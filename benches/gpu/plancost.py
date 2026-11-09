#!/usr/bin/env python3
"""plan-cost (design §3): fit the cost model's `eff` on half of the measured plan arms and judge the picks on the other half.

    plancost.py <arch> <results-<arch>.txt> [--out benches/results/evidence/plan-cost] [--mapal target/release/mapal]
                [--lat N]  (lat_ns calibrated: fixed at N, only eff fitted)

1. carry: the sweep's `probe` rows (default face) whose arm the current build re-emits — the launched kernel ids of the
   arm's emission (`--gpu-off=plan-cost`: the rules the sweep ran) equal the arm's measured `probe_kernels` ids;
2. emit each carried arm with `--plan-cands`: per tile site its candidates and their terms (`; mapal.cand`), the rates
   (`; mapal.rates`) and the plan it emitted (`; mapal.plan … combine= S=`); an arm's predicted time is the sum over its
   sites of the emitted candidate's ns, plus one constant per program (the work no candidate changes);
3. mirror: `site_ns` re-evaluated here from each line's inputs must equal the printed ns (to its printed 0.1 ns);
4. split the decisions by program (a program and its `@pc` rows in one half: no kernel timed in the fit is scored),
   2-fold per hash salt, each split used both ways (`SALTS`: 6 folds); a decision's null is max(the `null` arm, the
   spread of two arms that emit one plan, 2 %); per fold fit `eff_max_pm`, `eff_half` and `lat_ns` (log-time L1;
   plan-cost-terms: busiest SM, latency chain, warp occupancy — `lat_ns` 0 is the sprint `plan-cost` model, reported
   as `no-terms`, eff refit) on the fit half; report on the other: pick accuracy (the argmin arm within the null of
   the measured best), pairwise sign accuracy, median |log err| of each arm's predicted time anchored at the
   program's `on` arm (`pred_i − pred_on + meas_on`: no constant fitted on the half it scores); then mean / min / max
   over the folds and the per-fold terms − no-terms pick gain. A fitted rate on its grid's edge is flagged `edge` (a
   model finding, not a rate). `all`: the rates fitted on every decision. The sweep's ON arm is re-emitted at
   the rates it ran with (`SWEEP_RATES`), so its measured plan is carried;
5. compiler: every decision's program (half: fold 0's) emitted with plan-cost ON at the profile's rates; its plan is scored against
   the measured arm that emitted the same plan (within the null of the best), unknown where no arm did
   (`compiler-<arch>.tsv`, one row per decision).
Tables only go to --out; raw output stays outside git (CLAUDE.md).
"""
import hashlib, math, os, re, statistics as st, subprocess, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The plan arms (design §3, `pc-design/count2.py`): rungs and knobs that change a tile site's plan, not its residency.
PLAN = re.compile(r'^(on|off-(fill-wave|warp-fill|fit-first|row-split|fill-rows|wave-gate|splitk|splitk-atomic|'
                  r'panel-regs|prefetch)|on-(row-fit|warp-fill|streamk|async)|on-g-.*)$')


def site_ns(r, p):
    """`TargetProfile::site_ns` (crates/backends/llvm/src/profile.rs), op for op."""
    f = float
    rows, c, k, e = f(p['rows']) * 1.0, f(p['c']) * 1.0, f(p['k']) * 1.0, f(p['e'])
    blocks = max(float(math.ceil(f(p['blocks']) * 1.0 * 1.0)), 1.0)
    tiles = f(p['tiles']) * 1.0 * 1.0
    kb = f(p['kb']) * 1.0
    w = f(r['sms'] * max(p['per_sm'], 1))
    waves = float(math.ceil(blocks / w))
    tail = blocks - (waves - 1.0) * w
    warps = f(-(-p['threads'] // r['warp']))
    peak = f(r['fp32_lanes']) * f(r['clock_khz']) / 1e6 * 2.0
    l2, dram = f(r['l2_bytes_per_ns']), f(r['dram_bytes_per_ns'])
    tile = f(p['bm'] * p['bn'])
    terms = r.get('lat_ns', 0) > 0  # plan-cost-terms: busiest SM, latency chain

    def wave(bw):
        used = min(bw, f(r['sms']))
        busy = float(math.ceil(bw / f(r['sms']))) if terms else bw / used
        wps = busy * warps / f(r['schedulers'])
        if terms:
            q = max(wps, 1.0) * f(p['acc'])
            eff = f(r['eff_max_pm']) / 1000.0 * min(wps, 1.0) * q / (q + f(r['eff_half']))
        else:
            q = wps * f(p['acc'])
            eff = f(r['eff_max_pm']) / 1000.0 * q / (q + f(r['eff_half']))
        ff = busy * 2.0 * tile * kb / (peak * eff) if terms else bw * 2.0 * tile * kb / (used * peak * eff)
        # tile-efficiency T4: the busiest SM's LSU issue time floors the FFMA time
        ipc = r.get('lsu_ipc_pm', 0)
        if ipc > 0:
            uf = float(math.ceil(bw / f(r['sms']))) * warps * (kb / f(max(p['bk'], 1))) * f(p.get('lsu', 0)) \
                / (f(ipc) / 1000.0 * f(r['clock_khz']) / 1e6)
            ff = max(ff, uf)
        return ff, bw * f(p['bm'] + p['bn']) * kb * e / l2

    chain = 0.0
    if terms:
        trips, fly = max(float(math.ceil(kb / f(max(p['bk'], 1)))), 1.0), f(max(p['fly'], 1))
        chain = (trips + fly - 1.0) / fly * f(r['lat_ns'])
    ff, lf = wave(min(w, blocks))
    ft, lt = wave(tail)
    full = waves - 1.0
    out = rows * c * e
    ws = f(p['s']) if p['comb'] == 'splitk' else 1.0
    d_ns = ((rows * k + k * c) * e + out * ws) / dram
    kernel = max(full * max(ff, lf, chain) + max(ft, lt, chain), d_ns)
    atomics = {'atomic': f(p['s']) * rows * c, 'streamk': (tiles + blocks) * tile}.get(p['comb'], 0.0)
    combine = atomics / f(r['atomic_ops_per_ns'])
    gap, issue = f(r['gap_ns']), f(r['issue_ns'])
    zero = f(r['zero_ns']) + out / l2
    if p['comb'] == 'none':
        fixed = gap
    elif p['comb'] in ('atomic', 'streamk'):
        fixed = gap + zero + max(gap, issue - zero)
    else:
        combine += (ws + 1.0) * out / dram
        fixed = gap + max(gap, issue - kernel)
    return kernel + combine + fixed


def kv(line):
    return dict(x.split('=', 1) for x in line.split() if '=' in x)


def comb_of(label):
    return re.match(r'[a-z]+', label)[0]


def load(path):
    t, ks = defaultdict(list), defaultdict(set)
    for l in open(path, errors='replace'):
        m = re.match(r'probe prog=(\S+) face=_default arm=(\S+) rep=\d+ .*iter ms=([\d.]+)', l)
        if m:
            t[(m[1], m[2])].append(float(m[3]))
            continue
        # splitk cells (default-face legs `<cell>` / `<cell>.<arm>`; `_c`/`_r`/… are other faces): med_ms per pass
        m = re.match(r'pass=\d+ cell=(\S+) time leg=(\S+) .*med_ms=([\d.]+)', l)
        if m and (m[2] == m[1] or m[2].startswith(m[1] + '.')):
            t[(m[1], m[2][len(m[1]) + 1:] or 'on')].append(float(m[3]))
            continue
        m = re.match(r'pass=\d+ cell=(\S+) arm stem=(\S+) face=default\S* (.*)', l)
        if m and (m[2] == m[1] or m[2].startswith(m[1] + '.')):
            ks[(m[1], m[2][len(m[1]) + 1:] or 'on')] |= set(re.findall(r'=mapal_tile_([0-9a-f]{16})', m[3]))
            continue
        m = re.match(r'probe_kernels prog=(\S+) face=_default arm=(\S+) rep=\d+ (.*)', l)
        if m:
            ks[(m[1], m[2])] |= {k for k, _ in re.findall(r'(\w+) ms=([\d.]+)', m[3])}
    return t, ks


def flags(arm):
    if arm in ('on', 'null'):
        return []
    m = re.match(r'on-g-(\w+?)-(\d+)$', arm)
    if m:
        return [f'--geom={m[1]}={m[2]}']
    if arm.startswith('off-'):
        return [f'--gpu-off={arm[4:]}']
    if arm.startswith('on-'):
        return [f'--gpu-on={arm[3:]}']
    return None


# the plan-cost sweep's arms (there `on` is plan-cost ON, `off-plan-cost+linear` / `on-base` the rules)
# The rates the sweep's ON arm was emitted at (e32dcfb's profile): its measured plans re-emit only under them.
SWEEP_RATES = {'rtx-4090': {'eff_max_pm': 475, 'eff_half': 6, 'lat_ns': 0, 'lsu_ipc_pm': 0},
               'rtx-5090': {'eff_max_pm': 150, 'eff_half': 1, 'lat_ns': 0, 'lsu_ipc_pm': 0}}
PC_ARMS = {'on': [], 'on-base': ['--gpu-off=plan-cost,linear'], 'off-plan-cost+linear': ['--gpu-off=plan-cost,linear'],
           'off-streamk+splitk-atomic+splitk': ['--gpu-off=streamk,splitk-atomic,splitk']}


def emit(mapal, arch, prog, arm, work, cost_on=False, fl=None):
    src = f'{ROOT}/benches/algorithms/{prog}.mapal'
    fl = flags(arm) if fl is None else fl
    if fl is None or not os.path.exists(src):
        return None
    out = f'{work}/{prog}__{arm}' + ('__pc' if cost_on else '')
    # plan-cost ON forced where the arm does not turn it off: 4090/5090 ship it off (stopgap rows)
    on = [] if any('plan-cost' in x for x in fl) else ['--gpu-on=plan-cost,linear']
    if not os.path.exists(out + '.dev.ll'):  # a killed run leaves .ll alone
        r = subprocess.run([mapal, 'build', src, f'--target={arch}', '--emit-only', '--plan-cands', '-o', out]
                           + (on if cost_on else ['--gpu-off=plan-cost']) + fl, capture_output=True, text=True)
        if r.returncode or not os.path.exists(out + '.dev.ll'):
            return None
    ids = {f'{int(x) & (2 ** 64 - 1):016x}' for x in
           re.findall(r'@mapal_gpu_(?:launch\w*|gen\w*)\(i64 (-?\d+)', open(out + '.ll').read())}
    rates, cands, picks = {}, defaultdict(dict), {}
    for l in open(out + '.dev.ll'):
        if l.startswith('; mapal.rates'):
            rates = {a: int(b) for a, b in kv(l).items()}
        elif l.startswith('; mapal.cand'):
            f = kv(l)
            cands[f['site']][f['cand']] = f
        elif l.startswith('; mapal.plan') and ' site=' in l:
            f = kv(l)
            kid = f['kernel'].rsplit('_', 1)[-1]
            if kid not in ids:
                continue
            s = int(f['S'])
            lab = {'none': 'none', 'atomic': f'atomic{s}', 'splitk': f'splitk{s}',
                   'streamk': f"streamk{f['ctas']}x{f['iters']}"}[f['combine']]
            picks[f['site']] = lab
    return ids, rates, cands, picks


def shape(f):
    p = {k: int(f[k]) for k in ('rows', 'c', 'k', 'e', 'bm', 'bn', 'threads', 'per_sm', 'acc', 'tiles', 'blocks',
                                'kb', 's', 'bk', 'fly')}
    p['lsu'] = int(f.get('lsu', 0))  # tile-efficiency T4 (absent on older lines)
    p['comb'] = comb_of(f['cand'])
    return p


def plan_of(cands, picks):
    """Each site's emitted candidate shape (frozen, comparable across emissions); None if a pick has no candidate."""
    if any(lab not in cands.get(s, {}) for s, lab in picks.items()):
        return None
    return tuple(sorted((s, tuple(sorted(shape(cands[s][lab]).items()))) for s, lab in picks.items()))


def rung(arm):
    m = re.match(r'on-g-(\w+?)-\d+$', arm)
    return f'g-{m[1]}' if m else arm.split('-', 1)[-1]


SALTS = ('', 'b', 'c')


def folds(rows, salts=SALTS):
    """[(fit, hold)]: per salt a 2-way split by program (hash order, alternating), used both ways."""
    base = lambda x: x[0].split('@')[0]
    out = []
    for s in salts:
        progs = sorted({base(x) for x in rows}, key=lambda q: hashlib.md5((s + q).encode()).hexdigest())
        half = [[x for x in rows if progs.index(base(x)) % 2 == i] for i in (0, 1)]
        assert not {base(x) for x in half[0]} & {base(x) for x in half[1]}, 'a program in both halves'
        out += [tuple(half), tuple(half[::-1])]
    return out


def main():
    a = sys.argv[1:]
    arch, results = a[0], a[1]
    pc = [a[i + 1] for i, x in enumerate(a) if x == '--pc']  # plan-cost sweep results (ladder + changed programs)
    out = a[a.index('--out') + 1] if '--out' in a else f'{ROOT}/benches/results/evidence/plan-cost'
    mapal = a[a.index('--mapal') + 1] if '--mapal' in a else f'{ROOT}/target/release/mapal'
    work = os.environ.get('PLANCOST_WORK', f'/tmp/plancost-{arch}')
    os.makedirs(work, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    PC_ARMS['on'] = [f'--cost={k}={v}' for k, v in SWEEP_RATES.get(arch, {}).items()]
    rows, mirror, carried, dropped = [], [0, 0], 0, 0
    rates = None
    sources = [(load(results), '', lambda x: PLAN.match(x) and flags(x), False)]
    sources += [(load(x), '@pc', PC_ARMS.get, True) for x in pc]
    for (t, ks), tag, armfl, cost_on in sources:
        for prog in sorted({p for p, _ in t if (p, 'on') in t}):
            on = st.median(t[(prog, 'on')])
            null = abs(st.median(t[(prog, 'null')]) / on - 1) if (prog, 'null') in t else 0.02
            arms = []
            for arm in sorted({x for (q, x) in t if q == prog and armfl(x) is not None}):
                e = emit(mapal, arch, prog, arm, work, cost_on, armfl(arm))
                if e is None or not e[3]:
                    continue
                ids, r, cands, picks = e
                if ks.get((prog, arm)) and not ks[(prog, arm)] <= ids:
                    dropped += 1
                    continue
                carried += 1
                rates = rates or r
                for site in cands.values():
                    for c in site.values():
                        mirror[0] += 1
                        mirror[1] += abs(site_ns(r, shape(c)) - float(c['ns'])) <= 0.051
                plan = plan_of(cands, picks)
                if plan is not None:
                    arms.append((arm, st.median(t[(prog, arm)]) * 1e6, plan))
            # a decision: two or more carried arms that emit different plans; its class: the rungs that move the plan
            if len({x[2] for x in arms}) >= 2:
                ref = next((x[2] for x in arms if x[0] == 'on'), arms[0][2])
                cls = ','.join(sorted({rung(a) for a, _, pl in arms if pl != ref}))
                # two arms that emit one plan measure the box's null (`on-base` / `off-plan-cost+linear`)
                null = max([null] + [abs(x[1] / y[1] - 1) for i, x in enumerate(arms) for y in arms[i + 1:]
                                     if x[2] == y[2]])
                rows.append((prog + tag, cls, max(null, 0.02), arms))
    assert mirror[0] == mirror[1], f'mirror: {mirror[1]}/{mirror[0]} candidate ns reproduced'
    rows = sorted(rows)
    fs = folds(rows)
    hold0 = {x[0] for x in fs[0][1]}

    def judge(part, rr, emax, ehalf, lat=None):
        r = dict(rr, eff_max_pm=emax, eff_half=ehalf, lat_ns=rr.get('lat_ns', 0) if lat is None else lat)
        ns = {}  # one site shape, one rate set: one site_ns
        sq, picks, signs, errs = 0.0, [], [], []
        for prog, _, null, arms in part:
            pred = [sum(ns[p] if p in ns else ns.setdefault(p, site_ns(r, dict(p))) for _, p in pl)
                    for _, _, pl in arms]
            meas = [m for _, m, _ in arms]
            # the fit: one constant per program from its own (fit-half) arms
            cst = st.mean(m - q for m, q in zip(meas, pred))
            sq += sum(abs(math.log(max(q + cst, 1.0) / m)) for q, m in zip(pred, meas))  # L1: cliffs (spills)
            # the report: anchored at the program's `on` arm, nothing fitted on the scored arms
            on = next((i for i, x in enumerate(arms) if x[0] == 'on'), None)
            if on is not None:
                errs += [abs(math.log(max(q - pred[on] + meas[on], 1.0) / m))
                         for i, (q, m) in enumerate(zip(pred, meas)) if i != on]
            best = min(meas)
            picks.append(meas[pred.index(min(pred))] <= best * (1 + null))
            for i in range(len(arms)):
                for j in range(i + 1, len(arms)):
                    if abs(meas[i] / meas[j] - 1) > null:
                        signs.append((pred[i] < pred[j]) == (meas[i] < meas[j]))
        return sq, picks, signs, errs

    # the compiler's own pick (plan-cost ON, its built-in rates) on every decision: the measured arm that emitted the
    # same plan, ok when within max(null, 2 %) of the best; unknown where no arm did
    comp = []
    for prog, _, null, arms in rows:
        e = emit(mapal, arch, prog.split('@')[0], 'compiler', work, cost_on=True, fl=[])
        plan = e and plan_of(e[2], e[3])
        hit = [(a, m) for a, m, pl in arms if pl == plan]
        best = min(arms, key=lambda x: x[1])
        ok = '' if not hit else int(hit[0][1] <= best[1] * (1 + null))
        comp.append((prog, int(prog in hold0), null, hit[0][0] if hit else '?', best[0],
                     f'{best[1] and (hit[0][1] / best[1]):.3f}' if hit else '', ok))
    # the fit (log-time L1, fit half only): the declared fitted rates — eff_max_pm, eff_half and, with the terms,
    # lat_ns (0 = the sprint `plan-cost` model: no busiest-SM, latency-chain or warp-occupancy term)
    EM, EH, LAT = range(100, 1001, 25), [0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256], range(50, 1601, 50)
    lat_fixed = '--lat' in a  # lat_ns calibrated (`calib.cu`): not a fitted rate
    LAT = [int(a[a.index('--lat') + 1])] if lat_fixed else LAT
    grid = lambda part, lats: min((judge(part, rates, em, eh, lat)[0], em, eh, lat) for lat in lats
                                  for em in EM for eh in EH)[1:]
    # eff_half 0 is the floor (no ramp), not an edge
    edge = lambda em, eh, la: ','.join(n for n, hit in (('eff_max', em in (EM[0], EM[-1])), ('eff_half', eh == EH[-1]),
                                                          ('lat', not lat_fixed and la in (LAT[0], LAT[-1]))) if hit) or '-'
    seed = (rates['eff_max_pm'], rates['eff_half'], rates.get('lat_ns', 0))
    report = []
    for f, (fit, hold) in enumerate(fs):
        for label, (em, eh, la) in (('seed', seed), ('no-terms', grid(fit, [0])), ('terms', grid(fit, LAT))):
            _, picks, signs, errs = judge(hold, rates, em, eh, la)
            report.append((f, label, em, eh, la, edge(em, eh, la) if label != 'seed' else '-', len(fit), len(hold),
                           sum(picks), len(picks), sum(signs), len(signs), st.median(errs) if errs else float('nan')))
    allr = grid(rows, LAT)
    acc = lambda lab: [x[8] / x[9] for x in report if x[1] == lab]
    gain = [t[8] - n[8] for t, n in zip(*[[x for x in report if x[1] == lab] for lab in ('terms', 'no-terms')])]
    tsv = f'{out}/holdout-{arch}.tsv'
    with open(tsv, 'w') as fh:
        fh.write('fold\trates\teff_max_pm\teff_half\tlat_ns\tedge\tfit\tdecisions\tpick_ok\tpicks\tsign_ok\tpairs\t'
                 'median_abs_log_err\n')
        for x in report:
            fh.write('\t'.join(str(v) if not isinstance(v, float) else f'{v:.4f}' for v in x) + '\n')
        for lab in ('seed', 'no-terms', 'terms'):
            a = acc(lab)
            fh.write(f'mean/min/max\t{lab}\t\t\t\t\t\t\t{st.mean(a):.3f}\t{min(a):.3f}\t{max(a):.3f}\t\t\n')
        fh.write(f'gain\tterms-no-terms\t\t\t\t\t\t\t{",".join(map(str, gain))}\t\t\t\t\n')
        fh.write(f'all\tterms\t{allr[0]}\t{allr[1]}\t{allr[2]}\t{edge(*allr)}\t{len(rows)}\t\t\t\t\t\t\n')
    with open(f'{out}/decisions-{arch}.tsv', 'w') as fh:
        fh.write('program\thalf\tnull\tarm\tmeasured_ns\tpredicted_ns_all\n')
        r = dict(rates, eff_max_pm=allr[0], eff_half=allr[1], lat_ns=allr[2])
        for prog, _, null, arms in rows:
            for arm, m, pl in arms:
                fh.write(f'{prog}\t{"hold" if prog in hold0 else "fit"}\t{null:.4f}\t{arm}\t{m:.0f}\t'
                         f'{sum(site_ns(r, dict(p)) for _, p in pl):.0f}\n')
    print(f'{arch}: carried {carried} arms ({dropped} dropped: kernel ids not re-emitted), {len(rows)} decisions, '
          f'{len(fs)} program folds; mirror {mirror[1]}/{mirror[0]}; all-decision fit eff=({allr[0]},{allr[1]}) '
          f'lat={allr[2]} edge {edge(*allr)}')
    for x in report:
        print(f'  fold {x[0]} {x[1]:8} eff=({x[2]},{x[3]}) lat={x[4]} edge {x[5]}  picks {x[8]}/{x[9]}  '
              f'signs {x[10]}/{x[11]}  median|log err| {x[12]:.3f}')
    for lab in ('seed', 'no-terms', 'terms'):
        a = acc(lab)
        print(f'  {lab:8} hold picks mean {st.mean(a):.1%} min {min(a):.1%} max {max(a):.1%}')
    print(f'  terms - no-terms picks per fold: {gain}')
    ch = [c for c in comp if c[1] == 1]
    print(f'  fold-0 hold-out compiler picks {sum(c[6] == 1 for c in ch)}/{sum(c[6] != "" for c in ch)} '
          f'({sum(c[6] == "" for c in ch)} unknown: no arm emitted the pick)')
    with open(f'{out}/compiler-{arch}.tsv', 'w') as fh:
        fh.write('program\thalf\tnull\tcompiler_arm\tmeasured_best_arm\tcompiler_over_best\tok\n')
        for c in comp:
            fh.write(f'{c[0]}\t{"fit" if c[1] == 0 else "hold"}\t{c[2]:.4f}\t{c[3]}\t{c[4]}\t{c[5]}\t{c[6]}\n')


if __name__ == '__main__':
    main()
