#!/usr/bin/env python3
"""plan-cost (design §3): fit the cost model's `eff` on half of the measured plan arms and judge the picks on the other half.

    plancost.py <arch> <results-<arch>.txt> [--out benches/results/evidence/plan-cost] [--mapal target/release/mapal]

1. carry: the sweep's `probe` rows (default face) whose arm the current build re-emits — the launched kernel ids of the
   arm's emission (`--gpu-off=plan-cost`: the rules the sweep ran) equal the arm's measured `probe_kernels` ids;
2. emit each carried arm with `--plan-cands`: per tile site its candidates and their terms (`; mapal.cand`), the rates
   (`; mapal.rates`) and the plan it emitted (`; mapal.plan … combine= S=`); an arm's predicted time is the sum over its
   sites of the emitted candidate's ns, plus one constant per program (the work no candidate changes);
3. mirror: `site_ns` re-evaluated here from each line's inputs must equal the printed ns (to its printed 0.1 ns);
4. fit `eff_max_pm`, `eff_half` (log-time least squares) on the programs of one hash half; report on the other:
   pick accuracy (the argmin arm within max(null, 2 %) of the measured best), pairwise sign accuracy, median |log err|.
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

    def wave(bw):
        used = min(bw, f(r['sms']))
        wps = bw / used * warps / f(r['schedulers'])
        q = wps * f(p['acc'])
        rate = used * peak * (f(r['eff_max_pm']) / 1000.0 * q / (q + f(r['eff_half'])))
        return bw * 2.0 * tile * kb / rate, bw * f(p['bm'] + p['bn']) * kb * e / l2

    ff, lf = wave(min(w, blocks))
    ft, lt = wave(tail)
    full = waves - 1.0
    out = rows * c * e
    ws = f(p['s']) if p['comb'] == 'splitk' else 1.0
    d_ns = ((rows * k + k * c) * e + out * ws) / dram
    kernel = max(full * max(ff, lf) + max(ft, lt), d_ns)
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


def emit(mapal, arch, prog, arm, work):
    src = f'{ROOT}/benches/algorithms/{prog}.mapal'
    fl = flags(arm)
    if fl is None or not os.path.exists(src):
        return None
    out = f'{work}/{prog}__{arm}'
    if not os.path.exists(out + '.ll'):
        r = subprocess.run([mapal, 'build', src, f'--target={arch}', '--emit-only', '--plan-cands', '-o', out,
                            '--gpu-off=plan-cost'] + fl, capture_output=True, text=True)
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
                                'kb', 's')}
    p['comb'] = comb_of(f['cand'])
    return p


def main():
    a = sys.argv[1:]
    arch, results = a[0], a[1]
    out = a[a.index('--out') + 1] if '--out' in a else f'{ROOT}/benches/results/evidence/plan-cost'
    mapal = a[a.index('--mapal') + 1] if '--mapal' in a else f'{ROOT}/target/release/mapal'
    work = os.environ.get('PLANCOST_WORK', f'/tmp/plancost-{arch}')
    os.makedirs(work, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    t, ks = load(results)
    progs = sorted({p for p, _ in t if (p, 'on') in t})
    rows, mirror, carried, dropped = [], [0, 0], 0, 0
    rates = None
    for prog in progs:
        on = st.median(t[(prog, 'on')])
        null = abs(st.median(t[(prog, 'null')]) / on - 1) if (prog, 'null') in t else 0.02
        arms = []
        for arm in sorted({x for (q, x) in t if q == prog and PLAN.match(x)}):
            e = emit(mapal, arch, prog, arm, work)
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
            sites = [cands[s][lab] for s, lab in picks.items() if lab in cands.get(s, {})]
            if len(sites) == len(picks):
                arms.append((arm, st.median(t[(prog, arm)]) * 1e6, [shape(c) for c in sites]))
        # a decision: two or more carried arms that emit different plans
        if len({repr(x[2]) for x in arms}) >= 2:
            half = int(hashlib.md5(prog.encode()).hexdigest(), 16) % 2
            rows.append((prog, half, max(null, 0.02), arms))
    assert mirror[0] == mirror[1], f'mirror: {mirror[1]}/{mirror[0]} candidate ns reproduced'

    def judge(part, rr, emax, ehalf):
        r = dict(rr, eff_max_pm=emax, eff_half=ehalf)
        sq, picks, signs, errs = 0.0, [], [], []
        for prog, _, null, arms in part:
            pred = [sum(site_ns(r, p) for p in sh) for _, _, sh in arms]
            meas = [m for _, m, _ in arms]
            cst = st.mean(m - q for m, q in zip(meas, pred))
            le = [math.log(max(q + cst, 1.0) / m) for q, m in zip(pred, meas)]
            sq += sum(abs(x) for x in le)  # L1: a cliff arm (spills) must not set the slope
            errs += [abs(x) for x in le]
            best = min(meas)
            picks.append(meas[pred.index(min(pred))] <= best * (1 + null))
            for i in range(len(arms)):
                for j in range(i + 1, len(arms)):
                    if abs(meas[i] / meas[j] - 1) > null:
                        signs.append((pred[i] < pred[j]) == (meas[i] < meas[j]))
        return sq, picks, signs, errs

    fit = [x for x in rows if x[1] == 0]
    hold = [x for x in rows if x[1] == 1]
    best = None
    for emax in range(100, 1001, 25):  # at most the FFMA peak
        for ehalf in [0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]:
            sq = judge(fit, rates, emax, ehalf)[0]
            if best is None or sq < best[0]:
                best = (sq, emax, ehalf)
    _, emax, ehalf = best
    report = []
    for name, part in (('fit', fit), ('hold-out', hold)):
        for label, (em, eh) in (('seed', (rates['eff_max_pm'], rates['eff_half'])), ('fitted', (emax, ehalf))):
            _, picks, signs, errs = judge(part, rates, em, eh)
            report.append((name, label, em, eh, len(part), sum(picks), len(picks), sum(signs), len(signs),
                           st.median(errs) if errs else float('nan')))
    tsv = f'{out}/holdout-{arch}.tsv'
    with open(tsv, 'w') as fh:
        fh.write('half\trates\teff_max_pm\teff_half\tdecisions\tpick_ok\tpicks\tsign_ok\tpairs\tmedian_abs_log_err\n')
        for x in report:
            fh.write('\t'.join(str(v) if not isinstance(v, float) else f'{v:.4f}' for v in x) + '\n')
    with open(f'{out}/decisions-{arch}.tsv', 'w') as fh:
        fh.write('program\thalf\tnull\tarm\tmeasured_ns\tpredicted_ns_fitted\n')
        r = dict(rates, eff_max_pm=emax, eff_half=ehalf)
        for prog, half, null, arms in rows:
            for arm, m, sh in arms:
                fh.write(f'{prog}\t{"fit" if half == 0 else "hold"}\t{null:.4f}\t{arm}\t{m:.0f}\t'
                         f'{sum(site_ns(r, p) for p in sh):.0f}\n')
    print(f'{arch}: carried {carried} arms ({dropped} dropped: kernel ids not re-emitted), {len(rows)} decisions '
          f'({len(fit)} fit / {len(hold)} hold-out); mirror {mirror[1]}/{mirror[0]}; fitted eff_max_pm={emax} '
          f'eff_half={ehalf}')
    for x in report:
        print(f'  {x[0]:8} {x[1]:6} eff=({x[2]},{x[3]}) picks {x[5]}/{x[6]}  signs {x[7]}/{x[8]}  '
              f'median|log err| {x[9]:.3f}')


if __name__ == '__main__':
    main()
