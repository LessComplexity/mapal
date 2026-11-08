#!/usr/bin/env python3
"""plan-cost measure phase: the sweep tables from a box run (box_run.sh results) — tables only, no raw.

usage: plancost_sweep.py <out dir> <diff dir> <arch>=<results.txt>[:<ops dir>] ...
  <diff dir>  changed.py diff (emission.tsv: which programs changed at HEAD vs the base)
  <ops dir>   the box's prof/ops (ops_nsys.sh: nsys gpu traces + ptxas of the 512 chains' arms)
Arms: ON = shipped (HEAD), OFF = `off-plan-cost+linear`, BASE = `on-base` (the base commit's emission). An arm absent from a
program's rows emits the same as ON (the bundle drops equal arms): it is printed `=ON`.
Writes <out>/programs-<arch>.tsv, ladder-<arch>.tsv, gates-<arch>.tsv, ops-<arch>.tsv, summary.tsv.
"""
import csv
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

OFF, BASE = 'off-plan-cost+linear', 'on-base'
CHAINS = ['indep', 'twin', 'mlp', 'mlp2', 'fanout', 'diamond']
LADDER = {f'pt_{c}_{n}' for c in CHAINS for n in (512, 1024, 2048)}
FLOOR = 0.02  # a difference under 2 % is never a win or a loss
med = lambda xs: st.median(xs) if xs else None


def parse(path):
    probes, gate, torch, chain, cells = defaultdict(list), defaultdict(lambda: [0, 0]), {}, defaultdict(dict), \
        defaultdict(lambda: defaultdict(list))
    diff = [0, 0]
    for l in open(path, errors='replace'):
        if l.startswith('probe prog='):
            f = dict(re.findall(r'(\w+)=([^\s\[\]]+)', l))
            if (it := re.search(r'iter ms=\s*([-\d.eE+]+)', l)) and f.get('face') == '_default':
                probes[(f['prog'], f['arm'])].append(float(it[1]))
        elif l.startswith('value-gate prog='):
            m = re.match(r'value-gate prog=(\S+) arms=(\d+)(?: failed=(\d+))?', l)
            gate[m[1]][0] += int(m[2]) - int(m[3] or 0)
            gate[m[1]][1] += int(m[3] or 0)
        elif m := re.match(r'torch prog=(\S+) mode=(\w+) .*med_ms=([\d.]+) .*within=(\S+)', l):
            torch[(m[1], m[2])] = (float(m[3]), m[4])
        elif m := re.match(r'chain probe=(\S+) n=(\d+) leg=(\S+) .*?med_ms=([\d.]+)', l):
            chain[f'pt_{m[1]}_{m[2]}'][m[3]] = float(m[4])
        elif m := re.match(r'chain_values probe=(\S+) n=(\d+) leg=(\S+) .* gate=FAIL', l):
            chain[f'pt_{m[1]}_{m[2]}'].setdefault('_fail', set()).add(m[3])
        elif m := re.match(r'pass=\d+ cell=(mm_\S+) time leg=(\S+) .*?med_ms=([\d.]+)', l):
            cells[m[1]][m[2]].append(float(m[3]))
        elif m := re.match(r'(PASS|FAIL)\b', l):
            diff[m[1] == 'FAIL'] += 1
    return probes, gate, torch, chain, cells, diff


def row(name, on, off, base, null, others, changed, noise):
    """One program's table row: ON / OFF / BASE ms, the best other leg, ratios, the verdict ON vs BASE."""
    best = min(others.items(), key=lambda kv: kv[1]) if others else None
    t = max(null or 0, noise or 0, FLOOR)
    ref = base if base is not None else on
    v = '=' if base is None else ('win' if on < base * (1 - t) else 'loss' if on > base * (1 + t) else 'null')
    r = lambda x: '' if x is None or best is None else f'{x / best[1]:.3f}'
    f = lambda x: '=ON' if x is None else f'{x:.5g}'
    return dict(program=name, changed='y' if changed else 'n', on=f'{on:.5g}', off=f(off), base=f(base),
                null='' if null is None else f'{null:.3f}', tol=f'{t:.3f}', best_other=best[0] if best else '',
                best_ms='' if not best else f'{best[1]:.5g}', on_over_best=r(on), base_over_best=r(ref),
                on_over_base='' if base is None else f'{on / base:.3f}', verdict=v)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    emission = list(csv.DictReader(open(Path(sys.argv[2]) / 'emission.tsv'), delimiter='\t'))
    summary = []
    for spec in sys.argv[3:]:
        arch, rest = spec.split('=', 1)
        path, _, ops = rest.partition(':')
        probes, gate, torch, chain, cells, diff = parse(path)
        changed = {r['program'] for r in emission if r['arch'] == arch and r['face'] == 'default' and r['status'] == 'changed'}
        rows = []
        progs = sorted({p for p, _ in probes})
        for p in progs:
            g = lambda a: med(probes.get((p, a)))
            on = g('on')
            if on is None:
                continue
            nul = g('null')
            reps = probes[(p, 'on')]
            noise = (max(reps) - min(reps)) / on / 2 if len(reps) > 2 else None  # half the min-max spread of ON
            lib = {k: v for k, v in chain.get(p, {}).items() if not k.startswith('_')
                   and k not in chain.get(p, {}).get('_fail', set())}
            others = dict(lib)
            for mode in ('eager', 'compile'):
                if (p, mode) in torch and torch[(p, mode)][1] == 'yes':
                    others[f'torch-{mode}'] = torch[(p, mode)][0]
            rows.append(row(p, on, g(OFF), g(BASE), abs(nul / on - 1) if nul else None, others, p in changed, noise))
        for c, legs in sorted(cells.items()):
            if c not in legs:
                continue
            g = lambda a: med(legs.get(f'{c}.{a}' if a else c))
            on = g('')
            if on is None:
                continue
            nul = g('null')
            others = {k: med(v) for k, v in legs.items() if k in ('cublas', 'cutlass', 'cutlass_tuned')}
            rows.append(row(c, on, g(OFF), g(BASE), abs(nul / on - 1) if nul else None, others, c in changed, None))
        cols = list(rows[0])
        for name, sel in (('programs', rows), ('ladder', [r for r in rows if r['program'] in LADDER])):
            with open(out / f'{name}-{arch}.tsv', 'w') as fh:
                w = csv.DictWriter(fh, cols, delimiter='\t')
                w.writeheader()
                w.writerows(sel)
        # value gates: the probes' (per arm), the differential, the cells' bound
        with open(out / f'gates-{arch}.tsv', 'w') as fh:
            fh.write('section\tpass\tfail\n')
            fh.write(f'differential (every program, every arm)\t{diff[0]}\t{diff[1]}\n')
            fh.write(f'probe arms (programs {len(gate)})\t{sum(v[0] for v in gate.values())}\t{sum(v[1] for v in gate.values())}\n')
            for p, v in sorted(gate.items()):
                if v[1]:
                    fh.write(f'  probe {p}\t{v[0]}\t{v[1]}\n')
        # summary on the changed programs
        ch = [r for r in rows if r['changed'] == 'y' and r['base'] != '=ON']
        gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float('nan')
        ratios = [float(r['on_over_base']) for r in ch]
        wl = {k: sum(r['verdict'] == k for r in ch) for k in ('win', 'loss', 'null')}
        offr = [float(r['off']) / float(r['base']) for r in ch if r['off'] not in ('=ON',)]
        chains512 = [r for r in rows if r['program'] in {f'pt_{c}_512' for c in CHAINS}]
        summary.append((arch, len(ch), f'{gm(ratios):.3f}', wl['win'], wl['loss'], wl['null'], f'{gm(offr):.3f}',
                        ' '.join(f"{r['program'][3:-4]}:{r['base_over_best']}>{r['on_over_best']}" for r in chains512)))
        if ops:
            ops_table(Path(ops), out / f'ops-{arch}.tsv')
    with open(out / 'summary.tsv', 'w') as fh:
        fh.write('arch\tchanged_programs_with_base_arm\tgeomean_ON_over_BASE\twin\tloss\tnull\tgeomean_OFF_over_BASE\t'
                 'chain_512_BASE>ON_over_best_other\n')
        for s in summary:
            fh.write('\t'.join(map(str, s)) + '\n')


def ops_table(d, dest):
    """Per 512 chain and arm: device ops after the last data-generation kernel (the timed region), zero passes, gaps."""
    rows = []
    for f in sorted(d.glob('*.1.gpu.csv')):
        stem = f.name[:-len('.1.gpu.csv')]
        per = []
        for r in (1, 2, 3):
            g = d / f'{stem}.{r}.gpu.csv'
            if not g.exists():
                continue
            ls = [l for l in open(g, errors='replace').read().split('\n') if l and l[0].isdigit()]
            ops = []
            for x in csv.reader(ls):
                if len(x) >= 21:
                    ops.append((int(x[0]), int(x[1]), x[20]))
            last = max((i for i, o in enumerate(ops) if 'mapal_gen_' in o[2]), default=-1)
            t = [o for o in ops[last + 1:] if 'memcpy' not in o[2]]  # the timed region: after data generation, before the copies out
            if not t:
                continue
            kern = [o for o in t if 'mapal_tile' in o[2]]
            zero = [o for o in t if 'memset' in o[2]]
            gaps = [t[i + 1][0] - (t[i][0] + t[i][1]) for i in range(len(t) - 1)]
            per.append((len(t), len(zero), sum(o[1] for o in kern) / 1e3, sum(max(g, 0) for g in gaps) / 1e3,
                        (t[-1][0] + t[-1][1] - t[0][0]) / 1e3))
        if not per:
            continue
        m = [f'{med([p[i] for p in per]):.1f}' for i in range(5)]
        ptx = d / f'{stem}.ptxas'
        res = ''
        if ptx.exists():
            txt = ptx.read_text(errors='replace')
            res = ' '.join(f'{a}/{b}' for a, b in zip(re.findall(r'Used (\d+) registers', txt), re.findall(r'(\d+) bytes spill stores', txt)))
        rows.append([stem, *m, res])
    with open(dest, 'w') as fh:
        fh.write('program.arm\tops\tzero_passes\tsum_kernel_us\tidle_gaps_us\tspan_us\tregs/spill_stores_per_kernel\n')
        fh.writelines('\t'.join(r) + '\n' for r in rows)


if __name__ == '__main__':
    main()
