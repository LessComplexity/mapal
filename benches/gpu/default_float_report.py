#!/usr/bin/env python3
"""S63 default-float tables from box_run results files (integrated_bundle.sh, ADR-0040).

Kernel cells (splitk section: every arm + cuBLAS + stock/tuned CUTLASS in one tile_bench
process, two processes ABBA): per cell the median ms over both processes' samples of the
default face, its fixed-split (`_rc`), contract (`_c`) and exact (`_x`) faces, the libraries,
default / best library, and the null arm's max per-process |null/default - 1|. Per rung
(`streamk`, `splitk-atomic`) on the cells where it changes the launch: ON/OFF per process
(OFF = the next realization down) and the S63 keep rule. MLP (mlp_bench): event medians.
Whole programs (probe section): median `iter ms`. Gates and value-bound lines first.

usage: default_float_report.py <results files...>"""
import re
import sys
from collections import defaultdict

med = lambda xs: sorted(xs)[len(xs) // 2]
CHAIN = ['', '.off-streamk', '.off-streamk+splitk-atomic', '.off-streamk+splitk-atomic+splitk']
RUNG_OFF = {'.off-streamk': 'streamk', '.off-streamk+splitk-atomic': 'splitk-atomic',
            '.off-streamk+splitk-atomic+splitk': 'splitk'}
LIBS = ('cublas', 'cutlass', 'cutlass_tuned')


def parse(path):
    samples = defaultdict(lambda: defaultdict(list))  # cell -> leg -> [ms] (both passes)
    pmed = defaultdict(lambda: defaultdict(dict))      # cell -> leg -> pass -> median
    combine = {}                                        # (cell, stem) -> combine/S
    fnv = defaultdict(dict)                             # (cell, stem) -> pass -> fnv
    bound = defaultdict(set)
    flags = []
    mlp = defaultdict(lambda: defaultdict(dict))        # prog -> leg -> pass -> event med
    mlp_vals = []
    probes = defaultdict(list)
    for l in open(path, errors='replace'):
        if any(x in l for x in ('GATE-FAILED', 'GATE FAILED', 'VALUE-FAIL', 'VALIDATION FAILED',
                                'SPLITK PROGRAM FAILED', 'BUILD FAILED')):
            flags.append(l.strip())
        m = re.match(r'pass=(\d+) cell=mm_(\S+) (.*)', l)
        if m:
            p, cell, rest = int(m[1]), m[2], m[3]
            if a := re.match(r'arm stem=(\S+) .* S=(\d+) combine=(\S+)', rest):
                combine[(cell, a[1])] = f'{a[3]}' + (f' S={a[2]}' if a[3] in ('workspace', 'atomic') else '')
            if v := re.match(r'values leg=(\S+) .*within_bound=(\w+) fnv=(\w+)', rest):
                fnv[(cell, v[1])][p] = v[3]
                bound[v[2]].add((cell, v[1]))
            if t := re.match(r'time leg=(\S+) .*med_ms=([\d.]+)', rest):
                pmed[cell][t[1]][p] = float(t[2])
            if s := re.match(r'samples leg=(\S+) (\S*)', rest):
                samples[cell][s[1]] += [float(x) for x in s[2].split(',') if x]
            continue
        m = re.match(r'pass=(\d+) mlp=(\S+) (.*)', l)
        if m:
            p, prog, rest = int(m[1]), m[2], m[3]
            if t := re.match(r'time leg=(\S+) .*event_med_ms=([\d.]+)', rest):
                mlp[prog][t[1]][p] = float(t[2])
            if rest.startswith('values'):
                mlp_vals.append(f'{prog} {rest.strip()}')
            continue
        if l.startswith('probe prog='):
            f = dict(re.findall(r'(\w+)=([^\s\[\]]+)', l))
            if it := re.search(r'iter ms=\s*([-\d.eE+]+)', l):
                probes[(f['prog'], f['face'], f['arm'])].append(float(it[1]))
        if l.startswith('value-bound') or l.startswith('splitk_program'):
            flags.append(l.strip())
    return samples, pmed, combine, fnv, bound, flags, mlp, mlp_vals, probes


def report(path):
    samples, pmed, combine, fnv, bound, flags, mlp, mlp_vals, probes = parse(path)
    print(f'## {path}\n')
    print('gates / value checks:')
    for f in flags:
        print('  ', f[:260])
    print(f'  within_bound=yes: {len(bound["yes"])} arms, NO: {sorted(bound["NO"])}')
    cells = sorted(samples, key=lambda c: [int(x) for x in c.split('x')])
    print('\n| cell | default realization | Mapal default ms | fixed-split (_rc) ms | contract ms | exact ms | '
          'cuBLAS | CUTLASS | tuned CUTLASS | default / best library | null |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    rungs = []
    for cell in cells:
        s = samples[cell]
        leg = lambda suffix: f'mm_{cell}{suffix}'
        m = lambda suffix: med(s[leg(suffix)]) if s.get(leg(suffix)) else None
        lib = {k: med(s[k]) for k in LIBS if s.get(k)}
        best = min(lib.values()) if lib else None
        d = m('')
        nul = [abs(pmed[cell][leg('.null')][p] / pmed[cell][leg('')][p] - 1)
               for p in pmed[cell][leg('')] if p in pmed[cell].get(leg('.null'), {})]
        null = max(nul) if nul else None
        f = lambda x: f'{x:.4f}' if x is not None else '-'
        print(f'| {cell} | {combine.get((cell, leg("")), "?")} | {f(d)} | {f(m("_rc"))} | {f(m("_c"))} | {f(m("_x"))} | '
              f'{f(lib.get("cublas"))} | {f(lib.get("cutlass"))} | {f(lib.get("cutlass_tuned"))} | '
              f'{d / best:.3f} | {null:.3f} |' if d and best and null is not None else f'| {cell} | incomplete |')
        # The realization chain: each kept arm's OFF arm is the next kept arm down.
        kept = [a for a in CHAIN if leg(a) in pmed[cell]]
        for on, off in zip(kept, kept[1:]):
            # `off` turns one more rung off than the arm above it (an arm equal to the one
            # above was dropped by the bundle): that last rung changed the launch.
            r = RUNG_OFF[off]
            per = [pmed[cell][leg(on)][p] / pmed[cell][leg(off)][p] for p in sorted(pmed[cell][leg(on)])
                   if p in pmed[cell][leg(off)]]
            rungs.append((r, cell, combine.get((cell, leg(on))), combine.get((cell, leg(off))), per, null,
                          [pmed[cell][leg(on)][p] for p in sorted(pmed[cell][leg(on)])],
                          [pmed[cell][leg(off)][p] for p in sorted(pmed[cell][leg(off)])]))
    print('\n| rung | cell | ON realization | OFF realization | ON ms (p1/p2) | OFF ms (p1/p2) | ON/OFF (p1/p2) | null |')
    print('|---|---|---|---|---|---|---|---|')
    verdict = defaultdict(lambda: [0, 0, 0])  # rung -> [faster, slower, inside]
    for r, cell, con, coff, per, null, on, off in rungs:
        fmt = lambda xs, k=4: '/'.join(f'{x:.{k}f}' for x in xs)
        print(f'| {r} | {cell} | {con} | {coff} | {fmt(on)} | {fmt(off)} | {fmt(per, 3)} | {null:.3f} |')
        if all(x > 1 + null for x in per):
            verdict[r][1] += 1
        elif all(x < 1 - null for x in per):
            verdict[r][0] += 1
        else:
            verdict[r][2] += 1
    print('\n| rung | cells faster beyond null | slower beyond null | inside null | keep (S63 rule) |')
    print('|---|---|---|---|---|')
    for r, (fa, sl, ins) in sorted(verdict.items()):
        print(f'| {r} | {fa} | {sl} | {ins} | {"ON" if sl == 0 and fa > 0 else "OFF row"} |')
    print('\nrun-to-run fnv of the default-face arms (pass 1 / pass 2; the face permits a difference):')
    for (cell, stem), d in sorted(fnv.items()):
        if '_' in stem.split('.')[0].removeprefix(f'mm_{cell}'):
            continue
        print(f'  {stem}: {d.get(1)} / {d.get(2)} {"same" if d.get(1) == d.get(2) else "DIFFER"} ({combine.get((cell, stem))})')
    if mlp:
        print('\n| MLP | leg | event med ms (p1/p2) |')
        print('|---|---|---|')
        for prog in sorted(mlp):
            for leg, d in sorted(mlp[prog].items()):
                print(f'| {prog} | {leg} | {"/".join(f"{d[p]:.4f}" for p in sorted(d))} |')
        for v in mlp_vals:
            print('  ', v[:220])
    if probes:
        print('\n| program | face | arm | median iter ms | n |')
        print('|---|---|---|---|---|')
        for (p, f, a), v in sorted(probes.items()):
            print(f'| {p} | {f} | {a} | {med(v):.4f} | {len(v)} |')
    print()


for p in sys.argv[1:]:
    report(p)
