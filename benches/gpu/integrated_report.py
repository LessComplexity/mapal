#!/usr/bin/env python3
"""S62 integration session tables from box_run results files (integrated_bundle.sh).

Kernel cells (box_run's splitk section, every arm + cuBLAS + CUTLASS in one tile_bench
process, two processes ABBA): per cell x face, each pass's median ms of the baseline
(`.base`, 4e1102c's emitter), the integrated build, cuBLAS and CUTLASS, the ratios
base/int and int/cuBLAS, every LOO arm's OFF/ON (`.off-<rung>` time over the integrated
time; > 1 = the rung pays), and the null arm's |null/int - 1| (the box's variation).
Gate failures are listed first. Whole programs (probe section): median `iter ms` per arm
over the repetitions, base/int, OFF/ON per LOO arm, null.

usage: integrated_report.py <results files...>"""
import re
import sys
from collections import defaultdict

med = lambda xs: sorted(xs)[len(xs) // 2]
FACE = {'': 'exact', '_c': 'contract', '_r': 'reassoc', '_rc': 'reassoc+c'}


def split(stem):
    """`mm_<cell><face>[.<arm>]` -> (cell, face, arm)."""
    head, _, arm = stem.partition('.')
    m = re.match(r'mm_(\d+x\d+x\d+)(_rc|_r|_c)?$', head)
    return m[1], m[2] or '', arm or 'int'


def kernels(path):
    t = defaultdict(lambda: defaultdict(list))  # (cell) -> leg -> [med per pass]
    gates = []
    for l in open(path, errors='replace'):
        m = re.match(r'pass=(\d+) cell=mm_(\S+) (.*)', l)
        if not m:
            continue
        cell, rest = m[2], m[3]
        if 'GATE-FAILED' in rest or 'same_output=NO' in rest:
            gates.append(l.strip())
        tm = re.match(r'time leg=(\S+) .*med_ms=([\d.]+)', rest)
        if tm:
            t[cell][tm[1]].append(float(tm[2]))
    return t, gates


def probes(path):
    runs = defaultdict(list)
    fails = []
    for l in open(path, errors='replace'):
        if l.startswith('VALUE-FAIL') or l.startswith('VALIDATION FAILED'):
            fails.append(l.strip())
        if not l.startswith('probe prog='):
            continue
        f = dict(re.findall(r'(\w+)=([^\s\[\]]+)', l))
        it = re.search(r'iter ms=\s*([-\d.eE+]+)', l)
        if it:
            runs[(f['prog'], f['face'], f['arm'])].append(float(it[1]))
    return runs, fails


def report(path):
    print(f'## {path}\n')
    t, gates = kernels(path)
    print('gate failures:', len(gates))
    for g in gates:
        print('  ', g)
    nulls = []
    print('\n| cell | face | base ms | int ms | cuBLAS | CUTLASS | base/int | int/cuBLAS | null | LOO OFF/ON |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for cell in sorted(t, key=lambda c: [int(x) for x in c.split('x')]):
        legs = t[cell]
        lib = {k: v for k, v in legs.items() if k in ('cublas', 'cutlass')}
        for face in ('', '_c', '_r', '_rc'):
            arms = {split(k)[2]: v for k, v in legs.items() if k.startswith('mm_') and split(k)[1] == face}
            if 'int' not in arms:
                continue
            i = arms['int']
            fmt = lambda xs: '/'.join(f'{x:.4f}' for x in xs)
            r = lambda a, b: '/'.join(f'{x / y:.3f}' for x, y in zip(a, b))
            base = arms.get('base') or (t[cell].get(f'mm_{cell}.base') if face == '_r' else None) or \
                (t[cell].get(f'mm_{cell}_c.base') if face == '_rc' else None)
            nv = ''
            if 'null' in arms:
                d = [abs(x / y - 1) for x, y in zip(arms['null'], i)]
                nulls.append(max(d))
                nv = '/'.join(f'{x:.3f}' for x in d)
            loo = ' '.join(f'{a.removeprefix("off-")}={r(v, i)}' for a, v in sorted(arms.items()) if a.startswith(('off-', 'pre')))
            print(f'| {cell} | {FACE[face]} | {fmt(base) if base else "-"} | {fmt(i)} | {fmt(lib.get("cublas", []))} | '
                  f'{fmt(lib.get("cutlass", []))} | {r(base, i) if base else "-"} | {r(i, lib.get("cublas", []))} | {nv} | {loo} |')
    if nulls:
        print(f'\nkernel null variation: max {max(nulls):.3f}, median {med(nulls):.3f}')
    runs, fails = probes(path)
    print('\nprobe value failures:', len(fails))
    for f in fails:
        print('  ', f)
    progs = sorted({(p, f) for p, f, _ in runs})
    if progs:
        print('\n| program | face | base iter ms | int iter ms | base/int | null | LOO OFF/ON |')
        print('|---|---|---|---|---|---|---|')
    pn = []
    for p, f in progs:
        m = {a: med(v) for (pp, ff, a), v in runs.items() if (pp, ff) == (p, f)}
        on = m.get('on')
        if not on:
            continue
        nv = abs(m['null'] / on - 1) if 'null' in m else None
        if nv is not None:
            pn.append(nv)
        loo = ' '.join(f'{a.removeprefix("off-")}={m[a] / on:.3f}' for a in sorted(m) if a.startswith(('off-', 'pre')))
        b = m.get('base')
        print(f'| {p} | {f} | {b:.4f} | {on:.4f} | {b / on:.3f} | {nv if nv is None else round(nv, 3)} | {loo} |'
              if b else f'| {p} | {f} | - | {on:.4f} | - | {nv} | {loo} |')
    if pn:
        print(f'\nprobe null variation: max {max(pn):.3f}, median {med(pn):.3f}')
    print()


for p in sys.argv[1:]:
    report(p)
