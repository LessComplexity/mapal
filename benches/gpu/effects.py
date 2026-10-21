#!/usr/bin/env python3
"""ON vs OFF per rung from a box_run results file: off_med / on_med (>1 = ON faster),
min-max overlap flag, and the ON arm vs cuBLAS / CUTLASS, plus regs/spill per arm."""
import re
import sys

for path in sys.argv[1:]:
    rows, cur = {}, {}
    for l in open(path):
        m = re.match(r'# n=(\d+) face=(\S+)', l)
        if m:
            cur = {'n': int(m[1]), 'face': m[2]}
            continue
        m = re.match(r'mapal kernel=\S+ grid=(\S+) block=(\d+) smem=(\d+) regs=(\d+) local_bytes=(\d+)', l)
        if m:
            cur['regs'], cur['spill'] = int(m[4]), int(m[5])
        m = re.match(r'time leg=(\w+) .* min_ms=([\d.]+) med_ms=([\d.]+) max_ms=([\d.]+)', l)
        if m:
            cur[m[1]] = tuple(float(x) for x in m.groups()[1:])
        if l.startswith('ratio'):
            rows[(cur['n'], cur['face'])] = cur
    # (label, on-face, off-face) per face prefix
    pairs = []
    for f, base in (('_c', '_c'), ('_exact', '_exact')):
        a = f + '.'
        pairs += [
            (f'{f} async  fma-regs', base, a + 'off-fma-regs'),
            (f'{f} async  idx32', base, a + 'off-idx32'),
            (f'{f} async  affine', base, a + 'off-affine'),
            (f'{f} async  wide-fma', base, a + 'off-wide-fma'),
            (f'{f} sync   fma-regs', a + 'off-async', a + 'off-async+fma-regs'),
            (f'{f} sync   idx32', a + 'off-async', a + 'off-async+idx32'),
            (f'{f} sync   affine', a + 'off-async', a + 'off-async+affine'),
            (f'{f} sync   pad', a + 'off-async', a + 'off-async+pad'),
            (f'{f} sync   wide-fma', a + 'off-async', a + 'off-async+wide-fma'),
            (f'{f} item   sync-ON vs S52-shipped', a + 'off-async', a + 'off-fma-regs'),
        ]
    print(f"\n{path}")
    print("| arm | n | ON ms | OFF ms | ON/OFF speedup | overlap | norm (vs same-bench cuBLAS) | ON vs cuBLAS | ON vs CUTLASS | ON regs/spill | OFF regs/spill |")
    print("|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|")
    for label, on, off in pairs:
        for n in (1024, 2048, 4096, 8192):
            a, b = rows.get((n, on)), rows.get((n, off))
            if not a or not b:
                continue
            x, y = a['mapal'], b['mapal']
            ov = 'overlap' if not (x[2] < y[0] or y[2] < x[0]) else ''
            print(f"| {label} | {n} | {x[1]:.3f} | {y[1]:.3f} | {y[1] / x[1]:.3f} | {ov} | {(a['cublas'][1] / x[1]) / (b['cublas'][1] / y[1]):.3f} "
                  f"| {a['cublas'][1] / x[1]:.2f} | {a['cutlass'][1] / x[1]:.2f} "
                  f"| {a.get('regs')}/{a.get('spill')} | {b.get('regs')}/{b.get('spill')} |")
