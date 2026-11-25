#!/usr/bin/env python3
"""Mapal / best other per program from one local results file (site-geometry report.py's rule, one part):
Mapal = median `iter ms` of the shipped build (probe arm `on`, default face); best other = min over chain_bench legs
(`med_ms`, or `wall_med_ms` where printed) and torch legs (`med_ms`, `within=yes`; TF32 modes only for *_tf32 programs);
null = |median(null) / median(on) - 1|. Cells: the splitk section (tile_bench, one process per pass, ABBA): the
default-face build vs cuBLAS / CUTLASS stock / tuned, medians over passes."""
import re, statistics, sys
from collections import defaultdict
med = statistics.median
PROBE = re.compile(r'^probe prog=(\S+) face=(\S+) arm=(\S+) rep=\d+ .*?iter ms=\s*([-\d.eE+]+)')
TORCH = re.compile(r'^torch prog=(\S+) mode=(\S+) n=\d+ min_ms=\S+ med_ms=([\d.]+) .*within=(\w+)')
CHAIN = re.compile(r'^chain probe=(\S+) ((?:\w+=\S+ )*?)leg=(\S+) .*?(?:wall_med_ms|med_ms)=([\d.]+)')
CELL = re.compile(r'^pass=\d+ cell=(\S+) time leg=(\S+) .*?\bmed_ms=([\d.]+)')
probe = defaultdict(list); other = defaultdict(dict); cells = defaultdict(lambda: defaultdict(list))
for l in open(sys.argv[1], errors='replace'):
    if m := PROBE.match(l): probe[(m[1], m[3])].append(float(m[4])); continue
    if m := TORCH.match(l):
        if m[4] == 'yes' and ('tf32' in m[1] or not m[2].endswith('-tf32')):
            other[m[1]]['torch-' + m[2]] = float(m[3])
        continue
    if m := CHAIN.match(l):
        nums = '_'.join(x.split('=')[1] for x in m[2].split()); other[f'pt_{m[1]}_{nums}'][m[3]] = float(m[4]); continue
    if m := CELL.match(l): cells[m[1]][m[2]].append(float(m[3]))
print('| workload | Mapal ms | best other (leg) | Mapal/best | null |')
print('|---|---:|---|---:|---:|')
for p in sorted({k[0] for k in probe}):
    on = probe.get((p, 'on')); nl = probe.get((p, 'null'))
    if not on: continue
    m = med(on); o = other.get(p, {})
    k = min(o, key=o.get) if o else None
    null = f'{abs(med(nl) / m - 1):.1%}' if nl else '-'
    print(f'| {p[3:]} | {m:.4g} | {f"{o[k]:.4g} ({k})" if k else "no leg"} | {f"{m / o[k]:.2f}" if k else "-"} | {null} |')
print('\n| cell (default face) | Mapal ms | cuBLAS | CUTLASS | CUTLASS tuned | Mapal/best | null |')
print('|---|---:|---:|---:|---:|---:|---:|')
for c, legs in sorted(cells.items()):
    g = {k: med(v) for k, v in legs.items()}
    m = g.get(c); libs = {k: g[k] for k in ('cublas', 'cutlass', 'cutlass_tuned') if k in g}
    if m is None or not libs: continue
    b = min(libs.values()); n = g.get(c + '.null')
    print(f'| {c[3:]} | {m:.4g} | {g.get("cublas", float("nan")):.4g} | {g.get("cutlass", float("nan")):.4g} | '
          f'{g.get("cutlass_tuned", float("nan")):.4g} | {m / b:.2f} | {f"{abs(n / m - 1):.1%}" if n else "-"} |')
