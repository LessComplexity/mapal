#!/usr/bin/env python3
"""site-geometry smoke: ON/OFF per new rung from results-rtx-4090.txt (probes: median `iter ms` per arm, ABBA;
cells: tile_bench med_ms per leg). ratio = OFF / ON (> 1: the rung's ON arm is faster); null = |null / on - 1|.
usage: onoff.py results-rtx-4090.txt > onoff.tsv   (summary per rung on stderr)"""
import math
import re
import statistics
import sys
from collections import defaultdict

text = open(sys.argv[1]).read()
t = defaultdict(list)
for m in re.finditer(r'^probe prog=(\S+) face=(\S+) arm=(\S+) rep=\d+ .*?out=\[[^\]]*iter ms=([\d.]+)', text, re.M):
    t[(m[1] + ('' if m[2] == '_default' else m[2]), m[3])].append(float(m[4]))
for m in re.finditer(r'^pass=\d+ cell=\S+ time leg=(mm_\d+x\d+x\d+(?:_c|_x)?)(?:\.(\S+))? .*med_ms=([\d.]+)', text, re.M):
    t[(m[1], m[2] or 'on')].append(float(m[3]))
med = {k: statistics.median(v) for k, v in t.items()}
rows, by = [], defaultdict(list)
for (p, arm), v in sorted(med.items()):
    if not arm.startswith('off-') or (p, 'on') not in med:
        continue
    on = med[(p, 'on')]
    null = abs(med[(p, 'null')] / on - 1) if (p, 'null') in med else float('nan')
    r = v / on
    rows.append((p, arm[4:], f'{on:.4f}', f'{v:.4f}', f'{r:.3f}', f'{null:.3f}'))
    by[arm[4:]].append((r, null))
print('program\trung\ton_ms\toff_ms\toff/on\tnull')
for r in rows:
    print('\t'.join(r))
for rung, xs in sorted(by.items()):
    g = math.exp(sum(math.log(r) for r, _ in xs) / len(xs))
    win = sum(r - 1 > max(n, 0.01) for r, n in xs if not math.isnan(n))
    loss = sum(1 - r > max(n, 0.01) for r, n in xs if not math.isnan(n))
    nulls = [n for _, n in xs if not math.isnan(n)]
    print(f'{rung}: n={len(xs)} geomean OFF/ON={g:.3f} min={min(r for r, _ in xs):.3f} max={max(r for r, _ in xs):.3f} '
          f'win={win} loss={loss} null_max={max(nulls) if nulls else float("nan"):.3f}', file=sys.stderr)
