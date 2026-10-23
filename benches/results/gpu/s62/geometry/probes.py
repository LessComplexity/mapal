#!/usr/bin/env python3
"""S62 geometry: residual-MLP probe rows -> per (prog, face, arm) medians of the traced
kernel sum and the program's own iter ms (5 reps, one process each)."""
import re, sys, statistics as st
from collections import defaultdict
k, it = defaultdict(list), defaultdict(list)
for l in open(sys.argv[1]):
    m = re.match(r'probe_kernels prog=(\S+) face=(\S+) arm=(\S+) rep=(\d+) (.*)', l)
    if m:
        k[m.group(1, 2, 3)].append(sum(float(x) for x in re.findall(r'ms=([0-9.]+)', m[5])))
    m = re.match(r'probe prog=(\S+) face=(\S+) arm=(\S+) rep=(\d+) .*iter ms=([0-9.]+)', l)
    if m:
        it[m.group(1, 2, 3)].append(float(m[5]))
print('prog face arm | kernel-sum ms med (min-max) | iter ms med | kernel-sum arm/on')
for key in sorted(k):
    on = st.median(k[(key[0], key[1], 'on')])
    v = k[key]
    print(f'{key[0]} {key[1]} {key[2]} | {st.median(v):.4f} ({min(v):.3f}-{max(v):.3f}) | {st.median(it[key]):.4f} | {st.median(v) / on:.3f}')
