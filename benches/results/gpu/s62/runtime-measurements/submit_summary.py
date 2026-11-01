#!/usr/bin/env python3
"""Submission stage 2 (submit_bench.c): generated vs direct per-launch enqueue, two sessions.

usage: submit_summary.py <part> <stage2-session-1> <stage2-session-2>
Per kernel, the medians over its processes of generated_us_med, direct_us_med, their ratio and the
direct-vs-direct null ratio. The overhead counts when generated/direct > 1 + max(0.01, floor) in
both sessions, floor = the session's largest |null_direct_ratio - 1|.
"""
import re
import statistics
import sys

part, files = sys.argv[1], sys.argv[2:]
ses = []
for f in files:
    k = {}
    for l in open(f):
        if not l.startswith('submit_floor'):
            continue
        v = dict(re.findall(r'(\w+)=(\S+)', l))
        k.setdefault((v['kernel'], v['entry'], v['nptr']), []).append(v)
    ses.append(k)
fl = [max(abs(float(v['null_direct_ratio']) - 1) for vs in s.values() for v in vs) for s in ses]
th = [max(0.01, x) for x in fl]
med = lambda vs, key: statistics.median(float(v[key]) for v in vs)
print(f"### {part}: submission stage 2\n")
print(f"Floor |direct/direct - 1| per session: {', '.join(f'{x:.4f}' for x in fl)}.\n")
print("| kernel | entry | ptrs | generated us s1 / s2 | direct us s1 / s2 | generated/direct s1 / s2 | direct/direct null s1 / s2 | sign |")
print("|---|---|---|---|---|---|---|---|")
keys = sorted(set().union(*[set(s) for s in ses]))
signs = []
for key in keys:
    if not all(key in s for s in ses):
        continue
    g = [med(s[key], 'generated_us_med') for s in ses]
    d = [med(s[key], 'direct_us_med') for s in ses]
    r = [x / y for x, y in zip(g, d)]
    sg = '+' if all(x > 1 + t for x, t in zip(r, th)) else '-' if all(x < 1 - t for x, t in zip(r, th)) else '0'
    signs.append(sg)
    j = lambda xs, p=3: ' / '.join(f'{x:.{p}f}' for x in xs)
    nl = [med(s[key], 'null_direct_ratio') for s in ses]
    print(f"| {key[0][11:]} | {key[1]} | {key[2]} | {j(g)} | {j(d)} | {j(r)} | {j(nl)} | {sg} |")
print(f"\n{part}: {signs.count('+')} of {len(signs)} kernels generated > direct beyond the floor in both sessions.\n")
