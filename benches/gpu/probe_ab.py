#!/usr/bin/env python3
"""S62: the probe section's arms as medians — program `iter ms` (untraced process) and the traced
per-launch kernel ms summed — per program, face and arm, with OFF/ON (> 1 = ON faster).

usage: python3 benches/gpu/probe_ab.py <results-<profile>.txt> [prog-prefix]
"""
import re
import statistics
import sys
from collections import defaultdict


def main(path, prefix='pt_'):
    it, kern = defaultdict(list), defaultdict(list)
    for line in open(path):
        m = re.match(r'probe prog=(\S+) face=(\S+) arm=(\S+) rep=\d+ .*?out=\[.*?iter ms=([\d.e+-]+)', line)
        if m and m[1].startswith(prefix):
            it[m.groups()[:3]].append(float(m[4]))
        m = re.match(r'probe_kernels prog=(\S+) face=(\S+) arm=(\S+) rep=\d+ (.*)', line)
        if m and m[1].startswith(prefix):
            kern[m.groups()[:3]].append(sum(float(x) for x in re.findall(r'ms=([\d.]+)', m[4])))
    med = lambda v: statistics.median(v) if v else float('nan')
    print('| program | face | arm | reps | iter ms median (min-max) | traced kernel ms median | OFF/ON iter | OFF/ON kernel |')
    print('|---|---|---|---:|---:|---:|---:|---:|')
    for key in sorted(it):
        prog, face, arm = key
        on = (prog, face, 'on')
        r_it = med(it[key]) / med(it[on]) if arm != 'on' and on in it else None
        r_k = med(kern[key]) / med(kern[on]) if arm != 'on' and kern.get(on) and kern.get(key) else None
        print(f"| {prog} | {face} | {arm} | {len(it[key])} | {med(it[key]):.4f} ({min(it[key]):.4f}-{max(it[key]):.4f}) | "
              f"{med(kern[key]):.4f} | {'' if r_it is None else f'{r_it:.3f}'} | {'' if r_k is None else f'{r_k:.3f}'} |")


if __name__ == '__main__':
    main(*sys.argv[1:])
