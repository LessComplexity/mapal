#!/usr/bin/env python3
"""caching-memory: time and peak device memory per workload, from one or more box_run results files
(benches/results/evidence/local-rtx-5080/caching-memory.md).

Mapal: each arm's `iter ms` (median over its processes) and its peak device bytes (`mapal_gpu_peak`, printed by the
traced processes: every cuMemAlloc less every cuMemFree, the high-water mark). Torch: each leg's median ms and
`peak_mb` (torch.cuda.max_memory_allocated over the leg, its tensors included), gated legs only.

usage: caching_memory_report.py <results.txt>... [--steps N]   (decode rows: also ms per step = iter ms / N)
"""
import re
import statistics
import sys
from collections import defaultdict

PROBE = re.compile(r'^probe prog=(\S+) face=(\S+) arm=(\S+) rep=(\d+) .*?out=\[(.*?)\]')
PEAK = re.compile(r'mapal_gpu_peak bytes=(\d+)')
TORCH = re.compile(r'^torch prog=(\S+) mode=(\S+) .*?med_ms=([\d.]+) .*?within=(\S+) .*?peak_mb=([\d.na]+)')


def main(paths):
    ms, peak = defaultdict(list), {}
    torch = defaultdict(dict)
    for p in paths:
        for line in open(p, errors='replace'):
            m = PROBE.match(line)
            if m:
                prog, face, arm, _, out = m.groups()
                if face != '_default':
                    continue
                it = re.search(r'iter ms=([\d.e-]+)', out)
                if it:
                    ms[(prog, arm)].append(float(it.group(1)))
                pk = PEAK.search(line)
                if pk:
                    peak[(prog, arm)] = int(pk.group(1)) / 2 ** 20
                continue
            m = TORCH.match(line)
            if m:
                prog, mode, med, within, pmb = m.groups()
                if within == 'yes':
                    torch[prog][mode] = (float(med), float(pmb) if pmb != 'nan' else float('nan'))
    print('| workload | arm | Mapal ms | Mapal peak MB | best competitor (leg) | ms | its peak MB | Mapal / it |')
    print('|---|---|---:|---:|---|---:|---:|---:|')
    for (prog, arm) in sorted(ms):
        v = statistics.median(ms[(prog, arm)])
        pk = peak.get((prog, arm), float('nan'))
        legs = torch.get(prog, {})
        best = min(legs.items(), key=lambda kv: kv[1][0]) if legs else None
        if best:
            leg, (tm, tp) = best
            print(f'| {prog} | {arm} | {v:.3f} | {pk:.0f} | {leg} | {tm:.3f} | {tp:.0f} | {v / tm:.2f} |')
        else:
            print(f'| {prog} | {arm} | {v:.3f} | {pk:.0f} | — | | | |')
    print()
    print('| workload | torch leg | ms | peak MB |')
    print('|---|---|---:|---:|')
    for prog in sorted(torch):
        for leg, (tm, tp) in sorted(torch[prog].items(), key=lambda kv: kv[1][0]):
            print(f'| {prog} | {leg} | {tm:.3f} | {tp:.0f} |')


if __name__ == '__main__':
    main([a for a in sys.argv[1:] if not a.startswith('--')])
