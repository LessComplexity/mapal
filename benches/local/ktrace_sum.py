#!/usr/bin/env python3
"""Summarize a ktrace.cpp CUPTI trace (one TSV per traced process): per kernel name the launches, total and mean
device us, grid / block / registers / smem of its first launch; memcpy / memset totals; the device busy time and the
span from the first to the last record. `--window=<first>:<last>` keeps the records from the first launch of a kernel
whose name contains <first> to the end of the last launch of one containing <last> (a timed region).
usage: ktrace_sum.py <trace.tsv> [--window=a:b] [--top=N]"""
import sys
from collections import defaultdict


def short(name):
    for pre in ('void ', ):
        name = name.removeprefix(pre)
    return name.split('(')[0][:90]


def main():
    path = sys.argv[1]
    opts = dict(a.split('=', 1) for a in sys.argv[2:] if a.startswith('--'))
    top = int(opts.get('--top', 12))
    recs = []
    for line in open(path, errors='replace'):
        f = line.rstrip('\n').split('\t')
        if f[0] == 'K':
            recs.append(('K', short(f[1]), int(f[7]), int(f[8]), f[2], f[3], f[4], f[5], f[6]))
        elif f[0] == 'C':
            recs.append(('C', f'memcpy kind={f[2]}', int(f[3]), int(f[4]), int(f[1])))
        elif f[0] == 'S':
            recs.append(('S', 'memset', int(f[2]), int(f[3]), int(f[1])))
    recs.sort(key=lambda r: r[2])
    if '--window' in opts:
        a, b = opts['--window'].split(':')
        i0 = next(i for i, r in enumerate(recs) if r[0] == 'K' and a in r[1])
        i1 = max(i for i, r in enumerate(recs) if r[0] == 'K' and b in r[1])
        recs = recs[i0:i1 + 1]
    if not recs:
        print('no records')
        return
    agg = defaultdict(lambda: [0, 0, None])
    busy, last_end = 0, 0
    for r in recs:
        a = agg[(r[0], r[1])]
        a[0] += 1
        a[1] += r[3] - r[2]
        if a[2] is None:
            a[2] = r[4:]
        s, e = max(r[2], last_end), r[3]
        busy += max(0, e - s)
        last_end = max(last_end, e)
    span = (max(r[3] for r in recs) - recs[0][2]) / 1e3
    print(f'span_us={span:.1f} busy_us={busy / 1e3:.1f} idle_us={span - busy / 1e3:.1f} records={len(recs)}')
    for (k, n), (c, t, info) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:top]:
        extra = (f'grid={info[0]} block={info[1]} regs={info[2]} smem={int(info[3]) + int(info[4])}' if k == 'K'
                 else f'bytes={sum(r[4] for r in recs if r[1] == n)}')
        print(f'  {k} {n} n={c} total_us={t / 1e3:.1f} mean_us={t / 1e3 / c:.2f} {extra}')


if __name__ == '__main__':
    main()
