#!/usr/bin/env python3
"""S62 geometry: one part's two timing passes (results-<p>.txt, results-<p>-pass2.txt) -> tables.

Per (cell, face): ON = shipped, OFF = `off-row-fit+wave-gate+grid-share+panel-regs+fill-rows`
(absent: the all-off text equals ON). ms = median of the 60 events (2 processes x 30).
ON/OFF = ON ms / OFF ms, each process normalized by its own cuBLAS leg, mean of the 2 passes
(<1 = ON faster). null = max over passes of |shipped/null - 1| (normalized).
Per arm: arm/ON normalized per pass (>1 = the arm is slower than shipped).
usage: summ.py <results-p.txt> <results-p-pass2.txt>
"""
import re
import statistics
import sys

ALL5 = 'off-row-fit+wave-gate+grid-share+panel-regs+fill-rows'


def parse(path):
    rows, cur, bad = {}, None, []
    for l in open(path):
        m = re.match(r'# n=(\S+) face=(\S+)', l)
        if m:
            cur = (m[1],) + tuple(m[2].partition('.')[::2])
            rows[cur] = {}
            continue
        if cur is None:
            continue
        if 'bit_exact_vs_in_order_fold=NO' in l or 'MAPAL VALUES WRONG' in l:
            bad.append(cur)
        m = re.match(r'mapal kernel=\S+ (grid=\S+ block=\d+)', l)
        if m:
            rows[cur]['geom'] = m[1]
        if l.startswith('values leg=mapal') and 'bit_exact_vs_in_order_fold=yes' in l:
            rows[cur]['exact'] = True
        m = re.match(r'samples mapal=(\S+) cublas=(\S+) cutlass=(\S+)', l)
        if m:
            rows[cur]['s'] = [list(map(float, x.split(','))) for x in m.groups()]
    return {k: v for k, v in rows.items() if 's' in v}, bad


def med(x):
    return statistics.median(x)


def main(p1, p2):
    passes, bad = [], []
    for p in (p1, p2):
        r, b = parse(p)
        passes.append(r)
        bad += b
    keys = sorted(set(passes[0]) & set(passes[1]), key=lambda k: (k[0], k[1], k[2]))
    print('value failures:', bad or 'none',
          '| rows without bit_exact=yes:', [k for p in passes for k, v in p.items() if not v.get('exact')] or 'none')

    def norm(p, k):
        return med(p[k]['s'][0]) / med(p[k]['s'][1])

    def pooled(k, leg):
        return med(passes[0][k]['s'][leg] + passes[1][k]['s'][leg])

    cells = sorted({(k[0], k[1]) for k in keys}, key=lambda c: (c[0].count('x') == 0, c))
    print('\ncell face | ON ms | OFF ms | cuBLAS ms | CUTLASS ms | ON/OFF | ON/OFF raw | null | ON grid | OFF grid')
    for n, f in cells:
        on = (n, f, '')
        if on not in passes[0] or on not in passes[1]:
            continue
        nul = (n, f, 'null')
        null = max(abs(norm(p, on) / norm(p, nul) - 1) for p in passes) if nul in passes[0] and nul in passes[1] else float('nan')
        off = (n, f, ALL5)
        if off in passes[0] and off in passes[1]:
            r = statistics.mean(norm(p, on) / norm(p, off) for p in passes)
            offms, raw, og = f'{pooled(off, 0):.4f}', f'{pooled(on, 0) / pooled(off, 0):.3f}', passes[0][off].get('geom')
            r = f'{r:.3f} ({"/".join(f"{norm(p, on) / norm(p, off):.3f}" for p in passes)})'
        else:
            offms, raw, og, r = '= ON', '1 (same text)', '= ON', '1 (same text)'
        print(f'{n} {f} | {pooled(on, 0):.4f} | {offms} | {pooled(on, 1):.4f} | {pooled(on, 2):.4f} | {r} | {raw} | '
              f'{null:.3%} | {passes[0][on].get("geom")} | {og}')
    print('\narms: cell face arm | arm/ON per pass (normalized; >1 = arm slower) | arm ms | ON ms | arm grid')
    for k in keys:
        n, f, a = k
        if a in ('', 'null'):
            continue
        on = (n, f, '')
        rs = [norm(p, k) / norm(p, on) for p in passes]
        print(f'{n} {f} {a} | {" ".join(f"{x:.3f}" for x in rs)} | {pooled(k, 0):.4f} | {pooled(on, 0):.4f} | '
              f'{passes[0][k].get("geom")}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
