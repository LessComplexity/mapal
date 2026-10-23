#!/usr/bin/env python3
"""One row per part for the report tables: ranges of the within-session ratios over the cells.

usage: digest.py <part>:<session-1>,<session-2> ...
Signs follow summarize.py's plan rule (floor = the session's largest base-cell |null/on - 1|).
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../gpu'))
import arm_ratio as ar  # noqa: E402

AFFECTED = ('indep', 'diamond', 'mlp2')


def probe(prog):
    return prog.split('_')[-2]


def floor(d):
    r = [abs(ar.med(a['null']['iter']) / ar.med(a['on']['iter']) - 1)
         for a in d.values() if a.get('null') and a.get('on') and a['null']['iter'] and a['on']['iter']]
    return max([ar.EPS] + r)


def cells(ds, A, B, keep):
    """[(key, [ratio per session], sign)] over the cells both sessions measured."""
    th = [floor(d) for d in ds]
    out = []
    for key in sorted(set.intersection(*[set(d) for d in ds])):
        if not keep(key[0]):
            continue
        rs = [ar.ratio(d, key, A, B, 'iter') for d in ds]
        if any(r is None for r in rs):
            continue
        v = [r[0] for r in rs]
        s = '+' if all(x > 1 + t for x, t in zip(v, th)) else '-' if all(x < 1 - t for x, t in zip(v, th)) else '0'
        out.append((key, v, s))
    return out


def rng(cs):
    v = [x for _, vs, _ in cs for x in vs]
    return f"{min(v):.2f}–{max(v):.2f}" if v else '-'


def count(cs):
    return f"{sum(s == '+' for _, _, s in cs)}+/{sum(s == '-' for _, _, s in cs)}-/{len(cs)}"


base = lambda p: p.startswith('pt_') and not p.startswith(('pt_rq_', 'pt_rt_', 'pt_rg_'))
print("| part | steady predce/fs, affected (+/-/cells) | range | request clock predce/fs, affected | range "
      "| controls predce/fs (+/-/cells) | on/on-streams | off-arena/on | off-resident/on | off-devgen/on "
      "| rg/rt | floor s1 / s2 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for arg in sys.argv[1:]:
    part, fs = arg.split(':')
    ds = [ar.parse(f) for f in fs.split(',')]
    aff = lambda p: base(p) and probe(p) in AFFECTED
    rq = lambda p: p.startswith('pt_rq_') and probe(p) in AFFECTED
    ctl = lambda p: (base(p) or p.startswith('pt_rq_')) and probe(p) in ('twin', 'mlp')
    a, q, c = cells(ds, 'fs', 'predce', aff), cells(ds, 'fs', 'predce', rq), cells(ds, 'fs', 'predce', ctl)
    st = cells(ds, 'on-streams', 'on', base)
    arn = cells(ds, 'on', 'off-arena', base)
    rt = lambda p: p.startswith('pt_rt_')
    res, dev = cells(ds, 'on', 'off-resident', rt), cells(ds, 'on', 'off-devgen', rt)
    rg = []
    for d in ds:
        for key in d:
            if key[0].startswith('pt_rt_'):
                gk = (key[0].replace('pt_rt_', 'pt_rg_'), key[1])
                if gk in d and d[key].get('on') and d[gk].get('on'):
                    rg.append(ar.med(d[gk]['on']['iter']) / ar.med(d[key]['on']['iter']))
    rgs = f"{min(rg):.2f}–{max(rg):.2f}" if rg else '-'
    print(f"| {part} | {count(a)} | {rng(a)} | {count(q)} | {rng(q)} | {count(c)} | {rng(st)} ({count(st)}) "
          f"| {rng(arn)} ({count(arn)}) | {rng(res)} ({count(res)}) | {rng(dev)} ({count(dev)}) | {rgs} "
          f"| {' / '.join(f'{floor(d):.3f}' for d in ds)} |")
