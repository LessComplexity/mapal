#!/usr/bin/env python3
"""Retained (pt_rt_*) vs regenerated (pt_rg_*) immutable inputs, per cell, two sessions per part.

`programs` = pt_rg_X / pt_rt_X, arm on, each the whole program run in sequence on one box (the plan's
design); `arm` = the rg arm / on inside pt_rt_X (the pt_rg_X objects interleaved ABBA with on and
null, rg sessions only); `null` = null / on inside pt_rt_X. Sign: beyond 1 ± max(EPS, that session's
largest null/on deviation over these cells, else digest.py's base-cell floor) in both sessions.

usage: rg.py <part>:<s1>,<s2> ...
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../gpu'))
import arm_ratio as ar  # noqa: E402

med = ar.med


def floor(d):
    r = [abs(med(a['null']['iter']) / med(a['on']['iter']) - 1)
         for a in d.values() if a.get('null') and a.get('on') and a['null']['iter'] and a['on']['iter']]
    return max([ar.EPS] + r)


def ratio(a, b):
    """b/a on iter medians, disjoint flag."""
    if not a or not b or not a['iter'] or not b['iter']:
        return None
    return med(b['iter']) / med(a['iter']), max(a['iter']) < min(b['iter']) or max(b['iter']) < min(a['iter'])


def sign(rs, th):
    if any(r is None for r in rs):
        return ''
    v = [r[0] for r in rs]
    s = '+' if all(x > 1 + t for x, t in zip(v, th)) else '−' if all(x < 1 - t for x, t in zip(v, th)) else '0'
    return s + ('*' if s != '0' and all(r[1] for r in rs) else '')


def fmt(rs):
    return ' / '.join('-' if r is None else f"{r[0]:.3f}" for r in rs)


print("| part | cell | programs rg/rt s1 / s2 | sign | arm rg/on s1 / s2 | sign | null/on s1 / s2 | rt ms per request s1 / s2 |")
print("|---|---|---|---|---|---|---|---|")
for arg in sys.argv[1:]:
    part, fs = arg.split(':')
    ds = [ar.parse(f) for f in fs.split(',')]
    th = [floor(d) for d in ds]
    for key in sorted({k for d in ds for k in d if k[0].startswith('pt_rt_')}):
        gk = (key[0].replace('pt_rt_', 'pt_rg_'), key[1])
        prog = [ratio(d.get(key, {}).get('on'), d.get(gk, {}).get('on')) for d in ds]
        arm = [ratio(d.get(key, {}).get('on'), d.get(key, {}).get('rg')) for d in ds]
        nul = [ratio(d.get(key, {}).get('on'), d.get(key, {}).get('null')) for d in ds]
        per = ' / '.join(f"{med(d[key]['on']['iter']) / 8:.3f}" if key in d and d[key].get('on') else '-' for d in ds)
        cell = key[0][6:] + (' c' if key[1] == '_c' else '')
        print(f"| {part} | {cell} | {fmt(prog)} | {sign(prog, th)} | {fmt(arm)} | {sign(arm, th)} | {fmt(nul)} | {per} |")
