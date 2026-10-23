#!/usr/bin/env python3
"""Per-cell cross-part table of one arm pair: `B/A` on iter ms, `s1 / s2` per part, sign by the
summarize.py plan rule (session floor = largest base-cell |null/on - 1|), `*` disjoint in both.

usage: cells.py <A> <B> <prog-regex> <part>:<s1>,<s2> ...
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../gpu'))
import arm_ratio as ar  # noqa: E402


def floor(d):  # digest.py's session floor
    r = [abs(ar.med(a['null']['iter']) / ar.med(a['on']['iter']) - 1)
         for a in d.values() if a.get('null') and a.get('on') and a['null']['iter'] and a['on']['iter']]
    return max([ar.EPS] + r)


A, B, rx = sys.argv[1], sys.argv[2], re.compile(sys.argv[3])
parts = []
for arg in sys.argv[4:]:
    part, fs = arg.split(':')
    ds = [ar.parse(f) for f in fs.split(',')]
    parts.append((part, ds, [floor(d) for d in ds]))
keys = sorted({k for _, ds, _ in parts for d in ds for k in d if rx.fullmatch(k[0])})
print('| cell | ' + ' | '.join(p for p, _, _ in parts) + ' |')
print('|---|' + '---|' * len(parts))
for key in keys:
    row = []
    for _, ds, th in parts:
        rs = [ar.ratio(d, key, A, B, 'iter') if key in d else None for d in ds]
        if any(r is None for r in rs):
            row.append('-')
            continue
        v = [r[0] for r in rs]
        s = '+' if all(x > 1 + t for x, t in zip(v, th)) else '−' if all(x < 1 - t for x, t in zip(v, th)) else '0'
        star = '*' if s != '0' and all(r[1] for r in rs) else ''
        row.append(f"{v[0]:.2f} / {v[1]:.2f} {s}{star}")
    print(f"| {key[0][3:]}{key[1].replace('_exact', '').replace('_c', ' c')} | " + ' | '.join(row) + ' |')
