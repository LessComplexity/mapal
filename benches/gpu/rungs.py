#!/usr/bin/env python3
"""The GPU_RUNGS_OFF rule over one part's sessions (box_run results files, different boxes).
Per rung and (fill, face) class: fill = Under at n=1024 (warp-fill acts there on every part),
Filled at n>=2048; face = Exact for _exact, Contract for _c and _tf.
At one size, speedup s = (OFF ms / its bench's cuBLAS ms) / (ON ms / its cuBLAS ms), medians within one run. A size is
  stable+  when s > 1.01 in every session (ON faster, the sign holds across boxes),
  stable-  when s < 0.99 in every session,
and `*` marks disjoint min-max ranges. The rule: a rung ships ON in a class iff some size is stable+.
Status against what ships now: ok / CONTRADICTS (strong: a size stable with disjoint ranges in every
session the other way) / contradicts (weak: no stable+ size where it ships ON, or vice versa without
disjoint ranges). Composite arms (arch-rungs = the whole table, fma-regs = idx32+affine+pad+wide-fma)
are shown, never decided.
usage: python3 benches/gpu/rungs.py <results-session-1> [<results-session-2>...] """
import re, sys
from collections import defaultdict

EPS = 0.01  # no effect within ±1%: the same ratio differs by a median 0.5-1.2% between two boxes (S54 base vs confirm)


def rows_of(path):
    rows, cur = {}, {}
    for l in open(path):
        m = re.match(r'# n=(\d+) face=(\S+)', l)
        if m:
            cur = {'n': int(m[1]), 'key': m[2]}
            continue
        m = re.match(r'time leg=(\w+) .* min_ms=([\d.]+) med_ms=([\d.]+) max_ms=([\d.]+)', l)
        if m:
            cur[m[1]] = tuple(float(x) for x in m.groups()[1:])
        if l.startswith('ratio') and 'mapal' in cur:
            rows[(cur['n'], cur['key'])] = cur
    return rows


def arms(path):
    rows, out = rows_of(path), {}
    for (n, key), r in rows.items():
        if '.' not in key:
            continue
        face, arm = key.split('.', 1)
        if not (arm.startswith('off-') or arm.startswith('on-')) or (n, face) not in rows:
            continue
        base = rows[(n, face)]
        d, rung = arm.split('-', 1)
        on, off = (base, r) if d == 'off' else (r, base)
        disj = on['mapal'][2] < off['mapal'][0] or off['mapal'][2] < on['mapal'][0]
        # Normalized by each bench's own cuBLAS leg: cancels the clock drift between benches
        # (power-capped boxes drift 5-10% over a run; table.py's cuBLAS-normalized column).
        sp = (off['mapal'][1] / off['cublas'][1]) / (on['mapal'][1] / on['cublas'][1])
        out[(rung, face, n)] = (sp, disj, 'ON' if d == 'off' else 'OFF', on['mapal'][1], off['mapal'][1])
    return out


def classes(files):
    sess = [arms(f) for f in files]
    cls = defaultdict(lambda: defaultdict(dict))  # class -> (face, n) -> session -> value
    for i, a in enumerate(sess):
        for (rung, face, n), v in a.items():
            c = (rung, 'Under' if n == 1024 else 'Filled', 'Exact' if face == '_exact' else 'Contract')
            cls[c][(face, n)][i] = v
    out = []
    for c, sizes in sorted(cls.items()):
        shipped = {v[2] for s in sizes.values() for v in s.values()}
        full = {k: s for k, s in sizes.items() if len(s) == len(sess)}  # measured in every session
        pos = [k for k, s in full.items() if all(v[0] > 1 + EPS for v in s.values())]
        neg = [k for k, s in full.items() if all(v[0] < 1 - EPS for v in s.values())]
        strong_pos = [k for k in pos if all(v[1] for v in full[k].values())]
        strong_neg = [k for k in neg if all(v[1] for v in full[k].values())]
        verdict = 'ON' if pos else 'OFF'
        sh = '/'.join(sorted(shipped))
        if c[0] in ('arch-rungs', 'fma-regs'):
            status = 'composite'
        elif sh == verdict:
            status = 'ok'
        elif (verdict == 'OFF' and strong_neg) or (verdict == 'ON' and strong_pos):
            status = 'CONTRADICTS'
        else:
            status = 'contradicts'
        cells = '; '.join(
            f"{face} {n}: " + ' '.join(f"{s[i][0]:.3f}{'*' if s[i][1] else ''}" for i in sorted(s))
            for (face, n), s in sorted(sizes.items()))
        out.append((c, sh, verdict, status, cells, sizes))
    return out, len(sess)


if __name__ == '__main__':
    out, ns = classes(sys.argv[1:])
    print(f"| rung | fill | face | shipped | rule ({ns} session{'s' if ns > 1 else ''}) | status | OFF/ON speedup per size, per session (`*` disjoint) |")
    print('|---|---|---|---|---|---|---|')
    for (rung, fill, face), sh, v, st, cells, _ in out:
        print(f"| {rung} | {fill} | {face} | {sh} | {v} | {st} | {cells} |")
