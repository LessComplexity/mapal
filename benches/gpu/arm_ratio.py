#!/usr/bin/env python3
"""Arm A/B over box_run.sh probe rows (S62 runtime measurements).

Per (prog, face, arm): medians and min-max of the program's own `iter ms` and the process
`wall_us` (untraced process); the traced process's counts (allocs, h2d, d2h, launches,
events, bytes) and `alloc_ms + free_ms`; from `submit` rows (MAPAL_GPU_TRACE=2: no
post-launch synchronize) the launch enqueue `kernel_ms + gen_ms` and its fraction `f` of
the untraced `iter ms`. Every arm of one (prog, face) must print the same values (`out=`,
the clock excluded): a mismatch is printed as VALUE MISMATCH and fails the exit status.

`--pair A B` prints B/A on `iter ms` and `wall_us` per (prog, face) in each session, `*` for
disjoint min-max ranges; over several sessions (different boxes) a sign counts only when
the ratio is beyond 1 ± max(EPS, floor) in every session, floor = that session's largest
|null/on - 1| on `iter ms` (rungs.py's rule with the box's own noise floor).

usage: arm_ratio.py [--pair A B]... <results-session-1> [<results-session-2>...]
       arm_ratio.py --selftest
"""
import re
import statistics
import sys
from collections import defaultdict

EPS = 0.01  # rungs.py: no effect within ±1%
ROW = re.compile(r'(probe|submit) prog=(\S+) face=(\S+) arm=(\S+) rep=\d+ (?:wall_us=(\d+) .*?)?out=\[(.*?)\]')
KEYS = ('alloc_ms', 'free_ms', 'kernel_ms', 'gen_ms', 'h2d_ms', 'd2h_ms', 'allocs', 'h2d', 'd2h',
        'launches', 'gens', 'events', 'bytes_in', 'bytes_out')


def parse(path):
    """{(prog, face): {arm: {'iter': [], 'wall': [], 'trace': [dict], 'submit': [dict], 'out': set}}}"""
    d = defaultdict(lambda: defaultdict(lambda: {'iter': [], 'wall': [], 'trace': [], 'submit': [],
                                                 'out': set()}))
    for l in open(path):
        m = ROW.match(l)
        if not m:
            continue
        kind, prog, face, arm, wall, out = m.groups()
        a = d[(prog, face)][arm]
        it = re.search(r'iter ms=([\d.e+-]+)', out)
        t = {k: float(v) for k, v in re.findall(r'\b(' + '|'.join(KEYS) + r')=([\d.]+)', l)}
        if kind == 'probe':
            a['out'].add(out.split('iter ms=')[0].strip())
            if it:
                a['iter'].append(float(it[1]))
            a['wall'].append(int(wall) / 1000)
            a['trace'].append(t)
        else:
            a['out'].add(out.split('iter ms=')[0].strip())
            if it:
                t['iter'] = float(it[1])
            a['submit'].append(t)
    return d


def med(xs):
    return statistics.median(xs) if xs else float('nan')


def floor(d):
    """The session's noise floor: the largest |null/on - 1| on `iter ms`."""
    r = [abs(med(a['null']['iter']) / med(a['on']['iter']) - 1)
         for a in d.values() if a.get('null') and a.get('on') and a['on']['iter'] and a['null']['iter']]
    return max(r) if r else 0.0


def ratio(d, key, A, B, col):
    a, b = d[key].get(A), d[key].get(B)
    if not a or not b or not a[col] or not b[col]:
        return None
    disj = max(a[col]) < min(b[col]) or max(b[col]) < min(a[col])
    return med(b[col]) / med(a[col]), disj


def table(path, d):
    bad = 0
    print(f"## {path}  (noise floor |null/on-1| = {floor(d):.4f})")
    print("| prog | face | arm | n | iter ms med (min-max) | wall ms med | allocs | alloc+free ms | h2d | d2h "
          "| launches+gens | events | enqueue ms (f) | enqueue us/launch |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (prog, face), arms in sorted(d.items()):
        outs = {o for a in arms.values() for o in a['out']}
        if len(outs) > 1:
            bad += 1
            print(f"VALUE MISMATCH {prog} {face}: {sorted(outs)}")
        for arm, a in sorted(arms.items()):
            tr = a['trace']
            g = lambda k, src=tr: med([t.get(k, 0) for t in src])
            sub = a['submit']
            enq = med([t.get('kernel_ms', 0) + t.get('gen_ms', 0) for t in sub]) if sub else None
            nl = med([t.get('launches', 0) + t.get('gens', 0) for t in sub]) if sub else None
            it = a['iter']
            ec = (f"{enq:.3f} ({enq / med(it):.3f})" if enq is not None and it else "")
            eu = f"{1000 * enq / nl:.1f}" if enq is not None and nl else ""
            iters = f"{med(it):.4f} ({min(it):.4f}-{max(it):.4f})" if it else "-"
            print(f"| {prog} | {face} | {arm} | {len(it) or len(a['wall'])} | {iters} | {med(a['wall']):.1f} "
                  f"| {g('allocs'):.0f} | {g('alloc_ms') + g('free_ms'):.3f} | {g('h2d'):.0f} | {g('d2h'):.0f} "
                  f"| {g('launches') + g('gens'):.0f} | {g('events'):.0f} | {ec} | {eu} |")
    return bad


def pairs(files, ds, prs):
    fl = [max(EPS, floor(d)) for d in ds]
    print(f"\n## pairs (B/A; >1 = A faster). thresholds per session: {', '.join(f'{x:.4f}' for x in fl)}")
    print("| A | B | prog | face | iter B/A per session | wall B/A per session | sign (iter) |")
    print("|---|---|---|---|---|---|---|")
    for A, B in prs:
        keys = sorted(set().union(*[set(d) for d in ds]))
        for key in keys:
            it = [ratio(d, key, A, B, 'iter') for d in ds]
            wa = [ratio(d, key, A, B, 'wall') for d in ds]
            if all(x is None for x in it):
                continue
            f = lambda xs: ' '.join('-' if x is None else f"{x[0]:.4f}{'*' if x[1] else ''}" for x in xs)
            if any(x is None for x in it):
                sign = 'unmeasured in a session'
            elif all(x[0] > 1 + t for x, t in zip(it, fl)):
                sign = 'stable+ (B slower)'
            elif all(x[0] < 1 - t for x, t in zip(it, fl)):
                sign = 'stable- (B faster)'
            else:
                sign = 'none'
            print(f"| {A} | {B} | {key[0]} | {key[1]} | {f(it)} | {f(wa)} | {sign} |")


def selftest():
    import os
    import tempfile
    rows = []
    for r, (on, null, x) in enumerate([(1.0, 1.01, 1.2), (1.02, 1.0, 1.25), (0.99, 1.0, 1.22)]):
        for arm, v in (('on', on), ('null', null), ('x', x)):
            rows.append(f"probe prog=pt_p_512 face=_exact arm={arm} rep={r} wall_us=300000 traced_wall_us=1 "
                        f"out=[1.5 2.5 iter ms={v} ] mapal_gpu_trace_total alloc_ms=0.5 free_ms=0.25 allocs=7 "
                        f"h2d=0 d2h=4 launches=4 gens=2 events=1 rc=0 trace_rc=0")
        rows.append(f"submit prog=pt_p_512 face=_exact arm=on rep={r} out=[1.5 2.5 iter ms=2.0 ] "
                    f"mapal_gpu_trace_total kernel_ms=0.03 gen_ms=0.01 launches=4 gens=4 rc=0")
    p = tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False)
    p.write('\n'.join(rows) + '\n')
    p.close()
    d = parse(p.name)
    a = d[('pt_p_512', '_exact')]
    assert a['on']['iter'] == [1.0, 1.02, 0.99] and len(a['on']['submit']) == 3
    assert abs(floor(d) - 0.0) < 0.02
    r, disj = ratio(d, ('pt_p_512', '_exact'), 'on', 'x', 'iter')
    assert abs(r - 1.22) < 1e-9 and disj
    assert table(p.name, d) == 0
    with open(p.name, 'a') as f:
        f.write(rows[0].replace('out=[1.5', 'out=[9.5') + '\n')
    assert table(p.name, parse(p.name)) == 1
    os.unlink(p.name)
    print('selftest ok')


if __name__ == '__main__':
    args = sys.argv[1:]
    if args == ['--selftest']:
        selftest()
        sys.exit()
    prs = []
    while args[:1] == ['--pair']:
        prs.append((args[1], args[2]))
        args = args[3:]
    ds = [parse(f) for f in args]
    bad = sum(table(f, d) for f, d in zip(args, ds))
    if prs:
        pairs(args, ds, prs)
    sys.exit(1 if bad else 0)
