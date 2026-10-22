#!/usr/bin/env python3
"""S62 runtime measurements: the REPORT tables from two sessions per part.

usage: summarize.py <part> <results-session-1> <results-session-2>
Reads probe/submit rows with benches/gpu/arm_ratio.py's parser. Ratios are medians within one
session. `sign` is the plan's rule: |r - 1| > max(EPS, floor) in both sessions, floor = the
session's largest unchanged-code variation |null/on - 1| over the base probes' cells. `cell` is
the same test against that cell's own |null/on - 1| (a pt_rq_/pt_rt_/pt_rg_ cell takes its base
probe's). `*` = disjoint min-max ranges in every session.
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../gpu'))
import arm_ratio as ar  # noqa: E402

part, files = sys.argv[1], sys.argv[2:]
ds = [ar.parse(f) for f in files]
med = ar.med


def floors(d):
    f = {}
    for (prog, face), a in d.items():
        if a.get('null') and a.get('on') and a['null']['iter'] and a['on']['iter']:
            f[(prog, face)] = abs(med(a['null']['iter']) / med(a['on']['iter']) - 1)
    return f


fl = [floors(d) for d in ds]
th = [max(ar.EPS, max(f.values())) if f else ar.EPS for f in fl]


def cell_th(i, key):
    prog, face = key
    for p in (prog, 'pt_' + prog[6:] if prog.startswith(('pt_rq_', 'pt_rt_', 'pt_rg_')) else prog):
        if (p, face) in fl[i]:
            return max(ar.EPS, fl[i][(p, face)])
    return max(ar.EPS, statistics.median(fl[i].values())) if fl[i] else ar.EPS


def sign(rs, key=None):
    """rs: per-session (ratio, disjoint) or None. '+' = B slower than A in every session."""
    if any(r is None for r in rs):
        return 'n/a'
    t = [cell_th(i, key) if key else th[i] for i in range(len(rs))]
    if all(r[0] > 1 + x for r, x in zip(rs, t)):
        return '+' + ('*' if all(r[1] for r in rs) else '')
    if all(r[0] < 1 - x for r, x in zip(rs, t)):
        return '-' + ('*' if all(r[1] for r in rs) else '')
    return '0'


def fmt(rs):
    return ' / '.join('-' if r is None else f"{r[0]:.3f}{'*' if r[1] else ''}" for r in rs)


def tr(d, key, arm, k):
    a = d[key].get(arm)
    return med([t.get(k, 0) for t in a['trace']]) if a and a['trace'] else None


keys = sorted(set().union(*[set(d) for d in ds]))
bad = sum(len({o for a in d[k].values() for o in a['out']}) > 1 for d in ds for k in d)
print(f"### {part}\n")
print(f"Sessions: {', '.join(os.path.basename(f) for f in files)}. Null variation |null/on - 1| on iter, "
      f"median / max over base cells: {', '.join(f'{statistics.median(f.values()):.4f} / {max(f.values()):.4f}' if f else '-' for f in fl)}. "
      f"Value mismatches across arms: {bad}.\n")


def pair_table(title, A, B, progs, cols=('iter', 'wall'), extra=None):
    print(f"#### {title} ({B}/{A}; >1 = {A} faster)\n")
    hdr = "| prog | face | iter s1 / s2 | sign | cell floor s1 / s2 | cell | wall s1 / s2 | sign |" + (extra[0] if extra else "")
    print(hdr)
    print("|" + "---|" * (hdr.count('|') - 1))
    for key in keys:
        if not progs(key[0]):
            continue
        it = [ar.ratio(d, key, A, B, 'iter') for d in ds]
        if all(r is None for r in it):
            continue
        wa = [ar.ratio(d, key, A, B, 'wall') for d in ds]
        fls = ' / '.join(f"{cell_th(i, key):.3f}" for i in range(len(ds)))
        row = (f"| {key[0]} | {key[1]} | {fmt(it)} | {sign(it)} | {fls} | {sign(it, key)} | {fmt(wa)} "
               f"| {sign(wa)} |")
        if extra:
            row += extra[1](key)
        print(row)
    print()


def allocs(key):
    v = []
    for d in ds:
        a, b = tr(d, key, 'fs', 'allocs'), tr(d, key, 'predce', 'allocs')
        am = tr(d, key, 'fs', 'alloc_ms') or 0
        fm = tr(d, key, 'fs', 'free_ms') or 0
        bm = (tr(d, key, 'predce', 'alloc_ms') or 0) + (tr(d, key, 'predce', 'free_ms') or 0)
        v.append(f"{b:.0f}->{a:.0f} ({bm:.3f}->{am + fm:.3f} ms)" if a is not None and b is not None else '-')
    return ' ' + ' / '.join(v) + ' |'


base = lambda p: p.startswith('pt_') and not p.startswith(('pt_rq_', 'pt_rt_', 'pt_rg_'))
pair_table('Allocation elimination, steady clock: predce vs fs', 'fs', 'predce', base,
           extra=(" allocs predce->fs (alloc+free ms) s1 / s2 |", allocs))
pair_table('Allocation elimination, request clock: predce vs fs', 'fs', 'predce',
           lambda p: p.startswith('pt_rq_'), extra=(" allocs predce->fs (alloc+free ms) s1 / s2 |", allocs))
pair_table('Noise floor: null vs on', 'on', 'null', base)
pair_table('Streams: on (shipped OFF) vs on-streams', 'on-streams', 'on', base)
pair_table('Streams, gate off: on vs fs', 'fs', 'on', base)
pair_table('Arena: off-arena vs on', 'on', 'off-arena', base)
pair_table('Streams at one allocation policy: fs (chains, per-object buffers) vs off-arena (one stream, '
           'per-object buffers)', 'off-arena', 'fs', base)
pair_table('Retention: off-devgen vs on', 'on', 'off-devgen', lambda p: p.startswith('pt_rt_'))
pair_table('Retention: off-resident vs on', 'on', 'off-resident', lambda p: p.startswith('pt_rt_'))

print("#### Regenerated vs retained immutable inputs (pt_rg / pt_rt, arm on, 8 requests)\n")
print("| probe | face | rg/rt iter s1 / s2 | sign | per-request ms rt s1 / s2 | h2d, d2h, launches+gens per process rt / rg (s1) |")
print("|---|---|---|---|---|---|")
for key in keys:
    if not key[0].startswith('pt_rt_'):
        continue
    gk = (key[0].replace('pt_rt_', 'pt_rg_'), key[1])
    rs, per = [], []
    for d in ds:
        a, b = d[key].get('on'), d.get(gk, {}).get('on')
        if not a or not b or not a['iter'] or not b['iter']:
            rs.append(None)
            per.append('-')
            continue
        disj = max(a['iter']) < min(b['iter']) or max(b['iter']) < min(a['iter'])
        rs.append((med(b['iter']) / med(a['iter']), disj))
        per.append(f"{med(a['iter']) / 8:.3f}")
    inv = lambda k_, arm: '/'.join(f"{tr(ds[0], k_, arm, x) or 0:.0f}" for x in ('h2d', 'd2h')) + \
        f"/{(tr(ds[0], k_, arm, 'launches') or 0) + (tr(ds[0], k_, arm, 'gens') or 0):.0f}"
    print(f"| {key[0][6:]} | {key[1]} | {fmt(rs)} | {sign(rs)} | {' / '.join(per)} | "
          f"{inv(key, 'on')} vs {inv(gk, 'on') if gk in ds[0] else '-'} |")
print()

print("#### Submission (MAPAL_GPU_TRACE=2): launch enqueue = kernel_ms + gen_ms, f = enqueue / untraced iter ms\n")
print("| prog | face | arm | enqueue ms s1 / s2 | f s1 / s2 | us per launch s1 / s2 | launches+gens |")
print("|---|---|---|---|---|---|---|")
fmax = 0.0
for key in keys:
    for arm in ('on', 'fs'):
        e, f, u, n = [], [], [], None
        for d in ds:
            a = d[key].get(arm)
            if not a or not a['submit'] or not a['iter']:
                e.append('-'); f.append('-'); u.append('-')
                continue
            enq = med([t.get('kernel_ms', 0) + t.get('gen_ms', 0) for t in a['submit']])
            nl = med([t.get('launches', 0) + t.get('gens', 0) for t in a['submit']])
            fr = enq / med(a['iter'])
            fmax = max(fmax, fr) if key[0].startswith('pt_') else fmax
            e.append(f"{enq:.3f}"); f.append(f"{fr:.3f}"); u.append(f"{1000 * enq / nl:.1f}" if nl else '-')
            n = nl
        if n is None:
            continue
        print(f"| {key[0]} | {key[1]} | {arm} | {' / '.join(e)} | {' / '.join(f)} | {' / '.join(u)} | {n:.0f} |")
print(f"\nLargest f on {part}: {fmax:.3f}\n")
