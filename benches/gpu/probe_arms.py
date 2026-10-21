#!/usr/bin/env python3
"""A placement rung on the S54 probes, from box_run results files: per probe x n x face,
the shipped emission (`on`) against one arm (default `off-epilogue`) — Mapal's own clock
(`iter ms`, medians and min–max over the reps), the MAPAL_GPU_TRACE kernel time, the bytes
that crossed, whether both arms printed the same values — and the best library leg
(chain_bench.cu) against the shipped arm. Then one summary row per file.

With --vs-base, also: best library / Mapal on (`iter`), the S54 base session
(`base-<part>.txt` beside each `<item>-<part>.txt`) against this one, per probe, n, face.

usage: probe_arms.py [--arm off-<rung>] [--vs-base] <results files...>"""
import re
import sys
from collections import defaultdict

med = lambda xs: sorted(xs)[len(xs) // 2]
fields = lambda l: dict(kv.split('=', 1) for kv in re.findall(r'(\w+=[^\s\[\]]+)', l))


def load(path):
    runs, legs, gates, head, fails = defaultdict(list), defaultdict(dict), defaultdict(list), {}, []
    for l in open(path, errors='replace'):
        if l.startswith('differential:'):
            head['diff'] = l.split(':', 1)[1].strip()
        elif l.startswith('FAIL '):
            fails.append(l.strip()[:300])
        elif re.match(r'^(NVIDIA|Tesla)', l):
            head.setdefault('gpu', ', '.join(l.strip().split(', ')[:2]))
        elif l.startswith('probe prog='):
            f = fields(l)
            it = re.search(r'iter ms=\s*([-\d.eE+]+)', l)
            out = re.search(r'out=\[([^\]]*)\]', l)
            runs[(f['prog'], f['face'], f['arm'])].append({
                'iter': float(it[1]) if it else float('nan'),
                'kernel': float(f.get('kernel_ms', 'nan')),
                'xfer': float(f.get('d2h_ms', 'nan')) + float(f.get('h2d_ms', 'nan')),
                'af': float(f.get('alloc_ms', 'nan')) + float(f.get('free_ms', 'nan')),
                'launches': f.get('launches', '?'), 'out': int(f.get('bytes_out', 0)),
                'in': int(f.get('bytes_in', 0)),
                'vals': out[1].split('iter ms=')[0].split() if out else []})
        elif l.startswith('chain probe='):
            f = fields(l)
            legs[(f['probe'], f['n'])][f['leg']] = float(f['med_ms'])
        elif l.startswith('chain_values '):
            f = fields(l)
            if f['gate'] != 'pass':
                gates[(f['probe'], f['n'])].append(f"{f['leg']}:{f['gate']}")
    return head, fails, runs, legs, gates


def rng(xs):
    return f"{min(xs):.2f}–{max(xs):.2f}"


args = sys.argv[1:]
arm = 'off-epilogue'
if args[:1] == ['--arm']:
    arm, args = args[1], args[2:]
vs_base = args[:1] == ['--vs-base']
args = args[1:] if vs_base else args
summary = []
for path in args:
    head, fails, runs, legs, gates = load(path)
    name = path.split('/')[-1]
    print(f"\n### {name}: {head.get('gpu', '?')}; differential {head.get('diff', 'MISSING')}\n")
    for x in fails:
        print(f"- {x}")
    print(f"| probe | n | face | on iter ms (range) | {arm} iter ms (range) | {arm} / on | disjoint | on kernel ms "
          f"| {arm} kernel ms | on d2h+h2d ms | {arm} d2h+h2d ms | on launches | on bytes in / out | same values "
          "| best library ms (leg) | best / on iter | best / on kernel | gates |")
    print("|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|")
    ratios, mins, kr, bi, bk = defaultdict(list), [], [], defaultdict(list), defaultdict(list)
    key = lambda k: (k[0].rsplit('_', 1)[0], int(k[0].rsplit('_', 1)[1]), k[1])
    for prog, face, _ in sorted({k for k in runs if k[2] == 'on'}, key=key):
        on, off = runs[(prog, face, 'on')], runs.get((prog, face, arm), [])
        probe, n = prog[3:].rsplit('_', 1)
        oi, fi = [r['iter'] for r in on], [r['iter'] for r in off]
        ok, kk = med(oi), med([r['kernel'] for r in on])
        lg = legs.get((probe, n), {})
        best = min(lg.items(), key=lambda kv: kv[1]) if lg else None
        fc = face.strip('_')
        cells = ['', '', '', '', '', '']
        if off:
            r = med(fi) / ok
            ratios[fc].append(r)
            mins.append(min(fi) / min(oi))
            kf = med([x['kernel'] for x in off])
            kr.append(kf / kk)
            disj = max(oi) < min(fi) or max(fi) < min(oi)
            same = all(x['vals'] == on[0]['vals'] for x in on + off)
            cells = [f"{med(fi):.3f} ({rng(fi)})", f"{r:.2f}", 'yes' if disj else 'no', f"{kf:.3f}",
                     f"{med([x['xfer'] for x in off]):.2f}", 'yes' if same else 'NO']
        if best:
            bi[fc].append(best[1] / ok)
            bk[fc].append(best[1] / kk)
        print(f"| {probe} | {n} | {fc} | {ok:.3f} ({rng(oi)}) | {cells[0]} | {cells[1]} | {cells[2]} | {kk:.3f} "
              f"| {cells[3]} | {med([x['xfer'] for x in on]):.2f} | {cells[4]} | {on[0]['launches']} "
              f"| {on[0]['in']} / {on[0]['out']} | {cells[5]} "
              f"| {f'{best[1]:.3f} ({best[0]})' if best else ''} | {f'{best[1] / ok:.3f}' if best else ''} "
              f"| {f'{best[1] / kk:.2f}' if best and kk > 0 else ''} "
              f"| {' '.join(gates.get((probe, n), [])) or ('pass' if lg else '')} |")
    summary.append((name, head.get('gpu', '?'), head.get('diff', 'MISSING'), ratios, mins, kr, bi, bk))

print(f"\n### Summary: {arm} / on (>1 = the rung pays), best library / Mapal on (>1 = Mapal faster)\n")
print("| file | GPU | differential | off/on exact | off/on contract | off/on of the minima | off kernel / on kernel "
      "| best / on iter exact | best / on iter contract | best / on kernel exact | best / on kernel contract |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for name, gpu, diff, ratios, mins, kr, bi, bk in summary:
    r = lambda xs: f"{min(xs):.2f}–{max(xs):.2f}" if xs else ''
    print(f"| {name} | {gpu} | {diff} | {r(ratios['exact'])} | {r(ratios['c'])} | {r(mins)} | {r(kr)} | {r(bi['exact'])} "
          f"| {r(bi['c'])} | {r(bk['exact'])} | {r(bk['c'])} |")

if vs_base:
    import os
    base_of = lambda p: os.path.join(os.path.dirname(p), re.sub(r'^[^-]+-', 'base-', os.path.basename(p)))
    loaded = [(path, load(base_of(path)), load(path)) for path in args]
    progs = sorted({(k[0], k[1]) for _, _, (_, _, r, _, _) in loaded for k in r if k[2] == 'on'},
                   key=lambda k: (k[1], k[0].rsplit('_', 1)[0], int(k[0].rsplit('_', 1)[1])))
    print("\n### best library / Mapal iter: S54 base session → this session (best / Mapal kernel, this session)\n")
    print("| probe | n | face | " + " | ".join(os.path.basename(p) for p, _, _ in loaded) + " |")
    print("|---|---:|---|" + "---|" * len(loaded))
    for prog, face in progs:
        probe, n = prog[3:].rsplit('_', 1)
        cells = []
        for _, (_, _, br, bl, _), (_, _, er, el, _) in loaded:
            b, e, lb, le = br.get((prog, face, 'on')), er.get((prog, face, 'on')), bl.get((probe, n)), el.get((probe, n))
            if not (b and e and lb and le):
                cells.append('')
                continue
            bb, be = min(lb.values()), min(le.values())
            ie, ke = med([x['iter'] for x in e]), med([x['kernel'] for x in e])
            cells.append(f"{bb / med([x['iter'] for x in b]):.2f} → {be / ie:.2f} ({be / ke:.2f})")
        print(f"| {probe} | {n} | {face.strip('_')} | " + " | ".join(cells) + " |")
