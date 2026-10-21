#!/usr/bin/env python3
"""Tabulate a box_run results file: medians, TFLOP/s, ratios, overlap, geometry;
then every rung arm against the default it flips (ON vs OFF, medians, and the
same ratio normalized by each bench's own cuBLAS leg, which cancels clock drift
between benches); and its `== e2e` sections (box_profile.sh): iter ms against
the site trace."""
import re
import sys

PHASES = ('load', 'lookup', 'alloc', 'h2d', 'kernel', 'd2h', 'free')


def e2e_table(path):
    """`== e2e <program>` sections (box_profile.sh output, one process per
    `wall_ms=` line): the program's own `iter ms` against its site's trace."""
    runs, prog, cur = {}, None, {}
    for l in open(path):
        m = re.match(r'== e2e (\S+)', l)
        if m:
            prog, cur = m[1], {}
            continue
        if prog is None:
            continue
        if l.startswith('mapal_gpu_trace'):
            cur.update({p: float(re.search(rf'{p}_ms=([\d.]+)', l)[1]) for p in PHASES})
        elif l.startswith('iter ms='):
            cur['iter'] = float(l.split('=')[1])
        elif l.startswith('wall_ms='):
            cur['wall'] = float(l.split('=')[1])
            runs.setdefault(prog, []).append(cur)
            cur = {}
    if not runs:
        return
    med = lambda xs: sorted(xs)[len(xs) // 2]
    print("\n| program | procs | iter ms | trace sum | iter − sum (max abs) | load ms | site w/o load | wall ms |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for prog, rs in runs.items():
        total = [sum(r[p] for p in PHASES) for r in rs]
        diff = max((r['iter'] - t for r, t in zip(rs, total)), key=abs)
        print(f"| {prog} | {len(rs)} | {med([r['iter'] for r in rs]):.3f} | {med(total):.3f} | {diff:+.3f} "
              f"| {med([r['load'] for r in rs]):.1f} | {med([t - r['load'] for r, t in zip(rs, total)]):.3f} "
              f"| {med([r['wall'] for r in rs]):.0f} |")


for path in sys.argv[1:]:
    rows, cur = [], {}
    for l in open(path):
        m = re.match(r'# n=(\d+) face=(\S+)', l)
        if m:
            face, _, arm = m[2].partition('.')
            cur = {'n': m[1], 'key': m[2], 'face': {'_exact': 'exact', '_c': 'FMA SHIPPED', '_tf': 'TF32 tensor'}[face] if not arm
                   else {'_exact': 'exact', '_c': 'fma', '_tf': 'tf32'}[face] + ' ' + arm}
            continue
        m = re.match(r'mapal kernel=\S+ grid=(\S+) block=(\d+) smem=(\d+) regs=(\d+) local_bytes=(\d+)', l)
        if m:
            cur['geom'] = f"{m[1]}×{m[2]}t {int(m[3]) // 1024}K r{m[4]}" + (f" SPILL {m[5]}" if m[5] != '0' else '')
        m = re.match(r'time leg=(\w+) .* min_ms=([\d.]+) med_ms=([\d.]+) max_ms=([\d.]+) tflops_med=([\d.]+)', l)
        if m:
            cur[m[1]] = tuple(float(x) for x in m.groups()[1:])
        m = re.match(r'values leg=mapal .*(?:bit_exact_vs_in_order_fold|within_4x_cublas_tf32)=(\w+)', l)
        if m:
            cur['bits'] = m[1]
        if l.startswith('ratio'):
            rows.append(cur)
    print(f"\n{path}")
    print("| n | face | Mapal ms | TF/s | cuBLAS ms | TF/s | CUTLASS ms | TF/s | vs cuBLAS | vs CUTLASS | bits | geometry |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    ov = lambda a, b: '' if (a[2] < b[0] or b[2] < a[0]) else ' ~'
    for r in rows:
        m, c, t = r['mapal'], r['cublas'], r['cutlass']
        print(f"| {r['n']} | {r['face']} | {m[1]:.3f} | {m[3]:.1f} | {c[1]:.3f} | {c[3]:.1f} | {t[1]:.3f} | {t[3]:.1f} "
              f"| {c[1] / m[1]:.2f}×{ov(m, c)} | {t[1] / m[1]:.2f}×{ov(m, t)} | {r.get('bits')} | {r.get('geom')} |")

    by = {(r['n'], r['key']): r for r in rows}
    arms = [r for r in rows if '.' in r['key'] and r['key'].split('.', 1)[1][:3] in ('off', 'on-')]
    if arms:
        print("\n| n | face | rung | default | ON ms | OFF ms | ON/OFF speedup | cuBLAS-normalized |")
        print("|---|---|---|---|---:|---:|---:|---:|")
    for r in arms:
        face, arm = r['key'].split('.', 1)
        base = by.get((r['n'], face))
        if not base:
            continue
        direction, rung = arm.split('-', 1)
        on, off = (base, r) if direction == 'off' else (r, base)
        speed = off['mapal'][1] / on['mapal'][1]
        norm = (off['mapal'][1] / off['cublas'][1]) / (on['mapal'][1] / on['cublas'][1])
        print(f"| {r['n']} | {face} | {rung} | {'ON' if direction == 'off' else 'OFF'} | {on['mapal'][1]:.3f} "
              f"| {off['mapal'][1]:.3f} | {speed:.3f}{ov(on['mapal'], off['mapal'])} | {norm:.3f} |")
    e2e_table(path)
