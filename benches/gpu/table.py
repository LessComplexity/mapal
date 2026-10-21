#!/usr/bin/env python3
"""Tabulate a box_run results file: medians, TFLOP/s, ratios, overlap, geometry."""
import re
import sys

for path in sys.argv[1:]:
    rows, cur = [], {}
    for l in open(path):
        m = re.match(r'# n=(\d+) face=(\S+)', l)
        if m:
            cur = {'n': m[1], 'face': {'_exact': 'exact', '_c': 'FMA SHIPPED', '_tf': 'TF32 tensor'}.get(m[2], ('tf32 ' + m[2].removeprefix('_tf.')) if m[2].startswith('_tf.') else 'fma ' + m[2].removeprefix('_c.'))}
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

    # S53 decode e2e (box_run.sh): per program arm, wall and traced phases over
    # the reps; medians, min–max for the overlap call.
    wall, ph, launches, floor = {}, {}, {}, None
    for l in open(path):
        m = re.match(r'decode prog=(\S+) rep=\d+ rc=0 wall_s=([\d.]+)', l)
        if m:
            wall.setdefault(m[1], []).append(float(m[2]))
        m = re.match(r'decode_trace prog=(\S+) rep=\d+ phases (.*)', l)
        if m:
            d = dict(kv.split('=') for kv in m[2].split())
            ph.setdefault(m[1], []).append({k: float(v) for k, v in d.items()})
        m = re.match(r'decode_trace prog=(\S+) rep=1 launch kernel=(\S+) out=(\d+) n=(\d+)', l)
        if m:
            launches.setdefault(m[1], []).append(f"{m[4]}×{int(m[3]) // 1024}K")
        m = re.match(r'decode_floor T=\d+ d=\d+ .* med_ms=([\d.]+)', l)
        if m:
            floor = float(m[1])
    if wall:
        med = lambda v: sorted(v)[len(v) // 2]
        print("\n| decode 1024×256 | wall s (med, min–max) | ON/OFF | kernel ms | h2d ms | d2h ms | alloc+free ms | load ms | launches (n×out) |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for prog in sorted(wall, key=lambda p: (p.split('.')[0], '.off' not in p)):
            w = wall[prog]
            off = wall.get(prog.split('.')[0] + '.off-retain-delta') if '.off-' not in prog else None
            ratio = ''
            if off:
                ov = '' if (max(w) < min(off) or max(off) < min(w)) else ' ~'
                ratio = f"{med(off) / med(w):.2f}×{ov}"
            p = ph.get(prog, [])
            g = lambda k: med([x.get(k, 0) for x in p]) if p else float('nan')
            print(f"| {prog} | {med(w):.3f} ({min(w):.3f}–{max(w):.3f}) | {ratio} | {g('kernel_ms'):.1f} | {g('h2d_ms'):.1f} "
                  f"| {g('d2h_ms'):.1f} | {g('alloc_ms') + g('free_ms'):.1f} | {g('load_ms'):.1f} | {' '.join(launches.get(prog, []))} |")
        if floor is not None:
            print(f"\nhand-written cuBLAS sgemv KV cache (device time, whole loop): {floor:.3f} ms")
