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

    # S53 host-link arms: `e2e ...` / `floor ...` lines (box_run.sh `e2e`).
    e2e, floor = {}, {}
    for l in open(path):
        if l.startswith('e2e prog='):
            f = dict(kv.split('=', 1) for kv in re.findall(r'(\w+=[^\s\[]+)', l))
            key = (f['prog'], f['face'], f['arm'])
            ph = sum(float(f.get(k + '_ms', 0)) for k in ('alloc', 'h2d', 'kernel', 'gen', 'd2h', 'free'))
            e2e.setdefault(key, []).append((int(f['wall_us']) / 1e3, ph, float(f.get('load_ms', 0)),
                                            int(f.get('bytes_in', 0)), int(f.get('bytes_out', 0))))
        elif l.startswith('floor prog='):
            f = dict(kv.split('=', 1) for kv in re.findall(r'(\w+=\S+)', l))
            floor.setdefault(f['prog'], []).append((float(f['total_ms']), float(f['after_ctx_ms'])))
    if e2e:
        med = lambda xs: sorted(xs)[len(xs) // 2]
        rng = lambda xs: f"{min(xs):.1f}–{max(xs):.1f}"
        print("\n| program | face | arm | wall ms (med, range) | phases ms (med, excl. load) | bytes in | bytes out | arm/ON wall (>1 = ON faster) | arm/ON phases | cuBLAS floor ms (total / after ctx) |")
        print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for (p, face, arm), rows in e2e.items():
            w, ph = [r[0] for r in rows], [r[1] for r in rows]
            on = e2e.get((p, face, 'on'))
            rw = rp = ''
            if on and arm != 'on':
                rw = f"{med(w) / med([r[0] for r in on]):.2f}×"
                rp = f"{med(ph) / med([r[1] for r in on]):.2f}×"
            fl = floor.get(p)
            fs = f"{med([x[0] for x in fl]):.1f} / {med([x[1] for x in fl]):.2f}" if fl and arm == 'on' else ''
            print(f"| {p} | {face} | {arm} | {med(w):.1f} ({rng(w)}) | {med(ph):.2f} ({rng(ph)}) | {rows[0][3]} | {rows[0][4]} | {rw} | {rp} | {fs} |")
