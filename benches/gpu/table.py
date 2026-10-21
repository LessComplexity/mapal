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

    # S53 host-runtime: e2e process wall time per arm (box_profile.sh `e2e` rows).
    e2e = {}
    for l in open(path):
        if l.startswith('e2e '):
            kv = dict(x.split('=', 1) for x in l.split()[1:] if '=' in x)
            e2e.setdefault((int(kv['n']), kv['face']), {}).setdefault(kv['arm'], []).append(kv)
    if not e2e:
        continue
    from statistics import median as med
    col = lambda rows, k: [float(r.get(k, 'nan')) for r in rows]
    print("\n| n | face | arm | procs | wall ms med (min–max) | off − on ms | wait | init+ctx+module | lead | alloc | free | value |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for (n, face), arms in sorted(e2e.items()):
        on = [w / 1000 for w in col(arms.get('on', []), 'wall_us')]
        for arm, rows in arms.items():
            w = [x / 1000 for x in col(rows, 'wall_us')]
            load = [a + b + c for a, b, c in zip(col(rows, 'init_ms'), col(rows, 'ctx_ms'), col(rows, 'module_ms'))]
            d = '' if arm == 'on' or not on else f"{med(w) - med(on):+.1f}" + ('' if max(on) < min(w) or max(w) < min(on) else ' ~')
            print(f"| {n} | {face} | {arm} | {len(w)} | {med(w):.1f} ({min(w):.1f}–{max(w):.1f}) | {d} "
                  f"| {med(col(rows, 'wait_ms')):.1f} | {med(load):.1f} | {med(col(rows, 'lead_ms')):.1f} "
                  f"| {med(col(rows, 'alloc_ms')):.3f} | {med(col(rows, 'free_ms')):.3f} | {','.join(sorted({r['value'] for r in rows}))} |")
