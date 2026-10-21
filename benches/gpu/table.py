#!/usr/bin/env python3
"""Tabulate a box_run results file: medians, TFLOP/s, ratios, overlap, geometry;
then every rung arm against the default it flips (ON vs OFF, medians, and the
same ratio normalized by each bench's own cuBLAS leg, which cancels clock drift
between benches); then the whole-program rows (`e2e prog=`: process wall, the
program's own `iter ms`, the MAPAL_GPU_TRACE totals) and the decode rows."""
import re
import sys

med = lambda xs: sorted(xs)[len(xs) // 2]
rng = lambda xs: f"{min(xs):.1f}–{max(xs):.1f}"
# ` ~`: the two min–max ranges overlap.
ovl = lambda a, b: '' if (max(a) < min(b) or max(b) < min(a)) else ' ~'
PHASES = ('alloc', 'h2d', 'kernel', 'gen', 'd2h', 'free')


def fields(l):
    return dict(kv.split('=', 1) for kv in re.findall(r'(\w+=[^\s\[\]]+)', l))


def e2e_table(path):
    """`e2e prog=` lines: one process each, every arm interleaved per rep."""
    e2e, floor = {}, {}
    for l in open(path):
        if l.startswith('e2e prog='):
            f = fields(l)
            it = re.search(r'iter ms=\s*([-\d.eE+]+)', l)
            e2e.setdefault((f['prog'], f['face']), {}).setdefault(f['arm'], []).append({
                'wall': int(f['wall_us']) / 1e3,
                'phases': sum(float(f.get(p + '_ms', 0)) for p in PHASES),
                'load': float(f.get('load_ms', 0)), 'lead': float(f.get('lead_ms', 0)),
                'iter': float(it[1]) if it else None,
                'in': int(f.get('bytes_in', 0)), 'out': int(f.get('bytes_out', 0))})
        elif l.startswith('floor prog='):
            f = fields(l)
            floor.setdefault(f['prog'], []).append(float(f['total_ms']))
    if not e2e:
        return
    print("\n| program | face | arm | procs | wall ms med (range) | wall arm/on (>1 = on faster) | iter ms | "
          "phases ms excl. load | load wait ms | lead ms | bytes in | bytes out | cuBLAS floor ms |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (p, face), arms in e2e.items():
        on = arms.get('on')
        for arm, rs in arms.items():
            w = [r['wall'] for r in rs]
            ratio = f"{med(w) / med([r['wall'] for r in on]):.3f}{ovl(w, [r['wall'] for r in on])}" if on and arm != 'on' else ''
            its = [r['iter'] for r in rs if r['iter'] is not None]
            fl = f"{med(floor[p]):.1f}" if arm == 'on' and p in floor else ''
            print(f"| {p} | {face} | {arm} | {len(rs)} | {med(w):.1f} ({rng(w)}) | {ratio} "
                  f"| {f'{med(its):.3f}' if its else ''} | {med([r['phases'] for r in rs]):.2f} "
                  f"| {med([r['load'] for r in rs]):.1f} | {med([r['lead'] for r in rs]):.1f} "
                  f"| {rs[0]['in']} | {rs[0]['out']} | {fl} |")


def decode_table(path):
    """`decode prog=` lines (box_run.sh decode block)."""
    runs, launches, floor = {}, {}, None
    for l in open(path):
        if l.startswith('decode prog='):
            f = fields(l)
            runs.setdefault(f['prog'], []).append(f)
        elif l.startswith('decode_launch prog='):
            f = fields(l)
            launches.setdefault(f['prog'], []).append(f"{f['n']}×{f['kernel'][-6:]}")
        m = re.match(r'decode_floor T=\d+ d=\d+ .* med_ms=([\d.]+)', l)
        if m:
            floor = float(m[1])
    if not runs:
        return
    print("\n| decode | procs | wall s med (range) | on/arm (>1 = shipped faster) | kernel ms | h2d ms | d2h ms "
          "| alloc+free ms | load ms | launches |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for prog, rs in sorted(runs.items()):
        w = [float(r['wall_s']) for r in rs]
        base = runs.get(prog.split('.')[0])
        ratio = ''
        if base and '.' in prog:
            bw = [float(r['wall_s']) for r in base]
            ratio = f"{med(w) / med(bw):.2f}{ovl(w, bw)}"
        g = lambda k: med([float(r.get(k, 'nan')) for r in rs])
        print(f"| {prog} | {len(rs)} | {med(w):.3f} ({min(w):.3f}–{max(w):.3f}) | {ratio} | {g('kernel_ms'):.1f} "
              f"| {g('h2d_ms'):.1f} | {g('d2h_ms'):.1f} | {g('alloc_ms') + g('free_ms'):.1f} | {g('load_ms'):.1f} "
              f"| {' '.join(launches.get(prog, []))} |")
    if floor is not None:
        print(f"\nhand-written cuBLAS sgemv KV cache (device time, whole loop): {floor:.3f} ms")


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
    decode_table(path)
