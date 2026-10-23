#!/usr/bin/env python3
"""S62 `hostopt` / `place` from box_run results files: per program x face, the median
`iter ms` (Mapal's own clock) and process wall of every arm, each arm's ratio to the
shipped build (`on`; > 1 = `on` faster), the null arm's (a copy of `on`: the box's
variation), and the calibration facts the profile rows cite:

  DeviceCost.flops_per_ns  tile_bench mm_4096 contract face, Mapal leg tflops_med x 1000
  DeviceCost.bytes_per_ns  pt_pj_m1_n2048_k7168 `on`: its site's bytes / the site kernel's ms
  DeviceCost.launch_ns     pt_pc_64 all-device build (`on` in a run without costs, else
                           `off-place`): median iter ms / its 3 sites
  LaneRate.flops_per_ns    pt_pc_128 CPU build at MAPAL_PAR=1: 3 x 2 x 128^3 / iter ms
  LaneRate.bytes_per_ns    pt_pj_m1_n2048_k7168 CPU build at MAPAL_PAR=1: site bytes / iter ms
  device start (reported, no profile field): pt_pc_32's process wall, all-device - CPU build
  DispatchCost.dispatch_ns pt_pc_32 / pt_pc_64: (CPU build - its every-task-inline build,
                           `cpu-inline`) / the 3 tile tasks between the clocks (the S51 method)
  (no mm_4096 timing in the file: DeviceCost.flops_per_ns from pt_pj_m128_n18432_k7168
   contract face, its site's flops / the site kernel's ms)

usage: cpugpu.py <results files...>"""
import re
import sys
from collections import defaultdict

med = lambda xs: sorted(xs)[len(xs) // 2]
fields = lambda l: dict(kv.split('=', 1) for kv in re.findall(r'(\w+=[^\s\[\]]+)', l))
PJ_M1_BYTES = (7168 + 7168 * 2048 + 2048) * 4


def load(path):
    runs, kern, lane, mm, head = defaultdict(list), defaultdict(list), defaultdict(list), {}, []
    face = None
    for l in open(path, errors='replace'):
        if l.startswith(('lscpu: Model name', 'cpu_flags', '== ')) or re.match(r'^(NVIDIA|Tesla)', l):
            head.append(l.strip())
        elif l.startswith('probe prog='):
            f = fields(l)
            it = re.search(r'iter ms=\s*([-\d.eE+]+)', l)
            if it:
                runs[(f['prog'], f['face'], f['arm'])].append((float(it[1]), int(f['wall_us'])))
        elif l.startswith('probe_kernels prog='):
            f = fields(l)
            ms = [float(x) for x in re.findall(r'ms=([\d.]+)', l)]
            kern[(f['prog'], f['face'], f['arm'])].append(ms)
        elif l.startswith('lane prog='):
            f = fields(l)
            it = re.search(r'iter ms=\s*([-\d.eE+]+)', l)
            if it:
                lane[(f['prog'], f['face'])].append(float(it[1]))
        elif l.startswith('# n='):
            face = l.split()[1:]
        elif l.startswith('time leg=mapal') and face == ['n=4096', 'face=_c']:
            mm['tflops'] = float(fields(l)['tflops_med'])
    return head, runs, kern, lane, mm


for path in sys.argv[1:]:
    head, runs, kern, lane, mm = load(path)
    print(f"## {path}")
    for h in head:
        print(f"    {h}")
    progs = sorted({(p, fa) for p, fa, _ in runs})
    print("| program | face | on iter ms | arm: OFF/on iter (wall) | null/on iter |")
    print("| --- | --- | ---: | --- | ---: |")
    nulls = []
    for p, fa in progs:
        on = runs.get((p, fa, 'on'))
        if not on:
            continue
        oi, ow = med([x[0] for x in on]), med([x[1] for x in on])
        cells, nul = [], ''
        for (q, fb, arm), xs in sorted(runs.items()):
            if q != p or fb != fa or arm == 'on':
                continue
            ai, aw = med([x[0] for x in xs]), med([x[1] for x in xs])
            if arm == 'null':
                nul = f"{ai / oi:.3f}"
                nulls.append(abs(ai / oi - 1))
            else:
                cells.append(f"{arm} {ai / oi:.3f} ({aw / ow:.2f})")
        print(f"| {p} | {fa} | {oi:.4f} (n={len(on)}, wall {ow / 1000:.0f} ms) | {'; '.join(cells)} | {nul} |")
    if nulls:
        print(f"null |null/on - 1| over programs: median {med(nulls):.3f}, max {max(nulls):.3f}")
    # The calibration facts.
    cal = {}
    if 'tflops' in mm:
        cal['DeviceCost.flops_per_ns'] = round(mm['tflops'] * 1000)
    elif kern.get(('pt_pj_m128_n18432_k7168', '_c', 'on')):
        site_ms = med([max(x) for x in kern[('pt_pj_m128_n18432_k7168', '_c', 'on')] if x])
        cal['DeviceCost.flops_per_ns(pj_m128)'] = round(2 * 128 * 18432 * 7168 / (site_ms * 1e6))
    k = kern.get(('pt_pj_m1_n2048_k7168', '_exact', 'on'))
    if k:
        site_ms = med([max(x) for x in k if x])
        cal['DeviceCost.bytes_per_ns'] = round(PJ_M1_BYTES / (site_ms * 1e6))
    for arm in ('off-place', 'on'):
        r = runs.get(('pt_pc_64', '_exact', arm))
        if r:
            cal['DeviceCost.launch_ns'] = round(med([x[0] for x in r]) * 1e6 / 3)
            cal['(launch from arm)'] = arm
            break
    r = lane.get(('pt_pc_128', '_exact'))
    if r:
        cal['LaneRate.flops_per_ns'] = round(3 * 2 * 128 ** 3 / (med(r) * 1e6))
    r = lane.get(('pt_pj_m1_n2048_k7168', '_exact'))
    if r:
        cal['LaneRate.bytes_per_ns'] = round(PJ_M1_BYTES / (med(r) * 1e6))
    gpu_arm = next((a for a in ('off-place', 'off-hostopt', 'on') if runs.get(('pt_pc_32', '_exact', a))), None)
    if gpu_arm and runs.get(('pt_pc_32', '_exact', 'cpu')):
        w = lambda a: med([x[1] for x in runs[('pt_pc_32', '_exact', a)]])
        cal['device start ms (pc_32 wall, GPU - CPU build)'] = round(w(gpu_arm) / 1000 - w('cpu') / 1000, 1)
    for p in ('pt_pc_32', 'pt_pc_64'):
        a, b = runs.get((p, '_exact', 'cpu')), runs.get((p, '_exact', 'cpu-inline'))
        if a and b:
            cal[f'DispatchCost.dispatch_ns({p})'] = round(
                (med([x[0] for x in a]) - med([x[0] for x in b])) * 1e6 / 3)
    print("calibration:", ", ".join(f"{k}={v}" for k, v in cal.items()))
    print()
