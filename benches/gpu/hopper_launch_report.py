#!/usr/bin/env python3
"""hopper-launch: the table of a hopper_launch_box.sh run (results-<part>.txt).

Per program: each arm's median `iter ms` over its repetitions, the shipped arm's null (|null / on - 1|, its second
copy), the traced kernel time of the shipped arm and the old runtime, the host time a call spent on tensor maps
(`tmap_ms`) and the init's TMA warm (`tma_warm_ms`), and the fastest library leg (gemm_lib's TF32 legs, median).
usage: hopper_launch_report.py <results-file>
"""
import re
import statistics
import sys
from collections import defaultdict

runs = defaultdict(lambda: defaultdict(list))
trace = defaultdict(dict)
lib = defaultdict(dict)
probes = defaultdict(list)
for line in open(sys.argv[1], errors='replace'):
    m = re.match(r'run prog=(\S+) arm=(\S+) rep=\d+ out=\[(.*)\]', line)
    if m:
        t = re.search(r'iter ms=\s*([0-9.eE+-]+)', m.group(3))
        if t:
            runs[m.group(1)][m.group(2)].append(float(t.group(1)))
        continue
    m = re.match(r'trace prog=(\S+) arm=(\S+) (.*)', line)
    if m:
        f = dict(re.findall(r'(\w+)=([0-9.]+)', m.group(3).split('op=launch')[0]))
        ks = [float(x) for x in re.findall(r'op=launch kernel=\S+ ms=([0-9.]+)', m.group(3))]
        f['kernels'] = ks
        trace[m.group(1)][m.group(2)] = f
        continue
    m = re.match(r'gemm prog=(\S+) leg=(\S+) .*med_ms=([0-9.]+)', line)
    if m:
        lib[m.group(1)][m.group(2)] = float(m.group(3))
        continue
    m = re.match(r'probe rt=new rep=\d+ (self )?.*?flags=(\d+) first_us=([0-9.]+)', line)
    if m:
        probes[('self ' if m.group(1) else '') + m.group(2)].append(float(m.group(3)))

med = statistics.median
print('launch probes (first clocked TMA launch, us, median [min-max]):')
for k, v in sorted(probes.items()):
    print(f'  {k}: {med(v):.1f} [{min(v):.1f}-{max(v):.1f}] n={len(v)}')
arms = ['off', 'old', 'cold', 'one', 'on', 'order', 'rast']
print('| program | ' + ' | '.join(arms) + ' | null | on/old | on/off | kernel on / old (ms) | tmap_ms old / on |'
      ' tma_warm_ms | lib best (leg) |')
print('|' + '---|' * (len(arms) + 8))
for p, a in runs.items():
    m = {k: med(v) for k, v in a.items()}
    on = m.get('on')
    null = abs(m['null'] / on - 1) if on and 'null' in m else float('nan')
    cells = [f'{m[k]:.4f}' if k in m else '=' for k in arms]
    t = trace.get(p, {})
    ko = sum(t.get('on', {}).get('kernels', [])) if 'on' in t else float('nan')
    kd = sum(t.get('old', {}).get('kernels', [])) if 'old' in t else float('nan')
    tm = f"{t.get('old', {}).get('tmap_ms', '-')} / {t.get('on', {}).get('tmap_ms', '-')}"
    tw = t.get('on', {}).get('tma_warm_ms', '-')
    best = min(lib[p].items(), key=lambda x: x[1]) if lib.get(p) else None
    bs = f'{best[1]:.4f} ({best[0]})' if best else '-'
    print(f'| {p} | ' + ' | '.join(cells) + f' | {null:.1%} | {m["old"] / on:.2f} | {m["off"] / on:.2f} | '
          f'{ko:.3f} / {kd:.3f} | {tm} | {tw} | {bs} |')
