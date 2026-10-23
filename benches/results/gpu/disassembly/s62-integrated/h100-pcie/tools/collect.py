#!/usr/bin/env python3
"""Compact raw evidence for the repository from a fetched box trace.

usage: collect.py <trace dir> <libsass.json> <out raw dir>
- executed-sass/<kernel>.<module sha8>.sass: the executed function's text from the nvdisasm of the
  observer-saved module, + index.json (kernel, capture, module file + sha256, launches in region, grid)
- nsys/<run>_{kernels,api}.csv: kernel and CUDA API rows exported from each nsys sqlite
- capture-events.tar.xz: every observer events.tsv (+ module sha256 list)
"""
import csv
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from sass_stats import NVSEC  # noqa: E402

trace, ls_path, out = sys.argv[1:4]
os.makedirs(os.path.join(out, 'executed-sass'), exist_ok=True)
os.makedirs(os.path.join(out, 'nsys'), exist_ok=True)


def short(n):
    m = re.search(r'GemmShapeILi(\d+)ELi(\d+)ELi(\d+)E', n)
    if 'cutlass6Kernel' in n and m:
        return f'cutlass_Gemm_{m.group(1)}_{m.group(2)}_{m.group(3)}'
    n = re.sub(r'^_Z\w*?\d+(?=[a-z_])', '', n) if n.startswith('_Z') else n
    return re.sub(r'[^A-Za-z0-9_]', '_', n)[:80]


def section(text, name):
    lines, on = [], False
    for l in text.splitlines():
        m = NVSEC.match(l)
        if m:
            if on:
                break
            on = m.group(1) == name
        if on:
            lines.append(l)
    return '\n'.join(lines) + '\n'


index, cache = [], {}
for cap in json.load(open(ls_path)):
    cdir = cap['capture']
    for k in cap['kernels']:
        if not k.get('image') or not (k.get('in_region') or os.path.basename(cdir).startswith('cap_mlp_')):
            continue
        mod = os.path.join(cdir, k['image'])
        sha = hashlib.sha256(open(mod, 'rb').read()).hexdigest()
        fn = f'{short(k["name"])}.{sha[:8]}.sass'
        p = os.path.join(out, 'executed-sass', fn)
        if not os.path.exists(p):
            if mod not in cache:
                cache.clear(); cache[mod] = open(mod + '.nvdisasm', errors='replace').read()
            open(p, 'w').write(section(cache[mod], k['name']))
        index.append(dict(kernel=k['name'], capture=os.path.basename(cdir), module=k['image'], module_sha256=sha,
                          in_region_launches=k.get('in_region'), grid=k.get('grid'), block=k.get('block'),
                          dyn_smem=k.get('dyn_smem'), join=k['join'], sass=fn))
json.dump(index, open(os.path.join(out, 'executed-sass', 'index.json'), 'w'), indent=1)

for f in sorted(os.listdir(trace)):
    if not f.endswith('.sqlite'):
        continue
    c = sqlite3.connect(os.path.join(trace, f))
    S = dict(c.execute('select id, value from StringIds'))
    base = f[:-7]
    with open(os.path.join(out, 'nsys', base + '_kernels.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['start_ns', 'end_ns', 'corr', 'name', 'grid', 'block', 'regs', 'static_smem', 'dyn_smem', 'local_per_thread'])
        for r in c.execute('select start, end, correlationId, mangledName, gridX, gridY, gridZ, blockX, blockY, blockZ, '
                           'registersPerThread, staticSharedMemory, dynamicSharedMemory, localMemoryPerThread '
                           'from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
            w.writerow([r[0], r[1], r[2], S.get(r[3], ''), f'{r[4]}x{r[5]}x{r[6]}', f'{r[7]}x{r[8]}x{r[9]}', *r[10:]])
    if base.startswith(('nsys_mlp', 'nsys_rmlib')):
        with open(os.path.join(out, 'nsys', base + '_api.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['start_ns', 'end_ns', 'corr', 'name'])
            for r in c.execute('select start, end, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start'):
                w.writerow([r[0], r[1], r[2], S.get(r[3], '')])

caps = sorted(d for d in os.listdir(trace) if d.startswith('cap_') and os.path.isdir(os.path.join(trace, d)))
with open(os.path.join(trace, 'module-sha256.txt'), 'w') as fh:
    for d in caps:
        for m in sorted(os.listdir(os.path.join(trace, d))):
            if m.endswith('.cubin'):
                fh.write(f'{hashlib.sha256(open(os.path.join(trace, d, m), "rb").read()).hexdigest()}  {d}/{m}\n')
subprocess.run(['tar', '-C', trace, '-cJf', os.path.join(out, 'capture-events.tar.xz'), 'module-sha256.txt',
                *[f'{d}/events.tsv' for d in caps], *[f'{d}.out' for d in caps]], check=True)
print(f'{len(index)} executed-kernel rows, {len(os.listdir(os.path.join(out, "executed-sass"))) - 1} SASS files')
