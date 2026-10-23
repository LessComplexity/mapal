#!/usr/bin/env python3
"""Join ptxas -v resources, launch geometry and SASS mix of every Mapal sm_86 kernel.

usage: static_table.py <static_dir> <bundle_part_dir> > mapal-static.json
"""
import json, re, subprocess, sys
from pathlib import Path

st, part = Path(sys.argv[1]), Path(sys.argv[2])
here = Path(__file__).parent
out = []
for px in sorted(st.glob('*.ptxas')):
    v = px.stem
    res = {}
    cur = None
    for line in px.read_text().splitlines():
        m = re.search(r"Compiling entry function '(\S+)'", line)
        if m:
            cur = m.group(1); res[cur] = {}
        m = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', line)
        if m and cur:
            res[cur].update(stack=int(m[1]), spill_st=int(m[2]), spill_ld=int(m[3]))
        m = re.search(r'Used (\d+) registers, used (\d+) barriers(?:, (\d+) bytes smem)?', line)
        if m and cur:
            res[cur].update(regs=int(m[1]), barriers=int(m[2]), static_smem=int(m[3] or 0))
    launch = (part / f'{v}.launch')
    lw = launch.read_text().split() if launch.exists() else []
    launches = {}
    if lw:
        launches[lw[0]] = dict(grid=[int(lw[1]), int(lw[2])], block=int(lw[3]), dyn_smem=int(lw[4]))
        if len(lw) >= 11:
            launches[lw[6]] = dict(grid=[int(lw[7]), int(lw[8])], block=int(lw[9]), dyn_smem=int(lw[10]), splitk=int(lw[5]))
            launches[lw[0]]['splitk'] = int(lw[5])
    mix = {}
    r = subprocess.run([sys.executable, str(here / 'sass_mix.py'), str(st / f'{v}.sass')], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        d = json.loads(line); mix[d['function']] = d
    for k, rr in res.items():
        out.append({'variant': v, 'kernel': k, **rr, 'launch': launches.get(k), 'sass': mix.get(k)})
print(json.dumps(out, indent=1))
