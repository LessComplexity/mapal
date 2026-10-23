#!/usr/bin/env python3
"""Per admitted launch: executed-image resources (cuobjdump -res-usage) + SASS mix of that exact function.

usage: exec_static.py <admission.json> <sass_dir> > executed-static.json
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sass_mix import FUN, INS, summarize  # noqa: E402

adm = json.load(open(sys.argv[1]))
sd = Path(sys.argv[2])
cache = {}


def functions(stem):
    if stem in cache:
        return cache[stem]
    funs, cur = {}, None
    for line in (sd / f'{stem}.sass').read_text(errors='replace').splitlines():
        m = FUN.match(line)
        if m:
            cur = m.group(1); funs[cur] = []; continue
        if cur:
            m = INS.search(line)
            if m:
                funs[cur].append(m.groups())
    res, cur = {}, None
    for line in (sd / f'{stem}.res').read_text(errors='replace').splitlines():
        m = re.match(r'\s*Function (\S+):', line)
        if m:
            cur = m.group(1); continue
        m = re.search(r'REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)', line)
        if m and cur:
            res[cur] = dict(regs=int(m[1]), stack=int(m[2]), static_smem=int(m[3]), local=int(m[4]))
    cache[stem] = (funs, res)
    return cache[stem]


out = []
for cap, v in adm.items():
    if not v.get('pass'):
        out.append({'capture': cap, 'admitted': False, 'error': v.get('error')}); continue
    for s in v['selected']:
        stem = (cap + '/' + s['image']).replace('/', '_')
        funs, res = functions(stem)
        ins = funs.get(s['symbol'])
        out.append({'capture': cap, 'admitted': True, 'symbol': s['symbol'], 'grid': s['grid'], 'block': s['block'],
                    'dyn_smem': s['smem'], 'join': s['join'], 'image': s['image'], **res.get(s['symbol'], {}),
                    'sass': summarize(s['symbol'], ins) if ins else None})
print(json.dumps(out, indent=1))
