#!/usr/bin/env python3
"""Executed-image SASS: every kernel the observer saw execute (capture_join.py), its exact saved
module disassembled (cuobjdump -sass, -res-usage) and its hot loop measured (sass_stats.py).

usage: libsass.py <capture dir>... > libsass.json   (cuobjdump on PATH; SASS saved beside each image)
"""
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from capture_join import main as join  # noqa: E402
from sass_stats import analyse  # noqa: E402


def dump(cap, image, what):
    p = os.path.join(cap, image)
    out = f'{p}.{what}'
    if not os.path.exists(out):
        # SASS from nvdisasm (cuobjdump refuses some driver-loaded library images as "no device code")
        cmd = ['nvdisasm', '-c', p] if what == 'nvdisasm' else ['cuobjdump', f'-{what}', p]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        open(out, 'w').write(r.stdout)
    return out


def res_usage(path, name):
    txt = open(path, errors='replace').read()
    m = re.search(r'Function ' + re.escape(name) + r':\s*\n\s*(REG:.*)', txt)
    return dict(re.findall(r'(\w+(?:\[\d+\])?):(\d+)', m.group(1))) if m else {}


def main(caps):
    out = []
    for cap in caps:
        j = join(cap)
        rows = []
        seen = set()
        region = {}
        for k in j['kernels']:
            key = (k['name'], k['image'])
            if k['in_region']:
                region.setdefault(key, []).append(k['grid'])
        for k in j['kernels']:
            key = (k['name'], k['image'])
            if key in seen or not k['image']:
                if not k['image']:
                    rows.append(dict(name=k['name'], in_region=k['in_region'], join=k['join']))
                continue
            seen.add(key)
            k = dict(k, in_region=len(region.get(key, [])), region_grids=region.get(key, []))
            sass = dump(cap, k['image'], 'nvdisasm')
            res = res_usage(dump(cap, k['image'], 'res-usage'), k['name'])
            try:
                st = analyse(sass, k['name'])
            except SystemExit as e:
                st = dict(error=str(e))
            rows.append(dict(name=k['name'], grid=k['grid'], block=k['block'], dyn_smem=k['dyn_smem'],
                             in_region=k['in_region'], region_grids=k['region_grids'], image=k['image'], join=k['join'],
                             res=res, sass=st, sass_file=os.path.basename(sass)))
        out.append(dict(capture=cap, images=j['images'], image_bytes=j['image_bytes'], errors=j['errors'],
                        region=j['region'], kernels=rows))
    return out


if __name__ == '__main__':
    print(json.dumps(main(sys.argv[1:]), indent=1))
