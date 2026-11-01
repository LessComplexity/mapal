#!/usr/bin/env python3
"""Mapal's sm_90a kernels per cell/face: launch geometry (emitted .launch), ptxas -v resources,
theoretical occupancy and waves, hot-loop SASS mix (sass_stats.py).
usage: mapal_static.py <bundle/h100-pcie> <static dir> <sms> > mapal_static.json"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from occ import occupancy, waves  # noqa: E402
from sass_stats import analyse  # noqa: E402


def ptxas(path):
    out, cur = {}, None
    for line in open(path):
        m = re.search(r"Compiling entry function '(\S+)'", line)
        if m:
            cur = m.group(1); out[cur] = {}
        m = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', line)
        if m and cur:
            out[cur].update(stack=int(m.group(1)), spill_st=int(m.group(2)), spill_ld=int(m.group(3)))
        m = re.search(r'Used (\d+) registers', line)
        if m and cur:
            out[cur]['regs'] = int(m.group(1))
            s = re.search(r'(\d+) bytes smem', line)
            out[cur]['static_smem'] = int(s.group(1)) if s else 0
    return out


def kernel(stem, static, name, grid, block, smem, sms):
    res = ptxas(f'{static}/{stem}.ptxas').get(name, {})
    b, w, occ, lim = occupancy(block, res.get('regs', 0), smem + res.get('static_smem', 0))
    blocks = grid[0] * grid[1]
    r = dict(kernel=name, grid=grid, block=block, dyn_smem=smem, **res, blocks_per_sm=b, warps_per_sm=w,
             occupancy=round(occ, 3), limiter=lim, blocks=blocks, waves=round(waves(blocks, b, sms), 3),
             resident_warps_per_sm=round(min(blocks, b * sms) * ((block + 31) // 32) / sms, 2))
    r['sass'] = analyse(f'{static}/{stem}.sass', name)
    return r


def main(bdir, static, sms):
    out = {}
    for f in sorted(os.listdir(bdir)):
        if not f.endswith('.launch') or '.null' in f or '.base' in f or not f.startswith('mm_'):
            continue
        stem = f[:-7]
        t = open(f'{bdir}/{f}').read().split()
        e = dict(stem=stem, S=int(t[5]) if len(t) > 5 else 1)
        e['main'] = kernel(stem, static, t[0], [int(t[1]), int(t[2])], int(t[3]), int(t[4]), sms)
        if len(t) > 6:
            e['reduce'] = kernel(stem, static, t[6], [int(t[7]), int(t[8])], int(t[9]), int(t[10]), sms)
        out[stem] = e
    # MLP programs: the two tile launches from the host IR
    for f in sorted(os.listdir(bdir)):
        if not (f.startswith('pt_rm') and f.endswith('.host.ll')) or '.null' in f:
            continue
        stem = f[:-8]
        launches = re.findall(r'call void @mapal_gpu_launchv\(i64 (-?\d+), i32 \d+, i32 \d+, ptr %\w+, i32 (\d+), i32 (\d+), i32 (\d+), i32 (\d+)\)',
                              open(f'{bdir}/{f}').read())
        ks = []
        for i, gx, gy, blk, sm in launches:
            name = 'mapal_tile_%016x' % (int(i) & (2**64 - 1))
            ks.append(kernel(stem, static, name, [int(gx), int(gy)], int(blk), int(sm), sms))
        out[stem] = dict(stem=stem, launches=ks)
    return out


if __name__ == '__main__':
    print(json.dumps(main(sys.argv[1], sys.argv[2], int(sys.argv[3])), indent=1))
