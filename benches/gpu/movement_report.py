#!/usr/bin/env python3
"""S65 movement: per loop program x arch x build, what the facts require each trip (`; mapal.move`, the plan's
derivation) against what the host IR moves (evidence.py `host.move`): host copies, uploads, copy-outs, and how
the state travels.

usage: movement_report.py <raw dir>   (<raw>/static/<arch>/facts.json[.gz], the bundles' `evidence facts`)
"""
import gzip
import json
import os
import sys

ARCHS = ('rtx-4090', 'rtx-5090', 'h100-pcie', 'rtx-3090')
PROGS = ('pt_loop_2048', 'pt_loop_4096', 'pt_loopmm_1024_8', 'pt_rt_mlp2_1024', 'pt_decode_1024_256')
ARMS = (('', 'CPU+GPU'), ('.off-by-ref', 'CPU+GPU, S64 arm'), ('.cpu', 'CPU-only'), ('.cpu-off-by-ref', 'CPU-only, S64 arm'))
mb = lambda b: f'{b / 2 ** 20:.3f}' if b >= 2 ** 20 else f'{b / 2 ** 10:.1f} K' if b >= 1024 else str(b)


def main(raw):
    print('| program | arch | build | state (position:bytes:carried) | host copy / trip: facts → realized | '
          'h2d / trip: facts → realized | d2h / trip: facts → realized | MB copied per trip, S64 arm → this |')
    print('|---|---|---|---|---|---|---|---|')
    for a in ARCHS:
        p = os.path.join(raw, 'static', a, 'facts.json')
        if os.path.exists(p + '.gz'):
            facts = json.load(gzip.open(p + '.gz'))
        elif os.path.exists(p):
            facts = json.load(open(p))
        else:
            continue
        for prog in PROGS:
            for suffix, build in ARMS:
                e = facts.get(prog + suffix)
                if not e or not e.get('host', {}).get('move', {}).get('per_loop'):
                    continue
                for lp in e['host']['move']['per_loop']:
                    it = lp['intent']
                    off = facts.get(prog + ('.cpu-off-by-ref' if 'cpu' in suffix else '.off-by-ref'), {})
                    off_b = sum(x['copy_b'] for x in off.get('host', {}).get('move', {}).get('per_loop', []))
                    cmp = f'{mb(off_b)} → {mb(lp["copy_b"])}' if 'off' not in suffix else '-'
                    print(f'| {prog} | {a} | {build} | {it.get("carried")} | {mb(int(it.get("host_copy", 0)))} → '
                          f'{mb(lp["copy_b"])} ({lp["copy_n"]}) | {mb(int(it.get("h2d", 0)))} → {mb(lp["h2d_b"])} | '
                          f'{mb(int(it.get("d2h", 0)))} → {mb(lp["d2h_b"])} | {cmp} |')


if __name__ == '__main__':
    main(sys.argv[1])
