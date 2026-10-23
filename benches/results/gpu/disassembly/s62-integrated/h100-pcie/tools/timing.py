#!/usr/bin/env python3
"""Parse a disasm box results file into per-cell / per-program timing JSON.

usage: timing.py <results-h100-pcie.txt>  -> JSON {facts, cells, mlp, rmlib}
cells[cell][leg] = {pass: {med, min, max, n}, samples: [...], pooled_med}; legs are Mapal arm
stems (face suffix, `.null`), cublas, cutlass, cutlass_tuned. Also `arm` (JIT regs/local bytes,
grid) and `libarm` (CUTLASS resources) lines and the value-gate lines per leg.
"""
import json
import re
import statistics
import sys
from collections import defaultdict


def kv(s):
    return dict(re.findall(r'(\w+)=("[^"]*"|\[[^\]]*\]|\S+)', s))


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None  # upper median, as the harnesses print


def main(path):
    facts, cells, mlp, rmlib, gates = {}, defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), []
    arms, libarms, values = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for line in open(path, errors='replace'):
        line = line.rstrip('\n')
        m = re.match(r'pass=(\d) cell=(\S+) (.*)', line)
        if m:
            p, cell, rest = int(m.group(1)), m.group(2), m.group(3)
            if rest.startswith('facts'):
                facts.update(kv(rest))
            elif rest.startswith('device'):
                facts['device'] = rest
            elif rest.startswith('arm '):
                d = kv(rest); arms[cell][d['stem']] = d
            elif rest.startswith('libarm '):
                d = kv(rest); libarms[cell][d['leg']] = d
            elif rest.startswith('values '):
                d = kv(rest); values[cell].setdefault(d['leg'], []).append(d)
            elif rest.startswith('time '):
                d = kv(rest)
                cells[cell][d['leg']].setdefault('pass', {})[p] = dict(med=float(d['med_ms']), min=float(d['min_ms']),
                                                                        max=float(d['max_ms']), n=int(d['n']))
            elif rest.startswith('samples '):
                leg = rest.split()[1].split('=', 1)[1]
                xs = rest.split(None, 2)[2] if len(rest.split(None, 2)) > 2 else ''
                cells[cell][leg].setdefault('samples', []).extend(float(x) for x in xs.split(',') if x)
            continue
        m = re.match(r'mlp prog=(\S+) arm=(\S+) rep=(\d+) wall_us=(\d+) out=\[(.*)\]', line)
        if m:
            it = re.search(r'iter ms=\s*([0-9.eE+-]+)', m.group(5))
            mlp[m.group(1)][m.group(2)].append(dict(rep=int(m.group(3)), wall_us=int(m.group(4)),
                                                    iter_ms=float(it.group(1)) if it else None, out=m.group(5)))
            continue
        m = re.match(r'mlp_trace prog=(\S+) arm=(\S+) rep=(\d+) (.*)', line)
        if m:
            ks = re.findall(r'([0-9a-f]{16}) ms=([0-9.]+)', m.group(4))
            mlp[m.group(1)][m.group(2) + ':trace'].append(dict(rep=int(m.group(3)), kernels=[(k, float(v)) for k, v in ks],
                                                              total=kv(m.group(4).split('mapal_gpu_trace_total', 1)[-1])))
            continue
        m = re.match(r'pass=(\d) prog=(\S+) (rm|rm_values|rm_samples|libarm) (.*)', line)
        if m:
            d = kv(m.group(4)); d['pass'] = int(m.group(1))
            rmlib[m.group(2)][m.group(3)].append(d)
            continue
        if line.startswith(('PASS ', 'FAIL ', 'CELL GATE', 'mlp value gate', 'capture ', 'nsys ', 'built ', 'BUILD')):
            gates.append(line)
    out = dict(facts=facts, gates=gates, arms=arms, libarms=libarms, values=values, cells={}, mlp={}, rmlib=rmlib)
    for cell, legs in cells.items():
        out['cells'][cell] = {}
        for leg, d in legs.items():
            d['pooled_med'] = med(d.get('samples', []))
            out['cells'][cell][leg] = d
    for prog, arms_ in mlp.items():
        out['mlp'][prog] = {}
        for arm, rows in arms_.items():
            if arm.endswith(':trace'):
                ks = defaultdict(list)
                for r in rows:
                    for i, (_, ms) in enumerate(r['kernels']):
                        ks[i].append(ms)
                out['mlp'][prog][arm] = dict(kernel_ms=[med(v) for _, v in sorted(ks.items())], rows=rows)
            else:
                it = [r['iter_ms'] for r in rows if r['iter_ms'] is not None]
                out['mlp'][prog][arm] = dict(n=len(it), iter_med=med(it), iter_min=min(it) if it else None,
                                             iter_mean=statistics.mean(it) if it else None,
                                             wall_med_us=med([r['wall_us'] for r in rows]), iters=it,
                                             outs=sorted({re.sub(r'iter ms=.*', '', r['out']).strip() for r in rows}))
    return out


if __name__ == '__main__':
    print(json.dumps(main(sys.argv[1]), indent=1, default=str))
