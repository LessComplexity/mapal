#!/usr/bin/env python3
"""Markdown tables for REPORT.md from tables.json + executed-static.json + mapal-static.json.

usage: gen_report.py <tables.json> <executed-static.json> <mapal-static.json> > tables.md
"""
import json, re, statistics, sys

T = json.load(open(sys.argv[1]))
X = json.load(open(sys.argv[2]))
MS = json.load(open(sys.argv[3]))
SMS = 82
CELLS = ['16x2048x7168', '16x1024x4096', '16x18432x7168', '128x2048x7168', '4096x4096x4096']
IMPLS = [('Mapal E', '{c}', 'mm_{c}'), ('Mapal C', '{c}', 'mm_{c}_c'), ('Mapal reassoc+C', '{c}', 'mm_{c}_rc'),
         ('cuBLAS', '{c}', 'cublas'), ('CUTLASS stock', '{c}', 'cutlass'), ('CUTLASS tuned', '{c}', 'cutlass_tuned')]
ROWKEY = {'Mapal E': 'Mapal E', 'Mapal C': 'Mapal C', 'Mapal reassoc+C': 'Mapal reassoc+C', 'cuBLAS': 'cublas',
          'CUTLASS stock': 'cutlass', 'CUTLASS tuned': 'cutlass_tuned'}


def occ(regs, threads, smem):
    warps = -(-threads // 32)
    rw = -(-(max(regs, 1) * 32) // 256) * 256
    lim = {'regs': (65536 // rw) // warps, 'threads': 1536 // (warps * 32), 'smem': 102400 // (smem + 1024), 'blocks': 16}
    b = min(lim.values())
    return b, b * warps, [k for k, v in lim.items() if v == b][0]


def short(s):
    m = re.search(r'(cutlass_80_simt_sgemm_\w+?_nn|ampere_sgemm_\w+?_nn|splitKreduce_kernel|globalKernel|bias_relu|mapal_tile_[0-9a-f]+)', s)
    if m:
        return m.group(1)
    m = re.search(r'GemmShapeILi(\d+)ELi(\d+)ELi(\d+)', s)
    return 'cutlass::Gemm<%s,%s,%s>' % m.groups() if m else s[:30]


def kernels(cap):
    return [x for x in X if x['capture'] == cap and x.get('admitted')]


def nsys_us(cell, k):
    mapal = k['symbol'].startswith('mapal_')
    for r in T['nsys'].get(cell, []):
        if mapal and r['name'] != k['symbol']:
            continue
        if not mapal and r['name'].startswith('mapal_'):
            continue
        if [int(g) for g in r['grid']] == k['grid'] and [int(b) for b in r['block']] == k['block']:
            return r['median_us']
    return None


out = []
P = out.append

# ---- 1. timing + verdicts
P('## Same-box kernel timing (ms, CUDA events, L2 flushed before every sample)\n')
P('Two processes, 30 samples each per leg, legs round-robin with ABBA order; process 2 reverses the arm order. '
  '`null` = a second copy of the Contract PTX loaded as its own module in the same process.\n')
P('| cell | Mapal E | Mapal C | Mapal reassoc+C | cuBLAS | CUTLASS stock | CUTLASS tuned | null (C copy vs C) |')
P('|---|---|---|---|---|---|---|---|')
for c in CELLS:
    rows = T['cells'][c]['rows']
    f = lambda k: '%.4f / %.4f' % tuple(rows[k]['pass_medians_ms'])
    nv = T['cells'][c]['null_variation_per_pass']
    P(f"| {c} | {f('Mapal E')} | {f('Mapal C')} | {f('Mapal reassoc+C')} | {f('cublas')} | {f('cutlass')} | {f('cutlass_tuned')} | "
      f"{100 * nv[0]:.2f}% / {100 * nv[1]:.2f}% |")
P('')

P('## Per-cell verdicts (time ratio Mapal / other, per process; >1 = Mapal slower)\n')
P('Verdict rule: both processes outside the cell\'s null variation on the same side → better/worse; otherwise on-par.\n')
P('| cell | Mapal face | vs cuBLAS | vs CUTLASS stock | vs CUTLASS tuned |')
P('|---|---|---|---|---|')
VERD = {}
for c in CELLS:
    rows = T['cells'][c]['rows']
    null = max(T['cells'][c]['null_variation_per_pass'])
    for face in ('Mapal E', 'Mapal C', 'Mapal reassoc+C'):
        cells = []
        for other in ('cublas', 'cutlass', 'cutlass_tuned'):
            r = [a / b for a, b in zip(rows[face]['pass_medians_ms'], rows[other]['pass_medians_ms'])]
            if all(x > 1 + null for x in r):
                v = 'worse'
            elif all(x < 1 - null for x in r):
                v = 'better'
            else:
                v = 'on-par'
            VERD[(c, face, other)] = (r, v)
            cells.append(f'{v} {r[0]:.2f} / {r[1]:.2f}')
        P(f'| {c} | {face[6:]} | ' + ' | '.join(cells) + ' |')
P('')

# ---- 2. kernels and resources
P('## What each implementation executes (observer-admitted images; regs/smem/local from the executed image)\n')
P('Occupancy: sm_86 limits (64K regs, 1536 threads, 16 blocks, 100 KiB smem + 1 KiB/block reserved per SM). '
  'Waves = blocks / (82 SMs x resident blocks per SM). Time = nsys kernel median (3-rep traced process).\n')
P('| cell | impl | kernel | grid | block | regs | stack (spill) B | smem static+dyn B | blocks/SM (limit) | warps/SM | waves | nsys µs |')
P('|---|---|---|---|---|---|---|---|---|---|---|---|')
for c in CELLS:
    for name, _, leg in IMPLS:
        cap = f'{c}/' + leg.format(c=c)
        for k in kernels(cap):
            th = k['block'][0] * k['block'][1] * k['block'][2]
            blocks = k['grid'][0] * k['grid'][1] * k['grid'][2]
            sm = k.get('static_smem', 0) + k['dyn_smem']
            b, w, lim = occ(k.get('regs', 0), th, sm)
            waves = blocks / (SMS * b) if b else float('inf')
            P(f"| {c} | {name} | `{short(k['symbol'])}` | {'x'.join(map(str, k['grid']))} | {'x'.join(map(str, k['block']))} | "
              f"{k.get('regs')} | {k.get('stack')} | {k.get('static_smem')}+{k['dyn_smem']} | {b} ({lim}) | {w} | {waves:.2f} | "
              f"{nsys_us('cell_' + c, k)} |")
P('')

# ---- 3. SASS mix
P('## SASS instruction mix of the hot loop (static, executed image)\n')
P('MAC = FFMA (Contract) or FMUL+FADD pair (Exact). chains = independent accumulation chains in the loop body '
  '(accumulators per thread). LDG/LDS widths in bytes per thread per loop iteration.\n')
P('| cell | impl | kernel | code B | loop instr | MAC | FADD | LDG (128/64/32) | LDGSTS | LDS (128) | STS | BAR | LDL/STL | MAC/global ld | MAC/LDS | chains | MAC share |')
P('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
seen = set()
for c in CELLS:
    for name, _, leg in IMPLS:
        cap = f'{c}/' + leg.format(c=c)
        for k in kernels(cap):
            s = k.get('sass') or {}
            h = s.get('hot_loop') or {}
            if not h or not h.get('mac_steps'):
                continue
            key = (name, short(k['symbol']))
            if name.startswith('CUTLASS') and key in seen:
                continue
            seen.add(key)
            ldg = f"{h['LDG']} ({h['LDG.128']}/{h['LDG.64']}/{h['LDG.32']})"
            P(f"| {c if not name.startswith('CUTLASS') else 'all'} | {name} | `{short(k['symbol'])}` | {s['code_bytes']} | {h['instructions']} | {h['mac_steps']} | {h['FADD']} | {ldg} | "
              f"{h['LDGSTS']} | {h['LDS']} ({h['LDS.128']}) | {h['STS']} | {h['BAR']} | {h['LDL']}/{h['STL']} | {h['ffma_per_global_load']} | {h['ffma_per_shared_load']} | "
              f"{h['accumulator_regs']} | {h['mac_share']} |")
P('')

# ---- 4. hardware usage
P('## Hardware usage (from same-box time, known bytes and FLOPs)\n')
P(f"Peaks at the clocks sampled during the run: SM {T['clock_sm_mhz_median_under_load']:.0f} MHz → FP32 "
  f"{T['peak_fp32_tflops_at_clock']} TFLOP/s (82 SM x 128 lanes x 2); memory {T['clock_mem_mhz_median_under_load']:.0f} MHz → "
  f"{T['peak_dram_gbs']} GB/s (384-bit). Bytes = A + B + C once (+ the split-K workspace write and read where a separate "
  "reduce kernel runs); L2 is flushed before every sample, so A and B come from DRAM. FP32 pipe % counts 2 issue slots per "
  "MAC on Exact (FMUL + FADD).\n")
P('| cell | impl | ms | AI FLOP/B | GB/s | % DRAM peak | TFLOP/s | % FP32 peak | FP32 pipe % | bound |')
P('|---|---|---|---|---|---|---|---|---|---|')
for c in CELLS:
    M, N, K = map(int, c.split('x'))
    rows = T['cells'][c]['rows']
    for name, _, leg in IMPLS:
        r = rows[ROWKEY[name]]
        ms = r['median_ms_all']
        extra = 0
        cap = f'{c}/' + leg.format(c=c)
        ks = kernels(cap)
        if any('splitKreduce' in k['symbol'] for k in ks):
            z = max(k['grid'][2] for k in ks)
            extra = 2 * z * M * N * 4
        by = r['bytes_min'] + extra if not name.startswith('Mapal') else r['bytes_min']
        gbs = by / (ms / 1e3) / 1e9
        pbw = 100 * gbs / T['peak_dram_gbs']
        tf = 2 * M * N * K / (ms / 1e3) / 1e12
        pfp = 100 * tf / T['peak_fp32_tflops_at_clock']
        pipe = pfp * (2 if name == 'Mapal E' else 1)
        if pbw >= 60:
            b = 'memory'
        elif pipe >= 50:
            b = 'compute'
        elif ms < 0.01:
            b = 'launch'
        else:
            b = 'latency/occupancy'
        P(f'| {c} | {name} | {ms:.4f} | {2 * M * N * K / by:.1f} | {gbs:.0f} | {pbw:.1f} | {tf:.2f} | {pfp:.1f} | {pipe:.1f} | {b} |')
P('')
print('\n'.join(out))
