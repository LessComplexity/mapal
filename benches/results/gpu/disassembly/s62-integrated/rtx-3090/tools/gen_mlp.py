#!/usr/bin/env python3
"""MLP whole-program tables: Mapal arms (lazy/eager), nsys kernels, code size; library legs."""
import json, sqlite3, sys
from pathlib import Path
T = json.load(open(sys.argv[1])); MS = json.load(open(sys.argv[2])); X = json.load(open(sys.argv[3])); NS = Path(sys.argv[4])
code = {}
for r in MS:
    if r['variant'].startswith('pt_rm') and r['kernel'].startswith('mapal_tile'):
        code.setdefault(r['variant'], {})[r['kernel']] = (r['sass']['code_bytes'], r['regs'], r.get('spill_st', 0))
def nsk(prog):
    return [x for x in T['nsys'].get('prog_' + prog, []) if x['name'].startswith('mapal_tile')]
def api(prog, name):
    f = NS / f'prog_{prog}.sqlite'
    if not f.exists(): return None
    c = sqlite3.connect(f)
    v = c.execute("select r.end-r.start from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value=? order by r.start", (name,)).fetchall()
    return [x[0] / 1e3 for x in v]
P = print
P('## Residual MLP whole programs (Mapal `iter ms`, 12 processes per arm and mode, ABBA)\n')
P('`lazy` = driver default (CUDA_MODULE_LOADING unset: each function is loaded at its first launch, inside the timed region); '
  '`eager` = CUDA_MODULE_LOADING=EAGER (every function loaded at cuModuleLoadData, before the clock). Kernel µs from one nsys-traced process.\n')
P('| program | face | arm | iter ms lazy | iter ms eager | lazy − eager µs | kernel 1 µs (grid x block, regs) | kernel 2 µs (grid x block, regs) | kernel sum µs | eager − kernels µs | SASS code bytes (both kernels) | cuModuleLoadData µs (before clock) |')
P('|---|---|---|---|---|---|---|---|---|---|---|---|')
mp = T['mlp_programs']
for prog in ['pt_rm_m128_d256_h512', 'pt_rm_m128_d1024_h2048']:
    for face in ['', '_c']:
        for arm in ['', '.base', '.off-grid-share', '.off-panel-regs']:
            a = prog + face + arm
            if a not in mp: continue
            lz, eg = mp[a]['lazy']['median_ms'], mp[a]['eager']['median_ms']
            ks = nsk(a)
            kd = [f"{k['median_us']} ({'x'.join(k['grid'][:2])} x {k['block'][0]}, {k['regs']})" for k in ks]
            ksum = sum(k['median_us'] for k in ks) if ks else None
            cb = code.get(a, {})
            cbs = '+'.join(str(cb[k][0]) for k in sorted(cb)) if cb else ''
            ml = api(a, 'cuModuleLoadData')
            P(f"| {prog[6:]} | {'C' if face else 'E'} | {arm[1:] or 'on (integrated)'} | {lz:.4f} | {eg:.4f} | {1e3*(lz-eg):.1f} | "
              f"{kd[0] if kd else '—'} | {kd[1] if len(kd)>1 else '—'} | {('%.1f' % ksum) if ksum else '—'} | {('%.1f' % (1e3*eg-ksum)) if ksum else '—'} | {cbs} | {('%.0f' % ml[0]) if ml else '—'} |")
P('')
P('## Residual MLP library legs (rm_bench, two processes x 30 samples, L2 flushed before each sample)\n')
P('`first` = host wall of the first call in the process (library first use and lazy kernel load included); '
  '`event` = CUDA events around one call; `wall` = host wall, launch to synchronize (the form of Mapal\'s `iter ms`).\n')
P('| d | leg | launches | first ms | event ms (p1 / p2) | wall ms (p1 / p2) | kernels executed (observer) |')
P('|---|---|---|---|---|---|---|')
import re
def short(s):
    m = re.search(r'(cutlass_80_simt_sgemm_\w+?_nn|ampere_sgemm_\w+?_nn|splitKreduce_kernel|globalKernel|bias_relu)', s)
    if m: return m.group(1)
    m = re.search(r'GemmShapeILi(\d+)ELi(\d+)ELi(\d+)', s)
    return 'Gemm<%s,%s,%s>' % m.groups() if m else s[:30]
lib = T['mlp_library']
for d in (256, 1024):
    for leg in ('cublas', 'lt', 'cutlass', 'cutlass_tuned'):
        v = lib[f'd{d}|{leg}']; a, b = v['1'], v['2']
        ks = [x for x in X if x['capture'] == f'rm_{d}/{leg}' and x.get('admitted')]
        kd = ', '.join(f"`{short(k['symbol'])}` {'x'.join(map(str,k['grid']))}" for k in ks)
        P(f"| {d} | {leg} | {a['launches']} | {a['first_ms']:.3f} / {b['first_ms']:.3f} | {a['event_med_ms']:.4f} / {b['event_med_ms']:.4f} | {a['wall_med_ms']:.4f} / {b['wall_med_ms']:.4f} | {kd} |")
P('')
