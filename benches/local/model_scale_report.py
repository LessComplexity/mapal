#!/usr/bin/env python3
"""model-scale summary from box_run.sh results files (benches/local/model_scale_run.sh): per program, Mapal's
median `iter ms` per arm (shipped `on`, GPU-only `off-place`, `null`), the best like-for-like competitor — f32 legs
for an f32 program (torch eager / compile / sdpa, gemm_lib cuBLAS / cuBLASLt), TF32 legs for a `_tf32` one — with
`within=yes` only, and the torch bf16 reference (best of `bf16`, `bf16-compile`; never a like-for-like row).
A `_pv` program and its natural twin share the torch legs (the same layer). Decode rows (`dec`, `blkdec`) are per
step: the program's 16 steps / 16. `once` lines (model_scale_once.sh): one run, gated against the `_pv` twin's
oracle. usage: model_scale_report.py <results.txt>..."""
import re
import statistics
import sys
from collections import defaultdict

PROBE = re.compile(r'^probe prog=(\S+) face=_default arm=(\S+) rep=\d+ .*?iter ms=\s*([-\d.eE+]+)')
TORCH = re.compile(r'^torch prog=(\S+) mode=(\S+) n=\d+ min_ms=\S+ med_ms=([\d.]+) .*within=(\S+) maxdev=(\S+)')
GEMM = re.compile(r'^gemm prog=(\S+) leg=(\S+) .*med_ms=([\d.]+) .*within=(\S+) maxdev=(\S+)')
GATE = re.compile(r'^(value-bound|VALUE-FAIL) prog=(\S+) face=_default arm=(\S+)')
ONCE = re.compile(r'^once prog=(\S+) rc=0 .*?iter ms=\s*([-\d.eE+]+).*gate=(\w+)')  # model_scale_once.sh
STEPS = 16
mapal = defaultdict(lambda: defaultdict(list))
legs = defaultdict(dict)  # torch stem (base, `_pv` dropped) -> leg -> (ms, within, maxdev)
gates = defaultdict(set)
for path in sys.argv[1:]:
    for line in open(path, errors='replace'):
        if m := PROBE.match(line):
            mapal[m[1]][m[2]].append(float(m[3]))
        elif m := TORCH.match(line):
            legs[m[1].replace('_pv', '')]['torch-' + m[2]] = (float(m[3]), m[4], m[5])
        elif m := GEMM.match(line):
            legs[m[1]][m[2]] = (float(m[3]), m[4], m[5])
        elif m := ONCE.match(line):
            mapal[m[1]]['on'].append(float(m[2]))
            gates[m[1]].add(m[3] + ':once')
        elif m := GATE.match(line):
            gates[m[2]].add(('pass' if m[1] == 'value-bound' else 'FAIL') + ':' + m[3])


def per(stem):
    return STEPS if re.match(r'pt_llm_(blk)?dec_', stem) else 1


def gate_cell(g):
    """`<passed>/<arms gated>`, then the failing arms."""
    if not g:
        return '-'
    bad = sorted(x.split(':', 1)[1] for x in g if not x.startswith('pass'))
    return f'{len(g) - len(bad)}/{len(g)}' + (f' FAIL {",".join(bad)}' if bad else '')


def fmt(v):
    return f'{v:.4g}' if v is not None else '-'


print('| workload | unit | Mapal ms (shipped) | Mapal ms (GPU-only arm) | null | best competitor (leg) | ratio | '
      'torch bf16 reference | gates |')
print('|---|---|---:|---:|---:|---|---:|---|---|')
for stem in sorted(mapal):
    arms = {a: statistics.median(v) / per(stem) for a, v in mapal[stem].items()}
    on, gpu, nl = arms.get('on'), arms.get('off-place'), arms.get('null')
    tf32 = stem.endswith('_tf32')
    lg = legs.get(stem.replace('_pv', ''), {})
    like = {k: v[0] / per(stem) for k, v in lg.items() if v[1] == 'yes' and not k.startswith('torch-bf16')
            and (k.endswith('-tf32') == tf32)}
    best = min(like, key=like.get) if like else None
    bf = {k: v[0] / per(stem) for k, v in legs.get(stem.replace('_pv', '').removesuffix('_tf32'), {}).items()
          if k.startswith('torch-bf16')}
    bfk = min(bf, key=bf.get) if bf else None
    mine = min(x for x in (on, gpu) if x is not None)
    print(f'| {stem[3:]} | {"ms/step" if per(stem) > 1 else "ms"} | {fmt(on)} | {fmt(gpu)} | '
          f'{f"{abs(nl / on - 1):.1%}" if nl and on else "-"} | '
          f'{f"{like[best]:.4g} ({best})" if best else "no leg"} | {f"{on / like[best]:.2f}" if best and on else "-"}'
          f'{f" / {gpu / like[best]:.2f}" if best and gpu else ""} | '
          f'{f"{bf[bfk]:.4g} ({bfk[6:]}, {lg.get(bfk, (0, 0, "-"))[2]} dev)" if bfk else "-"} | '
          f'{gate_cell(gates.get(stem, set()))} |')
print('\nratio: Mapal shipped / best competitor (/ GPU-only arm); `null`: |null arm / shipped − 1|.')
print('\n| program | leg | ms | within | maxdev |')
print('|---|---|---:|---|---:|')
for stem in sorted(legs):
    for k, (ms, w, d) in sorted(legs[stem].items(), key=lambda kv: kv[1][0]):
        print(f'| {stem[3:]} | {k} | {ms / per(stem):.4g}{" /step" if per(stem) > 1 else ""} | {w} | {d} |')
