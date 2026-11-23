#!/usr/bin/env python3
"""Mapal vs the fastest competitor AT THE SAME PRECISION, per GPU part and algorithm family: one matrix from the
results files box_run.sh / gpu_run.sh / model_scale_run.sh write (gap.py's and model_scale_report.py's parsing).

usage: matrix.py [--out DIR] [--skip REGEX] <part>=<file>[@<commit>][,<file>[@<commit>]...] ...
  One column per part, in argument order. List a part's files oldest first: a later file replaces an earlier one
  per program (Mapal's `on` / `null` samples come from the newest file that timed the program `on`), per competitor
  leg (the newest file that timed that leg for the program; legs do not depend on Mapal's commit) and per GEMM cell.
  `@<commit>` labels a file (default: an 8-hex path component, else `?`). Writes DIR/matrix.md and DIR/matrix.json
  (DIR defaults to benches/results/evidence). The command that built the committed matrix is at the end of matrix.md.
  `--skip REGEX`: workloads (without `pt_`) matching it are not rows (e.g. the KV-cache twins `llm_kv[xn]_`, reported
  with their own HF-cache / naive torch legs in caching-memory.md and the cross-arch README).

Rules
- Mapal = median `iter ms` of the shipped build (probe arm `on`, face `_default`); null = |median(null) / median(on)
  - 1| from the same file. GEMM cells (tile_bench splitk section): the default-face build, medians over passes.
- Fairness: a row's best competitor is the fastest leg at the row's precision.
    f32     torch eager / compile / compile-cg / compile-ro / sdpa / sdpa-efficient / sdpa-compile(-cg); gemm_lib
            cuBLAS / cuBLASLt (`-fresh`: operands regenerated each repetition); chain_bench legs `chain-*` (cuBLAS,
            cuBLASLt, CUTLASS, hand CUDA, all CUBLAS_COMPUTE_32F); the cells' cuBLAS / CUTLASS / CUTLASS tuned;
            comp_bench `triton` (ieee dots), `liger`.
    _tf32   the TF32 legs: torch *-tf32, gemm_lib cublas-tf32 / lt-tf32 (+ `-fresh`), comp `triton-tf32`, `liger-tf32`.
    _bf16w  the bf16 legs: torch bf16 / bf16-compile(-cg / -ro) / bf16-cudnn / bf16-efficient (the layer in bfloat16),
            comp `*-bf16` (Liger, FlashAttention-2/3, FlashInfer, Triton).
    _bf16kv the bf16 legs (weights and KV cache stored bf16, f32 compute: tensor-cores' bf16 KV cache).
    _bf16   the bf16 legs (bf16 storage, products `: bf16 | f32`: torch bf16's face; `pj_*_bf16`, the layer twins).
    _f8w    fp8 legs (f8e4m3 weights with per-channel scales, f32 compute: fp8-leftovers): none is installed (no
            torchao), so the row is `(no leg)`; its `_bf16w` twin's best bf16 leg is its bf16 reference, never its best.
  A bf16-storage row without bf16 legs of its own takes its twin's (`legs_from`: the `_bf16w` twin, else the f32
  program): a bf16 leg rounds weights, KV cache and activations to bfloat16, so the twins run one computation
  (torch_bench.py `llm_workload` / `workload`); box_run.sh's TORCH_SKIP does not time it twice.
  Every leg counts only with `within=yes`; a CUDA-graph leg whose graph Inductor skipped (`cg=skipped`) does not
  count (it ran as a plain compile). `bf16_ref` = the row's (else its `_tf32`-less base stem's, else its `_bf16w`
  twin's) best bf16 leg: the real-world reference, reported apart, never the best of an f32 / tf32 row.
- Competitor families (`CFAMS`): every leg of a row is listed per part in the family tables ("Every competitor leg"),
  fastest bold; `not run` = the family has a leg for that row on another part, `unavailable` = the part's run
  reported the family's package missing. `base_*` = the best over the legs that existed before the competitors step
  (torch eager / compile / sdpa / bf16, gemm_lib, chain_bench, cells): the "what the new legs change" section.
- A `_pv` program's torch legs serve its natural twin (the same layer); `_pv` rows are not matrix rows. `px_` rows
  are not timed programs. LLM decode rows (`llm_dec`, `llm_blkdec`) are per step (the program's 16 steps / 16).
- Verdict: win / even / loss; even = |ratio - 1| <= max(null, 5 %). Geomeans over a family's rows per part.
- `old_*`: gap.py's mixed rule on the same data (every torch leg within, bf16 included, TF32 modes kept off
  non-tf32 rows; chain legs; no gemm_lib legs) — the flips section lists rows whose verdict differs.
"""
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

med = statistics.median
PROBE = re.compile(r'^probe prog=(\S+) face=_default arm=(\S+) rep=\d+ .*?iter ms=\s*([-\d.eE+]+)')
TORCH = re.compile(r'^torch prog=(\S+) mode=(\S+) n=\d+ min_ms=\S+ med_ms=([\d.]+) .*within=(\w+)')
COMP = re.compile(r'^comp prog=(\S+) leg=(\S+) family=\S+ prec=\S+ n=\d+ min_ms=\S+ med_ms=([\d.]+) .*within=(\w+)')
COMP_MISSING = re.compile(r'^comp family=(\S+) unavailable')
GEMM = re.compile(r'^gemm prog=(\S+) leg=(\S+) .*med_ms=([\d.]+) .*within=(\w+)')
CHAIN = re.compile(r'^chain probe=(\S+) ((?:\w+=\S+ )*?)leg=(\S+) .*?(?:wall_med_ms|med_ms)=([\d.]+)')
CELL = re.compile(r'^pass=\d+ cell=(\S+) time leg=(\S+) .*med_ms=([\d.]+)')
CELL_LIBS = {'cublas': 'cublas', 'cutlass': 'cutlass', 'cutlass_tuned': 'cutlass-tuned'}
STEPS = 16
FAMILIES = [  # (key, title, workload pattern)
    ('gemm', 'GEMM cells & projections', r'(mm|pj|epi)_'),
    ('mlp', 'MLP chains', r'(mlp|mlp2|smlp)_'),
    ('attn', 'Attention / softmax / long rows', r'(attn|causal|decode|gen|prefill|lsm|softmax_rows|rowsum)_'),
    ('llm', 'LLM decode / prefill / block (f32, tf32, bf16 storage)', r'llm_'),
    ('conv', 'Conv / FIR / elementwise', r'(conv|convmc|ew|ewconvew|fir|firew)_'),
    ('fan', 'Fan-in/out, loops, CPU+GPU', r'(diamond|fanout|indep|twin|loop|loopmm|rt)_'),
    ('other', 'Other', r''),
]
PRECS = ('f32', 'tf32', 'bf16w', 'bf16kv', 'bf16', 'f8w')
EVEN = 0.05
CFAMS = [  # (key, title): the competitor families of the every-leg tables, in column order
    ('vendor', 'vendor libs'), ('eager', 'torch eager'), ('compile', 'torch.compile'),
    ('cg', 'compile + CUDA graphs'), ('ro', 'reduce-overhead'), ('sdpa', 'SDPA'), ('fa', 'FlashAttention-2/3'),
    ('flashinfer', 'FlashInfer'), ('triton', 'hand Triton'), ('liger', 'Liger'), ('trtllm', 'TensorRT-LLM'),
    ('cuda', 'hand CUDA'),
]
COMP_FAM = {'triton': 'triton', 'liger': 'liger', 'fa2': 'fa', 'fa3': 'fa', 'flashinfer': 'flashinfer',
            'trtllm': 'trtllm'}
BASE_TORCH = {'eager', 'compile', 'eager-tf32', 'compile-tf32', 'sdpa', 'sdpa-tf32', 'sdpa-compile', 'bf16',
              'bf16-compile', 'naive', 'naive-compile'}


def cfam(leg):
    """A leg's competitor family."""
    head = leg.split('-')[0]
    if head in COMP_FAM:
        return COMP_FAM[head]
    if leg in ('chain-fused', 'chain-cache'):
        return 'cuda'
    if not leg.startswith('torch-'):
        return 'vendor'  # gemm_lib cuBLAS / cuBLASLt, chain_bench library legs, the cells' cuBLAS / CUTLASS
    m = leg[6:]
    if '-cg' in m:
        return 'cg'
    if '-ro' in m:
        return 'ro'
    if m.startswith('sdpa') or m in ('bf16-cudnn', 'bf16-efficient'):
        return 'sdpa'
    return 'compile' if 'compile' in m else 'eager'


def is_new(leg):
    """A leg the competitors step added (not in the 9e27d09c matrix's set)."""
    if leg.startswith('torch-'):
        return leg[6:] not in BASE_TORCH
    return leg.split('-')[0] in COMP_FAM


def family(w):
    return next(k for k, _, p in FAMILIES if re.match(p, w))


def row_prec(w):
    return next((p for suf, p in (('_bf16w', 'bf16w'), ('_bf16kv', 'bf16kv'), ('_bf16', 'bf16'), ('_tf32', 'tf32'),
                                  ('_f8w', 'f8w'))
                 if w.endswith(suf)), 'f32')


def twins(prog):
    """A bf16-storage program's twins whose bf16 legs are its own computation: the `_bf16w` twin, the f32 program."""
    base = re.sub(r'_(bf16w|bf16kv|bf16|tf32|f8w)$', '', prog)
    return [t for t in (base + '_bf16w', base) if t != prog]


def leg_prec(leg):
    return 'bf16' if leg.startswith('torch-bf16') or leg.endswith('-bf16') else 'tf32' if 'tf32' in leg else 'f32'


FAIR = {'f32': 'f32', 'tf32': 'tf32', 'bf16w': 'bf16', 'bf16kv': 'bf16', 'bf16': 'bf16', 'f8w': 'f8'}


def per(w):
    return STEPS if re.match(r'llm_(blk)?dec_', w) else 1


def unpv(prog):
    return re.sub(r'_pv(?=_|$)', '', prog)


def label(path):
    m = re.search(r'raw-evidence/[^/]+/(.+)/[^/]+$', path)
    return m[1] if m else Path(path).parent.name or path


def parse(path):
    """-> mapal {prog: {arm: [ms]}}, legs {stem: {leg: (ms, within, note)}}, cells {cell: {leg: [ms]}}, the
    competitor families the run reported unavailable. note: '' | 'gate' (within=no) | 'cg-skipped'."""
    mapal = defaultdict(lambda: defaultdict(list))
    legs = defaultdict(dict)
    cells = defaultdict(lambda: defaultdict(list))
    missing = set()
    for line in open(path, errors='replace'):
        if m := PROBE.match(line):
            mapal[m[1]][m[2]].append(float(m[3]))
        elif m := TORCH.match(line):
            skip = ' cg=skipped' in line
            ok = m[4] == 'yes' and not skip
            legs[unpv(m[1])]['torch-' + m[2]] = (float(m[3]), ok, 'cg-skipped' if skip else '' if ok else 'gate')
        elif m := COMP.match(line):
            legs[unpv(m[1])][m[2]] = (float(m[3]), m[4] == 'yes', '' if m[4] == 'yes' else 'gate')
        elif m := COMP_MISSING.match(line):
            missing.add(COMP_FAM.get(m[1], m[1]))
        elif m := GEMM.match(line):
            legs[unpv(m[1])][m[2]] = (float(m[3]), m[4] == 'yes', '' if m[4] == 'yes' else 'gate')
        elif m := CHAIN.match(line):
            nums = '_'.join(x.split('=')[1] for x in m[2].split())
            legs[f'pt_{m[1]}_{nums}']['chain-' + m[3]] = (float(m[4]), True, '')
        elif m := CELL.match(line):
            cells[m[1]][m[2]].append(float(m[3]))
    return mapal, legs, cells, missing


def load_part(specs):
    """Merge a part's files (oldest first) -> programs, legs, cells, sources."""
    progs, legs, cells, sources, missing = {}, defaultdict(dict), {}, [], set()
    for spec in specs:
        path, _, commit = spec.partition('@')
        if not commit:
            commit = next((c for c in Path(path).parts if re.fullmatch(r'[0-9a-f]{8}', c)), '?')
        src = {'file': path, 'step': label(path), 'commit': commit}
        sources.append(src)
        mapal, lg, cl, miss = parse(path)
        missing |= miss
        for prog, arms in mapal.items():
            if arms.get('on'):
                progs[prog] = (arms, src)
        for stem, d in lg.items():
            for leg, (ms, ok, note) in d.items():
                legs[stem][leg] = (ms, ok, src, note)
        for cell, d in cl.items():
            if d.get(cell):
                cells[cell] = (d, src)
    # a family a later pass of the same part ran (e.g. a source build retried after the run's first report) is not
    # unavailable there
    missing -= {cfam(leg) for d in legs.values() for leg in d}
    return progs, legs, cells, sources, missing


def verdict(ratio, null):
    if ratio is None:
        return None
    thr = max(null or 0.0, EVEN)
    return 'win' if ratio < 1 - thr else 'loss' if ratio > 1 + thr else 'even'


def pick(d):
    if not d:
        return None, None
    k = min(d, key=d.get)
    return d[k], k


def rows_for(part, progs, legs, cells):
    out = []
    for prog, (arms, src) in sorted(progs.items()):
        if not prog.startswith('pt_') or unpv(prog) != prog:
            continue
        w = prog[3:]
        s = per(w)
        on = med(arms['on']) / s
        null = abs(med(arms['null']) / s / on - 1) if arms.get('null') else None
        prec = row_prec(w)
        lg = legs.get(prog, {})
        lf, src_legs = lg, None
        if FAIR[prec] == 'bf16' and not any(leg_prec(k) == 'bf16' for k in lg):
            for t in twins(prog):
                if any(leg_prec(k) == 'bf16' for k in legs.get(t, {})):
                    lf, src_legs = legs[t], t
                    break
        fair = {k: v[0] / s for k, v in lf.items() if v[1] and leg_prec(k) == FAIR[prec]}
        every = {k: (round(v[0] / s, 6), v[3]) for k, v in lf.items() if leg_prec(k) == FAIR[prec]}
        best, bleg = pick(fair)
        tf32 = 'tf32' in prog
        old = {k: v[0] / s for k, v in lg.items() if v[1] and (k.startswith('torch-') or k.startswith('chain-'))
               and (tf32 or not k.endswith('-tf32')) and not is_new(k)}
        obest, oleg = pick(old)
        bf = {}
        for b in (prog, prog.removesuffix('_tf32'), prog.removesuffix('_tf32') + '_bf16w', *twins(prog)):
            bf = bf or {k: v[0] / s for k, v in legs.get(b, {}).items() if v[1] and leg_prec(k) == 'bf16'}
        bref, bfleg = pick(bf)
        out.append(row(part, w, prec, on, best, bleg, null, bref, bfleg, src, obest, oleg,
                       {k: round(v, 6) for k, v in sorted(fair.items(), key=lambda kv: kv[1])},
                       sorted({lf[k][2]['step'] for k in fair}), 'ms/step' if s > 1 else 'ms', every))
        out[-1]['legs_from'] = src_legs and src_legs[3:]
    for cell, (d, src) in sorted(cells.items()):
        g = {k: med(v) for k, v in d.items()}
        libs = {CELL_LIBS[k]: g[k] for k in CELL_LIBS if k in g}
        every = {k: (round(v, 6), '') for k, v in libs.items()}
        for k, v in legs.get(cell, {}).items():  # comp_bench's legs of the cell (hand Triton)
            if leg_prec(k) == 'f32':
                every[k] = (round(v[0], 6), v[3])
                if v[1]:
                    libs[k] = v[0]
        on = g[cell]
        null = abs(g[cell + '.null'] / on - 1) if cell + '.null' in g else None
        best, bleg = pick(libs)
        out.append(row(part, cell, 'f32', on, best, bleg, null, None, None, src, best, bleg,
                       {k: round(v, 6) for k, v in sorted(libs.items(), key=lambda kv: kv[1])}, [src['step']], 'ms',
                       every))
    return out


def row(part, w, prec, on, best, bleg, null, bref, bfleg, src, obest, oleg, fair, leg_steps, unit, every):
    ratio = on / best if best else None
    oratio = on / obest if obest else None
    base, bsleg = pick({k: v for k, v in fair.items() if not is_new(k)})
    bratio = on / base if base else None
    return dict(part=part, workload=w, family=family(w), precision=prec, unit=unit, mapal_ms=round(on, 6),
                best_ms=best and round(best, 6), best_leg=bleg, best_family=bleg and cfam(bleg),
                ratio=ratio and round(ratio, 4), null=null if null is None else round(null, 4),
                verdict=verdict(ratio, null), base_best_ms=base and round(base, 6), base_best_leg=bsleg,
                base_ratio=bratio and round(bratio, 4), base_verdict=verdict(bratio, null),
                bf16_ref_ms=bref and round(bref, 6), bf16_ref_leg=bfleg, commit=src['commit'], source=src['step'],
                leg_sources=leg_steps, legs=fair, every_leg=every, old_best_ms=obest and round(obest, 6),
                old_best_leg=oleg, old_ratio=oratio and round(oratio, 4), old_verdict=verdict(oratio, null))


def gm(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def wel(rs):
    v = [r['verdict'] for r in rs if r['verdict']]
    return f"{v.count('win')}/{v.count('even')}/{v.count('loss')}" if v else '-'


def f4(x):
    return f'{x:.4g}'


def cell_md(r):
    if r is None:
        return '-'
    if r['ratio'] is None:
        return f"{f4(r['mapal_ms'])} (no leg)"
    rt = f"{r['ratio']:.2f}"
    rt = f'**{rt}**' if r['verdict'] == 'win' else rt + '≈' if r['verdict'] == 'even' else rt
    nl = f" ±{r['null']:.0%}" if r['null'] and r['null'] > EVEN else ''
    return f"{rt} {f4(r['mapal_ms'])}/{f4(r['best_ms'])} ({r['best_leg']}){nl}"


def gcell(rs):
    g = gm([r['ratio'] for r in rs if r['ratio']])
    return f'{g:.3f} ({sum(1 for r in rs if r["ratio"])})' if g else '-'


def main(argv):
    out_dir = Path(__file__).resolve().parents[2] / 'benches/results/evidence'
    skip = None
    while argv[:1] in (['--out'], ['--skip']):
        if argv[0] == '--out':
            out_dir = Path(argv[1])
        else:
            skip = argv[1]
        argv = argv[2:]
    if not argv:
        sys.exit(__doc__)
    parts, allrows, srcs, missing = [], [], {}, {}
    for a in argv:
        part, _, files = a.partition('=')
        progs, legs, cells, sources, miss = load_part(files.split(','))
        rs = [r for r in rows_for(part, progs, legs, cells) if not (skip and re.search(skip, r['workload']))]
        parts.append(part)
        srcs[part] = sources
        missing[part] = miss
        allrows += rs
    by = {(r['part'], r['workload']): r for r in allrows}
    fam_order = {k: i for i, (k, _, _) in enumerate(FAMILIES)}
    names = sorted({r['workload'] for r in allrows},
                   key=lambda w: (fam_order[family(w)], PRECS.index(row_prec(w)), w))
    counted = [r for r in allrows if r['ratio']]
    old_parts = [p for p in parts if not any(is_new(l) for r in allrows if r['part'] == p for l in r['every_leg'])]
    common = [w for w in names if all(by.get((p, w), {}).get('ratio') for p in parts)]

    L = ['# Mapal vs the fastest competitor at the same precision', '',
         f'Built {date.today().isoformat()} by `benches/local/matrix.py` from existing results files (command at the '
         'end). Cell: ratio Mapal ms / best ms (best leg); **ratio < 1 = Mapal faster**. Bold = win, `≈` = even '
         '(|ratio − 1| ≤ max(null, 5 %)), plain = loss; `±n%` = a null above 5 %; `-` = not timed on that part. '
         'Decode rows of the LLM family are ms per step (16 steps / 16), every other row ms per program run.', '',
         '**Fairness:** f32 rows vs f32 legs, `_tf32` rows vs TF32 legs, `_bf16w` rows (bf16 weights, f32 '
         'activations), `_bf16kv` rows (weights and KV cache bf16, f32 compute) and `_bf16` rows (bf16 storage, '
         '`: bf16 | f32` products) vs the bf16 legs (bf16 throughout); a bf16-storage row without bf16 legs of its own '
         'takes its twin\'s (`_bf16w`, else the f32 program: the same bf16 computation); `_f8w` rows (fp8 weights, f32 '
         'compute) vs fp8 legs, of which none is installed: `(no leg)`, their `_bf16w` twin\'s bf16 leg is a reference '
         'only. The torch bf16 reference of an '
         'f32 / tf32 row is never '
         'its best; it is listed apart ([bf16 reference](#bf16-real-world-reference)). Torch / gemm_lib legs count '
         'only with `within=yes`.', '',
         '**Competitors:** cuBLAS, cuBLASLt (gemm_lib, flushed and `-fresh`; chain_bench), CUTLASS `b2f249a` (stock '
         'SIMT + tuned; chain_bench epilogues), hand CUDA (chain_bench `fused` / `streams` / `cache` / `gemv`), '
         'PyTorch 2.11 (cu130) eager, `torch.compile` mode max-autotune-no-cudagraphs (Inductor / Triton kernels), '
         'with CUDA graphs (`-cg`: mode max-autotune; `-ro`: mode reduce-overhead), SDPA (torch\'s pick, '
         '`-efficient`, `-cudnn`: cuDNN attention; eager and compiled), TF32 twins of each, torch bf16 / bf16-compile '
         '(SDPA flash); comp_bench.py: hand-written Triton (the v3.6.0 tutorials\' matmul / softmax / fused attention, '
         'autotuned), Liger 0.8.5 (RMSNorm, RoPE, SwiGLU), FlashAttention-2 2.8.3.post2 (flash-attn; built from '
         'source), FlashAttention-3 (its hopper/ build: sm_90a boxes), FlashInfer 0.7.5.post1. '
         '**Not included:** TensorRT-LLM (1.3.0 pins torch 2.14; see local-rtx-5080/competitors.md), vLLM / SGLang '
         'as servers. ' + (f'The {", ".join(old_parts)} column(s) carry legs from before the competitors step: '
                           'their new families read `not run`.' if old_parts else
                           'Every part ran the competitor legs (FlashAttention-3 on sm_90a only; `unavailable` = the '
                           'part\'s run reported the package missing or its build capped).'), '',
         '**Legs:** `torch-*` the PyTorch modes; `cublas` / `lt` (+ `-tf32`, `-fresh`: operands regenerated each '
         'repetition) gemm_lib cuBLAS / cuBLASLt; `chain-eager` cuBLAS sgemm + one CUDA kernel per step, '
         '`chain-streams` the same with independent parts on two streams, `chain-lt` / `chain-lt2` cuBLASLt with the '
         'fused epilogue (lt2: two streams), `chain-cutlass` CUTLASS SIMT with a fused epilogue, `chain-fused` a '
         'hand-fused CUDA kernel, `chain-cache` the hand-written KV-cache decode, `chain-gemv` sgemm + cublasSgemv; '
         '`cutlass` / `cutlass-tuned` the cells\' CUTLASS stock / tuned.', '',
         '| part | commit(s) | Mapal rows from (step@commit: rows) |', '|---|---|---|']
    for p in parts:
        cnt = defaultdict(int)
        for r in allrows:
            if r['part'] == p:
                cnt[(r['source'], r['commit'])] += 1
        commits = sorted({c for _, c in cnt})
        L.append(f'| {p} | {", ".join(commits)} | '
                 + '; '.join(f'{s}@{c}: {n}' for (s, c), n in sorted(cnt.items(), key=lambda kv: -kv[1])) + ' |')
    L += ['', '## Overall', '', '| geomean Mapal/best | ' + ' | '.join(parts) + ' |',
          '|---|' + '---:|' * len(parts)]
    for k, title, _ in FAMILIES:
        if any(r['family'] == k for r in counted):
            L.append(f'| {title} | ' + ' | '.join(gcell([r for r in counted if r['part'] == p and r['family'] == k])
                                                  for p in parts) + ' |')
    for pr in PRECS:
        L.append(f'| all {pr} rows | ' + ' | '.join(gcell([r for r in counted if r['part'] == p and
                                                           r['precision'] == pr]) for p in parts) + ' |')
    L.append('| **all rows** | ' + ' | '.join(gcell([r for r in counted if r['part'] == p]) for p in parts) + ' |')
    L.append(f'| rows timed on every part ({len(common)}) | ' + ' | '.join(
        gcell([by[(p, w)] for w in common]) for p in parts) + ' |')
    L.append('| win / even / loss | ' + ' | '.join(wel([r for r in counted if r['part'] == p]) for p in parts) + ' |')
    L += ['', 'Geomean cells: value (rows). ' + ('Every part times the same rows.' if len(common) == len(names) else
          'Parts time different row sets; compare columns on the "every part" row.')]

    for k, title, _ in FAMILIES:
        ws = [w for w in names if family(w) == k]
        if not ws:
            continue
        L += ['', f'## {title}', '', '| workload | prec | ' + ' | '.join(parts) + ' |',
              '|---|---|' + '---|' * len(parts)]
        for w in ws:
            L.append(f'| {w} | {row_prec(w)} | ' + ' | '.join(cell_md(by.get((p, w))) for p in parts) + ' |')
        fr = [r for r in counted if r['family'] == k]
        precs = [pr for pr in PRECS if any(r['precision'] == pr for r in fr)]
        if len(precs) > 1:
            for pr in precs:
                L.append(f'| geomean {pr} | | ' + ' | '.join(
                    gcell([r for r in fr if r['part'] == p and r['precision'] == pr]) for p in parts) + ' |')
        L.append('| **geomean** | | ' + ' | '.join(gcell([r for r in fr if r['part'] == p]) for p in parts) + ' |')
        L.append('| win / even / loss | | ' + ' | '.join(wel([r for r in fr if r['part'] == p]) for p in parts) + ' |')

    bases = [w for w in names if family(w) in ('gemm', 'llm') and row_prec(w) == 'f32'
             and any(by.get((p, w), {}).get('bf16_ref_ms') for p in parts)]
    L += ['', '## bf16 real-world reference', '',
          'The fastest bf16 leg (torch bf16 / bf16-compile(-cg / -ro) / bf16-cudnn / bf16-efficient, Liger, '
          'FlashAttention-2/3, FlashInfer, Triton — the layer in bfloat16; gated loosely against Mapal\'s f32 values) '
          'next to Mapal\'s own rows: cell = best bf16 leg ms (leg) · Mapal f32 / ref · Mapal `_bf16w` / ref (when '
          'the `_bf16w` twin exists). Not a like-for-like comparison: the reference rounds activations and the KV '
          'cache to bf16 too.', '', '| workload | ' + ' | '.join(parts) + ' |', '|---|' + '---|' * len(parts)]
    for w in bases:
        cs = []
        for p in parts:
            r, rw = by.get((p, w)), by.get((p, w + '_bf16w'))
            if not r or not r['bf16_ref_ms']:
                cs.append('-')
                continue
            ref = r['bf16_ref_ms']
            c = f"{f4(ref)} ({r['bf16_ref_leg'].removeprefix('torch-')}) · f32 {r['mapal_ms'] / ref:.2f}"
            if rw:
                c += f" · bf16w {rw['mapal_ms'] / ref:.2f}"
            cs.append(c)
        L.append(f'| {w} | ' + ' | '.join(cs) + ' |')

    flips = [r for r in allrows if r['verdict'] and r['old_verdict'] and r['verdict'] != r['old_verdict']]
    L += ['', '## Verdict flips vs the mixed-precision gap tables', '',
          'gap.py\'s rule on the same data (torch legs of any precision but TF32 on non-tf32 rows — so torch bf16 for '
          'f32 llm_ / pj_ rows —, chain legs, no gemm_lib legs) against the fair rule; it reproduces the gap tables '
          '(gap-<part>.md, gap.md) row for row. Rows a part\'s gap table did not cover (the RTX 5080\'s model-scale '
          'rows, reported in model-scale.md) are judged against that rule applied to the same data.', '',
          '| part | workload | old: best (leg) ratio | fair: best (leg) ratio | cause |', '|---|---|---|---|---|']
    for r in sorted(flips, key=lambda r: (parts.index(r['part']), r['workload'])):
        ol, nl = r['old_best_leg'], r['best_leg']
        why = ([f'old best a {leg_prec(ol)} leg'] if leg_prec(ol) != FAIR[r['precision']] else []) + (
            ['best now a gemm_lib leg'] if nl and not nl.startswith(('torch-', 'chain-')) else [])
        cause = ' + '.join(why) or 'other'
        L.append(f"| {r['part']} | {r['workload']} | {f4(r['old_best_ms'])} ({ol}) {r['old_ratio']:.2f} "
                 f"{r['old_verdict']} | {f4(r['best_ms']) if r['best_ms'] else '-'} ({nl}) "
                 f"{r['ratio']:.2f} {r['verdict']} | {cause} |")
    if not flips:
        L.append('| - | none | | | |')

    changed = [r for r in allrows if r['ratio'] and r['base_best_leg'] and r['best_leg'] != r['base_best_leg']]
    L += ['', '## What the new legs change', '',
          'Rows whose best leg is one the competitors step added (CUDA-graph torch.compile, SDPA backends, hand '
          'Triton, Liger, FlashAttention-2/3, FlashInfer), against the best over the legs before it (`base`).', '',
          '| part | workload | prec | base: best (leg) ratio | now: best (leg) ratio | verdict |',
          '|---|---|---|---|---|---|']
    for r in sorted(changed, key=lambda r: (parts.index(r['part']), fam_order[r['family']], r['workload'])):
        L.append(f"| {r['part']} | {r['workload']} | {r['precision']} | {f4(r['base_best_ms'])} "
                 f"({r['base_best_leg']}) {r['base_ratio']:.2f} | {f4(r['best_ms'])} ({r['best_leg']}) "
                 f"{r['ratio']:.2f} | {r['base_verdict']} → {r['verdict']} |")
    if not changed:
        L.append('| - | none | | | | |')
    lost = [r for r in changed if r['base_verdict'] == 'win' and r['verdict'] != 'win']
    L += ['', f'Mapal wins that disappear: {len(lost)}' + (' — ' + ', '.join(
        f"{r['part']} {r['workload']} ({r['best_leg']})" for r in lost) if lost else '') + '.']

    L += ['', '## Every competitor leg', '',
          'Per row and part, every leg at the row\'s precision, grouped by competitor family: `leg ms (Mapal/leg)`; '
          '**bold** = the row\'s fastest leg; `✗gate` = timed but outside the value gate (not counted); '
          '`cg-skipped` = Inductor ran the leg without its CUDA graph (not counted); `not run` = the family has a leg '
          'for this row on another part; `unavailable` = the part\'s run reported the package missing. Families: '
          + '; '.join(f'{t}' for _, t in CFAMS) + '. Columns with no leg on any row of a table are left out.']
    for k, title, _ in FAMILIES:
        ws = [w for w in names if family(w) == k]
        if not ws:
            continue
        fams = [(fk, ft) for fk, ft in CFAMS if any(cfam(l) == fk for w in ws for p in parts
                                                     for l in by.get((p, w), {}).get('every_leg', {}))]
        L += ['', f'### {title}', '', '| workload | part | Mapal ms | ' + ' | '.join(t for _, t in fams) + ' |',
              '|---|---|---:|' + '---|' * len(fams)]
        for w in ws:
            ran = {fk for p in parts for l in by.get((p, w), {}).get('every_leg', {}) for fk in [cfam(l)]}
            for p in parts:
                r = by.get((p, w))
                if not r:
                    continue
                cs = []
                for fk, _ in fams:
                    ls = sorted(((l, v) for l, v in r['every_leg'].items() if cfam(l) == fk), key=lambda lv: lv[1][0])
                    if not ls:
                        cs.append('unavailable' if fk in missing[p] and fk in ran else 'not run' if fk in ran else '')
                        continue
                    items = []
                    for l, (ms, note) in ls:
                        txt = f"{l} {f4(ms)}" + (f" ✗gate" if note == 'gate' else ' cg-skipped' if note else
                                                  f" ({r['mapal_ms'] / ms:.2f})")
                        items.append(f'**{txt}**' if l == r['best_leg'] else txt)
                    cs.append('<br>'.join(items))
                L.append(f"| {w} | {p} | {f4(r['mapal_ms'])} | " + ' | '.join(cs) + ' |')

    L += ['', '## Command', '', '```', 'python3 benches/local/matrix.py ' + (f"--skip '{skip}' " if skip else '') + '\\']
    for i, p in enumerate(parts):
        files = ','.join(f"{s['file']}@{s['commit']}" for s in srcs[p])
        L.append(f'  {p}={files}' + (' \\' if i < len(parts) - 1 else ''))
    L += ['```', '', 'Raw results files are local (`.claude/raw-evidence/`, not in git); `matrix.json` holds every row '
          '(all fair legs, sources, the bf16 reference, the old rule\'s best).', '']

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'matrix.md').write_text('\n'.join(L))
    meta = dict(generated=date.today().isoformat(), parts=parts, even_threshold=EVEN,
                sources={p: srcs[p] for p in parts}, families={k: t for k, t, _ in FAMILIES},
                competitor_families={k: t for k, t in CFAMS}, unavailable={p: sorted(missing[p]) for p in parts},
                rule='ratio = mapal_ms / best_ms; best = fastest within leg at the row precision '
                     '(f32 -> f32 legs, tf32 -> tf32 legs, bf16w / bf16kv / bf16 -> bf16 legs; legs_from: the twin '
                     'whose bf16 legs a bf16-storage row takes)')
    (out_dir / 'matrix.json').write_text(json.dumps(dict(meta=meta, rows=allrows), indent=1) + '\n')
    print(f'wrote {out_dir}/matrix.md, matrix.json: {len(allrows)} rows, {len(flips)} verdict flips')


if __name__ == '__main__':
    main(sys.argv[1:])
