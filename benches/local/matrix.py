#!/usr/bin/env python3
"""Mapal vs the fastest competitor AT THE SAME PRECISION, per GPU part and algorithm family: one matrix from the
results files box_run.sh / gpu_run.sh / model_scale_run.sh write (gap.py's and model_scale_report.py's parsing).

usage: matrix.py [--out DIR] <part>=<file>[@<commit>][,<file>[@<commit>]...] ...
  One column per part, in argument order. List a part's files oldest first: a later file replaces an earlier one
  per program (Mapal's `on` / `null` samples come from the newest file that timed the program `on`), per competitor
  leg (the newest file that timed that leg for the program; legs do not depend on Mapal's commit) and per GEMM cell.
  `@<commit>` labels a file (default: an 8-hex path component, else `?`). Writes DIR/matrix.md and DIR/matrix.json
  (DIR defaults to benches/results/evidence). The command that built the committed matrix is at the end of matrix.md.

Rules
- Mapal = median `iter ms` of the shipped build (probe arm `on`, face `_default`); null = |median(null) / median(on)
  - 1| from the same file. GEMM cells (tile_bench splitk section): the default-face build, medians over passes.
- Fairness: a row's best competitor is the fastest leg at the row's precision.
    f32     torch eager / compile / sdpa / sdpa-compile; gemm_lib cuBLAS / cuBLASLt (`-fresh`: operands regenerated
            each repetition); chain_bench legs `chain-*` (cuBLAS, cuBLASLt, CUTLASS, hand CUDA, all
            CUBLAS_COMPUTE_32F); the cells' cuBLAS / CUTLASS / CUTLASS tuned.
    _tf32   the TF32 legs: torch eager-tf32 / compile-tf32 / sdpa-tf32, gemm_lib cublas-tf32 / lt-tf32 (+ `-fresh`).
    _bf16w  the bf16 legs: torch bf16 / bf16-compile (the layer in bfloat16, SDPA flash).
  Torch and gemm_lib legs count only with `within=yes`. `bf16_ref` = the row's (else its `_tf32`-less base stem's)
  best torch bf16 leg: the real-world reference, reported apart, never the best of an f32 / tf32 row.
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
GEMM = re.compile(r'^gemm prog=(\S+) leg=(\S+) .*med_ms=([\d.]+) .*within=(\w+)')
CHAIN = re.compile(r'^chain probe=(\S+) ((?:\w+=\S+ )*?)leg=(\S+) .*?(?:wall_med_ms|med_ms)=([\d.]+)')
CELL = re.compile(r'^pass=\d+ cell=(\S+) time leg=(\S+) .*med_ms=([\d.]+)')
CELL_LIBS = {'cublas': 'cublas', 'cutlass': 'cutlass', 'cutlass_tuned': 'cutlass-tuned'}
STEPS = 16
FAMILIES = [  # (key, title, workload pattern)
    ('gemm', 'GEMM cells & projections', r'(mm|pj|epi)_'),
    ('mlp', 'MLP chains', r'(mlp|mlp2|smlp)_'),
    ('attn', 'Attention / softmax / long rows', r'(attn|causal|decode|gen|prefill|lsm|softmax_rows|rowsum)_'),
    ('llm', 'LLM decode / prefill / block (f32, tf32, bf16w)', r'llm_'),
    ('conv', 'Conv / FIR / elementwise', r'(conv|convmc|ew|ewconvew|fir|firew)_'),
    ('fan', 'Fan-in/out, loops, CPU+GPU', r'(diamond|fanout|indep|twin|loop|loopmm|rt)_'),
    ('other', 'Other', r''),
]
PRECS = ('f32', 'tf32', 'bf16w')
EVEN = 0.05


def family(w):
    return next(k for k, _, p in FAMILIES if re.match(p, w))


def row_prec(w):
    return 'bf16w' if w.endswith('_bf16w') else 'tf32' if w.endswith('_tf32') else 'f32'


def leg_prec(leg):
    return 'bf16' if leg.startswith('torch-bf16') else 'tf32' if 'tf32' in leg else 'f32'


FAIR = {'f32': 'f32', 'tf32': 'tf32', 'bf16w': 'bf16'}


def per(w):
    return STEPS if re.match(r'llm_(blk)?dec_', w) else 1


def unpv(prog):
    return re.sub(r'_pv(?=_|$)', '', prog)


def label(path):
    m = re.search(r'raw-evidence/[^/]+/(.+)/[^/]+$', path)
    return m[1] if m else Path(path).parent.name or path


def parse(path):
    """-> mapal {prog: {arm: [ms]}}, legs {stem: {leg: (ms, within)}}, cells {cell: {leg: [ms]}}"""
    mapal = defaultdict(lambda: defaultdict(list))
    legs = defaultdict(dict)
    cells = defaultdict(lambda: defaultdict(list))
    for line in open(path, errors='replace'):
        if m := PROBE.match(line):
            mapal[m[1]][m[2]].append(float(m[3]))
        elif m := TORCH.match(line):
            legs[unpv(m[1])]['torch-' + m[2]] = (float(m[3]), m[4] == 'yes')
        elif m := GEMM.match(line):
            legs[unpv(m[1])][m[2]] = (float(m[3]), m[4] == 'yes')
        elif m := CHAIN.match(line):
            nums = '_'.join(x.split('=')[1] for x in m[2].split())
            legs[f'pt_{m[1]}_{nums}']['chain-' + m[3]] = (float(m[4]), True)
        elif m := CELL.match(line):
            cells[m[1]][m[2]].append(float(m[3]))
    return mapal, legs, cells


def load_part(specs):
    """Merge a part's files (oldest first) -> programs, legs, cells, sources."""
    progs, legs, cells, sources = {}, defaultdict(dict), {}, []
    for spec in specs:
        path, _, commit = spec.partition('@')
        if not commit:
            commit = next((c for c in Path(path).parts if re.fullmatch(r'[0-9a-f]{8}', c)), '?')
        src = {'file': path, 'step': label(path), 'commit': commit}
        sources.append(src)
        mapal, lg, cl = parse(path)
        for prog, arms in mapal.items():
            if arms.get('on'):
                progs[prog] = (arms, src)
        for stem, d in lg.items():
            for leg, (ms, ok) in d.items():
                legs[stem][leg] = (ms, ok, src)
        for cell, d in cl.items():
            if d.get(cell):
                cells[cell] = (d, src)
    return progs, legs, cells, sources


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
        fair = {k: v[0] / s for k, v in lg.items() if v[1] and leg_prec(k) == FAIR[prec]}
        best, bleg = pick(fair)
        tf32 = 'tf32' in prog
        old = {k: v[0] / s for k, v in lg.items() if v[1] and (k.startswith('torch-') or k.startswith('chain-'))
               and (tf32 or not k.endswith('-tf32'))}
        obest, oleg = pick(old)
        bf = {k: v[0] / s for k, v in lg.items() if v[1] and leg_prec(k) == 'bf16'}
        if not bf:
            base = legs.get(prog.removesuffix('_tf32'), {})
            bf = {k: v[0] / s for k, v in base.items() if v[1] and leg_prec(k) == 'bf16'}
        bref, bfleg = pick(bf)
        out.append(row(part, w, prec, on, best, bleg, null, bref, bfleg, src, obest, oleg,
                       {k: round(v, 6) for k, v in sorted(fair.items(), key=lambda kv: kv[1])},
                       sorted({lg[k][2]['step'] for k in fair}), 'ms/step' if s > 1 else 'ms'))
    for cell, (d, src) in sorted(cells.items()):
        g = {k: med(v) for k, v in d.items()}
        libs = {CELL_LIBS[k]: g[k] for k in CELL_LIBS if k in g}
        on = g[cell]
        null = abs(g[cell + '.null'] / on - 1) if cell + '.null' in g else None
        best, bleg = pick(libs)
        out.append(row(part, cell, 'f32', on, best, bleg, null, None, None, src, best, bleg,
                       {k: round(v, 6) for k, v in sorted(libs.items(), key=lambda kv: kv[1])}, [src['step']], 'ms'))
    return out


def row(part, w, prec, on, best, bleg, null, bref, bfleg, src, obest, oleg, fair, leg_steps, unit):
    ratio = on / best if best else None
    oratio = on / obest if obest else None
    return dict(part=part, workload=w, family=family(w), precision=prec, unit=unit, mapal_ms=round(on, 6),
                best_ms=best and round(best, 6), best_leg=bleg, ratio=ratio and round(ratio, 4),
                null=null if null is None else round(null, 4), verdict=verdict(ratio, null),
                bf16_ref_ms=bref and round(bref, 6), bf16_ref_leg=bfleg, commit=src['commit'], source=src['step'],
                leg_sources=leg_steps, legs=fair, old_best_ms=obest and round(obest, 6), old_best_leg=oleg,
                old_ratio=oratio and round(oratio, 4), old_verdict=verdict(oratio, null))


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
    if argv[:1] == ['--out']:
        out_dir, argv = Path(argv[1]), argv[2:]
    if not argv:
        sys.exit(__doc__)
    parts, allrows, srcs = [], [], {}
    for a in argv:
        part, _, files = a.partition('=')
        progs, legs, cells, sources = load_part(files.split(','))
        rs = rows_for(part, progs, legs, cells)
        parts.append(part)
        srcs[part] = sources
        allrows += rs
    by = {(r['part'], r['workload']): r for r in allrows}
    fam_order = {k: i for i, (k, _, _) in enumerate(FAMILIES)}
    names = sorted({r['workload'] for r in allrows},
                   key=lambda w: (fam_order[family(w)], PRECS.index(row_prec(w)), w))
    counted = [r for r in allrows if r['ratio']]
    common = [w for w in names if all(by.get((p, w), {}).get('ratio') for p in parts)]

    L = ['# Mapal vs the fastest competitor at the same precision', '',
         f'Built {date.today().isoformat()} by `benches/local/matrix.py` from existing results files (command at the '
         'end). Cell: ratio Mapal ms / best ms (best leg); **ratio < 1 = Mapal faster**. Bold = win, `≈` = even '
         '(|ratio − 1| ≤ max(null, 5 %)), plain = loss; `±n%` = a null above 5 %; `-` = not timed on that part. '
         'Decode rows of the LLM family are ms per step (16 steps / 16), every other row ms per program run.', '',
         '**Fairness:** f32 rows vs f32 legs, `_tf32` rows vs TF32 legs, `_bf16w` rows (bf16 weights, f32 '
         'activations) vs the torch bf16 legs (bf16 throughout). The torch bf16 reference of an f32 / tf32 row is never '
         'its best; it is listed apart ([bf16 reference](#bf16-real-world-reference)). Torch / gemm_lib legs count '
         'only with `within=yes`.', '',
         '**Competitors:** cuBLAS, cuBLASLt (gemm_lib, flushed and `-fresh`; chain_bench), CUTLASS `b2f249a` (stock '
         'SIMT + tuned; chain_bench epilogues), hand CUDA (chain_bench `fused` / `streams` / `cache` / `gemv`), '
         'PyTorch 2.11 (cu130) eager, `torch.compile` mode max-autotune-no-cudagraphs (Inductor / Triton kernels), '
         'SDPA (eager and compiled), TF32 twins of each, torch bf16 / bf16-compile (SDPA flash). '
         '**Not included:** torch with CUDA graphs, hand-written Triton kernels, the FlashAttention-2/3 package, '
         'FlashInfer, cuDNN attention, TensorRT-LLM / vLLM.', '',
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
    L += ['', 'Geomean cells: value (rows). Parts time different row sets (the RTX 5080 adds 5080-only rows, the rented '
          'parts the full model-scale set); compare columns on the "every part" row.']

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
          'Torch bf16 / bf16-compile (the layer in bfloat16, SDPA flash; gated loosely against Mapal\'s f32 values) '
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
            c = f"{f4(ref)} ({r['bf16_ref_leg'][6:]}) · f32 {r['mapal_ms'] / ref:.2f}"
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

    L += ['', '## Command', '', '```', 'python3 benches/local/matrix.py \\']
    for i, p in enumerate(parts):
        files = ','.join(f"{s['file']}@{s['commit']}" for s in srcs[p])
        L.append(f'  {p}={files}' + (' \\' if i < len(parts) - 1 else ''))
    L += ['```', '', 'Raw results files are local (`.claude/raw-evidence/`, not in git); `matrix.json` holds every row '
          '(all fair legs, sources, the bf16 reference, the old rule\'s best).', '']

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'matrix.md').write_text('\n'.join(L))
    meta = dict(generated=date.today().isoformat(), parts=parts, even_threshold=EVEN,
                sources={p: srcs[p] for p in parts}, families={k: t for k, t, _ in FAMILIES},
                rule='ratio = mapal_ms / best_ms; best = fastest within leg at the row precision '
                     '(f32 -> f32 legs, tf32 -> tf32 legs, bf16w -> torch bf16 legs)')
    (out_dir / 'matrix.json').write_text(json.dumps(dict(meta=meta, rows=allrows), indent=1) + '\n')
    print(f'wrote {out_dir}/matrix.md, matrix.json: {len(allrows)} rows, {len(flips)} verdict flips')


if __name__ == '__main__':
    main(sys.argv[1:])
