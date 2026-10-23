#!/usr/bin/env python3
"""S64 evidence pass, E: the report tables from the raw evidence (benches/results/evidence/s64/).

usage: evidence_report.py <raw dir>
  <raw>/<arch>/evidence-<arch>.tsv     the intent check over the box run (static facts + `exec`)
  <raw>/static/<arch>/evidence.tsv     the same check from the bundle alone (used where the box one is absent)
  <raw>/<arch>/results-<arch>.txt      box_run.sh: splitk cells, mlp, probes (suite), torch legs, jit
  <raw>/judge.tsv                      rejudge.py judge (re-judged GPU_RUNGS_OFF rows, re-measured ON/OFF)
Prints markdown: (1) intended vs realized per rung x arch, (2) mismatches on shipped programs, (3) the
suite per workload x arch, (4) GEMM / MLP kernels vs the libraries, (5) re-judged rows. Medians only;
`null` = max over passes / reps of |null arm / shipped - 1|.
"""
import os
import re
import sys
from collections import Counter, defaultdict

ARCHS = ('rtx-4090', 'rtx-5090', 'h100-pcie', 'rtx-3090')
med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
f4 = lambda x: '-' if x is None else f'{x:.4f}' if x < 10 else f'{x:.2f}'
f3 = lambda x: '-' if x is None else f'{x:.3f}'


def tsv(raw, arch):
    for p in (os.path.join(raw, arch, f'evidence-{arch}.tsv'), os.path.join(raw, 'static', arch, 'evidence.tsv')):
        if os.path.exists(p) and os.path.getsize(p) > 0:
            rows = [line.rstrip('\n').split('\t') for line in open(p)]
            return [r for r in rows if len(r) >= 8 and r[0] != 'arch'], p
    return [], None


def intent_tables(raw):
    counts = defaultdict(Counter)  # (rung, arch) -> status counts, shipped programs only
    mism = defaultdict(set)        # (rung, arch) -> {(kernel-free stem, check, realized)}
    for a in ARCHS:
        rows, _ = tsv(raw, a)
        for arch, stem, k, rung, label, expr, real, st in (r[:8] for r in rows):
            if '.' in stem:  # an arm: C's evidence, not the shipped build
                continue
            counts[(rung, a)][st] += 1
            if st != 'ok':
                mism[(rung, a)].add((stem, label, real))
    rungs = sorted({r for r, _ in counts})
    print('### Intended vs realized (shipped programs; ok / MISMATCH / n-a)\n')
    print('| rung:state | ' + ' | '.join(ARCHS) + ' |')
    print('|---|' + '---|' * len(ARCHS))
    for r in rungs:
        cells = []
        for a in ARCHS:
            c = counts.get((r, a))
            cells.append(f'{c["ok"]} / {c["MISMATCH"]} / {c["n/a"]}' if c else '-')
        print(f'| {r} | ' + ' | '.join(cells) + ' |')
    print('\n### Findings: every mismatch on a shipped program\n')
    print('| arch | rung | check | programs | realized (first) |')
    print('|---|---|---|---|---|')
    for (r, a), s in sorted(mism.items()):
        by = defaultdict(list)
        for stem, label, real in sorted(s):
            by[label].append((stem, real))
        for label, xs in by.items():
            print(f'| {a} | {r} | {label} | {" ".join(sorted({x for x, _ in xs}))} | {xs[0][1][:120]} |')
    print()


def results(raw, a):
    p = os.path.join(raw, a, f'results-{a}.txt')
    return open(p, errors='replace').read().splitlines() if os.path.exists(p) else []


def suite_tables(raw):
    print('### Workload suite: iter ms (median of the reps), CPU-only / GPU-only / CPU+GPU, libraries, PyTorch\n')
    print('| workload | arch | CPU-only | GPU-only | CPU+GPU (shipped) | null | best library | torch eager | '
          'torch compile | shipped / best other |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for a in ARCHS:
        probes = defaultdict(list)
        chain = defaultdict(dict)
        torch = {}
        for l in results(raw, a):
            if l.startswith('probe prog='):
                f = dict(re.findall(r'(\w+)=([^\s\[\]]+)', l))
                if (it := re.search(r'iter ms=\s*([-\d.eE+]+)', l)) and f.get('face') == '_default':
                    probes[(f['prog'], f['arm'])].append(float(it[1]))
            elif m := re.match(r'chain probe=(\S+) (?:n=(\d+)|T=(\d+) d=(\d+)) leg=(\S+) .*?(?:med_ms|wall_med_ms)=([\d.]+)', l):
                key = f'{m[1]}_{m[2]}' if m[2] else f'{m[1]}_{m[3]}_{m[4]}'
                chain[key][m[5]] = float(m[6])
            elif m := re.match(r'torch prog=(\S+) mode=(\w+) .*med_ms=([\d.]+) .*within=(\S+)', l):
                torch[(m[1], m[2])] = (float(m[3]), m[4])
        progs = sorted({p for p, _ in probes})
        for p in progs:
            g = lambda arm: med(probes.get((p, arm), []))
            on, cpu, gpu, nul = g('on'), g('cpu'), g('off-place'), g('null')
            gpu_s = f4(gpu) if gpu is not None else '= shipped'
            null = max((abs(x / on - 1) for x in [nul] if x is not None and on), default=None)
            key = p.removeprefix('pt_')
            lib = chain.get(key) or chain.get(key.replace('decode_1024_256', 'decode_1024_256')) or {}
            if p.startswith('pt_decode_'):
                lib = chain.get('decode_1024_256', {})
            bl = min(lib.items(), key=lambda kv: kv[1]) if lib else None
            te, tc = torch.get((p, 'eager')), torch.get((p, 'compile'))
            others = [x for x in [bl[1] if bl else None, te[0] if te and te[1] == 'yes' else None,
                                  tc[0] if tc and tc[1] == 'yes' else None] if x]
            ratio = on / min(others) if on and others else None
            tf = lambda t: '-' if not t else f'{t[0]:.4f}' + ('' if t[1] == 'yes' else f' ({t[1]})')
            print(f'| {key} | {a} | {f4(cpu)} | {gpu_s} | {f4(on)} | {f3(null)} | '
                  f'{f"{bl[0]} {bl[1]:.4f}" if bl else "-"} | {tf(te)} | {tf(tc)} | {f3(ratio)} |')
    print()


def kernel_tables(raw):
    print('### GEMM cells and MLP kernels: Mapal vs cuBLAS / cuBLASLt / CUTLASS (median ms, both processes)\n')
    print('| cell | arch | Mapal default | realization | Mapal _c | Mapal _x | cuBLAS | cuBLASLt | CUTLASS | tuned CUTLASS | '
          'default / best library | null |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for a in ARCHS:
        samples = defaultdict(lambda: defaultdict(list))
        pmed = defaultdict(lambda: defaultdict(dict))
        comb = {}
        for l in results(raw, a):
            m = re.match(r'pass=(\d+) (cell|mlp)=(\S+) (.*)', l)
            if not m:
                continue
            p, cell, rest = int(m[1]), m[3], m[4]
            if s := re.match(r'samples leg=(\S+) (?:event=)?(\S*)', rest):
                samples[cell][s[1]] += [float(x) for x in s[2].split(',') if re.fullmatch(r'[\d.]+', x)]
            if t := re.match(r'time leg=(\S+) .*?(?:event_med_ms|med_ms)=([\d.]+)', rest):
                pmed[cell][t[1]][p] = float(t[2])
            if c := re.match(r'arm stem=(\S+) .* S=(\d+) combine=(\S+)', rest):
                comb[c[1]] = c[3] + (f' S={c[2]}' if c[3] in ('workspace', 'atomic') else '')
        for cell in sorted(pmed):
            s, pm = samples[cell], pmed[cell]
            mm = cell.startswith('mm_')
            base = cell if mm else f'{cell}_c'  # the MLP's default face is its contract face (S63)
            m_ = lambda leg: med(s[leg]) if s.get(leg) else (med(list(pm[leg].values())) if leg in pm else None)
            d = m_(base)
            libs = {k: m_(k) for k in ('cublas', 'cublaslt', 'cutlass', 'cutlass_tuned') if m_(k) is not None}
            best = min(libs.values()) if libs else None
            nul = [abs(pm[base + '.null'][p] / pm[base][p] - 1) for p in pm.get(base, {}) if p in pm.get(base + '.null', {})]
            print(f'| {cell} | {a} | {f4(d)} | {comb.get(base, "-")} | {f4(m_(cell + "_c")) if mm else "="} | '
                  f'{f4(m_(cell + "_x"))} | {f4(libs.get("cublas"))} | {f4(libs.get("cublaslt"))} | '
                  f'{f4(libs.get("cutlass"))} | {f4(libs.get("cutlass_tuned"))} | '
                  f'{f3(d / best) if d and best else "-"} | {f3(max(nul)) if nul else "-"} |')
    print()


def judge_table(raw):
    p = os.path.join(raw, 'judge.tsv')
    if not os.path.exists(p):
        return
    print('### Re-judged GPU_RUNGS_OFF rows\n')
    print('| row | part | rung | fill | face | cells | status | evidence (ON vs shipped) | re-measured ON/shipped |')
    print('|---|---|---|---|---|---|---|---|---|')
    for line in open(p):
        r = line.rstrip('\n').split('\t')
        if r[0] == 'row' or len(r) < 9:
            continue
        print('| ' + ' | '.join(x.replace('|', '/')[:220] for x in r[:9]) + ' |')
    print()


if __name__ == '__main__':
    raw = sys.argv[1]
    intent_tables(raw)
    kernel_tables(raw)
    suite_tables(raw)
    judge_table(raw)
