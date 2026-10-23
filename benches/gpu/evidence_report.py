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
    print('| workload | arch | CPU-only | GPU-only | CPU+GPU (shipped) | null | kernel / copy / host ms, launches, MB '
          '(traced rep) | best library | torch eager | torch compile | shipped / best other |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for a in ARCHS:
        probes = defaultdict(list)
        trace = {}
        chain = defaultdict(dict)
        torch = {}
        for l in results(raw, a):
            if l.startswith('probe prog='):
                f = dict(re.findall(r'(\w+)=([^\s\[\]]+)', l))
                if (it := re.search(r'iter ms=\s*([-\d.eE+]+)', l)) and f.get('face') == '_default':
                    probes[(f['prog'], f['arm'])].append(float(it[1]))
                    # MAPAL_GPU_TRACE phases of the shipped arm: what the iteration spends outside the device
                    if f['arm'] == 'on' and 'mapal_gpu_trace_total' in l and f['prog'] not in trace:
                        ph = sum(float(f.get(k, 0)) for k in ('h2d_ms', 'kernel_ms', 'gen_ms', 'd2h_ms'))
                        trace[f['prog']] = (max(0.0, float(it[1]) - ph), float(f.get('kernel_ms', 0)),
                                            float(f.get('h2d_ms', 0)) + float(f.get('d2h_ms', 0)),
                                            int(f.get('launches', 0)),
                                            (int(f.get('bytes_in', 0)) + int(f.get('bytes_out', 0))) / 2 ** 20)
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
            t = trace.get(p)
            tr = f'{t[1]:.2f} / {t[2]:.2f} / {t[0]:.2f}, {t[3]}, {t[4]:.0f}' if t else '-'
            print(f'| {key} | {a} | {f4(cpu)} | {gpu_s} | {f4(on)} | {f3(null)} | {tr} | '
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
            libs = {k: m_(k) for k in ('cublas', 'lt', 'cutlass', 'cutlass_tuned') if m_(k) is not None}
            best = min(libs.values()) if libs else None
            nul = [abs(pm[base + '.null'][p] / pm[base][p] - 1) for p in pm.get(base, {}) if p in pm.get(base + '.null', {})]
            print(f'| {cell} | {a} | {f4(d)} | {comb.get(base, "-")} | {f4(m_(cell + "_c")) if mm else "="} | '
                  f'{f4(m_(cell + "_x"))} | {f4(libs.get("cublas"))} | {f4(libs.get("lt"))} | '
                  f'{f4(libs.get("cutlass"))} | {f4(libs.get("cutlass_tuned"))} | '
                  f'{f3(d / best) if d and best else "-"} | {f3(max(nul)) if nul else "-"} |')
    print()


def legs(lines):
    """leg -> {process key: ms}: tile_bench / mlp_bench per-pass medians (`p1`, `p2`), probe `iter ms` per rep."""
    out = defaultdict(dict)
    for l in lines:
        if m := re.search(r'^pass=(\d) (?:cell|mlp)=\S+ time leg=(\S+) .*?(?:event_med_ms|med_ms)=([\d.]+)', l):
            out[m[2]].setdefault(f'p{m[1]}', float(m[3]))
        elif l.startswith('probe prog=') and (it := re.search(r'iter ms=\s*([-\d.eE+]+)', l)):
            f = dict(re.findall(r'(\w+)=([^\s\[\]]+)', l))
            face = '' if f['face'] == '_default' else f['face']
            out[f['prog'] + face + ('' if f['arm'] == 'on' else '.' + f['arm'])][f['rep']] = float(it[1])
    return out


def time_table(raw, title='How each optimization turned out in time (arms timed this session: ON / OFF, median over processes)'):
    """Per rung x arch, every arm this session timed: ON / OFF per process (an `.off-<r>` arm: shipped / arm;
    an `.on-<r>` arm: arm / shipped), against the program's null."""
    agg = defaultdict(list)
    for a in ARCHS:
        lg = legs(results(raw, a))
        for leg, arm in lg.items():
            m = re.match(r'(.+?)\.(on|off)-([\w+-]+)$', leg)
            if not m or m[1] not in lg:
                continue
            base, d, rung = m[1], m[2], m[3].split('+')[-1]
            sh = lg[base]
            ks = [k for k in arm if k in sh]
            if any(k.startswith('p') for k in ks):  # a kernel arm: its tile_bench / mlp_bench events
                ks = [k for k in ks if k.startswith('p')]
            if not ks:
                continue
            r = [(arm[k] / sh[k]) if d == 'on' else (sh[k] / arm[k]) for k in ks]
            root = re.sub(r'_(x|c|r|rc|tf)$', '', base)
            kind, nul = ks[0].startswith('p'), None
            for n0, s0 in ((root + '.null', root), (root + '_c.null', root + '_c')):
                nl, sh0 = lg.get(n0, {}), lg.get(s0, {})
                xs = [abs(nl[k] / sh0[k] - 1) for k in nl if k in sh0 and k.startswith('p') == kind]
                if xs:
                    nul = max(xs)
                    break
            sign = ('ON slower' if nul is not None and all(x > 1 + nul for x in r) else
                    'ON faster' if nul is not None and all(x < 1 - nul for x in r) else 'inside')
            agg[(rung, a)].append((base, med(r), nul, sign, 'shipped ON' if d == 'off' else 'shipped OFF'))
    print(f'### {title}\n')
    print('| rung | arch | shipped | cells: ON/OFF (null) | ON faster / slower / inside null |')
    print('|---|---|---|---|---|')
    for (rung, a), xs in sorted(agg.items()):
        c = Counter(x[3] for x in xs)
        cells = '; '.join(f'{b} {r:.3f} ({"-" if n is None else f"{n:.3f}"})' for b, r, n, _, _ in sorted(xs))
        print(f'| {rung} | {a} | {"/".join(sorted({x[4] for x in xs}))} | {cells} | '
              f'{c["ON faster"]} / {c["ON slower"]} / {c["inside"]} |')
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
    time_table(raw)
    if os.path.isdir(os.path.join(raw, 'warp-fill')):
        time_table(os.path.join(raw, 'warp-fill'), 'Follow-up: warp-fill ON on the MLP contract kernels (H100, 3090)')
    kernel_tables(raw)
    suite_tables(raw)
    judge_table(raw)
