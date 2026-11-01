#!/usr/bin/env python3
"""Baseline tables from the five base-<part>.txt (and -confirm.txt) results files.
usage: BASELINE_HEAD="$(cat <headline.md>)" python3 benches/gpu/baseline.py benches/results/gpu/validation/baseline > /tmp/baseline-tables.md
(BASELINE_HEAD: the hand-written headline printed under the title)"""
import os, re, subprocess, sys
from collections import defaultdict

D = sys.argv[1]
PARTS = [('rtx-3090', 'RTX 3090 (sm_86)'), ('a100', 'A100 (sm_80)'), ('rtx-4090', 'RTX 4090 (sm_89)'),
         ('h100-pcie', 'H100 PCIe (sm_90a)'), ('h100', 'H100 SXM (sm_90a)'), ('rtx-5090', 'RTX 5090 (sm_120)')]
med = lambda xs: sorted(xs)[len(xs) // 2]
fields = lambda l: dict(kv.split('=', 1) for kv in re.findall(r'(\w+=[^\s\[\]]+)', l))


def kernels(path):
    rows, cur = {}, {}
    for l in open(path):
        m = re.match(r'# n=(\d+) face=(\S+)', l)
        if m:
            cur = {'n': int(m[1]), 'key': m[2]}
            continue
        m = re.match(r'time leg=(\w+) .* min_ms=([\d.]+) med_ms=([\d.]+) max_ms=([\d.]+)', l)
        if m:
            cur[m[1]] = tuple(float(x) for x in m.groups()[1:])
        if l.startswith('ratio') and 'mapal' in cur:
            rows[(cur['n'], cur['key'])] = cur
    return rows


def arms(rows):
    out = {}
    for (n, key), r in rows.items():
        if '.' not in key:
            continue
        face, arm = key.split('.', 1)
        if not (arm.startswith('off-') or arm.startswith('on-')) or (n, face) not in rows:
            continue
        base = rows[(n, face)]
        d, rung = arm.split('-', 1)
        on, off = (base, r) if d == 'off' else (r, base)
        disj = on['mapal'][2] < off['mapal'][0] or off['mapal'][2] < on['mapal'][0]
        out[(rung, face, n)] = (off['mapal'][1] / on['mapal'][1], disj, 'ON' if d == 'off' else 'OFF',
                                on['mapal'][1], off['mapal'][1])
    return out


def probes(path):
    runs, legs, gates = defaultdict(list), defaultdict(dict), defaultdict(list)
    for l in open(path):
        if l.startswith('probe prog='):
            f = fields(l)
            it = re.search(r'iter ms=\s*([-\d.eE+]+)', l)
            runs[(f['prog'], f['face'], f['arm'])].append({
                'iter': float(it[1]) if it else float('nan'), 'wall': int(f['wall_us']) / 1e3,
                'kernel': float(f.get('kernel_ms', 'nan')), 'd2h': float(f.get('d2h_ms', 'nan')),
                'h2d': float(f.get('h2d_ms', 'nan')), 'launches': f.get('launches', '?'),
                'in': int(f.get('bytes_in', 0)), 'out': int(f.get('bytes_out', 0))})
        elif l.startswith('chain probe='):
            f = fields(l)
            legs[(f['probe'], f['n'])][f['leg']] = (float(f['med_ms']), f['launches'])
        elif l.startswith('chain_values '):
            f = fields(l)
            if f['gate'] != 'pass':
                gates[(f['probe'], f['n'])].append(f"{f['leg']}:{f['gate']}")
    return runs, legs, gates


def header(path):
    h = {'diff': '', 'trap': [0, 0], 'gpu': '', 'box': ''}
    for l in open(path):
        if l.startswith('differential:'):
            h['diff'] = l.split(':', 1)[1].strip()
        elif l.startswith('TRAP-'):
            h['trap'][0 if l.startswith('TRAP-PASS') else 1] += 1
        elif l.startswith('NVIDIA') or l.startswith('NVIDIA ') or re.match(r'^(NVIDIA|Tesla|A100)', l):
            h['gpu'] = h['gpu'] or ', '.join(l.strip().split(', ')[:2])
        elif l.startswith('# box:') and ' DONE ' in l:
            h['box'] = l.split('DONE', 1)[1].strip()
        elif l.startswith('# box:') and 'try ' in l:
            h.setdefault('tries', []).append(l)
    if h.get('tries'):
        m = re.search(r'dph=(\S+) pcie=(\S+) cpu_ghz=(\S+) host=(\S+) machine=(\S+) geo=(.*)', h['tries'][-1])
        if m:
            h['box'] = f"host {m[4]} machine {m[5]}, {m[6].strip()}, ${m[1]}/h, PCIe {m[2]} GB/s"
    return h


have = [(p, label, os.path.join(D, f'base-{p}.txt')) for p, label in PARTS if os.path.exists(os.path.join(D, f'base-{p}.txt'))]
K = {p: kernels(f) for p, _, f in have}
print("# GPU baseline: five architectures, multi-computation probes, library chains\n")
print(os.environ.get("BASELINE_HEAD", "") + "\n")
print("Sessions: `base-<part>.txt` = the base build (the S53 table; Ampere every rung ON), every arm, the probes "
      "and the library chains; `-confirm.txt` = the same build's kernels, decode and e2e on a different box; "
      "`-verify*.txt` = the S54 table's kernels (after the rows below), next to the base build's. `-prof.txt` = "
      "ptxas, SASS and nsys evidence. Medians; `~` = the min–max ranges overlap. Per-part detail: "
      "`python3 benches/gpu/table.py base-<part>.txt`; rung decisions: `python3 benches/gpu/rungs.py <sessions>`.\n")
print("## Boxes and correctness\n")
print("| part | GPU, driver | box | differential (GPU vs interpreter oracle, every arm, both faces) | trap-before-site |")
print("|---|---|---|---|---|")
for p, label, f in have:
    h = header(f)
    print(f"| {label} | {h['gpu']} | {h['box']} | {h['diff']} | {h['trap'][0]} pass / {h['trap'][1]} fail |")

ov = lambda a, b: '' if (a[2] < b[0] or b[2] < a[0]) else '~'
for face, title in (('_exact', 'exact (conformance face, bit-identical to the interpreter)'),
                    ('_c', 'contract (one fma per step)'), ('_tf', 'TF32 (tensor cores; bound-checked)')):
    print(f"\n## Kernels, {title}: cuBLAS ms / Mapal ms (>1 = Mapal faster), CUTLASS ms / Mapal ms\n")
    print("| n | " + " | ".join(label for p, label, _ in have) + " |")
    print("|---:|" + "---|" * len(have))
    for n in (1024, 1536, 2048, 3072, 4096, 8192):
        cells = []
        for p, _, _ in have:
            r = K[p].get((n, face))
            if not r:
                cells.append('')
                continue
            m, c, t = r['mapal'], r['cublas'], r['cutlass']
            cells.append(f"{c[1] / m[1]:.2f}{ov(m, c)} / {t[1] / m[1]:.2f}{ov(m, t)} ({m[1]:.3f} ms)")
        print(f"| {n} | " + " | ".join(cells) + " |")

print("\n## The S54 table on the box: new shipped kernel vs the base build's, one run\n")
print("`base-<part>-verify*.txt`: the kernel the S54 table ships at that session next to the base build's "
      "(`s54base`) and the earlier S54 tables' (`v1`, `v2`), interleaved in one run; ms medians, the same bench's "
      "cuBLAS ms in brackets; shipped/x = (x ms / its cuBLAS) / (shipped ms / its cuBLAS), >1 = the shipped table "
      "faster.\n")
print("| part | session | face | n | shipped ms (cuBLAS) | v1 ms (cuBLAS) | v2 ms (cuBLAS) | base ms (cuBLAS) | shipped/v1 | shipped/v2 | shipped/base |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for p, label, f in have:
    for tag in ('verify', 'verify2', 'verify3'):
        vf = os.path.join(D, f'base-{p}-{tag}.txt')
        if not os.path.exists(vf):
            continue
        R = kernels(vf)
        for face, fl in (('_exact', 'exact'), ('_c', 'fma'), ('_tf', 'tf32')):
            for n in (1024, 2048, 4096, 8192):
                new, old, v1, v2 = (R.get((n, face + x)) for x in ('', '.s54base', '.v1', '.v2'))
                if not new or not old:
                    continue
                r = lambda a, b: (a['mapal'][1] / a['cublas'][1]) / (b['mapal'][1] / b['cublas'][1])
                c = lambda x: f"{x['mapal'][1]:.3f} ({x['cublas'][1]:.3f})" if x else ''
                print(f"| {label} | {tag} | {fl} | {n} | {c(new)} | {c(v1)} | {c(v2)} | {c(old)} | "
                      f"{f'{r(v1, new):.3f}' if v1 else ''} | {f'{r(v2, new):.3f}' if v2 else ''} | {r(old, new):.3f} |")

print("\n## Multi-computation probes: Mapal vs the library (exact face, shipped emission)\n")
print("Mapal `iter` = the program's own clock around all its computations (transfers, host loops, syncs included); "
      "`kernel` = MAPAL_GPU_TRACE device time; library legs = cudaEvent GPU time on device-built data "
      "(chain_bench.cu, every leg gated against Mapal's values). eager = cuBLAS + one kernel per step; "
      "best fused = the fastest of cuBLASLt epilogues / CUTLASS epilogues / streams / gemv / one fused kernel.\n")
for p, label, f in have:
    runs, legs, gates = probes(f)
    if not runs:
        continue
    print(f"\n### {label}\n")
    print("| probe | n | Mapal iter ms exact / contract | Mapal kernel ms exact / contract | Mapal process wall ms | d2h + h2d ms | launches | bytes in / out "
          "| eager ms | best fused (leg) ms | eager / Mapal iter | best / Mapal iter | best / Mapal kernel exact / contract | gates |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for (prog, fc, arm), rs in sorted(runs.items(), key=lambda kv: (kv[0][0].rsplit('_', 1)[0], int(kv[0][0].rsplit('_', 1)[1]))):
        if fc != '_exact' or arm != 'on':
            continue
        probe, n = prog[3:].rsplit('_', 1)
        c = runs.get((prog, '_c', 'on'), [])
        it, itc = med([r['iter'] for r in rs]), med([r['iter'] for r in c] or [float('nan')])
        k = med([r['kernel'] for r in rs])
        kc = med([r['kernel'] for r in c] or [float('nan')])
        lg = legs.get((probe, n), {})
        eager = lg.get('eager')
        fused = {k2: v for k2, v in lg.items() if k2 != 'eager'}
        best = min(fused.items(), key=lambda kv: kv[1][0]) if fused else None
        b = min(lg.values(), key=lambda v: v[0])[0] if lg else None
        print(f"| {probe} | {n} | {it:.3f} / {itc:.3f} | {k:.3f} / {kc:.3f} | {med([r['wall'] for r in rs]):.0f} | {med([r['d2h'] + r['h2d'] for r in rs]):.2f} "
              f"| {rs[0]['launches']} | {rs[0]['in']} / {rs[0]['out']} "
              f"| {f'{eager[0]:.3f} ({eager[1]})' if eager else ''} "
              f"| {f'{best[1][0]:.3f} {best[0]} ({best[1][1]})' if best else ''} "
              f"| {f'{eager[0] / it:.3f}' if eager else ''} | {f'{b / it:.3f}' if b else ''} "
              f"| {f'{b / k:.2f} / {b / kc:.2f}' if b and k == k and k > 0 and kc > 0 else ''} "
              f"| {' '.join(gates.get((probe, n), [])) or ('pass' if lg else 'no library run')} |")

print("\n## Host rungs on the probes: arm iter ms / shipped iter ms (>1 = the rung pays), exact face\n")
print("| probe | n | " + " | ".join(f"{label} off-resident / off-devgen" for p, label, _ in have) + " |")
print("|---|---:|" + "---|" * len(have))
allp = {p: probes(f)[0] for p, _, f in have}
progs = sorted({k[0] for r in allp.values() for k in r}, key=lambda s: (s.rsplit('_', 1)[0], int(s.rsplit('_', 1)[1])))
for prog in progs:
    probe, n = prog[3:].rsplit('_', 1)
    cells = []
    for p, _, _ in have:
        r = allp[p]
        on = r.get((prog, '_exact', 'on'))
        xs = []
        for arm in ('off-resident', 'off-devgen'):
            a = r.get((prog, '_exact', arm))
            xs.append(f"{med([x['iter'] for x in a]) / med([x['iter'] for x in on]):.2f}" if a and on else '=')
        cells.append(' / '.join(xs))
    print(f"| {probe} | {n} | " + " | ".join(cells) + " |")

print("\n## Kernel rungs, ON vs OFF per part\n")
print("Speedup = (OFF ms / that bench's cuBLAS ms) / (ON ms / that bench's cuBLAS ms), medians (the cuBLAS leg cancels "
      "a power-capped box's clock drift); one number per session (s1 = `base-<part>.txt`, s2 = `-confirm.txt`, a "
      "different box); `*` = disjoint raw min–max ranges. fill: n=1024 under-filled, n>=2048 filled; face: `_exact` "
      "Exact, `_c` and `_tf` Contract. Rule: a rung ships ON in a class iff some size is > 1.01 in every session "
      "(±1% = no effect: one ratio moves a median 0.5–1.2% between two boxes). status: ok / CONTRADICTS (a size "
      "the other way in every session, disjoint in every session) / contradicts (the rule disagrees without that). "
      "`arch-rungs` (the whole table) and `fma-regs` (idx32+affine+pad+wide-fma) are composite arms.\n")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rungs import classes
for p, label, f in have:
    files = [f] + [x for x in [os.path.join(D, f'base-{p}-confirm.txt')] if os.path.exists(x)]
    out, ns = classes(files)
    print(f"\n### {label} ({ns} session{'s' if ns > 1 else ''})\n")
    print("| rung | fill | face | shipped (base build) | rule | status | speedup per size: s1 s2 |")
    print("|---|---|---|---|---|---|---|")
    for (rung, fill, face), sh, v, st, cells, _ in out:
        print(f"| {rung} | {fill} | {face} | {sh} | {v} | {st} | {cells} |")

for p, label, f in have:
    table = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'table.py')
    out = subprocess.run([sys.executable, table, f], capture_output=True, text=True).stdout
    # the whole-program and decode tables only (the kernels and probes are above)
    keep, blocks = [], out.split('\n\n')
    for b in blocks:
        if b.startswith('| program |') or b.startswith('| decode |') or b.startswith('hand-written'):
            keep.append(b)
    if keep:
        print(f"\n## Whole programs and decode, {label}\n")
        print('\n\n'.join(keep))
