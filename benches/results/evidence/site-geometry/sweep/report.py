#!/usr/bin/env python3
"""site-geometry sweep report: report.py <out.md>"""
import re, statistics, sys, os, math
from collections import defaultdict

REPO = '/Volumes/LessComplex/Personal/mapal/.claude/worktrees/wf_6788c915-fd1-6'
SW = f'{REPO}/benches/results/evidence/site-geometry/sweep'
ARCHS = ['rtx-4090', 'rtx-5090', 'h100-pcie', 'rtx-3090']
med = statistics.median
gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))

PROBE = re.compile(r'^probe prog=(\S+) face=(\S+) arm=(\S+) rep=\d+ wall_us=(\d+) out=\[(.*?)iter ms=\s*([-\d.eE+]+)')
TORCH = re.compile(r'^torch prog=(\S+) mode=(\S+) n=\d+ min_ms=\S+ med_ms=([\d.]+) .*within=(\w+)')
CH2 = re.compile(r'^chain probe=(\S+) ((?:\w+=\S+ )*?)leg=(\S+) .*wall_med_ms=([\d.]+)')
CELL = re.compile(r'^pass=(\d+) cell=(\S+) time leg=(\S+) .*med_ms=([\d.]+)')
GATE = re.compile(r'^differential: (\d+) pass, (\d+) fail')

probe = defaultdict(list); torch = {}; chain = defaultdict(dict)
cell = defaultdict(lambda: defaultdict(list))   # (arch, cell) -> leg -> [med per pass]
gate = {}
for a in ARCHS:
    p = f'{SW}/{a}/results-{a}.txt'
    if not os.path.exists(p):
        continue
    for l in open(p, errors='replace'):
        m = PROBE.match(l)
        if m:
            probe[(a, m[1], m[2], m[3])].append(float(m[6])); continue
        m = TORCH.match(l)
        if m:
            torch[(a, m[1], m[2])] = float(m[3]); continue
        m = CH2.match(l)
        if m:
            nums = '_'.join(x.split('=')[1] for x in m[2].split())
            chain[(a, f'pt_{m[1]}_{nums}')][m[3]] = float(m[4]); continue
        m = CELL.match(l)
        if m:
            cell[(a, m[2])][m[3]].append(float(m[4])); continue
        m = GATE.match(l)
        if m:
            gate[a] = (int(m[1]), int(m[2]))

import gzip, glob
carried_t = {}; carried_c = defaultdict(dict)
for a in ARCHS:
    for f in glob.glob(f'{REPO}/benches/results/evidence/recurrence-precision/sweep-{a}/results*.txt.gz'):
        for l in gzip.open(f, 'rt', errors='replace'):
            m = TORCH.match(l)
            if m:
                carried_t[(a, m[1], m[2])] = float(m[3]); continue
            m = CH2.match(l)
            if m:
                nums = '_'.join(x.split('=')[1] for x in m[2].split())
                carried_c[(a, f'pt_{m[1]}_{nums}')][m[3]] = float(m[4])

pm = lambda a, prog, face, arm: med(probe[(a, prog, face, arm)]) if probe.get((a, prog, face, arm)) else None
cm = lambda a, c, leg: med(cell[(a, c)][leg]) if cell[(a, c)].get(leg) else None

def best_other(a, prog):
    def gather(ch, tt):
        c = dict(ch.get((a, prog), {}))
        tf = 'tf32' in prog
        for (aa, p, mode), v in tt.items():
            if aa == a and p == prog and (tf or not mode.endswith('-tf32')):
                c['torch' if mode == 'eager' else 'torch-' + mode] = v
        return c
    c = gather(chain, torch); tag = ''
    if not any(k.startswith('torch') for k in c):     # torch legs missing in this run: the earlier sweep's (competitor unchanged)
        c2 = gather(carried_c, carried_t)
        c2.update(c); c = c2; tag = '*' if any(k.startswith('torch') for k in c) else ''
    if not c:
        return None, None
    k = min(c, key=c.get)
    return c[k], k + tag

prev = {}
for f in (f'{REPO}/benches/results/evidence/cpu-gpu-boundary/table.md',
          f'{REPO}/benches/results/evidence/recurrence-precision/report.md'):
    for l in open(f):
        c = [x.strip() for x in l.strip().strip('|').split('|')]
        if len(c) >= 6 and c[1] in ARCHS and re.match(r'^[\d.]+$', c[4]):
            prev[(c[0], c[1])] = c[4]

out = []; w = out.append
progs = sorted({k[1] for k in probe})
cells = sorted({k[1] for k in cell})

w('### Workload x arch (default face; Mapal = median iter ms of the shipped build; cells: tile_bench ms, passes median)\n')
w('(* = competitor leg carried from the recurrence-precision sweep: not measured on this box in this run)\n')
w('| workload | arch | Mapal ms | best other (leg) | Mapal/best | previous Mapal/best | measured / carried |')
w('|---|---|---|---|---|---|---|')
for prog in progs:
    wl = prog[3:]
    for a in ARCHS:
        m = pm(a, prog, '_default', 'on')
        if m is None:
            continue
        b, leg = best_other(a, prog)
        w(f'| {wl} | {a} | {m:.4g} | {f"{b:.4g} ({leg})" if b else "no library/torch leg"} | '
          f'{f"{m/b:.2f}" if b else "-"} | {prev.get((wl, a), "new")} | measured |')
for c in cells:
    for a in ARCHS:
        m = cm(a, c, c)
        if m is None:
            continue
        lib = {k: cm(a, c, k) for k in ('cublas', 'cutlass', 'cutlass_tuned') if cm(a, c, k)}
        k = min(lib, key=lib.get) if lib else None
        w(f'| {c} | {a} | {m:.4g} | {f"{lib[k]:.4g} ({k})" if k else "-"} | {f"{m/lib[k]:.2f}" if k else "-"} | n/a (no earlier cell row) | measured |')

# ON/OFF per rung: OFF time / ON time (>1 = ON faster)
w('\n### ON/OFF per rung (OFF/ON geomean over programs; W/L = ON faster/slower beyond max(box null, 2%); null = median per-program |null/on - 1|; W/L use each own null)\n')
w('| rung | arch | kind | n | geomean OFF/ON | wins | losses | max null | worst program (OFF/ON) |')
w('|---|---|---|---|---|---|---|---|---|')
R = defaultdict(list)
NL = defaultdict(float)
for (a, prog, face, arm) in list(probe):
    if face != '_default' or not (arm.startswith('off-') or arm.startswith('on-')) or arm.startswith('on-g-'):
        continue
    on, nl = pm(a, prog, face, 'on'), pm(a, prog, face, 'null')
    x = pm(a, prog, face, arm)
    if not on or not x:
        continue
    if arm.startswith('off-'):
        R[(arm[4:], a, 'probe')].append((prog[3:], x / on, abs(nl / on - 1) if nl else 0.0))
    else:
        R[('ON:' + arm[3:], a, 'probe')].append((prog[3:], on / x, abs(nl / on - 1) if nl else 0.0))
    if nl:
        NL[(a, 'probe')] = max(NL[(a, 'probe')], abs(nl / on - 1))
for (a, c) in list(cell):
    base = cm(a, c, c); nl = cm(a, c, c + '.null')
    if not base:
        continue
    if nl:
        NL[(a, 'cell')] = max(NL[(a, 'cell')], abs(nl / base - 1))
    for leg in cell[(a, c)]:
        if leg.startswith(c + '.off-'):
            R[(leg[len(c) + 5:], a, 'cell')].append((c, cm(a, c, leg) / base, abs(nl / base - 1) if nl else 0.0))
        elif leg.startswith(c + '.on-') and not leg.startswith(c + '.on-g-'):
            R[('ON:' + leg[len(c) + 4:], a, 'cell')].append((c, base / cm(a, c, leg), abs(nl / base - 1) if nl else 0.0))
for (rung, a, kind) in sorted(R):
    v = R[(rung, a, kind)]
    nl = med([n for _, _, n in v])
    wins = sum(1 for _, r, n in v if r > 1 + max(n, .02)); loss = sum(1 for _, r, n in v if r < 1 - max(n, .02))
    worst = min(v, key=lambda t_: t_[1])
    w(f'| {rung} | {a} | {kind} | {len(v)} | {gm([r for _, r, _ in v]):.3f} | {wins} | {loss} | {nl:.3f} | {worst[0]} {worst[1]:.2f} |')

w('\n### Per-program OFF/ON for the new rungs\n')
w('| program | rung | ' + ' | '.join(ARCHS) + ' |')
w('|---|---|' + '---|' * len(ARCHS))
for rung in ('fill-wave', 'fit-first', 'row-split', 'tf32-radix', 'ON:tf32-radix', 'ON:row-fit'):
    names = sorted({n for (r, a, k), v in R.items() if r == rung for n, _, _ in v})
    for n in names:
        cs = []
        for a in ARCHS:
            v = [r for k in ('probe', 'cell') for nn, r, _ in R.get((rung, a, k), []) if nn == n]
            cs.append(f'{v[0]:.2f}' if v else '=')
        w(f'| {n} | {rung} | ' + ' | '.join(cs) + ' |')

w('\n### Value gates (differential pass/fail)\n')
for a in ARCHS:
    if a in gate:
        w(f'- {a}: {gate[a][0]} pass, {gate[a][1]} fail')
open(sys.argv[1], 'w').write('\n'.join(out) + '\n')
print(len(out), 'lines')
