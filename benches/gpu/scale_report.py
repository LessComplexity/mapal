#!/usr/bin/env python3
"""per-site-choice scale ladders: how Mapal fares against every competitor leg as each task grows.

  scale_report.py report <ladder.tsv> <results-<part>.txt>... [--bundle <dir>] [--out <dir>]
      per task family and part a table: size, work units, Mapal AUTO ms (the tuned pick), RULES ms (candidate 0),
      each competitor's ms, Mapal / best other (< 1: Mapal faster), null (|null - AUTO| / AUTO), the gate, and the
      crossover size where the ratio passes 1.0. Writes <out>/<part>.tsv + .md (tables only) and <out>/combined.tsv
      (task, part, size, units, unit, ratio) for charting; a step the bundle skipped (`<part>/ladder-mem.tsv`) or a
      leg that failed (torch OOM, a gate) is a row / cell saying so. Default out: benches/results/evidence/scale.
  scale_report.py mem <part dir> <part> [--drop]
      predicted device bytes per ladder stem (the host IR's `mapal_gpu_arena` + constant `mapal_gpu_alloc` sizes)
      against the part's memory -> <part dir>/ladder-mem.tsv; --drop removes the files of a step that does not fit
      (90 % of the memory), so box_run.sh never sees it.
  scale_report.py estimate <ladder.tsv> [--bundle <dir>]
      box minutes per part: Mapal arms (gates, tune run, ABBA reps, traces) + library and torch legs.

Times: a whole program's `iter ms` (its own clock, `probe` lines: median over the reps), chain_bench / torch legs'
med_ms (`chain`, `torch` lines), a cell's tile_bench med_ms (median over the two passes). A cell's AUTO is what
`mapal_gpu_site` decides from its candidates' minima (`.pick<i>` legs): the argmin unless candidate 0 is within
max(3 %, 1 us)."""
import glob
import os
import re
import statistics
import sys
from collections import defaultdict

MEM_GB = {'rtx-4090': 24, 'rtx-5090': 32, 'rtx-3090': 24, 'h100-pcie': 80}
# Achieved f32 FMA rate (TFLOP/s) and DRAM bandwidth (GB/s) per part: the estimate's only inputs.
RATE = {'rtx-4090': (50, 1000), 'rtx-5090': (70, 1800), 'rtx-3090': (25, 900), 'h100-pcie': (35, 2000)}
med = statistics.median


def ladder(path):
    rows = [dict(zip(('stem', 'family', 'size', 'units', 'unit', 'gate'), l.rstrip('\n').split('\t')))
            for l in open(path) if not l.startswith('stem\t')]
    for r in rows:
        r['size'], r['units'] = int(r['size']), int(r['units'])
    return rows


def kv(line):
    return dict(re.findall(r'(\w+)=([^\s\[\]]+)', line))


def cell_leg(f):
    """tile_bench's leg name in a cell: `c0` (the shipped default face), `pick<i>`, `null`, `x`, `c`, … or a library."""
    leg, cell = f['leg'], f['cell']
    return leg if not leg.startswith(cell) else 'c0' if leg == cell else leg[len(cell) + 1:]


def parse(path):
    """stem -> {leg: [ms...]}, stem -> {leg: note} (a failed leg / gate)."""
    t, note = defaultdict(lambda: defaultdict(list)), defaultdict(dict)
    for line in open(path, errors='replace'):
        if line.startswith('probe prog=') and 'face=_default' in line:
            f, m = kv(line), re.search(r'iter ms=\s*([-0-9.e+]+)', line)
            if m:
                t[f['prog']][f['arm']].append(float(m.group(1)))
        elif line.startswith('chain probe='):
            f = kv(line)
            stem = f"pt_decode_{f['T']}_{f['d']}" if f['probe'] == 'decode' else f"pt_{f['probe']}_{f['n']}"
            ms = f.get('med_ms', f.get('wall_med_ms'))
            if ms:
                t[stem]['lib-' + f['leg']].append(float(ms))
        elif line.startswith('chain_values probe='):
            f = kv(line)
            stem = f"pt_decode_{f['T']}_{f['d']}" if f['probe'] == 'decode' else f"pt_{f['probe']}_{f['n']}"
            if f.get('gate') not in ('ok', 'yes', 'pass', None):
                note[stem]['lib-' + f['leg']] = f"gate {f['gate']}"
        elif line.startswith('torch prog='):
            f = kv(line)
            leg = 'torch-' + f.get('mode', '?')
            if 'med_ms' in f:
                t[f['prog']][leg].append(float(f['med_ms']))
                if f.get('within') == 'no':
                    note[f['prog']][leg] = 'values off'
            else:  # FAILED <exception> / NOWORKLOAD: the leg cannot run there
                m = re.search(r'(FAILED|NOWORKLOAD) (\w+)', line)
                note[f['prog']][leg if 'mode' in f else 'torch'] = ' '.join(m.groups()) if m else 'failed'
        elif ' cell=' in line and ' time leg=' in line:
            f = kv(line)
            leg = cell_leg(f)
            t[f['cell']][leg].append(float(f['med_ms']))
            t[f['cell']]['min:' + leg].append(float(f['min_ms']))
        elif ' cell=' in line and line.split(' ', 2)[2].startswith('values leg=') and 'GATE-FAILED' in line:
            f = kv(line)
            note[f['cell']][cell_leg(f)] = 'GATE-FAILED'
    return t, note


def site_pick(mins):
    """mapal_choice.h's decision over the candidates' minima: candidate 0 unless beaten by > max(3 %, 1 us)."""
    c0 = mins['c0']
    best = min(mins, key=mins.get)
    return best if c0 - mins[best] > max(0.03 * c0, 1e-3) else 'c0'


def row(step, t, note):
    """One ladder step: {auto, rules, null, legs{name: ms}, notes}."""
    s = t.get(step['stem'], {})
    if step['gate'] == 'cell':
        bad = note.get(step['stem'], {})  # a candidate that failed its gate is never picked
        mins = {k[4:]: min(v) for k, v in s.items()
                if k.startswith('min:') and (k[4:] == 'c0' or k[4:].startswith('pick')) and k[4:] not in bad}
        if 'c0' not in mins:
            return None
        pick = site_pick(mins)
        auto, rules = med(s[pick]), med(s['c0'])
        legs = {k: med(v) for k, v in s.items() if k in ('cublas', 'cutlass', 'cutlass_tuned')}
        null = abs(med(s['null']) / rules - 1) if s.get('null') else None
    else:
        if not s.get('on'):
            return None
        auto = med(s['on'])
        rules = med(s['rules']) if s.get('rules') else auto  # no site table: one build
        legs = {k: med(v) for k, v in s.items() if k.startswith(('lib-', 'torch-'))}
        null = abs(med(s['null']) / auto - 1) if s.get('null') else None
        if s.get('cpu'):
            legs['mapal-cpu'] = med(s['cpu'])
    notes = note.get(step['stem'], {})
    ok = {k: v for k, v in legs.items() if k not in notes and k != 'mapal-cpu'}
    best = min(ok.values()) if ok else None
    return dict(auto=auto, rules=rules, null=null, legs=legs, notes=notes, ratio=auto / best if best else None)


def crossover(rows):
    """The first size whose Mapal / best-other ratio is on the other side of 1.0 from the step before it."""
    r = [(s['size'], x['ratio']) for s, x in rows if x and x['ratio']]
    for (a, ra), (b, rb) in zip(r, r[1:]):
        if (ra < 1) != (rb < 1):
            return f"{b} ({ra:.2f} -> {rb:.2f})"
    return 'none' if r else 'n/a'


def mem_skips(bundle, part):
    """stem -> the reason the bundle skipped it (`ladder-mem.tsv`)."""
    p = os.path.join(bundle or '', part, 'ladder-mem.tsv')
    rows = [l.rstrip('\n').split('\t') for l in open(p)] if bundle and os.path.exists(p) else []
    return {r[0]: r[-1] for r in rows if r[-1].startswith('skip')}


def report(argv):
    lad, files, opt, it = ladder(argv[0]), [], {}, iter(argv[1:])
    for a in it:
        if a.startswith('--'):
            opt[a] = next(it)
        else:
            files.append(a)
    out = opt.get('--out', 'benches/results/evidence/scale')
    os.makedirs(out, exist_ok=True)
    comb = ['task\tpart\tsize\tunits\tunit\tratio']
    fams = list(dict.fromkeys(s['family'] for s in lad))
    for path in files:
        part = re.sub(r'^results-|\.txt$', '', os.path.basename(path))
        t, note = parse(path)
        skips = mem_skips(opt.get('--bundle'), part)
        md, tsv = [f'# scale ladders — {part}', ''], ['task\tsize\tunits\tunit\tgate\tauto_ms\trules_ms\tlegs\tratio\tnull\tnote']
        for fam in fams:
            steps = sorted((s for s in lad if s['family'] == fam), key=lambda s: s['size'])
            rows = [(s, row(s, t, note)) for s in steps]
            names = sorted({k for _, x in rows if x for k in x['legs']} | {k for _, x in rows if x for k in x['notes']})
            md += [f'## {fam} (crossover: {crossover(rows)})', '',
                   '| size | units | gate | AUTO ms | RULES ms | ' + ' | '.join(names) + ' | Mapal / best other | null |',
                   '|' + ' --- |' * (7 + len(names))]
            for s, x in rows:
                if x is None:
                    why = skips.get(s['stem'], 'not run (no timing in this results file)')
                    md.append(f"| {s['size']} | {s['units']:.3g} | {s['gate']} | {why} |" + ' |' * (4 + len(names)))
                    tsv.append(f"{fam}\t{s['size']}\t{s['units']}\t{s['unit']}\t{s['gate']}\t\t\t\t\t\t{why}")
                    continue
                cell = lambda k: (f"{x['legs'][k]:.4g}" if k in x['legs'] else '') + (f" ({x['notes'][k]})" if k in x['notes'] else '')
                ratio = f"{x['ratio']:.3f}" if x['ratio'] else 'n/a'
                null = f"{x['null']:.3f}" if x['null'] is not None else 'n/a'
                md.append(f"| {s['size']} | {s['units']:.3g} | {s['gate']} | {x['auto']:.4g} | {x['rules']:.4g} | "
                          + ' | '.join(cell(k) for k in names) + f' | {ratio} | {null} |')
                tsv.append(f"{fam}\t{s['size']}\t{s['units']}\t{s['unit']}\t{s['gate']}\t{x['auto']:.6g}\t{x['rules']:.6g}\t"
                           + ','.join(f'{k}={x["legs"][k]:.6g}' for k in sorted(x['legs'])) + f'\t{ratio}\t{null}\t'
                           + ','.join(f'{k}:{v}' for k, v in sorted(x['notes'].items())))
                if x['ratio']:
                    comb.append(f"{fam}\t{part}\t{s['size']}\t{s['units']}\t{s['unit']}\t{x['ratio']:.4f}")
            md.append('')
        open(os.path.join(out, f'{part}.md'), 'w').write('\n'.join(md))
        open(os.path.join(out, f'{part}.tsv'), 'w').write('\n'.join(tsv) + '\n')
    open(os.path.join(out, 'combined.tsv'), 'w').write('\n'.join(comb) + '\n')
    print(f'scale: {len(files)} part(s), {len(comb) - 1} ratios -> {out}')


def device_bytes(host_ll):
    """The device bytes a host object asks for up front: its arenas and constant-size allocations."""
    return sum(int(n) for n in re.findall(r'@mapal_gpu_(?:arena|alloc)\(ptr [^,]+, i64 (\d+)\)', host_ll))


def mem(argv):
    d, part, drop = argv[0], argv[1], '--drop' in argv
    lad = ladder(os.path.join(d, '..', 'tiles', 'ladder.tsv'))
    cap = MEM_GB[part] << 30
    out = ['stem\tdevice_bytes\tpart_bytes\ttables\tverdict']
    for stem in dict.fromkeys(s['stem'] for s in lad):
        ll = os.path.join(d, stem + '.host.ll')
        if not os.path.exists(ll):
            out.append(f'{stem}\t\t{cap}\t\tskip: no emission (compile failed?)')
            continue
        b = device_bytes(open(ll).read())
        dev = os.path.join(d, stem + '.dev.ll')
        tables = len(re.findall(r'^; mapal\.site ', open(dev).read(), re.M)) if os.path.exists(dev) else 0
        fits = b <= 0.9 * cap
        out.append(f'{stem}\t{b}\t{cap}\t{tables}\t' + ('fits' if fits else f'skip: {b / 2**30:.1f} GiB > 90 % of {MEM_GB[part]} GB'))
        if not fits and drop:
            for f in glob.glob(os.path.join(d, stem + '.*')) + [g for x in ('_x', '_c', '_r', '_rc')
                                                                 for g in glob.glob(os.path.join(d, stem + x + '.*'))]:
                os.remove(f)
    open(os.path.join(d, 'ladder-mem.tsv'), 'w').write('\n'.join(out) + '\n')
    print(f"mem {part}: {sum('skip' in l for l in out)} skipped, {sum(l.endswith('fits') for l in out)} fit")


def estimate(argv):
    """Box seconds per step: Mapal (5 value gates + 1 tune + PROBE_REPS 5 x arms on/rules/null x (1 + 2/5 traced) at
    0.4 s process start + 3 x ms each), chain_bench (3 legs x 38 runs), torch (2-4 modes x (40 s compile + 36 runs));
    a cell: tile_bench --splitk, 2 processes x ~12 legs x 33 runs + 6 whole programs."""
    lad = ladder(argv[0])
    rows = ['| part | steps | Mapal min | library min | torch min | total min |', '| --- | --- | --- | --- | --- | --- |']
    for part, (tf, bw) in RATE.items():
        mp = lib = tor = 0.0
        for s in {x['stem']: x for x in lad}.values():
            ms = (s['units'] * 4 * 5 / bw / 1e6 if s['unit'] == 'elem' else s['units'] / tf / 1e9) + 0.05
            if s['family'].startswith(('decode', 'prefill', 'gen')):
                ms += s['size'] * 0.02 if 'context' in s['family'] else 1024 * 0.02  # per-trip launches
            if s['gate'] == 'cell':
                lib += 2 * 12 * 33 * ms / 1e3 + 6 * (0.4 + 3 * ms / 1e3)
                continue
            proc = 0.4 + 3 * ms / 1e3
            mp += (5 + 1 + 5 * 3 * 1.4 + (1 if s['gate'] == 'xref' else 0)) * proc
            lib += 3 * 38 * ms / 1e3 + 2
            modes = 4 if ('tf32' in s['stem'] or 'convmc' in s['stem'] or '_mlp' in s['stem']) else 2
            tor += modes / 2 * 40 + modes * 36 * ms / 1e3
        rows.append(f"| {part} | {len({x['stem'] for x in lad})} | {mp / 60:.0f} | {lib / 60:.0f} | {tor / 60:.0f} | "
                    f"{(mp + lib + tor) / 60:.0f} |")
    print('\n'.join(rows))


def demo():
    """Self-check on a synthetic results file (two families, a skipped and a failed leg)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        L = os.path.join(tmp, 'ladder.tsv')
        open(L, 'w').write('stem\tfamily\tsize\tunits\tunit\tgate\n'
                           'pt_mlp_512\tchain-mlp\t512\t1\tflop\toracle\npt_mlp_1024\tchain-mlp\t1024\t8\tflop\txref\n'
                           'pt_mlp_2048\tchain-mlp\t2048\t64\tflop\txref\n'
                           'mm_256x256x256\tgemm-square-cell\t256\t1\tflop\tcell\n')
        R = os.path.join(tmp, 'results-rtx-4090.txt')
        lines = []
        for n, on, lib in ((512, 2.0, 1.0), (1024, 4.0, 5.0)):
            for r, x in enumerate((on, on * 1.01, on * 0.99)):
                lines.append(f'probe prog=pt_mlp_{n} face=_default arm=on rep={r} wall_us=1 out=[1 2 iter ms={x} ] rc=0')
                lines.append(f'probe prog=pt_mlp_{n} face=_default arm=rules rep={r} wall_us=1 out=[1 2 iter ms={x * 1.1} ] rc=0')
                lines.append(f'probe prog=pt_mlp_{n} face=_default arm=null rep={r} wall_us=1 out=[1 2 iter ms={x * 1.02} ] rc=0')
            lines.append(f'chain probe=mlp n={n} leg=eager launches=3 reps=30 min_ms={lib} med_ms={lib} max_ms={lib}')
            lines.append(f'torch prog=pt_mlp_{n} mode=eager n=30 min_ms=9 med_ms=9 max_ms=9 within=yes maxdev=0 compile_s=0')
        lines.append('torch prog=pt_mlp_1024 mode=compile FAILED OutOfMemoryError: CUDA out of memory')
        for p in (1, 2):
            for leg, ms in (('mm_256x256x256', 1.0), ('mm_256x256x256.pick1', 0.5), ('mm_256x256x256.pick2', 0.99),
                            ('cublas', 0.8), ('cutlass', 0.9)):
                lines.append(f'pass={p} cell=mm_256x256x256 time leg={leg} M=256 N=256 K=256 n=30 min_ms={ms} med_ms={ms} max_ms={ms}')
        open(R, 'w').write('\n'.join(lines) + '\n')
        t, note = parse(R)
        lad = ladder(L)
        x = row(lad[0], t, note)
        assert abs(x['auto'] - 2.0) < 1e-9 and abs(x['rules'] - 2.2) < 1e-9 and abs(x['ratio'] - 2.0) < 1e-9, x
        assert abs(x['null'] - 0.02) < 1e-9, x
        y = row(lad[1], t, note)
        assert abs(y['ratio'] - 0.8) < 1e-9 and y['notes']['torch-compile'] == 'FAILED OutOfMemoryError', y
        c = row(lad[3], t, note)
        assert abs(c['auto'] - 0.5) < 1e-9 and c['rules'] == 1.0 and abs(c['ratio'] - 0.625) < 1e-9, c
        assert site_pick({'c0': 1.0, 'pick1': 0.98}) == 'c0'  # within 3 %: candidate 0 stays
        assert crossover([(lad[0], x), (lad[1], y)]).startswith('1024'), crossover([(lad[0], x), (lad[1], y)])
        report([L, R, '--out', os.path.join(tmp, 'out')])
        md = open(os.path.join(tmp, 'out', 'rtx-4090.md')).read()
        assert 'crossover: 1024' in md and 'not run' in md and 'OutOfMemoryError' in md, md
        comb = open(os.path.join(tmp, 'out', 'combined.tsv')).read().splitlines()
        assert comb[0] == 'task\tpart\tsize\tunits\tunit\tratio' and len(comb) == 4, comb
    print('scale_report demo ok')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'demo'
    {'report': report, 'mem': mem, 'estimate': estimate, 'demo': lambda _: demo()}[cmd](sys.argv[2:])
