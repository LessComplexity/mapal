#!/usr/bin/env python3
"""site-geometry A3 — ADR-0034 D2: one generic search driver over the per-target knob table.

The table is benches/gpu/knobs.json, `{"<arch>": {"<knob>": [values..]}}`, the shipped value first; the emitter
takes a knob as `--geom=<knob>=<value>`. Nothing here knows a knob, a site kind or an arch.

plan   — per program and face, each knob's other values one at a time from the shipped table (coordinate search).
         A candidate's emission is hashed: equal to the shipped build => the knob reaches no site of the program
         (`drop`), equal to an earlier candidate's => the same arm (`alias`), its (arm, shipped) verdict already in
         the arch's result cache => `cached`, else `build`.
           search.py plan <out> <knobs.json> <arch> <program>.. [--faces default ..]
           -> <out>/plan.tsv (program face knob value hash arm status); prints the ARMS entries (integrated_bundle.sh)
bundle — plan, then integrated_bundle.sh: the programs (SUITE / CELLS) + the `build` arms, then preflight.py.
           search.py bundle <bundle> <knobs.json> <arch:sm> <program>..
pick   — box_run.sh's ABBA rows -> each knob's value for the arch. Per program s = t(arm) / t(shipped): a probe's
         median `iter ms` over the ABBA repetitions, a cell's tile_bench time over its own cuBLAS leg per pass, the
         passes averaged; null = |t(null) / t(shipped) - 1|. A value ships when its geometric-mean s over the
         programs the knob reaches beats 1 by more than the largest null and no program loses more than that null;
         no null => `open`. A candidate the value gate failed is an emitter bug (ADR-0034 D4): `gate-fail`.
         Written with the machine stamp and the results file (the citation); each verdict cached by (arm, shipped).
           search.py pick <out> <arch> <results-<arch>.txt>  -> <out>/best-<arch>.tsv
ponytail: one knob at a time from the shipped row (misses knob interactions); a second round from the picked row
is the upgrade.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import changed  # noqa: E402 — the emission hash, the tiles, the cache, the box stamp

FACE_SUFFIX = {'default': '', 'contract': '_c', 'exact': '_x'}


def arm_name(knob, value):
    """The bundle's stem suffix (integrated_bundle.sh: `tr '=.' '-p'` on `on-g-<knob>=<value>`)."""
    return f'on-g-{knob}={value}'.translate(str.maketrans('=.', '-p'))


def pair_key(arm_hash, shipped):
    return hashlib.sha256(f'{arm_hash}/{shipped}'.encode()).hexdigest()[:16]


def cache_rows(arch, leg):
    f = changed.CACHE / f'{arch}.tsv'
    rows = [l.split('\t') for l in f.read_text().splitlines()[1:]] if f.exists() else []
    return {r[0]: r for r in rows if r[1] == leg}


def plan(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    knobs = json.loads(Path(a.knobs).read_text()).get(a.arch, {})
    tiles = changed.tiles(changed.ROOT / 'benches/gpu/gen_tiles.py', out / 'tiles')
    missing = [p for p in a.programs if p not in tiles]
    if missing:
        sys.exit(f'search: no such program {missing}')
    verdicts = cache_rows(a.arch, 'search')
    jobs = int(os.environ.get('CARGO_BUILD_JOBS') or max(1, (os.cpu_count() or 4) - 2))
    cands = [(p, f, k, v) for p in a.programs for f in a.faces for k, vs in knobs.items() for v in vs[1:]]
    with cf.ThreadPoolExecutor(jobs) as pool:
        shipped = dict(zip([(p, f) for p in a.programs for f in a.faces], pool.map(
            lambda pf: changed.emission(changed.EMIT, tiles[pf[0]], a.arch, pf[1]),
            [(p, f) for p in a.programs for f in a.faces])))
        hashes = list(pool.map(lambda c: changed.emission(changed.EMIT, tiles[c[0]], a.arch, c[1],
                                                          [f'--geom={c[2]}={c[3]}']), cands))
    rows, seen, arms = [], {}, []
    for (p, f, k, v), h in zip(cands, hashes):
        sh, name = shipped[(p, f)], arm_name(k, v)
        if h == 'err':
            st = 'err'
        elif h == sh:
            st = 'drop'
        elif (p, f, h) in seen:
            st, name = 'alias', seen[(p, f, h)]
        elif pair_key(h, sh) in verdicts:
            st = 'cached'
        else:
            st = 'build'
            arms.append(f'{a.arch}:{p}:{FACE_SUFFIX[f]}:{k}={v}:geom')
        seen.setdefault((p, f, h), name)
        rows.append((p, f, k, v, h, name, st))
    changed.write(out / 'plan.tsv', ['program', 'face', 'knob', 'value', 'hash', 'arm', 'status'], rows)
    (out / 'shipped.tsv').write_text(''.join(f'{p}\t{f}\t{h}\n' for (p, f), h in shipped.items()))
    n = defaultdict(int)
    for r in rows:
        n[r[6]] += 1
    print(f'plan {a.arch}: {len(a.programs)} programs x {len(knobs)} knobs: '
          + ' '.join(f'{k}={v}' for k, v in sorted(n.items())), file=sys.stderr)
    return arms


def bundle(a):
    prof, sm = a.spec.split(':')[:2]
    a.out, a.arch = a.bundle, prof
    arms = plan(a)
    suite = [p for p in a.programs if p.startswith('pt_')]
    cells = [p[3:] for p in a.programs if re.fullmatch(r'mm_\d+x\d+x\d+', p)]
    env = dict(os.environ, SUITE=' '.join(suite), SUITE_C=' '.join(suite) if 'contract' in a.faces else '',
               PROGS='', CELLS=' '.join(cells), ARMS=' '.join(arms), SUITE_ARMS='0')
    root = changed.ROOT / 'benches/gpu'
    with open(Path(a.bundle) / f'integrated-{prof}.log', 'w') as log:
        subprocess.run([str(root / 'integrated_bundle.sh'), a.bundle, f'{prof}:{sm}'], env=env, check=True,
                       stdout=log, stderr=subprocess.STDOUT)
    sys.exit(subprocess.run([sys.executable, str(root / 'preflight.py'), a.bundle, prof]).returncode)


def times(text):
    """{(program, face, arm): t}: a probe's median `iter ms`; a cell's time over its pass's cuBLAS, passes averaged."""
    reps = defaultdict(list)
    for m in re.finditer(r'^probe prog=(\S+) face=(\S+) arm=(\S+) rep=\d+ .*?out=\[[^\]]*iter ms=([\d.]+)', text, re.M):
        reps[(m[1], changed.RESULT_FACE.get(m[2], m[2]), m[3])].append(float(m[4]))
    t = {k: statistics.median(v) for k, v in reps.items()}
    passes = defaultdict(dict)
    for m in re.finditer(r'^pass=(\d+) cell=(\S+) time leg=(\S+) .*med_ms=([\d.]+)', text, re.M):
        passes[(m[1], m[2])][m[3]] = float(m[4])
    ratios = defaultdict(list)
    for legs in passes.values():
        for stem, ms in legs.items():
            s = re.fullmatch(r'(mm_\d+x\d+x\d+)(_c|_x)?(?:\.(\S+))?', stem)
            if s and legs.get('cublas'):
                face = {'_c': 'contract', '_x': 'exact'}.get(s[2], 'default')
                ratios[(s[1], face, s[3] or 'on')].append(ms / legs['cublas'])
    t.update({k: sum(v) / len(v) for k, v in ratios.items()})
    return t


def pick(a):
    out, arch, text = Path(a.out), a.arch, Path(a.results).read_text(errors='replace')
    rows = changed.read(out / 'plan.tsv')
    shipped = {(p, f): h for p, f, h in (l.split('\t') for l in (out / 'shipped.tsv').read_text().splitlines())}
    t = times(text)
    failed = {(m[1], changed.RESULT_FACE.get(m[2], m[2]), m[3])
              for m in re.finditer(r'^VALUE-FAIL prog=(\S+) face=(\S+) arm=(\S+)', text, re.M)}
    # A cell candidate tile_bench's gate fails (and leaves out of the timing): ADR-0034 D4's emitter bug.
    failed |= {(m[1], {'_c': 'contract', '_x': 'exact'}.get(m[2], 'default'), m[3] or 'on')
               for m in re.finditer(r'^pass=\d+ cell=\S+ values leg=(mm_\d+x\d+x\d+)(_c|_x)?(?:\.(\S+))? .*GATE-FAILED',
                                    text, re.M)}
    stamp = changed.box_stamp(text)[0]
    verdicts = cache_rows(arch, 'search')
    s, null, gate, fresh = {}, {}, set(), []
    for p, f, k, v, h, arm, st in rows:
        key = (p, f, k, v)
        if st == 'drop':
            s[key] = 1.0
        elif (p, f, arm) in failed:
            gate.add((k, v))
        elif st == 'cached':
            r = dict(x.split('=', 1) for x in verdicts[pair_key(h, shipped[(p, f)])][3].split())
            s[key] = float(r['s'])
            if not math.isnan(float(r['null'])):
                null[(p, f)] = max(null.get((p, f), 0), float(r['null']))
        elif (p, f, arm) in t and (p, f, 'on') in t:
            s[key] = t[(p, f, arm)] / t[(p, f, 'on')]
            if (p, f, 'null') in t:
                null[(p, f)] = abs(t[(p, f, 'null')] / t[(p, f, 'on')] - 1)
            if st == 'build':
                fresh.append((pair_key(h, shipped[(p, f)]), 'search', '-',
                              f's={s[key]:.4f} null={null.get((p, f), float("nan")):.4f} prog={p} face={f} knob={k} '
                              f'value={v}', stamp))
    knobs = json.loads(Path(a.knobs).read_text()).get(arch, {}) if a.knobs else {}
    by_knob = defaultdict(lambda: defaultdict(dict))  # knob -> value -> (program, face) -> s
    reach = defaultdict(set)
    for p, f, k, v, h, arm, st in rows:
        if st != 'drop':
            reach[k].add((p, f))
        if (p, f, k, v) in s:
            by_knob[k][v][(p, f)] = s[(p, f, k, v)]
    best = []
    for k in sorted({r[2] for r in rows}):
        pf = reach[k]
        default = str(knobs[k][0]) if k in knobs else '?'
        if not pf:
            best.append((arch, k, default, default, '1.000', 0, '-', '-', '-', 'no-site', a.results, stamp))
            continue
        score = {}
        for v, per in by_knob[k].items():
            if (k, v) not in gate and pf <= per.keys():
                score[v] = (math.exp(sum(math.log(per[x]) for x in pf) / len(pf)), max(per.items(), key=lambda x: x[1]))
        nl = max((null[x] for x in pf if x in null), default=None)
        if not score:
            verdict, v, g, worst = 'open', default, float('nan'), (('-', '-'), float('nan'))
        else:
            v = min(score, key=lambda x: score[x][0])
            g, worst = score[v]
            verdict = 'open' if nl is None or len([x for x in pf if x in null]) < len(pf) else \
                'ship' if g < 1 - nl and worst[1] <= 1 + nl else 'default'
        if any(x[0] == k for x in gate):
            verdict += '+gate-fail:' + ','.join(sorted(str(x[1]) for x in gate if x[0] == k))
        best.append((arch, k, default, v if verdict.startswith('ship') else default, f'{g:.3f}', len(pf),
                     '-' if nl is None else f'{nl:.3f}', '/'.join(worst[0]), f'{worst[1]:.3f}', verdict,
                     a.results, stamp))
    changed.write(out / f'best-{arch}.tsv', ['arch', 'knob', 'default', 'best', 'gmean_s', 'programs', 'null',
                                             'worst_program', 'worst_s', 'verdict', 'citation', 'machine'], best)
    if fresh:
        changed.CACHE.mkdir(parents=True, exist_ok=True)
        f = changed.CACHE / f'{arch}.tsv'
        head = '' if f.exists() else 'key\tleg\tver\tvalue\tstamp\n'
        with f.open('a') as fh:
            fh.write(head + ''.join('\t'.join(r) + '\n' for r in fresh))
    for b in best:
        print('\t'.join(map(str, b[:10])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('plan', 'bundle'):
        c = sub.add_parser(name)
        c.add_argument('bundle' if name == 'bundle' else 'out')
        c.add_argument('knobs')
        c.add_argument('spec' if name == 'bundle' else 'arch')
        c.add_argument('programs', nargs='+')
        c.add_argument('--faces', nargs='+', default=['default'], choices=list(FACE_SUFFIX))
    c = sub.add_parser('pick')
    c.add_argument('out')
    c.add_argument('arch')
    c.add_argument('results')
    c.add_argument('--knobs', help='knobs.json, for the shipped value of each knob')
    a = ap.parse_args()
    if a.cmd == 'plan':
        print(' '.join(plan(a)))
    else:
        {'bundle': bundle, 'pick': pick}[a.cmd](a)


if __name__ == '__main__':
    main()
