#!/usr/bin/env python3
"""The GPU_RUNGS_OFF rule over one part's sessions (box_run results files, different boxes).

Evidence, per arm (`off-<rung>` = shipped ON, `on-<rung>` = shipped OFF), as an OFF/ON speedup
(>1 = ON faster), medians within one run:
  kernels  `# n=… face=<f>.<arm>` tile_bench rows, each bench normalized by its own cuBLAS leg
           (cancels the clock drift of a power-capped box between benches);
  probes   `probe prog=pt_<probe>_<n> … arm=<arm>` rows, the program's own `iter ms` (S54);
  e2e      `e2e prog=<prog> … arm=<arm>` rows, process wall;
  decode   `decode prog=decode_<T>_<d>[_c].<arm>` rows, process wall.
Probe, e2e and decode arms run interleaved in one session, so their ratio needs no reference leg.

A class is (rung, fill, face). Face: Exact (_exact), Contract (_c, the FMA face), Tf32 (_tf).
Fill is read from the grid the rung's decision is taken on: `Any` for a rung decided once per
module (gpu.rs `host_rungs`: its rows must cover every fill), else Under at n <= 1024 and Filled
above — `Gpu::warps_per_scheduler` of a site's base FMA grid on all five parts (profile.rs test
`warps_per_scheduler_reads_the_base_grid`).

At one size: stable+ when s > 1.01 in every session, stable- when s < 0.99 in every session,
`*` = disjoint min-max ranges. `rule` (plan-s54 §3): a rung ships ON in a class iff some size is
stable+ — it decides a rung measured for the first time (`--new`). A rung already in the table flips
only on evidence (`ships`): ON -> OFF when some size is stable- and none stable+, OFF -> ON the other
way; no effect (no stable size), a class some session did not measure, and mixed signs (stable+ at one
size, stable- at another: no (fill, face) key separates them) keep what ships. Composite arms
(arch-rungs = the whole table, fma-regs = idx32+affine+pad+wide-fma, `a+b` arms) are shown, never
decided. Every session's `null` arm (the shipped kernel benched again) prints the noise floor.

Then the RungOff rows that ship, merged across classes where they agree (a `Contract` row covers
the TF32 face too: profile.rs `Face::covers`), to paste into GPU_RUNGS_OFF.

S62: a kernel bench may be a cell `# n=<M>x<N>x<K>`; its fill is the site's own, read from
`--fill <tsv>` (`<profile> <M>x<N>x<K> <Under|Filled>` lines, gpu.rs `launch`'s fill key).
`--null-eps`: no effect within max(1%, the sessions' largest null-arm deviation) instead of 1%.

usage: python3 benches/gpu/rungs.py [--new <rung>,...] [--fill <tsv>] [--null-eps] <results-session-1> [...]
       python3 benches/gpu/rungs.py --selftest
"""
import os
import re
import statistics
import sys
from collections import defaultdict

EPS = 0.01  # no effect within ±1%: the same ratio differs by a median 0.5-1.2% between two boxes (S54 base vs confirm)
# Rungs decided once per module (gpu.rs `host_rungs`). S54's placement rungs are read there too,
# except `epilogue`, `raster` and `l2-gate`, which a site's own launch reads at its fill.
HOST = {'resident', 'devgen', 'eager-init', 'retain-delta', 'arena', 'devmap', 'cell-law', 'delta-h2d',
        'retain-resident', 'lane-stride', 'rowfold', 'rowfold-layout', 'streams', 'stream-gate', 'pinned',
        'pdl', 'l2-window'}
COMPOSITE = {'arch-rungs', 'fma-regs'}
FACE = {'_exact': 'Exact', '_c': 'Contract', '_tf': 'Tf32'}
FILL = {}  # (profile, 'MxNxK') -> fill, from --fill
PROFILE = '?'


def fill_of(rung, n):
    if rung in HOST:
        return 'Any'
    if isinstance(n, str) and not n.isdigit():
        return FILL.get((PROFILE, n), '?')
    return 'Under' if int(n) <= 1024 else 'Filled'


def profile_of(path):
    for l in open(path):
        m = re.match(r'== (\S+) sm_', l)
        if m:
            return m[1]
    return '?'


def kernel_rows(path):
    rows, cur = {}, {}
    for l in open(path):
        m = re.match(r'# n=(\S+) face=(\S+)', l)
        if m:
            cur = {'n': m[1], 'key': m[2]}
            continue
        m = re.match(r'time leg=(\w+) .* min_ms=([\d.]+) med_ms=([\d.]+) max_ms=([\d.]+)', l)
        if m:
            cur[m[1]] = tuple(float(x) for x in m.groups()[1:])
        if l.startswith('ratio') and 'mapal' in cur:
            rows[(cur['n'], cur['key'])] = cur
    return rows


def split_arm(arm):
    """`off-<rung>` / `on-<rung>` -> (shipped, rung); anything else (on, s52, PREV tags) -> None."""
    d, _, rung = arm.partition('-')
    return (('ON' if d == 'off' else 'OFF'), rung) if d in ('off', 'on') and rung else None


def kernel_arms(path):
    rows, out, nulls = kernel_rows(path), {}, null_ab(path)
    for (n, key), r in rows.items():
        face, _, arm = key.partition('.')
        sa = split_arm(arm)
        if not sa or (n, face) not in rows or face not in FACE:
            continue
        base = rows[(n, face)]
        on, off = (base, r) if sa[0] == 'ON' else (r, base)
        disj = on['mapal'][2] < off['mapal'][0] or off['mapal'][2] < on['mapal'][0]
        sp = (off['mapal'][1] / off['cublas'][1]) / (on['mapal'][1] / on['cublas'][1])
        # Against the shipped row's repeat (`null`) when the session has one: the shipped bench's own
        # offset (null s = shipped / repeat) cancels.
        ns = nulls.get(f'{face} {n}', 1.0)
        sp = sp * ns if sa[0] == 'ON' else sp / ns
        out[(sa[1], face, f'{face} {n}', n)] = (sp, disj, sa[0])
    return out


def null_ab(path):
    """The shipped kernel benched again as arm `null` (bundle.sh): the same text's cuBLAS-normalized
    ratio — the session's own noise floor for one A/B, position in the bench order included."""
    rows = kernel_rows(path)
    return {f'{face} {n}': (rows[(n, face)]['mapal'][1] / rows[(n, face)]['cublas'][1])
            / (r['mapal'][1] / r['cublas'][1])
            for (n, key), r in rows.items() for face in [key.removesuffix('.null')]
            if key.endswith('.null') and (n, face) in rows}


def program_arms(path):
    """Probe / e2e / decode arms: {(rung, face, size label, n): (speedup, disjoint, shipped)}."""
    xs = defaultdict(list)  # (kind, prog, face, arm) -> samples
    for l in open(path):
        m = re.match(r'(probe|e2e) prog=(\S+) face=(\S+) arm=(\S+) rep=\d+ wall_us=(\d+) .*?out=\[(.*?)\]', l)
        if m:
            kind, prog, face, arm, wall, out = m.groups()
            it = re.search(r'iter ms=([\d.e+-]+)', out)
            v = float(it[1]) if kind == 'probe' and it else int(wall) / 1000
            xs[(kind, prog, face, arm)].append(v)
            continue
        m = re.match(r'decode prog=(decode_\d+_\d+)(_c)?(?:\.(\S+))? rep=\d+ rc=0 wall_s=([\d.]+)', l)
        if m:
            prog, c, arm, wall = m.groups()
            xs[('decode', prog, '_c' if c else '_exact', arm or 'on')].append(float(wall) * 1000)
    out = {}
    for (kind, prog, face, arm), v in xs.items():
        sa = split_arm(arm)
        base = xs.get((kind, prog, face, 'on'))
        if not sa or not base:
            continue
        on, off = (base, v) if sa[0] == 'ON' else (v, base)
        disj = max(on) < min(off) or max(off) < min(on)
        n = max(int(x) for x in re.findall(r'_(\d+)', prog))
        label = f"{prog.removeprefix('pt_')}{face}"
        out[(sa[1], face, label, n)] = (statistics.median(off) / statistics.median(on), disj, sa[0])
    return out


def classes(files, new=()):
    """Per class: (class, shipped, rule, target, status, cells). `rule` is the plan's rule (ON iff
    some size is stable+); it decides a rung in `new` (measured for the first time). A rung already
    in the table flips only on evidence: ON -> OFF when some size is stable- and none stable+, OFF ->
    ON the other way round; no effect (no stable size) and mixed signs keep what ships."""
    sess = [{**kernel_arms(f), **program_arms(f)} for f in files]
    cls = defaultdict(lambda: defaultdict(dict))  # class -> size label -> session -> value
    for i, a in enumerate(sess):
        for (rung, face, label, n), v in a.items():
            cls[(rung, fill_of(rung, n), FACE[face])][label][i] = v
    out = []
    for c, sizes in sorted(cls.items()):
        shipped = {v[2] for s in sizes.values() for v in s.values()}
        full = {k: s for k, s in sizes.items() if len(s) == len(sess)}  # measured in every session
        pos = [k for k, s in full.items() if all(v[0] > 1 + EPS for v in s.values())]
        neg = [k for k, s in full.items() if all(v[0] < 1 - EPS for v in s.values())]
        strong_pos = [k for k in pos if all(v[1] for v in full[k].values())]
        strong_neg = [k for k in neg if all(v[1] for v in full[k].values())]
        sh = '/'.join(sorted(shipped))
        # A class some session did not measure is not decided here (one session is not a sign).
        rule = ('ON' if pos else 'OFF') if full or len(sess) == 1 else 'n/a'
        if c[0] in new:
            target = rule if rule != 'n/a' and not (pos and neg) else sh
        else:
            target = {('ON', False, True): 'OFF', ('OFF', True, False): 'ON'}.get((sh, bool(pos), bool(neg)), sh)
        if c[0] in COMPOSITE or '+' in c[0]:
            status = 'composite'
        elif rule == 'n/a':
            status = f'{max(len(s) for s in sizes.values())} of {len(sess)} sessions'
        elif pos and neg:
            status = 'mixed (held)'
        elif target != sh:
            status = 'FLIP (disjoint)' if strong_pos or strong_neg else 'FLIP'
        elif sh == rule:
            status = 'ok'
        else:
            status = 'no effect (held)'
        cells = '; '.join(
            f"{label}: " + ' '.join(f"{s[i][0]:.3f}{'*' if s[i][1] else ''}" for i in sorted(s))
            for label, s in sorted(sizes.items()))
        out.append((c, sh, rule, target, status, cells))
    return out, len(sess)


FACES = {'Any': ('Exact', 'Contract', 'Tf32'), 'Contract': ('Contract', 'Tf32'), 'Exact': ('Exact',),
         'Tf32': ('Tf32',)}
FILLS = {'Any': ('Any', 'Under', 'Filled'), 'Under': ('Under',), 'Filled': ('Filled',)}


def covers(key, cls):
    """profile.rs `gpu_rungs`: a row's fill and face cover a class's (a `Contract` row covers TF32)."""
    return cls[0] in FILLS[key[0]] and cls[1] in FACES[key[1]]


def merge(offs, ons):
    """The fewest (fill, face) keys covering every OFF class and no ON class, widest first; an
    unmeasured class is free either way. Returns (keys, OFF classes no key can take)."""
    todo, keys = set(offs), []
    for fi in ('Any', 'Under', 'Filled'):
        for fa in ('Any', 'Contract', 'Exact', 'Tf32'):
            k = (fi, fa)
            if any(covers(k, c) for c in todo) and not any(covers(k, c) for c in ons):
                keys.append(k)
                todo -= {c for c in todo if covers(k, c)}
    return keys, sorted(todo)


def rows(out, files):
    """The RungOff rows that ship: every class whose target is OFF, merged per rung."""
    by = defaultdict(lambda: ([], []))
    for (rung, fill, face), sh, _, target, st, cells in out:
        if st != 'composite':
            by[rung][0 if target == 'OFF' else 1].append(((fill, face), cells, sh))
    names = ' / '.join(os.path.basename(f) for f in files)
    res = []
    for rung, (offs, ons) in sorted(by.items()):
        keys, left = merge([c for c, _, _ in offs], [c for c, _, _ in ons])
        for k in keys:
            mine = [(cells, sh) for c, cells, sh in offs if covers(k, c)]
            ev = '; '.join(cells for cells, _ in mine)
            res.append((rung, k, any(sh != 'OFF' for _, sh in mine),
                        f"{names} OFF/ON (kernels cuBLAS-normalized, programs interleaved): {ev} "
                        f"(ms in the files)"))
        for c in left:
            res.append((rung, c, None, 'no key: OFF here, ON on the TF32 face (an FMA-only Face would take it)'))
    return res


def selftest():
    U, F, A = 'Under', 'Filled', 'Any'
    # An unmeasured class (no arm: the text did not change) is free, so the widest key.
    assert merge([(U, 'Contract'), (F, 'Contract')], []) == ([(A, A)], [])
    assert merge([(U, 'Tf32'), (F, 'Tf32')], [(U, 'Contract')]) == ([(A, 'Tf32')], [])
    assert merge([(A, 'Exact'), (A, 'Contract')], []) == ([(A, A)], [])
    assert merge([(U, 'Tf32')], [(U, 'Contract'), (F, 'Tf32')]) == ([(U, 'Tf32')], [])
    assert merge([(U, 'Contract')], [(U, 'Tf32')]) == ([], [(U, 'Contract')])
    assert merge([(U, 'Exact'), (F, 'Contract')], [(F, 'Exact')]) == ([(A, 'Contract'), (U, A)], [])
    assert fill_of('resident', 1024) == A and fill_of('min-blocks', 1024) == U and fill_of('pad', 2048) == F
    assert split_arm('off-warp-fill') == ('ON', 'warp-fill') and split_arm('on-async') == ('OFF', 'async')
    assert split_arm('s52') is None and split_arm('on') is None
    # The two rules on synthetic sessions: `off-x` (shipped ON) loses 5% at 1024 in both; `on-y`
    # (shipped OFF) gains 0.5% (no effect); `off-z` (new rung) gains 0.5% (no effect).
    import tempfile
    def bench(key, ms):
        return (f"# n=1024 face={key}\ntime leg=mapal n=30 min_ms={ms} med_ms={ms} max_ms={ms}\n"
                f"time leg=cublas n=30 min_ms=1 med_ms=1 max_ms=1\nratio x\n")
    text = (bench('_c', 1.0) + bench('_c.off-x', 0.95) + bench('_c.on-y', 0.995) + bench('_c.off-z', 1.005))
    paths = []
    for _ in range(2):
        f = tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False)
        f.write('== part sm_00\n' + text)
        f.close()
        paths.append(f.name)
    got = {c[0]: (sh, rule, target) for c, sh, rule, target, _, _ in classes(paths, {'z'})[0]}
    assert got == {'x': ('ON', 'OFF', 'OFF'), 'y': ('OFF', 'OFF', 'OFF'), 'z': ('ON', 'OFF', 'OFF')}, got
    got = {c[0]: target for c, _, _, target, _, _ in classes(paths)[0]}
    assert got == {'x': 'OFF', 'y': 'OFF', 'z': 'ON'}, got  # z not new: no effect holds ON
    for p in paths:
        os.unlink(p)
    print('selftest ok')


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest()
        sys.exit()
    args = sys.argv[1:]
    new = set(args[1].split(',')) if args[:1] == ['--new'] else set()
    files = args[2:] if new else args
    if files[:1] == ['--fill']:
        for line in open(files[1]):
            prof, cell, fill = line.split()
            FILL[(prof, cell)] = fill
        files = files[2:]
    if files[:1] == ['--null-eps']:
        files = files[1:]
        EPS = max([EPS] + [abs(v - 1) for f in files for v in null_ab(f).values()])
        print(f"EPS = {EPS:.4f} (max of 1% and the null arms' deviation)\n")
    PROFILE = profile_of(files[0])
    out, ns = classes(files, new)
    print(f"| rung | fill | face | shipped | rule ({ns} session{'s' if ns > 1 else ''}) | ships | status | OFF/ON speedup per size, per session (`*` disjoint) |")
    print('|---|---|---|---|---|---|---|---|')
    for (rung, fill, face), sh, v, target, st, cells in out:
        print(f"| {rung} | {fill} | {face} | {sh} | {v} | {target} | {st} | {cells} |")
    for f in files:
        nl = null_ab(f)
        if nl:
            print(f"\nnull A/B (shipped vs itself), {os.path.basename(f)}: max |s-1| = "
                  f"{max(abs(s - 1) for s in nl.values()):.3f}; "
                  + '; '.join(f"{k}: {s:.3f}" for k, s in sorted(nl.items())))
    print(f"\n// GPU_RUNGS_OFF rows that ship on {profile_of(files[0])} (new = a class that ships ON today)")
    for rung, (fill, face), new, cite in rows(out, files):
        if new is None:
            print(f"    // {rung} {fill} {face}: {cite}")
            continue
        print(f"    // {'new' if new else 'held'}\n    RungOff {{\n        profile: \"{profile_of(files[0])}\",\n"
              f"        rung: \"{rung}\",\n        fill: Fill::{fill},\n        face: Face::{face},\n"
              f"        cite: \"{cite}\",\n    }},")
