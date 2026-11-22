#!/usr/bin/env python3
"""Changed programs: the timed-suite programs and cells an A/B moves (benches/local/README.md).

Every suite program (benches/local/suite.txt, the default face) and every face of each cell is emitted for the
rtx-5080 on both arms of the A/B; host IR and device IR, less the `; mapal.` evidence comments, are compared
(changed.py `emission`: equal device IR is equal PTX under one LLVM). A program whose text is equal on both arms
links the same host object and loads the same PTX on both: its A/B is null by construction and it is not timed. The
others are bundled with the other arm (gpu_bundle.sh ARMS) and timed ABBA (box_run.sh's probe section, the arm order
reversed on even repetitions; the cells in tile_bench --splitk). A base whose runtime or timing harnesses differ
(changed.py RT_PATHS) moves every program. The full suite still runs once per step (the safety net).

  suite_diff.py <out> --off <rung>[,<rung>..]       HEAD shipped vs HEAD with the rungs OFF   (arm off-<r>[+<r>])
  suite_diff.py <out> --on <rung>[,<rung>..]        HEAD shipped vs HEAD with the rungs ON    (arm on-<r>[+<r>])
  suite_diff.py <out> --base <ref> | --base-emit P  HEAD vs the emitter at <ref> / P          (arm on-base)
  [--stems ..] [--cells ..] [--jobs N]
-> <out>/suite.tsv (stem face status head_hash arm_hash); prints STEMS / CELLS / ARMS / BASE_EMIT as shell
   assignments for gpu_bundle.sh (gpu_changed.sh evaluates them).
"""
import argparse
import concurrent.futures as cf
import os
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'benches/gpu'))
import changed  # noqa: E402

ARCH = 'rtx-5080'
# gpu_bundle.sh's faces: the suite runs the default face alone; a cell is built on all five (integrated_bundle.sh).
CELL_FACES = {'': [], '_x': ['--exact'], '_c': ['--order=sequential'], '_r': ['--no-contract', '--order=fixed'],
              '_rc': ['--order=fixed']}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    arm = ap.add_mutually_exclusive_group(required=True)
    arm.add_argument('--off', help='rungs OFF on the B arm')
    arm.add_argument('--on', help='rungs ON on the B arm')
    arm.add_argument('--base', help='a git ref: its emitter is the B arm')
    arm.add_argument('--base-emit', help='a prebuilt emitter as the B arm')
    ap.add_argument('--stems', nargs='*', help='default: benches/local/suite.txt')
    ap.add_argument('--cells', nargs='*', default=['4096x4096x4096', '16x2048x7168'])
    ap.add_argument('--jobs', type=int)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    changed.build_emit()
    head = changed.tiles(ROOT / 'benches/gpu/gen_tiles.py', out / 'tiles')
    stems = a.stems if a.stems is not None else (HERE / 'suite.txt').read_text().split()
    base_emit, moved_rt = None, False
    if a.base or a.base_emit:
        base_emit = Path(a.base_emit).resolve() if a.base_emit else changed.base_emitter(a.base)
        b_emit, b_flags, tag = base_emit, [], 'on-base'
        entry = lambda stem, face: f'{ARCH}:{stem}:{face}:base:base'  # noqa: E731
        moved_rt = bool(a.base) and changed.rt_stamp(a.base) != changed.rt_stamp('HEAD')
        if moved_rt:
            print(f'suite_diff: the runtime or a timing harness differs from {a.base} (changed.py RT_PATHS): every '
                  'program is timed, and its base side needs a full-suite run at the base (one bundle links one '
                  'runtime)', file=sys.stderr)
    else:
        d, rs = ('off', a.off) if a.off else ('on', a.on)
        b_emit, b_flags, tag = changed.EMIT, [f'--gpu-{d}={rs}'], f'{d}-{rs.replace(",", "+")}'
        entry = lambda stem, face: f'{ARCH}:{stem}:{face}:{rs}:{d}'  # noqa: E731
    rows = [(s, '', []) for s in stems] + [(f'mm_{c}', f, fl) for c in a.cells for f, fl in CELL_FACES.items()]
    missing = [r[0] for r in rows if r[0] not in head]
    if missing:
        sys.exit(f'not a gen_tiles.py program: {" ".join(sorted(set(missing)))}')
    jobs = a.jobs or max(1, (os.cpu_count() or 4) // 4)
    with cf.ThreadPoolExecutor(jobs) as pool:
        ha = list(pool.map(lambda r: changed.emission(changed.EMIT, head[r[0]], ARCH, 'default', r[2]), rows))
        hb = list(pool.map(lambda r: changed.emission(b_emit, head[r[0]], ARCH, 'default', r[2] + b_flags), rows))
    table, moved = [], []
    for (stem, face, _), x, y in zip(rows, ha, hb):
        st = 'err' if 'err' in (x, y) else 'runtime' if moved_rt else 'same' if x == y else 'differs'
        table.append((stem, face or '_default', st, x, y))
        if st in ('differs', 'runtime'):
            moved.append((stem, face))
        elif st == 'err':
            print(f'suite_diff: {stem}{face}: an arm does not emit ({x} / {y}): not timed', file=sys.stderr)
    changed.write(out / 'suite.tsv', ['stem', 'face', 'status', 'head_hash', f'{tag}_hash'], table)
    suite = [s for s in stems if (s, '') in moved]
    cells = [c for c in a.cells if any(s == f'mm_{c}' for s, _ in moved)]
    # tile_bench --splitk times every arm the bundle built for a cell, on every face.
    arms = [entry(s, '') for s in suite] + [entry(s, f) for s, f in moved if s.startswith('mm_')]
    n = {k: sum(r[2] == k for r in table) for k in ('same', 'differs', 'runtime', 'err')}
    print(f'suite_diff {tag}: {len(stems)} programs + {len(a.cells)} cells x {len(CELL_FACES)} faces: '
          + ' '.join(f'{k}={v}' for k, v in n.items() if v) + f' -> time {len(suite)} programs, {len(cells)} cells '
          f'({out}/suite.tsv)', file=sys.stderr)
    for k, v in [('STEMS', suite), ('CELLS', cells), ('ARMS', arms)]:
        print(f'{k}={shlex.quote(" ".join(v))}')
    if base_emit:
        print(f'BASE_EMIT={shlex.quote(str(base_emit))}')


if __name__ == '__main__':
    main()
