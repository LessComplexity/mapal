#!/usr/bin/env python3
"""site-geometry A1: measure only what changed.

diff   — emit every bench program (gen_tiles.py) per arch x face on a base ref and on this checkout. Host IR and
         device IR, less the `; mapal.` evidence comments, decide (equal device IR => equal PTX under one LLVM). For
         every changed row, each rung's OFF arm is emitted at HEAD; one whose emission differs from the shipped
         build is a differing arm. A row whose HEAD emission the arch's result cache already holds is `cached`.
           changed.py diff <out> [--base REF] [--base-emit PATH] [--archs ..] [--faces ..] [--rungs ..] [--jobs N]
           A runtime / harness change (RT_PATHS at the base vs the working tree) makes every row `changed`;
           OFF arms are probed only where the emission moved. Parts with GPU_RUNGS_OFF rows also probe
           `--gpu-on=<r>` for those rungs (arms `on-<r>`).
           -> <out>/emission.tsv  program arch face status base_hash head_hash  (same|changed|new|gone|cached)
              <out>/stamp         HEAD's runtime stamp (the cache's `ver` prefix)
              <out>/arms.tsv      program arch face arm hash tag                (changed rows' arms != shipped;
                                  tag new | cause (off at HEAD == off at base: the change is this rung's) | carried)
              <out>/changed.tsv   program status arch/face.. rungs              (the summary)
              <out>/tiles/        HEAD's programs (the cache's source hashes)
env    — one arch's selection as shell assignments for integrated_bundle.sh / bundle.sh (changed_bundle.sh);
         CPU-only changes go to one box (--cpu-box, default this arch).
           changed.py env <out> <arch> [--cpu-box ARCH]
ingest — a box run's rows into the result cache benches/results/cache/<arch>.tsv (key leg ver value stamp):
         key = the arm's emission hash (Mapal legs) or the program's source hash (library legs); ver = the
         runtime stamp | nvcc release | driver for Mapal, the torch version or the nvcc release for the libraries.
           changed.py ingest <out> <arch> <results-<arch>.txt>
The x86-64-v3 "arch" is the CPU-only build on the boxes' host profile (the `cpu` arm).
"""
import argparse
import concurrent.futures as cf
import hashlib
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMIT = Path(os.environ.get('EMIT', ROOT / 'target/release/examples/emit'))
CACHE = Path(os.environ.get('MAPAL_RESULT_CACHE') or ROOT / 'benches/results/cache')
ARCHS = ['rtx-4090', 'rtx-5090', 'h100-pcie', 'rtx-3090', 'x86-64-v3']
FACES = {'default': [], 'contract': ['--order=sequential'], 'exact': ['--exact']}
RESULT_FACE = {'_default': 'default', '_c': 'contract', '_x': 'exact', '_exact': 'exact'}
CPU = 'x86-64-v3'
# What a Mapal row's time runs through besides its emission: the runtime it links and the harnesses that time it
# (box_run.sh only orchestrates and gates; the library legs are keyed by their own source and toolchain).
RT_PATHS = ['crates/mapal-rt', 'benches/gpu/tile_bench.cu', 'benches/gpu/mlp_bench.cu']
sys.path.insert(0, str(Path(__file__).resolve().parent))


def sh(*cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def norm(text):
    return ''.join(l for l in text.splitlines(True) if not l.startswith('; mapal.'))


def emission(emit, src, arch, face, extra=()):
    """Hash of one emission's host IR + device IR (16 hex), `err` when the emitter refuses."""
    with tempfile.TemporaryDirectory() as t:
        dev = Path(t) / 'd.ll'
        cmd = [str(emit), str(src), '-', '--rewrite', f'--target={arch}', *FACES[face], *extra]
        if arch != CPU:
            cmd[5:5] = ['--host=x86-64-v3', f'--device-out={dev}']
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            return 'err'
        body = norm(r.stdout) + '\0' + (norm(dev.read_text()) if dev.exists() else '')
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def rungs(emit, src):
    """Every rung name the emitter accepts (its unknown-rung error lists them), aliases dropped."""
    r = subprocess.run([str(emit), str(src), '-', '--gpu-off=?'], capture_output=True, text=True)
    m = re.search(r'\(([^)]*)\)', r.stderr)
    return [n for n in m[1].split(', ') if n not in ('fma-regs', 'arch-select')] if m else []


def tiles(gen, out):
    out.mkdir(parents=True, exist_ok=True)
    sh(sys.executable, str(gen), str(out))
    return {p.stem: p for p in out.glob('*.mapal') if not p.stem.endswith(('_xabs', '_tfsrc'))}


def base_tree(ref, tmp):
    """(emitter, programs) at `ref`: the emitter is built once per commit (target/base-emit/<sha>)."""
    sha = sh('git', '-C', str(ROOT), 'rev-parse', ref).strip()
    gen = tmp / 'gen_tiles.py'
    gen.write_text(sh('git', '-C', str(ROOT), 'show', f'{sha}:benches/gpu/gen_tiles.py'))
    exe = ROOT / 'target/base-emit' / sha[:12] / 'emit'
    if not exe.exists():
        src = ROOT / 'target/base-emit/src'
        shutil.rmtree(src, ignore_errors=True)
        src.mkdir(parents=True)
        arc = subprocess.Popen(['git', '-C', str(ROOT), 'archive', sha], stdout=subprocess.PIPE)
        subprocess.run(['tar', '-x', '-m', '-C', str(src)], stdin=arc.stdout, check=True)
        arc.wait()
        print(f'building the {ref} emitter ({sha[:12]}) ...', file=sys.stderr)
        env = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / 'target/base-emit/target'))
        subprocess.run(['cargo', 'build', '-q', '--release', '--example', 'emit', '-p', 'mapal-backend-llvm'],
                       cwd=src, env=env, check=True)
        exe.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'target/base-emit/target/release/examples/emit', exe)
        shutil.rmtree(src)
    return exe, tiles(gen, tmp / 'tiles')


def rt_stamp(ref):
    """16 hex over RT_PATHS' blobs at `ref`; `HEAD` adds the working tree's uncommitted changes to them."""
    g = ['git', '-C', str(ROOT)]
    ids = sh(*g, 'ls-tree', '-r', sh(*g, 'rev-parse', ref).strip(), '--', *RT_PATHS)
    if ref == 'HEAD':
        ids += sh(*g, 'diff', 'HEAD', '--', *RT_PATHS)
    return hashlib.sha256(ids.encode()).hexdigest()[:16]


def cached_keys(arch, rt):
    """Mapal keys the cache holds measured on runtime stamp `rt`."""
    f = CACHE / f'{arch}.tsv'
    rows = [l.split('\t') for l in f.read_text().splitlines()] if f.exists() else []
    return {r[0] for r in rows if r[1:2] == ['mapal'] and r[2].split('|')[0] == rt}


def rungs_off_at(arch):
    """The rungs a GPU_RUNGS_OFF row ships OFF on `arch` (their ON arm is the one to re-judge)."""
    from evidence import rows_off
    return sorted({r for prof, r, *_ in rows_off() if prof == arch})


def build_emit():
    """The emitter this checkout's source builds (a no-op when fresh); EMIT= names another and skips it."""
    if 'EMIT' not in os.environ:
        subprocess.run(['cargo', 'build', '-q', '--release', '--example', 'emit', '-p', 'mapal-backend-llvm'],
                       cwd=ROOT, check=True)


def diff(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    build_emit()
    rt_base, rt = rt_stamp(a.base), rt_stamp('HEAD')
    (out / 'stamp').write_text(rt + '\n')
    head = tiles(ROOT / 'benches/gpu/gen_tiles.py', out / 'tiles')
    jobs = a.jobs or int(os.environ.get('CARGO_BUILD_JOBS') or max(1, (os.cpu_count() or 4) - 2))
    with tempfile.TemporaryDirectory() as t, cf.ThreadPoolExecutor(jobs) as pool:
        bemit, base = base_tree(a.base, Path(t)) if not a.base_emit else (a.base_emit, tiles(
            ROOT / 'benches/gpu/gen_tiles.py', Path(t) / 'tiles'))
        rows = [(p, arch, face) for p in sorted(set(head) | set(base)) for arch in a.archs for face in a.faces]
        hb = pool.map(lambda r: emission(bemit, base[r[0]], r[1], r[2]) if r[0] in base else '-', rows)
        hh = pool.map(lambda r: emission(EMIT, head[r[0]], r[1], r[2]) if r[0] in head else '-', rows)
        cached = {arch: cached_keys(arch, rt) for arch in a.archs}
        if rt != rt_base:
            print(f'runtime/harness changed since {a.base} ({rt_base} -> {rt}): every row is measured again',
                  file=sys.stderr)
        em = []
        for (p, arch, face), b, h in zip(rows, hb, hh):
            st = 'new' if b == '-' else 'gone' if h == '-' else 'same' if b == h and rt == rt_base else \
                'cached' if h in cached[arch] else 'changed'
            em.append((p, arch, face, st, b, h))
        # Arms only where the emission moved (a runtime-only change re-measures the shipped build).
        todo = [(p, arch, face, h) for p, arch, face, st, b, h in em if st in ('changed', 'new') and b != h]
        names = a.rungs or rungs(EMIT, next(iter(head.values())))
        known = set(rungs(bemit, next(iter(base.values())))) if base else set()
        ons = {arch: [r for r in rungs_off_at(arch) if not a.rungs or r in a.rungs] for arch in a.archs}
        probe = [(p, arch, face, h, f'off-{r}') for p, arch, face, h in todo for r in names] + \
                [(p, arch, face, h, f'on-{r}') for p, arch, face, h in todo for r in ons[arch]]
        flag = lambda arm: f'--gpu-{arm.replace("-", "=", 1)}'  # noqa: E731 — off-x -> --gpu-off=x
        offs = pool.map(lambda x: emission(EMIT, head[x[0]], x[1], x[2], [flag(x[4])]), probe)
        live = [(x, o) for x, o in zip(probe, offs) if o not in (x[3], 'err')]
        # The arm's tag: `new` (a rung the base lacks), `cause` (the arm at HEAD gives the base's arm: the change
        # is this rung's), `carried` (the change runs through both arms of its A/B).
        boffs = pool.map(lambda xo: emission(bemit, base[xo[0][0]], xo[0][1], xo[0][2], [flag(xo[0][4])])
                         if xo[0][0] in base and xo[0][4].split('-', 1)[1] in known else 'new', live)
        arms = [(p, arch, face, arm, o, 'new' if b == 'new' else 'cause' if b == o else 'carried')
                for ((p, arch, face, h, arm), o), b in zip(live, boffs)]
    write(out / 'emission.tsv', ['program', 'arch', 'face', 'status', 'base_hash', 'head_hash'], em)
    write(out / 'arms.tsv', ['program', 'arch', 'face', 'arm', 'hash', 'tag'], arms)
    summary = defaultdict(lambda: [set(), set(), set()])
    for p, arch, face, st, _, _ in em:
        if st in ('changed', 'new'):
            s = summary[p]
            s[0].add(st)
            s[1].add(f'{arch}/{face}')
    for p, _, _, arm, _, tag in arms:
        if tag != 'carried':
            summary[p][2].add(arm)
    write(out / 'changed.tsv', ['program', 'status', 'arch/face emission differs', 'new / cause arms (differ)'],
          [(p, '/'.join(sorted(s[0])), ' '.join(sorted(s[1])), ','.join(sorted(s[2])))
           for p, s in sorted(summary.items())])
    count = defaultdict(int)
    for r in em:
        count[r[3]] += 1
    print(f'diff {a.base} -> HEAD: {len(head)} programs x {len(a.archs)} archs x {len(a.faces)} faces: '
          + ' '.join(f'{k}={v}' for k, v in sorted(count.items()))
          + f'; {len(summary)} programs to measure; differing arms: '
          + ' '.join(f'{t}={sum(r[5] == t for r in arms)}' for t in ('new', 'cause', 'carried')) + f' -> {out}/changed.tsv')


def write(path, header, rows):
    path.write_text('\n'.join('\t'.join(map(str, r)) for r in [header, *rows]) + '\n')


def read(path):
    lines = path.read_text().splitlines()[1:] if path.exists() else []
    return [l.split('\t') for l in lines if l]


def env(a):
    """integrated_bundle.sh: SUITE (pt_*, default face) / SUITE_C (contract face too) / PROGS (pt_rm_*) / CELLS
    (mm_MxNxK) / ARMS (their `new` and `cause` arms; ALL_ARMS=1 adds the `carried` ones) / SUITE_CPU (the suite
    programs whose CPU-only build changed: their `cpu` arm); bundle.sh: ONLY (every other program: the differential)
    with its differing rungs as HOST_RUNGS / KERNEL_RUNGS / TF32_RUNGS (NOARMS where it has none) and DEFAULT (those
    whose default face moved: gated against the exact oracle within D2's bound, box_run.sh `_d`; UNCHECKED where no
    K bounds it — changed_bundle.sh stops on those unless UNCHECKED_OK=1). CPU-only changes
    are measured on the `--cpu-box` part alone."""
    out, arch = Path(a.out), a.arch
    cpu_box = a.cpu_box or arch
    moved = defaultdict(set)  # program -> faces changed on this arch (`cpu` for the CPU-only build)
    for p, ar, face, st, _, _ in read(out / 'emission.tsv'):
        if st in ('changed', 'new') and (ar == arch or (ar == CPU and arch == cpu_box)):
            moved[p].add(face if ar == arch else 'cpu')
    suite, suite_c, suite_cpu, progs, cells, only, default, unchecked, arms, rungs_only = \
        [], [], [], [], [], [], [], [], [], set()
    for p, faces in sorted(moved.items()):
        if p.startswith('pt_'):
            if not faces & {'default', 'contract', 'cpu'}:
                continue
            (progs if p.startswith('pt_rm_m') else suite).append(p)
            if 'contract' in faces and not p.startswith('pt_rm_m'):
                suite_c.append(p)
            if 'cpu' in faces and not p.startswith('pt_rm_m'):
                suite_cpu.append(p)
        elif re.fullmatch(r'mm_\d+x\d+x\d+', p):
            cells.append(p[3:])
        elif faces & {'exact', 'contract', 'default'}:
            only.append(p)
            if 'default' in faces:  # D2's bound needs the program's K (gen_tiles.py `.kmax`)
                (default if (out / 'tiles' / f'{p}.kmax').exists() else unchecked).append(p)
    timed = set(suite) | set(progs) | {f'mm_{c}' for c in cells}
    every = os.environ.get('ALL_ARMS') == '1'
    for p, ar, face, arm, _, tag in read(out / 'arms.tsv'):
        if ar != arch or (tag == 'carried' and not every):
            continue
        d, r = arm.split('-', 1)
        if p in timed and face in ('default', 'contract'):
            arms.append(f'{arch}:{p}:{"_c" if face == "contract" else ""}:{r}:{d}')
        elif p in only and face in ('exact', 'contract') and d == 'off':
            rungs_only.add(r)
    rr = ' '.join(sorted(rungs_only)) or ' '  # a blank, not empty: bundle.sh's `${X:-default}` keeps it
    noarms = '' if rungs_only else ' '.join(only)
    for k, v in [('SUITE', suite), ('SUITE_C', suite_c), ('SUITE_CPU', suite_cpu), ('PROGS', progs),
                 ('CELLS', cells), ('ARMS', arms), ('ONLY', only), ('DEFAULT', default), ('UNCHECKED', unchecked),
                 ('NOARMS', noarms.split())]:
        print(f"{k}='{' '.join(v)}'")
    for k in ('HOST_RUNGS', 'KERNEL_RUNGS', 'TF32_RUNGS'):
        print(f"{k}='{rr}'")


def box_stamp(text):
    """(GPU name|driver, nvcc release, torch version) off a box_run.sh results file."""
    gpu = re.search(r'^name, driver_version.*\n([^,\n]+), ([^,\n]+)', text, re.M)
    nvcc = (re.findall(r'Cuda compilation tools, release ([\d.]+)', text) or ['?'])[0]
    tv = (re.findall(r'^torch version=(\S+)', text, re.M) or ['?'])[0]
    return (f'{gpu[1].strip()}|{gpu[2].strip()}' if gpu else '?'), nvcc, tv


def src_hash(out, prog):
    f = Path(out) / 'tiles' / f'{prog}.mapal'
    return hashlib.sha256(f.read_bytes()).hexdigest()[:16] if f.exists() else f'src:{prog}'


def ingest(a):
    """Probe (`probe ... out=[.. iter ms=X]`), cell (`time leg=<stem>`), torch and chain rows -> the cache."""
    out, arch, text = Path(a.out), a.arch, Path(a.results).read_text(errors='replace')
    head = {(p, face): h for p, ar, face, _, _, h in read(out / 'emission.tsv') if ar == arch}
    cpu = {(p, face): h for p, ar, face, _, _, h in read(out / 'emission.tsv') if ar == CPU}
    arm_h = {(p, face, arm): h for p, ar, face, arm, h, _ in read(out / 'arms.tsv') if ar == arch}
    stamp, nvcc, tv = box_stamp(text)
    rt = (out / 'stamp').read_text().strip() if (out / 'stamp').exists() else '-'
    ver = f'{rt}|{nvcc}|{stamp.split("|")[-1]}'  # runtime stamp | nvcc | driver

    def key(p, face, arm):
        if arm in ('on', 'null'):
            return head.get((p, face))
        if arm == 'cpu':
            return cpu.get((p, face))
        return arm_h.get((p, face, arm)) or (f'{head[(p, face)]}/{arm}' if (p, face) in head else None)

    ms = defaultdict(list)
    for m in re.finditer(r'^probe prog=(\S+) face=(\S+) arm=(\S+) rep=\d+ .*?out=\[[^\]]*iter ms=([\d.]+)', text, re.M):
        ms[(m[1], RESULT_FACE.get(m[2], m[2]), m[3])].append(float(m[4]))
    lib = defaultdict(list)
    for m in re.finditer(r'^pass=\d+ cell=(\S+) time leg=(\S+) .*med_ms=([\d.]+)', text, re.M):
        s = re.fullmatch(r'(mm_\d+x\d+x\d+)(_x|_c|_r|_rc)?(?:\.(\S+))?', m[2])
        if s:
            face = {'_x': 'exact', '_c': 'contract', None: 'default'}.get(s[2], s[2])
            ms[(s[1], face, s[3] or 'on')].append(float(m[3]))
        else:
            lib[(m[1], m[2], nvcc)].append(float(m[3]))  # cell= carries mm_ (box_run.sh)
    for m in re.finditer(r'^torch prog=(\S+) mode=(\S+) .*med_ms=([\d.]+)', text, re.M):
        lib[(m[1], f'torch-{m[2]}', tv)].append(float(m[3]))
    for m in re.finditer(r'^chain probe=(\S+) n=(\d+) leg=(\S+) .*med_ms=([\d.]+)', text, re.M):
        lib[(f'pt_{m[1]}_{m[2]}', f'chain-{m[3]}', nvcc)].append(float(m[4]))
    rows = []
    for (p, face, arm), v in ms.items():
        k = key(p, face, arm)
        if k:
            rows.append((k, 'mapal-null' if arm == 'null' else 'mapal', ver,
                         f'med_ms={statistics.median(v):.6g} n={len(v)} prog={p} face={face} arm={arm}', stamp))
    for (p, leg, ver), v in lib.items():
        rows.append((src_hash(out, p), leg, ver, f'med_ms={statistics.median(v):.6g} n={len(v)} prog={p}', stamp))
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f'{arch}.tsv'
    new = not f.exists()
    with f.open('a') as fh:
        if new:
            fh.write('key\tleg\tver\tvalue\tstamp\n')
        fh.writelines('\t'.join(r) + '\n' for r in rows)
    print(f'ingest {arch}: {len(rows)} rows ({sum(r[1].startswith("mapal") for r in rows)} Mapal) -> {f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('diff')
    d.add_argument('out')
    d.add_argument('--base', default='recurrence-precision-latest')
    d.add_argument('--base-emit', help='a prebuilt base emitter (skips the build; programs from HEAD)')
    d.add_argument('--archs', nargs='+', default=ARCHS)
    d.add_argument('--faces', nargs='+', default=list(FACES), choices=list(FACES))
    d.add_argument('--rungs', nargs='+', help='the rungs whose arms to probe (default: every rung)')
    d.add_argument('--jobs', type=int)
    e = sub.add_parser('env')
    e.add_argument('out')
    e.add_argument('arch')
    e.add_argument('--cpu-box', help='the one part that measures CPU-only changes (default: this one)')
    i = sub.add_parser('ingest')
    i.add_argument('out')
    i.add_argument('arch')
    i.add_argument('results')
    a = ap.parse_args()
    {'diff': diff, 'env': env, 'ingest': ingest}[a.cmd](a)


if __name__ == '__main__':
    main()
