#!/usr/bin/env python3
"""site-geometry A2: fail a bundle on THIS machine, before renting, for anything box_run.sh would stop on.

Per part (box_run.sh's own rules, read off the bundle):
  - the harness: x86-64 ELF libmapal_rt.a, mapal_gpu.c, tile_bench.cu, chain_bench.cu, box_run.sh, the CUTLASS
    headers (cutlass/include/cutlass), value_gate.py where a probe answers to D2's bound, torch_bench.py unless
    TORCH=0;
  - the part directory with its host objects, and a non-empty launch line for every cell arm;
  - every differential program and arm has its face's oracle (tiles/<prog>[_c].expect; a TF32 one its off-epilogue);
  - every probe face has its gate: the default face D2's bound (tiles/<b>_x.expect + K) or tiles/<b>.expect,
    the others tiles/<b><face>.expect or their CPU build;
  - every probe has a library leg: chain_bench (box_run.sh's probe list) or a torch_bench.py workload
    (NO_LEG="<glob>.." names programs measured without one).
usage: preflight.py <bundle> <profile>...   (exit 1 on any finding; vast_run.sh runs it before renting)
"""
import fnmatch
import os
import re
import sys
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent


def nonempty(path):
    """box_run.sh's `-s` (an empty oracle or K fails there as "no bound")."""
    return path.exists() and path.stat().st_size > 0


def elf_x86_64(path):
    data = path.read_bytes()[:1 << 20]
    i = data.find(b'\x7fELF')
    return i >= 0 and data[i + 18:i + 20] == b'\x3e\x00'


def torch_workload(bundle):
    """stem -> True when torch_bench.py maps it (its workload() run with torch stubbed: construction only)."""
    src = bundle / 'torch_bench.py' if (bundle / 'torch_bench.py').exists() else HERE / 'torch_bench.py'
    ns = {'__name__': 'torch_bench'}
    with mock.patch.dict(sys.modules, {'torch': mock.MagicMock()}):
        exec(compile(src.read_text(), str(src), 'exec'), ns)

    def maps(stem):
        try:  # torch_bench.main's own aliasing
            ns['workload'](re.sub(r'^pt_gen_', 'pt_prefill_', stem.removesuffix('_tf32')))
            return True
        except Exception:  # noqa: BLE001 — any failure is "no leg"
            return False
    return maps


def check(bundle, prof):
    bad = []
    part = bundle / prof
    tiles = bundle / 'tiles'
    torch_on = os.environ.get('TORCH', '1') == '1'
    no_leg = os.environ.get('NO_LEG', '').split()
    lib = bundle / 'libmapal_rt.a'
    if not lib.exists() or not elf_x86_64(lib):
        bad.append('libmapal_rt.a missing or not x86-64 ELF (cargo build --release -p mapal-rt '
                   '--target x86_64-unknown-linux-gnu)')
    for f in ('mapal_gpu.c', 'tile_bench.cu', 'chain_bench.cu', 'box_run.sh'):
        if not (bundle / f).exists():
            bad.append(f'{f} missing')
    if not (bundle / 'cutlass/include/cutlass').is_dir():
        bad.append('cutlass/include/cutlass missing (the CUTLASS legs and every harness build)')
    objs = sorted(p.name[:-len('.host.o')] for p in part.glob('*.host.o')) if part.is_dir() else []
    if not objs:
        bad.append(f'{prof}/ missing or holds no host objects')
    for l in part.glob('mm_*.launch'):
        if not l.read_text().split():
            bad.append(f'{prof}/{l.name} empty (tile_bench reads kernel grid block smem from it)')
    probes = set()
    for b in objs:
        stem = b.split('.')[0]
        if b.startswith(('mm_', 'e2e_', 'loop_mm_')):
            continue
        if b.startswith('pt_'):
            if '.' not in b and not b.endswith(('_c', '_x')):
                probes.add(b)
            continue
        base = stem.removesuffix('_c')
        exp = tiles / (f'{base}_c.expect' if stem != base else f'{base}.expect')
        dflt = stem.endswith('_d') and nonempty(tiles / f'{stem[:-2]}.kmax')  # box_run.sh `_d`: D2's bound
        if dflt:
            exp = tiles / f'{stem[:-2]}.expect'
            if not (bundle / 'value_gate.py').exists():
                bad.append(f'{b}: default face without value_gate.py')
        if stem.endswith('_tf'):
            if f'{stem}.off-epilogue' not in objs:
                bad.append(f'{b}: TF32 program without its off-epilogue reference arm')
        elif not nonempty(exp):
            bad.append(f'{b}: missing oracle tiles/{exp.name}')
    box = (bundle / 'box_run.sh').read_text() if (bundle / 'box_run.sh').exists() else ''
    chain = (re.findall(r'case \$probe in ([^)]+)\) ;;', box) or [''])[0].split('|')
    maps = torch_workload(bundle) if probes and torch_on else (lambda s: False)
    if any(nonempty(tiles / f'{b}_x.expect') for b in probes) and not (bundle / 'value_gate.py').exists():
        bad.append('value_gate.py missing (the default face\'s bound)')
    if probes and torch_on and not (bundle / 'torch_bench.py').exists():
        bad.append('torch_bench.py missing (TORCH=0 to run without the torch legs)')
    for b in sorted(probes):
        kmax = nonempty(tiles / f'{b}.kmax') or re.search(r'_d\d+_h\d+', b)
        if not (nonempty(tiles / f'{b}_x.expect') and kmax) and not nonempty(tiles / f'{b}.expect'):
            bad.append(f'{b}: default face without its bound (tiles/{b}_x.expect + K, or tiles/{b}.expect)')
        for face in ('_c', '_x'):
            if any(o == b + face or o.startswith(f'{b}{face}.') for o in objs) and \
                    not nonempty(tiles / f'{b}{face}.expect') and f'{b}{face}.cpu' not in objs:
                bad.append(f'{b}{face}: no oracle (tiles/{b}{face}.expect or {b}{face}.cpu)')
        probe, n = b[3:].rsplit('_', 1)
        has_chain = n.isdigit() and any(fnmatch.fnmatchcase(probe, g) for g in chain if g)
        if not (has_chain or maps(b) or any(fnmatch.fnmatchcase(b, g) for g in no_leg)):
            bad.append(f'{b}: no library leg (chain_bench or a torch_bench.py workload; NO_LEG to accept)')
    return bad, len(probes), len(objs)


def main():
    bundle, profs = Path(sys.argv[1]), sys.argv[2:]
    if not profs:
        sys.exit(__doc__)
    fails = 0
    for prof in profs:
        bad, np, no = check(bundle, prof)
        for b in bad:
            print(f'PREFLIGHT {prof}: {b}')
        print(f'preflight {prof}: {"FAILED (%d)" % len(bad) if bad else "OK"} — {np} probes, {no} host objects')
        fails += len(bad)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
