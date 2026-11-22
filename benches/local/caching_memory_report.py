#!/usr/bin/env python3
"""caching-memory: time and peak device memory per workload, from one or more box_run results files
(benches/results/evidence/local-rtx-5080/caching-memory.md). A later file's rows replace an earlier one's for the
same program and arm (torch: program and leg).

Mapal: each arm's `iter ms` (median over its processes) and its peak device bytes (`mapal_gpu_peak`, printed by the
traced processes: every cuMemAlloc less every cuMemFree, the high-water mark). Torch: each leg's median ms and
`peak_mb` (torch.cuda.max_memory_allocated over the leg, its tensors included, the harness's L2-flush buffer not),
gated legs only. f32 legs (`eager`, `compile`, `sdpa`, `naive*`) are the like-for-like competitors; the `bf16*` legs
are the real-world reference, reported apart.

usage: caching_memory_report.py <results.txt>... [--peaks <torch lines>...]
  (`--peaks`: files whose torch legs give only `peak_mb`, e.g. one process per leg — torch_peak_proc.sh)
"""
import re
import statistics
import sys
from collections import defaultdict

torch_peak = {}
PROBE = re.compile(r'^probe prog=(\S+) face=(\S+) arm=(\S+) rep=(\d+) .*?out=\[(.*?)\]')
PEAK = re.compile(r'mapal_gpu_peak bytes=(\d+)')
TORCH = re.compile(r'^torch prog=(\S+) mode=(\S+) .*?med_ms=([\d.]+) .*?within=(\S+) .*?peak_mb=([\d.na-]+)')


def load(paths, peaks=()):
    ms, peak, torch = {}, {}, defaultdict(dict)
    for p in peaks:  # torch peak overrides only
        for line in open(p, errors='replace'):
            m = TORCH.match(line)
            if m and m.group(4) == 'yes':
                torch_peak[(m.group(1), m.group(2))] = float(m.group(5))
    for p in paths:
        cur = defaultdict(list)
        for line in open(p, errors='replace'):
            m = PROBE.match(line)
            if m:
                prog, face, arm, _, out = m.groups()
                if face != '_default':
                    continue
                it = re.search(r'iter ms=([\d.e-]+)', out)
                if it:
                    cur[(prog, arm)].append(float(it.group(1)))
                pk = PEAK.search(line)
                if pk:
                    peak[(prog, arm)] = int(pk.group(1)) / 2 ** 20
                continue
            m = TORCH.match(line)
            if m:
                prog, mode, med, within, pmb = m.groups()
                if within == 'yes':
                    pk = torch_peak.get((prog, mode), float(pmb) if pmb not in ('nan', '-') else float('nan'))
                    torch[prog][mode] = (float(med), pk)
        for k, v in cur.items():
            ms[k] = statistics.median(v)
    return ms, peak, torch


def best(legs, keep):
    got = [(k, v) for k, v in legs.items() if keep(k)]
    return min(got, key=lambda kv: kv[1][0]) if got else None


def fmt(b):
    return (f'{b[1][0]:.2f} ({b[0]}) | {b[1][1]:.0f}') if b else '— | —'


def main(paths, peaks):
    ms, peak, torch = load(paths, peaks)
    f32 = lambda k: not k.startswith('bf16')  # noqa: E731
    shapes = sorted({re.sub(r'^pt_llm_kv[xn]_', '', p) for (p, _) in ms if p.startswith('pt_llm_kv')})
    print('| shape | Mapal explicit ms / peak MB | Mapal naive ms / peak MB | torch HF-cache f32 ms (leg) | its MB '
          '| torch naive ms (leg) | its MB | torch bf16 ms (leg) | its MB |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for sh in shapes:
        x, n = f'pt_llm_kvx_{sh}', f'pt_llm_kvn_{sh}'
        mx, mn = ms.get((x, 'on')), ms.get((n, 'on'))
        cell = lambda p, v: f'{v:.2f} / {peak.get((p, "on"), float("nan")):.0f}' if v else '—'  # noqa: E731
        print(f'| {sh} | {cell(x, mx)} | {cell(n, mn)} | {fmt(best(torch.get(x, {}), f32))} '
              f'| {fmt(best(torch.get(n, {}), f32))} | {fmt(best(torch.get(x, {}), lambda k: not f32(k)))} |')
    print()
    print('| workload | arm | Mapal ms | Mapal peak MB | vs on (time) | vs on (peak) |')
    print('|---|---|---:|---:|---:|---:|')
    for (prog, arm) in sorted(ms):
        if arm in ('null', 'c1', 'pick0', 'pick1', 'pick2'):
            continue
        on, pon = ms.get((prog, 'on')), peak.get((prog, 'on'))
        v, pk = ms[(prog, arm)], peak.get((prog, arm), float('nan'))
        rel = f'{v / on:.2f}x' if on and arm != 'on' else ''
        prel = f'{pk - pon:+.0f} MB' if pon is not None and arm != 'on' else ''
        print(f'| {prog} | {arm} | {v:.3f} | {pk:.0f} | {rel} | {prel} |')
    print()
    print('| workload | Mapal ms | Mapal peak MB | best torch f32 (leg) ms | its peak MB | bf16 ref ms (leg) | its MB '
          '| Mapal / torch f32 |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for prog in sorted({p for (p, a) in ms if a == 'on'}):
        v, pk = ms[(prog, 'on')], peak.get((prog, 'on'), float('nan'))
        b = best(torch.get(prog, {}), f32)
        r = best(torch.get(prog, {}), lambda k: not f32(k))
        ratio = f'{v / b[1][0]:.2f}' if b else ''
        print(f'| {prog} | {v:.3f} | {pk:.0f} | {fmt(b)} | {fmt(r)} | {ratio} |')


if __name__ == '__main__':
    args = sys.argv[1:]
    cut = args.index('--peaks') if '--peaks' in args else len(args)
    main(args[:cut], args[cut + 1:])
