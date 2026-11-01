#!/usr/bin/env python3
"""S65: per probe x arm on one box — the median `iter ms`, ON/arm per repetition (ABBA within the run), the null,
and the traced iteration's split (kernel / copies / host residue ms, launches, MB moved).

usage: s65_arms.py <results-<arch>.txt> [prog...]
"""
import re
import sys
from collections import defaultdict

med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None


def load(path):
    reps = defaultdict(dict)  # (prog, arm) -> rep -> iter ms
    trace = {}
    vals = {}
    for l in open(path, errors='replace'):
        if l.startswith('value-bound ') or l.startswith('value-gate '):
            f = dict(re.findall(r'(\w+)=([^\s\[\]]+)', l))
            vals.setdefault(f.get('prog'), l.strip()[:160])
        if not l.startswith('probe prog='):
            continue
        f = dict(re.findall(r'(\w+)=([^\s\[\]]+)', l))
        it = re.search(r'iter ms=\s*([-\d.eE+]+)', l)
        if not it or f.get('face') != '_default':
            continue
        reps[(f['prog'], f['arm'])][int(f['rep'])] = float(it[1])
        if 'mapal_gpu_trace_total' in l and (f['prog'], f['arm']) not in trace:
            ph = {k: float(f.get(k, 0)) for k in ('h2d_ms', 'kernel_ms', 'gen_ms', 'd2h_ms')}
            trace[(f['prog'], f['arm'])] = dict(
                residue=max(0.0, float(it[1]) - sum(ph.values())), kernel=ph['kernel_ms'] + ph['gen_ms'],
                copies=ph['h2d_ms'] + ph['d2h_ms'], launches=int(f.get('launches', 0)),
                h2d=int(f.get('h2d', 0)), d2h=int(f.get('d2h', 0)),
                mb=(int(f.get('bytes_in', 0)) + int(f.get('bytes_out', 0))) / 2 ** 20)
    return reps, trace, vals


def main(path, progs):
    reps, trace, _ = load(path)
    progs = progs or sorted({p for p, _ in reps})
    print('prog\tarm\tmedian_ms\tbase/arm per rep (median, min..max)\tnull\ttrace kernel/copies/residue ms, launches, h2d/d2h, MB')
    for p in progs:
        on = reps.get((p, 'on'), {})
        nul = reps.get((p, 'null'), {})
        null = max((abs(nul[r] / on[r] - 1) for r in nul if r in on), default=None)
        for (pp, arm), rs in sorted(reps.items()):
            if pp != p:
                continue
            base = reps.get((p, 'cpu'), {}) if arm.startswith('cpu-') else on
            ratios = [base[r] / rs[r] for r in rs if r in base]
            t = trace.get((p, arm))
            ts = (f"{t['kernel']:.2f}/{t['copies']:.2f}/{t['residue']:.2f}, {t['launches']}, {t['h2d']}/{t['d2h']}, "
                  f"{t['mb']:.1f}") if t else '-'
            rr = f'{med(ratios):.3f} ({min(ratios):.3f}..{max(ratios):.3f})' if ratios and arm != 'on' else '-'
            print(f"{p}\t{arm}\t{med(list(rs.values())):.4f}\t{rr}\t{'-' if null is None else f'{null:.3f}'}\t{ts}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
