#!/usr/bin/env python3
"""S62: per-cell arm ratios of one box_run session run with PASSES=2 (ABBA arm order).

Each tile_bench row is normalized by its own cuBLAS leg (r = mapal_med / cublas_med; cancels the
clock drift between benches, rungs.py's convention). For a cell, face and pass:
  s(<arm>) = r(<arm>) / r(shipped)        > 1 = the shipped kernel is faster than that arm
  null     = |r(null) / r(shipped) - 1|   the shipped text benched again: this box's noise floor
A cell's arm absent from the bundle (identical PTX: bundle.sh drops it) prints `=`.

usage: python3 benches/gpu/abba.py <results-<profile>.txt> [arm ...]
       (default arms: off-asyncstage off-cell-fill+asyncstage)
"""
import re
import sys
from collections import defaultdict


def rows(path):
    """{(cell, face, arm, pass): {leg: med_ms}} plus the launch and value lines of each bench."""
    out, cur = defaultdict(dict), None
    for line in open(path):
        m = re.match(r'# n=(\S+) face=(_exact|_c|_tf)(?:\.(\S+))? pass=(\d+)( l2=warm)?', line)
        if m:
            cur = (m[1] + ('@warm' if m[5] else ''), m[2], m[3] or 'shipped', int(m[4]))
            continue
        m = re.match(r'time leg=(\w+) .* med_ms=([\d.]+)', line)
        if m and cur:
            out[cur][m[1]] = float(m[2])
        m = re.match(r'mapal kernel=\S+ grid=(\S+) block=(\d+) smem=(\d+) regs=(\d+) local_bytes=(\d+)', line)
        if m and cur:
            out[cur]['launch'] = f"{m[1]}/{m[2]}/{m[3]} regs={m[4]} local={m[5]}"
        m = re.match(r'values leg=mapal .* bit_exact_vs_in_order_fold=(\w+)', line)
        if m and cur:
            out[cur]['exact'] = m[1]
    return out


def main(path, arms):
    rs = rows(path)
    r = lambda key: rs[key]['mapal'] / rs[key]['cublas'] if key in rs and 'cublas' in rs[key] else None
    cells = sorted({(c, f) for c, f, _, _ in rs},
                   key=lambda x: ([int(v) for v in x[0].split('@')[0].split('x')], x[0], x[1]))
    passes = sorted({p for *_, p in rs})
    nulls = []
    print(f"| cell | face | pass | shipped ms | cuBLAS ms | CUTLASS ms | shipped/cuBLAS | "
          + " | ".join(f"s({a})" for a in arms) + " | null |")
    print("|---|---|---|---:|---:|---:|---:|" + "---:|" * (len(arms) + 1))
    for c, f in cells:
        for p in passes:
            on = rs.get((c, f, 'shipped', p))
            if not on or 'mapal' not in on:
                continue
            base = r((c, f, 'shipped', p))
            ss = []
            for a in arms:
                x = r((c, f, a, p))
                ss.append('=' if x is None else f"{x / base:.3f}")
            n = r((c, f, 'null', p))
            nl = '' if n is None else f"{abs(n / base - 1):.3f}"
            if n is not None:
                nulls.append(abs(n / base - 1))
            print(f"| {c} | {f} | {p} | {on['mapal']:.4f} | {on['cublas']:.4f} | {on.get('cutlass', 0):.4f} | "
                  f"{base:.3f} | " + " | ".join(ss) + f" | {nl} |")
    if nulls:
        print(f"\nnull: max {max(nulls):.3f}, median {sorted(nulls)[len(nulls) // 2]:.3f} over {len(nulls)} benches")
    bad = [k for k, v in rs.items() if v.get('exact') not in (None, 'yes')]
    print(f"value gate: {'every Mapal leg bit-exact' if not bad else 'NOT EXACT ' + str(bad)}")


def decide(path, fills_path, warm=False):
    """The acceptance rule per (rung, fill, face): a cell is stable+ when s > 1 + t in both passes,
    stable- when s < 1 - t in both, t = max(the cell's null in either pass, 0.01). A class ships ON
    iff some cell is stable+ and none stable-. asyncstage: s = r(off-asyncstage) / r(shipped);
    cell-fill: s = r(off-cell-fill+asyncstage) / r(off-asyncstage) (r(shipped) when asyncstage did
    not fire). Fill = the cell's base-grid class (fill-classes.txt, the GPU_RUNGS_OFF key)."""
    rs = rows(path)
    prof = re.match(r'== (\S+)', open(path).readline())[1]
    fill = {}
    for line in open(fills_path):
        f = line.split()
        if len(f) == 4 and f[0] == prof:
            fill[f[1]] = f[3]
    r = lambda key: rs[key]['mapal'] / rs[key]['cublas'] if key in rs and 'cublas' in rs[key] else None
    cls = defaultdict(list)
    for c, f in sorted({(c, f) for c, f, _, _ in rs}):
        if c.endswith('@warm') != warm:
            continue
        base = c.split('@')[0]
        dims = base if 'x' in base else f"{base}x{base}x{base}"
        ss = defaultdict(list)
        t = 0.01
        for p in (1, 2):
            on, nl = r((c, f, 'shipped', p)), r((c, f, 'null', p))
            if on is None:
                continue
            if nl is not None:
                t = max(t, abs(nl / on - 1))
            a, b = r((c, f, 'off-asyncstage', p)), r((c, f, 'off-cell-fill+asyncstage', p))
            if a is not None:
                ss['asyncstage'].append(a / on)
            # cell-fill applies only where the two arms launch differently (M128: one text).
            same = a is not None and rs[(c, f, 'off-asyncstage', p)].get('launch', '').split(' regs')[0] == \
                rs.get((c, f, 'off-cell-fill+asyncstage', p), {}).get('launch', '').split(' regs')[0]
            if b is not None and not same:
                ss['cell-fill'].append(b / (a if a is not None else on))
        for rung, v in ss.items():
            sign = '+' if all(x > 1 + t for x in v) else '-' if all(x < 1 - t for x in v) else '~'
            cls[(rung, fill.get(dims, '?'), {'_exact': 'Exact', '_c': 'Contract'}.get(f, f))].append(
                f"{c}{sign}[{'/'.join(f'{x:.3f}' for x in v)} t={t:.3f}]")
    print(f"\n{prof}{' (warm L2)' if warm else ''}: | rung | fill | face | ships | cells |")
    for k, cells in sorted(cls.items()):
        pos = any('+[' in x for x in cells)
        neg = any('-[' in x for x in cells)
        verdict = 'ON' if pos and not neg else ('OFF (mixed)' if pos and neg else 'OFF')
        print(f"| {k[0]} | {k[1]} | {k[2]} | {verdict} | {'; '.join(cells)} |")


def selftest():
    import os
    import tempfile
    bench = lambda key, p, ms, cub: (f"# n=16x2048x7168 face={key} pass={p}\n"
                                     f"time leg=mapal M=16 n=30 min_ms={ms} med_ms={ms} max_ms={ms}\n"
                                     f"time leg=cublas M=16 n=30 min_ms={cub} med_ms={cub} max_ms={cub}\n")
    f = tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False)
    f.write(bench('_c', 1, 1.0, 0.5) + bench('_c.off-asyncstage', 1, 3.0, 1.0) + bench('_c.null', 1, 1.02, 0.5))
    f.close()
    got = rows(f.name)
    assert got[('16x2048x7168', '_c', 'off-asyncstage', 1)]['mapal'] == 3.0
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(f.name, ['off-asyncstage'])
    os.unlink(f.name)
    assert '| 16x2048x7168 | _c | 1 | 1.0000 | 0.5000 | 0.0000 | 2.000 | 1.500 | 0.020 |' in buf.getvalue(), buf.getvalue()
    print('selftest ok')


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest()
    elif sys.argv[1] == '--decide':  # --decide <results> <fill-classes.txt> [--warm]
        decide(sys.argv[2], sys.argv[3], sys.argv[4:] == ['--warm'])
    else:
        main(sys.argv[1], sys.argv[2:] or ['off-asyncstage', 'off-cell-fill+asyncstage'])
