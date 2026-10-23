"""Final-build ON/OFF from a run's arms: ON = the arm byte-identical to the final emission."""
import re, sys
from collections import defaultdict
med = lambda xs: sorted(xs)[len(xs) // 2]
def load(p):
    runs = defaultdict(list)
    for l in open(p, errors='replace'):
        if l.startswith('probe prog='):
            f = dict(kv.split('=', 1) for kv in re.findall(r'(\w+=[^\s\[\]]+)', l))
            it = re.search(r'iter ms=\s*([-\d.eE+]+)', l)
            if it: runs[(f['prog'], f['face'], f['arm'])].append((float(it[1]), int(f['wall_us'])))
    return runs
path, *specs = sys.argv[1:]   # specs: prog:on_arm
runs = load(path)
print(f"## {path}")
print("| program | face | ON arm | ON iter ms | ON wall ms | off-place/ON iter (wall) | off-hostopt/ON iter (wall) | cpu/ON iter (wall) | null/null-src iter | n |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
for spec in specs:
    p, onarm = spec.split(':')
    for fa in ('_exact', '_c'):
        on = runs.get((p, fa, onarm))
        if not on: continue
        oi, ow = med([x[0] for x in on]), med([x[1] for x in on])
        def r(a):
            xs = runs.get((p, fa, a))
            if not xs: return '-'
            return f"{med([x[0] for x in xs]) / oi:.3f} ({med([x[1] for x in xs]) / ow:.2f})"
        nl = runs.get((p, fa, 'null')); src = runs.get((p, fa, 'on'))
        nul = f"{med([x[0] for x in nl]) / med([x[0] for x in src]):.3f}" if nl and src else '-'
        print(f"| {p} | {fa} | {onarm} | {oi:.4f} | {ow/1000:.1f} | {r('off-place')} | {r('off-hostopt')} | {r('cpu')} | {nul} | {len(on)} |")
