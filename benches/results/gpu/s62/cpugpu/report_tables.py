"""REPORT tables from run1: dual (= S54 GPU build, byte-identical) vs CPU-only, per arch."""
import re, sys
from collections import defaultdict
med = lambda xs: sorted(xs)[len(xs) // 2]
R = sys.argv[1]
files = {'rtx-4090': 'run1/results-rtx-4090.txt', 'rtx-5090': 'run1/results-rtx-5090.txt',
         'h100': 'run1/results-h100-pcie.txt', 'rtx-3090': 'run1/results-rtx-3090.partial-51025609.txt'}
data, dec, lib = {}, {}, {}
for a, f in files.items():
    runs = defaultdict(list); d = defaultdict(list); lb = {}
    for l in open(f"{R}/{f}", errors='replace'):
        if l.startswith('probe prog='):
            kv = dict(x.split('=', 1) for x in re.findall(r'(\w+=[^\s\[\]]+)', l))
            it = re.search(r'iter ms=\s*([-\d.eE+]+)', l)
            if it: runs[(kv['prog'], kv['face'], kv['arm'])].append((float(it[1]), int(kv['wall_us']) / 1000))
        elif l.startswith('decode prog='):
            kv = dict(x.split('=', 1) for x in re.findall(r'(\w+=[^\s\[\]]+)', l))
            d[kv['prog']].append(float(kv['wall_s']))
        elif l.startswith('chain probe=mlp2 n=4096'):
            kv = dict(x.split('=', 1) for x in re.findall(r'(\w+=[^\s\[\]]+)', l))
            lb[kv['leg']] = float(kv['med_ms'])
    data[a], dec[a], lib[a] = runs, d, lb
progs = sorted({p for r in data.values() for (p, _, _) in r})
print("| program | face | " + " | ".join(f"{a} dual iter ms | {a} cpu/dual iter | {a} cpu/dual wall" for a in files) + " |")
print("|---|---|" + "---:|" * (3 * len(files)))
for p in progs:
    for fa in ('_exact', '_c'):
        cells = []
        for a in files:
            on, cpu = data[a].get((p, fa, 'on')), data[a].get((p, fa, 'cpu'))
            if not on or not cpu:
                cells += ['-'] * 3; continue
            oi, ow = med([x[0] for x in on]), med([x[1] for x in on])
            cells += [f"{oi:.4f}", f"{med([x[0] for x in cpu]) / oi:.2f}", f"{med([x[1] for x in cpu]) / ow:.3f}"]
        print(f"| {p} | {fa.strip('_')} | " + " | ".join(cells) + " |")
print()
print("| decode_1024_256 wall s (median of 5) | " + " | ".join(files) + " |")
print("|---|" + "---:|" * len(files))
for arm in ('decode_1024_256', 'decode_1024_256.cpu', 'decode_1024_256_c', 'decode_1024_256_c.cpu'):
    print(f"| {arm} | " + " | ".join(f"{med(dec[a][arm]):.3f}" if dec[a].get(arm) else '-' for a in files) + " |")
print()
print("| mlp2_4096 (contract) ms | " + " | ".join(files) + " |")
print("|---|" + "---:|" * len(files))
row = lambda name, f: print(f"| {name} | " + " | ".join(f(a) for a in files) + " |")
row("Mapal dual iter ms (median)", lambda a: f"{med([x[0] for x in data[a][('pt_mlp2_4096', '_c', 'on')]]):.3f}" if data[a].get(('pt_mlp2_4096', '_c', 'on')) else '-')
for leg in ('eager', 'lt', 'lt2', 'cutlass'):
    row(f"chain_bench {leg} med_ms", lambda a, leg=leg: f"{lib[a][leg]:.3f}" if leg in lib[a] else '-')
