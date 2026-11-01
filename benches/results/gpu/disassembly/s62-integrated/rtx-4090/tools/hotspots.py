#!/usr/bin/env python3
"""s62 disasm (rtx-4090): whole-MLP hotspots from `nsys stats --format csv -r
cuda_api_trace,cuda_gpu_trace,...` of one Mapal program process (box_disasm.sh section 4).
The computation = from the end of the first cuCtxSynchronize (operands generated on the
device) to the end of the last cuCtxSynchronize before the result copies: the span the
program's own `iter ms` clocks. Inside it: the host gap before the first tile launch (no
traced CUDA call: the runtime's cuModuleGetFunction/cuFuncSetAttribute, where lazy loading
puts the function load), launch API time, gaps between launches, sync wait, and the GPU
kernels. Outside it: module load (JIT), alloc, copies, free.
usage: hotspots.py <csv>... -> one markdown row per file"""
import csv, io, re, sys
from pathlib import Path

def tables(p):
    out, cur = [], []
    for line in Path(p).read_text(errors="replace").splitlines():
        if not line.strip():
            if cur: out.append(cur); cur = []
            continue
        cur.append(line)
    if cur: out.append(cur)
    return [list(csv.DictReader(io.StringIO("\n".join(t)))) for t in out]

print("| process | iter ms (this nsys run) | span µs | host gap before 1st tile launch µs | launch API µs (n) | gaps between tile launches µs | final sync µs | tile kernels GPU µs (n) | kernel share of span | outside span: module load / alloc / free ms |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for p in sys.argv[1:]:
    t = tables(p)
    api = next(x for x in t if x and "Name" in x[0] and "Result" in x[0])
    gpu = next(x for x in t if x and "GrdX" in x[0])
    syncs = [r for r in api if r["Name"] == "cuCtxSynchronize"]
    s0 = int(syncs[0]["Start (ns)"]) + int(syncs[0]["Duration (ns)"])
    last = syncs[-1]; s1 = int(last["Start (ns)"]) + int(last["Duration (ns)"])
    launches = [r for r in api if r["Name"].startswith("cuLaunchKernel") and s0 <= int(r["Start (ns)"]) <= s1]
    gap0 = int(launches[0]["Start (ns)"]) - s0 if launches else 0
    lapi = sum(int(r["Duration (ns)"]) for r in launches)
    between = sum(int(b["Start (ns)"]) - int(a["Start (ns)"]) - int(a["Duration (ns)"]) for a, b in zip(launches, launches[1:]))
    tiles = [r for r in gpu if r["Name"].startswith("mapal_tile") and s0 <= int(r["Start (ns)"]) <= s1]
    kern = sum(int(r["Duration (ns)"]) for r in tiles)
    span = s1 - s0
    tot = lambda n: sum(int(r["Duration (ns)"]) for r in api if r["Name"].startswith(n)) / 1e6
    out = Path(p).with_suffix(".out")
    it = re.search(r"iter ms=([0-9.]+)", out.read_text()) if out.exists() else None
    name = Path(p).stem.removeprefix("mlp_")
    print(f"| {name} | {float(it.group(1)):.4f} | {span/1e3:.1f} | {gap0/1e3:.1f} | {lapi/1e3:.1f} ({len(launches)}) | {between/1e3:.1f} | "
          f"{int(last['Duration (ns)'])/1e3:.1f} | {kern/1e3:.1f} ({len(tiles)}) | {kern/span*100:.0f}% | "
          f"{tot('cuModuleLoadData'):.2f} / {tot('cuMemAlloc'):.2f} / {tot('cuMemFree'):.2f} |" if it else "")
