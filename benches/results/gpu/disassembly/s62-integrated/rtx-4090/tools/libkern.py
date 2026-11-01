#!/usr/bin/env python3
"""s62 disasm (rtx-4090): per-leg kernel-only GPU time of mlp_bench under nsys (reps=3).
Kernel launches in start order segment by the observer's per-call kernel count per leg:
first call x4 legs, gate x4, warm-up 3 per leg back to back (L2 warm), then 3 timed reps in
ABBA order after a 256 MiB flush (L2 cold). Prints per leg: launches/call, kernel-sum per call
(median of the warm-ups / median of the timed reps), and the GPU span of one timed call.
usage: libkern.py <mlplib csv> <obs dir> <prog>"""
import csv, io, statistics, sys
from pathlib import Path
p, obs, prog = sys.argv[1:4]
blocks, cur = [], []
for line in Path(p).read_text(errors="replace").splitlines():
    if not line.strip():
        if cur: blocks.append(cur); cur = []
    else: cur.append(line)
if cur: blocks.append(cur)
gpu = next(list(csv.DictReader(io.StringIO("\n".join(b)))) for b in blocks if "GrdX" in b[0])
ks = sorted((r for r in gpu if r["GrdX"]), key=lambda r: int(r["Start (ns)"]))
legs = ["eager", "lt", "cutlass", "cutlass_tuned"]
n = {l: len(Path(obs, f"mlp.{prog}.{l}", "kernels.tsv").read_text().splitlines()) - 1 for l in legs}
calls = []  # (leg, phase, rows)
i = 0
def take(leg, phase):
    global i
    calls.append((leg, phase, ks[i:i + n[leg]])); i += n[leg]
for l in legs: take(l, "first")
for l in legs: take(l, "gate")
for l in legs:
    for _ in range(3): take(l, "warm")
for r in range(3):
    for l in (legs if r % 2 == 0 else legs[::-1]): take(l, "timed")
assert i == len(ks), (i, len(ks))
ksum = lambda rows: sum(int(x["Duration (ns)"]) for x in rows) / 1e3
span = lambda rows: (int(rows[-1]["Start (ns)"]) + int(rows[-1]["Duration (ns)"]) - int(rows[0]["Start (ns)"])) / 1e3
for l in legs:
    w = [ksum(r) for (a, ph, r) in calls if a == l and ph == "warm"]
    t = [ksum(r) for (a, ph, r) in calls if a == l and ph == "timed"]
    s = [span(r) for (a, ph, r) in calls if a == l and ph == "timed"]
    names = " + ".join(x["Name"].split("<")[0].split("(")[0].replace("void ", "")[:28] for x in calls[-1][2]) if False else ""
    print(f"{prog} {l} launches={n[l]} kernel_us_warm={statistics.median(w):.1f} kernel_us_flushed={statistics.median(t):.1f} gpu_span_us_flushed={statistics.median(s):.1f}")
