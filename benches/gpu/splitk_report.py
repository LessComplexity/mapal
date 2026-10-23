#!/usr/bin/env python3
"""S62 `splitk` (ADR-0040): the per-cell table of a box's `tile_bench --splitk` rows.

Per cell and step face (`_r` exact steps, `_rc` fma steps), per process (pass):
ON = the derived split, OFF = `.off-splitk` (the ordered kernel), null = `.null`
(ON's PTX again: |null/ON - 1| is that process's noise). Keep iff ON/OFF <= 1 - 2·null
in both passes and every gate passed (bit-exact to the partitioned order, ADR-0040
bound, same output after timing, same FNV in both processes). Ratios are within one
process: ON/cuBLAS, ON/`_c` (Mapal Contract), and each forced-S arm over ON.

usage: splitk_report.py results-<prof>.txt
"""
import re
import sys
from collections import defaultdict

med = defaultdict(dict)        # (cell, pass) -> leg -> med_ms
gate = defaultdict(lambda: True)  # (cell, leg) -> every gate passed
fnv = defaultdict(set)          # (cell, leg) -> hashes seen (values + determinism, both passes)
S = {}                          # (cell, leg) -> S
progs = []
for line in open(sys.argv[1]):
    m = re.match(r"pass=(\d) cell=(\S+) (\w+) leg=(\S+) (.*)", line)
    if m:
        p, cell, kind, leg, rest = m.groups()
        if kind == "time":
            med[(cell, p)][leg] = float(re.search(r"med_ms=([\d.]+)", rest).group(1))
        if kind in ("values", "determinism") and not leg.startswith(("cublas", "cutlass")):
            gate[(cell, leg)] &= "NO" not in rest and "GATE-FAILED" not in rest
            h = re.search(r"fnv=(\w+)", rest)
            h and fnv[(cell, leg)].add(h.group(1))
            s = re.search(r"\bS=(\d+)", rest)
            s and S.setdefault((cell, leg), int(s.group(1)))
    elif line.startswith("splitk_program"):
        progs.append(line.strip())

cells = sorted({c for c, _ in med}, key=lambda c: [int(x) for x in c[3:].split("x")])
print("| cell | face | S | ON ms p1/p2 | OFF ms p1/p2 | null p1/p2 | ON/OFF p1/p2 | verdict | ON/cuBLAS | ON/_c | forced S: ms/ON |")
print("|---|---|---:|---|---|---|---|---|---|---|---|")
verdicts = []
for cell in cells:
    for f in ("_r", "_rc"):
        on = cell + f
        rows = [med[(cell, p)] for p in ("1", "2") if (cell, p) in med]
        if not rows or any(on not in r for r in rows):
            continue
        ok = all(gate[(cell, l)] and len(fnv[(cell, l)]) == 1 for l in rows[0] if l.startswith(on))
        fmt = lambda xs: "/".join(f"{x:.4f}" for x in xs)
        ons = [r[on] for r in rows]
        nulls = [abs(r[on + ".null"] / r[on] - 1) for r in rows if on + ".null" in r]
        offs = [r.get(on + ".off-splitk") for r in rows]
        if None in offs:
            verdict, ratio = "S=1 (no split)", "-"
        else:
            rs = [a / b for a, b in zip(ons, offs)]
            ratio = fmt(rs)
            keep = ok and len(nulls) == len(rs) and all(r <= 1 - 2 * n for r, n in zip(rs, nulls))
            lose = all(r > 1 + 2 * n for r, n in zip(rs, nulls))
            verdict = ("keep" if keep else "LOSS" if lose else "no stable effect") + ("" if ok else " (GATE FAILED)")
            verdicts.append((cell, f, verdict))
        forced = sorted((int(l.rsplit(".s", 1)[1]), l) for l in rows[0] if l.startswith(on + ".s"))
        fs = " ".join(f"S{s}:{fmt([r[l] / r[on] for r in rows if l in r])}" for s, l in forced)
        print(f"| {cell} | {f} | {S.get((cell, on), '?')} | {fmt(ons)} | {fmt(offs) if None not in offs else '-'} | "
              f"{fmt(nulls)} | {ratio} | {verdict} | {fmt([r[on] / r['cublas'] for r in rows])} | "
              f"{fmt([r[on] / r[cell + '_c'] for r in rows])} | {fs} |")
print()
for p in progs:
    print("-", p)
