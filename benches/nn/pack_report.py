#!/usr/bin/env python3
"""plan-s51 H1 reporter — the B-pack A/B, with its noise floor in the same table.

Reads `P_<shape>_<d>_<arm>_<width>` sample files written by `pack_ab.sh`.

Two things this prints that a bare ratio does not:

  * ON vs CTL — two byte-identical binaries. That column IS this run's noise
    floor (S39 measured -5.9%..+1.2% between byte-identical binaries here), and
    any ON/OFF effect not larger than it is not an effect.
  * DISJOINT vs OVERLAP — whether the distributions actually separate. The
    pre-registered thresholds in plan-s51 §3 are stated on disjointness, not on
    medians, so the medians alone cannot settle them.
"""

import sys
from pathlib import Path
from statistics import median


def load(path):
    vals = [float(x) for x in path.read_text().split()]
    return vals or None


def cell_key(name):
    # P_<shape>_<d>_<arm>_<width>
    _, shape, d, arm, width = name.split("_")
    return (shape, int(d)), arm, width


def sep(a, b):
    """Disjoint if the two ranges do not overlap at all."""
    return "DISJOINT" if max(a) < min(b) or max(b) < min(a) else "overlap"


def main(outdir):
    out = Path(outdir)
    series = {}
    for p in sorted(out.glob("P_*")):
        v = load(p)
        if v:
            key, arm, width = cell_key(p.name)
            series.setdefault((key, width), {})[arm] = v
    if not series:
        print("no samples")
        return 1

    for width in ("1t", "par"):
        rows = {k: a for (k, w), a in series.items() if w == width}
        if not rows:
            continue
        print(f"\n== {width} ==")
        print(f"{'cell':<14} {'ON med':>9} {'OFF med':>9} {'CTL med':>9} "
              f"{'OFF/ON':>8} {'sep':>9} {'CTL/ON':>8} {'floor':>9}")
        print("-" * 84)
        for (shape, d) in sorted(rows, key=lambda k: (k[1], k[0])):
            a = rows[(shape, d)]
            if not {"on", "off", "ctl"} <= a.keys():
                continue
            on, off, ctl = a["on"], a["off"], a["ctl"]
            mo, mf, mc = median(on), median(off), median(ctl)
            print(f"{shape+'_'+str(d):<14} {mo:9.4f} {mf:9.4f} {mc:9.4f} "
                  f"{mo/mf:8.3f}x {sep(on, off):>9} {mo/mc:8.3f}x {sep(on, ctl):>9}")
        print("\n  OFF/ON > 1 means the pack is a LOSS at that size (OFF is faster).")
        print("  CTL is ON's own bytes recompiled: CTL/ON should be ~1.000 and 'overlap'.")

    # The pre-registered verdict, evaluated rather than eyeballed.
    print("\n== plan-s51 §3 pre-registered thresholds ==")
    small = [(k, a) for (k, w), a in series.items() if w == "1t" and k[1] == 128]
    large = [(k, a) for (k, w), a in series.items() if w == "1t" and k[1] == 2048]
    verdict = []
    for label, cells, want_off_faster, bar in (
        ("d=128  OFF beats ON by >=15%, disjoint", small, True, 1.15),
        ("d=2048 ON beats OFF, disjoint", large, False, 1.0),
    ):
        if not cells:
            verdict.append(f"  NOT RUN  {label}")
            continue
        for k, a in cells:
            if not {"on", "off"} <= a.keys():
                continue
            on, off = a["on"], a["off"]
            r = median(on) / median(off)          # >1 == OFF faster
            ok = (r >= bar if want_off_faster else (1 / r) >= bar) and sep(on, off) == "DISJOINT"
            verdict.append(f"  {'PASS' if ok else 'FAIL'}  {k[0]}_{k[1]}: {label} "
                           f"(OFF/ON {r:.3f}x, {sep(on, off)})")
    print("\n".join(verdict))
    print("\n  BOTH must PASS for a crossover. A monotone result REFUTES the model "
          "and no\n  threshold is fitted — plan-s51 §3.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "benches/results-s51"))
