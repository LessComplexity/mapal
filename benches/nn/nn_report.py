#!/usr/bin/env python3
"""Turn benches/results-s49/S_* sample files into the plan-s49 tables.

Reporting rule (S33): min for 1-thread legs (frequency ramp makes slow outliers),
median for parallel legs (the clock-read race made fast ones). Both are printed
so the choice is visible rather than assumed.

Overlap factor  W = 2 * T(nn2) / T(nn2x2)  per leg: 1.0 = the two nets ran one
after the other, 2.0 = they fully overlapped. nn4 is the control — same 8 d^3
flops as nn2x2, zero available concurrency — so its W_ctl = 2*T(nn2)/T(nn4)
measures what the machine gives for the extra work with no DAG to exploit.
"""
import statistics
import sys
from pathlib import Path

LEGS = ["mapal-conf-1t", "mapal-conf-par", "mapal-fma-1t", "mapal-fma-par",
        "numpy-1t", "numpy-mt"]
SHAPES = ["nn2", "nn4", "nn2x2"]
FLOPS = {"nn2": 4, "nn4": 8, "nn2x2": 8}   # coefficient of d^3


def load(root: Path):
    data = {}
    for f in sorted(root.glob("S_*")):
        _, shape, d, leg = f.name.split("_", 3)
        vals = [float(x) for x in f.read_text().split()]
        if vals:
            data[(shape, int(d), leg)] = vals
    return data


def pick(vals, leg):
    return min(vals) if leg.endswith("-1t") else statistics.median(vals)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "benches/results-s49")
    data = load(root)
    if not data:
        raise SystemExit(f"no samples in {root}")
    sizes = sorted({d for _, d, _ in data})
    n = len(next(iter(data.values())))

    print(f"\n== ms per forward pass (n={n}; min for 1t, median for par) ==")
    head = f"{'shape':<7}{'d':>6}  " + "".join(f"{leg:>16}" for leg in LEGS)
    print(head)
    print("-" * len(head))
    for d in sizes:
        for shape in SHAPES:
            row = f"{shape:<7}{d:>6}  "
            for leg in LEGS:
                v = data.get((shape, d, leg))
                row += f"{pick(v, leg):>16.4f}" if v else f"{'-':>16}"
            print(row)

    print(f"\n== GF/s ==")
    print(head)
    print("-" * len(head))
    for d in sizes:
        for shape in SHAPES:
            row = f"{shape:<7}{d:>6}  "
            for leg in LEGS:
                v = data.get((shape, d, leg))
                if v:
                    gf = FLOPS[shape] * d ** 3 / (pick(v, leg) * 1e-3) / 1e9
                    row += f"{gf:>16.1f}"
                else:
                    row += f"{'-':>16}"
            print(row)

    print("\n== overlap factor  W = 2*T(nn2)/T(nn2x2)   [1 = serial, 2 = full overlap] ==")
    print("   W_ctl = 2*T(nn2)/T(nn4) is the SAME flops with no available concurrency")
    hdr = f"{'d':>6}  " + "".join(f"{leg:>16}" for leg in LEGS)
    print(hdr)
    print("-" * len(hdr))
    for d in sizes:
        for label, ctl in (("W", "nn2x2"), ("W_ctl", "nn4")):
            row = f"{d:>6}  " if label == "W" else f"{'  ctl':>6}  "
            for leg in LEGS:
                a, b = data.get(("nn2", d, leg)), data.get((ctl, d, leg))
                row += f"{2 * pick(a, leg) / pick(b, leg):>16.3f}" if a and b else f"{'-':>16}"
            print(row)

    print("\n== mapal / numpy speedup (>1 = mapal faster), fma face ==")
    hdr = f"{'shape':<7}{'d':>6}{'1t':>12}{'par-vs-numpy-mt':>20}{'par-vs-numpy-1t':>20}"
    print(hdr)
    print("-" * len(hdr))
    for d in sizes:
        for shape in SHAPES:
            g = lambda leg: data.get((shape, d, leg))
            def r(num, den):
                a, b = g(num), g(den)
                return f"{pick(a, num) / pick(b, den):>20.3f}" if a and b else f"{'-':>20}"
            m1, n1 = g("mapal-fma-1t"), g("numpy-1t")
            one = f"{pick(n1, 'numpy-1t') / pick(m1, 'mapal-fma-1t'):>12.3f}" if m1 and n1 else f"{'-':>12}"
            print(f"{shape:<7}{d:>6}{one}"
                  + r("numpy-mt", "mapal-fma-par") + r("numpy-1t", "mapal-fma-par"))


if __name__ == "__main__":
    main()
