#!/usr/bin/env python3
"""The summary of benches/local/cpu_shapes.sh: every leg's median, and three fair comparisons per workload,
each with its 95 % interval, p-value and verdict (benches/local/stats.py).

  one thread:     Mapal's best 1t build  vs  C++'s best 1t build (strict or -ffast-math), NumPy (1 thread)
  whole machine:  Mapal's best par build vs  C++'s best of its 1t and mt builds, strict or -ffast-math

"Best" = the build with the lower median; its whole sample is then compared.
usage: shapes_summary.py <log dir> <out.json>
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stats  # noqa: E402

LABEL = {"fir": "FIR filter, 1M", "conv2d": "conv2d 3x3, 1024²", "saxpy": "saxpy, 1M",
         "reduce": "sum reduction, 1M", "transpose": "transpose, 1024²", "gather": "gather x[idx[i]], 1M"}
LEGS = ["mapal-conf-1t", "mapal-fma-1t", "mapal-conf-par", "mapal-fma-par",
        "cpp-1t", "cpp-fast-1t", "cpp-mt", "cpp-fast-mt", "numpy-1t"]


def main() -> int:
    log, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    rows = []
    print("ms, median of n interleaved runs (one process per run). Ratio = competitor ÷ Mapal (> 1 = Mapal faster),")
    print("[95 % bootstrap interval], Mann–Whitney p; a verdict only when the interval excludes 1 and p < 0.05.\n")
    print("| workload | " + " | ".join(LEGS) + " | 1 thread vs C++ | 1 thread vs NumPy | whole machine vs C++ |")
    print("| --- |" + " ---: |" * (len(LEGS) + 3))
    for shape, name in LABEL.items():
        s = {}
        for leg in LEGS:
            f = log / f"L_{shape}_{leg}"
            if f.exists():
                s[leg] = [float(x) for x in f.read_text().split()]
        best = lambda legs: min((l for l in legs if l in s), key=lambda l: stats.median(s[l]))
        m1, mp = best(["mapal-conf-1t", "mapal-fma-1t"]), best(["mapal-conf-par", "mapal-fma-par"])
        c1, cw = best(["cpp-1t", "cpp-fast-1t"]), best(["cpp-1t", "cpp-fast-1t", "cpp-mt", "cpp-fast-mt"])
        cmp = {"one_thread_cpp": stats.compare(s[c1], s[m1]),
               "one_thread_numpy": stats.compare(s["numpy-1t"], s[m1]),
               "whole_machine_cpp": stats.compare(s[cw], s[mp])}
        picks = {"mapal_1t": m1, "mapal_par": mp, "cpp_1t": c1, "cpp_whole": cw}
        print(f"| {name} | " + " | ".join(f"{stats.median(s[l]):.4f}" if l in s else "–" for l in LEGS) + " | "
              + " | ".join(stats.fmt(cmp[k]) for k in ("one_thread_cpp", "one_thread_numpy", "whole_machine_cpp")) + " |")
        rows.append({"name": name, "shape": shape, "picks": picks, "compare": cmp,
                     "median_ms": {l.replace("-", "_"): stats.median(v) for l, v in s.items()},
                     "samples_ms": {l.replace("-", "_"): v for l, v in s.items()}})
    out.write_text(json.dumps({"stat": "median of interleaved runs; bootstrap 95 % interval; Mann–Whitney",
                               "workloads": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
