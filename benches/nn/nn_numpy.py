#!/usr/bin/env python3
"""NumPy CPU baseline for the linear-NN shapes (plan-s49).

Mirrors benches/nn/nn_gen.py exactly: the same integer generator, the same layer
order, the same printed values, and `iter ms=` bracketing the forward pass only —
data generation is outside every timed region, as in the Mapal sources.

nn2x2 is written the way a person writes two nets in numpy: one after the other.
That IS the baseline. Mapal's advantage claim is precisely that it does not need
the user to write the overlap.

Usage: nn_numpy.py <nn2|nn4|nn2x2> <d> <iters>

NN_WARMUP (default 3, as benches/matmul/numpy_bench.py): untimed forward passes
first, so the timed one finds BLAS's thread pool awake and its buffers mapped —
the state a Mapal leg's bracket finds its pool in (created and used by the data
generation before `time`). NN_WARMUP=0 is the old cold arm: one first call per
process, which on a 64-thread OpenBLAS cost 8.70 ms against 1.52 warm at nn2
d=1024 (local-rtx-5080).
"""
import os
import sys
import time

import numpy as np

SEEDS = {
    "x": (7, 13), "x2": (17, 3), "w1": (5, 29), "w2": (3, 41),
    "w3": (11, 7), "w4": (13, 17), "w1b": (19, 23), "w2b": (23, 31),
}


def mat(name: str, d: int) -> np.ndarray:
    mul, add = SEEDS[name]
    t = np.arange(d * d, dtype=np.int64)
    return ((t * mul + add) % 9 + 1).astype(np.float32).reshape(d, d)


def render(v) -> str:
    return np.format_float_positional(np.float32(v), unique=True, trim="-")


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} <nn2|nn4|nn2x2> <d> <iters>")
    shape, d, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    m = {name: mat(name, d) for name in SEEDS}

    warmup = int(os.environ.get("NN_WARMUP", "3"))
    for i in range(warmup + iters):
        start = time.perf_counter()
        if shape == "nn2":
            out = [m["x"] @ m["w1"] @ m["w2"]]
        elif shape == "nn4":
            out = [m["x"] @ m["w1"] @ m["w2"] @ m["w3"] @ m["w4"]]
        elif shape == "nn2x2":
            out = [m["x"] @ m["w1"] @ m["w2"], m["x2"] @ m["w1b"] @ m["w2b"]]
        else:
            raise SystemExit(f"unknown shape {shape}")
        if i >= warmup:
            print(f"iter ms={(time.perf_counter() - start) * 1000:.6f}")
    for o in out:
        print(render(o.flat[0]))
        print(render(o.flat[d * d - 1]))


if __name__ == "__main__":
    main()
