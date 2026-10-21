#!/usr/bin/env python3
"""Generate the GPU differential programs: tile sites over every staging path.

Each program builds a and b from fractional values (so a reordered fold changes
the printed bits), computes one tile site, and prints a whole-output checksum
plus two corner cells. The interpreter's stdout is the oracle.

usage: gen_tiles.py <outdir>
"""
import sys
from pathlib import Path

# name, M, N, K, a-layout, b-layout, elem
CASES = [
    ("rm_256", 256, 256, 256, "rm", "rm", "f32"),     # staged, vector both
    ("rm_128x384x96", 128, 384, 96, "rm", "rm", "f32"),
    ("at_256", 256, 256, 128, "t", "rm", "f32"),      # a column-major: scalar a staging
    ("bt_256", 256, 256, 128, "rm", "t", "f32"),      # b column-major: scalar b staging
    ("odd_100", 100, 100, 100, "rm", "rm", "f32"),    # indivisible: one thread per cell
    ("rm_128_f64", 128, 128, 128, "rm", "rm", "f64"),
]


def a_index(layout, M, K):
    return f"a[i * {K} + k]" if layout == "rm" else f"a[k * {M} + i]"


def b_index(layout, K, N):
    return f"b[k * {N} + j]" if layout == "rm" else f"b[j * {K} + k]"


def program(M, N, K, al, bl, elem):
    widen = f"widen_{elem}"
    zero = "0.0"
    return f"""fn main() {{
    {M * K} -> iota -> ta;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> {widen} -> x; x / 7.0 }} -> a;
    {K * N} -> iota -> tb;
    tb -> map {{ t -> (t * 11 + 57) % 103 - 51 -> {widen} -> x; x / 9.0 }} -> b;
    {K} -> iota -> krange;
    {M * N} -> iota -> tc;
    tc -> map {{ t ->
        t / {N} -> i;
        t % {N} -> j;
        ({zero}, krange) -> fold {{ acc, k -> acc + {a_index(al, M, K)} * {b_index(bl, K, N)} }}
    }} -> c;
    ({zero}, c) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
    c[0] -> println;
    c[{M * N - 1}] -> println;
}}
"""


out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for name, M, N, K, al, bl, elem in CASES:
    (out / f"{name}.mapal").write_text(program(M, N, K, al, bl, elem))
    print(name)


def bench(n):
    """The square f32 site the timing harness launches: only its kernel is
    used, so its data is the matmul benches' integer formula."""
    return f"""fn main() {{
    {n * n} -> iota -> ta;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 }} -> a;
    ta -> map {{ t -> (t * 7 + 57) % 101 - 50 -> widen_f32 }} -> b;
    {n} -> iota -> krange;
    ta -> map {{ t ->
        t / {n} -> i;
        t % {n} -> j;
        (0.0, krange) -> fold {{ acc, k -> acc + a[i * {n} + k] * b[k * {n} + j] }}
    }} -> c;
    c[0] -> println;
}}
"""


for n in (1024, 1536, 2048, 3072, 4096, 8192):
    (out / f"mm_{n}.mapal").write_text(bench(n))


def e2e(n):
    """The same site as `bench(n)`, self-timed around the site with the `time`
    builtin: what a Mapal program pays for the site end to end (transfers and
    launch included), next to the harness's kernel-only number."""
    return f"""fn main() {{
    {n * n} -> iota -> ta;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 }} -> a;
    ta -> map {{ t -> (t * 7 + 57) % 101 - 50 -> widen_f32 }} -> b;
    {n} -> iota -> krange;
    () -> time -> t0;
    ta -> map {{ t ->
        t / {n} -> i;
        t % {n} -> j;
        (0.0, krange) -> fold {{ acc, k -> acc + a[i * {n} + k] * b[k * {n} + j] }}
    }} -> c;
    () -> time -> t1;
    c[0] -> println;
    "iter ms=" -> print;
    t1 - t0 -> println;
}}
"""


for n in (1024, 4096):
    (out / f"e2e_{n}.mapal").write_text(e2e(n))
