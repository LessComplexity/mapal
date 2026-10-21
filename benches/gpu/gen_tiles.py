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


for n in (1024, 2048, 4096, 8192):
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


def decode(T, d, rewrite=False):
    """S53 `retain-delta`: a decode loop. Iteration t writes row t of the
    carried `x` (d unrolled `Update`s at `t*d + j`, the new row fed by the
    previous output), then `K = x·Wk`, `V = x·Wv` (row-local, invariant
    weights), `s = q_t·Kᵀ` and `o = s·V` (row-mixing). The compiler is never
    told any of that; the rule has to find that only row t of K and V changed.

    `rewrite=True` is the negative control: the iteration writes x[j·d + t]
    instead (a column, not a row), so the written cells are not one
    contiguous range and nothing is retained."""
    w = 16 * d  # scales from a numpy run: outputs stay O(0.01..0.3) at both sizes
    row = []
    for j in range(d):
        idx = f"{j} * {d} + t" if rewrite else f"t * {d} + {j}"
        row.append(
            f"                ((t * {d} + {j}) * 7 + 13) % 101 - 50 -> widen_f32 -> f{j};\n"
            f"                f{j} / 64.0 + o[{j}] -> v{j};\n"
            f"                x[{idx}] <- v{j};\n")
    proj = lambda w_: (
        f"tc -> map {{ c ->\n"
        f"                    c / {d} -> i;\n"
        f"                    c % {d} -> j;\n"
        f"                    (0.0, krange) -> fold {{ acc, k -> acc + x[i * {d} + k] * {w_}[k * {d} + j] }}\n"
        f"                }}")
    # The loop lives in `main`: the entry fn is the one whose big blocks go to
    # the arena, and a 1024x256 decode carries ~10 MB of arrays.
    return f"""fn main() {{
    {d * d} -> iota -> tw;
    tw -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> x; x / {w}.0 }} -> wk;
    tw -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> x; x / {w}.0 }} -> wv;
    {T * d} -> iota -> tc;
    {d} -> iota -> krange;
    {T} -> iota -> rrange;
    (0.0, {T * d}) -> fill -> x0;
    (0.0, {d}) -> fill -> o0;
    mut x: [f32; {T * d}] <- x0;
    mut o: [f32; {d}] <- o0;
    mut t: i32 <- 0;
    loop {{
        (t < {T}) -> {{
            -true-> {{
{''.join(row)}                {proj('wk')} -> kk;
                {proj('wv')} -> vv;
                krange -> map {{ k -> ((t * {d} + k) * 5 + 3) % 97 - 48 -> widen_f32 -> qv; qv / {w}.0 }} -> q;
                rrange -> map {{ r -> (0.0, krange) -> fold {{ acc, k -> acc + q[k] * kk[r * {d} + k] }} }} -> s;
                krange -> map {{ j -> (0.0, rrange) -> fold {{ acc, r -> acc + s[r] * vv[r * {d} + j] }} }} -> o;
                t + 1 -> t;
                -> loop;
            }}
            -false-> o -> out;
        }}
    }}
    out[0] -> println;
    out[{d - 1}] -> println;
    0.0 -> zero: f32;
    (zero, out) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
}}
"""


for T, d in ((16, 8), (1024, 256)):
    (out / f"decode_{T}_{d}.mapal").write_text(decode(T, d))
(out / "loop_rewrite_16_8.mapal").write_text(decode(16, 8, rewrite=True))
