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


# --- S53 `resident`: programs with more than one site, or a loop, where the
# question is where each object lives between sites (plan-s53 §2.3).

def operand(name, n, mul, add, mod, off, scale):
    """`name` = a formula over the index: closed form, so `devgen` may build it
    on the device; `scale` keeps loop products bounded."""
    return (f"    ta -> map {{ t -> (t * {mul} + {add}) % {mod} - {off} -> widen_f32 -> v; "
            f"v / {scale} }} -> {name};\n")


def site(out, a, b, n):
    return f"""    ta -> map {{ t ->
        t / {n} -> i;
        t % {n} -> j;
        (0.0, krange) -> fold {{ acc, k -> acc + {a}[i * {n} + k] * {b}[k * {n} + j] }}
    }} -> {out};
"""


def head(n):
    return f"fn main() {{\n    {n * n} -> iota -> ta;\n    {n} -> iota -> krange;\n"


def chain(n, timed=False, check=False):
    """F1 through a chain: `c` is read only by the second site."""
    t0, t1 = ("    () -> time -> t0;\n", "    () -> time -> t1;\n") if timed else ("", "")
    tail = f"    d[0] -> println;\n"
    if check:
        tail = f"    (0.0, d) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n" + tail + f"    d[{n * n - 1}] -> println;\n"
    if timed:
        tail += '    "iter ms=" -> print;\n    t1 - t0 -> println;\n'
    return (head(n) + operand("a", n, 7, 13, 101, 50, "7.0") + operand("b", n, 11, 57, 103, 51, "9.0")
            + operand("e", n, 5, 3, 97, 48, "11.0") + t0 + site("c", "a", "b", n) + site("d", "c", "e", n)
            + t1 + tail + "}\n")


def share(n, check=False):
    """F1 as a fan-out: `a` is read by two sites. `e` is read only by `Index`
    (one-element copies); `d` whole when checked."""
    tail = "    d[0] -> println;\n    e[0] -> println;\n"
    if check:
        tail = f"    (0.0, d) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n" + tail + f"    e[{n * n - 1}] -> println;\n"
    return (head(n) + operand("a", n, 7, 13, 101, 50, "7.0") + operand("b", n, 11, 57, 103, 51, "9.0")
            + operand("c", n, 5, 3, 97, 48, "11.0") + site("d", "a", "b", n) + site("e", "a", "c", n)
            + tail + "}\n")


def loop_mm(n, iters, check=False):
    """F2 + F3: `w` is loop-invariant, `x` is carried from site to site. `w`'s
    scale keeps |x| bounded over the iterations (no inf, no subnormals)."""
    tail = "    out[0] -> println;\n"
    if check:
        tail = (f"    0.0 -> z: f32;\n    (z, out) -> fold {{ acc, v -> acc + v }} -> total;\n"
                f"    total -> println;\n" + tail + f"    out[{n * n - 1}] -> println;\n")
    return (head(n) + operand("x0", n, 7, 13, 101, 50, "7.0")
            + operand("w", n, 11, 57, 103, 51, f"{8 * n}.0")
            + f"""    mut x: [f32; {n * n}] <- x0;
    mut it: i32 <- 0;
    loop {{
        (it < {iters}) -> {{
            -true-> {{
{site("x", "x", "w", n).replace("    ", "                ", 1)}                it + 1 -> it;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
""" + tail + "}\n")


# Differential versions (oracle-checked, both faces; the loop at 128 because
# the interpreter takes minutes per 256³ iteration).
(out / "chain_256.mapal").write_text(chain(256, check=True))
(out / "share_256.mapal").write_text(share(256, check=True))
(out / "loop_128_8.mapal").write_text(loop_mm(128, 8, check=True))
print("chain_256\nshare_256\nloop_128_8")
# Timed versions (MAPAL_GPU_TRACE + process wall, ON vs each OFF arm).
(out / "mm_chain_2048.mapal").write_text(chain(2048))
(out / "mm_share_1024.mapal").write_text(share(1024))
(out / "loop_mm_1024_8.mapal").write_text(loop_mm(1024, 8))
(out / "e2e_chain_4096.mapal").write_text(chain(4096, timed=True))
