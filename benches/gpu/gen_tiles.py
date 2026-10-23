#!/usr/bin/env python3
"""Generate the GPU differential programs: tile sites over every staging path.

Each program builds a and b from fractional values (so a reordered fold changes
the printed bits), computes one tile site, and prints a whole-output checksum
plus two corner cells. The interpreter's stdout is the oracle.

usage: gen_tiles.py <outdir>
"""
import re
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
    # S62 `cell-fill` / `asyncstage`: 16 rows, no register tile divides it —
    # one output per thread in a 16 x 16 block, 32 blocks (Under on every part).
    ("cell_16x512x256", 16, 512, 256, "rm", "rm", "f32"),
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
# S53 integration: a filled grid (2048², small k) — the size the measured
# tables switch rungs at (`wide-fma`, register staging), which the 256-sized
# cases never reach; its oracle is the CPU build (bundle.sh `wide_*`).
(out / "wide_2048x2048x64.mapal").write_text(program(2048, 2048, 64, "rm", "rm", "f32"))
# A float `%` in a closed-form operand: `devgen` must leave it to the host
# (fmodf vs NVPTX's frem expansion differ).
(out / "fmod_64.mapal").write_text(program(64, 64, 64, "rm", "rm", "f32").replace("x / 7.0", "x % 0.7"))
# S54 `raster`: a site whose grid is more than one wave and whose block panels
# over k (8 MB) leave the RTX 3090's 6 MB L2 — the differential of the grouped
# block order (32 block-rows in two groups of 16 on the 3090; row-major on every
# larger L2). Oracle: the CPU build (bundle.sh `deep_*`).
(out / "deep_4096x1024x8192.mapal").write_text(program(4096, 1024, 8192, "rm", "rm", "f32"))
# ... and the grouped order where the measured table turns the gate off (the RTX 5090:
# 512 blocks of 128×128, more than its 340-block wave).
(out / "wide_8192x1024x1024.mapal").write_text(program(8192, 1024, 1024, "rm", "rm", "f32"))
print("wide_2048x2048x64\nfmod_64\ndeep_4096x1024x8192\nwide_8192x1024x1024")
# S62: the M16 projections the timing cells launch, oracle-checked whole (CPU
# build: 235M steps is too long for the interpreter).
for M, N, K in ((16, 2048, 7168), (16, 1024, 4096)):
    (out / f"deep_{M}x{N}x{K}.mapal").write_text(program(M, N, K, "rm", "rm", "f32"))
# ... and the cell tile's epilogue store (a per-cell consumer) and its
# column-major store (`rowfold-layout`: a row fold is the only reader).
_cell = program(16, 512, 256, "rm", "rm", "f32")
(out / "cellepi_16x512x256.mapal").write_text(
    _cell.replace("} -> c;\n", "} -> c;\n    c -> map { v -> v * 0.5 + 1.0 } -> d;\n")
    .replace("(0.0, c)", "(0.0, d)").replace("c[0]", "d[0]").replace("c[8191]", "d[8191]"))
(out / "cellrow_16x512x256.mapal").write_text(
    _cell.replace("} -> c;\n", "} -> c;\n    16 -> iota -> rr;\n    512 -> iota -> jr;\n"
                  "    rr -> map { i -> (0.0, jr) -> fold { acc, j -> acc + c[i * 512 + j] } } -> r;\n")
    .replace("(0.0, c)", "(0.0, r)").replace("c[0]", "r[0]").replace("c[8191]", "r[15]"))
print("deep_16x2048x7168\ndeep_16x1024x4096\ncellepi_16x512x256\ncellrow_16x512x256")


def bench(n, N=None, K=None):
    """The f32 site the timing harness launches (`n × N × K`, square by
    default): only its kernel is used, so its data is the matmul benches'
    integer formula."""
    M, N, K = n, N or n, K or n
    if (M, N, K) == (n, n, n):
        ops = f"""    {n * n} -> iota -> ta;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 }} -> a;
    ta -> map {{ t -> (t * 7 + 57) % 101 - 50 -> widen_f32 }} -> b;
    {n} -> iota -> krange;
    ta -> map {{ t ->"""
    else:
        ops = f"""    {M * K} -> iota -> ta;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 }} -> a;
    {K * N} -> iota -> tb;
    tb -> map {{ t -> (t * 7 + 57) % 101 - 50 -> widen_f32 }} -> b;
    {K} -> iota -> krange;
    {M * N} -> iota -> tc;
    tc -> map {{ t ->"""
    return f"""fn main() {{
{ops}
        t / {N} -> i;
        t % {N} -> j;
        (0.0, krange) -> fold {{ acc, k -> acc + a[i * {K} + k] * b[k * {N} + j] }}
    }} -> c;
    c[0] -> println;
}}
"""


for n in (1024, 1536, 2048, 3072, 4096, 8192):
    (out / f"mm_{n}.mapal").write_text(bench(n))
# S62: the projection cells (M1/M16/M128 of N2048/K7168, N1024/K4096, the
# N18432 one) and the residual MLP's GEMM sites (M128, d256/h512 and
# d1024/h2048: p = x·Wo, h = p·Wup, m = h·Wdown), `mm_<M>x<N>x<K>`.
for M, N, K in ((16, 2048, 7168), (16, 1024, 4096), (16, 18432, 7168), (1, 2048, 7168),
                (128, 2048, 7168), (128, 1024, 4096),
                (128, 256, 256), (128, 512, 256), (128, 256, 512),
                (128, 1024, 1024), (128, 2048, 1024), (128, 1024, 2048)):
    (out / f"mm_{M}x{N}x{K}.mapal").write_text(bench(M, N, K))
# S62 `splitk`: the projection cells (M1/M16/M128 of N×K = 2048×7168, 1024×4096;
# M16 of 18432×7168) and the 4096³ control, as `program`'s fractional site so
# the whole program is also a differential. Only the kernel is timed (tile_bench
# --splitk); `bundle.sh` builds the reassociation faces for these stems.
for M, N, K in ((16, 2048, 7168), (16, 1024, 4096), (16, 18432, 7168), (1, 2048, 7168), (1, 1024, 4096),
                (128, 2048, 7168), (128, 1024, 4096), (4096, 4096, 4096)):
    (out / f"mm_{M}x{N}x{K}.mapal").write_text(program(M, N, K, "rm", "rm", "f32"))


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

# S53: the e2e site behind an input that traps (division by zero at t = 77), so
# the program exits 101 before its first site, while the eager init may still
# be running. The interpreter and the CPU build trap the same way.
(out / "e2e_trap.mapal").write_text(
    bench(1024).replace("(t * 7 + 13) % 101 - 50 -> widen_f32", "1000 / (t - 77) -> widen_f32"))

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

def decode(T, d, rewrite=False, scaled=False):
    """S53 `retain-delta`: a decode loop. Iteration t writes row t of the
    carried `x` (d unrolled `Update`s at `t*d + j`, the new row fed by the
    previous output), then `K = x·Wk`, `V = x·Wv` (row-local, invariant
    weights), `s = q_t·Kᵀ` and `o = s·V` (row-mixing). The compiler is never
    told any of that; the rule has to find that only row t of K and V changed.

    `rewrite=True` is the negative control: the iteration writes x[j·d + t]
    instead (a column, not a row), so the written cells are not one
    contiguous range and nothing is retained.

    `scaled=True` (S54): `K` and `V` read `xh = x·0.5`, a per-cell map of the
    written `x` — the row delta has to pass through it (`write_delta`'s Same
    consumers), so `xh` uploads its written row and `K`, `V` are retained."""
    w = 16 * d  # scales from a numpy run: outputs stay O(0.01..0.3) at both sizes
    row = []
    for j in range(d):
        idx = f"{j} * {d} + t" if rewrite else f"t * {d} + {j}"
        row.append(
            f"                ((t * {d} + {j}) * 7 + 13) % 101 - 50 -> widen_f32 -> f{j};\n"
            f"                f{j} / 64.0 + o[{j}] -> v{j};\n"
            f"                x[{idx}] <- v{j};\n")
    xs = "xh" if scaled else "x"
    half = "                x -> map { v -> v * 0.5 } -> xh;\n" if scaled else ""
    proj = lambda w_: (
        f"tc -> map {{ c ->\n"
        f"                    c / {d} -> i;\n"
        f"                    c % {d} -> j;\n"
        f"                    (0.0, krange) -> fold {{ acc, k -> acc + {xs}[i * {d} + k] * {w_}[k * {d} + j] }}\n"
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
{''.join(row)}{half}                {proj('wk')} -> kk;
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
(out / "decode_scaled_16_8.mapal").write_text(decode(16, 8, scaled=True))

# --- S54: multi-computation probes. Several computations in one program — a
# tile site with per-cell consumers, chains, fan-out, fan-in, independent nets,
# a reduction after a tile site, a loop with an activation — to show where
# kernel boundaries fall by default (plan-s41 amendment 9: fusion is placement).
# `px_<probe>_<n>`: the differential — checksum + corners of every output,
# both faces, the interpreter's stdout as the oracle. `pt_<probe>_<n>`: timed
# at 2048 / 4096 — `time` around the computations, then the corners of every
# output (the values `chain_bench.cu`'s library legs are gated against); the box
# runs them under MAPAL_GPU_TRACE. chain_bench.cu rebuilds each probe with the
# same operand laws (OPS, BIAS) — change both or neither.
OPS = {  # name: (mul, add, mod, off, divisor) — `(t·mul + add) % mod − off`, ÷ divisor
    'a': (7, 13, 101, 50, '7.0'), 'b': (11, 57, 103, 51, '9.0'), 'e': (5, 3, 97, 48, '11.0'),
    'f': (3, 1, 89, 44, '13.0'), 'g': (13, 7, 107, 53, '15.0'), 'p': (17, 29, 109, 54, '17.0'),
}
BIAS = {'bias': (13, 6, '5.0'), 'bias2': (11, 5, '4.0')}  # name: (mod, off, divisor) over the column j


def probe_ops(names, n, w=None):
    s = ''.join(operand(x, n, *OPS[x]) for x in names)
    if w:  # the loop's weight: ÷ n keeps |x| O(0.1..10) over 8 relu steps (numpy, n = 128..4096)
        s += operand(w, n, 11, 57, 103, 51, f"{n}.0")
    return s


def bias(name):
    mod, off, div = BIAS[name]
    return f"    krange -> map {{ t -> t % {mod} - {off} -> widen_f32 -> v; v / {div} }} -> {name};\n"


def relu(e):
    return f"{e} -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> r; r"


def expo(e):
    """exp(x) for x <= 0 as its limit form (1 + x/256)^256 — 8 squarings, the
    base clamped at 0. `exp` is not in the Core op set; chain_bench.cu's `expo`
    is the same function."""
    sq = "".join(f" v{i} * v{i} -> v{i + 1};" for i in range(8))
    return f"{e} -> x; x / 256.0 + 1.0 -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> v0;{sq} v8"


def row_fold(o, src, n, seed, step):
    """`o[i] = fold_j (seed, step) src[i·n + j]`: one in-order fold per row."""
    return (f"    krange -> map {{ i -> ({seed}, krange) -> fold {{ acc, j -> src_v; {step} }} }} -> {o};\n"
            .replace("src_v", f"{src}[i * {n} + j] -> e"))


def probes(n):
    """probe: (operands, vectors + computations, [(output, length)])."""
    S = lambda o, a, b: site(o, a, b, n)
    M = lambda o, src, e: f"    {src} -> map {{ v -> {e} }} -> {o};\n"
    T = lambda o, e: f"    ta -> map {{ t -> {e} }} -> {o};\n"
    N = n * n
    return {
        # per-cell consumers of a tile site (an epilogue; a column bias)
        'epi': ('ab', S('c', 'a', 'b') + M('d', 'c', 'v * 0.5 + 1.0'), [('d', N)]),
        'bias': ('ab', bias('bias') + S('c', 'a', 'b') + T('d', f'c[t] + bias[t % {n}]'), [('d', N)]),
        # per-cell only, no fold: three maps
        'ew': ('ab', M('x', 'a', 'v * 3.0') + M('y', 'x', 'v + 2.0') + T('z', 'y[t] * b[t]'), [('z', N)]),
        # a per-cell producer of a tile site's operand (a prologue)
        'pro': ('ab', M('a2', 'a', 'v * 0.25') + S('c', 'a2', 'b'), [('c', N)]),
        # two independent sites joined per cell
        'indep': ('abef', S('c', 'a', 'b') + S('d', 'e', 'f') + T('g', 'c[t] + d[t]'), [('g', N)]),
        # a chain through a per-cell step
        'mlp': ('abe', S('c', 'a', 'b') + M('h', 'c', 'v * 0.5') + S('d', 'h', 'e'), [('d', N)]),
        # one intermediate, three readers: a per-cell map, a tile site, the host
        'fanout': ('abe', S('c', 'a', 'b') + M('h', 'c', 'v * 0.5 + 1.0') + S('d', 'c', 'e'),
                   [('c', N), ('h', N), ('d', N)]),
        # fan-out to two sites, fan-in by a per-cell join
        'diamond': ('abeg', S('c', 'a', 'b') + S('d1', 'c', 'e') + S('d2', 'c', 'g') + T('h', 'd1[t] + d2[t]'),
                    [('h', N)]),
        # a residual: the site's own operand added back per cell
        'residual': ('ab', S('c', 'a', 'b') + T('d', 'c[t] + a[t]'), [('d', N)]),
        # a per-row in-order fold after a tile site
        'rowsum': ('ab', S('c', 'a', 'b')
                   + f"    krange -> map {{ i -> (0.0, krange) -> fold {{ acc, j -> acc + c[i * {n} + j] }} }} -> r;\n",
                   [('r', n)]),
        # softmax over the rows of a tile site's output, temperature n: a row max
        # (a fold over the site's output), exp(x − max) per cell, a row sum of
        # that (a fold over a per-cell output), the quotient per cell
        'softmax_rows': ('ab', S('c', 'a', 'b')
                         + row_fold('mx', 'c', n, '-1000000000.0',
                                    '(e > acc) -> { -true-> e; -false-> acc; }')
                         + T('ex', expo(f'(c[t] - mx[t / {n}]) / {n}.0'))
                         + row_fold('den', 'ex', n, '0.0', 'acc + e')
                         + T('y', f'ex[t] / den[t / {n}]'),
                         [('y', N)]),
        # two independent two-layer MLPs (bias + relu between), at once
        'mlp2': ('abefgp', bias('bias') + bias('bias2')
                 + S('c1', 'a', 'b') + T('h1', relu(f'c1[t] + bias[t % {n}]')) + S('d1', 'h1', 'e')
                 + S('c2', 'f', 'g') + T('h2', relu(f'c2[t] + bias2[t % {n}]')) + S('d2', 'h2', 'p'),
                 [('d1', N), ('d2', N)]),
    }


def loop_act(n, iters):
    """Carried state through a site and a per-cell activation: x ← relu(x·w)."""
    return ('a', f"""    mut x: [f32; {n * n}] <- a;
    mut it: i32 <- 0;
    loop {{
        (it < {iters}) -> {{
            -true-> {{
{site("y", "x", "w", n).replace("    ", "                ", 1)}                y -> map {{ v -> {relu('v')} }} -> x;
                it + 1 -> it;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
""", [('out', n * n)])


def probe_program(n, spec, timed, w=None, clock='iter'):
    """`clock='request'` (S62): the corner reads are bound before `t1` and
    printed after it, so every output's death — its free, and the one-element
    copies — lands inside the clock."""
    names, body, outs = spec
    s = head(n) + probe_ops(names, n, w)
    tail = "" if timed else "    0.0 -> zero: f32;\n"
    reads = ""
    for o, length in outs:
        if not timed:
            tail += f"    (zero, {o}) -> fold {{ acc, v -> acc + v }} -> s_{o};\n    s_{o} -> println;\n"
        if clock == 'request':
            reads += f"    {o}[0] -> r0_{o};\n    {o}[{length - 1}] -> r1_{o};\n"
            tail += f"    r0_{o} -> println;\n    r1_{o} -> println;\n"
        else:
            tail += f"    {o}[0] -> println;\n    {o}[{length - 1}] -> println;\n"
    if timed:
        # Declarations (a loop's `mut` state) stay outside the clock.
        lines = body.splitlines(keepends=True)
        decl = [x for x in lines if x.startswith("    mut ")]
        rest = [x for x in lines if not x.startswith("    mut ")]
        s += "".join(decl) + "    () -> time -> t0;\n" + "".join(rest) + reads + "    () -> time -> t1;\n"
        tail += '    "iter ms=" -> print;\n    t1 - t0 -> println;\n'
    else:
        s += body
    return s + tail + "}\n"


# The differential at 128 (the interpreter takes ~2 min per 256³ site and the
# probes hold 25 sites; 128 is still several blocks per site on every part).
# The loop at 256 × 3 steps: its carried array then reaches `heap_min_bytes`
# (256 KB), the storage the timed loops have — at 128 the 64 KB array stays an
# SSA aggregate and clang takes ~35 s per host object (~1 s at 256).
for name, spec in probes(128).items():
    (out / f"px_{name}_128.mapal").write_text(probe_program(128, spec, False))
(out / "px_loop_256.mapal").write_text(probe_program(256, loop_act(256, 3), False, w='w'))
# S54 `devmap`: `ew` read at eight cells across the grid instead of folded whole,
# so its cells run on the device (px_ew's whole host fold is R2's negative
# control: with every operand closed-form, it stays on the host).
_names, _body, _ = probes(128)['ew']
(out / "px_ewc_128.mapal").write_text(
    head(128) + probe_ops(_names, 128) + _body
    + "".join(f"    z[{i}] -> println;\n" for i in (0, 1, 255, 256, 8191, 8192, 12345, 128 * 128 - 1))
    + "}\n")
for n in (2048, 4096):
    for name, spec in probes(n).items():
        (out / f"pt_{name}_{n}.mapal").write_text(probe_program(n, spec, True))
    (out / f"pt_loop_{n}.mapal").write_text(probe_program(n, loop_act(n, 8), True, w='w'))
# S54 `epilogue`: the per-cell consumer in the store of the one-thread-per-cell
# kernel (100 is indivisible by every tile) and of an f64 site.
(out / "px_bias_100.mapal").write_text(probe_program(100, probes(100)['bias'], False))
(out / "px_mlp2_64_f64.mapal").write_text(
    probe_program(64, probes(64)['mlp2'], False).replace("widen_f32", "widen_f64").replace("zero: f32", "zero: f64"))
# S54 `streams`: the probes with independent sites (and `mlp`, `pdl`'s chain) at
# sizes whose grids are under one wave on every part — where concurrency can pay.
for n in (512, 1024):
    for name, spec in probes(n).items():
        if name in ('indep', 'diamond', 'mlp2', 'fanout', 'mlp'):
            (out / f"pt_{name}_{n}.mapal").write_text(probe_program(n, spec, True))
# `twin`: indep with its join left out — two independent sites whose outputs are
# read at the corners only, so nothing crosses whole and no host loop runs: the
# device-only form indep takes once its join is placed on the device (R1/R2).
# What streams can buy on a part, with no host work in the way.
def twin(n):
    return ('abef', site('c', 'a', 'b', n) + site('d', 'e', 'f', n), [('c', n * n), ('d', n * n)])


(out / "px_twin_128.mapal").write_text(probe_program(128, twin(128), False))
for n in (512, 1024, 2048, 4096):
    (out / f"pt_twin_{n}.mapal").write_text(probe_program(n, twin(n), True))
# `fan`: the twin's two outputs joined by a third site (g = c · d), every output
# read at the corners only — device-only, so streams open: the differential of
# two chains and the cross-chain wait before the join. g's corners depend on a
# whole row of c and a whole column of d.
fan_body = twin(128)[1] + site('g', 'c', 'd', 128)
(out / "px_fan_128.mapal").write_text(
    probe_program(128, ('abef', fan_body, [('c', 128 * 128), ('d', 128 * 128), ('g', 128 * 128)]), True)
    .replace('    () -> time -> t0;\n', '').replace('    () -> time -> t1;\n', '')
    .replace('    "iter ms=" -> print;\n    t1 - t0 -> println;\n', ''))

# --- S62 ordered tile geometry: skinny sites (M ≪ N). Differential: the row-fitted
# and scalar-row (`fill-rows`) tiles, M1's one-row grid and an f64 (two-lane) tile;
# oracle: the CPU build (bundle.sh `skinny_*`).
for name, M, N, K, elem in (("skinny_16x16384x512", 16, 16384, 512, "f32"),
                            ("skinny_16x2048x512", 16, 2048, 512, "f32"),
                            ("skinny_1x16384x512", 1, 16384, 512, "f32"),
                            ("skinny_8x4096x256_f64", 8, 4096, 256, "f64")):
    (out / f"{name}.mapal").write_text(program(M, N, K, "rm", "rm", elem))


def bench_mnk(M, N, K):
    """`bench(n)` at M × N × K: the timed kernel of a projection-shaped site."""
    return f"""fn main() {{
    {M * K} -> iota -> ta;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 }} -> a;
    {K * N} -> iota -> tb;
    tb -> map {{ t -> (t * 7 + 57) % 101 - 50 -> widen_f32 }} -> b;
    {K} -> iota -> krange;
    {M * N} -> iota -> tc;
    tc -> map {{ t ->
        t / {N} -> i;
        t % {N} -> j;
        (0.0, krange) -> fold {{ acc, k -> acc + a[i * {K} + k] * b[k * {N} + j] }}
    }} -> c;
    c[0] -> println;
}}
"""


for M in (1, 16, 128):
    for N, K in ((2048, 7168), (1024, 4096), (18432, 7168)):
        (out / f"mm_{M}x{N}x{K}.mapal").write_text(bench_mnk(M, N, K))


def rmlp_skip(M, d, h, timed):
    """A residual MLP layer: out = relu(x·w1)·w2 + x, x M×d, w1 d×h, w2 h×d —
    two skinny sites chained through a per-cell activation, the residual per cell."""
    t0, t1 = ("    () -> time -> t0;\n", "    () -> time -> t1;\n") if timed else ("", "")
    tail = f"    out[0] -> println;\n    out[{M * d - 1}] -> println;\n"
    if timed:
        tail += '    "iter ms=" -> print;\n    t1 - t0 -> println;\n'
    else:
        tail = f"    0.0 -> zero: f32;\n    (zero, out) -> fold {{ acc, v -> acc + v }} -> s;\n    s -> println;\n" + tail
    return f"""fn main() {{
    {M * d} -> iota -> tx;
    {d * h} -> iota -> tw;
    {M * h} -> iota -> tc;
    {d} -> iota -> kd;
    {h} -> iota -> kh;
    tx -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> x;
    tw -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> v; v / {d}.0 }} -> w1;
    tw -> map {{ t -> (t * 5 + 3) % 97 - 48 -> widen_f32 -> v; v / {h}.0 }} -> w2;
{t0}    tc -> map {{ t ->
        t / {h} -> i;
        t % {h} -> j;
        (0.0, kd) -> fold {{ acc, k -> acc + x[i * {d} + k] * w1[k * {h} + j] }}
    }} -> c;
    c -> map {{ v -> {relu("v")} }} -> hr;
    tx -> map {{ t ->
        t / {d} -> i;
        t % {d} -> j;
        (0.0, kh) -> fold {{ acc, k -> acc + hr[i * {h} + k] * w2[k * {d} + j] }}
    }} -> y;
    tx -> map {{ t -> y[t] + x[t] }} -> out;
{t1}{tail}}}
"""


(out / "px_rmlp_128x64x128.mapal").write_text(rmlp_skip(128, 64, 128, False))
for d, h in ((256, 512), (1024, 2048)):
    (out / f"pt_rmlp_128x{d}x{h}.mapal").write_text(rmlp_skip(128, d, h, True))


def rmlp(M, d, h, timed):
    """S62: the residual MLP of the fused-library comparison
    (benches/results/gpu/workloads/residual-mlp/fused-references/preparation/CONTRACT.md):
    p = x·Wo; hraw = p·Wup; h = relu(hraw + b1); mraw = h·Wdown; m = mraw + b2;
    y = (p + m) / 4 — three tile sites of M rows, two of them non-square."""
    def site_(o, a, b, n, k, kr, cells):
        return (f"    {cells} -> map {{ t ->\n        t / {n} -> i;\n        t % {n} -> j;\n"
                f"        (0.0, {kr}) -> fold {{ acc, k -> acc + {a}[i * {k} + k] * {b}[k * {n} + j] }}\n"
                f"    }} -> {o};\n")
    w = lambda name, it, mul, add, mod, off, div: (
        f"    {it} -> map {{ t -> (t * {mul} + {add}) % {mod} - {off} -> widen_f32 -> v; v / {div} }} -> {name};\n")
    s = (f"fn main() {{\n    {M * d} -> iota -> tx;\n    {d * d} -> iota -> two;\n    {d * h} -> iota -> tup;\n"
         f"    {h * d} -> iota -> tdn;\n    {M * h} -> iota -> th;\n    {d} -> iota -> dr;\n    {h} -> iota -> hr;\n"
         + w("x", "tx", 7, 13, 101, 50, "7.0") + w("wo", "two", 11, 57, 103, 51, f"{d}.0")
         + w("wup", "tup", 5, 3, 97, 48, f"{d}.0") + w("wdn", "tdn", 3, 1, 89, 44, f"{h}.0")
         + "    hr -> map { t -> t % 13 - 6 -> widen_f32 -> v; v / 5.0 } -> b1;\n"
         + "    dr -> map { t -> t % 11 - 5 -> widen_f32 -> v; v / 4.0 } -> b2;\n")
    body = (site_("p", "x", "wo", d, d, "dr", "tx") + site_("hraw", "p", "wup", h, d, "dr", "th")
            + f"    th -> map {{ t -> {relu(f'hraw[t] + b1[t % {h}]')} }} -> hh;\n"
            + site_("mraw", "hh", "wdn", d, h, "hr", "tx")
            + f"    tx -> map {{ t -> mraw[t] + b2[t % {d}] -> m; (p[t] + m) / 4.0 }} -> y;\n")
    tail = f"    y[0] -> println;\n    y[{M * d - 1}] -> println;\n"
    if timed:
        return s + "    () -> time -> t0;\n" + body + "    () -> time -> t1;\n" + tail + \
            '    "iter ms=" -> print;\n    t1 - t0 -> println;\n}\n'
    return s + body + "    0.0 -> zero: f32;\n    (zero, y) -> fold { acc, v -> acc + v } -> s_y;\n" \
        "    s_y -> println;\n" + tail + "}\n"


# Timed (`pt_`, the probe section: iter ms per arm) and oracle-checked (`deep_`,
# the CPU build: 42M-176M steps is too long for the interpreter).
for d, h in ((256, 512), (1024, 2048)):
    (out / f"pt_rmlp{d}x{h}_128.mapal").write_text(rmlp(128, d, h, True))
    (out / f"deep_rmlp{d}x{h}_128.mapal").write_text(rmlp(128, d, h, False))

# --- S62 `hostopt` / `place`: composed programs whose sites differ in size, so
# a per-site placement has something to decide. `pt_*`: timed like the probes
# (`time` around the computations, then the corners of every output); their
# oracle is the CPU build (bundle.sh), both faces.
def law(name, length, key):
    mul, add, mod, off, div = OPS[key]
    return (f"    {length} -> iota -> i_{name};\n"
            f"    i_{name} -> map {{ t -> (t * {mul} + {add}) % {mod} - {off} -> widen_f32 -> v; "
            f"v / {div} }} -> {name};\n")


def gsite(out, a, b, M, N, K):
    """`out = a·b`, a M×K and b K×N row-major: one tile site of any shape."""
    return (f"    {M * N} -> iota -> o_{out};\n    {K} -> iota -> k_{out};\n"
            f"    o_{out} -> map {{ t ->\n        t / {N} -> i;\n        t % {N} -> j;\n"
            f"        (0.0, k_{out}) -> fold {{ acc, k -> acc + {a}[i * {K} + k] * {b}[k * {N} + j] }}\n"
            f"    }} -> {out};\n")


def composed(ops, body, outs):
    s = "fn main() {\n" + "".join(law(*o) for o in ops) + "    () -> time -> t0;\n" + body
    s += "    () -> time -> t1;\n"
    for o, length in outs:
        s += f"    {o}[0] -> println;\n    {o}[{length - 1}] -> println;\n"
    return s + '    "iter ms=" -> print;\n    t1 - t0 -> println;\n}\n'


def small_chain(s, x=""):
    """Three dependent s³ sites (`c3{x} = ((a·b)·e)·f`)."""
    ops = [(f"sa{x}", s * s, 'a'), (f"sb{x}", s * s, 'b'), (f"se{x}", s * s, 'e'), (f"sf{x}", s * s, 'f')]
    body = (gsite(f"c1{x}", f"sa{x}", f"sb{x}", s, s, s) + gsite(f"c2{x}", f"c1{x}", f"se{x}", s, s, s)
            + gsite(f"c3{x}", f"c2{x}", f"sf{x}", s, s, s))
    return ops, body, [(f"c3{x}", s * s)]


# The projection families of the S56 composed cells (M rows of activations
# times a K×N weight): DeepSeek expert / dense, Qwen expert.
for M in (1, 16, 128):
    for N, K in ((2048, 7168), (1024, 4096), (18432, 7168)):
        (out / f"pt_pj_m{M}_n{N}_k{K}.mapal").write_text(
            composed([("a", M * K, 'a'), ("b", K * N, 'b')], gsite("c", "a", "b", M, N, K), [("c", M * N)]))
# The residual MLP chains: y = relu(x·W1 + bias)·W2 + x.
for M, d, h in ((128, 256, 512), (128, 1024, 2048)):
    body = (gsite("s1", "x", "w1", M, h, d)
            + f"    o_s1 -> map {{ t -> {relu(f's1[t] + bias[t % {h}]')} }} -> hh;\n"
            + gsite("s2", "hh", "w2", M, d, h)
            + "    o_s2 -> map { t -> s2[t] + x[t] } -> y;\n")
    bias_law = (f"    {h} -> iota -> i_bias;\n"
                f"    i_bias -> map {{ t -> t % 13 - 6 -> widen_f32 -> v; v / 5.0 }} -> bias;\n")
    src = composed([("x", M * d, 'a'), ("w1", d * h, 'b'), ("w2", h * d, 'e')], body, [("y", M * d)])
    (out / f"pt_rm_m{M}_d{d}_h{h}.mapal").write_text(src.replace("    () -> time -> t0;\n", bias_law + "    () -> time -> t0;\n"))
def mixed(n, s):
    """The small chain written first, then an independent `mlp` chain at n
    (c = a·b; h = c·0.5; d = h·e)."""
    ops, body, outs = small_chain(s)
    ops += [("a", n * n, 'a'), ("b", n * n, 'b'), ("e", n * n, 'e')]
    body += gsite("c", "a", "b", n, n, n) + "    c -> map { v -> v * 0.5 } -> h;\n" + gsite("d", "h", "e", n, n, n)
    return ops, body, outs + [("d", n * n)]


# The small chain alone (every site small: the whole program's placement), and
# mixed with a large chain — the cell where per-site placement differs.
for s in (32, 64, 128):
    (out / f"pt_pc_{s}.mapal").write_text(composed(*small_chain(s)))
    for n in (2048, 4096) if s > 32 else (2048,):
        (out / f"pt_pm_{n}_{s}.mapal").write_text(composed(*mixed(n, s)))
# The mixed cell's differential (oracle: the interpreter; 128 keeps it ~1 min).
(out / "px_pm_128_32.mapal").write_text(
    composed(*mixed(128, 32)).replace('    "iter ms=" -> print;\n    t1 - t0 -> println;\n', ''))
# --- S62 runtime measurements (benches/results/gpu/s62/runtime-measurements).
# `pt_rq_<probe>_<n>`: the request clock — each output's corner reads, and so
# its death and free, inside `time`.
for n, names in ((512, ('indep', 'diamond', 'mlp2', 'mlp')), (1024, ('indep', 'diamond', 'mlp2', 'mlp')),
                 (2048, ('mlp2',))):
    for name in names:
        (out / f"pt_rq_{name}_{n}.mapal").write_text(probe_program(n, probes(n)[name], True, clock='request'))
for n in (512, 1024, 2048):
    (out / f"pt_rq_twin_{n}.mapal").write_text(probe_program(n, twin(n), True, clock='request'))


def loop_requests(n, spec, regen, R=8):
    """R requests of one probe in a loop, `time` around the loop. A request's
    input is every site's left operand, its law shifted by the request index
    `it` — an invariant body is hoisted out of the loop whole, so without it
    the loop would time one request. The immutable inputs (right operands and
    bias vectors: no writer in the loop) are built before the loop
    (`regen=False`, `pt_rt_*`: retained) or inside the body (`regen=True`,
    `pt_rg_*`: the same bits rebuilt every request). The last request's
    outputs, joined per cell when there are several, are carried out."""
    names, body, outs = spec
    lines = body.splitlines(keepends=True)
    req = set(re.findall(r'acc \+ (\w+)\[i \*', body)) & set(names)

    def law(x):
        s = operand(x, n, *OPS[x])
        return s.replace("(t * ", "((t + it) * ") if x in req else s
    is_bias = lambda x: x.startswith("    krange -> map { t -> t % ")
    gen = "".join(law(x) for x in names if x not in req) + "".join(x for x in lines if is_bias(x))
    work = "".join(law(x) for x in names if x in req) + "".join(x for x in lines if not is_bias(x))
    if len(outs) > 1:
        work += f"    ta -> map {{ t -> {' + '.join(f'{o}[t]' for o, _ in outs)} }} -> h;\n"
        res = 'h'
    else:
        res = outs[0][0]
    ind = lambda s: "".join(" " * 12 + x for x in s.splitlines(keepends=True))
    N = n * n
    return (head(n) + ("" if regen else gen) + f"    (0.0, {N}) -> fill -> z0;\n"
            + f"    mut carry: [f32; {N}] <- z0;\n    mut it: i32 <- 0;\n    () -> time -> t0;\n"
            + f"    loop {{\n        (it < {R}) -> {{\n            -true-> {{\n"
            + ind((gen if regen else "") + work)
            + f"                {res} -> carry;\n                it + 1 -> it;\n                -> loop;\n"
            + "            }\n            -false-> carry -> out;\n        }\n    }\n"
            + "    () -> time -> t1;\n"
            + f"    out[0] -> println;\n    out[{N - 1}] -> println;\n"
            + '    "iter ms=" -> print;\n    t1 - t0 -> println;\n}\n')


# `pt_rt_*` / `pt_rg_*`: retained vs regenerated immutable inputs, same kernels.
for n in (1024, 2048):
    for name in ('mlp2', 'indep'):
        for tag, regen in (('rt', False), ('rg', True)):
            (out / f"pt_{tag}_{name}_{n}.mapal").write_text(loop_requests(n, probes(n)[name], regen))
# ... and their differential (oracle-checked, two requests at 128).
for tag, regen in (('rt', False), ('rg', True)):
    (out / f"px_{tag}_mlp2_128.mapal").write_text(
        loop_requests(128, probes(128)['mlp2'], regen, R=2)
        .replace('    () -> time -> t0;\n', '').replace('    () -> time -> t1;\n', '')
        .replace('    "iter ms=" -> print;\n    t1 - t0 -> println;\n', ''))


# --- S64 evidence suite (integrated_bundle.sh SUITE): the workloads beyond matmul, timed like the
# probes (`time` around the computation, then the corners of every output and `iter ms`). `<stem>.kmax`:
# the longest fold chain a printed value passes through, the K of ADR-0040 D2's default-face value gate
# (value_gate.py; box_run.sh reads it before the stem's `_d<d>_h<h>`).
def fir(n, taps):
    """A 1-D window site (`func/window.rs`): y[t] = fold_k c[k]·s[t + k] — the taps invariant, the
    signal sliding, one row of n lanes."""
    body = (f"    {n} -> iota -> o_y;\n    {taps} -> iota -> k_y;\n"
            f"    o_y -> map {{ t -> (0.0, k_y) -> fold {{ acc, k -> acc + c[k] * s[t + k] }} }} -> y;\n")
    return composed([("c", taps, 'e'), ("s", n + taps, 'a')], body, [("y", n)])


def conv(n, kk):
    """A 2-D convolution (`func/conv.rs`'s k÷div taps — a host site on every GPU profile: `gpu_site`
    has no `ksplit` encoding): y[i][j] = fold_k w[k]·img[(i + k/kk)·(n+kk-1) + j + k%kk]."""
    w = n + kk - 1
    body = (f"    {n * n} -> iota -> o_y;\n    {kk * kk} -> iota -> k_y;\n"
            f"    o_y -> map {{ t ->\n        t / {n} -> i;\n        t % {n} -> j;\n"
            f"        (0.0, k_y) -> fold {{ acc, k -> acc + w[k] * img[(i + k / {kk}) * {w} + j + k % {kk}] }}\n"
            f"    }} -> y;\n")
    return composed([("w", kk * kk, 'e'), ("img", w * w, 'a')], body, [("y", n * n)])


def timed_loop(src):
    """`time` around a program's top-level loop (its `mut` state is declared before it)."""
    src = src.replace("    loop {\n", "    () -> time -> t0;\n    loop {\n", 1)
    src = src.replace("\n    }\n    out[0] -> println;\n", "\n    }\n    () -> time -> t1;\n    out[0] -> println;\n", 1)
    assert src.count("time -> t") == 2, "one top-level loop"
    return src[:src.rindex("}")] + '    "iter ms=" -> print;\n    t1 - t0 -> println;\n}\n'


SUITE = {
    "pt_fir_4194304_16": (fir(1 << 22, 16), 16),
    "pt_conv_2048_5": (conv(2048, 5), 25),
    "pt_decode_1024_256": (timed_loop(decode(1024, 256)), 1024 * 256),
    "pt_loopmm_1024_8": (timed_loop(loop_mm(1024, 8)), 8 * 1024),
}
for stem, (src, _) in SUITE.items():
    (out / f"{stem}.mapal").write_text(src)
for stem, kmax in [(s, k) for s, (_, k) in SUITE.items()] + [
        (f"pt_{p}_{n}", f * n) for n in (2048, 4096)
        for p, f in (('softmax_rows', 2), ('rowsum', 2), ('ew', 8), ('fanout', 2), ('diamond', 2), ('indep', 1),
                     ('loop', 8), ('mlp', 2))] + [("pt_rt_mlp2_1024", 2 * 1024), ("pt_rt_mlp2_2048", 2 * 2048)]:
    (out / f"{stem}.kmax").write_text(f"{kmax}\n")
