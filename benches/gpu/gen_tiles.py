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

def decode(T, d, rewrite=False, scaled=False, causal=False, prefill=0, gen=0):
    """S53 `retain-delta`: a decode loop. Iteration t writes row t of the
    carried `x` — one map over its cells: row t gets the new row, fed by the
    previous output, every other cell keeps its value — then `K = x·Wk`,
    `V = x·Wv` (row-local, invariant weights), `s = q_t·Kᵀ` and `o = s·V`
    (row-mixing). The compiler is never told any of that; the rule has to
    find that only row t of K and V changed.

    `rewrite=True` is the negative control: the iteration writes column
    `t % d` of every row instead, so the written cells are not one contiguous
    range and nothing is retained.

    `scaled=True` (S54): `K` and `V` read `xh = x·0.5`, a per-cell map of the
    written `x` — the row delta has to pass through it (`write_delta`'s Same
    consumers), so `xh` uploads its written row and `K`, `V` are retained.

    `causal=True`: a real attention decode. Step t's scores cover rows
    `r <= t` only; softmax (max-subtracted, `expo`, over the masked sum) over
    them; `o = p·V / Σp`.

    `prefill=P` (implies causal): rows < P are a prompt. Before the loop, the
    prompt's K/V and its causal attention (every prompt row's output `po`,
    P x T scores at static max length T) are computed; the loop then decodes
    rows P..T-1 from the last prompt output. The loop still writes `K`, `V` as
    `x·W` over all T rows: that rows < P carry over from the prefill and only
    row t changes per step is the compiler's to find.

    `gen=P` (recurrence-precision): prefill's math as ONE loop from t = 0 —
    row t is the prompt row while `t < P`, the generated row after. Which trips
    batch, and that only the last prompt trip's output is read, is the
    compiler's to find (`trip_split`)."""
    w = 16 * d  # scales from a numpy run: outputs stay O(0.01..0.3) at both sizes
    causal = causal or gen > 0
    causal = causal or prefill > 0
    scale, ninf = repr(d ** 0.5), "-1000000000.0"

    def ex(e):  # `expo`
        return f"{e} -> exp"

    def rmax(row):  # the max of one row's T scores
        return f"({ninf}, rrange) -> fold {{ acc, r -> {row} -> e; (e > acc) -> {{ -true-> e; -false-> acc; }} }}"
    if rewrite:
        # Row j's cell of column t % d: the new value of row j (the fed-back
        # output damped: every row takes it, every trip).
        new = (f"(c % {d} == col) -> {{ -true-> {{ "
               f"((t * {d} + c / {d}) * 7 + 13) % 101 - 50 -> widen_f32 -> f; f / 64.0 + o[c / {d} % {d}] / 4.0 }} ")
        row = [f"                t % {d} -> col;\n"
               f"                tc -> map {{ c -> {new}-false-> x[c]; }} }} -> x;\n"]
    else:
        fed = (f"(t < {gen}) -> {{ -true-> f / 64.0; -false-> f / 64.0 + o[c % {d}]; }}" if gen
               else f"f / 64.0 + o[c % {d}]")
        new = (f"(c / {d} == t) -> {{ -true-> {{ "
               f"(c * 7 + 13) % 101 - 50 -> widen_f32 -> f; {fed} }} ")
        row = [f"                tc -> map {{ c -> {new}-false-> x[c]; }} }} -> x;\n"]
    xs = "xh" if scaled else "x"
    half = "                x -> map { v -> v * 0.5 } -> xh;\n" if scaled else ""
    proj = lambda w_, xs=xs: (
        f"tc -> map {{ c ->\n"
        f"                    c / {d} -> i;\n"
        f"                    c % {d} -> j;\n"
        f"                    (0.0, krange) -> fold {{ acc, k -> acc + {xs}[i * {d} + k] * {w_}[k * {d} + j] }}\n"
        f"                }}")
    if causal:
        attend = (
            f"                rrange -> map {{ r -> (r <= t) -> {{ -true-> {{ (0.0, krange) -> fold {{ acc, k -> acc + q[k] * kk[r * {d} + k] }} }} -false-> {ninf}; }} }} -> s;\n"
            f"                {rmax('s[r]')} -> mx;\n"
            f"                rrange -> map {{ r -> (r <= t) -> {{ -true-> {{ {ex(f'(s[r] - mx) / {scale}')} }} -false-> 0.0; }} }} -> p;\n"
            f"                (0.0, rrange) -> fold {{ acc, r -> acc + p[r] }} -> den;\n"
            f"                krange -> map {{ j -> (0.0, rrange) -> fold {{ acc, r -> acc + p[r] * vv[r * {d} + j] }} -> a; a / den }} -> o;\n")
    else:
        attend = (
            f"                rrange -> map {{ r -> (0.0, krange) -> fold {{ acc, k -> acc + q[k] * kk[r * {d} + k] }} }} -> s;\n"
            f"                krange -> map {{ j -> (0.0, rrange) -> fold {{ acc, r -> acc + s[r] * vv[r * {d} + j] }} }} -> o;\n")
    P = prefill
    if P:
        # The prompt: row r < P holds the decode's row law (`f / 64.0`, no fed-back `o`).
        init = (
            f"    tc -> map {{ c -> (c / {d} < {P}) -> {{ -true-> {{ (c * 7 + 13) % 101 - 50 -> widen_f32 -> f; f / 64.0 }} -false-> 0.0; }} }} -> x0;\n"
            f"    () -> time -> pf0;\n"
            f"    {proj('wk', 'x0')} -> k0;\n"
            f"    {proj('wv', 'x0')} -> v0;\n"
            f"    {P * d} -> iota -> pq;\n"
            f"    pq -> map {{ e -> (e * 5 + 3) % 97 - 48 -> widen_f32 -> qv; qv / {w}.0 }} -> qp;\n"
            f"    {P * T} -> iota -> ps_i;\n"
            f"    {P} -> iota -> prange;\n"
            f"    ps_i -> map {{ e -> e / {T} -> i; e % {T} -> r; (r <= i) -> {{ -true-> {{ (0.0, krange) -> fold {{ acc, k -> acc + qp[i * {d} + k] * k0[r * {d} + k] }} }} -false-> {ninf}; }} }} -> ps;\n"
            f"    prange -> map {{ i -> {rmax(f'ps[i * {T} + r]')} }} -> pmx;\n"
            f"    ps_i -> map {{ e -> e / {T} -> i; e % {T} -> r; (r <= i) -> {{ -true-> {{ {ex(f'(ps[e] - pmx[i]) / {scale}')} }} -false-> 0.0; }} }} -> pp;\n"
            f"    prange -> map {{ i -> (0.0, rrange) -> fold {{ acc, r -> acc + pp[i * {T} + r] }} }} -> pden;\n"
            f"    pq -> map {{ e -> e / {d} -> i; e % {d} -> j; (0.0, rrange) -> fold {{ acc, r -> acc + pp[i * {T} + r] * v0[r * {d} + j] }} -> a; a / pden[i] }} -> po;\n"
            f"    krange -> map {{ j -> po[{(P - 1) * d} + j] }} -> o0;\n")
        tail = "    (zero, po) -> fold { acc, v -> acc + v } -> ptotal;\n    ptotal -> println;\n"
    else:
        init = f"    (0.0, {T * d}) -> fill -> x0;\n    (0.0, {d}) -> fill -> o0;\n"
        tail = ""
    # The loop lives in `main`: the entry fn is the one whose big blocks go to
    # the arena, and a 1024x256 decode carries ~10 MB of arrays.
    return f"""fn main() {{
    {d * d} -> iota -> tw;
    tw -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> x; x / {w}.0 }} -> wk;
    tw -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> x; x / {w}.0 }} -> wv;
    {T * d} -> iota -> tc;
    {d} -> iota -> krange;
    {T} -> iota -> rrange;
{init}    mut x: [f32; {T * d}] <- x0;
    mut o: [f32; {d}] <- o0;
    mut t: i32 <- {P};
    loop {{
        (t < {T}) -> {{
            -true-> {{
{''.join(row)}{half}                {proj('wk')} -> kk;
                {proj('wv')} -> vv;
                krange -> map {{ k -> ((t * {d} + k) * 5 + 3) % 97 - 48 -> widen_f32 -> qv; qv / {w}.0 }} -> q;
{attend}                t + 1 -> t;
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
{tail}}}
"""


for T, d in ((16, 8), (1024, 256)):
    (out / f"decode_{T}_{d}.mapal").write_text(decode(T, d))
(out / "causal_16_8.mapal").write_text(decode(16, 8, causal=True))
(out / "prefill_16_8_8.mapal").write_text(decode(16, 8, prefill=8))
(out / "gen_16_8_8.mapal").write_text(decode(16, 8, gen=8))
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
    """exp(e): the `exp` pipeline builtin (ADR-0042). chain_bench.cu's `expo` and torch_bench.py's are `expf` /
    `torch.exp`."""
    return f"{e} -> exp"


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


def conv_mc(N, C, n, kk):
    """A multi-channel 2-D convolution, NCHW, C -> C channels, valid padding:
    y[b][co][i][j] = fold_{ci,dy,dx} w[co][ci][dy][dx]·img[b][ci][i+dy][j+dx]
    (torch conv2d's layouts)."""
    w, taps = n + kk - 1, C * kk * kk
    body = (f"    {N * C * n * n} -> iota -> o_y;\n    {taps} -> iota -> k_y;\n"
            f"    o_y -> map {{ t ->\n        t / {C * n * n} -> b;\n        t / {n * n} % {C} -> co;\n"
            f"        t / {n} % {n} -> i;\n        t % {n} -> j;\n"
            f"        (0.0, k_y) -> fold {{ acc, k -> k / {kk * kk} -> ci; k % {kk * kk} -> q; "
            f"acc + w[co * {taps} + k] * img[((b * {C} + ci) * {w} + i + q / {kk}) * {w} + j + q % {kk}] }}\n"
            f"    }} -> y;\n")
    return composed([("w", C * taps, 'e'), ("img", N * C * w * w, 'a')], body, [("y", N * C * n * n)])


def timed_loop(src):
    """`time` around a program's top-level loop (its `mut` state is declared before it)."""
    src = src.replace("    loop {\n", "    () -> time -> t0;\n    loop {\n", 1)
    src = src.replace("\n    }\n    out[0] -> println;\n", "\n    }\n    () -> time -> t1;\n    out[0] -> println;\n", 1)
    assert src.count("time -> t") == 2, "one top-level loop"
    return src[:src.rindex("}")] + '    "iter ms=" -> print;\n    t1 - t0 -> println;\n}\n'


SUITE = {
    "pt_fir_4194304_16": (fir(1 << 22, 16), 16),
    "pt_conv_2048_5": (conv(2048, 5), 25),
    # N8, 64 -> 64 channels, 3x3, 56x56 outputs (a ResNet stage); px_ is its interpreter-sized check.
    "pt_convmc_8_64_56_3": (conv_mc(8, 64, 56, 3), 64 * 9),
    "px_convmc_2_4_6_3": (conv_mc(2, 4, 6, 3), 4 * 9),
    "pt_decode_1024_256": (timed_loop(decode(1024, 256)), 1024 * 256),
    "pt_causal_1024_256": (timed_loop(decode(1024, 256, causal=True)), 1024 * 256),
    # Timed from the prefill's first op (`pf0`): prefill + decode is the workload.
    "pt_prefill_1024_512_256": (timed_loop(decode(1024, 256, prefill=512)).replace("t1 - t0", "t1 - pf0"), 1024 * 256),
    # prefill's math as one natural loop (`gen`): its twin, timed over the whole loop.
    "pt_gen_1024_512_256": (timed_loop(decode(1024, 256, gen=512)), 1024 * 256),
    "pt_loopmm_1024_8": (timed_loop(loop_mm(1024, 8)), 8 * 1024),
}
for stem, (src, _) in SUITE.items():
    (out / f"{stem}.mapal").write_text(src)
# S65 (ADR-0040 D2's M, absolute): `<stem>_xabs.mapal` is the program with every fold step's term replaced by
# its magnitude, so its exact CPU build prints M = |seed| + Σ|terms| through the whole fold chain for every
# printed value. Only programs whose printed values are fold chains of products and sums (a nonlinear map —
# relu, exp, a quotient — does not carry an error bound through this transform): the relative gate stays
# everywhere else (value_gate.py).
def xabs(src):
    step = re.compile(r'(fold \{ acc, (\w+) -> )acc \+ ([^{}]+?) \}')
    out, n = step.subn(lambda m: f"{m.group(1)}{m.group(3)} -> xt; (xt > 0.0) -> {{ -true-> xt; -false-> 0.0 - xt; }}"
                               f" -> xa; acc + xa }}", src)
    assert n > 0, "no fold step"
    return out


for n in (2048, 4096):
    (out / f"pt_rowsum_{n}_xabs.mapal").write_text(xabs((out / f"pt_rowsum_{n}.mapal").read_text()))
stated = set()  # the stems whose K is written where the program is
for stem, kmax in [(s, k) for s, (_, k) in SUITE.items()] + [
        (f"pt_{p}_{n}", f * n) for n in (2048, 4096)
        for p, f in (('softmax_rows', 2), ('rowsum', 2), ('ew', 8), ('fanout', 2), ('diamond', 2), ('indep', 1),
                     ('loop', 8), ('mlp', 2))] + [("pt_rt_mlp2_1024", 2 * 1024), ("pt_rt_mlp2_2048", 2 * 2048)]:
    stated.add(stem)
    (out / f"{stem}.kmax").write_text(f"{kmax}\n")


# --- generic-fusion: cross-algorithm chains (design §9), idiomatic Mapal. `pt_*` timed like the probes (`time`
# around the computation, then the corners of every output and `iter ms`); `px_*` the small differentials (the
# interpreter is the oracle: corners and a checksum of every output). `<stem>.kmax`: the longest fold chain.
def chain_program(ops, body, outs, timed):
    if timed:
        return composed(ops, body, outs)
    s = "fn main() {\n" + "".join(law(*o) for o in ops) + body + "    0.0 -> zero: f32;\n"
    for o, length in outs:
        s += (f"    (zero, {o}) -> fold {{ acc, v -> acc + v }} -> s_{o};\n    s_{o} -> println;\n"
              f"    {o}[0] -> println;\n    {o}[{length - 1}] -> println;\n")
    return s + "}\n"


def firew_rowsum(R, C, taps, timed=True):
    """FIR -> ew -> rowsum: a 1-D window over R·C samples, a per-sample activation, a sum per row of C."""
    body = (f"    {R * C} -> iota -> o_y;\n    {taps} -> iota -> k_y;\n"
            f"    o_y -> map {{ t -> (0.0, k_y) -> fold {{ acc, k -> acc + c[k] * s[t + k] }} }} -> y;\n"
            f"    y -> map {{ v -> {relu('v * 0.5')} }} -> h;\n"
            f"    {R} -> iota -> r_i;\n    {C} -> iota -> r_j;\n"
            f"    r_i -> map {{ i -> (0.0, r_j) -> fold {{ acc, j -> acc + h[i * {C} + j] }} }} -> r;\n")
    return chain_program([("c", taps, 'e'), ("s", R * C + taps - 1, 'a')], body, [("r", R)], timed)


def attention(T, d, timed=True):
    """matmul -> softmax -> matmul: s = q·Kᵀ / sqrt(d) (K read at the lane stride d), a row max, exp of the
    difference, a row sum, the quotient, p·V."""
    sd = f"{d ** 0.5:.1f}"
    body = (f"    {T * T} -> iota -> o_s;\n    {d} -> iota -> k_d;\n    {T} -> iota -> k_t;\n"
            f"    o_s -> map {{ e ->\n        e / {T} -> i;\n        e % {T} -> j;\n"
            f"        (0.0, k_d) -> fold {{ acc, k -> acc + q[i * {d} + k] * kk[j * {d} + k] }}\n    }} -> s;\n"
            f"    k_t -> map {{ i -> (-1000000000.0, k_t) -> fold {{ acc, j -> s[i * {T} + j] -> e; "
            f"(e > acc) -> {{ -true-> e; -false-> acc; }} }} }} -> mx;\n"
            f"    o_s -> map {{ t -> {expo(f'(s[t] - mx[t / {T}]) / {sd}')} }} -> ex;\n"
            f"    k_t -> map {{ i -> (0.0, k_t) -> fold {{ acc, j -> acc + ex[i * {T} + j] }} }} -> den;\n"
            f"    o_s -> map {{ t -> ex[t] / den[t / {T}] }} -> p;\n"
            f"    {T * d} -> iota -> o_o;\n"
            f"    o_o -> map {{ e ->\n        e / {d} -> i;\n        e % {d} -> j;\n"
            f"        (0.0, k_t) -> fold {{ acc, u -> acc + p[i * {T} + u] * vv[u * {d} + j] }}\n    }} -> o;\n")
    ops = [("q", T * d, 'a'), ("kk", T * d, 'b'), ("vv", T * d, 'e')]
    return chain_program(ops, body, [("o", T * d)], timed)


def long_softmax(R, L, timed=True):
    """long-rows: scores q·W (d = 16, a tile site: the rows start on the device) -> softmax over few long rows:
    a row max, exp of the difference, a row sum, the quotient — a vocabulary's logits, a long cache's scores."""
    body = (f"    {R * L} -> iota -> o_s;\n    16 -> iota -> k_d;\n    {R} -> iota -> r_i;\n    {L} -> iota -> k_t;\n"
            f"    o_s -> map {{ e ->\n        e / {L} -> i;\n        e % {L} -> j;\n"
            f"        (0.0, k_d) -> fold {{ acc, k -> acc + q[i * 16 + k] * w[k * {L} + j] }}\n    }} -> s;\n"
            f"    r_i -> map {{ i -> (-1000000000.0, k_t) -> fold {{ acc, j -> s[i * {L} + j] -> e; "
            f"(e > acc) -> {{ -true-> e; -false-> acc; }} }} }} -> mx;\n"
            f"    o_s -> map {{ t -> {expo(f'(s[t] - mx[t / {L}]) / 4.0')} }} -> ex;\n"
            f"    r_i -> map {{ i -> (0.0, k_t) -> fold {{ acc, j -> acc + ex[i * {L} + j] }} }} -> den;\n"
            f"    o_s -> map {{ t -> ex[t] / den[t / {L}] }} -> y;\n")
    return chain_program([("q", R * 16, 'a'), ("w", 16 * L, 'b')], body, [("y", R * L)], timed)


def ewconvew(n, kk, timed=True):
    """ew -> conv -> ew: an activation of the image, a kk×kk convolution (k÷kk, k%kk taps), an activation."""
    w = n + kk - 1
    body = (f"    img -> map {{ v -> v * 0.5 + 0.25 }} -> im;\n"
            f"    {n * n} -> iota -> o_y;\n    {kk * kk} -> iota -> k_y;\n"
            f"    o_y -> map {{ t ->\n        t / {n} -> i;\n        t % {n} -> j;\n"
            f"        (0.0, k_y) -> fold {{ acc, k -> acc + wk[k] * im[(i + k / {kk}) * {w} + j + k % {kk}] }}\n"
            f"    }} -> y;\n"
            f"    y -> map {{ v -> {relu('v')} }} -> z;\n")
    return chain_program([("wk", kk * kk, 'e'), ("img", w * w, 'a')], body, [("z", n * n)], timed)


def smlp(M, d, h, timed=True):
    """MLP: relu(x·W1 + b1)·W2."""
    body = (f"    {h} -> iota -> i_b1;\n"
            f"    i_b1 -> map {{ t -> t % 13 - 6 -> widen_f32 -> v; v / 5.0 }} -> b1;\n"
            + gsite("s1", "x", "w1", M, h, d)
            + f"    o_s1 -> map {{ t -> {relu(f's1[t] + b1[t % {h}]')} }} -> hh;\n"
            + gsite("s2", "hh", "w2", M, d, h))
    return chain_program([("x", M * d, 'a'), ("w1", d * h, 'b'), ("w2", h * d, 'e')], body, [("s2", M * d)], timed)


CHAINS = {
    "pt_firew_rowsum_2048x2048_16": (firew_rowsum(2048, 2048, 16), 16 + 2048),
    "pt_attn_1024_64": (attention(1024, 64), 64 + 2 * 1024),
    "pt_attn_1024_128": (attention(1024, 128), 128 + 2 * 1024),
    "pt_attn_2048_64": (attention(2048, 64), 64 + 2 * 2048),
    "pt_attn_2048_128": (attention(2048, 128), 128 + 2 * 2048),
    "pt_ewconvew_2048_5": (ewconvew(2048, 5), 25),
    "pt_smlp_4096_256_1024": (smlp(4096, 256, 1024), 256 + 1024),
    "pt_lsm_1_131072": (long_softmax(1, 131072), 16 + 2 * 131072),
    "pt_lsm_4_32768": (long_softmax(4, 32768), 16 + 2 * 32768),
    "pt_lsm_32_8192": (long_softmax(32, 8192), 16 + 2 * 8192),
    "px_firew_rowsum_32x32_16": (firew_rowsum(32, 32, 16, False), 16 + 32),
    "px_attn_64_16": (attention(64, 16, False), 16 + 2 * 64),
    "px_ewconvew_32_5": (ewconvew(32, 5, False), 25),
    "px_smlp_64_32_64": (smlp(64, 32, 64, False), 32 + 64),
    "px_lsm_2_512": (long_softmax(2, 512, False), 16 + 2 * 512),
}
for stem, (src, kmax) in CHAINS.items():
    (out / f"{stem}.mapal").write_text(src)
    stated.add(stem)
    (out / f"{stem}.kmax").write_text(f"{kmax}\n")


# flow: every timed program carries its gate's K (box_run.sh fails a default face without one — the
# facts-on-graph sweep bit-compared split-K / stream-K builds of pt_mlp2, pt_twin, pt_pj, pt_epi instead).
def fold_chain(src):
    """An upper bound on the rounded steps any value of a loop-free program passes through: per top-level
    statement, its folds' extents plus its arithmetic operators, plus the longest chain it reads."""
    assert "loop {" not in src, "a loop's K is stated where it is written"
    lens, chain, stmt = {}, {}, ""
    for line in src.splitlines():
        stmt += line + "\n"
        if not re.fullmatch(r"    \S.*;", line):
            continue
        it, m = re.fullmatch(r"    (\d+) -> iota -> (\w+);", line), re.search(r"-> (\w+)(?:: \w+)?;$", line)
        if it:
            lens[it.group(2)] = int(it.group(1))
        elif m:
            # site-geometry: a map's output is as long as what it maps (a fold over it is that long).
            src_map = re.match(r"    (\w+) -> map \{", stmt)
            if src_map and src_map.group(1) in lens:
                lens[m.group(1)] = lens[src_map.group(1)]
            w = sum(lens[r] for r in re.findall(r", (\w+)\) -> fold", stmt)) + len(re.findall(r" [-+*/] ", stmt))
            reads = set(re.findall(r"\w+", stmt)) - {m.group(1)}
            chain[m.group(1)] = w + max((chain[x] for x in reads if x in chain), default=0)
        stmt = ""
    return max(chain.values())


assert fold_chain(probe_program(64, probes(64)['mlp2'], True)) >= 2 * 64
assert fold_chain(probe_program(64, twin(64), True)) < 2 * 64  # two independent sites: one chain each
for n in (1024, 2048):  # one request's chain: the requests are independent (+ the outputs' join)
    for tag in ('rt', 'rg'):
        for name in ('mlp2', 'indep'):
            stem = f"pt_{tag}_{name}_{n}"
            if stem not in stated:
                stated.add(stem)
                (out / f"{stem}.kmax").write_text(f"{fold_chain(probe_program(n, probes(n)[name], True)) + 1}\n")
for f in sorted(out.glob("pt_*.mapal")):
    # (a `--ladder` program left in `out` by an earlier run carries its own `.kmax`)
    if f.stem not in stated and not (out / f"{f.stem}.kmax").exists():
        (out / f"{f.stem}.kmax").write_text(f"{fold_chain(f.read_text())}\n")


# recurrence-precision (ADR-0041): `: tf32 | f32` twins, the same face as torch's TF32 legs — every fold step
# admits TF32 products (`-> fold: tf32 | f32 { … }`). `.kmax` as their base program's.
def tf32_folds(src):
    return src.replace("-> fold {", "-> fold: tf32 | f32 {")


for stem in ("pt_convmc_8_64_56_3", "px_convmc_2_4_6_3", "pt_mlp_2048", "pt_mlp2_2048"):
    (out / f"{stem}_tf32.mapal").write_text(tf32_folds((out / f"{stem}.mapal").read_text()))
    if (out / f"{stem}.kmax").exists():
        (out / f"{stem}_tf32.kmax").write_text((out / f"{stem}.kmax").read_text())


# recurrence-precision smoke: M through fold chains, relu and bias adds. Every closed-form input by its
# magnitude makes every term, bias and relu input non-negative, so each printed value of the exact build is
# its chain's M (relu is 1-Lipschitz). The 4090 smoke failed pt_pc_32 and pt_mlp2_2048_tf32 on the relative
# fallback under cancellation (|exact| << M) with the error inside D2 / D3 of M.
def mabs(src):
    out, n = re.subn(r'-> widen_f32 -> v; v / ([0-9.]+) \}',
                     r'-> widen_f32 -> v; (v > 0.0) -> { -true-> v; -false-> 0.0 - v; } -> va; va / \1 }', src)
    assert n > 0, "no closed-form input"
    return out.replace("-> fold: tf32 | f32 {", "-> fold {")  # M is the exact face's


for stem in ("pt_pc_32", "pt_pc_64", "pt_pc_128", "pt_mlp_2048", "pt_mlp2_2048", "pt_mlp_2048_tf32", "pt_mlp2_2048_tf32"):
    (out / f"{stem}_xabs.mapal").write_text(mabs((out / f"{stem}.mapal").read_text()))


# site-geometry: every loop-free program's K, so its default face answers to D2's bound in the differential
# (box_run.sh `_d`); one fold_chain cannot bound (a loop, a fold over a function's argument) states its K where it
# is written, or its default face is reported unchecked (changed.py env UNCHECKED).
for f in sorted(out.glob("*.mapal")):
    if not (out / f"{f.stem}.kmax").exists():
        try:
            (out / f"{f.stem}.kmax").write_text(f"{fold_chain(f.read_text())}\n")
        except (AssertionError, KeyError):
            pass


# --- per-site-choice scale ladders (`--ladder`, opt-in: ladder_bundle.sh): every task family from small to the
# largest size that fits a part, from the generators above (only the size parameters change). `ladder.tsv`: stem,
# family, size label, work units, unit, gate (`oracle`: the exact CPU build; `xref`: the exact face on the device,
# the interpreter's order bit for bit; `cell`: tile_bench's f64 reference + the cell's exact face).
ORACLE_MAX = 2e10  # flops (elements for `elem`) the exact CPU oracle builds in well under a minute
if "--ladder" in sys.argv:
    lad = []

    def step(stem, fam, size, units, unit, src=None, kmax=None, gate=None):
        if src is not None:
            (out / f"{stem}.mapal").write_text(src)
        if kmax is not None:
            (out / f"{stem}.kmax").write_text(f"{kmax}\n")
        elif src is not None and not (out / f"{stem}.kmax").exists():
            (out / f"{stem}.kmax").write_text(f"{fold_chain(src)}\n")
        lad.append((stem, fam, size, int(units), unit, gate or ("oracle" if units <= ORACLE_MAX else "xref")))

    for n in (256, 512, 1024, 2048, 4096, 8192, 16384):
        c = f"mm_{n}x{n}x{n}"
        step(c, "gemm-square-cell", n, 2 * n ** 3, "flop", None if (out / f"{c}.mapal").exists()
             else program(n, n, n, "rm", "rm", "f32"), gate="cell")
        step(f"pt_pj_m{n}_n{n}_k{n}", "gemm-square", n, 2 * n ** 3, "flop",
             composed([("a", n * n, 'a'), ("b", n * n, 'b')], gsite("c", "a", "b", n, n, n), [("c", n * n)]))
    for N, K in ((2048, 7168), (1024, 4096)):
        for M in (1, 16, 128, 1024, 4096):
            c = f"mm_{M}x{N}x{K}"
            step(c, f"gemm-batch-n{N}k{K}-cell", M, 2 * M * N * K, "flop", None if (out / f"{c}.mapal").exists()
                 else program(M, N, K, "rm", "rm", "f32"), gate="cell")
            step(f"pt_pj_m{M}_n{N}_k{K}", f"gemm-batch-n{N}k{K}", M, 2 * M * N * K, "flop",
                 composed([("a", M * K, 'a'), ("b", K * N, 'b')], gsite("c", "a", "b", M, N, K), [("c", M * N)]))
    sites = dict(indep=2, twin=2, mlp=2, mlp2=4, fanout=2, diamond=3)
    for n in (512, 1024, 2048, 4096, 8192):
        for name, s in sites.items():
            src = probe_program(n, twin(n) if name == "twin" else probes(n)[name], True)
            step(f"pt_{name}_{n}", f"chain-{name}", n, s * 2 * n ** 3, "flop", src)
    for d in (64, 128):
        for T in (512, 1024, 2048, 4096, 8192):
            step(f"pt_attn_{T}_{d}", f"attention-d{d}", T, 4 * T * T * d, "flop", attention(T, d), d + 2 * T)
    for T, d in ((256, 256), (512, 256), (1024, 256), (2048, 256), (4096, 256), (1024, 1024), (1024, 2048), (1024, 4096)):
        ax = "context" if d == 256 else "width"
        size = T if d == 256 else d
        units = T * (4 * d * d + 4 * T * d)  # per trip: K, V's written row + the attention over T rows
        for fam, src in (("decode", timed_loop(decode(T, d))),
                         ("prefill", timed_loop(decode(T, d, prefill=T // 2)).replace("t1 - t0", "t1 - pf0")),
                         ("gen", timed_loop(decode(T, d, gen=T // 2)))):
            stem = f"pt_decode_{T}_{d}" if fam == "decode" else f"pt_{fam}_{T}_{T // 2}_{d}"
            # The exact CPU build runs the loop as written: its oracle only up to the suite's 1024 x 256.
            g = "oracle" if T * d <= 1024 * 256 else "xref"
            step(stem, f"{fam}-{ax}", size, units, "flop", src, T * d, g)
            if d == 256 and T == 1024:  # the shared point is also the width ladder's first step
                step(stem, f"{fam}-width", d, units, "flop", gate=g)
    for C, n in ((64, 56), (128, 56), (256, 56), (512, 56), (64, 112), (64, 224)):
        ax, size = ("channels", C) if n == 56 else ("res", n)
        src = conv_mc(8, C, n, 3)
        for face, s in (("", src), ("_tf32", tf32_folds(src))):
            for a, sz in ((ax, size),) + ((("res", 56),) if (C, n) == (64, 56) else ()):
                step(f"pt_convmc_8_{C}_{n}_3{face}", f"convmc{face or '_f32'}-{a}", sz,
                     2 * 8 * C * n * n * C * 9, "flop", s, C * 9)
    for n in (1024, 2048, 4096, 8192, 16384):
        for name in ("ew", "epi", "bias", "residual", "rowsum", "softmax_rows"):
            src = probe_program(n, probes(n)[name], True)
            ew = name == "ew"
            k = {"ew": 8 * n, "rowsum": 2 * n, "softmax_rows": 2 * n}.get(name)
            step(f"pt_{name}_{n}", name, n, n * n if ew else 2 * n ** 3, "elem" if ew else "flop", src, k)
            if name == "rowsum":
                (out / f"pt_rowsum_{n}_xabs.mapal").write_text(xabs(src))
    for e in (20, 22, 24, 26, 28):
        step(f"pt_fir_{1 << e}_16", "fir", 1 << e, 2 * 16 << e, "flop", fir(1 << e, 16), 16)
    for s in (32, 64, 128):
        step(f"pt_pc_{s}", "cpugpu-pc", s, 6 * s ** 3, "flop", composed(*small_chain(s)))
    for n, s in ((2048, 32), (4096, 64), (8192, 128)):
        step(f"pt_pm_{n}_{s}", "cpugpu-pm", n, 6 * s ** 3 + 4 * n ** 3, "flop", composed(*mixed(n, s)))
    (out / "ladder.tsv").write_text("stem\tfamily\tsize\tunits\tunit\tgate\n"
                                    + "".join("\t".join(map(str, r)) + "\n" for r in lad))


# --- model-scale: one Llama-3-8B / Qwen-7B-like layer (d_model 4096, 32 query heads x 128, 8 KV heads (GQA), SwiGLU
# FFN 14336, RMSNorm, RoPE) at shapes that fit a 16 GB card, f32. `pt_llm_dec_b<B>_l<L>`: S decode steps of the
# attention half (RMSNorm -> fused QKV -> RoPE -> KV-cache append -> GQA attention over the cache -> O -> residual),
# B sequences, cache capacity L ([B][L][Hk][dh] for K and V; step t writes position L - S + t and attends to every
# position up to it; the layer's output is the next step's input); `pt_llm_blkdec_*` the same with the FFN half
# (RMSNorm -> gate, up -> SiLU(gate)·up -> down -> residual): one whole block per step. `pt_llm_pre_t<T>` /
# `pt_llm_blkpre_t<T>`: causal prefill of T tokens. At T = 8192 the score arrays of all 32 heads (32·T² = 2³¹
# cells: past i32, and 8 GiB each for the scores and their exponentials) do not fit, so that attention runs as a
# loop over the 8 KV-head groups (4 query heads, T² scores each).
# exp is the `exp` pipeline builtin (ADR-0042). Not in the Core op set: sqrt (`llm_rms`: Heron's iteration, 16 steps from (1 + a)/2), sin / cos (the rotary tables are
# inputs, as inference engines precompute them). Operand laws: OPS's values with `t % mod` taken first, so no
# product leaves i32 at 2^28 cells (torch_bench.py `llm_*` builds the same tensors — change both or neither).
# Printed: out[0], out[last], Σ out² over the last row (a cancellation-free checksum of the last token, which reads
# every position), `iter ms`. `px_llm_*`: the same generators at a few-element scale, the interpreter their oracle.
LLM = dict(D=4096, H=32, Hk=8, dh=128, F=14336)
LLM_PX = dict(D=64, H=4, Hk=2, dh=16, F=96)


def llm_law(name, length, key, body):
    """`((t % mod)·mul + add) % mod − off` widened (OPS's values, i32-safe), then `body` of `v`."""
    mul, add, mod, off, _ = OPS[key]
    return (f"    {length} -> iota -> i_{name};\n"
            f"    i_{name} -> map {{ t -> (t % {mod} * {mul} + {add}) % {mod} - {off} -> widen_f32 -> v; {body} }} -> {name};\n")


def llm_div(k):
    """A weight's divisor: the law's values have std ~29, so std(W) = 1/sqrt(k) over a k-long fold."""
    return f"{29 * k ** 0.5:.1f}"


def llm_exp(z, n):
    """e^z: the `exp` pipeline builtin (ADR-0042; `n` names nothing now, kept for the callers)."""
    return f"{z} -> exp"


def llm_silu(v):
    """v·sigmoid(v) = v / (1 + e^-v)."""
    return f"{v} -> sv; 0.0 - sv -> exp -> en; sv / (1.0 + en)"


def llm_rms(row, D):
    """1 / sqrt(mean(row(i)²) + 1e-5) over i in rD: Heron's iteration, 16 steps from (1 + a) / 2 >= sqrt(a)."""
    return (f"(0.0, rD) -> fold {{ acc, i -> {row} -> rv; acc + rv * rv }} -> ss; ss / {D}.0 + 0.00001 -> ra; "
            f"0.5 + 0.5 * ra -> s0; (s0, r16) -> fold {{ acc, k -> 0.5 * (acc + ra / acc) }} -> sd; 1.0 / sd")


def llm_rope(src, base, c, pos, dh):
    """Llama's rotate_half RoPE of cell c of a head-concatenated vector at `src[base + c]`, position `pos`."""
    h = dh // 2
    return (f"{c} % {dh} -> ri; (ri < {h}) -> {{ "
            f"-true-> {src}[{base} + {c}] * cs[{pos} * {h} + ri] - {src}[{base} + {c} + {h}] * sn[{pos} * {h} + ri]; "
            f"-false-> {src}[{base} + {c}] * cs[{pos} * {h} + ri - {h}] + {src}[{base} + {c} - {h}] * sn[{pos} * {h} + ri - {h}]; }}")


def llm_rope_pv(src, base, c, pos, dh, closed=False):
    """`llm_rope` with every index provable (model-scale findings 2-3, the `_pv` diagnostics): the partner cell as
    `(ri + dh/2) % dh` of its head, the sign by a guard on the value; `closed`: the rotary values as the tables'
    closed form of (pos, ri % dh/2) — a decode's position is loop-carried, and no index on it is proven."""
    h = dh // 2
    s = (f"{c} / {dh} -> rh; {c} % {dh} -> ri; ri % {h} -> rj; (ri < {h}) -> {{ -true-> -1.0; -false-> 1.0; }} -> sg; "
         f"{src}[{base} + rh * {dh} + ri] -> va; {src}[{base} + rh * {dh} + (ri + {h}) % {dh}] -> vb; ")
    if closed:
        mf, af, modf, offf, _ = OPS["f"]
        mg, ag, modg, offg, _ = OPS["g"]
        return s + (f"({pos} * {h} + rj) % {modf} * {mf} + {af} -> cq; (cq % {modf} - {offf}) -> widen_f32 -> cv; "
                    f"({pos} * {h} + rj) % {modg} * {mg} + {ag} -> sq; (sq % {modg} - {offg}) -> widen_f32 -> sv; "
                    f"va * (cv / 50.0) + sg * vb * (sv / 50.0)")
    return s + f"va * cs[{pos} * {h} + rj] + sg * vb * sn[{pos} * {h} + rj]"


# half-precision (ADR-0043): `w16` stores the layer's weights (wqkv, wo, wg, wu, wd) as `bf16` — each the f32 law's
# value rounded once (`-> to_bf16`), read through its exact widening (`w[i] -> widen_f32`), products and folds f32:
# half the weight bytes a decode step moves. The torch legs build the same tensors (`.to(bfloat16)`).
LLM_WEIGHTS = ("wqkv", "wo", "wg", "wu", "wd")


def llm_ops(P, rows, positions, cache, ffn, w16=False):
    """The layer's inputs: (name, length, key, body of v)."""
    D, H, Hk, dh, F = P["D"], P["H"], P["Hk"], P["dh"], P["F"]
    NQ, half = (H + 2 * Hk) * dh, dh // 2
    ops = [("x0", rows * D, "a", "v / 29.0"), ("g1", D, "e", "1.0 + v / 400.0"),
           ("wqkv", D * NQ, "b", f"v / {llm_div(D)}"), ("wo", D * D, "e", f"v / {llm_div(D)}"),
           ("cs", positions * half, "f", "v / 50.0"), ("sn", positions * half, "g", "v / 50.0")]
    if cache:
        ops += [("kc0", cache, "p", "v / 29.0"), ("vc0", cache, "a", "v / 29.0")]
    if ffn:
        ops += [("g2", D, "b", "1.0 + v / 400.0"), ("wg", D * F, "f", f"v / {llm_div(D)}"),
                ("wu", D * F, "g", f"v / {llm_div(D)}"), ("wd", F * D, "p", f"v / {llm_div(F)}")]
    if w16:
        ops = [(n, ln, k, f"{b} -> to_bf16" if n in LLM_WEIGHTS else b) for n, ln, k, b in ops]
    return ops


def llm_w(name, idx, w16):
    """A fold step's read of weight `name` at `idx`: `(binding, value)` — the element itself, or (`w16`) its exact
    widening bound first."""
    if not w16:
        return "", f"{name}[{idx}]"
    return f"{name}[{idx}] -> widen_f32 -> wv; ", "wv"


def llm_ffn(P, src, dst, w16=False):
    """`dst = src + down(SiLU(gate(n)) · up(n))`, n = RMSNorm(src)."""
    D, F = P["D"], P["F"]
    (bg, vg), (bu, vu), (bd, vd) = (llm_w("wg", f"k * {F} + j", w16), llm_w("wu", f"k * {F} + j", w16),
                                    llm_w("wd", f"k * {D} + j", w16))
    return (f"    rR -> map {{ b -> {llm_rms(f'{src}[b * {D} + i]', D)} }} -> rinv2;\n"
            f"    rRD -> map {{ e -> {src}[e] * rinv2[e / {D}] * g2[e % {D}] }} -> hn;\n"
            f"    rRF -> map {{ e -> e / {F} -> b; e % {F} -> j; (0.0, rD) -> fold {{ acc, k -> {bg}acc + hn[b * {D} + k] * {vg} }} }} -> gt;\n"
            f"    rRF -> map {{ e -> e / {F} -> b; e % {F} -> j; (0.0, rD) -> fold {{ acc, k -> {bu}acc + hn[b * {D} + k] * {vu} }} }} -> up;\n"
            f"    rRF -> map {{ e -> {llm_silu('gt[e]')} -> sg; sg * up[e] }} -> hh;\n"
            f"    rRD -> map {{ e -> e / {D} -> b; e % {D} -> j; (0.0, rF) -> fold {{ acc, k -> {bd}acc + hh[b * {F} + k] * {vd} }} -> y2; {src}[e] + y2 }} -> {dst};\n")


def llm_head(P, rows, ffn, extra=""):
    D, H, Hk, dh, F = P["D"], P["H"], P["Hk"], P["dh"], P["F"]
    s = (f"    {rows} -> iota -> rR;\n    {D} -> iota -> rD;\n    {rows * D} -> iota -> rRD;\n"
         f"    {rows * (H + 2 * Hk) * dh} -> iota -> rRQ;\n    {dh} -> iota -> rdh;\n    16 -> iota -> r16;\n")
    if ffn:
        s += f"    {F} -> iota -> rF;\n    {rows * F} -> iota -> rRF;\n"
    return s + extra


def llm_tail(P, rows, timed, name="out"):
    """Four cells (the first, the last row's second, one mid, the last), then (untimed twins) Σ out² over the
    whole output. A loop's exit value is not visible to a later map or fold body (L1101, model-scale finding 1):
    every read here is a top-level index or a fold over the array itself."""
    D = P["D"]
    s = "    () -> time -> t1;\n" if timed else ""
    for c in (0, (rows - 1) * D + 1, rows * D // 2 + 3, rows * D - 1):
        s += f"    {name}[{c}] -> println;\n"
    if timed:
        s += '    "iter ms=" -> print;\n    t1 - t0 -> println;\n'
    else:
        s += f"    0.0 -> zero: f32;\n    (zero, {name}) -> fold {{ acc, v -> acc + v * v }} -> ssq;\n    ssq -> println;\n"
    return s + "}\n"


def llm_loop(cond, body, step, exit_):
    body = "".join("            " + ln.lstrip() + "\n" for ln in body.splitlines())
    return (f"    loop {{\n        ({cond}) -> {{\n            -true-> {{\n{body}"
            f"                {step};\n                -> loop;\n            }}\n            -false-> {exit_};\n        }}\n    }}\n")


def llm_decode(B, L, S, ffn, timed=True, P=LLM, pv=False, w16=False):
    """S decode steps of B sequences against a KV cache of capacity L (`w16`: bf16 weights, ADR-0043)."""
    D, H, Hk, dh = P["D"], P["H"], P["Hk"], P["dh"]
    G, KD, NQ, sq = H // Hk, Hk * dh, (H + 2 * Hk) * dh, f"{dh ** 0.5:.7f}"
    C = B * L * KD
    rope = (lambda *a: llm_rope_pv(*a, closed=True)) if pv else llm_rope
    (bq, vq), (bo, vo) = llm_w("wqkv", f"k * {NQ} + j", w16), llm_w("wo", f"k * {D} + j", w16)
    s = "fn main() {\n" + "".join(llm_law(*o) for o in llm_ops(P, B, L, C, ffn, w16))
    s += llm_head(P, B, ffn, f"    {B * H} -> iota -> rBH;\n    {B * H * L} -> iota -> rBHL;\n"
                             f"    {C} -> iota -> rC;\n    {L} -> iota -> rL;\n")
    s += (f"    mut kc: [f32; {C}] <- kc0;\n    mut vc: [f32; {C}] <- vc0;\n    mut x: [f32; {B * D}] <- x0;\n"
          f"    mut t: i32 <- 0;\n")
    s += "    () -> time -> t0;\n" if timed else ""
    body = (
        f"    {L - S} + t -> p;\n"
        f"    rR -> map {{ b -> {llm_rms(f'x[b * {D} + i]', D)} }} -> rinv;\n"
        f"    rRD -> map {{ e -> x[e] * rinv[e / {D}] * g1[e % {D}] }} -> xn;\n"
        f"    rRQ -> map {{ e -> e / {NQ} -> b; e % {NQ} -> j; (0.0, rD) -> fold {{ acc, k -> {bq}acc + xn[b * {D} + k] * {vq} }} }} -> qkv;\n"
        f"    rRD -> map {{ e -> e / {D} -> b; e % {D} -> c; {rope('qkv', f'b * {NQ}', 'c', 'p', dh)} }} -> q;\n"
        f"    rC -> map {{ c -> c / {L * KD} -> b; c / {KD} % {L} -> r; (r == p) -> {{ -true-> {{ c % {KD} -> u; "
        f"{rope('qkv', f'b * {NQ} + {D}', 'u', 'p', dh)} }} -false-> kc[c]; }} }} -> kc;\n"
        f"    rC -> map {{ c -> c / {L * KD} -> b; c / {KD} % {L} -> r; (r == p) -> {{ -true-> qkv[b * {NQ} + {D + KD} + c % {KD}]; -false-> vc[c]; }} }} -> vc;\n"
        f"    rBHL -> map {{ e -> e / {H * L} -> b; e / {L} % {H} -> h; e % {L} -> r; (r <= p) -> {{ -true-> {{ (0.0, rdh) -> fold {{ acc, i -> "
        f"acc + q[b * {D} + h * {dh} + i] * kc[((b * {L} + r) * {Hk} + h / {G}) * {dh} + i] }} -> dt; dt / {sq} }} -false-> -1000000000.0; }} }} -> s;\n"
        f"    rBH -> map {{ bh -> (-1000000000.0, rL) -> fold {{ acc, r -> s[bh * {L} + r] -> se; (se > acc) -> {{ -true-> se; -false-> acc; }} }} }} -> mx;\n"
        f"    rBHL -> map {{ e -> {llm_exp(f's[e] - mx[e / {L}]', 'x')} }} -> pe;\n"
        f"    rBH -> map {{ bh -> (0.0, rL) -> fold {{ acc, r -> acc + pe[bh * {L} + r] }} }} -> den;\n"
        f"    rRD -> map {{ e -> e / {D} -> b; e % {D} -> c; c / {dh} -> h; c % {dh} -> i; (0.0, rL) -> fold {{ acc, r -> "
        f"acc + pe[(b * {H} + h) * {L} + r] * vc[((b * {L} + r) * {Hk} + h / {G}) * {dh} + i] }} -> av; av / den[b * {H} + h] }} -> att;\n"
        f"    rRD -> map {{ e -> e / {D} -> b; e % {D} -> j; (0.0, rD) -> fold {{ acc, k -> {bo}acc + att[b * {D} + k] * {vo} }} -> y; x[e] + y }} -> {'h1' if ffn else 'x'};\n")
    if ffn:
        body += llm_ffn(P, "h1", "x", w16)
    s += llm_loop(f"t < {S}", body, "t + 1 -> t", "x -> out")
    return s + llm_tail(P, B, timed)


def llm_attend(P, B, L, kread, vread):
    """The decode step's GQA attention over positions <= p (scores, max, exp, sum, P·V): `kread(b, r, kv, i)` /
    `vread(...)` the K / V element of sequence b, position r, KV head kv, lane i."""
    D, H, Hk, dh = P["D"], P["H"], P["Hk"], P["dh"]
    G, sq = H // Hk, f"{dh ** 0.5:.7f}"
    return (
        f"    rBHL -> map {{ e -> e / {H * L} -> b; e / {L} % {H} -> h; e % {L} -> r; (r <= p) -> {{ -true-> {{ (0.0, rdh) -> fold {{ acc, i -> "
        f"acc + q[b * {D} + h * {dh} + i] * {kread('b', 'r', f'h / {G}', 'i')} }} -> dt; dt / {sq} }} -false-> -1000000000.0; }} }} -> s;\n"
        f"    rBH -> map {{ bh -> (-1000000000.0, rL) -> fold {{ acc, r -> s[bh * {L} + r] -> se; (se > acc) -> {{ -true-> se; -false-> acc; }} }} }} -> mx;\n"
        f"    rBHL -> map {{ e -> {llm_exp(f's[e] - mx[e / {L}]', 'x')} }} -> pe;\n"
        f"    rBH -> map {{ bh -> (0.0, rL) -> fold {{ acc, r -> acc + pe[bh * {L} + r] }} }} -> den;\n"
        f"    rRD -> map {{ e -> e / {D} -> b; e % {D} -> c; c / {dh} -> h; c % {dh} -> i; (0.0, rL) -> fold {{ acc, r -> "
        f"acc + pe[(b * {H} + h) * {L} + r] * {vread('b', 'r', f'h / {G}', 'i')} }} -> av; av / den[b * {H} + h] }} -> att;\n")


def llm_decode_kv(B, L, S, ffn, naive, timed=True, P=LLM):
    """caching-memory: S decode steps over a history of L positions per sequence (`hx0`: the layer input at every
    position; the steps write positions L - S + t), two ways a user writes the same layer:

    - explicit cache (`naive=False`, `pt_llm_kvx_*`): K and V of the whole history computed once (RMSNorm, the
      K|V columns of the fused projection, RoPE by position), then each step appends its token's row to the
      loop-carried caches `kc` / `vc` (as `llm_decode`);
    - naive (`naive=True`, `pt_llm_kvn_*`): no cache — the loop carries the history `hx` itself; each step writes
      its input row into it and recomputes RMSNorm, K, V and RoPE for every position from it. That only the
      written rows' K and V differ from the last step is the compiler's to find.

    Same math, same fold orders: the two print the same values up to the default face's reorderings."""
    D, H, Hk, dh = P["D"], P["H"], P["Hk"], P["dh"]
    KD, NQ, KK = Hk * dh, (H + 2 * Hk) * dh, 2 * Hk * dh
    C, HD = B * L * KD, B * L * D
    (bq, vq), (bo, vo) = llm_w("wqkv", f"k * {NQ} + j", False), llm_w("wo", f"k * {D} + j", False)
    ops = llm_ops(P, B, L, 0, ffn) + [("hx0", HD, "p", "v / 29.0")]
    s = "fn main() {\n" + "".join(llm_law(*o) for o in ops)
    s += llm_head(P, B, ffn, f"    {B * H} -> iota -> rBH;\n    {B * H * L} -> iota -> rBHL;\n"
                             f"    {C} -> iota -> rC;\n    {L} -> iota -> rL;\n    {B * L} -> iota -> rBL;\n"
                             f"    {HD} -> iota -> rHD;\n    {B * L * KK} -> iota -> rKK;\n"
                             f"    {B * L * NQ} -> iota -> rBLQ;\n")

    def kv_of(src, rinv, xn, kv):
        """RMSNorm of every history row, then its K|V columns: `kv[(b·L + r)·KK + …]`."""
        return (f"    rBL -> map {{ b -> {llm_rms(f'{src}[b * {D} + i]', D)} }} -> {rinv};\n"
                f"    rHD -> map {{ e -> {src}[e] * {rinv}[e / {D}] * g1[e % {D}] }} -> {xn};\n"
                f"    rKK -> map {{ e -> e / {KK} -> b; e % {KK} -> j; (0.0, rD) -> fold {{ acc, k -> "
                f"{bq}acc + {xn}[b * {D} + k] * {vq.replace('+ j', f'+ {D} + j')} }} }} -> {kv};\n")

    def rope_k(kv, dst):
        return (f"    rC -> map {{ c -> c / {KD} -> b; c % {KD} -> u; "
                f"{llm_rope(kv, f'b * {KK}', 'u', f'(b % {L})', dh)} }} -> {dst};\n")

    if naive:
        # The layer's projection over every position of the history, then RoPE by position: K for every row, the
        # current token's q from its own row — nothing about a cache.
        s += f"    mut hx: [f32; {HD}] <- hx0;\n    mut x: [f32; {B * D}] <- x0;\n    mut t: i32 <- 0;\n"
        s += "    () -> time -> t0;\n" if timed else ""
        body = (f"    {L - S} + t -> p;\n"
                f"    rHD -> map {{ c -> (c / {D} % {L} == p) -> {{ -true-> x[c / {L * D} * {D} + c % {D}]; -false-> hx[c]; }} }} -> hx;\n"
                f"    rBL -> map {{ b -> {llm_rms('hx[b * ' + str(D) + ' + i]', D)} }} -> rinvh;\n"
                f"    rHD -> map {{ e -> hx[e] * rinvh[e / {D}] * g1[e % {D}] }} -> xnh;\n"
                f"    rBLQ -> map {{ e -> e / {NQ} -> b; e % {NQ} -> j; (0.0, rD) -> fold {{ acc, k -> "
                f"{bq}acc + xnh[b * {D} + k] * {vq} }} }} -> qkvh;\n"
                f"    rC -> map {{ c -> c / {KD} -> b; c % {KD} -> u; "
                f"{llm_rope('qkvh', f'b * {NQ} + {D}', 'u', f'(b % {L})', dh)} }} -> kh;\n"
                f"    rRD -> map {{ e -> e / {D} -> b; e % {D} -> c; "
                f"{llm_rope('qkvh', f'(b * {L} + p) * {NQ}', 'c', 'p', dh)} }} -> q;\n")
        body += llm_attend(P, B, L, lambda b, r, kv, i: f"kh[(({b} * {L} + {r}) * {Hk} + {kv}) * {dh} + {i}]",
                           lambda b, r, kv, i: f"qkvh[({b} * {L} + {r}) * {NQ} + {D + KD} + {kv} * {dh} + {i}]")
    else:
        s += "    () -> time -> t0;\n" if timed else ""
        s += kv_of("hx0", "rinvh", "xnh", "kvh") + rope_k("kvh", "kc0")
        s += f"    rC -> map {{ c -> kvh[c / {KD} * {KK} + {KD} + c % {KD}] }} -> vc0;\n"
        s += (f"    mut kc: [f32; {C}] <- kc0;\n    mut vc: [f32; {C}] <- vc0;\n    mut x: [f32; {B * D}] <- x0;\n"
              f"    mut t: i32 <- 0;\n")
        body = (
            f"    {L - S} + t -> p;\n"
            f"    rR -> map {{ b -> {llm_rms(f'x[b * {D} + i]', D)} }} -> rinv;\n"
            f"    rRD -> map {{ e -> x[e] * rinv[e / {D}] * g1[e % {D}] }} -> xn;\n"
            f"    rRQ -> map {{ e -> e / {NQ} -> b; e % {NQ} -> j; (0.0, rD) -> fold {{ acc, k -> {bq}acc + xn[b * {D} + k] * {vq} }} }} -> qkv;\n"
            f"    rRD -> map {{ e -> e / {D} -> b; e % {D} -> c; {llm_rope('qkv', f'b * {NQ}', 'c', 'p', dh)} }} -> q;\n"
            f"    rC -> map {{ c -> c / {L * KD} -> b; c / {KD} % {L} -> r; (r == p) -> {{ -true-> {{ c % {KD} -> u; "
            f"{llm_rope('qkv', f'b * {NQ} + {D}', 'u', 'p', dh)} }} -false-> kc[c]; }} }} -> kc;\n"
            f"    rC -> map {{ c -> c / {L * KD} -> b; c / {KD} % {L} -> r; (r == p) -> {{ -true-> qkv[b * {NQ} + {D + KD} + c % {KD}]; -false-> vc[c]; }} }} -> vc;\n")
        body += llm_attend(P, B, L, lambda b, r, kv, i: f"kc[(({b} * {L} + {r}) * {Hk} + {kv}) * {dh} + {i}]",
                           lambda b, r, kv, i: f"vc[(({b} * {L} + {r}) * {Hk} + {kv}) * {dh} + {i}]")
    body += (f"    rRD -> map {{ e -> e / {D} -> b; e % {D} -> j; (0.0, rD) -> fold {{ acc, k -> {bo}acc + att[b * {D} + k] * {vo} }} -> y; "
             f"x[e] + y }} -> {'h1' if ffn else 'x'};\n")
    if ffn:
        body += llm_ffn(P, "h1", "x")
    s += llm_loop(f"t < {S}", body, "t + 1 -> t", "x -> out")
    return s + llm_tail(P, B, timed)


def llm_prefill(T, ffn, groups=False, timed=True, P=LLM, pv=False):
    """Causal prefill of T tokens; `groups`: the attention as a loop over the KV-head groups."""
    D, H, Hk, dh = P["D"], P["H"], P["Hk"], P["dh"]
    G, KD, NQ, sq = H // Hk, Hk * dh, (H + 2 * Hk) * dh, f"{dh ** 0.5:.7f}"
    nh = G if groups else H  # query heads per score array
    rope = llm_rope_pv if pv else llm_rope
    s = "fn main() {\n" + "".join(llm_law(*o) for o in llm_ops(P, T, T, 0, ffn))
    s += llm_head(P, T, ffn, f"    {T} -> iota -> rT;\n    {T * KD} -> iota -> rTK;\n"
                             f"    {nh * T} -> iota -> rHT;\n    {nh * T * T} -> iota -> rHTT;\n")
    if groups:
        s += f"    {T * G * dh} -> iota -> rTG;\n    {G * dh} -> iota -> rGd;\n    mut y: [f32; {T * D}] <- x0;\n    mut g: i32 <- 0;\n"
    s += "    () -> time -> t0;\n" if timed else ""
    s += (f"    rR -> map {{ b -> {llm_rms(f'x0[b * {D} + i]', D)} }} -> rinv;\n"
          f"    rRD -> map {{ e -> x0[e] * rinv[e / {D}] * g1[e % {D}] }} -> xn;\n"
          f"    rRQ -> map {{ e -> e / {NQ} -> i; e % {NQ} -> j; (0.0, rD) -> fold {{ acc, k -> acc + xn[i * {D} + k] * wqkv[k * {NQ} + j] }} }} -> qkv;\n"
          f"    rRD -> map {{ e -> e / {D} -> i; e % {D} -> c; {rope('qkv', f'i * {NQ}', 'c', 'i', dh)} }} -> q;\n"
          f"    rTK -> map {{ e -> e / {KD} -> i; e % {KD} -> u; {rope('qkv', f'i * {NQ} + {D}', 'u', 'i', dh)} }} -> kr;\n")
    hq = f"g * {G} + hh" if groups else "hh"  # a score array's head -> its query head, and that head's KV head
    kvh = "g" if groups else f"hh / {G}"
    attn = (
        f"    rHTT -> map {{ e -> e / {T * T} -> hh; e / {T} % {T} -> i; e % {T} -> r; (r <= i) -> {{ -true-> {{ (0.0, rdh) -> fold {{ acc, u -> "
        f"acc + q[i * {D} + ({hq}) * {dh} + u] * kr[r * {KD} + ({kvh}) * {dh} + u] }} -> dt; dt / {sq} }} -false-> -1000000000.0; }} }} -> s;\n"
        f"    rHT -> map {{ hi -> (-1000000000.0, rT) -> fold {{ acc, r -> s[hi * {T} + r] -> se; (se > acc) -> {{ -true-> se; -false-> acc; }} }} }} -> mx;\n"
        f"    rHTT -> map {{ e -> {llm_exp(f's[e] - mx[e / {T}]', 'x')} }} -> pe;\n"
        f"    rHT -> map {{ hi -> (0.0, rT) -> fold {{ acc, r -> acc + pe[hi * {T} + r] }} }} -> den;\n")
    pv = (f"(0.0, rT) -> fold {{ acc, r -> acc + pe[(hh * {T} + i) * {T} + r] * qkv[r * {NQ} + {D + KD} + ({kvh}) * {dh} + u] }} -> av; "
          f"av / den[hh * {T} + i]")
    if groups:
        # A group's 4 heads, then their rows of the O projection accumulated into the residual stream (the O
        # projection's sum split by head group: the exit value goes straight to the prints, finding 1).
        GD = G * dh
        attg = (f"    rTG -> map {{ e -> e / {GD} -> i; e % {GD} -> c; c / {dh} -> hh; c % {dh} -> u; {pv} }} -> attg;\n"
                f"    rRD -> map {{ e -> e / {D} -> i; e % {D} -> j; (0.0, rGd) -> fold {{ acc, k -> acc + attg[i * {GD} + k] * "
                f"wo[(g * {GD} + k) * {D} + j] }} -> yo; y[e] + yo }} -> y;\n")
        s += llm_loop(f"g < {Hk}", attn + attg, "g + 1 -> g", "y -> out")
        assert not ffn, "the grouped attention's output is the program's"
        return s + llm_tail(P, T, timed)
    s += attn + f"    rRD -> map {{ e -> e / {D} -> i; e % {D} -> c; c / {dh} -> hh; c % {dh} -> u; {pv} }} -> attf;\n"
    s += (f"    rRD -> map {{ e -> e / {D} -> i; e % {D} -> j; (0.0, rD) -> fold {{ acc, k -> acc + attf[i * {D} + k] * wo[k * {D} + j] }} -> y; "
          f"x0[e] + y }} -> {'h1' if ffn else 'out'};\n")
    if ffn:
        s += llm_ffn(P, "h1", "out")
    return s + llm_tail(P, T, timed)


def llm_kmax(P, ffn, L, steps, ssq=0):
    """An upper bound on the rounded steps a printed value passes through (fold_chain's rule, by hand: the loops):
    per step RMSNorm, QKV, scores, max, sum, P·V, O (+ the FFN), then (`ssq` cells) the untimed twins' Σ out²."""
    D, dh, F = P["D"], P["dh"], P["F"]
    return steps * (3 * D + 3 * L + dh + 80 + ((2 * D + F + 120) if ffn else 0)) + ssq


LLM_STEPS = 16  # decode steps per timed run (ms per step = iter ms / 16)
llm_progs = {}
for pv in ("", "_pv"):
    # `_pv`: the diagnostics of findings 2-3 — the same layer with the RoPE indices written provable
    # (`llm_rope_pv`), so the attention can be placed on the device; not benchmarks of the natural form.
    for B in (1, 8, 32):
        for L in (2048, 8192):
            llm_progs[f"pt_llm_dec_b{B}_l{L}{pv}"] = (llm_decode(B, L, LLM_STEPS, False, pv=bool(pv)),
                                                     llm_kmax(LLM, False, L, LLM_STEPS))
    llm_progs[f"pt_llm_blkdec_b1_l4096{pv}"] = (llm_decode(1, 4096, LLM_STEPS, True, pv=bool(pv)),
                                               llm_kmax(LLM, True, 4096, LLM_STEPS))
    llm_progs[f"pt_llm_pre_t2048{pv}"] = (llm_prefill(2048, False, pv=bool(pv)), llm_kmax(LLM, False, 2048, 1))
    llm_progs[f"pt_llm_blkpre_t2048{pv}"] = (llm_prefill(2048, True, pv=bool(pv)), llm_kmax(LLM, True, 2048, 1))
# half-precision (ADR-0043): the decode rows with bf16 weights (`_bf16w`) — bandwidth-bound, half the weight bytes.
for B in (1, 8, 32):
    for L in (2048, 8192):
        llm_progs[f"pt_llm_dec_b{B}_l{L}_bf16w"] = (llm_decode(B, L, LLM_STEPS, False, w16=True),
                                                   llm_kmax(LLM, False, L, LLM_STEPS))
llm_progs["pt_llm_blkdec_b1_l4096_bf16w"] = (llm_decode(1, 4096, LLM_STEPS, True, w16=True),
                                            llm_kmax(LLM, True, 4096, LLM_STEPS))
# caching-memory: the explicit-cache (`kvx`) and naive (`kvn`) twins over a history (`llm_decode_kv`): the attention
# half at the decode shapes whose naive state fits 16 GB, and the block.
for B, L in ((1, 2048), (1, 8192), (8, 2048), (8, 8192), (32, 2048)):
    for kind in ("kvx", "kvn"):
        llm_progs[f"pt_llm_{kind}_b{B}_l{L}"] = (llm_decode_kv(B, L, LLM_STEPS, False, kind == "kvn"),
                                                llm_kmax(LLM, False, L, LLM_STEPS + 1))
for kind in ("kvx", "kvn"):
    llm_progs[f"pt_llm_{kind}_blkdec_b1_l4096"] = (llm_decode_kv(1, 4096, LLM_STEPS, True, kind == "kvn"),
                                                   llm_kmax(LLM, True, 4096, LLM_STEPS + 1))
# (The grouped loop indexes by its counter, which no rewrite makes provable: no `_pv` twin.)
llm_progs["pt_llm_pre_t8192"] = (llm_prefill(8192, False, groups=True), llm_kmax(LLM, False, 8192, 1))
# The interpreter-sized twins (oracle: the interpreter; box_run.sh's differential, exact face and `_d`).
for pv in ("", "_pv"):
    llm_progs[f"px_llm_dec_b2_l16{pv}"] = (llm_decode(2, 16, 3, False, False, LLM_PX, bool(pv)),
                                          llm_kmax(LLM_PX, False, 16, 3, 128))
    llm_progs[f"px_llm_blkdec_b2_l16{pv}"] = (llm_decode(2, 16, 3, True, False, LLM_PX, bool(pv)),
                                             llm_kmax(LLM_PX, True, 16, 3, 128))
    llm_progs[f"px_llm_pre_t12{pv}"] = (llm_prefill(12, False, timed=False, P=LLM_PX, pv=bool(pv)),
                                       llm_kmax(LLM_PX, False, 12, 1, 768))
    llm_progs[f"px_llm_blkpre_t12{pv}"] = (llm_prefill(12, True, timed=False, P=LLM_PX, pv=bool(pv)),
                                          llm_kmax(LLM_PX, True, 12, 1, 768))
llm_progs["px_llm_dec_b2_l16_bf16w"] = (llm_decode(2, 16, 3, False, False, LLM_PX, w16=True),
                                       llm_kmax(LLM_PX, False, 16, 3, 128))
llm_progs["px_llm_blkdec_b2_l16_bf16w"] = (llm_decode(2, 16, 3, True, False, LLM_PX, w16=True),
                                          llm_kmax(LLM_PX, True, 16, 3, 128))
for kind in ("kvx", "kvn"):
    llm_progs[f"px_llm_{kind}_b2_l16"] = (llm_decode_kv(2, 16, 3, False, kind == "kvn", False, LLM_PX),
                                         llm_kmax(LLM_PX, False, 16, 4, 128))
    llm_progs[f"px_llm_{kind}_blkdec_b2_l16"] = (llm_decode_kv(2, 16, 3, True, kind == "kvn", False, LLM_PX),
                                                llm_kmax(LLM_PX, True, 16, 4, 128))
llm_progs["px_llm_preg_t12"] = (llm_prefill(12, False, groups=True, timed=False, P=LLM_PX),
                                llm_kmax(LLM_PX, False, 12, 1, 768))
# (d) the layer's projections alone: QKV (fused, N = 6144), O, gate/up (fused, N = 28672), down (K = 14336) —
# `composed`'s projection (torch_bench.py `pt_pj`, gemm_lib.cu) at m = 1, 16, 128, 2048.
for M in (1, 16, 128, 2048):
    for N, K in ((6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336)):
        llm_progs[f"pt_pj_m{M}_n{N}_k{K}"] = (
            composed([("a", M * K, 'a'), ("b", K * N, 'b')], gsite("c", "a", "b", M, N, K), [("c", M * N)]), K)
# The TF32 face (ADR-0041: `: tf32 | f32` on every fold) where GEMM products dominate: the prefills, the prefill
# block and the m >= 128 projections.
for stem in [s for s in llm_progs if s.startswith(("pt_llm_pre_t2048", "pt_llm_blkpre", "pt_pj_m128_", "pt_pj_m2048_"))]:
    src, k = llm_progs[stem]
    llm_progs[f"{stem}_tf32"] = (tf32_folds(src), k)
for stem, (src, k) in llm_progs.items():
    (out / f"{stem}.mapal").write_text(src)
    (out / f"{stem}.kmax").write_text(f"{k}\n")
    # The projections' D2 / D3 magnitude oracle (`mabs`): a corner whose |exact| << M under cancellation fails the
    # relative fallback inside the absolute bound (run 1: three TF32 corners at 1e-2 relative, |d| = 0.1 of M ~ 4e4).
    if stem.startswith("pt_pj_"):
        (out / f"{stem}_xabs.mapal").write_text(mabs(src))


# --- the formatter owns the layout of generated source too (CLAUDE.md "Source formatting"): every program this
# run wrote goes through `mapal fmt`, so benches/algorithms and the bundles hold formatted text. The CLI is $MAPAL,
# else `mapal` on PATH, else `cargo run -p mapal-cli`. Formatting never changes the parsed program (emission is
# byte-identical), and a program the formatter rejects is a generator bug, so a failure stops the run.
def format_written(paths):
    import os
    import shutil
    import subprocess

    if not paths:
        return
    cli = os.environ.get("MAPAL") or shutil.which("mapal")
    root = Path(__file__).resolve().parents[2]
    cmd = [cli] if cli else ["cargo", "run", "-q", "--release", "-p", "mapal-cli", "--"]
    for i in range(0, len(paths), 200):
        subprocess.run(cmd + ["fmt"] + [str(p) for p in paths[i:i + 200]], cwd=root, check=True,
                       stdout=subprocess.DEVNULL)


format_written(sorted(out.glob("*.mapal")))
