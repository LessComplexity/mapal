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
print("wide_2048x2048x64\nfmod_64\ndeep_4096x1024x8192")


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


def probe_program(n, spec, timed, w=None):
    names, body, outs = spec
    s = head(n) + probe_ops(names, n, w)
    tail = "" if timed else "    0.0 -> zero: f32;\n"
    for o, length in outs:
        if not timed:
            tail += f"    (zero, {o}) -> fold {{ acc, v -> acc + v }} -> s_{o};\n    s_{o} -> println;\n"
        tail += f"    {o}[0] -> println;\n    {o}[{length - 1}] -> println;\n"
    if timed:
        # Declarations (a loop's `mut` state) stay outside the clock.
        lines = body.splitlines(keepends=True)
        decl = [x for x in lines if x.startswith("    mut ")]
        rest = [x for x in lines if not x.startswith("    mut ")]
        s += "".join(decl) + "    () -> time -> t0;\n" + "".join(rest) + "    () -> time -> t1;\n"
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
for n in (2048, 4096):
    for name, spec in probes(n).items():
        (out / f"pt_{name}_{n}.mapal").write_text(probe_program(n, spec, True))
    (out / f"pt_loop_{n}.mapal").write_text(probe_program(n, loop_act(n, 8), True, w='w'))
