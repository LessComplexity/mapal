//! Mapal-source generators: `exp` and math builtins, and GEMM shapes.

use super::*;

// ---------------------------------------------------------------------------
// exp (ADR-0042): `x -> exp` over every branch of the oracle's algorithm.
// ---------------------------------------------------------------------------

/// A program taking `exp` of values that reach every branch of the oracle's
/// algorithm — normal results, overflow to `inf`, subnormal results, `0`,
/// `±inf` and `NaN` arguments (a quotient by a zero difference) — in `f32`
/// and `f64`, in a cell, in a fold body and through a row's max (a softmax);
/// sums and cells printed. `scale` sets the range (`2^scale / 8` times the
/// law's ±`h`).
#[derive(Clone, Debug)]
pub struct ExpProg {
    pub n: u64,
    pub rows: u64,
    pub mul: u64,
    pub add: u64,
    pub m: u64,
    pub scale: u8,
    pub wide: bool,
    pub nan: bool,
    pub softmax: bool,
    pub in_fold: bool,
}

pub fn exp_prog_strategy() -> impl Strategy<Value = ExpProg> {
    (
        prop::sample::select(vec![256u64, 512, 1024, 2048]),
        prop::sample::select(vec![1u64, 2, 4]),
        prop::sample::select(vec![7u64, 11, 13, 29]),
        0u64..64,
        prop::sample::select(vec![101u64, 211, 401]),
        0u8..6,
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(
            |(n, rows, mul, add, m, scale, wide, nan, softmax, in_fold)| ExpProg {
                n,
                rows,
                mul,
                add,
                m,
                scale,
                wide,
                nan,
                softmax,
                in_fold,
            },
        )
}

/// [`ExpProg`]'s source text. `x` is a tile site's output (`q · W`, `d =
/// 8`: the cells reading it run on the device), scaled.
pub fn exp_src(p: &ExpProg) -> String {
    let (n, r) = (p.n, p.rows);
    let l = n / r;
    let sc = f64::from(1u32 << p.scale) / 8.0;
    let mut s = format!(
        "fn main() {{\n    {n} -> iota -> t;\n    {r} -> iota -> rr;\n    {l} -> iota -> kl;\n    8 -> iota -> kd;\n    \
         {rq} -> iota -> tq;\n    tq -> map {{ e -> (e * {mul} + {add}) % 31 - 15 -> widen<f32> -> v; v / 4.0 }} -> q;\n    \
         {lw} -> iota -> tw;\n    tw -> map {{ e -> (e * 7 + 13) % {m} - {h} -> widen<f32> -> v; v / 8.0 }} -> w;\n    \
         t -> map {{ e -> e / {l} -> i; e % {l} -> j; (0.0, kd) -> fold {{ acc, d -> acc + q[i * 8 + d] * w[d * {l} + j] }} -> u; u * {sc:.3} }} -> x;\n",
        rq = r * 8,
        lw = l * 8,
        mul = p.mul,
        add = p.add,
        m = p.m,
        h = p.m / 2,
    );
    // A cell: `exp` of the value, or (every 7th cell from 0 and 1) of `NaN`
    // and of `±inf` (a quotient by a zero difference).
    let arg = if p.nan {
        "x[e] - x[e] -> z; (e % 7 == 0) -> { -true-> z / z; -false-> x[e] / z; } -> a; (e % 7 < 2) -> { -true-> a; -false-> x[e]; }".to_string()
    } else {
        "x[e]".to_string()
    };
    s += &format!("    t -> map {{ e -> {arg} -> exp }} -> y;\n");
    // `f64`: a range past f64's overflow and underflow.
    if p.wide {
        s += "    t -> map { e -> x[e] -> widen<f64> -> d; d * 8.0 -> exp } -> z;\n";
        s += "    0.0 -> z0: f64;\n    (z0, z) -> fold { acc, v -> acc + v / 1000000.0 } -> zs;\n";
    }
    // A fold whose step reads `exp` of its element.
    if p.in_fold {
        s += &format!(
            "    rr -> map {{ i -> (0.0, kl) -> fold {{ acc, j -> x[i * {l} + j] / 16.0 -> exp -> w; acc + w }} }} -> fs;\n"
        );
    }
    // A softmax over the rows (the rescale law's form).
    if p.softmax {
        s += &format!(
            "    rr -> map {{ i -> (-1000000000.0, kl) -> fold {{ acc, j -> x[i * {l} + j] -> w; (w > acc) -> {{ -true-> w; -false-> acc; }} }} }} -> mx;\n    \
             t -> map {{ e -> x[e] - mx[e / {l}] -> exp }} -> ex;\n    \
             rr -> map {{ i -> (0.0, kl) -> fold {{ acc, j -> acc + ex[i * {l} + j] }} }} -> den;\n    \
             t -> map {{ e -> ex[e] / den[e / {l}] }} -> sm;\n    sm[{a}] -> println;\n    den[{b}] -> println;\n",
            a = n / 3,
            b = r - 1
        );
    }
    for i in [0, 1, n / 5, n / 2 + 1, n - 1] {
        s += &format!("    y[{i}] -> println;\n");
    }
    if p.wide {
        s += &format!(
            "    z[{}] -> println;\n    z[{}] -> println;\n    zs -> println;\n",
            n / 7,
            n - 2
        );
    }
    if p.in_fold {
        s += &format!("    fs[{}] -> println;\n", r - 1);
    }
    s += "    0.0 -> y0: f32;\n    (y0, y) -> fold { acc, v -> acc + v / 1000000.0 } -> ys;\n    ys -> println;\n}\n";
    s
}

// ---------------------------------------------------------------------------
// math builtins (ADR-0046): `x -> sqrt`, `x -> sin`, `x -> cos` over every
// branch of their references.
// ---------------------------------------------------------------------------

/// A program taking `sqrt`, `sin` and `cos` (the ones `fns` selects: bit 0
/// `sqrt`, 1 `sin`, 2 `cos`) of values that reach every branch of the
/// references — moderate, large (Cody–Waite's range), huge (Payne–Hanek's:
/// `x·10³⁰`, near `f32::MAX`), tiny and subnormal (`x·10⁻⁴⁰`), near
/// multiples of `π/2`, and a cell law of special values (`±0`, `±∞`, NaN,
/// the smallest subnormal, negatives under `sqrt`) — in `f32` and `f64`, in
/// map cells (a tile site's output: device cells) and fold bodies; cells and
/// sums printed. `scale` sets the range (`2^scale / 8` times the law's ±`h`).
#[derive(Clone, Debug)]
pub struct MathProg {
    pub n: u64,
    pub rows: u64,
    pub mul: u64,
    pub add: u64,
    pub m: u64,
    pub scale: u8,
    pub fns: u8,
    pub wide: bool,
    pub in_fold: bool,
}

pub fn math_prog_strategy() -> impl Strategy<Value = MathProg> {
    (
        prop::sample::select(vec![256u64, 512, 1024]),
        prop::sample::select(vec![1u64, 2, 4]),
        prop::sample::select(vec![7u64, 11, 13, 29]),
        0u64..64,
        prop::sample::select(vec![101u64, 211, 401]),
        0u8..6,
        1u8..8,
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(
            |(n, rows, mul, add, m, scale, fns, wide, in_fold)| MathProg {
                n,
                rows,
                mul,
                add,
                m,
                scale,
                fns,
                wide,
                in_fold,
            },
        )
}

/// The builtins a [`MathProg`] takes.
pub fn math_fns(p: &MathProg) -> Vec<&'static str> {
    ["sqrt", "sin", "cos"]
        .into_iter()
        .enumerate()
        .filter(|(i, _)| p.fns & (1 << i) != 0)
        .map(|(_, f)| f)
        .collect()
}

/// [`MathProg`]'s source text.
pub fn math_src(p: &MathProg) -> String {
    let (n, r) = (p.n, p.rows);
    let l = n / r;
    let sc = f64::from(1u32 << p.scale) / 8.0;
    let mut s = format!(
        "fn main() {{\n    {n} -> iota -> t;\n    {r} -> iota -> rr;\n    {l} -> iota -> kl;\n    8 -> iota -> kd;\n    \
         {rq} -> iota -> tq;\n    tq -> map {{ e -> (e * {mul} + {add}) % 31 - 15 -> widen<f32> -> v; v / 4.0 }} -> q;\n    \
         {lw} -> iota -> tw;\n    tw -> map {{ e -> (e * 7 + 13) % {m} - {h} -> widen<f32> -> v; v / 8.0 }} -> w;\n    \
         t -> map {{ e -> e / {l} -> i; e % {l} -> j; (0.0, kd) -> fold {{ acc, d -> acc + q[i * 8 + d] * w[d * {l} + j] }} -> u; u * {sc:.3} }} -> x;\n    \
         16 -> iota -> ts;\n",
        rq = r * 8,
        lw = l * 8,
        mul = p.mul,
        add = p.add,
        m = p.m,
        h = p.m / 2,
    );
    // Decimal literals (Mapal has no exponent notation): `10^k` and `10^-k`.
    let big = |k: usize| format!("1{}.0", "0".repeat(k));
    let small = |k: usize| format!("0.{}1", "0".repeat(k - 1));
    // The argument law: by `e % 8` a range class of the cell's value.
    let arg32 = format!(
        "x[e] -> v; e % 8 -> c; (c == 0) -> {{ -true-> v; -false-> (c == 1) -> {{ -true-> v * {e6}; -false-> (c == 2) -> {{ -true-> v * {e30}; -false-> (c == 3) -> {{ -true-> v * {em40}; -false-> (c == 4) -> {{ -true-> v * 1.5707963; -false-> (c == 5) -> {{ -true-> v * 0.001; -false-> (c == 6) -> {{ -true-> v * 100.0; -false-> v * {e38}; }}; }}; }}; }}; }}; }}; }}",
        e6 = big(6),
        e30 = big(30),
        em40 = small(40),
        e38 = big(38),
    );
    // Special values, one a cell of `ts`: ±0, ±∞, NaN, the smallest
    // subnormal, −1, near `f32::MAX`, near `65·π/2`, the f32 worst
    // cancellation `16367173·2^72`.
    let special32 = format!(
        "(k == 0) -> {{ -true-> 0.0; -false-> (k == 1) -> {{ -true-> -1.0 * 0.0; -false-> (k == 2) -> {{ -true-> {inf}; -false-> (k == 3) -> {{ -true-> -{inf}; -false-> (k == 4) -> {{ -true-> z / z; -false-> (k == 5) -> {{ -true-> {sub}; -false-> (k == 6) -> {{ -true-> -1.0; -false-> (k == 7) -> {{ -true-> 340000000000000000000000000000000000000.0; -false-> (k == 8) -> {{ -true-> 102.10176; -false-> (k == 9) -> {{ -true-> 16367173.0 * 4722366482869645213696.0; -false-> x[k] * 3.0; }}; }}; }}; }}; }}; }}; }}; }}; }}; }}",
        inf = big(39),
        sub = small(45),
    );
    let fns = math_fns(p);
    for f in &fns {
        // `sqrt` of the square keeps the classes non-negative (their
        // negatives are the special law's).
        let a = if *f == "sqrt" { "a * a" } else { "a" };
        s += &format!("    t -> map {{ e -> {arg32} -> a; {a} -> {f} }} -> y_{f};\n");
        s += &format!(
            "    ts -> map {{ k -> 0.0 -> z0; z0 - z0 -> z; {special32} -> {f} }} -> s_{f};\n"
        );
    }
    if p.wide {
        let arg64 = format!(
            "x[e] -> widen<f64> -> d; e % 5 -> c; (c == 0) -> {{ -true-> d * 7.0; -false-> (c == 1) -> {{ -true-> d * {e300}; -false-> (c == 2) -> {{ -true-> d * {em310}; -false-> (c == 3) -> {{ -true-> d * {e15}; -false-> d * 1.5707963267948966; }}; }}; }}; }}",
            e300 = big(300),
            em310 = small(310),
            e15 = big(15),
        );
        for f in &fns {
            let a = if *f == "sqrt" { "a * a" } else { "a" };
            s += &format!("    t -> map {{ e -> {arg64} -> a; {a} -> {f} }} -> z_{f};\n");
            s += &format!(
                "    0.0 -> zz_{f}: f64;\n    (zz_{f}, z_{f}) -> fold {{ acc, v -> acc + v / 1000000.0 }} -> zs_{f};\n"
            );
        }
    }
    if p.in_fold {
        for f in &fns {
            s += &format!(
                "    rr -> map {{ i -> (0.0, kl) -> fold {{ acc, j -> x[i * {l} + j] -> v; v * v -> {f} -> w; acc + w }} }} -> fs_{f};\n"
            );
        }
    }
    for f in &fns {
        for i in [0, 1, 2, 3, 4, 5, 6, 7, n / 5, n / 2 + 3, n - 1] {
            s += &format!("    y_{f}[{i}] -> println;\n");
        }
        for k in 0..16 {
            s += &format!("    s_{f}[{k}] -> println;\n");
        }
        s += &format!(
            "    0.0 -> y0_{f}: f32;\n    (y0_{f}, y_{f}) -> fold {{ acc, v -> acc + v / 1000000.0 }} -> ys_{f};\n    ys_{f} -> println;\n"
        );
        if p.wide {
            for i in [0, 1, 2, 3, 4, n / 7, n - 2] {
                s += &format!("    z_{f}[{i}] -> println;\n");
            }
            s += &format!("    zs_{f} -> println;\n");
        }
        if p.in_fold {
            s += &format!("    fs_{f}[{}] -> println;\n", r - 1);
        }
    }
    s += "}\n";
    s
}

// --- large-gemm: GEMM shapes from decode GEMVs to model-scale projections ----

/// One `m × k · k × n` product (`y[i·n + j] = Σ_k a[i·k + k'] · b[k'·n + j]`,
/// `b` row-major, or transposed under `bt`) with small integer operands in
/// `-2 ..= 2`, so every partial sum is exact in f32 up to `k` of 2²² and any
/// cut or order of the fold prints the same bits. `scale` picks the extents:
///
/// - 0 skinny: `m` 1–32 (odd 3, 24), `n` 1024–4096, `k` 512–4096 — decode
///   projections, whose one-warp tile grid leaves the part short of warps;
/// - 1 under-filled deep: `m` 64–256 (odd 96, 192), `n` 1024–8192, `k`
///   4096–16384 (odd 6144, 12288) — a prefill chunk's projections;
/// - 2 filled: `m` 1024–4096, `n` 1024–8192, `k` 1024–8192 (odd 3072);
/// - 3 odd: extents a register tile need not divide (`m` 100, 136, 200; `n`
///   1000, 1500, 2040; `k` 1000, 4100, 8000).
///
/// Large extents are for plans only (emitted, never run); [`gemm_run_src`]
/// gives the sizes an emulator runs.
#[derive(Clone, Debug)]
pub struct GemmProg {
    pub scale: u8,
    pub dims: [u8; 3],
    pub bt: bool,
    pub law: [u8; 2],
}

pub fn gemm_prog_strategy() -> impl Strategy<Value = GemmProg> {
    (
        0u8..4,
        prop::array::uniform3(any::<u8>()),
        prop::bool::weighted(0.2),
        prop::array::uniform2(any::<u8>()),
    )
        .prop_map(|(scale, dims, bt, law)| GemmProg {
            scale,
            dims,
            bt,
            law,
        })
}

/// [`GemmProg`]'s `(m, n, k)`.
pub fn gemm_dims(p: &GemmProg) -> (u64, u64, u64) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let [dm, dn, dk] = p.dims;
    match p.scale {
        0 => (
            pick(dm, &[1, 3, 8, 16, 24, 32]),
            pick(dn, &[1024, 2048, 3072, 4096]),
            pick(dk, &[512, 1024, 4096]),
        ),
        1 => (
            pick(dm, &[64, 96, 128, 192, 256]),
            pick(dn, &[1024, 2048, 4096, 6144, 8192]),
            pick(dk, &[4096, 6144, 8192, 12288, 16384]),
        ),
        2 => (
            pick(dm, &[1024, 2048, 4096]),
            pick(dn, &[1024, 4096, 8192]),
            pick(dk, &[1024, 3072, 8192]),
        ),
        _ => (
            pick(dm, &[100, 136, 200]),
            pick(dn, &[1000, 1500, 2040]),
            pick(dk, &[1000, 4100, 8000]),
        ),
    }
}

/// The source of a [`GemmProg`] at `(m, n, k)`: the operands from a law in
/// `-2 ..= 2`, the product as a map over the output's cells folding `k`, and
/// five cells printed (the first, the last, the last row's first, a middle
/// one, the second row's last).
pub fn gemm_src_at(p: &GemmProg, m: u64, n: u64, k: u64) -> String {
    let bi = match p.bt {
        true => format!("b[j * {k} + k]"),
        false => format!("b[k * {n} + j]"),
    };
    gemm_src_reading(p, m, n, k, &bi, k * n)
}

/// gap-fixes merge (`stream-ld`): a [`GemmProg`] whose `b` is a window —
/// `b[j + k]`, `n + k - 1` elements, each read by up to `k` cells (the FIR's
/// signal: the lane and fold axes overlap) — at `(m, n, k)`.
pub fn gemm_window_src_at(p: &GemmProg, m: u64, n: u64, k: u64) -> String {
    gemm_src_reading(p, m, n, k, "b[j + k]", n + k - 1)
}

/// [`gemm_src_at`] with `b` read as `bi` from `kn` elements.
pub(super) fn gemm_src_reading(p: &GemmProg, m: u64, n: u64, k: u64, bi: &str, kn: u64) -> String {
    let (a1, a2) = (
        3 + 2 * u64::from(p.law[0] % 7),
        5 + 2 * u64::from(p.law[1] % 5),
    );
    let mut s = format!(
        "fn main() {{
    {mk} -> iota -> ia;
    ia -> map {{
        t -> (t * {a1} + 13) % 5 - 2 -> widen<f32>
    }} -> a;
    {kn} -> iota -> ib;
    ib -> map {{
        t -> (t * {a2} + 7) % 5 - 2 -> widen<f32>
    }} -> b;
    {k} -> iota -> kr;
    {mn} -> iota -> ic;
    ic -> map {{
        t ->
        t / {n} -> i;
        t % {n} -> j;
        (0.0, kr) -> fold {{
            acc, k -> acc + a[i * {k} + k] * {bi}
        }}
    }} -> y;
",
        mk = m * k,
        mn = m * n
    );
    let last = m * n - 1;
    for c in [0, last, (m - 1) * n, (m * n) / 2 + n / 3, 2 * n - 1] {
        s += &format!("    y[{}] -> println;\n", c.min(last));
    }
    s + "}\n"
}

/// ADR-0050 review: **a [`GemmProg`] with float operands whose values are
/// checked against the a-priori bound** — `a`, `b` non-integer (`v / 7`,
/// `v / 9` in f32: no order of the sum is exact), at `(m, n, k)`; the cells
/// `(i, j)` with `i ≡ r₀ (mod si)` and `j ≡ c₀ (mod sj)` (a cell in every
/// `si × sj` block, so every output tile of at least that size is sampled)
/// printed in order. Returns the source and the cells; [`gemm_bound_ref`]
/// gives each one's exact sum and `M = Σ_k |a·b|`.
pub fn gemm_bound_src_at(
    p: &GemmProg,
    m: u64,
    n: u64,
    k: u64,
    (si, sj): (u64, u64),
) -> (String, Vec<(u64, u64)>) {
    let ((a1, a2), (r0, c0)) = gemm_bound_laws(p, (si, sj));
    let bi = match p.bt {
        true => format!("b[j * {k} + q]"),
        false => format!("b[q * {n} + j]"),
    };
    let cells: Vec<(u64, u64)> = (r0..m)
        .step_by(si as usize)
        .flat_map(|i| (c0..n).step_by(sj as usize).map(move |j| (i, j)))
        .collect();
    let mut s = format!(
        "fn main() {{
    {mk} -> iota -> ia;
    ia -> map {{
        t -> (t * {a1} + 13) % 101 - 50 -> widen<f32> -> v;
        v / 7.0
    }} -> a;
    {kn} -> iota -> ib;
    ib -> map {{
        t -> (t * {a2} + 57) % 103 - 51 -> widen<f32> -> v;
        v / 9.0
    }} -> b;
    {k} -> iota -> kr;
    {mn} -> iota -> ic;
    ic -> map {{
        t ->
        t / {n} -> i;
        t % {n} -> j;
        (0.0, kr) -> fold {{
            acc, q -> acc + a[i * {k} + q] * {bi}
        }}
    }} -> y;
",
        mk = m * k,
        kn = k * n,
        mn = m * n,
    );
    for &(i, j) in &cells {
        s += &format!("    y[{}] -> println;\n", i * n + j);
    }
    (s + "}\n", cells)
}

/// [`gemm_bound_src_at`]'s operand laws `(a1, a2)` and first sampled cell.
pub(super) fn gemm_bound_laws(p: &GemmProg, (si, sj): (u64, u64)) -> ((u64, u64), (u64, u64)) {
    (
        (
            3 + 2 * u64::from(p.law[0] % 7),
            5 + 2 * u64::from(p.law[1] % 5),
        ),
        (u64::from(p.dims[0]) % si, u64::from(p.dims[1]) % sj),
    )
}

/// The exact sum and `M = Σ_k |a·b|` of each of [`gemm_bound_src_at`]'s
/// cells: the f32 operands as the program makes them (an integer widened,
/// divided in f32), their products and sums in f64 (exact products, a sum
/// within `k·2⁻⁵³·M` — far below any member's bound).
pub fn gemm_bound_ref(
    p: &GemmProg,
    (_m, n, k): (u64, u64, u64),
    (si, sj): (u64, u64),
    cells: &[(u64, u64)],
) -> Vec<(f64, f64)> {
    let ((a1, a2), _) = gemm_bound_laws(p, (si, sj));
    let a = |t: u64| ((((t * a1 + 13) % 101) as i64 - 50) as f32 / 7.0f32) as f64;
    let b = |t: u64| ((((t * a2 + 57) % 103) as i64 - 51) as f32 / 9.0f32) as f64;
    cells
        .iter()
        .map(|&(i, j)| {
            (0..k).fold((0.0, 0.0), |(s, mm), q| {
                let bt = match p.bt {
                    true => j * k + q,
                    false => q * n + j,
                };
                let x = a(i * k + q) * b(bt);
                (s + x, mm + x.abs())
            })
        })
        .collect()
}

/// [`GemmProg`]'s source at its own extents ([`gemm_dims`]).
pub fn gemm_src(p: &GemmProg) -> (String, (u64, u64, u64)) {
    let (m, n, k) = gemm_dims(p);
    (gemm_src_at(p, m, n, k), (m, n, k))
}

/// sweep-runtime-2 `streamk-epi`: a [`GemmProg`] at `(m, n, k)` whose product
/// `y` has cell-wise readers — an epilogue group (S54) — chosen by `chain`:
/// 0 an affine map (`y·½ + 1`), 1 a per-column bias then a relu (a guard),
/// 2 a join with a second closed-form array (`y + d`), 3 two members (an
/// affine map and its square plus `y`), with `y` itself printed too (the
/// site's output stored beside the members'). Integer operands in `-2 ..= 2`
/// and dyadic constants: every fold order and every cut of the fold prints
/// the same bits. Five cells of the last value printed.
pub fn gemm_epi_src_at(p: &GemmProg, m: u64, n: u64, k: u64, chain: u8) -> String {
    let base = gemm_src_at(p, m, n, k);
    let body = base
        .split("    y[")
        .next()
        .expect("the product's source")
        .to_string();
    let mn = m * n;
    let mut s = body;
    let out = match chain % 4 {
        0 => {
            s += "    ic -> map {\n        t -> y[t] * 0.5 + 1.0\n    } -> z;\n";
            "z"
        }
        1 => {
            s += &format!(
                "    {n} -> iota -> jb;\n    jb -> map {{\n        t -> (t * 3 + 1) % 7 - 3 -> widen<f32>\n    }} -> bv;\n"
            );
            s += &format!(
                "    ic -> map {{\n        t ->\n        y[t] + bv[t % {n}] -> u;\n        (u > 0.0) -> {{\n            -true-> u;\n            -false-> 0.0;\n        }}\n    }} -> z;\n"
            );
            "z"
        }
        2 => {
            s += "    ic -> map {\n        t -> (t * 5 + 2) % 9 - 4 -> widen<f32>\n    } -> d;\n";
            s += "    ic -> map {\n        t -> y[t] + d[t]\n    } -> z;\n";
            "z"
        }
        _ => {
            s += "    ic -> map {\n        t -> y[t] * 0.5 + 1.0\n    } -> z1;\n";
            s += "    ic -> map {\n        t -> z1[t] * z1[t] + y[t]\n    } -> z;\n";
            "z"
        }
    };
    let last = mn - 1;
    for c in [0, last, (m - 1) * n, mn / 2 + n / 3, 2 * n - 1] {
        s += &format!("    {out}[{}] -> println;\n", c.min(last));
    }
    if chain % 4 == 3 {
        s += &format!(
            "    y[{}] -> println;\n    z1[{last}] -> println;\n",
            mn / 3
        );
    }
    s + "}\n"
}
