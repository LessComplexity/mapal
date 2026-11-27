//! Mapal-source generators: split f32 products, wgmma sites, and the formatting check.

use super::*;

// --- f32-tensor-cores: split realizations of `{f32}` products ------------------

/// A sum of products under the default face's `{f32}` (no member set) whose
/// operands a split form may realize on a matrix instruction
/// (`mapal_ir::split`): both `f32` (tf32x3), both one narrow format
/// (narrow1), or `f32 × bf16` / `bf16 × f32` (bf16x3n, bf16x2n). `k` straddles every
/// form's `K_min` (8 … 512); a residual epilogue; `b` row-major or read
/// transposed; value laws a split meets at its edges — small rationals,
/// magnitudes spread over 2^±40, cancelling pairs (`|Σ| ≪ M`), small integers
/// (TF32-exact: `lo = 0`), an infinity and a NaN (the guard), the largest
/// finite value (the `hi` overflow select), operands near the least normal
/// value (parts below it: the absolute term), infinities of both signs in
/// the narrow operand of a mixed product (`b`; `a` for `bf16 × f32`; `b` for
/// one format) on the checked cells' rows / columns, the other operand
/// finite (a split part of `0` or of the other sign times `±inf`).
#[derive(Clone, Debug)]
pub struct SplitProg {
    pub dims: [u8; 3],
    /// 0: f32 × f32, 1: bf16 × bf16, 2: f16 × f16, 3: f32 × bf16, 6: bf16
    /// × f32.
    pub store: u8,
    /// 0 rationals, 1 wide, 2 cancelling, 3 integers, 4 inf / NaN, 5 the
    /// largest finite, 6 tiny, 7 ±inf in the narrow operand.
    pub law: u8,
    pub bt: bool,
    pub post: bool,
    pub seed: [u8; 2],
}

pub fn split_prog_strategy() -> impl Strategy<Value = SplitProg> {
    (
        (any::<u8>(), any::<u8>(), any::<u8>()),
        prop::sample::select(vec![0u8, 0, 1, 1, 2, 3, 6]),
        0u8..8,
        prop::bool::weighted(0.3),
        prop::bool::weighted(0.3),
        (any::<u8>(), any::<u8>()),
    )
        .prop_map(|(d, store, law, bt, post, (s0, s1))| SplitProg {
            dims: [d.0, d.1, d.2],
            store,
            law,
            bt,
            post,
            seed: [s0, s1],
        })
}

/// [`SplitProg`]'s source and `(cells, m, k)`: each checked cell prints its
/// value, its `M = Σ|a·b|` and `Σ(|a| + |b|)` (the absolute term's operand
/// mass), in that order.
pub fn split_src(p: &SplitProg) -> (String, (usize, u64, u64)) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    // Narrow operands: half the draws at sites whose flops per operand byte
    // reach a part's FFMA balance (`m`, `n` 256, `k` 512: 85 flop / B), where
    // the rule (not only its ON arm) takes the split.
    let (m, n, k) = match p.store {
        1 | 2 => (
            pick(p.dims[0], &[32, 128, 256, 256]),
            pick(p.dims[1], &[16, 128, 256, 256]),
            pick(p.dims[2], &[16, 48, 128, 512, 512]),
        ),
        _ => (
            pick(p.dims[0], &[16, 32, 64, 128, 256]),
            pick(p.dims[1], &[8, 16, 32, 64, 128, 256]),
            pick(p.dims[2], &[8, 16, 24, 32, 48, 64, 96, 128, 256, 512]),
        ),
    };
    split_src_at(p, m, n, k)
}

/// [`split_src`] at the given extents.
pub fn split_src_at(p: &SplitProg, m: u64, n: u64, k: u64) -> (String, (usize, u64, u64)) {
    let (a1, a2) = (
        u64::from(p.seed[0] % 13) * 2 + 3,
        u64::from(p.seed[1] % 11) * 2 + 5,
    );
    let (fa, fb) = match p.store {
        1 => (Some("bf16"), Some("bf16")),
        2 => (Some("f16"), Some("f16")),
        3 => (None, Some("bf16")),
        6 => (Some("bf16"), None),
        4 => (Some("f8e4m3"), Some("f8e4m3")),
        5 => (Some("f8e5m2"), Some("f8e5m2")),
        _ => (None, None),
    };
    // The narrow formats' range: the wide / tiny / largest laws keep their
    // spread inside it (f16: 2^±14 of headroom).
    let narrow = fa.is_some() || fb.is_some();
    let big = if narrow { "64.0" } else { "1000000000000.0" };
    let small = if narrow { "64.0" } else { "1000000000.0" };
    let p0 = u64::from(p.seed[0]) * 37 % (m * k);
    let p1 = u64::from(p.seed[1]) * 41 % (k * n);
    // `a` along `k` (row-major), `b` along `k` at stride `n` (or 1 when `bt`).
    let law_a = match p.law {
        1 => format!(
            "(t * {a1} + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0 -> x; (t % 3 == 0) -> {{ -true-> x * {big}; -false-> x / {small}; }} -> r; r"
        ),
        2 => format!(
            "((t / 2) * {a1} + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0 -> x; (t % 2 == 0) -> {{ -true-> x; -false-> 0.0 - x; }} -> r; r"
        ),
        3 => format!("(t * {a1} + 13) % 101 - 50 -> widen<f32> -> v; v"),
        4 => format!(
            "(t * {a1} + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0 -> x; (t == {p0}) -> {{ -true-> 1.0 / 0.0; -false-> x; }} -> r; r"
        ),
        5 if !narrow => format!(
            "(t * {a1} + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0 -> x; (t == {p0}) -> {{ -true-> 340282346638528859811704183484516925440.0; -false-> x; }} -> r; r"
        ),
        6 if !narrow => format!(
            "(t * {a1} + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0 -> x; x / 1000000000000000000000000000000000000.0"
        ),
        _ => format!("(t * {a1} + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0"),
    };
    // Cancelling pairs: `b` equal on the two `k` of a pair (whose `a` cancel)
    // but for a relative 2⁻¹⁰ step.
    let pair = if p.bt {
        "t / 2".to_string()
    } else {
        format!("t / {} * {n} + t % {n}", 2 * n)
    };
    let odd = if p.bt {
        "t % 2".to_string()
    } else {
        format!("t / {n} % 2")
    };
    let law_b = match p.law {
        2 => format!(
            "(({pair}) * {a2} + 57) % 103 - 51 -> widen<f32> -> v; {odd} -> widen<f32> -> s; v / 9.0 + v / 9.0 * s / 1024.0"
        ),
        3 => format!("(t * {a2} + 57) % 103 - 51 -> widen<f32> -> v; v"),
        4 => format!(
            "(t * {a2} + 57) % 103 - 51 -> widen<f32> -> v; v / 9.0 -> x; (t == {p1}) -> {{ -true-> 0.0 / 0.0; -false-> x; }} -> r; r"
        ),
        5 if !narrow => format!(
            "(t * {a2} + 57) % 103 - 51 -> widen<f32> -> v; v / 9.0 / 1000000000000000000000000000000.0"
        ),
        6 if !narrow => format!(
            "(t * {a2} + 57) % 103 - 51 -> widen<f32> -> v; v / 9.0 * 100000000000000000000.0"
        ),
        _ => format!("(t * {a2} + 57) % 103 - 51 -> widen<f32> -> v; v / 9.0"),
    };
    let cells = [
        u64::from(p.seed[0]) * 7 % (m * n),
        m * n / 2 + 1,
        m * n - 1 - u64::from(p.seed[1]) % n,
    ];
    // Law 7: `+inf` read by the first checked cell at `k / 2`, `−inf` by the
    // last at `k = 1`, in the narrow operand (`a` only where `a` alone is
    // narrow) — rows of `a` (`i·k + q`), columns of `b` (`q·n + j`, or
    // `j·k + q` when `bt`).
    let (law_a, law_b) = match p.law {
        7 => {
            let infs = |law: &str, (i0, i1): (u64, u64)| {
                format!(
                    "{law} -> w; (t == {i0}) -> {{ -true-> 1.0 / 0.0; -false-> w; }} -> u1; (t == {i1}) -> {{ -true-> 0.0 - 1.0 / 0.0; -false-> u1; }} -> u2; u2"
                )
            };
            let (c0, c2) = (cells[0], cells[2]);
            if fa.is_some() && fb.is_none() {
                (
                    infs(&law_a, ((c0 / n) * k + k / 2, (c2 / n) * k + 1)),
                    law_b,
                )
            } else {
                let bix = |j: u64, q: u64| if p.bt { j * k + q } else { q * n + j };
                (law_a, infs(&law_b, (bix(c0 % n, k / 2), bix(c2 % n, 1))))
            }
        }
        _ => (law_a, law_b),
    };
    let store = |f: Option<&str>| f.map_or(String::new(), |n| format!(" -> to<{n}>"));
    let read = |f: Option<&str>, x: &str| match f {
        Some(_) => format!("{x} -> widen<f32>"),
        None => x.to_string(),
    };
    let bi = if p.bt {
        format!("b[j * {k} + q]")
    } else {
        format!("b[q * {n} + j]")
    };
    let (ar, br) = (read(fa, &format!("a[i * {k} + q]")), read(fb, &bi));
    let post = if p.post {
        format!(
            " -> u; {} -> r; u * 0.5 + r",
            read(fa, &format!("a[i * {k}]"))
        )
    } else {
        String::new()
    };
    let abs =
        |x: &str, o: &str| format!("({x} > 0.0) -> {{ -true-> {x}; -false-> 0.0 - {x}; }} -> {o}");
    // An integer literal, a negative one as a difference.
    let int = |v: i64| match v < 0 {
        true => format!("(0 - {})", -v),
        false => v.to_string(),
    };
    let mut src = format!(
        "fn main() {{
    {mk} -> iota -> ia;
    ia -> map {{ t -> {law_a} }} -> a0;
    a0 -> map {{ x -> x{sa} }} -> a;
    {kn} -> iota -> ib;
    ib -> map {{ t -> {law_b} }} -> b0;
    b0 -> map {{ x -> x{sb} }} -> b;
    {mn} -> iota -> ic;
    {k} -> iota -> kr;
    ic -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, q -> {ar} -> x; {br} -> y; acc + x * y }}{post} }} -> y;
    3 -> iota -> ck;
    ck -> map: f32 {{ s -> {c0} + s * {d1} + s * (s - 1) / 2 * {d2} -> t; t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, q -> {ar} -> x; {br} -> y; x * y -> pv; {pabs}; acc + av }} }} -> mag;
    ck -> map: f32 {{ s -> {c0} + s * {d1} + s * (s - 1) / 2 * {d2} -> t; t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, q -> {ar} -> x; {br} -> y; {xabs}; {yabs}; acc + ax + ay }} }} -> sab;
",
        mk = m * k,
        kn = k * n,
        mn = m * n,
        sa = store(fa),
        sb = store(fb),
        pabs = abs("pv", "av"),
        xabs = abs("x", "ax"),
        yabs = abs("y", "ay"),
        // The three checked cells as `c0 + s·d1 + s(s−1)/2·d2`, `s` = 0, 1, 2:
        // `M` and `Σ(|a| + |b|)` of those cells only (the interpreter's
        // budget at 256 × 256 × 512).
        c0 = cells[0],
        d1 = int(cells[1] as i64 - cells[0] as i64),
        d2 = int(cells[2] as i64 - 2 * cells[1] as i64 + cells[0] as i64),
    );
    for (s, c) in cells.iter().enumerate() {
        src += &format!(
            "    y[{c}] -> println;\n    mag[{s}] -> println;\n    sab[{s}] -> println;\n"
        );
    }
    src += "}\n";
    (src, (cells.len(), m, k))
}

// --- hopper-wgmma-tma: TF32 GEMMs at warpgroup-tile extents -----------------

/// A TF32 GEMM `y = a · b` at extents a warpgroup tile divides or misses:
/// `m` 8–320, `n` 64–384, `k` 8–640 (from one tile below a box to twenty
/// stages: the barrier phases wrap), `a` rows `ci` apart (`pad`: 0 dense, 4 a
/// granule-whole pad, 3 a misaligned one the box refuses), the operands at an
/// offset (`off`: whole granules or not), `b` read as `k × n` or transposed
/// (`bt`: `n × k`, no `c × k` box). Integer-valued operands over 7 / 9: the
/// TF32 rounding of each operand is exercised, the bound of D3 judges the sum.
#[derive(Clone, Debug)]
pub struct WgmmaProg {
    pub dims: [u8; 3],
    pub pad: u8,
    pub off: u8,
    pub bt: bool,
    pub law: [u8; 2],
}

pub fn wgmma_prog_strategy() -> impl Strategy<Value = WgmmaProg> {
    (
        prop::array::uniform3(any::<u8>()),
        prop::sample::select(vec![0u8, 0, 0, 4, 3]),
        prop::sample::select(vec![0u8, 0, 0, 4, 1]),
        prop::bool::weighted(0.15),
        prop::array::uniform2(any::<u8>()),
    )
        .prop_map(|(dims, pad, off, bt, law)| WgmmaProg {
            dims,
            pad,
            off,
            bt,
            law,
        })
}

/// [`WgmmaProg`]'s source (`main` on the TF32 face) and its `(m, n, k)`.
/// Prints two cells of `y`, each followed by its `M` (`Σ |a·b|`).
pub fn wgmma_src(p: &WgmmaProg) -> (String, (u64, u64, u64)) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let m = pick(p.dims[0], &[8, 16, 64, 128, 192, 256, 320]);
    let n = pick(p.dims[1], &[64, 128, 192, 256, 384]);
    let k = pick(p.dims[2], &[8, 32, 64, 96, 128, 256, 640]);
    let ci = k + u64::from(p.pad);
    let (oa, ob) = (u64::from(p.off), u64::from(p.off) * 2);
    let (a1, a2) = (
        u64::from(p.law[0] % 13) * 2 + 3,
        u64::from(p.law[1] % 11) * 2 + 5,
    );
    let bi = if p.bt {
        format!("b[{ob} + j * {k} + k]")
    } else {
        format!("b[{ob} + k * {n} + j]")
    };
    let ai = format!("a[{oa} + i * {ci} + k]");
    let (c0, c1) = (
        u64::from(p.law[0]) * 7 % (m * n),
        m * n - 1 - u64::from(p.law[1]) % n,
    );
    let src = format!(
        "fn main(): tf32 | f32 {{
    {na} -> iota -> ia;
    ia -> map {{ t -> (t * {a1} + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0 }} -> a;
    {nb} -> iota -> ib;
    ib -> map {{ t -> (t * {a2} + 57) % 103 - 51 -> widen<f32> -> v; v / 9.0 }} -> b;
    {mn} -> iota -> ic;
    {k} -> iota -> kr;
    ic -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, k -> acc + {ai} * {bi} }} }} -> y;
    ic -> map: f32 {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, k -> {ai} * {bi} -> p; (p > 0.0) -> {{ -true-> p; -false-> 0.0 - p; }} -> q; acc + q }} }} -> mag;
    y[{c0}] -> println;
    mag[{c0}] -> println;
    y[{c1}] -> println;
    mag[{c1}] -> println;
}}
",
        na = oa + m * ci,
        nb = ob + k * n,
        mn = m * n
    );
    (src, (m, n, k))
}

// --- wgmma-16bit: 16-bit GEMMs and epilogues at warpgroup-tile extents ------

/// A GEMM `y = a · b` under `: bf16 | f32` or `: f16 | f32` at extents a
/// warpgroup tile divides or misses: `m` 16–256, `n` 8–256, `k` 64–1024 (one
/// stage to sixteen: the ring turns over), at most 2²⁴ products. `fa` / `fb`: each
/// operand stored f32 (0), `bf16` (1) or `f16` (2) — under `: f16 | f32` an
/// f32 operand is refused (no value-range fact), half of those programs store
/// both `f16`. `once`: the magnitudes read their own operand copies, so an
/// f32 `a` read by the product alone is stored rounded by its producer
/// (`PrecisionPlan::store`: the descriptor home); else it keeps an f32
/// reader (the register home). `bt`: `b` read `n × k` (k-major). `post`: an
/// epilogue on the cell — 1 a residual `u·0.5 + r`, 2 a gate `u·σ(u)·r`
/// (SiLU × up, `σ` through `exp`), `r` a row-closed read of another array.
#[derive(Clone, Debug)]
pub struct Wgmma16Prog {
    pub dims: [u8; 3],
    pub fa: u8,
    pub fb: u8,
    pub f16: bool,
    pub once: bool,
    pub bt: bool,
    pub post: u8,
    pub law: [u8; 2],
}

pub fn wgmma16_prog_strategy() -> impl Strategy<Value = Wgmma16Prog> {
    (
        (any::<u8>(), any::<u8>(), any::<u8>()),
        prop::sample::select(vec![0u8, 0, 1, 1, 2]),
        prop::sample::select(vec![1u8, 1, 1, 1, 0]),
        prop::bool::weighted(0.25),
        any::<bool>(),
        prop::bool::weighted(0.3),
        prop::sample::select(vec![0u8, 0, 1, 2]),
        (any::<u8>(), any::<u8>()),
    )
        .prop_map(|(d, fa, fb, f16, once, bt, post, (l0, l1))| {
            let both = f16 && l0 % 2 == 0;
            Wgmma16Prog {
                dims: [d.0, d.1, d.2],
                fa: if both { 2 } else { fa.min(1 + u8::from(f16)) },
                fb: if both { 2 } else { fb },
                f16,
                once,
                bt,
                post,
                law: [l0, l1],
            }
        })
}

/// [`Wgmma16Prog`]'s source and `(pairs, k)`: three cells of `y`, each
/// beside its `M = Σ|a·b|` (scaled as the epilogue scales `u`: `0.5` for the
/// residual, `|σ(u)·r| ≤ |r|·…` — the gate's pairs print `M·|r|·1.1`, a bound
/// of `|σ(u) + u·σ'(u)|·|r|` over every `u`).
pub fn wgmma16_src(p: &Wgmma16Prog) -> (String, (usize, u64)) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    // A deep `k` (768, 1024) on a 64- or 128-square tile: a small tile's
    // stage deepens (`tma-digits`' price), a larger one's ring turns over.
    let k0 = pick(p.dims[2], &[64, 128, 256, 768, 1024]);
    let (m, n) = match k0 >= 768 {
        true => [(64, 64), (128, 128), (64, 128), (128, 64)][usize::from(p.dims[0]) % 4],
        false => (
            pick(p.dims[0], &[16, 64, 128, 192, 256]),
            pick(p.dims[1], &[8, 64, 128, 192, 256]),
        ),
    };
    // At most 2²⁴ products a fold of the interpreter's.
    let k = [k0, 256, 128, 64]
        .into_iter()
        .find(|&k| m * n * k <= 1 << 24)
        .unwrap_or(64);
    let set = if p.f16 { ": f16 | f32" } else { ": bf16 | f32" };
    let (a1, a2) = (
        u64::from(p.law[0] % 13) * 2 + 3,
        u64::from(p.law[1] % 11) * 2 + 5,
    );
    let store = |f: u8| half_fmt(f).map_or(String::new(), |n| format!(" -> to<{n}>"));
    let read = |f: u8, x: &str| match half_fmt(f) {
        Some(_) => format!("{x} -> widen<f32>"),
        None => x.to_string(),
    };
    let bi = |b: &str| match p.bt {
        true => format!("{b}[j * {k} + q]"),
        false => format!("{b}[q * {n} + j]"),
    };
    let (ar, br) = (read(p.fa, &format!("a[i * {k} + q]")), read(p.fb, &bi("b")));
    let post = match p.post {
        1 => " -> u; r[t] -> rv; u * 0.5 + rv".to_string(),
        2 => " -> u; r[t] -> rv; 0.0 - u -> exp -> d; u / (1.0 + d) * rv".to_string(),
        _ => String::new(),
    };
    let cells = [
        u64::from(p.law[0]) * 7 % (m * n),
        m * n / 2 + 1,
        m * n - 1 - u64::from(p.law[1]) % n,
    ];
    let mut src = format!(
        "fn main() {{
    {mk} -> iota -> ia;
    ia -> map {{ t -> (t * {a1} + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0{sa} }} -> a;
    {kn} -> iota -> ib;
    ib -> map {{ t -> (t * {a2} + 57) % 103 - 51 -> widen<f32> -> v; v / 9.0{sb} }} -> b;
    {mn} -> iota -> ic;
    ic -> map {{ t -> (t * 3 + 7) % 17 - 8 -> widen<f32> -> v; v / 8.0 }} -> r;
    {k} -> iota -> kr;
    ic -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold{set} {{ acc, q -> {ar} -> x; {br} -> y; acc + x * y }}{post} }} -> y;
",
        mk = m * k,
        kn = k * n,
        mn = m * n,
        sa = store(p.fa),
        sb = store(p.fb),
    );
    let (am, bm) = if p.once {
        src += &format!(
            "    ia -> map {{ t -> (t * {a1} + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0{sa} }} -> am;
    ib -> map {{ t -> (t * {a2} + 57) % 103 - 51 -> widen<f32> -> v; v / 9.0{sb} }} -> bm;
",
            sa = store(p.fa),
            sb = store(p.fb),
        );
        (
            read(p.fa, &format!("am[i * {k} + q]")),
            read(p.fb, &bi("bm")),
        )
    } else {
        (ar, br)
    };
    let scale = match p.post {
        1 => " * 0.5".to_string(),
        2 => " * 1.1 * ra".to_string(),
        _ => String::new(),
    };
    src += &format!(
        "    ic -> map: f32 {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, q -> {am} -> x; {bm} -> y; x * y -> pv; (pv > 0.0) -> {{ -true-> pv; -false-> 0.0 - pv; }} -> av; acc + av }} -> s; r[t] -> rr; (rr > 0.0) -> {{ -true-> rr; -false-> 0.0 - rr; }} -> ra; s{scale} }} -> mag;
"
    );
    for c in cells {
        src += &format!("    y[{c}] -> println;\n    mag[{c}] -> println;\n");
    }
    src += "}\n";
    (formatted(src), (cells.len(), k))
}

/// A generated program laid out by the formatter (CLAUDE.md: generated source
/// is formatted too — the text the tests lower and the hardware runs dump).
/// The layout changes no token, so the program and its IR are the same.
pub fn formatted(src: String) -> String {
    mapal_syntax::format_source(&src)
        .unwrap_or_else(|d| panic!("generated program does not parse: {d:?}\n{src}"))
}

/// `mapal fmt --check` on a program a test lowers: the formatter leaves it
/// unchanged and no line is compacted past 120 columns.
pub fn assert_formatted(src: &str, what: &str) {
    let out = mapal_syntax::format_source(src)
        .unwrap_or_else(|d| panic!("{what} does not parse: {d:?}\n{src}"));
    assert_eq!(out, src, "{what} needs formatting");
    let long = src.lines().filter(|l| l.chars().count() > 120).count();
    assert_eq!(long, 0, "{what}: {long} lines past 120 columns\n{src}");
}
