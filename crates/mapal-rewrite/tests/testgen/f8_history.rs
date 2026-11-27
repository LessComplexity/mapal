//! Mapal-source generators: fp8 storage, history programs and GEMV sites.

use super::*;

// --- fp8 storage (ADR-0043 slice 3) -------------------------------------------

/// A sum of products over fp8 storage: `b` (`k × n`, or `n × k` read
/// transposed under `bt`) rounded once from an `f32` law into `f8e4m3` /
/// `f8e5m2` (`fmt` 0 / 1), at a range that stays normal, saturates past the
/// format's largest finite or reaches its subnormals (`range` 0 / 1 / 2);
/// `a` `f32` or the same fp8 format (`fa`); the fold under `{f32}`,
/// `: bf16 | f32` or `: f16 | f32` (`set` 0 / 1 / 2). The scale is
/// dataflow (`scale`): 0 none; 1 per tensor — the weight's `amax` by two
/// whole max folds, `b = w / (amax / max) -> to_f8`, the epilogue `u · s`;
/// 2 per column — `sc[j]` in the epilogue; 3 a factor inside the fold
/// (`acc + x·y·sf`, constant along `k`: `DepLaw::Factor`). Then cells
/// rounding into the format (every seventh `exp(1000) = inf`: saturates), a
/// loop carrying an fp8 array (`looped`) and a whole fold over widened
/// values.
#[derive(Clone, Debug)]
pub struct F8Prog {
    pub dims: [u8; 3],
    pub fmt: u8,
    pub fa: bool,
    pub set: u8,
    pub bt: bool,
    pub scale: u8,
    pub range: u8,
    pub looped: bool,
    pub law: [u8; 2],
}

pub fn f8_prog_strategy() -> impl Strategy<Value = F8Prog> {
    (
        (any::<u8>(), any::<u8>(), any::<u8>()),
        prop::sample::select(vec![0u8, 0, 1]),
        prop::bool::weighted(0.4),
        prop::sample::select(vec![0u8, 0, 1, 2]),
        prop::bool::weighted(0.4),
        prop::sample::select(vec![0u8, 1, 1, 2, 3]),
        prop::sample::select(vec![0u8, 0, 0, 1, 2]),
        prop::bool::weighted(0.4),
        (any::<u8>(), any::<u8>()),
    )
        .prop_map(
            |(d, fmt, fa, set, bt, scale, range, looped, (l0, l1))| F8Prog {
                dims: [d.0, d.1, d.2],
                fmt,
                fa,
                set,
                bt,
                scale,
                range,
                looped,
                law: [l0, l1],
            },
        )
}

/// The surface name of an [`F8Prog`]'s format.
pub fn f8_fmt(p: &F8Prog) -> &'static str {
    if p.fmt == 1 { "f8e5m2" } else { "f8e4m3" }
}

/// [`F8Prog`]'s source and its checks `(cells, k, m·n)`: the first `cells` pairs
/// of printed lines are a value and its `M = Σ|x·y·sf|` scaled as the value
/// (the epilogue's positive scale) over a `k`-step fold; the later lines
/// pass through a rounding into the format.
pub fn f8_src(p: &F8Prog) -> (String, (usize, u64, u64)) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let fmt = f8_fmt(p);
    let max = if p.fmt == 1 { "57344.0" } else { "448.0" };
    let set = match p.set {
        1 => ": bf16 | f32",
        2 => ": f16 | f32",
        _ => "",
    };
    let m = pick(p.dims[0], &[1, 4, 16, 32, 64]);
    let n = pick(p.dims[1], &[16, 32, 64, 128]);
    let k = pick(p.dims[2], &[16, 32, 64, 128, 256]);
    let (a1, a2) = (
        u64::from(p.law[0] % 13) * 2 + 3,
        u64::from(p.law[1] % 11) * 2 + 5,
    );
    let rng = match p.range {
        1 => "* 1000.0",
        2 if p.fmt == 1 => "* 0.000001",
        2 => "* 0.0001",
        _ => "/ 9.0",
    };
    let (mk, kn, mn) = (m * k, k * n, m * n);
    let mut s = format!(
        "fn main() {{\n    {mk} -> iota -> ia;\n    \
         ia -> map {{ t -> (t * {a1} + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0 }} -> af;\n    \
         {kn} -> iota -> ib;\n    \
         ib -> map {{ t -> (t * {a2} + 57) % 103 - 51 -> widen<f32> -> v; v {rng} }} -> bf;\n    \
         {n} -> iota -> inn;\n    {mn} -> iota -> ic;\n    {k} -> iota -> kr;\n"
    );
    // The scale, as dataflow.
    let (post, mpost, factor) = match p.scale {
        1 => {
            s += "    (0.0, ib) -> fold { mx, t -> bf[t] -> x; (x > mx) -> { -true-> x; -false-> mx; } } -> hi;\n    \
                  (0.0, ib) -> fold { mx, t -> 0.0 - bf[t] -> x; (x > mx) -> { -true-> x; -false-> mx; } } -> lo;\n    \
                  (hi > lo) -> { -true-> hi; -false-> lo; } -> amax;\n";
            s += &format!(
                "    amax / {max} -> s;\n    ib -> map {{ t -> bf[t] / s -> to<{fmt}> }} -> b;\n"
            );
            (" -> u; u * s".to_string(), " -> u; u * s".to_string(), "")
        }
        2 => {
            s += "    inn -> map { j -> j % 5 + 1 -> widen<f32> -> v; v * 0.25 } -> sc;\n";
            s += &format!("    ib -> map {{ t -> bf[t] -> to<{fmt}> }} -> b;\n");
            (
                " -> u; u * sc[j]".to_string(),
                " -> u; u * sc[j]".to_string(),
                "",
            )
        }
        _ => {
            s += &format!("    ib -> map {{ t -> bf[t] -> to<{fmt}> }} -> b;\n");
            (
                String::new(),
                String::new(),
                if p.scale == 3 { " * sf" } else { "" },
            )
        }
    };
    if p.scale == 3 {
        s += "    kr[3] -> widen<f32> -> f3;\n    f3 * 0.125 -> sf;\n";
    }
    let ar = if p.fa {
        s += &format!("    ia -> map {{ t -> af[t] -> to<{fmt}> }} -> a;\n");
        format!("a[i * {k} + q] -> widen<f32>")
    } else {
        format!("af[i * {k} + q]")
    };
    let bi = if p.bt {
        format!("b[j * {k} + q]")
    } else {
        format!("b[q * {n} + j]")
    };
    let abs = "(pv > 0.0) -> { -true-> pv; -false-> 0.0 - pv; } -> av; acc + av";
    s += &format!(
        "    ic -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold{set} {{ acc, q -> {ar} -> x; {bi} -> widen<f32> -> y; acc + x * y{factor} }}{post} }} -> y;\n    \
         ic -> map: f32 {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, q -> {ar} -> x; {bi} -> widen<f32> -> y; x * y{factor} -> pv; {abs} }}{mpost} }} -> mag;\n"
    );
    let cells = [
        u64::from(p.law[0]) * 7 % mn,
        mn / 2,
        mn - 1 - u64::from(p.law[1]) % n.min(mn),
    ];
    for c in cells {
        s += &format!("    y[{c}] -> println;\n    mag[{c}] -> println;\n");
    }
    // Cells rounding into the format: every seventh `exp(1000) = +inf`,
    // which saturates to the format's largest finite.
    s += &format!(
        "    ic -> map {{ e -> (e % 7 == 0) -> {{ -true-> 1000.0 -> exp -> to<{fmt}>; -false-> y[e] -> to<{fmt}>; }} }} -> z;\n"
    );
    let fin = if p.looped {
        s += &format!(
            "    mut st: [{fmt}; {mn}] <- z;\n    mut it: i32 <- 0;\n    loop {{\n        (it < 3) -> {{\n            -true-> {{\n                \
             ic -> map {{ e -> st[e] -> widen<f32> -> v; v * 1.5 + 0.25 -> to<{fmt}> }} -> st;\n                \
             it * 5 % {mn} -> c;\n                st[c] -> widen<f32> -> sv;\n                sv + 1.0 -> to<{fmt}> -> sn;\n                \
             st[c] <- sn;\n                it + 1 -> it;\n                -> loop;\n            }}\n            -false-> st -> fin;\n        }}\n    }}\n"
        );
        "fin"
    } else {
        "z"
    };
    for c in [0, 1, mn / 2, mn - 1] {
        s += &format!("    {fin}[{c}] -> widen<f32> -> println;\n");
    }
    s += &format!(
        "    0.0 -> zero: f32;\n    (zero, {fin}) -> fold {{ acc, v -> v -> widen<f32> -> x; acc + x }} -> sum;\n    sum -> println;\n}}\n"
    );
    (s, (cells.len(), k, mn))
}

/// caching-memory: a growing-history loop — the loop carries a history of `b`
/// sequences × `l` rows × `d` cells and writes the step's input row into it
/// at position `p0 + t` (one row of every sequence; past the end: no write),
/// then recomputes, for every row, a chain of row-local transforms (a row
/// fold of squares and a per-row scale; a capture-free `·2` map; a
/// projection through `n` columns; a rotation reading each row's partner
/// half — a window), and reads every cell of the result for the next input
/// (so a cell a delta missed shows). `wc` is an input the loop carries
/// without ever writing it (a program keeps it between trips): its
/// transpose, recomputed every trip, is the projection's weights. Small
/// integers throughout: every face and fold order agrees with the
/// interpreter.
#[derive(Clone, Debug)]
pub struct HistoryProg {
    pub b: u64,
    pub l: u64,
    pub d: u64,
    pub n: u64,
    pub trips: i64,
    pub p0: i64,
    /// One sequence written as `c ÷ d == p` (a row range, `Rows`) instead of
    /// `(c ÷ d) % l == p` (pieces).
    pub rows_form: bool,
    pub rowfold: bool,
    pub same: bool,
    pub rot: bool,
    /// The projection reads the carried constant's transpose.
    pub constant: bool,
    /// The row is written at the trip's end: the transforms read the history
    /// as the trip found it (its previous trip's row is the new one).
    pub late: bool,
    /// The history's initial value is built before the loop through a
    /// transform that dies there (a prebuilt cache's inputs): the loop's own
    /// buffers may take its bytes (`cone-span`).
    pub prebuilt: bool,
    /// Where the loop runs: 0 in `main`; 1 in a callee `step` (its exit
    /// value the result: `-> ret`) `main` calls once; 2 that callee called
    /// from `main`'s two-trip loop, each result the next call's `x0`. A
    /// callee's device loop leaves its result on the host for the caller.
    pub callee: u8,
    /// A callee's inputs besides `x0`: bit 0 `w`, bit 1 `g`, bit 2 `h0` a
    /// parameter, else built in the callee.
    pub params: u8,
}

pub fn history_strategy() -> impl Strategy<Value = HistoryProg> {
    (
        1u64..4,
        prop::sample::select(vec![4u64, 8, 16]),
        prop::sample::select(vec![4u64, 8, 16]),
        prop::sample::select(vec![4u64, 8, 16]),
        1i64..6,
        0i64..17,
        any::<bool>(),
        (
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
        ),
        (0u8..3, 0u8..8),
    )
        .prop_map(
            |(
                b,
                l,
                d,
                n,
                trips,
                p0,
                rows_form,
                (rowfold, same, rot, constant, late, prebuilt),
                (callee, params),
            )| {
                HistoryProg {
                    b,
                    l,
                    d,
                    n,
                    trips,
                    p0: p0 % l as i64,
                    rows_form: rows_form && b == 1,
                    rowfold,
                    same,
                    rot,
                    constant,
                    late,
                    prebuilt,
                    callee,
                    params,
                }
            },
        )
}

/// [`HistoryProg`]'s source text.
pub fn history_prog_src(p: &HistoryProg) -> String {
    let (b, l, d, n) = (p.b, p.l, p.d, p.n);
    let data = |name: &str, len: u64, a: u64, c: u64| {
        format!(
            "    {len} -> iota -> i_{name};\n    i_{name} -> map {{ t -> (t * {a} + {c}) % 5 - 2 -> widen<f32> }} -> {name};\n"
        )
    };
    // The inputs: built in `main` (a callee's parameters) or in the callee.
    let inputs = [
        ("w", d * n, 3, 1, p.params & 1 != 0),
        ("g", d, 2, 3, p.params & 2 != 0),
        ("x0", b * d, 7, 2, true),
        ("h0", b * l * d, 3, 4, p.params & 4 != 0),
    ];
    let (mut head, mut s) = (String::new(), String::new());
    let mut sig = Vec::new();
    for (name, len, a, c, param) in inputs {
        if p.callee == 0 || param {
            head += &data(name, len, a, c);
            sig.push((name, len));
        } else {
            s += &data(name, len, a, c);
        }
    }
    s += &format!(
        "    {} -> iota -> rBL;\n    {d} -> iota -> rD;\n    {} -> iota -> rHD;\n    {} -> iota -> rBLN;\n    \
         {} -> iota -> rLN;\n    {} -> iota -> rBD;\n    {} -> iota -> rDN;\n",
        b * l,
        b * l * d,
        b * l * n,
        l * n,
        b * d,
        d * n
    );
    // A prebuilt history: a scaled copy (dead once the next map has read
    // it, neighbour rows included), then the carried init.
    let init = if p.prebuilt {
        let hd = b * l * d;
        // Small row values around it: one dies before the scaled copy is
        // made, one lives past it (the arena's order of placement matters).
        s += &format!(
            "    rBL -> map {{ r -> h0[r * {d}] * 0.5 }} -> r1;\n    \
             rBL -> map {{ r -> h0[r * {d} + {d} - 1] + 1.0 }} -> r0;\n    \
             rBL -> map {{ r -> r0[r] * r0[(r + 1) % {bl}] }} -> m0;\n    \
             rHD -> map {{ e -> h0[e] * g[e % {d}] + m0[e / {d}] }} -> hs;\n    \
             rHD -> map {{ e -> hs[e] + hs[(e + {d}) % {hd}] + r1[e / {d}] }} -> hb;\n",
            bl = b * l
        );
        "hb"
    } else {
        "h0"
    };
    s += &format!(
        "    mut hx: [f32; {}] <- {init};\n    mut x: [f32; {}] <- x0;\n",
        b * l * d,
        b * d
    );
    if p.constant {
        s += &format!("    mut wc: [f32; {}] <- w;\n", d * n);
    }
    s += &format!(
        "    mut t: i32 <- 0;\n    loop {{\n        (t < {}) -> {{\n            -true-> {{\n                {} + t -> p;\n",
        p.trips, p.p0
    );
    let cond = if p.rows_form {
        format!("c / {d} == p")
    } else {
        format!("c / {d} % {l} == p")
    };
    let write = format!(
        "rHD -> map {{ c -> ({cond}) -> {{ -true-> x[c / {} * {d} + c % {d}]; -false-> hx[c]; }} }} -> hx;\n",
        l * d
    );
    // A late write reads the step's input before the consumer rebinds it.
    let mut body = if p.late {
        "x -> map { v -> v } -> xin;\n".to_string()
    } else {
        write.clone()
    };
    if p.rowfold {
        body += &format!(
            "rBL -> map {{ r -> (0.0, rD) -> fold {{ acc, i -> hx[r * {d} + i] -> v; acc + v * v }} }} -> rs;\n\
             rHD -> map {{ e -> hx[e] * g[e % {d}] + rs[e / {d}] }} -> xn;\n"
        );
    } else {
        body += &format!("rHD -> map {{ e -> hx[e] * g[e % {d}] }} -> xn;\n");
    }
    let src = if p.same {
        body += "xn -> map { v -> v * 2.0 } -> x2;\n";
        "x2"
    } else {
        "xn"
    };
    let wread = if p.constant {
        body += &format!("rDN -> map {{ e -> wc[e % {d} * {n} + e / {d}] }} -> wt;\n");
        format!("wt[j * {d} + k]")
    } else {
        format!("w[k * {n} + j]")
    };
    body += &format!(
        "rBLN -> map {{ e -> e / {n} -> r; e % {n} -> j; (0.0, rD) -> fold {{ acc, k -> acc + {src}[r * {d} + k] * {wread} }} }} -> pr;\n"
    );
    let res = if p.rot {
        let h = n / 2;
        body += &format!(
            "rBLN -> map {{ e -> e / {n} -> r; e % {n} -> u; (u < {h}) -> {{ -true-> pr[r * {n} + u] * 2.0 - pr[r * {n} + u + {h}]; \
             -false-> pr[r * {n} + u] + pr[r * {n} + u - {h}]; }} }} -> ro;\n"
        );
        "ro"
    } else {
        "pr"
    };
    body += &format!(
        "rBD -> map {{ e -> e / {d} -> bb; e % {d} -> dd; (0.0, rLN) -> fold {{ acc, i -> i % 3 - 1 -> widen<f32> -> cw; \
         acc + {res}[bb * {} + i] * cw }} -> s; dd -> widen<f32> -> df; (s + df > 0.0) -> {{ -true-> g[dd]; -false-> 0.0 - g[dd]; }} }} -> x;\n",
        l * n
    );
    if p.late {
        body += &write.replace("-true-> x[", "-true-> xin[");
    }
    // The carried constant goes round unchanged (a step that hands its
    // weights on, as a step function returning its whole state does).
    if p.constant {
        body += "wc -> wc;\n";
    }
    for line in body.lines() {
        s += &format!("                {line}\n");
    }
    let exit = if p.callee == 0 { "out" } else { "ret" };
    s += &format!(
        "                t + 1 -> t;\n                -> loop;\n            }}\n            -false-> x -> {exit};\n        }}\n    }}\n"
    );
    let mut prints = String::new();
    for c in 0..b * d {
        prints += &format!("    out[{c}] -> println;\n");
    }
    prints += "    0.0 -> zero: f32;\n    (zero, out) -> fold { acc, v -> acc + v } -> sum;\n    sum -> println;\n";
    let params: Vec<String> = sig
        .iter()
        .map(|(x, len)| format!("{x}: [f32; {len}]"))
        .collect();
    let step = format!(
        "fn step({}) -> [f32; {}] {{\n{s}}}\n\n",
        params.join(", "),
        b * d
    );
    // The call's argument: the one parameter, or their product.
    let args = |x0: &str| {
        let names: Vec<&str> = sig
            .iter()
            .map(|&(x, _)| if x == "x0" { x0 } else { x })
            .collect();
        match names.len() {
            1 => names[0].to_string(),
            _ => format!("({})", names.join(", ")),
        }
    };
    match p.callee {
        0 => format!("fn main() {{\n{head}{s}{prints}}}\n"),
        1 => format!(
            "{step}fn main() {{\n{head}    {} -> step -> out;\n{prints}}}\n",
            args("x0")
        ),
        _ => format!(
            "{step}fn main() {{\n{head}    mut xo: [f32; {}] <- x0;\n    mut u: i32 <- 0;\n    loop {{\n        (u < 2) -> {{\n            -true-> {{\n                {} -> step -> xr;\n                xr -> xo;\n                u + 1 -> u;\n                -> loop;\n            }}\n            -false-> xo -> out;\n        }}\n    }}\n{prints}}}\n",
            b * d,
            args("xo")
        ),
    }
}

// --- gemv-stream: matrix-vector sites (`vec-cells`, `stream-fly`) ------------

/// A streaming sum of products: `rows` (1–3) rows of an activation `a` against
/// a weight `w` of `cols` lanes — stored `f32`, `bf16` or `f16` (`fmt` 0 / 1 /
/// 2) — read `k × cols` (`layout` 0), `cols × k` (1: strided lanes), `k ×
/// cols` from an offset `off` elements in (2: the 16-byte alignment the
/// vector load needs holds or not), or one `k × cols` matrix per row (3: `w`
/// moves with the row). `shared`: every row reads one `a` row (`a`'s row
/// stride 0). `post`: an epilogue `u·0.5 + a[i·k]`. Operands non-negative
/// (no cancellation: a reordered sum stays within a relative `k · 2⁻²³`).
#[derive(Clone, Debug)]
pub struct GemvProg {
    pub rows: u64,
    pub cols: u64,
    pub k: u64,
    pub fmt: u8,
    pub layout: u8,
    pub off: u64,
    pub shared: bool,
    pub post: bool,
    pub mul: u64,
}

pub fn gemv_prog_strategy() -> impl Strategy<Value = GemvProg> {
    (
        prop::sample::select(vec![1u64, 1, 2, 3, 3]),
        prop::sample::select(vec![64u64, 72, 96, 100, 128, 256]),
        prop::sample::select(vec![8u64, 24, 64, 96, 128, 256, 512, 1024]),
        prop::sample::select(vec![0u8, 0, 1, 2]),
        prop::sample::select(vec![0u8, 0, 1, 2, 2, 3, 3]),
        prop::sample::select(vec![2u64, 4, 8, 16]),
        prop::bool::weighted(0.3),
        any::<bool>(),
        prop::sample::select(vec![3u64, 7, 11, 13]),
    )
        .prop_map(
            |(rows, cols, k, fmt, layout, off, shared, post, mul)| GemvProg {
                rows,
                cols,
                k,
                fmt,
                layout,
                off,
                shared,
                post,
                mul,
            },
        )
}

/// [`GemvProg`]'s source: the product as a map over `rows × cols` cells
/// folding `k`, every cell's value printed in a fold-free order (first, last,
/// a middle one, the second row's first), then the cells' sum.
pub fn gemv_src(p: &GemvProg) -> String {
    let (rows, cols, k) = (p.rows, p.cols, p.k);
    let n = rows * cols;
    let off = if p.layout == 2 { p.off } else { 0 };
    let wn = match p.layout {
        3 => rows * k * cols,
        _ => k * cols + off,
    };
    let fmt = ["", "bf16", "f16"][usize::from(p.fmt)];
    let wlaw = format!("(e * {} + 5) % 9 + 1 -> widen<f32> -> v; v / 8.0", p.mul);
    let wmap = match p.fmt {
        0 => wlaw,
        _ => format!("{wlaw} -> to<{fmt}>"),
    };
    let ar = if p.shared { k } else { rows * k };
    let mut s = format!(
        "fn main() {{\n    {n} -> iota -> t;\n    {k} -> iota -> kk;\n    {ar} -> iota -> ta;\n    \
         {wn} -> iota -> tw;\n    \
         ta -> map {{ e -> (e * 7 + 3) % 5 -> widen<f32> -> v; v / 4.0 }} -> a;\n    \
         tw -> map {{ e -> {wmap} }} -> w;\n"
    );
    let wi = match p.layout {
        1 => format!("w[j * {k} + q]"),
        2 => format!("w[q * {cols} + j + {off}]"),
        3 => format!("w[i * {kc} + q * {cols} + j]", kc = k * cols),
        _ => format!("w[q * {cols} + j]"),
    };
    let wv = match p.fmt {
        0 => wi,
        _ => format!("{wi} -> widen<f32>"),
    };
    let ai = if p.shared {
        "a[q]".to_string()
    } else {
        format!("a[i * {k} + q]")
    };
    let tail = if p.post {
        format!(
            " -> u; u * 0.5 + a[{}]",
            if p.shared {
                "0".into()
            } else {
                format!("i * {k}")
            }
        )
    } else {
        String::new()
    };
    s += &format!(
        "    t -> map {{ e -> e / {cols} -> i; e % {cols} -> j; (0.0, kk) -> fold {{ acc, q -> {wv} -> b; acc + {ai} * b }}{tail} }} -> y;\n"
    );
    for c in [0, n - 1, n / 2 + 1, cols.min(n - 1)] {
        s += &format!("    y[{c}] -> println;\n");
    }
    s += "    0.0 -> zero: f32;\n    (zero, y) -> fold { acc, v -> acc + v } -> sum;\n    sum -> println;\n}\n";
    s
}
