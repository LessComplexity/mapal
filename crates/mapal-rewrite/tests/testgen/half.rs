//! Mapal-source generators: narrow float storage and tensor-core products.

use super::*;

// --- half-precision (ADR-0043): narrow float storage --------------------------

/// A program over `bf16` / `f16` storage: an `f32` activation `a`
/// (`rows × k`), a narrow weight `w` (`k × cols`, or `cols × k` read
/// transposed — strided lanes) rounded once from an `f32` law at a scale that
/// reaches the format's normal range, its overflow (`f16` past 65504, `bf16`
/// past `f32`'s max) and its subnormals; a sum of products reading `w`'s
/// exact widening (a [`mapal_ir::CategoryIr::tile_plan_narrow`] site — with an
/// epilogue under `post`, both operands narrow under `both`); then cells that
/// round the products into the format, a guard selecting narrow values, a
/// loop carrying a narrow array through a map and an element update, and a
/// whole fold over widened narrow values.
#[derive(Clone, Debug)]
pub struct NarrowProg {
    pub rows: u64,
    pub cols: u64,
    pub k: u64,
    /// `true`: `f16`; else `bf16`.
    pub f16: bool,
    /// The weight law's scale: 0 a model weight (`1/1856`), 1 `1/8`, 2 past
    /// the format's overflow, 3 into its subnormals.
    pub scale: u8,
    pub mul: u64,
    pub add: u64,
    pub both: bool,
    pub strided: bool,
    pub post: bool,
    pub guard: bool,
    pub looped: bool,
}

pub fn narrow_prog_strategy() -> impl Strategy<Value = NarrowProg> {
    (
        (
            prop::sample::select(vec![1u64, 2, 4, 8, 32]),
            prop::sample::select(vec![64u64, 96, 256]),
            prop::sample::select(vec![8u64, 16, 64]),
            any::<bool>(),
            prop::sample::select(vec![0u8, 0, 1, 2, 3]),
            prop::sample::select(vec![3u64, 7, 11, 13]),
            0u64..32,
        ),
        (
            any::<bool>(),
            prop::bool::weighted(0.3),
            any::<bool>(),
            any::<bool>(),
            any::<bool>(),
        ),
    )
        .prop_map(
            |((rows, cols, k, f16, scale, mul, add), (both, strided, post, guard, looped))| {
                NarrowProg {
                    rows,
                    cols,
                    k,
                    f16,
                    scale,
                    mul,
                    add,
                    both,
                    strided,
                    post,
                    guard,
                    looped,
                }
            },
        )
}

/// [`NarrowProg`]'s source and the count of its leading printed lines that are
/// `f32` sums of exact operands (ADR-0040 D2's bound on the default face); the
/// later lines pass through a rounding into the format (ADR-0043 D3: a flip
/// across a tie per rounding).
pub fn narrow_src(p: &NarrowProg) -> (String, usize) {
    let fmt = if p.f16 { "f16" } else { "bf16" };
    let (rows, cols, k) = (p.rows, p.cols, p.k);
    let n = rows * cols;
    let scale = match (p.scale, p.f16) {
        (0, _) => "/ 1856.0".to_string(),
        (1, _) => "/ 8.0".to_string(),
        (2, true) => "* 2000.0".to_string(),
        (2, false) => "* 10000000000000000000000000000000000000.0".to_string(),
        (_, true) => "* 0.0000001".to_string(),
        (_, false) => "* 0.000000000000000000000000000000000000001".to_string(),
    };
    let mut s = format!(
        "fn main() {{\n    {n} -> iota -> t;\n    {k} -> iota -> kk;\n    {rk} -> iota -> ta;\n    \
         {kc} -> iota -> tw;\n    \
         ta -> map {{ e -> (e * {mul} + {add}) % 31 - 15 -> widen<f32> -> v; v / 4.0 }} -> a;\n    \
         tw -> map {{ e -> (e * 7 + 13) % 103 - 51 -> widen<f32> -> v; v {scale} -> to<{fmt}> }} -> w;\n",
        rk = rows * k,
        kc = k * cols,
        mul = p.mul,
        add = p.add,
    );
    // `a` itself narrow: both operands widened.
    let (abind, aval) = if p.both {
        s += &format!("    ta -> map {{ e -> a[e] -> to<{fmt}> }} -> an;\n");
        (
            "an[i * K + q] -> widen<f32> -> x; ".replace('K', &k.to_string()),
            "x".to_string(),
        )
    } else {
        (String::new(), format!("a[i * {k} + q]"))
    };
    let wi = if p.strided {
        format!("w[j * {k} + q]")
    } else {
        format!("w[q * {cols} + j]")
    };
    let tail = if p.post {
        format!(" -> u; u * 0.5 + a[i * {k}]")
    } else {
        String::new()
    };
    s += &format!(
        "    t -> map {{ e -> e / {cols} -> i; e % {cols} -> j; (0.0, kk) -> fold {{ acc, q -> {abind}{wi} -> widen<f32> -> b; acc + {aval} * b }}{tail} }} -> y;\n"
    );
    let mut tight = 0;
    for c in [0, n / 3, n - 1] {
        s += &format!("    y[{c}] -> println;\n");
        tight += 1;
    }
    // Cells rounding into the format; a guard selecting narrow values.
    if p.guard {
        s += &format!(
            "    t -> map {{ e -> (e % 3 == 0) -> {{ -true-> y[e] -> to<{fmt}>; -false-> w[e % {kc}]; }} }} -> z;\n",
            kc = k * cols
        );
    } else {
        s += &format!("    t -> map {{ e -> y[e] -> to<{fmt}> }} -> z;\n");
    }
    // A loop carrying a narrow array: a map over it and an element update.
    let fin = if p.looped {
        s += &format!(
            "    mut st: [{fmt}; {n}] <- z;\n    mut it: i32 <- 0;\n    loop {{\n        (it < 3) -> {{\n            -true-> {{\n                \
             t -> map {{ e -> st[e] -> widen<f32> -> v; v * 1.5 + 0.25 -> to<{fmt}> }} -> st;\n                \
             it * 5 % {n} -> c;\n                st[c] -> widen<f32> -> sv;\n                sv + 1.0 -> to<{fmt}> -> sn;\n                \
             st[c] <- sn;\n                it + 1 -> it;\n                -> loop;\n            }}\n            -false-> st -> fin;\n        }}\n    }}\n"
        );
        "fin"
    } else {
        "z"
    };
    for c in [0, 1, n / 2 + 1, n - 1] {
        s += &format!("    {fin}[{c}] -> widen<f32> -> println;\n");
    }
    s += &format!(
        "    0.0 -> zero: f32;\n    (zero, {fin}) -> fold {{ acc, v -> v -> widen<f32> -> x; acc + x }} -> sum;\n    sum -> println;\n}}\n"
    );
    (s, tight)
}

// --- tensor-cores (ADR-0043 D4 / D6): 16-bit products ------------------------

/// A sum of products whose multiplicands are 16-bit: operands stored `f32`,
/// `bf16` or `f16` (`fa`, `fb`: 0 / 1 / 2), the fold under `{f32}`,
/// `: bf16 | f32` or `: f16 | f32` (`set` 0 / 1 / 2). `attn`: an attention
/// head instead of a GEMM — scores `q·kᵀ` (`b` read `n × k`, scaled: a tile
/// site with an epilogue), a row softmax (`exp`), then `p·v` — both products
/// under the set, `k` / `v` the stored cache; `bt` there: two heads (the
/// sites read the head digit — `tile_plan_radix`). `bt` (GEMM): `b` read
/// `n × k` (`b[j·k + q]`); `post` (GEMM): an epilogue `u·0.5 + a`.
#[derive(Clone, Debug)]
pub struct HalfProg {
    pub dims: [u8; 3],
    pub fa: u8,
    pub fb: u8,
    pub set: u8,
    pub attn: bool,
    pub bt: bool,
    pub post: bool,
    pub law: [u8; 2],
}

pub fn half_prog_strategy() -> impl Strategy<Value = HalfProg> {
    (
        (any::<u8>(), any::<u8>(), any::<u8>()),
        prop::sample::select(vec![0u8, 1, 1, 2]),
        prop::sample::select(vec![0u8, 1, 1, 2]),
        prop::sample::select(vec![0u8, 1, 1, 1, 2]),
        prop::bool::weighted(0.3),
        prop::bool::weighted(0.3),
        any::<bool>(),
        (any::<u8>(), any::<u8>()),
    )
        .prop_map(|(d, fa, fb, set, attn, bt, post, (l0, l1))| HalfProg {
            dims: [d.0, d.1, d.2],
            // `: f16 | f32` rounds no f32 operand (no value-range fact, D4):
            // half its programs store both operands `f16`.
            fa: if set == 2 && l0 % 2 == 0 { 2 } else { fa },
            fb: if set == 2 && l0 % 2 == 0 { 2 } else { fb },
            set,
            attn,
            bt,
            post,
            law: [l0, l1],
        })
}

/// The format a [`HalfProg`] operand is stored in (`None`: f32).
pub fn half_fmt(f: u8) -> Option<&'static str> {
    match f {
        1 => Some("bf16"),
        2 => Some("f16"),
        _ => None,
    }
}

/// [`HalfProg`]'s source and the checks its printed lines take: `(cells,
/// k, k_out)` — each checked value is printed beside its `M = Σ|a·b|`
/// (value, magnitude pairs) for the first `cells` pairs over a `k`-step
/// fold. An attention program first prints the largest score magnitude,
/// and its output pairs (a `k_out`-step fold) follow the scores' (their `p`
/// passes through `exp` and the member's rounding: the test composes
/// ADR-0043 D3).
pub fn half_src(p: &HalfProg) -> (String, (usize, u64, u64)) {
    half_src_with(p, false)
}

/// [`half_src`]; `once` (sweep-gemm `round-store`): a GEMM's magnitudes read
/// their own copies of the operands (`am`, `bm`: the same laws), so the
/// product's operands have no f32 reader — every element read of an f32
/// operand rounds to the set's member (`PrecisionPlan::store`).
pub fn half_src_with(p: &HalfProg, once: bool) -> (String, (usize, u64, u64)) {
    half_src_full(p, once, false)
}

/// [`half_src_with`]`(p, true)` with the GEMM in a callee the inliner keeps
/// (a counted loop) that returns its operand `a` beside `y` and the
/// magnitudes (review-fixes: `round-store` must not store an array that
/// escapes its function). Two elements of the returned `a` are printed after
/// the cells' pairs, each beside a magnitude of 0 (exact: the caller reads
/// the producer's f32 values). Not for attention programs.
pub fn half_src_escaping(p: &HalfProg) -> (String, (usize, u64, u64)) {
    assert!(!p.attn, "a GEMM program");
    half_src_full(p, true, true)
}

/// precision `shared-cell`: a block — an elementwise map of its input `x`
/// (`law`: one of four forms, two constants) feeding a GEMM over an f32 or
/// `bf16` weight — chained over `copies` calls the inliner copies, so every
/// copy's map runs one shared body. The first copy's `x` is closed-form (an
/// `iota` law: a `cell-law` cell with `x`'s value forwarded); each later
/// copy's `x` is the previous GEMM's output (the map an S54 epilogue member
/// of that tile, its cell read as a one-cell buffer) — one body, two device
/// variants. The weight is square (`c × c`) so the copies chain; `fw` 1:
/// stored `bf16`.
#[derive(Clone, Debug)]
pub struct SharedCellProg {
    pub dims: [u8; 2],
    pub copies: u8,
    pub law: u8,
    pub consts: [u8; 2],
    pub fw: u8,
}

pub fn shared_cell_strategy() -> impl Strategy<Value = SharedCellProg> {
    (
        (any::<u8>(), any::<u8>()),
        2u8..=3,
        0u8..4,
        (any::<u8>(), any::<u8>()),
        prop::sample::select(vec![0u8, 1]),
    )
        .prop_map(|(d, copies, law, (c0, c1), fw)| SharedCellProg {
            dims: [d.0, d.1],
            copies,
            law,
            consts: [c0, c1],
            fw,
        })
}

/// [`SharedCellProg`]'s source: each copy's output prints three cells.
pub fn shared_cell_src(p: &SharedCellProg) -> String {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let t = pick(p.dims[0], &[8, 16, 32, 48]);
    let c = pick(p.dims[1], &[32, 64, 128]);
    let (s, o) = (
        f64::from(p.consts[0] % 7 + 1) / 8.0,
        f64::from(p.consts[1] % 9) / 16.0 - 0.25,
    );
    let off = match o < 0.0 {
        true => format!("- {:?}", -o),
        false => format!("+ {o:?}"),
    };
    let body = match p.law % 4 {
        0 => format!("x[i] * {s:?} {off}"),
        1 => format!("x[i] * x[i] * {s:?} {off}"),
        2 => format!("x[i] -> v; (v > 0.0) -> {{ -true-> v; -false-> v * {s:?}; }}"),
        _ => format!("x[i] -> v; v / (1.0 + v * v * {s:?}) {off}"),
    };
    let (tw, sw, ww) = match p.fw {
        1 => ("bf16", " -> to<bf16>", " -> widen<f32>"),
        _ => ("f32", "", ""),
    };
    let mut src = format!(
        "fn block(w: [{tw}; {cc}], x: [f32; {tc}]) -> [f32; {tc}] {{
    {tc} -> iota -> map {{
        i -> {body}
    }} -> mixed;
    {c} -> iota -> along;
    {tc} -> iota -> map {{
        cell ->
        cell / {c} -> t;
        cell % {c} -> row;
        (0.0, along) -> fold {{
            sum, column ->
            w[row * {c} + column]{ww} -> wv;
            sum + wv * mixed[t * {c} + column]
        }}
    }} -> ret;
}}
fn main() {{
    {cc} -> iota -> map {{
        e -> (e % 103 * 11 + 57) % 103 - 51 -> widen<f32> -> v;
        v / {wd:?}{sw}
    }} -> w;
    {tc} -> iota -> map {{
        e -> (e * 7 + 3) % 31 - 15 -> widen<f32> -> v;
        v / 9.0
    }} -> y0;
",
        cc = c * c,
        tc = t * c,
        wd = (c * 8) as f64,
    );
    for k in 1..=p.copies {
        src += &format!("    (w, y{}) -> block -> y{k};\n", k - 1);
    }
    let cells = [0, t * c / 2 + 1, t * c - 1];
    for k in 1..=p.copies {
        for x in cells {
            src += &format!("    y{k}[{x}] -> println;\n");
        }
    }
    src += "}\n";
    src
}

/// precision `shared-store`: one GEMM function called at `copies` sites
/// (the inliner's copies share every map and fold body — as a layer loop's
/// unrolled trips and a prompt's and a decode step's projections do), its
/// `a` produced in the function and read by `: bf16 | f32` products; the
/// sites in `read` (a bit per copy) also print elements of their copy's
/// `a` — an f32 reader, so that copy's `a` stays f32 while the others are
/// stored rounded (`PrecisionPlan::store`). `twin`: `a` read by two
/// projections through one helper (two folds of one body in a single
/// store's walk). `fb` 1: `b` stored `bf16`.
#[derive(Clone, Debug)]
pub struct SharedStoreProg {
    pub dims: [u8; 3],
    pub copies: u8,
    pub read: u8,
    pub twin: bool,
    pub fb: u8,
    pub law: [u8; 2],
}

pub fn shared_store_strategy() -> impl Strategy<Value = SharedStoreProg> {
    (
        (any::<u8>(), any::<u8>(), any::<u8>()),
        2u8..=3,
        any::<u8>(),
        any::<bool>(),
        prop::sample::select(vec![0u8, 1]),
        (any::<u8>(), any::<u8>()),
    )
        .prop_map(|(d, copies, read, twin, fb, (l0, l1))| SharedStoreProg {
            dims: [d.0, d.1, d.2],
            copies,
            read: read % (1 << copies),
            twin,
            fb,
            law: [l0, l1],
        })
}

/// [`SharedStoreProg`]'s source and `(pairs, k)`: each copy prints, per
/// projection, three cells beside their `M = Σ|a·b|` (from copies of the
/// operands no product reads); then each read copy prints two elements of
/// its `a` beside a magnitude of 0 (exact: the producer's f32 values).
pub fn shared_store_src(p: &SharedStoreProg) -> (String, (usize, u64)) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let m = pick(p.dims[0], &[16, 32, 48, 64]);
    let n = pick(p.dims[1], &[8, 16, 32, 64]);
    let k = pick(p.dims[2], &[16, 32, 64, 128]);
    let (mk, kn, mn) = (m * k, k * n, m * n);
    let (a1, a2) = (
        u64::from(p.law[0] % 13) * 2 + 3,
        u64::from(p.law[1] % 11) * 2 + 5,
    );
    let (bt, bs, br) = match p.fb {
        1 => ("bf16", " -> to<bf16>", " -> widen<f32>"),
        _ => ("f32", "", ""),
    };
    let projs = if p.twin { 2 } else { 1 };
    let mut src = format!(
        "fn project(a: [f32; {mk}], b: [{bt}; {kn}]) -> [f32; {mn}] {{
    {mn} -> iota -> ic;
    {k} -> iota -> kr;
    ic -> map {{
        t -> t / {n} -> i;
        t % {n} -> j;
        (0.0, kr) -> fold: bf16 | f32 {{
            acc, q -> a[i * {k} + q] -> x;
            b[q * {n} + j]{br} -> y;
            acc + x * y
        }}
    }} -> ret;
}}

fn magnitude(a: [f32; {mk}], b: [{bt}; {kn}]) -> [f32; {mn}] {{
    {mn} -> iota -> ic;
    {k} -> iota -> kr;
    ic -> map: f32 {{
        t -> t / {n} -> i;
        t % {n} -> j;
        (0.0, kr) -> fold {{
            acc, q -> a[i * {k} + q] -> x;
            b[q * {n} + j]{br} -> y;
            x * y -> pv;
            (pv > 0.0) -> {{
                -true-> pv;
                -false-> 0.0 - pv;
            }} -> av;
            acc + av
        }}
    }} -> ret;
}}

fn gemm(s: i32) -> ([f32; {mn}], [f32; {mn}], [f32; {mn}], [f32; {mn}], [f32; {mk}]) {{
    {mk} -> iota -> ia;
    ia -> map {{
        t -> (t * {a1} + 13 + s) % 101 - 50 -> widen<f32> -> v;
        v / 7.0
    }} -> a;
    ia -> map {{
        t -> (t * {a1} + 13 + s) % 101 - 50 -> widen<f32> -> v;
        v / 7.0
    }} -> am;
    {kn} -> iota -> ib;
    ib -> map {{
        t -> (t * {a2} + 57) % 103 - 51 -> widen<f32> -> v;
        v / 9.0{bs}
    }} -> b;
    ib -> map {{
        t -> (t * {a2} + 41) % 97 - 48 -> widen<f32> -> v;
        v / 9.0{bs}
    }} -> c;
    (a, b) -> project -> y;
    (am, b) -> magnitude -> my;
"
    );
    if p.twin {
        src += "    (a, c) -> project -> z;\n    (am, c) -> magnitude -> mz;\n";
    } else {
        src += "    y -> z;\n    my -> mz;\n";
    }
    src += "    (y, my, z, mz, a) -> ret;\n}\n\nfn main() {\n";
    let cells = [u64::from(p.law[0]) * 7 % mn, mn / 2 + 1, mn - 1];
    let mut pairs = 0;
    for c in 0..p.copies {
        src += &format!("    {} -> gemm -> r{c};\n", 3 * u64::from(c) + 1);
    }
    for c in 0..p.copies {
        for (out, mag) in [("0", "1"), ("2", "3")].into_iter().take(projs) {
            for x in cells {
                src +=
                    &format!("    r{c}.{out}[{x}] -> println;\n    r{c}.{mag}[{x}] -> println;\n");
                pairs += 1;
            }
        }
        if p.read >> c & 1 == 1 {
            for x in [u64::from(p.law[1]) * 5 % mk, mk - 1] {
                src += &format!("    r{c}.4[{x}] -> println;\n    0.0 -> println;\n");
                pairs += 1;
            }
        }
    }
    src += "}\n";
    (src, (pairs, k))
}

pub(super) fn half_src_full(p: &HalfProg, once: bool, ret: bool) -> (String, (usize, u64, u64)) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let set = match p.set {
        1 => ": bf16 | f32",
        2 => ": f16 | f32",
        _ => "",
    };
    let (a1, a2) = (
        u64::from(p.law[0] % 13) * 2 + 3,
        u64::from(p.law[1] % 11) * 2 + 5,
    );
    let store = |f: u8| half_fmt(f).map_or(String::new(), |n| format!(" -> to<{n}>"));
    let read = |f: u8, x: &str| match half_fmt(f) {
        Some(_) => format!("{x} -> widen<f32>"),
        None => x.to_string(),
    };
    let abs = "(pv > 0.0) -> { -true-> pv; -false-> 0.0 - pv; } -> av; acc + av";
    if !p.attn {
        let m = pick(p.dims[0], &[16, 32, 48, 64, 128]);
        let n = pick(p.dims[1], &[8, 16, 32, 64, 128]);
        let k = pick(p.dims[2], &[16, 32, 48, 64, 128, 256]);
        let bi = if p.bt {
            format!("b[j * {k} + q]")
        } else {
            format!("b[q * {n} + j]")
        };
        let (ar, br) = (read(p.fa, &format!("a[i * {k} + q]")), read(p.fb, &bi));
        let post = if p.post {
            format!(
                " -> u; a[i * {k}]{} -> r; u * 0.5 + r",
                if p.fa > 0 { " -> widen<f32>" } else { "" }
            )
        } else {
            String::new()
        };
        let cells = [
            u64::from(p.law[0]) * 7 % (m * n),
            m * n / 2 + 1,
            m * n - 1 - u64::from(p.law[1]) % n,
        ];
        let head = if ret {
            let ta = half_fmt(p.fa).unwrap_or("f32");
            format!(
                "fn gemm(n: i32) -> ([{ta}; {mk}], [f32; {mn}], [f32; {mn}], i32) {{
    mut s: i32 <- 0;
    mut z: i32 <- 0;
    loop {{
        (s < n) -> {{
            -true-> {{
                z + s -> z;
                s + 1 -> s;
                -> loop;
            }}
            -false-> z -> zz;
        }}
    }}
",
                mk = m * k,
                mn = m * n,
            )
        } else {
            "fn main() {\n".to_string()
        };
        let mut src = head
            + &format!(
                "    {mk} -> iota -> ia;
    ia -> map {{ t -> (t * {a1} + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0{sa} }} -> a;
    {kn} -> iota -> ib;
    ib -> map {{ t -> (t * {a2} + 57) % 103 - 51 -> widen<f32> -> v; v / 9.0{sb} }} -> b;
    {mn} -> iota -> ic;
    {k} -> iota -> kr;
    ic -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold{set} {{ acc, q -> {ar} -> x; {br} -> y; acc + x * y }}{post} }} -> y;
",
            mk = m * k,
            kn = k * n,
            mn = m * n,
            sa = store(p.fa),
            sb = store(p.fb),
        );
        let (am, bm) = if once {
            src += &format!(
                "    ia -> map {{ t -> (t * {a1} + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0{sa} }} -> am;
    ib -> map {{ t -> (t * {a2} + 57) % 103 - 51 -> widen<f32> -> v; v / 9.0{sb} }} -> bm;
",
                sa = store(p.fa),
                sb = store(p.fb),
            );
            (
                read(p.fa, &format!("am[i * {k} + q]")),
                read(p.fb, &bi.replacen("b[", "bm[", 1)),
            )
        } else {
            (ar, br)
        };
        src += &format!(
            "    ic -> map: f32 {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, q -> {am} -> x; {bm} -> y; x * y -> pv; {abs} }} }} -> mag;
"
        );
        if ret {
            src += "    a -> ret.0;
    y -> ret.1;
    mag -> ret.2;
    zz -> ret.3;
}
fn main() {
    1 -> gemm -> r;
    r.0 -> ao;
    r.1 -> y;
    r.2 -> mag;
";
        }
        // An epilogue's `0.5·u + r`: M scales by one half, `r` exact.
        for c in cells {
            src += &format!("    y[{c}] -> println;\n    mag[{c}] -> println;\n");
        }
        let mut pairs = cells.len();
        if ret {
            let widen = if p.fa > 0 { " -> widen<f32>" } else { "" };
            for x in [u64::from(p.law[1]) * 5 % (m * k), m * k - 1] {
                src += &format!("    ao[{x}]{widen} -> println;\n    0.0 -> println;\n");
            }
            src += "    r.3 -> println;\n";
            pairs += 2;
        }
        src += "}\n";
        return (src, (pairs, k, 0));
    }
    let s = pick(p.dims[0], &[16, 32, 64]);
    let d = pick(p.dims[1], &[16, 32, 64]);
    // `bt` (attention): two heads — the scores and `P·V` read `q`, `k`, `v`
    // at the head digit (`tile_plan_radix`'s sites).
    let hn: u64 = if p.bt { 2 } else { 1 };
    // One head: the plain forms (no head digit), as `tile_plan_post` /
    // `tile_plan_narrow` take them.
    let hd = |x: &str| {
        if hn == 1 {
            String::new()
        } else {
            format!("h * {s} + {x}")
        }
    };
    let row = |x: &str| match hn {
        1 => x.to_string(),
        _ => format!("({})", hd(x)),
    };
    let (qr, kr) = (
        read(p.fa, &format!("qh[{} * {d} + e]", row("i"))),
        read(p.fb, &format!("kc[{} * {d} + e]", row("j"))),
    );
    let vr = read(p.fb, &format!("vc[{} * {d} + c]", row("j")));
    let (hsc, hso) = match hn {
        1 => (format!("t / {s} -> i;"), format!("t / {d} -> i;")),
        _ => (
            format!("t / {} -> h; t / {s} % {s} -> i;", s * s),
            format!("t / {} -> h; t / {d} % {s} -> i;", s * d),
        ),
    };
    let prow = row("i");
    let (ns, nd) = (hn * s * s, hn * s * d);
    let cells = [
        u64::from(p.law[0]) * 5 % ns,
        ns - 1 - u64::from(p.law[1]) % s,
    ];
    let outs = [u64::from(p.law[1]) * 3 % nd, nd - 1];
    let mut src = format!(
        "fn main() {{
    {nd} -> iota -> it;
    it -> map {{ t -> (t * {a1} + 13) % 101 - 50 -> widen<f32> -> v; v / 29.0{sa} }} -> qh;
    it -> map {{ t -> (t * {a2} + 57) % 103 - 51 -> widen<f32> -> v; v / 31.0{sb} }} -> kc;
    it -> map {{ t -> (t * 11 + 7) % 97 - 48 -> widen<f32> -> v; v / 13.0{sb} }} -> vc;
    {ns} -> iota -> isc;
    {d} -> iota -> dr;
    {s} -> iota -> sr;
    {hs} -> iota -> hr;
    isc -> map {{ t -> {hsc} t % {s} -> j; (0.0, dr) -> fold{set} {{ acc, e -> {qr} -> x; {kr} -> y; acc + x * y }} -> u; u * 0.25 }} -> sc;
    isc -> map: f32 {{ t -> {hsc} t % {s} -> j; (0.0, dr) -> fold {{ acc, e -> {qr} -> x; {kr} -> y; x * y -> pv; {abs} }} -> u; u * 0.25 }} -> smag;
    hr -> map {{ r -> (-1000000000.0, sr) -> fold {{ m, j -> sc[r * {s} + j] -> x; (x > m) -> {{ -true-> x; -false-> m; }} }} }} -> mx;
    isc -> map {{ t -> t / {s} -> r; sc[t] - mx[r] -> exp }} -> ex;
    hr -> map {{ r -> (0.0, sr) -> fold {{ acc, j -> acc + ex[r * {s} + j] }} }} -> l;
    isc -> map {{ t -> t / {s} -> r; ex[t] / l[r] }} -> pr;
    it -> map {{ t -> {hso} t % {d} -> c; (0.0, sr) -> fold{set} {{ acc, j -> {vr} -> y; acc + pr[{prow} * {s} + j] * y }} }} -> o;
    it -> map: f32 {{ t -> {hso} t % {d} -> c; (0.0, sr) -> fold {{ acc, j -> {vr} -> y; pr[{prow} * {s} + j] * y -> pv; {abs} }} }} -> omag;
    (0.0, isc) -> fold {{ m, t -> smag[t] -> x; (x > m) -> {{ -true-> x; -false-> m; }} }} -> smax;
    smax -> println;
",
        hs = hn * s,
        sa = store(p.fa),
        sb = store(p.fb),
    );
    for c in cells {
        src += &format!("    sc[{c}] -> println;\n    smag[{c}] -> println;\n");
    }
    for c in outs {
        src += &format!("    o[{c}] -> println;\n    omag[{c}] -> println;\n");
    }
    src += "}\n";
    (src, (cells.len(), d, s))
}
