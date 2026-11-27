//! Mapal-source generators: onto-device indices, bounds fuzz, host parallelism, layers, slices.

use super::*;

// ---------------------------------------------------------------------------
// onto-device: guarded indices, loop-carried counters as indices, loop exit
// values read in later bodies (`bounds_proof`'s conditions and loop ranges,
// the exit-binding capture, `carry-made`).
// ---------------------------------------------------------------------------

/// A program whose reads are in bounds only by the conditions that hold
/// where they run: a rotate_half-style partner read under a guard on `c % m`
/// (seven forms: both poles, swapped operands, a `Phi`-valued index, a nested
/// guard, a conjunction, and an unprovable gather that keeps its check), a
/// loop whose counter (`t0`, step `±s`, four continue-condition forms) indexes
/// a table by the step's position `base + t` (optionally under a guard), and
/// the loop's exit state and a second loop's exit counter read inside later
/// map bodies. Every value is a small integer in `f32`, so every face and
/// every order is exact.
#[derive(Clone, Debug)]
pub struct OntoProg {
    /// The map extent (a multiple of `m`).
    pub n: u64,
    /// The guard modulus (a power of two).
    pub m: u64,
    /// The guarded read's form (0..7).
    pub guard: u8,
    /// Loop trips.
    pub trips: i64,
    /// Counter init.
    pub t0: i64,
    /// Counter step magnitude.
    pub step: i64,
    /// Counting down.
    pub down: bool,
    /// The continue condition's form (0..4).
    pub cform: u8,
    /// The position base.
    pub base: i64,
    /// The step's table read under a guard too.
    pub inloop_guard: bool,
    /// A second loop's exit counter (`K`) indexes a later body.
    pub exit_k: i64,
}

impl OntoProg {
    /// A second loop carries an array whose next value reads neither the
    /// state nor the counter (`z <- out / 2 + a` each trip, `out` the first
    /// loop's device-carried exit: made once on the device, hoisted out of
    /// the trip), its exit printed. Derived from the drawn fields (no
    /// draw of its own: the other programs stay the stream's).
    pub fn hoisted(&self) -> bool {
        (self.t0 + self.base + self.exit_k + self.trips) % 2 == 0
    }
}

pub fn onto_prog_strategy() -> impl Strategy<Value = OntoProg> {
    (
        prop::sample::select(vec![2u64, 4, 8, 16]),
        1u64..5,
        0u8..7,
        1i64..5,
        0i64..4,
        1i64..3,
        any::<bool>(),
        0u8..4,
        0i64..3,
        any::<bool>(),
        1i64..4,
    )
        .prop_map(
            |(m, reps, guard, trips, t0, step, down, cform, base, inloop_guard, exit_k)| OntoProg {
                n: m * reps * 4,
                m,
                guard,
                trips,
                t0,
                step,
                down,
                cform,
                base,
                inloop_guard,
                exit_k,
            },
        )
}

/// [`OntoProg`]'s source text.
pub fn onto_src(p: &OntoProg) -> String {
    let (n, m) = (p.n, p.m);
    let h = m / 2;
    // The counter's values on the advancing trips: `t0 + k·s` (up) or
    // `t0 + (trips − 1 − k)·s` (down), k < trips.
    let s = p.step;
    let (lo, hi) = (p.t0, p.t0 + (p.trips - 1) * s);
    let (init, next, cond) = if !p.down {
        let lim = hi + 1; // the least bound that exits after `hi`
        let cond = match p.cform {
            0 => format!("(t < {lim})"),
            1 => format!("({lim} > t)"),
            2 => format!("(t <= {})", lim - 1),
            _ => format!("(t + {s} <= {})", lim - 1 + s),
        };
        (lo, format!("t + {s}"), cond)
    } else {
        let lim = lo - 1; // the greatest bound that exits after `lo`
        let cond = match p.cform {
            0 => format!("(t > {lim})"),
            1 => format!("({lim} < t)"),
            2 => format!("(t >= {})", lim + 1),
            _ => format!("(t - {s} >= {})", lim + 1 - s),
        };
        (hi, format!("t - {s}"), cond)
    };
    let pmax = (p.base + hi) as u64;
    let lt = (pmax + 1) * n;
    let k = p.exit_k;
    let gread = match p.guard {
        0 => format!("(i < {h}) -> {{ -true-> a[c + {h}]; -false-> a[c - {h}]; }}"),
        1 => format!("(i >= {h}) -> {{ -true-> a[c - {h}]; -false-> a[c + {h}]; }}"),
        2 => format!("({h} > i) -> {{ -true-> a[c + {h}]; -false-> a[c - {h}]; }}"),
        3 => format!("(i < {h}) -> {{ -true-> c + {h}; -false-> c - {h}; }} -> kk; a[kk]"),
        4 => format!(
            "(i < {h}) -> {{ -true-> a[c + {h}]; -false-> (i < {}) -> {{ -true-> a[c - {h} + 1]; -false-> a[c - {h}]; }}; }}",
            m - 1
        ),
        5 => format!("(c >= {m} && i < {h}) -> {{ -true-> a[c - {m} + {h}]; -false-> a[c]; }}"),
        _ => "perm[c] -> j; a[j]".to_string(),
    };
    let tread = if p.inloop_guard {
        format!(
            "i % {m} -> u; (u < {h}) -> {{ -true-> tab[pp * {n} + i + {h}]; -false-> tab[pp * {n} + i - {h}]; }} -> tv"
        )
    } else {
        format!("tab[pp * {n} + i] -> tv")
    };
    format!(
        "fn main() {{
    {n} -> iota -> r;
    r -> map {{ t -> (t * 5 + 3) % 11 - 5 -> widen<f32> }} -> a;
    r -> map {{ t -> (t * 3 + 1) % {n} }} -> perm;
    {lt} -> iota -> it;
    it -> map {{ t -> (t * 7 + 1) % 13 - 6 -> widen<f32> }} -> tab;
    r -> map {{ c -> c % {m} -> i; {gread} }} -> g;
    mut x: [f32; {n}] <- g;
    mut t: i32 <- {init};
    loop {{
        {cond} -> {{
            -true-> {{
                {base} + t -> pp;
                r -> map {{ i -> {tread}; x[i] + tv }} -> x;
                {next} -> t;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
{zloop}
    mut u: i32 <- 0;
    loop {{
        (u < {k}) -> {{
            -true-> {{
                u + 1 -> u;
                -> loop;
            }}
            -false-> u -> un;
        }}
    }}
    {nk} -> iota -> rk;
    rk -> map {{ j -> out[j] * 2.0 + a[un + j] }} -> y;
    y[0] -> println;
    y[{last}] -> println;
    out[{mid}] -> println;
    0.0 -> zero: f32;
    (zero, y) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
}}
",
        base = p.base,
        zloop = if p.hoisted() {
            format!(
                "    mut z: [f32; {n}] <- a;\n    mut v: i32 <- 0;\n    loop {{\n        (v < {}) -> {{\n            -true-> {{\n                r -> map {{ i -> out[i] / 2.0 + a[i] }} -> z;\n                v + 1 -> v;\n                -> loop;\n            }}\n            -false-> z -> zo;\n        }}\n    }}\n    zo[{}] -> println;\n",
                p.trips + 1,
                n / 2 + 1
            )
        } else {
            String::new()
        },
        nk = n as i64 - k,
        last = n as i64 - k - 1,
        mid = n / 2 + 1,
    )
}

/// `bounds_proof`'s soundness fuzz: the [`OntoProg`] shapes with every
/// offset, threshold, comparison, step and limit drawn freely, so some reads
/// leave their array (the program traps) and some stay inside without the
/// condition that would prove it. A proof is sound when no proven read ever
/// leaves its array: the native build (proven reads unchecked) traps exactly
/// where the interpreter does.
#[derive(Clone, Debug)]
pub struct BoundsFuzz {
    pub n: u64,
    pub m: u64,
    pub op: u8,
    pub thr: i64,
    pub d1: i64,
    pub d2: i64,
    pub t0: i64,
    pub step: i64,
    pub cop: u8,
    pub lim: i64,
    pub base: i64,
    pub rows: u64,
}

pub fn bounds_fuzz_strategy() -> impl Strategy<Value = BoundsFuzz> {
    (
        prop::sample::select(vec![4u64, 8, 16]),
        1u64..4,
        0u8..6,
        0i64..24,
        0i64..24,
        0i64..24,
        -2i64..5,
        prop::sample::select(vec![-2i64, -1, 1, 2, 3]),
        0u8..6,
        -3i64..9,
        -1i64..3,
        (0u8..4, 0i64..4),
    )
        .prop_map(
            |(m, reps, op, thr, d1, d2, t0, step, cop, lim, base, (aligned, slack))| {
                // Offsets and thresholds around the modulus: a read is in
                // bounds, out by one, or in only under its condition; half
                // the guards are rotate_half's (`i < m/2`, `±m/2`) shifted
                // by at most one.
                let w = m as i64;
                let (op, thr, d1, d2) = if aligned < 2 {
                    let h = w / 2;
                    let e = d1 % 3 - 1;
                    (0, h + i64::from(aligned) * e, h, h)
                } else {
                    (op, thr % (w + 3) - 1, d1 % (w + 3) - 1, d2 % (w + 3) - 1)
                };
                // The table's rows: the furthest position the trips reach
                // (at most 6, the counter moving by `step` from `t0`), plus
                // a slack of −1..2 (out by one, exact, or spare).
                let far = (0..6).map(|k| base + t0 + k * step).max().unwrap_or(0);
                let rows = (far + slack).clamp(1, 64) as u64;
                BoundsFuzz {
                    n: m * reps * 2,
                    m,
                    op,
                    thr,
                    d1,
                    d2,
                    t0,
                    step,
                    cop,
                    lim,
                    base,
                    rows,
                }
            },
        )
}

pub(super) fn cmp_op(k: u8) -> &'static str {
    ["<", "<=", ">", ">=", "==", "!="][k as usize % 6]
}

/// [`BoundsFuzz`]'s source text.
pub fn bounds_fuzz_src(p: &BoundsFuzz) -> String {
    let (n, m) = (p.n, p.m);
    let op = cmp_op(p.op);
    let cop = cmp_op(p.cop);
    let lt = p.rows * n;
    format!(
        "fn main() {{
    {n} -> iota -> r;
    r -> map {{ t -> (t * 5 + 3) % 11 - 5 -> widen<f32> }} -> a;
    {lt} -> iota -> it;
    it -> map {{ t -> (t * 7 + 1) % 13 - 6 -> widen<f32> }} -> tab;
    r -> map {{ c -> c % {m} -> i; (i {op} {thr}) -> {{ -true-> a[c + {d1}]; -false-> a[c - {d2}]; }} }} -> g;
    mut x: [f32; {n}] <- g;
    mut t: i32 <- {t0};
    mut k: i32 <- 0;
    loop {{
        (t {cop} {lim} && k < 6) -> {{
            -true-> {{
                {base} + t -> pp;
                r -> map {{ i -> x[i] + tab[pp * {n} + i] }} -> x;
                t + {step} -> t;
                k + 1 -> k;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
    out[0] -> println;
    0.0 -> zero: f32;
    (zero, out) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
}}
",
        thr = p.thr,
        d1 = p.d1,
        d2 = p.d2,
        t0 = p.t0,
        lim = p.lim,
        base = p.base,
        step = p.step,
    )
}

/// onto-device `host-par`: a GPU build's host work. A device tile site `x`
/// (`q · w`, `rows × 512`), a gather of it through an unranged permutation
/// (trap-capable: it stays on the host, and so does everything it feeds),
/// then a chain of clean maps over the gathered array — each reading one or
/// two earlier arrays at shifted cells (`(c + s) % n`) and a scalar of the
/// host's — the host-par candidates.
#[derive(Clone, Debug)]
pub struct HostParProg {
    pub rows: u64,
    pub maps: Vec<[u8; 3]>,
}

pub fn host_par_strategy() -> impl Strategy<Value = HostParProg> {
    (
        prop::sample::select(vec![64u64, 128, 256, 512]),
        prop::collection::vec(prop::array::uniform3(any::<u8>()), 1..4),
    )
        .prop_map(|(rows, maps)| HostParProg { rows, maps })
}

/// [`HostParProg`]'s source text.
pub fn host_par_src(p: &HostParProg) -> String {
    let n = p.rows * 512;
    let mut s = format!(
        "fn main() {{
    {n} -> iota -> r;
    {rq} -> iota -> rq;
    32768 -> iota -> rw;
    64 -> iota -> kd;
    rq -> map {{ e -> (e * 5 + 3) % 31 - 15 -> widen<f32> -> v; v / 16.0 }} -> q;
    rw -> map {{ e -> (e * 7 + 13) % 101 - 50 -> widen<f32> -> v; v / 64.0 }} -> w;
    r -> map {{ e -> e / 512 -> i; e % 512 -> j; (0.0, kd) -> fold {{ acc, d -> acc + q[i * 64 + d] * w[d * 512 + j] }} }} -> x;
    r -> map {{ t -> (t * 7 + 3) % {n} }} -> perm;
    r -> map {{ c -> x[perm[c]] }} -> h0;
",
        rq = p.rows * 64
    );
    for (k, m) in p.maps.iter().enumerate() {
        let a = format!("h{}", m[0] as usize % (k + 1));
        let b = format!("h{}", m[1] as usize % (k + 1));
        let sh = 1 + m[2] as u64 % 7;
        let body = match m[2] % 3 {
            0 => format!("{a}[c] * 2.0 + {b}[(c + {sh}) % {n}]"),
            1 => format!("{a}[(c + {sh}) % {n}] - {b}[c] / 4.0"),
            _ => format!(
                "({a}[c] > 0.0) -> {{ -true-> {a}[c]; -false-> {b}[(c + {sh}) % {n}] * 0.5; }}"
            ),
        };
        s += &format!("    r -> map {{ c -> {body} }} -> h{};\n", k + 1);
    }
    let last = p.maps.len();
    s += &format!(
        "    h{last}[0] -> println;\n    h{last}[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, h{last}) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n}}\n",
        n - 1
    );
    s
}

/// **device-geometry generated programs** (`crates/backends/llvm/tests/device_geometry.rs`):
/// the four shapes the model-scale layer showed as device gaps, at sizes and
/// forms drawn freely — a sum of products whose map body goes on past its
/// fold (`tile-post`), attention over grouped query heads (`head-group`), a
/// per-sequence row append to a loop-carried `[B][S][W]` state
/// (`piece-delta`), and a row norm whose root is a second fold (`row-tree`).
/// Every value is a small integer in `f32` (or a quotient the host and the
/// device compute with the same body text from an exact sum), so every face
/// and every fold order agrees with the interpreter.
#[derive(Clone, Debug)]
pub struct LayerProg {
    /// 0 epilogue GEMM, 1 grouped heads, 2 append loop, 3 row norm.
    pub kind: u8,
    /// GEMM rows (1: a GEMV, the split-K reduce's epilogue), lanes, depth.
    pub m: u64,
    pub n: u64,
    pub k: u64,
    /// The epilogue's form (`kind` 0: 0..6; `kind` 1: the mask / divisor).
    pub epi: u8,
    /// Grouped heads: sequences, KV heads, query heads per KV head,
    /// positions, head width; QKᵀ (`true`) or P·V.
    pub b: u64,
    pub g: u64,
    pub per: u64,
    pub l: u64,
    pub d: u64,
    pub qk: bool,
    /// Append loop: sequences `b`, positions `l`, row width `d`, trips, the
    /// first position.
    pub trips: i64,
    pub p0: i64,
    /// Grouped heads: K/V shared by every `bb` (a prefill's query tokens).
    /// Row norm: rows `m`, width `n`, Heron steps `k`, a third local fold;
    /// its reader scales each cell by its row's factor.
    pub extra: bool,
}

pub fn layer_strategy() -> impl Strategy<Value = LayerProg> {
    (
        0u8..4,
        prop::sample::select(vec![1u64, 4, 8, 16, 32]),
        prop::sample::select(vec![8u64, 16, 32, 64, 128]),
        prop::sample::select(vec![4u64, 8, 16, 32, 64, 256]),
        0u8..7,
        (1u64..3, 1u64..3, prop::sample::select(vec![1u64, 2, 4, 4])),
        // Positions and head widths up to a staged tile's lane block (128).
        (
            prop::sample::select(vec![8u64, 16, 64, 128, 256]),
            prop::sample::select(vec![4u64, 8, 32, 128]),
        ),
        any::<bool>(),
        (1i64..4, 0i64..3),
        any::<bool>(),
    )
        .prop_map(
            |(kind, m, n, k, epi, (b, g, per), (l, d), qk, (trips, p0), extra)| LayerProg {
                kind,
                m,
                n,
                k,
                epi,
                b,
                g,
                per,
                l,
                d,
                qk,
                trips,
                p0,
                extra,
            },
        )
}

/// [`LayerProg`]'s source text.
pub fn layer_prog_src(p: &LayerProg) -> String {
    let data = |name: &str, len: u64, a: u64, c: u64, md: u64| {
        let h = md / 2;
        format!(
            "    {len} -> iota -> i_{name};\n    i_{name} -> map {{ t -> (t * {a} + {c}) % {md} - {h} -> widen<f32> }} -> {name};\n"
        )
    };
    match p.kind {
        // A sum of products, then elementwise terms of the same cell.
        0 => {
            let (m, n, k) = (p.m, p.n, p.k);
            let mut s = String::from("fn main() {\n");
            s += &data("a", m * k, 5, 3, 7);
            s += &data("w", k * n, 3, 1, 5);
            s += &data("x", m * n, 7, 2, 9);
            s += &data("bias", n, 2, 1, 5);
            s += &format!(
                "    {m} -> iota -> i_dv;\n    i_dv -> map {{ t -> t % 3 + 1 -> widen<f32> }} -> dv;\n"
            );
            s += &format!("    {k} -> iota -> rk;\n    {} -> iota -> o;\n", m * n);
            // A number the map captures: a whole fold of the residual.
            s += "    0.0 -> z0: f32;\n    (z0, bias) -> fold { acc, v -> acc + v } -> sc;\n";
            let fold = format!(
                "(0.0, rk) -> fold {{ acc, kk -> acc + a[i * {k} + kk] * w[kk * {n} + j] }}"
            );
            let body = match p.epi {
                0 => format!("{fold} -> y;\n        x[e] + y"),
                1 => format!("{fold} -> y;\n        y * 0.5 + bias[j]"),
                2 => format!(
                    "(j <= i + 2) -> {{\n            -true-> {{ {fold} -> y; y + 1.0 }}\n            -false-> -7.0;\n        }}"
                ),
                3 => format!("{fold} -> y;\n        y / dv[i]"),
                4 => format!("{fold} -> y;\n        y * 2.0 + sc"),
                5 => format!(
                    "{fold} -> y;\n        (y > 0.0) -> {{ -true-> y; -false-> x[e] * 0.25; }}"
                ),
                _ => format!("{fold} -> y;\n        x[e] + y * bias[j] - sc"),
            };
            s += &format!(
                "    o -> map {{\n        e -> e / {n} -> i;\n        e % {n} -> j;\n        {body}\n    }} -> c;\n"
            );
            s += &format!(
                "    c[0] -> println;\n    c[{}] -> println;\n    c[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, c) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n}}\n",
                m * n - 1,
                (m * n) / 2 + 1
            );
            s
        }
        // Attention over grouped query heads: QKᵀ (masked at a position the
        // program computes: a uniform) or P·V (divided by a per-row sum).
        1 => {
            let (b, g, per, l, d) = (p.b, p.g, p.per, p.l, p.d);
            let h = g * per;
            let mut s = String::from("fn main() {\n");
            s += &data("q", b * h * d, 5, 3, 7);
            // `extra`: one K/V for every `bb` — `bb` a query position (a
            // prefill's token), read by `q`/`pe` alone: the output's token
            // digit is then above the heads' (a digit-affine output).
            let (kn, kb) = match p.extra {
                true => (l * g * d, String::new()),
                false => (b * l * g * d, format!("bb * {l} + ")),
            };
            s += &data("kc", kn, 3, 1, 5);
            s += &data("pe", b * h * l, 7, 2, 5);
            s += &format!(
                "    {} -> iota -> i_den;\n    i_den -> map {{ t -> t % 4 + 1 -> widen<f32> }} -> den;\n",
                b * h
            );
            s += &format!("    {d} -> iota -> rd;\n    {l} -> iota -> rl;\n");
            // The mask position: a number computed from data (a uniform).
            s += &format!(
                "    0 -> z0: i32;\n    (z0, rd) -> fold {{ acc, v -> acc + v % 2 }} -> pp0;\n    pp0 % {l} -> pos;\n"
            );
            if p.qk {
                let post = match p.epi % 3 {
                    0 => "dt * 0.5".to_string(),
                    1 => "(r <= pos) -> { -true-> dt * 0.5; -false-> -1000.0; }".to_string(),
                    _ => "(r <= hh) -> { -true-> dt; -false-> 0.0 - dt; }".to_string(),
                };
                s += &format!(
                    "    {} -> iota -> o;\n    o -> map {{\n        e -> e / {} -> bb;\n        e / {l} % {h} -> hh;\n        e % {l} -> r;\n        (0.0, rd) -> fold {{\n            acc, u -> acc + q[(bb * {h} + hh) * {d} + u] * kc[(({kb}r) * {g} + hh / {per}) * {d} + u]\n        }} -> dt;\n        {post}\n    }} -> s;\n",
                    b * h * l,
                    h * l
                );
            } else {
                let post = match p.epi % 2 {
                    0 => "av / den[bb * {h} + hh]".replace("{h}", &h.to_string()),
                    _ => "av".to_string(),
                };
                s += &format!(
                    "    {} -> iota -> o;\n    o -> map {{\n        e -> e / {} -> bb;\n        e / {d} % {h} -> hh;\n        e % {d} -> u;\n        (0.0, rl) -> fold {{\n            acc, r -> acc + pe[(bb * {h} + hh) * {l} + r] * kc[(({kb}r) * {g} + hh / {per}) * {d} + u]\n        }} -> av;\n        {post}\n    }} -> s;\n",
                    b * h * d,
                    h * d
                );
            }
            let n = if p.qk { b * h * l } else { b * h * d };
            s += &format!(
                "    s[0] -> println;\n    s[{}] -> println;\n    s[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, s) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n}}\n",
                n - 1,
                n / 2 + 1
            );
            s
        }
        // A per-sequence row append to a carried `[B][S][W]` state.
        2 => {
            let (b, l, w) = (p.b, p.l.max(p.trips as u64 + p.p0 as u64 + 1), p.d);
            let n = b * l * w;
            let mut s = String::from("fn main() {\n");
            s += &data("kc0", n, 3, 1, 5);
            s += &data("nw", b * w, 5, 2, 7);
            s += &format!("    {n} -> iota -> rc;\n    {} -> iota -> rbw;\n", b * w);
            s += &format!(
                "    mut kc: [f32; {n}] <- kc0;\n    mut acc: [f32; {}] <- nw;\n    mut t: i32 <- 0;\n",
                b * w
            );
            s += &format!(
                "    loop {{\n        (t < {}) -> {{\n            -true-> {{\n                {} + t -> p;\n                rbw -> map {{ e -> acc[e] + 1.0 }} -> acc;\n                rc -> map {{\n                    c -> c / {} -> bb;\n                    c / {w} % {l} -> r;\n                    (r == p) -> {{\n                        -true-> acc[bb * {w} + c % {w}] * 2.0;\n                        -false-> kc[c];\n                    }}\n                }} -> kc;\n                t + 1 -> t;\n                -> loop;\n            }}\n            -false-> kc -> out;\n        }}\n    }}\n",
                p.trips,
                p.p0,
                l * w
            );
            s += &format!(
                "    out[0] -> println;\n    out[{}] -> println;\n    out[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, out) -> fold {{ a2, v -> a2 + v }} -> total;\n    total -> println;\n}}\n",
                n - 1,
                (p.p0 as u64) * w + 1
            );
            s
        }
        // A row norm: the sum of squares of a row, then a Heron root (a
        // second fold that loads nothing).
        _ => {
            let (rows, width) = (p.m.max(1), p.n.max(32));
            let steps = 4 + p.k % 8;
            let mut s = String::from("fn main() {\n");
            s += &data("x", rows * width, 5, 3, 7);
            s += &format!(
                "    {rows} -> iota -> rr;\n    {width} -> iota -> rw;\n    {steps} -> iota -> rh;\n"
            );
            let extra = if p.extra {
                "        4 -> iota -> r4;\n        (sd, r4) -> fold { a2, kk -> a2 * 1.0 } -> sd2;\n        1.0 / sd2"
            } else {
                "        1.0 / sd"
            };
            s += &format!(
                "    rr -> map {{\n        b -> (0.0, rw) -> fold {{\n            a1, i -> x[b * {width} + i] -> v;\n            a1 + v * v\n        }} -> ss;\n        ss / {width}.0 + 1.0 -> ra;\n        0.5 + 0.5 * ra -> s0;\n        (s0, rh) -> fold {{ a3, kk -> 0.5 * (a3 + ra / a3) }} -> sd;\n{extra}\n    }} -> rinv;\n"
            );
            // The norm's reader: each cell scaled by its row's factor.
            s += &format!(
                "    i_x -> map {{ e -> x[e] * rinv[e / {width}] }} -> xn;\n    rinv[0] -> println;\n    rinv[{}] -> println;\n    xn[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, xn) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n}}\n",
                rows - 1,
                rows * width - 1
            );
            s
        }
    }
}

/// **long-prefill `uniform-base` generated programs**
/// (`crates/backends/llvm/tests/device_geometry.rs`): a loop over `groups`
/// KV-head groups whose counter `gg` slices the operands — the group's query
/// heads (`q[(bb·h + gg·per + hh)·d + u]`), its KV head
/// (`kc[(r·g + gg)·d + u]`) and its weight panel (`w[(gg·K + k)·n + j]`) —
/// as an attention half (QKᵀ or P·V, plain, masked at a position the
/// counter moves, or scaled per row) and the group's projection added into
/// a carried `y`. Head widths include odd ones (`d = 3`: a pointer moved by
/// a non-vector step; `d = 4`: one 16-byte vector of `f32`); `b` rows the staged tiles take. The K/V
/// cache is stored `f32` or `bf16` (`narrow`: `to<bf16>`, read `-> widen<f32>` — ADR-0043's narrow
/// operand, its pointer moved by the counter in its own 2-byte elements). Every value is a small
/// integer or an exact quotient by 2 or 4, so every face and fold order agrees with the interpreter.
#[derive(Clone, Debug)]
pub struct SliceProg {
    /// Query rows (tokens or sequences), KV groups (the loop's trips),
    /// query heads a group, positions, head width, projection width.
    pub b: u64,
    pub groups: u64,
    pub per: u64,
    pub l: u64,
    pub d: u64,
    pub n: u64,
    /// QKᵀ (`true`) or P·V.
    pub qk: bool,
    /// 0 plain, 1 masked (QKᵀ: `r ≤ hh + gg`) / doubled, 2 halved / per-row
    /// divisor.
    pub epi: u8,
    /// K/V shared by every sequence (a prefill's tokens) or per sequence.
    pub shared: bool,
    /// A row fold reads the carry's init before the loop (its device copy
    /// then dies at the loop's entry: long-prefill `carry-take`).
    pub pre: bool,
    /// The K/V cache stored `bf16` (ADR-0043), read through `widen<f32>`.
    pub narrow: bool,
}

pub fn slice_strategy() -> impl Strategy<Value = SliceProg> {
    (
        (
            prop::sample::select(vec![4u64, 16, 32]),
            1u64..5,
            prop::sample::select(vec![1u64, 2, 4]),
        ),
        (
            prop::sample::select(vec![16u64, 64]),
            prop::sample::select(vec![3u64, 4, 8, 32]),
            prop::sample::select(vec![16u64, 64]),
        ),
        any::<bool>(),
        0u8..3,
        (any::<bool>(), any::<bool>(), any::<bool>()),
    )
        .prop_map(
            |((b, groups, per), (l, d, n), qk, epi, (shared, pre, narrow))| SliceProg {
                b,
                groups,
                per,
                l,
                d,
                n,
                qk,
                epi,
                shared,
                pre,
                narrow,
            },
        )
}

/// [`SliceProg`]'s source text.
pub fn slice_prog_src(p: &SliceProg) -> String {
    let data = |name: &str, len: u64, a: u64, c: u64, md: u64| {
        let h = md / 2;
        format!(
            "    {len} -> iota -> i_{name};\n    i_{name} -> map {{ t -> (t * {a} + {c}) % {md} - {h} -> widen<f32> }} -> {name};\n"
        )
    };
    let (b, g, per, l, d, n) = (p.b, p.groups, p.per, p.l, p.d, p.n);
    let h = g * per;
    // One group's output: `b` rows of `kk` (its heads' cells).
    let x = if p.qk { l } else { d };
    let kk = per * x;
    let (kn, kb) = match p.shared {
        true => (l * g * d, String::new()),
        false => (b * l * g * d, format!("bb * {l} + ")),
    };
    let mut s = String::from("fn main() {\n");
    s += &data("q", b * h * d, 5, 3, 7);
    s += &match p.narrow {
        true => data("kc", kn, 3, 1, 5)
            .replace("widen<f32> } -> kc;", "widen<f32> -> to<bf16> } -> kc;"),
        false => data("kc", kn, 3, 1, 5),
    };
    s += &data("pe", b * h * l, 7, 2, 5);
    s += &data("w", g * kk * n, 3, 2, 5);
    s += &data("y0", b * n, 5, 1, 9);
    s += &format!(
        "    {} -> iota -> i_den;\n    i_den -> map {{ t -> (t % 2 + 1) * 2 -> widen<f32> }} -> den;\n",
        b * h
    );
    s += &format!(
        "    {d} -> iota -> rd;\n    {l} -> iota -> rl;\n    {kk} -> iota -> rk;\n    {} -> iota -> o;\n    {} -> iota -> ry;\n",
        b * kk,
        b * n
    );
    if p.pre {
        s += &format!(
            "    {b} -> iota -> rb;\n    {n} -> iota -> rn;\n    rb -> map {{ i -> (0.0, rn) -> fold {{ acc, j -> acc + y0[i * {n} + (j * 3) % {n}] }} }} -> ys;\n    ys[0] -> println;\n"
        );
    }
    s += &format!(
        "    mut y: [f32; {}] <- y0;\n    mut gg: i32 <- 0;\n",
        b * n
    );
    // The cache's element: `f32` read in the product, or `bf16` widened
    // first (an exact widening the narrow tile site reads through).
    let kread = format!("kc[(({kb}r) * {g} + gg) * {d} + u]");
    let (kpre, kv) = match p.narrow {
        true => (
            format!("{kread} -> widen<f32> -> kv;\n                        "),
            "kv".to_string(),
        ),
        false => (String::new(), kread),
    };
    let att = if p.qk {
        let fold = format!(
            "(0.0, rd) -> fold {{\n                        acc, u -> {kpre}acc + q[(bb * {h} + gg * {per} + hh) * {d} + u] * {kv}\n                    }}"
        );
        let body = match p.epi {
            0 => format!("{fold}"),
            1 => format!(
                "(r <= hh + gg) -> {{\n                        -true-> {{ {fold} -> dt; dt }}\n                        -false-> -1000.0;\n                    }}"
            ),
            _ => format!("{fold} -> dt;\n                    dt * 0.5"),
        };
        format!(
            "                o -> map {{\n                    e -> e / {kk} -> bb;\n                    e / {l} % {per} -> hh;\n                    e % {l} -> r;\n                    {body}\n                }} -> s;\n"
        )
    } else {
        let fold = format!(
            "(0.0, rl) -> fold {{\n                        acc, r -> {kpre}acc + pe[(bb * {h} + gg * {per} + hh) * {l} + r] * {kv}\n                    }}"
        );
        let body = match p.epi {
            0 => format!("{fold}"),
            1 => format!("{fold} -> av;\n                    av * 2.0"),
            _ => format!("{fold} -> av;\n                    av / den[bb * {h} + gg * {per} + hh]"),
        };
        format!(
            "                o -> map {{\n                    e -> e / {kk} -> bb;\n                    e / {d} % {per} -> hh;\n                    e % {d} -> u;\n                    {body}\n                }} -> s;\n"
        )
    };
    s += &format!(
        "    loop {{\n        (gg < {g}) -> {{\n            -true-> {{\n{att}                ry -> map {{\n                    e -> e / {n} -> i;\n                    e % {n} -> j;\n                    (0.0, rk) -> fold {{\n                        acc, k -> acc + s[i * {kk} + k] * w[(gg * {kk} + k) * {n} + j]\n                    }} -> yo;\n                    y[e] + yo\n                }} -> y;\n                gg + 1 -> gg;\n                -> loop;\n            }}\n            -false-> y -> out;\n        }}\n    }}\n"
    );
    s += &format!(
        "    out[0] -> println;\n    out[{}] -> println;\n    out[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, out) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n}}\n",
        b * n - 1,
        (b * n) / 2 + 1
    );
    s
}
