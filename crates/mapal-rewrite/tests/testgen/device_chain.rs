//! Mapal-source generators: device-sized loops and chains of mixed algorithms.

use super::*;

// --- S65: device-sized loops, as Mapal source --------------------------------

/// A loop over `[f32; n·n]` arrays sized for a device (`n` 32 or 64): one or
/// two carried arrays (`x`, `y`) from closed-form inits and a loop-invariant
/// `w`; per trip `ops` over them — affine sites (`Σ_k A[i·n+k]·B[k·n+j]`),
/// per-cell maps (relu, scale, a two-array sum), guards choosing between
/// arrays, row writes into the state (`x[it·n + j] <- v`, the value one
/// element of another array); each state's next value picked from what the
/// trip made, the state itself or `w`; the exit forwards a state, read by one
/// element or whole. Built as source text (no parser dependency here).
#[derive(Clone, Debug)]
pub struct DeviceProg {
    pub big: bool,
    pub trips: u8,
    pub two: bool,
    pub ops: Vec<[u8; 3]>,
    pub next: [u8; 2],
    pub exit: bool,
    pub whole: bool,
}

pub fn device_prog_strategy() -> impl Strategy<Value = DeviceProg> {
    (
        prop::bool::weighted(0.25),
        1u8..5,
        any::<bool>(),
        prop::collection::vec(prop::array::uniform3(any::<u8>()), 1..5),
        prop::array::uniform2(any::<u8>()),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(|(big, trips, two, ops, next, exit, whole)| DeviceProg {
            big,
            trips,
            two,
            ops,
            next,
            exit,
            whole,
        })
}

/// [`DeviceProg`]'s source text.
pub fn device_src(p: &DeviceProg) -> String {
    let n = if p.big { 64 } else { 32 };
    let nn = n * n;
    let law = |name: &str, a: u32, b: u32, m: u32, scale: u32| {
        format!(
            "    ta -> map {{ t -> (t * {a} + {b}) % {m} - {h} -> widen<f32> -> v; v / {scale}.0 }} -> {name};\n",
            h = m / 2
        )
    };
    let mut s = format!("fn main() {{\n    {nn} -> iota -> ta;\n    {n} -> iota -> krange;\n");
    s += &law("a0", 7, 13, 101, 8 * n);
    s += &law("a1", 5, 3, 97, 8 * n);
    s += &law("w", 11, 57, 103, 4 * n);
    s += &format!("    mut x: [f32; {nn}] <- a0;\n");
    if p.two {
        s += &format!("    mut y: [f32; {nn}] <- a1;\n");
    }
    s += "    mut it: i32 <- 0;\n    loop {\n        (it < ";
    s += &format!("{}) -> {{\n            -true-> {{\n", p.trips);
    let ind = "                ";
    let mut arrs: Vec<String> = vec!["x".into()];
    if p.two {
        arrs.push("y".into());
    }
    arrs.push("w".into());
    let pick = |v: &[String], k: u8| v[k as usize % v.len()].clone();
    // The ops chain: each reads the one before (so every value the trip makes
    // is live — a loop runs only what feeds its next state or its exit), and
    // a second operand from anything made so far; the last is `x`'s next.
    let mut cur = "x".to_string();
    for (z, &[op, _, b]) in p.ops.iter().enumerate() {
        // (Most chains start at a site: a device unit to place.)
        let op = if z == 0 && op % 4 != 3 { 0 } else { op % 8 };
        let (x, y) = (cur.clone(), pick(&arrs, b));
        let z = format!("z{z}");
        s += ind;
        s += &match op {
            0 => format!(
                "ta -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, krange) -> fold {{ acc, k -> acc + {x}[i * {n} + k] * {y}[k * {n} + j] }} }} -> {z};\n"
            ),
            1 => format!(
                "{x} -> map {{ v -> v -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> r; r }} -> {z};\n"
            ),
            2 => format!("{x} -> map {{ v -> v * 0.5 }} -> {z};\n"),
            3 => format!("ta -> map {{ t -> {x}[t] + {y}[t] }} -> {z};\n"),
            4 => format!(
                "(it < {}) -> {{ -true-> {x}; -false-> {y}; }} -> {z};\n",
                b % (p.trips + 1)
            ),
            // A closed form of the cell and the trip counter (a uniform), read
            // by a site: built where the site runs, never uploaded.
            6 => format!(
                "ta -> map {{ t -> ((t + it * {c}) * 7 + 13) % 101 - 50 -> widen<f32> -> v; v / {s}.0 }} -> {z}g;\n{ind}ta -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, krange) -> fold {{ acc, k -> acc + {x}[i * {n} + k] * {z}g[k * {n} + j] }} }} -> {z};\n",
                c = u32::from(b % 5) + 1,
                s = 8 * n
            ),
            // A guarded dot product per cell over the trip counter (causal
            // attention's score): no tile site, a cell whose body folds.
            7 => format!(
                "ta -> map {{ t -> t / {n} -> i; t % {n} -> j; (j <= it * {c}) -> {{ -true-> {{ (0.0, krange) -> fold {{ acc, k -> acc + {x}[i * {n} + k] * {y}[k * {n} + j] }} }} -false-> -1.0; }} }} -> {z};\n",
                c = u32::from(b % 7) + 1
            ),
            _ => {
                // A row write into the state `x`: the trip's row, its cell `j`
                // one element of the chain (a copy-out when that one lives on
                // the device).
                let j = u32::from(b) % n;
                cur = "x".into();
                format!(
                    "{x}[it * {n} + {j}] * 0.5 + 0.25 -> {z}v;\n{ind}x[it * {n} + {j}] <- {z}v;\n"
                )
            }
        };
        if op != 5 {
            arrs.push(z.clone());
            cur = z;
        }
    }
    if cur != "x" {
        s += &format!("{ind}{cur} -> x;\n");
    }
    if p.two {
        // `y`'s next: itself, `w`, `x`'s new value or the trip's first array.
        let first = arrs[3.min(arrs.len() - 1)].clone();
        let ny = pick(
            &["y".to_string(), "w".to_string(), "x".to_string(), first],
            p.next[1],
        );
        if ny != "y" {
            s += &format!("{ind}{ny} -> y;\n");
        }
    }
    let out = if p.exit && p.two { "y" } else { "x" };
    s += &format!(
        "{ind}it + 1 -> it;\n{ind}-> loop;\n            }}\n            -false-> {out} -> out;\n        }}\n    }}\n"
    );
    s += &format!("    out[0] -> println;\n    out[{}] -> println;\n", nn - 1);
    if p.whole {
        s += "    0.0 -> zero: f32;\n    (zero, out) -> fold { acc, v -> acc + v } -> total;\n    total -> println;\n";
    }
    s + "}\n"
}

// --- generic-fusion: chains of mixed algorithms, as Mapal source -------------

/// A chain of 2..9 algorithm nodes over `f32` arrays of an `r × c` grid (`r`,
/// `c` in 4..16), `k`-deep tiles, `t`-tap windows: laws (iota maps), capture-free
/// maps (arithmetic, a relu / clamp guard, a constant-trip nested fold), cells
/// reading at the cell, its row, its column, a constant cell or an affine
/// in-row cell, tile sites (row-major, transposed `a` or `b`; chained through a
/// `c × c` operand), windows (FIR, 2-D `k÷div` taps, a weightless stencil and
/// box filter), row folds (sum, max, min), a fold's row value read per cell
/// (`f(x[t], v[t ÷ c])`), whole folds, and loops (≤ 4 trips) whose body maps,
/// writes one row naturally (`c / C == it`) or runs a row-local tile over the
/// state. Operands are picked from the arrays of the same shape made so far, so
/// chains fan out and fan in. Every sink — and every intermediate whose
/// `print` bit is set (a host reader) — prints a checksum and two corners.
#[derive(Clone, Debug)]
pub struct Chain {
    pub r: u8,
    pub c: u8,
    pub k: u8,
    pub taps: u8,
    pub nodes: Vec<[u8; 4]>,
    /// Also a row value broadcast along its row beside the column index
    /// (`v[t ÷ c] + j / 8`: a cell that reads no row of its grid).
    pub broadcast: bool,
    /// Every closed-form operand (and iota) made first, before the nodes —
    /// a program that loads its weights before it computes.
    pub hoist: bool,
    /// ADR-0045: the closed-form operands that come from a file instead —
    /// law `i` (in order of making) when bit `i % 8` is set — `"l<i>.bin" ->
    /// read -> l<i>: [f32; n]`, the file holding the law's own values
    /// ([`Chain::files`]): the same program, its weights loaded.
    pub read: u8,
}

pub fn chain_strategy() -> impl Strategy<Value = Chain> {
    (
        4u8..17,
        4u8..17,
        2u8..17,
        2u8..6,
        prop::collection::vec(prop::array::uniform4(any::<u8>()), 2..10),
    )
        .prop_map(|(r, c, k, taps, nodes)| Chain {
            r,
            c,
            k,
            taps,
            nodes,
            broadcast: false,
            hoist: false,
            read: 0,
        })
}

/// [`chain_strategy`]'s grammar on rows up to 128 cells wide and `k` up to
/// 64 (compile-time-graph A): most rows a power of two of 4..128 cells (a
/// narrow row's tile splits `k` across the row's spare lanes; a wide row
/// spans several warps, and so do a row-block unit's row groups and tree
/// folds), the rest any width in 4..128; with the row
/// broadcast node ([`Chain::broadcast`]), half of them with their operands
/// made first ([`Chain::hoist`]).
pub fn wide_chain_strategy() -> impl Strategy<Value = Chain> {
    (
        4u8..17,
        prop_oneof![
            3 => prop::sample::select(vec![4u8, 8, 16, 32, 64, 128]),
            1 => 4u8..17,
            1 => 17u8..129,
        ],
        2u8..65,
        2u8..6,
        prop::collection::vec(prop::array::uniform4(any::<u8>()), 2..10),
        any::<bool>(),
    )
        .prop_map(|(r, c, k, taps, nodes, hoist)| Chain {
            r,
            c,
            k,
            taps,
            nodes,
            broadcast: true,
            hoist,
            read: 0,
        })
}

/// [`wide_chain_strategy`] with operands read from files ([`Chain::read`]):
/// most programs read some of their closed-form operands, one in eight all
/// of them; the test writes [`Chain::files`] beside the program.
pub fn read_chain_strategy() -> impl Strategy<Value = Chain> {
    (
        wide_chain_strategy(),
        prop_oneof![7 => 1u8..=255, 1 => Just(255u8)],
    )
        .prop_map(|(c, read)| Chain { read, ..c })
}

/// An array's shape in a [`Chain`]: its length is a function of the grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Shape {
    /// `r × c`.
    M,
    /// `r` (a row value).
    V,
    /// `c` (a column value).
    W,
    /// `r × k`, `k × c`, `c × k` (tile operands), `c × c`.
    A,
    B,
    Bt,
    Q,
    /// `r·c + taps − 1` (a signal), `taps`.
    S,
    H,
    /// `(r + 2)·(c + 2)` (an image for 3 × 3 taps), `9` (the taps).
    Img,
    Kw,
}

pub(super) struct ChainCtx {
    r: u64,
    c: u64,
    k: u64,
    t: u64,
    s: String,
    pool: Vec<(String, Shape)>,
    iotas: Vec<u64>,
    laws: u32,
    /// [`Chain::hoist`]: the iotas and laws, emitted before the nodes.
    pre: Option<String>,
    /// [`Chain::read`] and the files its reads load.
    read: u8,
    files: Vec<(String, Vec<u8>)>,
}

impl ChainCtx {
    fn len(&self, sh: Shape) -> u64 {
        let (r, c, k, t) = (self.r, self.c, self.k, self.t);
        match sh {
            Shape::M => r * c,
            Shape::V => r,
            Shape::W => c,
            Shape::A => r * k,
            Shape::B | Shape::Bt => k * c,
            Shape::Q => c * c,
            Shape::S => r * c + t - 1,
            Shape::H => t,
            Shape::Img => (r + 2) * (c + 2),
            Shape::Kw => 9,
        }
    }

    /// `n -> iota -> i<n>`, once per length.
    fn iota(&mut self, n: u64) -> String {
        if !self.iotas.contains(&n) {
            self.iotas.push(n);
            let line = format!("    {n} -> iota -> ix{n};\n");
            match &mut self.pre {
                Some(pre) => *pre += &line,
                None => self.s += &line,
            }
        }
        format!("ix{n}")
    }

    /// A closed-form array of shape `sh` (values in about ±6, a few bits).
    fn law(&mut self, sh: Shape, key: u8) -> String {
        let n = self.len(sh);
        self.laws += 1;
        let name = format!("l{}", self.laws);
        let (mul, add, m) = [(7, 13, 101), (11, 57, 103), (5, 3, 97), (3, 1, 89)][key as usize % 4];
        let div = [7.0f32, 9.0, 11.0, 13.0][(key as usize / 4) % 4];
        let line = if self.read >> (self.laws % 8) & 1 == 1 {
            // ADR-0045: the same values, loaded — what the law's map computes
            // (an exact i32, widened exactly, one f32 division), little-endian.
            let file = format!("{name}.bin");
            let bytes = (0..n as i32)
                .flat_map(|t| (((t * mul + add) % m - m / 2) as f32 / div).to_le_bytes())
                .collect();
            self.files.push((file.clone(), bytes));
            format!("    \"{file}\" -> read -> {name}: [f32; {n}];\n")
        } else {
            let it = self.iota(n);
            format!(
                "    {it} -> map {{ t -> (t * {mul} + {add}) % {m} - {h} -> widen<f32> -> v; v / {div:.1} }} -> {name};\n",
                h = m / 2
            )
        };
        match &mut self.pre {
            Some(pre) => *pre += &line,
            None => self.s += &line,
        }
        self.pool.push((name.clone(), sh));
        name
    }

    /// An array of shape `sh` made so far (by `pick`), else a fresh law.
    fn pick(&mut self, sh: Shape, pick: u8) -> String {
        let have: Vec<String> = self
            .pool
            .iter()
            .filter(|(_, s)| *s == sh)
            .map(|(n, _)| n.clone())
            .collect();
        // One in four picks makes a fresh operand even when one exists.
        if have.is_empty() || pick % 4 == 3 {
            return self.law(sh, pick / 4);
        }
        have[pick as usize % have.len()].clone()
    }

    fn print(&mut self, name: &str, sh: Shape) {
        let n = self.len(sh);
        self.s += &format!(
            "    (zero, {name}) -> fold {{ acc, v -> acc + v }} -> sum_{name};\n    sum_{name} -> println;\n    {name}[0] -> println;\n    {name}[{}] -> println;\n",
            n - 1
        );
    }
}

impl Chain {
    /// The chain's Mapal source.
    pub fn source(&self) -> String {
        self.render().0
    }

    /// ADR-0045: the files the chain's reads load (`(name, bytes)`, the
    /// names relative to the program's data directory); none when
    /// [`Chain::read`] is 0.
    pub fn files(&self) -> Vec<(String, Vec<u8>)> {
        self.render().1
    }

    fn render(&self) -> (String, Vec<(String, Vec<u8>)>) {
        let (r, c, k, t) = (
            u64::from(self.r),
            u64::from(self.c),
            u64::from(self.k),
            u64::from(self.taps),
        );
        let mut x = ChainCtx {
            r,
            c,
            k,
            t,
            s: String::new(),
            pool: Vec::new(),
            iotas: Vec::new(),
            laws: 0,
            pre: self.hoist.then(String::new),
            read: self.read,
            files: Vec::new(),
        };
        let rc = r * c;
        let mut printed: Vec<String> = Vec::new();
        let last = self.nodes.len() - 1;
        for (z, &[op, p, q, flags]) in self.nodes.iter().enumerate() {
            let o = format!("n{z}");
            let tm = x.iota(rc);
            // Most nodes read an `r × c` array: start from a tile site.
            let ops = if self.broadcast { 17 } else { 16 };
            let op = if z == 0 && op % 3 != 0 { 6 } else { op % ops };
            let shape = match op {
                0 => {
                    let it = x.iota(rc);
                    let (mul, add) = [(3, 5), (13, 7)][p as usize % 2];
                    x.s += &format!(
                        "    {it} -> map {{ t -> (t * {mul} + {add}) % 31 - 15 -> widen<f32> -> v; v / 4.0 }} -> {o};\n"
                    );
                    Shape::M
                }
                1 => {
                    let a = x.pick(Shape::M, p);
                    let body = match q % 4 {
                        0 => "v * 0.5 + 0.25".to_string(),
                        1 => {
                            "v -> u; (u > 0.0) -> { -true-> u; -false-> 0.0; } -> w; w".to_string()
                        }
                        2 => "v -> u; (u > 1.5) -> { -true-> 1.5; -false-> u; } -> w; w * 2.0"
                            .to_string(),
                        _ => {
                            let kk = x.iota(3);
                            format!("(v, {kk}) -> fold {{ acc, j -> acc * 0.5 + 0.125 }}")
                        }
                    };
                    x.s += &format!("    {a} -> map {{ v -> {body} }} -> {o};\n");
                    Shape::M
                }
                2 => {
                    let (a, b) = (x.pick(Shape::M, p), x.pick(Shape::M, q));
                    x.s += &format!("    {tm} -> map {{ t -> {a}[t] * {b}[t] + 0.5 }} -> {o};\n");
                    Shape::M
                }
                3 | 14 => {
                    let (a, v) = (x.pick(Shape::M, p), x.pick(Shape::V, q));
                    let e = if op == 3 {
                        format!("{a}[t] - {v}[t / {c}]")
                    } else {
                        format!("{a}[t] / ({v}[t / {c}] * {v}[t / {c}] + 1.0)")
                    };
                    x.s += &format!("    {tm} -> map {{ t -> {e} }} -> {o};\n");
                    Shape::M
                }
                4 => {
                    let (a, w) = (x.pick(Shape::M, p), x.pick(Shape::W, q));
                    x.s += &format!("    {tm} -> map {{ t -> {a}[t] + {w}[t % {c}] }} -> {o};\n");
                    Shape::M
                }
                5 => {
                    let a = x.pick(Shape::M, p);
                    // An in-row cell, one cell twice (two addresses of one
                    // array), and another array at the row index or the row's
                    // first cell (a halo outside a cell-wide unit).
                    let e = match q % 4 {
                        0 => format!("{a}[t] * 0.5 + {a}[t / {c} * {c} + {}]", c - 1),
                        1 => format!("{a}[t] - {a}[3]"),
                        2 => {
                            let b = x.pick(Shape::M, q / 4);
                            format!("{a}[t] + {b}[t / {c}] * 0.5")
                        }
                        _ => {
                            let b = x.pick(Shape::M, q / 4);
                            format!("{a}[t] - {b}[t / {c} * {c}]")
                        }
                    };
                    x.s += &format!("    {tm} -> map {{ t -> {e} }} -> {o};\n");
                    Shape::M
                }
                6 | 7 => {
                    let (kr, ai, bi) = if op == 6 {
                        let (al, bl) = (q % 3 == 1, q % 3 == 2);
                        let a = x.pick(Shape::A, p);
                        let b = x.pick(if bl { Shape::Bt } else { Shape::B }, p / 2);
                        let ai = if al {
                            format!("{a}[k * {r} + i]")
                        } else {
                            format!("{a}[i * {k} + k]")
                        };
                        let bi = if bl {
                            format!("{b}[j * {k} + k]")
                        } else {
                            format!("{b}[k * {c} + j]")
                        };
                        (x.iota(k), ai, bi)
                    } else {
                        let a = x.pick(Shape::M, p);
                        let b = x.pick(Shape::Q, q);
                        let ai = format!("{a}[i * {c} + k]");
                        let bi = format!("{b}[k * {c} + j]");
                        (x.iota(c), ai, bi)
                    };
                    let site = |o: &str, ai: &str, bi: &str| {
                        format!(
                            "    {tm} -> map {{ t ->\n        t / {c} -> i;\n        t % {c} -> j;\n        (0.0, {kr}) -> fold {{ acc, k -> acc + {ai} * {bi} }}\n    }} -> {o};\n"
                        )
                    };
                    if op == 6 && flags & 0xC0 == 0xC0 {
                        // plan-cost `linear_plan`: a sum of two folds over one
                        // `k` range, each read only by the sum (operands may
                        // repeat: a shared `a` or `b`).
                        let (a2, b2) = (x.pick(Shape::A, p ^ 0x5a), x.pick(Shape::B, q ^ 0x33));
                        let (ai2, bi2) =
                            (format!("{a2}[i * {k} + k]"), format!("{b2}[k * {c} + j]"));
                        x.s += &site(&format!("{o}p"), &ai, &bi);
                        x.s += &site(&format!("{o}q"), &ai2, &bi2);
                        x.s += &format!("    {tm} -> map {{ t -> {o}p[t] + {o}q[t] }} -> {o};\n");
                    } else {
                        x.s += &site(&o, &ai, &bi);
                    }
                    Shape::M
                }
                8 => {
                    let a = x.pick(Shape::M, p);
                    let (tr, tc) = (x.iota(r), x.iota(c));
                    let (seed, step) = match q % 3 {
                        0 => ("0.0", "acc + e"),
                        1 => ("-1000000000.0", "(e > acc) -> { -true-> e; -false-> acc; }"),
                        _ => ("1000000000.0", "(e < acc) -> { -true-> e; -false-> acc; }"),
                    };
                    x.s += &format!(
                        "    {tr} -> map {{ i -> ({seed}, {tc}) -> fold {{ acc, j -> {a}[i * {c} + j] -> e; {step} }} }} -> {o};\n"
                    );
                    Shape::V
                }
                9 | 11 => {
                    let sg = x.pick(Shape::S, p);
                    let tt = x.iota(t);
                    let e = if op == 9 {
                        let h = x.pick(Shape::H, q);
                        format!("acc + {h}[k] * {sg}[t + k]")
                    } else {
                        format!("acc + {sg}[t + k]")
                    };
                    x.s += &format!(
                        "    {tm} -> map {{ t -> (0.0, {tt}) -> fold {{ acc, k -> {e} }} }} -> {o};\n"
                    );
                    Shape::M
                }
                10 => {
                    let img = x.pick(Shape::Img, p);
                    let tq = x.iota(9);
                    let at = format!("{img}[(i + k / 3) * {} + j + k % 3]", c + 2);
                    let e = if q % 2 == 0 {
                        let w = x.pick(Shape::Kw, q / 2);
                        format!("acc + {w}[k] * {at}")
                    } else {
                        format!("acc + {at}")
                    };
                    x.s += &format!(
                        "    {tm} -> map {{ t ->\n        t / {c} -> i;\n        t % {c} -> j;\n        (0.0, {tq}) -> fold {{ acc, k -> {e} }}\n    }} -> {o};\n"
                    );
                    Shape::M
                }
                12 => {
                    let v = x.pick(Shape::V, p);
                    x.s += &format!("    {v} -> map {{ e -> e * 0.5 + 1.0 }} -> {o};\n");
                    Shape::V
                }
                13 => {
                    // A loop carrying an `r × c` state.
                    let a = x.pick(Shape::M, p);
                    let trips = 1 + u64::from(q % 4);
                    let (sv, it) = (format!("st{z}"), format!("it{z}"));
                    let body = match flags % 3 {
                        0 => {
                            let y = x.pick(Shape::M, flags / 3);
                            format!(
                                "                {sv} -> map {{ v -> v * 0.5 + 0.25 }} -> u{z};\n                {tm} -> map {{ t -> u{z}[t] + {y}[t] }} -> w{z};\n"
                            )
                        }
                        1 => format!(
                            "                {tm} -> map {{ e -> (e / {c} == {it}) -> {{ -true-> {sv}[e] * 2.0 + 1.0; -false-> {sv}[e]; }} }} -> w{z};\n"
                        ),
                        _ => {
                            let qq = x.pick(Shape::Q, flags / 3);
                            let kc = x.iota(c);
                            format!(
                                "                {tm} -> map {{ t ->\n                    t / {c} -> i;\n                    t % {c} -> j;\n                    (0.0, {kc}) -> fold {{ acc, k -> acc + {sv}[i * {c} + k] * {qq}[k * {c} + j] }}\n                }} -> w0{z};\n                w0{z} -> map {{ v -> v * 0.125 }} -> w{z};\n"
                            )
                        }
                    };
                    x.s += &format!(
                        "    mut {sv}: [f32; {rc}] <- {a};\n    mut {it}: i32 <- 0;\n    loop {{\n        ({it} < {trips}) -> {{\n            -true-> {{\n{body}                w{z} -> {sv};\n                {it} + 1 -> {it};\n                -> loop;\n            }}\n            -false-> {sv} -> {o};\n        }}\n    }}\n"
                    );
                    // A name a loop's exit binds is read here, never captured
                    // by a later body (the lowering resolves captures before
                    // the loop's exits: `L1101`).
                    x.print(&o, Shape::M);
                    printed.push(o);
                    continue;
                }
                16 => {
                    let v = x.pick(Shape::V, p);
                    x.s += &format!(
                        "    {tm} -> map {{ t -> t % {c} -> j; j -> widen<f32> -> f; {v}[t / {c}] + f / 8.0 }} -> {o};\n"
                    );
                    Shape::M
                }
                _ if q % 2 == 0 => {
                    // A cell of one array's row-first cells, read by the next
                    // at the row index: outside the reader's own cell.
                    let (a, b) = (x.pick(Shape::M, p), x.pick(Shape::M, q / 2));
                    x.s += &format!(
                        "    {tm} -> map {{ t -> {a}[t / {c} * {c}] * 0.5 }} -> {o}h;\n    {tm} -> map {{ t -> {b}[t] + {o}h[t / {c}] }} -> {o};\n"
                    );
                    Shape::M
                }
                _ => {
                    // A whole fold, printed (a host reader of its operand's
                    // fold, not of the array).
                    let a = x.pick(Shape::M, p);
                    x.s += &format!(
                        "    (zero, {a}) -> fold {{ acc, v -> acc + v * 0.5 }} -> {o};\n    {o} -> println;\n"
                    );
                    continue;
                }
            };
            x.pool.push((o.clone(), shape));
            if z == last || flags & 0x30 == 0x30 {
                x.print(&o, shape);
                printed.push(o);
            }
        }
        if printed.is_empty() {
            // The last node was a whole fold: print the last array too.
            if let Some((o, sh)) = x
                .pool
                .iter()
                .rev()
                .find(|(o, _)| o.starts_with('n'))
                .cloned()
            {
                x.print(&o, sh);
            }
        }
        let src = format!(
            "fn main() {{\n    0.0 -> zero: f32;\n{}{}}}\n",
            x.pre.unwrap_or_default(),
            x.s
        );
        (src, x.files)
    }
}
