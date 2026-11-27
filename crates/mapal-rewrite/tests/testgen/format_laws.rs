//! Mapal-source generators: format generics, integer laws and widening laws.

use super::*;

// ---------------------------------------------------------------------------
// ADR-0051: format generics on generated programs.

/// A program of helpers generic over a size and a format (`fn f<n, t:
/// float>`, the `to` / `widen` builtins inside), each call giving a drawn
/// prefix of its arguments explicitly (`-> store<4, bf16>`, `-> store<4>`,
/// `-> store`) and the rest inferred — from the argument's format, or from the
/// result's annotation where the format reaches only the output. Its
/// hand-specialized twin writes one function per (helper, size, format) and
/// no explicit argument.
#[derive(Clone, Debug)]
pub struct FormatProg {
    /// Per call: the helper, a size draw, a format draw, the explicit-prefix
    /// draw (and whether the result is annotated).
    pub calls: Vec<[u8; 4]>,
    pub seed: u8,
}

/// A helper generic over `n` and the format `t` (kind `kind`): signature and
/// body with `@{e}` sizes (as [`GenericHelper`]) and `$t` for the format.
pub struct FormatHelper {
    pub name: &'static str,
    pub kind: &'static str,
    sig: &'static str,
    body: &'static str,
    /// The format reaches the output only: a call infers it from the
    /// result's annotation (or gives it explicitly).
    pub output_only: bool,
    /// A scalar helper (no size parameter); else an array helper over
    /// `[f32; n]` / `[t; n]`.
    pub scalar: bool,
}

pub const FORMAT_HELPERS: &[FormatHelper] = &[
    FormatHelper {
        name: "store",
        kind: "float",
        sig: "(x: [f32; @{n}]) -> [$t; @{n}]",
        body: "x -> map { v -> v -> to<$t> } -> ret;",
        output_only: true,
        scalar: false,
    },
    FormatHelper {
        name: "load",
        kind: "float",
        sig: "(x: [$t; @{n}]) -> [f32; @{n}]",
        body: "x -> map { v -> v -> widen<f32> } -> ret;",
        output_only: false,
        scalar: false,
    },
    FormatHelper {
        name: "total",
        kind: "float",
        sig: "(x: [$t; @{n}]) -> f32",
        body: "@{n} -> iota -> idx;\n(0.0, idx) -> fold { s, i -> x[i] -> widen<f32> -> v; s + v } -> ret;",
        output_only: false,
        scalar: false,
    },
    FormatHelper {
        name: "rescale",
        kind: "float",
        sig: "(x: [$t; @{n}], k: f32) -> [$t; @{n}]",
        body: "x -> map { v -> v -> widen<f32> -> * k -> to<$t> } -> ret;",
        output_only: false,
        scalar: false,
    },
    FormatHelper {
        name: "matvec",
        kind: "float",
        sig: "(w: [$t; @{n * n}], x: [f32; @{n}]) -> [f32; @{n}]",
        body: "@{n} -> iota -> cols;\n@{n} -> iota -> map { i -> (0.0, cols) -> fold { s, j -> w[i * @{n} + j] -> widen<f32> -> v; s + v * x[j] } } -> ret;",
        output_only: false,
        scalar: false,
    },
    FormatHelper {
        name: "pick",
        kind: "narrow",
        sig: "(x: f32) -> $t",
        body: "x -> to -> ret;",
        output_only: true,
        scalar: true,
    },
];

/// The formats a draw picks among (`float` helpers); a `narrow` helper skips
/// `f32`.
pub(super) const FORMAT_DRAWS: [&str; 5] = ["f32", "bf16", "f16", "f8e4m3", "f8e5m2"];

pub fn format_prog_strategy() -> impl Strategy<Value = FormatProg> {
    (
        prop::collection::vec(prop::array::uniform4(any::<u8>()), 1..7),
        any::<u8>(),
    )
        .prop_map(|(calls, seed)| FormatProg { calls, seed })
}

impl FormatProg {
    /// Each call: (helper index, size, format, explicit prefix length,
    /// annotated result).
    fn draws(&self) -> Vec<(usize, u64, &'static str, usize, bool)> {
        self.calls
            .iter()
            .map(|c| {
                let h = usize::from(c[0]) % FORMAT_HELPERS.len();
                let g = &FORMAT_HELPERS[h];
                let fmt = if g.kind == "narrow" {
                    FORMAT_DRAWS[1 + usize::from(c[2]) % 4]
                } else {
                    FORMAT_DRAWS[usize::from(c[2]) % 5]
                };
                let arity = if g.scalar { 1 } else { 2 };
                let prefix = usize::from(c[3]) % (arity + 1);
                // A format the output alone carries needs the stage or the
                // annotation; otherwise the annotation is a draw.
                let annotated = (g.output_only && prefix < arity) || c[3] / 3 % 2 == 0;
                (h, 1 + u64::from(c[1]) % 9, fmt, prefix, annotated)
            })
            .collect()
    }

    /// Every instance, by helper: its (size, format) pairs in the order
    /// lowering declares them (sizes numerically, then formats in ADR-0051's
    /// order `f64, f32, bf16, f16, f8e4m3, f8e5m2`; a scalar helper has no
    /// size).
    fn instances(&self) -> Vec<(usize, Vec<(u64, &'static str)>)> {
        let rank = |f: &str| {
            ["f64", "f32", "bf16", "f16", "f8e4m3", "f8e5m2"]
                .iter()
                .position(|x| *x == f)
        };
        let mut by: std::collections::BTreeMap<
            usize,
            std::collections::BTreeSet<(u64, Option<usize>, &'static str)>,
        > = Default::default();
        for (h, n, f, _, _) in self.draws() {
            let n = if FORMAT_HELPERS[h].scalar { 0 } else { n };
            by.entry(h).or_default().insert((n, rank(f), f));
        }
        by.into_iter()
            .map(|(h, s)| (h, s.into_iter().map(|(n, _, f)| (n, f)).collect()))
            .collect()
    }

    /// `(instance name, twin name)` pairs.
    pub fn renames(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for (h, pairs) in self.instances() {
            let g = &FORMAT_HELPERS[h];
            for (n, f) in pairs {
                if g.scalar {
                    out.push((format!("{}<{f}>", g.name), format!("{}_{f}", g.name)));
                } else {
                    out.push((
                        format!("{}<{n}, {f}>", g.name),
                        format!("{}_{n}_{f}", g.name),
                    ));
                }
            }
        }
        out
    }

    /// Calls giving at least one argument explicitly; calls whose format
    /// only the result's annotation determines; calls whose format the
    /// argument determines; instances at `f32` (the conversions' identity).
    pub fn counters(&self) -> [usize; 4] {
        let mut c = [0; 4];
        for (h, _, f, prefix, _) in self.draws() {
            let g = &FORMAT_HELPERS[h];
            let arity = if g.scalar { 1 } else { 2 };
            c[0] += usize::from(prefix > 0);
            c[1] += usize::from(g.output_only && prefix < arity);
            c[2] += usize::from(!g.output_only && prefix < arity);
            c[3] += usize::from(f == "f32");
        }
        c
    }

    /// The generic program (formatted).
    pub fn source(&self) -> String {
        self.program(false)
    }

    /// The hand-specialized twin (formatted).
    pub fn hand(&self) -> String {
        self.program(true)
    }

    fn program(&self, hand: bool) -> String {
        let mut src = String::new();
        for (h, pairs) in self.instances() {
            let g = &FORMAT_HELPERS[h];
            if hand {
                for (n, f) in &pairs {
                    let env = [("n", *n)];
                    let name = if g.scalar {
                        format!("{}_{f}", g.name)
                    } else {
                        format!("{}_{n}_{f}", g.name)
                    };
                    src.push_str(&format!(
                        "fn {name}{} {{\n{}\n}}\n\n",
                        render_sizes(g.sig, &env).replace("$t", f),
                        render_sizes(g.body, &env).replace("$t", f)
                    ));
                }
            } else {
                let list = if g.scalar {
                    format!("t: {}", g.kind)
                } else {
                    format!("n, t: {}", g.kind)
                };
                src.push_str(&format!(
                    "fn {}<{list}>{} {{\n{}\n}}\n\n",
                    g.name,
                    render_sizes(g.sig, &[]).replace("$t", "t"),
                    render_sizes(g.body, &[]).replace("$t", "t")
                ));
            }
        }
        src.push_str("fn main() {\n");
        for (k, (h, n, f, prefix, annotated)) in self.draws().into_iter().enumerate() {
            let g = &FORMAT_HELPERS[h];
            let (a, b) = (
                3 + (usize::from(self.seed) + 5 * k) % 7,
                (usize::from(self.seed) / 3 + k) % 11,
            );
            // The operand: an f32 closed form of the index (exact in every
            // drawn format's range), rounded to the format where the helper
            // takes `[t; n]`.
            // `matvec` reads an `n × n` matrix and an `n` vector.
            let len = if g.name == "matvec" { n * n } else { n };
            src.push_str(&format!(
                "    {len} -> iota -> map {{ i -> (i * {a} + {b}) % 13 - 6 -> widen<f32> -> * 0.125 }} -> x{k};\n"
            ));
            if g.name == "matvec" {
                src.push_str(&format!(
                    "    {n} -> iota -> map {{ i -> (i * {b} + {a}) % 7 - 3 -> widen<f32> }} -> u{k};\n"
                ));
            }
            if !g.output_only {
                src.push_str(&format!(
                    "    x{k} -> map {{ v -> v -> to<{f}> }} -> y{k};\n"
                ));
            }
            let callee = if hand {
                if g.scalar {
                    format!("{}_{f}", g.name)
                } else {
                    format!("{}_{n}_{f}", g.name)
                }
            } else {
                let args: Vec<String> = if g.scalar {
                    vec![f.to_string()]
                } else {
                    vec![n.to_string(), f.to_string()]
                };
                match prefix {
                    0 => g.name.to_string(),
                    p => format!("{}<{}>", g.name, args[..p].join(", ")),
                }
            };
            let ty = match (g.name, g.scalar) {
                (_, true) => f.to_string(),
                ("total", _) => "f32".to_string(),
                ("load" | "matvec", _) => format!("[f32; {n}]"),
                _ => format!("[{f}; {n}]"),
            };
            let bind = if annotated {
                format!("r{k}: {ty}")
            } else {
                format!("r{k}")
            };
            let input = match g.name {
                "store" => format!("x{k}"),
                "load" | "total" => format!("y{k}"),
                "rescale" => format!("(y{k}, 0.5)"),
                "matvec" => format!("(y{k}, u{k})"),
                _ => format!("x{k}[{}]", n - 1),
            };
            src.push_str(&format!("    {input} -> {callee} -> {bind};\n"));
            match g.name {
                "total" => src.push_str(&format!("    r{k} -> println;\n")),
                "pick" => src.push_str(&format!("    r{k} -> widen<f32> -> println;\n")),
                "load" | "matvec" => src.push_str(&format!(
                    "    (0.0, r{k}) -> fold {{ s, v -> s + v }} -> println;\n"
                )),
                _ => src.push_str(&format!(
                    "    (0.0, r{k}) -> fold {{ s, v -> v -> widen<f32> -> w; s + w }} -> println;\n"
                )),
            }
        }
        src.push_str("}\n");
        mapal_syntax::format_source(&src)
            .unwrap_or_else(|d| panic!("generated program does not parse: {d:?}\n{src}"))
    }
}

// ---------------------------------------------------------------------------
//  ADR-0051 D4: integer laws a `widen<f32>` converts (`BoundsProof::widen_range`)
// ---------------------------------------------------------------------------

/// An integer law of a map's index: `+ − ·` over the index and constants,
/// `÷ %` by a positive constant (trap-free), constants around 2²⁴ and products
/// that wrap past `i32`.
#[derive(Clone, Debug)]
pub enum IntLaw {
    Index,
    Const(i64),
    Bin(Box<IntLaw>, char, Box<IntLaw>),
}

/// Constants a law draws: small, around 2²⁴ (the exactness edge of `f32`),
/// and large enough that a product wraps.
pub(super) const LAW_CONSTS: [i64; 20] = [
    0, 1, 2, 3, 7, 13, 57, 255, 4096, 4097, 65536, 262144, 1000003, 16777215, 16777216, 16777217,
    33554432, 268435456, 1073741824, 2147483647,
];

/// Divisors (`÷ %`): positive, so no law traps; `16777217` / `16777218` sit on
/// either side of the edge a `%` alone keeps.
pub(super) const LAW_DIVISORS: [i64; 10] = [
    2, 3, 7, 101, 103, 4096, 1000003, 16777217, 16777218, 33554433,
];

/// The map's extents (`iota`): the index range the facts start from.
pub(super) const LAW_EXTENTS: [i64; 10] = [1, 2, 3, 5, 8, 33, 64, 65, 257, 4097];

pub(super) fn int_law_strategy() -> impl Strategy<Value = IntLaw> {
    let leaf = prop_oneof![
        3 => Just(IntLaw::Index),
        2 => (0..LAW_CONSTS.len(), any::<bool>()).prop_map(|(k, neg)| {
            let c = LAW_CONSTS[k];
            IntLaw::Const(if neg && c != 0 { -c } else { c })
        }),
        1 => (0i64..300).prop_map(IntLaw::Const),
    ];
    leaf.prop_recursive(4, 16, 2, |inner| {
        prop_oneof![
            3 => (inner.clone(), prop::sample::select(vec!['+', '-', '*']), inner.clone())
                .prop_map(|(a, op, b)| IntLaw::Bin(Box::new(a), op, Box::new(b))),
            2 => (inner, prop::sample::select(vec!['/', '%']), 0..LAW_DIVISORS.len())
                .prop_map(|(a, op, k)| {
                    IntLaw::Bin(Box::new(a), op, Box::new(IntLaw::Const(LAW_DIVISORS[k])))
                }),
        ]
    })
}

impl IntLaw {
    /// The law as Mapal text over `i` (a negative constant as `(0 - c)`).
    pub fn text(&self) -> String {
        match self {
            IntLaw::Index => "i".into(),
            IntLaw::Const(c) if *c < 0 => format!("(0 - {})", -c),
            IntLaw::Const(c) => c.to_string(),
            IntLaw::Bin(a, op, b) => format!("({} {op} {})", a.text(), b.text()),
        }
    }

    /// The law at `i` with `i32` arithmetic (wrapping `+ − ·`, truncating
    /// `÷ %`) and whether an intermediate left `i32` — for coverage counters
    /// only (the interpreter is the oracle).
    pub fn eval(&self, i: i64) -> (i32, bool) {
        match self {
            IntLaw::Index => (i as i32, false),
            IntLaw::Const(c) => (*c as i32, false),
            IntLaw::Bin(a, op, b) => {
                let ((x, wa), (y, wb)) = (a.eval(i), b.eval(i));
                let (v, exact) = match op {
                    '+' => (x.wrapping_add(y), i64::from(x) + i64::from(y)),
                    '-' => (x.wrapping_sub(y), i64::from(x) - i64::from(y)),
                    '*' => (x.wrapping_mul(y), i64::from(x) * i64::from(y)),
                    '/' => (x.wrapping_div(y), i64::from(x) / i64::from(y)),
                    _ => (x.wrapping_rem(y), i64::from(x) % i64::from(y)),
                };
                (v, wa || wb || i64::from(v) != exact)
            }
        }
    }

    /// Whether a `%` divides a dividend that wraps past `i32` at some index
    /// in `cells` (the case ADR-0051's `%` rule exists for).
    pub fn mod_of_wrapped(&self, cells: &[i64]) -> bool {
        match self {
            IntLaw::Bin(a, '%', _) => cells.iter().any(|&i| a.eval(i).1) || a.mod_of_wrapped(cells),
            IntLaw::Bin(a, _, b) => a.mod_of_wrapped(cells) || b.mod_of_wrapped(cells),
            _ => false,
        }
    }
}

/// A map over `extent -> iota` whose cells convert `law` with `widen<f32>` (or
/// its rounding twin `to<f32>`), under a guard arm `i < c` / `i >= c` when
/// drawn (the condition narrows the index where the conversion runs).
#[derive(Clone, Debug)]
pub struct WidenLawProg {
    pub extent: i64,
    pub law: IntLaw,
    /// `(below, c)`: the arm `i < c` (`below`) or `i >= c`.
    pub guard: Option<(bool, i64)>,
}

pub fn widen_law_strategy() -> impl Strategy<Value = WidenLawProg> {
    // A third of the laws are a scaled law's remainder, `(law · k + c) % d`
    // with `k` large: the product wraps past `i32` and only the `%` bounds
    // the result (the shape ADR-0051's `%` rule is for).
    let law = prop_oneof![
        2 => int_law_strategy(),
        1 => (int_law_strategy(), 9..LAW_CONSTS.len(), 0i64..300, 0..LAW_DIVISORS.len())
            .prop_map(|(l, k, c, d)| {
                let scaled = IntLaw::Bin(Box::new(l), '*', Box::new(IntLaw::Const(LAW_CONSTS[k])));
                let shifted = IntLaw::Bin(Box::new(scaled), '+', Box::new(IntLaw::Const(c)));
                IntLaw::Bin(Box::new(shifted), '%', Box::new(IntLaw::Const(LAW_DIVISORS[d])))
            }),
    ]
    // One line of a formatted program (no line past 120 columns).
    .prop_filter("a law fits a line", |l| l.text().len() <= 72);
    (
        0..LAW_EXTENTS.len(),
        law,
        prop::option::of((any::<bool>(), 0i64..70)),
    )
        .prop_map(|(e, law, guard)| WidenLawProg {
            extent: LAW_EXTENTS[e],
            law,
            guard,
        })
}

impl WidenLawProg {
    /// The indices where the conversion runs (the guard's arm), in order.
    pub fn cells(&self) -> Vec<i64> {
        (0..self.extent)
            .filter(|&i| match self.guard {
                Some((true, c)) => i < c,
                Some((false, c)) => i >= c,
                None => true,
            })
            .collect()
    }

    /// The program converting with `verb` (`widen` or `to`), formatted: each
    /// converted cell's sum printed (the conversion's value reaches output).
    pub fn source(&self, verb: &str) -> String {
        let law = self.law.text();
        let body = match self.guard {
            None => format!("i -> {law} -> {verb}<f32>"),
            Some((below, c)) => {
                let cmp = if below { "<" } else { ">=" };
                format!("i -> (i {cmp} {c}) -> {{ -true-> {law} -> {verb}<f32>; -false-> 0.0; }}")
            }
        };
        formatted(format!(
            "fn main() {{
    {extent} -> iota -> idx;
    idx -> map {{ {body} }} -> converted;
    (0.0, converted) -> fold {{ s, x -> s + x }} -> total;
    total -> println;
}}
",
            extent = self.extent
        ))
    }

    /// The oracle twin: the law's integers over exactly the cells where the
    /// conversion runs, their largest and smallest printed; `None` when no
    /// cell runs it.
    pub fn observer(&self) -> Option<String> {
        let cells = self.cells();
        let (first, n) = (*cells.first()?, cells.len());
        Some(formatted(format!(
            "fn main() {{
    {n} -> iota -> cells;
    cells -> map {{ j -> j + {first} -> i; {law} }} -> raw;
    (raw[0], raw) -> fold {{ m, x -> (x > m) -> {{ -true-> x; -false-> m; }} }} -> hi;
    (raw[0], raw) -> fold {{ m, x -> (x < m) -> {{ -true-> x; -false-> m; }} }} -> lo;
    hi -> println;
    lo -> println;
}}
",
            law = self.law.text()
        )))
    }
}
