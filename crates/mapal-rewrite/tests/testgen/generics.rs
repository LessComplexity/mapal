//! Mapal-source generators: size-generic helpers called at several sizes.

use super::*;

// --- size-generics: helpers called at several sizes, as Mapal source --------

/// ADR-0047: a program of size-generic helpers, each called at one or more
/// sizes, and its hand-specialized twin ([`GenericProg::hand`]): every
/// instance written out with literal sizes as `{helper}_{v₁}_{v₂}`, declared
/// where its generic is, by size tuple — the order monomorphization declares
/// them in, so the two lower to the same IR, names and spans aside.
/// Helpers: a bare size two parameters share (`dot`), a product with a solved
/// factor (`project`: `[f32; r * c]` beside `[f32; c]`), a sum (`window`:
/// `[f32; n + 2]`), a literal product (`pool`: `[f32; 2 * n]`), a repeated
/// unknown (`trace`: `[f32; n * n]`), a nested generic call (`norm` → `dot`),
/// a size only the result's annotation binds (`ramp`), and size expressions
/// in value position (`flip`: `r * c -> iota`, `m[r * c - 1 - i]`; `last`:
/// `x[n - 1]`).
#[derive(Clone, Debug)]
pub struct GenericProg {
    /// Per call: the helper (index into [`GENERIC_HELPERS`], mod its length),
    /// two size draws, and a third — even: the helper is called again at the
    /// sizes it draws.
    pub calls: Vec<[u8; 4]>,
    pub seed: u8,
}

/// A generic helper: its size names, its signature after the size list and
/// its body, both with `@{e}` size expressions (`e` over the size names, `+`
/// `-` `*`, written out by the generic and evaluated by the twin) and `$dot`
/// for its one nested generic call.
pub struct GenericHelper {
    pub name: &'static str,
    pub sizes: &'static [&'static str],
    sig: &'static str,
    body: &'static str,
}

pub const GENERIC_HELPERS: &[GenericHelper] = &[
    GenericHelper {
        name: "dot",
        sizes: &["n"],
        sig: "(a: [f32; @{n}], b: [f32; @{n}]) -> f32",
        body: "(a, b) -> zip -> map { p -> p.0 * p.1 } -> m;\n(0.0, m) -> fold { s, v -> s + v } -> ret;",
    },
    GenericHelper {
        name: "project",
        sizes: &["r", "c"],
        sig: "(w: [f32; @{r * c}], x: [f32; @{c}]) -> [f32; @{r}]",
        body: "@{c} -> iota -> cols;\n@{r} -> iota -> map { i -> (0.0, cols) -> fold { s, j -> s + w[i * @{c} + j] * x[j] } } -> ret;",
    },
    GenericHelper {
        name: "window",
        sizes: &["n"],
        sig: "(x: [f32; @{n + 2}]) -> [f32; @{n}]",
        body: "@{n} -> iota -> map { i -> x[i] + x[i + 1] * 0.5 + x[i + 2] * 0.25 } -> ret;",
    },
    GenericHelper {
        name: "pool",
        sizes: &["n"],
        sig: "(x: [f32; @{2 * n}]) -> [f32; @{n}]",
        body: "@{n} -> iota -> map { i -> x[2 * i] + x[2 * i + 1] } -> ret;",
    },
    GenericHelper {
        name: "trace",
        sizes: &["n"],
        sig: "(m: [f32; @{n * n}]) -> f32",
        body: "@{n} -> iota -> d;\n(0.0, d) -> fold { s, i -> s + m[i * @{n} + i] } -> ret;",
    },
    GenericHelper {
        name: "norm",
        sizes: &["n"],
        sig: "(x: [f32; @{n}]) -> f32",
        body: "(x, x) -> $dot -> ret;",
    },
    GenericHelper {
        name: "ramp",
        sizes: &["n"],
        sig: "(s: f32) -> [f32; @{n}]",
        body: "@{n} -> iota -> map { i -> i -> widen<f32> -> * s } -> ret;",
    },
    GenericHelper {
        name: "flip",
        sizes: &["r", "c"],
        sig: "(m: [f32; @{r * c}], v: [f32; @{c}]) -> [f32; @{r * c}]",
        body: "@{r * c} -> iota -> map { i -> m[@{r * c - 1} - i] * v[i % @{c}] } -> ret;",
    },
    GenericHelper {
        name: "last",
        sizes: &["n"],
        sig: "(x: [f32; @{n}]) -> f32",
        body: "x[@{n - 1}] -> ret;",
    },
];

pub fn generic_prog_strategy() -> impl Strategy<Value = GenericProg> {
    (
        prop::collection::vec(prop::array::uniform4(any::<u8>()), 1..7),
        any::<u8>(),
    )
        .prop_map(|(calls, seed)| GenericProg { calls, seed })
}

/// A size expression of the template language (`+`, `-`, `*` over names and
/// literals, left to right with `*` first) at `env`.
pub(super) fn eval_size(e: &str, env: &[(&str, u64)]) -> i64 {
    let atom = |t: &str| {
        env.iter()
            .find(|(k, _)| *k == t)
            .map_or_else(|| t.parse().expect("size atom"), |(_, v)| *v as i64)
    };
    let toks: Vec<&str> = e.split_whitespace().collect();
    let (mut sum, mut sign, mut prod) = (0, 1, atom(toks[0]));
    for pair in toks[1..].chunks(2) {
        match pair[0] {
            "*" => prod *= atom(pair[1]),
            op => {
                sum += sign * prod;
                sign = if op == "+" { 1 } else { -1 };
                prod = atom(pair[1]);
            }
        }
    }
    sum + sign * prod
}

/// `text` with every `@{e}` written out (`env` empty: the generic) or
/// evaluated (the twin), and `$dot` as `dot` or `dot_{n}`.
pub(super) fn render_sizes(text: &str, env: &[(&str, u64)]) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find("@{") {
        out.push_str(&rest[..at]);
        let end = at + rest[at..].find('}').expect("closed");
        let e = &rest[at + 2..end];
        if env.is_empty() {
            out.push_str(e);
        } else {
            out.push_str(&eval_size(e, env).to_string());
        }
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    let dot = match env.iter().find(|(k, _)| *k == "n") {
        Some((_, n)) if !env.is_empty() => format!("dot_{n}"),
        _ => "dot".to_string(),
    };
    out.replace("$dot", &dot)
}

impl GenericProg {
    /// Each call: (helper index, its size values in declared order).
    fn draws(&self) -> Vec<(usize, Vec<u64>)> {
        let mut out = Vec::new();
        for c in &self.calls {
            let h = usize::from(c[0]) % GENERIC_HELPERS.len();
            let sizes = |a: u8, b: u8| -> Vec<u64> {
                match GENERIC_HELPERS[h].sizes.len() {
                    1 => vec![1 + u64::from(a) % 9],
                    _ => vec![1 + u64::from(a) % 6, 1 + u64::from(b) % 6],
                }
            };
            out.push((h, sizes(c[1], c[2])));
            if c[3] % 2 == 0 {
                out.push((h, sizes(c[3] / 2, c[3] / 7)));
            }
        }
        out
    }

    /// Every instance the program needs, by helper index (`norm<n>` needs
    /// `dot<n>`), each helper's size tuples sorted.
    pub fn instances(&self) -> Vec<(usize, Vec<Vec<u64>>)> {
        let mut by: std::collections::BTreeMap<usize, std::collections::BTreeSet<Vec<u64>>> =
            Default::default();
        for (h, vs) in self.draws() {
            if GENERIC_HELPERS[h].name == "norm" {
                by.entry(0).or_default().insert(vs.clone());
            }
            by.entry(h).or_default().insert(vs);
        }
        by.into_iter()
            .map(|(h, s)| (h, s.into_iter().collect()))
            .collect()
    }

    /// Helpers called at two or more size tuples.
    pub fn multi_size(&self) -> usize {
        self.instances()
            .iter()
            .filter(|(_, vs)| vs.len() >= 2)
            .count()
    }

    /// Calls of `name` (`norm`: nested, `ramp`: annotation-bound, `trace`: a
    /// repeated unknown).
    pub fn calls_of(&self, name: &str) -> usize {
        self.draws()
            .iter()
            .filter(|(h, _)| GENERIC_HELPERS[*h].name == name)
            .count()
    }

    /// `(generic name, hand-written name)` of every instance.
    pub fn renames(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for (h, tuples) in self.instances() {
            let name = GENERIC_HELPERS[h].name;
            for vs in tuples {
                let list: Vec<String> = vs.iter().map(u64::to_string).collect();
                out.push((
                    format!("{name}<{}>", list.join(", ")),
                    format!("{name}_{}", list.join("_")),
                ));
            }
        }
        out
    }

    fn hand_name(h: usize, vs: &[u64]) -> String {
        let list: Vec<String> = vs.iter().map(u64::to_string).collect();
        format!("{}_{}", GENERIC_HELPERS[h].name, list.join("_"))
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
        for (h, tuples) in self.instances() {
            let g = &GENERIC_HELPERS[h];
            if hand {
                for vs in &tuples {
                    let env: Vec<(&str, u64)> =
                        g.sizes.iter().copied().zip(vs.iter().copied()).collect();
                    src.push_str(&format!(
                        "fn {}{} {{\n{}\n}}\n\n",
                        Self::hand_name(h, vs),
                        render_sizes(g.sig, &env),
                        render_sizes(g.body, &env)
                    ));
                }
            } else {
                src.push_str(&format!(
                    "fn {}<{}>{} {{\n{}\n}}\n\n",
                    g.name,
                    g.sizes.join(", "),
                    render_sizes(g.sig, &[]),
                    render_sizes(g.body, &[])
                ));
            }
        }
        src.push_str("fn main() {\n");
        // One operand array per call and parameter: a closed form of its index.
        let arr = |src: &mut String, name: &str, len: i64, k: usize| {
            let a = 3 + (usize::from(self.seed) + 5 * k) % 7;
            let b = (usize::from(self.seed) / 3 + k) % 11;
            src.push_str(&format!(
                "    {len} -> iota -> map {{ i -> (i * {a} + {b}) % 13 - 6 -> widen<f32> -> * 0.125 }} -> {name};\n"
            ));
        };
        for (k, (h, vs)) in self.draws().into_iter().enumerate() {
            let g = &GENERIC_HELPERS[h];
            let env: Vec<(&str, u64)> = g.sizes.iter().copied().zip(vs.iter().copied()).collect();
            let callee = if hand {
                Self::hand_name(h, &vs)
            } else {
                g.name.to_string()
            };
            let ev = |e: &str| eval_size(e, &env);
            let r = format!("r{k}");
            let scalar = match g.name {
                "dot" => {
                    arr(&mut src, &format!("a{k}"), ev("n"), k);
                    arr(&mut src, &format!("b{k}"), ev("n"), k + 1);
                    src.push_str(&format!("    (a{k}, b{k}) -> {callee} -> {r};\n"));
                    true
                }
                "project" | "flip" => {
                    arr(&mut src, &format!("a{k}"), ev("r * c"), k);
                    arr(&mut src, &format!("b{k}"), ev("c"), k + 1);
                    src.push_str(&format!("    (a{k}, b{k}) -> {callee} -> {r};\n"));
                    false
                }
                "window" | "pool" | "trace" | "norm" | "last" => {
                    let len = match g.name {
                        "window" => ev("n + 2"),
                        "pool" => ev("2 * n"),
                        "trace" => ev("n * n"),
                        _ => ev("n"),
                    };
                    arr(&mut src, &format!("a{k}"), len, k);
                    src.push_str(&format!("    a{k} -> {callee} -> {r};\n"));
                    matches!(g.name, "trace" | "norm" | "last")
                }
                _ => {
                    // `ramp`: only the result's annotation binds `n`.
                    src.push_str(&format!(
                        "    0.375 -> {callee} -> {r}: [f32; {}];\n",
                        ev("n")
                    ));
                    false
                }
            };
            if scalar {
                src.push_str(&format!("    {r} -> println;\n"));
            } else {
                src.push_str(&format!(
                    "    (0.0, {r}) -> fold {{ s, v -> s + v }} -> s{k};\n    s{k} -> println;\n    {r}[0] -> println;\n"
                ));
            }
        }
        src.push_str("}\n");
        mapal_syntax::format_source(&src)
            .unwrap_or_else(|d| panic!("generated program does not parse: {d:?}\n{src}"))
    }
}

/// ADR-0048 counters of one [`GenericProg::tops`] rendering: top-level
/// bindings, of them computed (not one literal), float literals bound at the
/// top level, uses of the size-generic alias `Vec<…>`, values read through a
/// `_` field, reads of a top-level array.
#[derive(Clone, Copy, Debug, Default)]
pub struct TopsUse {
    pub bindings: usize,
    pub computed: usize,
    pub floats: usize,
    pub aliases: usize,
    pub defaults: usize,
    pub tables: usize,
}

impl GenericProg {
    /// ADR-0048: the generic program written with top-level bindings — every
    /// operand length and the operands' scale a top-level name, declared in a
    /// seed-shuffled order after `main` (some lengths computed as a sum), and
    /// every helper's `[f32; e]` the size-generic alias `Vec<e>`. `pure`:
    /// nothing else, so it lowers to [`Self::source`]'s IR (names and spans
    /// aside). Otherwise also: every one-array operand passed wrapped in a
    /// size-generic struct `Tagged<n>` whose `_` is the array (read through
    /// it by the call's inference and argument), and each scalar result
    /// scaled by a top-level table of ones (`ones[k % 4]`, exact).
    pub fn tops(&self, pure: bool) -> (String, TopsUse) {
        let generic = self.program(false);
        let mut u = TopsUse::default();
        let mut tops: Vec<String> = Vec::new();
        let mut src = String::from("type Vec<n> {\n    _: [f32; n]\n}\n\n");
        if !pure {
            src.push_str("type Tagged<n> {\n    _: Vec<n>,\n    tag: i32,\n}\n\n");
        }
        let mut body = String::new();
        let mut in_main = false;
        let mut k_len = 0usize;
        for line in generic.lines() {
            if line.starts_with("fn ") && line.contains('<') && !in_main {
                // A helper's signature line: its `[f32; e]` → `Vec<e>`.
                let mut out = String::new();
                let mut rest = line;
                while let Some(at) = rest.find("[f32; ") {
                    out.push_str(&rest[..at]);
                    let end = at + rest[at..].find(']').expect("closed");
                    out.push_str(&format!("Vec<{}>", &rest[at + 6..end]));
                    u.aliases += 1;
                    rest = &rest[end + 1..];
                }
                out.push_str(rest);
                body.push_str(&out);
                body.push('\n');
                continue;
            }
            if line.starts_with("fn main") {
                in_main = true;
            }
            if in_main && let Some(arrow) = line.find(" -> iota -> map") {
                // `    L -> iota -> map { … * 0.125 } -> aK;`
                let len: u64 = line[..arrow].trim().parse().expect("length literal");
                let name = format!("len_{k_len}");
                k_len += 1;
                if len > 2 && (usize::from(self.seed) + k_len) % 3 != 0 {
                    u.computed += 1;
                    tops.push(format!("{} + {} -> {name};", len / 2, len - len / 2));
                } else {
                    tops.push(format!("{len} -> {name};"));
                }
                let rest = line[arrow..].replace("* 0.125", "* scale");
                body.push_str(&format!("    {name}{rest}\n"));
                continue;
            }
            if in_main && let Some(at) = line.find(": [f32; ") {
                // `ramp`'s annotation-bound result.
                let end = at + line[at..].find(']').expect("closed");
                let name = format!("len_{k_len}");
                k_len += 1;
                tops.push(format!("{} -> {name};", &line[at + 8..end]));
                body.push_str(&format!(
                    "{}: Vec<{name}>{}\n",
                    &line[..at],
                    &line[end + 1..]
                ));
                u.aliases += 1;
                continue;
            }
            if in_main {
                body.push_str(&line.replace("* 0.125", "* scale"));
            } else {
                body.push_str(line);
            }
            body.push('\n');
        }
        tops.push("0.125 -> scale;".to_string());
        u.floats += 1;
        if !pure {
            tops.push(
                "4 -> iota -> map {\n    i -> i * 0 + 1 -> widen<f32>\n} -> ones;".to_string(),
            );
            let mut out = String::new();
            for line in body.lines() {
                let t = line.trim_start();
                // `rK -> println;` of a scalar result: through the table.
                if let Some(r) = t.strip_suffix(" -> println;")
                    && let Some(k) = r.strip_prefix('r')
                    && let Ok(k) = k.parse::<usize>()
                {
                    out.push_str(&format!("    r{k} * ones[{}] -> println;\n", k % 4));
                    u.tables += 1;
                    continue;
                }
                // `aK -> f -> rK;` of a one-array call: through `_`.
                if let Some(rest) = t.strip_prefix('a')
                    && let Some((k, call)) = rest.split_once(" -> ")
                    && let Ok(k) = k.parse::<usize>()
                    && call.ends_with(&format!(" -> r{k};"))
                {
                    out.push_str(&format!(
                        "    Tagged {{ _: a{k}, tag: {k} }} -> t{k};\n    t{k} -> {call}\n"
                    ));
                    u.defaults += 1;
                    continue;
                }
                out.push_str(line);
                out.push('\n');
            }
            body = out;
        }
        // Shuffle the bindings by the seed (they are order-independent).
        let n = tops.len();
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (usize::from(self.seed) * 31 + i * 17) % (i + 1);
            order.swap(i, j);
        }
        u.bindings = n;
        src.push_str(&body);
        src.push('\n');
        for i in order {
            src.push_str(&tops[i]);
            src.push('\n');
        }
        let text = mapal_syntax::format_source(&src)
            .unwrap_or_else(|d| panic!("generated program does not parse: {d:?}\n{src}"));
        (text, u)
    }
}
