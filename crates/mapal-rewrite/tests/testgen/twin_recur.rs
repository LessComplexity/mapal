//! Mapal-source generators: twin programs, row-recurrent loops, mixed-radix and TF32 sites.

use super::*;

// --- generic-fusion natural forms: twin programs, as Mapal source -----------

/// A device-sized loop like [`DeviceProg`] whose writes into the state `x`
/// and whose cell exponents come in two forms with the same values — the
/// **natural** one a person writes (a guarded map over the cells: a row
/// `t ÷ n == it`, a column `t % n == c`, a range `L ≤ t < L + w`, one cell
/// `t == L`; a power as a fold over a range) and the **unrolled** one the
/// compiler used to need (`Update`s at each cell, a chain of squarings).
/// Per trip `ops` over sites, per-cell maps (relu, scale, a two-array sum, a
/// clamped power), guards choosing between arrays, row folds, and those
/// writes; one or two carried arrays, an element or whole exit read.
#[derive(Clone, Debug)]
pub struct TwinProg {
    pub big: bool,
    pub trips: u8,
    pub two: bool,
    pub ops: Vec<[u8; 3]>,
    pub next: u8,
    pub exit: bool,
    pub whole: bool,
    /// Also a whole fold of the chain read as a number by the next map
    /// (attention's running max or denominator): device bytes one op of the
    /// trip makes and a later one reads.
    pub numbers: bool,
}

pub fn twin_prog_strategy() -> impl Strategy<Value = TwinProg> {
    (
        prop::bool::weighted(0.25),
        1u8..5,
        any::<bool>(),
        prop::collection::vec(prop::array::uniform3(any::<u8>()), 1..6),
        any::<u8>(),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(|(big, trips, two, ops, next, exit, whole)| TwinProg {
            big,
            trips,
            two,
            ops,
            next,
            exit,
            whole,
            numbers: false,
        })
}

/// [`twin_prog_strategy`] with whole-fold numbers ([`TwinProg::numbers`]).
pub fn twin_number_strategy() -> impl Strategy<Value = TwinProg> {
    twin_prog_strategy().prop_map(|p| TwinProg { numbers: true, ..p })
}

/// [`TwinProg`]'s source text, `natural` or unrolled. `check = (a, b, len)`
/// instruments the trip (the write-delta soundness probe): it prints
/// `dz=<k>`, the number of cells of the state `x` that changed in the trip
/// outside `[a·it + b, a·it + b + len)`.
pub fn twin_src(p: &TwinProg, natural: bool, check: Option<(i64, i64, u64)>) -> String {
    let n: u32 = if p.big { 32 } else { 16 };
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
    s += &format!(
        "    mut it: i32 <- 0;\n    loop {{\n        (it < {}) -> {{\n            -true-> {{\n",
        p.trips
    );
    let ind = "                ";
    if check.is_some() {
        s += &format!("{ind}x -> xp;\n");
    }
    let mut arrs: Vec<String> = vec!["x".into()];
    if p.two {
        arrs.push("y".into());
    }
    arrs.push("w".into());
    let pick = |v: &[String], k: u8| v[k as usize % v.len()].clone();
    // A write into `x`: the natural guarded map over the cells, or one
    // `Update` per cell of `cells` (the new value of cell `c` is `val(c)`,
    // an expression of the index text `c`).
    let write = |z: &str,
                 guard: String,
                 cells: Vec<String>,
                 val: &dyn Fn(&str) -> String|
     -> String {
        if natural {
            format!(
                "{ind}ta -> map {{ t -> {guard} -> {{ -true-> {{ {} }} -false-> x[t]; }} }} -> x;\n",
                val("t")
            )
        } else {
            cells
                .iter()
                .enumerate()
                .map(|(k, c)| format!("{ind}{} -> {z}v{k};\n{ind}x[{c}] <- {z}v{k};\n", val(c)))
                .collect()
        }
    };
    let mut cur = "x".to_string();
    for (z, &[op, a, b]) in p.ops.iter().enumerate() {
        // (Most chains start at a site: a device unit to place.)
        let ops = if p.numbers { 11 } else { 10 };
        let op = if z == 0 && op % 4 != 3 { 0 } else { op % ops };
        let (x, y) = (cur.clone(), pick(&arrs, b));
        let z = format!("z{z}");
        // A row fold of the chain so far: one value per row (the vector a row
        // or column write reads).
        let rows = format!(
            "{ind}krange -> map {{ j -> (0.0, krange) -> fold {{ acc, k -> acc + {x}[j * {n} + k] }} }} -> {z}r;\n"
        );
        let made = match op {
            0 => Some(format!(
                "{ind}ta -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, krange) -> fold {{ acc, k -> acc + {x}[i * {n} + k] * {y}[k * {n} + j] }} }} -> {z};\n"
            )),
            1 => Some(format!(
                "{ind}{x} -> map {{ v -> v -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> r; r }} -> {z};\n"
            )),
            2 => Some(format!("{ind}{x} -> map {{ v -> v * 0.5 }} -> {z};\n")),
            3 => Some(format!(
                "{ind}ta -> map {{ t -> {x}[t] + {y}[t] }} -> {z};\n"
            )),
            4 => Some(format!(
                "{ind}(it < {}) -> {{ -true-> {x}; -false-> {y}; }} -> {z};\n",
                b % (p.trips + 1)
            )),
            9 => {
                // A power of the clamped cell: `2^k` by `k` squarings.
                let k = 1 + a % 4;
                let pow = if natural {
                    format!("{k} -> iota -> sq; (v0, sq) -> fold {{ acc, q -> acc * acc }}")
                } else {
                    (0..k)
                        .map(|i| format!("v{i} * v{i} -> v{};", i + 1))
                        .collect::<String>()
                        + &format!(" v{k}")
                };
                Some(format!(
                    "{ind}{x} -> map {{ v -> v * 0.125 + 0.5 -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> c; \
                     (c < 1.0) -> {{ -true-> c; -false-> 1.0; }} -> v0; {pow} }} -> {z};\n"
                ))
            }
            10 => {
                // A whole fold of the chain (a sum or a max), read as a
                // number by every cell of the next map.
                let (seed, step) = match a % 2 {
                    0 => ("0.0", "acc + e"),
                    _ => ("-1000000000.0", "(e > acc) -> { -true-> e; -false-> acc; }"),
                };
                Some(format!(
                    "{ind}({seed}, ta) -> fold {{ acc, t -> {x}[t] -> e; {step} }} -> {z}s;\n\
                     {ind}ta -> map {{ t -> {x}[t] - {z}s / {nn}.0 }} -> {z};\n"
                ))
            }
            _ => None,
        };
        if let Some(text) = made {
            s += &text;
            arrs.push(z.clone());
            cur = z;
            continue;
        }
        // The writes into the state `x`, row `it` (`it < trips ≤ 4 < n`).
        s += &match op {
            5 => {
                // Row `it`, each cell fed by its column's row sum.
                let val = |c: &str| {
                    let j = if c == "t" {
                        format!("t % {n}")
                    } else {
                        c.rsplit(" + ").next().unwrap().to_string()
                    };
                    format!("x[{c}] * 0.5 + {z}r[{j}] / {n}.0")
                };
                let cells = (0..n).map(|j| format!("it * {n} + {j}")).collect();
                rows.clone() + &write(&z, format!("(t / {n} == it)"), cells, &val)
            }
            6 => {
                // Column `(it + a) % n`, each cell fed by its row's sum.
                let val = |c: &str| {
                    let i = if c == "t" {
                        format!("t / {n}")
                    } else {
                        c.split(" * ").next().unwrap().to_string()
                    };
                    format!("x[{c}] * 0.5 + {z}r[{i}] / {n}.0")
                };
                let cells = (0..n).map(|i| format!("{i} * {n} + {z}c")).collect();
                rows.clone()
                    + &format!("{ind}(it + {a}) % {n} -> {z}c;\n")
                    + &write(&z, format!("(t % {n} == {z}c)"), cells, &val)
            }
            7 => {
                // A range from row `it`: cells `[o, o + w)` of it, into the
                // next row when `o + w > n` (`it·n + 2n < n²`: in bounds).
                let o = u32::from(a) % (n / 2);
                let w = 1 + u32::from(b) % (2 * n - o);
                let val = |c: &str| format!("x[{c}] * 0.5 + 0.25");
                let cells = (o..o + w).map(|j| format!("it * {n} + {j}")).collect();
                write(
                    &z,
                    format!("(t >= it * {n} + {o} && t < it * {n} + {})", o + w),
                    cells,
                    &val,
                )
            }
            _ => {
                // One cell of row `it`.
                let j = u32::from(a) % n;
                let val = |c: &str| format!("x[{c}] * 0.75 + 0.125");
                write(
                    &z,
                    format!("(t == it * {n} + {j})"),
                    vec![format!("it * {n} + {j}")],
                    &val,
                )
            }
        };
        cur = "x".into();
    }
    if cur != "x" {
        s += &format!("{ind}{cur} -> x;\n");
    }
    if p.two {
        // `y`'s next: itself, `w`, `x`'s new value or the trip's first array.
        let first = arrs[3.min(arrs.len() - 1)].clone();
        let ny = pick(
            &["y".to_string(), "w".to_string(), "x".to_string(), first],
            p.next,
        );
        if ny != "y" {
            s += &format!("{ind}{ny} -> y;\n");
        }
    }
    if let Some((a, b, len)) = check {
        let at = |c: i64| {
            if c < 0 {
                format!("it * {a} - {}", -c)
            } else {
                format!("it * {a} + {c}")
            }
        };
        s += &format!(
            "{ind}ta -> map {{ t -> (x[t] == xp[t]) -> {{ -true-> 0; -false-> 1; }} -> ch; \
             (t >= {} && t < {}) -> {{ -true-> 0; -false-> ch; }} }} -> dz;\n\
             {ind}(0, dz) -> fold {{ acc, v -> acc + v }} -> nd;\n{ind}\"dz=\" -> print;\n{ind}nd -> println;\n",
            at(b),
            at(b + len as i64)
        );
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

// --- recurrence-precision: row-recurrent loops, as Mapal source --------------

/// A counted loop over a row-written state `x: [f32; n·n]` and a row state
/// `o: [f32; n]` (`n` 16 or 32, 2..6 trips): trip `it` writes row `it` of `x`
/// naturally (a guarded map over the cells), its cells a closed form of the
/// cell — plus `o[t % n]` (the previous trip's row: a recurrence), or, with a
/// `prompt` P, only once `it ≥ P` (a guard on the counter). Then `ops` over
/// the trip's current matrix (from `x`) and vector (a closed form of the
/// counter): affine sites against a loop invariant `w`, per-cell matrix maps,
/// guarded row scores (`r <= it·c`: a nest cell over the counter), whole folds
/// (sum, max, min) of the vector read back per cell as a number,
/// vector·matrix nest cells, and row folds (sum, max) of the matrix. The
/// last vector is `o`'s next; the exit reads
/// `o` by one element or whole. `var` varies the row write the rungs were
/// built on: bit 0 — the prompt cell reads the loop invariant `w` (not a
/// closed form); bit 1 — the trip writes row `it + 1` (`b ≠ 0`); bits 2..3
/// zero (with a prompt) — an int division by `it − c` traps at an early
/// prompt trip `c`; bit 4 — a guard inside the written cell (`t % 3`); bit 5
/// — the trip writes the row's first half only (`len < d`); bit 6 — a second
/// row-written state `y` (two batched slots), read by `o`.
#[derive(Clone, Debug)]
pub struct RecurProg {
    pub big: bool,
    pub trips: u8,
    pub prompt: u8,
    pub ops: Vec<[u8; 2]>,
    pub whole: bool,
    pub var: u8,
}

pub fn recur_prog_strategy() -> impl Strategy<Value = RecurProg> {
    (
        any::<bool>(),
        2u8..7,
        any::<u8>(),
        prop::collection::vec(prop::array::uniform2(any::<u8>()), 2..7),
        any::<bool>(),
        any::<u8>(),
    )
        .prop_map(|(big, trips, p, ops, whole, var)| RecurProg {
            big,
            trips,
            // Half the programs have no prompt guard.
            prompt: if p % 2 == 0 { 0 } else { 1 + (p / 2) % trips },
            ops,
            whole,
            var,
        })
}

/// [`RecurProg`]'s source text.
pub fn recur_src(p: &RecurProg) -> String {
    let n: u32 = if p.big { 32 } else { 16 };
    let nn = n * n;
    let ind = "                ";
    let mut s = format!(
        "fn main() {{\n    {nn} -> iota -> ta;\n    {n} -> iota -> krange;\n    \
         ta -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen<f32> -> v; v / {}.0 }} -> w;\n    \
         (0.0, {nn}) -> fill -> x0;\n    (0.0, {n}) -> fill -> o0;\n    \
         mut x: [f32; {nn}] <- x0;\n{}    mut o: [f32; {n}] <- o0;\n    mut it: i32 <- 0;\n    loop {{\n        \
         (it < {}) -> {{\n            -true-> {{\n",
        4 * n,
        if p.var & 64 != 0 {
            format!("    (0.0, {nn}) -> fill -> y0;\n    mut y: [f32; {nn}] <- y0;\n")
        } else {
            String::new()
        },
        p.trips
    );
    let fed = |f: &str| match p.prompt {
        0 => format!("{f} / 64.0 + o[t % {n}]"),
        k => format!("(it < {k}) -> {{ -true-> {f} / 64.0; -false-> {f} / 64.0 + o[t % {n}]; }}"),
    };
    let cell = match p.var & 1 {
        0 => "(t * 7 + 13) % 101 - 50 -> widen<f32> -> f;".to_string(),
        // Reversed: not the cell's own index, so `w` is an operand (not `cell-law`).
        _ => format!("w[{} - t] * 64.0 -> f;", nn - 1),
    };
    let body = match p.var & 16 {
        0 => format!("{cell} {}", fed("f")),
        _ => format!(
            "{cell} (t % 3 == 0) -> {{ -true-> {{ {} }} -false-> f / 32.0; }}",
            fed("f")
        ),
    };
    let row = if p.var & 2 == 0 { "it" } else { "it + 1" };
    // The row's cells the trip writes: all, or (bit 5) its first half.
    let write = |body: &str, st: &str| match p.var & 32 {
        0 => format!("{{ {body} }}"),
        _ => format!(
            "{{ (t % {n} < {}) -> {{ -true-> {{ {body} }} -false-> {st}[t]; }} }}",
            n / 2
        ),
    };
    s += &format!(
        "{ind}ta -> map {{ t -> (t / {n} == {row}) -> {{ -true-> {} -false-> x[t]; }} }} -> x;\n",
        write(&body, "x")
    );
    if p.var & 64 != 0 {
        let g = format!("(t * 3 + 5) % 89 - 44 -> widen<f32> -> g; {}", fed("g"));
        s += &format!(
            "{ind}ta -> map {{ t -> (t / {n} == {row}) -> {{ -true-> {} -false-> y[t]; }} }} -> y;\n",
            write(&g, "y")
        );
    }
    let trap = (p.prompt > 0 && (p.var >> 2) & 3 == 0).then(|| (p.var >> 4) % p.prompt);
    let (dzs, dzt) = match trap {
        Some(c) => {
            s += &format!("{ind}100 / (it - {c}) -> dz;\n");
            (" dz -> to<f32> -> dq;", " + dq / 1000.0")
        }
        None => ("", ""),
    };
    s += &format!(
        "{ind}krange -> map {{ k -> ((it * {n} + k) * 5 + 3) % 97 - 48 -> widen<f32> -> qv;{dzs} qv / {}.0{dzt} }} -> q;\n",
        8 * n
    );
    let (mut mat, mut vec) = ("x".to_string(), "q".to_string());
    for (z, &[op, a]) in p.ops.iter().enumerate() {
        let z = format!("z{z}");
        s += ind;
        match op % 6 {
            0 => {
                s += &format!(
                    "ta -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, krange) -> fold {{ acc, k -> acc + {mat}[i * {n} + k] * w[k * {n} + j] }} }} -> {z};\n"
                );
                mat = z;
            }
            1 => {
                s += &if a % 2 == 0 {
                    format!(
                        "{mat} -> map {{ v -> (v > 0.0) -> {{ -true-> v; -false-> 0.0; }} }} -> {z};\n"
                    )
                } else {
                    format!("{mat} -> map {{ v -> v * 0.5 }} -> {z};\n")
                };
                mat = z;
            }
            2 => {
                let c = u32::from(a % 3) + 1;
                s += &format!(
                    "krange -> map {{ r -> (r <= it * {c}) -> {{ -true-> {{ (0.0, krange) -> fold {{ acc, k -> acc + {vec}[k] * {mat}[r * {n} + k] }} }} -false-> -1000.0; }} }} -> {z};\n"
                );
                vec = z;
            }
            3 => {
                let (seed, step) = match a % 3 {
                    0 => ("0.0", "acc + e"),
                    1 => ("-1000000.0", "(e > acc) -> { -true-> e; -false-> acc; }"),
                    _ => ("1000000.0", "(e < acc) -> { -true-> e; -false-> acc; }"),
                };
                s += &format!(
                    "({seed}, krange) -> fold {{ acc, r -> {vec}[r] -> e; {step} }} -> {z}m;\n"
                );
                s += &format!("{ind}krange -> map {{ r -> ({vec}[r] - {z}m) / 16.0 }} -> {z};\n");
                vec = z;
            }
            4 => {
                s += &format!(
                    "krange -> map {{ j -> (0.0, krange) -> fold {{ acc, r -> acc + {vec}[r] * {mat}[r * {n} + j] }} -> a; a * 0.25 }} -> {z};\n"
                );
                vec = z;
            }
            _ => {
                // A row fold of the matrix (a sum, or a running max).
                s += &if a % 2 == 0 {
                    format!(
                        "krange -> map {{ i -> (0.0, krange) -> fold {{ acc, k -> acc + {mat}[i * {n} + k] }} }} -> {z};\n"
                    )
                } else {
                    format!(
                        "krange -> map {{ i -> (-1000000.0, krange) -> fold {{ acc, k -> {mat}[i * {n} + k] -> e; (e > acc) -> {{ -true-> e; -false-> acc; }} }} }} -> {z};\n"
                    )
                };
                vec = z;
            }
        }
    }
    let last = if vec == "q" {
        format!("{mat}[j] * 0.5 + q[j]")
    } else {
        format!("{vec}[j]")
    };
    if p.var & 64 != 0 {
        // `y` read by a vector·matrix cell.
        s += &format!(
            "{ind}krange -> map {{ j -> (0.0, krange) -> fold {{ acc, r -> acc + {vec}[r] * y[r * {n} + j] }} -> a; a * 0.25 + {last} }} -> o;\n"
        );
    } else if vec == "q" {
        s += &format!("{ind}krange -> map {{ j -> {last} }} -> o;\n");
    } else {
        s += &format!("{ind}{vec} -> o;\n");
    }
    s += &format!(
        "{ind}it + 1 -> it;\n{ind}-> loop;\n            }}\n            -false-> o -> out;\n        }}\n    }}\n"
    );
    s += &format!("    out[0] -> println;\n    out[{}] -> println;\n", n - 1);
    if p.whole {
        s += "    0.0 -> zero: f32;\n    (zero, out) -> fold { acc, v -> acc + v } -> total;\n    total -> println;\n";
    }
    s + "}\n"
}

// --- recurrence-precision `radix-tile`: mixed-radix map/fold sites ----------

/// A multi-channel convolution `y[t] = Σ_k w[·]·img[·]` over constant div/mod
/// splits of the cell `t` and the tap `k`: batch `b` (1–2), `cin` (1–4) input
/// and `cout` (4/8/16) output channels, an `h × w` image (4/6/8 each side),
/// `kh × kw` taps (1–3); the output NCHW or NHWC, the weights `[co][k]` or
/// `[k][co]`, `k` channel-major or tap-major. `mix` reads the image at a
/// channel shifted by the output channel (`(ci + co) % cin`): a digit of `t`
/// read by both operands, which no tile admits. Built as source text.
#[derive(Clone, Debug)]
pub struct RadixProg {
    pub dims: [u8; 6],
    pub nhwc: bool,
    pub w_kio: bool,
    pub tap_major: bool,
    pub mix: bool,
}

pub fn radix_prog_strategy() -> impl Strategy<Value = RadixProg> {
    (
        prop::array::uniform6(any::<u8>()),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        prop::bool::weighted(0.2),
    )
        .prop_map(|(dims, nhwc, w_kio, tap_major, mix)| RadixProg {
            dims,
            nhwc,
            w_kio,
            tap_major,
            mix,
        })
}

/// [`RadixProg`]'s source text and its output length.
pub fn radix_src(p: &RadixProg) -> (String, u64) {
    let d = p.dims;
    let b = u64::from(d[0] % 2) + 1;
    // (`mix` needs two channels: `(ci + co) % 1` folds to a constant.)
    let cin = 1u64 << (d[1] % 5).max(u8::from(p.mix));
    let cout = 4u64 << (d[2] % 5);
    let (h, w) = (
        [4, 8, 6][usize::from(d[3] % 3)],
        [4, 8, 6][usize::from(d[4] % 3)],
    );
    let (kh, kw) = (u64::from(d[5] % 4) + 1, u64::from(d[5] / 4 % 4) + 1);
    let (ih, iw, kk) = (h + kh - 1, w + kw - 1, kh * kw);
    let (n, k) = (b * cout * h * w, cin * kk);
    let digits = if p.nhwc {
        format!(
            "t / {} -> b; t / {} % {h} -> i; t / {cout} % {w} -> j; t % {cout} -> co;",
            h * w * cout,
            w * cout
        )
    } else {
        format!(
            "t / {} -> b; t / {} % {cout} -> co; t / {w} % {h} -> i; t % {w} -> j;",
            cout * h * w,
            h * w
        )
    };
    let taps = if p.tap_major {
        format!("k / {cin} -> q; k % {cin} -> ci;")
    } else {
        format!("k / {kk} -> ci; k % {kk} -> q;")
    };
    let wa = if p.w_kio {
        format!("k * {cout} + co")
    } else {
        format!("co * {k} + k")
    };
    let ch = if p.mix {
        format!("(ci + co) % {cin}")
    } else {
        "ci".to_string()
    };
    let src = format!(
        "fn main() {{
    {wn} -> iota -> i_w;
    i_w -> map {{ t -> (t * 5 + 3) % 97 - 48 -> widen<f32> -> v; v / 11.0 }} -> wt;
    {imn} -> iota -> i_img;
    i_img -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0 }} -> img;
    {n} -> iota -> o_y;
    {k} -> iota -> k_y;
    o_y -> map {{ t ->
        {digits}
        (0.0, k_y) -> fold {{ acc, k -> {taps} acc + wt[{wa}] * img[((b * {cin} + {ch}) * {ih} + i + q / {kw}) * {iw} + j + q % {kw}] }}
    }} -> y;
    y[0] -> println;
    y[{last}] -> println;
    0.0 -> zero: f32;
    (zero, y) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
}}
",
        wn = cout * k,
        imn = b * cin * ih * iw,
        last = n - 1
    );
    (src, n)
}

// --- recurrence-precision `tf32-tile`: products admitting TF32 (ADR-0041) ---

/// An `m × k · k × n` product `y` whose fold's products carry a compute type
/// (`form`): 0 the step (`-> fold: tf32 | f32`), 1 the function
/// (`fn main(): tf32 | f32`), 2 the function with the step restoring `: f32`,
/// 3 none, 4 a `seq: tf32 | f32` block around the map. `bt` reads `b`
/// transposed. Beside it `mag`, the same cells' `M = Σ_k |a·b|` (ADR-0040's
/// magnitude) under `: f32`, so a value check needs no oracle of its own.
#[derive(Clone, Debug)]
pub struct Tf32Prog {
    pub dims: [u8; 3],
    pub bt: bool,
    pub form: u8,
    pub law: [u8; 2],
}

pub fn tf32_prog_strategy() -> impl Strategy<Value = Tf32Prog> {
    (
        prop::array::uniform3(any::<u8>()),
        prop::bool::weighted(0.25),
        prop::sample::select(vec![0u8, 0, 1, 1, 4, 4, 2, 3]),
        prop::array::uniform2(any::<u8>()),
    )
        .prop_map(|(dims, bt, form, law)| Tf32Prog {
            dims,
            bt,
            form,
            law,
        })
}

/// Whether [`Tf32Prog`]'s product `y` admits TF32.
pub fn tf32_admits(p: &Tf32Prog) -> bool {
    matches!(p.form, 0 | 1 | 4)
}

/// [`Tf32Prog`]'s source text and its `(m·n·k, k)`. Prints two cells of `y`,
/// each followed by its `M`.
pub fn tf32_src(p: &Tf32Prog) -> (String, (u64, u64)) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let m = pick(p.dims[0], &[16, 32, 64, 128]);
    let n = pick(p.dims[1], &[8, 16, 32, 64]);
    let k = pick(p.dims[2], &[8, 16, 32, 64, 128]);
    let (a1, a2) = (
        u64::from(p.law[0] % 13) * 2 + 3,
        u64::from(p.law[1] % 11) * 2 + 5,
    );
    let bi = if p.bt {
        format!("b[j * {k} + k]")
    } else {
        format!("b[k * {n} + j]")
    };
    let (fn_ty, step_ty) = match p.form {
        0 => ("", ": tf32 | f32"),
        1 => (": tf32 | f32", ""),
        2 => (": tf32 | f32", ": f32"),
        _ => ("", ""),
    };
    let y = format!(
        "ic -> map {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold{step_ty} {{ acc, k -> acc + a[i * {k} + k] * {bi} }} }}"
    );
    let y = if p.form == 4 {
        format!("ic -> seq: tf32 | f32 {{ {y} }}")
    } else {
        y
    };
    let (c0, c1) = (
        u64::from(p.law[0]) * 7 % (m * n),
        m * n - 1 - u64::from(p.law[1]) % n,
    );
    let src = format!(
        "fn main(){fn_ty} {{
    {mk} -> iota -> ia;
    ia -> map {{ t -> (t * {a1} + 13) % 101 - 50 -> widen<f32> -> v; v / 7.0 }} -> a;
    {kn} -> iota -> ib;
    ib -> map {{ t -> (t * {a2} + 57) % 103 - 51 -> widen<f32> -> v; v / 9.0 }} -> b;
    {mn} -> iota -> ic;
    {k} -> iota -> kr;
    {y} -> y;
    ic -> map: f32 {{ t -> t / {n} -> i; t % {n} -> j; (0.0, kr) -> fold {{ acc, k -> a[i * {k} + k] * {bi} -> p; (p > 0.0) -> {{ -true-> p; -false-> 0.0 - p; }} -> q; acc + q }} }} -> mag;
    y[{c0}] -> println;
    mag[{c0}] -> println;
    y[{c1}] -> println;
    mag[{c1}] -> println;
}}
",
        mk = m * k,
        kn = k * n,
        mn = m * n
    );
    (src, (m * n * k, k))
}
