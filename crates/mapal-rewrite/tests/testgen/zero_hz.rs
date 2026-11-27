//! Mapal-source generators: zeroing, horizontal groups and long-row folds.

use super::*;

/// kernel-merging `zero-fold`: a program whose tile sites add into a zeroed
/// output (few output cells, a long `k`: the atomic split-K candidates),
/// after device cells, in a loop trip or straight-line. `steps` is the
/// sequence: a cell (`[0, reads_o, from]`: an elementwise map of the trip
/// counter, reading the carried output `o` when `reads_o` is odd, and an
/// earlier site's output when `from` names one) or a site (`[1, q, _]`: the
/// GEMV-like fold of cell `q` against the weights). The last site's output is
/// the next carried `o`; a cell reading `o` after an earlier cell is an access
/// to the next site's bytes between them (the rule must not fold before it).
#[derive(Clone, Debug)]
pub struct ZeroProg {
    pub looped: bool,
    pub trips: u8,
    /// Output rows and columns (`rows · c` cells), the fold's length.
    pub rows: u64,
    pub c: u64,
    pub k: u64,
    pub steps: Vec<[u8; 3]>,
    /// prefill-residency: straight-line, the result reads the last site and
    /// every cell — an earlier site's output dies at the cells reading it,
    /// so a later site's output may take its bytes (the arena) and that
    /// site's zero is hoisted toward the reader's launch (a horizontal
    /// member's is its group's).
    pub drop_sites: bool,
}

pub fn zero_prog_strategy() -> impl Strategy<Value = ZeroProg> {
    (
        any::<bool>(),
        2u8..4,
        prop::sample::select(vec![1u64, 2, 4, 8]),
        prop::sample::select(vec![1u64, 8, 32, 64]),
        prop::sample::select(vec![256u64, 512, 1024]),
        prop::collection::vec(prop::array::uniform3(any::<u8>()), 2..7),
    )
        .prop_map(|(looped, trips, rows, c, k, steps)| ZeroProg {
            looped,
            trips,
            rows,
            c,
            k,
            steps,
            drop_sites: false,
        })
}

/// [`ZeroProg`]'s source text.
pub fn zero_src(p: &ZeroProg) -> String {
    let (r, c, k) = (p.rows, p.c, p.k);
    let rc = r * c;
    let mut s = format!(
        "fn main() {{\n    {kc} -> iota -> tw;\n    tw -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen<f32> -> v; v / 256.0 }} -> w;\n    {k} -> iota -> kr;\n    {rc} -> iota -> oc;\n    {rk} -> iota -> qc;\n",
        kc = k * c,
        rk = r * k
    );
    let ind = if p.looped { "                " } else { "    " };
    let mut body = String::new();
    let (mut cells, mut sites) = (0usize, 0usize);
    // Straight-line, the counter is a number the program computes (a cell
    // reading it is a kernel of its own, not a closed-form generator).
    let t = "t";
    let cell = |body: &mut String, cells: &mut usize, sites: usize, step: [u8; 3]| {
        let mut e = format!(
            "((({t} * {k} + e) * {}) + 3) % 97 - 48 -> widen<f32> -> v; v / 256.0",
            5 + step[1] % 3
        );
        if step[1] % 2 == 1 {
            e += &format!(" + o[e % {rc}]");
        }
        if sites > 0 && step[2] % 3 == 0 {
            e += &format!(" + y{}[e % {rc}] / 64.0", step[2] as usize % sites);
        }
        body.push_str(&format!("{ind}qc -> map {{ e -> {e} }} -> q{};\n", *cells));
        *cells += 1;
    };
    for &step in &p.steps {
        if step[0] % 2 == 0 {
            cell(&mut body, &mut cells, sites, step);
        } else {
            if cells == 0 {
                cell(&mut body, &mut cells, sites, [0, 0, 1]);
            }
            let q = step[1] as usize % cells;
            body.push_str(&format!(
                "{ind}oc -> map {{ e -> e / {c} -> i; e % {c} -> j; (0.0, kr) -> fold {{ acc, kk -> acc + q{q}[i * {k} + kk] * w[kk * {c} + j] }} }} -> y{sites};\n"
            ));
            sites += 1;
        }
    }
    if sites == 0 {
        body.push_str(&format!(
            "{ind}oc -> map {{ e -> e / {c} -> i; e % {c} -> j; (0.0, kr) -> fold {{ acc, kk -> acc + q0[i * {k} + kk] * w[kk * {c} + j] }} }} -> y0;\n"
        ));
        sites = 1;
    }
    // Every site's output reaches the result: the last is the next `o`, the
    // others add into it through one cell.
    // Read at shifted cells: no epilogue group takes a site's store (each
    // stays a site of its own, its output an arena member that may reuse a
    // dead operand's bytes).
    let last = sites - 1;
    let mut sum = format!("y{last}[e]");
    if p.drop_sites && !p.looped {
        for q in 0..cells {
            sum += &format!(" + q{q}[(e + {}) % {}] / 8.0", q + 1, r * k);
        }
    } else {
        for y in 0..last {
            sum += &format!(" + y{y}[(e + {}) % {rc}] / 8.0", y + 1);
        }
    }
    // One site: its output is the next state itself.
    let next = match sites {
        1 => format!("{ind}y0 -> o;\n"),
        _ => format!("{ind}oc -> map {{ e -> {sum} }} -> o;\n"),
    };
    if p.looped {
        s += &format!(
            "    (0.0, {rc}) -> fill -> o0;\n    mut o: [f32; {rc}] <- o0;\n    mut t: i32 <- 0;\n    loop {{\n        (t < {}) -> {{\n            -true-> {{\n{body}{next}{ind}t + 1 -> t;\n{ind}-> loop;\n            }}\n            -false-> o -> out;\n        }}\n    }}\n",
            p.trips
        );
    } else {
        s += &format!(
            "    (0.0, {rc}) -> fill -> o: [f32; {rc}];\n    (0, kr) -> fold {{ acc, v -> acc + v % 2 }} -> t;\n{body}    oc -> map {{ e -> {sum} }} -> out;\n"
        );
    }
    s + &format!(
        "    out[0] -> println;\n    out[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, out) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n}}\n",
        rc - 1
    )
}

/// kernel-merging `horizontal`: a program whose launches are independent
/// side by side, in a loop trip or straight-line. `steps` is the sequence: a
/// cell (`[0, salt, from]`: an elementwise map of the trip counter, reading
/// the carried `o` when `salt` is odd and an earlier site's output when
/// `from` names one — an edge that orders it after that site), a site (`[1,
/// q, _]`: the GEMV-like fold of cell `q` against the weights `w`, or `w2`
/// when `q` is odd), a row sum (`[2, y, _]`: each output row of site `y`
/// folded) — so cells reading only the counter, sites over different cells
/// and sums over different sites are unordered and adjacent. `kv` adds the
/// retained pattern in a loop (a KV cache's `x·Wk`, `x·Wv` over a state
/// written one row a trip, each site's row pass in its trip's branch).
#[derive(Clone, Debug)]
pub struct HzProg {
    pub looped: bool,
    pub trips: u8,
    /// Output rows and columns (`rows · c` cells), the fold's length.
    pub rows: u64,
    pub c: u64,
    pub k: u64,
    pub kv: bool,
    pub steps: Vec<[u8; 3]>,
    /// open-rules: sites whose map body goes on past the fold (`tile-post`:
    /// `u · 0.5 + q[i · k]`) — an epilogue group's split takes a workspace
    /// only, so its table holds no candidate a persistent kernel runs.
    pub post: bool,
}

pub fn hz_prog_strategy() -> impl Strategy<Value = HzProg> {
    (
        any::<bool>(),
        2u8..4,
        prop::sample::select(vec![4u64, 8]),
        prop::sample::select(vec![8u64, 32, 64]),
        prop::sample::select(vec![32u64, 64, 256]),
        any::<bool>(),
        prop::collection::vec(prop::array::uniform3(any::<u8>()), 2..8),
    )
        .prop_map(|(looped, trips, rows, c, k, kv, steps)| HzProg {
            looped,
            trips,
            rows,
            c,
            k,
            kv: kv && looped,
            steps,
            post: false,
        })
}

/// kernel-merging `calls`: [`HzProg`]'s programs over a wider size range —
/// one output row or column up to eight, folds up to 1024 long — so the
/// chains of dependent launches (a cell reading a site, a site over that
/// cell, a row sum over a site) hold both unsplit sites and sites whose rule
/// candidate adds into a zeroed output (few cells, a long fold: split-K
/// atomic), beside the retained pattern's row passes. open-rules: and,
/// under `post`, sites whose body goes on past the fold (`tile-post`).
pub fn calls_prog_strategy() -> impl Strategy<Value = HzProg> {
    (
        any::<bool>(),
        2u8..4,
        prop::sample::select(vec![1u64, 4, 8]),
        prop::sample::select(vec![1u64, 8, 32, 64]),
        prop::sample::select(vec![32u64, 64, 256, 1024]),
        any::<bool>(),
        prop::collection::vec(prop::array::uniform3(any::<u8>()), 2..8),
        any::<bool>(),
    )
        .prop_map(|(looped, trips, rows, c, k, kv, steps, post)| HzProg {
            looped,
            trips,
            rows,
            c,
            k,
            kv: kv && looped,
            steps,
            post,
        })
}

/// [`HzProg`]'s source text.
pub fn hz_src(p: &HzProg) -> String {
    let (r, c, k) = (p.rows, p.c, p.k);
    let rc = r * c;
    let mut s = format!(
        "fn main() {{\n    {kc} -> iota -> tw;\n    tw -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen<f32> -> v; v / 256.0 }} -> w;\n    tw -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen<f32> -> v; v / 256.0 }} -> w2;\n    {k} -> iota -> kr;\n    {c} -> iota -> cr;\n    {r} -> iota -> rr;\n    {rc} -> iota -> oc;\n    {rk} -> iota -> qc;\n",
        kc = k * c,
        rk = r * k
    );
    let ind = if p.looped { "                " } else { "    " };
    let mut body = String::new();
    let (mut cells, mut sites, mut sums) = (0usize, 0usize, 0usize);
    let t = "t";
    let cell = |body: &mut String, cells: &mut usize, sites: usize, step: [u8; 3]| {
        let mut e = format!(
            "((({t} * {k} + e) * {}) + 3) % 97 - 48 -> widen<f32> -> v; v / 256.0",
            5 + step[1] % 3
        );
        if step[1] % 2 == 1 {
            e += &format!(" + o[e % {rc}]");
        }
        if sites > 0 && step[2] % 3 == 0 {
            e += &format!(" + y{}[e % {rc}] / 64.0", step[2] as usize % sites);
        }
        body.push_str(&format!("{ind}qc -> map {{ e -> {e} }} -> q{};\n", *cells));
        *cells += 1;
    };
    for &step in &p.steps {
        match step[0] % 3 {
            0 => cell(&mut body, &mut cells, sites, step),
            1 => {
                if cells == 0 {
                    cell(&mut body, &mut cells, sites, [0, 0, 1]);
                }
                let q = step[1] as usize % cells;
                let wt = if step[1] % 2 == 1 { "w2" } else { "w" };
                // `tile-post`: the cell's body goes on past its fold.
                let post = match p.post && step[2] % 2 == 1 {
                    true => format!(" -> u; u * 0.5 + q{q}[i * {k}]"),
                    false => String::new(),
                };
                body.push_str(&format!(
                    "{ind}oc -> map {{ e -> e / {c} -> i; e % {c} -> j; (0.0, kr) -> fold {{ acc, kk -> acc + q{q}[i * {k} + kk] * {wt}[kk * {c} + j] }}{post} }} -> y{sites};\n"
                ));
                sites += 1;
            }
            _ if sites > 0 => {
                let y = step[1] as usize % sites;
                body.push_str(&format!(
                    "{ind}rr -> map {{ i -> (0.0, cr) -> fold {{ acc, j -> acc + y{y}[i * {c} + j] }} }} -> s{sums};\n"
                ));
                sums += 1;
            }
            _ => cell(&mut body, &mut cells, sites, step),
        }
    }
    if sites == 0 {
        if cells == 0 {
            cell(&mut body, &mut cells, sites, [0, 0, 1]);
        }
        body.push_str(&format!(
            "{ind}oc -> map {{ e -> e / {c} -> i; e % {c} -> j; (0.0, kr) -> fold {{ acc, kk -> acc + q0[i * {k} + kk] * w[kk * {c} + j] }} }} -> y0;\n"
        ));
        sites = 1;
    }
    // The retained pattern: row `t` of `x` written this trip, both sites over
    // all of `x` (their row passes the trip's), read at shifted cells.
    let mut kv = String::new();
    if p.kv {
        kv = format!(
            "{ind}qc -> map {{ e -> (e / {k} == t) -> {{ -true-> {{ (e * 7 + 13) % 101 - 50 -> widen<f32> -> f; f / 16.0 + o[e % {rc}] }} -false-> x[e]; }} }} -> x;\n\
             {ind}oc -> map {{ e -> e / {c} -> i; e % {c} -> j; (0.0, kr) -> fold {{ acc, kk -> acc + x[i * {k} + kk] * w[kk * {c} + j] }} }} -> ka;\n\
             {ind}oc -> map {{ e -> e / {c} -> i; e % {c} -> j; (0.0, kr) -> fold {{ acc, kk -> acc + x[i * {k} + kk] * w2[kk * {c} + j] }} }} -> va;\n"
        );
    }
    // Every output reaches the result, each read at a shifted cell (no
    // epilogue takes a site's store).
    let mut sum = format!("y{}[e]", sites - 1);
    for y in 0..sites - 1 {
        sum += &format!(" + y{y}[(e + {}) % {rc}] / 8.0", y + 1);
    }
    for x in 0..sums {
        sum += &format!(" + s{x}[(e + {}) % {r}] / 16.0", x + 1);
    }
    if p.kv {
        sum += &format!(" + ka[(e + 3) % {rc}] / 32.0 + va[(e + 5) % {rc}] / 32.0");
    }
    if p.looped {
        let xs = if p.kv {
            format!(
                "    (0.0, {rk}) -> fill -> x0;\n    mut x: [f32; {rk}] <- x0;\n",
                rk = r * k
            )
        } else {
            String::new()
        };
        s += &format!(
            "    (0.0, {rc}) -> fill -> o0;\n{xs}    mut o: [f32; {rc}] <- o0;\n    mut t: i32 <- 0;\n    loop {{\n        (t < {}) -> {{\n            -true-> {{\n{kv}{body}{ind}oc -> map {{ e -> {sum} }} -> o;\n{ind}t + 1 -> t;\n{ind}-> loop;\n            }}\n            -false-> o -> out;\n        }}\n    }}\n",
            p.trips
        );
    } else {
        s += &format!(
            "    (0.0, {rc}) -> fill -> o: [f32; {rc}];\n    (0, kr) -> fold {{ acc, v -> acc + v % 2 }} -> t;\n{body}    oc -> map {{ e -> {sum} }} -> out;\n"
        );
    }
    s + &format!(
        "    out[0] -> println;\n    out[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, out) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n}}\n",
        rc - 1
    )
}

/// long-rows: few rows of a long axis (`rows × len` cells) and a chain over
/// them — row folds (sum, max, min), per-cell maps of a row array and a
/// per-row value by every law `fold_combine` tells apart (`x · v`, `x ÷ v`,
/// `x ± v`, a polynomial of `x − v`, `v ÷ x`), per-row cells — the shapes a
/// long softmax, a norm or a logsumexp take. `nodes[n] = [kind, a, b, c]`
/// picks the operands among the arrays made so far; every fold and the last
/// cell print. `looped`: the chain runs in a two-trip loop over a carried
/// row array (its next value the chain's last cell).
#[derive(Clone, Debug)]
pub struct LongRow {
    pub rows: u64,
    pub len: u64,
    pub looped: bool,
    pub nodes: Vec<[u8; 4]>,
    /// A hash of `nodes` (no draw of its own: the other programs stay the
    /// stream's); a third of the programs (`mask % 3 == 0`) open with a
    /// masked softmax ([`LongRow::masked`]).
    pub mask: u8,
}

impl LongRow {
    /// The program opens with a softmax over `a0` masked to `−∞` past a
    /// per-row cut (rows wholly masked among them): the row max, `exp` of
    /// the shifted scaled score, its row sum (printed), before any loop. The
    /// other nodes are the unmasked program's.
    pub fn masked(&self) -> bool {
        self.mask % 3 == 0
    }
}

pub fn long_row_strategy() -> impl Strategy<Value = LongRow> {
    (
        prop::sample::select(vec![1u64, 2, 3, 4, 8]),
        prop::sample::select(vec![256u64, 512, 1024, 2048, 4096]),
        any::<bool>(),
        prop::collection::vec(prop::array::uniform4(any::<u8>()), 2..9),
    )
        .prop_map(|(rows, len, looped, nodes)| LongRow {
            rows,
            len,
            looped,
            mask: nodes.iter().fold(7u8, |h, n| {
                h.wrapping_mul(31).wrapping_add(n[0] ^ n[3].rotate_left(3))
            }),
            nodes,
        })
}

/// [`LongRow`]'s source text.
pub fn long_row_src(p: &LongRow) -> String {
    let (r, l) = (p.rows, p.len);
    let n = r * l;
    let ind = if p.looped { "                " } else { "    " };
    // `a0` is a tile site's output (`q · W`, `d = 8`): the rows start on the
    // device, where the folds over them run.
    let mut s = format!(
        "fn main() {{\n    {n} -> iota -> t;\n    {r} -> iota -> rr;\n    {l} -> iota -> kl;\n    8 -> iota -> kd;\n    {rq} -> iota -> tq;\n    tq -> map {{ e -> (e * 5 + 3) % 31 - 15 -> widen<f32> -> v; v / 16.0 }} -> q;\n    {lw} -> iota -> tw;\n    tw -> map {{ e -> (e * 7 + 13) % 101 - 50 -> widen<f32> -> v; v / 64.0 }} -> w;\n    t -> map {{ e -> e / {l} -> i; e % {l} -> j; (0.0, kd) -> fold {{ acc, d -> acc + q[i * 8 + d] * w[d * {l} + j] }} }} -> a0;\n    t -> map {{ e -> (e * 11 + 57) % 103 - 51 -> widen<f32> -> v; v / 64.0 }} -> b0;\n",
        rq = r * 8,
        lw = l * 8
    );
    // Row arrays (`a*`, all `n` cells, |x| ≲ 2) and per-row values (`v*`, one
    // a row; positive ones `p*`, ≥ 1).
    let mut rows_: Vec<String> = vec!["a0".into(), "b0".into()];
    let mut vals: Vec<String> = Vec::new();
    // ADR-0042: each row fold's `(value, folded array, max: true / min:
    // false)` — the shifts a softmax-like `exp` of a difference reads.
    let mut extremes: Vec<(String, String, bool)> = Vec::new();
    let mut pos: Vec<String> = Vec::new();
    let mut prints: Vec<String> = Vec::new();
    let mut body = String::new();
    if p.looped {
        rows_[0] = "x".into();
    }
    let (mut na, mut nv) = (0usize, 0usize);
    if p.masked() {
        // Row `i` keeps `cut − i · step` scores (`f32` `−1e39` is `−∞`):
        // slices wholly masked, one across the cut, rows with none.
        let cut = l * (1 + u64::from(p.mask / 3) % 3) / 4;
        let step = l / (1 + u64::from(p.mask / 9) % 4);
        s += &format!(
            "    t -> map {{ e -> e / {l} -> i; e % {l} -> j; (j + i * {step} < {cut}) -> {{ -true-> a0[e]; -false-> -1000000000000000000000000000000000000000.0; }} }} -> m0;\n    rr -> map {{ i -> (-1000000.0, kl) -> fold {{ acc, j -> m0[i * {l} + j] -> w; (w > acc) -> {{ -true-> w; -false-> acc; }} }} }} -> vm;\n    t -> map {{ e -> (m0[e] - vm[e / {l}]) / 4.0 -> exp }} -> am;\n    rr -> map {{ i -> (0.0, kl) -> fold {{ acc, j -> acc + am[i * {l} + j] }} }} -> sm;\n"
        );
    }
    let pick = |v: &[String], b: u8| v[b as usize % v.len()].clone();
    for (i, nd) in p.nodes.iter().enumerate() {
        let kind = if vals.is_empty() && nd[0] % 4 != 0 {
            0
        } else {
            nd[0] % 4
        };
        match kind {
            // A row fold: sum (scaled by the row), max or min.
            0 => {
                // Half the folds read the newest row array (the chain's tip).
                let a = match nd[1] % 2 {
                    0 => rows_.last().unwrap().clone(),
                    _ => pick(&rows_, nd[1] / 2),
                };
                let name = format!("v{nv}");
                nv += 1;
                let fold = match nd[2] % 4 {
                    0 | 3 => format!("(0.0, kl) -> fold {{ acc, j -> acc + {a}[i * {l} + j] }}"),
                    1 => {
                        extremes.push((name.clone(), a.clone(), true));
                        format!(
                            "(-1000000.0, kl) -> fold {{ acc, j -> {a}[i * {l} + j] -> w; (w > acc) -> {{ -true-> w; -false-> acc; }} }}"
                        )
                    }
                    _ => {
                        extremes.push((name.clone(), a.clone(), false));
                        format!(
                            "(1000000.0, kl) -> fold {{ acc, j -> {a}[i * {l} + j] -> w; (w < acc) -> {{ -true-> w; -false-> acc; }} }}"
                        )
                    }
                };
                body += &format!("{ind}rr -> map {{ i -> {fold} }} -> {name};\n");
                prints.push(format!("{name}[{}]", nd[3] as u64 % r));
                vals.push(name);
            }
            // A per-cell map of a row array and a per-row value.
            1 | 2 => {
                let a = pick(&rows_, nd[1]);
                let v = pick(&vals, nd[2]);
                let at = format!("{v}[e / {l}]");
                // Mostly the law forms; a quarter of the cells relu or multiply
                // rows, or take `exp` of a difference with a row's max / min.
                let form = match nd[3] {
                    0..160 => nd[3] % 6,
                    160..224 if !extremes.is_empty() => 8 + nd[3] % 3,
                    _ => 6 + nd[3] % 2,
                };
                let e = match form {
                    0 => format!("{a}[e] * {at} / 64.0"),
                    1 if !pos.is_empty() => format!("{a}[e] / {}[e / {l}]", pick(&pos, nd[2])),
                    2 => format!("{a}[e] - {at}"),
                    3 => format!("{a}[e] + {at}"),
                    4 => format!(
                        "({a}[e] - {at}) / 64.0 -> z; (z > -8.0) -> {{ -true-> 1.0 + z * (1.0 + z * 0.5); -false-> 0.0; }}"
                    ),
                    5 if !pos.is_empty() => {
                        format!("{}[e / {l}] / ({a}[e] * {a}[e] + 1.0)", pick(&pos, nd[2]))
                    }
                    6 => format!(
                        "({a}[e] > 0.0) -> {{ -true-> {a}[e]; -false-> 0.0 - {a}[e] / 4.0; }}"
                    ),
                    // ADR-0042: a softmax's exponential of the row's own max
                    // (min) — the rescale law's form, scaled or not — and of
                    // a difference with another row value (a rescale whose
                    // shift is no extreme of `x`: a phase in the consumer).
                    8 | 9 => {
                        let (v, src, up) = extremes[nd[2] as usize % extremes.len()].clone();
                        let d = if up {
                            format!("{src}[e] - {v}[e / {l}]")
                        } else {
                            format!("{v}[e / {l}] - {src}[e]")
                        };
                        if form == 8 {
                            format!("{d} -> exp")
                        } else {
                            format!("({d}) / 4.0 -> exp")
                        }
                    }
                    10 => format!("({a}[e] - {at}) / 256.0 -> exp"),
                    _ => format!("{a}[e] * {}[e] / 2.0", pick(&rows_, nd[2])),
                };
                // The law forms stay as written (a squash would hide the
                // law); a product of rows is squashed to stay bounded.
                let e = match form {
                    6 | 7 => format!("{e} -> y; y / (y * y / 64.0 + 1.0)"),
                    _ => e,
                };
                let name = format!("a{}", na + 1);
                na += 1;
                body += &format!("{ind}t -> map {{ e -> {e} }} -> {name};\n");
                rows_.push(name);
            }
            // A per-row cell: a positive scale from a value.
            _ => {
                let v = pick(&vals, nd[1]);
                let name = format!("p{}", pos.len());
                body += &format!(
                    "{ind}rr -> map {{ i -> 1.0 + {v}[i] * {v}[i] / 4096.0 }} -> {name};\n"
                );
                pos.push(name.clone());
                vals.push(name);
            }
        }
        let _ = i;
    }
    let last = rows_.last().unwrap().clone();
    let mut tail = format!(
        "    out[0] -> println;\n    out[{}] -> println;\n    out[{}] -> println;\n    0.0 -> zero: f32;\n    (zero, out) -> fold {{ acc, v -> acc + v }} -> total;\n    total -> println;\n",
        n / 2 + 3,
        n - 1
    );
    if p.masked() {
        tail += &format!("    sm[0] -> println;\n    sm[{}] -> println;\n", r - 1);
    }
    let prints: String = prints
        .iter()
        .map(|x| format!("    {x} -> println;\n"))
        .collect();
    if p.looped {
        // The next state reads the state (a carried array whose next value
        // reads nothing of it is another program: `long-rows.md` open item).
        let nexts =
            format!("{ind}t -> map {{ e -> x[e] / 4.0 + {last}[e] / 2.0 + a0[e] }} -> x;\n");
        s += &format!(
            "    mut x: [f32; {n}] <- b0;\n    mut it: i32 <- 0;\n    loop {{\n        (it < 2) -> {{\n            -true-> {{\n{body}{nexts}{ind}it + 1 -> it;\n{ind}-> loop;\n            }}\n            -false-> x -> out;\n        }}\n    }}\n"
        );
        s + &tail + "}\n"
    } else {
        s += &body;
        s += &format!("    {last} -> out;\n");
        s + &prints + &tail + "}\n"
    }
}
