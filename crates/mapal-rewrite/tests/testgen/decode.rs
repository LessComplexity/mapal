//! Mapal-source generators: decode, prefill, carry and block programs.

use super::*;

/// placement: a decode-like loop (model-ready `placement`) — a token's
/// layers called with a literal layer index (the second reads the first's
/// output, as a model's layers do: `x + a0 · 0`), each normalizing the hidden
/// row, projecting a key / value / query row (nested folds over
/// k-contiguous weights), rotating the key by its partner half (`i % w <
/// w/2`), appending both rows to a struct-typed loop state (`mut kv:
/// Cache`, which the rewriter carries as its leaves) in place, attending
/// over the cache's first `position + 1` rows and projecting back; the
/// logits' argmax picks the next token, whose embedding row is gathered (an
/// index the selecting fold ranges) and shifted by the token. Values are
/// small integers and each score its sign, so every sum is exact in any
/// order. Nothing prints inside the loop (the host-task ordering of a
/// printing loop's exit value is the scout's open `r2` / `p23`).
#[derive(Clone, Debug)]
pub struct DecodeProg {
    /// Hidden width `h`, row width `w`, vocabulary `v`, layers (1 or 2).
    pub h: u64,
    pub w: u64,
    pub v: u64,
    pub layers: u64,
    /// Cache positions `p`; the loop runs positions `[s0, s0 + trips)`.
    pub p: u64,
    pub s0: i64,
    pub trips: i64,
    /// The key row is rotated by its partner half.
    pub rot: bool,
    /// The cache's values row is appended before its keys row.
    pub swap: bool,
}

pub fn decode_strategy() -> impl Strategy<Value = DecodeProg> {
    (
        // 256: a lane of a warp-a-row fold walks 8 steps (`lanes-unroll`);
        // 1024: 32, over a grid of a few warps (`fly-unroll`).
        prop::sample::select(vec![32u64, 64, 256, 1024]),
        prop::sample::select(vec![4u64, 8]),
        prop::sample::select(vec![8u64, 16]),
        1u64..3,
        0i64..3,
        1i64..5,
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(|(h, w, v, layers, s0, trips, rot, swap)| DecodeProg {
            h,
            w,
            v,
            layers,
            p: (s0 + trips) as u64 + 1,
            s0,
            trips,
            rot,
            swap,
        })
}

pub fn decode_prog_src(g: &DecodeProg) -> String {
    let (h, w, v, l, p) = (g.h, g.w, g.v, g.layers, g.p);
    let (cn, half, lwh, lh) = (l * p * w, w / 2, l * w * h, l * h);
    let data = |name: &str, len: u64, a: u64, c: u64| {
        format!(
            "    {len} -> iota -> map {{\n        t -> (t * {a} + {c}) % 3 - 1 -> widen<f32>\n    }} -> {name};\n"
        )
    };
    let rotate = if g.rot {
        format!(
            "    {w} -> iota -> map {{\n        i ->\n        (i % {w} < {half}) -> {{\n            -true-> i + {half};\n            -false-> i - {half};\n        }} -> partner;\n        row[i] - row[partner]\n    }} -> ret;\n"
        )
    } else {
        format!("    {w} -> iota -> map {{\n        i -> row[i] * 2.0\n    }} -> ret;\n")
    };
    let appends = if g.swap {
        "    (cache.values, value, layer, position) -> append -> values;\n    (cache.keys, rotated, layer, position) -> append -> keys;\n"
    } else {
        "    (cache.keys, rotated, layer, position) -> append -> keys;\n    (cache.values, value, layer, position) -> append -> values;\n"
    };
    let mut s = String::new();
    s += &format!("type Cache {{\n    keys: [f32; {cn}],\n    values: [f32; {cn}]\n}}\n\n");
    s += &format!(
        "fn append(cache: [f32; {cn}], row: [f32; {w}], layer: i32, position: i32) -> [f32; {cn}] {{\n    {cn} -> iota -> map {{\n        cell ->\n        (cell / {w} == layer * {p} + position) -> {{\n            -true-> row[cell % {w}];\n            -false-> cache[cell];\n        }}\n    }} -> ret;\n}}\n\n"
    );
    s += &format!("fn rotate(row: [f32; {w}]) -> [f32; {w}] {{\n{rotate}}}\n\n");
    s += &format!(
        "fn project(weights: [f32; {lwh}], layer: i32, x: [f32; {h}]) -> [f32; {w}] {{\n    {h} -> iota -> columns;\n    {w} -> iota -> map {{\n        r -> (0.0, columns) -> fold {{\n            s, c -> s + weights[(layer * {w} + r) * {h} + c] * x[c]\n        }}\n    }} -> ret;\n}}\n\n"
    );
    s += &format!(
        "fn attend(q: [f32; {w}], keys: [f32; {cn}], values: [f32; {cn}], layer: i32, position: i32) -> [f32; {w}] {{\n    {w} -> iota -> dims;\n    {p} -> iota -> map {{\n        past ->\n        (past <= position) -> {{\n            -true-> {{\n                (0.0, dims) -> fold {{\n                    s, d -> s + q[d] * keys[(layer * {p} + past) * {w} + d]\n                }} -> dot;\n                (dot > 0.0) -> {{\n                    -true-> dot * 0.0 + 1.0;\n                    -false-> dot * 0.0 - 1.0;\n                }}\n            }}\n            -false-> 0.0;\n        }}\n    }} -> scores;\n    {p} -> iota -> pasts;\n    {w} -> iota -> map {{\n        d -> (0.0, pasts) -> fold {{\n            s, past -> s + scores[past] * values[(layer * {p} + past) * {w} + d]\n        }}\n    }} -> ret;\n}}\n\n"
    );
    s += &format!(
        "fn back(weights: [f32; {lwh}], layer: i32, a: [f32; {w}]) -> [f32; {h}] {{\n    {w} -> iota -> dims;\n    {h} -> iota -> map {{\n        i -> (0.0, dims) -> fold {{\n            s, d -> s + weights[(layer * {h} + i) * {w} + d] * a[d]\n        }}\n    }} -> ret;\n}}\n\n"
    );
    s += &format!(
        "fn layer_step(x: [f32; {h}], layer: i32, position: i32, wk: [f32; {lwh}], wv: [f32; {lwh}], wq: [f32; {lwh}], wo: [f32; {lwh}], gains: [f32; {lh}], cache: Cache) -> ([f32; {h}], Cache) {{\n    {h} -> iota -> map {{\n        i -> x[i] * gains[layer * {h} + i]\n    }} -> normed;\n    (wk, layer, normed) -> project -> key;\n    key -> rotate -> rotated;\n    (wv, layer, normed) -> project -> value;\n{appends}    (wq, layer, normed) -> project -> query;\n    (query, keys, values, layer, position) -> attend -> attended;\n    (wo, layer, attended) -> back -> out;\n    Cache {{\n        keys: keys,\n        values: values\n    }} -> next;\n    (out, next) -> ret;\n}}\n\n"
    );
    s += "fn main() {\n";
    s += &data("wk", lwh, 7, 1);
    s += &data("wv", lwh, 5, 2);
    s += &data("wq", lwh, 11, 0);
    s += &data("wo", lwh, 13, 1);
    s += &data("gains", lh, 3, 1);
    s += &data("emb", v * h, 17, 2);
    s += &format!(
        "    (0.0, {cn}) -> fill -> zeros;\n    Cache {{\n        keys: zeros,\n        values: zeros\n    }} -> empty;\n    {h} -> iota -> map {{\n        i -> emb[i]\n    }} -> x0;\n    {v} -> iota -> tokens;\n    {h} -> iota -> features;\n    mut x: [f32; {h}] <- x0;\n    mut kv: Cache <- empty;\n    mut position: i32 <- {s0};\n    loop {{\n        (position < {end}) -> {{\n            -true-> {{\n",
        s0 = g.s0,
        end = g.s0 + g.trips,
    );
    let ind = "                ";
    s += &format!(
        "{ind}(x, 0, position, wk, wv, wq, wo, gains, kv) -> layer_step -> l0;\n{ind}l0.0 -> a0;\n"
    );
    let (mut last, mut sum) = ("l0", "x[i] + a0[i]".to_string());
    if l == 2 {
        s += &format!(
            "{ind}{h} -> iota -> map {{\n{ind}    i -> x[i] + a0[i] * 0.0\n{ind}}} -> x1;\n{ind}(x1, 1, position, wk, wv, wq, wo, gains, l0.1) -> layer_step -> l1;\n{ind}l1.0 -> a1;\n"
        );
        last = "l1";
        sum += " + a1[i]";
    }
    s += &format!(
        "{ind}{h} -> iota -> map {{\n{ind}    i -> {sum}\n{ind}}} -> hidden;\n{ind}tokens -> map {{\n{ind}    t -> (0.0, features) -> fold {{\n{ind}        s, c -> s + emb[t * {h} + c] * hidden[c]\n{ind}    }}\n{ind}}} -> logits;\n{ind}((logits[0], 0), tokens) -> fold {{\n{ind}    best, t ->\n{ind}    (logits[t] > best.0) -> {{\n{ind}        -true-> (logits[t], t);\n{ind}        -false-> best;\n{ind}    }}\n{ind}}} -> winner;\n{ind}winner.1 -> token;\n{ind}token % 2 -> widen<f32> -> picked;\n{ind}{h} -> iota -> map {{\n{ind}    i -> emb[token * {h} + i] + picked\n{ind}}} -> next_x;\n{ind}next_x -> x;\n{ind}{last}.1 -> kv;\n{ind}position + 1 -> position;\n{ind}-> loop;\n            }}\n            -false-> x -> done;\n        }}\n    }}\n    done[0] -> println;\n    done[{m}] -> println;\n    done[{e}] -> println;\n}}\n",
        m = h / 2,
        e = h - 1,
    );
    s
}

/// full-model `once-range`: a prompt chunk written into a cache outside
/// every loop — `chunk` rows per layer through a range map at a captured
/// position (`t = cell ÷ w − layer·p − first`, `0 ≤ t < chunk`), attention
/// over the filled rows, then `trips` decode steps appending one row each
/// into the carried cache the prompt made (the loop's exit is the hidden
/// row, as a model's: the cache is read only through attention). `shared`:
/// the keys and values start from one zero fill (the first write's source
/// is read again by the second; else the values start from a distinct
/// fill); `reread`: an element of layer 0's keys is printed after layer 1
/// wrote over them (that write's source is not dead).
#[derive(Clone, Debug)]
pub struct PrefillProg {
    /// Hidden width `h`, row width `w`, layers (1 or 2), prompt rows `chunk`.
    pub h: u64,
    pub w: u64,
    pub layers: u64,
    pub chunk: u64,
    /// The prompt's first position; the cache holds `p` positions a layer.
    pub first: i64,
    pub p: u64,
    pub trips: i64,
    pub shared: bool,
    pub reread: bool,
}

pub fn prefill_strategy() -> impl Strategy<Value = PrefillProg> {
    (
        prop::sample::select(vec![16u64, 32, 64]),
        prop::sample::select(vec![4u64, 8]),
        1u64..3,
        1u64..5,
        0i64..3,
        0i64..3,
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(|(h, w, layers, chunk, first, trips, shared, reread)| {
            // The cache's size is no other array's, so a copy of its bytes
            // is a cache crossing: the chunk's scores (`chunk · p`) differ
            // from it by the chunk, the rest by spare positions past the last
            // (a model's context is longer than its run).
            let chunk = if chunk == layers * w {
                chunk - 1
            } else {
                chunk
            };
            let mut p = (first + chunk as i64 + trips) as u64 + 1;
            while [h, chunk * h, (chunk + 1) * h, layers * w * h, chunk * w, w]
                .contains(&(layers * p * w))
            {
                p += 1;
            }
            PrefillProg {
                h,
                w,
                layers,
                chunk,
                first,
                p,
                trips,
                shared,
                reread,
            }
        })
}

/// residency `carry-induct`: a loop whose carried array enters made on the
/// device — the product of a large site (`n × m` over `k`, a roofline the
/// device wins) — and is rewritten each trip from itself through a small
/// site (`x · w`, `n × m` over `m`, one the host's roofline would take) and
/// a map over the site and the state (`mix`: the next state reads the old
/// one too, or only the site). `second`: a second carried array the trip
/// scales alongside. `row`: each trip also gathers a host row `v` of `k`
/// (a token's embedding row, by an index the host makes), takes its sum of
/// squares (a one-cell nested fold) and a scaled copy (a norm) the next
/// state reads — `v` read on the device by the state's map too — and
/// projects a second row (a closed form of the index and the trip, `%` on
/// floats) through an `m × k` bf16 matrix (a one-row nest cell whose fold
/// walks the whole matrix). Values are small integers over powers
/// of two, exact in any order.
#[derive(Clone, Debug)]
pub struct CarryProg {
    pub n: u64,
    pub m: u64,
    pub k: u64,
    pub trips: i64,
    pub mix: bool,
    pub second: bool,
    pub row: bool,
}

pub fn carry_strategy() -> impl Strategy<Value = CarryProg> {
    (
        prop::sample::select(vec![2u64, 4, 8]),
        prop::sample::select(vec![16u64, 32, 64]),
        prop::sample::select(vec![1024u64, 2048]),
        1i64..4,
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(|(n, m, k, trips, mix, second, row)| CarryProg {
            n,
            m,
            k,
            trips,
            mix,
            second,
            row,
        })
}

/// The program [`CarryProg`] describes.
pub fn carry_prog_src(g: &CarryProg) -> String {
    let (n, m, k) = (g.n, g.m, g.k);
    let data = |name: &str, len: u64, a: u64, c: u64| {
        format!(
            "    {len} -> iota -> map {{\n        t -> (t * {a} + {c}) % 5 - 2 -> widen<f32>\n    }} -> {name};\n"
        )
    };
    let mut s = String::from("fn main() {\n");
    s += &data("a", n * k, 7, 1);
    s += &data("b", k * m, 5, 2);
    s += &data("w", m * m, 3, 0);
    s += &format!("    {k} -> iota -> inner;\n    {m} -> iota -> cols;\n");
    if g.row {
        s += &data("emb", 4 * k, 11, 3);
        // A bf16 matrix, as a checkpoint stores it: its one-row projection is
        // a nest cell (a warp a row), not a strided site.
        s += &data("proj", m * k, 13, 4).replace(
            "widen<f32>\n    } -> proj;",
            "widen<f32> -> to<bf16>\n    } -> proj;",
        );
    }
    s += &format!(
        "    {nm} -> iota -> map {{\n        c ->\n        c / {m} -> i;\n        c % {m} -> j;\n        (0.0, inner) -> fold {{\n            s, q -> s + a[i * {k} + q] * b[q * {m} + j]\n        }}\n    }} -> start;\n",
        nm = n * m
    );
    s += &format!(
        "    {nm} -> iota -> map {{\n        c -> (c % 7 - 3) -> widen<f32>\n    }} -> other0;\n",
        nm = n * m
    );
    s += "    mut x: [f32; N] <- start;\n"
        .replace("N", &(n * m).to_string())
        .as_str();
    if g.second {
        s += &format!("    mut y: [f32; {}] <- other0;\n", n * m);
    }
    s += "    mut trip: i32 <- 0;\n    loop {\n";
    s += &format!(
        "        (trip < {}) -> {{\n            -true-> {{\n",
        g.trips
    );
    s += &format!(
        "                {nm} -> iota -> map {{\n                    c ->\n                    c / {m} -> i;\n                    c % {m} -> j;\n                    (0.0, cols) -> fold {{\n                        s, q -> s + x[i * {m} + q] * w[q * {m} + j]\n                    }}\n                }} -> t;\n",
        nm = n * m
    );
    if g.row {
        s += &format!(
            "                [trip % 4] -> ids;\n                {k} -> iota -> map {{\n                    c -> emb[ids[0] * {k} + c]\n                }} -> v;\n                1 -> iota -> map {{\n                    r -> (0.0, inner) -> fold {{\n                        s, c -> s + v[r * {k} + c] * v[r * {k} + c]\n                    }}\n                }} -> squares;\n                {k} -> iota -> map {{\n                    c -> v[c] * 0.5 + squares[0] * 0.0625\n                }} -> scaled;\n                {k} -> iota -> map {{\n                    c -> c * 3 + trip -> widen<f32> -> f;\n                    f % 5.0 - 2.0\n                }} -> other;\n                {m} -> iota -> map {{\n                    r -> (0.0, inner) -> fold {{\n                        s, c ->\n                        proj[r * {k} + c] -> widen<f32> -> weight;\n                        s + weight * other[c]\n                    }}\n                }} -> u;\n"
        );
    }
    let next = match (g.mix, g.second) {
        (true, true) => "t[c] * 0.125 + x[c] * 0.5 + y[c] * 0.25",
        (true, false) => "t[c] * 0.125 + x[c] * 0.5",
        (false, true) => "t[c] * 0.125 + y[c] * 0.25",
        (false, false) => "t[c] * 0.125",
    };
    let next = match g.row {
        true => format!("{next} + u[c % {m}] * 0.0078125 + v[c] * 0.25 + scaled[c] * 0.5"),
        false => next.to_string(),
    };
    s += &format!(
        "                {nm} -> iota -> map {{\n                    c -> {next}\n                }} -> next;\n                next -> x;\n",
        nm = n * m
    );
    if g.second {
        s += &format!(
            "                {nm} -> iota -> map {{\n                    c -> y[c] * 0.5 + t[c] * 0.25\n                }} -> y2;\n                y2 -> y;\n",
            nm = n * m
        );
    }
    s += "                trip + 1 -> trip;\n                -> loop;\n            }\n            -false-> x -> done;\n        }\n    }\n";
    s += &format!(
        "    done[0] -> println;\n    done[{}] -> println;\n",
        n * m - 1
    );
    s += "}\n";
    s
}

/// The program [`PrefillProg`] describes.
pub fn prefill_prog_src(g: &PrefillProg) -> String {
    let (h, w, l, p, chunk) = (g.h, g.w, g.layers, g.p, g.chunk);
    let (cn, lwh) = (l * p * w, l * w * h);
    let data = |name: &str, len: u64, a: u64, c: u64| {
        format!(
            "    {len} -> iota -> map {{\n        t -> (t * {a} + {c}) % 5 - 2 -> widen<f32>\n    }} -> {name};\n"
        )
    };
    let mut s = String::new();
    s += &format!("type Cache {{\n    keys: [f32; {cn}],\n    values: [f32; {cn}],\n}}\n\n");
    s += &format!(
        "fn write_rows<tokens>(cache: [f32; {cn}], rows: [f32; tokens * {w}], layer: i32, first: i32) -> [f32; {cn}] {{\n    {cn} -> iota -> map {{\n        cell ->\n        cell / {w} - layer * {p} - first -> t;\n        (t >= 0 && t < tokens) -> {{\n            -true-> rows[t * {w} + cell % {w}];\n            -false-> cache[cell];\n        }}\n    }} -> ret;\n}}\n\n"
    );
    s += &format!(
        "fn project<tokens>(weights: [f32; {lwh}], layer: i32, x: [f32; tokens * {h}]) -> [f32; tokens * {w}] {{\n    {h} -> iota -> columns;\n    tokens * {w} -> iota -> map {{\n        cell ->\n        cell / {w} -> t;\n        cell % {w} -> r;\n        (0.0, columns) -> fold {{\n            s, c -> s + weights[(layer * {w} + r) * {h} + c] * x[t * {h} + c]\n        }}\n    }} -> ret;\n}}\n\n"
    );
    s += &format!(
        "fn back<tokens>(weights: [f32; {lwh}], layer: i32, a: [f32; tokens * {w}]) -> [f32; tokens * {h}] {{\n    {w} -> iota -> dims;\n    tokens * {h} -> iota -> map {{\n        cell ->\n        cell / {h} -> t;\n        cell % {h} -> i;\n        (0.0, dims) -> fold {{\n            s, d -> s + weights[(layer * {h} + i) * {w} + d] * a[t * {w} + d]\n        }}\n    }} -> ret;\n}}\n\n"
    );
    s += &format!(
        "fn attend<tokens>(q: [f32; tokens * {w}], keys: [f32; {cn}], values: [f32; {cn}], layer: i32, first: i32) -> [f32; tokens * {w}] {{\n    {w} -> iota -> dims;\n    {p} -> iota -> pasts;\n    tokens * {p} -> iota -> map {{\n        cell ->\n        cell / {p} -> t;\n        cell % {p} -> past;\n        (past <= first + t) -> {{\n            -true-> {{\n                (0.0, dims) -> fold {{\n                    s, d -> s + q[t * {w} + d] * keys[(layer * {p} + past) * {w} + d]\n                }} -> dot;\n                (dot > 0.0) -> {{\n                    -true-> dot * 0.0 + 1.0;\n                    -false-> dot * 0.0 - 1.0;\n                }}\n            }}\n            -false-> 0.0;\n        }}\n    }} -> scores;\n    tokens * {w} -> iota -> map {{\n        cell ->\n        cell / {w} -> t;\n        cell % {w} -> d;\n        (0.0, pasts) -> fold {{\n            s, past -> s + scores[t * {p} + past] * values[(layer * {p} + past) * {w} + d]\n        }}\n    }} -> ret;\n}}\n\n"
    );
    s += &format!(
        "fn layer_step<tokens>(x: [f32; tokens * {h}], layer: i32, first: i32, wk: [f32; {lwh}], wv: [f32; {lwh}], wq: [f32; {lwh}], wo: [f32; {lwh}], cache: Cache) -> ([f32; tokens * {h}], Cache) {{\n    (wk, layer, x) -> project -> key: [f32; tokens * {w}];\n    (wv, layer, x) -> project -> value: [f32; tokens * {w}];\n    (cache.keys, key, layer, first) -> write_rows -> keys;\n    (cache.values, value, layer, first) -> write_rows -> values;\n    (wq, layer, x) -> project -> query: [f32; tokens * {w}];\n    (query, keys, values, layer, first) -> attend -> attended;\n    (wo, layer, attended) -> back -> out: [f32; tokens * {h}];\n    tokens * {h} -> iota -> map {{\n        i -> x[i] + out[i]\n    }} -> after;\n    Cache {{\n        keys: keys,\n        values: values,\n    }} -> next;\n    (after, next) -> ret;\n}}\n\n"
    );
    s += "fn main() {\n";
    s += &data("wk", lwh, 7, 1);
    s += &data("wv", lwh, 5, 2);
    s += &data("wq", lwh, 11, 0);
    s += &data("wo", lwh, 13, 1);
    s += &data("emb", (chunk + 1) * h, 17, 2);
    s += &format!("    (0.0, {cn}) -> fill -> zeros;\n");
    let values0 = if g.shared {
        "zeros"
    } else {
        // A distinct start (an equal fill is the same value: one object),
        // closed-form in the cell index.
        s += &format!(
            "    {cn} -> iota -> map {{\n        i -> i % 3 - 1 -> widen<f32>\n    }} -> start;\n"
        );
        "start"
    };
    s += &format!(
        "    Cache {{\n        keys: zeros,\n        values: {values0},\n    }} -> empty;\n    {n} -> iota -> map {{\n        i -> emb[i]\n    }} -> prompt;\n    (prompt, 0, {first}, wk, wv, wq, wo, empty) -> layer_step -> l0;\n",
        n = chunk * h,
        first = g.first,
    );
    let mut last = "l0";
    if l == 2 {
        s += &format!(
            "    (l0.0, 1, {first}, wk, wv, wq, wo, l0.1) -> layer_step -> l1;\n",
            first = g.first
        );
        last = "l1";
        if g.reread {
            s += &format!("    l0.1.keys[{k}] -> println;\n", k = g.first as u64 * w);
        }
    }
    s += &format!("    {last}.0[{e}] -> println;\n", e = chunk * h - 1);
    let end = g.first + chunk as i64 + g.trips;
    s += &format!(
        "    {h} -> iota -> map {{\n        i -> emb[{chunk} * {h} + i]\n    }} -> x0;\n    mut x: [f32; {h}] <- x0;\n    mut kv: Cache <- {last}.1;\n    mut position: i32 <- {start};\n    loop {{\n        (position < {end}) -> {{\n            -true-> {{\n                (x, 0, position, wk, wv, wq, wo, kv) -> layer_step -> d0;\n",
        start = g.first + chunk as i64,
    );
    let mut dl = "d0";
    if l == 2 {
        s += "                (d0.0, 1, position, wk, wv, wq, wo, d0.1) -> layer_step -> d1;\n";
        dl = "d1";
    }
    s += &format!(
        "                {dl}.0 -> x;\n                {dl}.1 -> kv;\n                position + 1 -> position;\n                -> loop;\n            }}\n            -false-> x -> done;\n        }}\n    }}\n    done[0] -> println;\n    done[{m}] -> println;\n}}\n",
        m = h - 1,
    );
    s
}

/// bound-blkdec (the composed bound, ADR-0041 D5 `bound(o)`): a decode block
/// over `steps` trips — an optional RMS norm (a sum of squares and a Newton
/// square root), the fused QKV projection over bf16 weights, the step's key
/// and value rounded into a bf16 cache (`to<bf16>` of values that carry the
/// default face's deviation), causal scores over the cache, a running max,
/// `exp`, the sum, `P·V ÷ sum` (an epilogue the stored rounding ends:
/// `round-post`), the output projection with its residual and an optional
/// ReLU FFN (`ffn` its width, 0 none): the paths `round-store`,
/// `round-post` and a bf16 cache take. Inputs by closed integer laws (`law`
/// shifts them).
#[derive(Clone, Debug)]
pub struct BlockProg {
    pub rows: u64,
    pub d: u64,
    pub len: u64,
    pub steps: u64,
    pub norm: bool,
    pub ffn: u64,
    pub law: u64,
}

/// [`BlockProg`]s at the interpreter's scale.
pub fn block_strategy() -> impl Strategy<Value = BlockProg> {
    (
        prop::sample::select(vec![1u64, 2]),
        prop::sample::select(vec![8u64, 16, 32]),
        prop::sample::select(vec![4u64, 8, 16]),
        1u64..4,
        any::<bool>(),
        prop::sample::select(vec![0u64, 8, 24]),
        0u64..4,
    )
        .prop_map(|(rows, d, len, steps, norm, ffn, law)| BlockProg {
            rows,
            d,
            len,
            steps,
            norm,
            ffn,
            law,
        })
}

/// [`BlockProg`]'s source; `member`: every fold `: bf16 | f32` (else f32
/// products: the same exact face, no permission to round).
pub fn block_src(p: &BlockProg, member: bool) -> String {
    let (r, d, l, f) = (p.rows, p.d, p.len, p.ffn);
    let q = 3 * d;
    let k = if member { ": bf16 | f32 " } else { "" };
    let law = |n: u64, m: u64, a: u64, div: &str, narrow: bool| {
        let h = (m - 1) / 2;
        format!(
            "    {n} -> iota -> map {{\n        t -> (t % {m} * {s} + {a}) % {m} - {h} -> widen<f32> -> v;\n        v / {div}{tail}\n    }}",
            s = 7 + 2 * p.law,
            a = a + 5 * p.law,
            tail = if narrow { " -> to<bf16>" } else { "" },
        )
    };
    let (w, wf) = (format!("{}.0", 8 * d), format!("{}.0", 8 * f.max(1)));
    let mut s = String::from("fn main() {\n");
    s += &format!("{} -> x0;\n", law(r * d, 101, 13, "29.0", false));
    s += &format!("{} -> wqkv;\n", law(d * q, 103, 57, &w, true));
    s += &format!("{} -> wo;\n", law(d * d, 97, 3, &w, true));
    s += &format!("{} -> kc0;\n", law(r * l * d, 109, 29, "29.0", true));
    s += &format!("{} -> vc0;\n", law(r * l * d, 89, 1, "29.0", true));
    if f > 0 {
        s += &format!("{} -> wg;\n", law(d * f, 107, 7, &w, true));
        s += &format!("{} -> wd;\n", law(f * d, 97, 41, &wf, true));
    }
    for (n, name) in [
        (r, "rR"),
        (d, "rD"),
        (r * d, "rRD"),
        (r * q, "rRQ"),
        (r * l, "rRL"),
        (l, "rL"),
        (r * l * d, "rC"),
        (8, "rN"),
        (r * f.max(1), "rRF"),
        (f.max(1), "rF"),
    ] {
        s += &format!("    {n} -> iota -> {name};\n");
    }
    s += &format!(
        "    mut kc: [bf16; {c}] <- kc0;\n    mut vc: [bf16; {c}] <- vc0;\n    mut x: [f32; {rd}] <- x0;\n    mut t: i32 <- 0;\n    loop {{\n        (t < {steps}) -> {{\n            -true-> {{\n                {p0} + t -> p;\n",
        c = r * l * d,
        rd = r * d,
        steps = p.steps,
        p0 = l - p.steps,
    );
    let b = "                ";
    if p.norm {
        s += &format!(
            "{b}rR -> map {{\n{b}    b -> (0.0, rD) -> fold{k}{{\n{b}        acc, i -> x[b * {d} + i] -> v;\n{b}        acc + v * v\n{b}    }} -> ss;\n{b}    ss / {d}.0 + 0.01 -> ra;\n{b}    (0.5 + 0.5 * ra, rN) -> fold{k}{{\n{b}        acc, n -> 0.5 * (acc + ra / acc)\n{b}    }} -> sd;\n{b}    1.0 / sd\n{b}}} -> rinv;\n{b}rRD -> map {{\n{b}    e -> x[e] * rinv[e / {d}]\n{b}}} -> xn;\n"
        );
    } else {
        s += &format!("{b}rRD -> map {{\n{b}    e -> x[e] * 0.75\n{b}}} -> xn;\n");
    }
    s += &format!(
        "{b}rRQ -> map {{\n{b}    e -> e / {q} -> b;\n{b}    e % {q} -> j;\n{b}    (0.0, rD) -> fold{k}{{\n{b}        acc, c -> wqkv[c * {q} + j] -> widen<f32> -> w;\n{b}        acc + xn[b * {d} + c] * w\n{b}    }}\n{b}}} -> qkv;\n"
    );
    for (cache, off) in [("kc", d), ("vc", 2 * d)] {
        s += &format!(
            "{b}rC -> map {{\n{b}    c -> c / {ld} -> b;\n{b}    c / {d} % {l} -> r;\n{b}    (r == p) -> {{\n{b}        -true-> qkv[b * {q} + {off} + c % {d}] -> to<bf16>;\n{b}        -false-> {cache}[c];\n{b}    }}\n{b}}} -> {cache};\n",
            ld = l * d,
        );
    }
    s += &format!(
        "{b}rRL -> map {{\n{b}    e -> e / {l} -> b;\n{b}    e % {l} -> r;\n{b}    (r <= p) -> {{\n{b}        -true-> {{\n{b}            (0.0, rD) -> fold{k}{{\n{b}                acc, i -> kc[(b * {l} + r) * {d} + i] -> widen<f32> -> kv;\n{b}                acc + qkv[b * {q} + i] * kv\n{b}            }} -> dt;\n{b}            dt / {sc}\n{b}        }}\n{b}        -false-> -1000000000.0;\n{b}    }}\n{b}}} -> s;\n",
        sc = if d >= 16 { "4.0" } else { "2.0" },
    );
    s += &format!(
        "{b}rR -> map {{\n{b}    b -> (-1000000000.0, rL) -> fold{k}{{\n{b}        acc, r -> s[b * {l} + r] -> se;\n{b}        (se > acc) -> {{\n{b}            -true-> se;\n{b}            -false-> acc;\n{b}        }}\n{b}    }}\n{b}}} -> mx;\n{b}rRL -> map {{\n{b}    e -> s[e] - mx[e / {l}] -> exp\n{b}}} -> pe;\n{b}rR -> map {{\n{b}    b -> (0.0, rL) -> fold{k}{{\n{b}        acc, r -> acc + pe[b * {l} + r]\n{b}    }}\n{b}}} -> den;\n"
    );
    s += &format!(
        "{b}rRD -> map {{\n{b}    e -> e / {d} -> b;\n{b}    e % {d} -> i;\n{b}    (0.0, rL) -> fold{k}{{\n{b}        acc, r -> vc[(b * {l} + r) * {d} + i] -> widen<f32> -> vv;\n{b}        acc + pe[b * {l} + r] * vv\n{b}    }} -> av;\n{b}    av / den[b]\n{b}}} -> att;\n{b}rRD -> map {{\n{b}    e -> e / {d} -> b;\n{b}    e % {d} -> j;\n{b}    (0.0, rD) -> fold{k}{{\n{b}        acc, c -> wo[c * {d} + j] -> widen<f32> -> w;\n{b}        acc + att[b * {d} + c] * w\n{b}    }} -> y;\n{b}    x[e] + y\n{b}}} -> {h};\n",
        h = if f > 0 { "h" } else { "x" },
    );
    if f > 0 {
        s += &format!(
            "{b}rRF -> map {{\n{b}    e -> e / {f} -> b;\n{b}    e % {f} -> j;\n{b}    (0.0, rD) -> fold{k}{{\n{b}        acc, c -> wg[c * {f} + j] -> widen<f32> -> w;\n{b}        acc + h[b * {d} + c] * w\n{b}    }} -> g;\n{b}    (g > 0.0) -> {{\n{b}        -true-> g;\n{b}        -false-> 0.0;\n{b}    }}\n{b}}} -> hh;\n{b}rRD -> map {{\n{b}    e -> e / {d} -> b;\n{b}    e % {d} -> j;\n{b}    (0.0, rF) -> fold{k}{{\n{b}        acc, c -> wd[c * {d} + j] -> widen<f32> -> w;\n{b}        acc + hh[b * {f} + c] * w\n{b}    }} -> y2;\n{b}    h[e] + y2\n{b}}} -> x;\n"
        );
    }
    s += &format!(
        "{b}t + 1 -> t;\n{b}-> loop;\n            }}\n            -false-> x -> out;\n        }}\n    }}\n    out[0] -> println;\n    out[{last}] -> println;\n    0.0 -> zero: f32;\n    (zero, out) -> fold{k}{{\n        acc, v -> acc + v * v\n    }} -> ssq;\n    ssq -> println;\n}}\n",
        last = r * d - 1,
    );
    formatted(s)
}
