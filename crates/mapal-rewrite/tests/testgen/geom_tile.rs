//! Mapal-source generators: launch geometry of tile sites and tile sites at bench sizes.

use super::*;

// --- site-geometry: launch geometry of tile sites (`fill-wave`, `row-split`) --

/// An `m × k · k × n` product `y` with integer-valued operands — every partial
/// sum is exact in f32, so every cut of the fold prints the same bits — from a
/// skinny projection (`m` 1) to a small GEMM: `m` 1–64, `n` 32–512, `k`
/// 32–1024; `bt` reads `b` transposed. `retained`: the product runs in a loop
/// over a carried `x` (its `a`, `m ≥ 4` rows) whose trip `t` rewrites row `t`
/// and reads row `t` of `y` (decode's K/V rebuild: the retained one-row site).
/// `wide` (not retained): a few rows over many lanes — `m` 16–64, `n`
/// 16384–32768, `k` 32–64 (a convolution's implicit GEMM shape).
#[derive(Clone, Debug)]
pub struct GeomProg {
    pub dims: [u8; 3],
    pub bt: bool,
    pub retained: bool,
    pub law: [u8; 2],
    pub wide: bool,
    /// Operands `v / 7.0`, `v / 9.0`: partial sums round, so a cell whose `k`
    /// order moved prints other bits ([`order_prog_strategy`]).
    pub frac: bool,
    /// `f64` operands and products.
    pub f64: bool,
    /// `a` read along `i` (`a[q·m + i]`, `ci == 1`), not along `k`.
    pub ci: bool,
    /// `k = 8·odd` (40, 72, 136, 264): no power-of-two panel ≥ 16 divides it.
    pub k8: bool,
}

pub fn geom_prog_strategy() -> impl Strategy<Value = GeomProg> {
    (
        prop::array::uniform3(any::<u8>()),
        prop::bool::weighted(0.25),
        any::<bool>(),
        prop::array::uniform2(any::<u8>()),
        prop::bool::weighted(0.3),
    )
        .prop_map(|(dims, bt, retained, law, wide)| GeomProg {
            dims,
            bt,
            retained,
            law,
            wide: wide && !retained,
            frac: false,
            f64: false,
            ci: false,
            k8: false,
        })
}

/// [`geom_prog_strategy`] for order and layout oracles: non-integer operands
/// (`frac`) and, at random, `f64`, `ci == 1` and `k = 8·odd` (neither
/// retained nor wide: the interpreter is the oracle).
pub fn order_prog_strategy() -> impl Strategy<Value = GeomProg> {
    (
        geom_prog_strategy(),
        prop::bool::weighted(0.5),
        prop::bool::weighted(0.25),
        prop::bool::weighted(0.25),
    )
        .prop_map(|(p, f64, ci, k8)| GeomProg {
            retained: false,
            wide: false,
            frac: true,
            f64,
            ci,
            k8,
            ..p
        })
}

/// [`GeomProg`]'s source text and its `(m, n, k)`.
pub fn geom_src(p: &GeomProg) -> (String, (u64, u64, u64)) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let m = pick(
        p.dims[0],
        if p.retained {
            &[4, 8, 16, 64]
        } else if p.wide {
            &[16, 32, 64]
        } else {
            &[1, 2, 4, 16, 64]
        },
    );
    let n = pick(
        p.dims[1],
        if p.wide {
            &[16384, 32768]
        } else {
            &[32, 64, 128, 256, 512]
        },
    );
    let k = pick(
        p.dims[2],
        if p.wide {
            &[32, 64]
        } else if p.k8 {
            &[40, 72, 136, 264]
        } else {
            &[32, 64, 256, 1024]
        },
    );
    geom_src_at(p, m, n, k)
}

/// tile-efficiency T1: [`GeomProg`] at mid extents — rows 64–256 (192 not a
/// power of two), lanes 256–512, `k` 512–2048 — where a full register tile
/// divides the site and the fold is deep enough to cut, still small enough
/// for the emulator. `retained` and `wide` are ignored.
pub fn mid_src(p: &GeomProg) -> (String, (u64, u64, u64)) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let q = GeomProg {
        retained: false,
        wide: false,
        ..p.clone()
    };
    let m = pick(p.dims[0], &[64, 128, 192, 256]);
    let n = pick(p.dims[1], &[256, 512]);
    let k = pick(p.dims[2], &[512, 1024, 2048]);
    geom_src_at(&q, m, n, k)
}

/// [`geom_src`] at the extents `(m, n, k)`.
pub(super) fn geom_src_at(p: &GeomProg, m: u64, n: u64, k: u64) -> (String, (u64, u64, u64)) {
    let (a1, a2) = (
        u64::from(p.law[0] % 13) * 2 + 3,
        u64::from(p.law[1] % 11) * 2 + 5,
    );
    let bi = if p.bt {
        format!("b[j * {k} + q]")
    } else {
        format!("b[q * {n} + j]")
    };
    let ai = match p.ci && !p.retained {
        true => format!("q * {m} + i"),
        false => format!("i * {k} + q"),
    };
    let y = |a: &str| {
        format!(
            "ic -> map {{ c -> c / {n} -> i; c % {n} -> j; (0.0, kr) -> fold {{ acc, q -> acc + {a}[{ai}] * {bi} }} }} -> y;"
        )
    };
    let (da, db) = match p.frac {
        true => (" -> v; v / 7.0", " -> v; v / 9.0"),
        false => ("", ""),
    };
    let head = format!(
        "fn main() {{
    {mk} -> iota -> ia;
    ia -> map {{ t -> (t * {a1} + 13) % 101 - 50 -> widen<f32>{da} }} -> a;
    {kn} -> iota -> ib;
    ib -> map {{ t -> (t * {a2} + 57) % 103 - 51 -> widen<f32>{db} }} -> b;
    {mn} -> iota -> ic;
    {k} -> iota -> kr;
",
        mk = m * k,
        kn = k * n,
        mn = m * n
    );
    let body = if p.retained {
        format!(
            "    {n} -> iota -> nr;
    (0.0, {n}) -> fill -> o0;
    mut x: [f32; {mk}] <- a;
    mut o: [f32; {n}] <- o0;
    mut t: i32 <- 0;
    loop {{
        (t < {trips}) -> {{
            -true-> {{
                ia -> map {{ c -> (c / {k} == t) -> {{ -true-> (c * 3 + t * 5) % 23 - 11 -> widen<f32>; -false-> x[c]; }} }} -> x;
                {y}
                nr -> map {{ j -> y[t * {n} + j] }} -> o;
                t + 1 -> t;
                -> loop;
            }}
            -false-> o -> out;
        }}
    }}
    out[0] -> println;
    out[{last}] -> println;
    0.0 -> zero: f32;
    (zero, out) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
}}
",
            mk = m * k,
            trips = m.min(4),
            y = y("x"),
            last = n - 1
        )
    } else {
        // site-geometry integrate: a third of the products feed a bias + relu
        // epilogue (the tile kernel's epilogue group: an ordered launch beside
        // a cut fold's chooser).
        let epi = if p.law[1] % 3 == 0 {
            format!(
                "\n    {n} -> iota -> nb;\n    nb -> map {{ t -> t % 7 - 3 -> widen<f32> }} -> bias;\n    \
                 ic -> map {{ c -> y[c] + bias[c % {n}] -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> r; r }} -> z;"
            )
        } else {
            String::new()
        };
        let o = if epi.is_empty() { "y" } else { "z" };
        format!(
            "    {y}{epi}
    {o}[0] -> println;
    {o}[{last}] -> println;
    0.0 -> zero: f32;
    (zero, {o}) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
}}
",
            y = y("a"),
            last = m * n - 1
        )
    };
    let src = head + &body;
    match p.f64 {
        true => (src.replace("f32", "f64"), (m, n, k)),
        false => (src, (m, n, k)),
    }
}

// --- plan-cost blind set: tile sites at bench sizes --------------------------

/// A tile-site program at bench sizes: one `m × k · k × n` product (`kind` 0,
/// optionally a bias + relu epilogue), a chain of two or three (1, 2: each
/// product's `n` is the next one's `k`), or a sum of products — two over one `k`
/// range (3), three sharing `a` (4), two over two `k` ranges (5) — between a
/// `time` pair (the `benches/algorithms/pt_*` timer). Operands are integers in
/// ±3; `bt` reads each `b` transposed.
#[derive(Clone, Debug)]
pub struct TileProg {
    pub kind: u8,
    pub dims: [u8; 6],
    pub bt: bool,
    pub epi: bool,
    pub law: u8,
}

pub fn tile_prog_strategy() -> impl Strategy<Value = TileProg> {
    (
        0u8..6,
        prop::array::uniform6(any::<u8>()),
        prop::bool::weighted(0.25),
        prop::bool::weighted(0.3),
        any::<u8>(),
    )
        .prop_map(|(kind, dims, bt, epi, law)| TileProg {
            kind,
            dims,
            bt,
            epi,
            law,
        })
}

/// [`TileProg`]'s source and its products' `(m, n, k)`; `small` maps every
/// extent above 1 to 3–9 (same program) for the interpreter.
pub fn tile_src(p: &TileProg, small: bool) -> (String, Vec<(u64, u64, u64)>) {
    const M: &[u64] = &[1, 1, 16, 16, 7, 100, 333, 512, 1000, 1024, 2048, 3001, 4096];
    const N: &[u64] = &[
        64, 96, 250, 512, 777, 1024, 2048, 3000, 4096, 5120, 11008, 18432,
    ];
    const K: &[u64] = &[64, 100, 256, 513, 1024, 2048, 4096, 5000, 8192];
    let ext = |v: &[u64], d: u8| match v[usize::from(d) % v.len()] {
        x if small && x > 1 => x % 7 + 3,
        x => x,
    };
    let (m, k, k2) = (ext(M, p.dims[0]), ext(K, p.dims[1]), ext(K, p.dims[5]));
    let n = [ext(N, p.dims[2]), ext(N, p.dims[3]), ext(N, p.dims[4])];
    // (out, a, b, k, n) per product.
    let prods: Vec<(&str, &str, &str, u64, u64)> = match p.kind {
        0 => vec![("y", "a", "b0", k, n[0])],
        1 | 2 => [
            ("y1", "a", "b0", k, n[0]),
            ("y2", "y1", "b1", n[0], n[1]),
            ("y3", "y2", "b2", n[1], n[2]),
        ][..usize::from(p.kind) + 1]
            .to_vec(),
        3 => vec![("y1", "a", "b0", k, n[0]), ("y2", "a2", "b1", k, n[0])],
        4 => vec![
            ("y1", "a", "b0", k, n[0]),
            ("y2", "a", "b1", k, n[0]),
            ("y3", "a", "b2", k, n[0]),
        ],
        _ => vec![("y1", "a", "b0", k, n[0]), ("y2", "a2", "b1", k2, n[0])],
    };
    let mut iotas: Vec<u64> = Vec::new();
    let mut laws = Laws::default();
    let mut made: Vec<&str> = Vec::new();
    let mut body = String::new();
    for &(y, a, b, kk, nn) in &prods {
        if a.starts_with('a') && !made.contains(&a) {
            made.push(a);
            tile_law(&mut laws, p.law, a, tile_iota(&mut iotas, m * kk));
        }
        tile_law(&mut laws, p.law, b, tile_iota(&mut iotas, kk * nn));
        let bi = if p.bt {
            format!("{b}[j * {kk} + q]")
        } else {
            format!("{b}[q * {nn} + j]")
        };
        body += &format!(
            "    {ic} -> map {{ c -> c / {nn} -> i; c % {nn} -> j; (0.0, {kr}) -> fold {{ acc, q -> acc + {a}[i * {kk} + q] * {bi} }} }} -> {y};\n",
            ic = tile_iota(&mut iotas, m * nn),
            kr = tile_iota(&mut iotas, kk)
        );
    }
    let last = prods.last().unwrap();
    let (cells, ic) = (m * last.4, tile_iota(&mut iotas, m * last.4));
    let mut out = last.0.to_string();
    if p.kind >= 3 {
        let terms: Vec<String> = prods.iter().map(|t| format!("{}[c]", t.0)).collect();
        body += &format!("    {ic} -> map {{ c -> {} }} -> z;\n", terms.join(" + "));
        out = "z".into();
    } else if p.epi {
        tile_law(&mut laws, p.law, "bias", tile_iota(&mut iotas, last.4));
        body += &format!(
            "    {ic} -> map {{ c -> {out}[c] + bias[c % {nn}] -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> r; r }} -> z;\n",
            nn = last.4
        );
        out = "z".into();
    }
    let head: String = iotas
        .iter()
        .map(|n| format!("    {n} -> iota -> i{n};\n"))
        .collect();
    let laws = laws.text;
    let src = format!(
        "fn main() {{\n{head}{laws}    () -> time -> t0;\n{body}    () -> time -> t1;\n    {out}[0] -> println;\n    {out}[{}] -> println;\n    \"iter ms=\" -> print;\n    t1 - t0 -> println;\n}}\n",
        cells - 1
    );
    (
        src,
        prods.iter().map(|&(_, _, _, kk, nn)| (m, nn, kk)).collect(),
    )
}

/// `i<n>` (`n -> iota`), declared once.
pub(super) fn tile_iota(iotas: &mut Vec<u64>, n: u64) -> String {
    if !iotas.contains(&n) {
        iotas.push(n);
    }
    format!("i{n}")
}

/// A tile program's operand laws: their text and the draw state the next
/// law's multiplier is drawn from.
#[derive(Default)]
pub(super) struct Laws {
    text: String,
    /// Advanced by each law's fields ([`LAW_DRAW_WIDTH`] + its iota's,
    /// multiplier's, addend's and name's widths), never by how a law is
    /// spelled: a spelling change keeps every draw.
    drawn: usize,
}

/// The fixed width each law advances the draw by — the width of a law line's
/// fixed text when the draws were set (the generated set the tests' counters
/// were tuned on).
pub(super) const LAW_DRAW_WIDTH: usize = 57;

/// `name` over the iota `it`: integers in ±3, a law per name and `key`.
pub(super) fn tile_law(laws: &mut Laws, key: u8, name: &str, it: String) {
    let mul = [3, 5, 11, 13][(usize::from(key) + name.len() + laws.drawn) % 4];
    let add = name.len() * 2 + 1;
    laws.text += &format!(
        "    {it} -> map {{ t -> (t * {mul} + {add}) % 7 - 3 -> widen<f32> }} -> {name};\n"
    );
    laws.drawn +=
        LAW_DRAW_WIDTH + it.len() + mul.to_string().len() + add.to_string().len() + name.len();
}
