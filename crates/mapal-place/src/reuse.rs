//! **Reuse by distinct bytes** (ADR-0050 D4): the bytes a cache instance
//! misses on a grid's traversal, in closed forms over its structure — the
//! slices a wave, a walk window or a staggered window touches, counted as
//! *distinct* slices (interval unions per row and column), never the bytes
//! requested between two reads.
//!
//! - Lockstep waves (ordered, split): **exact** — every consecutive pair of a
//!   slice's reads is a pair inside one wave step or across waves, and the
//!   distinct slices between are interval unions over the waves between,
//!   linear in the panel step: `O(blocks)` a pair of waves, equal to the exact
//!   LRU simulation ([`crate::lru`]) byte for byte (property P1).
//! - The walk: a row's (column's) readers form an arithmetic progression
//!   modulo the run, whose gaps take at most three sizes ([`three_gap`]); the
//!   distinct slices in a gap are the union of the slots' windows — within a
//!   slice a row or column, or 5 %, of the simulation (P2).
//! - Staggered starts: the same window union over the slots a window holds.

/// A grid's traversal as the cache sees it: `gx × gy` output tiles, `panels`
/// `k` panels a tile, `slots` slots resident at once, slice bytes `sa` (an
/// `a` row panel), `sb` (a `b` column panel), `so` (an output tile), the
/// order, and whether a zero pass streams the output through the cache first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub gx: u64,
    pub gy: u64,
    pub panels: u64,
    pub slots: u64,
    pub sa: u64,
    pub sb: u64,
    pub so: u64,
    pub order: Order,
    pub zero: bool,
}

/// The traversal's access order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Waves of `slots` blocks in step, launch order by raster `h`, `s`
    /// partitions of `k` above the tiles (partition-major), their partials
    /// added into the tile (`atomic`) or written to a workspace tile each.
    Lockstep { h: u64, s: u64, atomic: bool },
    /// The same blocks, each starting when one before it ends.
    Stagger { h: u64, s: u64, atomic: bool },
    /// Stream-K: `ctas` blocks of `iters` consecutive units of the
    /// `tiles × panels` space over the raster order.
    Walk { h: u64, ctas: u64, iters: u64 },
    /// Whole waves of the raster order in step, the tail walked by `ctas`
    /// blocks of `iters` units.
    Hybrid { h: u64, ctas: u64, iters: u64 },
}

/// Missed bytes by slice family.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Reads {
    pub a: u64,
    pub b: u64,
    pub out: u64,
}

impl Reads {
    pub fn operands(&self) -> u64 {
        self.a + self.b
    }
    pub fn total(&self) -> u64 {
        self.a + self.b + self.out
    }
}

/// The tile `(row, col)` launch `l` runs under raster `h`: groups of `h`
/// block-rows walked column by column (`h ≤ 1`: row-major). A last group
/// shorter than `h` walks its own rows.
pub fn launch_tile(gx: u64, gy: u64, h: u64, l: u64) -> (u64, u64) {
    if h <= 1 {
        return (l / gx, l % gx);
    }
    let (grp, rem) = (l / (gx * h), l % (gx * h));
    let rows = h.min(gy - grp * h);
    (grp * h + rem % rows, rem / rows)
}

/// The walk's grid over `slots`: `iters` units a block so `ctas` blocks
/// cover `units`, no block empty.
pub fn walk_grid(units: u64, slots: u64) -> (u64, u64) {
    let iters = units.div_ceil(slots.max(1)).max(1);
    (units.div_ceil(iters), iters)
}

/// The bytes a cache of `keep` bytes misses on `g`, by its closed form.
pub fn reads(g: &Grid, keep: u64) -> Reads {
    let tiles = g.gx * g.gy;
    let mut r = match g.order {
        Order::Lockstep { h, s, atomic } => lockstep(g, h, s, atomic, 0, tiles, keep),
        Order::Stagger { h, s, atomic } => stagger(g, h, s, atomic, keep),
        Order::Walk { h, ctas, iters } => walk(g, h, 0, tiles, ctas, iters, keep),
        Order::Hybrid { h, ctas, iters } => {
            let whole = tiles / g.slots.max(1) * g.slots.max(1);
            let a = lockstep(g, h, 1, false, 0, whole, keep);
            // The last ordered wave's rows and columns: the tail's first
            // reader of one finds it where the rest of that wave and the
            // walk before the reader fit the keep.
            let w = g.slots.max(1);
            let last: Vec<(u64, u64)> = (whole.saturating_sub(w)..whole)
                .map(|t| launch_tile(g.gx, g.gy, h, t))
                .collect();
            let rows: std::collections::BTreeSet<u64> = last.iter().map(|x| x.0).collect();
            let cols: std::collections::BTreeSet<u64> = last.iter().map(|x| x.1).collect();
            let prev = Prev {
                rows,
                cols,
                outs: last.len() as u64,
            };
            let b = walk_after(g, h, whole, tiles - whole, ctas, iters, keep, Some(&prev));
            Reads {
                a: a.a + b.a,
                b: a.b + b.b,
                out: a.out + b.out,
            }
        }
    };
    if g.zero && !matches!(g.order, Order::Lockstep { .. }) {
        // The zero pass writes every output tile once.
        r.out += tiles * g.so;
    }
    r
}

/// **The bytes a lockstep grid's waves need**: each wave's distinct slices
/// (its rows' and columns' panels) — what its readers wait on together: a
/// slice the cache does not hold is fetched once for every reader of the
/// wave, so the fraction of these fetched from memory is the fraction of
/// requests that wait on memory's latency.
pub fn needed(g: &Grid, h: u64, s: u64) -> u64 {
    let s = s.max(1);
    let tiles = g.gx * g.gy;
    let pb = g.panels / s;
    let mut rows = std::collections::HashSet::new();
    let mut cols = std::collections::HashSet::new();
    let mut total = 0;
    let w = g.slots.max(1);
    for l in 0..tiles * s {
        if l % w == 0 {
            total += (rows.len() as u64 * g.sa + cols.len() as u64 * g.sb) * pb;
            rows.clear();
            cols.clear();
        }
        let (z, t) = (l / tiles, l % tiles);
        let (r, j) = launch_tile(g.gx, g.gy, h, t);
        rows.insert((r, z));
        cols.insert((j, z));
    }
    total + (rows.len() as u64 * g.sa + cols.len() as u64 * g.sb) * pb
}

/// `#{σ ∈ [0, n) : β·σ > c}`.
fn count_gt(beta: i128, c: i128, n: u64) -> u64 {
    let n = n as i128;
    let floor = |a: i128, b: i128| a.div_euclid(b);
    let k = match beta.signum() {
        0 => {
            if c < 0 {
                n
            } else {
                0
            }
        }
        1 => {
            // σ > c / β ⟺ σ ≥ ⌊c/β⌋ + 1.
            let lo = (floor(c, beta) + 1).clamp(0, n);
            n - lo
        }
        _ => {
            // σ < c / β (β < 0) ⟺ σ ≤ ⌈c/β⌉ − 1 = −⌊−c/β⌋ − 1 … as ⌊(−c − 1)/(−β)⌋.
            let hi = floor(-c - 1, -beta).clamp(-1, n - 1);
            hi + 1
        }
    };
    k as u64
}

/// Stamped flag sets over group ids (reset in O(touched)).
struct Flags {
    f: Vec<u8>,
    touched: Vec<usize>,
}

impl Flags {
    fn new(n: usize) -> Self {
        Flags {
            f: vec![0; n],
            touched: Vec::new(),
        }
    }
    fn set(&mut self, i: usize, bit: u8) {
        if self.f[i] == 0 {
            self.touched.push(i);
        }
        self.f[i] |= bit;
    }
    fn clear(&mut self) {
        for &i in &self.touched {
            self.f[i] = 0;
        }
        self.touched.clear();
    }
}

const IN_M: u8 = 1;
const IN_1: u8 = 2;
const IN_2: u8 = 4;
const POINT: u8 = 8;

#[derive(Clone, Copy)]
struct Blk {
    a: usize,
    b: usize,
    o: usize,
}

/// The cross-wave distance of a slice of family `fam` (0 `a`, 1 `b`) of
/// group `own`, last read at position `p1` of wave `w1` and next at `p2` of
/// wave `w2`, as `(α, β)`: the distinct bytes of other slices between the
/// two reads at panel step `σ` are `α + β·σ`.
#[allow(clippy::too_many_arguments)]
fn cross(
    waves: &[&[Blk]],
    fam: u8,
    own: usize,
    (w1, p1): (usize, usize),
    (w2, p2): (usize, usize),
    pb: u64,
    (sa, sb, so): (u64, u64, u64),
    fl: &mut [Flags; 3],
) -> (i128, i128) {
    let [fa, fb, fo] = fl;
    for w in &waves[w1 + 1..w2] {
        for x in w.iter() {
            fa.set(x.a, IN_M);
            fb.set(x.b, IN_M);
            fo.set(x.o, IN_M);
        }
    }
    for (q, x) in waves[w1].iter().enumerate() {
        fa.set(x.a, IN_1 | if q > p1 { POINT } else { 0 });
        let tail_b = if fam == 0 { q >= p1 } else { q > p1 };
        fb.set(x.b, IN_1 | if tail_b { POINT } else { 0 });
        fo.set(x.o, IN_M);
    }
    for (q, x) in waves[w2].iter().enumerate() {
        let head_a = if fam == 0 { q < p2 } else { q <= p2 };
        fa.set(x.a, IN_2 | if head_a { POINT } else { 0 });
        fb.set(x.b, IN_2 | if q < p2 { POINT } else { 0 });
    }
    let pb = pb as i128;
    let sum = |f: &Flags, skip: Option<usize>| -> (i128, i128) {
        let (mut al, mut be) = (0i128, 0i128);
        for &i in &f.touched {
            if Some(i) == skip {
                // The slice's own other panels: every one but itself.
                al += pb - 1;
                continue;
            }
            let x = f.f[i];
            if x & IN_M != 0 {
                al += pb;
                continue;
            }
            if x & IN_1 != 0 {
                al += pb - 1;
                be -= 1;
            }
            if x & IN_2 != 0 {
                be += 1;
            }
            if x & POINT != 0 {
                al += 1;
            }
        }
        (al, be)
    };
    let (aa, ba) = sum(fa, (fam == 0).then_some(own));
    let (ab, bb) = sum(fb, (fam == 1).then_some(own));
    let outs = fo.touched.len() as i128;
    let alpha = aa * sa as i128 + ab * sb as i128 + outs * so as i128;
    let beta = ba * sa as i128 + bb * sb as i128;
    for f in fl.iter_mut() {
        f.clear();
    }
    (alpha, beta)
}

/// **Lockstep waves, exact.** Tiles `first .. first + count` of the raster
/// order, `s` partitions a tile (partition-major launch), waves of `slots`.
pub fn lockstep(
    g: &Grid,
    h: u64,
    s: u64,
    atomic: bool,
    first: u64,
    count: u64,
    keep: u64,
) -> Reads {
    let s = s.max(1);
    let pb = g.panels / s;
    let w = g.slots.max(1) as usize;
    let (sa, sb, so) = (g.sa, g.sb, g.so);
    let tiles = g.gx * g.gy;
    let unique_out = !atomic || s <= 1;
    let blocks: Vec<Blk> = (0..count * s)
        .map(|l| {
            let (z, t) = (l / count.max(1), first + l % count.max(1));
            let (r, j) = launch_tile(g.gx, g.gy, h, t);
            let tile = r * g.gx + j;
            Blk {
                a: (r * s + z) as usize,
                b: (j * s + z) as usize,
                o: if unique_out && s > 1 {
                    (tile * s + z) as usize
                } else {
                    tile as usize
                },
            }
        })
        .collect();
    let waves: Vec<&[Blk]> = blocks.chunks(w).collect();
    let (na, nb) = ((g.gy * s) as usize, (g.gx * s) as usize);
    let no = (tiles * s) as usize;
    let mut fl = [Flags::new(na), Flags::new(nb), Flags::new(no)];
    let mut miss = Reads::default();
    let keep = keep as i128;
    // Operand families: the last (wave, position) each group was read at.
    for fam in 0u8..2 {
        let size = if fam == 0 { sa } else { sb } as i128;
        let n = if fam == 0 { na } else { nb };
        let mut last: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut seen = Flags::new(n.max(na).max(nb));
        let mut acc = 0u64;
        for (wi, wave) in waves.iter().enumerate() {
            for (q, x) in wave.iter().enumerate() {
                let gid = if fam == 0 { x.a } else { x.b };
                match last[gid] {
                    None => acc += pb * size as u64,
                    Some((w1, p1)) if w1 == wi => {
                        // Same wave, same step: the distinct slices of the
                        // blocks between (each one slice of each family).
                        let (lo_a, hi_a, lo_b, hi_b) = match fam {
                            0 => (p1 + 1, q, p1, q),
                            _ => (p1 + 1, q + 1, p1 + 1, q),
                        };
                        for y in &wave[lo_a..hi_a] {
                            seen.set(y.a, 1);
                        }
                        let da = seen.touched.len() as i128;
                        seen.clear();
                        for y in &wave[lo_b..hi_b] {
                            seen.set(y.b, 1);
                        }
                        let db = seen.touched.len() as i128;
                        seen.clear();
                        if da * sa as i128 + db * sb as i128 + size > keep {
                            acc += pb * size as u64;
                        }
                    }
                    Some((w1, p1)) => {
                        let (al, be) = cross(
                            &waves,
                            fam,
                            gid,
                            (w1, p1),
                            (wi, q),
                            pb,
                            (sa, sb, so),
                            &mut fl,
                        );
                        // Misses: σ with α + β·σ + size > keep.
                        acc += count_gt(be, keep - size - al, pb) * size as u64;
                    }
                }
                last[gid] = Some((wi, q));
            }
        }
        if fam == 0 {
            miss.a = acc;
        } else {
            miss.b = acc;
        }
    }
    // Outputs: the zero pass (every tile, in order), then each wave's
    // stores or partial adds after its steps.
    let mut groups_upto = vec![(0i128, 0i128); waves.len()];
    {
        let (mut sa_seen, mut sb_seen) = (Flags::new(na), Flags::new(nb));
        for (wi, wave) in waves.iter().enumerate() {
            for x in wave.iter() {
                sa_seen.set(x.a, 1);
                sb_seen.set(x.b, 1);
            }
            groups_upto[wi] = (sa_seen.touched.len() as i128, sb_seen.touched.len() as i128);
        }
    }
    let mut last: Vec<Option<(usize, usize)>> = vec![None; no];
    let mut red_before = Flags::new(no);
    let mut acc = 0u64;
    let pbi = pb as i128;
    let so_i = so as i128;
    if g.zero {
        acc += tiles * so;
    }
    for (wi, wave) in waves.iter().enumerate() {
        for (q, x) in wave.iter().enumerate() {
            let o = x.o;
            let d = match last[o] {
                None if !g.zero || o as u64 >= tiles => None,
                None => {
                    // From the zero pass: the tiles zeroed after it, the
                    // outputs added before it, every operand slice so far.
                    let after = (tiles - 1 - o as u64) as i128;
                    let before = red_before.touched.iter().filter(|&&i| i < o).count() as i128;
                    let (ga, gb) = groups_upto[wi];
                    Some((after + before) * so_i + ga * pbi * sa as i128 + gb * pbi * sb as i128)
                }
                Some((w1, p1)) if w1 == wi => {
                    let mut f = Flags::new(0);
                    std::mem::swap(&mut f, &mut fl[2]);
                    for y in &wave[p1 + 1..q] {
                        f.set(y.o, 1);
                    }
                    let n = f.touched.len() as i128;
                    f.clear();
                    std::mem::swap(&mut f, &mut fl[2]);
                    Some(n * so_i)
                }
                Some((w1, p1)) => {
                    let [fa, fb, fo] = &mut fl;
                    for y in &waves[w1][p1 + 1..] {
                        fo.set(y.o, 1);
                    }
                    for w in &waves[w1 + 1..wi] {
                        for y in w.iter() {
                            fo.set(y.o, 1);
                        }
                    }
                    for y in &waves[wi][..q] {
                        fo.set(y.o, 1);
                    }
                    for w in &waves[w1 + 1..=wi] {
                        for y in w.iter() {
                            fa.set(y.a, 1);
                            fb.set(y.b, 1);
                        }
                    }
                    let n_o = fo.touched.iter().filter(|&&i| i != o).count() as i128;
                    let d = n_o * so_i
                        + fa.touched.len() as i128 * pbi * sa as i128
                        + fb.touched.len() as i128 * pbi * sb as i128;
                    for f in fl.iter_mut() {
                        f.clear();
                    }
                    Some(d)
                }
            };
            if d.is_none_or(|d| d + so_i > keep) {
                acc += so;
            }
            last[o] = Some((wi, q));
            red_before.set(o, 1);
        }
    }
    miss.out = acc;
    miss
}

/// **The three-gap theorem**: the circular gaps between the points
/// `j·step mod modulus` (`j < n`) — at most three sizes — as `(gap, count)`,
/// zero gaps for repeated points included. `i₁` (`i₂`): the index of the
/// point nearest 0 from above (below); gaps of `u = i₁·step mod m` occur
/// `n − i₁` times, of `v = m − i₂·step mod m` `n − i₂` times, of `u + v`
/// `i₁ + i₂ − n` times (Sós).
pub fn three_gap(step: u64, n: u64, modulus: u64) -> Vec<(u64, u64)> {
    let m = modulus.max(1);
    let st = step % m;
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![(m, 1)];
    }
    let g = gcd(st, m);
    let period = m / g;
    if st == 0 {
        return vec![(0, n - 1), (m, 1)];
    }
    if n > period {
        // `period` distinct points `g` apart, each repeated.
        return vec![(0, n - period), (g, period)];
    }
    let (mut i1, mut u) = (0u64, u64::MAX);
    let (mut i2, mut v) = (0u64, u64::MAX);
    for j in 1..n {
        let p = (j as u128 * st as u128 % m as u128) as u64;
        if p < u {
            (i1, u) = (j, p);
        }
        if m - p < v {
            (i2, v) = (j, m - p);
        }
    }
    let mut out = vec![(u, n - i1), (v, n - i2)];
    if i1 + i2 > n {
        out.push((u + v, i1 + i2 - n));
    }
    out.retain(|&(_, c)| c > 0);
    out
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Distinct bytes the walk's blocks read between the read at unit `u1` of
/// block `c1` and the read at unit `u2` of block `c2`: every block's units
/// strictly between, the first unit's blocks after `c1` and the second's
/// before `c2` — per row and column the union of the blocks' `κ` windows,
/// and the pieces they end.
type Spans = std::collections::HashMap<u64, Vec<(u64, u64)>>;

/// Two consecutive readers `(unit, block)` and how many such pairs.
type ReaderPair = ((u64, u64), (u64, u64), u64);

/// The `κ` spans per row and column (and the pieces ended) the walk's
/// blocks read between `(u1, c1)` and `(u2, c2)` — see [`walk_window`];
/// `c1 = u64::MAX` takes unit `u1` whole.
fn walk_spans(
    g: &Grid,
    h: u64,
    (first, count): (u64, u64),
    (ctas, iters): (u64, u64),
    (u1, c1): (u64, u64),
    (u2, c2): (u64, u64),
) -> (Spans, Spans, std::collections::HashSet<u64>) {
    let p = g.panels;
    let total = count * p;
    let mut rows: Spans = Default::default();
    let mut cols: Spans = Default::default();
    let mut outs = std::collections::HashSet::new();
    for c in 0..ctas {
        let lo_u = if c1 == u64::MAX {
            u1
        } else {
            u1 + u64::from(c <= c1)
        };
        let hi_u = (u2 + 1).saturating_sub(u64::from(c >= c2)).min(iters);
        if hi_u <= lo_u {
            continue;
        }
        let lo = c * iters + lo_u;
        let hi = (c * iters + hi_u).min(total);
        let mut at = lo;
        while at < hi {
            let t = at / p;
            let k0 = at % p;
            let k1 = (hi - t * p).min(p);
            let (r, j) = launch_tile(g.gx, g.gy, h, first + t);
            rows.entry(r).or_default().push((k0, k1));
            cols.entry(j).or_default().push((k0, k1));
            if k1 == p || t * p + k1 == (c + 1) * iters {
                outs.insert(r * g.gx + j);
            }
            at = t * p + k1;
        }
    }
    (rows, cols, outs)
}

/// The distinct bytes of spans: per row and column the union of its `κ`
/// intervals.
fn distinct(
    g: &Grid,
    (mut rows, mut cols, outs): (Spans, Spans, std::collections::HashSet<u64>),
) -> u64 {
    let union = |v: &mut Vec<(u64, u64)>| -> u64 {
        v.sort_unstable();
        let (mut n, mut end) = (0, 0);
        for &(a, b) in v.iter() {
            let a = a.max(end);
            if b > a {
                n += b - a;
                end = b;
            }
        }
        n
    };
    let na: u64 = rows.values_mut().map(union).sum();
    let nb: u64 = cols.values_mut().map(union).sum();
    na * g.sa + nb * g.sb + outs.len() as u64 * g.so
}

/// Distinct bytes the walk's blocks read between the read at unit `u1` of
/// block `c1` and the read at unit `u2` of block `c2`: every block's units
/// strictly between, the first unit's blocks after `c1` and the second's
/// before `c2` — per row and column the union of the blocks' `κ` windows,
/// and the pieces they end.
#[doc(hidden)]
pub fn walk_window(
    g: &Grid,
    h: u64,
    fc: (u64, u64),
    ci: (u64, u64),
    a: (u64, u64),
    b: (u64, u64),
) -> u64 {
    distinct(g, walk_spans(g, h, fc, ci, a, b))
}

/// The three-gap theorem with each gap's index step: `(gap, count, Δj)` —
/// reader `j + Δj` follows reader `j` by `gap` (Sós: `u` at `i₁`, `v` at
/// `−i₂`, `u + v` at `i₁ − i₂`). `None` where points repeat (a period below
/// `n`).
pub fn three_gap_steps(step: u64, n: u64, modulus: u64) -> Option<Vec<(u64, u64, i64)>> {
    let m = modulus.max(1);
    let st = step % m;
    if n < 2 || st == 0 || n > m / gcd(st, m) {
        return None;
    }
    let (mut i1, mut u) = (0u64, u64::MAX);
    let (mut i2, mut v) = (0u64, u64::MAX);
    for j in 1..n {
        let p = (j as u128 * st as u128 % m as u128) as u64;
        if p < u {
            (i1, u) = (j, p);
        }
        if m - p < v {
            (i2, v) = (j, m - p);
        }
    }
    let mut out = vec![(u, n - i1, i1 as i64), (v, n - i2, -(i2 as i64))];
    if i1 + i2 > n {
        out.push((u + v, i1 + i2 - n, i1 as i64 - i2 as i64));
    }
    out.retain(|&(_, c, _)| c > 0);
    Some(out)
}

/// **The walk** over tiles `first .. first + count` of the raster order: a
/// slice's readers are the tiles of its row (column), at unit times
/// `(t·panels + κ) mod iters` — an arithmetic progression whose gaps take at
/// most three sizes ([`three_gap_steps`]: each gap's reader step gives the
/// blocks between, so the window is exact up to which blocks it holds); a
/// reader after a gap whose distinct bytes plus the slice exceed the keep
/// reads it again. `κ` is sampled past 64 panels (16 offsets, scaled), as the
/// readers' times only shift with it.
pub fn walk(g: &Grid, h: u64, first: u64, count: u64, ctas: u64, iters: u64, keep: u64) -> Reads {
    walk_after(g, h, first, count, ctas, iters, keep, None)
}

/// The lockstep wave a hybrid's tail walks after: its rows, columns and
/// output tiles.
pub struct Prev {
    pub rows: std::collections::BTreeSet<u64>,
    pub cols: std::collections::BTreeSet<u64>,
    pub outs: u64,
}

/// [`walk`] after the lockstep wave `prev` (a hybrid's tail).
#[allow(clippy::too_many_arguments)]
pub fn walk_after(
    g: &Grid,
    h: u64,
    first: u64,
    count: u64,
    ctas: u64,
    iters: u64,
    keep: u64,
    prev: Option<&Prev>,
) -> Reads {
    let p = g.panels.max(1);
    let iters = iters.max(1);
    if count == 0 {
        return Reads::default();
    }
    // Every panel offset on short folds; 16 evenly spread past 64.
    let samples = if p <= 64 { p } else { 16 };
    let mut memo: std::collections::HashMap<(u64, u64, u64, u64), u64> = Default::default();
    // The pair's window, with the other family's slice a reader's unit reads
    // beside it (an `a` reader's own `b` after it; a `b` reader's own `a`
    // before it) and the first reader's partials where its piece ends there.
    let mut d = |a: (u64, u64), b: (u64, u64), fam_a: bool| {
        *memo
            .entry((a.0, a.1, b.0, b.1 * 2 + u64::from(fam_a)))
            .or_insert_with(|| {
                let (mut rs, mut cs, mut os) =
                    walk_spans(g, h, (first, count), (ctas, iters), a, b);
                let (g1, g2) = (a.1 * iters + a.0, b.1 * iters + b.0);
                let tile = |gu: u64| launch_tile(g.gx, g.gy, h, first + gu / p);
                if fam_a {
                    let (_, j) = tile(g1);
                    cs.entry(j).or_default().push((g1 % p, g1 % p + 1));
                } else {
                    let (r, _) = tile(g2);
                    rs.entry(r).or_default().push((g2 % p, g2 % p + 1));
                }
                if g1 % p == p - 1 || a.0 == iters - 1 {
                    let (r, j) = tile(g1);
                    os.insert(r * g.gx + j);
                }
                distinct(g, (rs, cs, os))
            })
    };
    // The tiles of the walk by row and by column (raster order).
    let mut by_row: std::collections::BTreeMap<u64, Vec<u64>> = Default::default();
    let mut by_col: std::collections::BTreeMap<u64, Vec<u64>> = Default::default();
    for t in 0..count {
        let (r, j) = launch_tile(g.gx, g.gy, h, first + t);
        by_row.entry(r).or_default().push(t);
        by_col.entry(j).or_default().push(t);
    }
    let wave_outs = prev.map_or(0, |w| w.outs * g.so);
    let mut reads = |groups: &std::collections::BTreeMap<u64, Vec<u64>>,
                     size: u64,
                     fam_a: bool,
                     warm: Option<&std::collections::BTreeSet<u64>>|
     -> u64 {
        let other = if fam_a { g.sb } else { g.sa };
        let mut n = 0u64;
        for s in 0..samples {
            let kap = s * p / samples;
            for (key, ts) in groups.iter() {
                let at = |t: u64| ((t * p + kap) % iters, (t * p + kap) / iters);
                // The readers' gaps take at most three sizes
                // ([`three_gap_steps`]): where the widest window — every
                // block's units over the widest gap, both slices each, the
                // pieces they end — fits the keep, every reader after the
                // first finds the slice. Else each consecutive pair of
                // readers `(unit, block)` in time order, its window exact.
                let step_t = if ts.len() > 1 { ts[1] - ts[0] } else { 1 };
                let affine = ts.windows(2).all(|w| w[1] - w[0] == step_t);
                let widest = affine
                    .then(|| three_gap_steps(step_t * p, ts.len() as u64, iters))
                    .flatten()
                    .map(|gaps| gaps.iter().map(|x| x.0).max().unwrap_or(0));
                let all_fit = widest.is_some_and(|gw| {
                    (gw + 1) * ctas * (g.sa + g.sb + g.so) + other + size <= keep
                });
                let pairs: Vec<ReaderPair> = if all_fit {
                    Vec::new()
                } else {
                    let mut tc: Vec<(u64, u64)> = ts.iter().map(|&t| at(t)).collect();
                    tc.sort_unstable();
                    tc.windows(2).map(|w| (w[0], w[1], 1)).collect()
                };
                // The first reader: from memory, unless the wave before
                // read the slice and the rest of that wave plus the walk
                // before the reader (distinct, one union) fit.
                let first_at = ts.iter().map(|&t| at(t)).min().unwrap_or((0, 0));
                let fresh = warm.is_some_and(|w| w.contains(key)) && {
                    let (mut rs, mut cs, outs) =
                        walk_spans(g, h, (first, count), (ctas, iters), (0, u64::MAX), first_at);
                    let w = prev.expect("a wave before");
                    for r in &w.rows {
                        rs.entry(*r).or_default().push((kap + 1, p));
                    }
                    for c in &w.cols {
                        cs.entry(*c).or_default().push((kap + 1, p));
                    }
                    distinct(g, (rs, cs, outs)) + wave_outs + size <= keep
                };
                n += u64::from(!fresh);
                for (a, b, c) in pairs {
                    if d(a, b, fam_a) + size > keep {
                        n += c;
                    }
                }
            }
        }
        (n * size * p).div_ceil(samples)
    };
    let a = reads(&by_row, g.sa, true, prev.map(|w| &w.rows));
    let b = reads(&by_col, g.sb, false, prev.map(|w| &w.cols));
    // A piece's partials or store: a tile's pieces.
    let pieces: u64 = (0..count)
        .map(|t| {
            let c0 = t * p / iters;
            let c1 = ((t + 1) * p - 1) / iters;
            c1 - c0 + 1
        })
        .sum();
    Reads {
        a,
        b,
        out: pieces * g.so,
    }
}

/// **Staggered starts**: block `L` reads panel step `st` at time
/// `L·pb + st·slots` (in `1/slots` panel times). A slice's consecutive
/// readers are its row's (column's) blocks; the distinct bytes between are
/// the union of the active blocks' step windows, taken at the middle step.
pub fn stagger(g: &Grid, h: u64, s: u64, atomic: bool, keep: u64) -> Reads {
    let s = s.max(1);
    let tiles = g.gx * g.gy;
    let pb = (g.panels / s).max(1);
    let w = g.slots.max(1);
    let blocks: Vec<(u64, u64, u64)> = (0..tiles * s)
        .map(|l| {
            let (z, t) = (l / tiles, l % tiles);
            let (r, j) = launch_tile(g.gx, g.gy, h, t);
            (r, j, z)
        })
        .collect();
    let window = |l1: u64, l2: u64, st: u64| -> u64 {
        // Times (l1·pb + st·w, l2·pb + st·w): block L's steps in it.
        let (t1, t2) = (l1 * pb + st * w, l2 * pb + st * w);
        let first = (t1 + 1).saturating_sub(pb * w + pb) / pb;
        let mut rows: std::collections::HashMap<(u64, u64), Vec<(u64, u64)>> = Default::default();
        let mut cols: std::collections::HashMap<(u64, u64), Vec<(u64, u64)>> = Default::default();
        let mut outs = 0;
        for l in first..=l2.min(blocks.len() as u64 - 1) {
            let start = l * pb;
            if start >= t2 {
                break;
            }
            // Steps st' with t1 < start + st'·w < t2.
            let a = if t1 >= start { (t1 - start) / w + 1 } else { 0 };
            let b = (t2 - start).div_ceil(w).min(pb + 1);
            if b <= a {
                continue;
            }
            let (r, j, z) = blocks[l as usize];
            if b > pb {
                outs += 1;
            }
            let b = b.min(pb);
            if b > a {
                rows.entry((r, z)).or_default().push((a, b));
                cols.entry((j, z)).or_default().push((a, b));
            }
        }
        let union = |v: &mut Vec<(u64, u64)>| -> u64 {
            v.sort_unstable();
            let (mut n, mut end) = (0, 0);
            for &(a, b) in v.iter() {
                let a = a.max(end);
                if b > a {
                    n += b - a;
                    end = b;
                }
            }
            n
        };
        let na: u64 = rows.values_mut().map(union).sum();
        let nb: u64 = cols.values_mut().map(union).sum();
        na * g.sa + nb * g.sb + outs * g.so
    };
    let mid = pb / 2;
    type KeyOf<'a> = &'a dyn Fn(&(u64, u64, u64)) -> (u64, u64);
    let fam = |key: KeyOf, size: u64| -> u64 {
        let mut last: std::collections::HashMap<(u64, u64), u64> = Default::default();
        let mut n = 0;
        for (l, bk) in blocks.iter().enumerate() {
            let l = l as u64;
            let k = key(bk);
            match last.get(&k) {
                None => n += pb,
                Some(&l1) => {
                    if window(l1, l, mid) + size > keep {
                        n += pb;
                    }
                }
            }
            last.insert(k, l);
        }
        n * size
    };
    let a = fam(&|&(r, _, z)| (r, z), g.sa);
    let b = fam(&|&(_, j, z)| (j, z), g.sb);
    // Every block stores (or adds) its tile once.
    let _ = atomic;
    let outs = tiles * s;
    Reads {
        a,
        b,
        out: outs * g.so,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_gt_is_the_brute_count() {
        for beta in -5i128..=5 {
            for c in -30i128..=30 {
                for n in 0..12u64 {
                    let brute = (0..n as i128).filter(|&s| beta * s > c).count() as u64;
                    assert_eq!(count_gt(beta, c, n), brute, "{beta} {c} {n}");
                }
            }
        }
    }

    #[test]
    fn raster_walks_groups_column_by_column() {
        // gx 3, gy 4, h 2: (0,0) (1,0) (0,1) (1,1) (0,2) (1,2) (2,0) …
        let order: Vec<_> = (0..12).map(|l| launch_tile(3, 4, 2, l)).collect();
        assert_eq!(
            &order[..7],
            &[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 0)]
        );
        // A short last group (gy 3, h 2) walks its one row.
        let order: Vec<_> = (0..9).map(|l| launch_tile(3, 3, 2, l)).collect();
        assert_eq!(&order[6..], &[(2, 0), (2, 1), (2, 2)]);
    }
}
