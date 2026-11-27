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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

thread_local! {
    /// [`reads`] and [`needed`] by grid and keep: a search prices many
    /// schedules over one grid (its depths, layouts and occupancies).
    static READS: std::cell::RefCell<std::collections::HashMap<(Grid, u64), Reads>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    static NEEDED: std::cell::RefCell<std::collections::HashMap<(Grid, u64, u64), u64>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    static SPLIT: std::cell::RefCell<std::collections::HashMap<(Grid, u64), (Reads, Reads)>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// The bytes a cache of `keep` bytes misses on `g`, by its closed form
/// (memoized: a pure function of the grid and the keep).
pub fn reads(g: &Grid, keep: u64) -> Reads {
    if let Some(r) = READS.with(|c| c.borrow().get(&(*g, keep)).copied()) {
        return r;
    }
    let r = reads_uncached(g, keep);
    READS.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() > 1 << 16 {
            c.clear();
        }
        c.insert((*g, keep), r);
    });
    r
}

fn reads_uncached(g: &Grid, keep: u64) -> Reads {
    let tiles = g.gx * g.gy;
    let mut r = match g.order {
        Order::Lockstep { h, s, atomic } => lockstep(g, h, s, atomic, 0, tiles, keep),
        Order::Stagger { h, s, atomic } => stagger(g, h, s, atomic, keep),
        Order::Walk { h, ctas, iters } => walk(g, h, 0, tiles, ctas, iters, keep),
        Order::Hybrid { .. } => {
            let (w, t) = hybrid_split(g, keep);
            Reads {
                a: w.a + t.a,
                b: w.b + t.b,
                out: w.out + t.out,
            }
        }
    };
    // The zero pass writes every output tile once (the lockstep form, a
    // hybrid's waves included, counts its own).
    if g.zero && matches!(g.order, Order::Walk { .. } | Order::Stagger { .. }) {
        r.out += tiles * g.so;
    }
    r
}

/// [`hybrid_phases`] of a grid whose order is a hybrid, memoized (the
/// evaluator prices each phase by its own misses).
pub fn hybrid_split(g: &Grid, keep: u64) -> (Reads, Reads) {
    let Order::Hybrid { h, ctas, iters } = g.order else {
        return (reads(g, keep), Reads::default());
    };
    if let Some(r) = SPLIT.with(|c| c.borrow().get(&(*g, keep)).copied()) {
        return r;
    }
    let r = hybrid_phases(g, h, ctas, iters, keep);
    SPLIT.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() > 1 << 16 {
            c.clear();
        }
        c.insert((*g, keep), r);
    });
    r
}

/// **A hybrid's first reads by phase** (raster `h`): every operand slice
/// its whole waves read, and the walk's slices the waves never read — what
/// each phase misses at least (the evaluation's floor) and what a warm
/// operand's first reads are in each.
pub fn hybrid_first(g: &Grid, h: u64) -> (Reads, Reads) {
    let tiles = g.gx * g.gy;
    let w = g.slots.max(1);
    let tail = tiles - tiles / w * w;
    let mut rows = std::collections::HashSet::new();
    let mut cols = std::collections::HashSet::new();
    for t in tail..tiles {
        let (r, j) = launch_tile(g.gx, g.gy, h, t);
        rows.insert(r);
        cols.insert(j);
    }
    let waves = Reads {
        a: rows.len() as u64 * g.sa * g.panels,
        b: cols.len() as u64 * g.sb * g.panels,
        out: (tiles - tail) * g.so,
    };
    let (mut ra, mut cb) = (
        std::collections::HashSet::new(),
        std::collections::HashSet::new(),
    );
    for t in 0..tail {
        let (r, j) = launch_tile(g.gx, g.gy, h, t);
        if !rows.contains(&r) {
            ra.insert(r);
        }
        if !cols.contains(&j) {
            cb.insert(j);
        }
    }
    let walk = Reads {
        a: ra.len() as u64 * g.sa * g.panels,
        b: cb.len() as u64 * g.sb * g.panels,
        out: tail * g.so,
    };
    (waves, walk)
}

/// **A hybrid's two phases' misses** (ADR-0050 C5 layout): the whole waves
/// over the raster order's last `⌊tiles / slots⌋·slots` tiles (lockstep,
/// exact; the zero pass, where the grid has one, counted there), then the
/// walk over its first `tail` — a slice's first walk reader finds it where
/// the distinct bytes since the waves' last read of it fit the keep.
pub fn hybrid_phases(g: &Grid, h: u64, ctas: u64, iters: u64, keep: u64) -> (Reads, Reads) {
    let tiles = g.gx * g.gy;
    let w = g.slots.max(1);
    let whole = tiles / w * w;
    let tail = tiles - whole;
    let waves_r = lockstep(g, h, 1, false, tail, whole, keep);
    let blocks: Vec<(u64, u64)> = (tail..tiles)
        .map(|t| launch_tile(g.gx, g.gy, h, t))
        .collect();
    let prev = Prev {
        waves: blocks.chunks(w as usize).map(<[_]>::to_vec).collect(),
    };
    let walk_r = walk_after(g, h, 0, tail, ctas, iters, keep, Some(&prev));
    (waves_r, walk_r)
}

/// **The bytes a lockstep grid's waves need**: each wave's distinct slices
/// (its rows' and columns' panels) — what its readers wait on together: a
/// slice the cache does not hold is fetched once for every reader of the
/// wave, so the fraction of these fetched from memory is the fraction of
/// requests that wait on memory's latency.
pub fn needed(g: &Grid, h: u64, s: u64) -> u64 {
    if let Some(n) = NEEDED.with(|c| c.borrow().get(&(*g, h, s)).copied()) {
        return n;
    }
    let n = needed_uncached(g, h, s);
    NEEDED.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() > 1 << 16 {
            c.clear();
        }
        c.insert((*g, h, s), n);
    });
    n
}

fn needed_uncached(g: &Grid, h: u64, s: u64) -> u64 {
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

mod cross;
#[cfg(test)]
mod tests;
mod walk;

pub use self::cross::*;
pub use self::walk::*;
