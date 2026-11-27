//! Three-gap walks, the walk's reads after a prefix, and the staggered split.

use super::*;

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

pub(super) fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Distinct bytes the walk's blocks read between the read at unit `u1` of
/// block `c1` and the read at unit `u2` of block `c2`: every block's units
/// strictly between, the first unit's blocks after `c1` and the second's
/// before `c2` — per row and column the union of the blocks' `κ` windows,
/// and the pieces they end.
pub(super) type Spans = std::collections::HashMap<u64, Vec<(u64, u64)>>;

/// Two consecutive readers `(unit, block)` and how many such pairs.
pub(super) type ReaderPair = ((u64, u64), (u64, u64), u64);

/// The `κ` spans per row and column (and the pieces ended) the walk's
/// blocks read between `(u1, c1)` and `(u2, c2)` — see [`walk_window`];
/// `c1 = u64::MAX` takes unit `u1` whole.
pub(super) fn walk_spans(
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
pub(super) fn distinct(
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

/// The lockstep waves a hybrid's tail walks after: each wave's blocks'
/// `(row, column)` in launch order (every block reads its slices at a step
/// in that order, then the wave stores its tiles).
pub struct Prev {
    pub waves: Vec<Vec<(u64, u64)>>,
}

impl Prev {
    /// The wave and position of the last block reading row (`fam_a`) or
    /// column `key`.
    pub(super) fn last(&self, fam_a: bool, key: u64) -> Option<(usize, usize)> {
        self.waves.iter().enumerate().rev().find_map(|(wi, w)| {
            w.iter()
                .rposition(|&(r, j)| if fam_a { r } else { j } == key)
                .map(|q| (wi, q))
        })
    }
}

/// [`walk`] after the lockstep waves `prev` (a hybrid's tail).
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
    let tile = |gu: u64| launch_tile(g.gx, g.gy, h, first + gu / p);
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
    // The waves' slices after the read of a slice of family `fam_a` at
    // step `kap` by block `q` of wave `wi`: the blocks after it at that
    // step (and, for an `a` slice, its own block's `b`), every block of the
    // wave at the later steps, every later wave whole; the waves' stores.
    let after_waves = |w: &Prev,
                       (wi, q): (usize, usize),
                       kap: u64,
                       fam_a: bool,
                       rs: &mut Spans,
                       cs: &mut Spans,
                       os: &mut std::collections::HashSet<u64>| {
        for (i, wave) in w.waves.iter().enumerate().skip(wi) {
            for (x, &(r, j)) in wave.iter().enumerate() {
                let lo = match i == wi {
                    true => kap + 1,
                    false => 0,
                };
                rs.entry(r).or_default().push((lo, p));
                cs.entry(j).or_default().push((lo, p));
                if i == wi && (x > q || (x == q && fam_a)) {
                    if x > q {
                        rs.entry(r).or_default().push((kap, kap + 1));
                    }
                    cs.entry(j).or_default().push((kap, kap + 1));
                }
                os.insert(r * g.gx + j);
            }
        }
    };
    // The tiles of the walk by row and by column (raster order).
    let mut by_row: std::collections::BTreeMap<u64, Vec<u64>> = Default::default();
    let mut by_col: std::collections::BTreeMap<u64, Vec<u64>> = Default::default();
    for t in 0..count {
        let (r, j) = launch_tile(g.gx, g.gy, h, first + t);
        by_row.entry(r).or_default().push(t);
        by_col.entry(j).or_default().push(t);
    }
    let mut reads =
        |groups: &std::collections::BTreeMap<u64, Vec<u64>>, size: u64, fam_a: bool| -> u64 {
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
                    // The first reader: from memory, unless the waves before
                    // read the slice and everything since their last read of it
                    // — the rest of the waves, the walk before the reader
                    // (distinct, one union) — fits.
                    let first_at = ts.iter().map(|&t| at(t)).min().unwrap_or((0, 0));
                    let fresh = prev
                        .and_then(|w| w.last(fam_a, *key).map(|l| (w, l)))
                        .is_some_and(|(w, l)| {
                            let (mut rs, mut cs, mut os) = walk_spans(
                                g,
                                h,
                                (first, count),
                                (ctas, iters),
                                (0, u64::MAX),
                                first_at,
                            );
                            after_waves(w, l, kap, fam_a, &mut rs, &mut cs, &mut os);
                            if !fam_a {
                                // A `b` reader's own `a` before it.
                                let gu = first_at.1 * iters + first_at.0;
                                let (r, _) = tile(gu);
                                rs.entry(r).or_default().push((kap, kap + 1));
                            }
                            distinct(g, (rs, cs, os)) + size <= keep
                        });
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
    let a = reads(&by_row, g.sa, true);
    let b = reads(&by_col, g.sb, false);
    // A piece's partials or store, where its tile's pieces end in time order
    // (unit, block): the first from memory unless the zero pass left it and
    // everything since fits; each next one where the window since the one
    // before (its own unit's slices included: they precede its store) fits.
    let tiles = g.gx * g.gy;
    let mut out = 0u64;
    for t in 0..count {
        let c0 = t * p / iters;
        let c1 = ((t + 1) * p - 1) / iters;
        let mut ends: Vec<(u64, u64)> = (c0..=c1)
            .map(|c| {
                let end = ((t + 1) * p).min((c + 1) * iters) - 1;
                (end - c * iters, c)
            })
            .collect();
        ends.sort_unstable();
        let (r, j) = launch_tile(g.gx, g.gy, h, first + t);
        let id = r * g.gx + j;
        let own = |rs: &mut Spans, cs: &mut Spans, (u, c): (u64, u64)| {
            let gu = c * iters + u;
            let (r, j) = tile(gu);
            rs.entry(r).or_default().push((gu % p, gu % p + 1));
            cs.entry(j).or_default().push((gu % p, gu % p + 1));
        };
        let first_hit = g.zero && {
            let (mut rs, mut cs, mut os) =
                walk_spans(g, h, (first, count), (ctas, iters), (0, u64::MAX), ends[0]);
            own(&mut rs, &mut cs, ends[0]);
            // Every wave whole (the zero pass precedes them), its stores.
            for wave in prev.iter().flat_map(|w| &w.waves) {
                for &(r, j) in wave {
                    rs.entry(r).or_default().push((0, p));
                    cs.entry(j).or_default().push((0, p));
                    os.insert(r * g.gx + j);
                }
            }
            os.extend(id + 1..tiles);
            os.remove(&id);
            distinct(g, (rs, cs, os)) + g.so <= keep
        };
        out += u64::from(!first_hit) * g.so;
        for e in ends.windows(2) {
            let (mut rs, mut cs, mut os) =
                walk_spans(g, h, (first, count), (ctas, iters), e[0], e[1]);
            own(&mut rs, &mut cs, e[1]);
            os.remove(&id);
            if distinct(g, (rs, cs, os)) + g.so > keep {
                out += g.so;
            }
        }
    }
    Reads { a, b, out }
}

/// **Staggered starts**: block `L` reads panel step `st` at time
/// `L·pb + st·slots` (in `1/slots` panel times). A slice's consecutive
/// readers are its row's (column's) blocks; the distinct bytes between are
/// the union of every block's steps inside the window (the blocks launched
/// after the later reader that start before it included), per sampled step
/// (every step on short folds, 32 past 64) — within one slice a row or
/// column, or 5 %, of the simulation (P2).
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
    // A block's output: its tile's, or its partition's workspace tile.
    let okey = |tile: u64, z: u64| if atomic || s <= 1 { tile } else { tile * s + z };
    // The distinct bytes between block `l1`'s read at step `st` and block
    // `l2`'s (events in time order, a tie in launch order — the trace's).
    // `fam_a`: the slice is an `a` panel (its own event's `b` follows it);
    // else a `b` panel (its event's `a` precedes it).
    let window = |l1: u64, l2: u64, st: u64, fam_a: bool| -> u64 {
        let (t1, t2) = (l1 * pb + st * w, l2 * pb + st * w);
        let first = (t1 + 1).saturating_sub(pb * w + pb) / pb;
        let mut rows: std::collections::HashMap<(u64, u64), Vec<(u64, u64)>> = Default::default();
        let mut cols: std::collections::HashMap<(u64, u64), Vec<(u64, u64)>> = Default::default();
        let mut outs = std::collections::HashSet::new();
        // Every block started before the later read — the blocks launched
        // after its reader included (they start `pb / slots` panel times
        // apart, so `st·slots / pb` of them run before step `st`).
        for l in first..blocks.len() as u64 {
            let start = l * pb;
            if start > t2 || (start == t2 && l >= l2) {
                break;
            }
            // Steps st' (the output at st' = pb) whose event lies strictly
            // between the two reads: after (t1, l1), before (t2, l2).
            let a = match t1.checked_sub(start) {
                None => 0,
                Some(d) if d % w == 0 && l > l1 => d / w,
                Some(d) => d / w + 1,
            };
            let d2 = t2 - start;
            let b = match d2 % w == 0 && l < l2 {
                true => d2 / w + 1,
                false => d2.div_ceil(w),
            }
            .min(pb + 1);
            if b <= a {
                continue;
            }
            let (r, j, z) = blocks[l as usize];
            if b > pb {
                outs.insert(okey(r * g.gx + j, z));
            }
            let b = b.min(pb);
            if b > a {
                rows.entry((r, z)).or_default().push((a, b));
                cols.entry((j, z)).or_default().push((a, b));
            }
        }
        // The slice read beside it in its own event.
        if fam_a {
            let (_, j, z) = blocks[l1 as usize];
            cols.entry((j, z)).or_default().push((st, st + 1));
        } else {
            let (r, _, z) = blocks[l2 as usize];
            rows.entry((r, z)).or_default().push((st, st + 1));
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
        na * g.sa + nb * g.sb + outs.len() as u64 * g.so
    };
    // The window between a slice's two readers depends on the panel step
    // it is read at (the later reader's steps before it and the earlier
    // one's after it change sides as the step moves): every step on short
    // folds, 32 evenly spread past 64 — a step's miss stands for its share.
    let samples = if pb <= 64 { pb } else { 32 };
    type KeyOf<'a> = &'a dyn Fn(&(u64, u64, u64)) -> (u64, u64);
    let fam = |key: KeyOf, size: u64, fam_a: bool| -> u64 {
        let mut last: std::collections::HashMap<(u64, u64), u64> = Default::default();
        let mut n = 0;
        for (l, bk) in blocks.iter().enumerate() {
            let l = l as u64;
            let k = key(bk);
            match last.get(&k) {
                None => n += samples,
                Some(&l1) => {
                    for i in 0..samples {
                        let st = (2 * i + 1) * pb / (2 * samples);
                        if window(l1, l, st, fam_a) + size > keep {
                            n += 1;
                        }
                    }
                }
            }
            last.insert(k, l);
        }
        (n * pb).div_ceil(samples) * size
    };
    let a = fam(&|&(r, _, z)| (r, z), g.sa, true);
    let b = fam(&|&(_, j, z)| (j, z), g.sb, false);
    // Every block stores (or adds) its tile once.
    let outs = tiles * s;
    Reads {
        a,
        b,
        out: outs * g.so,
    }
}
