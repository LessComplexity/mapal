//! **The candidate space** (ADR-0050 D5 steps 1–4), generated from the facts
//! bottom-up — never from a table: (1) the operations the machine lists that
//! the site's precision set admits, with their roles; (2–3) each lowering's
//! group tiles, slot tiles, `bk`, depths and layouts within the capacities
//! ([`Lowering::slots`]); (4) traversals and their parameters over the slots
//! the footprint derives, filtered by the site's fold and the lowering's
//! declared traversals. The search (branch and bound over these) is a later
//! step; the coverage test checks the rules' launches are points of it.

use crate::lower::{Declared, Lowering, TravKind};
use crate::machine::Machine;
use crate::reuse::walk_grid;
use crate::schedule::{Comb, Fold, Schedule, Site, Traversal};

/// D5 step 1: the machine's operations whose operand formats the site's
/// precision set admits, both roles where the operation's shape allows the
/// transposed product.
pub fn ops(m: &Machine, s: &Site) -> Vec<(usize, bool)> {
    let mut out = Vec::new();
    for (i, o) in m.ops.iter().enumerate() {
        if s.fmts.contains(&o.fa) && s.fmts.contains(&o.fb) {
            out.push((i, false));
            out.push((i, true));
        }
    }
    out
}

/// The powers of two dividing `n` (1 first).
pub fn pow2_divisors(n: u64) -> Vec<u64> {
    (0..64)
        .map(|p| 1u64 << p)
        .take_while(|&h| h <= n.max(1))
        .filter(|&h| n.is_multiple_of(h))
        .collect()
}

/// The divisors of `n` in `[lo, hi]`.
pub fn divisors(n: u64, lo: u64, hi: u64) -> Vec<u64> {
    (lo.max(1)..=hi.min(n))
        .filter(|d| n.is_multiple_of(*d))
        .collect()
}

/// D5 step 4: the traversals of slot schedule `x` (declared counts `d`) —
/// raster heights dividing the grid's rows, `k` partitions dividing its
/// panels (atomic on a free fold, a workspace on a fixed one too), walks and
/// hybrids over every occupancy up to the footprint's (free fold), each kind
/// only where the lowering emits it; a persistent grid only where the
/// dispatch's round-robin jitter is measured.
pub fn traversals(
    m: &Machine,
    s: &Site,
    x: &Schedule,
    d: &Declared,
    kinds: &[TravKind],
) -> Vec<Traversal> {
    let units = m.units().max(1);
    let scratch = m.unit_scratch();
    let bytes: Vec<_> = scratch.map(|p| (p, d.smem)).into_iter().collect();
    let occ = m.occupancy(d.threads, d.regs, &bytes);
    let (bm, bn, bk) = (x.bm().max(1), x.bn().max(1), x.bk().max(1));
    let (gx, gy) = (s.n / bn, s.batch.max(1) * (s.m / bm));
    let tiles = gx * gy;
    let panels = s.k / bk;
    let mut out = Vec::new();
    if occ == 0 || tiles == 0 || panels == 0 {
        return out;
    }
    let has = |k: TravKind| kinds.contains(&k);
    let heights = pow2_divisors(gy);
    if has(TravKind::Ordered) {
        out.extend(heights.iter().map(|&h| Traversal::Ordered { h }));
    }
    let cut = s.kcut && s.fold != Fold::Sequential;
    if cut && panels >= 2 {
        for sp in divisors(panels, 2, panels) {
            if s.fold == Fold::Free && has(TravKind::SplitAtomic) {
                out.push(Traversal::Split {
                    s: sp,
                    comb: Comb::Atomic,
                    h: 1,
                });
            }
            if has(TravKind::SplitWorkspace) {
                out.push(Traversal::Split {
                    s: sp,
                    comb: Comb::Workspace,
                    h: 1,
                });
            }
        }
    }
    if cut && s.fold == Fold::Free && panels >= 2 {
        for u in 1..=occ {
            let w = units * u;
            if has(TravKind::Walk) && tiles * panels > w {
                let (ctas, iters) = walk_grid(tiles * panels, w);
                for &h in &heights {
                    out.push(Traversal::Walk { h, ctas, iters });
                }
            }
            let whole = tiles / w * w;
            if has(TravKind::Hybrid) && whole > 0 && whole < tiles {
                let (ctas, iters) = walk_grid((tiles - whole) * panels, w);
                out.push(Traversal::Hybrid { h: 1, ctas, iters });
            }
        }
    }
    if has(TravKind::Persistent) && m.dispatch.rr_jitter.is_some() {
        for u in 1..=occ {
            out.push(Traversal::Persistent {
                h: 1,
                ctas: units * u,
            });
        }
    }
    out
}

/// D5 steps 1–4 over `lows`: every feasible `(schedule, declared counts)`.
pub fn candidates(m: &Machine, s: &Site, lows: &[&dyn Lowering]) -> Vec<(Schedule, Declared)> {
    let mut out = Vec::new();
    for l in lows {
        for (op, swap) in l.ops(m, s) {
            for x in l.slots(m, s, op, swap) {
                let Ok(d) = l.declare(m, s, &x) else {
                    continue;
                };
                for t in traversals(m, s, &x, &d, l.traversals()) {
                    let y = Schedule {
                        trav: t,
                        ..x.clone()
                    };
                    if let Ok(d) = l.declare(m, s, &y) {
                        out.push((y, d));
                    }
                }
            }
        }
    }
    out
}
