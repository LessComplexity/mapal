//! The walk's result movement, shared-tile pieces and the lower bound.

use super::*;

/// A walk's result movement a block: tiles it stores whole (a lowering whose
/// whole pieces store), cells it adds (each shared tile's pieces; every
/// piece where whole pieces add too), and the pieces it crosses (each a
/// fill) — over `ctas` blocks of `iters` units, `tiles` tiles of `panels`
/// (on the floor no tile is shared: no adds).
pub(super) fn walk_ends(
    tiles: u64,
    panels: u64,
    iters: u64,
    ctas: u64,
    floor: bool,
    cells: f64,
    whole_stores: bool,
) -> (f64, f64, f64) {
    let ctas = ctas.max(1) as f64;
    // One pass: each tile's pieces (its units over the blocks' runs), shared
    // where more than one block holds it (none on the floor).
    let it = iters.max(1);
    let (mut shared, mut shared_tiles) = (0u64, 0u64);
    if !floor {
        for t in 0..tiles {
            let (c0, c1) = (t * panels / it, ((t + 1) * panels - 1) / it);
            if c1 > c0 {
                shared += c1 - c0 + 1;
                shared_tiles += 1;
            }
        }
    }
    let whole = tiles.saturating_sub(shared_tiles) as f64;
    let pieces = whole + shared as f64;
    match whole_stores {
        true => (whole / ctas, shared as f64 * cells / ctas, pieces / ctas),
        false => (0.0, pieces * cells / ctas, pieces / ctas),
    }
}

/// [`reuse::needed`], or unbounded on the floor (every request then waits
/// on the cache's latency alone: the latency terms' least).
pub(super) fn needed_of(floor: bool, g: &Grid, h: u64, s: u64) -> u64 {
    match floor {
        true => u64::MAX,
        false => reuse::needed(g, h, s),
    }
}

/// The pieces of shared tiles of a walk: a tile whose `panels` units span
/// blocks `c₀ … c₁` (`iters` units a block) has `c₁ − c₀ + 1` pieces when
/// shared, none otherwise.
pub fn shared_pieces(tiles: u64, panels: u64, iters: u64) -> u64 {
    let iters = iters.max(1);
    (0..tiles)
        .map(|t| {
            let c0 = t * panels / iters;
            let c1 = ((t + 1) * panels - 1) / iters;
            if c1 > c0 { c1 - c0 + 1 } else { 0 }
        })
        .sum()
}

/// **Admissible lower bound** (D5): the site's flops at every unit's peak
/// rate of the fastest admitted operation, and every operand byte read once
/// at the memory's rate — no schedule's evaluated time is below it.
pub fn lower_bound(m: &Machine, s: &Site, ops: &[usize], b: Bound) -> f64 {
    let ghz = m.ghz(b);
    let peak = ops
        .iter()
        .map(|&o| m.ops[o].rate.rate(b))
        .fold(0.0, f64::max)
        * ghz
        * m.units() as f64;
    let mem = m
        .device_memory()
        .and_then(|p| m.reach(p, crate::machine::ExecKind::Device))
        .map_or(f64::INFINITY, |r| r.total_bytes_per_ns.rate(b));
    let flops = s.flops() as f64 / peak.max(1e-9);
    // Operands resident from the previous call may read nothing.
    let b_ = s.batch.max(1);
    let once = b_ * s.m * s.n * (s.eo + s.epi)
        + if s.warm[0] { 0 } else { b_ * s.m * s.k * s.ea }
        + if s.warm[1] { 0 } else { b_ * s.k * s.n * s.eb };
    let bytes = once as f64 / mem.max(1e-9);
    flops.max(bytes)
}

/// The classes a count may name (for listings).
pub const CLASSES: [Class; 7] = [
    Class::Math,
    Class::Matrix,
    Class::Shared,
    Class::Global,
    Class::Barrier,
    Class::Convert,
    Class::Other,
];
