//! **An exact LRU trace simulator** (ADR-0050 D4): the access trace of a
//! grid's traversal, slice by slice, and the bytes an LRU cache of `keep`
//! bytes misses on it. Slices are runs of whole lines accessed contiguously,
//! so a line of slice `x` hits iff the distinct bytes of *other* slices
//! accessed since `x` was last accessed, plus `x`'s own bytes, fit the keep —
//! exactly what a line-granular LRU does (test `slices_are_line_lru`). The
//! simulator checks the closed forms' arithmetic ([`crate::reuse`]);
//! calibration probes check the hardware.

use crate::reuse::{Grid, Order, Reads, launch_tile};

/// A slice of the trace: `a(row, panel)`, `b(col, panel)` or an output key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Key {
    A(u64, u64),
    B(u64, u64),
    O(u64),
}

/// The trace of `g`'s traversal: `(slice, bytes)` in access order — the
/// zero pass first where the grid has one, then the kernel.
pub fn trace(g: &Grid) -> Vec<(Key, u64)> {
    let mut out = Vec::new();
    let tiles = g.gx * g.gy;
    let zero = |out: &mut Vec<(Key, u64)>| {
        if g.zero {
            out.extend((0..tiles).map(|t| (Key::O(t), g.so)));
        }
    };
    match g.order {
        Order::Lockstep { h, s, atomic } => {
            zero(&mut out);
            lockstep_trace(g, h, s, atomic, 0, tiles, &mut out);
        }
        Order::Stagger { h, s, atomic } => {
            zero(&mut out);
            stagger_trace(g, h, s, atomic, &mut out);
        }
        Order::Walk { h, ctas, iters } => {
            zero(&mut out);
            walk_trace(g, h, 0, tiles, ctas, iters, &mut out);
        }
        Order::Hybrid { h, ctas, iters } => {
            zero(&mut out);
            let whole = tiles / g.slots.max(1) * g.slots.max(1);
            lockstep_trace(g, h, 1, false, 0, whole, &mut out);
            walk_trace(g, h, whole, tiles - whole, ctas, iters, &mut out);
        }
    }
    out
}

/// The output key of block `(tile, part)`: a tile's own (ordered, atomic) or
/// its partition's workspace tile.
fn okey(tile: u64, part: u64, s: u64, atomic: bool) -> u64 {
    if atomic || s <= 1 {
        tile
    } else {
        tile * s + part
    }
}

/// Lockstep waves of `slots` blocks over the launch order (raster `h`,
/// partitions `s` above the tiles): each wave steps through its blocks'
/// panels together, every block's `a` then `b` slice a step, then writes
/// its output. Tiles `first .. first + count` of the raster order.
fn lockstep_trace(
    g: &Grid,
    h: u64,
    s: u64,
    atomic: bool,
    first: u64,
    count: u64,
    out: &mut Vec<(Key, u64)>,
) {
    let s = s.max(1);
    let pb = g.panels / s;
    let blocks: Vec<(u64, u64, u64, u64)> = (0..count * s)
        .map(|l| {
            let (z, t) = (l / count, first + l % count);
            let (r, j) = launch_tile(g.gx, g.gy, h, t);
            (r, j, z, okey(r * g.gx + j, z, s, atomic))
        })
        .collect();
    for wave in blocks.chunks(g.slots.max(1) as usize) {
        for st in 0..pb {
            for &(r, j, z, _) in wave {
                out.push((Key::A(r, z * pb + st), g.sa));
                out.push((Key::B(j, z * pb + st), g.sb));
            }
        }
        for &(_, _, _, o) in wave {
            out.push((Key::O(o), g.so));
        }
    }
}

/// Staggered starts: block `L` starts at `L·panels/slots` panel times and
/// reads panel `κ` at its start plus `κ`, then writes its output.
fn stagger_trace(g: &Grid, h: u64, s: u64, atomic: bool, out: &mut Vec<(Key, u64)>) {
    let s = s.max(1);
    let tiles = g.gx * g.gy;
    let pb = g.panels / s;
    let w = g.slots.max(1);
    // (time × w, block, step); step pb = the output.
    let mut ev: Vec<(u64, u64, u64)> = Vec::new();
    for l in 0..tiles * s {
        let start = l * pb;
        for st in 0..=pb {
            ev.push((start + st * w, l, st));
        }
    }
    ev.sort_unstable();
    for (_, l, st) in ev {
        let (z, t) = (l / tiles, l % tiles);
        let (r, j) = launch_tile(g.gx, g.gy, h, t);
        if st == pb {
            out.push((Key::O(okey(r * g.gx + j, z, s, atomic)), g.so));
        } else {
            out.push((Key::A(r, z * pb + st), g.sa));
            out.push((Key::B(j, z * pb + st), g.sb));
        }
    }
}

/// The walk over tiles `first .. first + count` of the launch order (raster
/// `h`): `ctas` blocks
/// of `iters` consecutive units of the `tiles × panels` space, block by
/// block within a unit; a piece's partials (or its store) where its tile's
/// panels end or its block's run does.
fn walk_trace(
    g: &Grid,
    h: u64,
    first: u64,
    count: u64,
    ctas: u64,
    iters: u64,
    out: &mut Vec<(Key, u64)>,
) {
    let total = count * g.panels;
    for u in 0..iters {
        for c in 0..ctas {
            let at = c * iters + u;
            if at >= total {
                continue;
            }
            let (t, kap) = (first + at / g.panels, at % g.panels);
            let (r, j) = launch_tile(g.gx, g.gy, h, t);
            out.push((Key::A(r, kap), g.sa));
            out.push((Key::B(j, kap), g.sb));
            if kap == g.panels - 1 || u == iters - 1 {
                out.push((Key::O(r * g.gx + j), g.so));
            }
        }
    }
}

/// A Fenwick tree over access positions.
#[allow(clippy::manual_isolate_lowest_one)]
struct Fenwick(Vec<i64>);

#[allow(clippy::manual_isolate_lowest_one)]
impl Fenwick {
    fn add(&mut self, mut i: usize, v: i64) {
        i += 1;
        while i < self.0.len() {
            self.0[i] += v;
            i += i & i.wrapping_neg();
        }
    }
    fn sum(&self, mut i: usize) -> i64 {
        // Sum of positions [0, i).
        let mut s = 0;
        while i > 0 {
            s += self.0[i];
            i -= i & i.wrapping_neg();
        }
        s
    }
}

/// **The bytes an LRU cache of `keep` bytes misses on `trace`**, by slice
/// family: a slice hits iff the distinct bytes of other slices since its
/// last access plus its own fit `keep` (the stack distance at line
/// granularity).
pub fn lru_bytes(trace: &[(Key, u64)], keep: u64) -> Reads {
    let mut last: std::collections::HashMap<Key, usize> = std::collections::HashMap::new();
    let mut f = Fenwick(vec![0; trace.len() + 1]);
    let mut r = Reads::default();
    for (p, &(key, size)) in trace.iter().enumerate() {
        let hit = match last.get(&key) {
            Some(&q) => {
                let d = (f.sum(p) - f.sum(q + 1)) as u64;
                f.add(q, -(size as i64));
                d + size <= keep
            }
            None => false,
        };
        f.add(p, size as i64);
        last.insert(key, p);
        if !hit {
            match key {
                Key::A(..) => r.a += size,
                Key::B(..) => r.b += size,
                Key::O(_) => r.out += size,
            }
        }
    }
    r
}

/// The bytes a **line-granular** LRU of `keep_lines` lines misses on
/// `trace` (sizes in lines): the reference the slice simulator equals.
pub fn lru_lines(trace: &[(Key, u64)], keep_lines: u64) -> Reads {
    let mut order: std::collections::VecDeque<(Key, u64)> = std::collections::VecDeque::new();
    let mut r = Reads::default();
    for &(key, size) in trace {
        for line in 0..size {
            let id = (key, line);
            let hit = match order.iter().position(|x| *x == id) {
                Some(i) => {
                    order.remove(i);
                    true
                }
                None => false,
            };
            order.push_front(id);
            while order.len() as u64 > keep_lines {
                order.pop_back();
            }
            if !hit {
                match key {
                    Key::A(..) => r.a += 1,
                    Key::B(..) => r.b += 1,
                    Key::O(_) => r.out += 1,
                }
            }
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The slice simulator is a line-granular LRU: every slice a run of
    /// whole lines (sizes in lines here), small grids of every order.
    #[test]
    fn slices_are_line_lru() {
        let orders = [
            Order::Lockstep {
                h: 1,
                s: 1,
                atomic: false,
            },
            Order::Lockstep {
                h: 2,
                s: 2,
                atomic: true,
            },
            Order::Stagger {
                h: 1,
                s: 1,
                atomic: false,
            },
            Order::Walk {
                h: 1,
                ctas: 5,
                iters: 7,
            },
            Order::Hybrid {
                h: 1,
                ctas: 3,
                iters: 4,
            },
        ];
        for order in orders {
            for keep in [0, 3, 8, 17, 40, 200] {
                let g = Grid {
                    gx: 4,
                    gy: 4,
                    panels: 4,
                    slots: 5,
                    sa: 2,
                    sb: 3,
                    so: 1,
                    order,
                    zero: matches!(order, Order::Walk { .. }),
                };
                let t = trace(&g);
                assert_eq!(
                    lru_bytes(&t, keep),
                    lru_lines(&t, keep),
                    "{order:?} keep {keep}"
                );
            }
        }
    }
}
