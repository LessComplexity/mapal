//! **One evaluator for every schedule** (ADR-0050 D4): predicted time =
//! over the schedule's phases (waves, a walk, a tail, a prologue), the
//! maximum over resources of work ÷ rate, summed, plus fixed costs.
//!
//! - **compute** — the busiest unit's work at the operation's rate, its
//!   efficiency (measured curve, else the latency rule: independent chains
//!   against latency × issue), at the sustained clock;
//! - **movement** — the device cache's misses by distinct bytes
//!   ([`crate::reuse`]) at the memory's rate; the panels into each unit at
//!   the cache's rate (device-wide and per unit);
//! - **latency** — bytes in flight the ring keeps a unit against the
//!   source's latency (Little's law);
//! - **issue** — the lowering's declared counts per class against the unit's
//!   pipes;
//! - **combine** — atomic adds as a resource at their place (overlapped while
//!   the output stays within the keep, a memory read-modify-write past it),
//!   workspace bytes and the reduce, the zero pass once, a per-piece
//!   pipeline fill;
//! - **fixed** — launches.
//!
//! No term reads a fit of a kernel's time; a marked fact read is named in
//! [`Terms::marked`].

use crate::fact::Bound;
use crate::lower::Declared;
use crate::machine::{Class, EffPoint, Machine, Pattern, Start};
use crate::reuse::{self, Grid, Order, Reads};
use crate::schedule::{Comb, Schedule, Site, Traversal};

/// One phase's resource times, ns.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Phase {
    pub compute: f64,
    pub issue: f64,
    pub dram: f64,
    pub cache: f64,
    pub unit_in: f64,
    pub latency: f64,
    pub atomics: f64,
    pub prologue: f64,
    pub ns: f64,
}

impl Phase {
    fn close(mut self) -> Self {
        self.ns = [
            self.compute,
            self.issue,
            self.dram,
            self.cache,
            self.unit_in,
            self.latency,
            self.atomics,
        ]
        .into_iter()
        .fold(0.0, f64::max)
            + self.prologue;
        self
    }

    /// The resource that bounds the phase.
    pub fn bound(&self) -> &'static str {
        let v = [
            (self.compute, "compute"),
            (self.issue, "issue"),
            (self.dram, "dram"),
            (self.cache, "cache"),
            (self.unit_in, "unit_in"),
            (self.latency, "latency"),
            (self.atomics, "atomics"),
        ];
        v.iter()
            .fold((0.0, "none"), |b, &(x, n)| if x > b.0 { (x, n) } else { b })
            .1
    }
}

/// The evaluator's terms for one schedule.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Terms {
    pub phases: Vec<Phase>,
    pub combine: f64,
    pub zero: f64,
    pub fixed: f64,
    pub ns: f64,
    /// Slots a unit holds (the footprint's occupancy, at most the
    /// schedule's), slots in all.
    pub per_unit: u64,
    pub slots: u64,
    /// The device cache's misses (bytes) and the keep they were counted at.
    pub reads: Reads,
    pub keep: u64,
    pub dram_bytes: u64,
    /// The efficiency the compute term took.
    pub eff: f64,
    /// Marked facts (derived / nominal) this evaluation read.
    pub marked: Vec<String>,
    /// Why the schedule is infeasible (empty: feasible).
    pub infeasible: String,
}

impl Terms {
    pub fn feasible(&self) -> bool {
        self.infeasible.is_empty()
    }
    /// The terms as one line (`; mapal.place`, the replay's evidence).
    pub fn line(&self) -> String {
        let ph: Vec<String> = self
            .phases
            .iter()
            .map(|p| {
                format!(
                    "{}:{:.0}[c{:.0} i{:.0} d{:.0} l2{:.0} u{:.0} lat{:.0} at{:.0} pro{:.0}]",
                    p.bound(),
                    p.ns,
                    p.compute,
                    p.issue,
                    p.dram,
                    p.cache,
                    p.unit_in,
                    p.latency,
                    p.atomics,
                    p.prologue
                )
            })
            .collect();
        format!(
            "ns={:.0} u={} slots={} eff={:.3} dram={:.1}MB reads=a{:.1}/b{:.1}/o{:.1}MB keep={:.1}MB combine={:.0} zero={:.0} fixed={:.0} {}",
            self.ns,
            self.per_unit,
            self.slots,
            self.eff,
            self.dram_bytes as f64 / 1e6,
            self.reads.a as f64 / 1e6,
            self.reads.b as f64 / 1e6,
            self.reads.out as f64 / 1e6,
            self.keep as f64 / 1e6,
            self.combine,
            self.zero,
            self.fixed,
            ph.join(" ")
        )
    }
}

/// The operation's efficiency with `chains` independent accumulator chains
/// a lane and `groups` resident groups a scheduler: the measured curve's
/// monotone envelope (the best point both coordinates of which the query
/// reaches), else the latency rule — `min(1, chains·groups / (latency ×
/// instructions a scheduler issues a clock))`, marked.
pub fn efficiency(m: &Machine, op: usize, chains: f64, groups: f64, b: Bound) -> (f64, bool) {
    let o = &m.ops[op];
    let dominated: Vec<&EffPoint> = o
        .eff
        .iter()
        .filter(|p| p.chains <= chains + 1e-9 && p.groups <= groups + 1e-9)
        .collect();
    if let Some(best) = dominated
        .iter()
        .map(|p| p.eff.rate(b))
        .fold(None, |a: Option<f64>, x| Some(a.map_or(x, |a| a.max(x))))
    {
        return (best.clamp(0.0, 1.0), false);
    }
    // Instructions one scheduler issues a clock through this row.
    let n = o.n.min();
    let per_instr = (2 * o.m * n * o.k * if o.lanes > 1 { 1 } else { m.group_lanes() }) as f64;
    let ipc = o.rate.rate(b) / per_instr / m.subs().max(1) as f64;
    let need = o.latency.cost(b) * ipc;
    // A scheduler holding a group hides latency with its groups' chains;
    // below one group a scheduler, the schedulers without one idle.
    let e = if need <= 0.0 {
        1.0
    } else {
        (chains * groups.max(1.0) / need).min(1.0)
    };
    (e * groups.min(1.0), true)
}

/// **Evaluate** schedule `x` of site `s` on machine `m` with its lowering's
/// declared counts `d`, every fact at bound `b`.
pub fn evaluate(m: &Machine, s: &Site, x: &Schedule, d: &Declared, b: Bound) -> Terms {
    evaluate_at(m, s, x, d, b, false)
}

/// **The evaluation's floor** (the search's cheap admissible bound, D5):
/// [`evaluate`] with the device cache missing every operand byte exactly
/// once — no term falls as misses rise (memory bytes, the miss ratio's
/// latency and fill), so no schedule's time is below it; it skips the
/// reuse closed forms (a walk's per-reader windows).
pub fn evaluate_floor(m: &Machine, s: &Site, x: &Schedule, d: &Declared, b: Bound) -> Terms {
    evaluate_at(m, s, x, d, b, true)
}

fn evaluate_at(m: &Machine, s: &Site, x: &Schedule, d: &Declared, b: Bound, floor: bool) -> Terms {
    let mut t = Terms::default();
    let mark = |t: &mut Terms, what: String| {
        if !t.marked.contains(&what) {
            t.marked.push(what);
        }
    };
    let units = m.units().max(1);
    let scratch = m.unit_scratch();
    let bytes: Vec<_> = scratch.map(|p| (p, d.smem)).into_iter().collect();
    let derived = m.occupancy(d.threads, d.regs, &bytes);
    if derived == 0 {
        t.infeasible = format!(
            "footprint: {} lanes x {} regs, {} B scratch fit no slot",
            d.threads, d.regs, d.smem
        );
        t.ns = f64::INFINITY;
        return t;
    }
    let u = if x.per_unit == 0 {
        derived
    } else {
        x.per_unit.min(derived)
    };
    let w = units * u;
    t.per_unit = u;
    t.slots = w;
    let op = &m.ops[x.op];
    let ghz = m.ghz(b);
    if m.dispatch.clock_khz.marked() {
        mark(&mut t, "dispatch.clock_khz".into());
    }
    let (bm, bn, bk) = (x.bm().max(1), x.bn().max(1), x.bk().max(1));
    let batch = s.batch.max(1);
    let (gx, gy) = (s.n / bn, batch * (s.m / bm));
    let tiles = gx * gy;
    let panels = (s.k / bk).max(1);
    if tiles == 0 {
        t.infeasible = "no tile divides the site".into();
        t.ns = f64::INFINITY;
        return t;
    }
    let (sa, sb, so) = (bm * bk * s.ea, bn * bk * s.eb, bm * bn * s.eo);
    // The device cache and the memory behind it.
    let cache = m.device_cache();
    let mem = m.device_memory();
    let pattern = match x.trav {
        Traversal::Walk { .. } | Traversal::Hybrid { .. } => Pattern::Rotated,
        _ => Pattern::Cyclic,
    };
    let keep_f = cache
        .and_then(|c| m.places[c].cache())
        .map(|c| c.keep.of(pattern));
    let keep = keep_f.map_or(0, |k| k.rate(b));
    if keep_f.is_some_and(|k| k.marked()) {
        mark(&mut t, format!("keep.{pattern:?}"));
    }
    t.keep = keep;
    let dram_bw = mem
        .and_then(|p| m.reach(p, crate::machine::ExecKind::Device))
        .map(|r| {
            if r.total_bytes_per_ns.marked() {
                mark(&mut t, "memory.bw".into());
            }
            r.total_bytes_per_ns.rate(b)
        })
        .unwrap_or(1.0)
        .max(1e-9);
    let mem_lat = mem
        .and_then(|p| m.reach(p, crate::machine::ExecKind::Device))
        .map_or(0.0, |r| r.latency_ns.cost(b));
    let (cache_bw, cache_lat, unit_bw) = cache
        .map(|c| {
            let dev = m.reach(c, crate::machine::ExecKind::Device);
            let unit = m.reach(c, crate::machine::ExecKind::Unit);
            if dev.is_some_and(|r| r.total_bytes_per_ns.marked())
                || unit.is_some_and(|r| r.bytes_per_ns.marked())
            {
                mark(&mut t, "cache.bw".into());
            }
            (
                dev.map_or(f64::INFINITY, |r| r.total_bytes_per_ns.rate(b)),
                dev.map_or(0.0, |r| r.latency_ns.cost(b)),
                unit.map_or(f64::INFINITY, |r| r.bytes_per_ns.rate(b)),
            )
        })
        .unwrap_or((f64::INFINITY, 0.0, f64::INFINITY));
    // Start timing: a measured start is read; an unmeasured one is the
    // worse of the two at the slow bound, the better at the fast one.
    let starts: Vec<Start> = match (m.dispatch.start_src.marked(), b) {
        (false, _) | (true, Bound::Value) => vec![m.dispatch.start],
        (true, _) => {
            mark(&mut t, "dispatch.start".into());
            vec![Start::Lockstep, Start::Stagger]
        }
    };
    let grid = |order: Order, zero: bool| Grid {
        gx,
        gy,
        panels,
        slots: w,
        sa,
        sb,
        so,
        order,
        zero,
    };
    let (s_parts, atomic) = match x.trav {
        Traversal::Split { s, comb, .. } => (s.max(1), comb == Comb::Atomic),
        _ => (1, false),
    };
    let zero = matches!(
        x.trav,
        Traversal::Split {
            comb: Comb::Atomic,
            ..
        } | Traversal::Walk { .. }
            | Traversal::Hybrid { .. }
    );
    let reads_of = |start: Start| -> Reads {
        let h = x.trav.raster();
        let order = match (x.trav, start) {
            (Traversal::Walk { h, ctas, iters }, _) => Order::Walk { h, ctas, iters },
            (Traversal::Hybrid { h, ctas, iters }, _) => Order::Hybrid { h, ctas, iters },
            (_, Start::Lockstep) => Order::Lockstep {
                h,
                s: s_parts,
                atomic,
            },
            (_, Start::Stagger) => Order::Stagger {
                h,
                s: s_parts,
                atomic,
            },
        };
        reuse::reads(&grid(order, false), keep)
    };
    // Warm operands: resident from the previous call where they fit.
    let once_a = gy * panels * sa;
    let once_b = gx * panels * sb;
    let mut reads = match floor {
        true => Reads {
            a: once_a,
            b: once_b,
            out: tiles * so,
        },
        false => starts
            .iter()
            .map(|&st| reads_of(st))
            .max_by_key(|r| match b {
                Bound::Fast => u64::MAX - r.operands(),
                _ => r.operands(),
            })
            .unwrap_or_default(),
    };
    if s.warm[0] && once_a <= keep {
        reads.a = reads.a.saturating_sub(once_a);
    }
    if s.warm[1] && once_b <= keep {
        reads.b = reads.b.saturating_sub(once_b);
    }
    t.reads = reads;
    let out_bytes = batch * s.m * s.n * s.eo;
    let epi_bytes = batch * s.m * s.n * s.epi;
    // The output: stored once; partial adds past the keep read and write it
    // again from memory each partition.
    let out_fits = out_bytes <= keep;
    let rmw = match x.trav {
        Traversal::Split {
            comb: Comb::Atomic,
            s,
            ..
        } if !out_fits => 2 * s * out_bytes,
        _ => 0,
    };
    let ws_bytes = match x.trav {
        Traversal::Split {
            comb: Comb::Workspace,
            s,
            ..
        } => s * out_bytes,
        _ => 0,
    };
    let dram_bytes = reads.operands() + out_bytes + epi_bytes + rmw;
    t.dram_bytes = dram_bytes;
    // The operation's rate a unit and its efficiency.
    if op.rate.marked() {
        mark(&mut t, format!("op.{}.rate", op.name));
    }
    let rate_unit = op.rate.rate(b) * ghz; // flops / ns
    let groups_per_sub = |slots: u64| (slots * d.groups) as f64 / m.subs().max(1) as f64;
    let eff_at = |t: &mut Terms, slots: u64| -> f64 {
        let (e, rule) = efficiency(m, x.op, d.chains as f64, groups_per_sub(slots), b);
        let what = format!("op.{}.eff", op.name);
        if rule && !t.marked.contains(&what) {
            t.marked.push(what);
        }
        e.max(1e-3)
    };
    let issue_ns = |trips: f64| -> f64 {
        // Per class against its pipe, and every class against the unit's
        // dispatch (one group-instruction a scheduler a clock).
        let mut worst: f64 = 0.0;
        let mut all = 0.0;
        for (c, n) in &d.issue {
            all += n;
            if let Some(p) = m.pipe(*c) {
                worst = worst.max(trips * n / (p.per_clk.rate(b) * ghz));
            }
        }
        worst.max(trips * all / (m.subs().max(1) as f64 * ghz))
    };
    let atomic_rate = m
        .dispatch
        .atomics
        .first()
        .map_or(0.0, |a| a.ops_per_ns.rate(b));
    // A phase of `blocks` slots of `trips` panels each over `w` slots.
    // `needed`: the bytes the phase's requests wait on together (a lockstep
    // wave's distinct slices; a walk's every request), of which
    // `cache_reads` come from memory.
    let phase =
        |t: &mut Terms, blocks: u64, trips: u64, flops_trip: f64, cache_reads: f64, needed: f64| {
            let nb = blocks.div_ceil(units); // the busiest unit's slots
            let resident = u.min(nb).max(1);
            let eff = eff_at(t, resident);
            t.eff = eff;
            let unit_trips = (nb * trips) as f64;
            let compute = unit_trips * flops_trip / (rate_unit * eff);
            let issue = issue_ns(unit_trips);
            let panel = (sa + sb) as f64;
            let unit_in = unit_trips * panel / unit_bw;
            let cache = (blocks * trips) as f64 * panel / cache_bw;
            let miss = (cache_reads / needed.max(1.0)).clamp(0.0, 1.0);
            let lat = cache_lat + miss * (mem_lat - cache_lat).max(0.0);
            let fly = (resident * d.in_flight).max(1) as f64;
            let latency = unit_trips * panel * lat / fly;
            let prologue = nb.div_ceil(resident) as f64 * d.fill as f64 * lat;
            Phase {
                compute,
                issue,
                cache,
                unit_in,
                latency,
                prologue,
                ..Phase::default()
            }
        };
    // The operation's work a trip: the useful flops, or what the lowering
    // declares it issues (a split form's passes).
    let flops_trip = match d.work {
        w if w > 0.0 => w,
        _ => (2 * bm * bn * bk) as f64,
    };
    match x.trav {
        Traversal::Ordered { .. } | Traversal::Split { .. } | Traversal::Persistent { .. } => {
            let blocks = tiles * s_parts;
            let needed = needed_of(
                floor,
                &grid(
                    Order::Walk {
                        h: 1,
                        ctas: 0,
                        iters: 0,
                    },
                    false,
                ),
                x.trav.raster(),
                s_parts,
            );
            let mut p = phase(
                &mut t,
                blocks,
                panels / s_parts,
                flops_trip,
                reads.operands() as f64,
                needed as f64,
            );
            p.dram = dram_bytes as f64 / dram_bw;
            if atomic {
                p.atomics = (s_parts * batch * s.m * s.n) as f64 / atomic_rate.max(1e-9);
            }
            t.phases.push(p.close());
        }
        Traversal::Walk { ctas, iters, .. } => {
            let requested = (tiles * panels * (sa + sb)) as f64;
            let mut p = phase(
                &mut t,
                ctas,
                iters,
                flops_trip,
                reads.operands() as f64,
                requested,
            );
            // A piece a tile a block crosses: its fill.
            let pieces = ctas + tiles;
            p.prologue = pieces.div_ceil(units) as f64 * d.fill as f64 * (cache_lat + mem_lat)
                / 2.0
                / u.min(ctas.div_ceil(units)).max(1) as f64;
            p.dram = dram_bytes as f64 / dram_bw;
            let shared = if floor {
                0
            } else {
                shared_pieces(tiles, panels, iters)
            };
            p.atomics = (shared * bm * bn) as f64 / atomic_rate.max(1e-9);
            t.phases.push(p.close());
        }
        Traversal::Hybrid { ctas, iters, .. } => {
            let whole = tiles / w * w;
            let tail = tiles - whole;
            let frac = |n: u64| n as f64 / tiles as f64;
            if whole > 0 {
                let needed = needed_of(
                    floor,
                    &grid(
                        Order::Walk {
                            h: 1,
                            ctas: 0,
                            iters: 0,
                        },
                        false,
                    ),
                    x.trav.raster(),
                    1,
                ) as f64;
                let mut p = phase(
                    &mut t,
                    whole,
                    panels,
                    flops_trip,
                    reads.operands() as f64 * frac(whole),
                    needed * frac(whole),
                );
                p.dram = dram_bytes as f64 * frac(whole) / dram_bw;
                t.phases.push(p.close());
            }
            if tail > 0 {
                let mut p = phase(
                    &mut t,
                    ctas,
                    iters,
                    flops_trip,
                    reads.operands() as f64 * frac(tail),
                    (tail * panels * (sa + sb)) as f64,
                );
                p.dram = dram_bytes as f64 * frac(tail) / dram_bw;
                let shared = if floor {
                    0
                } else {
                    shared_pieces(tail, panels, iters)
                };
                p.atomics = (shared * bm * bn) as f64 / atomic_rate.max(1e-9);
                t.phases.push(p.close());
            }
        }
    }
    // Combine: the workspace's reduce; the zero pass once.
    if ws_bytes > 0 {
        let rbytes = (ws_bytes + out_bytes) as f64;
        let bw = if ws_bytes <= keep { cache_bw } else { dram_bw };
        t.combine = rbytes / bw;
    }
    if zero {
        if m.dispatch.zero_bytes_per_ns.marked() {
            mark(&mut t, "dispatch.zero_bytes_per_ns".into());
        }
        // A hybrid whose whole pieces store zeroes the rows its walked
        // tiles span: the raster order's first `tiles − whole` (the
        // lowering's layout); every piece adding, the whole output.
        let zeroed = match x.trav {
            Traversal::Hybrid { h, .. } if d.whole_stores => {
                let whole = tiles / w * w;
                let h = h.max(1);
                let rows = (bm * h * (tiles - whole).div_ceil(gx * h)).min(batch * s.m);
                rows * s.n * s.eo
            }
            _ => out_bytes,
        };
        t.zero = m.dispatch.zero_ns.cost(b)
            + zeroed as f64 / m.dispatch.zero_bytes_per_ns.rate(b).max(1e-9);
    }
    // Fixed: the launches beyond their work (the reduce is its own).
    let launches = 1 + u64::from(ws_bytes > 0);
    let blocks = match x.trav {
        Traversal::Walk { ctas, .. } | Traversal::Persistent { ctas, .. } => ctas,
        _ => tiles * s_parts,
    };
    if let Some(l) = m.launch(x.launch) {
        if l.ns.marked() {
            mark(&mut t, "dispatch.launch".into());
        }
        t.fixed = launches as f64 * l.ns.cost(b)
            + blocks.saturating_sub(units) as f64 * l.per_slot_ps.cost(b) / 1e3;
    }
    t.ns = t.phases.iter().map(|p| p.ns).sum::<f64>() + t.combine + t.zero + t.fixed;
    t
}

/// [`reuse::needed`], or unbounded on the floor (every request then waits
/// on the cache's latency alone: the latency terms' least).
fn needed_of(floor: bool, g: &Grid, h: u64, s: u64) -> u64 {
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
