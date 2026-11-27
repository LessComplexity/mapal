//! **One evaluator for every schedule** (ADR-0050 D4) — **the cost model's
//! pipeline calculation** over the primitives (`cost-model.md`): every term
//! reads a measured or declared fact ([`Machine::cost`]: a primitive's rate,
//! its one use's latency and the concurrency that saturates it).
//!
//! - **a trip's steady state** — the slowest of compute (the operation at
//!   its measured efficiency for the resident groups), issue (the declared
//!   counts against the pipes), the scratchpad's fragment loads, and every
//!   mover: the panels into the unit by the declared mover (its rate, or —
//!   where the ring keeps less than its loaded latency × rate in flight —
//!   latency × work ÷ in flight), the device cache's feed, and the memory's
//!   pages by each operand's run and the rows the units stream at once
//!   ([`Flow`], [`steady`]);
//! - **a wave** — each piece's fill (the ring's first stage at its loaded
//!   latency) + trips × (steady + the declared barriers at their measured
//!   cost) + the drain (the last trip's shorter half) + the result's
//!   movement: the epilogue's stores and the partial adds at their measured
//!   rates, after the loop where the lowering declares it so ([`pipeline`]);
//! - **the grid** — its waves on the busiest unit (whole waves and the tail,
//!   each at its own residency) by the traversal: ordered, split, the walk
//!   and the hybrid's phases (misses by distinct bytes, [`crate::reuse`]);
//! - **fixed** — the zero pass, the workspace's reduce and the launches.
//!
//! No term reads a fit of a kernel's time; a marked fact read is named in
//! [`Terms::marked`].

use crate::fact::Bound;
use crate::lower::{Declared, Move};
use crate::machine::{Class, EffPoint, Machine, Pattern, PlaceId, PrimOf, Start};
use crate::reuse::{self, Grid, Order, Reads};
use crate::schedule::{Comb, Schedule, Site, Traversal};

/// One phase's resource times, ns: each resource's own total over the
/// phase's trips (the evidence: which bounds), its fills (`prologue`),
/// barriers, drains and the result's movement (`epilogue` stores, `atomics`
/// the partial adds), and the phase's time — their pipeline sum, never
/// below its memory bytes at the memory's rate.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Phase {
    pub compute: f64,
    pub issue: f64,
    pub dram: f64,
    pub cache: f64,
    pub unit_in: f64,
    pub latency: f64,
    pub scratch: f64,
    pub atomics: f64,
    pub epilogue: f64,
    pub barrier: f64,
    pub drain: f64,
    pub prologue: f64,
    pub ns: f64,
}

impl Phase {
    /// The resource that bounds the phase.
    pub fn bound(&self) -> &'static str {
        let v = [
            (self.compute, "compute"),
            (self.issue, "issue"),
            (self.dram, "dram"),
            (self.cache, "cache"),
            (self.unit_in, "unit_in"),
            (self.latency, "latency"),
            (self.scratch, "scratch"),
            (self.atomics, "atomics"),
            (self.epilogue, "epilogue"),
        ];
        v.iter()
            .fold((0.0, "none"), |b, &(x, n)| if x > b.0 { (x, n) } else { b })
            .1
    }
}

/// **One resource of a loop trip** (the cost model): its work a trip
/// (bytes, operations or flops), its rate (work a ns), the latency of one
/// use (ns: the loaded latency, the concurrency at the knee over the rate)
/// and the work the ring keeps in flight (∞: unbounded).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Flow {
    pub work: f64,
    pub rate: f64,
    pub latency: f64,
    pub fly: f64,
}

impl Flow {
    /// A flow bounded by its rate alone (unbounded in flight).
    pub fn rate_only(work: f64, rate: f64) -> Flow {
        Flow {
            work,
            rate,
            latency: 0.0,
            fly: f64::INFINITY,
        }
    }

    /// A time already priced, as a flow.
    pub fn time(ns: f64) -> Flow {
        Flow::rate_only(ns, 1.0)
    }

    /// Its time a trip: work ÷ rate where the ring keeps at least latency ×
    /// rate in flight, else latency × work ÷ in flight (Little's law: the
    /// larger of the two).
    pub fn ns(&self) -> f64 {
        if self.work <= 0.0 {
            return 0.0;
        }
        let thr = self.work / self.rate.max(1e-12);
        let lat = match self.fly {
            f if f.is_infinite() => 0.0,
            f => self.work * self.latency / f.max(1.0),
        };
        thr.max(lat)
    }
}

/// A trip's steady state: its slowest flow.
pub fn steady(flows: &[Flow]) -> f64 {
    flows.iter().map(Flow::ns).fold(0.0, f64::max)
}

/// **The pipeline calculation**: a loop of `trips` trips whose `compute`
/// and `movers` overlap — `fill` (the first use's latency) + trips × (the
/// steady state + `serial`, a trip's barriers) + the drain (the shorter of a
/// trip's compute and its movement: the last trip's other half). Monotone in
/// every rate and latency; with every ring unbounded and no fill or serial
/// term, within one trip of `trips` × the slowest resource (the max form).
pub fn pipeline(trips: f64, compute: &[Flow], movers: &[Flow], fill: f64, serial: f64) -> f64 {
    if trips <= 0.0 {
        return 0.0;
    }
    let (c, mv) = (steady(compute), steady(movers));
    fill + trips * (c.max(mv) + serial) + c.min(mv)
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
                    "{}:{:.0}[c{:.0} i{:.0} d{:.0} l2{:.0} u{:.0} lat{:.0} sh{:.0} at{:.0} st{:.0} bar{:.0} dr{:.0} pro{:.0}]",
                    p.bound(),
                    p.ns,
                    p.compute,
                    p.issue,
                    p.dram,
                    p.cache,
                    p.unit_in,
                    p.latency,
                    p.scratch,
                    p.atomics,
                    p.epilogue,
                    p.barrier,
                    p.drain,
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

/// **Which end of each fact group's intervals an evaluation reads** (D5:
/// facts with intervals are evaluated at their bounds, and a pick that flips
/// inside them is open — at mixed bounds too, a slow cache with a fast
/// memory): the operation (its rate, efficiency, latency, the clock, the
/// issue pipes), the device cache (its keep, rate and latency), the memory
/// behind it, and dispatch (start timing, launches, atomics, the zero pass).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Corner {
    pub compute: Bound,
    pub cache: Bound,
    pub memory: Bound,
    pub dispatch: Bound,
}

impl Corner {
    /// Every group at bound `b`.
    pub const fn all(b: Bound) -> Self {
        Corner {
            compute: b,
            cache: b,
            memory: b,
            dispatch: b,
        }
    }

    /// The sixteen corners: each group at its slow or its fast end (the
    /// all-slow corner first, the all-fast last).
    pub fn corners() -> Vec<Corner> {
        let side = |i: usize, bit: usize| match i >> bit & 1 {
            0 => Bound::Slow,
            _ => Bound::Fast,
        };
        (0..16)
            .map(|i| Corner {
                compute: side(i, 3),
                cache: side(i, 2),
                memory: side(i, 1),
                dispatch: side(i, 0),
            })
            .collect()
    }

    /// `compute/cache/memory/dispatch` as `s` (slow) or `f` (fast).
    pub fn name(&self) -> String {
        let c = |b: Bound| match b {
            Bound::Slow => 's',
            Bound::Value => 'v',
            Bound::Fast => 'f',
        };
        [self.compute, self.cache, self.memory, self.dispatch]
            .into_iter()
            .map(c)
            .collect()
    }
}

/// **Evaluate** schedule `x` of site `s` on machine `m` with its lowering's
/// declared counts `d`, every fact at bound `b`.
pub fn evaluate(m: &Machine, s: &Site, x: &Schedule, d: &Declared, b: Bound) -> Terms {
    evaluate_at(m, s, x, d, Corner::all(b), false)
}

/// [`evaluate`] at corner `c`: each fact group at its own bound.
pub fn evaluate_corner(m: &Machine, s: &Site, x: &Schedule, d: &Declared, c: Corner) -> Terms {
    evaluate_at(m, s, x, d, c, false)
}

/// **The evaluation's floor** (the search's cheap admissible bound, D5):
/// [`evaluate`] with the device cache missing every operand byte exactly
/// once — no term falls as misses rise (memory bytes, the miss ratio's
/// latency: a miss waits at least a hit's, and fill), so no schedule's time
/// is below it (`place_replay.rs` test `place_floor_is_admissible`, every
/// replay arm); it skips the reuse closed forms (a walk's per-reader
/// windows).
pub fn evaluate_floor(m: &Machine, s: &Site, x: &Schedule, d: &Declared, b: Bound) -> Terms {
    evaluate_at(m, s, x, d, Corner::all(b), true)
}

fn evaluate_at(m: &Machine, s: &Site, x: &Schedule, d: &Declared, c: Corner, floor: bool) -> Terms {
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
    let ghz = m.ghz(c.compute);
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
    let keep = keep_f.map_or(0, |k| k.rate(c.cache));
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
            r.total_bytes_per_ns.rate(c.memory)
        })
        .unwrap_or(1.0)
        .max(1e-9);
    let (cache_bw, unit_bw) = cache
        .map(|p| {
            let b = c.cache;
            let dev = m.reach(p, crate::machine::ExecKind::Device);
            let unit = m.reach(p, crate::machine::ExecKind::Unit);
            if dev.is_some_and(|r| r.total_bytes_per_ns.marked())
                || unit.is_some_and(|r| r.bytes_per_ns.marked())
            {
                mark(&mut t, "cache.bw".into());
            }
            (
                dev.map_or(f64::INFINITY, |r| r.total_bytes_per_ns.rate(b)),
                unit.map_or(f64::INFINITY, |r| r.bytes_per_ns.rate(b)),
            )
        })
        .unwrap_or((f64::INFINITY, f64::INFINITY));
    // Start timing: a measured start is read; an unmeasured one is the
    // worse of the two at the slow bound, the better at the fast one.
    let starts: Vec<Start> = match (m.dispatch.start_src.marked(), c.dispatch) {
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
        // The zero pass streams the output through the cache first.
        reuse::reads(&grid(order, zero), keep)
    };
    let out_bytes = batch * s.m * s.n * s.eo;
    // The zero pass's bytes: the whole output, or a hybrid's prefix where
    // its whole pieces store (the walked tiles are the raster order's first
    // `tiles − whole`: the lowering's layout); none without one.
    let zeroed = match x.trav {
        _ if !zero => 0,
        Traversal::Hybrid { h, .. } if d.whole_stores => {
            let whole = tiles / w * w;
            let h = h.max(1);
            let rows = (bm * h * (tiles - whole).div_ceil(gx * h)).min(batch * s.m);
            rows * s.n * s.eo
        }
        _ => out_bytes,
    };
    // Warm operands (resident from the previous call: `residence_plan`):
    // their first reads find them where they, and the zero pass streamed
    // through the cache ahead of the kernel, fit the keep — `a` first.
    let once_a = gy * panels * sa;
    let once_b = gx * panels * sb;
    let warm = {
        let mut held = zeroed;
        let mut w = [false; 2];
        for (i, once) in [(0, once_a), (1, once_b)] {
            if s.warm[i] && held + once <= keep {
                held += once;
                w[i] = true;
            }
        }
        w
    };
    // Per phase (a hybrid's waves and its walk; one phase else): the misses
    // by the closed form, or every operand slice once on the floor, less
    // each warm operand's first reads in that phase.
    let hybrid = match x.trav {
        Traversal::Hybrid { h, ctas, iters } => Some((h, ctas, iters)),
        _ => None,
    };
    let mut phase_reads: Vec<Reads> = match (floor, hybrid) {
        (true, None) => vec![Reads {
            a: once_a,
            b: once_b,
            out: tiles * so,
        }],
        (true, Some((h, ..))) => {
            let (wv, tl) = reuse::hybrid_first(
                &grid(
                    Order::Walk {
                        h,
                        ctas: 0,
                        iters: 0,
                    },
                    false,
                ),
                h,
            );
            vec![wv, tl]
        }
        (false, Some((h, ctas, iters))) => {
            let (wv, tl) = reuse::hybrid_split(&grid(Order::Hybrid { h, ctas, iters }, zero), keep);
            vec![wv, tl]
        }
        (false, None) => vec![
            starts
                .iter()
                .map(|&st| reads_of(st))
                .max_by_key(|r| match c.dispatch {
                    Bound::Fast => u64::MAX - r.operands(),
                    _ => r.operands(),
                })
                .unwrap_or_default(),
        ],
    };
    let firsts: Vec<Reads> = match hybrid {
        Some((h, ..)) => {
            let (wv, tl) = reuse::hybrid_first(
                &grid(
                    Order::Walk {
                        h,
                        ctas: 0,
                        iters: 0,
                    },
                    false,
                ),
                h,
            );
            vec![wv, tl]
        }
        None => vec![Reads {
            a: once_a,
            b: once_b,
            out: 0,
        }],
    };
    for (r, f) in phase_reads.iter_mut().zip(&firsts) {
        if warm[0] {
            r.a = r.a.saturating_sub(f.a);
        }
        if warm[1] {
            r.b = r.b.saturating_sub(f.b);
        }
    }
    let reads = phase_reads.iter().fold(Reads::default(), |a, r| Reads {
        a: a.a + r.a,
        b: a.b + r.b,
        out: a.out + r.out,
    });
    t.reads = reads;
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
    let rate_unit = op.rate.rate(c.compute) * ghz; // flops / ns
    let groups_per_sub = |slots: u64| (slots * d.groups) as f64 / m.subs().max(1) as f64;
    let eff_at = |t: &mut Terms, slots: u64| -> f64 {
        let (e, rule) = efficiency(m, x.op, d.chains as f64, groups_per_sub(slots), c.compute);
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
        for (class, n) in &d.issue {
            all += n;
            if let Some(p) = m.pipe(*class) {
                worst = worst.max(trips * n / (p.per_clk.rate(c.compute) * ghz));
            }
        }
        worst.max(trips * all / (m.subs().max(1) as f64 * ghz))
    };
    // ---- the cost model's primitives at their corners ----
    // A cost read from a marked fact is marked by its term; one borrowed
    // from a neighbouring measured row names that row (`term<-row`).
    let mark_cost = |t: &mut Terms, what: &str, c: &crate::machine::Cost| {
        let what = match c.borrowed {
            Some(row) => format!("{what}<-{row}"),
            None => what.to_string(),
        };
        if c.marked && !t.marked.iter().any(|x| *x == what) {
            t.marked.push(what);
        }
    };
    // The panels' mover from the cache (its unit rate, loaded latency) and
    // from the memory (loaded latency).
    let mover = |from: PlaceId, b: Bound| {
        let of = match d.mover {
            Move::Load(w) => PrimOf::Load { from, bytes: w },
            Move::Bulk(w) => PrimOf::Bulk { from, bytes: w },
            Move::Copy(w) => PrimOf::Copy { from, bytes: w },
            Move::Unsaid => PrimOf::Copy { from, bytes: 16 },
        };
        m.cost(of, b)
    };
    let (mv_cache, mv_mem) = match (cache, mem) {
        (Some(cp), Some(mp)) => (mover(cp, c.cache), mover(mp, c.memory)),
        (Some(cp), None) => (mover(cp, c.cache), mover(cp, c.cache)),
        (None, Some(mp)) => (mover(mp, c.memory), mover(mp, c.memory)),
        (None, None) => {
            let z = crate::machine::Cost {
                rate: f64::INFINITY,
                latency: 0.0,
                loaded: 0.0,
                marked: false,
                borrowed: None,
            };
            (z, z)
        }
    };
    mark_cost(&mut t, "prim.mover.cache", &mv_cache);
    mark_cost(&mut t, "prim.mover.memory", &mv_mem);
    // A miss's path is a hit's and then the memory's: a request that misses
    // waits at least what a hit waits (the memory's measured knee sits at
    // its own concurrency, the cache's at its own — on a part whose cache
    // knee lies further out, the RTX 3090's and the A100's, the memory's
    // loaded latency reads below the cache's). The latency the phase's
    // requests wait on so never falls as misses rise: the floor (misses at
    // their least) stays admissible.
    let miss_loaded = mv_mem.loaded.max(mv_cache.loaded);
    // The unit's ingress: its reach from the cache (one unit alone); the
    // device cache's feed: the mover's rate over the units, at most the
    // cache's device reach.
    let unit_rate = unit_bw;
    let feed_rate = (mv_cache.rate * units as f64).min(cache_bw);
    // The memory's rate for the operands' misses: the mover's stream from
    // the memory over the units, at most the memory's device reach. (The
    // panel stream's rate by run and rows, `PrimOf::Panel`, is a measured
    // fact no term reads: the replay contradicts it — cost-model.md.)
    let mem_rate = (mv_mem.rate * units as f64).min(dram_bw);
    let scratch = m.cost(
        PrimOf::Scratch {
            bytes: 16,
            matrix: d.scratch_matrix,
        },
        c.compute,
    );
    if d.scratch > 0 {
        mark_cost(&mut t, "prim.scratch", &scratch);
    }
    let bar = m.cost(PrimOf::Barrier { lanes: d.threads }, c.dispatch);
    if d.barriers > 0.0 {
        mark_cost(&mut t, "prim.barrier", &bar);
    }
    // The epilogue's stores: into the cache where the output fits its keep
    // (written back later), else to the memory.
    let out_fits = out_bytes <= keep;
    let store = match (out_fits, cache, mem) {
        (true, Some(p), _) | (false, _, Some(p)) => m.cost(
            PrimOf::Store {
                to: p,
                bytes: match d.store {
                    0 => 16,
                    w => w,
                },
            },
            if out_fits { c.cache } else { c.memory },
        ),
        _ => crate::machine::Cost {
            rate: dram_bw / units as f64,
            latency: 0.0,
            loaded: 0.0,
            marked: true,
            borrowed: None,
        },
    };
    let red = m.cost(
        PrimOf::Reduce {
            at: cache.unwrap_or(0),
            sharing: 1,
            sectors: match d.add_sectors {
                0 => 4,
                x => x,
            },
        },
        c.dispatch,
    );
    // ---- the pipeline: a phase of `blocks` slots of `trips` trips ----
    // `miss`: the share of the bytes the phase's requests wait on together
    // that come from memory (the loaded latency between the cache's and the
    // memory's); `mem_a` / `mem_b`: each operand's bytes from memory a slot
    // a trip; `ends`: a block's result movement — tiles stored, cells added,
    // pieces (each a fill).
    #[derive(Clone, Copy)]
    struct Ends {
        stores: f64,
        adds: f64,
        pieces: f64,
    }
    let pb = (sa + sb) as f64;
    let phase = |t: &mut Terms,
                 blocks: u64,
                 trips: u64,
                 flops_trip: f64,
                 miss: f64,
                 mem_ab: [f64; 2],
                 ends: Ends,
                 floor_bytes: f64|
     -> Phase {
        let nb = blocks.div_ceil(units); // the busiest unit's slots
        let r = u.min(nb).max(1);
        let lat = (1.0 - miss) * mv_cache.loaded + miss * miss_loaded;
        let mut ph = Phase::default();
        // Whole waves of `r` slots a unit, then the tail's.
        let per_wave = units * r;
        let full = blocks / per_wave;
        let rest = blocks % per_wave;
        // (count, slots a unit, slots in all)
        let waves = [
            (full, r, per_wave),
            (u64::from(rest > 0), rest.div_ceil(units), rest),
        ];
        let tr = trips as f64;
        let mut last = 0.0;
        for (count, rw, sw) in waves.into_iter().filter(|w| w.0 > 0) {
            let k = count as f64;
            let eff = eff_at(t, rw);
            t.eff = eff;
            let rwf = rw as f64;
            let swf = sw as f64;
            // Compute and issue: the unit's.
            let compute = Flow::rate_only(rwf * flops_trip, rate_unit * eff);
            let issue = Flow::time(issue_ns(rwf));
            // The panels into the unit: the mover's rate, its loaded latency
            // against the ring's bytes in flight.
            let unit_in = Flow::rate_only(rwf * pb, unit_rate);
            let fly = Flow {
                work: rwf * pb,
                rate: f64::INFINITY,
                latency: lat,
                fly: (rw * d.in_flight).max(1) as f64,
            };
            // The device cache's feed.
            let l2 = Flow::rate_only(swf * pb, feed_rate);
            // The scratchpad's fragment loads.
            let sh = Flow {
                work: rwf * d.scratch as f64,
                rate: scratch.rate,
                latency: scratch.loaded,
                fly: match d.scratch_fly {
                    0 => f64::INFINITY,
                    f => (rw * f) as f64,
                },
            };
            let serial = d.barriers * bar.latency;
            // Each piece's fill: the ring's first stage at its loaded latency.
            let fill = ends.pieces * lat;
            // The result's movement — the epilogue's stores (into the cache
            // where the output fits its keep, else to the memory) and the
            // partial adds (the reductions' rate; past the keep their
            // read-modify-write in the memory) — is issued and left (a store,
            // a `red`): a throughput beside the next wave's loops, a trip's
            // share of it, the memory's bytes in the memory's flow; only the
            // last wave's waits (the phase's drain, below).
            let st_bytes = swf * ends.stores * (so as f64);
            let st = st_bytes / (store.rate * units as f64).max(1e-12);
            let add_ops = swf * ends.adds;
            let adds = add_ops / (red.rate * units as f64).max(1e-12);
            let mem_bytes = match out_fits {
                true => 0.0,
                false => st_bytes + add_ops * 2.0 * s.eo as f64,
            };
            let dram = Flow::time(
                swf * (mem_ab[0] + mem_ab[1]) / mem_rate + mem_bytes / tr.max(1.0) / dram_bw,
            );
            let result = Flow::time((st + adds) / tr.max(1.0));
            let moving = [unit_in, fly, l2, dram, sh, result];
            let loop_ns = pipeline(tr, &[compute, issue], &moving, 0.0, serial);
            let wave = fill + loop_ns;
            // What is left at the wave's end: a lockstep wave's whole result
            // (its slots end together), a walk's blocks' last pieces only
            // (their pieces end at every offset: `1 / pieces` of it).
            let tail = match ends.pieces > 1.0 {
                true => 1.0 / ends.pieces,
                false => 1.0,
            };
            last = tail * (st + adds)
                + if st > 0.0 { store.latency } else { 0.0 }
                + if adds > 0.0 { red.latency } else { 0.0 };
            ph.ns += k * wave;
            ph.compute += k * tr * compute.ns();
            ph.issue += k * tr * issue.ns();
            ph.unit_in += k * tr * unit_in.ns();
            ph.latency += k * tr * fly.ns();
            ph.cache += k * tr * l2.ns();
            ph.dram += k * tr * dram.ns();
            ph.scratch += k * tr * sh.ns();
            ph.barrier += k * tr * serial;
            ph.drain += k * steady(&[compute, issue]).min(steady(&moving));
            ph.prologue += k * fill;
            ph.epilogue += k * st;
            ph.atomics += k * adds;
        }
        // The drain: the last wave's result, waited on (unless the lowering
        // overlaps its epilogue with the loop).
        if !d.epi_overlap {
            ph.ns += last;
        }
        // Never below the phase's memory bytes at the memory's rate.
        let floor_ns = floor_bytes / dram_bw;
        ph.dram = ph.dram.max(floor_ns);
        ph.ns = ph.ns.max(floor_ns);
        ph
    };
    // The operation's work a trip: the useful flops, or what the lowering
    // declares it issues (a split form's passes).
    let flops_trip = match d.work {
        w if w > 0.0 => w,
        _ => (2 * bm * bn * bk) as f64,
    };
    let miss_of = |cache_reads: f64, needed: f64| (cache_reads / needed.max(1.0)).clamp(0.0, 1.0);
    let per_trip = |r: &Reads, blocks: u64, trips: u64| {
        let n = (blocks * trips).max(1) as f64;
        [r.a as f64 / n, r.b as f64 / n]
    };
    let (bmn, epi_out) = ((bm * bn) as f64, (out_bytes + epi_bytes) as f64);
    match x.trav {
        Traversal::Ordered { .. } | Traversal::Split { .. } | Traversal::Persistent { .. } => {
            let blocks = tiles * s_parts;
            let trips = panels / s_parts;
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
            let ends = match (x.trav, atomic) {
                (Traversal::Split { .. }, true) => Ends {
                    stores: 0.0,
                    adds: bmn,
                    pieces: 1.0,
                },
                _ => Ends {
                    stores: 1.0,
                    adds: 0.0,
                    pieces: 1.0,
                },
            };
            let p = phase(
                &mut t,
                blocks,
                trips,
                flops_trip,
                miss_of(reads.operands() as f64, needed as f64),
                per_trip(&reads, blocks, trips),
                ends,
                dram_bytes as f64,
            );
            t.phases.push(p);
        }
        Traversal::Walk { ctas, iters, .. } => {
            let requested = (tiles * panels * (sa + sb)) as f64;
            let ends = walk_ends(tiles, panels, iters, ctas, floor, bmn, d.whole_stores);
            let p = phase(
                &mut t,
                ctas,
                iters,
                flops_trip,
                miss_of(reads.operands() as f64, requested),
                per_trip(&reads, ctas, iters),
                Ends {
                    stores: ends.0,
                    adds: ends.1,
                    pieces: ends.2,
                },
                dram_bytes as f64,
            );
            t.phases.push(p);
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
                let r = phase_reads[0];
                let p = phase(
                    &mut t,
                    whole,
                    panels,
                    flops_trip,
                    miss_of(r.operands() as f64, needed * frac(whole)),
                    per_trip(&r, whole, panels),
                    Ends {
                        stores: 1.0,
                        adds: 0.0,
                        pieces: 1.0,
                    },
                    r.operands() as f64 + epi_out * frac(whole),
                );
                t.phases.push(p);
            }
            if tail > 0 {
                let r = phase_reads[1];
                let requested = (tail * panels * (sa + sb)) as f64;
                let ends = walk_ends(tail, panels, iters, ctas, floor, bmn, d.whole_stores);
                let p = phase(
                    &mut t,
                    ctas,
                    iters,
                    flops_trip,
                    miss_of(r.operands() as f64, requested),
                    per_trip(&r, ctas, iters),
                    Ends {
                        stores: ends.0,
                        adds: ends.1,
                        pieces: ends.2,
                    },
                    r.operands() as f64 + epi_out * frac(tail),
                );
                t.phases.push(p);
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
        t.zero = m.dispatch.zero_ns.cost(c.dispatch)
            + zeroed as f64 / m.dispatch.zero_bytes_per_ns.rate(c.dispatch).max(1e-9);
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
        t.fixed = launches as f64 * l.ns.cost(c.dispatch)
            + blocks.saturating_sub(units) as f64 * l.per_slot_ps.cost(c.dispatch) / 1e3;
    }
    t.ns = t.phases.iter().map(|p| p.ns).sum::<f64>() + t.combine + t.zero + t.fixed;
    t
}

/// A walk's result movement a block: tiles it stores whole (a lowering whose
/// whole pieces store), cells it adds (each shared tile's pieces; every
/// piece where whole pieces add too), and the pieces it crosses (each a
/// fill) — over `ctas` blocks of `iters` units, `tiles` tiles of `panels`
/// (on the floor no tile is shared: no adds).
fn walk_ends(
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
