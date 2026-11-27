//! The cost model's result types: phases, flows, terms, efficiency and the bound corners.

use super::*;

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
