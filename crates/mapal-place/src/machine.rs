//! **The machine** (ADR-0050 D2): places, executors, operations, movers and
//! dispatch as facts. Nothing here assumes a GPU: a target defines the depth of
//! its own executor tree and the graph of its places, and its translation
//! (`Machine::of` in a backend) is the only target-specific code placement
//! reads. No field names a part, an architecture or an instruction-set version.

use crate::fact::{Bound, Fact, Row, Src};

/// Index of an executor level in [`Machine::exec`].
pub type ExecId = usize;
/// Index of a place in [`Machine::places`].
pub type PlaceId = usize;
/// Index of an operation row in [`Machine::ops`].
pub type OpId = usize;

/// What an executor level is, as placement reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecKind {
    /// Several machines joined by links.
    System,
    /// One machine (a GPU, a CPU socket, an ASIC).
    Device,
    /// The level work is dispatched to (an SM, a core, a PE tile).
    Unit,
    /// A unit's issue partition (a warp scheduler, an SMT thread).
    Sub,
    /// The lanes that issue together (a warp, a vector register).
    Group,
    /// One lane.
    Lane,
}

/// One level of the executor tree: `per_parent` instances under each instance
/// of its parent.
#[derive(Clone, Debug, PartialEq)]
pub struct Exec {
    pub name: &'static str,
    pub kind: ExecKind,
    pub parent: Option<ExecId>,
    pub per_parent: Fact<u64>,
}

/// An operand or accumulator format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fmt {
    F64,
    F32,
    /// f32 storage read with 10 explicit significand bits.
    Tf32,
    Bf16,
    F16,
    E4m3,
    E5m2,
}

impl Fmt {
    /// Bytes one element of this format occupies as an operand.
    pub fn bytes(self) -> u64 {
        match self {
            Fmt::F64 => 8,
            Fmt::F32 | Fmt::Tf32 => 4,
            Fmt::Bf16 | Fmt::F16 => 2,
            Fmt::E4m3 | Fmt::E5m2 => 1,
        }
    }
}

/// An operand's storage major order at a place: along `k` (K-major) or
/// along its own free axis (MN-major).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Major {
    K,
    Mn,
}

/// An operand's layout at a home (D3, D6): its major order, row padding in
/// elements and swizzle span in bytes (0: none). The banking cost of a layout
/// is a place fact ([`Banking`]); only the bit encoding is lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Layout {
    pub major: Major,
    pub pad: u64,
    pub swizzle: u64,
}

impl Layout {
    pub const K: Layout = Layout {
        major: Major::K,
        pad: 0,
        swizzle: 0,
    };
    pub const MN: Layout = Layout {
        major: Major::Mn,
        pad: 0,
        swizzle: 0,
    };
}

/// The `n` extents an operation takes per instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NSet {
    One(u64),
    /// Every multiple of `step` up to `max`.
    Step {
        step: u64,
        max: u64,
    },
}

impl NSet {
    pub fn admits(self, n: u64) -> bool {
        match self {
            NSet::One(x) => n == x,
            NSet::Step { step, max } => n > 0 && n <= max && n.is_multiple_of(step),
        }
    }
    pub fn min(self) -> u64 {
        match self {
            NSet::One(x) => x,
            NSet::Step { step, .. } => step,
        }
    }
}

/// One point of an operation's measured **efficiency curve** (D2): its
/// achieved fraction of the row's rate with `chains` independent accumulator
/// chains a lane (a group tile's accumulators per instruction) and `groups`
/// resident groups a scheduler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffPoint {
    pub chains: f64,
    pub groups: f64,
    pub eff: Fact<f64>,
}

/// **An operation as a row of geometry** (D2): any executor level's
/// instruction — the FMA lane is the `1 × 1 × 1` row, a warp matrix product,
/// a warpgroup product, a unit-pair product, a systolic array are rows of the
/// same table.
#[derive(Clone, Debug, PartialEq)]
pub struct Op {
    pub name: &'static str,
    /// The executor level that issues it (a lane, a group, …) and the lanes
    /// that issue one instruction together.
    pub exec: ExecId,
    pub lanes: u64,
    pub m: u64,
    pub n: NSet,
    pub k: u64,
    pub fa: Fmt,
    pub fb: Fmt,
    pub facc: Fmt,
    /// An auxiliary operand (block scales), if any.
    pub aux: Option<Fmt>,
    /// The places each of `a`, `b`, the accumulator may live in when it
    /// issues (registers, shared memory, tensor memory).
    pub homes: [&'static [PlaceId]; 3],
    /// Each operand's majors at its home (a descriptor operand may be K-major
    /// only).
    pub majors: [&'static [Major]; 2],
    /// 32-bit registers a lane holds per instruction's `a`, `b` and
    /// accumulator fragment (the fragment footprint, D2).
    pub frag: [u64; 3],
    /// Dependent-issue latency, unit clocks.
    pub latency: Fact<f64>,
    /// Instructions in flight before a wait (asynchronous issue).
    pub async_depth: u64,
    /// Flops one unit issues a clock through this row.
    pub rate: Fact<f64>,
    /// Measured efficiency points (empty: none measured — the evaluator
    /// takes the latency rule alone and marks it).
    pub eff: Vec<EffPoint>,
    /// Group widths (lanes) it issues at.
    pub widths: &'static [u64],
    /// Accumulator numerics (ADR-0043): unit roundoff `2^acc_u` and alignment
    /// guard bits.
    pub acc_u: Fact<i32>,
    pub acc_g: Fact<u32>,
}

impl Op {
    /// Flops of one instruction.
    pub fn flops(&self, n: u64) -> u64 {
        2 * self.m * n * self.k
    }
}

/// Shared-memory style banking (D2): `banks` banks of `width` bytes, a
/// broadcast of one word to every lane reading it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Banking {
    pub banks: u64,
    pub width: u64,
    pub broadcast: bool,
}

/// How executors (or addresses) map onto a cache's instances.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstanceMap {
    /// One instance every executor reaches.
    One,
    /// Instances by address interleave; every executor reaches every
    /// instance (near or far edges).
    Address { interleave: u64 },
    /// Instances by executor: `per` units share one.
    Executor { per: u64 },
}

/// A cache's measured **keep**: the largest working set it holds for a
/// cyclic, a rotated (a walk's) and a strided pattern — never the nominal
/// size — and its hit-against-distance curve (distance ÷ cyclic keep → hit
/// fraction; empty: a step at the keep).
#[derive(Clone, Debug, PartialEq)]
pub struct Keep {
    pub cyclic: Fact<u64>,
    pub rotated: Option<Fact<u64>>,
    pub strided: Option<Fact<u64>>,
    pub curve: Vec<(f64, f64)>,
}

impl Keep {
    /// The keep a pattern sees (the cyclic keep where its own is unmeasured).
    pub fn of(&self, p: Pattern) -> Fact<u64> {
        match p {
            Pattern::Cyclic => self.cyclic,
            Pattern::Rotated => self.rotated.unwrap_or(self.cyclic),
            Pattern::Strided => self.strided.unwrap_or(self.cyclic),
        }
    }
}

/// An access pattern a keep is measured for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    Cyclic,
    Rotated,
    Strided,
}

/// A per-access cache policy (a mover option).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Policy {
    Default,
    /// The line is the first to evict (streaming).
    EvictFirst,
    /// The line persists against other traffic (a persisting window).
    Persist,
}

/// A cache (D2): its instances, the map onto them, its replication rule (a
/// line read through two instances is held in both), its line and its keep.
#[derive(Clone, Debug, PartialEq)]
pub struct Cache {
    pub instances: Fact<u64>,
    pub map: InstanceMap,
    pub replicate: bool,
    pub line: u64,
    pub keep: Keep,
    pub policies: Vec<Policy>,
}

/// How a place is managed.
#[derive(Clone, Debug, PartialEq)]
pub enum Mgmt {
    /// A scratchpad the program addresses (registers, shared memory, tensor
    /// memory, an ASIC's SRAM, DRAM itself).
    Explicit,
    Cache(Box<Cache>),
}

/// An executor's edge to a place: the bandwidth one instance of `from`
/// reaches it at, the aggregate over every instance, the latency under load
/// and the transaction granule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reach {
    pub from: ExecId,
    pub bytes_per_ns: Fact<f64>,
    pub total_bytes_per_ns: Fact<f64>,
    pub latency_ns: Fact<f64>,
    pub granule: u64,
}

/// **A place** (D2): where bytes can live — one instance per instance of
/// executor level `exec` (registers a lane, shared memory a unit, an L2 a
/// device). Places form a graph: `backing` is where a miss or a spill goes;
/// `reach` the executors that address it.
#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    pub name: &'static str,
    pub exec: ExecId,
    pub capacity: Fact<u64>,
    pub mgmt: Mgmt,
    pub banking: Option<Banking>,
    /// Allocation granule (bytes) and the reserve each slot takes.
    pub granule: u64,
    pub reserve: u64,
    /// The most one slot may hold (0: the capacity).
    pub per_slot: u64,
    pub backing: Option<PlaceId>,
    pub reach: Vec<Reach>,
}

impl Place {
    pub fn cache(&self) -> Option<&Cache> {
        match &self.mgmt {
            Mgmt::Cache(c) => Some(c),
            Mgmt::Explicit => None,
        }
    }
}

/// **A capacity group** (D2): one physical capacity split between places
/// per launch (shared memory against L1); the schedule chooses the split
/// among `shares` of the first place.
#[derive(Clone, Debug, PartialEq)]
pub struct CapacityGroup {
    pub places: Vec<PlaceId>,
    pub total: Fact<u64>,
    pub shares: Vec<u64>,
}

/// A unit resource a slot (a dispatched block) occupies: its capacity a unit,
/// the most one slot may take, the most one lane may take, the allocation
/// granule (in the resource's own unit, per group of lanes) and the reserve a
/// slot costs. Occupancy is derived from a schedule's footprint against these
/// (D2: never a constant).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Resource {
    pub what: Res,
    pub cap: Fact<u64>,
    pub per_slot: u64,
    pub per_lane: u64,
    pub granule: u64,
    pub reserve: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Res {
    /// 32-bit registers.
    Regs,
    Threads,
    Slots,
    /// Bytes of an explicit place (its id).
    Bytes(PlaceId),
}

/// **The machine**: its executor tree, its places and capacity groups, its
/// unit resources and issue pipes, its operation rows, movers and dispatch.
/// `label` is for display only: no decision reads it (D1).
#[derive(Clone, Debug, PartialEq)]
pub struct Machine {
    pub label: String,
    pub exec: Vec<Exec>,
    pub places: Vec<Place>,
    pub groups: Vec<CapacityGroup>,
    pub res: Vec<Resource>,
    pub pipes: Vec<Pipe>,
    pub ops: Vec<Op>,
    pub movers: Vec<Mover>,
    pub dispatch: Dispatch,
    /// The cost model's primitives measured on the part (empty: every cost
    /// derived from the places' reaches and dispatch — [`Machine::cost`]).
    pub prims: Vec<Prim>,
}

impl Machine {
    /// The first executor level of `kind`.
    pub fn level(&self, kind: ExecKind) -> Option<ExecId> {
        self.exec.iter().position(|e| e.kind == kind)
    }

    /// Instances of executor level `e` in one instance of level `above`
    /// (the product of `per_parent` from `e` up to `above`, exclusive).
    pub fn count(&self, e: ExecId, above: ExecId) -> u64 {
        let mut n = 1;
        let mut at = Some(e);
        while let Some(i) = at {
            if i == above {
                return n;
            }
            n *= self.exec[i].per_parent.v;
            at = self.exec[i].parent;
        }
        n
    }

    /// Dispatch units on the device.
    pub fn units(&self) -> u64 {
        let d = self.level(ExecKind::Device).unwrap_or(0);
        self.count(self.dispatch.unit, d)
    }

    /// Lanes of one group (a warp).
    pub fn group_lanes(&self) -> u64 {
        match (self.level(ExecKind::Lane), self.level(ExecKind::Group)) {
            (Some(l), Some(g)) => self.count(l, g),
            _ => 1,
        }
    }

    /// Issue partitions (schedulers) of one unit.
    pub fn subs(&self) -> u64 {
        match self.level(ExecKind::Sub) {
            Some(s) => self.count(s, self.dispatch.unit),
            None => 1,
        }
    }

    pub fn resource(&self, what: Res) -> Option<&Resource> {
        self.res.iter().find(|r| r.what == what)
    }

    /// The first place whose instances belong to executor level `kind` and
    /// that is managed `explicit` (or as a cache).
    pub fn place_at(&self, kind: ExecKind, explicit: bool) -> Option<PlaceId> {
        self.places.iter().position(|p| {
            self.exec[p.exec].kind == kind && matches!(p.mgmt, Mgmt::Explicit) == explicit
        })
    }

    /// The device-level cache operands stream through, if any.
    pub fn device_cache(&self) -> Option<PlaceId> {
        self.place_at(ExecKind::Device, false)
    }

    /// The device-level explicit place operands live in (device memory).
    pub fn device_memory(&self) -> Option<PlaceId> {
        self.place_at(ExecKind::Device, true)
    }

    /// The unit's explicit scratchpad (shared memory), if any.
    pub fn unit_scratch(&self) -> Option<PlaceId> {
        self.place_at(ExecKind::Unit, true)
    }

    /// The lane's registers.
    pub fn lane_regs(&self) -> Option<PlaceId> {
        self.place_at(ExecKind::Lane, true)
    }

    /// An edge from executor level `from` to place `p`.
    pub fn reach(&self, p: PlaceId, from: ExecKind) -> Option<&Reach> {
        self.places[p]
            .reach
            .iter()
            .find(|r| self.exec[r.from].kind == from)
    }

    pub fn pipe(&self, class: Class) -> Option<&Pipe> {
        self.pipes.iter().find(|p| p.class == class)
    }

    pub fn launch(&self, mode: LaunchMode) -> Option<&LaunchCost> {
        self.dispatch
            .launches
            .iter()
            .find(|l| l.mode == mode)
            .or_else(|| {
                self.dispatch
                    .launches
                    .iter()
                    .find(|l| l.mode == LaunchMode::Plain)
            })
    }

    /// The unit clock, GHz, at bound `b`.
    pub fn ghz(&self, b: Bound) -> f64 {
        self.dispatch.clock_khz.rate(b) / 1e6
    }

    /// **Occupancy, derived from a footprint** (D2): slots of `threads` lanes
    /// at `regs` registers a lane and `bytes` of each explicit unit place one
    /// unit holds — the minimum over its resources, each at its allocation
    /// granule and per-slot reserve. 0: the footprint does not fit one slot.
    pub fn occupancy(&self, threads: u64, regs: u64, bytes: &[(PlaceId, u64)]) -> u64 {
        let lanes = self.group_lanes().max(1);
        let mut best = u64::MAX;
        for r in &self.res {
            let cap = r.cap.v;
            let need = match r.what {
                Res::Slots => 1,
                Res::Threads => threads,
                Res::Regs => {
                    if r.per_lane > 0 && regs > r.per_lane {
                        return 0;
                    }
                    // Registers are allocated a group at a time, rounded to
                    // the granule.
                    let groups = threads.div_ceil(lanes);
                    let per_group = (regs * lanes).next_multiple_of(r.granule.max(1));
                    groups * per_group
                }
                Res::Bytes(p) => {
                    let b = bytes.iter().find(|(q, _)| *q == p).map_or(0, |(_, b)| *b);
                    b.next_multiple_of(r.granule.max(1)) + r.reserve
                }
            };
            if r.per_slot > 0 && need.saturating_sub(r.reserve) > r.per_slot {
                return 0;
            }
            if need == 0 {
                continue;
            }
            best = best.min(cap / need);
        }
        if best == u64::MAX { 0 } else { best }
    }

    /// Every numeric fact with its provenance, one row each (the evidence
    /// listing, D8; a marked row is a fact a pick may be open on).
    pub fn rows(&self) -> Vec<Row> {
        let mut out = Vec::new();
        let mut put = |name: String, f: Fact<f64>| {
            out.push(Row {
                name,
                v: f.v,
                lo: f.lo,
                hi: f.hi,
                src: f.src,
            })
        };
        let u = |f: Fact<u64>| Fact {
            v: f.v as f64,
            lo: f.lo as f64,
            hi: f.hi as f64,
            src: f.src,
        };
        for e in &self.exec {
            put(format!("exec.{}.per_parent", e.name), u(e.per_parent));
        }
        for r in &self.res {
            put(format!("unit.{:?}.cap", r.what), u(r.cap));
        }
        for p in &self.pipes {
            put(format!("pipe.{:?}.per_clk", p.class), p.per_clk);
        }
        for p in &self.places {
            put(format!("place.{}.capacity", p.name), u(p.capacity));
            if let Some(c) = p.cache() {
                put(format!("place.{}.instances", p.name), u(c.instances));
                put(format!("place.{}.keep.cyclic", p.name), u(c.keep.cyclic));
                if let Some(k) = c.keep.rotated {
                    put(format!("place.{}.keep.rotated", p.name), u(k));
                }
                if let Some(k) = c.keep.strided {
                    put(format!("place.{}.keep.strided", p.name), u(k));
                }
            }
            for r in &p.reach {
                let from = self.exec[r.from].name;
                put(format!("place.{}.reach.{from}.bw", p.name), r.bytes_per_ns);
                put(
                    format!("place.{}.reach.{from}.total_bw", p.name),
                    r.total_bytes_per_ns,
                );
                put(
                    format!("place.{}.reach.{from}.lat_ns", p.name),
                    r.latency_ns,
                );
            }
        }
        for o in &self.ops {
            put(format!("op.{}.rate", o.name), o.rate);
            put(format!("op.{}.latency", o.name), o.latency);
            for (i, e) in o.eff.iter().enumerate() {
                put(
                    format!(
                        "op.{}.eff[{i}] chains={} groups={}",
                        o.name, e.chains, e.groups
                    ),
                    e.eff,
                );
            }
        }
        for m in &self.movers {
            match &m.kind {
                MoverKind::Bulk {
                    copy_ns, in_flight, ..
                } => {
                    put(format!("mover.{}.copy_ns", m.name), *copy_ns);
                    put(format!("mover.{}.in_flight", m.name), u(*in_flight));
                }
                MoverKind::Link {
                    bytes_per_ns,
                    pageable_pm,
                    ..
                } => {
                    put(format!("mover.{}.bw", m.name), *bytes_per_ns);
                    put(format!("mover.{}.pageable_pm", m.name), u(*pageable_pm));
                }
                _ => {}
            }
        }
        let d = &self.dispatch;
        put(
            format!("dispatch.start={:?}", d.start),
            Fact::point(0.0, d.start_src),
        );
        for l in &d.launches {
            put(format!("dispatch.launch.{:?}.ns", l.mode), l.ns);
            put(
                format!("dispatch.launch.{:?}.per_slot_ps", l.mode),
                l.per_slot_ps,
            );
        }
        put("dispatch.issue_ns".into(), d.issue_ns);
        put("dispatch.gap_ns".into(), d.gap_ns);
        for a in &d.atomics {
            put(
                format!("dispatch.atomics.{:?}.ops_per_ns", a.fmt),
                a.ops_per_ns,
            );
        }
        put("dispatch.zero_ns".into(), d.zero_ns);
        put("dispatch.zero_bytes_per_ns".into(), d.zero_bytes_per_ns);
        put("dispatch.clock_khz".into(), d.clock_khz);
        put("dispatch.barrier_clk".into(), d.barrier_clk);
        if let Some(g) = d.grid_sync_ns {
            put("dispatch.grid_sync_ns".into(), g);
        }
        for p in &self.prims {
            put(format!("prim.{}.rate", p.name), p.rate);
            put(format!("prim.{}.latency", p.name), p.latency);
            put(format!("prim.{}.conc", p.name), p.conc);
        }
        out
    }
}

mod cost;
mod prims;

pub use self::prims::*;
