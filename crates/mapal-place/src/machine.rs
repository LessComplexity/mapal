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

/// An instruction class with an issue rate (D2: issue rates per instruction
/// class are unit facts).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    /// Scalar / vector arithmetic (FFMA).
    Math,
    /// Matrix products.
    Matrix,
    /// Shared-memory loads and stores.
    Shared,
    /// Global loads, stores and asynchronous copies.
    Global,
    Barrier,
    Convert,
    /// Address arithmetic and the loop.
    Other,
}

/// A unit's issue pipe: group-instructions of `class` it issues a clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pipe {
    pub class: Class,
    pub per_clk: Fact<f64>,
}

/// What kind of mover.
#[derive(Clone, Debug, PartialEq)]
pub enum MoverKind {
    /// Program-issued lane copies of `bytes` a lane; `asynchronous`: no
    /// register round trip (completion by group wait).
    Lane { bytes: u64, asynchronous: bool },
    /// A bulk-copy engine: one lane issues a box of up to `box_dims`
    /// dimensions of `box_side` elements, in swizzle spans `swizzles`, its
    /// completion counted on a barrier; `copy_ns` per copy, `in_flight`
    /// copies overlapped, multicast to `cluster` slots.
    Bulk {
        granule: u64,
        box_dims: u64,
        box_side: u64,
        swizzles: Vec<u64>,
        tx_bytes: u64,
        cluster: u64,
        multicast: bool,
        copy_ns: Fact<f64>,
        in_flight: Fact<u64>,
    },
    /// A hardware prefetch stream: the run that locks it on and its reach.
    Prefetch { run: u64, reach: u64 },
    /// A link between machines: its rate (bytes per ns) by source, the
    /// pageable rate's share of the pinned one, and its copy engines.
    Link {
        bytes_per_ns: Fact<f64>,
        pageable_pm: Fact<u64>,
        engines: u64,
    },
}

/// **A mover** (D2): how bytes move from one place to another, who issues
/// it, whether it converts formats on the way, and its cache policies.
#[derive(Clone, Debug, PartialEq)]
pub struct Mover {
    pub name: &'static str,
    pub from: PlaceId,
    pub to: PlaceId,
    pub issuer: ExecId,
    pub kind: MoverKind,
    pub converts: bool,
    pub policies: Vec<Policy>,
}

/// The start timing of a grid's slots beyond its first wave (D2: measured).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Start {
    /// A wave's slots run in step: a slot starts when its wave's do.
    Lockstep,
    /// A slot starts when one before it ends: the starts spread over a
    /// slot's run.
    Stagger,
}

/// How a grid is launched (D3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaunchMode {
    Plain,
    /// Starts while the one before drains (programmatic dependent launch).
    Dependent,
    /// Every slot co-resident (a persistent grid).
    Cooperative,
    /// Slots in clusters of the given size.
    Cluster(u64),
}

/// A launch mode's device cost: start to start of dependent launches up to
/// one slot a unit, and what each slot beyond adds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LaunchCost {
    pub mode: LaunchMode,
    pub ns: Fact<f64>,
    pub per_slot_ps: Fact<f64>,
}

/// Atomic adds as a resource (D4) at the place that resolves them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Atomics {
    pub at: PlaceId,
    pub fmt: Fmt,
    pub ops_per_ns: Fact<f64>,
}

/// **Dispatch** (D2): how work reaches executors.
#[derive(Clone, Debug, PartialEq)]
pub struct Dispatch {
    /// The executor level a dispatched slot runs on.
    pub unit: ExecId,
    /// Start timing of slots past the first wave, and its source.
    pub start: Start,
    pub start_src: Src,
    /// Slots go round-robin over the units (the grid index's map).
    pub round_robin: bool,
    pub launches: Vec<LaunchCost>,
    /// Host issue time of one launch, and the device's idle time between two
    /// dependent ones.
    pub issue_ns: Fact<f64>,
    pub gap_ns: Fact<f64>,
    pub atomics: Vec<Atomics>,
    /// A zero pass: its fixed time and its fill rate.
    pub zero_ns: Fact<f64>,
    pub zero_bytes_per_ns: Fact<f64>,
    /// The unit clock under sustained load, kHz.
    pub clock_khz: Fact<f64>,
    /// One slot barrier, unit clocks.
    pub barrier_clk: Fact<f64>,
    pub grid_sync_ns: Option<Fact<f64>>,
    /// The jitter of a static round-robin over persistent slots (None:
    /// unmeasured — persistent traversals stay out of the space).
    pub rr_jitter: Option<Fact<f64>>,
}

/// **What a primitive is** (the cost model, ADR-0050): an operation row, a
/// mover's use from a place, a result's movement, a synchronization or a
/// launch — keyed by what it does and the place it reaches, never by an
/// instruction's or a part's name (the row's `name` is the probe's, for the
/// evidence).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrimOf {
    /// An operation row (compute): flops a clock a unit.
    Op(OpId),
    /// A lane's synchronous load of `bytes` from `from` into registers.
    Load {
        from: PlaceId,
        bytes: u64,
    },
    /// A lane's asynchronous copy of `bytes` from `from` into the unit
    /// scratchpad.
    Copy {
        from: PlaceId,
        bytes: u64,
    },
    /// A bulk-copy engine's box of `bytes` from `from`.
    Bulk {
        from: PlaceId,
        bytes: u64,
    },
    /// A tile's panel stream from `from`: `run` contiguous bytes of each of
    /// `rows` rows a unit streams at once, a trip, `fly` bytes in flight a
    /// unit (the memory's pages; a line's sectors missing at once).
    Panel {
        from: PlaceId,
        run: u64,
        rows: u64,
        fly: u64,
    },
    /// Unit-scratchpad loads of `bytes` a lane (`matrix`: the fragment
    /// loader's tiles); bytes a clock a unit.
    Scratch {
        bytes: u64,
        matrix: bool,
    },
    /// A lane's stores of `bytes` to `to`.
    Store {
        to: PlaceId,
        bytes: u64,
    },
    /// A tile staged in the unit scratchpad and stored to `to` (an
    /// epilogue's).
    Staged {
        to: PlaceId,
    },
    /// Reductions resolved at `at`: `sharing` lanes on one address (1: a
    /// lane its own; 0: every lane of the grid on a few), `sectors` 32-byte
    /// sectors one group instruction touches (4: a group's lanes contiguous;
    /// 8: a matrix accumulator's fragment rows; 32: a sector a lane).
    Reduce {
        at: PlaceId,
        sharing: u64,
        sectors: u64,
    },
    /// A slot barrier over `lanes` lanes (clocks).
    Barrier {
        lanes: u64,
    },
    /// A transaction barrier's phase (clocks).
    Phase,
    /// A barrier over every slot of a co-resident grid.
    GridSync,
    /// A launch's host issue, and the device's gap between dependent
    /// launches.
    Issue,
    Gap,
}

/// **A primitive's measured costs** (the cost model's rows): its rate when
/// saturated (per unit, per ns — per clock where `per_clk`), the latency of
/// one use (ns, or clocks where `per_clk`), and the concurrency a unit keeps
/// in flight to reach 90 % of the rate (in the rate's own unit: bytes,
/// operations or instructions). Every value a [`Fact`] with its interval.
#[derive(Clone, Debug, PartialEq)]
pub struct Prim {
    pub name: &'static str,
    pub of: PrimOf,
    pub per_clk: bool,
    pub rate: Fact<f64>,
    pub latency: Fact<f64>,
    pub conc: Fact<f64>,
}

/// A primitive's costs at one bound, in ns: its rate a unit (work a ns), its
/// one use's latency, its loaded latency (the concurrency at the knee over
/// the rate, at least the one use's), and whether it read a marked fact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cost {
    pub rate: f64,
    pub latency: f64,
    pub loaded: f64,
    pub marked: bool,
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
    /// The measured row of primitive `of`, if any.
    pub fn prim(&self, of: PrimOf) -> Option<&Prim> {
        self.prims.iter().find(|p| p.of == of)
    }

    /// **A primitive's costs at bound `b`** (ns; a clock row converted at the
    /// sustained clock): its measured row, else the nearest measured row of
    /// its kind (a panel stream: the largest run at most its own, then the
    /// fewest rows at least its own; a copy or load: the largest width at
    /// most its own), else derived from the machine's other facts and
    /// marked — a mover from the place's unit reach (rate) and loaded
    /// latency, a store or a panel from the memory's device reach over the
    /// units, the scratchpad from its banking, reductions from the
    /// atomics, barriers and launches from dispatch.
    pub fn cost(&self, of: PrimOf, b: Bound) -> Cost {
        let ghz = self.ghz(b).max(1e-9);
        let units = self.units().max(1) as f64;
        let row = |p: &Prim| {
            let (r, l, c) = (p.rate.rate(b), p.latency.cost(b), p.conc.cost(b));
            let (r, l) = match p.per_clk {
                true => (r * ghz, l / ghz),
                false => (r, l),
            };
            Cost {
                rate: r,
                latency: l,
                loaded: l.max(if r > 0.0 { c / r } else { 0.0 }),
                marked: p.rate.marked() || p.latency.marked() || p.conc.marked(),
            }
        };
        if let Some(p) = self.prim(of) {
            return row(p);
        }
        // The nearest measured row of the same kind.
        let near = match of {
            PrimOf::Panel {
                from,
                run,
                rows,
                fly,
            } => {
                // The largest measured run at most this one (else the
                // smallest), then the fewest rows at least these (else the
                // most), then the nearest bytes in flight (by ratio).
                let panels: Vec<(&Prim, u64, u64, u64)> = self
                    .prims
                    .iter()
                    .filter_map(|p| match p.of {
                        PrimOf::Panel {
                            from: f,
                            run: r,
                            rows: w,
                            fly: y,
                        } if f == from => Some((p, r, w, y)),
                        _ => None,
                    })
                    .collect();
                let run_ = panels
                    .iter()
                    .map(|x| x.1)
                    .filter(|&r| r <= run)
                    .max()
                    .or_else(|| panels.iter().map(|x| x.1).min());
                let at_run: Vec<_> = panels.iter().filter(|x| Some(x.1) == run_).collect();
                let rows_ = at_run
                    .iter()
                    .map(|x| x.2)
                    .filter(|&w| w >= rows)
                    .min()
                    .or_else(|| at_run.iter().map(|x| x.2).max());
                let ratio = |y: u64| ((y.max(1) as f64) / (fly.max(1) as f64)).ln().abs();
                at_run
                    .iter()
                    .filter(|x| Some(x.2) == rows_)
                    .min_by(|a, b| ratio(a.3).total_cmp(&ratio(b.3)))
                    .map(|x| x.0)
            }
            PrimOf::Copy { from, bytes } | PrimOf::Load { from, bytes } => {
                let copy = matches!(of, PrimOf::Copy { .. });
                self.prims
                    .iter()
                    .filter(|p| match p.of {
                        PrimOf::Copy { from: f, bytes: w } => copy && f == from && w <= bytes,
                        PrimOf::Load { from: f, bytes: w } => !copy && f == from && w <= bytes,
                        _ => false,
                    })
                    .max_by_key(|p| match p.of {
                        PrimOf::Copy { bytes, .. } | PrimOf::Load { bytes, .. } => bytes,
                        _ => 0,
                    })
            }
            PrimOf::Barrier { lanes } => self
                .prims
                .iter()
                .filter(|p| matches!(p.of, PrimOf::Barrier { .. }))
                .min_by_key(|p| match p.of {
                    PrimOf::Barrier { lanes: l } => l.abs_diff(lanes),
                    _ => u64::MAX,
                }),
            PrimOf::Reduce {
                at,
                sharing,
                sectors,
            } => self
                .prims
                .iter()
                .filter(|p| matches!(p.of, PrimOf::Reduce { at: a, sharing: h, .. } if a == at && h == sharing))
                .min_by_key(|p| match p.of {
                    // The fewest sectors at least these, else the most.
                    PrimOf::Reduce { sectors: x, .. } if x >= sectors => (0, x),
                    PrimOf::Reduce { sectors: x, .. } => (1, u64::MAX - x),
                    _ => (2, 0),
                }),
            PrimOf::Store { to, bytes } => self
                .prims
                .iter()
                .filter(|p| matches!(p.of, PrimOf::Store { to: t, bytes: w } if t == to && w <= bytes))
                .max_by_key(|p| match p.of {
                    PrimOf::Store { bytes, .. } => bytes,
                    _ => 0,
                }),
            _ => None,
        };
        if let Some(p) = near {
            return row(p);
        }
        // Derived from the other facts: marked.
        let reach = |p: PlaceId, k: ExecKind| self.reach(p, k);
        let derived = |rate: f64, latency: f64| Cost {
            rate,
            latency,
            loaded: latency,
            marked: true,
        };
        let device_mem = |p: PlaceId| {
            reach(p, ExecKind::Device)
                .map(|r| (r.total_bytes_per_ns.rate(b) / units, r.latency_ns.cost(b)))
        };
        match of {
            PrimOf::Op(o) => {
                let op = &self.ops[o];
                Cost {
                    rate: op.rate.rate(b) * ghz,
                    latency: op.latency.cost(b) / ghz,
                    loaded: op.latency.cost(b) / ghz,
                    marked: op.rate.marked() || op.latency.marked(),
                }
            }
            PrimOf::Load { from, .. } | PrimOf::Copy { from, .. } | PrimOf::Bulk { from, .. } => {
                match (reach(from, ExecKind::Unit), device_mem(from)) {
                    (Some(r), _) => derived(r.bytes_per_ns.rate(b), r.latency_ns.cost(b)),
                    (None, Some((r, l))) => derived(r, l),
                    _ => derived(f64::INFINITY, 0.0),
                }
            }
            PrimOf::Panel { from, run, .. } => {
                // Unmeasured: the copy from the place (the memory's pages
                // unpriced), its rate's interval opened below for runs
                // shorter than a line.
                let c = self.cost(PrimOf::Copy { from, bytes: 16 }, b);
                let short = (run as f64 / 128.0).clamp(0.25, 1.0);
                let rate = match b {
                    Bound::Slow => c.rate * short,
                    _ => c.rate,
                };
                Cost {
                    rate,
                    marked: true,
                    ..c
                }
            }
            PrimOf::Store { to, .. } | PrimOf::Staged { to } => match device_mem(to) {
                Some((r, l)) => derived(r, l),
                None => derived(f64::INFINITY, 0.0),
            },
            PrimOf::Scratch { .. } => {
                let bw = self
                    .unit_scratch()
                    .and_then(|p| self.places[p].banking)
                    .map_or(128.0, |k| (k.banks * k.width) as f64);
                derived(bw * ghz, 30.0 / ghz)
            }
            PrimOf::Reduce {
                sharing, sectors, ..
            } => {
                let a = self.dispatch.atomics.first();
                let r = a.map_or(0.0, |a| a.ops_per_ns.rate(b)) / units;
                let share = match sharing {
                    0 => 32.0,
                    s => s as f64,
                };
                Cost {
                    rate: r / share.sqrt() * (4.0 / sectors.max(4) as f64),
                    latency: self
                        .device_cache()
                        .and_then(|c| reach(c, ExecKind::Device))
                        .map_or(0.0, |r| r.latency_ns.cost(b)),
                    loaded: 0.0,
                    marked: true,
                }
            }
            PrimOf::Barrier { .. } | PrimOf::Phase => {
                let l = self.dispatch.barrier_clk.cost(b) / ghz;
                Cost {
                    rate: 1.0 / l.max(1e-9),
                    latency: l,
                    loaded: l,
                    marked: self.dispatch.barrier_clk.marked() || matches!(of, PrimOf::Phase),
                }
            }
            PrimOf::GridSync => {
                let g = self.dispatch.grid_sync_ns.unwrap_or(Fact::nominal(
                    1000.0,
                    500.0,
                    3000.0,
                    "grid sync unmeasured",
                ));
                Cost {
                    rate: 1.0 / g.cost(b).max(1e-9),
                    latency: g.cost(b),
                    loaded: g.cost(b),
                    marked: g.marked(),
                }
            }
            PrimOf::Issue => Cost {
                rate: 1.0 / self.dispatch.issue_ns.cost(b).max(1e-9),
                latency: self.dispatch.issue_ns.cost(b),
                loaded: self.dispatch.issue_ns.cost(b),
                marked: self.dispatch.issue_ns.marked(),
            },
            PrimOf::Gap => Cost {
                rate: 1.0 / self.dispatch.gap_ns.cost(b).max(1e-9),
                latency: self.dispatch.gap_ns.cost(b),
                loaded: self.dispatch.gap_ns.cost(b),
                marked: self.dispatch.gap_ns.marked(),
            },
        }
    }

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
