//! Instruction classes, pipes, movers, dispatch and the measured primitives.

use super::*;

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
/// the rate, at least the one use's), and whether it read a marked fact —
/// a row borrowed from a neighbouring measured primitive is marked and
/// names that row (`borrowed`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cost {
    pub rate: f64,
    pub latency: f64,
    pub loaded: f64,
    pub marked: bool,
    pub borrowed: Option<&'static str>,
}
