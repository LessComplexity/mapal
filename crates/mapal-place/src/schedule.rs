//! **A site and its schedule** (ADR-0050 D3). A site is what the dataflow
//! facts say about one sum-of-products tile site (extents, operand bytes, the
//! fold's freedom, the products its precision set admits, operands resident
//! from the previous call); a schedule is one placement of it on a machine.

use crate::machine::{ExecId, Fmt, LaunchMode, Layout, OpId, PlaceId, Policy};

/// The order a site's `k` fold may take (ADR-0040).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fold {
    /// Any order and grouping: cuts, splits and walks.
    Free,
    /// A fixed grouping replayed exactly: one partition count.
    Fixed,
    /// One in-order chain per cell.
    Sequential,
}

/// **A sum-of-products site** as placement reads it: `c[m × n] = Σ_k a·b`
/// (`batch` independent such products), operand and output element bytes as
/// stored, epilogue operand bytes a cell, the fold's freedom, whether `k`
/// may be cut at all (a window or mixed-radix site never is), the operation
/// formats its precision set admits (ADR-0041 / 0043, decided by the
/// backend), and operands resident from the previous call (`residence_plan`).
#[derive(Clone, Debug, PartialEq)]
pub struct Site {
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub batch: u64,
    pub ea: u64,
    pub eb: u64,
    pub eo: u64,
    pub epi: u64,
    pub fold: Fold,
    pub kcut: bool,
    pub fmts: Vec<Fmt>,
    pub warm: [bool; 2],
}

impl Site {
    pub fn flops(&self) -> u64 {
        2 * self.batch.max(1) * self.m * self.n * self.k
    }
    /// Every operand and output byte once.
    pub fn bytes_once(&self) -> u64 {
        let b = self.batch.max(1);
        b * (self.m * self.k * self.ea
            + self.k * self.n * self.eb
            + self.m * self.n * (self.eo + self.epi))
    }
}

/// What a schedule level's loop extent is bound to (D3): an executor level
/// (parallel over its instances) or a place (sequential blocking through it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bind {
    Exec(ExecId),
    /// The dispatched slot (a block) on the dispatch unit.
    Slot,
    Place(PlaceId),
}

/// One operand's home at a level: the place, its layout and its buffering
/// depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Home {
    pub place: PlaceId,
    pub layout: Layout,
    pub depth: u64,
}

/// **A level** (D3): a tile of `m × n × k` extents bound to an executor level
/// or a place, and where each operand lives at it. Levels run innermost
/// first: a lane's register tile, a group's tile, a slot's tile and its `k`
/// panel, then the grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Level {
    pub ext: [u64; 3],
    pub bind: Bind,
    pub homes: [Option<Home>; 2],
}

/// Partial sums' combine (D3: split `S`, atomic or workspace).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Comb {
    Atomic,
    Workspace,
}

/// **The traversal** (D3) of a grid of output tiles: ordered with a raster
/// group height `h` (1: row-major); split into `s` partitions of `k`; walked
/// (stream-K over `ctas` slots of `iters` units each, with or without a
/// raster); hybrid (whole waves ordered, the tail walked); persistent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Traversal {
    Ordered { h: u64 },
    Split { s: u64, comb: Comb, h: u64 },
    Walk { h: u64, ctas: u64, iters: u64 },
    Hybrid { h: u64, ctas: u64, iters: u64 },
    Persistent { h: u64, ctas: u64 },
}

impl Traversal {
    pub fn raster(self) -> u64 {
        match self {
            Traversal::Ordered { h }
            | Traversal::Split { h, .. }
            | Traversal::Walk { h, .. }
            | Traversal::Hybrid { h, .. }
            | Traversal::Persistent { h, .. } => h,
        }
    }
    /// `k` partitions a tile's fold is cut into (a walk: unbounded, 0).
    pub fn pieces(self) -> u64 {
        match self {
            Traversal::Ordered { .. } | Traversal::Persistent { .. } => 1,
            Traversal::Split { s, .. } => s,
            Traversal::Walk { .. } | Traversal::Hybrid { .. } => 0,
        }
    }
}

/// **A schedule** (D3): the lowering that emits it, the operation and the
/// operand roles (`swap`: the transposed product `cᵀ = bᵀ·aᵀ`), the levels,
/// the slot's composition (groups, producer groups and each role's register
/// share), a cluster tile with multicast per operand, the traversal, the
/// launch mode, the cache policy per operand, the occupancy it was costed at
/// (≤ the one its footprint derives), and the lowering's own encoding
/// parameters (D6: bit encodings that are not machine facts — named by the
/// lowering, read by no evaluator term).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Schedule {
    pub lowering: &'static str,
    pub op: OpId,
    pub swap: bool,
    pub levels: Vec<Level>,
    /// Groups a slot holds along `m` and `n`, and producer groups beside
    /// them.
    pub groups: [u64; 2],
    pub producers: u64,
    /// Registers a lane of a producer and of a consumer (0: the launch's).
    pub regs: [u64; 2],
    pub cluster: [u64; 2],
    pub multicast: [bool; 2],
    pub trav: Traversal,
    pub launch: LaunchMode,
    pub policy: [Policy; 2],
    pub per_unit: u64,
    pub params: Vec<(&'static str, u64)>,
}

impl Schedule {
    /// The level bound to the slot.
    pub fn slot(&self) -> &Level {
        self.levels
            .iter()
            .find(|l| l.bind == Bind::Slot)
            .expect("a schedule has a slot level")
    }
    /// The slot's output tile `(bm, bn)` and its panel depth `bk`.
    pub fn bm(&self) -> u64 {
        self.slot().ext[0]
    }
    pub fn bn(&self) -> u64 {
        self.slot().ext[1]
    }
    pub fn bk(&self) -> u64 {
        self.slot().ext[2]
    }
    /// Buffering depth of the slot's `k` panel (stages).
    pub fn depth(&self) -> u64 {
        self.slot()
            .homes
            .iter()
            .flatten()
            .map(|h| h.depth)
            .max()
            .unwrap_or(1)
    }
    /// The innermost level's tile (a lane's accumulators).
    pub fn lane_tile(&self) -> [u64; 3] {
        self.levels[0].ext
    }
    /// A lowering parameter by name.
    pub fn param(&self, name: &str) -> Option<u64> {
        self.params
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| *v)
    }
    /// Output tiles of the grid.
    pub fn tiles(&self, s: &Site) -> u64 {
        s.batch.max(1) * (s.m / self.bm().max(1)) * (s.n / self.bn().max(1))
    }
    /// Panels of a tile's fold.
    pub fn panels(&self, s: &Site) -> u64 {
        s.k / self.bk().max(1)
    }
    /// A one-line plan string (`; mapal.place`, `place_replay.tsv`).
    pub fn plan(&self) -> String {
        let t = match self.trav {
            Traversal::Ordered { h } => format!("ord(h{h})"),
            Traversal::Split { s, comb, h } => format!(
                "split(s{s},{},h{h})",
                match comb {
                    Comb::Atomic => "atomic",
                    Comb::Workspace => "ws",
                }
            ),
            Traversal::Walk { h, ctas, iters } => format!("walk(h{h},{ctas}x{iters})"),
            Traversal::Hybrid { h, ctas, iters } => format!("hybrid(h{h},{ctas}x{iters})"),
            Traversal::Persistent { h, ctas } => format!("persist(h{h},{ctas})"),
        };
        let lv: Vec<String> = self
            .levels
            .iter()
            .map(|l| format!("{}x{}x{}", l.ext[0], l.ext[1], l.ext[2]))
            .collect();
        let mut p = format!(
            "{}:op{}{} {} g{}x{}+{} d{} u{} {t}",
            self.lowering,
            self.op,
            if self.swap { "T" } else { "" },
            lv.join("/"),
            self.groups[0],
            self.groups[1],
            self.producers,
            self.depth(),
            self.per_unit
        );
        for (n, v) in &self.params {
            p += &format!(" {n}={v}");
        }
        p
    }
}
