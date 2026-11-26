//! **What a lowering declares** (ADR-0050 D2 "lowering-declared counts",
//! D6): the schedules it emits (its constraints narrow the space visibly),
//! and for each one its footprint (lanes, registers a lane, scratchpad bytes a
//! slot), its instruction counts per class per `k` trip, and its ring's
//! in-flight discipline. Each count is checked against `ptxas -v` and the
//! disassembly (D8); none is a fit of a kernel's time.

use crate::machine::{Class, Machine, OpId};
use crate::schedule::{Schedule, Site, Traversal};

/// A schedule's declared counts.
#[derive(Clone, Debug, PartialEq)]
pub struct Declared {
    /// Lanes a slot launches (producers included).
    pub threads: u64,
    /// Registers a lane holds (the largest role's; accumulators, fragments
    /// and the lowering's addresses).
    pub regs: u64,
    /// Bytes of the unit scratchpad a slot holds (every stage, barriers).
    pub smem: u64,
    /// Group-instructions a slot issues per `k` trip, by class.
    pub issue: Vec<(Class, f64)>,
    /// Bytes a slot keeps in flight from the device cache (its ring: e.g.
    /// `stages − 2` panels behind a `wait_group(stages − 2)`; one panel
    /// register-staged).
    pub in_flight: u64,
    /// `k` trips a piece's pipeline fill exposes before its first product.
    pub fill: u64,
    /// Accumulator chains a lane keeps independent per instruction (the
    /// group tile's instructions per `k` step a lane takes part in).
    pub chains: u64,
    /// Groups a slot issues the operation from (consumers).
    pub groups: u64,
}

impl Declared {
    pub fn count(&self, c: Class) -> f64 {
        self.issue
            .iter()
            .filter(|(k, _)| *k == c)
            .map(|(_, n)| n)
            .sum()
    }
}

/// Why a lowering does not emit a schedule (a declared constraint).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal(pub String);

impl Refusal {
    pub fn new(why: impl Into<String>) -> Self {
        Refusal(why.into())
    }
}

/// The traversal kinds a lowering emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TravKind {
    Ordered,
    SplitAtomic,
    SplitWorkspace,
    Walk,
    Hybrid,
    Persistent,
}

impl TravKind {
    pub fn of(t: Traversal) -> TravKind {
        match t {
            Traversal::Ordered { .. } => TravKind::Ordered,
            Traversal::Split {
                comb: crate::schedule::Comb::Atomic,
                ..
            } => TravKind::SplitAtomic,
            Traversal::Split { .. } => TravKind::SplitWorkspace,
            Traversal::Walk { .. } => TravKind::Walk,
            Traversal::Hybrid { .. } => TravKind::Hybrid,
            Traversal::Persistent { .. } => TravKind::Persistent,
        }
    }
}

/// **A lowering**: the kinds of schedule it emits and what each declares.
pub trait Lowering {
    fn id(&self) -> &'static str;
    /// The traversals it emits (a declared constraint: `wgmma` ordered only).
    fn traversals(&self) -> &'static [TravKind];
    /// The operations it emits.
    fn ops(&self, m: &Machine, s: &Site) -> Vec<(OpId, bool)>;
    /// D5 steps 2–3: its slot schedules for `op` (roles `swap`) — group
    /// tiles whose declared footprint fits the operation's home share, then
    /// slot tiles, `bk`, depths and layouts within a slot's capacities, at
    /// every feasible occupancy — each with an ordered traversal (step 4
    /// varies it).
    fn slots(&self, m: &Machine, s: &Site, op: OpId, swap: bool) -> Vec<Schedule>;
    /// Every declared constraint checked, the counts given.
    fn declare(&self, m: &Machine, s: &Site, x: &Schedule) -> Result<Declared, Refusal>;
}
