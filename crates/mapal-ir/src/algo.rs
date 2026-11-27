//! Graph algorithms (DESIGN §13): iterative Tarjan SCC, Kahn topological order,
//! and the `loop_structure` backend predicate.
//!
//! Everything here is `O(V + E)` and **recursion-free** (J1): Tarjan uses an
//! explicit work stack, Kahn is a worklist. The §16 deep-graph test (100k-object
//! chain) exercises the no-stack-overflow guarantee.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use slotmap::SecondaryMap;

use crate::graph::{CategoryIr, FuncId, MorphismId, ObjectId, ObjectKind, Operation};
use crate::ty::{Ty, Value, ty_contains_token};

mod algebra;
mod bounds;
mod cells;
mod elem;
mod folds;
mod guard;
mod last_use;
mod linear;
mod loops;
mod paths;
mod plan_types;
mod precision;
mod radix;
mod range;
mod site_plans;
mod statics;
mod stream;
#[cfg(test)]
mod tensor_form_tests;
mod tile_types;
mod tiles;
mod unchanged;
mod write_delta;
pub use self::last_use::*;
pub use self::plan_types::*;
pub use self::tile_types::*;
pub use algebra::{
    DepLaw, FmaPair, FoldCombine, FoldCombinePlan, Regroup, RescaleArg, RowDep, StepAlgebra,
    StepFlags, whole_rows,
};
pub use bounds::BoundsProof;
pub use linear::LinearSite;
pub use precision::{
    NarrowStore, PrecisionPlan, bound_coef, bound_coef_unit, bound_coef_units, gamma,
};
pub use radix::{AxisDigit, Digit, RadixAddr, RadixForm, RadixTile, uniform_at};
pub use range::FinitePlan;
pub use statics::{
    ATTEMPT_WORK, EMBED_BYTES, EVAL_WORK, Realize, StaticPlan, StaticRoot, ty_extent,
};
pub use stream::{
    BoxClass, GuardBox, GuardEnv, GuardSide, NoValues, RowLoad, StreamGroup, StreamGuard,
    StreamLaw, StreamPlan, StreamRole, extreme_up,
};
pub use unchanged::{CallKeep, UnchangedPlan};

/// A non-trivial loop SCC and its merge objects (DESIGN §7; the backend
/// capability predicate). One entry per non-trivial SCC.
#[derive(Clone, Debug, PartialEq)]
pub struct LoopScc {
    /// All objects in the SCC, in deterministic order.
    pub objects: Vec<ObjectId>,
    /// The `LoopMerge` objects within this SCC (≥1; >1 for nested loops).
    pub merges: Vec<ObjectId>,
}

/// The per-merge loop attribution the interp driver, rewrite replayer, and the
/// backend-llvm emitter all derive from one graph (DESIGN §3, BL7 — the one
/// source of truth for the rule whose two hand-maintained copies both regressed
/// in S12). Produced by [`CategoryIr::loop_plan`], which returns `None` for any
/// non-canonical shape (multi-merge SCC, ≠1 `LoopBack`, ≠1 attributed
/// `LoopExit`) — so `is_canonical`-style gates read `.is_some()` and the interp
/// / backend drivers, which only ever run on gated-canonical loops, `.expect()`.
#[derive(Clone, Debug, PartialEq)]
pub struct LoopPlan {
    /// The `LoopMerge` object this plan is for.
    pub merge: ObjectId,
    /// `source(LoopEnter → merge)` — the init value object (loop-invariant).
    pub init: ObjectId,
    /// `source(LoopBack → merge)` — the `(next_state, cond)` route object.
    pub back_route: ObjectId,
    /// The attributed `LoopExit` morphisms (canonical: exactly one).
    pub exits: Vec<MorphismId>,
    /// `source(the unique attributed LoopExit)` — the `(payload, cond)` route.
    pub exit_route: ObjectId,
    /// The SCC's objects (membership set; deterministic order).
    pub scc_objects: Vec<ObjectId>,
    /// Decide phase (ADR-0016): body morphisms whose target is backward-reachable
    /// from `exit_route` — the cond + exit-route feeders, run every iteration.
    pub decide_order: Vec<MorphismId>,
    /// Advance phase: the rest of the body (next-state), skipped on the exit step.
    pub advance_order: Vec<MorphismId>,
    /// Product objects assembled by `Pair` edges across the body (reset/iter).
    pub product_targets: Vec<ObjectId>,
}

/// Which cells of a loop-carried array one iteration writes (S53 `retain-delta`,
/// [`CategoryIr::write_delta`]): where the value crossing the back edge can
/// differ from the merge value it started the iteration as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delta {
    /// Only cells `[start, start + stride)` can differ. `next` is the object
    /// whose value crosses the back edge into this carried slot (the last
    /// write of the chain) — it differs from its own previous-iteration value
    /// only there too. A claim about cells: an interval a piecewise map
    /// writes ([`Outside`]) may reach past `[0, n)` (on a trip that writes
    /// nothing, anywhere), so a consumer moves or recomputes `[start, start +
    /// stride)` only when it lies inside the array (else the whole array).
    Rows {
        stride: u64,
        start: Affine,
        next: ObjectId,
    },
    /// device-geometry `piece-delta`: only the pieces `[start + step·j,
    /// start + step·j + w)`, `j ≥ 0`, within the array can differ — one row
    /// of each sequence of a `[B][L][…]` cache ([`Outside`] with `w > 1`).
    /// `next` as in [`Delta::Rows`]. `start` may lie before the array (a
    /// trip that writes nothing), as a row's may.
    Pieces {
        w: u64,
        step: u64,
        start: Affine,
        next: ObjectId,
    },
    /// Anything else: the whole array may differ (the status quo).
    Top,
}

/// `a·var + b` in the wrapping arithmetic of `var`'s int type, `var` an int
/// view of the loop's merge — the counter, as this iteration reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Affine {
    pub var: ObjectId,
    pub a: i64,
    pub b: i64,
}

/// A [`PathPlan`] task identifier. Task ids are indices into
/// [`PathPlan::tasks`].
pub type TaskId = usize;

/// One schedulable unit in a function's path plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Task {
    /// Whether the task is element-range splittable or sequential.
    pub kind: TaskKind,
    /// Tasks whose produced objects this task consumes.
    pub deps: Vec<TaskId>,
    /// Saturating critical-path weight to a sink.
    pub rank: u32,
    /// Earliest topo position of a trap-capable morphism in this task.
    pub trap_min: Option<u32>,
    /// Whether a trap-capable pure named call pins this task to the host spine.
    pub pinned: bool,
}

/// The execution shape of a [`Task`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaskKind {
    /// One element-range-splittable bulk operation.
    Split {
        /// The bulk morphism.
        site: MorphismId,
        /// Its statically known element count.
        n: u64,
    },
    /// Morphisms that execute sequentially, in topo order.
    Seq {
        /// The task's morphisms.
        morphisms: Vec<MorphismId>,
    },
}

/// One task-progress requirement at a host-spine checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaitEntry {
    /// The task whose progress gates the checkpoint.
    pub task: TaskId,
    /// `Some(w)` accepts a decided watermark ≥ `w` or task completion;
    /// `None` requires task completion.
    pub threshold: Option<u32>,
}

/// A host-spine observation point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    /// The token operation's topo position, or `u32::MAX` for function exit.
    pub topo: u32,
    /// Where the host waits: the topo position of the earliest host-built
    /// `Pair` / `Proj` edge that assembles the operation's value from a task's
    /// object — never before a pinned task the wait depends on runs — else
    /// `topo`.
    pub at: u32,
    /// Task progress required before the host may pass this point.
    pub wait: Vec<WaitEntry>,
}

/// The deterministic task DAG and host-spine checkpoints for one function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathPlan {
    /// Schedulable pure tasks. Task ids are vector indices.
    pub tasks: Vec<Task>,
    /// Token-operation checkpoints in topo order, followed by function exit.
    pub checkpoints: Vec<Checkpoint>,
}

/// One loop's complete driver region (plan-s40): SCC-incident morphisms,
/// machinery, and the `loop_plan` cones — exactly the set the flat walk hands
/// the driver. Private to [`CategoryIr::guard_plan`]: an arm owns a unit
/// atomically, represented in its own-list by the `LoopEnter` handle(s) alone;
/// internals never enter an own-list (the driver fires them).
struct LoopUnit {
    members: SecondaryMap<MorphismId, ()>,
    /// The `LoopEnter` handle(s) — what stands for the unit in an own-list.
    enters: Vec<MorphismId>,
    /// Drivable: single merge with a `loop_plan`, machinery unshared. A
    /// non-canonical unit never joins an arm (it runs unconditionally, which
    /// is always safe) — and `LiftLoops` consumes `loop_plan` facts too, so a
    /// shape the driver cannot run is a shape the rewriter cannot lift:
    /// refused raw ⇒ refused rewritten, no stability hole.
    canonical: bool,
    /// Any member trap-capable — the handle carries the whole unit's flag.
    can_trap: bool,
}

/// One arm of a Phi-position guard (plan-s39). `own` is the arm's exclusive
/// work: the morphisms whose every path to an observable root passes through
/// this arm's `Pair` edge — they fire only when the condition picks this arm.
///
/// Ownership is DIRECT: a nested guard's own morphisms belong to the nested
/// site (whose `phi` stays in this list as the gate that fires it), so the
/// own-lists of all sites partition the gated morphisms — each has exactly one
/// innermost owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardArm {
    /// The arm's value object (the triple's `Pair` slot 0/1 feeder).
    pub value: ObjectId,
    /// The boundary `Pair` edge writing `value` into the Phi triple.
    pub edge: MorphismId,
    /// Direct exclusive work in topo order; `edge` is last. Empty ⇒ the value
    /// is computed unconditionally (a parameter, constant, or shared object)
    /// and there is nothing to gate.
    pub own: Vec<MorphismId>,
    /// Whether the arm's TRANSITIVE exclusive work (this list plus any nested
    /// site's) holds a trap-capable morphism: integer `Div`/`Mod`, unproven
    /// `Index`, `Update`, or a `Call`/`Map`/`Fold` whose body is trap-capable.
    /// An ungated realization (today's `select`) is sound only when false.
    pub can_trap: bool,
    /// Whether the arm's exclusive work is worth skipping: it holds a bulk
    /// operation or a call. Scalar arithmetic is not — computing `x * -1`
    /// costs less than branching around it, and inside a map body a branch
    /// would cost the whole element loop its vectorization.
    pub heavy: bool,
    /// Whether [`CategoryIr::bounds_proof`] holds an `Index` in bounds only
    /// under this arm's condition (the arm's own work, or a nested site's or
    /// body's below it): the read is safe only while the arm runs alone, so
    /// the arm is gated (`can_trap` stays false — no trap check is emitted).
    pub cond_reads: bool,
}

/// A Phi-position guard site (plan-s39): the condition and the two arms'
/// exclusive work. An arm's work runs only if the condition picks that arm;
/// a Phi the query does not report (a non-builder triple shape, or an arm
/// reaching loop machinery) keeps today's strict both-arms semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardSite {
    /// The `Phi` morphism.
    pub phi: MorphismId,
    /// The `Bool` condition object (the triple's `Pair` slot 2 feeder).
    pub cond: ObjectId,
    pub on_true: GuardArm,
    pub on_false: GuardArm,
}

impl GuardSite {
    /// Whether consumers must gate this site. Legality first: a trap-capable
    /// arm can never be evaluated ungated, and no cost argument buys it. Then
    /// cost: a `heavy` arm is worth branching around. Everything else — two
    /// arms of scalar arithmetic — stays ungated, because for a total arm of a
    /// few register ops, gating and computing are the same work and the branch
    /// is the more expensive of the two. Every arm's own-list ends with its
    /// boundary `Pair` edge, so an "empty" arm is never literally empty; this
    /// predicate, not emptiness, is the test.
    pub fn gated(&self) -> bool {
        [&self.on_true, &self.on_false]
            .iter()
            .any(|a| a.can_trap || a.heavy || a.cond_reads)
    }
}

/// Pair-fed primitives whose product source can disappear into the operation.
fn is_pair_primitive(op: Operation) -> bool {
    matches!(
        op,
        Operation::Add
            | Operation::Sub
            | Operation::Mul
            | Operation::Div
            | Operation::Mod
            | Operation::Eq
            | Operation::Neq
            | Operation::Lt
            | Operation::Gt
            | Operation::Le
            | Operation::Ge
            | Operation::And
            | Operation::Or
            | Operation::Phi
            | Operation::Index
    )
}

fn safe_integer_divisor(object: &crate::graph::Object) -> bool {
    if object.kind != ObjectKind::Constant {
        return false;
    }
    match object.value.as_ref() {
        Some(Value::I32(v)) => *v != 0 && *v != -1,
        Some(Value::I64(v)) => *v != 0 && *v != -1,
        Some(Value::U8(v)) => *v != 0,
        _ => false,
    }
}

fn increment(counts: &mut SecondaryMap<ObjectId, u32>, o: ObjectId) {
    if let Some(&n) = counts.get(o) {
        counts.insert(o, n + 1);
    }
}
