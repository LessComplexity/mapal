//! `mapal-ir`: the Mapal-Core intermediate representation — the project's central
//! data structure (DESIGN `docs/components/ir/DESIGN.md`; ADR-0013).
//!
//! The IR is a category of **objects** (dataflow nodes) and **morphisms**
//! (directed dataflow edges). The one structural decision everything follows
//! from (ADR-0013, ERRATA LC-4): **all dataflow is adjacency edges; no morphism
//! payload carries value flow.** Product formation is per-slot `Pair` edges,
//! constants are source objects, loops are inline cycles through a `LoopMerge`
//! (the back edge is real and SCC-visible), and effects are linear world-token
//! threading (`Ty::IoToken`).
//!
//! Construction is gated: a graph is buildable **only** through the
//! invariant-enforcing [`IrBuilder`] (DESIGN §10) and sealed by global
//! validation — "ill-formed is unconstructible" (the §9 invariant ledger). An
//! independent [`validate`] re-derives the graph-shape invariants as the
//! property-test oracle (DESIGN §11). v1 graphs are **append-only then sealed**.
//!
//! Design highlights:
//! - **Determinism is sacred** (E2): no `HashMap` anywhere — `slotmap`
//!   SlotMap/SecondaryMap + insertion-ordered `Vec`s only (D2 / I12). Dumps,
//!   topo orders, and snapshots are byte-stable.
//! - **No unguarded recursion** (J1 / I10): `Ty` walks, Tarjan SCC, and Kahn
//!   topo are all iterative or depth-guarded (`MAX_TY_DEPTH = 64`).
//! - **Renderer-free values** (C3): no `Display` impls; Mermaid text comes from
//!   the plain function [`ty_label`] and [`CategoryIr::to_mermaid`].

mod access;
mod algo;
mod builder;
pub mod exp;
mod facts;
mod graph;
mod loc;
mod mermaid;
pub mod narrow;
pub mod read;
mod residence;
pub mod split;
mod ty;
mod unit;
mod validate;

pub use access::{AccessPlan, Hazard};
pub use algo::{
    Affine, AxisDigit, BoundsProof, BoxClass, CallKeep, CellRead, CellSite, Checkpoint, Delta,
    DepLaw, Digit, Edge, ElemPlan, ElemSrc, EmissionClass, EmissionPlan, FinitePlan, FmaPair,
    FoldCombine, FoldCombinePlan, Footprint, FootprintPlan, GuardArm, GuardBox, GuardSide,
    GuardSite, LastUsePlan, LinearSite, LoopPlan, LoopScc, MovePlan, MoveSite, NarrowStore,
    Outside, PathPlan, PrecisionPlan, RadixAddr, RadixForm, RadixTile, Reach, Regroup, RescaleArg,
    RowDep, RowFoldOp, RowFoldSite, StepAlgebra, StepFlags, StreamGroup, StreamGuard, StreamLaw,
    StreamPlan, StreamRole, Tap, Task, TaskId, TaskKind, TensorAxis, TensorDim, TensorForm,
    TileKSplit, TilePlan, TilePost, TileRead, TileSite, TripPhase, TripSplit, UnchangedPlan,
    WaitEntry, bound_coef, bound_coef_unit, bound_coef_units, extreme_up, gamma, uniform_at,
};
pub use builder::{Dest, FnBuilder, IrBuilder, IrError, LoopHandle};
pub use facts::FactArms;
pub use graph::{
    CategoryIr, FuncDef, FuncId, FuncKind, Morphism, MorphismId, Object, ObjectId, ObjectKind,
    Operation, Precision,
};
pub use loc::SourceLoc;
pub use mermaid::lint_mermaid;
pub use residence::{Residence, ResidencePlan};
pub use ty::{MAX_TY_DEPTH, NarrowFloat, Ty, Value, ty_contains_token, ty_label};
pub use unit::{RowMap, UnitGroup, UnitKind, UnitMember, UnitPlan};
pub use validate::{IrViolation, validate};
