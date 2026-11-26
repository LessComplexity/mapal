//! **mapal-place — the machine model** (ADR-0050).
//!
//! The compiler holds two kinds of facts: the dataflow's (`mapal-ir`) and the
//! hardware's. This crate is the second kind, target-agnostic: a [`Machine`]
//! of places (a graph of instances: explicit scratchpads and caches with
//! their instances, keeps and banking), executors of the target's own depth
//! with operations as rows of geometry, movers and dispatch — every value a
//! [`Fact`] with its provenance and interval. A [`Schedule`] places one site
//! on it; one [`evaluate`] prices every schedule; reuse is counted by
//! *distinct* bytes ([`reuse`]), checked against an exact LRU trace
//! simulation ([`lru`]). A backend contributes only its translation
//! (`Machine::of`) and its lowerings' declarations ([`Lowering`]).

pub mod eval;
pub mod fact;
pub mod lower;
pub mod lru;
pub mod machine;
pub mod reuse;
pub mod schedule;
pub mod search;
pub mod space;

pub use eval::{Phase, Terms, efficiency, evaluate, evaluate_floor, lower_bound};
pub use fact::{Bound, Fact, Row, Src};
pub use lower::{Declared, Lowering, Refusal, TravKind};
pub use machine::{
    Atomics, Banking, Cache, CapacityGroup, Class, Dispatch, EffPoint, Exec, ExecId, ExecKind, Fmt,
    InstanceMap, Keep, LaunchCost, LaunchMode, Layout, Machine, Major, Mgmt, Mover, MoverKind,
    NSet, Op, OpId, Pattern, Pipe, Place, PlaceId, Policy, Reach, Res, Resource, Start,
};
pub use reuse::{Grid, Order, Reads};
pub use schedule::{Bind, Comb, Fold, Home, Level, Schedule, Site, Traversal};
pub use search::{Cand, Opts, Ranked, Stats, point_bound, search, tile_bound};
