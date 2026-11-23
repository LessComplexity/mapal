//! Facts live on the graph: every [`crate::algo`] query is a pure function of
//! the sealed graph and a [`FuncId`], so it is computed at most once per
//! (graph, function) and stored here. Fact→fact dependencies and the backends
//! read the stored value; nothing recomputes a topo order, SCC set, loop plan
//! or bounds proof it has already derived.
//!
//! Slots are pre-allocated at seal (one per function, the set never changes)
//! and filled by [`OnceLock`], so the graph stays immutable and `Sync`. A slot
//! cannot be re-entered: a fact of `f` that needed itself would already
//! recurse forever without the store.

use std::sync::OnceLock;

use slotmap::SecondaryMap;

use crate::algo::{
    BoundsProof, CellSite, Delta, ElemPlan, EmissionPlan, FootprintPlan, GuardSite, LastUsePlan,
    LinearSite, LoopPlan, LoopScc, MovePlan, PathPlan, PrecisionPlan, RadixForm, RowFoldSite,
    StepAlgebra, TilePlan, TileSite, TripSplit,
};
use crate::graph::{CategoryIr, FuncId, MorphismId, ObjectId};

/// One function's stored facts.
#[derive(Default)]
pub(crate) struct FuncFacts {
    pub(crate) func_objects: OnceLock<Vec<ObjectId>>,
    pub(crate) emission_plan: OnceLock<EmissionPlan>,
    pub(crate) tile_plan: OnceLock<TilePlan>,
    pub(crate) tile_plan_strided: OnceLock<TilePlan>,
    pub(crate) tile_plan_post: OnceLock<TilePlan>,
    pub(crate) tile_plan_narrow: OnceLock<TilePlan>,
    pub(crate) radix_plan: OnceLock<SecondaryMap<ObjectId, RadixForm>>,
    pub(crate) tile_plan_radix: OnceLock<SecondaryMap<MorphismId, TileSite>>,
    pub(crate) precision_plan: OnceLock<PrecisionPlan>,
    pub(crate) step_algebra: OnceLock<StepAlgebra>,
    pub(crate) fold_combine: OnceLock<crate::algo::FoldCombinePlan>,
    pub(crate) move_plan: OnceLock<MovePlan>,
    pub(crate) row_fold: OnceLock<SecondaryMap<MorphismId, RowFoldSite>>,
    pub(crate) whole_fold: OnceLock<SecondaryMap<MorphismId, RowFoldSite>>,
    pub(crate) cell_plan: OnceLock<SecondaryMap<MorphismId, CellSite>>,
    pub(crate) linear_plan: OnceLock<SecondaryMap<MorphismId, LinearSite>>,
    pub(crate) footprint_plan: OnceLock<FootprintPlan>,
    pub(crate) elem_plan: OnceLock<ElemPlan>,
    pub(crate) path_plan: OnceLock<PathPlan>,
    pub(crate) sccs: OnceLock<Vec<Vec<ObjectId>>>,
    pub(crate) topo_order: OnceLock<Vec<MorphismId>>,
    pub(crate) loop_structure: OnceLock<Vec<LoopScc>>,
    /// Every merge of [`CategoryIr::loop_structure`] with its plan; a merge
    /// outside every loop SCC has none by definition.
    pub(crate) loop_plans: OnceLock<Vec<(ObjectId, Option<LoopPlan>)>>,
    pub(crate) last_use_plan: OnceLock<LastUsePlan>,
    pub(crate) bounds_proof: OnceLock<BoundsProof>,
    pub(crate) write_delta: OnceLock<SecondaryMap<ObjectId, Delta>>,
    pub(crate) unchanged_plan: OnceLock<crate::algo::UnchangedPlan>,
    pub(crate) call_keep: OnceLock<SecondaryMap<MorphismId, crate::algo::CallKeep>>,
    pub(crate) guard_plan: OnceLock<Vec<GuardSite>>,
    pub(crate) guard_structure: OnceLock<Vec<GuardSite>>,
    pub(crate) trip_split: OnceLock<Vec<TripSplit>>,
    pub(crate) unit_plan: OnceLock<crate::unit::UnitPlan>,
    pub(crate) stream_plan: OnceLock<crate::algo::StreamPlan>,
    pub(crate) finite_plan: OnceLock<crate::algo::FinitePlan>,
    pub(crate) access_plan: OnceLock<crate::access::AccessPlan>,
    pub(crate) residence_plan: OnceLock<crate::residence::ResidencePlan>,
}

/// Measurement arms of the facts themselves: each switch turns one fact rule
/// off so a build can be compared against the facts before it (an OFF arm
/// is a measurement flag, never a per-part setting). Default: every rule on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FactArms {
    /// onto-device `cond-bounds`: `bounds_proof` narrows ranges by the
    /// conditions holding where a value is used (guard arms, a loop's continue
    /// and exit conditions, `Phi` arms) and ranges loop-carried counters from
    /// `loop_plan`. Off: plain interval ranges over constants, iota elements
    /// and map/fold quantification.
    pub cond_bounds: bool,
    /// device-geometry `row-tree`: a cell's tree fold ([`crate::CellSite::tree`])
    /// is its one nested `Map`/`Fold` that loads array elements, beside any
    /// that load none (a norm's Heron root). Off: only a body whose one
    /// nested `Map`/`Fold` it is.
    pub row_tree: bool,
    /// device-geometry `piece-delta`: a piecewise map's set may be a row
    /// every `s` rows (`(t ÷ d) % s == r`: [`crate::Outside`] `w > 1`), and
    /// [`crate::Delta::Pieces`] its carried write. Off: no such set.
    pub piece_delta: bool,
    /// device-geometry `head-group`: a mixed-radix tile site may have batch
    /// digits — map-index digits both operands read (a KV head's query-head
    /// group, a sequence), row digits of `a` and block-uniform row digits of
    /// `b` ([`crate::CategoryIr::tile_plan_radix`]). Off: every digit is
    /// one operand's.
    pub head_group: bool,
    /// exp (ADR-0042 D7): `fold_combine` gives a value read only through
    /// `exp` of a difference [`crate::DepLaw::Rescale`]. Off
    /// (`--no-exp-rescale`): such a value is a phase.
    pub exp_rescale: bool,
    /// long-prefill `uniform-base`: a mixed-radix tile read's address may
    /// take terms of a number the map captures (a loop counter slicing a
    /// head group or a weight panel: [`crate::RadixTile::ua`]), each in
    /// bounds at its capture range's top. Off: such a read has no form (the
    /// map is a per-cell kernel).
    pub uniform_base: bool,
    /// caching-memory `unchanged`: [`crate::CategoryIr::unchanged_plan`]'s
    /// trip deltas reach through every map whose cells read rows of a
    /// changed array. Off: its trip deltas are `write_delta`'s alone.
    pub unchanged: bool,
    /// tensor-cores `narrow-radix`: a mixed-radix tile site's operand may be
    /// the exact widening of a narrow (`bf16` / `f16`) array element — a
    /// `bf16` KV cache read by attention's head digits
    /// ([`crate::CategoryIr::tile_plan_radix`], [`crate::TileRead::narrow`]).
    /// Off: such a read has no form (the map is a per-cell kernel).
    pub narrow_radix: bool,
    /// flash-attention `stream`: [`crate::CategoryIr::stream_plan`] proves
    /// a row-block group one pass over windows of its axis (each fold's
    /// state combined by its law). Off (`--gpu-off=stream-fact`): no stream
    /// group.
    pub stream: bool,
    /// flash-attention F7: [`crate::CategoryIr::finite_plan`] proves values
    /// finite by interval transfer. Off (`--gpu-off=finite`, `--no-finite`):
    /// nothing is proven finite.
    pub finite: bool,
    /// flash-attention F5: `stream_plan`'s guarded producer has its mask as
    /// a [`crate::GuardBox`] (masked windows skipped where F6 allows).
    /// Off (`--gpu-off=guard-box`, `--no-guard-box`): no mask form, every
    /// window computed.
    pub guard_box: bool,
    /// sweep-attention `unit-uniform`: `unit_plan`'s extended pass takes a
    /// mixed-radix tile whose operand address a captured number moves
    /// ([`crate::RadixTile::ua`] / `ub`: a loop counter slicing a KV head
    /// group, a decode position) — one value a launch, so the unit's launch
    /// moves that operand's pointer. Off (`--gpu-off=unit-uniform`): such a
    /// tile seeds and joins no extended group.
    pub unit_uniform: bool,
}

impl Default for FactArms {
    fn default() -> Self {
        FactArms {
            cond_bounds: true,
            row_tree: true,
            piece_delta: true,
            head_group: true,
            exp_rescale: true,
            uniform_base: true,
            unchanged: true,
            narrow_radix: true,
            stream: true,
            finite: true,
            guard_box: true,
            unit_uniform: true,
        }
    }
}

/// The graph's fact store.
#[derive(Default)]
pub(crate) struct Facts {
    per: SecondaryMap<FuncId, FuncFacts>,
    pub(crate) trap_capabilities: OnceLock<SecondaryMap<FuncId, bool>>,
    pub(crate) arms: FactArms,
}

impl Facts {
    pub(crate) fn new(funcs: impl Iterator<Item = FuncId>) -> Self {
        let mut per = SecondaryMap::new();
        for f in funcs {
            per.insert(f, FuncFacts::default());
        }
        Facts {
            per,
            trap_capabilities: OnceLock::new(),
            arms: FactArms::default(),
        }
    }
}

impl CategoryIr {
    /// The fact arms this graph's facts are computed under.
    pub fn fact_arms(&self) -> FactArms {
        self.facts.arms
    }

    /// The same graph with an empty fact store computed under `arms` (an OFF
    /// arm's build: every fact derived afresh, none carried over).
    pub fn refacted(&self, arms: FactArms) -> CategoryIr {
        let mut facts = Facts::new(self.funcs.keys());
        facts.arms = arms;
        CategoryIr {
            objects: self.objects.clone(),
            morphisms: self.morphisms.clone(),
            out_edges: self.out_edges.clone(),
            in_edges: self.in_edges.clone(),
            owner: self.owner.clone(),
            funcs: self.funcs.clone(),
            entry: self.entry,
            facts,
            precision: self.precision.clone(),
        }
    }

    pub(crate) fn facts_of(&self, f: FuncId) -> Option<&FuncFacts> {
        self.facts.per.get(f)
    }

    /// `f`'s stored fact in `slot`, computing it the first time. A foreign
    /// `f` (no slot) is computed and not stored.
    pub(crate) fn fact<T: Clone>(
        &self,
        f: FuncId,
        slot: impl FnOnce(&FuncFacts) -> &OnceLock<T>,
        compute: impl FnOnce() -> T,
    ) -> T {
        match self.facts_of(f) {
            Some(s) => slot(s).get_or_init(compute).clone(),
            None => compute(),
        }
    }
}
