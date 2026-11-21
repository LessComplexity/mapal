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
    pub(crate) guard_plan: OnceLock<Vec<GuardSite>>,
    pub(crate) guard_structure: OnceLock<Vec<GuardSite>>,
    pub(crate) trip_split: OnceLock<Vec<TripSplit>>,
    pub(crate) unit_plan: OnceLock<crate::unit::UnitPlan>,
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
}

impl Default for FactArms {
    fn default() -> Self {
        FactArms { cond_bounds: true }
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
