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
    LoopPlan, LoopScc, MovePlan, PathPlan, RadixForm, RowFoldSite, TilePlan, TileSite,
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
    pub(crate) move_plan: OnceLock<MovePlan>,
    pub(crate) row_fold: OnceLock<SecondaryMap<MorphismId, RowFoldSite>>,
    pub(crate) cell_plan: OnceLock<SecondaryMap<MorphismId, CellSite>>,
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
}

/// The graph's fact store.
#[derive(Default)]
pub(crate) struct Facts {
    per: SecondaryMap<FuncId, FuncFacts>,
    pub(crate) trap_capabilities: OnceLock<SecondaryMap<FuncId, bool>>,
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
        }
    }
}

impl CategoryIr {
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
