//! **`step_algebra`** (fold-algebra): what a function's arithmetic may become —
//! which float products may fuse with the add that reads them (one rounding,
//! FMA) and whether a fold body's step may be regrouped (lane partials + a
//! tree). Admission is algebra and dataflow only; the read geometry along the
//! fold axis is recorded for the layout a consumer picks, never for admission.
//! One fact, stored on the graph; the host emitter and the device kernels read
//! it, so both realize the same arithmetic.

use slotmap::SecondaryMap;

use crate::algo::RowFoldOp;
use crate::graph::{CategoryIr, FuncId, MorphismId, ObjectId, Operation};
use crate::ty::Ty;

/// A float `Add`/`Sub` that reads the product of a float `Mul` read by nothing
/// else: the pair may be one fused rounding where the face contracts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FmaPair {
    /// The `Mul` whose product is the fused operand.
    pub mul: MorphismId,
    /// The operand slot of the add/sub the product sits in (the first one
    /// when both operands are such products: one pair per add).
    pub slot: u32,
    /// The pair is `x − y` (`slot 0`: `a·b − c`, `slot 1`: `c − a·b`).
    pub sub: bool,
}

/// A fold body's step that may run as partial folds combined by its own `⊕`
/// ([`CategoryIr::fold_tree_step`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Regroup {
    /// The `Fold` morphism whose body this is.
    pub fold: MorphismId,
    /// The step morphism in the body (`acc ⊕ e`, or the max/min's `Phi`).
    pub step: MorphismId,
    pub op: RowFoldOp,
    /// Regrouping keeps the bits (integer steps). A float step's partials
    /// round differently: admitted where the order is free (ADR-0040 D2).
    pub exact: bool,
    /// Trip count (the folded array's length).
    pub k: u64,
    /// Per read of the step's term (`e`, or the factors when `e` is an
    /// [`FmaPair`] product): its stride along the fold axis — `Some(1)` the
    /// folded element itself, `Some(ck)` a whole fold's affine read, `None`
    /// unknown (a gather or a non-affine read). Layout, not admission.
    pub reads: Vec<Option<u64>>,
}

/// [`CategoryIr::step_algebra`]'s fields for one function.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepAlgebra {
    /// Keyed by the add/sub of each pair.
    pub fma: SecondaryMap<MorphismId, FmaPair>,
    /// Keyed by each pair's `Mul`: its add/sub.
    pub fused_mul: SecondaryMap<MorphismId, MorphismId>,
    /// Set when the function is a fold body whose step may be regrouped.
    pub regroup: Option<Regroup>,
}

/// The fast-math flags one realization gives a morphism: `contract` on both
/// halves of an [`FmaPair`] where the face contracts, `reassoc` on a float
/// regroup step where the order is free.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepFlags {
    pub contract: bool,
    pub reassoc: bool,
}

impl StepAlgebra {
    /// The one realization rule both backends read. `contract`: the face fuses
    /// steps (ADR-0040 D1); `free`: the order is unspecified; `pairs` /
    /// `regroup`: the two OFF arms (`--no-fma-pairs`, `--no-fold-regroup`).
    pub fn flags(
        &self,
        m: MorphismId,
        contract: bool,
        free: bool,
        pairs: bool,
        regroup: bool,
    ) -> StepFlags {
        StepFlags {
            contract: contract
                && pairs
                && (self.fma.contains_key(m) || self.fused_mul.contains_key(m)),
            reassoc: free
                && regroup
                && self
                    .regroup
                    .as_ref()
                    .is_some_and(|g| g.step == m && !g.exact),
        }
    }
}

impl CategoryIr {
    /// The deduced query `step_algebra : IR × FuncId → StepAlgebra`. Stored on
    /// the graph.
    pub fn step_algebra(&self, f: FuncId) -> StepAlgebra {
        self.fact(f, |s| &s.step_algebra, || self.step_algebra_compute(f))
    }

    fn step_algebra_compute(&self, f: FuncId) -> StepAlgebra {
        let mut alg = StepAlgebra::default();
        let Some(fd) = self.func(f) else {
            return alg;
        };
        for &s in &fd.morphisms {
            if let Some(p) = self.fma_pair(s) {
                alg.fused_mul.insert(p.mul, s);
                alg.fma.insert(s, p);
            }
        }
        alg.regroup = self.regroup_of(f, &alg);
        alg
    }

    fn is_float_obj(&self, o: ObjectId) -> bool {
        matches!(self.objects.get(o).map(|x| &x.ty), Some(Ty::Float { .. }))
    }

    /// `s` is a float add/sub one of whose operands is a single-reader float
    /// product, and `s` is the only reader of its operand pair.
    fn fma_pair(&self, s: MorphismId) -> Option<FmaPair> {
        let sm = self.morphisms.get(s)?;
        if !matches!(sm.op, Operation::Add | Operation::Sub) || !self.is_float_obj(sm.target) {
            return None;
        }
        if self.out_edges(sm.source) != [s] {
            return None;
        }
        (0..2).find_map(|slot| {
            let x = self.pair_slot_source(sm.source, slot)?;
            let &[mul] = self.in_edges(x) else {
                return None;
            };
            let mm = self.morphisms.get(mul)?;
            if mm.op != Operation::Mul || !self.is_float_obj(x) {
                return None;
            }
            // The product feeds this pair's one slot and nothing else.
            let &[e] = self.out_edges(x) else {
                return None;
            };
            let pe = &self.morphisms[e];
            (matches!(pe.op, Operation::Pair { slot: k, .. } if k == slot)
                && pe.target == sm.source)
                .then_some(FmaPair {
                    mul,
                    slot,
                    sub: sm.op == Operation::Sub,
                })
        })
    }

    /// `f`'s regroup when it is the body of a fold whose step admits partials.
    fn regroup_of(&self, f: FuncId, alg: &StepAlgebra) -> Option<Regroup> {
        let (fold, captures) = self.morphisms.iter().find_map(|(m, x)| match x.op {
            Operation::Fold { body, captures } if body == f => Some((m, captures)),
            _ => None,
        })?;
        let op = self.fold_tree_step(fold)?;
        let fd = self.func(f)?;
        let &[step] = self.in_edges(fd.output) else {
            return None;
        };
        let fm = &self.morphisms[fold];
        let k = match self.objects.get(fm.source)?.ty.component_ty(captures + 1)? {
            Ty::Array { size, .. } => *size,
            _ => return None,
        };
        let exact = !self.is_float_obj(fd.output);
        // The term `e`: the step's non-accumulator operand (a max/min's other
        // compared value); its reads along the fold axis.
        let acc = self
            .out_edges(fd.input)
            .iter()
            .map(|&p| &self.morphisms[p])
            .find(|p| p.op == (Operation::Proj { index: captures }))
            .map(|p| p.target)?;
        let elem = self
            .out_edges(fd.input)
            .iter()
            .map(|&p| &self.morphisms[p])
            .find(|p| {
                p.op == (Operation::Proj {
                    index: captures + 1,
                })
            })
            .map(|p| p.target);
        let sm = &self.morphisms[step];
        let term = match op {
            RowFoldOp::Arith { acc_first, .. } => {
                self.pair_slot_source(sm.source, u32::from(acc_first))?
            }
            RowFoldOp::Select { .. } => {
                let (p, q) = (
                    self.pair_slot_source(sm.source, 0)?,
                    self.pair_slot_source(sm.source, 1)?,
                );
                if p == acc { q } else { p }
            }
        };
        let factors: Vec<ObjectId> = match alg.fma.get(step) {
            Some(pair) => {
                let mm = &self.morphisms[pair.mul];
                vec![
                    self.pair_slot_source(mm.source, 0)?,
                    self.pair_slot_source(mm.source, 1)?,
                ]
            }
            None => vec![term],
        };
        let whole = self.whole_site_of(fold).map(|site| site.a.ck);
        // The folded element, or a component of it (a zipped pair's `p.0`),
        // advances one element per step.
        let of_elem = |mut o: ObjectId| loop {
            if Some(o) == elem {
                return true;
            }
            match self.in_edges(o) {
                [p] if matches!(self.morphisms[*p].op, Operation::Proj { .. }) => {
                    o = self.morphisms[*p].source;
                }
                _ => return false,
            }
        };
        let reads = factors
            .iter()
            .map(|&o| if of_elem(o) { Some(1) } else { whole })
            .collect();
        Some(Regroup {
            fold,
            step,
            op,
            exact,
            k,
            reads,
        })
    }

    /// The whole-fold site of `fold` in its own function, if it is one.
    fn whole_site_of(&self, fold: MorphismId) -> Option<crate::algo::RowFoldSite> {
        let owner = self.try_owner(self.morphisms.get(fold)?.target)?;
        self.whole_fold(owner).get(fold).cloned()
    }
}
