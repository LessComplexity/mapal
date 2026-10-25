//! **`precision_plan`** (ADR-0041 D5): where a function's products admit TF32,
//! what may be rounded once, where a rounded value must not travel, and each
//! tile site's a-priori bound. One fact, stored on the graph; the backends read
//! `set` (the realization, D6) and `bound` (the gates).

use slotmap::SecondaryMap;

use crate::graph::{CategoryIr, FuncId, MorphismId, ObjectId, Operation, Precision};
use crate::ty::Ty;

/// ADR-0041 D5's fields for one function.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrecisionPlan {
    /// `set(m)`: every float product of the function whose set is not `{f32}`
    /// (the innermost `: compute` around the body it was lowered from).
    pub set: SecondaryMap<MorphismId, Precision>,
    /// `round_at(v)`: values whose every reader is a product admitting `tf32`
    /// — rounded once at the producer, not per use (compute once).
    pub round_at: Vec<ObjectId>,
    /// `cross(e)`: `(v, reader)` where `v` also feeds a `tf32` product but
    /// `reader` does not admit `tf32` — no rounded copy of `v` may reach it.
    pub cross: Vec<(ObjectId, MorphismId)>,
    /// `bound(site)`: each tile site's coefficient of ADR-0040's
    /// `M = |seed| + Σ|a·b|` — D3's `(1 + u_t)²·(1 + γ_{K+1}(u_a)) − 1` where the
    /// product admits `tf32` (`u_t = 2⁻¹¹`, `u_a = 2⁻²³`: a tensor-core
    /// accumulator), else D2's `γ_{K+1}(2⁻²⁴)`.
    pub bound: SecondaryMap<MorphismId, f64>,
}

/// `γ_n(u) = n·u / (1 − n·u)`.
pub fn gamma(n: u64, u: f64) -> f64 {
    let nu = n as f64 * u;
    nu / (1.0 - nu)
}

/// The bound coefficient of a `k`-step fold whose products admit `tf32` or not.
pub fn bound_coef(k: u64, tf32: bool) -> f64 {
    if tf32 {
        let ut = 2f64.powi(-11);
        (1.0 + ut) * (1.0 + ut) * (1.0 + gamma(k + 1, 2f64.powi(-23))) - 1.0
    } else {
        gamma(k + 1, 2f64.powi(-24))
    }
}

impl CategoryIr {
    /// The deduced query `precision : IR × FuncId → PrecisionPlan` (ADR-0041
    /// D5). Stored on the graph.
    pub fn precision_plan(&self, f: FuncId) -> PrecisionPlan {
        self.fact(f, |s| &s.precision_plan, || self.precision_plan_compute(f))
    }

    fn precision_plan_compute(&self, f: FuncId) -> PrecisionPlan {
        let mut plan = PrecisionPlan::default();
        let Some(fd) = self.func(f) else {
            return plan;
        };
        for &m in &fd.morphisms {
            let p = self.precision(m);
            if p != Precision::F32 {
                plan.set.insert(m, p);
            }
        }
        // A value's readers: through the product a binary op takes (`Pair`
        // into its source tuple), the op; any other consumer as itself.
        let tf32 = |m: MorphismId| self.precision(m).admits_tf32();
        for o in self.func_objects(f) {
            if !matches!(self.objects.get(o).map(|x| &x.ty), Some(Ty::Float { .. })) {
                continue;
            }
            let mut readers = Vec::new();
            for &e in self.out_edges(o) {
                let em = &self.morphisms[e];
                if matches!(em.op, Operation::Pair { .. }) {
                    readers.extend(self.out_edges(em.target).iter().copied());
                } else {
                    readers.push(e);
                }
            }
            let (on, off): (Vec<_>, Vec<_>) = readers.into_iter().partition(|&r| tf32(r));
            if on.is_empty() {
                continue;
            }
            if off.is_empty() {
                plan.round_at.push(o);
            }
            plan.cross.extend(off.into_iter().map(|r| (o, r)));
        }
        let strided = self.tile_plan_strided(f);
        let radix = self.tile_plan_radix(f);
        for (m, s) in strided.sites.iter().chain(radix.iter()) {
            plan.bound.insert(m, bound_coef(s.k, s.tf32));
        }
        plan
    }
}
