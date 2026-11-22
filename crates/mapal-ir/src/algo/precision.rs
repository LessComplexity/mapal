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
    /// `round_at(v)`: values whose every reader is a product admitting a
    /// reduced member (`tf32`, `bf16`, `f16`) — rounded once at the producer,
    /// not per use (compute once).
    pub round_at: Vec<ObjectId>,
    /// `cross(e)`: `(v, reader)` where `v` also feeds a `tf32` product but
    /// `reader` does not admit `tf32` — no rounded copy of `v` may reach it.
    pub cross: Vec<(ObjectId, MorphismId)>,
    /// `bound(site)`: each tile site's coefficient of ADR-0040's
    /// `M = |seed| + Σ|a·b|` — D3's `(1 + u_t)²·(1 + γ_{K+1}(u_a)) − 1` where the
    /// product admits a reduced member (`u_t` its weakest's: `tf32` / `f16`
    /// 2⁻¹¹, `bf16` 2⁻⁸; `u_a = 2⁻²³`: a tensor-core accumulator), else D2's
    /// `γ_{K+1}(2⁻²⁴)` ([`bound_coef_unit`]).
    pub bound: SecondaryMap<MorphismId, f64>,
    /// ADR-0042 D6: each `Exp` of the function with its width's form — the
    /// oracle's constants and the default face's a-priori bound
    /// (`|got − eˣ| ≤ rel·eˣ + abs` on `[lo, hi]`).
    pub exp: SecondaryMap<MorphismId, crate::exp::ExpForm>,
    /// ADR-0043 D3: each `Narrow` (`to_bf16` / `to_f16`) of the function with
    /// its format's form — the rounding is exact on every face (IEEE
    /// `convertFormat`); a default-face gate adds `u·|x| + abs` per rounding
    /// whose operand carries that face's error (a flip across a tie).
    pub narrow: SecondaryMap<MorphismId, crate::narrow::NarrowForm>,
}

/// `γ_n(u) = n·u / (1 − n·u)`.
pub fn gamma(n: u64, u: f64) -> f64 {
    let nu = n as f64 * u;
    nu / (1.0 - nu)
}

/// The bound coefficient of a `k`-step fold whose products admit `tf32` or not.
pub fn bound_coef(k: u64, tf32: bool) -> f64 {
    bound_coef_unit(k, tf32.then(|| 2f64.powi(-11)))
}

/// The bound coefficient of a `k`-step fold whose weakest member rounds each
/// multiplicand with unit roundoff `ut` ([`Precision::unit`]): ADR-0041 D3's
/// `(1 + u_t)²·(1 + γ_{K+1}(u_a)) − 1` with a tensor-core accumulator's
/// `u_a = 2⁻²³` (one that truncates its aligned addends: the RTX 5080's
/// `mma.sync` bf16 / f16 / fp8, `benches/local/mma_calib.cu`); `None` (exact
/// f32 products): D2's `γ_{K+1}(2⁻²⁴)`.
pub fn bound_coef_unit(k: u64, ut: Option<f64>) -> f64 {
    match ut {
        Some(ut) => (1.0 + ut) * (1.0 + ut) * (1.0 + gamma(k + 1, 2f64.powi(-23))) - 1.0,
        None => gamma(k + 1, 2f64.powi(-24)),
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
            let mm = &self.morphisms[m];
            if mm.op == Operation::Exp
                && let Some(Ty::Float { bits }) = self.objects.get(mm.target).map(|o| &o.ty)
                && let Some(form) = crate::exp::ExpForm::of_bits(u32::from(*bits))
            {
                plan.exp.insert(m, form);
            }
            if mm.op == Operation::Narrow
                && let Some(Ty::Narrow(n)) = self.objects.get(mm.target).map(|o| &o.ty)
            {
                plan.narrow.insert(m, crate::narrow::NarrowForm::of(*n));
            }
        }
        // A value's readers: through the product a binary op takes (`Pair`
        // into its source tuple), the op; any other consumer as itself.
        let tf32 = |m: MorphismId| self.precision(m).reduced();
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
        let post = self.tile_plan_post(f);
        // ADR-0043: a narrow site's products read exact widenings — the f32
        // site's bound.
        let narrow = self.tile_plan_narrow(f);
        for (m, s) in strided
            .sites
            .iter()
            .chain(radix.iter())
            .chain(post.sites.iter())
            .chain(narrow.sites.iter())
        {
            plan.bound.insert(m, bound_coef_unit(s.k, s.unit()));
        }
        plan
    }
}
