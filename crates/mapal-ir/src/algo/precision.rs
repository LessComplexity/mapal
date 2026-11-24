//! **`precision_plan`** (ADR-0041 D5): where a function's products admit TF32,
//! what may be rounded once, where a rounded value must not travel, and each
//! tile site's a-priori bound. One fact, stored on the graph; the backends read
//! `set` (the realization, D6) and `bound` (the gates).

use slotmap::SecondaryMap;

use crate::graph::{
    CategoryIr, FuncId, Morphism, MorphismId, Object, ObjectId, ObjectKind, Operation, Precision,
};
use crate::ty::{NarrowFloat, Ty};

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
    /// ADR-0046 D9: each math builtin (`exp`, `sqrt`, `sin`, `cos`) of the
    /// function with its width's contract — the default face's a-priori
    /// bound a value gate composes through the builtin
    /// ([`crate::math::MathForm::compose`]).
    pub math: SecondaryMap<MorphismId, crate::math::MathForm>,
    /// ADR-0043 D3: each `Narrow` (`to_bf16` / `to_f16`) of the function with
    /// its format's form — the rounding is exact on every face (IEEE
    /// `convertFormat`); a default-face gate adds `u·|x| + abs` per rounding
    /// whose operand carries that face's error (a flip across a tie).
    pub narrow: SecondaryMap<MorphismId, crate::narrow::NarrowForm>,
    /// `round_at` over arrays (sweep-gemm `round-store`): each f32 array of
    /// the function a `Map` produces whose every element read — through the
    /// captures of nested maps and folds — is a multiplicand of a product
    /// admitting the `bf16` member, so its elements may be rounded once as
    /// they are stored ([`CategoryIr::narrow_stored`]): half the bytes, the
    /// conversion out of every reader. `bf16` only: an `f16` member rounds no
    /// f32 operand (ADR-0043 D4 needs a value-range fact, none is stated).
    /// An array that escapes its function (`last_use_plan`) is never stored
    /// narrow: its caller's type is f32.
    pub store: Vec<NarrowStore>,
}

/// One array of [`PrecisionPlan::store`]: where it lives, every object that
/// carries it and every element read — what [`CategoryIr::narrow_stored`]
/// rewrites.
#[derive(Clone, Debug, PartialEq)]
pub struct NarrowStore {
    /// The `Map`'s f32 result.
    pub array: ObjectId,
    /// The member its readers round to.
    pub fmt: NarrowFloat,
    /// Each object carrying the array: the array or a projection of a
    /// capture (`None`), a product with the array in a slot (`Some(slot)`:
    /// a capture tuple, an `Index` source, a body's input).
    pub carriers: Vec<(ObjectId, Option<u32>)>,
    /// Each element read: the `Index` and its element.
    pub reads: Vec<(MorphismId, ObjectId)>,
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
        Some(ut) => bound_coef_units(k, ut, ut, 2f64.powi(-23)),
        None => gamma(k + 1, 2f64.powi(-24)),
    }
}

/// ADR-0043 slice 3 (F5): D3's coefficient with one unit roundoff per
/// multiplicand — `(1 + u_a)·(1 + u_b)·(1 + γ_{K+1}(u_acc)) − 1`. An operand
/// stored narrow in a format the member embeds exactly contributes `0`
/// ([`crate::TileSite::units`]): two `f8e4m3` operands under `: bf16 | f32`
/// get `γ_{K+1}(2⁻²³)`. Equal to [`bound_coef_unit`] at equal units.
pub fn bound_coef_units(k: u64, ua: f64, ub: f64, uacc: f64) -> f64 {
    (1.0 + ua) * (1.0 + ub) * (1.0 + gamma(k + 1, uacc)) - 1.0
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
            if let Operation::Math(fun) = mm.op
                && let Some(Ty::Float { bits }) = self.objects.get(mm.target).map(|o| &o.ty)
                && let Some(form) = crate::math::MathForm::of(fun, u32::from(*bits))
            {
                plan.math.insert(m, form);
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
        for o in self.func_objects(f) {
            if let Some(s) = self.narrow_store(o) {
                plan.store.push(s);
            }
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
            let coef = match s.units() {
                Some((ua, ub)) => bound_coef_units(s.k, ua, ub, 2f64.powi(-23)),
                None => bound_coef_unit(s.k, None),
            };
            plan.bound.insert(m, coef);
        }
        plan
    }

    /// [`PrecisionPlan::store`]'s entry for `o`: an f32 array one `Map`
    /// writes (its body's one f32 result), every element read of which is a
    /// multiplicand of a product admitting `bf16`; `None` when any use of the
    /// array (or a capture of it) is something else, or when it escapes its
    /// function (`last_use_plan`: a callee's return carries it to a caller
    /// that reads f32).
    fn narrow_store(&self, o: ObjectId) -> Option<NarrowStore> {
        let Some(Ty::Array { elem, .. }) = self.objects.get(o).map(|x| &x.ty) else {
            return None;
        };
        if **elem != Ty::f32() {
            return None;
        }
        let [m] = self.in_edges(o) else {
            return None;
        };
        let Operation::Map { body, .. } = self.morphisms[*m].op else {
            return None;
        };
        let r = self.func(body)?.output;
        if self.objects[r].ty != Ty::f32()
            || self.in_edges(r).len() != 1
            || !self.out_edges(r).is_empty()
        {
            return None;
        }
        let mut s = NarrowStore {
            array: o,
            fmt: NarrowFloat::Bf16,
            carriers: Vec::new(),
            reads: Vec::new(),
        };
        if !self.store_carriers(o, &mut s) || s.reads.is_empty() {
            return None;
        }
        // The array, or a product carrying it, outlives its function
        // (`last_use_plan` rule 2: reaches the fn's output through `Pair`
        // fields — an un-inlined callee's return): its caller reads the f32
        // array, so the bytes stay f32. A body's input is a borrowed
        // parameter (always escaping) and carries nothing out by itself.
        let escapes = s.carriers.iter().any(|&(c, _)| {
            self.objects[c].kind != ObjectKind::Parameter
                && self
                    .try_owner(c)
                    .is_some_and(|f| self.last_use_plan(f).escapes(c))
        });
        if escapes {
            return None;
        }
        // Every element feeds products only (through the product a binary op
        // takes, as `round_at`'s readers), each admitting `bf16`.
        let bf16 = |r: MorphismId| {
            self.morphisms[r].op == Operation::Mul
                && self.precision(r).half() == Some(NarrowFloat::Bf16)
        };
        for &(_, e) in &s.reads {
            let mut readers = Vec::new();
            for &u in self.out_edges(e) {
                let um = &self.morphisms[u];
                if matches!(um.op, Operation::Pair { .. }) {
                    readers.extend(self.out_edges(um.target).iter().copied());
                } else {
                    readers.push(u);
                }
            }
            if readers.is_empty() || !readers.into_iter().all(bf16) {
                return None;
            }
        }
        Some(s)
    }

    /// Walks every use of the array object `x` (the array, or a capture's
    /// projection of it): an `Index` source (an element read), a capture of a
    /// nested `Map` / `Fold` (into its body's projection), a projection of a
    /// product carrying it. Any other use: `false`.
    fn store_carriers(&self, x: ObjectId, s: &mut NarrowStore) -> bool {
        s.carriers.push((x, None));
        for &e in self.out_edges(x) {
            let em = &self.morphisms[e];
            let Operation::Pair { slot, .. } = em.op else {
                return false;
            };
            let t = em.target;
            s.carriers.push((t, Some(slot)));
            for &u in self.out_edges(t) {
                let um = &self.morphisms[u];
                match um.op {
                    Operation::Index if slot == 0 => s.reads.push((u, um.target)),
                    Operation::Map { body, captures } | Operation::Fold { body, captures }
                        if slot < captures =>
                    {
                        let Some(p) = self.func(body).map(|d| d.input) else {
                            return false;
                        };
                        s.carriers.push((p, Some(slot)));
                        for &v in self.out_edges(p) {
                            match self.morphisms[v].op {
                                Operation::Proj { index } if index == slot => {
                                    if !self.store_carriers(self.morphisms[v].target, s) {
                                        return false;
                                    }
                                }
                                Operation::Proj { .. } => {}
                                _ => return false,
                            }
                        }
                    }
                    Operation::Proj { index } if index == slot => {
                        if !self.store_carriers(um.target, s) {
                            return false;
                        }
                    }
                    Operation::Proj { .. } => {}
                    _ => return false,
                }
            }
        }
        true
    }

    /// The graph with every [`PrecisionPlan::store`] array stored in its
    /// member's format (sweep-gemm `round-store`): the producer's body rounds
    /// its result once (`Narrow`), each carrier's type names the narrow
    /// array, and each element read widens exactly (`Widen`) before its
    /// products — `bf16(x)` per multiplicand, as each product's member rounds
    /// it at the read. An array whose rewrite would take a map off a site
    /// plan it is on ([`Self::site_classes`]: a reader or producer that
    /// cannot take the narrow format — the decode's `P·V` over a stored `p`,
    /// the flash stream over stored `q` / `k`) stays f32: each is taken in
    /// turn, kept where every site keeps
    /// its plans. `None` when no array is taken. The facts are derived afresh
    /// under the same arms.
    pub fn narrow_stored(&self) -> Option<CategoryIr> {
        let stores: Vec<NarrowStore> = self
            .funcs
            .keys()
            .flat_map(|f| self.precision_plan(f).store)
            .collect();
        if stores.is_empty() {
            return None;
        }
        let before = self.site_classes();
        let mut taken: Vec<&NarrowStore> = Vec::new();
        let mut out = None;
        for s in &stores {
            let mut g = self.refacted(self.fact_arms());
            for t in taken.iter().copied().chain([s]) {
                g.apply_store(t);
            }
            let after = g.site_classes();
            if before.iter().all(|x| after.contains(x)) {
                taken.push(s);
                out = Some(g);
            }
        }
        out
    }

    /// Every `(plan, map, group)` triple of the graph: the affine tile sites
    /// (strided, epilogue and narrow plans: one class), the mixed-radix
    /// sites, the row and whole folds, the per-cell maps (`group` the map
    /// itself), each row-block unit's members with their group's first
    /// member (`unit_plan`: a member that leaves its group, or a group that
    /// splits, is a lost class), and a streamed group's members with f32
    /// operands — what [`Self::narrow_stored`] keeps.
    fn site_classes(&self) -> std::collections::BTreeSet<(u8, MorphismId, MorphismId)> {
        let mut out = std::collections::BTreeSet::new();
        for f in self.funcs.keys() {
            // One class: a site over a stored operand moves from the strided
            // or epilogue plan to the narrow one (the same sites over
            // narrow operands), which is not a loss.
            for t in [
                self.tile_plan_strided(f),
                self.tile_plan_post(f),
                self.tile_plan_narrow(f),
            ] {
                out.extend(t.sites.keys().map(|m| (0, m, m)));
            }
            out.extend(self.tile_plan_radix(f).keys().map(|m| (3, m, m)));
            out.extend(self.row_fold(f).keys().map(|m| (4, m, m)));
            out.extend(self.whole_fold(f).keys().map(|m| (5, m, m)));
            out.extend(self.cell_plan(f).keys().map(|m| (6, m, m)));
            let units = self.unit_plan(f);
            for g in &units.groups {
                let Some(lead) = g.members.first().map(|x| x.m) else {
                    continue;
                };
                out.extend(g.members.iter().map(|x| (7, x.m, lead)));
            }
            // A streamed group (`stream_plan`: the flash form) reads its tile
            // members' operands as f32 elements — a member whose operand
            // turns narrow leaves the stream (the backend's stream kernel
            // takes no narrow element yet).
            for sg in self.stream_plan(f).groups {
                let Some(g) = units.groups.get(sg.group) else {
                    continue;
                };
                for x in &g.members {
                    let wide = match &x.kind {
                        crate::UnitKind::Tile(s) => s.a.narrow.is_none() && s.b.narrow.is_none(),
                        _ => true,
                    };
                    if wide {
                        out.insert((8, x.m, x.m));
                    }
                }
            }
        }
        out
    }

    fn apply_store(&mut self, s: &NarrowStore) {
        let Ty::Array { size, .. } = self.objects[s.array].ty else {
            unreachable!("a stored array")
        };
        let arr = Ty::Array {
            elem: Box::new(Ty::Narrow(s.fmt)),
            size,
        };
        for &(o, slot) in &s.carriers {
            let ty = &mut self.objects[o].ty;
            match (slot, ty) {
                (None, ty) => *ty = arr.clone(),
                (Some(i), Ty::Tuple(ts)) => ts[i as usize] = arr.clone(),
                (Some(_), ty) => unreachable!("a carrier product: {ty:?}"),
            }
        }
        // The producer rounds its result once.
        let m = self.in_edges(s.array)[0];
        let Operation::Map { body, .. } = self.morphisms[m].op else {
            unreachable!("a Map producer")
        };
        let r = self.funcs[body].output;
        let w = self.in_edges(r)[0];
        let loc = self.morphisms[w].loc;
        let r0 = self.mint_temp(body, Ty::f32(), None, loc);
        self.morphisms[w].target = r0;
        self.in_edges.insert(r0, vec![w]);
        self.objects[r].ty = Ty::Narrow(s.fmt);
        let n = self.mint_edge(body, r0, r, Operation::Narrow, loc, w);
        self.in_edges.insert(r, vec![n]);
        self.out_edges.insert(r0, vec![n]);
        // Each read widens exactly before its products.
        for &(ix, e) in &s.reads {
            let f = self.owner[e];
            let loc = self.objects[e].loc;
            let name = self.objects[e].name.take();
            let e1 = self.mint_temp(f, Ty::f32(), name, loc);
            let outs = std::mem::take(&mut self.out_edges[e]);
            for &u in &outs {
                self.morphisms[u].source = e1;
            }
            self.out_edges.insert(e1, outs);
            self.objects[e].ty = Ty::Narrow(s.fmt);
            let n = self.mint_edge(f, e, e1, Operation::Widen, loc, ix);
            self.out_edges.insert(e, vec![n]);
            self.in_edges.insert(e1, vec![n]);
        }
    }

    fn mint_temp(
        &mut self,
        f: FuncId,
        ty: Ty,
        name: Option<String>,
        loc: crate::SourceLoc,
    ) -> ObjectId {
        let o = self.objects.insert_with_key(|id| Object {
            id,
            ty,
            value: None,
            kind: ObjectKind::Temporary,
            name,
            loc,
        });
        self.owner.insert(o, f);
        self.out_edges.insert(o, Vec::new());
        self.in_edges.insert(o, Vec::new());
        o
    }

    /// A morphism `source → target` in `f`, listed right after `after` (a
    /// valid construction order). The caller wires the edge lists.
    fn mint_edge(
        &mut self,
        f: FuncId,
        source: ObjectId,
        target: ObjectId,
        op: Operation,
        loc: crate::SourceLoc,
        after: MorphismId,
    ) -> MorphismId {
        let m = self.morphisms.insert_with_key(|id| Morphism {
            id,
            source,
            target,
            op,
            loc,
        });
        let ms = &mut self.funcs[f].morphisms;
        let at = ms
            .iter()
            .position(|&x| x == after)
            .map_or(ms.len(), |i| i + 1);
        ms.insert(at, m);
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F5 reduces to the shared-unit form at equal units, and an exactly
    /// embedded operand drops its factor.
    #[test]
    fn per_operand_units() {
        for k in [1u64, 64, 4096] {
            for u in [2f64.powi(-8), 2f64.powi(-11)] {
                assert_eq!(
                    bound_coef_units(k, u, u, 2f64.powi(-23)),
                    bound_coef_unit(k, Some(u))
                );
                assert!(bound_coef_units(k, 0.0, u, 2f64.powi(-23)) < bound_coef_unit(k, Some(u)));
            }
            let (x, g) = (
                bound_coef_units(k, 0.0, 0.0, 2f64.powi(-23)),
                gamma(k + 1, 2f64.powi(-23)),
            );
            assert!((x - g).abs() <= 1e-9 * g, "{x} {g}");
        }
    }
}
