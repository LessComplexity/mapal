//! Element laws (`elem_plan`, `map_law`) and the interval / trap-freedom helpers they read.

use super::*;

impl CategoryIr {
    /// The deduced query `elem : IR × FuncId → ElemPlan` — for each array in
    /// `f`, what `out[i]` is as a closed function of the index
    /// (plan-s37-stage-structure §3.1).
    ///
    /// The law is the unique homomorphism from the producer DAG into the
    /// [`ElemSrc`] term algebra: `Iota ↦ Index`, `Fill ↦ Broadcast`,
    /// `Zip ↦ Pair(·,·)`, `Enumerate ↦ Pair(Index, ·)`, **everything else ↦ the
    /// cut**. Stage composition is nothing more than the recursion continuing
    /// past a cut: `Consumer[mid[i]] ∘ Producer{L} = Consumer[L(i)]`.
    ///
    /// Producers are recognized by an exact op-tag set rather than by "carries
    /// no body", because trap-freedom is a documented guarantee of these four
    /// tags specifically (`graph.rs`) — a future bodyless-looking op, or a
    /// `Call` whose totality is underivable, must not be admitted by shape.
    pub fn elem_plan(&self, f: FuncId) -> ElemPlan {
        self.fact(f, |s| &s.elem_plan, || self.elem_plan_compute(f))
    }

    pub(super) fn elem_plan_compute(&self, f: FuncId) -> ElemPlan {
        let mut plan = ElemPlan::default();
        let Some(def) = self.func(f) else {
            return plan;
        };
        let mut in_loop: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        for scc in self.loop_structure(f) {
            for o in scc.objects {
                in_loop.insert(o, ());
            }
        }
        for &m in &def.morphisms {
            let Some(morph) = self.morphisms.get(m) else {
                continue;
            };
            if !matches!(
                morph.op,
                Operation::Iota
                    | Operation::Fill
                    | Operation::Zip
                    | Operation::Enumerate
                    | Operation::Map { .. }
                    | Operation::Proj { .. }
            ) {
                continue;
            }
            if let Some(law) = self.elem_law(morph.target, &in_loop, 0) {
                plan.src.insert(morph.target, law);
            }
        }
        plan
    }

    /// gapfix `iota-law`: **an element law through a capture** — the law of
    /// a body fn's capture `Proj` of its input parameter, from the value every
    /// site of the body passes in that slot (its own fn's [`Self::elem_plan`],
    /// itself through captures one level further out). Only a law that names
    /// nothing of the enclosing fn crosses (`Index`, and pairs of such): a
    /// `Broadcast` / `Load` / `Apply` refers to objects the body cannot reach.
    /// Every site must agree (a body shared by sites passing different
    /// arrays has no one law). `None`: the cut — the capture is loaded.
    pub(super) fn capture_law(&self, o: ObjectId) -> Option<ElemSrc> {
        let m = self.in_edges(o).first().copied()?;
        let pm = self.morphisms.get(m)?;
        let Operation::Proj { index } = pm.op else {
            return None;
        };
        if self.objects.get(pm.source)?.kind != ObjectKind::Parameter {
            return None;
        }
        let of = self.try_owner(pm.source)?;
        if self.func(of)?.input != pm.source {
            return None;
        }
        fn closed(l: &ElemSrc) -> bool {
            match l {
                ElemSrc::Index => true,
                ElemSrc::Pair(a, b) => closed(a) && closed(b),
                _ => false,
            }
        }
        let mut law: Option<ElemSrc> = None;
        for (_, morph) in self.morphisms() {
            let k = match morph.op {
                Operation::Map { body, captures } | Operation::Fold { body, captures }
                    if body == of =>
                {
                    captures
                }
                _ => continue,
            };
            // Capture positions are 0..k (map: element at k; fold: acc at k).
            if index >= k {
                return None;
            }
            let outer = self.pair_slot_source(morph.source, index)?;
            let owner = self.try_owner(outer)?;
            let l = self.elem_plan(owner).src(outer).cloned().filter(closed)?;
            match &law {
                Some(p) if *p != l => return None,
                _ => law = Some(l),
            }
        }
        law
    }

    /// `elem_law_local` plus the three guards, recursing through structural
    /// producers. `None` is the cut.
    ///
    /// Guards, each preventing a specific wrong answer: a **single in-edge**
    /// (a multi-producer object has no unique law to substitute); **outside
    /// every loop SCC** (a loop-carried array's value differs per iteration, so
    /// recomputing at read time would observe a different iteration than the
    /// load did); and a **depth cap of 16** (the `tile_iota_size` /
    /// `element_range` precedent — bounds query cost, and cutting early is
    /// always sound). The cap is what keeps this within J1's spirit: recursion
    /// depth is bounded by a constant, never by graph size.
    pub(super) fn elem_law(
        &self,
        arr: ObjectId,
        in_loop: &SecondaryMap<ObjectId, ()>,
        depth: u32,
    ) -> Option<ElemSrc> {
        if depth >= 16 || in_loop.contains_key(arr) {
            return None;
        }
        let [definer] = self.in_edges(arr) else {
            return None;
        };
        let morph = self.morphisms.get(*definer)?;
        match morph.op {
            Operation::Iota | Operation::Fill => self.elem_law_local(arr),
            // gapfix `iota-law`: a capture of a body fn, by the law of what
            // its sites pass (a fold over a captured `iota`: no load).
            Operation::Proj { .. } => self.capture_law(arr),
            // `Zip`'s source is the 2-tuple product of the two arrays (ADR-0018).
            Operation::Zip => Some(ElemSrc::Pair(
                Box::new(self.elem_component(morph.source, 0, in_loop, depth + 1)),
                Box::new(self.elem_component(morph.source, 1, in_loop, depth + 1)),
            )),
            // `Enumerate`'s source is the bare array; the index half is free.
            Operation::Enumerate => Some(ElemSrc::Pair(
                Box::new(ElemSrc::Index),
                Box::new(self.elem_law(morph.source, in_loop, depth + 1).unwrap_or(
                    ElemSrc::Load {
                        source: morph.source,
                        slot: None,
                    },
                )),
            )),
            // `Map` joins only when its body is classifiable — see the
            // `ElemSrc::Apply` doc for why it is not a peer of the other four.
            Operation::Map { captures, .. } => {
                let mapped = if captures == 0 {
                    morph.source
                } else {
                    self.pair_slot_source(morph.source, captures)?
                };
                let inner = self.elem_law(mapped, in_loop, depth + 1);
                self.map_apply(*definer, inner)
            }
            _ => None,
        }
    }

    /// **One `Map`'s element law, its input cut** (generic-fusion): what
    /// `out[t]` is in terms of the mapped element as loaded — `elem_plan`'s
    /// `Map` arm without the loop-residence guard, so a map inside a loop
    /// (a row write into a carried state) has one too. `None` unless `m` is a
    /// `Map` whose body is classifiable.
    pub fn map_law(&self, m: MorphismId) -> Option<ElemSrc> {
        self.map_apply(m, None)
    }

    /// The `Apply` of map `m` over `inner` (`None`: the mapped element loaded).
    pub(super) fn map_apply(&self, m: MorphismId, inner: Option<ElemSrc>) -> Option<ElemSrc> {
        let morph = self.morphisms.get(m)?;
        let Operation::Map { body, captures } = morph.op else {
            return None;
        };
        if !self.body_is_classifiable(body) {
            return None;
        }
        let inner = inner.unwrap_or(ElemSrc::Load {
            source: morph.source,
            slot: (captures > 0).then_some(captures),
        });
        Some(ElemSrc::Apply {
            body,
            source: morph.source,
            captures,
            inner: Box::new(inner),
            array: morph.target,
            outside: self.map_outside(m),
        })
    }

    /// [`Outside`] of map `m`: its mapped array is an `iota` of the output's
    /// length (the element is the index `t`), its body's output a `Phi` on an
    /// index interval of `t` whose other arm is `x[t]`, `x` a captured array
    /// of the output's type read at `t` only.
    pub(super) fn map_outside(&self, m: MorphismId) -> Option<Outside> {
        let morph = self.morphisms.get(m)?;
        let Operation::Map { body, captures } = morph.op else {
            return None;
        };
        if captures == 0 {
            return None;
        }
        let out_ty = &self.objects.get(morph.target)?.ty;
        let Ty::Array { size: n, .. } = *out_ty else {
            return None;
        };
        if self.tile_iota_size(self.pair_slot_source(morph.source, captures)?) != Some(n) {
            return None;
        }
        let bd = self.func(body)?;
        let input = bd.input;
        let phi = self.tile_definer(bd.output, Operation::Phi)?;
        let arm = |k| self.pair_slot_source(phi.source, k);
        let (iv, inside_true) = self.index_interval(arm(2)?, input, captures, morph.source)?;
        let outer = arm(if inside_true { 1 } else { 0 })?;
        let ix = self.tile_definer(outer, Operation::Index)?;
        let slot = self
            .tile_input_proj_index(self.pair_slot_source(ix.source, 0)?, input)
            .filter(|&s| s < captures)?;
        let x = self.pair_slot_source(morph.source, slot)?;
        if self.objects.get(x)?.ty != *out_ty || !self.reads_only_at_t(bd, slot, captures) {
            return None;
        }
        // `lo + len` must not wrap where the interval meets `[0, n)`.
        if iv.d == 1 && iv.step == 1 && n.saturating_sub(1).saturating_add(iv.len) > i32::MAX as u64
        {
            return None;
        }
        // A piece never straddles the array's end.
        if iv.w > 1 && !n.is_multiple_of(iv.w) {
            return None;
        }
        let e = match iv.e {
            Some(k) => Some(self.pair_slot_source(morph.source, k)?),
            None => None,
        };
        Some(Outside {
            slot,
            d: iv.d,
            a: iv.a,
            e,
            b: iv.b,
            step: iv.step,
            len: iv.len,
            w: iv.w,
        })
    }

    /// Every use of capture `slot` in `bd` is an `Index` at the element `t`.
    pub(super) fn reads_only_at_t(
        &self,
        bd: &crate::graph::FuncDef,
        slot: u32,
        captures: u32,
    ) -> bool {
        self.out_edges(bd.input).iter().all(|&p| {
            let x = &self.morphisms[p];
            x.op != (Operation::Proj { index: slot })
                || self.out_edges(x.target).iter().all(|&q| {
                    let y = &self.morphisms[q];
                    matches!(y.op, Operation::Pair { slot: 0, .. })
                        && self.out_edges(y.target).iter().all(|&r| {
                            let z = &self.morphisms[r];
                            z.op == Operation::Index
                                && self
                                    .pair_slot_source(z.source, 1)
                                    .and_then(|a| self.tile_input_proj_index(a, bd.input))
                                    == Some(captures)
                        })
                })
        })
    }

    /// `cond` over the map element `t` (`Proj{captures}` of `input`) as a set
    /// of cells, and whether `t` is in it when `cond` holds: `t == X` / `!=`
    /// (`[X, X + 1)`), `t ÷ d == X` / `!=` (`[d·X, d·X + d)`), `t % s == X` /
    /// `!=` (`X + s·k`), or
    /// `L ≤ t ∧ t < L + len` in either operand order (`t ≥ L`, `L ≤ t`,
    /// `t > L − 1`; `t < H`, `H > t`, `t ≤ H − 1`), `X`, `L` affine in one
    /// captured int scalar.
    pub(super) fn index_interval(
        &self,
        cond: ObjectId,
        input: ObjectId,
        captures: u32,
        site: ObjectId,
    ) -> Option<(Interval, bool)> {
        let c = self.morphisms.get(*self.in_edges(cond).first()?)?;
        let (x, y) = (
            self.pair_slot_source(c.source, 0)?,
            self.pair_slot_source(c.source, 1)?,
        );
        let is_t = |o: ObjectId| self.tile_input_proj_index(o, input) == Some(captures);
        match c.op {
            Operation::Eq | Operation::Neq => {
                // `t` (one cell), `t ÷ d` (a row: `d` cells from `d·X`) or
                // `t % s` (a column: every `s`-th cell from `X`), `d`, `s`
                // literals.
                let lit = |o: ObjectId| self.tile_literal_u64(o).filter(|&k| k > 0);
                let split_of_t = |o: ObjectId| {
                    if is_t(o) {
                        return Some((1, 1, 1, 1));
                    }
                    let (op, g) = [Operation::Div, Operation::Mod]
                        .into_iter()
                        .find_map(|op| Some((op, self.tile_definer(o, op)?)))?;
                    let k = lit(self.pair_slot_source(g.source, 1)?)?;
                    let inner = self.pair_slot_source(g.source, 0)?;
                    if !is_t(inner) {
                        // `piece-delta`: `(t ÷ d) % s` — a row of `d` cells
                        // every `s` rows (pieces of `w = d`).
                        let dv = self.tile_definer(inner, Operation::Div)?;
                        let d = lit(self.pair_slot_source(dv.source, 1)?)?;
                        let ok = op == Operation::Mod
                            && self.fact_arms().piece_delta
                            && is_t(self.pair_slot_source(dv.source, 0)?);
                        return ok.then_some((d, d.checked_mul(k)?, u64::MAX, d));
                    }
                    Some(if op == Operation::Div {
                        (k, 1, k, 1)
                    } else {
                        (1, k, u64::MAX, 1)
                    })
                };
                let ((d, step, len, w), other) = match split_of_t(x) {
                    Some(s) => (s, y),
                    None => (split_of_t(y)?, x),
                };
                let mut e = None;
                let (a, b) = self.capture_affine(other, input, captures, site, &mut e, 0)?;
                Some((
                    Interval {
                        d,
                        a,
                        e,
                        b,
                        step,
                        len,
                        w,
                    },
                    c.op == Operation::Eq,
                ))
            }
            Operation::And => {
                let (bx, by) = (
                    self.t_bound(x, input, captures, site)?,
                    self.t_bound(y, input, captures, site)?,
                );
                let ((e1, a1, b1, d1), (e2, a2, b2, d2)) = match (bx, by) {
                    (Bound::Lo(e1, a1, b1, d1), Bound::Hi(e2, a2, b2, d2))
                    | (Bound::Hi(e2, a2, b2, d2), Bound::Lo(e1, a1, b1, d1)) => {
                        ((e1, a1, b1, d1), (e2, a2, b2, d2))
                    }
                    _ => return None,
                };
                // Rows `[a·e + b1, a·e + b2)` of `d` cells: `d·(b2 − b1)` cells
                // from `d·(a·e + b1)`.
                let rows = u64::try_from(b2.checked_sub(b1)?).ok().filter(|&l| l > 0)?;
                let len = rows.checked_mul(d1)?;
                (d1 == d2 && a1 == a2 && (e1 == e2 || a1 == 0)).then_some((
                    Interval {
                        d: d1,
                        a: a1,
                        e: e1.or(e2),
                        b: b1,
                        step: 1,
                        len,
                        w: 1,
                    },
                    true,
                ))
            }
            _ => None,
        }
    }

    /// One comparison of `t` against an affine bound: `t ≥ a·e + b` as
    /// `Lo(e, a, b, 1)`, `t < a·e + b` as `Hi(e, a, b, 1)` — or of the row
    /// `t ÷ d` shifted by an affine term (`t ÷ d − (a'·e + b') ≥ 0`, a chunk
    /// of rows written at a captured position), the shift moved to the bound.
    pub(super) fn t_bound(
        &self,
        cmp: ObjectId,
        input: ObjectId,
        captures: u32,
        site: ObjectId,
    ) -> Option<Bound> {
        let c = self.morphisms.get(*self.in_edges(cmp).first()?)?;
        let (x, y) = (
            self.pair_slot_source(c.source, 0)?,
            self.pair_slot_source(c.source, 1)?,
        );
        let mut e = None;
        let rows = |o: ObjectId, e: &mut Option<u32>| self.t_rows(o, input, captures, site, e, 0);
        // Normalize to `t ÷ d + shift op bound`.
        let (op, (d, sa, sb), bound) = if let Some(r) = rows(x, &mut e) {
            (c.op, r, y)
        } else if let Some(r) = rows(y, &mut e) {
            let flipped = match c.op {
                Operation::Lt => Operation::Gt,
                Operation::Gt => Operation::Lt,
                Operation::Le => Operation::Ge,
                Operation::Ge => Operation::Le,
                _ => return None,
            };
            (flipped, r, x)
        } else {
            return None;
        };
        let (ba, bb) = self.capture_affine(bound, input, captures, site, &mut e, 0)?;
        let (a, b) = (ba.checked_sub(sa)?, bb.checked_sub(sb)?);
        match op {
            Operation::Ge => Some(Bound::Lo(e, a, b, d)),
            Operation::Gt => Some(Bound::Lo(e, a, b.checked_add(1)?, d)),
            Operation::Lt => Some(Bound::Hi(e, a, b, d)),
            Operation::Le => Some(Bound::Hi(e, a, b.checked_add(1)?, d)),
            _ => None,
        }
    }

    /// `o` as `t ÷ d + a·e + b` (`(d, a, b)`): the map element `t` (`d = 1`)
    /// or `t ÷ d` (`d` a positive literal), plus or minus a term affine in one
    /// captured int scalar ([`Self::capture_affine`]). Depth-guarded (J1).
    pub(super) fn t_rows(
        &self,
        o: ObjectId,
        input: ObjectId,
        captures: u32,
        site: ObjectId,
        e: &mut Option<u32>,
        depth: u32,
    ) -> Option<(u64, i64, i64)> {
        if depth > 16 {
            return None;
        }
        if self.tile_input_proj_index(o, input) == Some(captures) {
            return Some((1, 0, 0));
        }
        let mm = self.morphisms.get(*self.in_edges(o).first()?)?;
        let x = self.pair_slot_source(mm.source, 0)?;
        let y = self.pair_slot_source(mm.source, 1)?;
        match mm.op {
            Operation::Div => {
                let d = self.tile_literal_u64(y).filter(|&k| k > 0)?;
                (self.tile_input_proj_index(x, input) == Some(captures)).then_some((d, 0, 0))
            }
            Operation::Add | Operation::Sub => {
                if let Some((d, ra, rb)) = self.t_rows(x, input, captures, site, e, depth + 1) {
                    let (ya, yb) = self.capture_affine(y, input, captures, site, e, depth + 1)?;
                    return Some(if mm.op == Operation::Add {
                        (d, ra.checked_add(ya)?, rb.checked_add(yb)?)
                    } else {
                        (d, ra.checked_sub(ya)?, rb.checked_sub(yb)?)
                    });
                }
                if mm.op != Operation::Add {
                    return None;
                }
                let (d, ra, rb) = self.t_rows(y, input, captures, site, e, depth + 1)?;
                let (xa, xb) = self.capture_affine(x, input, captures, site, e, depth + 1)?;
                Some((d, ra.checked_add(xa)?, rb.checked_add(xb)?))
            }
            _ => None,
        }
    }

    /// `o` as `a·e + b` over one captured int scalar `e` (`Proj{k}` of the
    /// body input, `k < captures`; the capture slot is recorded in `e`) and
    /// literals: `Add`, `Sub`, `Mul` by a literal. Depth-guarded (J1).
    pub(super) fn capture_affine(
        &self,
        o: ObjectId,
        input: ObjectId,
        captures: u32,
        site: ObjectId,
        e: &mut Option<u32>,
        depth: u32,
    ) -> Option<(i64, i64)> {
        if depth > 16 {
            return None;
        }
        let obj = self.objects.get(o)?;
        if obj.kind == ObjectKind::Constant {
            return match obj.value {
                Some(Value::I32(v)) => Some((0, v.into())),
                _ => None,
            };
        }
        if let Some(k) = self.tile_input_proj_index(o, input) {
            // placement `chain`: a capture the site passes as a constant (an
            // inlined layer's index) is that constant at this site.
            if k < captures
                && let Some(c) = self
                    .pair_slot_source(site, k)
                    .and_then(|c| self.objects.get(c))
                && c.kind == ObjectKind::Constant
                && let Some(Value::I32(v)) = c.value
            {
                return Some((0, v.into()));
            }
            return (k < captures && obj.ty == Ty::i32() && *e.get_or_insert(k) == k)
                .then_some((1, 0));
        }
        let mm = self.morphisms.get(*self.in_edges(o).first()?)?;
        if !matches!(mm.op, Operation::Add | Operation::Sub | Operation::Mul) {
            return None;
        }
        let x = self.pair_slot_source(mm.source, 0)?;
        let y = self.pair_slot_source(mm.source, 1)?;
        let (xa, xb) = self.capture_affine(x, input, captures, site, e, depth + 1)?;
        let (ya, yb) = self.capture_affine(y, input, captures, site, e, depth + 1)?;
        match mm.op {
            Operation::Add => Some((xa.checked_add(ya)?, xb.checked_add(yb)?)),
            Operation::Sub => Some((xa.checked_sub(ya)?, xb.checked_sub(yb)?)),
            _ if xa == 0 => Some((ya.checked_mul(xb)?, yb.checked_mul(xb)?)),
            _ if ya == 0 => Some((xa.checked_mul(yb)?, xb.checked_mul(yb)?)),
            _ => None,
        }
    }

    /// Whether a `Map` body may be **recomputed at a consumer's read site**.
    ///
    /// All three conjuncts are load-bearing, and each prevents a distinct way
    /// the oracle could be contradicted:
    ///
    /// - **trap-free** ([`Self::tile_trap_free`], the predicate the tiled path
    ///   already trusts): a body that can trap must fire at the producer's index
    ///   order, not at whatever order a consumer happens to read in.
    /// - **loop-free**: a body containing a loop could turn a `Trapped` outcome
    ///   into `Diverged` — a different observable class, not a slower path.
    /// - **effect-free**: `tile_trap_free`'s catch-all arm answers `true` for
    ///   `Print`/`TimeMs`. Its existing callers are covered by token threading;
    ///   this one is not, so the check is made here rather than inherited.
    ///
    /// A nested `Map`/`Fold` whose own body is classifiable is too (a fixed
    /// array is a finite trip; generic-fusion natural forms: an exponent as a
    /// fold over a range): it can neither trap nor diverge nor print.
    /// Depth-guarded (J1).
    pub(super) fn body_is_classifiable(&self, body: FuncId) -> bool {
        self.classifiable_at(body, 0)
    }

    pub(super) fn classifiable_at(&self, body: FuncId, depth: u32) -> bool {
        let Some(def) = self.func(body) else {
            return false;
        };
        if depth > 8 {
            return false;
        }
        let effectful = def.morphisms.iter().any(|m| {
            matches!(
                self.morphisms.get(*m).map(|x| x.op),
                Some(Operation::Print { .. } | Operation::TimeMs | Operation::Read)
            )
        });
        !effectful
            && self.loop_structure(body).is_empty()
            && self.trap_free_at(body, &self.bounds_proof(body), None, depth)
    }

    /// The law of component `slot` of `product`, cutting to a `Load` addressed
    /// as `(product, slot)` — the form the emitters' pointer helpers take.
    pub(super) fn elem_component(
        &self,
        product: ObjectId,
        slot: u32,
        in_loop: &SecondaryMap<ObjectId, ()>,
        depth: u32,
    ) -> ElemSrc {
        let cut = ElemSrc::Load {
            source: product,
            slot: Some(slot),
        };
        match self.pair_slot_source(product, slot) {
            Some(component) => self.elem_law(component, in_loop, depth).unwrap_or(cut),
            None => cut,
        }
    }

    pub(super) fn tile_literal_u64(&self, object: ObjectId) -> Option<u64> {
        let object = self.objects.get(object)?;
        if object.kind != ObjectKind::Constant {
            return None;
        }
        match object.value.as_ref()? {
            Value::I32(v) => u64::try_from(*v).ok(),
            Value::I64(v) => u64::try_from(*v).ok(),
            Value::U8(v) => Some((*v).into()),
            _ => None,
        }
    }

    /// Trap-freedom AND emission-completeness for a tiled body: the micro-kernel
    /// emits only the recognized chain, so anything it would skip must be
    /// provably unobservable — pure, trap-free, and with no nested body it
    /// cannot see into. `Call` and any Map/Fold other than the recognized fold
    /// (`allow_fold`) are rejected unless their own body is classifiable
    /// (pure, trap-free, loop-free — then skipping it is unobservable); any
    /// other subgraph could trap while the tiled form skips it (an R1
    /// divergence, not just a missed proof). `Call` is rejected outright.
    pub(super) fn tile_trap_free(
        &self,
        f: FuncId,
        bounds: &BoundsProof,
        allow_fold: Option<MorphismId>,
    ) -> bool {
        self.trap_free_at(f, bounds, allow_fold, 0)
    }

    pub(super) fn trap_free_at(
        &self,
        f: FuncId,
        bounds: &BoundsProof,
        allow_fold: Option<MorphismId>,
        depth: u32,
    ) -> bool {
        self.func(f).is_some_and(|fd| {
            fd.morphisms
                .iter()
                .all(|&m| self.morph_trap_free(m, bounds, allow_fold, depth))
        })
    }

    /// One morphism of [`Self::trap_free_at`]: it cannot trap (a nested
    /// body only when it is classifiable — pure, trap-free, loop-free).
    pub(super) fn morph_trap_free(
        &self,
        m: MorphismId,
        bounds: &BoundsProof,
        allow_fold: Option<MorphismId>,
        depth: u32,
    ) -> bool {
        let Some(morph) = self.morphisms.get(m) else {
            return false;
        };
        match morph.op {
            Operation::Div | Operation::Mod => {
                self.objects.get(morph.target).is_some_and(|target| {
                    matches!(target.ty, Ty::Float { .. })
                        || self
                            .pair_slot_source(morph.source, 1)
                            .and_then(|o| self.objects.get(o))
                            .is_some_and(safe_integer_divisor)
                })
            }
            Operation::Index => bounds.proven(m),
            Operation::Update => false,
            Operation::Call { .. } => false,
            Operation::Fold { body, .. } | Operation::Map { body, .. } => {
                allow_fold == Some(m) || self.classifiable_at(body, depth + 1)
            }
            _ => true,
        }
    }
}
