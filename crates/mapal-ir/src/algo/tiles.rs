//! Tile-site recognition: sum-of-products sites, their epilogues, strides and affine reads.

use super::*;

impl CategoryIr {
    pub(super) fn tile_site(&self, site: MorphismId, form: TileForm) -> Option<TileSite> {
        let map = self.morphisms.get(site)?;
        let Operation::Map {
            body,
            captures: map_captures,
        } = map.op
        else {
            return None;
        };
        if map_captures == 0 {
            return None;
        }

        let mapped = self.pair_slot_source(map.source, map_captures)?;
        let mapped_size = self.tile_iota_size(mapped)?;
        let Ty::Array {
            size: target_size, ..
        } = &self.objects.get(map.target)?.ty
        else {
            return None;
        };
        if *target_size != mapped_size {
            return None;
        }

        let body_def = self.func(body)?;
        let mut fold_id = None;
        for &m in &body_def.morphisms {
            let morph = self.morphisms.get(m)?;
            if matches!(morph.op, Operation::Fold { .. }) {
                if fold_id.replace(m).is_some() {
                    return None;
                }
            }
        }
        let fold_id = fold_id?;
        let fold = self.morphisms.get(fold_id)?;
        let post = if fold.target != body_def.output {
            Some(self.tile_post(map.source, body, fold_id, map_captures)?)
        } else {
            None
        };
        let with_post = |mut site: TileSite| -> Option<TileSite> {
            if let Some(p) = &post {
                // The site folds every cell: each read in bounds over the
                // whole iteration space, whatever arm the cell takes.
                let fits = |r: &TileRead| -> Option<bool> {
                    let arr = self.pair_slot_source(map.source, r.slot)?;
                    let Ty::Array { size, .. } = self.objects.get(arr)?.ty else {
                        return None;
                    };
                    let top = |c: u64, n: u64| c.checked_mul(n.saturating_sub(1));
                    let kpart = match r.ksplit {
                        Some(ks) => {
                            top(ks.cq, site.k / ks.div)?.checked_add(top(ks.cr, ks.div)?)?
                        }
                        None => top(r.ck, site.k)?,
                    };
                    let max = r
                        .base
                        .checked_add(top(r.ci, site.rows)?)?
                        .checked_add(top(r.clane, site.c)?)?
                        .checked_add(kpart)?;
                    Some(max < size)
                };
                if site.rows == 0 || site.c == 0 || site.k == 0 {
                    return None;
                }
                if fits(&site.a) != Some(true) || fits(&site.b) != Some(true) {
                    return None;
                }
                site.post = Some(p.clone());
            }
            Some(site)
        };
        let Operation::Fold {
            body: fold_body,
            captures: fold_captures,
        } = fold.op
        else {
            return None;
        };
        let depth = self
            .tile_iota_size(self.pair_slot_source(fold.source, fold_captures.checked_add(1)?)?)?;
        let seed_object = self
            .objects
            .get(self.through_pairs(self.pair_slot_source(fold.source, fold_captures)?))?;
        if seed_object.kind != ObjectKind::Constant {
            return None;
        }
        let seed = seed_object.value.clone()?;
        let body_bounds = self.bounds_proof(body);
        let fold_bounds = self.bounds_proof(fold_body);
        if !self.tile_trap_free(body, &body_bounds, Some(fold_id))
            || !self.tile_trap_free(fold_body, &fold_bounds, None)
        {
            return None;
        }

        let mut has_split = false;
        for &div_id in &body_def.morphisms {
            let Some((i, c)) =
                self.tile_split(div_id, body_def.input, map_captures, Operation::Div)
            else {
                continue;
            };
            for &mod_id in &body_def.morphisms {
                let Some((j, mod_c)) =
                    self.tile_split(mod_id, body_def.input, map_captures, Operation::Mod)
                else {
                    continue;
                };
                if mod_c != c {
                    continue;
                }
                has_split = true;
                if mapped_size % c != 0 {
                    continue;
                }
                let rows = mapped_size / c;
                if rows.checked_mul(c) != Some(mapped_size) {
                    continue;
                }
                let Some(site) = self.tile_fold_shape(
                    map.target,
                    map_captures,
                    body_def.input,
                    fold,
                    Some(i),
                    j,
                    rows,
                    c,
                    depth,
                    seed.clone(),
                    form,
                ) else {
                    continue;
                };
                return with_post(site);
            }
        }
        if has_split {
            return None;
        }
        if mapped_size == 0 {
            return None;
        }

        let lane = body_def.morphisms.iter().find_map(|m| {
            let morph = self.morphisms.get(*m)?;
            (self.tile_input_proj_index(morph.target, body_def.input) == Some(map_captures))
                .then_some(morph.target)
        })?;
        self.tile_fold_shape(
            map.target,
            map_captures,
            body_def.input,
            fold,
            None,
            lane,
            1,
            mapped_size,
            depth,
            seed,
            form,
        )
        .and_then(with_post)
    }

    /// [`TilePost`] of a map over `source` with body `body` whose one fold
    /// `fold` does not produce the body's output: every other morphism is
    /// pure cell work (no `Map`, loop, call or effect; `classifiable`), and
    /// the slots it reads are recorded. The fold's own operands reach it
    /// through the `Pair`s that assemble its source; any other use of a
    /// captured slot is the epilogue's.
    pub(super) fn tile_post(
        &self,
        source: ObjectId,
        body: FuncId,
        fold: MorphismId,
        captures: u32,
    ) -> Option<TilePost> {
        let bd = self.func(body)?;
        if !self.body_is_classifiable(body) {
            return None;
        }
        for &m in &bd.morphisms {
            match self.morphisms.get(m)?.op {
                Operation::Fold { .. } if m == fold => {}
                Operation::Map { .. }
                | Operation::Fold { .. }
                | Operation::Call(_)
                | Operation::LoopEnter
                | Operation::LoopBack
                | Operation::LoopExit => return None,
                _ => {}
            }
        }
        // The fold's operands reach it through the `Pair`s into its source.
        let fsrc = self.morphisms.get(fold)?.source;
        let mut reads = Vec::new();
        let mut uniform = Vec::new();
        let mut after = Vec::new();
        for &p in self.out_edges(bd.input) {
            let pm = self.morphisms.get(p)?;
            let Operation::Proj { index: slot } = pm.op else {
                continue;
            };
            if slot >= captures {
                continue;
            }
            let arr = self.pair_slot_source(source, slot)?;
            match self.objects.get(arr)?.ty {
                Ty::Array { .. } => {
                    if !self.feeds_only(pm.target, fsrc, 0) && !reads.contains(&slot) {
                        reads.push(slot);
                    }
                }
                Ty::Int { .. } | Ty::Float { .. } => {
                    if !uniform.contains(&slot) {
                        uniform.push(slot);
                    }
                    if !self.feeds_only(pm.target, fsrc, 0) && !after.contains(&slot) {
                        after.push(slot);
                    }
                }
                _ => return None,
            }
        }
        after.sort_unstable();
        reads.sort_unstable();
        uniform.sort_unstable();
        let guarded = self
            .guard_structure(body)
            .iter()
            .any(|g| g.on_true.own.contains(&fold) || g.on_false.own.contains(&fold));
        Some(TilePost {
            fold,
            reads,
            uniform,
            guarded,
            after,
            round: self.body_rounds(body),
        })
    }

    /// bf16-class `round-post`: the narrow format body `body`'s result is
    /// rounded to by its last step (`Narrow` from the f32 value) — a map
    /// storing its cells narrow ([`TilePost::round`]).
    pub(super) fn body_rounds(&self, body: FuncId) -> Option<crate::ty::NarrowFloat> {
        let r = self.func(body)?.output;
        let Ty::Narrow(n) = self.objects.get(r)?.ty else {
            return None;
        };
        let [m] = self.in_edges(r) else {
            return None;
        };
        let m = self.morphisms.get(*m)?;
        (m.op == Operation::Narrow && self.objects.get(m.source)?.ty == Ty::f32()).then_some(n)
    }

    /// The element type a tile site over map `map_target` writes, as its
    /// products see it: the array's own, or the f32 an epilogue's last step
    /// rounds to the array's narrow format ([`TilePost::round`]).
    pub(crate) fn tile_out_elem(&self, map_target: ObjectId) -> Option<Ty> {
        let Ty::Array { elem, .. } = &self.objects.get(map_target)?.ty else {
            return None;
        };
        if let Ty::Narrow(n) = **elem
            && let [m] = self.in_edges(map_target)
            && let Operation::Map { body, .. } = self.morphisms.get(*m)?.op
            && self.body_rounds(body) == Some(n)
        {
            return Some(Ty::f32());
        }
        Some((**elem).clone())
    }

    pub(super) fn tile_split(
        &self,
        m: MorphismId,
        body_input: ObjectId,
        element_slot: u32,
        op: Operation,
    ) -> Option<(ObjectId, u64)> {
        let morph = self.morphisms.get(m)?;
        if morph.op != op {
            return None;
        }
        let element = self.pair_slot_source(morph.source, 0)?;
        if self.tile_input_proj_index(element, body_input) != Some(element_slot) {
            return None;
        }
        let c = self.tile_literal_u64(self.pair_slot_source(morph.source, 1)?)?;
        (c != 0).then_some((morph.target, c))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn tile_fold_shape(
        &self,
        map_target: ObjectId,
        map_captures: u32,
        body_input: ObjectId,
        fold: &crate::graph::Morphism,
        i: Option<ObjectId>,
        lane: ObjectId,
        rows: u64,
        c: u64,
        k: u64,
        seed: Value,
        form: TileForm,
    ) -> Option<TileSite> {
        let strided = form.strided;
        let Operation::Fold {
            body,
            captures: fold_captures,
        } = fold.op
        else {
            return None;
        };
        let fold_def = self.func(body)?;
        // Fold-body analog of the map-body split (`tile_site`): bind a
        // `Div`/`Mod` pair on the fold's counted element to the derived axes
        // `(k÷div, k%div)` — conv2d's window taps. Rectangular windows only
        // (`k % div == 0`); a pair that fails this simply stays unbound and
        // the walker's `_ => None` arm refuses the site.
        let element_slot = fold_captures + 1;
        let ksplit_axes = fold_def.morphisms.iter().find_map(|&div_id| {
            let (kq, div) =
                self.tile_split(div_id, fold_def.input, element_slot, Operation::Div)?;
            fold_def.morphisms.iter().find_map(|&mod_id| {
                let (kr, mod_div) =
                    self.tile_split(mod_id, fold_def.input, element_slot, Operation::Mod)?;
                (mod_div == div && k % div == 0).then_some((kq, kr, div))
            })
        });
        let add = self.tile_definer(fold_def.output, Operation::Add)?;
        let add_lhs = self.pair_slot_source(add.source, 0)?;
        let add_rhs = self.pair_slot_source(add.source, 1)?;
        let lhs_acc = self.tile_input_proj_index(add_lhs, fold_def.input) == Some(fold_captures);
        let rhs_acc = self.tile_input_proj_index(add_rhs, fold_def.input) == Some(fold_captures);
        let (acc, product, add_acc_first) = match (lhs_acc, rhs_acc) {
            (true, false) => (add_lhs, add_rhs, true),
            (false, true) => (add_rhs, add_lhs, false),
            _ => return None,
        };

        let mul = self.tile_definer(product, Operation::Mul)?;
        let load0 = self.pair_slot_source(mul.source, 0)?;
        let load1 = self.pair_slot_source(mul.source, 1)?;
        let index_parts = |value: ObjectId| -> Option<(TileRead, Ty)> {
            // ADR-0043 (`form.narrow`): the product reads the exact widening
            // of a narrow array's element — the read is the element's.
            let (value, narrow) = match self.tile_definer(value, Operation::Widen) {
                Some(w) if form.narrow => match self.objects.get(w.source)?.ty {
                    Ty::Narrow(n) => (w.source, Some((n, self.objects.get(value)?.ty.clone()))),
                    _ => return None,
                },
                _ => (value, None),
            };
            let index = self.tile_definer(value, Operation::Index)?;
            let array = self.pair_slot_source(index.source, 0)?;
            let address = self.pair_slot_source(index.source, 1)?;
            let body_value =
                self.tile_fold_capture(array, fold.source, fold_def.input, fold_captures)?;
            let slot = self.tile_input_proj_index(body_value, body_input)?;
            if slot >= map_captures {
                return None;
            }
            let Ty::Array { elem, .. } = &self.objects.get(array)?.ty else {
                return None;
            };
            if self.objects.get(value)?.ty != **elem {
                return None;
            }
            let aff = self.tile_affine(
                address,
                fold.source,
                fold_def.input,
                fold_captures,
                i,
                lane,
                ksplit_axes.map(|(kq, kr, _)| (kq, kr)),
            )?;
            let ksplit = match ksplit_axes {
                Some((_, _, div)) if aff.cq != 0 || aff.cr != 0 => {
                    // A read is affine in raw `k` XOR in the derived axes —
                    // never mixed.
                    if aff.ck != 0 {
                        return None;
                    }
                    Some(TileKSplit {
                        div,
                        cq: aff.cq,
                        cr: aff.cr,
                    })
                }
                _ => None,
            };
            Some((
                TileRead {
                    slot,
                    base: aff.base,
                    ci: aff.ci,
                    ck: aff.ck,
                    clane: aff.clane,
                    ksplit,
                    narrow: narrow.as_ref().map(|(n, _)| *n),
                },
                // the product's operand type: the widening's target
                narrow.map_or_else(|| (**elem).clone(), |(_, t)| t),
            ))
        };
        let (read0, elem0) = index_parts(load0)?;
        let (read1, elem1) = index_parts(load1)?;

        if elem0 != elem1 {
            return None;
        }
        let (a, b, mul_a_first) = match (read0.clane, read1.clane) {
            (0, 1) => (read0, read1, true),
            (1, 0) => (read1, read0, false),
            // `tile_plan_strided`: `b` walks the lanes at stride `l` (`q·Kᵀ`, one row or 2-D).
            (0, l) if strided && l > 1 => (read0, read1, true),
            (l, 0) if strided && l > 1 => (read1, read0, false),
            _ => return None,
        };

        let acc_ty = self.objects.get(acc)?.ty.clone();
        let map_elem = self.tile_out_elem(map_target)?;
        if elem0 != acc_ty || elem0 != map_elem || seed.ty() != elem0 || !elem0.is_numeric() {
            return None;
        }
        Some(TileSite {
            rows,
            c,
            k,
            a,
            b,
            seed,
            elem: elem0,
            mul_a_first,
            add_acc_first,
            radix: None,
            tf32: self.precision(mul.id).admits_tf32(),
            half: self.precision(mul.id).half(),
            post: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn tile_affine(
        &self,
        object: ObjectId,
        fold_source: ObjectId,
        fold_input: ObjectId,
        fold_captures: u32,
        i: Option<ObjectId>,
        lane: ObjectId,
        ksplit_axes: Option<(ObjectId, ObjectId)>,
    ) -> Option<TileAffine> {
        if let Some(captured) =
            self.tile_fold_capture(object, fold_source, fold_input, fold_captures)
        {
            if i == Some(captured) {
                return Some(TileAffine {
                    ci: 1,
                    ..TileAffine::default()
                });
            }
            if captured == lane {
                return Some(TileAffine {
                    clane: 1,
                    ..TileAffine::default()
                });
            }
        }
        if let Some((kq, kr)) = ksplit_axes {
            if object == kq {
                return Some(TileAffine {
                    cq: 1,
                    ..TileAffine::default()
                });
            }
            if object == kr {
                return Some(TileAffine {
                    cr: 1,
                    ..TileAffine::default()
                });
            }
        }
        if self.tile_input_proj_index(object, fold_input) == fold_captures.checked_add(1) {
            return Some(TileAffine {
                ck: 1,
                ..TileAffine::default()
            });
        }
        if let Some(base) = self.tile_literal_u64(object) {
            return Some(TileAffine {
                base,
                ..TileAffine::default()
            });
        }

        let [m] = self.in_edges(object) else {
            return None;
        };
        let morph = self.morphisms.get(*m)?;
        let lhs = self.pair_slot_source(morph.source, 0)?;
        let rhs = self.pair_slot_source(morph.source, 1)?;
        match morph.op {
            Operation::Add => {
                let x = self.tile_affine(
                    lhs,
                    fold_source,
                    fold_input,
                    fold_captures,
                    i,
                    lane,
                    ksplit_axes,
                )?;
                let y = self.tile_affine(
                    rhs,
                    fold_source,
                    fold_input,
                    fold_captures,
                    i,
                    lane,
                    ksplit_axes,
                )?;
                x.add(y)
            }
            Operation::Mul => {
                let (scale, value) = if let Some(scale) = self.tile_literal_u64(lhs) {
                    (scale, rhs)
                } else {
                    (self.tile_literal_u64(rhs)?, lhs)
                };
                let value = self.tile_affine(
                    value,
                    fold_source,
                    fold_input,
                    fold_captures,
                    i,
                    lane,
                    ksplit_axes,
                )?;
                value.scale(scale)
            }
            _ => None,
        }
    }

    pub(crate) fn tile_definer(
        &self,
        object: ObjectId,
        op: Operation,
    ) -> Option<&crate::graph::Morphism> {
        let [m] = self.in_edges(object) else {
            return None;
        };
        let morph = self.morphisms.get(*m)?;
        (morph.op == op).then_some(morph)
    }

    pub(crate) fn tile_input_proj_index(&self, object: ObjectId, input: ObjectId) -> Option<u32> {
        let [m] = self.in_edges(object) else {
            return None;
        };
        let proj = self.morphisms.get(*m)?;
        let Operation::Proj { index } = proj.op else {
            return None;
        };
        (proj.source == input).then_some(index)
    }

    pub(super) fn tile_fold_capture(
        &self,
        object: ObjectId,
        fold_source: ObjectId,
        fold_input: ObjectId,
        captures: u32,
    ) -> Option<ObjectId> {
        let slot = self.tile_input_proj_index(object, fold_input)?;
        (slot < captures)
            .then(|| self.pair_slot_source(fold_source, slot))
            .flatten()
    }

    pub(super) fn tile_iota_size(&self, mut array: ObjectId) -> Option<u64> {
        for _ in 0..16 {
            let next = self
                .proj_of_pair(array)
                .or_else(|| self.capture_proj_of_input(array));
            let Some(next) = next else {
                break;
            };
            if next == array {
                return None;
            }
            array = next;
        }
        let Ty::Array { size, .. } = &self.objects.get(array)?.ty else {
            return None;
        };
        // Consumes the STRUCTURE fact, not the op tag. This used to read
        // `op == Operation::Iota`, which asks what the node is TAGGED rather
        // than what the array's element IS — and `tile_site` calls this twice
        // per site (the outer mapped array and the inner fold's trip count),
        // with a silent fallback to the scalar emitter when it answers `None`.
        // A tag check goes stale the moment anything else can produce an index
        // law; a fact check follows. Behavior today is identical — `Iota` is
        // still the only producer of `ElemSrc::Index` — and the cost of getting
        // this wrong is pinned in `tests/tile_sites_pin.rs` (4.0x on matmul
        // 1024, docs/performance/matmul/history/bit-exact-simd-tiles.md:46-48).
        matches!(self.elem_law_local(array), Some(ElemSrc::Index)).then_some(*size)
    }

    /// The one-step element law of `array` from its producer alone: the local
    /// half of [`elem_plan`]'s recursion, with no loop-residence or depth guard
    /// (the caller owns those). Returns `None` for the cut.
    pub(super) fn elem_law_local(&self, array: ObjectId) -> Option<ElemSrc> {
        let [definer] = self.in_edges(array) else {
            return None;
        };
        match self.morphisms.get(*definer)?.op {
            Operation::Iota => Some(ElemSrc::Index),
            Operation::Fill => Some(ElemSrc::Broadcast {
                source: self.morphisms.get(*definer)?.source,
                slot: 0,
            }),
            _ => None,
        }
    }
}
