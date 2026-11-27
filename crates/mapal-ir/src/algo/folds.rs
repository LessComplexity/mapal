//! Fold facts: regroupable fold steps, whole folds, row-fold and move sites.

use super::*;

impl CategoryIr {
    /// **A fold step the order may regroup** (recurrence-precision item 2):
    /// `fold`'s body is `acc ⊕ g(caps, k)` — `⊕` an `Add` or `Mul` with `acc`
    /// on either side, or a running max/min (`cmp(acc, e) ? · : ·` over
    /// exactly `{acc, e}`) — and `acc` feeds nothing but that step, so `g`
    /// never reads it. Where the order is free (ADR-0040) such a fold may run
    /// as partial folds from the step's identity combined by `⊕`.
    pub fn fold_tree_step(&self, fold: MorphismId) -> Option<RowFoldOp> {
        let Operation::Fold { body, captures } = self.morphisms.get(fold)?.op else {
            return None;
        };
        let fd = self.func(body)?;
        let accs: Vec<ObjectId> = self
            .out_edges(fd.input)
            .iter()
            .filter(|&&p| self.morphisms[p].op == (Operation::Proj { index: captures }))
            .map(|&p| self.morphisms[p].target)
            .collect();
        let &[acc] = accs.as_slice() else { return None };
        let [step] = self.in_edges(fd.output) else {
            return None;
        };
        let st = self.morphisms.get(*step)?;
        let (op, _) = self.fold_step(st, &|o| o == acc)?;
        if matches!(
            op,
            RowFoldOp::Arith {
                op: Operation::Sub,
                ..
            }
        ) {
            return None;
        }
        // Every use of `acc` is a slot of a product read only by the step
        // (and a running max/min's compare).
        let mut into = vec![st.source];
        if st.op == Operation::Phi {
            into.push(
                self.morphisms
                    .get(
                        *self
                            .in_edges(self.pair_slot_source(st.source, 2)?)
                            .first()?,
                    )?
                    .source,
            );
        }
        self.out_edges(acc)
            .iter()
            .all(|&e| {
                let pm = &self.morphisms[e];
                matches!(pm.op, Operation::Pair { .. }) && into.contains(&pm.target)
            })
            .then_some(op)
    }

    /// A fold body's step `acc ⊕ e` and its `e`: `⊕` an `Add`, `Sub` or `Mul`
    /// with `acc` on one side, or a running max/min (`cmp(acc, e) ? · : ·`
    /// over exactly `{acc, e}`). One classifier for [`Self::fold_core`] and
    /// [`Self::fold_tree_step`].
    pub(super) fn fold_step(
        &self,
        step: &crate::graph::Morphism,
        acc: &dyn Fn(ObjectId) -> bool,
    ) -> Option<(RowFoldOp, ObjectId)> {
        let slot = |k| self.pair_slot_source(step.source, k);
        match step.op {
            Operation::Add | Operation::Sub | Operation::Mul => {
                let (x, y) = (slot(0)?, slot(1)?);
                match (acc(x), acc(y)) {
                    (true, false) => Some((
                        RowFoldOp::Arith {
                            op: step.op,
                            acc_first: true,
                        },
                        y,
                    )),
                    (false, true) => Some((
                        RowFoldOp::Arith {
                            op: step.op,
                            acc_first: false,
                        },
                        x,
                    )),
                    _ => None,
                }
            }
            Operation::Phi => {
                let (p, q) = (slot(0)?, slot(1)?);
                let cmp = self.morphisms.get(*self.in_edges(slot(2)?).first()?)?;
                if !matches!(
                    cmp.op,
                    Operation::Lt | Operation::Gt | Operation::Le | Operation::Ge
                ) {
                    return None;
                }
                let x = self.pair_slot_source(cmp.source, 0)?;
                let y = self.pair_slot_source(cmp.source, 1)?;
                let e = if acc(x) { y } else { x };
                // Exactly `{acc, e}` on both sides: anything else is not a
                // running max/min of the read.
                let one_each = |u: ObjectId, v: ObjectId| {
                    acc(u) != acc(v) && [u, v].iter().all(|&o| acc(o) || o == e)
                };
                (one_each(x, y) && one_each(p, q)).then_some((
                    RowFoldOp::Select {
                        cmp: cmp.op,
                        acc_lhs: acc(x),
                        acc_if_true: acc(p),
                    },
                    e,
                ))
            }
            _ => None,
        }
    }

    /// **Whole folds** (recurrence-precision): a `Fold` of `f` itself (no
    /// enclosing map) over a counted range whose step reads one captured
    /// array at an affine address — a [`RowFoldSite`] of one row (`rows = 1`,
    /// `ci = 0`): `out = fold_k (seed, ⊕) a[base + ck·k]`. Its slot is the
    /// fold source's capture slot. A softmax's `max` and `sum` in a loop trip.
    pub fn whole_fold(&self, f: FuncId) -> SecondaryMap<MorphismId, RowFoldSite> {
        self.fact(
            f,
            |s| &s.whole_fold,
            || {
                let mut sites = SecondaryMap::new();
                for &m in self.func(f).map(|d| d.morphisms.as_slice()).unwrap_or(&[]) {
                    if let Some(site) = self.whole_fold_site(m) {
                        sites.insert(m, site);
                    }
                }
                sites
            },
        )
    }

    pub(super) fn whole_fold_site(&self, m: MorphismId) -> Option<RowFoldSite> {
        let fold = self.morphisms.get(m)?;
        if !matches!(fold.op, Operation::Fold { .. }) {
            return None;
        }
        let elem = self.objects.get(fold.target)?.ty.clone();
        if !elem.is_numeric() {
            return None;
        }
        let FoldCore {
            fbody,
            fcap,
            k,
            seed,
            op,
            array,
            address,
        } = self.fold_core(m)?;
        let fdef = self.func(fbody)?;
        let slot = self.tile_input_proj_index(array, fdef.input)?;
        let source = self.pair_slot_source(fold.source, slot)?;
        if slot >= fcap {
            return None;
        }
        let Ty::Array {
            elem: read_elem, ..
        } = &self.objects.get(source)?.ty
        else {
            return None;
        };
        if **read_elem != elem || self.objects.get(fdef.output)?.ty != elem || seed.ty() != elem {
            return None;
        }
        // No row: the fold's target stands in for the index, never a capture.
        let aff = self.tile_affine(
            address,
            fold.source,
            fdef.input,
            fcap,
            None,
            fold.target,
            None,
        )?;
        if aff.ci != 0 || aff.clane != 0 || aff.cq != 0 || aff.cr != 0 || k == 0 {
            return None;
        }
        if !self.tile_trap_free(fbody, &self.bounds_proof(fbody), None) {
            return None;
        }
        Some(RowFoldSite {
            rows: 1,
            k,
            a: TileRead {
                slot,
                base: aff.base,
                ci: 0,
                ck: aff.ck,
                clane: 0,
                ksplit: None,
                narrow: None,
            },
            w: 1,
            op,
            seed,
            elem,
        })
    }

    pub(super) fn row_fold_compute(&self, f: FuncId) -> SecondaryMap<MorphismId, RowFoldSite> {
        let mut sites = SecondaryMap::new();
        if let Some(fd) = self.func(f) {
            for &m in &fd.morphisms {
                if let Some(site) = self.row_fold_site(m) {
                    sites.insert(m, site);
                }
            }
        }
        sites
    }

    pub(super) fn row_fold_site(&self, site: MorphismId) -> Option<RowFoldSite> {
        let map = self.morphisms.get(site)?;
        let Operation::Map { body, captures } = map.op else {
            return None;
        };
        if captures == 0 {
            return None;
        }
        let rows = self.tile_iota_size(self.pair_slot_source(map.source, captures)?)?;
        let Ty::Array { size, elem } = &self.objects.get(map.target)?.ty else {
            return None;
        };
        if *size != rows || rows == 0 || !elem.is_numeric() {
            return None;
        }
        let elem = (**elem).clone();
        let body_def = self.func(body)?;
        let mut folds = body_def.morphisms.iter().copied().filter(|&m| {
            matches!(
                self.morphisms.get(m).map(|x| x.op),
                Some(Operation::Fold { .. })
            )
        });
        let fold_id = folds.next()?;
        if folds.next().is_some() {
            return None;
        }
        let fold = self.morphisms.get(fold_id)?;
        if fold.target != body_def.output {
            return None;
        }
        let FoldCore {
            fbody,
            fcap,
            k,
            seed,
            op,
            array,
            address,
        } = self.fold_core(fold_id)?;
        let fdef = self.func(fbody)?;
        let body_value = self.tile_fold_capture(array, fold.source, fdef.input, fcap)?;
        let read_slot = self.tile_input_proj_index(body_value, body_def.input)?;
        if read_slot >= captures {
            return None;
        }
        let Ty::Array {
            elem: read_elem, ..
        } = &self.objects.get(array)?.ty
        else {
            return None;
        };
        let acc_ty = &self.objects.get(fdef.output)?.ty;
        if **read_elem != elem || *acc_ty != elem || seed.ty() != elem {
            return None;
        }
        // The map index's axes as the fold captures them: the element itself,
        // or its split `(t÷w, t%w)` (`tile_site`'s split, one level up).
        let split = body_def.morphisms.iter().find_map(|&d| {
            let (q, w) = self.tile_split(d, body_def.input, captures, Operation::Div)?;
            body_def.morphisms.iter().find_map(|&m| {
                let (r, wm) = self.tile_split(m, body_def.input, captures, Operation::Mod)?;
                (wm == w && w > 1 && rows % w == 0).then_some((q, r, w))
            })
        });
        let (i, lane, w) = match split {
            Some((q, r, w)) => (q, r, w),
            None => {
                let t = (0..fcap).find_map(|s| {
                    let o = self.pair_slot_source(fold.source, s)?;
                    (self.tile_input_proj_index(o, body_def.input) == Some(captures)).then_some(o)
                })?;
                (t, t, 1)
            }
        };
        // The fold's own `(k÷div, k%div)`, rectangular windows only.
        let element = fcap + 1;
        let kaxes = fdef.morphisms.iter().find_map(|&d| {
            let (kq, div) = self.tile_split(d, fdef.input, element, Operation::Div)?;
            fdef.morphisms.iter().find_map(|&m| {
                let (kr, md) = self.tile_split(m, fdef.input, element, Operation::Mod)?;
                (md == div && k % div == 0).then_some((kq, kr, div))
            })
        });
        let aff = self.tile_affine(
            address,
            fold.source,
            fdef.input,
            fcap,
            Some(i),
            lane,
            kaxes.map(|(kq, kr, _)| (kq, kr)),
        )?;
        let ksplit = match kaxes {
            Some((_, _, div)) if aff.cq != 0 || aff.cr != 0 => {
                if aff.ck != 0 {
                    return None;
                }
                Some(TileKSplit {
                    div,
                    cq: aff.cq,
                    cr: aff.cr,
                })
            }
            _ if aff.cq != 0 || aff.cr != 0 => return None,
            _ => None,
        };
        if w == 1 && aff.clane != 0 {
            return None;
        }
        if !self.tile_trap_free(body, &self.bounds_proof(body), Some(fold_id))
            || !self.tile_trap_free(fbody, &self.bounds_proof(fbody), None)
        {
            return None;
        }
        Some(RowFoldSite {
            rows,
            k,
            a: TileRead {
                slot: read_slot,
                base: aff.base,
                ci: aff.ci,
                ck: aff.ck,
                clane: aff.clane,
                ksplit,
                narrow: None,
            },
            w,
            op,
            seed,
            elem,
        })
    }

    /// A fold's step and its one read, wherever the fold sits: a row fold's
    /// (inside a map body) or a whole fold's ([`CategoryIr::whole_fold`]).
    pub(super) fn fold_core(&self, fold_id: MorphismId) -> Option<FoldCore> {
        let fold = self.morphisms.get(fold_id)?;
        let Operation::Fold {
            body: fbody,
            captures: fcap,
        } = fold.op
        else {
            return None;
        };
        let k = self.tile_iota_size(self.pair_slot_source(fold.source, fcap.checked_add(1)?)?)?;
        let seed_object = self
            .objects
            .get(self.pair_slot_source(fold.source, fcap)?)?;
        if seed_object.kind != ObjectKind::Constant {
            return None;
        }
        let seed = seed_object.value.clone()?;
        let fdef = self.func(fbody)?;
        // ONE read: a second one is a second array or a second cell per step.
        let reads = fdef
            .morphisms
            .iter()
            .filter(|&&m| {
                self.morphisms
                    .get(m)
                    .is_some_and(|x| x.op == Operation::Index)
            })
            .count();
        if reads != 1 {
            return None;
        }
        let acc = |o: ObjectId| self.tile_input_proj_index(o, fdef.input) == Some(fcap);
        let [step] = self.in_edges(fdef.output) else {
            return None;
        };
        let (op, e) = self.fold_step(self.morphisms.get(*step)?, &acc)?;
        // `e` IS the read — any arithmetic on it is a second step.
        let index = self.tile_definer(e, Operation::Index)?;
        let array = self.pair_slot_source(index.source, 0)?;
        let address = self.pair_slot_source(index.source, 1)?;
        Some(FoldCore {
            fbody,
            fcap,
            k,
            seed,
            op,
            array,
            address,
        })
    }

    pub(super) fn move_site(&self, site: MorphismId) -> Option<MoveSite> {
        let map = self.morphisms.get(site)?;
        let Operation::Map { body, captures } = map.op else {
            return None;
        };
        // captures == 0 is a generator map: no array is read, so no address
        // exists to be affine in anything.
        if captures == 0 {
            return None;
        }
        let mapped = self.pair_slot_source(map.source, captures)?;
        let extent = self.tile_iota_size(mapped)?;
        let Ty::Array {
            size: target_size, ..
        } = &self.objects.get(map.target)?.ty
        else {
            return None;
        };
        if *target_size != extent || extent == 0 {
            return None;
        }

        let body_def = self.func(body)?;
        // THE arm condition: no fold. A body with one is a reduction and belongs
        // to `tile_site`; the two recognizers must never both answer for a site.
        if body_def.morphisms.iter().any(|&m| {
            matches!(
                self.morphisms.get(m).map(|x| x.op),
                Some(Operation::Fold { .. })
            )
        }) {
            return None;
        }
        // `(t÷C, t%C)` — the identical `tile_split` call `tile_site` makes on a
        // map body, one level up.
        let (q, c) = body_def
            .morphisms
            .iter()
            .find_map(|&m| self.tile_split(m, body_def.input, captures, Operation::Div))?;
        let (r, cm) = body_def
            .morphisms
            .iter()
            .find_map(|&m| self.tile_split(m, body_def.input, captures, Operation::Mod))?;
        if cm != c || c < 2 || extent % c != 0 {
            return None;
        }
        let rows = extent / c;
        if rows < 2 {
            return None;
        }

        // The body IS one read of a captured array: nothing else to emit, so the
        // permutation cannot reorder any other observable.
        let index = self.tile_definer(body_def.output, Operation::Index)?;
        let array = self.pair_slot_source(index.source, 0)?;
        let address = self.pair_slot_source(index.source, 1)?;
        if self.tile_input_proj_index(array, body_def.input)? >= captures {
            return None;
        }
        let Ty::Array { elem, size: len } = &self.objects.get(array)?.ty else {
            return None;
        };
        let aff = self.move_affine(address, q, r, 0)?;
        // Last, because it is the only expensive clause: the permutation
        // reorders visits, so a body that can trap could trap at a different
        // element. `tile_trap_free` is the predicate the tiled path already
        // trusts, and for the shapes we run it costs nothing — their `Index` is
        // already `bounds_proof`-proven.
        if !self.tile_trap_free(body, &self.bounds_proof(body), None) {
            return None;
        }
        Some(MoveSite {
            width: c,
            rows,
            cq: aff.cq,
            cr: aff.cr,
            elem: (**elem).clone(),
            len: *len,
        })
    }

    /// The map-body analog of [`Self::tile_affine`]: an address as
    /// `base + cq·(t÷C) + cr·(t%C)`, over Add / Mul-by-literal / literal. The
    /// coefficient fields are [`TileAffine`]'s own `cq`/`cr` — the same two
    /// derived axes it already carries for a fold's `k÷div`/`k%div`.
    pub(super) fn move_affine(
        &self,
        object: ObjectId,
        q: ObjectId,
        r: ObjectId,
        depth: u32,
    ) -> Option<TileAffine> {
        if depth >= 16 {
            return None;
        }
        if object == q {
            return Some(TileAffine {
                cq: 1,
                ..TileAffine::default()
            });
        }
        if object == r {
            return Some(TileAffine {
                cr: 1,
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
            Operation::Add => self
                .move_affine(lhs, q, r, depth + 1)?
                .add(self.move_affine(rhs, q, r, depth + 1)?),
            Operation::Mul => {
                let (scale, value) = match self.tile_literal_u64(lhs) {
                    Some(scale) => (scale, rhs),
                    None => (self.tile_literal_u64(rhs)?, lhs),
                };
                self.move_affine(value, q, r, depth + 1)?.scale(scale)
            }
            _ => None,
        }
    }
}
