//! DevEmit bulk ops: map, zip, enumerate, iota, fill, index, update, fold, literal and output.

use super::*;

impl<'a> DevEmit<'a> {
    /// A per-thread sequential Map loop (§1 inline form): the target local
    /// was declared in the prologue; the body call is trap-checked per
    /// element (the oracle's in-order first trap). ADR-0027: a capturing
    /// map's source is the `(c₁…cₖ, [T; n])` product — the captured
    /// components are per-thread reads of the same buffers/scalars, passed
    /// as the leading fields of the body-input argument.
    pub(super) fn emit_map(
        &mut self,
        source: ObjectId,
        target: ObjectId,
        body: FuncId,
        captures: u32,
    ) {
        let arr_obj = if captures == 0 {
            source
        } else {
            pair_component(self.ir, source, captures)
        };
        let arr_ty = self.obj_ty(arr_obj);
        let (elem, n) = array_parts(&arr_ty);
        let tgt_ty = self.obj_ty(target);
        let (uelem, _) = array_parts(&tgt_ty);
        let iv = self.tmp();
        // An erased source array (Array{Unit}) has no slot: the body
        // argument is omitted, and the per-thread loop still runs — the
        // launch form omits the kernel parameter the same way (F6). (The
        // residual expect is the invariant elem-lowers ⟹ source-lowers ⟹
        // a slot exists.)
        let src = self.slot(arr_obj);
        let elem_arg = if lower_ty(&elem).is_some() {
            Some(elem_expr(
                src.as_deref().expect("map source slot"),
                &iv,
                &elem,
            ))
        } else {
            None
        };
        // ADR-0027: the capture components of the source product, then the
        // element — the `(c₁…cₖ, elem)` body-input shape.
        let (pre, inloop, arg) = if captures == 0 {
            (Vec::new(), Vec::new(), elem_arg)
        } else {
            let pair_ty = self.obj_ty(self.ir.func(body).expect("body resolves").input);
            let mut parts: Vec<Option<String>> = Vec::new();
            for j in 0..captures {
                // WP-B (R-NODUP): read the capture THROUGH the source
                // product's field — a Named product gives `oN.fK` (one
                // reference), never a second copy of an Inline feeder's
                // expression; a dissolved product resolves to the field
                // source's expression, which then counts as its one use.
                parts.push(self.component_expr(source, j).map(|(_, v)| v));
            }
            parts.push(elem_arg);
            let mut varying = vec![false; captures as usize];
            varying.push(true);
            assemble_body_arg(&pair_ty, &parts, &varying)
        };
        for l in pre {
            self.line(1, l);
        }
        self.line(
            1,
            format!("for (unsigned long long {iv} = 0; {iv} < {n}ULL; {iv}++) {{"),
        );
        for l in inloop {
            self.line(2, l);
        }
        let name = self.quals.device_name(self.fnames, body);
        let mut args: Vec<String> = Vec::new();
        if let Some(a) = arg {
            args.push(a);
        }
        // An erased array-typed body result has no buffer (F6): no out
        // param, no store — the body call stands alone.
        let array_ret = matches!(uelem, Ty::Array { .. }) && lower_ty(&uelem).is_some();
        let tgt = self.slot(target);
        if array_ret {
            let dest = format!(
                "({} + ({iv} * {}ULL))",
                tgt.clone().expect("map target local"),
                flat_count(&uelem)
            );
            args.push(dest);
        }
        // #14: trap argument + per-element check only for a trap-capable
        // body (the oracle's in-order first trap needs no check where no
        // guard can fire).
        let capable = self.caps.get(body);
        if capable {
            args.push("trap".into());
        }
        let call = format!("{name}({})", args.join(", "));
        let ret = self.ret_default.clone();
        if array_ret {
            self.line(2, format!("{call};"));
        } else if lower_ty(&uelem).is_some() {
            let t = tgt.expect("map target local");
            self.line(2, format!("{t}[{iv}] = {call};"));
        } else {
            self.line(2, format!("{call};"));
        }
        if capable {
            self.line(2, format!("if (*trap) {ret}"));
        }
        self.line(1, "}");
    }

    pub(super) fn emit_zip(&mut self, source: ObjectId, target: ObjectId) {
        let Some(tgt) = self.slot(target) else {
            return; // fully-erased element product: no data, no traps
        };
        let src_ty = self.obj_ty(source);
        let a_ty = src_ty.component_ty(0).cloned().expect("zip a");
        let b_ty = src_ty.component_ty(1).cloned().expect("zip b");
        let (a_elem, n) = array_parts(&a_ty);
        let (b_elem, _) = array_parts(&b_ty);
        let (elem, _) = array_parts(&self.obj_ty(target));
        // An erased input array (e.g. Array{Unit}) has no slot; its element
        // component is erased in the output product too, so it stores nothing.
        let a = self.slot(pair_component(self.ir, source, 0));
        let b = self.slot(pair_component(self.ir, source, 1));
        let iv = self.tmp();
        self.line(
            1,
            format!("for (unsigned long long {iv} = 0; {iv} < {n}ULL; {iv}++) {{"),
        );
        let bare = residual_arity(&elem) == 1;
        for (k, comp_elem, arr) in [(0u32, &a_elem, &a), (1u32, &b_elem, &b)] {
            if let (Some(eidx), Some(arr)) = (erased_index(&elem, k), arr) {
                let lvalue = if bare {
                    format!("{tgt}[{iv}]")
                } else {
                    format!("{tgt}[{iv}].f{eidx}")
                };
                let rval = elem_expr(arr, &iv, comp_elem);
                self.line(2, format!("{lvalue} = {rval};"));
            }
        }
        self.line(1, "}");
    }

    pub(super) fn emit_enumerate(&mut self, source: ObjectId, target: ObjectId) {
        let Some(tgt) = self.slot(target) else { return };
        let src_ty = self.obj_ty(source);
        let (a_elem, n) = array_parts(&src_ty);
        let (elem, _) = array_parts(&self.obj_ty(target));
        // An erased source array (Array{Unit}) has no slot (F6): the i32
        // index component still stores per element; the element component
        // is erased from the output product too, so it stores nothing (the
        // launch form omits the parameter the same way).
        let a = self.slot(source);
        let iv = self.tmp();
        self.line(
            1,
            format!("for (unsigned long long {iv} = 0; {iv} < {n}ULL; {iv}++) {{"),
        );
        let bare = residual_arity(&elem) == 1;
        if let Some(eidx) = erased_index(&elem, 0) {
            let lvalue = if bare {
                format!("{tgt}[{iv}]")
            } else {
                format!("{tgt}[{iv}].f{eidx}")
            };
            self.line(2, format!("{lvalue} = (int32_t){iv};"));
        }
        if let Some(eidx) = erased_index(&elem, 1) {
            let lvalue = if bare {
                format!("{tgt}[{iv}]")
            } else {
                format!("{tgt}[{iv}].f{eidx}")
            };
            // eidx exists ⟹ the element lowers ⟹ the source array lowers
            // ⟹ it has a slot (the invariant, not a reachable panic).
            let arr = a.as_deref().expect("enumerate source slot");
            let rval = elem_expr(arr, &iv, &a_elem);
            self.line(2, format!("{lvalue} = {rval};"));
        }
        self.line(1, "}");
    }

    pub(super) fn emit_iota(&mut self, target: ObjectId) {
        let Some(tgt) = self.slot(target) else { return };
        // Elem ctype derived — same guard as `iota_kernel`.
        let (elem, n) = array_parts(&self.obj_ty(target));
        let elem_ct = lower_ty(&elem).expect("iota elem lowers");
        let iv = self.tmp();
        self.line(
            1,
            format!("for (unsigned long long {iv} = 0; {iv} < {n}ULL; {iv}++) {{"),
        );
        self.line(2, format!("{tgt}[{iv}] = ({elem_ct}){iv};"));
        self.line(1, "}");
    }

    pub(super) fn emit_fill(&mut self, source: ObjectId, target: ObjectId) {
        let Some(tgt) = self.slot(target) else { return };
        let (_, n) = array_parts(&self.obj_ty(target));
        let (_, val) = self.component_expr(source, 0).expect("fill value");
        let iv = self.tmp();
        self.line(
            1,
            format!("for (unsigned long long {iv} = 0; {iv} < {n}ULL; {iv}++) {{"),
        );
        self.line(2, format!("{tgt}[{iv}] = {val};"));
        self.line(1, "}");
    }

    /// The width-rule extension alone: materialize `(int64_t)idx` into a
    /// temp and return its name. This is the whole device form of a
    /// `bounds_proof`-PROVEN `Index` (S20): the proof puts the index
    /// statically in `[0, n)`, so the §3 guard can never fire and is dead
    /// text — the read below rides the same temp either way.
    pub(super) fn extend_only(&mut self, idx_expr: &str) -> String {
        let i64v = self.tmp();
        self.line(1, format!("int64_t {i64v} = {};", extend_index(idx_expr)));
        i64v
    }

    /// The width-rule extension + signed two-sided bounds guard, device form:
    /// materialize `(int64_t)idx` into a temp, then `idx < 0 || idx >= n`
    /// stores index_oob (kind+1 ⇒ 2u) and returns (§3). Returns the temp name.
    pub(super) fn guard_index(&mut self, idx_expr: &str, n: u64) -> String {
        let i64v = self.extend_only(idx_expr);
        let cond = bounds_cond(&i64v, n);
        let ret = self.ret_default.clone();
        self.line(1, format!("if ({cond}) {{ *trap = 2u; {ret} }}"));
        i64v
    }

    pub(super) fn emit_index(&mut self, m: MorphismId, source: ObjectId, target: ObjectId) {
        let src_ty = self.obj_ty(source);
        let arr_ty = src_ty.component_ty(0).cloned().expect("index array");
        let (elem, n) = array_parts(&arr_ty);
        let (_, idx) = self.component_expr(source, 1).expect("index idx operand");
        // S20: proven in-bounds ⇒ the §3 guard is dead text (trap-freedom is
        // exact) — extension temp only; unproven keeps the guard verbatim.
        let i64v = if self.proof.proven(m) {
            self.extend_only(&idx)
        } else {
            self.guard_index(&idx, n)
        };
        if lower_ty(&self.obj_ty(target)).is_none() {
            return; // erased element: the guard is the whole op
        }
        let arr = self
            .slot(pair_component(self.ir, source, 0))
            .expect("index array slot");
        if matches!(elem, Ty::Array { .. }) {
            // Array element: fresh per-thread local sub-copy (array targets
            // are always Named — the query keeps arrays out of Inline).
            let tgt = self.slot(target).expect("index target");
            let m = flat_count(&elem);
            let jv = self.tmp();
            self.line(
                1,
                format!("for (unsigned long long {jv} = 0; {jv} < {m}ULL; {jv}++) {{"),
            );
            self.line(
                2,
                format!("{tgt}[{jv}] = {arr}[((unsigned long long){i64v} * {m}ULL) + {jv}];"),
            );
            self.line(1, "}");
        } else {
            // A proven Index is guard-free, so its scalar result may be
            // Inline — route through store_obj (WP-B); an unproven one is
            // Named by the query (guarded producer) and stays a statement.
            self.store_obj(target, &format!("{arr}[(unsigned long long){i64v}]"));
        }
    }

    pub(super) fn emit_update(&mut self, source: ObjectId, target: ObjectId) {
        let src_ty = self.obj_ty(source);
        let arr_ty = src_ty.component_ty(0).cloned().expect("update array");
        let (elem, n) = array_parts(&arr_ty);
        let (_, idx) = self.component_expr(source, 1).expect("update idx operand");
        let i64v = self.guard_index(&idx, n);
        let in_place = self.in_place.contains_key(target);
        let tgt = if in_place {
            None
        } else {
            let Some(t) = self.slot(target) else {
                return; // erased element: the guard is the whole op
            };
            Some(t)
        };
        let src = self
            .slot(pair_component(self.ir, source, 0))
            .expect("update src");
        let (_, val) = self.component_expr(source, 2).expect("update value");
        let dst = match &tgt {
            Some(tgt) => {
                // §1 inline column: per-thread local copy, then the store.
                let iv = self.tmp();
                self.line(
                    1,
                    format!(
                        "for (unsigned long long {iv} = 0; {iv} < {}ULL; {iv}++) {{",
                        flat_count(&arr_ty)
                    ),
                );
                self.line(2, format!("{tgt}[{iv}] = {src}[{iv}];"));
                self.line(1, "}");
                tgt.clone()
            }
            // In place (plan-last-use §2 rule 4): the source array is dead
            // at this update — the copy is elided and the store lands in the
            // source's storage; the target aliases it from here on (no
            // produced local was declared for it).
            None => src.clone(),
        };
        if matches!(elem, Ty::Array { .. }) {
            let m = flat_count(&elem);
            let jv = self.tmp();
            self.line(
                1,
                format!("for (unsigned long long {jv} = 0; {jv} < {m}ULL; {jv}++) {{"),
            );
            self.line(
                2,
                format!("{dst}[((unsigned long long){i64v} * {m}ULL) + {jv}] = {val}[{jv}];"),
            );
            self.line(1, "}");
        } else {
            self.line(1, format!("{dst}[(unsigned long long){i64v}] = {val};"));
        }
        if in_place {
            self.slots.insert(target, dst);
        }
    }

    /// A per-thread sequential Fold loop — the oracle verbatim (acc slot 0,
    /// `(acc, e)` per step, in-order), acc in per-thread local storage (§5).
    /// An array-typed acc lives directly in the target's local array.
    /// ADR-0027: a capturing fold's source is `(c₁…cₖ, Acc, [T; n])` — the
    /// captures (per-thread reads of the same buffers/scalars) lead the
    /// body-input argument.
    pub(super) fn emit_fold(
        &mut self,
        source: ObjectId,
        target: ObjectId,
        body: FuncId,
        captures: u32,
    ) {
        let src_ty = self.obj_ty(source);
        let acc_ty = src_ty.component_ty(captures).cloned().expect("fold acc");
        let arr_ty = src_ty
            .component_ty(captures + 1)
            .cloned()
            .expect("fold array");
        let (elem, n) = array_parts(&arr_ty);
        let pair_ty = self.obj_ty(self.ir.func(body).expect("body resolves").input);
        // An erased array (Array{Unit}) has no slot; its element is erased
        // from the pair too, so the loop only re-calls the body.
        let arr = self.slot(pair_component(self.ir, source, captures + 1));
        // An erased-element acc (Array{Unit}) has no storage anywhere (F6):
        // treated like an erased scalar acc — the loop still calls the body.
        let array_acc = matches!(acc_ty, Ty::Array { .. }) && lower_ty(&acc_ty).is_some();

        // acc storage: array acc ⇒ the target local; scalar acc ⇒ a temp
        // initialized from the seed; erased acc ⇒ none.
        let acc_name = if array_acc {
            let tgt = self.slot(target).expect("fold target local");
            let (_, seed) = self.component_expr(source, captures).expect("fold seed");
            let jv = self.tmp();
            let m = flat_count(&acc_ty);
            self.line(
                1,
                format!("for (unsigned long long {jv} = 0; {jv} < {m}ULL; {jv}++) {{"),
            );
            self.line(2, format!("{tgt}[{jv}] = {seed}[{jv}];"));
            self.line(1, "}");
            Some(tgt)
        } else if let Some(ct) = lower_ty(&acc_ty) {
            let (_, seed) = self.component_expr(source, captures).expect("fold seed");
            let t = self.tmp();
            self.line(1, format!("{ct} {t} = {seed};"));
            Some(t)
        } else {
            None
        };

        let iv = self.tmp();
        // The (c₁…cₖ, acc, e) assembly under the residual remap — the
        // captures are the leading fields of the body-input product.
        let acc_expr = if lower_ty(&acc_ty).is_some() {
            acc_name.clone()
        } else {
            None
        };
        let earg = if lower_ty(&elem).is_some() {
            Some(elem_expr(
                arr.as_deref().expect("fold array slot"),
                &iv,
                &elem,
            ))
        } else {
            None
        };
        let mut parts: Vec<Option<String>> = Vec::new();
        for j in 0..captures {
            // WP-B (R-NODUP): through the source product's field — see
            // emit_map's capture loop.
            parts.push(self.component_expr(source, j).map(|(_, v)| v));
        }
        parts.push(acc_expr);
        parts.push(earg);
        let mut varying = vec![false; captures as usize];
        varying.extend([!array_acc, true]);
        let (pre, inloop, arg) = assemble_body_arg(&pair_ty, &parts, &varying);
        for l in pre {
            self.line(1, l);
        }
        self.line(
            1,
            format!("for (unsigned long long {iv} = 0; {iv} < {n}ULL; {iv}++) {{"),
        );
        for l in inloop {
            self.line(2, l);
        }
        let name = self.quals.device_name(self.fnames, body);
        let mut args: Vec<String> = Vec::new();
        if let Some(a) = arg {
            args.push(a);
        }
        if array_acc {
            args.push(acc_name.clone().unwrap());
        }
        // #14: trap argument + per-step check only for a trap-capable body.
        let capable = self.caps.get(body);
        if capable {
            args.push("trap".into());
        }
        let call = format!("{name}({})", args.join(", "));
        let ret = self.ret_default.clone();
        if array_acc {
            self.line(2, format!("{call};"));
        } else if acc_name.is_some() {
            self.line(2, format!("{} = {call};", acc_name.clone().unwrap()));
        } else {
            self.line(2, format!("{call};"));
        }
        if capable {
            self.line(2, format!("if (*trap) {ret}"));
        }
        self.line(1, "}");
        // Scalar acc lands in the target slot; array acc is already home.
        if !array_acc
            && acc_name.is_some()
            && let Some(tgt) = self.slot(target)
        {
            self.line(1, format!("{tgt} = {};", acc_name.unwrap()));
        }
    }

    /// A body-local array literal (§2 item 8): per-element stores into the
    /// per-thread local array; nested elements copy cellwise from the
    /// sub-array handle.
    pub(super) fn emit_literal(&mut self, target: ObjectId) {
        let tgt = match self.slot(target) {
            Some(t) => t,
            None => return, // erased element type: no representation
        };
        let arr_ty = self.obj_ty(target);
        let (elem, n) = array_parts(&arr_ty);
        let nested = matches!(elem, Ty::Array { .. });
        let m = flat_count(&elem);
        for k in 0..n {
            let src = pair_source(self.ir, target, k as u32).expect("literal element");
            let Some((_, expr)) = self.load_whole(src) else {
                continue; // erased element
            };
            if nested {
                let jv = self.tmp();
                self.line(
                    1,
                    format!("for (unsigned long long {jv} = 0; {jv} < {m}ULL; {jv}++) {{"),
                );
                self.line(2, format!("{tgt}[{} + {jv}] = {expr}[{jv}];", k * m));
                self.line(1, "}");
            } else {
                self.line(1, format!("{tgt}[{k}] = {expr};"));
            }
        }
    }

    pub(super) fn emit_output(&mut self, source: ObjectId, target: ObjectId) {
        if matches!(self.obj_ty(target), Ty::Array { .. }) {
            // An array Return: copy the source's cells into the fn's Return
            // local; the epilogue copies it to the caller's `out`.
            if let (Some(tgt), Some((_, src))) = (self.slot(target), self.load_whole(source)) {
                let m = flat_count(&self.obj_ty(target));
                let jv = self.tmp();
                self.line(
                    1,
                    format!("for (unsigned long long {jv} = 0; {jv} < {m}ULL; {jv}++) {{"),
                );
                self.line(2, format!("{tgt}[{jv}] = {src}[{jv}];"));
                self.line(1, "}");
            }
        } else if let Some((_, val)) = self.load_whole(source) {
            self.store_obj(target, &val);
        }
    }
}
