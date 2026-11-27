//! `Index` and `Update` emission, bounds guards and body-call arguments

use super::*;

impl<'a> FnEmit<'a> {
    pub(in crate::func) fn emit_index(
        &mut self,
        m: MorphismId,
        source: ObjectId,
        target: ObjectId,
    ) {
        let src_ty = self.obj_ty(source);
        let arr_ty = src_ty.component_ty(0).cloned().expect("index array");
        let (elem_llt, size) = array_parts(&arr_ty);
        if let Some(v) = self.cell_val(source) {
            self.store_obj(target, &elem_llt, &v);
            return;
        }
        let arr_llt = lower_ty(&arr_ty).expect("array lowers");
        let idx_ty = src_ty.component_ty(1).cloned().expect("index i");
        let elem_ty = arr_ty.component_ty(0).cloned().expect("array element");
        if has_arrays(&elem_ty)
            && !self
                .pair_source(source, 0)
                .is_some_and(|a| self.host_free(a))
        {
            // structure: an element holding arrays (a row of a nested array,
            // a struct of an array of structs) is read in place — a view of
            // its address when the plan made the target one, else copied
            // field by field; never one SSA aggregate.
            let arr_ptr = self
                .array_operand_ptr(source, Some(0))
                .expect("index array ptr");
            let i64idx = self.load_index(source, 1, &idx_ty);
            let i64idx = if self.bp.proven(m) {
                i64idx
            } else if matches!(self.guard_flavor, GuardFlavor::Host) {
                self.guard_index(&i64idx, size);
                i64idx
            } else {
                // A task / device body records the trap and reads cell 0.
                let oob = self.index_oob(&i64idx, size);
                let bad = self.label();
                let good = self.label();
                self.line(format!("br i1 {oob}, label %{bad}, label %{good}"));
                self.label_line(&bad);
                self.record_trap(m, 1);
                self.line(format!("br label %{good}"));
                self.label_line(&good);
                let c = self.tmp();
                self.line(format!("{c} = select i1 {oob}, i64 0, i64 {i64idx}"));
                c
            };
            let ep = self.tmp();
            self.line(format!(
                "{ep} = getelementptr {arr_llt}, ptr {arr_ptr}, i64 0, i64 {i64idx}"
            ));
            if let Some(dst) = self.slot(target) {
                let tp = self.slot_paths(target);
                self.copy_layout(&elem_ty, &dst, &tp, &ep, &[]);
            }
            return;
        }
        // S54 `arena`: a device-only array's element is copied out into a
        // scratch cell (its reads are proven, so no guard).
        if self
            .pair_source(source, 0)
            .is_some_and(|a| self.host_free(a))
        {
            assert!(
                self.bp.proven(m)
                    && self
                        .resident
                        .as_ref()
                        .is_some_and(|p| p.elem.contains_key(m)),
                "a device-only array's reads are proven one-element copy-outs"
            );
            let i64idx = self.load_index(source, 1, &idx_ty);
            let ep = self.scratch(&elem_llt);
            self.resident_elem(m, source, &ep, &i64idx);
            let v = self.tmp();
            self.line(format!("{v} = load {elem_llt}, ptr {ep}"));
            self.store_obj(target, &elem_llt, &v);
            return;
        }
        let arr_ptr = self
            .array_operand_ptr(source, Some(0))
            .expect("index array ptr");
        let i64idx = self.load_index(source, 1, &idx_ty);
        // A body's device variant (S54): a one-cell buffer holds cell `t`.
        let i64idx = if self.cell_read(source) {
            "0".to_string()
        } else {
            i64idx
        };
        // Guard elision (S20 `bounds_proof`; the vectorization unlock): the
        // plan proves the index statically inside `[0, size)` — the trap is
        // dead, so a proven `Index` emits just the GEP+load. Everything
        // unproven keeps the two-sided guard byte-identical.
        if !self.bp.proven(m) {
            if !matches!(self.guard_flavor, GuardFlavor::Host) {
                let oob = self.index_oob(&i64idx, size);
                let bad = self.label();
                let good = self.label();
                let done = self.label();
                self.line(format!("br i1 {oob}, label %{bad}, label %{good}"));
                self.label_line(&bad);
                self.record_trap(m, 1);
                self.line(format!("br label %{done}"));
                self.label_line(&good);
                let ep = self.tmp();
                self.line(format!(
                    "{ep} = getelementptr {arr_llt}, ptr {arr_ptr}, i64 0, i64 {i64idx}"
                ));
                let loaded = self.tmp();
                self.line(format!("{loaded} = load {elem_llt}, ptr {ep}"));
                self.line(format!("br label %{done}"));
                self.label_line(&done);
                let value = self.tmp();
                self.line(format!(
                    "{value} = phi {elem_llt} [zeroinitializer, %{bad}], [{loaded}, %{good}]"
                ));
                self.store_obj(target, &elem_llt, &value);
                return;
            }
            self.guard_index(&i64idx, size);
        }
        let ep = self.tmp();
        self.line(format!(
            "{ep} = getelementptr {arr_llt}, ptr {arr_ptr}, i64 0, i64 {i64idx}"
        ));
        self.resident_elem(m, source, &ep, &i64idx);
        let v = self.tmp();
        self.line(format!("{v} = load {elem_llt}, ptr {ep}"));
        self.store_obj(target, &elem_llt, &v);
    }

    pub(in crate::func) fn emit_update(
        &mut self,
        m: MorphismId,
        source: ObjectId,
        target: ObjectId,
    ) {
        let src_ty = self.obj_ty(source);
        let arr_ty = src_ty.component_ty(0).cloned().expect("update array");
        let (_elem_llt, size) = array_parts(&arr_ty);
        let arr_llt = lower_ty(&arr_ty).expect("array lowers");
        let idx_ty = src_ty.component_ty(1).cloned().expect("update i");
        // Last-use elision (suggestions #2; plan-last-use §2 rule 4): a dead
        // source shares its slot with the target — the memcpy is skipped and
        // the element store below lands in place. The copy path emits
        // byte-identical text to before.
        // The decision `prepare_storage` made (and laid the frame out by).
        let (tgt_slot, copy_from) = match self.update_aliases.get(target).copied() {
            Some(arr) => {
                let slot = self.slot(arr).expect("in-place update source slot");
                self.slots.insert(target, slot.clone());
                if self.ptr_resident.contains_key(arr) {
                    // A loop-owned view: the write lands through its pointer.
                    let p = self.tmp();
                    self.line(format!("{p} = load ptr, ptr {slot}"));
                    (p, None)
                } else {
                    (slot, None)
                }
            }
            None => (
                self.slot(target).expect("update target slot"),
                Some(
                    self.array_operand_ptr(source, Some(0))
                        .expect("update src ptr"),
                ),
            ),
        };
        let i64idx = self.load_index(source, 1, &idx_ty);
        if !matches!(self.guard_flavor, GuardFlavor::Host) {
            if let Some(src_arr_ptr) = copy_from {
                self.line(format!(
                    "call void @llvm.memcpy.p0.p0.i64(ptr {tgt_slot}, ptr {src_arr_ptr}, i64 ptrtoint (ptr getelementptr ({arr_llt}, ptr null, i64 1) to i64), i1 false)"
                ));
            }
            let oob = self.index_oob(&i64idx, size);
            let bad = self.label();
            let good = self.label();
            let done = self.label();
            self.line(format!("br i1 {oob}, label %{bad}, label %{good}"));
            self.label_line(&bad);
            self.record_trap(m, 1);
            self.line(format!("br label %{done}"));
            self.label_line(&good);
            let ep = self.tmp();
            self.line(format!(
                "{ep} = getelementptr {arr_llt}, ptr {tgt_slot}, i64 0, i64 {i64idx}"
            ));
            let (vllt, val) = self.load_component(source, 2).expect("update value");
            self.line(format!("store {vllt} {val}, ptr {ep}"));
            self.line(format!("br label %{done}"));
            self.label_line(&done);
            return;
        }
        self.guard_index(&i64idx, size);
        if let Some(src_arr_ptr) = copy_from {
            // memcpy source array → target (fresh array; ADR-0021). Size via the
            // gep-null sizeof constant expr (handles element padding).
            self.line(format!(
                "call void @llvm.memcpy.p0.p0.i64(ptr {tgt_slot}, ptr {src_arr_ptr}, i64 ptrtoint (ptr getelementptr ({arr_llt}, ptr null, i64 1) to i64), i1 false)"
            ));
        }
        let ep = self.tmp();
        self.line(format!(
            "{ep} = getelementptr {arr_llt}, ptr {tgt_slot}, i64 0, i64 {i64idx}"
        ));
        let elem_ty = arr_ty.component_ty(0).cloned().expect("array element");
        if has_arrays(&elem_ty)
            && let Some((src, spaths)) = self.component_value_ptr(source, 2)
        {
            // structure: an element of arrays written field by field.
            self.copy_layout(&elem_ty, &ep, &[], &src, &spaths);
            return;
        }
        let (vllt, val) = self.load_component(source, 2).expect("update value");
        self.line(format!("store {vllt} {val}, ptr {ep}"));
    }

    /// Load index component `k`, zero/sign-extended to i64 per its ty (S13
    /// type-directed rule: u8 zext, signed sext).
    pub(in crate::func) fn load_index(&mut self, agg: ObjectId, k: u32, idx_ty: &Ty) -> String {
        let (illt, idx) = self.load_component(agg, k).expect("index operand");
        if illt == "i64" {
            return idx;
        }
        let ext = if matches!(idx_ty, Ty::Int { signed: false, .. }) {
            "zext"
        } else {
            "sext"
        };
        let e = self.tmp();
        self.line(format!("{e} = {ext} {illt} {idx} to i64"));
        e
    }

    /// Trap when the i64 index is out of `[0, size)`. Unsigned (u8) indices skip
    /// the lower bound (they zero-extend, never negative — S13).
    pub(in crate::func) fn guard_index(&mut self, i64idx: &str, size: u64) {
        // The extension already erased signedness; but the operand's original
        // signedness decided zext vs sext. A zero-extended value is ≥ 0, so the
        // signed two-sided check is always correct on the i64 form.
        let oob = self.index_oob(i64idx, size);
        self.trap_if(&oob, 1);
    }

    pub(in crate::func) fn index_oob(&mut self, i64idx: &str, size: u64) -> String {
        let lo = self.tmp();
        self.line(format!("{lo} = icmp slt i64 {i64idx}, 0"));
        let hi = self.tmp();
        self.line(format!("{hi} = icmp sge i64 {i64idx}, {size}"));
        let oob = self.tmp();
        self.line(format!("{oob} = or i1 {lo}, {hi}"));
        oob
    }

    /// Assemble a capturing map/fold body's call operand (ADR-0027): the
    /// `(c₁…cₖ, rest…)` product in a fresh scratch — capture components `0..k`
    /// loaded from the op's source product (the broadcast edges), then the
    /// per-iteration `rest` components (`elem` for map; `acc, elem` for fold).
    /// Returns the `{llt} {val}` call operand. Erasure applies as usual: an
    /// erased component has no representation (`field_store` remaps via
    /// `erased_index`; a lowered capture is never erased — L1605).
    ///
    /// An Array capture travels **by reference** (suggestions #6): its scratch
    /// field is a `ptr` (matching the body fn's by-ref signature) holding the
    /// array's address — the forwarded capture pointer when the `Pair` feeder
    /// is itself ptr-resident (the transitive fold-in-map case), the source
    /// product's component address otherwise. No array bytes move per call.
    /// (`emit_call` reuses this template for a Named call's whole argument —
    /// every component a "capture", no `rest` — suggestions #8.)
    pub(in crate::func) fn body_call_arg(
        &mut self,
        source: ObjectId,
        captures: u32,
        arg_ty: &Ty,
        arg_llt: &str,
        rest: &[(&str, &str)],
    ) -> String {
        let buf = self.scratch(arg_llt);
        for i in 0..captures {
            if let Some(cty) = arg_ty.component_ty(i).cloned()
                && !matches!(cty, Ty::Array { .. })
                && has_arrays(&cty)
            {
                // structure: a captured product's arrays by address
                // (`capture_paths`).
                let (src, spaths) = self
                    .component_value_ptr(source, i)
                    .expect("body capture storage");
                let fp = self
                    .field_ptr(&buf, arg_ty, arg_llt, i)
                    .expect("capture field");
                self.copy_layout(&cty, &fp, &array_paths(&cty), &src, &spaths);
            } else if matches!(arg_ty.component_ty(i), Some(Ty::Array { .. })) {
                let addr = self
                    .array_operand_ptr(source, Some(i))
                    .expect("body capture addr");
                self.field_store(&buf, arg_ty, arg_llt, i, "ptr", &addr);
            } else {
                let (cllt, cval) = self.load_component(source, i).expect("body capture");
                self.field_store(&buf, arg_ty, arg_llt, i, &cllt, &cval);
            }
        }
        for (j, &(rllt, rval)) in rest.iter().enumerate() {
            self.field_store(&buf, arg_ty, arg_llt, captures + j as u32, rllt, rval);
        }
        let whole = self.tmp();
        self.line(format!("{whole} = load {arg_llt}, ptr {buf}"));
        format!("{arg_llt} {whole}")
    }
}
