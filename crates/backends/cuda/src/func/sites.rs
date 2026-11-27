//! FnEmit array-bulk launch sites (allocation, launch checks, map / zip / fold / update / index sites).

use super::*;

impl<'a> FnEmit<'a> {
    // --- WP3: array-bulk launch sites (kernel.rs owns the __global__ side) -

    /// `cudaMalloc` into the (already-declared) handle `slot` + register it
    /// in the allocation registry (DESIGN §2, allocation-based ownership).
    /// The per-buffer path — loop-cone sites in v1.0 (arena.rs); fn-zone
    /// members go through [`FnEmit::alloc_buffer`]'s pointer init instead.
    pub(super) fn malloc_buffer(&mut self, slot: &str, bytes: &str) {
        self.line(format!(
            "cu_check(cudaMalloc((void**)&{slot}, {bytes}), \"cudaMalloc({slot})\");"
        ));
        self.register_alloc(slot.to_string());
    }

    /// Allocate one construction site's buffer (plan-smart-arenas §3 — the
    /// single choke point, malloc_buffer's arena form): a fn-zone member's
    /// device address is the compile-time `arena0 + OFF` pointer init (no
    /// `cudaMalloc` at the site, nothing registered — the zone frees it); a
    /// loop-cone site keeps the per-buffer `cudaMalloc` + registry entry
    /// (v1.0). Either way the line lands at exactly the point
    /// `malloc_buffer` ran today, so zero-iteration `nullptr` semantics and
    /// per-iteration re-init are preserved verbatim. `ptr_ct` is the slot's
    /// lowered pointer type (the cast target); `bytes` the per-buffer
    /// path's cudaMalloc byte-count text.
    pub(super) fn alloc_buffer(&mut self, key: ArenaKey, slot: &str, ptr_ct: &str, bytes: &str) {
        match self.arena.as_ref().and_then(|p| p.offset(key)) {
            Some(off) => self.line(format!("{slot} = ({ptr_ct})(arena0 + {off}ULL);")),
            None => self.malloc_buffer(slot, bytes),
        }
    }

    /// One kernel launch + the §3 after-EVERY-launch trap check — the check
    /// rides only for a trap-CAPABLE site (#14): a trap-free kernel takes no
    /// trap pointer, so the launch passes no `d_trap` argument and no
    /// synchronizing readback follows (fewer syncs — the perf-visible part).
    /// Wherever any guard can fire the check-after-every-launch convention
    /// is kept verbatim (class parity, first-trap-wins).
    ///
    /// With #19a's `perf_timing` the launch is wrapped in CUDA events:
    /// `Record(start)` before it, `Record(stop)` + `Synchronize` +
    /// `ElapsedTime` after — the stop is recorded BEFORE the trap check, so
    /// the §3 convention is unchanged — printing one machine-readable
    /// `MAPAL_PERF launch=<kernel> ms=<%.4f>` line per EXECUTION (a cone
    /// launch prints per iteration). The elapsed sync is the only added
    /// host sync, and only where #14 had skipped the readback.
    pub(super) fn launch_and_check(
        &mut self,
        m: MorphismId,
        kname: &str,
        dims: &str,
        args: &[String],
        capable: bool,
    ) {
        let mut args = args.to_vec();
        if capable {
            args.push("d_trap".into());
        }
        let launch = format!("{kname}<<<{dims}>>>({});", args.join(", "));
        if self.perf {
            let i = *self.ev_ord.get(&m).expect("launch site has an event pair");
            self.line(format!(
                "cu_check(cudaEventRecord(fev{i}_start), \"cudaEventRecord\");"
            ));
            self.line(launch);
            self.line(format!(
                "cu_check(cudaEventRecord(fev{i}_stop), \"cudaEventRecord\");"
            ));
            self.line(format!(
                "cu_check(cudaEventSynchronize(fev{i}_stop), \"cudaEventSynchronize\");"
            ));
            self.line(format!(
                "cu_check(cudaEventElapsedTime(&mapal_perf_ms, fev{i}_start, fev{i}_stop), \"cudaEventElapsedTime\");"
            ));
            self.line("mapal_perf_total += mapal_perf_ms;");
            self.line(format!(
                "printf(\"MAPAL_PERF launch={kname} ms=%.4f\\n\", mapal_perf_ms);"
            ));
        } else {
            self.line(launch);
        }
        if capable {
            self.line("trap_check_after_launch();");
        }
    }

    /// The host side of one launch-form array-bulk op site: allocate the
    /// output buffer (a fn-zone pointer init, or a per-buffer cudaMalloc for
    /// a cone site — [`FnEmit::alloc_buffer`]), gather operands, launch,
    /// trap-check. Argument order mirrors the kernel's parameter list
    /// positionally (kernel.rs `emit_kernel` is the definitional side);
    /// ADR-0027 captures are the source product's leading components, passed
    /// after the existing operands, before the trap pointer. `capable` is
    /// the site's trap capability (#14, `TrapCaps::site`): it decides the
    /// trailing `d_trap` argument and the post-launch check.
    pub(super) fn emit_bulk_site(
        &mut self,
        m: MorphismId,
        kname: &str,
        source: ObjectId,
        target: ObjectId,
        op: Operation,
        capable: bool,
    ) {
        match op {
            Operation::Map { captures, .. } => {
                self.map_site(m, kname, source, target, captures, capable)
            }
            Operation::Zip => self.zip_site(m, kname, source, target, capable),
            Operation::Enumerate => self.enumerate_site(m, kname, source, target, capable),
            Operation::Iota => self.iota_site(m, kname, source, target, capable),
            Operation::Fill => self.fill_site(m, kname, source, target, capable),
            Operation::Update => self.update_site(m, kname, source, target, capable),
            Operation::Index => self.index_site(m, kname, source, target, capable),
            Operation::Fold { captures, .. } => {
                self.fold_site(m, kname, source, target, captures, capable)
            }
            _ => unreachable!("not a bulk-op site"),
        }
    }

    /// The ADR-0027 capture operands of a map/fold site: the source product's
    /// first `captures` components, each as a launch argument (a device
    /// buffer handle for an array capture, a by-value scalar otherwise) —
    /// appended after the site's existing operands, before `d_trap`.
    pub(super) fn push_capture_args(
        &mut self,
        args: &mut Vec<String>,
        source: ObjectId,
        captures: u32,
    ) {
        for j in 0..captures {
            let cap_obj = self.pair_source(source, j).expect("capture component");
            if let Some((_, v)) = self.load_whole(cap_obj) {
                args.push(v);
            }
        }
    }

    pub(super) fn map_site(
        &mut self,
        m: MorphismId,
        kname: &str,
        source: ObjectId,
        target: ObjectId,
        captures: u32,
        capable: bool,
    ) {
        // ADR-0027: k>0 ⇒ the mapped array is the source product's last
        // component; k=0 ⇒ the source IS the array.
        let arr_obj = if captures == 0 {
            source
        } else {
            self.pair_source(source, captures).expect("map array")
        };
        let arr_ty = self.obj_ty(arr_obj);
        let (_, n) = kernel::array_parts(&arr_ty);
        let tgt_ty = self.obj_ty(target);
        let mut args: Vec<String> = Vec::new();
        if let Some(ptr_ct) = lower_ty(&tgt_ty) {
            let slot = self.slot(target).expect("map target slot");
            self.alloc_buffer(
                ArenaKey::Obj(target),
                &slot,
                &ptr_ct,
                &kernel::buffer_bytes(&tgt_ty),
            );
            args.push(slot);
        }
        if lower_ty(&arr_ty).is_some() {
            args.push(self.slot(arr_obj).expect("map source slot"));
        }
        self.push_capture_args(&mut args, source, captures);
        self.launch_and_check(
            m,
            kname,
            &format!("{}, 256", kernel::grid_expr(n)),
            &args,
            capable,
        );
    }

    pub(super) fn zip_site(
        &mut self,
        m: MorphismId,
        kname: &str,
        source: ObjectId,
        target: ObjectId,
        capable: bool,
    ) {
        let src_ty = self.obj_ty(source);
        let a_ty = src_ty.component_ty(0).cloned().expect("zip a");
        let (_, n) = kernel::array_parts(&a_ty);
        let tgt_ty = self.obj_ty(target);
        let mut args: Vec<String> = Vec::new();
        if let Some(ptr_ct) = lower_ty(&tgt_ty) {
            let slot = self.slot(target).expect("zip target slot");
            self.alloc_buffer(
                ArenaKey::Obj(target),
                &slot,
                &ptr_ct,
                &kernel::buffer_bytes(&tgt_ty),
            );
            args.push(slot);
        }
        for k in 0..2 {
            let arr = self.pair_source(source, k).expect("zip input");
            if let Some(s) = self.slot(arr) {
                args.push(s);
            }
        }
        self.launch_and_check(
            m,
            kname,
            &format!("{}, 256", kernel::grid_expr(n)),
            &args,
            capable,
        );
    }

    pub(super) fn enumerate_site(
        &mut self,
        m: MorphismId,
        kname: &str,
        source: ObjectId,
        target: ObjectId,
        capable: bool,
    ) {
        let src_ty = self.obj_ty(source);
        let (_, n) = kernel::array_parts(&src_ty);
        let tgt_ty = self.obj_ty(target);
        let mut args: Vec<String> = Vec::new();
        if let Some(ptr_ct) = lower_ty(&tgt_ty) {
            let slot = self.slot(target).expect("enumerate target slot");
            self.alloc_buffer(
                ArenaKey::Obj(target),
                &slot,
                &ptr_ct,
                &kernel::buffer_bytes(&tgt_ty),
            );
            args.push(slot);
        }
        if lower_ty(&src_ty).is_some() {
            args.push(self.slot(source).expect("enumerate source slot"));
        }
        self.launch_and_check(
            m,
            kname,
            &format!("{}, 256", kernel::grid_expr(n)),
            &args,
            capable,
        );
    }

    pub(super) fn iota_site(
        &mut self,
        m: MorphismId,
        kname: &str,
        source: ObjectId,
        target: ObjectId,
        capable: bool,
    ) {
        let tgt_ty = self.obj_ty(target);
        let (_, n) = kernel::array_parts(&tgt_ty);
        let mut args = Vec::new();
        if let Some(ptr_ct) = lower_ty(&tgt_ty) {
            let slot = self.slot(target).expect("iota target slot");
            self.alloc_buffer(
                ArenaKey::Obj(target),
                &slot,
                &ptr_ct,
                &kernel::buffer_bytes(&tgt_ty),
            );
            args.push(slot);
        }
        let (_, count) = self.load_whole(source).expect("iota count");
        args.push(count);
        self.launch_and_check(
            m,
            kname,
            &format!("{}, 256", kernel::grid_expr(n)),
            &args,
            capable,
        );
    }

    pub(super) fn fill_site(
        &mut self,
        m: MorphismId,
        kname: &str,
        source: ObjectId,
        target: ObjectId,
        capable: bool,
    ) {
        let tgt_ty = self.obj_ty(target);
        let (_, n) = kernel::array_parts(&tgt_ty);
        let mut args = Vec::new();
        if let Some(ptr_ct) = lower_ty(&tgt_ty) {
            let slot = self.slot(target).expect("fill target slot");
            self.alloc_buffer(
                ArenaKey::Obj(target),
                &slot,
                &ptr_ct,
                &kernel::buffer_bytes(&tgt_ty),
            );
            args.push(slot);
        }
        let count = self.pair_source(source, 1).expect("fill count");
        let (_, count) = self.load_whole(count).expect("fill count operand");
        args.push(count);
        let value = self.pair_source(source, 0).expect("fill value");
        if let Some((_, value)) = self.load_whole(value) {
            args.push(value);
        }
        self.launch_and_check(
            m,
            kname,
            &format!("{}, 256", kernel::grid_expr(n)),
            &args,
            capable,
        );
    }

    pub(super) fn update_site(
        &mut self,
        m: MorphismId,
        kname: &str,
        source: ObjectId,
        target: ObjectId,
        capable: bool,
    ) {
        let src_ty = self.obj_ty(source);
        let arr_ty = src_ty.component_ty(0).cloned().expect("update array");
        let flat = kernel::flat_count(&arr_ty);
        let mut args: Vec<String> = Vec::new();
        if let Some(ptr_ct) = lower_ty(&arr_ty) {
            let slot = self.slot(target).expect("update target slot");
            let src = self.pair_source(source, 0).expect("update src");
            let src_slot = self.slot(src).expect("update src slot");
            if in_place_update(self.ir, &self.last_use, self.f, m, src) {
                // In place (plan-last-use §2 rule 4): the source array dies
                // at this update — no fresh buffer; the target handle IS the
                // source handle. The element-write kernel is race-free on
                // aliased out/src (each thread touches disjoint indices),
                // the bounds guard is unchanged, and the old array value is
                // dead, so the mutation is unobservable. An arena-member
                // site keeps its offset reserved-but-unused (the recorded
                // v1 simplification — capacity is slightly over-reserved).
                self.line(format!("{slot} = {src_slot};"));
            } else {
                self.alloc_buffer(
                    ArenaKey::Obj(target),
                    &slot,
                    &ptr_ct,
                    &kernel::buffer_bytes(&arr_ty),
                );
            }
            args.push(slot);
            args.push(src_slot);
        }
        // §3 width rule: extend the index per its Ty to int64_t (the C++
        // conversion is value-preserving — llvm's zext/sext split realized).
        let idx_obj = self.pair_source(source, 1).expect("update idx");
        let (_, idx) = self.load_whole(idx_obj).expect("update idx operand");
        args.push(kernel::extend_index(&idx));
        let val_obj = self.pair_source(source, 2).expect("update val");
        if lower_ty(&self.obj_ty(val_obj)).is_some() {
            let (_, v) = self.load_whole(val_obj).expect("update val operand");
            args.push(v);
        }
        self.launch_and_check(
            m,
            kname,
            &format!("{}, 256", kernel::grid_expr(flat)),
            &args,
            capable,
        );
    }

    pub(super) fn index_site(
        &mut self,
        m: MorphismId,
        kname: &str,
        source: ObjectId,
        target: ObjectId,
        capable: bool,
    ) {
        let src_ty = self.obj_ty(source);
        let arr_ty = src_ty.component_ty(0).cloned().expect("index array");
        let tgt_ty = self.obj_ty(target);
        let arr_obj = self.pair_source(source, 0).expect("index array");
        let idx_obj = self.pair_source(source, 1).expect("index idx");
        let (_, idx) = self.load_whole(idx_obj).expect("index idx operand");
        let mut args: Vec<String> = Vec::new();
        // The result buffer: a 1-cell device buffer for scalar elements (the
        // host memcpy's it D→H below — §2 item 5); an array-typed element
        // lands in a fresh device buffer that IS the target handle (§1).
        let mut cell: Option<String> = None;
        if let Some(ct) = lower_ty(&tgt_ty) {
            if matches!(tgt_ty, Ty::Array { .. }) {
                let slot = self.slot(target).expect("index target slot");
                self.alloc_buffer(
                    ArenaKey::Obj(target),
                    &slot,
                    &ct,
                    &kernel::buffer_bytes(&tgt_ty),
                );
                args.push(slot);
            } else {
                // The 1-cell readback buffer is declared at fn scope
                // (hoisted, nullptr-initialized): in a loop cone the site
                // iterates but the free at fn exit must still name it.
                let t = self.tmp();
                self.decls.push_str(&format!("  {ct}* {t} = nullptr;\n"));
                self.alloc_buffer(
                    ArenaKey::Cell(m),
                    &t,
                    &format!("{ct}*"),
                    &format!("sizeof({ct}) * 1ULL"),
                );
                args.push(t.clone());
                cell = Some(t);
            }
        }
        if lower_ty(&arr_ty).is_some() {
            args.push(self.slot(arr_obj).expect("index array slot"));
        }
        args.push(kernel::extend_index(&idx));
        self.launch_and_check(m, kname, "1, 1", &args, capable);
        if let Some(t) = cell {
            let ct = lower_ty(&tgt_ty).expect("index result lowers");
            let slot = self.slot(target).expect("index target slot");
            self.line(format!(
                "cu_check(cudaMemcpy(&{slot}, {t}, sizeof({ct}), cudaMemcpyDeviceToHost), \"cudaMemcpy(index)\");"
            ));
        }
    }

    pub(super) fn fold_site(
        &mut self,
        m: MorphismId,
        kname: &str,
        source: ObjectId,
        target: ObjectId,
        captures: u32,
        capable: bool,
    ) {
        let src_ty = self.obj_ty(source);
        // ADR-0027: the source product is `(c₁…cₖ, Acc, [T; n])` — the
        // captures shift the acc to component k, the array to k+1.
        let acc_ty = src_ty.component_ty(captures).cloned().expect("fold acc");
        let arr_ty = src_ty
            .component_ty(captures + 1)
            .cloned()
            .expect("fold array");
        let tgt_ty = self.obj_ty(target);
        let acc_obj = self.pair_source(source, captures).expect("fold seed");
        let arr_obj = self.pair_source(source, captures + 1).expect("fold array");
        let mut args: Vec<String> = Vec::new();
        // Scalar acc ⇒ 1-cell device buffer + D→H readback (§2 item 6); an
        // array acc stays as the result buffer (§1 — the target handle).
        let mut cell: Option<String> = None;
        if let Some(ct) = lower_ty(&tgt_ty) {
            if matches!(acc_ty, Ty::Array { .. }) {
                let slot = self.slot(target).expect("fold target slot");
                self.alloc_buffer(
                    ArenaKey::Obj(target),
                    &slot,
                    &ct,
                    &kernel::buffer_bytes(&acc_ty),
                );
                args.push(slot);
            } else {
                // The 1-cell readback buffer is declared at fn scope
                // (hoisted, nullptr-initialized), like Index's.
                let t = self.tmp();
                self.decls.push_str(&format!("  {ct}* {t} = nullptr;\n"));
                self.alloc_buffer(
                    ArenaKey::Cell(m),
                    &t,
                    &format!("{ct}*"),
                    &format!("sizeof({ct}) * 1ULL"),
                );
                args.push(t.clone());
                cell = Some(t);
            }
        }
        match &acc_ty {
            Ty::Array { .. } => {
                // An erased-element acc (Array{Unit}) has no slot: the seed
                // argument is omitted, matching the kernel's omitted acc0
                // parameter — the launch still runs (F6).
                if let Some(s) = self.slot(acc_obj) {
                    args.push(s);
                }
            }
            _ => {
                if lower_ty(&acc_ty).is_some() {
                    let (_, v) = self.load_whole(acc_obj).expect("fold seed operand");
                    args.push(v);
                }
            }
        }
        if lower_ty(&arr_ty).is_some() {
            args.push(self.slot(arr_obj).expect("fold array slot"));
        }
        self.push_capture_args(&mut args, source, captures);
        self.launch_and_check(m, kname, "1, 1", &args, capable);
        if let Some(t) = cell {
            let ct = lower_ty(&tgt_ty).expect("fold result lowers");
            let slot = self.slot(target).expect("fold target slot");
            self.line(format!(
                "cu_check(cudaMemcpy(&{slot}, {t}, sizeof({ct}), cudaMemcpyDeviceToHost), \"cudaMemcpy(fold)\");"
            ));
        }
    }
}
