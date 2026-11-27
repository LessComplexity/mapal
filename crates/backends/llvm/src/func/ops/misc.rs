//! task division, comparison, calls, print / time / read and in-place sources

use super::*;

impl<'a> FnEmit<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::func) fn emit_task_div(
        &mut self,
        m: MorphismId,
        target: ObjectId,
        op: Operation,
        llt: &str,
        a: &str,
        b: &str,
        signed: bool,
        min_dead: bool,
    ) {
        let zero = self.tmp();
        self.line(format!("{zero} = icmp eq {llt} {b}, 0"));
        let bad = self.label();
        let good = self.label();
        let done = self.label();
        self.line(format!("br i1 {zero}, label %{bad}, label %{good}"));
        self.label_line(&bad);
        self.record_trap(m, 0);
        self.line(format!("br label %{done}"));
        self.label_line(&good);

        let (good_value, good_block) = if signed && !min_dead {
            let min = int_min(llt);
            let minus_one = self.tmp();
            self.line(format!("{minus_one} = icmp eq {llt} {b}, -1"));
            let is_min = self.tmp();
            self.line(format!("{is_min} = icmp eq {llt} {a}, {min}"));
            let overflow = self.tmp();
            self.line(format!("{overflow} = and i1 {minus_one}, {is_min}"));
            let wrap = self.label();
            let normal = self.label();
            let wrapped = self.label();
            self.line(format!("br i1 {overflow}, label %{wrap}, label %{normal}"));
            self.label_line(&wrap);
            self.line(format!("br label %{wrapped}"));
            self.label_line(&normal);
            let real = self.tmp();
            let instruction = if op == Operation::Div { "sdiv" } else { "srem" };
            self.line(format!("{real} = {instruction} {llt} {a}, {b}"));
            self.line(format!("br label %{wrapped}"));
            self.label_line(&wrapped);
            let value = self.tmp();
            let wrap_value = if op == Operation::Div { min } else { "0" };
            self.line(format!(
                "{value} = phi {llt} [{wrap_value}, %{wrap}], [{real}, %{normal}]"
            ));
            (value, wrapped)
        } else {
            let instruction = match (signed, op) {
                (true, Operation::Div) => "sdiv",
                (true, Operation::Mod) => "srem",
                (false, Operation::Div) => "udiv",
                (false, Operation::Mod) => "urem",
                _ => unreachable!(),
            };
            let value = self.tmp();
            self.line(format!("{value} = {instruction} {llt} {a}, {b}"));
            (value, good)
        };
        self.line(format!("br label %{done}"));
        self.label_line(&done);
        let value = self.tmp();
        self.line(format!(
            "{value} = phi {llt} [0, %{bad}], [{good_value}, %{good_block}]"
        ));
        self.store_obj(target, llt, &value);
    }

    pub(in crate::func) fn emit_compare(
        &mut self,
        source: ObjectId,
        target: ObjectId,
        op: Operation,
    ) {
        let opty = self
            .obj_ty(source)
            .component_ty(0)
            .cloned()
            .expect("cmp ty");
        let (llt, a) = self.load_component(source, 0).expect("cmp a");
        let (_, b) = self.load_component(source, 1).expect("cmp b");
        let r = self.tmp();
        if is_float(&opty) {
            let pred = match op {
                Operation::Eq => "oeq",
                Operation::Neq => "une",
                Operation::Lt => "olt",
                Operation::Gt => "ogt",
                Operation::Le => "ole",
                Operation::Ge => "oge",
                _ => unreachable!(),
            };
            self.line(format!("{r} = fcmp {pred} {llt} {a}, {b}"));
        } else {
            let signed = matches!(opty, Ty::Int { signed: true, .. });
            let pred = match op {
                Operation::Eq => "eq",
                Operation::Neq => "ne",
                Operation::Lt => sign_pred(signed, "slt", "ult"),
                Operation::Gt => sign_pred(signed, "sgt", "ugt"),
                Operation::Le => sign_pred(signed, "sle", "ule"),
                Operation::Ge => sign_pred(signed, "sge", "uge"),
                _ => unreachable!(),
            };
            self.line(format!("{r} = icmp {pred} {llt} {a}, {b}"));
        }
        self.store_obj(target, "i1", &r);
    }

    /// A Named call (BL5 amendment, suggestions #8): top-level Array
    /// (components of the) argument go **by reference** — array parameters are
    /// observably read-only (Mapal value semantics; functional `Update` copies
    /// to a fresh alloca), so the address is observably the inline array, and
    /// no array bytes cross the call boundary. The lowering is per-signature
    /// (`lower_named_input_ty`), so every call site agrees with the callee's
    /// `FnEmit` by construction — host paths and body fns alike. Scalar-only
    /// arguments keep the by-value form byte-identical.
    pub(in crate::func) fn emit_call(&mut self, source: ObjectId, target: ObjectId, g: FuncId) {
        let callee = self.fnames[g].clone();
        let cfd = self.ir.func(g).expect("callee");
        let in_ty = self.obj_ty(cfd.input);
        let out_ty = self.obj_ty(cfd.output);
        let arg = match lower_named_input_ty(&in_ty) {
            None => String::new(),
            Some(arg_llt)
                if has_arrays(&in_ty) && !has_top_array(&in_ty) || nested_arrays(&in_ty) =>
            {
                // structure: arrays at any depth pass by address — the
                // argument assembled in scratch in the ABI's layout.
                let buf = self.scratch(&arg_llt);
                let (src, spaths) = self.value_ptr(source).expect("call arg storage");
                self.copy_layout(&in_ty, &buf, &array_paths(&in_ty), &src, &spaths);
                let whole = self.tmp();
                self.line(format!("{whole} = load {arg_llt}, ptr {buf}"));
                format!("{arg_llt} {whole}")
            }
            Some(_) if !has_top_array(&in_ty) => {
                let (llt, val) = self.load_whole(source).expect("call arg");
                format!("{llt} {val}")
            }
            Some(arg_llt) if arg_llt == "ptr" => {
                // A single surviving array argument: the bare source's address,
                // or the one array component's.
                let addr = if matches!(&in_ty, Ty::Array { .. }) {
                    self.array_operand_ptr(source, None)
                } else {
                    let k = (0u32..)
                        .find(|&k| matches!(in_ty.component_ty(k), Some(Ty::Array { .. })))
                        .expect("the surviving array component");
                    self.array_operand_ptr(source, Some(k))
                };
                format!("ptr {}", addr.expect("call array arg"))
            }
            Some(arg_llt) => {
                // The product argument, assembled in scratch: Array components
                // store their address, everything else its value — the
                // `body_call_arg` template with every component a "capture".
                self.body_call_arg(source, product_arity(&in_ty), &in_ty, &arg_llt, &[])
            }
        };
        if g != self.ir.entry() && out_by_ptr(&out_ty) {
            // structure: the callee writes its output into this storage.
            let llt = lower_ty(&out_ty).expect("an output of arrays lowers");
            let direct = self.slot_paths(target).is_empty();
            let dst = match self.slot(target) {
                Some(s) if direct => s,
                _ => self.scratch(&llt),
            };
            let sep = if arg.is_empty() { "" } else { ", " };
            self.line(format!("call void @{callee}({arg}{sep}ptr {dst})"));
            if !direct && let Some(t) = self.slot(target) {
                let tp = self.slot_paths(target);
                self.copy_layout(&out_ty, &t, &tp, &dst, &[]);
            }
            return;
        }
        match lower_ty(&out_ty) {
            None => self.line(format!("call void @{callee}({arg})")),
            Some(rty) => {
                let r = self.tmp();
                self.line(format!("{r} = call {rty} @{callee}({arg})"));
                self.store_obj(target, &rty, &r);
            }
        }
    }

    /// structure: a value of `ty` laid out by some of `layouts` holds array
    /// bytes (or the layouts differ) — it moves by `copy_layout`. A product
    /// every array of which is forwarded on every side is a small struct of
    /// pointers, moved whole as before.
    pub(in crate::func) fn holds_array_bytes(&self, ty: &Ty, layouts: &[Vec<Vec<u32>>]) -> bool {
        let all = array_paths(ty);
        !all.is_empty() && layouts.iter().any(|l| *l != all)
    }

    /// structure: the pointer positions of field `k` of `agg`'s slot (an
    /// array literal's elements are bytes).
    pub(in crate::func) fn field_paths(&self, agg: ObjectId, k: u32) -> Vec<Vec<u32>> {
        match self.obj_ty(agg) {
            Ty::Array { .. } => Vec::new(),
            _ => sub_paths(&self.slot_paths(agg), k),
        }
    }

    pub(in crate::func) fn emit_print(&mut self, source: ObjectId, newline: bool) {
        let nl = if newline { "true" } else { "false" };
        let pty = self
            .obj_ty(source)
            .component_ty(1)
            .cloned()
            .expect("print printable");
        if pty == Ty::Str {
            // Str comes only from a literal (I9s): the slot-1 Pair source is a
            // Str constant with a private global.
            let p_obj = self.pair_source(source, 1).expect("str print source");
            let g = self.strings.get(p_obj).expect("str global");
            let len = g.bytes.len();
            let name = g.name.clone();
            self.line(format!(
                "call void @mapal_print_str(ptr {name}, i64 {len}, i1 zeroext {nl})"
            ));
            return;
        }
        let (func, ze, tystr) = print_dispatch(&pty);
        let (_, val) = self.load_component(source, 1).expect("print value");
        // Param attr goes *after* the type in a call arg (`i8 zeroext %v`), like
        // the trailing newline `i1 zeroext` — attr-before-type is invalid LLVM.
        let ze = if ze { "zeroext " } else { "" };
        self.line(format!(
            "call void @{func}({tystr} {ze}{val}, i1 zeroext {nl})"
        ));
    }

    /// `time` (plan-time-builtin): the monotonic clock read. The source token
    /// is ordering-only and erases, and the `(IoToken, f64)` target residual-
    /// lowers to the bare `double` (ty.rs `lower_ty`: a one-component residual
    /// IS its component) — so the call result is the target object's value and
    /// no pair is materialized. Emission position in the block IS the ordering
    /// the token models.
    ///
    /// On a GPU profile the read first waits for the device (plan-s53
    /// `clock-fence`): a launch returns before its kernel ends, so without the
    /// sync the clock would close over work still running. `lib.rs` drops the
    /// call again from a module with no GPU site — nothing to wait for.
    pub(in crate::func) fn emit_time_ms(&mut self, target: ObjectId) {
        if self.profile.gpu().is_some() {
            self.line(gpu::GPU_SYNC);
        }
        let r = self.tmp();
        self.line(format!("{r} = call double @mapal_time_ms()"));
        self.store_obj(target, "double", &r);
    }

    /// `read` (ADR-0045): one call into the runtime, which resolves the path
    /// (the literal, `$MAPAL_DATA_DIR`, the recorded source directory) and
    /// reads the file's `n · |T|` bytes straight into the array's storage —
    /// the `(token, array)` pair itself has none (`prepare_storage`), so the
    /// bytes move once. An array nobody reads still has its file checked.
    ///
    /// In a task of the parallel entry (ADR-0045 D8) a failure is recorded
    /// at the read's topo position, like a task's trap
    /// (`mapal_read_task`), and delivered at the first checkpoint past it;
    /// on the spine and in a sequential function it exits at once.
    pub(in crate::func) fn emit_read(&mut self, m: MorphismId, source: ObjectId, target: ObjectId) {
        let p_obj = self.pair_source(source, 1).expect("read path source");
        let g = self.strings.get(p_obj).expect("read path global");
        let (path, plen) = (g.name.clone(), g.bytes.len());
        let dlen = self.ir.data_dir().map_or(0, str::len);
        let arr_ty = self
            .obj_ty(target)
            .component_ty(1)
            .cloned()
            .expect("read target is (token, array)");
        let bytes = mapal_ir::read::file_bytes(&arr_ty).expect("a readable array");
        let dst = match self.ir.read_values(target).first() {
            Some(&v) => self.slot(v).unwrap_or_else(|| "null".into()),
            None => "null".into(),
        };
        let dir = crate::module::DATA_DIR_GLOBAL;
        let args = format!("ptr {path}, i64 {plen}, ptr {dir}, i64 {dlen}, ptr {dst}, i64 {bytes}");
        if matches!(self.guard_flavor, GuardFlavor::Host) {
            self.line(format!("call void @mapal_read({args})"));
        } else {
            let topo = self.task_site(m);
            self.line(format!("call void @mapal_read_task({args}, i64 {topo})"));
            self.runtime_write = true;
        }
    }

    /// The source object of the `Pair{slot==k}` edge feeding aggregate `agg`.
    pub(in crate::func) fn pair_source(&self, agg: ObjectId, k: u32) -> Option<ObjectId> {
        for &m in self.ir.in_edges(agg) {
            let morph = self.ir.morphism(m).expect("morphism");
            if let Operation::Pair { slot, .. } = morph.op
                && slot == k
            {
                return Some(morph.source);
            }
        }
        None
    }

    /// `cell-law`'s device variant: the value standing for `source`'s array
    /// read — `%cv<s>` when the array is capture view `s` of the input and the
    /// slot arrives by value.
    pub(super) fn cell_val(&self, source: ObjectId) -> Option<String> {
        let arr = self.pair_source(source, 0)?;
        let &p = self.ir.in_edges(arr).first()?;
        let morph = self.ir.morphism(p)?;
        let Operation::Proj { index } = morph.op else {
            return None;
        };
        (morph.source == self.ir.func(self.f)?.input
            && self.cell_vals.iter().any(|&(s, _)| s == index))
        .then(|| format!("%cv{index}"))
    }

    /// Rule 4's in-place legality for the `Update` morphism `m` (plan-last-use
    /// §2; suggestions #2): the source array object, when the plan proves it
    /// `dead_after` this Update — every use ranked at/before it under rule 1's
    /// decide < `LoopExit` < advance < `LoopBack`, ¬escapes, ¬carried — so the
    /// whole-array memcpy may be skipped and the target may SHARE the source's
    /// slot (the element store lands in place; in the loop-carried matmul4
    /// shape the write goes straight into the merge's own storage and the
    /// back-edge copy becomes an identity). A ptr-resident (by-ref) source is
    /// never eligible — the store would land in caller memory (the plan's
    /// rule 2 already marks borrowed inputs escaping; this check is the
    /// emitted-text-level second line). `None` keeps the fresh-alloca copy.
    ///
    /// A piecewise map (generic-fusion natural forms, [`mapal_ir::Outside`]:
    /// outside an index interval each cell is its capture's own, which the
    /// body reads at that cell only) is the same write at a range: under the
    /// same legality it runs in place over the interval ([`Self::range_map`]).
    pub(in crate::func) fn in_place_source(&self, m: MorphismId) -> Option<ObjectId> {
        let morph = self.ir.morphism(m).expect("morphism resolves");
        let arr = match morph.op {
            Operation::Update => self.pair_source(morph.source, 0)?,
            Operation::Map { .. } => self.pair_source(morph.source, self.range_map(m)?.slot)?,
            _ => return None,
        };
        if self.ir.object(arr).expect("object resolves").kind == ObjectKind::Constant {
            return None;
        }
        let idx = self.lup.position(m)?;
        // S65 `by-ref`: a view's bytes are written where they live, when the
        // facts say no other view of them is read again (`frame.rs`).
        if !self.par_safe(arr, m) {
            return None;
        }
        if self.by_ref {
            return self.writes_in_place(m, arr, idx).then_some(arr);
        }
        if self.ptr_resident.contains_key(arr) {
            return None;
        }
        self.lup.dead_after(arr, idx).then_some(arr)
    }
}
