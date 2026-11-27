//! the op table — `emit_morphism` and the scalar operations it dispatches to
//!
//! Split out of the former single-file `func.rs` (S41); behaviour is byte-identical.

use super::*;

mod index;
mod misc;

impl<'a> FnEmit<'a> {
    /// Emit one morphism (DESIGN §2 op table). Called by the straight-line walk
    /// and by the loop driver for decide/advance cones. Under `resident` a
    /// producer may be built on the device instead, and its device copy is
    /// placed and freed around it (`func/resident.rs`).
    pub(crate) fn emit_morphism(&mut self, m: MorphismId) {
        if self.resident.is_none() {
            return self.emit_op(m);
        }
        // kernel-merging `horizontal`: a member's launch is captured; the
        // group launches after its last member, before that one's frees.
        let hz = self.resident.as_ref().and_then(|p| {
            let (g, i) = *p.hz_of.get(m)?;
            Some((g, i, i + 1 == p.hz[g].members.len()))
        });
        self.hz.cur = hz.map(|(g, i, _)| (g, i));
        if !self.resident_instead(m) {
            self.emit_op(m);
        }
        self.hz.cur = None;
        if let Some((g, _, true)) = hz {
            self.hz_launch(g);
        }
        self.resident_after(m);
    }

    fn emit_op(&mut self, m: MorphismId) {
        self.emit_checkpoint(m);
        let morph = self.ir.morphism(m).expect("morphism resolves");
        let op = morph.op;
        let source = morph.source;
        let target = morph.target;

        match op {
            Operation::Pair { slot, .. } => {
                if self.host_free(source) {
                    // S54 `arena`: the field of a device-only array is never
                    // read on the host (device sites take the device copy).
                    let (llt, dst) = self
                        .component_ptr(target, slot)
                        .expect("Pair array target ptr");
                    // `scalar-fold`: a device-only number's field is read by
                    // cells from its device buffer — nothing to store.
                    if !matches!(self.obj_ty(source), Ty::Array { .. }) {
                        return;
                    }
                    assert_eq!(llt, "ptr", "a device-only array rides a pointer-only field");
                    self.line(format!("store ptr null, ptr {dst}"));
                } else if matches!(self.obj_ty(source), Ty::Array { .. }) {
                    let sllt = lower_ty(&self.obj_ty(source)).expect("array lowers");
                    let src = self
                        .array_operand_ptr(source, None)
                        .expect("Pair array source ptr");
                    let (dllt, dst) = self
                        .component_ptr(target, slot)
                        .expect("Pair array target ptr");
                    if dllt == "ptr" {
                        self.line(format!("store ptr {src}, ptr {dst}"));
                    } else {
                        self.emit_memcpy(&dst, &src, &sllt);
                    }
                } else if self.holds_array_bytes(
                    &self.obj_ty(source),
                    &[self.slot_paths(source), self.field_paths(target, slot)],
                ) && let Some((src, spaths)) = self.value_ptr(source)
                    && let Some((_, dst)) = self.component_ptr(target, slot)
                {
                    // structure: a product holding array bytes moves field by
                    // field (`copy_layout`), never as one SSA aggregate.
                    let ty = self.obj_ty(source);
                    let dpaths = self.field_paths(target, slot);
                    self.copy_layout(&ty, &dst, &dpaths, &src, &spaths);
                } else if let Some(llt) = self.big_aggregate(source)
                    && let Some((cllt, dst)) = self.component_ptr(target, slot)
                {
                    debug_assert_eq!(cllt, llt, "a product field lowers as its type");
                    let src = self.slots[source].clone();
                    self.emit_memcpy(&dst, &src, &llt);
                } else if let Some((sllt, sval)) = self.load_whole(source)
                    && let Some((_, ptr)) = self.component_ptr(target, slot)
                {
                    self.line(format!("store {sllt} {sval}, ptr {ptr}"));
                }
            }
            Operation::Proj { .. } if self.ir.read_of(target).is_some() => {
                // ADR-0045: the read wrote the file's bytes into its first
                // array's storage; a second `Proj` of the same pair copies.
                let first = self.ir.read_values(source)[0];
                if first != target {
                    let src = self.slot(first).expect("read array slot");
                    let dst = self.slot(target).expect("read array slot");
                    let llt = lower_ty(&self.obj_ty(target)).expect("read array lowers");
                    self.emit_memcpy(&dst, &src, &llt);
                }
            }
            Operation::Proj { index } => {
                if self.host_free(target) {
                    // S65 `by-ref`: a device-carried view has no host bytes.
                } else if matches!(
                    self.obj_ty(source).component_ty(index),
                    Some(Ty::Array { .. })
                ) {
                    let src = self
                        .array_operand_ptr(source, Some(index))
                        .expect("Proj array source ptr");
                    let dst = self.slot(target).expect("Proj array target slot");
                    if self.ptr_resident.contains_key(target) {
                        self.line(format!("store ptr {src}, ptr {dst}"));
                    } else {
                        let llt = lower_ty(&self.obj_ty(target)).expect("Proj array lowers");
                        self.emit_memcpy(&dst, &src, &llt);
                    }
                } else if let Some(cty) = self.obj_ty(source).component_ty(index).cloned()
                    && self.holds_array_bytes(
                        &cty,
                        &[
                            sub_paths(&self.slot_paths(source), index),
                            self.slot_paths(target),
                        ],
                    )
                {
                    // structure: a product field of arrays (a struct
                    // parameter's, an element view's), field by field.
                    if let Some((src, spaths)) = self.component_value_ptr(source, index)
                        && let Some(dst) = self.slot(target)
                    {
                        let dpaths = self.slot_paths(target);
                        self.copy_layout(&cty, &dst, &dpaths, &src, &spaths);
                    }
                } else if let Some((cllt, val)) = self.load_component(source, index) {
                    self.store_obj(target, &cllt, &val);
                }
            }
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Div | Operation::Mod => {
                self.emit_arith(m, source, target, op);
            }
            Operation::Neg => {
                let (llt, val) = self.load_whole(source).expect("neg operand");
                let r = self.tmp();
                if is_float(&self.obj_ty(source)) {
                    self.line(format!("{r} = fneg {llt} {val}"));
                } else {
                    self.line(format!("{r} = sub {llt} 0, {val}"));
                }
                self.store_obj(target, &llt, &r);
            }
            Operation::Eq
            | Operation::Neq
            | Operation::Lt
            | Operation::Gt
            | Operation::Le
            | Operation::Ge => {
                self.emit_compare(source, target, op);
            }
            Operation::And | Operation::Or => {
                let (_, a) = self.load_component(source, 0).expect("logic a");
                let (_, b) = self.load_component(source, 1).expect("logic b");
                let iop = if op == Operation::And { "and" } else { "or" };
                let r = self.tmp();
                self.line(format!("{r} = {iop} i1 {a}, {b}"));
                self.store_obj(target, "i1", &r);
            }
            Operation::Not => {
                let (_, val) = self.load_whole(source).expect("not operand");
                let r = self.tmp();
                self.line(format!("{r} = xor i1 {val}, true"));
                self.store_obj(target, "i1", &r);
            }
            Operation::Math(f) => {
                // ADR-0042 / ADR-0046: one call; the module defines it per
                // side and face (`crate::math`).
                let (llt, val) = self.load_whole(source).expect("math operand");
                let bits = if llt == "double" { 64 } else { 32 };
                let r = self.tmp();
                self.line(format!(
                    "{r} = call {llt} {}({llt} {val})",
                    crate::math::symbol(f, bits)
                ));
                self.store_obj(target, &llt, &r);
            }
            Operation::Narrow => {
                // ADR-0043: one call; the module defines it per side
                // (`crate::narrow`).
                let Ty::Narrow(n) = self.obj_ty(target) else {
                    unreachable!("Narrow into a non-narrow ty passed validation")
                };
                let (_, val) = self.load_whole(source).expect("narrow operand");
                let t = crate::narrow::llvm_ty(n);
                let r = self.tmp();
                self.line(format!(
                    "{r} = call {t} {}(float {val})",
                    crate::narrow::symbol(n)
                ));
                self.store_obj(target, t, &r);
            }
            Operation::Widen => {
                let (sllt, val) = self.load_whole(source).expect("widen operand");
                let tllt = lower_ty(&self.obj_ty(target)).expect("widen target");
                // ADR-0043 slice 3: an fp8 element widens by the module's
                // `@mapal.widen_*` (exact; `crate::narrow`).
                if let Ty::Narrow(n) = self.obj_ty(source)
                    && n.bytes() == 1
                {
                    let (r, t) = (self.tmp(), self.tmp());
                    for l in crate::narrow::widen_lines(n, &sllt, &val, &tllt, &r, &t) {
                        self.line(l);
                    }
                    self.store_obj(target, &tllt, &r);
                    return;
                }
                let cvt = match (self.obj_ty(source), self.obj_ty(target)) {
                    (Ty::Int { bits: 32, .. }, Ty::Int { bits: 64, .. }) => "sext",
                    (Ty::Int { bits: 32, .. }, Ty::Float { .. }) => "sitofp",
                    (Ty::Float { bits: 32 }, Ty::Float { bits: 64 }) => "fpext",
                    // ADR-0043: exact.
                    (Ty::Narrow(_), Ty::Float { .. }) => "fpext",
                    _ => unreachable!("invalid Widen pair passed validation"),
                };
                let r = self.tmp();
                self.line(format!("{r} = {cvt} {sllt} {val} to {tllt}"));
                self.store_obj(target, &tllt, &r);
            }
            Operation::Phi => {
                if let Some(site) = self.gsites.get(m).cloned() {
                    // plan-s39: the condition picks the arm and only that
                    // arm's work runs. The cond's Pair edge (slot 2) is
                    // unconditional and already fired; each branch emits its
                    // arm's own-list, then lands the staged value in the
                    // target's slot.
                    let (_, c) = self.load_component(source, 2).expect("phi cond");
                    let bt = self.label();
                    let bf = self.label();
                    let bj = self.label();
                    self.line(format!("br i1 {c}, label %{bt}, label %{bf}"));
                    for (arm, blk, slot) in
                        [(&site.on_true, &bt, 0u32), (&site.on_false, &bf, 1u32)]
                    {
                        self.label_line(blk);
                        for &g in &arm.own {
                            let gm = self.ir.morphism(g).expect("morphism resolves");
                            if gm.op == Operation::LoopEnter {
                                // plan-s40: the handle stands for its whole
                                // loop unit — the driver CFG is emitted inside
                                // this branch.
                                crate::loops::emit_loop(self, gm.target);
                            } else {
                                self.emit_morphism(g);
                            }
                        }
                        if matches!(self.obj_ty(target), Ty::Array { .. }) {
                            let p = self
                                .array_operand_ptr(source, Some(slot))
                                .expect("phi arm array ptr");
                            let dst = self.slot(target).expect("phi array target slot");
                            if self.ptr_resident.contains_key(target) {
                                // S65 `by-ref`: the result is a view of the arm.
                                self.line(format!("store ptr {p}, ptr {dst}"));
                            } else {
                                let llt = lower_ty(&self.obj_ty(target)).expect("phi array lowers");
                                self.emit_memcpy(&dst, &p, &llt);
                            }
                        } else {
                            let (tllt, v) =
                                self.load_component(source, slot).expect("phi arm value");
                            self.store_obj(target, &tllt, &v);
                        }
                        self.line(format!("br label %{bj}"));
                    }
                    self.label_line(&bj);
                } else if matches!(self.obj_ty(target), Ty::Array { .. }) {
                    // Hand-built (non-builder) triple: strict select over both
                    // computed arms.
                    let t = self
                        .array_operand_ptr(source, Some(0))
                        .expect("phi then array ptr");
                    let e = self
                        .array_operand_ptr(source, Some(1))
                        .expect("phi else array ptr");
                    let (_, c) = self.load_component(source, 2).expect("phi cond");
                    let r = self.tmp();
                    self.line(format!("{r} = select i1 {c}, ptr {t}, ptr {e}"));
                    let dst = self.slot(target).expect("phi array target slot");
                    if self.ptr_resident.contains_key(target) {
                        self.line(format!("store ptr {r}, ptr {dst}"));
                    } else {
                        let llt = lower_ty(&self.obj_ty(target)).expect("phi array lowers");
                        self.emit_memcpy(&dst, &r, &llt);
                    }
                } else {
                    let (tllt, t) = self.load_component(source, 0).expect("phi then");
                    let (_, e) = self.load_component(source, 1).expect("phi else");
                    let (_, c) = self.load_component(source, 2).expect("phi cond");
                    let r = self.tmp();
                    self.line(format!("{r} = select i1 {c}, {tllt} {t}, {tllt} {e}"));
                    self.store_obj(target, &tllt, &r);
                }
            }
            Operation::Call(g) => self.emit_call(source, target, g),
            Operation::Map { body, captures } => {
                if !self.host_par_map(m) {
                    self.emit_map(m, source, target, body, captures)
                }
            }
            // `tile-post`: the epilogue variant takes the site's fold value.
            Operation::Fold { .. } if self.post_fold == Some(m) => {
                let llt = lower_ty(&self.obj_ty(target)).expect("a fold value lowers");
                self.store_obj(target, &llt, "%fv");
            }
            Operation::Fold { body, captures } => self.emit_fold(source, target, body, captures),
            Operation::Index => self.emit_index(m, source, target),
            Operation::Update => self.emit_update(m, source, target),
            // step 3b: an array every consumer rebuilds from its law is never
            // read, so the store loop that fills it is dead. Skipping it here
            // (rather than trusting DCE) also drops its `%Frame` field, which
            // DCE cannot do — the frame is one object shared across tasks.
            Operation::Zip | Operation::Enumerate | Operation::Iota | Operation::Fill
                if self.elided_arrays.contains_key(target) => {}
            Operation::Zip => self.emit_zip(source, target),
            Operation::Enumerate => self.emit_enumerate(source, target),
            Operation::Iota => self.emit_iota(source, target),
            Operation::Fill => self.emit_fill(source, target),
            Operation::Print { newline } => self.emit_print(source, newline),
            Operation::TimeMs => self.emit_time_ms(target),
            Operation::Read => self.emit_read(m, source, target),
            Operation::Output => {
                if matches!(self.obj_ty(source), Ty::Array { .. }) {
                    let src = self
                        .array_operand_ptr(source, None)
                        .expect("Output array source ptr");
                    let dst = self.slot(target).expect("Output array target slot");
                    let llt = lower_ty(&self.obj_ty(source)).expect("Output array lowers");
                    self.emit_memcpy(&dst, &src, &llt);
                } else if has_arrays(&self.obj_ty(source))
                    && let Some((src, spaths)) = self.value_ptr(source)
                    && let Some(dst) = self.slot(target)
                {
                    let ty = self.obj_ty(source);
                    self.copy_layout(&ty, &dst, &[], &src, &spaths);
                } else if let Some((llt, val)) = self.load_whole(source) {
                    self.store_obj(target, &llt, &val);
                }
            }
            Operation::LoopEnter | Operation::LoopBack | Operation::LoopExit => {
                unreachable!("loop ops are driver-owned")
            }
        }
        if self.watermark && self.local_trap_site(m) {
            self.emit_watermark(m);
        }
    }

    pub(super) fn emit_arith(
        &mut self,
        m: MorphismId,
        source: ObjectId,
        target: ObjectId,
        op: Operation,
    ) {
        let opty = self
            .obj_ty(source)
            .component_ty(0)
            .cloned()
            .expect("arith ty");
        let (llt, a) = self.load_component(source, 0).expect("arith a");
        let (_, b) = self.load_component(source, 1).expect("arith b");

        if is_float(&opty) {
            let fop = match op {
                Operation::Add => "fadd",
                Operation::Sub => "fsub",
                Operation::Mul => "fmul",
                Operation::Div => "fdiv",
                Operation::Mod => "frem",
                _ => unreachable!(),
            };
            // fold-algebra: the step algebra's pairs. Explicit faces stash a
            // pair's factors at the `Mul` and fuse them at its add.
            if self.explicit_fma() {
                if op == Operation::Mul && self.algebra().fused_mul.contains_key(m) {
                    self.fma_stash.insert(m, (a.clone(), b.clone(), self.block));
                }
                let pair = self.algebra().fma.get(m).copied();
                let stash = pair.and_then(|p| self.fma_stash.get(p.mul).cloned());
                if let (Some(p), Some((x, y, blk))) = (pair, stash)
                    && blk == self.block
                {
                    let (mut x, y) = (x, y);
                    let mut c = if p.slot == 0 { b.clone() } else { a.clone() };
                    if p.sub {
                        // `x·y − c` = fma(x, y, −c); `c − x·y` = fma(−x, y, c):
                        // negation is exact, one rounding either way.
                        let n = self.tmp();
                        let v = if p.slot == 0 { &c } else { &x };
                        self.line(format!("{n} = fneg {llt} {v}"));
                        if p.slot == 0 {
                            c = n;
                        } else {
                            x = n;
                        }
                    }
                    let bits = if llt == "double" { 64 } else { 32 };
                    let r = self.tmp();
                    self.line(format!(
                        "{r} = call {llt} @llvm.fma.f{bits}({llt} {x}, {llt} {y}, {llt} {c})"
                    ));
                    self.store_obj(target, &llt, &r);
                    return;
                }
            }
            let fl = match op {
                Operation::Add | Operation::Sub | Operation::Mul => self.step_flags(m),
                _ => "",
            };
            let r = self.tmp();
            self.line(format!("{r} = {fop}{fl} {llt} {a}, {b}"));
            self.store_obj(target, &llt, &r);
            return;
        }

        let signed = matches!(opty, Ty::Int { signed: true, .. });
        match op {
            Operation::Add | Operation::Sub | Operation::Mul => {
                let iop = match op {
                    Operation::Add => "add",
                    Operation::Sub => "sub",
                    Operation::Mul => "mul",
                    _ => unreachable!(),
                };
                let r = self.tmp();
                self.line(format!("{r} = {iop} {llt} {a}, {b}")); // no nsw/nuw (wraps, L1)
                self.store_obj(target, &llt, &r);
            }
            Operation::Div | Operation::Mod => {
                // #13 constant-divisor credit (S20): a literal non-zero divisor
                // makes the zero guard dead; a literal non-(−1) makes the
                // MIN/−1 guard dead (the oracle's behavior is identical — the
                // checks cannot fire).
                let dconst = const_int_operand(self.ir, source, 1);
                let zero_dead = matches!(dconst, Some(v) if v != 0);
                let min_dead = matches!(dconst, Some(v) if v != -1);
                if !zero_dead && !matches!(self.guard_flavor, GuardFlavor::Host) {
                    self.emit_task_div(m, target, op, &llt, &a, &b, signed, min_dead);
                    return;
                }
                if !zero_dead {
                    // Zero guard → mapal_trap(div_zero).
                    let z = self.tmp();
                    self.line(format!("{z} = icmp eq {llt} {b}, 0"));
                    self.trap_if(&z, 0);
                }

                if signed && !min_dead {
                    // MIN/-1 guard: Div ⇒ MIN, Mod ⇒ 0 (wrapping_div/rem parity).
                    let min = int_min(&llt);
                    let m1 = self.tmp();
                    self.line(format!("{m1} = icmp eq {llt} {b}, -1"));
                    let ismin = self.tmp();
                    self.line(format!("{ismin} = icmp eq {llt} {a}, {min}"));
                    let ov = self.tmp();
                    self.line(format!("{ov} = and i1 {m1}, {ismin}"));
                    let lov = self.label();
                    let lnorm = self.label();
                    let ldone = self.label();
                    self.line(format!("br i1 {ov}, label %{lov}, label %{lnorm}"));
                    self.label_line(&lov);
                    let ovval = if op == Operation::Div { min } else { "0" };
                    self.store_obj(target, &llt, ovval);
                    self.line(format!("br label %{ldone}"));
                    self.label_line(&lnorm);
                    let sop = if op == Operation::Div { "sdiv" } else { "srem" };
                    let r = self.tmp();
                    self.line(format!("{r} = {sop} {llt} {a}, {b}"));
                    self.store_obj(target, &llt, &r);
                    self.line(format!("br label %{ldone}"));
                    self.label_line(&ldone);
                } else {
                    let sop = match (signed, op) {
                        (true, Operation::Div) => "sdiv",
                        (true, Operation::Mod) => "srem",
                        (false, Operation::Div) => "udiv",
                        (false, Operation::Mod) => "urem",
                        _ => unreachable!(),
                    };
                    let r = self.tmp();
                    self.line(format!("{r} = {sop} {llt} {a}, {b}"));
                    self.store_obj(target, &llt, &r);
                }
            }
            _ => unreachable!(),
        }
    }
}
