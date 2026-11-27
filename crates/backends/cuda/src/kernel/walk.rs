//! DevEmit emission walk: the topological walk, loops, morphisms, arithmetic, compare and calls.

use super::*;

impl<'a> DevEmit<'a> {
    // --- the walk ----------------------------------------------------------

    /// Emit the twin: declarations, prologue, the topo walk, epilogue.
    pub(crate) fn emit(mut self) -> Result<String, EmitError> {
        let fd = self.ir.func(self.f).expect("func resolves");
        let in_ty = self.obj_ty(fd.input);
        let out_ty = self.obj_ty(fd.output);

        // Hoisted declarations (`o{ord}`): scalars/products as values,
        // produced arrays as per-thread locals, other arrays as handles.
        let mut ord = 0u32;
        let owned: Vec<(ObjectId, ObjectKind, Ty)> = self
            .ir
            .objects()
            .filter(|(id, _)| self.ir.try_owner(*id) == Some(self.f))
            .map(|(id, obj)| (id, obj.kind, obj.ty.clone()))
            .collect();
        for (id, kind, ty) in &owned {
            if *kind == ObjectKind::Constant {
                continue;
            }
            if lower_ty(ty).is_none() {
                ord += 1;
                continue;
            }
            // R-ONENAME: the parameter IS a variable — the input object's
            // slot is the literal `in` (no `o0 = in;` copy, no declaration).
            if *id == fd.input {
                self.slots.insert(*id, "in".to_string());
                ord += 1;
                continue;
            }
            // WP-B: Inline/Dissolved values own no local (never arrays —
            // the query keeps arrays Named; the guard is belt-and-braces).
            if !matches!(ty, Ty::Array { .. }) && self.expr_only(*id) {
                ord += 1;
                continue;
            }
            let name = format!("o{ord}");
            if let Ty::Array { .. } = ty {
                if self.in_place.contains_key(*id) {
                    // In-placed Update target: no produced local — the site's
                    // store lands in the SOURCE array's storage and the
                    // target's slot aliases it (inserted at the site).
                    ord += 1;
                    continue;
                }
                if self.produced.contains_key(*id) {
                    // F7: the per-thread local-array budget (the documented
                    // cell — over-budget is Unsupported, never improvised).
                    check_local_array_budget(
                        ty,
                        self.ir.object(*id).expect("object resolves").loc,
                    )?;
                    let base = flat_base_ct(ty).expect("array base lowers");
                    self.decls
                        .push_str(&format!("  {base} {name}[{}];\n", flat_count(ty)));
                } else {
                    let ct = lower_ty(ty).expect("array lowers");
                    self.decls.push_str(&format!("  {ct} {name};\n"));
                }
            } else {
                let ct = lower_ty(ty).expect("lowers");
                self.decls.push_str(&format!("  {ct} {name};\n"));
            }
            self.slots.insert(*id, name);
            ord += 1;
        }

        // No prologue copy: the input object's slot IS `in` (R-ONENAME).
        let _ = in_ty;

        // Driver-owned morphisms: everything in a loop plan's decide/advance
        // cones, plus anything incident to an SCC object — DESIGN §1's
        // walk-skip paragraph, the llvm rule (its func.rs:252–290) carried
        // verbatim, computed once. Plan membership is the precise rule: an
        // exit-only payload chain leaves the SCC but still belongs to the
        // decide cone; SCC incidence alone would re-emit it after the loop.
        let mut in_scc: SecondaryMap<ObjectId, ()> = SecondaryMap::new();
        let mut owned: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        for scc in self.ir.loop_structure(self.f) {
            for &mg in &scc.merges {
                if let Some(plan) = self.ir.loop_plan(self.f, mg) {
                    for &mo in plan.decide_order.iter().chain(plan.advance_order.iter()) {
                        owned.insert(mo, ());
                    }
                }
            }
            for o in scc.objects {
                in_scc.insert(o, ());
            }
        }

        for m in self.ir.topo_order(self.f) {
            let morph = self.ir.morphism(m).expect("morphism resolves");
            match morph.op {
                Operation::LoopEnter => self.emit_loop(morph.target)?,
                Operation::LoopBack | Operation::LoopExit => {}
                _ => {
                    if owned.contains_key(m)
                        || in_scc.contains_key(morph.source)
                        || in_scc.contains_key(morph.target)
                    {
                        continue; // driver-owned
                    }
                    self.emit_morphism(m)?;
                }
            }
        }

        // Epilogue: scalar returns by value; an array Return is copied into
        // the caller's `out` buffer (the out-param convention).
        match lower_ty(&out_ty) {
            Some(_) if !matches!(out_ty, Ty::Array { .. }) => {
                let os = self.slot(fd.output).expect("non-void return has a slot");
                self.line(1, format!("return {os};"));
            }
            Some(_) => {
                let os = self.slot(fd.output).expect("array return has a local");
                self.line(
                    1,
                    format!(
                        "for (unsigned long long j = 0; j < {}ULL; j++) {{",
                        flat_count(&out_ty)
                    ),
                );
                self.line(2, format!("out[j] = {os}[j];"));
                self.line(1, "}");
                self.line(1, "return;");
            }
            None => self.line(1, "return;"),
        }

        let sig = twin_signature(self.ir, self.f, self.fnames, self.caps);
        Ok(format!("{sig} {{\n{}{}}}\n", self.decls, self.body))
    }

    /// A canonical loop in the inline form (DESIGN §1's `loops` row, right
    /// column): the host driver's guard-first quartet (loops.rs) as
    /// per-thread sequential device code — decide cone → guard → advance
    /// cone → back edge. The merge is always a handle (never a produced
    /// local array), so init/back-edge copies are pointer swaps for carried
    /// arrays, exactly as on the host; the one cell-wise copy is an exit
    /// payload landing in a produced local array (the fn's array Return —
    /// the epilogue copies it to the caller's `out`).
    pub(super) fn emit_loop(&mut self, merge: ObjectId) -> Result<(), EmitError> {
        let plan = self
            .ir
            .loop_plan(self.f, merge)
            .expect("canonical loop (gated by emit's L3 capability check)");

        // Entry: init → merge local (pointer copy for a carried array).
        if let Some((_, val)) = self.load_whole(plan.init) {
            self.store_obj(plan.merge, &val);
        }

        self.line(1, "while (true) {");
        self.base += 1;
        // Decide/exit cone: guard cond + exit-route payload, every iteration.
        for &mo in &plan.decide_order {
            self.emit_morphism(mo)?;
        }
        let (_, cond) = self
            .component_expr(plan.exit_route, 1)
            .expect("loop guard cond");
        self.line(1, format!("if (!{cond}) {{ break; }}"));
        // Advance cone: the next-state, unreachable on the exit step.
        for &mo in &plan.advance_order {
            self.emit_morphism(mo)?;
        }
        // Back edge: one copy, sequenced last (the parallel assignment —
        // the route is a distinct local from the merge, so no temporaries).
        if let Some((_, val)) = self.component_expr(plan.back_route, 0) {
            self.store_obj(plan.merge, &val);
        }
        self.base -= 1;
        self.line(1, "}");

        // Exit: the payload materializes exactly once, here. A produced
        // (local-array) exit object takes the payload cell-wise; a handle
        // takes the pointer.
        for &ex in &plan.exits {
            let (route, tgt) = {
                let m = self.ir.morphism(ex).expect("exit morphism");
                (m.source, m.target)
            };
            if matches!(self.obj_ty(tgt), Ty::Array { .. }) && self.produced.contains_key(tgt) {
                if let Some((_, payload)) = self.component_expr(route, 0) {
                    let n = flat_count(&self.obj_ty(tgt));
                    let jv = self.tmp();
                    let tgt_slot = self.slot(tgt).expect("produced exit local");
                    self.line(
                        1,
                        format!("for (unsigned long long {jv} = 0; {jv} < {n}ULL; {jv}++) {{"),
                    );
                    self.line(2, format!("{tgt_slot}[{jv}] = {payload}[{jv}];"));
                    self.line(1, "}");
                }
            } else if let Some((_, val)) = self.component_expr(route, 0) {
                self.store_obj(tgt, &val);
            }
        }
        Ok(())
    }

    pub(super) fn emit_morphism(&mut self, m: MorphismId) -> Result<(), EmitError> {
        let morph = self.ir.morphism(m).expect("morphism resolves");
        let op = morph.op;
        let source = morph.source;
        let target = morph.target;

        match op {
            Operation::Pair { slot, .. } => {
                if matches!(self.obj_ty(target), Ty::Array { .. }) {
                    // Body-local array literal (§2 item 8): per-thread local
                    // initializer, emitted at the last Pair edge.
                    let seen = self.lit_seen.get(target).copied().unwrap_or(0) + 1;
                    self.lit_seen.insert(target, seen);
                    if seen == self.lit_total.get(target).copied().unwrap_or(0) {
                        self.emit_literal(target);
                    }
                } else if self.dissolved(target) {
                    // WP-B: a dissolved product never materializes — its
                    // consumers read the field sources through
                    // `component_expr`; the assembly text disappears.
                } else if let Some((_, sval)) = self.load_whole(source)
                    && let Some((_, lvalue)) = self.component_expr(target, slot)
                {
                    self.line(1, format!("{lvalue} = {sval};"));
                }
            }
            Operation::Proj { index } => {
                if let Some((_, val)) = self.component_expr(source, index) {
                    self.store_obj(target, &val);
                }
            }
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Div | Operation::Mod => {
                self.emit_arith(source, target, op);
            }
            Operation::Neg => {
                let (ct, val) = self.load_whole(source).expect("neg operand");
                if is_float(&self.obj_ty(source)) {
                    // `-({val})`: a folded negative constant would print
                    // `--1.5e0` — the decrement operator, ill-formed here.
                    self.store_obj(target, &format!("-({val})"));
                } else {
                    let uct = unsigned_twin(&self.obj_ty(source));
                    self.store_obj(target, &format!("({ct})(0 - ({uct}){val})"));
                }
            }
            Operation::Math(f) => {
                // ADR-0042 / ADR-0046: the reference (`module::math_helpers`).
                let (_, val) = self.load_whole(source).expect("math operand");
                let w = if matches!(self.obj_ty(source), Ty::Float { bits: 64 }) {
                    64
                } else {
                    32
                };
                self.store_obj(target, &format!("mapal_{}_f{w}({val})", f.name()));
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
                let (_, a) = self.component_expr(source, 0).expect("logic a");
                let (_, b) = self.component_expr(source, 1).expect("logic b");
                let sym = if op == Operation::And { "&" } else { "|" };
                self.store_obj(target, &format!("{a} {sym} {b}"));
            }
            Operation::Not => {
                let (_, val) = self.load_whole(source).expect("not operand");
                self.store_obj(target, &format!("!{val}"));
            }
            Operation::Widen => {
                let (_, val) = self.load_whole(source).expect("widen operand");
                let target_ct = lower_ty(&self.obj_ty(target)).expect("widen target");
                self.store_obj(target, &format!("({target_ct})({val})"));
            }
            Operation::Narrow => {
                // ADR-0043: one rounding to nearest even (`cvt.rn`).
                let (_, val) = self.load_whole(source).expect("narrow operand");
                let f = crate::module::narrow_fn(&self.obj_ty(target));
                self.store_obj(target, &format!("{f}({val})"));
            }
            Operation::Phi => {
                // plan-s39: strict select, and MEASURED UNREACHABLE. A guard
                // inside a map/fold body does not come here — bodies are
                // emitted as `__host__ __device__` fns by `func.rs`, which
                // gates them, so `map { x -> (x>100) -> {-true-> x/0; …} }`
                // already emits a real `if`/`else` on the device. This arm was
                // probed with a panic across every program in the repo (106
                // emissions: all examples + bench shapes + matmul, raw and
                // `--rewrite`) and the whole 163-test CUDA suite, and fired
                // ZERO times. Left strict rather than gated because there is no
                // program to verify a change against; if a shape ever reaches
                // here, gate it the way `func.rs` does.
                let (tct, t) = self.component_expr(source, 0).expect("phi then");
                let (_, e) = self.component_expr(source, 1).expect("phi else");
                let (_, c) = self.component_expr(source, 2).expect("phi cond");
                let tt = self.tmp();
                let te = self.tmp();
                self.line(1, format!("{tct} {tt} = {t};"));
                self.line(1, format!("{tct} {te} = {e};"));
                self.store_obj(target, &format!("{c} ? {tt} : {te}"));
            }
            Operation::Call(g) => self.emit_call(source, target, g),
            Operation::Iota => self.emit_iota(target),
            Operation::Fill => self.emit_fill(source, target),
            Operation::Map { body, captures } => self.emit_map(source, target, body, captures),
            Operation::Zip => self.emit_zip(source, target),
            Operation::Enumerate => self.emit_enumerate(source, target),
            Operation::Fold { body, captures } => self.emit_fold(source, target, body, captures),
            Operation::Index => self.emit_index(m, source, target),
            Operation::Update => self.emit_update(source, target),
            Operation::Print { .. } | Operation::TimeMs | Operation::Read => {
                unreachable!("token-free twins contain no Print/TimeMs/Read (E2)")
            }
            Operation::Output => self.emit_output(source, target),
            Operation::LoopEnter | Operation::LoopBack | Operation::LoopExit => {
                unreachable!("loop ops are driver-owned")
            }
        }
        Ok(())
    }

    pub(super) fn emit_arith(&mut self, source: ObjectId, target: ObjectId, op: Operation) {
        let opty = self
            .obj_ty(source)
            .component_ty(0)
            .cloned()
            .expect("arith ty");
        let (ct, a) = self.component_expr(source, 0).expect("arith a");
        let (_, b) = self.component_expr(source, 1).expect("arith b");

        if is_float(&opty) {
            let expr = match op {
                Operation::Add => format!("{a} + {b}"),
                Operation::Sub => format!("{a} - {b}"),
                Operation::Mul => format!("{a} * {b}"),
                Operation::Div => format!("{a} / {b}"),
                Operation::Mod if matches!(opty, Ty::Float { bits: 32 }) => {
                    format!("fmodf({a}, {b})")
                }
                Operation::Mod => format!("fmod({a}, {b})"),
                _ => unreachable!(),
            };
            self.store_obj(target, &expr);
            return;
        }

        let signed = matches!(opty, Ty::Int { signed: true, .. });
        match op {
            Operation::Add | Operation::Sub | Operation::Mul => {
                let sym = match op {
                    Operation::Add => "+",
                    Operation::Sub => "-",
                    Operation::Mul => "*",
                    _ => unreachable!(),
                };
                let uct = unsigned_twin(&opty);
                self.store_obj(target, &format!("({ct})(({uct}){a} {sym} ({uct}){b})"));
            }
            Operation::Div | Operation::Mod => {
                // #13: a literal non-zero constant divisor makes the zero
                // guard dead by construction (the oracle's behavior is
                // identical — the guard can never fire); a constant ≠ −1
                // makes the MIN/−1 value guard dead. A literal 0 keeps the
                // guard (it always fires, exactly as the oracle traps).
                let const_div = const_int_operand(self.ir, source, 1);
                if !matches!(const_div, Some(v) if v != 0) {
                    // §3: the device zero guard stores div_zero (kind+1 ⇒ 1u)
                    // and returns.
                    let ret = self.ret_default.clone();
                    self.line(1, format!("if ({b} == 0) {{ *trap = 1u; {ret} }}"));
                }
                if signed && !matches!(const_div, Some(v) if v != -1) {
                    // Guarded form: the target is Named by the query (a
                    // guarded producer), so the slot exists.
                    let slot = self.slot(target).expect("div/mod result slot");
                    let min = int_min(&ct);
                    let sym = if op == Operation::Div { "/" } else { "%" };
                    let ovval = if op == Operation::Div { min } else { "0" };
                    self.line(1, format!("if (({b} == -1) && ({a} == {min})) {{"));
                    self.line(2, format!("{slot} = {ovval};"));
                    self.line(1, "} else {");
                    self.line(2, format!("{slot} = {a} {sym} {b};"));
                    self.line(1, "}");
                } else {
                    let sym = if op == Operation::Div { "/" } else { "%" };
                    self.store_obj(target, &format!("{a} {sym} {b}"));
                }
            }
            _ => unreachable!(),
        }
    }

    pub(super) fn emit_compare(&mut self, source: ObjectId, target: ObjectId, op: Operation) {
        let (_, a) = self.component_expr(source, 0).expect("cmp a");
        let (_, b) = self.component_expr(source, 1).expect("cmp b");
        let sym = match op {
            Operation::Eq => "==",
            Operation::Neq => "!=",
            Operation::Lt => "<",
            Operation::Gt => ">",
            Operation::Le => "<=",
            Operation::Ge => ">=",
            _ => unreachable!(),
        };
        self.store_obj(target, &format!("{a} {sym} {b}"));
    }

    pub(super) fn emit_call(&mut self, source: ObjectId, target: ObjectId, g: FuncId) {
        let out_ty = self.obj_ty(self.ir.func(g).expect("callee resolves").output);
        let arg = self.load_whole(source).map(|(_, val)| val);
        let name = self.quals.device_name(self.fnames, g);
        let mut args: Vec<String> = Vec::new();
        if let Some(a) = arg {
            args.push(a);
        }
        // Array return ⇒ the callee is a Twin (the qualifier rule) and the
        // result is written into the caller's local via the out-param. An
        // erased array return has no buffer (F6): the callee's device form
        // is plain void and the call stores nothing.
        let array_ret = matches!(out_ty, Ty::Array { .. }) && lower_ty(&out_ty).is_some();
        if array_ret {
            let dest = self.slot(target).expect("array call result local");
            args.push(dest);
        }
        // #14: the trap argument and the post-call check ride only when the
        // callee can trap — a trap-free callee leaves the flag untouched, so
        // both are dead by construction (first-trap-wins is preserved by the
        // checks kept after capable calls, in order).
        let capable = self.caps.get(g);
        if capable {
            args.push("trap".into());
        }
        let call = format!("{name}({})", args.join(", "));
        let ret = self.ret_default.clone();
        if array_ret {
            self.line(1, format!("{call};"));
        } else if lower_ty(&out_ty).is_some() {
            let slot = self.slot(target).expect("call result slot");
            self.line(1, format!("{slot} = {call};"));
        } else {
            self.line(1, format!("{call};"));
        }
        if capable {
            self.line(1, format!("if (*trap) {ret}"));
        }
    }
}
