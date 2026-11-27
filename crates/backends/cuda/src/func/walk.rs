//! FnEmit emission walk: the topological walk, morphisms, arithmetic, calls, prints and literals.

use super::*;

impl<'a> FnEmit<'a> {
    // --- the walk ---------------------------------------------------------

    /// Emit the function: declarations, prologue, the topo walk, epilogue.
    pub fn emit(mut self) -> Result<String, EmitError> {
        let fd = self.ir.func(self.f).expect("func resolves");
        let in_ty = self.obj_ty(fd.input);
        let ret_ty = self.obj_ty(fd.output);

        // The arena plan (suggestions.md #18, plan-smart-arenas v1.0) —
        // deduced from the sealed graph, never stored; over ARENA_MAX_BYTES
        // is rule 4's compile-time Unsupported (the F7 precedent). Always
        // `None` for a HostDevice fn (the qualifier rule keeps launch-form
        // ops out of it), so no cudaMalloc ever reaches a __device__ pass.
        self.arena = arena::arena_plan(self.ir, self.f)?;

        // Hoisted declarations: one local per materialized (non-constant,
        // non-erased) object, `o{ord}` in deterministic object order — the
        // llvm ordinal scheme: constants consume no ordinal, erased
        // non-constants do. Device-handle locals (`T*`) are
        // `nullptr`-initialized: a loop-cone allocation site may not execute
        // (zero-iteration loop), and the free path must never read an
        // indeterminate handle.
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
            // R-ONENAME (WP-C): the parameter IS a variable — alias `in`,
            // no declaration, no prologue copy.
            if *id == fd.input && lower_ty(ty).is_some() {
                self.slots.insert(*id, "in".to_string());
                ord += 1;
                continue;
            }
            // Inline/Dissolved values own no local (arrays stay Named).
            if !matches!(ty, Ty::Array { .. }) && lower_ty(ty).is_some() && self.expr_only(*id) {
                ord += 1;
                continue;
            }
            if let Some(ct) = lower_ty(ty) {
                let name = format!("o{ord}");
                self.slots.insert(*id, name.clone());
                if ct.ends_with('*') {
                    self.decls.push_str(&format!("  {ct} {name} = nullptr;\n"));
                } else {
                    self.decls.push_str(&format!("  {ct} {name};\n"));
                }
            }
            ord += 1;
        }

        // The zone's base handle (plan §3): one arena per fn, malloc'd once
        // at fn entry; members get their `arena0 + OFF` pointer inits at the
        // points where today `malloc_buffer` runs.
        let arena_cap = self.arena.as_ref().map(|p| p.capacity);
        if arena_cap.is_some() {
            self.decls.push_str("  char* arena0 = nullptr;\n");
        }

        // #19a (perf_timing): one event pair per launch site (collect_sites
        // order — deterministic), created once per fn invocation; the
        // accumulators ride the fn as plain locals. Empty when off, so the
        // default text is byte-identical.
        if self.perf {
            for (i, site) in kernel::collect_sites(self.ir, self.f)
                .into_iter()
                .enumerate()
            {
                self.ev_ord.insert(site.m, i);
            }
        }
        let ev_count = self.ev_ord.len();
        if ev_count > 0 {
            for i in 0..ev_count {
                self.decls
                    .push_str(&format!("  cudaEvent_t fev{i}_start, fev{i}_stop;\n"));
            }
            self.decls
                .push_str("  float mapal_perf_total = 0.0f;\n  float mapal_perf_ms = 0.0f;\n");
        }

        // Prologue: the arena malloc, the event creates, then the parameter
        // moves into the input object's local.
        if let Some(cap) = arena_cap {
            self.line(format!(
                "cu_check(cudaMalloc((void**)&arena0, {cap}ULL), \"cudaMalloc(arena0)\");"
            ));
        }
        for i in 0..ev_count {
            self.line(format!(
                "cu_check(cudaEventCreate(&fev{i}_start), \"cudaEventCreate\");"
            ));
            self.line(format!(
                "cu_check(cudaEventCreate(&fev{i}_stop), \"cudaEventCreate\");"
            ));
        }
        // No `{o0} = in;` prologue copy — the input aliases `in` (WP-C).
        let _ = &in_ty;

        // The topo walk (DESIGN §1: one host statement per morphism; the
        // driver-ownership skip routes loop bodies to loops.rs's quartet).
        self.walk()?;

        // Epilogue: the MAPAL_PERF total + event destroys (#19a), then free
        // this fn's device buffers (value-guarded for an array return — see
        // emit_frees), then return the output local.
        if ev_count > 0 {
            self.line("printf(\"MAPAL_PERF total ms=%.4f\\n\", mapal_perf_total);");
            for i in 0..ev_count {
                self.line(format!(
                    "cu_check(cudaEventDestroy(fev{i}_start), \"cudaEventDestroy\");"
                ));
                self.line(format!(
                    "cu_check(cudaEventDestroy(fev{i}_stop), \"cudaEventDestroy\");"
                ));
            }
        }
        self.emit_frees();
        match lower_ty(&ret_ty) {
            Some(_) => {
                let os = self.slot(fd.output).expect("non-void return has a slot");
                self.line(format!("return {os};"));
            }
            None => {
                self.line("return;");
            }
        }

        let sig = fn_signature(self.ir, self.f, self.fnames, self.qual, self.caps);
        Ok(format!("{sig} {{\n{}{}}}\n", self.decls, self.body))
    }

    pub(super) fn walk(&mut self) -> Result<(), EmitError> {
        // Driver-owned morphisms: everything in a loop plan's decide/advance
        // cones, plus anything incident to an SCC object — DESIGN §1's
        // walk-skip paragraph, the llvm rule (its func.rs:252–290, the skip
        // itself :280–285) carried verbatim. Plan membership is the precise
        // rule: an exit-only payload chain (a computed exit value, an
        // exit-arm Print) leaves the SCC but still belongs to the decide
        // cone; skipping by SCC incidence alone would re-emit it after the
        // loop (dead recompute for values, a DOUBLE side effect for Print).
        // The loop driver emits the decide/advance cones; they emit nowhere
        // else.
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
                Operation::LoopEnter => crate::loops::emit_loop(self, morph.target)?,
                Operation::LoopBack | Operation::LoopExit => {}
                _ => {
                    if owned.contains_key(m)
                        || in_scc.contains_key(morph.source)
                        || in_scc.contains_key(morph.target)
                    {
                        continue; // driver-owned
                    }
                    if self.gated.contains_key(m) {
                        continue; // plan-s39: emitted inside its Phi's branch
                    }
                    self.emit_morphism(m)?;
                }
            }
        }
        Ok(())
    }

    /// Emit one morphism (DESIGN §1 op table, host column). Called by the
    /// straight-line walk and by the loop driver for the decide/advance
    /// cones.
    pub(crate) fn emit_morphism(&mut self, m: MorphismId) -> Result<(), EmitError> {
        let morph = self.ir.morphism(m).expect("morphism resolves");
        let op = morph.op;
        let source = morph.source;
        let target = morph.target;
        let loc = morph.loc;

        match op {
            Operation::Pair { slot, .. } => {
                if matches!(self.obj_ty(target), Ty::Array { .. }) {
                    // Array construction (pack_array literal): emitted once,
                    // at the target's last Pair edge in topo order (§2, BC11).
                    let seen = self.lit_seen.get(target).copied().unwrap_or(0) + 1;
                    self.lit_seen.insert(target, seen);
                    if seen == self.lit_total.get(target).copied().unwrap_or(0) {
                        self.emit_literal(target);
                    }
                } else if self.dissolved(target) {
                    // WP-C: dissolved wrapper — no materialization; consumers
                    // read the field sources through `component_expr`.
                } else if let Some((_, sval)) = self.load_whole(source)
                    && let Some((_, lvalue)) = self.component_expr(target, slot)
                {
                    self.line(format!("{lvalue} = {sval};"));
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
                    // BC2: wrapping_neg via the unsigned twin (INT_MIN neg is UB).
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
                // BC7: `&` / `|` on precomputed operands — never the
                // short-circuiting `&&` / `||`.
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
                if let Some(site) = self.gsites.get(m).cloned() {
                    // plan-s39: the condition picks the arm and only that
                    // arm's work runs. Each arm's own morphisms are emitted
                    // INSIDE its branch, so a `Div` guard, a launch, or a
                    // readback in the unchosen arm never executes. The result
                    // is `force_named`, so the in-branch assignment survives.
                    let (_, c) = self.component_expr(source, 2).expect("phi cond");
                    self.line(format!("if ({c}) {{"));
                    for (arm, slot) in [(&site.on_true, 0u32), (&site.on_false, 1u32)] {
                        self.indent += 1;
                        for &g in &arm.own {
                            self.emit_morphism(g)?;
                        }
                        let (_, v) = self.component_expr(source, slot).expect("phi arm");
                        self.store_obj(target, &v);
                        self.indent -= 1;
                        if slot == 0 {
                            self.line("} else {");
                        }
                    }
                    self.line("}");
                } else {
                    // Hand-built (non-builder) triple: strict select over both
                    // computed arms. The temps pin the llvm `select` shape; a
                    // ternary over the arm *expressions* would skip the untaken
                    // arm's trap, which for an ungated site is not the meaning.
                    let (tct, t) = self.component_expr(source, 0).expect("phi then");
                    let (_, e) = self.component_expr(source, 1).expect("phi else");
                    let (_, c) = self.component_expr(source, 2).expect("phi cond");
                    let tt = self.tmp();
                    let te = self.tmp();
                    self.line(format!("{tct} {tt} = {t};"));
                    self.line(format!("{tct} {te} = {e};"));
                    self.store_obj(target, &format!("{c} ? {tt} : {te}"));
                }
            }
            Operation::Call(g) => self.emit_call(source, target, g),
            Operation::Print { newline } => self.emit_print(source, newline),
            // plan-time-builtin: the clock read has no CUDA seam yet (the llvm
            // backend owns `mapal_time_ms`). The ✋ cell, not a silent host-clock
            // substitute — a device-side timing story is its own design.
            Operation::TimeMs => {
                return Err(EmitError::Unsupported {
                    feature: "the `time` builtin (no CUDA clock seam)".into(),
                    loc,
                });
            }
            // ADR-0045: the file read is the llvm backend's (`mapal_read` in
            // mapal-rt, then the residency machinery's upload); this
            // backend's arrays are device handles with no host read seam.
            Operation::Read => {
                return Err(EmitError::Unsupported {
                    feature: "the `read` builtin (no CUDA read seam)".into(),
                    loc,
                });
            }
            Operation::Output => {
                if let Some((_, val)) = self.load_whole(source) {
                    self.store_obj(target, &val);
                }
                // Allocation-registry escape (DESIGN §2): a returned buffer's
                // free duty transfers to the caller. Inert in WP2.
                if let Some(s) = self.slot(source) {
                    self.remove_alloc(&s);
                }
            }
            Operation::Map { .. }
            | Operation::Zip
            | Operation::Enumerate
            | Operation::Iota
            | Operation::Fill
            | Operation::Fold { .. }
            | Operation::Index
            | Operation::Update => {
                let kname = self.sites.get(&m).expect("bulk site registered").clone();
                let capable = self.caps.site(self.ir, m);
                self.emit_bulk_site(m, &kname, source, target, op, capable);
            }
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
                // IEEE at width; ÷0 is ±inf/NaN, no trap (ADR-0013 S13).
                Operation::Div => format!("{a} / {b}"),
                // fmod — llvm's open `frem` parity question transfers (§4b).
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
                // BC2: unsigned-cast wrapping — C++ signed overflow is UB.
                let uct = unsigned_twin(&opty);
                self.store_obj(target, &format!("({ct})(({uct}){a} {sym} ({uct}){b})"));
            }
            Operation::Div | Operation::Mod => {
                // #13 (kernel.rs's const_int_operand): a literal non-zero
                // constant divisor makes the zero guard dead by construction
                // (oracle behavior identical — it can never fire); a
                // constant ≠ −1 makes the MIN/−1 value guard dead. A literal
                // 0 keeps the guard (it always fires, as the oracle traps).
                let const_div = kernel::const_int_operand(self.ir, source, 1);
                if !matches!(const_div, Some(v) if v != 0) {
                    if self.qual == FnQual::HostDevice {
                        // BC8 (iv): one definition compiled for both sites. The
                        // host pass traps directly; the device pass stores
                        // div_zero (kind+1 ⇒ 1u) in the flag (the `d_trap`
                        // parameter shadows the global — see fn_signature) and
                        // returns (§3).
                        let ret = self.ret_default();
                        self.line(format!("if ({b} == 0) {{"));
                        self.line4("#ifdef __CUDA_ARCH__");
                        self.line4(format!("*d_trap = 1u; {ret}"));
                        self.line4("#else");
                        self.line4("mapal_trap(0);");
                        self.line4("#endif");
                        self.line("}");
                    } else {
                        // Zero guard → mapal_trap(div_zero) on the host (kind 0).
                        self.line(format!("if ({b} == 0) {{ mapal_trap(0); }}"));
                    }
                }
                if signed && !matches!(const_div, Some(v) if v != -1) {
                    // MIN/-1 → defined result (wrapping_div/rem parity):
                    // Div ⇒ MIN, Mod ⇒ 0 — a value guard, NOT a trap. The
                    // unguarded C++ `INT_MIN / -1` would be UB. Guarded ⇒
                    // the target is Named by the query — the slot exists.
                    let slot = self.slot(target).expect("div/mod result slot");
                    let min = int_min(&ct);
                    let sym = if op == Operation::Div { "/" } else { "%" };
                    let ovval = if op == Operation::Div { min } else { "0" };
                    self.line(format!("if (({b} == -1) && ({a} == {min})) {{"));
                    self.line4(format!("{slot} = {ovval};"));
                    self.line("} else {");
                    self.line4(format!("{slot} = {a} {sym} {b};"));
                    self.line("}");
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
        // One C++ operator per op: signedness rides the operand types; float
        // `==`/`!=`/`<`… match llvm's oeq/une/olt/… on unordered operands.
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
        let callee = self.fnames[g].clone();
        let out_ty = self.obj_ty(self.ir.func(g).expect("callee").output);
        let mut arg = match self.load_whole(source) {
            Some((_, val)) => val,
            None => String::new(),
        };
        // BC8 (iv): a `__host__ __device__` callee that CAN TRAP (#14)
        // takes the threaded trap pointer; host callers pass the host global
        // (unused on the host pass — its guards call mapal_trap directly,
        // §3). A trap-free callee's signature has no trap parameter, so the
        // call passes nothing.
        if self.quals.get(g) == FnQual::HostDevice && self.caps.get(g) {
            arg = if arg.is_empty() {
                "d_trap".to_string()
            } else {
                format!("{arg}, d_trap")
            };
        }
        match lower_ty(&out_ty) {
            None => self.line(format!("{callee}({arg});")),
            Some(_) => self.store_obj(target, &format!("{callee}({arg})")),
        }
    }

    pub(super) fn emit_print(&mut self, source: ObjectId, newline: bool) {
        let nl = if newline { "true" } else { "false" };
        let pty = self
            .obj_ty(source)
            .component_ty(1)
            .cloned()
            .expect("print printable");
        if pty == Ty::Str {
            // Str comes only from a literal (I9s): the slot-1 Pair source is
            // a Str constant with a private global; explicit length (the C
            // array's NUL terminator is not part of the payload).
            let p_obj = self.pair_source(source, 1).expect("str print source");
            let g = self.strings.get(p_obj).expect("str global");
            let len = g.bytes.len();
            let name = g.name.clone();
            self.line(format!(
                "mapal_print_str((const uint8_t*){name}, {len}, {nl});"
            ));
            return;
        }
        let func = print_dispatch(&pty).expect("printable Print operand");
        let (_, val) = self.component_expr(source, 1).expect("print value");
        self.line(format!("{func}({val}, {nl});"));
    }

    /// The source object of the `Pair{slot==k}` edge feeding aggregate `agg`.
    pub(super) fn pair_source(&self, agg: ObjectId, k: u32) -> Option<ObjectId> {
        kernel::pair_source(self.ir, agg, k)
    }

    /// The guard early-return for this fn's return type (`return {ret}{};` or
    /// `return;`) — used by the `__host__ __device__` Div/Mod guard's device
    /// branch. HostDevice fns never return arrays (the qualifier rule), so
    /// `{ct}{}` is always a scalar/product value-init here.
    pub(super) fn ret_default(&self) -> String {
        let fd = self.ir.func(self.f).expect("func resolves");
        let out_ty = &self.ir.object(fd.output).expect("output resolves").ty;
        match lower_ty(out_ty) {
            Some(ct) => format!("return {ct}{{}};"),
            None => "return;".to_string(),
        }
    }

    /// An array construction site (§2, BC11): a host data array + one H→D
    /// `cudaMemcpy`, per execution of the construction site. All-constant
    /// elements use a `static const` data array; computed elements a plain
    /// local (same one-memcpy shape); nested elements ride per-element
    /// device-to-device copies from the sub-array handles. The device buffer
    /// itself comes from [`FnEmit::alloc_buffer`] (a fn-zone pointer init,
    /// or a per-buffer cudaMalloc for a cone site).
    pub(super) fn emit_literal(&mut self, target: ObjectId) {
        let Some(slot) = self.slot(target) else {
            return; // erased element type: no representation
        };
        let arr_ty = self.obj_ty(target);
        let ptr_ct = lower_ty(&arr_ty).expect("literal array lowers");
        let (elem, n) = kernel::array_parts(&arr_ty);

        if matches!(elem, Ty::Array { .. }) {
            self.alloc_buffer(
                ArenaKey::Obj(target),
                &slot,
                &ptr_ct,
                &kernel::buffer_bytes(&arr_ty),
            );
            let m = kernel::flat_count(&elem);
            let base = kernel::flat_base_ct(&arr_ty).expect("literal base lowers");
            for k in 0..n {
                let src = self.pair_source(target, k as u32).expect("literal element");
                if let Some((_, expr)) = self.load_whole(src) {
                    self.line(format!(
                        "cu_check(cudaMemcpy({slot} + {}ULL, {expr}, sizeof({base}) * {m}ULL, cudaMemcpyDeviceToDevice), \"cudaMemcpy(literal)\");",
                        k * m
                    ));
                }
            }
            return;
        }

        let mut elems: Vec<String> = Vec::new();
        let mut all_const = true;
        for k in 0..n {
            let src = self.pair_source(target, k as u32).expect("literal element");
            let is_const = matches!(
                self.ir.object(src).expect("element resolves").kind,
                ObjectKind::Constant
            );
            let (_, expr) = self.load_whole(src).expect("literal element operand");
            elems.push(expr);
            all_const &= is_const;
        }
        let ct = lower_ty(&elem).expect("literal element lowers");
        let data = format!("lit{}", self.fresh());
        let init = elems.join(", ");
        if all_const {
            self.line(format!("static const {ct} {data}[{n}] = {{ {init} }};"));
        } else {
            self.line(format!("{ct} {data}[{n}] = {{ {init} }};"));
        }
        self.alloc_buffer(
            ArenaKey::Obj(target),
            &slot,
            &ptr_ct,
            &format!("sizeof({data})"),
        );
        self.line(format!(
            "cu_check(cudaMemcpy({slot}, {data}, sizeof({data}), cudaMemcpyHostToDevice), \"cudaMemcpy(literal)\");"
        ));
    }
}
