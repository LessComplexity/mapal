//! Emitter methods: loop exit arms, jump arms and slot exits.

use super::*;

impl Emitter<'_> {
    /// Lower the routing-guard chain's scrutinee (head + stages before the
    /// final Guard stage). Returns the cond object.
    pub(super) fn lower_scrutinee(&mut self, fb: &mut FnBuilder, chain: &Chain) -> ER<ObjectId> {
        let mut cur: Option<ObjectId> = match &chain.head {
            Some(h) => Some(self.emit_expr(fb, h)?),
            None => None,
        };
        let n = chain.stages.len();
        for (i, stage) in chain.stages.iter().enumerate() {
            let is_guard = i + 1 == n && matches!(stage.kind, StageKind::Guard(_));
            if is_guard {
                break;
            }
            let next = chain.stages.get(i + 1);
            cur = self.emit_stage(fb, stage, next, false, cur, &ChainCtx::Statement)?;
        }
        // A headless routing guard (`-> { -true-> … -false-> … }`) has no value
        // to route on — a user error, not a lower bug (ATK-13): report L1301
        // HeadlessChain, keeping L1901 unreachable from parse-clean input.
        cur.ok_or_else(|| {
            diag(
                LCode::HeadlessChain,
                chain.span,
                "routing guard has no scrutinee",
            )
        })
    }

    /// Emit a jump-arm payload's state updates (arm-local rebinds). The trailing
    /// `-> loop;` marker carries no value and is skipped (the back edge is wired
    /// by the routing-guard recipe).
    pub(super) fn emit_jump_arm_updates(
        &mut self,
        fb: &mut FnBuilder,
        payload: &ArmPayload,
    ) -> ER<()> {
        match payload {
            ArmPayload::Chain(c) => self.emit_jump_chain(fb, c),
            ArmPayload::Block(b) => {
                for item in &b.items {
                    if let BlockItem::Stmt(s) = item {
                        if let StmtKind::Chain(c) = &s.kind {
                            self.emit_jump_chain(fb, c)?;
                        } else {
                            self.emit_stmt(fb, s)?;
                        }
                    }
                }
                if let Some(t) = &b.tail {
                    self.emit_jump_chain(fb, t)?;
                }
                Ok(())
            }
        }
    }

    /// A jump-arm chain is the bare `-> loop` marker (after the state updates,
    /// which are separate statements). The chain itself is just `-> loop;`.
    pub(super) fn emit_jump_chain(&mut self, fb: &mut FnBuilder, chain: &Chain) -> ER<()> {
        let _ = fb;
        // The chain is the `-> loop` marker (headless, single LoopJump stage) —
        // carries no value; nothing to emit (the back edge is wired by the
        // routing-guard recipe).
        if chain.head.is_none()
            && chain.stages.len() == 1
            && matches!(chain.stages[0].kind, StageKind::LoopJump)
        {
            return Ok(());
        }
        // Otherwise it is a normal statement chain (a state update written as the
        // block tail). Emit it.
        self.emit_chain(fb, chain, ChainCtx::Statement)?;
        Ok(())
    }

    /// Emit an exit-arm payload against the snapshot (§8.5). `scrutinee` is the
    /// guard's incoming value (the cond) — a headless exit-arm chain that is not
    /// a bare `-> ret` marker seeds `cur := scrutinee` (§8.3 / ATK-19).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_exit_arm(
        &mut self,
        fb: &mut FnBuilder,
        payload: &ArmPayload,
        cond_exit: ObjectId,
        scrutinee: ObjectId,
        lh: &mapal_ir::LoopHandle,
        token_carried: bool,
        guard_sp: syn::SourceLoc,
    ) -> ER<()> {
        // L1504: an inner-loop exit must terminate in `-> ret` (the only nested
        // shape Core admits). `loop_depth >= 2` means we are emitting a nested
        // loop's routing guard.
        if self.loop_depth >= 2 && !exit_payload_is_ret(payload) {
            return Err(diag(
                LCode::NestedLoopShape,
                guard_sp,
                "an inner-loop exit must terminate in `-> ret`",
            ));
        }
        if self.defer_exit_arm(fb, payload, cond_exit, lh, token_carried, guard_sp)? {
            return Ok(());
        }
        // Determine the terminal: `-> ret`, a binding, or a bare value.
        let token_before = self.token;
        let (value, terminal) = self.exit_arm_value(fb, payload, scrutinee, guard_sp)?;
        // An arm not deferred past the loop feeds the exit route, which a loop
        // decides every trip (ADR-0016): an effect there would run every trip,
        // where the program runs it once.
        if self.token != token_before {
            return Err(diag(
                LCode::GuardArmEffectful,
                guard_sp,
                "an effect in a loop exit arm that cannot run after its loop",
            ));
        }
        let tok = if token_carried { self.token } else { None };
        match terminal {
            ExitTerminal::Ret => {
                // A value-less `-> ret` exit with no token can only be a pure
                // **Unit**-output fn here (D1's L1306 already rejected value-less
                // `-> ret` for non-Unit outputs). §8.5/LP-1: the payload is the
                // merge-state view itself (the merge object — any |U|), exiting to
                // `Dest::Fresh(None)` with zero Return writers (legal for Unit).
                if value.is_none() && tok.is_none() {
                    let merge = fb.merge_of(lh);
                    fb.loop_exit(lh, merge, cond_exit, Dest::Fresh(None), ir_loc(guard_sp))
                        .map_err(|e| ir_err(e, guard_sp))?;
                    return Ok(());
                }
                let payload_obj = match (value, tok) {
                    (Some(v), Some(t)) => self
                        .pack(fb, &[t, v], Dest::Fresh(None), guard_sp)
                        .map_err(|e| ir_err(e, guard_sp))?,
                    (None, Some(t)) => t, // value-less ret carrying the token.
                    (Some(v), None) => v,
                    (None, None) => unreachable!("handled above"),
                };
                fb.loop_exit(
                    lh,
                    payload_obj,
                    cond_exit,
                    Dest::Ret { slot: None },
                    ir_loc(guard_sp),
                )
                .map_err(|e| ir_err(e, guard_sp))?;
                // After a ret-terminal exit, the token register is consumed.
                if token_carried {
                    self.token = None;
                }
            }
            ExitTerminal::Bind(name, mutable) => {
                let v = value.ok_or_else(|| {
                    diag(LCode::FanoutNoValue, guard_sp, "exit binding with no value")
                })?;
                let payload_obj = match tok {
                    Some(t) => self
                        .pack(fb, &[t, v], Dest::Fresh(None), guard_sp)
                        .map_err(|e| ir_err(e, guard_sp))?,
                    None => v,
                };
                let exit = fb
                    .loop_exit(
                        lh,
                        payload_obj,
                        cond_exit,
                        Dest::Fresh(Some(name.clone())),
                        ir_loc(guard_sp),
                    )
                    .map_err(|e| ir_err(e, guard_sp))?;
                if token_carried {
                    self.record(exit, Ty::Tuple(vec![Ty::IoToken, self.tyq(v)]));
                    let nt = self.proj(fb, exit, 0, Dest::Fresh(None), guard_sp)?;
                    let nv = self.proj(fb, exit, 1, Dest::Fresh(Some(name.clone())), guard_sp)?;
                    self.token = Some(nt);
                    let vty = self.tyq(v);
                    // The exit value binds in the loop's enclosing scope (§8.5),
                    // so it is visible after the loop.
                    self.bind_new_enclosing(&name, nv, vty, mutable);
                } else {
                    let vty = self.tyq(v);
                    self.record(exit, vty.clone());
                    self.bind_new_enclosing(&name, exit, vty, mutable);
                }
            }
            ExitTerminal::BareValue => {
                let v = value
                    .ok_or_else(|| diag(LCode::FanoutNoValue, guard_sp, "exit with no value"))?;
                let payload_obj = match tok {
                    Some(t) => self
                        .pack(fb, &[t, v], Dest::Fresh(None), guard_sp)
                        .map_err(|e| ir_err(e, guard_sp))?,
                    None => v,
                };
                let exit = fb
                    .loop_exit(
                        lh,
                        payload_obj,
                        cond_exit,
                        Dest::Fresh(None),
                        ir_loc(guard_sp),
                    )
                    .map_err(|e| ir_err(e, guard_sp))?;
                if token_carried {
                    self.record(exit, Ty::Tuple(vec![Ty::IoToken, self.tyq(v)]));
                    let nt = self.proj(fb, exit, 0, Dest::Fresh(None), guard_sp)?;
                    self.token = Some(nt);
                } else {
                    let vty = self.tyq(v);
                    self.record(exit, vty);
                }
            }
        }
        Ok(())
    }

    /// Compute the exit arm's value and classify its terminal. The exit arm
    /// payload's final stage is a `-> ret`, a binding, or a bare value.
    pub(super) fn exit_arm_value(
        &mut self,
        fb: &mut FnBuilder,
        payload: &ArmPayload,
        scrutinee: ObjectId,
        guard_sp: syn::SourceLoc,
    ) -> ER<(Option<ObjectId>, ExitTerminal)> {
        // Get the final chain of the payload (its terminal). For a Block payload
        // with a tail, the tail is the terminal and all items emit as statements.
        // For a `;`-terminated Block (`tail: None`, e.g. `{ -> ret; }` or
        // `{ 99 -> print; -> ret; }`), peel the LAST chain item off as the
        // terminal and emit the preceding items as statements — statement-vs-tail
        // is identical (LD21 / ATK-11), mirroring `arm_is_routing`.
        let chain: &Chain = match payload {
            ArmPayload::Chain(c) => c,
            ArmPayload::Block(b) => match &b.tail {
                Some(t) => {
                    for item in &b.items {
                        if let BlockItem::Stmt(s) = item {
                            self.emit_stmt(fb, s)?;
                        }
                    }
                    t
                }
                None if slot_exit(b).is_some() => {
                    return self.slot_exit_value(fb, b, scrutinee, guard_sp);
                }
                None => {
                    // Find the last chain item (the terminal); emit everything
                    // before it as statements.
                    let last_chain_idx = b.items.iter().rposition(|it| {
                        matches!(it, BlockItem::Stmt(s) if matches!(s.kind, StmtKind::Chain(_)))
                    });
                    let Some(last_idx) = last_chain_idx else {
                        return Err(diag(
                            LCode::RoutingGuardShape,
                            b.span,
                            "exit arm with no terminal",
                        ));
                    };
                    for (i, item) in b.items.iter().enumerate() {
                        if i == last_idx {
                            continue;
                        }
                        if let BlockItem::Stmt(s) = item {
                            self.emit_stmt(fb, s)?;
                        }
                    }
                    match &b.items[last_idx] {
                        BlockItem::Stmt(s) => match &s.kind {
                            StmtKind::Chain(c) => c,
                            _ => unreachable!("rposition matched a Chain"),
                        },
                        _ => unreachable!("rposition matched a Stmt"),
                    }
                }
            },
        };
        // Inspect the last stage.
        let last = chain.stages.last();
        match last.map(|s| &s.kind) {
            Some(StageKind::Ret { proj }) => {
                if proj.is_some() {
                    // ret.k exit — fold into a value-then-ret (Core allows full
                    // exits; a slot exit is non-canonical → treat as L1306).
                    return Err(diag(
                        LCode::IncompleteReturn,
                        guard_sp,
                        "slot `ret.k` in a loop exit",
                    ));
                }
                // value = chain up to ret (or none for a bare `-> ret` marker).
                let val = self.exit_chain_value(fb, chain, scrutinee, true)?;
                Ok((val, ExitTerminal::Ret))
            }
            Some(StageKind::Bind { mut_span, name, .. }) => {
                let val = self.exit_chain_value(fb, chain, scrutinee, false)?;
                Ok((
                    val,
                    ExitTerminal::Bind(
                        name_text(self.source, *name).to_string(),
                        mut_span.is_some(),
                    ),
                ))
            }
            // A final bare-name `Expr(Var)` that resolves to no existing binding
            // (and is not a fn / `print`) is a fresh binding (LD1): the exit value
            // takes this name, bound in the enclosing scope (§8.5 / ATK-19). A
            // headless `-> c` exit arm seeds `cur := scrutinee` for the value.
            Some(StageKind::Expr(e)) if self.exit_bare_name_binding(e).is_some() => {
                let name = self.exit_bare_name_binding(e).unwrap();
                let val = self.exit_chain_value(fb, chain, scrutinee, false)?;
                Ok((val, ExitTerminal::Bind(name, false)))
            }
            _ => {
                // bare value: the whole chain produces the value.
                let val = self.exit_chain_value_all(fb, chain, scrutinee)?;
                Ok((val, ExitTerminal::BareValue))
            }
        }
    }

    /// structure (exit work runs once): an exit arm that computes — statements
    /// before its terminal, or stages beyond the terminal's own — runs AFTER
    /// the loop, on its exit value. The loop exits with the loop-varying values
    /// the arm reads (and the token); the arm's work is then ordinary
    /// straight-line code reading them. Its effects happen once, in order, as
    /// the source says, and none of its work joins the loop's decide phase
    /// (which runs every trip): a final norm and output projection in a
    /// layer loop's exit arm is computed once, not once per layer. `false`:
    /// the arm stays as before (no work, a headless or slot-wise terminal, a
    /// nested loop's ret exit).
    pub(super) fn defer_exit_arm(
        &mut self,
        fb: &mut FnBuilder,
        payload: &ArmPayload,
        cond_exit: ObjectId,
        lh: &mapal_ir::LoopHandle,
        token_carried: bool,
        guard_sp: syn::SourceLoc,
    ) -> ER<bool> {
        let block = match payload {
            ArmPayload::Block(b) => b.clone(),
            ArmPayload::Chain(c) => Block {
                items: Vec::new(),
                tail: Some(c.clone()),
                span: c.span,
                compute: None,
            },
        };
        if self.loop_depth >= 2 {
            return Ok(false);
        }
        // A slot-wise exit (`x -> ret.0; y -> ret.1;`): every statement runs
        // after the loop, its `ret.k` writes as the straight-line ones — in an
        // effectful fn (which returns whole, L1307) their values pack into the
        // one return after the arm's effects.
        let slots = slot_exit(&block).is_some();
        let packs = slots && self.in_effectful_fn();
        // The terminal chain: the tail, else the last statement chain.
        let stmts: Vec<&Stmt> = block
            .items
            .iter()
            .filter_map(|it| match it {
                BlockItem::Stmt(s) => Some(s),
                _ => None,
            })
            .collect();
        let (terminal, before): (&Chain, usize) = match &block.tail {
            Some(t) => (t, stmts.len()),
            None => match stmts.last().map(|s| &s.kind) {
                Some(StmtKind::Chain(c)) => (c, stmts.len() - 1),
                _ => return Ok(false),
            },
        };
        let bind = match terminal.stages.last().map(|s| &s.kind) {
            Some(StageKind::Ret { proj: None }) => None,
            Some(StageKind::Ret { proj: Some(_) }) if slots => None,
            Some(StageKind::Bind { mut_span, name, .. }) => Some((
                name_text(self.source, *name).to_string(),
                mut_span.is_some(),
            )),
            // `… -> name` with a fresh name binds the exit value (LD1).
            Some(StageKind::Expr(e)) => match self.exit_bare_name_binding(e) {
                Some(name) => Some((name, false)),
                None => return Ok(false),
            },
            _ => return Ok(false),
        };
        // The value-less `-> ret` marker after the arm's work (an effectful
        // unit fn's exit) reads no scrutinee: it defers like a headed terminal.
        let marker = terminal.head.is_none()
            && terminal.stages.len() == 1
            && matches!(terminal.stages[0].kind, StageKind::Ret { proj: None });
        if terminal.head.is_none() && !(marker && before > 0)
            || !slots && before == 0 && terminal.stages.len() < 2
        {
            return Ok(false);
        }
        // A packed slot exit reads every `ret.k` value after the loop: each
        // `ret.k` chain names its value (a headless one would read the cond).
        let arm_chains: Vec<&Chain> = stmts
            .iter()
            .filter_map(|s| match &s.kind {
                StmtKind::Chain(c) => Some(c),
                _ => None,
            })
            .chain(block.tail.as_ref())
            .collect();
        if packs
            && arm_chains
                .iter()
                .any(|c| ret_slot_of(c).is_some() && c.head.is_none())
        {
            return Ok(false);
        }
        // The loop-varying values the arm reads leave the loop as its exit.
        let mut names: Vec<(String, ObjectId, Ty, bool)> = Vec::new();
        for name in crate::typing::block_reads(self.source, &block, self.fn_sigs) {
            if let Some((b, _)) = self.scope.resolve(&name)
                && self.derived.contains(&b.obj)
            {
                names.push((name, b.obj, b.ty.clone(), b.mutable));
            }
        }
        let tok = if token_carried { self.token } else { None };
        let mut comps: Vec<ObjectId> = names.iter().map(|n| n.1).collect();
        comps.extend(tok);
        let payload_obj = match comps.len() {
            0 => fb.merge_of(lh),
            1 => comps[0],
            _ => self
                .pack(fb, &comps, Dest::Fresh(None), guard_sp)
                .map_err(|e| ir_err(e, guard_sp))?,
        };
        let exit = fb
            .loop_exit(
                lh,
                payload_obj,
                cond_exit,
                Dest::Fresh(None),
                ir_loc(guard_sp),
            )
            .map_err(|e| ir_err(e, guard_sp))?;
        let ty = self.tyq(payload_obj);
        self.record(exit, ty);
        for (k, (name, _, ty, mutable)) in names.iter().enumerate() {
            let obj = if comps.len() == 1 {
                exit
            } else {
                self.proj(
                    fb,
                    exit,
                    k as u32,
                    Dest::Fresh(Some(name.clone())),
                    guard_sp,
                )?
            };
            self.record(obj, ty.clone());
            self.rebind(name, obj, ty.clone(), *mutable);
        }
        if tok.is_some() {
            let nt = if comps.len() == 1 {
                exit
            } else {
                self.proj(fb, exit, names.len() as u32, Dest::Fresh(None), guard_sp)?
            };
            self.record(nt, Ty::IoToken);
            self.token = Some(nt);
        }
        // The arm's work, after the loop (outside it for the nesting rules).
        self.loop_depth -= 1;
        let emitted = if packs {
            self.emit_packed_slot_exit(fb, &block, cond_exit, guard_sp)
        } else {
            let mut emitted = Ok(());
            for s in &stmts[..before] {
                emitted = self.emit_stmt(fb, s);
                if emitted.is_err() {
                    break;
                }
            }
            if emitted.is_ok() {
                emitted = if marker {
                    self.emit_valueless_ret(fb, None, guard_sp)
                } else {
                    self.emit_chain(fb, terminal, ChainCtx::Statement)
                        .map(|_| ())
                };
            }
            emitted
        };
        self.loop_depth += 1;
        emitted?;
        if let Some((name, mutable)) = bind {
            let (b, _) = self
                .scope
                .resolve(&name)
                .ok_or_else(|| internal(guard_sp, "exit binding vanished"))?;
            let (obj, ty) = (b.obj, b.ty.clone());
            // The exit value binds in the loop's enclosing scope (§8.5).
            self.bind_new_enclosing(&name, obj, ty, mutable);
        }
        Ok(true)
    }

    /// A loop exit arm written slot by slot (`-false-> { x -> ret.0; y ->
    /// ret.1; }`): every statement runs in order, each `ret.k` chain's value is
    /// kept, and the complete set packs into the one exit value a full `-> ret`
    /// would carry (the canonical single `LoopExit` writer). An incomplete or
    /// repeated slot set is L1306, as at the top level. In an effectful fn the
    /// arm runs after the loop (`defer_exit_arm`); one that cannot is L1307:
    /// in the loop its effects would run every trip.
    pub(super) fn slot_exit_value(
        &mut self,
        fb: &mut FnBuilder,
        b: &Block,
        scrutinee: ObjectId,
        guard_sp: syn::SourceLoc,
    ) -> ER<(Option<ObjectId>, ExitTerminal)> {
        if self.in_effectful_fn() {
            return Err(diag(
                LCode::EffectfulReturnShape,
                guard_sp,
                "slot `ret.k` loop exit in an effectful function that cannot run after its loop",
            ));
        }
        let slots = self.slot_exit_slots(fb, b, scrutinee, guard_sp)?;
        let packed = self.pack_ret_slots(fb, slots, guard_sp)?;
        Ok((Some(packed), ExitTerminal::Ret))
    }

    /// An effectful fn's slot-wise loop exit, after its loop: the arm's
    /// statements in order, then the `ret.k` values packed into the one whole
    /// return (with the final token, as `x -> ret` writes it).
    pub(super) fn emit_packed_slot_exit(
        &mut self,
        fb: &mut FnBuilder,
        b: &Block,
        cond_exit: ObjectId,
        guard_sp: syn::SourceLoc,
    ) -> ER<()> {
        let slots = self.slot_exit_slots(fb, b, cond_exit, guard_sp)?;
        let packed = self.pack_ret_slots(fb, slots, guard_sp)?;
        self.emit_ret_existing(fb, packed, None, guard_sp, &ChainCtx::Statement)
    }

    /// Emit a slot-wise exit arm's statements in order and return each `ret.k`
    /// chain's slot and value.
    pub(super) fn slot_exit_slots(
        &mut self,
        fb: &mut FnBuilder,
        b: &Block,
        scrutinee: ObjectId,
        guard_sp: syn::SourceLoc,
    ) -> ER<Vec<(u64, ObjectId)>> {
        let mut slots: Vec<(u64, ObjectId)> = Vec::new();
        let tail = b.tail.as_ref().map(|t| (None, Some(t)));
        let items = b.items.iter().filter_map(|item| match item {
            BlockItem::Stmt(s) => Some((Some(s), None)),
            _ => None,
        });
        for (stmt, tail) in items.chain(tail) {
            let chain = match (stmt.map(|s| &s.kind), tail) {
                (Some(StmtKind::Chain(c)), _) | (None, Some(c)) => Some(c),
                _ => None,
            };
            match chain.and_then(|c| ret_slot_of(c).map(|k| (k, c))) {
                Some((k, c)) => {
                    let v = self
                        .exit_chain_value(fb, c, scrutinee, true)?
                        .ok_or_else(|| {
                            diag(
                                LCode::IncompleteReturn,
                                guard_sp,
                                "value-less `-> ret.k` with no value to write",
                            )
                        })?;
                    slots.push((k, v));
                }
                None => match (stmt, chain) {
                    (Some(s), _) => self.emit_stmt(fb, s)?,
                    (None, Some(c)) => {
                        self.emit_chain(fb, c, ChainCtx::Statement)?;
                    }
                    (None, None) => {}
                },
            }
        }
        Ok(slots)
    }

    /// Pack a slot-wise exit's `ret.k` values into the declared output (a
    /// tuple or a struct); every slot written exactly once, else L1306.
    pub(super) fn pack_ret_slots(
        &mut self,
        fb: &mut FnBuilder,
        mut slots: Vec<(u64, ObjectId)>,
        guard_sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let out = self
            .fn_sigs
            .get(&self.owner_name)
            .and_then(|s| s.ret.clone());
        let arity = match &out {
            Some(Ty::Tuple(ts)) => ts.len(),
            Some(Ty::Struct { fields, .. }) => fields.len(),
            _ => 0,
        };
        slots.sort_by_key(|&(k, _)| k);
        let complete =
            slots.len() == arity && slots.iter().enumerate().all(|(i, &(k, _))| k == i as u64);
        if !complete {
            return Err(diag(
                LCode::IncompleteReturn,
                guard_sp,
                "a loop exit's `ret.k` writes must cover every slot once",
            ));
        }
        let comps: Vec<ObjectId> = slots.into_iter().map(|(_, v)| v).collect();
        match out {
            Some(ty @ Ty::Struct { .. }) => {
                self.pack_struct(fb, ty, &comps, Dest::Fresh(None), guard_sp)
            }
            _ => self.pack(fb, &comps, Dest::Fresh(None), guard_sp),
        }
        .map_err(|e| ir_err(e, guard_sp))
    }

    /// If `e` is a bare `Var` that resolves to no existing binding and is not a
    /// fn / `print`, return the fresh binding name (LD1). Such a final stage in
    /// an exit arm is a binding terminal, not a wire-discarding stage.
    pub(super) fn exit_bare_name_binding(&self, e: &Expr) -> Option<String> {
        if let ExprKind::Var(n) = &e.kind {
            let text = name_text(self.source, *n);
            if self.scope.resolve(text).is_none()
                && !self.is_fn(text)
                && !crate::is_print_builtin(text)
                && !crate::is_time_builtin(text)
                && !crate::is_read_builtin(text)
                && !crate::is_pure_builtin(text)
            {
                return Some(text.to_string());
            }
        }
        None
    }

    /// Value of an exit chain up to (but excluding) the final Ret/Bind stage.
    /// A headless chain seeds `cur := scrutinee` (§8.3 / ATK-19) unless it is the
    /// bare `-> ret` marker (`is_ret` with no head and a single stage), which
    /// carries no value.
    pub(super) fn exit_chain_value(
        &mut self,
        fb: &mut FnBuilder,
        chain: &Chain,
        scrutinee: ObjectId,
        is_ret: bool,
    ) -> ER<Option<ObjectId>> {
        let n = chain.stages.len();
        // A headless bare `-> ret` marker carries no value (it never adopts the
        // scrutinee — §8.3). A headless `-> name` / `-> name -> …` exit chain is
        // NOT a bare marker: it seeds `cur := scrutinee`.
        if chain.head.is_none() && n == 1 && is_ret {
            return Ok(None);
        }
        // Lower head (or the seeded scrutinee) + all stages except the final one.
        let mut cur: Option<ObjectId> = match &chain.head {
            Some(h) => Some(self.emit_expr(fb, h)?),
            None => Some(scrutinee),
        };
        for (i, stage) in chain.stages.iter().enumerate() {
            if i + 1 == n {
                break; // skip the final Ret/Bind marker.
            }
            // A stage directly before the terminal `-> ret` marker must NOT
            // adopt `Dest::Ret` from the straight-line pin-2 lookahead: a loop
            // exit value feeds the LoopExit route, and only the LoopExit writes
            // the Return object. A direct Return writer here is read back by
            // the route's Pair edge — a `ret → route → ret` cycle that wedges
            // Kahn's topo_order (the Pair and LoopExit are never emitted).
            let next = if is_ret && i + 2 == n {
                None
            } else {
                chain.stages.get(i + 1)
            };
            cur = self.emit_stage(fb, stage, next, false, cur, &ChainCtx::Statement)?;
        }
        Ok(cur)
    }

    /// Value of an exit chain in bare-value position (all stages). A headless
    /// chain seeds `cur := scrutinee` (§8.3); a headed chain ignores the seed.
    pub(super) fn exit_chain_value_all(
        &mut self,
        fb: &mut FnBuilder,
        chain: &Chain,
        scrutinee: ObjectId,
    ) -> ER<Option<ObjectId>> {
        if chain.head.is_none() {
            self.emit_chain(fb, chain, ChainCtx::HeadlessSeed(scrutinee))
        } else {
            self.emit_chain(fb, chain, ChainCtx::Statement)
        }
    }
}
