//! Emitter methods: loop emission, loop bodies and routing guards.

use super::*;

impl Emitter<'_> {
    // --- loops --------------------------------------------------------------

    pub(super) fn emit_loop(&mut self, fb: &mut FnBuilder, l: &syn::LoopStmt) -> ER<()> {
        let sp = l.span;
        // 1. Discover the carried set: enclosing-scope `mut` names assigned in
        //    the body (params-then-bindings declaration order), token last iff
        //    the body has effects (§8.5/LD4).
        let mut assigned: Vec<String> = Vec::new();
        collect_loop_assigns(self.source, &l.body, &mut assigned);
        // Keep only names that resolve to an enclosing-scope mut binding, in
        // declaration order (= the order they were bound; we approximate with
        // resolution order, deduplicated).
        let mut carried: Vec<(String, Ty, ObjectId, u32)> = Vec::new();
        let mut seen: BTreeMap<String, ()> = BTreeMap::new();
        for name in &assigned {
            if seen.contains_key(name) {
                continue;
            }
            if let Some((b, _)) = self.scope.resolve(name)
                && b.mutable
            {
                seen.insert(name.clone(), ());
                carried.push((name.clone(), b.ty.clone(), b.obj, b.decl_seq));
            }
        }
        // Carried order = declaration order (LD4): params first, then bindings in
        // source order — exactly the `decl_seq` ordering.
        carried.sort_by_key(|(_, _, _, seq)| *seq);
        let carried: Vec<(String, Ty, ObjectId)> =
            carried.into_iter().map(|(n, t, o, _)| (n, t, o)).collect();
        let body_effectful = loop_body_has_effect(self.source, &l.body, self.fn_sigs);
        if carried.is_empty() {
            return Err(diag(
                LCode::LoopNoState,
                sp,
                "loop assigns no enclosing `mut` variable",
            ));
        }
        // L1504 (nested loops): a nested loop inside a token-carrying loop gives
        // the token a third route consumer; an inner loop assigning an
        // outer-carried `mut` closes a cross-merge cycle. Both are out of Core.
        if self.loop_depth > 0 {
            if self.enclosing_token_loop {
                return Err(diag(
                    LCode::NestedLoopShape,
                    sp,
                    "a nested loop inside a token-carrying loop is out of Core",
                ));
            }
            for (name, _, _) in &carried {
                if self.outer_carried.contains(name) {
                    return Err(diag(
                        LCode::NestedLoopShape,
                        sp,
                        format!("inner loop assigns the enclosing-loop-carried `{name}`"),
                    ));
                }
            }
        }

        // 2. init: the carried values (+ token), packed in carried order when
        //    |U| >= 2; the single object otherwise.
        let token_carried = body_effectful;
        let mut init_objs: Vec<ObjectId> = carried.iter().map(|(_, _, o)| *o).collect();
        let mut u_tys: Vec<Ty> = carried.iter().map(|(_, t, _)| t.clone()).collect();
        if token_carried {
            let tok = self
                .token
                .ok_or_else(|| internal(sp, "effectful loop without token"))?;
            init_objs.push(tok);
            u_tys.push(Ty::IoToken);
        }
        let u_single = init_objs.len() == 1;
        let init = if u_single {
            init_objs[0]
        } else {
            self.pack(fb, &init_objs, Dest::Fresh(None), sp)
                .map_err(|e| ir_err(e, sp))?
        };

        // 3. begin_loop + merge; rebind carried names to projs of the merge.
        let lh = fb.begin_loop(init, ir_loc(sp)).map_err(|e| ir_err(e, sp))?;
        let merge = fb.merge_of(&lh);
        let u_ty = if u_single {
            u_tys[0].clone()
        } else {
            Ty::Tuple(u_tys.clone())
        };
        self.record(merge, u_ty.clone());
        self.loop_depth += 1;
        // Seed the derives-from-merge set with the merge view (§7.3).
        let saved_derived = std::mem::take(&mut self.derived);
        self.derived.insert(merge);
        // Nesting state for L1504 (restored after the loop).
        let saved_token_loop = self.enclosing_token_loop;
        let saved_outer = std::mem::take(&mut self.outer_carried);
        self.enclosing_token_loop = saved_token_loop || token_carried;
        self.outer_carried = saved_outer.clone();
        for (name, _, _) in &carried {
            self.outer_carried.insert(name.clone());
        }

        self.scope.push();
        let n_state = carried.len();
        for (k, (name, ty, _)) in carried.iter().enumerate() {
            let obj = if u_single {
                merge
            } else {
                self.proj(fb, merge, k as u32, Dest::Fresh(Some(name.clone())), sp)?
            };
            self.derived.insert(obj);
            self.record(obj, ty.clone());
            self.rebind(name, obj, ty.clone(), true);
        }
        if token_carried {
            let tok = if u_single {
                merge
            } else {
                self.proj(fb, merge, n_state as u32, Dest::Fresh(None), sp)?
            };
            self.derived.insert(tok);
            self.record(tok, Ty::IoToken);
            self.token = Some(tok);
        }

        // 4. Lower the body items, then find the routing guard (final item/tail).
        let routing =
            self.lower_loop_body(fb, &l.body, &lh, &carried, token_carried, u_single, &u_ty)?;
        if !routing {
            self.scope.pop();
            self.loop_depth -= 1;
            self.derived = saved_derived;
            self.enclosing_token_loop = saved_token_loop;
            self.outer_carried = saved_outer;
            return Err(diag(
                LCode::LoopNoExit,
                sp,
                "loop has no routing guard / exit",
            ));
        }

        fb.end_loop(lh).map_err(|e| ir_err(e, sp))?;
        self.scope.pop();
        self.loop_depth -= 1;
        self.derived = saved_derived;
        self.enclosing_token_loop = saved_token_loop;
        self.outer_carried = saved_outer;

        // Poison the carried names in the enclosing scope (LD9/L1107).
        for (name, _, _) in &carried {
            self.scope.poison(name);
        }
        Ok(())
    }

    /// Lower a loop body: statements then the routing guard (final item or
    /// tail). Returns whether a routing guard was found and emitted.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower_loop_body(
        &mut self,
        fb: &mut FnBuilder,
        body: &Block,
        lh: &mapal_ir::LoopHandle,
        carried: &[(String, Ty, ObjectId)],
        token_carried: bool,
        u_single: bool,
        u_ty: &Ty,
    ) -> ER<bool> {
        // The routing guard is the final body item (statement or tail). Identify
        // it; lower all preceding items as statements.
        // Build the list of "items" including the tail as the last item.
        enum Item<'a> {
            Stmt(&'a Stmt),
            Tail(&'a Chain),
        }
        let mut items: Vec<Item> = Vec::new();
        for it in &body.items {
            if let BlockItem::Stmt(s) = it {
                items.push(Item::Stmt(s));
            }
        }
        if let Some(t) = &body.tail {
            items.push(Item::Tail(t));
        }
        let n = items.len();
        for (i, it) in items.into_iter().enumerate() {
            let is_last = i + 1 == n;
            match it {
                Item::Stmt(s) => {
                    if is_last {
                        // Last item: must be the routing guard (a chain whose
                        // last stage is a Guard with a routing arm).
                        if let StmtKind::Chain(c) = &s.kind
                            && let Some(g) = chain_routing_guard(c)
                        {
                            self.emit_routing_guard(
                                fb,
                                c,
                                g,
                                lh,
                                carried,
                                token_carried,
                                u_single,
                                u_ty,
                            )?;
                            return Ok(true);
                        }
                        // Not a routing guard in final position.
                        self.emit_stmt(fb, s)?;
                        return Ok(false);
                    }
                    self.emit_stmt(fb, s)?;
                }
                Item::Tail(c) => {
                    if let Some(g) = chain_routing_guard(c) {
                        self.emit_routing_guard(
                            fb,
                            c,
                            g,
                            lh,
                            carried,
                            token_carried,
                            u_single,
                            u_ty,
                        )?;
                        return Ok(true);
                    }
                    self.emit_chain(fb, c, ChainCtx::Statement)?;
                    return Ok(false);
                }
            }
        }
        Ok(false)
    }

    /// Emit a routing guard: the loop's route point (§8.5 step 4).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_routing_guard(
        &mut self,
        fb: &mut FnBuilder,
        chain: &Chain,
        arms: &[GuardArm],
        lh: &mapal_ir::LoopHandle,
        carried: &[(String, Ty, ObjectId)],
        token_carried: bool,
        u_single: bool,
        u_ty: &Ty,
    ) -> ER<()> {
        let guard_sp = chain.stages.last().map(|s| s.span).unwrap_or(chain.span);
        // Classify the two arms first (a structural check, independent of the
        // scrutinee): exactly one jump pole + one exit pole over {true,false},
        // else L1409. This must precede the scrutinee-type check so an
        // integer-discriminant routing guard reports L1409, not L1406.
        let (jump_arm, jump_pole, exit_arm, exit_pole) = classify_routing(arms, guard_sp)?;
        let _ = exit_pole;

        // The chain up to the guard computes the scrutinee (= cond).
        let cond = self.lower_scrutinee(fb, chain)?;
        let cty = self.ty_of(fb, cond);
        if cty != Ty::Bool {
            return Err(diag(
                LCode::GuardScrutineeType,
                guard_sp,
                "routing guard scrutinee is not bool",
            ));
        }
        // L1503: the cond's value must derive from the merge view (§7.3) — a
        // non-derived cond is a vacuous or unconditionally-divergent loop.
        if !self.derived.contains(&cond) {
            return Err(diag(
                LCode::LoopGuardShape,
                guard_sp,
                "loop guard condition does not derive from the loop state",
            ));
        }

        // Snapshot bindings + token register (§3 / SF-2c).
        let snap_scope = self.scope.snapshot();
        let snap_token = self.token;

        // --- jump arm: arm-local rebinds, then recomputed next-state -----------
        self.scope = snap_scope.clone();
        self.token = snap_token;
        self.scope.push();
        self.emit_jump_arm_updates(fb, &jump_arm.payload)?;
        // next-state = arm-local carried values (+ arm-local token), in order.
        let mut next_objs: Vec<ObjectId> = Vec::new();
        for (name, _, _) in carried {
            let (b, _) = self
                .scope
                .resolve(name)
                .ok_or_else(|| internal(guard_sp, "carried name vanished"))?;
            next_objs.push(b.obj);
        }
        if token_carried {
            next_objs.push(
                self.token
                    .ok_or_else(|| internal(guard_sp, "token vanished in jump arm"))?,
            );
        }
        // L1503: ≥1 component of the jump arm's next-state must derive from the
        // merge (a fully non-derived next-state is seal `LoopBackOutsideScc`).
        if !next_objs.iter().any(|o| self.derived.contains(o)) {
            self.scope.pop();
            return Err(diag(
                LCode::LoopGuardShape,
                guard_sp,
                "loop back-edge state does not derive from the loop state",
            ));
        }
        self.scope.pop();
        let next = if next_objs.len() == 1 {
            next_objs[0]
        } else {
            self.pack(fb, &next_objs, Dest::Fresh(None), guard_sp)
                .map_err(|e| ir_err(e, guard_sp))?
        };
        // Polarity: LoopBack fires on true. A `-false->` jump wraps cond in Not.
        let cond_back = if jump_pole {
            cond
        } else {
            self.unop(fb, Operation::Not, cond, Dest::Fresh(None), guard_sp)?
        };
        let _ = u_ty;
        let _ = u_single;
        fb.loop_back(lh, next, cond_back, ir_loc(guard_sp))
            .map_err(|e| ir_err(e, guard_sp))?;

        // --- exit arm: against the snapshot ------------------------------------
        self.scope = snap_scope;
        self.token = snap_token;
        // cond for exit is the complement of the jump pole.
        let cond_exit = if jump_pole {
            // jump on true → exit on false → builder LoopExit fires on false, so
            // pass the shared cond directly (LoopExit fires on false).
            cond
        } else {
            // jump on false → exit on true → LoopExit must fire on true, so pass Not(cond).
            self.unop(fb, Operation::Not, cond, Dest::Fresh(None), guard_sp)?
        };
        self.emit_exit_arm(
            fb,
            &exit_arm.payload,
            cond_exit,
            cond,
            lh,
            token_carried,
            guard_sp,
        )?;
        Ok(())
    }
}
