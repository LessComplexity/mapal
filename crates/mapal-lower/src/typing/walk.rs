//! The typing walk over blocks, statements, chains, stages, guards and map/fold bodies.

use super::*;

impl Walk<'_> {
    // --- the walk -----------------------------------------------------------

    pub(super) fn block(&mut self, block: &Block, scope: &mut ScopeStack<WTy>) {
        for item in &block.items {
            match item {
                BlockItem::Stmt(s) => self.stmt(s, scope),
                BlockItem::Arm(_) => {
                    // bare arm at block level — guard arms only appear inside a
                    // Guard stage; a top-level arm is structurally rejected by
                    // the parser. Defensive no-op.
                }
            }
        }
        if let Some(tail) = &block.tail {
            self.chain(tail, scope);
        }
    }

    pub(super) fn stmt(&mut self, stmt: &Stmt, scope: &mut ScopeStack<WTy>) {
        match &stmt.kind {
            StmtKind::Chain(c) => {
                self.chain(c, scope);
            }
            StmtKind::Bind(b) => {
                self.check_not_size(b.name);
                let v = self.expr(&b.value, scope);
                // `c[i] <- v` (ADR-0021): an *indexed* bind is an element write,
                // not a fresh binding. Type the index, unify `v` with `c`'s
                // element ty (so a bare literal resolves to the element width, not
                // a spurious i32 clash — this is how width-unification covers the
                // element-ty mismatch, L1201/L1203), and leave `c`'s binding
                // intact (its array ty is unchanged).
                if let Some(idx) = &b.index {
                    let _ = self.expr(idx, scope);
                    if let Some((bt, _)) = scope.resolve(name_text(self.source, b.name))
                        && let Some(elem) = array_elem_wty(&bt.clone())
                    {
                        self.unify(&v, &elem, b.span);
                    }
                    return;
                }
                if let Some(annot) = &b.ty
                    && let Some(t) = self.types.resolve_sized(
                        self.source,
                        annot,
                        &self.sizes.env,
                        &mut self.diags,
                    )
                {
                    self.unify_known(&v, &t, b.span);
                    scope.bind(name_text(self.source, b.name), WTy::Known(t));
                    return;
                }
                scope.bind(name_text(self.source, b.name), v);
            }
            StmtKind::Loop(l) => {
                scope.push_loop();
                self.block(&l.body, scope);
                scope.pop();
            }
            StmtKind::Error => {}
        }
    }

    pub(super) fn chain(&mut self, chain: &Chain, scope: &mut ScopeStack<WTy>) {
        self.chain_seeded(chain, WTy::Unknown, scope);
    }

    /// Like [`chain`], but a **headless** chain (no head expr) starts from `seed`
    /// instead of `Unknown` — the fanout wire threaded into each branch so a
    /// tuple-producing op (`enumerate`/`zip`) inside a branch types its downstream
    /// `map` body correctly (emit already threads the real wire; this keeps the
    /// D1 BlockSig in step).
    pub(super) fn chain_seeded(&mut self, chain: &Chain, seed: WTy, scope: &mut ScopeStack<WTy>) {
        // `() -> g`: the Unit witness a zero-parameter fn takes (anywhere
        // else `()` is not a value — emission's L1301).
        let zero_call = matches!(chain.stages.first().map(|s| &s.kind),
            Some(StageKind::Expr(e)) if matches!(&e.kind, ExprKind::Var(n)
                if self.fn_sigs.get(name_text(self.source, *n)).is_some_and(|g| g.params.is_empty())));
        let mut cur: WTy = match &chain.head {
            Some(h) if zero_call && matches!(h.kind, ExprKind::Unit) => WTy::Known(Ty::Unit),
            Some(h) => self.expr(h, scope),
            None => seed,
        };
        let mut prev: Option<&Expr> = chain.head.as_ref();
        let is_fn_tail = self.fn_tail == Some((chain.span.start, chain.span.end));
        for (i, stage) in chain.stages.iter().enumerate() {
            let next = chain.stages.get(i + 1);
            let outer = self.at_fn_tail;
            self.at_fn_tail = is_fn_tail && next.is_none();
            cur = self.stage(&stage.kind, stage.span, cur, prev, next, scope);
            self.at_fn_tail = outer;
            prev = match &stage.kind {
                StageKind::Expr(e) => Some(e),
                _ => None,
            };
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn stage(
        &mut self,
        kind: &StageKind,
        stage_span: syn::SourceLoc,
        cur: WTy,
        prev: Option<&Expr>,
        next: Option<&Stage>,
        scope: &mut ScopeStack<WTy>,
    ) -> WTy {
        match kind {
            StageKind::Expr(e) => {
                if let ExprKind::Var(n) = &e.kind {
                    let text = name_text(self.source, *n);
                    if scope.contains(text) {
                        // rebind: cur unifies into the existing binding's ty.
                        if let Some((b, _)) = scope.resolve(text) {
                            let bty = b.clone();
                            self.unify(&cur, &bty, e.span);
                        }
                        cur
                    } else if self.fn_sigs.get(text).is_some_and(|s| s.generic) {
                        // ADR-0047: a call to a size-generic function.
                        self.generic_call(text, &cur, next, e.span, None)
                    } else if let Some(sig) = self.fn_sigs.get(text).cloned() {
                        // call: cur (arg) unifies with declared input; result ty.
                        self.type_call(&sig, &cur, e.span);
                        match &sig.ret {
                            Some(r) => WTy::Known(r.clone()),
                            None => WTy::Unknown,
                        }
                    } else if crate::is_print_builtin(text) {
                        // print sink; result is "no value".
                        WTy::Unknown
                    } else if crate::is_read_builtin(text) {
                        // ADR-0045: the binding's annotation types the array
                        // (emit owns L1617 / L1618); nothing to unify here.
                        WTy::Unknown
                    } else if crate::is_time_builtin(text) {
                        // `() -> time` (plan-time-builtin): no source, f64
                        // milliseconds out. `cur` is whatever the wire-less
                        // head left (Unknown) and is deliberately not unified.
                        WTy::Known(Ty::f64())
                    } else if text == "zip" {
                        // zip: cur is a 2-tuple `([A;n], [B;n])`; result `[(A,B);n]`.
                        // Synthesize when resolvable so a downstream map sees the
                        // tuple element ty (emit re-derives + owns the L-codes).
                        self.zip_result(&cur)
                    } else if text == "enumerate" {
                        // enumerate: cur is `[A;n]`; result `[(i32, A);n]`.
                        self.enumerate_result(&cur)
                    } else if text == "iota" {
                        // ADR-0031: `n -> iota` — the count flows in. Size
                        // synthesized from the visible literal stage; emit
                        // re-derives and the builder owns static-n (L1612).
                        // A size parameter is a literal (ADR-0047 D4).
                        match prev.and_then(|p| self.int_const(p, scope)) {
                            Some(n) if n > 0 && n <= i32::MAX as u64 => {
                                WTy::Array(Box::new(WTy::Known(Ty::i32())), n)
                            }
                            // onto-device: an extent past i32 — the count is an
                            // `i64` literal and the elements its index type.
                            Some(n) if n <= i64::MAX as u64 && n > 0 => {
                                self.unify_known(&cur, &Ty::i64(), e.span);
                                WTy::Array(Box::new(WTy::Known(Ty::index_ty(n))), n)
                            }
                            _ => WTy::Unknown,
                        }
                    } else if text == "fill" {
                        // ADR-0031: `(x, n) -> fill` — elem from the tuple's
                        // first component, size from the visible literal
                        // (emit re-derives; L1613).
                        let elem = match &cur {
                            WTy::Tuple(ws) if ws.len() == 2 => ws[0].clone(),
                            _ => WTy::Unknown,
                        };
                        match prev.map(|p| &p.kind) {
                            Some(ExprKind::Tuple(xs)) if xs.len() == 2 => {
                                match self.int_const(&xs[1], scope) {
                                    Some(n) if n > 0 && n <= i32::MAX as u64 => {
                                        WTy::Array(Box::new(elem), n)
                                    }
                                    Some(n) if n > 0 && n <= i64::MAX as u64 => {
                                        if let WTy::Tuple(ws) = &cur {
                                            self.unify_known(&ws[1], &Ty::i64(), e.span);
                                        }
                                        WTy::Array(Box::new(elem), n)
                                    }
                                    _ => WTy::Unknown,
                                }
                            }
                            _ => WTy::Unknown,
                        }
                    } else if let Some(verb) = Verb::from_name(text) {
                        // ADR-0051 D4: `to` / `widen`, the target inferred
                        // from the context (emit owns L1614 / L1616 / L1708).
                        self.conversion(verb, None, &cur, next, e.span)
                    } else if let Some((verb, f)) = crate::formats::removed(text) {
                        self.diags.push(removed_conversion(text, verb, f, e.span));
                        WTy::Unknown
                    } else if mapal_ir::MathFn::from_name(text).is_some() {
                        // ADR-0042 / ADR-0046: `x -> exp` (`sqrt`, `sin`,
                        // `cos`) keeps the float width — a literal's variable
                        // flows through, so the width is whatever unifies
                        // with it (emit owns L1615).
                        match &cur {
                            WTy::Known(Ty::Float { .. }) | WTy::Var(_) => cur.clone(),
                            _ => WTy::Unknown,
                        }
                    } else {
                        // bind cur to a fresh name.
                        self.check_not_size(*n);
                        scope.bind(text, cur.clone());
                        cur
                    }
                } else if let ExprKind::Generic { name, args } = &e.kind {
                    // ADR-0051 D2: a stage with explicit generic arguments.
                    let text = name_text(self.source, *name);
                    if self.fn_sigs.get(text).is_some_and(|s| s.generic) && !scope.contains(text) {
                        self.generic_call(text, &cur, next, e.span, Some(args))
                    } else if let (Some(verb), false) =
                        (Verb::from_name(text), scope.contains(text))
                    {
                        self.conversion(verb, Some(args), &cur, next, e.span)
                    } else {
                        self.diags.push(no_generic_args(
                            self.source,
                            text,
                            scope.contains(text),
                            self.fn_sigs.contains_key(text),
                            e.span,
                        ));
                        WTy::Unknown
                    }
                } else {
                    // general expr stage — emission rejects (L1302). Walk for
                    // literals anyway so they resolve.
                    self.expr(e, scope);
                    WTy::Unknown
                }
            }
            StageKind::OpShorthand { expr } => {
                // The hole is `cur`; lower the expr with Hole↦cur.
                self.expr_with_hole(expr, &cur, scope)
            }
            StageKind::Bind { name, ty, .. } => {
                self.check_not_size(*name);
                if let Some(annot) = ty
                    && let Some(t) = self.types.resolve_sized(
                        self.source,
                        annot,
                        &self.sizes.env,
                        &mut self.diags,
                    )
                {
                    self.unify_known(&cur, &t, stage_span);
                    scope.bind(name_text(self.source, *name), WTy::Known(t.clone()));
                    return WTy::Known(t);
                }
                scope.bind(name_text(self.source, *name), cur.clone());
                cur
            }
            StageKind::Ret { proj } => {
                // Unify the returned value with the declared output (full) or
                // component (`ret.k`) ty so literals take the return width
                // (§7.2). Reports L1203 on an int/float clash via `unify_known`.
                if let Some(out) = self.owner_ret.clone() {
                    match proj {
                        None => self.unify_known(&cur, &out, stage_span),
                        // Skip the widening for a slot > u32::MAX (it would
                        // truncate and unify against the wrong component);
                        // emit.rs rejects the slot with L1205 (LOWER-RETK-TRUNC).
                        Some((k, _)) if *k <= u32::MAX as u64 => {
                            if let Some(c) = out.component_ty(*k as u32) {
                                self.unify_known(&cur, &c.clone(), stage_span);
                            }
                        }
                        Some(_) => {}
                    }
                }
                WTy::Unknown
            }
            StageKind::LoopJump => WTy::Unknown,
            StageKind::Guard(arms) => self.guard(arms, &cur, scope),
            StageKind::Fanout { branches, .. } => {
                for b in branches {
                    self.chain_seeded(b, cur.clone(), scope);
                }
                WTy::Unknown
            }
            StageKind::SeqBlock(body) => {
                // Type-walk the seq body in the enclosing scope (ADR-0019 pin b):
                // chain statements + tail seed from the wire `cur` (pin a), so a
                // headless chain resolves its literals against the seq input.
                // Bind/Loop are ordinary statements. Value ty is left Unknown
                // (like a fanout join): emission re-synthesizes the real ty.
                for item in &body.items {
                    if let BlockItem::Stmt(s) = item {
                        match &s.kind {
                            StmtKind::Chain(c) => self.chain_seeded(c, cur.clone(), scope),
                            _ => self.stmt(s, scope),
                        }
                    }
                }
                if let Some(tail) = &body.tail {
                    self.chain_seeded(tail, cur.clone(), scope);
                }
                WTy::Unknown
            }
            StageKind::MapFold { op, params, body } => {
                self.map_fold(*op, params, body, stage_span, &cur, scope)
            }
            StageKind::StmtBlock(_) | StageKind::Error(_) => WTy::Unknown,
        }
    }

    pub(super) fn guard(
        &mut self,
        arms: &[syn::GuardArm],
        cur: &WTy,
        scope: &mut ScopeStack<WTy>,
    ) -> WTy {
        // For value-match guards, each integer discriminant unifies with the
        // scrutinee (so a bare-int scrutinee literal resolves). Arm payloads
        // produce values that unify into a common ty.
        let mut arm_tys: Vec<WTy> = Vec::new();
        let mut is_routing = false;
        let routing_guard = arms.iter().any(|a| arm_is_routing(&a.payload));
        for arm in arms {
            if let GuardDiscr::Int(_) = arm.discr {
                // unify cur with i (cur is integer); we cannot construct a lit
                // var for the discr (it lives in the guard token), but we pin
                // cur to integer-ish by leaving it; emission range-checks.
            }
            // Detect routing (headless chain ending in LoopJump).
            if arm_is_routing(&arm.payload) {
                is_routing = true;
            }
            // A routing guard's exit arm binds its exit value in the loop's
            // enclosing scope (DESIGN §8.5, as emission does), so a later
            // map/fold body sees it as a capture. A bare final name binds only
            // when it was unbound before the arm (else it is a rebind or call).
            let exit_name = (routing_guard && !arm_is_routing(&arm.payload))
                .then(|| arm_exit_bind(&arm.payload))
                .flatten()
                .map(|n| name_text(self.source, n))
                .filter(|t| !scope.contains(t) && !self.fn_sigs.contains_key(*t));
            scope.push();
            let aty = self.arm_value(&arm.payload, cur, scope);
            let exit_bind = exit_name.and_then(|t| scope.resolve(t).map(|(w, _)| (t, w.clone())));
            scope.pop();
            if let Some((text, w)) = exit_bind {
                scope.bind_loop_exit(text, w);
            }
            arm_tys.push(aty);
        }
        if is_routing {
            return WTy::Unknown;
        }
        // Phi guard: unify arm tys pairwise; result is the common ty.
        for i in 1..arm_tys.len() {
            let (a, b) = (arm_tys[0].clone(), arm_tys[i].clone());
            self.unify(&a, &b, syn::SourceLoc::new(0, 0));
        }
        arm_tys.into_iter().next().unwrap_or(WTy::Unknown)
    }

    pub(super) fn arm_value(
        &mut self,
        payload: &ArmPayload,
        cur: &WTy,
        scope: &mut ScopeStack<WTy>,
    ) -> WTy {
        match payload {
            ArmPayload::Chain(c) => self.headless_chain_value(c, cur, scope),
            ArmPayload::Block(b) => {
                for item in &b.items {
                    if let BlockItem::Stmt(s) = item {
                        self.stmt(s, scope);
                    }
                }
                match &b.tail {
                    Some(t) => self.headless_chain_value(t, cur, scope),
                    None => WTy::Unknown,
                }
            }
        }
    }

    /// A headless chain in an arm position seeds cur := scrutinee unless it is a
    /// bare marker (-> ret / -> loop).
    pub(super) fn headless_chain_value(
        &mut self,
        chain: &Chain,
        scrutinee: &WTy,
        scope: &mut ScopeStack<WTy>,
    ) -> WTy {
        let mut cur = match &chain.head {
            Some(h) => self.expr(h, scope),
            None => scrutinee.clone(),
        };
        let mut prev: Option<&Expr> = chain.head.as_ref();
        for (i, stage) in chain.stages.iter().enumerate() {
            let next = chain.stages.get(i + 1);
            cur = self.stage(&stage.kind, stage.span, cur, prev, next, scope);
            prev = match &stage.kind {
                StageKind::Expr(e) => Some(e),
                _ => None,
            };
        }
        cur
    }

    pub(super) fn map_fold(
        &mut self,
        op: CollOp,
        params: &[syn::Name],
        body: &Block,
        stage_span: syn::SourceLoc,
        cur: &WTy,
        scope: &mut ScopeStack<WTy>,
    ) -> WTy {
        // Determine element / acc tys from cur (ADR-0048 D5: a struct reads
        // through its `_` array).
        let cur = &match (op, cur) {
            (CollOp::Map, _) => self.through_default(cur, |t| matches!(t, Ty::Array { .. })),
            (CollOp::Fold, WTy::Tuple(ws)) if ws.len() == 2 => WTy::Tuple(vec![
                ws[0].clone(),
                self.through_default(&ws[1], |t| matches!(t, Ty::Array { .. })),
            ]),
            (CollOp::Fold, WTy::Known(Ty::Tuple(ts))) if ts.len() == 2 => WTy::Tuple(vec![
                WTy::Known(ts[0].clone()),
                self.through_default(&WTy::Known(ts[1].clone()), |t| {
                    matches!(t, Ty::Array { .. })
                }),
            ]),
            _ => cur.clone(),
        };
        let cur_ty = self.wty_to_ty(cur);
        let idx = self.block_counter;
        self.block_counter += 1;
        let name = format!(
            "{}::{}@{}",
            self.owner_name,
            match op {
                CollOp::Map => "map",
                CollOp::Fold => "fold",
            },
            idx
        );
        // L1601: arity (map = 1, fold = 2 params).
        let expected_arity = match op {
            CollOp::Map => 1,
            CollOp::Fold => 2,
        };
        if params.len() != expected_arity {
            self.diags.push(diag(
                LCode::BlockArity,
                stage_span,
                format!(
                    "{} block takes {} parameter(s), found {}",
                    match op {
                        CollOp::Map => "map",
                        CollOp::Fold => "fold",
                    },
                    expected_arity,
                    params.len()
                ),
            ));
        }
        // L1604: the body must have a tail value.
        if body.tail.is_none() {
            self.diags.push(diag(
                LCode::BodyNoValue,
                body.span,
                "map/fold body has no tail value".to_string(),
            ));
        }
        // L1605: no effects (print / effectful call) inside the body.
        if let Some(sp) = body_effect_span(self.source, body, self.fn_sigs) {
            self.diags.push(diag(
                LCode::BodyEffectful,
                sp,
                "a map/fold body cannot print, read the clock or call something that does: its \
                 cells have no set order (they may run in parallel). Map to values, then print \
                 the result (`arr -> map { a -> a * 2 }`)"
                    .to_string(),
            ));
        }
        // ADR-0027: bodies may READ enclosing bindings (pure read captures —
        // broadcast edges, hidden leading body-input components). The one
        // remaining L1108 case is a WRITE to an enclosing name (an indexed
        // bind targeting it, ADR-0021's "rebind, never a fresh shadow").
        for p in params {
            self.check_not_size(*p);
        }
        // ADR-0047 D4: a size parameter is a constant in the body, never a
        // capture (seeded as a body-local name).
        let param_names: std::collections::BTreeSet<&str> = params
            .iter()
            .map(|p| name_text(self.source, *p))
            .chain(self.sizes.env.keys().map(String::as_str))
            .chain(self.types.globals().consts.keys().map(String::as_str))
            .collect();
        let caps = body_captures(self.source, body, &param_names, self.fn_sigs);
        if let Some((cap, sp)) = &caps.write {
            self.diags.push(diag(
                LCode::CaptureInBody,
                *sp,
                format!(
                    "map/fold body writes to the enclosing `{cap}` — a capture is read-only \
                     (body instances run per-element, in parallel, in an order you don't control; \
                     write your own local and pass the result out instead)"
                ),
            ));
        }
        // Resolve each read capture's ty from the enclosing scope (unknown
        // names error later in the body walk as L1101 — not duplicated here).
        // A capture keeps its literal vars (`[1.0, 2.0] -> xs` read in a
        // body as `f32`): the body's uses and the signature share them.
        let mut cap_tys: Vec<(String, Ty)> = Vec::new();
        let mut cap_ws: Vec<WTy> = Vec::new();
        for (cap, _sp) in &caps.reads {
            if let Some((w, _)) = scope.resolve(cap) {
                let w = w.clone();
                if let Some(t) = self.wty_to_ty(&w) {
                    cap_tys.push((cap.clone(), t));
                    cap_ws.push(w);
                }
            }
        }
        // Seed the body scope with the captures (leading components), so the
        // body walk types them and nested bodies can capture transitively.

        let mut body_scope: ScopeStack<WTy> = ScopeStack::new();
        for ((cap, _), w) in cap_tys.iter().zip(&cap_ws) {
            body_scope.bind(cap, w.clone());
        }
        // The fold seed and element WTys are extracted from `cur` *without*
        // resolving (so the seed's var threads into the body and unifies).
        let (seed_w, elem_w): (WTy, WTy) = match cur {
            WTy::Tuple(ws) if ws.len() == 2 => {
                let arr_elem = match &ws[1] {
                    WTy::Array(e, _) => (**e).clone(),
                    WTy::Known(Ty::Array { elem, .. }) => WTy::Known((**elem).clone()),
                    _ => WTy::Unknown,
                };
                (ws[0].clone(), arr_elem)
            }
            _ => (WTy::Unknown, WTy::Unknown),
        };
        let body_in: Ty = match op {
            CollOp::Map => {
                let elem = cur_ty.as_ref().and_then(array_elem).unwrap_or(Ty::i32());
                if let Some(p0) = params.first() {
                    body_scope.bind(name_text(self.source, *p0), WTy::Known(elem.clone()));
                }
                elem
            }
            CollOp::Fold => {
                // Bind acc to the seed WTy, t to the element WTy; walking the body
                // unifies them (acc + px.r ⇒ seed resolves to f32).
                if params.len() == 2 {
                    body_scope.bind(name_text(self.source, params[0]), seed_w.clone());
                    body_scope.bind(name_text(self.source, params[1]), elem_w.clone());
                }
                Ty::Unit // filled in after the body walk resolves acc.
            }
        };
        // Walk the body for literals + tail value ty.
        for item in &body.items {
            if let BlockItem::Stmt(s) = item {
                self.stmt(s, &mut body_scope);
            }
        }
        let tail_ty = match &body.tail {
            Some(t) => self.headless_chain_value(t, &WTy::Unknown, &mut body_scope),
            None => WTy::Unknown,
        };
        // For fold, the body result IS the acc ty; unify the tail value with the
        // seed so both resolve together.
        if matches!(op, CollOp::Fold) {
            self.unify(&seed_w, &tail_ty, stage_span);
        }
        // The body's output keeps its literal vars: a float literal in a map
        // body or a fold's seed takes its width from a later use or
        // annotation (`map { i -> 0.01 } -> start: [f32; 64]`); the block's
        // signature is read once the function's literals resolve.
        let out_fallback = match op {
            CollOp::Map => array_elem(&cur_ty.clone().unwrap_or(Ty::i32())).unwrap_or(Ty::i32()),
            CollOp::Fold => self.wty_to_ty(&seed_w).unwrap_or(Ty::f32()),
        };
        let body_out_w = match (self.wty_to_ty(&tail_ty), op) {
            (Some(_), _) => tail_ty.clone(),
            (None, CollOp::Fold) if self.wty_to_ty(&seed_w).is_some() => seed_w.clone(),
            (None, _) => WTy::Known(out_fallback.clone()),
        };
        // mask-frontend: a struct capture as its fields' captures (the body
        // scope above still types the struct by its name).
        let (cap_tys, cap_ws, records) =
            match cap_tys.iter().any(|(_, t)| matches!(t, Ty::Struct { .. })) {
                false => (cap_tys, cap_ws, Vec::new()),
                true => {
                    let mut tys = Vec::new();
                    let mut ws = Vec::new();
                    for (c, w) in cap_tys.iter().zip(&cap_ws) {
                        match &c.1 {
                            Ty::Struct { .. } => {
                                let (flat, _) = flatten_captures(std::slice::from_ref(c));
                                for f in flat {
                                    ws.push(WTy::Known(f.1.clone()));
                                    tys.push(f);
                                }
                            }
                            _ => {
                                tys.push(c.clone());
                                ws.push(w.clone());
                            }
                        }
                    }
                    let (_, records) = flatten_captures(&cap_tys);
                    (tys, ws, records)
                }
            };
        let caps: Vec<(WTy, Ty)> = cap_tys
            .iter()
            .zip(&cap_ws)
            .map(|((_, t), w)| (w.clone(), t.clone()))
            .collect();
        let comps = match op {
            CollOp::Fold => {
                let mut comps = caps;
                comps.push((seed_w.clone(), Ty::f32()));
                comps.push((elem_w.clone(), Ty::f32()));
                Some(comps)
            }
            CollOp::Map if !caps.is_empty() => {
                let mut comps = caps;
                comps.push((WTy::Known(body_in.clone()), body_in.clone()));
                Some(comps)
            }
            CollOp::Map => None,
        };
        self.pending_blocks.push(PendingBlock {
            span: (stage_span.start, stage_span.end),
            comps,
            body_in: (WTy::Known(body_in.clone()), body_in),
            body_out: (body_out_w.clone(), out_fallback),
            captures: cap_tys.into_iter().zip(cap_ws).collect(),
            records,
            name,
        });
        match op {
            CollOp::Map => WTy::Array(
                Box::new(body_out_w),
                cur_ty.as_ref().and_then(array_size).unwrap_or(1),
            ),
            CollOp::Fold => body_out_w,
        }
    }

    pub(super) fn type_call(&mut self, sig: &FnSig, arg: &WTy, span: syn::SourceLoc) {
        // A zero-param fn is uncallable in Core (DESIGN §6.3): a `v -> g` call
        // stage always supplies an argument, so calling one is an arg-vs-declared
        // -input mismatch (L1201), NOT a literal-width conflict. Do not unify the
        // arg against `Unit` (that would pin a literal var and misreport L1203 —
        // ATK-18).
        // `() -> g` calls a zero-parameter fn with the Unit witness.
        if sig.params.is_empty() && matches!(arg, WTy::Known(Ty::Unit)) {
            return;
        }
        if sig.params.is_empty() {
            self.diags.push(diag(
                LCode::TypeMismatch,
                span,
                "function takes no input; the call supplies an argument".to_string(),
            ));
            return;
        }
        // unify arg with the product of param tys (single → that ty).
        let in_ty = if sig.params.len() == 1 {
            sig.params[0].clone()
        } else {
            Ty::Tuple(sig.params.clone())
        };
        self.unify_known(arg, &in_ty, span);
    }

    /// D1 result ty for a `zip` stage: `([A;n],[B;n]) → [(A,B);n]` (ADR-0018).
    /// Unknown unless `cur` resolves to a 2-tuple of equal-size arrays — emission
    /// owns the misuse L-codes (L1606/L1607/L1608); D1 only threads the ty so a
    /// downstream `map` sees the tuple element.
    pub(super) fn zip_result(&mut self, cur: &WTy) -> WTy {
        let Some(Ty::Tuple(cs)) = self.wty_to_ty(cur) else {
            return WTy::Unknown;
        };
        if let [
            Ty::Array { elem: ae, size: an },
            Ty::Array { elem: be, size: bn },
        ] = cs.as_slice()
            && an == bn
        {
            return WTy::Known(Ty::Array {
                elem: Box::new(Ty::Tuple(vec![(**ae).clone(), (**be).clone()])),
                size: *an,
            });
        }
        WTy::Unknown
    }

    /// D1 result ty for an `enumerate` stage: `[A;n] → [(i32,A);n]` (ADR-0018).
    pub(super) fn enumerate_result(&mut self, cur: &WTy) -> WTy {
        match self.wty_to_ty(cur) {
            Some(Ty::Array { elem, size }) => WTy::Known(Ty::Array {
                elem: Box::new(Ty::Tuple(vec![Ty::i32(), (*elem).clone()])),
                size,
            }),
            _ => WTy::Unknown,
        }
    }
}
