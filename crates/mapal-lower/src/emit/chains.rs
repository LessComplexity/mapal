//! Emitter methods: chain emission and return writes.

use super::*;

impl Emitter<'_> {
    // --- chains -------------------------------------------------------------

    /// Lower a chain head-to-tail. Returns the final wire object (if any).
    pub(super) fn emit_chain(
        &mut self,
        fb: &mut FnBuilder,
        chain: &Chain,
        ctx: ChainCtx,
    ) -> ER<Option<ObjectId>> {
        let stages = &chain.stages;
        // The head's Dest (canonical ret-write, pin 2): a value-producing head
        // whose only continuation is `-> ret` (directly, or no stages in a return
        // context) targets Return. A bare pre-existing wire (Var/param/const) uses
        // `output()` at the `-> ret` stage instead.
        let head_is_value = chain
            .head
            .as_ref()
            .map(|h| expr_is_value_producing(&h.kind) && !self.is_size_literal(h))
            .unwrap_or(false);
        // A bare pre-existing-object tail (`fn f(x: i32) -> i32 { x }` / a body
        // tail `{ … x }`) must route through `output()` / `emit_ret_existing`
        // (LD21 + §8.1: bare pre-existing → Output), not pass `Dest::Ret` to the
        // `Var` arm (which would ignore it → seal `RetSlotMissing` / L1901; ATK-07a).
        let tail_is_return = matches!(ctx, ChainCtx::FnBodyReturn { .. } | ChainCtx::BodyReturn);
        if stages.is_empty()
            && !head_is_value
            && tail_is_return
            && let Some(h) = &chain.head
        {
            let wire = self.emit_expr(fb, h)?;
            self.emit_ret_existing(fb, wire, None, h.span, &ctx)?;
            return Ok(None);
        }
        let head_dest = if stages.is_empty() {
            self.lookahead_dest(None, &ctx, true)?
        } else if head_is_value {
            // §8.1's lookahead applies to heads too: `-> ret` targets Return
            // (pin 2), and a following binding names the head's object (LD17 —
            // `[…] -> image: …` labels the pack; `acc + i -> acc` labels the new
            // SSA object). The symbol itself still binds at the Bind/Var stage.
            self.lookahead_dest(stages.first(), &ctx, false)?
        } else {
            Dest::Fresh(None)
        };
        let head_to_dest = matches!(head_dest, Dest::Ret { .. });

        // Seed the wire.
        let mut cur: Option<ObjectId> = match &chain.head {
            // `()` (plan-time-builtin) is the wire-LESS head: it emits nothing
            // and seeds `None`, so `() -> time` reaches the `time` stage with
            // no source, exactly as the op's `IoToken → (IoToken, f64)`
            // signature wants. Any other stage then reports L1301 (no wire),
            // and `()` in a value position hits the `ExprKind::Unit` arm of
            // `emit_expr_dest`.
            Some(h) if matches!(h.kind, ExprKind::Unit) => None,
            Some(h) => {
                let r = self.emit_expr_dest(fb, h, head_dest.clone())?;
                if head_to_dest {
                    return Ok(None); // head wrote to Return; chain done.
                }
                Some(r)
            }
            None => match &ctx {
                ChainCtx::HeadlessSeed(seed) => Some(*seed),
                _ => None,
            },
        };

        // Whether the wire was produced by a value-producing primitive that took
        // the lookahead Dest (so a following `-> ret`/bind is already wired).
        let mut prev_to_dest = false;
        let mut i = 0;
        while i < stages.len() {
            let stage = &stages[i];
            let next = stages.get(i + 1);
            let is_last = i + 1 == stages.len();
            // A `Ret` stage following a value-producing stage is already wired by
            // the lookahead Dest (pin 2); it is a no-op marker. Only a `-> ret`
            // of a *pre-existing* wire (no preceding value-producing stage) needs
            // `output()` (§8.1).
            if let StageKind::Ret { proj } = &stage.kind {
                if !is_last {
                    return Err(diag(
                        LCode::RetMidChain,
                        stages[i + 1].span,
                        "stages after `-> ret` in one chain",
                    ));
                }
                if prev_to_dest {
                    return Ok(None); // already written to Return.
                }
                // A value-less `-> ret` (no wire): the statement-level / fn-body
                // bare return marker (ATK-09/ATK-10). The catalogue permits it.
                // §8.3: in a pure Unit fn it is a no-op; in an effectful B-absent
                // fn it writes the current token to Return (and consumes the
                // register — later token use → L1307); a non-Unit output was
                // already L1306'd in D1.
                let Some(wire) = cur else {
                    self.emit_valueless_ret(fb, *proj, stage.span)?;
                    return Ok(None);
                };
                self.emit_ret_existing(fb, wire, *proj, stage.span, &ctx)?;
                return Ok(None);
            }
            let value_producing = self.stage_writes_value(&stage.kind);
            cur = self.emit_stage(fb, stage, next, is_last, cur, &ctx)?;
            // The wire went to Return iff this stage was value-producing and the
            // next stage is `-> ret` (the lookahead chose Dest::Ret). In an
            // effectful B-present fn the producer took Fresh (a full-tuple writer
            // packs the token at the `Ret` marker — §8.1 / LD18), so the marker
            // must still reach `emit_ret_existing`.
            prev_to_dest = value_producing
                && !self.effectful_with_ret()
                && matches!(next.map(|n| &n.kind), Some(StageKind::Ret { .. }));
            i += 1;
        }
        Ok(cur)
    }

    /// Whether a stage emits a fresh-value primitive that takes the lookahead
    /// `Dest` (so a following `-> ret` is pre-wired; pin 2). A bare-name `Expr`
    /// stage counts for a pure fn or pure builtin; bindings and `print` do not.
    pub(super) fn stage_writes_value(&self, kind: &StageKind) -> bool {
        match kind {
            StageKind::OpShorthand { .. }
            | StageKind::Guard(_)
            | StageKind::MapFold { .. }
            | StageKind::Fanout { .. } => true,
            StageKind::Expr(e) => {
                if let ExprKind::Generic { name, .. } = &e.kind {
                    // ADR-0051: a conversion or a pure generic instance.
                    let text = name_text(self.source, *name);
                    return Verb::from_name(text).is_some()
                        || self.fn_sigs.get(text).is_some_and(|s| !s.effectful);
                }
                if let ExprKind::Var(n) = &e.kind {
                    let text = name_text(self.source, *n);
                    if crate::is_pure_builtin(text)
                        || crate::is_time_builtin(text)
                        || crate::is_read_builtin(text)
                    {
                        return true;
                    }
                    if let Some(sig) = self.fn_sigs.get(text) {
                        return !sig.effectful;
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// Lower one stage. `next` is the following stage (one-stage lookahead, LD2).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_stage(
        &mut self,
        fb: &mut FnBuilder,
        stage: &Stage,
        next: Option<&Stage>,
        is_last: bool,
        cur: Option<ObjectId>,
        ctx: &ChainCtx,
    ) -> ER<Option<ObjectId>> {
        let sp = stage.span;
        match &stage.kind {
            StageKind::Expr(e) => self.emit_expr_stage(fb, e, next, cur, ctx),
            StageKind::OpShorthand { expr } => {
                let wire =
                    cur.ok_or_else(|| diag(LCode::HeadlessChain, sp, "shorthand with no wire"))?;
                let dest = self.lookahead_dest(next, ctx, is_last)?;
                let r = self.emit_op_shorthand(fb, expr, wire, dest)?;
                self.maybe_bind_next(fb, next, r);
                Ok(Some(r))
            }
            StageKind::Bind { mut_span, name, ty } => {
                let wire =
                    cur.ok_or_else(|| diag(LCode::HeadlessChain, sp, "bind with no wire"))?;
                let annotated = ty.as_ref().and_then(|annot| {
                    self.types
                        .resolve_sized(self.source, annot, self.sizes, &mut Vec::new())
                });
                // ADR-0048 D5: a binding annotated with `_`'s type reads
                // through it.
                let wire = match &annotated {
                    Some(t) => self.coerce(fb, wire, t, sp)?,
                    None => wire,
                };
                let bn = name_text(self.source, *name).to_string();
                // An annotation states the value's type (ADR-0051 D3 reads it
                // as a determination): a value of another type is a mismatch,
                // never a binding relabelled with the annotation.
                if let (Some(want), Some(annot)) = (&annotated, ty) {
                    let have = self.ty_of(fb, wire);
                    if &have != want {
                        return Err(diag(
                            LCode::TypeMismatch,
                            annot.span,
                            format!(
                                "`{bn}` is annotated `{}`, but the value bound is `{}`",
                                mapal_ir::ty_label(want),
                                mapal_ir::ty_label(&have)
                            ),
                        ));
                    }
                }
                let bty = annotated.unwrap_or_else(|| self.ty_of(fb, wire));
                self.bind_new(&bn, wire, bty, mut_span.is_some());
                Ok(Some(wire))
            }
            StageKind::Ret { .. } => {
                // `Ret` is handled directly in `emit_chain` (the lookahead-Dest
                // pin-2 logic needs the preceding stage's context).
                Err(internal(sp, "ret stage reached emit_stage"))
            }
            StageKind::LoopJump => Err(diag(
                LCode::JumpMisplaced,
                sp,
                "`-> loop` outside a routing-guard jump arm",
            )),
            StageKind::Guard(arms) => {
                let wire =
                    cur.ok_or_else(|| diag(LCode::HeadlessChain, sp, "guard with no wire"))?;
                // Routing guards are handled by the loop recipe; a routing guard
                // reaching here is misplaced: inside a loop body but not the
                // final item → L1407; outside any loop → L1304.
                if arms
                    .iter()
                    .any(|a| crate::typing::arm_is_routing(&a.payload))
                {
                    if self.loop_depth > 0 {
                        return Err(diag(
                            LCode::RoutingGuardShape,
                            sp,
                            "routing guard is not the loop body's final item",
                        ));
                    }
                    return Err(diag(
                        LCode::JumpMisplaced,
                        sp,
                        "routing guard outside a loop body",
                    ));
                }
                let dest = self.lookahead_dest(next, ctx, is_last)?;
                let r = self.emit_phi_guard(fb, arms, wire, dest, sp)?;
                self.maybe_bind_next(fb, next, r);
                Ok(Some(r))
            }
            StageKind::Fanout { branches, .. } => {
                let wire =
                    cur.ok_or_else(|| diag(LCode::HeadlessChain, sp, "fanout with no wire"))?;
                self.emit_fanout(fb, branches, wire, next, ctx, sp)
            }
            StageKind::MapFold { op, params, body } => {
                let wire =
                    cur.ok_or_else(|| diag(LCode::HeadlessChain, sp, "map/fold with no wire"))?;
                let dest = self.lookahead_dest(next, ctx, is_last)?;
                let r = self.emit_map_fold(fb, *op, params, body, stage, wire, dest)?;
                self.maybe_bind_next(fb, next, r);
                Ok(Some(r))
            }
            StageKind::SeqBlock(body) => {
                let wire = cur.ok_or_else(|| diag(LCode::HeadlessChain, sp, "seq with no wire"))?;
                let outer = self.prec;
                self.prec = ir_prec(body.compute, outer);
                let r = self.emit_seq_block(fb, body, wire, next, ctx, sp);
                self.prec = outer;
                r
            }
            StageKind::StmtBlock(_) | StageKind::Error(_) => {
                Err(diag(LCode::UncleanTree, sp, "unclean stage"))
            }
        }
    }

    /// An `Expr` stage: a Var resolving to a local (rebind/bind), a fn (call),
    /// or `print`; anything else is L1302/L1106.
    pub(super) fn emit_expr_stage(
        &mut self,
        fb: &mut FnBuilder,
        e: &Expr,
        next: Option<&Stage>,
        cur: Option<ObjectId>,
        ctx: &ChainCtx,
    ) -> ER<Option<ObjectId>> {
        match &e.kind {
            ExprKind::Var(n) => {
                let text = name_text(self.source, *n).to_string();
                let wire = cur;
                if let Some((bind, poisoned)) =
                    self.scope.resolve(&text).map(|(b, p)| (b.clone(), p))
                {
                    if poisoned {
                        return Err(diag(
                            LCode::ReadAfterLoop,
                            e.span,
                            format!(
                                "`{text}` is a loop-carried name read after its loop; bind the exit value instead"
                            ),
                        ));
                    }
                    // Existing local in stage position = a rebind (mut) target.
                    let wire = wire
                        .ok_or_else(|| diag(LCode::HeadlessChain, e.span, "rebind with no wire"))?;
                    if !bind.mutable {
                        // ADR-0027: `-> name` of a read capture is a fresh
                        // body-local shadow in the current frame, never a
                        // rebind of the shared capture (D2b — rebinding a
                        // capture the body has already READ is the typing
                        // walk's teaching L1108, upstream of emission). The
                        // shadow pops with its scope; reads outside it still
                        // resolve to the capture.
                        if self.captures.contains(&text) {
                            let ty = self.ty_of(fb, wire);
                            self.bind_new(&text, wire, ty, false);
                            return Ok(Some(wire));
                        }
                        return Err(diag(
                            LCode::AssignImmutable,
                            e.span,
                            format!("cannot reassign `{text}`: not `mut`"),
                        ));
                    }
                    // Rebind to a fresh object naming `text` (the wire's value).
                    // The wire value already lives in `wire`; alias the symbol.
                    let ty = self.ty_of(fb, wire);
                    self.rebind(&text, wire, ty, true);
                    Ok(Some(wire))
                } else if self.fn_ids.contains_key(&text) || self.mono.generics.contains_key(&text)
                {
                    // call (ADR-0047: a generic one calls its instance). `() -> g`:
                    // the wire-less head calls a zero-parameter fn with the Unit
                    // witness (its input is `Unit`).
                    let zero = self.fn_sigs.get(&text).is_some_and(|s| s.params.is_empty());
                    let wire = match wire {
                        None if zero => {
                            let o = fb
                                .constant(mapal_ir::Value::Unit, ir_loc(e.span))
                                .map_err(|err| ir_err(err, e.span))?;
                            self.record(o, Ty::Unit)
                        }
                        w => {
                            w.ok_or_else(|| diag(LCode::HeadlessChain, e.span, "call with no arg"))?
                        }
                    };
                    let (fid, callee) = match self.fn_ids.get(&text) {
                        Some(&fid) => (fid, text.clone()),
                        None => self.generic_callee(&text, wire, next, ctx, e.span, None)?,
                    };
                    let dest = self.lookahead_dest(next, ctx, next.is_none())?;
                    let r = self.emit_call(fb, fid, &callee, wire, dest, e.span)?;
                    if let Some(r) = r {
                        self.maybe_bind_next(fb, next, r);
                    }
                    Ok(r)
                } else if crate::is_print_builtin(&text) {
                    let wire = wire
                        .ok_or_else(|| diag(LCode::HeadlessChain, e.span, "print with no value"))?;
                    self.emit_print(fb, wire, text == "println", e.span)?;
                    Ok(None) // print is a sink; chain ends.
                } else if crate::is_read_builtin(&text) {
                    // ADR-0045: `"file" -> read -> name: [T; n]`.
                    let wire = wire.ok_or_else(|| {
                        diag(
                            LCode::ReadSource,
                            e.span,
                            "`read` takes a file path: write `\"file.bin\" -> read -> name: [T; n]`",
                        )
                    })?;
                    let r = self.emit_read(fb, wire, next, ctx, e.span)?;
                    self.maybe_bind_next(fb, next, r);
                    Ok(Some(r))
                } else if crate::is_time_builtin(&text) {
                    // `() -> time` (plan-time-builtin): the ONE stage that
                    // takes no wire — the source is the IO token, not a value.
                    if wire.is_some() {
                        return Err(diag(
                            LCode::ExprStage,
                            e.span,
                            "`time` takes no value: write `() -> time`",
                        ));
                    }
                    let r = self.emit_time(fb, next, ctx, e.span)?;
                    self.maybe_bind_next(fb, next, r);
                    Ok(Some(r))
                } else if crate::is_pure_builtin(&text) {
                    let wire = wire.ok_or_else(|| {
                        diag(LCode::HeadlessChain, e.span, "builtin stage with no source")
                    })?;
                    let dest = self.lookahead_dest(next, ctx, next.is_none())?;
                    let r = match text.as_str() {
                        "zip" => self.emit_zip(fb, wire, dest, e.span)?,
                        "enumerate" => self.emit_enumerate(fb, wire, dest, e.span)?,
                        "iota" => self.emit_iota_stage(fb, wire, dest, e.span)?,
                        "fill" => self.emit_fill_stage(fb, wire, dest, e.span)?,
                        t if mapal_ir::MathFn::from_name(t).is_some() => self.emit_math(
                            fb,
                            wire,
                            mapal_ir::MathFn::from_name(t).expect("math builtin"),
                            dest,
                            e.span,
                        )?,
                        _ => {
                            let verb = Verb::from_name(&text).expect("conversion builtin");
                            self.emit_convert(fb, verb, None, wire, next, ctx, dest, e.span)?
                        }
                    };
                    self.maybe_bind_next(fb, next, r);
                    Ok(Some(r))
                } else if let Some((verb, f)) = crate::formats::removed(&text) {
                    Err(crate::typing::removed_conversion(&text, verb, f, e.span))
                } else {
                    // Unbound bare name = a fresh binding (LD1): the wire's value
                    // takes this name. The previous stage's lookahead Dest already
                    // named the object; bind the symbol to it.
                    let wire = wire.ok_or_else(|| {
                        diag(LCode::HeadlessChain, e.span, "binding with no wire")
                    })?;
                    let ty = self.ty_of(fb, wire);
                    self.bind_new(&text, wire, ty, false);
                    Ok(Some(wire))
                }
            }
            ExprKind::Generic { name, args } => {
                // ADR-0051 D2: a stage with explicit generic arguments — a
                // generic function's instance, or a conversion.
                let text = name_text(self.source, *name).to_string();
                let local = self.scope.resolve(&text).is_some();
                if !local && self.mono.generics.contains_key(&text) {
                    let wire =
                        cur.ok_or_else(|| diag(LCode::HeadlessChain, e.span, "call with no arg"))?;
                    let (fid, callee) =
                        self.generic_callee(&text, wire, next, ctx, e.span, Some(args))?;
                    let dest = self.lookahead_dest(next, ctx, next.is_none())?;
                    let r = self.emit_call(fb, fid, &callee, wire, dest, e.span)?;
                    if let Some(r) = r {
                        self.maybe_bind_next(fb, next, r);
                    }
                    Ok(r)
                } else if let (false, Some(verb)) = (local, Verb::from_name(&text)) {
                    let wire = cur.ok_or_else(|| {
                        diag(LCode::HeadlessChain, e.span, "builtin stage with no source")
                    })?;
                    let dest = self.lookahead_dest(next, ctx, next.is_none())?;
                    let r =
                        self.emit_convert(fb, verb, Some(args), wire, next, ctx, dest, e.span)?;
                    self.maybe_bind_next(fb, next, r);
                    Ok(Some(r))
                } else {
                    Err(crate::typing::no_generic_args(
                        self.source,
                        &text,
                        local,
                        self.fn_sigs.contains_key(&text),
                        e.span,
                    ))
                }
            }
            ExprKind::Member { base, field } => {
                // fn.param → L1106; else L1302.
                if let (ExprKind::Var(bn), MemberField::Named(_)) = (&base.kind, field) {
                    let bt = name_text(self.source, *bn);
                    if self.is_fn(bt) {
                        return Err(diag(
                            LCode::NamedParamApplication,
                            e.span,
                            "named-parameter partial application is out of Core",
                        ));
                    }
                }
                Err(diag(
                    LCode::ExprStage,
                    e.span,
                    "member expression does not consume the wire",
                ))
            }
            _ => Err(diag(
                LCode::ExprStage,
                e.span,
                "expression stage does not consume the wire",
            )),
        }
    }

    /// The one-stage-lookahead Dest (LD2). Returns `ER` because a `ret.k` slot
    /// `> u32::MAX` is rejected here with L1205 (the Session-04 truncation
    /// lesson). In an effectful fn with a declared return B present, a value
    /// producer whose next stage is `-> ret` does NOT target `Dest::Ret`: it
    /// produces Fresh and the `Ret` marker packs `(tok, value)` via
    /// `emit_ret_existing` (DESIGN §8.1 / LD18). For the fn-body / map-fold tail
    /// in such a fn, likewise produce Fresh and let `emit_ret_existing` /
    /// `emit_tail_as_return` pack the token.
    pub(super) fn lookahead_dest(
        &self,
        next: Option<&Stage>,
        ctx: &ChainCtx,
        is_last: bool,
    ) -> ER<Dest> {
        if let Some(n) = next {
            match &n.kind {
                StageKind::Ret { proj } => {
                    if self.effectful_with_ret() {
                        // Full-tuple writer: produce Fresh, packed at the Ret marker.
                        return Ok(Dest::Fresh(None));
                    }
                    let slot = Self::ret_slot(*proj)?;
                    Ok(Dest::Ret { slot })
                }
                StageKind::Bind { name, .. } => {
                    Ok(Dest::Fresh(Some(name_text(self.source, *name).to_string())))
                }
                StageKind::Expr(e) => {
                    if let ExprKind::Var(n) = &e.kind {
                        let text = name_text(self.source, *n);
                        // bare name binding (unbound or mut) → name it.
                        if !self.is_fn(text)
                            && !crate::is_print_builtin(text)
                            && !crate::is_time_builtin(text)
                            && !crate::is_read_builtin(text)
                            && !crate::is_pure_builtin(text)
                        {
                            return Ok(Dest::Fresh(Some(text.to_string())));
                        }
                    }
                    Ok(Dest::Fresh(None))
                }
                _ => Ok(Dest::Fresh(None)),
            }
        } else if is_last {
            // No next stage: this is the chain tail. If the chain context is a
            // return position, target Ret — but an effectful B-present fn packs
            // the token at the marker / tail, so produce Fresh there too.
            match ctx {
                ChainCtx::FnBodyReturn { effectful: false } => Ok(Dest::Ret { slot: None }),
                ChainCtx::BodyReturn => Ok(Dest::Ret { slot: None }),
                _ => Ok(Dest::Fresh(None)),
            }
        } else {
            Ok(Dest::Fresh(None))
        }
    }

    /// If the next stage is a binding, bind the produced object to that name
    /// (the Dest already named it, but the symbol must rebind too).
    pub(super) fn maybe_bind_next(
        &mut self,
        fb: &mut FnBuilder,
        next: Option<&Stage>,
        obj: ObjectId,
    ) {
        // Bare-name `Expr` stages self-bind when processed (LD1); only an explicit
        // `Bind` stage needs pre-binding here (the Dest already named it).
        if let Some(Stage {
            kind: StageKind::Bind { mut_span, name, ty },
            ..
        }) = next
        {
            let bty = ty
                .as_ref()
                .and_then(|t| {
                    self.types
                        .resolve_sized(self.source, t, self.sizes, &mut Vec::new())
                })
                .unwrap_or_else(|| self.ty_of(fb, obj));
            let bn = name_text(self.source, *name).to_string();
            self.bind_new(&bn, obj, bty, mut_span.is_some());
        }
    }

    // --- return writes ------------------------------------------------------

    /// `x -> ret` / `x -> ret.k` for a (possibly pre-existing) wire object.
    pub(super) fn emit_ret_existing(
        &mut self,
        fb: &mut FnBuilder,
        wire: ObjectId,
        proj: Option<(u64, syn::SourceLoc)>,
        sp: syn::SourceLoc,
        ctx: &ChainCtx,
    ) -> ER<()> {
        // A `Str` value can only feed `print`; returning it is L1206.
        if matches!(self.tyq(wire), Ty::Str) {
            return Err(diag(
                LCode::StrOutsidePrint,
                sp,
                "string value used outside `print`",
            ));
        }
        // In an effectful fn with a declared return, each surface ret-write
        // consumes *the* final token, so at most one surface return site exists.
        // A second ret-write finds the token already consumed (None) — that is
        // L1307 (>1 surface ret-write), not the pure-output L1201 path (ATK-16).
        if self.token.is_none() && self.effectful_with_ret() {
            return Err(diag(
                LCode::EffectfulReturnShape,
                sp,
                "more than one surface return in an effectful function (each consumes the final token)",
            ));
        }
        let effectful = matches!(ctx, ChainCtx::FnBodyReturn { effectful: true })
            || (self.token.is_some() && self.in_effectful_fn());
        if effectful {
            // full-tuple writer: pack(tok, wire) → Dest::Ret{None}. ret.k banned.
            if proj.is_some() {
                return Err(diag(
                    LCode::EffectfulReturnShape,
                    sp,
                    "slot return in an effectful function",
                ));
            }
            let tok = self
                .token
                .ok_or_else(|| internal(sp, "effectful ret without token"))?;
            let r = self
                .pack(fb, &[tok, wire], Dest::Ret { slot: None }, sp)
                .map_err(|e| ir_err(e, sp))?;
            let _ = r;
            self.token = None;
        } else {
            // Reject a `ret.k` slot > u32::MAX (the Session-04 truncation lesson,
            // L1205) before casting — mirrors the `x.k` member guard.
            let slot = Self::ret_slot(proj)?;
            fb.output(wire, slot, ir_loc(sp))
                .map_err(|e| ir_err(e, sp))?;
        }
        Ok(())
    }

    pub(super) fn in_effectful_fn(&self) -> bool {
        // An instance is its generic's effect (ADR-0047 D7): its signature
        // carries the flag under its own name.
        self.fn_sigs.get(&self.owner_name).map_or_else(
            || self.effects.is_effectful(&self.owner_name),
            |s| s.effectful,
        )
    }

    /// A value-less `-> ret` marker (no wire), §8.3 (ATK-09/ATK-10). The
    /// catalogue permits this statement-level / loop-exit / fn-body form. A
    /// `ret.k` slot here makes no sense (no value to write a slot of) — L1306. In
    /// a pure **Unit**-output fn the marker is a no-op (zero Return writers is
    /// legal for Unit — I-RET). In an **effectful B-absent** fn it writes the
    /// current token to Return (`output(tok, None)`) and consumes the register;
    /// any later token use then finds `None` → L1307. A non-Unit declared output
    /// was already L1306'd in D1 (return completeness), so it never reaches here
    /// with a live wire; defensively re-report L1306.
    pub(super) fn emit_valueless_ret(
        &mut self,
        fb: &mut FnBuilder,
        proj: Option<(u64, syn::SourceLoc)>,
        sp: syn::SourceLoc,
    ) -> ER<()> {
        if proj.is_some() {
            return Err(diag(
                LCode::IncompleteReturn,
                sp,
                "value-less `-> ret.k` with no value to write",
            ));
        }
        let b_present = self
            .fn_sigs
            .get(&self.owner_name)
            .map(|s| s.ret.is_some())
            .unwrap_or(false);
        if self.in_effectful_fn() {
            if b_present {
                // Effectful B-present: a value-less ret cannot complete the
                // `(IoToken × B)` writer (no B value) — D1's L1306.
                return Err(diag(
                    LCode::IncompleteReturn,
                    sp,
                    "value-less `-> ret` in a function with a declared return",
                ));
            }
            // Effectful B-absent: write the current token to Return (the sole
            // writer is the final token — §8.1, golden g). Consume the register.
            let tok = self.token.ok_or_else(|| {
                // The token was already consumed by an earlier ret-terminal site
                // (ATK-08): a second surface return in an effectful fn.
                diag(
                    LCode::EffectfulReturnShape,
                    sp,
                    "token already consumed by an earlier return",
                )
            })?;
            fb.output(tok, None, ir_loc(sp))
                .map_err(|e| ir_err(e, sp))?;
            self.token = None;
            return Ok(());
        }
        // Pure fn. A non-Unit pure output with a value-less ret is L1306 (D1);
        // defensively re-report. A Unit pure output: no-op.
        if b_present {
            return Err(diag(
                LCode::IncompleteReturn,
                sp,
                "value-less `-> ret` in a function with a declared return",
            ));
        }
        Ok(())
    }

    /// Validate a `ret.k` slot index, rejecting any `k > u32::MAX` with L1205
    /// (the Session-04 truncation lesson; DESIGN §4 L1205). Returns the slot as
    /// `Option<u32>` (None for a bare `-> ret`). Mirrors the `x.k` member guard.
    pub(super) fn ret_slot(proj: Option<(u64, syn::SourceLoc)>) -> ER<Option<u32>> {
        match proj {
            Some((k, sp)) => {
                if k > u32::MAX as u64 {
                    return Err(diag(
                        LCode::SlotOutOfRange,
                        sp,
                        "slot index exceeds u32::MAX",
                    ));
                }
                Ok(Some(k as u32))
            }
            None => Ok(None),
        }
    }

    /// Whether the enclosing fn is effectful with a declared return B present.
    /// In this case every surface ret-write lowers as a full-tuple writer
    /// `pack(tok, value) -> Dest::Ret { slot: None }` (DESIGN §8.1 / LD18), so a
    /// value-producing primitive whose next stage is `-> ret` must NOT take
    /// `Dest::Ret` — it produces Fresh, and `emit_ret_existing` packs the token.
    pub(super) fn effectful_with_ret(&self) -> bool {
        self.in_effectful_fn()
            && self
                .fn_sigs
                .get(&self.owner_name)
                .map(|s| s.ret.is_some())
                .unwrap_or(false)
    }
}
