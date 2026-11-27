//! Emitter methods: block and statement emission, fan-out blocks and `seq`.

use super::*;

impl Emitter<'_> {
    // --- block / statement emission ----------------------------------------

    pub(super) fn emit_block(&mut self, fb: &mut FnBuilder, block: &Block, ctx: BodyCtx) -> ER<()> {
        for item in &block.items {
            match item {
                BlockItem::Stmt(s) => self.emit_stmt(fb, s)?,
                BlockItem::Arm(_) => {}
            }
        }
        // The W11 tail chain is the final item (LD21).
        if let Some(tail) = &block.tail {
            match &ctx {
                BodyCtx::FnBody { ret, effectful } => {
                    // fn-body tail: virtual ret continuation when output non-Unit.
                    self.emit_tail_as_return(fb, tail, ret.clone(), *effectful)?;
                }
                BodyCtx::MapFoldBody { out_ty } => {
                    self.emit_body_tail(fb, tail, out_ty.clone())?;
                }
                BodyCtx::Plain => {
                    self.emit_chain(fb, tail, ChainCtx::Statement)?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn emit_stmt(&mut self, fb: &mut FnBuilder, stmt: &Stmt) -> ER<()> {
        match &stmt.kind {
            StmtKind::Chain(c) => {
                self.emit_chain(fb, c, ChainCtx::Statement)?;
                Ok(())
            }
            StmtKind::Bind(bnd) => {
                // `c[i] <- v` (ADR-0021 §2 wiring a): an *indexed* bind is a
                // rebind, never a fresh shadow. Resolve `c` as a rebindable local
                // (unbound → L1101, loop-poisoned → L1107, non-`mut` → L1104),
                // emit `Update(cur, i, v)`, then rebind the symbol to the fresh
                // array via the same `rebind()` a bare-Var chain rebind uses. Pure
                // — the builder threads no token (L1204/L1201 surface via `ir_err`
                // for a non-array target / element-ty clash; no new L-code needed).
                if let Some(idx_expr) = &bnd.index {
                    let bn = name_text(self.source, bnd.name).to_string();
                    let (bind, poisoned) = self
                        .scope
                        .resolve(&bn)
                        .map(|(b, p)| (b.clone(), p))
                        .ok_or_else(|| {
                            diag(LCode::UnknownName, bnd.span, format!("unknown name `{bn}`"))
                        })?;
                    if poisoned {
                        return Err(diag(
                            LCode::ReadAfterLoop,
                            bnd.span,
                            format!(
                                "`{bn}` is a loop-carried name updated after its loop; bind the exit value instead"
                            ),
                        ));
                    }
                    if !bind.mutable {
                        return Err(diag(
                            LCode::AssignImmutable,
                            bnd.span,
                            format!("cannot update `{bn}`: not `mut`"),
                        ));
                    }
                    let idx = self.emit_expr(fb, idx_expr)?;
                    let val = self.emit_expr(fb, &bnd.value)?;
                    let arr = self.update(
                        fb,
                        bind.obj,
                        idx,
                        val,
                        Dest::Fresh(Some(bn.clone())),
                        bnd.span,
                    )?;
                    let ty = self.ty_of(fb, arr);
                    self.rebind(&bn, arr, ty, true);
                    return Ok(());
                }
                let v = self.emit_expr(fb, &bnd.value)?;
                let ty = self.ty_of(fb, v);
                let bn = name_text(self.source, bnd.name).to_string();
                self.bind_new(&bn, v, ty, bnd.mut_span.is_some());
                Ok(())
            }
            StmtKind::Loop(l) => {
                // ADR-0041 D2: `loop: c { … }` — the body's products admit `c`.
                let outer = self.prec;
                self.prec = ir_prec(l.body.compute, outer);
                let r = self.emit_loop(fb, l);
                self.prec = outer;
                r
            }
            StmtKind::Error => Err(diag(LCode::UncleanTree, stmt.span, "error node")),
        }
    }

    /// Emit the fn-body tail as a return (a virtual `-> ret`), if the declared
    /// output is non-Unit; otherwise it is dead code (still walked for effects).
    pub(super) fn emit_tail_as_return(
        &mut self,
        fb: &mut FnBuilder,
        tail: &Chain,
        ret: Option<Ty>,
        effectful: bool,
    ) -> ER<()> {
        if ret.is_none() {
            // Unit output: the tail is dead code as a *value*; the final token (if
            // effectful) is written by `emit_fn`, not the tail (LD21 / ATK-17).
            // Walk it as a statement regardless of effectfulness — effects still
            // thread via `self.token`, and a fanout tail does not demand branch
            // values (the §6.2 yes|absent rows have no surface return).
            self.emit_chain(fb, tail, ChainCtx::Statement)?;
            return Ok(());
        }
        if effectful {
            // Effectful with B present: the tail value is a full-tuple writer —
            // produce it Fresh, then `pack(tok, value) -> Dest::Ret { slot: None }`
            // (§8.1 / LD18, LOWER-2). If the tail itself ends in `-> ret`, the Ret
            // marker already packed the token via `emit_ret_existing`.
            if chain_ends_in_ret(tail) {
                self.emit_chain(fb, tail, ChainCtx::FnBodyReturn { effectful: true })?;
                return Ok(());
            }
            // `RetValue` (not `Statement`): the tail is in return position, so a
            // tail-less `seq` here draws L1611 (ADR-0019 pin c), not the outer
            // L1306 fall-through. A valued tail is still handed back for the
            // token-pack below (RetValue writes nothing to Ret itself).
            let v = self.emit_chain(fb, tail, ChainCtx::RetValue)?;
            let value = v.ok_or_else(|| {
                diag(
                    LCode::IncompleteReturn,
                    tail.span,
                    "fn-body tail produces no value to return",
                )
            })?;
            let tok = self
                .token
                .ok_or_else(|| internal(tail.span, "effectful tail without token"))?;
            self.pack(fb, &[tok, value], Dest::Ret { slot: None }, tail.span)
                .map_err(|e| ir_err(e, tail.span))?;
            self.token = None;
            return Ok(());
        }
        // Non-Unit, pure: the tail value returns via the lookahead `Dest::Ret`.
        self.emit_chain(fb, tail, ChainCtx::FnBodyReturn { effectful: false })?;
        Ok(())
    }

    /// mask-frontend: struct capture `path` of type `ty` packed from its
    /// fields' captures ([`crate::typing::flatten_captures`]).
    pub(super) fn rebuild_record(
        &mut self,
        fb: &mut FnBuilder,
        path: &str,
        ty: &Ty,
        fields: &BTreeMap<String, ObjectId>,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let Ty::Struct { fields: decl, .. } = ty else {
            return fields
                .get(path)
                .copied()
                .ok_or_else(|| internal(sp, "a struct capture's field is captured"));
        };
        let mut comps = Vec::with_capacity(decl.len());
        for (f, ft) in decl {
            comps.push(self.rebuild_record(fb, &format!("{path}.{f}"), ft, fields, sp)?);
        }
        self.pack_struct(fb, ty.clone(), &comps, Dest::Fresh(None), sp)
            .map_err(|e| ir_err(e, sp))
    }

    pub(super) fn emit_body_tail(
        &mut self,
        fb: &mut FnBuilder,
        tail: &Chain,
        out_ty: Ty,
    ) -> ER<()> {
        let _ = out_ty;
        self.emit_chain(fb, tail, ChainCtx::BodyReturn)?;
        Ok(())
    }

    // --- fanout -------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_fanout(
        &mut self,
        fb: &mut FnBuilder,
        branches: &[Chain],
        source: ObjectId,
        next: Option<&Stage>,
        ctx: &ChainCtx,
        sp: syn::SourceLoc,
    ) -> ER<Option<ObjectId>> {
        // Branches are headless chains seeded with the fanout source, lowered in
        // the enclosing scope (no child scope). No join unless the chain
        // continues past the fanout.
        let continues =
            next.is_some() || matches!(ctx, ChainCtx::FnBodyReturn { .. } | ChainCtx::BodyReturn);
        let mut tails: Vec<ObjectId> = Vec::new();
        for br in branches {
            let r = self.emit_chain(fb, br, ChainCtx::HeadlessSeed(source))?;
            if continues {
                match r {
                    Some(o) => tails.push(o),
                    None => {
                        return Err(diag(
                            LCode::FanoutNoValue,
                            br.span,
                            "fanout branch produces no value",
                        ));
                    }
                }
            }
        }
        if !continues {
            return Ok(None);
        }
        // Materialize the join.
        if tails.len() == 1 {
            Ok(Some(tails[0])) // single-branch fanout: no pack (FAN-1).
        } else {
            let dest = self.lookahead_dest(next, ctx, next.is_none())?;
            let r = self.pack(fb, &tails, dest, sp).map_err(|e| ir_err(e, sp))?;
            self.maybe_bind_next(fb, next, r);
            Ok(Some(r))
        }
    }

    // --- seq ----------------------------------------------------------------

    /// `seq { … }` (ADR-0019): an ordered statement block in stage position.
    /// `seq` has **no IR footprint** (pin d) — its ordering guarantee *is* the
    /// token thread that source-order statement lowering already produces. The
    /// statements lower in order **in the enclosing scope** (no child scope —
    /// bindings escape, pin b); a headless chain statement seeds from the seq
    /// input `wire` (pin a — the old bare-chain branch form still means the
    /// same). The seq's value is its tail chain's value (pin c); a seq that
    /// continues (a following stage, or return position) with no tail value is
    /// L1611.
    pub(super) fn emit_seq_block(
        &mut self,
        fb: &mut FnBuilder,
        block: &Block,
        wire: ObjectId,
        next: Option<&Stage>,
        ctx: &ChainCtx,
        sp: syn::SourceLoc,
    ) -> ER<Option<ObjectId>> {
        // ponytail: a chain statement routes through `emit_chain(HeadlessSeed)`
        // (so a headless one seeds from the wire, pin a); Bind/Loop reuse
        // `emit_stmt`. This is the one place seq diverges from `emit_block`'s
        // non-seeding statement loop.
        for item in &block.items {
            if let BlockItem::Stmt(s) = item {
                match &s.kind {
                    StmtKind::Chain(c) => {
                        self.emit_chain(fb, c, ChainCtx::HeadlessSeed(wire))?;
                    }
                    _ => self.emit_stmt(fb, s)?,
                }
            }
        }
        let continues = next.is_some()
            || matches!(
                ctx,
                ChainCtx::FnBodyReturn { .. } | ChainCtx::BodyReturn | ChainCtx::RetValue
            );
        let Some(tail) = &block.tail else {
            if continues {
                return Err(diag(
                    LCode::SeqNoValue,
                    sp,
                    "chain continues past `seq` but the block has no tail value",
                ));
            }
            return Ok(None);
        };
        let value = self.emit_chain(fb, tail, ChainCtx::HeadlessSeed(wire))?;
        if !continues {
            return Ok(value);
        }
        let value = value.ok_or_else(|| {
            diag(
                LCode::SeqNoValue,
                tail.span,
                "`seq` tail produces no value but the chain continues",
            )
        })?;
        // The seq value comes bare off its tail (the tail lowered under
        // HeadlessSeed, so no lookahead `Dest::Ret` was taken). If the seq is the
        // final stage of a return-position chain, write the value to Return here —
        // the fn-body / map-fold-body tail handler ignores the returned wire and
        // relies on the last stage having targeted Ret. Otherwise hand the value
        // back as the wire: a following `-> ret` / bind / call stage consumes it
        // exactly as a pre-existing wire (so `SeqBlock` is not a
        // `stage_writes_value` — the `-> ret` marker routes through
        // `emit_ret_existing`).
        if next.is_none() && matches!(ctx, ChainCtx::FnBodyReturn { .. } | ChainCtx::BodyReturn) {
            self.emit_ret_existing(fb, value, None, tail.span, ctx)?;
            return Ok(None);
        }
        Ok(Some(value))
    }
}
