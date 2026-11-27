//! Emitter methods: declarations, function bodies and signatures (the emission walk's entry).

// ===========================================================================
//  The emission walk
// ===========================================================================

use super::*;

impl Emitter<'_> {
    /// Whether `text` names a function — fixed-size or generic (ADR-0047).
    pub(super) fn is_fn(&self, text: &str) -> bool {
        self.fn_ids.contains_key(text) || self.mono.generics.contains_key(text)
    }

    /// ADR-0047 D4: the value of a size parameter named `text` (no local
    /// shadows one: L1703).
    pub(super) fn size_value(&self, text: &str) -> Option<u64> {
        if self.scope.resolve(text).is_some() {
            return None;
        }
        self.sizes.get(text).copied()
    }

    /// D3 at a call to a generic function (emission side, authoritative): the
    /// arguments from the stage's explicit prefix (ADR-0051 D2), then the
    /// wire's exact type, then the type the result flows into
    /// ([`Self::expected_flow`]) — the instance's id, or, when this round did
    /// not declare it, the request that makes the next round declare it.
    pub(super) fn generic_callee(
        &mut self,
        text: &str,
        wire: ObjectId,
        next: Option<&Stage>,
        ctx: &ChainCtx,
        sp: syn::SourceLoc,
        args: Option<&[syn::SizeExpr]>,
    ) -> ER<(FuncId, String)> {
        let fd = self.mono.generics[text];
        let explicit = match args {
            Some(a) => crate::sizes::explicit_args(
                self.source,
                a,
                self.sizes,
                self.source,
                text,
                &fd.generics,
                sp,
            )?,
            None => Vec::new(),
        };
        let arg = self.tyq(wire);
        let expected = self.expected_flow(next, ctx);
        let values = crate::sizes::infer(
            self.source,
            self.types,
            fd,
            &arg,
            expected.as_ref(),
            &explicit,
            sp,
        )?;
        let name = crate::sizes::instance_name(text, &values);
        let inst: Instance = (text.to_string(), values.clone());
        let mut log = self.mono.log.borrow_mut();
        log.sites
            .insert((self.owner_name.clone(), sp.start, sp.end), values);
        match self.fn_ids.get(&name) {
            Some(&fid) => {
                log.called.insert(inst);
                Ok((fid, name))
            }
            None => {
                log.requested.insert(inst);
                Err(internal(
                    sp,
                    &format!("instance `{name}` not declared this round (ADR-0047 D5)"),
                ))
            }
        }
    }

    /// D3.4: the type a call's result is bound with — the next stage's
    /// annotation, or the owner's output at a `-> ret` (`ret.k`: that
    /// component).
    pub(super) fn expected_result(&self, next: Option<&Stage>) -> Option<Ty> {
        match next.map(|s| &s.kind) {
            Some(StageKind::Bind { ty: Some(t), .. }) => {
                self.types
                    .resolve_sized(self.source, t, self.sizes, &mut Vec::new())
            }
            Some(StageKind::Ret { proj: None }) => self
                .fn_sigs
                .get(&self.owner_name)
                .and_then(|s| s.ret.clone()),
            _ => None,
        }
    }

    /// ADR-0051 D3: the type a call's or a conversion's result flows into —
    /// the result's binding annotation, the owner's output at `-> ret`
    /// (`ret.k`: its component) or at the function body's tail (a return
    /// position, `ctx`), or the one parameter of a fixed function called next.
    /// The typing walk reads the same sources (`typing::Walk::expected_flow`).
    pub(super) fn expected_flow(&self, next: Option<&Stage>, ctx: &ChainCtx) -> Option<Ty> {
        if let Some(t) = self.expected_result(next) {
            return Some(t);
        }
        match next.map(|s| &s.kind) {
            None if matches!(ctx, ChainCtx::FnBodyReturn { .. } | ChainCtx::RetValue) => self
                .fn_sigs
                .get(&self.owner_name)
                .and_then(|s| s.ret.clone()),
            Some(StageKind::Ret { proj: Some((k, _)) }) => self
                .fn_sigs
                .get(&self.owner_name)
                .and_then(|s| s.ret.as_ref())
                .and_then(|r| u32::try_from(*k).ok().and_then(|k| r.component_ty(k)))
                .cloned(),
            Some(StageKind::Expr(e)) => match &e.kind {
                ExprKind::Var(n) => match self.fn_sigs.get(name_text(self.source, *n)) {
                    Some(sig)
                        if !sig.generic
                            && self.scope.resolve(name_text(self.source, *n)).is_none() =>
                    {
                        match sig.params.as_slice() {
                            [one] => Some(one.clone()),
                            _ => None,
                        }
                    }
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        }
    }

    /// D4: an integer expression over size parameters and literals, folded.
    pub(super) fn size_fold(&self, e: &Expr) -> Option<i128> {
        if self.sizes.is_empty() {
            return None;
        }
        crate::sizes::fold_value(self.source, e, &|t| self.size_value(t))
    }

    /// A size written as a value — `-n`, `rows * columns` — is a constant
    /// like `-4096` (a pre-existing object, never a lookahead `Dest`; D4).
    pub(super) fn is_size_literal(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Unary {
                op: UnOp::Neg,
                operand,
                ..
            } => matches!(&operand.kind, ExprKind::Var(n)
                if self.size_value(name_text(self.source, *n)).is_some()),
            ExprKind::Binary { .. } => self.size_fold(e).is_some(),
            _ => false,
        }
    }

    /// ADR-0048 D5: read `obj` through its `_` field(s) until its type fits
    /// `fits` — as is when it already does, else `obj._`, `obj._._`, … the
    /// first that fits; `None` when no step of the chain does.
    pub(super) fn read_default(
        &mut self,
        fb: &mut FnBuilder,
        obj: ObjectId,
        fits: impl Fn(&Ty) -> bool,
        sp: syn::SourceLoc,
    ) -> ER<Option<ObjectId>> {
        let ty = self.ty_of(fb, obj);
        let Some(k) = TypeTable::default_steps(&ty, fits) else {
            return Ok(None);
        };
        let mut cur = obj;
        for _ in 0..k {
            let idx = match self.ty_of(fb, cur) {
                Ty::Struct { fields, .. } => fields.iter().position(|(n, _)| n == "_"),
                _ => None,
            }
            .ok_or_else(|| internal(sp, "default field vanished"))?;
            cur = self.proj(fb, cur, idx as u32, Dest::Fresh(None), sp)?;
        }
        Ok(Some(cur))
    }

    /// D5 where a type is expected: `obj` as is when it has type `want`, else
    /// read through `_` (a tuple component by component) when that fits;
    /// otherwise `obj` unchanged (the consumer reports the mismatch).
    pub(super) fn coerce(
        &mut self,
        fb: &mut FnBuilder,
        obj: ObjectId,
        want: &Ty,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let ty = self.ty_of(fb, obj);
        if &ty == want {
            return Ok(obj);
        }
        if let (Ty::Tuple(have), Ty::Tuple(wants)) = (&ty, want)
            && have.len() == wants.len()
            && have
                .iter()
                .zip(wants)
                .all(|(h, w)| TypeTable::default_steps(h, |t| t == w).is_some())
        {
            let mut comps = Vec::with_capacity(wants.len());
            for (k, w) in wants.iter().enumerate() {
                let c = self.proj(fb, obj, k as u32, Dest::Fresh(None), sp)?;
                comps.push(self.coerce(fb, c, w, sp)?);
            }
            return self
                .pack(fb, &comps, Dest::Fresh(None), sp)
                .map_err(|e| ir_err(e, sp));
        }
        Ok(self
            .read_default(fb, obj, |t| t == want, sp)?
            .unwrap_or(obj))
    }

    /// ADR-0048 D3: a top-level scalar computed before the program runs, as
    /// the literal (or the annotated constant) it is.
    pub(super) fn global_const(
        &mut self,
        fb: &mut FnBuilder,
        c: &crate::globals::GlobalConst,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        use crate::globals::GlobalConst;
        let val = match c {
            GlobalConst::Int { value, pinned } => {
                let ty = pinned
                    .clone()
                    .or_else(|| self.lit_ty(sp))
                    .unwrap_or_else(|| Ty::index_ty(value.unsigned_abs() as u64));
                self.int_value(value.unsigned_abs() as u64, &ty, sp, *value < 0)?
            }
            GlobalConst::Float {
                value,
                pinned,
                literal,
            } => {
                let ty = pinned
                    .clone()
                    .or_else(|| self.lit_ty(sp))
                    .unwrap_or_else(Ty::f64);
                match (pinned, literal) {
                    (None, Some(lit)) => self.float_value(*lit, &ty)?,
                    _ if ty == Ty::f32() => Value::F32(*value as f32),
                    _ => Value::F64(*value),
                }
            }
            GlobalConst::Bool(b) => Value::Bool(*b),
        };
        self.constant(fb, val, sp)
    }

    /// Build a surface function body (D3).
    pub(super) fn emit_fn(&mut self, fd: &syn::FnDecl, mut fb: FnBuilder) -> ER<()> {
        let name = self.owner_name.clone();
        let sig = self.fn_sigs.get(&name).expect("sig").clone();
        let effectful = sig.effectful;
        let input = fb.input();

        // Bind params. Effectful: input is (IoToken, A) or IoToken alone.
        let (in_ty, _) = ir_signature(&sig);
        self.record(input, in_ty);
        if effectful {
            if sig.params.is_empty() {
                // input() IS the token.
                self.record(input, Ty::IoToken);
                self.token = Some(input);
            } else {
                // tok = proj(input, 0); params view = proj(input, 1).
                let tok = fb
                    .proj(input, 0, Dest::Fresh(None), ir_loc(fd.span))
                    .map_err(|e| ir_err(e, fd.span))?;
                self.record(tok, Ty::IoToken);
                self.token = Some(tok);
                self.bind_params(&mut fb, fd, &sig, input, Some(1))?;
            }
        } else {
            self.bind_params(&mut fb, fd, &sig, input, None)?;
        }

        // ADR-0048 D3: the top-level values this function reads that are not
        // literals — their chains, at the outermost scope (captured into
        // map / fold bodies like any enclosing value).
        let globals = self.types.globals();
        for i in globals.used_by(self.source, fd) {
            let chain = globals.items[i].chain.clone();
            self.emit_chain(&mut fb, &chain, ChainCtx::Statement)?;
        }

        // Emit the body block.
        self.emit_block(
            &mut fb,
            &fd.body,
            BodyCtx::FnBody {
                ret: sig.ret.clone(),
                effectful,
            },
        )?;

        // For an effectful B-absent fn whose token was not yet written to Return,
        // write the final token. (Pure fns: nothing to do; Unit returns may have
        // zero writers.)
        if effectful
            && sig.ret.is_none()
            && let Some(tok) = self.token
        {
            // Only write if Return has no writer yet (a bare-ret exit already
            // wrote it). We always write here for the straight-line case; a
            // ret-terminal loop exit consumes the token (self.token = None).
            fb.output(tok, None, ir_loc(fd.span))
                .map_err(|e| ir_err(e, fd.span))?;
        }

        fb.finish().map_err(|e| ir_err(e, fd.span))?;
        Ok(())
    }

    pub(super) fn bind_params(
        &mut self,
        fb: &mut FnBuilder,
        fd: &syn::FnDecl,
        sig: &FnSig,
        input: ObjectId,
        view_slot: Option<u32>,
    ) -> ER<()> {
        let multi = sig.params.len() >= 2;
        // Effectful: the params view = proj(input, view_slot), emitted once.
        let view = if let Some(vslot) = view_slot {
            let view_ty = if multi {
                Ty::Tuple(sig.params.clone())
            } else {
                sig.params[0].clone()
            };
            let v = fb
                .proj(input, vslot, Dest::Fresh(None), ir_loc(fd.span))
                .map_err(|e| ir_err(e, fd.span))?;
            self.record(v, view_ty);
            Some(v)
        } else {
            None
        };
        for (k, p) in fd.params.iter().enumerate() {
            let pname = name_text(self.source, p.name).to_string();
            let pty = sig.params[k].clone();
            let obj = if let Some(view) = view {
                if multi {
                    fb.proj(
                        view,
                        k as u32,
                        Dest::Fresh(Some(pname.clone())),
                        ir_loc(p.span),
                    )
                    .map_err(|e| ir_err(e, p.span))?
                } else {
                    view
                }
            } else if multi {
                fb.proj(
                    input,
                    k as u32,
                    Dest::Fresh(Some(pname.clone())),
                    ir_loc(p.span),
                )
                .map_err(|e| ir_err(e, p.span))?
            } else {
                input
            };
            self.record(obj, pty.clone());
            self.bind_new(&pname, obj, pty, p.mut_span.is_some());
        }
        Ok(())
    }

    /// Build a map/fold body fn (D2).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_body(
        &mut self,
        op: CollOp,
        params: &[syn::Name],
        body: &Block,
        body_in: &Ty,
        body_out: &Ty,
        caps: &[(String, Ty)],
        records: &[(String, Ty)],
        mut fb: FnBuilder,
    ) -> ER<()> {
        let input = fb.input();
        let span = body.span;
        // ADR-0027: bind each capture to its leading input projection (the
        // broadcast edge), then the visible params after them.
        let k = caps.len() as u32;
        self.captures = caps
            .iter()
            .map(|(c, _)| c.split('.').next().unwrap_or(c).to_string())
            .collect();
        self.record(input, body_in.clone());
        // mask-frontend: a struct capture's fields, by path, rebuilt below.
        let mut fields: BTreeMap<String, ObjectId> = BTreeMap::new();
        for (i, (cap, ty)) in caps.iter().enumerate() {
            let obj = fb
                .proj(input, i as u32, Dest::Fresh(None), ir_loc(span))
                .map_err(|e| ir_err(e, span))?;
            self.record(obj, ty.clone());
            match cap.contains('.') {
                true => {
                    fields.insert(cap.clone(), obj);
                }
                false => self.bind_new(&cap.clone(), obj, ty.clone(), false),
            }
        }
        for (name, ty) in records {
            let obj = self.rebuild_record(&mut fb, name, ty, &fields, span)?;
            self.bind_new(name, obj, ty.clone(), false);
        }
        match op {
            CollOp::Map => {
                // single param T = input (k=0: the whole input) or its last
                // component (k>0: after the captures).
                if let Some(p0) = params.first() {
                    let (obj, ty) = if k == 0 {
                        (input, body_in.clone())
                    } else {
                        let o = fb
                            .proj(input, k, Dest::Fresh(None), ir_loc(span))
                            .map_err(|e| ir_err(e, span))?;
                        let t = match body_in {
                            Ty::Tuple(ts) => ts[k as usize].clone(),
                            _ => return Err(internal(span, "captured map input not a product")),
                        };
                        self.record(o, t.clone());
                        (o, t)
                    };
                    let pn = name_text(self.source, *p0).to_string();
                    self.bind_new(&pn, obj, ty, false);
                }
            }
            CollOp::Fold => {
                // (caps…, Acc, T): captures at 0..k, acc at k, t at k+1.
                let acc = fb
                    .proj(input, k, Dest::Fresh(None), ir_loc(span))
                    .map_err(|e| ir_err(e, span))?;
                let t = fb
                    .proj(input, k + 1, Dest::Fresh(None), ir_loc(span))
                    .map_err(|e| ir_err(e, span))?;
                if let Ty::Tuple(ts) = body_in {
                    self.record(acc, ts[k as usize].clone());
                    self.record(t, ts[k as usize + 1].clone());
                    let p0 = name_text(self.source, params[0]).to_string();
                    let p1 = name_text(self.source, params[1]).to_string();
                    self.bind_new(&p0, acc, ts[k as usize].clone(), false);
                    self.bind_new(&p1, t, ts[k as usize + 1].clone(), false);
                }
            }
        }
        // Body: statements then tail = the body value, written canonically.
        self.emit_block(
            &mut fb,
            body,
            BodyCtx::MapFoldBody {
                out_ty: body_out.clone(),
            },
        )?;
        fb.finish().map_err(|e| ir_err(e, span))?;
        Ok(())
    }
}
