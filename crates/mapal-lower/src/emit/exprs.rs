//! Emitter methods: expressions, calls, print and literals.

use super::*;

impl Emitter<'_> {
    // --- expressions --------------------------------------------------------

    /// Lower an expression to an object (DESIGN §8.2).
    pub(super) fn emit_expr(&mut self, fb: &mut FnBuilder, expr: &Expr) -> ER<ObjectId> {
        self.emit_expr_dest(fb, expr, Dest::Fresh(None))
    }

    /// Lower an expression, with the outermost primitive taking `dest`.
    pub(super) fn emit_expr_dest(
        &mut self,
        fb: &mut FnBuilder,
        expr: &Expr,
        dest: Dest,
    ) -> ER<ObjectId> {
        let sp = expr.span;
        match &expr.kind {
            ExprKind::Int(v) => {
                let ty = self.lit_ty(sp).unwrap_or_else(Ty::i32);
                let val = self.int_value(*v, &ty, sp, false)?;
                self.constant(fb, val, sp)
            }
            ExprKind::Float => {
                let ty = self.lit_ty(sp).unwrap_or_else(Ty::f64);
                let val = self.float_value(sp, &ty)?;
                self.constant(fb, val, sp)
            }
            ExprKind::Bool(b) => self.constant(fb, Value::Bool(*b), sp),
            // `()` (plan-time-builtin) produces NO object: it is the wire-less
            // chain head, and `emit_chain` seeds `cur = None` for it before any
            // expression is emitted. Reaching here means it was used as a value
            // (`() + 1`, `x -> f(())`, a bare `();`) — L1301, the same code as
            // any other chain position with no wire.
            ExprKind::Unit => Err(diag(
                LCode::HeadlessChain,
                sp,
                "`()` is not a value: its only use is the head of `() -> time`",
            )),
            ExprKind::Str => {
                // A Str constant is a legal object; only a non-`print` *use*
                // (pack/binop/ret of a non-Str) is L1206 — the builder's
                // `StrOutsidePrint` maps to L1206. `print` consumes it fine.
                let raw = src_slice(self.source, sp);
                let unesc = mapal_syntax::unescape_string(raw);
                self.constant(fb, Value::Str(unesc), sp)
            }
            ExprKind::Var(n) => {
                let text = name_text(self.source, *n);
                match self.scope.resolve(text) {
                    Some((b, false)) => Ok(b.obj),
                    Some((_, true)) => Err(diag(
                        LCode::ReadAfterLoop,
                        sp,
                        format!("`{text}` read after its loop; bind the exit value instead"),
                    )),
                    None if self.types.globals().consts.contains_key(text) => {
                        let c = self.types.globals().consts[text].clone();
                        self.global_const(fb, &c, sp)
                    }
                    None if self.sizes.contains_key(text) => {
                        // ADR-0047 D4: a size is an integer literal.
                        let v = self.sizes[text];
                        let ty = self.lit_ty(sp).unwrap_or_else(|| Ty::index_ty(v));
                        let val = self.int_value(v, &ty, sp, false)?;
                        self.constant(fb, val, sp)
                    }
                    None => {
                        if self.is_fn(text) {
                            Err(diag(
                                LCode::FunctionAsValue,
                                sp,
                                format!("function `{text}` used as a value"),
                            ))
                        } else {
                            Err(diag(
                                LCode::UnknownName,
                                sp,
                                format!("unresolved name `{text}`"),
                            ))
                        }
                    }
                }
            }
            ExprKind::Hole => Err(internal(sp, "bare Hole in expression position")),
            ExprKind::Unary { op, operand, .. } => match op {
                UnOp::Neg => {
                    // Fold negative literals (pin 3) — a size parameter is
                    // one (ADR-0047 D4).
                    let lit = match &operand.kind {
                        ExprKind::Int(v) => Some(*v),
                        ExprKind::Var(n) => self.size_value(name_text(self.source, *n)),
                        _ => None,
                    };
                    if let Some(v) = lit {
                        let ty = self.lit_ty(operand.span).unwrap_or_else(Ty::i32);
                        let val = self.int_value(v, &ty, operand.span, true)?;
                        return self.constant(fb, val, sp);
                    }
                    if let ExprKind::Float = &operand.kind {
                        let ty = self.lit_ty(operand.span).unwrap_or_else(Ty::f64);
                        let val = self.float_value_neg(operand.span, &ty)?;
                        return self.constant(fb, val, sp);
                    }
                    let x = self.emit_expr(fb, operand)?;
                    self.unop(fb, Operation::Neg, x, dest, sp)
                }
                UnOp::Not => {
                    let x = self.emit_expr(fb, operand)?;
                    self.unop(fb, Operation::Not, x, dest, sp)
                }
            },
            ExprKind::Binary {
                op,
                lhs,
                rhs,
                op_span,
            } => {
                // ADR-0047 D4: a size expression is one integer literal.
                if let Some(v) = self.size_fold(expr) {
                    let ty = self.lit_ty(sp).unwrap_or_else(Ty::i32);
                    let val = self.int_value(v.unsigned_abs() as u64, &ty, sp, v < 0)?;
                    return self.constant(fb, val, sp);
                }
                let l = self.emit_expr(fb, lhs)?;
                let r = self.emit_expr(fb, rhs)?;
                self.binop(fb, map_binop(*op), l, r, dest, *op_span)
            }
            ExprKind::Member { base, field } => {
                let b = self.emit_expr(fb, base)?;
                let bty = self.ty_of(fb, b);
                match field {
                    MemberField::Named(fname) => {
                        let fn_ = name_text(self.source, *fname);
                        let has = |t: &Ty| match t {
                            Ty::Struct { fields, .. } => fields.iter().any(|(n, _)| n == fn_),
                            _ => false,
                        };
                        // ADR-0048 D5: a field of `_` reads through it.
                        let b = self.read_default(fb, b, has, sp)?.unwrap_or(b);
                        let bty = self.ty_of(fb, b);
                        let idx = match &bty {
                            Ty::Struct { fields, .. } => fields.iter().position(|(n, _)| n == fn_),
                            _ => None,
                        };
                        match idx {
                            Some(i) => self.proj(fb, b, i as u32, dest, sp),
                            None => Err(diag(
                                LCode::UnknownField,
                                sp,
                                format!("`{fn_}` is not a field"),
                            )),
                        }
                    }
                    MemberField::Index { value, span } => {
                        if matches!(bty, Ty::Array { .. }) {
                            return Err(diag(
                                LCode::NotAProduct,
                                *span,
                                "index on an array; use `arr[k]`",
                            ));
                        }
                        if *value > u32::MAX as u64 {
                            return Err(diag(
                                LCode::SlotOutOfRange,
                                *span,
                                "index exceeds u32::MAX",
                            ));
                        }
                        self.proj(fb, b, *value as u32, dest, sp)
                    }
                }
            }
            ExprKind::Index { base, index } => {
                let a = self.emit_expr(fb, base)?;
                let i = self.emit_expr(fb, index)?;
                self.index(fb, a, i, dest, sp)
            }
            ExprKind::Tuple(es) => {
                let mut comps = Vec::with_capacity(es.len());
                for e in es {
                    comps.push(self.emit_expr(fb, e)?);
                }
                self.pack(fb, &comps, dest, sp).map_err(|e| ir_err(e, sp))
            }
            ExprKind::Array(es) => {
                if es.is_empty() {
                    return Err(diag(LCode::EmptyArray, sp, "empty array literal"));
                }
                let mut comps = Vec::with_capacity(es.len());
                for e in es {
                    comps.push(self.emit_expr(fb, e)?);
                }
                self.pack_array(fb, &comps, dest, sp)
            }
            ExprKind::Struct { name, fields } => {
                let tname = name_text(self.source, *name);
                // ADR-0048 D5: an alias's literal `T { _: v }` is `v`.
                if let Some(td) = self.types.decl(tname)
                    && let [only] = td.fields.as_slice()
                    && name_text(self.source, only.name) == "_"
                    && let [fi] = fields.as_slice()
                    && name_text(self.source, fi.name) == "_"
                    && let Some(v) = &fi.value
                {
                    return self.emit_expr(fb, v);
                }
                // ADR-0048 D4: a size-generic type's literal takes its sizes
                // from its fields' types.
                let generic = self
                    .types
                    .decl(tname)
                    .filter(|td| !td.generics.is_empty())
                    .cloned();
                let sty = match &generic {
                    None => self.types.get(tname).cloned().ok_or_else(|| {
                        diag(LCode::UnknownType, sp, format!("unknown type `{tname}`"))
                    })?,
                    Some(td) => {
                        let mut objs: BTreeMap<String, ObjectId> = BTreeMap::new();
                        for fi in fields {
                            let fname = name_text(self.source, fi.name).to_string();
                            if let Some(e) = &fi.value {
                                objs.insert(fname, self.emit_expr(fb, e)?);
                            } else if let Some((b, false)) = self.scope.resolve(&fname) {
                                objs.insert(fname, b.obj);
                            }
                        }
                        let mut tys = Vec::new();
                        for f in &td.fields {
                            let fname = name_text(self.source, f.name);
                            let o = objs.get(fname).copied().ok_or_else(|| {
                                diag(LCode::TypeMismatch, sp, format!("missing field `{fname}`"))
                            })?;
                            tys.push(self.ty_of(fb, o));
                        }
                        let values =
                            crate::sizes::infer_type(self.source, self.types, td, &tys, sp)?;
                        let ty = self
                            .types
                            .instance(self.source, tname, &values, sp)
                            .ok_or_else(|| internal(sp, "generic type instance"))?;
                        let decl = match &ty {
                            Ty::Struct { fields, .. } => fields.clone(),
                            _ => return Err(internal(sp, "non-struct generic literal")),
                        };
                        let mut comps = Vec::with_capacity(decl.len());
                        for (fname, fty) in &decl {
                            let o = objs[fname];
                            comps.push(self.coerce(fb, o, fty, sp)?);
                        }
                        return self
                            .pack_struct(fb, ty, &comps, dest, sp)
                            .map_err(|e| ir_err(e, sp));
                    }
                };
                let decl = match &sty {
                    Ty::Struct { fields, .. } => fields.clone(),
                    _ => return Err(internal(sp, "non-struct in struct ctor")),
                };
                // Evaluate inits, reorder to decl order.
                let mut by_name: BTreeMap<String, ObjectId> = BTreeMap::new();
                for fi in fields {
                    let fname = name_text(self.source, fi.name).to_string();
                    let v = match &fi.value {
                        Some(e) => self.emit_expr(fb, e)?,
                        None => {
                            // pun: read the symbol.
                            match self.scope.resolve(&fname) {
                                Some((b, false)) => b.obj,
                                _ => {
                                    return Err(diag(
                                        LCode::UnknownName,
                                        fi.span,
                                        format!("`{fname}` not in scope"),
                                    ));
                                }
                            }
                        }
                    };
                    by_name.insert(fname, v);
                }
                let mut comps = Vec::with_capacity(decl.len());
                for (fname, _) in &decl {
                    match by_name.get(fname) {
                        Some(&o) => comps.push(o),
                        None => {
                            return Err(diag(
                                LCode::TypeMismatch,
                                sp,
                                format!("missing field `{fname}`"),
                            ));
                        }
                    }
                }
                self.pack_struct(fb, sty, &comps, dest, sp)
                    .map_err(|e| ir_err(e, sp))
            }
            // ADR-0031: `Call` is always P0108-rejected upstream (the ADR-0029
            // carve is gone — iota/fill are pipeline stages); no clean tree
            // reaches here with one.
            ExprKind::Call { .. } | ExprKind::Question(_) | ExprKind::Error => {
                Err(diag(LCode::UncleanTree, sp, "unclean expression node"))
            }
            // The parser builds explicit arguments in stage position only.
            ExprKind::Generic { .. } => Err(diag(
                LCode::ExprStage,
                sp,
                "explicit generic arguments `name<…>` stand only as a stage (ADR-0051 D2)",
            )),
        }
    }

    /// Lower an OpShorthand expr where the leftmost Hole leaf is `wire`.
    pub(super) fn emit_op_shorthand(
        &mut self,
        fb: &mut FnBuilder,
        expr: &Expr,
        wire: ObjectId,
        dest: Dest,
    ) -> ER<ObjectId> {
        self.emit_hole_expr(fb, expr, wire, dest)
    }

    pub(super) fn emit_hole_expr(
        &mut self,
        fb: &mut FnBuilder,
        expr: &Expr,
        wire: ObjectId,
        dest: Dest,
    ) -> ER<ObjectId> {
        let sp = expr.span;
        match &expr.kind {
            ExprKind::Hole => Ok(wire),
            ExprKind::Unary { op, operand, .. } => {
                let x = self.emit_hole_expr(fb, operand, wire, Dest::Fresh(None))?;
                let o = match op {
                    UnOp::Neg => Operation::Neg,
                    UnOp::Not => Operation::Not,
                };
                self.unop(fb, o, x, dest, sp)
            }
            ExprKind::Binary {
                op,
                lhs,
                rhs,
                op_span,
            } => {
                let l = self.emit_hole_expr(fb, lhs, wire, Dest::Fresh(None))?;
                let r = self.emit_hole_expr(fb, rhs, wire, Dest::Fresh(None))?;
                self.binop(fb, map_binop(*op), l, r, dest, *op_span)
            }
            _ => self.emit_expr_dest(fb, expr, dest),
        }
    }

    // --- calls / print ------------------------------------------------------

    pub(super) fn emit_call(
        &mut self,
        fb: &mut FnBuilder,
        callee: FuncId,
        callee_name: &str,
        arg: ObjectId,
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<Option<ObjectId>> {
        let csig = self.fn_sigs.get(callee_name).expect("callee sig").clone();
        // ADR-0048 D5: an argument reads through `_` where its own type does
        // not fit the parameter but `_`'s does.
        let want = match csig.params.as_slice() {
            [] => None,
            [one] => Some(one.clone()),
            ps => Some(Ty::Tuple(ps.to_vec())),
        };
        let before = self.ty_of(fb, arg);
        let arg = match &want {
            Some(w) => self.coerce(fb, arg, w, sp)?,
            None => arg,
        };
        // D5: an error at this call says whether `_` was read.
        let after = self.ty_of(fb, arg);
        let note = |mut d: syn::Diagnostic| {
            if let Some(n) = default_note(&before, &after) {
                d.message = format!("{} ({n})", d.message);
            }
            d
        };
        if csig.effectful {
            let tok = self.consume_token(sp)?;
            // arg' = pack(tok, arg), or the token alone for a zero-parameter
            // callee (`() -> g`: its input is the token).
            let arg2 = if csig.params.is_empty() {
                tok
            } else {
                self.pack(fb, &[tok, arg], Dest::Fresh(None), sp)
                    .map_err(|e| ir_err(e, sp))?
            };
            let r = fb
                .call(callee, arg2, Dest::Fresh(None), ir_loc(sp))
                .map_err(|e| note(ir_err(e, sp)))?;
            // Derives-from-merge: any derived operand ⇒ derived result (§7.3).
            // The call result derives from its argument (the packed `(tok, arg)`);
            // propagate to the result and both projs (ATK-12).
            self.prop(r, &[arg2, tok, arg]);
            if let Some(rty) = &csig.ret {
                // result is (IoToken, B). tok := proj(r,0); wire := proj(r,1).
                self.record(r, Ty::Tuple(vec![Ty::IoToken, rty.clone()]));
                let nt = self.proj(fb, r, 0, Dest::Fresh(None), sp)?;
                let wire = self.proj(fb, r, 1, Dest::Fresh(None), sp)?;
                // `proj` already propagates from its source `r`; both projs are
                // derived iff `r` is.
                self.token = Some(nt);
                let _ = dest;
                Ok(Some(wire))
            } else {
                // B absent: r : IoToken; tok := r directly (no proj). Chain ends.
                self.record(r, Ty::IoToken);
                self.token = Some(r);
                Ok(None)
            }
        } else {
            let out = csig.ret.clone().unwrap_or(Ty::Unit);
            let r = fb
                .call(callee, arg, dest, ir_loc(sp))
                .map_err(|e| note(ir_err(e, sp)))?;
            self.record(r, out);
            // Derives-from-merge: a pure-call result derives from its argument
            // (§7.3 / ATK-12), so a guard cond flowing through a pure call passes
            // the L1503 derivation test.
            self.prop(r, &[arg]);
            Ok(Some(r))
        }
    }

    /// The current token for an effect (print / effectful call). If the token
    /// register is `None` inside an *effectful* fn, the final token was already
    /// consumed by a ret-write or a ret-terminal loop exit (§8.5) — a later
    /// effect is L1307, not the internal "print in pure fn" path (ATK-08). In a
    /// genuinely pure fn the None is a lower bug (L1901).
    pub(super) fn consume_token(&self, sp: syn::SourceLoc) -> ER<ObjectId> {
        self.token.ok_or_else(|| {
            if self.in_effectful_fn() {
                diag(
                    LCode::EffectfulReturnShape,
                    sp,
                    "effect after the function's return consumed the token",
                )
            } else {
                internal(sp, "effect in a pure fn")
            }
        })
    }

    pub(super) fn emit_print(
        &mut self,
        fb: &mut FnBuilder,
        value: ObjectId,
        newline: bool,
        sp: syn::SourceLoc,
    ) -> ER<()> {
        let tok = self.consume_token(sp)?;
        // Printability: value ty must be numeric/bool/str.
        let vty = self.ty_of(fb, value);
        if !vty.is_printable() {
            return Err(diag(
                LCode::Unprintable,
                sp,
                format!("cannot print `{}`", ty_name(&vty)),
            ));
        }
        let res = if newline {
            fb.println(tok, value, ir_loc(sp))
        } else {
            fb.print(tok, value, ir_loc(sp))
        };
        let nt = res.map_err(|e| ir_err(e, sp))?;
        self.record(nt, Ty::IoToken);
        self.token = Some(nt);
        Ok(())
    }

    /// `time` (plan-time-builtin): consume the IO token, emit `TimeMs`, then
    /// split the `(IoToken, f64)` pair — slot 0 rebinds the token (the effect
    /// ordering, exactly as `print` rebinds it), slot 1 is the milliseconds
    /// value the chain carries on. The value proj takes the lookahead `Dest`,
    /// so `() -> time -> ret` writes Return directly.
    pub(super) fn emit_time(
        &mut self,
        fb: &mut FnBuilder,
        next: Option<&Stage>,
        ctx: &ChainCtx,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let tok = self.consume_token(sp)?;
        let pair = fb.time_ms(tok, ir_loc(sp)).map_err(|e| ir_err(e, sp))?;
        self.record(pair, Ty::Tuple(vec![Ty::IoToken, Ty::f64()]));
        let nt = fb
            .proj(pair, 0, Dest::Fresh(None), ir_loc(sp))
            .map_err(|e| ir_err(e, sp))?;
        self.record(nt, Ty::IoToken);
        self.token = Some(nt);
        let dest = self.lookahead_dest(next, ctx, next.is_none())?;
        let ms = fb
            .proj(pair, 1, dest, ir_loc(sp))
            .map_err(|e| ir_err(e, sp))?;
        self.record(ms, Ty::f64());
        Ok(ms)
    }

    /// `read` (ADR-0045): the wire is the path — a string literal, the only
    /// `Str` there is (L1618 otherwise); the next stage is the binding whose
    /// annotation types the array (L1617 when it is missing or is not a
    /// one-dimensional array of `i32`, `i64`, `u8`, `f32`, `f64`, `bf16` or
    /// `f16`). Then as `time`: consume the token, emit `Read`, split the
    /// `(IoToken, array)` pair — slot 0 rebinds the token, slot 1 is the
    /// array the binding names.
    pub(super) fn emit_read(
        &mut self,
        fb: &mut FnBuilder,
        wire: ObjectId,
        next: Option<&Stage>,
        ctx: &ChainCtx,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        if !matches!(self.tyq(wire), Ty::Str) {
            return Err(diag(
                LCode::ReadSource,
                sp,
                format!(
                    "`read` takes a string literal path, found `{}`",
                    ty_name(&self.tyq(wire))
                ),
            ));
        }
        let shape = "`read` needs its binding's type: an array or a struct / tuple \
                     whose every leaf is i32, i64, u8, f32, f64, bf16 or f16, with no \
                     padding (`\"file.bin\" -> read -> name: [T; n]`)";
        let annot = match next.map(|n| &n.kind) {
            Some(StageKind::Bind {
                ty: Some(annot), ..
            }) => annot,
            _ => return Err(diag(LCode::ReadAnnotation, sp, shape)),
        };
        let ty = self
            .types
            .resolve(self.source, annot, &mut Vec::new())
            .ok_or_else(|| diag(LCode::ReadAnnotation, sp, shape))?;
        if !mapal_ir::read::readable(&ty) {
            return Err(diag(
                LCode::ReadAnnotation,
                sp,
                format!("{shape}; found `{}`", ty_name(&ty)),
            ));
        }
        let tok = self.consume_token(sp)?;
        let pair = fb
            .read(tok, wire, &ty, ir_loc(sp))
            .map_err(|e| ir_err(e, sp))?;
        self.record(pair, Ty::Tuple(vec![Ty::IoToken, ty.clone()]));
        let nt = fb
            .proj(pair, 0, Dest::Fresh(None), ir_loc(sp))
            .map_err(|e| ir_err(e, sp))?;
        self.record(nt, Ty::IoToken);
        self.token = Some(nt);
        let dest = self.lookahead_dest(next, ctx, next.is_none())?;
        let arr = fb
            .proj(pair, 1, dest, ir_loc(sp))
            .map_err(|e| ir_err(e, sp))?;
        self.record(arr, ty);
        Ok(arr)
    }

    // --- literals -----------------------------------------------------------

    /// The resolved ty for the literal at `sp` (from D1), if recorded.
    pub(super) fn lit_ty(&self, sp: syn::SourceLoc) -> Option<Ty> {
        self.info.lit(sp).cloned()
    }

    /// Build the `Value` for an integer literal `v` of resolved ty `ty`,
    /// range-checking (L1202). `neg` negates (a folded `-lit`, pin 3).
    pub(super) fn int_value(&self, v: u64, ty: &Ty, sp: syn::SourceLoc, neg: bool) -> ER<Value> {
        macro_rules! oor {
            () => {
                diag(
                    LCode::LiteralOutOfRange,
                    sp,
                    format!(
                        "literal {}{v} does not fit `{}`",
                        if neg { "-" } else { "" },
                        ty_name(ty)
                    ),
                )
            };
        }
        match ty {
            Ty::Int {
                bits: 32,
                signed: true,
            } => {
                if neg {
                    let nv = -(v as i128);
                    if nv < i32::MIN as i128 {
                        return Err(oor!());
                    }
                    Ok(Value::I32(nv as i32))
                } else if v <= i32::MAX as u64 {
                    Ok(Value::I32(v as i32))
                } else {
                    Err(oor!())
                }
            }
            Ty::Int {
                bits: 64,
                signed: true,
            } => {
                if neg {
                    let nv = -(v as i128);
                    if nv < i64::MIN as i128 {
                        return Err(oor!());
                    }
                    Ok(Value::I64(nv as i64))
                } else if v <= i64::MAX as u64 {
                    Ok(Value::I64(v as i64))
                } else {
                    Err(oor!())
                }
            }
            Ty::Int {
                bits: 8,
                signed: false,
            } => {
                if neg {
                    return Err(oor!());
                }
                if v <= u8::MAX as u64 {
                    Ok(Value::U8(v as u8))
                } else {
                    Err(oor!())
                }
            }
            _ => Err(diag(
                LCode::LiteralTypeConflict,
                sp,
                format!("integer literal cannot have type `{}`", ty_name(ty)),
            )),
        }
    }

    /// Build the `Value` for a float literal at `sp` of resolved ty `ty`.
    pub(super) fn float_value(&self, sp: syn::SourceLoc, ty: &Ty) -> ER<Value> {
        self.float_value_signed(sp, ty, false)
    }
    pub(super) fn float_value_neg(&self, sp: syn::SourceLoc, ty: &Ty) -> ER<Value> {
        self.float_value_signed(sp, ty, true)
    }
    pub(super) fn float_value_signed(&self, sp: syn::SourceLoc, ty: &Ty, neg: bool) -> ER<Value> {
        let raw = src_slice(self.source, sp);
        match ty {
            Ty::Float { bits: 32 } => {
                let f: f32 = raw.parse().map_err(|_| {
                    diag(
                        LCode::LiteralOutOfRange,
                        sp,
                        format!("invalid f32 literal `{raw}`"),
                    )
                })?;
                Ok(Value::F32(if neg { -f } else { f }))
            }
            Ty::Float { bits: 64 } => {
                let f: f64 = raw.parse().map_err(|_| {
                    diag(
                        LCode::LiteralOutOfRange,
                        sp,
                        format!("invalid f64 literal `{raw}`"),
                    )
                })?;
                Ok(Value::F64(if neg { -f } else { f }))
            }
            _ => Err(diag(
                LCode::LiteralTypeConflict,
                sp,
                format!("float literal cannot have type `{}`", ty_name(ty)),
            )),
        }
    }
}
