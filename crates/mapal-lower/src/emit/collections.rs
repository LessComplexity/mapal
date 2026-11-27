//! Emitter methods: map/fold and the collection builtins (ADR-0018).

use super::*;

impl Emitter<'_> {
    // --- map / fold ---------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_map_fold(
        &mut self,
        fb: &mut FnBuilder,
        op: CollOp,
        _params: &[syn::Name],
        _body: &Block,
        stage: &Stage,
        wire: ObjectId,
        dest: Dest,
    ) -> ER<ObjectId> {
        let sp = stage.span;
        let bid = *self
            .body_ids
            .get(&(sp.start, sp.end))
            .ok_or_else(|| internal(sp, "missing body id"))?;
        let sig = self.info.block(sp).cloned();
        // ADR-0027: the body's read captures (leading hidden input components),
        // resolved from the enclosing scope into the op's source product.
        let caps: Vec<(String, Ty)> = sig.as_ref().map(|s| s.captures.clone()).unwrap_or_default();
        let mut cap_objs: Vec<ObjectId> = Vec::with_capacity(caps.len());
        for (cap, _) in &caps {
            // mask-frontend: a struct capture's field `name.f…`: the struct's
            // projections along the path.
            let mut path = cap.split('.');
            let root = path.next().unwrap_or(cap);
            // The poison bit travels with the capture: a loop-carried name
            // read after its loop is L1107 here exactly as on the direct-read
            // path (`ExprKind::Var`) — resolving through `.map(|(b, _)| …)`
            // would discard it and let the read reach the interpreter.
            match self.scope.resolve(root) {
                Some((b, false)) => {
                    let mut obj = b.obj;
                    for f in path {
                        let Ty::Struct { fields, .. } = self.ty_of(fb, obj) else {
                            return Err(internal(sp, "a field capture of a non-struct"));
                        };
                        let idx = fields
                            .iter()
                            .position(|(n, _)| n == f)
                            .ok_or_else(|| internal(sp, "a field capture's field"))?;
                        obj = match self.packed.get(&obj) {
                            Some(c) => c[idx],
                            None => self.proj(fb, obj, idx as u32, Dest::Fresh(None), sp)?,
                        };
                    }
                    cap_objs.push(obj)
                }
                Some((_, true)) => {
                    return Err(diag(
                        LCode::ReadAfterLoop,
                        sp,
                        format!("`{root}` read after its loop; bind the exit value instead"),
                    ));
                }
                None => return Err(internal(sp, "capture unbound in the enclosing scope")),
            }
        }
        // ADR-0048 D5: a mapped struct reads through its `_` array; a fold's
        // `(seed, s)` reads `s` so.
        let wire = match op {
            CollOp::Map => self
                .read_default(fb, wire, |t| matches!(t, Ty::Array { .. }), sp)?
                .unwrap_or(wire),
            CollOp::Fold => match self.ty_of(fb, wire) {
                Ty::Tuple(ts)
                    if ts.len() == 2
                        && !matches!(ts[1], Ty::Array { .. })
                        && TypeTable::default_steps(&ts[1], |t| matches!(t, Ty::Array { .. }))
                            .is_some() =>
                {
                    let seed = self.proj(fb, wire, 0, Dest::Fresh(None), sp)?;
                    let s = self.proj(fb, wire, 1, Dest::Fresh(None), sp)?;
                    let arr = self
                        .read_default(fb, s, |t| matches!(t, Ty::Array { .. }), sp)?
                        .unwrap_or(s);
                    self.pack(fb, &[seed, arr], Dest::Fresh(None), sp)
                        .map_err(|e| ir_err(e, sp))?
                }
                _ => wire,
            },
        };
        match op {
            CollOp::Map => {
                let wty = self.ty_of(fb, wire);
                let size = match &wty {
                    Ty::Array { size, .. } => *size,
                    _ => {
                        let n =
                            default_note(&wty, &wty).map_or(String::new(), |n| format!(" ({n})"));
                        return Err(diag(
                            LCode::MapNonArray,
                            sp,
                            format!("map applied to a non-array{n}"),
                        ));
                    }
                };
                let out = Ty::Array {
                    elem: Box::new(sig.as_ref().map(|s| s.body_out.clone()).unwrap_or(Ty::Unit)),
                    size,
                };
                let o = if cap_objs.is_empty() {
                    fb.map(bid, wire, dest, ir_loc(sp))
                        .map_err(|e| ir_err(e, sp))?
                } else {
                    fb.map_captured(bid, &cap_objs, wire, dest, ir_loc(sp))
                        .map_err(|e| ir_err(e, sp))?
                };
                // Derives-from-merge (§7.3): the op's inputs are its source
                // product — the captures + the array — so the result derives
                // iff any of them does (a captured map may legitimately feed
                // a loop guard; L1503's derivation test).
                let mut inputs = cap_objs.clone();
                inputs.push(wire);
                self.prop(o, &inputs);
                Ok(self.record(o, out))
            }
            CollOp::Fold => {
                let wty = self.ty_of(fb, wire);
                match &wty {
                    Ty::Tuple(ts) if ts.len() == 2 && matches!(ts[1], Ty::Array { .. }) => {}
                    _ => return Err(diag(LCode::FoldShape, sp, "fold wire is not (init, array)")),
                }
                let out = sig.as_ref().map(|s| s.body_out.clone()).unwrap_or(Ty::Unit);
                let o = if cap_objs.is_empty() {
                    fb.fold(bid, wire, dest, ir_loc(sp))
                        .map_err(|e| ir_err(e, sp))?
                } else {
                    // `fold_captured` takes the seed/array separately: project
                    // them out of the packed wire.
                    let (seed_ty, arr_ty) = match &wty {
                        Ty::Tuple(ts) => (ts[0].clone(), ts[1].clone()),
                        _ => unreachable!("fold wire shape checked above"),
                    };
                    let seed = fb
                        .proj(wire, 0, Dest::Fresh(None), ir_loc(sp))
                        .map_err(|e| ir_err(e, sp))?;
                    self.record(seed, seed_ty);
                    let arr = fb
                        .proj(wire, 1, Dest::Fresh(None), ir_loc(sp))
                        .map_err(|e| ir_err(e, sp))?;
                    self.record(arr, arr_ty);
                    fb.fold_captured(bid, &cap_objs, seed, arr, dest, ir_loc(sp))
                        .map_err(|e| ir_err(e, sp))?
                };
                // Derives-from-merge (§7.3): inputs are the captures + the
                // packed (seed, array) wire — same rule as the map arm.
                let mut inputs = cap_objs.clone();
                inputs.push(wire);
                self.prop(o, &inputs);
                Ok(self.record(o, out))
            }
        }
    }

    // --- collection builtins (ADR-0018) -------------------------------------

    /// `zip` (ADR-0018): source is a 2-tuple `([A;n], [B;n])`, result `[(A,B);n]`.
    /// Independently re-derives the shape here (owning L1606/L1607/L1608) before
    /// calling the builder, which re-checks defensively (the builder is not the
    /// diagnostic surface — LD12). Projects the two arrays out of the tuple wire
    /// (Pair-then-primitive; the builder re-pairs, exactly as `binop`).
    pub(super) fn emit_zip(
        &mut self,
        fb: &mut FnBuilder,
        wire: ObjectId,
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let wty = self.ty_of(fb, wire);
        let comps = match &wty {
            Ty::Tuple(ts) if ts.len() == 2 => ts.clone(),
            _ => {
                return Err(diag(
                    LCode::ZipNonTuple,
                    sp,
                    "zip source is not a 2-tuple of arrays",
                ));
            }
        };
        let (a_elem, a_size) = match &comps[0] {
            Ty::Array { elem, size } => ((**elem).clone(), *size),
            _ => {
                return Err(diag(
                    LCode::ZipNonArray,
                    sp,
                    "zip component 0 is not an array",
                ));
            }
        };
        let (b_elem, b_size) = match &comps[1] {
            Ty::Array { elem, size } => ((**elem).clone(), *size),
            _ => {
                return Err(diag(
                    LCode::ZipNonArray,
                    sp,
                    "zip component 1 is not an array",
                ));
            }
        };
        if a_size != b_size {
            return Err(diag(
                LCode::ZipSizeMismatch,
                sp,
                format!("zip arrays differ in length: {a_size} vs {b_size}"),
            ));
        }
        // Project the two arrays out of the tuple wire, then hand to the builder.
        let lhs = self.proj(fb, wire, 0, Dest::Fresh(None), sp)?;
        let rhs = self.proj(fb, wire, 1, Dest::Fresh(None), sp)?;
        let out = Ty::Array {
            elem: Box::new(Ty::Tuple(vec![a_elem, b_elem])),
            size: a_size,
        };
        let o = fb
            .zip(lhs, rhs, dest, ir_loc(sp))
            .map_err(|e| ir_err(e, sp))?;
        Ok(self.record(o, out))
    }

    /// `enumerate` (ADR-0018): source `[A;n]`, result `[(i32,A);n]`. Owns L1609
    /// (non-array) and L1610 (n > i32::MAX — the index `i32` could not name every
    /// element, F4/SND-3 precedent); the builder re-derives the bound defensively.
    pub(super) fn emit_enumerate(
        &mut self,
        fb: &mut FnBuilder,
        wire: ObjectId,
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let wty = self.ty_of(fb, wire);
        let (elem, size) = match &wty {
            Ty::Array { elem, size } => ((**elem).clone(), *size),
            _ => {
                return Err(diag(
                    LCode::EnumerateNonArray,
                    sp,
                    "enumerate applied to a non-array",
                ));
            }
        };
        if size > i32::MAX as u64 {
            return Err(diag(
                LCode::EnumerateOverflow,
                sp,
                format!("enumerate array length {size} exceeds i32::MAX"),
            ));
        }
        let out = Ty::Array {
            elem: Box::new(Ty::Tuple(vec![Ty::i32(), elem])),
            size,
        };
        let o = fb
            .enumerate(wire, dest, ir_loc(sp))
            .map_err(|e| ir_err(e, sp))?;
        Ok(self.record(o, out))
    }

    /// ADR-0051 D4: `x -> to<…>` / `x -> widen<…>` — the prelude's generic
    /// signature inferred like any generic call (the explicit prefix, the
    /// argument's format, the expected type: [`Self::expected_flow`]), then
    /// realized by the existing operations: `Narrow` (ADR-0043 rounding),
    /// `Widen` (ADR-0029 widening; `sitofp` rounds an `i32` to nearest), or
    /// nothing (the identity, erased). `widen<f32>` of an `i32` is logged for
    /// the facts to prove exact once the IR is sealed. Owns L1614 (`widen`),
    /// L1616 (`to`) and, through the inference, L1708–L1710.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_convert(
        &mut self,
        fb: &mut FnBuilder,
        verb: Verb,
        args: Option<&[syn::SizeExpr]>,
        wire: ObjectId,
        next: Option<&Stage>,
        ctx: &ChainCtx,
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let decl = verb.decl();
        let explicit = match args {
            Some(a) => crate::sizes::explicit_args(
                self.source,
                a,
                self.sizes,
                crate::formats::PRELUDE,
                verb.name(),
                &decl.generics,
                sp,
            )?,
            None => Vec::new(),
        };
        let source = self.ty_of(fb, wire);
        let code = match verb {
            Verb::To => LCode::InvalidNarrow,
            Verb::Widen => LCode::InvalidWiden,
        };
        if Format::of(&source).is_none() {
            return Err(diag(
                code,
                sp,
                format!(
                    "`{}` converts a number; got `{}` (a map converts each element: \
                     `xs -> map {{ x -> x -> {}<…> }}`)",
                    verb.name(),
                    ty_name(&source),
                    verb.name()
                ),
            ));
        }
        let expected = self.expected_flow(next, ctx);
        let values = crate::sizes::infer(
            crate::formats::PRELUDE,
            self.types,
            decl,
            &source,
            expected.as_ref(),
            &explicit,
            sp,
        )?;
        let (Arg::Format(target), Arg::Format(from)) = (values[0], values[1]) else {
            return Err(internal(sp, "conversion arguments are formats"));
        };
        let realize = crate::formats::admit(verb, from, target).map_err(|m| diag(code, sp, m))?;
        let o = match realize {
            Realize::Identity => {
                // Erased (D4): the wire is the value. A return position takes
                // it as a pre-existing value.
                if let Dest::Ret { slot } = dest {
                    self.emit_ret_existing(fb, wire, slot.map(|k| (u64::from(k), sp)), sp, ctx)?;
                }
                return Ok(wire);
            }
            Realize::Narrow(n) => fb
                .narrow(wire, n, dest, ir_loc(sp))
                .map_err(|e| ir_err(e, sp))?,
            Realize::Widen | Realize::WidenProven => fb
                .widen(wire, target.ty(), dest, ir_loc(sp))
                .map_err(|e| ir_err(e, sp))?,
        };
        if realize == Realize::WidenProven {
            self.mono
                .log
                .borrow_mut()
                .widens
                .push(crate::convert::WidenCheck {
                    target: o,
                    source: from,
                    format: target,
                    span: sp,
                    owner: self.owner_name.clone(),
                });
        }
        self.prop(o, &[wire]);
        Ok(self.record(o, target.ty()))
    }

    /// `x -> exp` (ADR-0042), `x -> sqrt` / `sin` / `cos` (ADR-0046): the
    /// builtin of a float scalar, its width kept. Owns L1615 (any other
    /// source type).
    pub(super) fn emit_math(
        &mut self,
        fb: &mut FnBuilder,
        wire: ObjectId,
        f: mapal_ir::MathFn,
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let source = self.ty_of(fb, wire);
        let name = f.name();
        if !matches!(source, Ty::Float { bits: 32 | 64 }) {
            return Err(diag(
                LCode::InvalidExp,
                sp,
                format!(
                    "`{name}` takes an `f32` or `f64`, got `{}`; widen an integer first (`x -> widen<f32> -> {name}`)",
                    ty_name(&source)
                ),
            ));
        }
        let o = fb
            .math(f, wire, dest, ir_loc(sp))
            .map_err(|e| ir_err(e, sp))?;
        self.prop(o, &[wire]);
        Ok(self.record(o, source))
    }

    /// `n -> iota` (ADR-0031): `[i32; n]` of `0..n-1` — the count flows in and
    /// must be a literal `Constant` (the ADR-0029 static-n rule, builder-owned:
    /// `NonStaticCount` / `EnumerateIndexOverflow`). Owns the L1612 teaching
    /// diagnostic; the output ty is the builder's (never re-derived here).
    pub(super) fn emit_iota_stage(
        &mut self,
        fb: &mut FnBuilder,
        wire: ObjectId,
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let o = fb.iota(wire, dest, ir_loc(sp)).map_err(|e| match e {
            IrError::NonStaticCount | IrError::EnumerateIndexOverflow => diag(
                LCode::IotaArgs,
                sp,
                "the iota count must be a positive literal flowing in — \
                 write `4096 -> iota` (a runtime size is out of Core, ADR-0023 territory)",
            ),
            other => ir_err(other, sp),
        })?;
        let out = fb.ty_of(o);
        Ok(self.record(o, out))
    }

    /// `(x, n) -> fill` (ADR-0031): `[T; n]` with every element `x` (T from x).
    /// The wire is the assembled 2-tuple; the builder's `fill_from` (the S21
    /// replay entry) consumes it, minting nothing — slot-1 must feed from a
    /// literal `Constant` (`NonStaticCount`). Owns the L1613 teaching
    /// diagnostic.
    pub(super) fn emit_fill_stage(
        &mut self,
        fb: &mut FnBuilder,
        wire: ObjectId,
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        let o = fb.fill_from(wire, dest, ir_loc(sp)).map_err(|e| match e {
            IrError::NonStaticCount => diag(
                LCode::FillArgs,
                sp,
                "fill takes `(value, count)` with a positive literal count ≤ i32::MAX — \
                 write `(x, 4096) -> fill` (a runtime size is out of Core, ADR-0023 territory)",
            ),
            other => ir_err(other, sp),
        })?;
        let out = fb.ty_of(o);
        Ok(self.record(o, out))
    }
}
