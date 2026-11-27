//! `FnBuilder` array primitives: zip, enumerate, widen / narrow, math, iota / fill, update, calls, map and fold (plain and captured).

use super::*;

impl FnBuilder<'_> {
    /// `Zip` (DESIGN §5.1; ADR-0018): `([A; n], [B; n]) → [(A, B); n]`. Builds the
    /// internal 2-tuple of the two arrays (Pair-then-primitive, exactly as
    /// `binop`), then applies `Zip`. Both arrays must share size `n`.
    pub fn zip(
        &mut self,
        lhs: ObjectId,
        rhs: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.check_obj(lhs)?;
        self.check_obj(rhs)?;
        let lty = self.ty_of(lhs);
        let rty = self.ty_of(rhs);
        let (a_elem, a_size) = match &lty {
            Ty::Array { elem, size } => ((**elem).clone(), *size),
            _ => return Err(IrError::NotAProduct),
        };
        let (b_elem, b_size) = match &rty {
            Ty::Array { elem, size } => ((**elem).clone(), *size),
            _ => return Err(IrError::NotAProduct),
        };
        if a_size != b_size {
            return Err(IrError::TypeMismatch {
                expected: lty.clone(),
                found: rty.clone(),
                loc,
            });
        }
        // Str never lives in a product/array (I9s) — such an array is already
        // unconstructible, but guard explicitly like `binop` for nested cases.
        if ty::ty_contains_str(&lty) || ty::ty_contains_str(&rty) {
            return Err(IrError::StrOutsidePrint);
        }
        // Internal (lhs, rhs) 2-tuple of the two arrays, then Zip.
        let pair_ty = Ty::Tuple(vec![lty.clone(), rty.clone()]);
        self.b.intake_ty(&pair_ty)?;
        let pair = self
            .b
            .mint_object(self.f, pair_ty, None, ObjectKind::Temporary, None, loc);
        self.b.add_edge(
            self.f,
            lhs,
            pair,
            Operation::Pair { slot: 0, arity: 2 },
            loc,
        );
        self.b.add_edge(
            self.f,
            rhs,
            pair,
            Operation::Pair { slot: 1, arity: 2 },
            loc,
        );
        let out_ty = Ty::Array {
            elem: Box::new(Ty::Tuple(vec![a_elem, b_elem])),
            size: a_size,
        };
        self.b.intake_ty(&out_ty)?;
        let plan = self.dest_target(&dest, &out_ty, loc)?;
        Ok(self.emit_to_dest(plan, pair, Operation::Zip, loc))
    }

    /// `Enumerate` (DESIGN §5.1; ADR-0018): `[A; n] → [(i32, A); n]`. Single
    /// source (direct edge, no internal pair). Rejects `n > i32::MAX` up front —
    /// the index `i32` could not name every element (F4/SND-3 precedent).
    pub fn enumerate(
        &mut self,
        arr: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.check_obj(arr)?;
        let aty = self.ty_of(arr);
        let (elem, size) = match &aty {
            Ty::Array { elem, size } => ((**elem).clone(), *size),
            _ => return Err(IrError::NotAProduct),
        };
        if size > i32::MAX as u64 {
            return Err(IrError::EnumerateIndexOverflow);
        }
        if ty::ty_contains_str(&aty) {
            return Err(IrError::StrOutsidePrint);
        }
        let out_ty = Ty::Array {
            elem: Box::new(Ty::Tuple(vec![Ty::i32(), elem])),
            size,
        };
        self.b.intake_ty(&out_ty)?;
        let plan = self.dest_target(&dest, &out_ty, loc)?;
        Ok(self.emit_to_dest(plan, arr, Operation::Enumerate, loc))
    }

    /// `Widen` (ADR-0029): one of `i32→i64`, `i32→f32`, `i32→f64`, or
    /// `f32→f64`. Direct unary edge; pure and trap-free.
    pub fn widen(
        &mut self,
        src: ObjectId,
        target: Ty,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.check_obj(src)?;
        let source = self.ty_of(src);
        if !matches!(
            (&source, &target),
            (
                Ty::Int {
                    bits: 32,
                    signed: true
                },
                Ty::Int {
                    bits: 64,
                    signed: true
                } | Ty::Float { bits: 32 | 64 }
            ) | (Ty::Float { bits: 32 }, Ty::Float { bits: 64 })
                | (Ty::Narrow(_), Ty::Float { bits: 32 | 64 })
        ) {
            return Err(IrError::InvalidWiden);
        }
        self.b.intake_ty(&target)?;
        let plan = self.dest_target(&dest, &target, loc)?;
        Ok(self.emit_to_dest(plan, src, Operation::Widen, loc))
    }

    /// `Narrow` (ADR-0043): an `f32` rounded once to `fmt` (ties to even).
    /// Rejects any other source type (`InvalidNarrow`; an `f64` source needs
    /// a single-rounding realization — ADR-0043 open item).
    pub fn narrow(
        &mut self,
        src: ObjectId,
        fmt: crate::ty::NarrowFloat,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.check_obj(src)?;
        if self.ty_of(src) != Ty::f32() {
            return Err(IrError::InvalidNarrow);
        }
        let target = Ty::Narrow(fmt);
        let plan = self.dest_target(&dest, &target, loc)?;
        Ok(self.emit_to_dest(plan, src, Operation::Narrow, loc))
    }

    /// `Exp` (ADR-0042): `eˣ` of a float scalar, the same width out. Rejects
    /// any other source type (`InvalidExp`).
    pub fn exp(&mut self, src: ObjectId, dest: Dest, loc: SourceLoc) -> Result<ObjectId, IrError> {
        self.math(crate::math::MathFn::Exp, src, dest, loc)
    }

    /// `Math(f)` (ADR-0046): a math builtin of a float scalar, the same
    /// width out. Rejects any other source type (`InvalidExp`, the one code
    /// of every math builtin).
    pub fn math(
        &mut self,
        f: crate::math::MathFn,
        src: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.check_obj(src)?;
        let ty = self.ty_of(src);
        if !matches!(ty, Ty::Float { bits: 32 | 64 }) {
            return Err(IrError::InvalidExp);
        }
        let plan = self.dest_target(&dest, &ty, loc)?;
        Ok(self.emit_to_dest(plan, src, Operation::Math(f), loc))
    }

    /// `Iota` (ADR-0029): `[i32; n]` of `0..n-1`. The count rides as a
    /// `Constant` integer source object so the static-n rule is a graph
    /// property (validate's `IotaCountMismatch`). Rejects a non-`Constant` or
    /// non-integer count (`NonStaticCount` — a runtime size is ADR-0023
    /// territory, out of Core) and `n == 0` / `n > i32::MAX` (the element type
    /// is pinned `i32` in v1; `Ty::Array` requires size ≥ 1).
    pub fn iota(
        &mut self,
        count: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        let n = self.static_count(count)?;
        if n > i64::MAX as u64 {
            return Err(IrError::EnumerateIndexOverflow);
        }
        // onto-device: the elements are the extent's index type (`i64` only
        // past `i32::MAX` cells).
        let out_ty = Ty::Array {
            elem: Box::new(Ty::index_ty(n)),
            size: n,
        };
        self.b.intake_ty(&out_ty)?;
        let plan = self.dest_target(&dest, &out_ty, loc)?;
        Ok(self.emit_to_dest(plan, count, Operation::Iota, loc))
    }

    /// `Fill` (ADR-0029): `[T; n]` with every element the source value. Builds
    /// the internal 2-tuple `(x, count)` (Pair-then-primitive, as `update`'s
    /// 3-tuple); the count must be a `Constant` integer (`NonStaticCount`).
    /// Rejects `n == 0` and a `Str`-carrying value (I9s).
    pub fn fill(
        &mut self,
        x: ObjectId,
        count: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        let n = self.static_count(count)?;
        self.check_obj(x)?;
        let xty = self.ty_of(x);
        if ty::ty_contains_str(&xty) {
            return Err(IrError::StrOutsidePrint);
        }
        // Internal (x, count) 2-tuple, then Fill (the `zip`/`update` pattern).
        let pair_ty = Ty::Tuple(vec![xty.clone(), self.ty_of(count)]);
        self.b.intake_ty(&pair_ty)?;
        let pair = self
            .b
            .mint_object(self.f, pair_ty, None, ObjectKind::Temporary, None, loc);
        self.b
            .add_edge(self.f, x, pair, Operation::Pair { slot: 0, arity: 2 }, loc);
        self.b.add_edge(
            self.f,
            count,
            pair,
            Operation::Pair { slot: 1, arity: 2 },
            loc,
        );
        let out_ty = Ty::Array {
            elem: Box::new(xty),
            size: n,
        };
        self.b.intake_ty(&out_ty)?;
        let plan = self.dest_target(&dest, &out_ty, loc)?;
        Ok(self.emit_to_dest(plan, pair, Operation::Fill, loc))
    }

    /// `Fill` from an EXISTING internal `(x, count)` 2-tuple — the replay
    /// entry (S21). Emits only the `Fill` edge, minting nothing, so rewrite
    /// replay is structure-faithful: a CSE'd duplicate tuple stays dead
    /// instead of being reborn by `fill`'s sugar mint every round (the S21
    /// fixpoint fix). Same static-n rule as `fill` (slot-1 feeder must be a
    /// `Constant` integer ≥ 1); validate's `IotaCountMismatch` twin re-checks
    /// independently.
    pub fn fill_from(
        &mut self,
        pair: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.check_obj(pair)?;
        let (xty, count) = match self.ty_of(pair) {
            Ty::Tuple(ts) if ts.len() == 2 => {
                let count = self
                    .b
                    .in_edges
                    .get(pair)
                    .and_then(|ms| {
                        ms.iter().find_map(|&m| {
                            let mo = &self.b.morphisms[m];
                            matches!(mo.op, Operation::Pair { slot: 1, .. }).then_some(mo.source)
                        })
                    })
                    .ok_or(IrError::NonStaticCount)?;
                (ts[0].clone(), count)
            }
            _ => return Err(IrError::NonStaticCount),
        };
        let n = self.static_count(count)?;
        if ty::ty_contains_str(&xty) {
            return Err(IrError::StrOutsidePrint);
        }
        let out_ty = Ty::Array {
            elem: Box::new(xty),
            size: n,
        };
        self.b.intake_ty(&out_ty)?;
        let plan = self.dest_target(&dest, &out_ty, loc)?;
        Ok(self.emit_to_dest(plan, pair, Operation::Fill, loc))
    }

    /// The `iota`/`fill` static-count rule (ADR-0029): a `Constant` integer
    /// object with value `1..` (0 is not a legal `Ty::Array` size).
    pub(super) fn static_count(&self, count: ObjectId) -> Result<u64, IrError> {
        let obj = self.b.objects.get(count).ok_or(IrError::UnknownObject)?;
        match (obj.kind, &obj.value) {
            (ObjectKind::Constant, Some(Value::I32(v))) if *v > 0 => Ok(*v as u64),
            (ObjectKind::Constant, Some(Value::I64(v))) if *v > 0 => Ok(*v as u64),
            (ObjectKind::Constant, Some(Value::U8(v))) if *v > 0 => Ok(*v as u64),
            _ => Err(IrError::NonStaticCount),
        }
    }

    /// `Update` (DESIGN §5.1; ADR-0021): `(Array{T,n}, I, T) → Array{T,n}`. Builds
    /// the internal 3-tuple `(arr, idx, val)` (Pair-then-primitive, as `phi`), then
    /// applies `Update`. `I` is a Core integer scalar (exactly `index`); `val` must
    /// match the element ty. Result is the same array ty (fresh array, slot `i`
    /// replaced); OOB is a run-time trap, not a shape error.
    pub fn update(
        &mut self,
        arr: ObjectId,
        idx: ObjectId,
        val: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.check_obj(arr)?;
        self.check_obj(idx)?;
        self.check_obj(val)?;
        let aty = self.ty_of(arr);
        let ity = self.ty_of(idx);
        let vty = self.ty_of(val);
        let elem = match &aty {
            Ty::Array { elem, .. } => (**elem).clone(),
            _ => return Err(IrError::NotAProduct),
        };
        if !ity.is_integer() {
            return Err(IrError::TypeMismatch {
                expected: Ty::i32(),
                found: ity,
                loc,
            });
        }
        if vty != elem {
            return Err(IrError::TypeMismatch {
                expected: elem,
                found: vty,
                loc,
            });
        }
        let triple_ty = Ty::Tuple(vec![aty.clone(), ity.clone(), vty.clone()]);
        self.b.intake_ty(&triple_ty)?;
        let triple = self
            .b
            .mint_object(self.f, triple_ty, None, ObjectKind::Temporary, None, loc);
        self.b.add_edge(
            self.f,
            arr,
            triple,
            Operation::Pair { slot: 0, arity: 3 },
            loc,
        );
        self.b.add_edge(
            self.f,
            idx,
            triple,
            Operation::Pair { slot: 1, arity: 3 },
            loc,
        );
        self.b.add_edge(
            self.f,
            val,
            triple,
            Operation::Pair { slot: 2, arity: 3 },
            loc,
        );
        let plan = self.dest_target(&dest, &aty, loc)?;
        Ok(self.emit_to_dest(plan, triple, Operation::Update, loc))
    }

    /// `Call(f)` (DESIGN §5.1): `f` is `FuncKind::Named`; arg ty = input ty.
    pub fn call(
        &mut self,
        callee: FuncId,
        arg: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        let cf = self.b.funcs.get(callee).ok_or(IrError::UnknownFunction)?;
        if cf.def.kind != FuncKind::Named {
            return Err(IrError::CallToBody);
        }
        let in_ty = self.b.objects[cf.def.input].ty.clone();
        let out_ty = self.b.objects[cf.def.output].ty.clone();
        self.check_obj(arg)?;
        let aty = self.ty_of(arg);
        if aty != in_ty {
            return Err(IrError::TypeMismatch {
                expected: in_ty,
                found: aty,
                loc,
            });
        }
        self.b.intake_ty(&out_ty)?;
        let plan = self.dest_target(&dest, &out_ty, loc)?;
        Ok(self.emit_to_dest(plan, arg, Operation::Call(callee), loc))
    }

    /// `Map{body}` (DESIGN §5.1): `Array{T,n} → Array{U,n}`; body `T → U`,
    /// `MapBody`, token-free.
    pub fn map(
        &mut self,
        body: FuncId,
        arr: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        let bf = self.b.funcs.get(body).ok_or(IrError::UnknownFunction)?;
        if bf.def.kind != FuncKind::MapBody {
            return Err(IrError::BodyKindMismatch);
        }
        let body_in = self.b.objects[bf.def.input].ty.clone();
        let body_out = self.b.objects[bf.def.output].ty.clone();
        if ty::ty_contains_token(&body_in) || ty::ty_contains_token(&body_out) {
            return Err(IrError::TokenInBody);
        }
        self.check_obj(arr)?;
        let aty = self.ty_of(arr);
        let (elem, size) = match &aty {
            Ty::Array { elem, size } => ((**elem).clone(), *size),
            _ => return Err(IrError::NotAProduct),
        };
        if elem != body_in {
            return Err(IrError::TypeMismatch {
                expected: body_in,
                found: elem,
                loc,
            });
        }
        let out_ty = Ty::Array {
            elem: Box::new(body_out),
            size,
        };
        self.b.intake_ty(&out_ty)?;
        let plan = self.dest_target(&dest, &out_ty, loc)?;
        Ok(self.emit_to_dest(plan, arr, Operation::Map { body, captures: 0 }, loc))
    }

    /// `Fold{body}` (DESIGN §5.1): `(Acc, Array{T,n}) → Acc`; body
    /// `(Acc, T) → Acc`, `FoldBody`, token-free.
    pub fn fold(
        &mut self,
        body: FuncId,
        seed_and_arr: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        let bf = self.b.funcs.get(body).ok_or(IrError::UnknownFunction)?;
        if bf.def.kind != FuncKind::FoldBody {
            return Err(IrError::BodyKindMismatch);
        }
        let body_in = self.b.objects[bf.def.input].ty.clone();
        let body_out = self.b.objects[bf.def.output].ty.clone();
        if ty::ty_contains_token(&body_in) || ty::ty_contains_token(&body_out) {
            return Err(IrError::TokenInBody);
        }
        self.check_obj(seed_and_arr)?;
        let pty = self.ty_of(seed_and_arr);
        // Source must be (Acc, Array{T, n}); body_in must be (Acc, T).
        let (acc_ty, t_ty) = match &body_in {
            Ty::Tuple(ts) if ts.len() == 2 => (ts[0].clone(), ts[1].clone()),
            _ => {
                return Err(IrError::TypeMismatch {
                    expected: body_in.clone(),
                    found: pty.clone(),
                    loc,
                });
            }
        };
        let expected_src = Ty::Tuple(vec![
            acc_ty.clone(),
            Ty::Array {
                elem: Box::new(t_ty.clone()),
                size: array_size_of(&pty).unwrap_or(1),
            },
        ]);
        // Validate the source is (Acc, Array{T,n}) for some n, and body_out==Acc.
        match &pty {
            Ty::Tuple(ts) if ts.len() == 2 => {
                if ts[0] != acc_ty {
                    return Err(IrError::TypeMismatch {
                        expected: acc_ty.clone(),
                        found: ts[0].clone(),
                        loc,
                    });
                }
                match &ts[1] {
                    Ty::Array { elem, .. } if **elem == t_ty => {}
                    _ => {
                        return Err(IrError::TypeMismatch {
                            expected: expected_src,
                            found: pty.clone(),
                            loc,
                        });
                    }
                }
            }
            _ => {
                return Err(IrError::TypeMismatch {
                    expected: expected_src,
                    found: pty.clone(),
                    loc,
                });
            }
        }
        if body_out != acc_ty {
            return Err(IrError::TypeMismatch {
                expected: acc_ty,
                found: body_out,
                loc,
            });
        }
        self.b.intake_ty(&body_out)?;
        let plan = self.dest_target(&dest, &body_out, loc)?;
        Ok(self.emit_to_dest(
            plan,
            seed_and_arr,
            Operation::Fold { body, captures: 0 },
            loc,
        ))
    }

    /// `Map{body, captures}` (ADR-0027): `(c₁…cₖ, Array{T,n}) → Array{U,n}`;
    /// body `(c₁…cₖ, T) → U`. The `caps` objects are packed with the array
    /// into the source product (the broadcast edges); the capture count is
    /// recorded on the op. `caps` empty ⇒ [`Self::map`].
    pub fn map_captured(
        &mut self,
        body: FuncId,
        caps: &[ObjectId],
        arr: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        if caps.is_empty() {
            return self.map(body, arr, dest, loc);
        }
        let bf = self.b.funcs.get(body).ok_or(IrError::UnknownFunction)?;
        if bf.def.kind != FuncKind::MapBody {
            return Err(IrError::BodyKindMismatch);
        }
        let body_in = self.b.objects[bf.def.input].ty.clone();
        let body_out = self.b.objects[bf.def.output].ty.clone();
        if ty::ty_contains_token(&body_in) || ty::ty_contains_token(&body_out) {
            return Err(IrError::TokenInBody);
        }
        // Body input must be the (caps…, elem) product with matching types.
        let comps = match &body_in {
            Ty::Tuple(ts) if ts.len() == caps.len() + 1 => ts.clone(),
            _ => {
                return Err(IrError::TypeMismatch {
                    expected: body_in.clone(),
                    found: self.ty_of(arr),
                    loc,
                });
            }
        };
        for (i, &c) in caps.iter().enumerate() {
            self.check_obj(c)?;
            let cty = self.ty_of(c);
            if cty != comps[i] {
                return Err(IrError::TypeMismatch {
                    expected: comps[i].clone(),
                    found: cty,
                    loc,
                });
            }
            // ADR-0027: a capture with no runtime representation would have no
            // body-input slot on the backends (LLVM panicked). `IoToken` is
            // already `TokenInBody` above; `Str` stays `StrOutsidePrint` (the
            // pack below), so neither is re-diagnosed here.
            if ty::ty_residual_empty(&cty) && !ty::ty_contains_str(&cty) {
                return Err(IrError::ErasedCapture);
            }
        }
        self.check_obj(arr)?;
        let (elem, size) = match &self.ty_of(arr) {
            Ty::Array { elem, size } => ((**elem).clone(), *size),
            _ => return Err(IrError::NotAProduct),
        };
        if elem != comps[caps.len()] {
            return Err(IrError::TypeMismatch {
                expected: comps[caps.len()].clone(),
                found: elem,
                loc,
            });
        }
        let mut all: Vec<ObjectId> = caps.to_vec();
        all.push(arr);
        let src = self.pack(&all, Dest::Fresh(None), loc)?;
        let out_ty = Ty::Array {
            elem: Box::new(body_out),
            size,
        };
        self.b.intake_ty(&out_ty)?;
        let plan = self.dest_target(&dest, &out_ty, loc)?;
        Ok(self.emit_to_dest(
            plan,
            src,
            Operation::Map {
                body,
                captures: caps.len() as u32,
            },
            loc,
        ))
    }

    /// `Fold{body, captures}` (ADR-0027): `(c₁…cₖ, Acc, Array{T,n}) → Acc`;
    /// body `(c₁…cₖ, Acc, T) → Acc`. `caps` empty ⇒ [`Self::fold`].
    pub fn fold_captured(
        &mut self,
        body: FuncId,
        caps: &[ObjectId],
        seed: ObjectId,
        arr: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        if caps.is_empty() {
            let seed_and_arr = self.pack(&[seed, arr], Dest::Fresh(None), loc)?;
            return self.fold(body, seed_and_arr, dest, loc);
        }
        let bf = self.b.funcs.get(body).ok_or(IrError::UnknownFunction)?;
        if bf.def.kind != FuncKind::FoldBody {
            return Err(IrError::BodyKindMismatch);
        }
        let body_in = self.b.objects[bf.def.input].ty.clone();
        let body_out = self.b.objects[bf.def.output].ty.clone();
        if ty::ty_contains_token(&body_in) || ty::ty_contains_token(&body_out) {
            return Err(IrError::TokenInBody);
        }
        // Body input must be (caps…, Acc, T); Acc = body_out.
        let comps = match &body_in {
            Ty::Tuple(ts) if ts.len() == caps.len() + 2 => ts.clone(),
            _ => {
                return Err(IrError::TypeMismatch {
                    expected: body_in.clone(),
                    found: self.ty_of(seed),
                    loc,
                });
            }
        };
        for (i, &c) in caps.iter().enumerate() {
            self.check_obj(c)?;
            let cty = self.ty_of(c);
            if cty != comps[i] {
                return Err(IrError::TypeMismatch {
                    expected: comps[i].clone(),
                    found: cty,
                    loc,
                });
            }
            // ADR-0027: a capture with no runtime representation would have no
            // body-input slot on the backends (LLVM panicked). `IoToken` is
            // already `TokenInBody` above; `Str` stays `StrOutsidePrint` (the
            // pack below), so neither is re-diagnosed here.
            if ty::ty_residual_empty(&cty) && !ty::ty_contains_str(&cty) {
                return Err(IrError::ErasedCapture);
            }
        }
        let acc_ty = comps[caps.len()].clone();
        let elem = comps[caps.len() + 1].clone();
        if body_out != acc_ty {
            return Err(IrError::TypeMismatch {
                expected: acc_ty,
                found: body_out,
                loc,
            });
        }
        self.check_obj(seed)?;
        if self.ty_of(seed) != acc_ty {
            return Err(IrError::TypeMismatch {
                expected: acc_ty,
                found: self.ty_of(seed),
                loc,
            });
        }
        self.check_obj(arr)?;
        match &self.ty_of(arr) {
            Ty::Array { elem: ae, .. } if **ae == elem => {}
            _ => {
                return Err(IrError::TypeMismatch {
                    expected: comps[caps.len() + 1].clone(),
                    found: self.ty_of(arr),
                    loc,
                });
            }
        }
        let mut all: Vec<ObjectId> = caps.to_vec();
        all.push(seed);
        all.push(arr);
        let src = self.pack(&all, Dest::Fresh(None), loc)?;
        self.b.intake_ty(&acc_ty)?;
        let plan = self.dest_target(&dest, &acc_ty, loc)?;
        Ok(self.emit_to_dest(
            plan,
            src,
            Operation::Fold {
                body,
                captures: caps.len() as u32,
            },
            loc,
        ))
    }
}
