//! `FnBuilder` core primitives: inputs, constants, destinations, products, unary / binary ops, `phi` and `index`.

use super::*;

impl FnBuilder<'_> {
    /// The function's input Parameter object.
    pub fn input(&self) -> ObjectId {
        self.b.funcs[self.f].def.input
    }

    /// ADR-0041 D1: record that the float product defining `product` admits
    /// `p`. A non-product (or an integer one) and `{f32}` record nothing.
    pub fn admit(&mut self, product: ObjectId, p: Precision) {
        let Some(&m) = self.b.in_edges.get(product).and_then(|v| v.first()) else {
            return;
        };
        let float = matches!(self.b.objects[product].ty, Ty::Float { .. });
        if p != Precision::F32 && float && self.b.morphisms[m].op == Operation::Mul {
            self.b.precision.insert(m, p);
        }
    }

    pub(super) fn output_id(&self) -> ObjectId {
        self.b.funcs[self.f].def.output
    }

    pub(super) fn output_ty(&self) -> Ty {
        self.b.objects[self.output_id()].ty.clone()
    }

    /// Resolve an object id, checking it exists and is owned by this function.
    pub(super) fn check_obj(&self, id: ObjectId) -> Result<(), IrError> {
        if !self.b.objects.contains_key(id) {
            return Err(IrError::UnknownObject);
        }
        if self.b.owner.get(id) != Some(&self.f) {
            return Err(IrError::WrongBuilder);
        }
        Ok(())
    }

    /// The builder-owned ty of an object in this fn — the one source of truth
    /// for result types the builder synthesizes itself (`iota`/`fill_from`
    /// array sizes, ADR-0031: lower reads this instead of re-deriving).
    pub fn ty_of(&self, id: ObjectId) -> Ty {
        self.b.objects[id].ty.clone()
    }

    /// A Constant source object (DESIGN §10); `ty := v.ty()` (I7).
    pub fn constant(&mut self, v: Value, loc: SourceLoc) -> Result<ObjectId, IrError> {
        let ty = v.ty();
        self.b.intake_ty(&ty)?;
        Ok(self
            .b
            .mint_object(self.f, ty, Some(v), ObjectKind::Constant, None, loc))
    }

    /// Resolve a `Dest` to a target object for a primitive producing `value_ty`,
    /// returning the object the caller should treat as "the result" plus the
    /// morphism wiring already done. The primitive's defining edge targets the
    /// returned object (for `Fresh`/`Ret{None}`) or a fresh component which then
    /// gets a `Pair` slot edge into Return (for `Ret{Some}`).
    ///
    /// Returns the object the produced value *lives in* (so callers can keep
    /// using it). For `Ret{Some(k)}` that is the fresh component object.
    pub(super) fn dest_target(
        &mut self,
        dest: &Dest,
        value_ty: &Ty,
        loc: SourceLoc,
    ) -> Result<DestPlan, IrError> {
        match dest {
            Dest::Fresh(name) => {
                let obj = self.b.mint_object(
                    self.f,
                    value_ty.clone(),
                    None,
                    ObjectKind::Temporary,
                    name.clone(),
                    loc,
                );
                Ok(DestPlan::Direct(obj))
            }
            Dest::Ret { slot: None } => {
                let ret = self.output_id();
                let out_ty = self.output_ty();
                if &out_ty != value_ty {
                    return Err(IrError::RetTypeMismatch);
                }
                Ok(DestPlan::Direct(ret))
            }
            Dest::Ret { slot: Some(k) } => {
                let out_ty = self.output_ty();
                if out_ty.product_arity().is_none() {
                    return Err(IrError::RetNotProduct);
                }
                // u64 arity capped to u32: an array of size > u32::MAX is not
                // slot-addressable (the `Pair { arity: u32 }` field can't name
                // it) — reject the slot write rather than truncate (F4/SND-3).
                let arity = out_ty.product_arity_u32().ok_or(IrError::SlotOutOfRange)?;
                if *k >= arity {
                    return Err(IrError::SlotOutOfRange);
                }
                let comp_ty = out_ty.component_ty(*k).unwrap().clone();
                if &comp_ty != value_ty {
                    return Err(IrError::RetTypeMismatch);
                }
                // Fresh component object holds the value; a Pair slot edge wires
                // it into Return.
                let comp = self.b.mint_object(
                    self.f,
                    value_ty.clone(),
                    None,
                    ObjectKind::Temporary,
                    None,
                    loc,
                );
                let ret = self.output_id();
                Ok(DestPlan::Slot {
                    comp,
                    ret,
                    slot: *k,
                    arity,
                })
            }
        }
    }

    /// Wire a primitive's defining edge to the dest plan's target, returning the
    /// object the produced value lives in.
    pub(super) fn emit_to_dest(
        &mut self,
        plan: DestPlan,
        source: ObjectId,
        op: Operation,
        loc: SourceLoc,
    ) -> ObjectId {
        match plan {
            DestPlan::Direct(tgt) => {
                self.b.add_edge(self.f, source, tgt, op, loc);
                tgt
            }
            DestPlan::Slot {
                comp,
                ret,
                slot,
                arity,
            } => {
                self.b.add_edge(self.f, source, comp, op, loc);
                self.b
                    .add_edge(self.f, comp, ret, Operation::Pair { slot, arity }, loc);
                comp
            }
        }
    }

    /// `π_index` out of a Tuple/Struct (DESIGN §5.1).
    pub fn proj(
        &mut self,
        src: ObjectId,
        index: u32,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.check_obj(src)?;
        let src_ty = self.ty_of(src);
        let comp = match &src_ty {
            Ty::Tuple(_) | Ty::Struct { .. } => src_ty
                .component_ty(index)
                .ok_or(IrError::SlotOutOfRange)?
                .clone(),
            _ => return Err(IrError::NotAProduct),
        };
        self.b.intake_ty(&comp)?;
        let plan = self.dest_target(&dest, &comp, loc)?;
        let out = self.emit_to_dest(plan, src, Operation::Proj { index }, loc);
        Ok(out)
    }

    /// Build a product object by packing `components` with `Pair` slot edges.
    /// `target_ty` is the (already-intaken) product ty; checks per-slot tys and
    /// I9s `Str` restriction. Returns the product object.
    pub(super) fn build_pack(
        &mut self,
        components: &[ObjectId],
        target_ty: Ty,
        dest: Dest,
        loc: SourceLoc,
        allow_str_components: bool,
    ) -> Result<ObjectId, IrError> {
        let arity = components.len() as u32;
        // Per-component ty checks + Str restriction.
        for (k, &c) in components.iter().enumerate() {
            self.check_obj(c)?;
            let cty = self.ty_of(c);
            let expected = target_ty.component_ty(k as u32).unwrap().clone();
            if cty != expected {
                return Err(IrError::TypeMismatch {
                    expected,
                    found: cty,
                    loc,
                });
            }
            if !allow_str_components && ty::ty_contains_str(&cty) {
                return Err(IrError::StrOutsidePrint);
            }
        }
        self.b.intake_ty(&target_ty)?;
        // Choose the product object per dest.
        let plan = self.dest_target(&dest, &target_ty, loc)?;
        // A Ret-slot dest: `dest_target` already minted the slot's component
        // (it type-checked the slot); the pack writes that component, then a
        // `Pair` edge wires it into Return. (mask-frontend: a second, fresh
        // product here left the minted component writerless — a struct
        // literal straight to `ret.1` crashed replay on it.)
        let (product, wire) = match plan {
            DestPlan::Direct(tgt) => (tgt, None),
            DestPlan::Slot {
                comp,
                ret,
                slot,
                arity,
            } => (comp, Some((ret, slot, arity))),
        };
        for (k, &c) in components.iter().enumerate() {
            self.b.add_edge(
                self.f,
                c,
                product,
                Operation::Pair {
                    slot: k as u32,
                    arity,
                },
                loc,
            );
        }
        if let Some((ret, slot, ret_arity)) = wire {
            self.b.add_edge(
                self.f,
                product,
                ret,
                Operation::Pair {
                    slot,
                    arity: ret_arity,
                },
                loc,
            );
        }
        Ok(product)
    }

    /// Tuple literal (DESIGN §10): `len ≥ 2` (`0`→`EmptyProduct`, `1`→`SingletonTuple`).
    pub fn pack(
        &mut self,
        components: &[ObjectId],
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        if components.is_empty() {
            return Err(IrError::EmptyProduct);
        }
        if components.len() == 1 {
            return Err(IrError::SingletonTuple);
        }
        // Synthesize the Tuple ty from components.
        let mut tys = Vec::with_capacity(components.len());
        for &c in components {
            self.check_obj(c)?;
            tys.push(self.ty_of(c));
        }
        let target = Ty::Tuple(tys);
        self.build_pack(components, target, dest, loc, false)
    }

    /// Struct literal (DESIGN §10): `ty` must be `Ty::Struct`; `components` in
    /// field order; `len == field count`.
    pub fn pack_struct(
        &mut self,
        ty: Ty,
        components: &[ObjectId],
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        let fields = match &ty {
            Ty::Struct { fields, .. } => fields.clone(),
            _ => return Err(IrError::NotAProduct),
        };
        // Defense in depth (mirrors `pack`): a zero-component struct literal would
        // mint a Temporary with zero in-edges. `intake_ty` already rejects a
        // zero-field `Ty::Struct` as `NonCoreType` (so this is unreachable via a
        // well-typed `(ty, components)` pair where the arities agree), but the
        // `EmptyProduct` guard makes the empty-product rejection local and explicit
        // here too, independent of the arity-vs-ty agreement (review TY-1).
        if components.is_empty() {
            return Err(IrError::EmptyProduct);
        }
        if components.len() != fields.len() {
            return Err(IrError::ArityMismatch);
        }
        self.b.intake_ty(&ty)?;
        self.build_pack(components, ty, dest, loc, false)
    }

    /// Array literal (DESIGN §10): `size = len ≥ 1`, all elem tys equal.
    pub fn pack_array(
        &mut self,
        components: &[ObjectId],
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        if components.is_empty() {
            return Err(IrError::EmptyProduct);
        }
        for &c in components {
            self.check_obj(c)?;
        }
        let elem = self.ty_of(components[0]);
        for &c in &components[1..] {
            let cty = self.ty_of(c);
            if cty != elem {
                return Err(IrError::TypeMismatch {
                    expected: elem.clone(),
                    found: cty,
                    loc,
                });
            }
        }
        let target = Ty::Array {
            elem: Box::new(elem),
            size: components.len() as u64,
        };
        self.build_pack(components, target, dest, loc, false)
    }

    /// Unary op (DESIGN §5.1): `{Neg, Not}` only.
    pub fn unop(
        &mut self,
        op: Operation,
        x: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.check_obj(x)?;
        let xty = self.ty_of(x);
        let out_ty = match op {
            Operation::Neg => {
                if !xty.is_numeric() {
                    return Err(IrError::TypeMismatch {
                        expected: Ty::i32(),
                        found: xty,
                        loc,
                    });
                }
                xty.clone()
            }
            Operation::Not => {
                if xty != Ty::Bool {
                    return Err(IrError::TypeMismatch {
                        expected: Ty::Bool,
                        found: xty,
                        loc,
                    });
                }
                Ty::Bool
            }
            _ => return Err(IrError::NotUnary),
        };
        self.b.intake_ty(&out_ty)?;
        let plan = self.dest_target(&dest, &out_ty, loc)?;
        Ok(self.emit_to_dest(plan, x, op, loc))
    }

    /// Binary op (DESIGN §5.1 / §10): the §10 membership set; packs `(lhs, rhs)`
    /// then applies the primitive (Pair-then-primitive, atomic).
    pub fn binop(
        &mut self,
        op: Operation,
        lhs: ObjectId,
        rhs: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.check_obj(lhs)?;
        self.check_obj(rhs)?;
        let lty = self.ty_of(lhs);
        let rty = self.ty_of(rhs);

        // Per-op typing dispatch (explicit membership; not enum-order range).
        let out_ty = match op {
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Div | Operation::Mod => {
                if !lty.is_numeric() {
                    return Err(IrError::TypeMismatch {
                        expected: lty.clone(),
                        found: lty.clone(),
                        loc,
                    });
                }
                if lty != rty {
                    return Err(IrError::TypeMismatch {
                        expected: lty,
                        found: rty,
                        loc,
                    });
                }
                lty.clone()
            }
            Operation::Eq | Operation::Neq => {
                // A ∈ N ∪ {Bool}.
                let ok = lty.is_numeric() || lty == Ty::Bool;
                if !ok {
                    return Err(IrError::TypeMismatch {
                        expected: Ty::Bool,
                        found: lty.clone(),
                        loc,
                    });
                }
                if lty != rty {
                    return Err(IrError::TypeMismatch {
                        expected: lty,
                        found: rty,
                        loc,
                    });
                }
                Ty::Bool
            }
            Operation::Lt | Operation::Gt | Operation::Le | Operation::Ge => {
                if !lty.is_numeric() {
                    return Err(IrError::TypeMismatch {
                        expected: Ty::i32(),
                        found: lty.clone(),
                        loc,
                    });
                }
                if lty != rty {
                    return Err(IrError::TypeMismatch {
                        expected: lty,
                        found: rty,
                        loc,
                    });
                }
                Ty::Bool
            }
            Operation::And | Operation::Or => {
                if lty != Ty::Bool {
                    return Err(IrError::TypeMismatch {
                        expected: Ty::Bool,
                        found: lty,
                        loc,
                    });
                }
                if rty != Ty::Bool {
                    return Err(IrError::TypeMismatch {
                        expected: Ty::Bool,
                        found: rty,
                        loc,
                    });
                }
                Ty::Bool
            }
            _ => return Err(IrError::NotBinary),
        };

        // Str may never be a binop operand (I9s) — none of these accept Str
        // anyway, but guard explicitly for nested cases.
        if ty::ty_contains_str(&lty) || ty::ty_contains_str(&rty) {
            return Err(IrError::StrOutsidePrint);
        }

        // Build the (lhs, rhs) 2-tuple internally, then apply.
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

        self.b.intake_ty(&out_ty)?;
        let plan = self.dest_target(&dest, &out_ty, loc)?;
        Ok(self.emit_to_dest(plan, pair, op, loc))
    }

    /// `Phi` (DESIGN §5.1): `(T, T, Bool) → T`; `T` must not contain `IoToken`.
    pub fn phi(
        &mut self,
        t: ObjectId,
        f: ObjectId,
        cond: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.check_obj(t)?;
        self.check_obj(f)?;
        self.check_obj(cond)?;
        let tty = self.ty_of(t);
        let fty = self.ty_of(f);
        let cty = self.ty_of(cond);
        if tty != fty {
            return Err(IrError::TypeMismatch {
                expected: tty,
                found: fty,
                loc,
            });
        }
        if cty != Ty::Bool {
            return Err(IrError::TypeMismatch {
                expected: Ty::Bool,
                found: cty,
                loc,
            });
        }
        if ty::ty_contains_token(&tty) {
            return Err(IrError::TokenInPhi);
        }
        if ty::ty_contains_str(&tty) {
            return Err(IrError::StrOutsidePrint);
        }
        let triple_ty = Ty::Tuple(vec![tty.clone(), fty.clone(), Ty::Bool]);
        self.b.intake_ty(&triple_ty)?;
        let triple = self
            .b
            .mint_object(self.f, triple_ty, None, ObjectKind::Temporary, None, loc);
        self.b.add_edge(
            self.f,
            t,
            triple,
            Operation::Pair { slot: 0, arity: 3 },
            loc,
        );
        self.b.add_edge(
            self.f,
            f,
            triple,
            Operation::Pair { slot: 1, arity: 3 },
            loc,
        );
        self.b.add_edge(
            self.f,
            cond,
            triple,
            Operation::Pair { slot: 2, arity: 3 },
            loc,
        );
        let plan = self.dest_target(&dest, &tty, loc)?;
        Ok(self.emit_to_dest(plan, triple, Operation::Phi, loc))
    }

    /// `Index` (DESIGN §5.1): `(Array{T,n}, I) → T`; `I` a Core integer scalar.
    pub fn index(
        &mut self,
        arr: ObjectId,
        idx: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.check_obj(arr)?;
        self.check_obj(idx)?;
        let aty = self.ty_of(arr);
        let ity = self.ty_of(idx);
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
        let pair_ty = Ty::Tuple(vec![aty.clone(), ity.clone()]);
        self.b.intake_ty(&pair_ty)?;
        let pair = self
            .b
            .mint_object(self.f, pair_ty, None, ObjectKind::Temporary, None, loc);
        self.b.add_edge(
            self.f,
            arr,
            pair,
            Operation::Pair { slot: 0, arity: 2 },
            loc,
        );
        self.b.add_edge(
            self.f,
            idx,
            pair,
            Operation::Pair { slot: 1, arity: 2 },
            loc,
        );
        self.b.intake_ty(&elem)?;
        let plan = self.dest_target(&dest, &elem, loc)?;
        Ok(self.emit_to_dest(plan, pair, Operation::Index, loc))
    }
}
