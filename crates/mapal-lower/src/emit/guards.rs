//! Emitter methods: guard arms (Phi).

use super::*;

impl Emitter<'_> {
    // --- guards (Phi) -------------------------------------------------------

    pub(super) fn emit_phi_guard(
        &mut self,
        fb: &mut FnBuilder,
        arms: &[GuardArm],
        scrutinee: ObjectId,
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        // Classify: bool guard vs value-match.
        let scrut_ty = self.ty_of(fb, scrutinee);
        let has_int = arms.iter().any(|a| matches!(a.discr, GuardDiscr::Int(_)));
        let has_bool = arms
            .iter()
            .any(|a| matches!(a.discr, GuardDiscr::True | GuardDiscr::False));
        if has_int && has_bool {
            return Err(diag(
                LCode::GuardArmMixed,
                sp,
                "bool and integer discriminants in one guard",
            ));
        }
        if has_int {
            self.emit_value_match(fb, arms, scrutinee, &scrut_ty, dest, sp)
        } else {
            self.emit_bool_guard(fb, arms, scrutinee, &scrut_ty, dest, sp)
        }
    }

    pub(super) fn emit_bool_guard(
        &mut self,
        fb: &mut FnBuilder,
        arms: &[GuardArm],
        scrutinee: ObjectId,
        scrut_ty: &Ty,
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        if *scrut_ty != Ty::Bool {
            return Err(diag(
                LCode::GuardScrutineeType,
                sp,
                "bool guard on non-bool scrutinee",
            ));
        }
        // Find true / false arms (a Default may stand for a missing pole).
        let mut t_arm: Option<&GuardArm> = None;
        let mut f_arm: Option<&GuardArm> = None;
        let mut default: Option<&GuardArm> = None;
        for a in arms {
            match a.discr {
                GuardDiscr::True => {
                    if t_arm.is_some() {
                        return Err(diag(
                            LCode::GuardArmDuplicate,
                            a.discr_span,
                            "duplicate `-true->`",
                        ));
                    }
                    t_arm = Some(a);
                }
                GuardDiscr::False => {
                    if f_arm.is_some() {
                        return Err(diag(
                            LCode::GuardArmDuplicate,
                            a.discr_span,
                            "duplicate `-false->`",
                        ));
                    }
                    f_arm = Some(a);
                }
                GuardDiscr::Default => {
                    // A duplicate `-_->` is a duplicate discriminant (L1402; ATK-15b).
                    if default.is_some() {
                        return Err(diag(
                            LCode::GuardArmDuplicate,
                            a.discr_span,
                            "duplicate `-_->` default arm",
                        ));
                    }
                    default = Some(a);
                }
                GuardDiscr::Int(_) => {
                    return Err(diag(
                        LCode::GuardArmMixed,
                        a.discr_span,
                        "integer arm in bool guard",
                    ));
                }
                GuardDiscr::OutOfCore => {
                    return Err(diag(LCode::UncleanTree, a.discr_span, "out-of-core arm"));
                }
            }
        }
        // A `-_->` may stand for THE single missing pole (§8.4, singular). With
        // both concrete poles present, the default is an unreachable arm (L1402;
        // ATK-15c). With BOTH poles missing (default-only), it cannot stand for
        // two poles → L1401 (ATK-15a).
        if let Some(d) = default
            && t_arm.is_some()
            && f_arm.is_some()
        {
            return Err(diag(
                LCode::GuardArmDuplicate,
                d.discr_span,
                "unreachable `-_->` default arm: both `-true->` and `-false->` are present",
            ));
        }
        let t = t_arm.or(default);
        let f = f_arm.or(default);
        let (t, f) = match (t, f) {
            // Default-only (both concrete poles missing): the default cannot
            // stand for both poles → L1401 (ATK-15a). `t` and `f` would alias the
            // same default arm here, which is exactly the missing-poles case.
            (Some(t), Some(f)) if t_arm.is_some() || f_arm.is_some() => (t, f),
            _ => {
                return Err(diag(
                    LCode::GuardArmMissing,
                    sp,
                    "bool guard without both poles",
                ));
            }
        };
        self.check_phi_arm(&t.payload)?;
        self.check_phi_arm(&f.payload)?;
        let t_val = self.emit_arm_value(fb, &t.payload, scrutinee)?;
        let f_val = self.emit_arm_value(fb, &f.payload, scrutinee)?;
        // Use the `self.phi` wrapper (NOT a bare `fb.phi`) so the result ty is
        // recorded and the derives-from-merge tag propagates — otherwise the
        // result defaults to Ty::Unit and is absent from `self.derived`,
        // falsely rejecting downstream print/route/loop uses (ATK-04).
        self.phi(fb, t_val, f_val, scrutinee, dest, sp)
    }

    pub(super) fn emit_value_match(
        &mut self,
        fb: &mut FnBuilder,
        arms: &[GuardArm],
        scrutinee: ObjectId,
        scrut_ty: &Ty,
        dest: Dest,
        sp: syn::SourceLoc,
    ) -> ER<ObjectId> {
        if !scrut_ty.is_integer() {
            return Err(diag(
                LCode::GuardScrutineeType,
                sp,
                "integer discriminants on non-integer scrutinee",
            ));
        }
        // Collect ordered int arms + the default (innermost).
        let mut int_arms: Vec<(&GuardArm, u64)> = Vec::new();
        let mut default: Option<&GuardArm> = None;
        let mut seen: BTreeMap<u64, ()> = BTreeMap::new();
        for a in arms {
            match a.discr {
                GuardDiscr::Int(k) => {
                    if seen.contains_key(&k) {
                        return Err(diag(
                            LCode::GuardArmDuplicate,
                            a.discr_span,
                            "duplicate discriminant",
                        ));
                    }
                    seen.insert(k, ());
                    int_arms.push((a, k));
                }
                GuardDiscr::Default => default = Some(a),
                GuardDiscr::True | GuardDiscr::False => {
                    return Err(diag(
                        LCode::GuardArmMixed,
                        a.discr_span,
                        "bool arm in value match",
                    ));
                }
                GuardDiscr::OutOfCore => {
                    return Err(diag(LCode::UncleanTree, a.discr_span, "out-of-core arm"));
                }
            }
        }
        let default = default
            .ok_or_else(|| diag(LCode::GuardArmMissing, sp, "value match without `-_->`"))?;
        // Check arms, compute conds in arm order, then right-fold phi.
        for (a, _) in &int_arms {
            self.check_phi_arm(&a.payload)?;
        }
        self.check_phi_arm(&default.payload)?;
        // cond_i = Eq(scrutinee, const k_i).
        let mut conds: Vec<ObjectId> = Vec::new();
        for (a, k) in &int_arms {
            let kval = self.int_value(*k, scrut_ty, a.discr_span, false)?;
            let kc = self.constant(fb, kval, a.discr_span)?;
            let c = self.binop(
                fb,
                Operation::Eq,
                scrutinee,
                kc,
                Dest::Fresh(None),
                a.discr_span,
            )?;
            conds.push(c);
        }
        // arm values.
        let mut arm_vals: Vec<ObjectId> = Vec::new();
        for (a, _) in &int_arms {
            arm_vals.push(self.emit_arm_value(fb, &a.payload, scrutinee)?);
        }
        let dval = self.emit_arm_value(fb, &default.payload, scrutinee)?;
        // Right-fold: innermost phi(e_n, e_d, c_n), outward; outermost takes dest.
        let n = int_arms.len();
        if n == 0 {
            return Err(diag(LCode::GuardArmMissing, sp, "value match with no arms"));
        }
        // Build innermost first.
        let mut acc = {
            let i = n - 1;
            self.phi(
                fb,
                arm_vals[i],
                dval,
                conds[i],
                if n == 1 {
                    dest.clone()
                } else {
                    Dest::Fresh(None)
                },
                int_arms[i].0.discr_span,
            )?
        };
        for i in (0..n - 1).rev() {
            let d = if i == 0 {
                dest.clone()
            } else {
                Dest::Fresh(None)
            };
            acc = self.phi(fb, arm_vals[i], acc, conds[i], d, int_arms[i].0.discr_span)?;
        }
        Ok(acc)
    }

    /// Phi-arm restrictions: no ret (L1405), no effects (L1404), no
    /// enclosing-mut assignment (L1408).
    pub(super) fn check_phi_arm(&self, payload: &ArmPayload) -> ER<()> {
        let block_or_chain: PhiArmScan = scan_phi_arm(self.source, self.fn_sigs, payload);
        if let Some(sp) = block_or_chain.ret_span {
            return Err(diag(
                LCode::GuardRetInPhiArm,
                sp,
                "`-> ret` inside a Phi-position arm",
            ));
        }
        if let Some(sp) = block_or_chain.effect_span {
            return Err(diag(
                LCode::GuardArmEffectful,
                sp,
                "effectful operation in a Phi-position arm",
            ));
        }
        if let Some(sp) = block_or_chain.loopjump_span {
            return Err(diag(
                LCode::JumpMisplaced,
                sp,
                "`-> loop` in a Phi-position arm",
            ));
        }
        // Enclosing-mut assignment detection.
        for (name, sp) in &block_or_chain.assigns {
            if let Some((b, _)) = self.scope.resolve(name)
                && b.mutable
            {
                return Err(diag(
                    LCode::AssignInPhiArm,
                    *sp,
                    format!("assignment to enclosing `mut` `{name}` in a Phi arm"),
                ));
            }
        }
        Ok(())
    }

    /// Emit an arm payload's value (in a child scope against the snapshot).
    pub(super) fn emit_arm_value(
        &mut self,
        fb: &mut FnBuilder,
        payload: &ArmPayload,
        scrutinee: ObjectId,
    ) -> ER<ObjectId> {
        self.scope.push();
        let r = self.emit_arm_value_inner(fb, payload, scrutinee);
        self.scope.pop();
        r
    }

    pub(super) fn emit_arm_value_inner(
        &mut self,
        fb: &mut FnBuilder,
        payload: &ArmPayload,
        scrutinee: ObjectId,
    ) -> ER<ObjectId> {
        match payload {
            ArmPayload::Chain(c) => {
                let r = self.emit_chain(fb, c, ChainCtx::HeadlessSeed(scrutinee))?;
                r.ok_or_else(|| diag(LCode::TypeMismatch, c.span, "guard arm produces no value"))
            }
            ArmPayload::Block(b) => {
                for item in &b.items {
                    if let BlockItem::Stmt(s) = item {
                        self.emit_stmt(fb, s)?;
                    }
                }
                match &b.tail {
                    Some(t) => {
                        let r = self.emit_chain(fb, t, ChainCtx::HeadlessSeed(scrutinee))?;
                        r.ok_or_else(|| {
                            diag(LCode::TypeMismatch, t.span, "guard arm produces no value")
                        })
                    }
                    None => Err(diag(
                        LCode::TypeMismatch,
                        b.span,
                        "guard arm produces no value",
                    )),
                }
            }
        }
    }
}
