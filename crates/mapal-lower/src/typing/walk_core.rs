//! Typing walk support: global constants, size parameters, expected tys, union-find and resolution.

use super::*;

impl Walk<'_> {
    /// ADR-0048 D3: a top-level scalar computed before the program runs — an
    /// unannotated number is a literal (its width from context), an
    /// annotated one a constant of its type.
    pub(super) fn global_const(&mut self, text: &str, span: syn::SourceLoc) -> WTy {
        use crate::globals::GlobalConst;
        match self.types.globals().consts[text].clone() {
            GlobalConst::Int {
                pinned: Some(t), ..
            }
            | GlobalConst::Float {
                pinned: Some(t), ..
            } => WTy::Known(t),
            GlobalConst::Int {
                value,
                pinned: None,
            } => {
                let var = self.fresh_var(span, false);
                if value > i128::from(i32::MAX) || value < i128::from(i32::MIN) {
                    self.pin(var, &Ty::i64(), span);
                }
                WTy::Var(var)
            }
            GlobalConst::Float { pinned: None, .. } => WTy::Var(self.fresh_var(span, true)),
            GlobalConst::Bool(_) => WTy::Known(Ty::Bool),
        }
    }

    /// ADR-0048 D5: a known struct read through its `_` field(s) until `fits`.
    pub(super) fn through_default(&mut self, w: &WTy, fits: impl Fn(&Ty) -> bool) -> WTy {
        if let Some(t) = self.wty_to_ty(w)
            && matches!(t, Ty::Struct { .. })
            && let Some(k) = TypeTable::default_steps(&t, &fits)
        {
            let mut cur = t;
            for _ in 0..k {
                cur = TypeTable::default_of(&cur).cloned().unwrap_or(Ty::Unit);
            }
            return WTy::Known(cur);
        }
        w.clone()
    }

    // --- size parameters (ADR-0047 D4) --------------------------------------

    /// The value of a size parameter named `text` not shadowed in `scope`.
    pub(super) fn size_value(&self, text: &str, scope: &ScopeStack<WTy>) -> Option<u64> {
        if scope.contains(text) {
            return None;
        }
        self.sizes.env.get(text).copied()
    }

    /// An integer literal's value, or a size expression's (D4: a size is an
    /// integer literal in every respect).
    pub(super) fn int_const(&self, e: &Expr, scope: &ScopeStack<WTy>) -> Option<u64> {
        match &e.kind {
            ExprKind::Int(n) => Some(*n),
            _ => self.size_fold(e, scope).and_then(|v| u64::try_from(v).ok()),
        }
    }

    /// D4: the value of an integer expression over size parameters and
    /// literals (`rows * columns`, `n - 1`), folded to one literal.
    pub(super) fn size_fold(&self, e: &Expr, scope: &ScopeStack<WTy>) -> Option<i128> {
        if self.sizes.env.is_empty() {
            return None;
        }
        crate::sizes::fold_value(self.source, e, &|t| self.size_value(t, scope))
    }

    /// L1703: a binding may not reuse a size parameter's name.
    pub(super) fn check_not_size(&mut self, name: syn::Name) {
        let text = name_text(self.source, name);
        if self.types.globals().is_top(text) {
            if !self.top_chain {
                self.diags.push(diag(
                    LCode::TopNameReused,
                    name.span,
                    format!(
                        "`{text}` is a top-level name; a local cannot reuse it — pick another name \
                         (ADR-0048 D1)"
                    ),
                ));
            }
        } else if self.sizes.env.contains_key(text) {
            self.diags.push(diag(
                LCode::SizeRebound,
                name.span,
                format!("`{text}` is a size parameter; it cannot be rebound — pick another name"),
            ));
        } else if self.sizes.env.format(text).is_some() {
            self.diags.push(diag(
                LCode::SizeRebound,
                name.span,
                format!(
                    "`{text}` is a format parameter (ADR-0051); it cannot be bound as a value — \
                     pick another name"
                ),
            ));
        }
    }

    /// D3 at a call to a generic function (typing side): emission's binding
    /// of this site when an earlier round found one, else this walk's own
    /// inference from `cur` (and the next stage's annotation), after the
    /// stage's explicit arguments (ADR-0051 D2). The result is the instance's
    /// output type; `Unknown` when neither reaches.
    pub(super) fn generic_call(
        &mut self,
        name: &str,
        cur: &WTy,
        next: Option<&Stage>,
        span: syn::SourceLoc,
        args: Option<&[syn::SizeExpr]>,
    ) -> WTy {
        let Some(fd) = self.sizes.generics.get(name).copied() else {
            return WTy::Unknown;
        };
        let explicit = match args {
            Some(a) => match crate::sizes::explicit_args(
                self.source,
                a,
                &self.sizes.env,
                self.source,
                name,
                &fd.generics,
                span,
            ) {
                Ok(v) => v,
                Err(d) => {
                    self.diags.push(d);
                    return WTy::Unknown;
                }
            },
            None => Vec::new(),
        };
        let key: SiteKey = (self.owner_name.clone(), span.start, span.end);
        let values = match self.sizes.sites.get(&key) {
            Some(v) => Some(v.clone()),
            None => self.wty_to_ty(cur).and_then(|arg| {
                let expected = self.expected_flow(next);
                crate::sizes::infer(
                    self.source,
                    self.types,
                    fd,
                    &arg,
                    expected.as_ref(),
                    &explicit,
                    span,
                )
                .ok()
            }),
        };
        let Some(values) = values else {
            self.size_use.unresolved += 1;
            return WTy::Unknown;
        };
        let env = crate::sizes::env_of(self.source, &fd.generics, &values, self.types.base_env());
        let effectful = self.effects.is_effectful(name);
        let Some(sig) = crate::sizes::instance_sig(self.source, self.types, fd, &env, effectful)
        else {
            return WTy::Unknown;
        };
        self.size_use.sites.insert(key, values.clone());
        self.size_use.instances.push((name.to_string(), values));
        self.type_call(&sig, cur, span);
        match sig.ret {
            Some(r) => WTy::Known(r),
            None => WTy::Unknown,
        }
    }

    /// ADR-0051 D4: `x -> to<…>` / `x -> widen<…>` (typing side): the target
    /// from the explicit argument, else the expected type (the next stage's
    /// annotation, the owner's output at `-> ret`, a fixed function's
    /// parameter the value flows into). Emission owns the admission (L1614 /
    /// L1616) and the undetermined target (L1708); a literal flowing into a
    /// narrow target is an `f32` (the one source), into an explicit
    /// `source` that format.
    pub(super) fn conversion(
        &mut self,
        verb: Verb,
        args: Option<&[syn::SizeExpr]>,
        cur: &WTy,
        next: Option<&Stage>,
        span: syn::SourceLoc,
    ) -> WTy {
        let decl = verb.decl();
        let explicit = match args {
            Some(a) => match crate::sizes::explicit_args(
                self.source,
                a,
                &self.sizes.env,
                crate::formats::PRELUDE,
                verb.name(),
                &decl.generics,
                span,
            ) {
                Ok(v) => v,
                Err(d) => {
                    self.diags.push(d);
                    return WTy::Unknown;
                }
            },
            None => Vec::new(),
        };
        let format_at = |i: usize| match explicit.get(i) {
            Some(Arg::Format(f)) => Some(*f),
            _ => None,
        };
        if let (Some(src), WTy::Var(_)) = (format_at(1), cur) {
            self.unify_known(cur, &src.ty(), span);
        }
        let target = format_at(0).or_else(|| {
            let want = self.expected_flow(next);
            want.as_ref().and_then(Format::of)
        });
        match target {
            Some(t) => {
                // A float literal flowing in takes its width from the
                // conversion: an `f32` into a narrow target (ADR-0043's one
                // source), the target itself into `f32` / `f64` (the identity).
                // An integer literal keeps its own width (`3 -> to<f32>`
                // converts an `i32`).
                if format_at(1).is_none()
                    && let WTy::Var(v) = cur
                    && t.is_float()
                {
                    let r = self.find(*v);
                    if matches!(self.state[r], LitState::FloatVar) {
                        let width = if t.narrow().is_some() {
                            Ty::f32()
                        } else {
                            t.ty()
                        };
                        self.unify_known(cur, &width, span);
                    }
                }
                WTy::Known(t.ty())
            }
            None => WTy::Unknown,
        }
    }

    /// ADR-0051 D3: the type a call's or a conversion's result flows into —
    /// the next stage's annotation, the owner's output at `-> ret` (`ret.k`:
    /// its component) or at the function body's tail, or the one parameter of
    /// a fixed function called next. Emission reads the same sources
    /// (`emit::Emitter::expected_flow`).
    pub(super) fn expected_flow(&mut self, next: Option<&Stage>) -> Option<Ty> {
        if let Some(t) = self.expected_result(next) {
            return Some(t);
        }
        match next.map(|s| &s.kind) {
            None if self.at_fn_tail => self.owner_ret.clone(),
            Some(StageKind::Ret { proj: Some((k, _)) }) => self
                .owner_ret
                .as_ref()
                .and_then(|r| u32::try_from(*k).ok().and_then(|k| r.component_ty(k)))
                .cloned(),
            _ => self.expected_param(next),
        }
    }

    /// ADR-0051 D3: the type a value flows into as the next stage's input —
    /// the one parameter of a fixed (non-generic) function called there.
    pub(super) fn expected_param(&self, next: Option<&Stage>) -> Option<Ty> {
        let Some(StageKind::Expr(e)) = next.map(|s| &s.kind) else {
            return None;
        };
        let ExprKind::Var(n) = &e.kind else {
            return None;
        };
        match self.fn_sigs.get(name_text(self.source, *n)) {
            Some(sig) if !sig.generic => match sig.params.as_slice() {
                [one] => Some(one.clone()),
                _ => None,
            },
            _ => None,
        }
    }

    /// D3.4: the type a call's result is bound with — the next stage's
    /// annotation (`-> y: [f32; 8]`), or the owner's output at a `-> ret`.
    pub(super) fn expected_result(&mut self, next: Option<&Stage>) -> Option<Ty> {
        match next.map(|s| &s.kind) {
            Some(StageKind::Bind { ty: Some(t), .. }) => {
                let mut sink = Vec::new();
                self.types
                    .resolve_sized(self.source, t, &self.sizes.env, &mut sink)
            }
            Some(StageKind::Ret { proj: None }) => self.owner_ret.clone(),
            _ => None,
        }
    }

    // --- union-find ---------------------------------------------------------

    pub(super) fn fresh_var(&mut self, span: syn::SourceLoc, float: bool) -> usize {
        let v = self.uf.len();
        self.uf.push(v);
        self.state.push(if float {
            LitState::FloatVar
        } else {
            LitState::IntVar
        });
        self.var_span.push((span.start, span.end));
        v
    }

    pub(super) fn find(&mut self, mut x: usize) -> usize {
        while self.uf[x] != x {
            self.uf[x] = self.uf[self.uf[x]];
            x = self.uf[x];
        }
        x
    }

    /// Unify a literal var with a concrete ty (pin it). Reports L1203 on a
    /// var/scalar-kind clash (IntVar↔Float etc.).
    pub(super) fn pin(&mut self, var: usize, ty: &Ty, span: syn::SourceLoc) {
        let r = self.find(var);
        match self.state[r].clone() {
            LitState::IntVar => {
                if matches!(ty, Ty::Int { .. }) {
                    self.state[r] = LitState::Fixed(ty.clone());
                } else {
                    self.diags.push(diag(
                        LCode::LiteralTypeConflict,
                        span,
                        format!(
                            "integer literal cannot have type `{}`",
                            crate::emit::ty_name(ty)
                        ),
                    ));
                    self.state[r] = LitState::Fixed(Ty::i32());
                }
            }
            LitState::FloatVar => {
                if matches!(ty, Ty::Float { .. }) {
                    self.state[r] = LitState::Fixed(ty.clone());
                } else {
                    self.diags.push(diag(
                        LCode::LiteralTypeConflict,
                        span,
                        format!(
                            "float literal cannot have type `{}`",
                            crate::emit::ty_name(ty)
                        ),
                    ));
                    self.state[r] = LitState::Fixed(Ty::f64());
                }
            }
            LitState::Fixed(prev) => {
                if &prev != ty {
                    self.diags.push(diag(
                        LCode::LiteralTypeConflict,
                        span,
                        format!(
                            "literal type conflict: `{}` vs `{}`",
                            crate::emit::ty_name(&prev),
                            crate::emit::ty_name(ty)
                        ),
                    ));
                }
            }
        }
    }

    /// Unify two literal vars (must be same kind).
    pub(super) fn union(&mut self, a: usize, b: usize, span: syn::SourceLoc) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match (self.state[ra].clone(), self.state[rb].clone()) {
            (LitState::Fixed(t), _) => {
                self.uf[rb] = ra;
                self.pin(ra, &t.clone(), span);
                // re-pin rb's prior fixed value if any handled by pin merge below
                if let LitState::Fixed(tb) = self.state[rb].clone() {
                    self.pin(ra, &tb, span);
                }
            }
            (_, LitState::Fixed(t)) => {
                self.uf[ra] = rb;
                self.pin(rb, &t, span);
            }
            (LitState::IntVar, LitState::IntVar) | (LitState::FloatVar, LitState::FloatVar) => {
                self.uf[rb] = ra;
            }
            _ => {
                // IntVar vs FloatVar — clash.
                self.diags.push(diag(
                    LCode::LiteralTypeConflict,
                    span,
                    "integer/float literal width unification clash".to_string(),
                ));
                self.uf[rb] = ra;
            }
        }
    }

    /// Unify a WTy with a concrete ty (the common case at annotation points),
    /// recursing into composites so an array/tuple of unsized literals takes the
    /// annotation's element/component widths.
    pub(super) fn unify_known(&mut self, w: &WTy, ty: &Ty, span: syn::SourceLoc) {
        match (w, ty) {
            (WTy::Var(v), _) => self.pin(*v, ty, span),
            (WTy::Tuple(ws), Ty::Tuple(ts)) if ws.len() == ts.len() => {
                for (cw, ct) in ws.iter().zip(ts.iter()) {
                    self.unify_known(cw, ct, span);
                }
            }
            (WTy::Array(ew, _), Ty::Array { elem, .. }) => {
                self.unify_known(ew, elem, span);
            }
            _ => {}
        }
    }

    /// Unify two WTys (binop operands, phi arms, etc.).
    pub(super) fn unify(&mut self, a: &WTy, b: &WTy, span: syn::SourceLoc) {
        match (a, b) {
            (WTy::Var(x), WTy::Var(y)) => self.union(*x, *y, span),
            (WTy::Var(x), WTy::Known(t)) | (WTy::Known(t), WTy::Var(x)) => self.pin(*x, t, span),
            (WTy::Tuple(xs), WTy::Tuple(ys)) if xs.len() == ys.len() => {
                for (cx, cy) in xs.iter().zip(ys.iter()) {
                    self.unify(cx, cy, span);
                }
            }
            (WTy::Array(ex, _), WTy::Array(ey, _)) => self.unify(ex, ey, span),
            (WTy::Var(x), other) | (other, WTy::Var(x)) => {
                if let Some(t) = self.materialize(other) {
                    self.pin(*x, &t, span);
                }
            }
            _ => {}
        }
    }

    /// Materialize a WTy to a concrete ty if all leaves are currently known
    /// (vars resolve to their current fixed/default state).
    pub(super) fn materialize(&mut self, w: &WTy) -> Option<Ty> {
        self.wty_to_ty(w)
    }

    // --- resolution ---------------------------------------------------------

    /// Resolve every literal var to a concrete ty (defaults i32/f64) and write
    /// `info.lit_ty`. Range-checking is done at emission against the resolved ty.
    pub(super) fn resolve_all(&mut self) {
        for v in 0..self.uf.len() {
            let r = self.find(v);
            let ty = match self.state[r].clone() {
                LitState::Fixed(t) => t,
                LitState::IntVar => Ty::i32(),
                LitState::FloatVar => Ty::f64(),
            };
            let span = self.var_span[v];
            self.info.lit_ty.insert(span, ty);
        }
        for b in std::mem::take(&mut self.pending_blocks) {
            let mut ty_of =
                |(w, fallback): &(WTy, Ty)| self.wty_to_ty(w).unwrap_or(fallback.clone());
            let body_in = match &b.comps {
                Some(comps) => Ty::Tuple(comps.iter().map(&mut ty_of).collect()),
                None => ty_of(&b.body_in),
            };
            let body_out = ty_of(&b.body_out);
            let captures = b
                .captures
                .into_iter()
                .map(|((name, t), w)| (name, ty_of(&(w, t))))
                .collect();
            self.info.blocks.insert(
                b.span,
                BlockSig {
                    body_in,
                    body_out,
                    captures,
                    records: b.records,
                    name: b.name,
                },
            );
        }
    }
}
