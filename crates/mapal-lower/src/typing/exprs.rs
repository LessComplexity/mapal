//! The typing walk over expressions.

use super::*;

impl Walk<'_> {
    // --- expressions --------------------------------------------------------

    pub(super) fn expr(&mut self, expr: &Expr, scope: &mut ScopeStack<WTy>) -> WTy {
        // ADR-0047 D4: a size expression in value position is one integer
        // literal (unpinned `Ty::index_ty`).
        if matches!(expr.kind, ExprKind::Binary { .. })
            && let Some(v) = self.size_fold(expr, scope)
        {
            let var = self.fresh_var(expr.span, false);
            if v > i128::from(i32::MAX) || v < i128::from(i32::MIN) {
                self.pin(var, &Ty::i64(), expr.span);
            }
            return WTy::Var(var);
        }
        match &expr.kind {
            ExprKind::Int(_) => WTy::Var(self.fresh_var(expr.span, false)),
            ExprKind::Float => WTy::Var(self.fresh_var(expr.span, true)),
            ExprKind::Bool(_) => WTy::Known(Ty::Bool),
            ExprKind::Str => WTy::Known(Ty::Str),
            ExprKind::Var(n) => {
                let text = name_text(self.source, *n);
                match scope.resolve(text) {
                    Some((b, _)) => b.clone(),
                    None if self.types.globals().consts.contains_key(text) => {
                        self.global_const(text, expr.span)
                    }
                    None => match self.sizes.env.get(text).copied() {
                        // ADR-0047 D4: an integer literal of the size's value;
                        // unpinned it is `Ty::index_ty(n)`.
                        Some(v) => {
                            let var = self.fresh_var(expr.span, false);
                            if v > i32::MAX as u64 {
                                self.pin(var, &Ty::i64(), expr.span);
                            }
                            WTy::Var(var)
                        }
                        None => WTy::Unknown,
                    },
                }
            }
            ExprKind::Hole => WTy::Unknown,
            ExprKind::Unary { op, operand, .. } => {
                let inner = self.expr(operand, scope);
                match op {
                    UnOp::Neg => inner,
                    UnOp::Not => WTy::Known(Ty::Bool),
                }
            }
            ExprKind::Binary {
                op,
                lhs,
                rhs,
                op_span,
            } => {
                let l = self.expr(lhs, scope);
                let r = self.expr(rhs, scope);
                self.unify(&l, &r, *op_span);
                use syn::BinOp::*;
                match op {
                    Add | Sub | Mul | Div | Mod => l,
                    Eq | Ne | Lt | Gt | Le | Ge | And | Or => WTy::Known(Ty::Bool),
                }
            }
            ExprKind::Member { base, field } => {
                let bty = self.expr(base, scope);
                match field {
                    syn::MemberField::Named(fname) => {
                        let fn_ = name_text(self.source, *fname);
                        let bty = self.through_default(&bty, |t| match t {
                            Ty::Struct { fields, .. } => fields.iter().any(|(n, _)| n == fn_),
                            _ => false,
                        });
                        let bt = self.wty_to_ty(&bty);
                        if let Some(Ty::Struct { fields, .. }) = bt.as_ref() {
                            let fn_ = name_text(self.source, *fname);
                            if let Some((_, t)) = fields.iter().find(|(n, _)| n == fn_) {
                                return WTy::Known(t.clone());
                            }
                        }
                        WTy::Unknown
                    }
                    syn::MemberField::Index { value, .. } => {
                        // A tuple of literals keeps its vars: `acc.1` of a
                        // `(1.0, 1.0)` seed takes the width its use gives.
                        if let WTy::Tuple(ws) = &bty
                            && let Some(w) = ws.get(*value as usize)
                        {
                            return w.clone();
                        }
                        let bt = self.wty_to_ty(&bty);
                        // Skip the widening for an index > u32::MAX (it would
                        // truncate and unify against the wrong component);
                        // emit.rs rejects it with L1205 (LOWER-MEMBERIDX-TYPING-TRUNC).
                        if *value <= u32::MAX as u64
                            && let Some(t) = bt.as_ref()
                            && let Some(c) = t.component_ty(*value as u32)
                        {
                            return WTy::Known(c.clone());
                        }
                        WTy::Unknown
                    }
                }
            }
            ExprKind::Index { base, index } => {
                let bty = self.expr(base, scope);
                let bty = self.through_default(&bty, |t| matches!(t, Ty::Array { .. }));
                let iv = self.expr(index, scope);
                // index is an integer; if a lit var, leave it (defaults i32) —
                // unless the literal itself is past i32 (an extent past 2³¹
                // indexes in `i64`, its index type).
                if let Some(n) = self.int_const(index, scope)
                    && n > i32::MAX as u64
                    && n <= i64::MAX as u64
                {
                    self.unify_known(&iv, &Ty::i64(), index.span);
                }
                // An array of literals keeps its element var (`[1.0, 2.0]`
                // read as `f32`).
                if let WTy::Array(e, _) = &bty {
                    return (**e).clone();
                }
                let bt = self.wty_to_ty(&bty);
                match bt.as_ref() {
                    Some(Ty::Array { elem, .. }) => WTy::Known((**elem).clone()),
                    _ => WTy::Unknown,
                }
            }
            ExprKind::Tuple(es) => {
                // Keep components as WTys so a later annotation can reach their
                // vars (composite literal-width inference).
                let ws: Vec<WTy> = es.iter().map(|e| self.expr(e, scope)).collect();
                if ws.len() >= 2 {
                    WTy::Tuple(ws)
                } else {
                    WTy::Unknown
                }
            }
            ExprKind::Array(es) => {
                let mut ws: Vec<WTy> = Vec::with_capacity(es.len());
                for e in es {
                    let w = self.expr(e, scope);
                    if let Some(prev) = ws.first() {
                        self.unify(prev, &w, e.span);
                    }
                    ws.push(w);
                }
                match ws.into_iter().next() {
                    Some(first) => WTy::Array(Box::new(first), es.len() as u64),
                    None => WTy::Unknown,
                }
            }
            ExprKind::Struct { name, fields } => {
                let tname = name_text(self.source, *name);
                // ADR-0048 D4: a size-generic type's literal, its sizes from
                // its fields' types (where they are known).
                if let Some(td) = self
                    .types
                    .decl(tname)
                    .filter(|td| !td.generics.is_empty())
                    .cloned()
                {
                    let mut tys = Vec::new();
                    let mut ws = Vec::new();
                    for f in &td.fields {
                        let fname = name_text(self.source, f.name);
                        let w = fields
                            .iter()
                            .find(|fi| name_text(self.source, fi.name) == fname)
                            .map(|fi| match &fi.value {
                                Some(v) => self.expr(v, scope),
                                None => scope
                                    .resolve(fname)
                                    .map(|(b, _)| b.clone())
                                    .unwrap_or(WTy::Unknown),
                            })
                            .unwrap_or(WTy::Unknown);
                        // A literal field (no type yet) carries no size.
                        tys.push(match &w {
                            WTy::Var(_) => Some(Ty::i32()),
                            _ => self.wty_to_ty(&w),
                        });
                        ws.push(w);
                    }
                    let Some(tys) = tys.into_iter().collect::<Option<Vec<Ty>>>() else {
                        return WTy::Unknown;
                    };
                    let Some(ty) =
                        crate::sizes::infer_type(self.source, self.types, &td, &tys, expr.span)
                            .ok()
                            .and_then(|vs| self.types.instance(self.source, tname, &vs, expr.span))
                    else {
                        return WTy::Unknown;
                    };
                    // The fields' literals take the instance's field types.
                    if let Ty::Struct { fields: decl, .. } = &ty {
                        for (w, (_, ft)) in ws.iter().zip(decl) {
                            self.unify_known(w, ft, expr.span);
                        }
                    }
                    return WTy::Known(ty);
                }
                if let Some(sty) = self.types.get(tname).cloned() {
                    if let Ty::Struct { fields: decl, .. } = &sty {
                        for fi in fields {
                            let fname = name_text(self.source, fi.name);
                            let declty = decl
                                .iter()
                                .find(|(n, _)| n == fname)
                                .map(|(_, t)| t.clone());
                            let w = match &fi.value {
                                Some(v) => self.expr(v, scope),
                                None => scope
                                    .resolve(fname)
                                    .map(|(b, _)| b.clone())
                                    .unwrap_or(WTy::Unknown),
                            };
                            if let Some(dt) = declty {
                                self.unify_known(&w, &dt, fi.span);
                            }
                        }
                    }
                    WTy::Known(sty)
                } else {
                    WTy::Unknown
                }
            }
            // ADR-0031: `ExprKind::Call` is always P0108-rejected upstream
            // (the ADR-0029 carve is gone); no clean tree contains one.
            _ => WTy::Unknown,
        }
    }

    /// Lower an OpShorthand expr where the single `Hole` leaf is `cur`.
    pub(super) fn expr_with_hole(
        &mut self,
        expr: &Expr,
        hole: &WTy,
        scope: &mut ScopeStack<WTy>,
    ) -> WTy {
        match &expr.kind {
            ExprKind::Hole => hole.clone(),
            ExprKind::Unary { op, operand, .. } => {
                let inner = self.expr_with_hole(operand, hole, scope);
                match op {
                    UnOp::Neg => inner,
                    UnOp::Not => WTy::Known(Ty::Bool),
                }
            }
            ExprKind::Binary {
                op,
                lhs,
                rhs,
                op_span,
            } => {
                let l = self.expr_with_hole(lhs, hole, scope);
                let r = self.expr_with_hole(rhs, hole, scope);
                self.unify(&l, &r, *op_span);
                use syn::BinOp::*;
                match op {
                    Add | Sub | Mul | Div | Mod => l,
                    _ => WTy::Known(Ty::Bool),
                }
            }
            _ => self.expr(expr, scope),
        }
    }

    /// Resolve a WTy to a concrete ty if currently known (vars resolve to their
    /// current fixed state, else `None` until `resolve_all`).
    pub(super) fn wty_to_ty(&mut self, w: &WTy) -> Option<Ty> {
        match w {
            WTy::Known(t) => Some(t.clone()),
            WTy::Var(v) => {
                let r = self.find(*v);
                match self.state[r].clone() {
                    LitState::Fixed(t) => Some(t),
                    LitState::IntVar => Some(Ty::i32()),
                    LitState::FloatVar => Some(Ty::f64()),
                }
            }
            WTy::Tuple(ws) => {
                let mut ts = Vec::with_capacity(ws.len());
                for cw in ws.clone() {
                    ts.push(self.wty_to_ty(&cw)?);
                }
                Some(Ty::Tuple(ts))
            }
            WTy::Array(ew, size) => {
                let e = self.wty_to_ty(ew)?;
                Some(Ty::Array {
                    elem: Box::new(e),
                    size: *size,
                })
            }
            WTy::Unknown => None,
        }
    }
}
