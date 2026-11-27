//! The walker's expressions: reads, field access and element types.

use super::*;

impl<'a> Walker<'a> {
    // --- expressions --------------------------------------------------------

    pub(super) fn expr(&mut self, e: &Expr) -> Ty {
        match &e.kind {
            ExprKind::Int(_) => Ty::Prim("i32".into()),
            ExprKind::Float => Ty::Prim("f32".into()),
            ExprKind::Str => Ty::Prim("str".into()),
            ExprKind::Bool(_) => Ty::Prim("bool".into()),
            ExprKind::Unit => Ty::Tuple(Vec::new()),
            ExprKind::Hole | ExprKind::Error => Ty::Unknown,
            ExprKind::Var(n) => self.read(*n),
            ExprKind::Unary { op, operand, .. } => {
                let t = self.expr(operand);
                match op {
                    mapal_syntax::UnOp::Not => Ty::Prim("bool".into()),
                    mapal_syntax::UnOp::Neg => t,
                }
            }
            ExprKind::Binary { op, lhs, rhs, .. } => {
                use mapal_syntax::BinOp::*;
                let l = self.expr(lhs);
                let r = self.expr(rhs);
                match op {
                    Eq | Ne | Lt | Gt | Le | Ge | And | Or => Ty::Prim("bool".into()),
                    _ if l.is_known() && !matches!(&l, Ty::Prim(p) if p == "i32") => l,
                    _ if r.is_known() => r,
                    _ => l,
                }
            }
            ExprKind::Member { base, field } => {
                let b = self.expr(base);
                match field {
                    MemberField::Named(f) => self.field_access(&b, *f),
                    MemberField::Index { value, .. } => match &b {
                        Ty::Tuple(ts) => ts.get(*value as usize).cloned().unwrap_or(Ty::Unknown),
                        _ => Ty::Unknown,
                    },
                }
            }
            ExprKind::Index { base, index } => {
                let b = self.expr(base);
                self.expr(index);
                self.elem_through_default(&b)
            }
            ExprKind::Tuple(es) => Ty::Tuple(es.iter().map(|e| self.expr(e)).collect()),
            ExprKind::Array(es) => {
                let ts: Vec<Ty> = es.iter().map(|e| self.expr(e)).collect();
                Ty::Array {
                    elem: Box::new(ts.into_iter().next().unwrap_or(Ty::Unknown)),
                    len: es.len().to_string(),
                }
            }
            ExprKind::Struct { name, fields } => {
                let tname = self.name(*name).to_string();
                if let Some(&id) = self.types.get(&tname) {
                    self.occ(name.span, id, Role::TypeUse);
                }
                for f in fields {
                    match &f.value {
                        Some(v) => {
                            let fname = self.name(f.name);
                            if let Some(&fid) = self.fields.get(&tname).and_then(|m| m.get(fname)) {
                                self.occ(f.name.span, fid, Role::Read);
                            }
                            self.expr(v);
                        }
                        // A pun `{ keys }` reads the local of that name and
                        // names the field: both occurrences, the local's first
                        // (definition goes to the value, as in Rust).
                        None => {
                            self.read(f.name);
                            let fname = self.name(f.name);
                            let field = self.fields.get(&tname).and_then(|m| m.get(fname)).copied();
                            if let Some(fid) = field {
                                self.occ(f.name.span, fid, Role::Read);
                            }
                            self.puns.push(f.name.span);
                        }
                    }
                }
                Ty::Named {
                    name: tname,
                    args: Vec::new(),
                }
            }
            ExprKind::Call { callee, args } => {
                let t = self.expr(callee);
                for a in args {
                    self.expr(a);
                }
                t
            }
            ExprKind::Question(inner) => self.expr(inner),
            // Built only in stage position (`stage`); walk it the same way.
            ExprKind::Generic { name, args } => {
                let wire = Wire {
                    ty: Ty::Unknown,
                    head: None,
                };
                self.generic_stage(*name, args, &wire)
            }
        }
    }

    pub(super) fn read(&mut self, n: Name) -> Ty {
        let text = self.name(n);
        if let Some((id, _)) = self.lookup_local(text) {
            self.occ(n.span, id, Role::Read);
            return self.defs[id].ty.clone();
        }
        if let Some(&id) = self.sizes.get(text) {
            self.occ(n.span, id, Role::Read);
            return match self.defs[id].kind {
                DefKind::SizeParam => Ty::Prim("i32".into()),
                _ => Ty::Unknown,
            };
        }
        if let Some(&id) = self.globals.get(text) {
            self.occ(n.span, id, Role::Read);
            return self.defs[id].ty.clone();
        }
        if let Some(&id) = self.fns.get(text) {
            self.occ(n.span, id, Role::Call);
            return self.defs[id].ty.clone();
        }
        if is_builtin_name(text) {
            self.lex(n.span, Class::Builtin);
        }
        Ty::Unknown
    }

    /// `base.f`: the field of the base's type, else of its `_` (ADR-0048 D5:
    /// a type with a default interpretation reads through it).
    pub(super) fn field_access(&mut self, base: &Ty, f: Name) -> Ty {
        let fname = self.name(f);
        let mut cur = base.clone();
        for _ in 0..8 {
            let Some(tname) = cur.named().map(str::to_string) else {
                return Ty::Unknown;
            };
            let Some(fields) = self.fields.get(&tname) else {
                return Ty::Unknown;
            };
            if let Some(&fid) = fields.get(fname) {
                self.occ(f.span, fid, Role::Read);
                return self.defs[fid].ty.clone();
            }
            match fields.get("_") {
                Some(&d) => cur = self.defs[d].ty.clone(),
                None => return Ty::Unknown,
            }
        }
        Ty::Unknown
    }

    pub(super) fn elem_through_default(&self, t: &Ty) -> Ty {
        let mut cur = t.clone();
        for _ in 0..8 {
            match &cur {
                Ty::Array { elem, .. } => return (**elem).clone(),
                Ty::Named { name, .. } => match self.fields.get(name).and_then(|m| m.get("_")) {
                    Some(&d) => cur = self.defs[d].ty.clone(),
                    None => return Ty::Unknown,
                },
                _ => return Ty::Unknown,
            }
        }
        Ty::Unknown
    }
}
