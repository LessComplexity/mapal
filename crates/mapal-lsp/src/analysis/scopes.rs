//! The walker's types and scopes: type syntax, frames, binding and lookup.

use super::*;

impl<'a> Walker<'a> {
    // --- types --------------------------------------------------------------

    pub(super) fn ty(&mut self, t: &mapal_syntax::Ty) -> Ty {
        match &t.kind {
            TyKind::Named(n) => self.type_name(*n, Vec::new()),
            TyKind::Generic { name, args } => {
                let args = args
                    .iter()
                    .map(|a| {
                        self.size_expr(a);
                        self.text(a.span).to_string()
                    })
                    .collect();
                self.type_name(*name, args)
            }
            TyKind::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| self.ty(t)).collect()),
            TyKind::Array {
                elem,
                len_span,
                size,
                ..
            } => {
                let elem = Box::new(self.ty(elem));
                let len = match size {
                    Some(s) => {
                        self.size_expr(s);
                        self.text(s.span).to_string()
                    }
                    None => self.text(*len_span).to_string(),
                };
                Ty::Array { elem, len }
            }
            TyKind::Dynamic(elem) => Ty::Array {
                elem: Box::new(self.ty(elem)),
                len: "?".into(),
            },
            TyKind::Error => Ty::Unknown,
        }
    }

    pub(super) fn type_name(&mut self, n: Name, args: Vec<String>) -> Ty {
        let text = self.name(n);
        if is_scalar(text) {
            self.lex(n.span, Class::Primitive);
            return Ty::Prim(text.to_string());
        }
        // ADR-0051 D1: a format parameter in a type position.
        if let Some(&id) = self.sizes.get(text)
            && self.defs[id].kind == DefKind::FormatParam
        {
            self.occ(n.span, id, Role::TypeUse);
            return Ty::Prim(text.to_string());
        }
        if let Some(&id) = self.types.get(text) {
            self.occ(n.span, id, Role::TypeUse);
        }
        Ty::Named {
            name: text.to_string(),
            args,
        }
    }

    pub(super) fn size_expr(&mut self, s: &SizeExpr) {
        match &s.kind {
            SizeKind::Int(_) => {}
            SizeKind::Name(n) => {
                let text = self.name(*n);
                // A format argument (`Matrix<64, 64, bf16>`, `to<bf16>`).
                if is_scalar(text) {
                    self.lex(n.span, Class::Primitive);
                } else if let Some(&id) = self.sizes.get(text) {
                    self.occ(n.span, id, Role::Read);
                } else if let Some(&id) = self.globals.get(text) {
                    self.occ(n.span, id, Role::Read);
                }
            }
            SizeKind::Binary { lhs, rhs, .. } => {
                self.size_expr(lhs);
                self.size_expr(rhs);
            }
        }
    }

    /// ADR-0041 D2: the precision members of a body's `: compute`.
    pub(super) fn compute(&mut self, b: &Block) {
        let Some(p) = b.compute else {
            return;
        };
        let spans: Vec<SourceLoc> = self
            .tokens
            .iter()
            .filter(|t| {
                t.kind == TokenKind::Ident
                    && t.span.start >= p.span.start
                    && t.span.end <= p.span.end
            })
            .map(|t| t.span)
            .collect();
        for s in spans {
            self.lex(s, Class::Primitive);
        }
    }

    // --- scopes -------------------------------------------------------------

    pub(super) fn push(&mut self, kind: FrameKind, end: u32) {
        self.frames.push(Frame {
            names: BTreeMap::new(),
            kind,
            end,
        });
    }

    pub(super) fn pop(&mut self) {
        self.frames.pop();
    }

    pub(super) fn lookup_local(&self, name: &str) -> Option<(DefId, usize)> {
        self.frames
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, f)| f.names.get(name).map(|&id| (id, i)))
    }

    pub(super) fn is_capture(&self, frame: usize) -> bool {
        self.frames[frame + 1..]
            .iter()
            .any(|f| f.kind == FrameKind::Body)
    }

    /// Bind a fresh local `name` into frame `target` (the innermost when
    /// `None`).
    pub(super) fn bind(
        &mut self,
        name: Name,
        kind: DefKind,
        mutable: bool,
        ty: Ty,
        span: SourceLoc,
        target: Option<usize>,
    ) -> DefId {
        let text = self.name(name).to_string();
        let Some(last) = self.frames.len().checked_sub(1) else {
            return usize::MAX;
        };
        let idx = target.unwrap_or(last).min(last);
        let id = self.def(Def {
            name: text.clone(),
            kind,
            name_span: name.span,
            span,
            ty,
            mutable,
            container: None,
            visible: SourceLoc::new(name.span.start, self.frames[idx].end.max(name.span.end)),
        });
        self.occ(name.span, id, Role::Decl);
        self.frames[idx].names.insert(text, id);
        id
    }

    /// The frame a routing guard's exit binding lands in: the one enclosing
    /// the innermost loop.
    pub(super) fn loop_exit_frame(&self) -> Option<usize> {
        self.frames
            .iter()
            .rposition(|f| f.kind == FrameKind::Loop)
            .map(|i| i.saturating_sub(1))
    }
}
