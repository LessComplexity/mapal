//! Parser methods for types and size expressions (DESIGN section 14.2).

use super::*;

impl<'a> Parser<'a> {
    // ======================================================================
    // §14.2 — types
    // ======================================================================

    /// Parse a type, guarded by the shared recursion-depth limit (DESIGN §16.1,
    /// J1). Type positions nest via `[`/`(`; deeply nested `[[[…` or `(…,…)`
    /// would otherwise recurse unboundedly and overflow the stack. On exceed,
    /// emit P0011 once and return a `TyKind::Error` placeholder, unwinding like
    /// the expression/block guard.
    pub(super) fn parse_type(&mut self) -> Ty {
        if !self.enter() {
            self.leave();
            return self.error_ty();
        }
        let ty = self.parse_type_inner();
        self.leave();
        ty
    }

    pub(super) fn parse_type_inner(&mut self) -> Ty {
        let start = self.cur_span().start;
        match self.kind() {
            TokenKind::Ident => {
                let name = Name {
                    span: self.bump().span,
                };
                if self.at(TokenKind::Lt)
                    && let Some(args) = self.try_parse_size_args()
                {
                    // ADR-0048 D4: a size-generic type at its sizes.
                    Ty {
                        kind: TyKind::Generic { name, args },
                        span: self.span(start, self.prev_end()),
                    }
                } else if self.at(TokenKind::Lt) {
                    // ⟦P0103⟧ generic args — keep the base name.
                    self.skip_balanced(TokenKind::Lt, TokenKind::Gt);
                    let end = self.prev_end();
                    self.diag(
                        "P0103",
                        self.span(start, end),
                        "generic type arguments are out of Mapal-Core (HANDOFF §4); \
                         planned for Core+1",
                    );
                    Ty {
                        kind: TyKind::Named(name),
                        span: self.span(start, end),
                    }
                } else {
                    Ty {
                        kind: TyKind::Named(name),
                        span: name.span,
                    }
                }
            }
            TokenKind::LBracket => self.parse_array_type(),
            TokenKind::LParen => self.parse_tuple_type(),
            _ => {
                self.diag("P0001", self.cur_span(), "expected a type");
                self.error_ty()
            }
        }
    }

    /// `'[' type ';' INT ']'` (fixed array) or ⟦P0104⟧ `'[' type ']'` (dynamic).
    pub(super) fn parse_array_type(&mut self) -> Ty {
        let start = self.cur_span().start;
        self.bump(); // `[`
        let elem = self.parse_type();
        if self.at(TokenKind::Semi) {
            self.bump();
            // The length: an integer literal or a size expression (ADR-0047 D2).
            let (len, len_span, size) = match self.parse_size_expr() {
                Some(SizeExpr {
                    kind: SizeKind::Int(n),
                    span,
                }) => (n, span, None),
                Some(e) => (0, e.span, Some(e)),
                None => (0, self.here(), None),
            };
            if self.eat(TokenKind::RBracket).is_none() {
                self.diag("P0002", self.cur_span(), "unclosed `[` in array type");
            }
            let span = self.node_span_covering(start, elem.span);
            Ty {
                kind: TyKind::Array {
                    elem: Box::new(elem),
                    len,
                    len_span,
                    size,
                },
                span,
            }
        } else {
            // ⟦P0104⟧ dynamic array / slice — kept.
            if self.eat(TokenKind::RBracket).is_none() {
                self.diag("P0002", self.cur_span(), "unclosed `[` in array type");
            }
            let span = self.node_span_covering(start, elem.span);
            self.diag(
                "P0104",
                span,
                "dynamic array / slice types (`[T]`) are out of Mapal-Core (HANDOFF §4); \
                 planned for Core+1",
            );
            Ty {
                kind: TyKind::Dynamic(Box::new(elem)),
                span,
            }
        }
    }

    /// ADR-0048 D4: `'<' size (',' size)* ','? '>'` after a type name — the
    /// sizes of a size-generic type. Speculative: anything else (a type
    /// argument `Vec<f32>`, an unclosed list) restores the cursor and the
    /// diagnostics and returns `None` (the caller's ⟦P0103⟧ path).
    pub(super) fn try_parse_size_args(&mut self) -> Option<Vec<SizeExpr>> {
        let (pos, ndiag, cooldown, depth) = (
            self.pos,
            self.diagnostics.len(),
            self.cooldown_at,
            self.depth_exceeded,
        );
        self.bump(); // `<`
        let mut args = Vec::new();
        let ok = loop {
            match self.parse_size_expr() {
                Some(e) => args.push(e),
                None => break false,
            }
            if self.eat(TokenKind::Comma).is_some() {
                if self.at(TokenKind::Gt) {
                    self.bump();
                    break true;
                }
                continue;
            }
            break self.eat(TokenKind::Gt).is_some();
        };
        // A type in argument position — a capitalized (type) name,
        // `Result<T, E>`, or `bool` — is a type argument. A number format
        // (`bf16`, `f32`, …) is a format argument (ADR-0051 D1:
        // `Matrix<64, 64, bf16>`), resolved by lowering against the
        // declared list.
        let typeish = |e: &SizeExpr| match e.kind {
            SizeKind::Name(n) => {
                let t = self.text(n.span);
                t.starts_with(|c: char| c.is_ascii_uppercase()) || t == "bool"
            }
            _ => false,
        };
        if ok && self.diagnostics.len() == ndiag && !args.iter().any(typeish) {
            return Some(args);
        }
        self.pos = pos;
        self.diagnostics.truncate(ndiag);
        self.cooldown_at = cooldown;
        self.depth_exceeded = depth;
        None
    }

    /// `size := term (('+' | '-') term)*` (ADR-0047 D2). `None` (with P0001)
    /// when no length is there at all; `/` and `%` are P0119.
    pub(super) fn parse_size_expr(&mut self) -> Option<SizeExpr> {
        if !self.enter() {
            self.leave();
            return None;
        }
        let out = self.parse_size_sum();
        self.leave();
        out
    }

    pub(super) fn parse_size_sum(&mut self) -> Option<SizeExpr> {
        let mut lhs = self.parse_size_term()?;
        loop {
            let op = match self.kind() {
                TokenKind::Plus => SizeOp::Add,
                TokenKind::Minus => SizeOp::Sub,
                _ => break,
            };
            self.bump();
            let rhs = self.parse_size_term()?;
            lhs = Self::size_binary(op, lhs, rhs);
        }
        Some(lhs)
    }

    pub(super) fn parse_size_term(&mut self) -> Option<SizeExpr> {
        let mut lhs = self.parse_size_atom()?;
        loop {
            match self.kind() {
                TokenKind::Star => {
                    self.bump();
                    let rhs = self.parse_size_atom()?;
                    lhs = Self::size_binary(SizeOp::Mul, lhs, rhs);
                }
                TokenKind::Slash | TokenKind::Percent => {
                    let sp = self.cur_span();
                    self.diag(
                        "P0119",
                        sp,
                        "a size expression takes `+`, `-` and `*` only: `/` and `%` are not \
                         invertible, so a call could not infer the size (ADR-0047)",
                    );
                    self.bump();
                    let _ = self.parse_size_atom();
                }
                _ => break,
            }
        }
        Some(lhs)
    }

    pub(super) fn parse_size_atom(&mut self) -> Option<SizeExpr> {
        let sp = self.cur_span();
        match self.kind() {
            TokenKind::Int => {
                self.bump();
                Some(SizeExpr {
                    kind: SizeKind::Int(self.parse_int(sp)),
                    span: sp,
                })
            }
            TokenKind::Ident => {
                self.bump();
                Some(SizeExpr {
                    kind: SizeKind::Name(Name { span: sp }),
                    span: sp,
                })
            }
            TokenKind::LParen => {
                let start = sp.start;
                self.bump();
                let inner = self.parse_size_expr()?;
                if self.eat(TokenKind::RParen).is_none() {
                    self.diag(
                        "P0002",
                        self.cur_span(),
                        "unclosed `(` in a size expression",
                    );
                }
                // Parentheses only group: the node keeps the inner shape with
                // the parenthesized span.
                Some(SizeExpr {
                    kind: inner.kind,
                    span: self.node_span_covering(start, inner.span),
                })
            }
            _ => {
                self.diag(
                    "P0001",
                    sp,
                    "array length must be an integer literal or a size expression \
                     (`[f32; 4096]`, `[f32; rows * columns]`)",
                );
                None
            }
        }
    }

    pub(super) fn size_binary(op: SizeOp, lhs: SizeExpr, rhs: SizeExpr) -> SizeExpr {
        let span = SourceLoc::new(lhs.span.start, rhs.span.end.max(lhs.span.end));
        SizeExpr {
            kind: SizeKind::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            },
            span,
        }
    }

    /// `'(' type (',' type)+ ','? ')'` — a tuple type, ≥2 elements.
    pub(super) fn parse_tuple_type(&mut self) -> Ty {
        let start = self.cur_span().start;
        self.bump(); // `(`
        let mut elems = Vec::new();
        while !self.at(TokenKind::RParen) && !self.at_eof() {
            let before = self.pos;
            elems.push(self.parse_type());
            if self.at(TokenKind::Comma) {
                self.bump();
            } else {
                break;
            }
            if self.pos == before {
                break;
            }
        }
        if self.eat(TokenKind::RParen).is_none() {
            self.diag("P0002", self.cur_span(), "unclosed `(` in tuple type");
        }
        // Cover the elements (a recovery element may sit outside the consumed
        // range, J2).
        let mut span = self.node_span(start);
        for e in &elems {
            span = SourceLoc::new(span.start.min(e.span.start), span.end.max(e.span.end));
        }
        if elems.len() < 2 {
            self.diag("P0001", span, "a tuple type needs at least two elements");
        }
        Ty {
            kind: TyKind::Tuple(elems),
            span,
        }
    }

    pub(super) fn error_ty(&self) -> Ty {
        Ty {
            kind: TyKind::Error,
            span: self.here(),
        }
    }
}
