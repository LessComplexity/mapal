//! Parser methods for expressions (DESIGN section 14.6).

use super::*;

impl<'a> Parser<'a> {
    // ======================================================================
    // §14.6 — expressions
    // ======================================================================

    pub(super) fn parse_expr(&mut self) -> Expr {
        if !self.enter() {
            self.leave();
            return self.error_expr();
        }
        let e = self.parse_or();
        self.leave();
        e
    }

    /// `or := and ( '||' and )*` — level 7.
    pub(super) fn parse_or(&mut self) -> Expr {
        let mut lhs = self.parse_and();
        while self.at(TokenKind::PipePipe) {
            let op_span = self.bump().span;
            let rhs = self.parse_and();
            lhs = self.binary(BinOp::Or, op_span, lhs, rhs);
        }
        lhs
    }

    /// `and := cmp ( '&&' cmp )*` — level 6.
    pub(super) fn parse_and(&mut self) -> Expr {
        let mut lhs = self.parse_cmp();
        while self.at(TokenKind::AmpAmp) {
            let op_span = self.bump().span;
            let rhs = self.parse_cmp();
            lhs = self.binary(BinOp::And, op_span, lhs, rhs);
        }
        lhs
    }

    /// `cmp := add ( CMPOP add )?` — level 5, NON-ASSOCIATIVE: a second CMPOP →
    /// P0007 (W14).
    pub(super) fn parse_cmp(&mut self) -> Expr {
        let lhs = self.parse_add();
        // ⟦P0009⟧ a `<-` where a comparison could go *inside* an expression:
        // `(i<-1)` lexes as `(i <- 1)` (W2); the meant comparison is `(i < -1)`.
        // Only fire when nested in a bracket group (`paren_depth > 0`); at
        // statement level a stray `<-` is the statement's concern (P0008/P0001).
        if self.at(TokenKind::BackArrow) && self.paren_depth > 0 {
            let op_span = self.bump().span;
            self.diag_fix(
                "P0009",
                op_span,
                "`<-` cannot appear inside an expression; `i<-1` lexes as `i <- 1`. \
                 For a comparison write `i < -1`",
                SuggestedFix {
                    span: op_span,
                    replacement: "< -".to_string(),
                    label: "did you mean `< -` (comparison with a negative)?",
                },
            );
            let rhs = self.parse_add();
            let neg = Expr {
                kind: ExprKind::Unary {
                    op: UnOp::Neg,
                    op_span,
                    operand: Box::new(rhs.clone()),
                },
                span: self.span(op_span.start, rhs.span.end.max(op_span.end)),
            };
            return self.binary(BinOp::Lt, op_span, lhs, neg);
        }
        if let Some(op) = cmp_op(self.kind()) {
            let op_span = self.bump().span;
            let rhs = self.parse_add();
            let result = self.binary(op, op_span, lhs, rhs);
            // A second comparison at this level is non-associative. Recovery
            // absorbs *all* further `CMPOP add` pairs under the single P0007 so a
            // chain like `a < b < c < d` reports exactly one P0007 with no P0001
            // cascade (DESIGN §17 P0007: one P0007 per chain).
            if cmp_op(self.kind()).is_some() {
                let span = self.cur_span();
                self.diag(
                    "P0007",
                    span,
                    "comparison operators are non-associative; parenthesize (e.g. `(a < b) < c`)",
                );
                // Parse and discard the rest at this level (lhs kept).
                while cmp_op(self.kind()).is_some() {
                    let _op_span = self.bump().span;
                    let _rhs = self.parse_add();
                }
            }
            result
        } else {
            lhs
        }
    }

    /// `add := mul ( ('+'|'-') mul )*` — level 4.
    pub(super) fn parse_add(&mut self) -> Expr {
        let mut lhs = self.parse_mul();
        loop {
            let op = match self.kind() {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => break,
            };
            let op_span = self.bump().span;
            let rhs = self.parse_mul();
            lhs = self.binary(op, op_span, lhs, rhs);
        }
        lhs
    }

    /// `mul := unary ( ('*'|'/'|'%') unary )*` — level 3.
    pub(super) fn parse_mul(&mut self) -> Expr {
        let mut lhs = self.parse_unary();
        loop {
            let op = match self.kind() {
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                TokenKind::Percent => BinOp::Mod,
                _ => break,
            };
            let op_span = self.bump().span;
            let rhs = self.parse_unary();
            lhs = self.binary(op, op_span, lhs, rhs);
        }
        lhs
    }

    /// `unary := ('-'|'!') unary | postfix` — binds tighter than `*`, looser than
    /// postfix (W15).
    pub(super) fn parse_unary(&mut self) -> Expr {
        let (op, op_span) = match self.kind() {
            TokenKind::Minus => (UnOp::Neg, self.cur_span()),
            TokenKind::Bang => (UnOp::Not, self.cur_span()),
            _ => return self.parse_postfix(),
        };
        self.bump();
        if !self.enter() {
            self.leave();
            return self.error_expr();
        }
        let operand = self.parse_unary();
        self.leave();
        let span = self.span(op_span.start, operand.span.end.max(op_span.end));
        debug_assert!(
            operand.span.start >= span.start && operand.span.end <= span.end,
            "unary operand span not within parent"
        );
        Expr {
            kind: ExprKind::Unary {
                op,
                op_span,
                operand: Box::new(operand),
            },
            span,
        }
    }

    /// Continue a binary expression climb from an existing `lhs` at `level`. Used
    /// by op-shorthand (lhs = Hole). Implements the §14.6 precedence levels.
    pub(super) fn parse_binop_rhs(&mut self, lhs: Expr, level: u8) -> Expr {
        match level {
            7 => {
                // `|| and`*
                let mut lhs = lhs;
                while self.at(TokenKind::PipePipe) {
                    let op_span = self.bump().span;
                    let rhs = self.parse_and();
                    lhs = self.binary(BinOp::Or, op_span, lhs, rhs);
                }
                lhs
            }
            6 => {
                let mut lhs = lhs;
                while self.at(TokenKind::AmpAmp) {
                    let op_span = self.bump().span;
                    let rhs = self.parse_cmp();
                    lhs = self.binary(BinOp::And, op_span, lhs, rhs);
                }
                // After consuming `&&` chains, a `||` may still follow.
                self.parse_binop_rhs(lhs, 7)
            }
            5 => {
                // Single comparison from lhs; level 5 is NON-ASSOCIATIVE, so a
                // second `CMPOP` is P0007 (W14), mirroring `parse_cmp`. Recovery
                // absorbs *all* further `CMPOP add` pairs under the single P0007
                // (one P0007 per chain, no P0001 cascade — DESIGN §17 P0007,
                // enforced on both expression paths).
                let lhs = if let Some(op) = cmp_op(self.kind()) {
                    let op_span = self.bump().span;
                    let rhs = self.parse_add();
                    let result = self.binary(op, op_span, lhs, rhs);
                    if cmp_op(self.kind()).is_some() {
                        let span = self.cur_span();
                        self.diag(
                            "P0007",
                            span,
                            "comparison operators are non-associative; parenthesize \
                             (e.g. `(a < b) < c`)",
                        );
                        while cmp_op(self.kind()).is_some() {
                            let _op_span = self.bump().span;
                            let _rhs = self.parse_add();
                        }
                    }
                    result
                } else {
                    lhs
                };
                self.parse_binop_rhs(lhs, 6)
            }
            4 => {
                let mut lhs = lhs;
                loop {
                    let op = match self.kind() {
                        TokenKind::Plus => BinOp::Add,
                        TokenKind::Minus => BinOp::Sub,
                        _ => break,
                    };
                    let op_span = self.bump().span;
                    let rhs = self.parse_mul();
                    lhs = self.binary(op, op_span, lhs, rhs);
                }
                self.parse_binop_rhs(lhs, 5)
            }
            3 => {
                let mut lhs = lhs;
                loop {
                    let op = match self.kind() {
                        TokenKind::Star => BinOp::Mul,
                        TokenKind::Slash => BinOp::Div,
                        TokenKind::Percent => BinOp::Mod,
                        _ => break,
                    };
                    let op_span = self.bump().span;
                    let rhs = self.parse_unary();
                    lhs = self.binary(op, op_span, lhs, rhs);
                }
                self.parse_binop_rhs(lhs, 4)
            }
            _ => lhs,
        }
    }

    /// Build a `Binary` node with J2 span checks.
    pub(super) fn binary(&self, op: BinOp, op_span: SourceLoc, lhs: Expr, rhs: Expr) -> Expr {
        let start = lhs.span.start;
        let end = rhs.span.end.max(op_span.end);
        debug_assert!(start <= end, "binary span start>end");
        debug_assert!(
            lhs.span.start >= start && lhs.span.end <= end,
            "binary lhs span not within parent"
        );
        debug_assert!(
            rhs.span.start >= start && rhs.span.end <= end,
            "binary rhs span not within parent"
        );
        Expr {
            kind: ExprKind::Binary {
                op,
                op_span,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            },
            span: self.span(start, end),
        }
    }

    /// `postfix := primary ( '.' IDENT | '.' INT | '[' expr ']' | '(' args ')'
    ///           | '?' )*`.
    pub(super) fn parse_postfix(&mut self) -> Expr {
        let mut base = self.parse_primary();
        // Each postfix wraps `base` one level deeper, so the run counts toward
        // the shared depth limit (J1): a loop builds the tree, but every walk
        // over it (and its `Drop`) recurses. Without this, `x(…)(…)…` — P0011
        // recovery in `((((…` included, where each further `(` reads as a
        // call — grows a tree as deep as the input is long.
        let mut wraps = 0u32;
        loop {
            let wraps_base = matches!(
                self.kind(),
                TokenKind::Dot | TokenKind::LBracket | TokenKind::LParen | TokenKind::Question
            );
            if wraps_base {
                if !self.enter() {
                    self.leave();
                    break;
                }
                wraps += 1;
            }
            match self.kind() {
                TokenKind::Dot => {
                    self.bump();
                    let field = match self.kind() {
                        TokenKind::Ident => MemberField::Named(Name {
                            span: self.bump().span,
                        }),
                        TokenKind::Int => {
                            let span = self.bump().span;
                            MemberField::Index {
                                value: self.parse_int(span),
                                span,
                            }
                        }
                        _ => {
                            self.diag(
                                "P0001",
                                self.cur_span(),
                                "expected a field name or tuple index after `.`",
                            );
                            // Place the recovery name at the consumed boundary so
                            // it stays within the member span (J2).
                            MemberField::Named(Name {
                                span: SourceLoc::empty_at(self.prev_end()),
                            })
                        }
                    };
                    let span = self.span(base.span.start, self.prev_end());
                    base = Expr {
                        kind: ExprKind::Member {
                            base: Box::new(base),
                            field,
                        },
                        span,
                    };
                }
                TokenKind::LBracket => {
                    self.bump();
                    self.paren_depth += 1;
                    let index = self.parse_expr();
                    self.paren_depth -= 1;
                    if self.eat(TokenKind::RBracket).is_none() {
                        self.diag("P0002", self.cur_span(), "unclosed `[` in index");
                    }
                    // Cover the index child (a recovery placeholder may sit just
                    // outside the consumed range, J2).
                    let span = SourceLoc::new(base.span.start, self.prev_end().max(index.span.end));
                    base = Expr {
                        kind: ExprKind::Index {
                            base: Box::new(base),
                            index: Box::new(index),
                        },
                        span,
                    };
                }
                TokenKind::LParen => {
                    // ⟦P0108⟧ call expression.
                    let args = self.parse_call_args();
                    // Cover the args (a recovery arg may sit just outside the
                    // consumed range, J2).
                    let mut end = self.prev_end();
                    for a in &args {
                        end = end.max(a.span.end);
                    }
                    let span = self.span(base.span.start, end);
                    // ADR-0031: no call expressions in Core, uniformly — the
                    // ADR-0029 iota/fill carve is superseded; they are pipeline
                    // stages (`n -> iota`, `(x, n) -> fill`).
                    let is_array_builtin = matches!(
                        &base.kind,
                        ExprKind::Var(n) if matches!(self.text(n.span), "iota" | "fill")
                    );
                    self.diag(
                        "P0108",
                        span,
                        if is_array_builtin {
                            "call expressions are out of Mapal-Core; `iota`/`fill` are \
                             pipeline stages — write `n -> iota` / `(x, n) -> fill` (ADR-0031)"
                        } else {
                            "call expressions are out of Mapal-Core (HANDOFF §4); \
                             use a tuple-input flow: `(args) -> f`"
                        },
                    );
                    base = Expr {
                        kind: ExprKind::Call {
                            callee: Box::new(base),
                            args,
                        },
                        span,
                    };
                }
                TokenKind::Question => {
                    // ⟦P0101⟧ `?` postfix (W23: bound to the stage target).
                    let q = self.bump().span;
                    self.diag(
                        "P0101",
                        q,
                        "the `?` operator is out of Mapal-Core (HANDOFF §4); planned for Core+1 \
                         error handling",
                    );
                    let span = self.span(base.span.start, q.end);
                    base = Expr {
                        kind: ExprKind::Question(Box::new(base)),
                        span,
                    };
                }
                TokenKind::Colon if self.peek_kind(1) == TokenKind::Colon => {
                    // ⟦P0109⟧ `::` path. Only when the two colons are
                    // span-adjacent (DESIGN §14.6).
                    let c0 = self.cur().span;
                    let c1 = self.peek(1).span;
                    if c0.end != c1.start {
                        break;
                    }
                    self.bump(); // first `:`
                    self.bump(); // second `:`
                    self.diag(
                        "P0109",
                        self.span(c0.start, c1.end),
                        "`::` paths are out of Mapal-Core (HANDOFF §4); planned for Core+1 modules",
                    );
                    // Consume the next token as the path segment if it is an
                    // Ident or any keyword (e.g. `List::map` ends at KwMap).
                    if self.at(TokenKind::Ident) || is_keyword(self.kind()) {
                        self.bump();
                    }
                    // Continue the postfix loop on the base expression; the span
                    // grows to the consumed segment.
                    let span = self.span(base.span.start, self.prev_end());
                    base = Expr {
                        kind: base.kind,
                        span,
                    };
                }
                _ => break,
            }
        }
        for _ in 0..wraps {
            self.leave();
        }
        base
    }

    /// `'(' args? ')'` — caller has checked the `(`. Returns the argument exprs.
    pub(super) fn parse_call_args(&mut self) -> Vec<Expr> {
        self.bump(); // `(`
        let mut args = Vec::new();
        self.paren_depth += 1;
        while !self.at(TokenKind::RParen) && !self.at_eof() {
            let before = self.pos;
            args.push(self.parse_expr());
            if self.at(TokenKind::Comma) {
                self.bump();
            } else {
                break;
            }
            if self.pos == before {
                break;
            }
        }
        self.paren_depth -= 1;
        if self.eat(TokenKind::RParen).is_none() {
            self.diag("P0002", self.cur_span(), "unclosed `(` in call");
        }
        args
    }

    pub(super) fn parse_primary(&mut self) -> Expr {
        let start_span = self.cur_span();
        match self.kind() {
            TokenKind::Int => {
                let span = self.bump().span;
                Expr {
                    kind: ExprKind::Int(self.parse_int(span)),
                    span,
                }
            }
            TokenKind::Float => {
                let span = self.bump().span;
                Expr {
                    kind: ExprKind::Float,
                    span,
                }
            }
            TokenKind::Str => {
                let span = self.bump().span;
                Expr {
                    kind: ExprKind::Str,
                    span,
                }
            }
            TokenKind::KwTrue => {
                let span = self.bump().span;
                Expr {
                    kind: ExprKind::Bool(true),
                    span,
                }
            }
            TokenKind::KwFalse => {
                let span = self.bump().span;
                Expr {
                    kind: ExprKind::Bool(false),
                    span,
                }
            }
            TokenKind::Ident => {
                // `IDENT '{' field-inits '}'` struct literal, else a Var.
                if self.peek_kind(1) == TokenKind::LBrace {
                    self.parse_struct_literal()
                } else {
                    let span = self.bump().span;
                    Expr {
                        kind: ExprKind::Var(Name { span }),
                        span,
                    }
                }
            }
            TokenKind::LParen => self.parse_paren_or_tuple(),
            TokenKind::LBracket => self.parse_array_literal(),
            TokenKind::BackArrow => {
                // ⟦P0009⟧ `<-` inside an expression (W2 hint).
                let span = self.bump().span;
                self.diag_fix(
                    "P0009",
                    span,
                    "`<-` cannot appear inside an expression; `i<-1` lexes as `i <- 1`. \
                     For a comparison write `i < -1`",
                    SuggestedFix {
                        span,
                        replacement: "< -".to_string(),
                        label: "did you mean `< -` (comparison with a negative)?",
                    },
                );
                self.error_expr()
            }
            TokenKind::Error => {
                // Lexer-diagnosed token: skip silently, ends the atom (C14).
                self.bump();
                self.error_expr()
            }
            _ => {
                // `ret`/`loop` as a chain head, or any other non-expression token.
                self.diag("P0001", start_span, "expected an expression");
                self.error_expr()
            }
        }
    }

    /// `IDENT '{' field-inits? '}'` — caller verified `Ident {`.
    pub(super) fn parse_struct_literal(&mut self) -> Expr {
        let start = self.cur_span().start;
        let name = Name {
            span: self.bump().span,
        };
        if !self.enter() {
            self.skip_balanced(TokenKind::LBrace, TokenKind::RBrace);
            self.leave();
            return Expr {
                kind: ExprKind::Struct {
                    name,
                    fields: Vec::new(),
                },
                span: self.node_span(start),
            };
        }
        self.bump(); // `{`
        let mut fields = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let before = self.pos;
            let f = self.parse_field_init();
            fields.push(f);
            if self.at(TokenKind::Comma) {
                self.bump();
            } else {
                break;
            }
            if self.pos == before {
                break;
            }
        }
        if self.eat(TokenKind::RBrace).is_none() {
            self.diag(
                "P0002",
                self.span(start, start + 1),
                "unclosed `{` in struct literal",
            );
        }
        self.leave();
        // Cover the field inits (a recovery value may sit outside the consumed
        // range, J2).
        let mut span = self.node_span(start);
        for f in &fields {
            span = SourceLoc::new(span.start.min(f.span.start), span.end.max(f.span.end));
        }
        Expr {
            kind: ExprKind::Struct { name, fields },
            span,
        }
    }

    /// `field-init := IDENT (':' expr)?` — pun shorthand allowed.
    pub(super) fn parse_field_init(&mut self) -> FieldInit {
        let start = self.cur_span().start;
        let name = self.expect_name("field name");
        let value = if self.at(TokenKind::Colon) {
            self.bump();
            self.paren_depth += 1;
            let v = self.parse_expr();
            self.paren_depth -= 1;
            Some(v)
        } else {
            None
        };
        let span = match &value {
            Some(v) => self.node_span_covering(start, v.span),
            None => self.node_span(start),
        };
        FieldInit { name, value, span }
    }

    /// `'(' expr ')'` (grouping) or `'(' expr (',' expr)+ ','? ')'` (tuple ≥2).
    pub(super) fn parse_paren_or_tuple(&mut self) -> Expr {
        let start = self.cur_span().start;
        self.bump(); // `(`
        // `()` is the Unit literal (plan-time-builtin): the wire-less chain
        // head, whose one sanctioned use is `() -> time`. It carries no value,
        // so every other position rejects it in lower (L1301), where the chain
        // context is known — the parser only records the shape.
        if self.at(TokenKind::RParen) {
            self.bump();
            return Expr {
                kind: ExprKind::Unit,
                span: self.span(start, self.prev_end()),
            };
        }
        // Inside parens: a `<-` is the W2/P0009 expression defect (gated below).
        self.paren_depth += 1;
        let result = self.parse_paren_or_tuple_inner(start);
        self.paren_depth -= 1;
        result
    }

    pub(super) fn parse_paren_or_tuple_inner(&mut self, start: u32) -> Expr {
        let first = self.parse_expr();
        if self.at(TokenKind::Comma) {
            // Tuple.
            let mut elems = vec![first];
            let mut trailing_only = false;
            while self.at(TokenKind::Comma) {
                self.bump();
                if self.at(TokenKind::RParen) {
                    trailing_only = true;
                    break;
                }
                let before = self.pos;
                elems.push(self.parse_expr());
                if self.pos == before {
                    break;
                }
            }
            if self.eat(TokenKind::RParen).is_none() {
                self.diag("P0002", self.cur_span(), "unclosed `(` in tuple");
            }
            let span = self.expr_list_span(start, &elems);
            // `(e,)` is not a 1-tuple → P0001.
            if elems.len() < 2 && trailing_only {
                self.diag(
                    "P0001",
                    span,
                    "`(e,)` is not a one-tuple; tuples need at least two elements",
                );
            }
            Expr {
                kind: ExprKind::Tuple(elems),
                span,
            }
        } else {
            // Grouping — no Paren node; nesting carries it (DESIGN §14.6).
            if self.eat(TokenKind::RParen).is_none() {
                self.diag("P0002", self.cur_span(), "unclosed `(`");
            }
            // The grouped expression's span widens to include the parentheses so
            // J2 nesting (child ⊆ parent) holds for downstream usage.
            let span = self.node_span_covering(start, first.span);
            Expr {
                kind: first.kind,
                span,
            }
        }
    }

    /// The span of an expression composite (tuple/array): `[start, prev_end]`
    /// widened to cover all element spans (a recovery element may land outside
    /// the consumed range, J2).
    pub(super) fn expr_list_span(&self, start: u32, elems: &[Expr]) -> SourceLoc {
        let mut span = self.node_span(start);
        for e in elems {
            span = SourceLoc::new(span.start.min(e.span.start), span.end.max(e.span.end));
        }
        span
    }

    /// `'[' array-elem (',' array-elem)* ','? ']'`. A `...expr` element → P0107.
    pub(super) fn parse_array_literal(&mut self) -> Expr {
        let start = self.cur_span().start;
        self.bump(); // `[`
        // `[]` → P0001 (no element type, not exhibited).
        if self.at(TokenKind::RBracket) {
            self.bump();
            self.diag(
                "P0001",
                self.span(start, self.prev_end()),
                "empty array literal `[]` has no element type and is not Mapal-Core surface",
            );
            return Expr {
                kind: ExprKind::Array(Vec::new()),
                span: self.span(start, self.prev_end()),
            };
        }
        let mut elems = Vec::new();
        self.paren_depth += 1;
        while !self.at(TokenKind::RBracket) && !self.at_eof() {
            let before = self.pos;
            if self.at(TokenKind::DotDotDot) {
                // ⟦P0107⟧ rest element.
                let dots = self.bump().span;
                let e = self.parse_expr();
                self.diag(
                    "P0107",
                    self.span(dots.start, e.span.end),
                    "rest elements (`...tail`) in array literals are out of Mapal-Core \
                     (HANDOFF §4); planned for Core+1 slices",
                );
                elems.push(e);
            } else {
                elems.push(self.parse_expr());
            }
            if self.at(TokenKind::Comma) {
                self.bump();
            } else {
                break;
            }
            if self.pos == before {
                break;
            }
        }
        self.paren_depth -= 1;
        if self.eat(TokenKind::RBracket).is_none() {
            self.diag(
                "P0002",
                self.span(start, start + 1),
                "unclosed `[` in array literal",
            );
        }
        let span = self.expr_list_span(start, &elems);
        Expr {
            kind: ExprKind::Array(elems),
            span,
        }
    }

    pub(super) fn error_expr(&self) -> Expr {
        Expr {
            kind: ExprKind::Error,
            span: self.here(),
        }
    }
}
