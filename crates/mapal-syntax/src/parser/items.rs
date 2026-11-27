//! Parser methods for programs and items (DESIGN section 14.1).

use super::*;

impl<'a> Parser<'a> {
    // ======================================================================
    // §14.1 — program / items
    // ======================================================================

    pub(super) fn parse_program(&mut self) -> Program {
        let start = self.cur_span().start;
        let mut items = Vec::new();
        while !self.at_eof() {
            let before = self.pos;
            if let Some(item) = self.parse_item() {
                items.push(item);
            }
            // Progress lemma (DESIGN §16.1): every item* iteration consumes ≥1
            // token or exits. If nothing was consumed, skip exactly one token
            // under cooldown to guarantee termination (J1).
            if self.pos == before && !self.at_eof() {
                let span = self.cur_span();
                self.diag("P0001", span, "unexpected token at top level");
                self.bump();
            }
            debug_assert!(
                self.pos > before || self.at_eof(),
                "item* failed to make progress at {before}"
            );
        }
        // Cover the items: a trailing recovery node (e.g. a degenerate fn whose
        // body reached EOF) may extend to `source.len()`, past the last consumed
        // token's end (J2).
        let mut span = self.node_span(start);
        for item in &items {
            let s = match item {
                Item::Fn(f) => f.span,
                Item::Type(t) => t.span,
                Item::Bind(b) => b.span,
                Item::Error(e) => *e,
            };
            span = SourceLoc::new(span.start.min(s.start), span.end.max(s.end));
        }
        Program { items, span }
    }

    /// `item := fn-decl | type-decl | @-annotation item | executor … | pub/use …
    ///        | statement` (DESIGN §14.1).
    pub(super) fn parse_item(&mut self) -> Option<Item> {
        match self.kind() {
            TokenKind::KwFn => Some(Item::Fn(self.parse_fn_decl())),
            TokenKind::KwType | TokenKind::KwCategory => {
                // KwCategory: parse as `type`, NO new diagnostic (L0004 emitted).
                Some(Item::Type(self.parse_type_decl()))
            }
            TokenKind::At => self.parse_annotated_item(),
            TokenKind::KwExecutor => Some(self.parse_executor_decl()),
            TokenKind::KwPub | TokenKind::KwUse => Some(self.parse_pub_use()),
            TokenKind::Error => {
                // Lexer-diagnosed token: skip silently (C14).
                self.bump();
                None
            }
            _ => {
                // ADR-0048 D1: a chain is a top-level binding; any other
                // statement is ⟦P0012⟧ (parsed for spans, stored as Error).
                Some(self.parse_top_level_statement())
            }
        }
    }

    /// ⟦P0102⟧ `'@' IDENT ( '(' …balanced… ')' )? item` — the annotation tokens
    /// are skipped, the annotated item is kept (DESIGN §14.1).
    pub(super) fn parse_annotated_item(&mut self) -> Option<Item> {
        let at_span = self.bump().span; // `@`
        self.diag(
            "P0102",
            at_span,
            "annotations (`@…`) are out of Mapal-Core (HANDOFF §4); planned for Core+1",
        );
        // Skip the annotation name and an optional balanced (...) group. The
        // name may be an identifier OR a keyword (`@executor(…)`), so accept both.
        if self.at(TokenKind::Ident) || is_keyword(self.kind()) {
            self.bump();
        }
        if self.at(TokenKind::LParen) {
            self.skip_balanced(TokenKind::LParen, TokenKind::RParen);
        }
        // The annotated item follows; if it is another annotation, recurse.
        if self.at_eof() {
            return None;
        }
        self.parse_item()
    }

    /// ⟦P0111⟧ `'executor' IDENT '{' …balanced… '}'` — skipped (DESIGN §14.1).
    pub(super) fn parse_executor_decl(&mut self) -> Item {
        let start = self.cur_span().start;
        let kw_span = self.bump().span; // `executor`
        self.diag(
            "P0111",
            kw_span,
            "`executor` declarations are out of Mapal-Core (HANDOFF §4); planned for post-M5",
        );
        if self.at(TokenKind::Ident) {
            self.bump();
        }
        if self.at(TokenKind::LBrace) {
            self.skip_balanced(TokenKind::LBrace, TokenKind::RBrace);
        }
        let end = self.prev_end();
        Item::Error(self.span(start, end))
    }

    /// ⟦P0112⟧ `('pub' | 'use') …to ';' or item start…` — skipped (DESIGN §14.1).
    pub(super) fn parse_pub_use(&mut self) -> Item {
        let start = self.cur_span().start;
        let kw_span = self.bump().span; // `pub` / `use`
        let what = self.text(kw_span);
        self.diag(
            "P0112",
            kw_span,
            format!("`{what}` is out of Mapal-Core (HANDOFF §4); planned for post-M5"),
        );
        // Skip to the next `;` (consume) or item-start keyword / EOF (leave).
        loop {
            match self.kind() {
                TokenKind::Semi => {
                    self.bump();
                    break;
                }
                TokenKind::Eof
                | TokenKind::KwFn
                | TokenKind::KwType
                | TokenKind::KwCategory
                | TokenKind::KwExecutor
                | TokenKind::At => break,
                _ => {
                    self.bump();
                }
            }
        }
        Item::Error(self.node_span(start))
    }

    /// `fn-decl := 'fn' IDENT sizes? '(' params? ')' ( '->' type )? block`.
    pub(super) fn parse_fn_decl(&mut self) -> FnDecl {
        let start = self.cur_span().start;
        self.bump(); // `fn`
        let name = self.expect_name("function name");
        // ADR-0047 D1 / ADR-0051 D1: the generic list `<rows, cache: float>`.
        let generics = if self.at(TokenKind::Lt) {
            self.parse_generic_params()
        } else {
            Vec::new()
        };
        let params = if self.at(TokenKind::LParen) {
            self.parse_params()
        } else {
            self.diag("P0001", self.cur_span(), "expected `(` after function name");
            Vec::new()
        };
        let ret_ty = if self.at(TokenKind::Arrow) {
            self.bump();
            Some(self.parse_type())
        } else {
            None
        };
        // ADR-0041 D2: the compute type follows the output (`-> T: tf32 | f32`).
        let compute = self.parse_compute();
        let mut body = if self.at(TokenKind::LBrace) {
            self.parse_block()
        } else {
            self.diag("P0001", self.cur_span(), "expected `{` for function body");
            self.empty_block()
        };
        body.compute = compute;
        let span = self.node_span_covering(start, body.span);
        FnDecl {
            name,
            generics,
            params,
            ret_ty,
            body,
            span,
        }
    }

    /// `generics := '<' generic (',' generic)* ','? '>'`, `generic := IDENT
    /// (':' IDENT)?` (ADR-0047 D1, ADR-0051 D1: a bare name is a size
    /// parameter, `name: kind` a format parameter) — caller has checked the
    /// `<`. An empty list or a missing kind after `:` is P0001; a missing `>`
    /// is P0002.
    pub(super) fn parse_generic_params(&mut self) -> Vec<GenericParam> {
        self.bump(); // `<`
        let mut sizes = Vec::new();
        while self.at(TokenKind::Ident) {
            let name = Name {
                span: self.bump().span,
            };
            let kind = if self.eat(TokenKind::Colon).is_some() {
                if self.at(TokenKind::Ident) {
                    Some(Name {
                        span: self.bump().span,
                    })
                } else {
                    self.diag(
                        "P0001",
                        self.cur_span(),
                        "expected a format kind after `:` — `float`, `narrow`, `int` or \
                         `number` (`fn f<cache: float>`, ADR-0051)",
                    );
                    None
                }
            } else {
                None
            };
            let span = self.node_span(name.span.start);
            sizes.push(GenericParam { name, kind, span });
            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
        }
        if sizes.is_empty() {
            self.diag(
                "P0001",
                self.cur_span(),
                "expected a size parameter name: `fn f<n>(x: [f32; n])` (ADR-0047), or a format \
                 parameter with its kind: `fn f<cache: float>` (ADR-0051)",
            );
        }
        if self.eat(TokenKind::Gt).is_none() {
            self.diag(
                "P0002",
                self.cur_span(),
                "unclosed `<` in the size parameter list",
            );
            // Recover at the parameter list or the body.
            while !matches!(
                self.kind(),
                TokenKind::LParen | TokenKind::LBrace | TokenKind::Eof | TokenKind::KwFn
            ) {
                self.bump();
            }
        }
        sizes
    }

    /// `compute := ':' member ('|' member)*` when the cursor is at `:` before
    /// a member (ADR-0041 D2) — on whatever opens a body; members `f32`,
    /// `tf32`, `bf16`, `f16` (ADR-0043 D4). The lone `|` the lexer flags
    /// (L0006) is this type's separator.
    pub(super) fn parse_compute(&mut self) -> Option<Precision> {
        let member = |t: &str| matches!(t, "f32" | "tf32" | "bf16" | "f16");
        if !(self.at(TokenKind::Colon)
            && self.peek_kind(1) == TokenKind::Ident
            && member(self.text(self.peek(1).span)))
        {
            return None;
        }
        let start = self.cur_span().start;
        self.bump(); // `:`
        let mut p = Precision {
            f32: false,
            tf32: false,
            bf16: false,
            f16: false,
            span: self.cur_span(),
        };
        loop {
            let sp = self.cur_span();
            match (self.cur().kind, self.text(sp)) {
                (TokenKind::Ident, "f32") => p.f32 = true,
                (TokenKind::Ident, "tf32") => p.tf32 = true,
                (TokenKind::Ident, "bf16") => p.bf16 = true,
                (TokenKind::Ident, "f16") => p.f16 = true,
                _ => {
                    self.diag(
                        "P0118",
                        sp,
                        "expected a precision: `f32`, `tf32`, `bf16` or `f16` (ADR-0041, ADR-0043)",
                    );
                    break;
                }
            }
            self.bump();
            let bar = self.cur_span();
            if self.cur().kind == TokenKind::Error && self.text(bar) == "|" {
                self.diagnostics
                    .retain(|d| !(d.code.0 == "L0006" && d.span == bar));
                self.bump();
                continue;
            }
            break;
        }
        p.span = self.span(start, self.prev_end());
        Some(p)
    }

    /// `params := param (',' param)* ','?` — caller has checked the `(`.
    pub(super) fn parse_params(&mut self) -> Vec<Param> {
        self.bump(); // `(`
        let mut params = Vec::new();
        while !self.at(TokenKind::RParen) && !self.at_eof() {
            let before = self.pos;
            params.push(self.parse_param());
            if self.at(TokenKind::Comma) {
                self.bump();
            } else if !self.at(TokenKind::RParen) {
                // Not a comma and not the close: stop the list.
                if self.pos == before {
                    // Avoid an infinite loop on a token a param can't start.
                    break;
                }
                break;
            }
            if self.pos == before {
                break;
            }
        }
        if self.eat(TokenKind::RParen).is_none() {
            self.diag("P0001", self.cur_span(), "expected `)` to close parameters");
        }
        params
    }

    pub(super) fn parse_param(&mut self) -> Param {
        let start = self.cur_span().start;
        let mut_span = self.eat(TokenKind::KwMut);
        let name = self.expect_name("parameter name");
        let ty = if self.at(TokenKind::Colon) {
            self.bump();
            self.parse_type()
        } else {
            self.diag(
                "P0001",
                self.cur_span(),
                "expected `:` and a parameter type",
            );
            self.error_ty()
        };
        let span = self.node_span_covering(start, ty.span);
        Param {
            mut_span,
            name,
            ty,
            span,
        }
    }

    /// `type-decl := ('type'|category) IDENT '{' field-list? '}'`.
    pub(super) fn parse_type_decl(&mut self) -> TypeDecl {
        let start = self.cur_span().start;
        self.bump(); // `type` or `category` (L0004 already emitted; no new diag)
        let name = self.expect_name("type name");
        // ADR-0048 D4 / ADR-0051 D1: `type Matrix<rows, columns, e: float>`.
        let generics = if self.at(TokenKind::Lt) {
            self.parse_generic_params()
        } else {
            Vec::new()
        };
        if self.at(TokenKind::LBrace) {
            self.bump();
        } else {
            self.diag("P0001", self.cur_span(), "expected `{` for type body");
        }
        let mut fields = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let before = self.pos;
            if self.at(TokenKind::Minus) {
                // ⟦P0105⟧ enum/coproduct variant: skip, decl kept.
                self.parse_enum_variant();
            } else {
                fields.push(self.parse_field());
            }
            if self.at(TokenKind::Comma) {
                self.bump();
            }
            if self.pos == before {
                // Progress lemma: skip one token we can't classify.
                self.diag("P0001", self.cur_span(), "expected a field or `}`");
                self.bump();
            }
        }
        if self.eat(TokenKind::RBrace).is_none() {
            self.diag("P0002", self.cur_span(), "unclosed `{` in type body");
        }
        TypeDecl {
            name,
            generics,
            fields,
            span: self.node_span(start),
        }
    }

    pub(super) fn parse_field(&mut self) -> Field {
        let start = self.cur_span().start;
        let name = self.expect_name("field name");
        let ty = if self.at(TokenKind::Colon) {
            self.bump();
            self.parse_type()
        } else {
            self.diag("P0001", self.cur_span(), "expected `:` and a field type");
            self.error_ty()
        };
        let span = self.node_span_covering(start, ty.span);
        Field { name, ty, span }
    }

    /// ⟦P0105⟧ `'-' IDENT ( '{' …balanced… '}' | '(' …balanced… ')' )` — an enum
    /// variant in a type body; the variant is skipped, the decl kept.
    pub(super) fn parse_enum_variant(&mut self) {
        let minus = self.bump().span; // `-`
        let mut end = minus.end;
        if self.at(TokenKind::Ident) {
            end = self.bump().span.end;
        }
        if self.at(TokenKind::LBrace) {
            self.skip_balanced(TokenKind::LBrace, TokenKind::RBrace);
            end = self.prev_end();
        } else if self.at(TokenKind::LParen) {
            self.skip_balanced(TokenKind::LParen, TokenKind::RParen);
            end = self.prev_end();
        }
        self.diag(
            "P0105",
            self.span(minus.start, end),
            "enum/coproduct variants in a `type` body are out of Mapal-Core (HANDOFF §4); \
             planned for Core+1 coproducts",
        );
    }
}
