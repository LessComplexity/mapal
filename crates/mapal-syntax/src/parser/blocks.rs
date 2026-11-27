//! Parser methods for blocks and statements (DESIGN section 14.3).

use super::*;

impl<'a> Parser<'a> {
    // ======================================================================
    // §14.3 — blocks, statements, chains
    // ======================================================================

    pub(super) fn empty_block(&self) -> Block {
        Block {
            items: Vec::new(),
            tail: None,
            span: self.here(),
            compute: None,
        }
    }

    /// `block := '{' block-item* tail? '}'` — caller has checked the `{`.
    pub(super) fn parse_block(&mut self) -> Block {
        let start = self.cur_span().start;
        if !self.enter() {
            // Depth guard: synthesize an empty block and skip the brace group.
            self.skip_balanced(TokenKind::LBrace, TokenKind::RBrace);
            self.leave();
            return Block {
                items: Vec::new(),
                tail: None,
                span: self.node_span(start),
                compute: None,
            };
        }
        self.bump(); // `{`
        // A non-stage block (fn body, loop body, payload block): guard arms are
        // not legitimate here, so a leading guard is stray → P0004.
        let (items, tail) = self.parse_block_body(false);
        if self.eat(TokenKind::RBrace).is_none() {
            self.diag("P0002", self.span(start, start + 1), "unclosed `{`");
        }
        self.leave();
        let span = self.block_span(start, &items, &tail);
        Block {
            items,
            tail,
            span,
            compute: None,
        }
    }

    /// A block span: `[start, prev_end]` widened to cover its body (J2).
    pub(super) fn block_span(
        &self,
        start: u32,
        items: &[BlockItem],
        tail: &Option<Chain>,
    ) -> SourceLoc {
        let base = self.node_span(start);
        match Self::body_extent(items, tail) {
            None => base,
            Some(ext) => self.span(base.start.min(ext.start), base.end.max(ext.end)),
        }
    }

    /// Parse a block's items and optional tail (the content between the braces).
    /// Mixing guard arms with non-arm statements draws P0006 (§14.3).
    ///
    /// `guard_ok` is true only for a stage-position block (`-> { … }`), where
    /// guard arms are legitimate (§14.4). Elsewhere a leading guard token is a
    /// **stray** guard → P0004 (W1), recovered as a statement.
    pub(super) fn parse_block_body(&mut self, guard_ok: bool) -> (Vec<BlockItem>, Option<Chain>) {
        let mut items: Vec<BlockItem> = Vec::new();
        let mut tail: Option<Chain> = None;
        while !self.at(TokenKind::RBrace) && !self.at_eof() {
            let before = self.pos;
            // ⟦P0001⟧ empty statement.
            if self.at(TokenKind::Semi) {
                let span = self.bump().span;
                self.diag("P0001", span, "empty statement (`;`) — removed");
                continue;
            }
            let class = self.classify_block_item();
            // A clean `Guard` token in a non-guard block is a *stray* guard (W1)
            // → P0004, recovered as a statement so the chain still parses.
            if !guard_ok && matches!(self.kind(), TokenKind::Guard(_)) {
                match self.parse_stray_guard_statement() {
                    StmtOrTail::Stmt(stmt) => items.push(BlockItem::Stmt(stmt)),
                    StmtOrTail::Tail(chain) => {
                        tail = Some(chain);
                        break;
                    }
                }
                if self.pos == before {
                    self.diag("P0001", self.cur_span(), "unexpected token in block");
                    self.bump();
                }
                continue;
            }
            match class {
                BlockItemClass::Arm => {
                    let arm = self.parse_guard_arm();
                    items.push(BlockItem::Arm(arm));
                }
                BlockItemClass::Statement => {
                    // A bare-expression chain ending at `}` is the block tail.
                    match self.parse_statement_or_tail() {
                        StmtOrTail::Stmt(stmt) => items.push(BlockItem::Stmt(stmt)),
                        StmtOrTail::Tail(chain) => {
                            tail = Some(chain);
                            // Tail must be the last thing before `}`.
                            break;
                        }
                    }
                }
            }
            if self.pos == before {
                // Progress lemma: skip one token under cooldown.
                let span = self.cur_span();
                self.diag("P0001", span, "unexpected token in block");
                self.bump();
            }
            debug_assert!(
                self.pos > before || self.at(TokenKind::RBrace) || self.at_eof(),
                "block-item* failed to make progress"
            );
        }
        // §14.3 mixing: if both arms and non-arm statements are present, P0006.
        self.check_arm_mixing(&items);
        (items, tail)
    }

    /// ⟦P0004⟧ stray guard arrow outside a guard block (W1). The lexer produced a
    /// clean `Guard` token because the context gate passed, but no guard block
    /// surrounds it. Emit P0004 with a fix built by **slicing the source lexeme**
    /// — `&source[span]` minus the trailing `->`, plus `" ->"` — never re-rendered
    /// from the clamped `GuardKind::Int` (which would rewrite an over-`u64`
    /// literal). Recover: an Int discriminant becomes a chain head
    /// `Unary(Neg, Int)` followed by an implicit arrow stage; other discriminants
    /// become an `Expr::Error` head.
    pub(super) fn parse_stray_guard_statement(&mut self) -> StmtOrTail {
        let start = self.cur_span().start;
        let TokenKind::Guard(gk) = self.kind() else {
            unreachable!("caller checked for a Guard token");
        };
        let guard_span = self.bump().span; // the `-D->` lexeme
        // Build the fix by slicing the source: drop the trailing `->`, add ` ->`.
        let lexeme = self.text(guard_span);
        let trimmed = lexeme.strip_suffix("->").unwrap_or(lexeme);
        let replacement = format!("{trimmed} ->");
        self.diag_fix(
            "P0004",
            guard_span,
            "stray guard arrow outside a guard block; `-7-> x` reads as a guard arm. \
             If you meant a negated value flowing to a target, add a space: `-7 -> x`",
            SuggestedFix {
                span: guard_span,
                replacement,
                label: "add a space inside the would-be guard arrow",
            },
        );
        // Recover the head expression.
        let head = match gk {
            GuardKind::Int(n) => {
                let operand = Expr {
                    kind: ExprKind::Int(n),
                    span: guard_span,
                };
                Expr {
                    kind: ExprKind::Unary {
                        op: UnOp::Neg,
                        op_span: guard_span,
                        operand: Box::new(operand),
                    },
                    span: guard_span,
                }
            }
            _ => Expr {
                kind: ExprKind::Error,
                span: guard_span,
            },
        };
        // The implicit arrow stage uses the guard lexeme's span as the arrow span;
        // the stage span starts at the guard lexeme (the implicit `->`) and is
        // widened to cover an inner expression placeholder (J2).
        let mut stages = Vec::new();
        let stage_kind = self.parse_stage_body();
        let mut stage_span = self.node_span(guard_span.start);
        if let Some(ext) = stage_kind_extent(&stage_kind) {
            stage_span =
                SourceLoc::new(stage_span.start.min(ext.start), stage_span.end.max(ext.end));
        }
        stages.push(Stage {
            arrow_span: guard_span,
            kind: stage_kind,
            span: stage_span,
        });
        // Continue any further `-> …` stages.
        while self.at(TokenKind::Arrow) {
            let before = self.pos;
            stages.push(self.parse_stage());
            if self.pos == before {
                break;
            }
        }
        // Cover the head and last stage (J2).
        let mut chain_span = self.node_span(start);
        chain_span = SourceLoc::new(
            chain_span.start.min(head.span.start),
            chain_span.end.max(head.span.end),
        );
        if let Some(s) = stages.last() {
            chain_span = SourceLoc::new(
                chain_span.start.min(s.span.start),
                chain_span.end.max(s.span.end),
            );
        }
        let chain = Chain {
            head: Some(head),
            stages,
            span: chain_span,
        };
        self.finish_chain(start, chain)
    }

    /// Detect arm/non-arm mixing and emit P0006 with the bidirectional hint for
    /// the `Minus Int Arrow` statement case (DESIGN §14.3 / §16).
    pub(super) fn check_arm_mixing(&mut self, items: &[BlockItem]) {
        let has_arm = items.iter().any(|i| matches!(i, BlockItem::Arm(_)));
        if !has_arm {
            return;
        }
        for item in items {
            if let BlockItem::Stmt(stmt) = item {
                // A statement among arms — P0006 (bidirectional hint when it is a
                // negated-int flow, which the user may have meant as a guard arm).
                // Cooldown is irrelevant here (we report at a settled position);
                // push directly so the mixing is always surfaced.
                let msg = "statement mixed with guard arms in the same block; \
                           a guard block contains only arms. \
                           If you meant a guard arm, remove the space: `-7->`";
                self.diagnostics
                    .push(Diagnostic::error("P0006", stmt.span, msg));
            }
        }
    }

    /// Classify the block-item at the cursor: guard arm vs statement
    /// (DESIGN §14.3 leading-`Minus` disambiguation rules 1–3).
    pub(super) fn classify_block_item(&self) -> BlockItemClass {
        // A clean Guard token is always an arm.
        if matches!(self.kind(), TokenKind::Guard(_)) {
            return BlockItemClass::Arm;
        }
        if self.at(TokenKind::Minus) {
            // Rule 1: `Minus (true|false|Ident‹_›) Arrow` (any spacing) → P0005 arm.
            if self.is_spaced_bool_or_default_arm() {
                return BlockItemClass::Arm;
            }
            // Rule 2: `Minus` span-adjacent to Ident≠_ or `[`, balanced group,
            // span-adjacent `Arrow` → P0106 pattern arm.
            if self.is_pattern_arm() {
                return BlockItemClass::Arm;
            }
            // Rule 3: otherwise an expression statement.
        }
        BlockItemClass::Statement
    }

    /// Rule 1 lookahead: `Minus (KwTrue|KwFalse|Ident‹_›) Arrow`, any spacing.
    pub(super) fn is_spaced_bool_or_default_arm(&self) -> bool {
        debug_assert!(self.at(TokenKind::Minus));
        let k1 = self.peek_kind(1);
        let is_disc = match k1 {
            TokenKind::KwTrue | TokenKind::KwFalse => true,
            TokenKind::Ident => self.text(self.peek(1).span) == "_",
            _ => false,
        };
        is_disc && self.peek_kind(2) == TokenKind::Arrow
    }

    /// Rule 2 lookahead: `Minus` span-adjacent to `Ident‹≠_›` or `[`, followed by
    /// a balanced `(…)`/`[…]` group (possibly empty), followed by a span-adjacent
    /// `Arrow` → pattern arm (`-Some(x)->`, `-None->`, `-[]->`, `-[h, ...t]->`).
    pub(super) fn is_pattern_arm(&self) -> bool {
        debug_assert!(self.at(TokenKind::Minus));
        let minus = self.cur().span;
        let head = self.peek(1);
        // Span-adjacency of `-` and the head.
        if minus.end != head.span.start {
            return false;
        }
        let is_head = match head.kind {
            TokenKind::Ident => self.text(head.span) != "_",
            TokenKind::LBracket => true,
            _ => false,
        };
        if !is_head {
            return false;
        }
        // Walk a balanced group of (...) and [...] from after the head, then
        // require a span-adjacent `->`. Bounded scan over the token slice.
        let mut i = self.pos + 1; // index of `head`
        let mut after = i + 1; // first token after the head
        // The head may be `[` (start of a group) or an Ident (then a group may
        // follow). Group brackets: `(` `)` and `[` `]`.
        if head.kind == TokenKind::LBracket {
            // The head itself opens a `[...]` group.
            after = self.scan_balanced_group(i);
            if after == usize::MAX {
                return false;
            }
        } else {
            // Ident head: an optional adjacent `(...)` or `[...]` group.
            i = after;
            if i < self.tokens.len()
                && matches!(self.tokens[i].kind, TokenKind::LParen | TokenKind::LBracket)
            {
                after = self.scan_balanced_group(i);
                if after == usize::MAX {
                    return false;
                }
            } else {
                after = i;
            }
        }
        // Now require an `Arrow` at `after`, span-adjacent to the prior token.
        if after >= self.tokens.len() {
            return false;
        }
        let arrow = self.tokens[after];
        if arrow.kind != TokenKind::Arrow {
            return false;
        }
        let prev_end = self.tokens[after - 1].span.end;
        arrow.span.start == prev_end
    }

    /// Scan a balanced bracket group starting at token index `open` (which must
    /// be `(` or `[`). Returns the index of the first token *after* the matching
    /// close, or `usize::MAX` if unbalanced before EOF.
    pub(super) fn scan_balanced_group(&self, open: usize) -> usize {
        let open_kind = self.tokens[open].kind;
        let close_kind = match open_kind {
            TokenKind::LParen => TokenKind::RParen,
            TokenKind::LBracket => TokenKind::RBracket,
            TokenKind::LBrace => TokenKind::RBrace,
            _ => return usize::MAX,
        };
        let mut depth = 0i32;
        let mut i = open;
        while i < self.tokens.len() {
            let k = self.tokens[i].kind;
            if k == TokenKind::Eof {
                return usize::MAX;
            }
            if k == open_kind {
                depth += 1;
            } else if k == close_kind {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            i += 1;
        }
        usize::MAX
    }

    /// Parse a statement, or recognize that a bare-expression chain ending at `}`
    /// is the block's tail (DESIGN §14.3 termination rule).
    pub(super) fn parse_statement_or_tail(&mut self) -> StmtOrTail {
        let start = self.cur_span().start;
        match self.kind() {
            TokenKind::KwLoop => StmtOrTail::Stmt(self.parse_loop_stmt()),
            // ⟦P0110⟧ statement-initial labeled block `:label { … }` (ADR-0012).
            // Recognized on the `Colon` with one token of lookahead; a second
            // `Colon` (a `::` path) is NOT a label, so require `Colon Ident LBrace`.
            TokenKind::Colon
                if self.peek_kind(1) == TokenKind::Ident
                    && self.peek_kind(2) == TokenKind::LBrace =>
            {
                StmtOrTail::Stmt(self.parse_sigiled_labeled_block())
            }
            TokenKind::KwVoid => {
                // ⟦P0113⟧ statement-initial `void` (with or without block).
                StmtOrTail::Stmt(self.parse_void_stmt())
            }
            // Statement-initial keyword stages that must follow `->` (W22).
            TokenKind::KwSeq | TokenKind::KwMap | TokenKind::KwFold => {
                let span = self.cur_span();
                let kw = self.text(span);
                self.diag(
                    "P0001",
                    span,
                    format!("`{kw}` must follow `->` (it is a stage, not a statement head)"),
                );
                // Recover: skip the keyword and its block if present.
                self.bump();
                if self.at(TokenKind::LBrace) {
                    self.skip_balanced(TokenKind::LBrace, TokenKind::RBrace);
                }
                StmtOrTail::Stmt(Stmt {
                    kind: StmtKind::Error,
                    span: self.node_span(start),
                })
            }
            _ => self.parse_chain_or_bind_stmt(),
        }
    }

    /// Parse a `loop`-introduced or `Ident {`-labeled loop / a chain / a bind.
    pub(super) fn parse_chain_or_bind_stmt(&mut self) -> StmtOrTail {
        // §14.5: statement-initial `Ident {` four-token scan (ADR-0011).
        if self.at(TokenKind::Ident)
            && self.peek_kind(1) == TokenKind::LBrace
            && self.ident_brace_is_loop()
        {
            return StmtOrTail::Stmt(self.parse_labeled_loop());
        }
        // bind-stmt: `mut`? IDENT (':' type)? '<-' expr.
        if self.looks_like_bind() {
            return StmtOrTail::Stmt(self.parse_bind_stmt());
        }
        // Otherwise a chain (statement or tail).
        self.parse_chain_statement()
    }

    /// §14.5 scan: from the `{` to its matching `}` (depth count), any `Semi`,
    /// `Arrow`, `BackArrow`, or `Guard` (at any depth) ⇒ labeled loop (ADR-0011).
    pub(super) fn ident_brace_is_loop(&self) -> bool {
        // Find the `{` (it is at pos+1).
        let open = self.pos + 1;
        debug_assert_eq!(self.tokens[open].kind, TokenKind::LBrace);
        let mut depth = 0i32;
        let mut i = open;
        while i < self.tokens.len() {
            match self.tokens[i].kind {
                TokenKind::LBrace => depth += 1,
                TokenKind::RBrace => {
                    depth -= 1;
                    if depth == 0 {
                        return false;
                    }
                }
                TokenKind::Semi | TokenKind::Arrow | TokenKind::BackArrow | TokenKind::Guard(_) => {
                    return true;
                }
                TokenKind::Eof => return false,
                _ => {}
            }
            i += 1;
        }
        false
    }

    /// ⟦P0110⟧ statement-initial labeled block `:label { … }` (ADR-0012). The
    /// `Custom` label `Name` span covers **just the identifier** (so its text is a
    /// valid name); the `:` sigil is folded into the diagnostic span and the
    /// `LoopStmt` span (which begins at the `Colon`). Parsed as a loop for
    /// recovery; the body goes through `parse_block` (depth guard, J1/J2).
    pub(super) fn parse_sigiled_labeled_block(&mut self) -> Stmt {
        let start = self.cur_span().start; // the `:`
        let colon = self.bump().span; // `:`
        let name = Name {
            span: self.bump().span, // IDENT
        };
        // Diagnostic span covers `:label` (sigil + ident).
        let sigil_span = self.span(colon.start, name.span.end);
        self.diag(
            "P0110",
            sigil_span,
            "labeled blocks (`:label { … }`, ADR-0012) are out of Mapal-Core (HANDOFF §4); \
             Mapal-Core's only loop introducer is `loop`. Planned for Core+1",
        );
        let body = self.parse_block();
        let span = self.node_span_covering(start, body.span);
        Stmt {
            kind: StmtKind::Loop(LoopStmt {
                label: LoopLabel::Custom(name),
                body,
                span,
            }),
            span,
        }
    }

    /// ⟦P0110⟧ recovery heuristic (demoted ADR-0011 scan, §14.5): a
    /// statement-initial `IDENT block` whose braces are loop-shaped (contain
    /// `Semi`/`Arrow`/`BackArrow`/`Guard`). Under ADR-0012 un-sigiled `Ident {`
    /// is *always* a struct literal grammatically; this path survives only to
    /// improve the error — it parses the body as a labeled block and points at
    /// the sigiled spelling.
    pub(super) fn parse_labeled_loop(&mut self) -> Stmt {
        let start = self.cur_span().start;
        let name = Name {
            span: self.bump().span,
        };
        let ident = self.text(name.span).to_string();
        self.diag(
            "P0110",
            name.span,
            format!(
                "labeled blocks are written `:{ident} {{ … }}` (ADR-0012) and are out of \
                 Mapal-Core (HANDOFF §4); Mapal-Core's only loop introducer is `loop`. \
                 Planned for Core+1"
            ),
        );
        let body = self.parse_block();
        let span = self.node_span_covering(start, body.span);
        Stmt {
            kind: StmtKind::Loop(LoopStmt {
                label: LoopLabel::Custom(name),
                body,
                span,
            }),
            span,
        }
    }

    /// `loop-stmt := 'loop' block`.
    pub(super) fn parse_loop_stmt(&mut self) -> Stmt {
        let start = self.cur_span().start;
        let kw = self.bump().span; // `loop`
        let compute = self.parse_compute();
        let mut body = if self.at(TokenKind::LBrace) {
            self.parse_block()
        } else {
            self.diag("P0001", self.cur_span(), "expected `{` for loop body");
            self.empty_block()
        };
        body.compute = compute;
        let span = self.node_span_covering(start, body.span);
        Stmt {
            kind: StmtKind::Loop(LoopStmt {
                label: LoopLabel::Loop(kw),
                body,
                span,
            }),
            span,
        }
    }

    /// ⟦P0113⟧ statement-initial `void` — both `void { … }` (fanout) and a bare
    /// `void;` keyword. Bare ⇒ skip + `Stmt::Error`.
    pub(super) fn parse_void_stmt(&mut self) -> Stmt {
        let start = self.cur_span().start;
        let kw = self.bump().span; // `void`
        if self.at(TokenKind::LBrace) {
            // void { … } — parse as a fanout-shaped block stage for recovery.
            self.diag(
                "P0113",
                kw,
                "`void` is out of Mapal-Core (HANDOFF §4); planned for Core+1",
            );
            let _block = self.parse_block();
            // Consume the trailing `;` if present (statement form).
            self.eat(TokenKind::Semi);
            Stmt {
                kind: StmtKind::Error,
                span: self.node_span(start),
            }
        } else {
            // bare `void;` — skip the keyword (and a trailing `;`), Stmt::Error.
            self.diag(
                "P0113",
                kw,
                "`void` is out of Mapal-Core (HANDOFF §4); planned for Core+1",
            );
            self.eat(TokenKind::Semi);
            Stmt {
                kind: StmtKind::Error,
                span: self.node_span(start),
            }
        }
    }

    /// Heuristic: does the cursor start a `bind-stmt`? `mut`? IDENT (':' type)?
    /// '<-' — we look for a `<-` before any `->`/`;`/`{`.
    pub(super) fn looks_like_bind(&self) -> bool {
        let mut i = self.pos;
        if self.tokens[i].kind == TokenKind::KwMut {
            i += 1;
        }
        if self.tokens.get(i).map(|t| t.kind) != Some(TokenKind::Ident) {
            return false;
        }
        i += 1;
        // Optional `: type` — scan until `<-`, `->`, `;`, `{`, or EOF, but a
        // BackArrow before any of those means a bind.
        let mut depth = 0i32;
        while i < self.tokens.len() {
            match self.tokens[i].kind {
                TokenKind::BackArrow if depth == 0 => return true,
                TokenKind::Arrow | TokenKind::Semi if depth == 0 => return false,
                TokenKind::LBrace if depth == 0 => return false,
                TokenKind::LParen | TokenKind::LBracket | TokenKind::Lt => depth += 1,
                TokenKind::RParen | TokenKind::RBracket | TokenKind::Gt => {
                    depth = (depth - 1).max(0);
                }
                TokenKind::Eof | TokenKind::RBrace => return false,
                _ => {}
            }
            i += 1;
        }
        false
    }

    /// `bind-stmt := 'mut'? IDENT ('[' expr ']')? (':' type)? '<-' expr` — a
    /// second `<-` in one statement draws P0008 (DESIGN §14.3 / §16). The
    /// optional `[' expr ']` is the element-update sugar `c[i] <- x` (ADR-0021):
    /// it excludes `mut` (P0013) and a type annotation (P0014); a nested target
    /// `c[i][j]` draws P0015.
    pub(super) fn parse_bind_stmt(&mut self) -> Stmt {
        let start = self.cur_span().start;
        let mut_span = self.eat(TokenKind::KwMut);
        let name = self.expect_name("binding name");
        // Optional element-update index `[' expr ']` (ADR-0021 §2).
        let index = if self.at(TokenKind::LBracket) {
            self.bump(); // `[`
            self.paren_depth += 1;
            let idx = self.parse_expr();
            self.paren_depth -= 1;
            if self.eat(TokenKind::RBracket).is_none() {
                self.diag("P0002", self.cur_span(), "unclosed `[` in update target");
            }
            // A nested index target `c[i][j] <- x` is out of this increment
            // (ADR-0021 §2, one-dimensional update only): P0015, extra
            // `[' … ']` groups consumed and dropped so the recovered node is a
            // clean one-dimensional indexed bind.
            while self.at(TokenKind::LBracket) {
                let open = self.cur_span();
                self.bump(); // `[`
                self.paren_depth += 1;
                let _ = self.parse_expr();
                self.paren_depth -= 1;
                let close = self
                    .eat(TokenKind::RBracket)
                    .unwrap_or_else(|| self.cur_span());
                self.diag(
                    "P0015",
                    self.span(open.start, close.end),
                    "nested element-update target `c[i][j] <- x` is out of Mapal-Core \
                     (ADR-0021: one-dimensional update only); planned for a later increment",
                );
            }
            // `mut c[i] <- x` — the index form is a rebind, not a fresh
            // declaration, so `mut` is meaningless (ADR-0021 §2): P0013.
            if let Some(m) = mut_span {
                self.diag(
                    "P0013",
                    m,
                    "`mut` is not allowed on an element-update target (`c[i] <- x` is a \
                     rebind, not a new binding)",
                );
            }
            Some(idx)
        } else {
            None
        };
        let ty = if self.at(TokenKind::Colon) {
            self.bump();
            Some(self.parse_type())
        } else {
            None
        };
        // `c[i]: T <- x` — an element-update target takes no type annotation
        // (ADR-0021 §2; the slot type is already fixed by the array): P0014.
        if let (Some(_), Some(t)) = (&index, &ty) {
            self.diag(
                "P0014",
                t.span,
                "an element-update target (`c[i] <- x`) takes no type annotation",
            );
        }
        // Consume `<-`.
        if self.eat(TokenKind::BackArrow).is_none() {
            self.diag("P0001", self.cur_span(), "expected `<-` in binding");
        }
        let value = self.parse_expr();
        // A second `<-` in the same statement → P0008 (W17), consume to `;`.
        if self.at(TokenKind::BackArrow) {
            let span = self.cur_span();
            self.diag(
                "P0008",
                span,
                "only one `<-` binding per statement (W17); chained `<-` is not Mapal-Core",
            );
            while !self.at(TokenKind::Semi) && !self.at(TokenKind::RBrace) && !self.at_eof() {
                self.bump();
            }
        }
        // Statement terminator `;`.
        self.expect_semi("binding statement");
        // Cover the bound value (and its type/index) so recovery placeholders nest (J2).
        let mut span = self.node_span_covering(start, value.span);
        if let Some(t) = &ty {
            span = SourceLoc::new(span.start.min(t.span.start), span.end.max(t.span.end));
        }
        if let Some(i) = &index {
            span = SourceLoc::new(span.start.min(i.span.start), span.end.max(i.span.end));
        }
        Stmt {
            kind: StmtKind::Bind(BindStmt {
                mut_span,
                name,
                index,
                ty,
                value,
                span,
            }),
            span,
        }
    }
}
