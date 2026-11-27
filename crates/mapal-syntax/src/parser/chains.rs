//! Parser methods for flow chains, stages and block stages.

use super::*;

impl<'a> Parser<'a> {
    // ======================================================================
    // chains
    // ======================================================================

    /// Parse a chain at statement position, then apply the §14.3 termination
    /// rule: `;` ⇒ statement; `}` ⇒ tail; block-final stage ⇒ statement w/o `;`;
    /// else P0001.
    pub(super) fn parse_chain_statement(&mut self) -> StmtOrTail {
        let start = self.cur_span().start;
        let chain = self.parse_chain();
        self.finish_chain(start, chain)
    }

    /// Apply the termination rule to a freshly-parsed chain.
    pub(super) fn finish_chain(&mut self, start: u32, chain: Chain) -> StmtOrTail {
        let bare_expr = chain.head.is_some() && chain.stages.is_empty();
        let last_is_block = chain
            .stages
            .last()
            .map(|s| stage_is_block_form(&s.kind))
            .unwrap_or(false);

        if self.at(TokenKind::Semi) {
            self.bump();
            if bare_expr {
                // A bare-expression chain as a `;`-terminated statement → P0003.
                self.diag(
                    "P0003",
                    chain.span,
                    "a bare expression is not a statement; flow it (`-> target`) or make it \
                     the block tail",
                );
            }
            let span = self.node_span_covering(start, chain.span);
            StmtOrTail::Stmt(Stmt {
                kind: StmtKind::Chain(chain),
                span,
            })
        } else if self.at(TokenKind::RBrace) || self.at_eof() {
            // Chain is the block's tail (a bare expr or an unterminated chain).
            StmtOrTail::Tail(chain)
        } else if last_is_block {
            // Block-final stage: `;` optional (W10). Also consume a stray `;`
            // handled above; here we have no `;`.
            self.eat(TokenKind::Semi);
            let span = self.node_span_covering(start, chain.span);
            StmtOrTail::Stmt(Stmt {
                kind: StmtKind::Chain(chain),
                span,
            })
        } else {
            // Missing `;`.
            self.diag("P0001", self.cur_span(), "expected `;` after statement");
            self.sync_to_stmt_end();
            let span = self.node_span_covering(start, chain.span);
            StmtOrTail::Stmt(Stmt {
                kind: StmtKind::Chain(chain),
                span,
            })
        }
    }

    /// `chain := expr stage* | stage+` (headed | headless).
    pub(super) fn parse_chain(&mut self) -> Chain {
        let start = self.cur_span().start;
        let head = if self.at(TokenKind::Arrow) {
            // Headless chain (e.g. `-> loop;`, fanout branches).
            None
        } else {
            Some(self.parse_expr())
        };
        let mut stages = Vec::new();
        while self.at(TokenKind::Arrow) {
            let before = self.pos;
            stages.push(self.parse_stage());
            if self.pos == before {
                break;
            }
        }
        // A mixed-direction chain (`a -> b <- c`) → P0001 (W18).
        if self.at(TokenKind::BackArrow) && (!stages.is_empty() || head.is_some()) {
            let span = self.cur_span();
            self.diag(
                "P0001",
                span,
                "`<-` cannot follow a `->` stage (mixed-direction chain, W18)",
            );
        }
        // Cover the head and the last stage so a recovery placeholder that lands
        // just outside the consumed range still nests within the chain (J2).
        let mut span = self.node_span(start);
        if let Some(h) = &head {
            span = SourceLoc::new(span.start.min(h.span.start), span.end.max(h.span.end));
        }
        if let Some(s) = stages.last() {
            span = SourceLoc::new(span.start.min(s.span.start), span.end.max(s.span.end));
        }
        Chain { head, stages, span }
    }

    /// `stage := '->' stage-body` — caller has checked the `->`.
    pub(super) fn parse_stage(&mut self) -> Stage {
        let arrow_span = self.bump().span; // `->`
        let start = arrow_span.start;
        let kind = self.parse_stage_body();
        // Cover the stage's children so a recovery placeholder that lands just
        // outside the consumed range still nests within the stage (J2).
        let mut span = self.node_span(start);
        if let Some(ext) = stage_kind_extent(&kind) {
            span = SourceLoc::new(span.start.min(ext.start), span.end.max(ext.end));
        }
        Stage {
            arrow_span,
            kind,
            span,
        }
    }

    pub(super) fn parse_stage_body(&mut self) -> StageKind {
        match self.kind() {
            TokenKind::KwRet => {
                let _kw = self.bump().span;
                // Optional `.INT` projection.
                let proj = if self.at(TokenKind::Dot) && self.peek_kind(1) == TokenKind::Int {
                    self.bump(); // `.`
                    let span = self.bump().span; // INT
                    Some((self.parse_int(span), span))
                } else {
                    None
                };
                StageKind::Ret { proj }
            }
            TokenKind::KwLoop => {
                self.bump();
                StageKind::LoopJump
            }
            TokenKind::KwSeq => {
                // ADR-0019: `seq { … }` is a statement block in stage position —
                // the ordinary block production, not a fanout. Guard arms stay
                // illegal in it (a clean guard token → stray-guard P0004).
                self.bump(); // `seq`
                let compute = self.parse_compute();
                let mut body = if self.at(TokenKind::LBrace) {
                    self.parse_block()
                } else {
                    self.diag("P0001", self.cur_span(), "expected `{` for seq block");
                    self.empty_block()
                };
                body.compute = compute;
                StageKind::SeqBlock(body)
            }
            TokenKind::KwVoid => {
                // ⟦P0113⟧ `-> void { … }`.
                let kw = self.bump().span;
                self.diag(
                    "P0113",
                    kw,
                    "`void` is out of Mapal-Core (HANDOFF §4); planned for Core+1",
                );
                let branches = self.parse_fanout_block();
                StageKind::Fanout {
                    kind: FanoutKind::Void(kw),
                    branches,
                }
            }
            TokenKind::KwMap => {
                let _kw = self.bump().span;
                self.parse_map_fold(CollOp::Map)
            }
            TokenKind::KwFold => {
                let _kw = self.bump().span;
                self.parse_map_fold(CollOp::Fold)
            }
            TokenKind::KwMut => {
                // `-> mut y` / `-> mut y: T` — mut binding stage.
                let mut_span = self.bump().span;
                let name = self.expect_name("binding name");
                let ty = if self.at(TokenKind::Colon) {
                    self.bump();
                    Some(self.parse_type())
                } else {
                    None
                };
                StageKind::Bind {
                    mut_span: Some(mut_span),
                    name,
                    ty,
                }
            }
            TokenKind::LBrace => {
                // §14.4 block-stage classification.
                self.parse_block_stage()
            }
            // op-shorthand: a leading binary operator starts a hole-expression.
            k if is_op_shorthand_lead(k) => self.parse_op_shorthand(),
            // `-> IDENT : T` typed binding stage, OR a general expression stage,
            // OR an out-of-Core `Ident {` collection operator.
            TokenKind::Ident => self.parse_ident_stage(),
            // ⟦P0110⟧ labeled jump `-> :label` (ADR-0012). Recognized on the
            // `Colon` followed by an `Ident`; a `::` path (`Colon Colon`) is NOT a
            // label and falls through to the general expression stage below.
            TokenKind::Colon if self.peek_kind(1) == TokenKind::Ident => {
                let colon = self.bump().span; // `:`
                let name_span = self.bump().span; // IDENT
                let span = self.span(colon.start, name_span.end);
                self.diag(
                    "P0110",
                    span,
                    "labeled jumps (`-> :label`, ADR-0012) are out of Mapal-Core (HANDOFF §4); \
                     Mapal-Core's only back-edge jump is `-> loop`. Planned for Core+1",
                );
                StageKind::Error(span)
            }
            _ => {
                // General expression stage (ADR-0005: a -> b + c -> d).
                let expr = self.parse_expr();
                StageKind::Expr(expr)
            }
        }
    }

    /// `-> IDENT …`: a typed binding stage (`-> x: i32`), an out-of-Core
    /// collection operator (`-> filter { … }`, P0114), or a general expression
    /// stage (member access etc.).
    pub(super) fn parse_ident_stage(&mut self) -> StageKind {
        // Out-of-Core collection operator: `Ident {` with op-block prefix.
        if self.peek_kind(1) == TokenKind::LBrace && self.ident_brace_is_op_block() {
            let name_span = self.cur_span();
            let op = self.text(name_span).to_string();
            self.bump(); // the Ident
            self.diag(
                "P0114",
                name_span,
                format!(
                    "`{op}` is out of Mapal-Core (HANDOFF §4): only `map`/`fold` are Core \
                     collection operators. Planned for Core+1"
                ),
            );
            let (params, body) = self.parse_op_block();
            return StageKind::MapFold {
                op: CollOp::Map,
                params,
                body,
            };
        }
        // Typed binding stage: `IDENT : type` (a lone Ident followed by a SINGLE
        // `:`). A span-adjacent `::` is an out-of-Core path (P0109), handled as a
        // general expression stage below, so exclude it here.
        if self.peek_kind(1) == TokenKind::Colon && !self.next_is_path_colons() {
            let name = Name {
                span: self.bump().span,
            };
            self.bump(); // `:`
            let ty = self.parse_type();
            return StageKind::Bind {
                mut_span: None,
                name,
                ty: Some(ty),
            };
        }
        // ADR-0051 D2: `-> name<args>` — explicit generic arguments. In stage
        // position a name followed by `<` opens the list (a comparison cannot
        // stand as a stage); a list that does not close where the stage ends
        // falls back to the expression stage.
        if self.peek_kind(1) == TokenKind::Lt
            && let Some(e) = self.try_parse_stage_args()
        {
            return StageKind::Expr(e);
        }
        // General expression stage (targets like `abs`, `print`, member access).
        let expr = self.parse_expr();
        StageKind::Expr(expr)
    }

    /// ADR-0051 D2: `IDENT '<' size (',' size)* ','? '>'` ending the stage —
    /// the callee's explicit arguments, a prefix of its declared list (sizes
    /// as size expressions, formats as names). Speculative: a list that does
    /// not close, or a `>` the stage does not end at, restores the cursor and
    /// the diagnostics and returns `None`.
    pub(super) fn try_parse_stage_args(&mut self) -> Option<Expr> {
        let (pos, ndiag, cooldown, depth) = (
            self.pos,
            self.diagnostics.len(),
            self.cooldown_at,
            self.depth_exceeded,
        );
        let start = self.cur_span().start;
        let name = Name {
            span: self.bump().span,
        };
        self.bump(); // `<`
        let mut args = Vec::new();
        let closed = loop {
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
        let ends_stage = matches!(
            self.kind(),
            TokenKind::Arrow
                | TokenKind::Semi
                | TokenKind::RBrace
                | TokenKind::RParen
                | TokenKind::RBracket
                | TokenKind::Comma
                | TokenKind::Eof
        );
        if closed && ends_stage {
            return Some(Expr {
                kind: ExprKind::Generic { name, args },
                span: self.span(start, self.prev_end()),
            });
        }
        self.pos = pos;
        self.diagnostics.truncate(ndiag);
        self.cooldown_at = cooldown;
        self.depth_exceeded = depth;
        None
    }

    /// Whether the next two tokens are a span-adjacent `::` (a path, P0109) — so
    /// the `IDENT :` typed-binding rule must not claim the first colon.
    pub(super) fn next_is_path_colons(&self) -> bool {
        self.peek_kind(1) == TokenKind::Colon
            && self.peek_kind(2) == TokenKind::Colon
            && self.peek(1).span.end == self.peek(2).span.start
    }

    /// op-block sniff: `Ident { IDENT (',' IDENT)* '->' …` (DESIGN §14.4).
    pub(super) fn ident_brace_is_op_block(&self) -> bool {
        let open = self.pos + 1; // the `{`
        debug_assert_eq!(self.tokens[open].kind, TokenKind::LBrace);
        let mut i = open + 1;
        // At least one Ident.
        if self.tokens.get(i).map(|t| t.kind) != Some(TokenKind::Ident) {
            return false;
        }
        i += 1;
        while self.tokens.get(i).map(|t| t.kind) == Some(TokenKind::Comma) {
            i += 1;
            if self.tokens.get(i).map(|t| t.kind) != Some(TokenKind::Ident) {
                return false;
            }
            i += 1;
        }
        self.tokens.get(i).map(|t| t.kind) == Some(TokenKind::Arrow)
    }

    /// `('map' | 'fold') op-block` — caller consumed the keyword.
    pub(super) fn parse_map_fold(&mut self, op: CollOp) -> StageKind {
        let compute = self.parse_compute();
        if !self.at(TokenKind::LBrace) {
            self.diag(
                "P0001",
                self.cur_span(),
                "expected `{` op-block after `map`/`fold`",
            );
            return StageKind::MapFold {
                op,
                params: Vec::new(),
                body: Block {
                    compute,
                    ..self.empty_block()
                },
            };
        }
        let (params, mut body) = self.parse_op_block();
        body.compute = compute;
        StageKind::MapFold { op, params, body }
    }

    /// `op-block := '{' IDENT (',' IDENT)* '->' block-item* tail? '}'`
    /// (DESIGN §14.3). A `(` pattern parameter → ⟦P0116⟧.
    pub(super) fn parse_op_block(&mut self) -> (Vec<Name>, Block) {
        let start = self.cur_span().start;
        if !self.enter() {
            self.skip_balanced(TokenKind::LBrace, TokenKind::RBrace);
            self.leave();
            return (Vec::new(), self.empty_block());
        }
        self.bump(); // `{`
        let mut params = Vec::new();
        // Parse params: plain idents only; `(` pattern → P0116. §14.3 grammar
        // `IDENT (',' IDENT)*` permits NO trailing comma and requires ≥1 param.
        let mut saw_param = false; // an Ident or a `(` destructuring pattern
        let mut trailing_comma: Option<SourceLoc> = None;
        loop {
            match self.kind() {
                TokenKind::Ident => {
                    saw_param = true;
                    trailing_comma = None;
                    params.push(Name {
                        span: self.bump().span,
                    });
                }
                TokenKind::LParen => {
                    // ⟦P0116⟧ destructuring op-block parameter.
                    saw_param = true;
                    trailing_comma = None;
                    let span = self.cur_span();
                    self.diag(
                        "P0116",
                        span,
                        "destructuring op-block parameters (`(x, y)`) are out of Mapal-Core \
                         (HANDOFF §4); planned for Core+1",
                    );
                    self.skip_balanced(TokenKind::LParen, TokenKind::RParen);
                }
                _ => break,
            }
            if self.at(TokenKind::Comma) {
                trailing_comma = Some(self.cur_span());
                self.bump();
            } else {
                break;
            }
        }
        // §14.3: a trailing comma before `->` is rejected; the params parsed so
        // far are kept so recovery continues (the `->` parse below still runs).
        if let Some(comma_span) = trailing_comma {
            self.diag(
                "P0001",
                comma_span,
                "op-block parameters take no trailing comma before `->`",
            );
        }
        // §14.3: an op-block needs at least one parameter (`{ -> … }` is invalid).
        if !saw_param {
            self.diag(
                "P0001",
                self.cur_span(),
                "op-block needs at least one parameter before `->`",
            );
        }
        if self.eat(TokenKind::Arrow).is_none() {
            self.diag(
                "P0001",
                self.cur_span(),
                "expected `->` after op-block parameters",
            );
        }
        // An op-block body is not a guard block: stray guards → P0004.
        let (items, tail) = self.parse_block_body(false);
        if self.eat(TokenKind::RBrace).is_none() {
            self.diag(
                "P0002",
                self.span(start, start + 1),
                "unclosed op-block `{`",
            );
        }
        self.leave();
        let span = self.block_span(start, &items, &tail);
        (
            params,
            Block {
                items,
                tail,
                span,
                compute: None,
            },
        )
    }

    /// `op-shorthand := BINOP hole-expr` — a leading binary operator after `->`.
    /// The piped value is the hole (leftmost leaf); parse by the precedence
    /// climber with a synthetic `Hole` lhs at the operator's level (DESIGN §14.3).
    pub(super) fn parse_op_shorthand(&mut self) -> StageKind {
        let op_tok = self.cur();
        let op = binop_of(op_tok.kind).expect("op-shorthand lead is a binop");
        let level = binop_level(op);
        let start = op_tok.span.start;
        let hole = Expr {
            kind: ExprKind::Hole,
            span: SourceLoc::empty_at(start),
        };
        // Climb from `hole` at the operator's level.
        let expr = self.parse_binop_rhs(hole, level);
        StageKind::OpShorthand { expr }
    }

    // ======================================================================
    // §14.4 — block-stage classification
    // ======================================================================

    /// A `{` in stage position parses as a generic block, then classifies
    /// (DESIGN §14.4): GuardBlock / Fanout / StmtBlock.
    pub(super) fn parse_block_stage(&mut self) -> StageKind {
        // Empty `{}` → Fanout(0) + P0010.
        if self.peek_kind(1) == TokenKind::RBrace {
            let start = self.cur_span().start;
            self.bump(); // `{`
            self.bump(); // `}`
            let end = self.prev_end();
            self.diag(
                "P0010",
                self.span(start, end),
                "empty block `{}` in stage position — classified as a zero-branch fanout",
            );
            return StageKind::Fanout {
                kind: FanoutKind::Plain,
                branches: Vec::new(),
            };
        }
        // Look at the first block-item: is it an arm? If so it's a GuardBlock.
        // We parse the block generically and classify by content.
        self.parse_classified_block()
    }

    /// Parse a stage-position block and classify it (DESIGN §14.4 table).
    pub(super) fn parse_classified_block(&mut self) -> StageKind {
        let start = self.cur_span().start;
        if !self.enter() {
            self.skip_balanced(TokenKind::LBrace, TokenKind::RBrace);
            self.leave();
            return StageKind::Error(self.node_span(start));
        }
        self.bump(); // `{`
        // Stage-position block (`-> { … }`): guard arms ARE legitimate here.
        let (items, tail) = self.parse_block_body(true);
        if self.eat(TokenKind::RBrace).is_none() {
            self.diag("P0002", self.span(start, start + 1), "unclosed `{`");
        }
        self.leave();
        let span = self.block_span(start, &items, &tail);

        let all_arms = !items.is_empty()
            && items.iter().all(|i| matches!(i, BlockItem::Arm(_)))
            && tail.is_none();
        let has_arm = items.iter().any(|i| matches!(i, BlockItem::Arm(_)));

        if all_arms {
            // GuardBlock — arm order preserved.
            let arms = items
                .into_iter()
                .map(|i| match i {
                    BlockItem::Arm(a) => a,
                    BlockItem::Stmt(_) => unreachable!("all_arms guarantees arms"),
                })
                .collect();
            return StageKind::Guard(arms);
        }
        if has_arm {
            // A tail among arms draws no P0006 from `check_arm_mixing` — it
            // scans items, and a tail is not an item. Name the mixing at the
            // tail's own span (B2).
            if let Some(t) = &tail {
                self.diagnostics.push(Diagnostic::error(
                    "P0006",
                    t.span,
                    "block tail mixed with guard arms in the same block; \
                     a guard block contains only arms and no tail",
                ));
            }
        }
        // Arms mixed with statements and/or a tail (B2, 2026-08-01 review):
        // NEVER recover by filtering — the old branch kept the arms and
        // dropped the rest, and a tail vanished with no diagnostic at all:
        // `{ -true-> 1; -false-> 2; 999 }` lost `999` from the AST while
        // parsing cleanly. Fall through to the StmtBlock arm instead: the
        // complete parsed block is retained, P0006 names the mixing, P0115
        // marks it out of Core, and lower rejects a StmtBlock stage loudly.
        // No accepted source loses data.

        // Fanout if all items are headless chains with no tail; else StmtBlock.
        let all_headless = !items.is_empty()
            && items.iter().all(|i| match i {
                BlockItem::Stmt(Stmt {
                    kind: StmtKind::Chain(c),
                    ..
                }) => c.head.is_none(),
                _ => false,
            })
            && tail.is_none();

        if all_headless {
            let branches = items
                .into_iter()
                .filter_map(|i| match i {
                    BlockItem::Stmt(Stmt {
                        kind: StmtKind::Chain(c),
                        ..
                    }) => Some(c),
                    _ => None,
                })
                .collect();
            StageKind::Fanout {
                kind: FanoutKind::Plain,
                branches,
            }
        } else {
            // ⟦P0115⟧ anonymous block stage.
            self.diag(
                "P0115",
                span,
                "anonymous block stage (`-> { … }` with statements/tail) is out of Mapal-Core \
                 (HANDOFF §4); user-guide §8.3 full-language form. Planned for Core+1",
            );
            StageKind::StmtBlock(Block {
                items,
                tail,
                span,
                compute: None,
            })
        }
    }

    /// A fanout block payload for the `void` stage: `'{' headless-chains '}'`.
    /// (After ADR-0019 `seq` is a statement block, so `void` is the only caller.)
    /// Non-chain statements (`x <- e` rebinds, `loop`s) do not belong in a fanout
    /// block: each is flagged with **P0117** at its span, replacing the old silent
    /// `filter_map` drop (ADR-0019 defect #3).
    pub(super) fn parse_fanout_block(&mut self) -> Vec<Chain> {
        if !self.at(TokenKind::LBrace) {
            self.diag("P0001", self.cur_span(), "expected `{` for fanout block");
            return Vec::new();
        }
        let start = self.cur_span().start;
        if !self.enter() {
            self.skip_balanced(TokenKind::LBrace, TokenKind::RBrace);
            self.leave();
            return Vec::new();
        }
        self.bump(); // `{`
        // A seq/void fanout block holds headless chains, not guard arms.
        let (items, tail) = self.parse_block_body(false);
        if self.eat(TokenKind::RBrace).is_none() {
            self.diag("P0002", self.span(start, start + 1), "unclosed `{`");
        }
        self.leave();
        let mut branches = Vec::new();
        for item in items {
            match item {
                BlockItem::Stmt(Stmt {
                    kind: StmtKind::Chain(c),
                    ..
                }) => branches.push(c),
                // ⟦P0117⟧ a rebind / loop is not a fanout branch — flag, don't drop
                // silently. Cooldown is irrelevant here: the cursor is settled past
                // `}`, so `self.diag`'s cooldown would collapse every drop after the
                // first into silence. Push directly, exactly as `check_arm_mixing`
                // does, so *each* dropped statement is surfaced (ADR-0019 defect #3).
                BlockItem::Stmt(Stmt {
                    kind: StmtKind::Bind(_) | StmtKind::Loop(_),
                    span,
                }) => self.diagnostics.push(Diagnostic::error(
                    "P0117",
                    span,
                    "only chains are fanout branches; `x <- e` / `loop` statements \
                     do not belong in a fanout block",
                )),
                // Error stmts and stray/spaced arms already carry their own
                // diagnostics (P0004/P0005/P0006), so they stay silent here.
                _ => {}
            }
        }
        // The final chain written without a trailing `;` is the block tail — it is a
        // branch too (the `;` is optional), not a silent drop (ADR-0019 defect #3).
        if let Some(c) = tail {
            branches.push(c);
        }
        branches
    }
}
