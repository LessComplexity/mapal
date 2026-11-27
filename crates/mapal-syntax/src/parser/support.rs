//! Shared parser helpers, error recovery and top-level statements.

use super::*;

impl<'a> Parser<'a> {
    // ======================================================================
    // shared helpers
    // ======================================================================

    /// Re-parse the digits of an `Int`/projection span, clamping to `u64::MAX`
    /// silently (the lexer already emitted L0008 on overflow; never double-report,
    /// C14).
    pub(super) fn parse_int(&self, span: SourceLoc) -> u64 {
        self.text(span).parse::<u64>().unwrap_or(u64::MAX)
    }

    /// Expect an identifier; on absence, emit P0001 and return a zero-width name
    /// at the consumed boundary (so it stays within any enclosing node, J2).
    pub(super) fn expect_name(&mut self, what: &str) -> Name {
        if self.at(TokenKind::Ident) {
            Name {
                span: self.bump().span,
            }
        } else {
            self.diag("P0001", self.cur_span(), format!("expected {what}"));
            Name { span: self.here() }
        }
    }

    /// Expect a statement-terminating `;`; on absence emit P0001.
    pub(super) fn expect_semi(&mut self, what: &str) {
        if self.eat(TokenKind::Semi).is_none() {
            self.diag(
                "P0001",
                self.cur_span(),
                format!("expected `;` after {what}"),
            );
            self.sync_to_stmt_end();
        }
    }

    /// Statement-level sync (DESIGN §16.1): skip to `;` (consume) or `}` (leave)
    /// or a statement-start keyword (leave).
    pub(super) fn sync_to_stmt_end(&mut self) {
        loop {
            match self.kind() {
                TokenKind::Semi => {
                    self.bump();
                    break;
                }
                TokenKind::RBrace
                | TokenKind::Eof
                | TokenKind::KwLoop
                | TokenKind::KwFn
                | TokenKind::KwType => break,
                _ => {
                    self.bump();
                }
            }
        }
    }

    /// Skip a balanced `open … close` group (depth-counted), consuming through
    /// the matching close. Used to skip annotation/executor/generic bodies. If
    /// the cursor is not at `open`, it is a no-op.
    pub(super) fn skip_balanced(&mut self, open: TokenKind, close: TokenKind) {
        if !self.at(open) {
            return;
        }
        let mut depth = 0i32;
        loop {
            let k = self.kind();
            if k == TokenKind::Eof {
                self.diag(
                    "P0002",
                    self.cur_span(),
                    "unclosed delimiter at end of input",
                );
                break;
            }
            if k == open {
                depth += 1;
            } else if k == close {
                depth -= 1;
                if depth == 0 {
                    self.bump();
                    break;
                }
            }
            self.bump();
        }
    }

    /// ADR-0048 D1: a top-level chain statement is a binding (`2048 ->
    /// hidden_size;`); any other statement is ⟦P0012⟧ (parsed for spans,
    /// stored as `Item::Error`). A chain without its `;` is P0001.
    pub(super) fn parse_top_level_statement(&mut self) -> Item {
        let start = self.cur_span().start;
        let span0 = self.cur_span();
        let before = self.pos;
        let parsed = self.parse_statement_or_tail();
        if self.pos == before {
            // Could not consume — skip one token to make progress (J1).
            self.diag(
                "P0012",
                span0,
                "statements are not allowed at the top level; wrap them in a function",
            );
            self.bump();
            return Item::Error(self.node_span(start));
        }
        match parsed {
            StmtOrTail::Stmt(Stmt {
                kind: StmtKind::Chain(chain),
                span,
            }) => Item::Bind(TopBind { chain, span }),
            StmtOrTail::Tail(chain) => {
                self.diag(
                    "P0001",
                    self.cur_span(),
                    "expected `;` after a top-level binding (`2048 -> hidden_size;`)",
                );
                let span = chain.span;
                Item::Bind(TopBind { chain, span })
            }
            StmtOrTail::Stmt(_) => {
                self.diag(
                    "P0012",
                    span0,
                    "only a binding chain (`2048 -> hidden_size;`) may stand at the top level; \
                     wrap other statements in a function",
                );
                Item::Error(self.node_span(start))
            }
        }
    }
}
