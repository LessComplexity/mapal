//! Parser methods for guard arms (DESIGN section 14.3).

use super::*;

impl<'a> Parser<'a> {
    // ======================================================================
    // §14.3 — guard arms
    // ======================================================================

    /// `guard-arm := GUARD arm-payload | spaced-arm | pattern-arm`.
    pub(super) fn parse_guard_arm(&mut self) -> GuardArm {
        let start = self.cur_span().start;
        let (discr, discr_span) = self.parse_guard_discr();
        let payload = self.parse_arm_payload();
        let payload_span = match &payload {
            ArmPayload::Chain(c) => c.span,
            ArmPayload::Block(b) => b.span,
        };
        let span = self.node_span_covering(start, payload_span);
        GuardArm {
            discr,
            discr_span,
            payload,
            span,
        }
    }

    /// Parse the discriminant portion of a guard arm, returning its kind and the
    /// span covering the discriminant lexeme(s) up to and including the `->`.
    pub(super) fn parse_guard_discr(&mut self) -> (GuardDiscr, SourceLoc) {
        if let TokenKind::Guard(gk) = self.kind() {
            let span = self.bump().span;
            return (guard_discr_of(gk), span);
        }
        // Spaced bool/default arm (rule 1) → P0005, recovered as the meant arm.
        if self.at(TokenKind::Minus) && self.is_spaced_bool_or_default_arm() {
            let start = self.cur_span().start;
            self.bump(); // `-`
            let disc = match self.kind() {
                TokenKind::KwTrue => {
                    self.bump();
                    GuardDiscr::True
                }
                TokenKind::KwFalse => {
                    self.bump();
                    GuardDiscr::False
                }
                _ => {
                    // Ident `_`.
                    self.bump();
                    GuardDiscr::Default
                }
            };
            let arrow = self.eat(TokenKind::Arrow).unwrap_or_else(|| self.here());
            let span = self.span(start, arrow.end);
            // Fix: remove interior whitespace (write `-true->`).
            let lexeme = self.canonical_arm_lexeme(disc);
            self.diag_fix(
                "P0005",
                span,
                "guard arrows are written without interior spaces",
                SuggestedFix {
                    span,
                    replacement: lexeme,
                    label: "remove the spaces inside the guard arrow",
                },
            );
            return (disc, span);
        }
        // Pattern arm (rule 2) → P0106; OutOfCore discr, payload parsed.
        if self.at(TokenKind::Minus) && self.is_pattern_arm() {
            let start = self.cur_span().start;
            self.bump(); // `-`
            // Skip head + optional balanced group up to the `->`.
            if self.at(TokenKind::Ident) {
                self.bump();
            }
            if self.at(TokenKind::LParen) {
                self.skip_balanced(TokenKind::LParen, TokenKind::RParen);
            } else if self.at(TokenKind::LBracket) {
                self.skip_balanced(TokenKind::LBracket, TokenKind::RBracket);
            }
            let arrow = self.eat(TokenKind::Arrow).unwrap_or_else(|| self.here());
            let span = self.span(start, arrow.end);
            self.diag(
                "P0106",
                span,
                "pattern guard arms (`-Some(x)->`, `-[h, ...t]->`) are out of Mapal-Core \
                 (HANDOFF §4); planned for Core+1 coproducts",
            );
            return (GuardDiscr::OutOfCore, span);
        }
        // Should not be reached (classify_block_item gates this), but stay total.
        let span = self.cur_span();
        (GuardDiscr::OutOfCore, span)
    }

    /// Build the canonical no-space guard lexeme for a spaced-arm fix.
    pub(super) fn canonical_arm_lexeme(&self, disc: GuardDiscr) -> String {
        match disc {
            GuardDiscr::True => "-true->".to_string(),
            GuardDiscr::False => "-false->".to_string(),
            GuardDiscr::Default => "-_->".to_string(),
            // Invariant: P0005 (the only caller) fires solely for the spaced
            // bool/default arms, so `disc` is always True/False/Default here;
            // Int/OutOfCore discriminants never reach this fix builder.
            GuardDiscr::Int(_) | GuardDiscr::OutOfCore => {
                unreachable!("canonical_arm_lexeme only builds the P0005 bool/default fix")
            }
        }
    }

    /// `arm-payload := chain (';' per statement rule) | block`.
    pub(super) fn parse_arm_payload(&mut self) -> ArmPayload {
        // A plain payload block (`-true-> { … }`) is NOT classified (§14.4).
        if self.at(TokenKind::LBrace) {
            let block = self.parse_payload_block();
            ArmPayload::Block(block)
        } else {
            // A chain payload (incl. bare expr `-true-> x;` and headless
            // `-false-> -> ret;`).
            let start = self.cur_span().start;
            let chain = self.parse_chain();
            // Termination: consume `;` if present; a `}` ends the enclosing block.
            self.eat(TokenKind::Semi);
            let _ = start;
            ArmPayload::Chain(chain)
        }
    }

    /// A plain payload block: statements + tail, NOT classified (§14.4).
    pub(super) fn parse_payload_block(&mut self) -> Block {
        let start = self.cur_span().start;
        if !self.enter() {
            self.skip_balanced(TokenKind::LBrace, TokenKind::RBrace);
            self.leave();
            return self.empty_block();
        }
        self.bump(); // `{`
        // An arm-payload block is NOT classified (§14.4): it holds statements and
        // a tail, never guard arms — a leading guard here is stray (P0004).
        let (items, tail) = self.parse_block_body(false);
        if self.eat(TokenKind::RBrace).is_none() {
            self.diag("P0002", self.span(start, start + 1), "unclosed `{`");
        }
        self.leave();
        let span = self.block_span(start, &items, &tail);
        // A trailing `;` after the payload block is legal (statement form).
        self.eat(TokenKind::Semi);
        Block {
            items,
            tail,
            span,
            compute: None,
        }
    }
}
