//! The Mapal-Core recursive-descent parser (DESIGN §13–§19).
//!
//! Two tiers per ADR-0005: an expression parser (precedence levels 1–7, §14.6)
//! and a statement-level flow-chain parser (`->`/`<-` over expression operands).
//! A flow is a statement, never an expression value.
//!
//! The parser is a module of `mapal-syntax`: [`parse`] runs [`crate::lex`]
//! internally and merges lex+parse diagnostics, stably sorted by
//! `(span.start, span.end)` (§15/§18). It is **total** (J1): a shared depth
//! counter (P0011 at 128) and the §16.1 progress lemma (every `*`-loop iteration
//! consumes ≥1 token or exits) guarantee it never panics or hangs on adversarial
//! input.
//!
//! Out-of-Core v0.2 surface is parsed precisely and rejected with a dedicated
//! P-code (C13); the structure is kept for span-precise downstream messages.
//! Lexer diagnostics are never duplicated (C14): `Error` tokens are skipped, not
//! re-reported.

use crate::ast::*;
use crate::diag::{Diagnostic, SuggestedFix};
use crate::loc::SourceLoc;
use crate::token::{GuardKind, Token, TokenKind};

mod blocks;
mod chains;
mod exprs;
mod guards;
mod items;
mod support;
mod types;

// Re-export so the §18 public API `pub use parser::{parse, ParseOutput};` is
// satisfiable; `ParseOutput` itself is defined with the other §15 node types in
// `ast.rs`. A glob (`ast::*`) plus this explicit path resolve to one item.
pub use crate::ast::ParseOutput;

/// Parse `source` into a [`ParseOutput`] (DESIGN §18).
///
/// Total function: never panics, never hangs (J1). Lexes internally; the
/// returned diagnostics are the lexer's diagnostics followed by the parser's,
/// stably sorted by `(span.start, span.end)` so lexer and parser reports
/// interleave by position (J6 preserves the lexer diagnostics verbatim).
pub fn parse(source: &str) -> ParseOutput {
    let lexed = crate::lex(source);
    let mut p = Parser::new(source, &lexed.tokens, lexed.diagnostics);
    let program = p.parse_program();
    let mut diagnostics = p.into_diagnostics();
    // Stable sort by (start, end): lexer + parser diagnostics interleave by
    // position; ties keep insertion order (lex before parse at the same span).
    diagnostics.sort_by_key(|d| (d.span.start, d.span.end));
    ParseOutput {
        program,
        diagnostics,
    }
}

/// The shared recursion-depth limit (DESIGN §16.1). Examples nest ≤ 8; exceeding
/// this triggers P0011 and unwinds — the totality guard (J1) against a
/// recursive-descent stack overflow.
const DEPTH_LIMIT: u32 = 128;

struct Parser<'a> {
    source: &'a str,
    tokens: &'a [Token],
    /// Cursor into `tokens`. `Eof` is guaranteed last, so `cur()` never indexes
    /// out of bounds.
    pos: usize,
    diagnostics: Vec<Diagnostic>,
    /// Shared expression/block recursion depth (P0011 guard).
    depth: u32,
    /// Cooldown (DESIGN §16.1): after a diagnostic, suppress further ones until
    /// the cursor advances. Holds the cursor position at which the last
    /// diagnostic fired; `None` means no cooldown is active.
    cooldown_at: Option<usize>,
    /// Set once a P0011 fires, to avoid a cascade of depth diagnostics while
    /// unwinding.
    depth_exceeded: bool,
    /// Bracket nesting depth for the *current expression* (parens, brackets,
    /// call args, tuples, struct fields, indices). A `<-` is a W2/P0009 expression
    /// defect only when `> 0`; at statement level (`== 0`) a stray `<-` is the
    /// statement's concern (P0008 double-bind / P0001 mixed-direction).
    paren_depth: u32,
}

impl<'a> Parser<'a> {
    fn new(source: &'a str, tokens: &'a [Token], lex_diagnostics: Vec<Diagnostic>) -> Self {
        debug_assert!(
            tokens.last().map(|t| t.kind) == Some(TokenKind::Eof),
            "token stream must end with Eof"
        );
        Parser {
            source,
            tokens,
            pos: 0,
            diagnostics: lex_diagnostics,
            depth: 0,
            cooldown_at: None,
            depth_exceeded: false,
            paren_depth: 0,
        }
    }

    fn into_diagnostics(self) -> Vec<Diagnostic> {
        self.diagnostics
    }

    // --- cursor helpers ---------------------------------------------------

    #[inline]
    fn cur(&self) -> Token {
        // Eof is the last token; clamp defensively.
        self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    #[inline]
    fn kind(&self) -> TokenKind {
        self.cur().kind
    }

    #[inline]
    fn peek_kind(&self, n: usize) -> TokenKind {
        let i = (self.pos + n).min(self.tokens.len() - 1);
        self.tokens[i].kind
    }

    /// The token `n` ahead (clamped to `Eof`).
    #[inline]
    fn peek(&self, n: usize) -> Token {
        let i = (self.pos + n).min(self.tokens.len() - 1);
        self.tokens[i]
    }

    #[inline]
    fn at(&self, k: TokenKind) -> bool {
        self.kind() == k
    }

    #[inline]
    fn at_eof(&self) -> bool {
        self.kind() == TokenKind::Eof
    }

    /// Advance the cursor one token (never past `Eof`). Clears any active
    /// cooldown once real progress is made.
    #[inline]
    fn bump(&mut self) -> Token {
        let t = self.cur();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }

    /// Consume `k` if present, returning its span; otherwise `None`.
    fn eat(&mut self, k: TokenKind) -> Option<SourceLoc> {
        if self.at(k) {
            Some(self.bump().span)
        } else {
            None
        }
    }

    /// The span of the current token (for placing "expected" diagnostics).
    #[inline]
    fn cur_span(&self) -> SourceLoc {
        self.cur().span
    }

    /// Lexeme text for a span.
    #[inline]
    fn text(&self, span: SourceLoc) -> &'a str {
        &self.source[span.start as usize..span.end as usize]
    }

    /// A zero-width span at the **consumed-input boundary** (`prev_end`) — the
    /// position where a recovery placeholder (missing name/type/expr) is located.
    /// Using `prev_end` rather than the current (unconsumed) token guarantees a
    /// placeholder never overshoots into a token belonging to the next
    /// production, so it always sits at or before any enclosing node's end. The
    /// node-span builders (`node_span`, `node_span_covering`, and the explicit
    /// `min`/`max` widenings) take the union with children, so a placeholder that
    /// lands *before* a node's captured start is still nested (J2).
    #[inline]
    fn here(&self) -> SourceLoc {
        SourceLoc::empty_at(self.prev_end())
    }

    /// Close a node that began at `start`, ending at the consumed boundary
    /// (`prev_end`). When the node consumed nothing (a pure-recovery node where
    /// the cursor never advanced past `start`), `prev_end < start`; collapse both
    /// ends to `start` so the span is a well-formed zero-width region exactly at
    /// the node's entry point, where its `here()` recovery children also sit (J2).
    fn node_span(&self, start: u32) -> SourceLoc {
        let end = self.prev_end();
        if end < start {
            SourceLoc::empty_at(start)
        } else {
            self.span(start, end)
        }
    }

    /// Close a node spanning `[start, prev_end]` **widened** to also cover an
    /// already-built child span (used when a synthesized/recovery child may lie
    /// outside the consumed range, e.g. a missing `{` body block placed at the
    /// current token, or an op-shorthand `Hole` lhs before the operator).
    /// Guarantees `child ⊆ result` (J2).
    fn node_span_covering(&self, start: u32, child: SourceLoc) -> SourceLoc {
        let end = self.prev_end();
        let lo = start.min(child.start);
        let hi = end.max(child.end).max(lo);
        self.span(lo, hi)
    }

    /// The covering span of a block's body (items + tail), for widening a block
    /// span so every body node nests within it (J2). Returns `None` for an empty
    /// body.
    fn body_extent(items: &[BlockItem], tail: &Option<Chain>) -> Option<SourceLoc> {
        let mut acc: Option<SourceLoc> = None;
        let mut widen = |s: SourceLoc| {
            acc = Some(match acc {
                None => s,
                Some(a) => SourceLoc::new(a.start.min(s.start), a.end.max(s.end)),
            });
        };
        for it in items {
            match it {
                BlockItem::Stmt(s) => widen(s.span),
                BlockItem::Arm(a) => widen(a.span),
            }
        }
        if let Some(t) = tail {
            widen(t.span);
        }
        acc
    }

    // --- diagnostics (cooldown) -------------------------------------------

    /// Whether a new diagnostic may fire (cooldown released, DESIGN §16.1):
    /// after a diagnostic, none more until the cursor advances ≥1 token.
    fn cooldown_ok(&self) -> bool {
        match self.cooldown_at {
            None => true,
            Some(at) => self.pos > at,
        }
    }

    /// Emit a diagnostic under the cooldown rule; no-op if on cooldown.
    fn diag(&mut self, code: &'static str, span: SourceLoc, message: impl Into<String>) {
        if !self.cooldown_ok() {
            return;
        }
        self.diagnostics
            .push(Diagnostic::error(code, span, message));
        self.cooldown_at = Some(self.pos);
    }

    /// Emit a diagnostic with a suggested fix under the cooldown rule.
    fn diag_fix(
        &mut self,
        code: &'static str,
        span: SourceLoc,
        message: impl Into<String>,
        fix: SuggestedFix,
    ) {
        if !self.cooldown_ok() {
            return;
        }
        self.diagnostics
            .push(Diagnostic::error(code, span, message).with_fix(fix));
        self.cooldown_at = Some(self.pos);
    }

    // --- span helper with J2 debug-assert ---------------------------------

    /// Build a node span `[start, end)` and assert it is well-formed (J2). The
    /// child-within-parent checks live at the use sites (each constructor asserts
    /// its children's spans are within the span it builds).
    fn span(&self, start: u32, end: u32) -> SourceLoc {
        debug_assert!(start <= end, "node span start>end: {start}..{end}");
        debug_assert!(
            end as usize <= self.source.len(),
            "node span out of bounds: {start}..{end} (len {})",
            self.source.len()
        );
        SourceLoc::new(start, end)
    }

    /// The end offset of the most recently consumed token (for closing a node
    /// span at the right boundary). Falls back to the current token start.
    fn prev_end(&self) -> u32 {
        if self.pos == 0 {
            self.cur_span().start
        } else {
            self.tokens[self.pos - 1].span.end
        }
    }

    // --- depth guard ------------------------------------------------------

    /// Enter a recursion level; returns `false` (and emits P0011 once) if the
    /// shared depth limit is exceeded (DESIGN §16.1, J1).
    fn enter(&mut self) -> bool {
        self.depth += 1;
        if self.depth > DEPTH_LIMIT {
            if !self.depth_exceeded {
                let span = self.cur_span();
                self.diag(
                    "P0011",
                    span,
                    "nesting too deep (over 128 levels); parsing stopped here to stay total",
                );
                self.depth_exceeded = true;
            }
            false
        } else {
            true
        }
    }

    #[inline]
    fn leave(&mut self) {
        self.depth -= 1;
        if self.depth <= DEPTH_LIMIT {
            self.depth_exceeded = false;
        }
    }
}

/// The classification of a block-item at the cursor (DESIGN §14.3).
enum BlockItemClass {
    Arm,
    Statement,
}

/// Result of parsing a block item that may be a statement or the block tail.
enum StmtOrTail {
    Stmt(Stmt),
    Tail(Chain),
}

// --- free functions -------------------------------------------------------

/// The covering span of a stage kind's direct children (for widening the stage
/// span so every child nests within it, J2). `None` for childless stages.
fn stage_kind_extent(kind: &StageKind) -> Option<SourceLoc> {
    fn widen(acc: &mut Option<SourceLoc>, s: SourceLoc) {
        *acc = Some(match *acc {
            None => s,
            Some(a) => SourceLoc::new(a.start.min(s.start), a.end.max(s.end)),
        });
    }
    let mut acc = None;
    match kind {
        StageKind::Expr(e) | StageKind::OpShorthand { expr: e } => widen(&mut acc, e.span),
        StageKind::Bind { ty: Some(ty), .. } => widen(&mut acc, ty.span),
        StageKind::Guard(arms) => {
            for a in arms {
                widen(&mut acc, a.span);
            }
        }
        StageKind::Fanout { branches, .. } => {
            for b in branches {
                widen(&mut acc, b.span);
            }
        }
        StageKind::MapFold { body, .. }
        | StageKind::StmtBlock(body)
        | StageKind::SeqBlock(body) => widen(&mut acc, body.span),
        _ => {}
    }
    acc
}

/// Whether a stage kind is a "block form" (guard/fanout/seq/map/fold/stmt-block):
/// for these the trailing `;` is optional (DESIGN §14.3 termination rule, W10).
fn stage_is_block_form(kind: &StageKind) -> bool {
    matches!(
        kind,
        StageKind::Guard(_)
            | StageKind::Fanout { .. }
            | StageKind::MapFold { .. }
            | StageKind::SeqBlock(_)
            | StageKind::StmtBlock(_)
    )
}

/// Whether `k` is a leading binary operator that starts an op-shorthand after
/// `->` (DESIGN §14.3). `!` is NOT a shorthand (unary, starts a general expr).
/// A leading `-` is always subtraction shorthand here (W12).
fn is_op_shorthand_lead(k: TokenKind) -> bool {
    matches!(
        k,
        TokenKind::Plus
            | TokenKind::Minus
            | TokenKind::Star
            | TokenKind::Slash
            | TokenKind::Percent
            | TokenKind::EqEq
            | TokenKind::BangEq
            | TokenKind::Lt
            | TokenKind::Gt
            | TokenKind::Le
            | TokenKind::Ge
            | TokenKind::AmpAmp
            | TokenKind::PipePipe
    )
}

/// Map a token kind to a binary operator (for op-shorthand and the climber).
fn binop_of(k: TokenKind) -> Option<BinOp> {
    Some(match k {
        TokenKind::Plus => BinOp::Add,
        TokenKind::Minus => BinOp::Sub,
        TokenKind::Star => BinOp::Mul,
        TokenKind::Slash => BinOp::Div,
        TokenKind::Percent => BinOp::Mod,
        TokenKind::EqEq => BinOp::Eq,
        TokenKind::BangEq => BinOp::Ne,
        TokenKind::Lt => BinOp::Lt,
        TokenKind::Gt => BinOp::Gt,
        TokenKind::Le => BinOp::Le,
        TokenKind::Ge => BinOp::Ge,
        TokenKind::AmpAmp => BinOp::And,
        TokenKind::PipePipe => BinOp::Or,
        _ => return None,
    })
}

/// The precedence level of a binary operator (DESIGN §14.6): mul=3, add=4,
/// cmp=5, and=6, or=7.
fn binop_level(op: BinOp) -> u8 {
    match op {
        BinOp::Mul | BinOp::Div | BinOp::Mod => 3,
        BinOp::Add | BinOp::Sub => 4,
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => 5,
        BinOp::And => 6,
        BinOp::Or => 7,
    }
}

/// A comparison operator token → its `BinOp` (level 5).
fn cmp_op(k: TokenKind) -> Option<BinOp> {
    Some(match k {
        TokenKind::EqEq => BinOp::Eq,
        TokenKind::BangEq => BinOp::Ne,
        TokenKind::Lt => BinOp::Lt,
        TokenKind::Gt => BinOp::Gt,
        TokenKind::Le => BinOp::Le,
        TokenKind::Ge => BinOp::Ge,
        _ => return None,
    })
}

/// Map a clean `GuardKind` to a tree `GuardDiscr`.
fn guard_discr_of(gk: GuardKind) -> GuardDiscr {
    match gk {
        GuardKind::True => GuardDiscr::True,
        GuardKind::False => GuardDiscr::False,
        GuardKind::Default => GuardDiscr::Default,
        GuardKind::Int(n) => GuardDiscr::Int(n),
    }
}

/// Whether a token kind is a keyword (used by the `::` path segment rule:
/// `List::map` ends at `KwMap`).
fn is_keyword(k: TokenKind) -> bool {
    matches!(
        k,
        TokenKind::KwFn
            | TokenKind::KwType
            | TokenKind::KwMut
            | TokenKind::KwLoop
            | TokenKind::KwRet
            | TokenKind::KwSeq
            | TokenKind::KwMap
            | TokenKind::KwFold
            | TokenKind::KwTrue
            | TokenKind::KwFalse
            | TokenKind::KwCategory
            | TokenKind::KwExecutor
            | TokenKind::KwPub
            | TokenKind::KwUse
            | TokenKind::KwVoid
    )
}

#[cfg(test)]
mod tests;
