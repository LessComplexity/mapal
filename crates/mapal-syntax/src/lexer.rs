//! The Mapal-Core lexer (DESIGN §4/§5): a single forward pass over the source,
//! maximal-munch, error-recovering, total (never panics).
//!
//! See `DESIGN.md` §4 (grammar), §5 (guard arrows / ADR-0010), §2 (diagnostic
//! codes L0001–L0008), and §8 (invariants I1–I5).

use crate::diag::{Diagnostic, SuggestedFix};
use crate::loc::SourceLoc;
use crate::token::{GuardKind, Token, TokenKind, keyword_kind};

/// The result of lexing: the token stream (always ending in `Eof`) and any
/// diagnostics produced during recovery.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LexOutput {
    pub tokens: Vec<Token>,
    pub diagnostics: Vec<Diagnostic>,
}

/// Lex `source` into tokens + diagnostics.
///
/// Total function: never panics, never fails (C2/C4). The returned `tokens`
/// always end with a zero-width `Eof`.
pub fn lex(source: &str) -> LexOutput {
    Lexer::new(source).run()
}

/// Scanner state. `pos` is a byte offset; `bytes` is the source as bytes for
/// O(1) lookahead. UTF-8 only matters inside comments/strings, which are
/// scanned char-boundary safely by re-deriving char widths from `source`.
struct Lexer<'a> {
    source: &'a str,
    bytes: &'a [u8],
    len: usize,
    pos: usize,
    tokens: Vec<Token>,
    diagnostics: Vec<Diagnostic>,
    /// Kind of the most recently emitted *token* (for the guard context gate).
    /// `None` at start of input. Trivia does not update this.
    last_kind: Option<TokenKind>,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Lexer {
            source,
            bytes: source.as_bytes(),
            len: source.len(),
            pos: 0,
            tokens: Vec::new(),
            diagnostics: Vec::new(),
            last_kind: None,
        }
    }

    fn run(mut self) -> LexOutput {
        while self.pos < self.len {
            let before = self.pos;
            self.skip_trivia();
            if self.pos >= self.len {
                break;
            }
            // If trivia advanced us to EOF or moved forward, re-check loop top.
            if self.pos != before && self.pos >= self.len {
                break;
            }
            self.scan_token();
            // I1: the scanner must always advance. Every `scan_*` path consumes
            // at least one byte; this is a defensive backstop.
            debug_assert!(
                self.pos > before || self.pos >= self.len,
                "lexer failed to advance at byte {before}"
            );
        }
        let eof_at = self.len as u32;
        self.push_token(TokenKind::Eof, SourceLoc::empty_at(eof_at));
        LexOutput {
            tokens: self.tokens,
            diagnostics: self.diagnostics,
        }
    }

    // --- byte helpers -----------------------------------------------------

    #[inline]
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    #[inline]
    fn peek_at(&self, n: usize) -> Option<u8> {
        self.bytes.get(self.pos + n).copied()
    }

    #[inline]
    fn bump(&mut self) {
        self.pos += 1;
    }

    // --- emission ---------------------------------------------------------

    fn push_token(&mut self, kind: TokenKind, span: SourceLoc) {
        // I2: spans ascending, non-overlapping, in-bounds, char-aligned.
        debug_assert!(span.start <= span.end, "span start after end: {span:?}");
        debug_assert!(
            span.end as usize <= self.len,
            "span out of bounds: {span:?} (len {})",
            self.len
        );
        debug_assert!(
            self.source.is_char_boundary(span.start as usize),
            "span start not on char boundary: {span:?}"
        );
        debug_assert!(
            self.source.is_char_boundary(span.end as usize),
            "span end not on char boundary: {span:?}"
        );
        if let Some(prev) = self.tokens.last() {
            debug_assert!(
                prev.span.start <= span.start && prev.span.end <= span.start,
                "non-ascending / overlapping spans: {:?} then {span:?}",
                prev.span
            );
        }
        self.tokens.push(Token::new(kind, span));
        self.last_kind = Some(kind);
    }

    fn span(&self, start: usize, end: usize) -> SourceLoc {
        SourceLoc::new(start as u32, end as u32)
    }

    // --- trivia -----------------------------------------------------------

    /// Skip whitespace, line comments, and `/* */` block comments (the last
    /// with an L0007 diagnostic). Char-boundary safe: comment bodies may hold
    /// arbitrary UTF-8.
    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(b' ') | Some(b'\t') | Some(b'\r') | Some(b'\n') => {
                    self.bump();
                }
                Some(b'/') if self.peek_at(1) == Some(b'/') => {
                    self.skip_line_comment();
                }
                Some(b'/') if self.peek_at(1) == Some(b'*') => {
                    self.skip_block_comment();
                }
                _ => break,
            }
        }
    }

    /// `//` to end of line (the `\n` itself is left to the whitespace branch).
    fn skip_line_comment(&mut self) {
        // Consume the two slashes.
        self.bump();
        self.bump();
        while let Some(b) = self.peek() {
            if b == b'\n' {
                break;
            }
            self.bump();
        }
    }

    /// `/* ... */` consumed as trivia, with L0007. Nesting is *not* honoured
    /// (Mapal has line comments only; this is purely friendly recovery).
    fn skip_block_comment(&mut self) {
        let start = self.pos;
        // Consume `/*`.
        self.bump();
        self.bump();
        let mut closed = false;
        while self.pos < self.len {
            if self.peek() == Some(b'*') && self.peek_at(1) == Some(b'/') {
                self.bump();
                self.bump();
                closed = true;
                break;
            }
            self.bump();
        }
        let end = self.pos;
        let span = self.span(start, end);
        let message = if closed {
            "block comments (`/* */`) are not Mapal syntax; use line comments (`//`)".to_string()
        } else {
            "block comments (`/* */`) are not Mapal syntax; use line comments (`//`) \
             — also unterminated, skipped to end of input"
                .to_string()
        };
        self.diagnostics
            .push(Diagnostic::error("L0007", span, message));
    }

    // --- token scanning ---------------------------------------------------

    fn scan_token(&mut self) {
        let b = match self.peek() {
            Some(b) => b,
            None => return,
        };

        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'_' => self.scan_ident_or_keyword(),
            b'0'..=b'9' => self.scan_number(),
            b'"' => self.scan_string(),
            b'-' => self.scan_minus(),
            b'<' => self.scan_lt(),
            b'>' => self.scan_gt(),
            b'=' => self.scan_eq(),
            b'!' => self.scan_bang(),
            b'&' => self.scan_amp(),
            b'|' => self.scan_pipe(),
            b'.' => self.scan_dot(),
            b'+' => self.single(TokenKind::Plus),
            b'*' => self.single(TokenKind::Star),
            b'/' => self.single(TokenKind::Slash),
            b'%' => self.single(TokenKind::Percent),
            b'(' => self.single(TokenKind::LParen),
            b')' => self.single(TokenKind::RParen),
            b'{' => self.single(TokenKind::LBrace),
            b'}' => self.single(TokenKind::RBrace),
            b'[' => self.single(TokenKind::LBracket),
            b']' => self.single(TokenKind::RBracket),
            b',' => self.single(TokenKind::Comma),
            b':' => self.single(TokenKind::Colon),
            b';' => self.single(TokenKind::Semi),
            b'?' => self.single(TokenKind::Question),
            b'@' => self.single(TokenKind::At),
            _ => self.scan_unknown(),
        }
    }

    /// Emit a one-byte token of `kind`.
    fn single(&mut self, kind: TokenKind) {
        let start = self.pos;
        self.bump();
        let span = self.span(start, self.pos);
        self.push_token(kind, span);
    }

    fn scan_ident_or_keyword(&mut self) {
        let start = self.pos;
        // First char already known to be [A-Za-z_].
        self.bump();
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'_' {
                self.bump();
            } else {
                break;
            }
        }
        let end = self.pos;
        let text = &self.source[start..end];
        let kind = keyword_kind(text).unwrap_or(TokenKind::Ident);
        let span = self.span(start, end);
        if kind == TokenKind::KwCategory {
            // L0004: reserved keyword; point at `type`, with a fix.
            let diag = Diagnostic::error(
                "L0004",
                span,
                "`category` is not a keyword in Mapal; the type-definition keyword is `type`",
            )
            .with_fix(SuggestedFix {
                span,
                replacement: "type".to_string(),
                label: "replace `category` with `type`",
            });
            self.diagnostics.push(diag);
        }
        self.push_token(kind, span);
    }

    /// Integer or float. Float requires digits on both sides of `.`
    /// (`ret.0` → `KwRet Dot Int`, `5.` → `Int Dot`). Integers exceeding u64
    /// clamp with L0008.
    fn scan_number(&mut self) {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                self.bump();
            } else {
                break;
            }
        }
        // Float? Need `.` followed by a digit.
        let is_float = self.peek() == Some(b'.')
            && self.peek_at(1).map(|b| b.is_ascii_digit()).unwrap_or(false);
        if is_float {
            // Consume the dot.
            self.bump();
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() {
                    self.bump();
                } else {
                    break;
                }
            }
            let span = self.span(start, self.pos);
            self.push_token(TokenKind::Float, span);
        } else {
            let end = self.pos;
            let span = self.span(start, end);
            // Range check for L0008 on integer literals.
            let text = &self.source[start..end];
            if text.parse::<u64>().is_err() {
                self.diagnostics.push(Diagnostic::error(
                    "L0008",
                    span,
                    "integer literal exceeds the maximum value (u64::MAX); value clamped",
                ));
            }
            self.push_token(TokenKind::Int, span);
        }
    }

    /// String literal `"..."` on a single line. Handles escapes (L0003 for
    /// unknown), and L0002 for raw newline / EOF before the closing quote.
    fn scan_string(&mut self) {
        let start = self.pos;
        // Consume opening quote.
        self.bump();
        let mut unterminated = false;
        loop {
            match self.peek() {
                None => {
                    unterminated = true;
                    break;
                }
                Some(b'\n') => {
                    // Raw newline before close: token ends here, '\n' is trivia.
                    unterminated = true;
                    break;
                }
                Some(b'"') => {
                    self.bump(); // closing quote
                    break;
                }
                Some(b'\\') => {
                    let esc_start = self.pos;
                    self.bump(); // backslash
                    match self.peek() {
                        Some(b'\\') | Some(b'"') | Some(b'n') | Some(b't') => {
                            self.bump();
                        }
                        Some(b'\n') | None => {
                            // Backslash at EOL/EOF: unknown escape, then the
                            // string is unterminated at the break point.
                            let span = self.span(esc_start, self.pos);
                            self.diagnostics.push(Diagnostic::error(
                                "L0003",
                                span,
                                "unknown escape sequence; backslash taken literally",
                            ));
                            unterminated = true;
                            break;
                        }
                        Some(_) => {
                            // Unknown escape: take the escaped char literally,
                            // continue scanning the string.
                            let esc_end = self.next_char_boundary(self.pos);
                            let span = self.span(esc_start, esc_end);
                            self.diagnostics.push(Diagnostic::error(
                                "L0003",
                                span,
                                "unknown escape sequence; character taken literally",
                            ));
                            self.pos = esc_end;
                        }
                    }
                }
                Some(_) => {
                    // Ordinary char (possibly multi-byte UTF-8): advance one char.
                    self.pos = self.next_char_boundary(self.pos);
                }
            }
        }
        let end = self.pos;
        let span = self.span(start, end);
        if unterminated {
            self.diagnostics.push(Diagnostic::error(
                "L0002",
                span,
                "unterminated string literal (reached end of line or end of input \
                 before the closing `\"`)",
            ));
        }
        self.push_token(TokenKind::Str, span);
    }

    /// Advance from a char-boundary byte offset to the start of the next char.
    fn next_char_boundary(&self, from: usize) -> usize {
        let mut i = from + 1;
        while i < self.len && !self.source.is_char_boundary(i) {
            i += 1;
        }
        i.min(self.len)
    }

    /// `-`: may begin a guard arrow (§5/ADR-0010), `->`, or `Minus`.
    fn scan_minus(&mut self) {
        if self.try_guard_arrow() {
            // The guard token was emitted and `pos` advanced; nothing else to do.
            return;
        }
        let start = self.pos;
        // Ordinary: `->` if `>` follows, else `-`.
        if self.peek_at(1) == Some(b'>') {
            self.bump();
            self.bump();
            let span = self.span(start, self.pos);
            self.push_token(TokenKind::Arrow, span);
        } else {
            self.bump();
            let span = self.span(start, self.pos);
            self.push_token(TokenKind::Minus, span);
        }
    }

    /// Attempt the guard-arrow production `-D->` with zero interior whitespace,
    /// gated on the previous-token context (`{`, `;`, `}`, or start of input).
    ///
    /// Returns `true` and emits a `Guard` token on success, `false` if the
    /// production does not apply (the caller falls back to ordinary `-`/`->`).
    fn try_guard_arrow(&mut self) -> bool {
        // Context gate: prev token must be one of `{`, `;`, `}`, or none.
        let ctx_ok = matches!(
            self.last_kind,
            None | Some(TokenKind::LBrace) | Some(TokenKind::Semi) | Some(TokenKind::RBrace)
        );
        if !ctx_ok {
            return false;
        }

        let start = self.pos;
        debug_assert_eq!(self.peek(), Some(b'-'));
        // Position just after the leading '-'.
        let mut i = start + 1;

        // Parse the discriminant D at byte offset i.
        let d_start = i;
        let (kind, after_d) = match self.match_discriminant(i) {
            Some(d) => d,
            None => return false,
        };
        i = after_d;

        // Require an immediately-following `->` (zero interior whitespace).
        if self.bytes.get(i).copied() != Some(b'-') {
            return false;
        }
        if self.bytes.get(i + 1).copied() != Some(b'>') {
            return false;
        }
        let end = i + 2;

        // Commit: emit the guard token spanning `start..end`.
        self.pos = end;
        let span = self.span(start, end);
        // L0008 (DESIGN §2/§5): an integer discriminant that overflows u64 is
        // clamped to u64::MAX *and* flagged. Detect overflow by re-parsing the
        // discriminant digits; an exact u64::MAX value is not flagged.
        if let GuardKind::Int(_) = kind {
            let digits = &self.source[d_start..after_d];
            if digits.parse::<u64>().is_err() {
                self.diagnostics.push(Diagnostic::error(
                    "L0008",
                    span,
                    "guard discriminant exceeds the maximum value (u64::MAX); value clamped",
                ));
            }
        }
        self.push_token(TokenKind::Guard(kind), span);
        true
    }

    /// Match a guard discriminant `D ∈ { true, false, _, [0-9]+ }` at byte
    /// offset `at`. Returns `(GuardKind, byte offset just past D)`.
    ///
    /// The match must be *exactly* the keyword/underscore/digit-run; for the
    /// keyword/underscore forms the next byte must terminate the discriminant
    /// (a guard discriminant is followed immediately by `-`), which is checked
    /// by the caller requiring `-` next. We additionally ensure `true`/`false`/
    /// `_` are not a prefix of a longer identifier by checking the following
    /// byte is `-` (the only valid continuation), preventing e.g. `-trueX->`
    /// from matching `true`.
    fn match_discriminant(&self, at: usize) -> Option<(GuardKind, usize)> {
        // Digit run.
        if self
            .bytes
            .get(at)
            .map(|b| b.is_ascii_digit())
            .unwrap_or(false)
        {
            let mut j = at;
            while self
                .bytes
                .get(j)
                .map(|b| b.is_ascii_digit())
                .unwrap_or(false)
            {
                j += 1;
            }
            let text = &self.source[at..j];
            let value = text.parse::<u64>().unwrap_or(u64::MAX);
            return Some((GuardKind::Int(value), j));
        }
        // `_`
        if self.bytes.get(at).copied() == Some(b'_') {
            // Ensure it's a lone `_`, not `_foo`: next byte must be `-`.
            if self.bytes.get(at + 1).copied() == Some(b'-') {
                return Some((GuardKind::Default, at + 1));
            }
            return None;
        }
        // `true`
        if self.source[at..].starts_with("true") {
            let next = at + 4;
            if self.bytes.get(next).copied() == Some(b'-') {
                return Some((GuardKind::True, next));
            }
            return None;
        }
        // `false`
        if self.source[at..].starts_with("false") {
            let next = at + 5;
            if self.bytes.get(next).copied() == Some(b'-') {
                return Some((GuardKind::False, next));
            }
            return None;
        }
        None
    }

    /// `<`: `<-` (BackArrow), `<=` (Le), or `<` (Lt).
    fn scan_lt(&mut self) {
        let start = self.pos;
        self.bump();
        let kind = match self.peek() {
            Some(b'-') => {
                self.bump();
                TokenKind::BackArrow
            }
            Some(b'=') => {
                self.bump();
                TokenKind::Le
            }
            _ => TokenKind::Lt,
        };
        let span = self.span(start, self.pos);
        self.push_token(kind, span);
    }

    /// `>`: `>=` (Ge) or `>` (Gt).
    fn scan_gt(&mut self) {
        let start = self.pos;
        self.bump();
        let kind = if self.peek() == Some(b'=') {
            self.bump();
            TokenKind::Ge
        } else {
            TokenKind::Gt
        };
        let span = self.span(start, self.pos);
        self.push_token(kind, span);
    }

    /// `=`: `==` (EqEq) or lone `=` (Error + L0005).
    fn scan_eq(&mut self) {
        let start = self.pos;
        self.bump();
        if self.peek() == Some(b'=') {
            self.bump();
            let span = self.span(start, self.pos);
            self.push_token(TokenKind::EqEq, span);
        } else {
            let span = self.span(start, self.pos);
            self.diagnostics.push(Diagnostic::error(
                "L0005",
                span,
                "unexpected `=`; use `<-` for binding or `==` for comparison",
            ));
            self.push_token(TokenKind::Error, span);
        }
    }

    /// `!`: `!=` (BangEq) or `!` (Bang).
    fn scan_bang(&mut self) {
        let start = self.pos;
        self.bump();
        let kind = if self.peek() == Some(b'=') {
            self.bump();
            TokenKind::BangEq
        } else {
            TokenKind::Bang
        };
        let span = self.span(start, self.pos);
        self.push_token(kind, span);
    }

    /// `&`: `&&` (AmpAmp) or lone `&` (Error + L0006).
    fn scan_amp(&mut self) {
        let start = self.pos;
        self.bump();
        if self.peek() == Some(b'&') {
            self.bump();
            let span = self.span(start, self.pos);
            self.push_token(TokenKind::AmpAmp, span);
        } else {
            let span = self.span(start, self.pos);
            self.diagnostics.push(Diagnostic::error(
                "L0006",
                span,
                "unexpected `&`; Mapal uses `&&` for logical and",
            ));
            self.push_token(TokenKind::Error, span);
        }
    }

    /// `|`: `||` (PipePipe) or lone `|` (Error + L0006).
    fn scan_pipe(&mut self) {
        let start = self.pos;
        self.bump();
        if self.peek() == Some(b'|') {
            self.bump();
            let span = self.span(start, self.pos);
            self.push_token(TokenKind::PipePipe, span);
        } else {
            let span = self.span(start, self.pos);
            self.diagnostics.push(Diagnostic::error(
                "L0006",
                span,
                "unexpected `|`; Mapal uses `||` for logical or",
            ));
            self.push_token(TokenKind::Error, span);
        }
    }

    /// `.`: `...` (DotDotDot) or `.` (Dot). A bare `..` yields two `Dot`s
    /// (handled by emitting one `Dot` at a time).
    fn scan_dot(&mut self) {
        let start = self.pos;
        if self.peek_at(1) == Some(b'.') && self.peek_at(2) == Some(b'.') {
            self.bump();
            self.bump();
            self.bump();
            let span = self.span(start, self.pos);
            self.push_token(TokenKind::DotDotDot, span);
        } else {
            self.bump();
            let span = self.span(start, self.pos);
            self.push_token(TokenKind::Dot, span);
        }
    }

    /// Unknown input: coalesce the maximal run of consecutive unknown chars
    /// into one `Error` token + one L0001 (DESIGN §2, W8). Char-boundary safe.
    fn scan_unknown(&mut self) {
        let start = self.pos;
        // Advance by whole chars while the current char does not start any
        // recognized token and is not trivia.
        loop {
            match self.peek() {
                None => break,
                Some(b) if self.starts_known(b) => break,
                Some(_) => {
                    self.pos = self.next_char_boundary(self.pos);
                }
            }
        }
        let end = self.pos;
        let span = self.span(start, end);
        let lexeme = &self.source[start..end];
        self.diagnostics.push(Diagnostic::error(
            "L0001",
            span,
            format!("unexpected character(s): `{lexeme}`"),
        ));
        self.push_token(TokenKind::Error, span);
    }

    /// Whether byte `b` can start a recognized token or trivia (so the
    /// unknown-run should stop). ASCII bytes that map to a token/trivia branch
    /// in `scan_token`/`skip_trivia` return true; everything else (non-ASCII,
    /// stray control chars, `'`, `$`, `#`, `~`, `^`, backtick, `\`) is unknown.
    fn starts_known(&self, b: u8) -> bool {
        matches!(
            b,
            b'A'..=b'Z' | b'a'..=b'z' | b'_'
                | b'0'..=b'9'
                | b'"'
                | b'-' | b'<' | b'>' | b'=' | b'!' | b'&' | b'|' | b'.'
                | b'+' | b'*' | b'/' | b'%'
                | b'(' | b')' | b'{' | b'}' | b'[' | b']'
                | b',' | b':' | b';' | b'?' | b'@'
                // trivia
                | b' ' | b'\t' | b'\r' | b'\n'
        )
    }
}

#[cfg(test)]
mod tests;
