//! Source layout over the existing lexer, preserving lexemes and line comments.

use crate::{Diagnostic, Severity, SourceLoc, Token, TokenKind, lex, parse};

/// Format valid Mapal source using four-space indentation and multiline braces.
///
/// No files are read or written. Invalid input returns the parser's diagnostics;
/// output is returned only after checking that it parses and retains every token
/// kind and lexeme. Comments and literal spellings are retained verbatim (CRLF
/// line endings become LF). Intentional line and paragraph breaks are retained.
pub fn format_source(source: &str) -> Result<String, Vec<Diagnostic>> {
    let diagnostics = parse(source).diagnostics;
    if diagnostics.iter().any(|d| d.severity == Severity::Error) {
        return Err(diagnostics);
    }

    let tokens = lex(source).tokens;
    let formatted = Formatter::new(source, &tokens).run();
    let output_tokens = lex(&formatted).tokens;
    let same_tokens = tokens.len() == output_tokens.len()
        && tokens
            .iter()
            .zip(&output_tokens)
            .all(|(a, b)| a.kind == b.kind && token_text(source, a) == token_text(&formatted, b));
    if !same_tokens
        || parse(&formatted)
            .diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error)
    {
        return Err(vec![Diagnostic::error(
            "F0001",
            SourceLoc::empty_at(0),
            "formatting could not preserve this source; the original text was left unchanged",
        )]);
    }
    Ok(formatted)
}

fn token_text<'a>(source: &'a str, token: &Token) -> &'a str {
    &source[token.span.start as usize..token.span.end as usize]
}

#[derive(Clone, Copy)]
struct Frame {
    open: TokenKind,
    indent: usize,
    /// A map/fold parameter list ends at its first arrow, not at its commas.
    params: bool,
}

struct Formatter<'a> {
    source: &'a str,
    tokens: &'a [Token],
    out: String,
    frames: Vec<Frame>,
    line_indent: usize,
    previous: Option<TokenKind>,
    previous_unary: bool,
    break_after: bool,
    after_params: bool,
}

impl<'a> Formatter<'a> {
    fn new(source: &'a str, tokens: &'a [Token]) -> Self {
        Self {
            source,
            tokens,
            out: String::with_capacity(source.len()),
            frames: Vec::new(),
            line_indent: 0,
            previous: None,
            previous_unary: false,
            break_after: false,
            after_params: false,
        }
    }

    fn run(mut self) -> String {
        let mut end = 0;
        for (i, token) in self.tokens.iter().enumerate() {
            let gap = &self.source[end..token.span.start as usize];
            let source_break = gap.contains('\n');
            let has_comment = gap.contains("//");
            self.trivia(gap);
            let kind = token.kind;
            if kind == TokenKind::Eof {
                break;
            }

            let closing = matches!(
                kind,
                TokenKind::RBrace | TokenKind::RParen | TokenKind::RBracket
            );
            let frame = if closing { self.frames.pop() } else { None };
            let mandatory = self.break_after
                || kind == TokenKind::RBrace
                || matches!(kind, TokenKind::Guard(_))
                || (self.previous == Some(TokenKind::RBrace) && !continues_after_brace(kind));
            if mandatory {
                self.newline();
            }
            if self.frames.is_empty()
                && matches!(kind, TokenKind::KwFn | TokenKind::KwType)
                && self.previous.is_some()
                && !has_comment
            {
                self.blank_line();
            }

            let indent = if let Some(frame) = frame {
                frame.indent
            } else {
                let base = self.body_indent();
                // A manually wrapped expression/chain is one level deeper than
                // its containing block, regardless of how many times it wraps.
                let continuation = source_break
                    && !mandatory
                    && !self.after_params
                    && self
                        .frames
                        .last()
                        .is_some_and(|f| f.open == TokenKind::LBrace)
                    && !matches!(self.previous, None | Some(TokenKind::LBrace));
                base + usize::from(continuation)
            };
            if self.needs_space(kind) {
                self.space();
            }
            self.write(token_text(self.source, token), indent);

            self.break_after = false;
            self.after_params = false;
            match kind {
                TokenKind::LBrace | TokenKind::LParen | TokenKind::LBracket => {
                    self.frames.push(Frame {
                        open: kind,
                        indent: self.line_indent,
                        params: kind == TokenKind::LBrace
                            && opens_map_fold(self.source, self.tokens, i),
                    });
                    self.break_after = kind == TokenKind::LBrace;
                }
                TokenKind::Arrow => {
                    if let Some(frame) = self.frames.last_mut() {
                        if frame.params {
                            frame.params = false;
                            self.after_params = true;
                        }
                    }
                }
                TokenKind::Semi => {
                    self.break_after = self
                        .frames
                        .last()
                        .is_some_and(|f| f.open == TokenKind::LBrace);
                }
                TokenKind::Comma => {
                    self.break_after = self
                        .frames
                        .last()
                        .is_some_and(|f| f.open == TokenKind::LBrace && !f.params);
                }
                _ => {}
            }
            self.previous_unary = matches!(kind, TokenKind::Bang)
                || (kind == TokenKind::Minus && !self.previous.is_some_and(ends_expression));
            self.previous = Some(kind);
            end = token.span.end as usize;
        }
        while self.out.ends_with('\n') {
            self.out.pop();
        }
        if !self.out.is_empty() {
            self.out.push('\n');
        }
        self.out
    }

    fn body_indent(&self) -> usize {
        self.frames.last().map_or(0, |f| f.indent + 1)
    }

    fn line_start(&self) -> bool {
        self.out.is_empty() || self.out.ends_with('\n')
    }

    fn write(&mut self, text: &str, indent: usize) {
        if self.line_start() {
            self.line_indent = indent;
            for _ in 0..indent {
                self.out.push_str("    ");
            }
        }
        self.out.push_str(text);
    }

    fn space(&mut self) {
        if !self.line_start() && !self.out.ends_with(' ') {
            self.out.push(' ');
        }
    }

    fn newline(&mut self) {
        if !self.line_start() {
            self.out.push('\n');
        }
    }

    fn blank_line(&mut self) {
        self.newline();
        if !self.out.is_empty() && !self.out.ends_with("\n\n") {
            self.out.push('\n');
        }
    }

    /// Gaps contain only whitespace and `//` comments after parsing succeeds.
    /// Write trailing comments before deferred statement/delimiter breaks.
    fn trivia(&mut self, mut gap: &str) {
        loop {
            let whitespace = gap.len() - gap.trim_start_matches([' ', '\t', '\r', '\n']).len();
            let lines = gap[..whitespace].bytes().filter(|&b| b == b'\n').count();
            if lines >= 2 {
                self.blank_line();
            } else if lines == 1 {
                self.newline();
            }
            gap = &gap[whitespace..];
            if gap.is_empty() {
                break;
            }
            debug_assert!(gap.starts_with("//"));
            let end = gap.find('\n').unwrap_or(gap.len());
            let comment = gap[..end].strip_suffix('\r').unwrap_or(&gap[..end]);
            self.space();
            self.write(comment, self.body_indent());
            gap = &gap[end..];
            // Also terminate a comment at EOF: it must never eat a later token.
            self.newline();
        }
    }

    fn needs_space(&self, next: TokenKind) -> bool {
        use TokenKind::*;
        let Some(previous) = self.previous else {
            return false;
        };
        if self.line_start() {
            return false;
        }
        if next == Dot {
            // `x.0 .1` must not become `x.0.1` (the lexer would read 0.1).
            return previous == Int;
        }
        if matches!(next, RParen | RBracket | Comma | Semi | Colon)
            || matches!(previous, LParen | LBracket | Dot)
            || self.previous_unary
        {
            return false;
        }
        if next == LParen && previous == Ident {
            return false;
        }
        if next == LBracket && ends_expression(previous) {
            return false;
        }
        true
    }
}

/// Only map/fold braces introduce parameters. Inspect the immediate opener,
/// allowing its optional `: f32 | tf32` annotation; do not infer parameters
/// from a body's first chain, which can equally begin with an identifier.
fn opens_map_fold(source: &str, tokens: &[Token], open: usize) -> bool {
    let is_op = |kind| matches!(kind, TokenKind::KwMap | TokenKind::KwFold);
    if open > 0 && is_op(tokens[open - 1].kind) {
        return true;
    }
    let mut i = open;
    while i > 0 {
        i -= 1;
        match (tokens[i].kind, token_text(source, &tokens[i])) {
            (TokenKind::Ident, "f32" | "tf32") | (TokenKind::Error, "|") => {}
            (TokenKind::Colon, _) => return i > 0 && is_op(tokens[i - 1].kind),
            _ => return false,
        }
    }
    false
}

fn ends_expression(kind: TokenKind) -> bool {
    use TokenKind::*;
    matches!(
        kind,
        Ident | Int | Float | Str | KwTrue | KwFalse | RParen | RBracket | RBrace
    )
}

fn continues_after_brace(kind: TokenKind) -> bool {
    use TokenKind::*;
    matches!(
        kind,
        Arrow | Comma | Semi | RParen | RBracket | Dot | LBracket
    ) || matches!(
        kind,
        Plus | Minus
            | Star
            | Slash
            | Percent
            | EqEq
            | BangEq
            | Lt
            | Gt
            | Le
            | Ge
            | AmpAmp
            | PipePipe
    )
}
