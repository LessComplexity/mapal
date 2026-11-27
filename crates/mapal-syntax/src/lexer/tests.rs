//! Unit tests for the lexer.

use super::*;
use crate::loc::LineIndex;
use crate::token::{GuardKind, TokenKind};

/// Lex and return just the kinds (dropping the trailing Eof).
fn kinds(src: &str) -> Vec<TokenKind> {
    let out = lex(src);
    let mut ks: Vec<TokenKind> = out.tokens.iter().map(|t| t.kind).collect();
    assert_eq!(ks.last(), Some(&TokenKind::Eof), "missing Eof");
    ks.pop();
    ks
}

fn lexeme<'a>(src: &'a str, t: &Token) -> &'a str {
    &src[t.span.start as usize..t.span.end as usize]
}

fn codes(src: &str) -> Vec<&'static str> {
    lex(src).diagnostics.iter().map(|d| d.code.0).collect()
}

use TokenKind::*;

// ---- §5 worked-consequences table (all 10 rows) ----------------------

#[test]
fn s5_row1_true_guard_after_brace() {
    // `-true->  x;` with prev token `{`.
    assert_eq!(
        kinds("{ -true->  x; }"),
        vec![LBrace, Guard(GuardKind::True), Ident, Semi, RBrace]
    );
}

#[test]
fn s5_row2_false_guard_after_rbrace() {
    // `-false-> -> ret;` with prev token `}`.
    // Construct: `{ } -false-> -> ret;`
    assert_eq!(
        kinds("{ } -false-> -> ret;"),
        vec![LBrace, RBrace, Guard(GuardKind::False), Arrow, KwRet, Semi]
    );
}

#[test]
fn s5_row3_default_guard_after_semi() {
    // `-_-> "unknown";` with prev token `;`.
    assert_eq!(
        kinds("a; -_-> \"unknown\";"),
        vec![Ident, Semi, Guard(GuardKind::Default), Str, Semi]
    );
}

#[test]
fn s5_row4_int_guard_after_brace() {
    assert_eq!(
        kinds("{ -42-> f;"),
        vec![LBrace, Guard(GuardKind::Int(42)), Ident, Semi]
    );
}

#[test]
fn s5_row5_spaced_minus_adjacency_fails() {
    // `-7 -> abs;` after `{`: adjacency fails → Minus Int Arrow Ident Semi.
    assert_eq!(
        kinds("{ -7 -> abs;"),
        vec![LBrace, Minus, Int, Arrow, Ident, Semi]
    );
}

#[test]
fn s5_row6_star_context_fails() {
    // `x * -1;` → Ident Star Minus Int Semi (prev Star; also adjacency
    // would fail since `;` follows `1`).
    assert_eq!(kinds("x * -1;"), vec![Ident, Star, Minus, Int, Semi]);
}

#[test]
fn s5_row7_ident_context_fails() {
    // `a-1->c;` → Ident Minus Int Arrow Ident Semi (prev Ident).
    assert_eq!(
        kinds("a-1->c;"),
        vec![Ident, Minus, Int, Arrow, Ident, Semi]
    );
}

#[test]
fn s5_row8_interior_space_adjacency_fails() {
    // `-true ->` after `{` → Minus KwTrue Arrow.
    assert_eq!(kinds("{ -true ->"), vec![LBrace, Minus, KwTrue, Arrow]);
}

#[test]
fn s5_row9_pattern_not_core_discriminant() {
    // `-Some(x)->` after `{` → Minus Ident LParen Ident RParen Arrow.
    assert_eq!(
        kinds("{ -Some(x)->"),
        vec![LBrace, Minus, Ident, LParen, Ident, RParen, Arrow]
    );
}

#[test]
fn s5_row10_float_discriminant_not_core() {
    // `-1.5->` after `{` → Minus Float Arrow.
    assert_eq!(kinds("{ -1.5->"), vec![LBrace, Minus, Float, Arrow]);
}

#[test]
fn s5_guard_at_start_of_input() {
    // Start-of-input context gate passes.
    assert_eq!(
        kinds("-true-> x;"),
        vec![Guard(GuardKind::True), Ident, Semi]
    );
}

#[test]
fn s5_guard_int_value_carried() {
    let out = lex("{ -123-> f;");
    let g = out.tokens.iter().find_map(|t| match t.kind {
        Guard(k) => Some(k),
        _ => None,
    });
    assert_eq!(g, Some(GuardKind::Int(123)));
}

// ---- §6 wart ledger rows W1–W9 ---------------------------------------

#[test]
fn w1_tight_negation_after_brace() {
    assert_eq!(
        kinds("{ -7->x;"),
        vec![LBrace, Guard(GuardKind::Int(7)), Ident, Semi]
    );
}

#[test]
fn w1_tight_negation_after_semi() {
    assert_eq!(
        kinds("a; -7->x;"),
        vec![Ident, Semi, Guard(GuardKind::Int(7)), Ident, Semi]
    );
}

#[test]
fn w1_tight_negation_after_rbrace() {
    // The `}` context specifically called out in W1.
    assert_eq!(
        kinds("{ } -7->x;"),
        vec![LBrace, RBrace, Guard(GuardKind::Int(7)), Ident, Semi]
    );
}

#[test]
fn w2_backarrow_maximal_munch() {
    // `i<-1` → Ident BackArrow Int.
    assert_eq!(kinds("i<-1"), vec![Ident, BackArrow, Int]);
}

#[test]
fn w3_ret_dot_zero_not_float() {
    assert_eq!(kinds("ret.0"), vec![KwRet, Dot, Int]);
}

#[test]
fn w3_five_dot_is_int_dot() {
    // `5.` → Int Dot (parser error later).
    assert_eq!(kinds("5."), vec![Int, Dot]);
}

#[test]
fn w4_double_dot_two_dots() {
    assert_eq!(kinds(".."), vec![Dot, Dot]);
}

#[test]
fn w4_range_in_expr() {
    // `0..255` (only appears in comments normally) → Int Dot Dot Int.
    assert_eq!(kinds("0..255"), vec![Int, Dot, Dot, Int]);
}

#[test]
fn w5_unicode_outside_strings_is_error() {
    // A lone non-ASCII char → one L0001 Error token.
    assert_eq!(kinds("∘"), vec![Error]);
    assert_eq!(codes("∘"), vec!["L0001"]);
}

#[test]
fn w5_unicode_in_comment_ok() {
    // `∘`, `×`, `—` in a comment are trivia.
    let out = lex("// ∘ × —\nx");
    assert!(out.diagnostics.is_empty());
    let ks: Vec<_> = out.tokens.iter().map(|t| t.kind).collect();
    assert_eq!(ks, vec![Ident, Eof]);
}

#[test]
fn w5_unicode_in_string_ok() {
    let out = lex("\"∘ × —\"");
    assert!(out.diagnostics.is_empty());
    assert_eq!(out.tokens[0].kind, Str);
}

#[test]
fn w6_crlf_is_trivia() {
    let out = lex("a\r\nb");
    assert!(out.diagnostics.is_empty());
    let ks: Vec<_> = out.tokens.iter().map(|t| t.kind).collect();
    assert_eq!(ks, vec![Ident, Ident, Eof]);
}

#[test]
fn w7_two_hundred_mhz() {
    // `200MHz` → Int Ident (maximal munch stops at the letter).
    let out = lex("200MHz");
    let ks: Vec<_> = out.tokens.iter().take(2).map(|t| t.kind).collect();
    assert_eq!(ks, vec![Int, Ident]);
    assert_eq!(lexeme("200MHz", &out.tokens[0]), "200");
    assert_eq!(lexeme("200MHz", &out.tokens[1]), "MHz");
}

#[test]
fn w7_underscored_and_hex_and_exp() {
    // `1_000` → Int(1) Ident(_000); `0xFF` → Int(0) Ident(xFF);
    // `1e10` → Int(1) Ident(e10).
    assert_eq!(kinds("1_000"), vec![Int, Ident]);
    assert_eq!(kinds("0xFF"), vec![Int, Ident]);
    assert_eq!(kinds("1e10"), vec![Int, Ident]);
}

#[test]
fn w8_coalesced_unknown_run() {
    // Adjacent unknown chars → one Error token + one L0001.
    let out = lex("$$$");
    let errs: Vec<_> = out.tokens.iter().filter(|t| t.kind == Error).collect();
    assert_eq!(errs.len(), 1);
    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code.0, "L0001");
    assert_eq!(lexeme("$$$", errs[0]), "$$$");
}

#[test]
fn w9_empty_input() {
    let out = lex("");
    assert_eq!(out.tokens.len(), 1);
    assert_eq!(out.tokens[0].kind, Eof);
    assert!(out.tokens[0].span.is_empty());
    assert!(out.diagnostics.is_empty());
}

// ---- operator munch boundaries ---------------------------------------

#[test]
fn munch_lt_variants() {
    assert_eq!(kinds("<"), vec![Lt]);
    assert_eq!(kinds("<="), vec![Le]);
    assert_eq!(kinds("<-"), vec![BackArrow]);
}

#[test]
fn munch_gt_variants() {
    assert_eq!(kinds(">"), vec![Gt]);
    assert_eq!(kinds(">="), vec![Ge]);
}

#[test]
fn munch_eq_bang_amp_pipe() {
    assert_eq!(kinds("=="), vec![EqEq]);
    assert_eq!(kinds("!="), vec![BangEq]);
    assert_eq!(kinds("!"), vec![Bang]);
    assert_eq!(kinds("&&"), vec![AmpAmp]);
    assert_eq!(kinds("||"), vec![PipePipe]);
}

#[test]
fn munch_arrow_vs_minus() {
    // Not in guard context (after Ident).
    assert_eq!(kinds("x->y"), vec![Ident, Arrow, Ident]);
    assert_eq!(kinds("x-y"), vec![Ident, Minus, Ident]);
}

#[test]
fn munch_dotdotdot() {
    assert_eq!(kinds("..."), vec![DotDotDot]);
    assert_eq!(
        kinds("[a, ...b]"),
        vec![LBracket, Ident, Comma, DotDotDot, Ident, RBracket]
    );
}

#[test]
fn munch_colon_colon() {
    // `::` → two Colon (out-of-Core path syntax). `map` is a hard keyword,
    // so `List::map` lexes the tail as KwMap (resolved by the parser).
    assert_eq!(kinds("List::map"), vec![Ident, Colon, Colon, KwMap]);
    assert_eq!(kinds("Foo::bar"), vec![Ident, Colon, Colon, Ident]);
}

#[test]
fn munch_slash_vs_comments() {
    // `/` alone is Slash; `//` is a line comment; `/*` is a block comment.
    assert_eq!(kinds("a / b"), vec![Ident, Slash, Ident]);
    let out = lex("a // comment\nb");
    let ks: Vec<_> = out.tokens.iter().map(|t| t.kind).collect();
    assert_eq!(ks, vec![Ident, Ident, Eof]);
    assert!(out.diagnostics.is_empty());
}

// ---- strings & escapes ----------------------------------------------

#[test]
fn string_basic() {
    let out = lex("\"hello\"");
    assert_eq!(out.tokens[0].kind, Str);
    assert!(out.diagnostics.is_empty());
    assert_eq!(lexeme("\"hello\"", &out.tokens[0]), "\"hello\"");
}

#[test]
fn string_known_escapes() {
    let out = lex("\"a\\n\\t\\\\\\\"b\"");
    assert!(out.diagnostics.is_empty());
    assert_eq!(out.tokens[0].kind, Str);
}

#[test]
fn string_unknown_escape_l0003() {
    let out = lex("\"a\\qb\"");
    assert_eq!(out.tokens[0].kind, Str);
    assert_eq!(codes("\"a\\qb\""), vec!["L0003"]);
}

#[test]
fn string_unterminated_eof_l0002() {
    let out = lex("\"abc");
    assert_eq!(out.tokens[0].kind, Str);
    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code.0, "L0002");
    // Token spans to EOF.
    assert_eq!(out.tokens[0].span.end as usize, "\"abc".len());
}

#[test]
fn string_unterminated_newline_l0002() {
    let out = lex("\"abc\nrest");
    assert_eq!(out.tokens[0].kind, Str);
    assert_eq!(out.diagnostics[0].code.0, "L0002");
    // Token ends at the newline; `rest` lexes as Ident.
    assert_eq!(lexeme("\"abc\nrest", &out.tokens[0]), "\"abc");
    assert_eq!(out.tokens[1].kind, Ident);
}

#[test]
fn unescape_helper() {
    use crate::token::unescape_string;
    assert_eq!(unescape_string("\"a\\nb\""), "a\nb");
    assert_eq!(unescape_string("\"\\t\\\\\\\"\""), "\t\\\"");
    // Unknown escape taken literally.
    assert_eq!(unescape_string("\"\\q\""), "q");
}

// ---- floats / ints ---------------------------------------------------

#[test]
fn float_basic() {
    assert_eq!(kinds("3.14"), vec![Float]);
    assert_eq!(kinds("0.0"), vec![Float]);
}

#[test]
fn int_basic() {
    assert_eq!(kinds("0"), vec![Int]);
    assert_eq!(kinds("255"), vec![Int]);
    assert_eq!(kinds("007"), vec![Int]); // leading zeros lex as written.
}

#[test]
fn float_needs_digit_after_dot() {
    // `5.` → Int Dot; `5.x` → Int Dot Ident.
    assert_eq!(kinds("5.x"), vec![Int, Dot, Ident]);
}

// ---- L-codes ---------------------------------------------------------

#[test]
fn l0004_category_with_fix() {
    let out = lex("category Foo");
    assert_eq!(out.tokens[0].kind, KwCategory);
    let d = &out.diagnostics[0];
    assert_eq!(d.code.0, "L0004");
    let fix = d.fix.as_ref().expect("fix present");
    assert_eq!(fix.replacement, "type");
}

#[test]
fn l0005_lone_eq() {
    let out = lex("=");
    assert_eq!(out.tokens[0].kind, Error);
    assert_eq!(out.diagnostics[0].code.0, "L0005");
}

#[test]
fn l0006_lone_amp_pipe() {
    assert_eq!(codes("&"), vec!["L0006"]);
    assert_eq!(codes("|"), vec!["L0006"]);
}

#[test]
fn l0007_block_comment_closed() {
    let out = lex("a /* skip */ b");
    let ks: Vec<_> = out.tokens.iter().map(|t| t.kind).collect();
    assert_eq!(ks, vec![Ident, Ident, Eof]);
    assert_eq!(out.diagnostics[0].code.0, "L0007");
}

#[test]
fn l0007_block_comment_unterminated() {
    let out = lex("a /* skip");
    let ks: Vec<_> = out.tokens.iter().map(|t| t.kind).collect();
    assert_eq!(ks, vec![Ident, Eof]);
    assert_eq!(out.diagnostics[0].code.0, "L0007");
}

#[test]
fn l0008_int_overflow_clamps() {
    // Way over u64::MAX.
    let src = "99999999999999999999999999";
    let out = lex(src);
    assert_eq!(out.tokens[0].kind, Int);
    assert_eq!(out.diagnostics[0].code.0, "L0008");
}

#[test]
fn l0008_guard_discriminant_overflow_no_panic() {
    // Over-u64 guard discriminant clamps to u64::MAX, no panic (I1), and is
    // flagged with L0008 (DESIGN §2/§5).
    let src = "{ -99999999999999999999-> f;";
    let out = lex(src);
    let g = out.tokens.iter().find_map(|t| match t.kind {
        Guard(k) => Some(k),
        _ => None,
    });
    assert_eq!(g, Some(GuardKind::Int(u64::MAX)));
    assert_eq!(codes(src), vec!["L0008"]);
}

#[test]
fn guard_max_u64_exact_no_diagnostic() {
    // The exact u64::MAX value is valid — clamped representation, but NOT
    // flagged (it did not overflow).
    let src = "{ -18446744073709551615-> f;";
    let out = lex(src);
    let g = out.tokens.iter().find_map(|t| match t.kind {
        Guard(k) => Some(k),
        _ => None,
    });
    assert_eq!(g, Some(GuardKind::Int(u64::MAX)));
    assert!(out.diagnostics.is_empty(), "exact u64::MAX must not flag");
}

// ---- invariants ------------------------------------------------------

#[test]
fn error_tokens_overlap_a_diagnostic() {
    // I4: every Error token overlaps >= 1 diagnostic span.
    let out = lex("= & | $$$ \x07");
    for t in out.tokens.iter().filter(|t| t.kind == Error) {
        let overlaps = out
            .diagnostics
            .iter()
            .any(|d| d.span.start < t.span.end && t.span.start < d.span.end || (d.span == t.span));
        assert!(overlaps, "Error token {t:?} has no overlapping diagnostic");
    }
}

/// Re-scan `source[start..end)` and assert it is trivia only — whitespace,
/// `//` line comments, and `/* */` block comments — mirroring `skip_trivia`.
/// This is the in-crate twin of the test-support coverage helper; it makes
/// the unit test below genuinely enforce I3 (not defer it elsewhere).
fn gap_is_trivia(src: &str, start: usize, end: usize) -> bool {
    let b = src.as_bytes();
    let mut i = start;
    while i < end {
        match b[i] {
            b' ' | b'\t' | b'\r' | b'\n' => i += 1,
            b'/' if b.get(i + 1) == Some(&b'/') => {
                i += 2;
                while i < end && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                i += 2;
                while i < end {
                    if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                        i += 2;
                        break;
                    }
                    i += 1;
                }
            }
            _ => return false,
        }
    }
    true
}

#[test]
fn spans_ascending_and_covering() {
    // I2 + I3 on a representative (guard-heavy, commented) source. Every
    // inter-token gap is re-scanned and must be whitespace/comments only, so
    // a dropped non-trivia byte would fail here too — not just in proptest.
    let src = "fn f(x: i32) -> i32 { // hdr\n (x > 0) -> { -true-> x; -false-> x * -1; } -> ret; }";
    let out = lex(src);
    let mut prev_end = 0u32;
    let mut covered = 0usize;
    for t in &out.tokens {
        assert!(t.span.start >= prev_end, "non-ascending: {t:?}");
        assert!(t.span.end <= src.len() as u32);
        // I3: the gap before this token is trivia only.
        assert!(
            gap_is_trivia(src, prev_end as usize, t.span.start as usize),
            "non-trivia gap in [{prev_end}..{}): a byte was dropped",
            t.span.start
        );
        prev_end = t.span.end;
        covered += (t.span.end - t.span.start) as usize;
    }
    // Eof is zero-width at end of input, so prev_end now == src.len(): every
    // byte is accounted for as a token span or a (verified-trivia) gap.
    assert_eq!(prev_end as usize, src.len(), "Eof must reach end of input");
    assert!(covered <= src.len());
}

// ---- LineIndex -------------------------------------------------------

#[test]
fn line_index_basic() {
    let src = "ab\ncde\nf";
    let idx = LineIndex::new(src);
    assert_eq!(idx.line_col(0), crate::loc::LineCol { line: 1, col: 1 });
    assert_eq!(idx.line_col(1), crate::loc::LineCol { line: 1, col: 2 });
    // Offset 3 is start of line 2 ('c').
    assert_eq!(idx.line_col(3), crate::loc::LineCol { line: 2, col: 1 });
    assert_eq!(idx.line_col(7), crate::loc::LineCol { line: 3, col: 1 });
}

#[test]
fn line_index_multibyte() {
    // '∘' is 3 bytes; columns must count chars, not bytes.
    let src = "x∘y\nz";
    let idx = LineIndex::new(src);
    // byte 0 = 'x' col 1; byte 1 = start of '∘' col 2; byte 4 = 'y' col 3.
    assert_eq!(idx.line_col(0), crate::loc::LineCol { line: 1, col: 1 });
    assert_eq!(idx.line_col(1), crate::loc::LineCol { line: 1, col: 2 });
    assert_eq!(idx.line_col(4), crate::loc::LineCol { line: 1, col: 3 });
    // After the '\n' (byte 5 is '\n', line 2 starts at byte 6).
    assert_eq!(idx.line_col(6), crate::loc::LineCol { line: 2, col: 1 });
}

#[test]
fn line_index_clamps_past_end() {
    let src = "ab";
    let idx = LineIndex::new(src);
    assert_eq!(idx.line_col(999), crate::loc::LineCol { line: 1, col: 3 });
}

// ---- examples sanity (zero diagnostics) ------------------------------

#[test]
fn examples_zero_diagnostics() {
    for name in ["abs", "fanout", "fir", "pipeline", "sepia", "sum_to_n"] {
        let path = format!("{}/../../examples/{name}.mapal", env!("CARGO_MANIFEST_DIR"));
        let src = std::fs::read_to_string(&path).expect("read example");
        let out = lex(&src);
        assert!(
            out.diagnostics.is_empty(),
            "{name}.mapal produced diagnostics: {:?}",
            out.diagnostics
        );
        assert!(
            !out.tokens.iter().any(|t| t.kind == Error),
            "{name}.mapal produced Error tokens"
        );
        assert_eq!(out.tokens.last().map(|t| t.kind), Some(Eof));
    }
}
