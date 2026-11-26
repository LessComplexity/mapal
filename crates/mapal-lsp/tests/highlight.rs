//! The shared highlighting fixture (`editors/test/fixture.mapal`): every
//! token's class, as `editors/test/fixture.tokens.tsv` records it, and the
//! server's semantic tokens against it.
//!
//! Each row is `line col text class static`: `class` is what the token is
//! (the server's answer for names, the lexer's for everything else);
//! `static` is what a highlighter that only sees the line can know — the Vim
//! syntax and the TextMate grammar must both say exactly that (their tests
//! read this file), and `=` means the same as `class`. They differ only for
//! names whose meaning lives elsewhere in the file (a parameter's use, a call
//! to a function with no `->` after it, a top-level value read in a body).
//!
//! `MAPAL_BLESS=1 cargo test -p mapal-lsp --test highlight` rewrites the file;
//! review the diff — it is the specification the editors are tested against.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mapal_lsp::Analysis;
use mapal_lsp::analysis::Class;
use mapal_lsp::server::{TOKEN_MODIFIERS, TOKEN_TYPES, encode_tokens, token_of};
use mapal_lsp::text::LineMap;
use mapal_syntax::{GuardKind, Token, TokenKind};

fn editors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../editors")
}

#[derive(Clone, PartialEq, Eq, Debug)]
struct Row {
    line: u32,
    col: u32,
    text: String,
    class: String,
    stat: String,
}

impl Row {
    fn tsv(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}",
            self.line, self.col, self.text, self.class, self.stat
        )
    }
}

/// `class → (vim, textmate, lsp)` from `editors/test/classes.tsv`.
fn classes() -> BTreeMap<String, (String, String, String)> {
    let text = std::fs::read_to_string(editors().join("test/classes.tsv")).unwrap();
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("class\t") && !l.is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (
                f[0].to_string(),
                (f[1].to_string(), f[2].to_string(), f[3].to_string()),
            )
        })
        .collect()
}

const ALL_CLASSES: [Class; 16] = [
    Class::FunctionDeclaration,
    Class::Function,
    Class::Builtin,
    Class::TypeDeclaration,
    Class::Type,
    Class::Primitive,
    Class::SizeDeclaration,
    Class::Size,
    Class::Global,
    Class::Binding,
    Class::Rebind,
    Class::ParameterDeclaration,
    Class::Parameter,
    Class::FieldDeclaration,
    Class::Field,
    Class::Variable,
];

fn lsp_name(class: Class) -> String {
    match token_of(class) {
        None => "-".into(),
        Some((t, None)) => t.into(),
        Some((t, Some(m))) => format!("{t}.{m}"),
    }
}

/// What a highlighter that sees one line at a time can know about the name
/// `toks[i]` — the rules the Vim syntax and the TextMate grammar implement.
fn static_name(
    src: &str,
    toks: &[Token],
    i: usize,
    lines: &LineMap,
    in_size: bool,
) -> &'static str {
    let text = |t: &Token| &src[t.span.start as usize..t.span.end as usize];
    let line = |t: &Token| lines.position(src, t.span.start).line;
    let w = text(&toks[i]);
    let here = line(&toks[i]);
    let prev = |k: usize| {
        i.checked_sub(k)
            .map(|j| &toks[j])
            .filter(|t| line(t) == here)
    };
    let next = |k: usize| {
        toks.get(i + k)
            .filter(|t| line(t) == here && t.kind != TokenKind::Eof)
    };
    let kind = |t: Option<&Token>| t.map(|t| t.kind);
    // A removed conversion name (ADR-0051 D6) stays reserved: shown as the
    // builtin it was (its use is the compiler's L1713).
    if mapal_lower::is_builtin(w) || mapal_lower::removed_builtin(w).is_some() {
        return "builtin";
    }
    if mapal_lower::SCALAR_TYPES.contains(&w) || mapal_lower::PRECISION_MEMBERS.contains(&w) {
        return "primitive";
    }
    if w == "_" {
        return "default-field";
    }
    if kind(prev(1)) == Some(TokenKind::KwFn) {
        return "function-declaration";
    }
    // ADR-0051 D1: a kind word after the colon of a declared parameter.
    if in_size
        && matches!(w, "float" | "narrow" | "int" | "number")
        && kind(prev(1)) == Some(TokenKind::Colon)
    {
        return "primitive";
    }
    if in_size {
        return "size";
    }
    if w.starts_with(|c: char| c.is_ascii_uppercase()) {
        return "type";
    }
    if matches!(kind(prev(1)), Some(TokenKind::Guard(_))) {
        return "variable";
    }
    let after_arrow = kind(prev(1)) == Some(TokenKind::Arrow)
        || (kind(prev(1)) == Some(TokenKind::KwMut) && kind(prev(2)) == Some(TokenKind::Arrow));
    let colon = kind(next(1)) == Some(TokenKind::Colon) && kind(next(2)) != Some(TokenKind::Colon);
    if after_arrow && (kind(next(1)) == Some(TokenKind::Semi) || colon) {
        return "binding";
    }
    if after_arrow && kind(next(1)) == Some(TokenKind::Arrow) {
        return "function";
    }
    if kind(prev(1)) == Some(TokenKind::KwMut) || kind(next(1)) == Some(TokenKind::BackArrow) {
        return "binding";
    }
    "variable"
}

/// Every token of `src` with its class and its one-line class.
fn rows(src: &str) -> Vec<Row> {
    assert!(
        src.is_ascii(),
        "the fixture is ASCII so columns are bytes and characters alike"
    );
    let a = Analysis::new(src);
    let lines = LineMap::new(src);
    let named: BTreeMap<u32, Class> = a
        .classified()
        .into_iter()
        .map(|(s, c)| (s.start, c))
        .collect();
    let toks: Vec<Token> = mapal_syntax::lex(src)
        .tokens
        .into_iter()
        .filter(|t| t.kind != TokenKind::Eof)
        .collect();
    let mut out = Vec::new();
    let mut push = |offset: u32, text: &str, class: &str, stat: &str| {
        let p = lines.position(src, offset);
        out.push(Row {
            line: p.line + 1,
            col: p.character + 1,
            text: text.to_string(),
            class: class.to_string(),
            stat: if stat == class {
                "=".into()
            } else {
                stat.to_string()
            },
        });
    };
    // Size lists: `<` after `fn name` / `type Name`, after a PascalCase
    // name (a generic type's sizes) or after a stage name behind an arrow
    // (`-> to<bf16>`, ADR-0051 D2), to its `>`.
    let mut size_depth = 0usize;
    for (i, t) in toks.iter().enumerate() {
        let text = &src[t.span.start as usize..t.span.end as usize];
        let prev_text = |k: usize| {
            i.checked_sub(k)
                .map(|j| &src[toks[j].span.start as usize..toks[j].span.end as usize])
        };
        let opens_size = t.kind == TokenKind::Lt
            && i >= 1
            && toks[i - 1].kind == TokenKind::Ident
            && (prev_text(1).is_some_and(|p| p.starts_with(|c: char| c.is_ascii_uppercase()))
                || (i >= 2
                    && matches!(
                        toks[i - 2].kind,
                        TokenKind::KwFn | TokenKind::KwType | TokenKind::Arrow
                    )));
        match t.kind {
            TokenKind::Ident => {
                let class = named.get(&t.span.start).map_or(
                    if text == "_" {
                        "default-field"
                    } else {
                        "variable"
                    },
                    |c| c.name(),
                );
                push(
                    t.span.start,
                    text,
                    class,
                    static_name(src, &toks, i, &lines, size_depth > 0),
                );
            }
            TokenKind::Int => push(t.span.start, text, "number", "number"),
            TokenKind::Float => push(t.span.start, text, "float", "float"),
            TokenKind::Str => {
                push(t.span.start, "\"", "string", "string");
                let bytes = text.as_bytes();
                let mut k = 1;
                while k + 1 < bytes.len() {
                    if bytes[k] == b'\\' {
                        push(t.span.start + k as u32, &text[k..k + 2], "escape", "escape");
                        k += 2;
                    } else {
                        k += 1;
                    }
                }
            }
            TokenKind::KwRet => push(t.span.start, text, "ret", "ret"),
            TokenKind::KwTrue | TokenKind::KwFalse => {
                push(t.span.start, text, "boolean", "boolean")
            }
            TokenKind::KwCategory => push(t.span.start, text, "reserved", "reserved"),
            TokenKind::KwFn
            | TokenKind::KwType
            | TokenKind::KwMut
            | TokenKind::KwLoop
            | TokenKind::KwSeq
            | TokenKind::KwMap
            | TokenKind::KwFold
            | TokenKind::KwExecutor
            | TokenKind::KwPub
            | TokenKind::KwUse
            | TokenKind::KwVoid => push(t.span.start, text, "keyword", "keyword"),
            TokenKind::Arrow | TokenKind::BackArrow => push(t.span.start, text, "arrow", "arrow"),
            TokenKind::Guard(g) => {
                push(t.span.start, "-", "guard", "guard");
                let inner = &text[1..text.len() - 2];
                let class = match g {
                    GuardKind::True | GuardKind::False => "guard-bool",
                    GuardKind::Default => "guard-default",
                    GuardKind::Int(_) => "guard-int",
                };
                push(t.span.start + 1, inner, class, class);
                push(t.span.end - 2, "->", "guard", "guard");
            }
            TokenKind::Lt if opens_size => {
                size_depth += 1;
                push(t.span.start, text, "size-bracket", "size-bracket");
            }
            TokenKind::Gt if size_depth > 0 => {
                size_depth -= 1;
                push(t.span.start, text, "size-bracket", "size-bracket");
            }
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
            | TokenKind::Bang => push(t.span.start, text, "operator", "operator"),
            // `|` between `: compute` members (ADR-0041 D2) lexes as an error
            // token the parser accepts there.
            TokenKind::Error if text == "|" => push(t.span.start, text, "operator", "operator"),
            _ => push(t.span.start, text, "punctuation", "punctuation"),
        }
    }
    // Comments are trivia to the lexer: `//` outside every token.
    let mut offset = 0u32;
    for l in src.split_inclusive('\n') {
        if let Some(i) = l.find("//") {
            let at = offset + i as u32;
            if !toks.iter().any(|t| t.span.start <= at && at < t.span.end) {
                push(at, "//", "comment", "comment");
                for tag in ["TODO", "FIXME", "XXX", "NOTE"] {
                    for (j, _) in l[i..].match_indices(tag) {
                        push(at + j as u32, tag, "todo", "todo");
                    }
                }
            }
        }
        offset += l.len() as u32;
    }
    out.sort_by_key(|r| (r.line, r.col));
    out
}

fn fixture() -> String {
    std::fs::read_to_string(editors().join("test/fixture.mapal")).unwrap()
}

const HEADER: &str = "# line\tcol\ttext\tclass\tstatic — generated by `MAPAL_BLESS=1 cargo test -p mapal-lsp --test highlight` (crates/mapal-lsp/tests/highlight.rs), reviewed by hand. `=`: the static grammars give the class itself.";

#[test]
fn fixture_tokens_are_the_committed_classes() {
    let src = fixture();
    let got = rows(&src);
    let path = editors().join("test/fixture.tokens.tsv");
    if std::env::var("MAPAL_BLESS").is_ok_and(|v| v == "1") {
        let mut text = format!("{HEADER}\n");
        for r in &got {
            text.push_str(&r.tsv());
            text.push('\n');
        }
        std::fs::write(&path, text).unwrap();
        return;
    }
    let want: Vec<String> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(str::to_string)
        .collect();
    let got: Vec<String> = got.iter().map(Row::tsv).collect();
    let diffs: Vec<String> = want
        .iter()
        .zip(&got)
        .filter(|(w, g)| w != g)
        .map(|(w, g)| format!("  want {w}\n  got  {g}"))
        .take(20)
        .collect();
    assert!(
        diffs.is_empty() && want.len() == got.len(),
        "{} rows committed, {} now; first differences:\n{}",
        want.len(),
        got.len(),
        diffs.join("\n")
    );
}

#[test]
fn classes_table_matches_the_server() {
    let table = classes();
    for c in ALL_CLASSES {
        let row = table
            .get(c.name())
            .unwrap_or_else(|| panic!("class `{}` missing from classes.tsv", c.name()));
        assert_eq!(
            row.2,
            lsp_name(c),
            "classes.tsv lsp column for `{}`",
            c.name()
        );
    }
    // Every LSP token in the table is in the server's legend, and each names
    // one class (so an editor can map it back without ambiguity).
    let mut seen = BTreeMap::new();
    for (class, (_, _, lsp)) in &table {
        if lsp == "-" {
            continue;
        }
        let (ty, m) = lsp
            .split_once('.')
            .map_or((lsp.as_str(), None), |(t, m)| (t, Some(m)));
        assert!(TOKEN_TYPES.contains(&ty), "{lsp}");
        assert!(m.is_none_or(|m| TOKEN_MODIFIERS.contains(&m)), "{lsp}");
        assert!(
            seen.insert(lsp.clone(), class.clone()).is_none(),
            "{lsp} maps two classes"
        );
    }
}

#[test]
fn semantic_tokens_match_the_fixture() {
    let src = fixture();
    let table = classes();
    let a = Analysis::new(&src);
    let lines = LineMap::new(&src);
    // Decode the server's tokens to `(line, col) → "type[.modifier]"`.
    let mut tokens = BTreeMap::new();
    let (mut line, mut col) = (0u32, 0u32);
    for t in encode_tokens(&a, &lines) {
        if t.delta_line > 0 {
            col = 0;
        }
        line += t.delta_line;
        col += t.delta_start;
        let ty = TOKEN_TYPES[t.token_type as usize];
        let name = match t.token_modifiers_bitset {
            0 => ty.to_string(),
            bits => format!("{ty}.{}", TOKEN_MODIFIERS[bits.trailing_zeros() as usize]),
        };
        assert!(
            t.token_modifiers_bitset.count_ones() <= 1,
            "one modifier at most"
        );
        tokens.insert((line + 1, col + 1), name);
    }
    let path = editors().join("test/fixture.tokens.tsv");
    let mut failures = Vec::new();
    let mut checked = 0;
    for l in std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let f: Vec<&str> = l.split('\t').collect();
        let (ln, cl, text, class) = (
            f[0].parse::<u32>().unwrap(),
            f[1].parse::<u32>().unwrap(),
            f[2],
            f[3],
        );
        let want = &table
            .get(class)
            .unwrap_or_else(|| panic!("unknown class {class}"))
            .2;
        let got = tokens.remove(&(ln, cl)).unwrap_or_else(|| "-".into());
        checked += 1;
        if &got != want {
            failures.push(format!(
                "{ln}:{cl} `{text}` ({class}): want {want}, got {got}"
            ));
        }
    }
    assert!(
        tokens.is_empty(),
        "semantic tokens on no fixture token: {tokens:?}"
    );
    assert!(
        failures.is_empty(),
        "{} of {checked}:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The names both grammars hard-code are the language's.
#[test]
fn grammar_name_lists_are_the_languages() {
    // The builtins and the removed conversion names (ADR-0051 D6), reserved
    // and highlighted as the builtins they were.
    let mut builtins: Vec<&str> = mapal_lower::builtin_names();
    builtins.extend(mapal_lower::REMOVED_BUILTINS);
    builtins.sort_unstable();
    let mut prims: Vec<&str> = mapal_lower::SCALAR_TYPES.to_vec();
    prims.extend(mapal_lower::PRECISION_MEMBERS);
    prims.sort_unstable();
    prims.dedup();
    let mut keywords: Vec<&str> = mapal_syntax::KEYWORDS
        .iter()
        .copied()
        .filter(|k| !matches!(*k, "ret" | "true" | "false" | "category"))
        .collect();
    keywords.sort_unstable();

    let vim = std::fs::read_to_string(editors().join("nvim/syntax/mapal.vim")).unwrap();
    let vim_list = |group: &str| {
        let mut v: Vec<String> = vim
            .lines()
            .filter(|l| l.starts_with(&format!("syn keyword {group} ")))
            .flat_map(|l| {
                l.split_whitespace()
                    .skip(3)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .collect();
        v.sort_unstable();
        v
    };
    assert_eq!(vim_list("flowBuiltin"), builtins, "mapal.vim flowBuiltin");
    assert_eq!(vim_list("flowPrimType"), prims, "mapal.vim flowPrimType");
    assert_eq!(vim_list("flowKeyword"), keywords, "mapal.vim flowKeyword");

    let tm: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(editors().join("vscode/syntaxes/mapal.tmLanguage.json")).unwrap(),
    )
    .unwrap();
    let tm_list = |scope: &str| {
        let pats = tm["repository"]["keywords"]["patterns"].as_array().unwrap();
        let p = pats
            .iter()
            .find(|p| p["name"] == scope)
            .unwrap_or_else(|| panic!("{scope}"));
        let m = p["match"].as_str().unwrap();
        let inner = m.trim_start_matches("\\b(").trim_end_matches(")\\b");
        let mut v: Vec<String> = inner.split('|').map(str::to_string).collect();
        v.sort_unstable();
        v
    };
    assert_eq!(
        tm_list("support.function.builtin.mapal"),
        builtins,
        "TextMate builtins"
    );
    assert_eq!(
        tm_list("support.type.primitive.mapal"),
        prims,
        "TextMate primitives"
    );
    assert_eq!(
        tm_list("keyword.control.mapal"),
        keywords,
        "TextMate keywords"
    );
}
