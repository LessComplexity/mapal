//! ADR-0051 D1/D2: format parameters with their kind in a declared list
//! (`fn f<n, cache: float>`, `type M<r, c, e: float>`), a stage's explicit
//! arguments (`x -> to<bf16>`, `-> project<4096, 2 * n>`) and their fallback
//! to an expression stage, a number format as a generic type's argument, the
//! parse errors, and the formatter's tight layout — idempotent, never
//! changing the parse.

use mapal_syntax::{
    BlockItem, Chain, ExprKind, FnDecl, Item, SizeKind, StageKind, StmtKind, TyKind, format_source,
    parse,
};

fn text(src: &str, s: mapal_syntax::SourceLoc) -> &str {
    &src[s.start as usize..s.end as usize]
}

fn one_fn(src: &str) -> FnDecl {
    let out = parse(src);
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    out.program
        .items
        .into_iter()
        .find_map(|i| match i {
            Item::Fn(fd) => Some(fd),
            _ => None,
        })
        .expect("a fn")
}

fn codes(src: &str) -> Vec<&'static str> {
    parse(src).diagnostics.iter().map(|d| d.code.0).collect()
}

/// The first statement chain of `fd`'s body.
fn first_chain(fd: &FnDecl) -> &Chain {
    match &fd.body.items[0] {
        BlockItem::Stmt(s) => match &s.kind {
            StmtKind::Chain(c) => c,
            other => panic!("not a chain: {other:?}"),
        },
        other => panic!("not a statement: {other:?}"),
    }
}

/// `(callee, [args as text])` of a stage with explicit arguments.
fn generic_stage<'s>(src: &'s str, kind: &StageKind) -> Option<(&'s str, Vec<&'s str>)> {
    match kind {
        StageKind::Expr(e) => match &e.kind {
            ExprKind::Generic { name, args } => Some((
                text(src, name.span),
                args.iter().map(|a| text(src, a.span)).collect(),
            )),
            _ => None,
        },
        _ => None,
    }
}

#[test]
fn declared_lists_hold_sizes_and_formats() {
    let src =
        "fn attend<positions, cache: float, w: narrow,>(keys: [cache; positions]) -> f32 { 1.0 }";
    let fd = one_fn(src);
    let entries: Vec<(&str, Option<&str>)> = fd
        .generics
        .iter()
        .map(|g| (text(src, g.name.span), g.kind.map(|k| text(src, k.span))))
        .collect();
    assert_eq!(
        entries,
        [
            ("positions", None),
            ("cache", Some("float")),
            ("w", Some("narrow"))
        ]
    );
    assert!(fd.generics[0].is_size() && !fd.generics[1].is_size());
    assert_eq!(text(src, fd.generics[1].span), "cache: float");
    // A type declaration takes the same list; a field's element is a name.
    let out = parse("type Matrix<rows, columns, element: float> { _: [element; rows * columns] }");
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    let Item::Type(td) = &out.program.items[0] else {
        panic!("not a type");
    };
    assert_eq!(td.generics.len(), 3);
    assert!(td.generics[2].kind.is_some());
}

#[test]
fn a_kind_is_a_name_after_the_colon() {
    assert!(codes("fn f<c:>(x: f32) -> f32 { x -> ret; }").contains(&"P0001"));
    assert!(codes("fn f<c: 4>(x: f32) -> f32 { x -> ret; }").contains(&"P0001"));
    // Any name parses; lowering checks the kind (L1711).
    assert!(codes("fn f<c: real>(x: f32) -> f32 { x -> ret; }").is_empty());
}

#[test]
fn explicit_arguments_at_a_stage() {
    let src = "fn main() { x -> to<bf16> -> project<4096, 2 * n,> -> widen<f32, bf16> -> y; }";
    let fd = one_fn(src);
    let stages: Vec<_> = first_chain(&fd)
        .stages
        .iter()
        .filter_map(|s| generic_stage(src, &s.kind))
        .collect();
    assert_eq!(
        stages,
        [
            ("to", vec!["bf16"]),
            ("project", vec!["4096", "2 * n"]),
            ("widen", vec!["f32", "bf16"]),
        ]
    );
    // The explicit arguments are size expressions (a format is a name).
    let ExprKind::Generic { args, .. } = (match &first_chain(&fd).stages[1].kind {
        StageKind::Expr(e) => &e.kind,
        _ => unreachable!(),
    }) else {
        panic!("not generic")
    };
    assert!(matches!(args[1].kind, SizeKind::Binary { .. }));
    // At the end of a chain, a body tail, a guard arm, a fanout branch.
    for src in [
        "fn main() { x -> map { v -> v -> to<bf16> } -> y; }",
        "fn main() { (c) -> { -true-> v -> to<bf16>; -false-> w; } -> y; }",
        "fn main() { x -> { -> to<bf16>; -> widen<f64>; } -> y; }",
        "fn main() { x -> to<bf16>; }",
    ] {
        assert!(codes(src).is_empty(), "{src}: {:?}", codes(src));
        assert!(src.contains("to<bf16>"));
    }
}

#[test]
fn a_comparison_is_no_argument_list() {
    // `<` that does not close where the stage ends falls back to the
    // expression stage (which lowering rejects: a comparison is no stage).
    for src in [
        "fn main() { a -> b < c -> y; }",
        "fn main() { a -> b < c; }",
        "fn main() { a -> b < 2 + c > d; }",
    ] {
        let out = parse(src);
        let fd = match &out.program.items[0] {
            Item::Fn(fd) => fd,
            _ => panic!(),
        };
        assert!(
            first_chain(fd)
                .stages
                .iter()
                .all(|s| generic_stage(src, &s.kind).is_none()),
            "{src}"
        );
    }
    // In a head (not stage position) `<` is a comparison.
    let src = "fn main() { a < b -> c; }";
    let fd = one_fn(src);
    assert!(matches!(
        first_chain(&fd).head.as_ref().map(|h| &h.kind),
        Some(ExprKind::Binary { .. })
    ));
}

#[test]
fn a_number_format_is_a_types_argument() {
    let src = "fn f(m: Matrix<64, n, bf16>, p: Pair<f16>) -> f32 { 1.0 }";
    let fd = one_fn(src);
    for p in &fd.params {
        let TyKind::Generic { args, .. } = &p.ty.kind else {
            panic!("not generic: {:?}", p.ty.kind)
        };
        let last = args.last().expect("an argument");
        assert!(matches!(last.kind, SizeKind::Name(_)));
    }
}

fn fmt(src: &str) -> String {
    format_source(src).unwrap_or_else(|d| panic!("{d:?}"))
}

#[test]
fn the_formatter_writes_the_lists_tight() {
    let src = "type Matrix < rows , columns , element : float > { _ : [ element ; rows * columns ] }
fn store < n , cache : float > ( x : [ f32 ; n ] ) -> [ cache ; n ] { x -> map { v -> v -> to < cache > } -> ret ; }
fn main ( ) { ( 0.5 , 4 ) -> fill -> xs ; xs -> store < 4 , bf16 > -> y ; y -> project < 2 * 2 , 4 > -> z : Matrix < 2 , 2 , bf16 > ; }";
    let out = fmt(src);
    assert_eq!(
        out,
        "type Matrix<rows, columns, element: float> {
    _: [element; rows * columns]
}

fn store<n, cache: float>(x: [f32; n]) -> [cache; n] {
    x -> map {
        v -> v -> to<cache>
    } -> ret;
}

fn main() {
    (0.5, 4) -> fill -> xs;
    xs -> store<4, bf16> -> y;
    y -> project<2 * 2, 4> -> z: Matrix<2, 2, bf16>;
}
"
    );
    // Idempotent, and the parse is the same before and after.
    assert_eq!(fmt(&out), out);
    let strip = |s: &str| {
        let toks = mapal_syntax::lex(s).tokens;
        toks.iter()
            .map(|t| text(s, t.span).to_string())
            .collect::<Vec<_>>()
    };
    assert_eq!(strip(src), strip(&out));
}

#[test]
fn the_formatter_keeps_a_comparison_spaced() {
    // `-> a < b` is not an argument list (no `>` closes it): spaced as before.
    let out = fmt("fn main() { 1 -> a < 2 -> y; }");
    assert!(out.contains("-> a < 2 -> y"), "{out}");
}
