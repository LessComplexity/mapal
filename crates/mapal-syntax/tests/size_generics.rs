//! ADR-0047 D1/D2: size parameter lists after a function's name, size
//! expressions in array lengths (`+`, `-`, `*`, literals, names, grouping),
//! the parse errors (P0001 empty list, P0002 unclosed `<`, P0119 `/` `%`), and
//! the formatter's tight `fn f<a, b>(` layout.

use mapal_syntax::{FnDecl, Item, SizeExpr, SizeKind, SizeOp, TyKind, format_source, lex, parse};

fn one_fn(src: &str) -> FnDecl {
    let out = parse(src);
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    match out.program.items.into_iter().next() {
        Some(Item::Fn(fd)) => fd,
        other => panic!("not a fn: {other:?}"),
    }
}

fn text(src: &str, s: mapal_syntax::SourceLoc) -> &str {
    &src[s.start as usize..s.end as usize]
}

/// The size expression as a fully parenthesized string: its parse tree.
fn shape(src: &str, e: &SizeExpr) -> String {
    match &e.kind {
        SizeKind::Int(n) => n.to_string(),
        SizeKind::Name(n) => text(src, n.span).to_string(),
        SizeKind::Binary { op, lhs, rhs } => {
            let o = match op {
                SizeOp::Add => "+",
                SizeOp::Sub => "-",
                SizeOp::Mul => "*",
            };
            format!("({} {o} {})", shape(src, lhs), shape(src, rhs))
        }
    }
}

fn size_of(ty: &mapal_syntax::Ty) -> (u64, Option<&SizeExpr>) {
    match &ty.kind {
        TyKind::Array { len, size, .. } => (*len, size.as_ref()),
        other => panic!("not an array: {other:?}"),
    }
}

#[test]
fn size_list_and_expressions_parse() {
    let src = "fn project<rows, columns>(weight: [bf16; rows * columns], x: [f32; columns]) \
               -> [f32; rows] { x[0] -> ret; }";
    let fd = one_fn(src);
    let names: Vec<&str> = fd.generics.iter().map(|g| text(src, g.name.span)).collect();
    assert_eq!(names, ["rows", "columns"]);
    let (len, e) = size_of(&fd.params[0].ty);
    assert_eq!(
        (len, shape(src, e.unwrap())),
        (0, "(rows * columns)".into())
    );
    assert_eq!(shape(src, size_of(&fd.params[1].ty).1.unwrap()), "columns");
    assert_eq!(
        shape(src, size_of(fd.ret_ty.as_ref().unwrap()).1.unwrap()),
        "rows"
    );
}

#[test]
fn precedence_grouping_and_literals() {
    let src = "fn f<a, b,>(x: [f32; a + b * 2 - 1], y: [[f32; (a + 1) * b]; 3], z: [f32; 64 * 64], \
               w: [f32; 4096]) -> f32 { x[0] -> ret; }";
    let fd = one_fn(src);
    assert_eq!(fd.generics.len(), 2, "a trailing comma is allowed");
    let s = |i: usize| shape(src, size_of(&fd.params[i].ty).1.unwrap());
    assert_eq!(s(0), "((a + (b * 2)) - 1)");
    // The outer dimension is a literal; the element's is grouped.
    let (len, outer) = size_of(&fd.params[1].ty);
    assert_eq!((len, outer), (3, None));
    let TyKind::Array { elem, .. } = &fd.params[1].ty.kind else {
        unreachable!()
    };
    assert_eq!(shape(src, size_of(elem).1.unwrap()), "((a + 1) * b)");
    // Literal arithmetic is a size expression too; one literal stays `len`.
    assert_eq!(s(2), "(64 * 64)");
    assert_eq!(size_of(&fd.params[3].ty), (4096, None));
    // Every size node's span lies inside its array type's (J2).
    let ty_span = fd.params[0].ty.span;
    let e = size_of(&fd.params[0].ty).1.unwrap();
    assert!(e.span.start >= ty_span.start && e.span.end <= ty_span.end);
}

#[test]
fn fixed_size_fns_have_no_sizes() {
    let fd = one_fn("fn f(x: [f32; 8]) -> f32 { x[0] -> ret; }");
    assert!(fd.generics.is_empty());
    assert_eq!(size_of(&fd.params[0].ty), (8, None));
}

fn codes(src: &str) -> Vec<&'static str> {
    parse(src).diagnostics.iter().map(|d| d.code.0).collect()
}

#[test]
fn division_and_modulo_are_not_sizes() {
    for src in [
        "fn f<n>(x: [f32; n / 2]) -> f32 { x[0] -> ret; }",
        "fn f<n>(x: [f32; n % 2]) -> f32 { x[0] -> ret; }",
    ] {
        assert_eq!(codes(src), ["P0119"], "{src}");
    }
}

#[test]
fn malformed_lists_and_lengths() {
    assert_eq!(
        codes("fn f<>(x: [f32; 4]) -> f32 { x[0] -> ret; }"),
        ["P0001"]
    );
    assert!(codes("fn f<n(x: [f32; n]) -> f32 { x[0] -> ret; }").contains(&"P0002"));
    assert!(codes("fn f<n>(x: [f32; ]) -> f32 { x[0] -> ret; }").contains(&"P0001"));
    assert!(codes("fn f<n>(x: [f32; (n + 1]) -> f32 { x[0] -> ret; }").contains(&"P0002"));
    // A type argument that is a type stays P0103 (out of Core); a number
    // format is a format argument (ADR-0051 D1), lowering resolves it.
    assert!(codes("fn f(x: Vec<Pixel>) { }").contains(&"P0103"));
    assert!(codes("fn f(x: Vec<bool>) { }").contains(&"P0103"));
    assert!(!codes("fn f(x: Vec<f32>) { }").contains(&"P0103"));
}

fn tokens(source: &str) -> Vec<(mapal_syntax::TokenKind, &str)> {
    lex(source)
        .tokens
        .into_iter()
        .map(|t| (t.kind, &source[t.span.start as usize..t.span.end as usize]))
        .collect()
}

#[test]
fn formatter_writes_the_list_tight() {
    let src = "fn  project < rows ,columns > ( weight : [ bf16 ; rows*columns ] , \
               x : [f32; (columns+1)-1] ) -> [f32;rows] { columns -> iota -> c; c -> ret; }";
    let out = format_source(src).unwrap();
    assert_eq!(
        out.lines().next().unwrap(),
        "fn project<rows, columns>(weight: [bf16; rows * columns], \
         x: [f32; (columns + 1) - 1]) -> [f32; rows] {"
    );
    assert_eq!(tokens(src), tokens(&out));
    assert_eq!(format_source(&out).unwrap(), out, "idempotent");
    // A comparison after a size list keeps its spaces.
    let cmp = format_source("fn f<n>(x: [f32; n]) -> bool { (n<2) -> ret; }").unwrap();
    assert!(cmp.contains("(n < 2) -> ret;"), "{cmp}");
}
