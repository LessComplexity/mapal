//! Parser tests: sigiled labels, termination rule, trailing commas, `ret` targets.

use super::*;

// ======================================================================
// ADR-0012 sigiled labeled blocks (`:label { … }`) and jumps (`-> :label`)
// ======================================================================

#[test]
fn sigiled_labeled_block_p0110() {
    // `:search { x -> y; }` statement ⇒ P0110, parsed as a custom-labeled loop.
    let src = "fn m() { :search { x -> y; } }";
    assert_eq!(pcodes(src), vec!["P0110"]);
    let f = parse_one_fn(src);
    match &f.body.items[0] {
        BlockItem::Stmt(Stmt {
            kind:
                StmtKind::Loop(LoopStmt {
                    label: LoopLabel::Custom(name),
                    span,
                    ..
                }),
            ..
        }) => {
            // Span convention: the `Custom` Name span is the bare identifier
            // (`search`), while the `LoopStmt` span begins at the `:` sigil.
            assert_eq!(
                &src[name.span.start as usize..name.span.end as usize],
                "search"
            );
            assert_eq!(&src[span.start as usize..span.start as usize + 1], ":");
        }
        other => panic!("expected a custom-labeled loop, got {other:?}"),
    }
}

#[test]
fn sigiled_labeled_block_diag_covers_sigil() {
    // The P0110 diagnostic span covers `:search` (sigil + ident).
    let src = "fn m() { :search { x -> y; } }";
    let out = parse(src);
    let d = out
        .diagnostics
        .iter()
        .find(|d| d.code.0 == "P0110")
        .expect("a P0110");
    assert_eq!(&src[d.span.start as usize..d.span.end as usize], ":search");
}

#[test]
fn labeled_jump_p0110_chain_terminates() {
    // `-> :search;` in a chain ⇒ one P0110; the `;` terminates the chain normally
    // and the surrounding fn body keeps parsing.
    let src = "fn m() { x -> :search; y -> ret; }";
    assert_eq!(pcodes(src), vec!["P0110"]);
    let f = parse_one_fn(src);
    // Two statements survived: the jump chain and the `y -> ret;` chain.
    assert_eq!(f.body.items.len(), 2, "chain terminated, both stmts parsed");
    // The jump stage is an Error stage; the `;` terminated the chain.
    let c = first_chain(&f);
    assert!(matches!(c.stages.last().unwrap().kind, StageKind::Error(_)));
}

#[test]
fn labeled_jump_diag_covers_sigil() {
    let src = "fn m() { x -> :search; }";
    let out = parse(src);
    let d = out
        .diagnostics
        .iter()
        .find(|d| d.code.0 == "P0110")
        .expect("a P0110");
    assert_eq!(&src[d.span.start as usize..d.span.end as usize], ":search");
}

#[test]
fn statement_initial_colon_non_ident_is_generic_error() {
    // A statement-initial `:` NOT followed by `Ident {` is NOT a label: keep the
    // pre-existing generic-error behavior (P0001), never P0110.
    let src = "fn m() { : 5 -> x; }";
    let codes = pcodes(src);
    assert!(
        !codes.contains(&"P0110"),
        "no P0110 for non-label `:`, got {codes:?}"
    );
    assert!(
        codes.contains(&"P0001"),
        "generic error expected, got {codes:?}"
    );
}

#[test]
fn path_colons_after_arrow_not_a_label() {
    // `x -> ::weird;` — a `::` path (P0109), NOT a labeled jump (no P0110). The
    // label form requires `Colon Ident`; a second `Colon` excludes it.
    let src = "fn m() { x -> ::weird; }";
    let codes = pcodes(src);
    assert!(
        !codes.contains(&"P0110"),
        "`::` is not a label, got {codes:?}"
    );
}

#[test]
fn nested_labeled_blocks_and_jump() {
    // `:outer { :inner { -> :inner; } }` — P0110 per labeled block (×2) plus
    // P0110 per labeled jump (×1) = three, and no panic. The headless `-> :inner;`
    // chain is the inner block's content.
    let src = "fn m() { :outer { :inner { -> :inner; } } }";
    assert_eq!(pcodes(src), vec!["P0110", "P0110", "P0110"]);
    assert_spans(src);
}

#[test]
fn unsigiled_jump_is_plain_var_stage_w25() {
    // W25: an un-sigiled `-> search;` is a plain name stage (variable flow), never
    // a jump — zero diagnostics.
    let src = "fn m() { x -> search; }";
    no_diags(src);
    let f = parse_one_fn(src);
    let c = first_chain(&f);
    match &c.stages.last().unwrap().kind {
        StageKind::Expr(Expr {
            kind: ExprKind::Var(_),
            ..
        }) => {}
        other => panic!("expected a plain Var stage, got {other:?}"),
    }
}

// ======================================================================
// termination-rule matrix (W10, W11): `;`, `}`, block-final, `};`
// ======================================================================

#[test]
fn term_semicolon_statement() {
    no_diags("fn m() { x -> y; }");
}

#[test]
fn term_block_tail_no_semi() {
    // A chain without `;` ending at `}` is the block's tail.
    let f = parse_one_fn("fn m() { x -> y }");
    assert!(f.body.tail.is_some());
    assert!(f.body.items.is_empty());
}

#[test]
fn term_block_final_stage_no_semi() {
    // fanout.mapal: `6 -> { … }` (block-final) followed by another statement
    // with no `;` after the brace.
    let src = "fn m() { 6 -> { -> a; -> b; } sq -> print; }";
    no_diags(src);
}

#[test]
fn term_block_final_with_semi_w10() {
    // `};` also legal (user-guide §5.2).
    let src = "fn m() { 6 -> { -> a; -> b; }; sq -> print; }";
    no_diags(src);
}

#[test]
fn term_missing_semicolon_p0001() {
    // `x -> y z` (two chains, no `;`) → P0001 expected `;`.
    let src = "fn m() { x -> y z -> w; }";
    assert!(pcodes(src).contains(&"P0001"));
}

#[test]
fn bare_expr_statement_p0003() {
    // A bare-expression chain as a `;`-terminated statement → P0003.
    assert_eq!(pcodes("fn m() { x; y -> ret; }"), vec!["P0003"]);
}

#[test]
fn bare_expr_tail_ok() {
    // The same bare expression as a block tail is legal.
    no_diags("fn m() { x }");
}

// ======================================================================
// trailing commas (DESIGN §14.6)
// ======================================================================

#[test]
fn trailing_comma_struct_decl_and_literal() {
    no_diags("type P { r: f32, g: f32, b: f32, }");
    no_diags("fn m() { P { r: a, g: b, } }");
}

#[test]
fn trailing_comma_array_and_tuple() {
    no_diags("fn m() { [1, 2, 3,] -> v; }");
    no_diags("fn m() { (a, b,) -> v; }");
}

#[test]
fn trailing_comma_params() {
    no_diags("fn f(a: i32, b: i32,) { a -> ret; }");
}

/// Extract the single MapFold stage from the first chain of a fn body.
fn first_map_fold(f: &FnDecl) -> (&[Name], &Block) {
    let c = first_chain(f);
    for st in &c.stages {
        if let StageKind::MapFold { params, body, .. } = &st.kind {
            return (params, body);
        }
    }
    panic!("expected a map/fold stage in the first chain");
}

#[test]
fn op_block_param_no_trailing_comma() {
    // op-block params: §14.3 grammar permits NO trailing comma before `->`. The
    // parser rejects it with P0001 but keeps the parsed param `x` and continues
    // recovery (the `->` and body still parse) — no panic, deterministic.
    let src = "fn m() { x -> map { x, -> a; } -> v; }";
    assert!(
        pcodes(src).contains(&"P0001"),
        "trailing-comma op-block param → P0001: {:?}",
        pcodes(src)
    );
    let f = parse_one_fn(src);
    let (params, _body) = first_map_fold(&f);
    assert_eq!(params.len(), 1, "the one param `x` is kept after recovery");
    let p = params[0].span;
    assert_eq!(&src[p.start as usize..p.end as usize], "x", "param is `x`");
}

#[test]
fn op_block_zero_params_p0001() {
    // §14.3: an op-block needs at least one parameter; `map { -> … }` has none →
    // P0001, recovered (the body still parses, MapFold kept with zero params).
    let src = "fn m() { x -> map { -> a; } -> v; }";
    assert!(
        pcodes(src).contains(&"P0001"),
        "zero-param op-block → P0001: {:?}",
        pcodes(src)
    );
    let f = parse_one_fn(src);
    let (params, _body) = first_map_fold(&f);
    assert!(params.is_empty(), "no params parsed");
}

// ======================================================================
// ret.0 targets, ret / loop
// ======================================================================

#[test]
fn ret_projection() {
    // `-> ret.0`.
    let f = parse_one_fn("fn m() { x -> ret.0; }");
    let c = first_chain(&f);
    match &c.stages[0].kind {
        StageKind::Ret { proj: Some((0, _)) } => {}
        other => panic!("expected ret.0, got {other:?}"),
    }
}

#[test]
fn ret_plain() {
    let f = parse_one_fn("fn m() { x -> ret; }");
    let c = first_chain(&f);
    assert!(matches!(c.stages[0].kind, StageKind::Ret { proj: None }));
}

#[test]
fn loop_jump_stage() {
    let f = parse_one_fn("fn m() { loop { -> loop; } }");
    match &f.body.items[0] {
        BlockItem::Stmt(Stmt {
            kind: StmtKind::Loop(l),
            ..
        }) => {
            let inner = &l.body.items[0];
            match inner {
                BlockItem::Stmt(Stmt {
                    kind: StmtKind::Chain(c),
                    ..
                }) => assert!(matches!(c.stages[0].kind, StageKind::LoopJump)),
                other => panic!("expected `-> loop;` chain, got {other:?}"),
            }
        }
        other => panic!("expected loop, got {other:?}"),
    }
}

#[test]
fn ret_as_chain_head_p0001_w20() {
    // `ret -> f;` → P0001 (ret is a stage form, not an expression).
    assert!(pcodes("fn m() { ret -> f; }").contains(&"P0001"));
}
