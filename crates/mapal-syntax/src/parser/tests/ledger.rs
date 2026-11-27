//! Parser tests: guard-payload matrix, section 17 ledger rows, stray guards.

use super::*;

// ======================================================================
// guard-payload F-matrix (expr / chain / block / headless / nested-guard)
// ======================================================================

#[test]
fn payload_bare_expr() {
    // `-true-> x;`
    no_diags("fn m() { (a) -> { -true-> x; -false-> y; } -> ret; }");
}

#[test]
fn payload_chain() {
    // `-false-> acc -> ret;` (sum_to_n / fir form).
    no_diags("fn m() { (a) -> { -true-> hi; -false-> acc -> ret; } }");
}

#[test]
fn payload_headless_chain() {
    // `-false-> -> ret;` (headless payload).
    no_diags("fn m() { (a) -> { -true-> hi; -false-> -> ret; } }");
}

#[test]
fn payload_block() {
    // `-true-> { … -> loop; }` mixes headed and headless statements legally.
    let src = "fn m() { loop { (a) -> { -true-> { x -> y; -> loop; } -false-> z -> ret; } } }";
    no_diags(src);
}

#[test]
fn payload_nested_guard() {
    // sepia: a `-false->` arm whose block payload contains another guard.
    let src = "fn m() { (a) -> { -true-> hi; -false-> { (b) -> { -true-> lo; -false-> v; } -> bounded; bounded } } -> ret; }";
    no_diags(src);
}

// ======================================================================
// §17 ledger rows
// ======================================================================

#[test]
fn w12_op_shorthand_minus_is_subtraction() {
    // `-> - 5` is subtraction shorthand: Binary(Sub, Hole, Int 5).
    let f = parse_one_fn("fn m() { x -> - 5; }");
    let c = first_chain(&f);
    match &c.stages[0].kind {
        StageKind::OpShorthand {
            expr:
                Expr {
                    kind:
                        ExprKind::Binary {
                            op: BinOp::Sub,
                            lhs,
                            ..
                        },
                    ..
                },
        } => assert!(matches!(lhs.kind, ExprKind::Hole)),
        other => panic!("expected `· - 5`, got {other:?}"),
    }
}

#[test]
fn w17_double_bind_p0008() {
    // `a <- b <- c` → P0008.
    assert!(pcodes("fn m() { x <- a <- b; }").contains(&"P0008"));
}

// --- ADR-0021 element-update sugar `c[i] <- x` --------------------------

/// The first block statement of the single fn, as a `BindStmt`, panicking
/// otherwise.
fn first_bind(src: &str) -> BindStmt {
    let f = parse_one_fn(src);
    match f.body.items.into_iter().next() {
        Some(BlockItem::Stmt(Stmt {
            kind: StmtKind::Bind(b),
            ..
        })) => b,
        other => panic!("expected first stmt to be a bind, got {other:?}"),
    }
}

#[test]
fn index_bind_top_level() {
    // `c[t] <- v;` at statement position: a clean indexed bind (ADR-0021).
    let src = "fn m() { c[t] <- v; }";
    no_diags(src);
    let b = first_bind(src);
    assert!(b.index.is_some(), "index target parsed");
    assert!(b.mut_span.is_none() && b.ty.is_none());
    assert_eq!(
        &src[b.name.span.start as usize..b.name.span.end as usize],
        "c"
    );
    // index is `t`, value is `v`.
    assert!(matches!(b.index.as_ref().unwrap().kind, ExprKind::Var(_)));
    assert!(matches!(b.value.kind, ExprKind::Var(_)));
}

#[test]
fn index_bind_in_loop_body() {
    // The motivating shape: `c[t] <- v;` inside a loop body. Clean parse.
    let src = "fn m() { loop { c[t] <- v; } }";
    no_diags(src);
}

#[test]
fn index_bind_computed_index() {
    // The index is a full expression (`a[i * 4 + k] <- x`), not just a name.
    let src = "fn m() { a[i * 4 + k] <- x; }";
    no_diags(src);
    let b = first_bind(src);
    assert!(matches!(
        b.index.as_ref().unwrap().kind,
        ExprKind::Binary { .. }
    ));
}

#[test]
fn index_bind_mut_rejected_p0013() {
    // `mut c[i] <- x` — the index form is a rebind, `mut` is meaningless: P0013.
    assert_eq!(pcodes("fn m() { mut c[i] <- x; }"), vec!["P0013"]);
}

#[test]
fn index_bind_type_annotation_rejected_p0014() {
    // `c[i]: T <- x` — an element-update target takes no type annotation: P0014.
    assert_eq!(pcodes("fn m() { c[i]: i32 <- x; }"), vec!["P0014"]);
}

#[test]
fn index_bind_nested_rejected_p0015() {
    // `c[i][j] <- x` — nested update is out of this increment: P0015. The
    // recovered node is a clean one-dimensional indexed bind.
    let src = "fn m() { c[i][j] <- x; }";
    assert_eq!(pcodes(src), vec!["P0015"]);
    let b = first_bind(src);
    assert!(b.index.is_some(), "outer index kept for recovery");
}

#[test]
fn index_chain_target_unaffected() {
    // The chain form `c[0] -> println;` is NOT a bind (no `<-`): untouched by
    // the sugar. `looks_like_bind` returns false (Arrow at depth 0), so this
    // stays a chain statement with an Index head. Regression guard.
    let src = "fn m() { c[0] -> println; }";
    no_diags(src);
    let f = parse_one_fn(src);
    assert!(matches!(
        f.body.items[0],
        BlockItem::Stmt(Stmt {
            kind: StmtKind::Chain(_),
            ..
        })
    ));
}

#[test]
fn w18_mixed_direction_p0001() {
    // `a -> b <- c` → P0001.
    assert!(pcodes("fn m() { a -> b <- c; }").contains(&"P0001"));
}

#[test]
fn w20_loop_in_expr_p0001() {
    // `loop + 1` → P0001 (loop is a stage form, not an expression).
    assert!(pcodes("fn m() { x -> y; loop + 1 -> z; }").contains(&"P0001"));
}

#[test]
fn w21_named_param_partial_is_member() {
    // `15 -> add.a;` parses as a member-expression stage (no special case).
    let f = parse_one_fn("fn m() { 15 -> add.a; }");
    let c = first_chain(&f);
    assert!(matches!(
        c.stages[0].kind,
        StageKind::Expr(Expr {
            kind: ExprKind::Member { .. },
            ..
        })
    ));
}

#[test]
fn w24_all_spaced_guard_block() {
    // Pinned regression: a block written entirely with spaced arms still
    // classifies as a GuardBlock; P0005 per arm, no P0006.
    let src = "fn m() { (a) -> { - true -> x; - false -> y; } -> ret; }";
    let pc = pcodes(src);
    assert_eq!(pc.iter().filter(|c| **c == "P0005").count(), 2, "P0005 ×2");
    assert!(!pc.contains(&"P0006"), "no mixing diagnostic: {pc:?}");
    let f = parse_one_fn(src);
    let c = first_chain(&f);
    assert!(
        matches!(c.stages[0].kind, StageKind::Guard(_)),
        "GuardBlock"
    );
}

// ======================================================================
// stray guard W1, spaced guard, mixing
// ======================================================================

#[test]
fn stray_guard_p0004() {
    // `-7-> x;` at statement-initial position (after `{`), no guard block → a
    // stray guard. P0004 with a source-sliced fix.
    let src = "fn m() { -7-> x; }";
    let out = parse(src);
    let pc: Vec<_> = out
        .diagnostics
        .iter()
        .filter(|d| d.code.0 == "P0004")
        .collect();
    assert_eq!(pc.len(), 1, "exactly one P0004");
    let fix = pc[0].fix.as_ref().expect("P0004 has a fix");
    // The fix slices the source lexeme minus the trailing `->`, plus ` ->`.
    assert_eq!(fix.replacement, "-7 ->");
}

#[test]
fn stray_guard_over_u64_slices_source() {
    // Over-u64 stray guard: the fix must slice the source lexeme verbatim, never
    // re-render from the clamped GuardKind::Int (which would rewrite the literal).
    let src = "fn m() { -99999999999999999999-> x; }";
    let out = parse(src);
    let d = out
        .diagnostics
        .iter()
        .find(|d| d.code.0 == "P0004")
        .expect("P0004");
    let fix = d.fix.as_ref().expect("fix");
    assert_eq!(fix.replacement, "-99999999999999999999 ->");
    // The lexer's L0008 is preserved and NOT duplicated (C14/J6).
    assert_eq!(codes(src).iter().filter(|c| **c == "L0008").count(), 1);
}

#[test]
fn spaced_guard_p0005_recovered_as_arm() {
    // `- true ->` inside a guard block → P0005, recovered as the meant arm.
    let src = "fn m() { (a) -> { - true -> x; -false-> y; } -> ret; }";
    assert!(pcodes(src).contains(&"P0005"));
}

#[test]
fn mixing_statement_among_arms_p0006() {
    // Pinned regression: `- 7 -> x;` (a statement) among arms → statement + P0006
    // with the bidirectional hint.
    let src = "fn m() { (a) -> { -true-> x; - 7 -> y; } -> ret; }";
    let pc = pcodes(src);
    assert!(pc.contains(&"P0006"), "P0006 present: {pc:?}");
    let out = parse(src);
    let d = out
        .diagnostics
        .iter()
        .find(|d| d.code.0 == "P0006")
        .unwrap();
    assert!(
        d.message.contains("-7->"),
        "bidirectional hint mentions `-7->`: {}",
        d.message
    );
}

#[test]
fn guard_block_with_tail_retains_the_tail_b2() {
    // B2 negative control (2026-08-01 review): `999` must survive in the AST
    // and the block must NOT classify as a clean two-arm guard. The old
    // recovery filtered to arms; the tail vanished with no diagnostic at all
    // (tails are not items, so check_arm_mixing never saw one).
    let src = "fn m() { true -> { -true-> 1; -false-> 2; 999 } -> ret; }";
    let pc = pcodes(src);
    assert!(pc.contains(&"P0006"), "tail mixing named: {pc:?}");
    assert!(
        pc.contains(&"P0115"),
        "StmtBlock marked out of Core: {pc:?}"
    );
    let f = parse_one_fn(src);
    let c = first_chain(&f);
    match &c.stages[0].kind {
        StageKind::StmtBlock(b) => {
            assert_eq!(
                b.items
                    .iter()
                    .filter(|i| matches!(i, BlockItem::Arm(_)))
                    .count(),
                2,
                "both arms retained: {:?}",
                b.items
            );
            assert!(b.tail.is_some(), "the `999` tail is retained, not deleted");
        }
        other => panic!("arms + tail must classify as StmtBlock, got {other:?}"),
    }
}

#[test]
fn guard_block_mixed_statement_retained_as_stmtblock() {
    // The statement-among-arms case takes the same retain-everything path:
    // P0006 (item mixing, from check_arm_mixing) + P0115, StmtBlock keeps
    // the arm AND the statement — nothing is filtered out.
    let src = "fn m() { (a) -> { -true-> x; - 7 -> y; } -> ret; }";
    let pc = pcodes(src);
    assert!(pc.contains(&"P0006"), "P0006 present: {pc:?}");
    assert!(pc.contains(&"P0115"), "P0115 present: {pc:?}");
    let f = parse_one_fn(src);
    let c = first_chain(&f);
    match &c.stages[0].kind {
        StageKind::StmtBlock(b) => {
            assert_eq!(b.items.len(), 2, "arm and statement both retained");
        }
        other => panic!("arms + statement must classify as StmtBlock, got {other:?}"),
    }
}
