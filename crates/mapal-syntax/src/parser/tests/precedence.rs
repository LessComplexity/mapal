//! Parser tests: examples, section 3.6 precedence, precedence/associativity table.

use super::*;

// ======================================================================
// Acceptance: examples parse with zero diagnostics (J4 — also in golden tests)
// ======================================================================

#[test]
fn examples_parse_clean() {
    for name in [
        "abs",
        "fanout",
        "fir",
        "pipeline",
        "sepia",
        "sum_to_n",
        "zip_demo",
        "vector_add",
        "seq_demo",
    ] {
        let path = format!("{}/../../examples/{name}.mapal", env!("CARGO_MANIFEST_DIR"));
        let src = std::fs::read_to_string(&path).expect("read example");
        let out = parse(&src);
        assert!(
            out.diagnostics.is_empty(),
            "{name}.mapal produced diagnostics: {:#?}",
            out.diagnostics
        );
        assert_spans(&src);
    }
}

// ======================================================================
// §3.6 precedence examples verbatim (DESIGN §20 item 4)
// ======================================================================

#[test]
fn s36_a_plus_b_arrow_c() {
    // `a + b -> c` : the head is `(a + b)`, one stage `-> c`.
    let f = parse_one_fn("fn m() { a + b -> c; }");
    let c = first_chain(&f);
    match &c.head.as_ref().unwrap().kind {
        ExprKind::Binary { op: BinOp::Add, .. } => {}
        other => panic!("expected Add head, got {other:?}"),
    }
    assert_eq!(c.stages.len(), 1);
}

#[test]
fn s36_a_arrow_b_plus_c_arrow_d() {
    // ADR-0005: `a -> b + c -> d` ≡ `a -> (b + c) -> d`. Head `a`, two stages;
    // the first stage is the expression `b + c`.
    let f = parse_one_fn("fn m() { a -> b + c -> d; }");
    let c = first_chain(&f);
    assert!(matches!(c.head.as_ref().unwrap().kind, ExprKind::Var(_)));
    assert_eq!(c.stages.len(), 2, "two stages");
    match &c.stages[0].kind {
        StageKind::Expr(Expr {
            kind: ExprKind::Binary { op: BinOp::Add, .. },
            ..
        }) => {}
        other => panic!("first stage should be `b + c`, got {other:?}"),
    }
}

#[test]
fn s36_x_arrow_f_dot_method() {
    // `x -> f.method` : head `x`, one stage which is a member access.
    let f = parse_one_fn("fn m() { x -> f.method; }");
    let c = first_chain(&f);
    assert_eq!(c.stages.len(), 1);
    match &c.stages[0].kind {
        StageKind::Expr(Expr {
            kind: ExprKind::Member { .. },
            ..
        }) => {}
        other => panic!("expected member-access stage, got {other:?}"),
    }
}

// ======================================================================
// precedence / associativity table (W14, W15)
// ======================================================================

#[test]
fn mul_binds_tighter_than_add() {
    // `-> + 2 * 3` ⇒ `· + (2 * 3)`. Use a pipeline so we exercise the climber.
    let f = parse_one_fn("fn m() { x -> + 2 * 3; }");
    let c = first_chain(&f);
    match &c.stages[0].kind {
        StageKind::OpShorthand {
            expr:
                Expr {
                    kind:
                        ExprKind::Binary {
                            op: BinOp::Add,
                            rhs,
                            ..
                        },
                    ..
                },
        } => {
            assert!(matches!(rhs.kind, ExprKind::Binary { op: BinOp::Mul, .. }));
        }
        other => panic!("expected `· + (2*3)`, got {other:?}"),
    }
}

#[test]
fn op_shorthand_simple_form() {
    // `-> + 5` ⇒ Binary(Add, Hole, Int 5).
    let f = parse_one_fn("fn m() { x -> + 5; }");
    let c = first_chain(&f);
    match &c.stages[0].kind {
        StageKind::OpShorthand {
            expr:
                Expr {
                    kind:
                        ExprKind::Binary {
                            op: BinOp::Add,
                            lhs,
                            rhs,
                            ..
                        },
                    ..
                },
        } => {
            assert!(matches!(lhs.kind, ExprKind::Hole), "hole is leftmost leaf");
            assert!(matches!(rhs.kind, ExprKind::Int(5)));
        }
        other => panic!("expected `· + 5`, got {other:?}"),
    }
}

#[test]
fn op_shorthand_mul_then_add() {
    // `-> * 2 + 1` ⇒ `(· * 2) + 1` = Binary(Add, Binary(Mul, Hole, Int 2), Int 1).
    let f = parse_one_fn("fn m() { x -> * 2 + 1; }");
    let c = first_chain(&f);
    match &c.stages[0].kind {
        StageKind::OpShorthand {
            expr:
                Expr {
                    kind:
                        ExprKind::Binary {
                            op: BinOp::Add,
                            lhs,
                            rhs,
                            ..
                        },
                    ..
                },
        } => {
            assert!(matches!(rhs.kind, ExprKind::Int(1)));
            match &lhs.kind {
                ExprKind::Binary {
                    op: BinOp::Mul,
                    lhs: ll,
                    ..
                } => {
                    assert!(matches!(ll.kind, ExprKind::Hole));
                }
                other => panic!("expected `(· * 2)`, got {other:?}"),
            }
        }
        other => panic!("expected `(· * 2) + 1`, got {other:?}"),
    }
}

#[test]
fn exactly_one_hole_leftmost() {
    // Invariant: exactly one Hole, leftmost leaf, nowhere else.
    let f = parse_one_fn("fn m() { x -> + 2 * 3; }");
    let c = first_chain(&f);
    let StageKind::OpShorthand { expr } = &c.stages[0].kind else {
        panic!("op-shorthand");
    };
    assert_eq!(count_holes(expr), 1);
}

fn count_holes(e: &Expr) -> usize {
    match &e.kind {
        ExprKind::Hole => 1,
        ExprKind::Binary { lhs, rhs, .. } => count_holes(lhs) + count_holes(rhs),
        ExprKind::Unary { operand, .. } => count_holes(operand),
        _ => 0,
    }
}

#[test]
fn unary_neg_precedence_w15() {
    // `x * -1` ⇒ `x * (-1)`  (abs.mapal). Note: in expression position, after
    // `*` a `-` is unary minus (not op-shorthand — that is stage-position only).
    let f = parse_one_fn("fn m() { x * -1 -> ret; }");
    let c = first_chain(&f);
    match &c.head.as_ref().unwrap().kind {
        ExprKind::Binary {
            op: BinOp::Mul,
            rhs,
            ..
        } => {
            assert!(matches!(rhs.kind, ExprKind::Unary { op: UnOp::Neg, .. }));
        }
        other => panic!("expected `x * (-1)`, got {other:?}"),
    }
}

#[test]
fn unary_neg_member_w15() {
    // `-x.f` ⇒ `-(x.f)` : unary looser than postfix member.
    let f = parse_one_fn("fn m() { -x.f -> ret; }");
    let c = first_chain(&f);
    match &c.head.as_ref().unwrap().kind {
        ExprKind::Unary {
            op: UnOp::Neg,
            operand,
            ..
        } => {
            assert!(matches!(operand.kind, ExprKind::Member { .. }));
        }
        other => panic!("expected `-(x.f)`, got {other:?}"),
    }
}

#[test]
fn not_and_precedence_w15() {
    // `!a && b` ⇒ `(!a) && b`.
    let f = parse_one_fn("fn m() { !a && b -> ret; }");
    let c = first_chain(&f);
    match &c.head.as_ref().unwrap().kind {
        ExprKind::Binary {
            op: BinOp::And,
            lhs,
            ..
        } => {
            assert!(matches!(lhs.kind, ExprKind::Unary { op: UnOp::Not, .. }));
        }
        other => panic!("expected `(!a) && b`, got {other:?}"),
    }
}

#[test]
fn chained_comparison_p0007_w14() {
    // `a < b < c` → P0007 (non-associative); lhs `a < b` kept.
    assert_eq!(pcodes("fn m() { a < b < c -> ret; }"), vec!["P0007"]);
}

#[test]
fn equality_chain_p0007() {
    // `a == b == c` → P0007.
    assert_eq!(pcodes("fn m() { a == b == c -> ret; }"), vec!["P0007"]);
}

#[test]
fn cmp_chain_cascade_single_p0007() {
    // `a < b < c < d` (4-deep chain) → exactly ONE P0007 and ZERO P0001: the
    // recovery loop absorbs every further `CMPOP add` pair under one P0007
    // (DESIGN §17 P0007: one per chain, no cascade).
    let pc = pcodes("fn m() { a < b < c < d -> ret; }");
    assert_eq!(pc.iter().filter(|c| **c == "P0007").count(), 1, "one P0007");
    assert_eq!(pc.iter().filter(|c| **c == "P0001").count(), 0, "no P0001");
}

#[test]
fn op_shorthand_cmp_chain_p0007() {
    // op-shorthand hole climber (level 5) must enforce non-associativity too:
    // `-> < a < b` → exactly one P0007, no generic P0001 (FIX: mirror parse_cmp
    // on the climber path; DESIGN §17 P0007 "enforced on both expression paths").
    let pc = pcodes("fn m() { x -> < a < b; }");
    assert_eq!(pc.iter().filter(|c| **c == "P0007").count(), 1, "one P0007");
    assert_eq!(pc.iter().filter(|c| **c == "P0001").count(), 0, "no P0001");
}
