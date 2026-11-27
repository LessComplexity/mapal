//! Parser tests: classification table, `seq` blocks, ADR-0011 scan cases.

use super::*;

// ======================================================================
// §14.4 classification table + arm-after-`}` lexer gate interplay
// ======================================================================

#[test]
fn guard_block_classification() {
    // abs-style guard block, all arms → StageKind::Guard, zero diags.
    let src = "fn m() { (x > 0) -> { -true-> x; -false-> y; } -> ret; }";
    no_diags(src);
    let f = parse_one_fn(src);
    let c = first_chain(&f);
    assert!(matches!(c.stages[0].kind, StageKind::Guard(_)));
}

#[test]
fn fanout_classification() {
    // `6 -> { -> a; -> b; }` → Fanout with 2 branches.
    let src = "fn m() { 6 -> { -> a; -> b; } sq -> print; }";
    let f = parse_one_fn(src);
    let c = first_chain(&f);
    match &c.stages[0].kind {
        StageKind::Fanout {
            kind: FanoutKind::Plain,
            branches,
        } => {
            assert_eq!(branches.len(), 2);
        }
        other => panic!("expected Fanout(2), got {other:?}"),
    }
}

#[test]
fn empty_block_fanout_zero_p0010() {
    // `x -> {}` → Fanout(0) + P0010.
    assert_eq!(pcodes("fn m() { x -> {}; }"), vec!["P0010"]);
}

// ======================================================================
// `seq` statement block (ADR-0019, WP1)
// ======================================================================

/// Extract the `SeqBlock` body of the first fn's leading chain's first stage.
/// The chain may be an item or the block tail (a `seq` stage is a block form,
/// so a trailing `;` is optional — an unterminated `seq` becomes the tail).
fn seq_body(src: &str) -> Block {
    let f = parse_one_fn(src);
    let c = match f.body.items.first() {
        Some(BlockItem::Stmt(Stmt {
            kind: StmtKind::Chain(c),
            ..
        })) => c,
        _ => f
            .body
            .tail
            .as_ref()
            .expect("a leading seq chain (item or tail)"),
    };
    match &c.stages[0].kind {
        StageKind::SeqBlock(b) => b.clone(),
        other => panic!("expected SeqBlock, got {other:?}"),
    }
}

#[test]
fn seq_statement_form_headed_chains() {
    // `data -> seq { data -> a; data -> b; }` — statement block, two headed
    // chain statements; no double-wrap, no fanout. Parses clean.
    let src = "fn m() { data -> seq { data -> a; data -> b; } }";
    no_diags(src);
    assert_spans(src);
    let b = seq_body(src);
    assert_eq!(b.items.len(), 2);
    assert!(b.tail.is_none());
    assert!(matches!(
        b.items[0],
        BlockItem::Stmt(Stmt {
            kind: StmtKind::Chain(_),
            ..
        })
    ));
}

#[test]
fn seq_headless_chains_compat() {
    // Compat pin 3: the old bare-chain branch form still parses. Headless
    // statements now live in the SeqBlock (they seed from the seq input in
    // lowering); the tree shape changes but the source is accepted unchanged.
    let src = "fn m() { data -> seq { -> a; -> b; } }";
    no_diags(src);
    let b = seq_body(src);
    assert_eq!(b.items.len(), 2);
    for it in &b.items {
        match it {
            BlockItem::Stmt(Stmt {
                kind: StmtKind::Chain(c),
                ..
            }) => assert!(c.head.is_none(), "headless chain seeds from seq input"),
            other => panic!("expected headless chain statement, got {other:?}"),
        }
    }
}

#[test]
fn seq_rebind_and_loop_kept() {
    // The ADR-0019 win: a `x <- e` rebind and a `loop { }` inside `seq` are
    // now first-class statements — kept, not silently dropped. Clean parse.
    let src = "fn m() { data -> seq { x <- 1; loop { } -> a; } }";
    no_diags(src);
    let b = seq_body(src);
    assert_eq!(b.items.len(), 3);
    assert!(matches!(
        b.items[0],
        BlockItem::Stmt(Stmt {
            kind: StmtKind::Bind(_),
            ..
        })
    ));
    assert!(matches!(
        b.items[1],
        BlockItem::Stmt(Stmt {
            kind: StmtKind::Loop(_),
            ..
        })
    ));
}

#[test]
fn seq_with_tail() {
    // A seq whose body ends in an unterminated chain — that chain is the
    // block's tail (the seq's value in lowering).
    let src = "fn m() { data -> seq { data -> a; b } }";
    no_diags(src);
    let b = seq_body(src);
    assert_eq!(b.items.len(), 1);
    assert!(b.tail.is_some(), "trailing chain is the seq tail");
}

#[test]
fn seq_empty() {
    // `seq { }` — empty statement block, zero diagnostics (no P0010; that is
    // the `-> {}` fanout-classification path, not seq).
    let src = "fn m() { data -> seq { } }";
    no_diags(src);
    let b = seq_body(src);
    assert!(b.items.is_empty());
    assert!(b.tail.is_none());
}

#[test]
fn seq_block_form_optional_semi() {
    // `seq { … }` is a block form (W10): the trailing `;` is optional, so a
    // following statement parses without a separator.
    let src = "fn m() { data -> seq { data -> a; } more -> print; }";
    no_diags(src);
    let f = parse_one_fn(src);
    assert_eq!(f.body.items.len(), 2, "seq stmt + following stmt");
}

#[test]
fn seq_guard_arm_illegal_p0004() {
    // Guard arms are illegal in a seq body (the ordinary block production). A
    // clean guard token there is a *stray* guard → P0004 (its message —
    // "guard arrow outside a guard block" — fits exactly; a seq body is not a
    // guard block). ADR-0019 pins guard arms illegal in `seq`.
    let src = "fn m() { data -> seq { -true-> x; } }";
    assert!(
        codes(src).contains(&"P0004"),
        "guard arm in seq rejected: {:?}",
        codes(src)
    );
}

#[test]
fn rebind_in_fanout_block_p0117() {
    // A `x <- e` rebind inside a fanout block (`void { … }`, the sole remaining
    // fanout-block form after seq migrated) is not a branch: P0117, not a
    // silent drop. The chain branch `-> a` is still kept.
    let src = "fn m() { data -> void { x <- 1; -> a; } }";
    let cs = codes(src);
    assert!(cs.contains(&"P0117"), "dropped rebind flagged: {cs:?}");
}

#[test]
fn loop_in_fanout_block_p0117() {
    // Likewise a `loop { }` statement dropped from a fanout block → P0117.
    let src = "fn m() { data -> void { loop { } -> a; } }";
    assert!(codes(src).contains(&"P0117"));
}

#[test]
fn fanout_block_each_dropped_stmt_draws_p0117() {
    // Regression (ADR-0019 defect #3): the cooldown must NOT collapse the second
    // and later drops in one block into silence. Each dropped non-chain statement
    // gets its own P0117; the chain branch is still kept.
    let count = |src: &str| codes(src).iter().filter(|c| **c == "P0117").count();
    assert_eq!(
        count("fn m() { data -> void { x <- 1; y <- 2; -> a; } }"),
        2,
        "two dropped rebinds → two P0117"
    );
    assert_eq!(
        count("fn m() { data -> void { x <- 1; y <- 2; z <- 3; -> a; } }"),
        3,
        "three dropped rebinds → three P0117"
    );
    assert_eq!(
        count("fn m() { data -> void { loop { } loop { } -> a; } }"),
        2,
        "two dropped loops → two P0117"
    );
    // The branch survives the drops.
    let f = parse_one_fn("fn m() { data -> void { x <- 1; y <- 2; -> a; }; }");
    match &first_chain(&f).stages[0].kind {
        StageKind::Fanout { branches, .. } => assert_eq!(branches.len(), 1),
        other => panic!("expected void Fanout, got {other:?}"),
    }
}

#[test]
fn fanout_block_unterminated_final_branch_kept() {
    // Regression (ADR-0019 defect #3): a final branch written without a trailing
    // `;` is the block tail — it is still a branch, not a silent drop.
    let branch_count = |src: &str| match &first_chain(&parse_one_fn(src)).stages[0].kind {
        StageKind::Fanout { branches, .. } => branches.len(),
        other => panic!("expected void Fanout, got {other:?}"),
    };
    assert_eq!(
        branch_count("fn m() { data -> void { -> a }; }"),
        1,
        "sole unterminated branch kept"
    );
    assert_eq!(
        branch_count("fn m() { data -> void { -> a; -> b }; }"),
        2,
        "trailing unterminated branch kept alongside the terminated one"
    );
}

#[test]
fn seq_arm_mixed_with_stmt_p0006() {
    // The plan's WP1 test matrix "arm in seq → P0006" is satisfiable via a
    // *spaced*/pattern arm mixed with a statement (only the *clean* guard-token
    // form routes to P0004 — see `seq_guard_arm_illegal_p0004`). Lock the P0006
    // path so a future refactor cannot lose it.
    let spaced = codes("fn m() { data -> seq { data -> a; - true -> y; } }");
    assert!(
        spaced.contains(&"P0006"),
        "spaced-arm mix → P0006: {spaced:?}"
    );
    let pattern = codes("fn m() { data -> seq { data -> a; -Some(x)-> y; } }");
    assert!(
        pattern.contains(&"P0006"),
        "pattern-arm mix → P0006: {pattern:?}"
    );
}

#[test]
fn arm_after_rbrace_lexer_gate() {
    // sepia-style nested guard whose `-false->` arm has a block payload that
    // ends in `}` then the next arm starts after `}` (lexer gate passes).
    let src = "fn m() { (a) -> { -true-> hi; -false-> { (b) -> { -true-> lo; -false-> v; } -> bounded; bounded } } -> ret; }";
    no_diags(src);
}

// ======================================================================
// ADR-0011 scan cases (§14.5)
// ======================================================================

#[test]
fn struct_literal_heads_chain() {
    // `Pixel { r: nr } -> ret;` — no `;`/arrow/guard inside braces ⇒ struct.
    let src = "fn m() { Pixel { r: nr, g: ng } -> ret; }";
    no_diags(src);
    let f = parse_one_fn(src);
    let c = first_chain(&f);
    assert!(matches!(
        c.head.as_ref().unwrap().kind,
        ExprKind::Struct { .. }
    ));
}

#[test]
fn labeled_loop_p0110() {
    // `search { x -> y; }` — arrows inside ⇒ labeled loop (P0110).
    let src = "fn m() { search { x -> y; } }";
    assert_eq!(pcodes(src), vec!["P0110"]);
    let f = parse_one_fn(src);
    match &f.body.items[0] {
        BlockItem::Stmt(Stmt {
            kind:
                StmtKind::Loop(LoopStmt {
                    label: LoopLabel::Custom(_),
                    ..
                }),
            ..
        }) => {}
        other => panic!("expected a custom-labeled loop, got {other:?}"),
    }
}

#[test]
fn labeled_loop_arrows_no_top_level_semi() {
    // Pinned regression: `search`-labeled loop whose body has arrows but no
    // top-level `;` ⇒ P0110 via the four-token scan.
    let src = "fn m() { search { x -> ret } }";
    assert_eq!(pcodes(src), vec!["P0110"]);
}

#[test]
fn empty_braces_struct_w13() {
    // `X { }` statement-initial → struct literal (no loop tokens). It becomes a
    // bare-expression tail of the fn body (no `;`), so no P0003.
    let src = "fn m() { X { } }";
    no_diags(src);
    let f = parse_one_fn(src);
    assert!(f.body.tail.is_some(), "struct-literal tail");
}

#[test]
fn flow_free_labeled_body_is_struct_w13() {
    // `X { x }` — no `;`/arrow/guard ⇒ struct literal (operationally-empty loop).
    let src = "fn m() { X { x } }";
    no_diags(src);
}

#[test]
fn demoted_scan_hint_mentions_sigil() {
    // The demoted ADR-0011 scan (§14.5) still parses a loop-shaped un-sigiled
    // `Ident {` as a labeled block for recovery, but its P0110 message now points
    // at the sigiled spelling `:search { … }`.
    let src = "fn m() { search { x -> y; } }";
    assert_eq!(pcodes(src), vec!["P0110"]);
    let out = parse(src);
    let d = out
        .diagnostics
        .iter()
        .find(|d| d.code.0 == "P0110")
        .expect("a P0110");
    assert!(
        d.message.contains(":search { … }"),
        "demoted-scan P0110 should hint the sigiled spelling, got {:?}",
        d.message
    );
}
