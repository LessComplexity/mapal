//! Parser tests: category recovery, regressions, totality, diagnostics merge.

use super::*;

// ======================================================================
// category recovery, `i<-1`, pinned regressions
// ======================================================================

#[test]
fn category_decl_recovery_no_double_report() {
    // `category Foo { … }` — L0004 already emitted by the lexer; the parser
    // parses it as a `type` with NO new diagnostic. Only one L0004, no P-code.
    let src = "category Foo { r: f32 }";
    let out = parse(src);
    let lcount = out
        .diagnostics
        .iter()
        .filter(|d| d.code.0 == "L0004")
        .count();
    assert_eq!(lcount, 1, "exactly one L0004 (no double report)");
    assert!(
        out.diagnostics.iter().all(|d| d.code.0 == "L0004"),
        "no new diagnostic from the parser: {:#?}",
        out.diagnostics
    );
    // And it parses as a Type item.
    assert!(matches!(out.program.items[0], Item::Type(_)));
}

#[test]
fn i_backarrow_one_p0009() {
    // `i<-1` inside an expression position → P0009 with the W2 hint.
    let src = "fn m() { (i<-1) -> ret; }";
    let out = parse(src);
    let d = out
        .diagnostics
        .iter()
        .find(|d| d.code.0 == "P0009")
        .expect("P0009");
    assert!(d.fix.is_some(), "P0009 has the `< -` fix");
}

#[test]
fn pattern_arm_only_block_p0106_twice() {
    // Pinned regression: `opt -> { -Some(x)-> y; -None-> 0; }` ⇒ P0106 ×2, no
    // spurious P0108.
    let src = "fn m() { opt -> { -Some(x)-> y; -None-> 0; } -> ret; }";
    let pc = pcodes(src);
    assert_eq!(
        pc.iter().filter(|c| **c == "P0106").count(),
        2,
        "P0106 ×2: {pc:?}"
    );
    assert!(!pc.contains(&"P0108"), "no spurious P0108: {pc:?}");
}

#[test]
fn rest_element_exactly_p0107() {
    // Pinned regression: `[head, ...tail]` ⇒ exactly P0107.
    let src = "fn m() { [head, ...tail] -> rest; }";
    assert_eq!(pcodes(src), vec!["P0107"]);
}

#[test]
fn list_map_path_exactly_p0109_chain_continues() {
    // Pinned regression: `List::map -> out;` ⇒ exactly P0109, chain continues.
    let src = "fn m() { data -> List::map -> out; }";
    assert_eq!(pcodes(src), vec!["P0109"]);
    // The chain still has the two stages after recovery.
    let f = parse_one_fn(src);
    let c = first_chain(&f);
    assert_eq!(c.stages.len(), 2, "chain continued past the `::` path");
}

#[test]
fn call_expression_p0108() {
    // `arr.len()` → P0108.
    assert!(pcodes("fn m() { x -> arr.len(); }").contains(&"P0108"));
}

#[test]
fn question_postfix_p0101() {
    // `f? -> g?` → P0101 ×2, one per `?`.
    let src = "fn m() { x -> f? -> g?; }";
    assert_eq!(pcodes(src).iter().filter(|c| **c == "P0101").count(), 2);
}

#[test]
fn annotation_p0102_item_kept() {
    // `@executor(ThreadPool) fn …` → P0102, the fn is kept.
    let src = "@executor(ThreadPool)\nfn m() { x -> ret; }";
    assert!(pcodes(src).contains(&"P0102"));
    let out = parse(src);
    assert!(
        out.program.items.iter().any(|i| matches!(i, Item::Fn(_))),
        "the annotated fn is kept"
    );
}

// ======================================================================
// totality / depth (J1) + determinism
// ======================================================================

#[test]
fn deep_nesting_p0011_no_panic() {
    // 300 nested groups exceed the depth limit; P0011 fires, no panic (J1).
    let src = format!(
        "fn m() {{ {}1{} -> ret; }}",
        "(".repeat(300),
        ")".repeat(300)
    );
    let out = parse(&src);
    assert!(
        out.diagnostics.iter().any(|d| d.code.0 == "P0011"),
        "expected P0011 on deep nesting"
    );
}

#[test]
fn deep_array_type_p0011_no_panic() {
    // J1 totality: a type with 300 nested `[` exceeds the depth limit (128). The
    // shared depth guard must fire P0011 and unwind without a stack overflow.
    let src = format!(
        "fn f(p: {}i32{}) {{ p -> ret; }}",
        "[".repeat(300),
        "]".repeat(300)
    );
    let out = parse(&src);
    assert!(
        out.diagnostics.iter().any(|d| d.code.0 == "P0011"),
        "expected P0011 on a deeply nested array type"
    );
}

#[test]
fn deep_tuple_type_p0011_no_panic() {
    // J1 totality: a type with 300 nested `(` tuple-types likewise hits the
    // shared depth guard → P0011, no panic.
    let src = format!(
        "fn f(p: {}i32, i32{}) {{ p -> ret; }}",
        "(".repeat(300),
        ")".repeat(300)
    );
    let out = parse(&src);
    assert!(
        out.diagnostics.iter().any(|d| d.code.0 == "P0011"),
        "expected P0011 on a deeply nested tuple type"
    );
}

/// The depth of an expression tree (a leaf is 1).
fn expr_depth(e: &Expr) -> usize {
    let below = match &e.kind {
        ExprKind::Unary { operand, .. } => expr_depth(operand),
        ExprKind::Binary { lhs, rhs, .. } => expr_depth(lhs).max(expr_depth(rhs)),
        ExprKind::Member { base, .. } | ExprKind::Question(base) => expr_depth(base),
        ExprKind::Index { base, index } => expr_depth(base).max(expr_depth(index)),
        ExprKind::Tuple(es) | ExprKind::Array(es) => es.iter().map(expr_depth).max().unwrap_or(0),
        ExprKind::Struct { fields, .. } => fields
            .iter()
            .filter_map(|f| f.value.as_ref().map(expr_depth))
            .max()
            .unwrap_or(0),
        ExprKind::Call { callee, args } => args
            .iter()
            .map(expr_depth)
            .max()
            .unwrap_or(0)
            .max(expr_depth(callee)),
        _ => 0,
    };
    below + 1
}

/// The deepest head expression of the first fn's chains.
fn deepest_head(src: &str) -> usize {
    let f = parse_one_fn(src);
    let chains = f.body.items.iter().filter_map(|i| match i {
        BlockItem::Stmt(Stmt {
            kind: StmtKind::Chain(c),
            ..
        }) => Some(c),
        _ => None,
    });
    chains
        .chain(f.body.tail.as_ref())
        .filter_map(|c| c.head.as_ref().map(expr_depth))
        .max()
        .unwrap_or(0)
}

#[test]
fn deep_nesting_builds_a_shallow_tree() {
    // J1 totality covers what comes after the parse too: every walk over the
    // tree (lowering, the language server, `Drop`) recurses. Past the depth
    // limit each further `(` of `((((…` reads as a call (P0108), and a run of
    // postfixes (`x.f.f.f…`, `x[0][0]…`) is one loop: both count toward the
    // limit, so the tree stays as shallow as the limit however long the input.
    // On a spawned thread's 2 MiB stack, as the language server checks files.
    let deep = std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(|| {
            let n = 40_000;
            let parens = format!("fn m() {{ {}1{} -> ret; }}", "(".repeat(n), ")".repeat(n));
            let members = format!("fn m() {{ x{} -> ret; }}", ".f".repeat(n));
            let indices = format!("fn m() {{ x{} -> ret; }}", "[0]".repeat(n));
            [parens, members, indices].map(|src| {
                assert!(codes(&src).contains(&"P0011"), "P0011 on {}", &src[..20]);
                deepest_head(&src)
            })
        })
        .unwrap()
        .join()
        .expect("parsed and dropped within a 2 MiB stack");
    for d in deep {
        assert!(d <= 2 * 128, "a tree {d} deep");
    }
    // A legitimate postfix run is untouched.
    assert_eq!(deepest_head("fn m() { a.b[0].c.1 -> ret; }"), 5);
}

#[test]
fn determinism() {
    let src = "fn m() { (x > 0) -> { -true-> x; -false-> y; } -> ret; }";
    assert_eq!(parse(src), parse(src));
}

#[test]
fn empty_input_no_panic() {
    let out = parse("");
    assert!(out.program.items.is_empty());
    assert!(out.diagnostics.is_empty());
}

#[test]
fn top_level_statement_p0012() {
    // A statement at the top level that is no binding chain → P0012
    // (a chain is a top-level binding, ADR-0048 D1).
    assert!(pcodes("x <- y;").contains(&"P0012"));
    assert!(!pcodes("x -> y;").contains(&"P0012"));
}

#[test]
fn lex_diags_preserved_j6() {
    // J6: parse diagnostics ⊇ lex diagnostics. The over-u64 int literal's L0008
    // survives into parse output.
    let src = "fn m() { 99999999999999999999999999 -> ret; }";
    let pout = parse(src);
    let lout = crate::lex(src);
    for ld in &lout.diagnostics {
        assert!(
            pout.diagnostics.contains(ld),
            "lex diagnostic not preserved: {ld:?}"
        );
    }
}

#[test]
fn int_clamp_no_double_report_c14() {
    // An over-u64 int literal: lexer emits L0008 once; the parser re-parses the
    // span and clamps silently (no new diagnostic).
    let src = "fn m() { 99999999999999999999999999 -> ret; }";
    assert_eq!(codes(src).iter().filter(|c| **c == "L0008").count(), 1);
    let f = parse_one_fn(src);
    let c = first_chain(&f);
    assert!(matches!(
        c.head.as_ref().unwrap().kind,
        ExprKind::Int(u64::MAX)
    ));
}

// ======================================================================
// diagnostics merge + sort (DESIGN §15/§18)
// ======================================================================

#[test]
fn diagnostics_sorted_by_span() {
    let src = "fn m() { x -> f? -> g?; }";
    let out = parse(src);
    let spans: Vec<(u32, u32)> = out
        .diagnostics
        .iter()
        .map(|d| (d.span.start, d.span.end))
        .collect();
    let mut sorted = spans.clone();
    sorted.sort_unstable();
    assert_eq!(spans, sorted, "diagnostics stably sorted by (start, end)");
}

#[test]
fn line_index_span_starts() {
    // The `?` postfix on line 2 is the sole diagnostic (P0101). Its span start
    // must map to the exact 1-based line:col of the `?` in the source. Line 1 is
    // `fn m() {`, line 2 is `  x -> f?;` — the `?` is the 9th column of line 2.
    let src = "fn m() {\n  x -> f?;\n}";
    let out = parse(src);
    assert_eq!(out.diagnostics.len(), 1, "exactly one diagnostic");
    let d = &out.diagnostics[0];
    assert_eq!(d.code.0, "P0101");
    let idx = LineIndex::new(src);
    let lc = idx.line_col(d.span.start);
    assert_eq!((lc.line, lc.col), (2, 9), "`?` is at line 2, col 9");
}

// --- ADR-0031: call expressions are uniformly out (the ADR-0029 carve died) --

#[test]
fn iota_fill_calls_are_p0108_rejected_with_teaching_text() {
    // ADR-0031: `iota`/`fill` are pipeline stages; their CALL forms reject
    // like every other call — with the arrow-form teaching message.
    let ds = parse("fn main() { iota(4) -> t; fill(1.0, 4) -> s; }\n").diagnostics;
    assert_eq!(ds.len(), 2, "{ds:?}");
    assert!(ds.iter().all(|d| d.code.0 == "P0108"), "{ds:?}");
    assert!(
        ds.iter().all(|d| d.message.contains("-> iota")),
        "teaching text names the arrow form: {ds:?}"
    );
    // An ordinary call keeps the generic P0108 rejection.
    assert!(
        pcodes("fn main() { foo(4) -> t; }\n").contains(&"P0108"),
        "{:?}",
        pcodes("fn main() { foo(4) -> t; }\n")
    );
    // The arrow forms parse clean.
    no_diags("fn main() { 4 -> iota -> t; (1.0, 4) -> fill -> s; }\n");
}
