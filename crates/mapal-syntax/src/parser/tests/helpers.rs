//! Shared parser-test helpers: diagnostics codes, span walkers.

use super::*;

// --- helpers --------------------------------------------------------------

pub(super) fn codes(src: &str) -> Vec<&'static str> {
    parse(src).diagnostics.iter().map(|d| d.code.0).collect()
}

/// The multiset of parser P-codes (lex L-codes excluded), as a sorted vec.
pub(super) fn pcodes(src: &str) -> Vec<&'static str> {
    let mut v: Vec<&'static str> = parse(src)
        .diagnostics
        .iter()
        .map(|d| d.code.0)
        .filter(|c| c.starts_with('P'))
        .collect();
    v.sort_unstable();
    v
}

pub(super) fn no_diags(src: &str) {
    let out = parse(src);
    assert!(
        out.diagnostics.is_empty(),
        "expected zero diagnostics for {src:?}, got {:#?}",
        out.diagnostics
    );
}

/// Parse and return the single fn body's first statement chain, panicking
/// otherwise. Convenience for inspecting expression/stage shapes.
pub(super) fn parse_one_fn(src: &str) -> FnDecl {
    let out = parse(src);
    match out.program.items.into_iter().next() {
        Some(Item::Fn(f)) => f,
        other => panic!("expected one fn, got {other:?}"),
    }
}

/// Walk every node and assert child spans are within parent spans and within
/// source bounds (J2). Returns the count of nodes visited.
pub(super) fn assert_spans(src: &str) {
    let out = parse(src);
    let len = src.len() as u32;
    for item in &out.program.items {
        if let Item::Fn(f) = item {
            assert!(f.span.end <= len);
            walk_block(&f.body, f.span, len);
        }
    }
}

pub(super) fn within(child: SourceLoc, parent: SourceLoc) -> bool {
    child.start >= parent.start && child.end <= parent.end
}

/// J2 helper: a `Ty` and its nested element types are within bounds and parent.
pub(super) fn walk_ty(t: &Ty, parent: SourceLoc, len: u32) {
    assert!(t.span.end <= len, "ty span out of bounds: {:?}", t.span);
    assert!(within(t.span, parent), "ty {:?} ⊄ {:?}", t.span, parent);
    match &t.kind {
        TyKind::Array { elem, .. } | TyKind::Dynamic(elem) => walk_ty(elem, t.span, len),
        TyKind::Tuple(es) => {
            for e in es {
                walk_ty(e, t.span, len);
            }
        }
        _ => {}
    }
}

pub(super) fn walk_block(b: &Block, parent: SourceLoc, len: u32) {
    assert!(b.span.end <= len, "block span out of bounds");
    assert!(within(b.span, parent), "block {:?} ⊄ {:?}", b.span, parent);
    for it in &b.items {
        match it {
            BlockItem::Stmt(s) => walk_stmt(s, b.span, len),
            BlockItem::Arm(a) => {
                assert!(within(a.span, b.span), "arm ⊄ block");
                walk_payload(&a.payload, a.span, len);
            }
        }
    }
    if let Some(t) = &b.tail {
        assert!(within(t.span, b.span), "tail ⊄ block");
        walk_chain(t, b.span, len);
    }
}

pub(super) fn walk_stmt(s: &Stmt, parent: SourceLoc, len: u32) {
    assert!(within(s.span, parent), "stmt {:?} ⊄ {:?}", s.span, parent);
    match &s.kind {
        StmtKind::Chain(c) => walk_chain(c, s.span, len),
        StmtKind::Bind(b) => walk_expr(&b.value, s.span, len),
        StmtKind::Loop(l) => walk_block(&l.body, s.span, len),
        StmtKind::Error => {}
    }
}

pub(super) fn walk_payload(p: &ArmPayload, parent: SourceLoc, len: u32) {
    match p {
        ArmPayload::Chain(c) => walk_chain(c, parent, len),
        ArmPayload::Block(b) => walk_block(b, parent, len),
    }
}

pub(super) fn walk_chain(c: &Chain, parent: SourceLoc, len: u32) {
    assert!(within(c.span, parent), "chain {:?} ⊄ {:?}", c.span, parent);
    if let Some(h) = &c.head {
        walk_expr(h, c.span, len);
    }
    for st in &c.stages {
        assert!(within(st.span, c.span), "stage ⊄ chain");
        match &st.kind {
            StageKind::Expr(e) | StageKind::OpShorthand { expr: e } => walk_expr(e, st.span, len),
            StageKind::Bind { ty: Some(ty), .. } => walk_ty(ty, st.span, len),
            StageKind::Guard(arms) => {
                for a in arms {
                    assert!(within(a.span, st.span), "stage-arm ⊄ stage");
                    assert!(within(a.discr_span, a.span), "arm discr ⊄ arm");
                    walk_payload(&a.payload, st.span, len);
                }
            }
            StageKind::Fanout { branches, .. } => {
                for br in branches {
                    walk_chain(br, st.span, len);
                }
            }
            StageKind::MapFold { params, body, .. } => {
                for p in params {
                    assert!(within(p.span, st.span), "map/fold param ⊄ stage");
                }
                walk_block(body, st.span, len)
            }
            StageKind::StmtBlock(body) | StageKind::SeqBlock(body) => {
                walk_block(body, st.span, len)
            }
            _ => {}
        }
    }
}

pub(super) fn walk_expr(e: &Expr, parent: SourceLoc, len: u32) {
    assert!(e.span.end <= len, "expr span out of bounds: {:?}", e.span);
    assert!(within(e.span, parent), "expr {:?} ⊄ {:?}", e.span, parent);
    match &e.kind {
        ExprKind::Unary { operand, .. } => walk_expr(operand, e.span, len),
        ExprKind::Binary { lhs, rhs, .. } => {
            walk_expr(lhs, e.span, len);
            walk_expr(rhs, e.span, len);
        }
        ExprKind::Member { base, .. } => walk_expr(base, e.span, len),
        ExprKind::Index { base, index } => {
            walk_expr(base, e.span, len);
            walk_expr(index, e.span, len);
        }
        ExprKind::Tuple(xs) | ExprKind::Array(xs) => {
            for x in xs {
                walk_expr(x, e.span, len);
            }
        }
        ExprKind::Struct { fields, .. } => {
            for f in fields {
                if let Some(v) = &f.value {
                    walk_expr(v, f.span, len);
                }
            }
        }
        ExprKind::Call { callee, args } => {
            walk_expr(callee, e.span, len);
            for a in args {
                walk_expr(a, e.span, len);
            }
        }
        ExprKind::Question(b) => walk_expr(b, e.span, len),
        _ => {}
    }
}

/// Extract the first statement's chain from a fn body.
pub(super) fn first_chain(f: &FnDecl) -> &Chain {
    match &f.body.items[0] {
        BlockItem::Stmt(Stmt {
            kind: StmtKind::Chain(c),
            ..
        }) => c,
        other => panic!("expected a chain statement, got {other:?}"),
    }
}
