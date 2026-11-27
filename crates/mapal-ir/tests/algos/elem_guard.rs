//! `elem_plan` and `guard_plan` tests.

use super::*;

// --- elem_plan (plan-s37-stage-structure) ---------------------------------
//
// The query answers "what is out[i]?" for each array. Only non-`Load` laws are
// recorded — absence means "load it", which is the status quo, so a consumer
// that ignores the plan is still correct.

/// `fn m(a: ([i32;16], ())) -> i32` with a body the caller fills; returns the
/// sealed graph, its FuncId, and whatever the body handed back.
pub(super) fn elem_fixture<F>(build: F) -> (mapal_ir::CategoryIr, mapal_ir::FuncId, Vec<ObjectId>)
where
    F: FnOnce(&mut mapal_ir::FnBuilder<'_>, ObjectId) -> Vec<ObjectId>,
{
    let mut b = IrBuilder::new();
    let f = b
        .declare(
            FuncKind::Named,
            "m",
            Ty::Tuple(vec![i32_arr(16), Ty::Unit]),
            Ty::i32(),
            L,
        )
        .unwrap();
    let probes;
    {
        let mut fb = b.build_fn(f).unwrap();
        let inp = fb.input();
        let a = fb.proj(inp, 0, Dest::Fresh(None), L).unwrap();
        probes = build(&mut fb, a);
        // Keep the fn well-formed: return a[0].
        let zero = fb.constant(Value::I32(0), L).unwrap();
        fb.index(a, zero, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    (ir, f, probes)
}

#[test]
fn elem_iota_is_the_index() {
    let (ir, f, p) = elem_fixture(|fb, _a| {
        let n = fb.constant(Value::I32(16), L).unwrap();
        vec![fb.iota(n, Dest::Fresh(None), L).unwrap()]
    });
    assert_eq!(ir.elem_plan(f).src(p[0]), Some(&ElemSrc::Index));
}

#[test]
fn elem_fill_is_a_broadcast() {
    let (ir, f, p) = elem_fixture(|fb, _a| {
        let x = fb.constant(Value::I32(7), L).unwrap();
        let n = fb.constant(Value::I32(16), L).unwrap();
        vec![fb.fill(x, n, Dest::Fresh(None), L).unwrap()]
    });
    assert!(matches!(
        ir.elem_plan(f).src(p[0]),
        Some(ElemSrc::Broadcast { slot: 0, .. })
    ));
}

/// `zip(iota, a)` — the left half resolves to the index law, the right half
/// cuts to a load. This is the shape saxpy's timed window is built from, and
/// the reason `Zip` needed no special case: it is `Pair(·, ·)`.
#[test]
fn elem_zip_pairs_a_resolved_half_with_a_cut_half() {
    let (ir, f, p) = elem_fixture(|fb, a| {
        let n = fb.constant(Value::I32(16), L).unwrap();
        let ix = fb.iota(n, Dest::Fresh(None), L).unwrap();
        vec![fb.zip(ix, a, Dest::Fresh(None), L).unwrap()]
    });
    let plan = ir.elem_plan(f);
    let Some(ElemSrc::Pair(lhs, rhs)) = plan.src(p[0]) else {
        panic!("zip should carry a Pair law, got {:?}", plan.src(p[0]));
    };
    assert_eq!(**lhs, ElemSrc::Index, "the iota half resolves");
    assert!(
        matches!(**rhs, ElemSrc::Load { slot: Some(1), .. }),
        "the parameter half cuts to a load addressed as (product, 1), got {rhs:?}"
    );
}

/// `Enumerate` needs no constructor of its own: it is `Pair(Index, ·)`, which
/// is the same statement as `enumerate a ≅ zip(iota n, a)`.
#[test]
fn elem_enumerate_is_pair_of_index_and_the_source() {
    let (ir, f, p) = elem_fixture(|fb, a| vec![fb.enumerate(a, Dest::Fresh(None), L).unwrap()]);
    let plan = ir.elem_plan(f);
    let Some(ElemSrc::Pair(lhs, rhs)) = plan.src(p[0]) else {
        panic!(
            "enumerate should carry a Pair law, got {:?}",
            plan.src(p[0])
        );
    };
    assert_eq!(**lhs, ElemSrc::Index);
    assert!(matches!(**rhs, ElemSrc::Load { slot: None, .. }));
}

/// The cut is the whole safety story: anything the query cannot read is simply
/// absent, and absent means "materialize and load" — exactly today's behavior.
#[test]
fn elem_cuts_on_a_parameter_array() {
    let (ir, f, p) = elem_fixture(|_fb, a| vec![a]);
    assert_eq!(ir.elem_plan(f).src(p[0]), None);
    assert!(ir.elem_plan(f).is_empty());
}

/// Nesting composes: `zip(iota, zip(iota, a))` resolves both index halves
/// through the recursion rather than through any per-op special case.
#[test]
fn elem_law_composes_through_nesting() {
    let (ir, f, p) = elem_fixture(|fb, a| {
        let n = fb.constant(Value::I32(16), L).unwrap();
        let ix = fb.iota(n, Dest::Fresh(None), L).unwrap();
        let inner = fb.zip(ix, a, Dest::Fresh(None), L).unwrap();
        let n2 = fb.constant(Value::I32(16), L).unwrap();
        let ix2 = fb.iota(n2, Dest::Fresh(None), L).unwrap();
        vec![fb.zip(ix2, inner, Dest::Fresh(None), L).unwrap()]
    });
    let plan = ir.elem_plan(f);
    let Some(ElemSrc::Pair(lhs, rhs)) = plan.src(p[0]) else {
        panic!("outer zip should carry a Pair law");
    };
    assert_eq!(**lhs, ElemSrc::Index);
    let ElemSrc::Pair(inner_l, _) = &**rhs else {
        panic!("the inner zip's law should recurse, got {rhs:?}");
    };
    assert_eq!(**inner_l, ElemSrc::Index, "the nested iota resolves too");
}

/// The `Apply` gate (plan-s37-stage-structure §7 precondition 10). A `Map` may
/// join the producer family only when its body can be **recomputed at a
/// consumer's read site** — which a body that can trap cannot, because the trap
/// would fire at the consumer's index order instead of the producer's. This is
/// S34's failure class one level up, so it is checked rather than assumed.
#[test]
fn elem_map_with_a_trapping_body_is_cut() {
    fn plan_for(divisor: i32) -> bool {
        let mut b = IrBuilder::new();
        let body = b
            .declare(FuncKind::MapBody, "m", Ty::i32(), Ty::i32(), L)
            .unwrap();
        {
            let mut fb = b.build_fn(body).unwrap();
            let p = fb.input();
            let d = fb.constant(Value::I32(divisor), L).unwrap();
            fb.binop(Operation::Div, p, d, Dest::Ret { slot: None }, L)
                .unwrap();
            fb.finish().unwrap();
        }
        let f = b
            .declare(FuncKind::Named, "f", Ty::Unit, Ty::i32(), L)
            .unwrap();
        let mapped;
        {
            let mut fb = b.build_fn(f).unwrap();
            let n = fb.constant(Value::I32(16), L).unwrap();
            let ix = fb.iota(n, Dest::Fresh(None), L).unwrap();
            mapped = fb.map(body, ix, Dest::Fresh(None), L).unwrap();
            let zero = fb.constant(Value::I32(0), L).unwrap();
            fb.index(mapped, zero, Dest::Ret { slot: None }, L).unwrap();
            fb.finish().unwrap();
        }
        let ir = b.seal(f).unwrap();
        assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
        matches!(ir.elem_plan(f).src(mapped), Some(ElemSrc::Apply { .. }))
    }

    assert!(
        !plan_for(0),
        "a body that can divide by zero must be cut, not recomputed"
    );
    assert!(
        plan_for(3),
        "positive control: a provably safe divisor is classifiable"
    );
}

// --- guard_plan (plan-s39: guards gate the flow) ------------------------------

/// The g1 shape: `phi(42, 7/0, a > 0) -> ret`. The trapping arm's work is
/// owned; the constant arm owns only its boundary edge.
pub(super) fn guard_g1() -> (mapal_ir::CategoryIr, mapal_ir::FuncId) {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "g1", Ty::i32(), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let a = fb.input();
        let zero = fb.constant(Value::I32(0), L).unwrap();
        let cond = fb
            .binop(Operation::Gt, a, zero, Dest::Fresh(Some("cond".into())), L)
            .unwrap();
        let t = fb.constant(Value::I32(42), L).unwrap();
        let seven = fb.constant(Value::I32(7), L).unwrap();
        let zero2 = fb.constant(Value::I32(0), L).unwrap();
        let e = fb
            .binop(
                Operation::Div,
                seven,
                zero2,
                Dest::Fresh(Some("bad".into())),
                L,
            )
            .unwrap();
        fb.phi(t, e, cond, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    (ir, f)
}

#[test]
fn guard_plan_owns_the_trapping_arm() {
    let (ir, f) = guard_g1();
    let sites = ir.guard_plan(f);
    assert_eq!(sites.len(), 1, "one Phi, one site");
    let s = &sites[0];

    // The condition is the Gt result.
    let gt = func_ops(&ir, f, |op| op == Operation::Gt)[0];
    assert_eq!(s.cond, ir.morphism(gt).unwrap().target);

    // True arm: a constant — owns only its boundary edge, cannot trap.
    assert_eq!(s.on_true.own, vec![s.on_true.edge]);
    assert!(!s.on_true.can_trap);

    // False arm: owns the Div, its staging Pair edges, and the boundary edge
    // (last); can trap.
    let div = func_ops(&ir, f, |op| op == Operation::Div)[0];
    assert!(s.on_false.own.contains(&div), "Div is arm-owned");
    assert_eq!(*s.on_false.own.last().unwrap(), s.on_false.edge);
    assert!(s.on_false.can_trap);

    // Disjoint own-lists; the condition's producer is owned by neither.
    for m in &s.on_true.own {
        assert!(!s.on_false.own.contains(m), "arms share {m:?}");
    }
    assert!(!s.on_true.own.contains(&gt));
    assert!(!s.on_false.own.contains(&gt));
}

#[test]
fn guard_plan_shared_value_owns_nothing_but_edges() {
    // phi(x, x, cond) — the same object feeds both arms: removing either edge
    // leaves x alive through the other, so neither arm owns x's producer.
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "sh", Ty::i32(), Ty::i32(), L)
        .unwrap();
    let mul;
    {
        let mut fb = b.build_fn(f).unwrap();
        let a = fb.input();
        let zero = fb.constant(Value::I32(0), L).unwrap();
        let cond = fb
            .binop(Operation::Gt, a, zero, Dest::Fresh(Some("c".into())), L)
            .unwrap();
        let two = fb.constant(Value::I32(2), L).unwrap();
        let x = fb
            .binop(Operation::Mul, a, two, Dest::Fresh(Some("x".into())), L)
            .unwrap();
        fb.phi(x, x, cond, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
        mul = x;
    }
    let ir = b.seal(f).unwrap();
    let sites = ir.guard_plan(f);
    assert_eq!(sites.len(), 1);
    let s = &sites[0];
    assert_eq!(s.on_true.own, vec![s.on_true.edge]);
    assert_eq!(s.on_false.own, vec![s.on_false.edge]);
    assert_eq!(s.on_true.value, mul);
    assert_eq!(s.on_false.value, mul);
    assert!(!s.on_true.can_trap && !s.on_false.can_trap);
}

#[test]
fn guard_plan_nested_sites_partition_ownership() {
    // outer: phi(t1, inner, c1); inner: phi(x2t, x2f, c2) — the golden_mermaid
    // nested shape. The outer false arm owns the inner Phi and c2's producer,
    // but NOT the inner arms' work (direct ownership).
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "nest", Ty::i32(), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let a = fb.input();
        let zero = fb.constant(Value::I32(0), L).unwrap();
        let c1 = fb
            .binop(Operation::Gt, a, zero, Dest::Fresh(Some("c1".into())), L)
            .unwrap();
        let ten = fb.constant(Value::I32(10), L).unwrap();
        let c2 = fb
            .binop(Operation::Lt, a, ten, Dest::Fresh(Some("c2".into())), L)
            .unwrap();
        let three = fb.constant(Value::I32(3), L).unwrap();
        let x2t = fb
            .binop(Operation::Add, a, three, Dest::Fresh(Some("x2t".into())), L)
            .unwrap();
        let four = fb.constant(Value::I32(4), L).unwrap();
        let x2f = fb
            .binop(Operation::Mul, a, four, Dest::Fresh(Some("x2f".into())), L)
            .unwrap();
        let inner = fb
            .phi(x2t, x2f, c2, Dest::Fresh(Some("inner".into())), L)
            .unwrap();
        let t1 = fb.constant(Value::I32(1), L).unwrap();
        fb.phi(t1, inner, c1, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let sites = ir.guard_plan(f);
    assert_eq!(sites.len(), 2, "inner and outer");
    // Sites are in topo order of the Phi: inner first.
    let (si, so) = (&sites[0], &sites[1]);

    let phis = func_ops(&ir, f, |op| op == Operation::Phi);
    assert_eq!(si.phi, phis[0]);
    assert_eq!(so.phi, phis[1]);

    let add = func_ops(&ir, f, |op| op == Operation::Add)[0];
    let mul = func_ops(&ir, f, |op| op == Operation::Mul)[0];
    let lt = func_ops(&ir, f, |op| op == Operation::Lt)[0];

    // Inner arms own their producers.
    assert!(si.on_true.own.contains(&add));
    assert!(si.on_false.own.contains(&mul));

    // Outer false arm: owns the inner Phi and the inner condition's producer,
    // but not the inner arms' work — that has exactly one owner, the inner site.
    assert!(so.on_false.own.contains(&si.phi));
    assert!(so.on_false.own.contains(&lt));
    assert!(!so.on_false.own.contains(&add));
    assert!(!so.on_false.own.contains(&mul));
    assert!(!so.on_false.own.contains(&si.on_true.edge));
    assert!(!so.on_false.own.contains(&si.on_false.edge));
}

#[test]
fn guard_plan_topo_order_and_edge_last() {
    let (ir, f) = guard_g1();
    let topo = ir.topo_order(f);
    let pos = |m: MorphismId| topo.iter().position(|&x| x == m).unwrap();
    let s = &ir.guard_plan(f)[0];
    for arm in [&s.on_true, &s.on_false] {
        for w in arm.own.windows(2) {
            assert!(pos(w[0]) < pos(w[1]), "own list not in topo order");
        }
        assert_eq!(*arm.own.last().unwrap(), arm.edge);
    }
}

// --- guard_plan (plan-s40: the arm owns the loop) -----------------------------

/// A loop whose exit value feeds one Phi arm exclusively (builder-built; L1406
/// keeps this shape out of surface Mapal). The loop body divides by the merge
/// (starts 0), so RUNNING it traps — the point is that the untaken arm's loop
/// must not run.
pub(super) fn loop_in_arm() -> (mapal_ir::CategoryIr, mapal_ir::FuncId) {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "la", Ty::i32(), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let cond = fb.constant(Value::Bool(true), L).unwrap();
        let t = fb.constant(Value::I32(42), L).unwrap();
        let zero = fb.constant(Value::I32(0), L).unwrap();
        let lh = fb.begin_loop(zero, L).unwrap();
        let merge = fb.merge_of(&lh);
        let ten = fb.constant(Value::I32(10), L).unwrap();
        let lc = fb
            .binop(Operation::Lt, merge, ten, Dest::Fresh(None), L)
            .unwrap();
        let seven = fb.constant(Value::I32(7), L).unwrap();
        let next = fb
            .binop(Operation::Div, seven, merge, Dest::Fresh(None), L)
            .unwrap();
        fb.loop_back(&lh, next, lc, L).unwrap();
        let ex = fb
            .loop_exit(&lh, merge, lc, Dest::Fresh(Some("ex".into())), L)
            .unwrap();
        fb.end_loop(lh).unwrap();
        fb.phi(t, ex, cond, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    (ir, f)
}

#[test]
fn guard_plan_arm_owns_a_loop_as_a_unit() {
    let (ir, f) = loop_in_arm();
    let sites = ir.guard_plan(f);
    assert_eq!(sites.len(), 1, "one Phi, one site");
    let s = &sites[0];

    // The false arm owns the loop through its LoopEnter handle alone.
    let enters: Vec<_> = s
        .on_false
        .own
        .iter()
        .filter(|&&m| ir.morphism(m).unwrap().op == Operation::LoopEnter)
        .collect();
    assert_eq!(enters.len(), 1, "exactly one handle: {:?}", s.on_false.own);

    // Internals stay out: no LoopBack/LoopExit, no cone morphisms.
    let lt = func_ops(&ir, f, |op| op == Operation::Lt)[0];
    let div = func_ops(&ir, f, |op| op == Operation::Div)[0];
    for &m in &s.on_false.own {
        let op = ir.morphism(m).unwrap().op;
        assert!(
            !matches!(op, Operation::LoopBack | Operation::LoopExit),
            "machinery in own: {op:?}"
        );
    }
    assert!(!s.on_false.own.contains(&lt), "cone Lt leaked into own");
    assert!(!s.on_false.own.contains(&div), "cone Div leaked into own");

    // The handle carries the unit's flags: a loop is heavy, and this one traps.
    assert!(s.on_false.heavy, "a loop is heavy by definition");
    assert!(s.on_false.can_trap, "the unit's Div must reach the flag");
    assert!(s.gated(), "trap-capable arm must gate");

    // The true arm is a bare constant.
    assert_eq!(s.on_true.own, vec![s.on_true.edge]);

    // Own stays topo-ordered, edge last.
    let topo = ir.topo_order(f);
    let pos = |m: MorphismId| topo.iter().position(|&x| x == m).unwrap();
    for w in s.on_false.own.windows(2) {
        assert!(pos(w[0]) < pos(w[1]), "own list not in topo order");
    }
    assert_eq!(*s.on_false.own.last().unwrap(), s.on_false.edge);
}

#[test]
fn guard_plan_loop_shared_with_outside_stays_unconditional() {
    // The exit value feeds BOTH arms: neither arm may own the loop, which then
    // runs unconditionally (always safe).
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "ls", Ty::i32(), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let cond = fb.constant(Value::Bool(true), L).unwrap();
        let zero = fb.constant(Value::I32(0), L).unwrap();
        let lh = fb.begin_loop(zero, L).unwrap();
        let merge = fb.merge_of(&lh);
        let ten = fb.constant(Value::I32(10), L).unwrap();
        let lc = fb
            .binop(Operation::Lt, merge, ten, Dest::Fresh(None), L)
            .unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        let next = fb
            .binop(Operation::Add, merge, one, Dest::Fresh(None), L)
            .unwrap();
        fb.loop_back(&lh, next, lc, L).unwrap();
        let ex = fb
            .loop_exit(&lh, merge, lc, Dest::Fresh(Some("ex".into())), L)
            .unwrap();
        fb.end_loop(lh).unwrap();
        let two = fb.constant(Value::I32(2), L).unwrap();
        let three = fb.constant(Value::I32(3), L).unwrap();
        let ta = fb
            .binop(Operation::Mul, ex, two, Dest::Fresh(None), L)
            .unwrap();
        let ea = fb
            .binop(Operation::Mul, ex, three, Dest::Fresh(None), L)
            .unwrap();
        fb.phi(ta, ea, cond, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let sites = ir.guard_plan(f);
    assert_eq!(sites.len(), 1);
    let s = &sites[0];
    for (tag, arm) in [("true", &s.on_true), ("false", &s.on_false)] {
        assert!(
            !arm.own
                .iter()
                .any(|&m| ir.morphism(m).unwrap().op == Operation::LoopEnter),
            "{tag} arm owns a loop the other arm also reads"
        );
    }
}

#[test]
fn guard_plan_gates_arm_inside_loop_body() {
    // The dual topology: a Phi INSIDE the loop body whose false arm exclusively
    // owns a trapping Div. The site gates; machinery stays out of its own-list.
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "lb", Ty::i32(), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let zero = fb.constant(Value::I32(0), L).unwrap();
        let lh = fb.begin_loop(zero, L).unwrap();
        let merge = fb.merge_of(&lh);
        let ten = fb.constant(Value::I32(10), L).unwrap();
        let lc = fb
            .binop(Operation::Lt, merge, ten, Dest::Fresh(None), L)
            .unwrap();
        let hundred = fb.constant(Value::I32(100), L).unwrap();
        let bc = fb
            .binop(Operation::Lt, merge, hundred, Dest::Fresh(None), L)
            .unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        let inc = fb
            .binop(Operation::Add, merge, one, Dest::Fresh(None), L)
            .unwrap();
        let seven = fb.constant(Value::I32(7), L).unwrap();
        let zc = fb.constant(Value::I32(0), L).unwrap();
        let bad = fb
            .binop(Operation::Div, seven, zc, Dest::Fresh(None), L)
            .unwrap();
        let next = fb.phi(inc, bad, bc, Dest::Fresh(None), L).unwrap();
        fb.loop_back(&lh, next, lc, L).unwrap();
        fb.loop_exit(&lh, merge, lc, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.end_loop(lh).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let sites = ir.guard_plan(f);
    assert_eq!(sites.len(), 1, "the in-body Phi is a site");
    let s = &sites[0];
    let div = func_ops(&ir, f, |op| op == Operation::Div)[0];
    assert!(s.on_false.own.contains(&div), "Div is arm-owned");
    assert!(s.on_false.can_trap);
    assert!(s.gated());
    for arm in [&s.on_true, &s.on_false] {
        for &m in &arm.own {
            let op = ir.morphism(m).unwrap().op;
            assert!(
                !matches!(
                    op,
                    Operation::LoopEnter | Operation::LoopBack | Operation::LoopExit
                ),
                "machinery in an in-body arm: {op:?}"
            );
        }
    }
}
