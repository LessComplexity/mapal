//! `last_use_plan` tests: death positions, escapes, carried sets.

use super::*;

// ---------------------------------------------------------------------------
// last_use_plan (docs/components/ir/plans/plan-last-use.md §2, rules 1-6)
// ---------------------------------------------------------------------------

/// Find the one morphism of `op` in `ir` (test helper — the shapes below
/// have exactly one of each).
pub(super) fn one_op(ir: &mapal_ir::CategoryIr, op: Operation) -> mapal_ir::MorphismId {
    let mut hits: Vec<mapal_ir::MorphismId> = ir
        .morphisms()
        .filter(|(_, m)| m.op == op)
        .map(|(id, _)| id)
        .collect();
    assert_eq!(hits.len(), 1, "expected exactly one {op:?}");
    hits.remove(0)
}

/// The matmul4-class shape as the smallest honest version: one array carried
/// through a canonical loop — `merge` is the carried array, the guard reads
/// it (`Index < n`), the advance cone writes it (`Update`), the exit
/// releases it. Returns the sealed graph and the entry id.
pub(super) fn carried_array_loop() -> (mapal_ir::CategoryIr, mapal_ir::FuncId) {
    let arr4 = Ty::Array {
        elem: Box::new(Ty::i32()),
        size: 4,
    };
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "build", Ty::i32(), arr4, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let n = fb.input();
        let z0 = fb.constant(Value::I32(0), L).unwrap();
        let z = fb
            .pack_array(&[z0, z0, z0, z0], Dest::Fresh(Some("z".into())), L)
            .unwrap();
        let lh = fb.begin_loop(z, L).unwrap();
        let merge = fb.merge_of(&lh);
        // guard: merge[0] < n (a decide-cone read of the carried array).
        let e = fb.index(merge, z0, Dest::Fresh(None), L).unwrap();
        let cond = fb.binop(Operation::Lt, e, n, Dest::Fresh(None), L).unwrap();
        // advance: merge' = update(merge, 0, 1).
        let one = fb.constant(Value::I32(1), L).unwrap();
        let upd = fb.update(merge, z0, one, Dest::Fresh(None), L).unwrap();
        fb.loop_back(&lh, upd, cond, L).unwrap();
        fb.loop_exit(&lh, merge, cond, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.end_loop(lh).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    (ir, f)
}

/// The same loop over a BORROWED init (the fn's array parameter feeds
/// `LoopEnter` directly).
pub(super) fn borrowed_init_loop() -> (mapal_ir::CategoryIr, mapal_ir::FuncId) {
    let arr4 = Ty::Array {
        elem: Box::new(Ty::i32()),
        size: 4,
    };
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "build", arr4.clone(), arr4, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let a = fb.input();
        let lh = fb.begin_loop(a, L).unwrap();
        let merge = fb.merge_of(&lh);
        let z0 = fb.constant(Value::I32(0), L).unwrap();
        let e = fb.index(merge, z0, Dest::Fresh(None), L).unwrap();
        let ten = fb.constant(Value::I32(10), L).unwrap();
        let cond = fb
            .binop(Operation::Lt, e, ten, Dest::Fresh(None), L)
            .unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        let upd = fb.update(merge, z0, one, Dest::Fresh(None), L).unwrap();
        fb.loop_back(&lh, upd, cond, L).unwrap();
        fb.loop_exit(&lh, merge, cond, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.end_loop(lh).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    (ir, f)
}

#[test]
fn last_use_carried_pair_update_result_carried_source_dead() {
    let (ir, f) = carried_array_loop();
    let plan = ir.last_use_plan(f);
    let update_m = one_op(&ir, Operation::Update);
    let update = ir.morphism(update_m).unwrap().target;
    let merge = ir
        .objects()
        .find(|(_, o)| o.kind == mapal_ir::ObjectKind::LoopMerge)
        .map(|(id, _)| id)
        .unwrap();
    let back_route = ir
        .morphism(one_op(&ir, Operation::LoopBack))
        .unwrap()
        .source;

    // Rule 3: the Update result and the route cross the LoopBack into merge.
    assert_eq!(plan.carried_by(update), Some(merge));
    assert_eq!(plan.carried_by(back_route), Some(merge));
    // …but the merge itself, the init literal, and the guard's Index are not
    // carried.
    assert_eq!(plan.carried_by(merge), None);
    // The guard cond rides slot 1 of the route — consumed, never carried.
    let cond = ir.morphism(one_op(&ir, Operation::Lt)).unwrap().target;
    assert_eq!(plan.carried_by(cond), None);

    // Rule 2: the merge does NOT escape through its own loop's exit (the
    // per-iteration release valve; the exit object itself is what escapes).
    assert!(!plan.escapes(merge));
    assert!(plan.escapes(ir.func(f).unwrap().output));
    // The Parameter always escapes (borrowed — never written in place).
    let input = ir.func(f).unwrap().input;
    assert!(plan.escapes(input));
    // The local init literal escapes too, conservatively: on the
    // zero-iteration path its buffer IS the returned buffer (the consumer's
    // pointer-value guard is the finer instrument — plan §3's note).
    let init = ir
        .morphism(one_op(&ir, Operation::LoopEnter))
        .unwrap()
        .source;
    assert!(plan.escapes(init));

    // Rule 4 / the matmul4 note: the merge (the Update's source) is dead
    // after the update — all its uses (the decide-cone Index read, the
    // exit-route Pair and its LoopExit pin) rank before the advance-cone
    // Update; it does not escape; it is not carried.
    let upd_pos = plan.position(update_m).unwrap();
    assert!(
        plan.dead_after(merge, upd_pos),
        "source dead_after the update"
    );
    // …but not earlier: at the decide-cone Index read it is still live.
    let idx_pos = plan.position(one_op(&ir, Operation::Index)).unwrap();
    assert!(!plan.dead_after(merge, idx_pos));
    assert!(!plan.dead_after(update, upd_pos), "carried is never dead");
    // death(merge) is the Update's position — its greatest use.
    assert_eq!(plan.death(merge), Some(upd_pos));

    // Rule 1's ranking: LoopExit between the cones, LoopBack past all body.
    let exit_pos = plan.position(one_op(&ir, Operation::LoopExit)).unwrap();
    let back_pos = plan.position(one_op(&ir, Operation::LoopBack)).unwrap();
    assert!(idx_pos < exit_pos, "decide before LoopExit");
    assert!(exit_pos < upd_pos, "LoopExit before advance");
    assert!(upd_pos < back_pos, "advance before LoopBack");
}

#[test]
fn last_use_borrowed_init_is_never_dead() {
    let (ir, f) = borrowed_init_loop();
    let plan = ir.last_use_plan(f);
    let enter_m = one_op(&ir, Operation::LoopEnter);
    let init = ir.morphism(enter_m).unwrap().source;
    let merge = ir
        .objects()
        .find(|(_, o)| o.kind == mapal_ir::ObjectKind::LoopMerge)
        .map(|(id, _)| id)
        .unwrap();

    // Rule 2: a Parameter feeding LoopEnter escapes — never dead, never
    // written in place ("Borrowed/init handles fail rule 2").
    assert!(plan.escapes(init));
    assert_eq!(plan.death(init), None);
    assert!(!plan.dead_after(init, plan.position(enter_m).unwrap()));
    assert!(!plan.dead_after(init, u32::MAX));
    // The merge's own dead_after stays true (the carried-update case of rule
    // 3) — so the CONSUMER's composition is "dead_after(source, idx) AND NOT
    // escapes(loop init)": the borrowed handle fails the second clause and
    // the loop falls back to the full-copy Update.
    let upd_pos = plan.position(one_op(&ir, Operation::Update)).unwrap();
    assert!(plan.dead_after(merge, upd_pos));
    assert!(plan.escapes(init), "the consumer's borrowed-init veto");
}

#[test]
fn last_use_chain_death_positions() {
    // x -> +1 -> a; a -> +2 -> b; b -> +3 -> ret, plus one Fresh object
    // nobody consumes (dead from definition).
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", Ty::i32(), Ty::i32(), L)
        .unwrap();
    let mut adds: Vec<mapal_ir::MorphismId> = Vec::new();
    let dead;
    {
        let mut fb = b.build_fn(f).unwrap();
        let x = fb.input();
        let one = fb.constant(Value::I32(1), L).unwrap();
        let two = fb.constant(Value::I32(2), L).unwrap();
        let three = fb.constant(Value::I32(3), L).unwrap();
        dead = fb
            .binop(Operation::Add, x, one, Dest::Fresh(Some("dead".into())), L)
            .unwrap();
        let a = fb
            .binop(Operation::Add, x, one, Dest::Fresh(Some("a".into())), L)
            .unwrap();
        let bb = fb
            .binop(Operation::Add, a, two, Dest::Fresh(Some("b".into())), L)
            .unwrap();
        fb.binop(Operation::Add, bb, three, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let plan = ir.last_use_plan(f);
    for (mid, morph) in ir.morphisms() {
        if morph.op == Operation::Add && morph.target != dead {
            adds.push(mid);
        }
    }
    adds.sort_by_key(|&m| plan.position(m).unwrap());
    let (add1, add2, add3) = (adds[0], adds[1], adds[2]);
    let a = ir.morphism(add1).unwrap().target;
    let bb = ir.morphism(add2).unwrap().target;

    // death is the consuming Add's position (through the operand-product
    // retention pin: a's handle lives until the product's own last use).
    assert_eq!(plan.death(a), Some(plan.position(add2).unwrap()));
    assert_eq!(plan.death(bb), Some(plan.position(add3).unwrap()));
    assert!(plan.dead_after(a, plan.position(add2).unwrap()));
    assert!(!plan.dead_after(a, plan.position(add1).unwrap()));
    // The Parameter escapes (rule 2).
    assert!(plan.escapes(ir.func(f).unwrap().input));
    assert_eq!(plan.death(ir.func(f).unwrap().input), None);
    // A use-less object: no use position (death None) but dead everywhere.
    assert_eq!(plan.death(dead), None);
    assert!(plan.dead_after(dead, 0));
}

#[test]
fn last_use_diamond_greatest_use_position() {
    // c -> a = c+1; c -> b = c+2; a, b -> s = a+b -> ret.
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", Ty::Unit, Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let c = fb.constant(Value::I32(5), L).unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        let two = fb.constant(Value::I32(2), L).unwrap();
        let a = fb
            .binop(Operation::Add, c, one, Dest::Fresh(Some("a".into())), L)
            .unwrap();
        let bb = fb
            .binop(Operation::Add, c, two, Dest::Fresh(Some("b".into())), L)
            .unwrap();
        fb.binop(Operation::Add, a, bb, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let plan = ir.last_use_plan(f);
    let mut adds: Vec<mapal_ir::MorphismId> = ir
        .morphisms()
        .filter(|(_, m)| m.op == Operation::Add)
        .map(|(id, _)| id)
        .collect();
    adds.sort_by_key(|&m| plan.position(m).unwrap());
    let (a1, a2, sum) = (adds[0], adds[1], adds[2]);
    let a = ir.morphism(a1).unwrap().target;
    let c = ir.morphism(a1).unwrap().source; // the (c, 1) operand product
    let c_obj = ir
        .in_edges(c)
        .iter()
        .map(|&m| ir.morphism(m).unwrap().source)
        .find(|&o| ir.object(o).unwrap().kind == mapal_ir::ObjectKind::Constant)
        .unwrap();

    // The shared feeder dies at the LATER of its two uses.
    let later = plan.position(a1).unwrap().max(plan.position(a2).unwrap());
    assert_eq!(plan.death(c_obj), Some(later));
    assert!(!plan.dead_after(c_obj, plan.position(a1).unwrap()));
    // Each arm dies at the Sum.
    assert_eq!(plan.death(a), Some(plan.position(sum).unwrap()));
    assert!(plan.dead_after(a, plan.position(sum).unwrap()));
}

#[test]
fn last_use_escape_via_pair_field() {
    // A locally-built array packed into a returned product escapes (rule 2's
    // "through Pair fields"): death ⊥, never dead_after.
    let arr2 = Ty::Array {
        elem: Box::new(Ty::i32()),
        size: 2,
    };
    let ret_ty = Ty::Tuple(vec![arr2.clone(), Ty::i32()]);
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "mk", Ty::i32(), ret_ty, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let x = fb.input();
        let one = fb.constant(Value::I32(1), L).unwrap();
        let two = fb.constant(Value::I32(2), L).unwrap();
        let a = fb
            .pack_array(&[one, two], Dest::Fresh(Some("a".into())), L)
            .unwrap();
        fb.pack(&[a, x], Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let plan = ir.last_use_plan(f);
    let a = ir
        .objects()
        .find(|(_, o)| o.name.as_deref() == Some("a"))
        .map(|(id, _)| id)
        .unwrap();
    assert!(plan.escapes(a));
    assert_eq!(plan.death(a), None, "escaping ⇒ death ⊥");
    assert!(!plan.dead_after(a, 0));
    assert!(!plan.dead_after(a, u32::MAX));
}

#[test]
fn last_use_two_sequential_loops_own_carried_sets() {
    // The S12 attribution trap: two sequential canonical loops in one fn —
    // each loop's carried set is its own, never the per-function union.
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "f", Ty::i32(), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let n = fb.input();
        let zero = fb.constant(Value::I32(0), L).unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        let two = fb.constant(Value::I32(2), L).unwrap();
        let lh1 = fb.begin_loop(zero, L).unwrap();
        let m1 = fb.merge_of(&lh1);
        let c1 = fb
            .binop(Operation::Lt, m1, n, Dest::Fresh(None), L)
            .unwrap();
        let i1 = fb
            .binop(Operation::Add, m1, one, Dest::Fresh(Some("i1".into())), L)
            .unwrap();
        fb.loop_back(&lh1, i1, c1, L).unwrap();
        let aa = fb
            .loop_exit(&lh1, m1, c1, Dest::Fresh(Some("aa".into())), L)
            .unwrap();
        fb.end_loop(lh1).unwrap();

        let lh2 = fb.begin_loop(zero, L).unwrap();
        let m2 = fb.merge_of(&lh2);
        let c2 = fb
            .binop(Operation::Lt, m2, n, Dest::Fresh(None), L)
            .unwrap();
        let j2 = fb
            .binop(Operation::Add, m2, two, Dest::Fresh(Some("j2".into())), L)
            .unwrap();
        fb.loop_back(&lh2, j2, c2, L).unwrap();
        let bb = fb
            .loop_exit(&lh2, m2, c2, Dest::Fresh(Some("bb".into())), L)
            .unwrap();
        fb.end_loop(lh2).unwrap();

        fb.binop(Operation::Add, aa, bb, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    let plan = ir.last_use_plan(f);
    let obj_named = |name: &str| {
        ir.objects()
            .find(|(_, o)| o.name.as_deref() == Some(name))
            .map(|(id, _)| id)
            .unwrap()
    };
    let merges: Vec<mapal_ir::ObjectId> = ir
        .objects()
        .filter(|(_, o)| o.kind == mapal_ir::ObjectKind::LoopMerge)
        .map(|(id, _)| id)
        .collect();
    let (m1, m2) = (merges[0], merges[1]);
    let (i1, j2) = (obj_named("i1"), obj_named("j2"));

    // Each loop's carried set is its own.
    assert_eq!(plan.carried_by(i1), Some(m1));
    assert_eq!(plan.carried_by(j2), Some(m2));
    assert_ne!(plan.carried_by(i1), Some(m2));
    assert_ne!(plan.carried_by(j2), Some(m1));
    // merge1 does not escape through its own exit even though its value
    // flows (via the exit object) into the post-loop Add — which CONSUMES it
    // (a scalar; computational edges carry no escape, only retention edges
    // do — rule 2's "through Pair fields and Phi arms").
    assert!(!plan.escapes(m1));
    assert!(!plan.escapes(obj_named("aa")));
    assert!(!plan.escapes(m2));
    // Rule 1 per loop: LoopBack past all of ITS loop's body morphisms.
    let backs: Vec<mapal_ir::MorphismId> = ir
        .morphisms()
        .filter(|(_, m)| m.op == Operation::LoopBack)
        .map(|(id, _)| id)
        .collect();
    let (lb1, lb2) = (backs[0], backs[1]);
    let add_of = |tgt: mapal_ir::ObjectId| {
        ir.in_edges(tgt)
            .iter()
            .map(|&m| ir.morphism(m).unwrap().id)
            .find(|&m| ir.morphism(m).unwrap().op == Operation::Add)
            .unwrap()
    };
    assert!(
        plan.position(lb1).unwrap() > plan.position(add_of(i1)).unwrap(),
        "loop1's LoopBack past its body"
    );
    assert!(
        plan.position(lb2).unwrap() > plan.position(add_of(j2)).unwrap(),
        "loop2's LoopBack past its body"
    );
}

/// S65 `view_death`: a projection of a product re-exposes the bytes the
/// product holds. `a` packed into `p`, `x = p.0` read after `a` is updated:
/// `dead_after` (copy semantics — `x` is its own bytes) lets the update write
/// `a` in place, `view_dead_after` (view semantics — `x` is `a`'s bytes) does
/// not. A projection of a scalar component is no view.
#[test]
fn view_death_follows_projections() {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", Ty::i32(), Ty::i32(), L)
        .unwrap();
    let (a, n);
    {
        let mut fb = b.build_fn(f).unwrap();
        let four = fb.constant(Value::I32(4), L).unwrap();
        a = fb.iota(four, Dest::Fresh(None), L).unwrap();
        n = fb.input();
        let p = fb.pack(&[a, n], Dest::Fresh(None), L).unwrap();
        let x = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
        let s = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
        let z = fb.constant(Value::I32(0), L).unwrap();
        let five = fb.constant(Value::I32(5), L).unwrap();
        let u = fb.update(a, z, five, Dest::Fresh(None), L).unwrap();
        let e = fb.index(x, z, Dest::Fresh(None), L).unwrap();
        let e2 = fb.index(u, z, Dest::Fresh(None), L).unwrap();
        let t = fb
            .binop(Operation::Add, e, e2, Dest::Fresh(None), L)
            .unwrap();
        let r = fb.binop(Operation::Add, t, s, Dest::Ret { slot: None }, L);
        r.unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let plan = ir.last_use_plan(f);
    let at = plan.position(one_op(&ir, Operation::Update)).unwrap();
    assert!(
        plan.dead_after(a, at),
        "copy semantics: x holds its own bytes"
    );
    assert!(
        !plan.view_dead_after(a, at),
        "view semantics: x is a's bytes"
    );
    assert!(plan.view_death(a) > plan.death(a));
    // The scalar's projection pins nothing: n's view death is its death.
    assert_eq!(plan.view_death(n), plan.death(n));
}

/// S65 `view_death` through a loop: the init array dies at the `LoopEnter`
/// under copy semantics, but its bytes are the merge's first state and,
/// through the exit, the exit object's — so they live as long as the exit
/// object is read.
#[test]
fn view_death_init_lives_through_the_loop_exit() {
    let (ir, f) = carried_array_loop();
    let plan = ir.last_use_plan(f);
    let enter_m = one_op(&ir, Operation::LoopEnter);
    let init = ir.morphism(enter_m).unwrap().source;
    let exit_m = one_op(&ir, Operation::LoopExit);
    let merge = ir.morphism(enter_m).unwrap().target;
    // The exit object is the fn's return: escaping, so the init's bytes do
    // too (the zero-trip path returns them) — the plan's escape agrees.
    assert!(plan.escapes(init) && plan.view_death(init).is_none());
    // The merge's own bytes die in the trip: its exit is no pin.
    let upd = plan.position(one_op(&ir, Operation::Update)).unwrap();
    assert_eq!(plan.view_death(merge), plan.death(merge));
    assert!(plan.view_death(merge).unwrap() <= upd);
    assert!(plan.position(exit_m).unwrap() < upd);
}

#[test]
fn last_use_determinism_run_twice_equal() {
    let (ir, f) = carried_array_loop();
    assert_eq!(ir.last_use_plan(f), ir.last_use_plan(f));
    let (ir, f) = borrowed_init_loop();
    assert_eq!(ir.last_use_plan(f), ir.last_use_plan(f));
}

#[test]
fn last_use_total_on_noncanonical_and_deep_graphs() {
    // The fused multi-merge graph (legal-but-degenerate): loop_plan is None
    // for both merges, so the plan is all-conservative — no carried set, no
    // re-ranking, full escape reachability — and must not panic (rule 6).
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "nested", Ty::i32(), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let c0 = fb.constant(Value::I32(0), L).unwrap();
        let c1 = fb.constant(Value::I32(0), L).unwrap();
        let outer = fb.begin_loop(c0, L).unwrap();
        let om = fb.merge_of(&outer);
        let inner = fb.begin_loop(c1, L).unwrap();
        let im = fb.merge_of(&inner);
        let ct = fb.constant(Value::Bool(true), L).unwrap();
        let cf = fb.constant(Value::Bool(false), L).unwrap();
        let inext = fb
            .binop(Operation::Add, om, im, Dest::Fresh(Some("inext".into())), L)
            .unwrap();
        fb.loop_back(&inner, inext, ct, L).unwrap();
        let onext = fb
            .binop(Operation::Add, im, om, Dest::Fresh(Some("onext".into())), L)
            .unwrap();
        fb.loop_back(&outer, onext, ct, L).unwrap();
        let _iout = fb
            .loop_exit(&inner, im, cf, Dest::Fresh(Some("iout".into())), L)
            .unwrap();
        fb.end_loop(inner).unwrap();
        fb.loop_exit(&outer, om, cf, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.end_loop(outer).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let plan = ir.last_use_plan(f);
    // No canonical plan ⇒ no carried classification (consumers fall back).
    for (o, _) in ir.objects() {
        assert_eq!(plan.carried_by(o), None, "non-canonical: nothing carried");
    }
    // Every morphism still has a position (the plain topo ranking).
    for (m, _) in ir.morphisms() {
        assert!(plan.position(m).is_some());
    }
    // The OUTER merge reaches the Return through retention edges (its exit
    // payload is the Return) ⇒ escapes (rule 2, conservative); the inner's
    // exit object is never consumed, so the inner merge genuinely does not
    // escape (its exit value is dropped). (loop_structure reports the merges
    // in insertion order: outer, then inner.)
    let (om, im) = (
        ir.loop_structure(f)[0].merges[0],
        ir.loop_structure(f)[0].merges[1],
    );
    assert!(plan.escapes(om));
    assert!(!plan.escapes(im));

    // The §16 deep chain (100k): the sweep stays linear and stackless (J1).
    let (ir, f) = {
        let n = 100_000usize;
        let mut b = IrBuilder::new();
        let f = b
            .declare(FuncKind::Named, "main", Ty::Unit, Ty::i32(), L)
            .unwrap();
        {
            let mut fb = b.build_fn(f).unwrap();
            let one = fb.constant(Value::I32(1), L).unwrap();
            let mut acc = fb.constant(Value::I32(0), L).unwrap();
            for i in 0..n {
                let dest = if i + 1 == n {
                    Dest::Ret { slot: None }
                } else {
                    Dest::Fresh(None)
                };
                acc = fb.binop(Operation::Add, acc, one, dest, L).unwrap();
            }
            fb.finish().unwrap();
        }
        (b.seal(f).unwrap(), f)
    };
    let plan = ir.last_use_plan(f);
    // The chain's second-to-last accumulator dies at the final Add.
    let mut adds: Vec<mapal_ir::MorphismId> = ir
        .morphisms()
        .filter(|(_, m)| m.op == Operation::Add)
        .map(|(id, _)| id)
        .collect();
    adds.sort_by_key(|&m| plan.position(m).unwrap());
    let last = *adds.last().unwrap();
    let penultimate_acc = ir.morphism(adds[adds.len() - 2]).unwrap().target;
    assert_eq!(
        plan.death(penultimate_acc),
        Some(plan.position(last).unwrap())
    );
}

#[test]
fn last_use_total_on_alg_suite_graphs() {
    // Totality sweep: every graph this suite builds plans without panic
    // (the mapal-rewrite testgen generator is NOT reachable from mapal-ir's
    // tests — it imports mapal_interp, which is downstream of mapal-ir — so
    // the plan §6.1 brute-force agreement row lives with the consumers).
    let (ir, f) = counting_loop();
    let _ = ir.last_use_plan(f);
    let (ir, f) = carried_array_loop();
    let _ = ir.last_use_plan(f);
    let (ir, f) = borrowed_init_loop();
    let _ = ir.last_use_plan(f);
    // And every fn of a multi-fn module: plan per fn, independently.
    let mut b = IrBuilder::new();
    let g = b
        .declare(FuncKind::Named, "g", Ty::i32(), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(g).unwrap();
        let x = fb.input();
        fb.output(x, None, L).unwrap();
        fb.finish().unwrap();
    }
    let f = b
        .declare(FuncKind::Named, "main", Ty::Unit, Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        fb.call(g, one, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    for (id, _) in ir.funcs() {
        let _ = ir.last_use_plan(id);
    }
}
