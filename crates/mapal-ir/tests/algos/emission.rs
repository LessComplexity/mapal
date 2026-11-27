//! Topo order around clock reads and `emission_plan` tests.

use super::*;

/// plan-s53 `clock-fence`, the GPU e2e shape (`benches/gpu/gen_tiles.py e2e`):
/// generate `a` and `b` from `ta`, `t0`, the site `ta -> map {a[t]}` with `a`
/// and `b` captured, `t1`, print. The site's source is the capture pack, written
/// on the site's line, so Kahn releases `t1` (its token is ready one line
/// earlier) BEFORE the site — and `t0` before the generation. That was S52's
/// `e2e_4096_c.host.ll`: `t1 - t0` timed the generation, and nothing of the site.
#[test]
fn topo_order_puts_the_site_between_the_clock_reads_and_the_generation_before() {
    let at = |line: u32| SourceLoc {
        start: line * 10,
        end: line * 10 + 9,
    };
    let arr = Ty::Array {
        elem: Box::new(Ty::i32()),
        size: 8,
    };
    let mut b = IrBuilder::new();
    let gen_body = identity_map_body(&mut b);
    let site_body = b
        .declare(
            FuncKind::MapBody,
            "site",
            Ty::Tuple(vec![arr.clone(), arr.clone(), Ty::i32()]),
            Ty::i32(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(site_body).unwrap();
        let input = fb.input();
        let a = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let t = fb.proj(input, 2, Dest::Fresh(None), L).unwrap();
        let cell = fb.index(a, t, Dest::Fresh(None), L).unwrap();
        fb.output(cell, None, L).unwrap();
        fb.finish().unwrap();
    }
    let f = b
        .declare(FuncKind::Named, "e2e", Ty::IoToken, Ty::IoToken, at(0))
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let token = fb.input();
        let eight = fb.constant(Value::I32(8), at(1)).unwrap();
        let ta = fb.iota(eight, Dest::Fresh(None), at(1)).unwrap();
        let a = fb.map(gen_body, ta, Dest::Fresh(None), at(2)).unwrap();
        let bb = fb.map(gen_body, ta, Dest::Fresh(None), at(3)).unwrap();
        let start = fb.time_ms(token, at(4)).unwrap();
        let token = fb.proj(start, 0, Dest::Fresh(None), at(4)).unwrap();
        let t0 = fb.proj(start, 1, Dest::Fresh(None), at(4)).unwrap();
        let c = fb
            .map_captured(site_body, &[a, bb], ta, Dest::Fresh(None), at(5))
            .unwrap();
        let stop = fb.time_ms(token, at(6)).unwrap();
        let token = fb.proj(stop, 0, Dest::Fresh(None), at(6)).unwrap();
        let t1 = fb.proj(stop, 1, Dest::Fresh(None), at(6)).unwrap();
        let zero = fb.constant(Value::I32(0), at(7)).unwrap();
        let head = fb.index(c, zero, Dest::Fresh(None), at(7)).unwrap();
        let token = fb.print(token, head, at(7)).unwrap();
        let elapsed = fb
            .binop(Operation::Sub, t1, t0, Dest::Fresh(None), at(8))
            .unwrap();
        let token = fb.print(token, elapsed, at(8)).unwrap();
        fb.output(token, None, at(9)).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));

    let order = ir.topo_order(f);
    let at_line = |line: u32| -> Vec<u32> {
        order
            .iter()
            .enumerate()
            .filter(|&(_, &m)| ir.morphism(m).unwrap().loc.start == line * 10)
            .map(|(i, _)| i as u32)
            .collect()
    };
    let reads = func_ops(&ir, f, |op| op == Operation::TimeMs);
    let (r0, r1) = (topo_pos(&ir, f, reads[0]), topo_pos(&ir, f, reads[1]));
    // The site: its capture pack and the Map itself, every one inside.
    let site = at_line(5);
    assert!(site.len() >= 2, "pack edges and the map: {site:?}");
    assert!(
        site.iter().all(|&p| r0 < p && p < r1),
        "the site lies between t0 ({r0}) and t1 ({r1}): {site:?}"
    );
    // The generation is written above t0, so it is outside the bracket.
    let generation: Vec<u32> = [1, 2, 3].into_iter().flat_map(at_line).collect();
    assert!(
        generation.iter().all(|&p| p < r0),
        "the generation precedes t0 ({r0}): {generation:?}"
    );
    // The readout is written below t1.
    assert!(at_line(7).iter().all(|&p| r1 < p), "the readout follows t1");
}

#[test]
fn emission_nested_product_dissolution_is_pair_built_only() {
    // inner = (x, 1); outer = (inner, 1); back = outer.0; ret back.1
    // outer: Pair-built, only-Proj consumers -> Dissolved.
    // inner: its consumer is a Pair edge (a field of outer) -> never
    // Dissolved; one effective consumer through the dissolved outer ->
    // product-typed Inline (the compound-literal case WP-B must handle).
    // back: Proj-PRODUCED tuple -> not Pair-built -> never Dissolved even
    // with only-Proj consumers; single consumer -> Inline.
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "nested", Ty::i32(), Ty::i32(), L)
        .unwrap();
    let inner;
    let outer;
    let back;
    {
        let mut fb = b.build_fn(f).unwrap();
        let x = fb.input();
        let one = fb.constant(Value::I32(1), L).unwrap();
        inner = fb.pack(&[x, one], Dest::Fresh(None), L).unwrap();
        outer = fb.pack(&[inner, one], Dest::Fresh(None), L).unwrap();
        back = fb.proj(outer, 0, Dest::Fresh(None), L).unwrap();
        fb.proj(back, 1, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let plan = ir.emission_plan(f);
    assert!(
        plan.class(outer).is_some_and(|c| c.is_dissolved()),
        "outer: Pair-built + only-Proj -> Dissolved"
    );
    assert!(
        plan.class(inner).is_some_and(|c| c.is_inline()),
        "inner: Pair-edge consumer blocks dissolution; effective count 1"
    );
    assert!(
        plan.class(back).is_some_and(|c| c.is_inline()),
        "back: Proj-produced, not dissolvable; single consumer -> Inline"
    );
}

#[test]
fn emission_proj_produced_tuple_fanout_is_named_not_dropped() {
    // The R-NODUP regression for the silent-count-drop bug: a Proj-produced
    // tuple read by TWO Projs must be Named. If it dissolved, its consumers
    // would vanish from the counts (pair_slot_source = None on a Proj-built
    // product), the shared computed field chain would classify Inline, and
    // the emitter would duplicate its text.
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "projfan", Ty::i32(), Ty::i32(), L)
        .unwrap();
    let s;
    let back;
    {
        let mut fb = b.build_fn(f).unwrap();
        let x = fb.input();
        let one = fb.constant(Value::I32(1), L).unwrap();
        s = fb
            .binop(Operation::Add, x, one, Dest::Fresh(None), L)
            .unwrap();
        let inner = fb.pack(&[s, one], Dest::Fresh(None), L).unwrap();
        let outer = fb.pack(&[inner, one], Dest::Fresh(None), L).unwrap();
        back = fb.proj(outer, 0, Dest::Fresh(None), L).unwrap();
        let r1 = fb.proj(back, 0, Dest::Fresh(None), L).unwrap();
        let r2 = fb.proj(back, 1, Dest::Fresh(None), L).unwrap();
        fb.binop(Operation::Add, r1, r2, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let plan = ir.emission_plan(f);
    assert!(
        plan.class(back).is_some_and(|c| c.is_named()),
        "Proj-produced tuple with 2 consumers must be Named (count intact)"
    );
    assert!(
        plan.class(s).is_some_and(|c| c.is_inline()),
        "the computed field chain stays single-reference"
    );
}

#[test]
fn emission_pure_chain_dissolves_products_and_inlines_values() {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "chain", Ty::i32(), Ty::i32(), L)
        .unwrap();
    let input;
    let one;
    let mut intermediates = Vec::new();
    {
        let mut fb = b.build_fn(f).unwrap();
        input = fb.input();
        one = fb.constant(Value::I32(1), L).unwrap();
        let mut value = input;
        for i in 0..4 {
            value = fb
                .binop(
                    Operation::Add,
                    value,
                    one,
                    if i == 3 {
                        Dest::Ret { slot: None }
                    } else {
                        Dest::Fresh(None)
                    },
                    L,
                )
                .unwrap();
            if i != 3 {
                intermediates.push(value);
            }
        }
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let plan = ir.emission_plan(f);
    let products: Vec<_> = ir
        .objects()
        .filter(|(o, obj)| ir.owner(*o) == f && obj.ty.product_arity().is_some())
        .map(|(o, _)| o)
        .collect();

    assert_eq!(products.len(), 4, "one internal product per binary op");
    assert!(
        products
            .iter()
            .all(|&o| plan.class(o).is_some_and(|c| c.is_dissolved()))
    );
    assert!(plan.class(input).is_some_and(|c| c.is_inline()));
    assert!(
        intermediates
            .iter()
            .all(|&o| plan.class(o).is_some_and(|c| c.is_inline()))
    );
    assert!(
        plan.class(ir.func(f).unwrap().output)
            .is_some_and(|c| c.is_named())
    );
    assert!(plan.class(one).is_none(), "constants never materialize");
}

#[test]
fn emission_fanout_has_one_named_split_point() {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "fanout", Ty::i32(), Ty::i32(), L)
        .unwrap();
    let shared;
    {
        let mut fb = b.build_fn(f).unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        shared = fb
            .binop(Operation::Add, fb.input(), one, Dest::Fresh(None), L)
            .unwrap();
        let two = fb.constant(Value::I32(2), L).unwrap();
        let three = fb.constant(Value::I32(3), L).unwrap();
        let left = fb
            .binop(Operation::Mul, shared, two, Dest::Fresh(None), L)
            .unwrap();
        let right = fb
            .binop(Operation::Sub, shared, three, Dest::Fresh(None), L)
            .unwrap();
        fb.binop(Operation::Add, left, right, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let plan = ir.emission_plan(f);
    let output = ir.func(f).unwrap().output;
    let named_non_output: Vec<_> = ir
        .objects()
        .filter(|(o, obj)| {
            ir.owner(*o) == f
                && *o != output
                && obj.kind != mapal_ir::ObjectKind::Constant
                && plan.class(*o).is_some_and(|c| c.is_named())
        })
        .map(|(o, _)| o)
        .collect();

    assert_eq!(named_non_output, vec![shared]);
    assert_eq!(
        ir.out_edges(shared)
            .iter()
            .filter(|&&m| matches!(ir.morphism(m).unwrap().op, Operation::Pair { .. }))
            .count(),
        2,
        "both consumer products reference the one split value"
    );
}

#[test]
fn emission_div_guard_classification_uses_divisor_facts() {
    let mut b = IrBuilder::new();
    let unsafe_f = b
        .declare(FuncKind::Named, "unsafe_div", Ty::i32(), Ty::i32(), L)
        .unwrap();
    let safe_f = b
        .declare(FuncKind::Named, "safe_div", Ty::i32(), Ty::i32(), L)
        .unwrap();
    let neg_one_f = b
        .declare(FuncKind::Named, "neg_one_div", Ty::i32(), Ty::i32(), L)
        .unwrap();
    let float_f = b
        .declare(FuncKind::Named, "float_div", Ty::f32(), Ty::f32(), L)
        .unwrap();
    let unsafe_div;
    let safe_div;
    let neg_one_div;
    let float_div;
    {
        let mut fb = b.build_fn(unsafe_f).unwrap();
        let twelve = fb.constant(Value::I32(12), L).unwrap();
        unsafe_div = fb
            .binop(Operation::Div, twelve, fb.input(), Dest::Fresh(None), L)
            .unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        fb.binop(Operation::Add, unsafe_div, one, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    {
        let mut fb = b.build_fn(safe_f).unwrap();
        let two = fb.constant(Value::I32(2), L).unwrap();
        safe_div = fb
            .binop(Operation::Div, fb.input(), two, Dest::Fresh(None), L)
            .unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        fb.binop(Operation::Add, safe_div, one, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    {
        let mut fb = b.build_fn(neg_one_f).unwrap();
        let minus_one = fb.constant(Value::I32(-1), L).unwrap();
        neg_one_div = fb
            .binop(Operation::Div, fb.input(), minus_one, Dest::Fresh(None), L)
            .unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        fb.binop(
            Operation::Add,
            neg_one_div,
            one,
            Dest::Ret { slot: None },
            L,
        )
        .unwrap();
        fb.finish().unwrap();
    }
    {
        let mut fb = b.build_fn(float_f).unwrap();
        let two = fb.constant(Value::F32(2.0), L).unwrap();
        float_div = fb
            .binop(Operation::Div, fb.input(), two, Dest::Fresh(None), L)
            .unwrap();
        let one = fb.constant(Value::F32(1.0), L).unwrap();
        fb.binop(Operation::Add, float_div, one, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(unsafe_f).unwrap();

    assert!(
        ir.emission_plan(unsafe_f)
            .class(unsafe_div)
            .is_some_and(|c| c.is_named())
    );
    assert!(
        ir.emission_plan(safe_f)
            .class(safe_div)
            .is_some_and(|c| c.is_inline())
    );
    assert!(
        ir.emission_plan(neg_one_f)
            .class(neg_one_div)
            .is_some_and(|c| c.is_named())
    );
    assert!(
        ir.emission_plan(float_f)
            .class(float_div)
            .is_some_and(|c| c.is_inline())
    );
}

#[test]
fn emission_index_guard_classification_uses_bounds_proof() {
    let array_ty = Ty::Array {
        elem: Box::new(Ty::i32()),
        size: 4,
    };
    let mut b = IrBuilder::new();
    let f = b
        .declare(
            FuncKind::Named,
            "indexes",
            Ty::Tuple(vec![array_ty, Ty::i32()]),
            Ty::i32(),
            L,
        )
        .unwrap();
    let proven;
    let unproven;
    {
        let mut fb = b.build_fn(f).unwrap();
        let input = fb.input();
        let array = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let dynamic = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        proven = fb.index(array, one, Dest::Fresh(None), L).unwrap();
        unproven = fb.index(array, dynamic, Dest::Fresh(None), L).unwrap();
        fb.binop(
            Operation::Add,
            proven,
            unproven,
            Dest::Ret { slot: None },
            L,
        )
        .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let plan = ir.emission_plan(f);

    assert!(plan.class(proven).is_some_and(|c| c.is_inline()));
    assert!(plan.class(unproven).is_some_and(|c| c.is_named()));
}

#[test]
fn emission_loop_merge_routes_and_cones_are_named() {
    let (ir, f) = counting_loop();
    let merge = ir.loop_structure(f)[0].merges[0];
    let loop_plan = ir.loop_plan(f, merge).unwrap();
    let plan = ir.emission_plan(f);
    let mut expected = vec![loop_plan.merge, loop_plan.back_route, loop_plan.exit_route];
    for &m in loop_plan
        .decide_order
        .iter()
        .chain(&loop_plan.advance_order)
        .chain(&loop_plan.exits)
    {
        let morph = ir.morphism(m).unwrap();
        expected.push(morph.source);
        expected.push(morph.target);
    }
    for o in expected {
        let obj = ir.object(o).unwrap();
        if obj.kind != mapal_ir::ObjectKind::Constant && !mapal_ir::ty_contains_token(&obj.ty) {
            assert!(
                plan.class(o).is_some_and(|c| c.is_named()),
                "loop object {o:?} must be Named"
            );
        }
    }
}

#[test]
fn emission_call_argument_product_and_array_literal_are_named() {
    let mut b = IrBuilder::new();
    let callee = b
        .declare(
            FuncKind::Named,
            "callee",
            Ty::Tuple(vec![Ty::i32(), Ty::i32()]),
            Ty::i32(),
            L,
        )
        .unwrap();
    let caller = b
        .declare(FuncKind::Named, "caller", Ty::Unit, Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(callee).unwrap();
        fb.proj(fb.input(), 0, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let call_arg;
    let array;
    {
        let mut fb = b.build_fn(caller).unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        let two = fb.constant(Value::I32(2), L).unwrap();
        call_arg = fb.pack(&[one, two], Dest::Fresh(None), L).unwrap();
        array = fb.pack_array(&[one, two], Dest::Fresh(None), L).unwrap();
        fb.call(callee, call_arg, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(caller).unwrap();
    let plan = ir.emission_plan(caller);

    assert!(plan.class(call_arg).is_some_and(|c| c.is_named()));
    assert!(plan.class(array).is_some_and(|c| c.is_named()));
}

pub(super) fn emission_effective_counts(
    ir: &mapal_ir::CategoryIr,
    f: mapal_ir::FuncId,
) -> Vec<(mapal_ir::ObjectId, u32)> {
    let plan = ir.emission_plan(f);
    let mut counts: Vec<_> = ir
        .objects()
        .filter(|(o, obj)| {
            ir.owner(*o) == f
                && obj.kind != mapal_ir::ObjectKind::Constant
                && !mapal_ir::ty_contains_token(&obj.ty)
                && !plan.class(*o).is_some_and(|c| c.is_dissolved())
        })
        .map(|(o, _)| (o, 0))
        .collect();
    let increment = |counts: &mut Vec<(mapal_ir::ObjectId, u32)>, o| {
        if let Some((_, count)) = counts.iter_mut().find(|(id, _)| *id == o) {
            *count += 1;
        }
    };

    for (o, _) in &counts.clone() {
        for &m in ir.out_edges(*o) {
            let morph = ir.morphism(m).unwrap();
            let transparent_pair = matches!(morph.op, Operation::Pair { .. })
                && plan.class(morph.target).is_some_and(|c| c.is_dissolved());
            if !transparent_pair {
                increment(&mut counts, *o);
            }
        }
    }
    for (product, _) in ir
        .objects()
        .filter(|(o, _)| ir.owner(*o) == f && plan.class(*o).is_some_and(|c| c.is_dissolved()))
    {
        for &m in ir.out_edges(product) {
            match ir.morphism(m).unwrap().op {
                Operation::Proj { index } => {
                    if let Some(source) = ir.in_edges(product).iter().find_map(|&pair| {
                        let morph = ir.morphism(pair).unwrap();
                        matches!(morph.op, Operation::Pair { slot, .. } if slot == index)
                            .then_some(morph.source)
                    }) {
                        increment(&mut counts, source);
                    }
                }
                _ => {
                    for &pair in ir.in_edges(product) {
                        let morph = ir.morphism(pair).unwrap();
                        if matches!(morph.op, Operation::Pair { .. }) {
                            increment(&mut counts, morph.source);
                        }
                    }
                }
            }
        }
    }
    counts
}

pub(super) fn emission_boundaries(
    ir: &mapal_ir::CategoryIr,
    f: mapal_ir::FuncId,
) -> Vec<mapal_ir::ObjectId> {
    let mut boundary = vec![ir.func(f).unwrap().output];
    let mut add = |o| {
        if !boundary.contains(&o) {
            boundary.push(o);
        }
    };
    for (o, obj) in ir.objects().filter(|(o, _)| ir.owner(*o) == f) {
        if obj.kind == mapal_ir::ObjectKind::LoopMerge || matches!(obj.ty, Ty::Array { .. }) {
            add(o);
        }
    }

    let bounds = ir.bounds_proof(f);
    for (m, morph) in ir.morphisms().filter(|(_, m)| ir.owner(m.source) == f) {
        let guarded = match morph.op {
            Operation::Div | Operation::Mod => {
                !matches!(ir.object(morph.target).unwrap().ty, Ty::Float { .. })
                    && ir
                        .in_edges(morph.source)
                        .iter()
                        .find_map(|&pair| {
                            let pair = ir.morphism(pair).unwrap();
                            matches!(pair.op, Operation::Pair { slot: 1, .. })
                                .then_some(pair.source)
                        })
                        .is_none_or(|o| {
                            let divisor = ir.object(o).unwrap();
                            divisor.kind != mapal_ir::ObjectKind::Constant
                                || !match divisor.value.as_ref() {
                                    Some(Value::I32(v)) => *v != 0 && *v != -1,
                                    Some(Value::I64(v)) => *v != 0 && *v != -1,
                                    Some(Value::U8(v)) => *v != 0,
                                    _ => false,
                                }
                        })
            }
            Operation::Index | Operation::Update => !bounds.proven(m),
            _ => false,
        };
        if guarded {
            add(morph.target);
        }
        match morph.op {
            Operation::Output => {
                add(morph.source);
                add(morph.target);
            }
            Operation::Call(_) => add(morph.source),
            Operation::Map { .. }
            | Operation::Fold { .. }
            | Operation::Zip
            | Operation::Enumerate
            | Operation::Update
            | Operation::Iota
            | Operation::Fill
                if ir
                    .object(morph.source)
                    .unwrap()
                    .ty
                    .product_arity()
                    .is_some() =>
            {
                add(morph.source);
            }
            Operation::LoopEnter | Operation::LoopBack | Operation::LoopExit => {
                add(morph.source);
                add(morph.target);
            }
            _ => {}
        }
    }
    for scc in ir.loop_structure(f) {
        for merge in scc.merges {
            if let Some(plan) = ir.loop_plan(f, merge) {
                for &m in plan.decide_order.iter().chain(&plan.advance_order) {
                    let morph = ir.morphism(m).unwrap();
                    add(morph.source);
                    add(morph.target);
                }
            }
        }
    }
    boundary
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

    #[test]
    fn emission_plan_totality_and_laws(
        ops in prop::collection::vec((0u8..5, any::<usize>(), any::<usize>()), 1..24)
    ) {
        let mut b = IrBuilder::new();
        let f = b.declare(FuncKind::Named, "generated", Ty::i32(), Ty::i32(), L).unwrap();
        {
            let mut fb = b.build_fn(f).unwrap();
            let mut pool = vec![
                fb.input(),
                fb.constant(Value::I32(0), L).unwrap(),
                fb.constant(Value::I32(1), L).unwrap(),
                fb.constant(Value::I32(-1), L).unwrap(),
                fb.constant(Value::I32(2), L).unwrap(),
            ];
            for (kind, a, c) in ops {
                let lhs = pool[a % pool.len()];
                let rhs = pool[c % pool.len()];
                let op = match kind {
                    0 => Operation::Add,
                    1 => Operation::Sub,
                    2 => Operation::Mul,
                    3 => Operation::Div,
                    _ => Operation::Mod,
                };
                pool.push(fb.binop(op, lhs, rhs, Dest::Fresh(None), L).unwrap());
            }
            fb.output(*pool.last().unwrap(), None, L).unwrap();
            fb.finish().unwrap();
        }
        let ir = b.seal(f).unwrap();
        let plan = ir.emission_plan(f);
        prop_assert_eq!(&plan, &ir.emission_plan(f), "R-DET");
        let counts = emission_effective_counts(&ir, f);
        let boundaries = emission_boundaries(&ir, f);

        for (o, obj) in ir.objects().filter(|(o, _)| ir.owner(*o) == f) {
            if obj.kind == mapal_ir::ObjectKind::Constant || mapal_ir::ty_contains_token(&obj.ty) {
                prop_assert!(plan.class(o).is_none());
                continue;
            }
            let class = plan.class(o).expect("classification is total");
            if class.is_inline() {
                let count = counts.iter().find(|(id, _)| *id == o).unwrap().1;
                prop_assert_eq!(count, 1, "R-NODUP for {:?}", o);
            }
            if class.is_dissolved() {
                prop_assert!(obj.ty.product_arity().is_some());
                prop_assert!(!boundaries.contains(&o), "boundary {:?}", o);
            }
            if boundaries.contains(&o) {
                prop_assert!(class.is_named(), "boundary {o:?} must be Named");
            }
        }
    }
}
