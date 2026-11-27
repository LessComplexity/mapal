//! `path_plan` tests: the parallel task DAG, host spine and checkpoints.

use super::*;

// --- path_plan (parallel task DAG + host-spine checkpoints) ------------------

pub(super) fn identity_map_body(b: &mut IrBuilder) -> mapal_ir::FuncId {
    let body = b
        .declare(FuncKind::MapBody, "identity", Ty::i32(), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(body).unwrap();
        fb.output(fb.input(), None, L).unwrap();
        fb.finish().unwrap();
    }
    body
}

pub(super) fn add_fold_body(b: &mut IrBuilder) -> mapal_ir::FuncId {
    let body = b
        .declare(
            FuncKind::FoldBody,
            "sum",
            Ty::Tuple(vec![Ty::i32(), Ty::i32()]),
            Ty::i32(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(body).unwrap();
        let input = fb.input();
        let acc = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let elem = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        fb.binop(Operation::Add, acc, elem, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    body
}

pub(super) fn task_for(plan: &PathPlan, morphism: MorphismId) -> usize {
    plan.tasks
        .iter()
        .position(|task| match &task.kind {
            TaskKind::Split { site, .. } => *site == morphism,
            TaskKind::Seq { morphisms } => morphisms.contains(&morphism),
        })
        .expect("morphism belongs to a task")
}

pub(super) fn func_ops(
    ir: &mapal_ir::CategoryIr,
    f: mapal_ir::FuncId,
    pred: impl Fn(Operation) -> bool,
) -> Vec<MorphismId> {
    ir.func(f)
        .unwrap()
        .morphisms
        .iter()
        .copied()
        .filter(|&m| pred(ir.morphism(m).unwrap().op))
        .collect()
}

pub(super) fn topo_pos(ir: &mapal_ir::CategoryIr, f: mapal_ir::FuncId, m: MorphismId) -> u32 {
    ir.topo_order(f).iter().position(|&site| site == m).unwrap() as u32
}

pub(super) fn diamond_path_fixture() -> (mapal_ir::CategoryIr, mapal_ir::FuncId) {
    let mut b = IrBuilder::new();
    let body = identity_map_body(&mut b);
    let output = Ty::Array {
        elem: Box::new(Ty::Tuple(vec![Ty::i32(), Ty::i32()])),
        size: 8,
    };
    let f = b
        .declare(FuncKind::Named, "diamond", i32_arr(8), output, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let input = fb.input();
        let left = fb.map(body, input, Dest::Fresh(None), L).unwrap();
        let right = fb.map(body, input, Dest::Fresh(None), L).unwrap();
        fb.zip(left, right, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    (ir, f)
}

#[test]
fn path_diamond_has_independent_maps_and_dataflow_join() {
    let (ir, f) = diamond_path_fixture();
    let plan = ir.path_plan(f);
    let maps = func_ops(&ir, f, |op| matches!(op, Operation::Map { .. }));
    let zip = func_ops(&ir, f, |op| op == Operation::Zip)[0];
    assert_eq!(maps.len(), 2);

    let left = task_for(&plan, maps[0]);
    let right = task_for(&plan, maps[1]);
    let join = task_for(&plan, zip);
    assert_ne!(left, right);
    assert!(plan.tasks[left].deps.is_empty());
    assert!(plan.tasks[right].deps.is_empty());

    // Zip consumes its Pair-built product; that scalar glue task depends on
    // both independent map producers.
    assert_eq!(plan.tasks[join].deps.len(), 1);
    let glue = plan.tasks[join].deps[0];
    assert_eq!(plan.tasks[glue].deps, vec![left, right]);
    assert!(!plan.is_single_path());
}

/// plan-s49. Every consumer of a `PathPlan` needs it to be a DAG; the backend
/// silently demotes a cyclic plan to a fully sequential emission
/// (`llvm/src/lib.rs::path_plan_is_acyclic`), and `path_plan`'s own
/// `debug_assert` is compiled out of release. So assert it here.
pub(super) fn assert_acyclic(plan: &PathPlan) {
    let mut remaining: Vec<usize> = plan.tasks.iter().map(|t| t.deps.len()).collect();
    let mut order: Vec<usize> = remaining
        .iter()
        .enumerate()
        .filter_map(|(id, &n)| (n == 0).then_some(id))
        .collect();
    let mut cursor = 0;
    while cursor < order.len() {
        let before = order[cursor];
        cursor += 1;
        for (after, task) in plan.tasks.iter().enumerate() {
            if task.deps.contains(&before) {
                remaining[after] -= 1;
                if remaining[after] == 0 {
                    order.push(after);
                }
            }
        }
    }
    assert_eq!(
        order.len(),
        plan.tasks.len(),
        "path_plan produced a CYCLIC task graph; {} of {} tasks are unreachable",
        plan.tasks.len() - order.len(),
        plan.tasks.len()
    );
}

/// plan-s49 regression. Two bulk ops chained through a scalar glue that BOTH
/// read one shared object — the shape of every multi-layer net, where each
/// matmul's capture tuple captures the same `k` range. Grouping scalar
/// morphisms by "shares an object in any role" fused the two glue groups into
/// one task, which then sat both before and after the first bulk op: a cycle,
/// and the whole program fell back to sequential emission. Glue is grouped by
/// dataflow and co-writing only, so the two groups stay apart.
#[test]
fn path_chained_bulk_ops_do_not_fuse_their_scalar_glue() {
    let mut b = IrBuilder::new();
    let fold_body = add_fold_body(&mut b);
    let f = b
        .declare(FuncKind::Named, "chained", i32_arr(8), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let input = fb.input();
        let zero = fb.constant(Value::I32(0), L).unwrap();
        // Glue 1 and glue 2 share a read of `input` and of nothing else.
        let first_in = fb.pack(&[zero, input], Dest::Fresh(None), L).unwrap();
        let first = fb.fold(fold_body, first_in, Dest::Fresh(None), L).unwrap();
        let second_in = fb.pack(&[first, input], Dest::Fresh(None), L).unwrap();
        fb.fold(fold_body, second_in, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let plan = ir.path_plan(f);
    assert_acyclic(&plan);

    let folds = func_ops(&ir, f, |op| matches!(op, Operation::Fold { .. }));
    assert_eq!(folds.len(), 2);
    let pairs = func_ops(&ir, f, |op| matches!(op, Operation::Pair { .. }));
    let glue1 = task_for(&plan, pairs[0]);
    let glue2 = task_for(&plan, *pairs.last().expect("pair morphisms"));
    assert_ne!(
        glue1, glue2,
        "sibling readers of one object were fused into a single scalar task"
    );

    // The chain is intact and one-directional: fold1 -> glue2 -> fold2.
    let fold1 = task_for(&plan, folds[0]);
    let fold2 = task_for(&plan, folds[1]);
    assert!(plan.tasks[fold1].deps.contains(&glue1));
    assert!(plan.tasks[glue2].deps.contains(&fold1));
    assert!(plan.tasks[fold2].deps.contains(&glue2));
    assert!(!plan.tasks[glue1].deps.contains(&fold1));
}

#[test]
fn path_fold_and_independent_map_have_no_edge() {
    let mut b = IrBuilder::new();
    let map_body = identity_map_body(&mut b);
    let fold_body = add_fold_body(&mut b);
    let f = b
        .declare(FuncKind::Named, "fork", i32_arr(8), i32_arr(8), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let input = fb.input();
        let zero = fb.constant(Value::I32(0), L).unwrap();
        let fold_input = fb.pack(&[zero, input], Dest::Fresh(None), L).unwrap();
        fb.fold(fold_body, fold_input, Dest::Fresh(None), L)
            .unwrap();
        fb.map(map_body, input, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let plan = ir.path_plan(f);
    let fold = task_for(
        &plan,
        func_ops(&ir, f, |op| matches!(op, Operation::Fold { .. }))[0],
    );
    let map = task_for(
        &plan,
        func_ops(&ir, f, |op| matches!(op, Operation::Map { .. }))[0],
    );

    assert!(matches!(
        plan.tasks[fold].kind,
        TaskKind::Seq { ref morphisms } if morphisms.len() == 1
    ));
    assert!(matches!(plan.tasks[map].kind, TaskKind::Split { n: 8, .. }));
    assert!(!plan.tasks[fold].deps.contains(&map));
    assert!(!plan.tasks[map].deps.contains(&fold));
}

#[test]
fn path_loop_scc_is_one_seq_task() {
    let (ir, f) = counting_loop();
    let plan = ir.path_plan(f);
    assert_eq!(plan.tasks.len(), 1);
    assert_eq!(
        plan.tasks[0].kind,
        TaskKind::Seq {
            morphisms: ir.topo_order(f)
        }
    );
    assert!(plan.is_single_path());
}

#[test]
fn path_effectful_loop_stays_entirely_on_host_spine() {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "countdown", Ty::IoToken, Ty::IoToken, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let token = fb.input();
        let three = fb.constant(Value::I32(3), L).unwrap();
        let init = fb.pack(&[three, token], Dest::Fresh(None), L).unwrap();
        let lh = fb.begin_loop(init, L).unwrap();
        let merge = fb.merge_of(&lh);
        let i = fb.proj(merge, 0, Dest::Fresh(None), L).unwrap();
        let token = fb.proj(merge, 1, Dest::Fresh(None), L).unwrap();
        let token = fb.print(token, i, L).unwrap();
        let zero = fb.constant(Value::I32(0), L).unwrap();
        let cond = fb
            .binop(Operation::Gt, i, zero, Dest::Fresh(None), L)
            .unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        let next_i = fb
            .binop(Operation::Sub, i, one, Dest::Fresh(None), L)
            .unwrap();
        let next = fb.pack(&[next_i, token], Dest::Fresh(None), L).unwrap();
        fb.loop_back(&lh, next, cond, L).unwrap();
        let token = fb
            .loop_exit(&lh, token, cond, Dest::Fresh(None), L)
            .unwrap();
        fb.end_loop(lh).unwrap();
        fb.output(token, None, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let print = func_ops(&ir, f, |op| matches!(op, Operation::Print { .. }))[0];
    let plan = ir.path_plan(f);

    assert!(plan.tasks.is_empty());
    assert_eq!(plan.checkpoints.len(), 2);
    assert_eq!(plan.checkpoints[0].topo, topo_pos(&ir, f, print));
    assert!(plan.checkpoints[0].wait.is_empty());
    assert_eq!(plan.checkpoints[1].topo, u32::MAX);
    assert!(plan.checkpoints[1].wait.is_empty());
}

#[test]
fn path_print_waits_for_value_producer_and_every_earlier_trap() {
    let mut b = IrBuilder::new();
    let body = identity_map_body(&mut b);
    let f = b
        .declare(
            FuncKind::Named,
            "print_mid",
            Ty::Tuple(vec![Ty::IoToken, Ty::i32()]),
            Ty::IoToken,
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let input = fb.input();
        let token = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let unknown = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        let eight = fb.constant(Value::I32(8), L).unwrap();
        let trap_array = fb.iota(eight, Dest::Fresh(None), L).unwrap();
        fb.index(trap_array, unknown, Dest::Fresh(None), L).unwrap();
        let value_array = fb.iota(eight, Dest::Fresh(None), L).unwrap();
        let mapped = fb.map(body, value_array, Dest::Fresh(None), L).unwrap();
        let zero = fb.constant(Value::I32(0), L).unwrap();
        let value = fb.index(mapped, zero, Dest::Fresh(None), L).unwrap();
        let token = fb.print(token, value, L).unwrap();
        fb.output(token, None, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let plan = ir.path_plan(f);
    let indexes = func_ops(&ir, f, |op| op == Operation::Index);
    assert_eq!(indexes.len(), 2);
    let trap = task_for(&plan, indexes[0]);
    let producer = task_for(&plan, indexes[1]);
    assert!(plan.tasks[trap].trap_min.is_some());
    assert_eq!(plan.tasks[producer].trap_min, None);

    assert_eq!(plan.checkpoints.len(), 2);
    let print = &plan.checkpoints[0];
    let mut expected = vec![
        WaitEntry {
            task: trap,
            threshold: Some(topo_pos(&ir, f, indexes[0])),
        },
        WaitEntry {
            task: producer,
            threshold: None,
        },
    ];
    expected.sort_unstable_by_key(|entry| entry.task);
    assert_eq!(print.wait, expected);
    assert_eq!(
        plan.checkpoints[1].wait,
        (0..plan.tasks.len())
            .map(|task| WaitEntry {
                task,
                threshold: None
            })
            .collect::<Vec<_>>()
    );
    assert!(!plan.is_single_path());
}

pub(super) fn constant_index_path(index: i32) -> (PathPlan, MorphismId, u32) {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "index", i32_arr(16), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let array = fb.input();
        let index = fb.constant(Value::I32(index), L).unwrap();
        fb.index(array, index, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let index = func_ops(&ir, f, |op| op == Operation::Index)[0];
    let topo = ir.topo_order(f).iter().position(|&m| m == index).unwrap() as u32;
    (ir.path_plan(f), index, topo)
}

#[test]
fn path_trap_min_exempts_only_bounds_proven_index() {
    let (proven, index, _) = constant_index_path(3);
    assert_eq!(proven.tasks[task_for(&proven, index)].trap_min, None);

    let (unproven, index, topo) = constant_index_path(20);
    assert_eq!(
        unproven.tasks[task_for(&unproven, index)].trap_min,
        Some(topo)
    );
}

#[test]
fn path_map_trap_min_uses_body_capability_at_site() {
    let mut b = IrBuilder::new();
    let trapping = b
        .declare(FuncKind::MapBody, "lookup", Ty::i32(), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(trapping).unwrap();
        let index = fb.input();
        let four = fb.constant(Value::I32(4), L).unwrap();
        let array = fb.iota(four, Dest::Fresh(None), L).unwrap();
        fb.index(array, index, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let safe = identity_map_body(&mut b);
    let f = b
        .declare(
            FuncKind::Named,
            "maps",
            Ty::Tuple(vec![Ty::IoToken, i32_arr(8)]),
            Ty::IoToken,
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let input = fb.input();
        let token = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let array = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        fb.map(trapping, array, Dest::Fresh(None), L).unwrap();
        fb.map(safe, array, Dest::Fresh(None), L).unwrap();
        let zero = fb.constant(Value::I32(0), L).unwrap();
        let token = fb.print(token, zero, L).unwrap();
        fb.output(token, None, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let maps = func_ops(&ir, f, |op| matches!(op, Operation::Map { .. }));
    let plan = ir.path_plan(f);

    let trapping_task_id = task_for(&plan, maps[0]);
    let trapping_task = &plan.tasks[trapping_task_id];
    let site_topo = topo_pos(&ir, f, maps[0]);
    assert_eq!(trapping_task.trap_min, Some(topo_pos(&ir, f, maps[0])));
    assert!(!trapping_task.pinned);
    assert_eq!(plan.tasks[task_for(&plan, maps[1])].trap_min, None);
    assert_eq!(
        plan.checkpoints[0].wait,
        vec![WaitEntry {
            task: trapping_task_id,
            threshold: Some(site_topo),
        }]
    );
}

#[test]
fn path_wait_uses_max_threshold_and_data_completion_wins() {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "thresholds", Ty::IoToken, Ty::IoToken, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let token = fb.input();
        let lhs = fb.constant(Value::I32(1), L).unwrap();
        let rhs = fb.constant(Value::I32(0), L).unwrap();
        // plan-s49: the two divisions must be CHAINED, not siblings. This test is
        // about one task carrying two trap sites at different topo positions, and
        // scalar components are dataflow components — two `Div`s that merely read
        // the same two constants have no order between them and are now two tasks,
        // correctly. Feeding the first into the second is what puts them in one.
        let first = fb
            .binop(Operation::Div, lhs, rhs, Dest::Fresh(None), L)
            .unwrap();
        let second = fb
            .binop(Operation::Div, first, rhs, Dest::Fresh(None), L)
            .unwrap();
        let token = fb.print(token, rhs, L).unwrap();
        let token = fb.print(token, second, L).unwrap();
        fb.output(token, None, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let plan = ir.path_plan(f);
    let divs = func_ops(&ir, f, |op| op == Operation::Div);
    let task = task_for(&plan, divs[0]);
    assert_eq!(task_for(&plan, divs[1]), task);
    assert_eq!(
        plan.checkpoints
            .iter()
            .map(|checkpoint| checkpoint.wait.clone())
            .collect::<Vec<_>>(),
        vec![
            vec![WaitEntry {
                task,
                threshold: Some(topo_pos(&ir, f, divs[1])),
            }],
            vec![WaitEntry {
                task,
                threshold: None,
            }],
            vec![WaitEntry {
                task,
                threshold: None,
            }],
        ]
    );
}

#[test]
fn path_fold_trap_min_uses_body_capability_at_site() {
    let mut b = IrBuilder::new();
    let body = b
        .declare(
            FuncKind::FoldBody,
            "divide",
            Ty::Tuple(vec![Ty::i32(), Ty::i32()]),
            Ty::i32(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(body).unwrap();
        let input = fb.input();
        let acc = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let elem = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        fb.binop(Operation::Div, acc, elem, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let f = b
        .declare(FuncKind::Named, "fold_div", i32_arr(8), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        let input = fb.input();
        let fold_input = fb.pack(&[one, input], Dest::Fresh(None), L).unwrap();
        fb.fold(body, fold_input, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let fold = func_ops(&ir, f, |op| matches!(op, Operation::Fold { .. }))[0];
    let plan = ir.path_plan(f);

    let task = &plan.tasks[task_for(&plan, fold)];
    assert_eq!(task.trap_min, Some(topo_pos(&ir, f, fold)));
    assert!(!task.pinned);
}

pub(super) fn call_trap_path(transitive: bool) -> (PathPlan, MorphismId, u32) {
    let mut b = IrBuilder::new();
    let args = Ty::Tuple(vec![Ty::i32(), Ty::i32()]);
    let leaf = b
        .declare(FuncKind::Named, "divide", args.clone(), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(leaf).unwrap();
        let input = fb.input();
        let lhs = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let rhs = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        fb.binop(Operation::Div, lhs, rhs, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let callee = if transitive {
        let middle = b
            .declare(FuncKind::Named, "middle", args.clone(), Ty::i32(), L)
            .unwrap();
        {
            let mut fb = b.build_fn(middle).unwrap();
            let input = fb.input();
            fb.call(leaf, input, Dest::Ret { slot: None }, L).unwrap();
            fb.finish().unwrap();
        }
        middle
    } else {
        leaf
    };
    let f = b
        .declare(FuncKind::Named, "caller", args, Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let input = fb.input();
        fb.call(callee, input, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let call = func_ops(&ir, f, |op| matches!(op, Operation::Call(_)))[0];
    let topo = topo_pos(&ir, f, call);
    (ir.path_plan(f), call, topo)
}

#[test]
fn path_trap_capable_calls_pin_direct_and_transitive_tasks() {
    for transitive in [false, true] {
        let (plan, call, topo) = call_trap_path(transitive);
        let task = &plan.tasks[task_for(&plan, call)];
        assert!(task.pinned);
        assert_eq!(task.trap_min, Some(topo));
    }
}

#[test]
fn path_rank_prefers_heavy_bulk_long_path_to_light_scalar_chain() {
    let mut b = IrBuilder::new();
    let body = identity_map_body(&mut b);
    let f = b
        .declare(FuncKind::Named, "rank", Ty::Unit, i32_arr(128), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        let mut scalar = one;
        for _ in 0..8 {
            scalar = fb
                .binop(Operation::Add, scalar, one, Dest::Fresh(None), L)
                .unwrap();
        }
        let n = fb.constant(Value::I32(128), L).unwrap();
        let range = fb.iota(n, Dest::Fresh(None), L).unwrap();
        fb.map(body, range, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let plan = ir.path_plan(f);
    let scalar = task_for(&plan, func_ops(&ir, f, |op| op == Operation::Add)[0]);
    let heavy = task_for(&plan, func_ops(&ir, f, |op| op == Operation::Iota)[0]);
    assert!(plan.tasks[heavy].rank > plan.tasks[scalar].rank);
}

/// The bench bracket (plan-time-builtin): data generation, `t0`, the kernel
/// map, `t1`, then `t1 - t0 -> print` and a `y[0]` readout. Two `TimeMs` reads
/// on the host spine with a bulk task between them and a scalar task after.
///
/// Spans ASCEND with construction order, as a lowered program's do: the
/// `TimeMs` fence is keyed on source position (the dataflow graph orders pure
/// work against a clock read not at all), so a fixture built with one shared
/// span would model a program written entirely on one line and fence nothing.
pub(super) fn time_bracket_fixture() -> (mapal_ir::CategoryIr, mapal_ir::FuncId) {
    // One span per source "line", ascending.
    let at = |line: u32| SourceLoc {
        start: line * 10,
        end: line * 10 + 9,
    };
    let mut b = IrBuilder::new();
    let body = identity_map_body(&mut b);
    let f = b
        .declare(FuncKind::Named, "bracket", Ty::IoToken, Ty::IoToken, at(0))
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let token = fb.input();
        // Data generation — OUTSIDE the bracket (the conv2d gen/kernel split).
        let eight = fb.constant(Value::I32(8), at(1)).unwrap();
        let data = fb
            .iota(eight, Dest::Fresh(Some("gen".into())), at(1))
            .unwrap();
        // `() -> time -> t0`: open the bracket.
        let start = fb.time_ms(token, at(2)).unwrap();
        let token = fb.proj(start, 0, Dest::Fresh(None), at(2)).unwrap();
        let t0 = fb
            .proj(start, 1, Dest::Fresh(Some("t0".into())), at(2))
            .unwrap();
        // The kernel: one bulk map over the generated array.
        let mapped = fb
            .map(body, data, Dest::Fresh(Some("y".into())), at(3))
            .unwrap();
        // `() -> time -> t1`: close it.
        let stop = fb.time_ms(token, at(4)).unwrap();
        let token = fb.proj(stop, 0, Dest::Fresh(None), at(4)).unwrap();
        let t1 = fb
            .proj(stop, 1, Dest::Fresh(Some("t1".into())), at(4))
            .unwrap();
        let elapsed = fb
            .binop(
                Operation::Sub,
                t1,
                t0,
                Dest::Fresh(Some("elapsed".into())),
                at(5),
            )
            .unwrap();
        let token = fb.print(token, elapsed, at(5)).unwrap();
        // Consume the kernel result so the map is not dead code.
        let zero = fb.constant(Value::I32(0), at(6)).unwrap();
        let head = fb.index(mapped, zero, Dest::Fresh(None), at(6)).unwrap();
        let token = fb.print(token, head, at(6)).unwrap();
        fb.output(token, None, at(7)).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    (ir, f)
}

#[test]
fn path_time_ms_fences_only_the_tasks_entirely_before_the_read() {
    // plan-time-builtin composition rule 1. A `TimeMs` consumes only the token,
    // so it has NO value producer to wait for and would read the clock while
    // the tasks it brackets are still in flight. It therefore fences: every
    // task written ENTIRELY before it in the SOURCE must complete. Source
    // order, because the graph supplies none — which is what makes `t1 - t0`
    // the work written between the two reads. plan-s33b turns the read into a
    // task, so the fence is stated as that task's `deps`.
    let (ir, f) = time_bracket_fixture();
    let plan = ir.path_plan(f);
    let reads = func_ops(&ir, f, |op| op == Operation::TimeMs);
    assert_eq!(reads.len(), 2, "t0 and t1");
    let kernel = task_for(
        &plan,
        func_ops(&ir, f, |op| matches!(op, Operation::Map { .. }))[0],
    );
    let gen_task = task_for(&plan, func_ops(&ir, f, |op| op == Operation::Iota)[0]);
    let readout = task_for(&plan, func_ops(&ir, f, |op| op == Operation::Index)[0]);
    let clock = |m: MorphismId| task_for(&plan, m);
    let fences = |m: MorphismId, task: usize| plan.tasks[clock(m)].deps.contains(&task);

    // Each read is its own task, pinned to the host spine: a worker running it
    // would put the clock wherever the pool happened to be.
    for &m in &reads {
        assert!(
            plan.tasks[clock(m)].pinned,
            "a clock read runs on the host spine"
        );
    }
    // t0 opens the bracket: the generation is written above it, so the read
    // depends on it — that is what excludes generation from the interval. The
    // kernel below it is untouched.
    assert!(
        fences(reads[0], gen_task),
        "t0 fences the generation written above it: {:?}",
        plan.tasks[clock(reads[0])].deps
    );
    assert!(
        !fences(reads[0], kernel),
        "t0 does not fence the kernel it opens: {:?}",
        plan.tasks[clock(reads[0])].deps
    );
    // t1 closes it: the kernel is now written above, so it is fenced too.
    assert!(
        fences(reads[1], kernel) && fences(reads[1], gen_task),
        "t1 fences the bracketed kernel: {:?}",
        plan.tasks[clock(reads[1])].deps
    );
    // …and the fence is "written before", not "everything": the post-bracket
    // readout task (the `y[0]` Index, written after t1) is in neither.
    assert!(
        !fences(reads[0], readout) && !fences(reads[1], readout),
        "neither read fences work written below it: {:?} / {:?}",
        plan.tasks[clock(reads[0])].deps,
        plan.tasks[clock(reads[1])].deps
    );
}

#[test]
fn path_time_ms_holds_back_the_work_written_after_it() {
    // plan-s33b, the half that was missing. `before` alone lets the bracketed
    // work START before the read that opens the bracket: the moment its inputs
    // are ready the DAG unlocks it, while the host is still walking toward the
    // clock. `t1 - t0` then measures an interval the work has already left —
    // measured at 6/100 threaded runs of fir 65 536 reading under 0.01 ms for a
    // ~0.07 ms kernel. The read is a DAG node, so `after` is ordinary edges.
    let (ir, f) = time_bracket_fixture();
    let plan = ir.path_plan(f);
    let reads = func_ops(&ir, f, |op| op == Operation::TimeMs);
    let t0 = task_for(&plan, reads[0]);
    let t1 = task_for(&plan, reads[1]);
    let kernel = task_for(
        &plan,
        func_ops(&ir, f, |op| matches!(op, Operation::Map { .. }))[0],
    );
    let gen_task = task_for(&plan, func_ops(&ir, f, |op| op == Operation::Iota)[0]);
    let readout = task_for(&plan, func_ops(&ir, f, |op| op == Operation::Index)[0]);

    assert!(
        plan.tasks[kernel].deps.contains(&t0),
        "the bracketed kernel cannot be dispatched before the read that opens the bracket: {:?}",
        plan.tasks[kernel].deps
    );
    assert!(
        plan.tasks[readout].deps.contains(&t1),
        "work written after the closing read waits for it: {:?}",
        plan.tasks[readout].deps
    );
    // The generation is written above t0, so it is a dependency OF the read,
    // never a dependent — holding it back would put it inside the interval.
    assert!(
        !plan.tasks[gen_task].deps.contains(&t0) && !plan.tasks[gen_task].deps.contains(&t1),
        "work written above a read is never held by it: {:?}",
        plan.tasks[gen_task].deps
    );
}

#[test]
fn path_time_ms_consumer_cone_stays_on_the_host_spine() {
    // `TimeMs` is the first spine op producing a VALUE, and tasks are
    // dispatched BEFORE the host writes that slot — a task consuming a clock
    // read races the write (the symptom was a NEGATIVE elapsed). The whole
    // consumer cone therefore stays on the spine.
    let (ir, f) = time_bracket_fixture();
    let plan = ir.path_plan(f);
    assert!(!plan.tasks.is_empty(), "the bulk work still parallelises");

    let named = |n: &str| {
        ir.objects()
            .find(|(_, o)| o.name.as_deref() == Some(n))
            .map(|(id, _)| id)
            .unwrap()
    };
    // The clock-carrying objects: the two `(IoToken, f64)` read results and the
    // `t0`/`t1`/`elapsed` chain hanging off them (proj, Sub, the print Pair).
    let mut clock: Vec<mapal_ir::ObjectId> = func_ops(&ir, f, |op| op == Operation::TimeMs)
        .iter()
        .map(|&m| ir.morphism(m).unwrap().target)
        .collect();
    clock.extend(["t0", "t1", "elapsed"].map(named));

    for task in &plan.tasks {
        let members: &[MorphismId] = match &task.kind {
            TaskKind::Split { site, .. } => std::slice::from_ref(site),
            TaskKind::Seq { morphisms } => morphisms,
        };
        for &m in members {
            let morph = ir.morphism(m).unwrap();
            // plan-s33b: the READ itself is a task — pinned, so the host spine
            // still executes it, at its own topo position, and the frame slot
            // is written there. What must not escape is the CONSUMER cone: a
            // pooled task reading that slot still races the write.
            if matches!(morph.op, Operation::TimeMs) {
                assert!(task.pinned, "the clock read stays on the host spine");
                continue;
            }
            assert!(
                !clock.contains(&morph.source) && !clock.contains(&morph.target),
                "clock-cone {:?} escaped onto a task",
                morph.op
            );
        }
    }
}

#[test]
fn path_plan_is_deterministic() {
    let (ir, f) = diamond_path_fixture();
    assert_eq!(ir.path_plan(f), ir.path_plan(f));
    let (ir, f) = time_bracket_fixture();
    assert_eq!(ir.path_plan(f), ir.path_plan(f));
}
