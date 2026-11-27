//! Rule R movement helpers (trip copies, required copies) and the carried-loop / capture differentials.

use super::*;

// --- S65 rule R on programs it was not designed on ---------------------------

/// One arm of the carried-loop sweep: `generic`, the exact face, the `by-ref`
/// rung on (shipped) or forced off (S64's by-value loop products).
pub(crate) fn emit_by_ref(ir: &CategoryIr, on: bool) -> String {
    let mut gpu = mapal_backend_llvm::GpuArms::default();
    if !on {
        *gpu.off.rung("by-ref").expect("rung") = true;
    }
    mapal_backend_llvm::emit_with_opts(
        ir,
        &mapal_backend_llvm::EmitOpts {
            target: "generic",
            gpu,
            ..mapal_backend_llvm::EmitOpts::exact()
        },
    )
    .expect("emits")
}

/// The array copies the emitted host IR makes per trip, over every loop the
/// emitter headed with `; mapal.move` (header block to latch): `llvm.memcpy`
/// calls, and whole-array values loaded or stored (a by-value forward).
pub(crate) fn trip_copies(ll: &str) -> (usize, usize) {
    let (mut copies, mut values) = (0, 0);
    for f in ll.split("\ndefine ").skip(1) {
        let lines: Vec<&str> = f.lines().collect();
        for (h, l) in lines.iter().enumerate() {
            let Some(rest) = l.strip_prefix("; mapal.move loop=") else {
                continue;
            };
            let header = rest.split_whitespace().next().expect("header");
            let back = format!("br label %{header}");
            let latch = (h..lines.len())
                .rev()
                .find(|&i| lines[i].trim() == back)
                .expect("latch");
            for x in &lines[h..=latch] {
                copies += x.contains("call void @llvm.memcpy") as usize;
                let t = x.trim();
                let value = (t.contains("= load [")
                    || t.contains("= load {")
                    || t.starts_with("store [")
                    || t.starts_with("store {"))
                    && t.split(", ptr").next().is_some_and(|ty| ty.contains(" x "));
                values += value as usize;
            }
        }
    }
    (copies, values)
}

/// Where an array position's bytes are, under handle semantics (every
/// forwarding morphism re-exposes them): an object's own, a loop state
/// position, or bytes the function does not own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Bytes {
    Own(mapal_ir::ObjectId),
    State(mapal_ir::ObjectId, Vec<u32>),
    Foreign,
}

/// **S65's movement oracle**, an independent reading of the IR facts: the
/// array copies `f`'s loops need per trip. Forwarding (`Pair`, `Proj`,
/// `Phi`, a loop's enter, back edge and exit) needs none; an `Update` needs
/// one unless its array's bytes are one buffer the function may overwrite —
/// its own (made in every loop running the update) or a loop state the loop
/// owns (every buffer the next state may be is the state's or made in the
/// trip, for that position alone) — and no other holder of those bytes made
/// before the update is read after it (`LastUsePlan::view_last_use`; a
/// guard-arm read runs at its `Phi`: any time; a whole-state read: any time).
pub(crate) fn required_trip_copies(ir: &CategoryIr, f: mapal_ir::FuncId) -> usize {
    use mapal_ir::{ObjectKind, Operation as Op};
    use std::collections::{HashMap, HashSet};
    let lup = ir.last_use_plan(f);
    let gated: HashSet<_> = ir
        .guard_plan(f)
        .into_iter()
        .filter(|g| g.gated())
        .flat_map(|g| g.on_true.own.into_iter().chain(g.on_false.own))
        .collect();
    let loops: Vec<(mapal_ir::LoopPlan, HashSet<mapal_ir::MorphismId>)> = ir
        .loop_structure(f)
        .into_iter()
        .flat_map(|s| s.merges)
        .filter_map(|m| ir.loop_plan(f, m))
        .map(|p| {
            let body = p
                .decide_order
                .iter()
                .chain(&p.advance_order)
                .copied()
                .collect();
            (p, body)
        })
        .collect();
    let mo = |m| ir.morphism(m).expect("morphism");
    let pair_src = |o, k| {
        ir.in_edges(o)
            .iter()
            .map(|&m| mo(m))
            .find(|x| matches!(x.op, Op::Pair { slot, .. } if slot == k))
            .map(|x| x.source)
    };
    let objs: Vec<mapal_ir::ObjectId> = ir
        .objects()
        .map(|(o, _)| o)
        .filter(|&o| ir.try_owner(o) == Some(f))
        .collect();
    let paths = |ty: &Ty| -> Vec<Vec<u32>> {
        fn go(ty: &Ty, pre: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
            match ty {
                Ty::Array { .. } => out.push(pre.clone()),
                Ty::Tuple(ts) => {
                    for (i, t) in ts.iter().enumerate() {
                        pre.push(i as u32);
                        go(t, pre, out);
                        pre.pop();
                    }
                }
                Ty::Struct { fields, .. } => {
                    for (i, (_, t)) in fields.iter().enumerate() {
                        pre.push(i as u32);
                        go(t, pre, out);
                        pre.pop();
                    }
                }
                _ => {}
            }
        }
        let mut out = Vec::new();
        go(ty, &mut Vec::new(), &mut out);
        out
    };
    let ty_of = |o| ir.object(o).expect("object").ty.clone();
    let bytes = |alias: &HashMap<_, _>, o, p: Vec<u32>| -> Vec<Bytes> {
        let mut out = Vec::new();
        let mut stack = vec![(o, p)];
        while let Some((o, p)) = stack.pop() {
            if p.is_empty()
                && let Some(&s) = alias.get(&o)
            {
                stack.push((s, p));
                continue;
            }
            let obj = ir.object(o).expect("object");
            let b = match obj.kind {
                ObjectKind::Parameter | ObjectKind::Constant => Some(Bytes::Foreign),
                ObjectKind::LoopMerge => Some(Bytes::State(o, p.clone())),
                _ => match ir.in_edges(o).first().map(|&m| mo(m)) {
                    Some(x) if matches!(x.op, Op::Pair { .. }) && !p.is_empty() => {
                        match pair_src(o, p[0]) {
                            Some(s) => {
                                stack.push((s, p[1..].to_vec()));
                                None
                            }
                            None => Some(Bytes::Foreign),
                        }
                    }
                    Some(x) if matches!(x.op, Op::Proj { .. } | Op::LoopExit | Op::Phi) => {
                        let pre: Vec<u32> = match x.op {
                            Op::Proj { index } => vec![index],
                            _ => vec![0],
                        };
                        stack.push((x.source, pre.iter().chain(&p).copied().collect()));
                        if x.op == Op::Phi {
                            stack.push((x.source, [1].iter().chain(&p).copied().collect()));
                        }
                        None
                    }
                    _ if p.is_empty() => Some(Bytes::Own(o)),
                    _ => Some(Bytes::Foreign),
                },
            };
            if let Some(b) = b
                && !out.contains(&b)
            {
                out.push(b);
            }
        }
        out
    };
    let holders = |alias: &HashMap<_, _>, b: &Bytes| -> Vec<mapal_ir::ObjectId> {
        objs.iter()
            .copied()
            .filter(|&o| {
                paths(&ty_of(o))
                    .into_iter()
                    .any(|q| bytes(alias, o, q).contains(b))
            })
            .collect()
    };
    let born = |o| {
        ir.in_edges(o)
            .iter()
            .filter_map(|&m| lup.position(m))
            .max()
            .unwrap_or(0)
    };
    let made_in = |body: &HashSet<_>, o| ir.in_edges(o).iter().any(|m| body.contains(m));
    let mut updates: Vec<mapal_ir::MorphismId> = ir
        .morphisms()
        .filter(|(_, x)| x.op == Op::Update && ir.try_owner(x.source) == Some(f))
        .map(|(m, _)| m)
        .collect();
    updates.sort_by_key(|&m| lup.position(m));
    // The parallel entry (`generic` is a CPU profile: a multi-path, acyclic
    // plan runs as tasks): a read before the write in another task must be
    // one the plan orders before it.
    let par: Option<(HashMap<mapal_ir::MorphismId, usize>, Vec<Vec<bool>>)> =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ir.path_plan(f)))
            .ok()
            .filter(|p| !p.is_single_path())
            .and_then(|p| {
                let n = p.tasks.len();
                let mut task = HashMap::new();
                let mut before = vec![vec![false; n]; n];
                for (t, tk) in p.tasks.iter().enumerate() {
                    match &tk.kind {
                        mapal_ir::TaskKind::Split { site, .. } => {
                            task.insert(*site, t);
                        }
                        mapal_ir::TaskKind::Seq { morphisms } => {
                            for &m in morphisms {
                                task.insert(m, t);
                            }
                        }
                    }
                    for &d in &tk.deps {
                        before[d][t] = true;
                    }
                }
                for k in 0..n {
                    for i in 0..n {
                        for j in 0..n {
                            if before[i][k] && before[k][j] {
                                before[i][j] = true;
                            }
                        }
                    }
                }
                (0..n).all(|i| !before[i][i]).then_some((task, before))
            });
    let ordered = |hs: &[mapal_ir::ObjectId], u: mapal_ir::MorphismId| {
        let Some((task, before)) = &par else {
            return true;
        };
        let Some(&tw) = task.get(&u) else {
            return false;
        };
        // (A loop's exit object reads its final state, after the loop.)
        let exit = |o| ir.in_edges(o).iter().any(|&m| mo(m).op == Op::LoopExit);
        hs.iter()
            .filter(|&&o| !exit(o))
            .flat_map(|&o| ir.out_edges(o).iter().copied())
            .all(|e| {
                e == u
                    || lup.position(e) >= lup.position(u)
                    || matches!(mo(e).op, Op::LoopEnter | Op::LoopBack | Op::LoopExit)
                        && !task.contains_key(&e)
                    || task.get(&e).is_some_and(|&tr| tr == tw || before[tr][tw])
            })
    };
    let mut alias: HashMap<mapal_ir::ObjectId, mapal_ir::ObjectId> = HashMap::new();
    let mut need = 0;
    for u in updates {
        let at = lup.position(u).expect("position");
        let arr = pair_src(mo(u).source, 0).expect("update array");
        let mine: Vec<&(mapal_ir::LoopPlan, HashSet<_>)> =
            loops.iter().filter(|(_, b)| b.contains(&u)).collect();
        // No holder made before the update is read after it.
        let quiet = |hs: Vec<mapal_ir::ObjectId>, body: Option<&HashSet<_>>| {
            hs.into_iter()
                .filter(|&o| born(o) < at && body.is_none_or(|b| made_in(b, o)))
                .all(|o| {
                    !ir.out_edges(o)
                        .iter()
                        .any(|&e| gated.contains(&e) && e != u && mo(e).target != mo(u).source)
                        && lup.view_last_use(o).is_none_or(|p| p <= at)
                        && (body.is_some() || (!lup.escapes(o) && lup.carried_by(o).is_none()))
                })
        };
        let ok = match &bytes(&alias, arr, vec![])[..] {
            [Bytes::Own(c)] => {
                let hs = holders(&alias, &Bytes::Own(*c));
                ir.object(*c).expect("object").kind == ObjectKind::Temporary
                    && mine.iter().all(|(_, b)| made_in(b, *c))
                    && ordered(&hs, u)
                    && quiet(hs, None)
            }
            [Bytes::State(m, p)] => {
                mine.iter()
                    .find(|(pl, _)| pl.merge == *m)
                    .is_some_and(|(pl, body)| {
                        let next = |q: &[u32]| {
                            bytes(
                                &HashMap::new(),
                                pl.back_route,
                                [0].iter().chain(q).copied().collect(),
                            )
                        };
                        let others: Vec<Bytes> = paths(&ty_of(*m))
                            .iter()
                            .filter(|q| *q != p)
                            .flat_map(|q| next(q))
                            .collect();
                        let owned = next(p).iter().all(|b| match b {
                            Bytes::State(m2, q) => m2 == m && q == p,
                            Bytes::Own(o) => made_in(body, *o) && !others.contains(b),
                            Bytes::Foreign => false,
                        });
                        let whole = ir
                            .out_edges(*m)
                            .iter()
                            .any(|&e| !matches!(mo(e).op, Op::Proj { .. }));
                        let hs = holders(&alias, &Bytes::State(*m, p.clone()));
                        owned && !whole && ordered(&hs, u) && quiet(hs, Some(body))
                    })
            }
            _ => false,
        };
        if ok {
            alias.insert(mo(u).target, arr);
        } else if !mine.is_empty() {
            need += 1;
        }
    }
    need
}

/// **S65 rule R, proven on unseen programs.** ≥ 256 generated `main`s whose
/// loops carry arrays (testgen `carry_prog_strategy`): one or two, one nested
/// with the accumulator in a sub-product, read through maps, folds, element
/// reads and captured maps, rewritten by update chains, chosen between by
/// guards whose arms own the work, swapped, replaced by invariants, zero to
/// eight trips, two loops in sequence — none of them written for the rule.
/// Each is (a) value-equal to the interpreter at `-O0` and `-O2` with the
/// rung on and off, and (b) moves per trip exactly the copies the facts
/// require ([`required_trip_copies`]): no forwarding morphism copies, and
/// every copy is an `Update` whose array is read again.
#[test]
fn differential_carried_loops_values_and_movement() {
    let Some(clang) = clang() else {
        eprintln!("SKIP differential_carried_loops: clang not found");
        return;
    };
    let _ = rt_lib();
    let mut runner = TestRunner::deterministic();
    let (mut jobs, mut moved, mut n) = (Vec::new(), Vec::new(), 0usize);
    let (mut trips, mut copies, mut required, mut loops) = (0usize, 0usize, 0usize, 0usize);
    for (count, trap_free) in [(224usize, true), (96, false)] {
        let strat = testgen::carry_prog_strategy(trap_free);
        for _ in 0..count {
            let prog = strat.new_tree(&mut runner).unwrap().current();
            let Built { ir, .. } = build(&prog);
            let rr = run(&ir, BUDGET);
            let Some((want_out, want_code)) = expect_native(&ir, &rr) else {
                continue; // diverged
            };
            let on = emit_by_ref(&ir, true);
            let (c, v) = trip_copies(&on);
            let need = required_trip_copies(&ir, ir.entry());
            loops += on.matches("; mapal.move loop=").count();
            trips += matches!(rr.outcome, Outcome::Done(_)) as usize;
            copies += c;
            required += need;
            if c != need || v != 0 {
                moved.push(format!("carry#{n}: {c} copies, {v} by-value forwards per trip; the facts require {need}\n{prog:?}"));
            }
            for (arm, ll) in [("on", on), ("off", emit_by_ref(&ir, false))] {
                if let Ok(d) = std::env::var("S65_DUMP") {
                    std::fs::write(format!("{d}/carry{n}_{arm}.ll"), &ll).unwrap();
                    std::fs::write(
                        format!("{d}/carry{n}.prog"),
                        format!("{prog:#?}\n{want_code}\n{want_out:?}"),
                    )
                    .unwrap();
                }
                jobs.push(Job {
                    tag: format!("carry#{n}/{arm}"),
                    ll,
                    want_out: want_out.clone(),
                    want_code,
                });
            }
            n += 1;
        }
    }
    assert!(n >= 256, "expected ≥256 carried-loop programs, got {n}");
    assert!(
        loops >= n,
        "every program has a carried loop: {loops} loops in {n}"
    );
    assert!(
        moved.is_empty(),
        "{} movement findings:\n{}",
        moved.len(),
        moved.join("\n")
    );
    let failures = run_jobs(&clang, &jobs);
    assert!(
        failures.is_empty(),
        "{} carried-loop divergences:\n{}",
        failures.len(),
        failures.join("\n")
    );
    eprintln!(
        "differential_carried_loops: {n} programs ({trips} run to the end), {loops} loops; \
         per-trip copies {copies} = required {required}; values OK on and off at -O0 and -O2"
    );
}
