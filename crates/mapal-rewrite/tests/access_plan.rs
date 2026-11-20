//! kernel-merging groundwork: `access_plan`, the compile-time dependency fact
//! (read / write sets per morphism, storage classes, RAW / WAR / WAW edges of
//! each loop trip) — on the decode step and on generated loops.

mod testgen;

use mapal_ir::{AccessPlan, CategoryIr, MorphismId, ObjectId, Operation};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "{:?}\n{src}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("{d:?}\n{src}"));
    mapal_rewrite::rewrite(ir).ir
}

/// The non-structural morphism making the object named `name` inside the
/// trip `order` (the last one, for a carried name made again each trip).
fn maker(ir: &CategoryIr, order: &[MorphismId], name: &str) -> usize {
    order
        .iter()
        .rposition(|&m| {
            let mo = ir.morphism(m).unwrap();
            !matches!(mo.op, Operation::Pair { .. } | Operation::Proj { .. })
                && ir.object(mo.target).unwrap().name.as_deref() == Some(name)
        })
        .unwrap_or_else(|| panic!("no maker of {name}"))
}

/// The decode step (`pt_decode_1024_256`): the state row `x`, the retained
/// GEMV rows `kk`, `vv`, the query `q`, the scores `s`, the output `o`. The
/// edges between their makers are the dataflow's: `kk`, `vv` after the row;
/// `s` after `kk` and `q`; `o` after `s` and `vv`, and after the row (it read
/// the last output, whose buffer the next one may take: WAR); `q` after
/// nothing of the trip.
#[test]
fn decode_trip_edges_are_the_dataflow() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../benches/algorithms/pt_decode_1024_256.mapal"
    ))
    .unwrap();
    let ir = lower(&src);
    let f = ir.entry();
    let plan = ir.access_plan(f);
    assert_eq!(plan, ir.access_plan(f), "stored once");
    let merge = ir.loop_structure(f)[0].merges[0];
    let (order, edges) = plan.trip(merge).expect("a trip");
    let names = ["x", "kk", "vv", "q", "s", "o"];
    let at: Vec<usize> = names.iter().map(|n| maker(&ir, order, n)).collect();
    let edge = |i: usize, j: usize| {
        edges[at[j]]
            .iter()
            .find(|(k, _)| *k == at[i])
            .map(|&(_, h)| h)
    };
    let mut got = Vec::new();
    for j in 0..names.len() {
        for i in 0..names.len() {
            if at[i] < at[j]
                && let Some(h) = edge(i, j)
            {
                got.push((names[i], names[j], h.raw, h.war, h.waw));
            }
        }
    }
    assert_eq!(
        got,
        vec![
            ("x", "kk", true, false, false),
            ("x", "vv", true, false, false),
            ("kk", "s", true, false, false),
            ("q", "s", true, false, false),
            ("x", "o", false, true, false),
            ("vv", "o", true, false, false),
            ("s", "o", true, false, false),
        ],
        "{got:?}"
    );
    // The carried output's view, next value and exit share one class; the
    // row's state and its next value another.
    let obj = |n: &str| -> Vec<ObjectId> {
        ir.objects()
            .filter(|(_, o)| o.name.as_deref() == Some(n))
            .map(|(id, _)| id)
            .collect()
    };
    let o_next = ir.morphism(order[at[5]]).unwrap().target;
    let x_next = ir.morphism(order[at[0]]).unwrap().target;
    let out = obj("out")[0];
    assert!(plan.shares(o_next, out));
    assert!(!plan.shares(o_next, x_next));
    let kk = ir.morphism(order[at[1]]).unwrap().target;
    assert_eq!(plan.class(kk), kk, "a trip temporary is its own class");
}

/// Every edge [`AccessPlan::trip`] stores is the brute-force pairwise
/// conflict of the stored read / write sets, every dataflow edge inside the
/// trip (a morphism reading the bytes an earlier one made) is a RAW edge, and
/// a RAW edge between two objects that share no class is a dataflow edge —
/// over generated loops (`twin_prog_strategy`, `twin_number_strategy`,
/// `recur_prog_strategy`, `device_prog_strategy`).
#[test]
fn trip_edges_are_the_conflicts_on_generated_loops() {
    let mut runner = TestRunner::deterministic();
    let mut srcs = Vec::new();
    for _ in 0..60 {
        let p = testgen::twin_prog_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        srcs.push(testgen::twin_src(&p, true, None));
        let p = testgen::twin_number_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        srcs.push(testgen::twin_src(&p, true, None));
        let p = testgen::recur_prog_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        srcs.push(testgen::recur_src(&p));
        let p = testgen::device_prog_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        srcs.push(testgen::device_src(&p));
    }
    let (mut trips, mut raw, mut war, mut waw, mut flow, mut carried) = (0, 0, 0, 0, 0, 0);
    for src in &srcs {
        let ir = lower(src);
        let f = ir.entry();
        let plan: AccessPlan = ir.access_plan(f);
        for scc in ir.loop_structure(f) {
            for &merge in &scc.merges {
                let Some((order, edges)) = plan.trip(merge) else {
                    continue;
                };
                trips += 1;
                for (j, &mj) in order.iter().enumerate() {
                    let want: Vec<usize> = (0..j)
                        .filter(|&i| {
                            let mi = order[i];
                            plan.hazard(
                                (plan.reads(mi), plan.writes(mi)),
                                (plan.reads(mj), plan.writes(mj)),
                                &|_, _| false,
                            )
                            .any()
                        })
                        .collect();
                    let have: Vec<usize> = edges[j].iter().map(|e| e.0).collect();
                    assert_eq!(have, want, "{src}");
                    for &(i, h) in &edges[j] {
                        raw += h.raw as usize;
                        war += h.war as usize;
                        waw += h.waw as usize;
                        // A RAW edge is a dataflow read, or a read of bytes
                        // the class shares (a carried buffer).
                        let ti = ir.morphism(order[i]).unwrap().target;
                        if h.raw && !plan.reads(mj).contains(&ti) {
                            let shared = ir
                                .objects()
                                .filter(|(o, _)| *o != ti && plan.shares(*o, ti))
                                .count();
                            assert!(shared > 0, "RAW without dataflow\n{src}");
                            carried += 1;
                        }
                    }
                    // Dataflow inside the trip: an earlier maker of a byte
                    // source of `mj` is an edge.
                    for &r in plan.reads(mj) {
                        for i in 0..j {
                            if plan.writes(order[i]).contains(&r) {
                                flow += 1;
                                assert!(
                                    edges[j].iter().any(|e| e.0 == i && e.1.raw),
                                    "missed RAW\n{src}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "trip_edges: {trips} trips over {} programs, {raw} RAW ({carried} through a carried class) / {war} WAR / {waw} WAW edges, {flow} dataflow reads",
        srcs.len()
    );
    assert!(trips > 0 && raw > 0 && war > 0 && flow > 0);
}
