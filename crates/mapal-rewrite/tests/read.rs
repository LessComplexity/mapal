//! ADR-0045 D6–D8: the facts that know about `read`, and the rewriter.
//!
//! - `unchanged_plan`: a read's array is the same at every call (no input
//!   slot) and on every trip of a loop it is read in.
//! - `access_plan`: the read writes its array's bytes; the `(token, array)`
//!   pair and the array are one storage class.
//! - `path_plan`: the read and its array's `Proj` are one task its readers
//!   depend on, a trap site (a failure is recorded at its position), and the
//!   next print's checkpoint waits for it.
//! - `rewrite` rebuilds the read and carries the graph's data directory; the
//!   oracle's output is the same before and after, on generated programs too
//!   (`testgen::read_chain_strategy`, coverage counted).

use mapal_interp::{Outcome, run};
use mapal_ir::{CategoryIr, MorphismId, ObjectId, Operation, TaskKind};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

mod testgen;

fn lower(src: &str, dir: &std::path::Path) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    mapal_lower::lower(src, &po.program)
        .unwrap_or_else(|d| panic!("lower: {d:?}\n{src}"))
        .with_data_dir(Some(dir.display().to_string()))
}

/// The entry's reads and, for each, its array object.
fn reads(ir: &CategoryIr) -> Vec<(MorphismId, ObjectId)> {
    ir.morphisms()
        .filter(|(_, m)| m.op == Operation::Read)
        .map(|(id, m)| (id, ir.read_values(m.target)[0]))
        .collect()
}

const LOOPED: &str = r#"fn main() {
    "weights.bin" -> read -> weights: [f32; 16];
    16 -> iota -> cells;
    mut x: [f32; 16] <- weights;
    mut step: i32 <- 0;
    loop {
        (step < 3) -> {
            -true-> {
                "bias.bin" -> read -> bias: [f32; 16];
                cells -> map {
                    t -> x[t] * weights[t] + bias[t]
                } -> x;
                step + 1 -> step;
                -> loop;
            }
            -false-> x -> out;
        }
    }
    out[15] -> println;
}
"#;

fn files(dir: &std::path::Path) {
    let v = |k: f32| {
        mapal_ir::read::encode(
            &(0..16)
                .map(|i| mapal_ir::Value::F32(i as f32 * k))
                .collect::<Vec<_>>(),
        )
    };
    std::fs::write(dir.join("weights.bin"), v(0.25)).unwrap();
    std::fs::write(dir.join("bias.bin"), v(0.5)).unwrap();
}

#[test]
fn unchanged_and_access_facts() {
    let dir = tempfile::tempdir().unwrap();
    files(dir.path());
    let ir = lower(LOOPED, dir.path());
    let f = ir.entry();
    let up = ir.unchanged_plan(f);
    let ap = ir.access_plan(f);
    let rs = reads(&ir);
    assert_eq!(rs.len(), 2);
    for &(m, arr) in &rs {
        assert!(up.same_every_call(arr), "a read's array has no input slot");
        let pair = ir.morphism(m).unwrap().target;
        assert!(ap.writes(m).contains(&arr), "the read writes its array");
        assert!(ap.shares(pair, arr), "pair and array are one buffer");
        assert_eq!(ir.read_of(arr), Some(m));
    }
    // The read in the loop body yields the same array every trip.
    let (_, bias) = rs[1];
    assert!(up.same_every_trip(bias), "bias changes per trip");
}

#[test]
fn path_plan_makes_the_read_a_task() {
    let dir = tempfile::tempdir().unwrap();
    files(dir.path());
    let src = r#"fn main() {
    "start" -> println;
    "weights.bin" -> read -> weights: [f32; 16];
    weights -> map {
        w -> w * 2.0
    } -> doubled;
    (0.0, doubled) -> fold {
        acc, w -> acc + w
    } -> total;
    total -> println;
}
"#;
    let ir = lower(src, dir.path());
    let plan = ir.path_plan(ir.entry());
    let (m, _) = reads(&ir)[0];
    let proj = ir
        .out_edges(ir.morphism(m).unwrap().target)
        .iter()
        .copied()
        .find(|&p| ir.read_of(ir.morphism(p).unwrap().target).is_some())
        .unwrap();
    let (id, task) = plan
        .tasks
        .iter()
        .enumerate()
        .find(|(_, t)| matches!(&t.kind, TaskKind::Seq { morphisms } if morphisms.contains(&m)))
        .expect("the read is a task");
    assert!(!task.pinned, "the read starts at launch");
    assert!(task.deps.is_empty());
    assert_eq!(
        task.kind,
        TaskKind::Seq {
            morphisms: vec![m, proj]
        }
    );
    let pos = ir
        .topo_order(ir.entry())
        .iter()
        .position(|&x| x == m)
        .unwrap() as u32;
    assert_eq!(task.trap_min, Some(pos), "a read is a trap site");
    // Its readers depend on it; the print after it waits for it.
    assert!(plan.tasks.iter().any(|t| t.deps.contains(&id)));
    let after: Vec<_> = plan
        .checkpoints
        .iter()
        .filter(|c| c.topo > pos && c.topo != u32::MAX)
        .collect();
    assert!(!after.is_empty());
    assert!(after.iter().all(|c| c.wait.iter().any(|w| w.task == id)));
    // No checkpoint at the read itself.
    assert!(plan.checkpoints.iter().all(|c| c.topo != pos));
}

#[test]
fn rewrite_keeps_the_read_and_the_data_dir() {
    let dir = tempfile::tempdir().unwrap();
    files(dir.path());
    let ir = lower(LOOPED, dir.path());
    let want = run(&ir, 1_000_000);
    assert!(
        matches!(want.outcome, Outcome::Done(_)),
        "{:?}",
        want.outcome
    );
    let rw = mapal_rewrite::rewrite(ir).ir;
    assert_eq!(
        rw.data_dir(),
        Some(dir.path().display().to_string().as_str())
    );
    assert_eq!(reads(&rw).len(), 2);
    assert_eq!(run(&rw, 1_000_000).output, want.output);
}

/// Generated chains with read operands: the rewritten program's oracle
/// output is the lowered one's, and the chain computing its operands gives
/// the same output (the files hold the laws' values). Coverage: reads made.
#[test]
fn generated_read_chains_rewrite() {
    let n = testgen::jobs().max(4) * 6;
    let mut runner = TestRunner::deterministic();
    let strat = testgen::read_chain_strategy();
    let chains: Vec<testgen::Chain> = (0..n)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let counts = testgen::par_count(&chains, |i, chain| {
        let dir = tempfile::tempdir().unwrap();
        let files = chain.files();
        for (name, bytes) in &files {
            std::fs::write(dir.path().join(name), bytes).unwrap();
        }
        let src = chain.source();
        let ir = lower(&src, dir.path());
        let rr = run(&ir, 2_000_000_000);
        assert!(
            matches!(rr.outcome, Outcome::Done(_)),
            "#{i}: {:?}\n{src}",
            rr.outcome
        );
        let rw = mapal_rewrite::rewrite(ir).ir;
        assert_eq!(reads(&rw).len(), files.len(), "#{i}");
        assert_eq!(
            run(&rw, 2_000_000_000).output,
            rr.output,
            "#{i} rewrite\n{src}"
        );
        let computed = testgen::Chain {
            read: 0,
            ..chain.clone()
        };
        let law = lower(&computed.source(), dir.path());
        assert_eq!(
            run(&law, 2_000_000_000).output,
            rr.output,
            "#{i} read vs computed\n{src}"
        );
        [files.len(), usize::from(!files.is_empty())]
    });
    eprintln!(
        "read chains: {n} programs, {} with reads, {} reads",
        counts[1], counts[0]
    );
    assert!(counts[0] > 0 && counts[1] > n / 2, "{counts:?}");
}
