//! ADR-0047 on generated programs (`testgen::generic_prog_strategy`): size-
//! generic helpers called at several sizes lower to the IR their
//! hand-specialized twins lower to (names and spans aside), and compute the
//! twin's values on the interpreter before and after `rewrite` — over
//! programs the change was not designed on. Coverage counters: helpers at two
//! or more sizes, nested generic calls, annotation-bound sizes, repeated
//! unknowns.

mod testgen;

use mapal_interp::{Outcome, run};
use mapal_ir::CategoryIr;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

const BUDGET: u64 = 50_000_000;

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(
        po.diagnostics.is_empty(),
        "parse: {:?}\n{src}",
        po.diagnostics
    );
    mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}\n{src}"))
}

/// The IR's `Debug` text without source spans, instance names mapped to the
/// twin's.
fn normalized(ir: &CategoryIr, renames: &[(String, String)]) -> String {
    let mut text: String = format!("{ir:#?}")
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            !(t.starts_with("start: ") || t.starts_with("end: "))
        })
        .collect::<Vec<_>>()
        .join("\n");
    for (from, to) in renames {
        text = text.replace(&format!("\"{from}"), &format!("\"{to}"));
    }
    text
}

fn output(ir: &CategoryIr, src: &str) -> String {
    let r = run(ir, BUDGET);
    assert!(
        matches!(r.outcome, Outcome::Done(_)),
        "{:?}\n{src}",
        r.outcome
    );
    r.output
}

#[test]
fn generic_programs_lower_to_their_twins_ir() {
    let n: usize = std::env::var("MAPAL_TESTGEN_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let mut runner = TestRunner::deterministic();
    let strat = testgen::generic_prog_strategy();
    let progs: Vec<_> = (0..n)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let [programs, instances, multi, nested, annotated, repeated] =
        testgen::par_count(&progs, |i, p| {
            let (src, hand) = (p.source(), p.hand());
            // `MAPAL_TESTGEN_DUMP=<dir>`: each program and its twin as files.
            if let Ok(dir) = std::env::var("MAPAL_TESTGEN_DUMP") {
                std::fs::write(format!("{dir}/generic_{i:03}.mapal"), &src).unwrap();
                std::fs::write(format!("{dir}/hand_{i:03}.mapal"), &hand).unwrap();
            }
            let (g, h) = (lower(&src), lower(&hand));
            assert_eq!(
                normalized(&g, &p.renames()),
                normalized(&h, &[]),
                "\n{src}\n{hand}"
            );
            let want = output(&h, &hand);
            assert_eq!(output(&g, &src), want, "\n{src}");
            let rewritten = mapal_rewrite::rewrite(g).ir;
            assert_eq!(output(&rewritten, &src), want, "rewritten\n{src}");
            [
                1,
                p.renames().len(),
                p.multi_size(),
                p.calls_of("norm"),
                p.calls_of("ramp"),
                p.calls_of("trace"),
            ]
        });
    eprintln!(
        "size generics: {programs} programs, {instances} instances, {multi} helpers at ≥ 2 sizes, \
         {nested} nested, {annotated} annotation-bound, {repeated} repeated-unknown calls"
    );
    assert!(multi > 0 && nested > 0 && annotated > 0 && repeated > 0);
}
