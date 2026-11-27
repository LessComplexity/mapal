//! ADR-0051 on generated programs (`testgen::format_prog_strategy`): helpers
//! generic over a size and a format (`fn store<n, t: float>`), called with a
//! drawn prefix of explicit arguments and the rest inferred (from the
//! argument's format or the result's annotation), lower to the IR of their
//! hand-specialized twins (names and spans aside) and compute the twin's
//! values on the interpreter, before and after `rewrite` — over programs the
//! change was not designed on. Coverage counters: explicit arguments,
//! annotation-bound formats, argument-bound formats, `f32` instances (the
//! conversions' identity).

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
fn format_generic_programs_lower_to_their_twins_ir() {
    let n: usize = std::env::var("MAPAL_TESTGEN_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(160);
    let mut runner = TestRunner::deterministic();
    let strat = testgen::format_prog_strategy();
    let progs: Vec<_> = (0..n)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let [
        programs,
        instances,
        explicit,
        by_result,
        by_argument,
        identity,
    ] = testgen::par_count(&progs, |i, p| {
        let (src, hand) = (p.source(), p.hand());
        if let Ok(dir) = std::env::var("MAPAL_TESTGEN_DUMP") {
            std::fs::write(format!("{dir}/format_{i:03}.mapal"), &src).unwrap();
            std::fs::write(format!("{dir}/format_hand_{i:03}.mapal"), &hand).unwrap();
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
        let c = p.counters();
        [1, p.renames().len(), c[0], c[1], c[2], c[3]]
    });
    eprintln!(
        "format generics: {programs} programs, {instances} instances, {explicit} calls with \
         explicit arguments, {by_result} formats from the result's annotation, {by_argument} from \
         the argument, {identity} f32 calls (identity conversions)"
    );
    assert!(explicit > 0 && by_result > 0 && by_argument > 0 && identity > 0);
}
