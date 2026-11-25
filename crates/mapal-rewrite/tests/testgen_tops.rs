//! ADR-0048 on generated programs (`testgen::generic_prog_strategy` rendered
//! by `GenericProg::tops`), over programs the change was not designed on:
//!
//! - written with top-level bindings (lengths and a scale, some computed, in a
//!   shuffled order) and the size-generic alias `Vec<n>` alone, a program
//!   lowers to the IR of the program without them (names and spans aside) —
//!   compile-time evaluation leaves literals, an alias costs nothing;
//! - with one-array operands passed through a size-generic struct's `_` and
//!   results scaled through a top-level table, it computes the same values
//!   on the interpreter, after `rewrite`, and after compile-time evaluation
//!   (`fold_static`) and `rewrite`;
//! - compile-time evaluation of the plain programs (whose operands are all
//!   laws) keeps every value, embeds roots, and drops their cones.
//!
//! Coverage counters are asserted non-zero.

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

/// The IR's `Debug` text without source spans.
fn normalized(ir: &CategoryIr) -> String {
    format!("{ir:#?}")
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            !(t.starts_with("start: ") || t.starts_with("end: "))
        })
        .collect::<Vec<_>>()
        .join("\n")
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
fn top_level_programs_lower_and_compute_like_their_plain_twins() {
    let n: usize = std::env::var("MAPAL_TESTGEN_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(96);
    let mut runner = TestRunner::deterministic();
    let strat = testgen::generic_prog_strategy();
    let progs: Vec<_> = (0..n)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let [
        programs,
        bindings,
        computed,
        aliases,
        defaults,
        tables,
        embedded,
        dropped,
    ] = testgen::par_count(&progs, |i, p| {
        let plain = p.source();
        let (pure, _) = p.tops(true);
        let (full, u) = p.tops(false);
        if let Ok(dir) = std::env::var("MAPAL_TESTGEN_DUMP") {
            std::fs::write(format!("{dir}/tops_pure_{i:03}.mapal"), &pure).unwrap();
            std::fs::write(format!("{dir}/tops_full_{i:03}.mapal"), &full).unwrap();
        }
        let base = lower(&plain);
        assert_eq!(
            normalized(&lower(&pure)),
            normalized(&base),
            "\n{pure}\n{plain}"
        );
        let want = output(&base, &plain);
        let full_ir = lower(&full);
        assert_eq!(output(&full_ir, &full), want, "\n{full}");
        let rewritten = mapal_rewrite::rewrite(lower(&full)).ir;
        assert_eq!(output(&rewritten, &full), want, "rewritten\n{full}");
        let (folded, _) = mapal_rewrite::fold_static(full_ir);
        let folded = mapal_rewrite::rewrite(folded).ir;
        assert_eq!(output(&folded, &full), want, "folded\n{full}");
        // Compile-time evaluation of the plain program.
        let objects = base.objects().count();
        let (static_ir, report) = mapal_rewrite::fold_static(base);
        assert_eq!(output(&static_ir, &plain), want, "static\n{plain}");
        let after = mapal_rewrite::rewrite(static_ir).ir;
        assert_eq!(output(&after, &plain), want, "static+rewrite\n{plain}");
        [
            1,
            u.bindings,
            u.computed,
            u.aliases,
            u.defaults,
            u.tables,
            report.embedded as usize,
            usize::from(after.objects().count() < objects),
        ]
    });
    eprintln!(
        "top-level: {programs} programs, {bindings} bindings ({computed} computed), {aliases} \
         alias uses, {defaults} `_` reads, {tables} table reads; compile-time evaluation: \
         {embedded} roots embedded, {dropped} programs smaller"
    );
    assert!(computed > 0 && aliases > 0 && defaults > 0 && tables > 0);
    assert!(embedded > 0 && dropped > 0);
}
