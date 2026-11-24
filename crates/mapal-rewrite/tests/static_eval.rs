//! ADR-0048 D3 — the `static_plan` fact and `fold_static`: which values are
//! computable before the program runs (no input slot, no IO token in their
//! cone), the rule's realization of each root (bytes against work), and the
//! values after compile-time evaluation (the oracle's, the cone dropped; a
//! trap left to run time).

use mapal_interp::{Outcome, run};
use mapal_ir::{CategoryIr, Operation, Realize};

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "{:?}", po.diagnostics);
    mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("{d:?}"))
}

fn realizations(ir: &CategoryIr) -> Vec<Realize> {
    ir.static_plan(ir.entry())
        .roots
        .iter()
        .map(|r| r.realize)
        .collect()
}

fn ops(ir: &CategoryIr, pick: impl Fn(Operation) -> bool) -> usize {
    ir.morphisms().filter(|(_, m)| pick(m.op)).count()
}

/// The oracle's output, before and after `fold_static` + `rewrite`, equal;
/// the folded IR.
fn folded(src: &str) -> (CategoryIr, String) {
    let want = run(&lower(src), 100_000_000);
    let (ir, _) = mapal_rewrite::fold_static(lower(src));
    let ir = mapal_rewrite::rewrite(ir).ir;
    let got = run(&ir, 100_000_000);
    assert_eq!(got.outcome, want.outcome, "{src}");
    assert_eq!(got.output, want.output, "{src}");
    (ir, got.output)
}

#[test]
fn a_reduction_of_laws_is_embedded_and_its_cone_dropped() {
    let src = "fn main() {
    64 -> iota -> map { i -> i * i % 7 } -> a;
    (0, a) -> fold { s, x -> s + x } -> total;
    total -> println;
}";
    assert_eq!(realizations(&lower(src)), vec![Realize::Embed]);
    let (ir, out) = folded(src);
    assert_eq!(out, "126\n");
    assert_eq!(
        ops(&ir, |o| matches!(
            o,
            Operation::Map { .. } | Operation::Fold { .. }
        )),
        0
    );
}

#[test]
fn a_value_of_an_effect_or_an_input_is_not_computable() {
    // `time` is an effect: its value, and everything from it, runs.
    let src = "fn main() {
    () -> time -> t;
    t * 2.0 -> u;
    u > 0.0 -> println;
}";
    let ir = lower(src);
    assert!(ir.static_plan(ir.entry()).roots.is_empty());
    // A function's parameter is a run-time input; a constant cone inside it
    // is computable.
    let src = "fn f(x: f32) -> f32 {
    4 -> iota -> map { i -> i -> widen_f32 } -> w;
    (0.0, w) -> fold { s, v -> s + v } -> six;
    x * six -> ret;
}
fn main() { 2.0 -> f -> println; }";
    let ir = lower(src);
    let f = ir.funcs().find(|(_, d)| d.name == "f").unwrap().0;
    let sp = ir.static_plan(f);
    assert_eq!(sp.roots.len(), 1);
    assert!(!sp.is_fixed(ir.func(f).unwrap().input));
    let (_, out) = folded(src);
    assert_eq!(out, "12\n");
}

#[test]
fn a_law_is_cheaper_than_its_value_and_a_large_value_stays_computed() {
    // A bare `iota` is as cheap as its cells: kept a law.
    let src = "fn f(x: [i32; 8]) -> i32 { x[3] -> ret; }
fn main() { 8 -> iota -> f -> println; }";
    assert_eq!(realizations(&lower(src)), vec![Realize::Embed]);
    // 8192 cells (32 KiB) of a law: too large to embed.
    let src = "fn h(x: [f32; 8192], k: i32) -> f32 { x[k] -> ret; }
fn main() {
    () -> time -> t;
    t > 0.0 -> on;
    on -> map_k -> k;
    8192 -> iota -> map { i -> i -> widen_f32 -> * 0.5 } -> big;
    (big, k) -> h -> println;
}
fn map_k(b: bool) -> i32 { 3 -> ret; }";
    let ir = lower(src);
    let roots = ir.static_plan(ir.entry()).roots;
    let big = roots
        .iter()
        .find(|r| r.cells == 8192)
        .expect("the table is a root");
    assert_eq!(big.realize, Realize::Compute);
    let (_, out) = folded(src);
    assert_eq!(out, "1.5\n");
}

#[test]
fn a_loop_is_attempted_under_the_budget() {
    // Heron's square root of 2 as a counted loop: work unknown before it runs.
    let src = "fn main() {
    (0, 1.0) -> mut s;
    loop {
        s.0 < 20 -> {
            -true-> { (s.0 + 1, (s.1 + 2.0 / s.1) * 0.5) -> s; -> loop; }
            -false-> { s.1 -> r; }
        }
    }
    r -> println;
}";
    let ir = lower(src);
    assert_eq!(realizations(&ir), vec![Realize::Attempt]);
    let (folded_ir, out) = folded(src);
    assert!(out.starts_with("1.41421"), "{out}");
    assert!(
        folded_ir
            .funcs()
            .all(|(f, _)| folded_ir.loop_structure(f).is_empty())
    );
}

#[test]
fn a_trapping_value_stays_to_trap_at_run_time() {
    let src = "fn main() {
    [1, 2, 3] -> a;
    a[7] -> x;
    x -> println;
}";
    let want = run(&lower(src), 1_000_000);
    assert!(matches!(want.outcome, Outcome::Trapped(_)));
    let (ir, report) = mapal_rewrite::fold_static(lower(src));
    assert_eq!(report.failed, 1);
    let got = run(&mapal_rewrite::rewrite(ir).ir, 1_000_000);
    assert_eq!(got.outcome, want.outcome);
}

#[test]
fn a_struct_value_is_embedded_as_a_pack_of_constants() {
    let src = "type P { a: [i32; 3], b: f32 }
fn make(k: i32) -> P { P { a: [k, k * 2, k * 3], b: 0.5 } -> ret; }
fn pick(p: P, i: i32) -> i32 { p.a[i] -> ret; }
fn main() {
    () -> time -> t;
    t > 0.0 -> on;
    on -> idx -> i;
    4 -> make -> p;
    (p, i) -> pick -> println;
}
fn idx(b: bool) -> i32 { 2 -> ret; }";
    let ir = lower(src);
    let roots = ir.static_plan(ir.entry()).roots;
    assert!(
        roots
            .iter()
            .any(|r| r.cells == 4 && r.realize == Realize::Embed)
    );
    let (_, out) = folded(src);
    assert_eq!(out, "12\n");
}
