//! ADR-0042 D1: `exp` is a pure pipeline builtin, `x -> exp` (or
//! `x -> exp -> y`), resolved by name like `widen_f32` — no call syntax. It
//! keeps the float width (`f32 → f32`, `f64 → f64`), is legal in map and fold
//! bodies and fanout, and is a reserved name; any other operand type is
//! L1615. The formatter keeps the stage as written.

mod common;
use common::{assert_rejects, lower_ok};
use mapal_ir::{CategoryIr, Operation, Ty};

/// The width of every `Exp` in `src`, in graph order.
fn exps(src: &str) -> Vec<Ty> {
    let ir: CategoryIr = lower_ok(src);
    ir.morphisms()
        .filter(|(_, m)| m.op == Operation::Exp)
        .map(|(_, m)| ir.object(m.target).unwrap().ty.clone())
        .collect()
}

#[test]
fn exp_in_pipeline_position_keeps_the_width() {
    // An unpinned literal takes its default width (f64); an annotation pins it.
    assert_eq!(exps("fn main() { 2.5 -> exp -> println; }"), [Ty::f64()]);
    assert_eq!(
        exps("fn main() { 2.5 -> x: f32; x -> exp -> println; }"),
        [Ty::f32()]
    );
    assert_eq!(
        exps("fn main() { 3 -> widen_f64 -> exp -> e; e -> println; }"),
        [Ty::f64()]
    );
    // In a map body and a fold body, bound and unbound.
    let src = "fn main() {
    8 -> iota -> r;
    r -> map { t -> t -> widen_f32 -> v; v / 4.0 -> exp } -> a;
    (0.0, r) -> fold { acc, k -> a[k] - 1.0 -> exp -> w; acc + w } -> s;
    s -> println;
}";
    assert_eq!(exps(src), [Ty::f32(), Ty::f32()]);
    // A literal flows through at the width it unifies with.
    assert_eq!(
        exps("fn main() { 0.5 -> exp -> e: f64; e -> println; }"),
        [Ty::f64()]
    );
}

#[test]
fn exp_of_a_non_float_is_l1615() {
    assert_rejects("fn main() { 3 -> exp -> println; }", "L1615");
    assert_rejects(
        "fn main() { 4 -> iota -> r; r -> map { t -> t -> exp } -> a; a[0] -> println; }",
        "L1615",
    );
}

#[test]
fn exp_is_reserved() {
    assert_rejects(
        "fn exp(x: f32) -> f32 { x -> ret; } fn main() { 1.0 -> exp -> println; }",
        "L1009",
    );
}

#[test]
fn formatter_keeps_the_stage() {
    let src = "fn main() {\n    2.0 -> exp -> e;\n    e -> println;\n}\n";
    let out = mapal_syntax::format_source(src).expect("formats");
    assert_eq!(out, src);
}
