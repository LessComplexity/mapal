//! ADR-0043 D1/D2: `bf16` and `f16` are storage types — arrays, bindings,
//! loop state and guards hold them; `to_bf16` / `to_f16` round an `f32` once
//! (L1616 otherwise; a literal source is an `f32`), `widen_f32` / `widen_f64`
//! widen them exactly; no arithmetic, comparison or print takes one; the
//! names are reserved. (The oracle's bits: `mapal-interp/tests/narrow.rs`.)

mod common;
use common::{assert_rejects, lower_ok};
use mapal_ir::{CategoryIr, Operation, Ty};

fn narrows(ir: &CategoryIr) -> Vec<Ty> {
    ir.morphisms()
        .filter(|(_, m)| m.op == Operation::Narrow)
        .map(|(_, m)| ir.object(m.target).unwrap().ty.clone())
        .collect()
}

#[test]
fn rounding_a_non_f32_is_l1616() {
    assert_rejects(
        "fn main() { 4 -> iota -> r; r[1] -> to_bf16 -> widen_f32 -> println; }",
        "L1616",
    );
    // An integer literal is typed `f32` by the stage first (L1203).
    assert_rejects(
        "fn main() { 3 -> to_bf16 -> widen_f32 -> println; }",
        "L1203",
    );
    assert_rejects(
        "fn main() { 3 -> widen_f64 -> to_f16 -> widen_f32 -> println; }",
        "L1616",
    );
    assert_rejects(
        "fn main() { 1.0 -> to_bf16 -> to_f16 -> widen_f32 -> println; }",
        "L1616",
    );
}

#[test]
fn no_arithmetic_comparison_or_print_on_narrow() {
    for src in [
        "fn main() { 1.0 -> to_bf16 -> a; a + a -> b; b -> widen_f32 -> println; }",
        "fn main() { 1.0 -> to_f16 -> a; a * 2.0 -> println; }",
        "fn main() { 1.0 -> to_bf16 -> a; (a < a) -> println; }",
        "fn main() { 1.0 -> to_bf16 -> println; }",
        "fn main() { 1.0 -> to_bf16 -> a; a -> widen_i64 -> println; }",
    ] {
        let po = mapal_syntax::parse(src);
        assert!(po.diagnostics.is_empty(), "{src}");
        assert!(mapal_lower::lower(src, &po.program).is_err(), "{src}");
    }
}

#[test]
fn names_are_reserved() {
    assert_rejects(
        "fn to_bf16(x: f32) -> f32 { x -> ret; } fn main() { 1.0 -> println; }",
        "L1009",
    );
    assert_rejects(
        "fn f16(x: f32) -> f32 { x -> ret; } fn main() { 1.0 -> println; }",
        "L1009",
    );
}

#[test]
fn formatter_keeps_the_stages_and_types() {
    let src = "fn main() {\n    mut w: [bf16; 4] <- z;\n    1.0 -> to_f16 -> h;\n    h -> widen_f32 -> println;\n}\n";
    let out = mapal_syntax::format_source(src).expect("formats");
    assert_eq!(out, src);
}

#[test]
fn conversions_lower_to_narrow_and_widen() {
    let ir = lower_ok(
        "fn main() {
    8 -> iota -> r;
    r -> map { t -> t -> widen_f32 -> v; v / 3.0 -> to_bf16 } -> a;
    r -> map { t -> t -> widen_f32 -> v; v * 9000.5 -> to_f16 } -> h;
    mut s: [bf16; 8] <- a;
    0.25 -> to_bf16 -> q;
    s[2] <- q;
    (0.0, r) -> fold { acc, k -> s[k] -> widen_f32 -> x; h[k] -> widen_f64 -> y; acc + x * 2.0 } -> tot;
    tot -> println;
}",
    );
    let n = narrows(&ir);
    assert!(n.contains(&Ty::bf16()) && n.contains(&Ty::f16()), "{n:?}");
    let widened: Vec<Ty> = ir
        .morphisms()
        .filter(|(_, m)| {
            m.op == Operation::Widen && matches!(ir.object(m.source).unwrap().ty, Ty::Narrow(_))
        })
        .map(|(_, m)| ir.object(m.target).unwrap().ty.clone())
        .collect();
    assert!(widened.contains(&Ty::f32()) && widened.contains(&Ty::f64()));
}
