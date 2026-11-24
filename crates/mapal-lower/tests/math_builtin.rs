//! ADR-0046 D1: `sqrt`, `sin` and `cos` are pure pipeline builtins like `exp`
//! — `x -> sqrt` (never `sqrt(x)`), the float width kept, legal in map and
//! fold bodies and fanout, reserved names; any other operand is L1615. The
//! formatter keeps the stages as written.

mod common;
use common::{assert_rejects, lower_ok};
use mapal_ir::{CategoryIr, MathFn, Operation, Ty};

/// Every math op of `src` with its width, in graph order.
fn maths(src: &str) -> Vec<(MathFn, Ty)> {
    let ir: CategoryIr = lower_ok(src);
    ir.morphisms()
        .filter_map(|(_, m)| match m.op {
            Operation::Math(f) => Some((f, ir.object(m.target).unwrap().ty.clone())),
            _ => None,
        })
        .collect()
}

#[test]
fn math_builtins_keep_the_width() {
    assert_eq!(
        maths("fn main() { 2.5 -> sqrt -> println; }"),
        [(MathFn::Sqrt, Ty::f64())]
    );
    assert_eq!(
        maths("fn main() { 2.5 -> x: f32; x -> sin -> println; }"),
        [(MathFn::Sin, Ty::f32())]
    );
    assert_eq!(
        maths("fn main() { 3 -> widen_f64 -> cos -> c; c -> println; }"),
        [(MathFn::Cos, Ty::f64())]
    );
    // Chained, in a map body and a fold body, beside `exp`.
    let src = "fn main() {
    8 -> iota -> r;
    r -> map { t -> t -> widen_f32 -> v; v / 4.0 -> sin -> exp } -> a;
    (0.0, r) -> fold { acc, k -> a[k] * a[k] -> sqrt -> w; w -> cos -> c; acc + c } -> s;
    s -> println;
}";
    assert_eq!(
        maths(src),
        [
            (MathFn::Sin, Ty::f32()),
            (MathFn::Exp, Ty::f32()),
            (MathFn::Sqrt, Ty::f32()),
            (MathFn::Cos, Ty::f32()),
        ]
    );
    // `Operation::Exp` is `Math(Exp)`.
    assert_eq!(Operation::Exp, Operation::Math(MathFn::Exp));
}

#[test]
fn math_of_a_non_float_is_l1615() {
    for f in ["sqrt", "sin", "cos"] {
        assert_rejects(&format!("fn main() {{ 3 -> {f} -> println; }}"), "L1615");
        assert_rejects(
            &format!(
                "fn main() {{ 4 -> iota -> r; r -> map {{ t -> t -> {f} }} -> a; a[0] -> println; }}"
            ),
            "L1615",
        );
        assert_rejects(&format!("fn main() {{ true -> {f} -> println; }}"), "L1615");
    }
}

#[test]
fn math_names_are_reserved() {
    for f in ["sqrt", "sin", "cos"] {
        assert_rejects(
            &format!(
                "fn {f}(x: f32) -> f32 {{ x -> ret; }} fn main() {{ 1.0 -> {f} -> println; }}"
            ),
            "L1009",
        );
    }
}

#[test]
fn formatter_keeps_the_stages() {
    let src =
        "fn main() {\n    2.0 -> sqrt -> r;\n    r -> sin -> s;\n    s -> cos -> println;\n}\n";
    assert_eq!(mapal_syntax::format_source(src).expect("formats"), src);
}
