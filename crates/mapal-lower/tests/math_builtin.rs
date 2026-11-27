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
        maths("fn main() { 3 -> widen<f64> -> cos -> c; c -> println; }"),
        [(MathFn::Cos, Ty::f64())]
    );
    // Chained, in a map body and a fold body, beside `exp`.
    let src = "fn main() {
    8 -> iota -> r;
    r -> map { t -> t -> widen<f32> -> v; v / 4.0 -> sin -> exp } -> a;
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

/// ADR-0046 D9: `precision_plan.math` records each math builtin with its
/// width's contract (and `exp` keeps ADR-0042's `exp` entry).
#[test]
fn precision_plan_records_each_builtin_and_its_contract() {
    let ir = lower_ok(
        "fn main() { 2.0 -> x: f32; x -> sqrt -> a; x -> sin -> b; 3.0 -> cos -> c; x -> exp -> d; \
         a + b + d -> println; c -> println; }",
    );
    let plan = ir.precision_plan(ir.entry());
    let mut got: Vec<(MathFn, u32, f64)> = plan
        .math
        .values()
        .map(|m| (m.f, m.bits, m.contract.rel))
        .collect();
    got.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        got,
        [
            (MathFn::Exp, 32, 2f64.powi(-21)),
            (MathFn::Sqrt, 32, 2f64.powi(-22)),
            (MathFn::Sin, 32, 2f64.powi(-21)),
            (MathFn::Cos, 64, 2f64.powi(-50)),
        ]
    );
    assert_eq!(plan.exp.len(), 1);
}
