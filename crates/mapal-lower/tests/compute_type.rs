//! ADR-0041 D2: `: compute` on anything that opens a body — a step
//! (`-> fold: tf32 | f32 { … }`), a block (`loop: … { … }`, `seq: … { … }`),
//! a function after its output (`fn f(a) -> T: … { … }`). Every float product
//! lowered from that body carries the set (D1); nested bodies inherit unless
//! they carry their own (innermost wins); `: f32` restores the default; a
//! called function's own type governs its body.

mod common;
use common::lower_ok;
use mapal_ir::{CategoryIr, Operation, Ty, Value};

/// `(products admitting tf32, f32-only products)` over every float `Mul`.
fn sets(src: &str) -> (usize, usize) {
    let ir: CategoryIr = lower_ok(src);
    let muls: Vec<bool> = ir
        .morphisms()
        .filter(|(_, m)| m.op == Operation::Mul)
        .filter(|(_, m)| matches!(ir.object(m.target).map(|o| &o.ty), Some(Ty::Float { .. })))
        .map(|(id, _)| ir.precision(id).admits_tf32())
        .collect();
    let on = muls.iter().filter(|&&t| t).count();
    (on, muls.len() - on)
}

const ARR: &str = "4 -> iota -> r; r -> map { t -> t -> widen_f32 -> v; v * 3.0 } -> a;";

#[test]
fn step_form_covers_its_own_body_only() {
    let src = format!(
        "fn main() {{ {ARR} (0.0, r) -> fold: tf32 | f32 {{ acc, k -> acc + a[k] * a[k] }} -> s; s -> println; }}"
    );
    assert_eq!(sets(&src), (1, 1));
    // `| f32` is implied (D3): `: tf32` alone admits the same.
    assert_eq!(sets(&src.replace(": tf32 | f32", ": tf32")), (1, 1));
}

#[test]
fn function_form_covers_nested_steps_and_f32_restores() {
    let body = format!("{ARR} (0.0, r) -> fold {{ acc, k -> acc + a[k] * a[k] }} -> s; s -> println;");
    assert_eq!(sets(&format!("fn main(): tf32 | f32 {{ {body} }}")), (2, 0));
    let restored = body.replace("r -> map {", "r -> map: f32 {");
    assert_eq!(sets(&format!("fn main(): tf32 | f32 {{ {restored} }}")), (1, 1));
    assert_eq!(sets(&format!("fn main() {{ {body} }}")), (0, 2));
}

#[test]
fn seq_block_covers_its_body() {
    let src = "fn main() { 2.0 -> x; x -> seq: tf32 | f32 { x * 3.0 -> y; y } -> z; z * 5.0 -> w; w -> println; }";
    assert_eq!(sets(src), (1, 1));
}

/// The function form follows the output; a call does not inherit the caller's
/// type, and a `loop:` block covers its body.
#[test]
fn loop_block_and_calls() {
    let cell = |fn_ty: &str, loop_ty: &str| {
        format!(
            "fn cell(a: [f32; 4]) -> f32{fn_ty} {{
    a[0] * a[1] -> s0;
    mut k: i32 <- 0;
    mut acc: f32 <- s0;
    loop{loop_ty} {{
        (k < 4) -> {{
            -true-> {{ acc + a[k] * a[k] -> acc; k + 1 -> k; -> loop; }}
            -false-> acc -> ret;
        }}
    }}
}}
fn main(): tf32 | f32 {{ {ARR} a -> cell -> c; c -> println; }}
"
        )
    };
    // main's map product admits tf32 (its own type); cell's are f32 (no inheritance).
    assert_eq!(sets(&cell("", "")), (1, 2));
    assert_eq!(sets(&cell("", ": tf32 | f32")), (2, 1));
    assert_eq!(sets(&cell(": tf32 | f32", "")), (3, 0));
    assert_eq!(sets(&cell(": tf32 | f32", ": f32")), (2, 1));
}

/// `precision_plan` (D5): a value every reader of which is a `tf32` product
/// rounds once at its producer (`round_at`); one an f32 reader shares is a
/// `cross` edge to that reader; `set` holds the `tf32` products only.
#[test]
fn precision_plan_round_at_and_cross() {
    let ir: CategoryIr = lower_ok(
        "fn main() { 2.0 -> x: f32; 7.0 -> u: f32; x -> seq: tf32 | f32 { x * u -> y; y } -> z; x * 5.0 -> w; z + w -> s; s -> println; }",
    );
    let (f, _) = ir.funcs().next().expect("main");
    let p = ir.precision_plan(f);
    let val = |o| ir.object(o).and_then(|x| x.value.clone());
    assert_eq!(p.set.len(), 1, "one tf32 product");
    // `u` (7.0) only feeds the tf32 product; `x` (2.0) also the f32 one, `w`.
    let rounded: Vec<_> = p.round_at.iter().map(|&o| val(o)).collect();
    assert_eq!(rounded, [Some(Value::F32(7.0))]);
    let crossed: Vec<_> = p.cross.iter().map(|&(o, r)| (val(o), ir.morphism(r).map(|m| m.op.clone()))).collect();
    assert_eq!(crossed, [(Some(Value::F32(2.0)), Some(Operation::Mul))]);
    assert!(p.bound.is_empty(), "no tile site");
}
