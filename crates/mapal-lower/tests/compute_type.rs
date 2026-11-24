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
    let body =
        format!("{ARR} (0.0, r) -> fold {{ acc, k -> acc + a[k] * a[k] }} -> s; s -> println;");
    assert_eq!(sets(&format!("fn main(): tf32 | f32 {{ {body} }}")), (2, 0));
    let restored = body.replace("r -> map {", "r -> map: f32 {");
    assert_eq!(
        sets(&format!("fn main(): tf32 | f32 {{ {restored} }}")),
        (1, 1)
    );
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
    let crossed: Vec<_> = p
        .cross
        .iter()
        .map(|&(o, r)| (val(o), ir.morphism(r).map(|m| m.op.clone())))
        .collect();
    assert_eq!(crossed, [(Some(Value::F32(2.0)), Some(Operation::Mul))]);
    assert!(p.bound.is_empty(), "no tile site");
}

/// ADR-0043 D4: the narrow members `bf16` / `f16` — a step's set carries
/// them (`Precision::half`, the weakest first), innermost wins as for
/// `tf32`, and the formatter keeps the annotation on the opener.
#[test]
fn narrow_members() {
    let half = |src: &str| -> Vec<Option<mapal_ir::NarrowFloat>> {
        let ir: CategoryIr = lower_ok(src);
        ir.morphisms()
            .filter(|(_, m)| m.op == Operation::Mul)
            .filter(|(_, m)| matches!(ir.object(m.target).map(|o| &o.ty), Some(Ty::Float { .. })))
            .map(|(id, _)| ir.precision(id).half())
            .collect()
    };
    let src = format!(
        "fn main() {{ {ARR} (0.0, r) -> fold: bf16 | f32 {{ acc, k -> acc + a[k] * a[k] }} -> s; s -> println; }}"
    );
    let h = half(&src);
    assert_eq!(h.iter().filter(|x| x.is_some()).count(), 1);
    assert!(h.contains(&Some(mapal_ir::NarrowFloat::Bf16)));
    let f = half(&src.replace("bf16 | f32", "f16 | f32"));
    assert!(f.contains(&Some(mapal_ir::NarrowFloat::F16)));
    // Both members: the weaker (`bf16`) is the set's semantics.
    let b = half(&src.replace("bf16 | f32", "f16 | bf16 | f32"));
    assert!(b.contains(&Some(mapal_ir::NarrowFloat::Bf16)));
    let ir: CategoryIr = lower_ok(&src);
    let unit = ir
        .morphisms()
        .find(|(id, _)| ir.precision(*id).half().is_some())
        .map(|(id, _)| ir.precision(id).unit());
    assert_eq!(unit, Some(Some(2f64.powi(-8))));
    // The formatter keeps the opener's annotation.
    let out = mapal_syntax::format_source(&src).expect("formats");
    assert!(out.contains("-> fold: bf16 | f32 {"), "{out}");
}

/// sweep-gemm `round-store` (`precision_plan.store`): an f32 array a map
/// produces whose every element read — through the captures of the nested
/// map and fold — is a multiplicand of a `bf16`-admitting product is rounded
/// once by its producer (`narrow_stored`: the producer's `Narrow`, each read's
/// `Widen`, the carriers' narrow array type); any other reader, an `f16`
/// member or a `{f32}` product keeps it f32.
#[test]
fn precision_plan_store_narrow() {
    let gemv = |set: &str, extra: &str| {
        format!(
            "fn main() {{ 64 -> iota -> ia; ia -> map {{ t -> t -> widen_f32 -> v; v / 7.0 }} -> a;
               256 -> iota -> ib; ib -> map {{ t -> t -> widen_f32 -> v; v / 9.0 -> to_bf16 }} -> w;
               4 -> iota -> o; 64 -> iota -> kk;
               o -> map {{ j -> (0.0, kk) -> fold{set} {{ acc, q -> w[q * 4 + j] -> widen_f32 -> y; acc + a[q] * y }} }} -> c;
               c[0] -> println; {extra} }}"
        )
    };
    let stores = |src: &str| -> usize {
        let ir: CategoryIr = lower_ok(src);
        ir.funcs()
            .map(|(f, _)| ir.precision_plan(f).store.len())
            .sum()
    };
    let src = gemv(": bf16 | f32", "");
    assert_eq!(stores(&src), 1, "`a` rounds once at its producer");
    let ir: CategoryIr = lower_ok(&src);
    let g = ir.narrow_stored().expect("rewritten");
    assert!(
        mapal_ir::validate(&g).is_empty(),
        "{:?}",
        mapal_ir::validate(&g)
    );
    assert!(g.narrow_stored().is_none(), "idempotent");
    let count = |ir: &CategoryIr, op: Operation| ir.morphisms().filter(|(_, m)| m.op == op).count();
    assert_eq!(
        count(&g, Operation::Narrow),
        count(&ir, Operation::Narrow) + 1
    );
    assert_eq!(
        count(&g, Operation::Widen),
        count(&ir, Operation::Widen) + 1
    );
    let narrow_arrays = |ir: &CategoryIr| {
        ir.objects()
            .filter(|(_, o)| matches!(&o.ty, Ty::Array { elem, .. } if **elem == Ty::Narrow(mapal_ir::NarrowFloat::Bf16)))
            .count()
    };
    assert!(narrow_arrays(&g) > narrow_arrays(&ir));
    // Any other reader keeps it f32: a print, a `{f32}` product, an `f16`
    // member (no value-range fact), an elementwise use.
    assert_eq!(stores(&gemv(": bf16 | f32", "a[3] -> println;")), 0);
    assert_eq!(stores(&gemv("", "")), 0);
    assert_eq!(stores(&gemv(": f16 | f32", "")), 0);
    assert_eq!(
        stores(&gemv(
            ": bf16 | f32",
            "a -> map { x -> x + 1.0 } -> b2; b2[0] -> println;"
        )),
        0
    );
    assert_eq!(
        stores(&gemv(
            ": bf16 | f32",
            "(0.0, kk) -> fold { s, q -> s + a[q] * 2.0 } -> z; z -> println;"
        )),
        0
    );
    // An array that escapes its function keeps its f32 bytes: a callee the
    // inliner keeps (it has a loop) returning `a` beside the product's cell —
    // its caller reads `[f32; 64]`. Returning only the cell, `a` is stored.
    let callee = |ret: &str, read: &str| {
        format!(
            "fn mk(n: i32) -> {ret} {{
                mut k: i32 <- 0;
                mut z: i32 <- 0;
                loop {{
                    (k < n) -> {{
                        -true-> {{
                            z + k -> z;
                            k + 1 -> k;
                            -> loop;
                        }}
                        -false-> z -> zz;
                    }}
                }}
                zz -> widen_f32 -> zf;
                64 -> iota -> ia;
                ia -> map {{ t -> t -> widen_f32 -> v; v / 7.0 + zf }} -> a;
                256 -> iota -> ib;
                ib -> map {{ t -> t -> widen_f32 -> v; v / 9.0 -> to_bf16 }} -> w;
                4 -> iota -> o;
                64 -> iota -> kk;
                o -> map {{ j -> (0.0, kk) -> fold: bf16 | f32 {{ acc, q -> w[q * 4 + j] -> widen_f32 -> y; acc + a[q] * y }} }} -> c;
                {read}
            }}
            fn main() {{ 3 -> mk -> r; {print} }}",
            print = if ret.starts_with('(') {
                "r.0 -> x; x[3] -> println; r.1 -> println;"
            } else {
                "r -> println;"
            }
        )
    };
    let escaping = callee("([f32; 64], f32)", "a -> ret.0; c[0] -> ret.1;");
    let kept = callee("f32", "c[0] -> ret;");
    let calls = |src: &str| {
        let ir: CategoryIr = lower_ok(src);
        ir.morphisms()
            .filter(|(_, m)| matches!(m.op, Operation::Call(_)))
            .count()
    };
    assert!(calls(&escaping) > 0, "the callee stays a call");
    assert_eq!(stores(&escaping), 0, "`a` escapes through the return");
    assert_eq!(stores(&kept), 1, "`a` dies in its callee: stored");
}
