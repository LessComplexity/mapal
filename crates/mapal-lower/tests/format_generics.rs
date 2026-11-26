//! ADR-0051: format generics. Format parameters with their kind (D1), explicit
//! arguments at a stage (D2), inference from the explicit prefix, the
//! argument and the expected type (D3), the conversion builtins `to` /
//! `widen` as generic signatures with their admitted pairs — `widen` exact by
//! the format's embedding, a compile-time value or a proven range (D4) —,
//! monomorphization and naming (D5), the removed names (D6), and every
//! rejection code (L1708–L1713, L1614, L1616).

mod common;
use common::{assert_rejects, lower_ok};
use mapal_ir::{CategoryIr, FuncKind, Operation, Ty};

fn named(ir: &CategoryIr) -> Vec<String> {
    ir.funcs()
        .filter(|(_, f)| f.kind == FuncKind::Named)
        .map(|(_, f)| f.name.clone())
        .collect()
}

fn lower_diags(src: &str) -> Vec<mapal_syntax::Diagnostic> {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "{:?}", po.diagnostics);
    mapal_lower::lower(src, &po.program).expect_err("expected a rejection")
}

/// The diagnostic with `code`, its message.
fn message(src: &str, code: &str) -> String {
    let ds = lower_diags(src);
    ds.iter()
        .find(|d| d.code.0 == code)
        .unwrap_or_else(|| panic!("no {code} in {ds:?}"))
        .message
        .clone()
}

fn count(ir: &CategoryIr, op: Operation) -> usize {
    ir.morphisms().filter(|(_, m)| m.op == op).count()
}

const TOTAL: &str = "fn total<n, t: float>(xs: [t; n]) -> f32 {
    n -> iota -> idx;
    (0.0, idx) -> fold { acc, i -> xs[i] -> widen<f32> -> x; acc + x } -> ret;
}
";

const STORE: &str = "fn store<n, cache: float>(x: [f32; n]) -> [cache; n] {
    x -> map { v -> v -> to<cache> } -> ret;
}
";

#[test]
fn a_format_parameter_binds_to_the_arguments_format() {
    let src = format!(
        "{TOTAL}
fn main() {{
    (0.5, 4) -> fill -> xs: [f32; 4];
    xs -> map {{ v -> v -> to<bf16> }} -> b;
    xs -> map {{ v -> v -> to<f16> }} -> h;
    b -> total -> sb;
    h -> total -> sh;
    xs -> total -> sx;
    sb -> println;
    sh -> println;
    sx -> println;
}}"
    );
    let ir = lower_ok(&src);
    // One instance per (size, format), formats by name in their order
    // (`f32`, `bf16`, `f16`, …: ADR-0047 D6's declared-order sort).
    assert_eq!(
        named(&ir),
        ["total<4, f32>", "total<4, bf16>", "total<4, f16>", "main"]
    );
}

#[test]
fn explicit_prefix_then_the_expected_type() {
    // Every argument given.
    let all = lower_ok(&format!(
        "{STORE}fn main() {{ (0.5, 4) -> fill -> xs: [f32; 4]; xs -> store<4, bf16> -> y; y[0] -> widen<f32> -> println; }}"
    ));
    assert_eq!(named(&all), ["store<4, bf16>", "main"]);
    // A prefix: the size given, the format from the result's annotation.
    let prefix = lower_ok(&format!(
        "{STORE}fn main() {{ (0.5, 4) -> fill -> xs: [f32; 4]; xs -> store<4> -> y: [f16; 4]; y[0] -> widen<f32> -> println; }}"
    ));
    assert_eq!(named(&prefix), ["store<4, f16>", "main"]);
    // Nothing given: the size from the argument, the format from the
    // annotation.
    let none = lower_ok(&format!(
        "{STORE}fn main() {{ (0.5, 4) -> fill -> xs: [f32; 4]; xs -> store -> y: [f8e4m3; 4]; y[0] -> widen<f32> -> println; }}"
    ));
    assert_eq!(named(&none), ["store<4, f8e4m3>", "main"]);
    // A size given explicitly is a size expression over the caller's sizes.
    let sized = lower_ok(&format!(
        "{STORE}fn outer<m>(x: [f32; m * 2]) -> [bf16; m * 2] {{ x -> store<m * 2, bf16> -> ret; }}
fn main() {{ (0.5, 6) -> fill -> xs: [f32; 6]; xs -> outer -> y; y[0] -> widen<f32> -> println; }}"
    ));
    assert_eq!(named(&sized), ["store<6, bf16>", "outer<3>", "main"]);
}

#[test]
fn a_format_flows_from_the_caller_into_the_callee() {
    // The caller's own format parameter as an explicit argument, and in a
    // binding annotation in the body.
    let src = format!(
        "{STORE}fn twice<n, c: float>(x: [f32; n]) -> [c; n] {{
    x -> store<n, c> -> once: [c; n];
    once -> ret;
}}
fn main() {{ (0.5, 4) -> fill -> xs: [f32; 4]; xs -> twice<4, bf16> -> y; y[1] -> widen<f32> -> println; }}"
    );
    let ir = lower_ok(&src);
    assert_eq!(named(&ir), ["store<4, bf16>", "twice<4, bf16>", "main"]);
}

#[test]
fn conversions_infer_from_the_context() {
    // A binding's annotation, the output at `-> ret`, a fixed function's
    // parameter the value flows into.
    let ir = lower_ok(
        "fn keep(v: bf16) -> f32 { v -> widen<f32> -> ret; }
fn half(x: f32) -> f16 { x -> to -> ret; }
fn main() {
    0.75 -> to -> a: bf16;
    a -> widen -> wide: f64;
    0.5 -> to -> keep -> k;
    2.0 -> half -> h;
    wide -> println;
    k -> println;
    h -> widen<f32> -> println;
}",
    );
    assert_eq!(count(&ir, Operation::Narrow), 3);
    // The narrow → f64, bf16 → f32 (twice: `keep`, its caller's tail) widenings.
    assert!(count(&ir, Operation::Widen) >= 3);
}

#[test]
fn an_explicit_source_must_agree() {
    let ok = lower_ok(
        "fn main() { 0.5 -> to<f32> -> x; x -> to<bf16, f32> -> b; b -> widen<f32, bf16> -> println; }",
    );
    assert_eq!(count(&ok, Operation::Narrow), 1);
    let m = message(
        "fn main() { 0.5 -> to<f32> -> x; x -> to<bf16, f16> -> b; }",
        "L1709",
    );
    assert!(m.contains("`source` = `f16`") && m.contains("`f32`"), "{m}");
}

#[test]
fn an_identity_conversion_is_erased() {
    let ir = lower_ok(
        "fn id(x: f32) -> f32 { x -> to<f32> -> ret; }
fn main() { 0.5 -> to<f32> -> x; x -> widen<f32> -> y; y -> id -> z; 3 -> widen<i32> -> i; z -> println; i -> println; }",
    );
    assert_eq!(count(&ir, Operation::Widen), 0);
    assert_eq!(count(&ir, Operation::Narrow), 0);
}

#[test]
fn literals_take_their_width_from_the_conversion() {
    // A float literal into `to<f32>` is an `f32` (the identity); into a
    // narrow target an `f32` rounded once; an integer literal keeps `i32`.
    let ir = lower_ok(
        "fn main() { 0.5 -> to<f32> -> a; 0.5 -> to<bf16> -> b; 3 -> to<f32> -> c; a -> println; b -> widen<f32> -> println; c -> println; }",
    );
    assert_eq!(count(&ir, Operation::Narrow), 1);
    let mut widens: Vec<String> = ir
        .morphisms()
        .filter(|(_, m)| m.op == Operation::Widen)
        .map(|(_, m)| mapal_ir::ty_label(&ir.object(m.source).expect("source").ty))
        .collect();
    widens.sort();
    assert_eq!(widens, ["bf16", "i32"]);
}

#[test]
fn widen_admits_the_embeddings() {
    let ir = lower_ok(
        "fn main() {
    0.5 -> to<bf16> -> b;
    0.5 -> to<f8e4m3> -> e;
    b -> widen<f32> -> b32;
    b -> widen<f64> -> b64;
    e -> widen<f32> -> e32;
    b32 -> widen<f64> -> d;
    3 -> widen<i64> -> l;
    3 -> widen<f64> -> f;
    b64 -> println; e32 -> println; d -> println; l -> println; f -> println;
}",
    );
    assert_eq!(count(&ir, Operation::Widen), 6);
}

#[test]
fn widen_of_an_i32_needs_the_facts() {
    // A compile-time constant within ±2²⁴.
    lower_ok("fn main() { 7 -> widen<f32> -> g; g -> println; }");
    lower_ok("fn main() { 16777216 -> widen<f32> -> g; g -> println; }");
    // A value range the bounds facts prove: an iota element, through `%`.
    lower_ok(
        "fn main() { 1000 -> iota -> map { t -> (t * 7 + 13) % 101 - 50 -> widen<f32> } -> v; v[3] -> println; }",
    );
    // `%` of a dividend past `i32` (wrapped) still lies within the divisor.
    lower_ok(
        "fn main() { 268435456 -> iota -> map { t -> (t * 11 + 57) % 103 - 51 -> widen<f32> } -> v; v[3] -> println; }",
    );
    // A value computable before the program runs, out of a fold the range
    // facts do not follow.
    lower_ok(
        "fn main() { 5 -> iota -> map { i -> i * i } -> sq; (0, sq) -> fold { s, x -> s + x } -> t; t -> widen<f32> -> println; }",
    );
    // Not exact: a constant past 2²⁴, an iota past 2²⁴, a function input.
    let m = message(
        "fn main() { 16777217 -> widen<f32> -> g; g -> println; }",
        "L1614",
    );
    assert!(m.contains("16777217") && m.contains("to<f32>"), "{m}");
    let m = message(
        "fn main() { 33554432 -> iota -> map { t -> t -> widen<f32> } -> v; v[3] -> println; }",
        "L1614",
    );
    assert!(m.contains("[0, 33554431]"), "{m}");
    let m = message(
        "fn f(x: i32) -> f32 { x -> widen<f32> -> ret; } fn main() { 3 -> f -> println; }",
        "L1614",
    );
    assert!(
        m.contains("bound it nowhere") && m.contains("to<f32>"),
        "{m}"
    );
    // `to<f32>` is the conversion that rounds: always admitted.
    lower_ok("fn f(x: i32) -> f32 { x -> to<f32> -> ret; } fn main() { 3 -> f -> println; }");
}

#[test]
fn widen_inside_a_generic_instance_is_proven_per_instance() {
    let src = "fn scale<n>(x: [i32; n]) -> [f32; n] { x -> map { v -> v % 1000 -> widen<f32> } -> ret; }
fn main() { 8 -> iota -> a; 3 -> iota -> b; a -> scale -> fa; b -> scale -> fb; fa[1] -> println; fb[1] -> println; }";
    assert_eq!(named(&lower_ok(src)), ["scale<3>", "scale<8>", "main"]);
}

#[test]
fn widen_rejects_a_conversion_that_rounds() {
    let m = message("fn main() { 0.5 -> widen<bf16> -> x; }", "L1614");
    assert!(m.contains("to<bf16>"), "{m}");
    let m = message(
        "fn main() { 0.5 -> to<f64> -> x; x -> widen<f32> -> y; }",
        "L1614",
    );
    assert!(m.contains("cannot widen `f64` to `f32`"), "{m}");
    // A pair no operation realizes yet says so.
    let m = message(
        "fn f(x: i64) -> f64 { x -> widen<f64> -> ret; } fn main() { 3 -> widen<i64> -> f -> println; }",
        "L1614",
    );
    assert!(m.contains("i64"), "{m}");
}

#[test]
fn to_rejects_the_open_pairs() {
    let m = message(
        "fn main() { 0.5 -> to<f64> -> x; x -> to<f32> -> y; }",
        "L1616",
    );
    assert!(m.contains("open"), "{m}");
    let m = message(
        "fn main() { 0.5 -> to<f32> -> x; x -> to<i32> -> y; }",
        "L1616",
    );
    assert!(m.contains("rounding-to-integer"), "{m}");
    let m = message("fn main() { 3 -> to<bf16> -> y; }", "L1616");
    assert!(m.contains("to<f32> -> to<bf16>"), "{m}");
    assert_rejects("fn main() { [1.0, 2.0] -> to<bf16> -> y; }", "L1616");
}

#[test]
fn an_undetermined_format_names_itself() {
    let m = message("fn main() { 0.5 -> to -> x; x -> println; }", "L1708");
    assert!(m.contains("`target` of `to`"), "{m}");
    // A format parameter only the body reads needs the stage to give it.
    let src = "fn fill_with<n, c: float>(x: [f32; n]) -> f32 { x[0] -> to<c> -> y; y -> widen<f32> -> ret; }
fn main() { (0.5, 4) -> fill -> xs: [f32; 4]; xs -> fill_with -> r; r -> println; }";
    let m = message(src, "L1708");
    assert!(m.contains("`c` of `fill_with`"), "{m}");
    lower_ok(&src.replace("-> fill_with ->", "-> fill_with<4, bf16> ->"));
}

#[test]
fn a_format_outside_its_kind() {
    // Given explicitly.
    let m = message(
        &format!(
            "{STORE}fn main() {{ (0.5, 4) -> fill -> xs: [f32; 4]; xs -> store<4, i32> -> y; }}"
        ),
        "L1710",
    );
    assert!(
        m.contains("`cache` of `store` is a `float`") && m.contains("`i32`"),
        "{m}"
    );
    // Inferred from the argument.
    let m = message(
        "fn first<n, t: narrow>(x: [t; n]) -> f32 { x[0] -> widen<f32> -> ret; }
fn main() { (0.5, 4) -> fill -> xs: [f32; 4]; xs -> first -> y; y -> println; }",
        "L1710",
    );
    assert!(
        m.contains("`t` of `first` is a `narrow`") && m.contains("`f32`"),
        "{m}"
    );
    // A format where a size is declared, a size where a format is.
    let m = message(
        &format!(
            "{STORE}fn main() {{ (0.5, 4) -> fill -> xs: [f32; 4]; xs -> store<bf16> -> y; }}"
        ),
        "L1710",
    );
    assert!(m.contains("`n` of `store` is a size"), "{m}");
    let m = message(
        &format!(
            "{STORE}fn main() {{ (0.5, 4) -> fill -> xs: [f32; 4]; xs -> store<4, 4> -> y; }}"
        ),
        "L1710",
    );
    assert!(m.contains("`cache` of `store` is a format"), "{m}");
    assert_rejects("fn main() { 0.5 -> to<8> -> y; }", "L1710");
}

#[test]
fn conflicting_formats() {
    // The explicit argument against the argument's format.
    let m = message(
        &format!(
            "{TOTAL}fn main() {{ (0.5, 4) -> fill -> xs: [f32; 4]; xs -> map {{ v -> v -> to<bf16> }} -> b; b -> total<4, f16> -> s; }}"
        ),
        "L1709",
    );
    assert!(m.contains("`t` = `f16`") && m.contains("`bf16`"), "{m}");
    // Two parameters of one format given two formats.
    let m = message(
        "fn mix<t: float>(a: t, b: t) -> f32 { a -> widen<f32> -> ret; }
fn main() { 0.5 -> to<bf16> -> a; 0.5 -> to<f16> -> b; (a, b) -> mix -> y; }",
        "L1709",
    );
    assert!(m.contains("conflicting formats"), "{m}");
}

#[test]
fn an_unknown_kind() {
    assert_rejects(
        "fn f<c: real>(x: f32) -> c { x -> to<c> -> ret; } fn main() { 3.0 -> f<f32> -> println; }",
        "L1711",
    );
    assert_rejects(
        "type Pair<e: decimal> { a: e, b: e } fn main() { 1 -> println; }",
        "L1711",
    );
}

#[test]
fn explicit_arguments_need_a_generic_callee() {
    // A fixed function, a value, a builtin without generic parameters, an
    // unknown name, too many arguments.
    assert_rejects(
        "fn f(x: f32) -> f32 { x -> ret; } fn main() { 0.5 -> f<4> -> x; }",
        "L1712",
    );
    assert_rejects("fn main() { 0.5 -> x; 0.25 -> x<4> -> y; }", "L1712");
    assert_rejects("fn main() { ([1.0], [2.0]) -> zip<1> -> z; }", "L1712");
    assert_rejects("fn main() { 0.5 -> nothing<4> -> y; }", "L1712");
    let m = message(
        &format!(
            "{STORE}fn main() {{ (0.5, 4) -> fill -> xs: [f32; 4]; xs -> store<4, bf16, f32> -> y; }}"
        ),
        "L1712",
    );
    assert!(m.contains("`store<n, cache: float>`"), "{m}");
    assert_rejects("fn main() { 0.5 -> to<bf16, f32, f32> -> y; }", "L1712");
}

#[test]
fn the_removed_names_name_their_replacement() {
    for (old, new) in [
        ("to_bf16", "to<bf16>"),
        ("to_f16", "to<f16>"),
        ("to_f8e4m3", "to<f8e4m3>"),
        ("to_f8e5m2", "to<f8e5m2>"),
        ("widen_f32", "widen<f32>"),
        ("widen_f64", "widen<f64>"),
        ("widen_i64", "widen<i64>"),
    ] {
        let m = message(&format!("fn main() {{ 0.5 -> {old} -> x; }}"), "L1713");
        assert!(m.contains(&format!("`{new}`")), "{old}: {m}");
    }
    let m = message("fn main() { 3 -> widen_f32 -> x; }", "L1713");
    assert!(m.contains("`to<f32>`"), "{m}");
    // The names stay reserved, as are `to` and `widen`.
    assert_rejects(
        "fn to_bf16(x: f32) -> f32 { x -> ret; } fn main() { 1 -> println; }",
        "L1009",
    );
    assert_rejects(
        "fn widen(x: f32) -> f32 { x -> ret; } fn main() { 1 -> println; }",
        "L1009",
    );
    assert_rejects("fn main() { 1 -> to; }", "L1708");
}

#[test]
fn a_format_parameter_is_no_value() {
    assert_rejects(
        "fn f<n, c: float>(x: [f32; n]) -> f32 { 1.0 -> c; x[0] } fn main() { [1.0] -> f<1, bf16> -> y; y -> println; }",
        "L1703",
    );
    // A name in a type position that is no format, type or declared format
    // parameter (no implicit parameters).
    let m = message(
        "fn f<n>(x: [cache; n]) -> f32 { 1.0 } fn main() { [1.0] -> f -> y; y -> println; }",
        "L1102",
    );
    assert!(m.contains("fn f<…, cache: float>"), "{m}");
    // Generic names are fresh across kinds.
    assert_rejects(
        "fn f<n, n: float>(x: [f32; n]) -> f32 { x[0] } fn main() { [1.0] -> f -> y; y -> println; }",
        "L1702",
    );
    assert_rejects(
        "fn f<n, bf16: float>(x: [f32; n]) -> f32 { x[0] } fn main() { [1.0] -> f -> y; y -> println; }",
        "L1009",
    );
}

#[test]
fn generic_types_take_format_parameters() {
    let src = "type Pair<e: float> { a: e, b: e }
type Matrix<rows, columns, element: float> { _: [element; rows * columns] }
fn first<r, c, e: float>(m: Matrix<r, c, e>) -> f32 { m[0] -> widen<f32> -> ret; }
fn sum<e: float>(p: Pair<e>) -> f32 { p.a -> widen<f32> -> a; p.b -> widen<f32> -> b; a + b }
fn main() {
    (0.5, 6) -> fill -> xs: [f32; 6];
    xs -> map { v -> v -> to<bf16> } -> w: Matrix<2, 3, bf16>;
    w -> first<2, 3> -> x;
    0.25 -> to<f16> -> h;
    Pair { a: h, b: h } -> p;
    p -> sum -> s;
    x -> println;
    s -> println;
}";
    let ir = lower_ok(src);
    assert_eq!(named(&ir), ["first<2, 3, bf16>", "sum<f16>", "main"]);
    assert!(
        ir.objects()
            .any(|(_, o)| matches!(&o.ty, Ty::Struct { name, .. } if name == "Pair<f16>")),
        "a generic struct's instance is named by its format"
    );
}

/// The IR's `Debug` text with source spans dropped and `renames` applied.
fn normalized(ir: &CategoryIr, renames: &[(&str, &str)]) -> String {
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

#[test]
fn an_instance_is_the_hand_specialized_function() {
    // D5: the format instance lowers to the IR of the function written for
    // that format (names and spans aside).
    let generic = lower_ok(&format!(
        "{STORE}{TOTAL}fn main() {{ (0.5, 4) -> fill -> xs: [f32; 4]; xs -> store<4, bf16> -> y; y -> total -> s; s -> println; }}"
    ));
    let fixed = lower_ok(
        "fn store_bf16(x: [f32; 4]) -> [bf16; 4] {
    x -> map { v -> v -> to<bf16> } -> ret;
}
fn total_bf16(xs: [bf16; 4]) -> f32 {
    4 -> iota -> idx;
    (0.0, idx) -> fold { acc, i -> xs[i] -> widen<f32> -> x; acc + x } -> ret;
}
fn main() { (0.5, 4) -> fill -> xs: [f32; 4]; xs -> store_bf16 -> y; y -> total_bf16 -> s; s -> println; }",
    );
    assert_eq!(
        normalized(
            &generic,
            &[
                ("store<4, bf16>", "store_bf16"),
                ("total<4, bf16>", "total_bf16")
            ]
        ),
        normalized(&fixed, &[])
    );
}
