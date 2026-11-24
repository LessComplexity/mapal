//! ADR-0048: top-level bindings, compile-time evaluation, `_` (a type's
//! default interpretation) and size-generic types. Acceptance programs run on
//! the oracle; every rejection code (L1801–L1809) and the size checks of a
//! generic type (L1702, L1706, L1802) have a test.

mod common;
use common::{assert_rejects, lower_err_codes, lower_ok};
use mapal_ir::{CategoryIr, ObjectKind, Ty, Value};

/// The oracle's stdout for `src`.
fn out(src: &str) -> String {
    let ir = lower_ok(src);
    let rr = mapal_interp::run(&ir, 10_000_000);
    assert!(
        matches!(rr.outcome, mapal_interp::Outcome::Done(_)),
        "{:?}",
        rr.outcome
    );
    rr.output
}

fn message(src: &str, code: &str) -> String {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "{:?}", po.diagnostics);
    let ds = mapal_lower::lower(src, &po.program).expect_err("expected a rejection");
    ds.iter()
        .find(|d| d.code.0 == code)
        .unwrap_or_else(|| panic!("no {code} in {ds:?}"))
        .message
        .clone()
}

/// The constants of `main`.
fn main_constants(ir: &CategoryIr) -> Vec<Value> {
    let main = ir.entry();
    ir.objects()
        .filter(|(o, x)| ir.owner(*o) == main && x.kind == ObjectKind::Constant)
        .filter_map(|(_, x)| x.value.clone())
        .collect()
}

#[test]
fn sizes_and_values_from_top_level_bindings_in_any_order() {
    // Used before it is declared, as a size and as a value; `head_size` is
    // computed from another binding (`/` is fine in a value).
    let src = "fn first(x: [f32; head_size]) -> f32 { x[0] -> ret; }
fn main() {
    head_size -> iota -> map { i -> i -> widen_f32 } -> h;
    h -> first -> println;
    hidden_size -> println;
    head_size * 3 -> println;
}
hidden_size / heads -> head_size;
64 -> hidden_size;
4 -> heads;
";
    assert_eq!(out(src), "0\n64\n48\n");
    // Each read is a literal: no call, no cone left in `main`.
    let ir = lower_ok(src);
    assert!(main_constants(&ir).contains(&Value::I32(64)));
}

#[test]
fn an_unannotated_number_takes_its_width_from_context() {
    let src = "3000000000 -> big;
7 -> small;
0.1 -> tenth;
fn main() {
    small -> widen_i64 -> a;
    a + big -> println;
    small + 1 -> println;
    1.5 -> f: f32;
    f * tenth -> println;
    2.5 * tenth -> println;
}
";
    // `tenth` is the literal `0.1` at each use: f32 there, f64 here.
    assert_eq!(out(src), "3000000007\n8\n0.15\n0.25\n");
}

#[test]
fn an_annotated_binding_is_a_constant_of_its_type() {
    let src = "0.5 -> half: f32;
12 -> n: i64;
fn main() {
    half -> println;
    n -> println;
}
";
    assert_eq!(out(src), "0.5\n12\n");
    let ir = lower_ok(src);
    assert!(main_constants(&ir).contains(&Value::F32(0.5)));
    assert!(main_constants(&ir).contains(&Value::I64(12)));
}

#[test]
fn a_computed_scalar_is_evaluated_before_the_program_runs() {
    // A fold over a law, a loop-free call: computed by the oracle at compile
    // time and a literal from then on — usable as a size.
    let src = "fn triangle(x: [i32; 4]) -> i32 {
    (0, x) -> fold { s, i -> s + i + 1 } -> ret;
}
4 -> iota -> triangle -> cells;
fn main() {
    cells -> iota -> map { i -> i * 2 } -> doubled;
    doubled -> last;
    last[cells - 1] -> println;
    cells -> println;
}
";
    let ir = lower_ok(src);
    assert!(main_constants(&ir).contains(&Value::I32(10)));
    assert_eq!(out(src), "18\n10\n");
}

#[test]
fn an_array_binding_is_its_chain_where_a_function_reads_it() {
    // A table read in `main` and captured into a map body of another function.
    let src = "8 -> iota -> map { i -> i * i } -> squares;
fn shifted(x: [i32; 8]) -> [i32; 8] { x -> map { v -> v + squares[3] } -> ret; }
fn main() {
    squares[7] -> println;
    squares -> shifted -> s;
    s[2] -> println;
}
";
    assert_eq!(out(src), "49\n13\n");
}

#[test]
fn a_struct_binding_and_a_tuple_binding() {
    let src = "type Pair { a: i32, b: f32 }
Pair { a: 3, b: 0.25 } -> pair;
(1, 2) -> both;
fn main() {
    pair.a -> println;
    pair.b -> println;
    both.1 -> println;
}
";
    assert_eq!(out(src), "3\n0.25\n2\n");
}

#[test]
fn several_names_in_one_chain() {
    let src = "16 -> lanes -> * 4 -> bytes;
fn main() { (lanes, bytes) -> t; t.1 -> println; }
";
    assert_eq!(out(src), "64\n");
}

#[test]
fn a_size_that_traps_or_is_no_integer_is_l1808() {
    let src = "1.5 -> width;
fn main() { [1.0, 2.0] -> x: [f32; width]; x[0] -> println; }
";
    assert_rejects(src, "L1808");
    let m = message(src, "L1808");
    assert!(m.contains("`width` is a top-level value"), "{m}");
    // A binding whose evaluation traps is realized at run time (its trap is
    // the program's), and cannot be a size.
    assert_rejects(
        "[1, 2] -> pair;
pair[5] -> bad;
fn main() { [1.0] -> x: [f32; bad]; x[0] -> println; }",
        "L1808",
    );
    assert_rejects(
        "0 - 4 -> negative;
fn main() { [1.0] -> x: [f32; negative]; x[0] -> println; }",
        "L1808",
    );
}

#[test]
fn rejections_l1801_to_l1807() {
    assert_rejects("4 -> mut n;\nfn main() { n -> println; }", "L1801");
    assert_rejects(
        "4 -> n;\nfn f(n: i32) -> i32 { n -> ret; }\nfn main() { 1 -> f -> println; }",
        "L1802",
    );
    assert_rejects("4 -> n;\nfn main() { 5 -> n; n -> println; }", "L1802");
    assert_rejects(
        "4 -> n;\nfn main() { 3 -> iota -> map { n -> n } -> x; x[0] -> println; }",
        "L1802",
    );
    assert_rejects(
        "4 -> n;\nfn f<n>(x: [f32; n]) -> f32 { x[0] -> ret; }\nfn main() { [1.0] -> f -> println; }",
        "L1802",
    );
    assert_rejects("4 -> ret;\nfn main() { }", "L1803");
    assert_rejects("4 -> n;\n5 -> n;\nfn main() { n -> println; }", "L1804");
    assert_rejects("fn n() { }\n4 -> n: i32;\nfn main() { }", "L1804");
    assert_rejects("4 -> println;\nfn main() { }", "L1805");
    assert_rejects(
        "fn noisy(x: i32) -> i32 { x -> println; x -> ret; }\n4 -> noisy -> n;\nfn main() { n -> println; }",
        "L1806",
    );
    assert_rejects(
        "a + 1 -> b;\nb + 1 -> a;\nfn main() { a -> println; }",
        "L1807",
    );
    // A cycle through a function's signature.
    let m = message(
        "fn f(x: [f32; m]) -> i32 { 1 -> ret; }\n[1.0] -> f -> m;\nfn main() { m -> println; }",
        "L1807",
    );
    assert!(m.contains("cycle"), "{m}");
}

#[test]
fn a_types_only_field_underscore_is_a_two_way_alias() {
    let src = "4 -> hidden_size;
type Hidden { _: [f32; hidden_size] }
fn total(x: Hidden) -> f32 { (0.0, x) -> fold { s, v -> s + v } -> ret; }
fn twice(x: [f32; 4]) -> Hidden { x -> map { v -> v * 2.0 } -> ret; }
fn main() {
    [1.0, 2.0, 3.0, 4.0] -> x: Hidden;
    x -> twice -> y;
    y -> total -> println;
    y[3] -> println;
}
";
    assert_eq!(out(src), "20\n8\n");
    // Zero cost: the alias is the array type itself.
    let ir = lower_ok(src);
    assert!(
        ir.objects()
            .all(|(_, o)| !matches!(&o.ty, Ty::Struct { name, .. } if name == "Hidden"))
    );
}

#[test]
fn a_types_default_field_reads_one_way_where_its_own_type_does_not_fit() {
    let src = "type Scaled { _: [f32; 3], scale: f32 }
type Outer { _: Scaled, tag: i32 }
fn sum(x: [f32; 3]) -> f32 { (0.0, x) -> fold { s, v -> s + v } -> ret; }
fn keep(x: Scaled) -> Scaled { x -> ret; }
fn main() {
    Scaled { _: [1.0, 2.0, 4.0], scale: 0.5 } -> s;
    s -> sum -> println;
    s[2] -> println;
    s -> map { v -> v * s.scale } -> m;
    m[1] -> println;
    (0.0, s) -> fold { a, v -> a + v } -> println;
    s -> keep -> k;
    k.scale -> println;
    Outer { _: s, tag: 7 } -> o;
    o -> sum -> println;
    o.scale -> println;
    o[0] -> println;
}
";
    assert_eq!(out(src), "7\n4\n1\n7\n0.5\n7\n0.5\n1\n");
}

#[test]
fn a_size_generic_type_instantiates_at_its_sizes() {
    let src = "2 -> rows_n;
type Matrix<rows, columns> { _: [f32; rows * columns] }
type Layer<n> { weight: Matrix<n, n>, bias: [f32; n] }
fn project<rows, columns>(weight: Matrix<rows, columns>, x: [f32; columns]) -> [f32; rows] {
    columns -> iota -> cs;
    rows -> iota -> map {
        r -> (0.0, cs) -> fold {
            s, c -> s + weight[r * columns + c] * x[c]
        }
    } -> ret;
}
fn apply<n>(layer: Layer<n>, x: [f32; n]) -> [f32; n] {
    (layer.weight, x) -> project -> y;
    n -> iota -> map { i -> y[i] + layer.bias[i] } -> ret;
}
fn main() {
    [1.0, 2.0, 3.0, 4.0] -> w: Matrix<rows_n, 2>;
    Layer { weight: w, bias: [10.0, 20.0] } -> layer: Layer<2>;
    [1.0, 1.0] -> x;
    (layer, x) -> apply -> y;
    y[0] -> println;
    y[1] -> println;
}
";
    assert_eq!(out(src), "13\n27\n");
    let ir = lower_ok(src);
    assert!(
        ir.objects()
            .any(|(_, o)| matches!(&o.ty, Ty::Struct { name, .. } if name == "Layer<2>"))
    );
}

#[test]
fn rejections_of_generic_types_l1809_l1702_l1706() {
    assert_rejects(
        "type M<r, c> { _: [f32; r * c] }\nfn main() { [1.0] -> x: M<1>; x[0] -> println; }",
        "L1809",
    );
    assert_rejects(
        "type M<r, c> { _: [f32; r * c] }\nfn main() { [1.0] -> x: M; x[0] -> println; }",
        "L1809",
    );
    assert_rejects(
        "type P { a: f32 }\nfn main() { P { a: 1.0 } -> x: P<2>; x.a -> println; }",
        "L1809",
    );
    assert_rejects("type M<r, r> { _: [f32; r] }\nfn main() { }", "L1702");
    assert_rejects("type M<r, c> { _: [f32; r] }\nfn main() { }", "L1706");
    assert_rejects("4 -> r;\ntype M<r> { _: [f32; r] }\nfn main() { }", "L1802");
}

#[test]
fn a_program_without_top_level_bindings_is_unchanged() {
    let codes = lower_err_codes("fn main() { x -> println; }");
    assert_eq!(codes, vec!["L1101".to_string()]);
}
