//! ADR-0047: size-generic functions. Acceptance (the user's `project`), the
//! inference cases (bare, product, sum, difference, a repeated unknown, the
//! result's annotation, nested generic calls, one function at several sizes,
//! sizes past 2³¹ cells), the size value's integer type (D4), every rejection
//! code (L1701–L1707, L1002, L1009, L1201, L1008), and the monomorphized IR
//! against the hand-specialized program's (D5: names and spans aside).

mod common;
use common::{assert_rejects, lower_err_codes, lower_ok};
use mapal_ir::{CategoryIr, FuncKind, ObjectKind, Ty, Value};

/// The named functions of `ir`, in declaration order.
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

const PROJECT: &str = "fn project<rows, columns>(weight: [bf16; rows * columns], x: [f32; columns]) -> [f32; rows] {
    columns -> iota -> column_indices;
    rows -> iota -> map { row -> (0.0, column_indices) -> fold { sum, column -> weight[row * columns + column] -> widen_f32 -> w; sum + w * x[column] } } -> ret;
}
";

#[test]
fn the_users_project_lowers_once_per_shape() {
    let src = format!(
        "type Layer {{ query: [bf16; 32], key: [bf16; 16] }}
{PROJECT}
fn main() {{
    32 -> iota -> map {{ i -> i -> widen_f32 -> to_bf16 }} -> q;
    16 -> iota -> map {{ i -> i -> widen_f32 -> to_bf16 }} -> k;
    Layer {{ query: q, key: k }} -> layer;
    8 -> iota -> map {{ i -> i -> widen_f32 }} -> normed;
    (layer.query, normed) -> project -> query;
    (layer.key, normed) -> project -> key;
    (layer.query, normed) -> project -> again;
    query[3] -> println;
    key[1] -> println;
    again[0] -> println;
}}"
    );
    let ir = lower_ok(&src);
    // Two shapes (4 × 8, 2 × 8): two instances, in size-tuple order at the
    // generic's position; the generic itself lowers to nothing.
    assert_eq!(named(&ir), ["project<2, 8>", "project<4, 8>", "main"]);
    let bodies: Vec<String> = ir
        .funcs()
        .filter(|(_, f)| f.kind != FuncKind::Named)
        .map(|(_, f)| f.name.clone())
        .collect();
    assert!(
        bodies.contains(&"project<4, 8>::map@0".to_string()),
        "{bodies:?}"
    );
}

#[test]
fn an_uncalled_generic_lowers_to_nothing() {
    let ir = lower_ok(&format!("{PROJECT}\nfn main() {{ 1 -> println; }}"));
    assert_eq!(named(&ir), ["main"]);
}

/// `main` calling `f` with each argument; the instances it gets.
fn instances(helper: &str, calls: &[&str]) -> Vec<String> {
    let mut body = String::new();
    for (i, c) in calls.iter().enumerate() {
        body.push_str(&format!("    {c} -> r{i};\n    r{i} -> println;\n"));
    }
    let src = format!("{helper}\nfn main() {{\n{body}}}");
    named(&lower_ok(&src))
        .into_iter()
        .filter(|n| n != "main")
        .collect()
}

#[test]
fn inference_bare_product_sum_difference() {
    // Bare: one name binds directly; two parameters share it.
    let dot = "fn dot<n>(a: [f32; n], b: [f32; n]) -> f32 {
    (a, b) -> zip -> map { p -> p.0 * p.1 } -> m;
    (0.0, m) -> fold { s, v -> s + v } -> ret;
}";
    assert_eq!(
        instances(
            dot,
            &["([1.0, 2.0], [3.0, 4.0]) -> dot", "([1.0], [3.0]) -> dot"]
        ),
        ["dot<1>", "dot<2>"]
    );
    // Product: `columns` binds from `x`, then `rows` = 12 / 4.
    let rows = "fn rows<r, c>(m: [f32; r * c], x: [f32; c]) -> f32 { m[r * c - 1] * x[c - 1] }";
    assert_eq!(
        instances(
            rows,
            &["(0.5, 12) -> fill -> m; (2.0, 4) -> fill -> x; (m, x) -> rows"]
        ),
        ["rows<3, 4>"]
    );
    // Sum and difference, each with its one unknown.
    let pad = "fn pad<n>(x: [f32; n + 2]) -> f32 { x[n + 1] }";
    assert_eq!(instances(pad, &["(1.0, 7) -> fill -> pad"]), ["pad<5>"]);
    let cut = "fn cut<n>(x: [f32; 10 - n]) -> f32 { x[0] }";
    assert_eq!(instances(cut, &["(1.0, 7) -> fill -> cut"]), ["cut<3>"]);
    // A nested parenthesized expression, inverted along its path.
    let nest = "fn nest<n>(x: [f32; (n - 1) * 3 + 2]) -> f32 { x[0] }";
    assert_eq!(instances(nest, &["(1.0, 14) -> fill -> nest"]), ["nest<5>"]);
}

#[test]
fn inference_repeated_unknown_by_bisection() {
    // A square matrix: `n * n` = 16 → n = 4 (D3.3, non-decreasing in n).
    let sq = "fn trace<n>(m: [f32; n * n]) -> f32 { n -> iota -> d; (0.0, d) -> fold { s, i -> s + m[i * n + i] } -> ret; }";
    assert_eq!(
        instances(
            sq,
            &["(1.0, 16) -> fill -> trace", "(1.0, 9) -> fill -> trace"]
        ),
        ["trace<3>", "trace<4>"]
    );
    let tri = "fn tri<n>(x: [f32; n * (n + 1)]) -> f32 { x[0] }";
    assert_eq!(instances(tri, &["(1.0, 30) -> fill -> tri"]), ["tri<5>"]);
    // No whole solution: L1705.
    assert_rejects(
        &format!("{sq}\nfn main() {{ (1.0, 15) -> fill -> trace -> t; t -> println; }}"),
        "L1705",
    );
}

#[test]
fn inference_from_the_results_annotation_or_ret() {
    let zeros = "fn zeros<n>(x: f32) -> [f32; n] { (x, n) -> fill -> ret; }";
    let src = format!("{zeros}\nfn main() {{ 1.5 -> zeros -> z: [f32; 8]; z[7] -> println; }}");
    assert_eq!(named(&lower_ok(&src)), ["zeros<8>", "main"]);
    // `-> ret` of a function whose output is declared.
    let src = format!(
        "{zeros}\nfn eight(x: f32) -> [f32; 8] {{ x -> zeros -> ret; }}
fn main() {{ 2.0 -> eight -> z; z[0] -> println; }}"
    );
    assert_eq!(named(&lower_ok(&src)), ["zeros<8>", "eight", "main"]);
    // Without either: L1706 at the call.
    let src = format!("{zeros}\nfn main() {{ 1.5 -> zeros -> z; z[0] -> println; }}");
    assert!(message(&src, "L1706").contains("cannot infer size `n` of `zeros`"));
}

#[test]
fn nested_generic_calls_flow_sizes_through() {
    let src = "fn dot<n>(a: [f32; n], b: [f32; n]) -> f32 {
    (a, b) -> zip -> map { p -> p.0 * p.1 } -> m;
    (0.0, m) -> fold { s, v -> s + v } -> ret;
}
fn first_row<r, c>(m: [f32; r * c], v: [f32; c]) -> f32 {
    c -> iota -> map { j -> m[j] } -> row;
    (row, v) -> dot -> ret;
}
fn norm<n>(x: [f32; n]) -> f32 { (x, x) -> dot -> ret; }
fn main() {
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] -> m;
    (m, [1.0, 1.0, 1.0]) -> first_row -> a;
    (m, [1.0, 2.0]) -> first_row -> b;
    m -> norm -> c;
    a -> println;
    b -> println;
    c -> println;
}";
    let ir = lower_ok(src);
    assert_eq!(
        named(&ir),
        [
            "dot<2>",
            "dot<3>",
            "dot<6>",
            "first_row<2, 3>",
            "first_row<3, 2>",
            "norm<6>",
            "main"
        ]
    );
    // Values: `crates/backends/llvm/tests/size_generics.rs`.
    let _ = ir;
}

/// Every constant's (type, value) in the function named `name`.
fn constants(ir: &CategoryIr, name: &str) -> Vec<(Ty, Value)> {
    let (f, _) = ir.funcs().find(|(_, f)| f.name == name).expect("fn");
    ir.objects()
        .filter(|(id, o)| ir.owner(*id) == f && o.kind == ObjectKind::Constant)
        .filter_map(|(_, o)| Some((o.ty.clone(), o.value.clone()?)))
        .collect()
}

#[test]
fn a_size_value_is_an_integer_literal() {
    // Unpinned: `Ty::index_ty(n)` — i32 below 2³¹, i64 past it (D4); the
    // arrays past 2³¹ cells are only lowered here, never built.
    let src = "fn show<n>(x: [i32; n]) { n -> println; }
fn main() {
    (0, 5) -> fill -> small;
    small -> show;
    (0, 3000000000) -> fill -> big;
    big -> show;
}";
    let ir = lower_ok(src);
    assert_eq!(named(&ir), ["show<5>", "show<3000000000>", "main"]);
    assert_eq!(constants(&ir, "show<5>"), [(Ty::i32(), Value::I32(5))]);
    assert_eq!(
        constants(&ir, "show<3000000000>"),
        [(Ty::i64(), Value::I64(3_000_000_000))]
    );
    // Pinned by context like any literal: an i64 operand, a u8 element.
    let src = "fn f<n>(x: [u8; n], k: i64) -> i64 { k * n }
fn g<n>(x: [u8; n]) -> u8 { x[0] + n }
fn main() { (7, 3) -> fill -> b: [u8; 3]; (b, 10) -> f -> y; y -> println; b -> g -> z; z -> println; }";
    let ir = lower_ok(src);
    assert_eq!(constants(&ir, "f<3>"), [(Ty::i64(), Value::I64(3))]);
    assert!(constants(&ir, "g<3>").contains(&(Ty::u8(), Value::U8(3))));
    // A size expression in value position is one literal (`r * c -> iota`,
    // `-n`), never an arithmetic morphism.
    let src = "fn flat<r, c>(m: [f32; r * c], v: [f32; c]) -> [f32; r * c] {
    r * c -> iota -> map { i -> m[r * c - 1 - i] * v[i % c] } -> ret;
}
fn neg<n>(x: [f32; n]) -> i32 { -n }
fn main() {
    ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 10.0]) -> flat -> y;
    y[0] -> println;
    y[5] -> println;
    y -> neg -> k;
    k -> println;
}";
    let ir = lower_ok(src);
    assert_eq!(constants(&ir, "neg<6>"), [(Ty::i32(), Value::I32(-6))]);
    // `r * c` (twice) and `r * c - 1` are constants of `flat<3, 2>`; the
    // arithmetic left is `- i` and `i % c` / the index products.
    let flat = constants(&ir, "flat<3, 2>");
    assert!(flat.contains(&(Ty::i32(), Value::I32(6))), "{flat:?}");
}

#[test]
fn rejections_l1701_unknown_size() {
    // In a generic signature, a fixed-size signature, a body annotation, a type.
    assert_rejects(
        "fn f<n>(x: [f32; m]) -> f32 { x[0] }\nfn main() { 1 -> println; }",
        "L1701",
    );
    assert_rejects(
        "fn f(x: [f32; n]) -> f32 { x[0] }\nfn main() { 1 -> println; }",
        "L1701",
    );
    assert_rejects(
        "fn f<n>(x: [f32; n]) -> f32 { x -> y: [f32; m]; y[0] }
fn main() { [1.0] -> f -> v; v -> println; }",
        "L1701",
    );
    assert_rejects(
        "type T { a: [f32; n] }\nfn main() { 1 -> println; }",
        "L1701",
    );
}

#[test]
fn rejections_l1702_duplicate_or_clashing_size() {
    for sig in [
        "fn f<n, n>(x: [f32; n]) -> f32 { x[0] }",
        "fn f<x>(x: [f32; x]) -> f32 { x[0] }",
        "fn f<main>(x: [f32; main]) -> f32 { x[0] }",
    ] {
        assert_rejects(&format!("{sig}\nfn main() {{ 1 -> println; }}"), "L1702");
    }
    // A reserved name is L1009.
    assert_rejects(
        "fn f<iota>(x: [f32; iota]) -> f32 { x[0] }\nfn main() { 1 -> println; }",
        "L1009",
    );
}

#[test]
fn rejections_l1703_rebinding_a_size() {
    let call = "fn main() { [1.0, 2.0] -> f -> y; y -> println; }";
    for body in [
        "3 -> n; n",
        "n <- 3; n",
        "x -> n: [f32; 2]; 0",
        "x -> map { n -> n } -> y; 0",
    ] {
        let src = format!("fn f<n>(x: [f32; n]) -> i32 {{ {body} }}\n{call}");
        let m = message(&src, "L1703");
        assert!(
            m.contains("`n` is a size parameter") && m.contains("(in `f<2>`)"),
            "{m}"
        );
    }
}

#[test]
fn rejections_l1704_conflicting_sizes() {
    let add = "fn add<n>(a: [f32; n], b: [f32; n]) -> [f32; n] { (a, b) -> zip -> map { p -> p.0 + p.1 } -> ret; }";
    let m = message(
        &format!(
            "{add}\nfn main() {{ ([1.0, 2.0], [1.0, 2.0, 3.0]) -> add -> c; c[0] -> println; }}"
        ),
        "L1704",
    );
    assert!(
        m.contains("`n` = 2 (from `a: [f32; n]`), but `b: [f32; n]` has 3"),
        "{m}"
    );
    // An overdetermined compound dimension.
    let rows = "fn rows<r, c>(m: [f32; r * c], v: [f32; c], o: [f32; r]) -> f32 { v[0] }";
    let m = message(
        &format!(
            "{rows}\nfn main() {{ (0.0, 12) -> fill -> m; (0.0, 3) -> fill -> v; (0.0, 5) -> fill -> o; (m, v, o) -> rows -> y; y -> println; }}"
        ),
        "L1704",
    );
    assert!(m.contains("so `r * c` = 15") && m.contains("has 12"), "{m}");
    // A zero factor: no size makes `0 * n` positive.
    assert_rejects(
        "fn z<n>(x: [f32; 0 * n + 1], y: [f32; 0 * n]) -> f32 { x[0] }
fn main() { ([1.0], [1.0]) -> z -> v; v -> println; }",
        "L1704",
    );
}

#[test]
fn rejections_l1705_not_a_whole_number() {
    let m = message(
        "fn rows<r, c>(m: [f32; r * c], v: [f32; c]) -> f32 { v[0] }
fn main() { (0.0, 10) -> fill -> m; (0.0, 3) -> fill -> v; (m, v) -> rows -> y; y -> println; }",
        "L1705",
    );
    assert!(m.contains("`r` = 10 / 3"), "{m}");
}

#[test]
fn rejections_l1706_undetermined_size() {
    // At the declaration: in no parameter or output type.
    assert_rejects(
        "fn f<n, k>(x: [f32; n]) -> f32 { x[0] }\nfn main() { 1 -> println; }",
        "L1706",
    );
    // At a call: only a product of two unknowns.
    assert_rejects(
        "fn f<r, c>(m: [f32; r * c]) -> f32 { m[0] }
fn main() { (0.0, 12) -> fill -> f -> y; y -> println; }",
        "L1706",
    );
}

#[test]
fn rejections_l1707_size_out_of_range() {
    // The output's dimension is not positive for this instance.
    let m = message(
        "fn shrink<n>(x: [f32; n]) -> [f32; n - 4] { n - 4 -> iota -> map { i -> x[i] } -> ret; }
fn main() { [1.0, 2.0, 3.0] -> shrink -> y; y[0] -> println; }",
        "L1707",
    );
    assert!(m.contains("`n - 4` is -1"), "{m}");
    // A solved size below 1.
    assert_rejects(
        "fn f<n>(x: [f32; n + 8]) -> f32 { x[0] }\nfn main() { [1.0, 2.0] -> f -> y; y -> println; }",
        "L1707",
    );
    // A literal dimension past 2⁶³ − 1.
    assert_rejects(
        "fn main() { 1 -> println; }\nfn f(x: [f32; 4294967296 * 4294967296]) -> f32 { x[0] }",
        "L1707",
    );
}

#[test]
fn rejections_shape_main_and_recursion() {
    // `main` takes no size parameters.
    assert_rejects("fn main<n>() { 1 -> println; }", "L1002");
    // An argument whose shape cannot match.
    assert_rejects(
        "fn f<n>(x: [f32; n]) -> f32 { x[0] }\nfn main() { 1.0 -> f -> y; y -> println; }",
        "L1201",
    );
    // The call graph stays acyclic on surface names.
    assert_rejects(
        "fn f<n>(x: [f32; n]) -> f32 { x -> f -> y; y }\nfn main() { [1.0] -> f -> y; y -> println; }",
        "L1008",
    );
}

#[test]
fn an_error_in_an_instance_is_reported_once_and_named() {
    let src = "fn f<n>(x: [f32; n]) -> f32 { 1 -> n; x[0] }
fn main() { [1.0] -> f -> a; [1.0, 2.0] -> f -> b; a -> println; b -> println; }";
    let ds = lower_diags(src);
    let l1703: Vec<_> = ds.iter().filter(|d| d.code.0 == "L1703").collect();
    assert_eq!(l1703.len(), 1, "{ds:?}");
    assert!(l1703[0].message.contains("(in `f<1>`)"));
}

/// The IR's `Debug` text with source spans dropped and instance names mapped
/// to the hand-written program's.
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
fn monomorphized_ir_is_the_hand_specialized_programs() {
    // Two instances of one generic, a size expression in value position, a
    // size in a guard, a nested generic call.
    let generic = "fn project<rows, columns>(weight: [bf16; rows * columns], x: [f32; columns]) -> [f32; rows] {
    columns -> iota -> column_indices;
    rows -> iota -> map { row -> (0.0, column_indices) -> fold { sum, column -> weight[row * columns + column] -> widen_f32 -> w; sum + w * x[column] } } -> ret;
}
fn clip<n>(y: [f32; n]) -> [f32; n] {
    n * 2 - 1 -> iota -> top;
    y -> map { v -> (v > 1.0) -> { -true-> 1.0; -false-> v } } -> ret;
}
fn both<n>(y: [f32; n]) -> f32 { y -> clip -> c; (0.0, c) -> fold { s, v -> s + v } -> ret; }
fn main() {
    12 -> iota -> map { i -> i -> widen_f32 -> * 0.25 -> to_bf16 } -> w;
    [1.0, 2.0, 3.0, 4.0] -> x4;
    [1.0, 2.0, 3.0] -> x3;
    (w, x4) -> project -> y;
    (w, x3) -> project -> z;
    y -> both -> a;
    z -> both -> b;
    a -> println;
    b -> println;
}";
    let hand = "fn project_3_4(weight: [bf16; 12], x: [f32; 4]) -> [f32; 3] {
    4 -> iota -> column_indices;
    3 -> iota -> map { row -> (0.0, column_indices) -> fold { sum, column -> weight[row * 4 + column] -> widen_f32 -> w; sum + w * x[column] } } -> ret;
}
fn project_4_3(weight: [bf16; 12], x: [f32; 3]) -> [f32; 4] {
    3 -> iota -> column_indices;
    4 -> iota -> map { row -> (0.0, column_indices) -> fold { sum, column -> weight[row * 3 + column] -> widen_f32 -> w; sum + w * x[column] } } -> ret;
}
fn clip_3(y: [f32; 3]) -> [f32; 3] {
    5 -> iota -> top;
    y -> map { v -> (v > 1.0) -> { -true-> 1.0; -false-> v } } -> ret;
}
fn clip_4(y: [f32; 4]) -> [f32; 4] {
    7 -> iota -> top;
    y -> map { v -> (v > 1.0) -> { -true-> 1.0; -false-> v } } -> ret;
}
fn both_3(y: [f32; 3]) -> f32 { y -> clip_3 -> c; (0.0, c) -> fold { s, v -> s + v } -> ret; }
fn both_4(y: [f32; 4]) -> f32 { y -> clip_4 -> c; (0.0, c) -> fold { s, v -> s + v } -> ret; }
fn main() {
    12 -> iota -> map { i -> i -> widen_f32 -> * 0.25 -> to_bf16 } -> w;
    [1.0, 2.0, 3.0, 4.0] -> x4;
    [1.0, 2.0, 3.0] -> x3;
    (w, x4) -> project_3_4 -> y;
    (w, x3) -> project_4_3 -> z;
    y -> both_3 -> a;
    z -> both_4 -> b;
    a -> println;
    b -> println;
}";
    let g = lower_ok(generic);
    let h = lower_ok(hand);
    let renames = [
        ("project<3, 4>", "project_3_4"),
        ("project<4, 3>", "project_4_3"),
        ("clip<3>", "clip_3"),
        ("clip<4>", "clip_4"),
        ("both<3>", "both_3"),
        ("both<4>", "both_4"),
    ];
    assert_eq!(normalized(&g, &renames), normalized(&h, &[]));
}

#[test]
fn fixed_size_programs_are_unchanged() {
    // No generic function: one round, the original pipeline, no instances;
    // literal arithmetic in a length is a size expression.
    let codes = lower_err_codes("fn main() { [1.0] -> f; }\nfn f(x: [f32; 2 - 2]) { }");
    assert_eq!(codes, ["L1707"]);
    let ir = lower_ok(
        "fn f(x: [f32; 2 * 2]) -> f32 { x[3] }\nfn main() { [1.0, 2.0, 3.0, 4.0] -> f -> y; y -> println; }",
    );
    assert_eq!(named(&ir), ["f", "main"]);
}

#[test]
fn a_call_the_typing_walk_cannot_type_takes_emissions_binding() {
    // `y` comes out of a `seq` block, which the typing walk leaves untyped:
    // it cannot infer `twice`'s size, so the map after the call would type
    // `v` as the fallback `i32` (an L1203 against `0.5`). Emission binds the
    // site from the exact type; the next round types it (D5). The same in
    // an instance body (`outer` → `twice`), a round per level.
    let generic = "fn twice<n>(x: [f32; n]) -> [f32; n] { x -> map { v -> v * 2.0 } -> ret; }
fn outer<n>(x: [f32; n]) -> f32 {
    x -> seq { -> map { v -> v + 1.0 } } -> y;
    y -> twice -> z;
    z -> map { v -> v + 0.5 } -> w;
    (0.0, w) -> fold { s, v -> s + v } -> ret;
}
fn main() {
    [1.0, 2.0, 3.0] -> a: [f32; 3];
    a -> seq { -> map { v -> v + 1.0 } } -> y;
    y -> twice -> z;
    z -> map { v -> v + 0.5 } -> w;
    w[0] -> println;
    a -> outer -> o;
    o -> println;
}";
    let hand = "fn twice_3(x: [f32; 3]) -> [f32; 3] { x -> map { v -> v * 2.0 } -> ret; }
fn outer_3(x: [f32; 3]) -> f32 {
    x -> seq { -> map { v -> v + 1.0 } } -> y;
    y -> twice_3 -> z;
    z -> map { v -> v + 0.5 } -> w;
    (0.0, w) -> fold { s, v -> s + v } -> ret;
}
fn main() {
    [1.0, 2.0, 3.0] -> a: [f32; 3];
    a -> seq { -> map { v -> v + 1.0 } } -> y;
    y -> twice_3 -> z;
    z -> map { v -> v + 0.5 } -> w;
    w[0] -> println;
    a -> outer_3 -> o;
    o -> println;
}";
    let (g, h) = (lower_ok(generic), lower_ok(hand));
    let renames = [("twice<3>", "twice_3"), ("outer<3>", "outer_3")];
    assert_eq!(normalized(&g, &renames), normalized(&h, &[]));
    // A real error downstream of such a call is still reported.
    let bad = "fn twice<n>(x: [f32; n]) -> [f32; n] { x -> map { v -> v * 2.0 } -> ret; }
fn main() {
    [1.0, 2.0, 3.0] -> a: [f32; 3];
    a -> seq { -> map { v -> v + 1.0 } } -> y;
    y -> twice -> z;
    z -> map { v -> v + true } -> w;
    w[0] -> println;
}";
    assert!(!lower_err_codes(bad).is_empty());
}

#[test]
fn a_generic_call_inside_a_map_body() {
    // The call site belongs to the instance whose body holds it (the map
    // body is part of `rowsums<r, c>`), so each instance binds its own.
    let src = "fn total<n>(x: [f32; n]) -> f32 { (0.0, x) -> fold { s, v -> s + v } -> ret; }
fn rowsums<r, c>(m: [f32; r * c], v: [f32; c]) -> [f32; r] {
    r -> iota -> map { i -> c -> iota -> map { j -> m[i * c + j] * v[j] } -> row; row -> total } -> ret;
}
fn main() {
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] -> m;
    (m, [1.0, 10.0]) -> rowsums -> y;
    (m, [1.0, 10.0, 100.0]) -> rowsums -> z;
    y[2] -> println;
    z[1] -> println;
}";
    assert_eq!(
        named(&lower_ok(src)),
        [
            "total<2>",
            "total<3>",
            "rowsums<2, 3>",
            "rowsums<3, 2>",
            "main"
        ]
    );
}
