//! ADR-0045 D1/D2/D6: `read` is an effectful pipeline builtin,
//! `"file.bin" -> read -> name: [T; n]` — a string literal path in, the
//! array the binding's annotation names out — resolved by name like `time`
//! (no call syntax). It threads the IO token (one `Read`, `(IoToken, Str) →
//! (IoToken, [T; n])`), so it is illegal in map and fold bodies (L1605) and
//! guard arms (L1404); a missing or non-array annotation is L1617, a path
//! that is not a string literal L1618, and `read` is a reserved name (L1009).
//! The formatter keeps the stage as written.

mod common;
use common::{assert_rejects, lower_ok};
use mapal_ir::{CategoryIr, NarrowFloat, Operation, Ty};

/// Each `Read`'s path literal and array type, in graph order.
fn reads(src: &str) -> Vec<(String, Ty)> {
    let ir: CategoryIr = lower_ok(src);
    ir.morphisms()
        .filter(|(_, m)| m.op == Operation::Read)
        .map(|(_, m)| {
            let path = ir
                .in_edges(m.source)
                .iter()
                .map(|&p| ir.morphism(p).unwrap())
                .find(|p| p.op == Operation::Pair { slot: 1, arity: 2 })
                .and_then(|p| ir.object(p.source).unwrap().value.clone())
                .expect("the path is a constant");
            let mapal_ir::Value::Str(path) = path else {
                panic!("the path is a Str")
            };
            let ty = ir
                .object(m.target)
                .unwrap()
                .ty
                .component_ty(1)
                .unwrap()
                .clone();
            (path, ty)
        })
        .collect()
}

fn arr(elem: Ty, size: u64) -> Ty {
    Ty::Array {
        elem: Box::new(elem),
        size,
    }
}

#[test]
fn read_takes_every_element_type() {
    for (name, elem) in [
        ("i32", Ty::i32()),
        ("i64", Ty::i64()),
        ("u8", Ty::u8()),
        ("f32", Ty::f32()),
        ("f64", Ty::f64()),
        ("bf16", Ty::Narrow(NarrowFloat::Bf16)),
        ("f16", Ty::Narrow(NarrowFloat::F16)),
    ] {
        let widen = match name {
            "bf16" | "f16" => " -> widen<f32>",
            _ => "",
        };
        let src = format!(
            "fn main() {{\n    \"data/{name}.bin\" -> read -> values: [{name}; 6];\n    values[5]{widen} -> println;\n}}\n"
        );
        assert_eq!(
            reads(&src),
            [(format!("data/{name}.bin"), arr(elem, 6))],
            "{src}"
        );
    }
}

#[test]
fn read_is_token_threaded_and_ordered() {
    // Two reads and two prints: one token chain, in source order.
    let src = r#"fn main() {
    "first" -> println;
    "a.bin" -> read -> a: [f32; 4];
    "b.bin" -> read -> b: [i32; 2];
    a[0] -> println;
    b[1] -> println;
}"#;
    assert_eq!(
        reads(src),
        [
            ("a.bin".into(), arr(Ty::f32(), 4)),
            ("b.bin".into(), arr(Ty::i32(), 2))
        ]
    );
    let ir = lower_ok(src);
    assert!(mapal_ir::validate(&ir).is_empty());
    // Each read's source pairs the token with the path, its target the next
    // token with the array.
    for (_, m) in ir.morphisms().filter(|(_, m)| m.op == Operation::Read) {
        assert_eq!(
            ir.object(m.source).unwrap().ty,
            Ty::Tuple(vec![Ty::IoToken, Ty::Str])
        );
        assert!(matches!(&ir.object(m.target).unwrap().ty, Ty::Tuple(t) if t[0] == Ty::IoToken));
    }
}

#[test]
fn read_in_an_effectful_function() {
    let src = r#"fn load(scale: f32) -> [f32; 3] {
    "w.bin" -> read -> w: [f32; 3];
    w -> map { v -> v * scale } -> ret;
}
fn main() {
    2.0 -> load -> w;
    w[2] -> println;
}"#;
    assert_eq!(reads(src), [("w.bin".into(), arr(Ty::f32(), 3))]);
}

#[test]
fn read_past_two_to_the_31_cells_takes_the_index_type() {
    // An extent past i32 is a type like any other (`Ty::index_ty` gives its
    // indices i64); lowering never materializes the bytes.
    let n = (1u64 << 31) + 8;
    let src = format!(
        "fn main() {{\n    \"big.bin\" -> read -> big: [u8; {n}];\n    big[0] -> println;\n}}\n"
    );
    assert_eq!(reads(&src), [("big.bin".into(), arr(Ty::u8(), n))]);
    assert_eq!(Ty::index_ty(n), Ty::i64());
}

#[test]
fn a_missing_or_non_array_annotation_is_l1617() {
    for src in [
        // no binding at all
        r#"fn main() { "w.bin" -> read -> println; }"#,
        // a binding without its type
        r#"fn main() { "w.bin" -> read -> w; w[0] -> println; }"#,
        // a scalar
        r#"fn main() { "w.bin" -> read -> w: f32; w -> println; }"#,
        // an element type a file cannot hold
        r#"fn main() { "w.bin" -> read -> w: [bool; 2]; w[0] -> println; }"#,
        r#"fn main() { "w.bin" -> read -> w: [f8e4m3; 2]; w[0] -> widen<f32> -> println; }"#,
        // a record whose layout needs padding (an f32 after a u8)
        r#"fn main() { "w.bin" -> read -> w: [(u8, f32); 2]; w[0].1 -> println; }"#,
    ] {
        assert_rejects(src, "L1617");
    }
    // full-model: a value of nested arrays, tuples or structs whose leaves are
    // readable scalars with no padding is the file's bytes, leaves in order.
    for src in [
        r#"fn main() { "w.bin" -> read -> w: [[f32; 2]; 2]; w[0][0] -> println; }"#,
        r#"fn main() { "w.bin" -> read -> w: [(f32, f32); 2]; w[0].0 -> println; }"#,
        r#"type L { g: [bf16; 2], w: [[bf16; 2]; 3] }
fn main() { "w.bin" -> read -> m: [L; 4]; m[1].w[2][0] -> widen<f32> -> println; }"#,
    ] {
        let po = mapal_syntax::parse(src);
        assert!(po.diagnostics.is_empty(), "{:?}", po.diagnostics);
        assert!(mapal_lower::lower(src, &po.program).is_ok(), "{src}");
    }
}

#[test]
fn a_path_that_is_not_a_string_literal_is_l1618() {
    assert_rejects(
        "fn main() { 3 -> read -> w: [i32; 2]; w[0] -> println; }",
        "L1618",
    );
    assert_rejects(
        "fn main() { 2.5 -> p; p -> read -> w: [f32; 2]; w[0] -> println; }",
        "L1618",
    );
}

#[test]
fn read_is_an_effect() {
    // In a map body (bodies are pure).
    assert_rejects(
        r#"fn main() { 4 -> iota -> r; r -> map { t -> "w.bin" -> read -> w: [f32; 2]; w[0] } -> a; a[0] -> println; }"#,
        "L1605",
    );
    // In a fold body.
    assert_rejects(
        r#"fn main() { 4 -> iota -> r; (0.0, r) -> fold { acc, t -> "w.bin" -> read -> w: [f32; 2]; acc + w[0] } -> s; s -> println; }"#,
        "L1605",
    );
}

#[test]
fn read_is_reserved() {
    assert_rejects(
        "fn read(x: i32) -> i32 { x -> ret; } fn main() { 1 -> read -> println; }",
        "L1009",
    );
}

#[test]
fn formatter_keeps_the_stage() {
    let src = "fn main() {\n    \"weights/embedding.bin\" -> read -> token_embedding: [bf16; 64];\n    token_embedding[0] -> widen<f32> -> println;\n}\n";
    let out = mapal_syntax::format_source(src).expect("formats");
    assert_eq!(out, src);
    let messy = "fn main(){\n\"weights/embedding.bin\"->read->token_embedding:[bf16;64];\ntoken_embedding[0]->widen<f32>->println;}\n";
    assert_eq!(mapal_syntax::format_source(messy).expect("formats"), src);
}
