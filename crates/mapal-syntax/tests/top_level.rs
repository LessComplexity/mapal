//! ADR-0048 surface: top-level binding chains (D1), a type's size list and a
//! size-generic type's sizes at a use (D4), `_` as a field name (D5) — the
//! parse trees, the diagnostics that stay (P0012, P0103, P0001) and the
//! formatter's layout.

use mapal_syntax::{Item, SizeKind, StageKind, TyKind, format_source, parse};

fn codes(src: &str) -> Vec<&'static str> {
    parse(src).diagnostics.iter().map(|d| d.code.0).collect()
}

#[test]
fn a_top_level_chain_is_a_binding_item() {
    let po = parse("2048 -> hidden_size;\nhidden_size / 2 -> half: i32;\nfn main() { }");
    assert!(po.diagnostics.is_empty(), "{:?}", po.diagnostics);
    let binds: Vec<_> = po
        .program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Bind(b) => Some(b),
            _ => None,
        })
        .collect();
    assert_eq!(binds.len(), 2);
    assert!(matches!(
        binds[1].chain.stages.last().map(|s| &s.kind),
        Some(StageKind::Bind { ty: Some(_), .. })
    ));
}

#[test]
fn other_top_level_statements_stay_p0012_and_a_missing_semicolon_is_p0001() {
    assert!(codes("x <- 5;\nfn main() { }").contains(&"P0012"));
    assert!(codes("loop { }\nfn main() { }").contains(&"P0012"));
    assert!(codes("2048 -> n\nfn main() { }").contains(&"P0001"));
}

#[test]
fn a_type_declares_sizes_and_a_use_states_them() {
    let src = "type Matrix<rows, columns> { _: [bf16; rows * columns] }
fn f(w: Matrix<n * 2, 64>) -> f32 { 1.0 -> ret; }
fn main() { }";
    let po = parse(src);
    assert!(po.diagnostics.is_empty(), "{:?}", po.diagnostics);
    let Item::Type(td) = &po.program.items[0] else {
        panic!("type item")
    };
    assert_eq!(td.sizes.len(), 2);
    assert_eq!(
        &src[td.fields[0].name.span.start as usize..td.fields[0].name.span.end as usize],
        "_"
    );
    let Item::Fn(fd) = &po.program.items[1] else {
        panic!("fn item")
    };
    let TyKind::Generic { args, .. } = &fd.params[0].ty.kind else {
        panic!("generic type use: {:?}", fd.params[0].ty.kind)
    };
    assert_eq!(args.len(), 2);
    assert!(matches!(args[0].kind, SizeKind::Binary { .. }));
    assert!(matches!(args[1].kind, SizeKind::Int(64)));
}

#[test]
fn a_type_argument_that_is_a_type_stays_p0103() {
    assert!(codes("fn f(x: Vec<f32>) { }").contains(&"P0103"));
    assert!(codes("fn f(x: Result<T, E>) { }").contains(&"P0103"));
    assert!(codes("fn f(x: Wrap<[f32; 4]>) { }").contains(&"P0103"));
    assert!(!codes("fn f(x: Wrap<n, 4>) { }").contains(&"P0103"));
}

#[test]
fn the_formatter_keeps_size_lists_tight_and_bindings_as_written() {
    let src = "2048 -> hidden_size;\n0.5 -> half;\ntype Matrix < rows , columns > { _: [bf16; rows * columns] }\ntype Layer { _: Matrix < hidden_size , hidden_size >, scale: f32 }\nfn main() { }\n";
    let out = format_source(src).expect("formats");
    assert!(
        out.starts_with("2048 -> hidden_size;\n0.5 -> half;\n"),
        "{out}"
    );
    assert!(out.contains("type Matrix<rows, columns> {"), "{out}");
    assert!(
        out.contains("    _: Matrix<hidden_size, hidden_size>,\n"),
        "{out}"
    );
    assert_eq!(format_source(&out).expect("idempotent"), out);
}
