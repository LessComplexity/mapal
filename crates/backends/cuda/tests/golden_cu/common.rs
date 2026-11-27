//! Shared Mapal sources and lowering helpers for the golden CUDA tests.

use super::*;

pub(crate) fn lower_src(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"))
}

pub(crate) fn build_example(name: &str) -> CategoryIr {
    let path = format!(
        "{}/../../../examples/{}.mapal",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let src = std::fs::read_to_string(&path).unwrap();
    lower_src(&src)
}

pub(crate) const IOTA_FILL_SRC: &str = r#"
fn main() {
    4 -> iota -> a;
    4 -> iota -> b;
    (7, 4) -> fill -> sevens;
    (a, b) -> zip -> map { p -> p.0 + p.1 } -> twice;
    (twice, sevens) -> zip -> map { p -> p.0 + p.1 } -> out;
    out[3] -> println;
}
"#;
