//! Shared Mapal sources and helpers for the golden LLVM IR tests.

use super::*;

pub(crate) const TILE_MATMUL_SRC: &str = r#"
fn matmul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    16 -> iota -> cells;
    4 -> iota -> ks;
    cells -> map { cell ->
        cell / 4 -> i;
        cell % 4 -> j;
        (0.0, ks) -> fold { acc, k -> acc + a[i * 4 + k] * b[k * 4 + j] }
    } -> c;
    c -> ret;
}
fn main() {
    [ -37.0, -30.0, -23.0, -16.0, -9.0, -2.0, 5.0, 12.0,
      19.0, 26.0, 33.0, 40.0, 47.0, -47.0, -40.0, -33.0] -> a: [f32; 16];
    [7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, -45.0,
     -38.0, -31.0, -24.0, -17.0, -10.0, -3.0, 4.0, 11.0] -> b: [f32; 16];
    (a, b) -> matmul -> c;
    c[0] -> println;
    c[15] -> println;
}
"#;

pub(crate) const TILE_MATMUL_F64_SRC: &str = r#"
fn matmul(a: [f64; 30], b: [f64; 100]) -> [f64; 120] {
    120 -> iota -> cells;
    5 -> iota -> ks;
    cells -> map { cell ->
        cell / 20 -> i;
        cell % 20 -> j;
        (0.0, ks) -> fold { acc, k -> acc + a[i * 5 + k] * b[k * 20 + j] }
    } -> c;
    c -> ret;
}
fn main() {
    30 -> iota -> ais;
    ais -> map { x -> (x * 5 + 7) % 67 - 33 -> widen<f64> } -> a;
    100 -> iota -> bis;
    bis -> map { x -> (x * 7 + 11) % 101 - 50 -> widen<f64> } -> b;
    (a, b) -> matmul -> c;
    c[0] -> println;
    c[119] -> println;
}
"#;

pub(crate) const TILE_FIR_SRC: &str = r#"
fn fir(w: [f32; 4], x: [f32; 19]) -> [f32; 16] {
    16 -> iota -> ts;
    4 -> iota -> kr;
    ts -> map { t ->
        (0.0, kr) -> fold { acc, k -> acc + w[k] * x[t + k] }
    } -> y;
    y -> ret;
}
fn main() {
    [1.0, -2.0, 3.0, -4.0] -> w: [f32; 4];
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
     11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0] -> x: [f32; 19];
    (w, x) -> fir -> y;
    y[0] -> println;
    y[15] -> println;
}
"#;

/// A K=300 packed matmul — deep enough that the KC k-panel nest applies under
/// the `generic` profile (kc = 128) and shallow enough that it does NOT under
/// `apple-m` (kc = 4096). Shared by the KC golden and the profile-gate test.
pub(crate) const KC_SRC: &str = r#"
fn matmul(a: [f32; 2400], b: [f32; 9600]) -> [f32; 256] {
    256 -> iota -> cells;
    300 -> iota -> ks;
    cells -> map { cell ->
        cell / 32 -> i;
        cell % 32 -> j;
        (0.0, ks) -> fold { acc, k -> acc + a[i * 300 + k] * b[k * 32 + j] }
    } -> c;
    c -> ret;
}
fn main() {
    2400 -> iota -> ta;
    ta -> map { t -> (t * 7 + 13) % 101 - 50 -> widen<f32> } -> a;
    9600 -> iota -> tb;
    tb -> map { t -> (t * 7 + 57) % 101 - 50 -> widen<f32> } -> b;
    (a, b) -> matmul -> c;
    c[0] -> println;
    c[255] -> println;
}
"#;

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

pub(crate) fn mapal_main(ll: &str) -> &str {
    let marker = ll.find("@mapal_main(").expect("mapal_main");
    let start = ll[..marker]
        .rfind("define internal ")
        .expect("mapal_main def");
    let end = start + ll[start..].find("\n}\n").expect("mapal_main end") + 3;
    &ll[start..end]
}

pub(crate) fn function_containing<'a>(ll: &'a str, needle: &str) -> &'a str {
    let marker = ll
        .find(needle)
        .unwrap_or_else(|| panic!("missing {needle}"));
    let start = if ll[marker..].starts_with("define internal ") {
        marker
    } else {
        ll[..marker]
            .rfind("define internal ")
            .expect("function start")
    };
    let end = start + ll[start..].find("\n}\n").expect("function end") + 3;
    &ll[start..end]
}
