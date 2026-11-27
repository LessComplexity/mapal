//! Shared emission and parsing helpers for the half-tile tests.

use super::*;

pub(crate) fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(
        po.diagnostics.is_empty(),
        "parse: {:?}\n{src}",
        po.diagnostics
    );
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}\n{src}"));
    mapal_rewrite::rewrite(ir).ir
}

/// The emission on `target` at `base`'s face, the rungs `off` off; `place`
/// off (every capable site on the device), each site its own kernel.
pub(crate) fn emission(
    ir: &CategoryIr,
    target: &'static str,
    base: EmitOpts,
    off: &[&str],
) -> Emission {
    let mut gpu = GpuArms::default();
    for r in [
        "place",
        "async",
        "pdl",
        "streams",
        "pinned",
        "horizontal",
        "calls",
    ]
    .iter()
    .chain(off)
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "generic",
        target,
        gpu,
        ..base
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// The plan lines of the 16-bit tensor-core kernels a module launches.
pub(crate) fn half_lines(dev: &str) -> Vec<String> {
    dev.lines()
        .filter(|l| l.starts_with("; mapal.plan ") && l.contains(" face=HalfMma "))
        .map(str::to_string)
        .collect()
}

pub(crate) fn floats(out: &str) -> Vec<f64> {
    out.split_whitespace()
        .map(|x| x.parse().expect("a float"))
        .collect()
}

pub(crate) const GEMV: &str = "fn main() {
    2048 -> iota -> ta;
    8192 -> iota -> tw;
    4096 -> iota -> t;
    64 -> iota -> kk;
    ta -> map { e -> (e * 7 + 3) % 31 - 15 -> widen<f32> -> v; v / 29.0 } -> a;
    tw -> map { e -> (e * 7 + 13) % 103 - 51 -> widen<f32> -> v; v / 1856.0 -> to<bf16> } -> w;
    t -> map { e -> e / 128 -> i; e % 128 -> j; (0.0, kk) -> fold: bf16 | f32 { acc, q -> w[q * 128 + j] -> widen<f32> -> b; acc + a[i * 64 + q] * b } } -> y;
    y[0] -> println;
    y[4095] -> println;
}
";
