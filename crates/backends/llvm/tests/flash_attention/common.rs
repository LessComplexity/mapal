//! Shared emission and field-parsing helpers for the flash-attention tests.

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

/// The emission on the rtx-5080 with the stream's geometry forced (`None`:
/// the rule's) and the rungs `off` off; the emulator's rungs off (`pdl`,
/// `streams`, `pinned`), `place` off so every capable site is on the device,
/// `horizontal` / `calls` off (a unit's own launch).
pub(crate) fn emission(
    ir: &CategoryIr,
    force: Option<(u64, u64, u64, u64, u64)>,
    off: &[&str],
) -> Emission {
    let mut gpu = GpuArms::default();
    gpu.on.flow = true;
    gpu.stream_force = force;
    for r in ["place", "pdl", "streams", "pinned", "horizontal", "calls"]
        .iter()
        .chain(off)
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "generic",
        target: "rtx-5080",
        gpu,
        ..EmitOpts::default()
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// The streamed units' intent lines.
pub(crate) fn stream_lines(em: &Emission) -> Vec<String> {
    em.device
        .as_deref()
        .unwrap_or("")
        .lines()
        .filter(|l| l.starts_with("; mapal.flow kernel=") && l.contains(" stream="))
        .map(str::to_string)
        .collect()
}

pub(crate) fn field<'l>(line: &'l str, key: &str) -> &'l str {
    line.split_whitespace()
        .find_map(|f| f.strip_prefix(key).and_then(|v| v.strip_prefix('=')))
        .unwrap_or("")
}

/// Every printed number within the default face's bound of the oracle's
/// (relative `1e-4`; the windows' rescales and the free order).
pub(crate) fn within(got: &str, want: &str) -> bool {
    let (g, w): (Vec<&str>, Vec<&str>) = (
        got.split_whitespace().collect(),
        want.split_whitespace().collect(),
    );
    g.len() == w.len()
        && g.iter().zip(&w).all(|(a, b)| {
            a == b
                || match (a.parse::<f64>(), b.parse::<f64>()) {
                    (Ok(x), Ok(y)) if x.is_nan() && y.is_nan() => true,
                    (Ok(x), Ok(y)) => (x - y).abs() <= 1e-4 * x.abs().max(y.abs()).max(1.0),
                    _ => false,
                }
        })
}

/// The emulator modes: free threads, each block's threads serialized between
/// barriers in order and reversed, the blocks in reverse order.
pub(crate) const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "1")],
];

pub(crate) fn program(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/../../../benches/algorithms/{name}.mapal",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}
