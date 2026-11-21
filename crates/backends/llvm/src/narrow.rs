//! **Narrow float storage** (ADR-0043): the module-level definitions of
//! `@mapal.to_bf16` / `@mapal.to_f16` that every `Narrow` morphism calls. A
//! body's text is the same on the host and the device (`devmap`), so the
//! realization is chosen where the module is put together, per side — and
//! both sides compute the oracle's bits ([`mapal_ir::narrow::round`]: one
//! rounding to nearest even, subnormals kept, `±inf` past the threshold):
//!
//! - **device** — `fptrunc`: `cvt.rn.bf16.f32` / `cvt.rn.f16.f32`, one
//!   instruction (the PTX ISA's round-to-nearest-even conversions keep
//!   subnormals).
//! - **host, `f16`** — `fptrunc`: F16C's `vcvtps2ph` (round to nearest even
//!   under the default MXCSR) on an x86 with the feature, the soft-float
//!   `__truncsfhf2` elsewhere.
//! - **host, `bf16`** — the integer sequence `(b + 0x7fff + lsb) >> 16`
//!   ([`mapal_ir::narrow::bf16_from_f32`]), NaN quieted: AVX512-BF16's
//!   `vcvtneps2bf16` flushes subnormal inputs to zero (no bound covers that
//!   on the exact faces), and the soft-float `__truncsfbf2` is a call per
//!   element; the sequence is five integer operations and vectorizes.
//!
//! Widening is `fpext` everywhere (exact: a shift for `bf16`, `cvt.f32.f16`
//! / `vcvtph2ps` for `f16`). Each definition is `internal alwaysinline`.

use mapal_ir::NarrowFloat;

/// The LLVM type of a narrow format.
pub(crate) fn llvm_ty(n: NarrowFloat) -> &'static str {
    match n {
        NarrowFloat::Bf16 => "bfloat",
        NarrowFloat::F16 => "half",
    }
}

/// The LLVM literal of a narrow constant (`0xR…` bfloat, `0xH…` half).
pub(crate) fn literal(n: NarrowFloat, bits: u16) -> String {
    match n {
        NarrowFloat::Bf16 => format!("0xR{bits:04X}"),
        NarrowFloat::F16 => format!("0xH{bits:04X}"),
    }
}

/// The symbol a body calls to round an `f32` into `n`.
pub(crate) fn symbol(n: NarrowFloat) -> &'static str {
    match n {
        NarrowFloat::Bf16 => "@mapal.to_bf16",
        NarrowFloat::F16 => "@mapal.to_f16",
    }
}

/// The definitions `text` calls and does not define yet, for one side.
pub(crate) fn defs(text: &str, device: bool) -> String {
    let mut out = String::new();
    for n in [NarrowFloat::Bf16, NarrowFloat::F16] {
        let (sym, t) = (symbol(n), llvm_ty(n));
        if !text.contains(&format!("{sym}("))
            || text.contains(&format!("define internal {t} {sym}("))
        {
            continue;
        }
        let body = match (n, device) {
            (NarrowFloat::Bf16, false) => "  %b = bitcast float %x to i32\n  \
                 %h = lshr i32 %b, 16\n  %l = and i32 %h, 1\n  %k = add i32 %l, 32767\n  \
                 %s = add i32 %b, %k\n  %r = lshr i32 %s, 16\n  %q = or i32 %h, 64\n  \
                 %u = fcmp uno float %x, 0.0\n  %v = select i1 %u, i32 %q, i32 %r\n  \
                 %w = trunc i32 %v to i16\n  %y = bitcast i16 %w to bfloat\n  ret bfloat %y\n"
                .to_string(),
            _ => format!("  %y = fptrunc float %x to {t}\n  ret {t} %y\n"),
        };
        out.push_str(&format!(
            "define internal {t} {sym}(float %x) alwaysinline nounwind willreturn memory(none) {{\nentry:\n{body}}}\n"
        ));
    }
    out
}
