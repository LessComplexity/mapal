//! **Narrow float storage** (ADR-0043): the module-level definitions of
//! `@mapal.to_bf16` / `@mapal.to_f16` / `@mapal.to_f8e4m3` /
//! `@mapal.to_f8e5m2` that every `Narrow` morphism calls, and of the fp8
//! widenings. A body's text is the same on the host and the device
//! (`devmap`), so the realization is chosen where the module is put
//! together, per side — and both sides compute the oracle's bits
//! ([`mapal_ir::narrow::round`]: one rounding to nearest even, subnormals
//! kept, `±inf` past the threshold — saturated to `±max` for fp8):
//!
//! - **device** — `fptrunc`: `cvt.rn.bf16.f32` / `cvt.rn.f16.f32`, one
//!   instruction (the PTX ISA's round-to-nearest-even conversions keep
//!   subnormals). fp8: `cvt.rn.satfinite.{e4m3,e5m2}x2.f32` where the part
//!   has it ([`crate::profile::Caps::cvt_f8`]; inline PTX — LLVM 23 selects
//!   its intrinsic only from PTX 8.1, past sm_89's default 7.8), else the
//!   host's integer sequence.
//! - **host, `f16`** — `fptrunc`: F16C's `vcvtps2ph` (round to nearest even
//!   under the default MXCSR) on an x86 with the feature, the soft-float
//!   `__truncsfhf2` elsewhere.
//! - **host, `bf16`** — the integer sequence `(b + 0x7fff + lsb) >> 16`
//!   ([`mapal_ir::narrow::bf16_from_f32`]), NaN quieted: AVX512-BF16's
//!   `vcvtneps2bf16` flushes subnormal inputs to zero (no bound covers that
//!   on the exact faces), and the soft-float `__truncsfbf2` is a call per
//!   element; the sequence is five integer operations and vectorizes.
//! - **host, fp8** — no x86 fp8 instruction: the integer sequence
//!   [`mapal_ir::narrow::f8_from_f32`] (clamp to `max`'s bits, round the
//!   significand by `+ (2^(d−1) − 1) + lsb`, re-bias; a subnormal by one f32
//!   addition onto the least subnormal's grid; NaN to the format's NaN) —
//!   integer operations and one exact-grid `fadd`, so it vectorizes.
//!
//! Widening is `fpext` for the 16-bit formats (exact: a shift for `bf16`,
//! `cvt.f32.f16` / `vcvtph2ps` for `f16`). An fp8 element (LLVM has no fp8
//! type: `i8` storage) widens by `@mapal.widen_f8e4m3` / `_f8e5m2` (and
//! `.v<N>` for an `<N x i8>` load): the f32 bits built from the byte —
//! sign to bit 31, the magnitude shifted into f32's exponent and
//! significand and re-biased; a subnormal `m·least` by `uitofp` and an exact
//! power-of-two `fmul`; e4m3's `0x7f` / e5m2's top exponent to NaN / `±inf`.
//! Integer operations, the same text on both sides, exact on every face.
//! Each definition is `internal alwaysinline`.

use mapal_ir::NarrowFloat;
use mapal_ir::narrow::{F8Seq, NarrowForm};

/// The LLVM type of a narrow format (`i8` storage for fp8: LLVM has no fp8
/// type).
pub(crate) fn llvm_ty(n: NarrowFloat) -> &'static str {
    match n {
        NarrowFloat::Bf16 => "bfloat",
        NarrowFloat::F16 => "half",
        NarrowFloat::F8e4m3 | NarrowFloat::F8e5m2 => "i8",
    }
}

/// The LLVM literal of a narrow constant (`0xR…` bfloat, `0xH…` half, a
/// signed `i8` for fp8).
pub(crate) fn literal(n: NarrowFloat, bits: u16) -> String {
    match n {
        NarrowFloat::Bf16 => format!("0xR{bits:04X}"),
        NarrowFloat::F16 => format!("0xH{bits:04X}"),
        NarrowFloat::F8e4m3 | NarrowFloat::F8e5m2 => format!("{}", bits as u8 as i8),
    }
}

/// The symbol a body calls to round an `f32` into `n`.
pub(crate) fn symbol(n: NarrowFloat) -> &'static str {
    match n {
        NarrowFloat::Bf16 => "@mapal.to_bf16",
        NarrowFloat::F16 => "@mapal.to_f16",
        NarrowFloat::F8e4m3 => "@mapal.to_f8e4m3",
        NarrowFloat::F8e5m2 => "@mapal.to_f8e5m2",
    }
}

/// The widening of `w` (of LLVM type `nty`: `n`'s scalar or `<N x …>`) to
/// `ty` (`float`, `double` or `<N x float>`), as an instruction's right-hand
/// side (`%v = <this>`): `fpext` for the 16-bit formats; for fp8 a call of
/// the module's `@mapal.widen_*` (an `f32` result; a `double` target is the
/// caller's `fpext` of it — [`widen_lines`]).
pub(crate) fn widen_rhs(n: NarrowFloat, nty: &str, w: &str, ty: &str) -> String {
    match n {
        NarrowFloat::Bf16 | NarrowFloat::F16 => format!("fpext {nty} {w} to {ty}"),
        NarrowFloat::F8e4m3 | NarrowFloat::F8e5m2 => {
            let lanes = nty
                .strip_prefix('<')
                .and_then(|s| s.split_once(" x "))
                .map(|(l, _)| l.trim().to_string());
            match lanes {
                Some(l) => format!("call {ty} {}.v{l}({nty} {w})", widen_symbol(n)),
                None => format!("call float {}(i8 {w})", widen_symbol(n)),
            }
        }
    }
}

/// [`widen_rhs`] as whole instructions into `v` — through an `f32`
/// temporary `t` for an fp8 element (or vector) widened to `double`; `t` is
/// unused otherwise (callers allocate it only for fp8, so a 16-bit body's
/// text and temporaries are unchanged).
pub(crate) fn widen_lines(
    n: NarrowFloat,
    nty: &str,
    w: &str,
    ty: &str,
    v: &str,
    t: &str,
) -> Vec<String> {
    if n.bytes() == 1 && ty.contains("double") {
        let fty = ty.replace("double", "float");
        vec![
            format!("{t} = {}", widen_rhs(n, nty, w, &fty)),
            format!("{v} = fpext {fty} {t} to {ty}"),
        ]
    } else {
        vec![format!("{v} = {}", widen_rhs(n, nty, w, ty))]
    }
}

/// The symbol of an fp8 widening.
fn widen_symbol(n: NarrowFloat) -> &'static str {
    match n {
        NarrowFloat::F8e4m3 => "@mapal.widen_f8e4m3",
        _ => "@mapal.widen_f8e5m2",
    }
}

/// The definitions `text` calls and does not define yet, for one side.
/// `cvt_f8`: the device has `cvt.rn.satfinite.{e4m3,e5m2}x2.f32`.
pub(crate) fn defs(text: &str, device: bool, cvt_f8: bool) -> String {
    let mut out = String::new();
    for n in NarrowFloat::ALL {
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
            (NarrowFloat::F8e4m3 | NarrowFloat::F8e5m2, true) if cvt_f8 => {
                let f = if n == NarrowFloat::F8e4m3 {
                    "e4m3"
                } else {
                    "e5m2"
                };
                format!(
                    "  %p = call i16 asm \"cvt.rn.satfinite.{f}x2.f32 $0, $1, $2;\", \"=h,f,f\"(float 0.0, float %x)\n  \
                     %y = trunc i16 %p to i8\n  ret i8 %y\n"
                )
            }
            (NarrowFloat::F8e4m3 | NarrowFloat::F8e5m2, _) => f8_round_body(n),
            _ => format!("  %y = fptrunc float %x to {t}\n  ret {t} %y\n"),
        };
        out.push_str(&format!(
            "define internal {t} {sym}(float %x) alwaysinline nounwind willreturn memory(none) {{\nentry:\n{body}}}\n"
        ));
    }
    // fp8 widenings: scalar and each vector width a body calls.
    for n in [NarrowFloat::F8e4m3, NarrowFloat::F8e5m2] {
        let sym = widen_symbol(n);
        let mut widths: Vec<Option<u64>> = Vec::new();
        let mut rest = text;
        while let Some(i) = rest.find(sym) {
            let tail = &rest[i + sym.len()..];
            let w = if tail.starts_with('(') {
                Some(None)
            } else if let Some(v) = tail.strip_prefix(".v") {
                let d: String = v.chars().take_while(|c| c.is_ascii_digit()).collect();
                v[d.len()..]
                    .starts_with('(')
                    .then(|| d.parse::<u64>().ok().map(Some))
                    .flatten()
            } else {
                None
            };
            if let Some(w) = w
                && !widths.contains(&w)
            {
                widths.push(w);
            }
            rest = tail;
        }
        for w in widths {
            let (name, it, ft) = match w {
                None => (sym.to_string(), "i8".to_string(), "float".to_string()),
                Some(l) => (
                    format!("{sym}.v{l}"),
                    format!("<{l} x i8>"),
                    format!("<{l} x float>"),
                ),
            };
            if text.contains(&format!("define internal {ft} {name}(")) {
                continue;
            }
            // A part with `cvt.rn.f16x2.e4m3x2` widens an `f8e4m3` pair in
            // one instruction (exact: e4m3 ⊂ f16), then `cvt.f32.f16`; the
            // integer form is ~13 operations an element (the 16-lane GEMV's
            // ALU share, gemv-stream `vec-cells`). `f8e5m2` is f16's top
            // byte: its integer form is already a shift.
            let body = match (n, device && cvt_f8) {
                (NarrowFloat::F8e4m3, true) => f8_widen_cvt_body(w),
                _ => f8_widen_body(n, w),
            };
            out.push_str(&format!(
                "define internal {ft} {name}({it} %b) alwaysinline nounwind willreturn memory(none) {{\nentry:\n{body}}}\n"
            ));
        }
    }
    out
}

/// An LLVM float literal: the `double` holding `v` exactly, in hex.
fn hex(v: f64) -> String {
    format!("0x{:016X}", v.to_bits())
}

/// The host's (and a part without `cvt_f8`'s) fp8 rounding: the integer
/// sequence [`mapal_ir::narrow::f8_from_f32`], line for line.
fn f8_round_body(n: NarrowFloat) -> String {
    let s = F8Seq::of(n).expect("an fp8 format");
    let half = (1u32 << (s.drop - 1)) - 1;
    format!(
        "  %b = bitcast float %x to i32\n  \
         %sh = lshr i32 %b, 24\n  %sg = and i32 %sh, 128\n  \
         %a = and i32 %b, 2147483647\n  %nan = icmp ugt i32 %a, 2139095040\n  \
         %lt = icmp ult i32 %a, {max}\n  %c = select i1 %lt, i32 %a, i32 {max}\n  \
         %ls = lshr i32 %c, {d}\n  %l = and i32 %ls, 1\n  \
         %r0 = add i32 %c, {half}\n  %r1 = add i32 %r0, %l\n  %r2 = lshr i32 %r1, {d}\n  \
         %nm = sub i32 %r2, {rebias}\n  \
         %cf = bitcast i32 %c to float\n  %y = fadd float %cf, {magic}\n  \
         %yb = bitcast float %y to i32\n  %sb = sub i32 %yb, {magic_bits}\n  \
         %isn = icmp uge i32 %c, {minn}\n  %m = select i1 %isn, i32 %nm, i32 %sb\n  \
         %v = or i32 %m, %sg\n  %q = or i32 %sg, 127\n  %o = select i1 %nan, i32 %q, i32 %v\n  \
         %t = trunc i32 %o to i8\n  ret i8 %t\n",
        max = s.max_bits,
        d = s.drop,
        rebias = s.rebias,
        magic = hex(f64::from(s.magic)),
        magic_bits = s.magic_bits,
        minn = s.min_normal_bits,
    )
}

/// An `f8e4m3` element (or `<l x i8>` vector) widened on a part with
/// `cvt.rn.f16x2.e4m3x2` ([`crate::profile::Caps::cvt_f8`]): each byte pair
/// (the lower byte to the lower half) to two `f16` values, exact, then to
/// `f32`. A scalar is the pair `(b, 0)`'s lower half. Inline PTX (LLVM 23
/// selects its intrinsic only from PTX 8.1).
fn f8_widen_cvt_body(lanes: Option<u64>) -> String {
    const CVT: &str = "asm \"cvt.rn.f16x2.e4m3x2 $0, $1;\", \"=r,h\"";
    match lanes {
        None => format!(
            "  %z = zext i8 %b to i16\n  %h = call i32 {CVT}(i16 %z)\n  \
             %t = trunc i32 %h to i16\n  %v = bitcast i16 %t to half\n  \
             %f = fpext half %v to float\n  ret float %f\n"
        ),
        Some(l) if l % 2 == 1 => f8_widen_body(NarrowFloat::F8e4m3, lanes),
        Some(l) => {
            let pairs = l / 2;
            let mut s = format!("  %p = bitcast <{l} x i8> %b to <{pairs} x i16>\n");
            for k in 0..pairs {
                s += &format!(
                    "  %p{k} = extractelement <{pairs} x i16> %p, i64 {k}\n  \
                     %h{k} = call i32 {CVT}(i16 %p{k})\n  \
                     %v{k} = bitcast i32 %h{k} to <2 x half>\n  \
                     %f{k} = fpext <2 x half> %v{k} to <2 x float>\n"
                );
            }
            // concatenate the pairs into one `<l x float>`.
            let mut prev = "poison".to_string();
            for k in 0..pairs {
                for j in 0..2 {
                    let x = format!("%e{k}_{j}");
                    s += &format!("  {x} = extractelement <2 x float> %f{k}, i64 {j}\n");
                    let r = format!("%r{k}_{j}");
                    s += &format!(
                        "  {r} = insertelement <{l} x float> {prev}, float {x}, i64 {}\n",
                        2 * k + j
                    );
                    prev = r;
                }
            }
            s + &format!("  ret <{l} x float> {prev}\n")
        }
    }
}

/// An fp8 element (or `<l x i8>` vector) widened to `f32`: the bits built
/// from the byte ([`mapal_ir::narrow::widen`], exact).
fn f8_widen_body(n: NarrowFloat, lanes: Option<u64>) -> String {
    let f = NarrowForm::of(n);
    let mbits = f.precision - 1;
    let bias = (1 - f.emin) as u32;
    let (it, ft, bt) = match lanes {
        None => ("i32".to_string(), "float".to_string(), "i1".to_string()),
        Some(l) => (
            format!("<{l} x i32>"),
            format!("<{l} x float>"),
            format!("<{l} x i1>"),
        ),
    };
    let src = match lanes {
        None => "i8".to_string(),
        Some(l) => format!("<{l} x i8>"),
    };
    let k = |v: u64| match lanes {
        None => format!("{v}"),
        Some(_) => format!("splat (i32 {v})"),
    };
    let kf = |v: f64| match lanes {
        None => hex(v),
        Some(_) => format!("splat (float {})", hex(v)),
    };
    // the first pattern of the top exponent: e4m3's NaN (`0x7f`, its only
    // non-finite), e5m2's `+inf` (`0x7c`, then the NaNs).
    let (special, special_mag) = match n {
        NarrowFloat::F8e4m3 => (0x7f, None),
        _ => (0x7c, Some(())),
    };
    let top = match special_mag {
        // e4m3: `0x7f` alone, the quiet NaN.
        None => format!(
            "  %isx = icmp eq {it} %m, {}\n  %mag2 = select {bt} %isx, {it} {}, {it} %mag\n",
            k(special),
            k(0x7fc0_0000)
        ),
        // e5m2: `0x7c..0x7f` keep their significand under an all-ones
        // exponent (`±inf`, then NaNs).
        Some(()) => format!(
            "  %isx = icmp uge {it} %m, {}\n  %lo = and {it} %m, {}\n  %los = shl {it} %lo, {}\n  \
             %spec = or {it} %los, {}\n  %mag2 = select {bt} %isx, {it} %spec, {it} %mag\n",
            k(special),
            k((1 << mbits) - 1),
            k(u64::from(23 - mbits)),
            k(0x7f80_0000)
        ),
    };
    format!(
        "  %z = zext {src} %b to {it}\n  %m = and {it} %z, {}\n  %s = and {it} %z, {}\n  \
         %sg = shl {it} %s, {}\n  %nb = shl {it} %m, {}\n  %nr = add {it} %nb, {}\n  \
         %fl = uitofp {it} %m to {ft}\n  %sub = fmul {ft} %fl, {}\n  %subb = bitcast {ft} %sub to {it}\n  \
         %isn = icmp ult {it} %m, {}\n  %mag = select {bt} %isn, {it} %subb, {it} %nr\n\
         {top}  %r = or {it} %mag2, %sg\n  %f = bitcast {it} %r to {ft}\n  ret {ft} %f\n",
        k(0x7f),
        k(0x80),
        k(24),
        k(u64::from(23 - mbits)),
        k(u64::from((127 - bias) << 23)),
        kf(f.least()),
        k(1 << mbits),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widen_rhs_names_the_width() {
        assert_eq!(
            widen_rhs(NarrowFloat::Bf16, "<4 x bfloat>", "%w", "<4 x float>"),
            "fpext <4 x bfloat> %w to <4 x float>"
        );
        assert_eq!(
            widen_rhs(NarrowFloat::F8e4m3, "<16 x i8>", "%w", "<16 x float>"),
            "call <16 x float> @mapal.widen_f8e4m3.v16(<16 x i8> %w)"
        );
        assert_eq!(
            widen_rhs(NarrowFloat::F8e5m2, "i8", "%w", "float"),
            "call float @mapal.widen_f8e5m2(i8 %w)"
        );
        let d = defs(
            "call float @mapal.widen_f8e5m2(i8 %w) call <4 x float> @mapal.widen_f8e5m2.v4(",
            true,
            true,
        );
        assert!(d.contains("define internal float @mapal.widen_f8e5m2(i8 %b)"));
        assert!(d.contains("define internal <4 x float> @mapal.widen_f8e5m2.v4(<4 x i8> %b)"));
        assert!(!d.contains("e4m3"));
        assert_eq!(literal(NarrowFloat::F8e4m3, 0xfe), "-2");
    }
}
