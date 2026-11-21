//! **Narrow float storage** (ADR-0043): `bf16` and `f16`, the oracle's
//! conversions and the facts the gates and cost model read.
//!
//! A narrow value is a binary32 value stored in two bytes; it takes no
//! arithmetic. `widen_f32` / `widen_f64` are exact (every narrow value is an
//! `f32`); `to_bf16` / `to_f16` round once, to nearest with ties to even,
//! from the `f32` or `f64` operand itself (never through an intermediate
//! rounding): subnormal results are kept, a magnitude at or past the
//! format's overflow threshold becomes `±inf`, `±0` and `±inf` keep their
//! sign, and a NaN stays a NaN (payload unspecified: no surface observes
//! it — a narrow value prints only after `widen_f32`, and every NaN prints
//! `NaN`). These are IEEE-754's `convertFormat` with `roundTiesToEven`:
//! every backend realizes them bit for bit on every face (the device's
//! `cvt.rn.bf16.f32` / `cvt.rn.f16.f32`, the host's F16C or the integer
//! sequence [`bf16_from_f32`] spells), so the conversions themselves add no
//! term to any face's bound; a value gate on the default face adds one
//! narrow ulp ([`NarrowForm::u`]) per rounding whose operand carries that
//! face's error (ADR-0043 D3: a flip across a rounding boundary).

use crate::ty::NarrowFloat;

/// One format's constants: its significand, its range and the default
/// face's per-rounding term.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NarrowForm {
    pub fmt: NarrowFloat,
    /// Significand bits, the implicit one included (`bf16` 8, `f16` 11).
    pub precision: u32,
    /// The least normal exponent (`bf16` −126, `f16` −14).
    pub emin: i32,
    /// The greatest finite exponent (`bf16` 127, `f16` 15).
    pub emax: i32,
    /// Unit roundoff `2^−precision`: a rounded normal value is within
    /// `u·|x|` of `x`.
    pub u: f64,
    /// Half the least subnormal: a rounded subnormal value is within this
    /// of `x` (`bf16` `2^−134`, `f16` `2^−25`).
    pub abs: f64,
    /// The largest finite value (`bf16` `(2 − 2^−7)·2^127`, `f16` 65504).
    pub max: f64,
}

impl NarrowForm {
    pub fn of(fmt: NarrowFloat) -> NarrowForm {
        let (precision, emin, emax) = match fmt {
            NarrowFloat::Bf16 => (8, -126, 127),
            NarrowFloat::F16 => (11, -14, 15),
        };
        let p = precision as i32;
        NarrowForm {
            fmt,
            precision,
            emin,
            emax,
            u: 2f64.powi(-p),
            abs: 2f64.powi(emin - p),
            max: (2.0 - 2f64.powi(1 - p)) * 2f64.powi(emax),
        }
    }

    /// `|round(x) − x|`'s bound for a finite `x` whose rounding does not
    /// overflow: `u·|x| + abs`.
    pub fn bound(&self, x: f64) -> f64 {
        self.u * x.abs() + self.abs
    }
}

/// `x` rounded once to `fmt` (ties to even), as `fmt`'s bit pattern. `x` is
/// an `f64` or an `f32` widened (exactly) to one: the rounding is from the
/// operand's own value.
pub fn round(fmt: NarrowFloat, x: f64) -> u16 {
    let f = NarrowForm::of(fmt);
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    let (inf, qnan) = match fmt {
        NarrowFloat::Bf16 => (0x7f80u16, 0x7fc0u16),
        NarrowFloat::F16 => (0x7c00u16, 0x7e00u16),
    };
    if x.is_nan() {
        return sign | qnan;
    }
    let a = x.abs();
    if a == 0.0 {
        return sign;
    }
    if a.is_infinite() {
        return sign | inf;
    }
    // The quantum at `a`'s binade (the subnormal quantum below `emin`):
    // `a / q` and `n · q` are exact (powers of two, far from f64's range
    // ends), so `n` is `a`'s rounding to the format's grid.
    if a < f64::MIN_POSITIVE {
        // an f64 subnormal: far below every narrow subnormal — rounds to 0
        return sign;
    }
    let e = (a.to_bits() >> 52) as i32 - 1023;
    let q = 2f64.powi(e.max(f.emin) - (f.precision as i32 - 1));
    let n = (a / q).round_ties_even();
    let r = n * q;
    if r > f.max {
        return sign | inf;
    }
    sign | encode(fmt, r as f32)
}

/// The bit pattern of a non-negative finite value exactly representable in
/// `fmt` (an `f32` carrying it exactly).
fn encode(fmt: NarrowFloat, v: f32) -> u16 {
    let b = v.to_bits();
    match fmt {
        NarrowFloat::Bf16 => (b >> 16) as u16,
        NarrowFloat::F16 => {
            if v == 0.0 {
                return 0;
            }
            let e = ((b >> 23) & 0xff) as i32 - 127;
            let m = b & 0x7f_ffff;
            if e >= -14 {
                (((e + 15) as u16) << 10) | (m >> 13) as u16
            } else {
                // subnormal: v = k · 2^−24, k < 1024
                (v as f64 * 2f64.powi(24)) as u16
            }
        }
    }
}

/// `f32 → bf16` with the integer sequence the host emits (ADR-0043 D5):
/// add `0x7fff + lsb` to the bits and keep the high half — round to nearest
/// even on every finite value (subnormals included; past the largest
/// finite it carries into the infinity pattern); a NaN keeps its sign and
/// is quieted. Equal to [`round`] on every non-NaN `f32`.
pub fn bf16_from_f32(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) as u16) | 0x0040;
    }
    let lsb = (b >> 16) & 1;
    (b.wrapping_add(0x7fff + lsb) >> 16) as u16
}

/// A narrow value widened to `f32` — exact.
pub fn widen(fmt: NarrowFloat, bits: u16) -> f32 {
    match fmt {
        NarrowFloat::Bf16 => f32::from_bits(u32::from(bits) << 16),
        NarrowFloat::F16 => {
            let sign = u32::from(bits & 0x8000) << 16;
            let e = u32::from((bits >> 10) & 0x1f);
            let m = u32::from(bits & 0x3ff);
            let mag = match e {
                0 => return f32::from_bits(sign) + sign_of(sign) * m as f32 * 2f32.powi(-24),
                31 => 0x7f80_0000 | (m << 13),
                _ => ((e + 127 - 15) << 23) | (m << 13),
            };
            f32::from_bits(sign | mag)
        }
    }
}

fn sign_of(sign: u32) -> f32 {
    if sign != 0 { -1.0 } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_narrow_value_round_trips() {
        for fmt in [NarrowFloat::Bf16, NarrowFloat::F16] {
            for b in 0..=u16::MAX {
                let w = widen(fmt, b);
                if w.is_nan() {
                    assert!(widen(fmt, round(fmt, f64::from(w))).is_nan());
                    continue;
                }
                assert_eq!(round(fmt, f64::from(w)), b, "{fmt:?} {b:#06x} = {w}");
                // the f32 source rounds the same as its f64 widening
                assert_eq!(bf16_or(fmt, w), b, "{fmt:?} {b:#06x}");
            }
        }
    }

    fn bf16_or(fmt: NarrowFloat, w: f32) -> u16 {
        match fmt {
            NarrowFloat::Bf16 => bf16_from_f32(w),
            NarrowFloat::F16 => round(fmt, f64::from(w)),
        }
    }

    /// The host's integer sequence and the oracle agree on every 97th `f32`
    /// (4.4·10⁷ values: every binade, subnormals, the overflow threshold,
    /// both infinities); NaNs stay NaNs.
    #[test]
    fn bf16_integer_sequence_is_the_oracle() {
        let mut b: u32 = 0;
        loop {
            let x = f32::from_bits(b);
            let (o, s) = (round(NarrowFloat::Bf16, f64::from(x)), bf16_from_f32(x));
            if x.is_nan() {
                assert!(
                    widen(NarrowFloat::Bf16, s).is_nan() && widen(NarrowFloat::Bf16, o).is_nan()
                );
            } else {
                assert_eq!(o, s, "{x:e} ({b:#010x})");
            }
            match b.checked_add(97) {
                Some(n) => b = n,
                None => break,
            }
        }
    }

    /// Ties go to even, from both widths; f64 rounds once (a value just
    /// above a tie that an intermediate f32 rounding would put on it rounds
    /// up); overflow at the threshold; subnormal results.
    #[test]
    fn ties_overflow_and_subnormals() {
        let bf = NarrowFloat::Bf16;
        // 1 + 2^-8 is a tie between 1 and 1 + 2^-7: even (1).
        assert_eq!(round(bf, 1.0 + 2f64.powi(-8)), 0x3f80);
        // 1 + 3·2^-8: tie between 1 + 2^-7 and 1 + 2^-6: even (1 + 2^-6).
        assert_eq!(round(bf, 1.0 + 3.0 * 2f64.powi(-8)), 0x3f82);
        // just above the tie, below f32's resolution: rounds up from f64 …
        assert_eq!(round(bf, 1.0 + 2f64.powi(-8) + 2f64.powi(-40)), 0x3f81);
        // … where an f32 intermediate would have rounded onto the tie (even).
        assert_eq!(
            bf16_from_f32((1.0 + 2f64.powi(-8) + 2f64.powi(-40)) as f32),
            0x3f80
        );
        let f = NarrowForm::of(NarrowFloat::F16);
        assert_eq!(f.max, 65504.0);
        assert_eq!(round(NarrowFloat::F16, 65504.0), 0x7bff);
        assert_eq!(round(NarrowFloat::F16, 65519.99), 0x7bff);
        assert_eq!(round(NarrowFloat::F16, 65520.0), 0x7c00);
        assert_eq!(round(NarrowFloat::F16, -1e9), 0xfc00);
        assert_eq!(round(NarrowFloat::F16, 2f64.powi(-24)), 0x0001);
        assert_eq!(round(NarrowFloat::F16, 2f64.powi(-25)), 0x0000); // tie to even: 0
        assert_eq!(round(NarrowFloat::F16, 3.0 * 2f64.powi(-25)), 0x0002);
        assert_eq!(round(NarrowFloat::F16, -0.0), 0x8000);
        assert_eq!(widen(NarrowFloat::F16, 0x8001), -(2f32.powi(-24)));
        let b = NarrowForm::of(bf);
        assert_eq!(b.max as f32, f32::from_bits(0x7f7f_0000));
        assert_eq!(round(bf, f64::from(f32::MAX)), 0x7f80);
        assert_eq!(round(bf, 2f64.powi(-133)), 0x0001);
        assert_eq!(widen(bf, 0x0001), 2f32.powi(-133));
        assert_eq!(round(bf, 1e-300), 0);
    }

    /// The host's F16C conversion (`vcvtps2ph`, round to nearest even) is
    /// the oracle on every 13th `f32` — the realization the host takes for
    /// `f16` (ADR-0043 D5).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn f16c_is_the_oracle() {
        if !std::arch::is_x86_feature_detected!("f16c") {
            return;
        }
        #[target_feature(enable = "f16c")]
        unsafe fn cvt(x: f32) -> u16 {
            use std::arch::x86_64::*;
            let v = _mm_set_ss(x);
            _mm_extract_epi16::<0>(_mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v)) as u16
        }
        let mut b: u32 = 0;
        loop {
            let x = f32::from_bits(b);
            let hw = unsafe { cvt(x) };
            let o = round(NarrowFloat::F16, f64::from(x));
            if x.is_nan() {
                assert!(
                    widen(NarrowFloat::F16, hw).is_nan() && widen(NarrowFloat::F16, o).is_nan()
                );
            } else {
                assert_eq!(o, hw, "{x:e} ({b:#010x})");
                assert_eq!(widen(NarrowFloat::F16, o).to_bits(), {
                    use std::arch::x86_64::*;
                    #[target_feature(enable = "f16c")]
                    unsafe fn back(h: u16) -> f32 {
                        _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(h as i16)))
                    }
                    unsafe { back(o) }.to_bits()
                });
            }
            match b.checked_add(13) {
                Some(n) => b = n,
                None => break,
            }
        }
    }
}
