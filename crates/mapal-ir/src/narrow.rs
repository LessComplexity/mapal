//! **Narrow float storage** (ADR-0043): `bf16`, `f16` and the fp8 formats
//! `f8e4m3` / `f8e5m2` (slice 3), the oracle's conversions and the facts the
//! gates and cost model read.
//!
//! A narrow value is a binary32 value stored in two bytes (one for fp8); it
//! takes no arithmetic. `widen_f32` / `widen_f64` are exact (every narrow
//! value is an `f32`); `to_bf16` / `to_f16` / `to_f8e4m3` / `to_f8e5m2`
//! round once, to nearest with ties to even, from the `f32` or `f64` operand
//! itself (never through an intermediate rounding): subnormal results are
//! kept, `±0` keeps its sign, and a NaN stays a NaN (payload unspecified: no
//! surface observes it — a narrow value prints only after `widen_f32`, and
//! every NaN prints `NaN`). Past the largest finite value the 16-bit formats
//! overflow to `±inf` (IEEE-754's `convertFormat` with `roundTiesToEven`);
//! the fp8 formats **saturate** to `±max` (the PTX ISA's
//! `cvt.rn.satfinite`, OCP's `SAT` mode: `±inf` included — clamping is
//! monotone and 1-Lipschitz, so the bound of D3 holds through it). Every
//! backend realizes them bit for bit on every face (the device's
//! `cvt.rn.bf16.f32` / `cvt.rn.f16.f32` / `cvt.rn.satfinite.e4m3x2.f32`, the
//! host's F16C or the integer sequences [`bf16_from_f32`] / [`f8_from_f32`]
//! spell), so the conversions themselves add no term to any face's bound; a
//! value gate on the default face adds one narrow ulp ([`NarrowForm::u`])
//! per rounding whose operand carries that face's error (ADR-0043 D3: a flip
//! across a rounding boundary).

use crate::ty::NarrowFloat;

/// One format's constants: its significand, its range and the default
/// face's per-rounding term.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NarrowForm {
    pub fmt: NarrowFloat,
    /// Significand bits, the implicit one included (`bf16` 8, `f16` 11,
    /// `f8e4m3` 4, `f8e5m2` 3).
    pub precision: u32,
    /// The least normal exponent (`bf16` −126, `f16` −14, `f8e4m3` −6,
    /// `f8e5m2` −14).
    pub emin: i32,
    /// The greatest finite exponent (`bf16` 127, `f16` 15, `f8e4m3` 8,
    /// `f8e5m2` 15).
    pub emax: i32,
    /// Unit roundoff `2^−precision`: a rounded normal value is within
    /// `u·|x|` of `x`.
    pub u: f64,
    /// Half the least subnormal: a rounded subnormal value is within this
    /// of `x` (`bf16` `2^−134`, `f16` `2^−25`, `f8e4m3` `2^−10`, `f8e5m2`
    /// `2^−17`).
    pub abs: f64,
    /// The largest finite value (`bf16` `(2 − 2^−7)·2^127`, `f16` 65504,
    /// `f8e4m3` 448 — its top significand at `emax` is the NaN — `f8e5m2`
    /// 57344).
    pub max: f64,
    /// The rounding clamps to `±max` instead of overflowing to `±inf` (the
    /// fp8 formats: `satfinite`).
    pub saturates: bool,
}

impl NarrowForm {
    pub fn of(fmt: NarrowFloat) -> NarrowForm {
        let (precision, emin, emax) = match fmt {
            NarrowFloat::Bf16 => (8, -126, 127),
            NarrowFloat::F16 => (11, -14, 15),
            NarrowFloat::F8e4m3 => (4, -6, 8),
            NarrowFloat::F8e5m2 => (3, -14, 15),
        };
        let p = precision as i32;
        let max = match fmt {
            // E4M3 "FN": `S.1111.111` is the NaN, so the top binade stops
            // one significand short (1.75·2^8).
            NarrowFloat::F8e4m3 => 448.0,
            _ => (2.0 - 2f64.powi(1 - p)) * 2f64.powi(emax),
        };
        NarrowForm {
            fmt,
            precision,
            emin,
            emax,
            u: 2f64.powi(-p),
            abs: 2f64.powi(emin - p),
            max,
            saturates: matches!(fmt, NarrowFloat::F8e4m3 | NarrowFloat::F8e5m2),
        }
    }

    /// `|round(x) − x|`'s bound for a finite `x` whose rounding does not
    /// overflow (or saturate): `u·|x| + abs`.
    pub fn bound(&self, x: f64) -> f64 {
        self.u * x.abs() + self.abs
    }

    /// The least positive (subnormal) value: `2^(emin − precision + 1)`.
    pub fn least(&self) -> f64 {
        2f64.powi(self.emin - self.precision as i32 + 1)
    }

    /// **Format embedding** (ADR-0043 slice 3, tensor-core admission over
    /// narrow storage): every finite value of this format is exactly a value
    /// of the format with `precision` significand bits, least subnormal
    /// `least` and largest finite `max` — `p_A ≤ p_B`, `least_A ≥ least_B`
    /// (both powers of two: A's grid is a coarsening of B's below B's least
    /// normal) and `max_A ≤ max_B`. So `f8e4m3`, `f8e5m2` ⊂ `f16` ⊂ TF32,
    /// `f8e4m3`, `f8e5m2` ⊂ `bf16` ⊂ TF32; `bf16` ⊄ `f16` and `f16` ⊄ `bf16`.
    /// A narrow operand read by an instruction whose format it embeds in is
    /// the instruction's operand unchanged (its multiplicand rounding is the
    /// identity: unit roundoff 0 in the bound, [`crate::bound_coef_units`]).
    pub fn embeds_in(&self, precision: u32, least: f64, max: f64) -> bool {
        self.precision <= precision && self.least() >= least && self.max <= max
    }

    /// [`Self::embeds_in`] another narrow format.
    pub fn embeds_in_form(&self, other: &NarrowForm) -> bool {
        self.embeds_in(other.precision, other.least(), other.max)
    }
}

/// The bit patterns of a format's infinity (`None`: none — the rounding
/// saturates) and quiet NaN.
fn specials(fmt: NarrowFloat) -> (Option<u16>, u16) {
    match fmt {
        NarrowFloat::Bf16 => (Some(0x7f80), 0x7fc0),
        NarrowFloat::F16 => (Some(0x7c00), 0x7e00),
        NarrowFloat::F8e4m3 => (None, 0x7f),
        NarrowFloat::F8e5m2 => (Some(0x7c), 0x7f),
    }
}

/// The sign bit of a format's pattern.
fn sign_bit(fmt: NarrowFloat) -> u16 {
    match fmt.bytes() {
        1 => 0x80,
        _ => 0x8000,
    }
}

/// `x` rounded once to `fmt` (ties to even), as `fmt`'s bit pattern (an fp8
/// pattern in the low byte). `x` is an `f64` or an `f32` widened (exactly)
/// to one: the rounding is from the operand's own value. The fp8 formats
/// saturate: a rounding past `max` (and `±inf`) gives `±max`.
pub fn round(fmt: NarrowFloat, x: f64) -> u16 {
    let f = NarrowForm::of(fmt);
    let sign: u16 = if x.is_sign_negative() {
        sign_bit(fmt)
    } else {
        0
    };
    let (inf, qnan) = specials(fmt);
    let top = |sign: u16| match (f.saturates, inf) {
        (false, Some(i)) => sign | i,
        _ => sign | encode(fmt, f.max as f32),
    };
    if x.is_nan() {
        return sign | qnan;
    }
    let a = x.abs();
    if a == 0.0 {
        return sign;
    }
    if a.is_infinite() {
        return top(sign);
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
        return top(sign);
    }
    sign | encode(fmt, r as f32)
}

/// The bit pattern of a non-negative finite value exactly representable in
/// `fmt` (an `f32` carrying it exactly).
fn encode(fmt: NarrowFloat, v: f32) -> u16 {
    let b = v.to_bits();
    if fmt == NarrowFloat::Bf16 {
        return (b >> 16) as u16;
    }
    if v == 0.0 {
        return 0;
    }
    let f = NarrowForm::of(fmt);
    let mbits = f.precision - 1;
    let e = ((b >> 23) & 0xff) as i32 - 127;
    let m = b & 0x7f_ffff;
    if e >= f.emin {
        let bias = 1 - f.emin;
        (((e + bias) as u16) << mbits) | (m >> (23 - mbits)) as u16
    } else {
        // subnormal: v = k · least, k < 2^mbits
        (f64::from(v) / f.least()) as u16
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

/// The constants of the fp8 integer sequence ([`f8_from_f32`]): the bits of
/// `max`, of the least normal `2^emin`, the f32 significand bits dropped,
/// the exponent re-bias (`(127 − bias) << mbits`) and the subnormal
/// grid's magic addend `2^(23 + log2 least)` with its bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct F8Seq {
    pub max_bits: u32,
    pub min_normal_bits: u32,
    pub drop: u32,
    pub rebias: u32,
    pub magic: f32,
    pub magic_bits: u32,
}

impl F8Seq {
    pub fn of(fmt: NarrowFloat) -> Option<F8Seq> {
        let f = NarrowForm::of(fmt);
        if !f.saturates {
            return None;
        }
        let mbits = f.precision - 1;
        let bias = (1 - f.emin) as u32;
        let magic = (f.least() * 2f64.powi(23)) as f32;
        Some(F8Seq {
            max_bits: (f.max as f32).to_bits(),
            min_normal_bits: (2f32).powi(f.emin).to_bits(),
            drop: 23 - mbits,
            rebias: (127 - bias) << mbits,
            magic,
            magic_bits: magic.to_bits(),
        })
    }
}

/// `f32 → f8e4m3 / f8e5m2` with the integer sequence the host emits (no
/// fp8 instruction on x86): `a = |x|` clamped to `max`'s bits (saturation:
/// rounding is monotone and `max` is on the grid, so clamping first is
/// clamping after; `±inf` lands on `max`), then a normal `a` rounds its
/// significand to nearest even by `a + (2^(drop−1) − 1) + lsb` and drops
/// `drop` bits, re-biased; a subnormal `a` rounds onto the grid of the least
/// subnormal by one f32 addition of `2^(23 + log2 least)` (round to nearest
/// even, the default mode: the sum's low significand bits are the count of
/// least subnormals). NaN gives the format's NaN. Every step is an integer
/// operation or one exact-grid f32 addition, so a loop of them vectorizes.
/// Equal to [`round`] on every `f32`.
pub fn f8_from_f32(fmt: NarrowFloat, x: f32) -> u8 {
    let s = F8Seq::of(fmt).expect("an fp8 format");
    let b = x.to_bits();
    let sign = (b >> 24) & 0x80;
    let a = b & 0x7fff_ffff;
    if a > 0x7f80_0000 {
        return (sign | 0x7f) as u8;
    }
    let c = a.min(s.max_bits);
    let lsb = (c >> s.drop) & 1;
    let normal = ((c + ((1 << (s.drop - 1)) - 1) + lsb) >> s.drop).wrapping_sub(s.rebias);
    let sub = (f32::from_bits(c) + s.magic)
        .to_bits()
        .wrapping_sub(s.magic_bits);
    let m = if c >= s.min_normal_bits { normal } else { sub };
    ((m | sign) & 0xff) as u8
}

/// An fp8 pattern as the `f16` pattern it widens through (exact: both fp8
/// formats embed in `f16`): `f8e5m2` is `f16`'s top byte (`b << 8`, the
/// infinities and NaNs included); `f8e4m3`'s magnitude bits shifted into
/// `f16`'s exponent and significand (`(b & 0x7f) << 7`) are `x·2⁻⁸` for every
/// normal *and* subnormal `x` (the bias differs by 8), the sign moved to bit
/// 15 — the device's integer widening and the host's, which multiply the
/// `f16`'s value by `2⁸` (exact) and map `0x7f` to the NaN.
pub fn f8_as_f16_bits(fmt: NarrowFloat, b: u8) -> u16 {
    let b = u16::from(b);
    match fmt {
        NarrowFloat::F8e5m2 => b << 8,
        NarrowFloat::F8e4m3 => ((b & 0x7f) << 7) | ((b & 0x80) << 8),
        _ => panic!("f8_as_f16_bits of {fmt:?}"),
    }
}

/// A narrow value widened to `f32` — exact.
pub fn widen(fmt: NarrowFloat, bits: u16) -> f32 {
    if fmt == NarrowFloat::Bf16 {
        return f32::from_bits(u32::from(bits) << 16);
    }
    let f = NarrowForm::of(fmt);
    let mbits = f.precision - 1;
    let ebits = 8 * fmt.bytes() as u32 - 1 - mbits;
    let sign = u32::from(bits & sign_bit(fmt)) << (32 - 8 * fmt.bytes() as u32);
    let e = (u32::from(bits) >> mbits) & ((1 << ebits) - 1);
    let m = u32::from(bits) & ((1 << mbits) - 1);
    let top = (1 << ebits) - 1;
    let bias = (1 - f.emin) as u32;
    let mag = match e {
        0 => return f32::from_bits(sign) + sign_of(sign) * (f64::from(m) * f.least()) as f32,
        // E4M3: the top exponent is finite but for `S.1111.111`.
        _ if e == top && fmt == NarrowFloat::F8e4m3 && m == (1 << mbits) - 1 => 0x7fc0_0000,
        _ if e == top && fmt != NarrowFloat::F8e4m3 => 0x7f80_0000 | (m << (23 - mbits)),
        _ => ((e + 127 - bias) << 23) | (m << (23 - mbits)),
    };
    f32::from_bits(sign | mag)
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
            _ => round(fmt, f64::from(w)),
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
        assert_eq!(widen(bf, 0x0001), f32::from_bits(0x0001_0000)); // 2^-133 (`powi` overflows past 2^-128)
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

    const F8: [NarrowFloat; 2] = [NarrowFloat::F8e4m3, NarrowFloat::F8e5m2];

    /// An independent reference: the nearest of the format's 256 widened
    /// values (ties to the even pattern), saturating to `±max` — no shared
    /// code with [`round`] beyond [`widen`], itself checked by hand values.
    fn nearest(fmt: NarrowFloat, x: f64) -> u16 {
        let f = NarrowForm::of(fmt);
        let s = if x.is_sign_negative() { 0x80 } else { 0 };
        if x.is_nan() {
            return s | 0x7f;
        }
        let a = x.abs().min(f.max);
        let mut best: (f64, u16) = (f64::INFINITY, 0);
        for b in 0u16..0x80 {
            let w = f64::from(widen(fmt, b));
            if !w.is_finite() {
                continue;
            }
            let d = (w - a).abs();
            if d < best.0 || (d == best.0 && b & 1 == 0) {
                best = (d, b);
            }
        }
        s | best.1
    }

    /// Every fp8 pattern widens and rounds back to itself (NaN to a NaN);
    /// the hand values of both formats; `embeds_in`'s table.
    #[test]
    fn every_f8_value_round_trips() {
        for fmt in F8 {
            for b in 0u16..=0xff {
                let w = widen(fmt, b);
                if w.is_nan() {
                    assert!(widen(fmt, round(fmt, f64::from(w))).is_nan());
                    continue;
                }
                if w.is_infinite() {
                    // e5m2's infinities saturate on the way back.
                    assert_eq!(round(fmt, f64::from(w)) & 0x7f, 0x7b, "{fmt:?} {b:#04x}");
                    continue;
                }
                assert_eq!(round(fmt, f64::from(w)), b, "{fmt:?} {b:#04x} = {w}");
                assert_eq!(u16::from(f8_from_f32(fmt, w)), b, "{fmt:?} {b:#04x}");
                // the f16 pattern it widens through is exact
                let h = widen(NarrowFloat::F16, f8_as_f16_bits(fmt, b as u8));
                let scale = if fmt == NarrowFloat::F8e4m3 {
                    256.0
                } else {
                    1.0
                };
                assert_eq!(h * scale, w, "{fmt:?} {b:#04x} through f16");
            }
        }
        let e4 = NarrowFloat::F8e4m3;
        assert_eq!(widen(e4, 0x7e), 448.0);
        assert_eq!(widen(e4, 0x01), 2f32.powi(-9));
        assert_eq!(widen(e4, 0x08), 2f32.powi(-6));
        assert_eq!(widen(e4, 0x38), 1.0);
        assert!(widen(e4, 0x7f).is_nan() && widen(e4, 0xff).is_nan());
        let e5 = NarrowFloat::F8e5m2;
        assert_eq!(widen(e5, 0x7b), 57344.0);
        assert_eq!(widen(e5, 0x7c), f32::INFINITY);
        assert_eq!(widen(e5, 0x01), 2f32.powi(-16));
        assert_eq!(widen(e5, 0x3c), 1.0);
        assert_eq!(NarrowForm::of(e4).abs, 2f64.powi(-10));
        assert_eq!(NarrowForm::of(e5).abs, 2f64.powi(-17));
        assert_eq!(NarrowForm::of(e5).max, 57344.0);
        // embedding: fp8 ⊂ f16, ⊂ bf16; bf16 ⊄ f16, f16 ⊄ bf16
        let form = NarrowForm::of;
        for a in F8 {
            for b in [NarrowFloat::F16, NarrowFloat::Bf16] {
                assert!(form(a).embeds_in_form(&form(b)), "{a:?} ⊂ {b:?}");
                assert!(!form(b).embeds_in_form(&form(a)), "{b:?} ⊄ {a:?}");
            }
        }
        assert!(!form(e4).embeds_in_form(&form(e5)) && !form(e5).embeds_in_form(&form(e4)));
        assert!(!form(NarrowFloat::Bf16).embeds_in_form(&form(NarrowFloat::F16)));
        assert!(!form(NarrowFloat::F16).embeds_in_form(&form(NarrowFloat::Bf16)));
        assert!(form(NarrowFloat::F16).embeds_in_form(&form(NarrowFloat::F16)));
    }

    /// Ties between adjacent fp8 values go to the even pattern; the
    /// saturation boundaries; subnormal endpoints; `±inf` and NaN.
    #[test]
    fn f8_ties_saturation_and_subnormals() {
        for fmt in F8 {
            for b in 0u16..0x7f {
                let (lo, hi) = (f64::from(widen(fmt, b)), f64::from(widen(fmt, b + 1)));
                if !hi.is_finite() || !lo.is_finite() {
                    continue;
                }
                let even = if b & 1 == 0 { b } else { b + 1 };
                assert_eq!(round(fmt, (lo + hi) / 2.0), even, "{fmt:?} tie {b:#04x}");
                assert_eq!(round(fmt, -(lo + hi) / 2.0), 0x80 | even);
            }
        }
        let e4 = NarrowFloat::F8e4m3;
        assert_eq!(round(e4, 448.0), 0x7e);
        assert_eq!(round(e4, 463.99), 0x7e);
        assert_eq!(round(e4, 464.0), 0x7e); // tie with 480 (not a value): saturates
        assert_eq!(round(e4, 1e30), 0x7e);
        assert_eq!(round(e4, f64::INFINITY), 0x7e);
        assert_eq!(round(e4, f64::NEG_INFINITY), 0xfe);
        assert_eq!(round(e4, 2f64.powi(-10)), 0x00); // half the least: tie to 0
        assert_eq!(round(e4, 3.0 * 2f64.powi(-10)), 0x02);
        assert_eq!(round(e4, -0.0), 0x80);
        assert_eq!(round(e4, f64::NAN) & 0x7f, 0x7f);
        let e5 = NarrowFloat::F8e5m2;
        assert_eq!(round(e5, 57344.0), 0x7b);
        assert_eq!(round(e5, 61440.0), 0x7b); // the tie with 65536: saturates
        assert_eq!(round(e5, f64::INFINITY), 0x7b);
        assert_eq!(round(e5, 2f64.powi(-17)), 0x00);
        assert_eq!(round(e5, 2f64.powi(-16)), 0x01);
        assert_eq!(round(e5, f64::NAN) & 0x7f, 0x7f);
        for fmt in F8 {
            assert_eq!(
                f8_from_f32(fmt, f32::INFINITY),
                round(fmt, f64::INFINITY) as u8
            );
            assert_eq!(f8_from_f32(fmt, f32::NAN) & 0x7f, 0x7f);
            assert_eq!(f8_from_f32(fmt, -f32::NAN) & 0x7f, 0x7f);
        }
    }

    /// The oracle, the host's integer sequence and an independent nearest
    /// search agree on every 97th `f32` (every binade, subnormals, the
    /// saturation threshold, both infinities); NaNs stay NaNs.
    #[test]
    fn f8_integer_sequence_is_the_oracle() {
        for fmt in F8 {
            let mut b: u32 = 0;
            loop {
                let x = f32::from_bits(b);
                let (o, s) = (round(fmt, f64::from(x)), f8_from_f32(fmt, x));
                if x.is_nan() {
                    assert!(widen(fmt, o).is_nan() && widen(fmt, u16::from(s)).is_nan());
                } else {
                    assert_eq!(o, u16::from(s), "{fmt:?} {x:e} ({b:#010x})");
                    // the nearest search is O(128): every 97·61th
                    if b % (97 * 61) == 0 {
                        assert_eq!(o, nearest(fmt, f64::from(x)), "{fmt:?} {x:e}");
                    }
                }
                match b.checked_add(97) {
                    Some(n) => b = n,
                    None => break,
                }
            }
        }
    }
}
