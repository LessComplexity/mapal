//! **`exp`** (ADR-0042): the oracle's exponential and the default face's
//! a-priori bound. The algorithm is the language's semantics on the exact,
//! contract and fixed faces: every backend computes it operation for
//! operation (separate IEEE steps, round to nearest even), so a device, the
//! host and the interpreter agree bit for bit. The default face admits any
//! realization within [`ExpForm::rel`]`·eˣ + `[`ExpForm::abs`] on
//! `[lo, hi]` that keeps the same special values (ADR-0042 D4).
//!
//! The algorithm, `E_b` for `b ∈ {32, 64}` (constants in [`ExpForm`]):
//!
//! ```text
//! xc = x > hi ? hi : x;  xc = xc < lo ? lo : xc;  xc = xc == xc ? xc : 0
//! kf = (xc · log2e + shifter) − shifter          (round to the nearest integer)
//! k  = (int) kf
//! r  = (xc − kf · ln2_hi) − kf · ln2_lo           (Cody–Waite: the first product exact)
//! p  = c_n; p = p · r + c_i for i = n−1 … 0       (Taylor, c_i = 1/i!)
//! k1 = k >> 1; k2 = k − k1                        (both scales normal)
//! y  = (p · 2^k1) · 2^k2                          (the second product rounds once if subnormal)
//! y  = x > hi ? +inf : y;  y = x < lo ? +0 : y;  y = x ≠ x ? x : y
//! ```
//!
//! Measured (`docs/decisions/ADR-0042-exp-builtin.md`): `E_32` exhaustively
//! over every f32 in `[lo, hi]` ≤ 1.221 ulp (relative 2^−23.21 on normal
//! results), its contracted form ≤ 0.939 ulp; `E_64` ≤ 1.0 ulp of glibc's
//! `exp` over 2·10⁸ samples.

/// One width's constants: the oracle's algorithm and the default face's
/// bound. Values are bit patterns so every backend prints the same number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpForm {
    /// Float width.
    pub bits: u32,
    /// `x > hi` is `+inf` (the largest `x` the algorithm keeps finite).
    pub hi: u64,
    /// `x < lo` is `+0` (below, `eˣ` rounds to zero).
    pub lo: u64,
    pub log2e: u64,
    /// `1.5 · 2^(p−1)`: adding and subtracting it rounds to an integer.
    pub shifter: u64,
    /// `ln 2` split: `kf · ln2_hi` is exact for every `k` the range reaches.
    pub ln2_hi: u64,
    pub ln2_lo: u64,
    /// Taylor degree `n` (`c_i = 1/i!` rounded to the width).
    pub degree: u32,
    /// The exponent bias and the significand bits (the `2^k` scales).
    pub bias: i64,
    pub mant: u32,
    /// The default face's bound: `|got − eˣ| ≤ rel·eˣ + abs` on `[lo, hi]`
    /// (ADR-0042 D4), as `rel = 2^−rel_log2`, `abs` the smallest subnormal.
    pub rel_log2: u32,
}

/// `E_32`.
pub const F32: ExpForm = ExpForm {
    bits: 32,
    hi: 0x42b1_7217,
    lo: 0xc2cf_f1b4,
    log2e: 0x3fb8_aa3b,
    shifter: 0x4b40_0000,
    ln2_hi: 0x3f31_7200,
    ln2_lo: 0x35bf_be8e,
    degree: 7,
    bias: 127,
    mant: 23,
    rel_log2: 21,
};

/// `E_64`.
pub const F64: ExpForm = ExpForm {
    bits: 64,
    hi: 0x4086_2e42_fefa_39ef,
    lo: 0xc087_4910_d52d_3051,
    log2e: 0x3ff7_1547_652b_82fe,
    shifter: 0x4338_0000_0000_0000,
    ln2_hi: 0x3fe6_2e42_fee0_0000,
    ln2_lo: 0x3dea_39ef_3579_3c76,
    degree: 13,
    bias: 1023,
    mant: 52,
    rel_log2: 50,
};

impl ExpForm {
    /// The form for a float width (`None` for any other).
    pub fn of_bits(bits: u32) -> Option<ExpForm> {
        match bits {
            32 => Some(F32),
            64 => Some(F64),
            _ => None,
        }
    }

    /// A constant's value, as an `f64` (exact for both widths).
    pub fn val(&self, c: u64) -> f64 {
        match self.bits {
            32 => f32::from_bits(c as u32) as f64,
            _ => f64::from_bits(c),
        }
    }

    /// The Taylor coefficients `c_0 … c_n` (`1/i!` rounded to the width),
    /// as bit patterns of the width.
    pub fn coefs(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.degree as usize + 1);
        let mut fact = 1f64;
        for i in 0..=self.degree {
            if i > 0 {
                fact *= i as f64;
            }
            let c = 1.0 / fact;
            out.push(match self.bits {
                32 => (c as f32).to_bits() as u64,
                _ => c.to_bits(),
            });
        }
        out
    }

    /// The default face's relative bound `2^−rel_log2`.
    pub fn rel(&self) -> f64 {
        2f64.powi(-(self.rel_log2 as i32))
    }

    /// The default face's absolute term: the smallest subnormal.
    pub fn abs(&self) -> f64 {
        match self.bits {
            32 => f32::from_bits(1) as f64,
            _ => f64::from_bits(1),
        }
    }

    /// The a-priori relative error of the reduced hardware form (ADR-0042
    /// D5): `2^r` approximated on `r ∈ [−½, ½]` with relative error
    /// `ex2_units · 2^−24`, after a Cody–Waite reduction whose rounded
    /// argument moves the result by at most `2^−24` relative; the two
    /// scalings are exact on normal results. `None` for `f64` (no hardware
    /// form).
    pub fn hw_rel(&self, ex2_units: u32) -> Option<f64> {
        (self.bits == 32).then(|| (ex2_units as f64 + 1.0) * 2f64.powi(-24))
    }

    /// The hardware form meets this width's bound.
    pub fn hw_admits(&self, ex2_units: u32) -> bool {
        self.hw_rel(ex2_units).is_some_and(|r| r <= self.rel())
    }
}

/// The oracle's `exp` on `f32` (`E_32`, module doc): ADR-0046's reference
/// definition [`crate::math::exp_def`] evaluated natively. Every step rounds
/// once — the exact face.
#[inline]
pub fn exp_f32(x: f32) -> f32 {
    crate::math::eval_f32(crate::math::MathFn::Exp, x)
}

/// The oracle's `exp` on `f64` (`E_64`, module doc).
#[inline]
pub fn exp_f64(x: f64) -> f64 {
    crate::math::eval_f64(crate::math::MathFn::Exp, x)
}

/// `got` is within the default face's bound of `eˣ` for width `form`
/// (ADR-0042 D4): the same special values as the oracle off `[lo, hi]`,
/// `|got − eˣ| ≤ rel·eˣ + abs` on it. `exact` is `eˣ` in a wider precision
/// (the caller's reference).
pub fn within_bound(form: &ExpForm, x: f64, got: f64, exact: f64) -> bool {
    if x.is_nan() {
        return got.is_nan();
    }
    if x > form.val(form.hi) {
        return got == f64::INFINITY;
    }
    if x < form.val(form.lo) {
        return got == 0.0;
    }
    (got - exact).abs() <= form.rel() * exact + form.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `eˣ` of an f32 argument to well below an f32 ulp: libm's `f64` exp.
    fn ref32(x: f32) -> f64 {
        (x as f64).exp()
    }

    /// A dense sweep (every 61st bit pattern of `[lo, hi]`, ~3.7·10⁷
    /// arguments) and the edge cases: the oracle's error in ulps of the
    /// exact result stays at the exhaustive measurement (1.221), far inside
    /// the default face's bound; the special values are the ADR's.
    #[test]
    fn exp_f32_dense_sweep_within_ulp_and_bound() {
        let (lo, hi) = (f32::from_bits(F32.lo as u32), f32::from_bits(F32.hi as u32));
        let mut worst = 0f64;
        let mut n = 0u64;
        let mut b: u32 = 0;
        loop {
            let x = f32::from_bits(b);
            if x >= lo && x <= hi {
                let (g, r) = (exp_f32(x) as f64, ref32(x));
                let ulp = if r < f32::MIN_POSITIVE as f64 {
                    F32.abs()
                } else {
                    2f64.powi(r.log2().floor() as i32 - 23)
                };
                worst = worst.max((g - r).abs() / ulp);
                assert!(within_bound(&F32, x as f64, g, r), "exp({x:e}) = {g:e}");
                n += 1;
            }
            match b.checked_add(61) {
                Some(nb) => b = nb,
                None => break,
            }
        }
        assert!(n > 30_000_000, "{n}");
        assert!(worst <= 1.25, "worst {worst} ulp");
        // The special values and the edges of the range.
        assert_eq!(exp_f32(0.0), 1.0);
        assert_eq!(exp_f32(-0.0), 1.0);
        assert_eq!(exp_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(exp_f32(f32::NEG_INFINITY), 0.0);
        assert!(exp_f32(f32::NAN).is_nan());
        assert!(exp_f32(hi).is_finite());
        assert_eq!(exp_f32(f32::from_bits(F32.hi as u32 + 1)), f32::INFINITY);
        assert_eq!(exp_f32(f32::from_bits(F32.lo as u32 + 1)), 0.0);
        assert_eq!(exp_f32(lo), f32::from_bits(1));
        assert_eq!(exp_f32(1e-30), 1.0);
        // Subnormal results round once.
        let x = -100.0f32;
        assert!(within_bound(&F32, x as f64, exp_f32(x) as f64, ref32(x)));
        assert!(exp_f32(x) > 0.0 && exp_f32(x) < f32::MIN_POSITIVE);
    }

    /// `E_64` against libm over a dense sample of `[lo, hi]` and `[−1, 1]`:
    /// within 1.5 ulp of libm's (itself within half an ulp), inside the bound.
    #[test]
    fn exp_f64_dense_sweep_within_ulp_and_bound() {
        let (lo, hi) = (f64::from_bits(F64.lo), f64::from_bits(F64.hi));
        let mut worst = 0f64;
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in 0..4_000_000u64 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            let x = if i % 2 == 0 {
                lo + (hi - lo) * u
            } else {
                2.0 * u - 1.0
            };
            let (g, r) = (exp_f64(x), x.exp());
            if r < f64::MIN_POSITIVE {
                assert!((g - r).abs() <= F64.abs() * 2.0, "{x:e}");
                continue;
            }
            let e = r.log2().floor() as i32 - 52;
            let ulp = 2f64.powi(e / 2) * 2f64.powi(e - e / 2);
            worst = worst.max((g - r).abs() / ulp);
        }
        assert!(worst <= 1.5, "worst {worst} ulp");
        assert_eq!(exp_f64(0.0), 1.0);
        assert_eq!(exp_f64(f64::INFINITY), f64::INFINITY);
        assert_eq!(exp_f64(f64::NEG_INFINITY), 0.0);
        assert!(exp_f64(f64::NAN).is_nan());
        assert!(exp_f64(hi).is_finite());
        assert_eq!(exp_f64(f64::from_bits(F64.hi + 1)), f64::INFINITY);
        assert_eq!(exp_f64(f64::from_bits(F64.lo + 1)), 0.0);
        assert!(exp_f64(lo) > 0.0);
    }

    /// The hardware form's admission is computed from the part's measured
    /// `ex2` error (units of `2^−24`): the RTX 5080's 2^−22.709 rounds up to 3
    /// units, `(3 + 1)·2^−24 = 2^−22 ≤ 2^−21`; a part above 7 units is not
    /// admitted; `f64` has no hardware form.
    #[test]
    fn hardware_form_admission_reads_the_measured_error() {
        assert!(F32.hw_admits(3));
        assert!(F32.hw_admits(7));
        assert!(!F32.hw_admits(8));
        assert!(!F64.hw_admits(0));
        assert_eq!(F32.coefs().len(), 8);
        assert_eq!(F64.coefs().len(), 14);
        assert_eq!(f32::from_bits(F32.shifter as u32), 12582912.0);
        assert_eq!(f64::from_bits(F64.shifter), 6755399441055744.0);
    }
}
