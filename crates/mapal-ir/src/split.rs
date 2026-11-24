//! **Split realizations of an f32 product on a tensor core** (f32-tensor-cores
//! F1; ADR-0044 candidate "split realizations of the default face"): the
//! a-priori coefficient of a sum of products whose f32 multiplicands are cut
//! into parts a matrix instruction multiplies exactly, against the default
//! face's own bound (ADR-0040 D2, `γ_{K+1}(2⁻²⁴)·M`), so the gate is unchanged.
//!
//! **The realization.** Per k-step (one instruction's `k`) and output tile, the
//! kept part products are issued smallest class first into a zeroed
//! accumulator, then added to the f32 register accumulator by one
//! round-to-nearest `FADD` (`j` k-steps a flush). The truncating accumulator
//! then rounds a chain of `j·P` instructions, never the whole `K`: its error
//! is paid once over the fold, against chunks whose magnitudes sum to `M`.
//!
//! **The datapath** ([`AccModel`], measured: `benches/local/mma_calib.cu`). One
//! instruction aligns its `n = k + 1` addends to the largest exponent, drops
//! each below `2^(e_max − 23 − g)` (`g` alignment guard bits), sums, normalizes
//! and truncates to 24 bits: `|d − exact| < c_g·u·M_i`, `c_g = 2·((k+1)·2⁻ᵍ + 1)`,
//! `M_i` the instruction's `|c| + Σ|a·b|`. The RTX 5080 keeps `g = 2` on every
//! warp format (TF32, bf16, f16, e4m3, e5m2).
//!
//! **The coefficient** (units of `u = 2⁻²⁴`, multiplying `M = |seed| + Σ|a_k·b_k|`):
//! `E(K, S, j) = e + C_j + ⌈K/(j·k)⌉ + S − 1` — the split error of a product
//! ([`SplitForm::e`]), the chain of one flush ([`SplitForm::chain`]), one
//! round-to-nearest flush per chunk, one combine per extra piece; admitted iff
//! `E·u ≤ γ_{K+1}(u)` ([`SplitForm::admits`]). D2's model (Higham's) holds
//! while every part stays a normal value: an operand below
//! [`SplitForm::exact_above`] (a part would round on the subnormal grid,
//! relative error up to `2⁻¹⁴` of the product for a TF32 `lo`) is sent, with
//! the non-finite ones, to the realization's guard — its warp recomputes the
//! tile by FFMA — so no absolute term enters the gate.

use crate::algo::gamma;

/// The f32 unit roundoff, `2⁻²⁴`.
pub const U: f64 = 1.0 / 16_777_216.0;

/// A split form: how the f32 operands are cut and which part products a
/// k-step issues.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SplitKind {
    /// `hi = rna_tf32(a)`, `lo = rna_tf32(a − hi)` (`a − hi` exact:
    /// Sterbenz); `lo·hi`, `hi·lo`, `hi·hi` on the TF32 instruction.
    Tf32x3,
    /// Three bf16 parts of each f32 operand (`rn(a)`, `rn` of the residual,
    /// what is left — exact above `2⁻¹¹⁰`); the six products with `i + j ≤ 2`
    /// on the bf16 instruction.
    Bf16x6,
    /// One operand stored narrow (its exact widening, ADR-0043), the other
    /// cut into three exact bf16 parts: three products, no split error.
    Bf16x3n,
    /// tensor-cores-2: one operand stored narrow, the other cut into two
    /// bf16 parts (`rn(a)`, `rn` of the residual): two products, split error
    /// `|a·b − (a₀ + a₁)·b| ≤ 2⁻¹⁶|a·b|` — admitted where the fold is long
    /// enough for D2 to cover it (`K ≥ 288` at `g = 2`).
    Bf16x2n,
    /// Both operands stored in one narrow format: one exact product a k-step.
    Narrow1,
}

/// **One split form's constants** (f32-tensor-cores §2.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitForm {
    pub kind: SplitKind,
    /// Parts each operand is cut into (a narrow operand: 1).
    pub parts_a: u32,
    pub parts_b: u32,
    /// The relative magnitude bound of each kept part product against
    /// `|a·b|`, in issue order (smallest class first): `P = classes.len()`
    /// instructions a k-step.
    pub classes: &'static [f64],
    /// The split error of one product: `|a·b − Σ kept| ≤ e·u·|a·b|` for
    /// operands whose parts stay normal.
    pub e: f64,
    /// The least `|x|` whose parts all stay normal values (or exact): `lo =
    /// rna(x − hi)` lies near `2⁻¹¹|x|`, normal for `|x| ≥ 2⁻¹¹⁵` (TF32); the
    /// third bf16 part near `2⁻¹⁶|x|`, for `|x| ≥ 2⁻¹¹⁰`. Below it a part
    /// rounds on the subnormal grid (half its least subnormal off: `2⁻¹³⁷`,
    /// `2⁻¹³⁴`); the realization's guard sends such an operand (not zero) to
    /// the FFMA path. 0: no part is rounded (a narrow operand).
    pub exact_above: f64,
}

/// TF32's correction class: `|lo| ≤ 2⁻¹¹(1 + 2⁻¹¹)|·|` — rounded up.
const RHO_TF32: f64 = 1.0 / 1024.0;
/// bf16's: `|a1| ≤ 2⁻⁸(1 + 2⁻⁸)|a|` — rounded up; `|a2| ≤ 2⁻¹⁶|a|`.
const RHO_BF16: f64 = 1.0 / 128.0;

const TF32X3_CLASSES: [f64; 3] = [RHO_TF32, RHO_TF32, 1.0];
const BF16X6_CLASSES: [f64; 6] = [
    RHO_BF16 * RHO_BF16,
    RHO_BF16 * RHO_BF16,
    RHO_BF16 * RHO_BF16,
    RHO_BF16,
    RHO_BF16,
    1.0,
];
const BF16X3N_CLASSES: [f64; 3] = [RHO_BF16 * RHO_BF16, RHO_BF16, 1.0];
const BF16X2N_CLASSES: [f64; 2] = [RHO_BF16, 1.0];
const NARROW1_CLASSES: [f64; 1] = [1.0];

impl SplitForm {
    pub fn of(kind: SplitKind) -> SplitForm {
        match kind {
            // `ab − (hh + hl + lh) = lo_a·lo_b + δ_a·b + δ_b·(a − δ_a)`,
            // `|lo| ≤ (1+2⁻¹¹)2⁻¹¹|·|`, `|δ| ≤ 2⁻²²|·|`: 3·2⁻²²(1 + 2⁻¹⁰)/u.
            SplitKind::Tf32x3 => SplitForm {
                kind,
                parts_a: 2,
                parts_b: 2,
                classes: &TF32X3_CLASSES,
                e: 12.0 * (1.0 + RHO_TF32),
                // TF32's least subnormal is 2⁻¹³⁶ (f32's exponent range, 10
                // fraction bits).
                exact_above: 2f64.powi(-115),
            },
            // The dropped (1,2), (2,1), (2,2): 2·2⁻⁸(1+2⁻⁸)·2⁻¹⁶ + 2⁻³².
            SplitKind::Bf16x6 => SplitForm {
                kind,
                parts_a: 3,
                parts_b: 3,
                classes: &BF16X6_CLASSES,
                e: (2.0 * 2f64.powi(-24) * (1.0 + 2f64.powi(-8)) + 2f64.powi(-32)) / U,
                exact_above: 2f64.powi(-110),
            },
            SplitKind::Bf16x3n => SplitForm {
                kind,
                parts_a: 3,
                parts_b: 1,
                classes: &BF16X3N_CLASSES,
                e: 0.0,
                exact_above: 2f64.powi(-110),
            },
            // `a − a₀ − a₁ = r − rn(r)`, `|r| ≤ 2⁻⁸|a|`: `|·| ≤ 2⁻⁸|r| ≤
            // 2⁻¹⁶|a|` while `r` is normal; on the subnormal grid at most
            // `2⁻¹³⁴`, within `2⁻¹⁶|a|` for `|a| ≥ 2⁻¹¹⁸` (the guard's
            // `2⁻¹¹⁰`, bf16x3n's, keeps it).
            SplitKind::Bf16x2n => SplitForm {
                kind,
                parts_a: 2,
                parts_b: 1,
                classes: &BF16X2N_CLASSES,
                e: 2f64.powi(-16) / U,
                exact_above: 2f64.powi(-110),
            },
            SplitKind::Narrow1 => SplitForm {
                kind,
                parts_a: 1,
                parts_b: 1,
                classes: &NARROW1_CLASSES,
                e: 0.0,
                exact_above: 0.0,
            },
        }
    }

    /// `P`: the part products a k-step issues.
    pub fn products(&self) -> u64 {
        self.classes.len() as u64
    }

    /// The chain term of one flush of `j` k-steps (units of `u·M_chunk`):
    /// each instruction rounds at most `c_g·u` of its own `|c| + Σ|a·b|`.
    /// One k-step (`j = 1`): instruction `p` carries the classes before it
    /// and its own, `Σ_p Σ_{q ≤ p} ρ_q`. More: every one of the `j·P`
    /// instructions carries at most the whole chunk, `j·P·Σ_q ρ_q`.
    pub fn chain(&self, j: u64, acc: AccModel, k: u64) -> f64 {
        let c = acc.c(k);
        let total: f64 = self.classes.iter().sum();
        if j <= 1 {
            let mut run = 0.0;
            let mut sum = 0.0;
            for r in self.classes {
                run += r;
                sum += run;
            }
            c * sum
        } else {
            c * (j * self.products()) as f64 * total
        }
    }

    /// **The realization's a-priori coefficient** `E(K, S, j)` (units of
    /// `u`, multiplying `M`): a fold of `kk` steps over an instruction of
    /// depth `k`, cut into `s` pieces combined by round-to-nearest adds,
    /// flushed every `j` k-steps. The `(1 + δ)` factors of the flush and
    /// combine chains are kept to second order by [`gamma`].
    pub fn coef(&self, kk: u64, s: u64, j: u64, k: u64, acc: AccModel) -> f64 {
        let j = j.max(1);
        let flushes = kk.div_ceil(j * k.max(1));
        let adds = flushes + s.max(1) - 1;
        // The split error and the chain of each chunk, carried through the
        // round-to-nearest adds that follow (`(1 + γ_adds)`), plus the adds.
        (self.e + self.chain(j, acc, k)) * (1.0 + gamma(adds, U)) + gamma(adds, U) / U
    }

    /// **Admission** (D2 verbatim): the largest flush depth `j` (a power of
    /// two dividing `kk / k`, at most `j_cap`) at which [`Self::coef`] meets
    /// `γ_{K+1}(u)`; `None` where even `j = 1` misses it.
    pub fn admits(&self, kk: u64, s: u64, k: u64, acc: AccModel, j_cap: u64) -> Option<u64> {
        let budget = gamma(kk + 1, U) / U;
        let steps = kk / k.max(1);
        if steps == 0 || !kk.is_multiple_of(k.max(1)) {
            return None;
        }
        let mut best = None;
        let mut j = 1;
        while j <= j_cap.max(1) && j <= steps && steps.is_multiple_of(j) {
            if self.coef(kk, s, j, k, acc) <= budget {
                best = Some(j);
            } else if best.is_some() {
                break;
            }
            j *= 2;
        }
        best
    }

    /// The smallest fold the form meets D2 at (`s` pieces, `j = 1`), a
    /// multiple of `k`; `None` past `limit`.
    pub fn k_min(&self, s: u64, k: u64, acc: AccModel, limit: u64) -> Option<u64> {
        (1..=limit / k.max(1))
            .map(|n| n * k)
            .find(|&kk| self.admits(kk, s, k, acc, 1).is_some())
    }
}

/// **The accumulator datapath of one matrix instruction** (f32-tensor-cores
/// §2.2): aligned truncation with `g` guard bits below the largest addend's
/// last bit, then a truncating normalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccModel {
    pub g: u32,
}

impl AccModel {
    /// `c_g`: one instruction of depth `k` is within `c_g·u·M_i` of its
    /// exact sum (`M_i = |c| + Σ|a·b|`).
    pub fn c(&self, k: u64) -> f64 {
        2.0 * ((k + 1) as f64 * 2f64.powi(-(self.g as i32)) + 1.0)
    }

    /// Whether a chain of `K/k` instructions into one accumulator (the
    /// member realization, ADR-0041 D3's `γ_{K+1}(2⁻²³)`) is covered:
    /// `((k+1)·2⁻ᵍ + 1)/k ≤ 1` — every `g ≥ 1` at `k ≥ 2`.
    pub fn member_sound(&self, k: u64) -> bool {
        ((k + 1) as f64 * 2f64.powi(-(self.g as i32)) + 1.0) <= k as f64
    }
}

/// `x`'s TF32 part, `cvt.rna.tf32.f32` (round to nearest, ties away, 10
/// fraction bits) of a finite `x`; the truncation where that rounds a finite
/// value to infinity (f32-tensor-cores §2.5's overflow select).
pub fn tf32_hi(x: f32) -> f32 {
    let b = x.to_bits();
    if (b & 0x7f80_0000) == 0x7f80_0000 {
        return x;
    }
    let r = f32::from_bits(b.wrapping_add(0x1000) & 0xffff_e000);
    if r.is_infinite() {
        f32::from_bits(b & 0xffff_e000)
    } else {
        r
    }
}

/// `x`'s TF32 parts `(hi, lo)`: `lo = rna(x − hi)`; `lo = 0` where `x` is
/// not finite.
pub fn tf32_parts(x: f32) -> (f32, f32) {
    let hi = tf32_hi(x);
    if !x.is_finite() {
        return (hi, 0.0);
    }
    (hi, tf32_hi(x - hi))
}

/// `x`'s three bf16 parts (each a bf16 value as an f32): `rn(x)`, `rn` of
/// the residual, `rn` of what is left.
pub fn bf16_parts(x: f32) -> [f32; 3] {
    let rn = |v: f32| {
        let bits = crate::narrow::round(crate::ty::NarrowFloat::Bf16, f64::from(v));
        f32::from_bits(u32::from(bits) << 16)
    };
    let p0 = rn(x);
    let r = x - p0;
    let p1 = rn(r);
    let p2 = rn(r - p1);
    [p0, p1, p2]
}

#[cfg(test)]
mod tests {
    use super::*;

    const G0: AccModel = AccModel { g: 0 };
    const G1: AccModel = AccModel { g: 1 };
    const G2: AccModel = AccModel { g: 2 };
    const G3: AccModel = AccModel { g: 3 };

    /// The design's K_min table (§2.3), and the measured `g = 2`'s.
    #[test]
    fn k_min_table_is_pinned() {
        let f = |k, kk| SplitForm::of(k).k_min(1, kk, G0, 4096);
        assert_eq!(f(SplitKind::Tf32x3, 8), Some(40));
        assert_eq!(f(SplitKind::Bf16x6, 16), Some(48));
        assert_eq!(f(SplitKind::Bf16x3n, 16), Some(48));
        assert_eq!(f(SplitKind::Narrow1, 16), Some(48));
        let at = |k, kk, g| SplitForm::of(k).k_min(1, kk, g, 4096);
        assert_eq!(at(SplitKind::Tf32x3, 8, G1), Some(32));
        assert_eq!(at(SplitKind::Tf32x3, 8, G3), Some(24));
        assert_eq!(at(SplitKind::Bf16x6, 16, G1), Some(32));
        assert_eq!(at(SplitKind::Bf16x6, 16, G3), Some(16));
        // The RTX 5080's measured datapath.
        assert_eq!(at(SplitKind::Tf32x3, 8, G2), Some(24));
        assert_eq!(at(SplitKind::Bf16x6, 16, G2), Some(16));
        assert_eq!(at(SplitKind::Narrow1, 16, G2), Some(16));
        // tensor-cores-2: the two-part cut's `2⁻¹⁶` split error needs a fold
        // whose `γ_{K+1}` covers 256 units.
        assert_eq!(at(SplitKind::Bf16x2n, 16, G2), Some(288));
        assert_eq!(at(SplitKind::Bf16x2n, 16, G0), Some(320));
    }

    /// tensor-cores-2: the two-part cut of an f32 is within `2⁻¹⁶|x|` on
    /// every binade edge at and above `2⁻¹¹⁰` and on pseudo-random values.
    #[test]
    fn bf16_two_parts_within_two_to_minus_16() {
        let mut xs = Vec::new();
        for e in -110..=127 {
            let b = 2f32.powi(e);
            xs.extend([b, b * (1.0 + f32::EPSILON), b * (2.0 - f32::EPSILON)]);
            // The guard takes what rounds to infinity (`2¹²⁸ − 2¹¹¹` and up).
            xs.retain(|x| x.to_bits() < 0x7f7f_8000);
        }
        let mut st = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..20_000 {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            let x = f32::from_bits((st as u32 & 0x007f_ffff) | 0x3f80_0000);
            xs.push(x * 2f32.powi((st >> 40) as i32 % 200 - 100));
        }
        for x in xs {
            let [p0, p1, _] = bf16_parts(x);
            let err = (f64::from(x) - f64::from(p0) - f64::from(p1)).abs();
            assert!(err <= 2f64.powi(-16) * f64::from(x).abs(), "{x:e}");
        }
    }

    /// `E` at K = 4096 (g = 0), the design's last column, within rounding.
    #[test]
    fn coef_at_4096_matches_the_design() {
        let e = |k, kk| SplitForm::of(k).coef(4096, 1, 1, kk, G0);
        assert!(
            (e(SplitKind::Tf32x3, 8) - 544.0).abs() < 1.0,
            "{}",
            e(SplitKind::Tf32x3, 8)
        );
        assert!(
            (e(SplitKind::Bf16x6, 16) - 295.0).abs() < 1.5,
            "{}",
            e(SplitKind::Bf16x6, 16)
        );
        assert!(
            (e(SplitKind::Narrow1, 16) - 292.0).abs() < 1.0,
            "{}",
            e(SplitKind::Narrow1, 16)
        );
    }

    /// `coef` is monotone in `K` and `S`; a deeper flush trades the chain
    /// (`j·P` instructions) for fewer flushes; admission respects the budget.
    #[test]
    fn coef_is_monotone() {
        for kind in [
            SplitKind::Tf32x3,
            SplitKind::Bf16x6,
            SplitKind::Bf16x3n,
            SplitKind::Narrow1,
        ] {
            let f = SplitForm::of(kind);
            let k = if kind == SplitKind::Tf32x3 { 8 } else { 16 };
            for g in [G0, G1, G2, G3] {
                let mut last = 0.0;
                for n in 1..64 {
                    let c = f.coef(n * k, 1, 1, k, g);
                    assert!(c >= last, "{kind:?} K {}", n * k);
                    last = c;
                    assert!(f.coef(n * k, 2, 1, k, g) > c);
                    assert!(f.chain(2, g, k) > f.chain(1, g, k));
                }
            }
            let j = f.admits(4096, 1, k, G2, 64).expect("admitted at 4096");
            assert!(j >= 2, "{kind:?}: a long fold flushes less often ({j})");
            assert!(f.coef(4096, 1, j, k, G2) <= gamma(4097, U) / U);
        }
    }

    /// A truncating chain of `K/k` instructions into one accumulator is
    /// covered by `γ_{K+1}(2⁻²³)` iff `g ≥ 1` (§2.2): the measured `g = 2`
    /// keeps ADR-0041 D3's member bound sound.
    #[test]
    fn member_bound_needs_a_guard_bit() {
        for k in [8, 16, 32] {
            assert!(!G0.member_sound(k));
            assert!(G1.member_sound(k) && G2.member_sound(k));
        }
    }

    /// The bf16 three-way split is exact (every binade edge at and above
    /// 2⁻¹¹⁰ and pseudo-random values); the TF32 residual is within 2⁻²²|x|.
    #[test]
    fn splits_are_exact_or_bounded() {
        let mut s: u32 = 0x1234_5678;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let mut xs = Vec::new();
        for e in -110..=127 {
            let b = 2f32.powi(e);
            xs.extend([
                b,
                -b,
                f32::from_bits(b.to_bits() - 1),
                f32::from_bits(b.to_bits() + 1),
            ]);
        }
        for _ in 0..(1 << 20) {
            let e = (next() % 237) as i32 - 110;
            let m = next() & 0x7f_ffff;
            let sign = (next() & 1) << 31;
            xs.push(f32::from_bits(sign | (((e + 127) as u32) << 23) | m));
        }
        for &x in &xs {
            if !x.is_finite() || x.abs() < 2f32.powi(-110) {
                continue;
            }
            let [p0, p1, p2] = bf16_parts(x);
            assert_eq!(
                f64::from(p0) + f64::from(p1) + f64::from(p2),
                f64::from(x),
                "{x:e}"
            );
            let (hi, lo) = tf32_parts(x);
            let r = (f64::from(x) - f64::from(hi) - f64::from(lo)).abs();
            assert!(r <= 2f64.powi(-22) * f64::from(x).abs(), "{x:e}: {r:e}");
            assert_eq!(hi.to_bits() & 0x1fff, 0);
            assert_eq!(lo.to_bits() & 0x1fff, 0);
        }
        // §2.5: the largest finite values keep a finite `hi`.
        let (hi, lo) = tf32_parts(f32::MAX);
        assert!(hi.is_finite() && lo.is_finite());
        let r = (f64::from(f32::MAX) - f64::from(hi) - f64::from(lo)).abs();
        assert!(r <= 2f64.powi(-22) * f64::from(f32::MAX));
    }

    /// The tf32x3 split error of one product stays within `e·u·|ab|` over
    /// random operands (the coefficient's first term).
    #[test]
    fn tf32x3_product_error_within_e() {
        let f = SplitForm::of(SplitKind::Tf32x3);
        let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            s ^= s << 7;
            s ^= s >> 9;
            s
        };
        let mut worst: f64 = 0.0;
        for _ in 0..(1 << 18) {
            let r = |v: u64| {
                f32::from_bits(0x3f80_0000 | (v as u32 & 0x7f_ffff))
                    * 2f32.powi((v >> 40) as i32 % 20 - 10)
            };
            let (a, b) = (r(next()), r(next()));
            let ((ah, al), (bh, bl)) = (tf32_parts(a), tf32_parts(b));
            let kept = f64::from(al) * f64::from(bh)
                + f64::from(ah) * f64::from(bl)
                + f64::from(ah) * f64::from(bh);
            let ex = f64::from(a) * f64::from(b);
            worst = worst.max((kept - ex).abs() / (U * ex.abs()));
        }
        assert!(worst <= f.e, "{worst}");
        assert!(worst > 1.0, "the probe reaches the split error ({worst})");
    }
}
