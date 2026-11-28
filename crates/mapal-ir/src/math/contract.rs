//! Contracts: each builtin's accuracy contract, special values and test arguments.

use super::*;

// ---------------------------------------------------------------------------
// Contracts.
// ---------------------------------------------------------------------------

/// A builtin's default-face bound at one width (ADR-0046 D3): `|got − f(x)| ≤
/// rel·|f(x)| + abs` for every `x` without a special value. The exact faces
/// are the reference's bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contract {
    pub f: MathFn,
    pub bits: u32,
    pub rel: f64,
    pub abs: f64,
}

pub(super) fn tiny(bits: u32) -> f64 {
    if bits == 64 {
        f64::from_bits(1)
    } else {
        f64::from(f32::from_bits(1))
    }
}

/// The contract of `f` at width `bits` (`None`: not a float width).
pub fn contract(f: MathFn, bits: u32) -> Option<Contract> {
    let wide = match bits {
        32 => false,
        64 => true,
        _ => return None,
    };
    let (rel_log2, abs) = match f {
        MathFn::Exp => {
            let form = ExpForm::of_bits(bits)?;
            return Some(Contract {
                f,
                bits,
                rel: form.rel(),
                abs: form.abs(),
            });
        }
        MathFn::Sqrt => (if wide { 51 } else { 22 }, 0.0),
        MathFn::Sin | MathFn::Cos => (if wide { 50 } else { 21 }, tiny(bits)),
    };
    Some(Contract {
        f,
        bits,
        rel: 2f64.powi(-rel_log2),
        abs,
    })
}

/// The value the contract demands exactly at `x` (on every face), or `None`
/// where the bound applies. A NaN demands any NaN.
pub fn special(f: MathFn, bits: u32, x: f64) -> Option<f64> {
    if x.is_nan() {
        return Some(f64::NAN);
    }
    match f {
        MathFn::Exp => {
            let form = ExpForm::of_bits(bits)?;
            if x > form.val(form.hi) {
                Some(f64::INFINITY)
            } else if x < form.val(form.lo) {
                Some(0.0)
            } else {
                None
            }
        }
        MathFn::Sqrt => {
            if x == 0.0 || x == f64::INFINITY {
                Some(x)
            } else if x < 0.0 {
                Some(f64::NAN)
            } else {
                None
            }
        }
        MathFn::Sin | MathFn::Cos => {
            if x.is_infinite() {
                Some(f64::NAN)
            } else if x == 0.0 {
                Some(if f == MathFn::Sin { x } else { 1.0 })
            } else {
                None
            }
        }
    }
}

/// `6381956970095103·2^797` (Kahan–McDonald's worst cancellation for `f64`): its sine and
/// cosine, correctly rounded (mpmath at 2,000 bits), as `(fn, x bits, value bits)`.
const HARD: [(MathFn, u64, u64); 2] = [
    (MathFn::Sin, 0x7506_ac5b_262c_a1ff, 0x3ff0_0000_0000_0000),
    (MathFn::Cos, 0x7506_ac5b_262c_a1ff, 0xbc21_4ae7_2e6b_a22f),
];

/// `f(x)` in `f64` (libm): the truth a width-32 result is checked against
/// (within `2⁻⁵²` relative of the real value; a width-64 check adds that).
pub fn truth(f: MathFn, x: f64) -> f64 {
    // At the hardest argument the platform's libm is not the truth: one whose argument
    // reduction loses the cancellation is off far beyond the contract (CI's Ubuntu runner
    // failed here while glibc 2.44 rounds correctly). The correctly rounded values instead.
    if let Some(&(_, _, v)) = HARD.iter().find(|h| h.0 == f && h.1 == x.to_bits()) {
        return f64::from_bits(v);
    }
    match f {
        MathFn::Exp => x.exp(),
        MathFn::Sqrt => x.sqrt(),
        MathFn::Sin => x.sin(),
        MathFn::Cos => x.cos(),
    }
}

/// `got` meets `f`'s contract at `x` (width `bits`) on the default face:
/// the special value where one is demanded (a zero's sign included), else
/// within the bound of `exact` (`slack`: the reference's own relative error,
/// `0` for an `f64`-checked f32).
pub fn within(f: MathFn, bits: u32, x: f64, got: f64, exact: f64, slack: f64) -> bool {
    let Some(c) = contract(f, bits) else {
        return false;
    };
    if let Some(s) = special(f, bits, x) {
        return if s.is_nan() {
            got.is_nan()
        } else {
            got == s && got.is_sign_negative() == s.is_sign_negative()
        };
    }
    (got - exact).abs() <= (c.rel + slack) * exact.abs() + c.abs
}

/// The contract's test domain at width `bits`: the special values, the
/// domain's edges, subnormals, huge `sin` / `cos` arguments (the f32 / f64
/// worst cancellations against multiples of `π/2`, the largest finite) and
/// seeded samples over the whole range — what a [`TargetDef`]'s declaration
/// is re-measured on.
pub fn test_args(f: MathFn, bits: u32) -> Vec<f64> {
    let wide = bits == 64;
    let of = |b: u64| -> f64 {
        if wide {
            f64::from_bits(b)
        } else {
            f64::from(f32::from_bits(b as u32))
        }
    };
    let mut v: Vec<f64> = vec![
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1.0,
        -1.0,
        0.5,
        2.0,
        of(1),
        of(3),
        if wide {
            f64::MIN_POSITIVE
        } else {
            f64::from(f32::MIN_POSITIVE)
        },
        if wide { f64::MAX } else { f64::from(f32::MAX) },
        if wide {
            -f64::MAX
        } else {
            -f64::from(f32::MAX)
        },
    ];
    match f {
        MathFn::Exp => {
            if let Some(form) = ExpForm::of_bits(bits) {
                for b in [form.hi, form.lo] {
                    v.push(form.val(b));
                    v.push(of(b + 1));
                    v.push(of(b - 1));
                }
            }
        }
        MathFn::Sin | MathFn::Cos => {
            // Near multiples of π/2 (k up to 2^20) and the known hard cases.
            for k in [1u64, 2, 3, 4, 7, 100, 355, 1 << 10, 1 << 20] {
                let t = k as f64 * std::f64::consts::FRAC_PI_2;
                for d in [-2i64, -1, 0, 1, 2] {
                    let b = if wide {
                        (t.to_bits() as i64 + d) as u64
                    } else {
                        ((t as f32).to_bits() as i64 + d) as u64
                    };
                    v.push(of(b));
                    v.push(-of(b));
                }
            }
            // The worst cancellations: f64 6381956970095103·2^797 (Kahan–
            // McDonald), f32 16367173·2^72; and large round numbers.
            if wide {
                v.push(6381956970095103.0 * 2f64.powi(797));
                v.push(1e22);
                v.push(1e300);
            } else {
                v.push(16367173.0 * 2f64.powi(72));
                v.push(f64::from(1e22f32));
                v.push(f64::from(1e30f32));
            }
        }
        MathFn::Sqrt => {
            v.push(4.0);
            v.push(0.25);
            v.push(of(0x7ff));
        }
    }
    // Seeded samples: uniform bit patterns (every exponent) and `[−4, 4]`.
    let mut s: u64 = 0x2545_f491_4f6c_dd1d ^ (f as u64) ^ u64::from(bits);
    for i in 0..3000 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let x = if i % 2 == 0 {
            of(if wide { s } else { s >> 32 })
        } else {
            (s >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 4.0
        };
        let x = if wide { x } else { f64::from(x as f32) };
        if !x.is_nan() {
            v.push(x);
        }
    }
    v
}
