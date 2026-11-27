//! Native evaluation: `Native<B>` runs a definition on plain integers (the interpreter, the constant folder).

use super::*;

// ---------------------------------------------------------------------------
// Native evaluation.
// ---------------------------------------------------------------------------

/// The definitions computed: every value a `u64` payload (a float's bits, an
/// integer in its low `w` bits, a bool 0 / 1). Monomorphized and inlined, it
/// is the reference at native speed — the interpreter's and the constant
/// folder's.
#[derive(Clone, Copy, Debug, Default)]
pub struct Native<const B: u32>;

pub(super) const M32: u64 = 0xffff_ffff;

#[inline(always)]
pub(super) fn mask(w: W, v: u64) -> u64 {
    match w {
        W::I32 => v & M32,
        W::I64 => v,
    }
}

impl<const B: u32> Native<B> {
    #[inline(always)]
    pub(super) fn fop(
        a: u64,
        b: u64,
        f32op: impl Fn(f32, f32) -> f32,
        f64op: impl Fn(f64, f64) -> f64,
    ) -> u64 {
        if B == 32 {
            f32op(f32::from_bits(a as u32), f32::from_bits(b as u32)).to_bits() as u64
        } else {
            f64op(f64::from_bits(a), f64::from_bits(b)).to_bits()
        }
    }

    #[inline(always)]
    pub(super) fn fval(a: u64) -> f64 {
        if B == 32 {
            f32::from_bits(a as u32) as f64
        } else {
            f64::from_bits(a)
        }
    }

    #[inline(always)]
    pub(super) fn of_f64(v: f64) -> u64 {
        if B == 32 {
            (v as f32).to_bits() as u64
        } else {
            v.to_bits()
        }
    }
}

impl<const B: u32> Arith for Native<B> {
    type V = u64;

    #[inline(always)]
    fn fbits(&self) -> u32 {
        B
    }
    #[inline(always)]
    fn contract(&mut self, _on: bool) {}
    #[inline(always)]
    fn name(&mut self, v: u64, _n: &str) -> u64 {
        v
    }
    #[inline(always)]
    fn name_ix(&mut self, v: u64, _p: &str, _i: usize) -> u64 {
        v
    }
    #[inline(always)]
    fn fc(&mut self, bits: u64) -> u64 {
        if B == 32 { bits & M32 } else { bits }
    }
    #[inline(always)]
    fn ic(&mut self, w: W, v: u64) -> u64 {
        mask(w, v)
    }
    #[inline(always)]
    fn f2(&mut self, op: F2, a: u64, b: u64) -> u64 {
        match op {
            F2::Add => Self::fop(a, b, |x, y| x + y, |x, y| x + y),
            F2::Sub => Self::fop(a, b, |x, y| x - y, |x, y| x - y),
            F2::Mul => Self::fop(a, b, |x, y| x * y, |x, y| x * y),
        }
    }
    #[inline(always)]
    fn fneg(&mut self, a: u64) -> u64 {
        if B == 32 {
            a ^ 0x8000_0000
        } else {
            a ^ (1 << 63)
        }
    }
    #[inline(always)]
    fn fma(&mut self, a: u64, b: u64, c: u64) -> u64 {
        if B == 32 {
            let f = |v: u64| f32::from_bits(v as u32);
            f(a).mul_add(f(b), f(c)).to_bits() as u64
        } else {
            let f = f64::from_bits;
            f(a).mul_add(f(b), f(c)).to_bits()
        }
    }
    #[inline(always)]
    fn fcmp(&mut self, p: FPred, a: u64, b: u64) -> u64 {
        let (x, y) = (Self::fval(a), Self::fval(b));
        u64::from(match p {
            FPred::Ogt => x > y,
            FPred::Olt => x < y,
            FPred::Oeq => x == y,
            FPred::Uno => x.is_nan() || y.is_nan(),
        })
    }
    #[inline(always)]
    fn i2(&mut self, w: W, op: I2, a: u64, b: u64) -> u64 {
        let n = w.bits();
        let r = match op {
            I2::Add => a.wrapping_add(b),
            I2::Sub => a.wrapping_sub(b),
            I2::Mul => a.wrapping_mul(b),
            I2::MulHi => match w {
                W::I32 => ((a & M32) * (b & M32)) >> 32,
                W::I64 => ((u128::from(a) * u128::from(b)) >> 64) as u64,
            },
            I2::And => a & b,
            I2::Or => a | b,
            I2::Xor => a ^ b,
            I2::Shl => {
                debug_assert!(b < u64::from(n), "shift {b} of an i{n}");
                a << (b & 63)
            }
            I2::LShr => {
                debug_assert!(b < u64::from(n), "shift {b} of an i{n}");
                mask(w, a) >> (b & 63)
            }
            I2::AShr => {
                debug_assert!(b < u64::from(n), "shift {b} of an i{n}");
                match w {
                    W::I32 => ((a as u32 as i32) >> (b & 31)) as u32 as u64,
                    W::I64 => ((a as i64) >> (b & 63)) as u64,
                }
            }
        };
        mask(w, r)
    }
    #[inline(always)]
    fn clz(&mut self, w: W, a: u64) -> u64 {
        match w {
            W::I32 => u64::from((a as u32).leading_zeros()),
            W::I64 => u64::from(a.leading_zeros()),
        }
    }
    #[inline(always)]
    fn icmp(&mut self, w: W, p: IPred, a: u64, b: u64) -> u64 {
        let (a, b) = (mask(w, a), mask(w, b));
        let s = |v: u64| match w {
            W::I32 => i64::from(v as u32 as i32),
            W::I64 => v as i64,
        };
        u64::from(match p {
            IPred::Eq => a == b,
            IPred::Ne => a != b,
            IPred::Ult => a < b,
            IPred::Ugt => a > b,
            IPred::Slt => s(a) < s(b),
            IPred::Sgt => s(a) > s(b),
        })
    }
    #[inline(always)]
    fn b2(&mut self, op: B2, a: u64, b: u64) -> u64 {
        match op {
            B2::And => a & b,
            B2::Or => a | b,
            B2::Xor => a ^ b,
        }
    }
    #[inline(always)]
    fn sel(&mut self, c: u64, a: u64, b: u64) -> u64 {
        if c != 0 { a } else { b }
    }
    #[inline(always)]
    fn cvt(&mut self, c: Cvt, a: u64) -> u64 {
        match c {
            Cvt::Bits | Cvt::FromBits => a,
            Cvt::FToSi(w) => {
                let v = if B == 32 {
                    f32::from_bits(a as u32) as i64
                } else {
                    f64::from_bits(a) as i64
                };
                mask(w, v as u64)
            }
            Cvt::SiToF(w) => {
                let v = match w {
                    W::I32 => i64::from(a as u32 as i32),
                    W::I64 => a as i64,
                };
                if B == 32 {
                    (v as f32).to_bits() as u64
                } else {
                    (v as f64).to_bits()
                }
            }
            Cvt::UiToF(w) => {
                let v = mask(w, a);
                if B == 32 {
                    (v as f32).to_bits() as u64
                } else {
                    (v as f64).to_bits()
                }
            }
            Cvt::ZExt | Cvt::Trunc => a & M32,
            Cvt::BoolToI(_) => a & 1,
        }
    }
    /// An ideal unit (the function correctly rounded from `f64`): the model a
    /// host-side check of a form evaluates; the device's own error is the
    /// unit's declared one.
    fn unit(&mut self, u: &Unit, a: u64) -> u64 {
        let x = Self::fval(a);
        Self::of_f64(match u.op {
            UnitOp::Ex2 => x.exp2(),
            UnitOp::Exp => x.exp(),
            UnitOp::Sqrt => x.sqrt(),
            UnitOp::Sin => x.sin(),
            UnitOp::Cos => x.cos(),
        })
    }
}
