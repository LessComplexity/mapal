//! Reference definitions of `exp` and `sqrt`, the special-value wrap and the `reference` / `eval_*` entry points.

use super::*;

// ---------------------------------------------------------------------------
// Reference definitions.
// ---------------------------------------------------------------------------

/// `exp`'s reference: ADR-0042's `E_b`, the same operations in the same
/// order (its printed text is ADR-0042's byte for byte).
pub fn exp_def<A: Arith>(a: &mut A, x: A::V) -> A::V {
    exp_body(a, x, None)
}

/// `E_b`, its middle either the reduction and the Taylor polynomial or (the
/// hardware form, ADR-0042 D5) the reduction as two FMAs and `ex2` of
/// `r·log2e`.
pub(super) fn exp_body<A: Arith>(a: &mut A, x: A::V, ex2: Option<&Unit>) -> A::V {
    let form = ExpForm::of_bits(a.fbits()).expect("exp of an f32 or f64");
    let w = if form.bits == 64 { W::I64 } else { W::I32 };
    let (hi, lo, zero) = (a.fc(form.hi), a.fc(form.lo), a.fc(0));
    let ch = a.fcmp(FPred::Ogt, x, hi);
    let ch = a.name(ch, "ch");
    let x0 = a.sel(ch, hi, x);
    let x0 = a.name(x0, "x0");
    let cl0 = a.fcmp(FPred::Olt, x0, lo);
    let cl0 = a.name(cl0, "cl0");
    let x1 = a.sel(cl0, lo, x0);
    let x1 = a.name(x1, "x1");
    let cn0 = a.fcmp(FPred::Uno, x1, zero);
    let cn0 = a.name(cn0, "cn0");
    let xc = a.sel(cn0, zero, x1);
    let xc = a.name(xc, "xc");
    let (log2e, sh) = (a.fc(form.log2e), a.fc(form.shifter));
    let (lhi, llo) = (a.fc(form.ln2_hi), a.fc(form.ln2_lo));
    let (kf, p) = match ex2 {
        Some(u) => {
            let t = a.fma(xc, log2e, sh);
            a.name(t, "t");
            let kf = a.fsub(t, sh);
            a.name(kf, "kf");
            let nk = a.fneg(kf);
            a.name(nk, "nk");
            let r0 = a.fma(nk, lhi, xc);
            a.name(r0, "r0");
            let r = a.fma(nk, llo, r0);
            a.name(r, "r");
            let av = a.fmul(r, log2e);
            a.name(av, "a");
            let p0 = a.unit(u, av);
            a.name(p0, "p0");
            (kf, p0)
        }
        None => {
            let t0 = a.fmul(xc, log2e);
            a.name(t0, "t0");
            let t = a.fadd(t0, sh);
            a.name(t, "t");
            let kf = a.fsub(t, sh);
            a.name(kf, "kf");
            a.contract(true);
            let h = a.fmul(kf, lhi);
            a.name(h, "h");
            let r0 = a.fsub(xc, h);
            a.name(r0, "r0");
            let lw = a.fmul(kf, llo);
            a.name(lw, "lw");
            let r = a.fsub(r0, lw);
            a.name(r, "r");
            let cs = form.coefs();
            let n = form.degree as usize;
            let mut p = a.fc(cs[n]);
            for i in (0..n).rev() {
                let m = a.fmul(p, r);
                a.name_ix(m, "m", i);
                let c = a.fc(cs[i]);
                p = a.fadd(m, c);
                a.name_ix(p, "p", i);
            }
            a.contract(false);
            (kf, p)
        }
    };
    let k = a.cvt(Cvt::FToSi(w), kf);
    a.name(k, "k");
    let one = a.ic(w, 1);
    let k1 = a.i2(w, I2::AShr, k, one);
    a.name(k1, "k1");
    let k2 = a.i2(w, I2::Sub, k, k1);
    a.name(k2, "k2");
    let (bias, mant) = (a.ic(w, form.bias as u64), a.ic(w, u64::from(form.mant)));
    let mut s = [k1, k2];
    for (i, k) in [k1, k2].into_iter().enumerate() {
        let b = a.i2(w, I2::Add, k, bias);
        a.name_ix(b, "bk", i + 1);
        let e = a.i2(w, I2::Shl, b, mant);
        a.name_ix(e, "ek", i + 1);
        s[i] = a.cvt(Cvt::FromBits, e);
        a.name_ix(s[i], "sk", i + 1);
    }
    let y0 = a.fmul(p, s[0]);
    a.name(y0, "y0");
    let y1 = a.fmul(y0, s[1]);
    a.name(y1, "y1");
    let inf = a.fc(if form.bits == 64 {
        f64::INFINITY.to_bits()
    } else {
        u64::from(f32::INFINITY.to_bits())
    });
    let y2 = a.sel(ch, inf, y1);
    a.name(y2, "y2");
    let cl = a.fcmp(FPred::Olt, x, lo);
    a.name(cl, "cl");
    let y3 = a.sel(cl, zero, y2);
    a.name(y3, "y3");
    let cn = a.fcmp(FPred::Uno, x, zero);
    a.name(cn, "cn");
    let y = a.sel(cn, x, y3);
    a.name(y, "y")
}

/// The float of the width's canonical quiet NaN.
pub(super) fn nan_bits(bits: u32) -> u64 {
    if bits == 64 {
        0x7ff8_0000_0000_0000
    } else {
        0x7fc0_0000
    }
}

pub(super) fn inf_bits(bits: u32) -> u64 {
    if bits == 64 {
        f64::INFINITY.to_bits()
    } else {
        u64::from(f32::INFINITY.to_bits())
    }
}

/// `sqrt`'s plain-operation definition: the correctly rounded `√x` (IEEE
/// `squareRoot`) by the restoring digit recurrence on the significand. The
/// significand `m` (subnormals normalized by `clz`) is shifted to `m2` with an
/// even exponent and `m2·2^(2Z)` square-rooted two bits a step (a 64-bit
/// remainder), giving one bit past the significand; that bit rounds (a
/// square root is never a tie).
pub fn sqrt_def<A: Arith>(a: &mut A, x: A::V) -> A::V {
    let bits = a.fbits();
    let wide = bits == 64;
    let w = if wide { W::I64 } else { W::I32 };
    // f32: m2 ∈ [2^24, 2^26), Z = 12, 13 + 12 steps; f64: m2 ∈ [2^52, 2^54),
    // Z = 27, 27 + 27 steps.
    let (mbits, ebits, bias, z, pairs) = if wide {
        (52u64, 11u64, 1023i64, 27u64, 27u64)
    } else {
        (23, 8, 127, 12, 13)
    };
    let ix = a.cvt(Cvt::Bits, x);
    let c_mb = a.ic(w, mbits);
    let expf = a.i2(w, I2::LShr, ix, c_mb);
    let c_emask = a.ic(w, (1 << ebits) - 1);
    let expf = a.i2(w, I2::And, expf, c_emask);
    let c_mmask = a.ic(w, (1 << mbits) - 1);
    let mant = a.i2(w, I2::And, ix, c_mmask);
    // Subnormal: normalize the significand by its leading zeros.
    let zero_i = a.ic(w, 0);
    let sub = a.icmp(w, IPred::Eq, expf, zero_i);
    let lz = a.clz(w, mant);
    let c_off = a.ic(w, u64::from(w.bits()) - mbits - 1);
    let nz = a.i2(w, I2::Sub, lz, c_off);
    let c_max = a.ic(w, mbits);
    let big = a.icmp(w, IPred::Ugt, nz, c_max);
    let nz = a.sel(big, c_max, nz);
    let msub = a.i2(w, I2::Shl, mant, nz);
    let c_hid = a.ic(w, 1 << mbits);
    let mnorm = a.i2(w, I2::Or, mant, c_hid);
    let m = a.sel(sub, msub, mnorm);
    let one = a.ic(w, 1);
    let esub = a.i2(w, I2::Sub, one, nz);
    let e = a.sel(sub, esub, expf);
    // x = m·2^E, E = e − bias − mbits; make the exponent even.
    let c_eb = a.ic(w, (bias + mbits as i64) as u64);
    let ee = a.i2(w, I2::Sub, e, c_eb);
    let odd = a.i2(w, I2::And, ee, one);
    let sh = if wide {
        odd
    } else {
        let two = a.ic(w, 2);
        a.i2(w, I2::Sub, two, odd)
    };
    let m2 = a.i2(w, I2::Shl, m, sh);
    let e2 = a.i2(w, I2::Sub, ee, sh);
    // The digit recurrence, in 64 bits.
    let m2 = if wide { m2 } else { a.cvt(Cvt::ZExt, m2) };
    let w64 = W::I64;
    let mut rem = a.ic(w64, 0);
    let mut res = a.ic(w64, 0);
    let (one64, two64, three64) = (a.ic(w64, 1), a.ic(w64, 2), a.ic(w64, 3));
    for i in 0..pairs + z {
        let r4 = a.i2(w64, I2::Shl, rem, two64);
        rem = if i < pairs {
            let s = a.ic(w64, 2 * (pairs - 1 - i));
            let pr = a.i2(w64, I2::LShr, m2, s);
            let pr = a.i2(w64, I2::And, pr, three64);
            a.i2(w64, I2::Or, r4, pr)
        } else {
            r4
        };
        let t = a.i2(w64, I2::Shl, res, two64);
        let t = a.i2(w64, I2::Or, t, one64);
        let lt = a.icmp(w64, IPred::Ult, rem, t);
        let d = a.i2(w64, I2::Sub, rem, t);
        rem = a.sel(lt, rem, d);
        let r2 = a.i2(w64, I2::Shl, res, one64);
        let ge = a.cvt(Cvt::BoolToI(w64), lt);
        let ge = a.i2(w64, I2::Xor, ge, one64);
        res = a.i2(w64, I2::Or, r2, ge);
    }
    let res = if wide { res } else { a.cvt(Cvt::Trunc, res) };
    let s = a.i2(w, I2::LShr, res, one);
    let round = a.i2(w, I2::And, res, one);
    // √x ≈ (s + round)·2^(E2/2 + 1 − Z): biased exponent E2/2 + 1 − Z +
    // mbits + bias; the significand's carry runs into the exponent.
    let half = a.i2(w, I2::AShr, e2, one);
    let c_be = a.ic(w, (1 - z as i64 + mbits as i64 + bias) as u64);
    let be = a.i2(w, I2::Add, half, c_be);
    let ebits_v = a.i2(w, I2::Shl, be, c_mb);
    let sm = a.i2(w, I2::Sub, s, c_hid);
    let yb = a.i2(w, I2::Add, ebits_v, sm);
    let yb = a.i2(w, I2::Add, yb, round);
    let y = a.cvt(Cvt::FromBits, yb);
    sqrt_specials(a, x, y)
}

/// `sqrt`'s special values around `y` (a positive finite `x`'s root):
/// `±0 → ±0`, `x < 0 → NaN`, `+∞ → +∞`, NaN → NaN.
pub(super) fn sqrt_specials<A: Arith>(a: &mut A, x: A::V, y: A::V) -> A::V {
    let bits = a.fbits();
    let zero = a.fc(0);
    let inf = a.fc(inf_bits(bits));
    let nan = a.fc(nan_bits(bits));
    let is0 = a.fcmp(FPred::Oeq, x, zero);
    let y = a.sel(is0, x, y);
    let neg = a.fcmp(FPred::Olt, x, zero);
    let y = a.sel(neg, nan, y);
    let isinf = a.fcmp(FPred::Oeq, x, inf);
    let y = a.sel(isinf, x, y);
    let un = a.fcmp(FPred::Uno, x, x);
    a.sel(un, x, y)
}

/// The contract's special values around `y` (a unit's result that need not
/// return them).
pub(super) fn wrap_specials<A: Arith>(f: MathFn, a: &mut A, x: A::V, y: A::V) -> A::V {
    let bits = a.fbits();
    match f {
        MathFn::Sqrt => sqrt_specials(a, x, y),
        MathFn::Exp => {
            let form = ExpForm::of_bits(bits).expect("exp of an f32 or f64");
            let (hi, lo, zero, inf) = (a.fc(form.hi), a.fc(form.lo), a.fc(0), a.fc(inf_bits(bits)));
            let ch = a.fcmp(FPred::Ogt, x, hi);
            let y = a.sel(ch, inf, y);
            let cl = a.fcmp(FPred::Olt, x, lo);
            let y = a.sel(cl, zero, y);
            let cn = a.fcmp(FPred::Uno, x, x);
            a.sel(cn, x, y)
        }
        MathFn::Sin | MathFn::Cos => {
            let zero = a.fc(0);
            let at0 = if f == MathFn::Sin {
                x
            } else {
                a.fc(if bits == 64 {
                    1f64.to_bits()
                } else {
                    u64::from(1f32.to_bits())
                })
            };
            let is0 = a.fcmp(FPred::Oeq, x, zero);
            let y = a.sel(is0, at0, y);
            let d = a.fsub(x, x);
            let un = a.fcmp(FPred::Uno, d, d);
            a.sel(un, d, y)
        }
    }
}

/// The reference definition of `f`.
#[inline]
pub fn reference<A: Arith>(f: MathFn, a: &mut A, x: A::V) -> A::V {
    match f {
        MathFn::Exp => exp_def(a, x),
        MathFn::Sqrt => sqrt_def(a, x),
        MathFn::Sin => sin_def(a, x),
        MathFn::Cos => cos_def(a, x),
    }
}

/// The reference of `f` on an `f32` (the interpreter's): `sqrt` as Rust's
/// correctly rounded square root (equal to [`sqrt_def`]: the correctly
/// rounded value is unique). `#[inline]`: a caller's optimization level
/// compiles it (the interpreter's).
#[inline]
pub fn eval_f32(f: MathFn, x: f32) -> f32 {
    if f == MathFn::Sqrt {
        return x.sqrt();
    }
    let mut n = Native::<32>;
    f32::from_bits(reference(f, &mut n, u64::from(x.to_bits())) as u32)
}

/// The reference of `f` on an `f64`.
#[inline]
pub fn eval_f64(f: MathFn, x: f64) -> f64 {
    if f == MathFn::Sqrt {
        return x.sqrt();
    }
    let mut n = Native::<64>;
    f64::from_bits(reference(f, &mut n, x.to_bits()))
}
