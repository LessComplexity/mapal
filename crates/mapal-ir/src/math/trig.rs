//! Reference definitions of `sin` and `cos`: Payne-Hanek reduction, polynomials and the final fix-up.

use super::*;

/// The bits of `2/π` with 32 zero bits in front, as 32-bit words (f32's
/// Payne–Hanek window: the significand's exponent picks four).
pub const TWO_OVER_PI_32: [u64; 8] = [
    0x0000_0000,
    0xa2f9_836e,
    0x4e44_1529,
    0xfc27_57d1,
    0xf534_ddc0,
    0xdb62_9599,
    0x3c43_9041,
    0xfe51_63ab,
];

/// The bits of `2/π` with 64 zero bits in front, as 64-bit words (f64's
/// window).
pub const TWO_OVER_PI_64: [u64; 20] = [
    0x0000_0000_0000_0000,
    0xa2f9_836e_4e44_1529,
    0xfc27_57d1_f534_ddc0,
    0xdb62_9599_3c43_9041,
    0xfe51_63ab_debb_c561,
    0xb724_6e3a_424d_d2e0,
    0x0649_2eea_09d1_921c,
    0xfe1d_eb1c_b129_a73e,
    0xe882_35f5_2ebb_4484,
    0xe99c_7026_b45f_7e41,
    0x3991_d639_8353_39f4,
    0x9c84_5f8b_bdf9_283b,
    0x1ff8_97ff_de05_980f,
    0xef2f_118b_5a0a_6d1f,
    0x6d36_7ecf_27cb_09b7,
    0x4f46_3f66_9e5f_ea2d,
    0x7527_bac7_ebe5_f17b,
    0x3d07_39f7_8a52_92ea,
    0x6bfb_5fb1_1f8d_5d08,
    0x5603_3046_fc7b_6bab,
];

/// `round(π/2 · 2^31)` and `round(π/2 · 2^63)`.
pub const PIO2_32: u64 = 0xc90f_daa2;
pub const PIO2_64: u64 = 0xc90f_daa2_2168_c235;

/// f32 `sin` on `[−π/4, π/4]` (Cephes `sinf`): `r + r·z·P(z)`.
pub(super) const SIN32: [f32; 3] = [-1.666_665_461_1e-1, 8.332_160_873_6e-3, -1.951_529_589_1e-4];
/// f32 `cos` on `[−π/4, π/4]` (Cephes `cosf`): `1 − z/2 + z²·Q(z)`.
pub(super) const COS32: [f32; 3] = [
    4.166_664_568_298_827e-2,
    -1.388_731_625_493_765e-3,
    2.443_315_711_809_948e-5,
];
/// f64 `__kernel_sin` (fdlibm) `S1 … S6`.
pub(super) const SIN64: [u64; 6] = [
    0xBFC5_5555_5555_5549,
    0x3F81_1111_1110_F8A6,
    0xBF2A_01A0_19C1_61D5,
    0x3EC7_1DE3_57B1_FE7D,
    0xBE5A_E5E6_8A2B_9CEB,
    0x3DE5_D93A_5ACF_D57C,
];
/// f64 `__kernel_cos` (fdlibm) `C1 … C6`.
pub(super) const COS64: [u64; 6] = [
    0x3FA5_5555_5555_554C,
    0xBF56_C16C_16C1_5177,
    0x3EFA_01A0_19CB_1590,
    0xBE92_7E4F_809C_52AD,
    0x3E21_EE9E_BDB4_B1C4,
    0xBDA8_FAE9_BE88_38D4,
];

/// `words[idx + off]` for `idx ∈ 0..n`, by selects over constant words.
pub(super) fn pick<A: Arith>(
    a: &mut A,
    w: W,
    idx: A::V,
    words: &[u64],
    off: usize,
    n: usize,
) -> A::V {
    let mut v = a.ic(w, words[off]);
    for i in 1..n {
        let ii = a.ic(w, i as u64);
        let c = a.icmp(w, IPred::Eq, idx, ii);
        let wv = a.ic(w, words[i + off]);
        v = a.sel(c, wv, v);
    }
    v
}

/// `(hi:lo) << o` keeping the high word, `o ∈ [0, n)` (`lo >> 1 >> (n−1−o)`:
/// every shift below the width).
pub(super) fn funnel<A: Arith>(a: &mut A, w: W, hi: A::V, lo: A::V, o: A::V) -> A::V {
    let one = a.ic(w, 1);
    let nm1 = a.ic(w, u64::from(w.bits()) - 1);
    let h = a.i2(w, I2::Shl, hi, o);
    let l = a.i2(w, I2::LShr, lo, one);
    let s = a.i2(w, I2::Sub, nm1, o);
    let l = a.i2(w, I2::LShr, l, s);
    a.i2(w, I2::Or, h, l)
}

/// The Payne–Hanek reduction: `x = q·π/2 + r`, `|r| ≤ π/4·(1 + ε)`, `q` an
/// `i32` in `0..4`, for every finite `x` (`|x| < π/4`: `r = x`, `q = 0`).
/// Integer arithmetic throughout; `r` carries one rounding.
pub(super) fn reduce<A: Arith>(a: &mut A, x: A::V) -> (A::V, A::V) {
    if a.fbits() == 64 {
        reduce64(a, x)
    } else {
        reduce32(a, x)
    }
}

pub(super) fn reduce32<A: Arith>(a: &mut A, x: A::V) -> (A::V, A::V) {
    let (w, l) = (W::I32, W::I64);
    let ix = a.cvt(Cvt::Bits, x);
    let c_abs = a.ic(w, 0x7fff_ffff);
    let ax = a.i2(w, I2::And, ix, c_abs);
    // |x| < f32(π/4) (the next f32 above π/4).
    let c_pio4 = a.ic(w, 0x3f49_0fdb);
    let small = a.icmp(w, IPred::Ult, ax, c_pio4);
    let c23 = a.ic(w, 23);
    let expo = a.i2(w, I2::LShr, ax, c23);
    let c_mm = a.ic(w, 0x7f_ffff);
    let m = a.i2(w, I2::And, ax, c_mm);
    let c_hid = a.ic(w, 0x80_0000);
    let m = a.i2(w, I2::Or, m, c_hid);
    // x = m·2^E, E = expo − 150; the window starts at 2/π's bit E − 1, the
    // padded table's bit k0 = expo − 120.
    let c120 = a.ic(w, 120);
    let k0 = a.i2(w, I2::Sub, expo, c120);
    let c5 = a.ic(w, 5);
    let wi = a.i2(w, I2::AShr, k0, c5);
    let c31 = a.ic(w, 31);
    let o = a.i2(w, I2::And, k0, c31);
    let t: [A::V; 4] = std::array::from_fn(|j| pick(a, w, wi, &TWO_OVER_PI_32, j, 5));
    let win: [A::V; 3] = std::array::from_fn(|j| funnel(a, w, t[j], t[j + 1], o));
    // The 96-bit product's bits [32, 96) (mod 2^96: multiples of 4 dropped).
    let m64 = a.cvt(Cvt::ZExt, m);
    let p: [A::V; 3] = std::array::from_fn(|j| {
        let v = a.cvt(Cvt::ZExt, win[j]);
        a.i2(l, I2::Mul, m64, v)
    });
    let c32 = a.ic(l, 32);
    let hi = a.i2(l, I2::Shl, p[0], c32);
    let hi = a.i2(l, I2::Add, hi, p[1]);
    let p2 = a.i2(l, I2::LShr, p[2], c32);
    let hi = a.i2(l, I2::Add, hi, p2);
    finish_reduce(a, x, ix, small, hi, None)
}

pub(super) fn reduce64<A: Arith>(a: &mut A, x: A::V) -> (A::V, A::V) {
    let l = W::I64;
    let ix = a.cvt(Cvt::Bits, x);
    let c_abs = a.ic(l, 0x7fff_ffff_ffff_ffff);
    let ax = a.i2(l, I2::And, ix, c_abs);
    // |x| ≤ f64(π/4) (below π/4).
    let c_pio4 = a.ic(l, 0x3fe9_21fb_5444_2d19);
    let small = a.icmp(l, IPred::Ult, ax, c_pio4);
    let c52 = a.ic(l, 52);
    let expo = a.i2(l, I2::LShr, ax, c52);
    let c_mm = a.ic(l, (1 << 52) - 1);
    let m = a.i2(l, I2::And, ax, c_mm);
    let c_hid = a.ic(l, 1 << 52);
    let m = a.i2(l, I2::Or, m, c_hid);
    // E = expo − 1075; bit k0 = expo − 1013 of the padded table.
    let c1013 = a.ic(l, 1013);
    let k0 = a.i2(l, I2::Sub, expo, c1013);
    let c6 = a.ic(l, 6);
    let wi = a.i2(l, I2::AShr, k0, c6);
    let c63 = a.ic(l, 63);
    let o = a.i2(l, I2::And, k0, c63);
    let t: [A::V; 4] = std::array::from_fn(|j| pick(a, l, wi, &TWO_OVER_PI_64, j, 17));
    let win: [A::V; 3] = std::array::from_fn(|j| funnel(a, l, t[j], t[j + 1], o));
    // The 192-bit product's bits [64, 192): top = L0 + H1 + carry, mid = L1 + H2.
    let l0 = a.i2(l, I2::Mul, m, win[0]);
    let h1 = a.i2(l, I2::MulHi, m, win[1]);
    let l1 = a.i2(l, I2::Mul, m, win[1]);
    let h2 = a.i2(l, I2::MulHi, m, win[2]);
    let mid = a.i2(l, I2::Add, l1, h2);
    let carry = a.icmp(l, IPred::Ult, mid, l1);
    let carry = a.cvt(Cvt::BoolToI(l), carry);
    let top = a.i2(l, I2::Add, l0, h1);
    let top = a.i2(l, I2::Add, top, carry);
    finish_reduce(a, x, ix, small, top, Some(mid))
}

/// From the product's top word (and, f64, the next): the quadrant, the
/// signed remainder's magnitude normalized and multiplied by `π/2` in fixed
/// point, one rounding to the width, the signs.
pub(super) fn finish_reduce<A: Arith>(
    a: &mut A,
    x: A::V,
    ix: A::V,
    small: A::V,
    top: A::V,
    mid: Option<A::V>,
) -> (A::V, A::V) {
    let l = W::I64;
    let w32 = W::I32;
    let wide = mid.is_some();
    let (c2, c62, c63, c64) = (a.ic(l, 2), a.ic(l, 62), a.ic(l, 63), a.ic(l, 64));
    let zero = a.ic(l, 0);
    let qv = a.i2(l, I2::LShr, top, c62);
    let q = a.cvt(Cvt::Trunc, qv);
    // The fraction in units of 2^-64 (f32) / 2^-128 (f64), rounded to the
    // nearest quadrant: a set top bit is the negative remainder.
    let fh = a.i2(l, I2::Shl, top, c2);
    let (fh, fl) = match mid {
        Some(mid) => {
            let mt = a.i2(l, I2::LShr, mid, c62);
            let fh = a.i2(l, I2::Or, fh, mt);
            let fl = a.i2(l, I2::Shl, mid, c2);
            (fh, Some(fl))
        }
        None => (fh, None),
    };
    let neg = a.icmp(l, IPred::Slt, fh, zero);
    let negi = a.cvt(Cvt::BoolToI(w32), neg);
    let q = a.i2(w32, I2::Add, q, negi);
    let (mag_top, n) = match fl {
        None => {
            let nf = a.i2(l, I2::Sub, zero, fh);
            let mag = a.sel(neg, nf, fh);
            let n = a.clz(l, mag);
            let big = a.icmp(l, IPred::Ugt, n, c63);
            let ns = a.sel(big, c63, n);
            let top = a.i2(l, I2::Shl, mag, ns);
            (top, n)
        }
        Some(fl) => {
            // −(fh:fl) as 128 bits.
            let nfl = a.i2(l, I2::Sub, zero, fl);
            let all = a.ic(l, u64::MAX);
            let nfh = a.i2(l, I2::Xor, fh, all);
            let lz = a.icmp(l, IPred::Eq, fl, zero);
            let lzi = a.cvt(Cvt::BoolToI(l), lz);
            let nfh = a.i2(l, I2::Add, nfh, lzi);
            let mh = a.sel(neg, nfh, fh);
            let ml = a.sel(neg, nfl, fl);
            let zh = a.icmp(l, IPred::Eq, mh, zero);
            let nh = a.clz(l, mh);
            let nl = a.clz(l, ml);
            let n_lo = a.i2(l, I2::Add, nl, c64);
            let n = a.sel(zh, n_lo, nh);
            let bh = a.icmp(l, IPred::Ugt, nh, c63);
            let na = a.sel(bh, c63, nh);
            let top_a = funnel(a, l, mh, ml, na);
            let bl = a.icmp(l, IPred::Ugt, nl, c63);
            let nb = a.sel(bl, c63, nl);
            let top_b = a.i2(l, I2::Shl, ml, nb);
            let top = a.sel(zh, top_b, top_a);
            (top, n)
        }
    };
    // r = mag·π/2: the normalized top word times π/2 in fixed point.
    let rf = if wide {
        let pio2 = a.ic(l, PIO2_64);
        let ph = a.i2(l, I2::MulHi, mag_top, pio2);
        a.cvt(Cvt::UiToF(l), ph)
    } else {
        let c32 = a.ic(l, 32);
        let t32 = a.i2(l, I2::LShr, mag_top, c32);
        let pio2 = a.ic(l, PIO2_32);
        let pr = a.i2(l, I2::Mul, t32, pio2);
        a.cvt(Cvt::UiToF(l), pr)
    };
    // The scale 2^(−63−n): f32 bits (64 − n) << 23, f64 (960 − n) << 52.
    let scale = if wide {
        let c960 = a.ic(l, 960);
        let e = a.i2(l, I2::Sub, c960, n);
        let c52 = a.ic(l, 52);
        let e = a.i2(l, I2::Shl, e, c52);
        a.cvt(Cvt::FromBits, e)
    } else {
        let n32 = a.cvt(Cvt::Trunc, n);
        let c = a.ic(w32, 64);
        let e = a.i2(w32, I2::Sub, c, n32);
        let c23 = a.ic(w32, 23);
        let e = a.i2(w32, I2::Shl, e, c23);
        a.cvt(Cvt::FromBits, e)
    };
    let r = a.fmul(rf, scale);
    // The signs: r < 0 iff the remainder or x is negative (not both); a
    // negative x's quadrant turns around.
    let iw = if wide { l } else { w32 };
    let zi = a.ic(iw, 0);
    let xneg = a.icmp(iw, IPred::Slt, ix, zi);
    let rneg = a.b2(B2::Xor, neg, xneg);
    let nr = a.fneg(r);
    let r = a.sel(rneg, nr, r);
    let z32 = a.ic(w32, 0);
    let nq = a.i2(w32, I2::Sub, z32, q);
    let q = a.sel(xneg, nq, q);
    let c3 = a.ic(w32, 3);
    let q = a.i2(w32, I2::And, q, c3);
    let r = a.sel(small, x, r);
    let q = a.sel(small, z32, q);
    (q, r)
}

/// `sin r` on `[−π/4, π/4]` (contractible).
pub(super) fn sin_poly<A: Arith>(a: &mut A, r: A::V) -> A::V {
    a.contract(true);
    let z = a.fmul(r, r);
    let y = if a.fbits() == 64 {
        let c: [A::V; 6] = std::array::from_fn(|i| a.fc(SIN64[i]));
        let v = a.fmul(z, r);
        let mut h = c[5];
        for i in (1..5).rev() {
            let t = a.fmul(z, h);
            h = a.fadd(c[i], t);
        }
        let t = a.fmul(z, h);
        let s = a.fadd(c[0], t);
        let t = a.fmul(v, s);
        a.fadd(r, t)
    } else {
        let c: [A::V; 3] = std::array::from_fn(|i| a.fc(u64::from(SIN32[i].to_bits())));
        let t = a.fmul(c[2], z);
        let t = a.fadd(t, c[1]);
        let t = a.fmul(t, z);
        let t = a.fadd(t, c[0]);
        let t = a.fmul(t, z);
        let t = a.fmul(t, r);
        a.fadd(t, r)
    };
    a.contract(false);
    y
}

/// `cos r` on `[−π/4, π/4]` (contractible).
pub(super) fn cos_poly<A: Arith>(a: &mut A, r: A::V) -> A::V {
    a.contract(true);
    let z = a.fmul(r, r);
    let half = a.fc(if a.fbits() == 64 {
        0.5f64.to_bits()
    } else {
        u64::from(0.5f32.to_bits())
    });
    let one = a.fc(if a.fbits() == 64 {
        1f64.to_bits()
    } else {
        u64::from(1f32.to_bits())
    });
    let y = if a.fbits() == 64 {
        let c: [A::V; 6] = std::array::from_fn(|i| a.fc(COS64[i]));
        let mut h = c[5];
        for i in (0..5).rev() {
            let t = a.fmul(z, h);
            h = a.fadd(c[i], t);
        }
        let rr = a.fmul(z, h);
        let hz = a.fmul(half, z);
        let wv = a.fsub(one, hz);
        let t = a.fsub(one, wv);
        let t = a.fsub(t, hz);
        let zr = a.fmul(z, rr);
        let t = a.fadd(t, zr);
        a.fadd(wv, t)
    } else {
        let c: [A::V; 3] = std::array::from_fn(|i| a.fc(u64::from(COS32[i].to_bits())));
        let t = a.fmul(c[2], z);
        let t = a.fadd(t, c[1]);
        let t = a.fmul(t, z);
        let t = a.fadd(t, c[0]);
        let t = a.fmul(t, z);
        let t = a.fmul(t, z);
        let hz = a.fmul(half, z);
        let t = a.fsub(t, hz);
        a.fadd(t, one)
    };
    a.contract(false);
    y
}

/// The quadrant's choice between `sin r` and `cos r` and its sign, `sin`'s
/// signed zero, a non-finite `x`'s NaN.
pub(super) fn trig_finish<A: Arith>(
    a: &mut A,
    x: A::V,
    q: A::V,
    s: A::V,
    c: A::V,
    cos: bool,
) -> A::V {
    let w32 = W::I32;
    let (one, two, three) = (a.ic(w32, 1), a.ic(w32, 2), a.ic(w32, 3));
    let q = if cos {
        let q1 = a.i2(w32, I2::Add, q, one);
        a.i2(w32, I2::And, q1, three)
    } else {
        q
    };
    let z = a.ic(w32, 0);
    let b0 = a.i2(w32, I2::And, q, one);
    let odd = a.icmp(w32, IPred::Ne, b0, z);
    let v = a.sel(odd, c, s);
    let b1 = a.i2(w32, I2::And, q, two);
    let flip = a.icmp(w32, IPred::Ne, b1, z);
    let nv = a.fneg(v);
    let v = a.sel(flip, nv, v);
    let fz = a.fc(0);
    let v = if cos {
        v
    } else {
        let is0 = a.fcmp(FPred::Oeq, x, fz);
        a.sel(is0, x, v)
    };
    // x − x: 0 for a finite x, NaN for ±∞ and NaN.
    let d = a.fsub(x, x);
    let un = a.fcmp(FPred::Uno, d, d);
    a.sel(un, d, v)
}

/// `sin`'s reference `S_b`: Payne–Hanek reduction, polynomial, quadrant.
pub fn sin_def<A: Arith>(a: &mut A, x: A::V) -> A::V {
    trig_def(a, x, false, (None, None))
}

/// `cos`'s reference `C_b`.
pub fn cos_def<A: Arith>(a: &mut A, x: A::V) -> A::V {
    trig_def(a, x, true, (None, None))
}

pub(super) fn trig_def<A: Arith>(
    a: &mut A,
    x: A::V,
    cos: bool,
    units: (Option<Unit>, Option<Unit>),
) -> A::V {
    let (q, r) = reduce(a, x);
    let s = match &units.0 {
        Some(u) => a.unit(u, r),
        None => sin_poly(a, r),
    };
    let c = match &units.1 {
        Some(u) => a.unit(u, r),
        None => cos_poly(a, r),
    };
    trig_finish(a, x, q, s, c, cos)
}
