//! ADR-0046: the references, their traces, the contracts and the
//! realization rule.

use super::*;

/// ADR-0042's `E_32` exactly as `exp.rs` wrote it before the framework (the
/// fixture `exp_def` must reproduce bit for bit).
fn e32_as_written(x: f32) -> f32 {
    let f = crate::exp::F32;
    let c = |b: u64| f32::from_bits(b as u32);
    let (hi, lo) = (c(f.hi), c(f.lo));
    let mut xc = if x > hi { hi } else { x };
    xc = if xc < lo { lo } else { xc };
    xc = if xc.is_nan() { 0.0 } else { xc };
    let t = xc * c(f.log2e) + c(f.shifter);
    let kf = t - c(f.shifter);
    let k = kf as i32;
    let r = (xc - kf * c(f.ln2_hi)) - kf * c(f.ln2_lo);
    let cs = f.coefs();
    let mut p = c(cs[f.degree as usize]);
    for i in (0..f.degree as usize).rev() {
        p = p * r + c(cs[i]);
    }
    let k1 = k >> 1;
    let k2 = k - k1;
    let s1 = f32::from_bits(((k1 + 127) as u32) << 23);
    let s2 = f32::from_bits(((k2 + 127) as u32) << 23);
    let mut y = (p * s1) * s2;
    if x > hi {
        y = f32::INFINITY;
    }
    if x < lo {
        y = 0.0;
    }
    if x.is_nan() {
        y = x;
    }
    y
}

fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

fn of32(b: u32) -> f64 {
    f64::from(f32::from_bits(b))
}

#[test]
fn names_resolve() {
    for f in MathFn::ALL {
        assert_eq!(MathFn::from_name(f.name()), Some(f));
    }
    assert_eq!(MathFn::from_name("tan"), None);
    assert_eq!(MathFn::Sqrt.label(), "Sqrt");
}

/// `exp` moved onto the framework with no behaviour change: the generic
/// definition evaluated natively is ADR-0042's `E_32` bit for bit (every
/// 97th f32 and the specials), and `E_64` at the samples.
#[test]
fn exp_def_is_adr_0042_bit_for_bit() {
    let mut b: u32 = 0;
    loop {
        let x = f32::from_bits(b);
        let want = e32_as_written(x);
        let got = eval_f32(MathFn::Exp, x);
        assert!(
            same(f64::from(got), f64::from(want)),
            "{x:e}: {got:e} vs {want:e}"
        );
        match b.checked_add(97) {
            Some(n) => b = n,
            None => break,
        }
    }
    for x in [0.0, -0.0, 1.0, -745.2, 709.0, f64::INFINITY, f64::NAN] {
        assert!(same(eval_f64(MathFn::Exp, x), crate::exp::exp_f64(x)));
    }
}

/// A trace evaluated step by step is the native evaluation, for every
/// builtin, width and test argument: what a backend prints is what the
/// interpreter computes.
#[test]
fn traces_compute_the_native_reference() {
    for f in MathFn::ALL {
        for bits in [32u32, 64] {
            let (mut t, x) = Trace::new(bits);
            let y = reference(f, &mut t, x);
            let def = t.finish(y);
            assert!(!def.uses(|o| matches!(o, Op::Fma | Op::Unit(..))), "{f:?}");
            for a in test_args(f, bits) {
                let (got, want) = if bits == 64 {
                    let mut n = Native::<64>;
                    let w = reference(f, &mut n, a.to_bits());
                    (f64::from_bits(def.eval(a.to_bits())), f64::from_bits(w))
                } else {
                    let xb = u64::from((a as f32).to_bits());
                    let mut n = Native::<32>;
                    let w = reference(f, &mut n, xb);
                    (of32(def.eval(xb) as u32), of32(w as u32))
                };
                assert!(same(got, want), "{f:?} f{bits} {a:e}: {got:e} vs {want:e}");
            }
        }
    }
}

/// `sqrt_def` (plain operations) is the correctly rounded square root —
/// every IEEE `sqrt`: f32 every 211th pattern, every subnormal step and the
/// binade edges; f64 seeded samples over every exponent.
#[test]
fn sqrt_def_is_correctly_rounded() {
    let d32 = |x: f32| -> f32 {
        let mut n = Native::<32>;
        f32::from_bits(sqrt_def(&mut n, u64::from(x.to_bits())) as u32)
    };
    let mut b: u32 = 0;
    let mut n = 0u64;
    loop {
        let x = f32::from_bits(b);
        let (g, w) = (d32(x), x.sqrt());
        assert!(
            same(f64::from(g), f64::from(w)),
            "{x:e} {b:#x}: {g:e} vs {w:e}"
        );
        n += 1;
        match b.checked_add(211) {
            Some(nb) => b = nb,
            None => break,
        }
    }
    assert!(n > 20_000_000 / 1_000);
    for b in (1u32..0x80_0000)
        .step_by(4093)
        .chain([1, 2, 3, 0x7f_ffff, 0x80_0000, 0x7f7f_ffff])
    {
        let x = f32::from_bits(b);
        assert_eq!(d32(x).to_bits(), x.sqrt().to_bits(), "{b:#x}");
    }
    let d64 = |x: f64| -> f64 {
        let mut n = Native::<64>;
        f64::from_bits(sqrt_def(&mut n, x.to_bits()))
    };
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..200_000 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let x = f64::from_bits(s);
        assert!(same(d64(x), x.sqrt()), "{x:e}");
    }
    for x in [
        0.0,
        -0.0,
        f64::INFINITY,
        -1.0,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::MAX,
        2.0,
    ] {
        assert!(same(d64(x), x.sqrt()), "{x:e}");
    }
    assert!(d32(-0.0).is_sign_negative() && d32(-0.0) == 0.0);
    assert!(d32(-1.0).is_nan() && d32(f32::NAN).is_nan());
}

/// The worst error of `S_32` / `C_32` in ulps of the f64 truth over a dense
/// f32 sweep (every 1021st pattern: the whole range, huge arguments
/// included; `math_f32_exhaustive` in `mapal-interp`'s optimized tests runs
/// every f32), inside the default face's bound everywhere.
#[test]
fn sin_cos_f32_within_contract_over_the_whole_range() {
    for f in [MathFn::Sin, MathFn::Cos] {
        let mut worst = 0f64;
        let mut b: u32 = 0;
        loop {
            let x = f32::from_bits(b);
            let got = f64::from(eval_f32(f, x));
            let t = truth(f, f64::from(x));
            assert!(
                within(f, 32, f64::from(x), got, t, 0.0),
                "{f:?}({x:e}) = {got:e}, truth {t:e}"
            );
            if t.is_finite() && t != 0.0 && x.is_finite() && x != 0.0 {
                let ulp = 2f64.powi(t.abs().log2().floor() as i32 - 23);
                worst = worst.max((got - t).abs() / ulp);
            }
            match b.checked_add(1021) {
                Some(n) => b = n,
                None => break,
            }
        }
        assert!(worst <= 2.0, "{f:?} worst {worst} ulp");
        eprintln!("{f:?} f32 worst {worst:.3} ulp");
    }
}

/// `S_64` / `C_64` against libm over seeded samples of every exponent and
/// `[−8, 8]`, and the worst cancellation `6381956970095103·2^797`.
#[test]
fn sin_cos_f64_within_contract() {
    for f in [MathFn::Sin, MathFn::Cos] {
        let mut worst = 0f64;
        let mut s: u64 = 0x2545_F491_4F6C_DD1D;
        for i in 0..400_000u32 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let x = if i % 2 == 0 {
                f64::from_bits(s)
            } else {
                (s >> 11) as f64 / (1u64 << 53) as f64 * 16.0 - 8.0
            };
            let got = eval_f64(f, x);
            let t = truth(f, x);
            assert!(
                within(f, 64, x, got, t, 2f64.powi(-52)),
                "{f:?}({x:e}) = {got:e}, libm {t:e}"
            );
            if t.is_finite() && t.abs() > 1e-300 {
                let ulp = 2f64.powi(t.abs().log2().floor() as i32 - 52);
                worst = worst.max((got - t).abs() / ulp);
            }
        }
        assert!(worst <= 2.5, "{f:?} worst {worst} ulp of libm");
        eprintln!("{f:?} f64 worst {worst:.3} ulp of libm");
        for x in test_args(f, 64) {
            assert!(
                within(f, 64, x, eval_f64(f, x), truth(f, x), 2f64.powi(-52)),
                "{x:e}"
            );
        }
    }
}

/// The special values of every builtin and width, on the reference.
#[test]
fn special_values() {
    for bits in [32u32, 64] {
        let e = |f: MathFn, x: f64| -> f64 {
            if bits == 64 {
                eval_f64(f, x)
            } else {
                f64::from(eval_f32(f, x as f32))
            }
        };
        for f in MathFn::ALL {
            for x in [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -1.0] {
                if let Some(s) = special(f, bits, x) {
                    let g = e(f, x);
                    assert!(
                        same(g, s) || (s.is_nan() && g.is_nan()),
                        "{f:?} f{bits} ({x}) = {g}, want {s}"
                    );
                }
            }
        }
        assert!(e(MathFn::Sin, -0.0).is_sign_negative());
        assert_eq!(e(MathFn::Cos, -0.0), 1.0);
        assert!(e(MathFn::Sqrt, -0.0).is_sign_negative());
        let tiny = if bits == 64 {
            f64::from_bits(1)
        } else {
            of32(1)
        };
        assert_eq!(e(MathFn::Sin, tiny), tiny);
        assert_eq!(e(MathFn::Sin, -tiny), -tiny);
        assert_eq!(e(MathFn::Cos, tiny), 1.0);
    }
}

/// The `2/π` words: the top 64 bits are `2/π`'s; the f32 table is the f64
/// table's halves.
#[test]
fn two_over_pi_words() {
    let v = TWO_OVER_PI_64[1] as f64 / 2f64.powi(64);
    assert!((v - 2.0 / std::f64::consts::PI).abs() < 1e-16);
    // The 32-bit table is the 64-bit one less its first 32 zero bits.
    for (i, w) in TWO_OVER_PI_32.iter().enumerate() {
        let h = if i % 2 == 1 {
            TWO_OVER_PI_64[(i + 1) / 2] >> 32
        } else {
            TWO_OVER_PI_64[i / 2] & 0xffff_ffff
        };
        assert_eq!(*w, h, "word {i}");
    }
    assert!((PIO2_64 as f64 / 2f64.powi(63) - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    assert!((PIO2_32 as f64 / 2f64.powi(31) - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    // fdlibm's coefficients are the documented decimals.
    let s1 = f64::from_bits(SIN64[0]);
    assert!((s1 + 1.666_666_666_666_663_2e-1).abs() < 1e-17);
    assert!((f64::from_bits(COS64[5]) + 1.135_964_755_778_819_5e-11).abs() < 1e-24);
}

/// The contract table (ADR-0046 D3).
#[test]
fn contract_table() {
    let c = |f, b| contract(f, b).unwrap();
    assert_eq!(c(MathFn::Exp, 32).rel, 2f64.powi(-21));
    assert_eq!(c(MathFn::Exp, 64).rel, 2f64.powi(-50));
    assert_eq!(c(MathFn::Sqrt, 32).rel, 2f64.powi(-22));
    assert_eq!(c(MathFn::Sqrt, 64).abs, 0.0);
    assert_eq!(c(MathFn::Sin, 32).abs, of32(1));
    assert_eq!(c(MathFn::Cos, 64).rel, 2f64.powi(-50));
    assert!(contract(MathFn::Sin, 16).is_none());
    let m = MathForm::of(MathFn::Exp, 32).unwrap();
    assert_eq!(m.cond(-3.0), 3.0);
    assert!((m.compose(2.0, 1e-7) - (2f64.powi(-21) + 2e-7)).abs() < 1e-20);
    assert_eq!(MathForm::of(MathFn::Sqrt, 64).unwrap().cond(9.0), 0.5);
}

const EX2_3: Unit = Unit {
    op: UnitOp::Ex2,
    bits: 32,
    sym: "llvm.nvvm.ex2.approx.ftz.f",
    err: Err::Bound {
        rel: 3.0 / 16_777_216.0,
        abs: 0.0,
    },
    lo: -0.5,
    hi: 0.5,
    cost: 1,
    specials: false,
    evidence: "test",
};

/// `exp`'s realization rule is ADR-0042's: the oracle off the default face
/// and under the OFF arm; on it the `ex2` form where the unit's error meets
/// the bound (f32 only), else the contracted steps where the face contracts.
#[test]
fn exp_realization_is_adr_0042s() {
    let units3 = [EX2_3];
    let t3 = Target {
        units: Box::leak(Box::new(units3)),
        defs: &[],
    };
    let ex2_9 = Unit {
        err: Err::Bound {
            rel: 9.0 / 16_777_216.0,
            abs: 0.0,
        },
        ..EX2_3
    };
    let t9 = Target {
        units: Box::leak(Box::new([ex2_9])),
        defs: &[],
    };
    let r = |bits, face, contract, hw, t: &Target| {
        choose(MathFn::Exp, bits, face, contract, hw, t).real
    };
    assert_eq!(r(32, Face::Exact, true, true, &t3), Real::Reference);
    assert_eq!(r(32, Face::Default, true, false, &t3), Real::Reference);
    assert_eq!(
        r(32, Face::Default, true, true, &t3),
        Real::Form(Form::ExpEx2(EX2_3))
    );
    assert_eq!(r(64, Face::Default, true, true, &t3), Real::Contracted);
    assert_eq!(r(32, Face::Default, true, true, &t9), Real::Contracted);
    assert_eq!(
        r(32, Face::Default, true, true, &Target::NONE),
        Real::Contracted
    );
    assert_eq!(
        r(32, Face::Default, false, true, &Target::NONE),
        Real::Reference
    );
}

/// An IEEE `sqrt` instruction is `Exact`: taken on every face (and under
/// the OFF arm); an approximate unit only on the default face and only
/// within the bound, never without evidence; no unit: the plain-operation
/// reference.
#[test]
fn sqrt_realizations_by_error_and_cost() {
    let rn = Unit {
        op: UnitOp::Sqrt,
        bits: 32,
        sym: "llvm.sqrt.f32",
        err: Err::Exact,
        lo: 0.0,
        hi: f64::INFINITY,
        // `sqrt.rn`'s measured 19.7 cycles a warp op: dearer than the
        // approximate unit even with its special values wrapped around it.
        cost: 20,
        specials: true,
        evidence: "IEEE 754 squareRoot",
    };
    let approx = Unit {
        sym: "llvm.nvvm.sqrt.approx.f",
        err: Err::Bound {
            rel: 2f64.powi(-23),
            abs: 0.0,
        },
        cost: 1,
        specials: false,
        evidence: "calib",
        ..rn
    };
    let loose = Unit {
        err: Err::Bound {
            rel: 2f64.powi(-20),
            abs: 0.0,
        },
        ..approx
    };
    let bare = Unit {
        evidence: "",
        ..approx
    };
    let t = |u: &[Unit]| Target {
        units: Box::leak(u.to_vec().into_boxed_slice()),
        defs: &[],
    };
    let r = |face, hw, tg: &Target| choose(MathFn::Sqrt, 32, face, true, hw, tg).real;
    let both = t(&[rn, approx]);
    assert_eq!(r(Face::Exact, true, &both), Real::Form(Form::Direct(rn)));
    assert_eq!(
        r(Face::Default, true, &both),
        Real::Form(Form::Direct(approx))
    );
    assert_eq!(r(Face::Default, false, &both), Real::Form(Form::Direct(rn)));
    assert_eq!(
        r(Face::Default, true, &t(&[rn, loose])),
        Real::Form(Form::Direct(rn))
    );
    assert_eq!(
        r(Face::Default, true, &t(&[rn, bare])),
        Real::Form(Form::Direct(rn))
    );
    // No unit: the plain-operation reference (nothing to contract).
    assert_eq!(r(Face::Default, true, &Target::NONE), Real::Reference);
    assert_eq!(r(Face::Exact, true, &Target::NONE), Real::Reference);
    let c = candidates(MathFn::Sqrt, 32, &t(&[bare]), true);
    let b = c
        .iter()
        .find(|c| c.real == Real::Form(Form::Direct(bare)))
        .unwrap();
    assert_eq!(
        admits(b, Face::Default, &Target::NONE),
        Err(Refusal::NoEvidence)
    );
    // The approximate unit without its own special values is wrapped.
    let def = trace(
        &choose(MathFn::Sqrt, 32, Face::Default, true, true, &both),
        &both,
    );
    assert!(def.uses(|o| matches!(o, Op::Unit(UnitOp::Sqrt, _))));
    assert!(def.uses(|o| *o == Op::Sel));
}

/// `sin` / `cos` take hardware units only where both units' composed error
/// is relative within the bound: an absolute-error unit (MUFU `sin.approx`'s
/// kind) is refused, a relative one is admitted and cheaper than the
/// polynomial.
#[test]
fn trig_units_admitted_by_their_error() {
    let pio4 = 0.7854;
    let s = Unit {
        op: UnitOp::Sin,
        bits: 32,
        sym: "llvm.nvvm.sin.approx.f",
        err: Err::Bound {
            rel: 0.0,
            abs: 2f64.powi(-21),
        },
        lo: -pio4,
        hi: pio4,
        cost: 1,
        specials: false,
        evidence: "calib",
    };
    let c = Unit {
        op: UnitOp::Cos,
        sym: "llvm.nvvm.cos.approx.f",
        ..s
    };
    let abs_t = Target {
        units: Box::leak(Box::new([s, c])),
        defs: &[],
    };
    for f in [MathFn::Sin, MathFn::Cos] {
        let ch = choose(f, 32, Face::Default, true, true, &abs_t);
        assert_eq!(ch.real, Real::Contracted, "{f:?}");
        let rel = Err::Bound {
            rel: 2f64.powi(-23),
            abs: 0.0,
        };
        let rel_t = Target {
            units: Box::leak(Box::new([Unit { err: rel, ..s }, Unit { err: rel, ..c }])),
            defs: &[],
        };
        let ch = choose(f, 32, Face::Default, true, true, &rel_t);
        assert!(
            matches!(ch.real, Real::Form(Form::Trig(..))),
            "{f:?} {ch:?}"
        );
        let cand = candidates(f, 32, &rel_t, true);
        let poly = cand.iter().find(|c| c.real == Real::Contracted).unwrap();
        assert!(ch.cost < poly.cost);
        // Never on the exact face.
        assert_eq!(
            choose(f, 32, Face::Exact, true, true, &rel_t).real,
            Real::Reference
        );
        // An absolute-error `sin` unit beside a relative `cos` unit (the RTX
        // 5080's measured pair): the polynomial's `sin r`, the unit's `cos r`.
        let mixed = Target {
            units: Box::leak(Box::new([s, Unit { err: rel, ..c }])),
            defs: &[],
        };
        let ch = choose(f, 32, Face::Default, true, true, &mixed);
        assert_eq!(
            ch.real,
            Real::Form(Form::Trig(None, Some(Unit { err: rel, ..c }))),
            "{f:?}"
        );
        let def = trace(&ch, &mixed);
        assert!(def.uses(|o| matches!(o, Op::Unit(UnitOp::Cos, _))));
        assert!(!def.uses(|o| matches!(o, Op::Unit(UnitOp::Sin, _))));
    }
}

fn liar_sin(t: &mut Trace, x: Val) -> Val {
    // sin x ≈ x: right near 0, wrong elsewhere.
    t.name(x, "x")
}

fn honest_sin(t: &mut Trace, x: Val) -> Val {
    sin_def(t, x)
}

/// **A target's definition override** is admitted only if it meets the
/// contract on the face in use: a plain-operation routine is re-measured
/// over the contract's test domain — one declaring a bound its trace
/// violates is refused, one declaring `Exact` that is the reference is
/// taken on the exact face; an external routine without evidence is not
/// admitted, with evidence only on the default face.
#[test]
fn target_overrides_are_checked_against_the_contract() {
    let bound = Err::Bound {
        rel: 2f64.powi(-22),
        abs: 0.0,
    };
    let liar = TargetDef {
        f: MathFn::Sin,
        bits: 32,
        body: DefBody::Steps(liar_sin),
        err: bound,
        cost: Some(1),
        evidence: "vendor sheet",
    };
    let t = Target {
        units: &[],
        defs: Box::leak(Box::new([liar])),
    };
    let cands = candidates(MathFn::Sin, 32, &t, true);
    let l = cands.iter().find(|c| c.real == Real::Target(0)).unwrap();
    assert!(matches!(
        admits(l, Face::Default, &t),
        Err(Refusal::Measured { .. })
    ));
    assert_eq!(
        choose(MathFn::Sin, 32, Face::Default, true, true, &t).real,
        Real::Contracted
    );
    // Declared outside the contract: refused before measuring.
    let wide = TargetDef {
        err: Err::Bound {
            rel: 2f64.powi(-12),
            abs: 0.0,
        },
        ..liar
    };
    let tw = Target {
        units: &[],
        defs: Box::leak(Box::new([wide])),
    };
    let w = candidates(MathFn::Sin, 32, &tw, true)[2];
    assert_eq!(admits(&w, Face::Default, &tw), Err(Refusal::Bound));
    // An honest routine declaring `Exact`: on every face, at its cost.
    let honest = TargetDef {
        body: DefBody::Steps(honest_sin),
        err: Err::Exact,
        cost: Some(3),
        evidence: "is the reference",
        ..liar
    };
    let th = Target {
        units: &[],
        defs: Box::leak(Box::new([honest])),
    };
    assert_eq!(
        choose(MathFn::Sin, 32, Face::Exact, true, true, &th).real,
        Real::Target(0)
    );
    // An `Exact` declaration its trace contradicts.
    let lie_exact = TargetDef {
        err: Err::Exact,
        ..liar
    };
    let tl = Target {
        units: &[],
        defs: Box::leak(Box::new([lie_exact])),
    };
    let c = candidates(MathFn::Sin, 32, &tl, true)[2];
    assert!(matches!(
        admits(&c, Face::Exact, &tl),
        Err(Refusal::Measured { .. })
    ));
    // An external symbol: evidence makes it admissible on the default face
    // only (a bound), never without.
    let sym = TargetDef {
        body: DefBody::Symbol("asic_sinf"),
        ..liar
    };
    let ts = Target {
        units: &[],
        defs: Box::leak(Box::new([
            sym,
            TargetDef {
                evidence: "",
                ..sym
            },
        ])),
    };
    let cs = candidates(MathFn::Sin, 32, &ts, true);
    assert_eq!(admits(&cs[2], Face::Default, &ts), Ok(()));
    assert_eq!(admits(&cs[2], Face::Exact, &ts), Err(Refusal::NotExact));
    assert_eq!(admits(&cs[3], Face::Default, &ts), Err(Refusal::NoEvidence));
    assert_eq!(
        choose(MathFn::Sin, 32, Face::Default, true, true, &ts).real,
        Real::Target(0)
    );
}

/// Costs follow the traced operations: the contracted reference fuses its
/// products, the `ex2` form is cheaper than the polynomial.
#[test]
fn costs_follow_the_traces() {
    let t = Target {
        units: Box::leak(Box::new([EX2_3])),
        defs: &[],
    };
    let c = candidates(MathFn::Exp, 32, &t, true);
    let cost = |r: Real| c.iter().find(|c| c.real == r).unwrap().cost;
    assert!(cost(Real::Contracted) < cost(Real::Reference));
    assert!(cost(Real::Form(Form::ExpEx2(EX2_3))) < cost(Real::Contracted));
    for f in [MathFn::Sin, MathFn::Cos] {
        let c32 = candidates(f, 32, &Target::NONE, false)[0].cost;
        let c64 = candidates(f, 64, &Target::NONE, false)[0].cost;
        eprintln!("{f:?} reference ops: f32 {c32}, f64 {c64}");
        assert!(c32 < c64 && c32 < 140 && c64 < 300);
    }
}
