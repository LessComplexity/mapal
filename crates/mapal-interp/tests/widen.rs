//! WP2 oracle contract for the four ADR-0029 widening lattice edges.

use mapal_interp::{Outcome, RValue, eval_call};
use mapal_ir::{Dest, FuncKind, IrBuilder, SourceLoc, Ty, Value, validate};

const L: SourceLoc = SourceLoc { start: 0, end: 0 };

fn check(input: Value, target: Ty, expected: Value) {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "widen", input.ty(), target.clone(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let src = fb.input();
        let widened = fb.widen(src, target, Dest::Fresh(None), L).unwrap();
        fb.output(widened, None, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    assert_eq!(
        eval_call(&ir, f, RValue::Scalar(input), 10),
        Outcome::Done(RValue::Scalar(expected))
    );
}

#[test]
fn widen_value_contract() {
    check(Value::I32(i32::MIN), Ty::i64(), Value::I64(i32::MIN as i64));
    check(
        Value::I32(16_777_217),
        Ty::f32(),
        Value::F32(16_777_217_i32 as f32),
    );
    check(
        Value::I32(-2_000_000_001),
        Ty::f64(),
        Value::F64(-2_000_000_001_i32 as f64),
    );
    check(Value::F32(0.1), Ty::f64(), Value::F64(0.1_f32 as f64));
}

/// ADR-0042: the interpreter's `exp` is `mapal_ir::exp`'s algorithm, at both
/// widths, the special values included.
#[test]
fn exp_value_contract() {
    let one = |input: Value| {
        let mut b = IrBuilder::new();
        let f = b
            .declare(FuncKind::Named, "e", input.ty(), input.ty(), L)
            .unwrap();
        {
            let mut fb = b.build_fn(f).unwrap();
            let src = fb.input();
            let e = fb.exp(src, Dest::Fresh(None), L).unwrap();
            fb.output(e, None, L).unwrap();
            fb.finish().unwrap();
        }
        let ir = b.seal(f).unwrap();
        assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
        match eval_call(&ir, f, RValue::Scalar(input), 10) {
            Outcome::Done(RValue::Scalar(v)) => v,
            o => panic!("{o:?}"),
        }
    };
    for x in [0.0f32, 1.0, -1.0, 88.0, 89.0, -100.0, -104.0, 1e-30] {
        assert_eq!(one(Value::F32(x)), Value::F32(mapal_ir::exp::exp_f32(x)));
    }
    for x in [0.0f64, 1.0, -1.0, 709.0, 710.0, -740.0, -746.0] {
        assert_eq!(one(Value::F64(x)), Value::F64(mapal_ir::exp::exp_f64(x)));
    }
    assert_eq!(one(Value::F32(0.0)), Value::F32(1.0));
    assert_eq!(one(Value::F32(f32::INFINITY)), Value::F32(f32::INFINITY));
    assert_eq!(one(Value::F64(f64::NEG_INFINITY)), Value::F64(0.0));
    assert!(matches!(one(Value::F32(f32::NAN)), Value::F32(x) if x.is_nan()));
}

/// ADR-0046: the interpreter's `sqrt`, `sin` and `cos` are the references
/// (`mapal_ir::math`) at both widths, the special values included — `sqrt`
/// the correctly rounded root, `sin` / `cos` of huge arguments exactly
/// reduced.
#[test]
fn math_value_contract() {
    use mapal_ir::MathFn;
    let one = |f: MathFn, input: Value| {
        let mut b = IrBuilder::new();
        let g = b
            .declare(FuncKind::Named, "m", input.ty(), input.ty(), L)
            .unwrap();
        {
            let mut fb = b.build_fn(g).unwrap();
            let src = fb.input();
            let e = fb.math(f, src, Dest::Fresh(None), L).unwrap();
            fb.output(e, None, L).unwrap();
            fb.finish().unwrap();
        }
        let ir = b.seal(g).unwrap();
        assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
        match eval_call(&ir, g, RValue::Scalar(input), 10) {
            Outcome::Done(RValue::Scalar(v)) => v,
            o => panic!("{o:?}"),
        }
    };
    for f in [MathFn::Sqrt, MathFn::Sin, MathFn::Cos] {
        for x in [0.0f32, -0.0, 1.0, -1.0, 2.0, 1e-40, 3.0e38, 1e30, 102.10176] {
            let want = mapal_ir::math::eval_f32(f, x);
            let got = one(f, Value::F32(x));
            assert!(
                matches!(got, Value::F32(g) if g.to_bits() == want.to_bits() || (g.is_nan() && want.is_nan())),
                "{f:?}({x}) = {got:?}, want {want}"
            );
        }
        for x in [
            0.0f64,
            1.0,
            -2.5,
            1e300,
            5e-324,
            6381956970095103.0 * 2f64.powi(797),
        ] {
            let want = mapal_ir::math::eval_f64(f, x);
            let got = one(f, Value::F64(x));
            assert!(
                matches!(got, Value::F64(g) if g.to_bits() == want.to_bits() || (g.is_nan() && want.is_nan())),
                "{f:?}({x}) = {got:?}"
            );
        }
    }
    assert_eq!(one(MathFn::Sqrt, Value::F32(2.0)), Value::F32(2f32.sqrt()));
    assert!(matches!(one(MathFn::Sqrt, Value::F32(-1.0)), Value::F32(x) if x.is_nan()));
    assert!(
        matches!(one(MathFn::Sqrt, Value::F64(-0.0)), Value::F64(x) if x == 0.0 && x.is_sign_negative())
    );
    assert!(matches!(one(MathFn::Sin, Value::F32(f32::INFINITY)), Value::F32(x) if x.is_nan()));
    assert!(
        matches!(one(MathFn::Sin, Value::F32(-0.0)), Value::F32(x) if x == 0.0 && x.is_sign_negative())
    );
    assert_eq!(one(MathFn::Cos, Value::F64(0.0)), Value::F64(1.0));
}
