//! Tests of zip, enumerate and update (ADR-0018, ADR-0021).

use super::*;

// --- zip / enumerate (ADR-0018) -------------------------------------------

#[test]
fn zip_builds_and_validates() {
    use crate::validate::validate;
    // main : ([i32;3], [i32;3]) -> [(i32,i32);3]
    let input = Ty::Tuple(vec![i32_arr(3), i32_arr(3)]);
    let out = Ty::Array {
        elem: Box::new(Ty::Tuple(vec![Ty::i32(), Ty::i32()])),
        size: 3,
    };
    let (mut b, f) = one_fn(input, out);
    {
        let mut fb = b.build_fn(f).unwrap();
        let p = fb.input();
        let a = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
        let bb = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
        fb.zip(a, bb, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty());
}

#[test]
fn enumerate_builds_and_validates() {
    use crate::validate::validate;
    // main : [f32;3] -> [(i32,f32);3]
    let input = Ty::Array {
        elem: Box::new(Ty::f32()),
        size: 3,
    };
    let out = Ty::Array {
        elem: Box::new(Ty::Tuple(vec![Ty::i32(), Ty::f32()])),
        size: 3,
    };
    let (mut b, f) = one_fn(input, out);
    {
        let mut fb = b.build_fn(f).unwrap();
        let a = fb.input();
        fb.enumerate(a, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty());
}

#[test]
fn zip_result_feeds_map() {
    use crate::validate::validate;
    // zip two [i32;3] then map (i32,i32)->i32 into ret.
    let mut b = IrBuilder::new();
    let pair = Ty::Tuple(vec![Ty::i32(), Ty::i32()]);
    let body = b
        .declare(FuncKind::MapBody, "mb", pair.clone(), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(body).unwrap();
        let pin = fb.input();
        let x = fb.proj(pin, 0, Dest::Fresh(None), L).unwrap();
        let y = fb.proj(pin, 1, Dest::Fresh(None), L).unwrap();
        fb.binop(Operation::Add, x, y, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let input = Ty::Tuple(vec![i32_arr(3), i32_arr(3)]);
    let f = b
        .declare(FuncKind::Named, "main", input, i32_arr(3), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let p = fb.input();
        let a = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
        let bb = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
        let z = fb.zip(a, bb, Dest::Fresh(None), L).unwrap();
        fb.map(body, z, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty());
}

#[test]
fn zip_size_mismatch_rejects() {
    let input = Ty::Tuple(vec![i32_arr(3), i32_arr(4)]);
    let out = Ty::Array {
        elem: Box::new(Ty::Tuple(vec![Ty::i32(), Ty::i32()])),
        size: 3,
    };
    let (mut b, f) = one_fn(input, out);
    let mut fb = b.build_fn(f).unwrap();
    let p = fb.input();
    let a = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
    let bb = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
    let e = fb.zip(a, bb, Dest::Fresh(None), L).unwrap_err();
    assert!(matches!(e, IrError::TypeMismatch { .. }));
}

#[test]
fn zip_non_array_rejects() {
    let (mut b, f) = one_fn(Ty::Tuple(vec![Ty::i32(), Ty::i32()]), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let p = fb.input();
    let a = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
    let bb = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
    let e = fb.zip(a, bb, Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e, IrError::NotAProduct);
}

#[test]
fn enumerate_non_array_rejects() {
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let e = fb.enumerate(x, Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e, IrError::NotAProduct);
}

#[test]
fn enumerate_oversize_rejects() {
    // An array of size > i32::MAX cannot be enumerated (index would overflow i32).
    let big = i32::MAX as u64 + 1;
    let (mut b, f) = one_fn(i32_arr(big), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let a = fb.input();
    let e = fb.enumerate(a, Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e, IrError::EnumerateIndexOverflow);
}

// --- Update (ADR-0021): (Array{T,n}, I, T) -> Array{T,n} -------------------

#[test]
fn update_builds_and_validates() {
    use crate::validate::validate;
    // main : [i32;3] -> [i32;3]; c[i] <- v with i, v constants.
    let (mut b, f) = one_fn(i32_arr(3), i32_arr(3));
    {
        let mut fb = b.build_fn(f).unwrap();
        let a = fb.input();
        let i = fb.constant(Value::I32(1), L).unwrap();
        let v = fb.constant(Value::I32(99), L).unwrap();
        fb.update(a, i, v, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
}

#[test]
fn update_non_array_rejects() {
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let i = fb.constant(Value::I32(0), L).unwrap();
    let e = fb.update(x, i, x, Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e, IrError::NotAProduct);
}

#[test]
fn update_index_not_int_rejects() {
    let (mut b, f) = one_fn(i32_arr(3), i32_arr(3));
    let mut fb = b.build_fn(f).unwrap();
    let a = fb.input();
    let bad = fb.constant(Value::F32(1.0), L).unwrap();
    let v = fb.constant(Value::I32(0), L).unwrap();
    let e = fb.update(a, bad, v, Dest::Fresh(None), L).unwrap_err();
    assert!(matches!(e, IrError::TypeMismatch { .. }));
}

#[test]
fn update_value_elem_mismatch_rejects() {
    let (mut b, f) = one_fn(i32_arr(3), i32_arr(3));
    let mut fb = b.build_fn(f).unwrap();
    let a = fb.input();
    let i = fb.constant(Value::I32(0), L).unwrap();
    let bad = fb.constant(Value::F32(1.0), L).unwrap();
    let e = fb.update(a, i, bad, Dest::Fresh(None), L).unwrap_err();
    assert!(matches!(e, IrError::TypeMismatch { .. }));
}
