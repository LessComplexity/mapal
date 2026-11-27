//! Tests of the captured map / fold constructors and the erased-capture rejections (ADR-0027).

use super::*;

// --- ADR-0027: captured Map/Fold constructors --------------------------------

/// A MapBody `(i32, i32) -> i32` (capture + elem, added).
fn declare_map_caps(b: &mut IrBuilder) -> FuncId {
    let body_in = Ty::Tuple(vec![Ty::i32(), Ty::i32()]);
    let body = b
        .declare(FuncKind::MapBody, "mbc", body_in, Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(body).unwrap();
        let pin = fb.input();
        let c = fb.proj(pin, 0, Dest::Fresh(None), L).unwrap();
        let x = fb.proj(pin, 1, Dest::Fresh(None), L).unwrap();
        fb.binop(Operation::Add, c, x, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    body
}

/// A FoldBody `(i32, i32, i32) -> i32` (capture + acc + elem).
fn declare_fold_caps(b: &mut IrBuilder) -> FuncId {
    let body_in = Ty::Tuple(vec![Ty::i32(), Ty::i32(), Ty::i32()]);
    let body = b
        .declare(FuncKind::FoldBody, "fbc", body_in, Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(body).unwrap();
        let pin = fb.input();
        let c = fb.proj(pin, 0, Dest::Fresh(None), L).unwrap();
        let acc = fb.proj(pin, 1, Dest::Fresh(None), L).unwrap();
        let x = fb.proj(pin, 2, Dest::Fresh(None), L).unwrap();
        let s = fb
            .binop(Operation::Add, acc, x, Dest::Fresh(None), L)
            .unwrap();
        fb.binop(Operation::Add, s, c, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    body
}

#[test]
fn map_captured_roundtrip_validates() {
    use crate::validate::validate;
    let mut b = IrBuilder::new();
    let body = declare_map_caps(&mut b);
    let f = b
        .declare(
            FuncKind::Named,
            "main",
            Ty::Tuple(vec![Ty::i32(), i32_arr(3)]),
            i32_arr(3),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let p = fb.input();
        let cap = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
        let a = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
        fb.map_captured(body, &[cap], a, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty());
}

#[test]
fn map_captured_capture_type_mismatch_rejects() {
    let mut b = IrBuilder::new();
    let body = declare_map_caps(&mut b);
    let f = b
        .declare(
            FuncKind::Named,
            "main",
            Ty::Tuple(vec![Ty::f32(), i32_arr(3)]),
            i32_arr(3),
            L,
        )
        .unwrap();
    let mut fb = b.build_fn(f).unwrap();
    let p = fb.input();
    let cap = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
    let a = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
    let e = fb
        .map_captured(body, &[cap], a, Dest::Fresh(None), L)
        .unwrap_err();
    assert!(matches!(e, IrError::TypeMismatch { .. }));
}

#[test]
fn map_captured_elem_mismatch_rejects() {
    // Capture type is right, but the array's element doesn't match the body
    // input's last component.
    let mut b = IrBuilder::new();
    let body = declare_map_caps(&mut b);
    let f = b
        .declare(
            FuncKind::Named,
            "main",
            Ty::Tuple(vec![
                Ty::i32(),
                Ty::Array {
                    elem: Box::new(Ty::f32()),
                    size: 3,
                },
            ]),
            Ty::Array {
                elem: Box::new(Ty::i32()),
                size: 3,
            },
            L,
        )
        .unwrap();
    let mut fb = b.build_fn(f).unwrap();
    let p = fb.input();
    let cap = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
    let a = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
    let e = fb
        .map_captured(body, &[cap], a, Dest::Fresh(None), L)
        .unwrap_err();
    assert!(matches!(e, IrError::TypeMismatch { .. }));
}

#[test]
fn map_captured_empty_delegates_to_map() {
    let mut b = IrBuilder::new();
    let body = declare_map_caps(&mut b);
    let f = b
        .declare(FuncKind::Named, "main", i32_arr(3), i32_arr(3), L)
        .unwrap();
    let mut fb = b.build_fn(f).unwrap();
    let a = fb.input();
    // Empty caps on a body whose input is (i32, i32) must behave as map():
    // the delegation checks body input == elem, so this one errors — the
    // k=0 path does not special-case. Pin the EXACT error: the body's
    // declared input as `expected`, the array element as `found`.
    let e = fb.map_captured(body, &[], a, Dest::Fresh(None), L);
    assert_eq!(
        e,
        Err(IrError::TypeMismatch {
            expected: Ty::Tuple(vec![Ty::i32(), Ty::i32()]),
            found: Ty::i32(),
            loc: L,
        })
    );
}

#[test]
fn fold_captured_roundtrip_validates() {
    use crate::validate::validate;
    let mut b = IrBuilder::new();
    let body = declare_fold_caps(&mut b);
    let f = b
        .declare(
            FuncKind::Named,
            "main",
            Ty::Tuple(vec![Ty::i32(), Ty::i32(), i32_arr(3)]),
            Ty::i32(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let p = fb.input();
        let cap = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
        let seed = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
        let a = fb.proj(p, 2, Dest::Fresh(None), L).unwrap();
        fb.fold_captured(body, &[cap], seed, a, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty());
}

#[test]
fn fold_captured_seed_mismatch_rejects() {
    let mut b = IrBuilder::new();
    let body = declare_fold_caps(&mut b);
    let f = b
        .declare(
            FuncKind::Named,
            "main",
            Ty::Tuple(vec![Ty::i32(), Ty::f32(), i32_arr(3)]),
            Ty::i32(),
            L,
        )
        .unwrap();
    let mut fb = b.build_fn(f).unwrap();
    let p = fb.input();
    let cap = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
    let seed = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
    let a = fb.proj(p, 2, Dest::Fresh(None), L).unwrap();
    let e = fb
        .fold_captured(body, &[cap], seed, a, Dest::Fresh(None), L)
        .unwrap_err();
    assert!(matches!(e, IrError::TypeMismatch { .. }));
}

// --- ADR-0027 review major #5: erased captures are rejected ------------------

/// A MapBody `(Unit, i32) -> i32` (Unit capture + elem): the body ignores the
/// capture and doubles the element.
fn declare_map_unit_cap(b: &mut IrBuilder) -> FuncId {
    let body_in = Ty::Tuple(vec![Ty::Unit, Ty::i32()]);
    let body = b
        .declare(FuncKind::MapBody, "muc", body_in, Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(body).unwrap();
        let pin = fb.input();
        let x = fb.proj(pin, 1, Dest::Fresh(None), L).unwrap();
        fb.binop(Operation::Add, x, x, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    body
}

#[test]
fn map_captured_unit_capture_rejects() {
    // A `Unit` capture has no runtime representation: validate-clean before,
    // an LLVM panic on emission — now rejected at the constructor.
    let mut b = IrBuilder::new();
    let body = declare_map_unit_cap(&mut b);
    let f = b
        .declare(
            FuncKind::Named,
            "main",
            Ty::Tuple(vec![Ty::Unit, i32_arr(3)]),
            i32_arr(3),
            L,
        )
        .unwrap();
    let mut fb = b.build_fn(f).unwrap();
    let p = fb.input();
    let cap = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
    let a = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
    let e = fb
        .map_captured(body, &[cap], a, Dest::Fresh(None), L)
        .unwrap_err();
    assert_eq!(e, IrError::ErasedCapture);
}

#[test]
fn map_captured_all_erased_product_capture_rejects() {
    // An all-erased product capture `(Unit, Unit)`: same empty residual.
    let body_in = Ty::Tuple(vec![Ty::Tuple(vec![Ty::Unit, Ty::Unit]), Ty::i32()]);
    let mut b = IrBuilder::new();
    let body = b
        .declare(FuncKind::MapBody, "mpc", body_in, Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(body).unwrap();
        let pin = fb.input();
        let x = fb.proj(pin, 1, Dest::Fresh(None), L).unwrap();
        fb.binop(Operation::Add, x, x, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let f = b
        .declare(
            FuncKind::Named,
            "main",
            Ty::Tuple(vec![Ty::Tuple(vec![Ty::Unit, Ty::Unit]), i32_arr(3)]),
            i32_arr(3),
            L,
        )
        .unwrap();
    let mut fb = b.build_fn(f).unwrap();
    let p = fb.input();
    let cap = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
    let a = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
    let e = fb
        .map_captured(body, &[cap], a, Dest::Fresh(None), L)
        .unwrap_err();
    assert_eq!(e, IrError::ErasedCapture);
}

#[test]
fn fold_captured_unit_capture_rejects() {
    let body_in = Ty::Tuple(vec![Ty::Unit, Ty::i32(), Ty::i32()]);
    let mut b = IrBuilder::new();
    let body = b
        .declare(FuncKind::FoldBody, "fuc", body_in, Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(body).unwrap();
        let pin = fb.input();
        let acc = fb.proj(pin, 1, Dest::Fresh(None), L).unwrap();
        let x = fb.proj(pin, 2, Dest::Fresh(None), L).unwrap();
        fb.binop(Operation::Add, acc, x, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let f = b
        .declare(
            FuncKind::Named,
            "main",
            Ty::Tuple(vec![Ty::Unit, Ty::i32(), i32_arr(3)]),
            Ty::i32(),
            L,
        )
        .unwrap();
    let mut fb = b.build_fn(f).unwrap();
    let p = fb.input();
    let cap = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
    let seed = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
    let a = fb.proj(p, 2, Dest::Fresh(None), L).unwrap();
    let e = fb
        .fold_captured(body, &[cap], seed, a, Dest::Fresh(None), L)
        .unwrap_err();
    assert_eq!(e, IrError::ErasedCapture);
}

#[test]
fn map_captured_partially_erased_product_capture_still_accepted() {
    // A capture with a SURVIVING component `(Unit, i32)` keeps a
    // representation (residual arity 1) — not the erased-capture hole.
    let body_in = Ty::Tuple(vec![Ty::Tuple(vec![Ty::Unit, Ty::i32()]), Ty::i32()]);
    let mut b = IrBuilder::new();
    let body = b
        .declare(FuncKind::MapBody, "mppc", body_in, Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(body).unwrap();
        let pin = fb.input();
        let x = fb.proj(pin, 1, Dest::Fresh(None), L).unwrap();
        fb.binop(Operation::Add, x, x, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let f = b
        .declare(
            FuncKind::Named,
            "main",
            Ty::Tuple(vec![Ty::Tuple(vec![Ty::Unit, Ty::i32()]), i32_arr(3)]),
            i32_arr(3),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let p = fb.input();
        let cap = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
        let a = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
        fb.map_captured(body, &[cap], a, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(crate::validate::validate(&ir).is_empty());
}
