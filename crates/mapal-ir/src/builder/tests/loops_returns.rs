//! Rejection-matrix tests: loops, the return object, seal entry errors and a clean end-to-end build.

use super::*;

// --- loops ----------------------------------------------------------------

#[test]
fn open_loop() {
    // A LoopHandle never end_loop'd → OpenLoop at finish.
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let _lh = fb.begin_loop(x, L).unwrap();
    let e = fb.finish().unwrap_err();
    assert_eq!(e, IrError::OpenLoop);
}

#[test]
fn loop_without_back() {
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let lh = fb.begin_loop(x, L).unwrap();
    let merge = fb.merge_of(&lh);
    let cond = fb.constant(Value::Bool(false), L).unwrap();
    // exit but no back.
    fb.loop_exit(&lh, merge, cond, Dest::Ret { slot: None }, L)
        .unwrap();
    let e = fb.end_loop(lh).unwrap_err();
    assert_eq!(e, IrError::LoopWithoutBack);
}

#[test]
fn loop_without_exit() {
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let lh = fb.begin_loop(x, L).unwrap();
    let merge = fb.merge_of(&lh);
    let one = fb.constant(Value::I32(1), L).unwrap();
    let next = fb
        .binop(Operation::Add, merge, one, Dest::Fresh(None), L)
        .unwrap();
    let cond = fb.constant(Value::Bool(true), L).unwrap();
    fb.loop_back(&lh, next, cond, L).unwrap();
    let e = fb.end_loop(lh).unwrap_err();
    assert_eq!(e, IrError::LoopWithoutExit);
}

#[test]
fn loop_back_outside_scc() {
    // A second loop_back whose source is parameter-derived (outside the SCC).
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let lh = fb.begin_loop(x, L).unwrap();
    let merge = fb.merge_of(&lh);
    let one = fb.constant(Value::I32(1), L).unwrap();
    let next = fb
        .binop(Operation::Add, merge, one, Dest::Fresh(None), L)
        .unwrap();
    let cond = fb.constant(Value::Bool(true), L).unwrap();
    fb.loop_back(&lh, next, cond, L).unwrap();
    // A second back edge whose source (x) is NOT in the SCC.
    fb.loop_back(&lh, x, cond, L).unwrap();
    // exit so end_loop passes locally.
    fb.loop_exit(&lh, merge, cond, Dest::Ret { slot: None }, L)
        .unwrap();
    fb.end_loop(lh).unwrap();
    fb.finish().unwrap();
    let e = b.seal(f).unwrap_err();
    assert_eq!(e, IrError::LoopBackOutsideScc);
}

// --- I-RET ----------------------------------------------------------------

#[test]
fn ret_not_product_slot_on_scalar() {
    // ret slot write on scalar output → RetNotProduct.
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let one = fb.constant(Value::I32(1), L).unwrap();
    let e = fb
        .binop(Operation::Add, x, one, Dest::Ret { slot: Some(0) }, L)
        .unwrap_err();
    assert_eq!(e, IrError::RetNotProduct);
}

#[test]
fn ret_type_mismatch() {
    let (mut b, f) = one_fn(Ty::i32(), Ty::Bool);
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let one = fb.constant(Value::I32(1), L).unwrap();
    // Add → i32, but output is Bool.
    let e = fb
        .binop(Operation::Add, x, one, Dest::Ret { slot: None }, L)
        .unwrap_err();
    assert_eq!(e, IrError::RetTypeMismatch);
}

#[test]
fn ret_mixed_writers() {
    // A full writer AND a slot writer to the same Return → RetMixedWriters.
    let out = Ty::Tuple(vec![Ty::i32(), Ty::i32()]);
    let (mut b, f) = one_fn(Ty::i32(), out);
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    // Full write of the whole tuple.
    let full = fb.pack(&[x, x], Dest::Fresh(None), L).unwrap();
    fb.output(full, None, L).unwrap();
    // Slot write of component 0.
    fb.output(x, Some(0), L).unwrap();
    let e = fb.finish().unwrap_err();
    assert_eq!(e, IrError::RetMixedWriters);
}

#[test]
fn ret_slot_conflict() {
    let out = Ty::Tuple(vec![Ty::i32(), Ty::i32()]);
    let (mut b, f) = one_fn(Ty::i32(), out);
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    fb.output(x, Some(0), L).unwrap();
    fb.output(x, Some(0), L).unwrap(); // same slot twice.
    let e = fb.finish().unwrap_err();
    assert_eq!(e, IrError::RetSlotConflict);
}

#[test]
fn ret_slot_missing() {
    let out = Ty::Tuple(vec![Ty::i32(), Ty::i32()]);
    let (mut b, f) = one_fn(Ty::i32(), out);
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    fb.output(x, Some(0), L).unwrap(); // slot 1 missing.
    let e = fb.finish().unwrap_err();
    assert_eq!(e, IrError::RetSlotMissing);
}

// --- seal entry errors ----------------------------------------------------

#[test]
fn no_entry() {
    let b = IrBuilder::new();
    // Seal with a default (never-issued) FuncId.
    let e = b.seal(FuncId::default()).unwrap_err();
    assert_eq!(e, IrError::NoEntry);
}

#[test]
fn entry_not_named() {
    let mut b = IrBuilder::new();
    let body = b
        .declare(FuncKind::MapBody, "mb", Ty::i32(), Ty::i32(), L)
        .unwrap();
    {
        let fb = b.build_fn(body).unwrap();
        let inp = fb.input();
        let mut fb = fb;
        fb.output(inp, None, L).unwrap();
        fb.finish().unwrap();
    }
    let e = b.seal(body).unwrap_err();
    assert_eq!(e, IrError::EntryNotNamed);
}

// --- token-not-escaping (loop U carries token, exit drops it) -------------

#[test]
fn token_not_escaping() {
    // A loop whose U is (i32, IoToken) but whose exit payload is i32 (drops the
    // token) → TokenNotEscaping (token-in ⇒ token-out, §8). Output is Unit so
    // I-RET is trivially satisfied and seal reaches the loop check.
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", Ty::IoToken, Ty::Unit, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let t0 = fb.input();
        let zero = fb.constant(Value::I32(0), L).unwrap();
        let init = fb.pack(&[zero, t0], Dest::Fresh(None), L).unwrap();
        let lh = fb.begin_loop(init, L).unwrap();
        let merge = fb.merge_of(&lh);
        // Recompute next = same U (project then repack to stay in SCC).
        let i = fb.proj(merge, 0, Dest::Fresh(None), L).unwrap();
        let tok = fb.proj(merge, 1, Dest::Fresh(None), L).unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        let i2 = fb
            .binop(Operation::Add, i, one, Dest::Fresh(None), L)
            .unwrap();
        let next = fb.pack(&[i2, tok], Dest::Fresh(None), L).unwrap();
        let cond = fb.constant(Value::Bool(true), L).unwrap();
        fb.loop_back(&lh, next, cond, L).unwrap();
        // Exit payload is i32 (drops the token) — the violation.
        let exit_i = fb.proj(merge, 0, Dest::Fresh(None), L).unwrap();
        let _out = fb
            .loop_exit(&lh, exit_i, cond, Dest::Fresh(None), L)
            .unwrap();
        fb.end_loop(lh).unwrap();
        fb.finish().unwrap();
    }
    let e = b.seal(f).unwrap_err();
    // The dropped token violates both the escape rule and the sink (I4b);
    // either is the correct rejection.
    assert!(matches!(
        e,
        IrError::TokenNotEscaping | IrError::TokenDropped
    ));
}

// --- a clean end-to-end build validates -----------------------------------

#[test]
fn clean_add_builds_and_validates() {
    use crate::validate::validate;
    // a + b -> ret with two params packed as input (i32, i32) → i32.
    let input = Ty::Tuple(vec![Ty::i32(), Ty::i32()]);
    let (mut b, f) = one_fn(input, Ty::i32());
    {
        let mut fb = b.build_fn(f).unwrap();
        let p = fb.input();
        let a = fb.proj(p, 0, Dest::Fresh(Some("a".into())), L).unwrap();
        let bb = fb.proj(p, 1, Dest::Fresh(Some("b".into())), L).unwrap();
        fb.binop(Operation::Add, a, bb, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty());
}
