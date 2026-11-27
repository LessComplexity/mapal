//! Rejection-matrix tests: declarations, function graphs, operator membership, type intake and products.

use super::*;
use crate::ty::MAX_TY_DEPTH;

// --- declaration / function-graph errors ----------------------------------

#[test]
fn duplicate_name() {
    let mut b = IrBuilder::new();
    b.declare(FuncKind::Named, "f", Ty::Unit, Ty::Unit, L)
        .unwrap();
    let e = b
        .declare(FuncKind::Named, "f", Ty::Unit, Ty::Unit, L)
        .unwrap_err();
    assert_eq!(e, IrError::DuplicateName);
}

#[test]
fn unknown_function_on_build() {
    // A FuncId never issued in this builder resolves to UnknownFunction. We
    // forge one by allocating many funcs in a foreign builder so its key index
    // is beyond anything this builder issued (slotmap version tags reset across
    // instances, so divergent slot *indices* are the reliable separator — the
    // DESIGN §10 "mixing builders is unsupported" note made concrete).
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "f", Ty::i32(), Ty::i32(), L)
        .unwrap();
    let mut b2 = IrBuilder::new();
    let mut g = b2
        .declare(FuncKind::Named, "g0", Ty::i32(), Ty::i32(), L)
        .unwrap();
    for i in 1..16 {
        g = b2
            .declare(FuncKind::Named, &format!("g{i}"), Ty::i32(), Ty::i32(), L)
            .unwrap();
    }
    let mut fb = b.build_fn(f).unwrap();
    let arg = fb.input();
    let e = fb.call(g, arg, Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e, IrError::UnknownFunction);
}

#[test]
fn already_built() {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "f", Ty::Unit, Ty::Unit, L)
        .unwrap();
    {
        let _fb = b.build_fn(f).unwrap();
    }
    match b.build_fn(f) {
        Err(IrError::AlreadyBuilt) => {}
        other => panic!("expected AlreadyBuilt, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn unfinished_function() {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", Ty::Unit, Ty::Unit, L)
        .unwrap();
    // Declared, never built.
    let e = b.seal(f).unwrap_err();
    assert_eq!(e, IrError::UnfinishedFunction);
}

#[test]
fn unknown_object_from_foreign_builder() {
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    // A foreign builder that has minted many more objects, so the id we pull is
    // a slot index this builder never allocated → UnknownObject.
    let mut b2 = IrBuilder::new();
    let f2 = b2
        .declare(FuncKind::Named, "m2", Ty::i32(), Ty::i32(), L)
        .unwrap();
    let foreign = {
        let mut fb2 = b2.build_fn(f2).unwrap();
        let mut last = fb2.input();
        for _ in 0..32 {
            last = fb2.constant(Value::I32(1), L).unwrap();
        }
        last
    };
    let mut fb = b.build_fn(f).unwrap();
    let e = fb
        .unop(Operation::Neg, foreign, Dest::Fresh(None), L)
        .unwrap_err();
    assert_eq!(e, IrError::UnknownObject);
}

#[test]
fn wrong_builder_cross_function() {
    // An object owned by function g, used inside function f.
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "f", Ty::i32(), Ty::i32(), L)
        .unwrap();
    let g = b
        .declare(FuncKind::Named, "g", Ty::i32(), Ty::i32(), L)
        .unwrap();
    let g_input = {
        let fb = b.build_fn(g).unwrap();
        let inp = fb.input();
        // finish g properly so it's not Unfinished later (not needed for this test).
        inp
    };
    let mut fb = b.build_fn(f).unwrap();
    let e = fb
        .unop(Operation::Neg, g_input, Dest::Fresh(None), L)
        .unwrap_err();
    assert_eq!(e, IrError::WrongBuilder);
}

// --- unop / binop membership ----------------------------------------------

#[test]
fn not_unary() {
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let e = fb
        .unop(Operation::Add, x, Dest::Fresh(None), L)
        .unwrap_err();
    assert_eq!(e, IrError::NotUnary);
}

#[test]
fn not_binary() {
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let e = fb
        .binop(Operation::Neg, x, x, Dest::Fresh(None), L)
        .unwrap_err();
    assert_eq!(e, IrError::NotBinary);
}

#[test]
fn binop_eq_bool_ok_lt_bool_mismatch() {
    // The reviewer-named hole: Eq admits Bool, Lt does not.
    let (mut b, f) = one_fn(Ty::Bool, Ty::Bool);
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let t = fb.constant(Value::Bool(true), L).unwrap();
    // Eq(bool, bool) → Ok.
    let ok = fb.binop(Operation::Eq, x, t, Dest::Fresh(None), L);
    assert!(ok.is_ok());
    // Lt(bool, bool) → TypeMismatch.
    let e = fb
        .binop(Operation::Lt, x, t, Dest::Fresh(None), L)
        .unwrap_err();
    assert!(matches!(e, IrError::TypeMismatch { .. }));
}

#[test]
fn type_mismatch_on_mixed_numeric() {
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let y = fb.constant(Value::I64(1), L).unwrap();
    let e = fb
        .binop(Operation::Add, x, y, Dest::Fresh(None), L)
        .unwrap_err();
    assert!(matches!(e, IrError::TypeMismatch { .. }));
}

// --- type intake (I9/I10) -------------------------------------------------

#[test]
fn non_core_type_bad_int_width() {
    let mut b = IrBuilder::new();
    let bad = Ty::Int {
        bits: 16,
        signed: true,
    };
    let e = b
        .declare(FuncKind::Named, "m", bad, Ty::Unit, L)
        .unwrap_err();
    assert_eq!(e, IrError::NonCoreType);
}

#[test]
fn non_core_singleton_tuple_ty() {
    let mut b = IrBuilder::new();
    let bad = Ty::Tuple(vec![Ty::i32()]);
    let e = b
        .declare(FuncKind::Named, "m", bad, Ty::Unit, L)
        .unwrap_err();
    assert_eq!(e, IrError::NonCoreType);
}

#[test]
fn non_core_zero_field_struct_ty() {
    // TY-1 (layer 1): a zero-field `Ty::Struct` is product `1` whose literal mints
    // an in-edge-less Temporary — it seals but fails validate (BadInEdges). Intake
    // rejects it as NonCoreType (mirrors Tuple arity ≥ 2 / Array size ≥ 1), making
    // it unconstructible anywhere a ty enters the builder. Asserted at the declare
    // entry point (input ty here; the output path shares `intake_ty`).
    let mut b = IrBuilder::new();
    let bad = Ty::Struct {
        name: "Empty".into(),
        fields: vec![],
    };
    let e = b
        .declare(FuncKind::Named, "m", bad, Ty::Unit, L)
        .unwrap_err();
    assert_eq!(e, IrError::NonCoreType);
}

#[test]
fn ty_too_deep() {
    let mut t = Ty::i32();
    for _ in 0..MAX_TY_DEPTH + 2 {
        t = Ty::Array {
            elem: Box::new(t),
            size: 1,
        };
    }
    let mut b = IrBuilder::new();
    let e = b.declare(FuncKind::Named, "m", t, Ty::Unit, L).unwrap_err();
    assert_eq!(e, IrError::TyTooDeep);
}

// --- products: proj / pack / pack_array / pack_struct ---------------------

#[test]
fn not_a_product_on_proj() {
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let e = fb.proj(x, 0, Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e, IrError::NotAProduct);
}

#[test]
fn slot_out_of_range_on_proj() {
    let (mut b, f) = one_fn(Ty::Tuple(vec![Ty::i32(), Ty::Bool]), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let e = fb.proj(x, 5, Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e, IrError::SlotOutOfRange);
}

#[test]
fn empty_product_pack() {
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let e = fb.pack(&[], Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e, IrError::EmptyProduct);
    let e2 = fb.pack_array(&[], Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e2, IrError::EmptyProduct);
}

#[test]
fn singleton_tuple() {
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let e = fb.pack(&[x], Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e, IrError::SingletonTuple);
}

#[test]
fn arity_mismatch_pack_struct() {
    let sty = Ty::Struct {
        name: "P".into(),
        fields: vec![("a".into(), Ty::i32()), ("b".into(), Ty::i32())],
    };
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let e = fb.pack_struct(sty, &[x], Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e, IrError::ArityMismatch);
}

#[test]
fn empty_product_pack_struct() {
    // TY-1 (layer 2): zero components → EmptyProduct, mirroring `pack`. This is the
    // defense-in-depth guard for the in-edge-less-Temporary hole. The EmptyProduct
    // check precedes both the arity check and `intake_ty`, so it fires for ANY zero
    // component list regardless of the named struct's declared field count — here
    // the struct ty even has fields (a genuine arity mismatch), yet the empty
    // component list is reported first as EmptyProduct. (A zero-FIELD struct ty is
    // already unconstructible via layer 1's intake; this guard is independent of it.)
    let sty = Ty::Struct {
        name: "P".into(),
        fields: vec![("a".into(), Ty::i32())],
    };
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let e = fb.pack_struct(sty, &[], Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e, IrError::EmptyProduct);
}

#[test]
fn value_ty_mismatch_is_unreachable_constant_is_sound() {
    // constant() always sets ty := v.ty(), so ValueTyMismatch cannot arise from
    // the public API. We assert constants are internally consistent (I7) and
    // document ValueTyMismatch as a defense for future direct-value APIs.
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let c = fb.constant(Value::I32(7), L).unwrap();
    assert_eq!(fb.b.objects[c].ty, Ty::i32());
    assert_eq!(fb.b.objects[c].value, Some(Value::I32(7)));
}
