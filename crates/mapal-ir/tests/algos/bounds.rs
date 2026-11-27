//! `bounds_proof` tests: provably in-bounds `Index`.

use super::*;

// --- bounds_proof (the S20 kernel-gap analysis: provably-in-bounds Index) -----

pub(super) fn i32_arr(n: u64) -> Ty {
    Ty::Array {
        elem: Box::new(Ty::i32()),
        size: n,
    }
}

pub(super) fn index_morphs(
    ir: &mapal_ir::CategoryIr,
    f: mapal_ir::FuncId,
) -> Vec<mapal_ir::MorphismId> {
    ir.morphisms()
        .filter(|(m, _)| ir.morphism(*m).unwrap().op == Operation::Index)
        .map(|(m, _)| m)
        .filter(|&m| {
            let mm = ir.morphism(m).unwrap();
            ir.try_owner(mm.source) == Some(f) || ir.try_owner(mm.target) == Some(f)
        })
        .collect()
}

#[test]
fn bounds_constant_index_in_range_proven_out_not() {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "m", i32_arr(16), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let a = fb.input();
        let three = fb.constant(Value::I32(3), L).unwrap();
        let x = fb.index(a, three, Dest::Fresh(None), L).unwrap();
        let twenty = fb.constant(Value::I32(20), L).unwrap();
        let y = fb.index(a, twenty, Dest::Fresh(None), L).unwrap();
        fb.binop(Operation::Add, x, y, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    let bp = ir.bounds_proof(f);
    let ms = index_morphs(&ir, f);
    assert_eq!(ms.len(), 2);
    assert!(bp.proven(ms[0]), "constant 3 < 16 proves");
    assert!(
        !bp.proven(ms[1]),
        "constant 20 >= 16 does not (the guard stays)"
    );
}

#[test]
fn bounds_iota_element_ranged() {
    let mut b = IrBuilder::new();
    let f = b
        .declare(
            FuncKind::Named,
            "m",
            Ty::Tuple(vec![i32_arr(16), Ty::Unit]),
            Ty::i32(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let inp = fb.input();
        let a = fb.proj(inp, 0, Dest::Fresh(None), L).unwrap();
        let n = fb.constant(Value::I32(16), L).unwrap();
        let tr = fb.iota(n, Dest::Fresh(None), L).unwrap();
        let three = fb.constant(Value::I32(3), L).unwrap();
        let e = fb.index(tr, three, Dest::Fresh(None), L).unwrap();
        fb.index(a, e, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    let bp = ir.bounds_proof(f);
    let ms = index_morphs(&ir, f);
    assert_eq!(ms.len(), 2);
    assert!(
        bp.proven(ms[1]),
        "the iota element [0,16) is in-bounds of [i32;16]"
    );
}

/// The acceptance shape: an enumerate'd map body with the matmul's affine
/// indexing (`i = t / 4`, `j = t % 4`, `a[i*4+j]`) — proven when `a` is big
/// enough, not when it isn't.
pub(super) fn enumerate_cell(arr_size: u64) -> (mapal_ir::CategoryIr, mapal_ir::FuncId) {
    let mut b = IrBuilder::new();
    let body = b
        .declare(
            FuncKind::MapBody,
            "cell",
            Ty::Tuple(vec![
                i32_arr(arr_size),
                Ty::Tuple(vec![Ty::i32(), Ty::i32()]),
            ]),
            Ty::i32(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(body).unwrap();
        let p = fb.input();
        let cap = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
        let e = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
        let t = fb.proj(e, 0, Dest::Fresh(None), L).unwrap();
        let four = fb.constant(Value::I32(4), L).unwrap();
        let i = fb
            .binop(Operation::Div, t, four, Dest::Fresh(None), L)
            .unwrap();
        let j = fb
            .binop(Operation::Mod, t, four, Dest::Fresh(None), L)
            .unwrap();
        let i4 = fb
            .binop(Operation::Mul, i, four, Dest::Fresh(None), L)
            .unwrap();
        let y = fb
            .binop(Operation::Add, i4, j, Dest::Fresh(None), L)
            .unwrap();
        fb.index(cap, y, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let f = b
        .declare(
            FuncKind::Named,
            "main",
            Ty::Tuple(vec![i32_arr(64), i32_arr(arr_size)]),
            i32_arr(64),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let inp = fb.input();
        let seed = fb.proj(inp, 0, Dest::Fresh(None), L).unwrap();
        let a = fb.proj(inp, 1, Dest::Fresh(None), L).unwrap();
        let en = fb.enumerate(seed, Dest::Fresh(None), L).unwrap();
        fb.map_captured(body, &[a], en, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    (ir, body)
}

#[test]
fn bounds_enumerate_affine_proven_when_sized() {
    let (ir, body) = enumerate_cell(64);
    let bp = ir.bounds_proof(body);
    let ms = index_morphs(&ir, body);
    assert_eq!(ms.len(), 1);
    assert!(bp.proven(ms[0]), "i*4+j <= 15*4+3 = 63 < 64 proves");
}

#[test]
fn bounds_enumerate_affine_not_proven_when_undersized() {
    let (ir, body) = enumerate_cell(32);
    let bp = ir.bounds_proof(body);
    let ms = index_morphs(&ir, body);
    assert_eq!(ms.len(), 1);
    assert!(!bp.proven(ms[0]), "63 >= 32 does not prove (guard stays)");
}

#[test]
fn bounds_literal_ramp_elements() {
    let mut b = IrBuilder::new();
    let f = b
        .declare(
            FuncKind::Named,
            "m",
            Ty::Tuple(vec![i32_arr(4), i32_arr(2)]),
            Ty::i32(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let inp = fb.input();
        let a4 = fb.proj(inp, 0, Dest::Fresh(None), L).unwrap();
        let a2 = fb.proj(inp, 1, Dest::Fresh(None), L).unwrap();
        let z = fb.constant(Value::I32(0), L).unwrap();
        let o = fb.constant(Value::I32(1), L).unwrap();
        let tw = fb.constant(Value::I32(2), L).unwrap();
        let th = fb.constant(Value::I32(3), L).unwrap();
        let kr = fb
            .pack_array(&[z, o, tw, th], Dest::Fresh(None), L)
            .unwrap();
        let e = fb.index(kr, o, Dest::Fresh(None), L).unwrap();
        let x = fb.index(a4, e, Dest::Fresh(None), L).unwrap();
        let y = fb.index(a2, e, Dest::Fresh(None), L).unwrap();
        fb.binop(Operation::Add, x, y, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    let bp = ir.bounds_proof(f);
    let ms = index_morphs(&ir, f);
    assert_eq!(ms.len(), 3);
    assert!(bp.proven(ms[1]), "ramp element [0,3] < 4");
    assert!(!bp.proven(ms[2]), "ramp element [0,3] >= 2 — guard stays");
}

#[test]
fn bounds_wrap_and_negative_bail() {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "m", i32_arr(300), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let a = fb.input();
        // u8: 16 * 17 = 272 > 255 — wraparound possible, range unknown.
        let x = fb.constant(Value::U8(16), L).unwrap();
        let y = fb.constant(Value::U8(17), L).unwrap();
        let m = fb
            .binop(Operation::Mul, x, y, Dest::Fresh(None), L)
            .unwrap();
        // 3 - 4 could wrap negative — range unknown.
        let three = fb.constant(Value::I32(3), L).unwrap();
        let four = fb.constant(Value::I32(4), L).unwrap();
        let d = fb
            .binop(Operation::Sub, three, four, Dest::Fresh(None), L)
            .unwrap();
        let p = fb.index(a, m, Dest::Fresh(None), L).unwrap();
        let q = fb.index(a, d, Dest::Fresh(None), L).unwrap();
        fb.binop(Operation::Add, p, q, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    let bp = ir.bounds_proof(f);
    let ms = index_morphs(&ir, f);
    assert_eq!(ms.len(), 2);
    assert!(!bp.proven(ms[0]), "u8 mul overflow — unknown — not proven");
    assert!(
        !bp.proven(ms[1]),
        "negative-going Sub — unknown — not proven"
    );
}

#[test]
fn bounds_determinism_and_totality() {
    let (ir, body) = enumerate_cell(64);
    let main = ir.entry();
    let a = ir.bounds_proof(body);
    let b2 = ir.bounds_proof(body);
    assert_eq!(a, b2, "same graph → same plan (L2)");
    // totality: every fn in the graph answers, including the entry.
    let _ = ir.bounds_proof(main);
    let _ = ir.bounds_proof(body);
}

/// The real bench shape: a map body computing `i = t/8`, `j = t%8` from the
/// enumerate element, with a nested fold capturing `i`/`j`/arrays over a
/// literal ramp — the fold's `a[i*8+k]`/`b[k*8+j]` reads. The capture-range
/// threading must prove BOTH Index reads (the S20 kernel-gap acceptance).
#[test]
fn bounds_capture_shape_matmul_proven() {
    let mut b = IrBuilder::new();
    // fold body: (i, j, a, b, acc, k) -> acc + a[i*8+k] * b[k*8+j]
    let dot = b
        .declare(
            FuncKind::FoldBody,
            "dot",
            Ty::Tuple(vec![
                Ty::i32(),
                Ty::i32(),
                i32_arr(64),
                i32_arr(64),
                Ty::i32(),
                Ty::i32(),
            ]),
            Ty::i32(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(dot).unwrap();
        let p = fb.input();
        let i = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
        let j = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
        let a = fb.proj(p, 2, Dest::Fresh(None), L).unwrap();
        let bb = fb.proj(p, 3, Dest::Fresh(None), L).unwrap();
        let acc = fb.proj(p, 4, Dest::Fresh(None), L).unwrap();
        let k = fb.proj(p, 5, Dest::Fresh(None), L).unwrap();
        let eight = fb.constant(Value::I32(8), L).unwrap();
        let i8 = fb
            .binop(Operation::Mul, i, eight, Dest::Fresh(None), L)
            .unwrap();
        let y1 = fb
            .binop(Operation::Add, i8, k, Dest::Fresh(None), L)
            .unwrap();
        let k8 = fb
            .binop(Operation::Mul, k, eight, Dest::Fresh(None), L)
            .unwrap();
        let y2 = fb
            .binop(Operation::Add, k8, j, Dest::Fresh(None), L)
            .unwrap();
        let xa = fb.index(a, y1, Dest::Fresh(None), L).unwrap();
        let xb = fb.index(bb, y2, Dest::Fresh(None), L).unwrap();
        let pr = fb
            .binop(Operation::Mul, xa, xb, Dest::Fresh(None), L)
            .unwrap();
        fb.binop(Operation::Add, acc, pr, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    // map body: (a, b, E) -> fold(dot, [i, j, a, b], 0, krange)
    let cell = b
        .declare(
            FuncKind::MapBody,
            "cell",
            Ty::Tuple(vec![
                i32_arr(64),
                i32_arr(64),
                Ty::Tuple(vec![Ty::i32(), Ty::i32()]),
            ]),
            Ty::i32(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(cell).unwrap();
        let p = fb.input();
        let a = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
        let bb = fb.proj(p, 1, Dest::Fresh(None), L).unwrap();
        let e = fb.proj(p, 2, Dest::Fresh(None), L).unwrap();
        let t = fb.proj(e, 0, Dest::Fresh(None), L).unwrap();
        let eight = fb.constant(Value::I32(8), L).unwrap();
        let i = fb
            .binop(Operation::Div, t, eight, Dest::Fresh(None), L)
            .unwrap();
        let j = fb
            .binop(Operation::Mod, t, eight, Dest::Fresh(None), L)
            .unwrap();
        let zero = fb.constant(Value::I32(0), L).unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        let two = fb.constant(Value::I32(2), L).unwrap();
        let three = fb.constant(Value::I32(3), L).unwrap();
        let four = fb.constant(Value::I32(4), L).unwrap();
        let five = fb.constant(Value::I32(5), L).unwrap();
        let six = fb.constant(Value::I32(6), L).unwrap();
        let seven = fb.constant(Value::I32(7), L).unwrap();
        let kr = fb
            .pack_array(
                &[zero, one, two, three, four, five, six, seven],
                Dest::Fresh(None),
                L,
            )
            .unwrap();
        let z = fb.constant(Value::I32(0), L).unwrap();
        fb.fold_captured(dot, &[i, j, a, bb], z, kr, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    // main: (seed, a, b) -> enumerate -> map_captured(cell, [a, b])
    let f = b
        .declare(
            FuncKind::Named,
            "main",
            Ty::Tuple(vec![i32_arr(64), i32_arr(64), i32_arr(64)]),
            i32_arr(64),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let inp = fb.input();
        let seed = fb.proj(inp, 0, Dest::Fresh(None), L).unwrap();
        let a = fb.proj(inp, 1, Dest::Fresh(None), L).unwrap();
        let bb = fb.proj(inp, 2, Dest::Fresh(None), L).unwrap();
        let en = fb.enumerate(seed, Dest::Fresh(None), L).unwrap();
        fb.map_captured(cell, &[a, bb], en, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    let bp = ir.bounds_proof(dot);
    let ms = index_morphs(&ir, dot);
    assert_eq!(ms.len(), 2);
    assert!(
        bp.proven(ms[0]) && bp.proven(ms[1]),
        "capture-threaded: i*8+k and k*8+j both prove (<= 7*8+7 = 63 < 64)"
    );
}
