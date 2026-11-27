//! The ADR-0027 erased-capture twin: `validate` independently re-derives the
//! empty-residual-capture rejection the builder owns (`ErasedCapture`). The
//! builder can never *produce* such a graph, so this corrupts a sealed one
//! (the in-crate privilege noted in `graph.rs`): the capture tuple
//! `(Unit, i32)` becomes `(Unit, Unit)`, leaving every edge consistently
//! typed so ONLY the erased-capture twin fires.

use super::{IrViolation, validate};
use crate::builder::{Dest, IrBuilder};
use crate::graph::FuncKind;
use crate::loc::SourceLoc;
use crate::ty::Ty;

const L: SourceLoc = SourceLoc { start: 0, end: 0 };

/// Replace the `(Unit, i32)` capture tuple with `(Unit, Unit)` anywhere it
/// nests in `ty`.
fn erase_capture_component(ty: &mut Ty) {
    if *ty == Ty::Tuple(vec![Ty::Unit, Ty::i32()]) {
        *ty = Ty::Tuple(vec![Ty::Unit, Ty::Unit]);
        return;
    }
    match ty {
        Ty::Array { elem, .. } => erase_capture_component(elem),
        Ty::Tuple(ts) => ts.iter_mut().for_each(erase_capture_component),
        Ty::Struct { fields, .. } => fields
            .iter_mut()
            .for_each(|(_, t)| erase_capture_component(t)),
        _ => {}
    }
}

#[test]
fn erased_capture_component_flagged() {
    let cap_ty = Ty::Tuple(vec![Ty::Unit, Ty::i32()]);
    let arr = Ty::Array {
        elem: Box::new(Ty::i32()),
        size: 3,
    };
    let mut b = IrBuilder::new();
    // A MapBody ((Unit, i32), i32) -> i32 that IGNORES its capture, so the
    // corruption keeps every body edge consistently typed.
    let body = b
        .declare(
            FuncKind::MapBody,
            "mb",
            Ty::Tuple(vec![cap_ty.clone(), Ty::i32()]),
            Ty::i32(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(body).unwrap();
        let pin = fb.input();
        let x = fb.proj(pin, 1, Dest::Fresh(None), L).unwrap();
        fb.binop(
            crate::graph::Operation::Add,
            x,
            x,
            Dest::Ret { slot: None },
            L,
        )
        .unwrap();
        fb.finish().unwrap();
    }
    let f = b
        .declare(
            FuncKind::Named,
            "main",
            Ty::Tuple(vec![cap_ty, arr.clone()]),
            arr,
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
    let mut ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty());
    for (_, obj) in ir.objects.iter_mut() {
        erase_capture_component(&mut obj.ty);
    }
    let v = validate(&ir);
    assert!(
        v.iter().any(|x| matches!(x, IrViolation::ErasedCapture(_))),
        "expected the ErasedCapture twin, got {v:?}"
    );
    assert_eq!(
        v.iter()
            .filter(|x| !matches!(x, IrViolation::ErasedCapture(_)))
            .count(),
        0,
        "only the erased-capture twin fires, got {v:?}"
    );
}
