//! The ADR-0018 enumerate-bound twin: `validate` independently re-derives the
//! `n ≤ i32::MAX` condition the builder rejects up front. The builder can never
//! *produce* an oversize-enumerate graph, so this corrupts a sealed one (the
//! in-crate privilege noted in `graph.rs`) to prove the validate pass flags it.

use super::{IrViolation, validate};
use crate::builder::{Dest, IrBuilder};
use crate::graph::FuncKind;
use crate::loc::SourceLoc;
use crate::ty::Ty;

const L: SourceLoc = SourceLoc { start: 0, end: 0 };

#[test]
fn oversize_enumerate_edge_flagged() {
    let input = Ty::Array {
        elem: Box::new(Ty::i32()),
        size: 3,
    };
    let out = Ty::Array {
        elem: Box::new(Ty::Tuple(vec![Ty::i32(), Ty::i32()])),
        size: 3,
    };
    let mut b = IrBuilder::new();
    let f = b.declare(FuncKind::Named, "main", input, out, L).unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let a = fb.input();
        fb.enumerate(a, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let mut ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty());
    // Corrupt every array size past the bound, keeping source/target sizes
    // equal so the typing shape still holds and ONLY the bound twin fires.
    let big = i32::MAX as u64 + 1;
    for (_, obj) in ir.objects.iter_mut() {
        if let Ty::Array { size, .. } = &mut obj.ty {
            *size = big;
        }
    }
    let v = validate(&ir);
    assert!(
        v.iter()
            .any(|x| matches!(x, IrViolation::EnumerateIndexOverflow(_))),
        "expected EnumerateIndexOverflow twin, got {v:?}"
    );
}
