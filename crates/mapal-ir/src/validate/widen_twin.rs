//! The ADR-0029 widening-lattice twin: corrupt a legal edge's source type so
//! validate independently rejects the now-illegal pair.

use super::{IrViolation, validate};
use crate::builder::{Dest, IrBuilder};
use crate::graph::FuncKind;
use crate::loc::SourceLoc;
use crate::ty::Ty;

const L: SourceLoc = SourceLoc { start: 0, end: 0 };

#[test]
fn invalid_widen_edge_flagged() {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", Ty::i32(), Ty::f64(), L)
        .unwrap();
    let input;
    {
        let mut fb = b.build_fn(f).unwrap();
        input = fb.input();
        let widened = fb.widen(input, Ty::f64(), Dest::Fresh(None), L).unwrap();
        fb.output(widened, None, L).unwrap();
        fb.finish().unwrap();
    }
    let mut ir = b.seal(f).unwrap();
    ir.objects[input].ty = Ty::i64();
    let v = validate(&ir);
    assert!(
        v.iter().any(|x| matches!(x, IrViolation::InvalidWiden(_))),
        "expected InvalidWiden twin, got {v:?}"
    );
}
