//! The ADR-0029 static-count twin: `validate` independently re-derives the
//! count-value tie the builder owns (`NonStaticCount` + the mint). The builder
//! can never *produce* a mismatched graph, so this corrupts a sealed one (the
//! in-crate privilege noted in `graph.rs`): the count constant's value drifts
//! from the array's static size, leaving every edge consistently typed so ONLY
//! the static-n twin fires.

use super::{IrViolation, validate};
use crate::builder::{Dest, IrBuilder};
use crate::graph::FuncKind;
use crate::loc::SourceLoc;
use crate::ty::{Ty, Value};

const L: SourceLoc = SourceLoc { start: 0, end: 0 };

fn drift(ir: &mut crate::graph::CategoryIr, from: Value, to: Value) {
    for (_, obj) in ir.objects.iter_mut() {
        if let Some(v) = &mut obj.value
            && *v == from
        {
            *v = to.clone();
        }
    }
}

#[test]
fn drifted_iota_count_flagged() {
    let out = Ty::Array {
        elem: Box::new(Ty::i32()),
        size: 5,
    };
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", Ty::Unit, out, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let n = fb.constant(Value::I32(5), L).unwrap();
        fb.iota(n, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let mut ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty());
    // Corrupt: the count constant says 4, the array says 5.
    drift(&mut ir, Value::I32(5), Value::I32(4));
    let v = validate(&ir);
    assert!(
        v.iter()
            .any(|x| matches!(x, IrViolation::IotaCountMismatch(_))),
        "expected IotaCountMismatch twin, got {v:?}"
    );
}

#[test]
fn drifted_fill_count_flagged() {
    let out = Ty::Array {
        elem: Box::new(Ty::f64()),
        size: 3,
    };
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", Ty::Unit, out, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let x = fb.constant(Value::F64(1.5), L).unwrap();
        let n = fb.constant(Value::I32(3), L).unwrap();
        fb.fill(x, n, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let mut ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty());
    // Corrupt: the count constant says 4, the array says 3.
    drift(&mut ir, Value::I32(3), Value::I32(4));
    let v = validate(&ir);
    assert!(
        v.iter()
            .any(|x| matches!(x, IrViolation::IotaCountMismatch(_))),
        "expected IotaCountMismatch twin, got {v:?}"
    );
}
