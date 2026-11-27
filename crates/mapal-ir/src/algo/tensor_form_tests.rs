//! Tests of `TileRead::tensor_form` and `TensorForm::box_of`.

use super::{TensorAxis, TileRead};

fn read(base: u64, ci: u64, ck: u64, clane: u64) -> TileRead {
    TileRead {
        slot: 0,
        base,
        ci,
        ck,
        clane,
        ksplit: None,
        narrow: None,
    }
}

/// Row-major `a` (`a[i·k + k]`), `b` as `k × n` and as `n × k`, an offset
/// base: each a dense tensor whose box of the panel's sides is exact.
#[test]
fn plain_reads_are_dense_tensors() {
    let a = read(0, 128, 1, 0).tensor_form(64, 32, 128).unwrap();
    assert_eq!(a.axes(), vec![TensorAxis::K, TensorAxis::I]);
    let (sides, exact) = a.box_of(|x| if x == TensorAxis::K { 32 } else { 16 });
    assert_eq!((sides, exact), (vec![32, 16], true));
    let b = read(64, 0, 256, 1).tensor_form(64, 256, 128).unwrap();
    assert_eq!(b.axes(), vec![TensorAxis::Lane, TensorAxis::K]);
    assert_eq!(b.base, 64);
    let bt = read(0, 0, 1, 128).tensor_form(64, 256, 128).unwrap();
    assert_eq!(bt.axes(), vec![TensorAxis::K, TensorAxis::Lane]);
    // A side that does not divide its extent is no exact box.
    assert!(!a.box_of(|_| 24).1);
    // A padded row stride (ci > k) is still dense.
    assert!(read(0, 136, 1, 0).tensor_form(64, 32, 128).is_some());
}

/// Overlapping axes (a window `a[i + k]`) and derived `k` digits are no
/// tensor; an axis of extent 1 or stride 0 is no dimension.
#[test]
fn overlapping_axes_are_refused() {
    assert!(read(0, 1, 1, 0).tensor_form(64, 32, 128).is_none());
    assert!(read(0, 64, 1, 0).tensor_form(64, 32, 128).is_none());
    let one = read(0, 128, 1, 0).tensor_form(1, 32, 128).unwrap();
    assert_eq!(one.axes(), vec![TensorAxis::K]);
    let mut w = read(0, 128, 0, 0);
    w.ksplit = Some(super::TileKSplit {
        div: 3,
        cq: 1,
        cr: 1,
    });
    assert!(w.tensor_form(64, 32, 9).is_none());
}
