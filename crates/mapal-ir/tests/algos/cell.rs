//! `cell_plan` tests: per-cell morphisms and their read footprints.

use super::*;

// --- cell_plan (S54: per-cell morphisms and their read footprints) ----------

pub(super) fn f32_arr(n: u64) -> Ty {
    Ty::Array {
        elem: Box::new(Ty::f32()),
        size: n,
    }
}

/// `ta -> map { t -> body(x, y, t) } -> out` over `n` cells, `x: [f32; nx]`,
/// `y: [f32; ny]` captured (both `fill`s). Returns the IR, `main` and the map.
pub(super) fn cell_fixture(
    n: u64,
    nx: u64,
    ny: u64,
    body: impl FnOnce(&mut mapal_ir::FnBuilder<'_>, ObjectId, ObjectId, ObjectId) -> ObjectId,
) -> (mapal_ir::CategoryIr, mapal_ir::FuncId, MorphismId) {
    let mut b = IrBuilder::new();
    let cell = b
        .declare(
            FuncKind::MapBody,
            "cell",
            Ty::Tuple(vec![f32_arr(nx), f32_arr(ny), Ty::i32()]),
            Ty::f32(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(cell).unwrap();
        let input = fb.input();
        let x = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let y = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        let t = fb.proj(input, 2, Dest::Fresh(None), L).unwrap();
        let v = body(&mut fb, x, y, t);
        let z = fb.constant(Value::F32(0.0), L).unwrap();
        fb.binop(Operation::Add, v, z, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let main = b
        .declare(FuncKind::Named, "main", Ty::Unit, f32_arr(n), L)
        .unwrap();
    {
        let mut fb = b.build_fn(main).unwrap();
        let one = fb.constant(Value::F32(1.0), L).unwrap();
        let cx = fb.constant(tile_count(nx), L).unwrap();
        let x = fb.fill(one, cx, Dest::Fresh(None), L).unwrap();
        let cy = fb.constant(tile_count(ny), L).unwrap();
        let y = fb.fill(one, cy, Dest::Fresh(None), L).unwrap();
        let cn = fb.constant(tile_count(n), L).unwrap();
        let ta = fb.iota(cn, Dest::Fresh(None), L).unwrap();
        fb.map_captured(cell, &[x, y], ta, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(main).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    let m = one_op(
        &ir,
        Operation::Map {
            body: cell,
            captures: 2,
        },
    );
    (ir, main, m)
}

/// `t op c` for a literal `c`.
pub(super) fn cell_t(
    fb: &mut mapal_ir::FnBuilder<'_>,
    op: Operation,
    t: ObjectId,
    c: i32,
) -> ObjectId {
    let k = fb.constant(Value::I32(c), L).unwrap();
    fb.binop(op, t, k, Dest::Fresh(None), L).unwrap()
}

/// `x[ix] + y[iy]`.
pub(super) fn cell_sum(
    fb: &mut mapal_ir::FnBuilder<'_>,
    x: ObjectId,
    ix: ObjectId,
    y: ObjectId,
    iy: ObjectId,
) -> ObjectId {
    let a = fb.index(x, ix, Dest::Fresh(None), L).unwrap();
    let b = fb.index(y, iy, Dest::Fresh(None), L).unwrap();
    fb.binop(Operation::Add, a, b, Dest::Fresh(None), L)
        .unwrap()
}

pub(super) fn cell_reads(
    n: u64,
    nx: u64,
    ny: u64,
    body: impl FnOnce(&mut mapal_ir::FnBuilder<'_>, ObjectId, ObjectId, ObjectId) -> ObjectId,
) -> Option<Vec<mapal_ir::CellRead>> {
    let (ir, main, m) = cell_fixture(n, nx, ny, body);
    let plan = ir.cell_plan(main);
    plan.get(m).map(|s| {
        assert_eq!(s.extent, n);
        s.reads.clone()
    })
}

#[test]
fn cell_plan_same_and_col_a_bias() {
    use mapal_ir::{CellRead, Footprint};
    // `c[t] + bias[t % 64]` — the bias probe's shape.
    let r = cell_reads(256, 256, 64, |fb, x, y, t| {
        let j = cell_t(fb, Operation::Mod, t, 64);
        cell_sum(fb, x, t, y, j)
    });
    assert_eq!(
        r,
        Some(vec![
            CellRead {
                slot: 0,
                foot: Footprint::Same
            },
            CellRead {
                slot: 1,
                foot: Footprint::Col { c: 64 }
            },
        ])
    );
}

#[test]
fn cell_plan_row_and_const() {
    use mapal_ir::{CellRead, Footprint};
    // `x[t / 16] + y[3]`: a per-row scale and one fixed cell.
    let r = cell_reads(256, 16, 8, |fb, x, y, t| {
        let i = cell_t(fb, Operation::Div, t, 16);
        let k = fb.constant(Value::I32(3), L).unwrap();
        cell_sum(fb, x, i, y, k)
    });
    assert_eq!(
        r,
        Some(vec![
            CellRead {
                slot: 0,
                foot: Footprint::Row { c: 16 }
            },
            CellRead {
                slot: 1,
                foot: Footprint::Const(3)
            },
        ])
    );
}

#[test]
fn cell_plan_other_address() {
    use mapal_ir::{CellRead, Footprint};
    // `x[t + 1]`: proven (x has n + 1 cells), but not a cell-only footprint.
    let r = cell_reads(256, 257, 256, |fb, x, y, t| {
        let s = cell_t(fb, Operation::Add, t, 1);
        cell_sum(fb, x, s, y, t)
    });
    assert_eq!(
        r,
        Some(vec![
            CellRead {
                slot: 0,
                foot: Footprint::Other
            },
            CellRead {
                slot: 1,
                foot: Footprint::Same
            },
        ])
    );
}

#[test]
fn cell_plan_refuses_a_trap_capable_body() {
    // `x[t]` with x one cell short: the read is not proven, the cell can trap.
    let r = cell_reads(256, 255, 256, |fb, x, y, t| cell_sum(fb, x, t, y, t));
    assert_eq!(r, None);
}

#[test]
fn cell_plan_records_a_nested_folds_reads() {
    // A tile site's cell holds a fold over its captures (cpu-gpu-boundary
    // `nested`): a, b and its k range read at any proven address, `Other`.
    let (ir, main, _) = tile_matmul_fixture(16, 16, 16);
    let plan = ir.cell_plan(main);
    let site = plan.values().next().expect("one cell site");
    assert_eq!(plan.len(), 1);
    assert_eq!(
        site.nested.len(),
        3,
        "a, b and the k range it walks: {site:?}"
    );
    assert!(
        site.reads
            .iter()
            .all(|r| r.foot == mapal_ir::Footprint::Other),
        "{site:?}"
    );
    assert!(site.uniform.is_empty());
    assert_eq!(site.work, 1 + 16, "one step plus the k = 16 fold: {site:?}");
    assert_eq!(ir.tile_plan(main).sites.len(), 1);
}

#[test]
fn cell_plan_capture_free_map_reads_its_element_same() {
    use mapal_ir::{CellRead, Footprint};
    // `x -> map { v -> v + 0.0 }`: the epilogue shape — the element is `x[t]`.
    let mut b = IrBuilder::new();
    let body = b
        .declare(FuncKind::MapBody, "epi", Ty::f32(), Ty::f32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(body).unwrap();
        let v = fb.input();
        let z = fb.constant(Value::F32(0.0), L).unwrap();
        fb.binop(Operation::Add, v, z, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let main = b
        .declare(FuncKind::Named, "main", Ty::Unit, f32_arr(64), L)
        .unwrap();
    {
        let mut fb = b.build_fn(main).unwrap();
        let one = fb.constant(Value::F32(1.0), L).unwrap();
        let n = fb.constant(tile_count(64), L).unwrap();
        let x = fb.fill(one, n, Dest::Fresh(None), L).unwrap();
        fb.map(body, x, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(main).unwrap();
    let m = one_op(&ir, Operation::Map { body, captures: 0 });
    let site = &ir.cell_plan(main)[m];
    assert_eq!(site.extent, 64);
    assert_eq!(
        site.reads,
        vec![CellRead {
            slot: 0,
            foot: Footprint::Same
        }]
    );
}
