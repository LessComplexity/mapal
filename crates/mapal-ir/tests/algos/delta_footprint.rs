//! `write_delta` and `footprint_plan` tests.

use super::*;

// --- write_delta (S53 `retain-delta`) ----------------------------------------

/// A canonical loop carrying `(x: [i32; n], t: i32)` that writes `x[idx(t, j)]
/// <- 1` for `j < d`, then `t + 1`. Returns the graph, the fn, and the `x` and
/// `t` merge views.
pub(super) fn update_loop(
    n: u64,
    d: i32,
    idx: impl Fn(&mut mapal_ir::FnBuilder<'_>, ObjectId, i32) -> ObjectId,
) -> (mapal_ir::CategoryIr, mapal_ir::FuncId, ObjectId, ObjectId) {
    let arr = Ty::Array {
        elem: Box::new(Ty::i32()),
        size: n,
    };
    let mut b = IrBuilder::new();
    let f = b.declare(FuncKind::Named, "w", Ty::i32(), arr, L).unwrap();
    let (xv, tv);
    {
        let mut fb = b.build_fn(f).unwrap();
        let z = fb.constant(Value::I32(0), L).unwrap();
        let x0 = fb
            .pack_array(&vec![z; n as usize], Dest::Fresh(None), L)
            .unwrap();
        let init = fb.pack(&[x0, z], Dest::Fresh(None), L).unwrap();
        let lh = fb.begin_loop(init, L).unwrap();
        let merge = fb.merge_of(&lh);
        xv = fb.proj(merge, 0, Dest::Fresh(None), L).unwrap();
        tv = fb.proj(merge, 1, Dest::Fresh(None), L).unwrap();
        let lim = fb.constant(Value::I32(4), L).unwrap();
        let cond = fb
            .binop(Operation::Lt, tv, lim, Dest::Fresh(None), L)
            .unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        let mut cur = xv;
        for j in 0..d {
            let i = idx(&mut fb, tv, j);
            cur = fb.update(cur, i, one, Dest::Fresh(None), L).unwrap();
        }
        let t1 = fb
            .binop(Operation::Add, tv, one, Dest::Fresh(None), L)
            .unwrap();
        let next = fb.pack(&[cur, t1], Dest::Fresh(None), L).unwrap();
        fb.loop_back(&lh, next, cond, L).unwrap();
        fb.loop_exit(&lh, xv, cond, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.end_loop(lh).unwrap();
        fb.finish().unwrap();
    }
    (b.seal(f).unwrap(), f, xv, tv)
}

/// `c·t + k` as builder morphisms.
pub(super) fn affine_ix(fb: &mut mapal_ir::FnBuilder<'_>, t: ObjectId, c: i32, k: i32) -> ObjectId {
    let c = fb.constant(Value::I32(c), L).unwrap();
    let k = fb.constant(Value::I32(k), L).unwrap();
    let ct = fb
        .binop(Operation::Mul, t, c, Dest::Fresh(None), L)
        .unwrap();
    fb.binop(Operation::Add, ct, k, Dest::Fresh(None), L)
        .unwrap()
}

pub(super) fn rows(d: mapal_ir::Delta) -> Option<(u64, ObjectId, i64, i64)> {
    match d {
        mapal_ir::Delta::Rows { stride, start, .. } => Some((stride, start.var, start.a, start.b)),
        _ => None,
    }
}

#[test]
fn write_delta_matmul4_cell_is_rows_1_t() {
    // matmul4_loop's `c[t] <- v`: one cell, at the counter.
    let (ir, f, xv, tv) = update_loop(16, 1, |_, t, _| t);
    let wd = ir.write_delta(f);
    assert_eq!(wd.len(), 1);
    assert_eq!(rows(wd[xv]), Some((1, tv, 1, 0)));
}

#[test]
fn write_delta_decode_row_is_rows_d_td() {
    // The decode loop's row write: `x[t·d + j] <- …` for `j < d`.
    let (ir, f, xv, tv) = update_loop(32, 8, |fb, t, j| affine_ix(fb, t, 8, j));
    assert_eq!(rows(ir.write_delta(f)[xv]), Some((8, tv, 8, 0)));
}

#[test]
fn write_delta_column_write_is_top() {
    // loop_rewrite: `x[j·d + t]` — d cells a stride apart, not one range.
    let (ir, f, xv, _) = update_loop(64, 8, |fb, t, j| affine_ix(fb, t, 1, 8 * j));
    assert_eq!(ir.write_delta(f)[xv], mapal_ir::Delta::Top);
}

#[test]
fn write_delta_non_affine_index_is_top() {
    // `x[t·t]`: the index is not affine in the counter.
    let (ir, f, xv, _) = update_loop(16, 1, |fb, t, _| {
        fb.binop(Operation::Mul, t, t, Dest::Fresh(None), L)
            .unwrap()
    });
    assert_eq!(ir.write_delta(f)[xv], mapal_ir::Delta::Top);
}

#[test]
fn write_delta_unwritten_carry_is_top() {
    // The array crosses the back edge unchanged: nothing to retain from.
    let (ir, f, xv, _) = update_loop(16, 0, |_, t, _| t);
    assert_eq!(ir.write_delta(f)[xv], mapal_ir::Delta::Top);
}

// --- footprint_plan (generic-fusion: every consumer's read edges) ----------

pub(super) fn affine(w: u64, base: u64, cq: u64, cr: u64, taps: &[(u64, u64)]) -> mapal_ir::Reach {
    mapal_ir::Reach::Affine {
        w,
        base,
        cq,
        cr,
        taps: taps
            .iter()
            .map(|&(coef, ext)| mapal_ir::Tap { coef, ext })
            .collect(),
    }
}

/// The `(slot, reach)` of every direct edge of `m` (no seen-through law).
pub(super) fn reaches(
    ir: &mapal_ir::CategoryIr,
    f: mapal_ir::FuncId,
    m: MorphismId,
) -> Vec<(u32, mapal_ir::Reach)> {
    ir.footprint_plan(f)
        .of(m)
        .iter()
        .filter(|e| e.law.is_none())
        .map(|e| (e.slot, e.reach.clone()))
        .collect()
}

pub(super) fn the_site(ir: &mapal_ir::CategoryIr, f: mapal_ir::FuncId) -> MorphismId {
    let plan = ir.tile_plan_strided(f);
    assert_eq!(plan.sites.len(), 1);
    plan.sites.iter().next().unwrap().0
}

#[test]
fn footprint_tile_reads_and_the_a_panel_rect() {
    // c[i·N + j] = Σ_k a[i·K + k]·b[k·N + j]: a is `w = N, cq = K, tap (1, K)`,
    // b `w = N, cr = 1, tap (N, K)`; a BM×BN block reads a BM×K panel of a.
    let (ir, main, _) = tile_matmul_fixture(64, 48, 32);
    let m = the_site(&ir, main);
    let r = reaches(&ir, main, m);
    assert_eq!(
        r,
        vec![
            (1, affine(48, 0, 32, 0, &[(1, 32)])),
            (2, affine(48, 0, 0, 1, &[(48, 32)]))
        ]
    );
    assert_eq!(r[0].1.rect(32, 16..32, 0..16), Some([16..32, 0..32]));
    // b's panel for the same block: all K rows, the block's BN columns.
    assert_eq!(r[1].1.rect(48, 16..32, 16..32), Some([0..32, 16..32]));
}

#[test]
fn footprint_fir_window_is_bn_plus_taps_minus_one() {
    let (ir, main, _) = tile_fir_fixture(64, 16, 1);
    let m = the_site(&ir, main);
    let r = reaches(&ir, main, m);
    assert_eq!(r[1], (1, affine(64, 0, 0, 1, &[(1, 16)])));
    // A block of 32 lanes reads 32 + 15 signal cells (one producer row).
    let x_len = 63 + 16;
    assert_eq!(r[1].1.rect(x_len, 0..1, 32..64), Some([0..1, 32..79]));
}

#[test]
fn footprint_conv_ksplit_halo() {
    // 5×5 taps over a 20-wide image: a BM×BN block reads (BM+4)×(BN+4).
    let (ir, main, _) = tile_conv2d_fixture_with(Conv2dFixture {
        side: 16,
        div: 5,
        mod_div: 5,
        depth: 25,
        mix_raw_k: false,
    });
    let m = the_site(&ir, main);
    let r = reaches(&ir, main, m);
    assert_eq!(r[1], (2, affine(16, 0, 20, 1, &[(20, 5), (1, 5)])));
    assert_eq!(r[1].1.rect(20, 4..8, 8..16), Some([4..12, 8..20]));
}

#[test]
fn footprint_cell_reads_same_row_col_const_affine() {
    // `c[t] + bias[t % 64]`.
    let (ir, main, m) = cell_fixture(256, 256, 64, |fb, x, y, t| {
        let j = cell_t(fb, Operation::Mod, t, 64);
        cell_sum(fb, x, t, y, j)
    });
    assert_eq!(
        reaches(&ir, main, m),
        vec![(0, affine(1, 0, 1, 0, &[])), (1, affine(64, 0, 0, 1, &[]))]
    );
    // `x[t / 16] + y[3]`.
    let (ir, main, m) = cell_fixture(256, 16, 8, |fb, x, y, t| {
        let i = cell_t(fb, Operation::Div, t, 16);
        let k = fb.constant(Value::I32(3), L).unwrap();
        cell_sum(fb, x, i, y, k)
    });
    assert_eq!(
        reaches(&ir, main, m),
        vec![(0, affine(16, 0, 1, 0, &[])), (1, affine(1, 3, 0, 0, &[]))]
    );
    // `x[t + 1] + y[t]`: affine where `Footprint` says `Other`.
    let (ir, main, m) = cell_fixture(256, 257, 256, |fb, x, y, t| {
        let s = cell_t(fb, Operation::Add, t, 1);
        cell_sum(fb, x, s, y, t)
    });
    assert_eq!(
        reaches(&ir, main, m),
        vec![(0, affine(1, 1, 1, 0, &[])), (1, affine(1, 0, 1, 0, &[]))]
    );
    assert_eq!(
        ir.cell_plan(main)[m].reads[0].foot,
        mapal_ir::Footprint::Other
    );
    // `x[2·(t / 16)] + y[t % 16 + 4·(t / 16)]`: one divisor, both axes.
    let (ir, main, m) = cell_fixture(256, 32, 76, |fb, x, y, t| {
        let i = cell_t(fb, Operation::Div, t, 16);
        let two = fb.constant(Value::I32(2), L).unwrap();
        let xi = fb
            .binop(Operation::Mul, i, two, Dest::Fresh(None), L)
            .unwrap();
        let j = cell_t(fb, Operation::Mod, t, 16);
        let i2 = cell_t(fb, Operation::Div, t, 16);
        let four = fb.constant(Value::I32(4), L).unwrap();
        let i4 = fb
            .binop(Operation::Mul, i2, four, Dest::Fresh(None), L)
            .unwrap();
        let yi = fb
            .binop(Operation::Add, j, i4, Dest::Fresh(None), L)
            .unwrap();
        cell_sum(fb, x, xi, y, yi)
    });
    assert_eq!(
        reaches(&ir, main, m),
        vec![(0, affine(16, 0, 2, 0, &[])), (1, affine(16, 0, 4, 1, &[]))]
    );
}

/// `rows -> map { i -> (0.0, kr) -> fold { acc, k -> acc + a[i·ci + k] } }`.
pub(super) fn row_fold_fixture(
    rows: u64,
    k: u64,
    ci: u64,
) -> (mapal_ir::CategoryIr, mapal_ir::FuncId, MorphismId) {
    let a_ty = f32_arr((rows - 1) * ci + k);
    let mut b = IrBuilder::new();
    let step = b
        .declare(
            FuncKind::FoldBody,
            "rf_step",
            Ty::Tuple(vec![a_ty.clone(), Ty::i32(), Ty::f32(), Ty::i32()]),
            Ty::f32(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(step).unwrap();
        let input = fb.input();
        let a = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let i = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        let acc = fb.proj(input, 2, Dest::Fresh(None), L).unwrap();
        let kk = fb.proj(input, 3, Dest::Fresh(None), L).unwrap();
        let c = fb.constant(Value::I32(ci as i32), L).unwrap();
        let ic = fb
            .binop(Operation::Mul, i, c, Dest::Fresh(None), L)
            .unwrap();
        let ix = fb
            .binop(Operation::Add, ic, kk, Dest::Fresh(None), L)
            .unwrap();
        let v = fb.index(a, ix, Dest::Fresh(None), L).unwrap();
        fb.binop(Operation::Add, acc, v, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let row = b
        .declare(
            FuncKind::MapBody,
            "rf_row",
            Ty::Tuple(vec![a_ty.clone(), i32_arr(k), Ty::i32()]),
            Ty::f32(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(row).unwrap();
        let input = fb.input();
        let a = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let kr = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        let i = fb.proj(input, 2, Dest::Fresh(None), L).unwrap();
        let z = fb.constant(Value::F32(0.0), L).unwrap();
        fb.fold_captured(step, &[a, i], z, kr, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let main = b
        .declare(FuncKind::Named, "main", Ty::Unit, f32_arr(rows), L)
        .unwrap();
    {
        let mut fb = b.build_fn(main).unwrap();
        let one = fb.constant(Value::F32(1.0), L).unwrap();
        let n = fb.constant(tile_count((rows - 1) * ci + k), L).unwrap();
        let a = fb.fill(one, n, Dest::Fresh(None), L).unwrap();
        let ck = fb.constant(tile_count(k), L).unwrap();
        let kr = fb.iota(ck, Dest::Fresh(None), L).unwrap();
        let cr = fb.constant(tile_count(rows), L).unwrap();
        let rr = fb.iota(cr, Dest::Fresh(None), L).unwrap();
        fb.map_captured(row, &[a, kr], rr, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(main).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    let m = one_op(
        &ir,
        Operation::Map {
            body: row,
            captures: 2,
        },
    );
    (ir, main, m)
}

#[test]
fn footprint_row_fold_reads_its_row() {
    let (ir, main, m) = row_fold_fixture(8, 16, 16);
    assert_eq!(ir.row_fold(main).len(), 1);
    let r = reaches(&ir, main, m);
    assert_eq!(r, vec![(0, affine(1, 0, 16, 0, &[(1, 16)]))]);
    // Rows 2..4 of the fold read rows 2..4 of a 16-wide producer, whole.
    assert_eq!(r[0].1.rect(16, 2..4, 0..1), Some([2..4, 0..16]));
}

/// `fn m(a: [i32; 16])`: `y = a -> map {v -> v + 1}` (a law over a load),
/// `z = y -> map {v -> v * 2}`, `ix = iota 16 -> map {t -> t + 1}`,
/// `w = ix -> map {v -> v * 2}`, `s = (0, a) -> fold`, `a[3]`.
pub(super) fn law_fixture() -> (mapal_ir::CategoryIr, mapal_ir::FuncId, Vec<ObjectId>) {
    let mut b = IrBuilder::new();
    let unary = |b: &mut IrBuilder, name: &str, op: Operation, k: i32| {
        let f = b
            .declare(FuncKind::MapBody, name, Ty::i32(), Ty::i32(), L)
            .unwrap();
        let mut fb = b.build_fn(f).unwrap();
        let v = fb.input();
        let c = fb.constant(Value::I32(k), L).unwrap();
        fb.binop(op, v, c, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
        f
    };
    let inc = unary(&mut b, "inc", Operation::Add, 1);
    let dbl = unary(&mut b, "dbl", Operation::Mul, 2);
    let sum = b
        .declare(
            FuncKind::FoldBody,
            "sum",
            Ty::Tuple(vec![Ty::i32(), Ty::i32()]),
            Ty::i32(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(sum).unwrap();
        let input = fb.input();
        let acc = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let v = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        fb.binop(Operation::Add, acc, v, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let f = b
        .declare(
            FuncKind::Named,
            "m",
            Ty::Tuple(vec![i32_arr(16), Ty::Unit]),
            Ty::i32(),
            L,
        )
        .unwrap();
    let out;
    {
        let mut fb = b.build_fn(f).unwrap();
        let inp = fb.input();
        let a = fb.proj(inp, 0, Dest::Fresh(None), L).unwrap();
        let y = fb.map(inc, a, Dest::Fresh(None), L).unwrap();
        let z = fb.map(dbl, y, Dest::Fresh(None), L).unwrap();
        let n = fb.constant(Value::I32(16), L).unwrap();
        let it = fb.iota(n, Dest::Fresh(None), L).unwrap();
        let ix = fb.map(inc, it, Dest::Fresh(None), L).unwrap();
        let w = fb.map(dbl, ix, Dest::Fresh(None), L).unwrap();
        let zero = fb.constant(Value::I32(0), L).unwrap();
        let s = fb
            .fold_captured(sum, &[], zero, a, Dest::Fresh(None), L)
            .unwrap();
        let three = fb.constant(Value::I32(3), L).unwrap();
        let a3 = fb.index(a, three, Dest::Fresh(None), L).unwrap();
        let z0 = fb.index(z, zero, Dest::Fresh(None), L).unwrap();
        let w0 = fb.index(w, zero, Dest::Fresh(None), L).unwrap();
        let p = fb
            .binop(Operation::Add, s, a3, Dest::Fresh(None), L)
            .unwrap();
        let q = fb
            .binop(Operation::Add, z0, w0, Dest::Fresh(None), L)
            .unwrap();
        fb.binop(Operation::Add, p, q, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
        out = vec![a, y, z, ix, w, s, a3];
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    (ir, f, out)
}

#[test]
fn footprint_laws_folds_and_host_reads() {
    let (ir, f, o) = law_fixture();
    let (a, y, z, ix, w, s, a3) = (o[0], o[1], o[2], o[3], o[4], o[5], o[6]);
    let plan = ir.footprint_plan(f);
    let writer = |x: ObjectId| *ir.in_edges(x).first().unwrap();
    // `z` reads `y` at `t`, and through `y`'s law the parameter `a` at `t`.
    let ez = plan.of(writer(z));
    assert!(ez.contains(&mapal_ir::Edge {
        consumer: writer(z),
        slot: 0,
        producer: y,
        reach: affine(1, 0, 1, 0, &[]),
        law: None,
    }));
    assert!(ez.contains(&mapal_ir::Edge {
        consumer: writer(z),
        slot: 0,
        producer: a,
        reach: affine(1, 0, 1, 0, &[]),
        law: Some(y),
    }));
    // `w` reads `ix`, whose element is a closed form of the index.
    assert_eq!(plan.of(writer(w))[0].producer, ix);
    assert_eq!(plan.of(writer(w))[0].reach, mapal_ir::Reach::Law);
    // A whole fold reads every cell; `a[3]` one.
    let es = plan.of(writer(s));
    assert_eq!(
        (es[0].producer, es[0].reach.clone()),
        (a, affine(1, 0, 0, 0, &[(1, 16)]))
    );
    assert_eq!(es[0].reach.rect(16, 0..1, 0..1), Some([0..1, 0..16]));
    assert_eq!(plan.of(writer(a3))[0].reach, affine(1, 3, 0, 0, &[]));
}

#[test]
fn footprint_rect_of_ranges_and_laws() {
    use mapal_ir::Reach;
    assert_eq!(
        Reach::Range { lo: 5, hi: 9 }.rect(16, 0..1, 0..1),
        Some([0..1, 5..9])
    );
    assert_eq!(
        Reach::Range { lo: 5, hi: 40 }.rect(16, 0..1, 0..1),
        Some([0..3, 0..16])
    );
    assert_eq!(Reach::Law.rect(16, 0..4, 0..4), Some([0..0, 0..0]));
    assert_eq!(Reach::Whole.rect(16, 0..4, 0..4), None);
    // A column term that leaves the producer row has no rectangle.
    assert_eq!(affine(4, 0, 0, 3, &[]).rect(8, 0..1, 0..4), None);
}

proptest! {
    /// `rect` is the tight hull of the enumerated addresses — never smaller,
    /// never larger — whenever it answers.
    #[test]
    fn footprint_rect_is_the_tight_hull(
        pw in 1u64..24,
        base in 0u64..64,
        cq in 0u64..40,
        cr in 0u64..6,
        taps in prop::collection::vec((0u64..30, 1u64..5), 0..3),
        r0 in 0u64..5, rn in 1u64..4,
        c0 in 0u64..5, cn in 1u64..4,
    ) {
        let reach = affine(8, base, cq, cr, &taps);
        let (rows, cols) = (r0..r0 + rn, c0..c0 + cn);
        let mut addrs = Vec::new();
        for q in rows.clone() {
            for r in cols.clone() {
                let mut acc = vec![base + cq * q + cr * r];
                for &(coef, ext) in &taps {
                    acc = acc.iter().flat_map(|&a| (0..ext).map(move |k| a + coef * k)).collect();
                }
                addrs.extend(acc);
            }
        }
        let lo = |f: &dyn Fn(u64) -> u64| addrs.iter().map(|&a| f(a)).min().unwrap();
        let hi = |f: &dyn Fn(u64) -> u64| addrs.iter().map(|&a| f(a)).max().unwrap();
        let (rlo, rhi) = (lo(&|a| a / pw), hi(&|a| a / pw));
        let (clo, chi) = (lo(&|a| a % pw), hi(&|a| a % pw));
        if let Some([rr, cc]) = reach.rect(pw, rows, cols) {
            prop_assert_eq!(rr, rlo..rhi + 1);
            prop_assert_eq!(cc, clo..chi + 1);
        }
    }
}
