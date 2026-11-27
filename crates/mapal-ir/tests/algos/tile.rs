//! `tile_plan` tests: matmul, FIR and conv2d tile sites and their refusals.

use super::*;

// --- tile_plan (bit-exact map/fold matmul tiling) ----------------------------

#[derive(Clone, Copy)]
pub(super) enum TileFixtureVariant {
    Standard,
    Postprocess,
    NonconstantSeed,
    RowMajorB,
    UndersizedA,
    DeadCall,
}

pub(super) fn tile_array(elem: &Ty, size: u64) -> Ty {
    Ty::Array {
        elem: Box::new(elem.clone()),
        size,
    }
}

pub(super) fn tile_count(n: u64) -> Value {
    Value::I32(i32::try_from(n).unwrap())
}

pub(super) fn tile_zero(elem: &Ty) -> Value {
    match elem {
        Ty::Float { bits: 32 } => Value::F32(0.0),
        Ty::Float { bits: 64 } => Value::F64(0.0),
        _ => panic!("tile fixture only uses f32/f64"),
    }
}

pub(super) fn tile_one(elem: &Ty) -> Value {
    match elem {
        Ty::Float { bits: 32 } => Value::F32(1.0),
        Ty::Float { bits: 64 } => Value::F64(1.0),
        _ => panic!("tile fixture only uses f32/f64"),
    }
}

pub(super) fn tile_matmul_fixture_with(
    rows: u64,
    c: u64,
    k: u64,
    elem: Ty,
    variant: TileFixtureVariant,
) -> (mapal_ir::CategoryIr, mapal_ir::FuncId, mapal_ir::FuncId) {
    let mapped_size = rows.checked_mul(c).unwrap();
    let full_a_size = rows.checked_mul(k).unwrap();
    let a_size = if matches!(variant, TileFixtureVariant::UndersizedA) {
        full_a_size - 1
    } else {
        full_a_size
    };
    let full_b_size = k.checked_mul(c).unwrap();
    let b_size = if matches!(variant, TileFixtureVariant::RowMajorB) {
        c.checked_sub(1)
            .unwrap()
            .checked_mul(c)
            .unwrap()
            .checked_add(k)
            .unwrap()
    } else {
        full_b_size
    };
    let a_ty = tile_array(&elem, a_size);
    let b_ty = tile_array(&elem, b_size);
    let k_ty = i32_arr(k);

    let mut b = IrBuilder::new();
    // DeadCall: a Named identity whose call in the map body is dead — the
    // micro-kernel would skip it, so tile_plan must refuse (a skipped callee
    // could trap in general; R1 requires refusal, not hope).
    let id_fn = matches!(variant, TileFixtureVariant::DeadCall).then(|| {
        let id_fn = b
            .declare(FuncKind::Named, "tile_id", Ty::i32(), Ty::i32(), L)
            .unwrap();
        let mut fb = b.build_fn(id_fn).unwrap();
        let input = fb.input();
        let z = fb.constant(Value::I32(0), L).unwrap();
        fb.binop(Operation::Add, input, z, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
        id_fn
    });
    let dot = b
        .declare(
            FuncKind::FoldBody,
            "tile_dot",
            Ty::Tuple(vec![
                a_ty.clone(),
                Ty::i32(),
                b_ty.clone(),
                Ty::i32(),
                elem.clone(),
                Ty::i32(),
            ]),
            elem.clone(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(dot).unwrap();
        let input = fb.input();
        let a = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let i = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        let bb = fb.proj(input, 2, Dest::Fresh(None), L).unwrap();
        let j = fb.proj(input, 3, Dest::Fresh(None), L).unwrap();
        let acc = fb.proj(input, 4, Dest::Fresh(None), L).unwrap();
        let kk = fb.proj(input, 5, Dest::Fresh(None), L).unwrap();
        let ca = fb.constant(tile_count(k), L).unwrap();
        let cb = fb.constant(tile_count(c), L).unwrap();
        let i_ca = fb
            .binop(Operation::Mul, i, ca, Dest::Fresh(None), L)
            .unwrap();
        let a_index = fb
            .binop(Operation::Add, i_ca, kk, Dest::Fresh(None), L)
            .unwrap();
        let b_index = if matches!(variant, TileFixtureVariant::RowMajorB) {
            let j_cb = fb
                .binop(Operation::Mul, j, cb, Dest::Fresh(None), L)
                .unwrap();
            fb.binop(Operation::Add, j_cb, kk, Dest::Fresh(None), L)
                .unwrap()
        } else {
            let k_cb = fb
                .binop(Operation::Mul, kk, cb, Dest::Fresh(None), L)
                .unwrap();
            fb.binop(Operation::Add, k_cb, j, Dest::Fresh(None), L)
                .unwrap()
        };
        let a_value = fb.index(a, a_index, Dest::Fresh(None), L).unwrap();
        let b_value = fb.index(bb, b_index, Dest::Fresh(None), L).unwrap();
        let product = fb
            .binop(Operation::Mul, a_value, b_value, Dest::Fresh(None), L)
            .unwrap();
        fb.binop(Operation::Add, acc, product, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }

    let cell = b
        .declare(
            FuncKind::MapBody,
            "tile_cell",
            Ty::Tuple(vec![k_ty.clone(), a_ty.clone(), b_ty.clone(), Ty::i32()]),
            elem.clone(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(cell).unwrap();
        let input = fb.input();
        let krange = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let a = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        let bb = fb.proj(input, 2, Dest::Fresh(None), L).unwrap();
        let t = fb.proj(input, 3, Dest::Fresh(None), L).unwrap();
        let columns = fb.constant(tile_count(c), L).unwrap();
        if let Some(id_fn) = id_fn {
            fb.call(id_fn, t, Dest::Fresh(None), L).unwrap();
        }
        let i = fb
            .binop(Operation::Div, t, columns, Dest::Fresh(None), L)
            .unwrap();
        let j = fb
            .binop(Operation::Mod, t, columns, Dest::Fresh(None), L)
            .unwrap();
        let zero = fb.constant(tile_zero(&elem), L).unwrap();
        let seed = if matches!(variant, TileFixtureVariant::NonconstantSeed) {
            fb.binop(Operation::Add, zero, zero, Dest::Fresh(None), L)
                .unwrap()
        } else {
            zero
        };
        let dest = if matches!(variant, TileFixtureVariant::Postprocess) {
            Dest::Fresh(None)
        } else {
            Dest::Ret { slot: None }
        };
        let folded = fb
            .fold_captured(dot, &[a, i, bb, j], seed, krange, dest, L)
            .unwrap();
        if matches!(variant, TileFixtureVariant::Postprocess) {
            let one = fb.constant(tile_one(&elem), L).unwrap();
            fb.binop(Operation::Add, folded, one, Dest::Ret { slot: None }, L)
                .unwrap();
        }
        fb.finish().unwrap();
    }

    let main = b
        .declare(
            FuncKind::Named,
            "tile_main",
            Ty::Unit,
            tile_array(&elem, mapped_size),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(main).unwrap();
        let mapped_count = fb.constant(tile_count(mapped_size), L).unwrap();
        let mapped = fb.iota(mapped_count, Dest::Fresh(None), L).unwrap();
        let k_count = fb.constant(tile_count(k), L).unwrap();
        let krange = fb.iota(k_count, Dest::Fresh(None), L).unwrap();
        let one = fb.constant(tile_one(&elem), L).unwrap();
        let a_count = fb.constant(tile_count(a_size), L).unwrap();
        let a = fb.fill(one, a_count, Dest::Fresh(None), L).unwrap();
        let b_count = fb.constant(tile_count(b_size), L).unwrap();
        let bb = fb.fill(one, b_count, Dest::Fresh(None), L).unwrap();
        fb.map_captured(cell, &[krange, a, bb], mapped, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(main).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    (ir, main, dot)
}

pub(super) fn tile_matmul_fixture(
    rows: u64,
    c: u64,
    k: u64,
) -> (mapal_ir::CategoryIr, mapal_ir::FuncId, mapal_ir::FuncId) {
    tile_matmul_fixture_with(rows, c, k, Ty::f64(), TileFixtureVariant::Standard)
}

#[test]
fn tile_matmul_site_recognized() {
    let (ir, main, _) = tile_matmul_fixture(512, 512, 512);
    let plan = ir.tile_plan(main);
    assert_eq!(plan.sites.len(), 1);
    let (site_id, site) = plan.sites.iter().next().unwrap();
    assert!(matches!(
        ir.morphism(site_id).unwrap().op,
        Operation::Map { .. }
    ));
    assert_eq!(
        site,
        &TileSite {
            rows: 512,
            c: 512,
            k: 512,
            a: TileRead {
                slot: 1,
                base: 0,
                ci: 512,
                ck: 1,
                clane: 0,
                ksplit: None,
                narrow: None,
            },
            b: TileRead {
                slot: 2,
                base: 0,
                ci: 0,
                ck: 512,
                clane: 1,
                ksplit: None,
                narrow: None,
            },
            seed: Value::F64(0.0),
            elem: Ty::f64(),
            mul_a_first: true,
            add_acc_first: true,
            radix: None,
            tf32: false,
            half: None,
            post: None,
        }
    );
}

#[test]
fn bounds_matmul_fold_body_proven_through_captures() {
    let (ir, _, fold_body) = tile_matmul_fixture(8, 8, 8);
    let bounds = ir.bounds_proof(fold_body);
    let indexes = index_morphs(&ir, fold_body);
    assert_eq!(indexes.len(), 2);
    assert!(indexes.into_iter().all(|m| bounds.proven(m)));
}

#[test]
fn tile_refuses_postprocessed_fold() {
    let (ir, main, _) =
        tile_matmul_fixture_with(2, 3, 4, Ty::f64(), TileFixtureVariant::Postprocess);
    assert!(ir.tile_plan(main).sites.is_empty());
}

#[test]
fn tile_refuses_nonconstant_seed() {
    let (ir, main, _) =
        tile_matmul_fixture_with(2, 3, 4, Ty::f64(), TileFixtureVariant::NonconstantSeed);
    assert!(ir.tile_plan(main).sites.is_empty());
}

#[test]
fn tile_refuses_nonunit_lane_stride() {
    let (ir, main, fold_body) =
        tile_matmul_fixture_with(2, 3, 4, Ty::f64(), TileFixtureVariant::RowMajorB);
    let bounds = ir.bounds_proof(fold_body);
    assert!(
        index_morphs(&ir, fold_body)
            .into_iter()
            .all(|m| bounds.proven(m)),
        "row-major b[j*C+k] is in bounds; refusal is clane=C"
    );
    assert!(ir.tile_plan(main).sites.is_empty());
}

#[test]
fn tile_refuses_unproven_index() {
    let (ir, main, _) =
        tile_matmul_fixture_with(2, 3, 4, Ty::f64(), TileFixtureVariant::UndersizedA);
    assert!(ir.tile_plan(main).sites.is_empty());
}

pub(super) fn tile_fir_fixture(
    m: u64,
    k: u64,
    lane_stride: u64,
) -> (mapal_ir::CategoryIr, mapal_ir::FuncId, mapal_ir::FuncId) {
    let elem = Ty::f32();
    let w_ty = tile_array(&elem, k);
    let x_size = m
        .checked_sub(1)
        .unwrap()
        .checked_mul(lane_stride)
        .unwrap()
        .checked_add(k)
        .unwrap();
    let x_ty = tile_array(&elem, x_size);
    let k_ty = i32_arr(k);

    let mut b = IrBuilder::new();
    let dot = b
        .declare(
            FuncKind::FoldBody,
            "tile_fir_dot",
            Ty::Tuple(vec![
                w_ty.clone(),
                x_ty.clone(),
                Ty::i32(),
                elem.clone(),
                Ty::i32(),
            ]),
            elem.clone(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(dot).unwrap();
        let input = fb.input();
        let w = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let x = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        let t = fb.proj(input, 2, Dest::Fresh(None), L).unwrap();
        let acc = fb.proj(input, 3, Dest::Fresh(None), L).unwrap();
        let kk = fb.proj(input, 4, Dest::Fresh(None), L).unwrap();
        let x_base = if lane_stride == 1 {
            t
        } else {
            let stride = fb.constant(tile_count(lane_stride), L).unwrap();
            fb.binop(Operation::Mul, t, stride, Dest::Fresh(None), L)
                .unwrap()
        };
        let x_index = fb
            .binop(Operation::Add, x_base, kk, Dest::Fresh(None), L)
            .unwrap();
        let w_value = fb.index(w, kk, Dest::Fresh(None), L).unwrap();
        let x_value = fb.index(x, x_index, Dest::Fresh(None), L).unwrap();
        let product = fb
            .binop(Operation::Mul, w_value, x_value, Dest::Fresh(None), L)
            .unwrap();
        fb.binop(Operation::Add, acc, product, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }

    let sample = b
        .declare(
            FuncKind::MapBody,
            "tile_fir_sample",
            Ty::Tuple(vec![w_ty.clone(), x_ty.clone(), k_ty.clone(), Ty::i32()]),
            elem.clone(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(sample).unwrap();
        let input = fb.input();
        let w = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let x = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        let kr = fb.proj(input, 2, Dest::Fresh(None), L).unwrap();
        let t = fb.proj(input, 3, Dest::Fresh(None), L).unwrap();
        let seed = fb.constant(Value::F32(0.0), L).unwrap();
        fb.fold_captured(dot, &[w, x, t], seed, kr, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }

    let main = b
        .declare(
            FuncKind::Named,
            "tile_fir_main",
            Ty::Unit,
            tile_array(&elem, m),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(main).unwrap();
        let one = fb.constant(Value::F32(1.0), L).unwrap();
        let w_count = fb.constant(tile_count(k), L).unwrap();
        let w = fb.fill(one, w_count, Dest::Fresh(None), L).unwrap();
        let x_count = fb.constant(tile_count(x_size), L).unwrap();
        let x = fb.fill(one, x_count, Dest::Fresh(None), L).unwrap();
        let k_count = fb.constant(tile_count(k), L).unwrap();
        let kr = fb.iota(k_count, Dest::Fresh(None), L).unwrap();
        let mapped_count = fb.constant(tile_count(m), L).unwrap();
        let mapped = fb.iota(mapped_count, Dest::Fresh(None), L).unwrap();
        fb.map_captured(sample, &[w, x, kr], mapped, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(main).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    (ir, main, dot)
}

#[test]
fn tile_fir_site_recognized() {
    let (ir, main, _) = tile_fir_fixture(64, 8, 1);
    let plan = ir.tile_plan(main);
    assert_eq!(plan.sites.len(), 1);
    assert_eq!(
        plan.sites.iter().next().unwrap().1,
        &TileSite {
            rows: 1,
            c: 64,
            k: 8,
            a: TileRead {
                slot: 0,
                base: 0,
                ci: 0,
                ck: 1,
                clane: 0,
                ksplit: None,
                narrow: None,
            },
            b: TileRead {
                slot: 1,
                base: 0,
                ci: 0,
                ck: 1,
                clane: 1,
                ksplit: None,
                narrow: None,
            },
            seed: Value::F32(0.0),
            elem: Ty::f32(),
            mul_a_first: true,
            add_acc_first: true,
            radix: None,
            tf32: false,
            half: None,
            post: None,
        }
    );
}

#[test]
fn tile_refuses_fir_nonunit_lane_stride() {
    let (ir, main, fold_body) = tile_fir_fixture(64, 8, 2);
    let bounds = ir.bounds_proof(fold_body);
    assert!(
        index_morphs(&ir, fold_body)
            .into_iter()
            .all(|m| bounds.proven(m)),
        "x[2*t+k] is in bounds; refusal is clane=2"
    );
    assert!(ir.tile_plan(main).sites.is_empty());
}

pub(super) fn tile_no_split_fixture() -> (mapal_ir::CategoryIr, mapal_ir::FuncId) {
    let elem = Ty::f64();
    let array = tile_array(&elem, 8);
    let krange = i32_arr(4);
    let mut b = IrBuilder::new();
    let fold_body = b
        .declare(
            FuncKind::FoldBody,
            "tile_1d_fold",
            Ty::Tuple(vec![array.clone(), Ty::i32(), elem.clone(), Ty::i32()]),
            elem.clone(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(fold_body).unwrap();
        let input = fb.input();
        let a = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let i = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        let acc = fb.proj(input, 2, Dest::Fresh(None), L).unwrap();
        let value = fb.index(a, i, Dest::Fresh(None), L).unwrap();
        fb.binop(Operation::Add, acc, value, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let map_body = b
        .declare(
            FuncKind::MapBody,
            "tile_1d_map",
            Ty::Tuple(vec![krange.clone(), array.clone(), Ty::i32()]),
            elem.clone(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(map_body).unwrap();
        let input = fb.input();
        let kr = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let a = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        let i = fb.proj(input, 2, Dest::Fresh(None), L).unwrap();
        let seed = fb.constant(Value::F64(0.0), L).unwrap();
        fb.fold_captured(fold_body, &[a, i], seed, kr, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let main = b
        .declare(FuncKind::Named, "tile_1d", Ty::Unit, array, L)
        .unwrap();
    {
        let mut fb = b.build_fn(main).unwrap();
        let mapped_count = fb.constant(Value::I32(8), L).unwrap();
        let mapped = fb.iota(mapped_count, Dest::Fresh(None), L).unwrap();
        let k_count = fb.constant(Value::I32(4), L).unwrap();
        let kr = fb.iota(k_count, Dest::Fresh(None), L).unwrap();
        let one = fb.constant(Value::F64(1.0), L).unwrap();
        let a_count = fb.constant(Value::I32(8), L).unwrap();
        let a = fb.fill(one, a_count, Dest::Fresh(None), L).unwrap();
        fb.map_captured(map_body, &[kr, a], mapped, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(main).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    (ir, main)
}

#[test]
fn tile_refuses_non_fma_1d_map() {
    let (ir, main) = tile_no_split_fixture();
    assert!(ir.tile_plan(main).sites.is_empty());
}

#[test]
fn tile_refuses_dead_call_in_map_body() {
    let (ir, main, _) = tile_matmul_fixture_with(2, 3, 4, Ty::f64(), TileFixtureVariant::DeadCall);
    assert!(ir.tile_plan(main).sites.is_empty());
}

#[test]
fn tile_matmul_f32_site_recognized() {
    let (ir, main, _) =
        tile_matmul_fixture_with(1024, 1024, 1024, Ty::f32(), TileFixtureVariant::Standard);
    let plan = ir.tile_plan(main);
    assert_eq!(plan.sites.len(), 1);
    let site = plan.sites.iter().next().unwrap().1;
    assert_eq!(site.elem, Ty::f32());
    assert_eq!(site.seed, Value::F32(0.0));
}

// --- tile_plan: fold-body derived axes (conv2d k÷div / k%div) ----------------

#[derive(Clone, Copy)]
pub(super) struct Conv2dFixture {
    /// Output side; rows == c == side, mapped == side².
    pub(super) side: u64,
    /// Window side; the fold body's `Div` divisor.
    pub(super) div: u64,
    /// The fold body's `Mod` divisor (== div when the pair matches).
    pub(super) mod_div: u64,
    /// Fold depth k (== div² when the window is rectangular).
    pub(super) depth: u64,
    /// Also add raw `k` to the img address (mixed raw/derived refusal pin).
    pub(super) mix_raw_k: bool,
}

pub(super) fn tile_conv2d_fixture_with(
    cfg: Conv2dFixture,
) -> (mapal_ir::CategoryIr, mapal_ir::FuncId, mapal_ir::FuncId) {
    let elem = Ty::f32();
    let stride = cfg.side + cfg.div - 1;
    let max_kq = (cfg.depth - 1) / cfg.div;
    let max_addr = (cfg.side - 1 + max_kq) * stride
        + (cfg.side - 1)
        + (cfg.mod_div - 1)
        + if cfg.mix_raw_k { cfg.depth - 1 } else { 0 };
    let img_ty = tile_array(&elem, max_addr + 1);
    let w_ty = tile_array(&elem, cfg.depth);
    let k_ty = i32_arr(cfg.depth);

    let mut b = IrBuilder::new();
    let dot = b
        .declare(
            FuncKind::FoldBody,
            "tile_conv_dot",
            Ty::Tuple(vec![
                w_ty.clone(),
                Ty::i32(),
                img_ty.clone(),
                Ty::i32(),
                elem.clone(),
                Ty::i32(),
            ]),
            elem.clone(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(dot).unwrap();
        let input = fb.input();
        let w = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let i = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        let img = fb.proj(input, 2, Dest::Fresh(None), L).unwrap();
        let j = fb.proj(input, 3, Dest::Fresh(None), L).unwrap();
        let acc = fb.proj(input, 4, Dest::Fresh(None), L).unwrap();
        let kk = fb.proj(input, 5, Dest::Fresh(None), L).unwrap();
        let div_c = fb.constant(tile_count(cfg.div), L).unwrap();
        let mod_c = fb.constant(tile_count(cfg.mod_div), L).unwrap();
        let stride_c = fb.constant(tile_count(stride), L).unwrap();
        let kq = fb
            .binop(Operation::Div, kk, div_c, Dest::Fresh(None), L)
            .unwrap();
        let kr = fb
            .binop(Operation::Mod, kk, mod_c, Dest::Fresh(None), L)
            .unwrap();
        let i_kq = fb
            .binop(Operation::Add, i, kq, Dest::Fresh(None), L)
            .unwrap();
        let row = fb
            .binop(Operation::Mul, i_kq, stride_c, Dest::Fresh(None), L)
            .unwrap();
        let row_j = fb
            .binop(Operation::Add, row, j, Dest::Fresh(None), L)
            .unwrap();
        let addr = fb
            .binop(Operation::Add, row_j, kr, Dest::Fresh(None), L)
            .unwrap();
        let addr = if cfg.mix_raw_k {
            fb.binop(Operation::Add, addr, kk, Dest::Fresh(None), L)
                .unwrap()
        } else {
            addr
        };
        let x = fb.index(img, addr, Dest::Fresh(None), L).unwrap();
        let w_v = fb.index(w, kk, Dest::Fresh(None), L).unwrap();
        let product = fb
            .binop(Operation::Mul, w_v, x, Dest::Fresh(None), L)
            .unwrap();
        fb.binop(Operation::Add, acc, product, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }

    let cell = b
        .declare(
            FuncKind::MapBody,
            "tile_conv_cell",
            Ty::Tuple(vec![k_ty.clone(), w_ty.clone(), img_ty.clone(), Ty::i32()]),
            elem.clone(),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(cell).unwrap();
        let input = fb.input();
        let krange = fb.proj(input, 0, Dest::Fresh(None), L).unwrap();
        let w = fb.proj(input, 1, Dest::Fresh(None), L).unwrap();
        let img = fb.proj(input, 2, Dest::Fresh(None), L).unwrap();
        let t = fb.proj(input, 3, Dest::Fresh(None), L).unwrap();
        let side_c = fb.constant(tile_count(cfg.side), L).unwrap();
        let i = fb
            .binop(Operation::Div, t, side_c, Dest::Fresh(None), L)
            .unwrap();
        let j = fb
            .binop(Operation::Mod, t, side_c, Dest::Fresh(None), L)
            .unwrap();
        let seed = fb.constant(Value::F32(0.0), L).unwrap();
        fb.fold_captured(
            dot,
            &[w, i, img, j],
            seed,
            krange,
            Dest::Ret { slot: None },
            L,
        )
        .unwrap();
        fb.finish().unwrap();
    }

    let mapped_size = cfg.side * cfg.side;
    let main = b
        .declare(
            FuncKind::Named,
            "tile_conv_main",
            Ty::Unit,
            tile_array(&elem, mapped_size),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(main).unwrap();
        let one = fb.constant(Value::F32(1.0), L).unwrap();
        let w_count = fb.constant(tile_count(cfg.depth), L).unwrap();
        let w = fb.fill(one, w_count, Dest::Fresh(None), L).unwrap();
        let img_count = fb.constant(tile_count(max_addr + 1), L).unwrap();
        let img = fb.fill(one, img_count, Dest::Fresh(None), L).unwrap();
        let k_count = fb.constant(tile_count(cfg.depth), L).unwrap();
        let krange = fb.iota(k_count, Dest::Fresh(None), L).unwrap();
        let mapped_count = fb.constant(tile_count(mapped_size), L).unwrap();
        let mapped = fb.iota(mapped_count, Dest::Fresh(None), L).unwrap();
        fb.map_captured(cell, &[krange, w, img], mapped, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(main).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    (ir, main, dot)
}

#[test]
fn tile_conv2d_site_recognized() {
    let (ir, main, _) = tile_conv2d_fixture_with(Conv2dFixture {
        side: 16,
        div: 3,
        mod_div: 3,
        depth: 9,
        mix_raw_k: false,
    });
    let plan = ir.tile_plan(main);
    assert_eq!(plan.sites.len(), 1);
    assert_eq!(
        plan.sites.iter().next().unwrap().1,
        &TileSite {
            rows: 16,
            c: 16,
            k: 9,
            a: TileRead {
                slot: 1,
                base: 0,
                ci: 0,
                ck: 1,
                clane: 0,
                ksplit: None,
                narrow: None,
            },
            b: TileRead {
                slot: 2,
                base: 0,
                ci: 18,
                ck: 0,
                clane: 1,
                ksplit: Some(TileKSplit {
                    div: 3,
                    cq: 18,
                    cr: 1,
                }),
                narrow: None,
            },
            seed: Value::F32(0.0),
            elem: Ty::f32(),
            mul_a_first: true,
            add_acc_first: true,
            radix: None,
            tf32: false,
            half: None,
            post: None,
        }
    );
}

#[test]
fn tile_refuses_conv2d_non_rectangular_window() {
    // k/3, k%3 present but depth 10 is not divisible by 3: the pair stays
    // unbound and the raw Div/Mod refuse the site.
    let (ir, main, _) = tile_conv2d_fixture_with(Conv2dFixture {
        side: 16,
        div: 3,
        mod_div: 3,
        depth: 10,
        mix_raw_k: false,
    });
    assert!(ir.tile_plan(main).sites.is_empty());
}

#[test]
fn tile_refuses_conv2d_divisor_mismatch() {
    let (ir, main, _) = tile_conv2d_fixture_with(Conv2dFixture {
        side: 16,
        div: 3,
        mod_div: 4,
        depth: 9,
        mix_raw_k: false,
    });
    assert!(ir.tile_plan(main).sites.is_empty());
}

#[test]
fn tile_refuses_conv2d_mixed_raw_and_derived_k() {
    // addr affine in raw k AND in (k÷3, k%3): mixed forms refuse (v1).
    let (ir, main, _) = tile_conv2d_fixture_with(Conv2dFixture {
        side: 16,
        div: 3,
        mod_div: 3,
        depth: 9,
        mix_raw_k: true,
    });
    assert!(ir.tile_plan(main).sites.is_empty());
}
