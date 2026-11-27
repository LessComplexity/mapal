//! Tests: GPU tile geometry, matrix instructions and warp fill.

use super::*;

/// The GPU tile geometry is arithmetic over device facts — these pins are
/// the derivation worked by hand, so a changed rule shows up as a changed
/// number here rather than as a slower kernel on a box.
#[test]
fn gpu_tile_geometry_is_derived_from_device_facts() {
    let f32 = Ty::f32();
    // `fill` on and off: `warp-fill` is a no-op on these 4096 sites.
    let g = |p: &TargetProfile, n, fill| {
        let x = p
            .gpu_tile(n, n, n, &f32, false, fill)
            .expect("divisible square site");
        (x.bm(), x.bn(), x.threads(), x.bk, x.blocks_per_sm)
    };
    for fill in [false, true] {
        // 255 registers ⇒ 8×8 accumulators; 65 536 / 128 / 2 ⇒ 256 threads;
        // 4090 4096²: 1024 blocks ≥ 128 SMs, 2 per SM ⇒ 49 KB ⇒ bk 32.
        assert_eq!(g(&RTX_4090, 4096, fill), (128, 128, 256, 32, 2));
        // H100's 228 KB per SM doubles the panel depth.
        assert_eq!(g(&H100, 4096, fill), (128, 128, 256, 64, 2));
    }
    // OFF arm, 1024²: 64 blocks < 128 SMs ⇒ halve x once ⇒ 128 blocks.
    assert_eq!(g(&RTX_4090, 1024, false), (128, 64, 128, 32, 4));
    // 132 SMs: 128 blocks is still short ⇒ halve y too.
    assert_eq!(g(&H100, 1024, false), (64, 64, 64, 32, 8));
    // A CPU profile has no GPU geometry, and an indivisible site has none.
    assert_eq!(GENERIC.gpu_tile(4096, 4096, 4096, &f32, true, true), None);
    assert_eq!(RTX_4090.gpu_tile(1000, 1000, 1000, &f32, true, true), None);
    // 5090: 24 resident blocks per SM (the box's device query, S53), which
    // the thread and register budgets never reach here: 1024² halves to
    // 64-thread blocks ⇒ 256 blocks ≥ 170 SMs, 8 per SM by registers ⇒
    // 11.8 KB each ⇒ bk 16.
    assert_eq!(RTX_5090.gpu().map(|g| g.max_blocks_per_sm), Some(24));
    assert_eq!(g(&RTX_5090, 4096, false), (128, 128, 256, 32, 2));
    assert_eq!(g(&RTX_5090, 1024, false), (64, 64, 64, 16, 8));

    // S53 `wide-fma`: 8×16 holds 128 + 2·24 + 32 = 208 ≤ 255 (16×16 would
    // be 352); 65 536 / 208 / 2 = 157 ⇒ 128 threads; ty·8 = tx·16 ⇒ 16 × 8
    // ⇒ a 128 × 128 block; 315 / 128 ⇒ 2 blocks per SM — on every part.
    let w = |p: &TargetProfile, n| {
        let x = p
            .gpu_tile(n, n, n, &f32, true, false)
            .expect("divisible square site");
        (
            x.tm,
            x.tn,
            x.bm(),
            x.bn(),
            x.threads(),
            x.bk,
            x.blocks_per_sm,
        )
    };
    assert_eq!(fma_held_regs(8, 16, 4), 208);
    assert_eq!(w(&RTX_4090, 4096), (8, 16, 128, 128, 128, 32, 2));
    assert_eq!(w(&H100_PCIE, 4096), (8, 16, 128, 128, 128, 64, 2));
    assert_eq!(w(&H100, 4096), (8, 16, 128, 128, 128, 64, 2));
    assert_eq!(w(&RTX_5090, 4096), (8, 16, 128, 128, 128, 32, 2));
    // A small site keeps the S52 tile, and the fill rule still halves it:
    // 1024² is 64 wide blocks < every part's SMs.
    for p in [&RTX_4090, &H100_PCIE, &H100, &RTX_5090] {
        assert_eq!(
            p.gpu_tile(1024, 1024, 1024, &f32, true, false),
            p.gpu_tile(1024, 1024, 1024, &f32, false, false),
            "{}",
            p.name
        );
    }
    assert_eq!(w(&RTX_4090, 1024), (8, 8, 128, 64, 128, 32, 4));
    // 2048² is 256 wide blocks ≥ 170 SMs (5090): wide, unhalved.
    assert_eq!(w(&RTX_5090, 2048), (8, 16, 128, 128, 128, 32, 2));
    // f64 counts two registers per element: 8 × 8 (224).
    let d = RTX_4090
        .gpu_tile(4096, 4096, 4096, &Ty::f64(), true, false)
        .expect("f64");
    assert_eq!((d.tm, d.tn, fma_held_regs(8, 8, 8)), (8, 8, 224));
    // With `warp-fill`, the 5090's 2048² wide grid (1024 warps < 1360) is
    // under-filled too: the S52 tile, then filled.
    assert_eq!(
        RTX_5090.gpu_tile(2048, 2048, 2048, &f32, true, true),
        RTX_5090.gpu_tile(2048, 2048, 2048, &f32, false, true)
    );
    // The 4090's (1024 warps = 1024) is not.
    assert_eq!(
        RTX_4090.gpu_tile(2048, 2048, 2048, &f32, true, true),
        RTX_4090.gpu_tile(2048, 2048, 2048, &f32, true, false)
    );
}

/// The fill a site's base geometry gives: 4090 1024² is 128 blocks of 4
/// warps over 128 × 4 schedulers = 1 (under-filled); 2048² is 256 blocks
/// of 8 warps = 4.
#[test]
fn warps_per_scheduler_reads_the_base_grid() {
    let f32 = Ty::f32();
    let wps = |p: &TargetProfile, n| {
        let g = p
            .gpu_tile(n, n, n, &f32, false, false)
            .expect("divisible square site");
        let blocks = (n / g.bm()) * (n / g.bn());
        p.gpu()
            .expect("gpu")
            .warps_per_scheduler(blocks, g.threads())
    };
    assert_eq!(wps(&RTX_4090, 1024), 1);
    assert_eq!(wps(&RTX_4090, 2048), 4);
    assert_eq!(wps(&RTX_5090, 1024), 0);
    assert_eq!(wps(&H100_PCIE, 1024), 1);
    // S54 arch-select: `benches/gpu/rungs.py` classes a measured size by
    // this fill — Under at n ≤ 1024, Filled above — on every measured part.
    for p in GPU_PARTS {
        for n in [512, 1024, 2048, 4096, 8192] {
            let under = wps(p, n) < GPU_RESIDENT_BLOCKS;
            assert_eq!(under, n <= 1024, "{} {n}: {}", p.name, wps(p, n));
        }
    }
}

/// The rung rules over the parts' properties (hardware-properties): the
/// `cp.async` panel pays only under-filled on a compute-rich part (the
/// 4090's FFMA per DRAM byte), register staging everywhere filled.
#[test]
fn gpu_rungs_are_the_property_rules() {
    assert!(RTX_4090.gpu_rungs(1, true).cp_async);
    assert!(!RTX_4090.gpu_rungs(4, true).cp_async);
    assert!(!RTX_4090.gpu_rungs(4, false).cp_async);
    assert!(RTX_4090.gpu_rungs(1, false).cp_async);
    assert!(!H100_PCIE.gpu_rungs(1, true).cp_async);
    assert!(!RTX_5090.gpu_rungs(0, true).cp_async);
    // S54: `devmap` and `cell-law` measured positive on all five parts
    // (`devmap-<part>.txt`; the evidence is on the `GpuRungs` fields).
    for p in GPU_PARTS {
        for contract in [false, true] {
            let r = p.gpu_rungs(GPU_RESIDENT_BLOCKS, contract);
            assert!(r.devmap && r.cell_law && r.arch, "{}", p.name);
        }
    }
    // Two parts with the same properties take the same rungs, whatever
    // their names: the 4070 Ti spelling is the rules over its own.
    for wps in [0, 1, 2, 4, 64] {
        for face in [Face::Exact, Face::Contract, Face::Tf32] {
            let g = CUDA_ADA.gpu().expect("gpu");
            assert_eq!(
                CUDA_ADA.gpu_rungs_on(wps, face),
                crate::rules::rungs(g, wps, face)
            );
        }
        for contract in [false, true] {
            assert_eq!(
                RTX_3090.gpu_rungs(wps, contract),
                RTX_3090.gpu_rungs_on(wps, Face::of(contract, None))
            );
        }
    }
    let tf32 = Some(crate::Reduced::Tf32);
    assert_eq!(Face::of(true, tf32), Face::Tf32);
    assert_eq!(Face::of(true, None), Face::Contract);
    assert_eq!(Face::of(false, None), Face::Exact);
}

/// S54: `Gpu::l2_bytes` is the box's `facts l2_bytes=` on every measured
/// part (`benches/results/gpu/validation/baseline/base-<part>.txt`), and the whitepaper
/// figure on the two parts no box has reported (the 4070 Ti, H100 SXM).
#[test]
fn l2_bytes_is_the_device_query() {
    let mb = 1024 * 1024;
    for (p, facts) in [
        (&RTX_3090, 6_291_456),
        (&A100, 41_943_040),
        (&RTX_4090, 75_497_472),
        (&H100_PCIE, 52_428_800),
        (&RTX_5090, 100_663_296),
        (&H100, 50 * mb),
        (&CUDA_ADA, 48 * mb),
    ] {
        assert_eq!(p.gpu().expect("gpu").l2_bytes, facts, "{}", p.name);
    }
}

/// S54 `raster`: the group height minimises the panels one wave reads,
/// keeps row-major under the L2 gate and on a one-wave grid, and only
/// ever divides the grid's height.
#[test]
fn raster_is_the_wave_panel_minimum_behind_the_l2_gate() {
    let g = RTX_3090.gpu().expect("gpu");
    let wave = 82 * 2;
    let fma = g.plan_cost().flops_per_ns;
    // 128×128 blocks, f32: one block's panels are 1 KB per k.
    assert_eq!(
        g.raster(128, 128, wave, 4096, 4096, 4096, 4, true, 4, fma),
        1
    ); // 4 MB ≤ 6 MB
    assert_eq!(
        g.raster(128, 128, wave, 8192, 8192, 8192, 4, true, 4, fma),
        16
    ); // 8 MB > 6 MB
    // h + ⌈164 / h⌉ panels: 8 → 29, 16 → 27, 32 → 38.
    assert_eq!(
        g.raster(128, 128, wave, 8192, 8192, 8192, 4, false, 4, 0),
        16
    );
    assert_eq!(
        g.raster(128, 128, wave, 4096, 4096, 4096, 4, false, 4, 0),
        16
    );
    // One wave (16 × 8 blocks ≤ 164): order changes nothing.
    assert_eq!(
        g.raster(128, 128, wave, 1024, 2048, 8192, 4, false, 4, 0),
        1
    );
    // A grid 3 blocks high: the largest power of two dividing it is 1.
    assert_eq!(g.raster(128, 128, wave, 384, 8192, 8192, 4, false, 4, 0), 1);
    let big = RTX_4090.gpu().expect("gpu");
    let fma4 = big.plan_cost().flops_per_ns;
    assert_eq!(
        big.raster(128, 128, 256, 8192, 8192, 8192, 4, true, 4, fma4),
        1
    ); // 8 MB ≤ 72 MB
    // gapfix tensor-tile (C11): the 3090's 4096 square re-reads `b` six
    // times (15 MB of reuse distance past its keep) but its FMA pipe
    // hides them (0.08 of the pipe): row-major; unpriced, the re-read
    // alone groups it.
    assert_eq!(g.b_rereads(128, 128, wave, 4096, 4096, 4096, 4), 6);
    assert_eq!(
        g.raster(128, 128, wave, 4096, 4096, 4096, 4, true, 4, 0),
        16
    );
    // The RTX 5080's bf16 projections on its 128 × 128 tile, 2 blocks an
    // SM: 2048 × 28672 × 4096 re-reads its 235 MB `b` in each of 16
    // block-rows (224 blocks a row past the 168-block wave; 0.93 of the
    // pipe) and groups; 2048 × 4096 × 14336 (26 MB of reuse distance)
    // and 2048 × 6144 × 4096 (15 MB) keep row-major in the 66 MB keep.
    let r = RTX_5080.gpu().expect("gpu");
    let half = r.matrix_flops_per_ns(&MATRIX_WARP[1]);
    assert_eq!(r.b_rereads(128, 128, 168, 2048, 28672, 4096, 2), 15);
    assert!(r.raster(128, 128, 168, 2048, 28672, 4096, 4, true, 2, half) > 1);
    assert_eq!(r.b_rereads(128, 128, 168, 2048, 4096, 14336, 2), 0);
    assert_eq!(
        r.raster(128, 128, 168, 2048, 4096, 14336, 4, true, 2, half),
        1
    );
    assert_eq!(r.b_rereads(128, 128, 168, 2048, 6144, 4096, 2), 0);
    assert_eq!(
        r.raster(128, 128, 168, 2048, 6144, 4096, 4, true, 2, half),
        1
    );
}

/// gap-fixes merge: **the `mma-fly` ring prices its panels under the
/// walk the site takes.** The RTX 5080's 128 × 28672 × 4096 bf16
/// projection (one block-row: row-major whatever the gate) keeps its ring
/// (`bk` 16 × 4, measured 0.400 → 0.344 ms); its 2048-row twin, grouped
/// by C11's gate, reads each panel with its group's blocks and keeps the
/// S52 panel (`bk` 32 × 2: 3.99 ms against the ring's 4.52) — priced
/// row-major (`--gpu-off=raster`) it would take the ring.
#[test]
fn mma_fly_prices_the_grouped_walk() {
    let tile = |rows: u64, walk: Option<(bool, u64)>| {
        RTX_5080
            .gpu_mma16_tile(
                rows,
                28672,
                4096,
                true,
                true,
                MmaFmt::Bf16,
                2,
                2,
                false,
                Depth::Fly { walk },
                None,
            )
            .map(|t| (t.bk, t.stages))
            .expect("a tensor tile")
    };
    let panel = |rows: u64| {
        RTX_5080
            .gpu_mma16_tile(
                rows,
                28672,
                4096,
                true,
                true,
                MmaFmt::Bf16,
                2,
                2,
                false,
                Depth::Panel,
                None,
            )
            .map(|t| (t.bk, t.stages))
            .expect("a tensor tile")
    };
    let walk = Some((true, 4));
    assert_eq!(tile(128, walk), (16, 4), "the 128-row ring");
    assert_eq!(
        tile(2048, walk),
        panel(2048),
        "the grouped walk: the S52 panel"
    );
    assert_eq!(panel(2048), (32, 2));
    assert_ne!(tile(2048, None), panel(2048), "row-major: the ring");
}

/// S54 R5: the delta upload and the strided one-row site ship on every
/// measured part. Retention on the resident buffer ships on the 3090,
/// A100, 4090 and 5090 (S65) and the H100 (cpu-gpu-boundary).
#[test]
fn r5_rungs_are_the_measured_table() {
    for p in GPU_PARTS {
        for wps in [0, 64] {
            for contract in [false, true] {
                let r = p.gpu_rungs(wps, contract);
                assert!(r.resident && r.delta_h2d && r.lane_stride, "{}", p.name);
                assert!(r.retain_resident, "{}", p.name);
            }
        }
    }
}

/// S54: `rowfold` ships on every part; the reader-chosen store order on
/// none (no property separates its two 1-2 % wins from its losses: open
/// in `rung_replay.tsv`).
#[test]
fn rowfold_rungs_are_the_property_rules() {
    for p in GPU_PARTS {
        for contract in [false, true] {
            let r = p.gpu_rungs(GPU_RESIDENT_BLOCKS, contract);
            assert!(r.rowfold, "{} {contract}", p.name);
            assert!(!r.rowfold_layout, "{} {contract}", p.name);
        }
    }
}

/// S53 `pad`: the register-staged transposed-A store. With all four lanes
/// of a row in one warp no 16-byte-aligned pad is conflict-free; two lanes
/// per row (one 32-byte sector) are, at a 4-word pad.
#[test]
fn gpu_tile_sync_a_is_derived_from_the_bank_geometry() {
    let f32 = Ty::f32();
    assert_eq!(conflict_free_pad(128, |l| (4 * (l % 4), l / 4)), None);
    for p in [&RTX_4090, &H100_PCIE, &RTX_5090] {
        let g = p
            .gpu_tile(4096, 4096, 4096, &f32, true, true)
            .expect("tile");
        for bk in [8, 16, 32] {
            let s = p.gpu_tile_sync_a(GpuTile { bk, ..g }, &f32);
            assert_eq!((s.a_row_lanes, s.pad_a), (2, 4), "{} bk {bk}", p.name);
        }
        // f64 keeps the S52 map: the bank model counts words.
        let d = p.gpu_tile_sync_a(g, &Ty::f64());
        assert_eq!((d.a_row_lanes, d.pad_a), (0, 0));
    }
}

/// The tensor-core geometry, worked by hand from the device facts.
#[test]
/// tensor-cores: capabilities are geometry — each matrix instruction's
/// shape, operand format and homes, and its measured accumulator; the
/// device query's table gives the warp 16-bit products from sm_80, fp8
/// from sm_89, the shared-memory `wgmma` forms on sm_90a and the
/// tensor-memory accumulator on sm_100a; every named profile lists what
/// its compute capability's table does.
fn matrix_instructions_are_geometry() {
    let warp16 = caps_of_cc(86, false)
        .matrix_op(MmaFmt::Bf16, Home::Regs)
        .expect("sm_86: bf16 mma.sync");
    assert_eq!(
        (warp16.shape.m, warp16.shape.n, warp16.shape.k),
        (16, 8, 16)
    );
    assert_eq!(warp16.shape.threads, 32);
    assert_eq!((warp16.b, warp16.acc), (Home::Regs, Home::Regs));
    assert!(
        warp16.acc_meets(-23) && !warp16.acc_meets(-24),
        "truncating (mma_calib.cu)"
    );
    assert!(
        caps_of_cc(86, false)
            .matrix_op(MmaFmt::E4m3, Home::Regs)
            .is_none()
    );
    let f8 = caps_of_cc(120, false)
        .matrix_op(MmaFmt::E4m3, Home::Regs)
        .expect("sm_120: e4m3 mma.sync");
    assert_eq!(f8.shape.k, 32);
    assert!(
        caps_of_cc(120, false)
            .matrix_op(MmaFmt::Bf16, Home::Smem)
            .is_none()
    );
    let wg = caps_of_cc(90, true)
        .matrix_op(MmaFmt::Bf16, Home::Smem)
        .expect("sm_90a: wgmma bf16");
    assert_eq!((wg.shape.m, wg.shape.k, wg.shape.threads), (64, 16, 128));
    assert_eq!((wg.b, wg.acc), (Home::Smem, Home::Regs));
    assert!(
        caps_of_cc(90, false)
            .matrix_op(MmaFmt::Bf16, Home::Smem)
            .is_none()
    );
    let tm = caps_of_cc(100, true)
        .matrix_op(MmaFmt::Bf16, Home::Smem)
        .expect("sm_100a: tcgen05");
    assert_eq!((tm.acc, tm.shape.threads, tm.shape.m), (Home::Tmem, 1, 128));
    assert!(caps_of_cc(75, false).matrix.is_empty());
    for (p, cc, a) in [
        (RTX_3090, 86, false),
        (A100, 80, false),
        (RTX_4090, 89, false),
        (H100, 90, true),
        (H100_PCIE, 90, true),
        (RTX_5090, 120, false),
        (RTX_5080, 120, false),
    ] {
        let g = p.gpu().expect("gpu");
        assert_eq!(g.caps.matrix, caps_of_cc(cc, a).matrix, "{}", p.name);
        assert_eq!(g.has_wgmma(), a, "{}", p.name);
    }
}

#[test]
fn gpu_mma_tile_is_derived_from_device_facts() {
    let t = |p: &TargetProfile, n| {
        let x = p
            .gpu_mma_tile(n, n, n, false, true)
            .expect("divisible square site");
        (
            x.wm(),
            x.wn(),
            x.bm(),
            x.bn(),
            x.bk,
            x.stages,
            x.pad_a,
            x.pad_b,
        )
    };
    // 255 regs ⇒ 16 m16n8 tiles = 64×32 warp tile; 65 536 / 128 / 2 ⇒ 8
    // warps ⇒ 2×4 ⇒ 128×128. Pads: A rows of bk+4 words put the 8 fragment
    // rows 4 banks apart; B rows of 128+8 put the 4 k-rows 8 banks apart.
    // 4090: 49 KB per block ⇒ bk 16 (two 18.4 KB stages), 2 stages.
    assert_eq!(t(&RTX_4090, 4096), (64, 32, 128, 128, 16, 2, 4, 8));
    // H100: 113 KB per block ⇒ bk 32 (35.8 KB) ⇒ 3 stages.
    assert_eq!(t(&H100, 4096), (64, 32, 128, 128, 32, 3, 4, 8));
    // Pre-sm_80 has no `mma.sync` TF32; a CPU has no tile at all.
    assert_eq!(GENERIC.gpu_mma_tile(4096, 4096, 4096, false, true), None);
    // The wide rule: 4×8 tiles (64×64 warp, 224 registers) ⇒ 292 resident
    // threads ⇒ 4 warps per block ⇒ 2×2 warps = 128×128 (CUTLASS's).
    let w = RTX_4090
        .gpu_mma_tile(4096, 4096, 4096, true, true)
        .expect("wide");
    assert_eq!(
        (w.wm(), w.wn(), w.bm(), w.bn(), w.bk, w.stages),
        (64, 64, 128, 128, 16, 2)
    );
}

/// S62: the skinny geometries the ordered-geometry rungs derive, worked
/// from the device facts; and the squares they leave alone.
#[test]
fn ordered_skinny_geometry_is_derived_from_shape_and_device_facts() {
    let f32 = Ty::f32();
    let on = crate::GpuRungs::ALL;
    let without = |names: &[&str]| {
        let mut r = on;
        for n in names {
            *r.rung(n).expect("a rung") = false;
        }
        r
    };
    let geo = |p: &TargetProfile, m, n, k, r| {
        p.gpu_tile_with(m, n, k, &f32, false, r).map(|x| {
            (
                x.tm,
                x.tn,
                x.bm(),
                x.bn(),
                x.threads(),
                x.bk,
                x.blocks_per_sm,
            )
        })
    };
    // 5090 16×2048×7168: 8×8 ⇒ fill (1360 warps) ⇒ 4×4 ⇒ `fill-rows` 1×4
    // (256 warps: tn keeps its vector); 256 threads halve to 8 × 4 ⇒ 8 × 16
    // blocks, 256 ≥ 170 SMs. `grid-share`: ⌈256 / 170⌉ = 2 per SM ⇒ 49 KB
    // ⇒ bk 512 (24 · 512 · 4 B); without it 16 per SM ⇒ 5.2 KB ⇒ bk 32.
    assert_eq!(
        geo(&RTX_5090, 16, 2048, 7168, on),
        Some((1, 4, 8, 16, 32, 512, 2))
    );
    assert_eq!(
        geo(&RTX_5090, 16, 2048, 7168, without(&["grid-share"])),
        Some((1, 4, 8, 16, 32, 32, 16))
    );
    // Wide: the vector floor has a geometry — 4×4 at 32 threads row-fits to
    // 16 × 32, 576 blocks ≥ 170 SMs — so `fill-rows` keeps it; ⌈576 / 170⌉ =
    // 4 per SM ⇒ bk 128. Likewise M128: a vector tile exists, no scalar rows.
    assert_eq!(
        geo(&RTX_5090, 16, 18432, 7168, on),
        Some((4, 4, 16, 32, 32, 128, 4))
    );
    for (m, n, k) in [(16, 18432, 7168), (128, 2048, 7168), (128, 1024, 4096)] {
        assert_eq!(
            geo(&RTX_5090, m, n, k, on),
            geo(&RTX_5090, m, n, k, without(&["fill-rows"])),
            "{m}x{n}x{k}"
        );
    }
    // `fill-rows` off: 4×4 at 32 threads is 32 rows tall ⇒ `row-fit` 16 × 32,
    // 64 blocks < 170 SMs ⇒ `wave-gate` ⇒ the one-thread-per-cell kernel;
    // with the gate off too, the S59 geometry — one SM's share, bk 512.
    assert_eq!(
        geo(&RTX_5090, 16, 2048, 7168, without(&["fill-rows"])),
        None
    );
    assert_eq!(
        geo(
            &RTX_5090,
            16,
            2048,
            7168,
            without(&["fill-rows", "wave-gate"])
        ),
        Some((4, 4, 16, 32, 32, 512, 1))
    );
    // Every new rung off is the S53 geometry: no tile fits 16 rows.
    let old = without(&["row-fit", "wave-gate", "grid-share", "fill-rows"]);
    assert_eq!(geo(&RTX_5090, 16, 2048, 7168, old), None);
    assert_eq!(
        RTX_5090.gpu_tile(16, 2048, 7168, &f32, false, true),
        RTX_5090.gpu_tile_with(16, 2048, 7168, &f32, false, old)
    );
    // M1: one row of 32 threads, 1 × 128 blocks. 2048 columns are 16
    // blocks, 0.09 of 170 SMs: gated ([`wave_full`]). 18432 columns are
    // 144 blocks, 0.85 of the 5090's SMs: kept; on 128- and 82-SM parts
    // no whole-copy panel fits a 100 KB SM (bk ≥ 128 for whole A
    // vectors, 129 · 128 · 4 B > its share).
    assert_eq!(geo(&RTX_5090, 1, 2048, 7168, on), None);
    assert_eq!(
        geo(&RTX_5090, 1, 18432, 7168, on),
        Some((1, 4, 1, 128, 32, 128, 1))
    );
    for p in [&RTX_4090, &RTX_3090] {
        assert_eq!(geo(p, 1, 18432, 7168, on), None, "{}", p.name);
    }
    // The A100's and H100's 164/228 KB hold a panel, but 144 blocks
    // fill 0.67 / 0.63 of the SM slots their two waves span: gated.
    for p in [&A100, &H100_PCIE] {
        assert_eq!(geo(p, 1, 18432, 7168, on), None, "{}", p.name);
    }
    // The squares: every subset of the new rungs, `fill` and `wide` both
    // ways, is the old geometry at 2048² and 4096² on every part; at 1024²
    // only `grid-share` moves anything (the 4090's 128 blocks: one per SM).
    let new = ["row-fit", "wave-gate", "grid-share", "fill-rows"];
    for p in GPU_PARTS {
        for n in [1024, 2048, 4096] {
            for mask in 0..16u32 {
                let names: Vec<&str> = (0..4)
                    .filter(|b| mask >> b & 1 == 0)
                    .map(|b| new[b])
                    .collect();
                for (wide, fill) in [(false, false), (false, true), (true, false), (true, true)] {
                    // (large-gemm's `streamk-fill` off: it moves the
                    // whole-file tile, asserted below.)
                    let r = crate::GpuRungs {
                        fill,
                        kmajor_a: false,
                        streamk_fill: false,
                        ..without(&names)
                    };
                    let got = p.gpu_tile_with(n, n, n, &f32, wide, r);
                    let want = p.gpu_tile(n, n, n, &f32, wide, fill);
                    if n > 1024 || !r.grid_share {
                        assert_eq!(got, want, "{} {n} {names:?}", p.name);
                    } else {
                        let (g, w) = (got.expect("tile"), want.expect("tile"));
                        assert_eq!((g.tm, g.tn, g.tx, g.ty), (w.tm, w.tn, w.tx, w.ty));
                    }
                }
            }
        }
    }
    // large-gemm `streamk-fill`: the whole-file tile only where its grid
    // gives the part its fill target — at 2048² its 256 blocks of four
    // warps fill the 5080 (672) and the 4090 (1024), not the 5090
    // (1360), which takes the S52 tile; at 4096² every part fills.
    let sk = crate::GpuRungs {
        streamk_fill: true,
        ..crate::GpuRungs::default()
    };
    for (p, n, held) in [
        (&RTX_5080, 2048, true),
        (&RTX_4090, 2048, true),
        (&RTX_5090, 2048, false),
        (&RTX_5090, 4096, true),
    ] {
        let t = p.gpu_tile_with(n, n, n, &f32, true, sk).expect("tile");
        assert_eq!(t.held > 0, held, "{} {n}", p.name);
    }
}

/// `warp-fill` (S53): the tile shrinks until the grid holds
/// 2 · 4 schedulers · SMs warps, before the thread grid splits for blocks.
#[test]
fn gpu_warp_fill_shrinks_the_tile_until_every_scheduler_has_two_warps() {
    let f32 = Ty::f32();
    let gpu = |p: &TargetProfile| *p.gpu().expect("gpu");
    assert_eq!(gpu(&RTX_4090).schedulers(), 4);
    assert_eq!(gpu(&RTX_4090).fill_warps(), 1024);
    let fma = |p: &TargetProfile, n, fill| {
        let x = p.gpu_tile(n, n, n, &f32, false, fill).expect("divisible");
        let warps = (n / x.tm) * (n / x.tn) / 32;
        (x.tm, x.tn, x.bm(), x.bn(), x.threads(), warps)
    };
    // 4090 1024²: 8×8 gives 512 warps (1 per scheduler) ⇒ halve tn ⇒ 8×4,
    // 1024 warps (2 per scheduler); 16×16 threads ⇒ 128×64, 128 blocks.
    assert_eq!(fma(&RTX_4090, 1024, false), (8, 8, 128, 64, 128, 512));
    assert_eq!(fma(&RTX_4090, 1024, true), (8, 4, 128, 64, 256, 1024));
    // H100 PCIe (114 SMs) needs 912 ⇒ 8×4; H100 SXM (132) 1056 and the
    // 5090 (170) 1360 ⇒ 4×4, 64×64 blocks, 2048 warps.
    assert_eq!(fma(&H100_PCIE, 1024, true), (8, 4, 128, 64, 256, 1024));
    assert_eq!(fma(&H100, 1024, true), (4, 4, 64, 64, 256, 2048));
    assert_eq!(fma(&RTX_5090, 1024, true), (4, 4, 64, 64, 256, 2048));
    // Never below one vector: a 256² site floors at 4×4 and then splits
    // the thread grid for blocks as before.
    assert_eq!(fma(&RTX_4090, 256, true).0, 4);
    assert_eq!(fma(&RTX_4090, 256, true).1, 4);

    let mma = |p: &TargetProfile, n, fill| {
        let x = p.gpu_mma_tile(n, n, n, true, fill).expect("divisible");
        (x.wm(), x.wn(), x.bm(), x.bn(), (n / x.wm()) * (n / x.wn()))
    };
    // Wide 64×64 warp tile at 1024²: 256 warps (0.5 per scheduler) ⇒
    // 64×32 ⇒ 32×32, 1024 warps; 2×2 warps per block ⇒ 64×64.
    assert_eq!(mma(&RTX_4090, 1024, false), (64, 64, 128, 64, 256));
    assert_eq!(mma(&RTX_4090, 1024, true), (32, 32, 64, 64, 1024));
    // No-op wherever the grid already fills the schedulers.
    for p in [&RTX_4090, &H100, &H100_PCIE, &RTX_5090] {
        for n in [3072, 4096] {
            assert_eq!(
                p.gpu_tile(n, n, n, &f32, false, true),
                p.gpu_tile(n, n, n, &f32, false, false)
            );
            assert_eq!(
                p.gpu_mma_tile(n, n, n, true, true),
                p.gpu_mma_tile(n, n, n, true, false)
            );
        }
    }
}

/// The Hopper `wgmma` geometry (S53), worked by hand from the device facts.
#[test]
fn gpu_wgmma_tile_is_derived_from_hopper_facts() {
    let t = |n| {
        let x = H100_PCIE
            .gpu_wgmma_tile(n, n, n)
            .expect("sm_90a, divisible square site");
        assert_eq!(x.stage_words() * 4 % 1024, 0, "stages start on atoms");
        (
            x.bm(),
            x.bn(),
            x.groups,
            x.bk,
            x.stages,
            x.pad_b,
            x.blocks_per_sm,
        )
    };
    // 4096: n = 256 (128 acc + 16 fragments + 72 ≤ 255 at bk 32);
    // 65 536 / 216 = 303 threads ⇒ 2 warpgroups in one block ⇒ 256 × 128;
    // the 227 KB share holds 4 bk-32 stages of 49 KB, not 3 of bk 64.
    // 2048 and 8192 already fill 114 SMs with it.
    for n in [2048, 4096, 8192] {
        assert_eq!(t(n), (256, 128, 2, 32, 4, 8, 1), "{n}");
    }
    // 1024: 4 × 8 blocks < 114 SMs ⇒ one warpgroup, then n 128 ⇒ 128
    // blocks; 3 resident ⇒ 75 KB ⇒ exactly three bk-32 stages.
    assert_eq!(t(1024), (128, 64, 1, 32, 3, 8, 3));
    // `wgmma` is sm_90a only: not Ada, not consumer Blackwell, not a CPU.
    for p in [&RTX_4090, &RTX_5090, &CUDA_ADA, &GENERIC] {
        assert_eq!(p.gpu_wgmma_tile(4096, 4096, 4096), None, "{}", p.name);
    }
    assert!(H100.gpu().is_some_and(Gpu::has_wgmma));
}

/// bugs-properties `ptx-isa`: each part's target encoding is its own
/// SM's (the PTX ISA table, as `caps_of_cc` builds it for the device
/// query), the fp8 conversions need PTX 8.1 on `sm_89` and 7.8 from
/// `sm_90`, and a module's declaration line round-trips.
#[test]
fn ptx_isa_per_part() {
    for (p, cc, a) in [
        (RTX_3090, 86, false),
        (A100, 80, false),
        (RTX_4090, 89, false),
        (CUDA_ADA, 89, false),
        (H100, 90, true),
        (H100_PCIE, 90, true),
        (RTX_5090, 120, false),
        (RTX_5080, 120, false),
    ] {
        let g = p.gpu().expect("gpu");
        let arch = format!("sm_{cc}{}", if a { "a" } else { "" });
        assert_eq!(g.arch, arch, "{}", p.name);
        assert_eq!(g.caps.ptx, ptx_of_target(cc, a), "{}", p.name);
        assert_eq!(g.caps.cvt_f8, caps_of_cc(cc, a).cvt_f8, "{}", p.name);
    }
    assert_eq!(caps_of_cc(89, false).cvt_f8, Some(Ptx::new(81, 89)));
    assert_eq!(caps_of_cc(86, false).cvt_f8, None);
    assert_eq!(
        (ptx_of_target(90, true).isa, ptx_of_target(120, false).isa),
        (80, 87)
    );
    let d = format!(
        "; mapal-backend-llvm device module\n{}; mapal.device target=x\n",
        Ptx::new(81, 89).declare()
    );
    assert_eq!(Ptx::declared(&d), Some(Ptx::new(81, 89)));
    assert_eq!(Ptx::new(81, 89).mattr(), "+ptx81");
    assert_eq!(Ptx::declared("; mapal-backend-llvm device module\n"), None);
}
