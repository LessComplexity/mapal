//! Tests: GPU cost facts (LDS broadcast, LSU floor, L2 keep).

use super::*;

/// hopper-wgmma-tma: the capabilities as geometry per part — the H100
/// (`sm_90a`) has the warpgroup product, the bulk-copy engine and
/// register reallocation; the RTX 5080 (`sm_120`) the engine alone (its
/// arch-specific `sm_120a` would add reallocation); the 4090 / 3090 / A100
/// none of the three. The version enters only through `caps_of_cc`.
#[test]
fn caps_are_geometry_per_part() {
    let caps = |p: &TargetProfile| p.gpu().expect("a GPU").caps;
    let h = caps(&H100);
    assert!(h.bulk.is_some() && h.reg_realloc == Some(REG_REALLOC_SM90));
    assert!(H100.gpu().is_some_and(Gpu::has_wgmma));
    let b = caps(&RTX_5080);
    assert!(b.bulk.is_some() && b.reg_realloc.is_none());
    for p in [&RTX_4090, &RTX_3090, &A100] {
        let c = caps(p);
        assert!(c.bulk.is_none() && c.reg_realloc.is_none(), "{}", p.name);
    }
    assert_eq!(caps_of_cc(90, true), CAPS_WARPGROUP_MMA);
    assert_eq!(caps_of_cc(120, true).reg_realloc, Some(REG_REALLOC_SM90));
    assert_eq!(caps_of_cc(120, false).reg_realloc, None);
    let bc = BULK_SM90;
    assert_eq!((bc.desc_bytes, bc.oob_zero), (128, true));
    // A box: inner 32 words = the 128-byte span; a padded outer stride
    // of whole granules fits, a misaligned one, a misaligned base, a
    // row past the span or a side past the engine's does not.
    let d = [(4096, 1), (2048, 4096)];
    assert!(bc.tensor_box_fits(&d, &[32, 256], 0, 4, Some(128)));
    assert!(bc.tensor_box_fits(&[(4096, 1), (2048, 4100)], &[32, 256], 4, 4, Some(128)));
    assert!(!bc.tensor_box_fits(&[(4096, 1), (2048, 4099)], &[32, 256], 0, 4, Some(128)));
    assert!(!bc.tensor_box_fits(&d, &[32, 256], 1, 4, Some(128)));
    assert!(!bc.tensor_box_fits(&d, &[64, 256], 0, 4, Some(128)));
    assert!(!bc.tensor_box_fits(&d, &[32, 512], 0, 4, Some(128)));
}

/// hopper-wgmma-tma R2, hand-worked on the H100 SXM (132 SMs, 65 536
/// registers, 227 KiB a block):
/// - 2048 × 4096 × 4096: consumers at (65 536 − 128·40)/256 = 236 → 232
///   registers hold n = 256 (128 + 16 + 40); two consumer warpgroups
///   (bn 128), 48 KiB a stage ⇒ 4 stages; 8 × 32 = 256 tiles.
/// - 2048³ (the TF32 MLP's site, which the S53 derivation refuses): two
///   consumers at n 256 give 128 tiles < 132; the filling geometry of
///   most flops per staged byte is n 128 with two consumers (256 tiles).
/// - The depth (R1's price, `tma_copy_ns` 159 measured): a stage's 2
///   copies (318 ns) within its consume time at 2048 TF32 flop/clk/SM
///   (1.807 GHz): n 256 × bn 128 × bk 32 takes 567 ns — 32; n 128 takes
///   283 ns at 32 — 64 (566 ns), three 64 KiB stages.
/// - 128 × 4096 × 4096: no geometry fills 132 SMs at these rows without a
///   split; the largest grid — n 32, one consumer (256 tiles).
/// - `ws-regs` off: every thread keeps 65 536 / 384 → 168 registers, so
///   n ≤ 2·(168 − 56) = 224 — 128 at 2048 rows (bk 64 by the price).
/// - No warpgroup product (the 5080, the 4090): `None`.
#[test]
fn wgmma_ws_tile_is_derived_from_hopper_properties() {
    let t = |r, c, k, rr| {
        let w = H100.gpu_wgmma_ws_tile(r, c, k, rr, true).expect("sm_90a");
        assert!(w.ws && w.ws_stage_bytes() % 1024 == 0);
        assert!(w.smem_bytes() <= H100.gpu().unwrap().smem_per_block);
        (w.n, w.groups, w.bk, w.stages, w.realloc)
    };
    assert_eq!(t(2048, 4096, 4096, true), (256, 2, 32, 4, Some((40, 232))));
    assert_eq!(H100.gpu_wgmma_tile(2048, 2048, 2048).map(|w| w.ws), None);
    assert_eq!(t(2048, 2048, 2048, true), (128, 2, 64, 3, Some((40, 232))));
    assert_eq!(t(128, 4096, 4096, true).0, 32);
    assert_eq!(t(128, 4096, 4096, true).1, 1);
    assert_eq!(t(2048, 4096, 4096, false), (128, 2, 64, 3, None));
    // Without `tma-digits` a stage is one swizzle row deep.
    let w = H100
        .gpu_wgmma_ws_tile(2048, 2048, 2048, true, false)
        .unwrap();
    assert_eq!((w.n, w.bk, w.stages), (128, 32, 7));
    // k not a whole swizzle row: no box of the descriptor's layout.
    assert_eq!(H100.gpu_wgmma_ws_tile(2048, 4096, 4100, true, true), None);
    for p in [&RTX_5080, &RTX_4090, &GENERIC] {
        assert_eq!(
            p.gpu_wgmma_ws_tile(2048, 4096, 4096, true, true),
            None,
            "{}",
            p.name
        );
    }
}

/// attention-chains: the stream's broadcast 16-byte loads a clock —
/// measured (`calib.cu lds bcast`: the RTX 5080's 0.246, the A100's
/// 0.241), else the banks' one 128-byte wavefront a clock over a warp's
/// 512 bytes; the same name overrides it.
#[test]
fn lds_bcast_is_measured_or_derived() {
    let c = |p: &TargetProfile| p.gpu().unwrap().cost.unwrap_or(NO_PLAN);
    assert_eq!(c(&RTX_5080).lds_bcast(32), 0.246);
    assert_eq!(c(&A100).lds_bcast(32), 0.241);
    assert_eq!(c(&H100).lds_bcast(32), 0.25);
    let mut over: CostOver = [None; DeviceCost::NAMES.len()];
    let i = DeviceCost::NAMES
        .iter()
        .position(|x| *x == "lds_bcast_pm")
        .unwrap();
    over[i] = Some(500);
    let g = H100.gpu().unwrap().with_cost(&over);
    assert_eq!(g.cost.unwrap().lds_bcast(32), 0.5);
}

/// hopper-wgmma-tma R1's price input: the engine's ns per copy, measured
/// where given, else the loaded latency over the copies it overlaps.
#[test]
fn tma_copy_ns_is_measured_or_derived() {
    let g = *H100.gpu().unwrap();
    assert_eq!(g.tma_copy_ns(), 159);
    assert_eq!(RTX_5080.gpu().unwrap().tma_copy_ns(), 197);
    let lat = RTX_5080.gpu().unwrap().cost.unwrap().lat_ns;
    assert!(lat > 0);
    let mut over: CostOver = [None; DeviceCost::NAMES.len()];
    let i = DeviceCost::NAMES
        .iter()
        .position(|x| *x == "tma_copy_ns")
        .unwrap();
    over[i] = Some(77);
    assert_eq!(g.with_cost(&over).tma_copy_ns(), 77);
    over[i] = Some(0);
    let derived = RTX_5080.gpu().unwrap().with_cost(&over).tma_copy_ns();
    assert_eq!(derived, lat / TMA_COPIES_IN_FLIGHT);
}

/// The matrix descriptor's constant fields for the S52 bench geometries:
/// K-major, 128-byte swizzle — LBO field 1 (unused, "assumed 1"), SBO =
/// one 8 × 128 B atom between 8-row groups, mode 1. Start address and
/// base offset stay zero (the stage is filled in per panel, on an atom).
#[test]
fn wgmma_descriptor_fields_are_the_128b_swizzle_k_major_ones() {
    for n in [1024, 2048, 4096, 8192] {
        let d = H100_PCIE
            .gpu_wgmma_tile(n, n, n)
            .expect("geometry")
            .desc_fields();
        assert_eq!(d & 0x3fff, 0, "start address");
        assert_eq!((d >> 16) & 0x3fff, 1, "LBO");
        assert_eq!((d >> 32) & 0x3fff, 1024 >> 4, "SBO");
        assert_eq!((d >> 49) & 0x7, 0, "base offset");
        assert_eq!(d >> 62, 1, "128-byte swizzle");
        assert_eq!(d, 0x4000_0040_0001_0000);
    }
}

/// The placeholder honesty pin. `cuda-ada` inherits CPU-shaped constants it
/// has no business claiming as GPU measurements; this test states that in
/// the suite so nobody reads them as swept. When a GPU realization starts
/// consuming one of these, it either gets its own value or this assertion
/// changes — either way the change is deliberate and reviewed.
#[test]
fn cuda_ada_cpu_shaped_fields_are_inherited_placeholders() {
    assert_eq!(CUDA_ADA.vec_bytes, GENERIC.vec_bytes);
    assert_eq!(CUDA_ADA.vec_regs, GENERIC.vec_regs);
    assert_eq!(CUDA_ADA.l2_bytes, GENERIC.l2_bytes);
    assert_eq!(CUDA_ADA.acc_vecs_per_row, GENERIC.acc_vecs_per_row);
}

/// S54 Ampere: the compute-capability facts (CUDA C Programming Guide
/// tables for 8.0 and 8.6; the `deviceQuery` values), the ISA gates they
/// open, and the geometry derived from them; the rung rows two S54 sessions
/// measured.
#[test]
fn ampere_profiles_carry_the_compute_capability_facts() {
    let f32 = Ty::f32();
    let facts = |p: &TargetProfile| {
        let g = *p.gpu().expect("gpu");
        (
            g.arch,
            g.sms,
            g.smem_per_sm,
            g.smem_per_block,
            g.max_threads_per_sm,
            g.max_blocks_per_sm,
        )
    };
    assert_eq!(facts(&RTX_3090), ("sm_86", 82, 102_400, 101_376, 1536, 16));
    assert_eq!(facts(&A100), ("sm_80", 108, 167_936, 166_912, 2048, 32));
    for p in [&RTX_3090, &A100] {
        assert_eq!(resolve(p.name).map(|r| r.name), Some(p.name));
        let g = *p.gpu().expect("gpu");
        // cp.async, ldmatrix (sm_75) and `mma.sync` TF32 all hold from
        // sm_80; `wgmma` is sm_90a only.
        assert!(g.has_mma_sync() && !g.has_wgmma(), "{}", p.name);
        assert_eq!(p.gpu_wgmma_tile(4096, 4096, 4096), None);
        assert!(p.gpu_mma_tile(4096, 4096, 4096, true, true).is_some());
        let t = p
            .gpu_tile(4096, 4096, 4096, &f32, false, false)
            .expect("tile");
        assert!(p.gpu_tile_async(t, 4096, &f32).is_some());
        // S54, both parts: register staging beats cp.async on every grid
        // and face; resident beats retain-delta on decode; tensor cores stay.
        for wps in [0, 1, 2, 64] {
            for contract in [false, true] {
                let r = p.gpu_rungs(wps, contract);
                assert!(!r.cp_async && !r.retain_delta && r.mma, "{} {wps}", p.name);
            }
        }
    }
    // 2 warps × 4 schedulers × SMs.
    assert_eq!(RTX_3090.gpu().map(Gpu::fill_warps), Some(656));
    assert_eq!(A100.gpu().map(Gpu::fill_warps), Some(864));
    let g = |p: &TargetProfile, n| {
        let x = p.gpu_tile(n, n, n, &f32, false, false).expect("square");
        (x.bm(), x.bn(), x.threads(), x.bk, x.blocks_per_sm)
    };
    // 3090: Ada's 100 KB SM ⇒ the 4090's 4096² tile; 1024² is 64 blocks
    // < 82 SMs ⇒ halve x once.
    assert_eq!(g(&RTX_3090, 4096), (128, 128, 256, 32, 2));
    assert_eq!(g(&RTX_3090, 1024), (128, 64, 128, 32, 4));
    // A100: 164 KB / 2 blocks − 1 KB ⇒ bk 64, as on Hopper.
    assert_eq!(g(&A100, 4096), (128, 128, 256, 64, 2));
    let m = |p: &TargetProfile| {
        let x = p.gpu_mma_tile(4096, 4096, 4096, false, true).expect("mma");
        (x.wm(), x.wn(), x.bm(), x.bn(), x.bk, x.stages)
    };
    assert_eq!(m(&RTX_3090), (64, 32, 128, 128, 16, 2));
    assert_eq!(m(&A100), (64, 32, 128, 128, 32, 2));
}

/// site-geometry `fit-first` (integrate review): the reshape is taken
/// only where its block tile divides the site, so it never loses the tile
/// OFF finds (rows 64 over `c = 2⁷·257`: a 256-lane block row does not
/// divide `c`; OFF's shrunk 64-lane one does). A forced tile knob the site
/// cannot take falls back to the derived tile; the fill key ignores it.
#[test]
fn fit_first_divides_and_forced_tiles_fall_back() {
    let f32 = Ty::f32();
    let mut seen = 0;
    for p in GPU_PARTS {
        let on = crate::GpuRungs::ALL;
        let off = crate::GpuRungs {
            fit_first: false,
            ..on
        };
        let t = |r| p.gpu_tile_with(64, 128 * 257, 1024, &f32, false, r);
        if let Some(o) = t(off) {
            seen += 1;
            let n = t(on).unwrap_or_else(|| panic!("{}: fit-first lost OFF's tile {o:?}", p.name));
            assert!(64 % n.bm() == 0 && (128 * 257) % n.bn() == 0, "{}", p.name);
        }
        // tm = 3 is no power of two: the derived tile stands.
        let mut forced = *p;
        if let Machine::Gpu(g) = &mut forced.machine {
            g.geom.tm = 3;
            g.geom.resident = 4;
        }
        let r = crate::GpuRungs::default();
        let derived = p.with_geom(Geom {
            resident: 4,
            ..GEOM
        });
        assert_eq!(
            forced.gpu_tile_split(4096, 4096, 4096, &f32, false, r, 1),
            derived.gpu_tile_split(4096, 4096, 4096, &f32, false, r, 1),
            "{}",
            p.name
        );
        assert_eq!(
            forced.gpu_tile(4096, 4096, 4096, &f32, false, true),
            p.gpu_tile(4096, 4096, 4096, &f32, false, true)
        );
    }
    assert!(seen > 0, "OFF tiles the shape somewhere");
}

/// ADR-0034 D1: the search's knob file lists every knob of [`Geom`] for
/// every measured part, the part's table value first (the shipped arm),
/// and an override replaces exactly its knob.
#[test]
fn knobs_json_lists_every_knob_table_value_first() {
    let json = include_str!("../../../../../../benches/gpu/knobs.json");
    // The SXM H100 (cross-arch replay) never ran the offline search.
    for p in GPU_PARTS
        .into_iter()
        .filter(|p| p.name != "rtx-5080" && p.name != "h100")
    {
        let mut g = p.gpu().expect("a GPU part").geom;
        let row = json
            .split(&format!("\"{}\":", p.name))
            .nth(1)
            .and_then(|r| r.split('}').next())
            .unwrap_or_else(|| panic!("knobs.json has no `{}` row", p.name));
        for n in Geom::NAMES {
            let v = *g.knob(n).expect("a listed knob");
            assert!(
                row.contains(&format!("\"{n}\": [{v}")),
                "{}: `{n}` must list {v} first",
                p.name
            );
        }
    }
    let mut over = GeomOver::default();
    over[1] = Some(4);
    assert_eq!(GEOM.with(&over), Geom { split: 4, ..GEOM });
    assert_eq!(GEOM.with(&GeomOver::default()), GEOM);
}

/// A 512³ f32 site's plan on the 4090 at `blocks` blocks of 128 threads,
/// 4 a SM, a 32×32 tile with 16 accumulators a thread.
fn shape(blocks: u64, s: u64, comb: PlanComb) -> PlanShape {
    PlanShape {
        rows: 512,
        c: 512,
        k: 512,
        e: 4,
        ea: 4,
        eb: 4,
        bm: 32,
        bn: 32,
        threads: 128,
        per_sm: 4,
        acc: 16,
        tiles: blocks / s,
        blocks,
        kb: 512 / s,
        s,
        comb,
        bk: 16,
        fly: 1,
        lsu: 0,
    }
}

/// tile-efficiency T4: the LSU floor is monotone in `lsu`, absent at
/// `lsu = 0` or without `lsu_ipc_pm`, and binds a 4×4 tile (FFMA/LSU 6,
/// under the break-even 8) but not an 8×8 one (13); the pick between them
/// holds under ±20 % on the rate.
#[test]
fn lsu_floor_is_monotone_and_separates_the_tiles() {
    let p = &RTX_4090;
    let base = shape(128, 1, PlanComb::None);
    let at = |lsu, p: &TargetProfile| p.site_ns(&PlanShape { lsu, ..base }).unwrap();
    let mut prev = at(0, p).ns;
    assert_eq!(at(0, p).u_ns, 0.0);
    for lsu in [8, 32, 64, 128, 256] {
        let c = at(lsu, p);
        assert!(c.ns >= prev && c.f_ns >= c.u_ns, "lsu={lsu}");
        prev = c.ns;
    }
    let over_ipc = |pm| {
        let mut over = [None; DeviceCost::NAMES.len()];
        let i = DeviceCost::NAMES.iter().position(|n| *n == "lsu_ipc_pm");
        over[i.unwrap()] = Some(pm);
        TargetProfile {
            machine: Machine::Gpu(RTX_4090.gpu().unwrap().with_cost(&over)),
            ..RTX_4090
        }
    };
    assert_eq!(at(256, &over_ipc(0)).u_ns, 0.0);
    // 512³: a 4×4 tile (16 acc; 2 LDS.128 a k step, 10 copies and stores
    // a panel ⇒ FFMA/LSU 256/42 ≈ 6) against an 8×8 tile split 8 ways
    // (64 acc; 4 LDS.128 a step, 12 a panel ⇒ 1024/76 ≈ 13.5).
    let small = PlanShape {
        lsu: 16 * 2 + 10,
        ..base
    };
    let big = PlanShape {
        bm: 128,
        bn: 128,
        threads: 256,
        per_sm: 1,
        acc: 64,
        tiles: 16,
        blocks: 16 * 8,
        kb: 64,
        s: 8,
        comb: PlanComb::Atomic,
        lsu: 16 * 4 + 12,
        ..base
    };
    let ratio = |s: &PlanShape| (s.acc * s.bk) as f64 / s.lsu as f64;
    assert!(ratio(&small) < 8.0 && ratio(&big) > 12.0);
    let pick = |q: &TargetProfile| {
        let (a, b) = (q.site_ns(&small).unwrap(), q.site_ns(&big).unwrap());
        assert!(a.u_ns > 0.0 && b.u_ns > 0.0);
        a.ns <= b.ns
    };
    let want = pick(p);
    for pm in [400, 450, 550, 600] {
        assert_eq!(pick(&over_ipc(pm)), want, "lsu_ipc_pm={pm}");
    }
}

#[test]
fn plan_rates_only_on_calibrated_parts() {
    for p in [&RTX_4090, &RTX_5090] {
        assert!(p.plan_rates().is_some(), "{}", p.name);
    }
    for p in [&H100_PCIE, &H100, &RTX_3090, &A100, &CUDA_ADA, &GENERIC] {
        assert!(p.plan_rates().is_none(), "{}", p.name);
        assert!(p.site_ns(&shape(256, 1, PlanComb::None)).is_none());
    }
    // A `--cost=` arm on an uncalibrated part leaves it uncalibrated
    // until every rate is given.
    let mut over = [None; DeviceCost::NAMES.len()];
    over[4] = Some(500);
    let g = RTX_3090.gpu().unwrap().with_cost(&over);
    assert_eq!(g.cost.unwrap().gap_ns, 500);
    assert_eq!(g.cost.unwrap().launch_ns, 18217);
}

/// plan-cost §5: `kernel_ns` is non-decreasing in every extent of a fixed
/// plan, steps up where the grid crosses a wave multiple, and the step is
/// the tail wave's time.
#[test]
fn kernel_ns_is_monotone_and_steps_at_waves() {
    let p = &RTX_4090;
    let ns = |sh: &PlanShape| p.site_ns(sh).unwrap();
    let base = shape(256, 1, PlanComb::None);
    let k0 = ns(&base).kernel_ns;
    for grow in [
        PlanShape { rows: 1024, ..base },
        PlanShape { c: 1024, ..base },
        PlanShape { k: 1024, ..base },
        PlanShape { kb: 1024, ..base },
        PlanShape {
            blocks: 300,
            ..base
        },
        PlanShape { acc: 8, ..base },
    ] {
        assert!(ns(&grow).kernel_ns >= k0, "{grow:?}");
    }
    let w = 128 * 4;
    let one = ns(&shape(w, 1, PlanComb::None));
    let two = ns(&shape(w + 1, 1, PlanComb::None));
    assert_eq!((one.waves, one.tail, two.waves, two.tail), (1, w, 2, 1));
    // The step is the tail wave: one block at a lone block's rate.
    let lone = ns(&shape(1, 1, PlanComb::None));
    let step = two.kernel_ns - one.kernel_ns;
    assert!(
        (step - lone.f_ns.max(lone.l_ns).max(lone.c_ns)).abs() <= 1e-6 * step,
        "{step}"
    );
    let mut prev = 0.0;
    for b in (64..=4 * w).step_by(64) {
        let t = ns(&shape(b, 1, PlanComb::None)).kernel_ns;
        assert!(t >= prev, "{b}");
        prev = t;
    }
    // The split's fixed cost: the zero pass and the idle it leaves (D2).
    let at = ns(&shape(512, 4, PlanComb::Atomic));
    assert_eq!((at.zeros, at.launches, at.atomics), (1, 1, 4 * 512 * 512));
    let ws = ns(&shape(512, 4, PlanComb::Workspace));
    assert_eq!((ws.zeros, ws.launches), (0, 2));
}

/// plan-cost-terms G4: a wave lasts as long as its busiest SM. On the
/// 5090 (170 SMs) 171 or 256 blocks leave some SMs with two: the same
/// FFMA time as 340 (two everywhere), not the average's `B / 170`.
#[test]
fn a_wave_is_its_busiest_sm() {
    let p = &RTX_5090;
    let f = |blocks| {
        let sh = PlanShape {
            per_sm: 2,
            k: 1 << 16,
            kb: 1 << 16,
            ..shape(blocks, 1, PlanComb::None)
        };
        p.site_ns(&sh).unwrap().f_ns
    };
    assert_eq!(f(171), f(340));
    assert_eq!(f(256), f(340));
    assert!(f(171) > 1.5 * f(170));
}

/// stream-fly: the split puts a block on every SM and the part's
/// measured knee in flight on each — a part that needs more bytes in
/// flight an SM (the H100 SXM's 32 KiB against the RTX 5080's 8) cuts
/// finer; a knee nobody measured is Little's law at the loaded latency;
/// neither: every SM streams. The 1 × 4096 × 4096 GEMV's grids: 4
/// blocks of 16-byte loads (32 KiB a block in flight), 16 of 4-byte
/// loads (8 KiB).
#[test]
fn split_stream_reads_the_in_flight_knee() {
    let panels = 4096;
    let s = |p: &TargetProfile, blocks, fly| {
        p.split_stream(blocks, fly, 1 << 20, panels, 4096, 4, false)
    };
    assert_eq!(RTX_5080.gpu().unwrap().stream_fly_per_sm(), 8192);
    assert_eq!(H100.gpu().unwrap().stream_fly_per_sm(), 32768);
    assert_eq!(s(&RTX_5080, 4, 32768), 32);
    assert_eq!(s(&H100, 4, 32768), 64);
    assert_eq!(s(&RTX_5080, 16, 8192), 8);
    // 16 blocks of 8 KiB on the H100: every SM at 16 partitions, its
    // knee at 32 (4 MiB) — the warp fill's 16 leaves half of it.
    assert_eq!(s(&H100, 16, 8192), 64);
    // Unmeasured knee: Little's law at `lat_ns` (the 5080's 573 ns over
    // its 960 GB/s peak: 6.5 KiB an SM); neither: every SM streams.
    let mut little = *RTX_5080.gpu().unwrap();
    little.cost = little.cost.map(|d| DeviceCost {
        fly_bytes_per_sm: 0,
        bar_ns: 0,
        ..d
    });
    assert_eq!(little.stream_fly_per_sm(), 960 * 573 / 84);
    let mut none = little;
    none.cost = None;
    assert_eq!(none.stream_fly_per_sm(), 0);
    let p = TargetProfile {
        machine: Machine::Gpu(none),
        ..RTX_5080
    };
    assert_eq!(s(&p, 4, 32768), 21_u64.next_power_of_two());
    // A workspace keeps its partials in the L2; a wave caps the grid.
    assert_eq!(
        RTX_5080.split_stream(4, 32768, 64, panels, 4096, 4, false),
        16
    );
}

/// plan-cost-terms Little's law: the bytes a grid pulls per ns grow with
/// the bytes it keeps in flight (blocks × `fly` panels) and saturate at
/// the part's DRAM rate; below it the latency chain sets the time.
#[test]
fn bandwidth_saturates_with_bytes_in_flight() {
    for p in [&RTX_4090, &RTX_5090] {
        let (_, d) = p.plan_rates().unwrap();
        let gemv = |s: u64, fly: u64| PlanShape {
            rows: 1,
            c: 1024,
            k: 1 << 14,
            bm: 1,
            bn: 256,
            threads: 256,
            per_sm: 6,
            acc: 1,
            tiles: 4,
            blocks: 4 * s,
            kb: (1 << 14) / s,
            s,
            comb: PlanComb::Atomic,
            bk: NAIVE_UNROLL,
            fly,
            ..shape(1, 1, PlanComb::None)
        };
        let mut prev = 0.0;
        for (s, fly) in [(1, 1), (2, 1), (4, 1), (4, 2), (16, 1), (64, 1), (64, 4)] {
            let c = p.site_ns(&gemv(s, fly)).unwrap();
            let rate = (1 << 14) as f64 * 1024.0 * 4.0 / c.kernel_ns;
            assert!(rate >= prev * (1.0 - 1e-9), "{} S={s} fly={fly}", p.name);
            assert!(rate <= d.dram_bytes_per_ns as f64 * 1.001, "{}", p.name);
            prev = rate;
        }
        assert!(
            prev >= d.dram_bytes_per_ns as f64 * 0.99,
            "{} saturates",
            p.name
        );
        // Few blocks: the chain, not DRAM, is the time.
        let one = p.site_ns(&gemv(1, 1)).unwrap();
        assert!(one.c_ns > one.d_ns, "{}", p.name);
    }
}

/// plan-cost-terms k-depth (G3): a split shortens each partition's panel
/// trips — the chain falls with `S` while the adds grow; without
/// `lat_ns` there is no chain and the busiest-SM / warp terms are off
/// (decision D8: the sprint `plan-cost` model, op for op).
#[test]
fn a_split_trades_trips_for_its_combine() {
    let p = &RTX_4090;
    let mut prev = f64::INFINITY;
    for s in [1, 2, 4, 8] {
        let c = p.site_ns(&shape(16 * s, s, PlanComb::Atomic)).unwrap();
        assert!(c.c_ns < prev && c.c_ns > 0.0, "S={s}");
        prev = c.c_ns;
    }
    let mut over = [None; DeviceCost::NAMES.len()];
    let lat = DeviceCost::NAMES
        .iter()
        .position(|n| *n == "lat_ns")
        .unwrap();
    over[lat] = Some(0);
    let q = TargetProfile {
        machine: Machine::Gpu(RTX_4090.gpu().unwrap().with_cost(&over)),
        ..RTX_4090
    };
    let c = q.site_ns(&shape(300, 1, PlanComb::None)).unwrap();
    assert_eq!(c.c_ns, 0.0);
    assert_eq!(c.kernel_ns, (c.f_ns.max(c.l_ns)).max(c.d_ns));
}

/// plan-cost D3: a crossover is where the two cost curves meet — `alt`
/// cheaper at the reported power of two, `pick` at every power nearer 1 —
/// over unsplit vs split plans of 4…512 tiles on both calibrated parts.
#[test]
fn crossover_is_where_the_curves_meet() {
    let mut met = 0;
    for p in [&RTX_4090, &RTX_5090] {
        let ns = |sh: &PlanShape, x: f64| p.site_ns_at(sh, x).unwrap().ns;
        for tiles in [4, 16, 64, 128, 512] {
            for (s, comb) in [2, 4, 8]
                .iter()
                .flat_map(|&s| [PlanComb::Atomic, PlanComb::Workspace].map(move |c| (s, c)))
            {
                let none = shape(tiles, 1, PlanComb::None);
                let split = shape(tiles * s, s, comb);
                let (pick, alt) = if ns(&split, 1.0) <= ns(&none, 1.0) {
                    (split, none)
                } else {
                    (none, split)
                };
                let Some(x) = p.crossover(&pick, &alt) else {
                    continue;
                };
                met += 1;
                let side = |y: f64| ns(&pick, y) <= ns(&alt, y);
                assert!(!side(x), "x{x}");
                assert!(side(1.0));
                for i in 1..(x.log2().abs() as i32) {
                    let y = f64::from(i).exp2();
                    assert!(side(y) && side(1.0 / y), "x{y}");
                }
            }
        }
    }
    assert!(met >= 4, "crossovers in range ({met})");
}

/// review: **each profile's `l2_keep_bytes` is its sweep's keep**
/// ([`l2_keep_of_sweep`]): the RTX 5080's (local, 2026-11-25, from 0.25
/// ×: plateau 4.1–4.3 TB/s, cliff past 1.01 ×) at 1.025 × its 64 MiB; the
/// RTX 3090's (cross-arch 5fcb401b `calib-l2keep`, 1.0–1.05 ×: plateau
/// 1.2 TB/s to 1.02 ×) at 1.03 × its 6 MiB — the derived L2 rate's
/// midpoint (1658) put it below 1.0 × and the profile at a placeholder
/// half; the A100's and H100's sweeps (5fcb401b, flat at the DRAM rate
/// from 1.0 ×) have no plateau: their keep lies below 1.0 × (their
/// placeholders, open until the sweep from 0.25 × runs).
#[test]
fn l2_keep_read_off_the_sweeps() {
    let r5080 = [
        (0.25, 3601.0),
        (0.375, 3901.0),
        (0.5, 4096.0),
        (0.625, 4144.0),
        (0.75, 4165.0),
        (0.8125, 4226.0),
        (0.875, 4248.0),
        (0.9375, 4267.0),
        (1.0, 4161.0),
        (1.005, 4182.0),
        (1.01, 4137.0),
        (1.015, 3737.0),
        (1.02, 3153.0),
        (1.025, 2619.0),
        (1.03, 2308.0),
        (1.035, 2065.0),
        (1.04, 1801.0),
        (1.045, 1608.0),
        (1.05, 1458.0),
    ];
    let r3090 = [
        (1.0, 1205.0),
        (1.005, 1211.0),
        (1.01, 1205.0),
        (1.015, 1201.0),
        (1.02, 1217.0),
        (1.025, 1166.0),
        (1.03, 1101.0),
        (1.035, 971.0),
        (1.04, 900.0),
        (1.045, 874.0),
        (1.05, 811.0),
    ];
    let a100 = [(1.0, 1563.0), (1.02, 1550.0), (1.05, 1528.0)];
    let h100 = [(1.0, 2579.0), (1.025, 2623.0), (1.05, 2608.0)];
    let keep = |p: &TargetProfile, sweep: &[(f64, f64)], dram: f64| {
        let g = p.gpu().expect("gpu");
        let r = l2_keep_of_sweep(sweep, dram).expect("a plateau");
        ((r * g.l2_bytes as f64) as u64, g.l2_keep_bytes())
    };
    let (want, got) = keep(&RTX_5080, &r5080, 917.0);
    assert_eq!(got, want, "rtx-5080");
    let (want, got) = keep(&RTX_3090, &r3090, 892.0);
    assert_eq!(got, want, "rtx-3090");
    for (p, sweep, dram) in [(&A100, &a100, 1487.0), (&H100, &h100, 3180.0)] {
        assert_eq!(l2_keep_of_sweep(sweep, dram), None, "{}", p.name);
        let g = p.gpu().expect("gpu");
        assert!(g.l2_keep_bytes() < g.l2_bytes, "{}: below 1.0 x", p.name);
    }
}

/// gaps-disasm C5 `split-price`, replayed on its sweep: the rtx-5080's
/// 16 × 2048 × 7168 one-warp split (16 tiles, 896 panels of `bk` 8) —
/// measured S = 7 / 8 / 14 / 28 / 56 at 100.4 / 88.1 / 79.9 / 86.0 /
/// 88.1 µs — prices S = 14 cheapest and every measured count within 6 %
/// of its time; a part without `split_cell_ps` is unpriced.
#[test]
fn split_price_replays_the_mm16_sweep() {
    let (blocks, panels, cells) = (16, 896, 16 * 2048);
    let bytes = (16 * 7168 + 7168 * 2048 + 16 * 2048) * 4;
    assert_eq!(
        RTX_5080.split_priced(blocks, 16 * 64, panels, cells, bytes),
        Some(14)
    );
    let g = RTX_5080.gpu().expect("gpu");
    let d = g.plan_cost();
    let price = |s: u64| {
        bytes as f64 / d.dram_bytes_per_ns as f64
            * crate::rules::grid_spread_pm(blocks * s, g.sms) as f64
            / 1000.0
            + (s * cells) as f64 * d.split_cell_ps as f64 / 1000.0
    };
    for (s, us) in [(7, 100.4), (8, 88.1), (14, 79.9), (28, 86.0), (56, 88.1)] {
        let err = (price(s) / 1000.0 - us).abs() / us;
        assert!(
            err < 0.06,
            "S={s}: priced {:.1} vs {us} us",
            price(s) / 1000.0
        );
    }
    for p in [&RTX_4090, &RTX_3090, &A100, &H100] {
        assert_eq!(
            p.split_priced(blocks, 16 * 64, panels, cells, bytes),
            None,
            "{}",
            p.name
        );
    }
}
