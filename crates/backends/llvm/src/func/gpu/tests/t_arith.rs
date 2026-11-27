//! Tests: arithmetic policy, stream-K price, registry, cooperative tiles, raster.

use super::*;

/// fold-algebra: a strided-lane site's step is an FMA pair of the step
/// algebra, and the host fuses it under `contract` — so the device does
/// too, on the Contract policy. The OFF arm (`--no-fma-pairs`) keeps the
/// unit-lane rule: separate `fmul` + `fadd`, the Exact policy.
#[test]
fn strided_contract_fuses_from_the_step_algebra() {
    let s = site(1, 256, 128, true, false, Ty::f32());
    for pairs in [true, false] {
        let arms = crate::GpuArms {
            pairs,
            ..crate::GpuArms::default()
        };
        let text = kernel_text(
            "strided",
            &s,
            &H100_PCIE,
            true,
            arms,
            None,
            false,
            None,
            Combine::None,
        );
        assert_eq!(text.contains("call float @llvm.fma"), pairs);
        assert_eq!(
            text.contains(" = fmul float ") && text.contains(" = fadd float "),
            !pairs
        );
        let exact = launch(&s, &H100_PCIE, false, arms, None);
        let requested = launch(&s, &H100_PCIE, true, arms, None);
        assert_eq!(
            requested.arithmetic,
            if pairs {
                Arithmetic::OrderedFma
            } else {
                Arithmetic::SeparateMulAdd
            }
        );
        if !pairs {
            assert_eq!(
                requested.rungs, exact.rungs,
                "separate arithmetic needs Exact policy"
            );
        }
    }
}

#[test]
fn actual_arithmetic_rejected_tf32_selects_scalar_policy() {
    use crate::profile::RTX_3090;
    let s = site(256, 256, 128, false, true, Ty::f32());
    let arms = crate::GpuArms::default();
    let text = kernel_text(
        "fallback",
        &s,
        &RTX_3090,
        true,
        arms,
        Some(Reduced::Tf32),
        false,
        None,
        Combine::None,
    );
    assert!(text.contains("call float @llvm.fma"));
    assert!(!text.contains("mma.sync") && !text.contains("wgmma.mma_async"));
    let scalar = launch(&s, &RTX_3090, true, arms, None);
    let requested = launch(&s, &RTX_3090, true, arms, Some(Reduced::Tf32));
    assert_eq!(
        requested.rungs, scalar.rungs,
        "rejected TF32 needs scalar policy"
    );
}

#[test]
fn actual_arithmetic_scalar_modes_preserve_order_and_geometry() {
    for p in PARTS {
        for n in [7, 256] {
            // naive and cooperative scalar realizations
            for elem in [Ty::f32(), Ty::f64()] {
                let mut s = site(n, n, 128, true, true, elem);
                s.mul_a_first = false;
                s.add_acc_first = false;
                s.seed = if s.elem == Ty::f32() {
                    Value::F32(2.0)
                } else {
                    Value::F64(2.0)
                };
                for contract in [false, true] {
                    let arms = crate::GpuArms::default();
                    let l = launch(&s, p, contract, arms, None);
                    let expected = if contract {
                        Arithmetic::OrderedFma
                    } else {
                        Arithmetic::SeparateMulAdd
                    };
                    assert_eq!(l.arithmetic, expected);
                    assert_eq!(l.tiled.is_some(), n == 256, "{} {s:?}", p.name);
                    let text = kernel_text(
                        "scalar",
                        &s,
                        p,
                        contract,
                        arms,
                        None,
                        false,
                        None,
                        Combine::None,
                    );
                    assert_eq!(text.contains("@llvm.fma"), contract);
                    assert_eq!(text.contains(" = fmul "), !contract);
                    assert_eq!(text.contains(" = fadd "), !contract);
                    // Exercise the same step builder with named values so
                    // the source order is checked independently of SSA ids.
                    let mut k = Kernel {
                        out: String::new(),
                        next: 0,
                        idx: "i64",
                        t: lower_ty(&s.elem).unwrap(),
                        site: &s,
                        arithmetic: l.arithmetic,
                        rungs: l.rungs,
                        profile_gpu: *p.gpu().unwrap(),
                        pdl: false,
                        async_tail: None,
                        sync_tail: None,
                        raster: (l.gx, l.raster),
                        col: false,
                        epi: None,
                        splitk: 1,
                        gy_out: l.gy,
                        z: None,
                        atomic: false,
                        terms: 1,
                        streamk: None,
                        whole: 0,
                        ticket: false,
                        part: false,
                        p0: None,
                        head: "entry",
                        tmap: Vec::new(),
                        streamed: [false; 2],
                    };
                    let result = k.step("%acc", "%a", "%b");
                    let t = &k.t;
                    if contract {
                        assert!(k.out.contains(&format!("({t} %a, {t} %b, {t} %acc)")));
                    } else {
                        assert!(k.out.contains(&format!("fmul {t} %b, %a")));
                        assert!(
                            k.out.contains(&format!("{result} = fadd {t} %v2, %acc")),
                            "{}",
                            k.out
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn actual_arithmetic_tensor_rejections_use_complete_scalar_plan() {
    use crate::profile::{Machine, RTX_3090};
    let old = TargetProfile {
        machine: Machine::Gpu(Gpu {
            arch: "sm_70",
            caps: crate::profile::caps_of_cc(70, false),
            ..*RTX_4090.gpu().unwrap()
        }),
        ..RTX_4090
    };
    let ordinary = site(256, 256, 128, true, true, Ty::f32());
    let mut cases = vec![
        ("type", site(256, 256, 128, true, true, Ty::f64())),
        ("a-layout", site(256, 256, 128, false, true, Ty::f32())),
        ("b-layout", site(256, 256, 128, true, false, Ty::f32())),
        ("rows", site(1, 256, 128, true, true, Ty::f32())),
        ("columns", site(256, 7, 128, true, true, Ty::f32())),
        ("reduction", site(256, 256, 7, true, true, Ty::f32())),
    ];
    for (name, field) in [
        ("a-base", 0),
        ("b-base", 1),
        ("a-row", 2),
        ("b-row", 3),
        ("a-lane", 4),
        ("b-row-dependence", 5),
    ] {
        let mut s = ordinary.clone();
        match field {
            0 => s.a.base = 1,
            1 => s.b.base = 1,
            2 => s.a.ci += 1,
            3 => s.b.ck += 1,
            4 => s.a.clane = 1,
            _ => s.b.ci = 1,
        }
        cases.push((name, s));
    }
    let off = force(GpuRungs {
        mma: false,
        ..GpuRungs::ALL
    });
    for p in [&RTX_3090, &RTX_4090, &H100_PCIE, &H100, &RTX_5090, &old] {
        for (why, s) in &cases {
            for contract in [false, true] {
                for arms in [crate::GpuArms::default(), force(GpuRungs::ALL), off] {
                    let scalar = launch(s, p, contract, arms, None);
                    let l = launch(s, p, contract, arms, Some(Reduced::Tf32));
                    assert_eq!(l, scalar, "{} {why} contract={contract}", p.name);
                    let text = kernel_text(
                        "rejected",
                        s,
                        p,
                        contract,
                        arms,
                        Some(Reduced::Tf32),
                        false,
                        None,
                        Combine::None,
                    );
                    // (The plan line names the site's set, `set=tf32|f32`.)
                    let code: String = text
                        .lines()
                        .filter(|l| !l.starts_with("; mapal.plan"))
                        .collect();
                    assert!(
                        !code.contains("tf32")
                            && !code.contains("mma.sync")
                            && !code.contains("wgmma.mma_async")
                    );
                    assert_eq!(
                        text.contains("@llvm.fma"),
                        l.arithmetic == Arithmetic::OrderedFma
                    );
                }
            }
        }
        // Valid operands with MMA forced off, and an unsupported device
        // with every optimization forced on, must still use scalar policy.
        for arms in if !p.gpu().unwrap().has_mma_sync() {
            vec![off, force(GpuRungs::ALL)]
        } else {
            vec![off]
        } {
            for contract in [false, true] {
                assert_eq!(
                    launch(&ordinary, p, contract, arms, Some(Reduced::Tf32)),
                    launch(&ordinary, p, contract, arms, None),
                );
            }
        }
    }
}

#[test]
fn actual_arithmetic_supported_tensor_controls_keep_tf32_policy() {
    let s = site(1024, 1024, 1024, true, true, Ty::f32());
    let mma = force(GpuRungs {
        wgmma: false,
        ..GpuRungs::ALL
    });
    for (p, arms, expected, instruction) in [
        (
            &RTX_4090,
            force(GpuRungs::ALL),
            Arithmetic::Tf32Mma,
            "mma.sync.aligned",
        ),
        (&H100_PCIE, mma, Arithmetic::Tf32Mma, "mma.sync.aligned"),
        (
            &H100_PCIE,
            force(GpuRungs::ALL),
            Arithmetic::Tf32Wgmma,
            "wgmma.mma_async",
        ),
    ] {
        for contract in [false, true] {
            let l = launch(&s, p, contract, arms, Some(Reduced::Tf32));
            assert_eq!(l.arithmetic, expected);
            assert_eq!(l.arithmetic.face(), Face::Tf32);
            assert_eq!(l.rungs, GpuRungs::ALL.with(arms));
            assert!(l.block <= p.gpu().unwrap().max_threads_per_block);
            assert!(l.smem <= p.gpu().unwrap().smem_per_block);
            let text = kernel_text(
                "tensor",
                &s,
                p,
                contract,
                arms,
                Some(Reduced::Tf32),
                false,
                None,
                Combine::None,
            );
            assert!(text.contains(instruction));
            assert!(!text.contains("@llvm.fma") && !text.contains(" = fmul "));
        }
    }
}

#[test]
fn actual_arithmetic_mixed_sites_keep_module_pdl_agreement() {
    let sites = [
        site(1, 256, 128, true, false, Ty::f32()),
        site(256, 256, 128, true, true, Ty::f32()),
    ];
    for p in [&H100_PCIE, &RTX_4090] {
        for enabled in [false, true] {
            let arms = force(GpuRungs {
                pdl: enabled,
                ..GpuRungs::ALL
            });
            let wait = pdl_part(p, true, arms);
            assert_eq!(wait, enabled && p.gpu().unwrap().has_griddepcontrol());
            let host = init_call(host_rungs(p, true, arms), false, wait, false, 0);
            let flags: u32 = host
                .split("i32 ")
                .nth(1)
                .unwrap()
                .split(')')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            assert_eq!(flags & GPU_INIT_PDL != 0, wait);
            // fold-algebra: the strided site fuses with the pairs arm on.
            let strided = if arms.pairs {
                Arithmetic::OrderedFma
            } else {
                Arithmetic::SeparateMulAdd
            };
            for (s, expected) in sites.iter().zip([strided, Arithmetic::OrderedFma]) {
                assert_eq!(launch(s, p, true, arms, None).arithmetic, expected);
                let text = kernel_text("mixed", s, p, true, arms, None, false, None, Combine::None);
                assert_eq!(text.contains("griddepcontrol.wait"), wait);
            }
        }
    }
}

/// sweep-gemm-2 `streamk-price` on the rtx-5080's under-filled sites (the
/// gen_tiles layouts): loopmm's 1024³ (128 tiles of 64 panels on 168
/// slots) streams; 1024 × 1024 × 256 (16 panels a tile: the pieces'
/// partials and the zero pass outweigh the three panels the walk saves)
/// and the 512³ cells (32 panels of a 64 × 32 tile) do not; the OFF arm
/// and a part without the rates (`DeviceCost` zero) keep the classes'
/// rule (no under-filled stream-K).
#[test]
fn streamk_price_separates_the_measured_sites() {
    use crate::profile::{A100, H100, H100_PCIE, RTX_3090, RTX_4090, RTX_5080, RTX_5090};
    let free = |s: &TileSite, p: &TargetProfile, arms| {
        launch_in(s, p, true, arms, None, Order::Free).combine
    };
    let arms = crate::GpuArms::default();
    let mut unpriced = crate::GpuArms::default();
    *unpriced.off.rung("streamk-price").expect("a rung") = true;
    let sk = |c: Combine| matches!(c, Combine::StreamK { .. });
    let loopmm = site(1024, 1024, 1024, true, true, Ty::f32());
    assert!(sk(free(&loopmm, &RTX_5080, arms)), "loopmm streams");
    assert!(
        !sk(free(&loopmm, &RTX_5080, unpriced)),
        "OFF: the class's rule"
    );
    for (m, n, k) in [(1024, 1024, 256), (512, 512, 512)] {
        let s = site(m, n, k, true, true, Ty::f32());
        assert!(!sk(free(&s, &RTX_5080, arms)), "{m}x{n}x{k}");
    }
    // review-fixes: a part that measured none of the three rates is
    // priced on the ones its properties derive (`Gpu::plan_cost`) —
    // never left unpriced (the OFF arm is the class's rule); the 512³
    // cells' 8 panels a tile refuse there as on the 5080.
    for p in [&A100, &H100, &H100_PCIE, &RTX_3090] {
        let g = p.gpu().expect("gpu");
        assert!(
            g.cost
                .is_none_or(|d| d.atomic_ops_per_ns == 0 || d.zero_ns == 0),
            "{}: unmeasured",
            p.name
        );
        let d = g.plan_cost();
        assert!(
            d.flops_per_ns > 0 && d.atomic_ops_per_ns > 0 && d.zero_ns > 0,
            "{}: {d:?}",
            p.name
        );
        // The FMA tile (the A100's contract face takes TF32x3 there).
        let mut fma = arms;
        *fma.off.rung("split-tile").expect("a rung") = true;
        let l = launch_in(&loopmm, p, true, fma, None, Order::Free);
        let t = l.tiled.expect("an FMA tile");
        let wave = l.wave(g);
        let blocks = l.gx * l.gy;
        assert!(
            streamk_priced(p, &loopmm, &t, wave, blocks, wave, 1, 1, 1).is_some(),
            "{}: priced",
            p.name
        );
        assert!(
            !sk(free(&loopmm, p, unpriced)),
            "{}: OFF the class's rule",
            p.name
        );
        let s = site(512, 512, 512, true, true, Ty::f32());
        assert!(!sk(free(&s, p, arms)), "{}: 512³", p.name);
    }
    // The measured parts' rates are their own.
    for p in [&RTX_5080, &RTX_4090, &RTX_5090] {
        let g = p.gpu().expect("gpu");
        let (d, m) = (g.plan_cost(), g.cost.expect("measured"));
        assert_eq!(
            (d.flops_per_ns, d.atomic_ops_per_ns, d.zero_ns),
            (m.flops_per_ns, m.atomic_ops_per_ns, m.zero_ns)
        );
    }
}

/// sweep-gemm-2 `warm-zero`: the init flag is set exactly for a module
/// whose launches zero an output — a direct zero call or a site table row
/// with zero bytes — and the rung is on.
#[test]
fn warm_zero_flag_follows_the_modules_zeros() {
    let row = |zero: u64| {
        format!(
            "%mapal_cand {{ i64 7, i64 0, i64 {zero}, i64 0, i32 4, i32 32, i32 256, i32 0, i32 0, i32 0 }}"
        )
    };
    let table = |zeros: &[u64]| {
        let rows: Vec<String> = zeros.iter().map(|&z| row(z)).collect();
        format!(
            "@f.cands1 = private unnamed_addr constant [{} x %mapal_cand] [{}]\n",
            zeros.len(),
            rows.join(", ")
        )
    };
    assert!(module_zeroes(&table(&[16384, 0])));
    assert!(module_zeroes(&table(&[0, 4096])));
    assert!(!module_zeroes(&table(&[0, 0])));
    // `streamk-epi`: a workspace without a reduce kernel is zeroed by the
    // runtime (integration 3: the merge of warm-zero with the ticket); a
    // split's workspace with its reduce kernel is not.
    let ws = |rid: i64, ws: u64| {
        format!(
            "@g.cands1 = private unnamed_addr constant [1 x %mapal_cand] [%mapal_cand {{ i64 7, i64 {rid}, \
                 i64 0, i64 {ws}, i32 84, i32 1, i32 256, i32 0, i32 0, i32 0 }}]\n"
        )
    };
    assert!(module_zeroes(&ws(0, 1 << 20)));
    assert!(!module_zeroes(&ws(-5, 1 << 20)));
    assert!(!module_zeroes(&ws(0, 0)));
    assert!(module_zeroes(
        "  call void @mapal_gpu_zero(i64 %d3, i64 65536)\n"
    ));
    assert!(!module_zeroes(
        "  call void @mapal_gpu_launchv(i64 1, i32 0)\n"
    ));
    let flags = |rungs: GpuRungs, zeroes: bool| -> u32 {
        init_call(rungs, false, false, zeroes, 0)
            .split("i32 ")
            .nth(1)
            .and_then(|s| s.split(')').next())
            .and_then(|s| s.parse().ok())
            .expect("a flag word")
    };
    assert_ne!(flags(GpuRungs::ALL, true) & GPU_INIT_WARM_ZERO, 0);
    assert_eq!(flags(GpuRungs::ALL, false) & GPU_INIT_WARM_ZERO, 0);
    let off = GpuRungs {
        warm_zero: false,
        ..GpuRungs::ALL
    };
    assert_eq!(flags(off, true) & GPU_INIT_WARM_ZERO, 0);
    // Every other bit is the rung-off module's.
    assert_eq!(
        flags(GpuRungs::ALL, true) & !GPU_INIT_WARM_ZERO,
        flags(off, true)
    );
    // hopper-launch `tma-warm`: shipped, the flag word is the one before
    // the rung (the runtime reads the module's recipes itself); the OFF
    // arm sets one bit and nothing else.
    assert_eq!(flags(GpuRungs::ALL, true) & GPU_INIT_TMA_COLD, 0);
    let cold = GpuRungs {
        tma_warm: false,
        ..GpuRungs::ALL
    };
    assert_eq!(
        flags(cold, true),
        flags(GpuRungs::ALL, true) | GPU_INIT_TMA_COLD
    );
    // gaps-disasm C1 `zero-host`: the zero kernel's bit exactly where the
    // module zeroes; the OFF arm clears that one bit.
    assert_ne!(flags(GpuRungs::ALL, true) & GPU_INIT_ZERO_KERNEL, 0);
    assert_eq!(flags(GpuRungs::ALL, false) & GPU_INIT_ZERO_KERNEL, 0);
    let memset = GpuRungs {
        zero_host: false,
        ..GpuRungs::ALL
    };
    assert_eq!(
        flags(memset, true),
        flags(GpuRungs::ALL, true) & !GPU_INIT_ZERO_KERNEL
    );
}

/// gaps-disasm C1 `load-dispatch`: one registry row a table (its first
/// call's site and flags, however many calls launch it), the registration
/// ahead of the init; none without a table or with the rung off.
#[test]
fn site_registry_lists_each_table_once_before_the_init() {
    let call = |site: i64, flags: u32, table: &str| {
        format!(
            "  call void @mapal_gpu_site(i64 {site}, i32 {flags}, i32 0, i32 3, ptr {table}, i32 3, ptr %p1, i32 2, ptr null)\n"
        )
    };
    let host = call(-5, 1, "@f.cands1") + &call(-5, 1, "@f.cands1") + &call(7, 8, "@g.cands4");
    let host = format!(
        "declare void @mapal_gpu_site(i64, i32, i32, i32, ptr, i32, ptr, i32, ptr)\n{host}"
    );
    let (n, reg) = site_registry(&host);
    assert_eq!(n, 2, "{reg}");
    assert!(reg.starts_with(SITES_DECL), "{reg}");
    assert!(
        reg.contains(
            "[2 x %mapal_site_reg] [%mapal_site_reg { i64 -5, ptr @f.cands1, i32 3, i32 1 }, \
                 %mapal_site_reg { i64 7, ptr @g.cands4, i32 3, i32 8 }]"
        ),
        "{reg}"
    );
    assert_eq!(
        site_registry("  call void @mapal_gpu_launchv(i64 1)\n"),
        (0, String::new())
    );
    let init = init_call(GpuRungs::ALL, false, false, false, n);
    assert!(
        init.starts_with(
            "  call void @mapal_gpu_sites(ptr @mapal_sites, i32 2)\n  call void @mapal_gpu_init("
        ),
        "{init}"
    );
    let off = GpuRungs {
        load_dispatch: false,
        ..GpuRungs::ALL
    };
    assert_eq!(
        init_call(off, false, false, false, n),
        init_call(GpuRungs::ALL, false, false, false, 0)
    );
}

/// S62: the skinny M16 projections take the cooperative ordered tile on the
/// 5090 — on S53 code both took the one-thread-per-cell kernel (`None`).
/// 2048 columns: `fill-rows` 1×4 at 8 × 4 threads, 256 blocks; two per SM
/// (`grid-share`) ⇒ bk 512; `panel-regs` (255 − 46 registers free for
/// the panel at ¾ register per `k`) ⇒ bk 256. 18432 columns (Filled):
/// the 5090 ships `row-fit` OFF on the exact face (no effect measured), so
/// the vector floor does not fit 16 rows and `fill-rows` takes 1×4 at
/// 16 × 16 threads, 288 blocks, `cp.async`; with `row-fit` the floor 4×4
/// row-fits to 16 × 32. The contract face ships `warp-fill` and
/// `wave-gate` OFF there: 8×8 row-fits to 144 blocks (gated: the
/// one-thread-per-cell kernel, 4.5× slower on the 5090). `panel-regs`
/// reads only grids that leave a scheduler without a warp: not 1024² or
/// 128 × 18432 (3 warps per scheduler), whose launch is its OFF arm's.
/// tile-efficiency T3 `async-depth`: a `cp.async` panel below `tm·tn·bk
/// ≥ 1024` yields to register staging only where staging's panel is
/// deeper (4090 128×1024²: 4×4 at bk 16 → staged bk 32); at equal depth
/// it stays (5090 16×18432×7168, bk 64). There is no `k` test: a `k`
/// with no deeper panel gives staging none either. OFF: the deepest
/// `cp.async` panel that fits.
#[test]
fn async_depth_yields_only_to_a_deeper_staged_panel() {
    let geo = |p: &TargetProfile, (m, n, k), off: bool| {
        let mut a = crate::GpuArms::default();
        a.off.async_depth = off;
        // The panel depth the launch's own SM share admits (`grid-share`,
        // on wherever the L2 share per SM holds deeper panels) is not
        // the yield this tests.
        a.off.grid_share = true;
        let s = site(m, n, k, true, true, Ty::f32());
        launch(&s, p, false, a, None)
            .tiled
            .map(|g| (g.tm * g.tn, g.bk, g.stages))
    };
    let mm = (128, 1024, 1024);
    assert_eq!(geo(&RTX_4090, mm, true), Some((16, 16, 2)));
    assert_eq!(geo(&RTX_4090, mm, false), Some((16, 32, 1)));
    let skinny = (16, 18432, 7168);
    assert_eq!(geo(&RTX_5090, skinny, false), geo(&RTX_5090, skinny, true));
}

#[test]
fn skinny_projections_take_the_ordered_cooperative_tile() {
    let geo = |n, contract, arms| {
        let s = site(16, n, 7168, true, true, Ty::f32());
        launch(&s, &RTX_5090, contract, arms, None)
            .tiled
            .map(|g| (g.tm, g.tn, g.bm(), g.bn(), g.threads(), g.bk, g.stages))
    };
    let shipped = crate::GpuArms::default();
    let off = |r: &str| {
        let mut a = crate::GpuArms::default();
        *a.off.rung(r).expect("a rung") = true;
        a
    };
    let on = |r: &str| {
        let mut a = crate::GpuArms::default();
        *a.on.rung(r).expect("a rung") = true;
        a
    };
    for contract in [false, true] {
        assert_eq!(
            geo(2048, contract, shipped),
            Some((1, 4, 8, 16, 32, 256, 1))
        );
    }
    // hardware-properties: `row-fit` ships filled on the exact face too
    // (its S62 exact cell here was null), so the 4 × 4 tile row-fits.
    assert_eq!(geo(18432, false, shipped), Some((4, 4, 16, 32, 32, 64, 1)));
    // tile-efficiency T2 / per-site-choice A2: the k-major panel's bank
    // pad (4 words a row at bk 64) leaves two `cp.async` stages of bk 64
    // over the share, so the padded async panel halves to bk 32 and
    // register staging's deeper panel — k-major, padded, conflict-free —
    // takes the site, as on the transposed arm.
    let mut kmajor = on("row-fit");
    *kmajor.on.rung("kmajor-a").expect("a rung") = true;
    // hardware-properties: the rules ship no `cp.async` panel filled (it
    // lost filled on every measured part), so both bank-plan arms take
    // register staging's panel.
    assert_eq!(geo(18432, false, kmajor), Some((4, 4, 16, 32, 32, 64, 1)));
    *kmajor.on.rung("bank-plan").expect("a rung") = true;
    assert_eq!(geo(18432, false, kmajor), Some((4, 4, 16, 32, 32, 64, 1)));
    let s = site(16, 18432, 7168, true, true, Ty::f32());
    let k = launch(&s, &RTX_5090, false, kmajor, None)
        .tiled
        .expect("tiled");
    assert_eq!((k.kmajor, k.pad_a), (true, 4));
    assert_eq!(panel_excess(&s, &k, &GpuRungs::ALL, false), [1; 4]);
    assert_eq!(
        geo(18432, false, on("row-fit")),
        Some((4, 4, 16, 32, 32, 64, 1))
    );
    assert_eq!(geo(18432, true, shipped), Some((8, 8, 16, 128, 32, 16, 1)));
    // hardware-properties: the gate forced on keeps the 144-block grid —
    // it fills 0.85 of the 5090's SM slots ([`crate::profile::wave_full`];
    // the S62 v2 cell: kept 0.46 ms, gated one thread a cell 2.07 ms).
    assert_eq!(
        geo(18432, true, on("wave-gate")),
        Some((8, 8, 16, 128, 32, 16, 1))
    );
    assert_eq!(
        geo(2048, false, off("panel-regs")),
        Some((1, 4, 8, 16, 32, 16, 1))
    );
    for (m, n, k) in [(1024, 1024, 1024), (128, 18432, 7168)] {
        for contract in [false, true] {
            let at = |arms| {
                let l = launch(
                    &site(m, n, k, true, true, Ty::f32()),
                    &RTX_5090,
                    contract,
                    arms,
                    None,
                );
                (l.tiled, l.smem)
            };
            assert_eq!(at(shipped), at(off("panel-regs")), "{m}x{n}x{k} {contract}");
        }
    }
}

/// Every staged copy of a skinny site's panels lands once, in the panel,
/// on every rung setting the S62 geometry rungs reach (row-fitted, gated
/// off, scalar-row register tiles, f32 and f64, every operand layout).
#[test]
fn skinny_cooperative_copies_cover_each_panel_once() {
    use crate::profile::{A100, RTX_3090};
    use std::collections::BTreeSet;
    let geometry = |r: GpuRungs, names: &[&str]| {
        let mut r = r;
        for n in names {
            *r.rung(n).expect("a rung") = false;
        }
        r
    };
    for p in [&A100, &RTX_3090].into_iter().chain(PARTS) {
        for m in [1, 16, 128] {
            for (n, depth) in [(18432, 7168), (2048, 7168), (1024, 4096)] {
                for (arm, brm) in [(true, true), (false, true), (true, false)] {
                    for elem in [Ty::f32(), Ty::f64()] {
                        let s = site(m, n, depth, arm, brm, elem);
                        let lanes = GPU_VEC_BYTES / elem_bytes_of(&s.elem);
                        for base in arms() {
                            for names in [
                                &[][..],
                                &["fill-rows"],
                                &["fill-rows", "wave-gate"],
                                &["grid-share"],
                                &["panel-regs"],
                            ] {
                                let r = force(geometry(base, names));
                                let Some(g) = launch(&s, p, false, r, None).tiled else {
                                    continue;
                                };
                                for (i, (map, loads)) in
                                    lane_maps(&s, &g, g.kmajor, false).into_iter().enumerate()
                                {
                                    let vec = if i == 0 {
                                        vector_a(&s) && (g.stages < 2 || g.kmajor)
                                    } else {
                                        vector_b(&s)
                                    };
                                    let width = if vec { lanes } else { 1 };
                                    let mut seen = BTreeSet::new();
                                    for copy in 0..loads {
                                        for tid in 0..g.threads() {
                                            let li = tid + copy * g.threads();
                                            let (row, col) = map.coords(&mut Num, &li);
                                            for v in 0..width {
                                                assert!(seen.insert((row, col + v)), "{g:?}");
                                            }
                                        }
                                    }
                                    let (rows, cols) = if i == 0 {
                                        (g.bm(), g.bk)
                                    } else {
                                        (g.bk, g.bn())
                                    };
                                    assert_eq!(seen.len() as u64, rows * cols, "{g:?}");
                                    assert!(seen.iter().all(|&(r, c)| r < rows && c < cols));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The rung settings whose geometry differs: shipped, register staging,
/// and each with the whole-file tile off.
pub(super) fn arms() -> Vec<GpuRungs> {
    let on = GpuRungs::ALL;
    let sync = GpuRungs {
        cp_async: false,
        ..on
    };
    [on, sync]
        .into_iter()
        .flat_map(|r| {
            [
                r,
                GpuRungs {
                    wide_fma: false,
                    ..r
                },
            ]
        })
        .collect()
}

/// S54 `raster` at the kernels the box times, on the tile each face
/// actually takes. The derivation (every rung on, `arch-rungs` off): the
/// 3090 groups at 8192 only (one block's panels, 8 MB, past its 6 MB L2),
/// every other part keeps row-major; without the gate every part groups
/// where the grid is more than a wave. What ships is the rules' tile with
/// raster on every part (the 3090's arms measured null): only the part
/// whose L2 the panels leave groups; the forced-ungated arm still
/// exercises grouping.
#[test]
fn raster_fires_where_one_blocks_panels_leave_the_l2() {
    use crate::profile::{A100, RTX_3090};
    let parts: [&TargetProfile; 6] = [&RTX_3090, &A100, &RTX_4090, &H100_PCIE, &H100, &RTX_5090];
    let faces = [(false, None), (true, None), (true, Some(Reduced::Tf32))];
    let arms = |gate_off: bool| {
        let mut off = GpuRungs::default();
        off.arch = true;
        off.l2_gate = gate_off;
        crate::GpuArms {
            on: GpuRungs::default(),
            off,
            splitk: 0,
            cands: false,
            free: false,
            ..crate::GpuArms::default()
        }
    };
    for p in parts {
        for n in [1024, 2048, 4096, 8192] {
            for (contract, reduced) in faces {
                let s = site(n, n, n, true, true, Ty::f32());
                let at = |a| launch(&s, p, contract, a, reduced);
                let why = format!("{} {n} {contract} {reduced:?}", p.name);
                let derived = at(arms(false));
                // gapfix tensor-tile (C11): beyond the one-block gate
                // (the 3090 at 8192), only the H100 PCIe's 8192 TF32
                // square — its `wgmma` pipe against 2 TB/s re-reads `b`
                // past its keep in 4 of 5 waves: DRAM-bound — groups.
                let reuse = p.name == "h100-pcie" && n == 8192 && reduced.is_some();
                assert_eq!(
                    derived.raster > 1,
                    (p.name == "rtx-3090" && n == 8192) || reuse,
                    "{why}"
                );
                // The tile grid's height (a persistent walk launches
                // `ctas × 1` and walks the tiles in this order).
                let gy = |l: &Launch| match l.wgmma {
                    Some(w) if w.ctas > 0 => s.rows / w.bm(),
                    _ => l.gy,
                };
                let ungated = at(arms(true));
                if n == 8192 {
                    assert!(ungated.raster > 1, "{why}");
                }
                assert!(gy(&ungated) % ungated.raster == 0, "{why}");
                let shipped = at(crate::GpuArms::default());
                if shipped.raster > 1 {
                    assert!((p.name == "rtx-3090" && n == 8192) || reuse, "{why}");
                }
                assert!(gy(&shipped) % shipped.raster == 0, "{why}");
            }
        }
    }
}
