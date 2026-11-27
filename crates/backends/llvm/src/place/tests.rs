//! Unit tests of the machine-model placement: profile translation, search, lowered launches.

use super::machine::prim_of;
use super::search::{PICKS, PROBE, search_site};
use super::*;
use crate::{EmitOpts, emit_detailed};
use mapal_place::space;

/// Every GPU profile (the hand-written parts; `native-gpu` is this box's
/// RTX 5080, `rtx-5080` here).
const PROFILES: [&str; 8] = [
    "cuda-ada",
    "rtx-4090",
    "h100",
    "h100-pcie",
    "rtx-5090",
    "rtx-5080",
    "rtx-3090",
    "a100",
];

#[test]
fn every_gpu_profile_translates_with_provenance() {
    for name in PROFILES {
        let p = crate::profile::resolve(name).expect("a profile");
        let m = machine_of(p).expect("a GPU machine");
        let rows = m.rows();
        assert!(m.units() > 0 && m.group_lanes() == 32, "{name}");
        // Every fit stays out of the machine: perturbing a fitted
        // DeviceCost field leaves the translation unchanged.
        let g = *p.gpu().unwrap();
        for f in FITS {
            let mut over: crate::profile::CostOver = [None; DeviceCost::NAMES.len()];
            let i = DeviceCost::NAMES.iter().position(|n| *n == f).unwrap();
            over[i] = Some(g.rate(f) * 3 + 7);
            let q = TargetProfile {
                machine: crate::profile::Machine::Gpu(g.with_cost(&over)),
                ..*p
            };
            assert_eq!(
                machine_of(&q).unwrap(),
                m,
                "{name}: the machine reads the fit {f}"
            );
        }
        // Unlabeled values are visible as nominal, never silent.
        let marked = rows.iter().filter(|r| r.src.marked()).count();
        eprintln!(
            "{name}: {} facts, {} marked (derived / nominal)",
            rows.len(),
            marked
        );
    }
    // The RTX 5080's keep is measured; the A100's and H100's half-L2
    // placeholders are nominal with the interval down to a quarter.
    let keep = |n: &str| {
        let m = machine_of(crate::profile::resolve(n).unwrap()).unwrap();
        m.places[L2].cache().unwrap().keep.cyclic
    };
    assert!(!keep("rtx-5080").marked());
    assert!(keep("a100").marked() && keep("a100").lo == 10 << 20);
    assert!(keep("h100").marked());
    // A part without a measured launch cost prices it as nominal, never
    // as the RTX 5080's measured one.
    let m = machine_of(crate::profile::resolve("rtx-5090").unwrap()).unwrap();
    assert!(m.dispatch.launches[0].ns.marked());
}

/// **Every measured row is read** (cost-rental review): on every part,
/// each `prim` row maps to a machine primitive or operation, and each
/// efficiency and latency row to an operation — none dropped by its
/// name (the H100's `wg.m64n128k16.bf16` rows measure the warpgroup
/// family `wg.m64n256k16.bf16` at `n = 128`).
#[test]
fn every_measured_row_is_read() {
    // Measured and read by no machine row: the one-block launch (a
    // grid's launch reads the `.sms` rows).
    const UNREAD: [&str; 2] = ["launch.issue.1", "launch.gap.1"];
    for name in [
        "rtx-3090", "a100", "rtx-4090", "h100", "rtx-5090", "rtx-5080",
    ] {
        let p = crate::profile::resolve(name).unwrap();
        let g = p.gpu().unwrap();
        let m = machine_of(p).unwrap();
        for &(row, _) in g.place.prim {
            if UNREAD.contains(&row) {
                continue;
            }
            let (of, _) = prim_of(row, &m.ops)
                .unwrap_or_else(|| panic!("{name}: the prim row {row} maps to no machine row"));
            match of {
                PrimOf::Op(o) => {
                    let op = &m.ops[o];
                    assert!(
                        !op.rate.marked() && !op.latency.marked(),
                        "{name}: {row} sets {}'s rate and latency",
                        op.name
                    );
                }
                _ => assert!(
                    m.prims.iter().any(|p| p.name == row),
                    "{name}: the prim row {row} is a machine primitive"
                ),
            }
        }
        for &(row, ..) in g.place.eff {
            assert!(
                m.ops
                    .iter()
                    .any(|o| measures(row, o.name, o.n) && !o.eff.is_empty()),
                "{name}: the efficiency row {row} is an operation's"
            );
        }
        for &(row, _) in g.place.latency {
            assert!(
                m.ops
                    .iter()
                    .any(|o| measures(row, o.name, o.n) && !o.latency.marked()),
                "{name}: the latency row {row} is an operation's"
            );
        }
    }
    // The H100's warpgroup bf16 family reads its measured member; the
    // register-`a` form, never measured, stays nominal.
    let m = machine_of(crate::profile::resolve("h100").unwrap()).unwrap();
    let op = |n: &str| m.ops.iter().find(|o| o.name == n).expect("the op");
    let wg = op("wg.m64n256k16.bf16");
    assert!(!wg.rate.marked() && !wg.latency.marked() && !wg.eff.is_empty());
    assert!(op("wg.m64n256k16.bf16.rs").latency.marked());
    assert!(!measures("wg.m64n128k16.bf16", "wg.m64n256k16.f16", wg.n));
    // A value from another box of the part keeps its own provenance
    // (the RTX 4090's L2 latency: an earlier box, not the rental run's).
    let m = machine_of(crate::profile::resolve("rtx-4090").unwrap()).unwrap();
    let lat = m
        .reach(L2, mapal_place::ExecKind::Unit)
        .unwrap()
        .latency_ns
        .src;
    assert!(
        matches!(lat, Src::Measured(c) if c.contains("earlier RTX 4090 box")),
        "the 4090's L2 latency cites its own box: {lat:?}"
    );
    assert!(!measures(
        "mma.m16n8k16.bf16",
        "mma.m16n16k16.bf16",
        NSet::One(16)
    ));
}

/// **The host launches the kernel the device emits** (cost-rental): on
/// every GPU part, each placed ordered site of the attention programs
/// launches the geometry its kernel is emitted with — the A100's
/// 1024 x 256 x 256 site launched the pick's 32 x 32 grid over the rules'
/// 64 x 32 kernel (the rules' tile stores column-major, the pick's not).
#[test]
fn placed_host_launch_is_the_device_kernel() {
    let mut arms = crate::GpuArms {
        free: true,
        site_cands: 1,
        ..crate::GpuArms::default()
    };
    let plain = arms;
    *arms.on.rung("machine-place").expect("the rung") = true;
    let (mut ordered, mut kept) = (0, 0);
    for prog in [
        "pt_decode_1024_256",
        "pt_causal_1024_256",
        "pt_mlp_2048",
        "pt_twin_2048",
    ] {
        let src = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../../benches/algorithms/{prog}.mapal")),
        )
        .expect("the program");
        let po = mapal_syntax::parse(&src);
        let ir = mapal_rewrite::rewrite(mapal_lower::lower(&src, &po.program).expect("lowers")).ir;
        for name in [
            "rtx-3090", "a100", "rtx-4090", "h100", "rtx-5090", "rtx-5080",
        ] {
            let p = crate::profile::resolve(name).unwrap();
            for (site, _) in &warm_sites(&ir, p.gpu().unwrap().l2_keep_bytes()) {
                let host = gpu::launch_in(site, p, true, arms, crate::Order::Free);
                // A site the search kept runs the OFF arm's kernel, of
                // any realization (the H100's kept 2048² walk).
                let rules =
                    self::rules(|| gpu::launch_in_rules(site, p, true, arms, crate::Order::Free));
                if search_site(site, p, true, arms, crate::Order::Free, &rules)
                    .is_some_and(|x| x.kept)
                {
                    kept += 1;
                    let mut dev = gpu::launch_for(site, p, true, arms, host.combine);
                    let off = gpu::launch_for(site, p, true, plain, host.combine);
                    dev.rungs.machine_place = false;
                    assert_eq!(
                        dev, off,
                        "{prog} on {name}: a kept site's kernel is the OFF arm's"
                    );
                }
                if host.combine != Combine::None {
                    continue;
                }
                ordered += 1;
                let dev = gpu::launch_for(site, p, true, arms, Combine::None);
                assert_eq!(
                    (host.tiled, host.mma, host.wgmma),
                    (dev.tiled, dev.mma, dev.wgmma),
                    "{prog} on {name}: the launched geometry is the kernel's"
                );
            }
        }
    }
    assert!(ordered > 0 && kept > 0, "ordered and kept sites placed");
}

/// **A fixed-order site launches the kernel the device emits**
/// (cost-rental, `OrderScope`): on every part, fixed-order sites whose
/// pick moves the launch off the rules' (a split's `S`, its tile) — the
/// kernel of the host's realization emitted before the host's search —
/// launch the device's whole kernel (grid, tile, rungs), never only its
/// tile (the H100's 16 x 2048 x 7168 split pick launched over the rules'
/// kernel: wrong values). Without `OrderScope` the kernel is the pick of
/// the arms' own order (`Sequential`: no split) and this fails.
#[test]
fn placed_fixed_order_kernel_is_the_host_launch() {
    let mut fixed = crate::GpuArms {
        free: false,
        site_cands: 1,
        ..crate::GpuArms::default()
    };
    *fixed.on.rung("machine-place").expect("the rung") = true;
    let mut moved = 0;
    for (m, n, k) in [
        (16, 2048, 7168),
        (16, 18432, 7168),
        (128, 4096, 4096),
        (128, 6144, 4096),
    ] {
        let site = crate::place_replay::replay_site(m, n, k, "f32", true);
        for name in [
            "rtx-3090", "a100", "rtx-4090", "h100", "rtx-5090", "rtx-5080",
        ] {
            let p = crate::profile::resolve(name).unwrap();
            let at = format!("{m} x {n} x {k} on {name}");
            let _order = OrderScope::new(crate::Order::Fixed);
            // The host's launch, searched alone.
            PICKS.with(|c| c.borrow_mut().clear());
            let host = gpu::launch_in(&site, p, false, fixed, crate::Order::Fixed);
            let rules =
                self::rules(|| gpu::launch_in_rules(&site, p, false, fixed, crate::Order::Fixed));
            moved += usize::from(host != rules);
            // The kernel of its realization emitted first, then the
            // host's launch.
            PICKS.with(|c| c.borrow_mut().clear());
            let dev = gpu::launch_for(&site, p, false, fixed, host.combine);
            let again = gpu::launch_in(&site, p, false, fixed, crate::Order::Fixed);
            assert_eq!(again, host, "{at}: one launch whichever comes first");
            // The device's launch is its realization's kernel (its
            // `combine` field the host's to set).
            assert_eq!(
                Launch {
                    combine: host.combine,
                    ..dev
                },
                host,
                "{at}: the host launches the kernel the device emits"
            );
        }
    }
    assert!(moved > 0, "the search moves some fixed-order launch");
}

/// **The warpgroup lowering's declared constraint** (cost-rental): a
/// K-major-only instruction (TF32) takes the transposed product.
#[test]
fn tf32_wgmma_direct_product_refused() {
    // cost-rental: on the H100 the search's direct TF32 warpgroup tile
    // was invalid PTX (`wgmma.mma_async` arguments mismatch): a
    // K-major-only instruction reads a lanes-major `b` only through the
    // transposed product. The rules' tile stays admitted; the same tile
    // made direct is refused.
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../benches/algorithms/pt_mlp_2048_tf32.mapal"),
    )
    .expect("the TF32 program");
    let po = mapal_syntax::parse(&src);
    let ir = mapal_rewrite::rewrite(mapal_lower::lower(&src, &po.program).expect("lowers")).ir;
    let p = crate::profile::resolve("h100").unwrap();
    let m = machine_of(p).unwrap();
    let arms = crate::GpuArms {
        free: true,
        site_cands: 1,
        ..crate::GpuArms::default()
    };
    let sites = warm_sites(&ir, p.gpu().unwrap().l2_keep_bytes());
    let mut seen = 0;
    for (site, _) in &sites {
        let rules = gpu::launch_in_rules(site, p, true, arms, crate::Order::Free);
        let Some(w) = rules.wgmma else { continue };
        if w.fmt != crate::profile::MmaFmt::Tf32 {
            continue;
        }
        seen += 1;
        assert!(!w.direct, "the rules' TF32 tile is the transposed product");
        let e = Emitter {
            id: if w.ws { WGMMA_WS } else { WGMMA },
            site,
            profile: p,
            contract: true,
            arms,
        };
        let s = site_of(site, true, crate::Order::Free);
        let at = |w| {
            let x = schedule_of(
                site,
                p,
                arms,
                &Launch {
                    wgmma: Some(w),
                    ..rules
                },
            );
            e.declare(&m, &s, &x)
        };
        assert!(at(w).is_ok(), "the rules' tile admitted");
        assert!(
            at(crate::profile::GpuWgmmaTile { direct: true, ..w }).is_err(),
            "a direct TF32 tile refused"
        );
    }
    assert!(
        seen > 0,
        "the program has a TF32 warpgroup site on the H100"
    );
}

/// **The vector-cells kernel's declared constraint** (cost-model ABBA):
/// on the 2048 x 2048 x 25 window program's own tile sites (mixed-radix
/// `b`), the one-thread-per-cell lowering refuses more than one cell a
/// thread — the search's 16-cell pick stored wrong values on the RTX
/// 5080 — and keeps admitting them on a plain GEMV row.
#[test]
fn naive_vector_cells_refuse_a_window_site() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../benches/algorithms/pt_conv_2048_5.mapal"),
    )
    .expect("the window program");
    let po = mapal_syntax::parse(&src);
    let ir = mapal_rewrite::rewrite(mapal_lower::lower(&src, &po.program).expect("lowers")).ir;
    let p = crate::profile::resolve("rtx-5080").unwrap();
    let m = machine_of(p).unwrap();
    let arms = crate::GpuArms {
        free: true,
        site_cands: 1,
        ..crate::GpuArms::default()
    };
    let sites = warm_sites(&ir, p.gpu().unwrap().l2_keep_bytes());
    let mut windows = 0;
    for (site, _) in &sites {
        let rules = gpu::launch_in_rules(site, p, true, arms, crate::Order::Free);
        if rules.tiled.is_some() || rules.mma.is_some() || rules.wgmma.is_some() {
            continue;
        }
        let e = Emitter {
            id: NAIVE,
            site,
            profile: p,
            contract: true,
            arms,
        };
        let s = site_of(site, true, crate::Order::Free);
        let at = |cells| {
            let x = schedule_of(site, p, arms, &Launch { cells, ..rules });
            e.declare(&m, &s, &x)
        };
        assert!(at(1).is_ok(), "one cell a thread stays admitted");
        if !gpu::naive_cells_admit(site, 16) {
            windows += 1;
            assert!(at(16).is_err(), "16 cells on a window site refused");
        }
    }
    assert!(windows > 0, "the program has a window site");
    let row = crate::place_replay::replay_site(1, 4096, 2048, "f32", true);
    assert!(
        gpu::naive_cells_admit(&row, 4),
        "a GEMV row admits vector cells"
    );
}

/// **ADR-0050 review: the one-thread-per-cell kernel's declared
/// registers cover what `ptxas` allocates** at every cell count the
/// search may give a thread (1–16, f32 and bf16 `b`), so the footprint's
/// occupancy is never above the kernel's. `PLACE_NAIVE_REGS=1` prints
/// the table; without `llc` / `ptxas`, nothing to check.
#[test]
fn naive_declared_registers_cover_ptxas() {
    use std::process::Command;
    let have = |t: &str| Command::new(t).arg("--version").output().is_ok();
    if !have("llc") || !have("ptxas") {
        return;
    }
    let arms = crate::GpuArms {
        free: true,
        site_cands: 1,
        ..crate::GpuArms::default()
    };
    let mut over = Vec::new();
    for part in ["a100", "rtx-3090", "rtx-4090", "h100", "rtx-5080"] {
        let p = crate::profile::resolve(part).unwrap();
        let g = p.gpu().unwrap();
        let m = machine_of(p).unwrap();
        for fmt in ["f32", "bf16"] {
            let site = crate::place_replay::replay_site(1, 4096, 2048, fmt, true);
            let rules = gpu::launch_in_rules(&site, p, true, arms, crate::Order::Free);
            assert!(
                rules.tiled.is_none() && rules.mma.is_none(),
                "a cell kernel"
            );
            for cells in [1, 2, 4, 8, 16] {
                let l = Launch { cells, ..rules };
                let x = schedule_of(&site, p, arms, &l);
                let e = Emitter {
                    id: NAIVE,
                    site: &site,
                    profile: p,
                    contract: true,
                    arms,
                };
                let s = site_of(&site, true, crate::Order::Free);
                let d = e.declare(&m, &s, &x).expect("declared");
                let text = gpu::lowered_text(&site, p, true, arms, &l).expect("lowers");
                let dir = std::env::temp_dir().join(format!(
                    "mapal-naive-regs-{}-{fmt}-{cells}",
                    std::process::id()
                ));
                std::fs::create_dir_all(&dir).unwrap();
                let (ll, ptx) = (dir.join("k.ll"), dir.join("k.ptx"));
                std::fs::write(&ll, gpu::kernel_module(&text)).unwrap();
                let mut llc = Command::new("llc");
                llc.args(["-O3", &format!("-mcpu={}", g.arch), "-nvptx-fma-level=0"]);
                if let Some(v) = crate::Ptx::declared(&gpu::kernel_module(&text)) {
                    llc.arg(format!("-mattr={}", v.mattr()));
                }
                assert!(llc.arg(&ll).arg("-o").arg(&ptx).status().unwrap().success());
                let out = Command::new("ptxas")
                    .args([&format!("-arch={}", g.arch), "-v"])
                    .arg(&ptx)
                    .arg("-o")
                    .arg(dir.join("k.cubin"))
                    .output()
                    .unwrap();
                let _ = std::fs::remove_dir_all(&dir);
                let log = String::from_utf8_lossy(&out.stderr).to_string();
                let regs: u64 = log
                    .split("Used ")
                    .nth(1)
                    .and_then(|r| r.split(' ').next())
                    .and_then(|r| r.parse().ok())
                    .expect("a register count");
                if std::env::var("PLACE_NAIVE_REGS").is_ok() {
                    eprintln!(
                        "naive {part} {fmt} cells={cells}: ptxas {regs}, declared {}",
                        d.regs
                    );
                }
                if regs > d.regs {
                    over.push(format!(
                        "{part} {fmt} cells={cells}: ptxas {regs}, declared {}",
                        d.regs
                    ));
                }
            }
        }
    }
    assert!(over.is_empty(), "registers above the declared: {over:?}");
}

/// **ADR-0050 review: the native GPU's machine is the hand profile's** —
/// the device query and the calibration line (its `place.` facts
/// included) translate to the RTX 5080's machine: every fact's value and
/// interval, and whether it is marked (what the minimax reads), equal;
/// only the cites' wording differs. Without a calibration line the start
/// timing and the clock fall back, marked.
#[test]
fn native_gpu_translates_to_the_hand_profiles_machine() {
    let q = crate::profile::DeviceQuery {
        cc: 120,
        sms: 84,
        warp: 32,
        max_threads_per_block: 1024,
        max_threads_per_sm: 1536,
        max_blocks_per_sm: 24,
        regs_per_sm: 65536,
        smem_per_sm: 102400,
        smem_per_block: 101_376,
        smem_reserved_per_block: 1024,
        l2_bytes: 67108864,
        clock_khz: 2_640_000,
        mem_clock_khz: 15_001_000,
        mem_bus_bits: 256,
    };
    let calib = crate::profile::COMMITTED_CALIB[0];
    let native = q.profile(Some(calib));
    let hand = crate::profile::resolve("rtx-5080").unwrap();
    let (a, b) = (machine_of(&native).unwrap(), machine_of(hand).unwrap());
    let rows = |m: &Machine| {
        m.rows()
            .into_iter()
            .map(|r| (r.name, r.v, r.lo, r.hi, r.src.marked()))
            .collect::<Vec<_>>()
    };
    let (ra, rb) = (rows(&a), rows(&b));
    assert_eq!(ra.len(), rb.len());
    for (x, y) in ra.iter().zip(&rb) {
        assert_eq!(x, y, "native-gpu against rtx-5080");
    }
    assert_eq!(a.dispatch.start, b.dispatch.start);
    assert_eq!(a.dispatch.start_src.marked(), b.dispatch.start_src.marked());
    // No calibration line: the probes' facts are absent, and visibly so
    // (marked), never silently the hand profile's.
    let none = q.profile(None);
    let c = machine_of(&none).unwrap();
    assert!(c.dispatch.start_src.marked() && c.dispatch.clock_khz.marked());
}

/// **ADR-0050 C4: the search on every GPU profile** — f32, TF32 and
/// bf16 GEMM sites of several shapes: a pick on each (or the rules' kept
/// within the band), every candidate it lowers through the gate, its
/// `launch_of` a launch the emitter's text takes; the walk's stats
/// printed (`PLACE_SEARCH=1`).
#[test]
fn the_search_places_gemm_sites_on_every_profile() {
    let shapes: &[(u64, u64, u64, &str)] = &[
        (128, 4096, 4096, "f32"),
        (1024, 1024, 1024, "tf32"),
        (512, 2048, 2048, "bf16"),
        (2048, 4096, 4096, "bf16"),
    ];
    // A debug build places two parts' two shapes; `--release` all.
    let debug = cfg!(debug_assertions);
    for name in PROFILES
        .into_iter()
        .filter(|n| !debug || ["rtx-5080", "h100"].contains(n))
    {
        let p = crate::profile::resolve(name).unwrap();
        for &(m, n, k, fmt) in shapes.iter().filter(|s| !debug || s.0 <= 512) {
            let site = crate::place_replay::replay_site(m, n, k, fmt, false);
            let arms = crate::GpuArms {
                free: true,
                site_cands: 1,
                ..crate::GpuArms::default()
            };
            let rules = gpu::launch_in_rules(&site, p, true, arms, crate::Order::Free);
            let t = std::time::Instant::now();
            let placed =
                search_site(&site, p, true, arms, crate::Order::Free, &rules).expect("a placement");
            let dt = t.elapsed();
            let text = gpu::lowered_text(&site, p, true, arms, &placed.launch);
            assert!(text.is_ok(), "{name} {m}x{n}x{k} {fmt}: {}", placed.plan);
            if std::env::var("PLACE_SEARCH").is_ok() {
                eprintln!(
                    "{name} {m}x{n}x{k} {fmt} {dt:.2?} kept={} {}",
                    placed.kept,
                    placed.line.trim_end()
                );
            }
        }
    }
}

/// The search's walk per lowering on one site (`PLACE_SEARCH=1`).
#[test]
fn search_cost_per_lowering() {
    if std::env::var("PLACE_SEARCH").is_err() {
        return;
    }
    let p = crate::profile::resolve("rtx-5080").unwrap();
    for (m, n, k, fmt) in [(128, 4096, 4096, "f32"), (512, 2048, 2048, "bf16")] {
        let site = crate::place_replay::replay_site(m, n, k, fmt, false);
        let arms = crate::GpuArms {
            free: true,
            site_cands: 1,
            ..crate::GpuArms::default()
        };
        let mach = machine_of(p).unwrap();
        let s = site_of(&site, true, crate::Order::Free);
        let rules = gpu::launch_in_rules(&site, p, true, arms, crate::Order::Free);
        let x = schedule_of(&site, p, arms, &rules);
        for e in Emitter::all(&site, p, true, arms) {
            let t = std::time::Instant::now();
            let inc = (e.id == lowering_of(&rules)).then(|| (0, x.clone()));
            SEARCHING.with(|c| c.set(true));
            LSU.with(|c| c.borrow_mut().clear());
            LAYOUT.with(|c| c.borrow_mut().clear());
            let r = mapal_place::search(&mach, &s, &[&e], &place_opts(), inc, &mut |_, _| Ok(()));
            SEARCHING.with(|c| c.set(false));
            eprintln!(
                "{m}x{n}x{k} {fmt} {}: {:.2?} {:?} {}",
                e.id,
                t.elapsed(),
                r.as_ref().map(|r| r.stats.clone()),
                r.map_or("-".into(), |r| format!(
                    "{} {:.0}",
                    r.pick.x.plan(),
                    r.pick.value()
                ))
            );
        }
    }
}

/// The suite's programs: every committed algorithm and ladder program.
fn suite() -> Vec<(String, String)> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../benches/algorithms");
    let mut out = Vec::new();
    for dir in [root.clone(), root.join("ladder")] {
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .expect("benches/algorithms")
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "mapal"))
            .collect();
        files.sort();
        for f in files {
            let src = std::fs::read_to_string(&f).expect("a program");
            out.push((f.file_stem().unwrap().to_string_lossy().into_owned(), src));
        }
    }
    out
}

fn lower(src: &str) -> Option<mapal_ir::CategoryIr> {
    let po = mapal_syntax::parse(src);
    if !po.diagnostics.is_empty() {
        return None;
    }
    let ir = mapal_lower::lower(src, &po.program).ok()?;
    Some(mapal_rewrite::rewrite(ir).ir)
}

/// The faces the suite emits a program on: the default face, and the
/// matrix-product cells on the four others (`gpu_bundle.sh`).
fn faces(stem: &str) -> Vec<Vec<&'static str>> {
    let mut f = vec![vec![]];
    if stem.starts_with("mm_") {
        f.extend([
            vec!["--exact"],
            vec!["--order=sequential"],
            vec!["--no-contract", "--order=fixed"],
            vec!["--order=fixed"],
        ]);
    }
    f
}

/// The order a face's flags emit under (`flags::apply`: `--exact` and
/// `--order=sequential` sequential, `--order=fixed` fixed, else free).
fn face_order(face: &[&str]) -> crate::Order {
    let mut opts = EmitOpts::default();
    for f in face {
        assert!(crate::flags::apply(&mut opts, f).expect("a flag"), "{f}");
    }
    opts.order
}

fn emit(
    ir: &mapal_ir::CategoryIr,
    target: &'static str,
    face: &[&str],
    probe: Option<bool>,
) -> (Option<String>, Vec<Seen>) {
    let mut opts = EmitOpts {
        host: "generic",
        target,
        ..EmitOpts::default()
    };
    for f in face {
        assert!(crate::flags::apply(&mut opts, f).expect("a flag"), "{f}");
    }
    PROBE.with(|p| {
        *p.borrow_mut() = probe.map(|roundtrip| Probe {
            roundtrip,
            seen: Vec::new(),
        })
    });
    let text = emit_detailed(ir, &opts)
        .ok()
        .map(|e| e.text + "\n;;device\n" + &e.device.unwrap_or_default());
    let seen = PROBE.with(|p| p.borrow_mut().take().map(|p| p.seen).unwrap_or_default());
    (text, seen)
}

/// **ADR-0050 C2 coverage: every launch the rules make for a tile site
/// of the suite, on every GPU profile, is a point of the candidate space
/// and lowers back to itself** — `schedule_of` the launch is inside its
/// lowering's coordinate domains with every declared constraint met and
/// an occupancy within its footprint's, its traversal is one the
/// traversal generator gives (persistent grids aside: out of step 1's
/// space), `launch_of` that schedule is the launch, and the emission
/// with every launch replaced by its schedule's is byte-identical.
#[test]
fn rules_launches_are_points_of_the_space() {
    // A debug build emits every eighth program (and every cell); a
    // release build (`--release`) the whole suite.
    let progs: Vec<_> = suite()
        .into_iter()
        .enumerate()
        .filter(|(i, (stem, _))| !cfg!(debug_assertions) || i % 8 == 0 || stem.starts_with("mm_"))
        .map(|(_, p)| p)
        .collect();
    let jobs = std::env::var("MAPAL_TESTGEN_JOBS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12usize);
    let chunks: Vec<Vec<(String, String)>> = {
        let mut c = vec![Vec::new(); jobs];
        for (i, p) in progs.into_iter().enumerate() {
            c[i % jobs].push(p);
        }
        c
    };
    type Report = (
        u64,
        u64,
        Vec<String>,
        std::collections::BTreeMap<String, u64>,
        Vec<String>,
    );
    let results: Vec<Report> = std::thread::scope(|sc| {
        let hs: Vec<_> = chunks
            .iter()
            .map(|chunk| {
                sc.spawn(move || {
                    let mut launches = 0u64;
                    let mut emissions = 0u64;
                    let mut fails = Vec::new();
                    let mut d6: std::collections::BTreeMap<String, u64> = Default::default();
                    let mut persist = Vec::new();
                    for (stem, src) in chunk {
                        let Some(ir) = lower(src) else { continue };
                        for target in PROFILES {
                            for face in faces(stem) {
                                let (text, seen) = emit(&ir, target, &face, Some(false));
                                let Some(text) = text else { continue };
                                emissions += 1;
                                let mut dedup = std::collections::HashSet::new();
                                for s in &seen {
                                    let key = format!("{:?}{:?}{}", s.site, s.launch, s.contract);
                                    if !dedup.insert(key) {
                                        continue;
                                    }
                                    launches += 1;
                                    let p = crate::profile::resolve(s.profile).unwrap();
                                    let m = machine_of(p).unwrap();
                                    let x = schedule_of(&s.site, p, s.arms, &s.launch);
                                    for (n, v) in &x.params {
                                        if n.starts_with("rung") {
                                            *d6.entry(format!(
                                                "{n}{}",
                                                crate::GpuRungs::NAMES[*v as usize]
                                            ))
                                            .or_default() += 1;
                                        }
                                    }
                                    let e = Emitter {
                                        id: lowering_of(&s.launch),
                                        site: &s.site,
                                        profile: p,
                                        contract: s.contract,
                                        arms: s.arms,
                                    };
                                    let tag = format!(
                                        "{stem}{} {target} {}x{}x{} {}",
                                        face.join(""),
                                        s.site.rows,
                                        s.site.c,
                                        s.site.k,
                                        x.plan()
                                    );
                                    let back = launch_of(&s.site, p, s.contract, s.arms, &x);
                                    if back != s.launch {
                                        fails.push(format!("{tag}: launch_of differs"));
                                        continue;
                                    }
                                    let d = match e.contains(&m, &x) {
                                        Ok(d) => d,
                                        Err(why) => {
                                            fails.push(format!("{tag}: {why}"));
                                            continue;
                                        }
                                    };
                                    let occ = m.occupancy(d.threads, d.regs, &[(SMEM, d.smem)]);
                                    if x.per_unit > occ {
                                        *d6.entry(
                                            "residency above the footprint's (rules' metadata)"
                                                .into(),
                                        )
                                        .or_default() += 1;
                                    }
                                    if let Traversal::Persistent { .. } = x.trav {
                                        persist.push(tag);
                                        continue;
                                    }
                                    // The fold's freedom is the
                                    // emission's face (its order), never
                                    // read off the launch checked.
                                    let order = face_order(&face);
                                    let site = site_of(&s.site, s.contract, order);
                                    let ts = space::traversals(&m, &site, &x, &d, e.traversals());
                                    if !ts.contains(&x.trav) {
                                        fails.push(format!(
                                            "{tag}: traversal outside the generator"
                                        ));
                                    }
                                }
                                // Byte-identical: every launch replaced
                                // by its schedule's.
                                let (rt, _) = emit(&ir, target, &face, Some(true));
                                if rt.as_deref() != Some(text.as_str()) {
                                    fails.push(format!(
                                        "{stem}{} {target}: round-trip text differs",
                                        face.join("")
                                    ));
                                }
                            }
                        }
                    }
                    (launches, emissions, fails, d6, persist)
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("a worker"))
            .collect()
    });
    let (mut launches, mut emissions) = (0, 0);
    let mut fails = Vec::new();
    let mut d6: std::collections::BTreeMap<String, u64> = Default::default();
    let mut persist = Vec::new();
    for (l, e, f, d, p) in results {
        launches += l;
        emissions += e;
        fails.extend(f);
        for (k, v) in d {
            *d6.entry(k).or_default() += v;
        }
        persist.extend(p);
    }
    fails.sort();
    eprintln!(
        "coverage: {emissions} emissions, {launches} distinct rules' launches, {} outside, {} persistent (out of step 1)",
        fails.len(),
        persist.len()
    );
    eprintln!("D6 rungs a launch carries beyond its site's table: {d6:?}");
    // The reasons, by class (digits dropped), with an example each.
    let mut why: std::collections::BTreeMap<String, (u64, String)> = Default::default();
    for f in &fails {
        let r = f.rsplit(": ").next().unwrap_or(f);
        let class: String = r
            .chars()
            .map(|c| if c.is_ascii_digit() { '#' } else { c })
            .collect();
        let e = why.entry(class).or_insert((0, f.clone()));
        e.0 += 1;
    }
    let mut why: Vec<_> = why.into_iter().collect();
    why.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    for (class, (n, ex)) in why.iter().take(40) {
        eprintln!("  {n}\t{class}\n      e.g. {ex}");
    }
    assert!(
        fails.is_empty(),
        "{} launches outside the space",
        fails.len()
    );
}
