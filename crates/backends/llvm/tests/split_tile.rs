//! f32-tensor-cores: `{f32}` products on tensor cores by split forms, and
//! tensor tiles cut along `k`.
//!
//! - `split-tile` (R-F, `func/gpu.rs` `split_operands` / `split_pays`): a
//!   default-face `{f32}` sum of products takes a split form
//!   (`mapal_ir::split`: tf32x3 for two f32 operands, narrow1 for two
//!   operands of one narrow format) where its a-priori coefficient meets
//!   ADR-0040 D2 at the site's `K` (the accumulator's measured guard bits,
//!   `MatrixOp::acc_g`) and the part's matrix rate pays for the form's passes
//!   against its FFMA rate — OFF arm `--gpu-off=split-tile`, ON arm
//!   `--gpu-on=split-tile` (the bound alone);
//! - `mma-combine` (R-T1, `mma_combine` / `mma_pieces`): a tensor tile's `k`
//!   cut into `S` pieces whose partials add into the zeroed output, where the
//!   cut lifts its grid's wave fill — OFF arm `--gpu-off=mma-combine`.
//!
//! Plans on hand-picked and generated sites across the profiled parts;
//! values through the GPU emulator on programs from the generator
//! (`testgen::SplitProg`, `testgen::gemm_prog_strategy`), in every emulator
//! mode, under the measured accumulator (`MAPAL_EMU_MMA_G=2`) and a
//! guard-less one (`=0`, where the fold is long enough for it), against the
//! interpreter within D2 (plus the split's absolute term), with coverage
//! counters.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::CategoryIr;
use mapal_ir::split::{AccModel, SplitForm, SplitKind};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use std::collections::HashMap;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(
        po.diagnostics.is_empty(),
        "parse: {:?}\n{src}",
        po.diagnostics
    );
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}\n{src}"));
    mapal_rewrite::rewrite(ir).ir
}

/// The emission on `target` at `base`'s face with the rungs `off` off and
/// `on` on; every capable site on the device, the rules' plan only.
fn emission(
    ir: &CategoryIr,
    target: &'static str,
    base: EmitOpts,
    off: &[&str],
    on: &[&str],
    splitk: u64,
) -> Emission {
    let mut gpu = GpuArms {
        site_cands: 1,
        splitk,
        ..GpuArms::default()
    };
    for r in ["place", "pdl", "streams", "pinned", "horizontal", "calls"]
        .iter()
        .chain(off)
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    for r in on {
        *gpu.on.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "generic",
        target,
        gpu,
        ..base
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// Each tensor kernel's `; mapal.plan` fields.
fn mma_plans(e: &Emission) -> Vec<HashMap<String, String>> {
    e.device
        .as_deref()
        .unwrap_or_default()
        .lines()
        .filter(|l| l.starts_with("; mapal.plan ") && l.contains(" kind=mma "))
        .map(|l| {
            l.split(' ')
                .filter_map(|f| f.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        })
        .collect()
}

fn num(p: &HashMap<String, String>, k: &str) -> u64 {
    p.get(k)
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("plan field {k}: {p:?}"))
}

fn split_of(e: &Emission) -> Option<String> {
    mma_plans(e).iter().find_map(|p| p.get("split").cloned())
}

fn floats(out: &str) -> Vec<f64> {
    out.split_whitespace()
        .map(|x| x.parse().expect("a float"))
        .collect()
}

const MEASURED: AccModel = AccModel { g: 2 };

/// A `rows × c × k` f32 GEMM under `{f32}`, operands stored as `fmt`.
fn gemm(rows: u64, c: u64, k: u64, fmt: Option<&str>) -> String {
    let p = testgen::SplitProg {
        dims: [0, 0, 0],
        store: match fmt {
            Some("bf16") => 1,
            Some("f16") => 2,
            _ => 0,
        },
        law: 0,
        bt: false,
        post: false,
        seed: [3, 5],
    };
    testgen::split_src_at(&p, rows, c, k).0
}

/// **The rule per part, per face, per `K`.** tf32x3 on a part whose TF32
/// rate pays three passes against its FFMA rate (the A100: 1024 against
/// 3 · 1.3 · 128 an SM a clock) and nowhere else by rule (GeForce 256, the
/// H100's `mma.sync` 532 against 998); every part under the ON arm; never
/// below the form's `K_min` (24 at the measured `g = 2`), nor on the exact
/// or sequential faces, nor under the OFF arm. narrow1 by rule on every part
/// with a 16-bit warp product (512 against 1.3 · 256 on GeForce).
#[test]
fn split_rule_by_rate_bound_and_face() {
    let ir = lower(&gemm(128, 256, 512, None));
    let at = |t: &'static str, base: EmitOpts, off: &[&str], on: &[&str]| {
        split_of(&emission(&ir, t, base, off, on, 0))
    };
    let d = EmitOpts::default;
    assert_eq!(at("a100", d(), &[], &[]).as_deref(), Some("tf32x3"));
    for t in [
        "rtx-5080",
        "rtx-4090",
        "rtx-3090",
        "rtx-5090",
        "h100",
        "h100-pcie",
    ] {
        assert_eq!(at(t, d(), &[], &[]), None, "{t}: the rate rule refuses");
        assert_eq!(
            at(t, d(), &[], &["split-tile"]).as_deref(),
            Some("tf32x3"),
            "{t}: the ON arm"
        );
    }
    assert_eq!(at("a100", d(), &["split-tile"], &[]), None, "OFF arm");
    let seq = EmitOpts {
        order: mapal_backend_llvm::Order::Sequential,
        ..d()
    };
    assert_eq!(at("a100", seq, &[], &["split-tile"]), None, "sequential");
    assert_eq!(
        at("a100", EmitOpts::exact(), &[], &["split-tile"]),
        None,
        "exact"
    );
    // Below K_min the bound refuses even the ON arm.
    let kmin = SplitForm::of(SplitKind::Tf32x3)
        .k_min(1, 8, MEASURED, 4096)
        .expect("k_min");
    assert_eq!(kmin, 24);
    let short = lower(&gemm(128, 256, 16, None));
    let e = emission(&short, "rtx-5080", d(), &[], &["split-tile"], 0);
    assert_eq!(split_of(&e), None, "K 16 < K_min 24");
    let e = emission(
        &lower(&gemm(128, 256, 32, None)),
        "rtx-5080",
        d(),
        &[],
        &["split-tile"],
        0,
    );
    assert_eq!(split_of(&e).as_deref(), Some("tf32x3"), "K 32 > K_min");
    // narrow1: both operands bf16 (or f16) under `{f32}`, by rule (a site
    // whose flops per operand byte reach every part's FFMA balance: the
    // 4090's 81.9 refuses a 64 flop / B one, as `half-fill`).
    for fmt in ["bf16", "f16"] {
        let ir = lower(&gemm(256, 256, 2048, Some(fmt)));
        for t in ["rtx-5080", "rtx-4090", "a100"] {
            let e = emission(&ir, t, d(), &[], &[], 0);
            let p = mma_plans(&e);
            assert_eq!(
                p.first().and_then(|p| p.get("split")).map(String::as_str),
                Some("narrow1"),
                "{t} {fmt}"
            );
            assert!(p[0]["op"].ends_with(fmt), "{p:?}");
        }
        let e = emission(&ir, "rtx-5080", d(), &["split-tile"], &[], 0);
        assert!(mma_plans(&e).is_empty(), "{fmt}: OFF arm");
    }
    // integration: two fp8 operands under `{f32}` are exact products too,
    // but their warp product is m16n8k32 — no split kernel issues it: the
    // FMA tile, by rule and under the ON arm, on parts listing fp8 products.
    for store in [4, 5] {
        let p = testgen::SplitProg {
            dims: [0, 0, 0],
            store,
            law: 0,
            bt: false,
            post: false,
            seed: [3, 5],
        };
        let ir = lower(&testgen::split_src_at(&p, 256, 256, 2048).0);
        for t in ["rtx-5080", "h100"] {
            for on in [&[][..], &["split-tile"][..]] {
                let e = emission(&ir, t, d(), &[], on, 0);
                assert_eq!(split_of(&e), None, "{t} fp8 store {store} {on:?}");
            }
        }
    }
    // f32 × bf16 (bf16x3n: no emitted form) keeps the FMA tile, ON arm too.
    let p = testgen::SplitProg {
        dims: [0, 0, 0],
        store: 3,
        law: 0,
        bt: false,
        post: false,
        seed: [3, 5],
    };
    let ir = lower(&testgen::split_src_at(&p, 128, 256, 512).0);
    let e = emission(&ir, "rtx-5080", d(), &[], &["split-tile"], 0);
    assert!(mma_plans(&e).is_empty());
    // The tf32x3 kernel's instructions: three TF32 products a tile and k8,
    // the zeroed chain, the round-to-nearest add, the guard's vote.
    let e = emission(&ir_tf32x3(), "rtx-5080", d(), &[], &["split-tile"], 0);
    let dev = e.device.as_deref().unwrap_or_default();
    for i in [
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
        "cvt.rna.tf32.f32",
        "vote.sync.any.pred",
        "float 0.0, float 0.0, float 0.0, float 0.0",
        "@llvm.fma.f32",
    ] {
        assert!(dev.contains(i), "{i}");
    }
}

fn ir_tf32x3() -> CategoryIr {
    lower(&gemm(64, 64, 64, None))
}

/// **`mma-combine` on generated TF32 and 16-bit shapes (plans).** A cut
/// tensor tile has 2 ≤ `S` ≤ 8 pieces of at least 8 panels each, and is
/// either short of one wave of its slots or compute-bound; under the OFF arm
/// no tensor tile is cut; a forced count cuts where it divides the panels.
#[test]
fn mma_combine_on_generated_shapes() {
    let mut runner = TestRunner::deterministic();
    let strat = testgen::gemm_prog_strategy();
    let progs: Vec<testgen::GemmProg> = (0..96)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let counts = testgen::par_check(&progs, |z, p| {
        let (mut cut, mut uncut, mut tensor) = (0usize, 0usize, 0usize);
        let (src, (m, n, k)) = testgen::gemm_src(p);
        let src = if z % 2 == 0 {
            src.replace("-> fold {", "-> fold: tf32 | f32 {")
        } else {
            src.replace("-> fold {", "-> fold: bf16 | f32 {")
        };
        let ctx = format!("#{z} {m}x{n}x{k}");
        let ir = lower(&src);
        let on = emission(&ir, "rtx-5080", EmitOpts::default(), &[], &[], 0);
        for t in mma_plans(&on) {
            tensor += 1;
            let s = num(&t, "S");
            if t["combine"] == "atomic" && s > 1 {
                cut += 1;
                let panels = k / num(&t, "bk");
                assert!((2..=8).contains(&s), "{ctx} {t:?}");
                assert!(panels % s == 0 && panels / s >= 8, "{ctx} {t:?}");
                let slots = 84 * num(&t, "blocks_per_sm");
                let blocks = num(&t, "gx") * num(&t, "gy");
                let flops = 2 * m * n * k;
                let compute = flops / (4 * (m * k + k * n + m * n)) >= 118;
                assert!(blocks < slots || compute, "{ctx} {t:?}");
            } else {
                uncut += 1;
            }
        }
        for t in mma_plans(&emission(
            &ir,
            "rtx-5080",
            EmitOpts::default(),
            &["mma-combine"],
            &[],
            0,
        )) {
            assert_eq!(t["combine"], "none", "{ctx} OFF arm {t:?}");
        }
        [cut, uncut, tensor]
    });
    let sum = |i: usize| counts.iter().map(|c| c[i]).sum::<usize>();
    eprintln!(
        "mma-combine plans: {} cut, {} uncut of {} tensor kernels",
        sum(0),
        sum(1),
        sum(2)
    );
    assert!(sum(0) >= 4 && sum(1) >= 1, "both sides reached");
}

const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_GRID", "2")],
    [("MAPAL_EMU_REVERSE", "1"), ("MAPAL_EMU_GRID", "1")],
];

/// **`mma-combine` values.** Generated GEMMs at runnable extents under the
/// TF32 and bf16 member sets, cut into a forced `S` (integer operands in
/// −2..2: exact on the tensor core, so every cut prints the interpreter's
/// bits), in every emulator mode and against the OFF arm.
#[test]
fn mma_combine_values_through_the_emulator() {
    let Some(cc) = emu::clang() else {
        eprintln!("no clang: emulator runs skipped");
        return;
    };
    let mut runner = TestRunner::deterministic();
    let strat = testgen::gemm_prog_strategy();
    let progs: Vec<testgen::GemmProg> = (0..8)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let counts = testgen::par_check(&progs, |z, p| {
        let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
        let (m, n, k) = (
            pick(p.dims[0], &[64, 128]),
            pick(p.dims[1], &[128, 256]),
            pick(p.dims[2], &[256, 512]),
        );
        let set = if z % 2 == 0 { "tf32" } else { "bf16" };
        // The TF32 tile reads `b` along its lanes (`vector_b`): row-major.
        let p = &testgen::GemmProg {
            bt: p.bt && set == "bf16",
            ..p.clone()
        };
        let src = testgen::gemm_src_at(p, m, n, k)
            .replace("-> fold {", &format!("-> fold: {set} | f32 {{"));
        let ctx = format!("#{z} {m}x{n}x{k} {set} bt={}", p.bt);
        let ir = lower(&src);
        let rr = run(&ir, 20_000_000_000);
        assert!(matches!(rr.outcome, Outcome::Done(_)), "{ctx}");
        let want = rr.output;
        let s = if z % 4 < 2 { 2 } else { 4 };
        let on = emission(&ir, "rtx-5080", EmitOpts::default(), &["half-fill"], &[], s);
        let cut = mma_plans(&on)
            .iter()
            .any(|t| t["combine"] == "atomic" && num(t, "S") == s);
        assert!(cut, "{ctx}: the forced cut {s}");
        let dev = |e: &Emission| e.device.clone().unwrap_or_default();
        let built = emu::build_emulated(&cc, &on.text, &dev(&on));
        let mut runs = 0;
        for (env, (got, _)) in MODES.iter().zip(built.run_each(&MODES)) {
            assert_eq!(got, want, "{ctx} {env:?}");
            runs += 1;
        }
        let off = emission(
            &ir,
            "rtx-5080",
            EmitOpts::default(),
            &["half-fill", "mma-combine"],
            &[],
            s,
        );
        assert_eq!(
            emu::run_emulated(&cc, &off.text, &dev(&off)),
            want,
            "{ctx} OFF"
        );
        [1usize, runs]
    });
    let sum = |i: usize| counts.iter().map(|c| c[i]).sum::<usize>();
    eprintln!(
        "mma-combine values: {} programs, {} emulated runs",
        sum(0),
        sum(1)
    );
    assert!(sum(0) == 8);
}

#[derive(Default, Clone, Copy, Debug)]
struct Cover {
    programs: u64,
    tf32x3: u64,
    narrow1: u64,
    refused_kmin: u64,
    refused_rate: u64,
    pieces: u64,
    post: u64,
    bt: u64,
    guard: u64,
    underflow: u64,
    overflow_sel: u64,
    lo_zero: u64,
    cancel: u64,
    fallback: u64,
    guardless: u64,
    narrow1_rule: u64,
}

impl std::ops::AddAssign for Cover {
    fn add_assign(&mut self, o: Cover) {
        self.programs += o.programs;
        self.tf32x3 += o.tf32x3;
        self.narrow1 += o.narrow1;
        self.refused_kmin += o.refused_kmin;
        self.refused_rate += o.refused_rate;
        self.pieces += o.pieces;
        self.post += o.post;
        self.bt += o.bt;
        self.guard += o.guard;
        self.underflow += o.underflow;
        self.overflow_sel += o.overflow_sel;
        self.lo_zero += o.lo_zero;
        self.cancel += o.cancel;
        self.fallback += o.fallback;
        self.guardless += o.guardless;
        self.narrow1_rule += o.narrow1_rule;
    }
}

/// One generated program: the emulated device under the ON arm (the 5080's
/// rate rule refuses tf32x3; narrow1 is the rule's) in every mode, under the
/// measured accumulator and — where `K` meets the form's guard-less `K_min`
/// — a guard-less one, cut into two pieces where the bound still admits
/// it, and under the OFF arm; each against the interpreter: a finite cell
/// within `2·γ_{K+1}(u)·M + abs·Σ(|a| + |b|)` (both sides within D2 of the
/// exact sum; the split's absolute term), a non-finite one its value class.
fn check(cc: &str, p: &testgen::SplitProg) -> Cover {
    let (src, (cells, _m, k)) = testgen::split_src(p);
    let ir = lower(&src);
    let rr = run(&ir, 4_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let want = floats(&rr.output);
    let gamma = mapal_ir::bound_coef_unit(k, None);
    let check_values = |got: &str, what: &str, abs: f64| {
        let g = floats(got);
        assert_eq!(g.len(), want.len(), "{what}\n{src}");
        for i in 0..cells {
            let (y, w, mm, sab) = (g[3 * i], want[3 * i], want[3 * i + 1], want[3 * i + 2]);
            if !w.is_finite() || !mm.is_finite() {
                // The guard: the oracle's value class (an infinity's sign, NaN).
                assert!(
                    y.is_nan() && w.is_nan() || y == w,
                    "{what} cell {i}: {y} vs {w} (M {mm})\n{src}"
                );
                continue;
            }
            let tol = 2.0 * gamma * mm + abs * sab + 2f64.powi(-22) * w.abs();
            assert!(
                (y - w).abs() <= tol,
                "{what} cell {i}: {y} vs {w} (M {mm}, tol {tol})\n{src}"
            );
        }
    };
    let d = EmitOpts::default;
    let on = emission(&ir, "rtx-5080", d(), &[], &["split-tile"], 0);
    let plans = mma_plans(&on);
    let split = plans.iter().find_map(|t| t.get("split").cloned());
    let rule = emission(&ir, "rtx-5080", d(), &[], &[], 0);
    let ruled = split_of(&rule);
    let kind = match split.as_deref() {
        Some("tf32x3") => Some(SplitKind::Tf32x3),
        Some("narrow1") => Some(SplitKind::Narrow1),
        _ => None,
    };
    // The guard sends operands whose parts leave the normal range to the
    // FFMA path: no absolute term.
    let abs = 0.0;
    let dev = |e: &Emission| e.device.clone().unwrap_or_default();
    let mut c = Cover {
        programs: 1,
        ..Cover::default()
    };
    if let Some(kind) = kind {
        let built = emu::build_emulated(cc, &on.text, &dev(&on));
        for mode in MODES {
            let env = [mode[0], mode[1], ("MAPAL_EMU_MMA_G", "2")];
            let (got, _) = built.run(&env);
            check_values(&got, &format!("ON {mode:?} g=2"), abs);
        }
        let kk = if kind == SplitKind::Tf32x3 { 8 } else { 16 };
        let guardless = SplitForm::of(kind)
            .admits(k, 1, kk, AccModel { g: 0 }, 1)
            .is_some();
        if guardless {
            let (got, _) = built.run(&[("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_MMA_G", "0")]);
            check_values(&got, "ON g=0", abs);
            c.guardless = 1;
        }
        // Two pieces, where the bound admits their combine.
        let cut = emission(&ir, "rtx-5080", d(), &[], &["split-tile"], 2);
        if mma_plans(&cut)
            .iter()
            .any(|t| t.contains_key("split") && t["combine"] == "atomic")
        {
            let (got, _) = emu::build_emulated(cc, &cut.text, &dev(&cut))
                .run(&[("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_MMA_G", "2")]);
            check_values(&got, "ON S=2", abs);
            c.pieces = 1;
        }
        match kind {
            SplitKind::Tf32x3 => c.tf32x3 = 1,
            _ => c.narrow1 = 1,
        }
        c.post = u64::from(p.post);
        c.bt = u64::from(p.bt);
        c.guard = u64::from(p.law == 4);
        c.overflow_sel = u64::from(p.law == 5);
        c.underflow = u64::from(p.law == 6);
        c.lo_zero = u64::from(p.law == 3);
        c.cancel = u64::from(p.law == 2);
        c.refused_rate = u64::from(kind == SplitKind::Tf32x3 && ruled.is_none());
        // The 5080's rate rule refuses tf32x3 (its TF32 rate is its FFMA
        // rate); narrow1 by rule where the site's flops per byte or grid
        // fill pays (`split_pays` 3.).
        assert!(
            kind != SplitKind::Tf32x3 || ruled.is_none(),
            "the 5080's rule refuses tf32x3\n{src}"
        );
        c.narrow1_rule = u64::from(ruled.as_deref() == Some("narrow1"));
    } else {
        let below = match p.store {
            0 => k < 24,
            1 | 2 => k < 16,
            _ => false,
        };
        c.refused_kmin = u64::from(below);
        c.fallback = u64::from(!below);
    }
    let off = emission(&ir, "rtx-5080", d(), &["split-tile"], &[], 0);
    assert!(split_of(&off).is_none(), "OFF arm");
    let got = emu::run_emulated(cc, &off.text, &dev(&off));
    check_values(&got, "OFF", 0.0);
    c
}

/// 64 generated programs (`testgen::SplitProg`): every coverage counter
/// non-zero.
#[test]
fn generated_programs_within_d2() {
    let Some(cc) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::SplitProg> = (0..64)
        .map(|_| {
            testgen::split_prog_strategy()
                .new_tree(&mut runner)
                .unwrap()
                .current()
        })
        .collect();
    if let Ok(dir) = std::env::var("MAPAL_SPLIT_DUMP") {
        for (i, p) in progs.iter().enumerate() {
            let _ = std::fs::write(format!("{dir}/split_{i:02}.mapal"), testgen::split_src(p).0);
        }
    }
    let covers = testgen::par_check(&progs, |_, p| check(&cc, p));
    let mut c = Cover::default();
    for k in covers {
        c += k;
    }
    eprintln!("split-tile generated: {c:?}");
    let all = [
        c.programs,
        c.tf32x3,
        c.narrow1,
        c.refused_kmin,
        c.refused_rate,
        c.pieces,
        c.post,
        c.bt,
        c.guard,
        c.underflow,
        c.overflow_sel,
        c.lo_zero,
        c.cancel,
        c.fallback,
        c.guardless,
        c.narrow1_rule,
    ];
    assert!(all.iter().all(|&x| x > 0), "{c:?}");
}
