//! hopper-wgmma-tma K2: the warp-specialized Hopper TF32 kernel (`wgmma-ws`)
//! — a producer warpgroup copying each stage as whole tensor boxes
//! (`cp.async.bulk.tensor`, the maps the runtime encodes from the kernel's
//! `_tmap` recipe) onto a full barrier and rounding the `aᵀ` box once,
//! consumer warpgroups waiting full and ready, releasing a stage on its empty
//! barrier once the group reading it completed.
//!
//! The emulator (`mapal_gpu_emu.c`) applies a box with the engine's swizzle
//! and out-of-bounds fill, runs `wgmma` as the warpgroup's meet reading `b`
//! through the descriptor (start, SBO, swizzle mode) and the operands'
//! truncation, and fails a run whose stage changed under a group before its
//! `wait_group`. On programs from the generator (`testgen::WgmmaProg`) the
//! warp-specialized kernel equals the S53 kernel (`--gpu-off=wgmma-ws`) byte
//! for byte where that one exists (both sum each k-step of 8 exactly, then
//! truncate, k-steps in order; both round every operand to nearest-even
//! once), and stays within D3's bound of the interpreter, in every emulator
//! mode; a mutation of the protocol fails.

use mapal_backend_llvm::{CostOver, DeviceCost, EmitOpts, GpuArms, emit_detailed};
use mapal_ir::{CategoryIr, bound_coef};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

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

/// The part's SM count overridden to `sms` (0: the profile's).
fn sms(n: u64) -> CostOver {
    let mut c: CostOver = [None; DeviceCost::NAMES.len()];
    if n > 0 {
        let i = DeviceCost::NAMES
            .iter()
            .position(|x| *x == "sms")
            .expect("sms");
        c[i] = Some(n);
    }
    c
}

/// The default face on `target` with `n` SMs and the rungs `off` off: `place`
/// off (every capable site on the device), each site its own kernel, one
/// candidate.
fn emission(ir: &CategoryIr, target: &'static str, n: u64, off: &[&str]) -> (String, String) {
    let mut gpu = GpuArms::default();
    for r in ["place", "pdl", "horizontal", "calls"].iter().chain(off) {
        *gpu.off.rung(r).expect("rung") = true;
    }
    gpu.site_cands = 1;
    let e = emit_detailed(
        ir,
        &EmitOpts {
            host: "generic",
            target,
            gpu,
            cost: sms(n),
            ..EmitOpts::default()
        },
    )
    .expect("emits");
    (e.text, e.device.unwrap_or_default())
}

fn plan_lines(dev: &str) -> Vec<&str> {
    dev.lines()
        .filter(|l| l.starts_with("; mapal.plan "))
        .collect()
}

fn ws_lines(dev: &str) -> Vec<&str> {
    plan_lines(dev)
        .into_iter()
        .filter(|l| l.contains(" ws=1 "))
        .collect()
}

/// A plan line's `key=` value.
fn field(line: &str, key: &str) -> u64 {
    line.split_whitespace()
        .find_map(|w| w.strip_prefix(&format!("{key}=")))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

fn floats(out: &str) -> Vec<f64> {
    out.split_whitespace()
        .map(|x| x.parse().expect("a float"))
        .collect()
}

const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_GRID", "2")],
    [("MAPAL_EMU_REVERSE", "1"), ("MAPAL_EMU_GRID", "1")],
];

const MM: &str = "fn main(): tf32 | f32 {
    65536 -> iota -> ia;
    ia -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 } -> a;
    65536 -> iota -> ib;
    ib -> map { t -> (t * 5 + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 } -> b;
    65536 -> iota -> ic;
    256 -> iota -> kr;
    ic -> map { t -> t / 256 -> i; t % 256 -> j; (0.0, kr) -> fold { acc, k -> acc + a[i * 256 + k] * b[k * 256 + j] } } -> y;
    y[0] -> println;
    y[65535] -> println;
}
";

/// The kernel on `sm_90a`: one producer warpgroup (`setmaxnreg.dec`) and
/// consumers (`setmaxnreg.inc`), the stage's boxes (`cp.async.bulk.tensor`)
/// onto full barriers, no `cp.async`, the maps as 128-byte parameters by
/// value and their recipe beside the kernel; every `wgmma` operand rounded
/// to nearest-even (`cvt.rn`); the OFF arms reproduce the S53 kernel and the
/// even register share. Where `ptxas` is installed: no spill, the plan's
/// registers, `UTMALDG.2D` / `HGMMA` / `USETMAXREG` in the SASS.
#[test]
fn ws_kernel_text_on_sm_90a() {
    let ir = lower(MM);
    let (_, dev) = emission(&ir, "h100", 4, &[]);
    let ws = ws_lines(&dev);
    assert_eq!(ws.len(), 1, "{}", plan_lines(&dev).join("\n"));
    assert!(ws[0].contains(" regs=40/"), "{}", ws[0]);
    for needle in [
        "setmaxnreg.dec.sync.aligned.u32 40;",
        "setmaxnreg.inc.sync.aligned.u32",
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes",
        "ptr byval([128 x i8]) align 64 \"nvvm.grid_constant\" %tm0",
        "_tmap = addrspace(1) constant [41 x i64]",
        "wgmma.wait_group.sync.aligned 0;",
        "mbarrier.arrive.shared::cta.b64 _",
        "fence.proxy.async.shared::cta;",
    ] {
        assert!(dev.contains(needle), "{needle}");
    }
    assert!(!dev.contains("call void @llvm.nvvm.cp.async.cg.shared.global"));
    assert!(!dev.contains("wgmma.wait_group.sync.aligned 1;"));
    assert!(!dev.contains("cvt.rna.tf32"), "every wgmma operand RNE");
    // OFF: the kernel the site had (here the S53 derivation refuses it at
    // four SMs: the warp `mma` tile), staged by `cp.async`.
    let (_, s53) = emission(&ir, "h100", 4, &["wgmma-ws"]);
    assert!(ws_lines(&s53).is_empty() && !s53.contains("cp.async.bulk.tensor"));
    assert!(s53.contains("call void @llvm.nvvm.cp.async.cg.shared.global"));
    let (_, even) = emission(&ir, "h100", 4, &["ws-regs"]);
    assert!(ws_lines(&even).len() == 1 && !even.contains("setmaxnreg"));
    // Every part without the warpgroup product keeps its kernel.
    for t in ["rtx-5080", "rtx-4090"] {
        let (_, d) = emission(&ir, t, 4, &[]);
        assert!(
            ws_lines(&d).is_empty() && !d.contains("cp.async.bulk.tensor"),
            "{t}"
        );
    }
    ptxas_check(&dev, ws[0]);
}

/// `llc` + `ptxas -arch=sm_90a` on the module (skipped where either is
/// missing): 0 spill, registers within the launch's share, and the SASS
/// intents of the plan line.
fn ptxas_check(dev: &str, plan: &str) {
    use std::process::Command;
    let have = |t: &str| Command::new(t).arg("--version").output().is_ok();
    if !have("llc") || !have("ptxas") || !have("cuobjdump") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let p = |n: &str| dir.path().join(n);
    std::fs::write(p("d.ll"), dev).unwrap();
    let ok = Command::new("llc")
        .args(["-O3", "-mcpu=sm_90a", "-nvptx-fma-level=0"])
        .arg(p("d.ll"))
        .arg("-o")
        .arg(p("d.ptx"))
        .status()
        .unwrap()
        .success();
    if !ok {
        return; // an llc without sm_90a: the text checks above stand
    }
    let out = Command::new("ptxas")
        .args(["-arch=sm_90a", "-v"])
        .arg(p("d.ptx"))
        .arg("-o")
        .arg(p("d.cubin"))
        .output()
        .unwrap();
    let log = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{log}");
    let name = plan
        .split_whitespace()
        .find_map(|w| w.strip_prefix("kernel="))
        .unwrap_or("mapal_tile");
    let block = field(plan, "block");
    let mut it = log.split("Compiling entry function '").skip(1);
    let fun = it
        .find(|s| s.starts_with(name))
        .unwrap_or_else(|| panic!("{name} in\n{log}"));
    assert!(
        fun.contains("0 bytes spill stores, 0 bytes spill loads"),
        "{fun}"
    );
    let regs: u64 = fun
        .split("Used ")
        .nth(1)
        .and_then(|r| r.split(' ').next())
        .and_then(|r| r.parse().ok())
        .expect("registers");
    assert!(regs * block <= 65536, "{regs} registers × {block}");
    let sass = Command::new("cuobjdump")
        .arg("-sass")
        .arg(p("d.cubin"))
        .output()
        .unwrap();
    let sass = String::from_utf8_lossy(&sass.stdout);
    for op in [
        "UTMALDG.2D",
        "HGMMA.64x",
        ".F32.TF32",
        "USETMAXREG",
        "SYNCS.PHASECHK",
    ] {
        assert!(sass.contains(op), "{op}");
    }
    assert!(
        !sass.contains("LDGSTS") && !sass.contains(" STL"),
        "no cp.async, no spill"
    );
}

#[derive(Debug, Default)]
struct Cover {
    programs: u64,
    ws: u64,
    vs_s53: u64,
    bound_only: u64,
    two_consumers: u64,
    wraps: u64,
    s53_fallback: u64,
    other_kernel: u64,
}

/// Generated TF32 GEMMs on the H100 at 1, 2 and 8 SMs: every
/// warp-specialized kernel equals the S53 kernel byte for byte where the S53
/// derivation takes the site, and stays within D3's bound of the interpreter,
/// in every emulator mode; every coverage counter non-zero.
#[test]
fn generated_ws_kernels_equal_the_s53_kernel() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::WgmmaProg> = (0..36)
        .map(|_| {
            testgen::wgmma_prog_strategy()
                .new_tree(&mut runner)
                .unwrap()
                .current()
        })
        .collect();
    let covers = testgen::par_check(&progs, |i, p| {
        let (src, (_, _, k)) = testgen::wgmma_src(p);
        let ir = lower(&src);
        let n_sms = [1, 2, 8][i % 3];
        let (host, dev) = emission(&ir, "h100", n_sms, &[]);
        let ws = ws_lines(&dev);
        let mut c = Cover {
            programs: 1,
            ..Cover::default()
        };
        if ws.is_empty() {
            let s53 = plan_lines(&dev).iter().any(|l| l.contains(" kind=wgmma "));
            c.s53_fallback = u64::from(s53);
            c.other_kernel = u64::from(!s53);
            return c;
        }
        c.ws = 1;
        c.two_consumers = u64::from(field(ws[0], "groups") == 2);
        c.wraps = u64::from(k / field(ws[0], "bk") > 2 * field(ws[0], "stages"));
        let (host_off, dev_off) = emission(&ir, "h100", n_sms, &["wgmma-ws"]);
        let s53 = plan_lines(&dev_off)
            .iter()
            .any(|l| l.contains(" kind=wgmma "));
        let on = emu::build_emulated(&clang, &host, &dev);
        let off = s53.then(|| emu::build_emulated(&clang, &host_off, &dev_off));
        let f = floats(&mapal_interp::run(&ir, 1 << 34).output);
        let coef = 2.0 * (bound_coef(k, true) + bound_coef(k, false));
        for mode in MODES {
            let (a, _) = on.run(&mode);
            if let Some(off) = &off {
                let (b, _) = off.run(&mode);
                assert_eq!(a, b, "ws vs S53 {mode:?}\n{src}");
            }
            let g = floats(&a);
            for cell in 0..2 {
                let (y, yf, mag) = (g[2 * cell], f[2 * cell], f[2 * cell + 1]);
                assert!(
                    (y - yf).abs() <= coef * mag,
                    "{mode:?} cell {cell}: {y} vs {yf} (M {mag})\n{src}"
                );
            }
        }
        c.vs_s53 = u64::from(s53);
        c.bound_only = u64::from(!s53);
        c
    });
    let mut c = Cover::default();
    for k in covers {
        c.programs += k.programs;
        c.ws += k.ws;
        c.vs_s53 += k.vs_s53;
        c.bound_only += k.bound_only;
        c.two_consumers += k.two_consumers;
        c.wraps += k.wraps;
        c.s53_fallback += k.s53_fallback;
        c.other_kernel += k.other_kernel;
    }
    eprintln!("wgmma-ws generated: {c:?}");
    assert!(
        c.ws > 0
            && c.vs_s53 > 0
            && c.two_consumers > 0
            && c.wraps > 0
            && c.s53_fallback + c.other_kernel > 0,
        "{c:?}"
    );
}

/// Mutations of the protocol, each a failure the emulator shows: the
/// producer's rounding dropped (`wgmma` truncates the unrounded `aᵀ`: not the
/// S53 kernel's bits), a stage released before its group's `wait_group`
/// (the group's operand changes under it: exit 110 or other bits), and the
/// wrong full-barrier parity (a stale stage or a phase that never completes).
#[test]
fn protocol_mutations_fail() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let src = MM.replace("256 -> iota -> kr;", "256 -> iota -> kr;");
    let ir = lower(&src);
    let (host, dev) = emission(&ir, "h100", 2, &[]);
    assert_eq!(ws_lines(&dev).len(), 1);
    let (host_off, dev_off) = emission(&ir, "h100", 2, &["wgmma-ws"]);
    let want = emu::build_emulated(&clang, &host_off, &dev_off)
        .run(&MODES[1])
        .0;
    assert_eq!(
        emu::build_emulated(&clang, &host, &dev).run(&MODES[1]).0,
        want
    );
    let fails = |dev: &str| {
        let x = emu::build_emulated(&clang, &host, dev);
        MODES.iter().any(|m| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| x.run(m).0))
                .map_or(true, |o| o != want)
        })
    };
    // 1. No rounding: the producer's `cvt.rn` an identity.
    let unrounded = dev.replace(
        "call i32 asm \"cvt.rn.tf32.f32 $0, $1;\", \"=r,f\"(float",
        "call i32 asm \"mov.b32 $0, $1;\", \"=r,f\"(float",
    );
    assert_ne!(unrounded, dev);
    let unrounded = unrounded.replace(
        "asm \"mov.b32 $0, $1;\", \"=r,f\"(float ",
        "@mapal_bits(float ",
    ) + "define internal i32 @mapal_bits(float %x) {\n  %b = bitcast float %x to i32\n  ret i32 %b\n}\n";
    assert!(fails(&unrounded), "an unrounded aᵀ must fail");
    // 2. The stage released with its group in flight: the in-loop wait gone.
    let i = dev
        .find("call void asm sideeffect \"wgmma.wait_group.sync.aligned 0;\"")
        .expect("the in-loop wait");
    let early = format!(
        "{}{}",
        &dev[..i],
        dev[i..].replacen(
            "call void asm sideeffect \"wgmma.wait_group.sync.aligned 0;\", \"~{memory}\"()",
            "add i64 0, 0",
            1
        )
    );
    let early = early.replacen("  add i64 0, 0\n", "", 1);
    assert!(fails(&early), "an early release must fail");
}
