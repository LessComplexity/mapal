//! wgmma-16bit (`wgmma16`): a 16-bit tensor tile on the part's warpgroup
//! product — the warp-specialized tile family of `wgmma-ws` (`tests/
//! hopper_tma.rs`) with the operand format a parameter. Where the part lists
//! `wgmma.m64nNk16` for the format (`Caps::matrix_group`) at a measured rate
//! above its warp `mma.sync.m16n8k16` (`rules::wgmma16_pays`), a bf16 / f16
//! site computes `c = a·b` as stored: `a` from a K-major descriptor where it
//! is stored in the format, else from registers (an f32 `a` rounded to the
//! format as its fragment is loaded); `b` from a descriptor, K-major or
//! MN-major (the transpose bit); TMA boxes onto an mbarrier ring fed by a
//! producer warpgroup.
//!
//! The emulator (`mapal_gpu_emu.c` `emu_wgmma16`) reads each descriptor
//! element through its canonical swizzled layout (a wrong LBO / SBO /
//! transpose bit is a wrong value) and sums each k16 step as the warp
//! `mma.m16n8k16` does, so on generated programs the warpgroup kernel equals
//! the warp tile (`--gpu-off=wgmma16`) byte for byte where both exist, and
//! stays within ADR-0041 D3's bound of the interpreter, in every emulator
//! mode.

use mapal_backend_llvm::{CostOver, DeviceCost, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run, run_weakest};
use mapal_ir::{CategoryIr, bound_coef_unit};
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

/// The default face on `target` with `n` SMs and the rungs `off` off
/// (`+name`: forced on): `place` off (every capable site on the device), each
/// site its own kernel, one candidate.
fn emission(ir: &CategoryIr, target: &'static str, n: u64, off: &[&str]) -> (String, String) {
    let mut gpu = GpuArms::default();
    for r in ["place", "pdl", "horizontal", "calls"].iter().chain(off) {
        match r.strip_prefix('+') {
            Some(on) => *gpu.on.rung(on).expect("rung") = true,
            None => *gpu.off.rung(r).expect("rung") = true,
        }
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

/// [`emission`] on the H100 profile with its placement priced (`place` on:
/// the device cost model the copies' price reads), `horizontal` / `calls` /
/// `pdl` off, the rungs `off` off.
fn placed(ir: &CategoryIr, off: &[&str]) -> (String, String) {
    let mut gpu = GpuArms::default();
    for r in ["pdl", "horizontal", "calls"].iter().chain(off) {
        *gpu.off.rung(r).expect("rung") = true;
    }
    gpu.site_cands = 1;
    let e = emit_detailed(
        ir,
        &EmitOpts {
            host: "generic",
            target: "h100",
            gpu,
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

/// The plan lines of 16-bit warpgroup kernels.
fn ws16_lines(dev: &str) -> Vec<&str> {
    plan_lines(dev)
        .into_iter()
        .filter(|l| l.contains(" ws=1 ") && l.contains(" op=m64n"))
        .collect()
}

/// The plan lines of the warp 16-bit tile.
fn mma16_lines(dev: &str) -> Vec<&str> {
    plan_lines(dev)
        .into_iter()
        .filter(|l| l.contains(" kind=mma ") && l.contains(" op=m16n8k16"))
        .collect()
}

/// A plan line's `key=` value.
fn field<'a>(line: &'a str, key: &str) -> &'a str {
    line.split_whitespace()
        .find_map(|w| w.strip_prefix(key).and_then(|v| v.strip_prefix('=')))
        .unwrap_or("")
}

fn num(line: &str, key: &str) -> u64 {
    field(line, key).parse().unwrap_or(0)
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

/// `benches/gpu/gen_tiles.py`'s `pt_pj_*_bf16` form at 256 × 256 × 256: `a`
/// f32, `b` stored bf16 (lanes-major), every fold `: bf16 | f32`.
const PJ: &str = "fn main(): bf16 | f32 {
    65536 -> iota -> ia;
    ia -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 } -> a;
    65536 -> iota -> ib;
    ib -> map { t -> (t * 5 + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 -> to_bf16 } -> b;
    65536 -> iota -> ic;
    256 -> iota -> kr;
    ic -> map { t -> t / 256 -> i; t % 256 -> j; (0.0, kr) -> fold { acc, k -> b[k * 256 + j] -> widen_f32 -> w; acc + a[i * 256 + k] * w } } -> y;
    y[0] -> println;
    y[65535] -> println;
}
";

/// The kernel text on `sm_90a`: the 16-bit warpgroup instruction with `a`
/// from registers (rounded `cvt.rn.bf16x2.f32`) and `b` MN-major (transpose
/// bit 1), the stage's boxes onto full barriers, no rounding pass (no ready
/// barrier wait), no `mma.sync`; OFF (`--gpu-off=wgmma16`) the warp tile.
/// Where `ptxas` is installed: no spill, `HGMMA.64x…BF16` in the SASS.
#[test]
fn ws16_kernel_text_on_sm_90a() {
    // `a` read by the product alone: stored rounded by its producer
    // (`PrecisionPlan::store`) — `a` from a descriptor; printed too: kept
    // f32 — `a` from registers, rounded in its fragment load.
    let kept = PJ.replace(
        "    y[0] -> println;",
        "    a[3] -> println;\n    y[0] -> println;",
    );
    for (src, home, ea) in [(PJ.to_string(), "smem", 2), (kept, "regs", 4)] {
        let ir = lower(&src);
        let (_, dev) = emission(&ir, "h100", 4, &[]);
        let ws = ws16_lines(&dev);
        assert_eq!(ws.len(), 1, "{}", plan_lines(&dev).join("\n"));
        let l = ws[0];
        assert_eq!(field(l, "face"), "HalfWgmma", "{l}");
        assert_eq!((field(l, "a_home"), field(l, "bt")), (home, "0"), "{l}");
        assert_eq!((num(l, "ea"), num(l, "eb")), (ea, 2), "{l}");
        for needle in [
            "k16.f32.bf16.bf16",
            "cp.async.bulk.tensor",
            "setmaxnreg.inc.sync.aligned.u32",
            "wgmma.commit_group.sync.aligned;",
        ] {
            assert!(dev.contains(needle), "{needle}");
        }
        assert_eq!(dev.contains("cvt.rn.bf16x2.f32"), home == "regs");
        assert_eq!(
            dev.contains("wgmma.wait_group.sync.aligned 1;"),
            home == "smem"
        );
        assert!(!dev.contains("mma.sync.aligned") && !dev.contains("cvt.rn.tf32"));
        ptxas_check(&dev, l);
    }
    let ir = lower(PJ);
    let (_, off) = emission(&ir, "h100", 4, &["wgmma16"]);
    assert!(ws16_lines(&off).is_empty() && !mma16_lines(&off).is_empty());
    // No part without the warpgroup product takes it.
    for t in ["rtx-5080", "rtx-4090", "a100"] {
        let (_, d) = emission(&ir, t, 4, &[]);
        assert!(ws16_lines(&d).is_empty(), "{t}");
    }
}

/// `llc` + `ptxas -arch=sm_90a` on the module (skipped where either is
/// missing): 0 spill, registers within the launch's share, and the SASS
/// intents of the plan line (`HGMMA` 16-bit, `UTMALDG`, `SYNCS`).
fn ptxas_check(dev: &str, plan: &str) -> String {
    use std::process::Command;
    let have = |t: &str| Command::new(t).arg("--version").output().is_ok();
    if !have("llc") || !have("ptxas") || !have("cuobjdump") {
        return String::new();
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
        return String::new();
    }
    let out = Command::new("ptxas")
        .args(["-arch=sm_90a", "-v"])
        .arg(p("d.ptx"))
        .arg("-o")
        .arg(p("d.cubin"))
        .output()
        .unwrap();
    let log = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(), "{log}");
    let name = field(plan, "kernel");
    let block = num(plan, "block");
    let mut it = log.split("Compiling entry function '").skip(1);
    let fun = it
        .find(|s| s.starts_with(name))
        .unwrap_or_else(|| panic!("{name} in\n{log}"));
    assert!(
        fun.contains("0 bytes spill stores, 0 bytes spill loads"),
        "{fun}"
    );
    assert!(!fun.contains("serialized"), "wgmma serialized: {fun}");
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
    let sass = String::from_utf8_lossy(&sass.stdout).to_string();
    for op in ["UTMALDG", "HGMMA.64x", ".BF16", "SYNCS.PHASECHK"] {
        assert!(sass.contains(op), "{op}");
    }
    assert!(
        !sass.contains("LDGSTS") && !sass.contains(" STL") && !sass.contains("HMMA.16816"),
        "no cp.async, no spill, no warp mma"
    );
    sass
}

/// A 192 × 64 × 256 bf16 GEMM with a residual epilogue beside an
/// independent f32 fold of the same extents (the generator's `wg16_01`): a
/// pair `horizontal` would merge (the warp-specialized kernel's one consumer
/// warpgroup: 256 threads, the fold's block).
const MERGED: &str = "fn main() {
    49152 -> iota -> ia;
    ia -> map { t -> (t * 21 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 } -> a;
    16384 -> iota -> ib;
    ib -> map { t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 -> to_bf16 } -> b;
    12288 -> iota -> ic;
    ic -> map { t -> (t * 3 + 7) % 17 - 8 -> widen_f32 -> v; v / 8.0 } -> r;
    256 -> iota -> kr;
    ic -> map { t -> t / 64 -> i; t % 64 -> j; (0.0, kr) -> fold: bf16 | f32 { acc, q -> a[i * 256 + q] -> x; b[q * 64 + j] -> widen_f32 -> y; acc + x * y } -> u; r[t] -> rv; u * 0.5 + rv } -> y;
    ia -> map { t -> (t * 21 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 } -> am;
    ib -> map { t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 -> to_bf16 } -> bm;
    ic -> map: f32 { t -> t / 64 -> i; t % 64 -> j; (0.0, kr) -> fold { acc, q -> am[i * 256 + q] -> x; bm[q * 64 + j] -> widen_f32 -> y; x * y } } -> mag;
    y[245] -> println;
    mag[245] -> println;
}
";

/// A warp-specialized launch never joins a merged launch (`horizontal`):
/// merged, every such member faulted on the H100 (`illegal memory access`,
/// 22 of 22 generated programs; alone 26 of 26 within the bound) — the
/// GEMM launches alone, the independent fold beside it keeps its own launch
/// or merges with others.
#[test]
fn ws16_launches_alone() {
    let ir = lower(MERGED);
    let mut gpu = GpuArms::default();
    for r in ["place", "pdl", "calls"] {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let e = emit_detailed(
        &ir,
        &EmitOpts {
            host: "generic",
            target: "h100",
            gpu,
            ..EmitOpts::default()
        },
    )
    .expect("emits");
    let dev = e.device.unwrap_or_default();
    assert_eq!(ws16_lines(&dev).len(), 1, "{}", plan_lines(&dev).join("\n"));
    assert!(
        !dev.lines()
            .any(|l| l.starts_with("define internal void @hz.") && l.contains("%tm0")),
        "a warp-specialized member in a merged launch"
    );
    assert!(
        dev.lines().any(
            |l| l.starts_with("define ptx_kernel") && l.contains("\"nvvm.grid_constant\" %tm0")
        ),
        "the kernel takes its maps itself"
    );
}

/// A stage deeper than one 128-byte swizzle row (`bk` 256: the `a` box's
/// digit columns, a k16 step's descriptor offset across them) — a 128 × 64 ×
/// 1024 GEMM on the profile's 132 SMs, its placement priced (the copies'
/// price deepens its small tile's stage) — on both `a` homes equals the warp
/// tile byte for byte in every emulator mode.
#[test]
fn deep_stages_equal_the_warp_tile() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let deep = PJ
        .replace("65536 -> iota -> ia", "131072 -> iota -> ia")
        .replace(
            "ic -> map { t -> t / 256 -> i; t % 256 -> j;",
            "ic -> map { t -> t / 64 -> i; t % 64 -> j;",
        )
        .replace("65536 -> iota -> ic", "8192 -> iota -> ic")
        .replace("256 -> iota -> kr", "1024 -> iota -> kr")
        .replace("b[k * 256 + j]", "b[k * 64 + j]")
        .replace("a[i * 256 + k]", "a[i * 1024 + k]")
        .replace("y[65535]", "y[8191]");
    let kept = deep.replace(
        "    y[0] -> println;",
        "    a[3] -> println;\n    y[0] -> println;",
    );
    for (src, home) in [(deep.clone(), "smem"), (kept, "regs")] {
        let ir = lower(&src);
        let (host, dev) = placed(&ir, &[]);
        let ws = ws16_lines(&dev);
        assert_eq!(ws.len(), 1, "{}", plan_lines(&dev).join("\n"));
        assert!(
            num(ws[0], "bk") > 64 && field(ws[0], "a_home") == home,
            "{}",
            ws[0]
        );
        let (host_off, dev_off) = placed(&ir, &["wgmma16"]);
        let on = emu::build_emulated(&clang, &host, &dev);
        let off = emu::build_emulated(&clang, &host_off, &dev_off);
        for mode in MODES {
            assert_eq!(on.run(&mode).0, off.run(&mode).0, "{mode:?}\n{src}");
        }
    }
}

/// The two `a` homes on the emulated device equal the warp tile
/// (`--gpu-off=wgmma16`) byte for byte in every mode.
#[test]
fn ws16_equals_the_warp_tile() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let kept = PJ.replace(
        "    y[0] -> println;",
        "    a[3] -> println;\n    y[0] -> println;",
    );
    for src in [PJ.to_string(), kept] {
        let ir = lower(&src);
        let (host, dev) = emission(&ir, "h100", 2, &[]);
        assert_eq!(ws16_lines(&dev).len(), 1);
        let (host_off, dev_off) = emission(&ir, "h100", 2, &["wgmma16"]);
        assert_eq!(mma16_lines(&dev_off).len(), 1);
        let on = emu::build_emulated(&clang, &host, &dev);
        let off = emu::build_emulated(&clang, &host_off, &dev_off);
        for mode in MODES {
            let (a, _) = on.run(&mode);
            let (b, _) = off.run(&mode);
            assert_eq!(a, b, "{mode:?}\n{src}");
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Cover {
    programs: u64,
    ws16: u64,
    a_regs: u64,
    a_smem: u64,
    b_mn: u64,
    b_k: u64,
    f16: u64,
    two_consumers: u64,
    wraps: u64,
    deep: u64,
    post: u64,
    gate: u64,
    vs_warp: u64,
    bound_only: u64,
    persist: u64,
    refused: u64,
}

impl std::ops::AddAssign for Cover {
    fn add_assign(&mut self, o: Cover) {
        self.programs += o.programs;
        self.ws16 += o.ws16;
        self.a_regs += o.a_regs;
        self.a_smem += o.a_smem;
        self.b_mn += o.b_mn;
        self.b_k += o.b_k;
        self.f16 += o.f16;
        self.two_consumers += o.two_consumers;
        self.wraps += o.wraps;
        self.deep += o.deep;
        self.post += o.post;
        self.gate += o.gate;
        self.vs_warp += o.vs_warp;
        self.bound_only += o.bound_only;
        self.persist += o.persist;
        self.refused += o.refused;
    }
}

/// One generated program on the H100 at `n_sms` SMs: the emulated device in
/// every mode within D3's bound of the weakest member's interpreter (the
/// set's `u_t`, `u_a = 2⁻²³`: each side within the coefficient of the exact
/// value, so within twice it of each other; the gate epilogue's `exp` form
/// adds `2⁻¹⁸` relative), equal byte for byte to the warp tile (`--gpu-off=
/// wgmma16`) where that one folds in order, and to the persistent walk
/// (`--gpu-on=ws-persist`) where its price takes the site.
fn check(clang: &str, i: usize, p: &testgen::Wgmma16Prog) -> Cover {
    let (src, (pairs, k)) = testgen::wgmma16_src(p);
    let ir = lower(&src);
    let rr = run(&ir, 4_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let weak = floats(&run_weakest(&ir, 4_000_000_000, &[]).output);
    let unit = Some(if p.f16 { 2f64.powi(-11) } else { 2f64.powi(-8) });
    let c = bound_coef_unit(k, unit) + bound_coef_unit(k, None);
    let rel = if p.post == 2 {
        2f64.powi(-18)
    } else {
        2f64.powi(-22)
    };
    let check_values = |got: &str, what: &str| {
        let g = floats(got);
        assert_eq!(g.len(), weak.len(), "{what}\n{src}");
        for q in 0..pairs {
            let (y, w, m) = (g[2 * q], weak[2 * q], weak[2 * q + 1]);
            let tol = 2.0 * c * m + rel * w.abs();
            assert!(
                (y - w).abs() <= tol,
                "{what} pair {q}: {y} vs {w} (M {m}, tol {tol})\n{src}"
            );
        }
    };
    // 0: the profile's 132 SMs (a small tile's stage deepens under the copies' price).
    let n_sms = [1, 2, 8, 0][i % 4];
    let (host, dev) = emission(&ir, "h100", n_sms, &["half-fill"]);
    let ws = ws16_lines(&dev);
    let mut cv = Cover {
        programs: 1,
        ..Cover::default()
    };
    let built = emu::build_emulated(clang, &host, &dev);
    let mut outs = Vec::new();
    for mode in MODES {
        let (got, _) = built.run(&mode);
        check_values(&got, &format!("on {mode:?}"));
        outs.push(got);
    }
    let Some(l) = ws.first() else {
        cv.refused = 1;
        return cv;
    };
    cv.ws16 = 1;
    cv.a_regs = u64::from(field(l, "a_home") == "regs");
    cv.a_smem = u64::from(field(l, "a_home") == "smem");
    cv.b_k = u64::from(field(l, "bt") == "1");
    cv.b_mn = u64::from(field(l, "bt") == "0");
    cv.f16 = u64::from(field(l, "op").ends_with(".f16"));
    cv.two_consumers = u64::from(num(l, "groups") == 2);
    cv.wraps = u64::from(k / num(l, "bk") > num(l, "stages"));
    cv.deep = u64::from(num(l, "bk") > 64);
    cv.post = u64::from(p.post == 1 && field(l, "epi") == "1");
    cv.gate = u64::from(p.post == 2 && field(l, "epi") == "1");
    // The warp tile, where it folds in order: the same k16 sums.
    let (host_off, dev_off) = emission(&ir, "h100", n_sms, &["half-fill", "wgmma16"]);
    let warp = mma16_lines(&dev_off);
    let ordered = warp.len() == 1 && field(warp[0], "combine") == "none";
    let off = emu::build_emulated(clang, &host_off, &dev_off);
    let (walk_h, walk_d) = emission(&ir, "h100", n_sms, &["half-fill", "+ws-persist"]);
    let walk = ws16_lines(&walk_d)
        .first()
        .is_some_and(|l| l.contains(" persist="));
    let walked = walk.then(|| emu::build_emulated(clang, &walk_h, &walk_d));
    cv.persist = u64::from(walk);
    for (mode, a) in MODES.iter().zip(&outs) {
        let (b, _) = off.run(mode);
        check_values(&b, &format!("off {mode:?}"));
        if ordered {
            assert_eq!(a, &b, "ws16 vs the warp tile {mode:?}\n{src}");
        }
        if let Some(w) = &walked {
            assert_eq!(a, &w.run(mode).0, "walk vs a CTA a tile {mode:?}\n{src}");
        }
    }
    cv.vs_warp = u64::from(ordered);
    cv.bound_only = u64::from(!ordered);
    cv
}

/// The real device's check (`MAPAL_WG16_DUMP=<dir>`, ignored otherwise): 48
/// generated programs (the generator's sequence past the emulator test's)
/// that take the warpgroup tile on the h100 profile, each as
/// `wg16_<i>.mapal` beside `wg16_<i>.want` — per printed pair the weakest
/// member's interpreter value and the tolerance [`check`] allows — for a
/// device run (`mapal build --target=h100`) compared on the part.
#[test]
#[ignore]
fn dump_hw_programs() {
    let Ok(dir) = std::env::var("MAPAL_WG16_DUMP") else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let mut kept = 0;
    for i in 0..400 {
        let p = testgen::wgmma16_prog_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        if i < 32 || kept == 48 {
            continue;
        }
        let (src, (pairs, k)) = testgen::wgmma16_src(&p);
        let ir = lower(&src);
        let (_, dev) = emission(&ir, "h100", 0, &[]);
        if ws16_lines(&dev).is_empty() {
            continue;
        }
        let weak = floats(&run_weakest(&ir, 4_000_000_000, &[]).output);
        let unit = Some(if p.f16 { 2f64.powi(-11) } else { 2f64.powi(-8) });
        let c = bound_coef_unit(k, unit) + bound_coef_unit(k, None);
        let rel = if p.post == 2 {
            2f64.powi(-18)
        } else {
            2f64.powi(-22)
        };
        let mut want = String::new();
        for q in 0..pairs {
            let (w, m) = (weak[2 * q], weak[2 * q + 1]);
            want += &format!("{w:e} {m:e} {:e}\n", 2.0 * c * m + rel * w.abs());
        }
        std::fs::write(format!("{dir}/wg16_{kept:02}.mapal"), &src).unwrap();
        std::fs::write(format!("{dir}/wg16_{kept:02}.want"), want).unwrap();
        kept += 1;
    }
    assert_eq!(kept, 48);
}

/// 32 generated programs (bf16 / f16 GEMMs at warpgroup-tile extents, both
/// `a` homes, both `b` majors, residual and gate epilogues) on the H100 at 1,
/// 2, 8 and its 132 SMs: every check of [`check`]; every coverage counter
/// non-zero but `deep` (a stage past one 128-byte row: the copies' price at
/// the profile's rates rarely asks it of so small a site —
/// [`deep_stages_equal_the_warp_tile`] pins it).
#[test]
fn generated_ws16_within_the_bound() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::Wgmma16Prog> = (0..32)
        .map(|_| {
            testgen::wgmma16_prog_strategy()
                .new_tree(&mut runner)
                .unwrap()
                .current()
        })
        .collect();
    let covers = testgen::par_check(&progs, |i, p| check(&clang, i, p));
    let mut c = Cover::default();
    for k in covers {
        c += k;
    }
    eprintln!("wgmma16 generated: {c:?}");
    let all = [
        c.ws16,
        c.a_regs,
        c.a_smem,
        c.b_mn,
        c.b_k,
        c.f16,
        c.two_consumers,
        c.wraps,
        c.post,
        c.gate,
        c.vs_warp,
        c.persist,
        c.refused,
    ];
    assert!(all.iter().all(|&x| x > 0), "{c:?}");
}
