//! gapfix tensor-tile (C3 `mma-fly`, C11 the raster's reuse distance) on
//! programs from the generator (`testgen::GemmProg`: GEMMs it was not
//! designed on), under a 16-bit (`fold: bf16 | f32`) or TF32 (`fold: tf32 |
//! f32`) product set:
//!
//! - **Plans, every part.** A tensor tile's ring differs from the S52 panel
//!   (`--gpu-off=mma-fly`) only on a DRAM-bound site, keeps the block and
//!   its grid, keeps at least the part's in-flight bytes past its wait
//!   (`blocks_per_sm · (stages − 2) ·` the stage's DRAM-streamed bytes ≥
//!   `stream_fly_per_sm`), never more blocks an SM, and its kernel waits
//!   `wait_group(stages − 2)`; a grouped walk (`raster > 1`) is exactly where
//!   one block's panels leave the L2 or row-major's `b` re-reads leave the
//!   site DRAM-bound, and it divides the grid's height.
//! - **Values, the emulator.** Runnable extents with the part's L2 keep and
//!   DRAM rate overridden (`--cost=`) so both rules act: integer operands
//!   (exact in bf16 / TF32, sums exact in f32), the emulated device equal to
//!   the interpreter in every emulator mode, ON and OFF.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed, resolve_profile};
use mapal_interp::{Outcome, run};
use mapal_ir::CategoryIr;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use std::collections::HashMap;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

const PARTS: [&str; 4] = ["rtx-5080", "rtx-4090", "a100", "h100"];

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

/// The GEMM under a product set: `set` 0 `bf16 | f32`, 1 `tf32 | f32`.
fn set_src(p: &testgen::GemmProg, (m, n, k): (u64, u64, u64), set: u8) -> String {
    let fold = match set {
        0 => "fold: bf16 | f32 {",
        _ => "fold: tf32 | f32 {",
    };
    let src = testgen::gemm_src_at(p, m, n, k);
    assert!(src.contains("(0.0, kr) -> fold {"), "{src}");
    src.replace("(0.0, kr) -> fold {", &format!("(0.0, kr) -> {fold}"))
}

fn emission(ir: &CategoryIr, target: &'static str, off: &[&str], extra: &[&str]) -> Emission {
    let mut gpu = GpuArms {
        site_cands: 1,
        ..GpuArms::default()
    };
    // `place` stays on: off, the part is unpriced (no `DeviceCost`: its L2
    // keep the nominal L2), not the profile these rules read.
    for r in ["pdl", "streams", "pinned", "horizontal", "calls"]
        .iter()
        .chain(off)
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let mut opts = EmitOpts {
        host: "generic",
        target,
        gpu,
        ..EmitOpts::default()
    };
    for f in extra {
        assert!(mapal_backend_llvm::flags::apply(&mut opts, f).expect("a flag"));
    }
    emit_detailed(ir, &opts).expect("emits")
}

/// Each tensor-tile kernel's plan fields and its body.
fn mma_kernels(e: &Emission) -> Vec<(HashMap<String, String>, String)> {
    let dev = e.device.as_deref().unwrap_or_default();
    let mut out = Vec::new();
    for (i, l) in dev.lines().enumerate() {
        if !(l.starts_with("; mapal.plan ") && l.contains(" kind=mma ")) {
            continue;
        }
        let f: HashMap<String, String> = l
            .split(' ')
            .filter_map(|f| f.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let name = &f["kernel"];
        let body: String = dev
            .lines()
            .skip(i)
            .skip_while(|x| !(x.starts_with("define ") && x.contains(name.as_str())))
            .take_while(|x| *x != "}")
            .collect::<Vec<_>>()
            .join("\n");
        out.push((f, body));
    }
    out
}

fn num(p: &HashMap<String, String>, k: &str) -> u64 {
    p.get(k)
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("plan field {k}: {p:?}"))
}

#[derive(Default, Clone, Copy, Debug)]
struct Cover {
    kernels: u64,
    fly: u64,
    fly_kept: u64,
    raster_reuse: u64,
    raster_rowmajor: u64,
}

impl std::ops::AddAssign for Cover {
    fn add_assign(&mut self, o: Cover) {
        self.kernels += o.kernels;
        self.fly += o.fly;
        self.fly_kept += o.fly_kept;
        self.raster_reuse += o.raster_reuse;
        self.raster_rowmajor += o.raster_rowmajor;
    }
}

/// Large extents (plans only): the generator's skinny-to-square GEMMs with
/// the column count widened (`n · 2^w`, up to 28672-wide projections) so
/// `b` streams from DRAM on every part.
fn plan_dims(p: &testgen::GemmProg) -> (u64, u64, u64) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let [dm, dn, dk] = p.dims;
    (
        pick(dm, &[64, 128, 256, 1024, 2048]),
        pick(dn, &[4096, 6144, 8192, 14336, 28672]),
        pick(dk, &[1024, 4096, 8192, 14336]),
    )
}

fn check_plans(p: &testgen::GemmProg, set: u8, part: &'static str) -> Cover {
    let (m, n, k) = plan_dims(p);
    let src = set_src(p, (m, n, k), set);
    let ctx = format!("{part} set={set} {m}x{n}x{k} bt={}", p.bt);
    let ir = lower(&src);
    let prof = resolve_profile(part).expect("a part");
    let g = prof.gpu().expect("a GPU part");
    let on = mma_kernels(&emission(&ir, part, &[], &[]));
    let off = mma_kernels(&emission(&ir, part, &["mma-fly"], &[]));
    assert_eq!(on.len(), off.len(), "{ctx} the same tensor kernels");
    let mut c = Cover::default();
    for ((t, body), (o, _)) in on.iter().zip(&off) {
        c.kernels += 1;
        let (bm, bn) = (num(t, "rows") / num(t, "gy"), num(t, "c") / num(t, "gx"));
        // A TF32 tile's panels are f32 (its plan line names no widths).
        let width = |f: &str| t.get(f).map_or(4, |_| num(t, f));
        let (ea, eb) = (width("ea"), width("eb"));
        let keep = g.l2_keep_bytes();
        let (a_dram, b_dram) = (m * k * ea > keep, k * n * eb > keep);
        // The S52 panel off a DRAM-resident operand, and on a tile that
        // converts its fragments as it reads them (TF32, an f32 panel).
        let op_bytes = if t.get("op").is_none_or(|o| o.contains("tf32")) {
            4
        } else {
            2
        };
        let narrow = op_bytes == 2 && ea == 2 && eb == 2;
        if !(a_dram || b_dram) || !narrow {
            assert_eq!(t, o, "{ctx}: the S52 panel");
        }
        for f in [
            "gx", "gy", "block", "mt", "nt", "warps_x", "warps_y", "raster",
        ] {
            assert_eq!(t[f], o[f], "{ctx}: mma-fly moved {f}");
        }
        let (stages, bps, bk) = (num(t, "stages"), num(t, "blocks_per_sm"), num(t, "bk"));
        if (num(t, "bk"), stages, bps) != (num(o, "bk"), num(o, "stages"), num(o, "blocks_per_sm"))
        {
            c.fly += 1;
            assert!(stages >= 3, "{ctx} {t:?}");
            assert!(bps <= num(o, "blocks_per_sm"), "{ctx} {t:?} / {o:?}");
            let stage_dram = u64::from(a_dram) * bm * bk * ea + u64::from(b_dram) * bk * bn * eb;
            assert!(
                bps * (stages - 2) * stage_dram >= g.stream_fly_per_sm(),
                "{ctx} ring short of the part's in-flight bytes {t:?}"
            );
            if bps == num(o, "blocks_per_sm") {
                c.fly_kept += 1;
            }
        }
        // The ring's wait: `stages − 2` groups left in flight.
        let wait = format!("@llvm.nvvm.cp.async.wait.group(i32 {})", stages - 2);
        assert!(body.contains(&wait), "{ctx}: {wait} in the kernel {t:?}");
        // C11: the grouped walk exactly where its gate says.
        let wave = g.sms * bps;
        let (gx, gy) = (num(t, "gx"), num(t, "gy"));
        // The warp product's rate (`Gpu::matrix_flops_per_ns`).
        let tf32 = t.get("op").is_none_or(|o| o.contains("tf32"));
        let per_clk = match tf32 {
            true => g.mma_tf32_per_clk,
            false => g.mma_half_per_clk,
        };
        let rate = g.sms * per_clk * g.clock_khz / 1_000_000;
        let rereads = g.b_rereads(bm, bn, wave, m, n, k, eb);
        let bytes = (rereads + 1) * n * k * eb + m * k * eb;
        let panels = (bm + bn) * k * 4 > g.l2_bytes;
        let reuse = rereads > 0 && g.dram_bound(bytes, 2 * m * n * k, rate);
        let walks = gx >= 2 && gy >= 2 && gx * gy > wave;
        let r = num(t, "raster");
        assert_eq!(
            r > 1,
            walks && (panels || reuse),
            "{ctx}: raster {r} rereads {rereads} bytes {bytes} rate {rate} wave {wave} panels {panels} reuse {reuse} {t:?}"
        );
        assert_eq!(gy % r, 0, "{ctx}: the group divides the grid");
        if r > 1 && !panels {
            c.raster_reuse += 1;
        }
        if walks && !panels && !reuse && rereads > 0 {
            c.raster_rowmajor += 1;
        }
    }
    c
}

/// 64 generated GEMMs × both product sets × four parts' properties: every
/// tensor tile's ring and walk as the rules state them; every coverage
/// counter non-zero.
#[test]
fn tensor_tile_rules_on_generated_shapes() {
    let mut runner = TestRunner::deterministic();
    let strat = testgen::gemm_prog_strategy();
    let progs: Vec<testgen::GemmProg> = (0..64)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let covers = testgen::par_check(&progs, |z, p| {
        let mut c = Cover::default();
        for part in PARTS {
            c += check_plans(p, (z % 2) as u8, part);
        }
        c
    });
    let mut c = Cover::default();
    for k in covers {
        c += k;
    }
    eprintln!("tensor-tile plans: {c:?}");
    assert!(c.kernels >= 64, "{c:?}");
    assert!(
        c.fly > 0 && c.fly_kept > 0,
        "the generator reaches mma-fly: {c:?}"
    );
    assert!(
        c.raster_reuse > 0,
        "the generator reaches the reuse gate: {c:?}"
    );
    assert!(
        c.raster_rowmajor > 0,
        "a re-reading grid the pipe hides stays row-major: {c:?}"
    );
}

const MODES: [[(&str, &str); 2]; 3] = [
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "1")],
];

/// The part's L2 keep overridden so the emulator's extents stream from
/// "DRAM", and its loaded latency so their panels outlast their products:
/// `mma-fly` acts.
const FLY: [&str; 2] = ["--cost=l2_keep_bytes=65536", "--cost=lat_ns=100000"];

/// The keep and the DRAM rate overridden so row-major's `b` re-reads leave
/// the site DRAM-bound: the grouped walk acts.
const REREAD: [&str; 2] = ["--cost=l2_keep_bytes=65536", "--cost=dram_bytes_per_ns=1"];

/// **Values through the emulator.** 6 generated GEMMs at runnable extents
/// (`m` 512, `n` 6144: 192 tiles past the 168-block wave; `k` 64), each
/// product set: the ring (`mma-fly`) and the grouped walk on, the emulated
/// device equal to the interpreter in every emulator mode, and the OFF arm
/// too.
#[test]
fn tensor_tile_values_through_the_emulator() {
    let Some(cc) = emu::clang() else {
        eprintln!("no clang: emulator runs skipped");
        return;
    };
    let mut runner = TestRunner::deterministic();
    let strat = testgen::gemm_prog_strategy();
    let progs: Vec<testgen::GemmProg> = (0..6)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let counts = testgen::par_check(&progs, |z, p| {
        let set = (z % 2) as u8;
        // The TF32 tile reads `b` along its lanes (an `n × k` `b` takes the
        // FMA tile): the TF32 programs keep `b` `k × n`.
        let p = &testgen::GemmProg {
            bt: p.bt && set == 0,
            ..p.clone()
        };
        let src = set_src(p, (512, 6144, 64), set);
        let ctx = format!("#{z} set={set} bt={}", p.bt);
        let ir = lower(&src);
        let rr = run(&ir, 20_000_000_000);
        assert!(matches!(rr.outcome, Outcome::Done(_)), "{:?}", rr.outcome);
        let want = rr.output;
        let on = emission(&ir, "rtx-5080", &[], &FLY);
        let walk = emission(&ir, "rtx-5080", &[], &REREAD);
        let off = emission(&ir, "rtx-5080", &["mma-fly"], &FLY);
        let ks = mma_kernels(&on);
        assert!(!ks.is_empty(), "{ctx}: a tensor tile\n{src}");
        let fly = ks.iter().any(|(t, _)| num(t, "stages") > 2);
        let raster = mma_kernels(&walk).iter().any(|(t, _)| num(t, "raster") > 1);
        let dev = |e: &Emission| e.device.clone().unwrap_or_default();
        let built = emu::build_emulated(&cc, &on.text, &dev(&on));
        let mut runs = 0;
        for (env, (got, _)) in MODES.iter().zip(built.run_each(&MODES)) {
            assert_eq!(got, want, "{ctx} {env:?}\n{src}");
            runs += 1;
        }
        let built = emu::build_emulated(&cc, &walk.text, &dev(&walk));
        for (env, (got, _)) in MODES[2..].iter().zip(built.run_each(&MODES[2..])) {
            assert_eq!(got, want, "{ctx} grouped walk {env:?}\n{src}");
            runs += 1;
        }
        assert_eq!(
            emu::run_emulated(&cc, &off.text, &dev(&off)),
            want,
            "{ctx} OFF"
        );
        [usize::from(fly), usize::from(raster), runs]
    });
    let sum = |i: usize| counts.iter().map(|c| c[i]).sum::<usize>();
    eprintln!(
        "tensor-tile values: {} ring kernels, {} grouped walks, {} emulated runs",
        sum(0),
        sum(1),
        sum(2)
    );
    assert!(sum(0) >= 3 && sum(1) >= 3, "both rules valued");
}
