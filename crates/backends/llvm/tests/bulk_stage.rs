//! TMA `bulk-stage`: on a part with the bulk-copy engine (`Caps::bulk`:
//! compute capability 9.0 and up — the RTX 5080, the H100), the warp TF32
//! `mma` tile's panels are staged by one warp's row copies
//! (`cp.async.bulk`) whose bytes a stage's transaction barrier counts
//! (`mbarrier.expect_tx`), every thread waiting on its phase
//! (`mbarrier.try_wait.parity`) — where the engine's geometry covers every
//! panel row. The copies move the same bytes into the same shared layout as
//! the `cp.async` arm, so on programs from the generator
//! (`testgen::Tf32Prog`, the TF32 GEMMs `tf32_tile.rs` judges) the emulated
//! device's values are the OFF arm's, byte for byte, in every emulator mode
//! (a copy is applied only when a thread waits on its barrier: a missing or
//! early wait reads stale shared memory), and within ADR-0041 D3's bound of
//! the interpreter.

use mapal_backend_llvm::{EmitOpts, GpuArms, emit_detailed};
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

/// The default face on `target` with the rungs `off` off and `on` on:
/// `place` off (every capable site on the device), each site its own kernel,
/// one candidate.
fn emission(ir: &CategoryIr, target: &'static str, off: &[&str], on: &[&str]) -> (String, String) {
    let mut gpu = GpuArms::default();
    for r in ["place", "pdl", "horizontal", "calls"].iter().chain(off) {
        *gpu.off.rung(r).expect("rung") = true;
    }
    for r in on {
        *gpu.on.rung(r).expect("rung") = true;
    }
    gpu.site_cands = 1;
    let e = emit_detailed(
        ir,
        &EmitOpts {
            host: "generic",
            target,
            gpu,
            ..EmitOpts::default()
        },
    )
    .expect("emits");
    (e.text, e.device.unwrap_or_default())
}

fn bulk_lines(dev: &str) -> usize {
    dev.lines()
        .filter(|l| l.starts_with("; mapal.plan ") && l.contains(" stage=bulk"))
        .count()
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

const MM: &str = "fn main() {
    16384 -> iota -> ia;
    ia -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 } -> a;
    65536 -> iota -> ib;
    ib -> map { t -> (t * 5 + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 } -> b;
    32768 -> iota -> ic;
    128 -> iota -> kr;
    ic -> map { t -> t / 256 -> i; t % 256 -> j; (0.0, kr) -> fold: tf32 | f32 { acc, k -> acc + a[i * 128 + k] * b[k * 256 + j] } } -> y;
    y[0] -> println;
}
";

/// The capability as geometry, per part, under the rung's ON arm: the RTX
/// 5080 (sm_120) and the H100 (sm_90a, its `wgmma` arm off) stage the TF32
/// tile's panels by bulk copies — one barrier a stage, its bytes counted —
/// the RTX 4090 (sm_89, no engine) by `cp.async`. The rules ship it off on
/// every part (the 5080's measured loss, `gemv-tma.md`): `cp.async`.
#[test]
fn staged_where_the_engine_is() {
    let ir = lower(MM);
    let on = &["bulk-stage"][..];
    for (target, off, on, bulk) in [
        ("rtx-5080", &[][..], on, true),
        ("h100", &["wgmma-hopper"][..], on, true),
        ("rtx-4090", &[][..], on, false),
        ("rtx-5080", &[][..], &[][..], false),
        ("h100", &["wgmma-hopper"][..], &[][..], false),
    ] {
        let (_, dev) = emission(&ir, target, off, on);
        assert!(dev.contains(" kind=mma "), "{target}: the warp mma tile");
        assert_eq!(bulk_lines(&dev) > 0, bulk, "{target} {off:?}");
        assert_eq!(
            dev.contains("cp.async.bulk.shared::cluster"),
            bulk,
            "{target}"
        );
        assert_eq!(dev.contains("mbarrier.try_wait.parity"), bulk, "{target}");
        assert_eq!(
            dev.contains("call void @llvm.nvvm.cp.async.cg.shared.global.16"),
            !bulk,
            "{target}"
        );
    }
}

#[derive(Debug, Default)]
struct Cover {
    programs: u64,
    bulk: u64,
    unstaged: u64,
    bt: u64,
}

/// 48 generated TF32 GEMMs on the RTX 5080: wherever the tile is staged by
/// bulk copies, the emulated device equals the `cp.async` arm byte for byte
/// in every emulator mode and stays within D3's bound of the interpreter;
/// every coverage counter non-zero.
#[test]
fn generated_tf32_tiles_equal_the_cp_async_arm() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::Tf32Prog> = (0..48)
        .map(|_| {
            testgen::tf32_prog_strategy()
                .new_tree(&mut runner)
                .unwrap()
                .current()
        })
        .filter(testgen::tf32_admits)
        .collect();
    let covers = testgen::par_check(&progs, |_, p| {
        let (src, (_, k)) = testgen::tf32_src(p);
        let ir = lower(&src);
        let (host, dev) = emission(&ir, "rtx-5080", &[], &["bulk-stage"]);
        let n = bulk_lines(&dev) as u64;
        let mut c = Cover {
            programs: 1,
            bulk: u64::from(n > 0),
            unstaged: u64::from(n == 0 && dev.contains(" kind=mma ")),
            bt: u64::from(p.bt && n == 0),
        };
        if n == 0 {
            return c;
        }
        let (host_off, dev_off) = emission(&ir, "rtx-5080", &["bulk-stage"], &[]);
        assert_eq!(bulk_lines(&dev_off), 0, "OFF\n{src}");
        let on = emu::build_emulated(&clang, &host, &dev);
        let off = emu::build_emulated(&clang, &host_off, &dev_off);
        let f = floats(&mapal_interp::run(&ir, 1 << 34).output);
        let coef = 2.0 * (bound_coef(k, true) + bound_coef(k, false));
        for mode in MODES {
            let (a, _) = on.run(&mode);
            let (b, _) = off.run(&mode);
            assert_eq!(a, b, "bulk vs cp.async {mode:?}\n{src}");
            let g = floats(&a);
            for cell in 0..2 {
                let (y, yf, mag) = (g[2 * cell], f[2 * cell], f[2 * cell + 1]);
                assert!(
                    (y - yf).abs() <= coef * mag,
                    "{mode:?} cell {cell}: {y} vs {yf} (M {mag})\n{src}"
                );
            }
        }
        c.programs = 1;
        c
    });
    let mut c = Cover::default();
    for k in covers {
        c.programs += k.programs;
        c.bulk += k.bulk;
        c.unstaged += k.unstaged;
        c.bt += k.bt;
    }
    eprintln!("bulk-stage generated: {c:?}");
    assert!(
        c.programs > 0 && c.bulk > 0 && c.unstaged + c.bt > 0,
        "{c:?}"
    );
}
