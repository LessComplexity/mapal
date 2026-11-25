//! ADR-0048 through the backends, over generated programs
//! (`testgen::generic_prog_strategy` rendered by `GenericProg::tops`): a
//! program written with top-level bindings, the size-generic alias `Vec<n>`,
//! values read through a size-generic struct's `_` and a top-level table
//! computes its plain twin's values on the native CPU and on the emulated
//! device in every emulator mode (serial 1, serial 2, reversed) — with
//! compile-time evaluation on (the shipped pipeline: `fold_static`, then
//! `rewrite`) and off — each against the interpreter on the plain program.
//! With compile-time evaluation on, every program's operands are laws, so its
//! printed values are embedded: the device unit, when one is left, holds only
//! what the rule keeps at run time.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::CategoryIr;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

const MODES: [[(&str, &str); 2]; 3] = [
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "1")],
];

fn lowered(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(
        po.diagnostics.is_empty(),
        "parse: {:?}\n{src}",
        po.diagnostics
    );
    mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}\n{src}"))
}

/// The shipped pipeline's IR: compile-time evaluation (`on`), then `rewrite`.
fn compiled(src: &str, on: bool) -> CategoryIr {
    let ir = lowered(src);
    let ir = if on {
        mapal_rewrite::fold_static(ir).0
    } else {
        ir
    };
    mapal_rewrite::rewrite(ir).ir
}

fn interp(ir: &CategoryIr, src: &str) -> String {
    let r = run(ir, 2_000_000_000);
    assert!(
        matches!(r.outcome, Outcome::Done(_)),
        "{:?}\n{src}",
        r.outcome
    );
    r.output
}

/// The exact face on `target`; a GPU target with every capable site on the
/// device (`place` off) and the rungs the emulator cannot run off.
fn emission(ir: &CategoryIr, target: &'static str) -> Emission {
    let mut gpu = GpuArms::default();
    for r in ["place", "async", "pdl", "streams", "pinned"] {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "generic",
        target,
        gpu,
        ..EmitOpts::exact()
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// The host program compiled at `-O2` and run (`MAPAL_PAR=2`); its stdout.
fn native(clang: &str, ll: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let (llp, exe) = (dir.path().join("p.ll"), dir.path().join("p"));
    std::fs::write(&llp, ll).unwrap();
    let ok = {
        let _slot = emu::slot();
        Command::new(clang)
            .args(["-O2", "-Wno-override-module"])
            .arg(&llp)
            .arg(emu::rt_lib())
            .args(["-lm", "-o"])
            .arg(&exe)
            .status()
            .unwrap()
            .success()
    };
    assert!(ok, "clang failed");
    let _slot = emu::slot();
    let mut child = Command::new(&exe)
        .env("MAPAL_PAR", "2")
        .env("MAPAL_SPIN", "0")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let start = Instant::now();
    while child.try_wait().unwrap().is_none() {
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "native run timed out"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "{}", out.status);
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Every emulator mode's stdout, against `want`. Whether a device unit ran.
fn emulated(clang: &str, em: &Emission, want: &str, ctx: &str) -> bool {
    let Some(dev) = em.device.as_ref() else {
        return false;
    };
    let built = emu::build_emulated(clang, &em.text, dev);
    for (env, (got, _)) in MODES.iter().zip(built.run_each(&MODES)) {
        assert_eq!(got, want, "{ctx} {env:?}");
    }
    true
}

#[test]
fn generated_top_level_programs_on_native_and_emulated_device() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP: clang not found");
        return;
    };
    let n: usize = std::env::var("MAPAL_TESTGEN_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    let mut runner = TestRunner::deterministic();
    let strat = testgen::generic_prog_strategy();
    let progs: Vec<_> = (0..n)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let [programs, device_off, device_on, defaults] = testgen::par_count(&progs, |_, p| {
        let plain = p.source();
        let (src, u) = p.tops(false);
        let want = interp(&lowered(&plain), &plain);
        let mut dev = [0usize; 2];
        for (k, on) in [false, true].into_iter().enumerate() {
            let ir = compiled(&src, on);
            let ctx = format!("static={on}\n{src}");
            assert_eq!(interp(&ir, &src), want, "{ctx}");
            assert_eq!(
                native(&clang, &emission(&ir, "generic").text),
                want,
                "{ctx}"
            );
            let em = emission(&ir, "rtx-5080");
            dev[k] = usize::from(emulated(&clang, &em, &want, &ctx));
        }
        [1, dev[0], dev[1], u.defaults]
    });
    eprintln!(
        "top-level (backends): {programs} programs, through the emulated device {device_off} \
         without compile-time evaluation and {device_on} with it (3 modes each), {defaults} \
         `_` reads"
    );
    assert!(device_off > 0 && defaults > 0);
}
