//! local-rtx-5080: `native`'s per-host dispatch fact and the width rule it
//! drives (`DispatchCost::lanes`, `FnEmit::task_width`).
//!
//! ONE `#[test]` on purpose: the fact is read once per process (`native`'s
//! `OnceLock`), so the binary pins `MAPAL_NATIVE_DISPATCH` before anything
//! resolves `native`, and a second test thread could race that. The forced
//! economics — a 800 ns wave over 8 lanes (100 ns a lane), 1 flop and 1 byte a
//! nanosecond — make the generated programs' small sites land on every answer
//! the rule has: inline, a partial width, the whole pool.
#![cfg(target_os = "linux")]

use std::process::Command;

use mapal_backend_llvm::{EmitOpts, emit_with_opts};
use mapal_interp::run;
use mapal_ir::CategoryIr;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

const LANES: u64 = 8;

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

/// `(n, kind, width)` of every `mapal_par_task` registration.
fn registrations(ll: &str) -> Vec<(u64, u64, u64)> {
    ll.lines()
        .filter_map(|l| l.split("call void @mapal_par_task(").nth(1))
        .map(|args| {
            let v: Vec<&str> = args.trim_end_matches(')').split(", ").collect();
            let num = |i: usize| v[i].rsplit(' ').next().unwrap().parse::<u64>().unwrap();
            (num(4), num(2), num(8))
        })
        .collect()
}

/// The rule's invariants on one emission: a width is a partial pool (`2..LANES`)
/// or `0`, never `1` (one lane is `kind = 0`), and an inline task has none.
fn check_registrations(ll: &str, tag: &str) {
    for (n, kind, width) in registrations(ll) {
        assert!(width != 1 && width < LANES, "{tag}: n={n} width={width}");
        assert!(
            kind == 1 || width == 0,
            "{tag}: inline task n={n} has width {width}"
        );
    }
}

/// Compile with the debug runtime and run on a `LANES` pool.
fn run_native(clang: &str, ll: &str, tag: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let (src, exe) = (dir.path().join("p.ll"), dir.path().join("p"));
    std::fs::write(&src, ll).unwrap();
    let ok = Command::new(clang)
        .args(["-O2", "-Wno-override-module"])
        .arg(&src)
        .arg(emu::rt_lib())
        .args(["-lpthread", "-ldl", "-lm", "-o"])
        .arg(&exe)
        .status()
        .is_ok_and(|s| s.success());
    assert!(ok, "{tag}: clang failed");
    let out = Command::new(&exe)
        .env("MAPAL_PAR", LANES.to_string())
        .output()
        .unwrap();
    assert!(out.status.success(), "{tag}: exit {:?}", out.status);
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn saxpy(n: u64) -> String {
    format!(
        "fn main() {{
    {n} -> iota -> ix;
    ix -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen<f32> }} -> x;
    ix -> map {{ t -> (t * 5 + 3) % 31 - 15 -> widen<f32> }} -> y0;
    (x, y0) -> zip -> map {{ p -> 2.5 * p.0 + p.1 }} -> y;
    y[0] -> println;
    y[{last}] -> println;
}}
",
        last = n - 1
    )
}

#[test]
fn width_rule_on_native() {
    // SAFETY: the only test in this binary, before any thread reads the env.
    unsafe { std::env::set_var("MAPAL_NATIVE_DISPATCH", format!("800,{LANES},1,1")) };
    let Some(clang) = emu::clang() else {
        return;
    };

    // The three answers on one elementwise site (a few bytes a cell at 1
    // byte/ns, 100 ns a lane): 16 cells inline, 256 cells a partial width
    // (sqrt of ~2 us / 100 ns), 65536 cells the whole pool.
    for (n, want) in [(16u64, "inline"), (256, "partial"), (65_536, "pool")] {
        let ir = lower(&saxpy(n));
        let ll = emit_with_opts(&ir, &EmitOpts::exact()).unwrap();
        check_registrations(&ll, &format!("saxpy_{n}"));
        let regs = registrations(&ll);
        assert!(
            regs.iter().any(|&(m, k, w)| m == n
                && match want {
                    "inline" => k == 0,
                    "partial" => k == 1 && w > 1,
                    _ => k == 1 && w == 0,
                }),
            "saxpy_{n}: no {want} registration among {regs:?}"
        );
        let rr = run(&ir, 400_000_000);
        assert_eq!(
            run_native(&clang, &ll, &format!("saxpy_{n}")),
            rr.output,
            "saxpy_{n}"
        );
    }

    // Generated chains (maps, cells, tile sites, windows, row and whole folds,
    // loops): the invariants hold and the values are the interpreter's, bit for
    // bit on the exact face, whatever width each site was given.
    let mut runner = TestRunner::deterministic();
    let strat = testgen::chain_strategy();
    let (mut partial, mut inline) = (0, 0);
    for z in 0..48 {
        let src = strat.new_tree(&mut runner).unwrap().current().source();
        let ir = lower(&src);
        let ll = emit_with_opts(&ir, &EmitOpts::exact()).unwrap();
        let tag = format!("chain#{z}");
        check_registrations(&ll, &tag);
        let regs = registrations(&ll);
        partial += regs.iter().filter(|r| r.1 == 1 && r.2 > 1).count();
        inline += regs.iter().filter(|r| r.1 == 0 && r.0 > 1).count();
        let rr = run(&ir, 400_000_000);
        assert_eq!(run_native(&clang, &ll, &tag), rr.output, "{tag}\n{src}");
    }
    assert!(
        partial > 0 && inline > 0,
        "partial {partial}, inline {inline}"
    );
    eprintln!("width_rule_on_native: {partial} partial widths, {inline} inlined splits");
}
