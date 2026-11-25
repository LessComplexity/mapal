//! model-ready `placement`: a decode step's work stays where its data lives.
//!
//! - **Views are their values** (rewriter): a `Proj` of a pack — an inlined
//!   call's argument tuple, a struct's field — forwards its feeder inside a
//!   loop too, and a loop state slot that is itself a struct of arrays is
//!   carried as its leaves (`mut kv: Cache`), so every fact sees the arrays.
//! - **Facts** (`mapal-ir`): sibling views of one input slot share a
//!   condition's narrowing (a rotation's partner `i ± w/2` under `i % w <
//!   w/2` is in bounds); a selecting fold's index component is ranged by its
//!   seed and items (an argmax token gathers an embedding row in bounds); a
//!   piecewise map's interval through a capture the site passes as a
//!   constant (an inlined layer's index) is affine in the other one; a
//!   nested fold's read of a captured array has its stride along the fold.
//! - **Rules** (backend): a cell a loop runs every trip is placed by the
//!   bytes it moves (a made-once operand is uploaded once); a chain of
//!   in-place range maps over a carried state shares the carry's buffer (a
//!   trip's appends to one cache); a nest cell whose fold reads an operand
//!   at unit stride takes a warp a cell.
//! - **Generated programs** (`testgen::decode_strategy`): decode-like loops
//!   — layers with a literal index, rotated keys, appends to a struct cache,
//!   attention over the filled rows, an argmax token's embedding row —
//!   against the interpreter in every emulator mode, on the CPU and with a
//!   placement rung off; the loop moves nothing to the device per trip.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::{CategoryIr, Ty};
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

fn interp(ir: &CategoryIr) -> String {
    let rr = run(ir, 2_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{:?}", rr.outcome);
    rr.output
}

/// The GPU emission on the rtx-5080 with the emulator's rungs off, and
/// `off` off too.
fn device_emission(ir: &CategoryIr, off: &[&str]) -> Emission {
    let mut gpu = GpuArms::default();
    for r in ["async", "pdl", "streams", "pinned", "horizontal", "calls"]
        .iter()
        .chain(off)
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "generic",
        target: "rtx-5080",
        gpu,
        ..EmitOpts::default()
    };
    emit_detailed(ir, &opts).expect("emits")
}

const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "1")],
];

/// The entry's loop movement lines (`; mapal.move loop=…`).
fn moves(em: &Emission) -> Vec<&str> {
    em.text
        .lines()
        .filter(|l| l.starts_with("; mapal.move loop=") && !l.contains("carried=-"))
        .collect()
}

fn check(clang: &str, i: usize, g: &testgen::DecodeProg) -> Vec<(&'static str, u64)> {
    let src = testgen::decode_prog_src(g);
    let ir = lower(&src);
    let want = interp(&ir);
    let mut c = vec![("programs", 1)];
    // The struct state is carried as its leaves: no loop merge of `main`
    // holds a product in a slot.
    let f = ir.entry();
    let flat = ir.loop_structure(f).iter().all(|scc| {
        scc.merges
            .iter()
            .all(|&m| match ir.object(m).map(|x| &x.ty) {
                Some(Ty::Tuple(parts)) => !parts
                    .iter()
                    .any(|t| matches!(t, Ty::Tuple(_) | Ty::Struct { .. })),
                _ => true,
            })
    });
    assert!(flat, "a struct slot stays in the loop state\n{src}");
    c.push(("flat-states", 1));
    let em = device_emission(&ir, &[]);
    let dev = em.device.as_deref().unwrap_or("");
    let mv = moves(&em);
    // Every trip's work on the device: nothing crosses up per trip, and
    // the cache's leaves are carried there.
    if mv.iter().any(|l| l.contains(":device")) {
        c.push(("device-carries", 1));
    }
    if mv.iter().all(|l| l.contains(" h2d=0 ")) {
        c.push(("trip-h2d-zero", 1));
        if g.rot {
            c.push(("rotated-on-device", 1));
        }
    }
    if dev.contains(" range=") {
        c.push(("range-cells", 1));
        if g.layers == 2 {
            c.push(("range-chains", 1));
        }
    }
    if dev.contains(" lanes=32") {
        c.push(("lane-cells", 1));
    }
    if dev.contains("kind=cell uniform=1 nested=0") {
        c.push(("gather-cells", 1));
    }
    let ex = emu::build_emulated(clang, &em.text, dev);
    for (mode, (got, _)) in MODES.iter().zip(ex.run_each(&MODES)) {
        assert_eq!(got, want, "{mode:?}\n{src}");
    }
    let cpu = emu::run_emulated(
        clang,
        &emit_detailed(&ir, &EmitOpts::default()).unwrap().text,
        "",
    );
    assert_eq!(cpu, want, "cpu\n{src}");
    if i % 3 == 0 {
        let arm = ["range-cell", "devmap", "cell-lanes"][(i / 3) % 3];
        let off = device_emission(&ir, &[arm]);
        let got = emu::run_emulated(clang, &off.text, off.device.as_deref().unwrap_or(""));
        assert_eq!(got, want, "{arm} off\n{src}");
        c.push(("off-arm-runs", 1));
    }
    c
}

/// Generated decode-like loops (`testgen::decode_strategy`), each on the
/// emulator in every mode, the CPU build and with a rung off — against the
/// interpreter. Every path is reached.
#[test]
fn placement_on_generated_programs() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::DecodeProg> = std::iter::repeat_with(|| {
        testgen::decode_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current()
    })
    .take(24)
    .collect();
    let results = testgen::par_check(&progs, |i, g| check(&clang, i, g));
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    for r in results.into_iter().flatten() {
        *c.entry(r.0).or_default() += r.1;
    }
    eprintln!("placement generated: {c:?}");
    for key in [
        "programs",
        "flat-states",
        "device-carries",
        "trip-h2d-zero",
        "rotated-on-device",
        "range-cells",
        "range-chains",
        "lane-cells",
        "gather-cells",
        "off-arm-runs",
    ] {
        assert!(c.get(key).copied().unwrap_or(0) > 0, "{key}: {c:?}");
    }
}
