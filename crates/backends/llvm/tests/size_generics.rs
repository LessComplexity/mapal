//! ADR-0047 through the backends: a size-generic program computes its
//! hand-specialized twin's values on the native CPU and on the emulated
//! device in every emulator mode (serial 1, serial 2, reversed), each against
//! the interpreter; with one instance per generic function the host text and
//! the device unit are the twin's byte for byte (D5); over generated programs
//! (`testgen::generic_prog_strategy`) the same values. No backend changed:
//! monomorphization hands them ordinary fixed-size functions.

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

/// The user's projection at two shapes and a reduction at both.
const PROJECT: &str =
    "fn project<rows, columns>(weight: [bf16; rows * columns], x: [f32; columns]) -> [f32; rows] {
    columns -> iota -> column_indices;
    rows -> iota -> map {
        row -> (0.0, column_indices) -> fold {
            sum, column -> weight[row * columns + column] -> widen_f32 -> w;
            sum + w * x[column]
        }
    } -> ret;
}

fn checksum<n>(y: [f32; n]) -> f32 {
    (0.0, y) -> fold {
        s, v -> s + v
    } -> ret;
}

fn main() {
    8192 -> iota -> map {
        i -> i % 7 - 3 -> widen_f32 -> * 0.25 -> to_bf16
    } -> w;
    128 -> iota -> map {
        i -> i % 5 -> widen_f32 -> + 1.0
    } -> x;
    64 -> iota -> map {
        i -> i % 3 -> widen_f32
    } -> x2;
    (w, x) -> project -> y;
    (w, x2) -> project -> z;
    y -> checksum -> cy;
    z -> checksum -> cz;
    cy -> println;
    cz -> println;
    y[63] -> println;
    z[127] -> println;
}
";

const PROJECT_HAND: &str = "fn project_64_128(weight: [bf16; 8192], x: [f32; 128]) -> [f32; 64] {
    128 -> iota -> column_indices;
    64 -> iota -> map {
        row -> (0.0, column_indices) -> fold {
            sum, column -> weight[row * 128 + column] -> widen_f32 -> w;
            sum + w * x[column]
        }
    } -> ret;
}

fn project_128_64(weight: [bf16; 8192], x: [f32; 64]) -> [f32; 128] {
    64 -> iota -> column_indices;
    128 -> iota -> map {
        row -> (0.0, column_indices) -> fold {
            sum, column -> weight[row * 64 + column] -> widen_f32 -> w;
            sum + w * x[column]
        }
    } -> ret;
}

fn checksum_64(y: [f32; 64]) -> f32 {
    (0.0, y) -> fold {
        s, v -> s + v
    } -> ret;
}

fn checksum_128(y: [f32; 128]) -> f32 {
    (0.0, y) -> fold {
        s, v -> s + v
    } -> ret;
}

fn main() {
    8192 -> iota -> map {
        i -> i % 7 - 3 -> widen_f32 -> * 0.25 -> to_bf16
    } -> w;
    128 -> iota -> map {
        i -> i % 5 -> widen_f32 -> + 1.0
    } -> x;
    64 -> iota -> map {
        i -> i % 3 -> widen_f32
    } -> x2;
    (w, x) -> project_64_128 -> y;
    (w, x2) -> project_128_64 -> z;
    y -> checksum_64 -> cy;
    z -> checksum_128 -> cz;
    cy -> println;
    cz -> println;
    y[63] -> println;
    z[127] -> println;
}
";

/// One instance per generic: a matrix–vector product over a square
/// `[f32; n * n]` (the bisection inference) and its twin.
const SQUARE: &str = "fn apply<n>(m: [f32; n * n], v: [f32; n]) -> [f32; n] {
    n -> iota -> cols;
    n -> iota -> map {
        i -> (0.0, cols) -> fold {
            s, j -> s + m[i * n + j] * v[j]
        }
    } -> ret;
}

fn main() {
    9216 -> iota -> map {
        i -> (i * 5 + 3) % 17 - 8 -> widen_f32 -> * 0.125
    } -> m;
    96 -> iota -> map {
        i -> i % 4 -> widen_f32
    } -> v;
    (m, v) -> apply -> y;
    y[0] -> println;
    y[95] -> println;
}
";

const SQUARE_HAND: &str = "fn apply_96(m: [f32; 9216], v: [f32; 96]) -> [f32; 96] {
    96 -> iota -> cols;
    96 -> iota -> map {
        i -> (0.0, cols) -> fold {
            s, j -> s + m[i * 96 + j] * v[j]
        }
    } -> ret;
}

fn main() {
    9216 -> iota -> map {
        i -> (i * 5 + 3) % 17 - 8 -> widen_f32 -> * 0.125
    } -> m;
    96 -> iota -> map {
        i -> i % 4 -> widen_f32
    } -> v;
    (m, v) -> apply_96 -> y;
    y[0] -> println;
    y[95] -> println;
}
";

#[test]
fn generic_programs_compute_their_twins_values_everywhere() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP: clang not found");
        return;
    };
    let mut device = 0;
    for (src, hand) in [(PROJECT, PROJECT_HAND), (SQUARE, SQUARE_HAND)] {
        let (g, h) = (lower(src), lower(hand));
        let want = interp(&h, hand);
        assert_eq!(interp(&g, src), want);
        assert_eq!(native(&clang, &emission(&g, "generic").text), want, "{src}");
        for target in ["rtx-5080", "h100"] {
            device += usize::from(emulated(&clang, &emission(&g, target), &want, target));
        }
    }
    assert_eq!(device, 4, "every case offloads on both targets");
}

#[test]
fn one_instance_per_generic_emits_the_twins_text() {
    // D5: with one instance per generic function (relative source order is
    // the twin's), the host text and the device unit are the twin's.
    let (g, h) = (lower(SQUARE), lower(SQUARE_HAND));
    for target in ["generic", "rtx-5080", "h100", "rtx-3090", "rtx-4090"] {
        let (eg, eh) = (emission(&g, target), emission(&h, target));
        assert_eq!(eg.text, eh.text, "{target} host");
        assert_eq!(eg.device, eh.device, "{target} device");
        if target != "generic" {
            assert!(eg.device.is_some(), "{target} offloads");
        }
    }
}

#[test]
fn generated_generic_programs_on_native_and_emulated_device() {
    let Some(clang) = emu::clang() else {
        eprintln!("SKIP: clang not found");
        return;
    };
    let n: usize = std::env::var("MAPAL_TESTGEN_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let mut runner = TestRunner::deterministic();
    let strat = testgen::generic_prog_strategy();
    let progs: Vec<_> = (0..n)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let [programs, multi, device, same_text] = testgen::par_count(&progs, |_, p| {
        let (src, hand) = (p.source(), p.hand());
        let (g, h) = (lower(&src), lower(&hand));
        let want = interp(&h, &hand);
        assert_eq!(interp(&g, &src), want, "\n{src}");
        assert_eq!(
            native(&clang, &emission(&g, "generic").text),
            want,
            "\n{src}"
        );
        let em = emission(&g, "rtx-5080");
        let dev = emulated(&clang, &em, &want, &src);
        // One instance per helper: the twin's host text and device unit.
        let single = p.instances().iter().all(|(_, vs)| vs.len() == 1);
        if single {
            let eh = emission(&h, "rtx-5080");
            assert_eq!(em.text, eh.text, "host text\n{src}");
            assert_eq!(em.device, eh.device, "device unit\n{src}");
        }
        [
            1,
            usize::from(p.multi_size() > 0),
            usize::from(dev),
            usize::from(single),
        ]
    });
    eprintln!(
        "size generics (backends): {programs} programs, {multi} with a helper at ≥ 2 sizes, \
         {device} through the emulated device (3 modes each), {same_text} with the twin's text"
    );
    assert!(multi > 0 && device > 0 && same_text > 0);
}
