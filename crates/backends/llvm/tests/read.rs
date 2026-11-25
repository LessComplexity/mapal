//! ADR-0045: the `read` builtin on every backend — the interpreter (the
//! oracle), the native CPU build (sequential and the parallel host entry) and
//! the GPU build's host against the emulated device in every emulator mode.
//!
//! - **Golden programs** (`fixtures/read/golden.mapal`): one committed file of
//!   every element type, each consumed by bulk work; every backend prints the
//!   oracle's bytes (`golden.out`).
//! - **Failures**: a missing file and a file of the wrong length exit 103 on
//!   every backend with the same message naming the file (and the byte
//!   counts), after the output written before the read.
//! - **Data movement**: the host reads the file straight into the array (no
//!   copy of it in the host IR); the GPU build uploads it once, outside every
//!   loop — the emulator's upload count and bytes do not grow with the trips.
//! - **Generated programs** (`testgen::read_chain_strategy`): chains whose
//!   closed-form operands come from files equal the same chains computing
//!   them, on every backend; the coverage counters are non-zero.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use mapal_backend_llvm::{EmitOpts, emit_detailed};
use mapal_interp::{Outcome, RunResult, run};
use mapal_ir::{CategoryIr, NarrowFloat, Operation, Ty, Value};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_GRID", "2")],
    [("MAPAL_EMU_REVERSE", "1"), ("MAPAL_EMU_GRID", "1")],
];

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/read")
}

/// Parse and lower `src`, its reads resolving against `dir`; rewritten
/// when `rewrite` (the shipped pipeline), as lowered otherwise.
fn lower_in(src: &str, dir: &Path, rewrite: bool) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program)
        .unwrap_or_else(|d| panic!("lower: {d:?}\n{src}"))
        .with_data_dir(Some(dir.display().to_string()));
    if rewrite {
        let ir = mapal_rewrite::rewrite(ir).ir;
        assert_eq!(ir.data_dir(), Some(dir.display().to_string().as_str()));
        ir
    } else {
        ir
    }
}

fn emission(ir: &CategoryIr, target: &'static str, base: EmitOpts) -> mapal_backend_llvm::Emission {
    let opts = EmitOpts {
        host: "generic",
        target,
        ..base
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// Compile a host module with the runtime and run it: stdout, stderr, code.
/// Time-boxed: a host waiting on a task it has yet to run fails, not hangs.
fn native(clang: &str, ll: &str, env: &[(&str, &str)]) -> (String, String, i32) {
    let dir = tempfile::tempdir().unwrap();
    let (p, exe) = (dir.path().join("p.ll"), dir.path().join("p"));
    std::fs::write(&p, ll).unwrap();
    let ok = {
        let _slot = emu::slot();
        Command::new(clang)
            .args(["-O2", "-Wno-override-module"])
            .arg(&p)
            .arg(emu::rt_lib())
            .arg("-o")
            .arg(&exe)
            .args(["-lm", "-ldl", "-lpthread"])
            .status()
            .unwrap()
            .success()
    };
    assert!(ok, "clang failed");
    let _slot = emu::slot();
    let mut child = Command::new(&exe)
        .env("MAPAL_PAR", "2")
        .env("MAPAL_SPIN", "0")
        .envs(env.iter().copied())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let start = Instant::now();
    while child.try_wait().unwrap().is_none() {
        if start.elapsed() > Duration::from_secs(60) {
            let _ = child.kill();
            panic!("native run timed out");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let out = child.wait_with_output().unwrap();
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

fn done(rr: &RunResult) -> &str {
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{:?}", rr.outcome);
    &rr.output
}

/// The committed fixtures are the documented values, little-endian.
#[test]
fn fixtures_are_the_documented_bytes() {
    let narrow = |n, bits: &[u16]| bits.iter().map(|&b| Value::Narrow(n, b)).collect();
    let want: Vec<(&str, Vec<Value>)> = vec![
        (
            "i32.bin",
            [0, 1, -1, i32::MIN, i32::MAX, 123_456_789, -77, 4096]
                .map(Value::I32)
                .into(),
        ),
        (
            "i64.bin",
            [0, -1, i64::MIN, i64::MAX, 1 << 40, -(1 << 33), 7, 3]
                .map(Value::I64)
                .into(),
        ),
        (
            "u8.bin",
            [0, 1, 127, 128, 254, 255, 9, 200].map(Value::U8).into(),
        ),
        (
            "f32.bin",
            [0.0, -0.0, 1.5, -2.25, f32::MAX, 1.4e-45, 0.1, 1e-3]
                .map(Value::F32)
                .into(),
        ),
        (
            "f64.bin",
            [
                0.0,
                -0.0,
                0.1,
                -2.5,
                1e300,
                5e-324,
                1e-150,
                std::f64::consts::E,
            ]
            .map(Value::F64)
            .into(),
        ),
        (
            "bf16.bin",
            narrow(
                NarrowFloat::Bf16,
                &[
                    0x3f80, 0xc040, 0x7f80, 0x0001, 0x4049, 0x8000, 0xff80, 0x3dcd,
                ],
            ),
        ),
        (
            "f16.bin",
            narrow(
                NarrowFloat::F16,
                &[
                    0x3c00, 0xc200, 0x7bff, 0x0001, 0x4248, 0x7c00, 0x8001, 0x3555,
                ],
            ),
        ),
    ];
    for (file, values) in want {
        let got = std::fs::read(fixtures().join(file)).unwrap();
        assert_eq!(got, mapal_ir::read::encode(&values), "{file}");
        let elem = values[0].ty();
        assert_eq!(mapal_ir::read::decode(&got, &elem), values, "{file}");
    }
}

/// Every element type, every backend: the oracle's bytes (`golden.out`).
#[test]
fn golden_every_element_type_on_every_backend() {
    let dir = fixtures();
    let src = std::fs::read_to_string(dir.join("golden.mapal")).unwrap();
    let want = std::fs::read_to_string(dir.join("golden.out")).unwrap();
    for rewrite in [false, true] {
        let ir = lower_in(&src, &dir, rewrite);
        assert_eq!(
            ir.morphisms()
                .filter(|(_, m)| m.op == Operation::Read)
                .count(),
            7
        );
        let rr = run(&ir, 10_000_000);
        assert_eq!(done(&rr), want, "interp (rewrite={rewrite})");
        let Some(clang) = emu::clang() else {
            continue;
        };
        // Native CPU: the exact face (sequential order) and the default face
        // (the parallel host entry, reads as pinned tasks).
        for (face, base) in [
            ("exact", EmitOpts::exact()),
            ("default", EmitOpts::default()),
        ] {
            let em = emission(&ir, "generic", base);
            assert_eq!(read_calls(&em.text).len(), 7);
            let (out, err, code) = native(&clang, &em.text, &[]);
            assert_eq!(
                (code, out.as_str()),
                (0, want.as_str()),
                "native {face}: {err}"
            );
        }
        // The GPU build (the cost model keeps these eight-element arrays on
        // the host; `device_reads_every_float_format` puts reads on the
        // device), every emulator mode.
        for (out, _, code) in gpu_runs(&clang, &ir) {
            assert_eq!(
                (code, out.as_str()),
                (0, want.as_str()),
                "GPU build (rewrite={rewrite})"
            );
        }
    }
}

/// The GPU build of `ir` run in every emulator mode — or, when no site
/// went to the device (S62: the build is the host's), natively.
fn gpu_runs(clang: &str, ir: &CategoryIr) -> Vec<(String, String, i32)> {
    let em = emission(ir, "rtx-5080", EmitOpts::exact());
    match em.device.as_deref() {
        Some(dev) => {
            let ex = emu::build_emulated(clang, &em.text, dev);
            MODES.iter().map(|m| ex.run_status(m)).collect()
        }
        None => vec![native(clang, &em.text, &[])],
    }
}

/// A program that prints, reads `file` as `[f32; 4]`, prints again.
fn failing(file: &str) -> String {
    format!(
        "fn main() {{\n    \"before\" -> println;\n    \"{file}\" -> read -> samples: [f32; 4];\n    samples -> map {{\n        s -> s * 2.0\n    }} -> doubled;\n    doubled[3] -> println;\n    \"after\" -> println;\n}}\n"
    )
}

/// A missing file and a file of the wrong length: exit 103 and one message
/// naming the resolved file (and the expected and actual byte counts), the
/// same text on every backend; nothing after the read runs.
#[test]
fn failures_exit_103_with_one_message_on_every_backend() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("short.bin"), [0u8; 15]).unwrap();
    let full = |f: &str| dir.path().join(f).display().to_string();
    let cases = [
        (
            "short.bin",
            format!(
                "mapal: read failed: {}: expected 16 bytes, found 15",
                full("short.bin")
            ),
        ),
        (
            "missing.bin",
            format!(
                "mapal: read failed: {}: No such file or directory (os error 2)",
                full("missing.bin")
            ),
        ),
    ];
    for (file, msg) in cases {
        let ir = lower_in(&failing(file), dir.path(), true);
        let rr = run(&ir, 1_000_000);
        assert_eq!(rr.outcome, Outcome::ReadFailed(msg.clone()), "interp");
        assert_eq!(
            mapal_interp::native_observable(&ir, &rr),
            Some((None, mapal_ir::read::EXIT_CODE))
        );
        let Some(clang) = emu::clang() else {
            continue;
        };
        for (face, base) in [
            ("exact", EmitOpts::exact()),
            ("default", EmitOpts::default()),
        ] {
            let em = emission(&ir, "generic", base);
            let (out, err, code) = native(&clang, &em.text, &[]);
            assert_eq!(code, 103, "native {face}");
            assert_eq!(err.trim_end(), msg, "native {face}");
            assert_eq!(out, "before\n", "native {face}: the output before the read");
        }
        for (out, err, code) in gpu_runs(&clang, &ir) {
            assert_eq!(code, 103, "GPU build");
            assert_eq!(err.trim_end(), msg, "GPU build");
            assert_eq!(out, "before\n");
        }
    }
}

/// A trap written before a read is the observable, not the read's failure
/// (the parallel host entry waits for every earlier trap site before the
/// read runs); a trap written after it never fires.
#[test]
fn failure_order_against_traps() {
    let dir = tempfile::tempdir().unwrap();
    let src = |before: bool| {
        let trap = "    2 -> iota -> two;\n    (0, two) -> fold {\n        acc, k -> acc + 7 / (k - 1)\n    } -> boom;\n    boom -> println;\n";
        let read =
            "    \"missing.bin\" -> read -> samples: [f32; 4];\n    samples[0] -> println;\n";
        let (a, b) = if before { (trap, read) } else { (read, trap) };
        format!("fn main() {{\n    \"start\" -> println;\n{a}{b}}}\n")
    };
    for before in [true, false] {
        let ir = lower_in(&src(before), dir.path(), true);
        let rr = run(&ir, 1_000_000);
        let want = if before { 101 } else { 103 };
        assert_eq!(
            mapal_interp::native_observable(&ir, &rr).map(|o| o.1),
            Some(want)
        );
        let Some(clang) = emu::clang() else {
            continue;
        };
        for base in [EmitOpts::exact(), EmitOpts::default()] {
            let em = emission(&ir, "generic", base);
            let (_, err, code) = native(&clang, &em.text, &[]);
            assert_eq!(code, want, "{err}");
        }
    }
}

/// `$MAPAL_DATA_DIR` redirects a relative path; an absolute literal is used
/// as written whatever the environment says.
#[test]
fn data_dir_override_and_absolute_paths() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let (src_dir, alt) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let values = |x: f32| mapal_ir::read::encode(&[Value::F32(x), Value::F32(x + 1.0)]);
    std::fs::write(src_dir.path().join("w.bin"), values(1.0)).unwrap();
    std::fs::write(alt.path().join("w.bin"), values(10.0)).unwrap();
    let rel = "fn main() {\n    \"w.bin\" -> read -> w: [f32; 2];\n    w[1] -> println;\n}\n";
    let ir = lower_in(rel, src_dir.path(), true);
    let em = emission(&ir, "generic", EmitOpts::exact());
    let alt_s = alt.path().display().to_string();
    assert_eq!(native(&clang, &em.text, &[]).0, "2\n");
    assert_eq!(
        native(&clang, &em.text, &[("MAPAL_DATA_DIR", &alt_s)]).0,
        "11\n"
    );
    // An empty value is no override.
    assert_eq!(native(&clang, &em.text, &[("MAPAL_DATA_DIR", "")]).0, "2\n");
    let abs = rel.replace("w.bin", &src_dir.path().join("w.bin").display().to_string());
    let ir = lower_in(&abs, Path::new("/nonexistent"), true);
    let em = emission(&ir, "generic", EmitOpts::exact());
    assert_eq!(
        native(&clang, &em.text, &[("MAPAL_DATA_DIR", &alt_s)]).0,
        "2\n"
    );
    // The oracle resolves the same way (the rule is `mapal_ir::read`'s).
    assert_eq!(done(&run(&ir, 1_000_000)), "2\n");
}

/// Every `mapal_read` / `mapal_read_task` call of a host module: its
/// destination and byte count.
fn read_calls(text: &str) -> Vec<(String, u64)> {
    text.lines()
        .filter(|l| {
            l.contains("call void @mapal_read(") || l.contains("call void @mapal_read_task(")
        })
        .map(|l| {
            let args: Vec<&str> = l[l.find('(').unwrap() + 1..l.rfind(')').unwrap()]
                .split(", ")
                .collect();
            let arg = |k: usize| args[k].split(' ').nth(1).unwrap().to_string();
            (arg(4), arg(5).parse().unwrap())
        })
        .collect()
}

/// A loop over weights read once: `trips` matrix-vector products, the
/// weights stored as `elem` (`f32`, or `bf16` / `f16` widened in the fold).
fn weights_loop(trips: u32, elem: &str) -> String {
    let term = if elem == "f32" {
        "acc + weights[r * 64 + k] * x[k]".to_string()
    } else {
        "weights[r * 64 + k] -> widen_f32 -> w;\n                        acc + w * x[k]".to_string()
    };
    format!(
        "fn main() {{
    \"weights.bin\" -> read -> weights: [{elem}; 4096];
    \"start.bin\" -> read -> start: [f32; 64];
    64 -> iota -> rows;
    mut x: [f32; 64] <- start;
    mut step: i32 <- 0;
    () -> time -> t0;
    loop {{
        (step < {trips}) -> {{
            -true-> {{
                rows -> map {{
                    r -> (0.0, rows) -> fold {{
                        acc, k -> {term}
                    }}
                }} -> y;
                y -> map {{
                    v -> v * 0.5
                }} -> x;
                step + 1 -> step;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
    () -> time -> t1;
    out[0] -> println;
    out[63] -> println;
}}
"
    )
}

/// Data movement (CLAUDE.md evidence rule), for weights stored as `f32`,
/// `bf16` and `f16`: the host IR reads each file into its array and never
/// copies that array; the GPU build's device sites read the weights, which
/// are uploaded once, before the bracketed loop — the emulator's uploads and
/// their bytes are the same at 1 and at 5 trips, in every emulator mode the
/// values the oracle's.
#[test]
fn device_reads_every_float_format_and_weights_move_once() {
    let dir = tempfile::tempdir().unwrap();
    let wv = |i: i32| ((i * 7 + 3) % 13 - 6) as f32 / 64.0;
    let x: Vec<Value> = (0..64)
        .map(|i| Value::F32(((i * 5 + 1) % 11 - 5) as f32 / 8.0))
        .collect();
    std::fs::write(dir.path().join("start.bin"), mapal_ir::read::encode(&x)).unwrap();
    for (elem, width) in [("f32", 4u64), ("bf16", 2), ("f16", 2)] {
        let w: Vec<Value> = (0..4096)
            .map(|i| match elem {
                "f32" => Value::F32(wv(i)),
                "bf16" => Value::Narrow(NarrowFloat::Bf16, mapal_ir::narrow::bf16_from_f32(wv(i))),
                _ => Value::Narrow(
                    NarrowFloat::F16,
                    mapal_ir::narrow::round(NarrowFloat::F16, f64::from(wv(i))),
                ),
            })
            .collect();
        std::fs::write(dir.path().join("weights.bin"), mapal_ir::read::encode(&w)).unwrap();
        let wbytes = 4096 * width;
        let mut seen = Vec::new();
        for trips in [1, 5] {
            let ir = lower_in(&weights_loop(trips, elem), dir.path(), true);
            let rr = run(&ir, 200_000_000);
            let want = done(&rr).to_string();
            // Host: one read into each array, and no host copy of either.
            let host = emission(&ir, "generic", EmitOpts::exact());
            let reads = read_calls(&host.text);
            assert_eq!(reads.len(), 2);
            for ((dst, got), bytes) in reads.iter().zip([wbytes, 256]) {
                assert_eq!(*got, bytes, "{dst}");
                assert!(
                    !host
                        .text
                        .lines()
                        .any(|l| l.contains("@llvm.memcpy") && l.contains(&format!("ptr {dst},"))),
                    "the read array {dst} is copied"
                );
            }
            // GPU: the weights' upload precedes the first clock read; the
            // run's uploads do not depend on the trips.
            let gpu = emission(&ir, "rtx-5080", EmitOpts::exact());
            let dev = gpu
                .device
                .as_deref()
                .expect("the products are device sites");
            let main = gpu
                .text
                .split("define internal void @mapal_main(")
                .nth(1)
                .unwrap();
            let up = main
                .lines()
                .position(|l| {
                    l.contains("@mapal_gpu_h2d") && l.ends_with(&format!("i64 {wbytes})"))
                })
                .expect("the weights' upload");
            let clock = main
                .lines()
                .position(|l| l.contains("@mapal_time_ms()"))
                .unwrap();
            assert!(up < clock, "{elem}: the upload is inside the bracket");
            let Some(clang) = emu::clang() else {
                continue;
            };
            let ex = emu::build_emulated(&clang, &gpu.text, dev);
            for mode in MODES {
                let (out, _) = ex.run(&mode);
                assert_eq!(out, want, "{elem}, {trips} trips, {mode:?}");
            }
            let (_, err) = ex.run(&MODES[0]);
            let totals = emu::totals(&err);
            seen.push((totals[0], totals[3]));
        }
        if let [(n1, b1), (n5, b5)] = seen[..] {
            assert_eq!((n1, b1), (n5, b5), "{elem}: uploads grow with the trips");
            assert!(b1 >= wbytes + 256, "{elem}: bytes in {b1}");
        }
    }
}

/// Generated programs whose closed-form operands come from files
/// (`testgen::read_chain_strategy`): the same values as the chain computing
/// them (the oracle on both), and the oracle's bytes natively and on the
/// emulated device. Coverage: reads, read bytes, device programs.
#[test]
fn generated_read_chains() {
    let n = testgen::jobs().max(4) * 4;
    let mut runner = TestRunner::deterministic();
    let strat = testgen::read_chain_strategy();
    let chains: Vec<testgen::Chain> = (0..n)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let clang = emu::clang();
    let counts = testgen::par_count(&chains, |i, chain| {
        let dir = tempfile::tempdir().unwrap();
        let files = chain.files();
        for (name, bytes) in &files {
            std::fs::write(dir.path().join(name), bytes).unwrap();
        }
        let src = chain.source();
        let ir = lower_in(&src, dir.path(), true);
        let reads = ir
            .morphisms()
            .filter(|(_, m)| m.op == Operation::Read)
            .count();
        assert_eq!(reads, files.len(), "#{i}");
        let rr = run(&ir, 2_000_000_000);
        let want = done(&rr).to_string();
        let computed = testgen::Chain {
            read: 0,
            ..chain.clone()
        };
        let law = lower_in(&computed.source(), dir.path(), true);
        assert_eq!(
            done(&run(&law, 2_000_000_000)),
            want,
            "#{i}: read vs computed\n{src}"
        );
        let Some(clang) = &clang else {
            return [reads, 0, 0];
        };
        let host = emission(&ir, "generic", EmitOpts::exact());
        let (out, err, code) = native(clang, &host.text, &[]);
        assert_eq!(
            (code, out.as_str()),
            (0, want.as_str()),
            "#{i} native: {err}\n{src}"
        );
        let gpu = emission(&ir, "rtx-5080", EmitOpts::exact());
        let mut device = 0;
        if let Some(dev) = gpu.device.as_deref() {
            device = 1;
            let ex = emu::build_emulated(clang, &gpu.text, dev);
            let (out, _) = ex.run(&MODES[i % MODES.len()]);
            assert_eq!(out, want, "#{i} emulated\n{src}");
        }
        let bytes: usize = files.iter().map(|(_, b)| b.len()).sum();
        [reads, bytes, device]
    });
    eprintln!(
        "read chains: {n} programs, {} reads, {} bytes read, {} with device sites",
        counts[0], counts[1], counts[2]
    );
    assert!(counts[0] > 0, "no generated program reads");
    if emu::clang().is_some() {
        assert!(counts[2] > 0, "no generated read program has a device site");
    }
}

/// Every readable element type is one the IR validates (`Ty` coverage of
/// `mapal_ir::read::readable` against the golden program's types).
#[test]
fn readable_types_match_the_golden_program() {
    let src = std::fs::read_to_string(fixtures().join("golden.mapal")).unwrap();
    let ir = lower_in(&src, &fixtures(), false);
    let mut elems: Vec<String> = ir
        .morphisms()
        .filter(|(_, m)| m.op == Operation::Read)
        .map(
            |(_, m)| match ir.object(m.target).unwrap().ty.component_ty(1) {
                Some(Ty::Array { elem, .. }) => mapal_ir::ty_label(elem),
                other => panic!("{other:?}"),
            },
        )
        .collect();
    elems.sort();
    assert_eq!(elems, ["bf16", "f16", "f32", "f64", "i32", "i64", "u8"]);
}

/// A read, then a loop calling a function that owns a loop, then a print:
/// the print's wait is placed by `Checkpoint::at`, never at the read's token
/// (it reads no bytes) ahead of the pinned call it needs — the native host
/// used to wait there for a task only it could run (hang).
#[test]
fn a_print_after_a_read_waits_after_the_pinned_call() {
    let src = "fn inner(x: [f32; 512], n: i32) -> [f32; 512] {
    mut y: [f32; 512] <- x;
    mut k: i32 <- 0;
    loop {
        (k < n) -> {
            -true-> {
                512 -> iota -> map {
                    i -> y[i] * 0.5 + y[(i + 1) % 512] * 0.25
                } -> y;
                k + 1 -> k;
                -> loop;
            }
            -false-> y -> ret;
        }
    }
}

fn main() {
    \"f32.bin\" -> read -> w: [f32; 8];
    512 -> iota -> map {
        i -> w[i % 4]
    } -> x0;
    mut x: [f32; 512] <- x0;
    mut t: i32 <- 0;
    loop {
        (t < 4) -> {
            -true-> {
                (x, t + 1) -> inner -> x;
                t + 1 -> t;
                -> loop;
            }
            -false-> x -> done;
        }
    }
    done[0] -> println;
    done[511] -> println;
}
";
    let dir = fixtures();
    for rewrite in [false, true] {
        let ir = lower_in(src, &dir, rewrite);
        let rr = run(&ir, 100_000_000);
        let want = done(&rr).to_string();
        let Some(clang) = emu::clang() else {
            continue;
        };
        for (face, base) in [
            ("exact", EmitOpts::exact()),
            ("default", EmitOpts::default()),
        ] {
            let em = emission(&ir, "generic", base);
            let (out, err, code) = native(&clang, &em.text, &[]);
            assert_eq!(
                (code, out.as_str()),
                (0, want.as_str()),
                "native {face} (rewrite={rewrite}): {err}"
            );
        }
        for (out, _, code) in gpu_runs(&clang, &ir) {
            assert_eq!((code, out.as_str()), (0, want.as_str()), "GPU build");
        }
    }
}
