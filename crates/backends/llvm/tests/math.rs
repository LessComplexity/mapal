//! Math builtins (ADR-0046): `x -> sqrt`, `x -> sin`, `x -> cos` on the host
//! and the device. The exact face is the reference op for op (or an `Exact`
//! unit: IEEE `sqrt`) — the interpreter's bytes on the host build and the
//! emulated device in every emulator mode, over generated programs reaching
//! every branch (huge arguments, subnormals, ±0, ±∞, NaN, `f64`, fold
//! bodies); the default face takes each side's cheapest admitted
//! realization (the host's contracted references, the RTX 5080's
//! `sqrt.approx` and `cos.approx` forms) within the contract; the OFF arm
//! (`--no-math-hw`) the exact face's choice; a target with no unit (an
//! accelerator) the plain-operation references, same bytes.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::CategoryIr;
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

/// The emission on `target` and `base`'s face, the rungs `off` off; the
/// emulator's rungs off and `place` off, so every capable map is a device
/// cell.
fn emission(ir: &CategoryIr, target: &'static str, base: EmitOpts, off: &[&str]) -> Emission {
    emission_on(ir, "generic", target, base, off)
}

fn emission_on(
    ir: &CategoryIr,
    host: &'static str,
    target: &'static str,
    base: EmitOpts,
    off: &[&str],
) -> Emission {
    let mut gpu = GpuArms::default();
    gpu.on.flow = true;
    for r in [
        "place",
        "async",
        "pdl",
        "streams",
        "pinned",
        "horizontal",
        "calls",
    ]
    .iter()
    .chain(off)
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host,
        target,
        gpu,
        ..base
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// Every printed value within `tol` relative (or the same text: `inf`,
/// `NaN`, `0`).
fn within(got: &str, want: &str, tol: f64) -> bool {
    let (g, w): (Vec<&str>, Vec<&str>) = (
        got.split_whitespace().collect(),
        want.split_whitespace().collect(),
    );
    g.len() == w.len()
        && g.iter().zip(&w).all(|(a, b)| {
            a == b
                || match (a.parse::<f64>(), b.parse::<f64>()) {
                    (Ok(x), Ok(y)) => (x - y).abs() <= tol * x.abs().max(y.abs()).max(1e-30),
                    _ => false,
                }
        })
}

const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_GRID", "2")],
    [("MAPAL_EMU_REVERSE", "1"), ("MAPAL_EMU_GRID", "1")],
];

fn prog(fns: u8, wide: bool, in_fold: bool) -> testgen::MathProg {
    testgen::MathProg {
        n: 512,
        rows: 2,
        mul: 7,
        add: 3,
        m: 101,
        scale: 3,
        fns,
        wide,
        in_fold,
    }
}

fn body<'t>(text: &'t str, def: &str) -> &'t str {
    text.split(def)
        .nth(1)
        .unwrap_or_else(|| panic!("no {def}"))
        .split("\n}\n")
        .next()
        .unwrap()
}

/// The realization each side and face takes (the intents): the exact face
/// the references on both sides (no fast-math flag, no approximate unit),
/// `sqrt` the IEEE instruction (`llvm.sqrt` on the host, `sqrt.rn` on the
/// device); the default face the host's contracted references and the
/// device's measured units — `sqrt.approx.f32` (2^−23.25 within 2^−22) and
/// `cos.approx.f32` as the `cos r` part (2^−21.58 + the remainder's
/// rounding within 2^−21), never `sin.approx` (absolute error); `f64` the
/// IEEE `sqrt` and the contracted references; `--no-math-hw` the exact
/// face's choice on the default face.
#[test]
fn realization_per_side_and_face() {
    let ir = lower(&testgen::math_src(&prog(7, true, false)));
    let d = |f: &str, t: &str| {
        format!(
            "define internal {t} @mapal.{f}.{}(",
            if t == "float" { "f32" } else { "f64" }
        )
    };
    let on = emission(&ir, "rtx-5080", EmitOpts::default(), &[]);
    let dev = on.device.as_deref().expect("device cells");
    for (f, t) in [
        ("sqrt", "float"),
        ("sin", "float"),
        ("cos", "float"),
        ("sin", "double"),
    ] {
        assert!(dev.contains(&d(f, t)), "device {f} {t}");
        assert!(
            on.text.contains(&d(f, t)) || !on.text.contains(&format!("@mapal.{f}.")),
            "host {f} {t}"
        );
    }
    let sq = body(dev, &d("sqrt", "float"));
    assert!(sq.contains("@llvm.nvvm.sqrt.approx.f(float %x)"), "{sq}");
    let sq64 = body(dev, &d("sqrt", "double"));
    assert!(sq64.contains("@llvm.nvvm.sqrt.rn.d(double %x)"), "{sq64}");
    for f in ["sin", "cos"] {
        let b = body(dev, &d(f, "float"));
        assert!(b.contains("@llvm.nvvm.cos.approx.f("), "{f}: {b}");
        assert!(!b.contains("sin.approx"), "{f}");
        assert!(
            b.contains("fmul contract float"),
            "{f}: the sin polynomial contracts"
        );
        assert!(
            b.contains("mul i64") && b.contains("@llvm.ctlz.i64"),
            "{f}: Payne–Hanek"
        );
    }
    let b64 = body(dev, &d("sin", "double"));
    assert!(!b64.contains("approx") && b64.contains("mul i128"));
    let host_sqrt = body(&on.text, &d("sqrt", "float"));
    assert!(
        host_sqrt.contains("@llvm.sqrt.f32(float %x)"),
        "{host_sqrt}"
    );
    for (side, text) in [("host", on.text.as_str()), ("device", dev)] {
        assert!(!text.contains("define internal float @mapal.sqrt.f32(float %x) alwaysinline nounwind willreturn memory(none) {\nentry:\n  %v.1 = bitcast"), "{side}: no digit recurrence where a unit is declared");
    }
    let exact = emission(&ir, "rtx-5080", EmitOpts::exact(), &[]);
    for t in [&exact.text, exact.device.as_deref().unwrap()] {
        for f in ["sin", "cos"] {
            let b = body(t, &d(f, "float"));
            assert!(
                !b.contains("contract") && !b.contains("approx") && !b.contains("@llvm.fma"),
                "{b}"
            );
        }
        assert!(
            body(t, &d("sqrt", "float")).contains("sqrt"),
            "an IEEE unit"
        );
        assert!(!body(t, &d("sqrt", "float")).contains("approx"));
    }
    let off = emission(&ir, "rtx-5080", EmitOpts::default(), &["math-hw"]);
    let offd = off.device.as_deref().unwrap();
    assert!(!offd.contains("approx.f(") || offd.contains("ex2.approx"));
    assert!(body(offd, &d("sqrt", "float")).contains("@llvm.nvvm.sqrt.rn.f(float %x)"));
    // The PTX carries the instructions (where llc is installed).
    if let Ok(o) = std::process::Command::new("llc")
        .args([
            "-O3",
            "-mcpu=sm_120",
            "-mattr=+ptx87",
            "-nvptx-fma-level=0",
            "-o",
            "-",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut c| {
            use std::io::Write;
            c.stdin.take().unwrap().write_all(dev.as_bytes())?;
            c.wait_with_output()
        })
        && o.status.success()
    {
        let ptx = String::from_utf8_lossy(&o.stdout);
        assert!(ptx.contains("sqrt.approx.f32"), "no sqrt.approx in the PTX");
        assert!(ptx.contains("cos.approx.f32"), "no cos.approx in the PTX");
        assert!(
            ptx.contains("mul.hi.u64") && ptx.contains("clz.b64"),
            "Payne–Hanek's integer ops"
        );
    }
}

/// One generated program's checks: the exact face on the host build and the
/// emulated device (generic, the RTX 5080) and on the unit-less host
/// (`plain-fp`: every builtin its plain-operation reference) is the
/// interpreter's bytes; the default face within the contract. `[programs, sqrt, sin, cos, f64, fold, huge-or-inf, nan,
/// subnormal, device-units]`.
fn check_math(clang: &str, p: &testgen::MathProg) -> [u64; 10] {
    let src = testgen::math_src(p);
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let want = rr.output.clone();
    for (host, target) in [
        ("generic", "generic"),
        ("generic", "rtx-5080"),
        ("plain-fp", "plain-fp"),
    ] {
        let ex = emission_on(&ir, host, target, EmitOpts::exact(), &[]);
        let dev = ex.device.as_deref().unwrap_or("");
        let got = emu::run_emulated(clang, &ex.text, dev);
        assert_eq!(got, want, "exact {host}/{target}\n{src}");
    }
    // The unit-less host on its default face: the contracted references.
    let pl = emission_on(&ir, "plain-fp", "plain-fp", EmitOpts::default(), &[]);
    if testgen::math_fns(p).contains(&"sqrt") {
        assert!(
            !pl.text.contains("@llvm.sqrt"),
            "plain-fp: the digit recurrence"
        );
    }
    let got = emu::run_emulated(clang, &pl.text, "");
    assert!(
        within(&got, &want, 1e-4),
        "plain-fp default: {got:?} vs {want:?}\n{src}"
    );
    let on = emission(&ir, "rtx-5080", EmitOpts::default(), &[]);
    let dev = on.device.as_deref().unwrap_or("");
    let got = emu::run_emulated(clang, &on.text, dev);
    assert!(
        within(&got, &want, 1e-4),
        "default: {got:?} vs {want:?}\n{src}"
    );
    let fns = testgen::math_fns(p);
    [
        1,
        u64::from(fns.contains(&"sqrt")),
        u64::from(fns.contains(&"sin")),
        u64::from(fns.contains(&"cos")),
        u64::from(p.wide),
        u64::from(p.in_fold),
        u64::from(want.contains("inf") || src.contains("0000000000000000000000000000.0")),
        u64::from(want.contains("NaN")),
        u64::from(src.contains("0.000000000000000000000000000000000000000000001")),
        u64::from(dev.contains("approx.f(")),
    ]
}

/// The exact faces are the references: generated programs over every
/// branch — the host build and the emulated device print the interpreter's
/// bytes; the default face within the bound (relative 1e-4 on the printed
/// sums, which also carry the order the face frees).
#[test]
fn exact_face_bytes_and_default_bound_on_generated_programs() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::MathProg> = (0..24)
        .map(|_| {
            testgen::math_prog_strategy()
                .new_tree(&mut runner)
                .unwrap()
                .current()
        })
        .collect();
    let counts = testgen::par_check(&progs, |_, p| check_math(&clang, p));
    let mut c = [0u64; 10];
    for k in counts {
        for i in 0..10 {
            c[i] += k[i];
        }
    }
    eprintln!(
        "math generated: programs {} sqrt {} sin {} cos {} f64 {} fold {} huge/inf {} nan {} subnormal {} device-units {}",
        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]
    );
    assert!(c.iter().all(|&x| x > 0), "{c:?}");
}

/// The exact face in every emulator mode (serial turns, grid widths, the
/// reversed block order): the same bytes.
#[test]
fn exact_face_in_every_emulator_mode() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let p = prog(7, true, true);
    let src = testgen::math_src(&p);
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    let ex = emission(&ir, "rtx-5080", EmitOpts::exact(), &[]);
    let b = emu::build_emulated(&clang, &ex.text, ex.device.as_deref().unwrap());
    for mode in MODES {
        let (got, _) = b.run(&mode);
        assert_eq!(got, rr.output, "{mode:?}");
    }
    let on = emission(&ir, "rtx-5080", EmitOpts::default(), &[]);
    let b = emu::build_emulated(&clang, &on.text, on.device.as_deref().unwrap());
    for mode in MODES {
        let (got, _) = b.run(&mode);
        assert!(within(&got, &rr.output, 1e-4), "{mode:?}");
    }
}

/// `L1615` for a non-float operand of every math builtin; the names are
/// reserved.
#[test]
fn surface_rejects_non_floats_and_reserves_the_names() {
    for f in ["sqrt", "sin", "cos"] {
        let src = format!("fn main() {{\n    3 -> {f} -> y;\n    y -> println;\n}}\n");
        let po = mapal_syntax::parse(&src);
        let e = mapal_lower::lower(&src, &po.program).expect_err("an i32 operand");
        assert!(format!("{e:?}").contains("L1615"), "{f}: {e:?}");
        let src = format!(
            "fn {f}(a: f32) -> f32 {{\n    a -> ret;\n}}\nfn main() {{\n    1.0 -> println;\n}}\n"
        );
        let po = mapal_syntax::parse(&src);
        let e = mapal_lower::lower(&src, &po.program).expect_err("reserved");
        assert!(format!("{e:?}").contains("L1009"), "{f}: {e:?}");
    }
}
