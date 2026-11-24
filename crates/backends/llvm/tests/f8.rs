//! fp8 storage (ADR-0043 slice 3): `f8e4m3` / `f8e5m2` on the host and the
//! device. The roundings saturate and are the oracle's bits on every face
//! (the host's integer sequence, the device's `cvt.rn.satfinite` where the
//! part has it — the same sequence where it does not); a widening builds the
//! f32 bits from the byte (exact, the same text on both sides); a sum of
//! products over an fp8 operand is a device tile site (`tile_plan_narrow`)
//! whose kernel loads one byte an element. Generated programs
//! (`testgen::F8Prog`: scales as dataflow — per tensor, per column, a factor
//! inside the fold —, saturation, subnormals, `±inf` inputs, `k × n` and
//! `n × k` panels, an fp8 `a`, loops carrying fp8 arrays) print the
//! interpreter's bytes on the exact face (host; the emulated device in every
//! emulator mode and every OFF arm) and stay within ADR-0040 D2 / ADR-0041 D3
//! (F5's per-operand units) / ADR-0043 D3 on the default face.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::{CategoryIr, NarrowFloat, bound_coef_unit};
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

/// The emission on `target` at `base`'s face, the rungs `off` off and `on`
/// forced on; the emulator's rungs off and `place` off, so every capable map
/// is a device cell (the `narrow.rs` harness).
fn emission(
    ir: &CategoryIr,
    target: &'static str,
    base: EmitOpts,
    off: &[&str],
    on: &[&str],
) -> Emission {
    let mut gpu = GpuArms::default();
    for r in on {
        *gpu.on.rung(r).expect("rung") = true;
    }
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
    .filter(|r| !on.contains(r))
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "generic",
        target,
        gpu,
        ..base
    };
    emit_detailed(ir, &opts).expect("emits")
}

fn num(s: &str) -> f64 {
    match s {
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        _ => s.parse().unwrap_or(f64::NAN),
    }
}

const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_GRID", "2")],
    [("MAPAL_EMU_REVERSE", "1"), ("MAPAL_EMU_GRID", "1")],
];

/// The roundings are facts (`precision_plan.narrow`: saturating forms), a
/// product over an fp8 weight is a narrow tile site whose operand is one
/// byte, and the module defines each conversion it calls: the device's
/// `cvt.rn.satfinite` on a part with `Caps::cvt_f8`, the integer sequence on
/// the host and on a part without it.
#[test]
fn f8_facts_and_realization_per_side() {
    let src = "fn main() {
    1024 -> iota -> ib;
    ib -> map { t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 -> to_f8e4m3 } -> b;
    ib -> map { t -> (t * 5 + 3) % 97 - 48 -> widen_f32 -> v; v * 1000.0 -> to_f8e5m2 } -> e;
    64 -> iota -> ia;
    ia -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 } -> a;
    16 -> iota -> ic;
    ic -> map { j -> (0.0, ia) -> fold { acc, k -> b[k * 16 + j] -> widen_f32 -> w; acc + a[k] * w } } -> y;
    y[3] -> println;
    e[7] -> widen_f64 -> println;
}
";
    let ir = lower(src);
    let f = ir.entry();
    let forms: Vec<_> = ir
        .funcs()
        .flat_map(|(g, _)| ir.precision_plan(g).narrow.into_iter().map(|(_, n)| n))
        .collect();
    assert!(!forms.is_empty() && forms.iter().all(|n| n.saturates));
    assert!(
        forms
            .iter()
            .any(|n| n.fmt == NarrowFloat::F8e4m3 && n.max == 448.0)
    );
    assert!(
        forms
            .iter()
            .any(|n| n.fmt == NarrowFloat::F8e5m2 && n.max == 57344.0)
    );
    let sites = ir.tile_plan_narrow(f);
    let (_, s) = sites.sites.iter().next().expect("an fp8 tile site");
    assert_eq!(s.b.narrow, Some(NarrowFloat::F8e4m3));
    assert_eq!(s.b.elem_bytes(&s.elem), 1);
    let host = emission(&ir, "generic", EmitOpts::exact(), &[], &[]);
    assert!(
        host.text
            .contains("define internal i8 @mapal.to_f8e4m3(float %x)")
    );
    assert!(
        host.text
            .contains("define internal float @mapal.widen_f8e5m2(i8 %b)")
    );
    assert!(!host.text.contains("cvt.rn.satfinite"));
    let dev = emission(&ir, "rtx-5080", EmitOpts::exact(), &[], &[]);
    let d = dev.device.as_deref().expect("a device unit");
    assert!(d.contains("cvt.rn.satfinite.e4m3x2.f32"), "{d}");
    assert!(d.contains("load i8, ptr addrspace(1)") || d.contains("x i8>, ptr addrspace(1)"));
    // the 3090's profile: no fp8 conversion instruction — the sequence.
    let old = emission(&ir, "rtx-3090", EmitOpts::exact(), &[], &[]);
    let d = old.device.as_deref().expect("a device unit");
    assert!(!d.contains("cvt.rn.satfinite") && d.contains("@mapal.to_f8e4m3"));
}

#[derive(Debug, Default)]
struct Cover {
    programs: u64,
    e4m3: u64,
    e5m2: u64,
    fp8_a: u64,
    transposed: u64,
    per_tensor: u64,
    per_column: u64,
    factor: u64,
    members: u64,
    saturated: u64,
    subnormal: u64,
    looped: u64,
    narrow_sites: u64,
    device_f8_tile: u64,
    device_f8_cell: u64,
}

impl std::ops::AddAssign for Cover {
    fn add_assign(&mut self, o: Cover) {
        self.programs += o.programs;
        self.e4m3 += o.e4m3;
        self.e5m2 += o.e5m2;
        self.fp8_a += o.fp8_a;
        self.transposed += o.transposed;
        self.per_tensor += o.per_tensor;
        self.per_column += o.per_column;
        self.factor += o.factor;
        self.members += o.members;
        self.saturated += o.saturated;
        self.subnormal += o.subnormal;
        self.looped += o.looped;
        self.narrow_sites += o.narrow_sites;
        self.device_f8_tile += o.device_f8_tile;
        self.device_f8_cell += o.device_f8_cell;
    }
}

fn check_f8(clang: &str, p: &testgen::F8Prog) -> Cover {
    let (src, (cells, k, mn)) = testgen::f8_src(p);
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let want = rr.output.clone();
    // Exact face: the interpreter's bytes on the host and the emulated device
    // in every emulator mode, and under every OFF arm.
    let host = emission(&ir, "generic", EmitOpts::exact(), &[], &[]);
    let got = emu::run_emulated(clang, &host.text, host.device.as_deref().unwrap_or(""));
    assert_eq!(got, want, "exact host\n{src}");
    let ex = emission(&ir, "rtx-5080", EmitOpts::exact(), &[], &[]);
    let built = emu::build_emulated(clang, &ex.text, ex.device.as_deref().unwrap_or(""));
    for mode in MODES {
        let (got, _) = built.run(&mode);
        assert_eq!(got, want, "exact device {mode:?}\n{src}");
    }
    for (off, on) in [
        (&["narrow-tile"][..], &[][..]),
        (&[], &["async"]),
        (&["vec-cells"], &[]),
    ] {
        let e = emission(&ir, "rtx-5080", EmitOpts::exact(), off, on);
        let got = emu::run_emulated(clang, &e.text, e.device.as_deref().unwrap_or(""));
        assert_eq!(got, want, "exact device off {off:?} on {on:?}\n{src}");
    }
    // the 3090's profile: the integer roundings on the device.
    let old = emission(&ir, "rtx-3090", EmitOpts::exact(), &[], &[]);
    let got = emu::run_emulated(clang, &old.text, old.device.as_deref().unwrap_or(""));
    assert_eq!(got, want, "exact device rtx-3090\n{src}");
    // Default face: each sum of products within 2·coef·M (both sides within
    // coef of the exact sum; F5's per-operand units are at most the shared
    // unit's) plus the epilogue's rounding; a value through a rounding into
    // the format within D3's flips (5 ulps of the format after the loop's
    // roundings); the whole fold over them within its count of such flips.
    let unit = match p.set {
        1 => Some(2f64.powi(-8)),
        2 => Some(2f64.powi(-11)),
        _ => None,
    };
    let coef = bound_coef_unit(k, unit);
    let f8 = if p.fmt == 1 {
        NarrowFloat::F8e5m2
    } else {
        NarrowFloat::F8e4m3
    };
    let form = mapal_ir::narrow::NarrowForm::of(f8);
    let ulp = 2.0 * form.u;
    let wl: Vec<&str> = want.lines().collect();
    let check_default = |got: &str, what: &str| {
        let gl: Vec<&str> = got.lines().collect();
        assert_eq!(gl.len(), wl.len(), "{what}\n{src}");
        for c in 0..cells {
            let (g, w, mag) = (num(gl[2 * c]), num(wl[2 * c]), num(wl[2 * c + 1]));
            let tol = 2.0 * coef * mag + 2f64.powi(-21) * w.abs() + 1e-30;
            assert!(
                (g - w).abs() <= tol,
                "{what} cell {c}: {g} vs {w} (M {mag})\n{src}"
            );
        }
        let n = gl.len();
        for i in 2 * cells..n - 1 {
            let (g, w) = (num(gl[i]), num(wl[i]));
            let tol = 5.0 * ulp * g.abs().max(w.abs()) + 5.0 * form.least();
            assert!(
                gl[i] == wl[i] || (g - w).abs() <= tol,
                "{what} line {i}: {g} vs {w}\n{src}"
            );
        }
        let (g, w) = (num(gl[n - 1]), num(wl[n - 1]));
        let mn = mn as f64;
        assert!(
            (g - w).abs() <= 5.0 * ulp * form.max * mn,
            "{what} sum: {g} vs {w}\n{src}"
        );
    };
    let on = emission(&ir, "rtx-5080", EmitOpts::default(), &[], &[]);
    let dev_on = on.device.as_deref().unwrap_or("").to_string();
    let built = emu::build_emulated(clang, &on.text, &dev_on);
    for mode in MODES {
        let (got, _) = built.run(&mode);
        check_default(&got, &format!("default device {mode:?}"));
    }
    let off = emission(&ir, "rtx-5080", EmitOpts::default(), &["narrow-tile"], &[]);
    let got = emu::run_emulated(clang, &off.text, off.device.as_deref().unwrap_or(""));
    check_default(&got, "default device narrow-tile OFF");
    let sites = ir.tile_plan_narrow(ir.entry());
    let f8_site = sites
        .sites
        .iter()
        .any(|(_, s)| s.b.narrow.is_some_and(|n| n.bytes() == 1));
    // A tile kernel's `k` loop over one-byte loads.
    let tile_kernels = dev_on
        .split("define ptx_kernel void @mapal_tile_")
        .skip(1)
        .filter(|k| k.contains("%kloop") && k.contains("@mapal.widen_f8"))
        .count() as u64;
    let cell_kernels = dev_on
        .split("define ")
        .filter(|d| d.contains("call i8 @mapal.to_f8"))
        .count() as u64;
    let fmax = form.max.to_string();
    Cover {
        programs: 1,
        e4m3: u64::from(p.fmt == 0),
        e5m2: u64::from(p.fmt == 1),
        fp8_a: u64::from(p.fa),
        transposed: u64::from(p.bt),
        per_tensor: u64::from(p.scale == 1),
        per_column: u64::from(p.scale == 2),
        factor: u64::from(p.scale == 3),
        members: u64::from(p.set > 0),
        saturated: wl
            .iter()
            .skip(2 * cells)
            .filter(|l| l.trim_start_matches('-') == fmax)
            .count() as u64,
        subnormal: u64::from(p.range == 2),
        looped: u64::from(p.looped),
        narrow_sites: u64::from(f8_site),
        device_f8_tile: u64::from(tile_kernels > 0),
        device_f8_cell: u64::from(cell_kernels > 0),
    }
}

/// 48 generated programs: exact face bytes (host; emulated device under
/// every emulator mode and OFF arm; the 3090's integer roundings), default
/// face within the bounds; every coverage counter non-zero.
#[test]
fn generated_f8_programs_exact_bytes_and_default_bounds() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::F8Prog> = (0..48)
        .map(|_| {
            testgen::f8_prog_strategy()
                .new_tree(&mut runner)
                .unwrap()
                .current()
        })
        .collect();
    let covers = testgen::par_check(&progs, |_, p| check_f8(&clang, p));
    let mut c = Cover::default();
    for k in covers {
        c += k;
    }
    eprintln!("f8 generated: {c:?}");
    let all = [
        c.programs,
        c.e4m3,
        c.e5m2,
        c.fp8_a,
        c.transposed,
        c.per_tensor,
        c.per_column,
        c.factor,
        c.members,
        c.saturated,
        c.subnormal,
        c.looped,
        c.narrow_sites,
        c.device_f8_tile,
        c.device_f8_cell,
    ];
    assert!(all.iter().all(|&x| x > 0), "{c:?}");
}

/// bugs-properties `ptx-isa` (cross-arch 5fcb401b finding 1: every `_f8w`
/// module for the RTX 4090 went out at `llc`'s `sm_89` default, PTX 7.8, and
/// `ptxas` rejected its `cvt.rn.f16x2.e4m3x2`, which `sm_89` encodes only
/// from PTX 8.1): over generated fp8 programs, on every GPU profile, the
/// device module declares an encoding (`; mapal.ptx`) at least its target's
/// own (`Caps::ptx`) and, where it emits the fp8 conversions, at least
/// theirs (`Caps::cvt_f8`); where `llc` and `ptxas` are installed, every
/// module compiles at that declaration and `ptxas` accepts it for the
/// profile's target — the bundle's preflight, per program.
#[test]
fn generated_f8_programs_declare_their_ptx_isa_per_profile() {
    use mapal_backend_llvm::{Ptx, resolve_profile};
    let have = |t: &str| {
        std::process::Command::new(t)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    };
    let tools = have("llc") && have("ptxas");
    const PROFILES: [&str; 8] = [
        "rtx-3090",
        "a100",
        "rtx-4090",
        "cuda-ada",
        "h100",
        "h100-pcie",
        "rtx-5090",
        "rtx-5080",
    ];
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::F8Prog> = (0..16)
        .map(|_| {
            testgen::f8_prog_strategy()
                .new_tree(&mut runner)
                .unwrap()
                .current()
        })
        .collect();
    // Per program and profile with a device unit: (profile, the module
    // carries an fp8 instruction, assembled by ptxas).
    let per = testgen::par_check(&progs, |z, p| {
        let (src, _) = testgen::f8_src(p);
        let ir = lower(&src);
        let mut seen = Vec::new();
        for (i, name) in PROFILES.iter().enumerate() {
            let g = resolve_profile(name)
                .and_then(|t| t.gpu())
                .expect("a GPU profile");
            let em = emission(&ir, name, EmitOpts::default(), &[], &[]);
            let Some(d) = em.device.as_deref() else {
                continue;
            };
            let dec = Ptx::declared(d).unwrap_or_else(|| panic!("#{z} {name}: no ; mapal.ptx"));
            assert!(
                dec >= g.caps.ptx,
                "#{z} {name}: {dec:?} below the target's {:?}",
                g.caps.ptx
            );
            let hw = d.contains("cvt.rn.f16x2.e4m3x2") || d.contains("cvt.rn.satfinite.e");
            if hw {
                let need = g
                    .caps
                    .cvt_f8
                    .expect("an fp8 instruction on a part that has it");
                assert!(
                    dec.isa >= need.isa && dec.sm >= need.sm,
                    "#{z} {name}: {dec:?} < {need:?}"
                );
            }
            let mut asm = false;
            if tools {
                let dir = tempfile::tempdir().unwrap();
                let (ll, ptx) = (dir.path().join("d.ll"), dir.path().join("d.ptx"));
                std::fs::write(&ll, d).unwrap();
                let llc = std::process::Command::new("llc")
                    .args(["-O3", &format!("-mcpu={}", g.arch), "-nvptx-fma-level=0"])
                    .arg(format!("-mattr={}", dec.mattr()))
                    .arg(&ll)
                    .arg("-o")
                    .arg(&ptx)
                    .output()
                    .unwrap();
                assert!(
                    llc.status.success(),
                    "#{z} {name}: llc\n{}\n{src}",
                    String::from_utf8_lossy(&llc.stderr)
                );
                let st = std::process::Command::new("ptxas")
                    .arg(format!("-arch={}", g.arch))
                    .arg(&ptx)
                    .args(["-o", "/dev/null"])
                    .output()
                    .unwrap();
                assert!(
                    st.status.success(),
                    "#{z} {name}: ptxas rejects the module at {dec:?}\n{}\n{src}",
                    String::from_utf8_lossy(&st.stderr)
                );
                asm = true;
            }
            seen.push((i, hw, asm));
        }
        seen
    });
    let mut hw = [0u64; PROFILES.len()];
    let (mut modules, mut assembled) = (0, 0);
    for (i, h, a) in per.into_iter().flatten() {
        modules += 1;
        hw[i] += u64::from(h);
        assembled += u64::from(a);
    }
    eprintln!(
        "ptx-isa: {modules} modules, {assembled} through ptxas, fp8 instructions per profile {hw:?}"
    );
    assert!(
        modules >= PROFILES.len() as u64,
        "every profile emits device units"
    );
    // The parts with the conversions emit them (sm_89 among them: the
    // encoding the declaration must carry); the 3090 / A100 never do.
    for (i, name) in PROFILES.iter().enumerate() {
        let cvt = resolve_profile(name)
            .and_then(|t| t.gpu())
            .expect("gpu")
            .caps
            .cvt_f8;
        assert_eq!(hw[i] > 0, cvt.is_some(), "{name}: {hw:?}");
    }
    if tools {
        assert_eq!(assembled, modules, "every module through ptxas");
    }
}
