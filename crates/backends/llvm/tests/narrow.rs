//! half-precision (ADR-0043): `bf16` / `f16` storage on the host and the
//! device. The conversions are the oracle's bits on every face (the host's
//! integer sequence for `bf16` and `fptrunc` for `f16`, the device's
//! `cvt.rn`); a sum of products over a narrow operand is a device tile site
//! (`tile_plan_narrow`) whose kernel loads the 2-byte format and widens it;
//! generated programs print the interpreter's bytes on the exact face (host,
//! emulated device in every emulator mode) and stay within ADR-0040 D2 /
//! ADR-0043 D3 on the default face.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::{CategoryIr, NarrowFloat, Operation};
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

/// The emission on `target` at `base`'s face, the rungs `off` off; the
/// emulator's rungs off and `place` off, so every capable map is a device
/// cell (the `exp.rs` harness).
fn emission(ir: &CategoryIr, target: &'static str, base: EmitOpts, off: &[&str]) -> Emission {
    emission_on(ir, target, base, off, &[])
}

/// [`emission`] with the rungs `on` forced on.
fn emission_on(
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

/// Every printed value within `tol` relative (or the same text).
fn within(got: &str, want: &str, tol: f64) -> bool {
    let (g, w): (Vec<&str>, Vec<&str>) = (
        got.split_whitespace().collect(),
        want.split_whitespace().collect(),
    );
    g.len() == w.len() && g.iter().zip(&w).all(|(a, b)| close(a, b, tol))
}

fn close(a: &str, b: &str, tol: f64) -> bool {
    a == b
        || match (a.parse::<f64>(), b.parse::<f64>()) {
            (Ok(x), Ok(y)) => (x - y).abs() <= tol * x.abs().max(y.abs()).max(1e-30),
            _ => false,
        }
}

const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_GRID", "2")],
    [("MAPAL_EMU_REVERSE", "1"), ("MAPAL_EMU_GRID", "1")],
];

const GEMV: &str = "fn main() {
    64 -> iota -> tk;
    6144 -> iota -> tw;
    96 -> iota -> tn;
    tk -> map { e -> (e * 7 + 3) % 31 - 15 -> widen<f32> -> v; v / 29.0 } -> x;
    tw -> map { e -> (e % 103 * 11 + 57) % 103 - 51 -> widen<f32> -> v; v / 1856.0 -> to<bf16> } -> w;
    96 -> iota -> tb;
    tb -> map { e -> e -> widen<f32> -> v; v * 1000.5 -> to<f16> } -> h;
    tw -> map { e -> (e % 97 * 5 + 3) % 97 - 48 -> widen<f32> -> v; v / 1856.0 } -> g;
    tn -> map { j -> (0.0, tk) -> fold { acc, k -> w[k * 96 + j] -> widen<f32> -> b; acc + x[k] * b } -> s; h[j] -> widen<f32> -> hv; s + hv } -> y;
    tn -> map { j -> (0.0, tk) -> fold { acc, k -> acc + x[k] * g[k * 96 + j] } } -> f;
    y[0] -> println;
    y[95] -> println;
    f[3] -> println;
}
";

/// D8: `tile_plan_narrow` holds the sum of products over the narrow
/// operand (with its epilogue), its read carries the format, the f32 sum of
/// products is in `tile_plan` and not there; `precision_plan` records each
/// rounding's form and the narrow site's bound (the f32 site's).
#[test]
fn narrow_sites_and_roundings_are_facts() {
    let src = GEMV.replace(" -> s; h[j] -> widen<f32> -> hv; s + hv", "");
    let ir = lower(&src);
    let f = ir.entry();
    let narrow = ir.tile_plan_narrow(f);
    let strict = ir.tile_plan(f);
    assert_eq!(narrow.sites.len(), 1, "{src}");
    assert_eq!(strict.sites.len(), 1);
    let (m, s) = narrow.sites.iter().next().unwrap();
    assert!(!strict.sites.contains_key(m), "disjoint plans");
    assert_eq!(s.b.narrow, Some(NarrowFloat::Bf16));
    assert_eq!(s.a.narrow, None);
    assert_eq!((s.rows, s.c, s.k), (1, 96, 64));
    assert_eq!(s.b.elem_bytes(&s.elem), 2);
    let pp = ir.precision_plan(f);
    let roundings: Vec<_> = ir
        .funcs()
        .flat_map(|(g, _)| ir.precision_plan(g).narrow.into_iter().map(|(_, n)| n))
        .collect();
    assert_eq!(roundings.len(), 2, "to<bf16> and to<f16>");
    assert!(roundings.iter().any(|n| n.u == 2f64.powi(-8)));
    assert!(
        roundings
            .iter()
            .any(|n| n.u == 2f64.powi(-11) && n.max == 65504.0)
    );
    assert_eq!(
        pp.bound.get(m).copied(),
        Some(mapal_ir::gamma(65, 2f64.powi(-24)))
    );
    // With its epilogue (`+ h[j]`), still a narrow site — carrying the post.
    let ir = lower(GEMV);
    let narrow = ir.tile_plan_narrow(ir.entry());
    assert_eq!(narrow.sites.len(), 1);
    assert!(narrow.sites.iter().next().unwrap().1.post.is_some());
    assert!(ir.funcs().any(|(_, d)| {
        d.morphisms
            .iter()
            .any(|&m| ir.morphism(m).unwrap().op == Operation::Narrow)
    }));
}

/// D5 / D6 / D8 intents: the host defines `to<bf16>` as the integer
/// sequence and `to<f16>` as `fptrunc`; the device both as `fptrunc`
/// (`cvt.rn`); the narrow site's kernel loads `bfloat` and widens it (one
/// thread a cell), its pointer an allocation's — aligned as the kernel's
/// other operands (regressions: `align 2` kept the bf16 KV cache's scores
/// kernel to 2-byte loads); `--gpu-off=narrow-tile`: no tile kernel reads
/// the format.
#[test]
fn realization_per_side() {
    let ir = lower(GEMV);
    let on = emission(&ir, "rtx-5080", EmitOpts::default(), &[]);
    let host = &on.text;
    let bf = "define internal bfloat @mapal.to_bf16(float %x)";
    let hf = "define internal half @mapal.to_f16(float %x)";
    let body = |t: &str, d: &str| -> String {
        t.split(d)
            .nth(1)
            .unwrap_or_else(|| panic!("{d} missing"))
            .split("\n}\n")
            .next()
            .unwrap()
            .to_string()
    };
    let dev = on.device.as_deref().expect("device");
    if host.contains("@mapal.to_bf16(") {
        assert!(body(host, bf).contains("add i32 %b, %k") && !body(host, bf).contains("fptrunc"));
    }
    assert!(body(dev, bf).contains("fptrunc float %x to bfloat"));
    assert!(body(dev, hf).contains("fptrunc float %x to half"));
    let tile = dev
        .split("define ptx_kernel void @mapal_tile_")
        .skip(1)
        .find(|k| k.contains("load bfloat"))
        .expect("a tile kernel loads the narrow operand");
    assert!(tile.contains("fpext bfloat"), "{tile}");
    let align = |p: &str| {
        let head = tile.lines().next().unwrap_or("");
        let i = head
            .find(&format!(" {p},"))
            .or_else(|| head.find(&format!(" {p})")));
        i.and_then(|i| head[..i].rsplit(" align ").next().map(str::to_string))
    };
    assert!(
        align("%b").is_some() && align("%b") == align("%a"),
        "{tile}"
    );
    let off = emission(&ir, "rtx-5080", EmitOpts::default(), &["narrow-tile"]);
    let dev_off = off.device.as_deref().unwrap_or("");
    let sites = |d: &str| d.lines().filter(|l| l.starts_with("; mapal.site ")).count();
    assert_eq!(sites(dev), 2, "the narrow and the f32 site");
    assert_eq!(
        sites(dev_off),
        1,
        "OFF: the f32 site alone; the narrow map a cell"
    );
    assert!(dev_off.contains("kind=cell"));
    // The PTX carries the conversions (where llc is installed).
    if let Ok(o) = std::process::Command::new("llc")
        .args(["-O3", "-mcpu=sm_120", "-o", "-"])
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
        for i in [
            "cvt.rn.bf16.f32",
            "cvt.rn.f16.f32",
            "cvt.f32.bf16",
            "ld.global.nc.b16",
        ] {
            assert!(ptx.contains(i), "no {i} in the PTX");
        }
    }
}

/// One generated program's checks; the coverage counters it adds.
#[derive(Default, Clone, Copy, Debug)]
struct Cover {
    programs: u64,
    f16: u64,
    bf16: u64,
    narrow_sites: u64,
    both: u64,
    strided: u64,
    post: u64,
    device_narrow_tile: u64,
    device_narrow_staged: u64,
    device_narrow_streamed: u64,
    device_narrow_cell: u64,
    inf: u64,
    nan: u64,
    subnormal: u64,
    guard: u64,
    looped: u64,
}

impl std::ops::AddAssign for Cover {
    fn add_assign(&mut self, o: Cover) {
        self.programs += o.programs;
        self.f16 += o.f16;
        self.bf16 += o.bf16;
        self.narrow_sites += o.narrow_sites;
        self.both += o.both;
        self.strided += o.strided;
        self.post += o.post;
        self.device_narrow_tile += o.device_narrow_tile;
        self.device_narrow_staged += o.device_narrow_staged;
        self.device_narrow_streamed += o.device_narrow_streamed;
        self.device_narrow_cell += o.device_narrow_cell;
        self.inf += o.inf;
        self.nan += o.nan;
        self.subnormal += o.subnormal;
        self.guard += o.guard;
        self.looped += o.looped;
    }
}

fn check_narrow(clang: &str, p: &testgen::NarrowProg) -> Cover {
    let (src, tight) = testgen::narrow_src(p);
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let want = rr.output.clone();
    // Exact face: the interpreter's bytes on the host and the emulated device
    // in every emulator mode.
    let host = emission(&ir, "generic", EmitOpts::exact(), &[]);
    let got = emu::run_emulated(clang, &host.text, host.device.as_deref().unwrap_or(""));
    assert_eq!(got, want, "exact host\n{src}");
    let ex = emission(&ir, "rtx-5080", EmitOpts::exact(), &[]);
    let dev = ex.device.as_deref().unwrap_or("");
    let built = emu::build_emulated(clang, &ex.text, dev);
    for mode in MODES {
        let (got, _) = built.run(&mode);
        assert_eq!(got, want, "exact device {mode:?}\n{src}");
    }
    // tensor-cores `narrow-async`: the `cp.async` pipeline forced on — a
    // narrow `b` panel streamed in its format, widened per fragment — keeps
    // each cell's order: the interpreter's bytes in every mode.
    let st = emission_on(&ir, "rtx-5080", EmitOpts::exact(), &[], &["async"]);
    let dev_st = st.device.as_deref().unwrap_or("").to_string();
    let streamed = dev_st
        .split("define ptx_kernel void @mapal_tile_")
        .skip(1)
        .filter(|k| k.contains("cp.async.ca.shared.global.8") && k.contains("fpext <"))
        .count() as u64;
    let built = emu::build_emulated(clang, &st.text, &dev_st);
    for mode in MODES {
        let (got, _) = built.run(&mode);
        assert_eq!(got, want, "exact device, async forced {mode:?}\n{src}");
    }
    // Default face: the f32 sums within D2 (a relative 1e-4 at these k), the
    // values through a rounding within D3's flips (4 ulps of the format
    // after the loop's three roundings and the update's).
    let ulp = if p.f16 { 2f64.powi(-10) } else { 2f64.powi(-7) };
    let check_default = |got: &str, what: &str| {
        let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
        assert_eq!(g.len(), w.len(), "{what}\n{src}");
        for (i, (a, b)) in g.iter().zip(&w).enumerate() {
            let tol = if i < tight { 1e-4 } else { 5.0 * ulp };
            assert!(close(a, b, tol), "{what} line {i}: {a} vs {b}\n{src}");
        }
    };
    let on = emission(&ir, "rtx-5080", EmitOpts::default(), &[]);
    let dev_on = on.device.as_deref().unwrap_or("").to_string();
    let built = emu::build_emulated(clang, &on.text, &dev_on);
    for mode in MODES {
        let (got, _) = built.run(&mode);
        check_default(&got, &format!("default device {mode:?}"));
    }
    let off = emission(&ir, "rtx-5080", EmitOpts::default(), &["narrow-tile"]);
    let got = emu::run_emulated(clang, &off.text, off.device.as_deref().unwrap_or(""));
    check_default(&got, "default device narrow-tile OFF");
    assert!(within(&got, &want, 1.0), "{src}");
    let sites = ir.tile_plan_narrow(ir.entry());
    let tile_kernels = dev_on
        .split("define ptx_kernel void @mapal_tile_")
        .skip(1)
        .filter(|k| k.contains("%kloop") && (k.contains("load bfloat") || k.contains("load half")))
        .count() as u64;
    // The register-staged tile over a narrow operand (its panel widened as it
    // is copied): a plan line of kind `tile` naming the format.
    let staged = dev_on
        .lines()
        .filter(|l| {
            l.starts_with("; mapal.plan ") && l.contains(" kind=tile ") && l.contains(" narrow=")
        })
        .count() as u64;
    let cell_kernels = dev_on
        .split("define ")
        .filter(|d| d.contains("store bfloat") || d.contains("store half"))
        .count() as u64;
    Cover {
        programs: 1,
        f16: u64::from(p.f16),
        bf16: u64::from(!p.f16),
        narrow_sites: sites.sites.len() as u64,
        both: u64::from(
            sites
                .sites
                .iter()
                .any(|(_, s)| s.a.narrow.is_some() && s.b.narrow.is_some()),
        ),
        strided: u64::from(
            sites
                .sites
                .iter()
                .any(|(_, s)| s.b.clane > 1 || s.a.clane > 1),
        ),
        post: u64::from(sites.sites.iter().any(|(_, s)| s.post.is_some())),
        device_narrow_tile: u64::from(tile_kernels > 0),
        device_narrow_staged: u64::from(staged > 0),
        device_narrow_streamed: u64::from(streamed > 0),
        device_narrow_cell: u64::from(cell_kernels > 0),
        inf: u64::from(want.contains("inf")),
        nan: u64::from(want.contains("NaN")),
        subnormal: u64::from(p.scale == 3),
        guard: u64::from(p.guard),
        looped: u64::from(p.looped),
    }
}

/// 64 generated programs: exact face bytes (host; emulated device under
/// every emulator mode), default face within the bounds (every mode), the
/// OFF arm too; every coverage counter non-zero.
#[test]
fn generated_programs_exact_bytes_and_default_bounds() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::NarrowProg> = (0..64)
        .map(|_| {
            testgen::narrow_prog_strategy()
                .new_tree(&mut runner)
                .unwrap()
                .current()
        })
        .collect();
    let covers = testgen::par_check(&progs, |_, p| check_narrow(&clang, p));
    let mut c = Cover::default();
    for k in covers {
        c += k;
    }
    eprintln!("narrow generated: {c:?}");
    let all = [
        c.programs,
        c.f16,
        c.bf16,
        c.narrow_sites,
        c.both,
        c.strided,
        c.post,
        c.device_narrow_tile,
        c.device_narrow_staged,
        c.device_narrow_streamed,
        c.device_narrow_cell,
        c.inf,
        c.nan,
        c.subnormal,
        c.guard,
        c.looped,
    ];
    assert!(all.iter().all(|&x| x > 0), "{c:?}");
}

/// tensor-cores `narrow-radix`: a decode step over a `bf16` KV cache
/// (`benches/algorithms/px_llm_dec_b2_l16_bf16kv`, the cache rows rounded
/// once as written): attention's scores and `P·V` read the cache through its
/// widening at the head digits — mixed-radix tile sites whose `b` is narrow
/// (`tile_plan_radix`, a `narrow=b:bf16 … radix=b` plan); the exact face is
/// the interpreter's bytes on the emulated device in every mode. OFF
/// (`--gpu-off=narrow-radix`): no such site.
#[test]
fn bf16_cache_attention_is_a_narrow_radix_site() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../benches/algorithms/px_llm_dec_b2_l16_bf16kv.mapal"
    ))
    .expect("the bf16-cache decode twin");
    let ir = lower(&src);
    let narrow_radix = ir
        .funcs()
        .flat_map(|(f, _)| ir.tile_plan_radix(f).into_iter().map(|(_, s)| s))
        .filter(|s| s.b.narrow == Some(NarrowFloat::Bf16) || s.a.narrow.is_some())
        .count();
    assert!(narrow_radix >= 2, "scores and P·V: {narrow_radix}");
    let lines = |e: &Emission| {
        e.device
            .as_deref()
            .unwrap_or("")
            .lines()
            .filter(|l| {
                l.starts_with("; mapal.plan ") && l.contains(" narrow=") && l.contains(" radix=")
            })
            .count()
    };
    let ex = emission(&ir, "rtx-5080", EmitOpts::exact(), &[]);
    assert!(lines(&ex) >= 2, "the narrow radix tiles");
    let off = emission(&ir, "rtx-5080", EmitOpts::exact(), &["narrow-radix"]);
    assert_eq!(lines(&off), 0, "OFF: per-cell maps");
    let Some(clang) = emu::clang() else {
        return;
    };
    let want = run(&ir, 4_000_000_000).output;
    let built = emu::build_emulated(&clang, &ex.text, ex.device.as_deref().unwrap_or(""));
    for mode in MODES {
        let (got, _) = built.run(&mode);
        assert_eq!(got, want, "exact device {mode:?}");
    }
}
