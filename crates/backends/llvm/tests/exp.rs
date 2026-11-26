//! exp (ADR-0042): `x -> exp` on the host and the device. The exact face is
//! the oracle's algorithm op for op — the interpreter's bytes on the host
//! build and the emulated device, over generated programs reaching every
//! branch (overflow, subnormal results, ±inf and NaN arguments, `f64`); the
//! default face takes the contracted polynomial on the host and the
//! `ex2.approx` form on the device, within the bound; `fold_combine`'s
//! rescale law lets `row-slices` fold a softmax's sum beside its max (online
//! softmax), one phase fewer, values within the bound in every emulator mode.

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

/// The emission on `target` and `base`'s face, `s` blocks a row forced
/// (0 = the rule's), the rungs `off` off; the emulator's rungs off and
/// `place` off, so every capable map is a device cell.
fn emission(
    ir: &CategoryIr,
    target: &'static str,
    base: EmitOpts,
    s: u64,
    off: &[&str],
) -> Emission {
    let mut gpu = GpuArms::default();
    gpu.on.flow = true;
    gpu.row_slices_s = s;
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
        host: "generic",
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

fn field<'l>(line: &'l str, key: &str) -> &'l str {
    line.split_whitespace()
        .find_map(|f| f.strip_prefix(key).and_then(|v| v.strip_prefix('=')))
        .unwrap_or("")
}

fn slice_lines(em: &Emission) -> Vec<String> {
    em.device
        .as_deref()
        .unwrap_or("")
        .lines()
        .filter(|l| l.starts_with("; mapal.flow kernel=") && l.contains(" slices="))
        .map(str::to_string)
        .collect()
}

const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_GRID", "2")],
    [("MAPAL_EMU_REVERSE", "1"), ("MAPAL_EMU_GRID", "1")],
];

/// The realization each side and face takes (the intents): the exact face
/// defines the oracle's steps on both sides (no fast-math flag, no `ex2`);
/// the default face the host's contracted polynomial and the device's
/// `ex2.approx.ftz.f32` after the reduction; `--gpu-off=exp-hw` the oracle's
/// steps on the default face; `f64` never `ex2`.
#[test]
fn realization_per_side_and_face() {
    let p = testgen::ExpProg {
        n: 1024,
        rows: 2,
        mul: 7,
        add: 3,
        m: 101,
        scale: 2,
        wide: true,
        nan: false,
        softmax: false,
        in_fold: false,
    };
    let src = testgen::exp_src(&p);
    let ir = lower(&src);
    let def32 = "define internal float @mapal.exp.f32(";
    let def64 = "define internal double @mapal.exp.f64(";
    let on = emission(&ir, "rtx-5080", EmitOpts::default(), 0, &[]);
    let dev = on.device.as_deref().expect("device cells");
    assert!(dev.contains(def32) && dev.contains(def64));
    assert!(dev.contains("call float @llvm.nvvm.ex2.approx.ftz.f(float %a)"));
    assert_eq!(
        dev.matches("ex2.approx").count(),
        2,
        "one call, one declaration"
    );
    assert!(on.text.contains(def32) && on.text.contains("fmul contract float"));
    assert!(!on.text.contains("ex2.approx") && !on.text.contains("reassoc float %kf"));
    let exact = emission(&ir, "rtx-5080", EmitOpts::exact(), 0, &[]);
    for t in [&exact.text, exact.device.as_deref().unwrap()] {
        assert!(t.contains(def32) && !t.contains("ex2.approx"));
        let body: String = t
            .split(def32)
            .nth(1)
            .unwrap()
            .split("\n}\n")
            .next()
            .unwrap()
            .into();
        assert!(
            !body.contains("contract") && !body.contains("@llvm.fma"),
            "{body}"
        );
    }
    let off = emission(&ir, "rtx-5080", EmitOpts::default(), 0, &["exp-hw"]);
    assert!(!off.device.as_deref().unwrap().contains("ex2.approx"));
    assert!(!off.text.contains("fmul contract float %kf"));
    // The PTX carries the hardware instruction (where llc is installed).
    if let Ok(o) = std::process::Command::new("llc")
        .args(["-O3", "-mcpu=sm_120", "-nvptx-fma-level=0", "-o", "-"])
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
        assert!(ptx.contains("ex2.approx.ftz.f32"), "no ex2 in the PTX");
    }
}

/// The exact face is the oracle's algorithm: generated programs over every
/// branch of it — the host build and the emulated device print the
/// interpreter's bytes; the default face (host contracted, device `ex2`)
/// within the bound (relative 1e-4 on the printed sums, which also carry the
/// order the face frees).
#[test]
fn exact_face_bytes_and_default_bound_on_generated_programs() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::ExpProg> = (0..48)
        .map(|_| {
            testgen::exp_prog_strategy()
                .new_tree(&mut runner)
                .unwrap()
                .current()
        })
        .collect();
    let counts = testgen::par_check(&progs, |_, p| check_exp(&clang, p));
    let mut c = [0u64; 6];
    for k in counts {
        for i in 0..6 {
            c[i] += k[i];
        }
    }
    eprintln!(
        "exp generated: programs {} inf {} nan {} subnormal-or-zero {} f64 {} device-ex2 {}",
        c[0], c[1], c[2], c[3], c[4], c[5]
    );
    assert!(c.iter().all(|&x| x > 0), "{c:?}");
}

/// One generated program's checks: `[programs, inf, nan, zero, f64, ex2]`.
fn check_exp(clang: &str, p: &testgen::ExpProg) -> [u64; 6] {
    let src = testgen::exp_src(p);
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let want = rr.output.clone();
    for target in ["generic", "rtx-5080"] {
        let ex = emission(&ir, target, EmitOpts::exact(), 0, &[]);
        let dev = ex.device.as_deref().unwrap_or("");
        assert!(
            target == "generic" || dev.contains("@mapal.exp.f32("),
            "{src}"
        );
        let got = emu::run_emulated(clang, &ex.text, dev);
        assert_eq!(got, want, "exact {target}\n{src}");
    }
    let on = emission(&ir, "rtx-5080", EmitOpts::default(), 0, &[]);
    let dev = on.device.as_deref().unwrap_or("");
    let got = emu::run_emulated(clang, &on.text, dev);
    assert!(
        within(&got, &want, 1e-4),
        "default: {got:?} vs {want:?}\n{src}"
    );
    [
        1,
        u64::from(want.contains("inf")),
        u64::from(want.contains("NaN")),
        u64::from(want.lines().any(|l| {
            l == "0"
                || l.contains("e-4")
                || l.starts_with("0.0000000000000000000000000000000000000")
        })),
        u64::from(p.wide),
        u64::from(dev.contains("ex2.approx")),
    ]
}

/// A softmax over `rows` rows of `len` scores (`q · W`, a tile site), the
/// `exp` builtin of the shifted score.
fn softmax(rows: u64, len: u64, scaled: bool) -> String {
    let n = rows * len;
    let ex = if scaled {
        format!("(x[e] - mx[e / {len}]) / 8.0 -> exp")
    } else {
        format!("x[e] - mx[e / {len}] -> exp")
    };
    format!(
        "fn main() {{
    {n} -> iota -> t;
    {rows} -> iota -> rr;
    {len} -> iota -> kl;
    8 -> iota -> kd;
    {rq} -> iota -> tq;
    tq -> map {{ e -> (e * 5 + 3) % 31 - 15 -> widen<f32> -> v; e / 8 + 1 -> widen<f32> -> rf; v * rf / 4.0 }} -> q;
    {lw} -> iota -> tw;
    tw -> map {{ e -> (e * 7 + 13) % 101 - 50 -> widen<f32> -> v; v / 16.0 }} -> w;
    t -> map {{ e -> e / {len} -> i; e % {len} -> j; (0.0, kd) -> fold {{ acc, d -> acc + q[i * 8 + d] * w[d * {len} + j] }} }} -> x;
    rr -> map {{ i -> (-1000000000.0, kl) -> fold {{ acc, j -> x[i * {len} + j] -> e; (e > acc) -> {{ -true-> e; -false-> acc; }} }} }} -> mx;
    t -> map {{ e -> {ex} }} -> ex;
    rr -> map {{ i -> (0.0, kl) -> fold {{ acc, j -> acc + ex[i * {len} + j] }} }} -> den;
    t -> map {{ e -> ex[e] / den[e / {len}] }} -> y;
    den[0] -> println;
    y[0] -> println;
    y[{mid}] -> println;
    y[{last}] -> println;
    0.0 -> zero: f32;
    (zero, y) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
}}
",
        rq = rows * 8,
        lw = len * 8,
        mid = n / 2 + 5,
        last = n - 1
    )
}

/// The rescale law in `row-slices` (online softmax): the denominator's
/// partials take each slice's own max (`mx`'s partial over the same slice)
/// and are rescaled by `exp(m_B − m)` when combined — `mx` and `den` fold in
/// phase 0, two phases and one grid barrier where the phase rule takes
/// three; values within the bound in every emulator mode; the OFF arm
/// (`--no-exp-rescale`) three phases, the exact face one block a row.
#[test]
fn rescale_folds_the_sum_beside_the_max() {
    let Some(clang) = emu::clang() else {
        return;
    };
    for (rows, len, scaled) in [(3, 1024, true), (1, 2048, false), (2, 512, true)] {
        let src = softmax(rows, len, scaled);
        let ir = lower(&src);
        let rr = run(&ir, 2_000_000_000);
        assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
        for s in [2u64, 4, 8] {
            let on = emission(&ir, "rtx-5080", EmitOpts::default(), s, &[]);
            let lines = slice_lines(&on);
            assert_eq!(lines.len(), 1, "s={s}");
            assert_eq!(field(&lines[0], "phases"), "2");
            assert_eq!(field(&lines[0], "laws"), "rescale");
            let dev = on.device.as_deref().unwrap();
            assert_eq!(dev.matches("call void @mapal_gsync(").count(), 1);
            let ex = emu::build_emulated(&clang, &on.text, dev);
            for mode in MODES {
                let (got, _) = ex.run(&mode);
                assert!(
                    within(&got, &rr.output, 1e-4),
                    "rows={rows} s={s} {mode:?}: {got:?} vs {:?}",
                    rr.output
                );
            }
            let off = emission(&ir, "rtx-5080", EmitOpts::default(), s, &["exp-rescale"]);
            let l = slice_lines(&off);
            assert_eq!(field(&l[0], "phases"), "3");
            assert_eq!(field(&l[0], "laws"), "");
            let got = emu::run_emulated(&clang, &off.text, off.device.as_deref().unwrap());
            assert!(within(&got, &rr.output, 1e-4), "{got:?} vs {:?}", rr.output);
        }
        let exact = emission(&ir, "rtx-5080", EmitOpts::exact(), 4, &[]);
        assert!(slice_lines(&exact).is_empty());
        let got = emu::run_emulated(&clang, &exact.text, exact.device.as_deref().unwrap());
        assert_eq!(got, rr.output, "exact face bytes");
    }
}

/// `−1e39`: an `f32` literal past the range, `−∞`.
const NEG_INF: &str = "-1000000000000000000000000000000000000000.0";

/// A softmax whose scores are masked to `−∞` past a per-row cut (`j + 100 i
/// < cut`): row 0 keeps `cut` scores, row 1 `cut − 100`, row 2 none; the
/// running max seeded at `seed` (a finite floor, or `−∞` itself).
fn masked_softmax(len: u64, cut: u64, seed: &str) -> String {
    let n = 3 * len;
    format!(
        "fn main() {{
    {n} -> iota -> t;
    3 -> iota -> rr;
    {len} -> iota -> kl;
    8 -> iota -> kd;
    24 -> iota -> tq;
    tq -> map {{ e -> (e * 5 + 3) % 31 - 15 -> widen<f32> -> v; v / 4.0 }} -> q;
    {lw} -> iota -> tw;
    tw -> map {{ e -> (e * 7 + 13) % 101 - 50 -> widen<f32> -> v; v / 16.0 }} -> w;
    t -> map {{ e -> e / {len} -> i; e % {len} -> j; (0.0, kd) -> fold {{ acc, d -> acc + q[i * 8 + d] * w[d * {len} + j] }} }} -> x;
    t -> map {{ e -> e / {len} -> i; e % {len} -> j; (j + i * 100 < {cut}) -> {{ -true-> x[e]; -false-> {NEG_INF}; }} }} -> xm;
    rr -> map {{ i -> ({seed}, kl) -> fold {{ acc, j -> xm[i * {len} + j] -> e; (e > acc) -> {{ -true-> e; -false-> acc; }} }} }} -> mx;
    t -> map {{ e -> (xm[e] - mx[e / {len}]) / 8.0 -> exp }} -> ex;
    rr -> map {{ i -> (0.0, kl) -> fold {{ acc, j -> acc + ex[i * {len} + j] }} }} -> den;
    t -> map {{ e -> ex[e] / den[e / {len}] }} -> y;
    den[0] -> println;
    den[1] -> println;
    den[2] -> println;
    y[0] -> println;
    y[{c1}] -> println;
    y[{cut}] -> println;
    y[{r1}] -> println;
    y[{r2}] -> println;
    0.0 -> zero: f32;
    (zero, kl) -> fold {{ acc, j -> acc + y[j] }} -> total;
    total -> println;
}}
",
        lw = len * 8,
        c1 = cut - 1,
        r1 = len + 5,
        r2 = 2 * len + 7,
    )
}

/// The rescale law over slices whose every score is `−∞` (a mask): the
/// slice's own max is the step's identity, so its terms take the stand-in
/// shift `0` (each `exp(−∞) = 0`) and the combine rescales by the true
/// shift's `exp(a(−∞, m))` — `0` beside a finite row max, `NaN` where the
/// row max is `−∞` too, as the sequential fold. Interpreter = every emulator
/// mode, at 2 / 4 / 8 slices a row (slices wholly masked, one straddling the
/// cut, a row wholly masked), the max seeded finite and at `−∞`. Without the
/// stand-in every row with a masked slice prints `NaN`.
#[test]
fn rescale_over_masked_slices() {
    let Some(clang) = emu::clang() else {
        return;
    };
    for seed in ["-1000000000.0", NEG_INF] {
        let src = masked_softmax(512, 150, seed);
        let ir = lower(&src);
        let rr = run(&ir, 2_000_000_000);
        assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
        // Row 2 is wholly masked: 0 / 0 under the finite seed, NaN through
        // `−∞ − (−∞)` under the infinite one; rows 0 and 1 finite.
        let rows: Vec<&str> = rr.output.lines().collect();
        assert!(rows[0].parse::<f64>().is_ok_and(f64::is_normal), "{rows:?}");
        assert!(rows[1].parse::<f64>().is_ok_and(f64::is_normal), "{rows:?}");
        assert_eq!(
            rows[2],
            if seed == NEG_INF { "NaN" } else { "0" },
            "{rows:?}"
        );
        for s in [2u64, 4, 8] {
            let on = emission(&ir, "rtx-5080", EmitOpts::default(), s, &[]);
            let lines = slice_lines(&on);
            let unit = lines.iter().find(|l| l.contains(" root=mx "));
            let unit = unit.unwrap_or_else(|| panic!("s={s}: {lines:#?}"));
            assert_eq!(field(unit, "phases"), "2", "s={s}");
            assert_eq!(field(unit, "laws"), "rescale", "s={s}");
            let dev = on.device.as_deref().unwrap();
            let ex = emu::build_emulated(&clang, &on.text, dev);
            for mode in MODES {
                let (got, _) = ex.run(&mode);
                assert!(
                    within(&got, &rr.output, 1e-4),
                    "seed={seed} s={s} {mode:?}: {got:?} vs {:?}",
                    rr.output
                );
            }
        }
    }
}
