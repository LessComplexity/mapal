//! tensor-cores (ADR-0043 D4 / D6): a sum of products whose multiplicands are
//! 16-bit — operands stored `bf16` / `f16`, or `f32` under `: bf16 | f32` —
//! takes the part's `mma.sync.m16n8k16` with f32 accumulators (`half-tile`):
//! 16-byte `cp.async` panels in each operand's storage format, `ldmatrix`
//! fragments of the narrow ones, an f32 operand rounded once to the member as
//! its fragment is read. On programs from the generator (`testgen::HalfProg`:
//! GEMMs and attention heads it was not designed on): the facts, the
//! realization per arm, and the emulated device's values within ADR-0041 D3's
//! bound (`u_t` the member's, `u_a = 2⁻²³`) in every emulator mode and under
//! the OFF arm.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run, run_weakest};
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

/// The emission on `target` at `base`'s face, the rungs `off` off; `place`
/// off (every capable site on the device), each site its own kernel.
fn emission(ir: &CategoryIr, target: &'static str, base: EmitOpts, off: &[&str]) -> Emission {
    let mut gpu = GpuArms::default();
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

/// The plan lines of the 16-bit tensor-core kernels a module launches.
fn half_lines(dev: &str) -> Vec<String> {
    dev.lines()
        .filter(|l| l.starts_with("; mapal.plan ") && l.contains(" face=HalfMma "))
        .map(str::to_string)
        .collect()
}

fn floats(out: &str) -> Vec<f64> {
    out.split_whitespace()
        .map(|x| x.parse().expect("a float"))
        .collect()
}

const GEMV: &str = "fn main() {
    2048 -> iota -> ta;
    8192 -> iota -> tw;
    4096 -> iota -> t;
    64 -> iota -> kk;
    ta -> map { e -> (e * 7 + 3) % 31 - 15 -> widen_f32 -> v; v / 29.0 } -> a;
    tw -> map { e -> (e * 7 + 13) % 103 - 51 -> widen_f32 -> v; v / 1856.0 -> to_bf16 } -> w;
    t -> map { e -> e / 128 -> i; e % 128 -> j; (0.0, kk) -> fold: bf16 | f32 { acc, q -> w[q * 128 + j] -> widen_f32 -> b; acc + a[i * 64 + q] * b } } -> y;
    y[0] -> println;
    y[4095] -> println;
}
";

/// D4 facts: `: bf16 | f32` is the product's set (`Precision::half`), its
/// tile site carries the member, and `precision_plan.bound` is D3's with
/// `u_t = 2⁻⁸`, `u_a = 2⁻²³`; the weakest member's interpreter rounds the f32
/// operand (another value), the f32 side does not.
#[test]
fn member_facts_and_bound() {
    let ir = lower(GEMV);
    let f = ir.entry();
    let sites = ir.tile_plan_narrow(f);
    let (m, s) = sites.sites.iter().next().expect("a narrow site");
    assert_eq!(s.half, Some(NarrowFloat::Bf16));
    assert_eq!(s.unit(), Some(2f64.powi(-8)));
    let pp = ir.precision_plan(f);
    assert_eq!(
        pp.bound.get(m).copied(),
        Some(bound_coef_unit(64, Some(2f64.powi(-8))))
    );
    let sets: Vec<_> = ir
        .funcs()
        .flat_map(|(g, _)| ir.precision_plan(g).set.into_iter().map(|(_, p)| p))
        .collect();
    let mul = sets.first().expect("the product's set");
    assert!(mul.bf16 && mul.f32 && !mul.tf32 && !mul.f16);
    let f32 = run(&ir, 1 << 30).output;
    let weak = run_weakest(&ir, 1 << 30, &[]).output;
    assert_ne!(f32, weak, "the member rounds `a`");
}

/// D6 / the realization per arm on the RTX 5080 (`half-fill` off: the
/// small site takes the tile wherever it fits): `mma.sync.m16n8k16.bf16` —
/// `cp.async` panels (the f32 `a` 4 bytes, the bf16 `b` 2), `ldmatrix.trans`
/// of `b`, `cvt.rn.bf16x2.f32` of `a`; OFF (`half-tile`) and the exact face:
/// no 16-bit `mma`. The plan names the member's set and the instruction.
/// `half-fill` on: this skinny site (8 flop / B against the 5080's 54.9,
/// 16 one-warp blocks) takes the FMA tile; a compute-bound one (256 × 512 ×
/// 512, 85 flop / B) the 16-bit tile.
#[test]
fn realization_per_arm() {
    let ir = lower(GEMV);
    let shipped = emission(&ir, "rtx-5080", EmitOpts::default(), &[]);
    assert!(half_lines(shipped.device.as_deref().unwrap_or("")).is_empty());
    let big = GEMV
        .replace("2048 -> iota -> ta", "131072 -> iota -> ta")
        .replace("8192 -> iota -> tw", "262144 -> iota -> tw")
        .replace("4096 -> iota -> t;", "131072 -> iota -> t;")
        .replace("64 -> iota -> kk", "512 -> iota -> kk")
        .replace("e / 128", "e / 512")
        .replace("e % 128", "e % 512")
        .replace("q * 128 + j", "q * 512 + j")
        .replace("i * 64 + q", "i * 512 + q")
        .replace("y[4095]", "y[131071]");
    let e = emission(&lower(&big), "rtx-5080", EmitOpts::default(), &[]);
    assert_eq!(
        half_lines(e.device.as_deref().unwrap_or("")).len(),
        1,
        "{big}"
    );
    let on = emission(&ir, "rtx-5080", EmitOpts::default(), &["half-fill"]);
    let dev = on.device.as_deref().expect("device");
    let lines = half_lines(dev);
    assert_eq!(lines.len(), 1, "{dev}");
    let l = &lines[0];
    for f in [
        " kind=mma ",
        " op=m16n8k16.bf16 ",
        " ea=4 eb=2",
        " set=bf16|f32 ",
        " narrow=b:bf16 ",
    ] {
        assert!(l.contains(f), "{f}: {l}");
    }
    for i in [
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        "ldmatrix.sync.aligned.m8n8.x",
        ".trans.shared.b16",
        "cvt.rn.bf16x2.f32",
        "@llvm.nvvm.cp.async.cg.shared.global.16",
    ] {
        assert!(dev.contains(i), "{i}");
    }
    let off = emission(
        &ir,
        "rtx-5080",
        EmitOpts::default(),
        &["half-tile", "half-fill"],
    );
    let dev_off = off.device.as_deref().unwrap_or("");
    assert!(half_lines(dev_off).is_empty());
    assert!(!dev_off.contains("m16n8k16"));
    let exact = emission(&ir, "rtx-5080", EmitOpts::exact(), &["half-fill"]);
    assert!(
        half_lines(exact.device.as_deref().unwrap_or("")).is_empty(),
        "the exact face keeps the oracle's f32 products in its order"
    );
    // Every part listing the warp bf16 product takes it; ids differ from the
    // f32 program's (the site keys its member).
    for t in ["rtx-3090", "a100", "rtx-4090", "rtx-5090", "h100-pcie"] {
        let e = emission(&ir, t, EmitOpts::default(), &["half-fill"]);
        assert_eq!(
            half_lines(e.device.as_deref().unwrap_or("")).len(),
            1,
            "{t}"
        );
    }
}

/// `{f32}` over two operands of one narrow format: exact products, so the
/// member's realization (the products chained on the truncating
/// accumulator) is not admitted — every profiled part truncates
/// (`mma_calib.cu`): no `half-tile` kernel under the default face's set.
/// f32-tensor-cores: the split form `narrow1` takes it instead (each k-step
/// from a zeroed accumulator, one round-to-nearest add: D2 at the site's
/// `K`, `split_tile.rs`), its own rung `split-tile`.
#[test]
fn exact_products_need_the_round_to_nearest_accumulator() {
    let src = GEMV
        .replace("fold: bf16 | f32", "fold")
        .replace("v / 29.0 }", "v / 29.0 -> to_bf16 }")
        .replace(
            "acc + a[i * 64 + q] * b",
            "a[i * 64 + q] -> widen_f32 -> x; acc + x * b",
        );
    let ir = lower(&src);
    let s = ir.tile_plan_narrow(ir.entry());
    let site = s.sites.values().next().expect("site");
    assert!(site.a.narrow.is_some() && site.b.narrow.is_some() && site.half.is_none());
    let e = emission(&ir, "rtx-5080", EmitOpts::default(), &["half-fill"]);
    let dev = e.device.as_deref().unwrap_or("");
    assert!(half_lines(dev).is_empty());
    // This GEMV is DRAM-bound and its grid small: `split_pays` keeps the FMA
    // tile; the ON arm (the bound alone) takes narrow1.
    let mut arms = GpuArms::default();
    *arms.on.rung("split-tile").expect("rung") = true;
    for r in [
        "place",
        "async",
        "pdl",
        "streams",
        "pinned",
        "horizontal",
        "calls",
    ] {
        *arms.off.rung(r).expect("rung") = true;
    }
    let forced = emit_detailed(
        &ir,
        &EmitOpts {
            host: "generic",
            target: "rtx-5080",
            gpu: arms,
            ..EmitOpts::default()
        },
    )
    .expect("emits");
    assert!(
        forced
            .device
            .as_deref()
            .unwrap_or("")
            .lines()
            .any(|l| l.starts_with("; mapal.plan ") && l.contains(" split=narrow1")),
        "the split form narrow1"
    );
    let e = emission(
        &ir,
        "rtx-5080",
        EmitOpts::default(),
        &["half-fill", "split-tile"],
    );
    assert!(!e.device.as_deref().unwrap_or("").contains("mma.sync"));
    // Under the program's `: bf16 | f32` the same operands take it.
    let ir = lower(&src.replace("-> fold {", "-> fold: bf16 | f32 {"));
    let e = emission(&ir, "rtx-5080", EmitOpts::default(), &["half-fill"]);
    let lines = half_lines(e.device.as_deref().unwrap_or(""));
    assert_eq!(lines.len(), 1);
    assert!(lines[0].contains(" ea=2 eb=2"), "{}", lines[0]);
}

const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_GRID", "2")],
    [("MAPAL_EMU_REVERSE", "1"), ("MAPAL_EMU_GRID", "1")],
];

#[derive(Default, Clone, Copy, Debug)]
struct Cover {
    programs: u64,
    gemm: u64,
    attn: u64,
    bf16_set: u64,
    f16_set: u64,
    half_kernels: u64,
    a_rounded: u64,
    a_narrow: u64,
    b_rounded: u64,
    b_narrow: u64,
    bt: u64,
    post: u64,
    fallback: u64,
    f16_kernels: u64,
    narrow_radix: u64,
}

impl std::ops::AddAssign for Cover {
    fn add_assign(&mut self, o: Cover) {
        self.programs += o.programs;
        self.gemm += o.gemm;
        self.attn += o.attn;
        self.bf16_set += o.bf16_set;
        self.f16_set += o.f16_set;
        self.half_kernels += o.half_kernels;
        self.a_rounded += o.a_rounded;
        self.a_narrow += o.a_narrow;
        self.b_rounded += o.b_rounded;
        self.b_narrow += o.b_narrow;
        self.bt += o.bt;
        self.post += o.post;
        self.fallback += o.fallback;
        self.f16_kernels += o.f16_kernels;
        self.narrow_radix += o.narrow_radix;
    }
}

/// One program: the emulated device (every mode, and the OFF arm) against
/// the interpreter.
///
/// Products and scores, `(y, M)` pairs: D3 from both sides — the device and
/// the weakest member's interpreter are each within the set's coefficient
/// of the exact value, so within twice it of each other.
///
/// An attention head's output pairs `(o, M_o)`: `o = Σ_j p_j·v_j` over the
/// device's own `p`, whose scores moved by at most `Δs` (the score pairs'
/// coefficient times the largest score magnitude `smax`), so each `p_j`
/// within `(e^{2Δs} − 1)·p_j` before the softmax's own roundings (the exp
/// form's bound, the sum and the division: 2⁻¹⁸ relative), then a rounding
/// to the member flips it at most `2u·p_j` (ADR-0043 D3): `|Δo| ≤ 2c·M_o +
/// (e^{2Δs} − 1 + 2⁻¹⁸ + 2u)·M_o`.
fn check(clang: &str, p: &testgen::HalfProg) -> Cover {
    let (src, (pairs, k, k_out)) = testgen::half_src(p);
    let ir = lower(&src);
    let rr = run(&ir, 4_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let weak = floats(&run_weakest(&ir, 4_000_000_000, &[]).output);
    let unit = match p.set {
        1 => Some(2f64.powi(-8)),
        2 => Some(2f64.powi(-11)),
        _ => None,
    };
    let c = |k: u64| bound_coef_unit(k, unit) + bound_coef_unit(k, None);
    // An attention head prints its largest score magnitude first.
    let (at, smax) = if p.attn { (1, weak[0]) } else { (0, 0.0) };
    let check_values = |got: &str, what: &str| {
        let g = floats(got);
        assert_eq!(g.len(), weak.len(), "{what}\n{src}");
        for i in 0..pairs {
            let (y, w, m) = (g[at + 2 * i], weak[at + 2 * i], weak[at + 2 * i + 1]);
            let tol = 2.0 * c(k) * m + 2f64.powi(-22) * w.abs();
            assert!(
                (y - w).abs() <= tol,
                "{what} pair {i}: {y} vs {w} (M {m}, tol {tol})\n{src}"
            );
        }
        if p.attn {
            // The scores' largest move: their coefficient at the largest
            // score magnitude (`smax`, printed first).
            let ds = 2.0 * c(k) * smax;
            let u = unit.unwrap_or(0.0);
            for i in pairs..(g.len() - at) / 2 {
                let (o, w, m) = (g[at + 2 * i], weak[at + 2 * i], weak[at + 2 * i + 1]);
                let rel = (2.0 * ds).exp_m1() + 2f64.powi(-18) + 2.0 * u;
                let tol = 2.0 * c(k_out) * m + rel * m + 2f64.powi(-22) * w.abs();
                assert!(
                    (o - w).abs() <= tol,
                    "{what} output {i}: {o} vs {w} (M {m}, tol {tol})\n{src}"
                );
            }
        }
    };
    // `half-fill` off: the small programs take the 16-bit tile wherever it
    // fits (the rule's own arm is `realization_per_arm`'s).
    let on = emission(&ir, "rtx-5080", EmitOpts::default(), &["half-fill"]);
    let dev = on.device.as_deref().unwrap_or("").to_string();
    let built = emu::build_emulated(clang, &on.text, &dev);
    for mode in MODES {
        let (got, _) = built.run(&mode);
        check_values(&got, &format!("default {mode:?}"));
    }
    let off = emission(
        &ir,
        "rtx-5080",
        EmitOpts::default(),
        &["half-tile", "half-fill"],
    );
    let dev_off = off.device.as_deref().unwrap_or("");
    assert!(half_lines(dev_off).is_empty(), "OFF: no 16-bit mma\n{src}");
    let got = emu::run_emulated(clang, &off.text, dev_off);
    check_values(&got, "half-tile OFF");
    let lines = half_lines(&dev);
    let has = |f: &str| u64::from(lines.iter().any(|l| l.contains(f)));
    let fallback = u64::from(p.set > 0 && lines.is_empty());
    Cover {
        programs: 1,
        gemm: u64::from(!p.attn),
        attn: u64::from(p.attn && !lines.is_empty()),
        bf16_set: u64::from(p.set == 1),
        f16_set: u64::from(p.set == 2),
        half_kernels: lines.len() as u64,
        a_rounded: has(" ea=4"),
        a_narrow: has(" ea=2"),
        b_rounded: has(" eb=4"),
        b_narrow: has(" eb=2"),
        bt: has(" bt=1"),
        post: u64::from(lines.iter().any(|l| l.contains(" post=1"))),
        fallback,
        f16_kernels: has(".f16 "),
        // tensor-cores `narrow-radix`: a head digit's site over a narrow cache.
        narrow_radix: u64::from(dev.lines().any(|l| {
            l.starts_with("; mapal.plan ") && l.contains(" narrow=") && l.contains(" radix=")
        })),
    }
}

/// 48 generated programs (GEMMs and attention heads over f32 / bf16 / f16
/// storage under `{f32}`, `: bf16 | f32`, `: f16 | f32`): the emulated
/// device within the bound in every emulator mode and under the OFF arm;
/// every coverage counter non-zero.
#[test]
fn generated_programs_within_the_member_bound() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::HalfProg> = (0..48)
        .map(|_| {
            testgen::half_prog_strategy()
                .new_tree(&mut runner)
                .unwrap()
                .current()
        })
        .collect();
    // The real device's check: `MAPAL_HALF_DUMP=<dir>` writes each program
    // for `mapal test --weakest --target=<part>` on the part itself.
    if let Ok(dir) = std::env::var("MAPAL_HALF_DUMP") {
        for (i, p) in progs.iter().enumerate() {
            let _ = std::fs::write(format!("{dir}/half_{i:02}.mapal"), testgen::half_src(p).0);
        }
    }
    let covers = testgen::par_check(&progs, |_, p| check(&clang, p));
    let mut c = Cover::default();
    for k in covers {
        c += k;
    }
    eprintln!("half-tile generated: {c:?}");
    let all = [
        c.programs,
        c.gemm,
        c.attn,
        c.bf16_set,
        c.f16_set,
        c.half_kernels,
        c.a_rounded,
        c.a_narrow,
        c.b_rounded,
        c.b_narrow,
        c.bt,
        c.post,
        c.fallback,
        c.f16_kernels,
        c.narrow_radix,
    ];
    assert!(all.iter().all(|&x| x > 0), "{c:?}");
}
