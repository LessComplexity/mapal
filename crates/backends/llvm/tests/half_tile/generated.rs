//! Generated programs within the member bound, rounded stores and shared bodies.

use super::*;

pub(crate) const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_GRID", "2")],
    [("MAPAL_EMU_REVERSE", "1"), ("MAPAL_EMU_GRID", "1")],
];

#[derive(Default, Clone, Copy, Debug)]
pub(crate) struct Cover {
    pub(crate) programs: u64,
    pub(crate) gemm: u64,
    pub(crate) attn: u64,
    pub(crate) bf16_set: u64,
    pub(crate) f16_set: u64,
    pub(crate) half_kernels: u64,
    pub(crate) a_rounded: u64,
    pub(crate) a_narrow: u64,
    pub(crate) b_rounded: u64,
    pub(crate) b_narrow: u64,
    pub(crate) bt: u64,
    pub(crate) post: u64,
    pub(crate) fallback: u64,
    pub(crate) f16_kernels: u64,
    pub(crate) narrow_radix: u64,
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
pub(crate) fn check(clang: &str, p: &testgen::HalfProg) -> Cover {
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

/// sweep-gemm `round-store`: what one generated GEMM covers.
#[derive(Default, Clone, Copy, Debug)]
pub(crate) struct StoreCover {
    pub(crate) programs: u64,
    /// Arrays `PrecisionPlan::store` rounds at their producer.
    pub(crate) stored: u64,
    /// An f32-stored operand read as a 2-byte panel on the ON arm.
    pub(crate) a_stored_tile: u64,
    pub(crate) b_stored_tile: u64,
    /// A stored operand on a site without the 16-bit tile (the FMA tile).
    pub(crate) fma_reads: u64,
    /// Refused: an f32 operand under `{f32}` or `: f16 | f32`.
    pub(crate) refused: u64,
    pub(crate) post: u64,
    /// An `a` the fact would store, returned from its function: kept f32.
    pub(crate) escaped: u64,
}

impl std::ops::AddAssign for StoreCover {
    fn add_assign(&mut self, o: StoreCover) {
        self.programs += o.programs;
        self.stored += o.stored;
        self.a_stored_tile += o.a_stored_tile;
        self.b_stored_tile += o.b_stored_tile;
        self.fma_reads += o.fma_reads;
        self.refused += o.refused;
        self.post += o.post;
        self.escaped += o.escaped;
    }
}

/// One generated GEMM whose operands only the product reads
/// (`testgen::half_src_with(p, true)`): the emulated device on the ON arm
/// (the arrays rounded by their producers) in every emulator mode, and on
/// the OFF arm (`round-store`: each reader rounds its copy), against the
/// weakest member's interpreter within D3 from both sides.
pub(crate) fn check_store(clang: &str, p: &testgen::HalfProg, esc: bool) -> StoreCover {
    let (src, (pairs, k, _)) = match esc {
        true => testgen::half_src_escaping(p),
        false => testgen::half_src_with(p, true),
    };
    let ir = lower(&src);
    let weak = floats(&run_weakest(&ir, 4_000_000_000, &[]).output);
    let unit = match p.set {
        1 => Some(2f64.powi(-8)),
        2 => Some(2f64.powi(-11)),
        _ => None,
    };
    let c = bound_coef_unit(k, unit) + bound_coef_unit(k, None);
    let check_values = |got: &str, what: &str| {
        let g = floats(got);
        assert_eq!(g.len(), weak.len(), "{what}\n{src}");
        for i in 0..pairs {
            let (y, w, m) = (g[2 * i], weak[2 * i], weak[2 * i + 1]);
            let tol = 2.0 * c * m + 2f64.powi(-22) * w.abs();
            assert!(
                (y - w).abs() <= tol,
                "{what} pair {i}: {y} vs {w} (M {m}, tol {tol})\n{src}"
            );
        }
    };
    let stored: usize = ir
        .funcs()
        .map(|(f, _)| ir.precision_plan(f).store.len())
        .sum();
    // The fact: an f32 operand under `: bf16 | f32` is stored rounded; one
    // under `{f32}` / `: f16 | f32`, or already narrow, is not.
    // An epilogue reads `a` itself (`u·0.5 + a[i·k]`): an f32 reader. A
    // returned `a` escapes its function: its caller reads f32.
    let a_rounds = p.set == 1 && p.fa == 0 && !p.post;
    let expect = usize::from(a_rounds && !esc) + usize::from(p.set == 1 && p.fb == 0);
    assert_eq!(stored, expect, "{src}");
    assert_eq!(ir.narrow_stored().is_some(), stored > 0);
    if let Some(g) = ir.narrow_stored() {
        assert!(mapal_ir::validate(&g).is_empty(), "{src}");
        assert!(g.narrow_stored().is_none(), "idempotent\n{src}");
    }
    let on = emission(&ir, "rtx-5080", EmitOpts::default(), &["half-fill"]);
    let dev = on.device.as_deref().unwrap_or("").to_string();
    let built = emu::build_emulated(clang, &on.text, &dev);
    for mode in MODES {
        let (got, _) = built.run(&mode);
        check_values(&got, &format!("round-store {mode:?}"));
    }
    let off = emission(
        &ir,
        "rtx-5080",
        EmitOpts::default(),
        &["half-fill", "round-store"],
    );
    let dev_off = off.device.as_deref().unwrap_or("");
    if stored == 0 {
        assert_eq!(dev, dev_off, "nothing stored: the OFF arm's text\n{src}");
    }
    let got = emu::run_emulated(clang, &off.text, dev_off);
    check_values(&got, "round-store OFF");
    let lines = half_lines(&dev);
    let narrow_line = |x: &str| {
        u64::from(
            p.set == 1
                && dev
                    .lines()
                    .any(|l| l.starts_with("; mapal.plan ") && l.contains(x)),
        )
    };
    StoreCover {
        programs: 1,
        stored: stored as u64,
        a_stored_tile: u64::from(p.fa == 0) * u64::from(lines.iter().any(|l| l.contains(" ea=2"))),
        b_stored_tile: u64::from(p.fb == 0) * u64::from(lines.iter().any(|l| l.contains(" eb=2"))),
        fma_reads: u64::from(stored > 0 && lines.is_empty())
            * (narrow_line("a:bf16") | narrow_line("b:bf16")),
        refused: u64::from(p.set != 1 && (p.fa == 0 || p.fb == 0)),
        post: u64::from(stored > 0 && p.post),
        escaped: u64::from(esc && a_rounds),
    }
}

/// sweep-gemm `round-store` on 40 generated GEMMs (f32 / bf16 / f16 storage
/// under `{f32}`, `: bf16 | f32`, `: f16 | f32`; `b` k × n or n × k;
/// epilogues; one in five in a kept callee returning its `a`): the fact per
/// operand (a returned `a` escapes: never stored), the rewritten graph valid
/// and idempotent, values within the bound on the emulated device in every
/// mode and under the OFF arm; every coverage counter non-zero.
#[test]
fn round_store_on_generated_programs() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    // The draws' operand formats and sets, leaned toward the case the rung
    // reads: f32 operands (half the `a`s, a third of the `b`s) under
    // `: bf16 | f32` (three programs in four).
    let progs: Vec<testgen::HalfProg> = (0..40)
        .map(|i| {
            let mut p = testgen::half_prog_strategy()
                .new_tree(&mut runner)
                .unwrap()
                .current();
            p.attn = false;
            if i % 4 != 3 {
                p.set = 1;
            }
            if i % 2 == 0 {
                p.fa = 0;
            }
            if i % 3 == 0 {
                p.fb = 0;
            }
            p
        })
        .collect();
    // One program in five returns its GEMM's `a` from a kept callee.
    let covers = testgen::par_check(&progs, |i, p| check_store(&clang, p, i % 5 == 1));
    let mut c = StoreCover::default();
    for k in covers {
        c += k;
    }
    eprintln!("round-store generated: {c:?}");
    let all = [
        c.programs,
        c.stored,
        c.a_stored_tile,
        c.b_stored_tile,
        c.fma_reads,
        c.refused,
        c.post,
        c.escaped,
    ];
    assert!(all.iter().all(|&x| x > 0), "{c:?}");
}
