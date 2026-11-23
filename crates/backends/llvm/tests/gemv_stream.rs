//! gemv-stream: a streaming sum of products (a matrix-vector product: its
//! flops per operand byte below the part's FFMA balance) on the
//! one-thread-per-cell kernel. `vec-cells`: a thread folds the `16 / e` lanes
//! one 16-byte load of `b` covers, where the layout facts prove the load;
//! `stream-fly`: the `k` split puts a block on every SM with the part's
//! measured in-flight knee in flight on each. On programs from the generator
//! (`testgen::GemvProg`: f32 / bf16 / f16 weights, unit, strided, offset and
//! per-row layouts, a shared activation, an epilogue): the exact face's bytes
//! and the default face within its bound, in every emulator mode and under
//! each OFF arm, at part sizes that reach the vector and split kernels at the
//! programs' own small extents (the SM count overridden: `--cost=sms=`).

use mapal_backend_llvm::{CostOver, DeviceCost, Emission, EmitOpts, GpuArms, emit_detailed};
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

/// The part's SM count overridden to `sms` (0: the profile's).
fn sms(n: u64) -> CostOver {
    let mut c: CostOver = [None; DeviceCost::NAMES.len()];
    if n > 0 {
        let i = DeviceCost::NAMES
            .iter()
            .position(|x| *x == "sms")
            .expect("sms");
        c[i] = Some(n);
    }
    c
}

/// The emission on the RTX 5080 at `base`'s face with `n` SMs, the rungs
/// `off` off; `place` off (every capable site on the device), each site its
/// own kernel, one candidate a site.
fn emission(ir: &CategoryIr, base: EmitOpts, n: u64, off: &[&str]) -> Emission {
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
    gpu.site_cands = 1;
    let opts = EmitOpts {
        host: "generic",
        target: "rtx-5080",
        gpu,
        cost: sms(n),
        ..base
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// The one-thread-per-cell plan lines of a device unit.
fn naive_lines(dev: &str) -> Vec<String> {
    dev.lines()
        .filter(|l| l.starts_with("; mapal.plan ") && l.contains(" kind=naive "))
        .map(str::to_string)
        .collect()
}

/// A plan line's `key=` value.
fn field(line: &str, key: &str) -> Option<u64> {
    line.split_whitespace()
        .find_map(|w| w.strip_prefix(&format!("{key}=")))
        .and_then(|v| v.parse().ok())
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
    4096 -> iota -> t;
    4096 -> iota -> kk;
    16777216 -> iota -> tw;
    kk -> map { e -> (e * 7 + 3) % 5 -> widen_f32 -> v; v / 4.0 } -> a;
    tw -> map { e -> (e * 11 + 5) % 9 + 1 -> widen_f32 -> v; v / 8.0 } -> w;
    t -> map { j -> (0.0, kk) -> fold { acc, q -> acc + a[q] * w[q * 4096 + j] } } -> y;
    y[0] -> println;
}
";

/// The rules on the measured parts' own properties: the RTX 5080 (84 SMs,
/// an 8 KiB knee) cuts the 1 × 4096 × 4096 GEMV into 32 partitions of four
/// cells a thread (4 blocks a partition: a block on every SM, 49 KiB in
/// flight on each); the H100 SXM (132 SMs, a 32 KiB knee) into 64. OFF
/// (`vec-cells`): one cell a thread, the warp fill's 8 partitions on the
/// 5080 (16 blocks each); OFF (`stream-fly`): the warp fill's 8 partitions
/// alone, whose vector grid (32 blocks) would leave SMs idle — one cell a
/// thread. The ordered (exact) kernel keeps one cell a thread: four blocks
/// would leave 80 SMs idle.
#[test]
fn rules_per_part_and_arm() {
    let ir = lower(GEMV);
    let line = |target: &'static str, base: EmitOpts, off: &[&str]| {
        let mut gpu = GpuArms::default();
        // `place` on: off, a part without plan rates drops its measured
        // costs (the H100's knee with them).
        for r in ["pdl"].iter().chain(off) {
            *gpu.off.rung(r).expect("rung") = true;
        }
        gpu.site_cands = 1;
        let e = emit_detailed(
            &ir,
            &EmitOpts {
                host: "generic",
                target,
                gpu,
                ..base
            },
        )
        .expect("emits");
        let l = naive_lines(e.device.as_deref().unwrap_or(""));
        assert_eq!(l.len(), 1, "{l:?}");
        let s = |k: &str| field(&l[0], k).unwrap_or(0);
        (s("cells"), s("S"), s("gx"), s("fly_sm"), s("knee"))
    };
    let d = EmitOpts::default();
    assert_eq!(line("rtx-5080", d, &[]), (4, 32, 4, 49932, 8192));
    assert_eq!(line("h100", d, &[]), (4, 64, 4, 63550, 32768));
    assert_eq!(line("rtx-5080", d, &["vec-cells"]), (1, 8, 16, 12483, 8192));
    assert_eq!(
        line("rtx-5080", d, &["stream-fly"]),
        (1, 8, 16, 12483, 8192)
    );
    assert_eq!(line("rtx-5080", EmitOpts::exact(), &[]).0, 1);
    // The PTX the vector kernel compiles to: 16-byte loads of `b`.
    let mut gpu = GpuArms::default();
    *gpu.off.rung("place").unwrap() = true;
    gpu.site_cands = 1;
    let e = emit_detailed(
        &ir,
        &EmitOpts {
            host: "generic",
            target: "rtx-5080",
            gpu,
            ..d
        },
    )
    .expect("emits");
    let dev = e.device.as_deref().unwrap_or("");
    assert!(
        dev.contains("load <4 x float>, ptr addrspace(1)"),
        "vector load"
    );
    assert!(dev.contains(", align 16"), "16-byte aligned");
}

/// A 1 × `c` × `k` GEMV ([`GEMV`]'s shape at other extents).
fn gemv(c: u64, k: u64) -> String {
    GEMV.replace("4096 -> iota -> t;", &format!("{c} -> iota -> t;"))
        .replace("4096 -> iota -> kk;", &format!("{k} -> iota -> kk;"))
        .replace(
            "16777216 -> iota -> tw;",
            &format!("{} -> iota -> tw;", c * k),
        )
        .replace("w[q * 4096 + j]", &format!("w[q * {c} + j]"))
}

/// regressions `vec-cells` (cross-arch 3a8e654f): on a part whose DRAM share
/// an SM is at least `DRAM_PER_SM_SPREAD` the vector cells' grid must spread
/// over the SMs as evenly as the scalar grid at the same split. 1 × 6144 ×
/// 4096: 192 vector blocks — on the H100's 132 SMs spread 1.375 against the
/// scalar grid's 1.03 (measured 0.829 ON / OFF): one cell a thread; on the
/// A100's 108, 1.125 both (1.035 won): four cells. 1 × 18432 × 7168 on the
/// A100: 126 vector blocks at S 7, 1.71 against 1.07 (0.953 lost). The RTX
/// 5080 (11.4 KB/µs an SM) takes the vector cells at every spread.
#[test]
fn vec_cells_spread_per_part() {
    let cells = |src: &str, target: &'static str| {
        let ir = lower(src);
        let mut gpu = GpuArms::default();
        *gpu.off.rung("pdl").expect("rung") = true;
        gpu.site_cands = 1;
        let e = emit_detailed(
            &ir,
            &EmitOpts {
                host: "generic",
                target,
                gpu,
                ..EmitOpts::default()
            },
        )
        .expect("emits");
        let l = naive_lines(e.device.as_deref().unwrap_or(""));
        assert_eq!(l.len(), 1, "{target}: {l:?}");
        (
            field(&l[0], "cells").unwrap_or(1),
            field(&l[0], "gx").unwrap_or(0) * field(&l[0], "S").unwrap_or(1),
        )
    };
    let n6144 = gemv(6144, 4096);
    assert_eq!(cells(&n6144, "rtx-5080").0, 4);
    assert_eq!(cells(&n6144, "a100"), (4, 192));
    assert_eq!(cells(&n6144, "h100").0, 1);
    let n18432 = gemv(18432, 7168);
    assert_eq!(cells(&n18432, "a100").0, 1);
    assert_eq!(cells(&n18432, "rtx-5080").0, 4);
}

/// The B1 decode's O projection: a GEMV whose map body goes on past its
/// fold (`h = x + o·W`, a `tile-post` site).
const GEMV_POST: &str = "fn main() {
    4096 -> iota -> t;
    4096 -> iota -> kk;
    16777216 -> iota -> tw;
    kk -> map { e -> (e * 7 + 3) % 5 -> widen_f32 -> v; v / 4.0 } -> a;
    t -> map { e -> (e * 3 + 1) % 7 -> widen_f32 -> v; v / 2.0 } -> x;
    tw -> map { e -> (e * 11 + 5) % 9 + 1 -> widen_f32 -> v; v / 8.0 } -> w;
    t -> map { j -> (0.0, kk) -> fold { acc, q -> acc + a[q] * w[q * 4096 + j] } -> s; x[j] + s } -> y;
    y[0] -> println;
}
";

/// cross-arch 3a8e654f: a `tile-post` site's split is a workspace whose
/// reduce launch runs the epilogue — launched through the site's table even
/// when the table holds one candidate (the single launch keeps an epilogue
/// group ordered). The H100 / A100 / 3090 planned S 64 / 32 / 32 for the B1
/// decode's O projection (the A100 now 16: one cell a thread, `vec-spread`) and ran the ordered 16-block kernel (H100 235 us a
/// call against 29 split). On every part: the post kernel launched is the
/// split (a block on every SM), its reduce evaluates the epilogue, the host
/// calls the table.
#[test]
fn post_split_launches_through_its_table() {
    let ir = lower(GEMV_POST);
    for (target, s, sms) in [
        ("h100", 64, 132),
        ("a100", 16, 108),
        ("rtx-3090", 32, 82),
        ("rtx-5080", 32, 84),
    ] {
        for cands in [1, 8] {
            let mut gpu = GpuArms::default();
            *gpu.off.rung("pdl").expect("rung") = true;
            gpu.site_cands = cands;
            let e = emit_detailed(
                &ir,
                &EmitOpts {
                    host: "generic",
                    target,
                    gpu,
                    ..EmitOpts::default()
                },
            )
            .expect("emits");
            let dev = e.device.as_deref().unwrap_or("");
            let posts: Vec<String> = naive_lines(dev)
                .into_iter()
                .filter(|l| l.contains(" post=1"))
                .collect();
            assert!(!posts.is_empty(), "{target}: a post site\n{dev}");
            assert!(
                posts.iter().any(|l| l.contains(" combine=splitk ")
                    && field(l, "S") == Some(s)
                    && field(l, "gx").unwrap_or(0) * s >= sms),
                "{target} cands {cands}: the planned split launched\n{posts:?}"
            );
            assert!(
                !posts.iter().any(|l| l.contains(" combine=none ")),
                "{target} cands {cands}: no ordered post kernel\n{posts:?}"
            );
            assert!(
                dev.lines()
                    .any(|l| l.contains(" kind=reduce ") && l.contains(" epi=1")),
                "{target}: the reduce runs the epilogue"
            );
            assert!(
                e.text.contains("call void @mapal_gpu_site("),
                "{target}: launched through its table"
            );
        }
        // integration: the OFF arm (`--gpu-off=post-table`) is the emission
        // before it — where the site keeps one candidate, the ordered kernel.
        let mut gpu = GpuArms::default();
        *gpu.off.rung("pdl").expect("rung") = true;
        *gpu.off.rung("post-table").expect("rung") = true;
        gpu.site_cands = 1;
        let e = emit_detailed(
            &ir,
            &EmitOpts {
                host: "generic",
                target,
                gpu,
                ..EmitOpts::default()
            },
        )
        .expect("emits");
        let dev = e.device.as_deref().unwrap_or("");
        assert!(
            naive_lines(dev)
                .iter()
                .any(|l| l.contains(" post=1") && l.contains(" combine=none ")),
            "{target} post-table off: the ordered post kernel"
        );
        assert!(
            !e.text.contains("call void @mapal_gpu_site("),
            "{target} post-table off"
        );
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Cover {
    programs: u64,
    vec_f32: u64,
    vec_narrow: u64,
    vec_split: u64,
    vec_atomic: u64,
    vec_workspace: u64,
    vec_post: u64,
    post_split: u64,
    vec_ordered: u64,
    vec_rows: u64,
    scalar_strided: u64,
    scalar_misaligned: u64,
    split_scalar: u64,
    spread_refused: u64,
    spread_kept: u64,
}

impl std::ops::AddAssign for Cover {
    fn add_assign(&mut self, o: Cover) {
        self.programs += o.programs;
        self.vec_f32 += o.vec_f32;
        self.vec_narrow += o.vec_narrow;
        self.vec_split += o.vec_split;
        self.vec_atomic += o.vec_atomic;
        self.vec_workspace += o.vec_workspace;
        self.vec_post += o.vec_post;
        self.post_split += o.post_split;
        self.vec_ordered += o.vec_ordered;
        self.vec_rows += o.vec_rows;
        self.scalar_strided += o.scalar_strided;
        self.scalar_misaligned += o.scalar_misaligned;
        self.split_scalar += o.split_scalar;
        self.spread_refused += o.spread_refused;
        self.spread_kept += o.spread_kept;
    }
}

fn check(clang: &str, p: &testgen::GemvProg) -> Cover {
    let src = testgen::gemv_src(p);
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let want = rr.output.clone();
    let mut lines = Vec::new();
    // Exact face (one SM: the ordered kernel takes the vector cells): the
    // interpreter's bytes in every mode, and OFF.
    for (n, off) in [(1, &[][..]), (1, &["vec-cells"][..])] {
        let ex = emission(&ir, EmitOpts::exact(), n, off);
        let dev = ex.device.as_deref().unwrap_or("").to_string();
        let built = emu::build_emulated(clang, &ex.text, &dev);
        for mode in MODES {
            let (got, _) = built.run(&mode);
            assert_eq!(got, want, "exact {n} SMs {off:?} {mode:?}\n{src}");
        }
        lines.extend(naive_lines(&dev));
    }
    // Default face: positive operands, so a reordered (split) sum stays
    // within a relative `k · 2⁻²³` of the interpreter's.
    let tol = (p.k + 2 * p.rows * p.cols) as f64 * 2f64.powi(-22);
    let check_default = |got: &str, what: &str| {
        let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
        assert_eq!(g.len(), w.len(), "{what}\n{src}");
        for (i, (a, b)) in g.iter().zip(&w).enumerate() {
            assert!(close(a, b, tol), "{what} line {i}: {a} vs {b}\n{src}");
        }
    };
    // regressions `vec-spread`: at 6 and 8 SMs the RTX 5080's DRAM share an
    // SM is 120–160 KB/µs, above `DRAM_PER_SM_SPREAD`: the vector cells only
    // where their grid spreads as evenly as one cell a thread's (6 SMs: a
    // count no power-of-two split divides, as 108 / 132 do not); OFF at any
    // spread.
    let mut vec_at: Vec<(u64, bool, bool)> = Vec::new();
    for (n, off) in [
        (2, &[][..]),
        (8, &[][..]),
        (84, &[][..]),
        (2, &["vec-cells"][..]),
        (2, &["stream-fly"][..]),
        (2, &["splitk-atomic"][..]),
        (6, &[][..]),
        (6, &["vec-spread"][..]),
        (8, &["vec-spread"][..]),
    ] {
        let on = emission(&ir, EmitOpts::default(), n, off);
        let dev = on.device.as_deref().unwrap_or("").to_string();
        let built = emu::build_emulated(clang, &on.text, &dev);
        for mode in MODES {
            let (got, _) = built.run(&mode);
            check_default(&got, &format!("default {n} SMs {off:?} {mode:?}"));
        }
        let ls = naive_lines(&dev);
        if off.is_empty() || off == ["vec-spread"] {
            let vec = ls.iter().any(|l| field(l, "cells").unwrap_or(1) > 1);
            vec_at.push((n, off.is_empty(), vec));
        }
        if off.contains(&"vec-cells") {
            assert!(
                ls.iter().all(|l| field(l, "cells").unwrap_or(1) == 1),
                "OFF: one cell a thread\n{src}"
            );
        }
        lines.extend(ls);
    }
    let vec_on = |n: u64, rule: bool| vec_at.iter().any(|&(m, r, v)| m == n && r == rule && v);
    let spread_refused = [6, 8].iter().any(|&n| vec_on(n, false) && !vec_on(n, true));
    let spread_kept = [6, 8].iter().any(|&n| vec_on(n, true));
    let vec: Vec<&String> = lines
        .iter()
        .filter(|l| field(l, "cells").unwrap_or(1) > 1)
        .collect();
    let has = |f: &dyn Fn(&str) -> bool| u64::from(vec.iter().any(|l| f(l)));
    let scalar_stream = lines
        .iter()
        .any(|l| field(l, "cells") == Some(1) && field(l, "S").unwrap_or(1) > 1);
    // The layout facts the vector load needs: where they fail, no vector.
    let misaligned = p.layout == 2 && (p.off * [4, 2, 2][usize::from(p.fmt)]) % 16 != 0;
    if p.layout == 1 || misaligned {
        assert!(vec.is_empty(), "no proof of the 16-byte load\n{src}");
    }
    Cover {
        programs: 1,
        vec_f32: has(&|l| field(l, "cells") == Some(4) && !l.contains(" narrow=")),
        vec_narrow: has(&|l| field(l, "cells") == Some(8) && l.contains(" narrow=b:")),
        vec_split: has(&|l| field(l, "S").unwrap_or(1) > 1),
        vec_atomic: has(&|l| l.contains(" combine=atomic ")),
        vec_workspace: has(&|l| l.contains(" combine=splitk ")),
        vec_post: has(&|l| l.contains(" post=1")),
        post_split: u64::from(
            lines
                .iter()
                .any(|l| l.contains(" post=1") && l.contains(" combine=splitk ")),
        ),
        vec_ordered: has(&|l| l.contains(" combine=none ")),
        vec_rows: has(&|l| field(l, "rows").unwrap_or(1) > 1),
        scalar_strided: u64::from(p.layout == 1 && !lines.is_empty()),
        scalar_misaligned: u64::from(misaligned && !lines.is_empty()),
        split_scalar: u64::from(scalar_stream),
        spread_refused: u64::from(spread_refused),
        spread_kept: u64::from(spread_kept),
    }
}

/// 64 generated programs: the exact face's bytes and the default face within
/// its bound in every emulator mode, at 1, 2, 6, 8 and 84 SMs and under the
/// `vec-cells`, `stream-fly`, `splitk-atomic` and `vec-spread` OFF arms; every coverage
/// counter non-zero.
#[test]
fn generated_programs_keep_values_on_every_arm() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::GemvProg> = (0..64)
        .map(|_| {
            testgen::gemv_prog_strategy()
                .new_tree(&mut runner)
                .unwrap()
                .current()
        })
        .collect();
    let covers = testgen::par_check(&progs, |_, p| check(&clang, p));
    let mut c = Cover::default();
    for k in covers {
        c += k;
    }
    eprintln!("gemv-stream generated: {c:?}");
    let all = [
        c.programs,
        c.vec_f32,
        c.vec_narrow,
        c.vec_split,
        c.vec_atomic,
        c.vec_workspace,
        c.vec_post,
        c.post_split,
        c.vec_ordered,
        c.vec_rows,
        c.scalar_strided,
        c.scalar_misaligned,
        c.split_scalar,
        c.spread_refused,
        c.spread_kept,
    ];
    assert!(all.iter().all(|&x| x > 0), "{c:?}");
}
