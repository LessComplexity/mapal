//! bf16 tensor-tile `mma-walk`: a 16-bit tensor tile's grid walked by
//! stream-K over every resident slot, on programs from the generator
//! (`testgen::GemmProg`: GEMMs it was not designed on) under the `fold: bf16
//! | f32` product set:
//!
//! - **Plans, every part.** A walked tensor tile (`combine=streamk
//!   walk=mma`) launches at most the part's resident slots (`ctas ≤ sms ×
//!   blocks_per_sm`), its `ctas × iters` units cover the `tiles × panels`
//!   space with no empty block, it keeps its whole warp tile (no block of
//!   fewer warps than the ordered tile's), and its kernel adds the shared
//!   tiles' partials (`atomicrmw fadd`) beside the whole tiles' plain store;
//!   its k loop is the ordered tile's (the warp tile's `mt·nt·bk/k` products
//!   a trip, walked or not); the OFF arm (`--gpu-off=mma-walk`) and the fixed
//!   order walk none.
//! - **Reuse, every part** (hopper-wgmma `walk-reuse`). The walk's readers
//!   of an operand slice sit where its rotation puts them; a slice the
//!   part's L2 keep does not hold between two readers is read again. No
//!   walk the price takes is made DRAM-bound by those re-reads where its
//!   operands read once are not (`Gpu::dram_bound`), and every forced walk
//!   (`--gpu-on=mma-walk`) that is, the default arm keeps ordered (some
//!   generated shapes are gated: coverage > 0).
//! - **Values, the emulator.** Runnable extents with the walk forced
//!   (`--gpu-on=mma-walk`, the measurement arm): integer operands (exact in
//!   bf16, sums exact in f32 in any order), plain GEMMs (the zeroed output's
//!   `red`) and epilogue groups (the tiles' workspace and ticket), the
//!   emulated device equal to the interpreter in every emulator mode, and the
//!   OFF arm too.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed, resolve_profile};
use mapal_interp::{Outcome, run};
use mapal_ir::CategoryIr;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use std::collections::HashMap;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

const PARTS: [&str; 5] = ["rtx-5080", "rtx-4090", "rtx-3090", "a100", "h100"];

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

/// `src` (a generated GEMM, or its epilogue group) under `fold: bf16 | f32`.
fn bf16(src: String) -> String {
    assert!(src.contains("(0.0, kr) -> fold {"), "{src}");
    src.replace("(0.0, kr) -> fold {", "(0.0, kr) -> fold: bf16 | f32 {")
}

fn emission(
    ir: &CategoryIr,
    target: &'static str,
    off: &[&str],
    on: &[&str],
    extra: &[&str],
) -> Emission {
    let mut gpu = GpuArms {
        site_cands: 1,
        ..GpuArms::default()
    };
    for r in ["pdl", "streams", "pinned", "horizontal", "calls"]
        .iter()
        .chain(off)
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    for r in on {
        *gpu.on.rung(r).expect("rung") = true;
    }
    let mut opts = EmitOpts {
        host: "generic",
        target,
        gpu,
        ..EmitOpts::default()
    };
    for f in extra {
        assert!(mapal_backend_llvm::flags::apply(&mut opts, f).expect("a flag"));
    }
    emit_detailed(ir, &opts).expect("emits")
}

/// Each tensor-tile kernel's plan fields and its body.
fn mma_kernels(e: &Emission) -> Vec<(HashMap<String, String>, String)> {
    let dev = e.device.as_deref().unwrap_or_default();
    let mut out = Vec::new();
    for (i, l) in dev.lines().enumerate() {
        if !(l.starts_with("; mapal.plan ") && l.contains(" kind=mma ")) {
            continue;
        }
        let f: HashMap<String, String> = l
            .split(' ')
            .filter_map(|f| f.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let name = &f["kernel"];
        let body: String = dev
            .lines()
            .skip(i)
            .skip_while(|x| !(x.starts_with("define ") && x.contains(name.as_str())))
            .take_while(|x| *x != "}")
            .collect::<Vec<_>>()
            .join("\n");
        out.push((f, body));
    }
    out
}

fn num(p: &HashMap<String, String>, k: &str) -> u64 {
    p.get(k)
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("plan field {k}: {p:?}"))
}

fn walked(p: &HashMap<String, String>) -> bool {
    p.get("combine").is_some_and(|c| c == "streamk")
}

/// A warp tensor tile's `mma.sync` in its emitted body against its warp
/// tile's products a k-loop trip (`mt · nt · bk / k` of the `m16n8kK` op);
/// `None` for another instruction (the warpgroup product).
fn mma_issue(t: &HashMap<String, String>, body: &str) -> Option<(u64, u64)> {
    let k = match t.get("op")?.split('.').next()? {
        "m16n8k16" => 16,
        "m16n8k8" => 8,
        _ => return None,
    };
    let want = num(t, "mt") * num(t, "nt") * num(t, "bk") / k;
    Some((body.matches("mma.sync").count() as u64, want))
}

/// Large extents (plans only): skinny-to-square projections, `k` deep
/// enough to be walked.
fn plan_dims(p: &testgen::GemmProg) -> (u64, u64, u64) {
    let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
    let [dm, dn, dk] = p.dims;
    (
        pick(dm, &[128, 256, 512, 1024, 2048]),
        pick(dn, &[512, 1024, 2048, 4096, 8192]),
        pick(dk, &[1024, 2048, 4096, 8192]),
    )
}

#[derive(Default, Clone, Copy, Debug)]
struct Cover {
    kernels: u64,
    walked: u64,
    kept: u64,
    shared: u64,
    gated: u64,
    loops: u64,
}

impl std::ops::AddAssign for Cover {
    fn add_assign(&mut self, o: Cover) {
        self.kernels += o.kernels;
        self.walked += o.walked;
        self.kept += o.kept;
        self.shared += o.shared;
        self.gated += o.gated;
        self.loops += o.loops;
    }
}

/// hopper-wgmma `walk-reuse`, the oracle: whether a walked plan's re-reads
/// make its site DRAM-bound where reading its operands once does not. Block
/// `c` runs units `[c·iters, (c+1)·iters)`, so tile `t`'s panel `κ` is read
/// at unit time `(t·panels + κ) mod iters`; a block-row's tiles read one `a`
/// slice, a column's one `b` slice, and a reader more than the keep's worth
/// of the walk's stream (`ctas` slices of each operand a unit) after the
/// previous one reads it again — sampled at eight panel offsets, as the
/// price does.
fn rereads_bind(t: &HashMap<String, String>, gpu: &Facts) -> bool {
    let (bm, bn) = (
        num(t, "warps_y") * num(t, "mt") * 16,
        num(t, "warps_x") * num(t, "nt") * 8,
    );
    let (rows, c, k, bk) = (num(t, "rows"), num(t, "c"), num(t, "k"), num(t, "bk"));
    let (gx, gy, panels) = (c / bn, rows / bm, k / bk);
    let (ctas, iters) = (num(t, "ctas"), num(t, "iters"));
    let (ea, eb) = (num(t, "ea"), num(t, "eb"));
    let (sa, sb) = (bm * bk * ea, bn * bk * eb);
    let near = gpu.keep() / (ctas * (sa + sb)).max(1);
    let samples = panels.clamp(1, 8);
    let runs = |mut ts: Vec<u64>| {
        ts.sort_unstable();
        1 + ts.windows(2).filter(|w| w[1] - w[0] > near).count() as u64
    };
    let mut read = 0;
    for s in 0..samples {
        let kappa = s * panels / samples;
        let at = |tile: u64| (tile * panels + kappa) % iters;
        for r in 0..gy {
            read += runs((0..gx).map(|j| at(r * gx + j)).collect()) * sa;
        }
        for j in 0..gx {
            read += runs((0..gy).map(|r| at(r * gx + j)).collect()) * sb;
        }
    }
    let once_ops = (gy * sa + gx * sb) * panels;
    let once = rows * k * ea + k * c * eb + rows * c * 4;
    let walk = once + (read * panels / samples).saturating_sub(once_ops);
    let flops = 2 * rows * c * k;
    gpu.bound(walk, flops) && !gpu.bound(once, flops)
}

/// The part's properties the oracle reads.
struct Facts(&'static str);

impl Facts {
    fn keep(&self) -> u64 {
        resolve_profile(self.0)
            .unwrap()
            .gpu()
            .unwrap()
            .l2_keep_bytes()
    }
    fn bound(&self, bytes: u64, flops: u64) -> bool {
        let g = resolve_profile(self.0).unwrap().gpu().unwrap();
        // The part's warp 16-bit product, flops a ns.
        let rate = g.sms * g.mma_half_per_clk * g.clock_khz / 1_000_000;
        g.dram_bound(bytes, flops, rate)
    }
}

fn check_plans(p: &testgen::GemmProg, part: &'static str) -> Cover {
    let (m, n, k) = plan_dims(p);
    let src = bf16(testgen::gemm_src_at(p, m, n, k));
    let ir = lower(&src);
    let ctx = format!("{part} {m}x{n}x{k} bt={}", p.bt);
    let gpu = *resolve_profile(part).expect("a part").gpu().expect("a GPU");
    let on = emission(&ir, part, &[], &[], &[]);
    let off = emission(&ir, part, &["mma-walk"], &[], &[]);
    let fixed = emission(&ir, part, &[], &[], &["--order=fixed"]);
    let mut c = Cover::default();
    let ks = mma_kernels(&on);
    let ordered = mma_kernels(&off);
    for (t, body) in &ks {
        c.kernels += 1;
        if !walked(t) {
            c.kept += 1;
            continue;
        }
        c.walked += 1;
        assert_eq!(
            t.get("walk").map(String::as_str),
            Some("mma"),
            "{ctx}: {t:?}"
        );
        let (bm, bn) = (
            num(t, "warps_y") * num(t, "mt") * 16,
            num(t, "warps_x") * num(t, "nt") * 8,
        );
        let tiles = (num(t, "rows") / bm) * (num(t, "c") / bn);
        let panels = num(t, "k") / num(t, "bk");
        let (ctas, iters) = (num(t, "ctas"), num(t, "iters"));
        assert!(
            ctas <= gpu.sms * num(t, "blocks_per_sm"),
            "{ctx}: every block resident: {t:?}"
        );
        assert!(
            ctas * iters >= tiles * panels && (ctas - 1) * iters < tiles * panels,
            "{ctx}: the walk covers the units: {t:?}"
        );
        // The whole warp tile: never fewer warps nor a smaller warp tile
        // than the ordered kernel's.
        for (o, _) in &ordered {
            let warps = |x: &HashMap<String, String>| num(x, "warps_x") * num(x, "warps_y");
            let tile = |x: &HashMap<String, String>| num(x, "mt") * num(x, "nt");
            assert!(
                warps(t) >= warps(o) || tile(t) > tile(o),
                "{ctx}: the walk keeps its tile: {t:?} against {o:?}"
            );
        }
        assert!(
            body.contains("atomicrmw fadd")
                && body.contains("st_whole:")
                && body.contains("st_part:"),
            "{ctx}: shared pieces add, whole tiles store"
        );
        c.shared +=
            u64::from((0..tiles).any(|x| x * panels / iters != ((x + 1) * panels - 1) / iters));
        assert!(
            !rereads_bind(t, &Facts(part)),
            "{ctx}: the walk's re-reads make it DRAM-bound: {t:?}"
        );
    }
    // A forced walk its re-reads bind: the default arm keeps the ordered
    // grid (or its cut).
    let forced = mma_kernels(&emission(&ir, part, &[], &["mma-walk"], &[]));
    // The walk's k loop is the ordered tile's (`walk-reuse`, correcting
    // disasm-rented's "the walk's loop halves", a static SASS count): the warp
    // tile's `mt·nt·bk/k` products a trip, walked or not, in one k-step body
    // (LLVM's peel of the ordered loop's last trip comes later, in `opt`).
    for (t, body) in forced.iter().chain(&ordered) {
        if let Some((got, want)) = mma_issue(t, body) {
            assert_eq!(got, want, "{ctx}: the k loop's products a trip: {t:?}");
            c.loops += u64::from(walked(t));
        }
    }
    if forced
        .iter()
        .any(|(t, _)| walked(t) && rereads_bind(t, &Facts(part)))
    {
        c.gated += 1;
        assert!(
            ks.iter().all(|(t, _)| !walked(t)),
            "{ctx}: a walk its re-reads bind, taken"
        );
    }
    assert!(
        ordered
            .iter()
            .all(|(t, _)| !walked(t) || t.get("walk").is_none_or(|w| w != "mma")),
        "{ctx}: the OFF arm walks no tensor tile"
    );
    assert!(
        mma_kernels(&fixed).iter().all(|(t, _)| !walked(t)),
        "{ctx}: a fixed order walks none"
    );
    c
}

#[test]
fn mma_walk_rules_on_generated_shapes() {
    let mut runner = TestRunner::deterministic();
    let strat = testgen::gemm_prog_strategy();
    let progs: Vec<testgen::GemmProg> = (0..48)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let covers = testgen::par_check(&progs, |_, p| {
        let mut c = Cover::default();
        for part in PARTS {
            c += check_plans(p, part);
        }
        c
    });
    let mut c = Cover::default();
    for k in covers {
        c += k;
    }
    eprintln!("mma-walk plans: {c:?}");
    assert!(c.kernels >= 48, "{c:?}");
    assert!(
        c.walked > 0 && c.kept > 0,
        "the price walks some, keeps some: {c:?}"
    );
    assert!(c.shared > 0, "walks with shared tiles: {c:?}");
    assert!(c.gated > 0, "walks the re-reads gate: {c:?}");
    assert!(c.loops > 0, "walked k loops compared: {c:?}");
}

const MODES: [[(&str, &str); 2]; 3] = [
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "1")],
];

/// **Values through the emulator.** 6 generated GEMMs at runnable extents
/// (`m` 512, `n` 6144, `k` 64: 192 tiles of two panels on the 168 slots,
/// three panels a block — whole and shared tiles), the walk forced, each
/// program plain (the zeroed output's adds) and as an epilogue group (the
/// tiles' workspace and ticket), the emulated device equal to the
/// interpreter in every emulator mode, and the OFF arm too.
#[test]
fn mma_walk_values_through_the_emulator() {
    let Some(cc) = emu::clang() else {
        eprintln!("no clang: emulator runs skipped");
        return;
    };
    let mut runner = TestRunner::deterministic();
    let strat = testgen::gemm_prog_strategy();
    let progs: Vec<testgen::GemmProg> = (0..6)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let (m, n, k) = (512, 6144, 64);
    let counts = testgen::par_check(&progs, |z, p| {
        let mut walks = [0usize; 3];
        for epi in [false, true] {
            let src = bf16(match epi {
                false => testgen::gemm_src_at(p, m, n, k),
                true => testgen::gemm_epi_src_at(p, m, n, k, z as u8),
            });
            let ctx = format!("#{z} epi={epi} bt={}", p.bt);
            let ir = lower(&src);
            let rr = run(&ir, 20_000_000_000);
            assert!(matches!(rr.outcome, Outcome::Done(_)), "{:?}", rr.outcome);
            let want = rr.output;
            let on = emission(&ir, "rtx-5080", &[], &["mma-walk"], &[]);
            let off = emission(&ir, "rtx-5080", &["mma-walk"], &[], &[]);
            let ks = mma_kernels(&on);
            let w = ks.iter().filter(|(t, _)| walked(t)).count();
            assert!(w > 0, "{ctx}: the forced walk\n{src}");
            walks[usize::from(epi)] += w;
            let dev = |e: &Emission| e.device.clone().unwrap_or_default();
            let built = emu::build_emulated(&cc, &on.text, &dev(&on));
            for (env, (got, _)) in MODES.iter().zip(built.run_each(&MODES)) {
                assert_eq!(got, want, "{ctx} {env:?}\n{src}");
                walks[2] += 1;
            }
            assert_eq!(
                emu::run_emulated(&cc, &off.text, &dev(&off)),
                want,
                "{ctx} OFF"
            );
        }
        walks
    });
    let sum = |i: usize| counts.iter().map(|c| c[i]).sum::<usize>();
    eprintln!(
        "mma-walk values: {} walked GEMMs, {} walked epilogue groups, {} emulated runs",
        sum(0),
        sum(1),
        sum(2)
    );
    assert!(sum(0) >= 6 && sum(1) >= 1, "both stores valued");
}
