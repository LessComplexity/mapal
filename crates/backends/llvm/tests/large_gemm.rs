//! large-gemm: the large-multiply geometry rules over physical properties.
//!
//! - `wide-fma` (`rules.rs`): the whole-file `8 × 16` tile on a contract face
//!   where the part issues more FFMA per DRAM byte than `FILL_BALANCE_PM`
//!   (the rtx-5080's 54.9) — OFF arm `--gpu-off=wide-fma`;
//! - `streamk-fill`: stream-K only over a launched tile grid that gives the
//!   part its fill target at its own geometry; a skinny grid (16 one-warp
//!   tiles) takes the `fill-wave` split — OFF arm `--gpu-off=streamk-fill`;
//! - `k-fill`: an under-filled site keeps the full S52 tile and fills the
//!   part along `k` by stream-K over every slot the tile holds (its
//!   residency, not `grid-share`'s one block an SM), each CTA folding at
//!   least `K_FILL_PANELS` panels — OFF arm `--gpu-off=k-fill`.
//!
//! Plans on generated GEMMs from decode GEMVs to model-scale projections,
//! odd extents included (`testgen::gemm_prog_strategy`, emitted only), with
//! coverage counters; values through the GPU emulator (free threads,
//! serialized in thread order, reversed) on generated programs at runnable
//! extents, ON against the OFF arms (integer operands: every cut of a fold
//! prints the same bits) and the interpreter where it is fast enough.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::CategoryIr;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use std::collections::HashMap;

mod emu;
#[path = "../../../mapal-rewrite/tests/testgen/mod.rs"]
mod testgen;

/// The rtx-5080's SMs and fill target (`resident` 2 warps × 4 schedulers × 84).
const SMS: u64 = 84;
const FILL_WARPS: u64 = 672;

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

/// The rtx-5080 emission with the rungs `off` off: the rules' plan only
/// (`--cands=1`), every capable site on the device, no launch-order rungs the
/// emulator does not run.
fn emission(ir: &CategoryIr, off: &[&str]) -> Emission {
    emission_at(ir, off, &[])
}

/// [`emission`] with `--geom=` knobs forced (`kfill=<panels>`).
fn emission_at(ir: &CategoryIr, off: &[&str], geom: &[&str]) -> Emission {
    let mut gpu = GpuArms {
        site_cands: 1,
        ..GpuArms::default()
    };
    for r in ["place", "pdl", "streams", "pinned", "horizontal", "calls"]
        .iter()
        .chain(off)
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let mut opts = EmitOpts {
        host: "generic",
        target: "rtx-5080",
        gpu,
        ..EmitOpts::default()
    };
    for g in geom {
        let f = format!("--geom={g}");
        assert!(mapal_backend_llvm::flags::apply(&mut opts, &f).expect("a knob"));
    }
    emit_detailed(ir, &opts).expect("emits")
}

/// Each tile kernel's `; mapal.plan` fields.
fn tile_plans(e: &Emission) -> Vec<HashMap<String, String>> {
    e.device
        .as_deref()
        .unwrap_or_default()
        .lines()
        .filter(|l| l.starts_with("; mapal.plan ") && l.contains(" kind=tile "))
        .map(|l| {
            l.split(' ')
                .filter_map(|f| f.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        })
        .collect()
}

fn num(p: &HashMap<String, String>, k: &str) -> u64 {
    p.get(k)
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("plan field {k}: {p:?}"))
}

fn warps(p: &HashMap<String, String>) -> u64 {
    num(p, "block").div_ceil(32)
}

/// **The rules on generated shapes (plans).** Every stream-K tile kernel is
/// either `k-fill`'s — an under-filled site (`wps < 2`) at the full S52 tile
/// (≥ 64 accumulators, ≥ 256 threads), its CTAs every resident slot (≥ 90 %
/// of `sms × blocks_per_sm`, one wave), ≥ 32 panels each — or a grid that
/// gives the part its fill target at its own geometry (`streamk-fill`).
/// The OFF arms: without `k-fill` no under-filled site streams; without
/// `streamk-fill` the grids it declined stream again (and its ON plan is the
/// `fill-wave` split there). The whole-file tile (`wide-fma`) holds its 128
/// accumulators where it is taken, and is gone under its OFF arm.
#[test]
fn large_gemm_rules_on_generated_shapes() {
    let mut runner = TestRunner::deterministic();
    let strat = testgen::gemm_prog_strategy();
    let progs: Vec<testgen::GemmProg> = (0..96)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let counts = testgen::par_check(&progs, |z, p| {
        let (mut kfill, mut filled_sk, mut declined, mut wide, mut shrunk_off) = (0, 0, 0, 0, 0);
        let mut odd_plans = 0;
        let (src, (m, n, k)) = testgen::gemm_src(p);
        let ctx = format!("#{z} {m}x{n}x{k} bt={}", p.bt);
        let ir = lower(&src);
        let on = emission(&ir, &[]);
        let on_p = tile_plans(&on);
        if p.scale == 3 {
            odd_plans += on_p.len();
        }
        // sweep-gemm-2 `streamk-price` (its own test below): an under-filled
        // site's priced walk over the launched grid — the stream-K kernels
        // its OFF arm does not have.
        let unpriced = tile_plans(&emission(&ir, &["streamk-price"]));
        assert_eq!(unpriced.len(), on_p.len(), "{ctx} the same tile kernels");
        for (t, u) in on_p.iter().zip(&unpriced) {
            if t.get("combine").map(String::as_str) == Some("streamk") {
                let (wps, ctas, iters) = (num(t, "wps"), num(t, "ctas"), num(t, "iters"));
                let slots = SMS * num(t, "blocks_per_sm");
                if wps < 2 && u["combine"] != "streamk" {
                    continue;
                }
                if wps < 2 {
                    kfill += 1;
                    assert!(num(t, "tm") * num(t, "tn") >= 64, "{ctx} {t:?}");
                    assert!(num(t, "block") >= 256, "{ctx} {t:?}");
                    assert!(iters >= 32, "{ctx} iters {iters}");
                    assert!(
                        ctas <= slots && 10 * ctas >= 9 * slots,
                        "{ctx} {ctas} of {slots}"
                    );
                    assert!(
                        t["rungs_on"].contains("k-fill"),
                        "{ctx} k-fill on its plan line"
                    );
                } else {
                    filled_sk += 1;
                    assert!(
                        num(t, "gx") * num(t, "gy") * warps(t) >= FILL_WARPS,
                        "{ctx} stream-K over a grid short of the fill target {t:?}"
                    );
                }
            }
            if num(t, "held") > 0 {
                wide += 1;
                assert_eq!(num(t, "tm") * num(t, "tn"), 128, "{ctx} {t:?}");
            }
        }
        // `k-fill` OFF: no under-filled site streams; its kernel is the
        // shrunk tile or threads the rules derive.
        let had_kfill = on_p
            .iter()
            .any(|t| t["combine"] == "streamk" && num(t, "wps") < 2);
        if had_kfill {
            // (`streamk-price` off too: its own under-filled walks.)
            let off = tile_plans(&emission(&ir, &["k-fill", "streamk-price"]));
            for t in &off {
                assert!(
                    !(t["combine"] == "streamk" && num(t, "wps") < 2),
                    "{ctx} k-fill off still streams {t:?}"
                );
                if num(t, "tm") * num(t, "tn") < 64 || num(t, "block") < 256 {
                    shrunk_off += 1;
                }
            }
        }
        // `streamk-fill` OFF: a grid short of its fill target streams again
        // (one block an SM, or a whole-file tile's half-warp blocks); ON took
        // the `fill-wave` split there, or the S52 tile whose grid fills.
        let off = tile_plans(&emission(&ir, &["streamk-fill"]));
        for (t, o) in on_p.iter().zip(&off) {
            let short = num(o, "gx") * num(o, "gy") * warps(o) < FILL_WARPS;
            if o["combine"] == "streamk" && num(o, "wps") >= 2 && short {
                declined += 1;
                let split =
                    (t["combine"] == "atomic" || t["combine"] == "splitk") && num(t, "S") > 1;
                let fills = t["combine"] == "streamk"
                    && num(t, "held") == 0
                    && num(t, "gx") * num(t, "gy") * warps(t) >= FILL_WARPS;
                assert!(split || fills, "{ctx} streamk-fill: {t:?}");
            }
        }
        // `wide-fma` OFF: no whole-file tile.
        if on_p.iter().any(|t| num(t, "held") > 0) {
            for t in tile_plans(&emission(&ir, &["wide-fma"])) {
                assert_eq!(num(&t, "held"), 0, "{ctx} wide-fma off {t:?}");
            }
        }
        [kfill, filled_sk, declined, wide, shrunk_off, odd_plans]
    });
    let sum = |i: usize| counts.iter().map(|c| c[i]).sum::<usize>();
    let (kfill, filled_sk, declined, wide, shrunk_off, odd_plans) =
        (sum(0), sum(1), sum(2), sum(3), sum(4), sum(5));
    eprintln!(
        "large-gemm plans: {kfill} k-fill stream-K kernels ({shrunk_off} shrunk under the OFF arm), \
         {filled_sk} filled stream-K, {declined} stream-K grids declined for the fill split, \
         {wide} whole-file tiles, {odd_plans} tile kernels at odd extents"
    );
    assert!(kfill >= 4, "the generator reaches k-fill ({kfill})");
    assert!(shrunk_off >= 4, "k-fill's OFF arm shrinks ({shrunk_off})");
    assert!(
        declined >= 2,
        "the generator reaches streamk-fill ({declined})"
    );
    assert!(filled_sk >= 2, "filled stream-K stays ({filled_sk})");
    assert!(wide >= 2, "the generator reaches wide-fma ({wide})");
}

const MODES: [[(&str, &str); 2]; 3] = [
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "1")],
];

fn interp(ir: &CategoryIr) -> String {
    let rr = run(ir, 20_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{:?}", rr.outcome);
    rr.output
}

/// **Values through the emulator.** Generated programs where each rule acts
/// at extents the emulator runs: skinny products (`m` 16, `n` 2048: 16
/// one-warp tiles, the `fill-wave` split under `streamk-fill`) and
/// under-filled products (`m` 64–128, `n` 256–1024, `k` 1024–2048, the
/// full tile's stream-K under `k-fill` — its panel floor forced to one,
/// `--geom=kfill=1`: at the rule's 32 panels a CTA the smallest such site
/// on 84 SMs folds 1.4 G products, past what the emulator runs), each
/// against the interpreter, ON under free threads, serialized in thread
/// order and reversed; the OFF arm too (integer operands: every cut of a
/// fold prints the same bits).
#[test]
fn large_gemm_values_through_the_emulator() {
    let Some(cc) = emu::clang() else {
        eprintln!("no clang: emulator runs skipped");
        return;
    };
    let mut runner = TestRunner::deterministic();
    let strat = testgen::gemm_prog_strategy();
    let progs: Vec<testgen::GemmProg> = (0..12)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let counts = testgen::par_check(&progs, |z, p| {
        let (mut kfill, mut fill_split, mut runs) = (0, 0, 0);
        let deep = z % 2 == 1;
        let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
        let (m, n, k) = match deep {
            true => (
                pick(p.dims[0], &[64, 128]),
                pick(p.dims[1], &[256, 512]),
                pick(p.dims[2], &[512, 1024]),
            ),
            false => (16, 2048, pick(p.dims[2], &[256, 512])),
        };
        let src = testgen::gemm_src_at(p, m, n, k);
        let ctx = format!("#{z} {m}x{n}x{k} bt={}", p.bt);
        let ir = lower(&src);
        let geom: &[&str] = if deep { &["kfill=1"] } else { &[] };
        let on = emission_at(&ir, &[], geom);
        let plans = tile_plans(&on);
        let rule = if deep { "k-fill" } else { "streamk-fill" };
        let acted = plans.iter().any(|t| match deep {
            true => t["combine"] == "streamk" && num(t, "wps") < 2,
            false => t["combine"] == "atomic" && num(t, "S") > 1,
        });
        if !acted {
            return [0, 0, 0];
        }
        match deep {
            true => kfill += 1,
            false => fill_split += 1,
        }
        let off = emission_at(&ir, &[rule], geom);
        let dev = |e: &Emission| e.device.clone().unwrap_or_default();
        let want = interp(&ir);
        assert_eq!(want.lines().count(), 5, "{ctx}\n{want}");
        assert_eq!(
            emu::run_emulated(&cc, &off.text, &dev(&off)),
            want,
            "{ctx} {rule} off"
        );
        let built = emu::build_emulated(&cc, &on.text, &dev(&on));
        for (env, (got, _)) in MODES.iter().zip(built.run_each(&MODES)) {
            assert_eq!(got, want, "{ctx} {rule} {env:?}\n{src}");
            runs += 1;
        }
        [kfill, fill_split, runs]
    });
    let sum = |i: usize| counts.iter().map(|c| c[i]).sum::<usize>();
    let (kfill, fill_split, runs) = (sum(0), sum(1), sum(2));
    eprintln!(
        "large-gemm values: {kfill} k-fill, {fill_split} fill-split programs, {runs} emulated runs"
    );
    assert!(
        kfill >= 2 && fill_split >= 2,
        "both rules valued ({kfill}, {fill_split})"
    );
}

/// [`emission_at`] with plan rates overridden (`--cost=<rate>=<n>`), the
/// sites placed by the rules (`place` on: a split's zeroed output is a
/// resident launch's; `--gpu-off=place` launches every site unsplit).
fn emission_cost(ir: &CategoryIr, off: &[&str], cost: &[&str]) -> Emission {
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
    let mut opts = EmitOpts {
        host: "generic",
        target: "rtx-5080",
        gpu,
        ..EmitOpts::default()
    };
    for c in cost {
        let f = format!("--cost={c}");
        assert!(mapal_backend_llvm::flags::apply(&mut opts, &f).expect("a rate"));
    }
    emit_detailed(ir, &opts).expect("emits")
}

/// **sweep-gemm-2 `streamk-price` on generated shapes.** An under-filled
/// site (`wps < 2`, its class ships `streamk` off) streams over its launched
/// grid only where the part's rates price the walk (the pieces' partials and
/// the zero pass) below the S63 count: plans on the generated GEMMs from
/// decode GEMVs to model-scale projections — each such kernel is gone under
/// `--gpu-off=streamk-price` (the panel-count rule of the classes that ship
/// `streamk`), keeps the launched tile and grid (`gx·gy` tiles), walks every
/// slot once (`ctas ≤ sms × blocks_per_sm`) and folds at least one panel a
/// CTA; and a part without the `red` rate (`--cost=atomic_ops_per_ns=0`) is
/// priced on the one its properties derive (review-fixes-2, `Gpu::plan_cost`:
/// the L2 rate's adds per byte — on the 5080, the reference part, exactly its
/// measured rate), so it streams the same sites: never the old rule where
/// unmeasured. Values through the emulator at
/// extents it runs (256–512 each side), the `red` rate raised and the zero
/// pass zeroed (`--cost=`) so the walk pays where the grid quantizes badly:
/// ON (free threads, serialized in thread order, reversed) against the
/// interpreter and the OFF arm (integer operands: every cut of a fold
/// prints the same bits) — `streamk-fill` off on both arms: at these extents
/// no tile grid gives the part its fill warps, and the walk under test is
/// the same kernel.
#[test]
fn streamk_price_on_generated_shapes() {
    let mut runner = TestRunner::deterministic();
    let strat = testgen::gemm_prog_strategy();
    let progs: Vec<testgen::GemmProg> = (0..96)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let counts = testgen::par_check(&progs, |z, p| {
        // Every other program at drawn extents in the band where the rule
        // acts — a tile grid that fills the part's warps (`streamk-fill`)
        // yet holds fewer tiles than its slots (768–1536 a side on the
        // 5080's 168 slots of `128 × 64` tiles), 32–256 panels: the
        // generator's own extents (decode GEMVs to model-scale) reach it
        // rarely.
        let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
        let (src, (m, n, k)) = match z % 2 {
            0 => testgen::gemm_src(p),
            _ => {
                let (m, n, k) = (
                    pick(p.dims[0], &[768, 1024, 1536]),
                    pick(p.dims[1], &[768, 1024, 1280]),
                    pick(p.dims[2], &[512, 1024, 2048, 4096]),
                );
                (testgen::gemm_src_at(p, m, n, k), (m, n, k))
            }
        };
        let ctx = format!("#{z} {m}x{n}x{k} bt={}", p.bt);
        let ir = lower(&src);
        let on = tile_plans(&emission_cost(&ir, &[], &[]));
        let off = tile_plans(&emission_cost(&ir, &["streamk-price"], &[]));
        let bare = tile_plans(&emission_cost(&ir, &[], &["atomic_ops_per_ns=0"]));
        assert_eq!(on.len(), off.len(), "{ctx}");
        let (mut priced, mut under) = (0, 0);
        for ((t, o), b) in on.iter().zip(&off).zip(&bare) {
            if num(t, "wps") >= 2 {
                continue;
            }
            under += 1;
            let (t_sk, o_sk) = (t["combine"] == "streamk", o["combine"] == "streamk");
            assert_eq!(
                b["combine"], t["combine"],
                "{ctx} the derived `red` rate prices as the measured {b:?}"
            );
            if !t_sk || o_sk {
                continue;
            }
            priced += 1;
            let (ctas, iters) = (num(t, "ctas"), num(t, "iters"));
            assert_eq!(
                (num(t, "gx"), num(t, "gy"), num(t, "block")),
                (num(o, "gx"), num(o, "gy"), num(o, "block")),
                "{ctx} the launched tile and grid {t:?}"
            );
            let tiles = num(t, "gx") * num(t, "gy");
            let panels = num(t, "k") / num(t, "bk");
            assert!(iters >= 1 && iters < panels * tiles, "{ctx} {t:?}");
            assert!(
                ctas <= SMS * num(t, "blocks_per_sm"),
                "{ctx} one wave {t:?}"
            );
            assert_eq!(ctas, (tiles * panels).div_ceil(iters), "{ctx} {t:?}");
        }
        [priced, under]
    });
    let sum = |i: usize| counts.iter().map(|c| c[i]).sum::<usize>();
    let (priced, under) = (sum(0), sum(1));
    eprintln!("streamk-price plans: {priced} priced walks of {under} under-filled tile kernels");
    assert!(
        priced >= 2,
        "the generator reaches streamk-price ({priced})"
    );

    // Values.
    let Some(cc) = emu::clang() else {
        eprintln!("no clang: emulator runs skipped");
        return;
    };
    let cheap = ["atomic_ops_per_ns=4000000", "zero_ns=1"];
    let progs: Vec<testgen::GemmProg> = (0..10)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let counts = testgen::par_check(&progs, |z, p| {
        let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
        let (m, n, k) = (
            pick(p.dims[0], &[256, 512]),
            pick(p.dims[1], &[256, 512]),
            pick(p.dims[2], &[256, 512]),
        );
        let src = testgen::gemm_src_at(p, m, n, k);
        let ctx = format!("#{z} {m}x{n}x{k} bt={}", p.bt);
        let ir = lower(&src);
        let on = emission_cost(&ir, &["streamk-fill"], &cheap);
        let off = emission_cost(&ir, &["streamk-fill", "streamk-price"], &cheap);
        let walks = tile_plans(&on)
            .iter()
            .zip(&tile_plans(&off))
            .filter(|(t, o)| {
                t["combine"] == "streamk" && num(t, "wps") < 2 && o["combine"] != "streamk"
            })
            .count();
        if walks == 0 {
            return [0, 0];
        }
        let dev = |e: &Emission| e.device.clone().unwrap_or_default();
        let want = interp(&ir);
        assert_eq!(want.lines().count(), 5, "{ctx}\n{want}");
        assert_eq!(
            emu::run_emulated(&cc, &off.text, &dev(&off)),
            want,
            "{ctx} streamk-price off"
        );
        let built = emu::build_emulated(&cc, &on.text, &dev(&on));
        let mut runs = 0;
        for (env, (got, _)) in MODES.iter().zip(built.run_each(&MODES)) {
            assert_eq!(got, want, "{ctx} streamk-price {env:?}\n{src}");
            runs += 1;
        }
        [1, runs]
    });
    let (valued, runs) = (
        counts.iter().map(|c| c[0]).sum::<usize>(),
        counts.iter().map(|c| c[1]).sum::<usize>(),
    );
    eprintln!("streamk-price values: {valued} programs, {runs} emulated runs");
    assert!(valued >= 2, "priced walks valued ({valued})");
}
