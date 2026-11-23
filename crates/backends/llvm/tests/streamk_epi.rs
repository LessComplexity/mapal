//! sweep-runtime-2 `streamk-epi`: an epilogue group (S54) keeps the stream-K
//! walk the rules give its site. Every piece adds its partials into a zeroed
//! workspace; the CTA that draws a tile's last ticket evaluates the group's
//! members on the whole sums. OFF arm `--gpu-off=streamk-epi`: the group's
//! ordered kernel (S54), which forfeits stream-K (sweep-runtime §1: the 2048
//! fan / MLP rows 4–9 %).
//!
//! Plans on generated GEMMs with cell-wise readers (`testgen::gemm_epi_src_at`:
//! an affine map, a bias + relu, a join, a two-member chain), at the
//! generator's extents from decode GEMVs to model-scale projections: the
//! group's combine is stream-K exactly where the same site without its group
//! (`--gpu-off=epilogue`) takes stream-K, never an atomic or workspace split,
//! and none under the OFF arm. Values through the GPU emulator (free threads,
//! serialized in thread order and reversed; blocks reversed) at runnable
//! extents where `k-fill` streams the site, against the interpreter and the
//! OFF arm (integer operands, dyadic constants: every cut prints the same
//! bits).

use mapal_backend_llvm::{CostOver, DeviceCost, Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::CategoryIr;
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use std::collections::HashMap;

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

/// The rtx-5080 emission with the rungs `off` off: the rules' plan only
/// (`--cands=1`), no launch-order rungs the emulator does not run, the
/// `--geom=` knobs `geom` forced.
fn emission(ir: &CategoryIr, off: &[&str], geom: &[&str]) -> Emission {
    emission_on(ir, off, geom, &[])
}

/// [`emission`] on the rtx-5080's properties with `props` replaced (a part
/// of 2 SMs: a stream-K walk over more tiles than blocks, at extents the
/// emulator runs).
fn emission_on(ir: &CategoryIr, off: &[&str], geom: &[&str], props: &[(&str, u64)]) -> Emission {
    emission_with(ir, off, geom, props, false)
}

/// [`emission_on`], the sites placed by the rules where `placed` (`place`
/// on: a split's zeroed output is a resident launch's — the shipped
/// emission, under which an under-filled site may stream by its price;
/// `--gpu-off=place` launches it unsplit).
fn emission_with(
    ir: &CategoryIr,
    off: &[&str],
    geom: &[&str],
    props: &[(&str, u64)],
    placed: bool,
) -> Emission {
    let mut cost: CostOver = [None; DeviceCost::NAMES.len()];
    for &(n, v) in props {
        cost[DeviceCost::NAMES.iter().position(|x| *x == n).expect(n)] = Some(v);
    }
    let mut gpu = GpuArms {
        site_cands: 1,
        ..GpuArms::default()
    };
    let place: &[&str] = if placed { &[] } else { &["place"] };
    for r in place
        .iter()
        .chain(&["pdl", "streams", "pinned", "horizontal", "calls"])
        .chain(off)
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let mut opts = EmitOpts {
        host: "generic",
        target: "rtx-5080",
        gpu,
        cost,
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

/// A grouped stream-K kernel's tiles: `(whole, shared)` — one block's
/// piece, or several blocks' (the walk of [`Kernel::ticket_split`]).
fn tile_kinds(e: &Emission) -> (u64, u64) {
    let num = |t: &HashMap<String, String>, k: &str| -> u64 { t[k].parse().expect(k) };
    let (mut whole, mut shared) = (0, 0);
    for t in tile_plans(e) {
        if t["epi"] != "1" || t["combine"] != "streamk" {
            continue;
        }
        let (panels, iters) = (num(&t, "k") / num(&t, "bk"), num(&t, "iters"));
        for x in 0..num(&t, "gx") * num(&t, "gy") {
            match x * panels / iters == ((x + 1) * panels - 1) / iters {
                true => whole += 1,
                false => shared += 1,
            }
        }
    }
    (whole, shared)
}

/// The combine of the grouped (`epi=1`) and plain (`epi=0`) kernels.
fn combines(e: &Emission, epi: &str) -> Vec<String> {
    tile_plans(e)
        .into_iter()
        .filter(|t| t["epi"] == epi)
        .map(|t| t["combine"].clone())
        .collect()
}

/// **The rule on generated shapes (plans).** A grouped site streams exactly
/// where its site alone streams, takes no other split, and its stream-K
/// kernel carries the ticket; the OFF arm keeps every group ordered.
#[test]
fn streamk_epi_rule_on_generated_shapes() {
    let mut runner = TestRunner::deterministic();
    let strat = testgen::gemm_prog_strategy();
    let progs: Vec<testgen::GemmProg> = (0..64)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    // Each program at the generator's extents, then (integration 3: the
    // merge with sweep-gemm-2 `streamk-price`) at drawn extents in the band
    // where an under-filled site streams by the part's rates — 768–1536 a
    // side, 512–4096 deep on the 5080's 168 slots — so a group's stream-K
    // walk is reached by each rule (`walk=class|fill|price`).
    let shapes: Vec<(usize, &testgen::GemmProg, bool)> = progs
        .iter()
        .enumerate()
        .flat_map(|(z, p)| [(z, p, false), (z, p, true)])
        .collect();
    let counts = testgen::par_check(&shapes, |_, &(z, p, band)| {
        let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
        let (m, n, k) = match band {
            false => testgen::gemm_dims(p),
            true => (
                pick(p.dims[0], &[768, 1024, 1536]),
                pick(p.dims[1], &[768, 1024, 1280]),
                pick(p.dims[2], &[512, 1024, 2048, 4096]),
            ),
        };
        let chain = (z % 4) as u8;
        let src = testgen::gemm_epi_src_at(p, m, n, k, chain);
        let ctx = format!("#{z} {m}x{n}x{k} chain {chain} band {band}");
        let ir = lower(&src);
        // The band's shapes under the shipped placement (`place` on).
        let emission =
            |ir: &CategoryIr, off: &[&str], geom: &[&str]| emission_with(ir, off, geom, &[], band);
        let on = emission(&ir, &[], &[]);
        let grouped = combines(&on, "1");
        if grouped.is_empty() {
            return [0; 6];
        }
        assert_eq!(grouped.len(), 1, "{ctx}: one group kernel {grouped:?}");
        let alone = emission(&ir, &["epilogue"], &[]);
        let plain = combines(&alone, "0");
        assert_eq!(plain.len(), 1, "{ctx}: one site kernel {plain:?}");
        let streams = plain[0] == "streamk";
        assert!(
            grouped[0] == "streamk" || grouped[0] == "none",
            "{ctx}: a group takes no atomic or workspace split ({grouped:?})"
        );
        assert_eq!(
            grouped[0] == "streamk",
            streams,
            "{ctx}: grouped {grouped:?}, alone {plain:?}"
        );
        let dev = on.device.as_deref().unwrap_or_default();
        assert_eq!(
            dev.contains("; streamk-epi: the tile's ticket"),
            streams,
            "{ctx}: the ticket where the group streams"
        );
        let off = emission(&ir, &["streamk-epi"], &[]);
        assert_eq!(combines(&off, "1"), vec!["none".to_string()], "{ctx} OFF");
        if !streams {
            return [0, 1, 1, 0, 0, 0];
        }
        // The group's walk is its site's (the plan line's `walk`): one wave
        // of CTAs over the launched tiles, every panel of every tile once.
        let t = tile_plans(&on)
            .into_iter()
            .find(|t| t["epi"] == "1")
            .expect("the group kernel");
        let walk = t.get("walk").cloned().unwrap_or_default();
        let num = |k: &str| t[k].parse::<u64>().expect(k);
        let (tiles, panels) = (num("gx") * num("gy"), num("k") / num("bk"));
        if walk == "price" {
            let (ctas, iters) = (num("ctas"), num("iters"));
            assert!(
                ctas * iters >= tiles * panels && (ctas - 1) * iters < tiles * panels,
                "{ctx} the priced walk covers its tiles once {t:?}"
            );
        }
        [
            1,
            0,
            1,
            usize::from(walk == "class"),
            usize::from(walk == "fill"),
            usize::from(walk == "price"),
        ]
    });
    let sum = |i: usize| counts.iter().map(|c| c[i]).sum::<usize>();
    let (tickets, ordered, groups) = (sum(0), sum(1), sum(2));
    let (class, fill, price) = (sum(3), sum(4), sum(5));
    eprintln!(
        "streamk-epi plans: {groups} groups, {tickets} stream-K tickets (walk class {class}, fill {fill}, \
         price {price}), {ordered} ordered"
    );
    assert!(
        tickets >= 4 && ordered >= 4,
        "both sides reached ({tickets}, {ordered})"
    );
    assert!(price >= 2, "a group at a priced walk reached ({price})");
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

/// **Values through the emulator.** Generated grouped products at extents
/// the emulator runs, every chain kind, two families: on the rtx-5080 (`m`
/// 64–128, `n` 256–512, `k` 512–1024: under-filled, streamed by `k-fill` at
/// a one-panel floor, `--geom=kfill=1` — every tile shared by several
/// blocks) and on its properties with 2 SMs (`m` 256–384, `n` 384–512,
/// `k` 256–512: the filled walk over more tiles than blocks — whole tiles
/// stored by their one block beside shared ones). The ticket kernel against
/// the interpreter under each emulator mode, and the OFF arm's ordered
/// group kernel (at `--cands=1` it launches without a table: the unsplit
/// geometry, not the stream-K walk's full tile).
#[test]
fn streamk_epi_values_through_the_emulator() {
    let Some(cc) = emu::clang() else {
        eprintln!("no clang: emulator runs skipped");
        return;
    };
    let mut runner = TestRunner::deterministic();
    let strat = testgen::gemm_prog_strategy();
    let progs: Vec<testgen::GemmProg> = (0..24)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let counts = testgen::par_check(&progs, |z, p| {
        let pick = |d: u8, v: &[u64]| v[usize::from(d) % v.len()];
        let small = z % 2 == 1;
        // integration 3: programs 16.. on the 5080's properties with 4 SMs
        // and `streamk-fill` off — the class rule's walk (`iters + 1 <
        // split`) over 6 tiles on 8 slots, many pieces a tile. A priced walk
        // (sweep-gemm-2 `streamk-price`) runs this same kernel: the rule
        // that chose stream-K only sets `(ctas, iters)`, which the walk's
        // pieces read alike (the plan test above reaches the priced walk).
        let priced = z >= 16;
        let (m, n, k) = match (priced, small) {
            (true, _) => (
                pick(p.dims[0], &[256, 384]),
                pick(p.dims[1], &[256, 384]),
                pick(p.dims[2], &[256, 512]),
            ),
            (_, false) => (
                pick(p.dims[0], &[64, 128]),
                pick(p.dims[1], &[256, 512]),
                pick(p.dims[2], &[512, 1024]),
            ),
            (_, true) => (
                pick(p.dims[0], &[256, 384]),
                pick(p.dims[1], &[384, 512]),
                pick(p.dims[2], &[256, 512]),
            ),
        };
        let chain = ((z / 2) % 4) as u8;
        let src = testgen::gemm_epi_src_at(p, m, n, k, chain);
        let ctx = format!("#{z} {m}x{n}x{k} chain {chain} small {small} priced {priced}");
        let ir = lower(&src);
        let (off0, geom, props): (&[&str], &[&str], &[(&str, u64)]) = match (priced, small) {
            (true, _) => (&["streamk-fill"], &[], &[("sms", 4)]),
            (_, false) => (&[], &["kfill=1"], &[]),
            (_, true) => (&[], &[], &[("sms", 2)]),
        };
        let on = emission_with(&ir, off0, geom, props, priced);
        if combines(&on, "1") != vec!["streamk".to_string()] {
            return [0, 0, 0, 0, 0];
        }
        let walk_class = tile_plans(&on)
            .iter()
            .any(|t| t["epi"] == "1" && t.get("walk").is_some_and(|w| w == "class"));
        let (whole, shared) = tile_kinds(&on);
        let off_arm: Vec<&str> = off0.iter().copied().chain(["streamk-epi"]).collect();
        let off = emission_with(&ir, &off_arm, geom, props, priced);
        let dev = |e: &Emission| e.device.clone().unwrap_or_default();
        let want = interp(&ir);
        assert!(want.lines().count() >= 5, "{ctx}\n{want}");
        assert_eq!(
            emu::run_emulated(&cc, &off.text, &dev(&off)),
            want,
            "{ctx} streamk-epi off"
        );
        let built = emu::build_emulated(&cc, &on.text, &dev(&on));
        let mut runs = 0;
        for (env, (got, _)) in MODES.iter().zip(built.run_each(&MODES)) {
            assert_eq!(got, want, "{ctx} streamk-epi {env:?}\n{src}");
            runs += 1;
        }
        [
            1,
            runs,
            whole as usize,
            shared as usize,
            usize::from(walk_class),
        ]
    });
    let sum = |i: usize| counts.iter().map(|c| c[i]).sum::<usize>();
    let (tickets, runs, whole, shared, class) = (sum(0), sum(1), sum(2), sum(3), sum(4));
    eprintln!(
        "streamk-epi values: {tickets} ticket programs ({whole} whole tiles, {shared} shared; {class} on the \
         class walk), {runs} emulated runs"
    );
    assert!(tickets >= 4, "the ticket valued on {tickets} programs");
    assert!(class >= 2, "the ticket valued on the class walk ({class})");
    assert!(
        whole >= 4 && shared >= 8,
        "both tile kinds valued ({whole}, {shared})"
    );
}
