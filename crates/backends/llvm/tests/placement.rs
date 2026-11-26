//! model-ready `placement`: a decode step's work stays where its data lives.
//!
//! - **Views are their values** (rewriter): a `Proj` of a pack — an inlined
//!   call's argument tuple, a struct's field — forwards its feeder inside a
//!   loop too, and a loop state slot that is itself a struct of arrays is
//!   carried as its leaves (`mut kv: Cache`), so every fact sees the arrays.
//! - **Facts** (`mapal-ir`): sibling views of one input slot share a
//!   condition's narrowing (a rotation's partner `i ± w/2` under `i % w <
//!   w/2` is in bounds); a selecting fold's index component is ranged by its
//!   seed and items (an argmax token gathers an embedding row in bounds); a
//!   piecewise map's interval through a capture the site passes as a
//!   constant (an inlined layer's index) is affine in the other one; a
//!   nested fold's read of a captured array has its stride along the fold.
//! - **Rules** (backend): a cell a loop runs every trip is placed by the
//!   bytes it moves (a made-once operand is uploaded once); a chain of
//!   in-place range maps over a carried state shares the carry's buffer (a
//!   trip's appends to one cache); a nest cell whose fold reads an operand
//!   at unit stride takes a warp a cell.
//! - **Generated programs** (`testgen::decode_strategy`): decode-like loops
//!   — layers with a literal index, rotated keys, appends to a struct cache,
//!   attention over the filled rows, an argmax token's embedding row —
//!   against the interpreter in every emulator mode, on the CPU and with a
//!   placement rung off; the loop moves nothing to the device per trip.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::{CategoryIr, Ty};
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

fn interp(ir: &CategoryIr) -> String {
    let rr = run(ir, 2_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{:?}", rr.outcome);
    rr.output
}

/// The GPU emission on the rtx-5080 with the emulator's rungs off, and
/// `off` off too.
fn device_emission(ir: &CategoryIr, off: &[&str]) -> Emission {
    let mut gpu = GpuArms::default();
    for r in ["async", "pdl", "streams", "pinned", "horizontal", "calls"]
        .iter()
        .chain(off)
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "generic",
        target: "rtx-5080",
        gpu,
        ..EmitOpts::default()
    };
    emit_detailed(ir, &opts).expect("emits")
}

const MODES: [[(&str, &str); 2]; 4] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_REVERSE", "0")],
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_REVERSE", "1")],
];

/// The entry's loop movement lines (`; mapal.move loop=…`).
fn moves(em: &Emission) -> Vec<&str> {
    em.text
        .lines()
        .filter(|l| l.starts_with("; mapal.move loop=") && !l.contains("carried=-"))
        .collect()
}

fn check(clang: &str, i: usize, g: &testgen::DecodeProg) -> Vec<(&'static str, u64)> {
    let src = testgen::decode_prog_src(g);
    let ir = lower(&src);
    let want = interp(&ir);
    let mut c = vec![("programs", 1)];
    // The struct state is carried as its leaves: no loop merge of `main`
    // holds a product in a slot.
    let f = ir.entry();
    let flat = ir.loop_structure(f).iter().all(|scc| {
        scc.merges
            .iter()
            .all(|&m| match ir.object(m).map(|x| &x.ty) {
                Some(Ty::Tuple(parts)) => !parts
                    .iter()
                    .any(|t| matches!(t, Ty::Tuple(_) | Ty::Struct { .. })),
                _ => true,
            })
    });
    assert!(flat, "a struct slot stays in the loop state\n{src}");
    c.push(("flat-states", 1));
    let em = device_emission(&ir, &[]);
    let dev = em.device.as_deref().unwrap_or("");
    let mv = moves(&em);
    // Every trip's work on the device: nothing crosses up per trip, and
    // the cache's leaves are carried there.
    if mv.iter().any(|l| l.contains(":device")) {
        c.push(("device-carries", 1));
    }
    if mv.iter().all(|l| l.contains(" h2d=0 ")) {
        c.push(("trip-h2d-zero", 1));
        if g.rot {
            c.push(("rotated-on-device", 1));
        }
    }
    if dev.contains(" range=") {
        c.push(("range-cells", 1));
        if g.layers == 2 {
            c.push(("range-chains", 1));
        }
    }
    if dev.contains(" lanes=32") {
        c.push(("lane-cells", 1));
    }
    if dev.contains("kind=cell uniform=1 nested=0") {
        c.push(("gather-cells", 1));
    }
    // `lanes-unroll`: every warp-a-cell fold loop's placeholder became its
    // own loop id, unrolled `NAIVE_UNROLL` trips.
    assert!(
        !dev.contains("@UNROLL@"),
        "a loop id left unnumbered\n{src}"
    );
    let unrolled = dev.matches("llvm.loop.unroll.count").count() as u64;
    if unrolled > 0 {
        c.push(("unrolled-lane-loops", unrolled));
    }
    // residency `fly-unroll`: a warp-a-cell fold over a grid too small to
    // keep the part's stream bytes in flight at 8 loads a lane unrolls
    // deeper (its own body variant, its own loop count).
    let deep = ["16", "32"]
        .iter()
        .map(|n| {
            dev.matches(&format!("\"llvm.loop.unroll.count\", i32 {n}}}"))
                .count() as u64
        })
        .sum::<u64>();
    if deep > 0 {
        assert!(
            dev.contains("_lanes_u"),
            "a deeper loop without its variant\n{src}"
        );
        c.push(("fly-unrolled-loops", deep));
    }
    let ex = emu::build_emulated(clang, &em.text, dev);
    for (mode, (got, _)) in MODES.iter().zip(ex.run_each(&MODES)) {
        assert_eq!(got, want, "{mode:?}\n{src}");
    }
    let cpu = emu::run_emulated(
        clang,
        &emit_detailed(&ir, &EmitOpts::default()).unwrap().text,
        "",
    );
    assert_eq!(cpu, want, "cpu\n{src}");
    if i % 3 == 0 {
        let arm = [
            "range-cell",
            "devmap",
            "cell-lanes",
            "lanes-unroll",
            "fly-unroll",
        ][(i / 3) % 5];
        let off = device_emission(&ir, &[arm]);
        let off_dev = off.device.as_deref().unwrap_or("");
        if arm == "lanes-unroll" {
            assert!(
                !off_dev.contains("llvm.loop.unroll.count"),
                "{arm} off\n{src}"
            );
        }
        if arm == "fly-unroll" {
            assert!(
                !off_dev.contains("_lanes_u") && !off_dev.contains("unroll.count\", i32 32"),
                "{arm} off\n{src}"
            );
        }
        let got = emu::run_emulated(clang, &off.text, off_dev);
        assert_eq!(got, want, "{arm} off\n{src}");
        c.push(("off-arm-runs", 1));
    }
    c
}

/// Generated decode-like loops (`testgen::decode_strategy`), each on the
/// emulator in every mode, the CPU build and with a rung off — against the
/// interpreter. Every path is reached.
#[test]
fn placement_on_generated_programs() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::DecodeProg> = std::iter::repeat_with(|| {
        testgen::decode_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current()
    })
    .take(24)
    .collect();
    let results = testgen::par_check(&progs, |i, g| check(&clang, i, g));
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    for r in results.into_iter().flatten() {
        *c.entry(r.0).or_default() += r.1;
    }
    eprintln!("placement generated: {c:?}");
    for key in [
        "programs",
        "flat-states",
        "device-carries",
        "trip-h2d-zero",
        "rotated-on-device",
        "range-cells",
        "range-chains",
        "lane-cells",
        "gather-cells",
        "unrolled-lane-loops",
        "fly-unrolled-loops",
        "off-arm-runs",
    ] {
        assert!(c.get(key).copied().unwrap_or(0) > 0, "{key}: {c:?}");
    }
}

/// The host lines of an emission's `; mapal.range once=` intent, by kind.
fn once_ranges(em: &Emission) -> (u64, u64) {
    let n = |k: &str| em.text.matches(&format!("; mapal.range once={k} ")).count() as u64;
    (n("inplace"), n("whole"))
}

/// Calls moving `bytes` between host and device (either way) in the host text.
fn crossings(em: &Emission, bytes: u64) -> usize {
    em.text
        .lines()
        .filter(|l| {
            (l.contains("@mapal_gpu_h2d(") || l.contains("@mapal_gpu_d2h("))
                && l.trim_end().ends_with(&format!("i64 {bytes})"))
        })
        .count()
}

fn check_prefill(clang: &str, i: usize, g: &testgen::PrefillProg) -> Vec<(&'static str, u64)> {
    let src = testgen::prefill_prog_src(g);
    let ir = lower(&src);
    let want = interp(&ir);
    let mut c = vec![("programs", 1)];
    let em = device_emission(&ir, &[]);
    let dev = em.device.as_deref().unwrap_or("");
    let (inplace, whole) = once_ranges(&em);
    // Every prompt write runs once on the device: over its dead source's
    // buffer, or (its source read again: the second cache's shared zero
    // fill, a reread layer) whole into its own.
    let writes = 2 * g.layers;
    let read_again = u64::from(g.shared) + u64::from(g.reread && g.layers == 2);
    assert_eq!(
        inplace + whole,
        writes,
        "prompt writes on the device\n{src}"
    );
    assert_eq!(whole, read_again, "whole-extent writes\n{src}");
    c.push(("inplace-writes", inplace));
    if whole > 0 {
        c.push(("whole-writes", whole));
    }
    // Movement: no cache crosses the boundary — the fills are zeroed on the
    // device, the prompt's writes stay there, the decode loop takes their
    // buffers.
    let cache_bytes = g.layers * g.p * g.w * 4;
    assert_eq!(crossings(&em, cache_bytes), 0, "a cache crosses\n{src}");
    c.push(("no-cache-copy", 1));
    if g.trips > 0 {
        c.push(("decode-after-prompt", 1));
    }
    let ex = emu::build_emulated(clang, &em.text, dev);
    for (mode, (got, _)) in MODES.iter().zip(ex.run_each(&MODES)) {
        assert_eq!(got, want, "{mode:?}\n{src}");
    }
    if i % 2 == 0 {
        // The OFF arm: no straight-line range cell at all.
        let off = device_emission(&ir, &["once-range"]);
        assert_eq!(once_ranges(&off), (0, 0), "once-range off\n{src}");
        let got = emu::run_emulated(clang, &off.text, off.device.as_deref().unwrap_or(""));
        assert_eq!(got, want, "once-range off\n{src}");
        c.push(("off-arm-runs", 1));
    } else {
        let cpu = emu::run_emulated(
            clang,
            &emit_detailed(&ir, &EmitOpts::default()).unwrap().text,
            "",
        );
        assert_eq!(cpu, want, "cpu\n{src}");
        c.push(("cpu-runs", 1));
    }
    c
}

/// full-model `once-range` on generated prompt chunks
/// (`testgen::prefill_strategy`): a range map outside every loop writes its
/// rows over its source's device buffer when nothing reads the source after
/// it, else the whole extent into its own; no cache crosses to the host,
/// and the decode loop after the prompt takes the prompt's buffers. Values
/// against the interpreter in every emulator mode, the CPU build and the OFF
/// arm; every path is reached.
#[test]
fn once_range_on_generated_prompts() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::PrefillProg> = std::iter::repeat_with(|| {
        testgen::prefill_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current()
    })
    .take(24)
    .collect();
    let results = testgen::par_check(&progs, |i, g| check_prefill(&clang, i, g));
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    for r in results.into_iter().flatten() {
        *c.entry(r.0).or_default() += r.1;
    }
    eprintln!("once-range generated: {c:?}");
    for key in [
        "programs",
        "inplace-writes",
        "whole-writes",
        "no-cache-copy",
        "decode-after-prompt",
        "off-arm-runs",
        "cpu-runs",
    ] {
        assert!(c.get(key).copied().unwrap_or(0) > 0, "{key}: {c:?}");
    }
}

/// The GPU emission against a host with measured costs (`x86-64-v3`'s
/// dispatch row): `place` prices every site and cell by its roofline on
/// both sides, so what stays on the device is what the residency facts
/// prove is there, not what an unpriced build places by default.
fn priced_emission(ir: &CategoryIr) -> Emission {
    let mut gpu = GpuArms::default();
    for r in ["async", "pdl", "streams", "pinned", "horizontal", "calls"] {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "x86-64-v3",
        target: "rtx-5080",
        gpu,
        ..EmitOpts::default()
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// `; mapal.host`'s `host_sites=` count: capable tile sites `place` kept on
/// the host.
fn host_sites(em: &Emission) -> u64 {
    em.text
        .lines()
        .find_map(|l| l.strip_prefix("; mapal.host "))
        .and_then(|l| {
            l.split_whitespace()
                .find_map(|w| w.strip_prefix("host_sites="))
        })
        .and_then(|n| n.parse().ok())
        .expect("a host line")
}

fn check_carry(clang: &str, i: usize, g: &testgen::CarryProg) -> Vec<(&'static str, u64)> {
    let src = testgen::carry_prog_src(g);
    let ir = lower(&src);
    let want = interp(&ir);
    let mut c = vec![("programs", 1)];
    let em = priced_emission(&ir);
    let dev = em.device.as_deref().unwrap_or("");
    // The trip's small site reads the carried state where the large site
    // made it: placed on the device (fed), nothing crosses per trip.
    assert_eq!(host_sites(&em), 0, "a site left on the host\n{src}");
    c.push(("fed-by-carry", 1));
    let mv = moves(&em);
    assert!(
        mv.iter().any(|l| l.contains(":device")
            && (g.row || l.contains(" h2d=0 "))
            && l.contains(" d2h=0 ")),
        "the loop's state crosses\n{}\n{src}",
        mv.join("\n")
    );
    c.push(("trip-moves-zero", 1));
    if g.mix {
        c.push(("mixed-next", 1));
    }
    if g.second {
        c.push(("second-carry", 1));
    }
    if g.row {
        // The one-row projection runs on the device (a warp-a-row nest
        // cell), beside the state it feeds.
        assert!(
            dev.contains("nested=2 work="),
            "the row's projection left the device\n{src}"
        );
        // residency `shared-upload`: the gathered row crosses at most once
        // a trip, for the state's map (beside the projected row the host
        // makes); its norm runs beside it on the device and adds no upload
        // of its own (the scaled row never crosses).
        let row = g.k * 4;
        let trip_h2d = mv
            .iter()
            .filter_map(|l| l.split_whitespace().find_map(|w| w.strip_prefix("h2d=")))
            .filter_map(|n| n.parse::<u64>().ok())
            .max()
            .unwrap_or(0);
        assert!(
            trip_h2d <= 2 * row + 4,
            "the trip uploads more than its row\n{}\n{src}",
            mv.join("\n")
        );
        c.push(("row-trips", 1));
    }
    let ex = emu::build_emulated(clang, &em.text, dev);
    for (mode, (got, _)) in MODES.iter().zip(ex.run_each(&MODES)) {
        assert_eq!(got, want, "{mode:?}\n{src}");
    }
    if i % 2 == 0 {
        // The OFF arm (`carry-made`: no carried view is device-made): the
        // priced rule alone decides, and the values are the same.
        let mut gpu = GpuArms::default();
        for r in [
            "async",
            "pdl",
            "streams",
            "pinned",
            "horizontal",
            "calls",
            "carry-made",
        ] {
            *gpu.off.rung(r).expect("rung") = true;
        }
        let opts = EmitOpts {
            host: "x86-64-v3",
            target: "rtx-5080",
            gpu,
            ..EmitOpts::default()
        };
        let off = emit_detailed(&ir, &opts).expect("emits");
        let got = emu::run_emulated(clang, &off.text, off.device.as_deref().unwrap_or(""));
        assert_eq!(got, want, "carry-made off\n{src}");
        c.push(("off-arm-runs", 1));
    }
    c
}

/// residency `carry-induct`, on generated loops (`testgen::carry_strategy`)
/// placed against measured costs: a carried array that enters the loop made
/// on the device (a large site's product) and is rewritten each trip from
/// itself is device-made on every trip, by induction — so the trip's small
/// site that reads it is fed and runs where it is, instead of the host's
/// roofline pulling it (and the state) off the device. Values against the
/// interpreter in every emulator mode and with the `carry-made` arm off.
#[test]
fn priced_placement_follows_the_carried_state() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::CarryProg> = std::iter::repeat_with(|| {
        testgen::carry_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current()
    })
    .take(16)
    .collect();
    let results = testgen::par_check(&progs, |i, g| check_carry(&clang, i, g));
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    for r in results.into_iter().flatten() {
        *c.entry(r.0).or_default() += r.1;
    }
    eprintln!("priced carry generated: {c:?}");
    for key in [
        "programs",
        "fed-by-carry",
        "trip-moves-zero",
        "mixed-next",
        "second-carry",
        "row-trips",
        "off-arm-runs",
    ] {
        assert!(c.get(key).copied().unwrap_or(0) > 0, "{key}: {c:?}");
    }
}

/// A prompt's head, straight-line: `n` rows gathered from a table through
/// run-time ids no proof bounds (`(x > n + 5)` never taken — a
/// `trap_cell_plan` cell), each row's sum of squares (a row fold), the rows
/// scaled by it and a gain (`norm`; else the rows go on as gathered), one
/// projection to 64 columns (a tile site), and with `res` the gathered rows'
/// first 64 columns added back (a residual). `(r, w, n)`: the table's rows, the width, the
/// prompt's rows.
fn prompt_head_src(r: u64, w: u64, n: u64, norm: bool, res: bool) -> String {
    let mut s = format!(
        "fn main() {{\n    {rw} -> iota -> i_t;\n    i_t -> map {{\n        x -> (x * 7 + 3) % 13 - 6 -> widen<f32> -> v;\n        v / 4.0\n    }} -> table;\n    {n} -> iota -> i_ids;\n    i_ids -> map {{\n        x -> (x > {far}) -> {{\n            -true-> 100000;\n            -false-> (x * 5 + 3) % {r};\n        }}\n    }} -> ids;\n    {nw} -> iota -> map {{\n        cell -> table[ids[cell / {w}] * {w} + cell % {w}]\n    }} -> rows;\n    {w} -> iota -> rw;\n    {w} -> iota -> i_g;\n    i_g -> map {{\n        x -> (x * 3 + 1) % 5 - 2 -> widen<f32> -> v;\n        v / 2.0 + 1.0\n    }} -> gain;\n    {ww} -> iota -> i_wt;\n    i_wt -> map {{\n        x -> (x * 5 + 2) % 9 - 4 -> widen<f32> -> v;\n        v / 16.0\n    }} -> weight;\n",
        rw = r * w,
        nw = n * w,
        ww = w * 64,
        far = n + 5,
    );
    let x = if norm {
        s += &format!(
            "    {n} -> iota -> map {{\n        t -> (0.0, rw) -> fold {{\n            sum, i -> sum + rows[t * {w} + i] * rows[t * {w} + i]\n        }}\n    }} -> squares;\n    {nw} -> iota -> map {{\n        cell -> squares[cell / {w}] / {w}.0 + 0.00001 -> sqrt -> root;\n        rows[cell] * (1.0 / root) * gain[cell % {w}]\n    }} -> normed;\n",
            nw = n * w
        );
        "normed"
    } else {
        "rows"
    };
    s += &format!(
        "    {n64} -> iota -> map {{\n        cell -> (0.0, rw) -> fold {{\n            acc, k -> acc + {x}[cell / 64 * {w} + k] * weight[k * 64 + cell % 64]\n        }}\n    }} -> y;\n",
        n64 = n * 64
    );
    let out = if res {
        s += &format!(
            "    {n64} -> iota -> map {{\n        cell -> rows[cell / 64 * {w} + cell % 64] + y[cell]\n    }} -> h;\n",
            n64 = n * 64
        );
        "h"
    } else {
        "y"
    };
    let last = n * 64 - 1;
    for c in [0, 65, last / 2, last] {
        s += &format!("    {out}[{c}] -> println;\n");
    }
    s + "}\n"
}

/// bf16-integrate `cand-made` on generated prompt heads
/// (`prompt_head_src`): a map an operand of which a device candidate makes is
/// a candidate itself, so the gather, its row fold and the norm's scale are
/// placed as one chain against measured costs (x86-64-v3's host, the 5080's
/// device) — the gather runs on the device as a checking cell beside the
/// projection that reads it, nothing gathered crosses up, and the values
/// equal the interpreter in every emulator mode. The OFF arm
/// (`--gpu-off=cand-made`) decides each map by its own roofline: where that
/// leaves the gather on the host (a chain the roofline prices only cell by
/// cell), the counter `moved` records it; its values equal too.
#[test]
fn cand_made_places_generated_prompt_heads() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut progs = Vec::new();
    for (i, w) in [256u64, 1024, 2048].into_iter().enumerate() {
        for (j, n) in [8u64, 64, 256, 512].into_iter().enumerate() {
            let r = [7u64, 40, 300][(i + j) % 3];
            progs.push((r, w, n, (i + j) % 4 != 3, (i + 2 * j) % 3 != 0));
        }
    }
    let results = testgen::par_check(&progs, |i, &(r, w, n, norm, res)| {
        let src = prompt_head_src(r, w, n, norm, res);
        let ir = lower(&src);
        let rr = run(&ir, 40_000_000_000);
        assert!(matches!(rr.outcome, Outcome::Done(_)), "{:?}", rr.outcome);
        let want = rr.output;
        let mut c = vec![("programs", 1u64)];
        let gathered = |em: &Emission| {
            em.device
                .as_deref()
                .unwrap_or("")
                .contains("@dev_trap_record(")
        };
        // The default face (order free, contracted, `sqrt`): the values
        // within a bound of the oracle's, not its bits.
        let close = |got: &str, what: &str| {
            let num = |t: &str| -> Vec<f64> {
                t.lines()
                    .map(|l| l.trim().parse().expect("a number"))
                    .collect()
            };
            let (g, w) = (num(got), num(&want));
            assert_eq!(g.len(), w.len(), "{what}\n{src}");
            for (x, y) in g.iter().zip(&w) {
                assert!(
                    (x - y).abs() <= 2f64.powi(-12) * y.abs().max(1.0),
                    "{what}: {x} vs {y}\n{src}"
                );
            }
        };
        let on = priced_emission(&ir);
        let mut gpu = GpuArms::default();
        for r in [
            "async",
            "pdl",
            "streams",
            "pinned",
            "horizontal",
            "calls",
            "cand-made",
        ] {
            *gpu.off.rung(r).expect("rung") = true;
        }
        let opts = EmitOpts {
            host: "x86-64-v3",
            target: "rtx-5080",
            gpu,
            ..EmitOpts::default()
        };
        let off = emit_detailed(&ir, &opts).expect("emits");
        // Candidates only grow: a gather the roofline alone put on the
        // device stays there.
        assert!(
            gathered(&on) || !gathered(&off),
            "cand-made took the gather off the device\n{src}"
        );
        if gathered(&on) {
            c.push(("gather-on-device", 1));
            if !gathered(&off) {
                c.push(("moved", 1));
            }
        }
        if norm {
            c.push(("with-norm", 1));
        }
        if res {
            c.push(("with-residual", 1));
        }
        let dev = on.device.as_deref().unwrap_or("");
        let ex = emu::build_emulated(&clang, &on.text, dev);
        for (mode, (got, _)) in MODES.iter().zip(ex.run_each(&MODES)) {
            close(&got, &format!("{mode:?}"));
        }
        if i % 2 == 0 || gathered(&on) != gathered(&off) {
            let got = emu::run_emulated(&clang, &off.text, off.device.as_deref().unwrap_or(""));
            close(&got, "cand-made off");
            c.push(("off-arm-runs", 1));
        }
        c
    });
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    for r in results.into_iter().flatten() {
        *c.entry(r.0).or_default() += r.1;
    }
    eprintln!("cand-made generated: {c:?}");
    for key in [
        "programs",
        "gather-on-device",
        "with-norm",
        "with-residual",
        "moved",
        "off-arm-runs",
    ] {
        assert!(c.get(key).copied().unwrap_or(0) > 0, "{key}: {c:?}");
    }
}
