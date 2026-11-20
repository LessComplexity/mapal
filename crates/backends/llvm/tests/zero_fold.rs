//! kernel-merging `zero-fold`: an atomic site's zero pass as the tail of an
//! earlier kernel the dependency fact (`access_plan`) orders after every
//! earlier access to the output's bytes — the rule's decisions on the decode
//! step and attention, values through the GPU emulator against the
//! interpreter and the OFF arm under every candidate pick (the emulator skips
//! a flagged site's own zero, so the tail is the only zero), free-running and
//! serialized, on hand-written and generated programs
//! (`testgen::zero_prog_strategy`).

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

fn bench(name: &str) -> CategoryIr {
    lower(
        &std::fs::read_to_string(format!(
            "{}/../../../benches/algorithms/{name}.mapal",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap(),
    )
}

/// The shipped emission on `target` (default face), the rung `on`.
fn shipped(ir: &CategoryIr, target: &'static str, on: bool) -> Emission {
    let mut gpu = GpuArms::default();
    *gpu.off.rung("zero-fold").unwrap() = !on;
    emit_detailed(
        ir,
        &EmitOpts {
            target,
            gpu,
            ..EmitOpts::default()
        },
    )
    .expect("emits")
}

/// The emulator's emission: its rungs off (`place`, so every capable site is
/// on the device), the default face (atomic candidates need a free order).
fn emulated(ir: &CategoryIr, on: bool) -> Emission {
    let mut gpu = GpuArms::default();
    for r in ["place", "async", "pdl", "streams", "pinned"] {
        *gpu.off.rung(r).unwrap() = true;
    }
    *gpu.off.rung("zero-fold").unwrap() = !on;
    emit_detailed(
        ir,
        &EmitOpts {
            host: "generic",
            target: "rtx-5080",
            gpu,
            ..EmitOpts::default()
        },
    )
    .expect("emits")
}

fn field<'l>(line: &'l str, key: &str) -> &'l str {
    line.split_whitespace()
        .find_map(|f| f.strip_prefix(key).and_then(|v| v.strip_prefix('=')))
        .unwrap_or("")
}

/// The host intent line `; mapal.zero.host …`.
fn host_line(em: &Emission) -> String {
    em.text
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("; mapal.zero.host "))
        .unwrap_or("")
        .to_string()
}

/// The site calls' flags, in order.
fn site_flags(em: &Emission) -> Vec<u32> {
    em.text
        .lines()
        .filter_map(|l| l.trim().strip_prefix("call void @mapal_gpu_site(i64 "))
        .map(|r| {
            r.split(", ")
                .nth(1)
                .unwrap()
                .strip_prefix("i32 ")
                .unwrap()
                .parse()
                .unwrap()
        })
        .collect()
}

fn within(got: &str, want: &str) -> bool {
    let (g, w): (Vec<&str>, Vec<&str>) = (
        got.split_whitespace().collect(),
        want.split_whitespace().collect(),
    );
    g.len() == w.len()
        && g.iter().zip(&w).all(|(a, b)| {
            a == b
                || match (a.parse::<f64>(), b.parse::<f64>()) {
                    (Ok(x), Ok(y)) => (x - y).abs() <= 1e-3 * x.abs().max(y.abs()).max(1.0),
                    _ => false,
                }
        })
}

fn zeros(err: &str) -> u64 {
    let l = err
        .lines()
        .find_map(|l| l.strip_prefix("emu_total "))
        .unwrap_or("");
    field(l, "zeros").parse().unwrap_or(0)
}

/// `pt_decode_1024_256` on the rtx-5080: the scores `s` and the output `o`
/// (atomic sites, 4 KB + 1 KB) zero in the query's cell kernel, the trip's
/// last kernel before them that touches neither (the row `x` reads the last
/// output, whose buffer the next one may take); the retained rows `kk`, `vv`
/// keep their zero (their range is the trip's). `pt_attn_1024_64`: `P·V`'s
/// output in the softmax unit. OFF: no tail, no flag, no intent line.
#[test]
fn decode_and_attention_fold_where_the_dataflow_allows() {
    let ir = bench("pt_decode_1024_256");
    let on = shipped(&ir, "rtx-5080", true);
    let h = host_line(&on);
    assert_eq!(
        (
            field(&h, "sites"),
            field(&h, "folded"),
            field(&h, "kernels")
        ),
        ("2", "2", "1"),
        "{h}"
    );
    assert!(
        on.text.contains("; mapal.zero sites=2 bytes=5120"),
        "{}",
        on.text
    );
    assert_eq!(site_flags(&on), vec![1, 1, 1, 1, 3, 3]);
    let dev = on.device.as_deref().unwrap();
    assert_eq!(dev.matches("i64 %zt, ptr addrspace(1) %zp0").count(), 1);
    let off = shipped(&ir, "rtx-5080", false);
    assert!(host_line(&off).is_empty() && !off.text.contains("; mapal.zero"));
    assert!(site_flags(&off).iter().all(|&f| f == 1));
    assert!(!off.device.as_deref().unwrap().contains("%zt"));

    let ir = bench("pt_attn_1024_64");
    let on = shipped(&ir, "rtx-5080", true);
    let h = host_line(&on);
    assert_eq!((field(&h, "sites"), field(&h, "folded")), ("1", "1"), "{h}");
    let unit = on
        .text
        .lines()
        .position(|l| l.contains("; unit"))
        .expect("the softmax unit");
    let zero = on
        .text
        .lines()
        .position(|l| l.contains("; mapal.zero sites=1 bytes=262144"))
        .expect("o's zero");
    assert!(zero < unit, "the zero rides the unit's launch");
    assert_eq!(site_flags(&on), vec![1, 3]);
}

/// A decode step small enough to emulate, its GEMV rows long enough for
/// atomic candidates: values equal the interpreter under every candidate pick
/// with the rule on and off, free-running and serialized; the rule's zero
/// tails replace two zero passes a trip. With the tails' stores removed (the
/// sites still flagged) the values are wrong: the sites rely on the tail.
#[test]
fn decode_values_with_the_zero_in_the_query_kernel() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let (trips, rows, d) = (3u64, 16u64, 256u64);
    let src = format!(
        "fn main() {{
    {dd} -> iota -> tw;
    tw -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> x; x / 64.0 }} -> wk;
    tw -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> x; x / 64.0 }} -> wv;
    {rd} -> iota -> tc;
    {d} -> iota -> krange;
    {rows} -> iota -> rrange;
    (0.0, {rd}) -> fill -> x0;
    (0.0, {d}) -> fill -> o0;
    mut x: [f32; {rd}] <- x0;
    mut o: [f32; {d}] <- o0;
    mut t: i32 <- 0;
    loop {{
        (t < {trips}) -> {{
            -true-> {{
                tc -> map {{ c -> (c / {d} == t) -> {{ -true-> {{ (c * 7 + 13) % 101 - 50 -> widen_f32 -> f; f / 16.0 + o[c % {d}] }} -false-> x[c]; }} }} -> x;
                tc -> map {{ c -> c / {d} -> i; c % {d} -> j; (0.0, krange) -> fold {{ acc, k -> acc + x[i * {d} + k] * wk[k * {d} + j] }} }} -> kk;
                tc -> map {{ c -> c / {d} -> i; c % {d} -> j; (0.0, krange) -> fold {{ acc, k -> acc + x[i * {d} + k] * wv[k * {d} + j] }} }} -> vv;
                krange -> map {{ k -> ((t * {d} + k) * 5 + 3) % 97 - 48 -> widen_f32 -> qv; qv / 64.0 }} -> q;
                rrange -> map {{ r -> (0.0, krange) -> fold {{ acc, k -> acc + q[k] * kk[r * {d} + k] }} }} -> s;
                krange -> map {{ j -> (0.0, rrange) -> fold {{ acc, r -> acc + s[r] * vv[r * {d} + j] }} }} -> o;
                t + 1 -> t;
                -> loop;
            }}
            -false-> o -> out;
        }}
    }}
    out[0] -> println;
    out[{last}] -> println;
    0.0 -> zero: f32;
    (zero, out) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
}}
",
        dd = d * d,
        rd = rows * d,
        last = d - 1
    );
    let ir = lower(&src);
    let rr = run(&ir, 4_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let on = emulated(&ir, true);
    let off = emulated(&ir, false);
    let h = host_line(&on);
    assert!(
        field(&h, "folded").parse::<u32>().unwrap() >= 1,
        "{h}\n{}",
        on.text
    );
    let (eon, eoff) = (
        emu::build_emulated(&clang, &on.text, on.device.as_deref().unwrap()),
        emu::build_emulated(&clang, &off.text, off.device.as_deref().unwrap()),
    );
    let mut fewer = 0;
    for pick in ["0", "1", "2", "3", "4", "5"] {
        let (a, ea) = eon.run(&[("MAPAL_GPU_PICK", pick)]);
        let (b, eb) = eoff.run(&[("MAPAL_GPU_PICK", pick)]);
        assert!(
            within(&a, &rr.output),
            "pick {pick} ON: {a:?} vs {:?}",
            rr.output
        );
        assert!(within(&b, &rr.output), "pick {pick} OFF: {b:?}");
        // A pick whose candidate adds into a zeroed output runs one zero
        // pass fewer a folded site and trip.
        assert!(
            zeros(&ea) <= zeros(&eb),
            "pick {pick}: {} vs {}",
            zeros(&ea),
            zeros(&eb)
        );
        fewer += usize::from(zeros(&ea) < zeros(&eb));
        for mode in ["1", "2"] {
            let (s, _) = eon.run(&[("MAPAL_GPU_PICK", pick), ("MAPAL_EMU_SERIAL", mode)]);
            assert!(within(&s, &rr.output), "pick {pick} serial {mode}");
        }
    }
    assert!(fewer > 0, "an atomic pick runs on the tail's zero");
    // Trials (the fake clock: the last candidates fastest) time every
    // candidate on the real output: each timed run and the pick after it
    // zero again, flag or not.
    let dir = tempfile::tempdir().unwrap();
    let wis = dir.path().join("w");
    let (t, err) = eon.run(&[
        (
            "MAPAL_EMU_NS",
            "1000000,900000,800000,700000,600000,500000,400000,300000",
        ),
        ("MAPAL_WISDOM", wis.to_str().unwrap()),
        ("MAPAL_GPU_TRACE", "1"),
    ]);
    assert!(err.contains("op=trial "), "trials ran\n{err}");
    assert!(
        within(&t, &rr.output),
        "after trials: {t:?} vs {:?}",
        rr.output
    );
    // The tails' stores gone, the flags kept: some pick reads garbage.
    let dev = on
        .device
        .as_deref()
        .unwrap()
        .lines()
        .filter(|l| !(l.trim_start().starts_with("store ") && l.contains("ptr addrspace(1) %zf")))
        .collect::<Vec<_>>()
        .join("\n");
    let bad = emu::build_emulated(&clang, &on.text, &dev);
    let caught = ["0", "1", "2", "3", "4", "5"]
        .iter()
        .any(|p| !within(&bad.run(&[("MAPAL_GPU_PICK", p)]).0, &rr.output));
    assert!(caught, "a missing zero tail must show");
}

/// Coverage of [`zero_fold_on_generated_programs`].
#[derive(Default, Debug)]
struct Cover {
    programs: usize,
    sites: usize,
    folded: usize,
    loops: usize,
    straight: usize,
    multi: usize,
    stopped: usize,
    none: usize,
    runs: usize,
}

/// Over generated programs (`zero_prog_strategy`): the OFF arm is the
/// emission without tails or flags; ON, every candidate pick, free-running
/// and serialized in thread order and reverse, equals the interpreter (within
/// the default face's bound) and the OFF arm; a folded program runs fewer zero
/// passes. Coverage asserted: folds in loop trips and straight-line code,
/// kernels zeroing several sites, walks stopped by an access to the output's
/// bytes, and sites no kernel could take.
#[test]
fn zero_fold_on_generated_programs() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let strat = testgen::zero_prog_strategy();
    let mut cv = Cover::default();
    for z in 0..120 {
        let p = strat.new_tree(&mut runner).unwrap().current();
        let src = testgen::zero_src(&p);
        let ir = lower(&src);
        let rr = run(&ir, 4_000_000_000);
        assert!(matches!(rr.outcome, Outcome::Done(_)), "#{z}\n{src}");
        let on = emulated(&ir, true);
        let off = emulated(&ir, false);
        assert!(host_line(&off).is_empty(), "#{z}");
        assert!(site_flags(&off).iter().all(|&f| f & 2 == 0), "#{z}");
        cv.programs += 1;
        let h = host_line(&on);
        let n = |k: &str| field(&h, k).parse::<usize>().unwrap_or(0);
        cv.sites += n("sites");
        cv.stopped += n("stopped");
        cv.none += n("none");
        let folded = n("folded");
        if folded == 0 {
            // No tail: the emission is the OFF arm's but for the intent.
            let strip = |t: &str| {
                t.lines()
                    .filter(|l| !l.trim().starts_with("; mapal.zero.host"))
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            assert_eq!(strip(&on.text), strip(&off.text), "#{z}");
            continue;
        }
        cv.folded += folded;
        if p.looped {
            cv.loops += 1;
        } else {
            cv.straight += 1;
        }
        // A kernel zeroing several sites.
        cv.multi += usize::from(folded > n("kernels"));
        let flagged = site_flags(&on).iter().filter(|&&f| f & 2 != 0).count();
        assert_eq!(
            flagged, folded,
            "#{z}: every folded site flagged\n{}",
            on.text
        );
        let (eon, eoff) = (
            emu::build_emulated(&clang, &on.text, on.device.as_deref().unwrap()),
            emu::build_emulated(&clang, &off.text, off.device.as_deref().unwrap()),
        );
        for pick in ["0", "1", "3", "7"] {
            let (a, ea) = eon.run(&[("MAPAL_GPU_PICK", pick)]);
            let (b, eb) = eoff.run(&[("MAPAL_GPU_PICK", pick)]);
            assert!(
                within(&a, &rr.output),
                "#{z} pick {pick}: {a:?} vs {:?}\n{src}",
                rr.output
            );
            assert!(
                within(&a, &b),
                "#{z} pick {pick}: ON {a:?} OFF {b:?}\n{src}"
            );
            assert!(zeros(&ea) <= zeros(&eb), "#{z} pick {pick}");
            for mode in ["1", "2"] {
                let (s, _) = eon.run(&[("MAPAL_GPU_PICK", pick), ("MAPAL_EMU_SERIAL", mode)]);
                assert!(
                    within(&s, &rr.output),
                    "#{z} pick {pick} serial {mode}\n{src}"
                );
            }
            cv.runs += 1;
        }
    }
    eprintln!("zero_fold_on_generated_programs: {cv:?}");
    assert!(
        cv.folded > 0
            && cv.loops > 0
            && cv.straight > 0
            && cv.multi > 0
            && cv.stopped > 0
            && cv.none > 0,
        "{cv:?}"
    );
}
