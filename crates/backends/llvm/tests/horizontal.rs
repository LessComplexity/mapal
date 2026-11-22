//! kernel-merging `horizontal`: independent launches of one sequence that
//! the dependency fact (`access_plan`) leaves unordered run as one grid — the
//! rule's decisions on the decode step, the OFF arm, values through the GPU
//! emulator against the interpreter and the OFF arm under every candidate pick
//! of the merged tables, free-running, serialized and with a grid's blocks in
//! reverse (members that share bytes race), on hand-written and generated
//! programs (`testgen::hz_prog_strategy`).

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

/// The shipped emission on `target` (default face), the rung `on`, on a named
/// host: a `native` one is priced by this machine's measured dispatch.
fn shipped(ir: &CategoryIr, target: &'static str, on: bool) -> Emission {
    let mut gpu = GpuArms::default();
    *gpu.off.rung("horizontal").unwrap() = !on;
    // `calls` (its tests) would take the groups into persistent chains.
    *gpu.off.rung("calls").unwrap() = true;
    emit_detailed(
        ir,
        &EmitOpts {
            target,
            host: "generic",
            gpu,
            ..EmitOpts::default()
        },
    )
    .expect("emits")
}

/// The emulator's emission: its rungs off (`place`, so every capable site is
/// on the device), the default face.
fn emulated(ir: &CategoryIr, on: bool) -> Emission {
    let mut gpu = GpuArms::default();
    for r in ["place", "async", "pdl", "streams", "pinned"] {
        *gpu.off.rung(r).unwrap() = true;
    }
    *gpu.off.rung("horizontal").unwrap() = !on;
    // `calls` (its tests) would take the groups into persistent chains.
    *gpu.off.rung("calls").unwrap() = true;
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

/// The host intent line `; mapal.hz.host …`.
fn host_line(em: &Emission) -> String {
    em.text
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("; mapal.hz.host "))
        .unwrap_or("")
        .to_string()
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

fn launches(err: &str) -> u64 {
    let l = err
        .lines()
        .find_map(|l| l.strip_prefix("emu_total "))
        .unwrap_or("");
    field(l, "launches").parse().unwrap_or(0)
}

/// The text without the rung's own lines (intent, merged launches).
fn strip(t: &str) -> String {
    t.lines()
        .filter(|l| !l.trim().starts_with("; mapal.hz"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// `pt_decode_1024_256` on the rtx-5080: the retained rows `kk`, `vv` (row
/// passes in their trip's branch) and the query cell `q` (it reads only the
/// counter) are one group launched at `q`: one merged table, its four tuples
/// the rows' unsplit candidates, each merged kernel a dispatch over the
/// members' block ranges. `x` (writes what both rows read) and `s`, `o`
/// (read them) stay out. OFF: no merged launch, no intent line.
#[test]
fn decode_merges_its_rows_and_query() {
    let ir = bench("pt_decode_1024_256");
    let on = shipped(&ir, "rtx-5080", true);
    let h = host_line(&on);
    assert_eq!(
        (
            field(&h, "groups"),
            field(&h, "members"),
            field(&h, "branchy"),
            field(&h, "refused")
        ),
        ("1", "3", "2", "0"),
        "{h}"
    );
    let specs: Vec<&str> = on
        .text
        .lines()
        .filter(|l| l.trim().starts_with("; mapal.hz kernel="))
        .collect();
    assert_eq!(specs.len(), 4, "{specs:?}");
    assert!(specs[0].contains("block=256 smem=0"), "{}", specs[0]);
    let dev = on.device.as_deref().unwrap();
    for s in &specs {
        let name = field(s.trim(), "kernel");
        assert!(
            dev.contains(&format!("define ptx_kernel void @{name}(")),
            "{name}"
        );
    }
    // A 32-thread row candidate in a 256-thread grid counts its own threads.
    assert!(dev.contains("barrier.sync.cnt(i32 0, i32 32)"));
    assert!(on.text.contains("@mapal_main.hz"), "the merged table");
    let off = shipped(&ir, "rtx-5080", false);
    assert!(host_line(&off).is_empty() && !off.text.contains("; mapal.hz"));
    assert!(!off.device.as_deref().unwrap().contains("@hz.mapal_tile_"));

    // `pt_rt_mlp2_1024`'s trip: the two input cells (4096 blocks each; their
    // last waves, 64 blocks each, share one) are one launch; `d1` and `c2`
    // (128-block GEMMs whose last waves fill one alone) stay two.
    let ir = bench("pt_rt_mlp2_1024");
    let h = host_line(&shipped(&ir, "rtx-5080", true));
    assert_eq!(
        (
            field(&h, "groups"),
            field(&h, "cells"),
            field(&h, "refused")
        ),
        ("1", "2", "waves:1"),
        "{h}"
    );
}

/// A decode step small enough to emulate: values equal the interpreter under
/// every pick of the merged table with the rule on and off, free-running and
/// serialized; the first trip (the whole sites in place) launches each member
/// alone, later trips one launch for three.
#[test]
fn decode_values_through_the_merged_launch() {
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
    assert_eq!(field(&h, "branchy"), "2", "{h}\n{}", on.text);
    let (eon, eoff) = (
        emu::build_emulated(&clang, &on.text, on.device.as_deref().unwrap()),
        emu::build_emulated(&clang, &off.text, off.device.as_deref().unwrap()),
    );
    for pick in ["0", "1", "2", "3", "4", "5"] {
        let (a, ea) = eon.run(&[("MAPAL_GPU_PICK", pick)]);
        let (b, eb) = eoff.run(&[("MAPAL_GPU_PICK", pick)]);
        assert!(
            within(&a, &rr.output),
            "pick {pick} ON: {a:?} vs {:?}",
            rr.output
        );
        assert!(within(&b, &rr.output), "pick {pick} OFF: {b:?}");
        // Two launches fewer every trip after the first.
        assert_eq!(
            launches(&ea) + 2 * (trips - 1),
            launches(&eb),
            "pick {pick}\n{ea}\n{eb}"
        );
        for mode in ["1", "2"] {
            let (s, _) = eon.run(&[("MAPAL_GPU_PICK", pick), ("MAPAL_EMU_SERIAL", mode)]);
            assert!(within(&s, &rr.output), "pick {pick} serial {mode}");
        }
        let (r, _) = eon.run(&[("MAPAL_GPU_PICK", pick), ("MAPAL_EMU_REVERSE", "1")]);
        assert!(within(&r, &rr.output), "pick {pick} reverse");
    }
}

/// Coverage of [`horizontal_on_generated_programs`].
#[derive(Default, Debug)]
struct Cover {
    programs: usize,
    groups: usize,
    members: usize,
    wide: usize,
    cells: usize,
    units: usize,
    tiles: usize,
    branchy: usize,
    tables: usize,
    padded: usize,
    refused: usize,
    loops: usize,
    straight: usize,
    fewer: usize,
    runs: usize,
}

/// Over generated programs (`hz_prog_strategy`): the OFF arm has no merged
/// launch; ON, every pick of the merged tables, free-running and serialized
/// in thread order and reverse, equals the interpreter (within the default
/// face's bound) and the OFF arm, with fewer launches where a group formed; a
/// program with no group is the OFF arm's text. Coverage asserted: groups in
/// loop trips and straight-line code, groups of three or more, retained rows,
/// merged tables, members of fewer threads than the merged block, refusals.
#[test]
fn horizontal_on_generated_programs() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let strat = testgen::hz_prog_strategy();
    let mut cv = Cover::default();
    let n: usize = std::env::var("MAPAL_TESTGEN_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);
    for z in 0..n {
        let p = strat.new_tree(&mut runner).unwrap().current();
        let src = testgen::hz_src(&p);
        let ir = lower(&src);
        let rr = run(&ir, 4_000_000_000);
        assert!(matches!(rr.outcome, Outcome::Done(_)), "#{z}\n{src}");
        let on = emulated(&ir, true);
        let off = emulated(&ir, false);
        assert!(host_line(&off).is_empty(), "#{z}");
        cv.programs += 1;
        let h = host_line(&on);
        let n = |k: &str| field(&h, k).parse::<usize>().unwrap_or(0);
        cv.refused += usize::from(!h.is_empty() && field(&h, "refused") != "0");
        let groups = n("groups");
        if groups == 0 {
            assert_eq!(strip(&on.text), strip(&off.text), "#{z}");
            continue;
        }
        cv.groups += groups;
        cv.members += n("members");
        cv.wide += usize::from(n("members") > 2 * groups);
        cv.branchy += n("branchy");
        cv.cells += n("cells");
        cv.units += n("units");
        cv.tiles += n("tiles") - n("branchy");
        let specs: Vec<&str> = on
            .text
            .lines()
            .filter(|l| l.trim().starts_with("; mapal.hz kernel="))
            .collect();
        cv.tables += usize::from(on.text.contains("@mapal_main.hz") && specs.len() > groups);
        cv.padded += usize::from(specs.iter().any(|s| {
            let b = field(s.trim(), "block");
            field(s.trim(), "members")
                .split(',')
                .any(|m| m.rsplit(':').next() != Some(b))
        }));
        if p.looped {
            cv.loops += 1;
        } else {
            cv.straight += 1;
        }
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
            assert!(launches(&ea) <= launches(&eb), "#{z} pick {pick}");
            cv.fewer += usize::from(launches(&ea) < launches(&eb));
            for mode in ["1", "2"] {
                let (s, _) = eon.run(&[("MAPAL_GPU_PICK", pick), ("MAPAL_EMU_SERIAL", mode)]);
                assert!(
                    within(&s, &rr.output),
                    "#{z} pick {pick} serial {mode}\n{src}"
                );
            }
            // A grid's blocks in reverse: two members sharing bytes race.
            let (r, _) = eon.run(&[("MAPAL_GPU_PICK", pick), ("MAPAL_EMU_REVERSE", "1")]);
            assert!(within(&r, &rr.output), "#{z} pick {pick} reverse\n{src}");
            cv.runs += 1;
        }
    }
    eprintln!("horizontal_on_generated_programs: {cv:?}");
    assert!(
        cv.groups > 0
            && cv.loops > 0
            && cv.straight > 0
            && cv.wide > 0
            && cv.cells > 0
            && cv.tiles > 0
            && cv.branchy > 0
            && cv.tables > 0
            && cv.padded > 0
            && cv.fewer > 0,
        "{cv:?}"
    );
}
