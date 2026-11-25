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
    // tensor-cores `staged-regs`: the rows' candidates whose register-staged
    // panel spills (three of the four unsplit tuples: 5 kernels, ptxas) are
    // refused — one merged tuple remains.
    assert_eq!(specs.len(), 1, "{specs:?}");
    assert!(specs[0].contains("block=256 smem=0"), "{}", specs[0]);
    let dev = on.device.as_deref().unwrap();
    for s in &specs {
        let name = field(s.trim(), "kernel");
        assert!(
            dev.contains(&format!("define ptx_kernel void @{name}(")),
            "{name}"
        );
    }
    // tensor-cores `staged-regs`: the 32-thread row candidates (whose
    // register-staged panels spilled) are refused; the one tuple left is a
    // 256-thread launch without a merged table.
    assert!(!dev.contains("barrier.sync.cnt(i32 0, i32 32)"));
    assert!(!on.text.contains("@mapal_main.hz"), "one tuple: no table");
    let off = shipped(&ir, "rtx-5080", false);
    assert!(host_line(&off).is_empty() && !off.text.contains("; mapal.hz"));
    assert!(!off.device.as_deref().unwrap().contains("@hz.mapal_tile_"));

    // `pt_rt_mlp2_1024`'s trip: the two input cells (4096 blocks each; their
    // last waves, 64 blocks each, share one) are one launch. `d1` and `c2`
    // (128-block GEMMs) walk their tiles by stream-K on the 5080 since
    // sweep-gemm-2 (`streamk-price`: one wave of CTAs each), so no merge is
    // weighed for them; with that rule off they are the 128-block launches
    // whose last waves fill one alone, refused (integration 3: the merged
    // workspace run found this expectation stale).
    let ir = bench("pt_rt_mlp2_1024");
    let h = host_line(&shipped(&ir, "rtx-5080", true));
    let key = |h: &str| {
        (
            field(h, "groups").to_string(),
            field(h, "cells").to_string(),
            field(h, "refused").to_string(),
        )
    };
    assert_eq!(key(&h), ("1".into(), "2".into(), "0".into()), "{h}");
    let mut gpu = GpuArms::default();
    *gpu.off.rung("calls").unwrap() = true;
    *gpu.off.rung("streamk-price").unwrap() = true;
    let unpriced = emit_detailed(
        &ir,
        &EmitOpts {
            target: "rtx-5080",
            host: "generic",
            gpu,
            ..EmitOpts::default()
        },
    )
    .unwrap();
    let h = host_line(&unpriced);
    assert_eq!(key(&h), ("1".into(), "2".into(), "waves:1".into()), "{h}");
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

impl Cover {
    fn add(&mut self, d: &Cover) {
        self.programs += d.programs;
        self.groups += d.groups;
        self.members += d.members;
        self.wide += d.wide;
        self.cells += d.cells;
        self.units += d.units;
        self.tiles += d.tiles;
        self.branchy += d.branchy;
        self.tables += d.tables;
        self.padded += d.padded;
        self.refused += d.refused;
        self.loops += d.loops;
        self.straight += d.straight;
        self.fewer += d.fewer;
        self.runs += d.runs;
    }
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
    let n: usize = std::env::var("MAPAL_TESTGEN_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);
    let progs: Vec<testgen::HzProg> = (0..n)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let covers = testgen::par_check(&progs, |z, p| {
        let mut cv = Cover::default();
        let src = testgen::hz_src(p);
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
            return cv;
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
            let runs = eon.run_each(&[
                [("MAPAL_GPU_PICK", pick), ("MAPAL_EMU_SERIAL", "0")],
                [("MAPAL_GPU_PICK", pick), ("MAPAL_EMU_SERIAL", "1")],
                [("MAPAL_GPU_PICK", pick), ("MAPAL_EMU_SERIAL", "2")],
                [("MAPAL_GPU_PICK", pick), ("MAPAL_EMU_REVERSE", "1")],
            ]);
            let (a, ea) = &runs[0];
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
            assert!(launches(ea) <= launches(&eb), "#{z} pick {pick}");
            cv.fewer += usize::from(launches(ea) < launches(&eb));
            for (mode, (s, _)) in ["1", "2"].iter().zip(&runs[1..3]) {
                assert!(
                    within(s, &rr.output),
                    "#{z} pick {pick} serial {mode}\n{src}"
                );
            }
            // A grid's blocks in reverse: two members sharing bytes race.
            let (r, _) = &runs[3];
            assert!(within(r, &rr.output), "#{z} pick {pick} reverse\n{src}");
            cv.runs += 1;
        }
        cv
    });
    let mut cv = Cover::default();
    for d in &covers {
        cv.add(d);
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

/// full-model: **a hoisted zero waits for the merged launch.** Two
/// projections, the rotary embedding of each (two cells the rule merges: the
/// first is deferred to the second's position), then two projections of the
/// rotated query. The arena gives the second pair's outputs the bytes of the
/// first pair's (dead after the rotations), and `zero-hoist` issues their
/// zero ops early — it once stopped at the deferred member's own position,
/// so the zero cleared `q` before the merged launch read it (the rotated
/// query and everything after it were 0: the readable Llama's prompt
/// computed wrong logits). The walk now stops at the group's launch (its last
/// member). Every emulator mode prints the oracle's values; the emission has
/// the merged launch and the hoisted zeros (else the test proves nothing).
/// Without the fix the reversed-grid and serial modes print 0 for the rotated
/// query (as the RTX 5080 did).
#[test]
fn a_hoisted_zero_waits_for_the_merged_launch() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let src = std::fs::read_to_string(format!(
        "{}/tests/fixtures/hoisted_zero_merged.mapal",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let ir = lower(&src);
    let rr = run(&ir, 20_000_000_000);
    let Outcome::Done(_) = rr.outcome else {
        panic!("oracle: {:?}", rr.outcome);
    };
    // The second pair split along `k` (atomic outputs, each zeroed first):
    // the sites this machine's measured profile splits by rule (`native-gpu`
    // on the RTX 5080), on the static part's profile.
    let mut opts = EmitOpts {
        host: "generic",
        target: "rtx-5080",
        ..EmitOpts::default()
    };
    mapal_backend_llvm::flags::apply(&mut opts, "--splitk=2").expect("a flag");
    let em = emit_detailed(&ir, &opts).expect("emits");
    assert!(em.text.contains("; mapal.hz kernel="), "no merged launch");
    let zero = em
        .text
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("; mapal.zero.host "))
        .unwrap_or("")
        .to_string();
    assert!(
        field(&zero, "hoisted").parse::<u32>().unwrap_or(0) > 0,
        "{zero}"
    );
    let ex = emu::build_emulated(&clang, &em.text, em.device.as_deref().unwrap());
    for mode in [
        [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_GRID", "3")],
        [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_GRID", "3")],
        [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_GRID", "2")],
        [("MAPAL_EMU_REVERSE", "1"), ("MAPAL_EMU_GRID", "1")],
    ] {
        let (out, err) = ex.run(&mode);
        assert!(
            within(&out, &rr.output),
            "{mode:?}\n{out}\nwant\n{}\n{err}",
            rr.output
        );
    }
}
