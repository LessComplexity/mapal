//! compile-time-graph B: launch regions — a device loop's trip tagged with its
//! compile-time dependencies (`func/region.rs`) and replayed as one graph by
//! the runtime. Through the GPU emulator, which replays a region in a seeded
//! random topological order (`MAPAL_EMU_SHUFFLE`): values equal the
//! interpreter and the OFF arm under every order, a tag that drops an edge is
//! caught, and the regions and nodes run are what the tags predict — on a
//! decode-shaped loop and on generated loops (`testgen::twin_prog_strategy`).

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

/// The exact face on `target`, the emulator's rungs off, the graph arm `on`.
fn emission(ir: &CategoryIr, target: &'static str, on: bool) -> Emission {
    let mut gpu = GpuArms::default();
    gpu.graph = on;
    for r in ["place", "async", "pdl", "streams", "pinned"] {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "generic",
        target,
        gpu,
        ..EmitOpts::exact()
    };
    emit_detailed(ir, &opts).expect("emits")
}

fn field<'l>(line: &'l str, key: &str) -> &'l str {
    line.split_whitespace()
        .find_map(|f| f.strip_prefix(key).and_then(|v| v.strip_prefix('=')))
        .unwrap_or("")
}

/// The `; mapal.region` intent lines and the tags `(op, deps, deps_out)`.
fn regions(host: &str) -> (Vec<String>, Vec<(i64, i64, i64)>) {
    let lines = host
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("; mapal.region "))
        .map(str::to_string)
        .collect();
    let tags = host
        .lines()
        .filter_map(|l| l.trim().strip_prefix("call void @mapal_gpu_tag(i32 "))
        .filter_map(|r| {
            let v: Vec<i64> = r
                .trim_end_matches(')')
                .split(", i64 ")
                .filter_map(|x| x.parse().ok())
                .collect();
            (v.len() == 3 && v[0] >= 0).then(|| (v[0], v[1], v[2]))
        })
        .collect();
    (lines, tags)
}

/// `emu_total`'s `graphs=` and `graph_nodes=`.
fn graph_totals(err: &str) -> (u64, u64) {
    let l = err
        .lines()
        .find_map(|l| l.strip_prefix("emu_total "))
        .unwrap_or("");
    let get = |k| field(l, k).parse().unwrap_or(0);
    (get("graphs"), get("graph_nodes"))
}

/// A decode step as `pt_decode`'s, small: a row of the state written in
/// place, two retained GEMV rows from it, a query, scores over every row, the
/// next output from the scores.
fn decode(trips: u64, rows: u64, d: u64) -> String {
    let rd = rows * d;
    let dd = d * d;
    format!(
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
                tc -> map {{ c ->
                    c / {d} -> i;
                    c % {d} -> j;
                    (0.0, krange) -> fold {{ acc, k -> acc + x[i * {d} + k] * wk[k * {d} + j] }}
                }} -> kk;
                tc -> map {{ c ->
                    c / {d} -> i;
                    c % {d} -> j;
                    (0.0, krange) -> fold {{ acc, k -> acc + x[i * {d} + k] * wv[k * {d} + j] }}
                }} -> vv;
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
        last = d - 1
    )
}

/// `pt_causal`'s step, small: the decode step with a masked score row, its
/// max and sum as whole folds (numbers in one-element device buffers) and the
/// output divided by the sum.
fn causal(trips: u64, rows: u64, d: u64) -> String {
    decode(trips, rows, d)
        .replace(
            &format!(
                "rrange -> map {{ r -> (0.0, krange) -> fold {{ acc, k -> acc + q[k] * kk[r * {d} + k] }} }} -> s;"
            ),
            &format!(
                "rrange -> map {{ r -> (r <= t) -> {{ -true-> {{ (0.0, krange) -> fold {{ acc, k -> acc + q[k] * kk[r * {d} + k] }} }} -false-> -1000000000.0; }} }} -> s;
                (-1000000000.0, rrange) -> fold {{ acc, r -> s[r] -> e; (e > acc) -> {{ -true-> e; -false-> acc; }} }} -> mx;
                rrange -> map {{ r -> (r <= t) -> {{ -true-> {{ (s[r] - mx) / 16.0 -> ez; ez / 256.0 + 1.0 }} -false-> 0.0; }} }} -> p;
                (0.0, rrange) -> fold {{ acc, r -> acc + p[r] }} -> den;"
            ),
        )
        .replace(
            &format!(
                "krange -> map {{ j -> (0.0, rrange) -> fold {{ acc, r -> acc + s[r] * vv[r * {d} + j] }} }} -> o;"
            ),
            &format!(
                "krange -> map {{ j -> (0.0, rrange) -> fold {{ acc, r -> acc + p[r] * vv[r * {d} + j] }} -> a; a / den }} -> o;"
            ),
        )
}

/// A whole fold's number is device bytes too: on the causal step the cell
/// reading `mx` waits for the fold making it, and `o` for `den` (the edges a
/// number-blind tag dropped: rtx-5080 causal values off under a site pick).
/// Values under every shuffled order and every site candidate equal the OFF
/// arm's.
#[test]
fn causal_region_waits_for_whole_folds() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let src = causal(6, 8, 8);
    assert!(src.contains("-> den;"), "{src}");
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let on = emission(&ir, "rtx-5080", true);
    let (lines, tags) = regions(&on.text);
    assert_eq!(lines.len(), 1, "{}", on.text);
    // Every op after the first reads something an earlier op makes.
    assert!(
        tags.iter()
            .skip(1)
            .all(|&(op, deps, _)| op == 3 || deps != 0),
        "{tags:?}"
    );
    let dev = on.device.as_deref().expect("a device unit");
    let built = emu::build_emulated(&clang, &on.text, dev);
    for pick in ["0", "1", "2", "3", "all"] {
        for seed in ["1", "4", "9"] {
            let (got, _) = built.run(&[("MAPAL_EMU_SHUFFLE", seed), ("MAPAL_GPU_PICK", pick)]);
            assert!(
                close(&got, &rr.output),
                "pick {pick} seed {seed}: {got:?} vs {:?}",
                rr.output
            );
        }
    }
}

/// Equal up to the last bits a site candidate's own fold order moves.
fn close(got: &str, want: &str) -> bool {
    let (g, w): (Vec<&str>, Vec<&str>) = (
        got.split_whitespace().collect(),
        want.split_whitespace().collect(),
    );
    g.len() == w.len()
        && g.iter()
            .zip(&w)
            .all(|(a, b)| match (a.parse::<f64>(), b.parse::<f64>()) {
                (Ok(x), Ok(y)) => (x - y).abs() <= 1e-4 * x.abs().max(y.abs()).max(1e-6),
                _ => a == b,
            })
}

/// The tags on `pt_decode_1024_256` (rtx-5080): one region of six ops — the
/// state row, the two retained GEMV rows, the query, the scores, the output
/// — whose edges are the dataflow's: `kk`, `vv` after the row; the scores
/// after `kk` and `q`; the output after the scores, `vv` and the row that
/// read the last output (its zero pass after that row only).
#[test]
fn decode_region_tags_are_the_dataflow() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../benches/algorithms/pt_decode_1024_256.mapal"
    ))
    .unwrap();
    let ir = lower(&src);
    let opts = EmitOpts {
        target: "rtx-5080",
        ..EmitOpts::default()
    };
    let em = emit_detailed(&ir, &opts).expect("emits");
    let (lines, tags) = regions(&em.text);
    assert_eq!(lines.len(), 1, "{}", em.text);
    assert_eq!(field(&lines[0], "ops"), "6");
    assert_eq!(
        tags,
        vec![
            (0, 0, 0),
            (1, 0b1, 0),
            (2, 0b1, 0),
            (3, 0, 0),
            (4, 0b1010, 0),
            (5, 0b10101, 0b1)
        ]
    );
    // The OFF arm: the emission before the arm.
    let off = emit_detailed(
        &ir,
        &EmitOpts {
            gpu: GpuArms {
                graph: false,
                ..GpuArms::default()
            },
            ..opts
        },
    )
    .expect("emits");
    assert!(!off.text.contains("@mapal_gpu_region_begin"));
    assert!(!off.text.contains("@mapal_gpu_tag"));
}

/// Values: the decode loop's region replayed in shuffled orders equals the
/// interpreter and the OFF arm; one graph a trip, every op a node. A tag with
/// its edges dropped is caught by some order.
#[test]
fn decode_region_values_under_every_order() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let src = decode(6, 8, 8);
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
    let on = emission(&ir, "rtx-5080", true);
    let off = emission(&ir, "rtx-5080", false);
    let (lines, _) = regions(&on.text);
    assert_eq!(lines.len(), 1, "{}", on.text);
    let ops: u64 = field(&lines[0], "ops").parse().unwrap();
    assert!(regions(&off.text).0.is_empty());
    let dev = on.device.as_deref().expect("a device unit");
    let built = emu::build_emulated(&clang, &on.text, dev);
    for seed in ["0", "1", "2", "3", "7", "11"] {
        let (got, err) = built.run(&[("MAPAL_EMU_SHUFFLE", seed)]);
        assert_eq!(got, rr.output, "seed {seed}");
        let (graphs, nodes) = graph_totals(&err);
        assert_eq!(graphs, 6, "one graph a trip: {err}");
        assert!(
            nodes >= 6 * ops,
            "every op a node or more (zero passes): {nodes}"
        );
    }
    let (got, _) = built.run(&[("MAPAL_GPU_GRAPH", "0")]);
    assert_eq!(got, rr.output, "runtime OFF");
    let got = emu::run_emulated(&clang, &off.text, off.device.as_deref().unwrap());
    assert_eq!(got, rr.output, "compile OFF");
    // A missing edge shows: every tag's dependencies dropped.
    let broken: String = on
        .text
        .lines()
        .map(
            |l| match l.trim().strip_prefix("call void @mapal_gpu_tag(i32 ") {
                Some(r) if !r.starts_with('-') => {
                    let op = r.split(',').next().unwrap();
                    format!("  call void @mapal_gpu_tag(i32 {op}, i64 0, i64 0)\n")
                }
                _ => format!("{l}\n"),
            },
        )
        .collect();
    let bad = emu::build_emulated(&clang, &broken, dev);
    let caught =
        (1..=16).any(|seed| bad.run(&[("MAPAL_EMU_SHUFFLE", &seed.to_string())]).0 != rr.output);
    assert!(caught, "no shuffled order exposed the dropped edges");
}

/// Generated loops (`testgen::twin_prog_strategy`, not designed on this
/// arm): with a region, the emulated values under shuffled replay equal the
/// interpreter and the OFF arm, each trip runs at most one graph per region,
/// and a region's tags name only earlier ops.
#[test]
fn launch_graph_on_generated_loops() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let _ = emu::rt_lib();
    let mut runner = TestRunner::deterministic();
    let strat = testgen::twin_prog_strategy();
    let (mut fired, mut findings) = (0, Vec::new());
    for z in 0..160 {
        let p = strat.new_tree(&mut runner).unwrap().current();
        let src = testgen::twin_src(&p, true, None);
        let ir = lower(&src);
        let on = emission(&ir, "rtx-4090", true);
        let (lines, tags) = regions(&on.text);
        if lines.is_empty() {
            continue;
        }
        fired += 1;
        for &(op, deps, out) in &tags {
            if deps >> op != 0 || out & !deps != 0 {
                findings.push(format!("twin#{z}: tag ({op}, {deps:b}, {out:b})"));
            }
        }
        let rr = run(&ir, 400_000_000);
        if !matches!(rr.outcome, Outcome::Done(_)) {
            continue;
        }
        let off = emission(&ir, "rtx-4090", false);
        let got = emu::run_emulated(&clang, &off.text, off.device.as_deref().unwrap());
        if got != rr.output {
            findings.push(format!("twin#{z} OFF: {got:?} vs {:?}", rr.output));
        }
        let built = emu::build_emulated(&clang, &on.text, on.device.as_deref().unwrap());
        for seed in ["0", "5", "9"] {
            let (got, err) = built.run(&[("MAPAL_EMU_SHUFFLE", seed)]);
            if got != rr.output {
                findings.push(format!(
                    "twin#{z} seed {seed}: {got:?} vs {:?}\n{src}",
                    rr.output
                ));
            }
            let (graphs, nodes) = graph_totals(&err);
            if graphs == 0 || nodes < 2 * graphs {
                findings.push(format!("twin#{z}: graphs {graphs} nodes {nodes}"));
            }
        }
    }
    assert!(findings.is_empty(), "{}", findings.join("\n"));
    assert!(fired > 0, "no generated loop had a launch region");
    eprintln!("launch_graph_on_generated_loops: {fired}/160 with a region");
}
