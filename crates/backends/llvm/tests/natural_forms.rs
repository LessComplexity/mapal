//! generic-fusion natural forms: the suite's programs are written the way a
//! person writes them — a row write as one map over the cells, an exponent as
//! a fold over a range — and the facts must give them the code their unrolled
//! twins (the shape the generators used to emit) get.
//!
//! - `map_law`'s [`Outside`]: a map whose cells outside a set are a capture's
//!   own (`(c ÷ d == t) ? new : x[c]`, `lo ≤ c < hi`, `c % s == t`, either arm).
//! - `write_delta` takes such a map as the write of its interval, so a decode
//!   loop's row write retains `K`, `V` exactly as the unrolled `Update`s do.
//! - the emitter runs it in place over the set (`in_place_source`).
//! - a nested fold with a classifiable body is a cell's work (`cell_plan`,
//!   `device_leaf`): softmax's `ex` stays one cell kernel.

use mapal_backend_llvm::{EmitOpts, GpuArms, emit_detailed};
use mapal_ir::{CategoryIr, Delta, ElemSrc, Operation, Outside};

mod emu;
use emu::{clang, rt_lib, run_emulated};

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"));
    mapal_rewrite::rewrite(ir).ir
}

/// The `Outside` of every map of `ir`'s entry, in program order.
fn outsides(ir: &CategoryIr) -> Vec<Option<Outside>> {
    let f = ir.entry();
    ir.topo_order(f)
        .into_iter()
        .filter(|&m| matches!(ir.morphism(m).unwrap().op, Operation::Map { .. }))
        .map(|m| match ir.map_law(m) {
            Some(ElemSrc::Apply { outside, .. }) => outside,
            _ => None,
        })
        .collect()
}

fn one_map(body: &str) -> String {
    format!(
        "fn main() {{
    64 -> iota -> tc;
    0.5 -> h: f32;
    (h, 64) -> fill -> x;
    3 -> r: i32;
    tc -> map {{ c -> {body} }} -> y;
    y[0] -> println;
    y[63] -> println;
}}
"
    )
}

#[test]
fn piecewise_maps_have_an_outside() {
    let new = "{ c -> widen_f32 -> f; f + x[c] }";
    let all = u64::MAX;
    // (d, a, b, step, len)
    let cases: [(&str, Option<(u64, i64, i64, u64, u64)>); 10] = [
        (&format!("(c / 8 == r) -> {{ -true-> {new} -false-> x[c]; }}"), Some((8, 1, 0, 1, 8))),
        (&format!("(r == c / 8) -> {{ -true-> {new} -false-> x[c]; }}"), Some((8, 1, 0, 1, 8))),
        // Swapped arms: the set is the false arm's.
        (&format!("(c / 8 != r) -> {{ -true-> x[c]; -false-> {new} }}"), Some((8, 1, 0, 1, 8))),
        (&format!("(c >= r * 8 && c < r * 8 + 8) -> {{ -true-> {new} -false-> x[c]; }}"), Some((1, 8, 0, 1, 8))),
        (&format!("(r * 4 - 2 <= c && r * 4 + 3 > c) -> {{ -true-> {new} -false-> x[c]; }}"), Some((1, 4, -2, 1, 5))),
        // One cell.
        (&format!("(c == r * 8 + 5) -> {{ -true-> {new} -false-> x[c]; }}"), Some((1, 8, 5, 1, 1))),
        // A column: every 8th cell from `r`.
        (&format!("(c % 8 == r) -> {{ -true-> {new} -false-> x[c]; }}"), Some((1, 1, 0, 8, all))),
        (&format!("(r + 1 != c % 8) -> {{ -true-> x[c]; -false-> {new} }}"), Some((1, 1, 1, 8, all))),
        // A body reading `x` off its own cell is no piecewise law, nor is a
        // set that is not one progression.
        ("(c / 8 == r) -> { -true-> { x[(c + 1) % 64] } -false-> x[c]; }", None),
        (&format!("(c % 8 == r || c / 8 == r) -> {{ -true-> {new} -false-> x[c]; }}"), None),
    ];
    for (body, want) in cases {
        let ir = lower(&one_map(body));
        let got = outsides(&ir).into_iter().flatten().map(|o| (o.d, o.a, o.b, o.step, o.len)).next();
        assert_eq!(got, want, "{body}");
    }
}

fn opts(target: &'static str, place: bool) -> EmitOpts {
    let mut gpu = GpuArms::default();
    for r in ["async", "streams", "pinned", "pdl"].into_iter().chain((!place).then_some("place")) {
        *gpu.off.rung(r).expect("rung") = true;
    }
    EmitOpts { host: "generic", target, gpu, ..EmitOpts::exact() }
}

fn native(clang: &str, ll: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let (p, x) = (dir.path().join("p.ll"), dir.path().join("p"));
    std::fs::write(&p, ll).unwrap();
    let ok = std::process::Command::new(clang)
        .args(["-O2", "-Wno-override-module"])
        .arg(&p)
        .arg(rt_lib())
        .arg("-o")
        .arg(&x)
        .status()
        .unwrap()
        .success();
    assert!(ok, "clang failed");
    let out = std::process::Command::new(&x).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn deltas(ir: &CategoryIr) -> Vec<Option<(u64, i64, i64)>> {
    ir.write_delta(ir.entry())
        .values()
        .map(|d| match *d {
            Delta::Rows { stride, start, .. } => Some((stride, start.a, start.b)),
            Delta::Top => None,
        })
        .collect()
}

/// The range map at the array's edges: a row below zero (empty), a range
/// reaching past either end, two writes whose union is one range, a column
/// past the row width (empty), a range wider than the array, and a device
/// site over the written state (retained: its delta upload falls back to
/// whole where the rows miss the array) — in place on the host, on the device
/// with every site placed there.
#[test]
fn range_map_at_the_edges_matches_the_oracle() {
    let Some(clang) = clang() else {
        eprintln!("skip: clang not found");
        return;
    };
    let rows = "fn main() {
    256 -> iota -> tc;
    16 -> iota -> krange;
    tc -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 256.0 } -> w;
    (0.25, 256) -> fill -> x0;
    mut x: [f32; 256] <- x0;
    mut z: [f32; 256] <- x0;
    mut t: i32 <- -2;
    loop {
        (t < 18) -> {
            -true-> {
                tc -> map { c -> (c / 16 == t) -> { -true-> { c -> widen_f32 -> f; f / 16.0 + z[c] } -false-> x[c]; } } -> x;
                tc -> map { c -> (c >= t * 16 - 3 && c < t * 16 + 2) -> { -true-> { x[c] * 0.5 + 1.0 } -false-> x[c]; } } -> x;
                tc -> map { c -> c / 16 -> i; c % 16 -> j; (0.0, krange) -> fold { acc, k -> acc + x[i * 16 + k] * w[k * 16 + j] } } -> y;
                y -> map { v -> v * 0.125 + 0.5 } -> z;
                t + 1 -> t;
                -> loop;
            }
            -false-> z -> out;
        }
    }
    0.0 -> zero: f32;
    (zero, out) -> fold { acc, v -> acc + v } -> s;
    s -> println;
    out[0] -> println;
    out[255] -> println;
}
";
    // The column write: `c % 16 == t` matches nothing for `t < 0` or `t ≥ 16`.
    let column = rows.replace(
        "(c / 16 == t) -> { -true-> { c -> widen_f32 -> f; f / 16.0 + z[c] } -false-> x[c]; }",
        "(c % 16 == t) -> { -true-> { c -> widen_f32 -> f; f / 16.0 + z[c] } -false-> x[c]; }",
    );
    // A range wider than the array: its delta is never an upload's.
    let wide = rows.replace("(c >= t * 16 - 3 && c < t * 16 + 2)", "(c >= t * 16 && c < t * 16 + 600)");
    for (src, want) in [(rows, Some((19, 16, -3))), (&column[..], None), (&wide[..], Some((600, 16, 0)))] {
        let ir = lower(src);
        // The writes union to `[16t − 3, 16t + 16)`; a column is no range.
        assert!(deltas(&ir).contains(&want), "{:?}", deltas(&ir));
        let rr = mapal_interp::run(&ir, 2_000_000_000);
        assert!(matches!(rr.outcome, mapal_interp::Outcome::Done(_)));
        let host = emit_detailed(&ir, &opts("generic", true)).unwrap().text;
        assert_eq!(native(&clang, &host), rr.output);
        for target in ["rtx-4090", "h100-pcie"] {
            let em = emit_detailed(&ir, &opts(target, false)).unwrap();
            let dev = em.device.as_ref().expect("the site on the device");
            assert_eq!(run_emulated(&clang, &em.text, dev), rr.output, "{target}");
        }
    }
}

/// Two loops whose states start from one array, each writing a row of it in
/// place (the integration's chain property test, chains #158 and #182): the
/// first loop's in-place range map must write the loop's own copy — the init
/// is read again by the second loop — as its `Update` twin does.
#[test]
fn range_map_over_a_shared_init_copies_it_at_the_entry() {
    let Some(clang) = clang() else {
        eprintln!("skip: clang not found");
        return;
    };
    let src = "fn main() {
    65 -> iota -> ix;
    ix -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 9.0 } -> n0;
    mut s1: [f32; 65] <- n0;
    mut i1: i32 <- 0;
    loop {
        (i1 < 4) -> {
            -true-> {
                ix -> map { e -> (e / 5 == i1) -> { -true-> s1[e] * 2.0 + 1.0; -false-> s1[e]; } } -> s1;
                i1 + 1 -> i1;
                -> loop;
            }
            -false-> s1 -> n1;
        }
    }
    n1[0] -> println;
    mut s3: [f32; 65] <- n0;
    mut i3: i32 <- 0;
    loop {
        (i3 < 1) -> {
            -true-> {
                ix -> map { e -> (e / 5 == i3) -> { -true-> s3[e] * 2.0 + 1.0; -false-> s3[e]; } } -> s3;
                i3 + 1 -> i3;
                -> loop;
            }
            -false-> s3 -> n3;
        }
    }
    n3[0] -> println;
    n0[0] -> println;
}
";
    let ir = lower(src);
    let rr = mapal_interp::run(&ir, 2_000_000_000);
    assert!(matches!(rr.outcome, mapal_interp::Outcome::Done(_)));
    let host = emit_detailed(&ir, &opts("generic", true)).unwrap().text;
    assert_eq!(native(&clang, &host), rr.output, "{host}");
}

/// softmax's `ex` with the exponent as a fold over a range: a device cell
/// (its own kernel, or with `flow` a member of the row unit, whose kernel
/// carries the fold's body), as with the 8 squarings unrolled, the same host
/// launches, and the oracle's values.
#[test]
fn a_fold_over_a_range_stays_in_the_cell() {
    let probe = |ex: &str| {
        format!(
            "fn main() {{
    4096 -> iota -> ta;
    64 -> iota -> krange;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> a;
    ta -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 }} -> b;
    ta -> map {{ t -> t / 64 -> i; t % 64 -> j; (0.0, krange) -> fold {{ acc, k -> acc + a[i * 64 + k] * b[k * 64 + j] }} }} -> c;
    krange -> map {{ i -> (-1000000000.0, krange) -> fold {{ acc, j -> c[i * 64 + j] -> e; (e > acc) -> {{ -true-> e; -false-> acc; }} }} }} -> mx;
    ta -> map {{ t -> (c[t] - mx[t / 64]) / 64.0 -> x; x / 256.0 + 1.0 -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> v0; {ex} }} -> ex;
    krange -> map {{ i -> (0.0, krange) -> fold {{ acc, j -> ex[i * 64 + j] -> e; acc + e }} }} -> den;
    ta -> map {{ t -> ex[t] / den[t / 64] }} -> y;
    0.0 -> zero: f32;
    (zero, y) -> fold {{ acc, v -> acc + v }} -> s;
    s -> println;
    y[0] -> println;
    y[4095] -> println;
}}
"
        )
    };
    let unrolled: String = (0..8).map(|i| format!("v{i} * v{i} -> v{};", i + 1)).collect::<String>() + " v8";
    let natural = "8 -> iota -> sq; (v0, sq) -> fold { acc, k -> acc * acc }";
    let (nat, unr) = (lower(&probe(natural)), lower(&probe(&unrolled)));
    let cells = |ir: &CategoryIr| ir.cell_plan(ir.entry()).len();
    assert_eq!(cells(&nat), cells(&unr));
    let kinds = |d: &Option<String>| -> Vec<String> {
        d.as_ref()
            .expect("a device unit")
            .lines()
            .filter_map(|l| l.strip_prefix("; mapal.plan kernel="))
            .filter_map(|l| l.split(" kind=").nth(1))
            .map(|k| k.split(' ').next().unwrap_or("").to_string())
            .collect()
    };
    let host_calls = |t: &str| t.lines().filter(|l| l.contains("call void @mapal_gpu_")).count();
    let Some(clang) = clang() else { return };
    let rr = mapal_interp::run(&nat, 400_000_000);
    for target in ["rtx-4090", "h100-pcie"] {
        let (a, b) = (emit_detailed(&nat, &opts(target, false)).unwrap(), emit_detailed(&unr, &opts(target, false)).unwrap());
        assert_eq!(kinds(&a.device), kinds(&b.device), "{target}");
        // A device cell: its own kernel, or (`flow`) a cell member of a unit.
        let dev = a.device.as_ref().unwrap();
        let unit_cell = dev.lines().any(|l| {
            l.starts_with("; mapal.flow kernel=") && (l.contains(";ex:cell:") || l.contains("=ex:cell:"))
        });
        assert!(kinds(&a.device).iter().any(|k| k == "cell") || unit_cell, "{:?}", kinds(&a.device));
        assert_eq!(host_calls(&a.text), host_calls(&b.text), "{target}");
        assert_eq!(run_emulated(&clang, &a.text, a.device.as_ref().unwrap()), rr.output, "{target}");
    }
}
