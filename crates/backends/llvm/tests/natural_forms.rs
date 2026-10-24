//! generic-fusion natural forms: the suite's programs are written the way a
//! person writes them — a row write as one map over the cells, an exponent as
//! a fold over a range — and the facts must give them the code their unrolled
//! twins (the shape the generators used to emit) get.
//!
//! - `map_law`'s [`Outside`]: a map whose cells outside an index interval are
//!   a capture's own (`(c ÷ d == t) ? new : x[c]`, `lo ≤ c < hi`, either arm).
//! - `write_delta` takes such a map as the write of its interval, so a decode
//!   loop's row write retains `K`, `V` exactly as the unrolled `Update`s do.
//! - the emitter runs it in place over the interval (`in_place_source`).
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
    (0.5, 64) -> fill -> x;
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
    let cases: [(&str, Option<(u64, i64, i64, u64)>); 7] = [
        // (d, a, b, len)
        (&format!("(c / 8 == r) -> {{ -true-> {new} -false-> x[c]; }}"), Some((8, 1, 0, 8))),
        (&format!("(r == c / 8) -> {{ -true-> {new} -false-> x[c]; }}"), Some((8, 1, 0, 8))),
        // Swapped arms: the interval is the false arm.
        (&format!("(c / 8 != r) -> {{ -true-> x[c]; -false-> {new} }}"), Some((8, 1, 0, 8))),
        (&format!("(c >= r * 8 && c < r * 8 + 8) -> {{ -true-> {new} -false-> x[c]; }}"), Some((1, 8, 0, 8))),
        (&format!("(r * 4 - 2 <= c && r * 4 + 3 > c) -> {{ -true-> {new} -false-> x[c]; }}"), Some((1, 4, -2, 5))),
        // A column (`c % 8 == r`) is no interval, and a body reading `x` off
        // its own cell is no piecewise law.
        (&format!("(c % 8 == r) -> {{ -true-> {new} -false-> x[c]; }}"), None),
        ("(c / 8 == r) -> { -true-> { x[(c + 1) % 64] } -false-> x[c]; }", None),
    ];
    for (body, want) in cases {
        let ir = lower(&one_map(body));
        let got = outsides(&ir).into_iter().flatten().map(|o| (o.d, o.a, o.b, o.len)).next();
        assert_eq!(got, want, "{body}");
    }
}

/// The decode loop (`benches/gpu/gen_tiles.py`'s), natural or unrolled.
fn decode(t_steps: u64, d: u64, elem: &str, column: bool, unrolled: bool) -> String {
    let float = elem == "f32";
    let (zero, w) = if float { ("0.0", "x / 128.0") } else { ("0", "x % 7") };
    let cell = |c: &str| {
        if float {
            format!("({c} * 7 + 13) % 101 - 50 -> widen_f32 -> f; f / 64.0")
        } else {
            format!("({c} * 7 + 13) % 101 - 50")
        }
    };
    let row = if unrolled {
        (0..d)
            .map(|j| {
                let idx = if column { format!("{j} * {d} + t") } else { format!("t * {d} + {j}") };
                format!("{} + o[{j}] -> v{j}; x[{idx}] <- v{j};\n", cell(&format!("(t * {d} + {j})")).replace("-> f;", &format!("-> f{j};")).replace("f / 64.0", &format!("f{j} / 64.0")))
            })
            .collect::<String>()
    } else if column {
        format!(
            "tc -> map {{ c -> c - t -> u; (u >= 0 && u % {d} == 0 && u < {dd}) -> {{ -true-> {{ {v} + o[u / {d}] }} -false-> x[c]; }} }} -> x;\n",
            dd = d * d,
            v = cell(&format!("(t * {d} + u / {d})"))
        )
    } else {
        format!("tc -> map {{ c -> (c / {d} == t) -> {{ -true-> {{ {v} + o[c % {d}] }} -false-> x[c]; }} }} -> x;\n", v = cell("c"))
    };
    let wf = |m: u64, a: u64| {
        let x = format!("(t * {m} + {a}) % 101 - 50");
        if float { format!("{x} -> widen_f32 -> x; {w}") } else { format!("{x} -> x; {w}") }
    };
    let proj = |wn: &str| {
        format!("tc -> map {{ c -> c / {d} -> i; c % {d} -> j; ({zero}, krange) -> fold {{ acc, k -> acc + x[i * {d} + k] * {wn}[k * {d} + j] }} }}")
    };
    let q = if float {
        format!("((t * {d} + k) * 5 + 3) % 97 - 48 -> widen_f32 -> qv; qv / 128.0")
    } else {
        format!("((t * {d} + k) * 5 + 3) % 9 - 4")
    };
    format!(
        "fn main() {{
    {dd} -> iota -> tw;
    tw -> map {{ t -> {wk} }} -> wk;
    tw -> map {{ t -> {wv} }} -> wv;
    {td} -> iota -> tc;
    {d} -> iota -> krange;
    {t_steps} -> iota -> rrange;
    ({zero}, {td}) -> fill -> x0;
    ({zero}, {d}) -> fill -> o0;
    mut x: [{elem}; {td}] <- x0;
    mut o: [{elem}; {d}] <- o0;
    mut t: i32 <- 0;
    loop {{
        (t < {t_steps}) -> {{
            -true-> {{
{row}                {kk} -> kk;
                {vv} -> vv;
                krange -> map {{ k -> {q} }} -> q;
                rrange -> map {{ r -> ({zero}, krange) -> fold {{ acc, k -> acc + q[k] * kk[r * {d} + k] }} }} -> s;
                krange -> map {{ j -> ({zero}, rrange) -> fold {{ acc, r -> acc + s[r] * vv[r * {d} + j] }} }} -> o;
                t + 1 -> t;
                -> loop;
            }}
            -false-> o -> out;
        }}
    }}
    out[0] -> println;
    out[{dl}] -> println;
}}
",
        dd = d * d,
        td = t_steps * d,
        dl = d - 1,
        wk = wf(7, 13),
        wv = wf(11, 57),
        kk = proj("wk"),
        vv = proj("wv"),
    )
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

fn opts(target: &'static str) -> EmitOpts {
    let mut gpu = GpuArms::default();
    for r in ["async", "streams", "pinned", "pdl"] {
        *gpu.off.rung(r).expect("rung") = true;
    }
    EmitOpts { host: "generic", target, gpu, ..EmitOpts::exact() }
}

fn move_lines(text: &str) -> Vec<String> {
    text.lines()
        .filter(|l| l.starts_with("; mapal.move"))
        .map(|l| l.split_once(" carried=").map_or(l, |(_, r)| r).to_string())
        .collect()
}

/// The natural decode gets the unrolled one's facts and movement: the same
/// `Δ` (row `t`, `d` cells from `t·d`; the column write `Top`), the same
/// per-trip movement on every GPU profile, the same launches; and its row
/// write runs in place over the row.
#[test]
fn natural_decode_moves_what_the_unrolled_one_moves() {
    for (elem, column) in [("f32", false), ("i32", false), ("f32", true)] {
        let nat = lower(&decode(16, 8, elem, column, false));
        let unr = lower(&decode(16, 8, elem, column, true));
        assert_eq!(deltas(&nat), deltas(&unr), "{elem} column={column}");
        for target in ["rtx-4090", "h100-pcie", "rtx-3090", "rtx-5090", "generic"] {
            let a = emit_detailed(&nat, &opts(target)).unwrap();
            let b = emit_detailed(&unr, &opts(target)).unwrap();
            assert_eq!(move_lines(&a.text), move_lines(&b.text), "{elem} column={column} {target}");
            let launches = |t: &str| t.matches("call void @mapal_gpu_launch").count();
            assert_eq!(launches(&a.text), launches(&b.text), "{elem} column={column} {target}");
        }
    }
    let rows = deltas(&lower(&decode(16, 8, "f32", false, false)));
    assert!(rows.contains(&Some((8, 8, 0))), "{rows:?}");
    // In place: the state is one buffer, and the row write's loop runs from
    // `max(8t, 0)` to `min(8t + 8, 128)`.
    let host = emit_detailed(&lower(&decode(16, 8, "f32", false, false)), &opts("generic")).unwrap().text;
    assert!(move_lines(&host).iter().all(|l| l.starts_with("0:512:inplace")), "{:?}", move_lines(&host));
    let bounds = host.lines().filter(|l| l.contains(" = select i1 ") && l.ends_with(", i64 128")).count();
    assert_eq!(bounds, 1, "one row-write loop clipped to the array");
}

#[test]
fn natural_decode_matches_the_oracle() {
    let Some(clang) = clang() else {
        eprintln!("skip: clang not found");
        return;
    };
    let _ = rt_lib();
    for (elem, column) in [("f32", false), ("f32", true), ("i32", false)] {
        let ir = lower(&decode(16, 8, elem, column, false));
        let rr = mapal_interp::run(&ir, 100_000_000);
        assert!(matches!(rr.outcome, mapal_interp::Outcome::Done(_)));
        for target in ["rtx-4090", "h100-pcie"] {
            let em = emit_detailed(&ir, &opts(target)).unwrap();
            let got = match &em.device {
                Some(dev) => run_emulated(&clang, &em.text, dev),
                None => native(&clang, &em.text),
            };
            assert_eq!(got, rr.output, "{elem} column={column} {target}");
        }
        assert_eq!(native(&clang, &emit_detailed(&ir, &opts("generic")).unwrap().text), rr.output);
    }
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

/// The range map at the array's edges: an interval below zero (empty), one
/// reaching past either end, two row writes whose union is one range, and a
/// device site over the written state (its delta upload falls back to whole
/// where the rows miss the array).
#[test]
fn range_map_at_the_edges_matches_the_oracle() {
    let Some(clang) = clang() else {
        eprintln!("skip: clang not found");
        return;
    };
    let src = "fn main() {
    4096 -> iota -> tc;
    4096 -> iota -> tw;
    64 -> iota -> krange;
    tw -> map { t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 4096.0 } -> w;
    (0.25, 4096) -> fill -> x0;
    mut x: [f32; 4096] <- x0;
    mut t: i32 <- -2;
    loop {
        (t < 66) -> {
            -true-> {
                tc -> map { c -> (c / 64 == t) -> { -true-> { c -> widen_f32 -> f; f / 64.0 + x[c] } -false-> x[c]; } } -> x;
                tc -> map { c -> (c >= t * 64 - 3 && c < t * 64 + 2) -> { -true-> { x[c] * 0.5 + 1.0 } -false-> x[c]; } } -> x;
                tc -> map { c -> c / 64 -> i; c % 64 -> j; (0.0, krange) -> fold { acc, k -> acc + x[i * 64 + k] * w[k * 64 + j] } } -> y;
                y -> map { v -> v * 0.125 + 0.5 } -> x;
                t + 1 -> t;
                -> loop;
            }
            -false-> x -> out;
        }
    }
    0.0 -> zero: f32;
    (zero, out) -> fold { acc, v -> acc + v } -> s;
    s -> println;
    out[0] -> println;
    out[4095] -> println;
}
";
    let ir = lower(src);
    // The two writes union to `[64t − 3, 64t + 64)`.
    let ds = deltas(&ir);
    assert!(ds.iter().all(Option::is_none) || ds.contains(&Some((67, 64, -3))), "{ds:?}");
    let rr = mapal_interp::run(&ir, 2_000_000_000);
    assert!(matches!(rr.outcome, mapal_interp::Outcome::Done(_)));
    assert_eq!(native(&clang, &emit_detailed(&ir, &opts("generic")).unwrap().text), rr.output);
    for target in ["rtx-4090", "h100-pcie"] {
        let em = emit_detailed(&ir, &opts(target)).unwrap();
        let got = match &em.device {
            Some(dev) => run_emulated(&clang, &em.text, dev),
            None => native(&clang, &em.text),
        };
        assert_eq!(got, rr.output, "{target}");
    }
}

/// softmax's `ex` with the exponent as a fold over a range: one cell kernel,
/// as with the 8 squarings unrolled, and the oracle's values.
#[test]
fn a_fold_over_a_range_stays_in_the_cell() {
    let probe = |ex: &str| {
        format!(
            "fn main() {{
    16384 -> iota -> ta;
    128 -> iota -> krange;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> a;
    ta -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 }} -> b;
    ta -> map {{ t -> t / 128 -> i; t % 128 -> j; (0.0, krange) -> fold {{ acc, k -> acc + a[i * 128 + k] * b[k * 128 + j] }} }} -> c;
    krange -> map {{ i -> (-1000000000.0, krange) -> fold {{ acc, j -> c[i * 128 + j] -> e; (e > acc) -> {{ -true-> e; -false-> acc; }} }} }} -> mx;
    ta -> map {{ t -> (c[t] - mx[t / 128]) / 128.0 -> x; x / 256.0 + 1.0 -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> v0; {ex} }} -> ex;
    krange -> map {{ i -> (0.0, krange) -> fold {{ acc, j -> ex[i * 128 + j] -> e; acc + e }} }} -> den;
    ta -> map {{ t -> ex[t] / den[t / 128] }} -> y;
    0.0 -> zero: f32;
    (zero, y) -> fold {{ acc, v -> acc + v }} -> s;
    s -> println;
    y[0] -> println;
    y[16383] -> println;
}}
"
        )
    };
    let unrolled: String = (0..8).map(|i| format!("v{i} * v{i} -> v{};", i + 1)).collect::<String>() + " v8";
    let natural = "8 -> iota -> sq; (v0, sq) -> fold { acc, k -> acc * acc }";
    let (nat, unr) = (lower(&probe(natural)), lower(&probe(&unrolled)));
    let cells = |ir: &CategoryIr| ir.cell_plan(ir.entry()).len();
    assert_eq!(cells(&nat), cells(&unr));
    let (a, b) = (emit_detailed(&nat, &opts("rtx-4090")).unwrap(), emit_detailed(&unr, &opts("rtx-4090")).unwrap());
    let kinds = |d: &Option<String>| {
        d.as_ref().unwrap().lines().filter(|l| l.starts_with("; mapal.plan kernel=")).map(|l| l.split(" kind=").nth(1).unwrap_or("").split(' ').next().unwrap().to_string()).collect::<Vec<_>>()
    };
    assert_eq!(kinds(&a.device), kinds(&b.device));
    let host_calls = |t: &str| t.lines().filter(|l| l.contains("call void @mapal_gpu_")).count();
    assert_eq!(host_calls(&a.text), host_calls(&b.text));
    let Some(clang) = clang() else { return };
    let rr = mapal_interp::run(&nat, 400_000_000);
    assert_eq!(run_emulated(&clang, &a.text, a.device.as_ref().unwrap()), rr.output);
}
