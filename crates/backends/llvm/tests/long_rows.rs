//! long-rows `row-slices`: a flow unit's rows over several blocks — each
//! fold's partials combined by `fold_combine`'s law, the phases its laws
//! leave separated by grid barriers of one cooperative launch. Values through
//! the GPU emulator against the interpreter (free threads, serialized in
//! thread order and reversed, the persistent grid capped below the items so
//! the blocks stride), the intent the fact predicts, on hand-written and
//! generated programs; the rule's prices on the parts' measured properties.

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

/// The emission on `target` and `base`'s face with `s` blocks a row forced
/// (0 = the rule's; 1 = one block a row) and the rungs `off` off; the
/// emulator's rungs off (`async`, `pdl`, `streams`, `pinned`), `place` off
/// so every capable site is on the device, `horizontal` / `calls` off (a
/// unit's own launch).
fn emission(
    ir: &CategoryIr,
    target: &'static str,
    base: EmitOpts,
    s: u64,
    off: &[&str],
) -> Emission {
    let mut gpu = GpuArms::default();
    gpu.on.flow = true;
    gpu.row_slices_s = s;
    for r in [
        "place",
        "async",
        "pdl",
        "streams",
        "pinned",
        "horizontal",
        "calls",
    ]
    .iter()
    .chain(off)
    {
        *gpu.off.rung(r).expect("rung") = true;
    }
    let opts = EmitOpts {
        host: "generic",
        target,
        gpu,
        ..base
    };
    emit_detailed(ir, &opts).expect("emits")
}

/// The sliced units' intent lines.
fn slice_lines(em: &Emission) -> Vec<String> {
    em.device
        .as_deref()
        .unwrap_or("")
        .lines()
        .filter(|l| l.starts_with("; mapal.flow kernel=") && l.contains(" slices="))
        .map(str::to_string)
        .collect()
}

fn field<'l>(line: &'l str, key: &str) -> &'l str {
    line.split_whitespace()
        .find_map(|f| f.strip_prefix(key).and_then(|v| v.strip_prefix('=')))
        .unwrap_or("")
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
                    (Ok(x), Ok(y)) => (x - y).abs() <= 1e-4 * x.abs().max(y.abs()).max(1.0),
                    _ => false,
                }
        })
}

/// The emulator modes every sliced build runs under: free threads, each
/// block's threads serialized between barriers in order and reversed, and
/// the cooperative grid capped at 1 and 2 blocks (every block strides over
/// several items, a barrier's arrivals from a block that ran others first).
const MODES: [[(&str, &str); 2]; 5] = [
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "2"), ("MAPAL_EMU_GRID", "3")],
    [("MAPAL_EMU_SERIAL", "0"), ("MAPAL_EMU_GRID", "1")],
    [("MAPAL_EMU_SERIAL", "1"), ("MAPAL_EMU_GRID", "2")],
];

/// A softmax over `rows` rows of `len` scores (`q · W`, a tile site; row `i`
/// scaled by `i + 1`, so no two rows share a max or a sum): the row max, the
/// polynomial exponential of the shifted score, its row sum, the quotient.
fn softmax(rows: u64, len: u64) -> String {
    let n = rows * len;
    format!(
        "fn main() {{
    {n} -> iota -> t;
    {rows} -> iota -> rr;
    {len} -> iota -> kl;
    8 -> iota -> kd;
    {rq} -> iota -> tq;
    tq -> map {{ e -> (e * 5 + 3) % 31 - 15 -> widen_f32 -> v; e / 8 + 1 -> widen_f32 -> rf; v * rf / 16.0 }} -> q;
    {lw} -> iota -> tw;
    tw -> map {{ e -> (e * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 64.0 }} -> w;
    t -> map {{ e -> e / {len} -> i; e % {len} -> j; (0.0, kd) -> fold {{ acc, d -> acc + q[i * 8 + d] * w[d * {len} + j] }} }} -> x;
    rr -> map {{ i -> (-1000000000.0, kl) -> fold {{ acc, j -> x[i * {len} + j] -> e; (e > acc) -> {{ -true-> e; -false-> acc; }} }} }} -> mx;
    t -> map {{ e -> (x[e] - mx[e / {len}]) / 8.0 -> z; z / 256.0 + 1.0 -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> v0; 8 -> iota -> sq; (v0, sq) -> fold {{ acc, k -> acc * acc }} }} -> ex;
    rr -> map {{ i -> (0.0, kl) -> fold {{ acc, j -> acc + ex[i * {len} + j] }} }} -> den;
    t -> map {{ e -> ex[e] / den[e / {len}] }} -> y;
    y[0] -> println;
    y[{mid}] -> println;
    y[{last}] -> println;
    0.0 -> zero: f32;
    (zero, y) -> fold {{ acc, v -> acc + v }} -> total;
    total -> println;
}}
",
        rq = rows * 8,
        lw = len * 8,
        mid = n / 2 + 5,
        last = n - 1
    )
}

/// The softmax sliced: `max` in phase 0, `ex` / `den` in phase 1 (the
/// polynomial of `x − mx` is no law: a phase), `y` in phase 2 — three
/// phases, two grid barriers, one cooperative launch; values within the
/// default face's bound of the interpreter in every emulator mode; the OFF
/// arm and the exact face one block a row.
#[test]
fn sliced_softmax_values_and_phases() {
    let Some(clang) = emu::clang() else {
        return;
    };
    for (rows, len) in [(3, 1024), (1, 2048)] {
        let src = softmax(rows, len);
        let ir = lower(&src);
        let rr = run(&ir, 2_000_000_000);
        assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
        for s in [2u64, 4, 8] {
            let on = emission(&ir, "rtx-5080", EmitOpts::default(), s, &[]);
            let lines = slice_lines(&on);
            assert_eq!(
                lines.len(),
                1,
                "s={s}: {}",
                on.device.as_deref().unwrap_or("")
            );
            assert_eq!(field(&lines[0], "slices"), s.to_string());
            assert_eq!(field(&lines[0], "phases"), "3");
            assert_eq!(field(&lines[0], "laws"), "");
            assert!(on.text.contains("call void @mapal_gpu_launch_coop("));
            let dev = on.device.as_deref().unwrap();
            assert_eq!(dev.matches("call void @mapal_gsync(").count(), 2);
            assert!(dev.contains("define internal void @mapal_gsync("));
            let ex = emu::build_emulated(&clang, &on.text, dev);
            for mode in MODES {
                let (got, _) = ex.run(&mode);
                assert!(
                    within(&got, &rr.output),
                    "rows={rows} s={s} {mode:?}: {got:?} vs {:?}",
                    rr.output
                );
            }
        }
        // OFF arm and the exact face: one block a row, as before.
        for (base, off) in [
            (EmitOpts::default(), &["row-slices"][..]),
            (EmitOpts::exact(), &[][..]),
        ] {
            let em = emission(&ir, "rtx-5080", base, 4, off);
            assert!(slice_lines(&em).is_empty());
            assert!(!em.text.contains("launch_coop"));
            let got = emu::run_emulated(&clang, &em.text, em.device.as_deref().unwrap());
            assert!(within(&got, &rr.output), "{got:?} vs {:?}", rr.output);
        }
    }
}

/// `fold_combine`'s laws drop phases: a row sum of `x · v / 64` with `v` a
/// row sum of `x` folds `x / 64` beside `v` and multiplies by `v` once a row
/// (a product factor); a row max of `x − m` with `m` a row max folds `x` and
/// subtracts `m` (a shift); a row sum of `x ÷ p` with `p` a per-row cell of
/// a row min divides once (a quotient factor): two phases each, where the
/// phase rule alone takes three. The law's identity is the read the partials
/// take: the values are the interpreter's.
#[test]
fn laws_drop_phases() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let (rows, len) = (2u64, 1024u64);
    let n = rows * len;
    let head = format!(
        "fn main() {{
    {n} -> iota -> t;
    {rows} -> iota -> rr;
    {len} -> iota -> kl;
    8 -> iota -> kd;
    {rq} -> iota -> tq;
    tq -> map {{ e -> (e * 5 + 3) % 31 - 15 -> widen_f32 -> v; v / 16.0 }} -> q;
    {lw} -> iota -> tw;
    tw -> map {{ e -> (e * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 64.0 }} -> w;
    t -> map {{ e -> e / {len} -> i; e % {len} -> j; (0.0, kd) -> fold {{ acc, d -> acc + q[i * 8 + d] * w[d * {len} + j] }} }} -> x;
",
        rq = rows * 8,
        lw = len * 8
    );
    let tail = "    0.0 -> zero: f32;\n    (zero, y) -> fold { acc, u -> acc + u } -> total;\n    total -> println;\n}\n";
    let cases = [
        // (body, laws, phases)
        (
            format!(
                "    rr -> map {{ i -> (0.0, kl) -> fold {{ acc, j -> acc + x[i * {len} + j] }} }} -> v;
    t -> map {{ e -> x[e] * v[e / {len}] / 64.0 }} -> a;
    rr -> map {{ i -> (0.0, kl) -> fold {{ acc, j -> acc + a[i * {len} + j] }} }} -> s1;
    t -> map {{ e -> x[e] + s1[e / {len}] }} -> y;
    s1[1] -> println;
"
            ),
            "mul",
            "2",
        ),
        (
            format!(
                "    rr -> map {{ i -> (-1000000.0, kl) -> fold {{ acc, j -> x[i * {len} + j] -> e; (e > acc) -> {{ -true-> e; -false-> acc; }} }} }} -> m;
    t -> map {{ e -> x[e] - m[e / {len}] }} -> a;
    rr -> map {{ i -> (-1000000.0, kl) -> fold {{ acc, j -> a[i * {len} + j] -> e; (e > acc) -> {{ -true-> e; -false-> acc; }} }} }} -> m2;
    t -> map {{ e -> x[e] * m2[e / {len}] }} -> y;
    m2[0] -> println;
"
            ),
            "sub",
            "2",
        ),
        (
            format!(
                "    rr -> map {{ i -> (1000000.0, kl) -> fold {{ acc, j -> x[i * {len} + j] -> e; (e < acc) -> {{ -true-> e; -false-> acc; }} }} }} -> m;
    rr -> map {{ i -> 1.0 + m[i] * m[i] }} -> p;
    t -> map {{ e -> x[e] / p[e / {len}] }} -> a;
    rr -> map {{ i -> (0.0, kl) -> fold {{ acc, j -> acc + a[i * {len} + j] }} }} -> s1;
    t -> map {{ e -> a[e] - s1[e / {len}] }} -> y;
    s1[1] -> println;
"
            ),
            "div",
            "2",
        ),
    ];
    for (body, laws, phases) in cases {
        let src = format!("{head}{body}    y[7] -> println;\n{tail}");
        let ir = lower(&src);
        let rr = run(&ir, 2_000_000_000);
        assert!(matches!(rr.outcome, Outcome::Done(_)), "{src}");
        let on = emission(&ir, "rtx-5080", EmitOpts::default(), 4, &[]);
        let lines = slice_lines(&on);
        assert_eq!(
            lines.len(),
            1,
            "{src}\n{}",
            on.device.as_deref().unwrap_or("")
        );
        assert_eq!(field(&lines[0], "laws"), laws, "{src}");
        assert_eq!(field(&lines[0], "phases"), phases, "{src}");
        let ex = emu::build_emulated(&clang, &on.text, on.device.as_deref().unwrap());
        for mode in MODES {
            let (got, _) = ex.run(&mode);
            assert!(
                within(&got, &rr.output),
                "{laws} {mode:?}: {got:?} vs {:?}\n{src}",
                rr.output
            );
        }
    }
}

/// Generated programs (the change was not designed on them): few rows of a
/// long axis and a chain of row folds, cells of a row and a per-row value
/// by every law, per-row cells (`testgen::long_row_strategy`), straight-line
/// or in a loop; a third open with a softmax over scores masked to `−∞`
/// (slices and rows wholly masked: the rescale law's identity shift). Every
/// unit the slicing admits runs at two slice counts under every emulator
/// mode against the interpreter and the OFF arm.
#[test]
fn row_slices_on_generated_long_rows() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    // The emulator runs every thread on the host: rows of at most 4096
    // cells in all.
    let progs: Vec<testgen::LongRow> = std::iter::repeat_with(|| {
        testgen::long_row_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current()
    })
    .filter(|p| p.rows * p.len <= 4096)
    .take(160)
    .collect();
    let results = testgen::par_check(&progs, |_, p| check_long_row(&clang, p));
    for r in results.into_iter().flatten() {
        *c.entry(r.0).or_default() += r.1;
    }
    eprintln!("row-slices generated: {c:?}");
    for key in [
        "programs",
        "sliced",
        "phases>1",
        "phases>2",
        "law",
        "law:factor",
        "law:shift",
        "law:rescale",
        "law:rescale masked",
        "stored fold",
        "stored row cell",
        "per-row cell",
        "one row",
        "looped",
        "runs",
    ] {
        assert!(c.get(key).copied().unwrap_or(0) > 0, "{key}: {c:?}");
    }
}

/// One generated program's checks; its coverage counts.
fn check_long_row(clang: &str, p: &testgen::LongRow) -> Vec<(&'static str, u64)> {
    let src = testgen::long_row_src(p);
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    if !matches!(rr.outcome, Outcome::Done(_)) {
        return Vec::new();
    }
    let mut out = vec![("programs", 1)];
    let off = emission(&ir, "rtx-5080", EmitOpts::default(), 4, &["row-slices"]);
    assert!(slice_lines(&off).is_empty());
    let got = emu::run_emulated(clang, &off.text, off.device.as_deref().unwrap_or(""));
    assert!(
        within(&got, &rr.output),
        "OFF: {got:?} vs {:?}\n{src}",
        rr.output
    );
    for s in [2u64, 8] {
        let on = emission(&ir, "rtx-5080", EmitOpts::default(), s, &[]);
        let lines = slice_lines(&on);
        if lines.is_empty() {
            continue;
        }
        out.push(("sliced", lines.len() as u64));
        for l in &lines {
            let phases: u32 = field(l, "phases").parse().unwrap();
            out.push(("phases>1", u64::from(phases > 1)));
            out.push(("phases>2", u64::from(phases > 2)));
            for law in field(l, "laws").split(',').filter(|x| !x.is_empty()) {
                out.push(("law", 1));
                if law == "rescale" && p.masked() {
                    out.push(("law:rescale masked", 1));
                }
                out.push((
                    match law {
                        "mul" | "div" => "law:factor",
                        "rescale" => "law:rescale",
                        _ => "law:shift",
                    },
                    1,
                ));
            }
            let members = field(l, "members");
            let store = field(l, "store");
            for m in members.split(';') {
                let name = m.split(':').next().unwrap_or("");
                let stored = store.split(';').any(|x| x == name);
                if m.contains(":rowfold:") && stored {
                    out.push(("stored fold", 1));
                }
                if m.contains(":cell:1x1") {
                    out.push(("per-row cell", 1));
                }
                if m.contains(":cell:") && !m.contains(":cell:1x1") && stored {
                    out.push(("stored row cell", 1));
                }
            }
            if field(l, "rows") == "1" {
                out.push(("one row", 1));
            }
        }
        if p.looped {
            out.push(("looped", 1));
        }
        let ex = emu::build_emulated(clang, &on.text, on.device.as_deref().unwrap());
        let modes = [MODES[0], MODES[2], MODES[4]];
        for (mode, (got, _)) in modes.iter().zip(ex.run_each(&modes)) {
            assert!(
                within(&got, &rr.output),
                "s={s} {mode:?}: {got:?} vs {:?}\n{src}",
                rr.output
            );
            out.push(("runs", 1));
        }
    }
    out
}

/// The rule on the parts' measured properties (`row_iter_ns`, `gsync_ns`,
/// `coop_launch_ns`, `coop_first_ns`): one row of 131072 slices (the
/// rtx-5080's one block a row walks 1536 lane-iterations at 87 ns; 64 blocks
/// a row walk 24 and meet twice, plus the process's first cooperative
/// launch, 38 us, where the unit runs once); four rows of 32768 slice inside
/// a loop (its trips share that first launch) and not outside one; four rows
/// of 2048 and 32 of 8192 keep one block a row (the cooperative launch and
/// barriers cost more than the walk saves); a part without `row_iter_ns`
/// slices nothing; the OFF arm neither.
#[test]
fn rule_slices_long_rows_on_measured_properties() {
    let pick = |src: &str, target: &'static str, off: &[&str]| {
        let ir = lower(src);
        let em = emission(&ir, target, EmitOpts::default(), 0, off);
        slice_lines(&em)
            .first()
            .map(|l| field(l, "slices").parse::<u64>().unwrap())
    };
    let looped = |rows: u64, len: u64| {
        let n = rows * len;
        format!(
            "fn main() {{
    {n} -> iota -> t;
    {rows} -> iota -> rr;
    {len} -> iota -> kl;
    8 -> iota -> kd;
    {rq} -> iota -> tq;
    tq -> map {{ e -> (e * 5 + 3) % 31 - 15 -> widen_f32 -> v; v / 16.0 }} -> q;
    {lw} -> iota -> tw;
    tw -> map {{ e -> (e * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 64.0 }} -> w;
    t -> map {{ e -> e / {len} -> i; e % {len} -> j; (0.0, kd) -> fold {{ acc, d -> acc + q[i * 8 + d] * w[d * {len} + j] }} }} -> x0;
    mut x: [f32; {n}] <- x0;
    mut it: i32 <- 0;
    loop {{
        (it < 4) -> {{
            -true-> {{
                rr -> map {{ i -> (-1000000000.0, kl) -> fold {{ acc, j -> x[i * {len} + j] -> e; (e > acc) -> {{ -true-> e; -false-> acc; }} }} }} -> mx;
                t -> map {{ e -> x[e] - mx[e / {len}] -> z; z / 256.0 + 1.0 }} -> ex;
                rr -> map {{ i -> (0.0, kl) -> fold {{ acc, j -> acc + ex[i * {len} + j] }} }} -> den;
                t -> map {{ e -> x[e] + ex[e] / den[e / {len}] }} -> x;
                it + 1 -> it;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
    out[0] -> println;
}}
",
            rq = rows * 8,
            lw = len * 8
        )
    };
    assert_eq!(pick(&softmax(1, 131072), "rtx-5080", &[]), Some(64));
    assert_eq!(pick(&softmax(4, 32768), "rtx-5080", &[]), None);
    assert_eq!(pick(&looped(4, 32768), "rtx-5080", &[]), Some(64));
    assert_eq!(pick(&softmax(4, 2048), "rtx-5080", &[]), None);
    assert_eq!(pick(&softmax(32, 8192), "rtx-5080", &[]), None);
    assert_eq!(pick(&softmax(1, 131072), "rtx-4090", &[]), None);
    assert_eq!(pick(&softmax(1, 131072), "rtx-5080", &["row-slices"]), None);
}
