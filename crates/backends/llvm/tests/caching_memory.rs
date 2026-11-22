//! caching-memory: the unchanged-since-last-time fact (`unchanged_plan`) and
//! its consumers.
//!
//! - **The fact.** Across a loop's trips, a value the body computes from a
//!   row-written carried array differs from its last trip's value only in
//!   the rows whose inputs differ: a row fold, a per-row scale, a
//!   projection and a rotation reading its row's partner half carry the
//!   delta (`Rows` for one written range, `Pieces` for a row of every
//!   sequence; a window that reaches into the neighbour row widens it); a
//!   map reading a changing number, or a whole changed array, is `Top`; a
//!   transform of a carried array the loop never writes is unchanged.
//!   Across calls: every value's input slots (constants only: the same at
//!   every call; a clock read: all of them).
//! - **The consumers.** `retain-cell`: a device cell whose output has a row
//!   delta launches over the delta after the loop's first trip;
//!   `retain-pieces`: a retained tile site whose operand changes in one row
//!   of every sequence computes those rows as one gathered site, placed by
//!   `mapal_gpu_copy2d`; the S53 one-row site reads the derived deltas too.
//!   The `unchanged` fact arm turns the derivation off (`write_delta`'s
//!   deltas only).
//! - **Generated programs** (`testgen::history_strategy`): growing-history
//!   loops — one or more sequences, a write in range or piece form, past the
//!   end too — with row folds, per-row scales, capture-free maps,
//!   projections, windows and a carried constant's transpose, against the
//!   interpreter in every emulator mode and with each consumer off.

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, emit_detailed};
use mapal_interp::{Outcome, run};
use mapal_ir::{CategoryIr, Delta, FactArms, ObjectId};
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

/// Not rewritten: `rewrite` evaluates an all-constant program's small loops.
fn lower_raw(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"))
}

fn interp(ir: &CategoryIr) -> String {
    let rr = run(ir, 2_000_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{:?}", rr.outcome);
    rr.output
}

/// The entry's object named `name` that a loop body computes (the last one
/// so named: a rebound state's next value).
fn named(ir: &CategoryIr, name: &str) -> ObjectId {
    let up = ir.unchanged_plan(ir.entry());
    let mut hits: Vec<ObjectId> = up
        .trip
        .keys()
        .chain(up.same.keys())
        .filter(|&o| ir.object(o).and_then(|x| x.name.as_deref()) == Some(name))
        .collect();
    hits.sort();
    *hits
        .last()
        .unwrap_or_else(|| panic!("no loop value `{name}`"))
}

/// `(kind, w, step, a, b)`: a delta's shape (`Rows`: `w = stride`, step 0).
fn shape(d: Delta) -> (char, u64, u64, i64, i64) {
    match d {
        Delta::Rows { stride, start, .. } => ('R', stride, 0, start.a, start.b),
        Delta::Pieces { w, step, start, .. } => ('P', w, step, start.a, start.b),
        Delta::Top => ('T', 0, 0, 0, 0),
    }
}

fn delta_of(ir: &CategoryIr, name: &str) -> (char, u64, u64, i64, i64) {
    let o = named(ir, name);
    shape(ir.unchanged_plan(ir.entry()).delta(o).unwrap_or(Delta::Top))
}

/// A decode-shaped loop over a history of `b` sequences × 8 rows × 4 cells
/// written at position `t + 2`, with the transforms the fact must see
/// through.
fn history(b: u64) -> String {
    format!(
        "fn main() {{
    {hd} -> iota -> i_h;
    i_h -> map {{ t -> (t * 3 + 1) % 5 - 2 -> widen_f32 }} -> h0;
    {bd} -> iota -> i_x;
    i_x -> map {{ t -> t % 3 -> widen_f32 }} -> x0;
    16 -> iota -> i_w;
    i_w -> map {{ t -> (t * 2 + 1) % 5 - 2 -> widen_f32 }} -> w;
    4 -> iota -> g;
    {bl} -> iota -> rBL;
    4 -> iota -> rD;
    {hd} -> iota -> rHD;
    {bl4} -> iota -> rBLN;
    {bd} -> iota -> rBD;
    mut hx: [f32; {hd}] <- h0;
    mut x: [f32; {bd}] <- x0;
    mut t: i32 <- 0;
    loop {{
        (t < 3) -> {{
            -true-> {{
                t + 2 -> p;
                rHD -> map {{ c -> (c / 4 % 8 == p) -> {{ -true-> x[c / 32 * 4 + c % 4]; -false-> hx[c]; }} }} -> hx;
                rBL -> map {{ r -> (0.0, rD) -> fold {{ acc, i -> hx[r * 4 + i] -> v; acc + v * v }} }} -> rs;
                rHD -> map {{ e -> hx[e] * 2.0 + rs[e / 4] }} -> xn;
                rBLN -> map {{ e -> e / 4 -> r; e % 4 -> j; (0.0, rD) -> fold {{ acc, k -> acc + xn[r * 4 + k] * w[k * 4 + j] }} }} -> pr;
                rBLN -> map {{ e -> e / 4 -> r; e % 4 -> u; (u < 2) -> {{ -true-> pr[r * 4 + u] - pr[r * 4 + u + 2]; -false-> pr[r * 4 + u] + pr[r * 4 + u - 2]; }} }} -> ro;
                rBLN -> map {{ e -> p -> widen_f32 -> pf; ro[e] + pf }} -> rp;
                rBD -> map {{ e -> (0.0, rBLN) -> fold {{ acc, i -> acc + ro[i] }} -> s; s * 0.0 + rp[e] }} -> x;
                t + 1 -> t;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
    out[0] -> println;
}}
",
        hd = b * 32,
        bd = b * 4,
        bl = b * 8,
        bl4 = b * 32,
    )
}

/// One sequence: each derived array changes in the written row (`Pieces`
/// with one piece: the write is `(c ÷ 4) % 8 == p`), the rotation's window
/// in two; a map reading the changing number, and a fold over a whole
/// changed array, everywhere.
#[test]
fn unchanged_derives_row_deltas() {
    let ir = lower(&history(1));
    // hx: 4 cells at 4·(t + 2), a piece every 32 (one inside).
    assert_eq!(delta_of(&ir, "rs"), ('P', 1, 8, 1, 2));
    assert_eq!(delta_of(&ir, "xn"), ('P', 4, 32, 4, 8));
    assert_eq!(delta_of(&ir, "pr"), ('P', 4, 32, 4, 8));
    // Row r's partner half `u ± 2`, `u < 4`, reaches rows r − 1 and r + 1
    // (the guard on `u` is not a range fact here): rows p − 1 .. p + 1.
    assert_eq!(delta_of(&ir, "ro"), ('P', 12, 32, 4, 4));
    assert_eq!(delta_of(&ir, "rp").0, 'T');
    assert_eq!(delta_of(&ir, "x").0, 'T');
    assert!(ir.unchanged_plan(ir.entry()).derived >= 4);
}

/// Three sequences: the same deltas, one piece a sequence.
#[test]
fn unchanged_pieces_per_sequence() {
    let ir = lower(&history(3));
    assert_eq!(delta_of(&ir, "rs"), ('P', 1, 8, 1, 2));
    assert_eq!(delta_of(&ir, "pr"), ('P', 4, 32, 4, 8));
    assert_eq!(delta_of(&ir, "ro"), ('P', 12, 32, 4, 4));
}

/// The row written at the trip's end: the transforms read the merge view,
/// which differs from its last trip's value in the row that trip wrote —
/// one position back (`t + 1`).
#[test]
fn unchanged_view_is_the_previous_write() {
    let src = history(1)
        .replace(
            "                rHD -> map { c -> (c / 4 % 8 == p) -> { -true-> x[c / 32 * 4 + c % 4]; -false-> hx[c]; } } -> hx;\n",
            "                x -> map { v -> v } -> xin;\n",
        )
        .replace(
            "                t + 1 -> t;",
            "                rHD -> map { c -> (c / 4 % 8 == p) -> { -true-> xin[c / 32 * 4 + c % 4]; -false-> hx[c]; } } -> hx;\n                t + 1 -> t;",
        );
    let ir = lower(&src);
    assert_eq!(delta_of(&ir, "rs"), ('P', 1, 8, 1, 1));
    assert_eq!(delta_of(&ir, "pr"), ('P', 4, 32, 4, 4));
}

/// The fact arm off: `write_delta`'s deltas only — nothing derived.
#[test]
fn unchanged_off_is_write_delta() {
    let ir = lower(&history(2)).refacted(FactArms {
        unchanged: false,
        ..FactArms::default()
    });
    let up = ir.unchanged_plan(ir.entry());
    assert_eq!(up.derived, 0);
    // Every entry is write_delta's (a view's at the previous trip), or a
    // carried write's back-edge value.
    let wd = ir.write_delta(ir.entry());
    for (o, d) in up.trip.iter() {
        let from_wd = wd.contains_key(o)
            || wd.values().any(|w| {
                matches!(*w, Delta::Rows { next, .. } | Delta::Pieces { next, .. } if next == o)
            });
        assert!(from_wd, "{o:?} {d:?}");
    }
}

/// A row written in range form (`c ÷ 4 == p`, one sequence): `Rows`.
#[test]
fn unchanged_rows_form() {
    let src = history(1).replace("(c / 4 % 8 == p)", "(c / 4 == p)");
    let ir = lower(&src);
    assert_eq!(delta_of(&ir, "rs"), ('R', 1, 0, 1, 2));
    assert_eq!(delta_of(&ir, "pr"), ('R', 4, 0, 4, 8));
    assert_eq!(delta_of(&ir, "ro"), ('R', 12, 0, 4, 4));
}

/// A carried array the loop never writes, and its transform: unchanged on
/// every trip; across calls, a value of constants only is the same at
/// every call, one of the clock is not.
#[test]
fn unchanged_constant_carry_and_calls() {
    let src = "fn main() {
    8 -> iota -> i_w;
    i_w -> map { t -> t % 3 -> widen_f32 } -> w0;
    8 -> iota -> r;
    mut wc: [f32; 8] <- w0;
    mut acc: f32 <- 0.0;
    mut t: i32 <- 0;
    () -> time -> t0;
    loop {
        (t < 3) -> {
            -true-> {
                r -> map { e -> wc[7 - e] * 2.0 } -> wt;
                (0.0, wt) -> fold { a, v -> a + v } -> s;
                acc + s -> acc;
                wc -> wc;
                t + 1 -> t;
                -> loop;
            }
            -false-> acc -> out;
        }
    }
    () -> time -> t1;
    out -> println;
    t1 - t0 -> println;
}
";
    let ir = lower_raw(src);
    let f = ir.entry();
    let up = ir.unchanged_plan(f);
    let wt = named(&ir, "wt");
    assert!(up.same_every_trip(wt), "{up:?}");
    assert!(up.delta(wt).is_none());
    // `wt` is made of constants only; the clock reads are not.
    assert!(up.same_every_call(wt));
    let clock: Vec<ObjectId> = ir
        .objects()
        .filter(|(o, x)| ir.try_owner(*o) == Some(f) && x.name.as_deref() == Some("t1"))
        .map(|(o, _)| o)
        .collect();
    assert!(!clock.is_empty() && clock.iter().all(|&o| !up.same_every_call(o)));
}

/// The GPU emission on the rtx-5080, `place` off (every capable site on the
/// device) and the emulator's rungs off, with `off` off too.
fn device_emission(ir: &CategoryIr, off: &[&str]) -> Emission {
    let mut gpu = GpuArms::default();
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

/// The retained objects the emission counts (`; mapal.move … retained=N`).
fn retained(em: &Emission) -> u64 {
    em.text
        .lines()
        .filter_map(|l| l.split("retained=").nth(1))
        .filter_map(|v| v.split_whitespace().next()?.parse::<u64>().ok())
        .sum()
}

/// The two-sequence history on the emulator: the retained cells launch
/// their delta kernels, the projection's gathered rows are placed by
/// `copy2d`, and every mode, every consumer's OFF arm and the CPU build
/// print the interpreter's values.
#[test]
fn retained_history_on_the_emulator() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let ir = lower(&history(2));
    let want = interp(&ir);
    let em = device_emission(&ir, &[]);
    let dev = em.device.as_deref().unwrap_or("");
    assert!(dev.contains("retain="), "no retained cell kernel");
    assert!(
        em.text.contains("call void @mapal_gpu_copy2d("),
        "no gathered rows"
    );
    assert!(retained(&em) >= 4, "{}", retained(&em));
    let ex = emu::build_emulated(&clang, &em.text, dev);
    for (mode, (got, _)) in MODES.iter().zip(ex.run_each(&MODES)) {
        assert_eq!(got, want, "{mode:?}");
    }
    for arm in ["retain-cell", "retain-pieces", "unchanged"] {
        let off = device_emission(&ir, &[arm]);
        assert!(retained(&off) < retained(&em), "{arm}");
        let got = emu::run_emulated(&clang, &off.text, off.device.as_deref().unwrap_or(""));
        assert_eq!(got, want, "{arm} off");
    }
}

fn check_history(clang: &str, i: usize, p: &testgen::HistoryProg) -> Vec<(&'static str, u64)> {
    let src = testgen::history_prog_src(p);
    let ir = lower(&src);
    let want = interp(&ir);
    let f = ir.entry();
    let up = ir.unchanged_plan(f);
    let mut c = vec![("programs", 1)];
    let derived = up
        .trip
        .values()
        .filter(|d| !matches!(d, Delta::Top))
        .count() as u64;
    c.push(("delta-arrays", derived));
    if up.derived > 0 {
        c.push(("derived-facts", 1));
    }
    if up.trip.values().any(|d| matches!(d, Delta::Rows { .. })) {
        c.push(("rows-facts", 1));
    }
    if up.trip.values().any(|d| matches!(d, Delta::Pieces { .. })) {
        c.push(("piece-facts", 1));
    }
    if up.same.keys().any(|o| {
        matches!(
            ir.object(o).map(|x| &x.ty),
            Some(mapal_ir::Ty::Array { .. })
        )
    }) {
        c.push(("same-arrays", 1));
    }
    let em = device_emission(&ir, &[]);
    let dev = em.device.as_deref().unwrap_or("");
    let cells = dev.matches(" retain=").count() as u64;
    c.push(("retained-cell-kernels", cells));
    c.push(("retained", retained(&em)));
    if em.text.contains("call void @mapal_gpu_copy2d(") {
        c.push(("gathered-sites", 1));
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
        let arm = ["unchanged", "retain-cell", "retain-pieces"][(i / 3) % 3];
        let off = device_emission(&ir, &[arm]);
        let got = emu::run_emulated(clang, &off.text, off.device.as_deref().unwrap_or(""));
        assert_eq!(got, want, "{arm} off\n{src}");
        c.push(("off-arm-runs", 1));
    }
    c
}

/// Generated growing-history loops (`testgen::history_strategy`), each on
/// the emulator in every mode, the CPU build, and a third with a consumer
/// off — against the interpreter. Every path is reached.
#[test]
fn caching_on_generated_programs() {
    let Some(clang) = emu::clang() else {
        return;
    };
    let mut runner = TestRunner::deterministic();
    let progs: Vec<testgen::HistoryProg> = std::iter::repeat_with(|| {
        testgen::history_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current()
    })
    .take(96)
    .collect();
    let results = testgen::par_check(&progs, |i, p| check_history(&clang, i, p));
    let mut c = std::collections::BTreeMap::<&str, u64>::new();
    for r in results.into_iter().flatten() {
        *c.entry(r.0).or_default() += r.1;
    }
    eprintln!("caching-memory generated: {c:?}");
    for key in [
        "programs",
        "delta-arrays",
        "derived-facts",
        "rows-facts",
        "piece-facts",
        "same-arrays",
        "retained-cell-kernels",
        "retained",
        "gathered-sites",
        "off-arm-runs",
    ] {
        assert!(c.get(key).copied().unwrap_or(0) > 0, "{key}: {c:?}");
    }
}
