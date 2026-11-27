//! Golden IR for conv tiles, the parallel matmul cap, heap frames, parallel loops and time brackets.

use super::*;

pub(crate) const TILE_CONV_SRC: &str = r#"
fn main() {
    324 -> iota -> ti;
    ti -> map { t -> (t * 7 + 13) % 101 - 50 -> widen<f32> } -> img;
    9 -> iota -> kr;
    kr -> map { t -> (t * 5 + 3) % 31 - 15 -> widen<f32> } -> w;
    256 -> iota -> ts;
    ts -> map { t ->
        t / 16 -> i;
        t % 16 -> j;
        (0.0, kr) -> fold { acc, k -> acc + w[k] * img[(i + k / 3) * 18 + j + k % 3] }
    } -> y;
    y[0] -> println;
    y[255] -> println;
}
"#;

#[test]
fn golden_tile_map_shape_conv() {
    let ir = mapal_rewrite::rewrite(lower_src(TILE_CONV_SRC)).ir;
    // ADR-0038 D4: the `<16 x float>` counts below are `generic`'s geometry.
    let tiled = emit_with_opts(
        &ir,
        &EmitOpts {
            host: "generic",
            target: "generic",
            ..EmitOpts::exact()
        },
    )
    .unwrap();
    let tiled_fn = function_containing(&tiled, " = alloca [16 x float]");
    // The S28 conv rung cashed the k-split record as an unrolled (kq, kr) tap
    // nest. S31 added two things on top (plan-s31-deduced-blocking):
    //   item 2 — the constant-TJ MAIN tile runs on <TJ x elem> SSA values;
    //   item 4 — interior full-window rows run TI at a time, because the
    //            RECORD says b slides (`i_reuse(b) == Sliding{1}`).
    // The nest is therefore three row regions on one counter: head (TI=1),
    // blocked interior (TI=4), tail (TI=1).
    assert!(
        !tiled_fn.contains(" = call float @fn"),
        "tiled conv map must not call its per-cell body:\n{tiled_fn}"
    );
    // The S27c priced refusal, still cashed: zero div/mod in the tile nest.
    assert!(
        !tiled_fn.contains("sdiv") && !tiled_fn.contains("srem"),
        "conv tile nest must contain no sdiv/srem (the taps constant-fold):\n{tiled_fn}"
    );

    // --- item 4: the tap-row union is HOISTED, which is the whole point ------
    // A TI=1 row reaches tap-rows 0..3, so its largest row offset is cq·2 = 36.
    // Offsets 54/72/90 (cq·3, cq·4, cq·5) can only come from a block whose
    // union of tap-rows is (TI−1)·q + k/div = 6 — six image rows serving four
    // output rows. Their presence IS the reuse.
    for off in ["54", "72", "90"] {
        assert!(
            tiled_fn.lines().any(|line| {
                line.contains(" = add i64 ") && line.trim_end().ends_with(&format!(", {off}"))
            }),
            "tap-row offset {off} proves the blocked union was hoisted:\n{tiled_fn}"
        );
    }
    // The reuse factor itself, read off the emission. Head and tail are TI=1
    // vector tiles (9 taps each: 9 loads, 9 FMAs); everything else is the
    // blocked tile. It must issue 18 vector loads for 36 FMAs — exactly the
    // 2.0x that `reuse::distinct_runs` predicts at TI=4, and NOT the 3x of
    // suggestion #11 (that is the TI -> infinity limit).
    let vec_loads = tiled_fn.matches(" = load <16 x float>, ").count();
    let vec_fmuls = tiled_fn.matches(" = fmul <16 x float> ").count();
    assert_eq!(
        (vec_loads, vec_fmuls),
        (36, 54),
        "region decomposition:\n{tiled_fn}"
    );
    let (blocked_loads, blocked_fmuls) = (vec_loads - 2 * 9, vec_fmuls - 2 * 9);
    assert_eq!(
        (blocked_loads, blocked_fmuls),
        (18, 36),
        "the blocked tile must reuse each tap-row across the block:\n{tiled_fn}"
    );
    assert_eq!(blocked_fmuls / blocked_loads, 2, "2.0x reuse at TI=4");

    // --- item 2: no main tile touches accumulator memory ---------------------
    // Every `[16 x float]` GEP is an accumulator access. What remains is the
    // REMAINDER path only: head + tail (11 each) plus the blocked range's four
    // per-row remainders (4 x 11). The main tiles contribute none.
    assert_eq!(
        tiled_fn.matches("getelementptr [16 x float]").count(),
        66,
        "only remainder tiles may address accumulator memory:\n{tiled_fn}"
    );
    assert_eq!(
        tiled_fn.matches(" = fadd <16 x float> ").count(),
        54,
        "one vector accumulation per tap per blocked row:\n{tiled_fn}"
    );
    assert_eq!(
        tiled_fn.matches("store <16 x float> ").count(),
        6,
        "one vector store per main tile: head + tail + TI blocked rows:\n{tiled_fn}"
    );
    assert_eq!(
        tiled_fn.matches(" = insertelement <16 x float> ").count(),
        60,
        "a seed splat per accumulator plus one w[k] splat per FMA:\n{tiled_fn}"
    );
    // No main-path lane loops survive anywhere: seed, taps and store are one
    // vector operation each in every main tile.
    assert_eq!(
        tiled_fn
            .lines()
            .filter(|line| {
                let line = line.trim();
                line.contains(" = icmp uge i64 %t") && line.ends_with(", 16")
            })
            .count(),
        0,
        "no main tile may keep a constant-TJ lane loop:\n{tiled_fn}"
    );
    insta::assert_snapshot!("tile_nest_shape_conv", tiled_fn);
}

#[test]
fn golden_parallel_matmul_cap() {
    let src = r#"
fn main() {
    64 -> iota -> a;
    64 -> iota -> b;
    4 -> iota -> ks;
    16 -> iota -> cells;
    cells -> map { cell_id ->
        cell_id / 4 -> i;
        cell_id % 4 -> j;
        (0, ks) -> fold { acc, k ->
            acc + a[i * 4 + k] * b[k * 4 + j] -> partial;
            a[100] -> guarded;
            partial + guarded
        } -> value;
        value
    } -> c;
    c[0] -> println;
}
"#;
    let ll = emit(&lower_src(src)).unwrap();
    assert!(ll.contains("%Frame = type {"), "parallel frame:\n{ll}");
    assert!(
        ll.contains("define internal void @task0(i64 %lo, i64 %hi, ptr %frame)"),
        "parallel task functions:\n{ll}"
    );
    assert!(
        ll.contains("@ckpt0_entries = private unnamed_addr constant"),
        "packed checkpoint entries:\n{ll}"
    );
    let fold_body = ll
        .split("define internal ")
        .skip(1)
        .filter_map(|s| s.split("\n}\n").next())
        .find(|s| {
            s.contains("getelementptr [64 x i32]") && s.contains("call void @mapal_par_trap(i64 ")
        })
        .expect("fold body");
    assert!(
        fold_body.contains("call void @mapal_par_trap(i64 "),
        "fold body guards speculate into the parallel trap flag:\n{fold_body}"
    );
    assert!(
        !fold_body.contains("call void @mapal_trap(i32"),
        "parallel fold body must not directly trap:\n{fold_body}"
    );
    insta::assert_snapshot!("parallel_matmul_cap", ll);
}

/// A map over `n` cells: two `[n x i32]` frame fields, so `n` alone decides
/// whether the entry frame crosses the heap-lowering threshold.
pub(crate) fn heap_src(n: u32) -> String {
    format!(
        "\nfn main() {{\n    {n} -> iota -> t;\n    \
         t -> map {{ x -> (x * 7 + 13) % 101 - 50 }} -> a;\n    \
         a[{}] -> println;\n}}\n",
        n - 1
    )
}

/// plan-s29 emission item 4 (heap lowering). The parallel entry packs every
/// array into ONE `%Frame`, so `%Frame` is the block that blows the stack —
/// macOS caps the main thread at 64 MB hard, and 2048² f32 ×3 plus the packed
/// panel is ~67 MB. At or above `func.rs:HEAP_MIN_BYTES` (256 KB) the frame
/// becomes a `mapal_rt_alloc` arena block; every field access stays the same
/// `getelementptr %Frame, ptr %frame, …` because an `alloca` result and a
/// `mapal_rt_alloc` result are both just a `ptr`. `mapal_main` then drops
/// exactly one `mapal_rt_free_all`, AFTER `mapal_par_finish` — composition rule
/// 4: no task can still be reading arena memory past the join.
///
/// The size operand is pinned deliberately. It is the emitter's own
/// struct-layout arithmetic (`func.rs:llt_bytes`), verified against LLVM's
/// `ptrtoint (ptr getelementptr (%Frame, ptr null, i32 1) to i64)`, and an
/// under-count is a silent heap overflow rather than a loud failure.
///
/// The n=64 twin is the negative control: below the threshold NOTHING moves,
/// down to the declaration block — which is what keeps every other golden in
/// this file byte-identical.
///
/// Note (plan-s37-stage-structure step 3b): this program's `iota` is now
/// ELIDED — its only consumer rebuilds the element from the index — so the
/// frame holds one array where it used to hold two, and the arena block halved
/// from 800024 to 400024 bytes. What the test is for is unaffected: 390 KB is
/// still over `HEAP_MIN_BYTES` (256 KB), so heap lowering still fires, there is
/// still exactly one arena block and one teardown. Only the pinned literals
/// moved, and they are pinned precisely so a change like this cannot pass
/// unnoticed.
#[test]
fn golden_heap_lowered_frame() {
    let big = emit(&lower_src(&heap_src(100_000))).unwrap();
    assert!(
        big.contains("declare ptr @mapal_rt_alloc(i64, i64)\ndeclare void @mapal_rt_free_all()\n"),
        "the arena ABI is declared:\n{big}"
    );
    assert!(
        big.contains("%Frame = type { [100000 x i32], { ptr, i32 }, i32, i32 }"),
        "the surviving array is a frame field:\n{big}"
    );
    assert!(
        !big.contains("alloca %Frame"),
        "a 390 KB frame must not be a stack block:\n{big}"
    );
    assert!(
        big.contains("  %frame = call ptr @mapal_rt_alloc(i64 400024, i64 8)\n"),
        "the frame is one arena block at LLVM's own sizeof(%Frame):\n{big}"
    );
    let host = mapal_main(&big);
    assert_eq!(
        host.matches("call void @mapal_rt_free_all()").count(),
        1,
        "exactly one teardown:\n{host}"
    );
    assert!(
        host.find("call void @mapal_par_finish")
            .expect("host joins")
            < host
                .find("call void @mapal_rt_free_all")
                .expect("host tears down"),
        "the teardown follows the join:\n{host}"
    );

    let small = emit(&lower_src(&heap_src(64))).unwrap();
    assert!(
        small.contains("  %frame = alloca %Frame\n"),
        "below the threshold the frame stays on the stack:\n{small}"
    );
    assert!(
        !small.contains("mapal_rt_"),
        "a program that heap-allocates nothing gains no declaration:\n{small}"
    );
}

/// S24 review-find pin: a checkpoint INSIDE an effectful loop also fires
/// BEFORE the loop is entered. The loop's seed/entry glue reads task-produced
/// frame slots, so `mapal_par_wait`+`mapal_par_check` must precede the loop CFG
/// in the host body — the per-iteration hook alone would let the first entry
/// read race a still-running task.
#[test]
fn parallel_effectful_loop_waits_before_entry() {
    let src = r#"
fn main() {
    64 -> iota -> t;
    t -> map { x -> (x * 7) % 5 } -> a;
    (0, a) -> fold { acc, x -> acc + x } -> s0;
    s0 % 10 -> s;
    mut i: i32 <- s;
    loop {
        (i > 0) -> {
            -true-> { i -> println; i - 1 -> i; -> loop; }
            -false-> i -> done;
        }
    }
    done -> println;
}
"#;
    let ll = emit(&lower_src(src)).unwrap();
    let host = ll
        .split("define internal void @mapal_main(")
        .nth(1)
        .expect("host fn");
    let wait = host
        .find("call void @mapal_par_wait")
        .expect("host emits a checkpoint wait");
    let first_label = host.find("\nbb").expect("the loop CFG's first label");
    assert!(
        wait < first_label,
        "the in-loop checkpoint's wait+check must precede the loop CFG:\n{host}"
    );
    assert!(
        host.matches("call void @mapal_par_wait").count() >= 2,
        "both the pre-loop and per-iteration checkpoint hooks exist:\n{host}"
    );
}

#[test]
fn parallel_scalar_guard_publishes_watermark() {
    let src = r#"
fn main() {
    4 -> iota -> xs;
    xs[1] -> divisor;
    8 / divisor -> value;
    value -> println;
}
"#;
    let ll = emit(&lower_src(src)).unwrap();
    assert!(
        ll.contains("call void @mapal_par_trap(i64 "),
        "scalar divide speculates:\n{ll}"
    );
    assert!(
        ll.contains("call void @mapal_par_watermark(i64 "),
        "scalar guard publishes its decided watermark:\n{ll}"
    );
}

/// plan-time-builtin: the `time` extern is declared, a bracketed program emits
/// exactly two `mapal_time_ms` calls on the host spine in chain order, and each
/// carries its checkpoint. The S29 fence is the point of the pin: each read
/// waits for every task written entirely BEFORE it in the source, so `t0`
/// fences the generation above the bracket and `t1` fences that PLUS the
/// bracketed kernel — `t1 - t0` is the work written between them, and the
/// generation is excluded (the S28 gen-boundary finding). Keyed on source
/// order because the dataflow graph gives none: pure work is unordered against
/// a clock read, so topo order is free to put all of it on either side, and
/// does. The `fsub` after the second read is the second fence: a `TimeMs`
/// result's consumer cone stays on the host spine, so it can never race the
/// host's write of the clock value.
/// plan-time-builtin rule 1, the loop case: a clock read inside a loop body
/// runs once per ITERATION. Four of lower's six effect detectors keyed on
/// `print` alone when `time` landed, and the two that did not learn it —
/// `emit.rs:effect_chain` (via `loop_body_has_effect`) and `scan_phi_arm` —
/// classify a loop body as pure, which hoists the read out of the cycle: one
/// timestamp for every iteration, silently. This pins the emitted position.
#[test]
fn time_inside_a_loop_stays_inside_the_loop() {
    let src = r#"
fn main() {
    0 -> mut i;
    loop {
        () -> time -> t;
        t -> println;
        i + 1 -> i;
        (i < 3) -> {
            -true-> -> loop;
            -false-> -> ret;
        }
    }
}
"#;
    let ll = emit(&lower_src(src)).unwrap();
    let host = mapal_main(&ll);
    let read = host
        .find("call double @mapal_time_ms()")
        .expect("the clock read is emitted");
    // The loop header is the block the back edge targets; the read must sit
    // after that label and before the back edge that closes the cycle.
    let header = host.find("br label %bb1").expect("loop entry branch");
    let back = host.rfind("br label %bb1").expect("loop back edge");
    assert!(
        header < read && read < back,
        "the clock read belongs inside the loop body, not hoisted above it:\n{host}"
    );
}

#[test]
fn time_bracket_fences_the_tasks_it_brackets() {
    let src = r#"
fn main() {
    8192 -> iota -> xs;
    () -> time -> t0;
    xs -> map { x -> (x * 7) % 13 } -> ys;
    (0, ys) -> fold { acc, y -> acc + y } -> total;
    () -> time -> t1;
    total -> println;
    t1 - t0 -> elapsed;
    elapsed -> println;
}
"#;
    let ll = emit(&lower_src(src)).unwrap();
    assert!(
        ll.contains("declare double @mapal_time_ms()"),
        "the clock extern is declared:\n{ll}"
    );
    assert_eq!(
        ll.matches("call double @mapal_time_ms()").count(),
        2,
        "both clock reads survive, neither CSE'd nor DCE'd:\n{ll}"
    );
    let host = mapal_main(&ll);
    // plan-s33b: each read is a pinned task. It still executes on the host
    // spine, at its own topo position — `mapal_par_run_pinned` is a host call —
    // but as a DAG node, so the work written after it has an edge to wait on.
    let pins: Vec<u32> = host
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix("call void @mapal_par_pin(ptr %h, i32 ")
        })
        .map(|rest| rest.trim_end_matches(')').parse().expect("pinned task id"))
        .collect();
    assert_eq!(pins.len(), 2, "one pinned task per clock read:\n{host}");
    let runs: Vec<usize> = host
        .match_indices("call void @mapal_par_run_pinned(")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(runs.len(), 2, "the host spine runs both reads:\n{host}");
    // The `i32 <len>` argument of the last `mapal_par_wait` before `at`.
    let wait_len = |at: usize| -> u32 {
        let w = host[..at]
            .rfind("call void @mapal_par_wait(")
            .expect("a wait precedes each clock read");
        host[w..]
            .lines()
            .next()
            .unwrap()
            .rsplit("i32 ")
            .next()
            .unwrap()
            .trim_end_matches(')')
            .parse()
            .expect("wait entry count")
    };
    assert!(
        wait_len(runs[0]) > 0,
        "t0 fences the generation written above the bracket:\n{host}"
    );
    assert!(
        wait_len(runs[1]) > wait_len(runs[0]),
        "t1 fences that PLUS the bracketed kernel — strictly more than t0, \
         which is what makes the interval the work written between them:\n{host}"
    );
    // The half that was missing: the bracketed work depends on t0, so it cannot
    // be dispatched before the read that opens the bracket happens.
    assert!(
        host.contains(&format!(
            "call void @mapal_par_dep(ptr %h, i32 {}, i32 ",
            pins[0]
        )),
        "the work written after t0 waits for it:\n{host}"
    );
    assert!(
        host.find("fsub double").expect("the elapsed subtraction") > runs[1],
        "the clock values' consumer cone stays on the host spine, after both reads:\n{host}"
    );
}

/// S20 #6/#8 regression: capture/product staging may forward an array address,
/// but must never materialize the large array as a first-class SSA value.
#[test]
fn capture_array_staging_never_loads_whole() {
    let src = r#"
fn main() {
    32 -> iota -> a;
    32 -> iota -> ids;
    ids -> map { i -> a[i] } -> b;
    b[31] -> println;
}
"#;
    let ll = emit(&lower_src(src)).unwrap();
    assert_eq!(
        ll.matches("load [32 x").count(),
        0,
        "arrays larger than the small-class boundary never load whole:\n{ll}"
    );
}
