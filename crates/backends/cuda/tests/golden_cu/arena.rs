//! Arena allocation gates and perf timing instrumentation goldens.

use super::*;

// --- plan-smart-arenas §7: the structural perf gates (#18) -------------------

/// Whole-module `cudaMalloc` count EXCLUDING the prelude's `d_trap` pair
/// (plan §7's counting convention).
pub(crate) fn arena_mallocs(cu: &str) -> usize {
    cu.matches("cu_check(cudaMalloc(").count() - 1
}

/// Whole-module `cudaFree` count EXCLUDING the prelude's `d_trap` pair.
pub(crate) fn arena_frees(cu: &str) -> usize {
    cu.matches("cu_check(cudaFree(").count() - 1
}

/// plan-smart-arenas §7's gates, counted on emitted text (deterministic by
/// L2): a fn's non-loop-cone buffer cudaMallocs collapse into ONE arena
/// malloc, the per-buffer frees into one zone release. Loop-cone sites stay
/// per-buffer in v1.0 (the honest debt, pinned below).
#[test]
fn arena_gates_plan_section_7() {
    // The marathon's flagship shape: 8 fn-scope buffers (4 literals +
    // enumerate + map + 2 readback cells) → ONE zone (2 KiB: 8 × 256 B
    // slots). §7's gate was ≤3/≤2 (its "before" of 12 predates W1 — the W1
    // text was 8 + d_trap); the mechanism lands at 1/1.
    let cu = emit(&lower_src(ONE_KERNEL_MATMUL_SRC)).unwrap();
    assert_eq!(arena_mallocs(&cu), 1, "{cu}");
    assert_eq!(arena_frees(&cu), 1, "{cu}");
    assert!(
        cu.contains("cu_check(cudaMalloc((void**)&arena0, 2048ULL), \"cudaMalloc(arena0)\");"),
        "{cu}"
    );
    // Every member gets its compile-time 256 B-aligned pointer init…
    assert_eq!(cu.matches("= (double*)(arena0 + ").count(), 6, "{cu}");
    assert_eq!(
        cu.matches("= (FlowProd_int32_t_double*)(arena0 + ").count(),
        1,
        "{cu}"
    );
    assert_eq!(cu.matches("= (int32_t*)(arena0 + ").count(), 1, "{cu}");
    // …and W1's launch count is unchanged. The §3 checks drop honestly:
    // every launch is trap-free (the two constant readbacks prove (S20);
    // the map launch too — the fold's captured-index reads prove via the
    // S20c capture-range flow and its `t / 4` / `t % 4` are #13-safe
    // constant divisions).
    let main_start = cu.find("static void mapal_main() {").unwrap();
    let main_end = cu[main_start..].find("\n}\n").unwrap() + main_start;
    let main_def = &cu[main_start..main_end];
    assert_eq!(main_def.matches("<<<").count(), 4, "{main_def}");
    assert_eq!(
        main_def.matches("trap_check_after_launch();").count(),
        0,
        "{main_def}"
    );

    // vector_add: main's 7 buffers → one zone (fn1/fn2 are
    // __host__ __device__ scalars — no buffers, no zones). Gate was ≤3/≤2.
    let cu = emit(&build_example("vector_add")).unwrap();
    assert_eq!(arena_mallocs(&cu), 1, "{cu}");
    assert_eq!(arena_frees(&cu), 1, "{cu}");

    // fir: main's two literals → main's zone; fn0's two Index readback
    // cells are advance-cone sites → per-buffer in v1.0 (plan §5's recorded
    // scoping). DEVIATION from §7's fir row (≤3/≤2): the recorded mechanism
    // lands at 3/3 (4/4 with d_trap) — the §7 cell was calibrated without
    // noticing both of fir's cells are cone sites. Recorded in the plan's
    // Status line and suggestions #18.
    let cu = emit(&build_example("fir")).unwrap();
    assert_eq!(arena_mallocs(&cu), 3, "{cu}");
    assert_eq!(arena_frees(&cu), 3, "{cu}");

    // micro_loop_update: rule 4's in-place update killed the cone site's
    // per-buffer malloc AND free — what remains is build's zone (the
    // top-level literal, whose buffer the in-placed update now writes and
    // returns) + main's zone (the readback cell): 2/2 excluding d_trap =
    // 3/3 with it. (The v1.1 debt note moved: the cone Update site no
    // longer allocates at all; its arena offset stays reserved-but-unused —
    // the recorded v1 simplification.)
    let cu = emit(&lower_src(LOOP_UPDATE_SRC)).unwrap();
    assert_eq!(arena_mallocs(&cu), 2, "{cu}");
    assert_eq!(arena_frees(&cu), 2, "{cu}");
    // The zero-iteration contract: every pointer local still declares
    // nullptr-initialized, zone member or not.
    assert!(cu.contains("int32_t* o12 = nullptr;"), "{cu}");
}

#[test]
fn iota_fill_are_arena_members() {
    let cu = emit(&lower_src(IOTA_FILL_SRC)).unwrap();
    assert_eq!(arena_mallocs(&cu), 1, "{cu}");
    assert_eq!(arena_frees(&cu), 1, "{cu}");
    assert!(
        cu.contains("cu_check(cudaMalloc((void**)&arena0, 2048ULL), \"cudaMalloc(arena0)\");"),
        "{cu}"
    );
    assert_eq!(cu.matches("= (int32_t*)(arena0 + ").count(), 6, "{cu}");
    assert_eq!(
        cu.matches("= (FlowProd_int32_t_int32_t*)(arena0 + ")
            .count(),
        2,
        "{cu}"
    );
}

/// The arena swap preserves L2: same IR → byte-identical text, zones
/// included (the assignment walk is topo-ordered, the offsets map is
/// lookup-only).
#[test]
fn arena_emit_twice_byte_equal() {
    for name in ["vector_add", "fir", "sepia"] {
        let a = emit(&build_example(name)).unwrap();
        let b = emit(&build_example(name)).unwrap();
        assert_eq!(a, b, "{name}: emit is not byte-deterministic");
    }
    let a = emit(&lower_src(ONE_KERNEL_MATMUL_SRC)).unwrap();
    let b = emit(&lower_src(ONE_KERNEL_MATMUL_SRC)).unwrap();
    assert_eq!(a, b, "one-kernel matmul: emit is not byte-deterministic");
}

// --- suggestions.md #19a: kernel-time instrumentation ------------------------

/// With `perf_timing` on, every launch site is wrapped in CUDA events and
/// the machine-readable `MAPAL_PERF` lines print; with it off (the `emit`
/// default) the text contains none of that machinery, and the launch /
/// trap-check counts are identical either way (the §3 convention is
/// untouched — the stop event is recorded before the trap check).
#[test]
fn perf_timing_instruments_launches() {
    let ir = build_example("vector_add");
    let on = emit_with_opts(&ir, &EmitOpts { perf_timing: true }).unwrap();
    for needle in [
        // One event pair per launch site, created once at fn entry.
        "cudaEvent_t fev0_start, fev0_stop;",
        "cudaEvent_t fev4_start, fev4_stop;",
        "cu_check(cudaEventCreate(&fev0_start), \"cudaEventCreate\");",
        // The launch wrap: Record(start) … launch … Record(stop) → Sync →
        // ElapsedTime → the MAPAL_PERF line.
        "cu_check(cudaEventRecord(fev0_start), \"cudaEventRecord\");",
        "cu_check(cudaEventRecord(fev0_stop), \"cudaEventRecord\");",
        "cu_check(cudaEventSynchronize(fev0_stop), \"cudaEventSynchronize\");",
        "cu_check(cudaEventElapsedTime(&mapal_perf_ms, fev0_start, fev0_stop), \"cudaEventElapsedTime\");",
        "mapal_perf_total += mapal_perf_ms;",
        "printf(\"MAPAL_PERF launch=k0_0 ms=%.4f\\n\", mapal_perf_ms);",
        // The per-fn total at fn end, then the destroys.
        "printf(\"MAPAL_PERF total ms=%.4f\\n\", mapal_perf_total);",
        "cu_check(cudaEventDestroy(fev0_stop), \"cudaEventDestroy\");",
    ] {
        assert!(on.contains(needle), "missing `{needle}` in:\n{on}");
    }
    // vector_add's 5 launch sites (zip, map, two deduped Index, fold) → 5
    // event pairs and 5 MAPAL_PERF launch lines; one total line (only
    // mapal_main launches — fn1/fn2 are pure-scalar __host__ __device__).
    assert_eq!(on.matches("cudaEventCreate(&fev").count(), 10, "{on}");
    assert_eq!(on.matches("MAPAL_PERF launch=").count(), 5, "{on}");
    assert_eq!(on.matches("MAPAL_PERF total ms=").count(), 1, "{on}");
    // The event stop is recorded BEFORE the trap check where the check
    // rides. vector_add's own sites are all trap-free now (S20: the two
    // constant readbacks prove), so the ordering pins on an unproven
    // readback (`17 % 5` is statically [0,4] ⊄ [0,4)): its Index site
    // keeps the §3 convention, and the stop event precedes the check.
    let idx_ir =
        lower_src("fn main() {\n    [1, 2, 3, 4] -> a: [i32; 4];\n    a[17 % 5] -> println;\n}\n");
    let idx_on = emit_with_opts(&idx_ir, &EmitOpts { perf_timing: true }).unwrap();
    let rec = idx_on.find("cudaEventRecord(fev0_stop)").unwrap();
    let chk = idx_on.find("trap_check_after_launch();").unwrap();
    assert!(rec < chk, "stop recorded before the trap check:\n{idx_on}");

    // Default off: no event machinery anywhere, and `emit` is byte-equal
    // to the defaulted options (the differential/goldens ride the default).
    let off = emit(&ir).unwrap();
    assert!(!off.contains("cudaEvent"), "{off}");
    assert!(!off.contains("MAPAL_PERF"), "{off}");
    assert_eq!(
        off,
        emit_with_opts(&ir, &EmitOpts::default()).unwrap(),
        "emit == emit_with_opts(default)"
    );
    // Launch and trap-check counts are option-invariant.
    assert_eq!(
        on.matches("<<<").count(),
        off.matches("<<<").count(),
        "launch count unchanged"
    );
    assert_eq!(
        on.matches("trap_check_after_launch();").count(),
        off.matches("trap_check_after_launch();").count(),
        "trap-check count unchanged"
    );
    // Instrumentation is deterministic (L2) — the event ordinals ride
    // collect_sites order.
    assert_eq!(
        on,
        emit_with_opts(&ir, &EmitOpts { perf_timing: true }).unwrap(),
        "instrumented emit is byte-deterministic"
    );
}

/// A cone (in-loop) launch is instrumented too — per EXECUTION, so a loop
/// prints one MAPAL_PERF line per iteration — and the fn-scope events are
/// still created once, at fn entry (not per iteration).
#[test]
fn perf_timing_instruments_cone_launches_once_created() {
    let on = emit_with_opts(&lower_src(LOOP_UPDATE_SRC), &EmitOpts { perf_timing: true }).unwrap();
    let build_start = on.find("static int32_t* fn").expect("build def:\n{on}");
    let build_end = on[build_start..]
        .find("\n}\n")
        .map(|e| build_start + e)
        .unwrap();
    let build = &on[build_start..build_end];
    // One site (the advance-cone Update): one pair, created at fn top,
    // destroyed at fn exit — the Record/print lines sit inside the loop.
    assert_eq!(build.matches("cudaEventCreate(&fev").count(), 2, "{build}");
    assert_eq!(build.matches("MAPAL_PERF launch=").count(), 1, "{build}");
    let brk = build.find(") { break; }").unwrap();
    let rec = build.find("MAPAL_PERF launch=").unwrap();
    assert!(
        brk < rec,
        "the cone launch's MAPAL_PERF line is in-loop:\n{build}"
    );
    assert!(build.contains("MAPAL_PERF total ms="), "{build}");
}
