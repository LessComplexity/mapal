//! Structural-line goldens: abs, sum, sepia, one-kernel matmul and twin folds.

use super::*;

/// The flagship's load-bearing lines, pinned robustly (snapshot-orthogonal).
#[test]
fn abs_structural_lines() {
    let cu = emit(&build_example("abs")).unwrap();
    for needle in [
        "#include <cstdint>",
        "[[noreturn]] void mapal_trap(uint32_t kind);",
        "static unsigned int* d_trap = nullptr;",
        // The negated arm: unsigned-cast wrapping multiply (BC2).
        "(int32_t)((uint32_t)",
        // Phi: both arms into temporaries, then the strict select (BC7).
        "? t0 : t1;",
        // Direct C++ call to the abs fn + its host-only definition.
        "static int32_t fn0(int32_t in)",
        "mapal_print_i32(",
        "  trap_init();",
        "int main() {",
        "  return 0;",
    ] {
        assert!(cu.contains(needle), "missing `{needle}` in:\n{cu}");
    }
    // No kernels, no launches: nothing device-side but the trap flag.
    assert!(!cu.contains("__global__"), "{cu}");
    assert!(!cu.contains("<<<"), "{cu}");
}

/// sum_to_n's WP4 quartet, pinned robustly (snapshot-orthogonal): init copy,
/// `while (true)`, the guard-first break, the back edge, the exit copy.
#[test]
fn sum_to_n_structural_lines() {
    let cu = emit(&build_example("sum_to_n")).unwrap();
    for needle in [
        // Entry: init → merge local, then the host-driven loop.
        "while (true) {",
        // Guard-first: the decide cone's cond guards the advance cone.
        "if (!",
        ") { break; }",
        // The exit copy and the scalar return.
        "return o1;",
    ] {
        assert!(cu.contains(needle), "missing `{needle}` in:\n{cu}");
    }
    // Exactly one loop in the module.
    assert_eq!(cu.matches("while (true) {").count(), 1, "{cu}");
    // The break precedes the back edge; the back edge precedes the exit copy.
    let brk = cu.find(") { break; }").unwrap();
    let back = cu.find("o13.f0;").expect("back edge in:\n{cu}");
    let close = cu[back..].find("\n  }\n").map(|i| back + i).unwrap();
    let exit = cu.find("o1 = o14.f0;").expect("exit copy in:\n{cu}");
    assert!(brk < back, "guard-first: break before back edge:\n{cu}");
    assert!(back < close, "back edge inside the loop body:\n{cu}");
    assert!(close < exit, "exit copy after the loop:\n{cu}");
    // Scalar program: no kernels, no launches.
    assert!(!cu.contains("__global__"), "{cu}");
    assert!(!cu.contains("<<<"), "{cu}");
}

/// The loop-driven `Update` program's load-bearing shapes, pinned robustly:
/// the in-place update (plan-last-use §2 rule 4) — NO per-iteration malloc
/// in the loop body, the target handle IS the source handle (`o12 = o5;`),
/// the element-write kernel launch + its §3 trap check unchanged (out == src
/// is race-free: disjoint per-thread indices), the back-edge pointer swap —
/// and the zone release's range-test veto as the escape guard, now
/// load-bearing: the returned handle is the in-placed init's zone buffer, so
/// the veto fires and the caller inherits the bounded-leak duty (§2
/// amendment (ii)'s recorded shape).
#[test]
fn loop_update_structural_lines() {
    let cu = emit(&lower_src(LOOP_UPDATE_SRC)).unwrap();
    for needle in [
        // The BC5 element-write kernel exists for the Update site (its text
        // is unchanged — out == src writes each index at most once).
        "out[i] = ((int64_t)i == idx) ? val : src[i];",
        // Its launch sits inside the loop, with the §3 check after it.
        "trap_check_after_launch();",
        // Rule 4's in-place update: the target handle is the source handle —
        // no fresh buffer is allocated for the update.
        "o12 = o5;",
        // The loop-carried escape, tier 2 (#18, plan rule 3): the zone
        // release's range-test veto — the returned handle pointing into the
        // arena pins the whole zone.
        "bool escaped0 = ((char*)o1 >= (char*)arena0 && (char*)o1 < (char*)arena0 + 256ULL);",
        "if (!escaped0) {",
    ] {
        assert!(cu.contains(needle), "missing `{needle}` in:\n{cu}");
    }
    // The Update launch is inside the advance cone: after the guard's break.
    let brk = cu.find(") { break; }").unwrap();
    let launch = cu.find("<<<").expect("a kernel launch in:\n{cu}");
    assert!(
        brk < launch,
        "guard-first: the Update launch is post-guard:\n{cu}"
    );
    // Rule 4's payoff: NO cudaMalloc inside build at all for the update —
    // the per-iteration malloc+copy is gone (the one remaining malloc is the
    // fn zone's); the in-placed target is never a registered allocation, so
    // build frees nothing per buffer either.
    let build_start = cu.find("static int32_t* fn").expect("build def:\n{cu}");
    let build_end = cu[build_start..]
        .find("\n}\n")
        .map(|e| build_start + e)
        .unwrap();
    let build = &cu[build_start..build_end];
    assert_eq!(
        build.matches("cu_check(cudaMalloc((void**)&o").count(),
        0,
        "in place: no fresh buffer for the cone Update:\n{build}"
    );
    assert!(
        !build.contains("cudaFree(o12)"),
        "the in-placed target is not a registered allocation:\n{build}"
    );
    // The one launch passes the source handle as out AND src (o12 == o5).
    assert!(
        build.contains(">>>(o12, o5, (int64_t)o6, o10, d_trap);"),
        "{build}"
    );
    // Every remaining free in build is the zone release under the range-test
    // veto (the returned handle points into the arena this time).
    for line in build.lines().filter(|l| l.contains("cudaFree")) {
        assert!(
            line.contains("cu_check(cudaFree(arena0)"),
            "guarded free: {line}\n{cu}"
        );
    }
}

/// S13 orchestrator-review regression (llvm `golden_ll.rs:198` ported): an
/// exit-only computed payload (`acc + 12345`, consumed only by the exit
/// route) belongs to the decide cone but lives outside the loop SCC. The
/// walk must not re-emit it after the loop — the constant appears exactly
/// once in the module.
#[test]
fn exit_only_payload_emitted_once() {
    let src = r#"
fn f(n: i32) -> i32 {
    mut i: i32 <- 0;
    mut acc: i32 <- 0;
    loop {
        (i < n) -> {
            -true-> { acc + 2 -> acc; i + 1 -> i; -> loop; }
            -false-> acc + 12345 -> ret;
        }
    }
}
fn main() { 4 -> f -> r; r -> println; }
"#;
    let cu = emit(&lower_src(src)).unwrap();
    assert_eq!(
        cu.matches("12345").count(),
        1,
        "exit-only payload must be emitted exactly once (driver-owned):\n{cu}"
    );
}

/// The exit-arm `Print` sibling (the OF-1/R1 miscompile class): countdown's
/// `n -> println` rides the token chain into the decide cone — it must be
/// emitted exactly once (a double emission is a DOUBLE side effect per
/// iteration) and it must sit inside the loop (it prints the final 0 on the
/// exit step — ADR-0016, the interp oracle's `5\n4\n3\n2\n1\n0\n`).
#[test]
fn exit_arm_print_emitted_once() {
    // interp acceptance.rs's COUNTDOWN_SRC verbatim.
    let src = r#"
fn countdown(mut n: i32) {
    loop {
        n -> println;
        (n > 0) -> {
            -true-> { n - 1 -> n; -> loop; }
            -false-> -> ret;
        }
    }
}
fn main() { 5 -> countdown; }
"#;
    let cu = emit(&lower_src(src)).unwrap();
    assert_eq!(
        cu.matches("mapal_print_i32(o").count(),
        1,
        "the exit-arm print must be emitted exactly once:\n{cu}"
    );
    // The one emission is inside the while body (runs on the exit step).
    let loop_start = cu.find("while (true) {").unwrap();
    let brk = cu.find(") { break; }").unwrap();
    let print = cu.find("mapal_print_i32(o").unwrap();
    assert!(
        loop_start < print && print < brk,
        "the decide-cone print precedes the guard:\n{cu}"
    );
}

#[test]
fn determinism_emit_twice_byte_equal() {
    for name in [
        "abs",
        "calc",
        "fanout",
        "fir",
        "pipeline",
        "sepia",
        "seq_demo",
        "sum_to_n",
        "vector_add",
        "zip_demo",
    ] {
        let a = emit(&build_example(name)).unwrap();
        let b = emit(&build_example(name)).unwrap();
        assert_eq!(a, b, "{name}: emit is not byte-deterministic");
    }
}

/// sepia's load-bearing WP3 shapes, pinned robustly (snapshot-orthogonal):
/// the BC8 `__host__ __device__` single definition (clamp), the struct-
/// element literal upload, the Pixel map/fold kernels. #14's trim: sepia's
/// whole call graph is trap-free (no Div/Mod/Index/Update anywhere), so no
/// trap parameters, no `d_trap` arguments, and no post-launch readbacks —
/// the launches and their geometry are unchanged.
#[test]
fn sepia_structural_lines() {
    let cu = emit(&build_example("sepia")).unwrap();
    for needle in [
        // clamp: one two-site definition — trap-free (#14), no trap pointer.
        "static __host__ __device__ float fn0(FlowProd_float_float_float in)",
        // The Pixel literal: computed elements ⇒ plain local data array +
        // one H→D memcpy (§2, BC11).
        "FlowProd_float_float_float lit0[16] = { o2,",
        "cudaMemcpyHostToDevice",
        // The map kernel calls the body per element; the fold kernel is the
        // single-thread oracle loop (BC4) — both trap-free, no trap args.
        "out[i] = fn2(in[i]);",
        "acc = fn3(pair);",
        // Scalar fold acc: 1-cell buffer (a fn-zone member — #18's arena
        // pointer init) + D→H readback (§2 item 6).
        "= (float*)(arena0 + ",
        "\"cudaMemcpy(fold)\"",
    ] {
        assert!(cu.contains(needle), "missing `{needle}` in:\n{cu}");
    }
    // The map body calls clamp — no trap pointer threaded anywhere (#14:
    // sepia has no trap-capable fn or site, so the readback never follows
    // a launch; the prelude's [[maybe_unused]] definition always remains).
    assert!(cu.contains("fn0(o15);"), "{cu}");
    assert!(!cu.contains("trap_check_after_launch();"), "{cu}");
    // No twin needed (all bodies are pure-scalar): no __device__-only fns.
    assert!(!cu.contains("static __device__"), "{cu}");
}

/// ADR-0027's acceptance shape (the S16 bench note's finding #3, closed):
/// the one-kernel matmul — an outer map over the 16 cells with the inner
/// fold over k inside the outer body's `__device__` twin, reading the
/// captured operand matrices through the kernel's extra parameters.
pub(crate) const ONE_KERNEL_MATMUL_SRC: &str = r#"
fn main() {
    [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0] -> a;
    [ 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0] -> b;
    [0, 1, 2, 3] -> krange;
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] -> seed;
    seed -> enumerate -> map { p ->
        p.0 -> t;
        t / 4 -> i;
        t % 4 -> j;
        (0.0, krange) -> fold { acc, k -> acc + a[i * 4 + k] * b[k * 4 + j] }
    } -> c;
    c[0] -> println;
    c[15] -> println;
}
"#;

/// ADR-0027's acceptance shape (the S16 bench note's finding #3, closed):
/// the one-kernel matmul. The outer map over the 16 cells is ONE
/// elementwise `__global__`; the fold over k is a per-thread loop INSIDE
/// the outer body's `__device__` twin, reading the captured operand
/// matrices through the kernel's extra parameters. Emitter-quality passes
/// (#17/#12) shrink the module further: main's two Index sites share ONE
/// deduplicated kernel (#17 — four structurally identical Index kernels
/// collapse to the first site's name), and the two Twin bodies get NO host
/// definition at all — nothing on the host path calls them (#12), so their
/// launch-form sites (the inner fold's host-side kernel) emit nothing.
#[test]
fn golden_one_kernel_matmul() {
    let cu = emit(&lower_src(ONE_KERNEL_MATMUL_SRC)).unwrap();

    // Exactly ONE __global__ for the outer map: the elementwise kernel with
    // the captured arrays (krange, b, a) as extra parameters, positionally
    // after `in` — and **no trap pointer** (S20c: the fold's Index reads
    // prove in-bounds via the capture-range flow, and the map body's `t / 4`
    // / `t % 4` are #13-safe constant divisions — the whole chain is
    // trap-free).
    assert!(
        cu.contains(
            "__global__ void k0_1(double* out, FlowProd_int32_t_double* in, int32_t* cap0, double* cap1, double* cap2) {"
        ),
        "{cu}"
    );
    assert_eq!(
        cu.matches("= d_fn2(pair);").count(),
        1,
        "exactly one kernel calls the outer body twin:\n{cu}"
    );
    // The kernel's body call: the captures lead the body-input assembly,
    // then the enumerated (t, seed) element.
    for needle in [
        "pair.f0 = cap0;",
        "pair.f1 = cap1;",
        "pair.f2 = cap2;",
        "pair.f3 = in[i];",
        "out[i] = d_fn2(pair);",
    ] {
        assert!(cu.contains(needle), "missing `{needle}` in:\n{cu}");
    }
    // The launch passes the captured buffers positionally after the mapped
    // array (out, in, krange, a, b — first-use order, ADR-0027; no `d_trap`).
    assert!(
        cu.contains(
            "k0_1<<<(unsigned int)((16ULL + 255ULL) / 256ULL), 256>>>(o8, o6, o4, o2, o3);"
        ),
        "{cu}"
    );

    // The inner fold is a per-thread loop INSIDE the outer body's twin
    // (d_fn2) — the S16 acceptance shape. (The ` {` suffix selects the
    // definition over the prototype.) The twin carries no trap param and no
    // per-step check: the fold body chain is proven trap-free (S20c).
    let d_fn2_start = cu
        .find("static __device__ double d_fn2(FlowProd_int32_tp_doublep_doublep_FlowProd_int32_t_double in) {")
        .expect("d_fn2 def");
    let d_fn2_end = cu[d_fn2_start..].find("\n}\n").unwrap() + d_fn2_start;
    let d_fn2 = &cu[d_fn2_start..d_fn2_end];
    assert!(
        d_fn2.contains("for (unsigned long long t1 = 0; t1 < 4ULL; t1++) {"),
        "{d_fn2}"
    );
    assert!(d_fn2.contains("t0 = d_fn1(pair);"), "{d_fn2}");
    assert!(!d_fn2.contains("trap"), "{d_fn2}");
    // #13: the `t / 4` / `t % 4` guards are elided (constant non-zero
    // divisor) — plain divisions, no `*trap = 1u` anywhere in the twin.
    // WP-B: the divisions inline into the body-arg fields directly — the
    // `(t,4)` wrapper products dissolved, the constants read as themselves.
    assert!(d_fn2.contains(".f1 = (o6 / 4);"), "{d_fn2}");
    assert!(d_fn2.contains(".f3 = (o6 % 4);"), "{d_fn2}");
    assert!(!d_fn2.contains("*trap = 1u"), "{d_fn2}");
    // The whole module is exactly THREE kernels: the enumerate, the outer
    // map, and ONE deduplicated Index kernel (#17 — main's two Index sites
    // and the Twin bodies' host-side Index sites shared one structural
    // shape; the dead twins' sites emit nothing, #12).
    assert_eq!(
        cu.matches("__global__ void").count(),
        3,
        "enumerate + map + one deduped index kernel:\n{cu}"
    );
    // #12: neither Twin body has a host caller, so neither gets a host
    // definition — no fn1/fn2 prototypes or definitions, and no host-side
    // fold kernel (k2_0) anywhere in the module.
    assert!(!cu.contains("static double fn"), "{cu}");
    assert!(!cu.contains("k2_0"), "{cu}");
    // #14 + S20c: EVERY kernel here is trap-free — the enumerate was already
    // free; the map kernel goes free (the fold's Index reads prove via the
    // capture-range flow, the map body's `t / 4` / `t % 4` are #13-safe
    // constant divisions); the two Index readbacks are constants — proven.
    assert!(
        cu.contains("__global__ void k0_0(FlowProd_int32_t_double* out, double* a) {"),
        "{cu}"
    );
    assert!(
        cu.contains("k0_0<<<(unsigned int)((16ULL + 255ULL) / 256ULL), 256>>>(o6, o5);"),
        "{cu}"
    );
    // The device path out of main is exactly enumerate + map + two index
    // readbacks (both launching the one deduped Index kernel) — the fold
    // contributes no launch. The S20 refinement + capture-range flow splits
    // the §3 checks honestly: every launch is trap-free (enumerate; the map
    // — fold Index reads proven + #13-safe divisions; both readbacks —
    // constants proven), so NO launch carries `d_trap` and NO readback
    // follows any launch.
    let main_start = cu.find("static void mapal_main() {").unwrap();
    let main_end = cu[main_start..].find("\n}\n").unwrap() + main_start;
    let main_def = &cu[main_start..main_end];
    assert_eq!(main_def.matches("<<<").count(), 4, "{main_def}");
    assert_eq!(main_def.matches("k0_2<<<").count(), 2, "{main_def}");
    assert_eq!(main_def.matches(", d_trap);").count(), 0, "{main_def}");
    assert_eq!(
        main_def.matches("trap_check_after_launch();").count(),
        0,
        "{main_def}"
    );
    // The proven readback kernel (k0_2): no §3 bounds guard, no trap
    // parameter — the plain global read stays, verbatim.
    let k2_start = cu.find("__global__ void k0_2(").expect("index kernel");
    let k2_end = cu[k2_start..].find("\n}\n").unwrap() + k2_start;
    let k2 = &cu[k2_start..k2_end];
    assert!(
        k2.starts_with("__global__ void k0_2(double* result, double* arr, int64_t idx) {"),
        "{k2}"
    );
    assert!(!k2.contains("trap"), "{k2}");
    assert!(
        k2.contains("*result = arr[(unsigned long long)idx];"),
        "{k2}"
    );

    // The index arithmetic reads the captured `a`/`b` on device: the inner
    // fold body (d_fn1) — its bounds guards are ELIDED (S20c: both reads
    // prove in-bounds via the capture-range flow), the plain global loads
    // stay verbatim, and the twin carries no trap param. (First-use capture
    // order, ADR-0027: (a, i, b, j, acc, elem).)
    let d_fn1_start = cu
        .find("static __device__ double d_fn1(FlowProd_doublep_int32_t_doublep_int32_t_double_int32_t in) {")
        .expect("d_fn1 def");
    let d_fn1_end = cu[d_fn1_start..].find("\n}\n").unwrap() + d_fn1_start;
    let d_fn1 = &cu[d_fn1_start..d_fn1_end];
    assert!(!d_fn1.contains("trap"), "{d_fn1}");
    // WP-B: the proven loads inline into the return chain — still plain
    // guardless global reads through the width-rule i64 temps.
    for needle in [
        "o2[(unsigned long long)t0]", // a[i * 4 + k]
        "o4[(unsigned long long)t1]", // b[k * 4 + j]
    ] {
        assert!(d_fn1.contains(needle), "missing `{needle}` in:\n{d_fn1}");
    }
}

#[test]
fn captured_twin_fold_hoists_invariant_fields() {
    let cu = emit(&lower_src(ONE_KERNEL_MATMUL_SRC)).unwrap();
    let start = cu
        .find(
            "static __device__ double d_fn2(FlowProd_int32_tp_doublep_doublep_FlowProd_int32_t_double in) {",
        )
        .expect("d_fn2 def");
    let end = cu[start..].find("\n}\n").unwrap() + start;
    let twin = &cu[start..end];
    let loop_pos = twin
        .find("for (unsigned long long t1 = 0; t1 < 4ULL; t1++) {")
        .unwrap();

    for field in 0..4 {
        assert!(twin.find(&format!("pair.f{field} =")).unwrap() < loop_pos);
    }
    for field in 4..6 {
        assert!(twin.find(&format!("pair.f{field} =")).unwrap() > loop_pos);
    }
}

/// S23 regression (the box differential's `neg operand` class): an in-twin
/// fold's SCALAR result consumed by a pure scalar op. The query may class it
/// Inline (one consumer, no boundary), but the fold emits as a loop — no
/// expression form exists; DevEmit must force-Name it. Pre-fix this panicked
/// in `load_whole` (27/640 testgen emissions); the local `emit_sweep` example
/// runs the full deterministic sweep without nvcc.
#[test]
fn twin_fold_scalar_result_into_scalar_op_is_named() {
    let src = r#"
fn main() {
    [1, 2, 3, 4] -> xs;
    [10, 20] -> seeds;
    seeds -> map { s ->
        (s, xs) -> fold { acc, x -> acc + x } -> t;
        t * 3
    } -> c;
    c[0] -> println;
    c[1] -> println;
}
"#;
    let cu = emit(&lower_src(src)).unwrap();
    assert!(cu.contains("for (unsigned long long"), "fold loop missing");
    assert_eq!(
        cu.matches("* (uint32_t)3").count(),
        1,
        "fold result duplicated:\n{cu}"
    );
}
