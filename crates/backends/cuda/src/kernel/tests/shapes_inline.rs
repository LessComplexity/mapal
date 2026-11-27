//! Tests of kernel text shapes, literal uploads, inline-form device twins and module invariants.

use super::*;

// --- kernel text shapes --------------------------------------------------

#[test]
fn elementwise_kernel_shape_and_grid() {
    let cu = emit_src(
        "fn main() {\n    [1, 2, 3, 4] -> a: [i32; 4];\n    \
             a -> map { x -> x * 2 } -> b;\n    b[17 % 5] -> println;\n}\n",
    );
    // BC3: 64-bit thread index, bounds-guarded, n baked as u64.
    assert!(
        cu.contains(
            "unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;"
        ),
        "{cu}"
    );
    assert!(cu.contains("if (i < 4ULL) {"), "{cu}");
    // Grid = ceil(n/256) computed in 64-bit, at the launch.
    assert!(
        cu.contains("<<<(unsigned int)((4ULL + 255ULL) / 256ULL), 256>>>"),
        "{cu}"
    );
    // The map kernel calls the body — the body is trap-free (#14), so
    // no trap pointer rides the kernel or the call.
    assert!(cu.contains("out[i] = fn1(in[i]);"), "{cu}");
    // The launch's Index sibling CAN trap (`17 % 5` is statically
    // [0,4] ⊄ [0,4) — the S20 proof can't clear it): the §3 check
    // follows it.
    assert!(cu.contains("trap_check_after_launch();"), "{cu}");
}

#[test]
fn enumerate_kernel_index_is_thread_index() {
    let ir = build_example("zip_demo");
    let cu = crate::emit(&ir).unwrap();
    assert!(cu.contains("out[i].f0 = (int32_t)i;"), "{cu}");
    assert!(cu.contains("out[i].f1 = a[i];"), "{cu}");
}

#[test]
fn iota_fill_host_kernels_are_64_bit_trap_free_and_deduped() {
    let cu = emit_src(
        "fn main() {\n    4 -> iota -> a;\n    5 -> iota -> b;\n    \
             (7, 4) -> fill -> c;\n    (8, 5) -> fill -> d;\n}\n",
    );
    assert_eq!(cu.matches("__global__ void").count(), 2, "{cu}");
    assert!(
        cu.contains("__global__ void k0_0(int32_t* out, long long n) {"),
        "{cu}"
    );
    assert!(
        cu.contains("long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;"),
        "{cu}"
    );
    assert!(cu.contains("if (i < n) out[i] = (int32_t)i;"), "{cu}");
    assert!(
        cu.contains("__global__ void k0_2(int32_t* out, long long n, int32_t x) {"),
        "{cu}"
    );
    assert!(cu.contains("if (i < n) out[i] = x;"), "{cu}");
    assert_eq!(cu.matches("k0_0<<<").count(), 2, "{cu}");
    assert_eq!(cu.matches("k0_2<<<").count(), 2, "{cu}");
    assert!(!cu.contains("unsigned int* trap"), "{cu}");
    assert!(!cu.contains("trap_check_after_launch();"), "{cu}");
}

#[test]
fn update_kernel_full_copy_and_guard() {
    let src = "fn main() {\n    mut a: [i32; 4] <- [1, 2, 3, 4];\n    \
                   a[1] <- 9;\n    a[1] -> println;\n}\n";
    let cu = emit_src(src);
    // BC5: the full-copy kernel + bounds guard (index_oob ⇒ 2u, §3's
    // kind+1 flag encoding; width rule).
    assert!(
        cu.contains("if (idx < 0 || idx >= (int64_t)4) { *trap = 2u; return; }"),
        "{cu}"
    );
    assert!(
        cu.contains("out[i] = ((int64_t)i == idx) ? val : src[i];"),
        "{cu}"
    );
    // The host launch passes the extended index + the value by value.
    assert!(cu.contains(", (int64_t)1, 9, d_trap);"), "{cu}");
}

#[test]
fn bounds_guard_width_rule_never_size_t() {
    // u8 (unsigned → zext semantics) and i64 (signed → sext semantics)
    // both realize the width rule as the value-preserving `(int64_t)`
    // conversion + the signed two-sided compare; never size_t. The
    // boundary-OOB constant `3` (size is 3) is exactly the index the
    // S20 proof can never clear — the guard must stay where it fires.
    for idx_decl in ["3 -> i: u8;", "3 -> i: i64;", "3 -> i;"] {
        let src = format!(
            "fn main() {{\n    [1, 2, 3] -> a: [i32; 3];\n    {idx_decl}\n    \
                 a[i] -> println;\n}}\n"
        );
        let cu = emit_src(&src);
        assert!(
            cu.contains("idx < 0 || idx >= (int64_t)3"),
            "idx_decl {idx_decl}:\n{cu}"
        );
        assert!(!cu.contains("(size_t)"), "{cu}");
    }
}

#[test]
fn index_kernel_scalar_readback_d2h() {
    let ir = build_example("vector_add");
    let cu = crate::emit(&ir).unwrap();
    // 1-cell device buffer (a fn-zone member — arena pointer init, #18),
    // <<<1, 1>>> launch, D→H into the host local.
    assert!(cu.contains("int32_t* t"), "{cu}");
    assert!(cu.contains("= (int32_t*)(arena0 + "), "{cu}");
    assert!(cu.contains("<<<1, 1>>>"), "{cu}");
    // The index readback specifically — the prelude's trap-flag memcpy
    // would satisfy a bare "cudaMemcpyDeviceToHost" in every module.
    assert!(
        cu.contains("cudaMemcpyDeviceToHost), \"cudaMemcpy(index)\""),
        "{cu}"
    );
    assert!(
        cu.contains("*result = arr[(unsigned long long)idx];"),
        "{cu}"
    );
}

#[test]
fn index_kernel_array_elem_copies_in_kernel() {
    let src = "fn main() {\n    [[1, 2], [3, 4]] -> m: [[i32; 2]; 2];\n    \
                   m[1] -> row: [i32; 2];\n    row[0] -> println;\n}\n";
    let cu = emit_src(src);
    // Array element: fresh device buffer + in-kernel device-to-device copy.
    assert!(
        cu.contains("for (unsigned long long j = 0; j < 2ULL; j++) {"),
        "{cu}"
    );
    assert!(
        cu.contains("result[j] = arr[((unsigned long long)idx * 2ULL) + j];"),
        "{cu}"
    );
    // No D→H readback for the array result (it stays on device); the
    // only D→H is the scalar row[0] index + the trap-flag reads.
    assert!(!cu.contains("cudaMemcpy(row"), "{cu}");
}

#[test]
fn fold_kernel_single_thread_oracle_order() {
    let ir = build_example("vector_add");
    let cu = crate::emit(&ir).unwrap();
    // BC4: single-thread kernel, the oracle loop verbatim — acc slot 0,
    // (acc, e) per step, in-order. The fold body is trap-free (#14):
    // no trap parameter and no per-step check (vector_add's only
    // would-be trap sources are its two Index sites — constant
    // readbacks the S20 proof clears, so they trim too).
    assert!(cu.contains("int32_t acc = acc0;"), "{cu}");
    assert!(cu.contains("pair.f0 = acc;"), "{cu}");
    assert!(cu.contains("pair.f1 = arr[i];"), "{cu}");
    assert!(cu.contains("acc = fn2(pair);"), "{cu}");
    assert!(cu.contains("*result = acc;"), "{cu}");
    // Scalar acc: 1-cell buffer + D→H readback at the site.
    assert!(cu.contains("\"cudaMemcpy(fold)\""), "{cu}");
}

#[test]
fn fold_kernel_array_acc_stays_on_device() {
    let src = "fn main() {\n    \
                   ([0, 0], [[1, 2], [3, 4]]) -> fold { acc, row -> acc } -> r: [i32; 2];\n    \
                   r[1] -> println;\n}\n";
    let cu = emit_src(src);
    // §5: acc in per-thread local storage (statically bounded).
    assert!(cu.contains("int32_t acc[2];"), "{cu}");
    assert!(cu.contains("acc[j] = acc0[j];"), "{cu}");
    // Array acc ⇒ twin body with the out-param convention. The body is
    // a pure passthrough — trap-free (#14): no trap argument.
    assert!(cu.contains("d_fn"), "{cu}");
    assert!(cu.contains(", acc);"), "{cu}");
    // The result is the device buffer — written back in-kernel.
    assert!(cu.contains("result[j] = acc[j];"), "{cu}");
}

// --- literal uploads ------------------------------------------------------

#[test]
fn literal_upload_static_const_one_memcpy() {
    let ir = build_example("vector_add");
    let cu = crate::emit(&ir).unwrap();
    assert!(
        cu.contains("static const int32_t lit0[16] = { 0, 1, 2, 3, 4, 5, 6, 7"),
        "{cu}"
    );
    // Exactly one H→D memcpy per literal (two literals here) — BC11.
    assert_eq!(cu.matches("cudaMemcpyHostToDevice").count(), 2, "{cu}");
    // Both buffers are fn-zone members (#18): arena pointer inits, and
    // the zone release replaces the per-buffer frees.
    assert!(cu.contains("o2 = (int32_t*)(arena0 + 0ULL);"), "{cu}");
    assert!(cu.contains("o3 = (int32_t*)(arena0 + 256ULL);"), "{cu}");
    assert!(!cu.contains("cudaFree(o2)"), "{cu}");
    assert!(cu.contains("cudaFree(arena0)"), "{cu}");
}

#[test]
fn literal_computed_elements_plain_local() {
    // sepia's [Pixel; 16]: struct-valued elements — plain local data
    // array (not static const), same one-memcpy shape.
    let ir = build_example("sepia");
    let cu = crate::emit(&ir).unwrap();
    assert!(
        cu.contains("FlowProd_float_float_float lit0[16] = { o2,"),
        "{cu}"
    );
    assert!(!cu.contains("static const FlowProd"), "{cu}");
    assert_eq!(cu.matches("cudaMemcpyHostToDevice").count(), 1, "{cu}");
}

#[test]
fn literal_nested_elements_device_to_device() {
    let src = "fn main() {\n    [[1, 2], [3, 4]] -> m: [[i32; 2]; 2];\n    \
                   m[1] -> row: [i32; 2];\n    row[0] -> println;\n}\n";
    let cu = emit_src(src);
    // The inner arrays upload H→D (each is a scalar literal); the outer
    // is a flat buffer filled by device-to-device copies.
    assert_eq!(cu.matches("cudaMemcpyDeviceToDevice").count(), 2, "{cu}");
    // The outer buffer is a fn-zone member (#18) — arena-resident, not a
    // per-buffer cudaMalloc. Its OFFSET is deliberately not pinned here:
    // `arena_plan` assigns 256 B slots walking `topo_order`, and since
    // plan-s38 that walk breaks ties on source position, so which member
    // lands at which slot is an ordering fact, not a correctness one (it
    // moved 512 → 256 when the tie-break changed, with every region still
    // disjoint). Slot disjointness and alignment are pinned where they
    // belong, on the plan itself: `arena::tests::
    // offsets_are_disjoint_aligned_and_topo_ordered`.
    assert!(cu.contains("o4 = (int32_t*)(arena0 + "), "{cu}");
    // What IS a correctness property: each inner literal is copied into
    // the outer flat buffer at its own element offset, two elements wide.
    assert!(
        cu.contains("cudaMemcpy(o4 + 0ULL, o2, sizeof(int32_t) * 2ULL"),
        "{cu}"
    );
    assert!(
        cu.contains("cudaMemcpy(o4 + 2ULL, o3, sizeof(int32_t) * 2ULL"),
        "{cu}"
    );
}

// --- inline form (device twins) -------------------------------------------

#[test]
fn twin_inline_map_is_per_thread_sequential_loop() {
    // A map body containing a nested map: the twin runs the inner map as
    // a per-thread sequential loop; the outer kernel calls the twin.
    // Both bodies are trap-free (#14): no trap threading anywhere.
    let src = "fn main() {\n    [1, 2] -> map { x -> [x, x] -> map { y -> y + 1 } } -> rs;\n    \
                   rs[0][0] -> println;\n}\n";
    let cu = emit_src(src);
    assert!(cu.contains("static __device__"), "{cu}");
    // The twin's inner-map loop, no trap references at all (#14).
    let twin = twin_slice(&cu);
    assert!(twin.contains("for (unsigned long long"), "{twin}");
    assert!(!twin.contains("trap"), "{twin}");
    // The outer kernel calls the twin with the out-offset destination.
    assert!(cu.contains("d_fn"), "{cu}");
}

#[test]
fn twin_iota_fill_are_local_loops() {
    let src = r#"
fn main() {
    [1, 2] -> map { x ->
        4 -> iota -> a;
        (x, 4) -> fill -> b;
        a[1] + b[2]
    } -> rs;
    rs[0] -> println;
}
"#;
    let cu = emit_src(src);
    let twin = twin_slice(&cu);
    assert_eq!(twin.matches("for (unsigned long long").count(), 2, "{twin}");
    assert!(twin.contains("] = (int32_t)"), "{twin}");
    assert!(twin.contains("] = o"), "{twin}");
    assert!(!twin.contains("<<<"), "{twin}");
}

#[test]
fn twin_body_local_literal_initializer() {
    // §2 item 8: a body-local array literal is a per-thread local
    // initializer, never a host static.
    let src = "fn main() {\n    [1, 2] -> map { x -> [x, x + 1][0] } -> rs;\n    \
                   rs[0] -> println;\n}\n";
    let cu = emit_src(src);
    let twin = twin_slice(&cu);
    // Per-thread local array + per-element stores.
    assert!(twin.contains("[2];"), "{twin}");
    assert!(twin.contains("[0] = "), "{twin}");
    assert!(twin.contains("[1] = "), "{twin}");
    // The host static+memcpy mechanism does NOT appear in the twin.
    assert!(!twin.contains("cudaMemcpy"), "{twin}");
    assert!(!twin.contains("static const"), "{twin}");
}

#[test]
fn twin_index_guard_sets_flag_and_returns() {
    let src = "fn main() {\n    [1, 2] -> map { x -> [x, x + 1][x] } -> rs;\n    \
                   rs[0] -> println;\n}\n";
    let cu = emit_src(src);
    let twin = twin_slice(&cu);
    // The device bounds guard: extend to int64_t, two-sided compare,
    // set the flag + return (§3) — never mapal_trap on device.
    assert!(twin.contains("int64_t t"), "{twin}");
    assert!(twin.contains("< 0 ||"), "{twin}");
    assert!(twin.contains("*trap = 2u; return int32_t{};"), "{twin}");
    assert!(!twin.contains("mapal_trap"), "{twin}");
}

#[test]
fn twin_proven_index_elides_guard() {
    // The S20 row-1 half, in-twin: the constant-index read
    // `[x, x + 1][0]` is provably in-bounds (0 < 2) — the §3 guard is
    // dead text, so the twin emits the extension temp + the plain
    // per-thread read and nothing else. With its only trap source
    // elided the twin is trap-free (#14): no trap parameter, no flag
    // references at all (the unproven sibling above keeps everything).
    let src = "fn main() {\n    [1, 2] -> map { x -> [x, x + 1][0] } -> rs;\n    \
                   rs[0] -> println;\n}\n";
    let cu = emit_src(src);
    let twin = twin_slice(&cu);
    assert!(twin.contains("int64_t t"), "{twin}");
    assert!(!twin.contains("< 0 ||"), "{twin}");
    assert!(twin.contains("[(unsigned long long)t"), "{twin}");
    assert!(!twin.contains("trap"), "{twin}");
}

#[test]
fn twin_div_guard_sets_flag_and_returns() {
    // The body has a literal (bulk ⇒ twin) AND a Div: the device zero
    // guard stores the div_zero encoding (1u) and returns.
    let src = "fn main() {\n    [10, 0] -> map { x -> [100 / x, x][0] } -> rs;\n    \
                   rs[0] -> println;\n}\n";
    let cu = emit_src(src);
    let twin = twin_slice(&cu);
    assert!(
        twin.contains("== 0) { *trap = 1u; return int32_t{}; }"),
        "{twin}"
    );
}

#[test]
fn twin_float_neg_of_negative_constant_is_parenthesized() {
    // The twin (inline-form) Neg has the same rule as the host table
    // (F5): `-({val})`, never the ill-formed `--1.5e0`.
    let src = "fn main() {\n    \
                   [1.0, 2.0] -> map { x ->\n        \
                       -1.5 -> c;\n        \
                       -c -> d;\n        \
                       [d, x][0] -> ret\n    \
                   } -> rs;\n    \
                   rs[0] -> println;\n}\n";
    let cu = emit_src(src);
    let twin = twin_slice(&cu);
    // WP-B: the Neg may inline into its consumer — the pin is the
    // parenthesized form itself, wherever it lands, and the absence of
    // the ill-formed `--` ANYWHERE in the twin.
    assert!(!twin.contains("--"), "{twin}");
    assert!(twin.contains("-(-1.5e0)"), "{twin}");
}

#[test]
fn trap_flag_stores_encode_kind_plus_one() {
    // §3's flag encoding: the flag is cudaMemset-zeroed (0 = quiescent),
    // so a guard stores kind + 1 — div_zero ⇒ 1u, index_oob ⇒ 2u — and
    // the host readback decodes with mapal_trap(kind - 1) back to the
    // mapal-rt kinds 0/1. A bare-kind store would collide: div_zero's 0
    // would read back as "no trap" (the R1 class cross).
    let src = "fn main() {\n    [10, 0] -> map { x -> [100 / x, x][x] } -> rs;\n    \
                   rs[0] -> println;\n}\n";
    let cu = emit_src(src);
    let twin = twin_slice(&cu);
    // Exact store texts: div_zero ⇒ 1u, index_oob ⇒ 2u.
    assert!(
        twin.contains("== 0) { *trap = 1u; return int32_t{}; }"),
        "{twin}"
    );
    assert!(
        twin.contains(") { *trap = 2u; return int32_t{}; }"),
        "{twin}"
    );
    // The host decode maps 1→0 (div_zero) and 2→1 (index_oob).
    assert!(cu.contains("mapal_trap(kind - 1);"), "{cu}");
}

#[test]
fn host_device_fn_div_guard_is_ifdef_split() {
    // A pure-scalar body with a Div: single __host__ __device__
    // definition; the guard is compiled per-site by the preprocessor.
    let src = "fn main() {\n    [10, 0] -> map { x -> 100 / x } -> rs;\n    \
                   rs[0] -> println;\n}\n";
    let cu = emit_src(src);
    assert!(cu.contains("static __host__ __device__"), "{cu}");
    assert!(cu.contains("#ifdef __CUDA_ARCH__"), "{cu}");
    assert!(cu.contains("*d_trap = 1u; return int32_t{};"), "{cu}");
    assert!(cu.contains("#else"), "{cu}");
    assert!(cu.contains("mapal_trap(0);"), "{cu}");
    assert!(cu.contains("#endif"), "{cu}");
}

#[test]
fn device_call_trap_check_after_call() {
    // The nested-map twin calls the inner body per element and checks
    // the flag after each call (first-trap-wins, §3) — the inner body
    // here is trap-CAPABLE (the Div's zero guard), so the full trap
    // convention rides (#14 trims it only for trap-free callees).
    let src = "fn main() {\n    [1, 2] -> map { x -> [x, x] -> map { y -> y / x } } -> rs;\n    \
                   rs[0][0] -> println;\n}\n";
    let cu = emit_src(src);
    let twin = twin_slice(&cu);
    assert!(twin.contains(", trap);"), "{twin}");
    assert!(twin.contains("if (*trap) return;"), "{twin}");
}

#[test]
fn twin_signature_array_return_uses_out_param() {
    let ir = lower_src(
        "fn main() {\n    ([0, 0], [[1, 2], [3, 4]]) -> fold { acc, row -> acc } -> r;\n    \
             r[0] -> println;\n}\n",
    );
    let fnames = fnames_of(&ir);
    let caps = TrapCaps::analyze(&ir);
    for (id, fd) in ir.funcs() {
        if fd.kind == FuncKind::FoldBody {
            let sig = twin_signature(&ir, id, &fnames, &caps);
            // The body is a pure passthrough — trap-free (#14), so the
            // trailing trap parameter is trimmed.
            assert_eq!(
                sig,
                "static __device__ void d_fn1(FlowProd_int32_tp_int32_tp in, int32_t* out)"
            );
        }
    }
}

// --- module-level invariants ----------------------------------------------

#[test]
fn emission_order_device_before_kernels_before_host() {
    let ir = build_example("vector_add");
    let cu = crate::emit(&ir).unwrap();
    let dev = cu.find("__device__").expect("device fn");
    let kern = cu.find("__global__").expect("kernel");
    let proto = cu.find("static void mapal_main();").expect("host proto");
    let def = cu.find("static void mapal_main() {").expect("host def");
    assert!(dev < kern, "device fns before kernels");
    assert!(kern < proto, "kernels before host prototypes");
    assert!(proto < def, "prototypes before definitions");
}

#[test]
fn vector_add_launch_count_and_trap_checks() {
    let ir = build_example("vector_add");
    let cu = crate::emit(&ir).unwrap();
    // zip + map + 2 index + fold = 5 launches (launch count is a
    // per-site property — unchanged by #17's dedup, #14's trim, and the
    // S20 proof refinement).
    assert_eq!(cu.matches("<<<").count(), 5, "{cu}");
    // §3's check-after-EVERY-launch, trimmed by #14 to the sites that
    // can trap — now zero: vector_add's only would-be trap sources are
    // its two Index sites, and both are constant readbacks (`c[0]`,
    // `c[15]`) the S20 bounds proof clears (0 < 16, 15 < 16) ⇒
    // trap-free, no readback follows any launch.
    assert_eq!(cu.matches("trap_check_after_launch();").count(), 0, "{cu}");
}

#[test]
fn host_device_fn_called_from_both_sites() {
    // sq is called from the map body (device) AND from main (host): one
    // definition serves both. sq is trap-FREE (#14: `x * x` has no
    // guard), so its two-site signature takes no trap pointer and every
    // call site passes nothing — the definition shape is otherwise the
    // BC8 (iv) single-definition rule.
    let src = "fn sq(x: i32) -> i32 {\n    x * x -> ret;\n}\n\
                   fn main() {\n    [1, 2] -> map { x -> x -> sq } -> rs;\n    \
                   3 -> sq -> s;\n    rs[0] -> println;\n    s -> println;\n}\n";
    let cu = emit_src(src);
    assert!(
        cu.contains("static __host__ __device__ int32_t fn0(int32_t in)"),
        "{cu}"
    );
    // The kernel calls the map body; the map body (itself a HostDevice
    // fn) calls sq — and so does main; nobody threads a trap pointer.
    assert!(cu.contains("out[i] = fn2(in[i]);"), "{cu}");
    // WP-C (R-ONENAME): the HostDevice body calls sq with the param
    // read in place — no extraction local.
    assert!(cu.contains("= fn0(in);"), "{cu}");
    let main_start = cu.find("static void mapal_main() {").unwrap();
    let host = &cu[main_start..];
    assert!(host.contains("fn0(3)"), "{host}");
    // Exactly one definition of sq (no twin).
    assert_eq!(cu.matches("fn0(int32_t in").count(), 2, "{cu}"); // proto + def
}

#[test]
fn emit_twice_byte_equal() {
    for name in ["vector_add", "zip_demo", "sepia"] {
        let a = crate::emit(&build_example(name)).unwrap();
        let b = crate::emit(&build_example(name)).unwrap();
        assert_eq!(a, b, "{name}: emit is not byte-deterministic");
    }
}
