//! Tests of the recorded cells: embedded arrays, loops in bodies, erased elements, local-array budget and captures.

use super::*;

// --- F3: the recorded Unsupported cell — arrays embedded in products ---
//
// Deep-value semantics break under the handle model (DESIGN §5): a
// product whose residual contains an array lowers to a struct with a
// `T*` field, and on device that field can only hold a per-thread
// local-memory or buffer-interior pointer — both dangle the moment the
// struct escapes into global memory. The honest move is the recorded
// cell, never a silent miscompile.

/// The F3 cell, asserted on `emit`.
fn assert_product_array_cell(src: &str) {
    let ir = lower_src(src);
    let e = crate::emit(&ir).unwrap_err();
    assert!(
        matches!(
            e,
            EmitError::Unsupported { ref feature, .. }
                if feature == "arrays embedded in products on device"
        ),
        "expected the F3 Unsupported cell, got {e:?}"
    );
}

#[test]
fn map_body_returning_product_with_array_is_unsupported() {
    // Repro 1: the map's output element type is a product containing an
    // array — the elementwise kernel would store per-thread local
    // pointers into the global output buffer.
    assert_product_array_cell(
        "fn main() {\n    [1, 2] -> map { x -> ([x, x], x) } -> rs;\n    \
             rs[0] -> p;\n    p.1 -> println;\n}\n",
    );
}

#[test]
fn map_body_local_product_with_array_is_unsupported() {
    // Repro 1b: the body only PASSES the product through a device-side
    // local (scalar return) — the twin's local struct still has no
    // honest storage for the array field.
    assert_product_array_cell(
        "fn main() {\n    [1, 2] -> map { x ->\n        \
             ([x, x], x) -> p;\n        p.1 -> ret\n    } -> rs;\n    \
             rs[0] -> println;\n}\n",
    );
}

#[test]
fn zip_over_nested_arrays_is_unsupported() {
    // Repro 2: zip producing ([i32;2], [i32;2]) elements — the kernel
    // would store buffer-INTERIOR pointers into global memory; a
    // consuming Index kernel would dereference them across launches
    // (after the pointee buffers' frees, or as garbage).
    assert_product_array_cell(
        "fn main() {\n    \
             ([[1, 2], [3, 4]], [[5, 6], [7, 8]]) -> zip -> z;\n    \
             z[0] -> p;\n    p.0 -> a;\n    a[0] -> println;\n}\n",
    );
}

#[test]
fn host_device_fn_product_with_array_return_is_unsupported() {
    // Repro 3: a body-reachable pure fn (no bulk ops, param passthrough)
    // returning a product-with-array — classified HostDevice, its single
    // definition would return a pointer-field struct BY VALUE on the
    // device pass.
    let src = "fn g(a: [i32; 2]) -> ([i32; 2], i32) {\n    (a, 5) -> ret;\n}\n\
                   fn main() {\n    \
                   [1, 2] -> map { x -> [x, x + 1] -> g -> p;\n        p.1 -> ret\n    } -> rs;\n    \
                   rs[0] -> println;\n}\n";
    assert_product_array_cell(src);
}

#[test]
fn product_array_cell_does_not_fire_on_supported_shapes() {
    // Host-side products holding handles are fine (the struct fields
    // are host copies of device pointers — F2's mk shape).
    let mk = "fn mk(x: i32) -> ([i32; 2], i32) {\n    [1, 2] -> a;\n    (a, x) -> ret;\n}\n\
                  fn main() {\n    5 -> mk -> p;\n    p.0 -> arr;\n    arr[1] -> println;\n}\n";
    assert!(crate::emit(&lower_src(mk)).is_ok(), "host-side product");
    // Arrays OF products (sepia's [Pixel; 16] — products of scalars).
    assert!(
        crate::emit(&build_example("sepia")).is_ok(),
        "array of scalar products"
    );
    // Products of products without arrays.
    let nested = "fn main() {\n    5 -> x;\n    ((x, x), true) -> p;\n    \
                      p.0 -> q;\n    q.0 -> println;\n}\n";
    assert!(
        crate::emit(&lower_src(nested)).is_ok(),
        "nested scalar products"
    );
    // The array-acc fold's (acc, e) pair: a product-with-array passed
    // BY VALUE to the body twin — registers only, never global memory.
    let fold = "fn main() {\n    \
                    ([0, 0], [[1, 2], [3, 4]]) -> fold { acc, row -> acc } -> r: [i32; 2];\n    \
                    r[1] -> println;\n}\n";
    assert!(
        crate::emit(&lower_src(fold)).is_ok(),
        "the by-value (acc, e) pair"
    );
    // Scalar products on device (the ordinary map/zip shapes).
    assert!(
        crate::emit(&build_example("vector_add")).is_ok(),
        "scalar products"
    );
}

// --- WP4: loops in bodies (inline-form quartets) -------------------------

#[test]
fn body_fn_scalar_loop_is_host_device_quartet() {
    // §1's inline-form cell: a canonical loop inside a map body lowers
    // as a top-level loop of the synthesized body fn. A pure-scalar
    // body is BC8 case (iv): ONE `__host__ __device__` definition —
    // the quartet is plain C++, valid on both passes.
    let src = "fn main() {\n    \
                       [1, 2, 3, 4] -> a: [i32; 4];\n    \
                       a -> map { x ->\n    \
                           mut i: i32 <- 0;\n    \
                           mut acc: i32 <- x;\n    \
                           loop {\n    \
                               (i < 2) -> {\n    \
                                   -true-> { acc + 10 -> acc; i + 1 -> i; -> loop; }\n    \
                                   -false-> acc -> r;\n    \
                               }\n    \
                           }\n    \
                           r -> ret\n    \
                       } -> rs;\n    \
                       rs[1] -> println;\n}\n";
    let cu = emit_src(src);
    // The body fn's single two-site definition carries the quartet.
    let def_start = cu
        .find("static __host__ __device__ int32_t fn")
        .expect("a HostDevice body fn:\n{cu}");
    let def_end = cu[def_start..].find("\n}\n").unwrap() + def_start;
    let def = &cu[def_start..def_end];
    assert_eq!(def.matches("while (true) {").count(), 1, "{def}");
    assert!(def.contains(") { break; }"), "{def}");
    // Guard-first: the break precedes the advance-cone add and the back
    // edge; the exit copy follows the loop close.
    let brk = def.find(") { break; }").unwrap();
    let add = def.find("(uint32_t)").expect("advance arith:\n{def}");
    assert!(brk < add, "guard-first: break before advance cone:\n{def}");
}

#[test]
fn twin_body_loop_is_per_thread_sequential_quartet() {
    // §1's inline-form cell, Twin case: the body contains a launch-form
    // op (the `[10, 20]` literal + Index), so BC8 makes it a twin; its
    // canonical loop is emitted as a per-thread sequential quartet. The
    // speculative `[10, 20][i]` read lives in the advance cone — on the
    // exit step (i = 2) it never executes (guard-first, ADR-0016).
    // (An ARRAY-carried loop exit in a body is rejected by the lower —
    // L1201 — before it can reach the backend; the scalar-carried shape
    // is the one the surface language produces today.)
    let src = "fn main() {\n    \
                       [1, 2] -> map { x ->\n    \
                           mut i: i32 <- 0;\n    \
                           mut acc: i32 <- x;\n    \
                           loop {\n    \
                               (i < 2) -> {\n    \
                                   -true-> { acc + [10, 20][i] -> acc; i + 1 -> i; -> loop; }\n    \
                                   -false-> acc -> r;\n    \
                               }\n    \
                           }\n    \
                           r -> ret\n    \
                       } -> rs;\n    \
                       rs[1] -> println;\n}\n";
    let cu = emit_src(src);
    let twin = twin_slice(&cu);
    assert_eq!(twin.matches("while (true) {").count(), 1, "{twin}");
    assert!(twin.contains(") { break; }"), "{twin}");
    // The loop-invariant literal is constructed once (outside the loop,
    // per-thread); the Index's int64 guard + read is inside the advance
    // cone — after the break (guard-first).
    let lit = twin
        .find("[2];")
        .expect("per-thread literal local:\n{twin}");
    let wh = twin.find("while (true) {").unwrap();
    assert!(lit < wh, "loop-invariant literal outside the loop:\n{twin}");
    let brk = twin.find(") { break; }").unwrap();
    let idx = twin
        .find("o9.f0 = o8;")
        .expect("the Index's operand pair:\n{twin}");
    assert!(
        brk < idx,
        "guard-first: the speculative Index is post-guard:\n{twin}"
    );
    // onto-device `cond-bounds`: on the advancing trips `i ∈ [0, 1]`
    // (`loop_plan`: init 0, step 1, continue `i < 2`), so the in-loop
    // Index is proven and carries no device bounds guard (index_oob ⇒ 2u).
    assert!(!twin.contains("*trap = 2u;"), "{twin}");
}

// --- F6: erased-element arrays degrade gracefully (never a panic) ------
//
// The surface cannot write a `()` literal (P0001), so the erased-element
// shapes are built with IrBuilder: a MapBody `i32 → Unit` (zero writers,
// I-RET) mapped over an i32 literal yields a `[Unit; n]` value, which
// has no runtime representation (L4).

fn arr_ty(elem: Ty, size: u64) -> Ty {
    Ty::Array {
        elem: Box::new(elem),
        size,
    }
}

/// `m1: i32 → Unit` — the Unit-array factory body (no morphisms).
fn declare_unit_body(b: &mut IrBuilder) -> FuncId {
    let m1 = b
        .declare(FuncKind::MapBody, "m1", Ty::i32(), Ty::Unit, L)
        .unwrap();
    {
        let fb = b.build_fn(m1).unwrap();
        fb.finish().unwrap();
    }
    m1
}

#[test]
fn fold_erased_acc_degrades_gracefully() {
    // F6 sites func.rs:844 + kernel.rs:686: a launch-form fold whose acc
    // is an erased-element array ([Unit;2]) — no acc0 kernel parameter,
    // no seed argument at the launch, no result buffer; the kernel still
    // launches and runs the body per element. (This body is a pure
    // passthrough — trap-free, so #14 drops the trap threading; a
    // trap-capable body would keep it.) The fold body's return is the
    // erased
    // array: with no device buffer to write, the out-param convention
    // is moot, so the body classifies HostDevice (void on both sites).
    let mut b = IrBuilder::new();
    let m1 = declare_unit_body(&mut b);
    let fb_id = b
        .declare(
            FuncKind::FoldBody,
            "fb",
            Ty::Tuple(vec![arr_ty(Ty::Unit, 2), Ty::i32()]),
            arr_ty(Ty::Unit, 2),
            L,
        )
        .unwrap();
    {
        let mut fb = b.build_fn(fb_id).unwrap();
        let i = fb.input();
        let acc = fb.proj(i, 0, Dest::Fresh(None), L).unwrap();
        fb.output(acc, None, L).unwrap();
        fb.finish().unwrap();
    }
    let main = b
        .declare(FuncKind::Named, "main", Ty::Unit, Ty::Unit, L)
        .unwrap();
    {
        let mut fb = b.build_fn(main).unwrap();
        let c1 = fb.constant(Value::I32(1), L).unwrap();
        let c2 = fb.constant(Value::I32(2), L).unwrap();
        let a = fb.pack_array(&[c1, c2], Dest::Fresh(None), L).unwrap();
        let u = fb.map(m1, a, Dest::Fresh(None), L).unwrap();
        let src = fb.pack(&[u, a], Dest::Fresh(None), L).unwrap();
        let _r = fb.fold(fb_id, src, Dest::Fresh(None), L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(main).unwrap();
    let cu = crate::emit(&ir).unwrap();
    // No acc0 anywhere (kernel param, host seed arg, or local).
    assert!(!cu.contains("acc0"), "{cu}");
    // The fold kernel still launches and runs the oracle loop, calling
    // the (void) body per element. The body is trap-free (#14): no trap
    // argument on the call and no per-step check.
    assert!(cu.contains("<<<1, 1>>>"), "{cu}");
    assert!(
        cu.contains("for (unsigned long long i = 0; i < 2ULL; i++) {"),
        "{cu}"
    );
    assert!(cu.contains("(arr[i]);"), "{cu}");
    assert!(!cu.contains("if (*trap)"), "{cu}");
}

/// The main wrapper shared by the inline-form tests: `[1,2] -> map mb ->
/// rs; rs[0] -> ret` (`mb` already built and finished).
fn seal_main_over(mut b: IrBuilder, mb: FuncId) -> CategoryIr {
    let main = b
        .declare(FuncKind::Named, "main", Ty::Unit, Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(main).unwrap();
        let c1 = fb.constant(Value::I32(1), L).unwrap();
        let c2 = fb.constant(Value::I32(2), L).unwrap();
        let a = fb.pack_array(&[c1, c2], Dest::Fresh(None), L).unwrap();
        let rs = fb.map(mb, a, Dest::Fresh(None), L).unwrap();
        let z = fb.constant(Value::I32(0), L).unwrap();
        let r = fb.index(rs, z, Dest::Fresh(None), L).unwrap();
        fb.output(r, None, L).unwrap();
        fb.finish().unwrap();
    }
    b.seal(main).unwrap()
}

#[test]
fn twin_map_erased_source_degrades_gracefully() {
    // F6 site kernel.rs:1289: the inline map's SOURCE is the erased
    // [Unit;2] — the twin must omit the body argument but still run the
    // per-thread loop (the launch form omits the parameter the same
    // way). m2: Unit → i32, a constant body.
    let mut b = IrBuilder::new();
    let m1 = declare_unit_body(&mut b);
    let m2 = b
        .declare(FuncKind::MapBody, "m2", Ty::Unit, Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(m2).unwrap();
        let c = fb.constant(Value::I32(5), L).unwrap();
        fb.output(c, None, L).unwrap();
        fb.finish().unwrap();
    }
    let mb = b
        .declare(FuncKind::MapBody, "mb", Ty::i32(), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(mb).unwrap();
        let x = fb.input();
        // A body-local literal: the bulk op that makes mb a Twin.
        let arr = fb.pack_array(&[x, x], Dest::Fresh(None), L).unwrap();
        let u = fb.map(m1, arr, Dest::Fresh(None), L).unwrap();
        let w = fb.map(m2, u, Dest::Fresh(None), L).unwrap();
        // Index by the element `x` itself ([1,2] statically ⊄ [0,2)) —
        // the S20 proof can't clear it, so the twin keeps its §3 guard
        // (a proven constant index would elide it).
        let r = fb.index(w, x, Dest::Fresh(None), L).unwrap();
        fb.output(r, None, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = seal_main_over(b, mb);
    let cu = crate::emit(&ir).unwrap();
    let twin = twin_slice(&cu);
    // The per-thread loop over the erased source runs (n = 2), storing
    // the body result. The constant body is trap-free (#14): its call
    // passes no trap argument — though the twin itself stays
    // trap-capable (its unproven Index guard) and keeps the parameter.
    assert!(
        twin.contains("for (unsigned long long") && twin.contains("< 2ULL;"),
        "{twin}"
    );
    assert!(twin.contains("= fn1();"), "{twin}");
    assert!(twin.contains("*trap = 2u;"), "{twin}");
}

#[test]
fn twin_enumerate_erased_source_degrades_gracefully() {
    // F6 site kernel.rs:1364: enumerate over the erased [Unit;2] — the
    // i32 index component (the thread index) still stores per element;
    // the erased element component stores nothing.
    let mut b = IrBuilder::new();
    let m1 = declare_unit_body(&mut b);
    let mb = b
        .declare(FuncKind::MapBody, "mb", Ty::i32(), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(mb).unwrap();
        let x = fb.input();
        // A body-local literal: the bulk op that makes mb a Twin.
        let arr = fb.pack_array(&[x, x], Dest::Fresh(None), L).unwrap();
        let u = fb.map(m1, arr, Dest::Fresh(None), L).unwrap();
        let e = fb.enumerate(u, Dest::Fresh(None), L).unwrap();
        let z = fb.constant(Value::I32(0), L).unwrap();
        let p = fb.index(e, z, Dest::Fresh(None), L).unwrap();
        let r = fb.proj(p, 0, Dest::Fresh(None), L).unwrap();
        fb.output(r, None, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = seal_main_over(b, mb);
    let cu = crate::emit(&ir).unwrap();
    let twin = twin_slice(&cu);
    assert!(
        twin.contains("for (unsigned long long") && twin.contains("< 2ULL;"),
        "{twin}"
    );
    // The index store survives: `{tgt}[{iv}] = (int32_t){iv};`.
    assert!(twin.contains("= (int32_t)"), "{twin}");
}

// --- F7: the per-thread local-array budget (documented cell) -----------

/// A map body constructing an `n`-element body-local literal — the twin
/// materializes it as a per-thread local C array (§2 item 8).
fn twin_local_array_src(n: usize) -> String {
    let elems = vec!["x"; n].join(", ");
    format!(
        "fn main() {{\n    [1, 2] -> map {{ x -> [{elems}] -> big;\n    big[0] -> ret\n    }} -> rs;\n    rs[0] -> println;\n}}\n"
    )
}

/// A launch-form fold with an `n`-element array acc — the single-thread
/// fold kernel copies it into a per-thread local (BC4).
fn fold_acc_array_src(n: usize) -> String {
    let elems = vec!["0"; n].join(", ");
    format!(
        "fn main() {{\n    ([{elems}], [[1, 2], [3, 4]]) -> fold {{ acc, row -> acc }} -> r;\n    r[0] -> println;\n}}\n"
    )
}

#[test]
fn twin_local_array_at_budget_emits_over_is_unsupported() {
    // 4096 i32 = 16384 B: exactly at the budget — emits.
    let cu = crate::emit(&lower_src(&twin_local_array_src(4096))).unwrap();
    assert!(cu.contains("int32_t o"), "{cu}");
    // 4097 i32 = 16388 B: over — the documented Unsupported cell.
    let e = crate::emit(&lower_src(&twin_local_array_src(4097))).unwrap_err();
    assert!(
        matches!(
            e,
            EmitError::Unsupported { ref feature, .. }
                if feature == "per-thread local array over 16384 bytes (16388 bytes)"
        ),
        "expected the F7 cell, got {e:?}"
    );
}

#[test]
fn fold_kernel_acc_at_budget_emits_over_is_unsupported() {
    // 4096 i32 = 16384 B: exactly at the budget — emits (the kernel
    // carries the acc in per-thread local storage).
    let cu = crate::emit(&lower_src(&fold_acc_array_src(4096))).unwrap();
    assert!(cu.contains("int32_t acc[4096];"), "{cu}");
    // 4097 i32 = 16388 B: over — the documented Unsupported cell.
    let e = crate::emit(&lower_src(&fold_acc_array_src(4097))).unwrap_err();
    assert!(
        matches!(
            e,
            EmitError::Unsupported { ref feature, .. }
                if feature == "per-thread local array over 16384 bytes (16388 bytes)"
        ),
        "expected the F7 cell, got {e:?}"
    );
}

// --- ADR-0027: captures — map/fold bodies read enclosing bindings ------

/// The one fn of `kind` in the module.
fn fn_of_kind(ir: &CategoryIr, kind: FuncKind) -> FuncId {
    ir.funcs()
        .find(|(_, fd)| fd.kind == kind)
        .map(|(id, _)| id)
        .expect("a fn of the kind")
}

#[test]
fn capturing_map_threads_capture_through_kernel_launch_and_body_call() {
    // ADR-0027 launch form: the captured scalar is an extra kernel
    // parameter (positionally after the existing operands, before the
    // trap pointer); the launch passes the source product's capture
    // component; the per-thread body call gets it as the leading
    // body-input field.
    let src = "fn main() {\n    3 -> scale;\n    [1, 2, 3] -> a;\n    \
                   a -> map { x -> x * scale } -> b;\n    b[1] -> println;\n}\n";
    let ir = lower_src(src);
    // The F3 cell must not trip: the captured source product is ordinary
    // dataflow (the transient-aggregate exemption covers it), and the
    // pure-scalar body classifies HostDevice exactly as before (BC8 iv).
    let quals = Qualifiers::analyze(&ir);
    assert_eq!(
        quals.get(fn_of_kind(&ir, FuncKind::MapBody)),
        FnQual::HostDevice
    );
    let cu = crate::emit(&ir).unwrap();
    // The kernel's extra parameter, after `in`. The body (`x * scale`)
    // is trap-free (#14): no trailing trap parameter.
    assert!(
        cu.contains("__global__ void k0_0(int32_t* out, int32_t* in, int32_t cap0) {"),
        "{cu}"
    );
    // The body call's capture argument: the leading field of the
    // assembled `(scale, x)` body input.
    assert!(cu.contains("pair.f0 = cap0;"), "{cu}");
    assert!(cu.contains("pair.f1 = in[i];"), "{cu}");
    assert!(cu.contains("out[i] = fn1(pair);"), "{cu}");
    // The launch's extra argument (the constant capture folds to `3`),
    // after the `in` operand — no `d_trap` (trap-free site, #14).
    assert!(
        cu.contains("k0_0<<<(unsigned int)((3ULL + 255ULL) / 256ULL), 256>>>(o4, o2, 3);"),
        "{cu}"
    );
}

#[test]
fn capturing_fold_threads_capture_through_kernel_launch_and_body_call() {
    // The Fold sibling: source `(base, acc, [T; n])` with the acc shifted
    // to component 1 — same parameter/launch/body-call threading.
    let src = "fn main() {\n    10 -> base;\n    [1, 2, 3] -> a;\n    \
                   (0, a) -> fold { acc, x -> acc + x * base } -> r;\n    r -> println;\n}\n";
    let ir = lower_src(src);
    let quals = Qualifiers::analyze(&ir);
    assert_eq!(
        quals.get(fn_of_kind(&ir, FuncKind::FoldBody)),
        FnQual::HostDevice
    );
    let cu = crate::emit(&ir).unwrap();
    // The body (`acc + x * base`) is trap-free (#14): no trap parameter.
    assert!(
        cu.contains(
            "__global__ void k0_0(int32_t* result, int32_t acc0, int32_t* arr, int32_t cap0) {"
        ),
        "{cu}"
    );
    // The `(base, acc, x)` assembly: the capture leads, then the acc
    // local, then the element — the ADR-0027 body-input shape.
    assert!(cu.contains("pair.f0 = cap0;"), "{cu}");
    assert!(cu.contains("pair.f1 = acc;"), "{cu}");
    assert!(cu.contains("pair.f2 = arr[i];"), "{cu}");
    assert!(cu.contains("acc = fn1(pair);"), "{cu}");
    // WP-C: the fold seed (a constant) inlines into the launch arg.
    assert!(cu.contains("k0_0<<<1, 1>>>(t1, 0, o5, 10);"), "{cu}");
}

#[test]
fn capturing_map_with_bulk_body_classifies_twin_and_inlines_captures() {
    // ADR-0027 Q3 (transitive captures) + the inline form: the outer map
    // body contains a capturing fold (a launch-form op) ⇒ the outer body
    // classifies Twin exactly as before (captures don't change BC8); the
    // inner fold's per-thread loop passes the through-captured scalar as
    // the leading body-input field; the inner fold's own body stays a
    // pure-scalar HostDevice. The F3 cell stays silent over the captured
    // wiring (the `emit` unwraps).
    let src = "fn main() {\n    10 -> scale;\n    [[1, 2], [3, 4]] -> m;\n    \
                   m -> map { row -> (0, row) -> fold { acc, x -> acc + x * scale } } -> rs;\n    \
                   rs[1] -> println;\n}\n";
    let ir = lower_src(src);
    let quals = Qualifiers::analyze(&ir);
    assert_eq!(quals.get(fn_of_kind(&ir, FuncKind::MapBody)), FnQual::Twin);
    assert_eq!(
        quals.get(fn_of_kind(&ir, FuncKind::FoldBody)),
        FnQual::HostDevice
    );
    let cu = crate::emit(&ir).unwrap();
    // The outer kernel: the capture is the extra parameter and the
    // leading body-input field; the element is the sub-array pointer.
    // Every body in the chain is trap-free (#14): no trap parameters,
    // no trap arguments, no per-step checks.
    assert!(
        cu.contains("__global__ void k0_0(int32_t* out, int32_t* in, int32_t cap0) {"),
        "{cu}"
    );
    assert!(cu.contains("pair.f0 = cap0;"), "{cu}");
    assert!(cu.contains("pair.f1 = (in + (i * 2ULL));"), "{cu}");
    assert!(cu.contains("out[i] = d_fn2(pair);"), "{cu}");
    assert!(
        cu.contains("k0_0<<<(unsigned int)((2ULL + 255ULL) / 256ULL), 256>>>(o6, o4, 10);"),
        "{cu}"
    );
    // The twin's per-thread fold loop: the through-captured `scale` is a
    // per-thread read of the twin's input field, leading the inner fold
    // body's `(scale, acc, x)` argument.
    let twin = twin_slice(&cu);
    assert!(
        twin.contains("for (unsigned long long t1 = 0; t1 < 2ULL; t1++) {"),
        "{twin}"
    );
    // WP-B (R-NODUP): the through-captured scalar is read through the
    // Named fold-operand product's field — one reference, no extraction
    // local, no re-computation; WP-D hoists the reference before the
    // loop.
    assert!(twin.contains("pair.f0 = o7.f0;"), "{twin}");
    assert!(twin.contains("pair.f1 = t0;"), "{twin}");
    assert!(twin.contains("pair.f2 = o6[t1];"), "{twin}");
    assert!(twin.contains("t0 = fn1(pair);"), "{twin}");
}
