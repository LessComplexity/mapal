//! Tests of the ABI measure, the qualifier matrix, dead host twins and the trap-capability pre-pass.

use super::*;

// --- abi_sizeof (the arena measure — plan-smart-arenas §3) --------------

fn tup(ts: Vec<Ty>) -> Ty {
    Ty::Tuple(ts)
}

fn arr(elem: Ty, size: u64) -> Ty {
    Ty::Array {
        elem: Box::new(elem),
        size,
    }
}

#[test]
fn abi_sizeof_scalars_at_width() {
    assert_eq!(abi_sizeof(&Ty::i32()), Some(4));
    assert_eq!(abi_sizeof(&Ty::i64()), Some(8));
    assert_eq!(abi_sizeof(&Ty::u8()), Some(1));
    assert_eq!(abi_sizeof(&Ty::f32()), Some(4));
    assert_eq!(abi_sizeof(&Ty::f64()), Some(8));
    assert_eq!(abi_sizeof(&Ty::Bool), Some(1));
    // Erased types have no representation to size.
    assert_eq!(abi_sizeof(&Ty::Unit), None);
    assert_eq!(abi_sizeof(&Ty::IoToken), None);
    assert_eq!(abi_sizeof(&Ty::Str), None);
}

#[test]
fn abi_sizeof_products_follow_c_layout() {
    // The drift case: (i32, bool) is 4 + 1 with tail padding to the
    // widest alignment (4) ⇒ 8 — nominal_sizeof says 5.
    let padded = tup(vec![Ty::i32(), Ty::Bool]);
    assert_eq!(nominal_sizeof(&padded), Some(5));
    assert_eq!(abi_sizeof(&padded), Some(8));
    // (f64, i32): 8 + 4, tail-padded to 8 ⇒ 16 (nominal 12).
    assert_eq!(abi_sizeof(&tup(vec![Ty::f64(), Ty::i32()])), Some(16));
    // (i32, f64): 4 + pad4 + 8 ⇒ 16 either way round.
    assert_eq!(abi_sizeof(&tup(vec![Ty::i32(), Ty::f64()])), Some(16));
    // (i32, i32): no padding anywhere ⇒ 8, nominal agrees.
    assert_eq!(abi_sizeof(&tup(vec![Ty::i32(), Ty::i32()])), Some(8));
    // sepia's Pixel (u8 × 3): alignment 1, no padding ⇒ 3.
    assert_eq!(
        abi_sizeof(&tup(vec![Ty::u8(), Ty::u8(), Ty::u8()])),
        Some(3)
    );
    // A handle field rides at pointer width: (i64, [f32; 4]) ⇒ 16.
    assert_eq!(
        abi_sizeof(&tup(vec![Ty::i64(), arr(Ty::f32(), 4)])),
        Some(16)
    );
    // Nested: ((i32, i32), bool) = 8 + 1, tail pad to 4 ⇒ 12.
    let nested = tup(vec![tup(vec![Ty::i32(), Ty::i32()]), Ty::Bool]);
    assert_eq!(abi_sizeof(&nested), Some(12));
}

#[test]
fn abi_sizeof_residual_one_is_the_bare_component() {
    // A residual-1 product lowers to its surviving component — no
    // wrapper, no tail padding (the ty.rs residual rule).
    assert_eq!(abi_sizeof(&tup(vec![Ty::IoToken, Ty::i32()])), Some(4));
    assert_eq!(abi_sizeof(&tup(vec![Ty::Bool, Ty::Unit])), Some(1));
    // Fully erased ⇒ no representation at all.
    assert_eq!(abi_sizeof(&tup(vec![Ty::Unit, Ty::IoToken])), None);
}

#[test]
fn buffer_bytes_of_is_flat_base_times_flat_count() {
    // The numeric twin of buffer_bytes' `sizeof(base) * flat` text.
    assert_eq!(buffer_bytes_of(&arr(Ty::f64(), 16)), Some(128));
    // Nested arrays peel to the flat base.
    assert_eq!(buffer_bytes_of(&arr(arr(Ty::i32(), 2), 3)), Some(24));
    // An array of a padded product: 8 B element × 16.
    let pixel_prod = arr(tup(vec![Ty::i32(), Ty::Bool]), 16);
    assert_eq!(buffer_bytes_of(&pixel_prod), Some(128));
    assert_eq!(
        buffer_bytes(&pixel_prod),
        "sizeof(FlowProd_int32_t_bool) * 16ULL"
    );
    // Erased base ⇒ no buffer.
    assert_eq!(buffer_bytes_of(&arr(Ty::Unit, 4)), None);
}

#[test]
fn align_up_rounds_to_powers_of_two() {
    assert_eq!(align_up(0, 256), 0);
    assert_eq!(align_up(1, 256), 256);
    assert_eq!(align_up(256, 256), 256);
    assert_eq!(align_up(257, 256), 512);
    assert_eq!(align_up(8, 8), 8);
    assert_eq!(align_up(9, 8), 16);
}

// --- BC8 qualifier matrix ----------------------------------------------

#[test]
fn qualifier_token_bearing_is_host_only() {
    let ir = lower_src("fn main() {\n    7 -> println;\n}\n");
    let quals = Qualifiers::analyze(&ir);
    assert_eq!(qual_of(&ir, &quals, "main"), FnQual::HostOnly);
}

#[test]
fn qualifier_pure_scalar_not_body_reachable_is_host_only() {
    // abs-like: token-free, no bulk ops, called only from main — plain
    // host fn, NO __host__ __device__ (keeps scalar programs WP2-stable).
    let src = "fn f(x: i32) -> i32 {\n    x + 1 -> ret;\n}\n\
                   fn main() {\n    5 -> f -> println;\n}\n";
    let ir = lower_src(src);
    let quals = Qualifiers::analyze(&ir);
    assert_eq!(qual_of(&ir, &quals, "f"), FnQual::HostOnly);
    let cu = crate::emit(&ir).unwrap();
    assert!(!cu.contains("__host__ __device__"), "{cu}");
    assert!(cu.contains("static int32_t fn0(int32_t in)"), "{cu}");
}

#[test]
fn qualifier_body_reachable_pure_scalar_is_host_device() {
    // sepia: clamp is token-free, no bulk ops, called from the map body.
    let ir = build_example("sepia");
    let quals = Qualifiers::analyze(&ir);
    assert_eq!(qual_of(&ir, &quals, "clamp"), FnQual::HostDevice);
    assert_eq!(qual_of(&ir, &quals, "main"), FnQual::HostOnly);
    // The map/fold bodies are pure-scalar: single definitions.
    assert_eq!(qual_of(&ir, &quals, "main::map@0"), FnQual::HostDevice);
    assert_eq!(qual_of(&ir, &quals, "main::fold@1"), FnQual::HostDevice);
}

#[test]
fn qualifier_body_reachable_bulk_is_twin() {
    // The map body constructs a literal (a launch-form op) ⇒ twin.
    let src = "fn main() {\n    [1, 2] -> map { x -> [x, x + 1][0] } -> rs;\n    \
                   rs[0] -> println;\n}\n";
    let ir = lower_src(src);
    let quals = Qualifiers::analyze(&ir);
    assert_eq!(qual_of(&ir, &quals, "main::map@0"), FnQual::Twin);
    // The twin exists; no __host__ __device__ definition for it.
    let cu = crate::emit(&ir).unwrap();
    assert!(cu.contains("static __device__ int32_t d_fn"), "{cu}");
}

#[test]
fn qualifier_array_return_body_is_twin_even_without_bulk_ops() {
    // The fold body returns the acc array by pass-through — no bulk op,
    // but an array return forces the twin (out-param convention).
    let src = "fn main() {\n    \
                   ([0, 0], [[1, 2], [3, 4]]) -> fold { acc, row -> acc } -> r;\n    \
                   r[0] -> println;\n}\n";
    let ir = lower_src(src);
    let quals = Qualifiers::analyze(&ir);
    assert_eq!(qual_of(&ir, &quals, "main::fold@0"), FnQual::Twin);
}

#[test]
fn qualifier_bulk_is_transitive_through_calls() {
    // h contains no bulk op but calls g, which folds ⇒ both twins.
    let src = "fn g(a: [i32; 2]) -> i32 {\n    (0, a) -> fold { acc, x -> acc + x } -> ret;\n}\n\
                   fn h(a: [i32; 2]) -> i32 {\n    a -> g -> ret;\n}\n\
                   fn main() {\n    [[1, 2], [3, 4]] -> map { row -> row -> h } -> rs;\n    \
                   rs[0] -> println;\n}\n";
    let ir = lower_src(src);
    let quals = Qualifiers::analyze(&ir);
    assert_eq!(qual_of(&ir, &quals, "g"), FnQual::Twin);
    assert_eq!(qual_of(&ir, &quals, "h"), FnQual::Twin);
    assert_eq!(qual_of(&ir, &quals, "main"), FnQual::HostOnly);
    // The inner fold body is pure-scalar: host+device single.
    assert_eq!(qual_of(&ir, &quals, "g::fold@0"), FnQual::HostDevice);
}

#[test]
fn qualifier_twin_propagates_callerward_through_calls() {
    // F4: g is a Twin (array return, no bulk ops); f is body-reachable,
    // pure-scalar, and CALLS g. The array-return Twin rule applied only
    // at decision time would classify f HostDevice — and its single
    // `__host__ __device__` definition would call g's HOST definition on
    // the device pass (nvcc hard error). The fixpoint must propagate
    // (bulk ∨ array-return) callerward through Call edges.
    //
    // The lens-B repro verbatim (classification only — its `(b, 5) -> p`
    // pair is a device-side product-with-array local, the F3 cell, so
    // the module no longer emits; the nvcc hard error is gone either
    // way):
    let repro = "fn g(a: [i32; 2]) -> [i32; 2] {\n    a -> ret;\n}\n\
                     fn f(a: [i32; 2]) -> i32 {\n    a -> g -> b;\n    (b, 5) -> p;\n    p.1 -> ret;\n}\n\
                     fn main() {\n    \
                     [1, 2] -> map { x -> [x, x + 1] -> f -> r;\n        r -> ret\n    } -> rs;\n    \
                     rs[0] -> println;\n}\n";
    let ir = lower_src(repro);
    let quals = Qualifiers::analyze(&ir);
    assert_eq!(qual_of(&ir, &quals, "g"), FnQual::Twin);
    assert_eq!(qual_of(&ir, &quals, "f"), FnQual::Twin);

    // The F3-clean variant (f consumes nothing from b): emits, and f's
    // device twin calls g's TWIN (d_g), never g's host definition.
    let src = "fn g(a: [i32; 2]) -> [i32; 2] {\n    a -> ret;\n}\n\
                   fn f(a: [i32; 2]) -> i32 {\n    a -> g -> b;\n    b -> g -> c;\n    7 -> ret;\n}\n\
                   fn main() {\n    \
                   [1, 2] -> map { x -> [x, x + 1] -> f -> r;\n        r -> ret\n    } -> rs;\n    \
                   rs[0] -> println;\n}\n";
    let ir = lower_src(src);
    let quals = Qualifiers::analyze(&ir);
    assert_eq!(qual_of(&ir, &quals, "g"), FnQual::Twin);
    assert_eq!(qual_of(&ir, &quals, "f"), FnQual::Twin);
    let fnames = fnames_of(&ir);
    let (mut d_f, mut d_g) = (String::new(), String::new());
    for (id, fd) in ir.funcs() {
        match fd.name.as_str() {
            "f" => d_f = format!("d_{}", fnames[id]),
            "g" => d_g = format!("d_{}", fnames[id]),
            _ => {}
        }
    }
    let cu = crate::emit(&ir).unwrap();
    let twin = twin_slice(&cu);
    // f's twin DEFINITION: the `{d_f}(` occurrence whose param list
    // closes into ` {` (prototypes and call sites end with `;`).
    let needle = format!("{d_f}(");
    let f_start = twin
        .match_indices(&needle)
        .find(|(i, _)| {
            let after = &twin[i + needle.len()..];
            after
                .find(')')
                .is_some_and(|p| after[p + 1..].trim_start().starts_with('{'))
        })
        .map(|(i, _)| i)
        .expect("f twin def");
    let f_end = twin[f_start..].find("\n}\n").unwrap() + f_start;
    let f_twin = &twin[f_start..f_end];
    assert!(
        f_twin.contains(&format!("{d_g}(")),
        "f's twin calls g's twin:\n{f_twin}"
    );
}

// --- #12: dead host-side twins -------------------------------------------

#[test]
fn dead_host_twin_skips_host_definition_and_kernels() {
    // The map body constructs a literal (bulk ⇒ Twin) and is called only
    // from the map kernel — no host path. Its host definition and its
    // Index site's kernel are dead text (#12): main's map + index sites
    // are the only kernels.
    let src = "fn main() {\n    [1, 2] -> map { x -> [x, x + 1][0] } -> rs;\n    \
                   rs[0] -> println;\n}\n";
    let ir = lower_src(src);
    let body = ir
        .funcs()
        .find(|(_, fd)| fd.kind == FuncKind::MapBody)
        .map(|(id, _)| id)
        .expect("a map body");
    let live = host_reachable(&ir);
    assert!(dead_host_twin(&Qualifiers::analyze(&ir), &live, body));
    let cu = crate::emit(&ir).unwrap();
    // The twin (device form) survives; the host definition is gone.
    assert!(cu.contains("static __device__ int32_t d_fn"), "{cu}");
    assert!(!cu.contains("static int32_t fn"), "{cu}");
    // main's map + index sites only (the body's host-side Index emits
    // no kernel — before #17/#12 it was the duplicate k1_0).
    assert_eq!(cu.matches("__global__ void").count(), 2, "{cu}");
    // Launch count unchanged: one map launch, one index launch.
    let main_start = cu.find("static void mapal_main() {").unwrap();
    let main_def = &cu[main_start..];
    assert_eq!(main_def.matches("<<<").count(), 2, "{main_def}");
}

#[test]
fn host_called_twin_keeps_host_definition_and_kernels() {
    // g is body-reachable (called from the map body) AND host-called
    // (from main) ⇒ Twin with a live host path: the host definition and
    // its fold-site kernel stay (#12's boundary).
    let src = "fn g(a: [i32; 2]) -> i32 {\n    (0, a) -> fold { acc, x -> acc + x } -> ret;\n}\n\
                   fn main() {\n    [1, 2] -> map { x -> [x, x + 1] -> g } -> rs;\n    \
                   [3, 4] -> g -> s;\n    rs[0] -> println;\n    s -> println;\n}\n";
    let ir = lower_src(src);
    let g = ir
        .funcs()
        .find(|(_, fd)| fd.name == "g")
        .map(|(id, _)| id)
        .expect("g");
    let quals = Qualifiers::analyze(&ir);
    let live = host_reachable(&ir);
    assert_eq!(quals.get(g), FnQual::Twin);
    assert!(!dead_host_twin(&quals, &live, g));
    let cu = crate::emit(&ir).unwrap();
    // Both definitions of g: the host fn and the device twin.
    assert!(cu.contains("static __device__ int32_t d_fn"), "{cu}");
    assert!(cu.contains("static int32_t fn"), "{cu}");
    // g's host definition launches its fold kernel (main calls g).
    let gname = cu
        .match_indices("static int32_t fn")
        .next()
        .map(|(i, _)| i)
        .expect("g host def");
    let g_def = &cu[gname..];
    assert!(g_def.contains("<<<1, 1>>>"), "{g_def}");
}

// --- #14: trap-capability pre-pass ---------------------------------------

#[test]
fn trap_caps_int_div_capable_float_div_not() {
    // An integer Div is a §3 zero guard ⇒ capable — except a literal
    // non-zero, non-−1 constant divisor, which cannot trap (the S20 #13
    // credit); a float Div never guards (ADR-0013's S13: ÷0 is ±inf/NaN)
    // ⇒ trap-free.
    let body_caps = |src: &str| {
        let ir = lower_src(src);
        let caps = TrapCaps::analyze(&ir);
        ir.funcs()
            .find(|(_, fd)| fd.kind == FuncKind::MapBody)
            .map(|(id, _)| caps.get(id))
            .expect("a map body")
    };
    assert!(body_caps(
        "fn main() {\n    [1, 2] -> map { x -> x / 2 -> y; y -> z; z / y } -> rs;\n    rs[0] -> println;\n}\n"
    ));
    assert!(!body_caps(
        "fn main() {\n    [1, 2] -> map { x -> x / 2 } -> rs;\n    rs[0] -> println;\n}\n"
    ));
    assert!(!body_caps(
        "fn main() {\n    [1.0, 2.0] -> map { x -> x / 2.0 } -> rs;\n    rs[0] -> println;\n}\n"
    ));
}

#[test]
fn widen_is_trap_free_device_scalar_with_no_kernel_site() {
    let src = "fn main() {\n    2 -> iota -> a;\n    a -> map { x -> x -> widen<f64> } -> b;\n    b[1] -> println;\n}\n";
    let ir = lower_src(src);
    let body = ir
        .funcs()
        .find(|(_, fd)| fd.kind == FuncKind::MapBody)
        .map(|(id, _)| id)
        .expect("map body");
    assert!(!TrapCaps::analyze(&ir).get(body));
    let cu = crate::emit(&ir).unwrap();
    assert!(cu.contains("__host__ __device__ double"), "{cu}");
    assert!(cu.contains("(double)("), "{cu}");
    // Iota + Map + Index only: Widen is scalar and owns no kernel site.
    assert_eq!(cu.matches("__global__ void").count(), 3, "{cu}");
}

#[test]
fn trap_caps_flows_callerward_through_calls() {
    // g has the unproven Index (capable — `a[u]`'s index is a load out
    // of a parameter array: statically unknown, so the S20 proof can't
    // clear it); the map body only CALLS g ⇒ capable transitively; the
    // map site rides the body (TrapCaps::site), so the kernel keeps the
    // full convention — while the float-div sibling kernel trims it.
    let src = "fn g(a: [i32; 2]) -> i32 {\n    a[0] -> u;\n    a[u] -> ret;\n}\n\
                   fn main() {\n    [1, 2] -> map { x -> [x, x] -> g } -> rs;\n    \
                   rs[4 % 3] -> println;\n}\n";
    let ir = lower_src(src);
    let caps = TrapCaps::analyze(&ir);
    let g = ir
        .funcs()
        .find(|(_, fd)| fd.name == "g")
        .map(|(id, _)| id)
        .expect("g");
    let body = ir
        .funcs()
        .find(|(_, fd)| fd.kind == FuncKind::MapBody)
        .map(|(id, _)| id)
        .expect("a map body");
    assert!(caps.get(g));
    assert!(caps.get(body)); // callerward through the Call edge
    // The site rule: the map site rides its capable body.
    let site_m = collect_sites(&ir, body).first().map(|s| s.m);
    let main = ir.entry();
    let main_sites = collect_sites(&ir, main);
    assert!(site_m.is_none()); // the body's sites are inline-form
    assert!(
        caps.site(&ir, main_sites[0].m),
        "the map kernel rides the capable body"
    );
    // The Index site itself: `4 % 3` is statically [0,2] ⊄ [0,2) — the
    // S20 proof can't clear it, so the site stays capable too.
    assert!(caps.site(&ir, main_sites[1].m));
    let cu = crate::emit(&ir).unwrap();
    // Both launches keep the §3 check (map body + index site capable).
    assert_eq!(cu.matches("trap_check_after_launch();").count(), 2, "{cu}");
}

#[test]
fn trap_free_float_body_trims_kernel_param_arg_and_check() {
    // The whole trim on one module (#14): a trap-free float-div body —
    // the kernel drops the trap parameter, the launch drops `d_trap`,
    // and no readback follows. The unproven Index sibling keeps
    // everything (`4 % 3` is statically [0,2] ⊄ [0,2): the S20 proof
    // can't clear it — a proven constant readback would trim too).
    let src = "fn main() {\n    [1.0, 2.0] -> map { x -> x / 2.0 } -> rs;\n    \
                   rs[4 % 3] -> println;\n}\n";
    let cu = emit_src(src);
    let kstart = cu.find("__global__ void k0_0(").expect("map kernel");
    let kend = cu[kstart..].find("\n}\n").unwrap() + kstart;
    let kern = &cu[kstart..kend];
    assert!(!kern.contains("trap"), "trap-free kernel text:\n{kern}");
    // One launch (the Index) keeps d_trap + the readback.
    assert_eq!(cu.matches(", d_trap);").count(), 1, "{cu}");
    assert_eq!(cu.matches("trap_check_after_launch();").count(), 1, "{cu}");
    // Launch count is a per-site property: unchanged.
    assert_eq!(cu.matches("<<<").count(), 2, "{cu}");
}

#[test]
fn proven_index_trims_guard_param_arg_and_check_unproven_keeps() {
    // The S20 refinement end to end: a `bounds_proof`-proven Index site
    // (the constant readback `a[0]`) is trap-free — its kernel drops the
    // §3 guard AND the trap parameter, its launch drops `d_trap` and the
    // readback — while the unproven sibling (`a[17 % 5]`, statically
    // [0,4] ⊄ [0,4)) keeps the full convention verbatim. The two
    // otherwise-identical Index sites now emit TWO kernels (#17's shape
    // key carries the guard/parameter text — proven and unproven are
    // different shapes); launch count is unchanged.
    let src = "fn main() {\n    [1, 2, 3, 4] -> a: [i32; 4];\n    \
                   a[0] -> x;\n    a[17 % 5] -> y;\n    x + y -> println;\n}\n";
    let ir = lower_src(src);
    let caps = TrapCaps::analyze(&ir);
    let main = ir.entry();
    let sites = collect_sites(&ir, main);
    assert!(!caps.site(&ir, sites[0].m), "a[0]: proven ⇒ trap-free");
    assert!(caps.site(&ir, sites[1].m), "a[17 % 5]: unproven ⇒ capable");
    assert!(caps.get(main)); // the unproven site holds main capable
    let cu = crate::emit(&ir).unwrap();
    // The trap-free Index kernel: no guard text, no trap parameter.
    let free = "__global__ void k0_0(int32_t* result, int32_t* arr, int64_t idx) {";
    assert!(cu.contains(free), "{cu}");
    let fstart = cu.find(free).unwrap();
    let fend = cu[fstart..].find("\n}\n").unwrap() + fstart;
    let fkern = &cu[fstart..fend];
    assert!(!fkern.contains("trap"), "{fkern}");
    assert!(
        fkern.contains("*result = arr[(unsigned long long)idx];"),
        "{fkern}"
    );
    // The capable sibling keeps guard + parameter, verbatim.
    assert!(
        cu.contains(
            "__global__ void k0_1(int32_t* result, int32_t* arr, int64_t idx, unsigned int* trap) {"
        ),
        "{cu}"
    );
    assert!(
        cu.contains("if (idx < 0 || idx >= (int64_t)4) { *trap = 2u; return; }"),
        "{cu}"
    );
    // Launches: two, unchanged — one drops `d_trap`, one keeps it.
    assert_eq!(cu.matches("<<<").count(), 2, "{cu}");
    assert_eq!(cu.matches(", d_trap);").count(), 1, "{cu}");
    assert_eq!(cu.matches("trap_check_after_launch();").count(), 1, "{cu}");
    assert_eq!(cu.matches("__global__ void").count(), 2, "{cu}");
}
