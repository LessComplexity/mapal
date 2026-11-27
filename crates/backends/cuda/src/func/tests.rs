//! Unit tests of the host function emitter.

use super::*;
use mapal_ir::{Dest, FuncKind, IrBuilder, SourceLoc};

const L: SourceLoc = SourceLoc { start: 0, end: 0 };

fn lower_src(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"))
}

fn emit_src(src: &str) -> String {
    crate::emit(&lower_src(src)).unwrap()
}

#[test]
fn int_arith_wraps_via_unsigned_casts() {
    let cu = emit_src(
        "fn main() {\n    3 + 4 -> a;\n    10 - 3 -> s;\n    2 * 6 -> m;\n    \
             a + s + m -> t;\n    t -> println;\n}\n",
    );
    // BC2: every int op is `({cty})(({ucty})a {sym} ({ucty})b)`. WP-C:
    // the operand wrappers dissolved — constants inline as literals
    // inside the casts (the shape pin is unchanged).
    assert!(cu.contains("(int32_t)((uint32_t)"), "{cu}");
    assert!(cu.contains(" + (uint32_t)"), "{cu}");
    assert!(cu.contains(" - (uint32_t)"), "{cu}");
    assert!(cu.contains(" * (uint32_t)"), "{cu}");
    // Constants render as decimal literals, inline in the expressions.
    assert!(cu.contains("(uint32_t)3"), "{cu}");
    assert!(cu.contains("(uint32_t)10"), "{cu}");
}

#[test]
fn neg_wraps_via_unsigned_cast() {
    let cu = emit_src("fn main() {\n    5 -> x;\n    -x -> y;\n    y -> println;\n}\n");
    assert!(cu.contains("(int32_t)(0 - (uint32_t)"), "{cu}");
}

#[test]
fn div_mod_guards_host_shape() {
    let src = "fn f(a: i32, b: i32) -> i32 {\n    a / b -> q;\n    a % b -> r;\n    \
                   q + r -> ret;\n}\nfn main() {\n    (7, 3) -> f -> v;\n    v -> println;\n}\n";
    let cu = emit_src(src);
    // Zero guard → direct host mapal_trap(0) (DESIGN §3).
    assert_eq!(cu.matches("== 0) { mapal_trap(0); }").count(), 2, "{cu}");
    // MIN/-1 value guards: Div ⇒ MIN, Mod ⇒ 0.
    assert_eq!(cu.matches("== -1) && (").count(), 2, "{cu}");
    // The overflow arms: Div stores INT32_MIN, Mod stores 0 (both inside
    // the `} else {` frame; the prelude's `unsigned int kind = 0;` is not
    // an overflow store).
    assert!(cu.contains("= INT32_MIN;\n  } else {"), "{cu}");
    assert!(cu.contains("= 0;\n  } else {"), "{cu}");
    // The division/modulo proper, guarded.
    assert!(cu.contains(" / "), "{cu}");
    assert!(cu.contains(" % "), "{cu}");
}

#[test]
fn unsigned_div_has_only_the_zero_guard() {
    let src = "fn f(a: u8, b: u8) -> u8 {\n    a / b -> ret;\n}\n\
                   fn main() {\n    200 -> x: u8;\n    3 -> y: u8;\n    \
                   (x, y) -> f -> v;\n    v -> println;\n}\n";
    let cu = emit_src(src);
    assert!(cu.contains("== 0) { mapal_trap(0); }"), "{cu}");
    // Unsigned: no MIN/-1 guard anywhere.
    assert!(!cu.contains("== -1"), "{cu}");
}

// --- #13: constant-divisor guard elision --------------------------------

#[test]
fn const_divisor_elides_both_guards_host() {
    // Literal non-zero divisor (≠ −1): the zero guard and the MIN/−1
    // value guard are dead by construction — a plain division remains.
    let src = "fn f(a: i32) -> i32 {\n    a / 4 -> q;\n    a % 4 -> r;\n    \
                   q + r -> ret;\n}\nfn main() {\n    7 -> f -> v;\n    v -> println;\n}\n";
    let cu = emit_src(src);
    assert!(!cu.contains("mapal_trap(0)"), "{cu}");
    assert!(!cu.contains("== -1"), "{cu}");
    assert!(cu.contains(" / "), "{cu}");
    assert!(cu.contains(" % "), "{cu}");
}

#[test]
fn const_zero_divisor_keeps_the_zero_guard() {
    // A literal 0 divisor: the guard ALWAYS fires (the oracle traps the
    // same way) — eliding is for provably-dead guards only.
    let src = "fn f(a: i32) -> i32 {\n    a / 0 -> ret;\n}\n\
                   fn main() {\n    7 -> f -> v;\n    v -> println;\n}\n";
    let cu = emit_src(src);
    assert!(cu.contains("== 0) { mapal_trap(0); }"), "{cu}");
}

#[test]
fn const_neg_one_divisor_keeps_only_the_min_guard() {
    // A literal −1 divisor: the zero guard is dead (elided); the MIN/−1
    // value guard is LIVE (it decides on the dividend) and stays.
    let src = "fn f(a: i32) -> i32 {\n    a / -1 -> ret;\n}\n\
                   fn main() {\n    7 -> f -> v;\n    v -> println;\n}\n";
    let cu = emit_src(src);
    assert!(!cu.contains("mapal_trap(0)"), "{cu}");
    assert!(cu.contains("== -1) && ("), "{cu}");
    assert!(cu.contains("= INT32_MIN;"), "{cu}");
}

#[test]
fn const_divisor_elides_guards_in_twin_and_host_device() {
    // Device paths (#13 applies to the twin AND the `__host__ __device__`
    // definition): a constant-divisor Div drops the device zero guard —
    // and with it the `#ifdef __CUDA_ARCH__` split.
    let src = "fn main() {\n    [10, 20] -> map { x -> x / 4 } -> rs;\n    \
                   rs[0] -> println;\n}\n";
    let cu = emit_src(src);
    assert!(cu.contains("static __host__ __device__"), "{cu}");
    assert!(!cu.contains("#ifdef __CUDA_ARCH__"), "{cu}");
    assert!(!cu.contains("*d_trap = 1u"), "{cu}");
    // The twin sibling (a body with a literal AND a constant-divisor
    // Div): no `*trap = 1u` in device code.
    let src = "fn main() {\n    [10, 20] -> map { x -> [x / 4, x][0] } -> rs;\n    \
                   rs[0] -> println;\n}\n";
    let cu = emit_src(src);
    assert!(!cu.contains("*trap = 1u"), "{cu}");
    assert!(cu.contains(" / "), "{cu}");
}

#[test]
fn float_div_mod_never_had_guards() {
    // #13 touches nothing on the float path (÷0 is ±inf/NaN by
    // ADR-0013's S13 amendment — never a guard).
    let cu = emit_src(
        "fn main() {\n    7.5 / 2.0 -> d;\n    7.5 % 2.0 -> m;\n    \
                           d + m -> t;\n    t -> println;\n}\n",
    );
    // No guard CALLS (the prelude's mapal_trap declaration is always
    // present).
    assert!(!cu.contains("mapal_trap(0);"), "{cu}");
    assert!(!cu.contains("== 0) {"), "{cu}");
    assert!(cu.contains(" / "), "{cu}");
    assert!(cu.contains("fmod("), "{cu}");
}

#[test]
fn float_arith_is_plain_and_mod_is_fmod() {
    let cu = emit_src("fn main() {\n    7.5 % 2.0 -> m;\n    m -> println;\n}\n");
    // f64 Mod → fmod; WP-C: operands and result inline (the wrapper
    // product dissolved, the single-consumer chain nests into print).
    assert!(cu.contains("fmod(7.5e0, 2e0)"), "{cu}");
    assert!(cu.contains("mapal_print_f64("), "{cu}");
    let src = "fn main() {\n    7.5 -> x: f32;\n    2.0 -> y: f32;\n    \
                   x % y -> m;\n    m -> println;\n}\n";
    let cu = emit_src(src);
    assert!(cu.contains("fmodf("), "{cu}");
    assert!(cu.contains("mapal_print_f32("), "{cu}");
}

#[test]
fn float_neg_of_negative_constant_is_parenthesized() {
    // The lower folds `-1.5` into a Constant on the raw path; Neg must
    // emit `-({val})` — `--1.5e0` is ill-formed C++ (the decrement
    // operator).
    let cu = emit_src("fn main() {\n    -1.5 -> x;\n    -x -> y;\n    y -> println;\n}\n");
    // WP-C: the Neg may inline — pin the parenthesized form wherever
    // it lands; `--` never appears in an expression position (comment
    // banners legitimately contain `---`).
    assert!(!cu.contains("= --"), "{cu}");
    assert!(!cu.contains("(--"), "{cu}");
    assert!(cu.contains("-(-1.5e0)"), "{cu}");
}

#[test]
fn phi_is_strict_select_over_temporaries() {
    let src = "fn abs(x: i32) -> i32 {\n    (x > 0) -> {\n        -true->  x;\n        \
                   -false-> x * -1;\n    } -> ret;\n}\n\
                   fn main() {\n    -7 -> abs -> r;\n    r -> println;\n}\n";
    let cu = emit_src(src);
    // Both arms materialize into temporaries before the ternary (BC7).
    assert!(cu.contains("int32_t t0 = "), "{cu}");
    assert!(cu.contains("int32_t t1 = "), "{cu}");
    assert!(cu.contains("? t0 : t1;"), "{cu}");
}

#[test]
fn and_or_are_bitwise_never_short_circuit() {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", Ty::Bool, Ty::Bool, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let i = fb.input();
        let and = fb
            .binop(Operation::And, i, i, Dest::Fresh(None), L)
            .unwrap();
        fb.binop(Operation::Or, and, i, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let cu = crate::emit(&ir).unwrap();
    assert!(cu.contains(" & "), "{cu}");
    assert!(cu.contains(" | "), "{cu}");
    assert!(!cu.contains("&&"), "{cu}");
    assert!(!cu.contains("||"), "{cu}");
}

#[test]
fn residual_one_pair_is_bare_no_struct() {
    let cu = emit_src("fn main() {\n    7 -> println;\n}\n");
    // The (IoToken, i32) print pair is residual-1: a bare int32_t local —
    // no FlowProd type anywhere in the module.
    assert!(!cu.contains("FlowProd"), "{cu}");
    assert!(cu.contains("mapal_print_i32(o"), "{cu}");
}

#[test]
fn residual_two_pair_uses_named_struct_fields() {
    let src = "fn f(a: i32, b: i32) -> i32 {\n    a + b -> ret;\n}\n\
                   fn main() {\n    (1, 2) -> f -> v;\n    v -> println;\n}\n";
    let cu = emit_src(src);
    assert!(
        cu.contains("struct FlowProd_int32_t_int32_t {\n    int32_t f0;\n    int32_t f1;\n};"),
        "{cu}"
    );
    // Pair writes and arith reads go through the .f{erased} fields.
    assert!(cu.contains(".f0 = "), "{cu}");
    assert!(cu.contains(".f1 = "), "{cu}");
    assert!(cu.contains(".f0 + (uint32_t)"), "{cu}");
}

#[test]
fn u8_print_routes_to_flow_print_u8() {
    let cu = emit_src("fn main() {\n    200 -> x: u8;\n    x -> println;\n}\n");
    // A call (not the prelude's decl): mapal_print_u8 on a local.
    assert!(cu.contains("mapal_print_u8(o"), "{cu}");
    // No i32 print CALL (the prelude declares every print fn).
    assert!(!cu.contains("mapal_print_i32(o"), "{cu}");
}

#[test]
fn str_print_passes_pointer_and_explicit_len() {
    let cu = emit_src("fn main() {\n    \"hi\" -> print;\n}\n");
    assert!(cu.contains("static const char str0[] = \"hi\";"), "{cu}");
    assert!(
        cu.contains("mapal_print_str((const uint8_t*)str0, 2, false);"),
        "{cu}"
    );
}

#[test]
fn literals_render_at_width() {
    assert_eq!(const_literal(&Value::I32(7)), "7");
    assert_eq!(const_literal(&Value::I32(-7)), "-7");
    assert_eq!(const_literal(&Value::I32(i32::MIN)), "INT32_MIN");
    assert_eq!(const_literal(&Value::I64(i64::MIN)), "INT64_MIN");
    assert_eq!(const_literal(&Value::I64(-9)), "-9");
    assert_eq!(const_literal(&Value::U8(255)), "255");
    assert_eq!(const_literal(&Value::Bool(true)), "true");
    assert_eq!(const_literal(&Value::F64(f64::NAN)), "NAN");
    assert_eq!(const_literal(&Value::F64(f64::INFINITY)), "HUGE_VAL");
    assert_eq!(const_literal(&Value::F64(f64::NEG_INFINITY)), "-HUGE_VAL");
    assert_eq!(const_literal(&Value::F32(f32::INFINITY)), "HUGE_VALF");
}

#[test]
fn float_literals_round_trip() {
    // Every finite literal parses back bit-exactly (Rust's parser is
    // correctly rounded, like the target C++ compilers).
    for x in [
        4080.0f64,
        5.375,
        -0.0,
        0.1,
        1e300,
        1e-300,
        f64::MAX,
        f64::MIN_POSITIVE,
    ] {
        let lit = const_literal(&Value::F64(x));
        let back: f64 = lit.parse().unwrap();
        assert_eq!(back.to_bits(), x.to_bits(), "f64 literal {lit} for {x}");
    }
    for x in [4080.0f32, 5.375, -0.0, 0.1, 1e30, f32::MAX] {
        let lit = const_literal(&Value::F32(x));
        let back: f32 = lit.trim_end_matches('f').parse().unwrap();
        assert_eq!(back.to_bits(), x.to_bits(), "f32 literal {lit} for {x}");
    }
}

/// WP2's placeholder is gone: vector_add now emits a complete module —
/// kernels, launches, literal uploads, the whole WP3 surface.
#[test]
fn array_bulk_ops_emit_wp3() {
    let path = format!(
        "{}/../../../examples/vector_add.mapal",
        env!("CARGO_MANIFEST_DIR")
    );
    let src = std::fs::read_to_string(&path).unwrap();
    let cu = crate::emit(&lower_src(&src)).unwrap();
    assert!(cu.contains("__global__"), "{cu}");
    // S20: vector_add's only would-be trap sources are its two constant
    // readbacks — the bounds proof clears both, so no launch is followed
    // by the §3 check at all.
    assert!(!cu.contains("trap_check_after_launch();"), "{cu}");
    assert!(cu.contains("cudaMemcpyHostToDevice"), "{cu}");
}

#[test]
fn allocation_registry_skeleton() {
    // A trivial sealed main to host an FnEmit.
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", Ty::Unit, Ty::Unit, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let i = fb.input();
        fb.output(i, None, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let mut fnames: SecondaryMap<FuncId, String> = SecondaryMap::new();
    fnames.insert(ir.entry(), "mapal_main".to_string());
    let strings: SecondaryMap<ObjectId, StrGlobal> = SecondaryMap::new();
    let quals = crate::kernel::Qualifiers::analyze(&ir);
    let caps = crate::kernel::TrapCaps::analyze(&ir);
    let live = crate::kernel::host_reachable(&ir);
    let kernels = crate::kernel::emit_kernel_set(&ir, &fnames, &quals, &caps, &live);

    let mut fe = FnEmit::new(&ir, ir.entry(), &fnames, &strings, &quals, &caps, &kernels);
    fe.register_alloc("buf0".into());
    fe.register_alloc("buf1".into());
    fe.register_alloc("buf2".into());
    // The escape move removes exactly buf1; frees emit in allocation order.
    fe.remove_alloc("buf1");
    fe.remove_alloc("never_registered"); // no-op, no panic
    fe.emit_frees();
    assert_eq!(
        fe.body,
        "  cu_check(cudaFree(buf0), \"cudaFree(buf0)\");\n  \
             cu_check(cudaFree(buf2), \"cudaFree(buf2)\");\n"
    );
    // The registry is drained by emission.
    assert!(fe.allocs.is_empty());
}

#[test]
fn empty_registry_emits_no_frees() {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", Ty::Unit, Ty::Unit, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let i = fb.input();
        fb.output(i, None, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let cu = crate::emit(&ir).unwrap();
    // WP2's scalar programs register nothing: no cudaFree in any fn body.
    assert!(!cu.contains("cudaFree(buf"), "{cu}");
    // The one cudaFree LINE is the trap-flag free in main.
    let free_lines: Vec<&str> = cu.lines().filter(|l| l.contains("cudaFree")).collect();
    assert_eq!(free_lines.len(), 1, "{cu}");
    assert!(free_lines[0].contains("d_trap"), "{cu}");
}

// --- F2: the return-escape guard covers every array-typed component ----

/// The definition slice of the host fn whose signature contains `needle`.
fn fn_def<'a>(cu: &'a str, needle: &str) -> &'a str {
    let start = cu.find(needle).expect("fn def");
    let end = cu[start..].find("\n}\n").map(|e| start + e).unwrap();
    &cu[start..end]
}

/// Frees emitted at fn-epilogue indent (unguarded — the value guard
/// indents its free one level deeper). A bare free of a buffer that
/// escapes through the return is the use-after-free (DESIGN §2).
fn bare_free_lines(def: &str) -> Vec<&str> {
    def.lines()
        .filter(|l| l.starts_with("  cu_check(cudaFree("))
        .collect()
}

#[test]
fn product_return_with_array_component_guards_free() {
    // F2 repro (a): mk returns a product holding the locally-allocated
    // array `a`. The name-based registry escape (Output) cannot see the
    // buffer inside the struct; the free must be value-guarded against
    // the returned struct's array field, or the caller gets a dangling
    // field (use-after-free). #18: the literal buffer is a zone member,
    // so the guard is the zone release's range-test veto (plan rule 3 —
    // the same pointer-VALUE comparison class, at zone granularity).
    let src = "fn mk(x: i32) -> ([i32; 2], i32) {\n    [1, 2] -> a;\n    (a, x) -> ret;\n}\n\
                   fn main() {\n    5 -> mk -> p;\n    p.0 -> arr;\n    arr[1] -> println;\n}\n";
    let cu = emit_src(src);
    let def = fn_def(&cu, "fn0(int32_t in) {");
    // The zone release is vetoed iff the return struct's array field
    // (.f0 by the erased-index remap) points into the arena.
    assert!(
        def.contains("(char*)o1.f0 >= (char*)arena0 && (char*)o1.f0 < (char*)arena0 + 256ULL"),
        "range-test veto against the return field:\n{def}"
    );
    assert!(def.contains("if (!escaped0) {"), "{def}");
    assert_eq!(
        bare_free_lines(def),
        Vec::<&str>::new(),
        "no buffer may be freed bare in an array-component return:\n{def}"
    );
    // The returned struct field still holds the handle past fn exit.
    assert!(def.contains("return o"), "{def}");
}

#[test]
fn phi_selected_array_return_guards_free() {
    // F2 repro (b): a Phi over two locally-allocated buffers feeds the
    // return. The Output's name-based removal matches nothing in the
    // registry (the Phi allocated nothing); the selected buffer escapes
    // and must survive, the unselected one is freed — by pointer
    // equality against the return local. #18: both literals share the
    // fn zone, so the veto is one range test on the returned handle —
    // an escaping buffer pins its whole zone (rule 3's recorded
    // bounded-leak tradeoff for the unselected buffer).
    let src = "fn sel(c: bool) -> [i32; 2] {\n    \
                   c -> {\n        -true-> [1, 2];\n        -false-> [3, 4];\n    } -> ret;\n}\n\
                   fn main() {\n    true -> sel -> r;\n    r[0] -> println;\n}\n";
    let cu = emit_src(src);
    let def = fn_def(&cu, "fn0(bool in) {");
    // Both literal buffers are zone members (two pointer inits); the
    // single zone release is vetoed by the returned handle's range test.
    assert_eq!(def.matches("= (int32_t*)(arena0 + ").count(), 2, "{def}");
    assert_eq!(def.matches("cu_check(cudaFree(").count(), 1, "{def}");
    assert!(
        def.contains("(char*)o1 >= (char*)arena0 && (char*)o1 < (char*)arena0 + 512ULL"),
        "range-test veto on the returned handle:\n{def}"
    );
    assert_eq!(
        bare_free_lines(def),
        Vec::<&str>::new(),
        "no bare frees — the escaped buffer is runtime-selected:\n{def}"
    );
}

#[test]
fn nested_product_return_guards_free_recursively() {
    // F2, nested shape: an array two products deep — the guard must
    // recurse to the inner field lvalue (.f0.f0 by the remap). #18: the
    // recursion now shows up inside the zone release's range test.
    let src = "fn mk(x: i32) -> (([i32; 2], i32), bool) {\n    \
                   [1, 2] -> a;\n    ((a, x), true) -> ret;\n}\n\
                   fn main() {\n    5 -> mk -> p;\n    p.0 -> q;\n    q.0 -> arr;\n    \
                   arr[1] -> println;\n}\n";
    let cu = emit_src(src);
    let def = fn_def(&cu, "fn0(int32_t in) {");
    assert!(
        def.contains("(char*)o1.f0.f0 >= (char*)arena0"),
        "the veto's range test recurses into nested products:\n{def}"
    );
    assert_eq!(
        bare_free_lines(def),
        Vec::<&str>::new(),
        "no bare frees:\n{def}"
    );
}

// --- plan-last-use §2 rule 4: the in-place Update ------------------------

/// fn f(a: [i32; 4]) -> [i32; 4] { update(a, 0, 1) -> ret } — one Update
/// whose source is the borrowed parameter itself.
fn param_update_cu() -> String {
    let arr4 = Ty::Array {
        elem: Box::new(Ty::i32()),
        size: 4,
    };
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", arr4.clone(), arr4, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let a = fb.input();
        let z = fb.constant(Value::I64(0), L).unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        fb.update(a, z, one, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    crate::emit(&ir).unwrap()
}

#[test]
fn in_place_update_borrowed_parameter_source_keeps_full_copy() {
    // Rule 4's first half: a Parameter source escapes (the plan's rule
    // 2) — never written in place. The target gets its own fn-zone
    // pointer init and the launch's out/src stay distinct handles.
    let cu = param_update_cu();
    assert!(
        cu.contains("= (int32_t*)(arena0 + 0ULL);"),
        "the full-copy target buffer (a zone member) stays:\n{cu}"
    );
    let def = fn_def(&cu, "mapal_main(int32_t* in) {");
    // No in-place handle copy: the launch's out/src stay distinct
    // handles (the parameter's slot is only read, never aliased into).
    let launch = def
        .lines()
        .find(|l| l.contains("<<<"))
        .expect("the update launch");
    let args = launch.split(">>>(").nth(1).expect("launch args");
    let mut operands = args.split(',').map(str::trim);
    let out = operands.next().expect("out arg");
    let src = operands.next().expect("src arg");
    assert_ne!(out, src, "out/src stay distinct (full copy):\n{def}");
}

#[test]
fn in_place_update_dead_literal_source_writes_in_place() {
    // A locally-built array read nowhere after the update (rule 4's
    // dead_after): the target handle IS the source handle — no pointer
    // init, no fresh buffer for the update.
    let arr4 = Ty::Array {
        elem: Box::new(Ty::i32()),
        size: 4,
    };
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", Ty::Unit, arr4, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let one = fb.constant(Value::I32(1), L).unwrap();
        let two = fb.constant(Value::I32(2), L).unwrap();
        let three = fb.constant(Value::I32(3), L).unwrap();
        let four = fb.constant(Value::I32(4), L).unwrap();
        let z = fb
            .pack_array(&[one, two, three, four], Dest::Fresh(None), L)
            .unwrap();
        let idx = fb.constant(Value::I64(0), L).unwrap();
        let nine = fb.constant(Value::I32(9), L).unwrap();
        fb.update(z, idx, nine, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let cu = crate::emit(&ir).unwrap();
    let def = fn_def(&cu, "mapal_main() {");
    let launch = def
        .lines()
        .find(|l| l.contains("<<<"))
        .expect("the update launch");
    let args = launch.split(">>>(").nth(1).expect("launch args");
    let mut operands = args.split(',').map(str::trim);
    let out = operands.next().expect("out arg").to_string();
    let src = operands.next().expect("src arg").to_string();
    // In place: the target was assigned the source handle (no zone
    // pointer init for it), so out == src at the launch.
    assert!(
        def.contains(&format!("{out} = {src};")),
        "the in-place handle copy:\n{def}"
    );
    assert!(
        !def.contains(&format!("{out} = (int32_t*)(arena0")),
        "no fresh buffer for the update target:\n{def}"
    );
    // The literal's own buffer (the zone member the update writes) is
    // still there, and the launch/guard text is unchanged.
    assert!(def.contains("= (int32_t*)(arena0 + 0ULL);"), "{def}");
    assert!(def.contains("trap_check_after_launch();"), "{def}");
}
