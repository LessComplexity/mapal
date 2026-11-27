//! Golden IR for the examples, micro programs, updates, loops and captures.

use super::*;

#[test]
fn golden_examples() {
    for name in EXAMPLES {
        let ir = build_example(name);
        let ll = emit(&ir).unwrap();
        insta::assert_snapshot!(format!("example_{name}"), ll);
    }
}

// --- micro shapes ---------------------------------------------------------

/// Signed `Div`/`Mod`: zero-guard trap block + `MIN/-1` guard (Div ⇒ MIN, Mod ⇒
/// 0), plus wrapping `add` (no nsw).
#[test]
fn golden_micro_arith() {
    let src = r#"
fn f(a: i32, b: i32) -> i32 {
    a / b -> q;
    a % b -> r;
    q + r -> ret;
}
fn main() {
    (7, 3) -> f -> v;
    v -> println;
}
"#;
    let ll = emit(&lower_src(src)).unwrap();
    insta::assert_snapshot!("micro_arith", ll);
}

/// `Update` (ADR-0021): index guard + `llvm.memcpy` + dynamic GEP store. Built
/// via `IrBuilder` (surface rebind sugar is a later WP).
#[test]
fn golden_micro_update() {
    let mut b = IrBuilder::new();
    let arr_ty = Ty::Array {
        elem: Box::new(Ty::i32()),
        size: 4,
    };
    let f = b
        .declare(FuncKind::Named, "main", arr_ty.clone(), arr_ty, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let a = fb.input();
        let i = fb.constant(Value::I32(2), L).unwrap();
        let v = fb.constant(Value::I32(99), L).unwrap();
        fb.update(a, i, v, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    let ll = emit(&ir).unwrap();
    insta::assert_snapshot!("micro_update", ll);
}

#[test]
fn widen_emits_conversion_instructions() {
    let input = Ty::Tuple(vec![Ty::i32(), Ty::f32()]);
    let output = Ty::Tuple(vec![Ty::i64(), Ty::f32(), Ty::f64(), Ty::f64()]);
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "widen", input, output, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let arg = fb.input();
        let i = fb.proj(arg, 0, Dest::Fresh(None), L).unwrap();
        let f32 = fb.proj(arg, 1, Dest::Fresh(None), L).unwrap();
        let i64 = fb.widen(i, Ty::i64(), Dest::Fresh(None), L).unwrap();
        let i_f32 = fb.widen(i, Ty::f32(), Dest::Fresh(None), L).unwrap();
        let i_f64 = fb.widen(i, Ty::f64(), Dest::Fresh(None), L).unwrap();
        let f_f64 = fb.widen(f32, Ty::f64(), Dest::Fresh(None), L).unwrap();
        fb.pack(&[i64, i_f32, i_f64, f_f64], Dest::Ret { slot: None }, L)
            .unwrap();
        fb.finish().unwrap();
    }
    let ll = emit(&b.seal(f).unwrap()).unwrap();
    assert_eq!(ll.matches(" = sext i32 ").count(), 1, "{ll}");
    assert_eq!(ll.matches(" = sitofp i32 ").count(), 2, "{ll}");
    assert_eq!(ll.matches(" = fpext float ").count(), 1, "{ll}");
}

// --- last-use Update elision (suggestions #2; plan-last-use §2 rule 4) -----

/// The matmul4-class loop form — a loop-carried array rebuilt per iteration by
/// `c[t] <- v` — elides the whole-array memcpy: the plan proves the update's
/// source (the merge's Proj view) `dead_after` the update (decide-cone reads
/// and the exit-route pin rank before the advance-cone update; ¬escapes,
/// ¬carried), so the element store lands in the carried storage in place and
/// the Update target has no separate array alloca. Other semantically owed
/// array transfers use `llvm.memcpy` per WP3b and are not part of this count.
#[test]
fn golden_update_inplace_carried_loop() {
    let src = r#"
fn build(n: i32) -> [i32; 4] {
    mut c: [i32; 4] <- [0, 0, 0, 0];
    mut t: i32 <- 0;
    loop {
        (t < 4) -> {
            -true-> {
                t + 1 -> v;
                c[t] <- v;
                t + 1 -> t;
                -> loop;
            }
            -false-> c -> ret;
        }
    }
}
fn main() {
    0 -> build -> r;
    r[2] -> println;
}
"#;
    let build_of = |ll: &str| -> String {
        ll.split("define internal void @fn")
            .nth(1)
            .and_then(|s| s.split("\n}\n").next())
            .expect("build fn")
            .to_string()
    };
    let mut gpu = GpuArms::default();
    *gpu.off.rung("by-ref").expect("rung") = true;
    let s64 = emit_with_opts(
        &lower_src(src),
        &EmitOpts {
            gpu,
            ..EmitOpts::exact()
        },
    )
    .unwrap();
    let build = build_of(&s64);
    // structure: the returned array is the caller's storage (`ptr %out`),
    // never an alloca of the callee.
    assert_eq!(
        build.matches("alloca [4 x i32]").count(),
        2,
        "the carried Update target shares its source slot (no third array alloca):\n{build}"
    );
    // S65 `by-ref`: the state is the literal's buffer (it dies at the entry),
    // the Update writes it in place through the state's pointer, and the
    // loop products forward that pointer — the exit copies the value out into
    // the caller's storage (the caller observes it).
    let ll = emit(&lower_src(src)).unwrap();
    let build = build_of(&ll);
    assert_eq!(build.matches("alloca [4 x i32]").count(), 1, "{build}");
    assert_eq!(build.matches("@llvm.memcpy").count(), 1, "{build}");
    insta::assert_snapshot!("update_inplace_carried_loop", ll);
}

/// The elision's negative pins: when the plan cannot prove the source dead,
/// the fresh-alloca copy stays.
///
/// 1. Escape: the pre-update array is packed into the returned product (rule
///    2's escape through `Pair` fields) — the update keeps its memcpy.
/// 2. By-ref source: a Named fn's bare-Array parameter arrives as `ptr`
///    (suggestions #8) — ptr-resident, borrowed caller memory, never written
///    in place even though the rebind is its only non-escape use. The `ptr`
///    signature assertion pins that the by-ref shape is actually exercised.
#[test]
fn golden_update_memcpy_kept_when_not_dead() {
    // Escape via a Pair field of the output product.
    let escape = r#"
fn f(n: i32) -> ([i32; 4], [i32; 4]) {
    mut a: [i32; 4] <- [1, 2, 3, 4];
    a -> orig;
    a[1] <- 9;
    (orig, a) -> ret;
}
fn main() {
    0 -> f -> r;
    r.0[1] -> println;
    r.1[1] -> println;
}
"#;
    let ll = emit(&lower_src(escape)).unwrap();
    let f = ll
        .split("define internal void @fn")
        .nth(1)
        .and_then(|s| s.split("\n}\n").next())
        .expect("escape f");
    assert_eq!(
        f.matches("call void @llvm.memcpy").count(),
        3,
        "the two owed return-field copies plus the escaping Update copy stay:\n{f}"
    );

    // Borrowed by-ref source (ptr-resident): the in-place write would land in
    // caller memory — vetoed, memcpy stays.
    let byref = r#"
fn f(a: [i32; 4]) -> [i32; 4] {
    mut c: [i32; 4] <- a;
    c[1] <- 99;
    c -> ret;
}
fn main() {
    [1, 2, 3, 4] -> f -> r;
    r[1] -> println;
}
"#;
    let ll = emit(&lower_src(byref)).unwrap();
    let fdef = ll
        .lines()
        .find(|l| l.starts_with("define internal void @fn"))
        .expect("f's define line");
    assert!(
        fdef.contains("(ptr"),
        "f's array parameter arrives by reference (the ptr-resident shape):\n{fdef}"
    );
    assert_eq!(
        ll.matches("call void @llvm.memcpy").count(),
        2,
        "the borrowed Update copy and owed output copy stay:\n{ll}"
    );
}

/// Two sequential canonical loops in one fn (S12 P0 shape) — each its own
/// guard-first quartet, exit attribution by route-feeder membership.
#[test]
fn golden_micro_two_loops() {
    let src = r#"
fn f(n: i32) -> i32 {
    mut i: i32 <- 0;
    mut a: i32 <- 0;
    loop {
        (i < n) -> {
            -true-> { a + 2 -> a; i + 1 -> i; -> loop; }
            -false-> a -> aa;
        }
    }
    mut j: i32 <- 0;
    mut b: i32 <- 0;
    loop {
        (j < n) -> {
            -true-> { b + 3 -> b; j + 1 -> j; -> loop; }
            -false-> b -> bb;
        }
    }
    aa + bb -> ret;
}
fn main() {
    4 -> f -> r;
    r -> println;
}
"#;
    let ll = emit(&lower_src(src)).unwrap();
    insta::assert_snapshot!("micro_two_loops", ll);
}

// --- ADR-0027 captures ----------------------------------------------------

/// A source-level capturing map: the body reads the enclosing `scale`, lowered
/// to a hidden leading input component — the body fn's input is `(cap, elem)`
/// and the map loop's call passes the assembled product (the capture is a real
/// leading argument, broadcast to every element). One map loop, no other
/// counted loop.
#[test]
fn golden_capture_map() {
    let src = r#"
fn main() {
    3 -> scale;
    [1, 2, 3] -> a;
    a -> map { x -> x * scale } -> b;
    b[1] -> println;
}
"#;
    let ll = emit(&lower_src(src)).unwrap();
    assert_eq!(
        ll.matches("icmp uge").count(),
        1,
        "exactly one counted (map) loop:\n{ll}"
    );
    let call = ll
        .lines()
        .find(|l| l.contains(" = call i32 @fn"))
        .expect("map body call site");
    assert!(
        call.contains("{ i32, i32 }"),
        "the body call passes the (capture, elem) product: {call}"
    );
    insta::assert_snapshot!("capture_map", ll);
}

/// A source-level capturing fold: `acc + x * scale` reads the enclosing
/// `scale` — the step fn's input is `(cap, acc, elem)` and the fold loop's
/// call passes all three (capture leading, per ADR-0027).
#[test]
fn golden_capture_fold() {
    let src = r#"
fn main() {
    3 -> scale;
    [1, 2, 3] -> a;
    (0, a) -> fold { acc, x -> acc + x * scale } -> total;
    total -> println;
}
"#;
    let ll = emit(&lower_src(src)).unwrap();
    assert_eq!(
        ll.matches("icmp uge").count(),
        1,
        "exactly one counted (fold) loop:\n{ll}"
    );
    let call = ll
        .lines()
        .find(|l| l.contains(" = call i32 @fn"))
        .expect("fold body call site");
    assert!(
        call.contains("{ i32, i32, i32 }"),
        "the step call passes the (capture, acc, elem) product: {call}"
    );
    insta::assert_snapshot!("capture_fold", ll);
}

/// The ADR-0021 motivating program in its natural ADR-0027 form: the
/// one-kernel matmul — a map over `enumerate` whose body is a fold over the
/// captured `a`/`b` (the body also captures the loop-derived `i`/`j`). ONE
/// elementwise map loop around one fold loop; the captured arrays reach the
/// inner body's `Index` operands as leading body-input components — **by
/// reference** (suggestions #6): the first-k Array components of each body
/// input lower to `ptr`, so no array bytes cross the per-element/per-step
/// call boundary (the 64 KB-per-call memcpy measured at S19).
#[test]
fn golden_capture_one_kernel_matmul() {
    // `a` doubles as the enumerate seed (only its length matters). N=4.
    let src = r#"
fn matmul(a: [f32; 16], b: [f32; 16], ks: [i32; 4]) -> [f32; 16] {
    a -> enumerate -> map { p ->
        p.0 / 4 -> i;
        p.0 % 4 -> j;
        (0.0, ks) -> fold { acc, k -> acc + a[i * 4 + k] * b[k * 4 + j] } -> cell;
        cell
    } -> c;
    c -> ret;
}
fn main() {
    [ -37.0, -30.0, -23.0, -16.0, -9.0, -2.0, 5.0, 12.0,
      19.0, 26.0, 33.0, 40.0, 47.0, -47.0, -40.0, -33.0] -> a: [f32; 16];
    [7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, -45.0,
     -38.0, -31.0, -24.0, -17.0, -10.0, -3.0, 4.0, 11.0] -> b: [f32; 16];
    [0, 1, 2, 3] -> ks: [i32; 4];
    (a, b, ks) -> matmul -> c;
    c[0] -> println;
    c[15] -> println;
}
"#;
    let ll = emit(&lower_src(src)).unwrap();
    // Three counted loops total: enumerate + ONE map (in `matmul`) + the fold
    // (inside the map body fn) — the map is one loop, not per-element calls.
    assert_eq!(
        ll.matches("icmp uge").count(),
        3,
        "enumerate + ONE map loop + ONE fold loop:\n{ll}"
    );
    // Exactly two body call sites: the map body per element (in the map loop)
    // and the fold body per step (inside the map body) — no other per-element
    // call structure. Both pass their array captures as `ptr` fields.
    assert_eq!(
        ll.matches(" = call float @fn").count(),
        2,
        "one map-body call site + one fold-body call site:\n{ll}"
    );
    assert!(
        ll.contains(" = call float @fn") && ll.contains("({ ptr, ptr, ptr, { i32, float } } %"),
        "the map-body call passes the three array captures (ks, a, b) by reference:\n{ll}"
    );
    assert!(
        ll.contains("({ ptr, i32, ptr, i32, float, i32 } %"),
        "the fold-step call passes the captured a/b by reference (i/j by value):\n{ll}"
    );
    // The fold body (the only fn with `fmul` — the dot-product step): its
    // input product carries the captured `a`/`b` as `ptr`; each capture Proj
    // lands in an `alloca ptr`, and `a[i*4+k]` / `b[k*4+j]` index through the
    // forwarded pointers with dynamic (i64) GEPs — no inline array copies in
    // the body-input product.
    let fold_body = ll
        .split("define internal ")
        .skip(1)
        .filter_map(|s| s.split("\n}\n").next())
        .find(|s| s.contains("fmul"))
        .expect("the fold body fn");
    assert!(
        fold_body.contains("{ ptr, i32, ptr, i32, float, i32 } %arg"),
        "the fold body's array captures arrive by reference:\n{fold_body}"
    );
    assert_eq!(
        fold_body.matches("alloca ptr").count(),
        2,
        "each captured-array Proj lands in an `alloca ptr` slot:\n{fold_body}"
    );
    assert_eq!(
        fold_body
            .matches("getelementptr [16 x float], ptr %t")
            .count(),
        2,
        "a[i*4+k] and b[k*4+j] index the captured arrays through the forwarded pointers:\n{fold_body}"
    );
    insta::assert_snapshot!("capture_one_kernel_matmul", ll);
}
