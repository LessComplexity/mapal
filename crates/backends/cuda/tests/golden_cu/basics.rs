//! Golden CUDA for the examples, micro programs and update/copy behaviour.

use super::*;

#[test]
fn golden_examples() {
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
        let cu = emit(&build_example(name)).unwrap();
        insta::assert_snapshot!(format!("example_{name}"), cu);
    }
}

/// Signed `Div`/`Mod`: zero-guard host trap + `MIN/-1` value guard
/// (Div ⇒ `INT32_MIN`, Mod ⇒ `0`), plus unsigned-cast wrapping `Add`.
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
    let cu = emit(&lower_src(src)).unwrap();
    insta::assert_snapshot!("micro_arith", cu);
}

#[test]
fn widen_is_scalar_and_emits_host_c_casts() {
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
    let cu = emit(&b.seal(f).unwrap()).unwrap();
    assert!(!cu.contains("__global__"), "{cu}");
    assert!(!cu.contains("arena0"), "{cu}");
    assert_eq!(cu.matches("(int64_t)(").count(), 1, "{cu}");
    assert_eq!(cu.matches("(float)(").count(), 1, "{cu}");
    assert_eq!(cu.matches("(double)(").count(), 2, "{cu}");
}

#[test]
fn golden_iota_fill() {
    let cu = emit(&lower_src(IOTA_FILL_SRC)).unwrap();
    insta::assert_snapshot!("iota_fill", cu);
}

/// Two sequential canonical loops in one fn (S12 P0 shape, the llvm
/// `golden_micro_two_loops` source verbatim) — each its own guard-first
/// quartet, exit attribution by route-feeder membership.
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
    let cu = emit(&lower_src(src)).unwrap();
    insta::assert_snapshot!("micro_two_loops", cu);
}

/// The ADR-0021 motivating pattern, smallest honest version: a loop-carried
/// `mut c` array built by `c[t] <- v` per iteration and returned — the merge
/// is a host handle, the back edge a pointer swap, and each iteration's
/// `Update` is IN PLACE (plan-last-use §2 rule 4: the carried source dies at
/// the update — no fresh buffer, no per-iteration malloc; the launch is
/// unchanged, writing the source handle itself). The escape through the loop
/// exit rides the zone release's range-test veto: the returned handle IS the
/// in-placed, zone-resident init buffer.
pub(crate) const LOOP_UPDATE_SRC: &str = r#"
fn build(n: i32) -> [i32; 4] {
    [0, 0, 0, 0] -> z: [i32; 4];
    mut c: [i32; 4] <- z;
    mut t: i32 <- 0;
    loop {
        (t < n) -> {
            -true-> { c[t] <- t * 10; t + 1 -> t; -> loop; }
            -false-> c -> ret;
        }
    }
}
fn main() {
    4 -> build -> c;
    c[2] -> println;
}
"#;

#[test]
fn golden_micro_loop_update() {
    let cu = emit(&lower_src(LOOP_UPDATE_SRC)).unwrap();
    insta::assert_snapshot!("micro_loop_update", cu);
}

// --- plan-last-use §2 rule 4: the in-place Update (structural pins) ---------

/// Rule 4's escape half: the update's source escapes through a returned
/// product's field (rule 2's "through Pair fields") — the full copy +
/// fresh buffer stay.
#[test]
fn update_escaping_source_keeps_full_copy() {
    let src = r#"
fn mk(x: i32) -> ([i32; 4], [i32; 4]) {
    [1, 2, 3, 4] -> a: [i32; 4];
    mut b: [i32; 4] <- a;
    b[0] <- x;
    (a, b) -> ret;
}
fn main() {
    9 -> mk -> p;
    p.1 -> b;
    b[0] -> println;
}
"#;
    let cu = emit(&lower_src(src)).unwrap();
    let mk_start = cu.find("static FlowProd").expect("mk def:\n{cu}");
    let mk_end = cu[mk_start..].find("\n}\n").map(|e| mk_start + e).unwrap();
    let mk = &cu[mk_start..mk_end];
    // Two zone pointer inits (the literal a AND the update target) — the
    // in-place form would skip the target's.
    assert_eq!(
        mk.matches("= (int32_t*)(arena0 + ").count(),
        2,
        "escape: the update keeps its own fresh buffer:\n{mk}"
    );
    // The launch's out/src are distinct handles.
    let launch = mk.lines().find(|l| l.contains("<<<")).expect("launch");
    let args = launch.split(">>>(").nth(1).expect("launch args");
    let mut operands = args.split(',').map(str::trim);
    assert_ne!(
        operands.next().expect("out"),
        operands.next().expect("src"),
        "out/src stay distinct:\n{mk}"
    );
}

/// Rule 4's ordering half: the source read AFTER the update (a later use of
/// the old value) vetoes the in-place write — the full copy keeps the old
/// buffer's contents intact for the read.
#[test]
fn update_source_read_after_write_keeps_full_copy() {
    let src = r#"
fn main() {
    [1, 2, 3, 4] -> a: [i32; 4];
    mut b: [i32; 4] <- a;
    b[0] <- 9;
    a[1] -> println;
    b[0] -> println;
}
"#;
    let cu = emit(&lower_src(src)).unwrap();
    // The update target gets its own fresh (zone) buffer and the launch's
    // out/src stay distinct handles — the in-place form would have aliased
    // them and corrupted the later `a[1]` read.
    let launch = cu.lines().find(|l| l.contains("<<<")).expect("launch");
    let args = launch.split(">>>(").nth(1).expect("launch args");
    let mut operands = args.split(',').map(str::trim);
    let out = operands.next().expect("out");
    let src = operands.next().expect("src");
    assert_ne!(out, src, "out/src stay distinct:\n{cu}");
    assert!(
        cu.contains(&format!("{out} = (int32_t*)(arena0 + ")),
        "the update keeps its own fresh buffer:\n{cu}"
    );
}

/// Rule 4's consumer half (mapal-ir's `last_use_borrowed_init_is_never_dead`
/// pin): a loop carrying a BORROWED init (the fn's array parameter feeds the
/// LoopEnter) never writes in place and is never freed at the back edge —
/// the per-iteration full-copy malloc + the value-guarded per-buffer free
/// stay (today's O(k·n) behavior, the conservative default).
#[test]
fn loop_update_borrowed_init_keeps_full_copy() {
    let src = r#"
fn build(z: [i32; 4], n: i32) -> [i32; 4] {
    mut c: [i32; 4] <- z;
    mut t: i32 <- 0;
    loop {
        (t < n) -> {
            -true-> { c[t] <- t * 10; t + 1 -> t; -> loop; }
            -false-> c -> ret;
        }
    }
}
fn main() {
    ([1, 2, 3, 4], 4) -> build -> c;
    c[2] -> println;
}
"#;
    let cu = emit(&lower_src(src)).unwrap();
    let build_start = cu.find("static int32_t* fn").expect("build def:\n{cu}");
    let build_end = cu[build_start..]
        .find("\n}\n")
        .map(|e| build_start + e)
        .unwrap();
    let build = &cu[build_start..build_end];
    // The borrowed init vetoes the in-place update: the per-iteration
    // full-copy buffer stays (one cudaMalloc inside the loop body).
    let brk = build.find(") { break; }").unwrap();
    let malloc = build
        .find("cu_check(cudaMalloc((void**)&o")
        .expect("per-buffer malloc");
    assert!(brk < malloc, "the full-copy malloc stays in-loop:\n{build}");
    // The update target stays a registered allocation: its value-guarded
    // per-buffer free at fn exit is the tier-1 escape guard.
    assert_eq!(build.matches("!= o1) {").count(), 1, "{build}");
    // No back-edge free either (the plan says borrowed): no cudaFree inside
    // the loop body.
    let loop_close = build[brk..].find("\n  }\n").map(|i| brk + i).unwrap();
    assert!(
        !build[brk..loop_close].contains("cudaFree"),
        "no back-edge free where the plan says borrowed:\n{build}"
    );
}

/// The map-in-loop carried shape — suggestions.md #2's back-edge freeing:
/// the carried buffer is NOT produced by an Update, so it keeps its
/// per-iteration malloc (a registered cone-site buffer), but the back edge
/// frees the merge's OUTGOING handle under the pointer-value init guard —
/// the per-iteration buffers no longer accumulate to fn exit. The plan's
/// half: `dead_after(merge, LoopBack)` (the guard cond reads only `t`) and
/// an owned init literal; the consumer's half: the producer is a registered
/// allocation and the init is not borrowed. The producer's final instance
/// still rides the fn-exit escape value guard (the second line of defense).
#[test]
fn map_in_loop_back_edge_frees_outgoing_handle() {
    let src = r#"
fn build(n: i32) -> [i32; 4] {
    [1, 2, 3, 4] -> z: [i32; 4];
    mut c: [i32; 4] <- z;
    mut t: i32 <- 0;
    loop {
        (t < n) -> {
            -true-> { c -> map { x -> x + 1 } -> c; t + 1 -> t; -> loop; }
            -false-> c -> ret;
        }
    }
}
fn main() {
    4 -> build -> c;
    c[0] -> println;
}
"#;
    let cu = emit(&lower_src(src)).unwrap();
    let build_start = cu.find("static int32_t* fn").expect("build def:\n{cu}");
    let build_end = cu[build_start..]
        .find("\n}\n")
        .map(|e| build_start + e)
        .unwrap();
    let build = &cu[build_start..build_end];
    for needle in [
        // The per-iteration map buffer (registered) stays — this class is
        // not in-place-able, it is FREED instead.
        "cu_check(cudaMalloc((void**)&o9, sizeof(int32_t) * 4ULL), \"cudaMalloc(o9)\");",
        // The back-edge free: the merge's outgoing handle under the
        // pointer-value init guard (the first iteration holds the init's
        // zone buffer — never a registered allocation — so the guard skips
        // it; later iterations hold the previous producer buffer).
        "if (o4.f0 != o3.f0) {",
        "cu_check(cudaFree(o4.f0), \"cudaFree(o4.f0)\");",
        // The final instance is still freed at fn exit under the escape
        // value guard (the emitted-text second line of defense, unchanged).
        "if (o9 != o1) {",
    ] {
        assert!(build.contains(needle), "missing `{needle}` in:\n{build}");
    }
    // Sequencing: the free sits after the guard's break and before the
    // back-edge swap — the outgoing handle dies exactly at the swap.
    let brk = build.find(") { break; }").unwrap();
    let free = build.find("cudaFree(o4.f0)").unwrap();
    let swap = build.find("o4 = o13.f0;").expect("the back-edge swap");
    assert!(
        brk < free && free < swap,
        "the back-edge free precedes the swap:\n{build}"
    );
    // One free per iteration replaces accumulate-to-exit: exactly one
    // cudaFree inside the loop body, and the fn-exit frees are unchanged.
    let loop_close = build[swap..].find("\n  }\n").map(|i| swap + i).unwrap();
    assert_eq!(
        build[brk..loop_close].matches("cu_check(cudaFree(").count(),
        1,
        "{build}"
    );
    // The freed shape is deterministic (L2): same IR → byte-identical text.
    assert_eq!(
        cu,
        emit(&lower_src(src)).unwrap(),
        "back-edge freeing is byte-deterministic"
    );
}
/// a body-local array whose only use is the update skips the per-thread
/// copy — the store lands in the source array directly, and the target
/// declares no produced local.
#[test]
fn twin_update_dead_source_skips_copy() {
    let src = r#"
fn main() {
    [1, 2, 3, 4] -> arr: [i32; 4];
    arr -> map { x ->
        mut local: [i32; 2] <- [10, 20];
        local[0] <- x;
        local[0]
    } -> out;
    out[2] -> println;
}
"#;
    let cu = emit(&lower_src(src)).unwrap();
    let twin_start = cu.find("static __device__").expect("a device twin:\n{cu}");
    let twin_end = cu[twin_start..]
        .find("\n}\n")
        .map(|e| twin_start + e)
        .unwrap();
    let twin = &cu[twin_start..twin_end];
    // The twin's update wrote the source array directly: the bounds guard
    // and the direct store stay, the per-thread copy loop is gone — the
    // twin has no for-loop at all in this shape.
    assert!(twin.contains("< 0 ||"), "{twin}");
    assert!(!twin.contains("for (unsigned long long"), "{twin}");
    let store = twin
        .lines()
        .find(|l| l.contains("[(unsigned long long)t") && l.contains("] = "))
        .expect("the direct store:\n{twin}");
    assert!(!store.contains("].f"), "scalar element store:\n{twin}");
}
