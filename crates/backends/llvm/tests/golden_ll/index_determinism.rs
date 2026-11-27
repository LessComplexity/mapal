//! Golden IR for index-guard elision, determinism, nested loops, exit payloads and profile gates.

use super::*;

// --- proven-Index guard elision (S20 `bounds_proof`; the vectorization unlock) ---

/// A proven `Index` — the plan shows its index statically inside `[0, n)` —
/// emits NO bounds guard (the trap is dead); everything unproven keeps the
/// two-sided guard + trap byte-identical. The map body below is the
/// matmul-cell class: an enumerate'd map whose body indexes a captured array
/// with the affine `h * 2 + r` (`h = p.0 / 2 ∈ [0, 1]`, `r = p.0 % 2 ∈ [0,
/// 1]` ⇒ `h*2+r ∈ [0, 3] < 4`). Its `Index` guard is gone (the index-guard
/// shape is the only `icmp slt`/`icmp sge` source here; the signed `Div`/`Mod`
/// zero/`MIN/-1` guards are `icmp eq` and stay). The unproven sibling — `cell`
/// indexing by its unknown i32 parameter — keeps exactly one guard.
#[test]
fn golden_proven_index_guard_elision() {
    let src = r#"
fn cell(a: [f64; 4], i: i32) -> f64 {
    a[i] -> r;
    r -> ret;
}
fn main() {
    [10.0, 20.0, 30.0, 40.0] -> a: [f64; 4];
    a -> enumerate -> map { p ->
        p.0 / 2 -> h;
        p.0 % 2 -> r;
        a[h * 2 + r] -> x;
        x * 2.0 -> y;
        y
    } -> b;
    (b, 2) -> cell -> r2;
    b[1] -> fst;
    r2 + fst -> t;
    t -> println;
}
"#;
    let ll = emit(&lower_src(src)).unwrap();
    let fns: Vec<&str> = ll
        .split("define internal ")
        .skip(1)
        .filter_map(|s| s.split("\n}\n").next())
        .collect();
    // The affine map body (the only fn with `fmul`): the proven Index's guard
    // is elided — no `icmp slt`/`icmp sge`/`or`-trap chain, just the GEP+load
    // (the div/mod `icmp eq` guards are a different op's and stay).
    let map_body = fns
        .iter()
        .find(|s| s.contains("fmul"))
        .expect("the affine map body fn");
    assert_eq!(
        map_body.matches("icmp slt").count() + map_body.matches("icmp sge").count(),
        0,
        "the proven Index emits no bounds guard:\n{map_body}"
    );
    assert_eq!(
        map_body.matches("getelementptr [4 x double]").count(),
        1,
        "the elided Index still emits its GEP+load:\n{map_body}"
    );
    // The unproven sibling (index by the unknown parameter): today's full
    // guard, byte-identical — one slt/sge/or chain + one trap call.
    let cell = fns
        .iter()
        .find(|s| s.contains("{ ptr, i32 } %arg"))
        .expect("the cell fn");
    assert_eq!(
        cell.matches("icmp slt").count(),
        1,
        "an unproven Index keeps the lower-bound compare:\n{cell}"
    );
    assert_eq!(
        cell.matches("icmp sge").count(),
        1,
        "an unproven Index keeps the upper-bound compare:\n{cell}"
    );
    assert_eq!(
        cell.matches("call void @mapal_trap(i32 1)").count(),
        1,
        "an unproven Index keeps the index_oob trap:\n{cell}"
    );
    insta::assert_snapshot!("proven_index_guard_elision", ll);
}

/// The FnAttrs refinement: a fn whose only trap-looking ops are PROVEN
/// `Index`es cannot trap (the guards are elided), so it joins the clean set —
/// `readonly nounwind willreturn`, and a bare by-ref `ptr` parameter
/// additionally carries `noalias nocapture readonly`. Before the refinement
/// both fns below emitted attribute-free (any `Index` counted as
/// trap-capable); the map body also pins the zero-`icmp` cell shape.
#[test]
fn golden_proven_index_fn_clean_attrs() {
    let src = r#"
fn sum4(a: [f64; 4]) -> f64 {
    a[0] -> x;
    a[1] -> y;
    a[2] -> z;
    a[3] -> w;
    x + y -> s1;
    z + w -> s2;
    s1 + s2 -> ret;
}
fn main() {
    [10.0, 20.0, 30.0, 40.0] -> a: [f64; 4];
    a -> enumerate -> map { p -> a[p.0] } -> b;
    b -> sum4 -> s;
    s -> println;
}
"#;
    let ll = emit(&lower_src(src)).unwrap();
    let fns: Vec<&str> = ll
        .split("define internal ")
        .skip(1)
        .filter_map(|s| s.split("\n}\n").next())
        .collect();
    // The cell body (enumerate `.0` index ⇒ proven): no guard icmps at all,
    // and it is attribute-clean.
    let cell_body = fns
        .iter()
        .find(|s| s.contains("{ ptr, { i32, double } } %arg"))
        .expect("the map body fn");
    assert_eq!(
        cell_body.matches("icmp").count(),
        0,
        "the proven cell body has no guard instructions:\n{cell_body}"
    );
    assert!(
        cell_body
            .lines()
            .next()
            .unwrap()
            .contains("readonly nounwind willreturn"),
        "the cell body is attribute-clean:\n{cell_body}"
    );
    // sum4 (four constant indices, all proven): clean, and its bare-Array
    // by-ref parameter carries the full `noalias nocapture readonly` set.
    let sum4 = fns
        .iter()
        .find(|s| s.contains("ptr noalias nocapture readonly %arg"))
        .expect("sum4's by-ref ptr param carries the clean param attrs");
    assert!(
        sum4.lines()
            .next()
            .unwrap()
            .contains("readonly nounwind willreturn"),
        "sum4 is attribute-clean:\n{sum4}"
    );
    assert_eq!(
        sum4.matches("icmp").count(),
        0,
        "sum4's constant indices emit no guards:\n{sum4}"
    );
    insta::assert_snapshot!("proven_index_fn_clean_attrs", ll);
}

// --- determinism (L2) -----------------------------------------------------

#[test]
fn determinism_emit_twice_byte_equal() {
    for name in ["sum_to_n", "sepia", "vector_add"] {
        let a = emit(&build_example(name)).unwrap();
        let b = emit(&build_example(name)).unwrap();
        assert_eq!(a, b, "{name}: emit is not byte-deterministic");
    }
}

#[test]
fn perf_timing_golden() {
    let opts = EmitOpts {
        host: "generic",
        perf_timing: true,
        ..EmitOpts::exact()
    };
    let parallel = emit_with_opts(&build_example("abs"), &opts).unwrap();
    let main = mapal_main(&parallel);
    assert!(
        main.starts_with(
            "define internal void @mapal_main() {\nentry:\n  call void @mapal_perf_begin()\n"
        ),
        "perf begin is the first entry instruction:\n{main}"
    );
    let finish = main
        .find("call void @mapal_par_finish")
        .expect("parallel finish");
    let end = main.find("call void @mapal_perf_end()").expect("perf end");
    let ret = main.rfind("ret void").expect("return");
    assert!(finish < end && end < ret, "parallel timer order:\n{main}");
    insta::assert_snapshot!("perf_timing_mapal_main", main);

    let sequential = emit_with_opts(&lower_src("fn main() {}\n"), &opts).unwrap();
    let main = mapal_main(&sequential);
    assert!(
        !main.contains("mapal_par_finish"),
        "sequential fixture:\n{main}"
    );
    assert!(
        main.starts_with(
            "define internal void @mapal_main() {\nentry:\n  call void @mapal_perf_begin()\n"
        ),
        "sequential perf begin:\n{main}"
    );
    assert!(
        main.ends_with("  call void @mapal_perf_end()\n  ret void\n}\n"),
        "sequential perf end:\n{main}"
    );
}

/// The goldens' helper is the exact face plus the profile pin and nothing
/// else (local-rtx-5080: the pin replaced the ambient `native`).
#[test]
fn default_opts_are_byte_identical() {
    let ir = build_example("vector_add");
    let mut opts = EmitOpts::exact();
    (opts.target, opts.host) = ("generic", "generic");
    assert_eq!(emit(&ir).unwrap(), emit_with_opts(&ir, &opts).unwrap());
}

// --- Unsupported pin (L3) -------------------------------------------------

/// A multi-merge nested loop (two loops cross-fed into one SCC): not the
/// canonical quartet. Shape copied from `mapal-rewrite/tests/identity.rs`.
pub(crate) fn multi_merge_nested_loop() -> CategoryIr {
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "nested", Ty::i32(), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let c0 = fb.constant(Value::I32(0), L).unwrap();
        let c1 = fb.constant(Value::I32(0), L).unwrap();
        let outer = fb.begin_loop(c0, L).unwrap();
        let om = fb.merge_of(&outer);
        let inner = fb.begin_loop(c1, L).unwrap();
        let im = fb.merge_of(&inner);
        let ct = fb.constant(Value::Bool(true), L).unwrap();
        let cf = fb.constant(Value::Bool(false), L).unwrap();
        let inext = fb
            .binop(Operation::Add, om, im, Dest::Fresh(Some("inext".into())), L)
            .unwrap();
        fb.loop_back(&inner, inext, ct, L).unwrap();
        let onext = fb
            .binop(Operation::Add, im, om, Dest::Fresh(Some("onext".into())), L)
            .unwrap();
        fb.loop_back(&outer, onext, ct, L).unwrap();
        fb.loop_exit(&inner, im, cf, Dest::Fresh(Some("iout".into())), L)
            .unwrap();
        fb.end_loop(inner).unwrap();
        fb.loop_exit(&outer, om, cf, Dest::Ret { slot: None }, L)
            .unwrap();
        fb.end_loop(outer).unwrap();
        fb.finish().unwrap();
    }
    b.seal(f).expect("fused nested loops seal")
}

#[test]
fn nested_loop_is_unsupported() {
    let ir = multi_merge_nested_loop();
    let e = emit(&ir).unwrap_err();
    assert!(
        matches!(e, EmitError::Unsupported { ref feature, .. } if feature == "nested loops"),
        "expected Unsupported {{ nested loops }}, got {e:?}"
    );
}

/// S13 orchestrator-review regression: an exit-only computed payload
/// (`acc + 12345` consumed only by the exit route) belongs to the decide cone
/// but lives outside the loop SCC. The walk must not re-emit it after the loop
/// (dead recompute for values; a DOUBLE side effect for an exit-arm Print) —
/// the constant appears exactly once in the module.
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
    let ll = emit(&lower_src(src)).unwrap();
    assert_eq!(
        ll.matches("12345").count(),
        1,
        "exit-only payload must be emitted exactly once (driver-owned):\n{ll}"
    );
}

/// plan-s31-target-profiles, the headline property: the KC nest turns itself
/// off **by derivation**, not by a default-off flag.
///
/// `apple-m` differs from `generic` in exactly one machine fact — a 16 MB L2
/// instead of 512 KB — so its k-panel is `(l2/2) / (nc x sizeof) = 4096`, which
/// is deeper than this site's K=300. The gate `site.k > tile_kc` therefore never
/// opens, and asking for `--kc` on this machine is a no-op. That is S29/S30's
/// measured verdict (the nest is a 3x loss on M4 Pro) reproduced as arithmetic.
///
/// The strong form of the assertion: apple-m WITH the nest requested is
/// byte-equal to generic WITHOUT it — the fallback is the real j-outer nest,
/// not a differently-shaped near-miss.
#[test]
fn profile_closes_the_kc_gate_by_derivation() {
    let ir = mapal_rewrite::rewrite(lower_src(KC_SRC)).ir;
    let kc_on = |target| {
        emit_with_opts(
            &ir,
            &EmitOpts {
                host: "generic",
                kc_nest: true,
                contract: true,
                target,
                ..EmitOpts::exact()
            },
        )
        .expect("emits")
    };

    // TI x KC = 4 x 128 — the a-panel pack scratch, present only in the nest.
    let apack = "alloca [512 x float], align 64";
    assert!(
        kc_on("generic").contains(apack),
        "generic (512 KB L2) must still open the KC gate at K=300"
    );
    assert!(
        !kc_on("apple-m").contains(apack),
        "apple-m (16 MB L2) must close the KC gate by derivation at K=300"
    );

    // ADR-0038 D4: this leg IS the generic reference — name it, or a detected
    // non-NEON profile emits a different j-outer nest and the byte-equality
    // below fails on a healthy machine.
    let generic_kc_off = emit_with_opts(
        &ir,
        &EmitOpts {
            host: "generic",
            contract: true,
            target: "generic",
            ..EmitOpts::exact()
        },
    )
    .expect("emits");
    assert_eq!(
        kc_on("apple-m"),
        generic_kc_off,
        "a closed gate must fall back to the j-outer nest byte-for-byte"
    );
}

/// An unknown profile name is an error, never a silent fall back to `generic`:
/// a typo that quietly emits the default numbers is the failure the table
/// exists to remove (plan rule 3).
#[test]
fn unknown_profile_is_an_error_not_a_silent_default() {
    let ir = lower_src(KC_SRC);
    let err = emit_with_opts(
        &ir,
        &EmitOpts {
            host: "generic",
            target: "apple_m",
            ..EmitOpts::exact()
        },
    )
    .expect_err("unknown profile must not emit");
    let EmitError::Internal(msg) = err else {
        panic!("expected Internal, got {err:?}");
    };
    assert!(
        msg.contains("apple_m") && msg.contains("generic, apple-m, zen3"),
        "{msg}"
    );
}
