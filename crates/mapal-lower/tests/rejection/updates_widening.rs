//! Rejection tests: array element update, iota/fill and numeric widening (L1612-L1614).

use super::*;

// --- ADR-0021: array element update `c[i] <- v` -----------------------------
// The indexed bind is a *rebind* (wiring point a). No new L-codes: a non-array
// target reuses L1204 (index on non-array), an element-ty clash reuses L1201/
// L1203 (width-unification), a non-`mut` target reuses L1104, and an assignment
// in a Phi arm reuses L1408 (wiring point c).

#[test]
fn array_update_non_mut_is_l1104() {
    // `c` bound non-`mut`; `c[0] <- 9` is an immutable rebind → L1104.
    let src = r#"fn main() {
    [0, 0, 0, 0] -> c: [i32; 4];
    c[0] <- 9;
    c[0] -> println;
}
"#;
    assert_rejects(src, "L1104");
}

#[test]
fn array_update_unbound_is_l1101() {
    // `d` never bound; `d[0] <- 9` targets an unknown name → L1101.
    let src = r#"fn main() {
    d[0] <- 9;
}
"#;
    assert_rejects(src, "L1101");
}

#[test]
fn array_update_non_array_target_is_l1204() {
    // `c` is a scalar; `c[0] <- 9` is an index-write on a non-array → L1204.
    let src = r#"fn main() {
    mut c: i32 <- 5;
    c[0] <- 9;
    c -> println;
}
"#;
    assert_rejects(src, "L1204");
}

#[test]
fn array_update_elem_ty_clash_is_l1201() {
    // A concrete element-ty clash (`bool` written into an `i32` array) is caught
    // by width-unification → L1201.
    let src = r#"fn main() {
    mut c: [i32; 4] <- [0, 0, 0, 0];
    c[0] <- true;
    c[0] -> println;
}
"#;
    assert_rejects(src, "L1201");
}

#[test]
fn array_update_in_phi_arm_is_l1408() {
    // `c[0] <- 1` inside a Phi-position arm rebinds an enclosing `mut`
    // unconditionally → L1408 (wiring point c: scan_stmt records the indexed
    // bind's target).
    let src = r#"fn main() {
    mut c: [i32; 4] <- [0, 0, 0, 0];
    5 -> x: i32;
    (x > 0) -> {
        -true-> { c[0] <- 1; x }
        -false-> x
    } -> r: i32;
    r -> println;
}
"#;
    assert_rejects(src, "L1408");
}

#[test]
fn lone_indexed_update_loop_not_l1502() {
    // A loop whose ONLY enclosing-mut mutation is `c[0] <- …` must NOT be
    // L1502-rejected (wiring point b: collect_assigns_stmt records the indexed
    // bind's target, so the mut array joins the carried set). It lowers clean.
    let src = r#"fn main() {
    mut c: [i32; 4] <- [1, 0, 0, 0];
    loop {
        c[0] <- c[0] + 1;
        (c[0] < 4) -> {
            -true-> -> loop;
            -false-> c -> result;
        }
    }
    result[0] -> println;
}
"#;
    let codes = lower_err_codes(src);
    assert!(
        !codes.iter().any(|c| c == "L1502"),
        "L1502 must not fire for a lone indexed-update loop; got {codes:?}"
    );
    // In fact it lowers clean.
    let _ = lower_ok(src);
}

// --- L1612–L1613: iota / fill (ADR-0031 pipeline form) -----------------------

#[test]
fn l1612_iota_tuple_wire_rejects() {
    // The old call-arity case in arrow clothes: a tuple is not a count.
    assert_rejects("fn main() { (4, 5) -> iota -> t; }\n", "L1612");
}

#[test]
fn l1612_iota_non_literal_count_rejects() {
    // An array wire is not a static count (a runtime size is ADR-0023).
    assert_rejects(
        "fn main() { 4 -> iota -> t; t -> k; k -> iota -> u; }\n",
        "L1612",
    );
}

#[test]
fn l1612_iota_zero_rejects_and_oversize_is_width_owned() {
    assert_rejects("fn main() { 0 -> iota -> t; }\n", "L1612");
    // onto-device: an extent past i32 is an `i64` count with `i64` elements
    // (the extent's index type) — accepted; past i64 the literal does not fit.
    assert_rejects(
        "fn main() { 18446744073709551615 -> iota -> t; }\n",
        "L1202",
    );
}

#[test]
fn l1613_fill_wire_shape_rejects() {
    // Not a 2-tuple: a scalar and a 3-tuple both miss the (value, count) shape.
    assert_rejects("fn main() { 1.0 -> fill -> s; }\n", "L1613");
    assert_rejects("fn main() { (1.0, 2, 3) -> fill -> s; }\n", "L1613");
}

#[test]
fn l1613_fill_zero_rejects_and_oversize_is_width_owned() {
    assert_rejects("fn main() { (1.0, 0) -> fill -> s; }\n", "L1613");
    // Same width-system ownership as iota: an extent past i32 is an `i64`
    // count (onto-device), past i64 the literal does not fit (L1202).
    assert_rejects(
        "fn main() { (1.0, 18446744073709551615) -> fill -> s; }\n",
        "L1202",
    );
}

#[test]
fn iota_bound_literal_count_lowers() {
    // ADR-0031 consequence: a NAME bound to a literal is the same Constant
    // object — still static. Strictly more expressive than the old AST check.
    lower_ok("fn main() { 4 -> n; n -> iota -> t; t[2] -> println; }\n");
}

// --- L1614: explicit numeric widening (ADR-0029) ----------------------------

#[test]
fn l1614_invalid_widen_sources_reject() {
    assert_rejects(
        "fn main() { 5 -> x: i64; x -> widen<f64> -> y; }\n",
        "L1614",
    );
    assert_rejects("fn main() { [1, 2] -> widen<i64> -> y; }\n", "L1614");
    // A float literal takes the conversion's width (the identity, ADR-0051);
    // an `f64` does not widen to `f32`.
    assert_rejects(
        "fn main() { 1.0 -> to<f64> -> x; x -> widen<f32> -> y; }\n",
        "L1614",
    );
}

#[test]
fn l1614_message_names_the_pair_and_the_rounding_conversion() {
    // ADR-0051 D4: the message names the pair and, where `to` realizes it,
    // the conversion that rounds; where nothing does, it says so.
    let l1614 = |src: &str| {
        let po = mapal_syntax::parse(src);
        assert!(po.diagnostics.is_empty(), "{:?}", po.diagnostics);
        let ds = mapal_lower::lower(src, &po.program).unwrap_err();
        ds.into_iter()
            .find(|d| d.code.0 == "L1614")
            .unwrap_or_else(|| panic!("expected L1614"))
            .message
    };
    let m = l1614("fn main() { 0.5 -> widen<bf16> -> y; }\n");
    assert!(
        m.contains("cannot widen `f32` to `bf16`") && m.contains("`to<bf16>`"),
        "{m}"
    );
    let m = l1614("fn main() { 5 -> x: i64; x -> widen<f64> -> y; }\n");
    assert!(
        m.contains("cannot widen `i64` to `f64`") && m.contains("yet"),
        "{m}"
    );
}
