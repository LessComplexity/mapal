//! Rejection tests: top-level structure, names and scope, types and literals (L1000-L1209).

use super::*;

// --- L1000–L1010: top-level structure ---------------------------------------

#[test]
fn l1001_no_main() {
    assert_rejects("fn f(x: i32) -> i32 { x -> ret; }\n", "L1001");
}

#[test]
fn l1002_main_shape() {
    // main with a param.
    assert_rejects("fn main(x: i32) { x -> print; }\n", "L1002");
}

#[test]
fn l1003_duplicate_fn() {
    assert_rejects("fn f() {}\nfn f() {}\nfn main() {}\n", "L1003");
}

#[test]
fn l1004_duplicate_type() {
    assert_rejects(
        "type P { x: i32 }\ntype P { y: i32 }\nfn main() {}\n",
        "L1004",
    );
}

#[test]
fn l1005_duplicate_field() {
    assert_rejects("type P { x: i32, x: i32 }\nfn main() {}\n", "L1005");
}

#[test]
fn l1006_duplicate_param() {
    assert_rejects("fn f(x: i32, x: i32) {}\nfn main() {}\n", "L1006");
}

#[test]
fn l1007_recursive_type() {
    assert_rejects("type A { b: B }\ntype B { a: A }\nfn main() {}\n", "L1007");
}

#[test]
fn l1008_recursive_call() {
    assert_rejects(
        "fn f() { -> g; }\nfn g() { -> f; }\nfn main() {}\n",
        "L1008",
    );
}

#[test]
fn l1009_reserved_name() {
    assert_rejects("fn print() {}\nfn main() {}\n", "L1009");
}

#[test]
fn l1009_reserved_collection_builtins() {
    // ADR-0018: `zip`/`enumerate` are builtins resolved by name, so a user `fn`
    // of either name collides exactly like `print` — L1009.
    assert_rejects("fn zip() {}\nfn main() {}\n", "L1009");
    assert_rejects("fn enumerate() {}\nfn main() {}\n", "L1009");
    assert_rejects("fn widen<i64>() {}\nfn main() {}\n", "L1009");
    assert_rejects("fn widen<f32>() {}\nfn main() {}\n", "L1009");
    assert_rejects("fn widen<f64>() {}\nfn main() {}\n", "L1009");
}

#[test]
fn l1009_reserved_time() {
    // plan-time-builtin: `time` is a reserved stage name like `print`/`iota`,
    // so a user fn of that name collides — L1009.
    assert_rejects("fn time() {}\nfn main() {}\n", "L1009");
}

#[test]
fn l1010_empty_type() {
    assert_rejects("type Empty {}\nfn main() {}\n", "L1010");
}

// --- L1101–L1108: names / scope ---------------------------------------------

#[test]
fn l1101_unknown_name() {
    assert_rejects("fn main() { nope -> print; }\n", "L1101");
}

#[test]
fn l1102_unknown_type() {
    assert_rejects("fn f(x: Nope) {}\nfn main() {}\n", "L1102");
}

#[test]
fn l1103_unknown_field() {
    assert_rejects(
        "type P { x: i32 }\nfn f(p: P) -> i32 { p.y -> ret; }\nfn main() {}\n",
        "L1103",
    );
}

#[test]
fn l1104_assign_immutable() {
    assert_rejects(
        "fn f(x: i32) -> i32 { x + 1 -> x; x -> ret; }\nfn main() {}\n",
        "L1104",
    );
}

#[test]
fn l1105_function_as_value() {
    // a fn name in expression position.
    assert_rejects("fn g() {}\nfn main() { g + 1 -> print; }\n", "L1105");
}

#[test]
fn l1106_named_param_application() {
    assert_rejects(
        "fn add(a: i32, b: i32) -> i32 { a + b -> ret; }\nfn main() { 15 -> add.a; }\n",
        "L1106",
    );
}

#[test]
fn l1107_read_after_loop() {
    let src = r#"fn sum(n: i32) -> i32 {
    mut i: i32 <- 0;
    mut acc: i32 <- 0;
    loop {
        (i < n) -> {
            -true-> { acc + i -> acc; i + 1 -> i; -> loop; }
            -false-> acc -> ret;
        }
    }
    acc -> ret;
}
fn main() {}
"#;
    assert_rejects(src, "L1107");
}

#[test]
fn l1107_capture_of_loop_carried_name_after_loop() {
    // ADR-0027 review blocker #2: a poisoned (loop-carried) name read as a
    // map-body capture after its loop must report L1107 like the direct read
    // — the capture resolution must not discard the poison bit (before the
    // fix it lowered, and the interp panicked).
    let src = r#"fn main() {
    mut acc: i32 <- 0;
    mut i: i32 <- 0;
    loop {
        (i < 3) -> {
            -true-> { acc + i -> acc; i + 1 -> i; -> loop; }
            -false-> acc -> done;
        }
    }
    [1, 2] -> map { x -> x + acc } -> ys: [i32; 2];
}
"#;
    assert_rejects(src, "L1107");
}

// (The pre-ADR-0027 `l1108_capture_in_body` row moved to tests/captures.rs as
// the positive pin `capture_map_lowers_with_capture_input_product`: reads of
// enclosing bindings are legal captures now — D1.)

#[test]
fn l1108_rebind_of_read_capture_names_the_variable() {
    // ADR-0027 review major #4 (D2b/D3): a chain rebind `-> k` of a name the
    // body has already captured (read) must report the TEACHING L1108 naming
    // `k` — not fall through to emit's generic L1104 ("not `mut`").
    let src = r#"fn main() {
    5 -> k;
    [1, 2] -> map { e -> e + k -> d; e + 1 -> k; d + k } -> ys: [i32; 2];
}
"#;
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "{:?}", po.diagnostics);
    let ds = match mapal_lower::lower(src, &po.program) {
        Ok(_) => panic!("must reject the rebind of a read capture"),
        Err(ds) => ds,
    };
    let d = ds
        .iter()
        .find(|d| d.code.0 == "L1108")
        .unwrap_or_else(|| panic!("expected L1108, got {ds:?}"));
    assert!(d.message.contains("`k`"), "names the variable: {d:?}");
    assert!(
        !ds.iter().any(|d| d.code.0 == "L1104"),
        "not the generic L1104: {ds:?}"
    );
}

// --- L1201–L1209: types / literals ------------------------------------------

#[test]
fn l1201_type_mismatch() {
    // adding an i32 and a bool.
    assert_rejects(
        "fn f(x: i32, b: bool) -> i32 { x + b -> ret; }\nfn main() {}\n",
        "L1201",
    );
}

#[test]
fn l1202_literal_out_of_range() {
    // 300 does not fit u8.
    assert_rejects("fn f() -> u8 { 300 -> ret; }\nfn main() {}\n", "L1202");
}

#[test]
fn l1203_literal_type_conflict() {
    // an int literal unified with an f32 annotation.
    assert_rejects("fn f() -> f32 { 5 -> ret; }\nfn main() {}\n", "L1203");
}

#[test]
fn l1204_not_a_product() {
    // `.0` index on a scalar.
    assert_rejects(
        "fn f(x: i32) -> i32 { x.0 -> ret; }\nfn main() {}\n",
        "L1204",
    );
}

#[test]
fn l1205_slot_out_of_range() {
    // ret.5 on a 2-tuple output. (Also covers the u64→u32 lesson via OOB.)
    assert_rejects(
        "fn f() -> (i32, i32) { 1 -> ret.0; 2 -> ret.5; }\nfn main() {}\n",
        "L1205",
    );
}

#[test]
fn l1206_str_outside_print() {
    // a string literal returned (not feeding print).
    assert_rejects("fn f() -> i32 { \"hi\" -> ret; }\nfn main() {}\n", "L1206");
}

#[test]
fn l1207_unprintable() {
    // printing a tuple.
    assert_rejects("fn main() { (1, 2) -> print; }\n", "L1207");
}

#[test]
fn l1208_empty_array() {
    assert_rejects("fn f(a: [i32; 0]) {}\nfn main() {}\n", "L1208");
}

#[test]
fn l1209_type_too_deep() {
    // 65 levels of array nesting (parser guards at 128; 65 is lower's job).
    let mut ty = String::from("i32");
    for _ in 0..65 {
        ty = format!("[{ty}; 1]");
    }
    let src = format!("fn f(x: {ty}) {{}}\nfn main() {{}}\n");
    assert_rejects(&src, "L1209");
}
