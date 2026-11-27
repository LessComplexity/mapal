//! Rejection tests: loops, map/fold, zip/enumerate (L1501-L1610).

use super::*;

// --- L1501–L1504: loops ------------------------------------------------------

#[test]
fn l1501_loop_no_exit() {
    // a loop body with no routing guard at all (a plain Phi guard only).
    let src = r#"fn f() {
    mut i: i32 <- 0;
    loop {
        (i < 10) -> { -true-> i + 1; -false-> i; } -> i;
    }
}
fn main() {}
"#;
    assert_rejects(src, "L1501");
}

#[test]
fn l1502_loop_no_state() {
    // a loop assigning no enclosing mut.
    let src = r#"fn f() {
    loop {
        (1 < 2) -> {
            -true-> -> loop;
            -false-> -> ret;
        }
    }
}
fn main() {}
"#;
    assert_rejects(src, "L1502");
}

#[test]
fn l1503_non_derived_cond() {
    // the guard cond does not derive from the loop state (`5 -> acc; (acc > 0)`).
    let src = r#"fn f() {
    mut acc: i32 <- 0;
    loop {
        5 -> acc;
        (acc > 0) -> {
            -true-> { acc + 1 -> acc; -> loop; }
            -false-> acc -> ret;
        }
    }
}
fn main() {}
"#;
    assert_rejects(src, "L1503");
}

#[test]
fn l1503_non_derived_next_state() {
    // the back-edge next-state is merge-independent (`5 -> acc; -> loop`).
    let src = r#"fn f(n: i32) -> i32 {
    mut acc: i32 <- 0;
    mut i: i32 <- 0;
    loop {
        (i < n) -> {
            -true-> { 5 -> acc; 5 -> i; -> loop; }
            -false-> acc -> ret;
        }
    }
}
fn main() {}
"#;
    assert_rejects(src, "L1503");
}

#[test]
fn l1504_nested_loop_updates_outer() {
    // an inner loop assigning a mut also carried by the enclosing loop (`acc` is
    // assigned in the outer body AND the inner body).
    let src = r#"fn f(n: i32) -> i32 {
    mut i: i32 <- 0;
    mut acc: i32 <- 0;
    loop {
        acc + i -> acc;
        (i < n) -> {
            -true-> {
                loop {
                    (acc < 10) -> {
                        -true-> { acc + 1 -> acc; -> loop; }
                        -false-> acc -> ret;
                    }
                }
                i + 1 -> i;
                -> loop;
            }
            -false-> acc -> ret;
        }
    }
}
fn main() {}
"#;
    assert_rejects(src, "L1504");
}

#[test]
fn l1504_inner_exit_not_ret() {
    // a nested loop whose inner exit does not terminate in `-> ret`.
    let src = r#"fn f(n: i32) -> i32 {
    mut i: i32 <- 0;
    mut total: i32 <- 0;
    loop {
        (i < n) -> {
            -true-> {
                mut j: i32 <- 0;
                loop {
                    (j < 3) -> {
                        -true-> { j + 1 -> j; -> loop; }
                        -false-> j -> total;
                    }
                }
                i + 1 -> i;
                -> loop;
            }
            -false-> total -> ret;
        }
    }
}
fn main() {}
"#;
    assert_rejects(src, "L1504");
}

#[test]
fn l1504_loop_in_effectful_loop() {
    // a nested loop inside a token-carrying loop.
    let src = r#"fn f(n: i32) {
    mut i: i32 <- 0;
    mut j: i32 <- 0;
    loop {
        i -> print;
        (i < n) -> {
            -true-> {
                loop {
                    (j < 3) -> {
                        -true-> { j + 1 -> j; -> loop; }
                        -false-> j -> ret;
                    }
                }
                i + 1 -> i;
                -> loop;
            }
            -false-> -> ret;
        }
    }
}
fn main() {}
"#;
    assert_rejects(src, "L1504");
}

// --- L1601–L1605: map / fold -------------------------------------------------

#[test]
fn l1601_block_arity() {
    // a map block with two params.
    assert_rejects(
        "fn main() { [1, 2] -> map { a, b -> a } -> ys: [i32; 2]; }\n",
        "L1601",
    );
}

#[test]
fn l1602_map_non_array() {
    // map applied to a non-array wire.
    assert_rejects("fn main() { 5 -> map { x -> x + 1 } -> ys; }\n", "L1602");
}

#[test]
fn l1603_fold_shape() {
    // fold wire not a 2-tuple `(init, array)`.
    assert_rejects(
        "fn main() { 5 -> fold { acc, x -> acc + x } -> r; }\n",
        "L1603",
    );
}

#[test]
fn l1604_body_no_value() {
    // a map body with no tail value.
    assert_rejects(
        "fn main() { [1, 2] -> map { x -> x + 1 -> y; } -> ys: [i32; 2]; }\n",
        "L1604",
    );
}

#[test]
fn l1605_body_effectful() {
    // a print inside a map body.
    assert_rejects(
        "fn main() { [1, 2] -> map { x -> x -> print; x } -> ys: [i32; 2]; }\n",
        "L1605",
    );
}

#[test]
fn l1605_body_time() {
    // plan-time-builtin: `time` is an effect site exactly like `print`, so a
    // clock read inside a map body is the same L1605 (bodies are token-free).
    assert_rejects(
        "fn main() { [1, 2] -> map { x -> () -> time -> t; x } -> ys: [i32; 2]; }\n",
        "L1605",
    );
}

// --- L1606–L1610: zip / enumerate (ADR-0018) --------------------------------

#[test]
fn l1606_zip_non_tuple_scalar() {
    // zip source is a scalar, not a 2-tuple.
    assert_rejects("fn main() { 5 -> zip -> x; }\n", "L1606");
}

#[test]
fn l1606_zip_non_tuple_arity3() {
    // zip source is a 3-tuple, not a 2-tuple.
    assert_rejects("fn main() { (1, 2, 3) -> zip -> x; }\n", "L1606");
}

#[test]
fn l1607_zip_component_not_array() {
    // component 0 of the 2-tuple is a scalar, not an array.
    assert_rejects(
        "fn main() { [1, 2, 3] -> a: [i32; 3]; (5, a) -> zip -> x; }\n",
        "L1607",
    );
}

#[test]
fn l1608_zip_size_mismatch() {
    // the two arrays differ in length.
    assert_rejects(
        "fn main() { [1, 2] -> a: [i32; 2]; [3, 4, 5] -> b: [i32; 3]; (a, b) -> zip -> x; }\n",
        "L1608",
    );
}

#[test]
fn l1609_enumerate_non_array() {
    // enumerate applied to a scalar wire.
    assert_rejects("fn main() { 5 -> enumerate -> x; }\n", "L1609");
}

#[test]
fn l1610_enumerate_oversize() {
    // an array whose length exceeds i32::MAX cannot be enumerated (the index
    // `i32` could not name every element).
    assert_rejects(
        "fn f(a: [i32; 2147483648]) -> [(i32, i32); 2147483648] { a -> enumerate -> ret; }\nfn main() {}\n",
        "L1610",
    );
}
