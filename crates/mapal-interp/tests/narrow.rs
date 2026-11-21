//! ADR-0043 D2/D3: the oracle's narrow conversions — one rounding to nearest
//! even from the `f32` value (subnormals kept, `±inf` past the threshold),
//! exact widening — through bindings, arrays, updates, guards and loops.

use mapal_interp::{Outcome, run};

fn out(src: &str) -> String {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "{src}");
    let ir = mapal_lower::lower(src, &po.program).expect("lowers");
    let r = run(&ir, 100_000_000);
    assert!(matches!(r.outcome, Outcome::Done(_)), "{src}");
    r.output
}

#[test]
fn storage_types_and_conversions() {
    let src = "fn main() {
    8 -> iota -> r;
    r -> map { t -> t -> widen_f32 -> v; v / 3.0 -> to_bf16 } -> a;
    r -> map { t -> t -> widen_f32 -> v; v * 9000.5 -> to_f16 } -> h;
    mut s: [bf16; 8] <- a;
    0.25 -> to_bf16 -> q;
    s[2] <- q;
    (0.0, r) -> fold { acc, k -> s[k] -> widen_f32 -> x; h[k] -> widen_f64 -> y; acc + x * 2.0 } -> tot;
    tot -> println;
    h[7] -> widen_f64 -> println;
    a[1] -> widen_f32 -> println;
}";
    // bf16(1/3) = 0.333984375; f16(7 · 9000.5 = 63003.5): ulp 32, 63003.5 / 32 = 1968.86 → 1969 · 32 = 63008.
    let o = out(src);
    let lines: Vec<&str> = o.lines().collect();
    assert_eq!(lines[1], "63008");
    assert_eq!(lines[2], "0.33398438");
}

#[test]
fn rounding_edges_in_the_oracle() {
    // f16: overflow at the threshold, a subnormal, ties to even; bf16 keeps f32's range.
    let o = out("fn main() {
    65519.0 -> to_f16 -> widen_f32 -> println;
    65520.0 -> to_f16 -> widen_f32 -> println;
    0.0000001 -> to_f16 -> widen_f32 -> println;
    1.00390625 -> to_bf16 -> widen_f32 -> println;
    1.01171875 -> to_bf16 -> widen_f32 -> println;
    100000000000000000000000000000000000000.0 -> to_bf16 -> widen_f32 -> println;
}");
    assert_eq!(
        o,
        "65504\ninf\n0.00000011920929\n1\n1.015625\n99692100000000000000000000000000000000\n"
    );
}

#[test]
fn guards_loops_and_captures_move_narrow_values() {
    let o = out("fn main() {
    4 -> iota -> r;
    r -> map { t -> t -> widen_f32 -> v; v + 0.5 -> to_bf16 } -> a;
    1.75 -> to_bf16 -> c;
    r -> map { t -> (t > 1) -> { -true-> c; -false-> a[t]; } } -> g;
    mut st: [bf16; 4] <- g;
    mut i: i32 <- 0;
    loop {
        (i < 2) -> {
            -true-> {
                r -> map { t -> st[t] -> widen_f32 -> v; v * 2.0 -> to_bf16 } -> st;
                i + 1 -> i;
                -> loop;
            }
            -false-> st -> fin;
        }
    }
    fin[0] -> widen_f32 -> println;
    fin[3] -> widen_f32 -> println;
}");
    assert_eq!(o, "2\n7\n");
}
