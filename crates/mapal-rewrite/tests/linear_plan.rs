//! plan-cost `linear_plan` (design §4a, D6): a sum of folds over one `k` range,
//! each read only by the sum, is one linear site; anything else is not.

mod testgen;

use mapal_ir::{CategoryIr, bound_coef};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "{:?}\n{src}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("{d:?}\n{src}"));
    mapal_rewrite::rewrite(ir).ir
}

/// `n × n` operands `a`, `b`, `e`, `f`, `g`, then `body`, then a print of `out`.
fn prog(body: &str) -> String {
    let mut s = String::from(
        "fn main() {\n    16 -> iota -> ta;\n    4 -> iota -> kr;\n    2 -> iota -> k2;\n",
    );
    for (x, m) in [("a", 7), ("b", 11), ("e", 5), ("f", 3), ("g", 13)] {
        s += &format!(
            "    ta -> map {{ t -> (t * {m} + 1) % 17 - 8 -> widen<f32> -> v; v / 4.0 }} -> {x};\n"
        );
    }
    s + body + "    out[0] -> println;\n    out[15] -> println;\n}\n"
}

/// A 4 × 4 tile site `name = x · y` over `range` from `seed`.
fn mm(name: &str, x: &str, y: &str, range: &str, seed: &str) -> String {
    format!(
        "    ta -> map {{ t -> t / 4 -> i; t % 4 -> j; ({seed}, {range}) -> fold {{ acc, k -> acc + {x}[i * 4 + k] * {y}[k * 4 + j] }} }} -> {name};\n"
    )
}

/// The linear sites of `main`: `(terms, shared a, shared b)` each.
fn sites(src: &str) -> Vec<(usize, bool, bool)> {
    let ir = lower(src);
    let f = ir.entry();
    assert_eq!(ir.linear_plan(f), ir.linear_plan(f), "stored once");
    ir.linear_plan(f)
        .values()
        .map(|s| {
            let n = s.terms.len() as u64;
            assert_eq!(s.bound, bound_coef(4 + n - 1, false));
            let (a, b) = (s.operands[0].0, s.operands[0].1);
            (
                s.terms.len(),
                s.operands.iter().all(|o| o.0 == a),
                s.operands.iter().all(|o| o.1 == b),
            )
        })
        .collect()
}

#[test]
fn sums_of_folds_over_one_range_are_linear() {
    // indep: two independent terms.
    let indep = format!(
        "{}{}    ta -> map {{ t -> c[t] + d[t] }} -> out;\n",
        mm("c", "a", "b", "kr", "0.0"),
        mm("d", "e", "f", "kr", "0.0")
    );
    assert_eq!(sites(&prog(&indep)), [(2, false, false)]);
    // diamond: both terms read `c` as `a` (the distributive form).
    let diamond = format!(
        "{}{}{}    ta -> map {{ t -> d1[t] + d2[t] }} -> out;\n",
        mm("c", "a", "b", "kr", "0.0"),
        mm("d1", "c", "e", "kr", "0.0"),
        mm("d2", "c", "g", "kr", "0.0")
    );
    assert_eq!(sites(&prog(&diamond)), [(2, true, false)]);
    // T = 3.
    let three = format!(
        "{}{}{}    ta -> map {{ t -> c[t] + d[t] + h[t] }} -> out;\n",
        mm("c", "a", "b", "kr", "0.0"),
        mm("d", "e", "f", "kr", "0.0"),
        mm("h", "g", "b", "kr", "0.0")
    );
    assert_eq!(sites(&prog(&three)), [(3, false, false)]);
}

#[test]
fn other_sums_are_not_linear() {
    let two = |c: &str, d: &str, tail: &str| {
        format!("{c}{d}    ta -> map {{ t -> c[t] + d[t] }} -> out;\n{tail}")
    };
    let cases = [
        // A nonzero seed.
        two(
            &mm("c", "a", "b", "kr", "1.0"),
            &mm("d", "e", "f", "kr", "0.0"),
            "",
        ),
        // A term read elsewhere.
        two(
            &mm("c", "a", "b", "kr", "0.0"),
            &mm("d", "e", "f", "kr", "0.0"),
            "    c[3] -> println;\n",
        ),
        // Unequal `k` range.
        two(
            &mm("c", "a", "b", "kr", "0.0"),
            &mm("d", "e", "f", "k2", "0.0"),
            "",
        ),
        // A difference, a scaled term, a term off its cell.
        format!(
            "{}{}    ta -> map {{ t -> c[t] - d[t] }} -> out;\n",
            mm("c", "a", "b", "kr", "0.0"),
            mm("d", "e", "f", "kr", "0.0")
        ),
        format!(
            "{}{}    ta -> map {{ t -> c[t] + d[t] * 2.0 }} -> out;\n",
            mm("c", "a", "b", "kr", "0.0"),
            mm("d", "e", "f", "kr", "0.0")
        ),
        format!(
            "{}{}    ta -> map {{ t -> c[t] + d[(t + 1) % 16] }} -> out;\n",
            mm("c", "a", "b", "kr", "0.0"),
            mm("d", "e", "f", "kr", "0.0")
        ),
        // A sum of a fold and a plain array.
        format!(
            "{}    ta -> map {{ t -> c[t] + e[t] }} -> out;\n",
            mm("c", "a", "b", "kr", "0.0")
        ),
    ];
    for body in cases {
        assert_eq!(sites(&prog(&body)), [], "{body}");
    }
    // An escaping term: `c` is the function's output.
    let esc = "fn two(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {\n    16 -> iota -> ta;\n    4 -> iota -> kr;\n".to_string()
        + &mm("c", "a", "b", "kr", "0.0")
        + &mm("d", "b", "a", "kr", "0.0")
        + "    ta -> map { t -> c[t] + d[t] } -> out;\n    out[0] -> println;\n    c\n}\n"
        + &prog("    (a, b) -> two -> out;\n");
    let ir = lower(&esc);
    assert!(
        ir.funcs().all(|(f, _)| ir.linear_plan(f).is_empty()),
        "{esc}"
    );
    // ...and the same function returning the sum is linear.
    let ok = esc.replace("    c\n}", "    out\n}");
    let ir = lower(&ok);
    assert_eq!(
        ir.funcs()
            .map(|(f, _)| ir.linear_plan(f).len())
            .sum::<usize>(),
        1,
        "{ok}"
    );
}

/// Unequal `(rows, c)`: an 8-cell term beside a 16-cell sum.
#[test]
fn unequal_extents_are_not_linear() {
    let body = format!(
        "{}    8 -> iota -> t8;\n    t8 -> map {{ t -> t / 4 -> i; t % 4 -> j; (0.0, kr) -> fold {{ acc, k -> acc + e[i * 4 + k] * f[k * 4 + j] }} }} -> d;\n    ta -> map {{ t -> c[t] + d[t % 8] }} -> out;\n",
        mm("c", "a", "b", "kr", "0.0")
    );
    assert_eq!(sites(&prog(&body)), []);
}

/// The generator's sum node reaches `linear_plan` (coverage for the
/// emulator's value tests, `crates/backends/llvm/tests/generic_fusion.rs`).
#[test]
fn chains_hit_linear_sites() {
    let mut runner = TestRunner::deterministic();
    let strat = testgen::chain_strategy();
    let mut hit = 0;
    for _ in 0..256 {
        let ch = strat.new_tree(&mut runner).unwrap().current();
        let ir = lower(&ch.source());
        hit += ir.linear_plan(ir.entry()).len();
    }
    assert!(hit >= 8, "linear sites in 256 chains: {hit}");
}
