//! bound-blkdec: `run_bound`, the composed bound of the default face (ADR-0041
//! D5 `bound(o)`) — each operation's own deviation from the IR facts,
//! composed through the program's derivatives: a member's known roundings
//! per `(product, slot)`, a regrouped fold's order, a narrow rounding that
//! absorbs a deviation short of a tie, a Newton step's contraction.

use mapal_interp::{Bound, Outcome, run_bound, run_weakest};

fn ir(src: &str) -> mapal_ir::CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "{src}");
    mapal_lower::lower(src, &po.program).expect("lowers")
}

fn bounded(src: &str) -> (Vec<f64>, Vec<Bound>) {
    let (r, b) = run_bound(&ir(src), 100_000_000);
    assert!(matches!(r.outcome, Outcome::Done(_)), "{src}");
    let v = r
        .output
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();
    (v, b)
}

fn gamma(n: f64, u: f64) -> f64 {
    n * u / (1.0 - n * u)
}

fn bf16(x: f64) -> f64 {
    let n = mapal_ir::NarrowFloat::Bf16;
    f64::from(mapal_ir::narrow::widen(n, mapal_ir::narrow::round(n, x)))
}

const DOT: &str = "fn main() {
    16 -> iota -> map {
        t -> t -> widen<f32> -> v;
        v / 3.0 + 0.1
    } -> x;
    16 -> iota -> map {
        t -> t % 5 - 2 -> widen<f32> -> v;
        v / 8.0 -> to<bf16>
    } -> w;
    16 -> iota -> rk;
    (0.0, rk) -> fold: bf16 | f32 {
        acc, k -> w[k] -> widen<f32> -> wv;
        acc + x[k] * wv
    } -> y;
    y -> println;
}";

/// A dot product of f32 `x` (off the bf16 grid) and bf16 `w` under
/// `: bf16 | f32`: `x`'s known roundings as one `(product, slot)` — their
/// signed sum `|Σ δ_k·w_k|` — plus the free order on a tensor core's
/// accumulator, `2·γ_17(2⁻²³)·Σ|x_k·w_k|`. The weakest oracle is inside.
#[test]
fn member_roundings_and_order_on_a_dot() {
    let (v, b) = bounded(DOT);
    let (mut e, mut m) = (0.0, 0.0);
    for t in 0..16 {
        let x = f64::from(t as f32 / 3.0 + 0.1);
        let w = f64::from((t % 5 - 2) as f32 / 8.0);
        e += (bf16(x) - x) * w;
        m += (x * w).abs();
    }
    let order = 2.0 * gamma(17.0, 2f64.powi(-23)) * m;
    assert!((b[0].member - e.abs()).abs() <= 1e-12, "{:?} vs {e}", b[0]);
    assert!((b[0].order - order).abs() <= 1e-12, "{:?} vs {order}", b[0]);
    assert!(
        (b[0].total - (e.abs() + order)).abs() <= 1e-12,
        "{:?}",
        b[0]
    );
    assert_eq!(b[0].flip, 0.0);
    let weak: f64 = run_weakest(&ir(DOT), 100_000_000, &[])
        .output
        .trim()
        .parse()
        .unwrap();
    assert!((weak - v[0]).abs() <= b[0].total, "{weak} vs {}", v[0]);
}

/// The same dot without `: bf16 | f32`: no member rounds, the order is the
/// f32 accumulator's; an exact value rounded to bf16 deviates nowhere.
#[test]
fn f32_products_and_exact_roundings() {
    let src = DOT.replace("fold: bf16 | f32", "fold");
    let (_, b) = bounded(&src);
    assert_eq!(b[0].member, 0.0);
    assert!(b[0].order > 0.0 && b[0].order < 1e-5, "{:?}", b[0]);
    let exact = "fn main() {
    2.5 -> x;
    x / 3.0 -> to<bf16> -> y;
    y -> widen<f32> -> println;
}";
    let (_, b) = bounded(exact);
    assert_eq!(b[0].total, 0.0, "{:?}", b[0]);
}

/// A narrow rounding of a value that carries a deviation: short of the near
/// tie it gives the exact face's bits (bound 0), else one grid step at most.
#[test]
fn narrow_rounding_absorbs_short_of_a_tie() {
    let src = DOT.replace(
        "    y -> println;",
        "    y -> println;\n    y -> to<bf16> -> z;\n    z -> widen<f32> -> println;",
    );
    let (v, b) = bounded(&src);
    let (y, e) = (v[0], b[0].total);
    let r = bf16(y);
    let ulp = 2f64.powi(y.abs().log2().floor() as i32 - 7);
    let near = (ulp / 2.0 - (r - y).abs()).max(0.0);
    if e < near {
        assert_eq!(b[1].total, 0.0, "{:?} (e {e}, near {near})", b[1]);
    } else {
        assert!(b[1].total <= ulp + e, "{:?} (e {e}, ulp {ulp})", b[1]);
    }
}

/// Newton's square root under `: bf16 | f32` (each step's `t` may round):
/// the steps contract, so the bound is the last rounding's — about
/// `u_t·|sd|` — not 16 of them, as a per-operation composition would count.
#[test]
fn a_newton_fold_contracts() {
    let src = "fn main() {
    2.7 -> ra;
    16 -> iota -> rn;
    (1.0, rn) -> fold: bf16 | f32 {
        acc, k -> 0.5 * (acc + ra / acc)
    } -> sd;
    sd -> println;
}";
    let (v, b) = bounded(src);
    let ut = 2f64.powi(-8);
    assert!((v[0] - 2.7f64.sqrt()).abs() < 1e-6, "{v:?}");
    assert!(
        b[0].total > 0.0 && b[0].total <= 2.0 * ut * v[0],
        "{:?}",
        b[0]
    );
}
