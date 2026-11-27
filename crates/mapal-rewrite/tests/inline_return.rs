//! Inline: a callee's Return written straight by a call the plan also
//! inlines (`(x, 2.0) -> scaled -> ret;` inside an inlined `doubled`) — the
//! inner callee is inlined at the Return too, never emitted as a `Call` of a
//! function `live_functions` dropped (was: `invalid SecondaryMap key` in
//! `replay::inline_return`). The readable model's `new_value` has this shape.

fn lower(src: &str) -> mapal_ir::CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"))
}

const SRC: &str = r#"
fn scaled(x: [f32; 64], factor: f32) -> [f32; 64] {
    64 -> iota -> map {
        i -> x[i] * factor
    } -> ret;
}

fn doubled(x: [f32; 64]) -> [f32; 64] {
    (x, 2.0) -> scaled -> ret;
}

fn bumped(x: [f32; 64]) -> [f32; 64] {
    x -> doubled -> ret;
}

fn main() {
    64 -> iota -> map {
        i -> i -> widen<f32>
    } -> x;
    x -> doubled -> y;
    y -> bumped -> z;
    z[63] -> println;
}
"#;

#[test]
fn inline_return_written_by_an_inlined_call() {
    let want = mapal_interp::run(&lower(SRC), 1_000_000).output;
    assert_eq!(want, "252\n");
    let res = mapal_rewrite::rewrite(lower(SRC));
    assert!(mapal_ir::validate(&res.ir).is_empty());
    // Every call is inlined: no `Call` survives.
    assert!(
        res.ir
            .morphisms()
            .all(|(_, m)| !matches!(m.op, mapal_ir::Operation::Call(_)))
    );
    assert_eq!(mapal_interp::run(&res.ir, 1_000_000).output, want);
}
