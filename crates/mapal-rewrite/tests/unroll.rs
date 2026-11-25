//! R-UN (`mapal_rewrite::unroll`): a counted loop whose trips are known before
//! the run is replayed trip by trip where a trip takes a record of an
//! invariant array of records at the counter (`model.layers[layer]`). Then the
//! function has no loop, and the inliner takes it (a single call site whatever
//! its size) — also from inside another loop (a decode step's layer loop). A
//! loop over a stacked array stays a loop. The values are the oracle's on
//! every program.

use mapal_ir::{CategoryIr, Operation};
use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(
        po.diagnostics.is_empty(),
        "parse: {:?}\n{src}",
        po.diagnostics
    );
    mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}\n{src}"))
}

fn loops(ir: &CategoryIr) -> usize {
    ir.funcs().map(|(f, _)| ir.loop_structure(f).len()).sum()
}

fn calls(ir: &CategoryIr) -> usize {
    ir.morphisms()
        .filter(|(_, m)| matches!(m.op, Operation::Call(_)))
        .count()
}

/// A decode-like program: `steps` trips of an outer loop, each calling a
/// function whose loop runs `layers` trips over per-layer structs (`view`:
/// read as `stack[layer]`; else through a stacked array at `layer · width`).
fn program(layers: u32, steps: u32, width: u32, view: bool, scale: u32) -> String {
    let read = if view {
        "stack[layer].weight[i]"
    } else {
        "flat[layer * width + i]"
    };
    format!(
        r#"
{layers} -> layer_count;
{width} -> width;

type Layer {{ weight: [f32; width], bias: f32 }}

fn step(x: [f32; width], stack: [Layer; layer_count], flat: [f32; layer_count * width]) -> [f32; width] {{
    mut h: [f32; width] <- x;
    mut layer: i32 <- 0;
    loop {{
        (layer < layer_count) -> {{
            -true-> {{
                width -> iota -> map {{
                    i -> h[i] * {read} + stack[layer].bias
                }} -> h;
                layer + 1 -> layer;
                -> loop;
            }}
            -false-> h -> ret;
        }}
    }}
}}

fn main() {{
    layer_count -> iota -> map {{
        l ->
        width -> iota -> map {{
            i -> (l * {scale} + i) % 7 - 3 -> widen_f32
        }} -> w;
        l -> widen_f32 -> b;
        Layer {{
            weight: w,
            bias: b,
        }} -> made;
        made
    }} -> stack;
    layer_count * width -> iota -> map {{
        c -> stack[c / width].weight[c % width]
    }} -> flat;
    width -> iota -> map {{
        i -> i -> widen_f32
    }} -> x0;
    mut x: [f32; width] <- x0;
    mut s: i32 <- 0;
    loop {{
        (s < {steps}) -> {{
            -true-> {{
                (x, stack, flat) -> step -> x;
                x[0] -> println;
                s + 1 -> s;
                -> loop;
            }}
            -false-> s -> done;
        }}
    }}
}}
"#
    )
}

#[test]
fn a_layer_loop_called_per_step_unrolls_and_inlines() {
    let ir = lower(&program(3, 2, 4, true, 5));
    let want = mapal_interp::run(&ir, 10_000_000).output;
    assert_eq!(loops(&ir), 2);
    let res = mapal_rewrite::rewrite(ir);
    assert!(mapal_ir::validate(&res.ir).is_empty());
    let unrolled = res
        .report
        .applied
        .iter()
        .find(|(p, _)| *p == mapal_rewrite::PassId::Unroll)
        .map_or(0, |(_, n)| *n);
    assert_eq!(unrolled, 1, "{:?}", res.report);
    // The step's loop is gone, the step inlined: one loop (the steps) left.
    assert_eq!((loops(&res.ir), calls(&res.ir)), (1, 0));
    assert_eq!(mapal_interp::run(&res.ir, 10_000_000).output, want);
}

/// A counted loop whose trips take no record of an array of records stays a
/// loop.
#[test]
fn a_plain_counted_loop_stays() {
    let src = r#"
fn main() {
    8 -> iota -> map {
        i -> i -> widen_f32
    } -> x0;
    mut x: [f32; 8] <- x0;
    mut k: i32 <- 0;
    loop {
        (k < 4) -> {
            -true-> {
                8 -> iota -> map {
                    i -> x[i] * 0.5 + 1.0
                } -> x;
                k + 1 -> k;
                -> loop;
            }
            -false-> x -> out;
        }
    }
    out[3] -> println;
}
"#;
    let res = mapal_rewrite::rewrite(lower(src));
    assert!(
        res.report
            .applied
            .iter()
            .all(|(p, _)| *p != mapal_rewrite::PassId::Unroll)
    );
}

/// Generated programs: the layer count, the steps, the width, view or
/// stacked reads. Every program's rewritten graph prints the oracle's output
/// with no call left; the coverage counters are non-zero.
#[test]
fn generated_layer_loops_keep_values() {
    let mut runner = TestRunner::new(Config {
        cases: 48,
        failure_persistence: None,
        ..Config::default()
    });
    use std::cell::Cell;
    let (unrolled, views, stacked) = (Cell::new(0u32), Cell::new(0u32), Cell::new(0u32));
    let strategy = (1u32..=6, 1u32..=3, 1u32..=9, any::<bool>(), 1u32..=11);
    runner
        .run(&strategy, |(layers, steps, width, view, scale)| {
            let ir = lower(&program(layers, steps, width, view, scale));
            let want = mapal_interp::run(&ir, 50_000_000).output;
            let res = mapal_rewrite::rewrite(ir);
            prop_assert!(mapal_ir::validate(&res.ir).is_empty());
            prop_assert_eq!(mapal_interp::run(&res.ir, 50_000_000).output, want);
            // Record views unroll and the step inlines (stacked reads keep
            // the step's loop unless R-LF / R-LM lift it).
            if view {
                prop_assert_eq!(calls(&res.ir), 0);
            }
            if res
                .report
                .applied
                .iter()
                .any(|(p, _)| *p == mapal_rewrite::PassId::Unroll)
            {
                unrolled.set(unrolled.get() + 1);
            }
            let side = if view { &views } else { &stacked };
            side.set(side.get() + 1);
            Ok(())
        })
        .unwrap();
    let (unrolled, views, stacked) = (unrolled.get(), views.get(), stacked.get());
    eprintln!("unroll coverage: unrolled {unrolled}, views {views}, stacked {stacked}");
    assert!(unrolled > 0 && views > 0 && stacked > 0);
}
