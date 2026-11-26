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
            i -> (l * {scale} + i) % 7 - 3 -> widen<f32>
        }} -> w;
        l -> widen<f32> -> b;
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
        i -> i -> widen<f32>
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
        i -> i -> widen<f32>
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
/// A prompt pass and decode steps over the same per-layer records, the way
/// a model is written: the records read whole from `file`, one size-generic
/// `step<rows>` called once over a prompt of `rows` rows before the loop and
/// once a trip over one row — two instances, each with its own unrolled
/// layer loop and its own constant indices `l` into `stack`.
fn prompt_program(file: &str, layers: u32, steps: u32, width: u32, rows: u32) -> String {
    format!(
        r#"
{layers} -> layer_count;
{width} -> width;

type Layer {{ weight: [f32; width], bias: f32 }}

fn step<rows>(x: [f32; rows * width], stack: [Layer; layer_count]) -> [f32; rows * width] {{
    mut h: [f32; rows * width] <- x;
    mut layer: i32 <- 0;
    loop {{
        (layer < layer_count) -> {{
            -true-> {{
                stack[layer] -> record;
                rows * width -> iota -> map {{
                    i -> h[i] * record.weight[i % width] + record.bias
                }} -> h;
                layer + 1 -> layer;
                -> loop;
            }}
            -false-> h -> ret;
        }}
    }}
}}

fn main() {{
    "{file}" -> read -> stack: [Layer; layer_count];
    {rows} * width -> iota -> map {{
        i -> i % 5 -> widen<f32>
    }} -> prompt;
    (prompt, stack) -> step -> prefilled: [f32; {rows} * width];
    prefilled[0] -> println;
    width -> iota -> map {{
        i -> prefilled[i] / 8.0
    }} -> x0;
    mut x: [f32; width] <- x0;
    mut s: i32 <- 0;
    loop {{
        (s < {steps}) -> {{
            -true-> {{
                (x, stack) -> step -> x;
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

/// The views of one array at constant indices, by (array, index value),
/// that more than one object holds — the duplicates the value scope of CSE
/// merges (`model.layers[l]` read by a prompt pass and by the decode steps).
fn duplicate_views(ir: &CategoryIr) -> usize {
    let mut seen = std::collections::BTreeMap::<String, u32>::new();
    for (_, m) in ir.morphisms() {
        if m.op != Operation::Index {
            continue;
        }
        let mut slots: Vec<(u32, mapal_ir::ObjectId)> = ir
            .in_edges(m.source)
            .iter()
            .filter_map(|&e| {
                let x = ir.morphism(e)?;
                match x.op {
                    Operation::Pair { slot, .. } => Some((slot, x.source)),
                    _ => None,
                }
            })
            .collect();
        slots.sort();
        let [(_, arr), (_, at)] = slots[..] else {
            continue;
        };
        let Some(v) = ir.object(at).and_then(|o| o.value.clone()) else {
            continue;
        };
        *seen.entry(format!("{arr:?}|{v:?}")).or_default() += 1;
    }
    seen.values().filter(|&&n| n > 1).count()
}

/// prefill-residency: a prompt pass and the decode steps over the same
/// per-layer records read each record once — the value scope of CSE keys a
/// constant index by its value outside guard arms, so `stack[l]` is one
/// object (one host view, one device copy) — with the oracle's values.
#[test]
fn prompt_and_steps_share_their_record_views() {
    let mut runner = TestRunner::new(Config {
        cases: 32,
        failure_persistence: None,
        ..Config::default()
    });
    use std::cell::Cell;
    let shared = Cell::new(0u32);
    let dir = tempfile::tempdir().unwrap();
    let strategy = (1u32..=6, 1u32..=3, 1u32..=9, 2u32..=5, 1u32..=11);
    runner
        .run(&strategy, |(layers, steps, width, rows, scale)| {
            // Each layer's record: `width` weights, then the bias (f32 LE).
            let bytes: Vec<u8> = (0..layers)
                .flat_map(|l| {
                    (0..=width).map(move |i| match i {
                        i if i == width => l as f32,
                        i => ((l * scale + i) % 7) as f32 - 3.0,
                    })
                })
                .flat_map(f32::to_le_bytes)
                .collect();
            let file = dir
                .path()
                .join(format!("stack_{layers}_{width}_{scale}.bin"));
            std::fs::write(&file, bytes).unwrap();
            let src = prompt_program(&file.display().to_string(), layers, steps, width, rows);
            let ir = lower(&src);
            let want = mapal_interp::run(&ir, 50_000_000).output;
            let res = mapal_rewrite::rewrite(ir);
            prop_assert!(mapal_ir::validate(&res.ir).is_empty());
            prop_assert_eq!(mapal_interp::run(&res.ir, 50_000_000).output, want);
            prop_assert_eq!(calls(&res.ir), 0);
            prop_assert_eq!(duplicate_views(&res.ir), 0, "{}", src);
            shared.set(shared.get() + 1);
            Ok(())
        })
        .unwrap();
    assert!(shared.get() > 0);
}

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
