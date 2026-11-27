//! mask-frontend: forms a readable program writes that the compiler had to
//! be written around — a struct literal written straight to a return slot,
//! a struct captured by a map body and read by its fields. Each lowers to
//! the graph its hand-narrowed twin does, and keeps the oracle's values
//! through every rewrite.

use mapal_ir::{CategoryIr, ObjectKind, Operation, Ty};

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "{:?}\n{src}", po.diagnostics);
    mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("{d:?}\n{src}"))
}

/// The compiler's own pipeline: the compile-time evaluation, then the
/// rewrites (`mapal build`'s `rewritten`).
fn rewritten(ir: CategoryIr) -> CategoryIr {
    let (ir, _) = mapal_rewrite::fold_static(ir);
    mapal_rewrite::rewrite(ir).ir
}

fn writerless(ir: &CategoryIr) -> usize {
    ir.objects()
        .filter(|(o, x)| x.kind == ObjectKind::Temporary && ir.in_edges(*o).is_empty())
        .count()
}

/// Every map / fold capture (a source product's slot before the mapped
/// array) whose type is a struct.
fn struct_captures(ir: &CategoryIr) -> usize {
    ir.morphisms()
        .filter_map(|(_, m)| match m.op {
            Operation::Map { captures, .. } | Operation::Fold { captures, .. } if captures > 0 => {
                Some((m.source, captures))
            }
            _ => None,
        })
        .map(|(src, k)| {
            ir.in_edges(src)
                .iter()
                .filter(|&&e| {
                    let x = ir.morphism(e).expect("edge");
                    matches!(x.op, Operation::Pair { slot, .. } if slot < k)
                        && matches!(ir.object(x.source).expect("slot").ty, Ty::Struct { .. })
                })
                .count()
        })
        .sum()
}

/// A layer loop over records calling a step that writes its new state as a
/// struct literal straight to `ret.1` — the builder packs the slot's own
/// component (it minted a second, writerless one: replay panicked on it,
/// `reconstruct_temp`). No temporary without a writer, before or after the
/// rewrites; the values the oracle's, `--static` on or off.
#[test]
fn a_struct_literal_written_to_a_ret_slot() {
    let src = "type Weights {\n    w: [f32; 4],\n}\n\ntype State {\n    a: [f32; 4],\n    b: [f32; 4],\n}\n\n\
fn step(x: [f32; 4], weights: Weights, s: State) -> ([f32; 4], State) {\n    4 -> iota -> map {\n        i -> x[i] * weights.w[i] + s.a[i]\n    } -> y;\n    4 -> iota -> map {\n        i -> s.b[i] + y[i]\n    } -> z;\n    y -> ret.0;\n    State {\n        a: y,\n        b: z,\n    } -> ret.1;\n}\n\n\
fn main() {\n    4 -> iota -> map {\n        i -> i -> widen<f32>\n    } -> x0;\n    3 -> iota -> map {\n        l -> Weights {\n            w: x0,\n        }\n    } -> layers;\n    State {\n        a: x0,\n        b: x0,\n    } -> s0;\n    mut x: [f32; 4] <- x0;\n    mut s: State <- s0;\n    mut l: i32 <- 0;\n    loop {\n        (l < 3) -> {\n            -true-> {\n                (x, layers[l], s) -> step -> r;\n                r.0 -> x;\n                r.1 -> s;\n                l + 1 -> l;\n                -> loop;\n            }\n            -false-> (s, x) -> out;\n        }\n    }\n    out.0.b[3] -> println;\n    out.1[2] -> println;\n}\n";
    let ir = lower(src);
    assert_eq!(writerless(&ir), 0);
    let want = mapal_interp::run(&ir, 10_000_000).output;
    assert_eq!(want, "255\n54\n");
    let rw = rewritten(ir);
    assert_eq!(writerless(&rw), 0);
    assert_eq!(mapal_interp::run(&rw, 10_000_000).output, want);
    let rw = mapal_rewrite::rewrite(lower(src)).ir;
    assert_eq!(mapal_interp::run(&rw, 10_000_000).output, want);
}

/// The attention of a readable model reading its cache through the struct
/// (`cache.keys[…]` in a map body and in the fold nested in its guard arm):
/// the captured struct is its fields — no map or fold captures a struct —
/// and after the rewrites the program has the tile sites and stream groups
/// of its twin passing the arrays, with the same values.
#[test]
fn a_captured_struct_is_its_fields() {
    let program = |via: bool| {
        let (keys, values) = match via {
            true => ("cache.keys", "cache.values"),
            false => ("keys", "values"),
        };
        format!(
            "type Cache {{\n    keys: [f32; 256],\n    values: [f32; 256],\n}}\n\n\
fn main() {{\n    512 -> iota -> map {{\n        i -> (i * 7 + 3) % 11 - 5 -> widen<f32> -> v;\n        v / 4.0\n    }} -> query;\n    256 -> iota -> map {{\n        i -> (i * 5 + 1) % 9 - 4 -> widen<f32> -> v;\n        v / 4.0\n    }} -> keys;\n    256 -> iota -> map {{\n        i -> (i * 3 + 2) % 13 - 6 -> widen<f32> -> v;\n        v / 4.0\n    }} -> values;\n    Cache {{\n        keys: keys,\n        values: values,\n    }} -> cache;\n    16 -> iota -> dims;\n    16 -> iota -> past;\n    512 -> iota -> map {{\n        e -> e / 16 -> row;\n        e % 16 -> r;\n        row / 2 -> t;\n        (r <= t) -> {{\n            -true-> {{\n                (0.0, dims) -> fold {{\n                    acc, d -> acc + query[row * 16 + d] * {keys}[r * 16 + d]\n                }} -> dot;\n                dot / 4.0\n            }}\n            -false-> -1000000000.0;\n        }}\n    }} -> scores;\n    32 -> iota -> map {{\n        row -> (-1000000000.0, past) -> fold {{\n            acc, r -> scores[row * 16 + r] -> x;\n            (x > acc) -> {{\n                -true-> x;\n                -false-> acc;\n            }}\n        }}\n    }} -> top;\n    512 -> iota -> map {{\n        e -> scores[e] - top[e / 16] -> exp\n    }} -> p;\n    32 -> iota -> map {{\n        row -> (0.0, past) -> fold {{\n            acc, r -> acc + p[row * 16 + r]\n        }}\n    }} -> total;\n    512 -> iota -> map {{\n        e -> e / 16 -> row;\n        e % 16 -> u;\n        (0.0, past) -> fold {{\n            acc, r -> acc + p[row * 16 + r] * {values}[r * 16 + u]\n        }} -> o;\n        o / total[row]\n    }} -> out;\n    out[0] -> println;\n    out[511] -> println;\n    (0.0, out) -> fold {{\n        acc, v -> acc + v\n    }} -> sum;\n    sum -> println;\n}}\n"
        )
    };
    let (via, plain) = (lower(&program(true)), lower(&program(false)));
    assert_eq!(struct_captures(&via), 0);
    let want = mapal_interp::run(&plain, 100_000_000).output;
    assert_eq!(mapal_interp::run(&via, 100_000_000).output, want);
    let (via, plain) = (rewritten(via), rewritten(plain));
    assert_eq!(mapal_interp::run(&via, 100_000_000).output, want);
    let (fv, fp) = (via.entry(), plain.entry());
    let sites = |ir: &CategoryIr, f| {
        (
            ir.tile_plan_strided(f).sites.len(),
            ir.tile_plan_post(f).sites.len(),
            ir.tile_plan_radix(f).len(),
            ir.unit_plan(f).groups.len(),
            ir.stream_plan(f).groups.len(),
        )
    };
    assert_eq!(sites(&via, fv), sites(&plain, fp));
    assert_eq!(ir_stream(&via), 1);
}

fn ir_stream(ir: &CategoryIr) -> usize {
    ir.stream_plan(ir.entry()).groups.len()
}
