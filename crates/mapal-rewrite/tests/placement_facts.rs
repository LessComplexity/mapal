//! model-ready `placement`: the facts and rewrites a decode step's device
//! placement reads — each on a small program, the program's value against the
//! interpreter on the raw graph.
//!
//! - `bounds_proof`: sibling views of one input slot share a condition's
//!   narrowing (a rotation's partner); a selecting fold's index component is
//!   ranged by its seed and items (an argmax token's embedding row).
//! - `map_law`'s `Outside` through a capture the site passes as a constant.
//! - `step_algebra`'s regroup reads: a captured array's stride along the fold.
//! - The rewriter: a struct-typed loop state is carried as its leaves; a
//!   view of an inlined call's argument tuple inside a loop is its value.

use mapal_interp::{Outcome, run};
use mapal_ir::{CategoryIr, ElemSrc, MorphismId, ObjectId, Operation, Ty};

fn parse_lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "{:?}\n{src}", po.diagnostics);
    mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("{d:?}\n{src}"))
}

fn lower(src: &str) -> CategoryIr {
    mapal_rewrite::rewrite(parse_lower(src)).ir
}

fn output(ir: &CategoryIr) -> String {
    let rr = run(ir, 200_000_000);
    assert!(matches!(rr.outcome, Outcome::Done(_)), "{:?}", rr.outcome);
    rr.output
}

fn named(ir: &CategoryIr, n: &str) -> ObjectId {
    ir.objects()
        .filter(|(_, o)| o.name.as_deref() == Some(n))
        .map(|(id, _)| id)
        .last()
        .unwrap_or_else(|| panic!("no {n}"))
}

fn site_of(ir: &CategoryIr, n: &str) -> MorphismId {
    ir.in_edges(named(ir, n))[0]
}

/// Every `Index` of `m`'s body (a `Map`) proven in bounds.
fn body_proven(ir: &CategoryIr, m: MorphismId) -> bool {
    let Some(Operation::Map { body, .. }) = ir.morphism(m).map(|x| x.op) else {
        panic!("not a map");
    };
    let bp = ir.bounds_proof(body);
    let idx: Vec<MorphismId> = ir.func(body).unwrap().morphisms.clone();
    idx.iter()
        .filter(|&&g| ir.morphism(g).unwrap().op == Operation::Index)
        .all(|&g| bp.proven(g))
}

const ROTATE: &str = "fn main() {
    16 -> iota -> map {
        i -> i % 5 -> widen_f32
    } -> row;
    16 -> iota -> map {
        i ->
        (i % 8 < 4) -> {
            -true-> i + 4;
            -false-> i - 4;
        } -> partner;
        row[i] - row[partner]
    } -> rotated;
    rotated[3] -> println;
    rotated[12] -> println;
}
";

/// `i` read by the condition and by the arm are two views of the body's
/// input: the condition's narrowing of one (`i % 8 < 4` ⇒ `i ≤ 11`) holds
/// for both, so `row[i ± 4]` is in bounds and the map is a cell site.
#[test]
fn sibling_views_share_a_conditions_narrowing() {
    let ir = lower(ROTATE);
    let m = site_of(&ir, "rotated");
    assert!(body_proven(&ir, m), "partner read not proven");
    assert!(ir.cell_plan(ir.entry()).contains_key(m), "no cell site");
    assert_eq!(output(&ir), output(&parse_lower(ROTATE)));
}

const ARGMAX: &str = "fn main() {
    12 -> iota -> map {
        i -> (i * 7 + 3) % 11 -> widen_f32
    } -> logits;
    48 -> iota -> map {
        i -> i % 9 -> widen_f32
    } -> table;
    12 -> iota -> tokens;
    ((logits[0], 0), tokens) -> fold {
        best, t ->
        (logits[t] > best.0) -> {
            -true-> (logits[t], t);
            -false-> best;
        }
    } -> winner;
    winner.1 -> token;
    4 -> iota -> map {
        i -> table[token * 4 + i]
    } -> row;
    row[0] -> println;
    row[3] -> println;
}
";

/// An argmax's index component is the seed's `0` or an item of `[0, 12)`:
/// the gather `table[token · 4 + i]` is in bounds (no trap), a cell site.
#[test]
fn a_selecting_folds_index_is_ranged_by_its_items() {
    let ir = lower(ARGMAX);
    let m = site_of(&ir, "row");
    assert!(body_proven(&ir, m), "gather not proven");
    assert!(ir.cell_plan(ir.entry()).contains_key(m), "no cell site");
    assert_eq!(output(&ir), output(&parse_lower(ARGMAX)));
}

const APPEND: &str =
    "fn append(cache: [f32; 64], row: [f32; 4], layer: i32, position: i32) -> [f32; 64] {
    64 -> iota -> map {
        cell ->
        (cell / 4 == layer * 8 + position) -> {
            -true-> row[cell % 4];
            -false-> cache[cell];
        }
    } -> ret;
}

fn main() {
    (0.0, 64) -> fill -> zeros;
    4 -> iota -> map {
        i -> i -> widen_f32
    } -> row;
    mut cache: [f32; 64] <- zeros;
    mut position: i32 <- 0;
    loop {
        (position < 3) -> {
            -true-> {
                (cache, row, 1, position) -> append -> written;
                written -> cache;
                position + 1 -> position;
                -> loop;
            }
            -false-> cache -> done;
        }
    }
    done[33] -> println;
    done[45] -> println;
}
";

/// The layer index the site passes as the literal `1` is that constant at
/// this site: the written row `4·(8 + position)` is affine in `position`
/// alone, an in-place range map.
#[test]
fn a_constant_capture_is_its_value_in_a_range_law() {
    let ir = lower(APPEND);
    let m = site_of(&ir, "written");
    let Some(ElemSrc::Apply {
        outside: Some(ou), ..
    }) = ir.map_law(m)
    else {
        panic!("no range law: {:?}", ir.map_law(m));
    };
    assert_eq!((ou.d, ou.a, ou.b, ou.len), (4, 1, 8, 4), "{ou:?}");
    assert!(ou.e.is_some(), "the position capture");
    assert_eq!(output(&ir), output(&parse_lower(APPEND)));
}

const PROJECT: &str = "fn main() {
    96 -> iota -> map {
        i -> i % 7 - 3 -> widen_f32
    } -> weights;
    32 -> iota -> map {
        i -> i % 3 -> widen_f32
    } -> x;
    32 -> iota -> columns;
    3 -> iota -> map {
        r -> (0.0, columns) -> fold {
            s, c -> s + weights[r * 32 + c] * x[c]
        }
    } -> y;
    y[0] -> println;
    y[2] -> println;
}
";

/// A row of `weights` and `x` are each read at unit stride along the fold
/// (`radix_plan`'s `k` digit of their addresses).
#[test]
fn a_captured_reads_stride_along_the_fold() {
    let ir = lower(PROJECT);
    let m = site_of(&ir, "y");
    let Some(Operation::Map { body, .. }) = ir.morphism(m).map(|x| x.op) else {
        panic!("not a map");
    };
    let fold_body = ir
        .func(body)
        .unwrap()
        .morphisms
        .iter()
        .find_map(|&g| match ir.morphism(g).unwrap().op {
            Operation::Fold { body, .. } => Some(body),
            _ => None,
        })
        .expect("nested fold");
    let r = ir.step_algebra(fold_body).regroup.expect("regroups");
    assert_eq!(r.reads, vec![Some(1), Some(1)], "{r:?}");
}

const STATE: &str = "type Pair2 {
    keys: [f32; 8],
    values: [f32; 8]
}

fn bump(p: Pair2, x: [f32; 8]) -> Pair2 {
    8 -> iota -> map {
        i -> p.keys[i] + x[i]
    } -> keys;
    8 -> iota -> map {
        i -> p.values[i] * 2.0
    } -> values;
    Pair2 {
        keys: keys,
        values: values
    } -> ret;
}

fn main() {
    8 -> iota -> map {
        i -> i -> widen_f32
    } -> x;
    Pair2 {
        keys: x,
        values: x
    } -> start;
    mut s: Pair2 <- start;
    mut n: i32 <- 0;
    loop {
        (n < 3) -> {
            -true-> {
                (s, x) -> bump -> s;
                n + 1 -> n;
                -> loop;
            }
            -false-> s -> done;
        }
    }
    done.keys[5] -> println;
    done.values[7] -> println;
}
";

/// A struct-typed state slot is carried as its leaves (no product in the
/// merge's slots); inside the loop no `Proj` reads a pack (an inlined
/// call's argument tuple, the struct's fields are their values).
#[test]
fn a_struct_state_is_carried_as_its_leaves() {
    let raw = parse_lower(STATE);
    let ir = lower(STATE);
    let f = ir.entry();
    let sccs = ir.loop_structure(f);
    assert!(!sccs.is_empty());
    for scc in &sccs {
        for &m in &scc.merges {
            let Some(Ty::Tuple(parts)) = ir.object(m).map(|x| &x.ty) else {
                continue;
            };
            assert!(
                !parts
                    .iter()
                    .any(|t| matches!(t, Ty::Tuple(_) | Ty::Struct { .. })),
                "{parts:?}"
            );
        }
        for &o in &scc.objects {
            let Some(&p) = ir.in_edges(o).first() else {
                continue;
            };
            let pm = ir.morphism(p).unwrap();
            if let Operation::Proj { .. } = pm.op {
                let packed = !ir.in_edges(pm.source).is_empty()
                    && ir
                        .in_edges(pm.source)
                        .iter()
                        .all(|&q| matches!(ir.morphism(q).unwrap().op, Operation::Pair { .. }));
                assert!(!packed, "a view of a pack in the loop");
            }
        }
    }
    assert_eq!(output(&ir), output(&raw));
}

/// P2 (rewrite DESIGN §1.2) holds for the views the placement merge
/// forwards: a `Proj` of a pack inside a loop SCC may alias a loop invariant
/// (in no SCC) when it feeds no route, and `is_consistent` — the check
/// `replay` asserts in debug builds — accepts every plan of every round.
/// Checked explicitly so a release build (where the assert is compiled out)
/// sees it too.
#[test]
fn every_rewrite_plan_is_consistent_with_forwarded_loop_views() {
    use mapal_rewrite::{
        analyze_const_fold, analyze_cse, analyze_dce, analyze_inline, analyze_lift,
        analyze_map_fusion, replay,
    };
    let passes: [fn(&CategoryIr) -> mapal_rewrite::RewritePlan; 6] = [
        analyze_inline,
        analyze_lift,
        analyze_const_fold,
        analyze_cse,
        analyze_dce,
        analyze_map_fusion,
    ];
    let mut cur = parse_lower(STATE);
    let mut invariant_views = 0;
    for _ in 0..mapal_rewrite::MAX_ROUNDS {
        let mut any = false;
        for pass in passes {
            let plan = pass(&cur);
            if plan.is_empty() {
                continue;
            }
            assert!(plan.is_consistent(&cur), "{plan:?}");
            let mut in_loop = std::collections::HashSet::new();
            for (f, _) in cur.funcs() {
                for scc in cur.loop_structure(f) {
                    in_loop.extend(scc.objects.iter().copied());
                }
            }
            invariant_views += plan
                .alias
                .iter()
                .filter(|&(k, v)| in_loop.contains(&k) && !in_loop.contains(v))
                .count();
            cur = replay(&cur, &plan).expect("canonical graph replays");
            any = true;
        }
        if !any {
            break;
        }
    }
    assert!(invariant_views > 0, "no loop view forwarded an invariant");
    assert_eq!(output(&cur), output(&parse_lower(STATE)));
}
