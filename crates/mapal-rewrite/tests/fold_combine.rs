//! long-rows: `fold_combine` — how a fold's state splits into blocks of its
//! axis, and the law each value constant along the axis enters its term by
//! (a factor or a shift applied once after the combine, or a phase).

mod testgen;

use mapal_ir::{CategoryIr, DepLaw, FoldCombinePlan, MorphismId, ObjectId, Operation, RowFoldOp};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;

fn lower(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "{:?}\n{src}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("{d:?}\n{src}"));
    mapal_rewrite::rewrite(ir).ir
}

fn named(ir: &CategoryIr, n: &str) -> ObjectId {
    ir.objects()
        .find(|(_, o)| o.name.as_deref() == Some(n))
        .map(|(id, _)| id)
        .unwrap_or_else(|| panic!("no {n}"))
}

/// The site whose output is `n`.
fn site_of(ir: &CategoryIr, n: &str) -> MorphismId {
    let o = named(ir, n);
    ir.in_edges(o)[0]
}

/// `(value name, law)` of the site making `n`.
fn deps(ir: &CategoryIr, plan: &FoldCombinePlan, n: &str) -> Vec<(String, DepLaw)> {
    let c = plan
        .sites
        .get(site_of(ir, n))
        .unwrap_or_else(|| panic!("{n} is no fold site"));
    c.deps
        .iter()
        .map(|d| {
            let name = ir
                .object(d.value)
                .and_then(|o| o.name.clone())
                .unwrap_or_default();
            (name, d.law)
        })
        .collect()
}

/// `pt_attn_1024_64`: the row max reads the scores (a tile's output: no
/// dep); the denominator reads `ex = poly(s − mx)` — no exponential law, a
/// phase after `mx`; `P·V` reads `p = ex ÷ den`: `den` a quotient factor
/// (applied once after the sum), `mx` a phase through `ex`.
#[test]
fn attention_softmax_laws() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../benches/algorithms/pt_attn_1024_64.mapal"
    ))
    .unwrap();
    let ir = lower(&src);
    let plan = ir.fold_combine(ir.entry());
    let mx = &plan.sites[site_of(&ir, "mx")];
    assert!(matches!(mx.op, RowFoldOp::Select { .. }) && !mx.exact);
    assert_eq!((mx.rows, mx.k), (1024, 1024));
    assert!(mx.deps.is_empty());
    assert_eq!(deps(&ir, &plan, "den"), [("mx".into(), DepLaw::Phase)]);
    assert_eq!(
        deps(&ir, &plan, "o"),
        [
            ("mx".into(), DepLaw::Phase),
            ("den".into(), DepLaw::Factor { div: true })
        ]
    );
    // Stored: the second query reads it.
    assert_eq!(ir.fold_combine(ir.entry()), plan);
}

/// A norm's per-row scale is a product factor of the projection that reads
/// the scaled rows; a max over `x − c` is the shifted max; `x ÷ (c + 100)`
/// is a phase, `c ÷ (x + 100)` a factor; a max over `x · c` is a phase (the
/// sign of `c` is unknown).
#[test]
fn norm_shift_and_phase_laws() {
    let src = "fn main() {
    4096 -> iota -> t;
    64 -> iota -> r;
    64 -> iota -> k;
    t -> map { e -> (e * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 64.0 } -> x;
    t -> map { e -> (e * 11 + 57) % 103 - 51 -> widen_f32 -> v; v / 64.0 } -> w;
    t -> map { e -> x[e] * x[e] } -> sq;
    r -> map { i -> (0.0, k) -> fold { acc, j -> acc + sq[i * 64 + j] } } -> ss;
    r -> map { i -> 1.0 / (ss[i] / 64.0 + 0.5) } -> rinv;
    t -> map { e -> x[e] * rinv[e / 64] * 2.0 } -> xn;
    t -> map { e -> e / 64 -> i; e % 64 -> j; (0.0, k) -> fold { acc, u -> acc + xn[i * 64 + u] * w[u * 64 + j] } } -> y;
    t -> map { e -> x[e] - ss[e / 64] } -> sh;
    r -> map { i -> (-1000000.0, k) -> fold { acc, j -> sh[i * 64 + j] -> v; (v > acc) -> { -true-> v; -false-> acc; } } } -> m2;
    t -> map { e -> ss[e / 64] / (x[e] + 100.0) } -> q;
    r -> map { i -> (0.0, k) -> fold { acc, j -> acc + q[i * 64 + j] } } -> s4;
    t -> map { e -> x[e] / (ss[e / 64] + 100.0) } -> q2;
    r -> map { i -> (0.0, k) -> fold { acc, j -> acc + q2[i * 64 + j] } } -> s3;
    t -> map { e -> x[e] * ss[e / 64] } -> sc;
    r -> map { i -> (-1000000.0, k) -> fold { acc, j -> sc[i * 64 + j] -> v; (v > acc) -> { -true-> v; -false-> acc; } } } -> m3;
    y[5] -> println;
    m2[3] -> println;
    s3[2] -> println;
    s4[2] -> println;
    m3[1] -> println;
}
";
    let ir = lower(src);
    let plan = ir.fold_combine(ir.entry());
    assert_eq!(
        deps(&ir, &plan, "y"),
        [("rinv".into(), DepLaw::Factor { div: false })]
    );
    assert_eq!(
        deps(&ir, &plan, "m2"),
        [("ss".into(), DepLaw::Shift { sub: true })]
    );
    assert_eq!(deps(&ir, &plan, "s3"), [("ss".into(), DepLaw::Phase)]);
    // `ss ÷ (x + 100)` is `ss · (1 ÷ (x + 100))`: linear in `ss`.
    assert_eq!(
        deps(&ir, &plan, "s4"),
        [("ss".into(), DepLaw::Factor { div: false })]
    );
    assert_eq!(deps(&ir, &plan, "m3"), [("ss".into(), DepLaw::Phase)]);
    assert!(plan.sites[site_of(&ir, "ss")].deps.is_empty());
    assert_eq!(plan.phase_values(site_of(&ir, "s3")), [named(&ir, "ss")]);
}

/// One row (`pt_causal_1024_256`'s trip): the whole folds over the scores;
/// the denominator's term reads the max as a number capture — a phase.
#[test]
fn whole_folds_read_their_row_value_as_a_number() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../benches/algorithms/pt_causal_1024_256.mapal"
    ))
    .unwrap();
    let ir = lower(&src);
    let mut seen = 0;
    for (f, _) in ir.funcs() {
        let plan = ir.fold_combine(f);
        let whole = ir.whole_fold(f);
        for (m, c) in &plan.sites {
            if whole.contains_key(m) && !c.deps.is_empty() {
                assert!(c.deps.iter().all(|d| d.law == DepLaw::Phase), "{c:?}");
                assert_eq!(c.rows, 1);
                seen += 1;
            }
        }
    }
    assert!(seen >= 1, "the denominator's whole fold reads the max");
}

/// Generated programs (the change was not designed on them): every fold
/// site of `row_fold`, `whole_fold` and `tile_plan_strided` has its combine;
/// every dep's value is read by its `via` cell, is not the term itself, and a
/// factor / shift only stands under the step its law needs.
#[test]
fn every_fold_site_has_a_combine_on_generated_programs() {
    let mut runner = TestRunner::deterministic();
    let mut srcs = Vec::new();
    for i in 0..120 {
        let p = testgen::long_row_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        srcs.push(testgen::long_row_src(&p));
        if i % 3 != 0 {
            continue;
        }
        let p = testgen::twin_prog_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        srcs.push(testgen::twin_src(&p, true, None));
        let p = testgen::wide_chain_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        srcs.push(p.source());
    }
    let (mut sites, mut factor, mut shift, mut phase) = (0, 0, 0, 0);
    for src in &srcs {
        let ir = lower(src);
        for (f, _) in ir.funcs() {
            let plan = ir.fold_combine(f);
            let tiles = ir.tile_plan_strided(f);
            for m in ir
                .row_fold(f)
                .keys()
                .chain(ir.whole_fold(f).keys())
                .chain(tiles.sites.keys())
            {
                let c = plan.sites.get(m).unwrap_or_else(|| panic!("{m:?}\n{src}"));
                sites += 1;
                for d in &c.deps {
                    assert_ne!(d.value, c.term, "{src}");
                    let via = ir.morphism(d.via).unwrap();
                    assert!(matches!(via.op, Operation::Map { .. }));
                    match d.law {
                        DepLaw::Factor { .. } => {
                            factor += 1;
                            assert!(matches!(
                                c.op,
                                RowFoldOp::Arith {
                                    op: Operation::Add,
                                    ..
                                }
                            ));
                        }
                        DepLaw::Shift { .. } => {
                            shift += 1;
                            assert!(matches!(c.op, RowFoldOp::Select { .. }));
                        }
                        DepLaw::Phase => phase += 1,
                    }
                }
            }
        }
    }
    eprintln!("fold_combine: {sites} sites, {factor} factor / {shift} shift / {phase} phase deps");
    assert!(sites > 100 && factor > 0 && shift > 0 && phase > 0);
}
