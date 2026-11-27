//! The shared differential generator draws every ADR-0029 widening edge.

mod testgen;

use mapal_ir::{Operation, Ty, validate};
use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::TestRunner;
use testgen::{build, prog_strategy};

#[test]
fn random_programs_cover_widen_lattice() {
    let mut runner = TestRunner::deterministic();
    let mut seen = [false; 4];
    for (count, trap_free) in [(128, false), (64, true)] {
        let strat = prog_strategy(trap_free, false);
        for _ in 0..count {
            let prog = strat.new_tree(&mut runner).unwrap().current();
            let ir = build(&prog).ir;
            assert!(validate(&ir).is_empty());
            for (_, m) in ir.morphisms().filter(|(_, m)| m.op == Operation::Widen) {
                let source = &ir.object(m.source).unwrap().ty;
                let target = &ir.object(m.target).unwrap().ty;
                match (source, target) {
                    (s, t) if *s == Ty::i32() && *t == Ty::i64() => seen[0] = true,
                    (s, t) if *s == Ty::i32() && *t == Ty::f32() => seen[1] = true,
                    (s, t) if *s == Ty::i32() && *t == Ty::f64() => seen[2] = true,
                    (s, t) if *s == Ty::f32() && *t == Ty::f64() => seen[3] = true,
                    pair => panic!("generator emitted illegal Widen pair: {pair:?}"),
                }
            }
        }
    }
    assert!(
        seen.iter().all(|x| *x),
        "missing Widen lattice edges: {seen:?}"
    );
}

/// The edge of `f32`'s exact integers (ADR-0051 D4).
const EXACT: i128 = 1 << 24;

fn lower(src: &str) -> Result<mapal_ir::CategoryIr, Vec<mapal_syntax::Diagnostic>> {
    let po = mapal_syntax::parse(src);
    assert!(
        po.diagnostics.is_empty(),
        "parse: {:?}\n{src}",
        po.diagnostics
    );
    mapal_lower::lower(src, &po.program)
}

/// Every `Widen` of an `i32` to `f32` in `ir`: its range where it runs
/// (`BoundsProof::widen_range`).
fn i32_f32_widen_ranges(ir: &mapal_ir::CategoryIr) -> Vec<Option<(i128, i128)>> {
    let mut out = Vec::new();
    for (f, fd) in ir.funcs() {
        let bp = ir.bounds_proof(f);
        for &m in &fd.morphisms {
            let Some(x) = ir.morphism(m).filter(|x| x.op == Operation::Widen) else {
                continue;
            };
            let (s, t) = (
                &ir.object(x.source).unwrap().ty,
                &ir.object(x.target).unwrap().ty,
            );
            if *s == Ty::i32() && *t == Ty::f32() {
                out.push(bp.widen_range(m));
            }
        }
    }
    out
}

fn printed(ir: &mapal_ir::CategoryIr, src: &str) -> Vec<String> {
    let r = mapal_interp::run(ir, 50_000_000);
    assert!(
        matches!(r.outcome, mapal_interp::Outcome::Done(_)),
        "{:?}\n{src}",
        r.outcome
    );
    r.output.lines().map(str::to_string).collect()
}

/// ADR-0051 D4 on generated programs (`testgen::widen_law_strategy`): integer
/// laws of a map's index over `+ − ·k ÷k %k` — constants around 2²⁴, products
/// wrapping past `i32`, a guard arm narrowing the index — converted to `f32`.
/// On the interpreter (the oracle twin prints the law's extremes over exactly
/// the cells the conversion runs on): the facts' range of each conversion
/// (`BoundsProof::widen_range`, the `%` rule included) holds every value it
/// converts; `widen<f32>` is admitted only where every value is exact
/// (within ±2²⁴), and wherever the facts' range is; otherwise it is L1614
/// and `to<f32>` (always admitted) is the conversion. Over programs the rules
/// were not designed on.
#[test]
fn widen_f32_is_admitted_only_where_every_value_is_exact() {
    let n: usize = std::env::var("MAPAL_TESTGEN_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(240);
    let mut runner = TestRunner::deterministic();
    let strat = testgen::widen_law_strategy();
    let progs: Vec<_> = (0..n)
        .map(|_| strat.new_tree(&mut runner).unwrap().current())
        .collect();
    let [
        programs,
        ranged,
        admitted,
        rejected,
        near_edge,
        mod_wrapped,
        guarded,
    ] = testgen::par_count(&progs, |i, p| {
        let (widen, to) = (p.source("widen"), p.source("to"));
        testgen::assert_formatted(&widen, "widen program");
        if let Ok(dir) = std::env::var("MAPAL_TESTGEN_DUMP") {
            std::fs::write(format!("{dir}/widen_law_{i:03}.mapal"), &widen).unwrap();
        }
        // `to<f32>` of an `i32` is always admitted: the facts' range of the
        // conversion where it runs.
        let to_ir = lower(&to).unwrap_or_else(|d| panic!("to<f32>: {d:?}\n{to}"));
        let ranges = i32_f32_widen_ranges(&to_ir);
        assert_eq!(ranges.len(), 1, "one conversion\n{to}");
        let range = ranges[0];
        printed(&to_ir, &to);
        // The law's values where the conversion runs, on the interpreter.
        let observed = p.observer().map(|obs| {
            let ir = lower(&obs).unwrap_or_else(|d| panic!("observer: {d:?}\n{obs}"));
            let out = printed(&ir, &obs);
            let v: Vec<i128> = out.iter().map(|l| l.trim().parse().unwrap()).collect();
            assert_eq!(v.len(), 2, "{out:?}\n{obs}");
            (v[1], v[0])
        });
        if let (Some((a, b)), Some((lo, hi))) = (range, observed) {
            assert!(
                a <= lo && hi <= b,
                "the facts' range [{a}, {b}] misses the values [{lo}, {hi}]\n{to}"
            );
        }
        let exact = |r: (i128, i128)| -EXACT <= r.0 && r.1 <= EXACT;
        let mut c = [1, usize::from(range.is_some()), 0, 0, 0, 0, 0];
        match lower(&widen) {
            Ok(ir) => {
                if let Some(o) = observed {
                    assert!(exact(o), "widen<f32> admitted over {o:?}\n{widen}");
                    c[4] = usize::from(o.0.abs().max(o.1.abs()) >= EXACT / 2);
                }
                printed(&ir, &widen);
                c[2] = 1;
                c[5] = usize::from(p.law.mod_of_wrapped(&p.cells()));
                c[6] = usize::from(p.guard.is_some());
            }
            Err(ds) => {
                assert!(
                    ds.iter().all(|d| d.code.0 == "L1614"),
                    "a rejected widen<f32> is L1614 only: {ds:?}\n{widen}"
                );
                assert!(
                    !range.is_some_and(exact),
                    "the facts bound it to {range:?}, yet widen<f32> is rejected\n{widen}"
                );
                c[3] = 1;
            }
        }
        c
    });
    eprintln!(
        "widen<f32> laws: {programs} programs, {ranged} ranged, {admitted} admitted \
         ({near_edge} past 2²³, {mod_wrapped} through a % of a wrapped product, \
         {guarded} under a guard), {rejected} rejected"
    );
    assert!(ranged * 2 > programs, "the facts range most laws");
    assert!(admitted > 0 && rejected > 0, "both outcomes reached");
    assert!(near_edge > 0, "an admitted law reaches past 2²³");
    assert!(mod_wrapped > 0, "a % of a wrapped product is admitted");
    assert!(guarded > 0, "a guard arm's narrowing is used");
}
