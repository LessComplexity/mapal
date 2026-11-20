//! kernel-merging groundwork: `residence_plan` — where a value's bytes live
//! between its producer and its last reader (the working set from one to the
//! other against a cache's capacity).

mod testgen;

use mapal_ir::{CategoryIr, ObjectId, Residence};
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

const MB: u64 = 1 << 20;

/// `pt_attn_1024_64`: the scores `s`, `ex` and `p` are 4 MB each (the index
/// domains none). While `s` waits for its last reader (`ex`), `s`, `ex`, the
/// row maxima and `vv` are live; while `p` waits for `P·V`, `ex` (read by
/// `p`'s maker), `p` and `vv` — never `s`. The working sets fit an L2 of
/// 64 MB (the rtx-5080's) and none of 4 MB; a number lives on chip.
#[test]
fn attention_rows_live_in_l2() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../benches/algorithms/pt_attn_1024_64.mapal"
    ))
    .unwrap();
    let ir = lower(&src);
    let f = ir.entry();
    let plan = ir.residence_plan(f);
    assert_eq!(plan, ir.residence_plan(f), "stored once");
    let (s, ex, p, q, mx) = (
        named(&ir, "s"),
        named(&ir, "ex"),
        named(&ir, "p"),
        named(&ir, "q"),
        named(&ir, "mx"),
    );
    assert_eq!(plan.bytes(s), 4 * MB);
    assert_eq!(plan.bytes(mx), 4096);
    // `s` is live with `ex` (its last reader writes it) and the operands.
    assert!(plan.working_set(s) >= plan.bytes(s) + plan.bytes(ex) + plan.bytes(q));
    // `p` never meets `s` (dead at `ex`'s maker).
    assert!(plan.working_set(p) < plan.bytes(s) + plan.bytes(ex) + plan.bytes(p));
    for o in [s, ex, p] {
        assert_eq!(plan.at(o, 64 * MB), Residence::L2);
        assert_eq!(plan.at(o, 4 * MB), Residence::Dram);
    }
    // A number: no array bytes.
    let iter_ms = ir
        .objects()
        .find(|(_, o)| matches!(o.ty, mapal_ir::Ty::Float { .. }) && o.name.is_none())
        .map(|(id, _)| id)
        .unwrap();
    assert_eq!(plan.at(iter_ms, 0), Residence::Chip);
}

/// Over generated programs: every array's working set holds at least its
/// own bytes, equals a naive scan of the live classes over its interval,
/// and decides the residence at every capacity.
#[test]
fn working_set_is_the_peak_of_the_live_bytes_on_generated_programs() {
    let mut runner = TestRunner::deterministic();
    let mut srcs = Vec::new();
    for _ in 0..40 {
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
        let p = testgen::recur_prog_strategy()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        srcs.push(testgen::recur_src(&p));
    }
    let (mut arrays, mut l2, mut dram) = (0, 0, 0);
    for src in &srcs {
        let ir = lower(src);
        let f = ir.entry();
        let plan = ir.residence_plan(f);
        let lup = ir.last_use_plan(f);
        let acc = ir.access_plan(f);
        let end = ir
            .objects()
            .flat_map(|(o, _)| ir.in_edges(o).to_vec())
            .filter_map(|m| lup.position(m))
            .max()
            .unwrap_or(0);
        // Each array's own interval, then its class's union.
        let live: Vec<(ObjectId, u64, u32, u32)> = ir
            .objects()
            .filter(|(o, _)| plan.bytes(*o) > 0)
            .map(|(o, _)| {
                let made: Vec<u32> = ir
                    .in_edges(o)
                    .iter()
                    .filter(|&&m| !acc.writes(m).is_empty())
                    .filter_map(|&m| lup.position(m))
                    .collect();
                let birth = made.iter().copied().min().unwrap_or(0);
                let death = if lup.escapes(o) {
                    end
                } else {
                    lup.death(o)
                        .unwrap_or(birth)
                        .max(lup.view_death(o).unwrap_or(0))
                        .max(birth)
                };
                (o, plan.bytes(o), birth, death)
            })
            .collect();
        let class = |o: ObjectId| {
            let c = acc.class(o);
            let mine: Vec<_> = live.iter().filter(|x| acc.class(x.0) == c).collect();
            (
                mine.iter().map(|x| x.1).max().unwrap(),
                mine.iter().map(|x| x.2).min().unwrap(),
                mine.iter().map(|x| x.3).max().unwrap(),
            )
        };
        for &(o, b, _, _) in &live {
            arrays += 1;
            let (_, lo, hi) = class(o);
            let mut peak = 0;
            for p in lo..=hi {
                let mut seen = Vec::new();
                let mut at = 0;
                for &(x, _, _, _) in &live {
                    let c = acc.class(x);
                    let (cb, clo, chi) = class(x);
                    if !seen.contains(&c) && clo <= p && p <= chi {
                        seen.push(c);
                        at += cb;
                    }
                }
                peak = peak.max(at);
            }
            let w = plan.working_set(o);
            assert!(w >= b, "{src}");
            assert_eq!(w, peak, "{src}");
            assert_eq!(plan.at(o, w), Residence::L2);
            assert_eq!(plan.at(o, w - 1), Residence::Dram);
            if w <= 64 * 1024 {
                l2 += 1;
            } else {
                dram += 1;
            }
        }
    }
    eprintln!(
        "working_set: {arrays} arrays over {} programs, {l2} within 64 KB, {dram} above",
        srcs.len()
    );
    assert!(arrays > 0 && l2 > 0 && dram > 0);
}
