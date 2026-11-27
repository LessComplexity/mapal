//! The cost model's pipeline calculation (ADR-0050, `cost-model.md`): its
//! terms' properties over generated flows, and the evaluator over a machine
//! whose every primitive is a measured row.
//!
//! - **Q1** a flow's time never rises with its rate or its bytes in flight,
//!   never falls with its latency or its work;
//! - **Q2** the pipeline never rises with any flow's rate, never falls with
//!   any latency, the fill or the serial term;
//! - **Q3** with every ring unbounded (no latency term), no fill and no serial
//!   term, a trip's steady state is the max over the resources of work ÷ rate,
//!   and the loop within one trip of `trips` × it (the max form);
//! - **Q4** the evaluator on the toy machine with every primitive a term
//!   reads measured, in both latency regimes and at every bound: never
//!   slower with any primitive's rate up, never faster with its latency or
//!   its concurrency up, never below the admissible bound or the search's
//!   floor.

mod toy;

use mapal_place::reuse::walk_grid;
use mapal_place::*;
use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};

fn runner(cases: u32) -> TestRunner {
    TestRunner::new(Config {
        cases,
        failure_persistence: None,
        ..Config::default()
    })
}

fn flow() -> impl Strategy<Value = Flow> {
    (
        1.0f64..1e6,
        0.01f64..1e3,
        0.0f64..2e3,
        prop_oneof![Just(f64::INFINITY), 1.0f64..1e6],
    )
        .prop_map(|(work, rate, latency, fly)| Flow {
            work,
            rate,
            latency,
            fly,
        })
}

#[test]
fn q1_a_flow_is_monotone_in_its_rate_latency_and_flight() {
    runner(2000)
        .run(&(flow(), 1.0f64..4.0), |(f, x)| {
            let t = f.ns();
            let up = |g: Flow| g.ns();
            let rate = up(Flow {
                rate: f.rate * x,
                ..f
            });
            let fly = up(Flow {
                fly: f.fly * x,
                ..f
            });
            let lat = up(Flow {
                latency: f.latency * x,
                ..f
            });
            let work = up(Flow {
                work: f.work * x,
                ..f
            });
            prop_assert!(rate <= t * (1.0 + 1e-12));
            prop_assert!(fly <= t * (1.0 + 1e-12));
            prop_assert!(lat >= t * (1.0 - 1e-12));
            prop_assert!(work >= t * (1.0 - 1e-12));
            // Below the knee (in flight < latency × rate) it is latency-bound.
            if f.fly.is_finite() && f.fly < f.latency * f.rate {
                prop_assert!((t - f.work * f.latency / f.fly.max(1.0)).abs() <= 1e-9 * t.max(1.0));
            } else {
                prop_assert!((t - f.work / f.rate).abs() <= 1e-9 * t.max(1.0));
            }
            Ok(())
        })
        .expect("Q1");
}

#[test]
fn q2_the_pipeline_is_monotone_in_every_rate_and_latency() {
    let strat = (
        prop::collection::vec(flow(), 1..4),
        prop::collection::vec(flow(), 1..5),
        1.0f64..1e4,
        0.0f64..1e4,
        0.0f64..1e3,
        0usize..9,
        1.5f64..4.0,
    );
    runner(2000)
        .run(&strat, |(c, mv, trips, fill, serial, which, x)| {
            let t = pipeline(trips, &c, &mv, fill, serial);
            let (mut c2, mut m2) = (c.clone(), mv.clone());
            // Every rate up: never slower.
            for f in c2.iter_mut().chain(m2.iter_mut()) {
                f.rate *= x;
            }
            prop_assert!(pipeline(trips, &c2, &m2, fill, serial) <= t * (1.0 + 1e-12));
            // One flow's rate up, one flow's latency up.
            let (mut c3, mut m3) = (c.clone(), mv.clone());
            let n = c3.len() + m3.len();
            let i = which % n;
            let f = if i < c3.len() {
                &mut c3[i]
            } else {
                &mut m3[i - c.len()]
            };
            f.rate *= x;
            prop_assert!(pipeline(trips, &c3, &m3, fill, serial) <= t * (1.0 + 1e-12));
            let (mut c4, mut m4) = (c.clone(), mv.clone());
            let f = if i < c4.len() {
                &mut c4[i]
            } else {
                &mut m4[i - c.len()]
            };
            f.latency *= x;
            prop_assert!(pipeline(trips, &c4, &m4, fill, serial) >= t * (1.0 - 1e-12));
            // The fill and the serial term: never faster.
            prop_assert!(pipeline(trips, &c, &mv, fill * x + 1.0, serial) >= t * (1.0 - 1e-12));
            prop_assert!(pipeline(trips, &c, &mv, fill, serial * x + 1.0) >= t * (1.0 - 1e-12));
            Ok(())
        })
        .expect("Q2");
}

#[test]
fn q3_unbounded_rings_reduce_to_the_max_form() {
    let strat = (
        prop::collection::vec(flow(), 1..4),
        prop::collection::vec(flow(), 1..5),
        1.0f64..1e4,
    );
    runner(2000)
        .run(&strat, |(c, mv, trips)| {
            let open = |v: &[Flow]| -> Vec<Flow> {
                v.iter()
                    .map(|f| Flow {
                        fly: f64::INFINITY,
                        ..*f
                    })
                    .collect()
            };
            let (c, mv) = (open(&c), open(&mv));
            let max = c
                .iter()
                .chain(mv.iter())
                .map(|f| f.work / f.rate)
                .fold(0.0, f64::max);
            let all: Vec<Flow> = c.iter().chain(mv.iter()).copied().collect();
            prop_assert!((steady(&all) - max).abs() <= 1e-9 * max);
            let p = pipeline(trips, &c, &mv, 0.0, 0.0);
            prop_assert!(
                p >= trips * max * (1.0 - 1e-12),
                "{p} below {}",
                trips * max
            );
            prop_assert!(
                p <= (trips + 1.0) * max * (1.0 + 1e-12),
                "{p} above {}",
                (trips + 1.0) * max
            );
            Ok(())
        })
        .expect("Q3");
}

/// **Q4** over every primitive an evaluator term reads (`toy::prim_rows`:
/// no row that no term reads, so no case is vacuous), in both latency
/// regimes (the memory's knee latency above the cache's, and below it — the
/// RTX 3090's and the A100's rows), at every bound: raising one row's rate
/// never slows the schedule, raising its latency or its concurrency never
/// speeds it, and neither the admissible bound nor the search's floor
/// ([`evaluate_floor`], what the branch and bound prunes with) is ever above
/// the evaluation.
#[test]
fn q4_the_evaluator_over_primitives_is_monotone_and_admissible() {
    let rows = toy::prim_rows(false).len();
    let strat = (
        (0u32..3, 0u32..3, 2u32..4, 2u32..4, 2u32..5, 1u64..4),
        (4u32..9, 4u32..9, 7u32..12),
        0u32..4,
        0usize..rows,
        1.5f64..4.0,
        any::<bool>(),
    );
    runner(400)
        .run(
            &strat,
            |((tmp, tnp, txp, typ, bkp, st), (mp, np, kp), tk, which, x, below)| {
                let (tm, tn, tx, ty, bk) = (1 << tmp, 1 << tnp, 1 << txp, 1 << typ, 1 << bkp);
                let (m, n, k) = (1u64 << mp, 1u64 << np, 1u64 << kp);
                let s = Site {
                    m,
                    n,
                    k,
                    batch: 1,
                    ea: 4,
                    eb: 4,
                    eo: 4,
                    epi: 0,
                    fold: Fold::Free,
                    kcut: true,
                    fmts: vec![Fmt::F32],
                    warm: [false, false],
                };
                let x0 = toy::schedule(tm, tn, tx, ty, bk, st, Traversal::Ordered { h: 1 });
                if m % x0.bm() != 0 || n % x0.bn() != 0 || k % bk != 0 {
                    return Ok(());
                }
                let d = Declared {
                    mover: Move::Copy(16),
                    scratch: (x0.bm() + x0.bn()) * bk * 4,
                    barriers: 1.0,
                    ..toy::declared(&x0)
                };
                let trav = match tk {
                    0 => Traversal::Ordered { h: 1 },
                    1 => Traversal::Split {
                        s: 2.min(k / bk),
                        comb: Comb::Atomic,
                        h: 1,
                    },
                    2 => {
                        let tiles = (m / x0.bm()) * (n / x0.bn());
                        let (ctas, iters) = walk_grid(tiles * (k / bk), 16 * 2);
                        Traversal::Walk { h: 1, ctas, iters }
                    }
                    _ => Traversal::Split {
                        s: 2.min(k / bk),
                        comb: Comb::Workspace,
                        h: 1,
                    },
                };
                let xs = Schedule { trav, ..x0 };
                let one = vec![1.0; rows];
                let mut up = one.clone();
                up[which] = x;
                let m0 = toy::primed(below, &one, &one, &one);
                for b in [Bound::Slow, Bound::Value, Bound::Fast] {
                    let t0 = evaluate(&m0, &s, &xs, &d, b);
                    if !t0.feasible() {
                        return Ok(());
                    }
                    let name = toy::prim_rows(below)[which].0;
                    for (what, mx, faster) in [
                        ("rate", toy::primed(below, &up, &one, &one), true),
                        ("latency", toy::primed(below, &one, &up, &one), false),
                        ("concurrency", toy::primed(below, &one, &one, &up), false),
                    ] {
                        let t1 = evaluate(&mx, &s, &xs, &d, b);
                        let ok = match faster {
                            true => t1.ns <= t0.ns * (1.0 + 1e-9),
                            false => t1.ns >= t0.ns * (1.0 - 1e-9),
                        };
                        prop_assert!(
                            ok,
                            "{name} {what} up at {b:?}: {} -> {}\n{}\n{}",
                            t0.ns,
                            t1.ns,
                            t0.line(),
                            t1.line()
                        );
                    }
                    let lb = lower_bound(&m0, &s, &[0], b);
                    prop_assert!(lb <= t0.ns, "bound {} above {} at {b:?}", lb, t0.ns);
                    let floor = evaluate_floor(&m0, &s, &xs, &d, b).ns;
                    prop_assert!(
                        floor <= t0.ns * (1.0 + 1e-9),
                        "the floor {floor} above the value {} at {b:?} (memory below: {below})",
                        t0.ns
                    );
                }
                Ok(())
            },
        )
        .expect("Q4");
}
