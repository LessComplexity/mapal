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
//! - **Q4** the evaluator on the toy machine with measured primitives: never
//!   slower with any primitive's rate up, never faster with any latency or
//!   concentration up, never below the admissible bound.

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

/// The toy machine with every cost model primitive a measured row, each
/// rate scaled by `r[i]` and each latency by `l[i]` (the primitive `i`).
fn primed(r: &[f64; 8], l: &[f64; 8]) -> Machine {
    let mut m = toy::machine(&[1.0; 9]);
    let row = |i: usize, name, of, per_clk, rate: f64, lat: f64, conc: f64| Prim {
        name,
        of,
        per_clk,
        rate: Fact::measured(rate * r[i], "toy"),
        latency: Fact::measured(lat * l[i], "toy"),
        conc: Fact::measured(conc, "toy"),
    };
    let (l2, mem) = (toy::L2, 3);
    m.prims = vec![
        row(
            0,
            "cp.l2",
            PrimOf::Copy {
                from: l2,
                bytes: 16,
            },
            false,
            50.0,
            150.0,
            16384.0,
        ),
        row(
            1,
            "cp.mem",
            PrimOf::Copy {
                from: mem,
                bytes: 16,
            },
            false,
            35.0,
            350.0,
            8192.0,
        ),
        row(
            2,
            "panel",
            PrimOf::Panel {
                from: mem,
                run: 64,
                rows: 128,
                fly: 8192,
            },
            false,
            25.0,
            350.0,
            8192.0,
        ),
        row(
            3,
            "lds",
            PrimOf::Scratch {
                bytes: 16,
                matrix: false,
            },
            true,
            128.0,
            30.0,
            8192.0,
        ),
        row(
            4,
            "bar",
            PrimOf::Barrier { lanes: 256 },
            true,
            0.03,
            30.0,
            1.0,
        ),
        row(
            5,
            "st",
            PrimOf::Store { to: mem, bytes: 16 },
            false,
            30.0,
            300.0,
            512.0,
        ),
        row(
            6,
            "red",
            PrimOf::Reduce {
                at: l2,
                sharing: 1,
                sectors: 4,
            },
            false,
            20.0,
            150.0,
            512.0,
        ),
        row(7, "op", PrimOf::Op(0), true, 256.0, 4.0, 32.0),
    ];
    m
}

#[test]
fn q4_the_evaluator_over_primitives_is_monotone_and_admissible() {
    let strat = (
        (0u32..3, 0u32..3, 2u32..4, 2u32..4, 2u32..5, 1u64..4),
        (4u32..9, 4u32..9, 7u32..12),
        0u32..4,
        0usize..8,
        1.5f64..4.0,
    );
    runner(400)
        .run(
            &strat,
            |((tmp, tnp, txp, typ, bkp, st), (mp, np, kp), tk, which, x)| {
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
                let one = [1.0; 8];
                let t0 = evaluate(&primed(&one, &one), &s, &xs, &d, Bound::Value);
                if !t0.feasible() {
                    return Ok(());
                }
                let mut r = one;
                r[which] = x;
                let t1 = evaluate(&primed(&r, &one), &s, &xs, &d, Bound::Value);
                prop_assert!(
                    t1.ns <= t0.ns * (1.0 + 1e-9),
                    "prim {which} rate up: {} -> {}\n{}\n{}",
                    t0.ns,
                    t1.ns,
                    t0.line(),
                    t1.line()
                );
                let t2 = evaluate(&primed(&one, &r), &s, &xs, &d, Bound::Value);
                prop_assert!(
                    t2.ns >= t0.ns * (1.0 - 1e-9),
                    "prim {which} latency up: {} -> {}\n{}\n{}",
                    t0.ns,
                    t2.ns,
                    t0.line(),
                    t2.line()
                );
                let lb = lower_bound(&primed(&one, &one), &s, &[0], Bound::Value);
                prop_assert!(lb <= t0.ns, "bound {} above {}", lb, t0.ns);
                Ok(())
            },
        )
        .expect("Q4");
}
