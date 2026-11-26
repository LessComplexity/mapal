//! ADR-0050 C1's property tests, over generated grids and schedules:
//!
//! - **P1** lockstep ordered and split: the closed form equals the exact LRU
//!   simulation byte for byte, every family (operands, outputs, the zero
//!   pass), every raster, partition count, slot count and keep.
//! - **P2** the walk and the hybrid: the closed form within one slice a row
//!   (column) and panel sample, or 5 %, of the simulation; the staggered
//!   start's window union reported against it.
//! - **P3** every closed form reads each slice at least once and at most once
//!   a reader, and never more with a larger keep.
//! - **P4** the evaluator: the time never rises with any rate, never falls
//!   with any extent; no schedule is below the admissible bound.
//! - **P6** the three-gap theorem equals the brute-force sorted gaps.
//!
//! (P5, the search's invariance under candidate order and rate noise, is the
//! search's: ADR-0050 C4.)

use mapal_place::lru::{lru_bytes, trace};
use mapal_place::reuse::{Grid, Order, reads, three_gap, walk_grid};
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

/// Generated lockstep grids: tiles, panels, slots, raster, partitions, slice
/// sizes and a keep around the grid's working sets.
fn lockstep_grid() -> impl Strategy<Value = (Grid, u64)> {
    (
        1u64..=12,
        1u64..=12,
        1u64..=24,
        1u64..=40,
        0u32..=3,
        prop_oneof![Just(1u64), Just(2), Just(3), Just(4)],
        any::<bool>(),
        any::<bool>(),
        (1u64..=6, 1u64..=6, 1u64..=6),
        0.0f64..1.6,
    )
        .prop_map(|(gx, gy, p, w, hp, s, atomic, zero, (sa, sb, so), kf)| {
            let h = (1u64 << hp).min(gy);
            let panels = p * s;
            let atomic = atomic && s > 1;
            let g = Grid {
                gx,
                gy,
                panels,
                slots: w,
                sa,
                sb,
                so,
                order: Order::Lockstep { h, s, atomic },
                zero: zero && atomic,
            };
            let ws = (gy * sa + gx * sb) * panels + gx * gy * so * s;
            (g, (kf * ws as f64) as u64)
        })
}

#[test]
fn p1_lockstep_closed_form_is_the_lru() {
    runner(400)
        .run(&lockstep_grid(), |(g, keep)| {
            let sim = lru_bytes(&trace(&g), keep);
            let cf = reads(&g, keep);
            prop_assert_eq!(cf, sim, "{:?} keep {}", g, keep);
            Ok(())
        })
        .expect("P1");
    // The cited grids at their real slices (ADR-0050 Context, C11): equal.
    for (gx, gy, p, w, keep, h) in [
        (32u64, 16u64, 128u64, 164u64, 6_480_199u64, 1u64),
        (16, 16, 128, 216, 20 << 20, 1),
        (32, 24, 256, 168, 68_786_585, 1),
        (32, 16, 128, 164, 6_480_199, 4),
    ] {
        let g = Grid {
            gx,
            gy,
            panels: p,
            slots: w,
            sa: 128 * 32 * 2,
            sb: 128 * 32 * 2,
            so: 128 * 128 * 4,
            order: Order::Lockstep {
                h,
                s: 1,
                atomic: false,
            },
            zero: false,
        };
        assert_eq!(reads(&g, keep), lru_bytes(&trace(&g), keep), "{g:?}");
    }
}

/// The walk's and hybrid's closed forms against the simulation.
#[test]
fn p2_walk_and_hybrid_within_a_slice_or_five_percent() {
    let strat = (
        2u64..=16,
        1u64..=12,
        2u64..=32,
        3u64..=40,
        any::<bool>(),
        0.0f64..1.4,
    );
    let worst = std::cell::Cell::new(0.0f64);
    let stag = std::cell::RefCell::new(Vec::<f64>::new());
    runner(200)
        .run(&strat, |(gx, gy, p, w, hybrid, kf)| {
            let tiles = gx * gy;
            let (sa, sb, so) = (4, 4, 8);
            let order = if hybrid && tiles > w && !tiles.is_multiple_of(w) {
                let tail = tiles - tiles / w * w;
                let (ctas, iters) = walk_grid(tail * p, w);
                Order::Hybrid { h: 1, ctas, iters }
            } else {
                let (ctas, iters) = walk_grid(tiles * p, w);
                Order::Walk { h: 1, ctas, iters }
            };
            let g = Grid {
                gx,
                gy,
                panels: p,
                slots: w,
                sa,
                sb,
                so,
                order,
                zero: true,
            };
            let ws = (gy * sa + gx * sb) * p;
            let keep = (kf * ws as f64) as u64;
            let sim = lru_bytes(&trace(&g), keep);
            let cf = reads(&g, keep);
            for (c, s, slice, lines) in [(cf.a, sim.a, sa, gy), (cf.b, sim.b, sb, gx)] {
                let diff = c.abs_diff(s) as f64;
                // One read a slice (a row or column panel) either way, or 5 %.
                let tol = (0.05 * s as f64).max((lines * slice * p) as f64);
                worst.set(worst.get().max(diff / s.max(1) as f64));
                prop_assert!(
                    diff <= tol,
                    "{:?} keep {}: closed {} sim {} (tol {})",
                    g,
                    keep,
                    c,
                    s,
                    tol
                );
            }
            // Staggered starts (reported): the window union against the
            // simulation of the same blocks.
            let gs = Grid {
                order: Order::Stagger {
                    h: 1,
                    s: 1,
                    atomic: false,
                },
                zero: false,
                ..g
            };
            let sim = lru_bytes(&trace(&gs), keep).operands();
            let cf = reads(&gs, keep).operands();
            stag.borrow_mut().push(cf as f64 / sim.max(1) as f64);
            Ok(())
        })
        .expect("P2");
    let mut stag = stag.into_inner();
    let worst = worst.get();
    stag.sort_by(f64::total_cmp);
    eprintln!(
        "P2: walk/hybrid worst relative {worst:.3}; stagger closed/sim median {:.3} [{:.3} .. {:.3}]",
        stag[stag.len() / 2],
        stag[0],
        stag[stag.len() - 1]
    );
}

#[test]
fn p3_reads_once_at_most_every_reader_monotone_in_keep() {
    runner(150)
        .run(&(lockstep_grid(), 0u32..3), |((g, _), kind)| {
            let tiles = g.gx * g.gy;
            let g = match kind {
                0 => g,
                1 => {
                    let (ctas, iters) = walk_grid(tiles * g.panels, g.slots);
                    Grid {
                        order: Order::Walk { h: 1, ctas, iters },
                        zero: true,
                        ..g
                    }
                }
                _ => Grid {
                    order: Order::Stagger {
                        h: 1,
                        s: 1,
                        atomic: false,
                    },
                    ..g
                },
            };
            let once = (g.gy * g.sa + g.gx * g.sb) * g.panels;
            let every = tiles * (g.sa + g.sb) * g.panels;
            let ws = once + tiles * g.so * 4;
            let mut last = u64::MAX;
            for keep in [0, ws / 8, ws / 3, ws / 2, ws, 4 * ws] {
                let r = reads(&g, keep).operands();
                prop_assert!(
                    once <= r && r <= every,
                    "{:?}: {} outside [{}, {}]",
                    g,
                    r,
                    once,
                    every
                );
                prop_assert!(r <= last, "{:?}: more reads with a larger keep", g);
                last = r;
            }
            prop_assert_eq!(reads(&g, 64 * ws).operands(), once);
            Ok(())
        })
        .expect("P3");
}

#[test]
fn p6_three_gap_is_the_sorted_gaps() {
    runner(2000)
        .run(&(1u64..200, 1u64..60, 1u64..300), |(step, n, m)| {
            let mut pts: Vec<u64> = (0..n).map(|j| j * step % m).collect();
            pts.sort_unstable();
            let mut brute: std::collections::BTreeMap<u64, u64> = Default::default();
            for w in pts.windows(2) {
                *brute.entry(w[1] - w[0]).or_default() += 1;
            }
            *brute.entry(pts[0] + m - pts[n as usize - 1]).or_default() += 1;
            let mut got: std::collections::BTreeMap<u64, u64> = Default::default();
            for (gp, c) in three_gap(step, n, m) {
                *got.entry(gp).or_default() += c;
            }
            prop_assert_eq!(got.clone(), brute, "step {} n {} m {}", step, n, m);
            prop_assert!(got.keys().filter(|&&g| g > 0).count() <= 3);
            Ok(())
        })
        .expect("P6");
}

// ---- P4: the evaluator on a generated machine ------------------------------

mod toy;

#[test]
fn p4_time_is_monotone_in_rates_and_extents_and_above_the_bound() {
    let strat = (
        (0u32..3, 0u32..3, 2u32..4, 2u32..4, 2u32..5, 1u64..4),
        (4u32..9, 4u32..9, 7u32..12),
        0u32..4,
        0usize..9,
        0u32..3,
    );
    runner(300)
        .run(
            &strat,
            |((tmp, tnp, txp, typ, bkp, st), (mp, np, kp), tk, rate, ext)| {
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
                let d = toy::declared(&x0);
                let mach = toy::machine(&[1.0; 9]);
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
                let x = Schedule { trav, ..x0 };
                let t0 = evaluate(&mach, &s, &x, &d, Bound::Value);
                if !t0.feasible() {
                    return Ok(());
                }
                // Every rate scaled up: never slower.
                let mut r = [1.0; 9];
                r[rate] = 2.0;
                let t1 = evaluate(&toy::machine(&r), &s, &x, &d, Bound::Value);
                prop_assert!(
                    t1.ns <= t0.ns * (1.0 + 1e-9),
                    "rate {} up: {} -> {}\n{}\n{}",
                    rate,
                    t0.ns,
                    t1.ns,
                    t0.line(),
                    t1.line()
                );
                // An extent doubled (the same schedule): never faster.
                let s2 = match ext {
                    0 => Site {
                        m: 2 * m,
                        ..s.clone()
                    },
                    1 => Site {
                        n: 2 * n,
                        ..s.clone()
                    },
                    _ => Site {
                        k: 2 * k,
                        ..s.clone()
                    },
                };
                // The same schedule: a walk keeps its blocks (more units each).
                let x2 = match (x.trav, ext) {
                    (Traversal::Walk { ctas, .. }, _) => {
                        let tiles = (s2.m / x.bm()) * (s2.n / x.bn());
                        let iters = (tiles * (s2.k / bk)).div_ceil(ctas);
                        Schedule {
                            trav: Traversal::Walk { h: 1, ctas, iters },
                            ..x.clone()
                        }
                    }
                    _ => x.clone(),
                };
                let t2 = evaluate(&mach, &s2, &x2, &d, Bound::Value);
                prop_assert!(
                    t2.ns >= t0.ns * (1.0 - 1e-9),
                    "extent {} doubled: {} -> {}\n{}\n{}",
                    ext,
                    t0.ns,
                    t2.ns,
                    t0.line(),
                    t2.line()
                );
                // Admissible: no evaluated time below the bound.
                let lb = lower_bound(&mach, &s, &[0], Bound::Value);
                prop_assert!(lb <= t0.ns, "bound {} above {}", lb, t0.ns);
                Ok(())
            },
        )
        .expect("P4");
}

/// ADR-0050 Context / R2: the cited walks' operand bytes by distinct bytes —
/// the closed form against the exact simulation (the design check's LRU:
/// RTX 3090 2048 × 4096 × 4096 956 MB, A100 2048 × 2048 × 8192 1074 MB, RTX
/// 5080 3072 × 4096 × 8192 124 MB where the requested-bytes rule read 1685),
/// bf16 operands, 128 × 128 tiles at two slots a unit.
#[test]
fn cited_walks_by_distinct_bytes() {
    let arms = [
        // units, keep, m, n, k, bk, LRU operand MB (design check)
        (82u64, 6_480_199u64, 2048u64, 4096u64, 4096u64, 32u64, 956.0),
        (108, 20 << 20, 2048, 2048, 8192, 64, 1074.0),
        (84, 68_786_585, 3072, 4096, 8192, 32, 124.0),
    ];
    for (units, keep, m, n, k, bk, cited) in arms {
        let (gx, gy, p) = (n / 128, m / 128, k / bk);
        let (ctas, iters) = walk_grid(gx * gy * p, units * 2);
        let g = Grid {
            gx,
            gy,
            panels: p,
            slots: units * 2,
            sa: 128 * bk * 2,
            sb: 128 * bk * 2,
            so: 128 * 128 * 4,
            order: Order::Walk { h: 1, ctas, iters },
            zero: true,
        };
        let sim = lru_bytes(&trace(&g), keep).operands() as f64 / 1e6;
        let cf = reads(&g, keep).operands() as f64 / 1e6;
        eprintln!(
            "walk {m}x{n}x{k} keep {keep}: closed {cf:.0} MB, LRU {sim:.0} MB (design check {cited})"
        );
        // The design check's LRU had no output stream in the trace for the
        // first two; within 10 % of it, and the closed form within 5 % of
        // this simulation.
        assert!(
            (sim - cited).abs() <= 0.10 * cited.max(100.0),
            "{m}x{n}x{k}: LRU {sim} vs {cited}"
        );
        assert!(
            (cf - sim).abs() <= 0.05 * sim.max(100.0),
            "{m}x{n}x{k}: closed {cf} vs LRU {sim}"
        );
    }
}
