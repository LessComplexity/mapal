//! ADR-0050 C4: **the search's properties** over generated sites on the toy
//! machine and a toy lowering with a coordinate space:
//!
//! - **B&B** the branch and bound's best equals the exhaustive argmin over
//!   every point and traversal (the bounds are admissible: nothing better
//!   is ever cut).
//! - **P5** the pick is invariant under the enumeration order (every domain
//!   reversed), and rate noise of ±1 % moves only picks the search marks
//!   open (a runner-up within the null).
//! - **Gate** a refusal of the post-lowering gate takes the next candidate;
//!   the incumbent stays within the band.

use mapal_place::lower::{Declared, Lowering, Refusal, TravKind};
use mapal_place::*;
use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};

mod toy;

fn runner(cases: u32) -> TestRunner {
    TestRunner::new(Config {
        cases,
        failure_persistence: None,
        ..Config::default()
    })
}

/// The toy lowering: `[op, tm, tn, tx, ty, bk, stages]`.
struct Toy;

const TILE: [u64; 4] = [1, 2, 4, 8];
const LANES: [u64; 4] = [4, 8, 16, 32];

impl Lowering for Toy {
    fn id(&self) -> &'static str {
        "toy"
    }
    fn traversals(&self) -> &'static [TravKind] {
        &[
            TravKind::Ordered,
            TravKind::SplitAtomic,
            TravKind::SplitWorkspace,
            TravKind::Walk,
            TravKind::Hybrid,
        ]
    }
    fn ops(&self, _m: &Machine, _s: &Site) -> Vec<(usize, bool)> {
        vec![(0, false)]
    }
    fn slots(&self, _m: &Machine, _s: &Site, _op: usize, _swap: bool) -> Vec<Schedule> {
        Vec::new()
    }
    fn declare(&self, _m: &Machine, s: &Site, x: &Schedule) -> Result<Declared, Refusal> {
        if !s.m.is_multiple_of(x.bm()) || !s.n.is_multiple_of(x.bn()) || !s.k.is_multiple_of(x.bk())
        {
            return Err(Refusal::new("no tails"));
        }
        let d = toy::declared(x);
        if d.threads > 1024 {
            return Err(Refusal::new("threads"));
        }
        Ok(d)
    }
    fn dims(&self) -> usize {
        7
    }
    fn search_domain(&self, _m: &Machine, s: &Site, i: usize, _pre: &[u64]) -> Vec<u64> {
        match i {
            1 | 2 => TILE.to_vec(),
            3 | 4 => LANES.to_vec(),
            5 => [8, 16, 32]
                .into_iter()
                .filter(|b| s.k.is_multiple_of(*b))
                .collect(),
            _ => vec![1, 2, 3],
        }
    }
    fn point(&self, _m: &Machine, _s: &Site, c: &[u64], _swap: bool) -> Option<Schedule> {
        Some(toy::schedule(
            c[1],
            c[2],
            c[3],
            c[4],
            c[5],
            c[6],
            Traversal::Ordered { h: 1 },
        ))
    }
    fn tile_of(&self, pre: &[u64], _swap: bool) -> Option<(u64, u64)> {
        (pre.len() >= 5).then(|| (pre[4] * pre[1], pre[3] * pre[2]))
    }
}

fn site(m: u64, n: u64, k: u64) -> Site {
    Site {
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
    }
}

/// Every point and traversal priced: the exhaustive best.
fn exhaustive(m: &Machine, s: &Site) -> f64 {
    let l = Toy;
    let mut best = f64::INFINITY;
    for tm in TILE {
        for tn in TILE {
            for tx in LANES {
                for ty in LANES {
                    for bk in l.search_domain(m, s, 5, &[]) {
                        for st in 1..=3 {
                            let mut x =
                                toy::schedule(tm, tn, tx, ty, bk, st, Traversal::Ordered { h: 1 });
                            let Ok(d) = l.declare(m, s, &x) else { continue };
                            let occ = m.occupancy(d.threads, d.regs, &[(toy::SMEM, d.smem)]);
                            if occ == 0 {
                                continue;
                            }
                            x.per_unit = occ;
                            for t in space::traversals(m, s, &x, &d, l.traversals()) {
                                let y = Schedule {
                                    trav: t,
                                    ..x.clone()
                                };
                                let d = l.declare(m, s, &y).unwrap();
                                let e = evaluate(m, s, &y, &d, Bound::Value);
                                if e.feasible() {
                                    best = best.min(e.ns);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    best
}

fn ok(_: &Schedule, _: &Declared) -> Result<(), String> {
    Ok(())
}

#[test]
fn bnb_is_the_exhaustive_best() {
    let strat = (5u32..8, 5u32..8, 4u32..8);
    runner(12)
        .run(&strat, |(mp, np, kp)| {
            let s = site(1 << mp, 1 << np, 1 << kp);
            let m = toy::machine(&[1.0; 9]);
            let lows: [&dyn Lowering; 1] = [&Toy];
            let Some(r) = search(&m, &s, &lows, &Opts::default(), None, &mut ok) else {
                prop_assert!(exhaustive(&m, &s).is_infinite());
                return Ok(());
            };
            let best = exhaustive(&m, &s);
            prop_assert!(
                (r.top[0].value() - best).abs() <= best * 1e-12,
                "{}: b&b {} exhaustive {best} ({:?})",
                r.top[0].x.plan(),
                r.top[0].value(),
                r.stats
            );
            Ok(())
        })
        .expect("B&B");
}

#[test]
fn p5_order_invariant_and_noise_flips_only_open_picks() {
    let strat = (
        (5u32..8, 5u32..8, 4u32..8),
        proptest::collection::vec(prop::bool::ANY, 9),
    );
    let flips = std::cell::Cell::new(0);
    let cases = std::cell::Cell::new(0);
    runner(12)
        .run(&strat, |((mp, np, kp), signs)| {
            let s = site(1 << mp, 1 << np, 1 << kp);
            let m = toy::machine(&[1.0; 9]);
            let lows: [&dyn Lowering; 1] = [&Toy];
            let o = Opts::default();
            let Some(a) = search(&m, &s, &lows, &o, None, &mut ok) else {
                return Ok(());
            };
            let rev = Opts {
                reverse: true,
                ..o.clone()
            };
            let b = search(&m, &s, &lows, &rev, None, &mut ok).expect("the same space");
            prop_assert_eq!(a.pick.x.plan(), b.pick.x.plan());
            // ±1 % on every rate.
            let mut r = [1.0; 9];
            for (i, up) in signs.iter().enumerate() {
                r[i] = if *up { 1.01 } else { 0.99 };
            }
            let noisy = search(&toy::machine(&r), &s, &lows, &o, None, &mut ok).expect("a pick");
            cases.set(cases.get() + 1);
            if noisy.pick.x.plan() != a.pick.x.plan() {
                flips.set(flips.get() + 1);
                prop_assert!(
                    !a.open.is_empty(),
                    "{} -> {} under noise, not marked open",
                    a.pick.x.plan(),
                    noisy.pick.x.plan()
                );
            }
            Ok(())
        })
        .expect("P5");
    eprintln!(
        "P5: {} of {} picks moved under ±1% noise, each marked open",
        flips.get(),
        cases.get()
    );
}

#[test]
fn the_gate_takes_the_next_and_the_band_keeps_the_incumbent() {
    let s = site(128, 128, 128);
    let m = toy::machine(&[1.0; 9]);
    let lows: [&dyn Lowering; 1] = [&Toy];
    let o = Opts::default();
    let a = search(&m, &s, &lows, &o, None, &mut ok).expect("a pick");
    // The gate refuses the pick: the next is taken.
    let first = a.pick.x.plan();
    let mut refuse = |x: &Schedule, _: &Declared| match x.plan() == first {
        true => Err("spills".to_string()),
        false => Ok(()),
    };
    let b = search(&m, &s, &lows, &o, None, &mut refuse).expect("the next");
    assert_ne!(b.pick.x.plan(), first);
    assert_eq!(b.stats.gated.len(), 1);
    // An incumbent slower by less than the band stays; beyond it, it goes.
    let inc = a.top.last().expect("a runner-up").x.clone();
    let gap = a.top.last().unwrap().value() / a.best.value() - 1.0;
    let within = Opts {
        band: vec![("all".into(), gap + 0.01)],
        ..o.clone()
    };
    let kept = search(&m, &s, &lows, &within, Some((0, inc.clone())), &mut ok).unwrap();
    assert!(kept.kept && kept.pick.x.plan() == inc.plan());
    if gap > 0.001 {
        let beyond = Opts {
            band: vec![("all".into(), gap / 2.0)],
            ..o
        };
        let moved = search(&m, &s, &lows, &beyond, Some((0, inc.clone())), &mut ok).unwrap();
        assert!(!moved.kept && moved.pick.x.plan() == a.best.x.plan());
    }
}
