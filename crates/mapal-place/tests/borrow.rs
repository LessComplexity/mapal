//! **A borrowed row is marked, names its row, and errs slow** (cost-rental
//! review): [`Machine::cost`] of a primitive with no row of its own reads the
//! nearest measured row of its kind — on the side that bounds it where one is
//! measured (a narrower copy, a smaller box, fewer bytes in flight, more
//! lanes, more sectors), else the other side scaled by its mechanism's bound
//! — and marks it with the row's name; an exact row is the measurement,
//! unmarked.

mod toy;

use mapal_place::*;

fn row(name: &'static str, of: PrimOf, per_clk: bool, rate: f64, lat: f64, conc: f64) -> Prim {
    Prim {
        name,
        of,
        per_clk,
        rate: Fact::measured(rate, "toy"),
        latency: Fact::measured(lat, "toy"),
        conc: Fact::measured(conc, "toy"),
    }
}

fn machine() -> Machine {
    let mut m = toy::machine(&[1.0; 9]);
    let (l2, mem) = (toy::L2, toy::MEM);
    m.prims = vec![
        row(
            "tma.4096",
            PrimOf::Bulk {
                from: l2,
                bytes: 4096,
            },
            false,
            40.0,
            250.0,
            16384.0,
        ),
        row(
            "tma.16384",
            PrimOf::Bulk {
                from: l2,
                bytes: 16384,
            },
            false,
            60.0,
            260.0,
            32768.0,
        ),
        row(
            "cp.4",
            PrimOf::Copy {
                from: mem,
                bytes: 4,
            },
            false,
            8.0,
            300.0,
            4096.0,
        ),
        row(
            "cp.16",
            PrimOf::Copy {
                from: mem,
                bytes: 16,
            },
            false,
            10.0,
            300.0,
            4096.0,
        ),
        row(
            "bar.128",
            PrimOf::Barrier { lanes: 128 },
            true,
            0.05,
            20.0,
            1.0,
        ),
        row(
            "bar.512",
            PrimOf::Barrier { lanes: 512 },
            true,
            0.02,
            45.0,
            1.0,
        ),
        row(
            "red.4",
            PrimOf::Reduce {
                at: l2,
                sharing: 1,
                sectors: 4,
            },
            false,
            5.0,
            140.0,
            512.0,
        ),
        row(
            "red.8",
            PrimOf::Reduce {
                at: l2,
                sharing: 1,
                sectors: 8,
            },
            false,
            2.5,
            140.0,
            512.0,
        ),
        row(
            "panel.64.fly8k",
            PrimOf::Panel {
                from: mem,
                run: 64,
                rows: 128,
                fly: 8192,
            },
            false,
            6.0,
            300.0,
            8192.0,
        ),
    ];
    m
}

#[test]
fn a_borrowed_row_is_marked_names_its_row_and_errs_slow() {
    let m = machine();
    let b = Bound::Value;
    let cost = |of| m.cost(of, b);
    let l2 = toy::L2;
    // An exact row: the measurement, unmarked.
    let exact = cost(PrimOf::Bulk {
        from: l2,
        bytes: 16384,
    });
    assert_eq!(
        (exact.rate, exact.marked, exact.borrowed),
        (60.0, false, None)
    );
    // A box between the rows takes the smaller (a box's fixed cost amortized
    // over fewer bytes: at most as fast), marked and named.
    let mid = cost(PrimOf::Bulk {
        from: l2,
        bytes: 21845,
    });
    assert_eq!(
        (mid.rate, mid.marked, mid.borrowed),
        (60.0, true, Some("tma.16384"))
    );
    let between = cost(PrimOf::Bulk {
        from: l2,
        bytes: 8192,
    });
    assert_eq!((between.rate, between.borrowed), (40.0, Some("tma.4096")));
    // Below every row: the smallest, its rate scaled down by the bytes.
    let small = cost(PrimOf::Bulk {
        from: l2,
        bytes: 1024,
    });
    assert_eq!((small.rate, small.borrowed), (10.0, Some("tma.4096")));
    // A copy narrower than every row: scaled the same way.
    let narrow = cost(PrimOf::Copy {
        from: toy::MEM,
        bytes: 2,
    });
    assert_eq!((narrow.rate, narrow.borrowed), (4.0, Some("cp.4")));
    // A barrier over more lanes than one row takes the next row up (more
    // lanes arriving: at least as slow), past the most its latency scaled by
    // the lanes.
    let ghz = m.ghz(b);
    let bar = cost(PrimOf::Barrier { lanes: 384 });
    assert_eq!(bar.borrowed, Some("bar.512"));
    assert!((bar.latency - 45.0 / ghz).abs() < 1e-9);
    let wide = cost(PrimOf::Barrier { lanes: 1024 });
    assert!((wide.latency - 90.0 / ghz).abs() < 1e-9);
    // A reduction over more sectors than any row: the most-sector row's rate
    // scaled by the sectors (sector throughput), never the faster row as is.
    let red = cost(PrimOf::Reduce {
        at: l2,
        sharing: 1,
        sectors: 32,
    });
    assert_eq!((red.rate, red.borrowed), (2.5 * 8.0 / 32.0, Some("red.8")));
    let red6 = cost(PrimOf::Reduce {
        at: l2,
        sharing: 1,
        sectors: 6,
    });
    assert_eq!((red6.rate, red6.borrowed), (2.5, Some("red.8")));
    // A panel stream shorter and thinner in flight than the row: scaled by
    // both (Little's law for the flight).
    let panel = cost(PrimOf::Panel {
        from: toy::MEM,
        run: 32,
        rows: 128,
        fly: 4096,
    });
    assert_eq!(
        (panel.rate, panel.borrowed),
        (6.0 * 0.5 * 0.5, Some("panel.64.fly8k"))
    );
    // Every borrowed cost is marked.
    for c in [mid, between, small, narrow, bar, wide, red, red6, panel] {
        assert!(c.marked);
    }
}
