//! Tests of the dispatch width arithmetic.

use super::DispatchCost;

fn cost(dispatch_ns: u64, lanes: u64) -> DispatchCost {
    DispatchCost {
        flops_per_ns: 100,
        bytes_per_ns: 10,
        dispatch_ns,
        lanes,
    }
}

/// No per-lane fact (every hand-measured profile): no width, so the
/// threshold rule and `width = 0` stay byte for byte.
#[test]
fn unmeasured_lanes_give_no_width() {
    for lanes in [0, 1] {
        assert_eq!(cost(8_300, lanes).width(1_000_000), None);
    }
}

/// `sqrt(T / w)`, one lane inline, the whole pool `0`, never above it.
#[test]
fn width_is_the_square_root_rule() {
    let c = cost(64_000, 64); // 1 us a lane
    assert_eq!(c.width(0), Some(1));
    assert_eq!(c.width(1_500), Some(1));
    assert_eq!(c.width(4_000), Some(2));
    assert_eq!(c.width(100_000), Some(10));
    assert_eq!(c.width(3_900_000), Some(62));
    assert_eq!(c.width(5_000_000), Some(0));
}

/// Monotone in the estimate up to the pool, then the pool.
#[test]
fn width_is_monotone() {
    let c = cost(97_000, 64);
    let mut last = 1;
    for ns in (0..60).map(|i| 1u64 << (i / 2)) {
        let w = c.width(ns).unwrap();
        let w = if w == 0 { 64 } else { w };
        assert!(w >= last, "ns={ns}: {w} < {last}");
        last = w;
    }
    assert_eq!(last, 64);
}

#[cfg(target_os = "linux")]
#[test]
fn cache_line_round_trips_and_keys_exactly() {
    use super::host_dispatch::{line, lookup, parse};
    let d = cost(97_521, 64);
    let text = format!("{}{}", line("k1", &cost(1, 2)), line("k2", &d));
    assert_eq!(lookup(&text, "k2"), Some(d));
    assert_eq!(lookup(&text, "k"), None, "a key prefix is not a hit");
    assert_eq!(parse("97521,64,100,10"), Some(d));
    for bad in ["", "1,2,3", "0,64,1,1", "1,64,0,1", "x,1,1,1", "1,2,3,4,5"] {
        assert_eq!(parse(bad), None, "{bad:?}");
    }
}

/// A measurement on a busy machine is not cached: the 1-minute load or the
/// tasks runnable now (less the reader) above a quarter of the online
/// CPUs. This box idle, under another project's test suite, and under a
/// burst the 1-minute average has not caught yet; unreadable is quiet.
#[cfg(target_os = "linux")]
#[test]
fn a_busy_machine_does_not_cache_its_wave() {
    use super::cpu_list_len;
    use super::host_dispatch::busy;
    assert_eq!(cpu_list_len("0-63\n"), 64);
    assert_eq!(cpu_list_len("0,4"), 2);
    assert_eq!(cpu_list_len(""), 1);
    assert!(!busy("3.41 10.50 77.43 2/1234 5678\n", 64));
    assert!(busy("41.20 30.02 12.00 3/1234 5678\n", 64));
    assert!(busy("2.00 1.00 1.00 40/1234 5678\n", 64));
    assert!(
        !busy("16.00 1.00 1.00 17/1234 5678\n", 64),
        "a quarter is quiet"
    );
    assert!(!busy("", 64));
    assert!(!busy("x y z", 64));
}
