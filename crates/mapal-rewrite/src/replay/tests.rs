//! Unit tests of the replayer helpers.

use super::split_capture_source;

#[test]
fn split_capture_source_partitions_caps_and_data() {
    let (caps, data) = split_capture_source(&[10, 20, 30], 2, 1, "Map");
    assert_eq!(caps, &[10, 20]);
    assert_eq!(data, &[30]);
    let (caps, data) = split_capture_source(&[10, 20, 30, 40], 2, 2, "Fold");
    assert_eq!(caps, &[10, 20]);
    assert_eq!(data, &[30, 40]);
}

#[test]
#[should_panic(expected = "replay: Map{k=2} source product arity expected 3, found 4")]
fn split_capture_source_over_arity_is_loud() {
    // Over-arity must NOT silently truncate in release (ADR-0027 review
    // major #8 — the old `debug_assert` let the extra slot drop).
    let f = [10, 20, 30, 40];
    let _ = split_capture_source(&f, 2, 1, "Map");
}

#[test]
#[should_panic(expected = "replay: Fold{k=1} source product arity expected 3, found 2")]
fn split_capture_source_under_arity_is_loud() {
    // A non-Pair-fed source yields a SHORT feeder list (`slot_feeders`
    // filters to Pair edges) — the same check names that invariant too.
    let f = [10, 20];
    let _ = split_capture_source(&f, 1, 2, "Fold");
}
