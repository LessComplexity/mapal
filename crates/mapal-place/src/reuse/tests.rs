//! Tests for the reuse counts.

use super::*;

#[test]
fn count_gt_is_the_brute_count() {
    for beta in -5i128..=5 {
        for c in -30i128..=30 {
            for n in 0..12u64 {
                let brute = (0..n as i128).filter(|&s| beta * s > c).count() as u64;
                assert_eq!(count_gt(beta, c, n), brute, "{beta} {c} {n}");
            }
        }
    }
}

#[test]
fn raster_walks_groups_column_by_column() {
    // gx 3, gy 4, h 2: (0,0) (1,0) (0,1) (1,1) (0,2) (1,2) (2,0) …
    let order: Vec<_> = (0..12).map(|l| launch_tile(3, 4, 2, l)).collect();
    assert_eq!(
        &order[..7],
        &[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 0)]
    );
    // A short last group (gy 3, h 2) walks its one row.
    let order: Vec<_> = (0..9).map(|l| launch_tile(3, 3, 2, l)).collect();
    assert_eq!(&order[6..], &[(2, 0), (2, 1), (2, 2)]);
}
