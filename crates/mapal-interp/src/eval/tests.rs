//! Tests for `round_tf32`.

use super::round_tf32;

#[test]
fn tf32_keeps_ten_bits_round_to_nearest_even() {
    assert_eq!(round_tf32(1.0), 1.0);
    // 1 + 2^-10 is a TF32 value; 1 + 2^-11 ties to even (1.0); 1 + 3·2^-11 ties up.
    assert_eq!(round_tf32(1.0 + 2f32.powi(-10)), 1.0 + 2f32.powi(-10));
    assert_eq!(round_tf32(1.0 + 2f32.powi(-11)), 1.0);
    assert_eq!(
        round_tf32(1.0 + 3.0 * 2f32.powi(-11)),
        1.0 + 2.0 * 2f32.powi(-10)
    );
    assert_eq!(round_tf32(-1.0 - 2f32.powi(-12)), -1.0);
    assert!(round_tf32(f32::NAN).is_nan());
    assert_eq!(round_tf32(f32::MAX), f32::INFINITY);
    // Every rounding is within u_t = 2^-11 relative.
    for x in [0.1f32, 3.14159, 1e-30, 6.0e7] {
        assert!(((round_tf32(x) - x) / x).abs() <= 2f32.powi(-11));
    }
}
