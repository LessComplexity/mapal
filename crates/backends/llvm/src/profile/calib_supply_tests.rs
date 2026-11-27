//! Tests of the calibration supply.

use super::*;

/// residency `calib-supply`: every committed line parses for its own key
/// into measured rates (a part someone measured is priced with no manual
/// step); a key no line carries resolves to `None`, naming what was
/// tried, so the build says it runs unpriced.
#[test]
fn committed_lines_price_their_parts_and_a_miss_is_named() {
    let mut keys = 0;
    for text in COMMITTED_CALIB {
        for line in text
            .lines()
            .map(|l| l.split('#').next().unwrap_or("").trim())
        {
            let Some((key, _)) = line.split_once('\t') else {
                continue;
            };
            let over = calib_line(text, key).expect("a committed line parses");
            assert!(over.iter().filter(|v| v.is_some()).count() >= 10, "{key}");
            let (src, got) = calib_supply(key);
            assert!(!matches!(src, CalibSource::None(_)), "{key}: {src:?}");
            assert!(got.is_some_and(|t| calib_line(&t, key).is_some()));
            keys += 1;
        }
    }
    assert!(keys >= 1);
    let (src, got) = calib_supply("cc=1 sms=1 l2=1 mem=1x1");
    assert!(got.is_none());
    assert!(
        matches!(&src, CalibSource::None(t) if t.contains("committed")),
        "{src:?}"
    );
}
