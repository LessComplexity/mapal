//! Provenance helpers shared by the `PLACE_*` measured-fact tables.

use super::*;

/// A measured value's cite (the profile's comment carries the detail).
pub(super) const fn m(c: &'static str) -> Src {
    Src::Measured(c)
}
/// A Mapal kernel's fitted or achieved rate: kept as a replay line, never
/// read by the placement evaluator (ADR-0050 D2, "fits").
pub(super) const FIT: Src = Src::Derived("fit of a Mapal kernel (replay only)");
pub(super) const DATASHEET: Src = Src::Declared("datasheet");
