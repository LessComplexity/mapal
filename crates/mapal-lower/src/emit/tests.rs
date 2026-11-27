//! Unit tests for the emission pass.

use super::*;

#[test]
fn token_in_body_maps_to_user_facing_l1605_not_internal() {
    // ADR-0027 review minor #13: a token-typed capture reaches
    // `map_captured` → `IrError::TokenInBody` (constructible only via
    // IrBuilder — the surface can't name tokens, so this unit test drives
    // the mapping directly). It must surface as the user-facing body
    // L-code (L1605 territory: bodies are effect-free), never L1901
    // "internal lowering error".
    let d = ir_err(IrError::TokenInBody, syn::SourceLoc { start: 0, end: 0 });
    assert_eq!(d.code.0, "L1605", "{d:?}");
    assert!(!d.message.contains("internal"), "{d:?}");
}
