## Why

An interpreter-accepted recurrent checkpoint program panics during optimization with an unmapped feeder. Existing minimized evidence reproduces the failure without GPU execution; the exact feeder provenance must be established before repairing the shared loop replayer.

## What Changes

- Trace the failing pass, canonical route and feeder in an isolated snapshot of b4152c5, binding both the tiny674 and original5907 failures to actual object metadata.
- If confirmed, repair reconstruction of required acyclic loop-route work using the existing canonical loop plan, with regressions that force replay and preserve values, effects and trap behavior.
- Produce a reviewable temporary patch and proof receipts while keeping the shared source, HEAD and frozen compiler/runtime binaries unchanged for the approved GPU correctness pilot.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `rewrite-equivalence`: optimization preserves observed loop outputs and computed exit values, including the zero-iteration boundary, without an internal reconstruction failure.

## Impact

Planned touch sites are `crates/mapal-rewrite/src/replay.rs`, temporary diagnostics in `src/driver.rs`, and focused rewrite regression tests. The existing `mapal_ir::LoopPlan` remains the authority. No runtime ABI, dependency, numerical policy or GPU mechanism changes. Production application and documentation reconciliation remain separate pending work after temporary proof and review.
