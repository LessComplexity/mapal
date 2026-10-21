# Review — S58 grouped-writer ordering

Scope: the ordering-only patch in `resident.rs` and its stream regressions. Small-M
selection and dead allocation elimination are separate, unaccepted candidates.

## FRAMEWORK §4.5 coherence laws

- [x] 1. Placement honesty — grouped members resolve to their physical GPU writer before
  feeder events and host readback; old emission fails four targeted regressions.
- [x] 2. Transmission well-typing — existing one-element/whole copies and runtime ABI
  remain unchanged. Same-Gmem stream events order accesses without moving an array.
- [x] 3. Placement totality — canonical aliases use the writer's chain; absent ownership
  stays legacy. A focused alias/unknown test covers that boundary.
- [x] 4. Dependency mediation — the existing event and producer-stream readback realize
  cross-stream and GPU-to-host readiness; unordered legacy consumers trigger the
  conservative existing stream veto. Runtime/API evidence is preserved in S57.
- [x] 5. Composition soundness — arena birth and stream lookup share `grouped_writer`;
  no independent ownership cache is introduced. Fifty focused checks and fourteen
  unchanged device LLVM units support the finite composition boundary.
- [x] 6. `runsAt` is a relation — CPU/device realizations and numerical faces remain
  independent. No operation is forced to one physical location by this correction.

## Validation and limits

Required-clang workspace: 1,188 passed, zero failed, one ignored. Ordinary RTX 5090:
200 frozen and 200 corrected processes pass; root independently rechecked raw output
and exact inventories after retrieval. Since both revisions pass, no observed old-code
race or speedup is claimed. These checks do not prove every possible program.

The arena's no-streams guard stays. `LastUsePlan` death is not GPU completion, and free
calls may mediate outstanding work. Allocation elimination remains open until its own
emitted ordering, pointers, frees, GPU values and measured savings are checked.

## Model delta and smell check

No new Dat, Trn, Loc, Trm, component or port. R7's existing intended relation is now
realized for grouped members; the implementation map and architecture/status were
reconciled. The ownership query is deduced from existing epilogue groups, not copied
into a second state record. Existing diagrams therefore require no new arrow.
