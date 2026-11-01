## Why

Stream setup allocates some raw tile outputs even when fusion proves those outputs are
never materialized. Frozen generated code shows these slots are allocated and freed but
never loaded; removing their allocations applies an existing compiler fact across dataflow
shapes without weakening the arena's concurrency guard. Read-only review also found a
possible prerequisite: readiness lookup must follow a stored fused output's actual writer,
so eliminating an incidental allocator wait cannot expose an unordered read.

## What Changes

- After the frozen five-profile validation and independent confirmations finish, make
  stream setup use the same output-storage facts as kernel emission.
- Investigate the unproved actual-writer lookup concern using emitted LLVM and runtime
  evidence as needed, including host readback, cross-chain readers and the existing veto
  for unordered legacy consumers. Discard it if refuted. No readiness fix is selected until
  the evidence establishes a defect; any justified fix must use existing group/alias facts.
- Remove only allocations proved unnecessary. Preserve stored fanout/host-visible outputs,
  existing buffer ownership, required completion ordering, timer placement and numerical semantics.
- Verify independent-join, diamond and two-chain examples, with single-chain and independent
  product controls, across sizes and arithmetic faces. Measure actual allocation counts,
  requested live bytes and ordinary execution time separately.
- Record all regressions and no-effect cases. An allocation reduction does not by itself
  establish an iteration or process-wall speedup.

## Capabilities

### New Capabilities

None. This is an internal code-generation optimization under the existing language and
runtime contract; `.openspec.yaml` opts out of spec deltas.

### Modified Capabilities

None. No new syntax, CLI setting, runtime ABI or numerical policy.

## Impact

Primary touch site: `crates/backends/llvm/src/func/resident.rs:FnEmit::chain_prologue`,
using existing `Epi.store_out` and member-storage facts consumed by
`FnEmit::emit_site_resident`. The readiness prerequisite involves `ResidentPlan::stream_of`
and `FnEmit::chains` in the same file as investigation sites, not yet authorized edits to
unproved behavior. Add integrated streams-plus-fusion tests in the existing LLVM
backend tests and reconcile the NVPTX implementation map, architecture and status.

The source remains frozen at `b4152c5` until `verify-s54-gpu-selection` completes its
hardware gates. This proposal authorizes no change during that baseline. It does not
remove `plan.chain.is_empty()` from the arena guard, introduce pooled ownership, hoist live
allocations, or implement CUDA graph replay. Those are separate candidates requiring
their own proofs and measurements.

Evidence rule from the user: every proposed correction must be grounded in an artifact
(generated LLVM, disassembly or runtime evidence), with a reproducer and a stated
acceptance/refutation condition. Static reasoning identifies what to inspect; it is not
permission to fix an unproved concern. Separate temporary emission using the frozen
compiler is allowed; source or binary changes remain deferred until baseline validation ends.
