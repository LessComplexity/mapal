## Context

See proposal.md. Cargo autodiscovers both backends' `examples/emit.rs` into one output name. The GPU harness consumes LLVM's `target/release/examples/emit`; the active CUDA emitter consumer is `benches/matmul/regen.sh`.

## Goals / Non-Goals

**Goals:** distinct developer executable artifacts, correct registration of both existing profile tests, current implementation mappings, and repeatable local verification.

**Non-Goals:** GPU rung selection, hardware-detection fallback, or test timeouts. Earlier S54 failures are not proven to originate in the example collision.

## Decisions

- Rename CUDA's file to `crates/backends/cuda/examples/emit_cuda.rs`; Cargo's existing autodiscovery supplies the unique target. LLVM keeps `emit`, avoiding edits across the current GPU bundle pipeline. Update active CUDA commands and explanatory comments; immutable session and performance records keep their historical commands.
- Move the stranded SME-kc documentation and test attribute onto `sme_kc_reproduces_the_measured_optimum`. This removes one duplicate registration and restores one omitted registration; expected aggregate count is unchanged.
- Repair the six known drift references to their existing definitions, using code-search evidence. No implementation behavior changes to make docs pass.
- Preserve guard ownership while folding. The full suite found a reproducible counterexample: folding a guarded `Index(...) % 1` to zero removes its dependency edge and exposes the previously untaken Index as unconditional work. In `equations.rs:analyze_const_fold`, derive the guarded target set from the existing `CategoryIr::guard_plan` ownership lists and exclude those targets from scalar folding/forwarding. Keep top-level Phi selection/drop behavior. A nested owned Phi also defers; subsequent fixpoint rounds can fold work after its enclosing guard resolves. Protect all gated arms, including unknown conditions, rather than special-casing the generated seed or loop-armed guards.
- Expand a guarded `LoopEnter` handle through the existing `loop_plan`'s `decide_order`, `advance_order` and `exits`; protect their target objects as well. Ownership lists deliberately compress each loop unit to one handle, so reading only that list omits non-SCC exit payloads. Independent review demonstrated that folding such a payload otherwise makes replay panic. Reuse the canonical driver-cone facts, without a separate reachability/ownership algorithm; retain the existing SCC exclusion.
- Model delta: the existing `Trn analyze_const_fold : CategoryIr → RewritePlan` remains total, but its partial alias/constify candidates exclude guard-owned targets, preserving rewrite DESIGN R1/R4 and FRAMEWORK §4.5 composition soundness. `Dat GuardSite` stays derived by `mapal_ir::guard_plan`; no parallel ownership model, new runtime Loc or Trm. Emitter rename and profile-test registration have no runtime model delta. Map the CUDA CLI binding at `examples/emit_cuda.rs:main` in its implementation doc.

## Risks / Trade-offs

- Old developer commands naming CUDA `--example emit` stop working → update active callers and usage text; retain historical logs unchanged.
- macOS execution contention and sandbox sysctl restrictions can invalidate the local test baseline → run required-clang tests with normal host access and serial test execution, preserving logs.
- Guard-owned folding becomes conservative while the guard survives. This can defer safe optimization; correctness takes priority. The existing fixpoint retries after enclosing guard resolution. Verify unknown and constant conditions, selected/unselected traps, unused results, and the preserved random seed, including a negative control against the old implementation.

## Migration Plan

Build all workspace examples, verify unique example target names and both backend outputs, repair the discovered rewrite defect with focused regression evidence, then run the full host suite twice. Reconcile mappings/status and the coherence review, require clean drift, and archive the OpenSpec change.
