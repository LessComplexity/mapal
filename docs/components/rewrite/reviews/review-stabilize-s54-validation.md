# Review — stabilize S54 validation

2026-10-21. Checks the built change against FRAMEWORK §4.5 and rewrite DESIGN R1/R4.

## Coherence laws

- [x] 1. Placement honesty — `CategoryIr → RewritePlan → CategoryIr` remains in the compiler process; no runtime datum or placement was added. Guarded dependency targets are derived from `CategoryIr::guard_plan` and `loop_plan`.
- [x] 2. Transmission well-typing — no runtime transmission changed. The CUDA developer executable was renamed to `emit_cuda`; LLVM retains `emit`, so their output artifacts no longer collide.
- [x] 3. Placement totality — the existing replay/emitter realizations retain their signatures and locations. Candidate exclusion narrows optimization, not the set of programs accepted by the rewrite driver.
- [x] 4. Dependency mediation — `equations.rs:analyze_const_fold` no longer detaches dependencies from guarded work. Guard-owned loop handles expand through existing decide/advance/exit orders; SCC membership continues to protect the cycle.
- [x] 5. Composition soundness — directed negative controls reproduced `Done → Trapped` and a replay panic before the fixes. Selected/unselected trap, unconditional trap, constant/dynamic condition, nested guard and loop-payload checks preserve outcomes after the fix. The original random seed is retained.
- [x] 6. runsAt is a relation — no runtime realization or location changed; the compiler's pure analysis still feeds the shared replayer.

## Model delta actually shipped

The existing partial alias/constify candidates exclude guard-owned target objects, including compressed loop units' driver cones. Rewrite DESIGN §2 and IMPLEMENTATION map this restriction. There is no parallel guard ownership model, runtime API, new component or precision policy.

The validation work also separates the CUDA/LLVM example targets, restores one omitted SME profile test while removing a duplicate registration, and repairs stale implementation mappings. CUDA, LLVM, IR and lower mappings now point at their actual code; CUDA arena coloring remains explicitly planned.

## Evidence and limits

- Independent reviewer reproduced the guard-owned loop failure, then verified the fix under ConstFold alone and the full pipeline. Forty nested unknown-guard comparisons also passed.
- All 81 rewrite crate tests passed. The calc snapshots changed only after checking its unchanged five-line oracle output: the final graph shrank from 107 to 81 objects; deferred guard resolution uses 11 fixpoint rounds instead of 4 (cap 32).
- Workspace example metadata has unique names; both built executables emit the expected formats, missing arguments return usage errors, and the old CUDA target name is rejected.
- Required-clang release workspace tests passed twice with normal host access and serial execution: **1176 passed / 0 failed / 1 ignored** (78.0 s and 65.7 s). All workspace examples build. Final drift: **0 dead / 460 references**. Full logs are preserved in the local consolidation archive under `validation-s56/`. This review does not certify S54 GPU correctness or performance; those require the separate `verify-s54-gpu-selection` change.

## Modeling checks

One existing guard/loop analysis supplies membership. No persisted ownership cache or new plan channel was introduced. Safe folds inside a surviving guard may defer; preserving evaluation outcome takes priority over that optimization opportunity.
