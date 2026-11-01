# FRAMEWORK §4.5 — loop-route replay repair

Reviewed final patch `32384fc81b26faff5006019b37d330daab9475bc2e1aa799afde2091282d8203`.
The independent [code review](../../../../benches/results/gpu/validation/loop-route-replay/review-gpu-owner.md)
and [diagnostic proof](../../../../benches/results/gpu/validation/loop-route-replay/PROOF.md)
bind the actual missing feeders, old-code failures and corrected outcomes. Root reviewed
the same production diff and formatted tests before applying them. No blocker remains
for this narrow repair; GPU ordering and large native emission latency are separate.

Existing Dat are `CategoryIr`, `LoopPlan`, `RewritePlan` and replay's object map. Existing
Trn is `replay`, concretely `replay.rs:{reconstruct_loop,reconstruct_a,feed}`. This remains
a pure in-process graph reconstruction under the component's existing DESIGN physical
pair; there is no new location, transmission, runtime API or device-selection policy.

1. **Placement honesty:** each route feeder must be mapped before reconstruction consumes
   it. Tiny674 and original5907 prove the old SCC-only region omitted required acyclic
   work. Existing decide/advance target facts now include it before `loop_exit`.
2. **Transmission well-typing:** no physical transmission or new ABI is introduced.
   `FnBuilder` reconstruction and independent `validate()` preserve the typed graph.
3. **Placement totality:** the existing `is_canonical` gate and whole-graph identity
   fallback for noncanonical loops are unchanged; existing nested-loop tests pass.
4. **Dependency mediation:** canonical driver membership and the same global object
   order preserve feeder-before-consumer construction. The existing done/remap maps
   avoid duplicate reconstruction; no independent loop ownership analysis is added.
5. **Composition soundness:** existing `reconstruct_a` preserves alias/drop/internal-pack
   and constify handling. Five new negative-controlled tests force empty-plan replay and
   check all six passes plus the full pipeline, effects, zero/multiple trips, computed
   exit values, real constify plans, sequential ownership and guarded traps. Full rewrite
   validation is86/0; complete original checkpoint interpreter results match.
6. **runsAt relation:** no new placement alternative is selected or removed. Backend
   emission is downstream of the same repaired graph transformation. Eight previously
   failing GPU emission configurations complete; no GPU execution is claimed.

Shared required-clang validation passes **1,181/0/1 ignored**, formatting passes, and all
workspace release examples build. The first sandboxed run failed only three native
hardware-detection tests, with direct `sysctl` permission-denial evidence; the unchanged
source passes the complete suite with host access. See the
[workspace receipt](../../../../benches/results/gpu/validation/loop-route-replay/evidence/shared-workspace-verification.json).
No numerical relaxation, observation removal or speculative performance fix was applied.

The implementation map was reconciled first, then the existing authoritative DESIGN.md
(the component has no separate ARCHITECTURE.md), then component/root status. Original
native Exact/Contract and one no-tile emission timeout remain documented and unresolved.
