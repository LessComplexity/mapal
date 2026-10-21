# Component: backend-nvptx

Status: **built inside the LLVM backend; final S54 validation matrix passed; targeted ordering remains open.**
Last updated: 2026-10-21 · S57 frozen validation complete; generic proof work continuing.

The S41 packaging choice is realized as the GPU leg of `mapal-backend-llvm`, with the
CUDA-driver runtime in `mapal-rt`. The component has no separate backend crate.
[ARCHITECTURE.md](ARCHITECTURE.md) owns the placement rules;
[IMPLEMENTATION.md](IMPLEMENTATION.md) maps their Dat/Trn/Loc/Trm to current code.
This replaces the obsolete S41 “planned — no code” status.

## Built

- GPU profiles, device LLVM/PTX generation, tiled and tensor-core realizations, host launch
  glue and CUDA-driver transmissions are present. The implementation map names their seams.
- S54 placement composes epilogues, cell kernels, row folds, residency/death, retained rows,
  delta uploads and independent streams in one resident plan. Grid raster and L2 windows
  select existing realizations; they add no numerical semantics.
- The per-profile `GPU_RUNGS_OFF` table and forced measurement arms are present. Table
  presence does not establish that every selection is accepted.
- Current smoke execution checks values and expected traps before timing. Remote failures
  preserve results and propagate failure through the rental wrapper, which destroys its instance.

Dependencies: IR geometry, bounds, loops, uses and task dependencies; LLVM host/device emission;
GPU runtime transmissions. Device capabilities and selection remain backend facts.
FRAMEWORK §4.5 laws 1–6 and ADR-0020's differential duty apply across these boundaries.
A staged shared-memory panel is readable only after its barrier completes the transmission.

## Current evidence

- Host validation after the recovered guarded-ConstFold repair: two required-clang workspace
  runs, each **1,176 passed / 0 failed / 1 ignored**; all release examples build. See
  [S56 validation baseline](../../sessions/2026-10-21-s56-validation-baseline.md).
- Fresh RTX 5090 smoke: **66 value passes / 14 expected-trap passes / 0 failures**. Source,
  binary and bundle hashes plus rental cleanup are recorded in
  [the smoke manifest](../../../benches/results-s54-gpu/s56-smoke-rtx-5090.json).
- Historical `integrated-*-prefix.txt` files are complete runs made **before** the arena repair,
  and contain failures. They do not certify the current integrated build.
- Selection recovery is complete at source `b4152c5`: RTX 5090 streams OFF; H100 retention
  and L2-window OFF; the RTX 5090 derived L2 gate restored. The two independent 5090 hosts
  favor streams OFF in all 32 changed-cell comparisons with disjoint iteration ranges.
  H100 and raster choices follow the declared conservative acceptance rule; the report
  preserves traced-kernel and TF32 counterevidence. See
  [per-architecture selection](../../performance/s54-per-arch-selection.md).
- All three selection runs passed complete timing, value, resource and nsys inventories,
  and their rentals were independently confirmed absent. The H100 host repeats a previously
  measured physical host. Older selection runners omit some process-status fields, explicitly
  disclosed in their receipts; final bundles enforce those fields.
- The final source passes required-clang workspace tests: **1,176 / 0 / 1 ignored** in 76.5 s.
  All five final bundles are built. RTX 3090, A100, RTX 4090, H100 PCIe and RTX 5090 primary runs
  pass all required result/profiling/cleanup gates. The independent RTX 4090 confirmation
  also passes and is cleaned up; its favorable FMA4096 median ratios repeat, while the
  FMA8192 win does not. H100 PCIe passed all 812 differential cases and 22 traps,
  every measurement/profiling gate, and independent cleanup at 20:37:43 UTC; its
  7,956 exported numeric fields and 4,140 raw kernel samples independently check.
  The replacement RTX 5090 confirmation also passes all gates and was independently
  absent at 20:55:19 UTC. Its 8,050 numeric fields and 4,320 raw kernel samples check.
  Streams OFF has lower medians in all eight changed cells, with six disjoint ranges
  and two overlaps. Host timing differences remain unexplained; they are not attributed
  to driver or power settings without evidence.

## Reconciliation and targeted proof still open

Completed validation: [verify-s54-gpu-selection](../../../openspec/changes/archive/2026-10-21-verify-s54-gpu-selection/proposal.md).
Generic ordering/allocation and composed-workload plans remain active.

1. The final five-profile matrix and both independent confirmations are complete and
   all corresponding task rentals are independently absent.
2. Both S54 reports and the §4.5 review are reconciled; drift passes and the validation
   change is archived. The uncovered composition duties below remain explicit.

Separate proof duty discovered during validation: [stored-member ordering](../../performance/s57-ordering-evidence.md).
Frozen emitted LLVM demonstrates missing explicit producer edges in host-readback and
different-chain tile cases. GPU consequences are not yet tested. The §4.5 review leaves
those composition/readiness duties open; the benchmark matrix must not be described as
proof for all programs. No ordering fix is applied. The reviewed isolated candidate has
four old-code regression failures, two passing controls, 50 passing focused checks and
14 unchanged device LLVM units; runtime/API completion proof is recorded. Both approved
ordinary GPU attempts failed at allocation before upload and are cleaned up. No further
attempt is authorized. Allocation elimination remains deferred; the arena guard stays.

The [large/composed preparation](../../performance/s57-large-composed-evidence.md) adds
96 successful temporary primitive emissions and small arithmetic certification.
The RTX 5090 16K/rectangular extension validates 98/100 invocations; both failed arms
are the same requested-TF32 fallback and remain rejected. All six 16K base/repeat
sampled value gates pass. This is separate from the frozen matrix.
The recurrent checkpoint replay panic is now repaired through existing loop-driver facts.
All eight previously failing GPU emissions complete and full interpreter checkpoints
match; actual composed GPU correctness/timing remains open. Native large emission still
reaches a bounded timeout. The required-clang workspace passes 1,181/0/1 ignored.

## Continuing performance work

The user authorized a generic compiler phase after frozen validation: use derived
dependency, lifetime and placement facts across algorithms, starting with allocation
under streams and evaluating explicit graph construction afterward. Plan it through
OpenSpec and preserve guards, carries, aliases and host observations as proof boundaries.
The [current comparison](../../performance/s54-current-library-comparison.md) and
[integrated report](../../performance/s54-gpu-fusion.md) distinguish kernel, program,
retention and allocation effects across families. Exact-face and Ampere/Hopper TF32
library gaps remain separate from placement acceptance. ADR-0039's precision policy remains
an explicit decision for the user; this validation does not change it. The older clock-fence
helper and eager-init race questions also remain open until independently checked.

Plans of record: [S41 NVPTX leg](plans/plan-s41-the-nvptx-leg.md),
[S54 placement](plans/plan-s54-fusion-placement.md). Historical measurements and decisions
remain immutable in [the S54 handoff](../../sessions/2026-10-21-s54-fusion-placement.md).
