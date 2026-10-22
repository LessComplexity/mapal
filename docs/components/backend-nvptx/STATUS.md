# Component: backend-nvptx

Status: **built inside the LLVM backend; S54 validation passed; S58 ordering correction integrated and checked.**
Last updated: 2026-10-21 · S60 parallel preparation reviewed; GPU run awaits budget approval.

Standalone [image-observation tooling](../../../benches/gpu/capture_images/README.md)
is saved with real CUDA/CUPTI 12.8 header compilation, eleven local test groups and
independently reproduced context/geometry rejections. Linux injection and actual executed
cubin/SASS capture remain unvalidated; this does not close the S57 disassembly gap.

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

The [stored-member ordering investigation](../../performance/s57-ordering-evidence.md)
proved missing explicit producer edges in host-readback and different-chain tile cases.
The [S58 correction](../../performance/s58-parallel-development.md) is now integrated:
four old-code regression failures, two passing controls, 50 focused checks and 14 unchanged
device LLVM units support the runtime/API completion proof. A renewed RTX 5090 attempt
passed 200 ordinary processes on the frozen revision and 200 on the correction, with
independent raw-output validation after retrieval. Both passed; there is no observed race
failure or measured speedup. Required-clang workspace: **1,188 passed / 0 failed / 1 ignored**.
Allocation elimination remains unintegrated; the arena guard stays. S58's isolated
`chain_prologue` candidate reuses selected-group `store_out` to omit only raw storage
already unused by the launch. Its old-code regression fails, 51 focused and 59 backend
library tests pass, and all 28 emitted device LLVM units match. The isolated full
required-clang workspace passes 1,190 tests with zero failures and one ignored. Surviving host bodies
preserve allocation positions and dependencies. Its own ordinary GPU gate remains
required because an omitted allocation also removes an incidental wait in its free.
See the [candidate evidence](../../performance/s58-parallel-development.md).

The [large/composed preparation](../../performance/s57-large-composed-evidence.md) adds
96 successful temporary primitive emissions and small arithmetic certification.
The RTX 5090 16K/rectangular extension validates 98/100 invocations; both failed arms
are the same requested-TF32 fallback and remain rejected. All six 16K base/repeat
sampled value gates pass. This is separate from the frozen matrix.
The recurrent checkpoint replay panic is now repaired through existing loop-driver facts.
All eight previously failing GPU emissions complete and full interpreter checkpoints
match. S58 small-matrix correctness passes for Mapal, handwritten CUDA/cuBLAS and PyTorch
eager. The first compiled PyTorch attempt stopped at missing `Python.h`; after an explicit
development-header installation, a later run passes. Root rechecked its 1,024 checkpoint
and 128 final values against both frozen numerical budgets. This is small-matrix
correctness, not a composed timing result. Native large emission still reaches a bounded timeout.

The [S59 comparison](../../performance/s59-smallm-comparison.md) completes 234/234
records and admits 28/30 timing cells. Wide M16 Exact/Contract/TF32 improve 4.82–5.18×;
smaller M16 Exact/Contract regress 37–47%, beyond observed null/control variation.
Two M1 TF32 gates fail in both revisions. The blanket candidate is not promoted.
Root rehashed 948 files and recomputed 10,080 raw CUDA-event samples. Runtime geometry
and resources are recorded; actual executed SASS/counters remain missing. Instance 57130775
is independently absent. Shared source, allocation behavior and the arena guard are unchanged.

[S60 parallel preparation](../../performance/s60-parallel-preparation.md) now has two
resident-input composed MLP packets with matched clock ordering, explicit numerical
coverage and independent readers. Two isolated geometry alternatives each pass 918,000
comparisons; the caller proof shows why changing geometry admission can alter the
preliminary fill key and selected arithmetic route. Neither is promoted. Allocation,
optional kernel comparison and executed-image packets are sealed with real local
cancellation/retrieval checks. No S60 rental/upload or GPU result exists; the explicit
budget-extension request is pending. Recurrence/retention and broader controls remain open.

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
