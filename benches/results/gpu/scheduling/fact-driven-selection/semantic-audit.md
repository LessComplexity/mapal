# Compile-time GPU fact-flow audit

Read-only audit; no production edit or test run. Semble MCP focused searches for TileSite→GPUlaunch and arithmetic/reduction permissions stalled for several minutes without output and were terminated. Reads degraded to root-provided known paths and current implementation-map links. Root owns OpenSpec `derive-gpu-schedules-from-facts`; this is code evidence for its review, not a separate plan.

## What already exists

`mapal-ir/src/algo.rs:TilePlan`248 and `TileRead`255 carry geometry independent of target: operand slots, constant offset, row/lane/reduction strides and optional derived reduction axes. `TileSite`403 adds rows/c/k, seed/element and original multiply/add operand order. `tile_site`1361 admits one guarded-free Map containing one Fold; `tile_fold_shape`1515 recognizes one accumulator Add and one product Mul, traces both affine reads, retains source operand order and normalizes operand roles. Strided extra lane sites are restricted through the recognizer at1619–1625 and GPU `tile_plan`202. These facts are present: lack of generic GEMM geometry is not the explanation.

`profile.rs:Gpu`125 carries warp, SM count, register/thread limits, shared-memory limits and L2. `TargetProfile::gpu_tile`2129 computes register microtile, thread-grid/resource budgets, shared panel depth and divisibility legality. `gpu.rs:launch`270 joins those dimensions with affine separability (`a.clane==0 && b.ci==0`), vector/alignment facts and hardware choices; preliminary geometry derives Fill before `rungs_at`284, then geometry is queried again with selected rungs. `profile.rs:gpu_rungs_on`1747 is still a measured OFF table keyed by profile/architecture, Fill and Face. This is an existing combination of derivation and measured selection, not a general schedule optimizer.

The emitted semantics explicitly preserve each cell's increasing-k chain (`gpu.rs`14–20; `Kernel::step`1516). Exact emits separate mul/add in original operand order1532/1538. Ordered contraction emits one llvm.fma1522. `EmitOpts::contract`lib.rs430 grants single-step FMA contraction; `Reduced`405 explicitly permits TF32's value change. Neither grants reduction reassociation, split-K, a tree sum, narrower storage or an arbitrary numerical budget. No new permission may be inferred from more hardware facts.

Existing dataflow placement also consumes real facts: `resident.rs:epilogues`698 joins CellSite footprints, exact outside readers, source/topological readiness and loop region to producer stores. Last-use/partition/retention/stream mechanisms are mapped in backend-nvptx/IMPLEMENTATION.md. They should not be described as absent.

## Concrete new propagation defect

`gpu.rs:launch`284–288 keys per-site rung selection with `Face::of(contract,reduced)` from global requested flags. `kernel_text`990–994 separately narrows actual scalar contraction to `contract && site.b.clane==1`: strided-lane sites retain the host's separate multiply/add even under requested Contract. Therefore their emitted scalar arithmetic is Exact while its site policy can be selected using Contract rows. `profile.rs:Face`315–344 documents the key as the face of the kernel actually emitted, making this a mismatch between existing data and its consumer, rather than a speculative missing optimization.

A second instance is requested TF32 followed by actual scalar fallback: `launch`292–297 and downstream MMA/WGMMA geometry admit tensor kernels only on supported type/access/hardware/shape. Requested Reduced alone still selected Tf32 at287 even when neither tensor realization survives and `kernel_text`1013–1016 emits scalar. Permission and actual realization need distinct representations.

Minimum correction: deduce a typed actual arithmetic realization on existing backend `Launch` (or equivalent derived local datum), with separate scalar OrderedFma/SeparateMulAdd and actual selected TF32 MMA/WGMMA cases. A single scalar eligibility helper must feed both selection and emission. Derive Face from the realized step/kernel; explicitly resolve tensor rejection to scalar policy without inventing reduced precision or a new permission. This adds no machine fact to shared IR and changes no source fold ordering. It is a selection-consistency correction; no speed benefit is established.

PDL must keep its separate module invariant. `gpu.rs:pdl`96–97 uses global `host_rungs(contract)`; kernel_text997 calls that same function. Runtime init uses the same module policy, so a dependent launch always has its matching kernel wait. Changing only the kernel to a per-site arithmetic face could break this relation. Keep module PDL/host residency policy consistent or introduce a separately proven module-wide derivation; do not conflate it with per-site arithmetic selection.

## Explicitly rejected repeated proposal

The thread-grid fitting hole at `profile.rs`2175–2226 is real but is already the prior `optimize-ordered-smallm-gpu` candidate. Its design records wide M16 gains4.82–5.18× alongside smaller-family regressions37–47%, plus918000 legality comparisons and selection/Fill coupling. Rebalancing ty/tx again is not a new missing mechanism or an accepted production fix. No performance-based proposal follows from this audit. Generic geometry/resource costs remain a separate hardware-agent question with complete candidate/face/Fill context and measurement duties.

## Focused legality/emission checks for the correction

- One-row strided lane site, Contract requested: effective scalar step SeparateMulAdd, Exact-keyed per-site rungs, emitted separate fmul/fadd, increasing-k recurrence and original operand order unchanged.
- Ordinary unit-lane site, Exact and Contract: unchanged separate/FMA respectively; emitted face equals policy face on both scalar-naive and cooperative routes. Operand reversal and nonzero seed controls preserve the ordered contract.
- Reduced requested with unsupported architecture, access/vector alignment or tensor geometry, and explicit MMA OFF: scalar fallback's effective mode and rungs agree; no tf32 conversion/mma appears. Supported TF32 path still emits its selected tensor realization and Tf32 policy.
- Forced rung ON/OFF precedence survives, and preliminary Fill uses the unchanged legacy geometry until separately reviewed.
- Mixed strided/unit-lane module: each per-site face is correct while module PDL flag and all kernel griddepcontrol waits remain mutually consistent. Host residency/ordering/allocation policies stay at their existing module scope.
- Existing scalar/strided/TF32/epilogue/row-fold emission and differential tests provide conformance; no broad GPU timing claim until new exact-image ordinary gates and actual distributions exist.

## Architectural mapping

FRAMEWORK §3 and §4.5: extend existing Launch rather than invent a parallel schedule record. Existing Dat TileSite/TileRead and backend Gpu/EmitOpts supply source facts/device facts/permissions. Planned deduced Dat actual arithmetic realization belongs to backend Launch, not shared IR. Existing Trn launch resolves placement; Kernel::step realizes ordered arithmetic; Face/rungs_at consume the chosen mode. Existing Loc Reg/Smem/Gmem and Trm panel copies/stores/runtime launches remain unchanged. Composition soundness requires selection's arithmetic mode = emitter's mode, and module PDL hostflag = matching kernel wait. GPU geometry optimization beyond this correction remains an explicit open question, not an asserted speedup.
