## Context

`gpu.rs:launch` currently derives a legacy base Fill from `TileSite` and `TargetProfile::gpu_tile`, then keys `rungs_at` with global `Face::of(contract,reduced)`. `kernel_text` independently constrains scalar contraction to `contract && site.b.clane==1`; tensor type/layout/architecture/geometry restrictions can also reject requested TF32 after its policy was selected. `Face` documents the kernel's actual arithmetic, so this is a proved mismatch. Read-only evidence is retained in the compiler semantic audit and existing IR fact map.

## Goals / Non-Goals

Goals: derive the legal realized arithmetic from existing facts before applying its policy; share it with the emitter; make fallback explicit and deterministic. Preserve source fold order and host/device execution agreement.

No new reduction reassociation, numerical tolerance, precision permission, hardware fact in IR, geometry formula, profitability threshold or global-optimality guarantee. Existing small-M tile fitting regressed measured cases and is not promoted. The current `regs_est` after warp-fill shrinking is a documented conservative policy, not an assumed bug.

## Decisions

Extend existing `Dat Launch` with a typed actual arithmetic realization, rather than create a parallel scheduler record (FRAMEWORK §3). Inputs remain existing `Dat TileSite/TileRead`, backend emission permissions and `Dat Gpu/TargetProfile` facts. A scalar eligibility derivation determines separate multiply/add versus ordered FMA from the same source/layout condition currently used by emission.

`Trn launch : TileSite × TargetProfile × Permissions × Arms → Launch` first preserves the legacy preliminary Fill classification. Where TF32 permission and type/layout/device eligibility hold, it attempts the existing tensor realizations under their TF32 policy. If neither tensor realization is available, it selects the scalar policy using the derived scalar arithmetic and derives existing scalar geometry/staging from that policy. A successful tensor plan retains its tensor policy. Explicit forced arms keep their existing precedence and cannot create an otherwise illegal tensor path.

`Trn Kernel::step` consumes the selected launch's scalar arithmetic fact rather than rederiving it from global flags. Increasing-K recurrence, source multiply/add operand order, seed and epilogue behavior remain invariant. Existing `Loc Reg/Smem/Gmem` and `Trm` panel copies, stores and runtime launch interfaces are unchanged.

Module-wide `pdl(profile,contract,arms)` remains shared by runtime initialization and every emitted kernel wait. It is a module execution contract, distinct from per-site arithmetic policy. Do not switch only one side to a site-specific face.

FRAMEWORK §4.5 composition soundness: policy face equals actual realization; scalar fallback never grants reassociation or reduced precision; transmitted operands and geometry still satisfy existing layout/resource checks; module launch flags and waits agree. The planned realization is `gpu.rs:launch`, a small backend arithmetic type/helper, and `kernel_text`/`Kernel::step`. Root will map exact symbols after implementation.

## Risks / Trade-offs

- Policy changes on fallback sites can alter staging/resources → preserve numerical instructions and ordered recurrence, exercise emission/resource boundaries and record that GPU performance is unmeasured.
- Tensor selection and policy have a dependency → resolve it explicitly by tensor attempt followed by scalar fallback; do not iterate heuristics or change legacy Fill classification.
- Requested precision and actual arithmetic are different → retain the request for eligibility/kernel identity while the plan records the selected realization.
- PDL is module-wide → mixed-site tests check matching host flag/kernel waits.

## Migration Plan

Capture pre-fix failing policy/emission invariants on existing APIs, implement the shared fact, run focused and existing GPU emission/semantics tests, independently review the diff and emitted controls. Reconcile architecture/implementation/status and preserve exact evidence. Fresh GPU numerical/timing validation remains separate; report the absence of a measured speed gain.
