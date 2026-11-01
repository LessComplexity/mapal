## Why

The backend already joins affine dataflow, ordered reductions and hardware resources to derive GPU placement. A code-backed inconsistency remains: per-site policy is keyed by requested arithmetic permissions, while strided accesses and unsupported tensor routes can emit a different arithmetic realization. The compiler should select policy for the kernel it can actually emit.

## What Changes

- Derive one typed actual arithmetic realization on the existing GPU launch plan from source/layout facts, arithmetic permissions, tensor feasibility and hardware capability.
- Use that fact consistently for per-site policy selection and scalar step emission; resolve rejected tensor attempts through the scalar policy.
- Preserve ordered reductions, operand order, existing geometry derivation, forced-arm precedence and module-wide host/device dependent-launch agreement.
- Add negative/boundary and emitted-code checks proving the inconsistency before correction and the shared invariant afterward. Retain existing successful scalar/tensor control emissions where their realized policy already agrees.
- Record measured performance and new GPU validation separately; this correction carries no speedup or global-optimality claim.

## Capabilities

No external numerical contract, language syntax, CLI, runtime ABI or new permission changes. This is an internal selection-consistency correction; specs are explicitly skipped. General schedule profitability remains governed by the existing measured policy and future controlled experiments.

## Impact

`crates/backends/llvm/src/func/gpu.rs`, focused backend tests, backend-nvptx architecture/implementation/status and performance evidence. Device facts remain outside shared IR. Existing small-M experiments remain isolated; their measured regressions preclude blanket promotion.
