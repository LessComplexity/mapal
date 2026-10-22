# Review — realized arithmetic drives GPU policy

Reviewed the implemented `derive-gpu-schedules-from-facts` correction against FRAMEWORK §4.5. Production scope is `crates/backends/llvm/src/func/gpu.rs`; shared IR, target resource formulas and runtime interfaces are unchanged.

## Coherence laws

- [x] 1. Placement honesty — `Launch::arithmetic` describes the selected scalar or tensor realization. Strided Contract and rejected TF32 no longer select a policy for arithmetic absent from the kernel.
- [x] 2. Transmission well-typing — panel copies, argument types and stores retain their existing paths. The selected arithmetic is consumed by device emission; tensor rejection is resolved before scalar emission.
- [x] 3. Placement totality — unsupported tensor type, layout, alignment, geometry, architecture and forced-OFF cases resolve through the existing scalar path; focused tests cover these partiality boundaries.
- [x] 4. Dependency mediation — module PDL remains derived by the shared `pdl` function for host flags and kernel waits. A real mixed scalar/tensor module preserves both waits and matching host flags; no stream/arena interface changed.
- [x] 5. Composition soundness — `Arithmetic::scalar` is the single eligibility derivation and `Arithmetic::face` supplies the matching policy key. Scalar emission consumes that result. Original fold order, seed and operand semantics remain the existing behavior.
- [x] 6. runsAt is a relation — existing scalar/tiled/MMA/WGMMA realizations remain available under their legality gates; permission to use a route is distinct from its actual selection.

## Model delta and evidence

Existing `Dat Launch` gains `Dat Arithmetic`; `Trn launch` resolves it and `Trn kernel_text`/`Kernel::step` consume it. The component implementation map and architecture §9 record this built connection. There is no parallel scheduler record, duplicate shared geometry or new numerical permission. Legacy preliminary Fill and intentional warp-fill budgeting remain unchanged.

Two original-API regressions fail before the correction after confirming the emitted arithmetic. All 65 library and 176 integration tests pass, with one ignored optional baseline. Eight successful controls preserve host LLVM, device LLVM and PTX bytes; five affected controls preserve arithmetic inventories and dependent-launch wait counts. Optimized LLVM comparison normalizes source paths only. The independent reviewer rehashes 104 emitted files and reports no blocking finding.

See the [verification report](../../../performance/gpu/disassembly/fact-driven-scheduling.md) and its raw receipts. The first restricted native-detection failures and the integration run's unset forced-clang guard are disclosed there. No new GPU runtime numerical, ptxas/SASS-resource or timing validation is claimed. Broader schedule profitability is an open performance question, not a failed invariant of this correction.
