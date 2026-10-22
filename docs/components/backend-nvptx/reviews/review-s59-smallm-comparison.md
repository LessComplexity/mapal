# S59 evidence reconciliation — FRAMEWORK §4.5

No production compiler change is promoted. Read IMPLEMENTATION, then ARCHITECTURE,
then reconcile component/root STATUS against the measured report. No mapping, atom,
port, runtime ABI or architecture diagram change is needed.

Existing model: Dat operand panels/ordered accumulators; Trn `Kernel::step` and
`Kernel::tiled`; Loc Gmem/Smem/Reg; Trm cooperative panel copies and output stores.
The isolated placement candidate changes `profile.rs:TargetProfile::gpu_tile`,
using the existing `func/gpu.rs` realizations. Its proof and patch remain in S58.

- Placement honesty and transmission typing: existing panel/barrier/type mechanisms
  are unchanged. S59 admits 28 cells under recorded gates, not all shapes.
- Locality/coherence: no IR machine fact or precision-policy change; Exact/Contract
  preserve bounded ordered-fold gates. Two shared M1 TF32 failures remain excluded.
- Composition: this is a single-kernel experiment. It certifies no allocation,
  retention, multi-kernel completion or composed-PyTorch speedup.
- Totality and acceptance: 234 identities are complete. Four performance regressions
  reject blanket promotion despite numerical/protocol acceptance on admitted cells.
- Evidence boundary: root rehashes 948 artifacts and recomputes 10,080 samples;
  executed SASS and dynamic counters remain open. Geometry bounds do not prove a
  full causal explanation of the regressions.

OpenSpec remains active. The existing production model and implementation stay intact.
