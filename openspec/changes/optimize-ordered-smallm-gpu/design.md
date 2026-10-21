## Context
See proposal.md for measured motivation. `TargetProfile::gpu_tile` derives a square register/thread geometry, then reduces total threads only while the grid underfills SMs. Its final row-divisibility check rejects skinny M16 sites even when the existing cooperative emitter could cover them. Semble MCP retrieval exceeded a bounded wait and was terminated; known `func/gpu.rs` and literal symbol lookup located the implementation.

## Goals / Non-Goals
Fit existing cooperative panels to row extents using shape and hardware facts. Preserve M1 fallback, M128/square controls, all ordered reductions, precision and storage semantics. No split-K, new kernel emitter, named workload gates or rounding changes.

## Decisions
Extend `GpuTile` placement geometry rather than introduce another representation (FRAMEWORK §3). Preserve the original wide fallback and occupancy shrinking unchanged. Only afterward, when positive rows are smaller than `ty*tm`, halve `ty` and double `tx` until the tile fits or `ty == 1`. This preserves the final thread count and all existing register microtile, divisibility, whole-copy and shared-memory checks. Every previously successful positive-row geometry necessarily has `bm <= rows`, so the adjustment is a no-op; `rows == 0` explicitly bypasses it. No mirrored-column feature is included. M1 cannot satisfy the existing vector-wide row microtile and retains fallback.

No Dat or Trn semantic model delta. Existing `Dat`: operand panels and ordered accumulators. Existing `Trn`: `Kernel::step` and `Kernel::tiled`. Existing `Loc`: global memory, shared panels, thread registers. Existing `Trm`: panel cooperative copies and output stores. Planned placement morphism `gpu_tile? : Shape × DeviceFacts → GpuTile` is partial: returns no geometry unless every existing legality/resource check passes. Its only change is thread-grid shape.

FRAMEWORK §4.5: placement honesty requires every shared operand to be loaded before the existing barrier; transmission well-typing is unchanged because emitter/types stay shared; placement totality requires positive geometry and complete tile coverage; composition soundness requires bitwise ordered folds and unchanged epilogues. Implemented mappings remain `profile.rs:TargetProfile::gpu_tile`, `func/gpu.rs:Kernel::tiled`, `Kernel::step`, and `lane_maps`.

## Risks / Trade-offs
- Skinny grids may lower reuse or change occupancy → actual before/after GPU timing and resources decide acceptance.
- Newly selected rungs may exercise a different emission combination → preserve LLVM/PTX and compare exact ordered values for Exact/Contract plus bounds/aliases/epilogues.
- M1 stays scalar → explicitly report it as a boundary control, with no speedup claim.

## Migration Plan
First add a test that fails on the frozen snapshot, then apply the isolated geometry edit and run CPU/kernel proofs. Parent reviews and integrates the patch before actual integrated GPU comparison; revert only this change if gates fail. Root owns reconciled component/root docs, §4.5 review and drift/archive.

## Rejected bounded candidate
Preserving thread count by doubling `tx` when halving `ty` was tested first and failed the existing whole-copy/shared-capacity checks on A100 M16×18432×7168. After occupancy shrinking, bm8/bn256 needs bk16 for every thread to load whole A vectors, exceeding its shared-memory share. Halving `ty` alone yields bm16/bn128 and legal cooperative panels. The final candidate explicitly retains at least one warp or falls back.

## Rejected second candidate and narrower correction
Pre-fill halving of `ty` without increasing `tx` passed the target M16 tests but failed a broader 2,440-case geometry inventory: 135 previously successful geometries changed and 75 became unavailable. This candidate is rejected. The final candidate keeps the original occupancy derivation first, then redistributes only an oversized positive-row grid while preserving its final thread count. A diagnostic baseline/candidate inventory covers boundary rows 4/8/16/32/48/64/96/128, all three N/K families and transposed dimensions, f32/f64, wide/fill, tiny/zero/ragged controls and five profiles. The zero-row guard preserves the original behavior without claiming the caller supports empty GPU work.
