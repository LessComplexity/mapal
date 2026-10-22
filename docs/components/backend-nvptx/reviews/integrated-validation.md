# FRAMEWORK §4.5 review integrated validation

Historical provenance: S57.

This is a local coherence review alongside the **completed frozen GPU validation matrix**. A subsequent frozen-emitter
investigation found missing explicit ordering for some stored epilogue outputs; those duties
remain open below. The separate targeted ordering investigation remains open; finite benchmark coverage
does not discharge every composition duty.

The [ordering evidence report](../../../performance/gpu/memory/ordering-investigation.md) links the frozen
LLVM fixtures, interpreter oracles and unchanged-source receipts. It includes refuted
same-stream/row-fold concerns and distinguishes missing emitted edges from untested GPU
wrong-value consequences. No speculative source fix has been made.

- [ ] 1. **Placement honesty.** Device readers require complete device operands;
   `resident.rs:FnEmit::emit_site_resident` emits cross-chain waits before a reader's
   launch, while `gpu.rs:FnEmit::emit_tiled_map_gpu` and `resident.rs` supply the
   existing host/device paths. `tests/streams.rs:diamond_waits_on_its_join_edge_only`
   pins the dependency before its consuming launch; `tests/resident.rs` pins copy
   placement and element reads. However, fresh emitted LLVM places a stored epilogue
   member's writer on stream2 and its first readback on the legacy path with no intervening
   runtime call. Existing tests do not cover that composition. Array presence alone does
   not establish readiness; the targeted evidence and correction remain open.
- [x] 2. **Transmission well-typing.** Copies/stages/stores cross the locations named in
   ARCHITECTURE §4; launch carries Host-to-device control. Same-`Gmem` readiness is
   ordering/control, not another array transmission. `mapal_gpu.c:event` records and
   waits on CUDA streams without copying a buffer. `tests/retain_delta.rs:
   delta_upload_is_stride_times_elem_bytes` pins row extents and offsets;
   `tests/resident.rs:l2_window_covers_the_chain_intermediate_then_the_most_read_operand`
   distinguishes cache policy from data movement.
- [x] 3. **Placement totality.** `ops.rs:FnEmit::emit_morphism` falls back to `emit_op`
   whenever no resident replacement is selected. The existing host realization
   remains available; rejecting a device shape does not invent a numerical rule.
   `tests/retain_delta.rs:cpu_profiles_are_untouched` and
   `site_produced_carry_is_top_and_byte_identical` cover important fallback boundaries.
- [ ] 4. **Dependency mediation.** `mapal_gpu.c:mapal_gpu_h2d`, `mapal_gpu_d2h` and
   `mapal_gpu_d2h_on` mediate location-crossing data dependencies. Same-stream order
   and `mapal_gpu_event` mediate shared-`Gmem` producer/consumer readiness;
   `launch_on` selects PDL launch attributes and `gpu.rs:Kernel::header` emits the
   corresponding device dependency wait/release. The revised §4/R3/§6
   and IMPLEMENTATION ordering row preserve this distinction. A second frozen-emitter
   fixture shows a stored member feeding a different chain with no emitted event; its
   intervening allocation calls use already-populated slots. Runtime consequences remain
   untested. Tracing synchronizes `launch_on`, so traced kernels cannot certify untraced
   stream overlap or rule out an ordering defect.
- [ ] 5. **Composition soundness.** `core.rs:FnEmit::set_gpu_arms` forms epilogue groups
   before residency; `resident.rs:Member::store` records outside-needed results.
   Existing `tests/epilogue.rs:fanout_stores_the_shared_output_once`,
   `mlp2_two_groups_store_only_what_the_next_site_reads` and
   `a_launch_never_passes_two_objects_on_the_same_arena_bytes` pin composition,
   surviving outputs and overlapping-lifetime safety. These checks do not establish
   stored-member readiness after fusion and stream placement. That composition is now
   a targeted open duty. The five primary profiles now pass their declared numerical
   matrix; that finite coverage does not close the uncovered ordering cases.
- [x] 6. **`runsAt` is a relation.** Host, epilogue and separate device-cell realizations
   remain alternative placements of existing bodies (`gpu.rs:device_module`,
   `resident.rs:FnEmit::cell_sites`). `tests/retain_delta.rs:
   resident_path_matches_the_oracle_emulated` and
   `int_decode_matches_the_oracle_on_and_off` exercise alternative realizations
   against an oracle; hardware differential coverage is still required.

All Rust paths above are under `crates/backends/llvm/`; `mapal_gpu.c` is
`crates/mapal-rt/gpu/mapal_gpu.c`. Test names identify existing coverage, not a claim
that a new GPU execution occurred during this review.

The harness changes add validation records, no runtime Dat/Trn/Loc/Trm. In
`benches/gpu/box_run.sh`, e2e/probe `rc` is the ordinary process status and `trace_rc`
the traced status; decode retains traced `rc` and adds ordinary `wall_rc`. Each
record is written before a nonzero status fails the run. Shell `pipefail` preserves
the ordinary program's failure through `tr`. `benches/gpu/test_harness.py` tests
plausible-output failures in either phase and remote result retrieval before cleanup.

Local evidence: H100-focused profile **43/43**, resident **13/13**, retain-delta
**14/14**; final 5090 streams **11/11** and raster **3/3**; independently rerun stub harness **12/12**. Logs:
`/private/tmp/mapal-s56-gpu/h100-profile-tests.log`,
`h100-resident-retain-tests.log`, and `coherence-harness-tests.log`.

**Five primary hardware gates passed:** RTX 3090, A100, RTX 4090, H100 PCIe and RTX 5090 have
inventory-exact values/traps, timing/status/library/resource/profiling evidence and
independent rental-absence receipts. The integrated report preserves every family,
loss and null anomaly; a separate checker verified 38,577 numeric fields and 19,800
kernel samples. These passes do not close the targeted ordering duties above.

The independent RTX 4090 confirmation also passes, with a distinct physical machine,
complete gates and confirmed cleanup. Its 7,494 numeric fields and 3,780 raw kernel
samples independently check. Favorable FMA4096 median ratios repeat; the FMA8192 win
does not. The report preserves that counterevidence and the host/power differences.

**Independent RTX 5090 confirmation passed:** its complete gates, distinct machine and
independent cleanup are recorded. All 8,050 exported numeric fields and 4,320 raw kernel
samples independently check. Streams OFF has lower medians in all eight changed cells;
two ranges overlap. The report preserves the larger TF32 gap and host timing differences
without attributing a cause. Across all seven runs, 54,121 numeric fields and 27,900 raw
kernel samples check. These benchmark gates do not prove generic asynchronous allocation
safety or close the targeted ordering duties above.

## Model reconciliation

The [independently reviewed disassembly comparison](../../../performance/gpu/disassembly/cublas-cutlass-gap.md)
binds exact library profiler/fatbin names and distinguishes offline Mapal assembly
from its unsaved driver-JIT image. It establishes small-M route and TF32 conversion
differences, not causal performance fixes. Current `Reduced::Tf32` requires `.rna`;
changing its tie rule is a semantic decision. Kernel library errors are diagnostics,
not independently passed accuracy gates. Frozen host LLVM separately establishes
CPU vectorization around preserved GPU calls, with no claim of measured overlap.
These findings add evidence and open questions, no new runtime Dat/Trn/Loc/Trm.

No runtime objects, placements or transmissions were added. The selected rows choose
existing realizations. The implementation map now records the built LLVM GPU leg,
and the architecture corrects same-Gmem ordering to control rather than a transmission.
Derived IR geometry remains separate from backend device facts; the root map points to
these component rows instead of duplicating them (FRAMEWORK §§3, 4.3, 5).
