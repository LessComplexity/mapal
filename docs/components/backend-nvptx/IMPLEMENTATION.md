# backend-nvptx — implementation map

Maps [ARCHITECTURE.md](ARCHITECTURE.md) §§2–7 to the GPU leg inside the LLVM backend,
not a separate backend crate (FRAMEWORK §4.2, §4.5, §6.3). Paths are repository-relative.
The mechanisms below are built; disputed per-architecture selections are resolved, while
the frozen S54 validation matrix and both confirmations pass. This map records code locations, not performance verdicts.
STATUS distinguishes finite benchmark acceptance from the targeted ordering proof still open.

## Dat — facts and placement records

| Architecture object | Realized at |
| --- | --- |
| Static arrays and tile geometry (§2) | `crates/mapal-ir/src/ty.rs:Ty`; `crates/mapal-ir/src/algo.rs:TileSite`, `crates/mapal-ir/src/algo.rs:TileRead` |
| Cell read footprints and row folds (§2) | `crates/mapal-ir/src/algo.rs:Footprint`, `crates/mapal-ir/src/algo.rs:CellSite`, `crates/mapal-ir/src/algo.rs:RowFoldSite`; deduced by `crates/mapal-ir/src/algo.rs:CategoryIr::cell_plan` and `crates/mapal-ir/src/algo.rs:CategoryIr::row_fold` |
| Bounds, loop state, changed rows, death and task dependencies (§2) | `crates/mapal-ir/src/algo.rs:CategoryIr::bounds_proof`, `crates/mapal-ir/src/algo.rs:CategoryIr::loop_plan`, `crates/mapal-ir/src/algo.rs:CategoryIr::write_delta`, `crates/mapal-ir/src/algo.rs:CategoryIr::last_use_plan`, `crates/mapal-ir/src/algo.rs:CategoryIr::path_plan` |
| Device facts and selection keys (§5, §7) | `crates/backends/llvm/src/profile.rs:Gpu`, `crates/backends/llvm/src/profile.rs:TargetProfile`, `crates/backends/llvm/src/profile.rs:Fill`, `crates/backends/llvm/src/profile.rs:Face`, `crates/backends/llvm/src/profile.rs:RungOff` |
| Rung switches and forced measurement arms (§7) | `crates/backends/llvm/src/lib.rs:GpuRungs`, `crates/backends/llvm/src/lib.rs:GpuArms`; table `crates/backends/llvm/src/profile.rs:GPU_RUNGS_OFF` |
| Per-function placement, groups, cells and carries (R1/R2/R5/R6) | `crates/backends/llvm/src/func/resident.rs:ResidentPlan`, `crates/backends/llvm/src/func/resident.rs:Epi`, `crates/backends/llvm/src/func/resident.rs:Cell`, `crates/backends/llvm/src/func/resident.rs:Carry`; `crates/backends/llvm/src/func/retain.rs:Retained` |
| Device arena and selected launch (R6, §7) | `crates/backends/llvm/src/func/resident.rs:Arena`; `crates/backends/llvm/src/func/gpu.rs:Launch` |

These are derived records over the existing IR and profile. Machine facts stay in the
backend; the geometry queries stay in `mapal-ir` (ARCHITECTURE §2; FRAMEWORK §4.5 law 5).

## Trn — placement and emission

| Clause / transformation | Realized at |
| --- | --- |
| Mixed host/device emission and current sequential host restriction (§8) | `crates/backends/llvm/src/lib.rs:Emission`, `crates/backends/llvm/src/lib.rs:emit_detailed`; realization ladder `crates/backends/llvm/src/func/tile.rs:FnEmit::emit_tiled_map_realize`; pre-emission rewrite `crates/backends/llvm/examples/emit.rs:main`; host/device compilation `benches/gpu/bundle.sh:emit_one` |
| Host helper effect attributes and targeted row-fold copy hint (§8) | `crates/backends/llvm/src/func/mod.rs:FnAttrs::analyze`; `crates/backends/llvm/src/func/drive.rs:FnEmit::emit`; `crates/backends/llvm/src/func/resident.rs:FnEmit::emit_site_resident` |
| R0: admit affine floating tile sites, including the lane-stride arm | `crates/backends/llvm/src/func/gpu.rs:gpu_site`, `crates/backends/llvm/src/func/gpu.rs:tile_plan`; `crates/mapal-ir/src/algo.rs:CategoryIr::tile_plan_strided` |
| R0/R0′: device row folds and sole-reader store layout | `crates/backends/llvm/src/func/rowfold.rs:rowfold_sites`; `crates/backends/llvm/src/func/gpu.rs:rowfold_kernel_text` |
| R1: form epilogue groups before residency; evaluate members at the tile store | `crates/backends/llvm/src/func/core.rs:FnEmit::set_gpu_arms`; `crates/backends/llvm/src/func/resident.rs:FnEmit::epilogues`; `crates/backends/llvm/src/func/gpu.rs:Kernel::store_cells` |
| R2: select cells and carries together; emit independent cell kernels | `crates/backends/llvm/src/func/resident.rs:FnEmit::cell_sites`; `crates/backends/llvm/src/func/gpu.rs:cell_kernel_text` |
| R3/R4: resident device launch or ordinary host operation | `crates/backends/llvm/src/func/gpu.rs:FnEmit::emit_tiled_map_gpu`; `crates/backends/llvm/src/func/ops.rs:FnEmit::emit_morphism` (resident replacement or `emit_op` fallback) |
| R5: prove row-local retention, launch into the resident row and upload changed rows | `crates/backends/llvm/src/func/retain.rs:retained_sites`; `crates/backends/llvm/src/func/resident.rs:FnEmit::emit_row_resident`, `crates/backends/llvm/src/func/resident.rs:FnEmit::emit_delta_h2d` |
| R6: classify device uses, keep/upload/element reads, death and arena offsets | `crates/backends/llvm/src/func/resident.rs:FnEmit::resident_plan`, `crates/backends/llvm/src/func/resident.rs:FnEmit::resident_after`, `crates/backends/llvm/src/func/resident.rs:FnEmit::arena`, `crates/backends/llvm/src/func/resident.rs:color`, `crates/backends/llvm/src/func/resident.rs:FnEmit::host_free` |
| R7: order independent chains and dependent launches | `crates/backends/llvm/src/func/resident.rs:FnEmit::chains`; `crates/backends/llvm/src/func/gpu.rs:pdl` |
| Grid raster and L2-window policy (§7) | `crates/backends/llvm/src/profile.rs:Gpu::raster`; `crates/backends/llvm/src/func/resident.rs:FnEmit::l2_window` |
| Site/host rung selection and device translation unit (§3, §7) | `crates/backends/llvm/src/func/gpu.rs:rungs_at`, `crates/backends/llvm/src/func/gpu.rs:host_rungs`, `crates/backends/llvm/src/func/gpu.rs:device_module`; `crates/backends/llvm/src/profile.rs:TargetProfile::gpu_rungs_on` |

The existing composition limits remain explicit: `set_gpu_arms` forms groups before calling
`resident_plan`, which selects cells and creates an arena only when the stream-chain map is empty. `retained_sites` refuses guarded
sites and unproven row-local shapes. These are ARCHITECTURE §5/§7's placement constraints,
not exceptions introduced by this map (FRAMEWORK §4.5 laws 1 and 5).

Planned correction, still unapplied: the [ordering candidate](../../performance/s57-ordering-evidence.md#reviewed-ordering-candidate--unapplied)
reuses grouped ownership for `ResidentPlan::stream_of` and the legacy-consumer veto.
Its private `grouped_writer` helper factors the existing `FnEmit::arena` query; it is a
candidate symbol, not shared implementation. Shared source retains the missing member
dependency cases documented there until ordinary GPU validation and integration.

The [saved host LLVM audit](../../../benches/results-s54-gpu/s57-host-optimization/host-gpu-call-effects-audit.md)
distinguishes these existing host hints from device attributes and explicit runtime
dependencies. In the inspected recurrence, GPU declarations expose neither device memory
effects nor completion tokens to LLVM; ordinary host vectorization still occurs. LLVM's
host stack lifetime markers do not realize `LastUsePlan` for device allocations. No
missed optimization or benefit from stronger attributes has been established.

## Loc / Trm — physical realization

| Architecture location / transmission | Realized at |
| --- | --- |
| `Host`, `Gmem`, `Smem`, `Reg` (§4) | Host emission `crates/backends/llvm/src/func/gpu.rs:FnEmit::emit_tiled_map_gpu`; device address spaces, shared memory and thread registers in `crates/backends/llvm/src/func/gpu.rs:DEVICE_PREAMBLE` and `crates/backends/llvm/src/func/gpu.rs:kernel_text` |
| Device allocation, arena base and release (R6) | `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_alloc`, `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_arena`, `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_free` |
| `Host → Gmem` whole/row copies; `Gmem → Host` whole/element copies (§4, R5/R6) | `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_h2d`, `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_d2h`, `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_d2h_on`; byte offsets/extents emitted by `crates/backends/llvm/src/func/resident.rs:FnEmit::emit_row_resident` and `crates/backends/llvm/src/func/resident.rs:FnEmit::emit_delta_h2d` |
| Launch control, pointer arguments and cache policy (§4, R3, §7) | `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_launch`, `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_launchv`, `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_launch_l2` |
| Runtime ordering/control within shared `Gmem`, and host clock fence (R7, §6 laws 1/4; ordering is not an array transmission) | `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_event`, `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_sync`; clock call emitted by `crates/backends/llvm/src/func/ops.rs:FnEmit::emit_time_ms` |
| Shared-memory staging, fragments and output stores (§4) | `crates/backends/llvm/src/func/gpu.rs:DEVICE_PREAMBLE` (shared address space, async-copy and barrier intrinsics); `crates/backends/llvm/src/func/gpu.rs:kernel_text` and `crates/backends/llvm/src/func/gpu.rs:Kernel::store_cells` |
| Host page registration (R7) | `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_pin`, `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_unpin` |

The CUDA-driver runtime realizes the Host/Gmem boundary; device text realizes the intra-SM
staging and stores. Its trace is diagnostic: `launch_on` synchronizes when tracing, so traced
kernel durations do not measure untraced stream overlap (`crates/mapal-rt/gpu/mapal_gpu.c:launch_on`).

## Harness and validation boundary

| Artifact / check | Realized at |
| --- | --- |
| Differential programs and multi-computation probes | `benches/gpu/gen_tiles.py:probes`, `benches/gpu/gen_tiles.py:probe_program` |
| Host/device objects, PTX and measurement arms | `benches/gpu/bundle.sh:emit_one`, `benches/gpu/bundle.sh:arms` |
| Value/trap smoke gate before timing | `benches/gpu/box_run.sh:DIFFERENTIAL_ONLY` (script-entry mode; rejects failed or empty smoke coverage); host linking `benches/gpu/box_run.sh:link` |
| Ordinary/traced timed-process status | `benches/gpu/box_run.sh:trace_rc`, `benches/gpu/box_run.sh:wall_rc` (probe/e2e: `rc` ordinary, `trace_rc` traced; decode: `wall_rc` ordinary, `rc` traced; nonzero fails the run) |
| Remote failure propagation and rental cleanup | `benches/gpu/vast_run.sh:run_rc`, `benches/gpu/vast_run.sh:cleanup` |
| Static launch/copy/arena placement counts | `benches/gpu/fusion_report.py:analyze`; static call counts do not establish runtime correctness |
| Kernel A/B, null controls and program measurements | `benches/gpu/rungs.py:kernel_arms`, `benches/gpu/rungs.py:null_ab`, `benches/gpu/rungs.py:program_arms`; summaries `benches/gpu/table.py:probe_table`, `benches/gpu/table.py:decode_table` |

The selection table's presence is not final validation. Acceptance still requires complete
integrated correctness and measurement evidence, independent confirmations and reconciled
selection/report status. Numerical-policy and unplaced-shape exclusions remain in
[ARCHITECTURE §8](ARCHITECTURE.md#8-what-is-not-placed-here); this map adds none.
