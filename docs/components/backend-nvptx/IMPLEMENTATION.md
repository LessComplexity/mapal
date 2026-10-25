# backend-nvptx — implementation map

Maps [ARCHITECTURE.md](ARCHITECTURE.md) §§2–7 to the GPU leg inside the LLVM backend,
not a separate backend crate (FRAMEWORK §4.2, §4.5, §6.3). Paths are repository-relative.
The mechanisms below are built; disputed per-architecture selections are resolved, while
the frozen S54 validation matrix and both confirmations pass. This map records code locations, not performance verdicts.
STATUS distinguishes finite benchmark acceptance from the S58 ordering correction and
remaining allocation/performance work.
The isolated [S59 small-M candidate](../../performance/gpu/kernels/small-m-geometry-comparison.md) is
not promoted because measured regressions accompany its wins; it adds no built row here.
The [S60 caller audit](../../performance/gpu/workloads/composed-benchmark-preparation.md) establishes an
existing dependency: `func/gpu.rs:launch` queries `TargetProfile::gpu_tile` before
`rungs_at` to derive `Fill`, then queries geometry again with selected wide/fill rungs.
Helper admission can therefore change rung selection as well as final dimensions.
The isolated A/B experiments add no built mechanism to this map.

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
| Actual arithmetic on the selected launch (§9) | `crates/backends/llvm/src/func/gpu.rs:Arithmetic`; existing `Launch::arithmetic` carries the selected separate, ordered-FMA, TF32-MMA or TF32-WGMMA realization |

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
| R7: resolve grouped physical writers and order independent chains, feeder events and element readback | `crates/backends/llvm/src/func/resident.rs:grouped_writer`, `crates/backends/llvm/src/func/resident.rs:ResidentPlan::stream_of`, `crates/backends/llvm/src/func/resident.rs:FnEmit::chains`; `crates/backends/llvm/src/func/gpu.rs:pdl` |
| R6/R7: omit raw chain storage when the selected group does not store it | `crates/backends/llvm/src/func/resident.rs:FnEmit::chain_prologue`; existing `Epi::store_out`; regression and live-storage controls in `crates/backends/llvm/tests/streams.rs:grouped_chains_allocate_only_materialized_raw_outputs`, `crates/backends/llvm/tests/streams.rs:raw_storage_controls_keep_their_observable_pointers` |
| Grid raster and L2-window policy (§7) | `crates/backends/llvm/src/profile.rs:Gpu::raster`; `crates/backends/llvm/src/func/resident.rs:FnEmit::l2_window` |
| Site/host rung selection and device translation unit (§3, §7) | `crates/backends/llvm/src/func/gpu.rs:rungs_at`, `crates/backends/llvm/src/func/gpu.rs:host_rungs`, `crates/backends/llvm/src/func/gpu.rs:device_module`; `crates/backends/llvm/src/profile.rs:TargetProfile::gpu_rungs_on` |
| Resolve actual arithmetic before selecting per-site policy (§9) | `crates/backends/llvm/src/func/gpu.rs:Arithmetic::scalar`, `crates/backends/llvm/src/func/gpu.rs:Arithmetic::face`, `crates/backends/llvm/src/func/gpu.rs:launch` |
| Consume selected arithmetic during scalar emission (§9) | `crates/backends/llvm/src/func/gpu.rs:kernel_text`, `crates/backends/llvm/src/func/gpu.rs:Kernel::step`; tensor emission is selected from the same `Launch` |
| S62 ordered skinny geometry (§10): `row-fit`, `wave-gate`, `grid-share`, `panel-regs`, `fill-rows`; output partition only, each cell's k fold whole on one thread | `crates/backends/llvm/src/profile.rs:TargetProfile::gpu_tile_with`, `crates/backends/llvm/src/func/gpu.rs:panel_regs`, `crates/backends/llvm/src/func/gpu.rs:Launch::wave`; regression `crates/backends/llvm/src/func/gpu.rs:skinny_projections_take_the_ordered_cooperative_tile` |
| S62 cell-fill tile and staged panels (§10): `cell-fill` (1×1 register tile, threads from the register budget), `asyncstage` (multi-stage cp.async; OFF on all four parts) | `crates/backends/llvm/src/profile.rs:TargetProfile::gpu_cell_tile`, `crates/backends/llvm/src/profile.rs:block_threads`, `crates/backends/llvm/src/profile.rs:square_register_tile`, `crates/backends/llvm/src/profile.rs:TargetProfile::gpu_tile_stages`; regression `crates/backends/llvm/src/func/gpu.rs:starved_m16_site_takes_the_cell_tile_and_streams` |
| S62 reassociation face split-K (§10, ADR-0040): S partial folds into a workspace slab each, fixed-order reduce | `crates/backends/llvm/src/profile.rs:TargetProfile::split_count`, `crates/backends/llvm/src/func/gpu.rs:kernel_id_split`, `crates/backends/llvm/src/func/gpu.rs:Kernel::at_z`, `crates/backends/llvm/src/func/gpu.rs:Kernel::split_seed`, `crates/backends/llvm/src/func/gpu.rs:reduce_kernel_text`, `crates/backends/llvm/src/func/gpu.rs:max_index`; regressions `crates/backends/llvm/tests/splitk.rs`, `crates/backends/llvm/src/func/gpu.rs:cell_tile_split_stores_into_its_slab` (the 9a9cc20 slab-offset fix) |
| S62 cost-gated device placement (§10): `place` keeps a site on the host where its roofline time does not pay for the launch | `crates/backends/llvm/src/func/gpu.rs:placement`, `crates/backends/llvm/src/func/gpu.rs:device_made`, `crates/backends/llvm/src/profile.rs:TargetProfile::device_pays`, `crates/backends/llvm/src/profile.rs:DeviceCost`, `crates/backends/llvm/src/profile.rs:roofline_ns`, `crates/backends/llvm/src/func/core.rs:FnEmit::on_device`, `crates/backends/llvm/src/func/resident.rs:FnEmit::cell_pays` |
| S62 host half of a GPU profile (§10): `hostopt` composes `<gpu>+<host>` so host code keeps CPU optimizations | `crates/backends/llvm/src/profile.rs:with_host`, `crates/backends/llvm/src/profile.rs:X86_64_V3`, `crates/backends/llvm/src/profile.rs:TargetProfile::named`; test `crates/backends/llvm/tests/host_profile.rs` |

The existing composition limits remain explicit: `set_gpu_arms` forms groups before calling
`resident_plan`, which selects cells and creates an arena only when the stream-chain map is empty. `retained_sites` refuses guarded
sites and unproven row-local shapes. These are ARCHITECTURE §5/§7's placement constraints,
not exceptions introduced by this map (FRAMEWORK §4.5 laws 1 and 5).

The [S58 ordering correction](../../performance/gpu/memory/ordering-and-allocation-development.md) reuses
grouped ownership for `ResidentPlan::stream_of` and the legacy-consumer veto.
Its private `grouped_writer` helper factors the existing `FnEmit::arena` query.
Canonical aliases resolve to the physical writer before selecting a stream; unknown
ownership remains on the legacy path. Runtime calls and allocation/free policy are unchanged.

The [S61 allocation correction](../../performance/gpu/current-results.md) subsequently
changes only raw storage admission in `chain_prologue`: an existing selected group with
`store_out=false` needs no raw buffer. Stored members, allocation positions, stream
dependencies, arena eligibility and the runtime ABI stay as before. The zero-initialized
unused slot reaches `mapal_gpu_free`'s existing zero-slot return. Its own ordinary GPU
gate and separate allocation counters pass; that result establishes neither a universal
race proof nor a latency gain.

The [saved host LLVM audit](../../../benches/results/gpu/host-dispatch/optimization-audit/host-gpu-call-effects-audit.md)
distinguishes these existing host hints from device attributes and explicit runtime
dependencies. In the inspected recurrence, GPU declarations expose neither device memory
effects nor completion tokens to LLVM; ordinary host vectorization still occurs. LLVM's
host stack lifetime markers do not realize `LastUsePlan` for device allocations. No
missed optimization or benefit from stronger attributes has been established.

## Loc / Trm — physical realization

| Architecture location / transmission | Realized at |
| --- | --- |
| `Host`, `Gmem`, `Smem`, `Reg` (§4) | Host emission `crates/backends/llvm/src/func/gpu.rs:FnEmit::emit_tiled_map_gpu`; device address spaces, shared memory and thread registers in `crates/backends/llvm/src/func/gpu.rs:DEVICE_PREAMBLE` and `crates/backends/llvm/src/func/gpu.rs:kernel_text` |
| Device allocation, arena base and release (R6) | `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_alloc`, `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_arena`, `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_free` (an `entry-arena` member released with its arena: `arena_member`) |
| `entry-arena`: arena made at function entry, loop-cone slots seeded from it | `crates/backends/llvm/src/func/resident.rs:FnEmit::entry_arena` (called by `FnEmit::resident_teardown`; reuses `color`) |
| Kernel loading before the clock (`preload`) | `crates/mapal-rt/gpu/mapal_gpu.c:preload` (from `init_all`), lookup in `crates/mapal-rt/gpu/mapal_gpu.c:kernel`; flag `crates/backends/llvm/src/func/gpu.rs:GPU_INIT_PRELOAD` |
| `Host → Gmem` without a host wait (`async-h2d`, `stage`) | `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_h2d_async` (page-locked ring `stage_buf`, flag `crates/backends/llvm/src/func/gpu.rs:GPU_INIT_STAGE`); emitted by `crates/backends/llvm/src/func/resident.rs:FnEmit::resident_teardown` |
| `Host → Gmem` on a chain's stream (`chain-h2d`) | `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_h2d_on`; readers' chain from `crates/backends/llvm/src/func/resident.rs:FnEmit::chains` (`ResidentPlan::upload_chain`), emitted by `FnEmit::resident_after` |
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
| S62 per-arch A/B: ABBA order, value gate before timing, leave-one-out rung arms, report tables | `benches/gpu/integrated_bundle.sh:loo`, `benches/gpu/integrated_bundle.sh:emit_one`, `benches/gpu/abba.py:decide`, `benches/gpu/arm_ratio.py:ratio`, `benches/gpu/integrated_report.py:report`, `benches/gpu/splitk_report.py`; submission floor `benches/gpu/submit_bench.c:main` |
| Executed-image observation tooling: exact PTX, pending load/context and callback-lifetime cubin copy (four S61 Linux/GPU Contract captures admitted) | `benches/gpu/capture_images/capture.cpp:Pending`, `benches/gpu/capture_images/capture.cpp:callback`, `benches/gpu/capture_images/capture.cpp:writefile`, `benches/gpu/capture_images/capture.cpp:InitializeInjection` |
| Observation admission: completed-process hashes, ELF/function, module/context joins and exact launch contract; always excludes ordinary timing | `benches/gpu/capture_images/validate.py:validate`, `benches/gpu/capture_images/validate.py:elf_kernel`; bounded launcher `benches/gpu/capture_images/run_capture.py` (script entry) |

The selection table's presence is not final validation. Acceptance still requires complete
integrated correctness and measurement evidence, independent confirmations and reconciled
selection/report status. Numerical-policy and unplaced-shape exclusions remain in
[ARCHITECTURE §8](ARCHITECTURE.md#8-what-is-not-placed-here); this map adds none.

The separate sealed library observer v2 now has live evidence for its callback invocation,
module/function and exact-entry joins: three admitted legs and four selected entries.
Its `capture.cpp` callback and `admit.py` `admit` transformation remain standalone tooling
in the retained source archive, not a replacement for the repository observer above or
a compiler/runtime mechanism. [Source seal and repairs](../../../benches/results/gpu/disassembly/library-kernels/counter-free-capture/observation-schema-repair/README.md)
and [live realization evidence](../../../benches/results/gpu/disassembly/library-kernels/counter-free-capture/rtx-5090/README.md)
bind those functions to exact source and outputs. No production implementation row changes.
