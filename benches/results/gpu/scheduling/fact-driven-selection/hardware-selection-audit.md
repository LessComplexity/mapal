# Compile-time hardware selection audit

Verdict: hardware/shape-driven GPU selection already exists. The concrete arithmetic-selection consistency defect identified in the independent semantic audit is the smallest supported mechanism correction. The register estimate retained across warp-fill shrink is intentional conservative policy; it is not a proved stale-fact bug. No production edit, test rerun, GPU action or profitability claim is made here.

Semble searches returned current `profile.rs`, `func/gpu.rs`, `func/tile.rs`, the warp-fill test and optimization plan. Archived duplicate hits were not treated as the live source. `hardware-audit-evidence.json` seals the source/plan and referenced independent audit. The original warp-fill patch and exact comment blame are retained alongside this report.

## Existing derivation and its boundaries

| Existing atom / transformation | Source | What it actually does |
|---|---|---|
| Dat `TileSite` / `TileRead` with independent outputs, ordered reduction and affine addressing; Loc target `Gpu` | `mapal-ir/src/algo.rs` records mapped in IR fact report; `profile.rs:Gpu`125 | Source extents/addressing and hardware resource limits are already available. They are not inferred from benchmark names. |
| Trn placement | `func/tile.rs:FnEmit::emit_tiled_map_realize`48–64; `gpu.rs:gpu_site`189 | Supported floating affine sites enter the GPU realization; ksplit address decomposition is refused. This is a compile-time fact gate. |
| Trn geometry and capability | `gpu.rs:launch`270–407; `profile.rs:TargetProfile::gpu_tile`2129–2252 | Affine separability/vector eligibility, shape/type, target registers/threads/SM/shared-memory and rungs determine one existing `Launch`. Scalar cooperative, naive, MMA and WGMMA routes are explicit. |
| Dat measured selection policy | `profile.rs:gpu_rungs_on`1747–1775; `gpu.rs:rungs_at` invoked at284 | Base geometry gives Fill; profile/architecture, Fill and Face index measured OFF rows, then measurement arms apply. This is a combination of derivation and recorded policy, not a general cost-ranking search. |
| Dat/Trm realized launch | `gpu.rs:Launch`220, `Launch::wave`242; `GpuTile`172 and its resource methods1247 | Carries grid/block/shared/geometry/raster to lowering. Resource ceilings are estimates/limits; `blocks_per_sm` is not an observation of achieved occupancy. Runtime launches/transfers and `ResidentPlan` consume already derived storage/completion facts. |

The function is a deterministic staged heuristic: register tile, thread-grid budget, fill adjustment, divisibility and whole-copy/shared-memory gates, then largest fitting power-of-two panel. These inline gates are reusable material for a future candidate checker. Their existence is not evidence that a generic ranking mechanism would be profitable, or that all legal geometry is searched today. `DispatchCost` is explicitly measured CPU-lane/task-pool economics (`profile.rs:DispatchCost`1362); its fields alone do not price GPU submission or prove they should be reused there.

## Register estimate: intentional, source-proved

`gpu_tile`2143–2160 derives `(tm,tn,regs_est)`. The nonwide estimate is twice the unshrunk square accumulator budget. The warp-fill loop2161–2171 reduces final `tm/tn` while leaving that budget unchanged; `resident_threads`2173 and subsequent block/shared budgets still read it.

This ordering is explicitly prescribed by the current function documentation2107–2111: register tile shrinks to increase grid warps, while threads per block stay those of the unshrunk tile. It is not an accidental omission inferred from variable naming.

Commit `8aa994458d06b344bbf4e4be3403ad25b64fa11a` deliberately changes the old `regs_est=2*tm*tn` into `regs_est=2*t*t` **before** introducing mutable/shrinking `tm/tn`, and adds the unchanged-thread documentation in the same patch. Later wide/FMA integration (`2cfcc522`, `9ab94d7c` in blame) retains this distinction. The plan's warp-fill section383–392 prescribes shrinking the register tile before changing the thread grid; it contains an earlier expected geometry sketch, so the current implementation and test pins are the authoritative exact output.

Current test `gpu_warp_fill_shrinks_the_tile_until_every_scheduler_has_two_warps`4113–4162 explicitly pins RTX5090 fill geometry to4×4 register tile,64×64 block,256threads; it does not recompute a new thread budget from the smaller tile. The same test pins unchanged controls at3072/4096 and device-dependent shrink choices. `gpu_tile_geometry_is_derived_from_device_facts`3785 pins normal target-resource derivations.

Therefore: retain this policy in a semantic consistency fix. Recomputing the budget would change threads, residency estimates, shared-memory share, panel depth and potentially route/Fill. That would require a separately reviewed policy/measurement intervention. Neither a smaller final microtile nor observed JIT registers proves that such a change is correct or profitable. No existing claim of actual register occupancy is licensed by this estimate.

## Smallest supported correction

The independent semantic audit proves a distinct missing connection: `launch` keys site policy from requested `Face::of(contract,reduced)` while scalar emission separately narrows contraction for strided lanes, and requested reduced precision can resolve to a scalar fallback. Thus the policy's arithmetic class can differ from the emitted realization. See the exact audit at `/private/tmp/mapal-library-image-review/compiler-semantic-audit/AUDIT.md`; this report does not duplicate its implementation work.

The minimal generic implementation should reuse the existing backend `Launch`: derive a typed actual scalar step/selected tensor realization from existing source permissions, affine facts and target capability/geometry; let both Face/rung selection and emission consume that same datum. It is not a benchmark-shaped rule, new hardware field in shared IR, new reduction permission or new geometry search. Preserve increasing-K/operand order, legacy Fill geometry and measured policy until their own evidence changes. Preserve the separate module-wide PDL/host-flag/kernel-wait invariant rather than changing only one consumer to a per-site Face.

Focused tests should prove the connection rather than assert performance: strided Contract→separate scalar arithmetic/Exact site Face; ordinary Exact/Contract unchanged; requested TF32 rejection→the actual scalar Face without TF32 instructions; supported selected tensor path→TF32 realization/Face; forced ON/OFF precedence; mixed-site module PDL consistency; host/device agreement on the same Launch. Use existing emission/property/differential fixtures and negative cases. No latency threshold belongs in these semantic tests.

A future generalized legality/cost interface can expose derived candidate legality and honest unknown costs, but should not be smuggled into this fix. Previous blanket small-M/thread-grid fitting alternatives already have measured wins and regressions; repeating or promoting them under a new name is not a new mechanism. The experiment matrix separates such future controlled discrimination from the current propagation correction.
