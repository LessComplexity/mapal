# S54 research: the per-architecture instruction matrix and the selection mechanism

Date: 2026-10-21 · HEAD 6d105b0 (`s53-gpu-opt`) · Sources: `profile.rs`, `lib.rs` (`GpuRungs`), `func/gpu.rs`,
`mapal_gpu.c`, `tile_bench.cu` (`facts` lines in `benches/results-s5{2,3}-gpu/`), the S53 notes (`creative`,
`kernel`, `link`), NVIDIA tuning guides / PTX ISA (links at the end). Nothing below is measured on Ampere;
every "expected" is a mechanism with its OFF arm named, not a number.

## 1. What `Gpu` has, and what the two Ampere profiles must contain

`Gpu` holds what `tile_bench.cu:device_facts` prints, minus `l2_bytes`, `sm_clock_khz`, `mem_clock_khz`.
**Add `l2_bytes`** (read by `l2-window` and `raster` below); clocks stay out (no decision reads them).
The device query, not the tuning guide, is the fact: the Blackwell guide says 12.0 has "128 KB smem/SM,
32 blocks/SM"; every 5090 `facts` line says `smem_per_sm=102400 max_blocks_per_sm=24`. The profile is
the query's cross-compilation spelling (ADR-0038 D4); the first box run checks it.

| field | `rtx-3090` (sm_86, GA102) | `a100` (sm_80, GA100) | source |
|---|---|---|---|
| `arch` | `"sm_86"` | `"sm_80"` | — |
| `sms` | 82 | 108 (SXM4-40GB and PCIe-40GB alike) | TechPowerUp GA102; A100 whitepaper |
| `regs_per_thread` / `regs_per_sm` | 255 / 65536 | 255 / 65536 | Ampere tuning guide |
| `smem_per_sm` | 102400 (100 KB) | 167936 (164 KB; an A800 box prints 167936) | Ampere tuning guide, forum |
| `smem_per_block` (opt-in) | 101376 (99 KB) | 166912 (163 KB) | CC table |
| `smem_reserved_per_block` | 1024 | 1024 | sm_80+ |
| `max_threads_per_sm` | 1536 (48 warps) | 2048 (64 warps) | Ampere tuning guide |
| `max_blocks_per_sm` | 16 | 32 | Ampere tuning guide |
| `max_threads_per_block` / `warp` | 1024 / 32 | 1024 / 32 | — |
| `l2_bytes` (new) | 6291456 (6 MB) | 41943040 (40 MB) | GA102 whitepaper; Ampere tuning guide |

Both go into `PROFILES` (`profile.rs:2179`) and `GPU_MEASURED` once measured; `gpu_rungs` inherits by
`arch` string, so sm_80 and sm_86 are two measured parts, never each other's proxy. **A100 variant:
`A100_SXM4`** — vast today: SXM4 4 offers from $0.695/h (40 GB at the floor), PCIe 2 from $0.734/h;
same SM and per-SM facts, SXM4 differs only in HBM clock and TDP (printed, not read). Profile `a100`,
docstring "SXM4-40GB box; PCIe is the same `Gpu`". RTX 3090: 4 offers from $0.173/h. The `h100` SXM
`facts` line exists (`sms=132`, S53 confirm boxes), so that profile is checked too.

## 2. Instruction / feature matrix, by `Gpu::sm()`

Rules read off the PTX ISA target notes (ISA-table facts, the `has_mma_sync` precedent), never a part fact.

| feature | sm_80 | sm_86 | sm_89 | sm_90a | sm_120 | what Mapal emits today |
|---|---|---|---|---|---|---|
| `cp.async` 4/8/16 B, `.ca`; `.cg` only at 16 B; `.L2::prefetch_size` hint | ✓ | ✓ | ✓ | ✓ | ✓ | `ca.4/8`, `cg.16` (`gpu.rs:47–49`), no hint |
| `mma.sync.m16n8k8.tf32` (k4 is the same unit halved, no rung) | ✓ | ✓ | ✓ | ✓ | ✓ | `mma` rung |
| `ldmatrix.x4` | ✓ (sm_75+) | ✓ | ✓ | ✓ | ✓ | `ldmatrix` rung, OFF 4090 |
| `ld.global.nc`, `ld.global.L2::evict_*` via `createpolicy` (sm_80+) | ✓ | ✓ | ✓ | ✓ | ✓ | `ld.global.nc.v4` in the gen kernel only |
| L2 persistence window (`accessPolicyWindow`, "L2 Cache Residency Management" ≥ 8.x) | ✓ | ✓ | ✓ | ✓ | ✓ | none |
| `cp.async.bulk` (1-D, mbarrier) / `cp.async.bulk.tensor` (TMA) | ✗ | ✗ | ✗ | ✓ | ✓ (PTX: sm_90+; verify on the box) | none |
| TMA `.multicast::cluster` | ✗ | ✗ | ✗ | ✓ (sm_90a) | ✗ (forum: consumer parts lack multicast) | none |
| thread-block clusters + DSMEM (`mapa`, `barrier.cluster`) | ✗ | ✗ | ✗ | ✓ (portable 8) | ✓ per Blackwell guide, size 8 (forum) — query `CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH` | none |
| `griddepcontrol` (PDL) | ✗ | ✗ | ✗ | ✓ | ✓ (sm_90+) | none |
| `wgmma`, `setmaxnreg` | ✗ | ✗ | ✗ | ✓ **sm_90a only** | ✗ | `wgmma` rung |
| `tcgen05` | ✗ | ✗ | ✗ | ✗ | ✗ (sm_100a/101a/103a only) | — |
| FP32 per SM per clock vs sm_80 | 1× | 2× | 2× | 2× | 2× | `exact` = mul.rn + add.rn, 2 issues/MAC |
| warps / blocks per SM | 64 / 32 | 48 / 16 | 48 / 24 | 64 / 32 | 48 / 24 | `fill`, `min-blocks` |
| smem per SM | 164 KB | 100 KB | 100 KB | 228 KB | 100 KB (box) | `gpu_tile_async` stages |
| L2 | 40 MB | 6 MB | 72 MB | 50 MB | 96 MB | not read |
| concurrent kernels / streams / graphs / events | ✓ all | ✓ | ✓ | ✓ | ✓ | one stream (`cuLaunchKernel(..., 0, ...)`), `cuCtxSynchronize` |

## 3. Rungs expected to flip on Ampere (hypotheses, each with its arm)

- **`async` (OFF filled on 4090, OFF all on H100, OFF under-filled on 5090).** The measured cause on Ada
  is depth: a 2-block share of 49 KB holds 2 stages at bk=16 (`s53-research-kernel` §2.1). On A100 the
  share is 81 KB → 4–5 stages, cuBLAS's depth. Expect ON filled on sm_80. H100 had 3 stages and lost
  to spills, so the flip is not certain; it is the first A100 arm to read. On the 3090 the share is Ada's
  (100 KB) → the 4090 rows.
- **`mma`/`wide` on A100.** FP32 19.5 TF vs TF32 156 TF (8×; Ada/Blackwell ≈ 2×): the TF32 face is where
  the A100 can win and the FMA tile cannot; `wide` (64×64 warp tile) and `ldmatrix` (≈0 on Ada, where the
  tile was not issue-bound) are re-measured there, not inherited.
- **`min-blocks` / `warp-fill`.** A100 holds 16 warps/SM at the 2-block cap vs 64 possible; the fill target
  is `GPU_RESIDENT_BLOCKS · schedulers · sms` = 864 warps on 108 SMs, 656 on the 3090 — n=1024 is
  under-filled on both (FMA 512 warps), so `warp-fill` is the n≤1024 rung as on Ada.
- **`unroll`, `prefetch`, `interleave`, `fma-regs`.** The 3090's SM is the 4090's at 1.7 GHz with 16
  blocks/SM: expect the 4090 rows. The A100's 1× FP32 datapath makes the exact face issue-bound sooner,
  so `pad`/`affine` (issue slots traded for bank conflicts) may reverse — H100's exact-face rows
  (`idx32`/`affine`/`pad` OFF) are the closer analogue.
- **The 3090's 6 MB L2** has no analogue on the measured parts (72/96/50 MB hold one 64 MB operand at
  4096): a row-major grid re-reads B from HBM every wave, so the `raster` rung (kernel §3 R9) is
  measurable only there.

## 4. `GPU_RUNGS_OFF` for fusion / placement rungs

`RungOff {profile, rung, fill, face, cite}` already carries host rungs as `Fill::Any` rows (`resident`,
`devgen`, `retain-delta`, via `gpu::host_rungs`). The new rungs are of two kinds:

1. **Per-site placement** — `devmap` (a k=0 per-cell map becomes a device site: the `ew` probe, 0 device
   code today), `epilogue` (a per-cell consumer of a tile site's output folds into its store: `epi`, `bias`),
   `prologue` (already `devgen`'s composition: `pro` fused). Their effect depends on the *host kernel's*
   grid (register pressure under the 128-reg cap) and face, so the fill key is the kernel the consumer
   folds into; `Fill`/`Face` fit unchanged. The `ew` chain with no fold has its own grid (one thread per
   cell): fill from `warps_per_scheduler(cells / threads, threads)`.
2. **Cross-site** — `streams` (independent sites: `indep`), `pdl`, `graph`, `l2-window`. The fill of a
   pair is the union grid's: two under-filled grids is the only class where `streams` can pay (at
   n≥2048 one grid already fills every part, link §3). Face is `Any`.

**Minimal extension: none to the struct.** One rule in `gpu_rungs`'s caller: a fused or concurrent rung
reads `Fill` from the grid it ends up in (host kernel, own grid, or union); the row's `cite` names the
probe program (`fuse/<name>.mapal`) and the ON/OFF medians, as today. If a cross-site rung ever shows a
sign that `Fill` does not separate (e.g. `l2-window` by operand bytes vs `l2_bytes`), the extension is a
third key `Fit {Any, InL2, OverL2}` computed from `l2_bytes` — a derivation, not a knob, and only when a
measurement needs it.

## 5. New per-arch instructions worth a rung (ranked by expected effect × arches)

| rung | ISA gate | data fact it reads (what a library cannot know) | arches | expected |
|---|---|---|---|---|
| `devmap` + `epilogue` (placement, no new instruction) | any | consumer's `elem_plan` law, trap-free; `Footprint` ⊆ block tile (amendment 9) | all 5 | removes a d2h + host pass: H100 4096 3.2 ms d2h vs 4.2 ms kernel |
| `streams` + pinned (`cuMemAllocHost`) | any | `path_plan` independence; `last_use_plan` | all 5 | `indep` serial → concurrent; under-filled only |
| `graph` (`cuGraph` replay of a loop body) | any | `loop_plan` static shapes + `retain`'s fixed buffers | all 5 | decode: launch overhead × sites × T |
| `l2-window` (`accessPolicyWindow` + `createpolicy evict_last`) | sm_80+ | exact use count: B read by every row-block; a retained object read T times | all 5; strongest 3090, A100 | +3–8% at 8192 (R9's bound) |
| `raster` (grid order from `l2_bytes`) | any | `TileRead` strides + `l2_bytes` | 3090 first | only measurable there |
| `pdl` (`griddepcontrol`, launch attr `PROGRAMMATIC_STREAM_SERIALIZATION`) | sm_90+ | which operands of site 2 are independent of site 1 (`mlp`: `e` loads before `c` completes) | H100, 5090 | hides prologue + launch gap in chains |
| `bulk` (`cp.async.bulk` 1-D, mbarrier) for the K-major A panel; `tma` (descriptor from `TileRead` strides) | sm_90+ | panel = affine box, one issuing thread | H100, 5090 (box-verify sm_120) | frees `cp.async` issue slots; H100 TF32 prerequisite (creative #6) |
| `cluster` A-panel multicast | sm_90a | two row-blocks share A (grid reuse count) | H100 | halves L2→smem for A; low priority |
| `l2-prefetch` (`cp.async .L2::128B`) | sm_80+ | next panel row = next sector | all 5 | one qualifier, likely ≤2% |
| `warpspec` (`setmaxnreg`) | sm_90a | `g2s` as a `Trm` on its own `TrnLoc` (creative #10) | H100 | H100 TF32 last 15% |

Not worth a rung: `mma.m16n8k4` (half a k8), `ld.global.nc` on the tile kernel (operands already go
through `cp.async`/smem), `tcgen05` (no measured part has it), clusters on sm_120 (no multicast, so no
A-sharing; DSMEM alone buys the FMA tile nothing).

## 6. Multiple computations

The probe's losses are placement problems, arch-independent: `epi`/`bias` → `epilogue`, `ew` → `devmap`,
`mlp` → `resident` once the map is a device site. Arch-specific gains come after: `indep` → `streams`
(all), `mlp` → `pdl` (sm_90+), loops → `graph` (all), reused operands → `l2-window` (sm_80+, sized by
`l2_bytes`). Each is a `GpuRungs` field + `--gpu-off` + a `bundle.sh` arm, differential N/N on both faces
over every probe program. Order: A100 + 3090 baseline (every existing arm) → `devmap`/`epilogue` on all
five → `streams`/`graph` → `pdl`/`bulk` on H100 + 5090 → `l2-window`/`raster`, 3090 first.

## Sources

- Ampere tuning guide: https://docs.nvidia.com/cuda/archive/13.0.3/ampere-tuning-guide/index.html
- Hopper tuning guide: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
- Blackwell tuning guide (12.0 "128 KB / 32 blocks" — the 5090 query says 100 KB / 24 and wins): https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- CC feature/spec tables: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- PTX ISA 8.0 notes (`sm_90a`, `wgmma`, `setmaxnreg`, `cp.async.bulk`; clusters sm_90+): https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html
- libcudacxx: `cp.async.bulk.tensor` unicast SM_90 / multicast SM_90a; `setmaxnreg` SM_90a: https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html
- PDL SM90+: https://docs.modular.com/mojo/stdlib/gpu/grid_controls/PDL
- Consumer Blackwell clusters (forum, unofficial): https://forums.developer.nvidia.com/t/thread-block-clustering-in-blackwell-gpus/320471/
- RTX 3090 facts: https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622 ; A800 smem 167936: https://forums.developer.nvidia.com/t/question-about-max-shared-memory-in-block-and-multiprocessor/283345
- Local: `tile_bench.cu:78`, `facts` lines in `benches/results-s53-gpu/integrated-*.txt`, `profile.rs`, `lib.rs:74`, `benches/results-s54-gpu/fusion-probe-static.txt`.
