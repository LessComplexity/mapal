# Accepted selection kernel evidence

Read-only inspection, 2026-10-21. Scope: saved `selection-h100-a`, `selection-5090-a2` and `selection-5090-b` under `/private/tmp/mapal-s56-gpu`. These are accepted selection runs, **not final integrated-build measurements**. Their register/memory/instruction differences establish implementation facts; none alone establishes the cause of a timing gap.

## Provenance and actual launch binding

`current-library-kernels.csv` records all 36 measured cells: three sessions × 1024/2048/4096/8192 ×Exact/Contract/TF32. Every referenced PTX SHA256 still matches its saved file. Each selected results section identifies the actual Mapal entry, grid, block, dynamic shared memory and driver-reported registers/local bytes; those names match the recorded PTX entries. The CSV's `source_line` points to the section header, **not** the resource line: read forward to its matching `mapal kernel=...` line. Looking backward incorrectly selects the preceding arm.

The three `verification.json` files accept resources and Nsight coverage and record these archive hashes, independently rechecked here:

| Session / archive | SHA256 |
|---|---|
| H100 `selection-h100-a/prof-h100-pcie.tgz` | `f9d73ea390b23305a75c30db450463a581cf29e67d2063b603eb530a0f772cdb` |
|5090 A2 `selection-5090-a2/prof-rtx-5090.tgz`|`c1526049b7c95f337a3a73837b0f0b158f1512623500375ec975fa1aa02194e9`|
|5090 B `selection-5090-b/prof-rtx-5090.tgz`|`1ca8006e7cfc7aa4a5e1c595133e6a6688f3318795a61254b7afcb4a15322c17`|

Direct executed-kernel evidence is stronger than the side-compiled SASS: each archive has `prof/run-{1024,4096,8192}.sqlite` for Contract and `prof/run_mm_4096_tf.sqlite`. Joining `CUPTI_ACTIVITY_KIND_KERNEL.demangledName` to `StringIds.id` gives the actual names, resource attributes, geometry and intervals. Each compared kernel has 8 observed launches in these profiler runs. There is no corresponding library timeline for 2048 or Exact, or TF32 sizes other than 4096; their timing records do not supply those missing instruction/launch traces.

## Measured resource differences

These tuples are **registers/thread, threads/block, dynamic shared bytes/block**, read from the actual kernel activity records. Static shared and local memory/thread are0 for these particular profiled kernels. The two5090 hosts agree except Mapal TF32 registers, 124 (A2)/122 (B).

| Profile / traced face,size | Mapal | cuBLAS | Pinned CUTLASS |
|---|---|---|---|
| H100 Contract, 1024 |157,128,6272|121,128,12672|137,256,16640|
| H100 Contract, 4096/8192 |127,256,16640|255,128,33792|137,256,16640|
| H100 TF32, 4096 |212,256,200704|168,384,231424|226,128,65536|
|5090 Contract, 1024|79,256,8448|128,128,30720|124,256,16640|
|5090 Contract, 4096/8192|128,256,16640|210,256,49152|124,256,16640|
|5090 TF32, 4096|124/122,256,37888|234,128,98304|230,128,65536|

Their grids also differ: at H100 TF32, 4096, Mapal 32×16, cuBLAS 114×1, CUTLASS 32×32; at5090 Contract4096, Mapal/CUTLASS 32×32 versus cuBLAS 256×2. These are directly observed scheduling geometries, not proof of achieved occupancy, utilization or a bottleneck.

For the unprofiled Exact cells, launch attributes still exist in the ordinary results: H100 1024 has 106 registers/0 local bytes, then 128/0 at 2048/4096/8192. 5090 Exact 1024 has 77/0 on both hosts, then 128 registers with 24 local bytes on A2 versus 8 on B at 2048/4096/8192, for identical saved PTX. Likewise5090 TF32 registers differ between drivers at 1024(76/66) and larger sizes(124/122). These facts prevent treating PTX or an independently assembled cubin as a unique description of the driver's executed image. They do not measure local-memory traffic or prove its timing cost.

## Instructions and library identities actually supported

Entry-scoped inspection of all measured Mapal PTX proves these emitted arithmetic paths:

- Exact: separate `mul.rn.f32`/`add.rn.f32`; Contract: `fma.rn.f32` on both profiles at all four sizes.
- H100 TF32: `wgmma.mma_async...m64n128k8` at 1024, `m64n256k8` at 2048/4096/8192, with async copies/waits.
-5090 TF32: `mma.sync...m16n8k8...tf32.tf32`, with async copies/waits at all four sizes. 5090 Exact 2048+ also contains async-copy instructions; that presence alone does not show useful copy/compute overlap.

The actual cuBLAS names in `prof/kern-*.csv` differ by device/size. H100 Contract uses named `sm80_xmma...ffma` kernels with 64×64×8 at 1024 and 128×128×8 at 4096/8192; TF32, 4096 names `sm90_xmma...tf32...128x128x32...warpgroupsize1x1x1`. 5090 Contract names `cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5...>` at 1024 and `...256x128_8x4...` at 4096/8192; TF32, 4096 names `...cutlass_80_tensorop_s1688gemm_128x128_32x3...`. These Kernel2 entries are the observed cuBLAS leg, **not the separate pinned CUTLASS leg**. The latter is the long `cutlass::Kernel<...MmaPipelined...128,128,8...>` FP32 template or `...MmaMultistage...128,128,16...>` TF32 template. Names establish selection identity; cuBLAS machine instructions are not captured.

SASS is present, with an important provenance limit. `prof/mm_4096*.{ptxas,cubin,sass}` were separately assembled from saved PTX by `box_profile.sh`; timing loads PTX via the CUDA driver. `mm_4096.null.ptx` is byte-identical to the Exact input and supplies its recorded SASS alias. Entry-scoped side-compiled SASS shows H100/5090 Exact FMUL+FADD, Contract FFMA, H100 TF32 HGMMA and5090 TF32 HMMA;5090 Exact also has LDL/STL instructions. The offline ptxas log reports 24 stack bytes and 20 spill-store/20 spill-load bytes there. That is an assembler fact, not a dynamic spill count or proof that every driver's JIT has the same spills.

`prof/tile_bench.sass` contains the compiled pinned CUTLASS kernels: SIMT uses FFMA and its TF32 TensorOp kernel uses HMMA/LDGSTS. This is executable fatbin disassembly, whereas Mapal SASS is a side-compiled module. Neither is a dynamic instruction count. All three selected 5090 Mapal 4096 cubins and the CUTLASS harness SASS are byte-identical between A2/B, while some driver-reported resources differ. **No cuBLAS shared-library SASS or exact executed Mapal JIT machine image is saved.**

## What cannot explain the gaps yet

There are no saved hardware-counter measurements: all three `ncu-*.csv` files are empty. The harness documents the container counter-permission limitation, but these saved logs do not show a per-run `ERR_NVGPUCTRPERM` attempt; absence of data is the supported claim. Achieved occupancy, tensor/SM utilization, cache hit rates, DRAM/shared bandwidth, instruction issue/stalls and dynamic spill traffic therefore remain unmeasured. Registers, shared bytes, static opcodes, grid shapes and timing ratios cannot substitute for them.

The ordinary CUDA-event kernel comparisons and Nsight kernel identities establish which tested implementations ran and how long those runs took. Their numerical contracts differ from Mapal Exact; the pinned CUTLASS configurations are specific choices, not an architecture-wide optimum. Whole-program Mapal probes include host/transmission/allocation work while the library probe legs use device-event timing. Saved MLP2 timelines can establish particular overlap/gap facts, as documented in `placement-gap-triage.md`, but do not identify a single-kernel bottleneck or general SM utilization.

Finally, the 5090 Contract/TF32, 4096/8192 selection runs retain the pre-final raster policy. They cannot be combined into a fabricated final-table measurement. No ratios are reproduced here; use `current-library-comparison.md`/its CSV for those already qualified numbers. This inspection proposes no tuning or fixes and performed no build, GPU action or source change.
