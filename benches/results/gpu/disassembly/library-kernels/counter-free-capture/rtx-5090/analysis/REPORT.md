# Actual Mapal, cuBLAS and CUTLASS kernel images

All three selected calls are independently admitted for **M16/N2048/K7168, Contract**: Mapal launches1kernel, cuBLAS2, and selected FP32-SIMT CUTLASS16×64×16 launches1. Exact ordinary executable, selected candidate, PTX, pre/post numerical gates and complete selected image/function/context/stream/launch joins pass. The reader rechecked4,608 raw words. This is a separate diagnostic execution; **no profiling duration is admitted**. [Independent admission](../independent-review/admission.json), [complete identities and anchors](comparison.json).

The captured Mapal image is the frozen original independent-output realization, predating compiler fix50a57e5. It does not validate or time that fix. The diagnostic RTX5090 host reports NVIDIA driver610.43.02 / CUDA driver API13030(13.3), distinct from earlier timing hosts. Decoder12.8.90 and nvcc12.8.93 are separate tool identities. All four listings targetsm_120 and contain unknown `?trans`/scheduling annotations; full decoder compatibility is not claimed. [Host](../tools/gpu-version.txt), [decoder](../tools/decoder-version.txt), [queried resources](harness-metadata.json).

| Actual selected entry | Grid | Threads/block | Registers/thread; local bytes | Dynamic shared bytes |
| --- | --- | --- | --- | --- |
| Mapal independent-output |128×1×1|256|40;0|0|
| cuBLAS GEMM entry |8×2×32|128|not recorded|25,600|
| cuBLAS reduction entry |64×1×1|32×16=512|not recorded|0|
| CUTLASS16×64×16 |1×32×1|128|56;0|10,496|

CUTLASS reports static shared0. Mapal/cuBLAS static shared attributes were not recorded; ELF section sizes and largest register names are not substituted for those queries. Exact full symbols and module/text/SASS hashes are in comparison.json. The cuBLAS GEMM symbol contains `cutlass_80_simt_sgemm_128x32_8x5_nn_align1`; its named axes/stages are not equated to Mapal axes. The separately launched `cublasLt::splitKreduce_kernel` is an observed entry, rather than an inference from an installed library image.

The decoded code shows these concrete differences:

- **Mapal independently accumulates one output per work item.** The loaded PTX computes flattenedCTA×256+thread output indices, an increasing-K FMA recurrence and one output store ([PTX](../inputs/mapal-loaded.ptx),lines30–85). The SASS has global-load operands at0x0150/0x0180, one dependent FMA chain beginning0x02d0/0x02f0 and one STG at0x0dd0. Its decoded entry contains no LDS/STS/BAR or call mnemonics. [Mapal SASS](../disassembly/mapal-sass-0/stdout),lines37–41,61–67,237.
- **Selected CUTLASS stages through shared memory and maintains several accumulators.** LDG0x0630, STS0x0ba0, BAR0x0d20, LDS.1280x0d70, and FFMAs0x1050–0x1080 target distinct accumulator registers. Its recorded configuration is threadblock16×64×16, warp8×32×16,2stages,128threads, split_k_slices1. [CUTLASS SASS](../disassembly/cutlass-sass-0/stdout),lines113,200,224,229,276–279; [configuration](harness-metadata.json).
- **The cuBLAS GEMM entry contains asynchronous global-to-shared copies and dependency/barrier operations.** LDGSTS0x0a60, LDGDEPBAR0x0c90, LDS.1280x1b20 and independent accumulator FFMAs0x1cf0–0x1d30 are directly visible. LDGSTS is NVIDIA's asynchronous global-to-shared instruction; its presence does not measure overlap or transferred bytes. [cuBLAS GEMM SASS](../disassembly/cublas-sass-0/stdout),lines184,219,452,482–486; [NVIDIA instruction reference](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-binary-utilities/index.html#instruction-set-reference).
- **The second cuBLAS entry contains loads, FP32 additions and output stores.** Loads0x0470–0x05b0 feed FADD0x05f0 onward. GridZ32 on the first dispatch and the separately observed reduction entry support a partitioned/reduction organization; exact partition extents, reduction order and heuristic selection are unrecorded. [Reduction SASS](../disassembly/cublas-sass-1/stdout),lines88–119. On the256 sampled cells, Mapal and CUTLASS match ordered-FMA bits256/256; cuBLAS matches5/256, while all pass their declared componentwise gates. Matching CUTLASS samples is not a universal ordered-bit promise.

Decoded scalar FFMA is present in all four entries. No MMA-family mnemonic appears in these complete selected-entry listings; this is a static listing observation, not a dynamic tensor-core utilization claim. HFMA2 zero-initialization idioms and optional epilogue paths are not treated as evidence of reduced-precision GEMM or executed activation. Unknown annotations remain explicit.

| Decoded entry | Instruction rows | FFMA | LDG | LDS / STS | LDGSTS | BAR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mapal |232|64|128|0 /0|0|0|
| cuBLAS GEMM |1,680|288|40|113 /64|50|24|
| cuBLAS reduction |880|11|134|0 /0|0|0|
| CUTLASS |736|136|32|50 /36|0|10|

These are static listing counts including padding, branches and optional paths, **not executed operation counts or runtime costs**. cuBLAS GEMM's113 LDS rows include22 explicitly `@!PT` constant-false instructions. Decoder addresses cover each selected text range contiguously; no dynamic cache traffic, occupancy, stalls or instruction-path counters were collected.

The evidence supports concrete experiments, not automatic promotion:

| Observed distinction | Existing fact/consumer | Smallest discriminating next test |
| --- | --- | --- |
| Independent-output Mapal versus shared, multi-accumulator code | Dat `TileSite`/`TileRead` geometry, strides, ordered fold; Trn `gpu.rs:launch`, `Kernel::naive/tiled`; Dat `Gpu`/`GpuTile`; Loc Gmem/Smem/Reg | Ordered output coarsening/geometry and staging experiments using existing legality gates; vary aspect and panel depth separately, retain frozen controls/nulls and independent confirmation. Prior shared-panel variants had regressions; static differences alone do not select a winner. |
| cuBLAS LDGSTS/dependency code | Existing affine access/alignment and target capability facts; backend panel-copy realization | A separately legal ordered-copy/pipeline alternative, holding arithmetic/output mapping fixed where possible; inspect its exact images and measure ordinary distributions. Presence alone proves no current latency cause or missing shared-IR field. |
| Extra cuBLAS reduction stage and changed ordered bits | Existing ordered `TileSite` fold and scalar step; `EmitOpts::contract` permits contraction, not reassociation | Keep ordered-FMA unchanged. Any parallel partial/tree alternative needs an explicitly separate numerical permission, a-priori error bound, workspace/completion plan, and full numerical/ordinary tests. `TileKSplit` address decomposition is not that permission. |
| Image cannot explain host/API or lifetime cost | Dat last-use/CellSite/footprint; Trn `ResidentPlan`, `FnEmit::arena/epilogues`; Loc device storage and Trm launch/transfer completion | Same-kernel matched submission/retention tests, plus inclusive allocation timing. This primitive harness uses preallocated CUDA buffers; its timings exclude initial buffer allocation and final release. Whole-program arena/lifetime costs require a separate inclusive measurement and cannot be declared this kernel gap's cause. |

The source fact map predates50a57e5 and must retain that identity. Its arithmetic-policy/emitter mismatch is already corrected in current source; this capture is not evidence that it remains missing. Shape/layout/order/hardware facts and cooperative realizations already exist. The remaining performance questions concern selecting and realizing legal schedules, and measuring whole GPU flow under the correct numerical contract.
