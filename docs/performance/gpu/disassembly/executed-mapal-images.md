# Actually executed Mapal kernels

All four frozen old/S58 Contract captures independently join exact input PTX, actual CUDA context/context UID, module ID, function handle/name and successful launch geometry to callback-saved CUDA ELF text and exact-entry nvdisasm output.924input and69output files independently rehashed; ordinary_timing_admissible=false.

| Case | Grid / threads / sharedB / registers | Kernel text bytes | Static instruction rows | FFMA | LDG | LDS | STS | BAR |
|---|---|---|---|---|---|---|---|---|
| dense-old | 1152x1 / 256 / 0 / 40 | 4480 | 280 | 48 | 96 | 0 | 0 | 0 |
| dense-candidate | 576x1 / 32 / 1664 / 79 | 3840 | 240 | 128 | 6 | 16 | 6 | 2 |
| qwen_expert-old | 64x1 / 256 / 0 / 40 | 3712 | 232 | 64 | 128 | 0 | 0 | 0 |
| qwen_expert-candidate | 32x1 / 32 / 1664 / 79 | 3840 | 240 | 128 | 6 | 16 | 6 | 2 |

All runtime local_bytes are0. Both S58 cases visibly contain cooperative shared loads/stores and barriers in the actually launched function; the old cases have none of those mnemonics. These are static instruction inventories including loop bodies and padding, not executed instruction totals or memory transactions. Static LDG/FFMA counts cannot be compared as dynamic ratios across different loop unrolling/tiles. No achieved occupancy, traffic, issue stalls or timing cause is established.

Each row has exact full-module and kernel-text byte hashes plus SASS/event/receipt hashes and every launch identity in `images-attempt05-independent/analysis.json`. Cubin hashes apply to actual callback bytes; no cuBLAS/CUTLASS image or B image is included. No Exact or TF32 face is captured. The capture preserves sampled256-cell ordered-FMA completion, not an exhaustive full-size oracle.

Inputs `/private/tmp/mapal-s60-kernel/diagnostics`; immutable result `/private/tmp/mapal-s61-gpu/attempt-05/retrieved/trace/kernel-diagnostics`. CPU readiness under sibling `cupti-layout-readiness` succeeds with real package/header/library/RPATH checks and the frozen capture source unchanged.

Evidence: [complete experiment directory](../../../../benches/results/gpu/disassembly/executed-mapal-images).

The callback ELF notes identify driver-JIT ptxas 13.3.44 targeting sm_120a, while the saved standalone nvdisasm is toolkit 12.8. Some scheduling annotations are not fully decoded. Shared-load/store/barrier presence is supported by PTX and queried resources; detailed newer-ISA interpretations require a matching decoder. An HFMA2 involving constants/zero registers does not establish FP16 GEMM.
