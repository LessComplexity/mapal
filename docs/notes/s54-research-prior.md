# S54 research: prior art and the uniqueness check

Date: 2026-10-21 · Branch `s53-gpu-opt` @ 6d105b0 · Builds on the S53 research notes and does not repeat them.

## 0. A finding about the probes

Every probe ends with `(0.0, d) -> fold { acc + v }`: an exact left fold over all n² cells. ADR-0028 D4 keeps float folds serial. That fold is the host reader that forces the whole-`d` d2h in `resident.rs`. The epilogue is not what forces it. Moving the epilogue onto the device therefore saves the host map pass and **zero transfer bytes**.

Library reductions (`torch.sum`, cub `DeviceReduce` even in CCCL 3.1's gpu-to-gpu deterministic mode, EVT's `atomic_add` scalar reduction) compute a different value than a left fold. A fair fusion benchmark therefore needs two variants of each probe:

- **(a)** with the sum, labelled "library value ≠ oracle value".
- **(b)** with element reads only, or a per-row sum. Per-row left folds are exact and parallel over rows.

## 1. Comparison

| system | fusion / placement decision | multiple computations (independent, chains, loops, reduce-after-GEMM) | per-arch choice | lacks (Mapal proves) · stronger |
|---|---|---|---|---|
| XLA:GPU | Cost-model priority fusion, multi-output and sibling fusion. A GEMM becomes cuBLASLt with rewritten epilogues or a Triton fusion tiled by **indexing maps / SymbolicTileAnalysis**. Only the jit region is on the device. | Horizontal fusion; command buffers (CUDA graphs). Reduce-after-GEMM is a separate fusion. Loops: `while` with in-place dynamic-update-slice. | GEMM autotuner: cuBLAS algorithms vs Triton configs, measured, cached. | **Lacks:** host program, write deltas, bit-identity (reassociates float reductions). **Stronger:** element-level affine footprints, the same class as `TileRead`/`ElemSrc`, so not unique to Mapal; a cost model; tuned tiles. |
| TVM / Ansor / Relax | Rules by op pattern (`kElemWise` … `kOutEWiseFusable`, `kOpaque`) over a post-dominator tree. `StaticPlanBlockMemory`. | A complex op absorbs elementwise outputs but never chains into another complex op, so GEMM→reduce or GEMM→GEMM is 2 kernels. | Ansor/MetaSchedule: learned cost model plus on-device measurement. | **Lacks:** exactness, deltas, host view. **Stronger:** per-shape search. |
| Triton + Inductor | Fuses pointwise/reduction nodes by bytes saved. GEMM epilogue fusion only under max-autotune with Triton or CUTLASS (EVT) templates; prologue fusion limited. | **Combo kernels** put N independent kernels in one launch (experimental, off by default). cudagraph trees. Split reductions. | Runtime benchmarking of configs, cuBLAS (ATen) included. | **Lacks:** values across graph breaks / host Python; numerics drift from eager (approximate division, libdevice ULP), with only knobs to emulate eager. **Stronger:** arbitrary pointwise epilogue DAGs, measured per shape. |
| Halide | Schedule (`compute_at`/`store_at`) is separate from the algorithm. Bounds inference by intervals. **Sliding window / storage folding** recomputes only a producer's new rows as the consumer advances. | Within one pipeline. | Autoschedulers (Adams 2019 learned model, Li 2018, Anderson 2021). | **Lacks:** retention is chosen by the schedule, not deduced; `rfactor` reorders floats. **Stronger:** the closest prior art to `retain-delta`. |
| Futhark (Dex similar, no GPU bar) | Rule-based SOAC fusion (producer–consumer and horizontal). Uniqueness types for in-place updates. | Incremental flattening with multi-versioned code. | Autotuned version thresholds. | **Lacks:** `reduce` may use any order, so no exact float face; host is outside. **Stronger:** nested parallelism. |
| JAX | jit → XLA. Arrays stay on the device. `x[0]` slices on the device before transfer. Donation is explicit. | Async dispatch overlaps host Python. KV caches are hand-written (`dynamic_update_slice`). | XLA's | **Lacks:** deduced residency and deltas; the user owns both. |
| TensorRT | Pattern-rule layer fusion, incl. horizontal merge of layers sharing an input. | Auxiliary streams; user-set persisting-L2 limit. | Tactic timing per GPU. | **Lacks:** whole program, exactness. |
| cuBLASLt | Fixed epilogue enum: `BIAS`, `RELU(_BIAS)`, `GELU(_BIAS)`, `*_AUX`, `DRELU/DGELU`, `BGRADA/B`. Also alpha/beta. | **Grouped GEMM** (cuBLAS ≥ 12.5): N GEMMs in one launch. | Heuristic algorithm query. | **Lacks:** consumers, arbitrary laws, chains. **Stronger:** SASS kernels (Mapal's exact face is 0.48–0.66×). |
| CUTLASS EVT | Arbitrary epilogue DAG (topological visitor): broadcast, auxiliary load/store, row/column/scalar reductions (cross-CTA through a GMEM workspace; scalar via `atomic_add`). The b2b example fuses a GEMM chain only when the first GEMM's N fits one CTA tile. | Grouped GEMM, Stream-K. | Profiler (exhaustive). | **Lacks:** residency, order-pinned reductions. **Stronger:** the epilogue machinery Mapal must match. |
| FlashAttention 2/3 | Hand-derived GEMM–softmax–GEMM fusion by online-softmax reassociation (not bit-identical to naive). | — | Hand-written per arch (FA3: wgmma, TMA, warp specialization). | Under D4, Mapal cannot derive this without the tree fold (ADR-0028 deferred option). |
| CUDA Graphs / PDL | Mechanisms, not deciders. Graphs remove launch cost, keep concurrent branches, and keep IF/WHILE conditional nodes on the device (12.4+). **PDL needs cc ≥ 9.0** (H100, 5090). | — | — | — |
| Research | Rammer (OSDI'20): independent operators in one persistent kernel. Welder (OSDI'23): tile-graph fusion. **MPK** (arXiv 2512.22219): SM-level task graph in one megakernel, 1.2–6.7×. Neptune (2025): loop-carried reductions fused with algebraic correction (inexact). CGCM (PLDI'11): automatic CPU–GPU communication per allocation. | | | MPK is the bar for "a boundary only where needed". |

## 2. Uniqueness check

**Not unique; Mapal must match these:** affine element footprints (XLA indexing maps, Halide bounds), use counts, independence (Rammer, combo kernels, grouped GEMM), closed-form producers fused into consumers (XLA, Inductor), immutability (HLO, JAX), shared-operand GEMM merge (TensorRT, hand-fused QKV).

**Unique, with evidence:**

1. **U0: every placement is value-invariant against an oracle (enabler).** Fusion is not a numerics decision, so it can default ON with no tolerance budget. Evidence: S53 differential 126/126 and 134/134; `devgen` refusing float `%` (`fmodf` ≠ NVPTX `frem` in 2051/4096 cells). Inductor, XLA and EVT all drift.
2. **O1: residency over the host program with element-exact host reads.** Mapal sees host loops, `println`/`time` fences and `LastUsePlan` deaths, and moves only the cells a host morphism reads. Evidence: e2e_4096 went from 128 MB in / 64 MB out to 0 / 4 B (S53). CGCM works per allocation; JAX gets the same bytes only when the user writes the slice. Today's gap: per-cell maps are never placed on the device (`gpu_site` needs k>0).
3. **O2: write deltas pushed through tile footprints** (`write_delta` → `Delta::Rows`, affine in the counter). The KV cache becomes a consequence. Prior art (Halide sliding window, schedule-directed; arXiv 2307.14988, runtime) does not derive it from a naive loop. Evidence: the fact exists, but the win is unshown. S53 measured `retain-delta` below `resident` alone (decode 0.63–1.11 s). `resident`×`retain-delta` is S54 queue item 1.
4. **O3: order of use across a `Trm`.** The host fold reads `d` row-major; the tile site finishes row panels (`TileRead` + `LastUsePlan` positions). An event per panel plus chunked d2h overlaps transfer and host fold with the kernel. Frameworks need hand-written chunked async copies. Composed S52 H100 n=4096 bound: d2h 3.2 ms against a 4.2 ms kernel. Not measured.
5. **O4: L2 persistence derived from reuse distance.** Reuse distance comes from `LastUsePlan` positions; bytes come from the type. Comparing them to the `l2_bytes` fact gives an `accessPolicyWindow` (sm_80+) on a chain intermediate between launches. L2 sizes (spec): 3090 6 MB, A100 40 MB, H100 50 MB, 4090 72 MB, 5090 96 MB, so this is a per-arch rung by construction. TensorRT only exposes a user-set limit. Not measured.

## 3. Baselines a fair benchmark must include

Rules for every probe:

- Inputs already on the device; a CUDA-graph version of every eager baseline (at n=512 launch cost is comparable to one GEMM).
- Mapal's `devgen` inputs (closed-form a, b) in a separate column: a library cannot know an input is closed-form.

| probe | Mapal at 6d105b0 | must include | exactness note |
|---|---|---|---|
| `epi` c=a@b; d=c·0.5+1 | 1 launch + host map | cuBLASLt alpha=0.5 + `EPILOGUE_BIAS` (bias=1); EVT; eager GEMM+ew; Inductor max-autotune | ×0.5 is exact, and cuBLASLt applies alpha·acc+bias in Mapal's order |
| `bias` d=c+bias[j] | same | cuBLASLt `EPILOGUE_BIAS` (row-major-as-transpose indexes j); EVT RowBroadcast; eager; `RELU_BIAS` variant | — |
| `ew` 3 per-cell maps | host only | one fused elementwise kernel (Inductor/Triton); eager 3 kernels | report GB/s against peak |
| `pro` a2=a·0.25; c=a2@b | fused (devgen) | cuBLAS alpha=0.25 (1 launch); Inductor prologue | ×0.25 exact, so the library ties with zero cost |
| `indep` c=a@b; d=e@f; g=c+d | 2 serial launches + host add | (1) d=e@f, then a@b with C=d, beta=1 (2 launches, no add kernel); (2) grouped GEMM + add; (3) two streams + add | acc+1·C equals c+d per cell. Concat-K is **not** exact (it changes fold order) |
| `mlp` c=a@b; h=c·0.5; d=h@e | c d2h, host h, h2d | 2 cuBLAS calls with alpha=0.5 on the first, c resident; graph; CUTLASS b2b only where N0 fits a CTA tile | — |
| diamond c→(c·2, c+1)→product | not probed | EVT topological visitor (1 launch); GEMM + one Inductor kernel; eager 4 | cuBLASLt cannot express it |
| fan-out d=c@e; f=c@g | not probed | concat-N single GEMM (bit-identical per cell); 2 calls with c resident | Mapal can prove the merge exact from identical `TileRead` |
| reduce after GEMM (row sum) | not probed | EVT Row/ColReduction; GEMM + `torch.sum(dim=1)` | library order differs, so label it |
| whole sum (every probe) | host left fold | cub `DeviceReduce` run-to-run and gpu-to-gpu | different value (see §0) |
| loop x←x·W ×8; decode | resident (S53) | eager cuBLAS with device buffers; CUDA graph / WHILE node; PyTorch with a hand-written KV cache; FlashInfer | — |

**Per-arch rungs these imply:**

- PDL: sm_90/sm_120 only.
- Grouped/combo launch: all five arches, measured.
- L2 window: sm_80+, sized by `l2_bytes`.

Sources: [XLA indexing analysis](https://mlir.llvm.org/OpenMeetings/2024-09-12-XLA-Indexing-Analysis.pdf) · [Inductor fusion](https://pytorch.org/blog/why-is-pytorch-compile-so-fast-kernel-fusion/) · [combo kernels](https://hosted-files.sched.co/pytorchconferenceeu2026/d2/Combo%20Kernels_%20Horizontal%20fusion%20optimization%20in%20torch.compile.pdf) · [Inductor numerics](https://gitcode.com/rmch/pytorch/issues/11) · [TVM fusion](https://tvm.apache.org/docs/arch/fusion.html) · [EVT](https://research.colfax-intl.com/epilogue_visitor_tree/) · [cuBLASLt epilogues](https://docs.nvidia.com/cuda/nvmath-python/1.0.0/bindings/generated/nvmath.bindings.cublasLt.Epilogue.html) · [grouped GEMM](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates) · [PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) · [CCCL determinism](https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl) · [MPK](https://arxiv.org/abs/2512.22219) · [Neptune](https://arxiv.org/html/2510.08726v1) · [incremental NNs](https://arxiv.org/pdf/2307.14988)
