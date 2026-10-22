# Residual MLP with competitive fused library references

Updated 2026-10-22. On one RTX5090, tuned fused cuBLASLt takes **0.032659 / 0.059620 ms** at the two widths, versus Mapal **0.073090 / 0.128935 ms**. It is **2.238× / 2.163× faster**. Tuned fused CUTLASS also has lower observed medians at both widths, narrowly at the larger width. Mapal beats the tested eager and graph-disabled Inductor arms. This does not establish the cause of the Mapal gap.

## Same-device whole-program comparison

The fixed graph is `p=X·Wo; h=ReLU(p·Wup+b1); m=h·Wdown+b2; y=(p+m)/4`.
All rows are synchronized host-wall upper medians of30 measured invocations after5 warmups. Lower is better. Compilation, tuning, setup and input preparation are outside these steady clocks. These are whole-program measurements, distinct from the small-M primitive CUDA-event results.

| Implementation | M128/d256/h512, ms | M128/d1024/h2048, ms |
| --- | ---: | ---: |
| Mapal ordered FMA | 0.073090 | 0.128935 |
| Handwritten naive FMA, original controls | 0.054452 | 0.333153 |
| cuBLAS pedantic, original controls | 0.039069 | 0.092890 |
| Tuned cuBLASLt, unfused | 0.036255 | 0.063416 |
| Tuned cuBLASLt, fused | 0.032659 | 0.059620 |
| Tuned CUTLASS, unfused | 0.047192 | 0.127954 |
| Tuned CUTLASS, fused | 0.046711 | 0.123867 |
| PyTorch eager | 0.115424 | 0.140602 |
| PyTorch Inductor, CUDA graphs disabled | 0.178659 | 0.208715 |

Native references and original Mapal/PyTorch controls ran in sequential phases on the same physical GPU, UUID `GPU-45dae734-7dbd-fc3f-ea9a-35302e563150`, driver580.142, nvcc12.8.93 and cuBLAS/cuBLASLt120804. PyTorch is2.13.0+cu130 with CUDA13.0 and both precision settings recorded as IEEE. Driver API13000 is a separate version field. UUID/model/driver and GPU-idle checks passed between phases. Independent-host repetition and randomized phase order were not performed.

The new native harness also preserves naive and cuBLAS controls. Their medians are0.055024/0.333596 and0.039480/0.093041 ms respectively. The table uses the original control harness for these rows; no mixed minimum across repetitions or harnesses is selected.

## What fusion changes in these references

| Library and fixed selected configuration | Unfused→fused, smaller ms | Unfused/fused speedup | Unfused→fused, larger ms | Unfused/fused speedup |
| --- | ---: | ---: | ---: | ---: |
| cuBLASLt, identical selected GEMM algorithm configurations across modes | 0.036255→0.032659 | 1.110× | 0.063416→0.059620 | 1.064× |
| CUTLASS,32×64×8 block /16×32×8 warp | 0.047192→0.046711 | 1.010× | 0.127954→0.123867 | 1.033× |

These correspond to latency reductions of9.92%/5.99% for cuBLASLt and1.02%/3.19% for CUTLASS. They are observed median differences from separate fresh processes, not confidence intervals. The small CUTLASS difference needs repetition before a strong performance claim.

Fused library calls apply bias/ReLU in the up projection and bias in the down projection. The residual-plus-quarter operation remains explicit. The unfused path requests3 GEMMs plus3 pointwise calls; the fused path requests3 GEMMs plus1 pointwise call. These are source/API request counts, **not an observed count of internal library kernels**. Actual library image/dispatch capture remains pending.

## Selection and arithmetic contract

CUTLASS searched7 separately compiled, two-stage FP32 SIMT geometries with K8, instruction1×1×1, split-K1 and scalar alignment. All7 compiled and passed tiny gates. Both shapes and both fusion modes selected block32×64×8, warp16×32×8 (candidate2). The128×128 default remains a separate held-out control: unfused/fused0.092180/0.087933 ms at the smaller width and0.293545/0.291312 ms at the larger width. Tuning matters more here than the measured epilogue delta.

cuBLASLt uses FP32 storage and `COMPUTE_32F_PEDANTIC`, with `NVIDIA_TF32_OVERRIDE=0`. It requests up to8 supported heuristics per site, then performs one fixed p/h/m coordinate sweep using9 warmup-excluded whole-program samples. Selected filtered indices are[0,1,0] at the smaller width and[0,3,0] at the larger width for both fusion modes. Exact opaque algorithm bytes/configuration are frozen across fresh confirmation processes. This finite sweep does not prove a global optimum. CUDA12.8's supported bias layout is implemented by a column-major interpretation of the transposed GEMM, preserving the original row-major bytes and logical graph.

Mapal and naive CUDA require the ordered FMA scalar result. Library and PyTorch arms must pass the frozen componentwise budgets against **both** operationwise and FMA scalar references. Passing those budgets is not a bit-exact ordered-reduction guarantee and does not identify hardware instruction units. FMA is ordinary multiply-add with one rounding; tensor-core use requires separate evidence.

## Retention, allocation and limits

All native variants retain the same input/tensor buffers and reserve even the unused raw intermediates:12 explicit tensor buffers totaling2,493,440/25,702,400 bytes. cuBLASLt additionally reserves32MiB workspace outside clocks. Private library allocations are unmeasured. This controlled native comparison intentionally does not claim a memory-allocation benefit from fusion.

Mapal's frozen original controls request an arena of1,838,080/22,032,384 bytes outside clocks. They regenerate immutable inputs before every timed invocation; native and PyTorch arms retain inputs. That difference can change cache state. Neither the whole-program gap nor the library fusion deltas establish a causal Mapal retention or allocation gain. This frozen compiler predates the separately validated unused-output allocation correction.

PyTorch uses the previously reviewed pinned fullgraph/static setup with CUDA graphs disabled; these rows do not represent a best-tuned graph-enabled PyTorch baseline. Recurrent/incremental programs,16K composed workloads, repeated-host confirmation and a generic algorithm selector remain open.

Tiny cases export full inputs, stages and final output. Representative shapes check256 selected cells per stage/final request against the frozen full-width-reference contract; the larger reference covers selected full rows. The fused checker obtains raw GEMM checkpoints through separate checker calls outside clocks: those values are not observations of hidden fused accumulators. Each native timed final output is exported and checked after the clock.

## Evidence and next action

The [raw experiment and independent checks](../../../../benches/results/gpu/workloads/residual-mlp/fused-references/rtx-5090/README.md) preserve all builds, tuning trials, selections, fresh confirmations and original controls. A separate root implementation imports no harness checker and verifies770,816 exported native IEEE words and both clocks across148 executions (2,020 timing pairs including warmups). No native arm was rejected or silently replaced. Full protocol/selection and original-control admission is recorded alongside it.

Next, obtain exact executed cuBLAS/CUTLASS images through the reviewed counter-free callback path, then compare instruction families, reduction organization, tiling and launch structure against the [existing Mapal images](../disassembly/executed-mapal-images.md) and [IR fact map](../disassembly/ir-facts-and-realization-gaps.md). Static disassembly alone cannot attribute stalls, cache behavior or the measured time gap. The failed Nsight counter diagnostic is retained separately.

[Earlier residual MLP run](residual-mlp.md) · [Primitive CUTLASS tuning](../kernels/competitive-cutlass.md) · [Current GPU results](../current-results.md)
