# Summary — 2026-10-25 → 28

Ratios are **Mapal time ÷ the best competitor** on the same program and GPU (cuBLAS, cuBLASLt, CUTLASS, PyTorch,
`torch.compile`). Below 1 = Mapal is faster. GPUs: RTX 4090 / RTX 5090 / H100 / RTX 3090.

## What we did

| step | what changed |
| --- | --- |
| cpu-gpu-boundary | Work stays on the GPU instead of bouncing to the CPU: loop counters passed into kernels, maps with inner sums run on the GPU, kernels loaded and memory allocated before the clock starts |
| recurrence-precision | Prefill and decode come out of the loop's dataflow on their own (no hardcoded cases); many threads per cell; convolution as a matrix multiply; the language gets compute types, `fold: tf32 \| f32 { … }` (ADR-0041) |
| site-geometry | Measure only what changed (emission diff + result cache); per-GPU table of sizing settings; TF32 inside the convolution tile; the sizing rules that won on some programs and lost on others were traced to missing facts |
| test harness | Fixed the laptop hangs: the tests started ~196 threads on 14 cores and never timed out emulated GPU runs; the test oracle now runs optimized (777 → 683 s) |
| rules adopted | "A mixed result is a missing fact" (add the fact that separates the cases, don't switch the optimization off); "measure only what changed" |

## Results

Each row is compared against the fastest of all competitors measured for it; the last column names which one won.

| workload | before | now | fastest competitor |
| --- | --- | --- | --- |
| causal attention | 2.87 / 2.74 / 5.62 / 2.89 | **0.18 / 0.26 / 0.25 / 0.14** | `torch.compile` (its matrix multiplies call cuBLAS) |
| attention prefill | 5.57 / 5.15 / 8.62 / 5.51 | **0.18 / 0.22 / 0.26 / 0.15** | `torch.compile` |
| gen (prefill + decode in one natural loop) | — | **0.17 / 0.24 / 0.25 / 0.15** | `torch.compile`; also matches the hand-written prefill (0.95–0.99×) |
| decode | 1.96 / 1.34 / 3.58 / 1.89 | 1.01 / **0.82** / 1.51 / 1.00 | hand-written KV-cache loop on cuBLAS (5090: cuBLAS without a cache) |
| multi-channel convolution | ~2000–5000 (CPU only) | 1.38 / **0.89** / 1.18 / 1.43 | PyTorch → cuDNN (FP32) |
| elementwise chain (ew_4096) | 2.59 / 5.33 / 4.75 / 2.27 | **0.46 / 0.45** / 1.01 / **0.61** | a hand-fused CUDA kernel or `torch.compile` |
| 2-layer MLP, loop-built inputs (rt_mlp2) | 4.08 / 5.6 / 2.27 / 2.78 | **0.75** / 1.00 / 1.17 / **0.88** | `torch.compile` |
| conv, FIR, conv chains, loop_4096 | ahead | **0.45–0.97** everywhere | best of CUTLASS / cuBLAS / PyTorch / `torch.compile` |
| 2048 MLP family | ~1.0 / 1.1 / 1.3 / 1.0 | same | CUTLASS, cuBLASLt or `torch.compile` (varies by GPU) |

Cells at or below the best competitor: 26 of 92 at the start → 54 of 120 after recurrence-precision. Later sweeps
re-time only changed programs.

## Still behind

| family | ratio | why |
| --- | --- | --- |
| decode on H100 | 1.51 | one-row kernel launched once per step; how split sums are combined is not chosen per GPU |
| small problems (512-size chains) | 1.5–2.2 (4090/5090) | per-kernel overhead and block sizing at small sizes |
| matrix multiply on 5090 / H100 | 1.1–1.9 | sizing tuned for the 4090 |
| narrow matrix × matrix/vector (pj) | 0.42–1.86 | wins on 4090, loses at larger row counts and on H100/5090 |
| attention at other sizes | 1.1–2.1 (5090 attn 0.82) | too few threads per row |
| TF32 vs PyTorch TF32 | 1.06–2.62 | TF32 path not on everywhere yet |

## Next steps (not yet approved)

1. **One sizing model for every kind of site.** Compute, from the dataflow graph and each GPU's numbers:
   - the work each block does after a long sum is split;
   - how evenly the last round of blocks covers the compute units;
   - the cost of combining split sums (adding straight into the result vs scratch memory plus one extra pass);
   - whether the inputs are already in the GPU's cache.

   Every sizing rule reads these; the per-GPU search only settles near-ties and times every program a setting
   changes. Targets: small-size chains, matrix multiply on 5090/H100, narrow matrix × matrix/vector, decode on H100,
   convolution block shape.
2. **Attention at other sizes**: more threads per row for the row sums and max.
3. **TF32 everywhere it's allowed**: in the convolution tile on H100/5090, and fill the H100's compute units for
   TF32 matrix multiply (it runs a mostly empty second round today).
4. **Fair attention check**: run FlashAttention / vLLM on the same prefill, decode and causal shapes.
5. Small: run each test while iterating only for what changed; check whether the runtime's thread spinning ignores
   other load on the machine.

## Notes

- Competitors are libraries and PyTorch on the same programs, not FlashAttention / vLLM-class inference kernels.
- Agent tokens: ~1.9 M (cpu-gpu-boundary), ~1.9 M + interrupted runs (recurrence-precision), ~2.05 M (site-geometry).
- Details: `docs/performance/evidence/{cpu-gpu-boundary,recurrence-precision,site-geometry}.md`, session logs in
  `docs/sessions/`.
