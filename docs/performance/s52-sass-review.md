# S52 — SASS review: Mapal vs CUTLASS (RTX 4090, n=4096)

Static review of `prof/mapal.sass` (Mapal shipped FP32 kernel, v4) and `prof/tile_bench.sass`
(CUTLASS SIMT `MmaPipelined 128x128x8` and TF32 `MmaMultistage 128x128x16`). No perf counters
(vast blocks them). Evidence tarballs: `prof-*.tgz` from `benches/gpu/box_profile.sh`.

## FP32, per k step

| per k step | Mapal | CUTLASS SIMT |
| --- | ---: | ---: |
| FFMA | 64 | 64 |
| LDS.128 | 4 | 4 |
| LDG | 0.25 (.128) | 1 (.32) |
| STS | 0.5×.32 + 0.125×.128 | 1×.32 |
| BAR | 0.125 (2/panel) | 0.125 (1/tile) |
| LDL/STL (spills) | 0.31 / 0 | 0 / 0 |
| int/addr/pred ops | 1.4 | 8.4 |
| total | 70.8 | 78.6 |
| FFMA share | 90.4% | 81.4% |
| registers | 128 (cap), 52 B spill | ≤ R119 |

Mapal issues *fewer* instructions per k; the gap (≈5%) is pipeline stalls:

1. **Prefetch issued late.** Under the 128-register cap ptxas sinks the next-panel LDGs behind ~85%
   of the panel's FFMAs (0x42e0–0x4330); only ~15% of the FFMAs cover their latency, and the
   consuming STS sits between two BARs. CUTLASS issues its LDGs after 12% of the FFMAs.
2. **4-way bank conflict on transposed-A STS** (512 B row stride) on the barrier-bracketed path.
3. **Spills on the critical path**: 64-bit global pointers reloaded (LDL.64) to feed the LDGs;
   64-bit loop counter.

cuBLAS ran `cutlass_80_simt_sgemm_256x128_8x4` (4-stage cp.async, 4-byte A copies) on Ada and
`sm80_xmma_gemm_..._128x128x8_stage3_..._ffma` on Hopper; at n=4096 it buys ≈0 over CUTLASS-128.

## TF32 (Mapal PTX vs CUTLASS SASS)

| per 16-k tile, per warp | Mapal (v4) | CUTLASS |
| --- | --- | --- |
| warp tile / HMMA | 64×32 / 32 | 64×64 / 64 |
| A fragments | 32 ld.shared.b32 | 8 LDSM.x4 |
| stages | 2 | 4 |
| smem layout | padded | XOR swizzle |
| fragment double-buffer across panels | no | yes |

## Ranked changes (review's list; status after bundle 7)

1. FP32 cp.async staging — **built as rung `async`; measured NEGATIVE at ≥2048 on all three parts**
   (+27% only at 1024 on 5090). Likely: 4-byte A copies (48/thread/panel) + still spilling.
2. TF32 XOR swizzle + deeper pipeline + cross-panel fragment prefetch — not built.
3. TF32 whole-thread register rule (64×64 warp) — **built as `wide`: +16–19% H100, +1–2% 4090, ~0 5090.**
4. FP32 A-load lane map (2 lanes per row) + pad — not built.
5. 32-bit indexing from site extents — not built.
6. TF32 ldmatrix.x4 — **built: ~0 effect on all three.**
7. 256×128 FP32 tile — only after the pipeline is fixed.
8. TF32 smem-staged epilogue — not built.
