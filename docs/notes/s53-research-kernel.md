# S53 research: the kernel gap, per architecture

Sources: `benches/results/gpu/kernels/initial-comparisons/v4–v6` (`table.py`), bundle-7 `prof-*.tgz` (ptxas, SASS, nsys), the SASS
review, `gpu.rs`, `profile.rs`. Ratios are within-run, cuBLAS ms / Mapal ms (>1 = Mapal faster).

## 1. Measured gap (v6)

**Label correction.** The 0.95× headline is the **contract face** (`_c`, `llvm.fma`). The **exact face**
(`_exact`, `mul.rn` + `add.rn`, the default) is 0.42–0.64× cuBLAS at ≥2048: two instructions per MAC,
0.54–0.70× the contract face on the same kernel. Every rule below applies to both faces unchanged.

| part | face | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| 4090 | contract, shipped (async ON) | 0.78 / 1.28 | 0.92 / 0.92 | 0.87 / 0.88 | 0.87 / 0.88 |
| 4090 | contract, `off-async` | 0.53 / 0.87 | 0.93 / 0.93 | 0.95 / 0.95 | 0.98 / 1.00 |
| 4090 | exact | 0.58 / 0.96 | 0.64 / 0.64 | 0.59 / 0.59 | 0.57 / 0.57 |
| 4090 | TF32 | 0.48 / 0.74 | 0.89 / 0.90 | 0.91 / 0.93 | 0.93 / 0.95 |
| H100 PCIe | contract, `off-async` | 0.63 / 0.89 | 0.77 / 1.01 | 0.95 / 1.06 | 0.87 / 1.07 |
| H100 PCIe | exact | 0.37 / 0.52 | 0.44 / 0.57 | 0.46 / 0.52 | 0.42 / 0.54 |
| H100 PCIe | TF32 | 0.51 / 0.73 | 0.49 / 0.77 | 0.43 / 0.74 | 0.34 / 0.74 |
| 5090 | contract, `off-async` | 0.48 / 1.00 | 0.80 / 0.94 | 0.85 / 0.94 | 0.85 / 0.96 |
| 5090 | exact | 0.59 / 1.24 | 0.53 / 0.62 | 0.57 / 0.64 | 0.57 / 0.66 |
| 5090 | TF32 | 0.42 / 0.69 | 0.86 / 0.87 | 0.90 / 0.89 | 0.86 / 0.89 |

Cells: vs cuBLAS / vs CUTLASS. CUTLASS ms / cuBLAS ms at ≥2048: FP32 4090 0.96–1.01, 5090 1.00–1.18,
H100 1.11–1.30; TF32 4090/5090 0.98–1.04, H100 1.56–2.13 (wgmma).

**Shipped-default regression.** `GpuRungs::default()` has `cp_async: true` everywhere. Its off arm: async
costs 2/8/8% (4090), 6/8/8% (H100), 2/3/4% (5090) at 2048/4096/8192; it wins only at 1024 on the 4090
(+46%) and 5090 (+27%), and loses 6% there on H100.

**Profile fact error.** `RTX_5090.max_blocks_per_sm = 32`; the box reports 24.

## 2. Causes, by evidence

1. **Ada/Blackwell async: two stages.** The 2-block share is 49.0 KB; a bk=16 stage with `pad_a = 8` is
   16.5 KB, so 2 stages and `wait.group 0` (`DEPBAR.LE SB0,0x0`): one panel in flight, the lookahead
   register prefetch already has. No spills (r125, 0 local), one barrier per panel, A copies already folded
   to immediates (`LDGSTS.E [R89+0x880],[R76.64+0x50]`). cuBLAS's Ada kernel (`256x128_8x4`) uses the same
   4-byte A copies with 4 stages, so depth is the difference.
2. **H100 async:** 3 stages, but 64 B spilled; 7 `LDL` in the per-panel issue block.
3. **The winning register-prefetch kernel spills 48–56 B on every part:** four 64-bit global pointers,
   reloaded by `LDL.64` inside the loop right before the prefetch `LDG`s (4090 0x39c0–0x3a80). Every index
   is `i64` (`Kernel::op`), one pointer per copy. The SASS review adds the transposed-A `STS` 4-way
   conflict and the late-sunk prefetch.
4. **Register tile = half the file.** cuBLAS holds 128 accumulators per thread (Hopper `128x128x8_stage3`,
   128 threads; Ada/BW 256×128, 256 threads); Mapal 64. That is the H100 FMA gap (Mapal = CUTLASS-128
   ×1.01–1.07) and the 5090's; on the 4090 it buys cuBLAS nothing.
5. **1024 is warp-starved.** Four sub-partitions per SM, each issuing only its own warps. 4090: FMA 512
   warps on 512 schedulers; TF32 256, half the schedulers empty. Async pays only here. cuBLAS beats
   CUTLASS-128 1.4–2.1× at 1024 with a kernel nobody has seen (nsys ran only at 4096).
6. **TF32 on Ada/BW: the same 2-stage drain** (bk=16; pads are 13.5% of a stage). CUTLASS: same tile,
   4 stages, XOR swizzle, fragment double-buffer. Not issue-bound: `ldmatrix` ≈0, `wide` ±2%.
7. **TF32 on H100: issue-bound, then the wrong instruction.** `cvt.rna` lowers to `FSETP` + predicated
   `IADD`: 128 `FSETP` per 128 `HMMA`. `wide` paid +16–19% only here. Beyond that, `wgmma`.

## 3. Changes, as derivation rules

| # | Rule | Evidence / bound | Effort |
|---|---|---|---|
| R1 | **Per-arch rung record.** `TargetProfile::gpu_rungs: Option<GpuRungs>`, set only on measured parts, each `false` citing its results row (`DispatchCost` discipline). Unmeasured parts get only rungs positive on every measured arch. Async keyed by R3's regime (warps per scheduler < 2): ON there on sm_89/sm_120, OFF elsewhere and on sm_90. `ldmatrix`: no positive cell at ≥2048, so OFF. Effects depend on the base (min-blocks +18–34% on the v4 register path, ±2% on v6 async), so a record is one configuration measured whole. | 4090 contract 0.87 → 0.95/0.98 at 4096/8192 (own off arm) | S |
| R2 | **One base, 32-bit offsets.** One 64-bit base per operand, advanced per panel; each copy's offset is `off(tid) + imm(p)`, computed in Rust from the lane map (`tid < threads` is known); `i32` when `bounds_proof` bounds the byte extent below 2³¹. | 48–64 B spills, every part | S–M |
| R3 | **Warp fill.** Grid warps ≥ 2 · subpartitions · `sms` (2: `GPU_RESIDENT_BLOCKS`'s reason, per scheduler; subpartitions = 4 from a `Gpu::sm()` table). Shrink the register tile (FMA) or warp tile (TF32 64×64 → 32×32) before the block. Overrides `wide`. | 1024: FMA 0.48–0.78, TF32 0.42–0.51 | S |
| R4 | **Zero-pad swizzle, then stages.** Column `c ^ s(row)` replaces padding, `s` the smallest mask the `conflict_free_pad` search finds that keeps 16-byte runs whole (FMA A: `c ^ 8·(k mod 4)`). Unpadded stages are 16.0 KB, so 3 fit the 49.0 KB share at bk=16 (FMA and TF32). R1 records which derived candidate wins: 2 blocks × 3 stages, or 1 block × max stages (CUTLASS's 4). `cp.async` only with ≥3 stages. | Ada/BW async −2…−8%; TF32 0.86–0.93 where cuBLAS = CUTLASS | M |
| R5 | **A-staging lane map** (register path). Fewest lanes per A row whose run fills a 32 B sector (2 × 16 B), then `conflict_free_pad` for that store: pad 4, conflict-free (now 4-way). | SASS review cause 2 | S |
| R6 | **Whole-file FMA tile.** `wide`'s rule: accumulators + 2·(tm+tn) + addresses ≤ 255 → 8×16 (208 regs); threads 65 536/208/2 rounded down to a power of two = 128; 128×128 block. cuBLAS's Hopper kernel, derived. After R2. | cuBLAS/CUTLASS-128: H100 1.11–1.30, 5090 1.00–1.18, 4090 ≈1.0 | M |
| R7 | **TF32 issue diet.** No per-fragment `cvt.rna` when the operand is already TF32 (ADR-0039 D2 `to_tf32`); interim, convert each staged element once in shared memory, not once per reading warp. Fixes the tie mismatch too (kernel `rna`, ADR RNE). Plus fragment double-buffering. | H100 `mma.sync` 0.74× CUTLASS-sm80 | S |
| R8 | **Hopper TF32 `wgmma`, operands swapped.** TF32 shared-memory operands must be K-major; B is N-major. Compute Cᵀ = Bᵀ·Aᵀ: B becomes the register A-operand (`LDS` from any layout, rounded there), A the K-major shared-memory B-operand, pre-rounded (R7) since nothing converts it in flight. Epilogue transposes through shared memory. Then rungs `tma`, `warpspec`. Check RS-tf32 in the PTX ISA first. When B is read by more than one site, its K-major rounded copy is a restructured operand kept for the later uses: the general use-pattern cache deduction (an LLM's KV cache is one instance), not a GPU special case. | H100 TF32 0.34–0.49 | L |
| R9 | **L2 rasterization, measure first.** H100 8192: cuBLAS +9% over its 4096 rate, Mapal and CUTLASS −5%; a cuBLAS kernel switch explains that as well. Build (bands of `round(√(W·bn/bm))` tile rows, `W = sms·blocks_per_sm`, new `Gpu::l2_bytes`) only if nsys shows the same cuBLAS kernel at both sizes. | weak | S |

**Not proposed.** *Split-K / Stream-K:* `exact`/`contract` pin the in-order chain and ADR-0039 D3's `block`
keeps "the k-steps themselves in order"; a serial split-K is legal but adds no parallelism. It needs
ADR-0028 D4's deferred canonical-tree re-pin; R3 is the legal 1024 fix. *Wave quantization* (5090 4096:
3.01 waves) is paid by cuBLAS and CUTLASS too. *256×128 tiles* only through R6 (on Ada cuBLAS's buys
0.99–1.04×). *Shared-memory epilogue* only for R8: `STG.64` already fills 32 B sectors. *16-byte A copies:*
cuBLAS reaches its rate with 4-byte ones; revisit as the TMA prerequisite.

## 4. Test plan

- Each rule: a `GpuRungs` field, `emit --gpu-off=<name>`, an entry in `bundle.sh`'s arm list (R4/R6's off
  arm is today's geometry). Add `_exact` arms: rung costs differ by face (H100 exact 4096: 0.57 → 0.46,
  v4 → v6).
- One vast session per part, `PROFILE=1`, ON vs OFF interleaved against cuBLAS and CUTLASS, 1024–8192.
  Gate: differential N/N PASS, `bits` = yes, runtime `regs`/`local_bytes` recorded (driver JIT and offline
  ptxas disagree: 5090 async 0 vs 68 B spilled).
- After R1 moves the shipped base, re-run every off arm on it.
- `box_profile.sh`: nsys at 1024 and 8192 too (sets R3's target, decides R9).
- Append rows to `benches/results/gpu/kernels/initial-comparisons/` as they land; ship ON per arch only where positive.

**Order:** R1 (+ the 5090 fact) → R2 + R5 → R3 → R4 → R6 → R7 → R8; R9 after nsys.
