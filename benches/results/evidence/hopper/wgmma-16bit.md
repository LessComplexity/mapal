# wgmma-16bit — the 16-bit tensor tiles on Hopper's warpgroup product

Branch `hopper/wgmma16` (from `c808622c`). The bf16 / f16 tile sites on a part whose matrix capability lists the
warpgroup 16-bit product (`wgmma.m64nNk16`) at a measured rate above its warp `mma.sync.m16n8k16` now take the
warp-specialized tile family of `wgmma-ws` with the operand format a parameter. Measured on one rented H100 80GB HBM3
SXM (vast.ai offer 50189139, 1980 MHz busy, 700 W, Broadwell host), $1.83. Raw output (never in git):
`.claude/raw-evidence/hopper/wgmma-16bit/` (`out.tgz`, `box1/`, `final/`, `vast.log`, `spend.txt`).

| | before (`--gpu-off=wgmma16`: `mma.sync`) | after (`wgmma16`) | gpt-fast (same box) |
|---|---:|---:|---:|
| full model bf16 class TTFT 128 / 512 / 2048 ms | 3.92 / 7.44 / 24.81 | **2.70 / 5.60 / 18.11** | 3.58 / 4.47 / 11.45 |
| Mapal / gpt-fast | 1.09 / 1.66 / 2.17 | **0.75 / 1.25 / 1.58** | — |
| decode ms a token (128 / 512 / 2048) | 1.98 / 2.01 / 2.06 | 1.98 / 2.00 / 2.06 (unchanged: GEMV, m = 1) | 1.33 / 1.34 / 1.42 |
| kernel suite: the 10 changed programs, off / on | — | **1.13–2.23×** (every program, null 0–2.5 %) | — |
| projections vs cuBLAS bf16 (m = 2048 / m = 128) | 2.1–2.6× / 1.5–2.5× | **1.05–1.33× / 1.38–1.59×** | — |

Tokens: on and off identical (128 / 128 at every prompt); against HF bf16 eager 43 / 7 / 66 tokens equal before the
first difference (the five-GPU round's 43 / 7 / 66: the bf16 class's own rounding, unchanged).

## Capability, rule, tile

**Capability as geometry** (`profile.rs`). `MatrixOp` gains `mn_major`: a shared-memory operand may be MN-major (the
instruction's transpose bit). `MATRIX_WARPGROUP` lists the 16-bit `wgmma.m64nNk16` (bf16, f16) twice — `a` from a
descriptor and `a` from registers (`group16`), both `b` from a descriptor, the accumulator in registers, `mn_major`;
the TF32 and fp8 forms stay K-major only. `Caps::matrix_group(fmt, a)` finds the warpgroup shape. `Gpu` gains
`wgmma_half_per_clk` (a calibration name; `mma_rates_of_cc` 4 entries); `matrix_flops_per_clk` keys the pipe by the
issuing group (`b`'s home) — unchanged for every listed op; `stage_consume_ns` takes the op itself.

**Rate** (`calib.cu matrix`, new `wgmma` kinds: two warpgroups an SM, `m64n256k{16,8}` back to back from zeroed shared
operands, a group in flight, clock64 spans), on the box:

| op | flops an SM a clock | Tflop/s |
|---|---:|---:|
| `mma.m16n8k16.bf16` (warp, registers) | 2702 | 658 |
| `wgmma.m64n256k16.bf16` (`a`, `b` descriptors) | **4096** | 998 |
| `wgmma.m64n256k16.bf16` (`a` registers) | **4096** | 982–997 |
| `wgmma.m64n256k8.tf32` | 2048 | 494 |

**Rule** (`rules::wgmma16_pays`): the site's format has a warpgroup shape for `a`'s home and its rate exceeds the warp
product's (4096 > 2702 on the H100; no other profiled part lists one: unchanged). Read in `launch_split` before the
warp tile: `b` must be stored in the format (the descriptor operand), `a`'s home is the descriptor where `a` is stored
in the format (the bf16 class's activations: `PrecisionPlan::store` rounds them at their producer) and registers where
`a` is f32 (rounded to the format pairwise in its fragment load, `cvt.rn.bf16x2.f32`). OFF arm: `--gpu-off=wgmma16`.

**Tile — one family over the matrix op** (`GpuWgmmaTile` + `WsForm`, `gpu_wgmma_ws_tile_of`; `gpu_wgmma_ws_tile`
is now its TF32 form). The form's `MatrixOp` decides the orientation: `mn_major` → `c = a·b` as stored (consumer
warpgroup `cw` owns rows `64·cw..` × `n` columns, `bm = 64·g`); else the TF32 product transposed (`bn = 64·g`). The
depth unit is one 128-byte swizzle row of the format (32 TF32 words, 64 16-bit elements); an f32 `a` spans two
digit columns a row; a register `a` holds `4·bk/16` fragment registers; a lanes-major `b` lands in whole 128-byte lane
slabs (`n` a multiple of 64), a k-major one in `n` rows (`n` any multiple of 8: `m64n8` tiles occur). A fold one
stage deep (`k = bk`) is one stage (16-bit only). Same kernel (`wgmma_ws`): the producer's boxes are one code path
over the element widths (`ws_tensors` per-operand recipes, k digits past one swizzle row, `tma-digits` for
lanes-major slabs); the 16-bit form has no rounding pass and no ready barrier. Consumers: `a` descriptor (K-major,
SBO one atom, a k16 step 32 bytes into the digit columns) or fragments; `b` descriptor K-major or MN-major
(`desc_mn_fields`: LBO the slab stride `128·bk`, SBO one 8-row atom, a k16 step 16 rows; transpose bit 1);
`wgmma.mma_async…m64n{n}k16.f32.{bf16|f16}`; both-descriptor tiles keep a group in flight (`wait_group 1`, the
previous stage released after commit), register-`a` tiles wait before reloading (the TF32 protocol). Epilogue: each
pair of columns one 8-byte store, through `store_cells` where there is an epilogue (residual, gate). The
persistent walk (`ws-persist`) and `setmaxnreg` shares are the family's.

**Found and fixed on the part**: a warp-specialized kernel merged into a horizontal launch faulted (`illegal memory
access`, 22 of 22 generated programs that merged; 26 of 26 alone passed). Two changes: the merged member took its
tensor maps byval as an internal function (a local copy — `device_fn` now passes the kernel parameter's address,
unit-tested), and, since the merged member still faulted with that fixed, a warp-specialized launch now launches
alone (`hz_shape`); the model's and the suite's emissions are unchanged by it (text-equal).

## Tests (`crates/backends/llvm/tests/wgmma16.rs`, emulator `emu_wgmma16`)

The emulator reads each descriptor element through its canonical 128-byte swizzled layout (K-major `start +
(mn/8)·SBO + (mn%8)·128 + 2k`; MN-major `start + (mn/64)·LBO + (k/8)·SBO + (k%8)·128 + 2·(mn%64)`), `a` fragments in
the m64k16 layout, each k16 step summed as the warp `mma.m16n8k16` (`mma_sum`); smem operands recorded until their
`wait_group`.

- `ws16_kernel_text_on_sm_90a`: both `a` homes; `ptxas -arch=sm_90a`: no spill, no `wgmma` serialization, HGMMA BF16,
  UTMALDG, SYNCS, no HMMA / LDGSTS; OFF → the warp tile; RTX 5080 / 4090 / A100 never take it.
- `ws16_equals_the_warp_tile`, `deep_stages_equal_the_warp_tile` (`bk` 256: digit columns): byte for byte with
  `--gpu-off=wgmma16`, every emulator mode.
- `generated_ws16_within_the_bound` (`testgen::Wgmma16Prog`, 32 programs at 1 / 2 / 8 / 132 SMs): within ADR-0041 D3
  of the weakest interpreter, equal to the warp tile where it folds in order, equal to the persistent walk; coverage
  e.g. ws16 22, a regs 6, a smem 16, b MN 17, b K 5, f16 3, two consumers 5, ring turns 7, residual 3, gate 3, walk 3,
  refused 10.
- `ws16_launches_alone`; `horizontal::a_tensor_map_parameter_becomes_its_address`; `rung_rules_replay` (174 / 82 /
  18 / 22: the two `wgmma16` lines picked, one `ws-persist` mixed line).
- Unchanged: `hopper_tma` (TF32 ws vs S53 byte for byte), `gpu_wgmma`, `bulk_stage`, `tf32_tile`, `site_geometry`,
  `site_choice`, `half_tile`, `horizontal`, lib (138). Against `c808622c` (`suite_diff.py --base c808622c --arch`,
  every timed program and both cells' five faces): the rtx-5080 147 / 147 text-equal; the h100 133 / 147, the other
  14 the 10 programs `--gpu-off=wgmma16` moves and 4 TF32 programs whose code is equal (an indented `; mapal.calls.host`
  evidence comment: the ws launch no longer a `calls` candidate, `formed=0` before).
- **On the H100**: 48 generated programs (`dump_hw_programs`) + 24 of them on the persistent walk: **72 / 72 within
  the bound** (`hw_check.py`); the suite's value gates 10 / 10 (`on`, `null`, `off-wgmma16` each within D2 / D3; on
  and off print identical values).
- Formatting (review fix, after `b3ad75f4`): `testgen::wgmma16_src` / `block_src` returned compacted source
  (`mapal fmt --check` on `wg16_00`: needs formatting, 2 lines past 120 columns); both now return
  `format_source`'s layout, as do the fixed `PJ` / `MERGED` programs (tests `wg16_sources_are_formatted`,
  `block_sources_are_formatted`: 64 draws each). The layout changes no token: the H100 runs' dumped programs
  (`wg16_00`, `01`, `05`, `17`, `33`) and their formatted text emit the same host and device IR byte for byte at
  `--target=h100`, so the 72 / 72 above stand. `deep_stages_equal_the_warp_tile` builds its variant of `PJ` by
  checked replacements (each present once), so a layout change can no longer leave it the original.

## Kernel suite on the box (`box_run.sh`, ABBA × 5, value-gated; `gpu_changed`: `suite_diff.py --arch h100 --off
wgmma16` → 10 programs differ, 127 text-equal: null by construction)

Program boundary ms (median); cuBLAS bf16 = `gemm_lib.cu --bf16` (new: `cublasGemmEx` / cuBLASLt over bf16 A, B, f32 C,
COMPUTE_32F), best leg; % of the measured 998 Tflop/s.

| shape (m × n × k) | off: mma.sync | on: wgmma | off / on | cuBLAS bf16 (leg) | on / cuBLAS | on Tflop/s (% peak) | cuBLAS Tflop/s (% peak) | kernel only: on / cuBLAS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 × 4096 × 4096 | 0.0651 | 0.0440 | 1.48 | 0.0291 (lt-fresh) | 1.51 | 98 (10 %) | 148 (15 %) | 0.0275 / 0.0200 |
| 128 × 6144 × 4096 | 0.0760 | 0.0571 | 1.33 | 0.0358 (lt-fresh) | 1.59 | 113 (11 %) | 180 (18 %) | 0.0400 / 0.0273 |
| 128 × 28672 × 4096 | 0.1538 | 0.1357 | 1.13 | 0.0984 (lt-fresh) | 1.38 | 222 (22 %) | 306 (31 %) | 0.1075 / 0.0898 |
| 128 × 4096 × 14336 | 0.1793 | 0.0951 | 1.89 | 0.0604 (lt-fresh) | 1.57 | 158 (16 %) | 249 (25 %) | 0.0730 / 0.0516 |
| 2048 × 4096 × 4096 | 0.2482 | 0.1358 | 1.83 | 0.1021 (lt-fresh) | 1.33 | 506 (51 %) | 673 (67 %) | 0.1145 / 0.0948 |
| 2048 × 6144 × 4096 | 0.3594 | 0.1897 | 1.89 | 0.1481 (lt) | 1.28 | 543 (54 %) | 696 (70 %) | 0.1750 / 0.1399 |
| 2048 × 28672 × 4096 | 1.5367 | 0.6989 | 2.20 | 0.6671 (lt-fresh) | 1.05 | 688 (69 %) | 721 (72 %) | 0.6760 / 0.6420 |
| 2048 × 4096 × 14336 | 0.7900 | 0.3536 | 2.23 | 0.3127 (lt) | 1.13 | 680 (68 %) | 769 (77 %) | 0.3415 / 0.3013 |
| llm_pre_t2048_bf16 | 1.1629 | 0.8609 | 1.35 | | | | | |
| llm_blkpre_t2048_bf16 | 3.7814 | 2.0861 | 1.81 | | | | | |

## Evidence pass — intent → realized → explanation

Intent (`intent.py` `wgmma16`, plus `wgmma-ws` / `narrow-tile` / `min-blocks` now stating the 16-bit family): 16-bit
`wgmma` in PTX, HGMMA in SASS, no HMMA; `bk/16` HGMMA a hot-loop trip, no block barrier in it; the stage as tensor
boxes (UTMALDG on every issue path, transaction barriers, no LDGSTS); `a`'s home realized (descriptor: no LDS in the
hot loop; registers: LDS + F2FP); no spill. Realized (`evidence.py check` over the bundle with ptxas 13.5 sm_90a):
every `wgmma16` check ok on every changed program; remaining mismatches are pre-existing (the device-exact oracle
builds, one OFF arm).

| kernel (2048 × 28672 × 4096) | SASS a k-stage (64 deep) | copies | epilogue | grid / block / regs / smem |
|---|---|---|---|---|
| Mapal off (`mma.sync`) | 128 HMMA.16816 a warp tile, 32 LDSM | 48 LDGSTS.128 | 64 STG.E.64 | 224 × 16 / 128 / 207 / 107 KB |
| Mapal on (`wgmma16`) | 4 HGMMA.64x256x16 a consumer warpgroup | 2 UTMALDG (2-d `a`, 3-d `b`), on 2 issue paths | 64 STG.E.64 from registers | 112 × 16 / 384 / 168 (40 / 232 under `setmaxnreg`) / 197 KB |
| cuBLAS `nvjet_sm90_tss_256x128_64x4_1x2_h_bz_coopA` | 8 HGMMA.64x128x16 a consumer warpgroup | UTMALDG.3D **.MULTICAST** (cluster 1 × 2) | STS → **UTMASTG.3D** | **2 × 66 persistent** / 384 / 168 / 213 KB |

Explanation: off → on is the instruction's rate and staging — the HMMA tile re-reads its panels through LDSM every k16
and copies by LDGSTS; the warpgroup reads both operands through descriptors (the 2.0–2.2× at m = 2048 against the
1.52× rate ratio: the warp tile also loses its issue slots to LDSM / LDGSTS). On against cuBLAS (1.05–1.33× at
m = 2048): cuBLAS's kernel is persistent over 132 CTAs, multicasts one operand across a 2-CTA cluster (half its L2 →
SMEM traffic for that operand) and stores its tile through shared memory by TMA under the next tile's mainloop;
Mapal's CTA-a-tile kernel stores from registers after its last k-step. At m = 128 (1.38–1.59×) cuBLAS takes 64 × 64
tiles 64 deep in 13 stages with a split-K (2) and its reduce; Mapal's 64 × 64 tile is 256 deep in 3 stages (the
copies' price at `tma_copy_ns` 159 ns), no split: the first stage's latency is a quarter of the k loop, unhidden. The
program-boundary ratio adds the host dispatch on this box's Broadwell host (kernel-only 1.07–1.38×).

## The full model on the box (bf16 class, ABBA × 3 × 3 processes, gpt-fast warmup 2 reps 5)

| prompt | TTFT off → on ms | gpt-fast | on / gpt-fast (before) | decode on / gpt-fast ms |
|---:|---:|---:|---:|---:|
| 128 | 3.92 → **2.70** | 3.58 | **0.75** (1.09) | 1.975 / 1.327 |
| 512 | 7.44 → **5.60** | 4.47 | **1.25** (1.66) | 2.001 / 1.335 |
| 2048 | 24.81 → **18.11** | 11.45 | **1.58** (2.17) | 2.057 / 1.423 |

Where the 2048 prefill goes (CUPTI, the TTFT window, busy 18.03 of 18.41 ms on; off 24.62 of 25.02):

| per prefill | Mapal off | Mapal on | gpt-fast |
|---|---:|---:|---:|
| gate + up projections (2 × 2048 × 8192 × 2048 a layer) | 7.49 ms | **3.81** | nvjet 256 × 128 (with qkv, o, down: ≈ 5.6 ms all projections) |
| down (2048 × 2048 × 8192) | 3.90 | **2.16** | |
| q, o, k / v projections | 3.30 | **2.00** | |
| attention (the streamed unit: 16 × `mapal_tile_…` 1024 × 128 threads, `stream=b64xkb32:mma:cp`) | 7.0 | 7.0 | flex attention 16 × 93 µs ≈ 1.5 ms |

The projections are now 8.0 ms against cuBLAS's ≈ 5.6 ms in gpt-fast; the streamed attention (mma.sync, 32-key
windows, 22 MUFU.EX2, 32 HMMA.16816, LDGSTS — 0.44 ms a layer, 39 Tflop/s) is 4.7× Triton's flex attention
(HGMMA.64x128x16 for QKᵀ, HGMMA.64x64x16 for PV, 93 µs a layer) and is now the largest single gap: the next item.

## `ws-persist` on the 16-bit tile (tried, not shipped)

`--gpu-on=ws-persist` / shipped, ABBA × 7 iter ms on the same box: 0.954–1.040 (wins m128 × 4096 × 14336 4.6 %,
m128 × 6144 × 4096 2 %; loses 2048 × 28672 × 4096 4.0 %, 128 × 4096 × 4096 3.7 %); the model's TTFT 2048 / 512
18.05 / 18.15, 5.48 / 5.59 ms (−0.6 / −2 %). Mixed, as on TF32 (`rung_replay.tsv`); the walk's static round robin and
its register-stored epilogue are what cuBLAS's persistent kernel does differently (TMA store, cluster multicast).

## Open items

1. **The streamed attention unit on `wgmma`** (FA3 form): QKᵀ with `b = K` k-major from a descriptor, PV with `a =
   P` from registers (this branch's register home) and `b = V` MN-major (this branch's transpose bit), 64- or
   128-key windows. The 2048 prefill trace above is the motivation (7.0 of 18.4 ms).
2. **The projection epilogue and cluster** against cuBLAS: stage the tile through shared memory and store by TMA
   under the next tile's mainloop (a persistent walk whose epilogue overlaps), and a 2-CTA cluster multicasting the
   shared operand (`Caps::bulk` multicast geometry). The 1.05–1.33× at m = 2048.
3. **Small-m depth**: the copies' price picks 3 stages 256 deep at m = 128 where cuBLAS runs 13 stages 64 deep with
   split-K; the price reads `tma_copy_ns` 159 ns. Open: what separates a deep stage from many shallow ones (the fill
   latency against the copy count), and a split 16-bit warpgroup tile.
4. **bf16 weights × f32 activations through the split forms** (the f32 class): a split's per-k-step zeroed
   accumulator and round-to-nearest add do not map onto `wgmma`'s in-place asynchronous accumulate without one FADD
   per accumulator a k16 step; not attempted. **An f32 `b`** (the descriptor operand) refuses the warpgroup tile
   (the transposed form would take it in registers). **Radix / batched GEMMs** keep `rx.is_none()`: their tensor boxes
   need the batch digits as map dimensions.
5. **The merged launch**: why a warp-specialized member still faulted inside a merged kernel with its maps' address
   fixed (it launches alone now).

## For integration

- CLAUDE.md facts table, the backend `Gpu` row: add `wgmma_half_per_clk` beside `mma_half_per_clk` /
  `wgmma_tf32_per_clk` (`calib.cu matrix` `wgmma` kinds), `Caps::matrix_group` and `MatrixOp::mn_major` (a descriptor
  operand's transpose bit) to the capability list, and `rules::wgmma16_pays` beside `rules::rungs`.
- `rung_replay.tsv`: two `wgmma16` lines (h100, `a=smem` / `a=regs`) and one `ws-persist` h100 `mixed` line; the
  replay counts (picked / null / mixed / open) are now 174 / 82 / 18 / 22 — the evidence file that reports them needs
  the same numbers.
- `benches/gpu/evidence.py` `KERNEL_RUNGS` gains `wgmma16`; `benches/local/suite_diff.py` gains `--arch`;
  `gemm_lib.cu` gains `--bf16` (box_run.sh passes it for `_bf16` projections).
- README: the H100 bf16-class row (TTFT 2.70 / 5.60 / 18.11 ms against gpt-fast 3.58 / 4.47 / 11.45 on the same box)
  with a link here.
- No tailored fix.
