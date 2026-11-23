# Hopper `wgmma` + TMA: warp-specialized TF32 tiles fed by whole-tile tensor boxes (topic `hopper-wgmma-tma`)

Design: `.claude/sprints/hopper/design.md` (K2, R1–R3, F1). Branch `sprint/hopper` from `55ba3e42`. Raw output (box logs,
results, calibrations, probes): `.claude/raw-evidence/local-rtx-5080/hopper/` (never in git).

**Spend: $1.79 of the $10 budget** — three H100 SXM sessions on one vast.ai offer ($2.334/h, label
`mapal-hopper-h100`, each instance destroyed on exit and checked gone): s1 13.4 min ($0.52), s2 21.8 min ($0.85),
s3 10.7 min ($0.42).

## What changed

| kind | name | what it is |
|---|---|---|
| capability | `Caps::reg_realloc` (`RegRealloc {24, 256, 8}`) | `setmaxnreg` as geometry; `caps_of_cc`: the arch-specific targets from 9.0 (`sm_90a`; `sm_120a`, not the 5080's `sm_120`) |
| capability | `BulkCopy::{desc_bytes, oob_zero}`, `tensor_box_fits` | the 128-byte by-value map; N-d boxes: ≤ 5 dims, sides ≤ 256 dividing the extents, inner row = the swizzle span, outer strides and base whole granules, ≤ one barrier phase |
| capability | `MatrixOp::flops_per_clk` | the warpgroup products' dense per-SM rate (datasheet: TF32 2048, 16-bit 4096, 8-bit 8192 flop/clk/SM); `Gpu::stage_consume_ns` reads it (the FFMA rate stands in where 0). *Integration: merged into f32-tensor-cores P1 — `stage_consume_ns` reads `Gpu::matrix_flops_per_clk` (`wgmma_tf32_per_clk` 2048 on the H100, the same price); the per-op field is gone.* |
| property | `DeviceCost::tma_copy_ns` (`Gpu::tma_copy_ns`) | the bulk-copy engine's ns per copy, `benches/local/tma_calib.cu`: **RTX 5080 197**, **H100 SXM 159** (measured); unmeasured: `lat_ns / TMA_COPIES_IN_FLIGHT` |
| fact | `TileRead::tensor_form` → `TensorForm {base, dims}`, `TensorForm::box_of` (mapal-ir F1) | a tile operand as a dense tensor over (I, Lane, K): dims innermost first, no two axes overlapping; the box that is a panel and whether the boxes tile it exactly. `None`: `ksplit` (window / im2col class) or overlapping axes |
| rule | R2 `gpu_wgmma_ws_tile` (`wgmma-ws`) | one producer + 1–2 consumer warpgroups, one block per SM by construction (smem from `smem_per_block`, so the 2048² MLP site the S53 derivation refused takes `wgmma`); consumer registers under R3 `(regs_per_sm − 128·40)/(128·g)` → 232; fill by flops per staged byte among the geometries that cover the SMs |
| rule | R1 price (`tma-digits`) | a stage's 2 copies (`2·tma_copy_ns`) within its consume time at the matrix rate: the shallowest depth (32·d words, d ≤ 4) that pays, else the deepest — 2048×4096: bk 32; 2048² (n 128): 64; m128 (n 32): 128 |
| kernel | K2 `wgmma_ws` | producer: rounds tile `j − (stages−1)` in place (RNE; `wgmma` truncates) behind `ready[s]`, then thread 0 waits `empty[s]`, `expect_tx`, issues the `aᵀ` box (k digits: one 2-/3-d copy) and the `b` panel (lane digits: one 3-d copy of all `bn/32` slabs); consumers wait `full` + `ready`, wait the previous group (`wait_group 0`, the PTX rule on `a` fragment registers) and release its stage, load `bᵀ` fragments through the box swizzle, `bk/8` `wgmma`, commit |
| runtime | `_tmap` recipe → `cuTensorMapEncodeTiled` | the kernel's maps travel as the device global `<kernel>_tmap`; the launch encodes each once per pointer and passes it by value after the pointers (host ABI unchanged; `tmaps=` on the trace total) |
| emulator | boxes, arrive, `wgmma` | boxes with the engine's swizzle and OOB zero (alignment checked), plain `mbarrier.arrive`, `wgmma` as the warpgroup's meet reading `b` through the descriptor with operand truncation, and a check that no group's operand changes before its `wait_group` |

OFF arms (measurement flags): `--gpu-off=wgmma-ws` (the S53 kernel; `mma.sync` where S53 refuses), `--gpu-off=ws-regs`
(no `setmaxnreg`, the even share), `--gpu-off=tma-digits` (a 2-d box per 32-word `b` slab, bk 32: the s1 form).

## Properties measured (`tma_calib.cu`: one elected thread an SM, every SM, L2-resident, ns per copy)

| copy | RTX 5080 depth 1 → ≥2 | H100 SXM depth 1 → ≥4 |
|---|---:|---:|
| row 128 B (`cp.async.bulk`) | 234 → 198 | 187 → 158 |
| box 1 KiB / 4 KiB | 235 / 271 → 197 | 194 / 209 → 159 |
| box 16 KiB | 408 → 289 (4.8 TB/s) | 302 → 159 (13.5 TB/s) |
| box 32 KiB | 701 → 654 | 465 → 352 (12.3 TB/s) |

The engine costs a fixed ~160–200 ns a copy up to 16 KiB however deep the ring: the property that separates the
rejected row staging (96–144 copies a stage) from boxes, and a 5-copy stage from a 2-copy one (below).

## Hardware probes (RTX 5080, real device; `probe/` in the raw dir)

- The runtime path end to end (recipe global → `cuModuleGetGlobal` → encode → 128-byte by-value parameter): a
  9 × 40 f32 tensor, box 8 × 32 at (16, 3), 128-byte swizzle — **0 of 256 words differ from the emulator's box model**
  (dense box, XOR on the shared address bits, OOB zero; unswizzled it would differ in 152); one encode for two launches.
- A 3-d map whose outer stride (128 B) is below the middle one: one box `{32, bk, q}` lands the `q` column slabs
  exactly as `q` 2-d boxes — **0 of 4096 words differ** (`tma-digits`).

## Static evidence (`ptxas -arch=sm_90a`, SASS; intents `wgmma-ws`, `ws-regs` in `intent.py`)

`pt_pj_m2048_n4096_k4096_tf32` (n 256, 2 consumers, bk 32, 4 stages): 168 registers at launch, 0 spill,
`USETMAXREG.DEALLOC 0x28` / `TRY_ALLOC 0xe8`; a trip: `UTMALDG.2D` + `UTMALDG.3D`, 4 × `HGMMA.64x256x8.F32.TF32`
(`gdesc` B), `WARPGROUP.ARRIVE` / `DEPBAR`, `SYNCS.*TRANS64`; no `LDGSTS`, no `BAR` in the loop. The s2 bundle's
evidence: `wgmma-ws:on` 120 checks ok, `:off` 15 ok, `ws-regs:on` 40 ok (`sass_stats` now counts `USETMAXREG`).

## Timed: H100 SXM (rented, same box both sessions; ABBA, 7 reps, median `iter ms`; value gates first: every arm of every program within D3 of the device-exact oracle)

Session 2 (shipped code). `slabs` = `--gpu-off=tma-digits` (the s1 kernel: 5 copies a stage at n 256, bk 32);
OFF = `--gpu-off=wgmma-ws`; null = the shipped build's copy; best = gemm_lib's best cuBLAS / cuBLASLt TF32 leg on the
same box (`-fresh`: operands regenerated, Mapal's boundary).

| workload | OFF | slabs | ON | OFF/ON (null) | fastest competitor (leg) | ON/best | explanation |
|---|---:|---:|---:|---:|---|---:|---|
| mlp2_2048_tf32 | 0.5725 | 0.5576 | **0.4594** | **1.25** (0.5 %) | torch-compile-cg-tf32 0.2145 (matrix) | 2.14 | OFF is `mma.sync` (S53 refused); ON n 128 × 2 consumers, bk 64: 2 copies per 2.1 Mflop (318 ns of 566) against 5 per 1.05 (795 of 283): the engine's per-copy cost was the stage rate |
| mlp_2048_tf32 | 0.3041 | 0.3020 | **0.2551** | **1.19** (1.8 %) | torch-compile-cg-tf32 0.1107 (matrix) | 2.30 | as above |
| pj_m128_n6144_k4096_tf32 | 0.1753 | 0.2148 | **0.1450** | **1.21** (2.1 %) | lt-tf32-fresh 0.0523 | 2.77 | n 32, 1 consumer, bk 128: 2 copies per 0.52 Mflop (slabs: 3 per 0.13) |
| pj_m128_n28672_k4096_tf32 | 0.2794 | 0.2845 | **0.2486** | **1.12** (0.9 %) | lt-tf32-fresh 0.1855 | 1.34 | as above; DRAM-side |
| pj_m128_n4096_k14336_tf32 | 0.2778 | 0.5386 | 0.2641 | 1.05 (4.1 %) | lt-tf32 0.1059 | 2.49 | within ~null |
| pj_m128_n4096_k4096_tf32 | 0.1219 | 0.1955 | 0.1265 | 0.96 (1.6 %) | cublas-tf32-fresh 0.0442 | 2.86 | 256 tiles on 132 SMs (1.94 waves) at 32 rows a tile; no split (R6 open) |
| pj_m2048_n28672_k4096_tf32 | 2.0566 | 1.9921 | **1.9667** | **1.05** (0.2 %) | lt-tf32 1.3192 | 1.49 | n 256: the price already met at bk 32 (2 copies 318 of 567 ns) |
| pj_m2048_n4096_k14336_tf32 | 1.0292 | 1.0291 | 1.0092 | 1.02 (0.0 %) | cublas-tf32 0.6186 | 1.63 | |
| pj_m2048_n6144_k4096_tf32 | 0.4855 | 0.4825 | 0.4790 | 1.01 (0.1 %) | lt-tf32-fresh 0.2652 | 1.81 | |
| pj_m2048_n4096_k4096_tf32 | 0.3461 | 0.3542 | 0.3428 | 1.01 (0.9 %) | cublas-tf32-fresh 0.1800 | 1.90 | |
| llm_blkpre_t2048_tf32 (s3) | 6.7255 | 6.7418 | 6.6694 | 1.01 (0.1 %) | triton-tf32 3.47 (matrix) | 1.92 | only its projections move; attention stays on radix tiles (P4b open) |
| llm_pre_t2048_tf32 (s3) | 3.2463 | 3.2676 | 3.2642 | 0.99 (0.1 %) | triton-tf32 1.372 (matrix) | 2.38 | as above |

Session 3 (5 reps, `LADDER=1`: the LLM programs, carried in s1/s2, timed) replicates s2 on every other row within
its null (mlp2 1.24, mlp 1.17, m128 n6144 1.21, n28672 1.13, m2048 1.01–1.07) except **pj_m128_n4096_k4096: 0.87**
(s2 0.96) — a consistent loss, the mixed result below (open 5).

Session 1 (the slabs form shipped then; same box type): m128 sites 0.53–0.99 of OFF, m2048 0.98–1.04, MLP 0.98–1.02;
`--gpu-off=ws-regs` 0.560 vs 0.353 ms (pj 2048×4096: n 128 at the even 168-register share) and 0.348 vs 0.307 (mlp).
That loss on m128 and the flat MLP are what the per-copy property explains and the priced depth removes (s2).

## The library's Hopper kernels (CUPTI on the box, `ktrace.cpp`; gemm_lib's TF32 legs, s3)

| site | cuBLAS kernel (most launched) | grid | block, regs, smem |
|---|---|---|---|
| 2048 × 4096 × 4096 | `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize128x256x32_warpgroupsize2x1x1_execute_segment_k_off` | 66 × 2 = **132** (persistent: one CTA an SM) | 384, 168, 226 KiB |
| 128 × 4096 × 4096 | `…tilesize64x128x32_warpgroupsize1x1x1_execute_segment_k_off` | 16 × 4 = 64 | 384, 168, 226 KiB |

The library's 2048 tile is ours (128 × 256 × 32, two consumer warpgroups and a producer, 384 threads at 168
registers) — the difference is the grid: a persistent CTA an SM walking the tiles (its prologue and pipeline fill
once, the next tile's boxes under this tile's epilogue) against our 256 CTAs in 1.94 waves. Its m128 tile is 64 rows
of a 64-CTA grid (b read twice), ours 32 rows (b read 4 times, 2 waves).

## Tests

- `profile.rs`: `caps_are_geometry_per_part`, `wgmma_ws_tile_is_derived_from_hopper_properties` (hand-worked 2048×4096²,
  2048³, 128×4096², `ws-regs` off, no digits, refusals), `tma_copy_ns_is_measured_or_derived`; mapal-ir
  `tensor_form_tests` (2).
- `tests/hopper_tma.rs`: the kernel text and `ptxas -arch=sm_90a` + SASS (0 spill, registers × block ≤ the file,
  UTMALDG / HGMMA / USETMAXREG / SYNCS, no LDGSTS / STL); 36 generated TF32 GEMMs (`testgen::WgmmaProg`: m 8–320,
  n 64–384, k 8–640, padded / misaligned row strides, offset bases, `b` k × n or n × k) at 1 / 2 / 8 SMs in four
  emulator modes (serial 0 / 1 / 2, reverse blocks) — every warp-specialized kernel equals the S53 kernel byte for byte
  where S53 takes the site, the 2-d slab boxes byte for byte, and D3's bound of the interpreter — coverage: 36
  programs, 18 warp-specialized kernels, 8 also as 2-d slabs, 9 byte-identical to S53, 9 bound-only, 8 with two
  consumers, 3 wrapping the barrier phases, 2 S53 fallbacks, 16 other kernels (refusals: n × k `b`, misaligned rows,
  k 8); `deep_stages_equal_one_row_stages`: the priced bk 64 (3-d k-digit `aᵀ` box, the descriptor over two atom
  columns) equals the bk-32 stages byte for byte in every mode; mutations fail — no producer rounding (other bits),
  a stage released before its group's wait (exit 110: the operand changed under the group). These generated runs keep
  `place` off, which drops the H100's uncalibrated rates and with them the price (bk 32): the deep stages are covered
  by the dedicated test and by the device runs (MLP bk 64, m128 bk 128, value-gated).
- `gpu_wgmma.rs` pins the S53 kernel under `--gpu-off=wgmma-ws`. Backend unit tests 118 passed.

## Open problems attacked

1. **TMA lost 4.3–14.5× (row copies)** — closed for the warpgroup tile: whole-tile boxes, the engine's per-copy cost
   measured as a property, and the R1 price choosing the stage depth (2 copies a stage). Not built: K1 (boxes for the
   `mma.sync` / FMA / 16-bit tiles on the 5080), so the row form keeps its rule-off and the 5080 runs no TMA kernel.
2. **The 2048² TF32 MLP refused `wgmma`** — closed (one block per SM by construction): 1.19–1.25× its `mma.sync` OFF.
3. **The `wait_group 1` stage hazard (S53)** — closed by construction (a stage released only after the wait that
   completes its group; the emulator fails an early release). `wait_group 0` before the fragment loads is the PTX rule
   on `a` fragment registers (ptxas inserted no wait for the pipelined form).
4. **m2048 TF32 sites at 1.5–1.9× cuBLAS** — attempted, open: the copy count is no longer the limit there (n 256 meets
   the price at bk 32) and ON ≈ OFF. Candidates the evidence leaves: the in-stage rounding pass (64 KiB of shared
   traffic a stage beside 64 KiB of `wgmma` B reads and 48 KiB of box writes: over 128 B/clk if `wgmma` shares the
   port), R4 rounding at the producer (`round_at` has no consumer yet), multicast (R7) halving the shared operand's L2
   traffic. Counters are refused in vast containers, so the decision needs an arm per hypothesis.
5. **m128 sites: a mixed result (n6144 / n28672 1.13–1.21×, n4096 × k4096 0.87–0.96×)** — attempted, open. What
   separates them in the plans: at n4096 the WS grid is 4 row blocks × 64 = 256 CTAs at one an SM (2 waves), so
   the 64 MB `b` is read by row blocks 0–1 in wave 1 and again by 2–3 in wave 2 (DRAM: 64 MB over 50 MB of L2),
   while S53's 2–3 resident blocks an SM hold all 256 in one wave; at n6144 / n28672 every arm streams `b` over
   several waves anyway. The fact is the operand's re-read across waves against the L2 (`raster`'s, which answers 1
   here) — the persistent grid in raster order (the library's form above) or R6's split would read `b` once; not
   built.
6. **`acc_u` of `wgmma` on the H100** — not measured (no `mma_calib wgmma`); every format keeps the assumed −23 and
   the TF32 value gates pass on the device.

## For integration

README line (`local-rtx-5080/README.md`):

`| hopper-wgmma-tma | [hopper.md](hopper.md) | warp-specialized TF32 wgmma fed by 2 tensor boxes a stage (R1 price on the measured tma_copy_ns: 5080 197 / H100 159 ns), producer-rounded aT, setmaxnreg; H100 SXM: MLP tf32 1.17-1.25x, m128 0.87-1.21x, m2048 1.01-1.07x of OFF; $1.79 |`

CLAUDE.md facts-catalog row:

`| tensor_form (TileRead::tensor_form → TensorForm, box_of) | a tile operand as a dense ≤ 3-axis tensor over (I, Lane, K): dims innermost first, none overlapping; the box that is a panel and whether such boxes tile it exactly (refused: ksplit windows / im2col, overlapping axes) |`

CLAUDE.md "backend: Gpu properties" row additions: `tma_copy_ns` (the bulk-copy engine's ns per copy, `tma_calib.cu`),
`stage_consume_ns` (a stage's flops at the matrix rate, `MatrixOp::flops_per_clk`), `Caps::reg_realloc`.
