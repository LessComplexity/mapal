# gaps-disasm — the RTX 5080's loss rows, disassembled against the leaders (2026-11-25)

Analysed commit `f1d3a04f` (the integrated confirmation-fixes head); matrix 5080 column 0.965 over 120 rows, 51 of
them losses (`ratio > 1`, Mapal / best same-precision leg). Four family diagnoses (gemm-f32, tensor-tf32,
tensor-16bit, attention) merged here into one cause catalog, ranked, and turned into a fix-sprint plan.
Read-only analysis: no compiler code changed.

**Method.** nsys / ncu do not run on this box (no stall counters). Time splits come from CUPTI activity records
(`ktrace.so`: kernels, memsets, API calls) on both sides; Mapal SASS from the bundle PTX through ptxas 13 +
`cuobjdump` / `nvdisasm`; the leaders' SASS from the module images they actually loaded (modcap + `nvdisasm`;
Triton cubins from its cache); per-CTA timelines from PTX instrumented with `globaltimer` / `smid` / `clock64`;
A/B arms built with the existing emit flags (`--splitk`, `--geom`, `--gpu-on/off`) or PTX edits through
`MAPAL_PTX`, same-session ABBA, every arm value-gated. Raw captures (never in git):
`.claude/raw-evidence/local-rtx-5080/gaps-disasm/{gemm-f32,tensor-tf32,tensor-16bit,attention,plan}/`.

**Gain arithmetic.** Each loss row's after-ratio is its family's estimate; its log gain is split over the
causes that explain it (shares in `plan/rank.py`). A cause's gain is `exp(Σ share·ln(now/after) / 120)` over
the whole column. Rank = gain % × families touched / risk (1 low, 2 medium, 3 high). All causes together:
0.965 → ~0.914 (optimistic: estimates overlap); sprint 1 alone: 0.965 → ~0.925. The 51 loss rows: geomean
1.262 → ~1.108.

## Ranked cause catalog

Families: f32 = gemm/mlp/fan f32, tf32, bf16 = 16-bit gemm + llm, attn = attention/softmax/long rows.
Verdict: **confirmed** = the time split, the SASS/IR and (where one exists) an A/B arm all agree, spot-checked
here; **plausible** = mechanism seen in SASS, size not shown by an arm; **unconfirmed** = no arm, inferred.

| # | cause | rows | families | evidence (spot-check) | generic fix | gain (120 rows) | branch |
|---|---|---:|---|---|---|---:|---|
| 1 | **C1 host path in the clock**: per-call site dispatch (t0 → first driver call 1.6–2.7 µs), an atomic site's zero as a `cuMemsetD8Async` API call (3.7–5.2 µs host), host enqueue of dependent launches after the device drains (2.0–2.45 µs each) | 10 | f32, mlp, fan, bf16, attn | **confirmed**: `pj_m1_n4096` API trace: memset 3.66 µs + launch 2.72 µs + sync around a 17.4 µs kernel at parity with gemvx (17.7); `lsm_32_8192` Mapal kernel gaps 3.33 + 0.51 µs after 3 dependent launches | site pick + `CUfunction` resolved at module load; an atomic output's zero folded into the preceding kernel / arena init (`zero_fold`) or a zero kernel, never the memset API; property `DeviceCost::launch_host_ns` priced in `dep_launch_at` so the merge rule takes fewer launches where kernels are shorter than the host enqueue | 0.96 % | B1 |
| 2 | **C5 split / stream-K form**: stream-K accumulates every tile with `REDG` into a zeroed output; splits limited to divisors of the k-steps; a one-wave atomic grid over-split; the 16-bit/TF32 k-cut pays `S·cells` atomics where the leaders use a serial in-place fixup | 13 | f32, mlp, tf32, bf16, attn | **confirmed**: `softmax_rows_4096` trace: 64 MiB memset 41.2 µs + stream-K tile (168 CTAs) with 384 static `REDG.E.ADD.F32`; `mm_16x2048x7168` emitted arms S=14 70.3 µs (= cuBLAS 0.0799) vs shipped S=56 78.7 | stream-K as data-parallel + stream-K (owned tiles `STG`, remainder tiles fix up through workspace + ticket, `stream.rs` split_combine), an ordered serial fixup for k-cuts, uneven last k-slice (`trip_split`), the combine priced as a tail of a one-wave grid | 0.48 % | B3 |
| 3 | **C3 in-flight depth below Little's law**: 2-stage cp.async rings waiting `DEPBAR.LE 0x0` (one panel in flight); TF32 warp tiles shrunk to 4–8 HMMA per k-step with single-buffered fragments; FMA tiles' register-staged next-panel loads sunk by ptxas to 180–390 ns of cover vs `lat_ns` 573 | 10 | f32, tf32, bf16 | **confirmed** (bf16): `pj_m128_n28672_k4096_bf16` Mapal `DEPBAR.LE SB0,0x0` vs Triton `0x4`, same 128×128×32 geometry; ABBA `--gpu-on=mma-depth` 0.398 → 0.342 ms. TF32 m128: Mapal 0x0 + 96 FSETP vs cuBLAS 0x4 + F2FP; **plausible** only for the TF32 size: the best existing arm (bk16×5 + sk8, 0.129) closes 13 % of the gap to 0.084 | stages from Little's law (bytes in flight per SM ≥ `dram_bytes_per_us_per_sm × lat`) where `residence_plan` puts the operand in DRAM and the busiest SM's DRAM time ≥ ~0.85 of its pipe time; swizzled unpadded panels, `wait_group(stages−2)`; warp tile floored at the matrix pipe's ILP (new `MatrixOp` latency/interval from `mma_calib.cu`), fill from k-cuts; FMA tiles: realized cover property → cp.async stages incl. under stream-K | 0.88 % (mma 0.84, fma 0.04) | B2 (mma), S2 fma-tile |
| 4 | **C8 closed-form values loaded or materialized**: the decode norm folds over a loop-captured iota `rD` and loads it (dependent index-then-data `LDG`), as one CTA, 17–19 µs per step vs 1–1.5; a closed-form bias `b1[t%1024]` materialized by its own kernel | 12 | bf16, bf16kv, bf16w, mlp | **confirmed**: source `pt_llm_dec_b32_l2048_bf16kv.mapal` `b -> (0.0, rD) -> fold { … x[b*4096+i] … }` over `rD = 4096 -> iota` declared outside the loop; norm SASS (attention): `LDG rD[i]` → `IMAD.WIDE` → `LDG x` | the cell-fold emitter reads an iota domain by its `elem_plan` law through loop captures (no load); a row fold whose rows are fewer than the SMs spreads over CTAs (`fold_combine` partials / `row_tree`); a closed-form operand is evaluated in its consumer's epilogue | 0.28 % | B4 |
| 5 | **C4 grid quantization on the shared matrix pipe**: 16-bit / TF32 tile grids sized to reach `fill_warps`, not per-SM rounds (512 tiles / 84 SMs = 6.10 → 7 rounds; 768 → 10 vs 9.14) | 10 | tf32, bf16 | **confirmed** (mechanism): `pj_m2048_n4096_k4096_bf16` grid 32×16 = 512 CTAs × 128 thr, 218 regs, 37.9 KB; predicted 7/6.25 = 1.12 vs kernel 1.129, per-round cost equal to cuBLAS (92.8–95.2 vs 93.7–94.4 µs); existing knobs lose (`resident=4` +6–8.5 %, `split=2` +4.7–11.7 %); the fix itself has no arm yet | tile and k-walk chosen by `⌈pieces / SMs⌉ × piece_ns / efficiency(warp tile)`: 128×64 / 64×64 blocks at the same warp tile, `n = 8j` widths, a k-balanced split with C5's ordered combine | 0.44 % | B2 |
| 6 | **C12 narrow / small-m DRAM operand path**: the decode scores kernel stages the bf16 k-major K per element (64 `LDG.E.64` + 64 `PRMT` + 128 `SHF` per trip) into smem at an 8-way bank conflict; 1-warp CTAs (5 warps/SM) with no copy/compute overlap; 663–679 GB/s vs the same-geometry P.V at 796–817 | 8 | bf16, bf16kv, bf16w | **confirmed** (both mechanisms present): `k_3bede` 64 `LDG.E.64.CONSTANT`, 132 `STS`, 64 `LDS.128`; IR store index `%v45 = ((tid&7)·4)<<7 \| tid>>3` words, +128 words per store → 4 banks per warp = 8-way. **unconfirmed** split between the conflict and the width (no arm separates them) | `kvec_b` copy layout from the bank geometry (pitch +4 words or XOR swizzle on n, the LDS map following); register-staged / radix copies of a narrow operand sized in bytes (16-B lane loads = 8 bf16) from `tile_plan_narrow` `elem_bytes`; small-m DRAM sites stream b register-direct in 16-B evict-first loads or narrow cp.async stages, 4–8 warps | 0.36 % | B5 |
| 7 | **C11 raster gate reads one block's panels, not the b-panel reuse distance** | 3 | tf32, bf16 | **confirmed**: `Gpu::raster` keeps row-major while `(bm+bn)·k·elem ≤ l2_bytes`; `pj_m2048_n28672_k4096_bf16` ABBA `--gpu-off=l2-gate` 4.571/4.581 → 3.991/3.987 ms (1.153 → 1.007); TF32 twin −5 %, blkpre −2 %; null where `gx ≤ wave` | gate on `gx > wave` and `gx·bn·k·elem > l2_keep_bytes`; change the raster only (today's OFF arm also moves the k14336 split) | 0.17 % | B2 |
| 8 | **C7 row-unit realization**: one row per 256-thread CTA, barrier-per-level smem trees (`tree_total`), two global passes of scalar loads, IEEE quotient per element by a row-uniform divisor, guarded exp | 7 | attn, bf16 | **confirmed** (SASS): Mapal softmax `k_8845` 19 `BAR`, 0 `SHFL`, 16 scalar `LDG`, a `CALL` (fdiv slow path) vs Triton 5 `BAR`, 26–28 `SHFL`, `LDG.E.128`, 2 `MUFU.RCP` per CTA; no arm | shuffle levels inside a warp + one smem exchange (the `Regroup` / tree-step fact admits the order); row buffer colored into registers when `row/threads` fits `regs_per_resident_thread`, read once as 16-B vectors; rows per CTA from `wave_full`; `precision_plan` admits `x · (1/d)` for a divisor constant along the axis (`fold_combine` Factor) with one rounding in the bound; exp form by argument range | 0.43 % | B4 |
| 9 | **C9 same-cell epilogue loses the producer's split or stays its own kernel**: silu·mul merged into the up GEMV drops its k-split → 56 CTAs on 84 SMs at 691 GB/s (siblings 851–892); silu·mul as its own kernel in prefill; two products into one cell as two tiles | 4 | fan, bf16, bf16kv, bf16w | **plausible**: attention and 16-bit traces agree on the 56-CTA kernel (+40 µs/step); no arm | the epilogue applies after the combine (`fold_combine`), so the merged site keeps the split; a same-cell consumer joins its tile's epilogue (`unit_plan` / `tile_plan_post`); a sum of products over one cell map as one tile site over the concatenated fold axis | 0.16 % | S2 stream-mma |
| 10 | **C10 streamed attention off the matrix instruction**: TF32 attention runs FFMA only (608 FFMA, 0 HMMA, 2005 µs) vs Triton's 256 `HMMA.1688.F32.TF32` (795 µs); the bf16 stream at 37 % of peak (2-warp CTAs, f32-staged windows); its f32 output not round-stored, so the O-proj loop converts (16 `F2FP` per 32 HMMA) | 4 | tf32, bf16 | **confirmed** (SASS + trace): Triton `_attn_fwd` 256 HMMA; Mapal attention kernel 1024 CTAs × 256, 167 regs, FFMA only | `stream::mma_fmt` admits TF32 m16n8k8, P's accumulator→A relayout as an in-quad shuffle derived from the `MatrixOp` fragment geometry; bf16-staged windows + `ldmatrix`, 4-warp blocks; `precision_plan.store` extended to a stream group's stored output | 0.38 % | S2 stream-mma |
| 11 | **C2 read-once operand on the default L2 policy**: Mapal loads a read-once operand larger than the L2 keep with `LDG.E.128.CONSTANT` where cuBLAS gemvx streams it `LDG.E.EF` (evict-first) | 4 (+2 minor) | f32 | **confirmed**: gemvx SASS 119 `LDG.E.EF` + 119 `LDG.E.STRONG.SM`; Mapal 32 `LDG.E.128.CONSTANT`; `pj_m16_n6144` tile 83.6 → 67.0 µs with `ld.global.cs.nc` (16 loads rewritten), pj_m1_n6144 69.0 → 58.5, pj_m1_k14336 232 → 210, pj_m128_n6144 −8.3 µs; null on k14336-m128 and n4096_k4096 | per tile-site operand: grid reuse 1, no later reader (`residence_plan`), bytes > `Gpu::l2_keep_bytes` → streaming loads; capability: L2 eviction priority (`.cs` / `createpolicy L2::evict_first`) | 0.32 % | B5 |
| 12 | **C13 conv window operand not contiguous in k** (`tensor_form` refuses im2col windows) → FFMA implicit-GEMM tile at 41 % of peak vs cuDNN's NHWC transpose (12.5 µs) + TF32 tile at 67 % | 1 | tf32 | **confirmed** (trace): cuDNN nchwToNhwc ×2 + `xmma_fprop_implicit_gemm…tf32` + nhwcToNchw; Mapal one tile 392 CTAs, 163 regs | a `move_plan` data-move site making the window's k contiguous once so the TF32 tile is admitted (priced against the FFMA tile) | 0.22 % | S2 conv-move |
| 13 | **C6 FFMA tile shared-pipe floor**: T4 counts LSU at `lsu_ipc_pm` (LDS.32 rate), so 8×8 (16 FFMA / LDS.128) and 8×16 (21.3) price equal; `k_fill` forces wide-fma off under stream-K | 4 | f32, mlp, attn | **confirmed** (counts): Mapal `pj_m128_n4096_k14336` 1024 FFMA / 64 LDS.128 / 0 LDGSTS vs cuBLAS 256×128 1152 / 54 + 48 LDGSTS; wisdom n28672 8×8 841 vs 8×16 796 µs, both priced 752.1 | T4 in shared-pipe clocks (128-bit shared ops at `lds_bcast_pm`, 32-bit at `lsu_ipc_pm`); `k_fill` offers 8×16 as a stream-K candidate | 0.06 % | S2 fma-tile |
| 14 | **C14 bf16w precision path**: f32 activations × bf16 weights stay on the FFMA tile (66 % of FFMA peak); the realized matrix:FFMA ratio is ~2.8× while the peak ratio (2.0×) is under `SPLIT_RATE_PM` (1.6 × products) | 2 | bf16w | **unconfirmed**: no arm; the f32-KV bf16w rows are the program's own storage (f32 attention at 869 GB/s, 90 % of DRAM), like-for-like = the `_bf16` / `_bf16kv` twins | admit `bf16x2n` by realized pipe fractions (a measured per-tile-kind fraction property) instead of peak rates | 0.27 % | S2 bf16w-admit |
| — | **H harness boundary**: leaders event-timed after a 256 MB flush hide their host dispatch (torch-compile 30.6 µs/call under CUPTI); Mapal is a one-call wall clock. Leader-leg variance: fa2 b8 l8192 8.73 ms today vs 7.79 in the matrix | 2 (+re-bases ~10) | bf16, bf16w (+ every small row) | **confirmed**: CUPTI e0 → 2× `cuLaunchKernel` → e1 spans 30.6 µs behind the flush; fa2 re-measured twice | time every leg with the program's boundary (host clock around call + synchronize, idle queue) and report device span beside wall; re-time fa2 | 0.13 % | B1 |

## Spot-checks (this step)

| cause | row | what was checked | verdict |
|---|---|---|---|
| C1 | pj_m1_n4096_k4096 | `trace/pj_m1_n4096.api.2.tsv`: timed call = `cuMemsetD8Async` 3.66 µs → `cuLaunchKernel` 2.72 µs → `cuCtxSynchronize` 19.75 µs | confirmed |
| C1 | lsm_32_8192 | `attention/ktrace/m_lsm_32_8192.tsv`: kernels 2.05 / 3.07 / 3.07 µs, gaps 3.33 + 0.51; leader 3.33 + 2.82, gap 0.26 | confirmed |
| C2 | pj_m16_n6144_k4096 | `exp/m16.{base,cs}.tsv.gz`: `mapal_tile_1e61…` 48×16 × 32 thr, 160 regs: 83.6 → 67.0 µs; `pj_m16_cs.ptx` 16 `ld.global.cs`; gemvx/Mapal SASS load mix above | confirmed |
| C3 | pj_m128_n28672_k4096_bf16 | `abba_mma_depth.txt` 0.3996 / 0.3970 → 0.3422 / 0.3416 ms; DEPBAR 0x0 vs 0x4 in `m_tile.sass` / `l_triton_chosen.sass`; same arm loses +3 % / +4 % / +12 % on k14336 / k4096 / m2048 (the gate is required) | confirmed |
| C3 | pj_m128_n4096_k4096_tf32 | `m128_mapal.sass` 32 HMMA, 96 FSETP, `DEPBAR.LE 0x0`; `m128_cublas.sass` 16 HMMA, 48 F2FP, `DEPBAR.LE 0x4` ×2 | mechanism confirmed, size plausible |
| C4 | pj_m2048_n4096_k4096_bf16 / n6144 | `m.ktrace.tsv` grid 32×16 × 128 thr, 218 regs, 37 888 B; `abba_m2048_fill.txt` resident-4 1.013/1.019 and split-2 1.047/1.044 vs 0.936 ms | mechanism confirmed, fix unmeasured |
| C5 | softmax_rows_4096 | `m_softmax_rows_4096.tsv` memset 67 108 864 B 41.2 µs, tile 168×128; SASS 384 `REDG.E.ADD.F32.FTZ.RN.STRONG.GPU` | confirmed |
| C6 | pj_m128_n4096_k14336 | static counts in `sass/*.sass` (table row 13) | confirmed |
| C7 | attn_2048_64 / attn_1024_64 | `sass-mapal/k_8845…sass` vs `sass-leader/{attn_2048_64,a1024}_softmax.sass` BAR / SHFL / MUFU / LDG counts | confirmed (no arm) |
| C8 | llm_dec_b32_l2048_bf16kv | the program source (fold over captured `rD`); attention's norm SASS excerpt | confirmed |
| C10 | llm_pre_t2048_tf32 | `triton_attn_fwd_tf32.sass` 256 HMMA; `kt_llmpre.mapal.tsv` attention kernel 1024 × 256 thr, 167 regs | confirmed |
| C11 | pj_m2048_n28672_k4096_bf16 | `abba_n28672_l2gate.txt`; `profile.rs` `Gpu::raster` gate expression | confirmed |
| C12 | llm_dec_b32_l8192_bf16kv | `k_3bede.sass` load/store mix; `k_3bede.ll` index algebra (`%v45`, stores at +512 B) | both mechanisms confirmed, split unconfirmed |
| C13 | convmc_8_64_56_3_tf32 | `kt_conv.{lead,mapal}.tsv` kernel lists | confirmed |
| C9, C14 | blkdec_b1 / pre_bf16w | family traces only | plausible / unconfirmed |

## Corrections across the family reports

- **TF32 pj rows carry host time too.** tensor-tf32 reports `outside 0` (it counted idle between kernels); the
  matrix ratio is the one-call wall: `pj_m128_n4096_k4096_tf32` is 134.7 µs wall vs a 125.3 µs kernel, against
  cuBLAS's 84.0 µs event bracket. C1 / H apply to every single-kernel pj row (f32, tf32, bf16: 4–18 µs outside
  vs 2–5 for the leader); the estimates above do not count it for the tensor rows.
- **Decode scores kernel: one kernel, two diagnoses.** attention: the 8-way STS conflict; tensor-16bit: per-element
  narrow staging. Both are in `k_3bede`; no arm separates them, so one branch (B5) owns the whole copy and A/Bs
  each half (PTX-edit arms) before any claim of shares.
- **blkdec_b1 56-CTA GEMV**: attention calls it grid spread, tensor-16bit the merged epilogue losing the split —
  the same kernel; the dataflow cause is the merge (C9), the grid-spread pricing (`grid_spread_pm`) is its check.
- **llm_dec_b8_l8192_{bf16,bf16w}**: the fa2 leg ran 8.73 ms twice today vs 7.79 in the matrix; the bf16 row is a
  win today (0.94). Re-time before working on it.
- **Plan model under-prices the FMA tiles it ships** (gemm-f32): pj_m128_n6144 plan 169.1 vs 192.6 µs realized,
  pj_m128_n4096_k14336 375.9 vs 435, pj_m128_n28672 752.1 vs 781.3, mm_16 cell 68.3 vs 78.7. Each fix adds its
  measurement as a `rung_replay.tsv` line.
- Prior explanations corrected by the families: indep_512 / mlp_512 have no zero pass in the clock;
  pj_m1_n4096_k4096 kernels are at parity (not L2 residency); mm_16 ships S=56 atomic (not 84-CTA stream-K);
  attn_1024_64's QKᵀ beats cuBLAS (the loss is the softmax unit + P.V atomic + memset); lsm_4_32768's kernels
  are faster than the leader's (the loss is two launch gaps); pj m2048 n28672 bf16 is the raster gate; the
  `mma-depth` and `mma_pieces` "mixed" notes are explained by the missing warp-loop efficiency (C3 / C4).
- Open after all causes: pj_m1_n6144 keeps 10.5 µs of kernel gap after streaming loads (58.5 vs 48.4 µs; S=16
  k-slices put the DRAM rows on ~42 of 96 CTAs; S=4 measured worse — unverified).

## Per-row table

Time split: kernel µs Mapal / leader; outside = wall − kernel (Mapal) / host time in the leader's bracket.
Decode rows: per 16-step program unless "/step".

| workload | prec | ratio | leader leg + kernel | Mapal kernel | time split | cause | fix |
|---|---|---:|---|---|---|---|---|
| pj_m1_n6144_k4096 | f32 | 1.621 | cublas-fresh: gemvx<…,9> 48×(128×4), LDG.E.EF | GEMV S=16 atomic, LDG.E.128.CONSTANT | k 70.8/48.4; out 12.2/2.8 | C2 + C1 | streaming loads (69.0→58.5); zero off host path |
| pj_m1_n4096_k4096 | f32 | 1.540 | lt-fresh: gemvx 64×(64×8) | GEMV S=32 atomic | k 17.4/17.7; out 14.2/2.8 | C1 | dispatch at load; zero off host path |
| pj_m128_n6144_k4096 | f32 | 1.127 | cublas-fresh: cutlass_80 simt 256x128_8x4 sK7 | 8×8 stream-K 167×256, 127 regs, reg-staged | k 192.6/176.9; out 10.5/3.3 | C2 + C6 + C3 | streaming B (−8.3 µs); 8×16; cp.async depth |
| pj_m128_n4096_k14336 | f32 | 1.111 | cublas-fresh: 256x128_8x4 sK10 | 8×8 stream-K 168×256, 128 regs | k 435.0/398.8; out 11.1/2.6 | C6 + C3 | width-weighted T4; 8×16 under stream-K |
| pj_m16_n6144_k4096 | f32 | 1.096 | lt-fresh: 128x32_8x5 sK3 | 768 one-warp CTAs S=16 atomic, 160 regs | k 83.0/85.1; out 13.6/3.0 | C1 + C2 | host path; streaming 83.6→67.0 (→0.88) |
| mm_16x2048x7168 | f32 | 1.086 | cublas: 128x32_8x5 sK9 + splitKreduce | S=56 atomic, 896 one-warp CTAs, 1.84M REDG | k 78.7/75.4; out 7.3/4.5 | C5 | combine as a tail → S≈14 (70.3) |
| pj_m128_n28672_k4096 | f32 | 1.076 | lt-fresh: 256x128_8x4 (336 CTAs) | 8×16 stream-K, reg-staged; zero 12.4 µs | k 781.3/742.8; out 20.5/2.7 | C3 + C5 | realized cover → cp.async stages; zero fold |
| pj_m1_n4096_k14336 | f32 | 1.067 | cublas-fresh: gemvx sK4 + reduce | GEMV S=28 atomic | k 233.1/227.3; out 11.7/2.1 | C2 + C1 | streaming 232→210; host path |
| indep_512 | f32 | 1.253 | torch-compile: triton_tem_fused_add_0 (1 kernel, 64×64×32, 3-stage) | 2 tiles 16.0+16.2 (128×128 thr, 4×4) | k 32.2/29.2; out 8.9/3.6 | C1 + C9 (H) | dispatch at load; products over one cell as one site |
| mlp_512 | f32 | 1.166 | torch-compile: 2× triton_tem_fused_mm_mul | 2 tiles 13.5+15.7 | k 29.2/27.6; out 9.0/5.2 | C1 (H) | dispatch at load; equal boundary |
| smlp_4096_256_1024 | f32 | 1.061 | torch-compile: triton addmm_relu + cutlass 128x64_8x5 | bias kernel + 2 products (4×8 stream-K) + 4 MB zero | k 144.8/142.8; out 13.8/6.7 | C5 + C8 + C6 | zero fold; bias by law in the epilogue; 8×8 tile |
| pj_m128_n4096_k4096_tf32 | tf32 | 1.604 | cublas-tf32-fresh: s1688gemm 64x64_16x6 sK8 serial | 64×64, 8 warps 32×16, bk32×2, DEPBAR 0x0 | k 125.3/84.0; out ~9.4/0 | C3 + C5 | ILP floor; Little's-law stages; serial fixup |
| pj_m128_n6144_k4096_tf32 | tf32 | 1.451 | lt-tf32-fresh: nvjet_sm120 tf32 80x128x32 TMA | 64×128, 32×32 warps, bk16×3, S=8 atomic | k 182/127 | C3 + C5 | same |
| pj_m128_n4096_k14336_tf32 | tf32 | 1.301 | cublas-tf32-fresh: 64x64_16x6 sK7 | 64×64 bk32×2 S=8 atomic | k 396/311.3 | C3 + C5 | same |
| pj_m128_n28672_k4096_tf32 | tf32 | 1.182 | cublas-tf32-fresh: 64x64_16x6 (896 CTAs) | 128×128, 64×64 warps, bk16×2, 224 CTAs; DRAM 75 % vs 89 % | k 688/591.9 | C3 | bytes in flight by Little's law |
| pj_m2048_n6144_k4096_tf32 | tf32 | 1.153 | cublas-tf32-fresh: 64x64_16x6 (3072 CTAs) | 128×128, 768 CTAs (9.14/SM → 10) | k 2005/1748 | C4 | tile by per-SM rounds |
| pj_m2048_n4096_k4096_tf32 | tf32 | 1.150 | lt-tf32-fresh: nvjet 128x80x32 TMA | 128×128 S=4 atomic (33.5M red) | k 1330/1155 | C4 + C5 | n=8j tiles; ordered combine |
| pj_m2048_n28672_k4096_tf32 | tf32 | 1.125 | lt-tf32-fresh: nvjet 256x128x32 | 128×128, 3584 CTAs, raster 1 | k 9060/8060 | C11 + C4 | reuse-distance raster (−5 %); 8-warp CTA |
| llm_pre_t2048_tf32 | tf32 | 1.176 | triton-tf32: _attn_fwd 795 µs + cutlass s1688 | attention FFMA-only 2005 µs | k 5600/5200; out 2/300 | C10 | TF32 streamed attention on mma |
| llm_blkpre_t2048_tf32 | tf32 | 1.092 | triton-tf32: cutlass 256x128_16x3 / 128x128_32x3 + _attn_fwd | 128×128 bk16×2 GEMMs (+1.44 ms), FFMA attention (+1.15 ms) | k 21260/19230 | C10 + C4 + C11 | |
| convmc_8_64_56_3_tf32 | tf32 | 1.179 | torch-eager-tf32: cuDNN nchwToNhwc + xmma tf32 + nhwcToNchw | FFMA implicit-GEMM 392 CTAs, 163 regs | k 71.5/56.7; out 0/13 | C13 | window move → TF32 tile |
| pj_m128_n4096_k14336_bf16 | bf16 | 1.299 | torch-bf16: cutlass bf16 128x64_64x3 sK5 serial | 64×64, 32×16 warps, 2 stages, 128 CTAs | k 204.6/167.2; out 18.0/4.8 | C4 + C1 | per-SM rounds; ordered k-split |
| pj_m128_n28672_k4096_bf16 | bf16 | 1.254 | triton-bf16: matmul 128x128x32 s4 (DEPBAR 0x4) | same geometry, 2 stages, DEPBAR 0x0 | k 394.4/313.2; out 4.3/2.2 | C3 | Little's-law stages (arm → 1.077) |
| pj_m128_n4096_k4096_bf16 | bf16 | 1.187 | torch-bf16: cutlass 128x64_64x3 (64 CTAs) | 128 CTAs 64×64 (32×16 warps) | k 54.9/50.3; out 8.0/2.9 | C1/H + C4 | equal boundary; tile by per-SM work |
| pj_m2048_n28672_k4096_bf16 | bf16 | 1.153 | torch-bf16-compile: triton tem 128x64 GROUP_M | 128×128 grid 224×16 row-major | k 4565.7/3947.5 | C11 | reuse-distance raster (arm → 1.007) |
| pj_m2048_n4096_k4096_bf16 | bf16 | 1.127 | compile-ro: cutlass 64x64_32x6 (2048 CTAs) | 512 tiles 128×128, 7 rounds vs 6.10 | k 666.7/590.6 | C4 | per-SM rounds |
| pj_m2048_n4096_k14336_bf16 | bf16 | 1.082 | compile-cg: cutlass 128x64_64x3 serial split 2 | S=4 atomic, 4×33.5 MB RED + zero | k 2233.9/2092.2; out 27.9/0 | C5 | ordered combine priced vs RED |
| pj_m2048_n6144_k4096_bf16 | bf16 | 1.061 | compile-ro: cutlass 64x64_32x6 | 768 tiles, 10 rounds vs 9.14 | k 929.2/867.1 | C4 | per-SM rounds |
| llm_pre_t2048_bf16 | bf16 | 1.372 | compile-cg: cutlass 64x64_32x6, triton addmm, flash_fwd | stream attention 751 vs 405; O-proj on f32 a 805 vs 594; rope/norm 148 vs 48 | k 2637/1906 | C10 + C7 + C4 | bf16 stream + ldmatrix; round-store stream output |
| llm_blkpre_t2048_bf16 | bf16 | 1.185 | compile: triton fused_mm_mul_silu / addmm_mul_silu, flash_fwd | down 7 rounds + residual epilogue; silu·mul own kernel | k 10008/7938 | C4 + C9 + C10 | |
| llm_dec_b32_l8192_bf16 | bf16 | 1.091 | flashinfer: BatchPrefillWithPagedKV (HMMA, one pass) | twin of bf16kv (identical sites, inferred) | k 26.58/21.25 ms; out 0/3.11 ms | C12 | kvec_b layout + 16-B narrow loads |
| llm_dec_b32_l2048_bf16 | bf16 | 1.087 | compile: triton tem QKV+norm (cp.async rings) | 1-warp reg-staged projections 660 GB/s; 1-CTA norm 282 vs 27 µs | k 8.12/6.93 ms; out 0/0.47 | C12 + C8 | |
| llm_dec_b8_l8192_bf16 | bf16 | 1.053 | fa2: flash_fwd_splitkv + combine | attention 3158+2702+264 µs | leader 8.73 ms today (7.79 matrix) | H + C12 | re-time; narrow staging |
| llm_blkdec_b1_l4096_bf16kv | bf16kv | 1.100 | compile: triton red fused norm+mm, mm_mul_silu | 56-CTA GEMV 691 GB/s; 2 norms 17.9+16.8 µs | /step 612.8/557.3 | C9 + C8 | epilogue keeps split; norm by law |
| llm_dec_b32_l8192_bf16kv | bf16kv | 1.092 | flashinfer: BatchPrefillWithPagedKV | scores 809.7 µs/step at 663 GB/s (8-way STS), P.V 656.9 at 817 | /step k 1696/1368; out 1/219 | C12 | |
| llm_dec_b32_l2048_bf16kv | bf16kv | 1.088 | compile: triton tem QKV+norm, red QK (LDG.E.EF.128) | QKV 76.6 µs 657 GB/s 1-warp split-K 8; scores 766 GB/s | /step 506.6/431.7; out 0/32 | C12 + C8 | |
| llm_dec_b8_l8192_bf16kv | bf16kv | 1.052 | fa2: splitkv + combine | scores 197.8 µs/step 679 GB/s | /step 515.9/457 | C12 + C8 | |
| llm_pre_t2048_bf16w | bf16w | 3.296 | compile-cg (bf16 activations, tensor cores) | FFMA tiles, QKV 2550 µs (66 % FFMA peak) | k 6299/1906 | C14 | realized-rate admission (unconfirmed) |
| llm_blkpre_t2048_bf16w | bf16w | 3.152 | compile (bf16) | FFMA tiles + f32 attention | k 25397/7938 | C14 | same |
| llm_dec_b32_l8192_bf16w | bf16w | 1.798 | flashinfer (bf16 cache) | f32 KV attention 869 GB/s (roofline) | k 43.2/21.3 ms | storage (+C8) | like-for-like = bf16 twin |
| llm_dec_b32_l2048_bf16w | bf16w | 1.692 | compile (bf16 cache) | f32 KV 5.00+4.95 ms; FMA-tile projections | k 12.6/6.93 ms | storage + C8 + C12 | |
| llm_dec_b8_l8192_bf16w | bf16w | 1.584 | fa2 | f32 KV 5.00+4.96 ms | leader 8.73 today | storage + H | |
| llm_dec_b8_l2048_bf16w | bf16w | 1.262 | compile-cg | norm 290 vs 23 µs; projections +170 | k 4761/3100; out 0/676 | C8 + C12 + storage | |
| llm_blkdec_b1_l4096_bf16w | bf16w | 1.129 | compile: triton red fused | merged up+silu cell kernel on 56 CTAs, 664 GB/s | k 10148/8436; out 0/641 | C9 + C8 | |
| lsm_32_8192 | f32 | 1.882 | torch-compile: tem mm + red softmax | 3 dependent launches 2.05+3.07+3.07 | k 7.96/5.93; out 10.1/4.3 | C1 + C7 | launch_host_ns merge; row unit |
| attn_1024_64 | f32 | 1.284 | compile-cg: cutlass 128x64 + triton per softmax + 64x64 sK8 + reduce | softmax 8.26 vs 3.23 (19 BAR, 0 SHFL); P.V split-8 atomic + 2.4 µs memset | k 28.6/23.6; out 9.1/7.1 | C7 + C5 | |
| attn_2048_128 | f32 | 1.235 | sdpa-compile-cg: cutlass 128x128_8x4 sK5 | P.V split 4 (64 CTAs) 43.5 vs 31.8; softmax 24.1 vs 15.2 | k 107.3/85.5 | C5 + C7 | uneven split → 5 |
| lsm_4_32768 | f32 | 1.184 | compile-cg: 6 graph kernels | 3 launches, kernels faster (8.1 vs 9.74) | out 8.1/4.6 | C1 | launch_host_ns merge |
| attn_2048_64 | f32 | 1.154 | compile-ro: cutlass + triton red softmax | softmax 20.8 vs 11.65 | k 70.2/63.9 | C7 + C6 | |
| softmax_rows_4096 | f32 | 1.076 | compile-ro: cutlass 256x128_8x4 serial sK3 | stream-K all-tile REDG + 64 MB memset 41–44 µs | k 3751.5/3537.3 | C5 + C7 | DP + SK |
| rowsum_4096 | f32 | 1.071 | compile-ro: cutlass 256x128 + triton red sum (LDG.E.EF.128) | same GEMM + memset 72 µs; row kernel scalar LDG 66.4 vs 38.5 | k 3745.4/3498.5 | C5 + C7 | DP + SK; 16-B rows |

## The changes, per cause

| cause | change (fact / property / capability / rule) | files | generated-program property test | OFF arm | other parts, by property |
|---|---|---|---|---|---|
| C1 | runtime: site pick + `CUfunction` at load; zero of an atomic output folded / a zero kernel; property `DeviceCost::launch_host_ns` (`calib.cu`) priced in `dep_launch_at` / `launch_at` | `crates/mapal-rt/gpu/mapal_gpu.c`, `mapal_choice.h`, `func/zero.rs`, `profile.rs` (`dep_launch_at`), `rules.rs` (`row_split_pays`, pdl rule), `benches/gpu/calib.cu` | extend `zero_fold_on_generated_programs` (atomic sites: no memset API in the timed region; values vs interpreter) and `calls_issue_is_monotone_on_generated_programs` (launches non-increasing as `launch_host_ns` grows) | `--gpu-off=load-dispatch`, `--gpu-off=zero-host`; `launch_host_ns = 0` = unpriced | every part: host enqueue is the box's CPU + driver (measured per box); small programs (attn, mlp) everywhere |
| H | harness boundary: wall clock around call + synchronize for every leg, device span reported beside it; fa2 re-timed | `benches/gpu/torch_bench.py` (`timed`), `gemm_lib.cu` (`time_once`), `comp_bench.py` | — (harness) | old bracket kept as a column | every part's small rows re-base |
| C2 | rule: tile-site operand with grid reuse 1, no later reader (`residence_plan`), bytes > `Gpu::l2_keep_bytes` → streaming loads; capability `Caps` L2 eviction priority | `func/tile.rs` / `func/vec.rs` / `func/gpu.rs` panel + GEMV load emission, `profile.rs` (`Caps`), `intent.py` / `evidence.py` (`LDG.E.EF` intent) | new `streaming_loads_on_generated_programs`: operands around the keep; evict-first exactly on the qualifying operands; values equal | `--gpu-off=stream-ld` | 3090 (6 MB L2) qualifies more operands; A100 (40 MB) / H100 (50 MB) decode GEMVs; risk: operands re-read within the keep |
| C3 | rule: stages by Little's law (`residence_plan` = Dram, `dram_bytes_per_us_per_sm × lat`), busiest-SM DRAM/pipe ≥ ~0.85 gate; property `MatrixOp` latency / interval (`mma_calib.cu`) as the warp tile's ILP floor; swizzled panels, `wait_group(stages−2)`, double-buffered fragments; capability single-instruction TF32 rounding (`cvt.rn.tf32` → `F2FP`); FMA tiles: realized cover property vs `lat_ns`, cp.async stages under stream-K | `profile.rs` (`gpu_mma_geom_parts`, `mma_depth_pick`), `func/gpu.rs` (mma emitter ~L7990–8410), `rules.rs` (`mma_depth`, `warp_fill`); fma: `func/gpu.rs` `k_fill`, `gpu_tile_async`, `rules.rs` cp_async | extend `mma_combine_on_generated_shapes` / `large_gemm_rules_on_generated_shapes`: stage count = the rule; DEPBAR depth = stages−2; HMMA per warp k-step ≥ floor; values vs bound; intent check `cover_ns ≥ lat_ns` | `mma-depth` (now rule-driven), `--gpu-off=mma-ilp`, `cp-async` | A100 / H100: more DRAM bytes per µs per SM → deeper stages (their TF32 geomeans 1.646 / 2.625); H100 also needs its TMA / wgmma capability; ILP floor per part from `mma_calib` |
| C4 | rule: tile + k-walk by `⌈pieces/SMs⌉ × piece_ns / efficiency(warp tile)`; candidates 128×64, 64×64 at the same warp tile, `n = 8j` | `profile.rs` (`gpu_mma_geom_parts` candidates), `func/gpu.rs` (`mma_pieces`, `MmaCut::ns`) | `large_gemm_rules_on_generated_shapes` over random (M, N, K) × the five parts' property sets; `rung_replay` lines | `--gpu-off=tile-rounds` | SM count decides: 512 tiles = 4.0 rounds on 128 SMs (4090: no loss predicted), 4.74 on 108 (A100), 3.9 on 132 |
| C5 | DP + SK stream-K (owned tiles `STG`, SK tiles via workspace + ticket); ordered serial fixup for k-cuts; uneven last slice (`trip_split`); combine priced as a tail of a one-wave grid | `func/gpu.rs` (`streamk_priced`, `launch_split`, tile-epilogue combine, `mma_combine`), `func/stream.rs` (`split_combine` reused), `mapal-ir/src/algo.rs` (`trip_split`), `profile.rs` (new `combine_ns`) | `streamk_price_on_generated_shapes`, `streamk_epi_rule_on_generated_shapes`: no whole-output zero when DP tiles store, REDG count = SK tiles × cells, slices cover k once; values vs interpreter, determinism on fixed-order | `--gpu-off=dp-sk`, `--gpu-off=ordered-combine`, `--splitk` | atomic rate (`atomic_ops_per_ns`) and slots differ per part; every part running stream-K drops the zero pass |
| C6 | T4 floor in shared-pipe clocks (`lds_bcast_pm` for 128-bit, `lsu_ipc_pm` for 32-bit); `k_fill` offers 8×16 under stream-K | `profile.rs` (`site_ns` / `site_ns_at`, `PlanShape.lsu`), `func/flow.rs` (lsu count ~L2038), `func/gpu.rs` `k_fill`, `rung_replay.tsv` | `large_gemm_rules_on_generated_shapes` + replay of the n28672 pair (841 vs 796) | `--gpu-off=lsu-width`, `wide-fma` | 64-lane SMs (`wide_fma_pays`) differ; per-part `lds_bcast_pm` from `calib.cu` |
| C7 | `tree_total` with shuffle levels + one smem exchange; row buffer in registers when it fits; rows per CTA from `wave_full`; `precision_plan`: reciprocal of an axis-constant divisor (one extra rounding in the bound); exp form by argument range | `func/flow.rs` (`tree_total`, unit row buffers), `rules.rs` (rows per CTA), `mapal-ir/src/algo.rs` (`precision_plan`) | `row_block_on_generated_chains`, `tree_gate_on_generated_chains`: values within the default-face bound; exact / fixed-order faces keep the smem order | `--gpu-off=shfl-tree`, `--gpu-off=row-regs`, `--gpu-off=recip` | all parts (shuffle and bank geometry universal); A100 attention 1.029 |
| C8 | iota fold domain read by its `elem_plan` law through loop captures; a row fold with rows < SMs spread over CTAs; closed-form operand evaluated in its consumer's epilogue | `func/rowfold.rs`, `func/flow.rs` (cell nested fold), `mapal-ir/src/algo.rs` (`elem_plan` through captures) | new `iota_law_on_generated_loops`: a fold over a captured iota loads no array; values equal | `--gpu-off=iota-law`, `--gpu-off=norm-spread` | all parts (removes a dependent load; spread by SM count) |
| C9 | epilogue after the combine keeps the split (`fold_combine`); same-cell consumer joins the tile epilogue (`unit_plan` / `tile_plan_post`) | `func/gpu.rs` epilogue placement, `mapal-ir/src/algo.rs` (`tile_plan_post`) | `round_store_on_generated_programs`-style: merged epilogue keeps the producer's grid | `--gpu-off=epi-split` | all parts (grid vs SMs per part) |
| C10 | `stream::mma_fmt` admits TF32 m16n8k8; P relayout derived from `MatrixOp` fragment maps; bf16-staged windows + ldmatrix; `precision_plan.store` over stream outputs | `func/stream.rs` (`mma_fmt`, `stream_ns`), `profile.rs` (`MatrixOp` fragment geometry), `mapal-ir/src/algo.rs` (`precision_plan` store) | `stream_mma_on_generated_attention` extended to TF32 and bf16 windows: values within bound, HMMA present | `--gpu-off=stream-tf32` | every sm_80+ part has m16n8k8 TF32; H100 also wgmma |
| C11 | raster gate on `gx > wave ∧ gx·bn·k·elem > l2_keep_bytes`, raster only | `profile.rs` (`Gpu::raster`), `rung_replay.tsv` | `large_gemm_rules_on_generated_shapes`: raster ≠ 1 iff the reuse distance exceeds the keep; split choice unchanged | `l2-gate` (now the reuse-distance gate) | 3090 (6 MB L2) is the gate's anchor and must keep its 8192 decision; 4090 (72 MB) fewer cases |
| C12 | `kvec_b` layout from bank geometry (TargetProfile: 32 banks × 4 B); narrow copies sized in bytes; small-m DRAM sites register-direct / narrow cp.async, 4–8 warps | `func/gpu.rs` (`kvec_b`, lane maps, `panel_excess`, register-staged copy width) | `kmajor_b_on_generated_decode`: per-warp store banks distinct (static from the lane map), 16-B narrow lane loads; values | `--gpu-off=kvec-swizzle`, `--gpu-off=narrow-width` | bank geometry is the same on all five parts → all decode rows move |
| C13 | `move_plan` window move to channel-last so `tensor_form` admits; priced vs the FFMA tile | `mapal-ir/src/algo.rs` (`move_plan` / `tensor_form` windows), `func/gpu.rs` | new generated conv/window programs: move admitted iff the moved operand has a tensor form; values | `--gpu-off=window-move` | all TF32 parts |
| C14 | admission of `bf16x2n` by a measured realized-fraction property per tile kind | `rules.rs` (`split_rate_pays`), `profile.rs` | generated bf16w programs: admission per realized rate, values within bound | `--gpu-off=realized-admit` | only parts whose peak matrix:FFMA ratio sits under the rule (GeForce: 3090, 4090, 5080); A100 / H100 already admit |

## Sprint plan

**Day 0 (serial, one small commit on the base, before the branches fork):** the new properties and capabilities
and their calibration, so no two branches edit the property tables: `DeviceCost::launch_host_ns`
(`calib.cu`), `MatrixOp` issue latency / interval and fragment-map geometry (`mma_calib.cu`), `Caps` L2
eviction priority, TargetProfile shared-memory bank geometry. Each defaults to "unmeasured = today's text".

**Sprint 1 — five parallel worktree branches, disjoint functions:**

| branch | causes | owns (functions) | must not touch | expected |
|---|---|---|---|---|
| `host-path` | C1, H | `mapal_gpu.c` / `mapal_choice.h` dispatch + zero issue; `func/zero.rs`; `profile.rs` `dep_launch_at` / `launch_at`; `rules.rs` `row_split_pays` / pdl rule; `benches/gpu/{torch_bench.py,gemm_lib.cu,comp_bench.py}` timing | tile emitters, `site_ns` | 0.96 % (+ re-based small rows) |
| `tensor-tile` | C3 (mma), C4, C11 | `profile.rs` `gpu_mma_geom_parts`, `mma_depth_pick`, `Gpu::raster`; `func/gpu.rs` mma emitter (stages, swizzle, wait depth, fragments, cvt), `mma_pieces`, `MmaCut::ns`; `rules.rs` `mma_depth` / `warp_fill` | `mma_combine`, `streamk_priced`, `kvec_b` | 1.45 % |
| `split-combine` | C5 | `func/gpu.rs` `streamk_priced`, `launch_split`, FMA-tile epilogue combine, `mma_combine`; `func/stream.rs` `split_combine` (shared helper); `mapal-ir` `trip_split`; new `profile.rs` `combine_ns` | `site_ns` body, `MmaCut::ns` (tensor-tile calls `combine_ns` at integration) | 0.48 % |
| `row-units` | C7, C8 | `func/flow.rs` `tree_total` + unit row buffers + cell nested fold; `func/rowfold.rs`; `rules.rs` rows per CTA; `mapal-ir` `precision_plan` reciprocal, `elem_plan` through captures | flow.rs lsu count, merge rule | 0.70 % |
| `operand-path` | C2, C12 | `func/tile.rs` / `func/vec.rs` / `func/gpu.rs` load instruction emission (cache policy, width), `kvec_b`, lane maps, `panel_excess`; the small-m DRAM site rule | mma emitter copies (tensor-tile), `k_fill` | 0.68 % |

Sprint-1 total ≈ 0.965 → 0.925 if the estimates hold. Order inside each branch: the A/B arm first (PTX edit or
existing flag) on the 5080, then the rule, then `gpu_changed.sh` ABBA of every suite program whose text moves and
the full suite once; `evidence.tsv` per run; each cited measurement a `rung_replay.tsv` line.

**Sprint 2 (after sprint 1 merges):** `fma-tile` (C6 + C3-fma: `site_ns` T4, `k_fill`, cover property — waits on
`split-combine` because both price stream-K), `stream-mma` (C10 + C9: `stream.rs` `mma_fmt` / `stream_ns`,
`precision_plan.store`, epilogue placement), `conv-move` (C13), `bf16w-admit` (C14, evidence first: an arm forcing
`bf16x2n` on the bf16w prefill rows).

**Captures needed from the rented parts** (`cross-arch/f1d3a04f/<part>/gaps/`): C1 — API timing on each box's
host (launch / memset API cost differs by CPU); C2 — the leaders' gemv load policy and Mapal's pj_m1 / pj_m16
time split on the 3090 (6 MB L2) and A100 / H100 (HBM); C3 — TF32 m128 rows' DEPBAR depth and CTA timelines on
A100 / H100 (their TF32 losses are larger; H100 separately on wgmma / TMA); C4 — m2048 bf16 / TF32 tile counts
and per-round cost on the 4090 (128 SMs: predicts no quantization loss) and A100 (108); C5 — softmax_rows /
rowsum stream-K vs serial split on A100 / H100 (atomic rate); C11 — the 3090 m2048_n28672 and 8192 anchor
(the gate must keep it); C14 — realized FFMA vs bf16 tile fractions on the 3090 / 4090. C7, C8, C12 are
geometry-universal: the 5080 SASS / IR suffices; C12 still needs its local split arm.
