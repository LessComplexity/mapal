# hopper — disasm-rented: what got worse between rounds, and the bf16 full model against gpt-fast, from the compile path to SASS to the timeline (2026-11-26)

Branch `hopper/disasm` on `c808622c` (local-rtx-5080 with the five-GPU round's evidence). No compiler code changed.
Question (the user): *esp study the things that got worse, from compile path to asm/ptx to runtime.* Every row that
moved the wrong way between the f1d3a04f round and the five-GPU round (`b8429388` / `9be9886d`), per part, traced
through three layers: (a) the compile path (the emitted plan lines, site candidates, rules, the emission of both
compilers for the same source on the same part profile), (b) PTX / SASS (`ptxas -v` + `cuobjdump -sass` of both
emissions, instruction mix per kernel), (c) runtime — a **same-box A/B** of the f1d3a04f build (its emitter + runtime)
against the `b8429388` build (its emitter + runtime) on rented H100 / A100 / RTX 4090 / RTX 3090 boxes, ABBA × 10,
plus CUPTI timelines (`ktrace.cpp`) and the box's host round-trip costs (`lat.c`). Then the bf16-class full model
(Llama-3.2-1B, `examples/llama.mapal` via `variant.py --compute bf16`) against gpt-fast at prompts 512 / 2048 with
decode, CUPTI on both sides and SASS of the top kernels on both sides (gpt-fast's Triton cubins from the inductor cache,
cuBLAS `nvjet` / `ampere_*` kernels from the f1d3a04f gap captures and `modcap`), and the H100 kernel suite's worst
bf16 / tf32 / GEMM rows against the cuBLAS Hopper kernels.

## Regression table — every row that got worse (f1d3a04f → five-GPU), where it changed, and why

`then → now` = Mapal / best same-precision leg, both rounds under the current fair rule (`matrix.py` rebuilt over the
f1d3a04f raw); `now` in the event bracket (the f1d3a04f definition, like for like) and, after the slash, at the program
boundary (the five-GPU round's clock). "Worse" = the ratio moved by more than max(null, 5 %). Counts: 3090 44 rows,
A100 19, 4090 36, H100 50; the RTX 5080's local integrations (confirmation-fixes `39cd2c87` → gapfix-integrate
`d945c641`): none beyond its null (`ew_4096` ×1.05 = the 5 % floor; Mapal geomean ×0.956).

| row(s) | part | then → now | compile path | PTX / SASS | runtime | class | fix |
|---|---|---|---|---|---|---|---|
| `pj_m2048_n4096_k4096_bf16` | rtx-3090 | 1.10 → 1.48 / 1.36 (Mapal 1.151 → 1.315 ms) | `mma-walk` (bf16/tensor-tile, in `b8429388`) walks the 512-tile grid by stream-K: `walk=mma combine=streamk ctas=164 iters=400` where f1d3a04f launched the ordered grid (`combine=none`, 32×16); `--gpu-off=mma-walk` on the new emitter gives back f1d3a04f's kernel byte for byte (`mapal_tile_067321548751b91f`) | ordered → walk: static HMMA 128 → 64 (the k loop's body halves), BRA 2 → 7, regs 220 → 238, + RED.F32 ×128 (partials) + a zero pass kernel | **same 3090 box** (EPYC 7R32, 370 W, 1 935 MHz busy), CUPTI: ordered 1 082 µs (f1d3a04f = the OFF arm, 1 082 / 1 084) vs walk 1 275 µs + zero 39 µs; iter 1.095 → 1.322 ms (×1.21, all 10 pairs); control `pj_m2048_n6144_k4096_bf16` 1.550 = 1.550 | **Mapal code regression** | the walk's main loop as the ordered tile's (2 k-steps a trip, piece bounds once a tile segment), and the walk priced at the rate the part realizes (replay lines below; `mma_walk_replay`) |
| (the same rule in the full model) | a100 | new this round (no "then") | the 2048-row down projection (2048 × 2048 × 8192, residual epilogue) walks: `walk=mma combine=streamk`, 216 CTAs | walk: HMMA 128 / trip (= ordered) but +FADD 192, LDG 208 (workspace + ticket), RED 128, ATOMG 1, STG 80, BAR 5, regs 242 | **same A100 box**, the model built ON and `--gpu-off=mma-walk` (CUPTI): walk 788 µs a layer vs ordered 761 µs (×1.04); prompt busy 58.78 vs 58.57 ms; TTFT medians 57.5 vs 53.4 ms (host clock, ±10 % spread). The ordered grid is wave-quantized (256 tiles on 216 slots = 2 rounds, 59 %: 90 Tflop/s against gate+up's 145–174), which the walk was priced to remove (~−40 %) and did not | **Mapal code** (same rule: realizes ~0 of the priced gain) | as above; the down projection's wave quantization stays open either way |
| 19 small rows, Mapal-side (`lsm_4_32768` 1.56 → 2.52 / 1.34, `lsm_1_131072`, `lsm_32_8192`, `pj_m1_n4096_k4096_bf16` 1.12 → 1.61 / 0.92, `pj_m1_n6144_k4096_bf16`, `attn_1024_64`, `ew_4096`, `fir_4194304_16`, `conv_2048_5`, `ewconvew`, `firew_rowsum`, `mlp_512`, `indep_512`, `pj_m16_*`, `pj_m1_n4096_k14336*`, `gen_1024_512_256`, …) | h100 | Mapal ×1.07–1.74 (e.g. 0.0241 → 0.0419 ms) | launched kernels' SASS identical on 13 of 21 tested rows (renames = site-candidate hashes); on the other 8 the row-fold trees moved from shared memory to shuffles (`lsm` BAR 38 → 6, LDS 68 → 8, SHFL 0 → 20) or a tile gained a few registers (`pj_m16_n4096_k4096_bf16` 107 → 117); new `zero.host hoisted=1` + runtime `ZERO_KERNEL` (flag 128) | none that costs (the A/B) | **same H100 host as the five-GPU round** (host 214845, Xeon 8568Y+): new / old 0.81–1.02 on all 21 rows (`lsm_4_32768` 0.928, `pj_m1_n4096_k4096_bf16` 0.883, `ew_4096` 1.015 [0.65, 1.24]); the f1d3a04f traces (EPYC 9454 host) had d2h ×0.59, lead ×0.50, alloc ×0.39 of this host's; `lat.c` here: launch+sync 8.0 µs, 8-byte D2H 10.7 µs, alloc+free 3.5 µs (RTX 4090 box: 3.9 / 3.9 / 0.9) | **box: the host's round-trip cost** (a sub-0.1 ms program pays its launches and two synchronous 8-byte D2H at the host's round-trip cost) | host round-trip as a measured `TargetProfile` host property; one gathered copy-out per program (the probes read 2 corners = 2 synchronous D2H) |
| 31 rows, competitor-side (heavy tensor-core legs: `llm_pre_t2048*`, `llm_blkpre_*`, `pj_m128_*`, `pj_m2048_*`, `causal_1024_256`, …) | h100 | best leg ×0.57–0.72, Mapal ×0.67–0.85 | — | — | f1d3a04f's box was capped at 500 W, this round's at 700 W (1 980 MHz busy both): cuBLAS `nvjet` wgmma legs gain more from the power headroom than Mapal's `mma.sync` kernels | **box: power limit** | — (a part property, recorded; the gap itself is row 1 of the cause table) |
| 7 rows, Mapal-side (`attn_2048_64` 1.46 → 2.86 evt, `attn_2048_128`, `rt_mlp2_1024` 1.26 → 1.49, `llm_dec_b32_l2048_bf16w`, `pj_m16_n18432_k7168`, `pj_m128_n4096_k{4096,14336}_bf16`) | a100 | Mapal ×1.05–2.00 | identical kernels on 6 of 7; `llm_dec_b32_l2048_bf16w` re-planned (stream `s:site` → `s:cost`; an `mma` kernel 165 → 255 regs, swizzle 8×4) | identical / improved | the round's A100 box (EPYC 7713) ran its **null arm** ±50 % (`attn_2048_64` on 0.368 vs null 0.183 ms); its tuner `confirm` reverted `pj_m16_n18432_k7168`'s pick on pairs 0.614 / 1.432 / 1.388 (f1d3a04f kept it at 0.955); same-box today (EPYC 7K62): new / old 0.91–1.00 on all 7, pairs within [0.85, 1.13] | **box noise + tuner on noise** | `confirm_pick.py`: a pick reverts only when the pairs agree in sign beyond the box's null, else re-measure |
| 12 rows, competitor-side | a100 | best ×0.87–0.90 | — | — | host EPYC 7532 → 7713, the same ±30–70 % host noise (README "drift") | box | — |
| `decode_1024_256` | rtx-4090 | 0.41 → 0.88 | same kernels (SASS identical, 26 of 26) | — | Mapal 21.0 → 12.27 ms (×0.58, the SM clock 1.2 → 2.45 GHz); `chain-cache` (hand CUDA, 6 144 launches) 51.3 → 13.95 ms (×0.27): launch-bound — 8.3 µs a launch on the f1d3a04f box (EPYC 7713, wall 35–60 ms), 2.27 µs on the Ryzen 9950X; same-box A/B new / old 0.997; CUPTI: Mapal 2 052 launches, 9.7 ms of kernels in 10.8 ms | **box: clock lock + host** (competitor side) | — |
| attention / loops: `causal_1024_256` 0.14 → 0.35, `gen_1024_512_256` 0.14 → 0.35, `prefill_1024_512_256` 0.14 → 0.36 | rtx-4090 | Mapal ×0.50, best (`torch-compile-cg`) ×0.20–0.25 | row folds on shuffles (BAR 39 → 23, LDS 34 → 4, SHFL 21 → 31) | — | same-box new / old 0.908 / 0.912; the torch CUDA-graph loop legs replay ~1 000 graphs a run — host + clock bound | **box: clock lock + host** | — |
| LLM family 0.903 → 1.053 (37 rows; 22 with the same `torch-bf16-compile` leg ×0.60) | rtx-4090 | Mapal ×0.70, best ×0.59 | bf16 LLM rows re-planned (stream `s:site` → `s:budget`, kernel regs 143 → 177 / 127 → 193) — faster | — | same-box new / old 0.95–0.99; f1d3a04f gap capture (1.2 GHz box): torch-bf16-compile 144 kernels, 2.8 ms of 11.4 ms outside kernels (host gaps) against Mapal's 12 launches and 13 µs — a faster host closes the competitor's gaps, not Mapal's | **box: clock lock + host** (competitor side) | — (the remaining bf16 decode gap is causes 2, 4, 6) |
| 4 GEMM / tf32 rows (`pj_m128_n{4096,6144}_k4096_tf32`, `pj_m1_n2048_k7168`, `pj_m1_n28672_k4096`) | rtx-4090 | best (cuBLAS) ×0.54, Mapal ×0.66 | identical kernels | — | same-box new / old 1.006 / 0.909 | box (clock) | — |
| 43 rows, competitor-side (all families) | rtx-3090 | best ×0.80–0.84, Mapal ×0.93 | — | — | power 270 → 390 W, host EPYC 7702P → Ryzen 3900X | box (power + host) | — |

**Verdict.** One Mapal code regression: the bf16/tensor-tile `mma-walk` rule on Ampere (sm_86 suite row, sm_80 model
down projection). Everything else that got worse is the box — a clock-locked 4090 and a 500 W H100 / 270 W 3090
flattering Mapal last round, host round-trip costs and host noise this round — confirmed by the same-box A/B (new /
old ≤ 1.02 on 37 of 38 rows tested; the 38th is the walk). The program-boundary clock itself moved every part 5–11 % in
Mapal's favour (measurement boundary, not a regression).

## Ranked cause table — the bf16 class and the regressions, by time at stake

| # | cause | parts | rows / phases | evidence (SASS / timeline) | fix (generic: fact / property / rule) | covered by wgmma-16bit? |
|---:|---|---|---|---|---|---|
| 1 | 16-bit tensor tiles on `mma.sync` on Hopper (no wgmma, no TMA, no warp specialization) | h100 | prefill projections; suite bf16 rows 2.1–2.5× | Mapal `mma` m16n8k16: HMMA 128 + LDGSTS 48 + LDSM 32 a trip, 128 threads, 2 blocks / SM, 207–242 regs; 2048-row gate+up 235 µs (293 Tflop/s). Leader `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA`: HGMMA 64×N×16, UTMALDG, SYNCS (mbarrier), 384 threads (producer + 2 consumer warpgroups), cluster 1×2, persistent 2×64 grid. Full model at 2048: projections 14.6 ms vs 5.1 (cuBLAS less its full-vocabulary head) | the tensor tile on the part's matrix instruction as geometry (`Caps::matrix_op`: 64×N×16 from smem descriptors, TMA boxes, mbarrier stages) | **yes** |
| 2 | decode GEMV cells load bf16 weights with **scalar 16-bit loads** | all parts (largest on h100) | every decode step | PTX `ld.global.b16` ×16–64 a thread trip on h100 / a100 / rtx-4090 (SASS LDG.U16); h100 GEMVs at 41–65 % of 3.35 TB/s (q / o 6.2 µs for 8.4 MB, gate / up 15.4 µs for 33.5 MB, down 18.6 µs); gpt-fast's Triton GEMVs: LDG.E.EF.128 (16 B, evict-first) ×64 + EL.64 activations; the suite's tile GEMV (`pj_m1_*_bf16`) already uses 32-bit loads + PRMT | narrow element reads in cell kernels vectorized along the contiguous fold axis (16 B, widened in registers) from facts at hand: the read's stride along the fold axis (`step_algebra` per-read stride) and its storage bytes (`TileRead::narrow` / `elem_bytes`; the cell path has no `tile_plan_narrow` twin — extend `cell_plan` with it) | no |
| 3 | attention (`stream-mma`) on `mma.sync` with K / V by scalar 16-bit loads | all | prefill attention: h100 6.87 vs 1.51 ms, a100 11.96 vs 2.78, rtx-4090 5.53 vs 3.20 at 2048 | Mapal flow kernel `stream=b64xkb32`, 128 threads, HMMA 32 + FFMA 318 + MUFU 72 + LDG.U16 32 + LDGSTS 4; flex attention (Triton): HGMMA 64×64×16 / 64×128×16, LDGSTS 96, 255 regs, 1 warpgroup | the stream unit on the part's matrix op (`stream::mma_fmt` reading `Caps`), K / V windows staged by the copy engine (cp.async / TMA box) into shared memory, window geometry from the pipe rate (full-model.md open 1) | partly (if it extends `stream::mma_fmt`) |
| 4 | per-token host round trip: the full f32 logits (513 KB) copied out, argmax on the host, token copied in | all; largest on slow-sync hosts | every decode step | CUPTI: h100 245 µs idle + 14 µs D2H of a 2.02 ms step (12 %); a100 91 + 28 µs; rtx-4090 66 + 21 µs; gpt-fast fuses argmax into its head GEMV (`triton_red_fused__unsafe_view_argmax_mm…`) and never syncs per token | the argmax as a device row fold (max with index; `fold_combine` already proves the max law), a 4-byte copy-out; the loop's next trip waits only on its launch | no |
| 5 | `mma-walk` on Ampere does not realize its price | rtx-3090, a100 | 3090 suite row ×1.21 (walk 1 275 + 39 vs ordered 1 082 µs; priced ~845); a100 model down projection walk 788 vs ordered 761 µs (priced ~−40 %) | SASS row 1 of the regression table; the price (`MmaWalk::ns`) assumes the ordered tile's panel rate; the 5080 (JIT of the very 3090 PTX) realizes it (628 + 21 vs 667 µs) | the walk's loop = the ordered loop; the walk priced by its measured panel rate (a `DeviceCost` line per part: walked vs ordered panel ns, `mma_calib.cu`), replay lines `mma_walk_replay`; the A100's 2-round down projection then wants the cut or a smaller tile (`mma_pieces`) | no |
| 6 | decode launch count: horizontal merges refused on h100, single-block norm folds, tiny elementwise cells | h100 most | decode | h100 340 kernels a step vs 245 (a100 / rtx-4090) vs gpt-fast 200; `mapal.hz.host … refused=occupancy:31,waves:35` (h100) vs `waves:1` (a100): gate+up (1 024 blocks each on 1 056 slots) fails "the last waves fit one wave", so 64 GEMVs launch alone; 33 RMSNorm folds a step at grid 1 × 256 threads (1.9 µs each); 65 elementwise cells at 8 × 256 (1.6 µs) | price a merge by the launches it saves (`DeviceCost` launch / ramp ns against the members' tails), not waves alone (a mixed result: the waves rule's missing property); a row fold whose only reader is a GEMV joins it as its prologue (row-closed `unit_plan`) | no |
| 7 | f32 elementwise cells in prefill (norm, RoPE, SiLU·up, residual on f32 activations) | all | prefill | h100 2.99 vs 1.73 ms, a100 5.42 vs 2.80, rtx-4090 4.57 vs 2.52 at 2048 (gpt-fast: bf16 activations, norm fused into the next GEMM) | activations' storage format from their readers (`precision_plan.store` past multiplicand-only reads), cells into the GEMM epilogue (`tile_plan_post`) | no |
| 8 | TF32 wgmma tiles round operands in a register pass | h100 | suite tf32 rows 1.6–1.8× | Mapal: per trip LDS ×80 → F2FP.TF32.F32.PACK_B ×128 → STS.128 ×16 → HGMMA 64×64×8 ×16; cuBLAS `sm90_xmma_gemm_f32f32_tf32f32` HGMMA 64×128×8 straight from the TMA tile | round once at the producer (`precision_plan.round_at` / `NarrowStore` for the tf32 face: weights stored tf32-rounded once), then wgmma reads the TMA box | no |
| 9 | host round-trip cost (box property) at small programs | h100, a100 | the regression table's 26 Mapal-side rows | `lat.c` per box (below); a 0.04 ms program pays two 8-byte synchronous D2H (10.7 µs each on the Xeon host) | measured host round trip as a `TargetProfile` host property; one gathered copy-out per program | no |
| 10 | tuner `confirm` on a noisy box | a100 | `pj_m16_n18432_k7168`, `rt_mlp2_1024` | pairs 0.614 / 1.432 / 1.388 → revert | sign agreement beyond the box's null before a revert | no |

gpt-fast's own waste, for the record: its prefill computes the vocabulary head for every prompt position
(`ampere_bf16_s16816gemm_bf16_128x64` 1002 × 32 blocks: 7.58 ms of its 41.0 ms at 2048 on the 4090; `nvjet …320x128`
1.38 ms on the H100); Mapal computes it for the last row (0.18–0.58 ms).

## Per part — full model, bf16 class, against gpt-fast (this study's boxes; CUPTI, the last run of each process)

Untraced: `run_mapal.py` / `run_gptfast.py` medians (warmup 2, reps 3). CUPTI: `ktana.py` over `ktrace.cpp` (prompt pass
= the first launch after the weights to the logits copy; decode = median of 126 steps). Gaps = idle between kernels.

| part (box) | n | Mapal TTFT / decode ms | gpt-fast TTFT / decode | Mapal prompt: kernels, busy | gpt-fast prompt: kernels, busy | Mapal step: kernels, busy / span | gpt-fast step |
|---|---:|---|---|---|---|---|---|
| h100 (host 214845 = the round's, 700 W, 1 980 MHz) | 512 | 7.60 / 2.096 | 4.14 / 1.405 | 277, 7.06 | 185, 2.95 | 340, 1.72 / 2.02 | 200, 1.38 / 1.46 |
| h100 | 2048 | 24.91 / 2.219 (reps 2.15–2.98) | 10.82 / 1.509 | 277, 24.46 | 184, 9.79 | 340, 1.76 / 2.12 | 200, 1.37 / 1.48 |
| a100 (EPYC 7K62, 400 W, 1 410 MHz) | 512 | 16.47 / 3.926 | 8.97 / 2.492 | 292, 14.61 | 185, 8.31 | 245, 3.20 / 3.66 | 200, 2.53 / 2.56 |
| a100 | 2048 | 57.16 / 3.806 | 28.91 / 2.559 | 325, 51.02 | 184, 27.42 | 245, 3.28 / 3.73 | 200, 2.63 / 2.67 |
| rtx-4090 (Ryzen 5 9600X, **250 W cap**, 2 355 MHz) | 512 | 11.41 / 3.393 | 11.10 / 2.956 | 309, 11.19 | 185, 9.65 | 245, 3.13 / 3.39 | 200, 2.93 / 2.95 |
| rtx-4090 | 2048 | 40.51 / 3.487 | 46.70 / 3.035 | 293, 41.07 | 184, 41.05 | 245, 3.22 / 3.49 | 200, 3.02 / 3.05 |

Mapal's prompt pass is device-bound on every part (gaps ≤ 0.46 ms); gpt-fast's prefill (Python dispatch, no graph)
has 1.8–4.7 ms of host gaps under CUPTI (its untraced TTFT is close to its busy time). Both decode steps are
device-bound but Mapal's span adds the per-token host round trip (cause 4).

**Prompt pass by family (ms; Mapal | gpt-fast):**

| part, n | projections | attention | elementwise / norm / rope / silu | head |
|---|---|---|---|---|
| h100 512 | 5.32 (gate+up 1.94, down 1.82, q/o 1.03, k/v 0.53) \| 1.66 | 0.78 \| 0.27 | 0.99 \| 0.61 | one row \| 0.39 |
| h100 2048 | 14.63 (7.51, 3.86, 2.28, 0.98) \| 5.14 | 6.87 \| 1.51 | 2.99 \| 1.73 | one row \| 1.38 |
| a100 512 | 10.81 \| 5.96 | 1.58 \| 0.50 | 2.02 \| 0.74 | one row \| 1.06 |
| a100 2048 | 33.39 (12.61, **12.32 walked**, 5.61, 2.84) \| 17.67 | 11.96 \| 2.78 | 5.42 \| 2.80 | one row \| 4.12 |
| rtx-4090 512 | 9.18 \| 7.12 | 0.64 \| 0.36 | 1.16 \| 0.36 | one row \| 1.79 |
| rtx-4090 2048 | 30.38 (15.96, 8.14, 4.53, 1.78) \| 27.72 | 5.53 \| 3.20 | 4.57 \| 2.52 | one row (0.58) \| 7.58 |

**Decode step by kind (µs a step, n = 512):**

| part | GEMVs (Mapal: kinds) | attention | elementwise + norms | host round trip | gpt-fast (GEMV incl. fused norms / attention) |
|---|---|---|---|---|---|
| h100 | 1 328: 65 q/k/v/o/gate/up + head cells (LDG.U16) 859 incl. the head 175, down 297 (16 × 18.6), hz 172 | 198 | 184 (113 launches) + 61 (33 one-block norm folds) | 259 | 1 040 (121) / 218 (64) |
| a100 | 2 659: hz gate+up 869, q/k/v `calls` chain 692 (16 × 43 µs; first member 8 blocks), o + head 543, down 555 | 308 | 195 (82) + 44 (17 folds) | 119 | 2 036 / 369 |
| rtx-4090 | 2 855: hz 1 177, o + head 716, down 604, calls 358 | 164 | 89 (82) + 24 (17 folds) | 87 | 2 629 / 277 |

**Top kernels' SASS (h100, 2048):** Mapal gate+up `mapal_tile_cd5b7712ebb41ccc` 64×16 × 128 thr, regs 207, 110 KB smem:
HMMA 128, LDGSTS 48, LDSM 32 a trip (cause 1); down `…98791a20c3e79e2a` + residual epilogue: + FADD 128, LDG 128; attention
`…a5331067ffff0688` (flow, `stream=b64xkb32`): HMMA 32, FFMA 318, MUFU 72, LDG.U16 32, LDGSTS 4, BRA 78 (cause 3); decode
gate / up cell `…` 1024 × 256 thr: FFMA 16, LDG.U16 32 a trip (cause 2); head 16032 × 256: LDG.U16 16 (90 % of DRAM).
gpt-fast: `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` 2 × 64 × 384 thr, 213 KB smem; flex attention
`triton_tem_fused…flex_attention…` HGMMA 64×64×16 ×32 + 64×128×16, LDGSTS 96; decode GEMV
`triton_red_fused_mm_t_view_12`: FFMA 512, LDG.E.EF.128 ×64.

## H100 kernel suite — the worst bf16 / tf32 / GEMM rows against the cuBLAS Hopper kernels

Ratios from the five-GPU round (program boundary); Mapal's kernel = the shipped launch (`.launch`), SASS from the
bundle's PTX (`ptxas -arch=sm_90a`); the leader's from the f1d3a04f gap captures (`gaps/<row>/leader-1.sass`).

| row | ratio | Mapal kernel (traced ms) | SASS a trip | leader kernel | leader SASS | cause # |
|---|---:|---|---|---|---|---|
| `pj_m2048_n28672_k4096_bf16` | 2.43 | `mma` m16n8k16, 224 × 16 × 128 thr, bk 64, 3 stages, 2 blocks / SM (1.53) | HMMA 128, LDGSTS 48, LDSM 32, regs 241 | `nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_NNT` 2 × 66 × 384 | HGMMA 64×64×16 ×40, UTMALDG 12, UTMASTG 10, SYNCS 39, cluster 1×2 | 1 |
| `pj_m2048_n4096_k4096_bf16` | 2.25 | same geometry, 32 × 16 (0.24) | same | `nvjet` (same family) | — | 1 |
| `pj_m128_n4096_k14336_bf16` | 2.52 | `mma` mt 1 nt 2, warps 4 × 2, bk 256, 64 × 4 (0.165) | HMMA 64, LDGSTS 24, LDSM 64 | `nvjet_sm90_tst_96x128_64x6_2x1_v_bz_splitK_NNN` + `splitKreduce` | HGMMA, TMA | 1 |
| `pj_m128_n4096_k4096_bf16` | 2.13 | as above (0.057), **8 B spill** | HMMA 64, LDGSTS 24, LDSM 64 | `nvjet_sm90_tst_64x64_64x13_2x1_v_bz_NNT` 2 × 64 × 384 | HGMMA 64×64×16 ×8, UTMALDG 4, SYNCS 49 | 1 |
| `llm_pre_t2048_bf16` / `llm_blkpre_t2048_bf16` | 2.92 / 2.76 | projections as above + the stream attention | — | `nvjet` + flex / SDPA | — | 1, 3 |
| `pj_m128_n6144_k4096_tf32` | 1.82 | `wgmma` Tf32, 96 × 2 × 256 thr, bk 128, 3 stages (0.091) | HGMMA 64×64×8 ×16, UTMALDG 2, **LDS 80 → F2FP.TF32 128 → STS 16**, regs 255 | `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize64x128x32` | HGMMA 64×128×8 ×12, UTMALDG 8 | 8 |
| `pj_m128_n4096_k14336_tf32` | 1.83 | `wgmma` Tf32, bk 128, 4 stages (0.19) | HGMMA ×16, F2FP 96, LDS 72 | `sm90_xmma…tilesize128x128x32` | HGMMA 64×128×8 ×24, UTMALDG 12 | 8 |
| `pj_m2048_n4096_k4096_tf32` | 1.67 | `wgmma` Tf32, 32 × 8 × 384 thr, bk 32 (0.29) | HGMMA ×4, F2FP 80, LDS 32 | cuBLASLt tf32 | — | 8 |
| `pj_m1_n4096_k4096` (f32) | 1.44 | naive split S = 64, `combine=atomic` (0.034 traced) | FFMA 128, LDG.EF.128 32 + LDG 32, REDG 4 | `gemvx::kernel` 128 × 32 | LDG ×242, no split | 9 (two synchronous D2H at 10.7 µs on a 43 µs program) |

## Box conditions and host round trips (`lat.c`, medians of 2 000)

| box | GPU, power, busy clock | host | launch + sync µs | enqueue | 8-byte D2H sync | alloc+free 1 MiB | event sync | spend |
|---|---|---|---:|---:|---:|---:|---:|---:|
| h100 (offer 48704103, **host 214845 = the five-GPU round's**) | H100 SXM 80 GB, 700 W, 1 980 / 1 980 | Xeon Platinum 8568Y+ | 7.98 | 2.54 | 10.71 | 3.48 | 1.73 | $2.18 |
| a100 (57328780) | A100-SXM4-40GB, 400 W, 1 410 / 1 410 | EPYC 7K62 | 9.73 | 3.32 | 11.44 | 2.35 | 1.64 | $0.18 + $0.17 (walk arm) |
| rtx-4090 (48857502) | RTX 4090, **250 W** (SW power cap), 2 355 / 3 135 | Ryzen 5 9600X | 3.88 | 1.31 | 3.91 | 0.87 | 0.20 | $0.08 |
| rtx-3090 (35413600) | RTX 3090, 370 W, 1 935 / 2 115 | EPYC 7R32 | 5.89 | 2.69 | 6.56 | 2.29 | 0.56 | $0.03 |
| local rtx-5080 | RTX 5080 | Threadripper 9970X | 4.81 | 1.52 | 4.57 | 0.98 | 0.21 | — |

Spend: **$2.63** of the $12 cap (h100 $2.18, a100 $0.18 + $0.17, rtx-4090 $0.08, rtx-3090 $0.03). No box was replaced (FFMA burn ≥ 0.75 × max with no HW / thermal slowdown); the 4090's 250 W cap is noted where it
matters (its GEMM-heavy gpt-fast prefill). Every `mapal-da-*` instance destroyed and verified gone; nothing else
touched.

## For integration

- **Replay lines** (`crates/backends/llvm/src/rung_replay.tsv` / `mma_walk_replay`): rtx-3090
  `pt_pj_m2048_n4096_k4096_bf16` walk (164 CTAs × 400 iters) 1 275 µs + zero 39 µs vs ordered 1 082 µs → refuse;
  a100 2048 × 2048 × 8192 residual-epilogue walk (216 CTAs) 788 µs vs ordered 761 µs (same box) → refuse; rtx-5080 (JIT
  of the 3090 PTX) walk 628.5 + 21.2 vs 666.6 µs → walk. The rule must reproduce all three from properties.
- **Facts table** (CLAUDE.md), proposed rows: `cell_plan` narrow reads — a cell's array read whose element is the exact
  widening of a `bf16` / `f16` element, its storage bytes and its stride along the cell's fold axis (the vector width
  the emitter may load: cause 2); `TargetProfile` host round trip (launch + sync, 8-byte D2H sync, alloc + free, from
  `lat.c` / `calib.cu`) and the board power limit as measured box properties (cause 9; the regression table's box
  classes); `DeviceCost` walked-vs-ordered panel rate (cause 5).
- **Tailored fixes**: none (`docs/tailored-fixes.md` unchanged).

## Raw and tools

`.claude/raw-evidence/hopper/disasm-rented/` (not in git): `<part>/out.tgz` (box log, A/B lines, CUPTI TSVs, model
JSONs, gpt-fast modules and Triton cubins), `compile-path/` (`ecmp-<part>.txt`: plan diff + per-kernel ptxas / SASS mix
of both emissions; `regress.json`; the f1d3a04f matrix rebuilt under the current rule), `scripts/` (`dr_box.sh`,
`dr_rent.sh`, `lat.c`, `modcap.cpp`, `emitcmp.sh`, `kmix.py`, `ktana.py`, `ktstep.py`, `ktdec.py`, `abana.py`,
`classify.py`), `local/` (the 3090 PTX JIT on the RTX 5080).
