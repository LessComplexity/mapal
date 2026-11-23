# f32-tensor-cores — `{f32}` products on tensor cores by split forms; TF32 / 16-bit tiles cut along `k` (2026-11-23)

Branch `sprint/f32-tensor-cores` @ 55ba3e42 + this step. RTX 5080 (sm_120, 84 SMs, 99 KiB opt-in smem, 64 MiB L2),
CUDA 13, clang/LLVM 23. Design: `.claude/sprints/f32-tensor-cores/design.md`. Protocol: the shared bench lock per
timed run, no game in the foreground, CPUs 16–23,48–55, Tctl checked (runs paused above 85 C); ABBA (arms
interleaved, order reversed on even repetitions, 6–8 an arm) of only the programs whose emission an arm changes
(text compared per program: equal text = the same object, null by construction); value gates first (every arm
printed the interpreter's values within D2). Raw output (calibrations, ABBA logs, SASS mixes, torch legs):
`.claude/raw-evidence/local-rtx-5080/f32-tensor-cores/` (not in git).

**Result.** (1) The accumulator question the design raised is answered by measurement: every warp format keeps
**two alignment guard bits** (`g = 2`), so ADR-0041 D3's member bound (which needs `g ≥ 1`) is sound, and the
split realization's a-priori coefficient meets D2 from `K = 24` (tf32x3) / `16` (bf16 forms). (2) On this part the
TF32 `mma.sync` issues **256 flops an SM a clock — exactly its FFMA rate** — and the 16-bit products 512: the rate
rule refuses tf32x3 here (forced ON it measures 2.2–2.5× slower, the loser line) and takes **narrow1** (`{f32}` over
two bf16 / f16 operands): **2.41× / 2.14×** the FFMA tile, **0.40× / 0.47× the fastest same-face competitor**
(torch f32 over the widened operands). (3) **`mma-combine`**: TF32 and 16-bit tiles now cut `k` where the wave model
says the cut fills the part: 10 changed programs, **all faster** (0.77–0.985), mlp_2048_tf32 / mlp2_2048_tf32 now
**1.03 / 1.04×** the fastest TF32 leg (were 1.09 / 1.10). (4) `mma-depth` (deeper panels at the register-first
residency) lost every measured site: off, its ON arm measures it.

## S0 — measured properties (`benches/local/mma_calib.cu`, `benches/gpu/calib.cu matrix`)

| probe | TF32 m16n8k8 | bf16 / f16 m16n8k16 | e4m3 / e5m2 m16n8k32 |
|---|---|---|---|
| final rounding (`1 + 1.5u`, `1 − u/2`) | truncation (`1`, `1 − u`) | truncation | truncation |
| guard bits `g` (`c = 1` + `K` addends of `2^−(23+j)`; `g0_j`, `gneg_j` variants agree) | **2** (`j` 1, 2 kept, 3 dropped) | **2** | **2** |
| subnormal product / operand / `c` | kept / kept / kept | kept / kept / kept | kept / kept |
| random worst, units of `2⁻²⁴·M` an instruction | 4.08 | 4.63 / 5.27 (as before) | 4.00 / 4.36 |
| issue rate, flops an SM a clock (32 warps, register chains, clock64) | **256** (= FFMA 128 lanes × 2) | **512** / 512 | — |
| tf32x3 realization's pattern (3 products from zero + 4 FADD a tile) | 243 (95 % of plain) | — | — |

Split realizations of one k-step (`c + Σ` over random f32 operands, 2²⁰ instructions; the bound is `e + C₁ + 1`):

| form | uniform 2^±8 | wide 2^±30 | cancelling pairs | a-priori `e + C₁ + 1` at `g = 2` |
|---|---:|---:|---:|---:|
| tf32x3 from zero + RN add (shipped) | 9.57 | 9.22 | 5.60 | 19.5 |
| tf32x3 chained on `c` (ADR-0041 "Later") | 10.34 | 9.19 | 5.53 | — |
| bf16x6 from zero + RN add | 5.76 | 4.44 | 0.87 | 13.9 |
| bf16x6 chained on `c` | 9.68 | 9.12 | 0.81 | — |

FFMA under load: 61.2 Tflop/s peak, 57.0 sustained (2.65 GHz, cool box); TF32 59.9 / 59.0, bf16 125.6 / 125.6.

## What landed

| id | where | what |
|---|---|---|
| F1 `mapal_ir::split` | `crates/mapal-ir/src/split.rs` | `SplitKind` (tf32x3, bf16x6, bf16x3n, narrow1), `SplitForm { classes, e, exact_above }`, `AccModel { g }` (`c_g = 2((k+1)2⁻ᵍ + 1)`), `coef(K, S, j, k, acc) = (e + C_j)(1 + γ_n) + γ_n/u` (`n` = flushes + combines; `C₁` = `c_g·Σ_p Σ_{q≤p} ρ_q`, `C_j` = `c_g·jP·Σρ`), `admits` (D2 verbatim: `coef·u ≤ γ_{K+1}(u)`), `k_min`, `member_sound` (`g ≥ 1`); the split helpers (`tf32_parts` with §2.5's overflow select, `bf16_parts`). Pure numerics over stored facts (`TileSite.k`, `TileRead::narrow`, the product set): no new graph fact |
| P1 | `profile.rs` `Gpu::{mma_tf32_per_clk, mma_half_per_clk, wgmma_tf32_per_clk}`, `matrix_flops_per_clk(op)`, `matrix_flops_per_ns(op)`; `mma_rates_of_cc` (native fallback); `--cost=` / calibration names | the matrix products' flops an SM a clock: 5080 measured 256 / 512; 3090 / 4090 / 5090 datasheet 256 / 512; A100 1024 / 2048; H100 `mma.sync` 532 (realized lower bound), `wgmma` 2048 |
| P2 | `MatrixOp::acc_g` (`MMA_ACC_G = 2`), `acc_model()` | the accumulator's guard bits (measured, every format) |
| R-F `split-tile` | `func/gpu.rs` `split_operands`, `split_pays`; `rules.rs` `split_rate_pays`, `SPLIT_RATE_PM` | a default-face `{f32}` product takes a split form where (1) its coefficient meets D2 at `K` (always, also forced), (2) `matrix_flops_per_ns ≥ 1.3·P·ffma_flops_per_ns`, (3) the site's flops / operand byte reach the FFMA balance or its grid fills the schedulers. 5080: tf32x3 refused (256 vs 998), narrow1 taken (512 vs 333); A100: tf32x3 taken; H100 `mma.sync`: refused |
| emission | `Kernel::mma` (tf32x3), `Kernel::mma16` (narrow1), `split_tf32`, `mma_tf32_call`, `split_cold` | per k-step and m16n8 tile: products from a zeroed accumulator (`lo·hi`, `hi·lo`, `hi·hi`), one RN `fadd`; tf32x3's guard: an infinite operand or one below `exact_above` (2⁻¹¹⁵, its `lo` would round on the subnormal grid) ORs a flag, one `vote.sync.any` a warp after the loop, the warp's tile recomputed by FFMA from global memory (cold path); `hi` rounded to infinity keeps the truncation |
| geometry | `profile.rs` `gpu_mma_split_tile`, `gpu_mma_geom_parts` | a split tile counts each fragment's parts and a temporary accumulator a tile (ptxas issues a k-step's products before their adds: narrow1's 4 × 8 tile at 254 registers spilled 64 B), warps rounded to a power of two |
| R-T1 `mma-combine` | `func/gpu.rs` `mma_combine`, `mma_pieces`; `Kernel::mma` / `mma16` take a piece (`at_z` offsets, `split_seed`, `slab` / `store_o`) | a tensor tile (TF32, 16-bit, split) cut into `S` pieces (atomic; workspace where no atomic) minimizing `⌈blocks·S/slots⌉·(P/S + stages)` where it saves > 5 % — not where a ≥ 1-wave grid streams a site below twice the part's matrix balance; a split form's pieces re-admitted with `S − 1` combines |
| R-T2 `mma-depth` | `profile.rs` `mma_depth_pick`; `--geom=bk=`, `--geom=stages=` now reach the tensor tile | the panel keeping the most bytes in flight at the residency — **off** (lost every site) |
| emulator | `mapal_gpu_emu.c` `mma_sum` (`MAPAL_EMU_MMA_G`: aligned truncation with `g` guard bits), `emu_vote_any`; `tests/emu` | the measured datapath in the emulator (unset: the exact sum truncated once, as before) |
| S0 probes | `mma_calib.cu` (TF32 section, `g_j` / `g0_j` / `gneg_j`, subnormals, split realizations), `calib.cu matrix` | above |
| intents | `intent.py` `split-tile`, `mma-combine`, `mma-depth`; `evidence.py` counts VOTE | |

OFF arms (measurement flags, never per-part): `--gpu-off=split-tile` (the FFMA tile), `--gpu-on=split-tile` (the bound
alone, past the rate rule), `--gpu-off=mma-combine` (tensor tiles never cut), `--gpu-on=mma-depth`.

## Intent → realized (ptxas -v, `cuobjdump -sass`, sm_120)

| kernel | intent | realized |
|---|---|---|
| narrow1 2048 × 4096 × 4096 (bf16 × bf16, `{f32}`) | HMMA.16816 a tile and k16, its FADD flush, LDGSTS panels, LDSM fragments, no spill | 2 × 4 tile, bk 32, 3 stages: 16 HMMA.16816 + 64 FADD in the k loop (one HMMA, four FADD a tile and k16), 9 LDGSTS, 8 LDSM, 0 FFMA, 0 spill; cut S = 2: 32 RED |
| narrow1 128 × 4096 × 4096 | as above | 2 × 2, bk 64: 32 HMMA + 128 FADD, 24 LDSM, 0 spill |
| tf32x3 (forced) 2048 × 4096 × 4096 | 3 HMMA.1688 a tile and k8, FADD flush, VOTE once, FFMA only in the cold path, no spill | 48 HMMA.1688, 96 FADD, 1 VOTE, 128 FFMA (cold path), 12 LDGSTS, 0 spill (3 blocks / SM) |
| `mma-combine` TF32 / bf16 tiles (10 programs) | same HMMA / LDGSTS as the uncut kernel; RED = 4·mt·nt a thread; no spill | HMMA, LDGSTS, LDSM, LDS identical ON / OFF in every kernel; RED 128 (4 × 8), 32 (2 × 4), 16 (2 × 2); 0 spill |

## Results (ms, ABBA medians; best competitor from `matrix.md`'s rtx-5080 column unless marked)

| workload | OFF | ON | fastest competitor (leg) | ON / best | explanation |
|---|---:|---:|---|---:|---|
| narrow1 2048×4096×4096 (bf16 × bf16, `{f32}`) | 1.6673 | **0.6917** | 1.7178 (torch f32 over the widened operands, this box) | **0.40** | HMMA.16816 at 512 flops / SM clk vs FFMA 256; cut S = 2. The bf16 tensor-core legs (accumulation on the tensor core: a weaker bound) 0.5840 / 0.5884 → 1.18 |
| narrow1 128×4096×4096 | 0.1363 | **0.0637** | 0.1360 (torch f32, this box) | **0.47** | as above (bf16 TC leg 0.0532 → 1.20) |
| tf32x3 forced 2048×4096×4096 (f32) | 1.8377 | 4.1132 | 1.77 (torch-compile) | — | loser line: TF32 = FFMA rate, three passes; the rule refuses |
| tf32x3 forced 128×4096×4096 (f32) | 0.1487 | 0.3790 | 0.1352 (cublas-fresh) | — | as above |
| mlp_2048_tf32 | 0.8313 | **0.7850** | 0.7598 (torch-compile-ro-tf32) | 1.03 | first GEMM cut S = 8 (256 blocks → 13 waves of pieces); RED 128 |
| mlp2_2048_tf32 | 1.6673 | **1.5681** | 1.514 (torch-compile-ro-tf32) | 1.04 | as above |
| pj_m128_n6144_k4096_tf32 | 0.2387 | **0.1839** | 0.127 (lt-tf32-fresh) | 1.45 | 96 blocks on 168 slots → S = 8 |
| pj_m128_n4096_k14336_tf32 | 0.4618 | **0.4138** | 0.3113 (cublas-tf32-fresh) | 1.33 | 128 blocks, 448 panels → S = 8 |
| pj_m2048_n4096_k4096_tf32 | 1.4175 | **1.3405** | 1.161 (lt-tf32-fresh) | 1.15 | 512 blocks (3.05 waves) → S = 8 |
| pj_m2048_n4096_k14336_tf32 | 4.9008 | **4.3702** | 4.241 (lt-tf32-fresh) | 1.03 | as above |
| pj_m128_n6144_k4096_bf16 | 0.1181 | **0.1026** | 0.0881 (torch-bf16) | 1.16 | 16-bit tile cut |
| pj_m128_n4096_k14336_bf16 | 0.2486 | 0.2449 | 0.172 (torch-bf16) | 1.42 | null |
| pj_m2048_n4096_k4096_bf16 | 0.7649 | **0.7111** | 0.598 (torch-bf16-compile-cg) | 1.19 | |
| pj_m2048_n4096_k14336_bf16 | 2.4546 | **2.2806** | 2.089 (torch-bf16-compile-cg) | 1.09 | |

Programs the rules leave unchanged (text equal ON / OFF): every other suite program (`benches/local/suite.txt`, 33 of
35), pj m128 / m2048 n28672 (TF32, bf16), m2048 n6144, m128 n4096 k4096, convmc, the prefills. No suite program has two
narrow operands under `{f32}`, so `split-tile` changes no suite program on this part (the rule refuses tf32x3).

Measured but not shipped (the replay's loser lines): `mma-depth` (`bk` 16 × 5 stages vs 32 × 2: 128 × 4096 × 4096
0.143 / 0.135, 128 × 4096 × 14336 0.490 / 0.454, 128 × 28672 × 4096 0.722 / 0.696, 2048 × 4096 × 4096 1.494 /
1.418); cuts the wave model refuses (128 × 28672 × 4096 S = 2 / 4 / 8: 0.747 / 0.744 / 0.764 vs 0.696; 2048 × 6144 ×
4096 S = 8 1.009×; 128 × 4096 × 4096 S = 8 0.98× and 1.08× in two sessions).

## Tests

- `crates/mapal-ir/src/split.rs` (6): the K_min table pinned (g 0 / 1 / 3 the design's 40 / 32 / 24, 48 / 32 / 16;
  the measured g 2: tf32x3 24, bf16 forms 16), `E` at K 4096 (544 / 295 / 292), `coef` monotone in `K` and `S`,
  `member_sound` iff `g ≥ 1`, the bf16 three-way split exact over 2²⁰ values and every binade edge ≥ 2⁻¹¹⁰, the
  TF32 residual ≤ 2⁻²²|x|, the tf32x3 product error ≤ `e`.
- `crates/backends/llvm/tests/split_tile.rs` (4): the rule per part / face / `K` / OFF and ON arms (A100 takes
  tf32x3, every GeForce part and the H100 refuse; narrow1 by rule; below K_min refused even forced; exact and
  sequential faces never); `mma-combine` plans on 96 generated GEMMs (TF32 and bf16 members); `mma-combine` values:
  8 generated programs × forced S 2 / 4 × 4 emulator modes + OFF, equal to the interpreter (32 emulated runs);
  **64 generated `SplitProg`s** (`testgen::split_prog_strategy`: f32 / bf16 / f16 / mixed storage, seven value laws —
  rationals, 2^±40, cancelling pairs, TF32-exact integers, ±inf + NaN, the largest finite, operands near the least
  normal; `bt`, residual epilogues; `K` 8–512 straddling every K_min) in 4 emulator modes under the measured datapath
  (`MAPAL_EMU_MMA_G=2`), under `g = 0` where K meets its K_min, cut into two pieces, and under the OFF arm — every
  finite cell within `2γ_{K+1}(u)·M`, every non-finite cell the interpreter's class. Coverage (all non-zero):
  64 programs, 9 tf32x3 (ON arm) and 26 narrow1 kernels (5 by the 5080's rule), 9 tf32x3 refused by the rate
  rule, 1 below K_min, 28 FFMA fallbacks (f32 × bf16, `b` transposed under TF32, odd `k`), 23 cut into two pieces, 33
  under the guard-less datapath, 8 epilogues, 8 `bt`, 2 guard (inf / NaN), 3 underflow, 7 overflow-select, 7
  TF32-exact, 4 cancelling.
- `func::gpu` `mma_pieces_on_the_measured_cells` (the measured picks and refusals), `rules.rs` `rung_rules_replay`
  (six new lines: **122 picked / 68 null / 9 mixed / 18 open**), `half_tile.rs` (`{f32}` two-narrow now narrow1).
- **Real device**: the same 64 generated programs (`MAPAL_SPLIT_DUMP`) built for the 5080 under
  `--gpu-on=split-tile` and run on it: every program with a split kernel (32: 26 narrow1, 6 tf32x3 — the rest place
  their small sites on the host) printed the interpreter's values within `2γ_{K+1}(u)·M` (worst 0.011 of the
  tolerance), **32 / 32 PASS** (`device_check.sh`, raw `device_check*.txt`). (`mapal test --weakest` fixes the order,
  which turns the split off: not usable for this check.)
- Workspace subset run (`cargo test --release -p mapal-backend-llvm -p mapal-ir`): **696 passed, 0 failed** (54
  targets; `streams.rs`' pre-existing `pdl_is_hopper_and_later_only` passed too). `c0-rules.tsv` stays: its pinned
  flags gain the two new OFF arms (the only moved row, the rtx-3090's mlp_2048_tf32, is `mma-combine`'s cut).

## Open problems attacked

- **Is the member bound sound on a truncating accumulator?** (design §2.2) — **closed**: `g = 2` measured on every
  format; `AccModel::member_sound` pins the condition, `MatrixOp::acc_g` carries it.
- **`{f32}` over two narrow operands never took a tensor core** (`half_tile.rs`) — **closed** by narrow1 (2.1–2.4×).
- **Tensor tiles never split** (`large-gemm.md` §6) — **closed** for the uniform split-K pieces (`mma-combine`, both
  kernels); stream-K over a tensor tile (a runtime trip count in the `cp.async` prologue) stays open — the wave model
  found the uniform cut enough on every measured winner.
- **Deeper TF32 pipelines** (design R-T2) — **attempted, open**: more stages at the register-first residency lost
  everywhere (the fact the rule lacks is the k-step's barrier + wait cost against the second block's cover); the
  depth-first 128 × 128 × 32 × 3-stage geometry needs XOR-swizzled panels (96 KiB) — not attempted.
- **narrow1's 4 × 8 tile** was 7 % faster than the spill-free 2 × 4 uncut (0.7317 vs 0.7882 ms) with a 64 B address
  spill in the loop — **closed**: the register estimate counts the chain temporaries (one accumulator a tile: ptxas
  issues a k-step's products before their adds), and with `mma-combine`'s cut the spill-free 2 × 4 tile is the fastest
  measured (ABBA, 2048 × 4096 × 4096: 2 × 4 cut S = 2 **0.6913**, 4 × 4 at 208 registers 0.7144, 4 × 8 spilling
  0.7308, FFMA 1.6750 ms).
- **The split's absolute term** (design §2.5 underflow) — **closed without weakening D2**: the guard sends operands
  below `exact_above` to the FFMA path instead of adding `abs·Σ(|a|+|b|)` to the gate.
- Not attempted: bf16x6 / bf16x3n emission (the A100's and H100's candidates; F1 carries their numerics), the `wgmma`
  split (S3, after the hopper design's R1 / R6), F3 radix contiguity / `tf32-radix` re-measure, `j > 1` flushes, the
  H100 `mma.sync` rate measurement and the A100 winner line (`calib.cu matrix`, the forced / ruled arms at the next
  rented round: the rate rule's θ = 1.3 has the 5080's loser and winner lines only).

## Commands

`nvcc -O3 -arch=sm_120 benches/local/mma_calib.cu`; `calib ffma matrix`; arms by `emit … --gpu-off=<rung>` /
`--gpu-on=split-tile` / `--splitk=S` / `--geom=bk=,stages=`, `opt -O3`, `llc -mcpu=sm_120 -nvptx-fma-level=0`,
linked with `mapal_gpu.c`; ABBA under the bench lock; `ptxas -v` + `cuobjdump -sass`; torch legs with
`venv-torch-cu130` (median of 30 after 10 warm, CUDA events).

## For integration

README line (`benches/results/evidence/local-rtx-5080/README.md`):

**f32-tensor-cores** (`f32-tensor-cores.md`): the warp accumulator keeps two guard bits on every format (the member
bound is sound) and the 5080's TF32 rate equals its FFMA rate (256 flops / SM clk; 16-bit 512) — `mapal_ir::split`
(split forms' a-priori coefficient against D2), `split-tile` (narrow1: `{f32}` over two bf16 operands 2.1–2.4× the
FFMA tile, 0.40–0.47× torch f32; tf32x3 taken by the rate rule on the A100 only), `mma-combine` (TF32 / 16-bit tiles
cut along `k` by a wave model: 10 programs 0.77–0.985, mlp_2048_tf32 1.09 → 1.03× the best TF32 leg),
`mma-depth` measured losing (off).

Calibration line (`calib/rtx-5080.calib`, measured by `calib.cu matrix`; the profile already carries the same
values): append `mma_tf32_per_clk=256 mma_half_per_clk=512`.

CLAUDE.md facts-table rows:

| query | what it proves |
| --- | --- |
| `mapal_ir::split` (`SplitForm::{coef, admits, k_min}`, `AccModel`) | a split realization of an `{f32}` product on a matrix instruction (tf32x3, bf16x6, bf16x3n, narrow1: parts, kept products smallest class first, each k-step from a zeroed accumulator then one round-to-nearest add): its a-priori coefficient `(e + C_j)(1+γ_n) + γ_n/u` against D2's `γ_{K+1}(u)` at the site's `K`, `S` pieces and the accumulator's measured guard bits; the operand range below which a part leaves the normal grid (`exact_above`: the guard's FFMA path) |
| backend: `Gpu::{mma_tf32_per_clk, mma_half_per_clk, wgmma_tf32_per_clk}`, `matrix_flops_per_ns(op)`, `MatrixOp::acc_g`, `rules::split_rate_pays`, `mma_pieces` | a matrix product's flops an SM a clock (`calib.cu matrix`) against the FFMA rate (a split form of `P` passes pays at `1.3·P`); the accumulator's guard bits (`mma_calib.cu`); a tensor tile's `k` cut by its waves on the slots it holds (`⌈blocks·S/slots⌉·(P/S + stages)`) |
