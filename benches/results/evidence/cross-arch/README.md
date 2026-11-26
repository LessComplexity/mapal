# cross-arch — the property rules and the suite on four rented parts (2026-11-21)

Branch `local-rtx-5080` @ `8bba2a50` (kernel merging, hardware properties, model-scale suite, long rows, onto device) +
this round's tooling (`f4e6e270` vast wrappers, `3968f59c` / `bd9ae103` / `73360fbc` calibration on the box,
`72ee9297` per-instance ssh key). Four vast.ai boxes in parallel, CUDA 13.0.2 devel image, torch 2.11.0+cu130.

Cross-part matrix (120 rows + 7 fp8-weight rows, every competitor leg at the program boundary, five parts with the
RTX 5080, the rented four on `b8429388`), Mapal vs the fastest same-precision competitor: [matrix.md](../matrix.md);
the latest round, with the full model on every part:
[b8429388 / 9be9886d (five-GPU round)](#b8429388--9be9886d-five-gpu-round-bf16-integration-every-competitor-leg-at-the-program-boundary-the-full-model).

**Bundle, one per part** (`integrated_bundle.sh`, built locally, `LADDER=1`): the README GPU suite (35 programs and
the two GEMM cells of `benches/local/gpu_bundle.sh`); the model-scale suite — the 12 natural Llama-layer programs
(decode B 1/8/32 × L 2048/8192, block decode, prefill 2048 / 8192, block prefill, TF32 twins) and 24 projections
(m 1/16/128/2048 × QKV / O / gate-up / down, TF32 at m ≥ 128), each answering to its exact face built for the device
(`XREF`), plus the five `px_llm_*` interpreter differentials; and the **flipped arm of every rung the rules ship
differently on that part than on the rtx-5080** (`emit --rungs` per part, diffed): rtx-3090 grid-share / panel-regs /
warp-fill / pinned ON (splitk ON on the cells' exact face); a100 warp-fill / min-blocks / pinned ON (fill-rows, unroll
ON exact); rtx-4090 async / kmajor-a / wide-fma OFF, pinned ON; h100 row-split OFF, min-blocks / warp-fill / pdl /
pinned ON (fill-rows ON exact). An arm whose emission equals the shipped build is dropped — every `pinned` arm (no
program of the suite copies a host array inside a loop). Built arms: 152 / 87 / 75 / 185.

**On each box**: `calib.cu` (every kind once, `gsync` and `copy` three times; new `facts` line and `ffma` kind), then
`box_run.sh` REPS=30 — value gates before any timing (differentials, every arm and site candidate of every probe, the
cells' split programs), probes ABBA × 5 with a traced process, the tune pass, `gemm_lib` cuBLAS / cuBLASLt (flushed and
`-fresh`), `chain_bench`, torch eager / compile (max-autotune-no-cudagraphs) / SDPA / TF32 / bf16 reference
(`TORCH_MIN=50`), the cells vs cuBLAS / CUTLASS stock / tuned; `evidence.py check`; `box_profile.sh` (ptxas, SASS,
nsys of the library kernels). `LADDER=1` times every program (none `carried`); the 156 ladder steps themselves were not
bundled (they do not fit the 3.5 h cap next to the model-scale suite).

## Boxes and cost

| part | GPU (driver, power limit) | host CPU | offer | $/h | hours | cost | status |
|---|---|---|---|---:|---:|---:|---|
| rtx-3090 | RTX 3090 24 GB (595.99, 350 W) | EPYC 7232P 8c | 31458820 | 0.214 | 1.61 | $0.35 | done |
| (3090 tries) | — | — | 59858132 / 61766354 / 61710265 | 0.21–0.25 | 0.35 | $0.09 | image pull never finished; CDI device error; created stopped |
| a100 | A100-SXM4-40GB (610.57, **270 W**) | EPYC 7B12 | 61533544 | 0.438 | 1.39 | $0.61 | done |
| rtx-4090 | RTX 4090 24 GB (610.57, **250 W**) | Ryzen 9 5950X | 59321201 | 0.417 | 1.19 | $0.50 | done |
| h100 | H100 80GB HBM3 SXM (615.71, 700 W) | EPYC 9684X | 56442968 | 2.55 | 1.55 | $3.96 | done |
| (h100 try) | — | — | 59816186 | 2.55 | 0.04 | $0.11 | port mapping error |
| **total** | | | | | | **$5.60** | every `mapal-xarch-*` instance destroyed (checked by id) |

$/h is the instance's `dph_total` (60 GB disk included); hours from creation to the destroy check. The account's credit
fell $21.08 over the round; another project's B200 billed through the same window.

## Value gates — all pass

| part | differential (px_llm ×4 builds, candidates) | probe gates (programs / arms·candidates within D2/D3) | device-exact oracles | split programs | gemm_lib legs within | torch legs within |
|---|---|---|---:|---|---:|---:|
| rtx-3090 | 23 / 23 | 71 / 71, 634 | 60 | 8 / 8 | 112 / 112 | 91 / 93 |
| a100 | 30 / 30 | 71 / 71, 652 | 60 | 8 / 8 | 112 / 112 | 220 / 234 |
| rtx-4090 | 32 / 32 | 71 / 71, 656 | 60 | 11 / 11 | 112 / 112 | 219 / 234 |
| h100 | 30 / 30 | 71 / 71, 752 | 60 | 8 / 8 | 112 / 112 | 220 / 234 |

No Mapal value failed on any part, arm, face or site candidate. Every torch leg outside its tolerance is a
cross-precision leg (a TF32 mode on an f32 program or the reverse, and Triton's TF32 against Mapal's TF32 on the MLPs:
the f32 tolerance 1e-3, as on the 5080) or `pt_attn_1024_64`'s SDPA leg (maxdev 1.951e-3 identically on every part:
the leg's formulation, eager / compile within 2–4e-7). One harness defect: `box_run.sh` ran an arm's `.pick<i>.launch`
(a tile_bench leg) as a program — rc 127, a false `SPLITK PROGRAM FAILED` on the 4090's OFF arms; fixed (`73360fbc`).

## Properties: the profile against the box

`calib.cu` on the box against `profile.rs` (`RTX_3090`, `A100`, `RTX_4090`, `H100`); the clock is the FFMA peak under
load over `sms · fp32_lanes · 2`. Device-query facts (SMs, L2, shared memory, registers, threads / blocks per SM, memory
clock and bus) equal the profiles on all four parts.

| property (rule) | rtx-3090 profile → box | a100 | rtx-4090 | h100 |
|---|---|---|---|---|
| SM clock under FFMA load, kHz | 1 695 000 → **1 561 273** | 1 410 000 → 1 408 892 | 2 520 000 → **1 955 131** | 1 980 000 → **1 807 192** |
| FFMA flop/ns | 35 581 → 32 774 | 19 491 → 19 477 | 82 575 → 64 066 | 66 908 → 61 069 |
| FFMA per DRAM byte, ‰ (`BALANCE_PM` 70 000, `FILL_BALANCE_PM` 45 000) | 38 013 → 35 014 | 12 534 → 12 524 | **81 919 → 63 556** | 19 960 → 18 218 |
| DRAM / µs / SM (`DRAM_PER_SM` 14 600) | 11 414 | 14 398 (1.4 % under) | 7 875 | 25 393 |
| regs / resident thread (`REGS` 32), L2 / SM (`L2_PER_SM` 128 Ki) | 42, 75 Ki | 32, 379 Ki | 42, 576 Ki | 32, 388 Ki |
| DRAM read (datasheet → `calib bw`), GB/s | 936 → 891 | 1555 → 1488 | 1008 → 958 | 3352 → 3186 |
| dependent launch / PDL / cooperative, ns | 0 → 1472 / — / 2866 | 0 → 2050 / — / 4714 | 0 → 1440 / — / 2660 | 0 → 1722 / **734** / 1839 |
| grid barrier at ≤ sms blocks, ns | 514 | 1113 | 502 | 910 |
| pageable / pinned H2D, ‰ (`PAGEABLE_PM` 900) | 0 → 948 | 0 → **760** | 0 → 961 | 0 → **841** |

**Deltas that flip a rule** (`rules.rs` `rungs`, replayed on the measured values; the replay reproduces `emit --rungs`
for all five profiles, 0 mismatches):

| part | rule | threshold | profile → measured | rungs that flip | ON / OFF timing on the box |
|---|---|---|---|---|---|
| rtx-4090 | `wide-fma`, `async`, `kmajor-a` (`BALANCE_PM`) | 70 000 ‰ | 81 919 → 63 556 (the box's 250 W cap: 1.96 GHz) | all three ON → OFF at every fill | the shipped ON **wins**: wide-fma 17 / 26 programs, kmajor-a 10 / 26, async 3 / 20, 0 losses — reading the measured clock would lose |
| h100 | `pdl` (`pdl_launch_ns < dep_launch_ns`) | — | unmeasured → 734 < 1722 | OFF → ON everywhere | mixed: ON wins 6 (gen / causal / prefill_1024 −20 %, decode_1024 −8 %, pj_m1_n2048, decode B1 −3 %), loses 8 (smlp +49 %, indep_512 +29 %, rt_mlp2_1024 +19 %, block prefill TF32 +12 %, block decode +10 %), 57 null |
| h100, a100 | `pinned` (`PAGEABLE_PM`) | 900 ‰ | unmeasured → 841 / 760 | OFF → ON | no effect on this suite (every `pinned` arm emits the shipped build) |

No flip on the rtx-3090 (35.0 against `FILL_BALANCE_PM` 45). Near a threshold: the a100's 14 398 B/µs/SM is 1.4 %
under `DRAM_PER_SM` (an 80 GB A100's HBM2e at 1593 MHz is 18 880: `row-split` ON).

Native-gpu calibration lines (`DeviceQuery::calib_key` + measured rates):
```
cc=86 sms=82 l2=6291456 mem=9751000x384	dram_bytes_per_ns=891 l2_bytes_per_ns=2447 gsync_ns=514 dep_launch_ns=1472 coop_launch_ns=2866 pageable_pm=948 fp32_lanes=128 clock_khz=1561273
cc=80 sms=108 l2=41943040 mem=1215000x5120	dram_bytes_per_ns=1488 l2_bytes_per_ns=2793 gsync_ns=1113 dep_launch_ns=2050 coop_launch_ns=4714 pageable_pm=760 fp32_lanes=64 clock_khz=1408892
cc=89 sms=128 l2=75497472 mem=10501000x384	dram_bytes_per_ns=958 l2_bytes_per_ns=4300 gsync_ns=502 dep_launch_ns=1440 coop_launch_ns=2660 pageable_pm=961 fp32_lanes=128 clock_khz=1955131
cc=90 sms=132 l2=52428800 mem=2619000x5120	dram_bytes_per_ns=3186 l2_bytes_per_ns=3990 gsync_ns=910 dep_launch_ns=1722 pdl_launch_ns=734 coop_launch_ns=1839 pageable_pm=841 fp32_lanes=128 clock_khz=1807192
```
(`calib bw` at 1024-thread blocks with L = 8 never launched on these parts — registers — and printed 2²⁰ GB/s; dropped
here, `calib.cu` now prints `launch_failed` there, `73360fbc`.)

## Rules: shipped decision against its flipped arm

Median `iter ms` (5 processes, ABBA) of the flipped arm over the shipped build's **rules pick** (candidate 0: a
flipped arm's site has no wisdom, so it runs its own candidate 0; for an untabled program, the shipped build); beyond
max(null, 2 %) counts. The arm forces the rung at every site, so the fill of the sites it changed is given (`wps=` of
`; mapal.plan`).

| part | flipped arm (part-specific decision) | programs | rule wins | null | rule loses | geomean flipped / shipped |
|---|---|---:|---:|---:|---:|---:|
| rtx-3090 | on-grid-share (off: L2 / SM < 128 Ki) | 53 | 12 | 35 | 6 | 1.045 |
| rtx-3090 | on-panel-regs (same) | 51 | 1 | 47 | 3 | 1.005 |
| rtx-3090 | on-warp-fill (off at wps 1: balance < 45) | 40 | 5 | 29 | 6 | 0.990 |
| a100 | on-min-blocks (off filled: 32 regs / thread) | 43 | 2 | 35 | 6 | 1.007 |
| a100 | on-warp-fill | 38 | 5 | 24 | 9 | 0.960 |
| rtx-4090 | off-async / off-kmajor-a / off-wide-fma | 20 / 26 / 26 | 3 / 10 / 17 | 17 / 16 / 9 | 0 / 0 / 0 | 1.013 / 1.032 / 1.032 |
| h100 | off-row-split (on: 25.4 KB/µs/SM ≥ 14.6) | 4 | 0 | 0 | **4** | **0.851** |
| h100 | on-min-blocks (off: 32 regs, 228 KiB smem) | 66 | 7 | 41 | **18** | 0.987 |
| h100 | on-pdl (off: unmeasured) | 71 | 8 | 57 | 6 | 1.005 |
| h100 | on-warp-fill | 37 | 6 | 26 | 5 | 1.049 |

**Rules that lose beyond null — the missing property each points at (none is a per-part switch):**

1. **`row-split` on the H100 SXM: 4 / 4 lose** — decode_1024_256 0.811, gen 0.860, causal 0.866, prefill 0.869.
   Explained (trace): the split adds two launches a trip — decode 6144 launches against 4098 OFF, `kernel_ms` 51.1
   against 38.3 — on rows of 256–512 floats a step; the rule reads DRAM bytes per µs per SM only. Missing: the row
   pass's own time against the launches the split adds (`dep_launch_ns` 1722 here). The H100 PCIe's 1.41–1.56× wins
   (the rule's citation) were long single passes.
2. **`min-blocks` off on the H100 SXM: 18 lose, 7 win** — block decode 0.696 (its changed sites are under-filled,
   wps 0: the `SMEM_PER_SM_MANY_BLOCKS` branch the H100 PCIe set), 17 more programs 0.946–0.979 at filled sites (epi,
   loop, pj m2048, mlp, softmax, rowsum …); it wins on m1 / m16 / m128 projections and mlp_512 (1.02–1.10). The evidence check flags 1107
   `face-registers` "derived blocks/SM resident at ptxas registers + smem" mismatches on this part, and block decode's
   shipped kernels spill 4.3 KB (`ptxas.spill_st`). Open: the SXM and the PCIe H100 share every per-SM fact; what
   separates them (SMs, clock, HBM rate) is not yet measured against this rung.
3. **`warp-fill` off at one warp per scheduler (balance < 45 ‰): mixed on all three low-balance parts.** At wps 1 it
   loses on attn (3090 0.675, a100 0.812), convmc (a100 0.73, h100 0.949), pj_m16_n2048 (h100 0.913) and wins on
   loopmm (1.05–1.17), rt_mlp2 (1.11), pj_m128_n6144 (1.39), pj_m1_n28672 (h100 5.83). The arm also changed filled
   sites, where every part (the 5080 too) ships it off: the m16 projections lose there on all three (a100 0.61–0.77,
   h100 0.71–0.82, 3090 0.86–0.93) and decode B32 / B8 (0.83–0.95), while others win (h100 m16 n18432 1.40, n4096 ×
   k14336 1.16 — the a100's 0.61 —, a100 m16 n28672 1.12).
   Missing fact, open: the balance does not separate these; the m16 projections' fitted grids against the SMs
   (wave quantization) are the candidate.
4. **`min-blocks` off on the A100 (filled): 6 lose by 3–5 %** (the m16 projections, pj_m2048_n4096_k14336), 2 win.
5. **`grid-share` / `panel-regs` off on the 3090 (75 KiB L2 per SM): net wins** (grid-share 12 W / 6 L, geomean
   1.045); the losses are decode B8 (0.86 / 0.95), decode_1024 (0.864), convmc (0.87 / 0.91). Open.
6. **`BALANCE_PM` against a measured clock** (properties table): the 4090 at 63.6 ‰ still wins all three rungs; the
   threshold sits between the 5090's 58.5 (its E1 clock under load) and 63.6, not at 70, once the clock is the
   measured one. `rung_replay.tsv` gains a measurement; the rule is unchanged here.
7. **`pdl` measured ON on the H100: mixed.** It wins the long dependent chains (the 1024-step loops at wps 0) and loses
   where launches were independent or few (smlp, indep_512, rt_mlp2_1024). Missing fact, open: whether a launch's
   predecessor is its producer (a PDL wait on an independent kernel serializes it).

**Tune pass (AUTO) against rules (candidate 0)**: AUTO faster beyond null on 17 / 51 (3090), 19 / 53 (a100), 6 / 53
(4090), 40 / 54 (h100) tabled programs; slower on 2 / 6 / 6 / 2 — on the two power-capped boxes the trials pick a
candidate that loses in steady state (a100 rt_mlp2_2048 1.090, rowsum_4096 1.074, loop_4096 1.073; 4090 block decode
1.110). Open: the trial runs at a clock the sustained run does not keep.

## Performance — Mapal / best competitor

Best competitor: the fastest leg within its gate — f32 legs for an f32 program, TF32 for `_tf32` (cuBLAS / cuBLASLt
flushed `cuBLAS`, `Lt` and fresh-operand `-f`, torch eager `te` / compile `tc`, SDPA, `chain_bench` streams / fused /
cache …); bf16 is never like-for-like. Decode rows per step. Null = |null / on − 1| of the shipped build.

| suite | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|
| README GPU suite, geomean (programs, wins) | 0.89 (24, 10) | 0.84 (31, 19) | 0.68 (31, 25) | 1.03 (31, 7) |
| model-scale layer, geomean (12) | 5.11 | 4.73 | 3.18 | 5.69 |
| model-scale projections, geomean (28) | 1.21 (2 wins) | 1.15 (8) | 1.12 (6) | 1.46 (0) |
| GEMM cells 16×2048×7168 / 4096³ | 1.08 / 1.11 | 1.73 / 1.10 | 1.14 / 1.05 | 1.86 / 1.17 |

**Caveats.** The 4090 (250 W) and A100 (270 W) are power-capped: a library leg's 30 back-to-back repetitions throttle
(4090 cuBLAS at m 2048 × 28672 × 4096: min 8.36, median 14.77 ms) while a Mapal process is a short burst, so their
large-GEMM "wins" (4090 pj_m2048 0.58–0.64, a100 0.77–0.93) are not like-for-like; the a100's prefill rows have 25–63 %
null (iter ms 139–420 ms across processes of one binary). The 3090's torch legs hit `TORCH_MIN` at the 8192 prefill
(`torch.compile` autotune OOM retries): seven README programs have no competitor there (—).

### README GPU suite

| workload | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---:|---:|---:|---:|
| attn_1024_64 | 1.85 (fused, 7.2%) | 1.22 (fused, 4.0%) | 1.31 (tc, 0.9%) | 1.55 (tc, 22.9%) |
| causal_1024_256 | 0.16 (tc, 25.4%) | 0.16 (tc, 2.1%) | 0.11 (tc, 0.1%) | 0.31 (tc, 0.0%) |
| conv_2048_5 | 0.71 (tc, 0.1%) | 0.56 (te, 1.8%) | 0.55 (te, 0.6%) | 0.47 (te, 0.4%) |
| convmc_8_64_56_3 | 1.97 (te, 2.7%) | 2.24 (te, 2.8%) | 1.74 (te, 0.1%) | 1.61 (te, 0.1%) |
| convmc_8_64_56_3_tf32 | 1.77 (te, 1.7%) | 3.90 (te32, 1.2%) | 2.06 (te32, 0.0%) | 3.45 (te32, 5.1%) |
| decode_1024_256 | 0.83 (cache, 11.2%) | 0.49 (nocache, 0.1%) | 0.60 (cache, 0.6%) | 1.33 (cache, 0.9%) |
| diamond_2048 | 1.03 (streams, 0.4%) | 0.92 (Lt, 0.1%) | 0.94 (tc, 0.6%) | 1.14 (tc, 0.0%) |
| epi_4096 | 0.96 (Lt, 0.0%) | 0.87 (Lt, 0.0%) | 0.58 (te, 0.2%) | 1.10 (Lt, 0.5%) |
| ew_4096 | 0.62 (tc, 2.7%) | 0.98 (tc, 0.4%) | 0.36 (tc, 0.3%) | 1.37 (tc, 0.5%) |
| ewconvew_2048_5 | 0.50 (tc, 0.1%) | 0.46 (te, 0.2%) | 0.47 (tc, 0.2%) | 0.49 (tc, 0.1%) |
| fanout_2048 | 1.04 (streams, 0.1%) | 0.94 (streams, 0.0%) | 0.94 (streams, 0.5%) | 1.14 (streams, 0.1%) |
| fir_4194304_16 | 0.66 (tc, 1.1%) | 0.49 (te, 3.8%) | 0.56 (tc, 0.4%) | 0.54 (te, 0.1%) |
| firew_rowsum_2048x2048_16 | 0.74 (tc, 1.2%) | 0.69 (tc, 0.3%) | 0.61 (tc, 0.2%) | 0.58 (tc, 0.9%) |
| gen_1024_512_256 | 0.15 (tc, 12.5%) | 0.15 (tc, 1.3%) | 0.11 (tc, 0.5%) | 0.30 (tc, 0.4%) |
| indep_2048 | 1.05 (streams, 0.2%) | 0.98 (tc, 0.1%) | 0.95 (tc, 0.2%) | 1.14 (Lt, 0.1%) |
| indep_512 | 1.80 (streams, 4.1%) | 1.46 (streams, 6.8%) | 1.08 (Lt, 2.0%) | 1.74 (streams, 0.3%) |
| loop_4096 | 0.97 (Lt, 0.4%) | 1.38 (te, 6.3%) | 0.63 (tc, 0.0%) | 1.09 (Lt, 0.0%) |
| loopmm_1024_8 | — | 1.16 (te, 0.1%) | 1.15 (tc, 0.1%) | 1.14 (tc, 0.2%) |
| mlp2_2048 | 1.09 (lt2, 0.1%) | 0.92 (te, 0.2%) | 0.90 (tc, 0.1%) | 1.10 (lt2, 0.0%) |
| mlp2_2048_tf32 | — | 1.01 (tc32, 0.2%) | 0.78 (te32, 0.4%) | 2.43 (tc32, 0.2%) |
| mlp_2048 | 1.04 (Lt, 0.1%) | 0.99 (tc, 0.2%) | 0.92 (tc, 0.2%) | 1.11 (Lt, 0.0%) |
| mlp_2048_tf32 | — | 1.24 (te32, 0.7%) | 0.80 (te32, 0.0%) | 2.38 (te32, 0.0%) |
| mlp_512 | 1.39 (Lt, 25.0%) | 1.28 (Lt, 0.4%) | 1.15 (Lt, 5.6%) | 1.57 (Lt, 0.2%) |
| prefill_1024_512_256 | — | 0.16 (tc, 2.2%) | 0.11 (tc, 0.0%) | 0.29 (tc, 0.1%) |
| rowsum_2048 | 1.10 (gemv, 0.1%) | 0.86 (eager, 0.0%) | 0.95 (tc, 0.0%) | 1.16 (eager, 0.1%) |
| rowsum_4096 | 1.06 (eager, 0.0%) | 0.94 (te, 0.1%) | 0.66 (gemv, 0.1%) | 1.08 (eager, 0.4%) |
| rt_mlp2_1024 | — | 1.23 (tc, 0.2%) | 0.77 (tc, 0.0%) | 1.01 (tc, 0.1%) |
| rt_mlp2_2048 | — | 1.11 (tc, 8.9%) | 0.57 (tc, 0.0%) | 1.05 (tc, 0.0%) |
| smlp_4096_256_1024 | 1.12 (cutlass, 1.1%) | 1.21 (eager, 0.3%) | 0.98 (tc, 0.1%) | 1.17 (Lt, 0.3%) |
| softmax_rows_4096 | 1.04 (fused, 0.2%) | 0.86 (tc, 0.0%) | 0.99 (tc, 0.2%) | 1.09 (tc, 0.3%) |
| twin_2048 | — | 0.92 (tc, 0.0%) | 0.95 (tc, 0.2%) | 1.13 (tc, 0.1%) |

### Model scale (the layer; decode per step)

| workload | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---:|---:|---:|---:|
| decode B1 L2048 | 2.06 (tc) | 1.36 (tc) | 1.20 (tc) | 2.03 (tc) |
| decode B1 L8192 | 3.13 (tc) | 1.80 (tc) | 1.60 (tc) | 2.18 (tc) |
| decode B8 L2048 | 2.05 (tc) | 2.29 (tc) | 1.36 (tc) | 3.04 (tc) |
| decode B8 L8192 | 1.55 (tc) | 2.41 (tc) | 1.19 (tc) | 2.72 (tc) |
| decode B32 L2048 | 1.91 (tc) | 2.02 (tc) | 1.25 (tc) | 2.19 (tc) |
| decode B32 L8192 | 1.46 (tc) | 2.43 (tc) | 1.10 (tc) | 2.03 (tc) |
| block decode B1 L4096 | 2.01 (tc) | 1.92 (tc) | 1.62 (tc) | 2.66 (tc) |
| prefill T2048 | 17.5 (tc) | 8.01 (tc, 32 % null) | 7.85 (tc) | 11.5 (tc) |
| prefill T2048 TF32 | 24.1 (tc32) | 24.9 (tc32, 63 % null) | 13.2 (tc32) | 40.9 (tc32) |
| prefill T8192 | 15.6 (tc) | 18.8 (tc, 26 % null) | 8.96 (tc) | 18.3 (tc) |
| block prefill T2048 | 17.2 (tc) | 7.64 (tc) | 9.33 (tc) | 7.73 (tc) |
| block prefill T2048 TF32 | 24.6 (tc32) | 34.2 (tc32) | 17.5 (tc32) | 40.2 (tc32) |

Null ≤ 2.1 % except as marked. The gaps are the 5080's findings 6–9 on every part (GEMM + residual and attention as
per-cell kernels, the whole-cache update, one-thread RMSNorm: `local-rtx-5080/model-scale.md`); the TF32 rows add that
those cell kernels have no TF32 path, worst on the H100 whose TF32 competitor runs on its MMA units.

| projection (m × n × k) | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---:|---:|---:|---:|
| m1 QKV 6144×4096 / O 4096×4096 | 1.30 / 1.52 | 1.49 / 1.79 | 1.43 / 1.57 | 1.74 / 2.17 |
| m1 gate-up 28672×4096 / down 4096×14336 | 1.08 / 1.12 | 1.09 / 1.11 | 1.17 / 1.46 | 1.12 / 1.35 |
| m16 QKV / O | 1.35 / 1.41 | 1.24 / 1.10 | 1.14 / 1.52 | 1.38 / 1.31 |
| m16 gate-up / down | 1.08 / 1.12 | 1.07 / 0.77 | 1.02 / 1.03 | 1.15 / 1.11 |
| m128 QKV / O | 1.31 / 1.61 | 0.84 / 1.36 | 1.29 / 1.26 | 1.77 / 1.65 |
| m128 gate-up / down | 0.92 / 1.60 | 0.82 / 1.37 | 0.96 / 1.15 | 1.03 / 1.51 |
| m128 TF32 QKV / O / gate-up / down | 1.62 / 1.32 / 1.16 / 0.79 | 1.60 / 1.63 / 1.11 / 1.87 | 1.51 / 1.49 / 1.00 / 1.26 | 2.81 / 1.89 / 1.31 / 2.25 |
| m2048 QKV / O | 1.11 / 1.04 | 0.77 / 0.83 | 0.60 / 0.96 | 1.11 / 1.15 |
| m2048 gate-up / down | 1.04 / 1.10 | 1.00 / 0.93 | 0.64 / 0.58 | 1.09 / 1.13 |
| m2048 TF32 QKV / O / gate-up / down | 1.19 / 1.08 / 1.16 / 1.20 | 1.44 / 1.27 / 0.93 / 0.97 | 1.05 / 1.12 / 1.23 / 0.95 | 1.74 / 1.75 / 1.60 / 1.70 |
| README suite pj: m1 × 2048 × 7168 / m1 × 18432 × 7168 | 1.52 / 1.05 | 1.70 / 1.09 | 1.06 / 1.17 | 2.11 / 1.12 |
| README suite pj: m16 × 2048 × 7168 / m16 × 18432 × 7168 | 1.58 / 1.06 | 1.07 / 1.04 | 1.70 / 1.03 | 1.40 / 1.02 |

Per-row legs and nulls: `analysis/combined.md` in the raw directory.

## Evidence (intent check, `evidence-<part>.tsv`)

| part | checks | mismatches | largest mismatch groups |
|---|---:|---:|---|
| rtx-3090 | 54 749 | 1 736 | kmajor-a 128-bit LDS (393), fma-pairs (389), min-blocks no-spill (256), face-registers no-spill (256) |
| a100 | 43 743 | 2 865 | face-registers residency at ptxas regs + smem (837), fma-pairs (398), streamk one wave (381), kmajor-a (224) |
| rtx-4090 | 42 137 | 1 853 | fma-pairs (394), kmajor-a (240), prefetch LDG cover (231), min-blocks / face-registers no-spill (182 each) |
| h100 | 61 649 | 3 831 | face-registers residency (1107), streamk one wave (429), kmajor-a (407), fma-pairs (386), prefetch (352) |

Each mismatch is a finding (CLAUDE.md evidence pass); the residency group on the 2048-thread parts (a100, h100) sits
with the `min-blocks` losses above.

## Open items (missing facts)

Items 1, 4, 5 and 6 are addressed in `rules.md` (properties, replay, the 5080's runs); 2 and 3 remain
open there with what the round separates.

1. `row-split`: the split's added launches per trip against the pass it splits (H100 SXM 4 / 4 lose, explained).
2. `min-blocks` on the 32-register parts: the H100 SXM and the A100 lose where the H100 PCIe won; and the under-filled
   `SMEM_PER_SM_MANY_BLOCKS` branch (block decode 0.70).
3. `warp-fill`: at wps 1 the balance does not separate winners from losers; at filled sites the m16 projections win
   on three parts.
4. `BALANCE_PM` placed on boost-era clocks; a measured clock moves the 4090 below it while its rungs still win.
5. `pdl`: dependence per launch edge (independent launches lose under PDL).
6. Tune trials under power caps pick candidates that lose in steady state.
7. `emit --rungs` takes a profile name only: the replay above is `rules.rs` in Python (checked against `emit`); the
   native-gpu path with these calibration lines would make it the compiler's own.

Raw (local, not in git): `.claude/raw-evidence/cross-arch/<part>/` (results, run output, `calib-<part>.txt`,
`evidence-<part>.tsv`, profile / trace / tune archives, wisdom, bundle log and arm list), `analysis/` (per-part
property, arm, model-scale and gap tables; `combined.md`), `scripts/` (build, launch, watchdog, analysis).

## 7a2b67bf (device geometry, exp, cross-arch rules, large GEMM)

Branch `local-rtx-5080` @ `7a2b67bf` (device geometry: tile-post, head-group, piece-delta, row-tree; `exp` builtin with
MUFU.EX2 on the default face and the rescale law; the cross-arch rules of `rules.md`; large GEMM: wide-fma, streamk-fill,
k-fill) + this round's tooling (branch `cross-arch-device-geometry`). Four vast.ai boxes in parallel, CUDA 13.0.2 devel
image, torch 2.11.0+cu130, the method above (`integrated_bundle.sh` `LADDER=1`, `box_run.sh` REPS=30, `evidence.py`
check, `box_profile.sh`), plus `exp_calib.cu` on every box and a trace sweep (`SWEEP_PROF=1`) for the gap analysis.

**Bundle, one per part.** The 35 README programs + two GEMM cells, the 12 model-scale layer programs and 24 projections
(`XREF`), the 5 `px_llm_*` differentials, and new: the three long-row softmaxes `pt_lsm_*` (exp's rescale law). Arms:
every rung `emit --rungs` ships differently on the part than on the rtx-5080 (3090 grid-share / panel-regs / warp-fill /
wide-fma ON, splitk ON exact; a100 warp-fill / min-blocks / wide-fma ON, fill-rows / unroll ON exact; 4090 async /
kmajor-a / wide-fma OFF; h100 wide-fma ON, row-split ON forced, fill-rows ON exact — `pdl` now ships ON on the H100 for
repeating launches, so its arm is OFF, plus `pdl-early` ON on the five tile chains); the large-GEMM rungs every part ships
(streamk-fill, k-fill OFF); the device-geometry kernels (all four OFF on 4 layer programs, each OFF on 2); `exp-hw` OFF
on 7 exp programs. The H100 bundle left out min-blocks / warp-fill ON (open items, unchanged rules; its only offer
was $4.88/h when the bundles were built).

**Tooling findings (fixed, `cross-arch-device-geometry`).** (1) A kernel's name hashes the rung set it was emitted
under, so an arm whose rung changes no kernel still renamed its site tables' candidates and llc numbered their labels
differently: the bundle's byte compare kept **535 of 1 038** arms that equal the shipped build (off-pdl on straight-line
programs, off-k-fill / off-streamk-fill where no site takes them …). `armdedup.py` compares bodies with names, labels and
comments canonical; the bundles here were deduplicated before upload (3090 134, a100 143, 4090 127, h100 80 arms timed).
(2) The `SWEEP_MIN` edit to `vast_run.sh`'s `ENVS` lost a space (`SWEEP_MIN='25'TORCH_MIN='50'`): `box_run.sh` cut the
torch legs at its 15 min default and the sweep's deadline was no number. The A100 had finished (torch legs for **23 of
74** programs); on the other three the profile pass was patched before it ran (`torch_rest.sh`: the legs of every timed
program without one, then the sweep), so torch covers 3090 61, 4090 72, H100 45 programs (H100: 13 min budget). A
second A100 rental for its torch legs was not made (outside the approved four). Fixed in `vast_run.sh`.

### Boxes and cost

| part | GPU (driver, power limit) | host CPU | offer | $/h | hours | cost | status |
|---|---|---|---|---:|---:|---:|---|
| rtx-3090 | RTX 3090 24 GB (580.173.02, **200 W**) | EPYC 7502 | 57929792 | 0.217 | 1.61 | $0.35 | done (+ torch remainder) |
| (3090 tries) | — | — | 61732982 / 57089696 | 0.18–0.20 | 0.30 | $0.06 | image pull never finished (15 min); created stopped |
| a100 | A100-SXM4-40GB (610.57, **270 W**) | EPYC 7B12 | 61833146 | 0.438 | 0.96 | $0.42 | done; torch legs 23 / 74 programs |
| rtx-4090 | RTX 4090 24 GB (610.57, 350 W) | EPYC 7402 | 58765415 | 0.421 | 1.92 | $0.81 | done (+ torch remainder; 215 MB profile at 0.4 MB/s) |
| h100 | H100 80GB HBM3 SXM (595.91, 700 W) | Xeon Platinum 8480+ | 59816179 | 2.411 | 1.35 | $3.25 | done (+ torch remainder) |
| **total** | | | | | | **$4.88** | every `mapal-xarch-*` instance destroyed (checked by id and by label) |

$/h is the instance's `dph_total`; hours from creation to the destroy. The account's credit fell $40.90 over the window
(208.78 → 167.88): other projects' B200 / B300 / H100 billed through it. **Power limits decide which tables are
like-for-like:** the 200 W 3090 settles to 1.277 of its 1.588 GHz burst clock after 3.7 s (`calib ffma`) and its
library legs' median / min is 1.13 (max 2.2); the 270 W A100's GEMM legs throttle to median / min 1.31 (max 4.8: cuBLAS
2048 × 28672 × 4096 108 ms against a 22 ms minimum) — a short Mapal process does not. The 4090 (350 W this time, 250 W
last round) and the H100 (700 W) hold 1.01: their tables are the like-for-like ones.

### Value gates — all pass

| part | differential | probe gates (programs) | arms · candidates within D2/D3 | device-exact oracles | split programs | gemm_lib legs within | torch legs within | mlp_bench bit-exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rtx-3090 | 23 / 23 | 74 / 74 | 699 / 699 | 60 / 60 | 9 / 9 | 112 / 112 | 201 / 211 | 27 / 27 |
| a100 | 30 / 30 | 74 / 74 | 736 / 736 | 60 / 60 | 9 / 9 | 112 / 112 | 67 / 67 | 45 / 45 |
| rtx-4090 | 30 / 30 | 74 / 74 | 735 / 735 | 60 / 60 | 11 / 11 | 112 / 112 | 229 / 240 | 34 / 34 |
| h100 | 30 / 30 | 74 / 74 | 675 / 675 | 60 / 60 | 8 / 8 | 112 / 112 | 139 / 147 | 34 / 34 |

No Mapal value failed on any part, face, arm or site candidate (the new device-geometry kernels, the hardware `exp`, the
large-GEMM tiles and the lsm programs included). Every torch leg outside its tolerance is a cross-precision leg (TF32
mode on `mlp_2048` / `mlp2_2048` or f32 mode on their `_tf32` twins, maxdev 1.4–4.7e-3 against the f32 1e-3), as last
round.

### Properties: the profile against the box

| property | rtx-3090 profile → box | a100 | rtx-4090 | h100 |
|---|---|---|---|---|
| FFMA clock, burst / sustained 4 s, kHz | 1 561 000 → 1 588 212 / **1 277 156** (settle 3.7 s) | 1 410 000 → 1 408 892 / 1 408 851 | 2 520 000 → **2 224 274** / 2 213 007 | 1 807 000 → 1 819 042 / 1 819 302 |
| FFMA per DRAM byte, ‰ (`FILL_BALANCE_PM` 45 000) | 35 008 → 35 618 | 12 534 → 12 524 | 81 919 → 72 306 | 18 216 → 18 338 |
| DRAM B / µs / SM (`DRAM_PER_SM_LOW` 9 200, `DRAM_PER_SM` 14 600) | 11 414 | 14 398 | 7 875 | 25 393 |
| DRAM / L2 / one-SM read, GB/s | 891 / 2 302 / 62 | 1 486 / 2 776 / 82 | 958 / 4 922 / 87 | 3 186 / 4 067 / 149 |
| dependent launch / PDL / cooperative, ns | 1 472 → 1 416 / — / 2 802 | 2 050 → 2 118 / — / 4 787 | 1 440 → 1 208 / — / 2 402 | 1 722 → 1 754 / 743 / 1 755 |
| grid barrier, ns | 509 | 1 112 | 440 | 931 |
| pageable / pinned H2D, ‰ (`PAGEABLE_PM` 900) | 940 | 750 | 707 | **295** |
| `ex2.approx.ftz.f32` on [−½, ½] (`Caps::ex2`), exhaustive | 2^−22.709 → **3** | 3 | 3 | 3 |
| exp over every f32, worst ulp: hardware form / expf / __expf | 2.068 / 1.931 / 63.78 | same | same | same |

Device-query facts equal the profiles on all four. **No rung flips on the measured values** except `pinned` (a host's
pageable ratio, unchanged suite: no program copies a host array in a loop). `Caps::ex2` and the hardware form's ulp are
bit-for-bit the 5080's on Ampere, Ada and Hopper: the SFU's `ex2` is one algorithm across these parts, and the bound rule
`(u + 1)·2⁻²⁴ ≤ 2⁻²¹` admits the hardware form on all of them. `row_split_pays` on the measured H100 values: the 256 ×
256 row alone 262 144 B / 149 B/ns − / 3 186 B/ns = 1 675 ns saved < 2 × 1 754 ns of launches → priced off, as replayed.

Calibration lines (`DeviceQuery::calib_key` + measured):
```
cc=86 sms=82 l2=6291456 mem=9751000x384	dram_bytes_per_ns=891 l2_bytes_per_ns=2302 gsync_ns=509 dep_launch_ns=1416 coop_launch_ns=2802 pageable_pm=940 fp32_lanes=128 clock_khz=1588212
cc=80 sms=108 l2=41943040 mem=1215000x5120	dram_bytes_per_ns=1486 l2_bytes_per_ns=2776 gsync_ns=1112 dep_launch_ns=2118 coop_launch_ns=4787 pageable_pm=750 fp32_lanes=64 clock_khz=1408892
cc=89 sms=128 l2=75497472 mem=10501000x384	dram_bytes_per_ns=958 l2_bytes_per_ns=4922 gsync_ns=440 dep_launch_ns=1208 coop_launch_ns=2402 pageable_pm=707 fp32_lanes=128 clock_khz=2224274
cc=90 sms=132 l2=52428800 mem=2619000x5120	dram_bytes_per_ns=3186 l2_bytes_per_ns=4067 gsync_ns=931 dep_launch_ns=1754 pdl_launch_ns=743 coop_launch_ns=1755 pageable_pm=295 fp32_lanes=128 clock_khz=1819042
```

### Rules: the cross-arch fixes against their replay

Median `iter ms` (5 processes, ABBA) of the arm over the shipped build's rules pick, beyond max(null, 2 %) (`arms.py`).
`flipped / shipped` > 1: the shipped rule wins.

| rule (part) | predicted by the replay (`rules.md`) | measured on the box | verdict |
|---|---|---|---|
| `pdl` on repeating launches, exit trigger (h100) | loops ON: gen 0.788, causal 0.790, prefill 0.796, decode 0.922, dec B1 0.972 of plain; straight-line plain | OFF / ON: gen **1.154**, causal **1.140**, dec B1 L2048 **1.180** / L8192 **1.132**, pre T8192 1.062, dec B8 1.035–1.055, dec B32 1.013–1.037, rt_mlp2_1024 **1.029** (an open loss in the replay), blkdec 1.012 (open loss → null); decode_1024 0.986 and prefill_1024 0.988 (9 % null) null — 9 W / 9 null / 1 L (lsm_32_8192 0.936: every arm of that program 0.93, the shipped arm's own spread). Straight-line modules emit no `griddepcontrol` (their OFF arm = shipped, dropped) | **held** (the loop gains 13–18 %, decode_1024's 8 % not seen) |
| `pdl-early` (h100) | the entry trigger is the candidate for the tile-chain losses | ON / shipped: rt_mlp2_1024 **1.214**, blkdec 1.025, three null | **confirmed**: the entry trigger was the H100 tile-chain loss; the exit trigger is right |
| `row-split` priced (h100) | the four 256-float rows off: 4 losses → wins (0.851 of the split) | priced off (above); rows last round → now (ms): decode 24.86 → 18.45, gen 16.81 → 10.82, causal 33.34 → 21.71, prefill 17.08 → 13.08 (with the exit trigger and exp); the split is a table axis there (the tune pass measures it) | **held** |
| operand-stream rungs on DRAM / SM < 9 200 (4090) | ON kept at any clock | 350 W box, 2.22 GHz: wide-fma OFF loses 19 / 27 (0 wins), kmajor-a OFF loses 5 / 24 (0 wins); async OFF 2 W / **3 L** (pj_m128_n6144 0.848, n4096_k14336 0.890, mlp_512 0.816 in a 9.8 % null), 23 null | **held** for wide-fma / kmajor-a; async: new loser at the k-filled m128 sites (below) |
| steady-state tune trials (capped boxes) | 5 of 16 steady-state-slower picks revert | AUTO slower than rules beyond null: 3090 (200 W, settles 3.7 s) **0** (last round 2); a100 (270 W) **3** (last 6: rt_mlp2_2048 / rowsum_4096 / loop_4096 gone; new pre_t8192 1.338, pj_m16 ×2 1.03); 4090 3 (blkdec **1.130**, last round 1.110 — persists); h100 3 (attn 1.082, mlp_512 1.074, pj_m16_n6144 1.067). AUTO faster beyond null on 22 / 19 / 12 / 38 of 56–57 tabled programs | mostly **held**; blkdec (4090) and pre_t8192 (a100) open |

**New rules on the other parts.**

| rule | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|
| device geometry, all four OFF (dec B1 / B32 L2048, blkdec, pre T2048): OFF / shipped | 6.32× (4 W) | 4.22× | 4.53× | 4.19× |
| tile-post / head-group OFF (dec B8, pre T2048) | 7.30× / 4.87× | 6.39× / 4.54× | 8.73× / 7.97× | 6.40× / 4.70× |
| row-tree / piece-delta OFF | 1.06 (1 W 1 null) / 1.30 | 1.22 / 1.50 | 1.12 / 1.57 | not bundled |
| `exp-hw` OFF (7 / 5 programs) | 3 W 4 null, 1.022 | 3 W 4 null, 1.044 | 2 W 5 null, 1.037 | 2 W 2 null (1 L: the lsm_32 outlier), 1.032 |
| wide-fma (ships ON only on the 4090: DRAM/SM < 9 200, or contract and ≥ 45 ‰) | **ON wins 12 / 25, 0 L** (0.93–0.97: mlp_2048, pj_m128_n28672, pj_m2048, pre T2048, indep, fanout, diamond, twin, rowsum …): the rule loses | ON 2 W (pj_m2048_n28672 0.752 in an 18 % null, pre T2048 0.958) / **9 L** (1.02–1.31): the rule holds | OFF loses 19 / 27: holds | **mixed**: ON wins the m2048 projections 0.934–0.964, smlp 0.727, loop 0.970, epi 0.973; loses mlp_2048 1.037, mlp2 1.052, pj_m128_n28672 1.167 |
| streamk-fill (all parts) OFF | 6 W / 1 L (pj_m16_n18432 0.946) | 13 W / 0 L (1.05–2.43) | 2 W / 1 L / 13 null | 11 W / 1 L (lsm outlier): m16 / m128 projections 1.35–2.28, decode B32 1.13–1.34 |
| k-fill (all parts) OFF | 3 W (pj_m128 1.10–1.80) | 3 W (1.07–1.70) | 10 null | 3 W (1.35–1.71) |

Device geometry and `exp` hold on every part: 0 losses over 4 parts. `streamk-fill` and `k-fill` hold where the site
takes them (4090: null). **wide-fma is the new mixed rule**: FFMA per DRAM byte does not separate the parts — the 3090
(35.6 ‰) wins it, the H100 (18.3 ‰) wins it on grids of many waves and loses on grids under ~2 waves, the A100 (12.5 ‰)
loses it. What does separate them: the A100 is the only 64-lane part (its FFMA rate per SM is half its shared-load
rate, so the 128-accumulator tile's lower LDS issue per FFMA buys nothing), and on the 128-lane parts the losers are the
sites whose wide-tile grid covers the SMs fewer than ~2 times (mlp_2048: 256 blocks on 132 SMs; pj_m128_n28672: 224)
while the same sites win on the 82-SM 3090 (3.1 / 2.7 waves). Missing facts: **FP32 lanes per SM against shared-load
bandwidth per SM** (part) and **the wide tile's grid in waves of SMs** (site).

**Open rules, re-measured.** `warp-fill` ON at one warp a scheduler: a100 11 W / 8 L, 3090 4 W / **11 L** (decode B8
L2048 0.473, attn 0.787, the m16 projections 0.87–0.92) — still mixed. `grid-share` ON on the 3090: 2 W / **6 L** (decode
B8 L2048 0.493 / L8192 0.710, convmc 0.86) — last round the shipped OFF won 12 / 6; with the device-geometry decode
kernels the OFF now loses net (geomean ON 0.962). `min-blocks` ON on the a100: 2 W / 3 L / 42 null. **New loser,
`calls` (kernel merging priced on measured launch / barrier costs):** on the H100, `smlp_4096_256_1024` and
`firew_rowsum_2048x2048_16` now run as one persistent kernel (`; mapal.calls.host chains=1 per_ns=2749 sep_ns=3444` +
`mapal.hz` folding a 16 384-block and a 2 048-block kernel): smlp 199 µs against last round's 70 + 55 µs, firew 79 µs
against 31 + 12 µs — the price saves ~0.7 µs of launch and loses 36–74 µs of phase time.

**Evidence check** (`evidence-<part>.tsv`): 3090 117 036 checks / 3 820 mismatches, a100 112 007 / 7 894, 4090 106 184 /
4 786, h100 92 350 / 8 379. Largest groups: face-registers residency (a100 2 304, h100 1 870), **`pdl: griddepcontrol at
kernel entry` (h100 1 545: the intent still expects the entry trigger the code moved to the exit — intent.py is stale)**,
prefetch LDG cover (4090 503, h100 747), kmajor-a 128-bit LDS (3090 627, 4090 485), fma-pairs (468–559 each).

### Performance — Mapal / best competitor, last round (8bba2a50) → now

Best competitor as above (f32 legs for f32 programs, TF32 for `_tf32`, bf16 never). Geomeans over the rows that have a
competitor in both rounds. **Like-for-like: rtx-4090 and h100**; the 3090 (200 W) and a100 (270 W) rows against library
legs that throttle over 30 back-to-back repetitions are optimistic for Mapal (projections 0.29–0.61 on the a100), and the
a100 has torch legs for 23 programs only.

| suite | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|
| README GPU suite (rows, wins now) | 0.89 → **0.67** (24, 18) | 0.84 → **0.71** (24, 17) | 0.67 → **0.73** (30, 19) | 1.07 → **1.06** (27, 6) |
| model-scale layer, llm_* (rows, wins) | 5.11 → **0.78** (12, 9) | 3.47 → **0.63** (7, 5) | 3.18 → **0.99** (12, 8) | 5.69 → **0.92** (12, 6) |
| model-scale projections, pj_* | 1.21 → **0.98** (28, 10) | 1.15 → **0.62** (28, 21) | 1.12 → **1.24** (28, 3) | 1.46 → **1.47** (28, 2) |
| GEMM cells 16×2048×7168 / 4096³ | 1.08 / 1.11 → **0.90 / 1.52** (21.6 % null) | 1.73 / 1.10 → **1.02 / 1.02** | 1.14 / 1.05 → **1.13 / 1.00** | 1.86 / 1.17 → **1.85 / 1.18** |

The layer is the device-geometry result: from 3.2–5.7× torch-compile f32 to 0.63–0.99, decode now **0.26–0.88** on every
part. The 4090's projections and README suite moved against Mapal because its library legs no longer throttle (350 W vs
250 W): its last-round "wins" at m 2048 (0.58–0.64) are 0.88–1.02 now.

| layer row (decode per step) | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---:|---:|---:|---:|
| decode B1 L2048 / L8192 | 2.06 / 3.13 → **0.75 / 0.73** | 1.36 / 1.80 → **0.60 / 0.59** | 1.20 / 1.60 → **0.77 / 0.88** | 2.03 / 2.18 → **0.59 / 0.53** |
| decode B8 L2048 / L8192 | 2.05 / 1.55 → **0.77 / 0.40** | 2.29 / 2.41 → — | 1.36 / 1.19 → **0.69 / 0.44** | 3.04 / 2.72 → **0.47 / 0.42** |
| decode B32 L2048 / L8192 | 1.91 / 1.46 → **0.26 / 0.32** | 2.02 / 2.43 → **0.39 / 0.30** | 1.25 / 1.10 → **0.48 / 0.40** | 2.19 / 2.03 → **0.40 / 0.26** |
| block decode B1 L4096 | 2.01 → **0.70** | 1.92 → **1.01** | 1.62 → **1.37** | 2.66 → **1.20** |
| prefill T2048 / TF32 | 17.5 / 24.1 → **0.59 / 0.78** | 8.01 / 24.9 → — | 7.85 / 13.2 → **0.99 / 1.19** | 11.5 / 40.9 → **1.12 / 1.10** |
| block prefill T2048 / TF32 | 17.2 / 24.6 → **1.13 / 1.07** | 7.64 / 34.2 → **0.89 / 1.08** | 9.33 / 17.5 → **0.92 / 1.18** | 7.73 / 40.2 → **1.11 / 1.90** |
| prefill T8192 | 15.6 → **9.73** | 18.8 → — | 8.96 → **12.9** | 18.3 → **18.7** |
| lsm 1×131072 / 4×32768 / 32×8192 (new) | 11.1 / 3.33 / 2.99 | — | 8.79 / 3.10 / 3.02 | 9.21 / 3.03 / 4.71 |

| README row (like-for-like parts) | rtx-4090 | h100 |
|---|---:|---:|
| attn_1024_64 | 1.31 → 1.61 (tc) | 1.55 → 1.56 (fused) |
| causal / gen / prefill / decode_1024 | 0.11 / 0.11 / 0.11 / 0.60 → 0.09 / 0.09 / 0.10 / 0.37 | 0.31 / 0.30 / 0.29 / 1.33 → 0.15 / 0.17 / — / 0.76 |
| convmc / convmc TF32 | 1.74 / 2.06 → 1.72 / 2.22 | 1.61 / 3.45 → 1.63 / 3.10 |
| mlp_512 / indep_512 | 1.15 / 1.08 → 1.41 / 1.19 | 1.57 / 1.74 → 1.84 / 1.78 |
| mlp_2048 / mlp2_2048 / epi / loop | 0.92 / 0.90 / 0.58 / 0.63 → 0.95 / 0.91 / 0.89 / 0.88 | 1.11 / 1.10 / 1.10 / 1.09 → 1.12 / 1.11 / 1.12 / 1.09 |
| mlp / mlp2 TF32 | 0.80 / 0.78 → 1.05 / 1.21 | 2.38 / 2.43 → 2.79 / 2.67 |
| smlp / firew_rowsum | 0.98 / 0.61 → 0.82 / 0.64 | 1.17 / 0.58 → **1.78 / 1.33** (`calls`, above) |
| softmax_rows_4096 / rowsum_4096 / ew_4096 | 0.99 / 0.66 / 0.36 → 0.88 / 1.01 / 0.42 | 1.09 / 1.08 / 1.37 → 1.07 / 1.11 / 1.02 |

Every row, leg and null for all four parts: `analysis/compare.md` in the raw directory.

### Gap analysis — how the compiler maps dataflow geometry onto each part

Rows where Mapal trails beyond null on a like-for-like part (4090, H100), and rows whose gap differs much between parts;
the capped parts' rows where the gap survives the throttled legs. Library side: nsys traces of the library / torch legs
(`SWEEP_PROF`: kernel names, grid, block, registers, smem) and the cells' cuBLAS / CUTLASS launches; Triton SASS from
the inductor cubins; Mapal side: its traces and the bundle's SASS (`evidence/*.sass`). Counters are refused in vast
containers (no ncu). Class: **(a)** a capability the part has that the compiler does not use, **(b)** a property the
rules read wrongly or not at all, **(c)** a dataflow fact missing. A row marked *open* has no library SASS behind it.

| part | workload | Mapal / best | what the library does on that part | Mapal's realization | class | missing property / capability / fact (geometry) |
|---|---|---:|---|---|---|---|
| h100 | 4096³ cell; m2048 projections | 1.18; 1.10–1.15 | cuBLAS `sm80_xmma_gemm_f32f32…tilesize256x128x8_stage3_ffma`: 16 × 32 blocks of 256 threads, 254 regs, 68 KB smem — 128 accumulators a thread, 3 stages (torch's f32 prefill GEMMs: the same kernel) | 264 blocks × 256, 113 regs, 17 KB; 1 024 FFMA per 64 LDS.128 (4 FFMA per shared float), REDG stream-K | b | FP32 lanes per SM against shared-load bandwidth per SM, and the wide tile's grid in waves of SMs: the 128-accumulator tile (`wide-fma`) is read from FFMA per DRAM byte (18 ‰ → off) — its ON arm wins these sites 0.93–0.96 |
| h100, 4090, 3090 | 16 × 2048 × 7168 cell; m16 projections | h100 1.85, 1.31–1.54; 4090 1.13, 1.04–1.91 | cuBLASLt split-K 38: `cutlass_80_simt_sgemm_128x32_8x5` grid 8 × 2 × 38 (608 blocks), 5-stage cp.async, 80 regs, 27 KB + `splitKreduce` (41 + 3 µs on the h100) | 256 blocks, per k-step 64 FFMA against 128 scalar LDS and 4 LDG.128 (2 shared loads per FFMA), 89 µs | b + a | the 16-row tile's shared-load issue per FFMA (the m operand as scalar LDS, no 128-bit fragment along k for a 16-row panel) and the split's waves on 132 SMs; the 5-stage cp.async depth the library keeps on the skinny panel |
| h100, 4090 | m1 projections (GEMV) | h100 1.16–2.37; 4090 1.18–1.79 (3090 1.01–1.29) | cuBLAS-fresh / Lt-fresh GEMV (`gemvx::kernel` 96 × 64·8 threads, 128 regs in the torch traces); Triton's GEMV (`triton_red_fused_mm`) uses LDG.128 / LDSM | k-split tiles 16 × 8 / 16 × 16 blocks of 256, 40 regs, 128 **scalar** LDG a thread + RED (all parts) | b | bytes in flight per SM against DRAM bytes per µs per SM × latency: 128–256 blocks of 4-byte loads cannot cover the H100's 25.4 KB/µs/SM (the gap grows with that property: 3090 11.4 → 1.0–1.3, 4090 7.9 but 2.2 GHz → 1.2–1.8, h100 → 2.4); missing: the GEMV site's load width and k-split depth from that product |
| h100 | TF32: mlp_2048 / mlp2 TF32, pj TF32, blkpre TF32, convmc TF32 | 2.79 / 2.67, 1.34–2.95, 1.90, 3.10 | torch TF32 / cuBLAS TF32 legs — kernels not traced this round (gemm_lib runs outside nsys) — *open* | blkpre_tf32: HGMMA (wgmma) fed by LDGSTS (32 HGMMA, 960 LDGSTS, WARPGROUP.ARRIVE / DEPBAR), no UTMALDG / SYNCS; mlp_2048_tf32: HMMA (mma.sync) + LDSM, no wgmma | a | Hopper's TMA (bulk tensor copies), mbarrier-tracked stage pipeline and warp-specialized producer warpgroups (and thread-block clusters) are not modeled: wgmma is issued but staged by cp.async from the same warps; the TF32 MLP site does not take wgmma at all |
| 4090 | TF32 projections / mlp TF32 | 1.00–1.57 / 1.05–1.21 | cuBLAS TF32 (mma.sync class) — not traced — *open* | HMMA + LDSM + LDGSTS | a (open) | stage depth of the TF32 panel (the library's multistage cp.async); not separated without the library SASS |
| 4090, h100, all | attn_1024_64 | 4090 1.61, h100 1.56, 3090 1.72, a100 1.49 | torch-compile: Triton `triton_per_fused_div_exp_max_sub_sum` (1 024 blocks × 256, MUFU) + cuBLAS sgemm; chain_bench `fused`: `ampere_sgemm_32x128_tn` + `softmax_row` | 4090: one flow kernel, **64 blocks** × 256 on 128 SMs, 101 KB smem, 128 regs (89 % of the time); h100: three tile kernels 20 + 14 + 23 µs (16 × 16 blocks); 3090: five tile kernels | b | the flow unit's grid against the SM count (4090: 1 024 query rows in 64 blocks leave half the SMs idle — rows are not split further when blocks < SMs), and the unit's per-part price (the H100, with 228 KB smem per SM, keeps three kernels) |
| all | lsm 1 × 131072 / 4 × 32768 / 32 × 8192 | 8.8–11.1 / 3.0–3.3 / 3.0–4.7 | torch-sdpa / compile: Triton `triton_red_fused__softmax…` reductions 16 blocks × 512 per row-set, MUFU | one block a row (`lsm_4`: 4 blocks of 256 hold 85–88 % of the time on the 3090 / 4090 traces) | b | SM count against rows: outside a loop the row-slices rule prices the first cooperative launch (exp.md open 2) and keeps one block a row; missing: the slice count from the SMs and the cooperative launch measured per part (4787 ns a100 … 1755 ns h100) |
| h100 | smlp, firew_rowsum | 1.78 (was 1.17), 1.33 (was 0.58) | cuBLASLt two kernels / torch-compile | `calls` + horizontal merge: one persistent kernel (199 µs / 79 µs against 70 + 55 / 31 + 12 µs as two) | b | the `calls` price reads launch (1754 ns) vs barrier (931 ns) only; the phases' time under one shared grid (a 16 384-block phase folded onto the 2 048-block grid) is not in it |
| h100, 4090 | mlp_512, indep_512 | h100 1.84 / 1.78, 4090 1.41 / 1.19 | cuBLASLt one kernel per GEMM / streams | 2–4 straight-line launches, ~10–20 µs each | b | launch cost against site time: the same price as above, the other way — short phases (≪ dep_launch × sites) are where merging pays |
| all | convmc 8 × 64 × 56 × 3 (f32 / TF32) | 4090 1.72 / 2.22, h100 1.63 / 3.10, 3090 1.62, a100 1.53 | torch eager: cuDNN `conv2d_grouped_direct_kernel` (1 568 blocks × 1 024, 43 regs) | implicit-GEMM tiles: 392–784 blocks × 64 threads, 107–128 regs, LDGSTS where async | b | warps per scheduler of the fitted grid: warp-fill ON at wps 1 wins convmc 0.71 (a100) and grid-share ON 0.86 (3090) — the open `warp-fill` fact (fitted grid's wave quantization) |
| all | prefill T8192 | 9.7–18.7 | torch-compile: cuBLAS f32 GEMMs + Triton fused softmax | per-cell kernels for the 2³¹-cell scores and the KV-group loop (5080 finding 10) | c | the scores' cell index in i64 and the KV-group loop's `g·4` in the address keep the attention off tile sites: an affine-index fact across the loop counter and the i64 range |
| 4090 | block decode B1 L4096 | 1.37 | torch-compile: Triton reductions fusing RMSNorm with the GEMV (`triton_red_fused_add_mean_mm_mul_pow_rsqrt`, 1 536 blocks × 256) | GEMV tiles 16 × 8 × 256 scalar-LDG + the cache's attention units; the tune pass picks a candidate 1.130 slower than rules | b | the GEMV in-flight bytes (row 3) and the trial's steady state on this site (open since last round) |
| 4090 | async at the m128 projections | off arm 0.848 / 0.890 | — | `k-fill` fills these sites along k; `cp.async` is chosen for the under-filled site before the fill | b | the async rule reads the site's fill before `k-fill` changes it: the fill after k-fill is the property |

**Across parts, ranked by suite time.**
1. **Missing dataflow fact, all parts (c): prefill T8192** — 9.7–18.7×, by far the most time of any row (one program).
2. **Under-filled grids against the SM count (b), architecture-general:** attention (1.5–1.7×), the long-row softmaxes
   (3–11×), the skinny GEMM / m16 projections (1.1–1.9×). The flow unit and the row slices are sized from the rows, not
   from `sms`; the gap grows with SMs (H100 132, 4090 128).
3. **GEMV bytes in flight (b), architecture-general but scaling with DRAM B/µs/SM:** m1 projections and the decode
   layer's projections (decode still beats torch-compile 0.26–0.88 because torch's GEMV is no better); worst on the H100.
4. **Hopper capability set (a): TMA + mbarrier pipeline + warp specialization** — the TF32 rows on the H100 only
   (1.3–3.0×); wgmma is used, the rest of the Hopper pipeline is not. *Open* until the library SASS is captured.
5. **The 128-accumulator tile's rule (b): lanes per SM and grid waves** — 1.1–1.2× on the H100's large f32 GEMMs; the
   3090 wins it today with the rule off. A rule correction, not a new capability.
6. **`calls` / launch pricing (b):** merging priced on launch vs barrier ns only — loses where phases are long (H100
   smlp / firew), and the short straight-line programs that would gain (mlp_512 / indep_512) stay unmerged.
7. **Conv window fill (b, open `warp-fill`):** convmc 1.5–1.7× on every part.

Architecture-general: 1, 2, 3, 5, 6, 7 (each reads or misses a property every part has: SMs, DRAM per SM, lanes per
SM, launch cost); specific to one capability set: 4 (Hopper TMA / mbarrier / clusters).

### Open items (missing facts)

1. `wide-fma`: FP32 lanes per SM vs shared-load bandwidth per SM, and the wide grid's waves (3090 wins ON 12 / 0, H100
   mixed, A100 loses).
2. `calls`: the phases' time under the merged grid in the price (H100 smlp 1.55×, firew 2.3× slower than unmerged).
3. `async` after `k-fill`: the site's fill after the k split (4090 m128 projections).
4. Flow-unit / row-slice grids against `sms` (attention, lsm) and GEMV bytes in flight per SM (m1, decode projections).
5. Hopper: TMA / mbarrier staging for wgmma; the TF32 MLP site without wgmma — capture the library's SASS first.
6. `warp-fill` (still mixed on a100 / 3090), `grid-share` on the 3090 (now loses net), `min-blocks` (a100 null).
7. Tune: 4090 block decode 1.130, a100 pre T8192 1.338.
8. `intent.py`: the `pdl` intent still expects the entry trigger (1 545 H100 mismatches).
9. The a100's torch legs (23 / 74 programs) and the capped boxes' library legs: a like-for-like a100 / 3090 row needs an
   uncapped box or sustained-clock-matched Mapal runs.

Raw (local, not in git): `.claude/raw-evidence/cross-arch/7a2b67bf/<part>/` (results, run output, `calib-<part>.txt` with
the `exp` lines, `evidence-<part>.tsv`, profile / trace / tune archives — `prof2-<part>.tgz` the patched profile with the
trace sweep —, wisdom, rest logs), `analysis/` (props, arms, gap, ms, sweep per part, `compare.md`), `scripts/` (build,
launch, watchdog, armdedup, torch_rest, rest, gates, props, gapsweep, compare).

## 60f0f8ce (half precision, review fixes, chain pricing)

Branch `local-rtx-5080` @ `60f0f8ce`: half precision (bf16 / f16 storage, ADR-0043: the bf16-weight decode rows), the
review fixes (the rescale law over masked slices, a next state made once is no device carry, the `row-slices` /
`exp-rescale` / `exp-hw` intents), `calls` priced with the members' body change (`coop_body_cycles`: the RTX 5080's
`calib.cu body` measurement, 4390 cycles, derived per part at its clock) and the named-host dispatch fix (bundles emit
with `--host=x86-64-v3`) + this round's tooling (branch `cross-arch-half-precision`: the `force` re-judge arm
`--calls=force` in `integrated_bundle.sh`; `box_run.sh` runs the torch legs bf16-weight rows first, projections last).
Four vast.ai boxes in parallel, CUDA 13.0.2 devel image, torch 2.11.0+cu130; `integrated_bundle.sh` `LADDER=1`,
`box_run.sh` REPS=30, `evidence.py` check, `calib.cu` (with `body`) in all three calibration passes, `exp_calib.cu`.

**Bundle, one per part** (82 timed programs on the 4090, 81 elsewhere): the 35 README programs + the two GEMM cells, the
12 f32 layer programs + **7 bf16-weight rows** (`_bf16w`: decode B 1/8/32 × L 2048/8192, block decode B1 L4096), 24
projections, the 3 lsm programs (XREF for the layer and projections), 9 px differentials (+ the two bf16w twins, + the
`_pv` prefill twins in which the h100 forms chains). Arms: **`calls` OFF where the price forms chains** (4090: causal /
gen / prefill and block decode with its `_bf16w` / `_pv` twins; 3090: causal / gen / prefill), **`--calls=force` where
it refuses them** (a100 / h100: causal / gen / prefill; 3090: block decode), `narrow-tile` OFF on three bf16w rows per
part. Against the 7a2b67bf bundles, 62–68 shipped programs a part emit identically (host IR less `; mapal.` lines, PTX);
the rest changed by their `calls` decision: chains that formed on the boundaries alone in 7a2b67bf are now refused by
the body price (3090 decode, rt_mlp2, block decode, decode B1, block prefill TF32; a100 causal / gen / prefill, block
decode, decode B1; 4090 decode, diamond, block prefill, decode B1 / B8; h100 smlp, firew_rowsum, lsm_32_8192, prefill,
block prefill, mlp / mlp2 TF32), and the 3090 / 4090 causal family keeps its chain at the higher price.

**Run notes.** Each box's torch legs were checked before its destroy (`ops_nsys.sh` in the bundle: the legs of every
timed program without one, then nsys traces of every `calls` arm): torch covers **81 / 81 programs on every part** (82 on
the 4090). The h100's `box_run.sh` was cut by its own `RUN_MIN` (72 min, set short for the then $3.70–4.56/h offers) 22
min into the torch legs: its `vast_run.sh` was paused, the remainder (38 programs) run by hand, then resumed — no
`== done`, so no timing section (mlp_bench) and no profile pass on the h100. Starts that never ran: 3090 one (image
pull past 15 min), 4090 three (one pull, two created stopped).

### Boxes and cost

| part | GPU (driver, power limit) | host CPU | offer | $/h | hours | cost | status |
|---|---|---|---|---:|---:|---:|---|
| rtx-3090 | RTX 3090 24 GB (595.91.07, **350 W**) | EPYC 7532 | 61720834 | 0.193 | 1.52 | $0.29 | done |
| (3090 try) | — | EPYC 7452 | 61869758 | 0.206 | 0.29 | $0.06 | image pull never finished |
| a100 | A100-SXM4-40GB (595.91.07, **300 W**) | Ryzen 7 7700 (8 cores) | 56853932 | 0.580 | 0.98 | $0.57 | done |
| rtx-4090 | RTX 4090 24 GB (580.178.04, **480 W**) | EPYC 9354P | 51934648 | 0.384 | 1.09 | $0.42 | done |
| (4090 tries) | — | EPYC 7742 / … | 60749252, 61538557, 61804555 | 0.36–0.42 | 0.30 | $0.12 | pull never finished; two created stopped |
| h100 | H100 80GB HBM3 SXM (595.91.07, 700 W) | Xeon Platinum 8480+ | 59816261 | 2.278 | 1.85 | $4.21 | done (box_run cut in its torch legs, remainder by hand) |
| **total** | | | | | | **$5.68** | every `mapal-xarch-*` instance destroyed (checked by id and by label) |

$/h is the instance's `dph_total`; hours from creation to destroy. The account's credit fell $28.84 over the window
(315.50 → 286.66): other projects' B200 / H100 / H200 billed through it. **All four tables are like-for-like this
round**: the library legs hold their clocks on every box (`gemm_lib` median / min 1.007–1.012; 7a2b67bf: 3090 1.13 at
200 W, a100 1.31 at 270 W). The h100 box is noisy on host-bound programs (13 rows with a null above 5 %: causal 13 %,
ew 16 %, lsm_4 15 %; median null 0.5 %, the others 0.1–0.2 %).

### Value gates — all pass

| part | differential | probe gates (programs) | arms · candidates within D2/D3 | device-exact oracles | split programs | gemm_lib legs within | torch legs within | mlp_bench bit-exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rtx-3090 | 41 / 41 | 81 / 81 | 656 / 656 | 67 / 67 | 8 / 8 | 112 / 112 | 267 / 281 | 27 / 27 |
| a100 | 54 / 54 | 81 / 81 | 690 / 690 | 67 / 67 | 8 / 8 | 112 / 112 | 269 / 281 | 30 / 30 |
| rtx-4090 | 56 / 56 | 82 / 82 | 715 / 715 | 68 / 68 | 8 / 8 | 112 / 112 | 274 / 286 | 30 / 30 |
| h100 | 54 / 54 | 81 / 81 | 685 / 685 | 67 / 67 | 8 / 8 | 112 / 112 | 265 / 277 | — (timing section cut) |

No Mapal value failed on any part, face, arm or site candidate: the bf16-weight rows (XREF within D2, the px twins bit
for bit on the exact face and within the bound on `_d`, the narrow-tile OFF arms), the forced and OFF chains, the lsm
rows (the masked-slice fix: values only — no flow unit is sliced on these four parts, see the evidence check). Every
torch leg outside its tolerance is a cross-precision leg (TF32 mode on the f32 `mlp_2048` / `mlp2_2048` / `rt_mlp2_*`
programs, f32 mode on their `_tf32` twins; maxdev 1.4–2.6·10⁻³ against 10⁻³), as in both previous rounds.

### Properties: the profile against the box

Device-query facts equal the profiles on all four. Measured (block 256, grid ≤ sms; `props.py`): FFMA clock burst /
sustained 3090 1 559 / 1 377 MHz (350 W), a100 1 394 / 1 394, 4090 2 219 / 2 207, h100 1 820 / 1 821; DRAM read 892 /
1 483 / 959 / 3 177 GB/s; dependent / cooperative launch 1 508 / 2 842, 2 448 / 5 040, 1 214 / 2 427, 1 940 / 1 946 ns
(h100 PDL 752); grid barrier 506 / 1 104 / 438 / 931 ns. `Caps::ex2` = 3 on all four (2^−22.709, the 5080's). The only
rung that flips on the measured values is `pinned` (pageable ratio 501 / 970 / 567 / 246 ‰; the a100 now above 900),
as before — no program copies a host array in a loop.

### `calls`: the chains against their price

Median `iter ms` (5 processes ABBA) of the arm over the shipped build; the 4090's block-decode arms against the shipped
build's AUTO picks (identical kernels keep their wisdom). h100: the minimum per arm (its shipped arms spread 1.26–2.38×).

| part | programs | price (per vs sep ns, profile) | arm | arm / shipped | null | verdict |
|---|---|---|---|---|---:|---|
| rtx-4090 | causal / gen / prefill | formed (6 912 vs 8 640) | OFF | **1.164 / 1.159 / 1.166** | ≤ 0.4 % | the chain wins |
| rtx-4090 | block decode, `_bf16w`, `_pv` | formed (7 916 vs 11 520) | OFF | 1.000 / 1.000 / 1.001 | 0.0 % | **formed, not realized** (below) |
| rtx-3090 | causal / gen / prefill | formed (8 248 vs 8 832) | OFF | **1.356 / 1.359 / 1.377** (min 1.14 / 1.36 / 1.14: OFF bimodal on this host) | ≤ 0.4 % | the chain wins |
| rtx-3090 | block decode | refused (cost ×3) | force | 1.004 | 0.0 % | null (1 of 3 forced chains realized, below) |
| a100 | causal / gen / prefill | refused (13 392 vs 12 300) | force | 1.013 / 1.015 / 1.017 | ≤ 0.1 % | refusal holds (marginal) |
| h100 | causal / gen / prefill | refused (8 818 vs 4 404: PDL) | force | 1.105 / 1.110 / 1.134 (min) | 3.5–13 % | refusal holds |

Every decision on the causal family has the measured sign on all four parts. Why (nsys, `prof/chains`): on the 4090 the
persistent kernel (40 registers, 128 × 256, 6 levels, 5 barrier arrivals) takes 15.7 µs a trip against its six members'
16.3 µs as separate kernels, and OFF's 7 launches a trip leave the device idle between them — the chain saves the
launches and its bodies do not slow. On the a100 the forced kernel takes 41.3 µs a trip against the members' 37.5 µs
(+10 %): the saved launches do not cover the slower bodies (+1.3–1.7 %).

**Formed, not realized (4090 block decode).** The host records `formed=1` (8 units) but the default-face module has no
cooperative launch and no persistent kernel (intent `calls`: `host.chain.coop=0 host.chain.formed=1`; PTX identical to
the OFF arm, only the host's launch order differs; the exact-face `.xref` build does carry it). Repro: `emit
pt_llm_blkdec_b1_l4096.mapal - --rewrite --target=rtx-4090 --host=x86-64-v3 --device-out=x.ll` → `; mapal.calls.host
chains=2 formed=1 …`, and the module's PTX has no grid-barrier code (0 `red.release.gpu` / `ld.acquire` / `membar`
lines; causal's module 40, the block decode's `.xref` 128). Its OFF arm is the shipped program (1.000). The same
count gap in every block (FFN) program whose chains form: 3090 forced block decode 3 formed / 1 cooperative launch,
h100 px block prefill 2 / 1, the 4090's px block decode / prefill twins 1 / 0 and 2 / 1.

**Across rounds: the refusals the body price added** (Mapal ms 7a2b67bf → now; rows whose emission is identical moved
0.98 / 0.94 / 0.96 / 1.06 geomean on 3090 / a100 / 4090 / h100): **decode_1024_256 3090 17.66 → 26.10 (bimodal: 20.4 /
26–27 ms), 4090 12.23 → 14.73 (1.20)** — the refusal loses; a100 causal / gen / prefill 0.98–0.99, decode B1 0.89 / 0.91,
4090 diamond 0.915, h100 **smlp 0.198 → 0.135 (0.68), firew_rowsum 0.088 → 0.046 (0.52)**, lsm_32_8192 0.58, mlp / mlp2
TF32 0.89 / 0.93 — those refusals win. Decode's formed chain (7a2b67bf) was one persistent kernel of 11.2 µs a trip
(3090; 4090 8.3 µs); refused, its three members (a horizontal unit and two GEMVs on 1 and 4 blocks) take 16.9 µs a trip
host-measured while the trip takes 19.9–25.5 µs: the separate form is host-issue-bound on these EPYC hosts (on the
5080's Threadripper the forced decode chain was 1.447 slower).

**The body term, measured on each part** (`calib.cu body`, three passes; derived = 4390 cycles at the profile clock):

| part | derived (ns) | measured 4 rounds (ns) | measured / derived | one round, 2 / 3 / 4 levels (ns) | 2 rounds (ns) | the price on the measured body (+ the measured boundaries) |
|---|---:|---:|---:|---|---:|---|
| rtx-3090 | 2 812 | 1 357 [1 310–1 588] | 0.48 | −119 / −167 / −230 | 214 | causal family stays; block decode forms 2 of 3 (its forced arm: null) |
| a100 | 3 113 | 2 699 [2 635–2 715] | 0.87 | −365 / 40 / 71 | 5 144 | unchanged; + boundaries (2 448 / 5 040 / 1 104 ns) → the causal family forms (forced: +1.3–1.7 %) |
| rtx-4090 | 1 742 | 3 294 [3 247–3 588] | 1.89 | −183 / −238 / −373 | 2 421 | causal family still forms (8 464 vs 8 640); + boundaries → refused (it wins 1.16) |
| h100 | 2 429 | −276 [−829 … −241] | −0.11 | −119 / −112 / 24 | −447 | smlp / firew_rowsum re-form (2 750 vs 3 444): the 7a2b67bf losers (1.55 / 2.3×) |

The derived value transfers to no part (measured / derived −0.11 … 1.89), and the measured one mis-decides too: alone on
the h100 (smlp / firew), with the measured boundaries also on the a100 and 4090. What separates the shapes is
**rounds**: every one-round chain measures −0.37 … +0.07 µs on all four parts, the grid-stride forms 0.2–5.1 µs on the
3090 / a100 / 4090 and none on the h100 (−0.45 / −0.28 µs, though its persistent kernel holds 5 blocks a SM against the
members' 8). The causal and decode chains are one round (≤ 128-block levels). **Missing facts** (a mixed result:
decode's chain wins on the 3090 / 4090, loses on the 5080): (1) the chain's widest level in rounds of the persistent
grid (site) — the body term per round, ≈ 0 for one round; (2) the host's issue time per launch (host property): the
price reads the device's dependent-launch latency, but on these hosts the separate form is bounded by the host; (3) the
phases' time in the merged grid (open since 7a2b67bf: the h100 smlp / firew phases, a100 members +10 % inside the
persistent kernel).

### Half precision: the bf16-weight rows against torch bf16

ms per decode step (16-step `iter ms` / 16). f32: the f32 program; bf16w: shipped; OFF: `--gpu-off=narrow-tile`;
torch bf16: the faster of `bf16` / `bf16-compile` (`c` compile, `e` eager) — bf16 weights, activations and KV cache.

| row | rtx-3090: f32 → bf16w (/f32) · torch bf16 → /tb | a100 | rtx-4090 | h100 |
|---|---|---|---|---|
| dec B1 L2048 | 0.277 → **0.185** (0.67) · 0.250 → **0.74** | 0.215 → **0.170** (0.79) · 0.155 → **1.09** | 0.242 → **0.154** (0.64) · 0.172 → **0.89** | 0.116 → **0.096** (0.83) · 0.293 → **0.33** |
| dec B1 L8192 | 0.378 → **0.287** (0.76) · 0.302 → **0.95** | 0.276 → **0.231** (0.84) · 0.198 → **1.16** | 0.344 → **0.258** (0.75) · 0.208 → **1.24** | 0.166 → **0.141** (0.85) · 0.168 → **0.84** |
| dec B8 L2048 | 1.023 → **0.838** (0.82) · 0.356 → **2.35** | 0.398 → **0.347** (0.87) · 0.229 → **1.52** | 0.497 → **0.417** (0.84) · 0.236 → **1.77** | 0.207 → **0.190** (0.92) · 0.257 → **0.74** |
| dec B8 L8192 | 1.801 → **1.615** (0.90) · 0.740 → **2.18** | 0.809 → **0.762** (0.94) · 0.423 e → **1.80** | 1.132 → **1.051** (0.93) · 0.460 → **2.29** | 0.468 → **0.454** (0.97) · 0.290 → **1.56** |
| dec B32 L2048 | 1.224 → **1.196** (0.98) · 0.725 → **1.65** | 0.859 → **0.867** (**1.01**) · 0.475 → **1.83** | 1.129 → **1.023** (0.91) · 0.454 → **2.25** | 0.471 → **0.491** (**1.04**) · 0.304 → **1.62** |
| dec B32 L8192 | 4.282 → **4.244** (0.99) · 2.412 → **1.76** | 2.535 → **2.541** (**1.00**) · 1.206 e → **2.11** | 3.765 → **3.661** (0.97) · 1.354 → **2.70** | 1.038 → **1.054** (**1.02**) · 0.562 → **1.88** |
| block dec B1 L4096 | 1.133 → **0.712** (0.63) · 0.679 → **1.05** | 0.959 → **0.790** (0.82) · 0.452 → **1.75** | 1.494 → **0.595** (0.40) · 0.572 → **1.04** | 0.569 → **0.352** (0.62) · 0.429 → **0.82** |
| geomean | /f32 **0.81**, /tb **1.41** | **0.89**, **1.57** | **0.75**, **1.61** | **0.88**, **0.97** |

The narrow tile's OFF arm (per-cell kernels) loses on every part: 1.19–3.40 (3090), 1.67–3.05 (a100), 1.50–1.70 (4090),
2.02–2.98 (h100). B1 reaches or passes torch bf16 on the 3090 / 4090 / h100 (the h100's torch B1 legs are slow on its
noisy host: 0.293 ms against the 5080's 0.143); B ≥ 8 trails 1.5–2.7× — attention over the f32 KV cache, as on the 5080
(ADR-0043 open 3). The 4090's block decode 0.40 is partly the f32 program's tuned pick (AUTO 1.126 × its rules pick,
open item 7 of 7a2b67bf, unchanged). **The mixed result: B32 on the a100 / h100, bf16w 1.00–1.04 of f32** (0.91–0.98 on
the 3090 / 4090). Traced projections (`probe_kernels`, ms a 16-step run, mean of 2 reps): h100 QKV 32 × 6144 × 4096 f32
1.004 → bf16w 1.192, O 0.632 → 0.750 — the f32 tile (`LDGSTS` 24, 1 024 FFMA a trip, 128 registers) moves its 100 MB a
step at 1.6 TB/s, the register-staged narrow tile (no `LDGSTS`, 256 FFMA a trip) its 50 MB at 0.68 TB/s: neither DRAM-
(3.2 TB/s) nor FFMA-bound (42 / 35 % of the measured FFMA rate) — latency-bound, the narrow tile with fewer loads in
flight; a100 QKV 2.167 → 2.170, O 1.394 → 1.508: FFMA-bound (62 % of its 64-lane FP32 peak at 32 rows), where the format
cannot help and the narrow tile is the less efficient one. The 4090: QKV 2.479 → 1.124, O 1.248 → 0.712 (−55 / −43 %;
the 50 MB of bf16 weights fit its 72 MB L2). **Missing realization / property**: the narrow panel's bytes in flight
against DRAM bytes per µs per SM (h100 25.4, a100 14.4 KB against the 4090's 7.9) — `cp.async` narrow panels (ADR-0043
open 4); and the site's FFMA per weight byte against the part's ridge (a100 12.4 FFMA / B): above it the format buys
nothing.

### Performance — Mapal / best competitor, 7a2b67bf → now

Best competitor as before (f32 legs for f32 programs, TF32 for `_tf32`, bf16 never). Geomeans over the rows with a
competitor in both rounds (`compare.py`).

| suite | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|
| README GPU suite (rows, wins now) | 0.69 → **0.90** (27, 10) | 0.71 → **1.00** (24, 8) | 0.73 → **0.79** (30, 19) | 1.06 → **1.05** (27, 7) |
| model-scale layer, f32 llm_* | 0.78 → **1.04** (12, 7) | 0.63 → **0.79** (7, 4) | 0.99 → **1.04** (12, 6) | 0.92 → **0.94** (12, 6) |
| projections, pj_* | 0.98 → **1.20** (28, 1) | 0.62 → **1.23** (28, 5) | 1.24 → **1.22** (28, 4) | 1.47 → **1.51** (28, 0) |
| GEMM cells 16×2048×7168 / 4096³ | 0.90 / 1.52 → **1.02 / 1.11** | 1.02 / 1.02 → **1.02 / 1.04** | 1.13 / 1.00 → **1.13 / 0.97** | 1.85 / 1.18 → **1.84 / 1.18** |
| bf16w decode vs torch bf16 (7, new) | **1.41** (2) | **1.57** (0) | **1.61** (1) | **0.97** (4) |

**What moved is the competitor, not Mapal.** Over the rows emitted identically in both rounds, Mapal's own ms moved
0.98 / 0.94 / 0.96 / 1.06 (geomean; the h100 within its noise), the best competitor's 0.77 / 0.58 / 0.98 / 1.04: the
3090 (200 → 350 W) and a100 (270 → 300 W) library legs no longer throttle, and faster hosts speed torch's launch-bound
loops (a100 causal torch-compile 236.6 → 72.6 ms on a Ryzen 7700; 4090 causal 0.09 → 0.18 with Mapal 18.50 → 18.38
ms). The capped boxes' last-round wins (3090 / a100 projections 0.62–0.98, mlp_2048 0.41–0.58) were the throttle.

| row (Mapal / best, now) | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---:|---:|---:|---:|
| causal / gen / prefill | 0.13 / 0.13 / 0.13 | 0.51 / 0.48 / 0.49 | 0.18 / 0.18 / 0.19 | 0.19 / 0.13 / 0.13 |
| decode_1024 (was) | 0.79 (0.55) | 0.96 (0.50) | 0.75 (0.37) | 0.73 (0.76) |
| smlp / firew_rowsum | 1.23 / 0.74 | 1.18 / 0.64 | 0.99 / 0.61 | **1.21 / 0.69** (was 1.78 / 1.33) |
| attn_1024_64 | 1.83 | 1.32 | 1.47 | 2.40 (null 8.9 %) |
| mlp_2048 / mlp_512 | 1.06 / 1.50 | 1.13 / 1.24 | 0.96 / 1.03 | 1.11 / 2.21 |
| decode B1 L2048 / B32 L2048 (f32) | 0.76 / 0.45 | 0.87 / 0.47 | 0.88 / 0.49 | 0.47 / 0.40 |
| block decode / prefill T2048 | 0.94 / 1.16 | 1.14 / 1.01 | 1.42 / 1.04 | 1.25 / 1.12 |
| prefill T8192 | 13.7 | 12.3 | 13.5 | 19.1 |
| lsm 1×131072 / 32×8192 | 11.2 / 2.69 | 10.8 / 1.67 | 8.39 / 2.44 | 5.81 / 2.52 |

Every row, leg and null for all four parts: `analysis/compare.md` in the raw directory.

### Evidence (intent check, `evidence-<part>.tsv`)

3090 45 247 checks / 2 471 mismatches, a100 50 773 / 4 691, 4090 58 090 / 3 819, h100 52 540 / 5 808. The new intents:
`narrow-tile` 446 / 452, 706 / 744, 852 / 896, 753 / 776 ok — the misses are `no spill` (8–244 B) and, on the a100, the
format-load check on `.xref` builds; most spilling kernels are table candidates, but **the block decode's narrow gate /
up site (1 × 14336 × 4096) ships a 255-register tile of 32-thread blocks spilling 216–244 B as its rules pick**
(candidate 0, launched: 132 µs a step on the 4090, as long as the naive GEMV beside it); `calls` 17 / 18, 10 / 10, 53 /
60, 20 / 22 — every miss a block (FFN) program with more chains formed than cooperative launches; `calls-kernel` **one
barrier arrival too many** on the exact-face chains (`.xref`, px: 8 arrivals for 8 levels, the intent expects levels −
1); `exp-hw` 264 / 444 … 345 / 528 — **every miss is an exact-face build** (`.xref`, px exact: the face that forbids the
hardware form; intent.py applies the intent there — stale scope, like `pdl`'s entry trigger, still 876 h100 misses).
`row-slices` / `exp-rescale` apply to no kernel on these parts: no flow unit is sliced (lsm one block a row). Largest
groups as before (fma-pairs 714–785, face-registers residency 971–1 015 a100 / h100, prefetch cover 213–543).

### Gap analysis — ranking update

1. **Missing dataflow fact (c): prefill T8192** — 12.3–19.1×, unchanged (one program, the most time).
2. **Under-filled grids against the SM count (b)**: lsm 1.7–11.2× (one block a row: no slices on any rented part),
   attention 1.3–2.4×, the skinny GEMM / m16 projections. Unchanged.
3. **`calls` pricing (b), up from 6**: the body term per chain from the 5080's grid-stride chain refuses decode's
   one-round chain on the 3090 / 4090 (1.2–1.5× slower than formed) while the causal family's decisions are right on
   all four; missing: rounds of the chain, the host's issue time per launch, the phases' time in the merged grid; and
   the 4090's block-decode chain formed but not realized.
4. **bf16 decode at B ≥ 8 (a / b, new)**: 1.5–2.7× torch bf16 — the f32 KV cache (attention), and on the high-bandwidth
   parts (a100 / h100) the register-staged narrow tile below DRAM bandwidth (B32 bf16w ≥ f32): `cp.async` narrow panels.
5. **GEMV / narrow-panel bytes in flight per SM (b)**: m1 projections 1.0–2.9× (h100 worst), and now the narrow
   tile — one property (DRAM B/µs/SM against the site's loads in flight) behind both.
6. **Hopper TMA / mbarrier / warp specialization (a)**: h100 TF32 rows 1.4–3.2×, unchanged (no library SASS yet).
7. **The 128-accumulator tile's rule (b)** and **conv window fill (b)**: unchanged (convmc f32 1.7–2.3×).

### Open items (missing facts)

1. `calls`: the chain's rounds over the persistent grid (body term per round), the host's issue time per launch, the
   phases' time in the merged grid; the 4090 block decode's formed-but-not-realized chain.
2. Half precision: `cp.async` narrow panels (bytes in flight on a100 / h100), the bf16 KV cache (B ≥ 8 gap), the
   block decode's spilling 255-register rules pick at the narrow m = 1 gate / up site.
3. intent.py: `exp-hw` and `calls-kernel` scoped onto exact-face builds; `pdl` entry trigger (876 h100).
4. Tune: the 4090 block decode AUTO 1.126 × rules (unchanged), h100 causal AUTO 1.374 × rules (noisy box).
5. `row-slices` on the rented parts: lsm stays one block a row (1.7–11×) — the first cooperative launch's price.
6. Unchanged from 7a2b67bf: `wide-fma` lanes / waves, `async` after `k-fill`, flow-unit grids against `sms`, Hopper
   TMA, `warp-fill` / `grid-share` / `min-blocks`.

Raw (local, not in git): `.claude/raw-evidence/cross-arch/60f0f8ce/<part>/` (results, run output, `calib-<part>.txt`
with the `body` and `exp` lines, `evidence-<part>.tsv`, profile archives with `prof/chains` nsys traces, tune, wisdom,
rest logs), `analysis/` (gates, props, body, arms, chainarms, gap, ms, bf16, compare, samecode, chains-decisions per
part), `scripts/` (build, stems, launch, watchdog, ops_nsys, rest, rest_now, analyze, body, chainarms, kdiff, ntrace,
drift, bf16, intents, cost, samecode, chains, offers).

## 19a8a44e (all competitors)

Branch `local-rtx-5080` @ `19a8a44e`: the long prefill as tile sites (`8354b6de`), caching-memory (`unchanged_plan`,
retained cells / gathered rows, `retain-same`, `cone-span`, `arena-fit`; `3cdb0b9d`, `834e64cb`, `e966e535`) and the
competitors step (`6dfd0d95`, `e3f841c1`: CUDA-graph `torch.compile`, SDPA backends, hand Triton, Liger,
FlashAttention-2/3, FlashInfer) + this round's tooling (`comp_build.sh` jobs from the container's CPU quota and memory
limit, FA3 on CUDA 13, `vast_part.sh` `OFFER_Q`, `matrix.py --skip`). Four vast.ai boxes in parallel, CUDA 13.0.2 devel
image, torch 2.11.0+cu130 (Triton 3.6.0), Liger 0.8.5, FlashInfer 0.7.5.post1 (the part's jit-cache wheel),
flash-attn v2.8.3.post2 built on each box (FA3: the H100). TensorRT-LLM not attempted (competitors.md). Matrix over all
five parts, every leg per row by family, same-precision rule: [matrix.md](../matrix.md).

**Bundle, one per part** (96 timed programs + the two GEMM cells, `LADDER=1`): the 60f0f8ce set (README suite, the f32 /
TF32 / bf16-weight layer programs, 24 projections, lsm, XREF) + the RTX 5080's three long-rows rows (`attn_2048_64`,
`attn_2048_128`, `softmax_rows_2048`: every part now times the 5080's 86 matrix rows) + the **12 KV-cache twins**
(explicit `kvx` / naive `kvn`, B 1/8/32 × L 2048/8192, block decode; XREF) and their four px differentials. Arms (the
same on every part, caching-memory.md §3): `unchanged` OFF (kvn B1 / B8), `retain-pieces` OFF (kvn B8), `cone-span`
OFF (kvx B8 / B32, prefill T8192), `arena-fit` OFF (prefill / block prefill T2048), `arena` OFF (prefill T2048, kvx B8),
`piece-delta` OFF (kvx B8); `retain-same` OFF equals the shipped emission on all 12 KV twins (no loop value there is the
same on every trip) and was dropped: 11 arms built a part. **On each box**: `box_run.sh` REPS=30 as before (probes ×5
ABBA with a traced process reporting `mapal_gpu_peak`), torch legs `TORCH_MIN=100` (every program finished: 96 / 96 on
every part), then FA2 (and FA3) built with nothing timed beside it, `comp_bench.py` `COMP_MIN=35`, `TIMING=0` (the
single-arm square-cell section off: the cells' rows come from the splitk section); `box_profile.sh` + this round's
`ops_nsys.sh`: torch's peak device bytes one process per (program, leg) on the 22 KV + f32 layer rows
(`torch_peak_proc.sh`'s legs), and nsys traces of the rows a new leg leads (`leads.py`: Mapal's process, the new leg,
the old best torch leg).

**Run notes.** (1) **a100, first box** (EPYC 7K62 host): from the second or third process of a program on, every Mapal
kernel ran ×1.27 slower (pj_m2048_n28672_k4096: processes 1–2 26.7 ms, 3–15 34.2 ms; cuBLAS right after 26.0 ms; the
60f0f8ce a100 26.86 ms), the whole Mapal column 1.19× its 60f0f8ce value against 1.08× for the gemm_lib legs — then the
host dropped the ssh connection 176 min in (comp_bench at the projections) and the box was destroyed. Its results
(streamed stdout, `a100-box1/`) are not in any table. A replacement A100 SXM4 on the 60f0f8ce a100's host (Ryzen 7 7700)
ran the whole bundle: every Mapal row within 15 % of 60f0f8ce but `llm_pre_t8192`, pj_m2048_n28672 26.86 ms in all 15
processes. (2) **H100 FA3**: `comp_build.sh`'s FA3 build failed in 29 s — its setup.py swaps a toolkit other than 12.8
for NVIDIA's nvcc 12.6 front end but compiles against the image's CUDA 13 headers, whose CCCL moved to `include/cccl`
(`fatal error: cuda/std/utility`). Rebuilt on the box with the image's nvcc 13.0 (93 s; setup.py's download skipped, now
`comp_build.sh`'s path on CUDA ≥ 13) beside `box_profile.sh`'s untimed sections; its 17 `fa3-bf16` legs timed alone on
the GPU afterwards. (3) **Power limits**: the 3090 box ran at 260 W and the 4090 at 330 W (60f0f8ce: 350 / 480 W): the
4090's cuBLAS m2048 × 28672 GEMM 8.29 → 11.06 ms. Every leg of a part shares its box, so the tables are like-for-like
within a part, not across rounds. (4) `nproc` showed the host's 96–256 threads against a 16–32-core quota: the FA builds
take their jobs from the cgroup (`comp_build.sh`), 8–15 jobs, FA2 892–1436 s.

### Boxes and cost

| part | GPU (driver, power limit) | host CPU | offer | $/h | hours | cost | status |
|---|---|---|---|---:|---:|---:|---|
| rtx-3090 | RTX 3090 24 GB (615.71.09, **260 W**) | EPYC 7742 | 61732988 | 0.270 | 3.13 | $0.84 | done |
| (a100 box 1) | A100-SXM4-40GB (595.91.07, 400 W) | EPYC 7K62 | 57328781 | 0.563 | 3.04 | $1.71 | Mapal kernels ×1.27 after warm-up; host lost at 176 min — not used |
| (a100 rest try) | — | Ryzen 7 7700 | 56853932 | 0.580 | 0.05 | $0.03 | stopped for the full rerun |
| a100 | A100-SXM4-40GB (595.91.07, 300 W) | Ryzen 7 7700 | 56853932 | 0.591 | 1.94 | $1.15 | done |
| rtx-4090 | RTX 4090 24 GB (580.126.09, **330 W**) | EPYC 7K62 | 55549899 | 0.390 | 3.10 | $1.21 | done |
| h100 | H100 80GB HBM3 SXM (595.91.07, 700 W) | Xeon Gold 6448Y | 29019361 | 2.360 | 2.80 | $6.61 | done (FA3 rebuilt on the box) |
| **total** | | | | | | **$11.55** | every `mapal-xarch-*` instance destroyed (checked by label) |

$/h is the instance's `dph_total`, hours from creation to destroy (`scripts/cost.py`). The H100 offer skipped the
60f0f8ce H100 host (noisy on host-bound rows). The account's credit fell $17.40 over the window (238.31 → 220.91);
other projects' instances bill through it.

### Value gates — all pass

| part | differential | probe gates (programs) | arms · candidates within D2/D3 | device-exact oracles | split programs | gemm_lib legs within | torch legs within | competitor legs within |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rtx-3090 | 59 / 59 | 96 / 96 | 834 / 834 | 79 / 79 | 8 / 8 | 112 / 112 | 622 / 650 | 132 / 136 |
| a100 | 78 / 78 | 96 / 96 | 871 / 871 | 79 / 79 | 8 / 8 | 112 / 112 | 626 / 650 | 133 / 136 |
| rtx-4090 | 69 / 69 | 96 / 96 | 881 / 881 | 79 / 79 | 8 / 8 | 112 / 112 | 628 / 650 | 132 / 136 |
| h100 | 78 / 78 | 96 / 96 | 865 / 865 | 79 / 79 | 8 / 8 | 112 / 112 | 626 / 650 | 149 / 153 |

No Mapal value failed on any part, face, arm or site candidate: the KV twins (XREF within D2, the four px twins bit for
bit on the exact face), the caching / memory arms, the long prefill's tile sites. **Competitor legs outside**: torch —
every one a TF32 leg at the 1e-3 gate (the f32 `mlp*_2048` / `rt_mlp2_*` rows' TF32 modes and the `_tf32` twins' f32
modes, 24 a part; the 3090 adds `compile-tf32` / `compile-cg-tf32` of `mlp_2048_tf32` / `mlp2_2048_tf32` at 3.7 /
4.7·10⁻³: Inductor's Triton TF32 template, the 5080's same miss), as in every round;
comp — `triton` FAILED on `lsm_1_131072` (all parts) and `lsm_4_32768` (not the a100: 164 KiB of shared memory) —
the tutorial softmax holds a row in one program (OutOfResources) — and `triton-tf32` on `mlp_2048_tf32` /
`mlp2_2048_tf32` at 3.7 / 4.7·10⁻³ against 10⁻³: the 5080's same four. Every CUDA-graph leg ran its graph (266 a part,
0 `cg=skipped`).

**Legs per part** (within / timed): torch eager / compile 282–285 / 296, CUDA graphs (`-cg`, `-ro`) 252–255 / 266, SDPA
backends 88 / 88, Liger 29 / 29, FA2 17 / 17, FlashInfer 17 / 17, hand Triton 69–70 / 73, FA3 17 / 17 (H100), TensorRT-
LLM `unavailable` (not installable on torch 2.11).

### Old competitor set → every leg

Geomean Mapal / best over the same 86 rows on every part; old set = the legs before the competitors step (torch eager /
compile / SDPA / bf16, cuBLAS, cuBLASLt, CUTLASS, hand CUDA), re-timed on the same box. Verdict: even = |ratio − 1| ≤
max(null, 5 %). The 5080 column is the competitors step's local run.

| part | old set | every leg | win / even / loss: old → every | rows whose best is a new leg (> 5 % faster) | Mapal wins that disappear |
|---|---:|---:|---|---|---|
| rtx-5080 | 0.980 | **1.001** | 18/24/44 → 18/22/46 | 28 (10: graphs 5, Liger 3, FlashInfer 1, Triton 1) | none |
| rtx-3090 | 1.008 | **1.054** | 36/9/41 → 32/9/45 | 49 (18: graphs 12, Triton 3, SDPA-efficient 2, Liger 1) | llm_pre_t2048 (sdpa-efficient), mlp_2048 (compile-ro, → even), mlp_2048_tf32 (compile-ro-tf32), pj_m128_n28672_k4096 (compile-ro, → even) |
| a100 | 1.157 | **1.199** | 16/15/55 → 16/13/57 | 40 (13: graphs 6, Triton 3, SDPA 2, Liger 2) | none |
| rtx-4090 | 1.031 | **1.081** | 30/12/44 → 24/13/49 | 41 (20: graphs 15, Liger 3, Triton 2) | llm_pre_t2048, llm_pre_t8192 (Liger), pj_m2048_n6144_k4096 (compile-ro), rt_mlp2_1024 (compile-cg); epi_4096, mlp2_2048 (compile-ro, → even) |
| h100 | 1.154 | **1.203** | 17/1/68 → 16/1/69 | 36 (18: graphs 14, Liger 3, Triton 1) | llm_dec_b8_l2048_bf16w (bf16-compile-cg) |

The new families that lead: **CUDA graphs** (`-cg` / `-ro`) on most changed rows — launch- and host-bound programs
(attention / lsm rows, the bf16 decode steps, small MLPs, the causal family's steps) — **Liger** (its RMSNorm / RoPE /
SwiGLU + SDPA's memory-efficient f32 attention) on every f32 prefill, **hand Triton** on every TF32 prefill (the fused-
attention tutorial) and a few TF32 projections / softmax rows, **cuDNN SDPA** on the 3090 / a100 bf16 B32 decode.
FlashAttention-2, FlashAttention-3 and FlashInfer are the fastest leg of **no row on the rented parts** (their decode
legs run the rest of the layer as torch eager ops: FA3 B32 L8192 0.706 ms a step on the H100 against bf16-compile-ro's
0.494).

| row: Mapal / best (leg) [old set] | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|
| causal_1024_256 | 0.40 compile-cg [0.34] | 0.14 compile-cg [0.12] | 0.50 compile-cg [0.49] | 0.10 compile-cg [0.09] | 0.18 compile-cg [0.16] |
| llm_pre_t8192 | 1.53 liger [0.95] | 1.58 liger [1.04] | 2.17 liger [1.14] | 1.61 liger [0.85] | 1.98 liger [1.17] |
| llm_pre_t2048 | 1.19 liger [1.02] | 1.08 sdpa-efficient [0.93] | 1.27 liger [1.03] | 1.14 liger [0.84] | 1.36 liger [1.12] |
| llm_pre_t2048_tf32 | 1.64 triton-tf32 [1.23] | 1.52 triton-tf32 [1.15] | 2.62 triton-tf32 [2.00] | 1.69 triton-tf32 [1.15] | 2.35 triton-tf32 [2.12] |
| llm_dec_b32_l8192 (f32) | 0.37 compile | 0.38 compile-ro | 0.39 compile-cg | 0.39 compile | 0.27 compile |
| llm_dec_b1_l2048_bf16w | 1.02 bf16-compile | 0.72 bf16-compile | 1.07 bf16-compile | 0.64 bf16-compile-cg [0.60] | 0.63 bf16-compile-cg [0.46] |
| llm_dec_b8_l2048_bf16w | 1.56 bf16-compile | 2.32 bf16-compile | 1.49 bf16-compile | 1.43 bf16-compile | 1.15 bf16-compile-cg [0.92] |
| llm_dec_b32_l8192_bf16w | 2.43 flashinfer [2.30] | 2.39 bf16-cudnn [1.32] | 2.61 bf16-cudnn [2.11] | 2.44 bf16-compile-cg [2.43] | 2.16 bf16-compile-ro [2.10] |
| lsm_1_131072 | 0.70 compile-cg [0.64] | 12.0 compile-cg [10.3] | 13.8 compile-cg [10.8] | 9.92 compile-cg [8.49] | 13.7 compile-cg [7.33] |
| attn_1024_64 | 1.36 compile-cg [1.32] | 2.18 compile-ro [1.71] | 1.17 sdpa-compile-cg [1.05] | 2.27 sdpa-compile-cg [1.91] | 2.10 compile-ro [1.73] |
| pj_m2048_n6144_k4096 | 1.02 compile-cg | 0.93 eager | 1.00 compile-cg | 1.06 compile-ro [0.83] | 1.12 eager |
| rt_mlp2_1024 | 1.01 compile | 0.85 compile-cg [0.84] | 1.59 compile-cg [1.25] | 1.07 compile-cg [0.92] | 1.09 compile-ro [1.06] |

Every row, leg and family per part: matrix.md ("What the new legs change", "Every competitor leg").

### Why (nsys, `prof/leads`)

- **Liger on the f32 prefills** (all parts, the 5080's cause): the H100's T8192 Liger leg is `fmha_cutlassF_f32_aligned_
  64x128_rf_sm80` 12.0 ms + two SIMT sgemms 6.7 ms each a call — the memory-efficient kernel streams key blocks with
  the scores on chip and skips the masked ones; Mapal's attention materializes the scores per KV-head group (long-
  prefill.md open 1 and 2). The same programs' **old-set** verdict moved the other way: Mapal's prefill T8192 went from
  12–19× (60f0f8ce) to 0.85–1.17× torch-compile (`8354b6de`: 1731 → 154 ms on the 3090, 1431 → 132 a100, 753 → 65
  4090, 828 → 52 H100).
- **CUDA graphs on launch-bound rows**: the H100 bf16 B8 decode's plain `bf16-compile` step is a handful of µs-scale
  kernels (cuDNN SDPA 26.9 µs, the fused RMSNorm + projection Triton template 19.0 µs, an `nvjet` GEMM 13.5 µs, …) with
  the host's launch gaps between them; the graph replays the step: 3.30 → 2.66 ms a run, 0.166 ms a step against Mapal's
  0.190. The lsm / attention rows likewise (lsm_1_131072 H100: compile 42.5 → compile-cg 22.7 µs); Mapal's lsm stays one
  block a row on the rented parts (open 2).
- **`compile-ro` beating cuBLAS on f32 GEMMs** (4090 pj_m2048_n6144 1.83 vs cuBLAS 2.31 ms, epi_4096 compile-ro 2.61
  vs compile 3.28; 3090 mlp_2048, pj_m128_n28672): the eager / compile call runs `cutlass_80_simt_sgemm_256x128_8x4`
  (cuBLAS's pick, 2.41 ms on the 4090 epi); inside the captured graph the GEMM is faster, but nsys traced graphs as one
  node (its default `--cuda-graph-trace=graph`), so which algorithm the capture selects is **open** (open 3). On the
  3090 host (EPYC 7742) the graph legs mostly save launch gaps: mlp_2048 eager 2.06 / compile 2.12 / graphs 1.63–1.65
  ms over the same two 735 µs GEMMs.
- **cuDNN SDPA on bf16 B32 decode** (3090 / a100): `cudnn_generated_fort_native_sdpa_sm80_flash_fprop` 736 µs a step
  against torch's own flash kernel 961 µs (a100); Mapal reads the f32 KV cache (ADR-0043 open 3).

### Mapal across rounds (60f0f8ce → now, same rows)

Geomean of Mapal ms now / before over the 83 rows both matrices hold: 3090 0.962, a100 0.971, 4090 1.041, h100 0.912;
the old set's best moved 1.057 / 0.998 / 1.092 / 0.946 (the 4090's 330 W cap slows both). Moves beyond 15 %: the
long prefill T8192 **0.06–0.09×** on all four; 3090 llm_blkpre_t2048 1.33, mm_4096³ 1.34 (260 W); 4090 host-bound
small programs 1.2–1.75 (decode_1024 14.7 → 25.8 ms, indep_512, mlp_512, lsm_32: an EPYC 7K62 host against
60f0f8ce's EPYC 9354P) and mm_4096³ 1.46 (330 W); h100 host-bound rows **0.56–0.84** (causal 38.4 → 21.4 ms, decode,
attention: the Xeon Gold host against the noisy 8480+ one).

### Peak device memory: Mapal vs torch

Mapal: `mapal_gpu_peak` of the shipped build (traced processes, the median); torch: one process per (program, leg), the
fastest f32 leg among those measured (compile on every row but the naive twins: naive-compile), peak from the leg's
inputs on (caching-memory.md's method). The 22 KV-twin and f32 layer rows:

| part | geomean Mapal / torch | Mapal lower | Mapal higher |
|---|---:|---:|---|
| rtx-3090 | 0.89 | 19 | prefill T2048 1.44, T8192 1.56, block prefill 1.24 |
| a100 | 0.88 | 19 | 1.46, 1.56, 1.25 |
| rtx-4090 | 0.86 | 19 | 1.43, 1.56, 1.24 |
| h100 | 0.86 | 19 | 1.40, 1.54, 1.22 |

Mapal's bytes are the same on every part within 1–3 % (one arena: explicit KV B32 L2048 2 216–2 230 MB against
torch-compile's 3 242–3 266, decode B32 L8192 2 285–2 299 against 3 278–3 302); torch's B1 decode peaks move with the
part's kernels (dec B1 L2048 183 MB on the 3090, 226–250 elsewhere). The prefills hold 1.2–1.6× torch's peak on every
part: the materialized score array (T2048: 512 MiB) — the 5080's finding, part-independent. Per row and part:
`analysis/peaks.md`.

### The automatic KV cache (naive twins) against torch

Mapal naive ms (/ torch naive, / torch HF-cache f32); explicit / HF-cache; peak MB naive / torch naive:

| shape | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|
| B1 L2048 | 9.0 (0.10, 1.16); 0.79; 286 / 313 | 9.2 (0.09, 1.48); 0.89; 286 / 322 | 5.7 (0.12, 0.98); 0.77; 285 / 361 | 4.0 (0.11, 0.98); 0.65; 291 / 363 |
| B1 L8192 | 27.2 (0.08, 1.68); 0.86; 650 / 748 | 28.4 (0.08, 1.94); 0.90; 651 / 748 | 13.8 (0.07, 1.40); 0.87; 651 / 748 | 11.4 (0.08, 1.40); 0.70; 655 / 772 |
| B8 L2048 | 59.4 (0.09, 1.85); 1.00; 1128 / 1321 | 53.8 (0.07, 2.00); 0.86; 1136 / 1321 | 24.5 (0.06, 1.25); 0.71; 1136 / 1321 | 21.4 (0.08, 1.63); 0.76; 1150 / 1345 |
| B8 L8192 | 249 (0.09, 2.15); 0.88; 4023 / 4780 | 202 (0.07, 2.13); 0.84; 4031 / 4772 | 109 (0.07, 1.53); 0.62; 4031 / 4780 | 78.5 (0.07, 1.81); 0.76; 4041 / 4804 |
| B32 L2048 | 240 (0.09, 2.05); 0.80; 4026 / 4778 | 203 (0.07, 2.13); 0.84; 4030 / 4778 | 107 (0.07, 1.54); 0.64; 4030 / 4778 | 78.1 (0.07, 1.78); 0.75; 4040 / 4802 |
| block B1 L4096 | 28.6 (0.16, 1.19); 0.92; 1080 / 1130 | 27.0 (0.14, 1.54); 1.11; 1080 / 1130 | 23.3 (0.23, 1.24); 1.10; 1081 / 1154 | 13.6 (0.19, 1.46); 1.15; 1085 / 1156 |

The naive program (no cache written) is **0.06–0.23× torch running the same naive program** at 0.79–0.96× its peak on
every part, and 0.98–2.15× torch's HF-cache program — even with it at B1 L2048 on the 4090 / H100, behind it as the
history grows (the first trip's full projection over the history: caching-memory.md open 1, demand). The explicit twin
takes 0.62–1.00× HF-cache torch on the decode shapes and 0.92–1.15× on the block decode. The 5080 (caching-memory.md):
naive 0.08–0.22× torch naive, 1.19–1.54× HF-cache.

### Caching / memory arms (time × ON / peak vs ON)

| workload | OFF | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|
| kvn B1 L2048 | unchanged | 14.4× / +0 | 10.5× / +0 | 6.34× / +0 | 9.72× / +0 |
| kvn B8 L2048 | unchanged | 19.6× / +0 | 14.5× / +0 | 15.3× / +0 | 14.8× / +0 |
| kvn B8 L2048 | retain-pieces | 19.2× / +0 | 14.2× / +0 | 14.9× / +0 | 14.3× / +0 |
| kvx B8 L2048 | piece-delta | 1.20× / +0 | 1.14× / +0 | 1.32× / +0 | 1.46× / +0 |
| kvx B8 L2048 | arena | 1.26× / −2 MB | 1.32× / −10 | 1.60× / −10 | 1.10× / −24 |
| prefill T2048 | arena | 2.10× / +0 | 1.47× / +0 | 3.32× / +0 | 1.34× / +0 |
| kvx B8 / B32 L2048, prefill T8192 | cone-span | 1.00× / +133 / +531 / +176 MB on every part | | | |
| prefill / block prefill T2048 | arena-fit | 1.00× / +129 MB on every part | | | |

Every mechanism keeps its sign on all four parts; the peak deltas are the same bytes everywhere (the arena's packing is
part-independent), the time of the OFF arms is not: without `arena` the allocations land in the timed region (1.10–3.32×,
largest on the slow-host 4090 / 3090 boxes), without the derived delta (`unchanged`) the projection over the history runs
whole every trip (6–20×).

### Evidence (intent check, `evidence-<part>.tsv`)

3090 71 161 checks / 3 789 mismatches, a100 81 328 / 7 727, 4090 84 765 / 5 126, h100 85 211 / 9 311. The caching /
memory intents hold on every part (`arena`, `entry-arena`, `carry-take`, `piece-delta`, `resident`, `retain-cell`,
`retain-pieces`: 0 mismatches); `by-ref` "trace allocs == predicted" 160 a part (the entry arena's one allocation where
two are predicted: caching-memory's standing item). The largest groups as in 60f0f8ce (`fma-pairs` contracting scalar
tiles 1 056–1 221, a100 / H100 `face-registers` residency 1 644 / 1 774, H100 `pdl` entry 982, `prefetch` cover
324–927); `calls` misses
are the block programs' formed-not-realized chains (4090 10 / 74), `exp-hw` the exact-face builds — both open since
60f0f8ce.

### Open items (missing facts)

1. **Flash-style streamed attention** (long-prefill open 1 / 2): now the largest LLM gap on every part — f32 prefill
   1.08–2.17× Liger / SDPA-efficient, TF32 prefill 1.52–2.62× the Triton tutorial, peak 1.2–1.6× torch.
2. **`row_iter_ns` is unmeasured on the rented profiles** (0: `row-slices` slices nothing; only the 5080's profile
   carries it, and `calib.cu` has no row-walk kind): lsm_1_131072 runs one block (227–348 µs at 512 elements a lane,
   slope 436–682 ns a lane-element over 32 → 512) against compile-cg's 23–26 µs, 9.9–13.8×. The property belongs in the
   calibration (a `row` kind), measured per part.
3. **The graph-captured cuBLAS algorithm** (4090 / 3090 GEMM rows): trace with `--cuda-graph-trace=node` to name the
   kernel the capture runs, then put the shape's best algorithm's geometry against Mapal's tile.
4. bf16 decode at B ≥ 8: 1.15–2.61× (the f32 KV cache, ADR-0043 open 3) — unchanged.
5. The a100 first box's ×1.27 Mapal-only slowdown after warm-up (cuBLAS unaffected, the box gone before a clock or
   ncu look): not reproduced on the replacement — recorded, open.
6. Unchanged from 60f0f8ce: `calls` (rounds, host issue time, phases), Hopper TMA / warp specialization (H100 TF32 rows
   1.3–3.1×, geomean 2.05), `cp.async` narrow panels.

Raw (local, not in git): `.claude/raw-evidence/cross-arch/19a8a44e/<part>/` (results, run output, `calib-<part>.txt`,
`evidence-<part>.tsv`, `prof-<part>.tgz` with `prof/leads` nsys traces and `prof/torch-peak-proc.txt`, torch / FA build
logs, wisdom, tune), `a100-box1/` (the lost box's streamed output), `analysis/` (gates, props, gap, ms, cm, cmtab, peaks,
compcmp, evidence-sum, rowiter per part), `scripts/` (stems, build, launch, watchdog, ops_nsys, leads, h100-patch/ the
FA3 rebuild, analyze, cost, compcmp, peaks, cmtab, evsum, legcount, ktop, keyrows, rowiter).

## 3a8e654f (final code, all competitors)

Branch `local-rtx-5080` @ `3a8e654f`, the final code: tensor cores (`fedc07ca`: 16-bit `mma.sync` tiles, the bf16 KV
cache as narrow radix sites, `narrow-async`, `staged-regs`), GEMV from the in-flight knee (`27258765`: `stream-fly`,
`vec-cells`, the bulk-copy engine as geometry), the open rules (`8aee53e7`: `calls` by rounds and phases on every part,
`wide-fma` by FP32 lanes and the wide grid, derived `row-slices` costs) and the review fixes (`3a8e654f`). The
19a8a44e round repeated on the same tooling (`60bc9bc7`) + this round's (`b7679aff`: the matrix's bf16-storage classes,
`TORCH_SKIP`, the FA3 build as rebuilt by hand). Four vast.ai boxes in parallel, CUDA 13.0.2 devel image, torch
2.11.0+cu130 (Triton 3.6.0), Liger 0.8.5, FlashInfer 0.7.5.post1, flash-attn v2.8.3.post2 built on every box, FA3 on
the H100 — **scripted this time** (`comp_build.sh`: the image's nvcc 13, 89 s); TensorRT-LLM not installable on torch
2.11 (competitors.md), skipped. The RTX 5080 on the same code with every leg, locally. Matrix over all five parts,
every leg by family, same-precision rule, the KV twins skipped (`--skip 'llm_kv[xn]_'`): [matrix.md](../matrix.md).

**Bundle, one per part** (130 timed programs + the two GEMM cells, `LADDER=1`): the 19a8a44e set (96: README suite +
long rows, f32 / TF32 / bf16-weight layers, 24 projections, lsm, the 12 KV-cache twins) + the **34 tensor-cores rows**
(`_bf16kv` × 7: weights and KV cache bf16, f32 compute; `_bf16` × 7: the same storage, `: bf16 | f32` products; prefill /
block prefill `_bf16w` / `_bf16`; 16 `pt_pj_*_bf16`), all against their exact face built for the device (`XREF`), and 16
px differentials (5 new). Their torch / comp legs are their twins' (`TORCH_SKIP`, matrix `legs_from`: a bf16 leg rounds
weights, cache and activations to bfloat16, so `_bf16kv`, `_bf16`, `_bf16w` and the f32 program run one bf16
computation). **Arms** — this round's rules, OFF, dropped where equal to the shipped emission (built 44 / 22 / 46 / 30
per part): `row-slices` (lsm ×3), `half-tile` (pj m128 / m2048 × 6144 × 4096 bf16), `narrow-radix` (decode B8 L2048
bf16kv), `stream-fly` and `vec-cells` (the six m = 1 projections), `wide-fma` (README suite + f32 projections), `calls`
(decode, causal, gen, rt_mlp2, block decode); the caching arms of 19a8a44e were not repeated. **On each box**: as
19a8a44e (`box_run.sh` REPS=30, probes ABBA × 5 with a traced process, `TORCH_MIN=100` — 96 / 96 programs on every
part —, the FA builds alone on the GPU, `comp_bench.py` `COMP_MIN` 30–35 — 64 / 64 —, `box_profile.sh` + `ops_nsys.sh`:
torch peaks one process per (program, leg), nsys of the rows a new leg leads) + a follow-up pushed to the running
3090 / 4090 / H100 (`decarms.sh`, finding 1).

**Run notes.** (1) **3090, first box** (EPYC 7742, 260 W, 19a8a44e's host): the host closed the ssh connection 20 min in
(probes, `pt_llm_dec_b32_l8192`), no results fetched — its streamed output (`rtx-3090-box1/`, 39 programs) is in no
table; replaced by an i9-9900 box (350 W). The A100 ran on 19a8a44e's replacement host (Ryzen 7 7700, 300 W); the 4090
on an EPYC 7702P at 450 W (19a8a44e: 330 W); the H100 on an EPYC 9454 at **500 W** (19a8a44e: 700 W; every H100 leg
moved: geomean best ms ×1.19, Mapal ×1.21 — like-for-like within the part, not across rounds). (2) **RTX 5080**: two
local bundles (bS: suite + long rows + lsm, 41; bM: model scale + tensor-cores rows + KV twins, 89) under the bench
protocol; both runs' torch sections met interference episodes (resS: 9 programs with legs at max > 2 × med, prefill
compile-cg 177 vs 19 ms; resM: decode legs 1.18–1.25× the competitors step's): every leg re-timed (`resS2` / `resM2`,
comp_run.sh at load < 3) — the re-timed legs equal the competitors step's (geomean 1.018 / 1.010); the matrix takes the
newest. (3) The in-flight knee measured on each box (`calib.cu bw fly`) equals the profile's: 3090 8 KiB, A100 16, 4090
4, H100 32.

### Boxes and cost

| part | GPU (driver, power limit) | host CPU | offer | $/h | hours | cost | status |
|---|---|---|---|---:|---:|---:|---|
| (3090 box 1) | RTX 3090 24 GB (615.71.09, 260 W) | EPYC 7742 | 61732988 | 0.270 | 0.38 | $0.10 | ssh closed by the host at 20 min — not used |
| rtx-3090 | RTX 3090 24 GB (610.57.04, 350 W) | i9-9900 | 62074391 | 0.250 | 2.68 | $0.67 | done |
| a100 | A100-SXM4-40GB (595.91.07, 300 W) | Ryzen 7 7700 | 56853932 | 0.591 | 2.11 | $1.25 | done |
| rtx-4090 | RTX 4090 24 GB (615.71.09, 450 W) | EPYC 7702P | 61950717 | 0.436 | 3.16 | $1.38 | done |
| h100 | H100 80GB HBM3 SXM (595.91.07, 500 W) | EPYC 9454 | 61868876 | 2.289 | 2.53 | $5.79 | done (FA3 by the script) |
| **total** | | | | | | **$9.19** | every `mapal-xarch-*` instance destroyed (checked by label) |

Budget $15 (watchdog at $14.5). The H100 skipped the noisy 8480+ host (743471), the A100 19a8a44e's slow-Mapal EPYC
7K62 host (410852). The account's credit fell $66.2 over the window (446.10 → 379.87): other projects' instances bill
through it.

### Value gates — all pass

| part | differential | probe gates (programs) | arms · candidates within D2/D3 | device-exact oracles | split programs | gemm_lib legs within | torch legs within | competitor legs within | B1 decode arms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rtx-5080 | 114 / 114 | 130 / 130 | 1140 / 1140 | 129 / 129 | 8 / 8 | 112 / 112 | 618 / 650 | 132 / 136 | — |
| rtx-3090 | 74 / 74 | 130 / 130 | 1118 / 1118 | 129 / 129 | 8 / 8 | 112 / 112 | 622 / 650 | 132 / 136 | 40 / 40 |
| a100 | 96 / 96 | 130 / 130 | 1098 / 1098 | 129 / 129 | 8 / 8 | 112 / 112 | 626 / 650 | 133 / 136 | — |
| rtx-4090 | 101 / 101 | 130 / 130 | 1179 / 1179 | 129 / 129 | 8 / 8 | 112 / 112 | 628 / 650 | 132 / 136 | 50 / 50 |
| h100 | 104 / 104 | 130 / 130 | 1129 / 1129 | 129 / 129 | 8 / 8 | 112 / 112 | 626 / 650 | 149 / 153 | 50 / 50 |

No Mapal value failed on any part, face, arm, site candidate or follow-up arm: the 34 tensor-cores rows within D3 at
`u_t = 2⁻⁸` (`_bf16`) / D2 (`_bf16kv`), the 16 px differentials (the five bf16 twins included). **Competitor legs
outside**, as every round: torch's TF32 legs at the 1e-3 gate (the f32 `mlp*_2048` / `rt_mlp2_*` rows' TF32 modes and
the `_tf32` twins' f32 modes); comp `triton` on `lsm_1_131072` (all parts) and `lsm_4_32768` (not the A100: 164 KiB of
shared memory) — OutOfResources —, `triton-tf32` on the two TF32 MLPs (3.7 / 4.7·10⁻³); the 5080 (16 GB) adds four
SDPA-math legs out of memory (decode B32 L8192, prefill T8192). Every CUDA-graph leg ran its graph (0 `cg=skipped`).
Legs per rented part (within / timed): torch 282–285 / 296, graphs 252–255 / 266, SDPA 88 / 88, Liger 29 / 29, FA2 17 /
17, FlashInfer 17 / 17, Triton 69–70 / 73, FA3 17 / 17 (H100).

### Performance — 19a8a44e → 3a8e654f, the 86 rows both rounds hold

Geomean Mapal / best competitor at the same precision (every leg), per class and family; win / even / loss (even =
|ratio − 1| ≤ max(null, 5 %)). The 5080's previous column is the competitors step's (f7a74cdb, the same Mapal bytes on
its 86 rows: Mapal ms geomean 1.000).

| part | all | f32 | tf32 | bf16w | GEMM | MLP | attention | LLM | conv | fan / loops | W / E / L |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rtx-5080 | 1.001 → **0.993** | 0.905 → **0.898** | 1.303 → **1.287** | 1.582 → **1.591** | 1.156 → **1.146** | 1.097 → **1.090** | 0.841 → **0.838** | 1.077 → **1.067** | 0.525 → **0.508** | 1.061 → **1.075** | 18/22/46 → 19/19/48 |
| rtx-3090 | 1.054 → **1.008** | 0.995 → **0.924** | 1.212 → **1.172** | 1.408 → **1.740** | 1.113 → **1.077** | 1.142 → **1.133** | 1.051 → **0.764** | 1.041 → **1.172** | 0.868 → **0.741** | 0.995 → **1.057** | 32/9/45 → 16/19/51 |
| a100 | 1.199 → **1.200** | 1.080 → **1.040** | 1.721 → **1.702** | 1.633 → **2.432** | 1.229 → **1.215** | 1.249 → **1.248** | 1.318 → **0.992** | 1.149 → **1.454** | 0.953 → **0.950** | 1.206 → **1.205** | 16/13/57 → 13/14/59 |
| rtx-4090 | 1.081 → **1.061** | 1.010 → **0.978** | 1.359 → **1.356** | 1.337 → **1.443** | 1.230 → **1.280** | 1.236 → **1.205** | 0.969 → **0.742** | 1.011 → **1.054** | 0.838 → **0.815** | 1.052 → **1.105** | 24/13/49 → 17/21/48 |
| h100 | 1.203 → **1.223** | 1.073 → **1.051** | 2.051 → **1.918** | 1.318 → **2.216** | 1.430 → **1.378** | 1.594 → **1.545** | 1.096 → **0.986** | 0.991 → **1.268** | 0.894 → **0.876** | 1.207 → **1.165** | 16/1/69 → 14/4/68 |

Without the six B1 decode rows (finding 1): 3090 1.072 → **0.996**, A100 1.209 → **1.145**, 4090 1.096 → **1.074**,
H100 1.242 → **1.179**, 5080 0.997 → 0.990 — the final code gains 2–7 % on the 80 other rows of every rented part; the
B1 decode regression takes it back on the 3090 / A100 / H100. The 3090's win → even shift is its box: the i9-9900 host
runs torch's host-bound legs faster (EPYC 7742 before), e.g. `indep_2048` compile-ro 1.85 → compile 1.49 ms.

**This round's new rows** (34, bf16 storage; geomean Mapal / best bf16 leg, W/E/L): 5080 **1.318** (6/2/26), 3090
1.495 (0/2/32), A100 2.613 (0/0/34), 4090 **1.228** (8/3/23), H100 2.923 (0/0/34). All 120 rows: 1.076 / 1.127 /
1.496 / 1.106 / 1.566 (W/E/L 25/21/74, 16/21/83, 13/14/93, 25/24/71, 14/4/102).

| tensor-cores rows (geomean) | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---:|---:|---:|---:|---:|
| pj m1 `_bf16` / best bf16 leg (the f32 row / best f32) | **0.72** (1.34) | 1.08 (1.04) | 1.25 (1.31) | **0.78** (1.63) | 1.50 (1.62) |
| pj m16 `_bf16` | 0.97 (1.00) | 1.29 (1.14) | 1.97 (1.03) | 1.00 (1.36) | 2.30 (1.16) |
| pj m128 `_bf16` | 1.36 (1.11) | 1.29 (1.04) | 2.06 (1.10) | 1.15 (1.18) | 2.98 (1.25) |
| pj m2048 `_bf16` | 1.22 (1.01) | 1.10 (1.06) | 1.97 (1.03) | 1.32 (1.03) | 2.85 (1.12) |
| decode `_bf16kv` / its `_bf16w` twin, Mapal ms (the bf16 KV cache) | **0.86** | **0.93** | 1.23 | **0.78** | 1.31 |
| decode `_bf16` / `_bf16kv`, Mapal ms (the bf16 face) | 1.00 | 1.00 | 1.00 | 1.00 | 1.01 |
| decode `_bf16kv` / best bf16 leg | 1.37 | 1.62 | 2.99 | 1.12 | 2.90 |
| prefill T2048 `_bf16` / best bf16 leg | 3.05 | 2.91 | 6.54 | 3.55 | 6.79 |
| block prefill T2048 `_bf16` / best bf16 leg | 1.68 | 1.55 | 2.87 | 1.83 | 3.61 |

### This round's rules on the hardware (same-box arms, OFF / ON; ON win where > 1 + max(null, 3 %))

| rule (OFF arm) | rtx-3090 | a100 | rtx-4090 | h100 | the replay / step predicted |
|---|---|---|---|---|---|
| `row-slices` (lsm 1×131072 / 4×32768 / 32×8192) | **11.0 / 3.56 / 1.41** | **14.3 / 3.36 / 1.10** | **8.14 / 2.43** / 0.97 null | **9.94 / 2.17 / 3.56** | open-rules (d): rented parts 1 block a row → 64 / 64 / 8–32 slices — **confirmed** |
| `calls` (decode / causal / gen; rt_mlp2, block decode) | **1.17 / 1.12 / 1.13**; null | null (formed) | **2.32 / 2.15 / 2.15**; null | decode refused (PDL); null | open-rules (a): decode formed on the 3090 / 4090 (W), the A100's causal chain (✗ open: now null), H100 refused — **confirmed** |
| `wide-fma` (suite + f32 projections) | 12 win / 8 null / 1 loss | not built (64 FP32 lanes) | 18 / 4 / 0 | 4 / 4 / 0 (the wide grid) | open-rules (1): by FP32 lanes and the H100's wave — **confirmed** |
| `half-tile` (pj m128 / m2048 × 6144 bf16) | **1.82 / 2.74** | **3.75 / 7.09** | **2.44 / 2.38** | **2.08 / 4.02** | tensor-cores: the 16-bit tile over the FMA tile on every part |
| `narrow-radix` (decode B8 L2048 bf16kv) | **1.52** | **1.80** | **1.32** | **2.36** | the bf16 cache as radix sites over per-cell kernels |
| `stream-fly` (six m = 1 projections) | 0 / 5 null / 0 | **5 / 0 / 0** (1.08–1.15) | 1 / 3 / **2 loss** (0.81, 0.92) | **6 / 0 / 0** (1.05–1.44) | gemv-tma: the knee in flight; 3090 knee = the warp fill (8 KiB): null as predicted; 4090 knee 4 KiB: mixed |
| `vec-cells` (the same six) | 0 / 6 null / 0 | 1 / 1 / **4 loss** (0.93–0.97) | 0 / 3 / **3 loss** (0.80–0.97) | 1 / 2 / **3 loss** (0.83–0.96) | gemv-tma (5080: LDG.128 wins) — **a mixed result** |

Per row and part: `analysis/arms.md`. The calls wins are the 4090's largest single moves: `decode_1024_256` 25.8 →
12.5 ms (0.82 → **0.41** chain-cache), `causal_1024_256` 0.10 compile-cg.

### New losers and what they point at

1. **B1 decode, 1.2–4.3× its 19a8a44e time on the 3090 / A100 / H100** (`dec_b1_l2048` H100 0.115 → 0.326 ms a step,
   0.57 → 1.57 torch-compile-cg; A100 0.86 → 1.88; 3090 0.76 → 0.94; the `_bf16w` and block-decode twins and the B1 KV
   twins alike; 4090 / 5080 unchanged). Measured on the boxes (`decarms.sh`: the B1 programs' OFF arms built after the
   bundle and run after `box_run`, ABBA × 5, every run within its bound; nsys kernel sums):

   | part | program | 19a8a44e | shipped (AUTO / rules) | `stream-fly` OFF | `staged-regs` OFF | `vec-cells` OFF |
   |---|---|---:|---:|---:|---:|---:|
   | h100 | dec_b1_l2048 | 1.841 | 5.208 / 5.208 | **2.025** (0.39) | **1.908** (0.37) | 5.143 (0.99) |
   | h100 | dec_b1_l2048_bf16w | 1.512 | 5.205 / 5.214 | **1.897** (0.36) | **1.546** (0.30) | 5.150 (0.99) |
   | rtx-3090 | dec_b1_l2048 | 4.416 | 5.195 / 5.196 | **4.287** (0.83) | = shipped | **4.317** (0.83) |
   | rtx-3090 | dec_b1_l2048_bf16w | 2.954 | 4.574 / 4.565 | **3.022** (0.66) | = shipped | 4.567 (1.00) |
   | rtx-4090 | dec_b1_l2048 | 3.798 | 3.887 / 3.955 | 3.934 (1.01) | 3.951 (1.02) | 3.940 (1.01) |

   (ms per 16-step run.) The time is one kernel: the O projection with its residual epilogue (`h1 = x + o·wo`, a
   `tile-post` site), H100 235 µs a call shipped against 25–29 µs in either OFF arm (3090 137 vs 79 µs). Its plan
   (`facts.json`): `kind naive, post 1, combine none, S 1, gx 16, fly_sm 8192, knee 32768` — **no k split: 16 blocks on
   132 SMs** (82 on the 3090), a quarter of the knee in flight on the 16 it reaches; OFF, the same site runs
   `combine splitk, S 16` (256 blocks). `stream-fly` cuts a streaming site into k partitions for the knee, but a post
   site's partitions must combine in a workspace before the epilogue (`epi_order`, open-rules (b)) and the rule leaves
   it unsplit. On the H100 / A100 `staged-regs` (the panel-alone budget) is what makes it a one-thread-per-cell site:
   it refuses the site's register-staged tile candidates, which ptxas fits in 62–70 registers without a spill; OFF
   either rule, the site keeps the warp-fill split (S 16). **Points at**: the post site's
   combine in `stream-fly`'s partition rule (a workspace split, or the warp-fill split it replaced, where atomics are
   not admissible) and the fill its intent names — "a block on every SM" — which `intent.py` checks only on non-post
   sites (the shipped kernel passes every intent; H100 `stream-fly` 441 / 441). The 4090 (knee 4 KiB) and the 5080 keep
   a split there. The H100's `decode_1024_256` (18.6 → 28.4 ms, kernel time 38.9 → 47.6 ms) lost its register-staged
   candidates the same way (`staged-regs`; no arm: open).
2. **The bf16 KV cache on the HBM parts**: `_bf16kv` takes 1.33–1.75× its f32-cache twin's time at B ≥ 8 on the A100 /
   H100 (dec B8 L8192 H100 0.838 vs 0.480 ms a step), 0.60–0.88× on the 3090 / 4090 / 5080; `narrow-radix` itself wins
   against per-cell kernels everywhere (1.3–2.4×). The halved cache bytes pay where the attention is DRAM-bound
   (GDDR, ~1 TB/s) and not where HBM feeds it (1.5–3.3 TB/s): the narrow path's widening (8-byte `cp.async`, PRMT per
   pair) and smaller panels cost more than the bytes save. **Points at**: the site's bytes against the part's FFMA
   balance (`ffma_per_dram_byte_pm`) deciding the narrow radix read — open (no SASS diff measured on the HBM parts).
3. **16-bit tiles on the A100 / H100**: pj `_bf16` m ≥ 16 1.97–2.98× torch bf16 (A100 / H100) against 0.97–1.36× on the
   GeForce parts. The H100 runs `mma.sync.m16n8k16` (128 a kernel, `cvt.rn.bf16x2` of the f32 `a` fragments) — no
   `wgmma` / TMA for 16-bit tiles, `bk` 16 from the f32 `a` panel (tensor-cores open); the A100's tensor rate is where
   the gap is widest. **Points at**: the matrix-op geometry with shared-memory descriptor operands (`Home::Smem`, the
   H100's `wgmma` in `Caps::matrix`) not mapped for half tiles, and the `a` operand's 4 bytes.
4. **`vec-cells`**: mixed on three parts (ON loses 3–20 % on 3–4 of six GEMVs on the A100 / 4090 / H100, null on the
   3090, wins on the 5080): the condition lumps cases a part property separates — candidates are the knee against the
   vector grid's warps in flight (A100 16 KiB, H100 32 KiB) and the 4090's 4 KiB knee; open.

### Gains that hold (cause on the hardware)

- **lsm rows** (`row-slices`, derived costs): `lsm_1_131072` 0.24–0.35 → 0.025–0.032 ms (9.9–13.8 → **0.96–1.69**×
  compile-cg), `lsm_4_32768` 2.8–5.0 → 1.1–2.0; the 5080 0.64 / 1.16 unchanged.
- **4090 / 3090 formed chains** (`calls` rounds): decode 0.41 / 0.75 chain-cache, causal 0.10 / 0.14 compile-cg.
- **Projections m = 1 on the A100 / H100** (`stream-fly`, standalone): A100 0.88–0.91× its 19a8a44e time on four of six,
  H100 OFF / ON 1.05–1.44× — still 1.5–2.2× cuBLAS / cuBLASLt at m = 1.
- **Attention family** (lsm + the long rows): 3090 1.051 → 0.764, A100 1.318 → 0.992, 4090 0.969 → 0.742, H100 1.096 →
  0.986.

Key rows (Mapal / best, leg; `analysis/keyrows.md`; every row, leg and family per part: matrix.md):

| row | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|
| lsm_1_131072 | 0.64 compile-ro | 0.96 compile-cg | 0.97 compile-cg | 1.18 compile-cg | 1.69 compile-cg |
| decode_1024_256 | 0.50 chain-cache | 0.75 chain-cache | 0.96 chain-cache | 0.41 chain-cache | 1.45 chain-cache |
| llm_dec_b1_l2048 | 0.88 compile-ro | 0.94 compile | 1.88 compile | 0.77 compile | 1.57 compile-cg |
| llm_blkdec_b1_l4096_bf16w | 1.16 bf16-compile | 1.68 bf16-compile | 4.57 bf16-compile-cg | 1.03 bf16-compile-cg | 5.05 bf16-compile-ro |
| llm_dec_b32_l8192 (f32) | 0.37 compile-cg | 0.39 compile | 0.39 compile-cg | 0.40 compile | 0.27 compile-cg |
| llm_dec_b32_l8192_bf16kv | 1.78 bf16-cudnn | 2.03 bf16-cudnn | 3.69 bf16-cudnn | 1.57 bf16-compile | 3.44 bf16-compile-cg |
| pj_m2048_n6144_k4096_bf16 | 1.23 bf16-compile-cg | 1.12 bf16-compile-ro | 1.99 bf16-compile-cg | 1.25 bf16-compile | 2.77 bf16 |
| llm_pre_t8192 | 1.49 sdpa-efficient | 1.51 liger | 2.17 liger | 1.84 liger | 1.81 liger |
| llm_pre_t2048_bf16 | 3.05 bf16-compile-cg | 2.91 bf16-compile | 6.54 bf16-compile-cg | 3.55 bf16-compile-cg | 6.79 bf16-compile-cg |

**Worst rows per part** (all 120): every part's top three are the bf16 prefills (`pre_t2048_bf16w` 4.3–14.5×,
`blkpre_t2048_bf16w` 3.3–12.5×, `pre_t2048_bf16` 2.9–6.8× torch bf16: attention over radix FMA tiles, flash-style
streaming open; the `_bf16w` rows compute f32); then A100 / H100 the B1 block decode (4.6 / 5.1×, finding 1) and the
bf16 decode at B32 (3.4–3.7×); 3090 lsm_32_8192 2.56 and the bf16 decode at B ≥ 8 (2.0–2.5×); 4090 lsm_32_8192 5.09
(`row-slices` null there), `convmc_tf32` 2.31, the GEMVs m ≤ 16 2.1–2.2× cuBLASLt; 5080 the bf16w decode at B ≥ 8
(2.1–2.5×), `pj_m128_n6144_tf32` 1.88.

### Peak device memory and the automatic KV cache

Peak: Mapal's `mapal_gpu_peak` (traced processes) against torch one process per (program, leg), the fastest f32 leg
measured (`torch_peak_proc.sh`'s legs; the 5080 locally), the 22 KV-twin and f32 layer rows:

| part | geomean Mapal / torch | Mapal lower | Mapal higher |
|---|---:|---:|---|
| rtx-5080 | 0.88 | 19 | block prefill 1.37, prefill T2048 1.60, T8192 1.60 |
| rtx-3090 | 0.89 | 19 | 1.24, 1.44, 1.56 |
| a100 | 0.87 | 19 | 1.24, 1.44, 1.56 |
| rtx-4090 | 0.84 | 19 | 1.24, 1.43, 1.56 |
| h100 | 0.85 | 19 | 1.22, 1.40, 1.54 |

Unchanged from 19a8a44e (the arena's bytes are part-independent; the prefills hold the materialized scores).

Naive twins (no cache in the program; Mapal's derived cache): Mapal naive ms (/ torch naive, / torch HF-cache f32);
explicit / HF-cache; peak MB naive / torch naive:

| shape | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|
| B1 L2048 | 6.4 (0.14, 1.24); 0.93; 286 / 313 | 9.8 (0.13, 1.31); 0.94; 286 / 313 | 13.3 (0.14, 2.14); 1.54; 286 / 322 | 5.6 (0.17, 0.96); 0.77; 285 / 353 | 8.7 (0.17, 1.89); 1.35; 290 / 356 |
| B1 L8192 | 16.2 (0.09, 1.55); 0.94; 651 / 748 | 27.9 (0.10, 1.97); 0.93; 650 / 748 | 32.5 (0.09, 2.22); 1.18; 650 / 748 | 13.3 (0.10, 1.39); 0.89; 651 / 748 | 19.6 (0.09, 1.95); 1.05; 654 / 772 |
| B8 L2048 | 28.7 (0.08, 1.39); 0.75; 1140 / 1321 | 59.2 (0.11, 2.07); 0.94; 1128 / 1321 | 53.7 (0.07, 1.99); 0.86; 1136 / 1313 | 23.7 (0.09, 1.23); 0.73; 1136 / 1321 | 32.7 (0.07, 1.82); 0.72; 1150 / 1345 |
| B8 L8192 | 110 (0.08, 1.39); 0.65; 4031 / 4780 | 186 (0.08, 1.83); 0.84; 4023 / 4780 | 202 (0.07, 2.13); 0.84; 4031 / 4772 | 94.4 (0.09, 1.34); 0.62; 4031 / 4780 | 112 (0.06, 1.68); 0.75; 4041 / 4804 |
| B32 L2048 | 110 (0.07, 1.43); 0.66; 4030 / 4778 | 177 (0.08, 1.77); 0.75; 4026 / 4778 | 203 (0.07, 2.13); 0.84; 4030 / 4778 | 91.6 (0.09, 1.36); 0.73; 4030 / 4778 | 114 (0.06, 1.71); 0.75; 4040 / 4802 |
| block B1 L4096 | 23.0 (0.20, 1.10); 0.93; 1080 / 1130 | 32.8 (0.21, 1.43); 1.17; 1079 / 1130 | 31.7 (0.16, 1.81); 1.37; 1080 / 1130 | 20.3 (0.27, 1.06); 0.94; 1080 / 1154 | 20.1 (0.18, 1.72); 1.30; 1084 / 1154 |

The naive program is **0.06–0.27× torch running it** at 0.78–0.98× its peak on every part, and 0.96–2.22× torch's
HF-cache program; the explicit twin 0.62–0.94× HF-cache at B ≥ 8. The B1 cells moved with finding 1 (A100 B1 L2048
naive 9.2 → 13.3 ms, explicit / HF 0.89 → 1.54; H100 4.0 → 8.7, 0.65 → 1.35); B ≥ 8 as in 19a8a44e (the H100 box's
500 W aside: 78.5 → 112 ms at B8 L8192).

### Evidence (intent check, `evidence-<part>.tsv`)

3090 101 276 checks / 3 643 mismatches, A100 108 911 / 8 837, 4090 115 602 / 4 479, H100 112 743 / 8 732, 5080 (bM)
106 226 / 4 670. This code's intents: `half-tile` 0 mismatches on every part (75–146 kernels), `vec-cells` 0,
`row-slices` 0 (38–143), `slice-launches` 0, `calls` / `calls-kernel` 0 (formed = cooperative launches), `pdl` 0 (H100
4 185), `bulk-stage` 0 (shipped off); `stream-fly` 16 / 18 / 49 misses on the 3090 / 4090 / 5080, none on the A100 /
H100 — and none on the post site of finding 1 (not checked there); `staged-regs` 150–291 misses a part (an 8 × 8
register-staged table candidate spilling 4–36 B, the step's standing item); `narrow-async` 0–42. The largest groups as
in 19a8a44e (A100 / H100 `face-registers` residency 2 151 / 2 275, `kmajor-a` 128-bit STS, `prefetch` cover).

### Open items (missing facts)

1. **`stream-fly` on `tile-post` sites** (finding 1): the partition rule must take the post site's workspace combine
   (or keep the warp-fill split) and fill the SMs; `intent.py`'s `stream-fly` check extended to post sites. Replay: the
   `decarm prog=` lines in `results-<part>.txt` and `<part>/decarms/`.
2. **`staged-regs`' panel-alone budget** refuses register-staged tile candidates ptxas fits without a spill (62–70
   registers: the H100 B1 decode's O projection; the H100 decode_1024_256's candidates): the budget lumps them with the
   spilling 8 × 8 panels.
3. **The narrow radix read against the part's balance** (finding 2) and **`vec-cells`' separating property**
   (finding 4).
4. **16-bit tiles on Hopper** (`wgmma`, TMA) and the f32 `a` panel (finding 3); the bf16 prefills' attention as flash-
   style streamed tiles (long-prefill open 1 / 2; T8192 1.49–2.17× Liger / SDPA-efficient, unchanged).
5. Unchanged: lsm_32_8192 on the 4090 (5.09×; `row-slices` null there), m ≤ 16 GEMVs 1.5–2.2× cuBLASLt, Hopper TF32
   (1.92), the graph-captured cuBLAS algorithm (19a8a44e open 3).

Raw (local, not in git): `.claude/raw-evidence/cross-arch/3a8e654f/<part>/` (results, run output, `calib-<part>.txt`,
`evidence-<part>.tsv`, `prof-<part>.tgz` with `prof/leads`, `prof/decarms` and `prof/torch-peak-proc.txt`, torch / FA
build logs, wisdom, tune), `<part>/decarms/` (the follow-up's kernel sums), `rtx-3090-box1/` (the lost box's streamed
output), `rtx-5080/` (resS / resM, the re-timed resS2 / resM2, torch-peak-proc.txt), `analysis/` (gates, props, gap,
ms, cm, cmtab, peaks, xcmp, arms, bf16tab, keyrows, evidence-sum), `scripts/` (stems, build, build_all, launch,
watchdog, ops_nsys, leads, local_bundle, local_run, local_rerun(M), arms_bundle, decarms, push_decarms, analyze, xcmp,
armtab, decsum, kfacts, bf16tab, keyrows, cost, peaks, cmtab, evsum, legcount).

## 5fcb401b (confirmation, all competitors)

Branch `local-rtx-5080` @ `5fcb401b`: the 3a8e654f round's losers attacked (`370637ed` regressions: the B1 decode's
O-projection split launched through its table — `post-table` —, the bf16 cache's 16-byte narrow loads, `vec-cells` by
the grid's spread — `vec-spread`), f32 on tensor cores (`dfdcb5e1`: `split-tile` tf32x3 / narrow1, `mma-combine`),
Hopper `wgmma` fed by TMA (`3092b270`: `wgmma-ws`, `tma-digits`, `ws-regs`), fp8 storage + leftovers (`ad451a13`: the
`_f8w` rows), the two open-problems sweeps (`unit-uniform`, `round-store`, `stream-mma`, `kmajor-b`, `warm-zero`,
`streamk-price`, `streamk-epi`) and the review fixes (`5fcb401b`: unmeasured parts priced from the 5080's rates). The
3a8e654f round repeated on its tooling — four vast.ai boxes, CUDA 13.0.2 devel image, torch 2.11.0+cu130 (Triton
3.6.0), Liger 0.8.5, FlashInfer 0.7.5.post1, flash-attn v2.8.3.post2 built on every box (`NVCC_THREADS=2`), FA3 on the
H100, TensorRT-LLM not installable on torch 2.11 (skipped) — and the RTX 5080 locally on the same code with every leg.
Matrix over all five parts on 5fcb401b (`--skip 'llm_kv[xn]_'`, the bf16kv / bf16 / f8w classes): [matrix.md](../matrix.md).

**Bundle, one per part** (137 timed programs + the two GEMM cells, `LADDER=1`): the 3a8e654f set (130) + the **7
`_f8w` decode rows** (f8e4m3 weights, per-channel scales, f32 compute; no fp8 leg exists: `(no leg)` in the matrix, their
`_bf16w` twin's bf16 leg a reference) and 18 px differentials (2 new). `TORCH_SKIP` adds `_f8w`. **Arms** — the rules
since 3a8e654f, OFF, built where the part's emission differs (77 / 69 / 52 / 78 on the 3090 / A100 / 4090 / H100):
`post-table` (B1 decode), `kmajor-b` (decode B ≥ 8 / L8192), `vec-spread` + `vec-cells` (the six m = 1 GEMVs),
`wgmma-ws` (the TF32 rows), `split-tile` (f32 GEMMs / MLPs), `mma-combine` (TF32 / bf16 tiles), `stream` +
`unit-uniform` (prefills, attention), `stream-mma` (bf16 prefills), `streamk-price` + `streamk-epi` (README suite),
`warm-zero` (m1 / m16 GEMVs), `round-store` (bf16 rows). **On each box** as 3a8e654f (`box_run.sh` REPS=30, probes
ABBA × 5 with a traced process, the tune pass now with `confirm_pick.py`, `TORCH_MIN=100`, the FA builds alone on the
GPU, `comp_bench.py`), then `ops_nsys.sh` + **`l2keep.cu`** (the L2's keep of a cyclic stream, `calib-l2keep` lines),
torch peaks one process per (program, leg). `calib.cu` (every kind incl. `bar`, `body`, `fly`, `matrix`), `exp_calib`
and `tma_calib` (H100; it does not build below sm_90) ran on every box.

**Run notes.** (1) **H100 market**: verified H100 SXM offers stood at $4.4–5.1/h when the round started; a $2.47
offer (EPYC 9684X, 700 W) appeared while the bundles built and was rented before its bundle finished (`vast_run.sh`
`WAIT_FILE`, preflight run by hand; 14 min held, $0.6). Its profile pass was cut to `ops_nsys.sh`'s calibration and
peaks (`box_profile_h100.sh`: the budget left ten minutes after `box_run`). (2) **RTX 4090, four rentals**: box 1
never reached running (22 min); box 2 stopped at the differential — the `_f8w` programs do not load on sm_89
(**finding 1**); the f8w programs were moved out of the 4090 bundle and box 3 (the same Ryzen 3900X host) ran 29 min
of probes until its GPU hung in `pt_llm_pre_t2048_bf16.off-stream` rep 4 (3 reps before ran 5.9 ms; the process
spinning 38 min, `nvidia-smi` in D state, no inter-block wait in that arm's kernels: the host) — destroyed by hand;
box 4 (Xeon E5-2640 v4, 20 threads, 450 W) ran the whole bundle; its `RUN_MIN` (175, set for the budget at the
relaunch) was held by stopping `vast_run`'s `timeout` (`extend_4090.sh`) once the H100 had ended; FA2 took 2 367 s
there; its profile pass hit 25 min before the tar (no `prof-rtx-4090.tgz`; peaks and `calib-l2keep` are in the results
file). Its torch section timed the bf16-storage twins too (130 programs; the matrix takes a row's own bf16 legs where
present — the same computation as its twin's). (3) **Power limits / hosts** (like-for-like within a part, not across
rounds): 3090 250 W (3a8e654f 350; sustained 1 424 MHz), A100 400 W (300), 4090 450 W (450; sustained 2 224 MHz on the
Broadwell host), H100 700 W (500; best ms ×0.82 against 3a8e654f's box). (4) **RTX 5080**: `local_run.sh` S, M, peaks
under the bench lock at load < 3 (resS 10 min, resM 37 min); no competitor leg with max > 2 × med or med > 1.3 × min
(257 / 522 legs): no re-timing this round.

### Boxes and cost

| part | GPU (driver, power limit) | host CPU | offer | $/h | hours | cost | status |
|---|---|---|---|---:|---:|---:|---|
| rtx-3090 | RTX 3090 24 GB (580.159.03, 250 W) | Ryzen 9 3950X | 44880349 | 0.249 | 2.96 | $0.74 | done |
| a100 | A100-SXM4-40GB (595.91.07, 400 W) | EPYC 9124 | 54218901 | 0.716 | 2.64 | $1.89 | done |
| (4090 box 1) | — | EPYC 7K62 | 59820949 | 0.473 | 0.31 | $0.15 | never reached running |
| (4090 box 2) | RTX 4090 24 GB | Ryzen 9 3900X | 51365776 | 0.484 | 0.08 | $0.04 | stopped at the differential: f8w PTX rejected on sm_89 |
| (4090 start) | — | — | 54464714 | 0.428 | 0.00 | $0.00 | not started |
| (4090 box 3) | RTX 4090 24 GB | Ryzen 9 3900X | 51365776 | 0.484 | 1.15 | $0.56 | GPU hung mid-probes — destroyed |
| rtx-4090 | RTX 4090 24 GB (595.91.07, 450 W) | Xeon E5-2640 v4 | 57617079 | 0.422 | 3.80 | $1.60 | done (`RUN_MIN` held; profile cut) |
| h100 | H100 80GB HBM3 SXM (615.71.09, 700 W) | EPYC 9684X | 56442966 | 2.578 | 3.55 | $9.14 | done (rented while its bundle built) |
| **total** | | | | | | **$14.12** | every `mapal-xarch-*` instance destroyed (checked by label) |

Budget $15 (watchdog at $14.5, raised to $14.9 after the second 4090 replacement). The account's credit fell $97.3 over
the window (424.55 → 327.22): other projects' instances bill through it.

### Value gates

| part | differential | probe gates (programs) | arms · candidates within D2/D3 | device-exact oracles | split programs | gemm_lib legs within | torch legs within | competitor legs within |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rtx-5080 | 128 / 128 | 137 / 137 | 1205 / 1205 | 136 / 136 | 8 / 8 | 112 / 112 | 618 / 650 | 131 / 136 |
| rtx-3090 | 80 / 80 | 137 / 137 | 1198 / 1198 | 136 / 136 | 8 / 8 | 112 / 112 | 622 / 650 | 132 / 136 |
| a100 | 64 / 64 | 137 / 137 | 944 / 944 | 136 / 136 | 8 / 8 | 112 / 112 | 626 / 650 | 133 / 136 |
| rtx-4090 | 104 / 104 (box 2: 10 f8w FAIL) | 130 / 130 | 1157 / 1157 | 129 / 129 | 8 / 8 | 112 / 112 | 969 / 994 | 132 / 136 |
| h100 | 113 / 113 | 137 / 137 | 1227 / 1227 | 136 / 136 | 8 / 8 | 112 / 112 | 626 / 650 | 149 / 153 |

**Finding 1 — a Mapal failure: the `_f8w` programs do not load on the RTX 4090 (sm_89).** Every f8w PTX (25: the 7
rows, their arms and the 2 px twins) carries `.version 7.8 .target sm_89` and `cvt.rn.f16x2.e4m3x2` (76 in
`pt_llm_dec_b1_l2048_f8w`); `cuModuleLoadData` fails ("a PTX JIT compilation failed", process exit 102). Repro:
`ptxas -arch=sm_89 target/xarch-4090-f8w/rtx-4090/pt_llm_dec_b1_l2048_f8w.ptx` → `Feature 'cvt.f16x2.e4m3x2 on sm_89'
requires PTX ISA .version 8.1 or later` (157 errors; `rtx-4090/f8w-ptxas-sm89.txt`). `Caps::cvt_f8` (cc ≥ 89) is right
about the instruction, but the capability also carries the PTX ISA that encodes it: `llc -mcpu=sm_89` defaults to
PTX 7.8 (sm_120: 8.7, sm_90a: 8.0 — why the 5080 and H100 load the same text). **Points at**: the capability's PTX ISA
(`+ptx81` on the device module wherever the cvt is emitted for sm_89), and a preflight check on `ptxas` errors — the
bundle's evidence pass logged all 25 errors at build time and nothing read them. The 3090 / A100 widen in software
(no cvt) and pass. The 4090's f8w rows are `-` in the matrix.

No other Mapal value failed on any part, face, arm or candidate. **Harness, not Mapal**: the A100's GEMM cell
`mm_4096x4096x4096` (now `split-tile` tf32x3) fails `tile_bench`'s gate — `bit_exact_vs_reference=NO
within_bound=yes` (maxrel 3.0·10⁻⁴ vs f64): the gate holds non-atomic arms to the in-order fold's bits, and a split
realization's bits are the default face's own (D2 holds); the matrix drops that cell. Competitor legs outside: torch's
TF32 legs at 1e-3 (as every round), `triton` on `lsm_1_131072` / `lsm_4_32768` (OutOfResources), `triton-tf32` on
the TF32 MLPs, the 5080's Liger decode B32 L8192 (out of memory, 16 GB) and four SDPA-math legs.

### Performance — 3a8e654f → 5fcb401b (every leg, same precision; even = |ratio − 1| ≤ max(null, 5 %))

| part | all | f32 | tf32 | bf16w | bf16kv | bf16 | GEMM | MLP | attention | LLM | conv | fan / loops | W / E / L |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rtx-5080 | 1.076 → **0.979** | 0.898 → **0.869** | 1.287 → **1.186** | 1.938 → **1.662** | 1.375 → **1.043** | 1.195 → **0.982** | 1.108 → **1.045** | 1.090 → **1.054** | 0.838 → 0.858 | 1.312 → **1.036** | 0.508 → 0.529 | 1.075 → **1.032** | 25/21/74 → 30/39/51 |
| rtx-3090 | 1.127 → **1.009** | 0.924 → **0.893** | 1.172 → **1.148** | 2.066 → **1.498** | 1.623 → **1.000** | 1.356 → **1.134** | 1.113 → **1.099** | 1.133 → **1.119** | 0.764 → 0.827 | 1.458 → **1.004** | 0.741 → 0.833 | 1.057 → **0.977** | 16/21/83 → 43/17/60 |
| a100 | 1.501 → **1.242** | 1.040 → 1.104 | 1.702 → **1.646** | 3.559 → **2.156** | 2.990 → **1.124** | 2.206 → **1.228** | 1.391 → **1.288** | 1.248 → 1.537 | 0.992 → 1.029 | 2.285 → **1.232** | 0.950 → 0.978 | 1.205 → 1.494 | 13/14/92 → 24/9/86 |
| rtx-4090 | 1.106 → **0.948** | 0.978 → **0.868** | 1.356 → 1.366 | 1.807 → **1.379** | 1.122 → **0.826** | 1.146 → **0.898** | 1.195 → **1.104** | 1.205 → **1.155** | 0.742 → **0.600** | 1.217 → **0.908** | 0.815 → **0.739** | 1.105 → 1.102 | 25/24/71 → 36/27/57 |
| h100 | 1.566 → **1.318** | 1.051 → **1.006** | 1.918 → 2.625 | 3.168 → **2.194** | 2.900 → **1.103** | 2.634 → **1.646** | 1.647 → **1.623** | 1.545 → 1.617 | 0.986 → **0.935** | 2.101 → **1.215** | 0.876 → 0.942 | 1.165 → 1.201 | 14/4/102 → 22/3/95 |

Geomean Mapal / best over the rows both matrices hold (120; the A100 119: its GEMM cell's gate, above); bold = better.
The 86 rows of the competitors step / the 34 bf16-storage rows: 5080 0.993 → **0.945** / 1.318 → **1.068**, 3090
1.008 → **0.952** / 1.495 → **1.171**, A100 1.200 → **1.188** / 2.613 → **1.388**, 4090 1.061 → **0.946** / 1.228 →
**0.952**, H100 1.223 → **1.181** / 2.923 → **1.740**. Mapal ms geomean new / old: 0.916 / 0.955 / 0.883 / 0.944 /
0.690 (the H100 box at 700 W: its best legs ×0.82). The A100's f32 / MLP / fan rows and the H100's TF32 rows moved the
other way: findings 2 and 3.

### The rules on the hardware (same-box arms; cell: geomean OFF / ON, > 1 = the shipped ON faster, [range], W / null / L)

| rule (OFF arm) | rtx-3090 | a100 | rtx-4090 | h100 | predicted (step) | verdict |
|---|---|---|---|---|---|---|
| `post-table` (B1 decode, 6 rows) | 1.63 [1.25–3.42] 6/0/0 | 3.54 [2.65–6.13] 3/0/0 | = shipped | 3.12 [1.37–7.04] 6/0/0 | the planned split launched on the 3090 / A100 / H100, the 4090 unchanged (regressions 1) | **confirmed** |
| `vec-spread` (m = 1 GEMVs) | = shipped | 1.07 [1.03–1.14] 5/0/0 | = shipped | 1.13 [1.07–1.29] 5/0/0 | the scalar grid on the HBM parts' losers (regressions 3) | **confirmed** |
| `vec-cells` (the same) | 1.01 0/6/0 | 1.00 0/1/0 | 1.02 3/3/0 | 0.99 0/3/0 | no loss left | **confirmed** (3a8e654f: 3–4 losses a part) |
| `kmajor-b` (decode B ≥ 8, L8192) | 1.30 [1.22–1.36] 5/0/0 | 1.19 [1.02–1.41] 4/1/0 | 1.21 [1.07–1.32] 5/0/0 | 1.13 [1.04–1.22] 5/0/0 | 5080 1.45 (B32 L8192 bf16kv, sweep-attention-2) | **confirmed** |
| `stream-mma` (bf16 prefills) | 1.24 [1.11–1.38] | 2.54 [1.78–3.64] | 1.32 [1.13–1.55] | 1.86 [1.45–2.39] | 5080 1.12–1.43 | **confirmed** |
| `round-store` (12 bf16 rows) | 1.02 4/7/1 | 1.12 7/5/0 | 1.08 6/6/0 | 1.17 11/1/0 | 5080 1.01–1.19, m128 n28672 0.97 | **confirmed** (3090 one loss: m128 n6144 0.96) |
| `warm-zero` (m1 / m16 GEMVs) | 1.02 2/10/0 | 1.06 8/4/0 | 1.07 6/6/0 | 1.06 7/5/0 | 5080 1.02–1.30 on one-call atomic splits | **confirmed** (largest on m1 n2048 / n4096: 1.04–1.23) |
| `streamk-price` | 1.07 3/2/0 | = shipped | = shipped | 1.76 1/0/0 | where the part's rates price an under-filled site | **confirmed** |
| `streamk-epi` (epilogue groups) | 0.99 [0.74–1.13] 2/8/2 | = shipped | = shipped | 1.02 3/5/0 | 5080 1.04–1.10 | **mixed: 3090 loses** (finding 6) |
| `stream` (prefills, attention) | 1.28 [0.96–2.04] 7/2/1 | 1.16 [0.43–3.27] 4/1/5 | 1.60 [0.82–3.37] 8/1/1 | 1.09 [0.65–2.47] 4/4/2 | priced on every part (review-fixes-2 §2), timed here first | **mixed: HBM parts lose at long rows** (finding 4) |
| `unit-uniform` (prefill T8192) | 1.50 | 0.75 | 2.08 | 0.65 | 5080 1.85 | **mixed** (finding 4) |
| `mma-combine` (TF32 / bf16 tiles) | 1.09 [0.93–1.49] 3/0/3 | 1.03 [0.84–1.34] 3/0/3 | 0.97 [0.86–1.09] 1/1/2 | 0.90 0/0/1 | 5080 1.02–1.30 (ON / OFF 0.77–0.985) | **mixed on every part** (finding 5) |
| `split-tile` (f32 GEMMs, tf32x3) | = shipped | **0.70 [0.56–0.79] 0/0/12** | = shipped | = shipped | the rate rule takes tf32x3 on the A100 only (unmeasured) | **refuted** (finding 2) |
| `wgmma-ws` (TF32 rows) | — | — | — | **0.76 [0.39–1.02] 0/3/9** | H100 1.01–1.25 (hopper.md) | **refuted in program time; confirmed per kernel** (finding 3) |

Per row and part: `analysis/arms.md`, summary `analysis/armsum.md`.

**The regressions' predictions, measured** (`analysis/feat.md`):

| | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|
| `llm_dec_b1_l2048` ms/step 3a8e654f → now (/ best now) | 0.239 → 0.240 (0.92) | 0.326 → **0.269** (0.77) | 0.471 → **0.214** (0.80) | 0.243 → 0.270 (0.66) | 0.326 → **0.110** (0.57) |
| `llm_blkdec_b1_l4096_bf16w` ms/step | 0.635 → 0.627 | 1.135 → **0.706** | 2.062 → **0.792** | 0.647 → 0.708 | 1.525 → **0.493** |
| bf16 KV cache: `_bf16kv` / f32-cache twin, Mapal ms, geomean B ≥ 8 | 0.78 → **0.67** | 0.89 → **0.74** | 1.40 → **0.82** | 0.69 → 0.68 | 1.56 → **0.85** |

The B1 decode is back below its 19a8a44e time on the 3090 / A100 / H100 (H100 0.115 then, 0.110 now) and wins against
torch on every part; the 4090's 1.05–1.13 × 3a8e654f is its box (a Xeon E5-2640 v4 host against an EPYC 7702P;
`post-table` = shipped there). The bf16 cache now pays on the HBM parts too (the scores kernel's 32 16-byte loads a thread), as predicted.

### Properties: measured on the box against what the compiler prices (`analysis/props-<part>.md`)

| property (priced: measured in profile.rs, else derived) | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|
| sustained clock MHz (profile) | 2 783 (2 450) | 1 424 (1 561) | 1 409 (1 410) | 2 224 (2 520) | 1 806 (1 807) |
| DRAM read GB/s (plan rate) | 917 (890) | 892 (891) | 1 487 (1 488) | 855 (981) | 3 180 (3 186) |
| in-flight knee KiB a SM (`fly`) | 8 (8) | 8 (8) | 16 (16) | 4 (4) | 32 (32) |
| `bar_ns`, 256 threads (derived: 41 cycles at the clock) | 16.9 (17) | **33.5** (26) | **38.1** (29) | **22.5** (16) | 21.1 (22) |
| `coop_body_cycles` (derived 4 390) | 2 879 (4 390) | 2 102 | 4 048 | **7 292** | 348 |
| panel-load latency, 573 × idle DRAM latency / the 5080's (derived: 573 at the clock) | 573 | 515 (899) | 662 (995) | 409 (380) | 585 (776) |
| `mma.sync` TF32 / 16-bit flops a SM a clock | 256 / 512 | 256 / 512 | 1 012 / 2 024 (1 024 / 2 048) | 256 / 512 | **1 351 / 2 702 (532 / 1 064)** |
| tf32x3 issue pattern (`split3`) flops a SM a clock | 254 | 252 | 824 | 252 | 1 079 |
| `tma_copy_ns` (box ≤ 4 KiB, depth ≥ 2) | 197 (197) | — | — | — | 158 (159) |
| `l2_keep_bytes` (`l2keep.cu`; profile: nominal L2 on the rented parts) | — (profile 1.025 × L2) | **1.03 × L2** (review: the plateau's midpoint; < 1.0 read against the derived L2 rate) | **< 1.0 × L2** | 1.015 × L2 | **< 1.0 × L2** |
| `pageable_pm` (`pinned` below 900) | 685 (695) | 951 (0) | 794 (0) | 741 (0) | 833 (0) |

**Would a measured value flip a rule?** Replayed through the emitter (`replay_flip.sh`: the part's profile against the
same profile with the measured value as `--cost=`): the **H100's measured `mma.sync` rates flip `split_rate_pays` for
tf32x3** (1 351 ≥ 1.3 · 3 · 256) — `pj_m2048_n4096_k4096`, `mlp_2048`, `pj_m128_n6144_k4096` change emission — toward
the realization that loses 12 / 12 on the A100 (finding 2): measuring the rate without fixing the split's staging
would move the H100's f32 GEMMs onto it. `pinned` flips in the rule table on the A100 / 4090 / H100 (741–833 < 900) but
moves no suite program (3 replayed per part). The measured `bar_ns` / panel latency / `l2_keep_bytes` (A100 stream,
prefill and attention), `coop_body_cycles` (the `calls` programs on all four), the GeForce parts' sustained clocks and
the 4090's DRAM rate move no emission. Two derivations are off by a stable factor: `bar_ns` (Ampere / Ada measure 54–57
cycles, the 5080 / H100 41–45) and the latency (DRAM latency is ns, not SM cycles: the clock-scaled value is 1.3–1.8×
the measured one); the L2 keeps less than its nominal size on the 3090 / A100 / H100 (a 1.0 × L2 cyclic stream re-reads
at the DRAM rate), which the tune's cold flag reads as resident.

### New losers and what they point at

2. **`split-tile` on the A100 (tf32x3 on f32 GEMMs): OFF / ON 0.56–0.79 on all 12 programs** (f32 class 1.040 →
   1.104, MLP 1.248 → 1.537, prefill T8192 132 → 209 ms with `stream` below). The rate rule's side holds — `calib.cu
   matrix` measures the split's issue at 824 flops a SM a clock (2.1× the FFMA's useful rate) — the kernel does not:
   `pj_m2048_n4096_k4096` runs 7.09 ms against the FFMA tile's 5.21 (19 % of the TF32 peak over three products). SASS:
   the split kernel's hot loop has **40 LDG and 0 LDGSTS** (the intent "16-byte cp.async panels" fails on 113 A100
   kernels, `evidence-a100.tsv`): its operand panels are register-staged (LDG → split → STS) at 2 stages, 3 blocks
   an SM, 145 registers, where the FFMA tile streams them by `cp.async`. *(Corrected by bugs-properties /
   tensor-cores-2: the 40 LDG are the guard's cold FFMA loop, which the checker took as hot by counting an HMMA like
   an FFMA; the k loop holds 8 LDGSTS and 0 LDG and is issue-bound — 1 612 instructions a 96-HMMA panel.)* **Points at**: the split form's operand
   staging (split on the fragment after an async panel copy) and `split_rate_pays` reading peak rates only — the
   split tile's realized rate at its staging is the property the rule lacks; the H100 flip above makes it urgent.
3. **`wgmma-ws` on the H100: per kernel it wins, per program it loses.** Traced kernel alone (`probe_kernels`): ON
   0.073 / 0.096 + 0.094 / 0.092 / 1.845 ms against OFF 0.076 / 0.134 + 0.129 / 0.133 / 2.000 (m128 × 4096, the MLP,
   m128 × 6144, m2048 × 28672: 1.04–1.45×, hopper.md's 1.01–1.25 confirmed on the device). The program's iteration:
   ON 0.212 / 0.350 / 0.235 against OFF 0.083 / 0.265 / 0.145 ms — each ws call carries 120–270 µs outside its kernel
   (OFF 2–7 µs),
   varying by repetition (m2048 × 4096: ON 0.435–0.553 ms, the same build's null copy 0.417–0.476, OFF 0.303–0.311;
   kernel 0.287 vs 0.296), absent in hopper.md's session (ON iteration = kernel). TF32 class 1.918 → 2.625. cuBLAS
   runs `sm90_xmma_gemm_…_tilesize128x256x32_warpgroupsize2x1x1` on a persistent 66 × 2 grid (ktrace): our tile at 256
   CTAs. **Points at**: the steady-state cost of a TMA kernel's launch (2 `__grid_constant__` tensor maps, 1 block a
   SM at 196 KiB, `setmaxnreg`) that `MAPAL_GPU_TRACE`'s synchronized timing does not see — an nsys timeline of the
   untraced loop names it; no `DeviceCost` property prices a TMA launch today.
4. **`stream` at long rows on the HBM parts**: A100 `llm_pre_t8192` 0.75, `attn_2048_128` 0.43, prefill T2048 0.90–0.96;
   H100 `llm_pre_t8192` 0.65, `attn_2048_64` 0.90 — while the GDDR parts win (3090 7/2/1, 4090 8/1/1: T8192 1.47 /
   2.01) — a mixed result. The A100 takes row blocks of 8–32 on every streamed site (`b8xkb32:rt1g8t64:cp:s8` for
   attention d = 128, `b16xkb64` at T8192, where the 3090 / 4090 take 64 and the H100 128). Its evidence line prices
   the d = 128 stream at 0.256 ms against the split realization's 0.373 (`ns_stream` / `ns_split`); measured 0.732
   against OFF's 0.317 — the stream priced 2.9× fast, the split 1.2× slow (T8192: priced 6.7 vs 13.2 ms a launch,
   measured program ON 209 / OFF 157 ms; H100 2.2 vs 4.3, measured 86.6 / 56.5). The measured `bar_ns` / latency / L2
   keep do not move it (replayed). The stream side is priced on derived plan rates on the A100 / H100 (no measured
   FMA-tile fraction, LSU issue or stream rate; review-fixes-2 §2) and the materialized side's DRAM traffic is cheap
   where `ffma_per_dram_byte` is 12.6 / 18 (A100 / H100) against 38–82 on the GDDR parts. **Points at**: a measured
   stream rate per part (a `calib.cu` stream kind: the row-block kernel's issue at the part's lanes and LSU) and the
   two sides' price at the part's FFMA / DRAM balance. Llama prefill T8192 against Liger: 3090 1.06, 4090 0.93, 5080
   0.86, A100 3.15, H100 3.35.
5. **`mma-combine` mixed on every part**: the TF32 MLP cuts lose on the 3090 / A100 (0.84–0.95), `pj_m128_n6144` cuts
   win on the 3090 (1.27–1.49) and lose on the 4090 / H100 (0.86–0.92). The wave model `⌈blocks·S/slots⌉·(P/S +
   stages)` counts waves, not the combine's `red` traffic (`atomic_ops_per_ns`, derived on the 3090 / A100 / H100) nor
   the part's tile efficiency — the property that separates the parts.
6. **`streamk-epi` on the 3090**: `loop_4096` 0.74, `epi_4096` 0.94 (H100 3 / 5 / 0, 5080 wins): the ticket's shared
   tiles' partials against a 6 MiB L2 (75 KiB a SM, `grid-share` off) — unpriced.
7. Unchanged / structural: the `_bf16w` prefills (f32 products on bf16 weights) 2.9–3.5× torch bf16 on the GeForce
   parts, 11.8–17.5× on the A100 / H100 (tensor-core bf16 against FFMA); the H100's 16-bit tiles on `mma.sync` (pj
   bf16 m2048 2.62×; no `wgmma` for 16-bit); `lsm_32_8192` 2.3–2.9×; `stream-fly`'s post-site intents now checked on
   the HBM parts (A100 112, H100 226 misses: "a block on every SM", "the knee in flight").

Key rows (Mapal / best, leg; `analysis/keyrows.md`; every row: matrix.md):

| row | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|
| lsm_1_131072 | 0.78 compile-cg | 1.07 compile-ro | 1.00 compile-cg | 0.87 compile-ro | 1.45 compile-cg |
| decode_1024_256 | 0.53 chain-cache | 0.74 chain-cache | 0.79 chain-cache | 0.35 chain-cache | 1.41 chain-cache |
| llm_dec_b1_l2048 | 0.92 compile-ro | 0.77 compile | 0.80 compile | 0.66 compile | 0.57 compile-cg |
| llm_dec_b32_l8192 (f32) | 0.29 compile-cg | 0.28 compile-ro | 0.30 compile-ro | 0.31 compile | 0.26 compile-cg |
| llm_dec_b32_l8192_bf16kv | 1.16 bf16-compile | 1.18 bf16-cudnn | 1.45 bf16-cudnn | 1.17 bf16-compile | 1.63 bf16-compile-cg |
| pj_m2048_n6144_k4096_bf16 | 1.06 bf16-compile-ro | 1.02 bf16-compile-ro | 1.48 bf16-compile-cg | 1.13 bf16-compile-cg | 2.54 bf16-compile |
| pj_m2048_n4096_k4096_tf32 | 1.15 lt-tf32-fresh | 1.04 compile-ro-tf32 | 1.47 lt-tf32 | 1.13 lt-tf32 | 2.99 lt-tf32-fresh |
| llm_pre_t8192 | 0.86 liger | 1.06 liger | 3.15 liger | 0.93 liger | 3.35 liger |
| llm_pre_t2048_bf16 | 1.37 bf16-compile-cg | 1.26 bf16-compile-ro | 1.94 bf16-compile-cg | 1.41 bf16-compile-cg | 3.53 bf16-compile-ro |

### Peak device memory, the automatic KV cache, the fp8 rows

Peak (`mapal_gpu_peak` against torch one process per (program, leg), the fastest f32 leg measured; 22 KV-twin and f32
layer rows): geomean Mapal / torch 5080 0.77 (21 lower; `dec_b1_l2048` 1.03), 3090 0.76, A100 0.75, 4090 0.73, H100 0.73
— **lower on all 22 rows of every rented part** (3a8e654f: 0.84–0.89 with the prefills at 1.22–1.60): prefill T2048
0.33–0.42, T8192 0.41, block prefill 0.64–0.73 — the streamed attention stores no score array (`analysis/peaks*.md`).

Naive twins (Mapal naive ms (/ torch naive, / torch HF-cache f32); explicit / HF-cache; peak MB naive / torch naive):

| shape | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|
| B1 L2048 | 6.4 (0.13, 0.92); 0.68; 286 / 313 | 8.9 (0.10, 1.16); 0.80; 286 / 313 | 11.2 (0.08, 1.71); 0.94; 286 / 322 | 6.1 (0.18, 0.85); 0.70; 285 / 353 | 3.9 (0.11, 0.95); 0.64; 291 / 356 |
| B1 L8192 | 15.9 (0.09, 1.24); 0.73; 651 / 748 | 26.8 (0.08, 1.72); 0.85; 651 / 748 | 35.5 (0.08, 2.25); 1.01; 651 / 748 | 13.7 (0.11, 1.20); 0.77; 651 / 748 | 11.0 (0.08, 1.35); 0.64; 655 / 772 |
| B8 L2048 | 27.9 (0.08, 1.12); 0.59; 1140 / 1321 | 66.5 (0.09, 2.03); 0.88; 1128 / 1321 | 70.2 (0.07, 1.98); 0.85; 1136 / 1321 | 23.6 (0.09, 1.08); 0.61; 1136 / 1321 | 21.3 (0.08, 1.63); 0.73; 1150 / 1345 |
| B8 L8192 | 112 (0.07, 1.17); 0.50; 4031 / 4780 | 241 (0.09, 2.03); 0.82; 4023 / 4780 | 285 (0.06, 2.18); 0.76; 4031 / 4780 | 86.8 (0.09, 1.12); 0.55; 4031 / 4780 | 78.3 (0.07, 1.80); 0.72; 4041 / 4804 |
| B32 L2048 | 107 (0.07, 1.16); 0.52; 4030 / 4778 | 229 (0.08, 1.95); 0.77; 4026 / 4778 | 284 (0.06, 2.38); 0.86; 4026 / 4778 | 88.0 (0.09, 1.21); 0.59; 4030 / 4778 | 78.4 (0.07, 1.76); 0.71; 4040 / 4802 |
| block B1 L4096 | 23.0 (0.18, 1.13); 0.95; 1080 / 1130 | 28.5 (0.15, 1.20); 0.94; 1080 / 1130 | 30.7 (0.13, 1.55); 1.04; 1080 / 1130 | 22.6 (0.30, 1.00); 0.89; 1080 / 1154 | 13.3 (0.18, 1.45); 1.14; 1085 / 1154 |

The naive program is 0.06–0.30× torch running it at 0.80–0.96× its peak, 0.85–2.38× torch's HF-cache program; the
explicit twin 0.50–0.88× HF-cache at B ≥ 8 (`kmajor-b`: the 5080's B8 L8192 0.65 → 0.50). The A100 naive rows grew
(B8 L8192 202 → 285 ms: its projections on `split-tile`, finding 2).

fp8 rows (`analysis/f8tab.md`; Mapal ms a step / its `_bf16w` twin, geomean): 5080 0.74, 3090 1.03, A100 0.92, H100
0.82 (B1 L2048: 5080 0.36, the others 0.87–0.92); against the bf16 reference leg 1.00–1.26 (B1 rows 0.37–0.90, B32 1.33–2.10). 4090:
finding 1.

### Evidence (intent check, `evidence-<part>.tsv`)

5080 (bM) 120 191 checks / 5 755 mismatches, 3090 130 172 / 4 960, A100 116 728 / 8 670, 4090 138 067 / 5 167, H100
140 971 / 9 034. This code's intents: `wgmma-ws` 81 / 81, `ws-regs` 22 / 22 (H100), `stream` 270–397 a part, `stream-mma`,
`unit-uniform`, `mma-combine`, `vec-cells` and `round-store` 0 misses (H100 one); `kmajor-b` 0 on the 3090 / A100 /
4090 (5080 12, H100 45); `streamk-epi` 13–14 on the A100 / H100; **`split-tile` 113 / 749 on the A100** (finding 2:
no LDGSTS in the hot loop); `stream-fly` 34–226 (the post sites now checked on the HBM parts); the standing groups
(`kmajor-a` 128-bit STS / LDS, `staged-regs` spills, `face-registers`, `prefetch` cover) as in 3a8e654f.

### Open items (missing facts)

1. The capability's PTX ISA (`cvt.f16x2.e4m3x2` on sm_89 needs PTX 8.1) and a preflight on `ptxas` errors (finding 1).
2. The split realization's operand staging and its realized rate in `split_rate_pays` — before the H100's measured
   `mma.sync` rate (1 351 / 2 702, profile 532 / 1 064) is put in its profile (finding 2).
3. The TMA kernel's steady-state launch cost (`wgmma-ws`: 120–270 µs a call outside the kernel on the H100; finding 3).
4. A measured stream rate per part and the two sides' price at the FFMA / DRAM balance (finding 4); `mma-combine`'s
   combine traffic (5); `streamk-epi`'s shared tiles against the L2 (6).
5. Derivations to replace by measurement: `bar_ns` (Ampere / Ada 54–57 cycles), the panel latency (ns, not cycles),
   `l2_keep_bytes` (< 1.0 × L2 on the A100 / H100; the 3090's sweep read off its plateau: 1.03 × L2, review —
   `l2_keep_of_sweep`), `coop_body_cycles` (348–7 292 measured, 4 390 derived).

Status after the confirmation-fixes merge (`local-rtx-5080/integration.md` § Confirmation-fixes merge): 1 closed
(`Ptx`, `Caps::ptx`, the `ptxas` preflight: bugs-properties); 2 closed (the split tile priced at its issue,
`split_issue_pm`; the integer cut: A100 tf32x3 1.43–1.54× the FFMA tile on 11 of its 12 programs — tensor-cores-2;
review: `pj_m128_n6144_k4096` still loses 0.809 under the shipped tf32x3, so 2 stays **open** for that site, and the
merge's whole 2048-row tiles are timed whole only at 2048³); 3 closed (the first TMA
launch off the clock, recipes from the module text: hopper-launch); 4 closed for the A100 (the stream's interleaved
chunks and half-serial price at `lds_bcast_pm`: attention-chains; `mma-combine` priced in ns: bugs-properties;
`streamk-epi`'s L2 rule: bugs-properties), the H100's long-row stream open; 5 measured (`bar_cycles`, `lat_ns`,
`coop_body_cycles`), `l2_keep_bytes` below 1.0 × L2 open. New open: `l2-gate`'s wave footprint, `ws-persist`'s
separating fact (hopper-launch), the GeForce bf16x2n tile's realized pipe fraction (tensor-cores-2).

Raw (local, not in git): `.claude/raw-evidence/cross-arch/5fcb401b/<part>/` (results, run output, `calib-<part>.txt`,
`evidence-<part>.tsv`, `prof-<part>.tgz`, torch / FA build logs, wisdom, tune; 4090: the f8w-abort and gpu-hang boxes'
output, `f8w-ptxas-sm89.txt`, `extend.log`), `rtx-5080/` (resS, resM, torch-peak-proc.txt, the calibrations),
`analysis/` (gates, props, gap, ms, cm, cmtab, peaks, xcmp, arms, armsum, feat, f8tab, bf16tab, keyrows, rowsets,
evidence-sum, cost, matrix-3a8e654f.json), `scripts/` (stems, build, build_all, launch, watchdog, h100_wait,
ready_when_built, box_profile_h100, extend_4090, ops_nsys, leads, local_bundle, local_run, analyze, props5, kplan,
replay_stream, replay_flip, armtab, armsum, xcmp, feat, f8tab, rowsets, gates, peaks, cmtab, evsum, legcount, cost).

## f1d3a04f (confirmation, all competitors)

Branch `local-rtx-5080` @ `f1d3a04f`: the integration of the four fix branches over 5fcb401b's losers —
`fix/bugs-properties` (the PTX ISA as a capability + a `ptxas` preflight, measured matrix / barrier / latency / body
properties, `split-issue`, `cut-price`, `streamk-epi`'s L2 rule), `fix/tensor-cores` (split tiles priced at their issue,
bf16 weights × f32 activations on the bf16 warp product: `split-least`), `fix/hopper-launch` (the TMA kernel's first
launch off the clock: `tma-warm`, recipes from the module text, a map cache; the persistent walk `ws-persist`, shipped
off), `fix/attention-chains` (interleaved consumer chunks, the stream window priced at the broadcast-load rate
`lds_bcast_pm`) and `39cd2c87` (a split tile's k cut at the rate its panels realize); `f1d3a04f`'s emission is
`39cd2c87`'s. 5fcb401b's tooling (CUDA 13.0.2 devel image, torch 2.11.0+cu130 / Triton 3.6.0, Liger 0.8.5, FlashInfer
0.7.5.post1, flash-attn v2.8.3.post2 built on every box with `NVCC_THREADS=2`, FA3 on the H100; TensorRT-LLM does not
install on torch 2.11 — skipped) with one change of order: **the competitor legs ran first** (phase A, while the
integrated commit was being built; the harness at 8ff66d34 — `torch_bench.py`, `comp_bench.py`, `triton_kernels.py`,
`comp_build.sh`, `gen_tiles.py` are byte-identical at `f1d3a04f`, so no leg was re-run; their value gate read the
5fcb401b round's printed values of the same part), then the Mapal legs on `f1d3a04f` (phase C: `box_run.sh` with
`TORCH=0 COMP=0`, then `box_profile.sh`), the gap captures, and the fix follow-up (phase D). Matrix over all five parts on
`f1d3a04f` (`--skip 'llm_kv[xn]_'`): [matrix.md](../matrix.md); the RTX 5080 column is the integration step's full suite
with every competitor leg (`local-rtx-5080/confirmation-fixes/res{S,M}` @ `39cd2c87`, emission-identical).

**Bundle, one per part**: 5fcb401b's 137 timed programs + the two GEMM cells (`LADDER=1`), the 4090 again with its
seven `_f8w` rows. **Arms** — the rungs the fix branches touched, OFF (ON for `ws-persist`), kept where the part's
emission differs beyond kernel names: `split-tile` / `split-issue` (the f32 suite, projections m16–m2048, prefills),
`split-least` (`_bf16w`), `wgmma-ws` / `tma-warm` / `ws-persist` (TF32), `stream` / `unit-uniform` (prefills,
attention), `mma-combine` (TF32 / bf16 tiles), `streamk-epi` (README suite). **Calibration** on every box: `calib.cu`
every kind incl. the new `lds bcast`, `bar`, `body`, `fly`, `matrix`; `exp_calib`; `tma_calib` (H100); `mma_calib`
(builds from sm_89: its fp8 `mma`); `l2keep.cu` from 0.25 × L2.

**Run notes — like-for-like within a part only.** (1) The **H100** box runs at **500 W** (5fcb401b's 700 W; FFMA
sustained 1 703 vs 1 807 MHz): the same `wgmma` kernels take 1.56–2.09× their 5fcb401b time and cuBLAS's TF32 legs
~2× (`lt-tf32`, 2 048 × 28 672 × 4 096: 2.59 vs 1.28 ms). (2) The **RTX 4090** box is **clock-locked at 1.2 GHz by its
host** (nvidia-smi 1 200 MHz under load, no throttle reason; FFMA sustained 984 MHz vs 2 224 on 5fcb401b's box): every
leg ×1.47, ratios like-for-like, but not a 4090's compute : DRAM balance — emitted at the measured clock 70 of the 137
programs change (replay, below). (3) The **A100** box (EPYC 7532, driver 595.71.05) runs the shipped kernels (same
hashes) 1.25–1.45× slower than 5fcb401b's A100 while its cuBLAS legs run 0.80–0.91× — finding 3. (4) The first 3090 offer
was an Ivy Bridge Xeon (no AVX2 for the x86-64-v3 host objects): destroyed before it started; the offer filter now skips
pre-Haswell hosts. (5) `ncu` is refused in vast containers (`ERR_NVGPUCTRPERM`) on all four boxes.

### Boxes and cost

| part | GPU (driver, power limit) | host CPU | offer | $/h | hours | cost | status |
|---|---|---|---|---:|---:|---:|---|
| (3090 box 1) | — | Xeon E5-2660 v2 | 62318004 | 0.222 | 0.03 | $0.01 | no AVX2: destroyed before start |
| rtx-3090 | RTX 3090 24 GB (595.99.02, 270 W) | EPYC 7702P | 60544709 | 0.271 | 4.15 | $1.12 | done (A, C, gaps 40, D) |
| a100 | A100-SXM4-40GB (595.71.05, 400 W) | EPYC 7532 | 51341859 | 0.892 | 4.59 | $4.09 | done (A, C, gaps 40, D) |
| rtx-4090 | RTX 4090 24 GB (595.84, 450 W; clocks locked 1.2 GHz) | EPYC 7713 | 47076390 | 0.421 | 4.19 | $1.77 | done (A, C, gaps 40; D: no change) |
| h100 | H100 80GB HBM3 SXM (595.91.07, 500 W) | EPYC 9454 | 61868876 | 2.289 | 3.79 | $8.68 | done (A, C, gaps 40, D) |
| **total** | | | | | | **$15.67** | every `mapal-xarch-*` destroyed (checked by label) |

Budget $40 (raised to $55 for the gap captures); watchdog at 362 min a box.

### Value gates — no Mapal failure on any part

| part | differential | probe gates (programs) | arms · candidates within D2/D3 | device-exact oracles | split programs | gemm_lib legs within | torch legs within | competitor legs within |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rtx-5080 | 128 / 128 | 137 / 137 | 1205 / 1205 | 136 / 136 | 8 / 8 | 112 / 112 | 619 / 650 | 131 / 136 |
| rtx-3090 | 80 / 80 | 137 / 137 | 1161 / 1161 | 136 / 136 | 8 / 8 | 112 / 112 | 622 / 650 | 129 / 133 |
| a100 | 64 / 64 | 137 / 137 | 986 / 986 | 136 / 136 | 8 / 8 | 112 / 112 | 626 / 650 | 133 / 136 |
| rtx-4090 | 118 / 118 | 137 / 137 | 1220 / 1220 | 136 / 136 | 8 / 8 | 112 / 112 | 624 / 650 | 132 / 136 |
| h100 | 113 / 113 | 137 / 137 | 1194 / 1194 | 136 / 136 | 8 / 8 | 112 / 112 | 626 / 650 | 149 / 153 |

**5fcb401b finding 1 closed**: every PTX of the four bundles passed `ptxas` at build (the module declares `+ptx81` where
it emits `cvt.f16x2.e4m3x2` for sm_89), and every `_f8w` module on the 4090 (114 PTX: the 7 rows, their arms and table picks, 2 px twins) passes its
differential and probe gates. Outside, competitors only (as every round): torch's TF32 legs at 1e-3 (1.4–2.6·10⁻³),
`triton` on `lsm_1_131072` / `lsm_4_32768` (OutOfResources), `triton-tf32` on the TF32 MLPs. The A100's GEMM cell keeps
`tile_bench`'s bit-exact gate off (`split-tile`: the default face's bits, within D2) — the matrix drops that cell.

### Performance — 5fcb401b → f1d3a04f (every leg, same precision; even = |ratio − 1| ≤ max(null, 5 %))

| part | all | f32 | tf32 | bf16w | bf16kv | bf16 | GEMM | MLP | attention | LLM | conv | fan / loops | W / E / L |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rtx-5080 | 0.979 → **0.965** | 0.869 → **0.859** | 1.186 → 1.183 | 1.662 → **1.615** | 1.043 → **1.014** | 0.982 → **0.967** | 1.045 → **1.033** | 1.054 → 1.048 | 0.858 → **0.840** | 1.036 → **1.015** | 0.529 → 0.534 | 1.032 → 1.023 | 30/39/51 → 32/37/51 |
| rtx-3090 | 1.009 → **0.997** | 0.893 → 0.891 | 1.148 → 1.139 | 1.498 → **1.419** | 1.000 → **0.938** | 1.134 → 1.122 | 1.099 → 1.116 | 1.119 → 1.131 | 0.827 → **0.742** | 1.004 → **0.962** | 0.833 → 0.879 | 0.977 → 1.026 | 43/17/60 → 42/18/60 |
| a100 | 1.242 → **1.185** | 1.104 → **0.953** | 1.646 → 2.087 | 2.156 → **1.682** | 1.124 → **1.035** | 1.228 → 1.423 | 1.288 → 1.389 | 1.537 → **1.263** | 1.029 → **0.853** | 1.232 → **1.121** | 0.978 → 1.242 | 1.494 → **1.019** | 24/9/86 → 41/6/72 |
| rtx-4090 | 0.948 → 0.954 | 0.868 → 0.869 | 1.366 → **1.239** | 1.379 → 1.377 | 0.826 → 0.863 | 0.898 → 0.958 | 1.104 → **1.075** | 1.155 → **1.123** | 0.600 → 0.678 | 0.908 → 0.903 | 0.739 → 0.879 | 1.102 → **1.040** | 36/27/57 → 42/26/52 |
| h100 | 1.318 → **1.204** | 1.006 → **0.974** | 2.625 → **1.688** | 2.194 → **1.749** | 1.103 → **1.022** | 1.646 → 1.620 | 1.623 → **1.511** | 1.617 → **1.381** | 0.935 → 0.933 | 1.215 → **1.031** | 0.942 → **0.908** | 1.201 → 1.183 | 22/3/95 → 25/7/88 |

Geomean Mapal / best over the rows both matrices hold (120; the A100 119); bold = better. The 86 rows of the
competitors step / the 34 bf16-storage rows: 5080 0.945 → **0.934** / 1.068 → **1.048**, 3090 0.952 → **0.943** /
1.171 → **1.147**, A100 1.188 → **1.089** / 1.388 → 1.463, 4090 0.946 → **0.935** / 0.952 → 1.003, H100 1.181 →
**1.075** / 1.740 → **1.605**. The A100's TF32 / bf16 / conv columns and the 4090's bf16 / attention columns moved the
other way on their boxes (run notes 2–3, finding 3).

### The fixes on the hardware (same-box arms; cell: geomean OFF / ON, > 1 = the shipped ON faster, [range], W / null / L)

| rule (arm) | rtx-3090 | a100 | rtx-4090 | h100 | predicted (fix step) | verdict |
|---|---|---|---|---|---|---|
| `split-tile` (f32, tf32x3) | = shipped | 1.39 [0.74–1.70] 26/0/1 | = shipped | = shipped | A100 1.43–1.54 (tensor-cores-2) | **confirmed** (GEMM-heavy rows 1.44–1.70); m128 × 6144 × 4096 0.74 (finding 2) |
| `split-issue` (OFF: 5fcb401b's cut) | = shipped | 1.42 [0.78–1.87] 26/0/1 | = shipped | = shipped | the cut at the split's issue | **confirmed**, the same loser |
| `split-least` (bf16w × f32) | = shipped | 1.11 [1.03–1.23] 4/0/0 | = shipped | 1.09 [1.01–1.17] 3/1/0 | A100 / H100 PCIe prefills 1.69–2.67 over 5fcb401b | **confirmed** (prefills 1.13–1.23 over OFF; 1.20–2.42 over 5fcb401b, below) |
| `wgmma-ws` (TF32, H100) | — | — | — | 1.09 [0.99–1.40] 4/8/0 | 11 / 12 wins | **partly**: no loss left (5fcb401b 0/3/9), 4 wins / 8 null |
| `tma-warm` | 1.00 0/13/0 | 1.00 1/11/1 | 0.99 0/12/1 | 1.01 1/12/0 | the TMA first launch before the clock | null everywhere (the H100's warm-up is in `wgmma-ws`'s ON) |
| `ws-persist` (ON arm; OFF / ON < 1 = ON faster) | — | — | — | 1.01 [0.98–1.04] 1/11/0 | measured mixed, shipped off | **off confirmed** |
| `stream` | 1.38 [1.00–2.05] 10/1/0 | 1.42 [1.05–3.20] 11/0/0 | 1.27 [0.95–2.19] 9/2/0 | 1.29 [0.97–2.38] 9/2/0 | A100 attn 2048 × 128 0.94 → 0.28 ms | **confirmed**: the HBM parts' long-row losses gone |
| `unit-uniform` (prefill T8192) | 1.75 | 1.47 | 1.12 | 1.50 | 5fcb401b: A100 0.75, H100 0.65 | **confirmed** |
| `mma-combine` | 1.17 [1.01–1.45] 3/1/0 | 1.15 [0.98–1.49] 3/1/0 | 1.04 [0.96–1.14] 3/1/2 | = shipped | 18 / 20 by the cut's price | **confirmed** on the 3090 / A100; 4090 bf16 m128 0.96–0.97 (attempted-open in bugs-properties) |
| `streamk-epi` | 1.05 1/1/0 | 1.05 1/1/0 | = shipped | 0.99 [0.82–1.08] 5/1/2 | 3090's losers off by the L2 rule | **3090 confirmed** (loop_4096 0.74 gone); **H100: two new losses** (finding 1) |

Per row and part: `analysis/arms-<part>.md`.

### Predicted vs measured, per fix

| fix | predicted | measured on the rented parts |
|---|---|---|
| fp8 on sm_89 (PTX ISA capability) | the 4090's `_f8w` modules assemble | all 114 f8w PTX pass `ptxas`, the 7 rows and 2 px twins their gates; Mapal ms / its `_bf16w` twin 0.78 (B1 L2048 0.47), / the bf16 reference 0.86 (4090 box clock-locked) |
| tf32x3 split tile, A100 vs the FFMA tile | 1.43–1.54× | 1.44–1.70× on the 21 GEMM-heavy rows, 1.39 over all 27 (26 W / 1 L); A100 f32 class 1.104 → 0.953 |
| bf16 weights × f32 on tensor cores (prefills were 11.8–17.5× torch bf16) | A100 1.74 / 2.67, H100 PCIe 1.69 / 2.16 faster | A100 `llm_pre_t2048_bf16w` 16.27 → 8.32 ms (1.95×), `blkpre` 57.2 → 23.7 (2.42×); H100 6.13 → 4.74 (1.29×), 22.1 → 18.4 (1.20×) on the 500 W box; against torch bf16 now 8.05 / 5.90 (A100), 7.28 / 6.64 (H100) |
| H100 `wgmma-ws` + TMA first launch | 11 / 12 TF32 wins vs the non-ws kernel | 4 W / 8 null / 0 L; a call's time outside its kernel 120–270 µs → 2–15 µs (iteration ≈ kernel); TF32 class 2.625 → 1.688 |
| `ws-persist` vs cuBLAS / CUTLASS Hopper | the walk prices `(⌈tiles/ctas⌉ − 1)·…`, shipped off | ON null on 11 / 12; cuBLAS runs `sm90_xmma_gemm_…_tilesize128x256x32_warpgroupsize2x1x1` on a persistent 66 × 2 grid (`segment_k_on` at m128): the TF32 rows are 1.26–2.37× the leader |
| the stream at long rows on HBM parts | priced at the broadcast rate | A100 `attn_2048_128` 0.732 → 0.292 ms, prefill T8192 208.6 → 93.7 ms (vs Liger 3.15 → 1.59); H100 T8192 86.6 → 50.4 ms (3.35 → 1.28); `stream` OFF / ON ≥ 0.95 on every row of every part |
| A100 MLP chains, fan-in/out + loops (split rate with the ALU issue) | 1.27–1.85× | A100 MLP 1.537 → 1.263, fan / loops 1.494 → 1.019; 20 A100 losses closed (`mlp_2048` 1.34 → 0.90, `loop_4096` 1.32 → 0.90, `fanout_2048` 1.34 → 0.91, `pj_m2048_n4096_k4096` 1.32 → 0.91, …) |
| `streamk-epi` on the 3090 | its losers off | 1 W / 1 null / 0 L (5fcb401b 2/8/2, `loop_4096` 0.74) |
| `mma-combine` | 18 / 20 by the cut's price | 9 W / 3 null / 2 L over the 3090 / A100 / 4090 |

### Properties: measured on the box against the integrated profile (`analysis/replay-flips.txt`)

| property: measured (profile) | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|
| FFMA sustained clock MHz | 1 437 (1 561) | 1 395 (1 410) | **984** (2 520; host lock) | 1 703 (1 807; 500 W) |
| DRAM read GB/s | 892 (891) | 1 488 (1 488) | 824 (981) | 3 193 (3 186) |
| `lds bcast` warp loads a clock × 1000 (`lds_bcast_pm`) | 245 (250 derived) | 241 (241) | 245 (250 derived) | 247 (250 derived) |
| barrier, 256 threads, cycles (`bar_cycles`) | 56.8 (57) | 53.7 (54) | 57.3 (57) | 41.8 (42) |
| `coop_body_cycles` | 1 996 (2 102) | **3 046** (4 048) | 7 021 (7 292) | 307 (348) |
| `mma.sync` TF32 / 16-bit flops a SM a clock | 256 / 512 (=) | 1 012 / 2 024 (=) | 256 / 512 (=) | 1 351 / 2 702 (=) |
| tf32x3 issue (`split3`) | 252 | 824 | 252 | 1 079 |
| `tma_copy_ns` (box ≤ 4 KiB, depth 2) | — | — | — | 158 (159) |
| L2 keep (`l2keep.cu`: the re-read rate's edge) | ~1.025 × L2 (0.5 ×) | no edge: ≤ 1.12 × DRAM from 0.25 × (0.5 ×) | ~1.02 × L2 (nominal) | **0.625–0.75 × L2** (0.5 ×) |
| idle DRAM latency ns (profile `lat_ns`) | 318 (452) | 403 (728) | 166 (380) | 354 (629) |
| `pageable_pm` | 748 | 852 | 881 | 844 |

**Would a measured value flip a rule?** Every one of the 137 programs emitted on the part's profile against the same
with the measured value as `--cost=` (`replay_all.sh`): the H100's L2 keep (0.69 × L2), its 500 W clock and body, the
A100's body (3 046), the 3090's keep (1.025 × L2), body and clock — **no program moves**. The 4090 box's clock (984 MHz)
moves **70 of 137** — its column times emission priced for the nominal clock on a part running at 39 % of it. The
measured `lds_bcast_pm` is within 2–4 % of the bank-derived 250 on every part; `bar_cycles` and the matrix rates match
the profiles (bugs-properties' measurements, now confirmed on a second box of each part).

### New losers and what they point at

1. **`streamk-epi` on the H100** (new): `epi_4096` 0.82, `loop_4096` 0.835 (shipped ON slower). ON = the stream-K walk
   on 264 CTAs (2 an SM, 234 registers) 4.38 ms; OFF = the 1 024-tile grid of the same tile 3.62 ms — 1 024 tiles on
   264 slots are 3.88 waves (a 3 % tail): the walk's fix-up and its single long wave cost 21 % to save 3 %. The L2 rule
   reads the L2 share / keep; the measured keep does not move it (replayed). **Points at** the walk's realized rate
   against the grid's tail fraction `(⌈w⌉ − w) / ⌈w⌉` — the property that separates these rows from the 3090's wins.
2. **`split-tile` on the A100's `pj_m128_n6144_k4096`** (0.74): ON = the split tile on 96 × 2 = 192 CTAs at 70 KB, no
   k cut (192 of 216 slots: 0.89 of a wave), 0.661 ms; OFF = the FFMA tile's stream-K walk on 216 CTAs, 0.534 ms;
   `pj_m128_n4096_k4096` takes S = 8 (64 × 16 CTAs) and wins 1.10. **Points at** the cut's price just under one wave:
   an issue-bound split tile pays its full per-SM issue over the whole k on every CTA (`mma_pieces` counts one wave at
   0.89 of the slots), where a walk evens the FFMA tile.
3. **The A100 box** (not code): the shipped kernels (same hashes) run 1.25–1.45× their 5fcb401b time — and **bimodally
   by process**: the TF32 projection's one binary takes 2.78–2.84 ms in some processes and 3.48–3.62 ms in others
   (`analysis/a100-clock-test.txt`; 5fcb401b's box: 2.86 ± 0.01 in all ten), an offline-`ptxas` cubin is no faster than
   the driver's JIT, and cuBLAS stays at 1.85–1.91 ms (SW power cap, 1 290–1 318 MHz). This is the A100's TF32 (1.646 →
   2.087), bf16 (1.228 → 1.423) and conv (0.978 → 1.242) columns and its six new losses (the m1 / m16 bf16 GEMVs,
   `ew_4096`, `lsm_1_131072`, the m16 f32 projections). **Points at** a per-process placement / clock state no
   calibration measures: a `calib.cu` kind repeating one tile kernel across processes would separate box state from code.
4. **The 4090's bf16 projections** (`pj_m128_n4096_k4096_bf16` 1.05 → 1.33, `…_k14336_bf16` 1.02 → 1.24, `pj_m16_n4096`
   0.82 → 1.19, `pj_m16_n6144` 0.72 → 1.08) on the clock-locked box: emission picked for 2.52 GHz (tiles, `mma-combine`
   cuts — 0.96–0.97 OFF / ON here) against library kernels that pick per launch. **Points at** the sustained clock as a
   run-time property (the profile's is nominal; at the measured clock 70 programs change).
5. At the threshold, same kernels where checked: the 3090's TF32 m2048 projections 1.01–1.05 → 1.07–1.10 (vs
   `cublas-tf32-fresh`), its `mm_4096` cell 1.09 → 1.10; the 5080's two `fa2-bf16` decode rows 1.04 → 1.05; the H100's
   `pj_m1_n4096_k4096_bf16` 1.01 → 1.12 (500 W).
6. Unchanged / structural: the bf16w prefills still 5.9–8.1× torch bf16 on the A100 / H100 (two or three bf16 products a
   split against torch's one); 16-bit tiles on `mma.sync` on the H100 (no 16-bit `wgmma`: `pj_m2048_n6144_k4096_bf16`
   2.21×); the H100's TF32 tile against cuBLAS's 128 × 256 × 32 two-warpgroup persistent kernel (1.26–2.37×);
   `convmc_8_64_56_3_tf32` 3.4–5.2× cuDNN's tensor-op implicit GEMM.

Key rows (Mapal / best, leg; every row: matrix.md):

| row | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|---|
| lsm_1_131072 | 0.74 compile-cg | 1.24 compile-cg | 1.16 compile-ro | 0.93 compile-ro | 1.44 sdpa-compile-cg |
| decode_1024_256 | 0.53 chain-cache | 0.53 chain-cache | 0.86 chain-nocache | 0.41 chain-cache | 1.45 chain-cache |
| llm_dec_b1_l2048 | 0.91 compile-ro | 0.75 compile | 0.81 compile-ro | 0.72 compile | 0.52 compile-cg |
| llm_dec_b32_l8192 (f32) | 0.27 compile-cg | 0.29 compile | 0.30 compile-cg | 0.29 compile | 0.25 compile |
| llm_dec_b32_l8192_bf16kv | 1.09 flashinfer-bf16 | 1.10 bf16-cudnn | 1.25 bf16-cudnn | 1.02 bf16-compile | 1.56 bf16-compile-ro |
| pj_m2048_n6144_k4096_bf16 | 1.06 bf16-compile-ro | 1.02 bf16-compile-cg | 1.94 bf16 | 1.05 bf16-compile-cg | 2.21 bf16 |
| pj_m2048_n4096_k4096_tf32 | 1.15 lt-tf32-fresh | 1.00 triton-tf32 | 1.91 lt-tf32 | 1.12 lt-tf32 | 1.33 lt-tf32-fresh |
| llm_pre_t8192 | 0.84 liger | 1.14 sdpa-efficient | 1.59 liger | 1.04 liger | 1.28 liger |
| llm_pre_t2048_bf16 | 1.37 bf16-compile-cg | 1.34 bf16-compile-cg | 2.62 bf16-compile-cg | 1.27 bf16-compile-cg | 2.71 bf16-compile-ro |

### Peak device memory, the automatic KV cache, the fp8 rows

Peak (`mapal_gpu_peak` against torch one process per (program, leg), the fastest f32 leg): geomean Mapal / torch 3090
0.76, A100 0.75, 4090 0.73, H100 0.73 — **lower on all 22 layer rows of every rented part** (prefill T8192 0.41).

Naive twins (Mapal naive ms (/ torch naive, / torch HF-cache f32); explicit / HF-cache; peak MB naive / torch naive):

| shape | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---|---|---|---|
| B1 L2048 | 8.9 (0.11, 1.13); 0.78; 286 / 313 | 8.9 (0.09, 1.14); 0.73; 286 / 329 | 8.2 (0.13, 1.00); 0.74; 285 / 354 | 5.2 (0.10, 1.14); 0.61; 291 / 356 |
| B1 L8192 | 26.7 (0.08, 1.74); 0.87; 651 / 748 | 26.1 (0.07, 1.47); 0.73; 651 / 748 | 21.3 (0.09, 1.40); 0.80; 651 / 748 | 15.8 (0.07, 1.57); 0.70; 655 / 772 |
| B8 L2048 | 56.2 (0.09, 1.81); 0.93; 1128 / 1321 | 38.0 (0.05, 1.38); 0.78; 1136 / 1321 | 37.5 (0.08, 1.28); 0.64; 1136 / 1321 | 33.4 (0.07, 1.85); 0.71; 1150 / 1345 |
| B8 L8192 | 215.7 (0.07, 1.96); 0.82; 4023 / 4780 | 142.8 (0.05, 1.49); 0.60; 4031 / 4780 | 136.2 (0.07, 1.36); 0.61; 4031 / 4780 | 111.3 (0.06, 1.63); 0.70; 4041 / 4804 |
| B32 L2048 | 206.5 (0.08, 1.88); 0.80; 4026 / 4778 | 142.7 (0.05, 1.48); 0.73; 4026 / 4778 | 136.5 (0.07, 1.36); 0.61; 4030 / 4778 | 111.9 (0.06, 1.68); 0.72; 4040 / 4802 |
| block B1 L4096 | 28.6 (0.16, 1.22); 0.95; 1080 / 1130 | 26.8 (0.14, 1.41); 1.08; 1080 / 1122 | 27.0 (0.20, 1.08); 0.89; 1080 / 1154 | 16.5 (0.15, 1.43); 1.01; 1085 / 1154 |

The A100's naive rows are back below 5fcb401b's (B8 L8192 285 → 143 ms: its projections on the issue-priced split).
fp8 rows (`analysis/f8tab.md`; Mapal ms a step / its `_bf16w` twin; / the bf16 reference, geomean): 5080 0.76 / 1.01,
3090 1.03 / 1.18, A100 0.99 / 1.11, 4090 0.78 / 0.86 (first timing on sm_89), H100 0.84 / 0.99.

### Evidence (intent check, `evidence-<part>.tsv`)

3090 115 566 checks / 4 391 mismatches, A100 121 915 / 8 380, 4090 130 184 / 5 375, H100 128 584 / 8 352. This code's
intents: `split-tile` 1 416 / 1 417 on the A100 (5fcb401b: 113 misses — the hot loop the checker reads, corrected in
bugs-properties) and 287 / 287 on the H100, `wgmma-ws` 201 / 201, `ws-persist` 100 / 100, `stream` 333–418 a part and
`mma-combine` 25–127 a part with no miss; `streamk-epi` 302 / 342 (A100), 101 / 114 (H100); the standing groups
(`min-blocks`, `face-registers`) as in 5fcb401b.

### Gap captures (every losing row's leader next to Mapal, before each box was destroyed)

Per part, the rows with ratio > 1.05 beyond null (from the part's own matrix: 3090 59, A100 72, 4090 51, H100 86),
up to 40, round-robin over (family, precision) by gap (`gaprows.py`): nsys (`--cuda-graph-trace=node`) of Mapal's
program and of the leader's leg — the kernel trace, kernel and API summaries and the timed iteration
(`gapsum.py`: the launches between two L2 flushes, its kernels, gaps and time outside kernels); Mapal's module SASS
(`ptxas` + `cuobjdump`); `ncu` refused (`ERR_NVGPUCTRPERM`) on all four, so the leader's SASS comes from the row's
Triton cache (`cuobjdump` of the cubin) or from the cubin the leader's process loaded (a CUPTI `MODULE_LOADED`
injection: cuBLAS's `nvjet` kernels are not in libcublasLt's fatbins). 40 rows a part (3090: 33 module + 34 Triton
leader kernels; A100: 42 + 27; 4090: 36 + 31; H100: 53 + 15, 1 not loaded). Raw:
`.claude/raw-evidence/cross-arch/f1d3a04f/<part>/gaps/` (`index.tsv`: workload, precision, ratio, null, leader leg, ms,
both sides' iteration summary and top kernels, the SASS source; one directory a row). The A100's archive is the slim one
(no nsys reports, caches or cubins: the full archive fetched at ~45 KB/s from that host).

### Fix follow-up: final HEAD 2972c3d1 (phase D)

`2972c3d1` (review: a narrow infinity takes the split guard, the 3090's L2 keep read off its sweep, claims corrected)
changes the shipped emission of 7 of the 137 programs on the 3090 (`decode_1024_256`, `causal_1024_256`,
`prefill_1024_512_256`, `gen_1024_512_256`, `mlp_512`, `indep_512`, `smlp_4096_256_1024`), 6 on the A100 and the H100 (the
B32 `_bf16w` / `_bf16kv` decodes, the two bf16w prefills), none on the 4090 (`emit_cmp.sh`). Re-timed on the same boxes,
final shipped against the integrated emission as `.on-base`, ABBA × 5 (`dbuild.sh`, gates pass): **H100 6 / 6 null**
(0.998–1.017); **A100 2 wins / 4 null** (`llm_dec_b32_l2048_bf16w` 1.031, `llm_dec_b32_l8192_bf16w` 1.045); **3090 6 null,
`smlp_4096_256_1024` 0.970** (the final build 3 % slower, at the threshold). The matrix stays on `f1d3a04f`.

### Open items (missing facts)

1. `streamk-epi`'s tail fraction against the walk's realized rate (H100 `epi_4096` / `loop_4096` 0.82 / 0.835).
2. The split tile's k cut just under one wave (A100 `pj_m128_n6144_k4096` 0.74).
3. A per-process tile-rate spread (the A100 box: one binary 2.8 or 3.5 ms by process) — a calibration kind repeating one
   kernel across processes; and the sustained clock at run time (the 4090 box's lock moves 70 programs' emission).
4. The L2 keep below / above the profile (H100 edge 0.625–0.75 × L2 against 0.5; the A100's L2 re-read never exceeds
   1.12 × DRAM) — no program moves on the suite.
5. Hopper: 16-bit `wgmma`; the TF32 tile against cuBLAS's 128 × 256 two-warpgroup persistent kernel; `ws-persist`'s
   separating fact. The bf16w prefills against torch bf16 (5.9–8.1× on the HBM parts).

Raw (local, not in git): `.claude/raw-evidence/cross-arch/f1d3a04f/<part>/` (`results-<part>-comp.txt` the competitor legs
+ calibration of phase A, `results-<part>.txt` the Mapal legs, `results-<part>-d.txt` phase D, `calib-<part>.txt`,
`evidence-<part>.tsv`, `prof-<part>.tgz`, `prof-<part>-comp.tgz`, wisdom, tune, `gaps/`), `analysis/` (arms, gates via
`gates.py`, cm / cmtab, peaks, f8tab, xcmp, replay-flips, a100-clock-test, the `emit_cmp` tables), `scripts/` (comp_box,
xbox, launch, watchdog, phase_bc, build, stems, ops_c, gaps / gap_box / gapsum / gaprows / regap, emit_cmp, dbuild,
replay_all, mkmatrix, the analysis scripts).

## b8429388 / 9be9886d (five-GPU round: bf16 integration, every competitor leg at the program boundary, the full model)

Branch `local-rtx-5080`: the bf16 integration `b8429388` (merges of bf16/tensor-tile `mma-walk`, bf16/class
`finite-check` / `narrow-window` / `round-post` / `device-trap`, prefill/residency's last three commits, `cand-made`)
and its reviewed final `9be9886d` (review fixes: a copy-out is a `device-trap` checkpoint; checker and test changes).
Four vast.ai boxes, the f1d3a04f round's tooling (CUDA 13.0.2 devel image, torch 2.11.0+cu130 / Triton 3.6.0, Liger
0.8.5, FlashInfer 0.7.5.post1, flash-attn v2.8.3.post2 built on every box with `NVCC_THREADS=2`, FA3 on the H100,
`TORCH_SKIP` for the bf16-storage twins) with two changes. **(1) Every competitor leg is timed at the program
boundary** (`fb909121`, gaps-disasm H): device idle, then the host clock around the call and a synchronize, the leg's
host dispatch inside its clock as a Mapal program's dispatch and launches are inside its own `time` reads; the former
event bracket is still printed as `dev_med_ms`. **(2) The full model** (Llama-3.2-1B-Instruct, `unsloth/`, no token)
against seven inference stacks on every box, type to type, as `local-rtx-5080/full-model.md` does on the RTX 5080.
Matrix over all five parts on `b8429388` (the RTX 5080 column: the sources of `matrix.md`'s command at that commit,
`gapfix-integrate` @ d945c641 / d450014d / 9abeb446 + `gapfix-boundary` chain legs): [matrix.md](../matrix.md). Raw
output (never in git): `.claude/raw-evidence/cross-arch/five-gpu/<part>/`, scripts in `five-gpu/scripts/`.

**Phases, one rental a part.** A (before the integrated commit existed): box check (device facts + a 30 s FFMA burn:
a box whose busy SM clock is below 0.6 × its max, or that reports a hardware / thermal slowdown, is replaced), the
installs with the GPU idle (the kernel-suite stack, FA2 / FA3, and `~/.cache/mapal/model-compare/setup_box.sh`: HF
transformers 5.22.2, gpt-fast 6ecad9b, vLLM 0.33.0, SGLang 0.5.23, TensorRT-LLM 1.3.0, llama.cpp 0ce7f21 with bf16 /
f16 / f32 GGUFs), `calib.cu` every kind (incl. `math`, `lds bcast`) + `gsync` / `copy` / `body` twice more,
`exp_calib`, `tma_calib` (H100), `mma_calib` (sm_89+), `l2keep.cu`, then `torch_bench.py` / `comp_bench.py` over the
137 timed programs + the two GEMM cells (30 reps, the f1d3a04f round's reference values as their gate), the torch
peaks, and the full-model engines in both classes. B (`b8429388`): `box_run.sh` (`TORCH=0 COMP=0 CALIB=0`, the
bundle's 137 programs + cells + 18 px differentials, shipped emission, no arms: differential, probe value gates,
ABBA × 5, the tune pass, `gemm_lib` / `tile_bench` / `chain_bench` / `mlp_bench` legs), the full model (the six
builds linked on the box, weights exported on the box, the same-class HF references `hf_reference.py` first), then
competitor legs re-timed back to back (drift). C (`9be9886d`): the emission compared, the changed builds re-timed.

**Run notes.** (1) **H100 market**: one verified H100 SXM offer all night ($4.44 + disk = $4.625/h); the box started
at 23:40 so that phase A ended as its bundle was ready (02:10). (2) **A100 box 1 was destroyed at 130 min by the
watchdog** (the f1d3a04f watchdog's per-part default caps, 130 min A100 / 85 min H100, were not overridden at launch —
an orchestration error; the H100's cap was raised before it fired): $1.61 lost, phase A redone on a second box.
(3) **Harness fixes on the boxes** (runner-level, measurement unchanged): the f32 runner copy of `run_sglang.py` hangs
in `engine.get_server_info()` → the bf16 class runs the original runner, its shutdown bounded (60 s) after the result
is written; TensorRT-LLM needs `libnuma1` and, in a root container, Open MPI 5's `*_ALLOW_RUN_AS_ROOT` for its executor
spawn; `setup_box.sh`'s driver parse failed on driver 610 (`CUDA UMD Version`). On the RTX 4090, SGLang 128 / 512 and
TensorRT-LLM were re-run after phase A on the same box. (4) **Peak memory**: the 3090 container's NVML hides process
memory — competitor peaks there (†) are the device's growth while the runner ran, Mapal's `n/a`.

### Boxes and cost

| part | GPU (driver, power limit) | host CPU | offer | $/h | hours | cost | status |
|---|---|---|---|---:|---:|---:|---|
| (3090 tries) | — | EPYC 7742 / 7663, Xeon 6330N | 61732988, 55384239, 52870151 | 0.30–0.41 | 0.31 | $0.13 | never running (status error) |
| rtx-3090 | RTX 3090 24 GB (595.99.02, 390 W) | Ryzen 9 3900X | 52497112 | 0.389 | 5.66 | $2.20 | A, B, C |
| (a100 box 1) | A100-SXM4-40GB (610.57.04, 370 W) | EPYC 7532 | 62381177 | 0.736 | 2.21 | $1.62 | destroyed by the watchdog's 130 min cap mid phase A |
| a100 | A100-SXM4-40GB (590.48.01, 400 W) | EPYC 7713 | 36878334 | 0.770 | 4.38 | $3.38 | A, B, C |
| rtx-4090 | RTX 4090 24 GB (595.99.02, 600 W; default 450 W, max 3120 MHz) | Ryzen 9 9950X | 61175417 | 0.511 | 5.95 | $3.04 | A (+ SGLang / TRT-LLM re-run), B, C |
| h100 | H100 80GB HBM3 SXM (615.71.09, 700 W) | Xeon Platinum 8568Y+ | 41555537 | 4.625 | 4.26 | $19.72 | A, B, C |
| **total** | | | | | | **$30.09** | every `mapal-five-*` destroyed (checked by label) |

Budget $40 (watchdog at $39.5, 600 min a box). $/h is the instance's own (disk included).

### Box conditions (no box throttled; none replaced for it)

| part | FFMA burn: busy / max SM clock | power, temp under the burn | through the runs | calibration lines |
|---|---|---|---|---|
| rtx-3090 | 1 950 / 2 130 MHz (0.92) | 350 W of 390, 45 °C | 1 860–1 972 MHz, ≤ 45 °C (engines); 336 W, 42 °C after B | 510 (8 `math`, 8 `lds bcast`, 19 `l2keep`; no `mma_calib`: fp8 `mma` needs sm_89) |
| a100 | 1 410 / 1 410 MHz (1.00) | 166 W of 400, 51 °C | 1 365–1 410 MHz, ≤ 50 °C | 522 (same kinds; no `mma_calib`) |
| rtx-4090 | 2 445 / 3 120 MHz (0.78: the board's boost table; 2 445 sustained) | 255 W of 600, 59 °C | 2 445 MHz, ≤ 47 °C | 653 (+ 131 `mma`) |
| h100 | 1 980 / 1 980 MHz (1.00) | 384 W of 700, 45 °C | 1 980 MHz, ≤ 42 °C | 680 (+ 131 `mma`, 19 `tma`) |

Unlike f1d3a04f (4090 clock-locked at 1.2 GHz, H100 at 500 W), every part ran at its full clock and power. **Host
noise** (drift, below): the A100's and H100's host-bound legs moved up to ±30–70 % between phase A and B on the same
box; the RTX parts' within ±5 %.

### Value gates

| part | differential, first pass | differential, held | probe gates (programs) | arms · candidates within D2/D3 | device-exact oracles | split programs | gemm_lib legs within | torch legs within | competitor legs within | full model f32 tokens vs HF f32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| rtx-3090 | 86 / 88 | 86 / 86 | 137 / 137 | 1130 / 1130 | 136 / 136 | 8 / 8 | 112 / 112 | 622 / 650 | 132 / 136 | 128 / 128 / 128 |
| a100 | 112 / 116 | 112 / 112 | 137 / 137 | 897 / 897 | 136 / 136 | 8 / 8 | 112 / 112 | 622 / 650 | 133 / 136 | 128 / 128 / 128 |
| rtx-4090 | 124 / 128 | 124 / 124 | 137 / 137 | 1175 / 1175 | 136 / 136 | 8 / 8 | 112 / 112 | 624 / 650 | 132 / 136 | 128 / 128 / 128 |
| h100 | 112 / 116 | 112 / 112 | 137 / 137 | 1133 / 1133 | 136 / 136 | 8 / 8 | 112 / 112 | 626 / 650 | 149 / 153 | 128 / 128 / 128 |

**Finding 1 — a Mapal value failure on every part: `px_llm_blkdec_b2_l16_bf16` (the bf16-storage block-decode
differential) is outside ADR-0040 D2's bound.** Element 2 of 5: 0.13053171 against the exact face's 0.12759829, rel
2.30·10⁻² > 1.62·10⁻² (K = 2168), the same value on the 3090, A100, 4090 and H100, in the shipped build and its
`.off-place` arm, every site candidate. At gapfix-integrate (`d945c641`) the RTX 5080 printed 0.1293837 for it
(rel 1.40·10⁻², within). Between them bf16/class put the `_bf16` class's KV cache in bf16 (`narrow-window`): a cached
key / value is now rounded once more than the bound counts, or the bound's `--bf16` term misses the cache's rounding.
**Points at**: the bf16 class's a-priori bound (`precision_plan.bound` with the cache's `NarrowForm` in the chain) —
open for the compiler; the timed `pt_llm_blkdec_b1_l4096_bf16` passed its probe gate on every part (its probe gate is
the program's own oracle within its bound). `box_run.sh` stops timing on any differential failure, so on the first
three boxes B ran a second pass with that program's builds held out of the differential (kept in `mb/<part>/held/`;
the A100 ran the differential over everything first, then the held pass). Outside, competitors only (as every round):
torch's TF32 legs at 1e-3, `triton` on the two long `lsm` rows, `triton-tf32` on the TF32 MLPs. The A100's GEMM cell
keeps `tile_bench`'s bit-exact gate off (split tile: the default face's bits, within D2) — the matrix drops it.

**Finding 2 — on the A100 the bf16 class's greedy tokens vary run to run**: 3–5 distinct sequences in 5 repetitions at
512 and 2048 (1 at 128; the f32 class 1 everywhere; the 3090, 4090 and H100 1 everywhere) — as on the RTX 5080 at
512 (`full-model.md`: the 16-bit tensor tile's stream-K walk, `mma-walk`, adds a tile's partials in arrival order;
`--gpu-off=mma-walk` is deterministic), meeting near ties; the tables give the first repetition's tokens.

### Kernel suite — Mapal / best competitor at the same precision (program boundary; even = |ratio − 1| ≤ max(null, 5 %))

| | rtx-5080 | rtx-3090 | a100 | rtx-4090 | h100 |
|---|---:|---:|---:|---:|---:|
| **all rows** (119 on every part) | 0.876 | **0.920** | 1.006 | **0.911** | 1.101 |
| all rows, event bracket (`dev_med_ms`, the f1d3a04f definition) | — | 1.011 | 1.134 | 0.964 | 1.233 |
| f1d3a04f, event bracket | 0.965 | 0.997 | 1.185 | 0.954 (1.2 GHz box) | 1.204 (500 W box) |
| W / E / L (120 rows) | 46/44/30 | 39/30/51 | 57/11/51 | 49/25/46 | 30/8/82 |
| GEMM cells & projections | 0.926 13/19/15 | 1.048 7/17/23 | 1.161 14/8/24 | 0.935 13/15/19 | 1.373 2/5/40 |
| MLP chains | 0.974 1/5/0 | 1.094 0/1/5 | 1.075 3/0/3 | 0.945 4/0/2 | 1.285 0/0/6 |
| attention / softmax / long rows | 0.639 9/5/0 | 0.517 9/0/5 | 0.628 12/0/2 | 0.647 12/2/0 | 0.790 5/1/8 |
| LLM decode / prefill / block | 0.987 16/7/14 | 0.973 18/5/14 | 1.013 18/2/17 | 1.053 10/6/21 | 0.975 19/1/17 |
| conv / FIR / elementwise | 0.493 6/0/1 | 0.695 5/0/2 | 0.944 4/1/2 | 0.646 5/0/2 | 0.805 4/0/3 |
| fan-in/out, loops, CPU+GPU | 0.979 1/8/0 | 1.030 0/7/2 | 0.977 6/0/3 | 0.957 5/2/2 | 1.108 0/1/8 |
| f32 rows (66) | 0.772 | 0.801 | 0.819 | 0.807 | 0.902 |
| tf32 rows (13) | 1.154 | 1.149 | 1.716 | 1.208 | 1.646 |
| bf16w rows (9) | 1.559 | 1.463 | 1.575 | 1.634 | 1.581 |
| bf16kv rows (7) | 0.971 | 0.935 | 0.919 | 1.097 | 0.896 |
| bf16 rows (25) | 0.846 | 1.005 | 1.135 | 0.835 | 1.409 |

Geomean Mapal / best leg (family cells: geomean, W/E/L). The program boundary is the fair clock (both sides' host
dispatch inside it); it moves every part 5–11 % in Mapal's favour against the event bracket (launch-bound torch
legs). **Like for like with f1d3a04f** (event bracket, the same harness otherwise): the A100 0.954× of its ratio
(1.185 → 1.134), the 3090 1.01× and the H100 1.02× (even), the 4090 1.01× (f1d3a04f's box was clock-locked). The
`_f8w` rows stay `(no leg)`. The full table, every leg and the verdict flips: [matrix.md](../matrix.md); the event
variant: `five-gpu/matrix-event-bracket/`.

**Drift** (competitor legs re-timed on the same box right after Mapal's suite, against phase A): rtx-3090 70 legs,
geomean 0.997 [0.88–1.04], gpt-fast / llama.cpp / HF at 512 within 4 %; rtx-4090 72 legs, 0.997 [0.95–1.02], model
legs within 1 %; a100 70 legs, 1.033 [0.73–1.42], 26 beyond 5 % (host-bound: m1 projections, B1 decode, attention),
llama.cpp bf16 TTFT at 512 43.4 → 18.3 ms; h100 72 legs, 1.003 [0.66–1.68], 23 beyond 5 % (the B1 decode's FA2 / FA3
/ FlashInfer / Liger legs 0.66–0.70, i.e. phase A slower), model decode within 7 %. The matrix uses phase A's legs;
on the A100 and H100 a host-bound row's ratio carries that box's ±30 % host noise.

**Phase C (`9be9886d`)**: the shipped emission of all 137 programs on all four parts, and of the six full-model
builds, is identical to `b8429388`'s (host and device text less the `; mapal.` comments; null by construction). The
runtime changed (`mapal_gpu_d2h` polls an armed trap), so the full model was re-timed on every box with the final
runtime: final / integrated TTFT 0.998–1.013 and decode 0.998–1.001 on the 3090, 0.998–1.002 / 0.998–0.999 on the
4090, 0.957–1.004 / 0.983–1.014 on the H100, 0.886–1.132 / 0.767–0.959 on the A100 (its integrated pass's decode
3.85–4.77 ms against 3.56–3.89: host noise), tokens equal except the A100's bf16 class (finding 2). The kernel suite's
runtime safety net (the full suite on the final runtime) did not run on the rented parts (budget). The tables below
are the final pass.

### Full model — Llama-3.2-1B-Instruct, batch 1, 128 generated, prompts 128 / 512 / 2048, medians of 5 (warmup 2)

Definitions (`model-compare/common.py`): TTFT = submit → first token on the host; decode = (t₁₂₈ − t₂) / 126 ms per
token; e2e = t₁₂₈; peak = the process's GPU memory. **Type to type**, no gain claimed across classes: the **f32 class**
is the program as written (Mapal's default face: f32 activations on the bf16 weights, within ADR-0040 D2) against the
f32-compute engines (`--dtype f32`, TF32 off: gpt-fast, vLLM, llama.cpp f32 GGUF, HF eager, HF static + compile;
SGLang has no f32 path and TensorRT-LLM's f32 output is wrong — not timed); the **bf16 class** is
`benches/model/variant.py --compute bf16` (`fold: bf16 | f32` on the three sums of products, the bf16 KV cache)
against the bf16 engines (gpt-fast, vLLM, SGLang, TensorRT-LLM, llama.cpp bf16, HF eager, HF static + compile;
llama.cpp f16 is shown, not compared). Mapal: `mapal build --target=<part> --host=x86-64-v3` per prompt length, the
host object linked on the box; `benches/model/run_mapal.py`. Tokens: equal before the first difference against the
same-class HF reference run on the box (`hf_reference.py`, eager attention; f32: TF32 off). Bold = the class's best.

#### rtx-3090 — f32 class

| tool | TTFT 128 ms | TTFT 512 | TTFT 2048 | decode 128 ms/token | decode 512 | decode 2048 | e2e 512 ms | peak MB (128 / 512 / 2048) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Mapal** (as written: f32 activations on the bf16 weights) | 19.37 | **55.98** | **225.7** | **3.974** | **4.038** | **4.222** | 565.3 | n/a / n/a / n/a |
| gpt-fast f32 | 19.17 | 73.42 | 327.0 | 6.072 | 6.152 | 6.295 | 855.5 | 6188† / 6484† / 7546† |
| vLLM f32 | 20.00 | 59.50 | 280.4 | 6.142 | 6.269 | 6.615 | 855.7 | 12498† / 12542† / 12738† |
| llama.cpp f32 GGUF | **18.67** | 58.77 | 274.7 | 6.372 | 6.611 | 7.248 | 900.5 | 5068† / 5120† / 5316† |
| HF eager f32 | 19.62 | 64.58 | 340.4 | 10.948 | 11.124 | 10.765 | 1477.2 | 5084† / 5184† / 6474† |
| HF static + compile f32 | 23.17 | 67.57 | 342.9 | 7.428 | 8.693 | 13.734 | 1172.5 | 5134† / 5264† / 6732† |

Mapal / best tool: TTFT 1.04 (llama.cpp-f32) / 0.95 (llama.cpp-f32) / 0.82 (llama.cpp-f32); decode 0.65 (gpt-fast-f32) / 0.66 (gpt-fast-f32) / 0.67 (gpt-fast-f32).
Tokens equal before the first difference, Mapal f32 vs hf-f32 (`hf_reference.py`, eager, on the box): 128 / 128 / 128 of 128.

#### rtx-3090 — bf16 class

| tool | TTFT 128 ms | TTFT 512 | TTFT 2048 | decode 128 ms/token | decode 512 | decode 2048 | e2e 512 ms | peak MB (128 / 512 / 2048) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Mapal** (`fold: bf16 \| f32`) | 10.62 | 23.84 | 91.65 | 3.960 | 4.015 | 4.104 | 529.5 | n/a / n/a / n/a |
| gpt-fast (max-autotune, CUDA graphs) | **6.86** | 23.02 | 85.05 | **3.228** | **3.274** | **3.295** | 439.3 | 3306† / 3428† / 3948† |
| vLLM 0.33.0 | 14.02 | 21.48 | 72.31 | 3.655 | 3.668 | 3.717 | 487.1 | 13048† / 13060† / 13180† |
| SGLang 0.5.23 | 8.48 | **20.38** | 71.53 | 3.516 | 3.546 | 3.612 | 470.6 | 14711† / 180† / 216† |
| TensorRT-LLM 1.3.0 | 10.86 | 20.50 | **70.25** | 3.537 | 3.594 | 3.681 | 477.0 | 13694† / 13285† / 13748† |
| llama.cpp bf16 | 8.57 | 24.29 | 91.72 | 3.510 | 3.539 | 3.613 | 474.0 | 2724† / 2766† / 2816† |
| HF eager | 12.22 | 22.93 | 81.44 | 10.611 | 10.120 | 9.655 | 1308.0 | 2712† / 2766† / 2948† |
| HF static cache + compile | 12.45 | 24.52 | 83.18 | 4.215 | 4.641 | 6.420 | 613.9 | 2762† / 2810† / 3020† |
| *(llama.cpp f16 — another type, not compared)* | *7.90* | *18.31* | *69.26* | *3.511* | *3.520* | *3.604* | *465.9* | *2724† / 2768† / 2818†* |

Mapal / best tool: TTFT 1.55 (gpt-fast) / 1.17 (sglang) / 1.30 (trtllm); decode 1.23 (gpt-fast) / 1.23 (gpt-fast) / 1.25 (gpt-fast).
Tokens equal before the first difference, Mapal bf16 vs hf-bf16-eager (`hf_reference.py`, eager, on the box): 49 / 128 / 66 of 128.

#### a100 — f32 class

| tool | TTFT 128 ms | TTFT 512 | TTFT 2048 | decode 128 ms/token | decode 512 | decode 2048 | e2e 512 ms | peak MB (128 / 512 / 2048) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Mapal** (as written: f32 activations on the bf16 weights) | **16.19** | **40.00** | **137.0** | **3.559** | **3.819** | **3.889** | 513.5 | 2814 / 2892 / 3174 |
| gpt-fast f32 | 27.13 | 88.73 | 349.1 | 4.137 | 4.212 | 4.374 | 625.4 | 6392 / 6692 / 7754 |
| vLLM f32 | 29.24 | 76.77 | 315.6 | 4.549 | 4.439 | 5.049 | 643.0 | 21656 / 21700 / 21896 |
| llama.cpp f32 GGUF | 22.14 | 87.91 | 322.1 | 4.652 | 5.099 | 5.669 | 738.3 | 5220 / 5272 / 5468 |
| HF eager f32 | 26.24 | 84.94 | 352.1 | 12.658 | 14.907 | 13.354 | 1982.0 | 5268 / 5368 / 6658 |
| HF static + compile f32 | 28.53 | 79.83 | 352.1 | 5.748 | 7.012 | 10.741 | 988.4 | 5362 / 5468 / 6942 |

Mapal / best tool: TTFT 0.73 (llama.cpp-f32) / 0.52 (vllm-f32) / 0.43 (vllm-f32); decode 0.86 (gpt-fast-f32) / 0.91 (gpt-fast-f32) / 0.89 (gpt-fast-f32).
Tokens equal before the first difference, Mapal f32 vs hf-f32 (`hf_reference.py`, eager, on the box): 128 / 128 / 128 of 128.

#### a100 — bf16 class

| tool | TTFT 128 ms | TTFT 512 | TTFT 2048 | decode 128 ms/token | decode 512 | decode 2048 | e2e 512 ms | peak MB (128 / 512 / 2048) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Mapal** (`fold: bf16 \| f32`) | 8.57 | 16.10 | 59.84 | 3.663 | 3.689 | 3.768 | 480.9 | 2806 / 3070 / 3264 |
| gpt-fast (max-autotune, CUDA graphs) | **4.58** | 11.55 | 31.02 | **2.352** | **2.384** | **2.394** | 314.8 | 3512 / 3634 / 4174 |
| vLLM 0.33.0 | 19.64 | 19.70 | 29.76 | 3.081 | 2.888 | 2.966 | 386.2 | 22208 / 22220 / 22340 |
| SGLang 0.5.23 | 9.20 | **11.14** | 29.98 | 3.535 | 3.182 | 3.224 | 414.5 | 23774 / 23790 / 23806 |
| TensorRT-LLM 1.3.0 | 15.62 | 16.54 | **28.17** | 2.750 | 2.867 | 2.938 | 380.3 | 22034 / 22014 / 22088 |
| llama.cpp bf16 | 51.75 | 43.42 | 130.6 | 2.724 | 2.794 | 2.856 | 398.9 | 2898 / 2940 / 2990 |
| HF eager | 12.17 | 14.31 | 43.89 | 10.477 | 11.772 | 11.382 | 1510.4 | 2898 / 2952 / 3134 |
| HF static cache + compile | 16.57 | 17.72 | 42.55 | 3.762 | 4.128 | 6.066 | 541.1 | 2996 / 3040 / 3282 |
| *(llama.cpp f16 — another type, not compared)* | *49.14* | *14.76* | *56.38* | *2.746* | *2.746* | *2.847* | *364.2* | *2896 / 2940 / 2990* |

Mapal / best tool: TTFT 1.87 (gpt-fast) / 1.44 (sglang) / 2.12 (trtllm); decode 1.56 (gpt-fast) / 1.55 (gpt-fast) / 1.57 (gpt-fast).
Tokens equal before the first difference, Mapal bf16 vs hf-bf16-eager (`hf_reference.py`, eager, on the box): 49 / 32 / 57 of 128.

#### rtx-4090 — f32 class

| tool | TTFT 128 ms | TTFT 512 | TTFT 2048 | decode 128 ms/token | decode 512 | decode 2048 | e2e 512 ms | peak MB (128 / 512 / 2048) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Mapal** (as written: f32 activations on the bf16 weights) | **7.42** | **20.10** | **75.65** | **3.416** | **3.470** | **3.611** | 457.3 | 2798 / 2854 / 3136 |
| gpt-fast f32 | 10.31 | 37.34 | 159.7 | 5.533 | 5.543 | 5.713 | 742.0 | 6416 / 6722 / 7784 |
| vLLM f32 | 10.57 | 31.57 | 142.5 | 5.632 | 5.697 | 5.886 | 755.1 | 12618 / 12662 / 12858 |
| llama.cpp f32 GGUF | 11.96 | 33.74 | 144.8 | 5.801 | 5.987 | 6.497 | 794.6 | 5190 / 5242 / 5438 |
| HF eager f32 | 11.99 | 35.30 | 201.4 | 7.288 | 7.693 | 8.555 | 1012.2 | 5226 / 5326 / 6616 |
| HF static + compile f32 | 13.66 | 36.49 | 203.1 | 6.650 | 7.690 | 11.823 | 1013.6 | 5348 / 5454 / 6968 |

Mapal / best tool: TTFT 0.72 (gpt-fast-f32) / 0.64 (vllm-f32) / 0.53 (vllm-f32); decode 0.62 (gpt-fast-f32) / 0.63 (gpt-fast-f32) / 0.63 (gpt-fast-f32).
Tokens equal before the first difference, Mapal f32 vs hf-f32 (`hf_reference.py`, eager, on the box): 128 / 128 / 128 of 128.

#### rtx-4090 — bf16 class

| tool | TTFT 128 ms | TTFT 512 | TTFT 2048 | decode 128 ms/token | decode 512 | decode 2048 | e2e 512 ms | peak MB (128 / 512 / 2048) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Mapal** (`fold: bf16 \| f32`) | 6.55 | 11.94 | 41.65 | 3.431 | 3.468 | 3.557 | 448.9 | 2778 / 2828 / 3018 |
| gpt-fast (max-autotune, CUDA graphs) | **4.54** | **10.27** | 39.44 | **2.919** | **2.926** | **2.990** | 382.3 | 3500 / 3634 / 4162 |
| vLLM 0.33.0 | 6.08 | 10.74 | 34.70 | 3.112 | 3.118 | 3.173 | 406.8 | 13168 / 13180 / 13300 |
| SGLang 0.5.23 | 5.62 | 10.97 | 34.14 | 3.003 | 3.007 | 3.054 | 393.0 | 14758 / 14758 / 14794 |
| TensorRT-LLM 1.3.0 | 6.37 | 10.53 | **33.72** | 3.006 | 3.044 | 3.106 | 397.2 | 13800 / 13796 / 13854 |
| llama.cpp bf16 | 6.19 | 11.75 | 41.92 | 3.112 | 3.147 | 3.206 | 411.9 | 2860 / 2900 / 2950 |
| HF eager | 6.42 | 11.11 | 37.37 | 4.465 | 4.459 | 4.586 | 577.5 | 2854 / 2908 / 3090 |
| HF static cache + compile | 7.55 | 11.95 | 38.31 | 3.531 | 3.798 | 5.041 | 494.5 | 2976 / 3024 / 3284 |
| *(llama.cpp f16 — another type, not compared)* | *6.30* | *10.49* | *36.81* | *3.117* | *3.138* | *3.208* | *409.4* | *2860 / 2902 / 2952* |

Mapal / best tool: TTFT 1.44 (gpt-fast) / 1.16 (gpt-fast) / 1.24 (trtllm); decode 1.18 (gpt-fast) / 1.19 (gpt-fast) / 1.19 (gpt-fast).
Tokens equal before the first difference, Mapal bf16 vs hf-bf16-eager (`hf_reference.py`, eager, on the box): 49 / 7 / 61 of 128.

#### h100 — f32 class

| tool | TTFT 128 ms | TTFT 512 | TTFT 2048 | decode 128 ms/token | decode 512 | decode 2048 | e2e 512 ms | peak MB (128 / 512 / 2048) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Mapal** (as written: f32 activations on the bf16 weights) | 10.76 | **21.79** | **60.89** | **2.055** | **2.057** | **2.139** | 281.1 | 3432 / 3518 / 3296 |
| gpt-fast f32 | **10.37** | 33.18 | 148.9 | 2.220 | 2.159 | 2.256 | 308.0 | 6664 / 6994 / 8040 |
| vLLM f32 | 14.90 | 32.67 | 140.9 | 2.230 | 2.300 | 2.516 | 328.4 | 45236 / 45236 / 45236 |
| llama.cpp f32 GGUF | 29.22 | 32.27 | 121.8 | 2.867 | 3.157 | 4.144 | 435.1 | 5408 / 5460 / 5656 |
| HF eager f32 | 11.91 | 29.62 | 131.9 | 5.676 | 5.861 | 9.083 | 774.7 | 5462 / 5582 / 6856 |
| HF static + compile f32 | 17.28 | 33.43 | 128.1 | 3.562 | 4.360 | 7.216 | 585.9 | 5672 / 5766 / 7176 |

Mapal / best tool: TTFT 1.04 (gpt-fast-f32) / 0.74 (hf-eager-f32) / 0.50 (llama.cpp-f32); decode 0.93 (gpt-fast-f32) / 0.95 (gpt-fast-f32) / 0.95 (gpt-fast-f32).
Tokens equal before the first difference, Mapal f32 vs hf-f32 (`hf_reference.py`, eager, on the box): 128 / 128 / 128 of 128.

#### h100 — bf16 class

| tool | TTFT 128 ms | TTFT 512 | TTFT 2048 | decode 128 ms/token | decode 512 | decode 2048 | e2e 512 ms | peak MB (128 / 512 / 2048) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Mapal** (`fold: bf16 \| f32`) | 4.13 | 7.60 | 25.26 | 2.039 | 2.061 | 2.170 | 267.3 | 2908 / 2958 / 3156 |
| gpt-fast (max-autotune, CUDA graphs) | **3.18** | **4.04** | **10.65** | **1.369** | **1.324** | **1.435** | 172.9 | 3756 / 3878 / 4440 |
| vLLM 0.33.0 | 12.21 | 12.38 | 13.39 | 1.641 | 1.725 | 1.611 | 232.0 | 45384 / 45384 / 45384 |
| SGLang 0.5.23 | 9.81 | 9.30 | 12.21 | 1.566 | 1.558 | 1.569 | 207.5 | 46632 / 46632 / 46656 |
| TensorRT-LLM 1.3.0 | 17.75 | 17.89 | 18.49 | 1.410 | 1.452 | 1.496 | 202.6 | 42640 / 42652 / 42694 |
| llama.cpp bf16 | 23.90 | 14.78 | 28.12 | 1.932 | 1.848 | 1.877 | 250.5 | 3100 / 3140 / 3190 |
| HF eager | 7.57 | 11.20 | 14.25 | 5.924 | 8.996 | 8.827 | 1153.8 | 3122 / 3156 / 3326 |
| HF static cache + compile | 8.63 | 12.88 | 17.78 | 2.207 | 2.371 | 2.637 | 314.2 | 3270 / 3350 / 3522 |
| *(llama.cpp f16 — another type, not compared)* | *9.96* | *11.73* | *34.09* | *1.681* | *1.923* | *1.925* | *256.8* | *3100 / 3142 / 3192* |

Mapal / best tool: TTFT 1.30 (gpt-fast) / 1.88 (gpt-fast) / 2.37 (gpt-fast); decode 1.49 (gpt-fast) / 1.56 (gpt-fast) / 1.51 (gpt-fast).
Tokens equal before the first difference, Mapal bf16 vs hf-bf16-eager (`hf_reference.py`, eager, on the box): 43 / 7 / 66 of 128.

#### Five-part summary — Mapal / best engine of the class (< 1 = Mapal faster)

| part | class | TTFT 128 | TTFT 512 | TTFT 2048 | decode 128 | decode 512 | decode 2048 | tokens vs HF (128 / 512 / 2048) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| rtx-5080 | f32 | **0.62** | **0.62** | **0.53** | **0.55** | **0.58** | **0.58** | 128 / 128 / 128 |
| rtx-5080 | bf16 | 1.13 | 1.05 | 1.15 | 1.10 | 1.10 | 1.06 | 128 / 32 / 61 (`full-model.md`, first repetition) |
| rtx-3090 | f32 | 1.04 | **0.95** | **0.82** | **0.65** | **0.66** | **0.67** | 128 / 128 / 128 |
| rtx-3090 | bf16 | 1.55 | 1.17 | 1.30 | 1.23 | 1.23 | 1.25 | 49 / 128 / 66 |
| a100 | f32 | **0.73** | **0.52** | **0.43** | **0.86** | **0.91** | **0.89** | 128 / 128 / 128 |
| a100 | bf16 | 1.87 | 1.44 | 2.12 | 1.56 | 1.55 | 1.57 | 49 / 32 / 57 |
| rtx-4090 | f32 | **0.72** | **0.64** | **0.53** | **0.62** | **0.63** | **0.63** | 128 / 128 / 128 |
| rtx-4090 | bf16 | 1.44 | 1.16 | 1.24 | 1.18 | 1.19 | 1.19 | 49 / 7 / 61 |
| h100 | f32 | 1.04 | **0.74** | **0.50** | **0.93** | **0.95** | **0.95** | 128 / 128 / 128 |
| h100 | bf16 | 1.30 | 1.88 | 2.37 | 1.49 | 1.56 | 1.51 | 43 / 7 / 66 |

The RTX 5080 (`local-rtx-5080/full-model.md`, `3c629486` = `b8429388`'s emission, measured by the local step):
f32 class 0.62 / 0.62 / 0.53 TTFT and 0.55 / 0.58 / 0.58 decode against the best f32 engine; bf16 class 1.13 / 1.05 /
1.15 TTFT and 1.10 / 1.10 / 1.06 decode against the best bf16 engine; f32 tokens 128 / 128 / 128.

**Five parts**: the f32 class leads every f32-compute engine on decode on all five parts (0.55–0.95×) and on TTFT
from 512 up (0.43–0.95×); at 128 it ties or trails by 4 % on the 3090 and H100 (llama.cpp f32, gpt-fast f32). The
bf16 class trails the best bf16 engine on every part: closest on the RTX 5080 (1.05–1.15× TTFT, 1.06–1.10× decode),
then the 4090 / 3090 (1.16–1.55×, 1.18–1.25×), furthest on the datacenter parts where gpt-fast's CUDA-graph decode
and TensorRT-LLM / SGLang prefill are strongest (A100 1.44–2.12× / 1.55–1.57×, H100 1.30–2.37× / 1.49–1.56×).
Mapal's decode step is ~2.0–4.2 ms on every part while gpt-fast's single-graph step is 1.32–3.33 ms: the launch-bound
step the RTX 5080 report attributes to the host's issue cost (`full-model.md`, "Decode is bound by the host's
launches") is the larger share on the faster GPUs.

## hopper-wgmma (16-bit warpgroup tiles on Hopper, the rented disasm, the walk's reads)

Branch `local-rtx-5080`: merges of `hopper/wgmma16` (`779f4a15`, evidence [wgmma-16bit](../hopper/wgmma-16bit.md)) and
`hopper/disasm` (`16f46432`, [disasm-rented](../hopper/disasm-rented.md)); `fix/bound-blkdec` was already in
(`9a8bcd07`). Then the one Mapal regression the disasm found, attacked here (`walk-reuse`). No rental in this step; the
two branches spent $1.83 (one H100 SXM, 47 min) and $2.63 (H100 / A100 / 4090 / 3090).

**H100, the bf16 class on its warpgroup product** (`wgmma.m64nNk16` as geometry: `a` from shared memory or registers,
`b` in either major, `Caps::matrix_group`, `rules::wgmma16_pays`, OFF `--gpu-off=wgmma16`; same H100 SXM box, gpt-fast
on it):

| prompt | TTFT before → after (ms) | gpt-fast (ms) | Mapal / gpt-fast before → after |
|---|---|---:|---|
| 128 | 3.92 → 2.70 | 3.58 | 1.09 → **0.75** |
| 512 | 7.44 → 5.60 | 4.47 | 1.66 → 1.25 |
| 2048 | 24.81 → 18.11 | 11.45 | 2.17 → 1.58 |

Decode unchanged (1.98–2.06 against 1.33–1.42 ms a token). The 10 changed suite programs (the bf16 projections and the
two bf16 prefills) 1.13–2.23× faster, 2048-row projections at 51–69 % of the measured 998 Tflop/s (cuBLAS 67–77 %); 72 /
72 generated programs within the bound on the part, value gates 10 / 10, tokens on = off. The warp-specialized kernel
launches alone (a horizontal merge passed its tensor maps by value: a fault; root cause open); `ws-persist` on the
16-bit tile measured mixed (0.95–1.04) and stays off.

**What got worse between rounds** (disasm-rented, same-box A/B of the f1d3a04f and b8429388 builds): new / old ≤ 1.02 on
37 of 38 rows — the rest is the boxes (a clock-locked 4090 and slow hosts flattering the earlier round, 500 / 270 W
power caps on its H100 / 3090, the H100 host's 10.7 µs synchronous copy-out, a noisy A100 with the tuner reverting on
noise). The one Mapal regression, `mma-walk` on Ampere, is fixed below. Ranked causes of the bf16 class's gap: (1)
16-bit tiles on `mma.sync` on Hopper — done above; (2) decode GEMVs load bf16 weights 2 bytes at a time (H100 41–65 % of
DRAM; gpt-fast 16-byte loads); (3) attention on `mma.sync` with 2-byte K / V loads (2048: H100 6.87 vs 1.51 ms); (4) the
per-token host round trip (513 KB of logits out, argmax on the host: H100 259 µs of a 2.02 ms step); (6) 340 decode
launches a step on the H100 against gpt-fast's 200; (7) f32 activations in prefill elementwise cells; (8) TF32 `wgmma`
operands rounded in a register pass.

**The walk's reads (`walk-reuse`).** The disasm's reading of the 3090 SASS (static HMMA 128 → 64: "the walk's loop
halves") is a static count: on sm_86 both loops run 64 HMMA, 16 LDSM and 8 LDGSTS a trip; the ordered kernel's other 64
are LLVM's peel of its last trip (`opt` remark "peeled loop by 1 iterations"), the walk's one extra branch its issue
under a run-time bound. What the walk changes is where an operand slice's readers sit: block `c` runs units `[c·iters,
(c+1)·iters)`, so the tiles that share an `a` (`b`) slice read it at times the walk's rotation (`panels` against
`iters`) scatters; between two readers the walk streams `ctas` slices a unit. The 3090's 6 MiB keep holds 2.4 units of
that stream and its readers sit 16 apart: 990 MB read for 50 MB of operands, 1.11 ms at the DRAM rate against a 1.05 ms
pipe (measured 1275 + 39 against the ordered grid's 1082 µs); the A100's 20 MiB, 3 units against 8: 1.1 GB (788 against
761 µs); the 5080's 64 MiB holds the very 3090 grid's readers (628.5 + 21.2 against 666.6). The fix reads properties
only: `MmaWalk::walk_bytes` (each slice once per run of readers the part's `l2_keep_bytes` holds), the walk priced no
faster than its reads, and refused where its re-reads make the site DRAM-bound (`Gpu::dram_bound`, C11's line) when
reading once does not. Rejected on the way: pricing every arm at its bytes over the `DRAM_BOUND_PM` rate refused
`pt_pj_m128_n6144_k4096_bf16`'s walk on the 5080, which wins (below). Local RTX 5080 rows (ABBA, CUPTI, µs with zero
passes; the replay lines `walk=MxNxK`, raw `.claude/raw-evidence/hopper/integrate/walk/`):

| site | OFF (the rules' grid / cut) | walk | reads (once) | price |
|---|---:|---:|---|---|
| 3072 × 4096 × 8192 | 1836.0 | 1829.2 + 39.8 | 1.7 GB (168 MB): DRAM-bound | refuse ✓ |
| 2048 × 4096 × 8192 | cut 4: 1297.9 + 21.0 | **1184.7 + 21.0** | 305 MB | walk ✓ |
| 4096 × 2048 × 8192 | cut 4: 1292.9 + 20.8 | **1198.4 + 20.7** | 404 MB | walk ✓ |
| `pt_pj_m128_n6144_k4096_bf16` | cut 8: 75.3 + 2.0 | **64.0 + 1.9** | 55 MB, once | walk ✓ |
| `pt_pj_m2048_n6144_k4096_bf16` | **929.2** | 1005.2 + 37.5 | 190 MB | refuse ✓ |
| `pt_pj_m128_n28672_k4096_bf16` | 332.8 | 325.5 + 6.8 | 251 MB, once | refuse (null) |

Replay: `mma_walk_replay` 17 measured walks, 14 picked, 3 open (the two forced 64 × 128 wins: the warp tile's realized
efficiency; the 3090 grid on the 5080, 2.5 % inside the 5 % margin); `rung_rules_replay` 304 lines: 181 picked, 83 null,
18 mixed, 22 open. Emission against the pre-fix HEAD (211 suite programs, seven profiles): RTX 5080, 4090, A100, H100 /
H100 PCIe unchanged; RTX 3090 `pt_pj_m2048_n4096_k4096_bf16` back to the ordered grid (the f1d3a04f kernel); RTX 5090
`pt_pj_m2048_n4096_k14336_bf16` walk → cut 8 (2.1 GB of reads against 210 MB on its nominal 96 MiB keep: predicted, not
measured). The A100 model's down projection (residual epilogue) back to the ordered grid. The RTX 5080 against
`c808622c`: the 211 programs and the full model (both classes, 128 / 512 / 2048) emit the same text; tokens f32 128 /
128 = HF f32, bf16 128 / 128 = HF bf16 at prompt 128. Tests: `walk_reuse_bounds` (64 generated grids: each slice read at
least once and at most once a reader, monotone in the keep), `mma_walk_rules_on_generated_shapes` on five parts with the
RTX 3090 (48 generated GEMMs: every taken walk ungated, 27 forced walks gated and kept ordered). Full workspace suite
(`cargo test --release --workspace`, once): 1728 passed, 0 failed, 11 ignored.

**Open.** The walk's order itself (blocks whose panel offsets align so readers meet within the keep) — what would make
the walk pay on small-L2 parts; the 5090's measured keep and its one flipped program; the A100 down projection's
two-round grid (256 tiles on 216 slots; an epilogue group takes no cut); from wgmma-16bit: attention on the warpgroup
product (FA3-style; 7.0 of 18.4 ms of the H100's 2048 prefill against flex attention's 1.5), the projections' epilogue
by TMA store, persistent walk and cluster multicast, small-`m` stage depth, the split forms, the merged-launch fault's
root cause; from disasm-rented: causes 2–4 and 6–8, the host round trip as a measured property, a tuner revert only on
pairs that agree beyond the null. The facts-table rows both branches propose for CLAUDE.md (`wgmma_half_per_clk`,
`Caps::matrix_group`, `MatrixOp::mn_major`, `rules::wgmma16_pays`) are not applied here.
