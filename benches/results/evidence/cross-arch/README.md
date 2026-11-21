# cross-arch — the property rules and the suite on four rented parts (2026-11-21)

Branch `local-rtx-5080` @ `8bba2a50` (kernel merging, hardware properties, model-scale suite, long rows, onto device) +
this round's tooling (`f4e6e270` vast wrappers, `3968f59c` / `bd9ae103` / `73360fbc` calibration on the box,
`72ee9297` per-instance ssh key). Four vast.ai boxes in parallel, CUDA 13.0.2 devel image, torch 2.11.0+cu130.

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
