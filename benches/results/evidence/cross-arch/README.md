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
