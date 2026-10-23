# S62 integrated: standing, hardware use and gaps

[Disassembly index](README.md) · build `6a0e3a0` (`s62-integrated`) · evidence
[`benches/results/gpu/disassembly/s62-integrated/`](../../../../benches/results/gpu/disassembly/s62-integrated/),
[`benches/results/gpu/s62/integrated/`](../../../../benches/results/gpu/s62/integrated/)

Ratio = Mapal / other; above 1 = Mapal slower. A verdict needs the ratio outside that box's null band.
Faces: E Exact, C Contract, R/r ordered reassoc, RC/rc reassoc+contract. GEMM cells: L2-flushed kernel ms.
MLP: kernel-only (5090 steady events; 4090/H100/3090 nsys) or whole-program iter ms
(shipped = LAZY loading, eager = `CUDA_MODULE_LOADING=EAGER`). 3090 library MLP legs flush L2, Mapal's do not (favours Mapal).
No ncu counters (blocked on vast): usage figures come from timing, static/executed SASS and known bytes/FLOPs.

## Standing

| part | vs cuBLAS | vs cuBLASLt (MLP) | vs stock CUTLASS | vs tuned CUTLASS 16x64x16 | vs own base 4e1102c |
| --- | --- | --- | --- | --- | --- |
| RTX 5090 | slower on every GEMM cell, 1.02 (16x18432 E) – 3.41 (128x2048 R); 4096³ C/RC 1.11; 16x18432 E/R/RC 1.02–1.04. MLP kernels: C/RC faster (d256 0.64, d1024 0.81), E/R slower (1.03/1.02 d256, 1.52 d1024) | d256 C/RC 0.94, E/R 1.51/1.50; d1024 C/RC 1.44/1.45, E/R 2.72. First call: Lt 18.6–19.8 ms, cuBLAS 1.02–1.06 ms | faster on all M16/M128 (0.15–0.83) and MLP (0.29–0.54); 4096³ C/RC 1.01, E/R 1.56/1.57 | faster on 16x2048, 16x1024, 4096³ (0.28–0.77), 16x18432 E/R/RC (0.93–0.95), 128x2048 C/RC (0.66/0.71); slower 16x18432 C 1.25, 128x2048 E/R 1.40/1.41. MLP d256 C/RC 1.00/0.99, E/R 1.60/1.59; d1024 C/RC 0.97, E/R 1.83 | MLP kernels C/RC 0.75–0.77, E/R 1.22–1.45; whole program shipped 1.41–2.19, eager 0.77–1.39 |
| RTX 4090 | faster 16x18432 E/C 0.97, 4096³ C/rc 0.94; par 16x18432 r/rc 0.98; slower 16x2048 1.04–1.08, 16x1024 1.05–1.93, 128x2048 1.09–1.37, 4096³ E/r 1.63/1.62. MLP kernels E 1.75/1.96, C 1.22/1.41; warm E 1.28/1.16, C 1.71/1.08; cold < 0.02 (cuBLAS first call 17.4 ms) | kernels E 2.09/2.01, C 1.46/1.44; warm E 1.49/1.21, C 1.98/1.13; cold d256 E 1.19, C 3.38, d1024 < 0.02 (Lt first call 12 ms) | faster everywhere except 4096³ E/r 1.63/1.62; 4096³ C 0.94; MLP faster except cold d256 C 1.00 | faster on every kernel cell (0.32–0.90) except 16x18432 1.00–1.01. MLP kernels E 1.40/1.19, C 0.98/0.86; warm d256 E 1.03, C 1.37, d1024 0.85/0.79; cold d256 C 1.94, d1024 C 1.41, E 0.68/0.91 | MLP C shipped d256 2.29, d1024 1.70; eager 1.58/1.21 (G9) |
| H100 PCIe | slower on every M16/M128 cell (1.13–6.38) and 4096³ E/R 1.69; faster 4096³ C/RC 0.97. MLP shipped 2.58–6.57, eager 1.31–4.41, kernels 1.84–5.62 | shipped 2.65–7.63, eager 1.52–4.54, kernels 2.09–5.87 | faster on all M16/M128 (0.08–0.92), 4096³ C/RC 0.91; 4096³ E/R 1.60. MLP shipped E 0.93/0.56, C 2.22/1.20; eager and kernels 0.36–0.96 | faster 16x2048, 16x1024 (0.23–0.53), 4096³ (0.40–0.71), 128x2048 R 0.68; slower 16x18432 1.05–1.45, 128x2048 E 1.13, C 2.14, RC 1.10. MLP shipped 1.40–7.34, eager 1.10–3.12, kernels 1.12–3.91 | MLP d1024 C shipped 0.4957 vs 0.6226 (win); d256 C 1.33 |
| RTX 3090 | slower everywhere except 16x18432 C 1.00; 1.03 (4096³ C/reassoc, 16x18432 E) – 3.93 (128x2048 C). MLP shipped 1.97–3.27, eager 1.22–3.12 | shipped 2.10–3.50, eager 1.35–3.34 | faster on all M16/M128 (0.15–0.94); 4096³ E 1.74, C 1.02, reassoc 1.01; MLP 0.56–0.93 (eager 0.40–0.88) | faster 16x2048, 16x1024 (0.48–0.67), 128x2048 E/reassoc 0.71/0.60, 4096³ (0.39–0.67); slower 16x18432 E/C/reassoc 1.05/1.02/1.43, 128x2048 C 1.30. MLP shipped 1.13–2.22; eager d1024 E 0.98, others 1.07–1.83 | MLP E shipped d256 1.58, d1024 1.09; eager 1.09/1.03 |

| across parts | result |
| --- | --- |
| stock CUTLASS | Mapal wins every M16/M128 cell on all four parts; loses 4096³ E everywhere (1.56–1.74) and 4096³ C on 3090 only (1.02) |
| cuBLAS | parity or win only at 4096³ C (4090 0.94, H100 0.97) and 16x18432 (4090 0.97–0.98, 3090 C 1.00) |
| cuBLASLt | every MLP comparison a loss, except 5090 d256 C/RC 0.94 and cold calls (Lt first call 12–19 ms) |
| shipped program vs base | slower on every part (G5), except H100 d1024 C 0.4957 vs 0.6226 ms |

## Hardware use

| part | DRAM GB/s | FP32 TFLOP/s | note |
| --- | --- | --- | --- |
| 5090 | 1792 | 104.8 | 450 W cap |
| 4090 | 1008 | 82.6 | |
| H100 PCIe | 2039 | 51.2 | |
| 3090 | 936 | 41.25 | at 1965 MHz, 380 W; 4096³ ran 1605–1680 MHz under the power cap |

| cell (FLOP/B) | part | Mapal | cuBLAS |
| --- | --- | --- | --- |
| 16x2048 (7.8–8), % DRAM | 5090 | 40–44% (C/R) | 71% |
| | 4090 | 57–60% | 66% |
| | H100 | 17–28% | 38.6% |
| | 3090 | 51–63% | 73% |
| 16x1024 (7.8–8), % DRAM | 5090 | 21–26% | 58% |
| | 4090 | 25–49% | 50% |
| | H100 | 8.3–18.7% | 25.8% |
| | 3090 | 22–31% | 57% |
| 16x18432 (8), % DRAM | 5090 | 85% (C 63%) | 87% |
| | 4090 | 85% | 83% (tuned CUTLASS 85%) |
| | H100 | 41–57% | 64% |
| | 3090 | E/C 88–91%, rc 65% | 91% (tuned CUTLASS 93%) |
| 128x2048 (59, ridge), % FP32 | 5090 | E 11.3%, C 24.0% | 38.1% |
| | 4090 | E 22.3%, C 28.0% | 30.6% |
| | H100 | 8.2–25.9% | 52.2% |
| | 3090 | 14–31% (E pipe 52%) | 56% |
| 4096³ C, % FP32 | 5090 / 4090 / H100 / 3090 | 46.5 / 72.2 / 60.6 / 57% | 51.7 / 68.1 / 58.5 / 59% |
| 4096³ E, % FP32 (% of halved roof) | 5090 / 4090 / H100 / 3090 | 30 (60) / 41.8 (84) / 34.6 (69) / 33 (67)% | — |

Stock CUTLASS (1.5–12%) and tuned CUTLASS (4.4–29%) sit below Mapal on M16 DRAM use.

## Hotspots

| hotspot | Mapal | libraries |
| --- | --- | --- |
| occupancy, skinny/ridge cells | 0.14–3 resident warps/SM; one-warp CTAs at 194–255 regs on every part | 6–16 warps/SM via split-K: cuBLAS 4–43 ways (32 on 5090 16x2048, 43 on H100 16x1024), cuBLASLt 4–8 on MLP |
| spills | 5090 C/RC M16 152 B at 255 regs; 3090 16x1024 E 144 B; 3090 128x2048 rc 24 B; 5090 4096³ E 8 B | — |
| async copies | cp.async only in 5090 16x18432 E (8 LDGSTS), 5090 4096³ E (10), 4090 16x1024 E (K16 panel); else LDG.128 → STS | 3–5-stage cp.async/LDGSTS everywhere (`cutlass_80_simt_sgemm_*`, `ampere_sgemm_*` on sm_86/89/120; `sm80_xmma` on sm_90) |
| M16 accumulators | 1–4 | cuBLAS 32 |
| M16 FFMA per shared load | 0.5–2.0 | 5.7–10.7 |
| 4096³ accumulators | 64 (5090), 128 (3090) | 128 |
| 4096³ FFMA per shared load | 5090 16, 3090 21.3 | 5090 20, 3090 21.3 |
| instruction mix | E and R: FMUL+FADD, 2 FP issue slots per MAC; C and RC: 1 FFMA | FFMA |
| MLP kernels | latency/launch-bound; d1024 FP32 5.5–23.1%; 2 launches | cuBLASLt 23.6–33.3%; "fused" epilogue = split-K reduce epilogue, 4 launches (3090 d256: separate globalKernel) |
| MLP whole program | host-dominated: 5090 GPU idle 54–79% of the window; 4090 LAZY kernel share 8–57%; 3090 lazy load 0.16–0.33 µs per KB SASS, images 4.4–7.8× base | — |
| outside the window | no H2D copy or allocation inside; CUDA init + context ~240 ms (5090) | — |

## Ranked gaps

| rank | gap | where | ms attributed | worst ratio |
| --- | --- | --- | --- | --- |
| 1 | G1 E/R issue FMUL+FADD (2 FP instr per MAC) on compute-bound and ridge cells | 4096³ E/R all parts (3090 E only); 128x2048 E/R 5090, E 4090/3090; 4090 MLP E kernels | 17.2459 | 3.41 (5090 128x2048 R vs cuBLAS) |
| 2 | G2 grid underfill: ordered faces never split K, reassoc split too small, MLP never split, 0.14–3 warps/SM | M16 cells all faces; 128x2048 (5090 C/RC, 4090 C/rc, H100 E/R, 3090 rc); MLP M128 kernels all faces | 2.9733 | 3.36 (H100 128x2048 E); M16 3.11 (H100 16x1024 C) |
| 3 | G3 mainloop not pipelined: loads after math or single-stage LDG→STS, 2 barriers per K panel | H100 16x18432 all faces; 5090 4096³ C/RC (+E/R tile share); 3090 4096³ C/rc (+E share) | 2.1950 | 1.55 |
| 4 | G4 per-face tile selection gives one face a slower tile (often one-warp, 64-acc, ~254-reg C CTA) | H100 128x2048 C/RC, MLP C; 3090 128x2048 C, 16x1024 C/rc, MLP C; 5090 16x18432 C; 4090 16x1024 E | 1.5094 | 6.38 (H100 128x2048 C vs cuBLAS) |
| 5 | G5 CUDA lazy function loading inside the timed span; images 4.4–7.8× larger | all parts, residual-MLP programs d256/d1024 × E/C, LAZY | 1.2275 (16 arms) | 7.63 vs cuBLASLt (H100 d256 C); 2.29 vs base (4090 d256 C) |
| 6 | G6 reassoc split-K wave quantization; S=4 bimodal | 3090 16x18432 rc; 4090 128x2048 r | 0.2893 | 1.43 |
| 7 | G7 integrated rungs make Exact MLP kernels slower (panel-regs 8× deeper panel; grid-share on 3090 d1024) | 5090 MLP E/R; 3090 MLP E | 0.1124 | 1.45 vs own base |
| 8 | G8 DRAM-roof residual on 16x18432 | 5090 E/R/RC (+C rest); 3090 E/C (+rc rest); 4090 on par | 0.0872 | 1.05 |
| 9 | G9 per-launch `cuModuleGetFunction` + `cuFuncSetAttribute` (> 48 KB smem) | 4090 MLP C, EAGER | 0.0276 | 1.98 vs cuBLASLt; 1.58 vs base |

Attribution: G1 share = Mapal E/R − max(Mapal C/RC same cell, best library); 3090 128x2048 E counts its whole row.
G2 = Mapal − best library after the G1/G4/G6/G7 shares. G4 = worse face − max(better face, best library).
G5 = shipped LAZY iter − EAGER iter. G9 = EAGER integrated − EAGER base.

## Gap detail

| gap | split (ms) | ratios | mechanism |
| --- | --- | --- | --- |
| G1 | 4096³ 16.6607 (5090 E 1.5514 + R 1.5678; 4090 E 1.5380 + r 1.5124; H100 E 3.1775 + R 3.1783; 3090 E 4.1350); 128x2048 0.5565 (5090 E 0.1679 + R 0.1597; 4090 E 0.0415; 3090 E 0.1874); 4090 MLP d256 0.0069 + d1024 0.0221 | vs cuBLAS 4096³ E/R: 5090 1.72/1.73, 4090 1.63/1.62, H100 1.69/1.69, 3090 1.77; 128x2048 E/R: 5090 3.37/3.41, 4090 1.37, 3090 2.14. vs stock CUTLASS 4096³ E/R 1.56–1.74. Own E/C: 4096³ 1.55/1.73/1.75/1.71, 128x2048 5090 2.12, 4090 1.26 | E roof = ½ FP32 peak. E vs halved roof: 4090 84%, H100 69%, 3090 67%, 5090 60%. 5090 128x2048 E/R issue-bound at 3 warps/SM. 4096³ C on the same tile class beats cuBLAS on 4090/H100: the loss is ordered separate rounding, not the tile. Unused fact: instr/MAC (2 vs 1) does not feed panel depth or register blocking |
| G2 | M16 0.8236 (H100 0.4089, 5090 0.2174, 3090 0.1659, 4090 0.0314); 128x2048 1.5589 (H100 0.9500, 3090 0.3225, 5090 0.2458, 4090 0.0406); MLP 0.5908 (H100 0.2520, 3090 0.1673, 5090 0.1269, 4090 0.0446) | 16x1024 E/C/R/RC vs cuBLAS: H100 2.77/3.11/1.38/1.40, 5090 2.40/2.20/1.80/1.80; 16x2048 H100 2.25/2.08/1.71/1.38; MLP kernels vs Lt: H100 E 2.09/2.62, 4090 E 2.09/2.01 C 1.46/1.44 | 5090 M16: 1.5 warps/SM, 4 acc vs split-K 32, 12 warps/SM, 32 acc. 4090: M=16 split in two 8-row blocks, B crosses L2→SM twice (117 vs 58.7 MB). H100 16x1024: 64 CTAs on 114 SMs, 1 acc, 0.5 MAC/LDS vs split-K 43. Reassoc S=2 on 5090 vs cuBLAS 32; H100 RC at S=8 closes most of it. MLP: one-warp CTAs everywhere; reassoc face not applied to MLP sites (5090 R = E within 0.3%) |
| G3 | 5090 4096³ 1.1182; 3090 4096³ 0.5520; H100 16x18432 0.5248 (E 0.2248, C 0.1593, R 0.0885, RC 0.0522) | H100 16x18432 vs cuBLAS 1.55/1.39/1.22/1.13, vs tuned 1.45/1.30/1.14/1.05; 5090 4096³ C/RC 1.11; 3090 4096³ C/rc 1.03 | H100 vs tuned CUTLASS: equal warps/SM (10.1), acc (8), MAC/LDS (4.0); load cover 0–3 MAC (next panel after math) vs 108–112 (loop top); DRAM 46.0 vs 59.7%. 3090: same 8 warps/SM and 21.3 FFMA/LDS as cuBLAS; 4-stage async + 1 barrier vs LDG→STS + 2 barriers. 5090: 64 acc/128 regs vs 128 acc/210 regs, 4-stage LDGSTS |
| G4 | H100 0.8717 (128x2048 C 0.4239, RC 0.1768; MLP C 0.0590 + 0.2120); 3090 0.4861; 5090 16x18432 C 0.1208; 4090 16x1024 E 0.0308 | within Mapal: H100 128x2048 C/E 1.90, MLP C/E 2.51/2.24; 3090 128x2048 C/E 1.83; 5090 16x18432 C/E 1.35; 4090 16x1024 E/C 1.81 | H100 128x2048 C: 32-thread CTAs, 254 regs, 64 acc, 1.1 warps/SM, 8.2% FP32 (E: 4-warp CTAs, 16 acc). 3090 Filled/Contract rows turning grid-share and panel-regs off not applied to 128x2048. 4090 16x1024 E: K16 panel at 248 GB/s vs C's K512 at 450 GB/s; row `rtx-4090 grid-share Under/Exact` keeps grid-share off |
| G5 | 5090 0.4712; H100 0.3577; 4090 0.3162; 3090 0.0824 | LAZY → EAGER integrated/base: 4090 d256 C 2.29 → 1.58, d1024 C 1.70 → 1.21; 5090 d256 E 1.96 → 1.13, C 1.49 → 0.88, d1024 E 2.19 → 1.39, C 1.41 → 0.77; 3090 d256 E 1.58 → 1.09 | Driver loads each function at its first `cuModuleGetFunction`, called per launch inside the span (`mapal_gpu.c:226`). 4090 d256 C: 186 of 210 µs is loading. Unused fact: host IR lists every launch id |
| G6 | 3090 16x18432 rc 0.2540; 4090 128x2048 r 0.0353 | 3090 rc 1.41× own C, 1.40 vs cuBLAS, 1.43 vs tuned; 4090 r 1.24 vs cuBLAS | 3090: S=2 gives 1152 blocks on 984 slots (1.17 waves), tail drops DRAM 91 → 65%. 4090: S=4 adds 8.4 MB workspace; process medians 0.2039 vs 0.1638 |
| G7 | 5090 d256 E +0.0059, R +0.0058, d1024 E/R +0.0368 each; 3090 d256 E +0.0032, d1024 E +0.0239 | 5090 vs 4e1102c 1.23/1.22/1.45; 3090 1.10/1.18 vs rung-off | 5090: 64 k/trip vs 8, 167 regs vs 79; C kernels got faster. 3090 d256 E: 185 regs vs 80. Stall cause unmeasured (no ncu) |
| G8 | 5090 E 0.0061, R 0.0123, RC 0.0102, C rest 0.0061; 3090 E 0.0325, C 0.0100, rc rest 0.0100 | 5090 vs cuBLAS 1.02/1.04/1.03; 3090 E/C vs tuned 1.05/1.02 | 85–91% vs 87–93% DRAM; 3090 E also pays 2 issue slots/MAC (G1) |
| G9 | d256 C +0.0145 (0.0394 vs 0.0249); d1024 C +0.0131 (0.0768 vs 0.0637) | warm vs Lt 1.98/1.13 | integrated C: GetFunction + SetAttribute + Launch per launch (57–86 KB dyn smem); not seen on 5090/3090 |

## Shared mechanisms

| id | mechanism | derived from (IR fact + TargetProfile fact) | emission | rung / OFF |
| --- | --- | --- | --- | --- |
| M1 | reassoc split-K sizing: S = clamp(⌈target_warps·sms / (blocks·warps_per_block)⌉, 1, K/panel), rounded down so blocks·S ≤ resident slots | `tile_plan` (M, N, K, blocks), `loop_plan` k-trips; sms, regs/SM, smem/SM | partials to workspace preallocated at init; fixed-order reduce kernel | `splitk` (replaces fixed S); OFF S=1 |
| M2 | ordered-face parallelism: acc/thread = max(4, M·N / (32·target_warps·sms)); CTA covers all M rows (B read once) | `tile_plan` M, N; output independence; sms | reqntid/maxntid + minctasm annotations; no reduction | `acc-fit`; OFF = current rows |
| M3 | bytes-in-flight pipeline: stages = clamp(⌊(smem/SM − A tile) / panel_bytes⌋ − 1, 3, 5); next-panel cp.async at loop top | affine panel loop, `bounds_proof`, 16-B alignment; smem/SM, cp.async by sm, dram_bw × latency | `cp.async.cg.shared.global.16` + commit/wait(stages−1), one `barrier0` per panel, `!invariant.load` | extends `prefetch_depth`; OFF = LDG→STS |
| M4 | face-aware tile chooser: instr/MAC (2 E/R, 1 C/RC) feeds register blocking and panel depth; no one-warp CTA when blocks < 2·sms | `elem_plan` rounding fact; sms | one chooser in `tile_plan`; rows face-agnostic unless measured | `face-tile`; OFF = today's rows |
| M5 | host: resolve every launch id on the eager-init thread, cache `CUfunction` per id, set MAX_DYNAMIC_SHARED once | host IR launch-id list | `mapal_gpu.c` init + launch path | `fn-cache`; OFF = per-launch lookup |

## Close and beyond

| gap | close | per part | beyond libraries | effort / order |
| --- | --- | --- | --- | --- |
| G1 | (a) sm ≥ 100: pair two independent outputs into `<2 x float>` fmul/fadd → FMUL2/FADD2, bits identical; profile gains `f32x2`. (b) others: M4 + M3 | 5090 (a)+(b); 4090 M4; H100/3090 (b) | fuse bias/ReLU/residual into the Exact kernel from `path_plan` + `last_use`; skip dead outputs | M; order 1 |
| G2 | R/RC: M1. E/C: M2 + M3. MLP: apply reassoc face + M1; M4 forbids one-warp CTAs | H100/5090 target 8–12 warps; 4090 C already 1.0× | L2 persisting window for B with use count > 1 (fits L2 on 4090/5090/H100, not 3090); MLP h dies after GEMM2 → one launch, h in smem/DSMEM on sm_90 | L; after M1–M4, G3; order 2 |
| G3 | M3 wherever `async` is on; load at loop top closes H100 16x18432 | all | cross-op prefetch: next launch's first panel cp.async in the previous tail (PDL exists) | M; order 1 |
| G4 | M4; drop `rtx-4090 grid-share Under/Exact` and 3090 Filled/Contract rows only after A/B | H100/3090 128x2048 C, MLP C; 5090 16x18432 C; 4090 16x1024 E | — | S; order 1 |
| G5 | M5 (beats `CUDA_MODULE_LOADING=EAGER`: no env) | all | preload only kernels on `path_plan` | S; order 0 |
| G6 | M1 one-wave clamp; workspace from a preallocated arena (partials die at reduce) | 3090 16x18432 rc; 4090 128x2048 r | — | S; part of M1 |
| G7 | panel-regs only if ≥ 4 warps/SM survive; grid-share off when blocks ≥ sms | 5090, 3090 | superseded by G2 MLP fusion | S; order 0 |
| G8 | none (within null on 5090; 3090 E share is G1(b)) | — | — | 0 |
| G9 | M5 | 4090 C | — | S; with G5 |

| legality | rule |
| --- | --- |
| E/C | no cross-thread/CTA reduction; M2 only reassigns outputs; per-output serial k order kept; f32x2 lane-wise RN; `fmuladd` only for C/RC |
| R/RC | reduce in fixed s = 0..S−1 order: deterministic, differs from serial only under the opt-in face (ADR-0040) |
| M3 | cp.async needs a full-panel `bounds_proof` (else predicated LDG) and 16-B affine alignment; `wait.group` before the panel's first LDS |
| M5 | function handles are per-module constants |

## Pre-registered acceptance

One box per part, same session, ABBA, ≥ 2 processes × 30 events, value gate vs interpreter first.

| test | cells | arms | keep rule |
| --- | --- | --- | --- |
| T0 host | MLP d256/d1024 E/C whole program | fn-cache OFF/ON | ON ≤ EAGER iter in G5, else remove |
| T1 pipeline + face | 4096³ all faces; H100 16x18432; 3090 4096³ | async old / M3; face-tile OFF/ON; cuBLAS; CUTLASS | ON beats OFF by > null; no cell loses > null, else gate by part row |
| T2 f32x2 | 5090 4096³ E/R, 128x2048 E/R | OFF/ON, cuBLAS | E/C ratio ≤ 1.2; bits equal |
| T3 fill | M16, 128x2048, MLP kernels, all faces | acc-fit / splitk OFF/ON; cuBLAS / cuBLASLt | ON ≤ 1.15× library or remove; RC/rc run-to-run identical |
| T4 retention | 16x1024x4096 looped ×32 | L2 window OFF/ON | ON beats OFF by > null on 4090/5090/H100; 3090 row off |

## Order

| order | work | size |
| --- | --- | --- |
| 0 | G5/G9 (M5), G7 gates | S, no deps |
| 1 | G3 (M3), G4 (M4), G1 (f32x2 + M4) | M, parallel |
| 2 | G2/G6 (M1, M2, MLP reassoc, fusion, retention) | L, after 1 |
| — | G8 | none |

Touch sites: `crates/backends/llvm/src/func/gpu.rs` (`tile_plan`, `prefetch_depth`, `kernel_text`, `launch`),
`crates/backends/llvm/src/profile.rs` (`GPU_RUNGS_OFF`), `crates/mapal-rt/gpu/mapal_gpu.c` (`mapal_gpu_init`, per-launch lookup at line 226).
