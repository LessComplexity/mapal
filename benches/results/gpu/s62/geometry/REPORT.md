# S62 ordered tile geometry (`geometry`)

Branch `s62/geometry`: d380843 (five rungs), cc49cae (fact gates), c862247 + 5dff8df (per-part
`GPU_RUNGS_OFF` rows). Verdict: **partial**. The projection kernels win on every part, but two regressions on the
residual-MLP probes have no fact gate yet, so every part is `gate`:
- the Exact-face M128 kernels on the 5090 and H100 PCIe (traced kernel ms);
- whole-program single-iteration time on all four parts (`iter ms`).

Rungs, each with an OFF arm (`emit --gpu-off=<name>`): `row-fit`, `wave-gate`, `grid-share`,
`panel-regs`, `fill-rows`. Gates added after v1:
- `fill-rows` only where the vector-floor tile has no geometry on the site's rows;
- `panel-regs` only where `Gpu::warps_per_scheduler(grid, threads) == 0`.

Runs (each: one box per part, two ABBA passes × 30 events, 3 legs round-robin, value gates first):
- `v1/`: d380843 with every rung arm.
- `v2/`: cc49cae with every rung arm. The `GPU_RUNGS_OFF` rows come from this run.
- `v3/`: c862247 (shipped table). Shipped vs all five rungs off (= base, the 4e1102c kernels).

## Shipped vs base kernels (v3)

Ratios are cuBLAS-normalized per pass (<1 = shipped faster). "same PTX" means the arm emitted
identical text and was dropped.

### rtx-5090

| M×N×K | face | shipped ms | base ms | cuBLAS ms | CUTLASS ms | shipped/base (pass 1/2) |
|---|---|---:|---:|---:|---:|---|
| 16×2048×7168 | Exact | 0.0922 | 0.2601 | 0.0512 | 0.5161 | 0.356 (0.357/0.354) |
| 16×2048×7168 | Contract | 0.0819 | 0.2662 | 0.0512 | 0.5161 | 0.304 (0.302/0.306) |
| 16×1024×4096 | Exact | 0.0492 | 0.1536 | 0.0184 | 0.3011 | 0.320 (0.320/0.320) |
| 16×1024×4096 | Contract | 0.0471 | 0.1475 | 0.0184 | 0.3011 | 0.319 (0.319/0.319) |
| 16×18432×7168 | Exact | 0.3461 | 2.1642 | 0.3400 | 0.5612 | 0.159 (0.160/0.159) |
| 16×18432×7168 | Contract | 0.4628 | 2.1038 | 0.3400 | 0.5612 | 0.220 (0.220/0.220) |
| 1×2048×7168 | Exact | 0.2662 | same PTX | 0.0451 | 0.5161 | — |
| 1×2048×7168 | Contract | 0.2539 | same PTX | 0.0451 | 0.5140 | — |
| 1×1024×4096 | Exact | 0.1577 | same PTX | 0.0225 | 0.3011 | — |
| 1×1024×4096 | Contract | 0.1495 | same PTX | 0.0225 | 0.3011 | — |
| 1×18432×7168 | Exact | 0.3911 | same PTX | 0.3502 | 0.5571 | — |
| 1×18432×7168 | Contract | 0.3850 | same PTX | 0.3523 | 0.5561 | — |
| 128×2048×7168 | Exact | 0.3031 | 0.6339 | 0.0942 | 0.5632 | 0.479 (0.477/0.481) |
| 128×2048×7168 | Contract | 0.1495 | 0.6369 | 0.0942 | 0.5632 | 0.235 (0.234/0.236) |
| 128×1024×4096 | Exact | 0.1475 | 0.3543 | 0.0922 | 0.3092 | 0.409 (0.407/0.411) |
| 128×1024×4096 | Contract | 0.0901 | 0.3543 | 0.0901 | 0.3092 | 0.254 (0.256/0.251) |
| 128×18432×7168 | Exact | 1.0773 | same PTX | 0.5929 | 0.6707 | — |
| 128×18432×7168 | Contract | 0.7588 | same PTX | 0.6021 | 0.6697 | — |
| 1024² | Exact | 0.0696 | same PTX | 0.0471 | 0.0942 | — |
| 1024² | Contract | 0.0594 | same PTX | 0.0471 | 0.0942 | — |
| 4096² | Exact | 4.0740 | same PTX | 2.3869 | 2.6137 | — |
| 4096² | Contract | 2.6363 | same PTX | 2.3931 | 2.6224 | — |

### rtx-4090

| M×N×K | face | shipped ms | base ms | cuBLAS ms | CUTLASS ms | shipped/base (pass 1/2) |
|---|---|---:|---:|---:|---:|---|
| 16×2048×7168 | Exact | 0.1034 | 0.5028 | 0.0952 | 0.4925 | 0.206 (0.207/0.205) |
| 16×2048×7168 | Contract | 0.1004 | 0.5243 | 0.0963 | 0.5345 | 0.191 (0.191/0.191) |
| 16×1024×4096 | Exact | 0.0696 | 0.2918 | 0.0369 | 0.3073 | 0.239 (0.240/0.239) |
| 16×1024×4096 | Contract | 0.0389 | 0.1361 | 0.0363 | 0.3073 | 0.289 (0.284/0.294) |
| 16×18432×7168 | Exact | 0.6195 | 2.9266 | 0.6369 | 1.0445 | 0.207 (0.201/0.213) |
| 16×18432×7168 | Contract | 0.6175 | 3.1068 | 0.6359 | 0.9667 | 0.199 (0.199/0.198) |
| 1×2048×7168 | Exact | 0.2385 | same PTX | 0.1055 | 0.5141 | — |
| 1×2048×7168 | Contract | 0.2253 | same PTX | 0.1065 | 0.5340 | — |
| 1×1024×4096 | Exact | 0.1434 | same PTX | 0.0328 | 0.3072 | — |
| 1×1024×4096 | Contract | 0.1300 | same PTX | 0.0328 | 0.3081 | — |
| 1×18432×7168 | Exact | 0.6124 | same PTX | 0.6236 | 1.0061 | — |
| 1×18432×7168 | Contract | 0.6124 | same PTX | 0.6236 | 1.0066 | — |
| 128×2048×7168 | Exact | 0.1976 | same PTX | 0.1516 | 0.4997 | — |
| 128×2048×7168 | Contract | 0.1679 | 0.1915 | 0.1546 | 0.5407 | 0.872 (0.870/0.874) |
| 128×1024×4096 | Exact | 0.1239 | same PTX | 0.0840 | 0.2888 | — |
| 128×1024×4096 | Contract | 0.0676 | 0.1260 | 0.0891 | 0.3113 | 0.521 (0.532/0.510) |
| 128×18432×7168 | Exact | 1.8329 | same PTX | 0.7107 | 1.0455 | — |
| 128×18432×7168 | Contract | 1.0885 | same PTX | 0.7097 | 1.0271 | — |
| 1024² | Exact | 0.0814 | same PTX | 0.0507 | 0.0875 | — |
| 1024² | Contract | 0.0666 | 0.0676 | 0.0532 | 0.0891 | 0.981 (0.977/0.985) |
| 4096² | Exact | 4.2353 | same PTX | 2.6056 | 2.6030 | — |
| 4096² | Contract | 2.5093 | same PTX | 2.6419 | 2.6507 | — |

### h100-pcie

| M×N×K | face | shipped ms | base ms | cuBLAS ms | CUTLASS ms | shipped/base (pass 1/2) |
|---|---|---:|---:|---:|---:|---|
| 16×2048×7168 | Exact | 0.1598 | 0.5540 | 0.0731 | 0.9200 | 0.286 (0.286/0.287) |
| 16×2048×7168 | Contract | 0.1479 | 0.8587 | 0.0728 | 0.9207 | 0.172 (0.171/0.172) |
| 16×1024×4096 | Exact | 0.1073 | 0.3009 | 0.0307 | 0.5244 | 0.360 (0.357/0.364) |
| 16×1024×4096 | Contract | 0.4058 | 0.4870 | 0.0308 | 0.5242 | 0.834 (0.833/0.835) |
| 16×18432×7168 | Exact | 0.5939 | 2.0426 | 0.3880 | 1.8185 | 0.291 (0.291/0.291) |
| 16×18432×7168 | Contract | 0.5355 | 2.1396 | 0.3879 | 1.8182 | 0.251 (0.251/0.250) |
| 1×2048×7168 | Exact | 0.5047 | same PTX | 0.0458 | 0.9204 | — |
| 1×2048×7168 | Contract | 0.5672 | same PTX | 0.0458 | 0.9200 | — |
| 1×1024×4096 | Exact | 0.2732 | same PTX | 0.0175 | 0.5236 | — |
| 1×1024×4096 | Contract | 0.2934 | same PTX | 0.0176 | 0.5239 | — |
| 1×18432×7168 | Exact | 0.4707 | 0.5489 | 0.3090 | 1.8169 | 0.857 (0.857/0.858) |
| 1×18432×7168 | Contract | 0.6028 | same PTX | 0.3087 | 1.8169 | — |
| 128×2048×7168 | Exact | 0.4475 | 0.4474 | 0.1321 | 0.9293 | 1.000 (0.998/1.001) |
| 128×2048×7168 | Contract | 0.8466 | 1.0184 | 0.1329 | 0.9290 | 0.829 (0.827/0.831) |
| 128×1024×4096 | Exact | 0.2760 | 0.2842 | 0.0517 | 0.5333 | 0.957 (0.953/0.960) |
| 128×1024×4096 | Contract | 0.4569 | 0.5745 | 0.0510 | 0.5335 | 0.792 (0.792/0.792) |
| 128×18432×7168 | Exact | 3.0629 | same PTX | 1.0459 | 1.8504 | — |
| 128×18432×7168 | Contract | 1.7927 | same PTX | 1.0457 | 1.8507 | — |
| 1024² | Exact | 0.2162 | same PTX | 0.0991 | 0.1352 | — |
| 1024² | Contract | 0.1462 | same PTX | 0.0989 | 0.1352 | — |
| 4096² | Exact | 7.2989 | same PTX | 4.3116 | 4.6063 | — |
| 4096² | Contract | 4.1646 | same PTX | 4.3107 | 4.6065 | — |

### rtx-3090

| M×N×K | face | shipped ms | base ms | cuBLAS ms | CUTLASS ms | shipped/base (pass 1/2) |
|---|---|---:|---:|---:|---:|---|
| 16×2048×7168 | Exact | 0.1285 | 0.3502 | 0.0881 | 0.7383 | 0.356 (0.340/0.372) |
| 16×2048×7168 | Contract | 0.1311 | 0.6062 | 0.0891 | 0.7670 | 0.216 (0.216/0.217) |
| 16×1024×4096 | Exact | 0.0655 | 0.1649 | 0.0358 | 0.4403 | 0.397 (0.397/0.397) |
| 16×1024×4096 | Contract | 0.1567 | same PTX | 0.0358 | 0.4403 | — |
| 16×18432×7168 | Exact | 0.6733 | 8.5160 | 0.6375 | 1.7464 | 0.078 (0.078/0.078) |
| 16×18432×7168 | Contract | 0.6451 | 8.2800 | 0.6369 | 1.7449 | 0.077 (0.077/0.077) |
| 1×2048×7168 | Exact | 0.2406 | same PTX | 0.0881 | 0.6845 | — |
| 1×2048×7168 | Contract | 0.2334 | same PTX | 0.0891 | 0.7281 | — |
| 1×1024×4096 | Exact | 0.1464 | same PTX | 0.0256 | 0.4249 | — |
| 1×1024×4096 | Contract | 0.1403 | same PTX | 0.0256 | 0.4403 | — |
| 1×18432×7168 | Exact | 0.6083 | same PTX | 0.6083 | 1.5785 | — |
| 1×18432×7168 | Contract | 0.6052 | same PTX | 0.6083 | 1.5759 | — |
| 128×2048×7168 | Exact | 0.3932 | 0.3625 | 0.1843 | 0.7670 | 1.003 (1.005/1.001) |
| 128×2048×7168 | Contract | 0.5150 | 0.6881 | 0.1853 | 0.7731 | 0.746 (0.749/0.744) |
| 128×1024×4096 | Exact | 0.1997 | 0.2007 | 0.0727 | 0.4106 | 0.972 (0.979/0.966) |
| 128×1024×4096 | Contract | 0.2970 | 0.3834 | 0.0788 | 0.4444 | 0.738 (0.750/0.726) |
| 128×18432×7168 | Exact | 3.2143 | same PTX | 1.6998 | 1.8104 | — |
| 128×18432×7168 | Contract | 1.9220 | same PTX | 1.7910 | 1.9077 | — |
| 1024² | Exact | 0.1941 | same PTX | 0.1157 | 0.1178 | — |
| 1024² | Contract | 0.1475 | 0.1536 | 0.1167 | 0.1188 | 0.966 (0.960/0.972) |
| 4096² | Exact | 11.4007 | same PTX | 6.5823 | 6.7707 | — |
| 4096² | Contract | 6.9468 | same PTX | 6.8485 | 6.9929 | — |

## Residual MLP probes (v3, median of 5, one process each)

The arm is all five rungs off. Kernel ms is the traced kernel sum. Iter ms is the program's own
single-iteration timer, untraced. Ratios are base/shipped (<1 = shipped slower).

| part | probe | face | shipped kernel ms | base kernel ms | ratio | shipped iter ms | base iter ms | ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|
| rtx-5090 | rmlp_128x256x512 | Exact | 0.0410 | 0.0340 | 0.829 | 0.1387 | 0.0617 | 0.445 |
| rtx-5090 | rmlp_128x256x512 | Contract | 0.0260 | 0.0340 | 1.308 | 0.0953 | 0.0642 | 0.674 |
| rtx-5090 | rmlp_128x1024x2048 | Exact | 0.1270 | 0.0890 | 0.701 | 0.2544 | 0.1093 | 0.430 |
| rtx-5090 | rmlp_128x1024x2048 | Contract | 0.0730 | 0.0900 | 1.233 | 0.1464 | 0.1039 | 0.710 |
| rtx-4090 | rmlp_128x256x512 | Exact | 0.0300 | same PTX | — | 0.0563 | same PTX | — |
| rtx-4090 | rmlp_128x256x512 | Contract | 0.0260 | 0.0250 | 0.962 | 0.1332 | 0.0663 | 0.498 |
| rtx-4090 | rmlp_128x1024x2048 | Exact | 0.0840 | same PTX | — | 0.1140 | same PTX | — |
| rtx-4090 | rmlp_128x1024x2048 | Contract | 0.0650 | 0.0660 | 1.015 | 0.1350 | 0.0942 | 0.698 |
| h100-pcie | rmlp_128x256x512 | Exact | 0.0640 | 0.0510 | 0.797 | 0.2097 | 0.1084 | 0.517 |
| h100-pcie | rmlp_128x256x512 | Contract | 0.1100 | 0.1220 | 1.109 | 0.3411 | 0.1570 | 0.460 |
| h100-pcie | rmlp_128x1024x2048 | Exact | 0.2280 | 0.1830 | 0.803 | 0.4067 | 0.2053 | 0.505 |
| h100-pcie | rmlp_128x1024x2048 | Contract | 0.3680 | 0.4570 | 1.242 | 0.4460 | 0.4729 | 1.060 |
| rtx-3090 | rmlp_128x256x512 | Exact | 0.0420 | 0.0410 | 0.976 | 0.1334 | 0.0623 | 0.467 |
| rtx-3090 | rmlp_128x256x512 | Contract | 0.0620 | 0.0680 | 1.097 | 0.1420 | 0.1054 | 0.742 |
| rtx-3090 | rmlp_128x1024x2048 | Exact | 0.1660 | 0.1630 | 0.982 | 0.2932 | 0.1849 | 0.631 |
| rtx-3090 | rmlp_128x1024x2048 | Contract | 0.2120 | 0.2930 | 1.382 | 0.3069 | 0.3143 | 1.024 |

In the probes, the shipped Exact face is slower on the 5090 (base/shipped 0.70, 0.83) and on H100 PCIe (0.80, 0.80).
The v2 per-rung arms (`v2/<part>/probes.txt`, traced kernel ms off/on) put it on
`panel-regs`: 5090 0.694/0.879, H100 0.792/0.785. `grid-share` contributes too: 5090 0.752/0.848,
H100 0.827/0.846. In the same (fill, face) class, `panel-regs` measures 2.04–2.43× on the 5090 mm
cells and 1.05–1.51× on the H100 mm cells. Nothing in this change separates the two cases. On the
5090, the probe's 128×2048 site and the mm cell 128×2048×7168 emit the same launch: 4×4 tile,
64 threads, 256 blocks, 32768 B smem. The mm bench flushes L2 before each launch and has
K = 4096–7168. The probe does not flush, and its K is 256–2048.

## Per-rung decisions (v2, OFF/ON per pass, rungs.py `--null-eps`)

| part | rows added to `GPU_RUNGS_OFF` (OFF/ON) | classes held ON with a stable loss somewhere |
|---|---|---|
| rtx-5090 | wave-gate Filled Contract (16×18432 0.223/0.225); row-fit Filled Exact (0.988/0.994, no effect) | grid-share Under Exact (128×2048 0.795; 16×N 1.22–1.54); wave-gate Under Exact (1×1024 0.915; 16×N 2.07–2.42); panel-regs/grid-share Exact probes |
| rtx-4090 | grid-share Under Exact (mm no effect; probe kernel ms 0.848/0.760); row-fit Filled (16×18432 0.963–0.965) | none |
| h100-pcie | grid-share Filled Contract (1.013/1.014, no effect); row-fit Filled (0.856–0.901); wave-gate Under Contract (16×1024 0.832/0.834) | grid-share Under Exact (128×1024 0.749, 16×1024 0.84; 1×18432 1.16); wave-gate Under Exact (M1 0.82–0.85; 16×1024 2.0); panel-regs/grid-share Exact probes |
| rtx-3090 | grid-share Filled (16×2048 Contract 0.838/0.831); panel-regs Filled (16×2048 0.86–0.94); row-fit Filled (0.962–0.998) | grid-share Under Exact (128×1024 0.84; 16×1024 1.45) |

Full class tables: `v2/<part>/rungs.txt`. v1 (`v1/<part>/rungs.txt`) produced the two gates:
- `fill-rows` against an existing vector tile: 0.53–0.96× in 12 of 19 cells. Against the
  one-thread-per-cell kernel it measured 1.15–5.7× in every cell.
- `panel-regs`: 35 of 38 cells fall on the measured side of the 0-warps-per-scheduler line.
  v1 1024² Exact on the 5090 was 0.786×; that cell is now the same PTX.

## Correctness

- Differential (GPU stdout = interpreter/CPU oracle, every program × every arm), all 0 fail.
  - v1: 313 / 288 / 262 / 257 pass.
  - v2: 191 / 160 / 193 / 190 pass.
  - v3: 136 / 138 / 138 / 138 pass (5090 / 4090 / H100 / 3090).
- tile_bench: Exact and Contract bit-exact vs the in-order fold in every bench. That is
  202/192/194/180, 182/176/172/170 and 112/118/118/116 per run. The TF32 face is bounded.
- Regression `skinny_projections_take_the_ordered_cooperative_tile`: fails on 4e1102c with
  left `None`, right `Some((1, 4, 8, 16, 32, 256, 1))` (`old-code-regression-4e1102c.log`).
- Local tests: `MAPAL_REQUIRE_CLANG=1 cargo test -p mapal-backend-llvm` gives 244 pass / 0 fail /
  1 ignored. `MAPAL_REQUIRE_CLANG=1 cargo test --workspace --release` gives 1199 / 0 / 1.

## Null variation (shipped vs itself, max |s−1| over cells, pass 1 / pass 2)

| run | rtx-5090 | rtx-4090 | h100-pcie | rtx-3090 |
|---|---|---|---|---|
| v1 | 2.8% / 8.5% | 2.7% (max) | 1.0% (max) | 7.7% (max) |
| v2 | 7.9% / 3.8% | 2.9% (max) | 2.6% (max) | 4.3% (max) |
| v3 | 4.4% / 4.9% | 2.0% / 7.0% | 0.7% / 3.4% | 2.7% / 6.8% |

On the cells of 0.3 ms and up, the max null per run is 0.7–2.3% in v1 and 0.4–2.6% in v2. In v3
it is 0.7–3.4%, except 6.9% on the 4090's 128×18432 Exact (a cell this change leaves as the same PTX).

## Register spills in shipped kernels (driver JIT, `local_bytes`)

| part | cell | spill | shipped vs base |
|---|---|---|---|
| rtx-5090 | 16×2048 and 16×1024 Contract (255 regs) | 152 B | 3.1–3.3× faster |
| H100 PCIe, rtx-3090 | 16×1024 Exact | 144 B | 2.5–2.8× faster |
| H100 PCIe | 1×18432 Exact | 1424 B | 1.17× faster |

## Open issues

- Exact face, M128 residual-MLP probes on the 5090 and H100 PCIe: shipped kernel time is 1.21–1.43×
  base. The rungs involved are `panel-regs` and `grid-share`, held ON because their class has
  stable mm-cell gains. This needs a fact-derived gate that separates these sites from the mm cells.
- Classes held ON with a stable loss in one cell are listed in the per-rung table above.
- Whole-program regression on the residual-MLP probes, by the program's own single-iteration
  `iter ms`. Shipped takes 1.35–2.33× the base iter time in 12 of the 14 probe × face cells
  compared (every part, both faces; table above), including cells where its kernel ms is equal or
  better.
  - `v4-probe-order/rtx-4090` reran the probes with the arm order on, base, on, 10 reps. On
    rmlp_128x256x512 Contract, iter ms was 0.1328 and 0.1339 at positions 1 and 3, against base
    0.0611 at position 2. Kernel ms were 0.0375 / 0.0350 / 0.0370. So the excess is not the
    launch order.
  - The shipped probe PTX is 3.8–7.4× the base lines (deep register panels; 3153–7271 vs
    731–1355). The mechanism behind the excess is not established.
  - This blocks a keep verdict for small chained programs on all four parts.
- M1 cells keep the one-thread-per-cell kernel wherever `wave-gate` is ON. With the gate off, M1
  takes a 1×128 register-staged tile that spills 1.4 KB. That tile takes 0.82–0.85× the gate-on
  kernel's time on H100 PCIe and 1.19–1.25× on the 3090; on the 5090 and 4090 the difference is
  within the null.

## Rentals (vast.ai; every instance created here is destroyed)

| run | rtx-5090 | rtx-4090 | h100-pcie | rtx-3090 |
|---|---|---|---|---|
| v1 | 57293231 | 57294918 | 57294920 | 57294919 |
| v2 | 57297489 | 57297492 | 57297490 | 57297491 |
| v3 | 57298823 | 57298826 | 57298825 | 57298824 |
| v4 probe order | — | 57300259 (upload failed), 57301011 | — | — |

Earlier interrupted attempt (no numbers): 57289827–57289830, destroyed.
