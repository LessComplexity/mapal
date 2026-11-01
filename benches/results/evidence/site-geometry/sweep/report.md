### Workload x arch (default face; Mapal = median iter ms of the shipped build; cells: tile_bench ms, passes median)

(* = competitor leg carried from the recurrence-precision sweep: not measured on this box in this run)

| workload | arch | Mapal ms | best other (leg) | Mapal/best | previous Mapal/best | measured / carried |
|---|---|---|---|---|---|---|
| attn_1024_128 | rtx-4090 | 0.07731 | 0.0451 (torch-compile) | 1.71 | new | measured |
| attn_1024_128 | rtx-5090 | 0.05706 | 0.0348 (torch-compile) | 1.64 | new | measured |
| attn_1024_64 | rtx-4090 | 0.0705 | 0.0338 (torch-compile) | 2.09 | 1.97 | measured |
| attn_1024_64 | rtx-5090 | 0.05261 | 0.0638 (torch-compile) | 0.82 | 1.39 | measured |
| attn_2048_128 | rtx-4090 | 0.1227 | 0.0891 (torch-compile) | 1.38 | new | measured |
| attn_2048_128 | rtx-5090 | 0.109 | 0.085 (torch-compile) | 1.28 | new | measured |
| attn_2048_64 | rtx-4090 | 0.08182 | 0.0696 (torch-compile) | 1.18 | new | measured |
| attn_2048_64 | rtx-5090 | 0.08135 | 0.0714 (torch-compile) | 1.14 | new | measured |
| causal_1024_256 | rtx-4090 | 43.97 | 201.7 (torch-compile) | 0.22 | 0.21 | measured |
| causal_1024_256 | rtx-5090 | 48.15 | 196.1 (torch-compile) | 0.25 | 0.40 | measured |
| causal_1024_256 | h100-pcie | 44.01 | 175.2 (torch-compile) | 0.25 | 0.54 | measured |
| causal_1024_256 | rtx-3090 | 46.92 | 286.7 (torch-compile*) | 0.16 | 0.16 | measured |
| convmc_8_64_56_3 | rtx-4090 | 0.09483 | 0.0686 (torch) | 1.38 | 1.66 | measured |
| convmc_8_64_56_3 | rtx-5090 | 0.08566 | 0.0758 (torch) | 1.13 | 1.04 | measured |
| convmc_8_64_56_3 | h100-pcie | 0.1168 | 0.0987 (torch) | 1.18 | 1.29 | measured |
| convmc_8_64_56_3 | rtx-3090 | 0.1653 | 0.1024 (torch*) | 1.61 | 1.68 | measured |
| convmc_8_64_56_3_tf32 | rtx-4090 | 0.08635 | 0.0451 (torch-eager-tf32) | 1.91 | 2.07 | measured |
| convmc_8_64_56_3_tf32 | rtx-5090 | 0.08518 | 0.0471 (torch-eager-tf32) | 1.81 | 1.36 | measured |
| convmc_8_64_56_3_tf32 | h100-pcie | 0.1164 | 0.0444 (torch-eager-tf32) | 2.62 | 2.38 | measured |
| convmc_8_64_56_3_tf32 | rtx-3090 | 0.1671 | 0.087 (torch*) | 1.92 | 1.69 | measured |
| decode_1024_256 | rtx-4090 | 36.08 | 30.32 (cache) | 1.19 | 1.03 | measured |
| decode_1024_256 | rtx-5090 | 47 | 42.67 (nocache) | 1.10 | 1.52 | measured |
| decode_1024_256 | h100-pcie | 34.75 | 22.99 (cache) | 1.51 | 2.48 | measured |
| decode_1024_256 | rtx-3090 | 38.74 | 32.7 (cache*) | 1.18 | 1.12 | measured |
| diamond_512 | rtx-4090 | 0.07248 | 0.0389 (torch-compile) | 1.86 | new | measured |
| diamond_512 | rtx-5090 | 0.06318 | 0.0348 (torch-compile) | 1.82 | new | measured |
| fanout_512 | rtx-4090 | 0.05927 | 0.0307 (torch-compile) | 1.93 | new | measured |
| fanout_512 | rtx-5090 | 0.0448 | 0.0266 (torch-compile) | 1.68 | new | measured |
| gen_1024_512_256 | rtx-4090 | 21.95 | 108.5 (torch-compile) | 0.20 | 0.16 | measured |
| gen_1024_512_256 | rtx-5090 | 24.77 | 101.9 (torch-compile) | 0.24 | 0.44 | measured |
| gen_1024_512_256 | h100-pcie | 22.17 | 90.08 (torch-compile) | 0.25 | 0.52 | measured |
| gen_1024_512_256 | rtx-3090 | 25.93 | 153.6 (torch-compile*) | 0.17 | 0.12 | measured |
| indep_512 | rtx-4090 | 0.06013 | 0.0276 (torch-compile) | 2.18 | new | measured |
| indep_512 | rtx-5090 | 0.04913 | 0.0225 (torch-compile) | 2.18 | new | measured |
| mlp2_512 | rtx-4090 | 0.1024 | 0.0573 (torch-compile) | 1.79 | new | measured |
| mlp2_512 | rtx-5090 | 0.08431 | 0.0553 (torch-compile) | 1.52 | new | measured |
| mlp_512 | rtx-4090 | 0.0624 | 0.0297 (torch-compile) | 2.10 | new | measured |
| mlp_512 | rtx-5090 | 0.04214 | 0.0266 (torch-compile) | 1.58 | new | measured |
| pj_m128_n1024_k4096 | rtx-4090 | 0.06959 | 0.0532 (torch-compile) | 1.31 | new | measured |
| pj_m128_n1024_k4096 | rtx-5090 | 0.08565 | 0.0676 (torch-compile) | 1.27 | new | measured |
| pj_m128_n2048_k7168 | rtx-4090 | 0.1764 | 0.1505 (torch) | 1.17 | new | measured |
| pj_m128_n2048_k7168 | rtx-5090 | 0.1426 | 0.0942 (torch) | 1.51 | new | measured |
| pj_m128_n2048_k7168 | rtx-3090 | 0.2891 | no library/torch leg | - | new | measured |
| pj_m16_n1024_k4096 | rtx-4090 | 0.04266 | 0.0358 (torch) | 1.19 | new | measured |
| pj_m16_n1024_k4096 | rtx-5090 | 0.03819 | 0.0205 (torch) | 1.86 | new | measured |
| pj_m16_n1024_k4096 | rtx-3090 | 0.1158 | no library/torch leg | - | new | measured |
| pj_m16_n18432_k7168 | rtx-4090 | 0.6862 | 0.638 (torch) | 1.08 | new | measured |
| pj_m16_n18432_k7168 | h100-pcie | 0.6217 | 0.3114 (torch-compile) | 2.00 | new | measured |
| pj_m16_n18432_k7168 | rtx-3090 | 0.635 | no library/torch leg | - | new | measured |
| pj_m16_n2048_k7168 | rtx-4090 | 0.06616 | 0.0963 (torch) | 0.69 | 0.86 | measured |
| pj_m16_n2048_k7168 | rtx-5090 | 0.07198 | 0.0553 (torch) | 1.30 | 1.21 | measured |
| pj_m16_n2048_k7168 | h100-pcie | 0.1193 | 0.0756 (torch) | 1.58 | 1.35 | measured |
| pj_m1_n1024_k4096 | rtx-4090 | 0.02146 | 0.0184 (torch-compile) | 1.17 | new | measured |
| pj_m1_n1024_k4096 | rtx-5090 | 0.02346 | 0.0184 (torch-compile) | 1.27 | new | measured |
| pj_m1_n1024_k4096 | h100-pcie | 0.02152 | 0.0181 (torch) | 1.19 | new | measured |
| pj_m1_n1024_k4096 | rtx-3090 | 0.04428 | no library/torch leg | - | new | measured |
| pj_m1_n18432_k7168 | rtx-4090 | 0.6335 | 0.6267 (torch) | 1.01 | new | measured |
| pj_m1_n18432_k7168 | rtx-5090 | 0.3221 | 0.3461 (torch) | 0.93 | new | measured |
| pj_m1_n18432_k7168 | h100-pcie | 0.3728 | 0.3086 (torch) | 1.21 | new | measured |
| pj_m1_n18432_k7168 | rtx-3090 | 0.6228 | no library/torch leg | - | new | measured |
| pj_m1_n2048_k7168 | rtx-4090 | 0.04022 | 0.0963 (torch-compile) | 0.42 | new | measured |
| pj_m1_n2048_k7168 | rtx-5090 | 0.03604 | 0.0471 (torch) | 0.77 | new | measured |
| pj_m1_n2048_k7168 | h100-pcie | 0.06561 | 0.0467 (torch) | 1.40 | new | measured |
| pj_m1_n2048_k7168 | rtx-3090 | 0.1113 | no library/torch leg | - | new | measured |
| prefill_1024_512_256 | rtx-4090 | 22.69 | 110.2 (torch-compile) | 0.21 | 0.21 | measured |
| prefill_1024_512_256 | rtx-5090 | 24.95 | 103.1 (torch-compile) | 0.24 | 0.38 | measured |
| prefill_1024_512_256 | h100-pcie | 23.13 | 89.45 (torch-compile) | 0.26 | 0.53 | measured |
| prefill_1024_512_256 | rtx-3090 | 27.71 | 152 (torch-compile*) | 0.18 | 0.16 | measured |
| rmlp1024x2048_128 | rtx-4090 | 0.1033 | no library/torch leg | - | new | measured |
| rmlp1024x2048_128 | rtx-5090 | 0.1002 | no library/torch leg | - | new | measured |
| rmlp256x512_128 | rtx-5090 | 0.05106 | no library/torch leg | - | new | measured |
| rmlp256x512_128 | h100-pcie | 0.05082 | no library/torch leg | - | new | measured |
| rq_diamond_512 | rtx-4090 | 0.1221 | no library/torch leg | - | new | measured |
| rq_diamond_512 | rtx-5090 | 0.1298 | no library/torch leg | - | new | measured |
| rq_indep_512 | rtx-4090 | 0.08978 | no library/torch leg | - | new | measured |
| rq_indep_512 | rtx-5090 | 0.1175 | no library/torch leg | - | new | measured |
| rq_mlp2_512 | rtx-4090 | 0.1723 | no library/torch leg | - | new | measured |
| rq_mlp2_512 | rtx-5090 | 0.1617 | no library/torch leg | - | new | measured |
| rq_mlp_512 | rtx-4090 | 0.1054 | no library/torch leg | - | new | measured |
| rq_mlp_512 | rtx-5090 | 0.1136 | no library/torch leg | - | new | measured |
| rq_twin_512 | rtx-4090 | 0.1371 | no library/torch leg | - | new | measured |
| rq_twin_512 | rtx-5090 | 0.1377 | no library/torch leg | - | new | measured |
| twin_512 | rtx-4090 | 0.05885 | 0.0307 (torch-compile) | 1.92 | new | measured |
| twin_512 | rtx-5090 | 0.04938 | 0.0266 (torch-compile) | 1.86 | new | measured |
| mm_128x1024x1024 | rtx-4090 | 0.0333 | 0.0297 (cublas) | 1.12 | n/a (no earlier cell row) | measured |
| mm_128x1024x1024 | rtx-5090 | 0.0307 | 0.0164 (cublas) | 1.87 | n/a (no earlier cell row) | measured |
| mm_128x1024x2048 | rtx-4090 | 0.0425 | 0.04055 (cublas) | 1.05 | n/a (no earlier cell row) | measured |
| mm_128x1024x2048 | rtx-5090 | 0.0451 | 0.0369 (cublas) | 1.22 | n/a (no earlier cell row) | measured |
| mm_128x1024x4096 | rtx-4090 | 0.0609 | 0.0855 (cublas) | 0.71 | n/a (no earlier cell row) | measured |
| mm_128x1024x4096 | rtx-5090 | 0.0717 | 0.0901 (cublas) | 0.80 | n/a (no earlier cell row) | measured |
| mm_128x2048x1024 | rtx-4090 | 0.0374 | 0.0456 (cublas) | 0.82 | n/a (no earlier cell row) | measured |
| mm_128x2048x1024 | rtx-5090 | 0.0471 | 0.0328 (cublas) | 1.44 | n/a (no earlier cell row) | measured |
| mm_128x2048x7168 | rtx-4090 | 0.1638 | 0.1521 (cublas) | 1.08 | n/a (no earlier cell row) | measured |
| mm_128x2048x7168 | rtx-5090 | 0.1679 | 0.0922 (cublas) | 1.82 | n/a (no earlier cell row) | measured |
| mm_128x2048x7168 | rtx-3090 | 0.3609 | 0.2581 (cublas) | 1.40 | n/a (no earlier cell row) | measured |
| mm_128x256x256 | rtx-5090 | 0.0082 | 0.0092 (cublas) | 0.89 | n/a (no earlier cell row) | measured |
| mm_128x256x256 | h100-pcie | 0.01185 | 0.0135 (cublas) | 0.88 | n/a (no earlier cell row) | measured |
| mm_128x256x512 | rtx-4090 | 0.01145 | 0.0123 (cublas) | 0.93 | n/a (no earlier cell row) | measured |
| mm_128x256x512 | rtx-5090 | 0.0102 | 0.0092 (cublas) | 1.11 | n/a (no earlier cell row) | measured |
| mm_128x512x256 | rtx-4090 | 0.0123 | 0.01125 (cublas) | 1.09 | n/a (no earlier cell row) | measured |
| mm_128x512x256 | rtx-5090 | 0.0102 | 0.0102 (cublas) | 1.00 | n/a (no earlier cell row) | measured |
| mm_128x512x256 | h100-pcie | 0.014 | 0.0138 (cublas) | 1.01 | n/a (no earlier cell row) | measured |
| mm_16x1024x4096 | rtx-4090 | 0.0389 | 0.03685 (cublas) | 1.06 | n/a (no earlier cell row) | measured |
| mm_16x1024x4096 | rtx-5090 | 0.0348 | 0.0205 (cublas) | 1.70 | n/a (no earlier cell row) | measured |
| mm_16x1024x4096 | rtx-3090 | 0.0993 | 0.0379 (cublas) | 2.62 | n/a (no earlier cell row) | measured |
| mm_16x18432x7168 | rtx-4090 | 0.6707 | 0.6154 (cutlass_tuned) | 1.09 | n/a (no earlier cell row) | measured |
| mm_16x18432x7168 | h100-pcie | 0.6089 | 0.41 (cublas) | 1.49 | n/a (no earlier cell row) | measured |
| mm_16x18432x7168 | rtx-3090 | 0.9113 | 0.8791 (cublas) | 1.04 | n/a (no earlier cell row) | measured |
| mm_16x2048x7168 | rtx-4090 | 0.0963 | 0.09575 (cublas) | 1.01 | n/a (no earlier cell row) | measured |
| mm_16x2048x7168 | rtx-5090 | 0.0645 | 0.0532 (cublas) | 1.21 | n/a (no earlier cell row) | measured |
| mm_16x2048x7168 | h100-pcie | 0.1092 | 0.0762 (cublas) | 1.43 | n/a (no earlier cell row) | measured |
| mm_1x1024x4096 | rtx-4090 | 0.03125 | 0.03225 (cublas) | 0.97 | n/a (no earlier cell row) | measured |
| mm_1x1024x4096 | rtx-5090 | 0.0164 | 0.02455 (cublas) | 0.67 | n/a (no earlier cell row) | measured |
| mm_1x1024x4096 | h100-pcie | 0.0245 | 0.018 (cublas) | 1.36 | n/a (no earlier cell row) | measured |
| mm_1x1024x4096 | rtx-3090 | 0.0297 | 0.0266 (cublas) | 1.12 | n/a (no earlier cell row) | measured |
| mm_1x18432x7168 | rtx-4090 | 0.6134 | 0.6134 (cutlass_tuned) | 1.00 | n/a (no earlier cell row) | measured |
| mm_1x18432x7168 | rtx-5090 | 0.3338 | 0.3482 (cublas) | 0.96 | n/a (no earlier cell row) | measured |
| mm_1x18432x7168 | h100-pcie | 0.3666 | 0.3101 (cublas) | 1.18 | n/a (no earlier cell row) | measured |
| mm_1x18432x7168 | rtx-3090 | 0.6052 | 0.6052 (cublas) | 1.00 | n/a (no earlier cell row) | measured |
| mm_1x2048x7168 | rtx-4090 | 0.0922 | 0.106 (cublas) | 0.87 | n/a (no earlier cell row) | measured |
| mm_1x2048x7168 | rtx-5090 | 0.0461 | 0.0451 (cublas) | 1.02 | n/a (no earlier cell row) | measured |
| mm_1x2048x7168 | h100-pcie | 0.06655 | 0.04625 (cublas) | 1.44 | n/a (no earlier cell row) | measured |
| mm_1x2048x7168 | rtx-3090 | 0.0778 | 0.0911 (cublas) | 0.85 | n/a (no earlier cell row) | measured |

### ON/OFF per rung (OFF/ON geomean over programs; W/L = ON faster/slower beyond max(box null, 2%); null = median per-program |null/on - 1|; W/L use each own null)

| rung | arch | kind | n | geomean OFF/ON | wins | losses | max null | worst program (OFF/ON) |
|---|---|---|---|---|---|---|---|---|
| ON:async | h100-pcie | cell | 2 | 0.575 | 0 | 2 | 0.025 | mm_128x256x256 0.53 |
| ON:async | h100-pcie | probe | 1 | 0.487 | 0 | 1 | 0.061 | rmlp256x512_128 0.49 |
| ON:async | rtx-5090 | cell | 1 | 0.804 | 0 | 1 | 0.000 | mm_128x256x256 0.80 |
| ON:async | rtx-5090 | probe | 1 | 0.747 | 0 | 1 | 0.045 | rmlp256x512_128 0.75 |
| ON:row-fit | h100-pcie | cell | 1 | 1.505 | 1 | 0 | 0.000 | mm_16x18432x7168 1.50 |
| ON:row-fit | h100-pcie | probe | 1 | 1.510 | 1 | 0 | 0.004 | pj_m16_n18432_k7168 1.51 |
| ON:row-fit | rtx-3090 | cell | 1 | 1.434 | 1 | 0 | 0.000 | mm_16x18432x7168 1.43 |
| ON:row-fit | rtx-3090 | probe | 1 | 1.021 | 1 | 0 | 0.003 | pj_m16_n18432_k7168 1.02 |
| ON:streamk | rtx-3090 | cell | 2 | 0.993 | 1 | 1 | 0.003 | mm_128x2048x7168 0.85 |
| ON:streamk | rtx-3090 | probe | 2 | 0.955 | 1 | 1 | 0.010 | pj_m128_n2048_k7168 0.85 |
| ON:streamk | rtx-5090 | cell | 6 | 0.996 | 4 | 2 | 0.000 | mm_16x2048x7168 0.73 |
| ON:streamk | rtx-5090 | probe | 9 | 0.986 | 4 | 2 | 0.018 | pj_m16_n1024_k4096 0.73 |
| ON:tf32-radix | rtx-3090 | probe | 1 | 1.260 | 1 | 0 | 0.011 | convmc_8_64_56_3_tf32 1.26 |
| ON:tf32-radix | rtx-5090 | probe | 1 | 1.216 | 1 | 0 | 0.010 | convmc_8_64_56_3_tf32 1.22 |
| ON:warp-fill | rtx-3090 | cell | 2 | 1.212 | 1 | 1 | 0.003 | mm_128x2048x7168 0.70 |
| ON:warp-fill | rtx-3090 | probe | 2 | 1.153 | 1 | 1 | 0.010 | pj_m128_n2048_k7168 0.76 |
| arch-rungs | h100-pcie | cell | 3 | 1.340 | 3 | 0 | 0.017 | mm_1x18432x7168 1.05 |
| arch-rungs | h100-pcie | probe | 2 | 2.572 | 2 | 0 | 0.033 | rmlp256x512_128 2.48 |
| arch-rungs | rtx-5090 | cell | 1 | 1.244 | 1 | 0 | 0.000 | mm_128x256x256 1.24 |
| arch-rungs | rtx-5090 | probe | 1 | 0.910 | 0 | 1 | 0.045 | rmlp256x512_128 0.91 |
| fill-rows | h100-pcie | cell | 2 | 1.643 | 2 | 0 | 0.002 | mm_16x2048x7168 1.36 |
| fill-rows | h100-pcie | probe | 2 | 1.618 | 2 | 0 | 0.015 | pj_m16_n2048_k7168 1.34 |
| fill-rows | rtx-3090 | cell | 1 | 4.125 | 1 | 0 | 0.000 | mm_16x18432x7168 4.13 |
| fill-rows | rtx-3090 | probe | 1 | 3.507 | 1 | 0 | 0.003 | pj_m16_n18432_k7168 3.51 |
| fill-rows | rtx-4090 | cell | 3 | 1.336 | 3 | 0 | 0.000 | mm_16x1024x4096 1.20 |
| fill-rows | rtx-4090 | probe | 3 | 1.641 | 3 | 0 | 0.034 | pj_m16_n1024_k4096 1.40 |
| fill-wave | h100-pcie | cell | 6 | 0.921 | 1 | 4 | 0.006 | mm_1x2048x7168 0.79 |
| fill-wave | h100-pcie | probe | 9 | 1.009 | 5 | 3 | 0.004 | pj_m1_n2048_k7168 0.79 |
| fill-wave | rtx-3090 | cell | 5 | 1.024 | 1 | 0 | 0.001 | mm_16x1024x4096 0.99 |
| fill-wave | rtx-3090 | probe | 9 | 0.990 | 1 | 3 | 0.012 | pj_m1_n2048_k7168 0.90 |
| fill-wave | rtx-4090 | cell | 12 | 0.976 | 3 | 5 | 0.000 | mm_128x1024x1024 0.80 |
| fill-wave | rtx-4090 | probe | 27 | 1.010 | 9 | 9 | 0.026 | rq_twin_512 0.81 |
| fill-wave | rtx-5090 | cell | 13 | 1.070 | 8 | 2 | 0.000 | mm_128x2048x1024 0.65 |
| fill-wave | rtx-5090 | probe | 28 | 0.918 | 4 | 15 | 0.032 | indep_512 0.76 |
| fit-first | h100-pcie | cell | 1 | 0.937 | 0 | 1 | 0.000 | mm_16x18432x7168 0.94 |
| fit-first | h100-pcie | probe | 1 | 0.932 | 0 | 1 | 0.004 | pj_m16_n18432_k7168 0.93 |
| fit-first | rtx-3090 | cell | 1 | 0.947 | 0 | 1 | 0.000 | mm_16x18432x7168 0.95 |
| fit-first | rtx-3090 | probe | 3 | 0.919 | 0 | 2 | 0.003 | convmc_8_64_56_3_tf32 0.88 |
| fit-first | rtx-4090 | cell | 1 | 0.923 | 0 | 1 | 0.003 | mm_16x18432x7168 0.92 |
| fit-first | rtx-4090 | probe | 2 | 1.035 | 1 | 1 | 0.004 | pj_m16_n18432_k7168 0.92 |
| fit-first | rtx-5090 | probe | 2 | 0.792 | 0 | 2 | 0.006 | convmc_8_64_56_3 0.79 |
| panel-regs | h100-pcie | cell | 2 | 1.422 | 2 | 0 | 0.025 | mm_128x512x256 1.35 |
| panel-regs | h100-pcie | probe | 1 | 1.706 | 1 | 0 | 0.061 | rmlp256x512_128 1.71 |
| prefetch | h100-pcie | cell | 2 | 2.055 | 2 | 0 | 0.025 | mm_128x512x256 1.90 |
| prefetch | h100-pcie | probe | 1 | 2.184 | 1 | 0 | 0.061 | rmlp256x512_128 2.18 |
| prefetch | rtx-5090 | cell | 3 | 2.203 | 3 | 0 | 0.000 | mm_128x512x256 1.61 |
| prefetch | rtx-5090 | probe | 1 | 1.265 | 1 | 0 | 0.045 | rmlp256x512_128 1.26 |
| radix-tile | rtx-3090 | probe | 2 | 13.169 | 2 | 0 | 0.006 | convmc_8_64_56_3_tf32 13.11 |
| radix-tile | rtx-4090 | probe | 2 | 10.984 | 2 | 0 | 0.010 | convmc_8_64_56_3 10.45 |
| radix-tile | rtx-5090 | probe | 2 | 8.956 | 2 | 0 | 0.006 | convmc_8_64_56_3 8.92 |
| resident | h100-pcie | cell | 6 | 2.616 | 5 | 0 | 0.006 | mm_128x512x256 1.02 |
| resident | h100-pcie | probe | 9 | 73.034 | 9 | 0 | 0.004 | rmlp256x512_128 43.04 |
| resident | rtx-3090 | cell | 5 | 2.102 | 3 | 0 | 0.001 | mm_16x1024x4096 0.99 |
| resident | rtx-3090 | probe | 9 | 72.847 | 9 | 0 | 0.012 | pj_m128_n2048_k7168 39.82 |
| resident | rtx-4090 | cell | 12 | 1.182 | 5 | 5 | 0.000 | mm_128x1024x1024 0.78 |
| resident | rtx-4090 | probe | 27 | 71.600 | 27 | 0 | 0.026 | rq_twin_512 22.53 |
| resident | rtx-5090 | cell | 13 | 1.460 | 9 | 2 | 0.000 | mm_128x2048x1024 0.65 |
| resident | rtx-5090 | probe | 28 | 75.220 | 28 | 0 | 0.032 | rq_twin_512 24.05 |
| retain-resident | h100-pcie | probe | 3 | 1.502 | 3 | 0 | 0.001 | prefill_1024_512_256 1.47 |
| retain-resident | rtx-3090 | probe | 3 | 2.612 | 3 | 0 | 0.026 | prefill_1024_512_256 2.45 |
| row-split | h100-pcie | probe | 4 | 1.462 | 4 | 0 | 0.002 | prefill_1024_512_256 1.41 |
| row-split | rtx-3090 | probe | 4 | 0.852 | 0 | 3 | 0.017 | prefill_1024_512_256 0.84 |
| row-split | rtx-4090 | probe | 4 | 0.850 | 0 | 4 | 0.009 | causal_1024_256 0.84 |
| row-split | rtx-5090 | probe | 4 | 0.916 | 0 | 2 | 0.134 | decode_1024_256 0.74 |
| splitk | h100-pcie | cell | 6 | 2.614 | 5 | 0 | 0.006 | mm_128x512x256 1.02 |
| splitk | h100-pcie | probe | 8 | 2.213 | 7 | 0 | 0.005 | rmlp256x512_128 1.03 |
| splitk-atomic | rtx-3090 | cell | 1 | 1.258 | 1 | 0 | 0.002 | mm_128x2048x7168 1.26 |
| splitk-atomic | rtx-3090 | probe | 1 | 1.201 | 1 | 0 | 0.012 | pj_m128_n2048_k7168 1.20 |
| tf32-radix | rtx-4090 | probe | 1 | 1.099 | 1 | 0 | 0.012 | convmc_8_64_56_3_tf32 1.10 |
| warp-fill | h100-pcie | cell | 2 | 1.642 | 2 | 0 | 0.002 | mm_16x2048x7168 1.36 |
| warp-fill | h100-pcie | probe | 2 | 1.606 | 2 | 0 | 0.015 | pj_m16_n2048_k7168 1.32 |
| warp-fill | rtx-3090 | cell | 1 | 3.685 | 1 | 0 | 0.000 | mm_16x18432x7168 3.68 |
| warp-fill | rtx-3090 | probe | 1 | 3.507 | 1 | 0 | 0.003 | pj_m16_n18432_k7168 3.51 |
| warp-fill | rtx-4090 | cell | 5 | 1.392 | 5 | 0 | 0.000 | mm_16x1024x4096 1.19 |
| warp-fill | rtx-4090 | probe | 5 | 1.289 | 3 | 2 | 0.012 | convmc_8_64_56_3 0.84 |
| warp-fill | rtx-5090 | cell | 1 | 1.244 | 1 | 0 | 0.000 | mm_128x256x256 1.24 |
| warp-fill | rtx-5090 | probe | 3 | 1.056 | 1 | 2 | 0.010 | convmc_8_64_56_3 0.95 |
| wave-gate | rtx-3090 | cell | 1 | 0.887 | 0 | 1 | 0.005 | mm_16x1024x4096 0.89 |
| wave-gate | rtx-3090 | probe | 1 | 0.903 | 0 | 1 | 0.008 | pj_m16_n1024_k4096 0.90 |

### Per-program OFF/ON for the new rungs

| program | rung | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 |
|---|---|---|---|---|---|
| attn_1024_128 | fill-wave | 1.07 | 0.88 | = | = |
| attn_1024_64 | fill-wave | 1.04 | 0.80 | = | = |
| attn_2048_128 | fill-wave | 0.98 | 1.01 | = | = |
| attn_2048_64 | fill-wave | 1.07 | 0.99 | = | = |
| causal_1024_256 | fill-wave | 0.98 | 1.12 | 1.11 | 1.00 |
| decode_1024_256 | fill-wave | 1.01 | 0.85 | 1.30 | 0.91 |
| diamond_512 | fill-wave | 0.88 | 0.82 | = | = |
| fanout_512 | fill-wave | 0.95 | 0.85 | = | = |
| gen_1024_512_256 | fill-wave | 1.00 | 1.17 | 1.10 | 1.00 |
| indep_512 | fill-wave | 0.87 | 0.76 | = | = |
| mlp2_512 | fill-wave | 0.90 | 0.94 | = | = |
| mlp_512 | fill-wave | 0.89 | 0.91 | = | = |
| mm_128x1024x1024 | fill-wave | 0.80 | 1.00 | = | = |
| mm_128x1024x2048 | fill-wave | 0.98 | 1.14 | = | = |
| mm_128x1024x4096 | fill-wave | 1.08 | 1.26 | = | = |
| mm_128x2048x1024 | fill-wave | 0.90 | 0.65 | = | = |
| mm_128x2048x7168 | fill-wave | 1.00 | 0.90 | = | 1.13 |
| mm_128x256x256 | fill-wave | = | 1.24 | 1.15 | = |
| mm_128x256x512 | fill-wave | 1.07 | 1.21 | = | = |
| mm_128x512x256 | fill-wave | 0.88 | 1.00 | 1.01 | = |
| mm_16x1024x4096 | fill-wave | 0.97 | 1.30 | = | 0.99 |
| mm_16x2048x7168 | fill-wave | 1.04 | 1.27 | 0.90 | = |
| mm_1x1024x4096 | fill-wave | 1.02 | 1.00 | 0.91 | 1.00 |
| mm_1x18432x7168 | fill-wave | 1.00 | 1.06 | 0.81 | 1.01 |
| mm_1x2048x7168 | fill-wave | 1.00 | 1.11 | 0.79 | 1.00 |
| pj_m128_n1024_k4096 | fill-wave | 1.26 | 0.80 | = | = |
| pj_m128_n2048_k7168 | fill-wave | 1.06 | 0.98 | = | 1.14 |
| pj_m16_n1024_k4096 | fill-wave | 1.36 | 1.05 | = | 0.98 |
| pj_m16_n2048_k7168 | fill-wave | 1.25 | 0.99 | 0.89 | = |
| pj_m1_n1024_k4096 | fill-wave | 1.14 | 0.97 | 1.05 | 1.02 |
| pj_m1_n18432_k7168 | fill-wave | 1.00 | 1.12 | 0.82 | 1.00 |
| pj_m1_n2048_k7168 | fill-wave | 0.90 | 0.97 | 0.79 | 0.90 |
| prefill_1024_512_256 | fill-wave | 1.00 | 1.00 | 1.09 | 0.98 |
| rmlp1024x2048_128 | fill-wave | 1.05 | 0.84 | = | = |
| rmlp256x512_128 | fill-wave | = | 0.88 | 1.03 | = |
| rq_diamond_512 | fill-wave | 0.91 | 0.81 | = | = |
| rq_indep_512 | fill-wave | 1.20 | 0.82 | = | = |
| rq_mlp2_512 | fill-wave | 0.99 | 0.95 | = | = |
| rq_mlp_512 | fill-wave | 1.08 | 0.92 | = | = |
| rq_twin_512 | fill-wave | 0.81 | 0.91 | = | = |
| twin_512 | fill-wave | 0.83 | 0.78 | = | = |
| convmc_8_64_56_3 | fit-first | 1.17 | 0.79 | = | 0.88 |
| convmc_8_64_56_3_tf32 | fit-first | = | 0.80 | = | 0.88 |
| mm_16x18432x7168 | fit-first | 0.92 | = | 0.94 | 0.95 |
| pj_m16_n18432_k7168 | fit-first | 0.92 | = | 0.93 | 0.99 |
| causal_1024_256 | row-split | 0.84 | 1.06 | 1.45 | 0.86 |
| decode_1024_256 | row-split | 0.85 | 0.74 | 1.56 | 0.85 |
| gen_1024_512_256 | row-split | 0.85 | 0.99 | 1.44 | 0.86 |
| prefill_1024_512_256 | row-split | 0.86 | 0.90 | 1.41 | 0.84 |
| convmc_8_64_56_3_tf32 | tf32-radix | 1.10 | = | = | = |
| convmc_8_64_56_3_tf32 | ON:tf32-radix | = | 1.22 | = | 1.26 |
| mm_16x18432x7168 | ON:row-fit | = | = | 1.50 | 1.43 |
| pj_m16_n18432_k7168 | ON:row-fit | = | = | 1.51 | 1.02 |

### Value gates (differential pass/fail)

- rtx-4090: 113 pass, 0 fail
- rtx-5090: 48 pass, 0 fail
- h100-pcie: 100 pass, 0 fail
- rtx-3090: 50 pass, 0 fail
