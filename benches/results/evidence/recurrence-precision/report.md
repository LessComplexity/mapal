# recurrence-precision sweep (narrowed: measure only what changed)

Changed list: changed.tsv (25 programs; 18 timed pt_* programs). Unchanged rows are carried from cpu-gpu-boundary/table.md. Boxes ran with E2E=0 and TIMING=0 (no square-cell, e2e or mm sections). Mapal = median iter ms (5 ABBA reps, shipped build, default face). The torch legs of pt_gen, the *_tf32 programs and mlp/mlp2 ran in a second box session per arch (torch_bench could not map them in the first); in those rows Mapal is that session's own on-arm (same box). The 3090's pass-1 profile archive was lost with its box, so its nsys rows come from a separate 3090 rental (sweep-rtx-3090/prof-rtx-3090.tgz).

### Workload x arch (default face; Mapal = median `iter ms` of the shipped build on this box)

| workload | arch | Mapal ms | best other (leg) | Mapal/best | cpu-gpu-boundary Mapal/best | measured / carried |
|---|---|---|---|---|---|---|
| causal_1024_256 | rtx-4090 | 40.48 | 194.8 (torch-compile) | 0.21 | 0.62 | measured |
| causal_1024_256 | rtx-5090 | 33.57 | 83.59 (torch-compile) | 0.40 | 1.30 | measured |
| causal_1024_256 | h100-pcie | 55.53 | 103 (torch-compile) | 0.54 | 1.41 | measured |
| causal_1024_256 | rtx-3090 | 44.75 | 286.7 (torch-compile) | 0.16 | 0.75 | measured |
| convmc_8_64_56_3 | rtx-4090 | 0.1124 | 0.0676 (torch) | 1.66 | 17.77 | measured |
| convmc_8_64_56_3 | rtx-5090 | 0.06149 | 0.0594 (torch) | 1.04 | 12.36 | measured |
| convmc_8_64_56_3 | h100-pcie | 0.1049 | 0.0812 (torch) | 1.29 | 20.33 | measured |
| convmc_8_64_56_3 | rtx-3090 | 0.172 | 0.1024 (torch) | 1.68 | 27.57 | measured |
| convmc_8_64_56_3_tf32 | rtx-4090 | 0.1079 | 0.0522 (torch-eager-tf32) | 2.07 | new | measured |
| convmc_8_64_56_3_tf32 | rtx-5090 | 0.06113 | 0.045 (torch-eager-tf32) | 1.36 | new | measured |
| convmc_8_64_56_3_tf32 | h100-pcie | 0.0944 | 0.0396 (torch-eager-tf32) | 2.38 | new | measured |
| convmc_8_64_56_3_tf32 | rtx-3090 | 0.1466 | 0.087 (torch) | 1.69 | new | measured |
| decode_1024_256 | rtx-4090 | 31.98 | 31.02 (cache) | 1.03 | 1.66 | measured |
| decode_1024_256 | rtx-5090 | 29.68 | 19.49 (cache) | 1.52 | 1.60 | measured |
| decode_1024_256 | h100-pcie | 51.91 | 20.96 (cache) | 2.48 | 2.95 | measured |
| decode_1024_256 | rtx-3090 | 36.88 | 33.05 (cache) | 1.12 | 1.69 | measured |
| gen_1024_512_256 | rtx-4090 | 14.1 | 85.67 (torch-compile) | 0.16 | new | measured |
| gen_1024_512_256 | rtx-5090 | 16.84 | 38.45 (torch-compile) | 0.44 | new | measured |
| gen_1024_512_256 | h100-pcie | 26.93 | 51.95 (torch-compile) | 0.52 | new | measured |
| gen_1024_512_256 | rtx-3090 | 19.01 | 153.6 (torch-compile) | 0.12 | new | measured |
| mlp2_2048_tf32 | rtx-4090 | 0.9841 | 0.8499 (torch-eager-tf32) | 1.16 | new | measured |
| mlp2_2048_tf32 | rtx-5090 | 0.8981 | 0.7803 (torch-compile-tf32) | 1.15 | new | measured |
| mlp2_2048_tf32 | h100-pcie | 0.7457 | 0.3996 (torch-compile-tf32) | 1.87 | new | measured |
| mlp2_2048_tf32 | rtx-3090 | 2.829 | 2.426 (torch-compile-tf32) | 1.17 | new | measured |
| mlp_2048_tf32 | rtx-4090 | 0.485 | 0.4567 (torch-compile-tf32) | 1.06 | new | measured |
| mlp_2048_tf32 | rtx-5090 | 0.4449 | 0.3851 (torch-compile-tf32) | 1.16 | new | measured |
| mlp_2048_tf32 | h100-pcie | 0.3662 | 0.209 (torch-eager-tf32) | 1.75 | new | measured |
| mlp_2048_tf32 | rtx-3090 | 1.405 | 1.006 (torch-compile-tf32) | 1.40 | new | measured |
| prefill_1024_512_256 | rtx-4090 | 20.63 | 98.48 (torch-compile) | 0.21 | 0.64 | measured |
| prefill_1024_512_256 | rtx-5090 | 17.22 | 45.1 (torch-compile) | 0.38 | 1.25 | measured |
| prefill_1024_512_256 | h100-pcie | 28.47 | 53.97 (torch-compile) | 0.53 | 1.30 | measured |
| prefill_1024_512_256 | rtx-3090 | 23.78 | 152 (torch-compile) | 0.16 | 0.81 | measured |
| rg_indep_1024 | rtx-4090 | 1.146 | no library/torch leg | - | new | measured |
| rg_indep_1024 | rtx-5090 | 0.9822 | no library/torch leg | - | new | measured |
| rg_indep_1024 | h100-pcie | 1.996 | no library/torch leg | - | new | measured |
| rg_indep_1024 | rtx-3090 | 3.261 | no library/torch leg | - | new | measured |
| rg_indep_2048 | rtx-4090 | 5.303 | no library/torch leg | - | new | measured |
| rg_indep_2048 | rtx-5090 | 4.544 | no library/torch leg | - | new | measured |
| rg_indep_2048 | h100-pcie | 10.45 | no library/torch leg | - | new | measured |
| rg_indep_2048 | rtx-3090 | 16.51 | no library/torch leg | - | new | measured |
| rg_mlp2_1024 | rtx-4090 | 2.103 | no library/torch leg | - | new | measured |
| rg_mlp2_1024 | rtx-5090 | 1.88 | no library/torch leg | - | new | measured |
| rg_mlp2_1024 | h100-pcie | 3.931 | no library/torch leg | - | new | measured |
| rg_mlp2_1024 | rtx-3090 | 6.028 | no library/torch leg | - | new | measured |
| rg_mlp2_2048 | rtx-4090 | 10.3 | no library/torch leg | - | new | measured |
| rg_mlp2_2048 | rtx-5090 | 9.114 | no library/torch leg | - | new | measured |
| rg_mlp2_2048 | h100-pcie | 20.42 | no library/torch leg | - | new | measured |
| rg_mlp2_2048 | rtx-3090 | 32.02 | no library/torch leg | - | new | measured |
| rowsum_2048 | rtx-4090 | 0.3176 | 0.3328 (torch) | 0.95 | new | measured |
| rowsum_2048 | rtx-5090 | 0.2808 | 0.2601 (torch) | 1.08 | new | measured |
| rowsum_2048 | h100-pcie | 0.6345 | 0.5253 (torch-compile) | 1.21 | new | measured |
| rowsum_2048 | rtx-3090 | 1.018 | 0.9144 (torch-compile) | 1.11 | new | measured |
| rowsum_4096 | rtx-4090 | 2.56 | 2.554 (torch) | 1.00 | 1.05 | measured |
| rowsum_4096 | rtx-5090 | 2.097 | 1.966 (torch-compile) | 1.07 | 1.10 | measured |
| rowsum_4096 | h100-pcie | 3.598 | 3.752 (torch-compile) | 0.96 | 0.97 | measured |
| rowsum_4096 | rtx-3090 | 6.859 | 7.174 (torch) | 0.96 | 0.95 | measured |
| rt_indep_1024 | rtx-4090 | 1.075 | no library/torch leg | - | new | measured |
| rt_indep_1024 | rtx-5090 | 0.9617 | no library/torch leg | - | new | measured |
| rt_indep_1024 | h100-pcie | 1.961 | no library/torch leg | - | new | measured |
| rt_indep_1024 | rtx-3090 | 3.233 | no library/torch leg | - | new | measured |
| rt_indep_2048 | rtx-4090 | 5.233 | no library/torch leg | - | new | measured |
| rt_indep_2048 | rtx-5090 | 4.514 | no library/torch leg | - | new | measured |
| rt_indep_2048 | h100-pcie | 10.4 | no library/torch leg | - | new | measured |
| rt_indep_2048 | rtx-3090 | 16.49 | no library/torch leg | - | new | measured |
| rt_mlp2_1024 | rtx-4090 | 2.03 | 1.783 (torch-compile) | 1.14 | new | measured |
| rt_mlp2_1024 | rtx-5090 | 1.84 | 1.534 (torch-compile) | 1.20 | new | measured |
| rt_mlp2_1024 | h100-pcie | 3.898 | 2.684 (torch-compile) | 1.45 | new | measured |
| rt_mlp2_1024 | rtx-3090 | 5.948 | 6.015 (torch-compile) | 0.99 | new | measured |
| rt_mlp2_2048 | rtx-4090 | 10.22 | 13.69 (torch-compile) | 0.75 | 1.12 | measured |
| rt_mlp2_2048 | rtx-5090 | 9.063 | 9.024 (torch-compile) | 1.00 | 1.12 | measured |
| rt_mlp2_2048 | h100-pcie | 20.48 | 17.54 (torch-compile) | 1.17 | 1.27 | measured |
| rt_mlp2_2048 | rtx-3090 | 31.97 | 36.33 (torch-compile) | 0.88 | 1.00 | measured |

Carried (emission identical, not re-run): 72 rows from cpu-gpu-boundary/table.md, e.g.:

| workload | arch | Mapal ms | best other (leg) | Mapal/best | status |
|---|---|---|---|---|---|
| attn_1024_64 | h100-pcie | 0.0665 | 0.0333 (torch-compile) | 2.00 | carried (emission identical) |
| attn_1024_64 | rtx-3090 | 0.1033 | 0.0584 (fused) | 1.77 | carried (emission identical) |
| attn_1024_64 | rtx-4090 | 0.0707 | 0.0358 (torch-compile) | 1.97 | carried (emission identical) |
| attn_1024_64 | rtx-5090 | 0.0371 | 0.0266 (torch-compile) | 1.39 | carried (emission identical) |
| conv_2048_5 | h100-pcie | 0.0605 | 0.1207 (torch) | 0.50 | carried (emission identical) |
| conv_2048_5 | rtx-3090 | 0.1117 | 0.1649 (torch-compile) | 0.68 | carried (emission identical) |
| conv_2048_5 | rtx-4090 | 0.0559 | 0.0891 (torch-compile) | 0.63 | carried (emission identical) |
| conv_2048_5 | rtx-5090 | 0.0370 | 0.0819 (torch) | 0.45 | carried (emission identical) |
| diamond_2048 | h100-pcie | 1.8841 | 1.4444 (streams) | 1.30 | carried (emission identical) |
| diamond_2048 | rtx-3090 | 2.5340 | 2.5006 (streams) | 1.01 | carried (emission identical) |
| diamond_2048 | rtx-4090 | 0.9555 | 0.98 (cutlass) | 0.98 | carried (emission identical) |
| diamond_2048 | rtx-5090 | 0.8902 | 0.7988 (torch-compile) | 1.11 | carried (emission identical) |
| epi_4096 | h100-pcie | 3.5572 | 3.6524 (lt) | 0.97 | carried (emission identical) |
| epi_4096 | rtx-3090 | 5.5166 | 6.2351 (lt) | 0.88 | carried (emission identical) |
| epi_4096 | rtx-4090 | 2.5192 | 2.5262 (lt) | 1.00 | carried (emission identical) |
| epi_4096 | rtx-5090 | 2.2472 | 2.1187 (torch-compile) | 1.06 | carried (emission identical) |
| ew_4096 | h100-pcie | 0.1155 | 0.1148 (torch-compile) | 1.01 | carried (emission identical) |
| ew_4096 | rtx-3090 | 0.1477 | 0.2406 (fused) | 0.61 | carried (emission identical) |
| ew_4096 | rtx-4090 | 0.1029 | 0.2243 (fused) | 0.46 | carried (emission identical) |
| ew_4096 | rtx-5090 | 0.0588 | 0.1311 (fused) | 0.45 | carried (emission identical) |
| ewconvew_2048_5 | h100-pcie | 0.0760 | 0.1488 (torch-compile) | 0.51 | carried (emission identical) |
| ewconvew_2048_5 | rtx-3090 | 0.1319 | 0.2499 (torch-compile) | 0.53 | carried (emission identical) |
| ewconvew_2048_5 | rtx-4090 | 0.0674 | 0.126 (torch-compile) | 0.53 | carried (emission identical) |
| ewconvew_2048_5 | rtx-5090 | 0.0530 | 0.1044 (torch-compile) | 0.51 | carried (emission identical) |
| fanout_2048 | h100-pcie | 1.2505 | 1.0202 (streams) | 1.23 | carried (emission identical) |
| fanout_2048 | rtx-3090 | 1.6837 | 1.6568 (streams) | 1.02 | carried (emission identical) |
| fanout_2048 | rtx-4090 | 0.6415 | 0.6615 (streams) | 0.97 | carried (emission identical) |
| fanout_2048 | rtx-5090 | 0.5721 | 0.5286 (streams) | 1.08 | carried (emission identical) |
| fir_4194304_16 | h100-pcie | 0.0426 | 0.076 (torch) | 0.56 | carried (emission identical) |
| fir_4194304_16 | rtx-3090 | 0.0757 | 0.1198 (torch-compile) | 0.63 | carried (emission identical) |
| fir_4194304_16 | rtx-4090 | 0.0396 | 0.0645 (torch-compile) | 0.61 | carried (emission identical) |
| fir_4194304_16 | rtx-5090 | 0.0260 | 0.0512 (torch) | 0.51 | carried (emission identical) |
| firew_rowsum_2048x2048_16 | h100-pcie | 0.0476 | 0.0835 (torch-compile) | 0.57 | carried (emission identical) |
| firew_rowsum_2048x2048_16 | rtx-3090 | 0.1031 | 0.1423 (torch-compile) | 0.72 | carried (emission identical) |
| firew_rowsum_2048x2048_16 | rtx-4090 | 0.0454 | 0.0686 (torch-compile) | 0.66 | carried (emission identical) |
| firew_rowsum_2048x2048_16 | rtx-5090 | 0.0314 | 0.0573 (torch-compile) | 0.55 | carried (emission identical) |
| indep_2048 | h100-pcie | 1.2689 | 0.9377 (streams) | 1.35 | carried (emission identical) |
| indep_2048 | rtx-3090 | 1.7005 | 1.5667 (torch-compile) | 1.09 | carried (emission identical) |
| indep_2048 | rtx-4090 | 0.6511 | 0.6605 (cutlass) | 0.99 | carried (emission identical) |
| indep_2048 | rtx-5090 | 0.5763 | 0.5325 (lt) | 1.08 | carried (emission identical) |
| loop_4096 | h100-pcie | 28.45 | 29.4662 (lt) | 0.97 | carried (emission identical) |
| loop_4096 | rtx-3090 | 46.35 | 48.2826 (lt) | 0.96 | carried (emission identical) |
| loop_4096 | rtx-4090 | 19.74 | 20.8425 (cutlass) | 0.95 | carried (emission identical) |
| loop_4096 | rtx-5090 | 16.70 | 17.2974 (lt) | 0.97 | carried (emission identical) |
| loopmm_1024_8 | h100-pcie | 0.6872 | 0.6145 (torch-compile) | 1.12 | carried (emission identical) |
| loopmm_1024_8 | rtx-3090 | 1.4554 | 0.897 (torch-compile) | 1.62 | carried (emission identical) |
| loopmm_1024_8 | rtx-4090 | 0.4904 | 0.3738 (torch-compile) | 1.31 | carried (emission identical) |
| loopmm_1024_8 | rtx-5090 | 0.4537 | 0.3635 (torch-compile) | 1.25 | carried (emission identical) |
| mlp2_2048 | h100-pcie | 2.5071 | 1.7998 (lt2) | 1.39 | carried (emission identical) |
| mlp2_2048 | rtx-3090 | 3.2897 | 3.1795 (lt2) | 1.03 | carried (emission identical) |
| mlp2_2048 | rtx-4090 | 1.2895 | 1.281 (lt2) | 1.01 | carried (emission identical) |
| mlp2_2048 | rtx-5090 | 1.2054 | 1.081 (streams) | 1.12 | carried (emission identical) |
| mlp_2048 | h100-pcie | 1.2477 | 1.0208 (lt) | 1.22 | carried (emission identical) |
| mlp_2048 | rtx-3090 | 1.6991 | 1.6312 (lt) | 1.04 | carried (emission identical) |
| mlp_2048 | rtx-4090 | 0.6351 | 0.6543 (cutlass) | 0.97 | carried (emission identical) |
| mlp_2048 | rtx-5090 | 0.5750 | 0.5264 (lt) | 1.09 | carried (emission identical) |
| pj_m16_n2048_k7168 | h100-pcie | 0.0939 | 0.0695 (torch-compile) | 1.35 | carried (emission identical) |
| pj_m16_n2048_k7168 | rtx-3090 | 0.1278 | 0.0819 (torch-compile) | 1.56 | carried (emission identical) |
| pj_m16_n2048_k7168 | rtx-4090 | 0.0811 | 0.0942 (torch-compile) | 0.86 | carried (emission identical) |
| pj_m16_n2048_k7168 | rtx-5090 | 0.0670 | 0.0553 (torch) | 1.21 | carried (emission identical) |
| smlp_4096_256_1024 | h100-pcie | 0.1850 | 0.1554 (torch-compile) | 1.19 | carried (emission identical) |
| smlp_4096_256_1024 | rtx-3090 | 0.3115 | 0.2406 (torch-compile) | 1.29 | carried (emission identical) |
| smlp_4096_256_1024 | rtx-4090 | 0.1224 | 0.1126 (torch-compile) | 1.09 | carried (emission identical) |
| smlp_4096_256_1024 | rtx-5090 | 0.1034 | 0.0881 (torch-compile) | 1.17 | carried (emission identical) |
| softmax_rows_4096 | h100-pcie | 3.6296 | 3.7487 (torch-compile) | 0.97 | carried (emission identical) |
| softmax_rows_4096 | rtx-3090 | 5.9604 | 6.2157 (torch-compile) | 0.96 | carried (emission identical) |
| softmax_rows_4096 | rtx-4090 | 2.6301 | 2.5446 (torch-compile) | 1.03 | carried (emission identical) |
| softmax_rows_4096 | rtx-5090 | 2.3255 | 2.1258 (torch-compile) | 1.09 | carried (emission identical) |
| twin_2048 | h100-pcie | 1.2414 | 1.0223 (torch-compile) | 1.21 | carried (emission identical) |
| twin_2048 | rtx-3090 | 1.6567 | 1.6404 (torch-compile) | 1.01 | carried (emission identical) |
| twin_2048 | rtx-4090 | 0.6337 | 0.6697 (torch) | 0.95 | carried (emission identical) |
| twin_2048 | rtx-5090 | 0.5550 | 0.5294 (torch-compile) | 1.05 | carried (emission identical) |

### ON/OFF per new rung (OFF iter ms / ON iter ms; >1 = faster ON), `null` = shipped re-run / ON - 1

| program | face | rung | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 |
|---|---|---|---|---|---|---|
| causal_1024_256 | c | range-cell | 1.20 | 1.04 | 1.09 | 1.09 |
| causal_1024_256 | c | scalar-fold | 1.16 | 0.96 | 0.98 | 0.97 |
| causal_1024_256 | default | cell-lanes | 2.04 (null 0.6%) | 2.49 (null 0.0%) | 2.57 (null 0.0%) | 3.07 (null 0.0%) |
| causal_1024_256 | default | range-cell | 1.47 (null 0.6%) | 1.20 (null 0.0%) | 1.23 (null 0.0%) | 1.29 (null 0.0%) |
| causal_1024_256 | default | scalar-fold | 1.37 (null 0.6%) | 1.37 (null 0.0%) | 1.34 (null 0.0%) | 1.44 (null 0.0%) |
| convmc_8_64_56_3 | c | radix-tile | 8.80 | 8.88 | 9.14 | 8.59 |
| convmc_8_64_56_3 | default | radix-tile | 8.92 (null 0.8%) | 11.48 (null 0.1%) | 15.62 (null 9.7%) | 15.47 (null 0.2%) |
| convmc_8_64_56_3_tf32 | c | radix-tile | 8.75 | 8.88 | 9.48 | 8.56 |
| convmc_8_64_56_3_tf32 | default | radix-tile | 8.96 (null 0.6%) | 11.51 (null 0.2%) | 17.22 (null 2.0%) | 15.42 (null 0.7%) |
| decode_1024_256 | c | range-cell | 1.34 | 1.04 | 1.11 | 1.15 |
| decode_1024_256 | default | range-cell | 1.67 (null 2.0%) | 1.14 (null 0.4%) | 1.32 (null 0.3%) | 1.37 (null 0.3%) |
| gen_1024_512_256 | c | range-cell | 2.33 | 2.02 | 2.13 | 2.10 |
| gen_1024_512_256 | c | scalar-fold | 1.14 | 0.96 | 0.99 | 0.97 |
| gen_1024_512_256 | c | trip-batch | 1.91 | 1.95 | 1.94 | 1.92 |
| gen_1024_512_256 | default | cell-lanes | 2.10 (null 2.6%) | 2.56 (null 0.0%) | 2.64 (null 0.0%) | 3.17 (null 0.0%) |
| gen_1024_512_256 | default | range-cell | 2.94 (null 2.6%) | 2.39 (null 0.0%) | 2.54 (null 0.0%) | 2.55 (null 0.0%) |
| gen_1024_512_256 | default | scalar-fold | 1.38 (null 2.6%) | 1.37 (null 0.0%) | 1.34 (null 0.0%) | 1.42 (null 0.0%) |
| gen_1024_512_256 | default | trip-batch | 1.92 (null 2.6%) | 1.99 (null 0.0%) | 1.99 (null 0.0%) | 1.97 (null 0.0%) |
| mlp2_2048_tf32 | default | tf32-tile | 1.24 (null 0.5%) | 1.26 (null 0.1%) | 3.40 (null 0.1%) | 1.14 (null 0.0%) |
| mlp_2048_tf32 | default | tf32-tile | 1.25 (null 0.2%) | 1.25 (null 0.1%) | 3.38 (null 0.3%) | 1.18 (null 0.1%) |
| prefill_1024_512_256 | c | range-cell | 1.22 | 1.04 | 1.10 | 1.09 |
| prefill_1024_512_256 | c | scalar-fold | 1.17 | 0.96 | 1.00 | 0.97 |
| prefill_1024_512_256 | default | cell-lanes | 2.10 (null 0.3%) | 2.52 (null 0.0%) | 2.59 (null 0.0%) | 3.07 (null 0.1%) |
| prefill_1024_512_256 | default | range-cell | 1.50 (null 0.3%) | 1.19 (null 0.0%) | 1.28 (null 0.0%) | 1.25 (null 0.1%) |
| prefill_1024_512_256 | default | scalar-fold | 1.37 (null 0.3%) | 1.37 (null 0.0%) | 1.37 (null 0.0%) | 1.39 (null 0.1%) |
| rowsum_2048 | default | fold-lanes | 0.98 (null 1.3%) | = | = | = |
| rowsum_4096 | default | fold-lanes | 0.99 (null 0.6%) | = | = | = |

### pt_gen vs pt_prefill (iter ms; gen/prefill)

| arch | face | gen | prefill | gen/prefill |
|---|---|---|---|---|
| rtx-4090 | default | 20.45 | 20.63 | 0.99 |
| rtx-4090 | c | 48.42 | 48.98 | 0.99 |
| rtx-5090 | default | 16.84 | 17.22 | 0.98 |
| rtx-5090 | c | 50.48 | 50.83 | 0.99 |
| h100-pcie | default | 27.91 | 28.47 | 0.98 |
| h100-pcie | c | 81.43 | 81.73 | 1.00 |
| rtx-3090 | default | 22.63 | 23.78 | 0.95 |
| rtx-3090 | c | 80.4 | 81.64 | 0.98 |

### convmc and the TF32 programs vs torch (ms; torch med)

| program | arch | Mapal | torch FP32 (eager/compile) | torch TF32-on (eager/compile) | Mapal/best FP32 | Mapal/best TF32-on |
|---|---|---|---|---|---|---|
| convmc_8_64_56_3 | rtx-4090 | 0.1124 | 0.0676/0.1014 | 0.044/0.0871 | 1.66 | 2.55 |
| convmc_8_64_56_3 | rtx-5090 | 0.06149 | 0.0594/0.1004 | 0.0471/0.0655 | 1.04 | 1.31 |
| convmc_8_64_56_3 | h100-pcie | 0.1049 | 0.0812/0.1288 | 0.0399/0.0525 | 1.29 | 2.63 |
| convmc_8_64_56_3 | rtx-3090 | 0.172 | 0.1024/0.1896 | 0.1024/0.1667 | 1.68 | 1.68 |
| convmc_8_64_56_3_tf32 | rtx-4090 | 0.1079 | 0.0737/0.1157 | 0.0522/0.0942 | 1.46 | 2.07 |
| convmc_8_64_56_3_tf32 | rtx-5090 | 0.06113 | 0.0594/0.1004 | 0.045/0.0635 | 1.03 | 1.36 |
| convmc_8_64_56_3_tf32 | h100-pcie | 0.0944 | 0.0811/0.128 | 0.0396/0.0508 | 1.16 | 2.38 |
| convmc_8_64_56_3_tf32 | rtx-3090 | 0.1466 | 0.087/0.17 | 0.087/0.1321 | 1.69 | 1.69 |
| mlp2_2048_tf32 | rtx-4090 | 0.9841 | 1.303/1.269 | 0.8499/0.9042 | 0.78 | 1.16 |
| mlp2_2048_tf32 | rtx-5090 | 0.8981 | 1.102/1.081 | 0.7967/0.7803 | 0.83 | 1.15 |
| mlp2_2048_tf32 | h100-pcie | 0.7457 | 2.11/2.079 | 0.442/0.3996 | 0.36 | 1.87 |
| mlp2_2048_tf32 | rtx-3090 | 2.829 | 3.753/3.938 | 2.78/2.426 | 0.75 | 1.17 |
| mlp_2048_tf32 | rtx-4090 | 0.485 | 0.6482/0.6461 | 0.4588/0.4567 | 0.75 | 1.06 |
| mlp_2048_tf32 | rtx-5090 | 0.4449 | 0.5387/0.5387 | 0.3912/0.3851 | 0.83 | 1.16 |
| mlp_2048_tf32 | h100-pcie | 0.3662 | 1.036/1.034 | 0.209/0.2091 | 0.35 | 1.75 |
| mlp_2048_tf32 | rtx-3090 | 1.405 | 1.944/2.211 | 1.466/1.006 | 0.72 | 1.40 |

### Value gates (probe programs passing before timing / failing)

- rtx-4090: 18 pass, 0 fail
- rtx-5090: 18 pass, 0 fail
- h100-pcie: 18 pass, 0 fail
- rtx-3090: 18 pass, 0 fail

### Evidence check (intent vs realized), whole bundle including OFF arms

- rtx-4090: 3384 ok, 4 mismatch (px_rowsum_128 unroll / face-registers, shipped and off-fold-lanes; a px program, not timed)
- rtx-5090: 3832 ok, 0 mismatch
- h100-pcie: 3999 ok, 0 mismatch
- rtx-3090: 3825 ok, 24 mismatch: convmc and convmc_tf32 radix-tile tile spills (ptxas spill_st=72, spill_ld=40) with min-blocks and face-registers on it; the loop programs' off-streamk+splitk-atomic arm spills 8-16 B; convmc `_c` prefetch LDG cover 0

### Rentals

28 created, 28 destroyed with -y (rentals.log); 13 produced results, 15 ended without (failed-starts/ holds the two launches that died on a bundle error: missing oracle for non-pt arms, missing cutlass dir). No stocklab-* instance touched.

nsys per-kernel for the losing changed programs: nsys-losers.md
