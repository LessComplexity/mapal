### Intended vs realized (shipped programs; ok / MISMATCH / n-a)

| rung:state | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 |
|---|---|---|---|---|
| arch-rungs:on | 36 / 0 / 0 | 36 / 0 / 0 | 33 / 0 / 0 | 36 / 0 / 0 |
| arena:on | 16 / 0 / 0 | 16 / 0 / 0 | 16 / 0 / 0 | 16 / 0 / 0 |
| async:off | 20 / 0 / 0 | 27 / 0 / 0 | 27 / 0 / 0 | 27 / 0 / 0 |
| async:on | 13 / 1 / 0 | - | - | - |
| by-ref:on | 24 / 0 / 0 | 24 / 0 / 0 | 24 / 0 / 0 | 24 / 0 / 0 |
| by-ref:runtime | 105 / 0 / 0 | 105 / 0 / 0 | 105 / 0 / 0 | 105 / 0 / 0 |
| cpu-tiling:on | 2 / 0 / 0 | 2 / 0 / 0 | 2 / 0 / 0 | 2 / 0 / 0 |
| devgen:on | 21 / 0 / 0 | 21 / 0 / 0 | 21 / 0 / 0 | 21 / 0 / 0 |
| devmap:on | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| eager-init:on | 21 / 0 / 0 | 21 / 0 / 0 | 21 / 0 / 0 | 21 / 0 / 0 |
| epilogue:on | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| face-fma:on | 66 / 2 / 0 | 66 / 2 / 0 | 60 / 2 / 0 | 66 / 2 / 0 |
| face-registers:on | 81 / 0 / 0 | 80 / 1 / 0 | 81 / 0 / 0 | 81 / 0 / 0 |
| face-separate:on | 4 / 0 / 0 | 4 / 0 / 0 | 4 / 0 / 0 | 4 / 0 / 0 |
| grid-share:on | 26 / 0 / 0 | 27 / 0 / 0 | 6 / 0 / 0 | 1 / 0 / 0 |
| hostopt:on | 5 / 0 / 0 | 5 / 0 / 0 | 5 / 0 / 0 | 5 / 0 / 0 |
| idx32:on | 26 / 0 / 0 | 23 / 0 / 0 | 31 / 0 / 0 | 24 / 0 / 0 |
| interleave:on | 27 / 0 / 0 | 27 / 0 / 0 | 27 / 0 / 0 | 20 / 0 / 0 |
| min-blocks:off | - | - | 26 / 0 / 0 | - |
| min-blocks:on | 54 / 0 / 0 | 53 / 1 / 0 | 2 / 0 / 0 | 54 / 0 / 0 |
| module:on | 47 / 0 / 0 | 47 / 0 / 0 | 46 / 0 / 0 | 48 / 0 / 0 |
| panel-regs:on | 1 / 0 / 0 | 26 / 1 / 0 | 26 / 0 / 0 | 1 / 0 / 0 |
| pdl:off | - | 36 / 0 / 0 | 33 / 0 / 0 | - |
| prefetch:on | 20 / 0 / 0 | 27 / 0 / 0 | 26 / 1 / 0 | 27 / 0 / 0 |
| resident:on | 21 / 0 / 0 | 21 / 0 / 0 | 21 / 0 / 0 | 21 / 0 / 0 |
| row-fit:on | 7 / 0 / 0 | 27 / 0 / 0 | 1 / 0 / 0 | 7 / 0 / 0 |
| rowfold:on | 4 / 0 / 0 | 4 / 0 / 0 | 4 / 0 / 0 | 4 / 0 / 0 |
| splitk-atomic-zero:on | 1 / 0 / 0 | 8 / 0 / 0 | 5 / 0 / 0 | 5 / 0 / 0 |
| splitk-atomic:on | 6 / 0 / 0 | 4 / 0 / 0 | 2 / 0 / 0 | 16 / 0 / 0 |
| splitk:on | - | - | 4 / 0 / 0 | 2 / 0 / 0 |
| streamk:on | - | 24 / 0 / 0 | 18 / 0 / 0 | - |
| unroll:on | 54 / 0 / 0 | 54 / 0 / 0 | 54 / 0 / 0 | 54 / 0 / 0 |
| warp-fill:on | 27 / 0 / 0 | 8 / 0 / 0 | 25 / 0 / 0 | 21 / 0 / 0 |
| wide-fma:on | 57 / 0 / 0 | - | - | 57 / 0 / 0 |
| window:on | 4 / 2 / 0 | 4 / 2 / 0 | 4 / 2 / 0 | 4 / 2 / 0 |

### Findings: every mismatch on a shipped program

| arch | rung | check | programs | realized (first) |
|---|---|---|---|---|
| rtx-4090 | async:on | LDGSTS in the hot loop, no register-staged STS | pt_attn_1024_64 | sass.hot.ldgsts=0 sass.hot.sts=0 |
| h100-pcie | face-fma:on | PTX hot loop: fma.rn, no separate mul | pt_firew_rowsum_2048x2048_16 | ptx.hot.fma=16 ptx.hot.mul=1 |
| h100-pcie | face-fma:on | SASS hot loop: FFMA, no FMUL | pt_firew_rowsum_2048x2048_16 | sass.hot.ffma=16 sass.hot.fmul=1 |
| rtx-3090 | face-fma:on | PTX hot loop: fma.rn, no separate mul | pt_firew_rowsum_2048x2048_16 | ptx.hot.fma=16 ptx.hot.mul=1 |
| rtx-3090 | face-fma:on | SASS hot loop: FFMA, no FMUL | pt_firew_rowsum_2048x2048_16 | sass.hot.ffma=16 sass.hot.fmul=1 |
| rtx-4090 | face-fma:on | PTX hot loop: fma.rn, no separate mul | pt_firew_rowsum_2048x2048_16 | ptx.hot.fma=16 ptx.hot.mul=1 |
| rtx-4090 | face-fma:on | SASS hot loop: FFMA, no FMUL | pt_firew_rowsum_2048x2048_16 | sass.hot.ffma=16 sass.hot.fmul=1 |
| rtx-5090 | face-fma:on | PTX hot loop: fma.rn, no separate mul | pt_firew_rowsum_2048x2048_16 | ptx.hot.fma=16 ptx.hot.mul=1 |
| rtx-5090 | face-fma:on | SASS hot loop: FFMA, no FMUL | pt_firew_rowsum_2048x2048_16 | sass.hot.ffma=16 sass.hot.fmul=1 |
| rtx-5090 | face-registers:on | no spill | pt_pj_m16_n2048_k7168 | ptxas.spill_st=208 ptxas.spill_ld=208 |
| rtx-5090 | min-blocks:on | ptxas met it without spilling | pt_pj_m16_n2048_k7168 | ptxas.spill_st=208 |
| rtx-5090 | panel-regs:on | the prefetch panel fits the register file: no spill | pt_pj_m16_n2048_k7168 | ptxas.spill_st=208 |
| h100-pcie | prefetch:on | the next panel's LDGs are covered by this panel's math | pt_attn_1024_64 | sass.hot.ldg_cover_min=0 |
| h100-pcie | window:on | LLVM fully unrolled the constant-trip tap walk (loop-unroll FullyUnrolled) | pt_conv_2048_5 pt_ewconvew_2048_5 | remarks.opt.full_unroll=0 |
| rtx-3090 | window:on | LLVM fully unrolled the constant-trip tap walk (loop-unroll FullyUnrolled) | pt_conv_2048_5 pt_ewconvew_2048_5 | remarks.opt.full_unroll=0 |
| rtx-4090 | window:on | LLVM fully unrolled the constant-trip tap walk (loop-unroll FullyUnrolled) | pt_conv_2048_5 pt_ewconvew_2048_5 | remarks.opt.full_unroll=0 |
| rtx-5090 | window:on | LLVM fully unrolled the constant-trip tap walk (loop-unroll FullyUnrolled) | pt_conv_2048_5 pt_ewconvew_2048_5 | remarks.opt.full_unroll=0 |

### How each optimization turned out in time (arms timed this session: ON / OFF, median over processes)

| rung | arch | shipped | cells: ON/OFF (null) | ON faster / slower / inside null |
|---|---|---|---|---|
| by-ref/cpu | h100-pcie | shipped ON | pt_causal_1024_256.cpu 0.925 (-); pt_decode_1024_256.cpu 0.921 (-); pt_loop_4096.cpu 0.934 (-); pt_loopmm_1024_8.cpu 0.955 (-); pt_prefill_1024_512_256.cpu 0.930 (-); pt_rt_mlp2_2048.cpu 0.979 (-) | 0 / 0 / 6 |
| by-ref/cpu | rtx-3090 | shipped ON | pt_causal_1024_256.cpu 0.965 (-); pt_decode_1024_256.cpu 0.960 (-); pt_loop_4096.cpu 0.953 (-); pt_loopmm_1024_8.cpu 0.892 (-); pt_prefill_1024_512_256.cpu 0.964 (-); pt_rt_mlp2_2048.cpu 0.972 (-) | 0 / 0 / 6 |
| by-ref/cpu | rtx-4090 | shipped ON | pt_causal_1024_256.cpu 0.961 (-); pt_decode_1024_256.cpu 0.952 (-); pt_loop_4096.cpu 0.980 (-); pt_loopmm_1024_8.cpu 0.911 (-); pt_prefill_1024_512_256.cpu 0.960 (-); pt_rt_mlp2_2048.cpu 0.993 (-) | 0 / 0 / 6 |
| by-ref/cpu | rtx-5090 | shipped ON | pt_causal_1024_256.cpu 0.959 (-); pt_decode_1024_256.cpu 0.957 (-); pt_loop_4096.cpu 0.960 (-); pt_loopmm_1024_8.cpu 0.920 (-); pt_prefill_1024_512_256.cpu 0.965 (-); pt_rt_mlp2_2048.cpu 0.982 (-) | 0 / 0 / 6 |
| flow | h100-pcie | shipped OFF | pt_attn_1024_64 0.813 (0.501); pt_firew_rowsum_2048x2048_16 0.523 (0.294); pt_rowsum_4096 0.959 (0.013); pt_softmax_rows_4096 0.879 (0.007) | 3 / 0 / 1 |
| flow | rtx-3090 | shipped OFF | pt_attn_1024_64 0.709 (0.472); pt_firew_rowsum_2048x2048_16 0.774 (0.053); pt_rowsum_4096 0.984 (0.010); pt_softmax_rows_4096 0.928 (0.005) | 3 / 0 / 1 |
| flow | rtx-4090 | shipped OFF | pt_attn_1024_64 0.788 (0.300); pt_firew_rowsum_2048x2048_16 0.711 (0.166); pt_rowsum_4096 1.056 (0.033); pt_softmax_rows_4096 0.884 (0.012) | 2 / 1 / 1 |
| flow | rtx-5090 | shipped OFF | pt_attn_1024_64 0.690 (0.176); pt_firew_rowsum_2048x2048_16 0.604 (0.254); pt_rowsum_4096 0.982 (0.014); pt_softmax_rows_4096 0.920 (0.021) | 3 / 0 / 1 |
| retain-resident | h100-pcie | shipped OFF | pt_causal_1024_256 0.678 (0.027); pt_prefill_1024_512_256 0.787 (0.025) | 2 / 0 / 0 |
| splitk | h100-pcie | shipped ON | pt_decode_1024_256 0.466 (0.035); pt_pj_m16_n2048_k7168 1.441 (0.376) | 1 / 0 / 1 |
| splitk | rtx-3090 | shipped ON | pt_attn_1024_64 0.918 (0.472); pt_causal_1024_256 1.005 (0.163); pt_decode_1024_256 0.512 (0.979); pt_pj_m16_n2048_k7168 1.918 (0.329); pt_prefill_1024_512_256 1.004 (0.028) | 0 / 1 / 4 |
| splitk | rtx-4090 | shipped ON | pt_attn_1024_64 1.035 (0.300); pt_decode_1024_256 0.677 (0.134) | 1 / 0 / 1 |
| splitk | rtx-5090 | shipped ON | pt_decode_1024_256 0.667 (0.097) | 1 / 0 / 0 |
| splitk-atomic | h100-pcie | shipped ON | pt_decode_1024_256 0.993 (0.035) | 0 / 0 / 1 |
| splitk-atomic | rtx-3090 | shipped ON | pt_attn_1024_64 0.620 (0.472); pt_causal_1024_256 1.001 (0.163); pt_decode_1024_256 0.567 (0.979); pt_loopmm_1024_8 1.157 (0.920); pt_prefill_1024_512_256 1.002 (0.028); pt_smlp_4096_256_1024 1.081 (0.120) | 0 / 0 / 6 |
| splitk-atomic | rtx-4090 | shipped ON | pt_attn_1024_64 0.575 (0.300); pt_decode_1024_256 0.931 (0.134) | 0 / 0 / 2 |
| splitk-atomic | rtx-5090 | shipped ON | pt_decode_1024_256 0.909 (0.097) | 0 / 0 / 1 |
| streamk | h100-pcie | shipped ON | pt_attn_1024_64 0.985 (0.501); pt_causal_1024_256 1.006 (0.027); pt_decode_1024_256 1.006 (0.035); pt_loopmm_1024_8 0.682 (0.146); pt_prefill_1024_512_256 1.003 (0.025); pt_smlp_4096_256_1024 0.855 (0.257) | 1 / 0 / 5 |
| streamk | rtx-5090 | shipped ON | pt_diamond_2048 0.984 (0.014); pt_fanout_2048 0.981 (0.020); pt_mlp2_2048 0.984 (0.027); pt_mlp_2048 0.975 (0.029); pt_rowsum_4096 1.009 (0.014); pt_rt_mlp2_2048 1.014 (0.200); pt_softmax_rows_4096 1.004 (0.021); pt_twin_2048 0.967 (0.174) | 0 / 0 / 8 |

### GEMM cells and MLP kernels: Mapal vs cuBLAS / cuBLASLt / CUTLASS (median ms, both processes)

| cell | arch | Mapal default | realization | Mapal _c | Mapal _x | cuBLAS | cuBLASLt | CUTLASS | tuned CUTLASS | default / best library | null |
|---|---|---|---|---|---|---|---|---|---|---|---|

### Workload suite: iter ms (median of the reps), CPU-only / GPU-only / CPU+GPU, libraries, PyTorch

| workload | arch | CPU-only | GPU-only | CPU+GPU (shipped) | null | kernel / copy / host ms, launches, MB (traced rep) | best library | torch eager | torch compile | shipped / best other |
|---|---|---|---|---|---|---|---|---|---|---|
| attn_1024_64 | rtx-4090 | 10.09 | = shipped | 0.2873 | 0.106 | 0.13 / 0.06 / 0.07, 6, 0 | fused 0.0369 | 0.0891 | 0.0328 | 8.758 |
| causal_1024_256 | rtx-4090 | 4199.12 | = shipped | 563.30 | 0.023 | 20.34 / 18.79 / 521.61, 2048, 6 | - | 448.4516 | 196.2046 | 2.871 |
| conv_2048_5 | rtx-4090 | 1.6167 | = shipped | 0.0703 | 0.012 | 0.05 / 0.05 / 0.00, 1, 0 | - | 0.0922 | 0.0881 | 0.798 |
| convmc_8_64_56_3 | rtx-4090 | 228.12 | = shipped | 270.83 | 0.051 | - | - | 0.0523 | 0.1014 | 5178.470 |
| decode_1024_256 | rtx-4090 | 3847.20 | = shipped | 56.05 | 0.001 | 41.89 / 15.63 / 0.00, 4096, 4 | cache 28.6490 | 207.1870 | 178.1320 | 1.956 |
| diamond_2048 | rtx-4090 | 66.57 | = shipped | 0.9847 | 0.005 | 0.91 / 0.06 / 0.00, 3, 0 | cutlass 0.9871 | 0.9515 | 0.9155 | 1.076 |
| epi_4096 | rtx-4090 | 165.27 | = shipped | 2.4464 | 0.004 | 2.38 / 0.07 / 0.00, 1, 0 | lt 2.8232 | 2.9102 | 2.7424 | 0.892 |
| ew_4096 | rtx-4090 | 6.3536 | = shipped | 0.5850 | 0.002 | 0.09 / 0.07 / 0.43, 1, 0 | fused 0.2263 | 0.5213 | 0.2261 | 2.588 |
| ewconvew_2048_5 | rtx-4090 | 2.3908 | = shipped | 0.0968 | 0.004 | 0.05 / 0.06 / 0.00, 1, 0 | - | 0.1413 | 0.1229 | 0.788 |
| fanout_2048 | rtx-4090 | 50.45 | = shipped | 0.6896 | 0.007 | 0.61 / 0.09 / 0.00, 2, 0 | streams 0.6308 | 0.6369 | 0.6369 | 1.093 |
| fir_4194304_16 | rtx-4090 | 1.1813 | = shipped | 0.0536 | 0.040 | 0.04 / 0.06 / 0.00, 1, 0 | - | 0.0655 | 0.0634 | 0.845 |
| firew_rowsum_2048x2048_16 | rtx-4090 | 8.4117 | = shipped | 0.0884 | 0.027 | 0.08 / 0.07 / 0.00, 2, 0 | - | 0.0911 | 0.0675 | 1.310 |
| loop_4096 | rtx-4090 | 17329.95 | = shipped | 19.49 | 0.004 | 19.24 / 0.06 / 0.11, 8, 0 | cutlass 22.2679 | 21.8737 | 21.7180 | 0.898 |
| loopmm_1024_8 | rtx-4090 | 227.98 | = shipped | 0.7311 | 0.058 | 0.53 / 0.05 / 0.54, 8, 0 | - | 0.4301 | 0.4434 | 1.700 |
| mlp2_2048 | rtx-4090 | 84.66 | = shipped | 1.3443 | 0.004 | 1.21 / 0.08 / 0.00, 4, 0 | cutlass 1.3466 | - | - | 0.998 |
| mlp_2048 | rtx-4090 | 50.46 | = shipped | 0.6900 | 0.003 | 0.62 / 0.06 / 0.00, 2, 0 | cutlass 0.6154 | - | - | 1.121 |
| pj_m16_n2048_k7168 | rtx-4090 | 12.82 | = shipped | 0.1487 | 0.067 | 0.08 / 0.07 / 0.00, 1, 0 | - | 0.0963 | 0.0932 | 1.595 |
| prefill_1024_512_256 | rtx-4090 | 2209.95 | = shipped | 578.57 | 0.020 | 10.18 / 10.04 / 591.77, 1026, 7 | - | 229.9300 | 103.8674 | 5.570 |
| rowsum_4096 | rtx-4090 | 178.95 | = shipped | 2.5147 | 0.002 | 2.47 / 0.07 / 0.00, 2, 0 | gemv 2.7515 | 2.7740 | 2.4023 | 1.047 |
| rt_mlp2_2048 | rtx-4090 | 7880.09 | = shipped | 47.42 | 0.023 | 10.04 / 21.77 / 15.30, 32, 272 | - | 14.8564 | 11.6204 | 4.081 |
| smlp_4096_256_1024 | rtx-4090 | 8.1147 | = shipped | 0.2658 | 0.167 | 0.12 / 0.07 / 0.01, 2, 0 | eager 0.1321 | 0.1577 | 0.1034 | 2.570 |
| softmax_rows_4096 | rtx-4090 | 186.06 | = shipped | 2.8796 | 0.001 | 2.82 / 0.05 / 0.00, 5, 0 | fused 2.8387 | 4.4217 | 2.7853 | 1.034 |
| twin_2048 | rtx-4090 | 44.03 | = shipped | 0.6794 | 0.008 | 0.60 / 0.07 / 0.00, 2, 0 | - | 0.7107 | 0.6083 | 1.117 |
| attn_1024_64 | rtx-5090 | 10.19 | = shipped | 0.2022 | 0.031 | 0.11 / 0.07 / 0.00, 6, 0 | fused 0.0328 | 0.1344 | 0.0266 | 7.602 |
| causal_1024_256 | rtx-5090 | 4167.00 | = shipped | 542.67 | 0.004 | 23.12 / 21.03 / 498.45, 2048, 6 | - | 465.6916 | 198.2957 | 2.737 |
| conv_2048_5 | rtx-5090 | 2.1273 | = shipped | 0.0597 | 0.017 | 0.04 / 0.09 / 0.00, 1, 0 | - | 0.0840 | 0.0840 | 0.710 |
| convmc_8_64_56_3 | rtx-5090 | 235.54 | = shipped | 256.62 | 0.005 | - | - | 0.0635 | 0.1065 | 4041.338 |
| decode_1024_256 | rtx-5090 | 3829.72 | = shipped | 55.27 | 0.006 | 46.32 / 17.84 / 0.00, 4096, 4 | nocache 41.3870 | 217.3882 | 182.5570 | 1.335 |
| diamond_2048 | rtx-5090 | 77.30 | = shipped | 1.0014 | 0.003 | 0.95 / 0.06 / 0.00, 3, 0 | lt 0.9073 | 0.8827 | 0.8745 | 1.145 |
| epi_4096 | rtx-5090 | 186.66 | = shipped | 2.5851 | 0.004 | 2.56 / 0.07 / 0.00, 1, 0 | lt 2.4310 | 2.4730 | 2.4115 | 1.072 |
| ew_4096 | rtx-5090 | 11.22 | = shipped | 0.6984 | 0.027 | 0.09 / 0.09 / 0.45, 1, 0 | fused 0.1311 | 0.2417 | 0.1331 | 5.327 |
| ewconvew_2048_5 | rtx-5090 | 3.4841 | = shipped | 0.0869 | 0.037 | 0.04 / 0.08 / 0.00, 1, 0 | - | 0.1188 | 0.1065 | 0.816 |
| fanout_2048 | rtx-5090 | 55.04 | = shipped | 0.6859 | 0.007 | 0.65 / 0.11 / 0.00, 2, 0 | streams 0.5839 | 0.5919 | 0.5837 | 1.175 |
| fir_4194304_16 | rtx-5090 | 1.4159 | = shipped | 0.0430 | 0.023 | 0.03 / 0.08 / 0.00, 1, 0 | - | 0.0512 | 0.0512 | 0.840 |
| firew_rowsum_2048x2048_16 | rtx-5090 | 9.4147 | = shipped | 0.0895 | 0.065 | 0.07 / 0.06 / 0.00, 2, 0 | - | 0.0758 | 0.0573 | 1.562 |
| loop_4096 | rtx-5090 | 17483.86 | = shipped | 19.09 | 0.001 | 18.87 / 0.07 / 0.09, 8, 0 | lt 19.9455 | 20.7391 | 20.6428 | 0.957 |
| loopmm_1024_8 | rtx-5090 | 228.42 | = shipped | 0.7443 | 0.012 | 0.52 / 0.06 / 0.07, 8, 0 | - | 0.4731 | 0.3666 | 2.030 |
| mlp2_2048 | rtx-5090 | 96.66 | = shipped | 1.4043 | 0.001 | 1.32 / 0.09 / 0.00, 4, 0 | lt2 1.2322 | - | - | 1.140 |
| mlp_2048 | rtx-5090 | 58.49 | = shipped | 0.6868 | 0.001 | 0.65 / 0.07 / 0.00, 2, 0 | lt 0.5796 | - | - | 1.185 |
| pj_m16_n2048_k7168 | rtx-5090 | 29.66 | = shipped | 0.1255 | 0.044 | 0.07 / 0.09 / 0.00, 1, 0 | - | 0.0553 | 0.0553 | 2.269 |
| prefill_1024_512_256 | rtx-5090 | 2141.05 | = shipped | 529.51 | 0.002 | 11.72 / 11.60 / 508.26, 1026, 7 | - | 241.2187 | 102.8902 | 5.146 |
| rowsum_4096 | rtx-5090 | 198.04 | = shipped | 2.6869 | 0.003 | 2.63 / 0.07 / 0.00, 2, 0 | eager 2.4371 | 2.4648 | 2.4094 | 1.115 |
| rt_mlp2_2048 | rtx-5090 | 8382.99 | = shipped | 58.69 | 0.027 | 10.96 / 26.39 / 19.80, 32, 272 | - | 13.4544 | 10.4755 | 5.602 |
| smlp_4096_256_1024 | rtx-5090 | 9.0642 | = shipped | 0.1731 | 0.020 | 0.12 / 0.07 / 0.00, 2, 0 | lt 0.1024 | 0.1475 | 0.0942 | 1.837 |
| softmax_rows_4096 | rtx-5090 | 244.03 | = shipped | 2.8802 | 0.005 | 2.86 / 0.07 / 0.00, 5, 0 | fused 2.5262 | 3.0976 | 2.4504 | 1.175 |
| twin_2048 | rtx-5090 | 52.62 | = shipped | 0.6703 | 0.002 | 0.62 / 0.09 / 0.00, 2, 0 | - | 0.6390 | 0.5817 | 1.152 |
| attn_1024_64 | h100-pcie | 9.0063 | = shipped | 0.1844 | 0.051 | 0.14 / 0.04 / 0.00, 6, 0 | fused 0.0428 | 0.0972 | 0.0328 | 5.623 |
| causal_1024_256 | h100-pcie | 2846.59 | = shipped | 568.86 | 0.011 | 35.69 / 145.76 / 396.02, 2048, 2050 | - | 236.0562 | 101.1347 | 5.625 |
| conv_2048_5 | h100-pcie | 0.7550 | = shipped | 0.0682 | 0.001 | 0.06 / 0.05 / 0.00, 1, 0 | - | 0.1219 | 0.1211 | 0.563 |
| convmc_8_64_56_3 | h100-pcie | 148.72 | = shipped | 178.87 | 0.011 | - | - | 0.0808 | 0.1269 | 2213.758 |
| decode_1024_256 | h100-pcie | 2722.51 | = shipped | 71.00 | 0.000 | 66.84 / 13.73 / 0.00, 5120, 4 | cache 19.8480 | 93.8089 | 91.8106 | 3.577 |
| diamond_2048 | h100-pcie | 38.30 | = shipped | 1.9237 | 0.000 | 1.89 / 0.05 / 0.00, 3, 0 | streams 1.4606 | 1.5636 | 1.5638 | 1.317 |
| epi_4096 | h100-pcie | 90.44 | = shipped | 3.5934 | 0.001 | 3.56 / 0.05 / 0.00, 1, 0 | lt 3.7795 | 4.0551 | 3.9101 | 0.951 |
| ew_4096 | h100-pcie | 2.0163 | = shipped | 0.5448 | 0.026 | 0.10 / 0.05 / 0.32, 1, 0 | fused 0.1248 | 0.2688 | 0.1147 | 4.750 |
| ewconvew_2048_5 | h100-pcie | 1.1845 | = shipped | 0.0872 | 0.005 | 0.06 / 0.05 / 0.00, 1, 0 | - | 0.1599 | 0.1453 | 0.600 |
| fanout_2048 | h100-pcie | 28.27 | = shipped | 1.3015 | 0.001 | 1.27 / 0.07 / 0.00, 2, 0 | streams 1.0336 | 1.0506 | 1.0491 | 1.259 |
| fir_4194304_16 | h100-pcie | 0.6957 | = shipped | 0.0479 | 0.007 | 0.04 / 0.05 / 0.00, 1, 0 | - | 0.0764 | 0.0764 | 0.627 |
| firew_rowsum_2048x2048_16 | h100-pcie | 4.5125 | = shipped | 0.1121 | 0.004 | 0.10 / 0.05 / 0.00, 2, 0 | - | 0.1050 | 0.0838 | 1.338 |
| loop_4096 | h100-pcie | 11381.99 | = shipped | 28.83 | 0.007 | 28.75 / 0.06 / 0.00, 8, 0 | lt 30.5280 | 30.9782 | 31.0244 | 0.945 |
| loopmm_1024_8 | h100-pcie | 164.24 | = shipped | 0.8032 | 0.003 | 0.69 / 0.03 / 0.00, 8, 0 | - | 0.7029 | 0.6380 | 1.259 |
| mlp2_2048 | h100-pcie | 48.58 | = shipped | 2.5831 | 0.002 | 2.54 / 0.07 / 0.00, 4, 0 | lt2 1.8686 | - | - | 1.382 |
| mlp_2048 | h100-pcie | 28.38 | = shipped | 1.2913 | 0.001 | 1.27 / 0.05 / 0.00, 2, 0 | lt 1.0295 | - | - | 1.254 |
| pj_m16_n2048_k7168 | h100-pcie | 6.3301 | = shipped | 0.2378 | 0.023 | 0.10 / 0.05 / 0.02, 2, 0 | - | 0.0697 | 0.0693 | 3.432 |
| prefill_1024_512_256 | h100-pcie | 1478.53 | = shipped | 450.56 | 0.000 | 17.88 / 71.08 / 368.95, 1026, 1028 | - | 121.0940 | 52.2544 | 8.622 |
| rowsum_4096 | h100-pcie | 104.58 | = shipped | 3.7894 | 0.001 | 3.77 / 0.05 / 0.00, 2, 0 | gemv 3.8489 | 3.8684 | 3.8479 | 0.985 |
| rt_mlp2_2048 | h100-pcie | 5545.68 | = shipped | 40.83 | 0.001 | 20.58 / 12.55 / 7.54, 32, 272 | - | 21.6604 | 18.0017 | 2.268 |
| smlp_4096_256_1024 | h100-pcie | 4.5216 | = shipped | 0.2329 | 0.013 | 0.19 / 0.05 / 0.00, 2, 0 | lt 0.1565 | 0.1978 | 0.1609 | 1.488 |
| softmax_rows_4096 | h100-pcie | 119.04 | = shipped | 4.2019 | 0.000 | 4.17 / 0.05 / 0.00, 5, 0 | fused 4.0280 | 4.8947 | 3.9075 | 1.075 |
| twin_2048 | h100-pcie | 24.32 | = shipped | 1.2872 | 0.002 | 1.26 / 0.06 / 0.00, 2, 0 | - | 1.0620 | 1.0297 | 1.250 |
| attn_1024_64 | rtx-3090 | 18.13 | = shipped | 0.2423 | 0.089 | 0.20 / 0.08 / 0.00, 6, 0 | fused 0.0584 | 0.1649 | 0.0584 | 4.148 |
| causal_1024_256 | rtx-3090 | 4091.42 | = shipped | 568.86 | 0.025 | 25.00 / 19.77 / 541.70, 2048, 6 | - | 441.9482 | 196.8773 | 2.889 |
| conv_2048_5 | rtx-3090 | 3.4690 | = shipped | 0.1323 | 0.015 | 0.11 / 0.07 / 0.00, 1, 0 | - | 0.1843 | 0.1638 | 0.808 |
| convmc_8_64_56_3 | rtx-3090 | 398.37 | = shipped | 396.79 | 0.016 | - | - | 0.0790 | 0.1700 | 5022.692 |
| decode_1024_256 | rtx-3090 | 3847.49 | = shipped | 53.39 | 0.007 | 49.89 / 21.23 / 0.00, 4096, 4 | cache 28.3180 | 210.2313 | 197.3555 | 1.885 |
| diamond_2048 | rtx-3090 | 122.30 | = shipped | 2.5936 | 0.000 | 2.53 / 0.07 / 0.00, 3, 0 | streams 2.7986 | 2.8959 | 2.9266 | 0.927 |
| epi_4096 | rtx-3090 | 295.71 | = shipped | 5.5562 | 0.001 | 5.48 / 0.07 / 0.00, 1, 0 | lt 7.0093 | 7.0871 | 7.0031 | 0.793 |
| ew_4096 | rtx-3090 | 23.07 | = shipped | 0.5454 | 0.079 | 0.15 / 0.06 / 0.34, 1, 0 | fused 0.2417 | 0.5642 | 0.2406 | 2.267 |
| ewconvew_2048_5 | rtx-3090 | 6.2069 | = shipped | 0.1524 | 0.023 | 0.11 / 0.07 / 0.00, 1, 0 | - | 0.3000 | 0.2478 | 0.615 |
| fanout_2048 | rtx-3090 | 89.48 | = shipped | 1.7536 | 0.001 | 1.67 / 0.08 / 0.00, 2, 0 | streams 1.8975 | 1.9804 | 1.9845 | 0.924 |
| fir_4194304_16 | rtx-3090 | 3.1672 | = shipped | 0.0892 | 0.010 | 0.07 / 0.08 / 0.00, 1, 0 | - | 0.1208 | 0.1208 | 0.738 |
| firew_rowsum_2048x2048_16 | rtx-3090 | 12.88 | = shipped | 0.1527 | 0.012 | 0.14 / 0.07 / 0.00, 2, 0 | - | 0.2294 | 0.1423 | 1.073 |
| loop_4096 | rtx-3090 | 17091.64 | = shipped | 49.54 | 0.007 | 50.38 / 0.07 / 0.68, 8, 0 | lt 52.9633 | 54.4819 | 54.7758 | 0.935 |
| loopmm_1024_8 | rtx-3090 | 228.36 | = shipped | 1.6598 | 0.003 | 1.46 / 0.06 / 0.05, 8, 0 | - | 1.1530 | 0.9676 | 1.715 |
| mlp2_2048 | rtx-3090 | 152.73 | = shipped | 3.4026 | 0.001 | 3.28 / 0.07 / 0.00, 4, 0 | lt2 3.5881 | - | - | 0.948 |
| mlp_2048 | rtx-3090 | 90.65 | = shipped | 1.7666 | 0.001 | 1.68 / 0.06 / 0.00, 2, 0 | lt 1.8340 | - | - | 0.963 |
| pj_m16_n2048_k7168 | rtx-3090 | 39.40 | = shipped | 0.3488 | 0.000 | 0.14 / 0.06 / 0.00, 2, 0 | - | 0.0901 | 0.0819 | 4.258 |
| prefill_1024_512_256 | rtx-3090 | 2121.96 | = shipped | 552.96 | 0.004 | 12.63 / 10.69 / 528.07, 1026, 7 | - | 227.5830 | 100.4145 | 5.507 |
| rowsum_4096 | rtx-3090 | 307.57 | = shipped | 5.9534 | 0.002 | 5.91 / 0.07 / 0.00, 2, 0 | eager 6.9233 | 6.9386 | 7.1107 | 0.860 |
| rt_mlp2_2048 | rtx-3090 | 8006.77 | = shipped | 88.68 | 0.000 | 26.67 / 45.22 / 15.59, 32, 272 | - | 40.6773 | 31.9468 | 2.776 |
| smlp_4096_256_1024 | rtx-3090 | 13.01 | = shipped | 0.4092 | 0.009 | 0.30 / 0.05 / 0.00, 2, 0 | cutlass 0.2662 | 0.3963 | 0.2417 | 1.693 |
| softmax_rows_4096 | rtx-3090 | 326.37 | = shipped | 6.5033 | 0.001 | 6.45 / 0.07 / 0.00, 5, 0 | fused 7.1803 | 9.0819 | 7.0308 | 0.925 |
| twin_2048 | rtx-3090 | 81.11 | = shipped | 1.7299 | 0.013 | 1.66 / 0.10 / 0.00, 2, 0 | - | 1.8667 | 1.6722 | 1.034 |

