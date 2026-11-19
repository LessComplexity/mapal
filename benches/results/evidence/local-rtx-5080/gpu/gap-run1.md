| workload | Mapal ms | best other (leg) | Mapal/best | null |
|---|---:|---|---:|---:|
| causal_1024_256 | 24.14 | 56.35 (torch-compile) | 0.43 | 0.3% |
| conv_2048_5 | 0.06946 | 0.1516 (torch-eager) | 0.46 | 0.1% |
| decode_1024_256 | 19.1 | 16.05 (cache) | 1.19 | 1.5% |
| diamond_2048 | 1.515 | 1.37 (torch-compile) | 1.11 | 0.6% |
| ew_4096 | 0.0925 | 0.2437 (fused) | 0.38 | 1.4% |
| ewconvew_2048_5 | 0.07933 | 0.1864 (torch-compile) | 0.43 | 2.8% |
| fanout_2048 | 1.077 | 0.9216 (torch-compile) | 1.17 | 1.2% |
| fir_4194304_16 | 0.0469 | 0.1044 (torch-eager) | 0.45 | 2.6% |
| firew_rowsum_2048x2048_16 | 0.05531 | 0.1085 (torch-compile) | 0.51 | 1.0% |
| gen_1024_512_256 | 11.07 | 24.14 (torch-compile) | 0.46 | 2.4% |
| indep_2048 | 1.103 | 0.9155 (torch-compile) | 1.20 | 2.3% |
| indep_512 | 0.1043 | 0.0366 (streams) | 2.85 | 3.4% |
| loop_4096 | 28.95 | 29.23 (torch-eager) | 0.99 | 2.6% |
| loopmm_1024_8 | 0.7245 | 0.5857 (torch-compile) | 1.24 | 3.4% |
| mlp2_2048 | 2.002 | 1.773 (lt2) | 1.13 | 1.2% |
| mlp2_2048_tf32 | 1.671 | 1.558 (torch-eager-tf32) | 1.07 | 0.1% |
| mlp_2048 | 1.024 | 0.9134 (lt) | 1.12 | 0.1% |
| mlp_2048_tf32 | 0.8304 | 0.7608 (torch-eager-tf32) | 1.09 | 0.3% |
| mlp_512 | 0.1053 | 0.043 (lt) | 2.45 | 5.8% |
| pj_m16_n18432_k7168 | 0.6666 | 0.6164 (torch-compile) | 1.08 | 0.2% |
| pj_m16_n2048_k7168 | 0.1255 | 0.0819 (torch-eager) | 1.53 | 1.1% |
| prefill_1024_512_256 | 11.15 | 28.25 (torch-compile) | 0.39 | 0.3% |
| rowsum_2048 | 0.5434 | 0.4628 (eager) | 1.17 | 1.3% |
| rowsum_4096 | 3.83 | 3.609 (gemv) | 1.06 | 0.1% |
| rt_mlp2_1024 | 2.666 | 2.528 (torch-compile) | 1.05 | 0.3% |
| rt_mlp2_2048 | 16.36 | 16.01 (torch-compile) | 1.02 | 0.1% |
| softmax_rows_4096 | 4.04 | 3.759 (fused) | 1.07 | 3.7% |
| twin_2048 | 1.028 | 0.9175 (torch-compile) | 1.12 | 0.5% |

| cell (default face) | Mapal ms | cuBLAS | CUTLASS | CUTLASS tuned | Mapal/best | null |
|---|---:|---:|---:|---:|---:|---:|
| 16x2048x7168 | 0.1126 | 0.0799 | 0.5263 | 0.385 | 1.41 | 0.0% |
| 4096x4096x4096 | 4.806 | 3.951 | 4.115 | 10.21 | 1.22 | 0.4% |
