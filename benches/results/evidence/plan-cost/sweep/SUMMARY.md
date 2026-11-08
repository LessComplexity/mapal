# plan-cost measure sweep (HEAD 1973bea vs base d87ab13)

ON = HEAD, OFF = HEAD `--gpu-off=plan-cost,linear`, BASE = d87ab13 emitter (same runtime). Times in ms (program `iter ms`, median of 9 ABBA reps, one process per rep; cells: median of 2 passes x 30). `=ON` = the arm emits the same as ON (bundle drops equal arms). null = |null copy / ON - 1|; tol = max(null, half ON min-max spread, 0.02); verdict ON vs BASE. Boxes: 4090 (driver 570.133.07, Ryzen 3 3300X host, offer 10964882), 5090 (offer 56812843).
Emission diff HEAD vs base (changed.py): h100-pcie 0, rtx-3090 0 on every face; rtx-4090 60 and rtx-5090 112 default-face programs, contract/exact 0.

## Summary on changed programs with a BASE arm

| arch | changed_programs_with_base_arm | geomean_ON_over_BASE | win | loss | null | geomean_OFF_over_BASE | chain_512_BASE>ON_over_best_other |
|---|---|---|---|---|---|---|---|
| rtx-4090 | 45 | 1.026 | 24 | 8 | 13 | 1.000 | diamond:1.249>1.033 fanout:1.169>1.067 indep:1.242>1.072 mlp2:1.365>1.239 mlp:1.165>1.070 twin:1.258>0.970 |
| rtx-5090 | 77 | 1.033 | 23 | 34 | 20 | 0.997 | diamond:1.524>1.208 fanout:1.384>1.232 indep:1.754>1.294 mlp2:1.859>1.766 mlp:1.333>1.150 twin:1.711>1.193 |

## Value gates

rtx-4090

| section | pass | fail |
|---|---|---|
| differential (every program, every arm) | 45 | 0 |
| probe arms (programs 44) | 160 | 0 |

rtx-5090

| section | pass | fail |
|---|---|---|
| differential (every program, every arm) | 95 | 0 |
| probe arms (programs 69) | 272 | 0 |

## Size ladders (chains 512 / 1024 / 2048)

rtx-4090

| program | changed | on | off | base | null | best_other | best_ms | on_over_best | base_over_best | on_over_base | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pt_diamond_1024 | n | 0.17688 | =ON | =ON | 0.000 | streams | 0.1352 | 1.308 | 1.308 |  | = |
| pt_diamond_2048 | n | 1.1394 | =ON | =ON | 0.001 | torch-compile | 0.9339 | 1.220 | 1.220 |  | = |
| pt_diamond_512 | y | 0.048671 | 0.060671 | 0.058811 | 0.026 | lt | 0.0471 | 1.033 | 1.249 | 0.828 | win |
| pt_fanout_1024 | n | 0.12347 | =ON | =ON | 0.001 | streams | 0.0923 | 1.338 | 1.338 |  | = |
| pt_fanout_2048 | n | 0.76171 | =ON | =ON | 0.005 | streams | 0.6236 | 1.221 | 1.221 |  | = |
| pt_fanout_512 | y | 0.03712 | 0.040571 | 0.040671 | 0.005 | eager | 0.0348 | 1.067 | 1.169 | 0.913 | win |
| pt_indep_1024 | n | 0.12232 | =ON | =ON | 0.001 | streams | 0.0899 | 1.361 | 1.361 |  | = |
| pt_indep_2048 | n | 0.75762 | =ON | =ON | 0.002 | lt | 0.6287 | 1.205 | 1.205 |  | = |
| pt_indep_512 | y | 0.037291 | 0.04317 | 0.043211 | 0.001 | streams | 0.0348 | 1.072 | 1.242 | 0.863 | win |
| pt_mlp2_1024 | n | 0.24037 | =ON | =ON | 0.002 | lt2 | 0.1669 | 1.440 | 1.440 |  | = |
| pt_mlp2_2048 | n | 1.5401 | =ON | =ON | 0.001 | lt2 | 1.2524 | 1.230 | 1.230 |  | = |
| pt_mlp2_512 | y | 0.070971 | 0.07773 | 0.078191 | 0.002 | streams | 0.0573 | 1.239 | 1.365 | 0.908 | win |
| pt_mlp_1024 | n | 0.12169 | =ON | =ON | 0.000 | lt | 0.0891 | 1.366 | 1.366 |  | = |
| pt_mlp_2048 | n | 0.74636 | =ON | =ON | 0.001 | lt | 0.6185 | 1.207 | 1.207 |  | = |
| pt_mlp_512 | y | 0.03723 | 0.04072 | 0.040541 | 0.005 | lt | 0.0348 | 1.070 | 1.165 | 0.918 | win |
| pt_twin_1024 | n | 0.12172 | =ON | =ON | 0.000 | torch-eager | 0.0911 | 1.336 | 1.336 |  | = |
| pt_twin_2048 | n | 0.75388 | =ON | =ON | 0.002 | torch-compile | 0.6216 | 1.213 | 1.213 |  | = |
| pt_twin_512 | y | 0.03675 | 0.0454 | 0.04766 | 0.008 | torch-eager | 0.0379 | 0.970 | 1.258 | 0.771 | win |

rtx-5090

| program | changed | on | off | base | null | best_other | best_ms | on_over_best | base_over_best | on_over_base | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pt_diamond_1024 | n | 0.18212 | =ON | =ON | 0.005 | streams | 0.129 | 1.412 | 1.412 |  | = |
| pt_diamond_2048 | y | 0.92735 | 0.84166 | 0.84227 | 0.001 | torch-compile | 0.7762 | 1.195 | 1.085 | 1.101 | loss |
| pt_diamond_512 | y | 0.051943 | 0.066043 | 0.065533 | 0.002 | streams | 0.043 | 1.208 | 1.524 | 0.793 | win |
| pt_fanout_1024 | n | 0.12186 | =ON | =ON | 0.001 | torch-compile | 0.0922 | 1.322 | 1.322 |  | = |
| pt_fanout_2048 | y | 0.652 | 0.59813 | 0.59988 | 0.002 | streams | 0.5164 | 1.263 | 1.162 | 1.087 | loss |
| pt_fanout_512 | y | 0.038191 | 0.043382 | 0.042902 | 0.022 | streams | 0.031 | 1.232 | 1.384 | 0.890 | win |
| pt_indep_1024 | n | 0.12331 | =ON | =ON | 0.006 | streams | 0.0819 | 1.506 | 1.506 |  | = |
| pt_indep_2048 | y | 0.61276 | 0.57307 | 0.57354 | 0.001 | lt | 0.5202 | 1.178 | 1.103 | 1.068 | loss |
| pt_indep_512 | y | 0.037139 | 0.049149 | 0.050349 | 0.003 | streams | 0.0287 | 1.294 | 1.754 | 0.738 | win |
| pt_mlp2_1024 | n | 0.2529 | =ON | =ON | 0.013 | lt2 | 0.1528 | 1.655 | 1.655 |  | = |
| pt_mlp2_2048 | y | 1.2523 | 1.1453 | 1.1425 | 0.001 | streams | 1.0419 | 1.202 | 1.097 | 1.096 | loss |
| pt_mlp2_512 | y | 0.078429 | 0.083928 | 0.082519 | 0.023 | lt2 | 0.0444 | 1.766 | 1.859 | 0.950 | null |
| pt_mlp_1024 | n | 0.12208 | =ON | =ON | 0.002 | torch-compile | 0.0901 | 1.355 | 1.355 |  | = |
| pt_mlp_2048 | y | 0.61698 | 0.56338 | 0.56186 | 0.001 | lt | 0.5161 | 1.195 | 1.089 | 1.098 | loss |
| pt_mlp_512 | y | 0.037719 | 0.041689 | 0.04372 | 0.019 | lt | 0.0328 | 1.150 | 1.333 | 0.863 | win |
| pt_twin_1024 | n | 0.1205 | =ON | =ON | 0.009 | torch-eager | 0.0881 | 1.368 | 1.368 |  | = |
| pt_twin_2048 | y | 0.64667 | 0.55928 | 0.55594 | 0.002 | torch-compile | 0.5181 | 1.248 | 1.073 | 1.163 | loss |
| pt_twin_512 | y | 0.036621 | 0.052101 | 0.052541 | 0.012 | torch-eager | 0.0307 | 1.193 | 1.711 | 0.697 | win |

## Device ops between the timers, 512 chains (nsys, median of 3 processes; us)

rtx-4090

| program.arm | ops | zero_passes | sum_kernel_us | idle_gaps_us | span_us |
|---|---|---|---|---|---|
| pt_diamond_512 | 3.0 | 0.0 | 42.1 | 1.3 | 43.5 |
| pt_diamond_512.off-plan-cost+linear | 5.0 | 2.0 | 46.8 | 7.9 | 57.1 |
| pt_diamond_512.on-base | 5.0 | 2.0 | 46.7 | 8.1 | 57.2 |
| pt_fanout_512 | 2.0 | 0.0 | 28.1 | 0.6 | 28.7 |
| pt_fanout_512.off-plan-cost+linear | 3.0 | 1.0 | 30.1 | 3.9 | 35.3 |
| pt_fanout_512.on-base | 3.0 | 1.0 | 30.2 | 6.1 | 37.7 |
| pt_indep_512 | 2.0 | 0.0 | 28.6 | 0.6 | 29.2 |
| pt_indep_512.off-plan-cost+linear | 3.0 | 1.0 | 30.6 | 6.6 | 38.7 |
| pt_indep_512.on-base | 3.0 | 1.0 | 30.7 | 6.5 | 38.4 |
| pt_mlp2_512 | 4.0 | 0.0 | 55.6 | 2.0 | 57.4 |
| pt_mlp2_512.off-plan-cost+linear | 6.0 | 2.0 | 60.1 | 9.3 | 72.1 |
| pt_mlp2_512.on-base | 6.0 | 2.0 | 60.2 | 8.1 | 71.3 |
| pt_mlp_512 | 2.0 | 0.0 | 28.0 | 0.6 | 28.6 |
| pt_mlp_512.off-plan-cost+linear | 3.0 | 1.0 | 30.7 | 1.3 | 33.2 |
| pt_mlp_512.on-base | 3.0 | 1.0 | 30.7 | 4.2 | 36.1 |
| pt_twin_512 | 2.0 | 0.0 | 27.8 | 0.7 | 28.5 |
| pt_twin_512.off-plan-cost+linear | 4.0 | 2.0 | 32.2 | 7.4 | 41.9 |
| pt_twin_512.on-base | 4.0 | 2.0 | 32.2 | 7.3 | 41.9 |

rtx-5090

| program.arm | ops | zero_passes | sum_kernel_us | idle_gaps_us | span_us |
|---|---|---|---|---|---|
| pt_diamond_512 | 3.0 | 0.0 | 36.8 | 3.6 | 40.4 |
| pt_diamond_512.off-plan-cost+linear | 5.0 | 2.0 | 36.4 | 20.8 | 58.7 |
| pt_diamond_512.on-base | 5.0 | 2.0 | 36.2 | 21.3 | 59.6 |
| pt_fanout_512 | 2.0 | 0.0 | 24.9 | 2.1 | 27.1 |
| pt_fanout_512.off-plan-cost+linear | 3.0 | 1.0 | 25.0 | 21.1 | 47.2 |
| pt_fanout_512.on-base | 3.0 | 1.0 | 24.8 | 21.7 | 47.1 |
| pt_indep_512 | 2.0 | 0.0 | 24.7 | 1.6 | 26.3 |
| pt_indep_512.off-plan-cost+linear | 3.0 | 1.0 | 24.5 | 14.1 | 39.5 |
| pt_indep_512.on-base | 3.0 | 1.0 | 24.3 | 13.5 | 38.5 |
| pt_mlp2_512 | 4.0 | 0.0 | 50.1 | 3.8 | 53.9 |
| pt_mlp2_512.off-plan-cost+linear | 6.0 | 2.0 | 49.7 | 34.9 | 85.9 |
| pt_mlp2_512.on-base | 6.0 | 2.0 | 49.6 | 30.3 | 81.7 |
| pt_mlp_512 | 2.0 | 0.0 | 24.5 | 1.0 | 25.4 |
| pt_mlp_512.off-plan-cost+linear | 3.0 | 1.0 | 24.6 | 22.1 | 47.6 |
| pt_mlp_512.on-base | 3.0 | 1.0 | 24.4 | 21.1 | 46.3 |
| pt_twin_512 | 2.0 | 0.0 | 24.7 | 2.0 | 26.8 |
| pt_twin_512.off-plan-cost+linear | 4.0 | 2.0 | 24.0 | 20.8 | 46.8 |
| pt_twin_512.on-base | 4.0 | 2.0 | 24.2 | 20.4 | 46.3 |

## ON slower than BASE beyond tol

rtx-4090

| program | on | off | base | null | tol | on_over_base |
|---|---|---|---|---|---|---|
| pt_decode_1024_256 | 46.536 | 23.676 | 24.048 | 0.000 | 0.029 | 1.935 |
| pt_pj_m16_n2048_k7168 | 0.070681 | 0.05669 | 0.056721 | 0.006 | 0.038 | 1.246 |
| pt_pj_m1_n1024_k4096 | 0.03806 | 0.0172 | 0.01703 | 0.002 | 0.052 | 2.235 |
| pt_pj_m1_n2048_k7168 | 0.098161 | 0.02812 | 0.02739 | 0.001 | 0.020 | 3.584 |
| mm_128x256x512 | 0.01225 | 0.0113 | 0.0113 | 0.004 | 0.020 | 1.084 |
| mm_16x2048x7168 | 0.09935 | 0.09615 | 0.0963 | 0.001 | 0.020 | 1.032 |
| mm_1x1024x4096 | 0.084 | 0.0307 | 0.0307 | 0.000 | 0.020 | 2.736 |
| mm_1x2048x7168 | 0.2202 | 0.0922 | 0.0922 | 0.000 | 0.020 | 2.388 |

rtx-5090

| program | on | off | base | null | tol | on_over_base |
|---|---|---|---|---|---|---|
| pt_bias_4096 | 2.1322 | 2.0875 | 2.0902 | 0.002 | 0.020 | 1.020 |
| pt_decode_1024_256 | 48.48 | 41.986 | 36.698 | 0.000 | 0.020 | 1.321 |
| pt_diamond_2048 | 0.92735 | 0.84166 | 0.84227 | 0.001 | 0.020 | 1.101 |
| pt_epi_4096 | 2.1267 | 2.0545 | 2.0539 | 0.002 | 0.020 | 1.035 |
| pt_fanout_2048 | 0.652 | 0.59813 | 0.59988 | 0.002 | 0.020 | 1.087 |
| pt_indep_2048 | 0.61276 | 0.57307 | 0.57354 | 0.001 | 0.020 | 1.068 |
| pt_loop_4096 | 16.345 | 15.986 | 15.985 | 0.001 | 0.020 | 1.023 |
| pt_mlp2_2048 | 1.2523 | 1.1453 | 1.1425 | 0.001 | 0.020 | 1.096 |
| pt_mlp_2048 | 0.61698 | 0.56338 | 0.56186 | 0.001 | 0.020 | 1.098 |
| pt_pj_m1_n1024_k4096 | 0.03076 | 0.02453 | 0.02508 | 0.024 | 0.029 | 1.226 |
| pt_pj_m1_n18432_k7168 | 0.39231 | 0.32419 | 0.32448 | 0.003 | 0.020 | 1.209 |
| pt_pj_m1_n2048_k7168 | 0.070931 | 0.03549 | 0.03522 | 0.002 | 0.020 | 2.014 |
| pt_pm_2048_128 | 0.63384 | 0.60483 | 0.60236 | 0.001 | 0.020 | 1.052 |
| pt_pm_2048_32 | 0.62928 | 0.57421 | 0.5751 | 0.001 | 0.020 | 1.094 |
| pt_pm_2048_64 | 0.63784 | 0.58322 | 0.58003 | 0.006 | 0.020 | 1.100 |
| pt_pro_2048 | 0.34367 | 0.29957 | 0.29793 | 0.002 | 0.020 | 1.154 |
| pt_residual_4096 | 2.1416 | 2.0839 | 2.085 | 0.002 | 0.020 | 1.027 |
| pt_rg_indep_2048 | 5.1226 | 4.7117 | 4.7077 | 0.001 | 0.020 | 1.088 |
| pt_rg_mlp2_2048 | 9.8174 | 9.43 | 9.4163 | 0.000 | 0.020 | 1.043 |
| pt_rowsum_2048 | 0.3397 | 0.2923 | 0.292 | 0.000 | 0.020 | 1.163 |
| pt_rq_mlp2_2048 | 1.2938 | 1.1827 | 1.1858 | 0.002 | 0.020 | 1.091 |
| pt_rq_twin_2048 | 0.68398 | 0.59529 | 0.59666 | 0.001 | 0.020 | 1.146 |
| pt_rt_indep_2048 | 5.0533 | 4.6405 | 4.6438 | 0.000 | 0.020 | 1.088 |
| pt_rt_mlp2_2048 | 9.7312 | 9.3316 | 9.3371 | 0.000 | 0.020 | 1.042 |
| pt_softmax_rows_2048 | 0.34675 | 0.30008 | 0.30346 | 0.001 | 0.020 | 1.143 |
| pt_twin_2048 | 0.64667 | 0.55928 | 0.55594 | 0.002 | 0.020 | 1.163 |
| mm_128x1024x1024 | 0.043 | 0.0307 | 0.0307 | 0.000 | 0.020 | 1.401 |
| mm_128x1024x2048 | 0.0737 | 0.045 | 0.045 | 0.000 | 0.020 | 1.638 |
| mm_128x256x256 | 0.0102 | 0.0082 | 0.0082 | 0.000 | 0.020 | 1.244 |
| mm_16x1024x4096 | 0.0369 | 0.0327 | 0.0327 | 0.000 | 0.020 | 1.128 |
| mm_16x2048x7168 | 0.0809 | 0.0635 | 0.0635 | 0.000 | 0.020 | 1.274 |
| mm_1x1024x4096 | 0.0553 | 0.0164 | 0.0164 | 0.000 | 0.020 | 3.372 |
| mm_1x18432x7168 | 0.384 | 0.3338 | 0.3338 | 0.003 | 0.020 | 1.150 |
| mm_1x2048x7168 | 0.1495 | 0.0451 | 0.0451 | 0.007 | 0.020 | 3.315 |

Per program x arch tables (ON / OFF / BASE, null, best other leg, ratios): `programs-<arch>.tsv`; intent check rows (plan-cost, linear, mismatches): `evidence-<arch>.tsv`. Raw: release evidence-raw-2026-11-09 (RAW-manifest.tsv).
