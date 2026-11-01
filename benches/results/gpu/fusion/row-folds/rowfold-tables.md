# S54 rowfold — five parts

Build: `s54/rowfold` (raster-swap build, 595eef2 and later). Sessions: `rowfold-<part>.txt` and `-confirm.txt`
(two boxes per part; `-v1` / `-v2` = the two earlier builds). Differential 26/26 on every part and session.
Mapal: the program's own clock around the computations (`iter ms`), medians of 7, session 1 / session 2; kernels:
MAPAL_GPU_TRACE launch times (site + fold, row-major store). Library: chain_bench.cu, cuBLAS sgemm + warp-per-row
sum (`eager`) or sgemv (`gemv`), or one fused warp-per-row softmax kernel (`fused`) — library order != oracle order.

## rowfold: device fold vs host fold, and vs the library

| part | probe | n | face | Mapal iter ms, rowfold ON | Mapal iter ms, OFF (host fold) | OFF/ON | Mapal kernels ms (site + fold) | best library ms (leg) | library / Mapal iter | library / Mapal kernels |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| rtx-3090 | rowsum | 2048 | _exact | 2.55 / 3.07 | 6.45 / 8.44 | 2.53 / 2.75 | 1.43 | 0.95 (gemv) | 0.37 | 0.67 |
| rtx-3090 | rowsum | 2048 | _c | 1.95 / 2.41 | 5.90 / 7.83 | 3.03 / 3.25 | 0.85 | 0.95 (gemv) | 0.49 | 1.11 |
| rtx-3090 | rowsum | 4096 | _exact | 12.13 / 13.33 | 27.61 / 34.82 | 2.28 / 2.61 | 8.99 | 6.70 (eager) | 0.55 | 0.75 |
| rtx-3090 | rowsum | 4096 | _c | 8.75 / 9.70 | 24.33 / 31.11 | 2.78 / 3.21 | 5.62 | 6.70 (eager) | 0.77 | 1.19 |
| rtx-3090 | softmax_rows | 2048 | _exact | 10.37 / 14.38 | 10.74 / 15.63 | 1.04 / 1.09 | 1.43 | 0.88 (fused) | 0.08 | 0.61 |
| rtx-3090 | softmax_rows | 2048 | _c | 9.87 / 13.85 | 10.03 / 14.74 | 1.02 / 1.06 | 0.87 | 0.88 (fused) | 0.09 | 1.00 |
| rtx-3090 | softmax_rows | 4096 | _exact | 44.17 / 57.43 | 46.78 / 63.99 | 1.06 / 1.11 | 9.25 | 7.03 (fused) | 0.16 | 0.76 |
| rtx-3090 | softmax_rows | 4096 | _c | 41.44 / 53.26 | 43.38 / 59.73 | 1.05 / 1.12 | 5.75 | 7.03 (fused) | 0.17 | 1.22 |
| a100 | rowsum | 2048 | _exact | 2.94 / 5.15 | 12.58 / 16.83 | 4.28 / 3.27 | 2.43 | 0.99 (eager) | 0.34 | 0.41 |
| a100 | rowsum | 2048 | _c | 1.84 / 3.71 | 11.42 / 15.46 | 6.20 / 4.17 | 1.31 | 0.99 (eager) | 0.54 | 0.76 |
| a100 | rowsum | 4096 | _exact | 16.19 / 22.16 | 54.41 / 71.07 | 3.36 / 3.21 | 15.57 | 7.28 (eager) | 0.45 | 0.47 |
| a100 | rowsum | 4096 | _c | 8.82 / 12.64 | 46.98 / 61.35 | 5.32 / 4.85 | 8.16 | 7.28 (eager) | 0.83 | 0.89 |
| a100 | softmax_rows | 2048 | _exact | 20.33 / 17.24 | 25.99 / 18.46 | 1.28 / 1.07 | 2.44 | 1.04 (fused) | 0.05 | 0.43 |
| a100 | softmax_rows | 2048 | _c | 19.27 / 15.89 | 24.87 / 16.88 | 1.29 / 1.06 | 1.33 | 1.04 (fused) | 0.05 | 0.78 |
| a100 | softmax_rows | 4096 | _exact | 83.85 / 63.93 | 107.69 / 71.29 | 1.28 / 1.12 | 15.58 | 7.50 (fused) | 0.09 | 0.48 |
| a100 | softmax_rows | 4096 | _c | 76.22 / 56.83 | 99.98 / 63.78 | 1.31 / 1.12 | 8.17 | 7.50 (fused) | 0.10 | 0.92 |
| rtx-4090 | rowsum | 2048 | _exact | 1.38 / 1.71 | 4.79 / 6.01 | 3.48 / 3.52 | 0.53 | 0.33 (eager) | 0.24 | 0.62 |
| rtx-4090 | rowsum | 2048 | _c | 1.24 / 1.45 | 4.72 / 5.77 | 3.79 / 3.99 | 0.43 | 0.33 (eager) | 0.26 | 0.75 |
| rtx-4090 | rowsum | 4096 | _exact | 6.46 / 6.93 | 20.69 / 23.63 | 3.20 / 3.41 | 4.02 | 2.57 (eager) | 0.40 | 0.64 |
| rtx-4090 | rowsum | 4096 | _c | 4.98 / 5.36 | 19.37 / 22.48 | 3.89 / 4.20 | 2.56 | 2.57 (eager) | 0.52 | 1.00 |
| rtx-4090 | softmax_rows | 2048 | _exact | 7.76 / 9.67 | 8.98 / 11.40 | 1.16 / 1.18 | 0.54 | 0.35 (fused) | 0.05 | 0.66 |
| rtx-4090 | softmax_rows | 2048 | _c | 7.61 / 9.43 | 8.88 / 11.21 | 1.17 / 1.19 | 0.44 | 0.35 (fused) | 0.05 | 0.81 |
| rtx-4090 | softmax_rows | 4096 | _exact | 34.40 / 38.65 | 41.65 / 45.51 | 1.21 / 1.18 | 4.03 | 2.66 (fused) | 0.08 | 0.66 |
| rtx-4090 | softmax_rows | 4096 | _c | 33.22 / 37.23 | 39.74 / 44.48 | 1.20 / 1.19 | 2.58 | 2.66 (fused) | 0.08 | 1.03 |
| h100-pcie | rowsum | 2048 | _exact | 1.91 / 3.87 | 7.72 / 9.09 | 4.04 / 2.35 | 1.17 | 0.52 (gemv) | 0.27 | 0.45 |
| h100-pcie | rowsum | 2048 | _c | 1.40 / 1.56 | 6.71 / 15.34 | 4.80 / 9.85 | 0.70 | 0.52 (gemv) | 0.37 | 0.74 |
| h100-pcie | rowsum | 4096 | _exact | 7.40 / 8.57 | 26.91 / 58.87 | 3.64 / 6.87 | 6.45 | 3.81 (gemv) | 0.52 | 0.59 |
| h100-pcie | rowsum | 4096 | _c | 4.69 / 5.46 | 23.86 / 51.66 | 5.09 / 9.47 | 3.76 | 3.81 (gemv) | 0.81 | 1.01 |
| h100-pcie | softmax_rows | 2048 | _exact | 13.18 / 19.95 | 14.22 / 20.19 | 1.08 / 1.01 | 1.18 | 0.55 (fused) | 0.04 | 0.47 |
| h100-pcie | softmax_rows | 2048 | _c | 8.87 / 18.92 | 18.68 / 19.80 | 2.11 / 1.05 | 0.72 | 0.55 (fused) | 0.06 | 0.77 |
| h100-pcie | softmax_rows | 4096 | _exact | 39.94 / 60.29 | 48.31 / 62.92 | 1.21 / 1.04 | 6.47 | 3.93 (fused) | 0.10 | 0.61 |
| h100-pcie | softmax_rows | 4096 | _c | 42.16 / 61.03 | 44.40 / 58.42 | 1.05 / 0.96 | 3.78 | 3.93 (fused) | 0.09 | 1.04 |
| rtx-5090 | rowsum | 2048 | _exact | 1.28 / 1.37 | 6.75 / 6.79 | 5.26 / 4.98 | 0.52 | 0.26 (eager) | 0.20 | 0.50 |
| rtx-5090 | rowsum | 2048 | _c | 1.00 / 1.20 | 6.54 / 6.67 | 6.55 / 5.57 | 0.34 | 0.26 (eager) | 0.26 | 0.77 |
| rtx-5090 | rowsum | 4096 | _exact | 4.37 / 4.48 | 26.74 / 26.28 | 6.12 / 5.86 | 3.43 | 1.98 (eager) | 0.45 | 0.58 |
| rtx-5090 | rowsum | 4096 | _c | 3.07 / 3.18 | 25.43 / 25.18 | 8.28 / 7.91 | 2.15 | 1.98 (eager) | 0.65 | 0.92 |
| rtx-5090 | softmax_rows | 2048 | _exact | 11.84 / 11.98 | 13.82 / 13.69 | 1.17 / 1.14 | 0.53 | 0.28 (eager) | 0.02 | 0.54 |
| rtx-5090 | softmax_rows | 2048 | _c | 11.73 / 12.44 | 13.61 / 13.31 | 1.16 / 1.07 | 0.34 | 0.28 (eager) | 0.02 | 0.82 |
| rtx-5090 | softmax_rows | 4096 | _exact | 46.20 / 45.97 | 54.60 / 53.94 | 1.18 / 1.17 | 3.47 | 2.07 (fused) | 0.04 | 0.60 |
| rtx-5090 | softmax_rows | 4096 | _c | 44.59 / 44.53 | 54.29 / 52.62 | 1.22 / 1.18 | 2.18 | 2.07 (fused) | 0.05 | 0.95 |

## The rule per part (benches/gpu/probe_rungs.py, two sessions)

### rtx-3090

| rung | face | rule on iter (2 sessions) | rule on device ms | per probe: OFF/ON iter (`*` disjoint) [OFF/ON traced kernel ms], per session |
|---|---|---|---|---|
| rowfold | _c | ON | OFF | rowsum_2048: 2.979* [0.931] 3.242* [0.927]; rowsum_4096: 2.746* [0.967] 3.253* [0.963]; softmax_rows_2048: 1.016 [0.913] 1.065* [0.914]; softmax_rows_4096: 1.047* [0.946] 1.122* [0.964] |
| rowfold | _exact | ON | OFF | rowsum_2048: 2.555* [0.962] 2.745* [0.960]; rowsum_4096: 2.254* [0.969] 2.556* [0.973]; softmax_rows_2048: 1.035 [0.952] 1.087* [0.948]; softmax_rows_4096: 1.059* [1.013] 1.114* [0.978] |
| rowfold-layout | _c | OFF | OFF | rowsum_2048: 0.984 [1.000] 0.997 [0.999]; rowsum_4096: 0.987 [0.998] 1.014 [0.999] |
| rowfold-layout | _exact | OFF | OFF | rowsum_2048: 1.008 [1.004] 0.999 [1.003]; rowsum_4096: 0.990 [0.990] 0.978 [0.993] |

### a100

| rung | face | rule on iter (2 sessions) | rule on device ms | per probe: OFF/ON iter (`*` disjoint) [OFF/ON traced kernel ms], per session |
|---|---|---|---|---|
| rowfold | _c | ON | OFF | rowsum_2048: 6.237* [0.946] 4.191* [0.946]; rowsum_4096: 5.254* [0.949] 4.768* [0.951]; softmax_rows_2048: 1.291* [0.920] 1.062* [0.927]; softmax_rows_4096: 1.312* [0.967] 1.122 [0.969] |
| rowfold | _exact | ON | OFF | rowsum_2048: 4.333* [0.977] 3.310* [0.977]; rowsum_4096: 3.382* [0.989] 3.224* [0.990]; softmax_rows_2048: 1.278* [0.957] 1.071* [0.961]; softmax_rows_4096: 1.284* [0.983] 1.115 [0.984] |
| rowfold-layout | _c | OFF | OFF | rowsum_2048: 1.007 [1.015] 1.005 [1.013]; rowsum_4096: 0.987 [0.980] 0.983* [0.981] |
| rowfold-layout | _exact | ON | ON | rowsum_2048: 1.012 [1.015] 1.012 [1.013]; rowsum_4096: 1.006 [1.006] 1.005 [1.006] |

### rtx-4090

| rung | face | rule on iter (2 sessions) | rule on device ms | per probe: OFF/ON iter (`*` disjoint) [OFF/ON traced kernel ms], per session |
|---|---|---|---|---|
| rowfold | _c | ON | OFF | rowsum_2048: 4.235* [1.251] 4.105* [0.976]; rowsum_4096: 3.922* [0.965] 4.181* [0.962]; softmax_rows_2048: 1.167* [0.902] 1.190* [0.871]; softmax_rows_4096: 1.196* [0.941] 1.195* [0.945] |
| rowfold | _exact | ON | OFF | rowsum_2048: 3.551* [0.961] 3.541* [0.991]; rowsum_4096: 3.117* [0.925] 3.331* [0.943]; softmax_rows_2048: 1.157* [0.918] 1.179 [0.917]; softmax_rows_4096: 1.211* [0.963] 1.178* [0.961] |
| rowfold-layout | _c | ON | ON | rowsum_2048: 1.116* [1.381] 1.030 [1.104]; rowsum_4096: 1.008 [1.023] 0.997 [1.013] |
| rowfold-layout | _exact | OFF | OFF | rowsum_2048: 1.021 [1.027] 1.006 [1.067]; rowsum_4096: 0.974* [0.959] 0.977* [0.975] |

### h100-pcie

| rung | face | rule on iter (2 sessions) | rule on device ms | per probe: OFF/ON iter (`*` disjoint) [OFF/ON traced kernel ms], per session |
|---|---|---|---|---|
| rowfold | _c | ON | OFF | rowsum_2048: 4.060 [0.864] 3.421 [0.870]; rowsum_4096: 4.641 [0.882] 9.187 [0.888]; softmax_rows_2048: 2.107 [0.873] 1.047 [0.883]; softmax_rows_4096: 1.053 [0.941] 0.957 [0.943] |
| rowfold | _exact | ON | OFF | rowsum_2048: 1.992 [0.834] 3.924 [0.846]; rowsum_4096: 2.861 [0.795] 5.879* [0.811]; softmax_rows_2048: 1.078 [0.926] 1.012 [0.931]; softmax_rows_4096: 1.209 [0.964] 1.044 [0.967] |
| rowfold-layout | _c | OFF | OFF | rowsum_2048: 0.846 [0.967] 0.347 [0.968]; rowsum_4096: 0.912 [0.932] 0.971 [0.938] |
| rowfold-layout | _exact | OFF | OFF | rowsum_2048: 0.493 [0.892] 1.672 [0.900]; rowsum_4096: 0.787 [0.822] 0.856 [0.837] |

### rtx-5090

| rung | face | rule on iter (2 sessions) | rule on device ms | per probe: OFF/ON iter (`*` disjoint) [OFF/ON traced kernel ms], per session |
|---|---|---|---|---|
| rowfold | _c | ON | OFF | rowsum_2048: 6.219* [0.894] 5.550* [0.900]; rowsum_4096: 8.136* [0.931] 7.670* [0.928]; softmax_rows_2048: 1.161* [0.860] 1.070 [0.862]; softmax_rows_4096: 1.218* [0.947] 1.182* [0.944] |
| rowfold | _exact | ON | OFF | rowsum_2048: 5.582* [0.947] 4.983* [0.949]; rowsum_4096: 6.064* [0.961] 5.818* [0.961]; softmax_rows_2048: 1.167* [0.912] 1.143 [0.914]; softmax_rows_4096: 1.182* [0.965] 1.173* [0.965] |
| rowfold-layout | _c | OFF | OFF | rowsum_2048: 0.949 [1.018] 0.997 [1.022]; rowsum_4096: 0.982 [0.972] 0.970 [0.966] |
| rowfold-layout | _exact | OFF | OFF | rowsum_2048: 1.062 [1.026] 1.001 [1.029]; rowsum_4096: 0.991 [0.987] 0.992 [0.986] |

## rowfold-layout on the device clock: traced kernel ms (site + fold), column-major vs row-major

```
rowfold-rtx-3090.txt | _exact 2048 1.424 vs 1.429; _c 2048 0.854 vs 0.854; _exact 4096 9.081 vs 8.994; _c 4096 5.637 vs 5.624
rowfold-rtx-3090-confirm.txt | _exact 2048 1.589 vs 1.594; _c 2048 0.946 vs 0.945; _exact 4096 9.856 vs 9.790; _c 4096 5.936 vs 5.929
rowfold-a100.txt | _exact 2048 2.392 vs 2.428; _c 2048 1.292 vs 1.311; _exact 4096 15.479 vs 15.568; _c 4096 8.326 vs 8.162
rowfold-a100-confirm.txt | _exact 2048 3.080 vs 3.120; _c 2048 1.658 vs 1.680; _exact 4096 19.911 vs 20.027; _c 4096 10.694 vs 10.488
rowfold-rtx-4090.txt | _exact 2048 0.518 vs 0.532; _c 2048 0.315 vs 0.435; _exact 4096 4.189 vs 4.018; _c 4096 2.507 vs 2.564
rowfold-rtx-4090-confirm.txt | _exact 2048 0.535 vs 0.571; _c 2048 0.327 vs 0.361; _exact 4096 4.267 vs 4.160; _c 4096 2.605 vs 2.640
rowfold-h100-pcie.txt | _exact 2048 1.311 vs 1.170; _c 2048 0.728 vs 0.704; _exact 4096 7.851 vs 6.451; _c 4096 4.033 vs 3.760
rowfold-h100-pcie-confirm.txt | _exact 2048 1.521 vs 1.369; _c 2048 0.846 vs 0.819; _exact 4096 9.101 vs 7.615; _c 4096 4.735 vs 4.441
rowfold-rtx-5090.txt | _exact 2048 0.506 vs 0.519; _c 2048 0.330 vs 0.336; _exact 4096 3.478 vs 3.432; _c 4096 2.216 vs 2.153
rowfold-rtx-5090-confirm.txt | _exact 2048 0.490 vs 0.504; _c 2048 0.321 vs 0.328; _exact 4096 3.378 vs 3.330; _c 4096 2.161 vs 2.088
```
