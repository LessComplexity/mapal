# Whole machine against multi-threaded NumPy (OpenBLAS)

Machine: AMD Ryzen Threadripper 9970X, 32 cores / 64 threads, clocks capped at 4.5 GHz; clang 23.1.5; NumPy 2.5.5 (official wheel, OpenBLAS 0.3.34); 2026-11-28. Every value gate passed before timing (`numpy-mt-gate.txt`).

Method: medians of 21 interleaved runs; each side at its better of 32 / 64 threads. Ratio = NumPy time ÷ Mapal time,
95 % bootstrap interval, Mann-Whitney p; a side wins only when the interval excludes 1 and p < 0.05.

Command: `PYTHON=<venv>/bin/python CYCLES=21 bash benches/local/cpu_numpy_mt.sh <dir>`. NumPy must be the official wheel
with OpenBLAS; the distribution NumPy linked to the reference BLAS is about 250x slower on matrix multiply.
Samples: `numpy-mt.json`.

ms, median of interleaved runs; each side at its better of 32 / 64 threads. Ratio = NumPy ÷ Mapal (> 1 = Mapal
faster), [95 % bootstrap interval], Mann–Whitney p; a verdict only when the interval excludes 1 and p < 0.05.

| workload | Mapal 32 | Mapal 64 | NumPy 32 | NumPy 64 | NumPy ÷ Mapal |
| --- | ---: | ---: | ---: | ---: | --- |
| matrix multiply 1024² | 0.840 | 0.806 | 0.877 | 1.486 | 1.09× [1.04, 1.16] p<0.001 faster (Mapal 64, NumPy 32 threads) |
| matrix multiply 2048² | 3.935 | 3.603 | 5.642 | 6.005 | 1.57× [1.48, 1.61] p<0.001 faster (Mapal 64, NumPy 32 threads) |
| matrix multiply 4096² | 36.710 | 37.038 | 22.697 | 28.666 | 0.62× [0.60, 0.64] p<0.001 slower (Mapal 32, NumPy 32 threads) |
| 2-layer network, width 512 | 0.396 | 0.429 | 0.515 | 0.497 | 1.25× [1.13, 1.37] p<0.001 faster (Mapal 32, NumPy 64 threads) |
| 4-layer network, width 512 | 0.759 | 0.821 | 1.051 | 1.008 | 1.33× [1.24, 1.46] p<0.001 faster (Mapal 32, NumPy 64 threads) |
| 2-layer network, width 1024 | 1.604 | 1.591 | 1.603 | 1.787 | 1.01× [0.98, 1.04] p=0.252 no clear difference (Mapal 64, NumPy 32 threads) |
| 4-layer network, width 1024 | 3.096 | 3.136 | 3.155 | 3.511 | 1.02× [0.99, 1.10] p=0.061 no clear difference (Mapal 32, NumPy 32 threads) |
| 2-layer network, width 2048 | 8.425 | 7.329 | 8.009 | 8.781 | 1.09× [1.04, 1.46] p<0.001 faster (Mapal 64, NumPy 32 threads) |
| 4-layer network, width 2048 | 17.289 | 14.581 | 20.823 | 18.574 | 1.27× [1.18, 1.33] p<0.001 faster (Mapal 64, NumPy 64 threads) |
