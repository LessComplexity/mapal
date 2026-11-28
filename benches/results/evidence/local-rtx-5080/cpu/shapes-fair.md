# CPU shapes against careful C++ and NumPy

Machine: AMD Ryzen Threadripper 9970X, 32 cores / 64 threads, clocks capped at 4.5 GHz; clang 23.1.5; NumPy 2.5.5 (official wheel, OpenBLAS 0.3.34); 2026-11-28. Every value gate passed before timing (`shapes-gate.txt`).

Method: medians of 31 interleaved runs, one process per run. Ratio = competitor time ÷ Mapal time (above 1: Mapal
faster), with its 95 % bootstrap interval and the Mann-Whitney p-value. A side wins only when the interval excludes 1
and p < 0.05. The C++ is written carefully (register-blocked tiles, blocked transpose, a thread pool started before
the timer) and built strict and with -ffast-math; each comparison takes C++'s faster build. "Whole machine" is Mapal's
64-thread build against the faster of C++'s one-thread and threaded builds. NumPy runs these on one thread.

Command: `PYTHON=<venv>/bin/python CYCLES=31 bash benches/local/cpu_shapes.sh <dir>` (NumPy must be the official wheel
with OpenBLAS; the distribution NumPy linked to the reference BLAS is about 250x slower on matrix multiply).
Samples: `shapes-fair.json`.

ms, median of n interleaved runs (one process per run). Ratio = competitor ÷ Mapal (> 1 = Mapal faster),
[95 % bootstrap interval], Mann–Whitney p; a verdict only when the interval excludes 1 and p < 0.05.

| workload | mapal-conf-1t | mapal-fma-1t | mapal-conf-par | mapal-fma-par | cpp-1t | cpp-fast-1t | cpp-mt | cpp-fast-mt | numpy-1t | 1 thread vs C++ | 1 thread vs NumPy | whole machine vs C++ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FIR filter, 1M | 1.0034 | 1.0014 | 0.3140 | 0.2851 | 0.9546 | 0.9553 | 0.2679 | 0.2801 | 6.2366 | 0.95× [0.94, 0.95] p<0.001 slower | 6.23× [6.16, 6.24] p<0.001 faster | 0.94× [0.84, 1.07] p=0.477 no clear difference |
| conv2d 3x3, 1024² | 0.1303 | 0.1095 | 0.1946 | 0.1774 | 0.1663 | 0.1661 | 0.2197 | 0.2261 | 2.6011 | 1.52× [1.50, 1.54] p<0.001 faster | 23.76× [23.53, 24.06] p<0.001 faster | 0.94× [0.88, 1.07] p=0.468 no clear difference |
| saxpy, 1M | 0.1314 | 0.1170 | 0.1221 | 0.1184 | 0.1155 | 0.1127 | 0.2439 | 0.2463 | 0.2449 | 0.96× [0.93, 0.99] p=0.003 slower | 2.09× [2.02, 2.14] p<0.001 faster | 0.95× [0.93, 0.99] p=0.023 slower |
| sum reduction, 1M | 0.0340 | 0.0337 | 0.0786 | 0.0770 | 0.4666 | 0.0339 | 0.2120 | 0.1631 | 0.1454 | 1.01× [0.99, 1.01] p=0.800 no clear difference | 4.31× [4.29, 4.35] p<0.001 faster | 0.44× [0.40, 0.46] p<0.001 slower |
| transpose, 1024² | 1.1805 | 1.1780 | 0.3020 | 0.3117 | 0.9444 | 0.9398 | 0.2865 | 0.2781 | 6.1368 | 0.80× [0.79, 0.81] p<0.001 slower | 5.21× [4.69, 5.37] p<0.001 faster | 0.92× [0.82, 0.98] p=0.002 slower |
| gather x[idx[i]], 1M | 0.6665 | 0.6370 | 0.3083 | 0.2977 | 0.5246 | 0.5162 | 0.3758 | 0.3785 | 0.9029 | 0.81× [0.81, 0.82] p<0.001 slower | 1.42× [1.38, 1.44] p<0.001 faster | 1.26× [1.17, 1.35] p<0.001 faster |
