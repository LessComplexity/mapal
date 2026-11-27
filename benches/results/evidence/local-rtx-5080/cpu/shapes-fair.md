# CPU shapes, fair pairs: Threadripper 9970X, 2026-11-27

Machine: AMD Ryzen Threadripper 9970X, 32 cores / 64 threads; clang 23.1.5; NumPy 2.5.5; target=native, PAR=64,
25 interleaved cycles, every value gate passed before timing. Source of the numbers: `shapes-fair.json`.

Command: `P1CPU=12 bash benches/local/cpu_shapes.sh <dir>`

Times in ms. Par and mt legs: median of 25; 1t legs: best of 25.

| workload | mapal-conf-par | mapal-fma-par | mapal-conf-1t | mapal-fma-1t | cpp-1t | cpp-mt | numpy-1t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FIR filter, 1M | 0.1877 | 0.1724 | 1.0045 | 1.0006 | 20.8443 | 1.7100 | 22.8975 |
| conv2d 3x3, 1024² | 0.1066 | 0.1044 | 0.1202 | 0.0990 | 0.1587 | 0.9747 | 1.9387 |
| saxpy, 1M | 0.0838 | 0.0750 | 0.1187 | 0.1097 | 0.1125 | 0.9154 | 0.2390 |
| sum reduction, 1M | 0.0439 | 0.0438 | 0.0340 | 0.0337 | 0.4716 | 9.0436 | 0.1573 |
| transpose, 1024² | 0.3159 | 0.2623 | 1.1806 | 1.1575 | 2.8223 | 1.3495 | 3.8453 |
| gather x[idx[i]], 1M | 0.2725 | 0.2208 | 0.5335 | 0.5351 | 0.5177 | 0.9265 | 0.8352 |

Fair pairs (competitor time / Mapal time; above 1: Mapal faster). Mapal is the better of its two legs of the same
precision face. The harness's own "best Mapal vs C++ mt" and "vs NumPy 1t" columns mix thread counts and are not used.

| workload | 1t: vs NumPy | 1t: vs C++ 1t | whole machine: Mapal par vs C++ best of (1t, mt) |
| --- | ---: | ---: | ---: |
| FIR filter, 1M | 22.9 | 20.8 | 9.92 |
| conv2d 3x3, 1024² | 19.6 | 1.60 | 1.52 |
| saxpy, 1M | 2.18 | 1.03 | 1.50 |
| sum reduction, 1M | 4.67 | 14.0 | 10.8 |
| transpose, 1024² | 3.32 | 2.44 | 5.14 |
| gather x[idx[i]], 1M | 1.57 | 0.97 | 2.34 |

Notes: the harness's cpp-mt starts its threads on every call and is slower than its own 1t at 1M elements, so the
C++ whole-machine baseline is the faster of cpp-1t and cpp-mt (cpp-1t for conv2d, saxpy, sum, gather; cpp-mt for FIR,
transpose). NumPy runs these operations on one thread. Mapal's sum is a vectorized reordered sum within its stated
error bound; C++ and NumPy keep strict order. Mapal's own one-thread sum (0.0337) is faster than its par leg (0.0438).
