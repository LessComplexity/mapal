# S54 `streams` / `pinned` / `pdl` — results

Item R7 of plan-s54 (branch `s54/streams`). Every number: median of the program's own `iter ms` (the clock around its computations) over interleaved reps, exact / contract face; `*` = disjoint min–max; ratios are OFF/ON (> 1 = ON faster). Raw rows: `streams-<part>[-confirm|-final|-shipped].txt`; static counts: `streams-fusion-report.txt`.

## Sessions

| part | session | differential | trap-before-site | box copy GB/s h2d pageable/pinned · d2h pageable/pinned |
|---|---|---:|---:|---|
| RTX 3090 (sm_86) | s1: first build | 52/52 | 4/4 | 8.54/11.82 · 10.35/13.19 |
| RTX 3090 (sm_86) | s2: runtime v2 | 36/36 | 2/2 | 13.04/13.44 · 10.10/13.53 |
| RTX 3090 (sm_86) | s3: device-only rule | 32/32 | 2/2 | 9.68/13.41 · 8.51/13.50 |
| A100 SXM4 (sm_80) | s1: first build | 52/52 | 4/4 | 18.36/21.90 · 16.86/19.85 |
| A100 SXM4 (sm_80) | s2: runtime v2 | 36/36 | 2/2 | 13.44/23.84 · 10.09/19.86 |
| A100 SXM4 (sm_80) | s3: device-only rule | 32/32 | 2/2 | 11.48/11.75 · 11.60/12.25 |
| A100 SXM4 (sm_80) | s4: shipped table | 32/32 | 2/2 | 7.91/10.90 · 9.84/11.20 |
| RTX 4090 (sm_89) | s1: first build | 52/52 | 4/4 | 13.31/13.43 · 12.43/13.18 |
| RTX 4090 (sm_89) | s3: device-only rule | 32/32 | 2/2 | 11.59/14.59 · 10.79/12.79 |
| H100 PCIe (sm_90a) | s1: first build | 74/74 | 6/6 | 24.75/57.21 · 21.25/53.47 |
| H100 PCIe (sm_90a) | s2: runtime v2 | 60/60 | 4/4 | 19.50/25.62 · 4.53/5.12 |
| H100 SXM (sm_90a, profile h100) | s3: device-only rule | 58/58 | 4/4 | 9.37/54.98 · 15.48/55.03 |
| RTX 5090 (sm_120) | s1: first build | 74/74 | 6/6 | 9.11/25.84 · 8.65/28.17 |
| RTX 5090 (sm_120) | s2: runtime v2 | 60/60 | 4/4 | 18.57/52.31 · 12.43/25.15 |
| RTX 5090 (sm_120) | s3: device-only rule | 58/58 | 4/4 | 14.20/14.23 · 13.67/14.13 |
| RTX 5090 (sm_120) | s4: shipped table | 58/58 | 4/4 | 13.71/23.53 · 9.67/21.45 |

## What ships (`GPU_RUNGS_OFF`)

| rung | RTX 3090 | A100 | RTX 4090 | H100 PCIe (and SXM) | RTX 5090 |
|---|---|---|---|---|---|
| `streams` | OFF | **ON** | OFF | OFF | **ON** |
| `pinned` | OFF | OFF | OFF | OFF | OFF |
| `pdl` | n/a (sm_86) | n/a (sm_80) | n/a (sm_89) | OFF | OFF |

- **streams ON, A100.** On the device-only shape, n=512: 1.04, 1.07–1.09 and 1.07–1.11× (s2, s3, s4). At n=2048 the last waves merge (two grids of about 1.2 waves run in 3 waves instead of 4): 1.27–1.32, 1.27–1.37 and 1.32–1.43× (s2 and s3 with the gate forced open, s4 shipped).
- **streams ON, RTX 5090.** n=512: 1.04, 1.05–1.06 and 1.01–1.11× (s2, s3, s4). n=4096 (last waves merge): 1.04–1.05×, disjoint (s4).
- **streams OFF, RTX 4090.** n=512: 0.94–0.99×. n=1024: 0.94–0.95× (s3).
- **streams OFF, RTX 3090.** n=512: 0.99–1.01× in s2 and s3, i.e. no effect.
- **streams OFF, H100.** The H100 PCIe s2 box was too noisy to read (its reps spread from 0.45 to 4.1 ms at n=512), and no third PCIe box was offered. On the SXM, n=512 measured 1.01–1.03× in one session, inside the noise floor.
- **The gate where it says no.** Forcing streams open measured 0.94–0.98× on the RTX 4090 at 2048 and 0.96–0.99× on the RTX 5090 at 1024–2048. On the RTX 3090 at 1024–2048 the sign was mixed. On the A100 it measured 1.08–1.24× at 1024 in s2 and s4 but 0.97× in s3, and 1.08–1.10× at 4096 in s4: the A100 gains even where the wave estimate (tile residency 2 blocks/SM) says the grids do not merge.
- **pinned.** One crossing never repays the registration: 0.06–0.09 ms per MB at 64 MB and 0.3–0.5 at 1 MB, against at most 0.08 ms per MB saved per crossing. In the loop, the sign follows the host. Boxes whose pageable copy runs at 0.3–0.6 of pinned gained 17–21% (RTX 5090 s1 and s2, H100 SXM) or 4–5% (A100 s1 and s2). Boxes whose pageable copy equals pinned lost 1–5% (RTX 5090 s3, A100 s3). The 3090 and 4090 showed no stable sign. That is a fact of the host, not of the part.
- **pdl.** No probe kept its sign across two sessions on the H100 or the RTX 5090.
- **Host-joined probes (indep, diamond, mlp2), in the first two builds.** diamond and mlp2 ran slower on streams on every part, at 0.60–0.96× (a few cells on the noisy H100 s2 box read above 1). indep at 512 flipped sign between sessions on the same part (0.69–1.21×). Where the first build opened them at 2048–4096, they ran at 0.63–0.93× (one cell, A100 indep_2048 exact, at 1.24). They no longer open streams, so their shipped text is the one-stream text.


## streams on the device-only shape (`twin`: two independent sites, corners read)

| part | n | session | streams ms | one stream ms | one/streams (>1 = streams faster) | streams (min) | one stream (min) |
|---|---:|---|---:|---:|---:|---:|---:|
| RTX 3090 (sm_86) | 512 | s2: runtime v2 (gate open) | 0.971 / 0.941 | 0.967 / 0.945 | 0.997 / 1.004 | 0.773 / 0.921 | 0.779 / 0.743 |
| RTX 3090 (sm_86) | 512 | s3: device-only rule (gate open) | 0.706 / 0.700 | 0.711 / 0.696 | 1.007 / 0.994 | 0.528 / 0.540 | 0.536 / 0.539 |
| RTX 3090 (sm_86) | 1024 | s2: runtime v2 (gate closed: arm forces streams) | 2.131 / 2.035 | 2.156 / 1.983 | 1.012 / 0.974 | 1.856 / 1.785 | 1.742 / 1.636 |
| RTX 3090 (sm_86) | 1024 | s3: device-only rule (gate closed: arm forces streams) | 1.749 / 1.623 | 1.725 / 1.585 | 0.987 / 0.977 | 1.473 / 1.363 | 1.373 / 1.294 |
| RTX 3090 (sm_86) | 2048 | s2: runtime v2 (gate closed: arm forces streams) | 5.138 / 4.193 | 5.392 / 4.093 | 1.049 / 0.976 | 5.015 / 4.052 | 5.250 / 4.048 |
| RTX 3090 (sm_86) | 2048 | s3: device-only rule (gate closed: arm forces streams) | 4.681 / 3.802 | 5.017 / 3.710 | 1.072 / 0.976 | 4.655 / 3.777 | 4.988 / 3.684 |
| A100 SXM4 (sm_80) | 512 | s2: runtime v2 (gate open) | 1.000 / 0.937 | 1.040 / 0.972 | 1.040 / 1.038 | 0.777 / 0.928 | 1.020 / 0.950 |
| A100 SXM4 (sm_80) | 512 | s3: device-only rule (gate open) | 0.969 / 0.898 | 1.054 / 0.960 | 1.088 / 1.069 | 0.651 / 0.597 | 0.724 / 0.658 |
| A100 SXM4 (sm_80) | 512 | s4: shipped table (gate open) | 0.483 / 0.437 | 0.518 / 0.486 | 1.073 / 1.111 | 0.461 / 0.370 | 0.501 / 0.451 |
| A100 SXM4 (sm_80) | 1024 | s2: runtime v2 (gate closed: arm forces streams) | 2.370 / 1.977 | 2.670 / 2.143 | 1.126 / 1.084 | 2.284 / 1.913 | 2.249 / 2.067 |
| A100 SXM4 (sm_80) | 1024 | s3: device-only rule (gate closed: arm forces streams) | 2.315 / 1.949 | 2.250 / 1.885 | 0.972 / 0.967 | 1.689 / 1.314 | 1.883 / 1.519 |
| A100 SXM4 (sm_80) | 1024 | s4: shipped table (gate closed: arm forces streams) | 1.209 / 0.908 | 1.495 / 1.112 | 1.237 / 1.225 | 1.120 / 0.862 | 1.470 / 1.066 |
| A100 SXM4 (sm_80) | 2048 | s2: runtime v2 (gate closed: arm forces streams) | 6.229 / 3.818 | 7.936 / 5.030 | 1.274 / 1.318 | 6.118 / 3.783 | 7.872 / 5.003 |
| A100 SXM4 (sm_80) | 2048 | s3: device-only rule (gate closed: arm forces streams) | 6.204 / 3.716 | 7.866 / 5.083 | 1.268 / 1.368 | 5.955 / 3.563 | 7.737 / 4.890 |
| A100 SXM4 (sm_80) | 2048 | s4: shipped table (gate open) | 4.562 / 2.653 | 6.039 / 3.807 | 1.324 / 1.435 | 4.406 / 2.597 | 5.953 / 3.764 |
| A100 SXM4 (sm_80) | 4096 | s4: shipped table (gate closed: arm forces streams) | 33.316 / 17.574 | 35.950 / 19.169 | 1.079 / 1.091 | 30.803 / 16.171 | 33.454 / 17.956 |
| RTX 4090 (sm_89) | 512 | s3: device-only rule (gate open) | 0.448 / 0.387 | 0.442 / 0.364 | 0.987 / 0.940 | 0.375 / 0.373 | 0.358 / 0.358 |
| RTX 4090 (sm_89) | 1024 | s3: device-only rule (gate open) | 0.961 / 0.926 | 0.911 / 0.873 | 0.948 / 0.943 | 0.941 / 0.899 | 0.882 / 0.859 |
| RTX 4090 (sm_89) | 2048 | s3: device-only rule (gate closed: arm forces streams) | 2.488 / 2.094 | 2.387 / 1.970 | 0.959 / 0.941 | 2.458 / 2.082 | 2.356 / 1.955 |
| H100 PCIe (sm_90a) | 512 | s2: runtime v2 (gate open) | 4.089 / 3.570 | 3.802 / 0.855 | 0.930 / 0.239 | 0.461 / 0.465 | 0.454 / 0.561 |
| H100 PCIe (sm_90a) | 1024 | s2: runtime v2 (gate closed: arm forces streams) | 1.597 / 1.690 | 1.928 / 2.506 | 1.207 / 1.483 | 1.085 / 1.312 | 1.342 / 1.602 |
| H100 PCIe (sm_90a) | 2048 | s2: runtime v2 (gate closed: arm forces streams) | 4.357 / 2.367 | 5.666 / 3.647 | 1.301 / 1.541 | 2.842 / 2.100 | 3.795 / 2.204 |
| H100 SXM (sm_90a, profile h100) | 512 | s3: device-only rule (gate open) | 0.752 / 0.739 | 0.763 / 0.762 | 1.015 / 1.031 | 0.448 / 0.483 | 0.554 / 0.736 |
| H100 SXM (sm_90a, profile h100) | 1024 | s3: device-only rule (gate closed: arm forces streams) | 1.438 / 1.365 | 1.452 / 1.402 | 1.010 / 1.028 | 1.023 / 1.016 | 1.185 / 0.768 |
| H100 SXM (sm_90a, profile h100) | 2048 | s3: device-only rule (gate closed: arm forces streams) | 2.674 / 2.064 | 2.795 / 2.093 | 1.045 / 1.014 | 2.257 / 1.685 | 2.250 / 1.415 |
| RTX 5090 (sm_120) | 512 | s2: runtime v2 (gate open) | 0.138 / 0.147 | 0.143 / 0.802 | 1.037 / 5.438 | 0.119 / 0.113 | 0.130 / 0.133 |
| RTX 5090 (sm_120) | 512 | s3: device-only rule (gate open) | 0.159 / 0.154 | 0.167 / 0.163 | 1.049 / 1.059 | 0.136 / 0.137 | 0.148 / 0.152 |
| RTX 5090 (sm_120) | 512 | s4: shipped table (gate open) | 0.668 / 0.621 | 0.672 / 0.691 | 1.006 / 1.113 | 0.523 / 0.512 | 0.517 / 0.521 |
| RTX 5090 (sm_120) | 1024 | s2: runtime v2 (gate closed: arm forces streams) | 0.325 / 0.295 | 0.325 / 0.298 | 1.003 / 1.008 | 0.318 / 0.283 | 0.307 / 0.287 |
| RTX 5090 (sm_120) | 1024 | s3: device-only rule (gate closed: arm forces streams) | 0.379 / 0.345 | 0.364 / 0.332 | 0.960 / 0.962 | 0.371 / 0.333 | 0.352 / 0.316 |
| RTX 5090 (sm_120) | 1024 | s4: shipped table (gate closed: arm forces streams) | 1.328 / 1.281 | 1.337 / 1.329 | 1.007 / 1.037 | 1.027 / 1.000 | 1.025 / 1.018 |
| RTX 5090 (sm_120) | 2048 | s2: runtime v2 (gate closed: arm forces streams) | 1.299 / 0.884 | 1.262 / 0.871 | 0.972 / 0.986 | 1.293 / 0.873 | 1.254 / 0.856 |
| RTX 5090 (sm_120) | 2048 | s3: device-only rule (gate closed: arm forces streams) | 1.435 / 1.017 | 1.391 / 0.981 | 0.970 / 0.964 | 1.427 / 0.993 | 1.359 / 0.972 |
| RTX 5090 (sm_120) | 2048 | s4: shipped table (gate closed: arm forces streams) | 1.956 / 1.574 | 1.969 / 1.599 | 1.007 / 1.016 | 1.927 / 1.547 | 1.942 / 1.570 |
| RTX 5090 (sm_120) | 4096 | s4: shipped table (gate open) | 7.804 / 5.447 | 8.207 / 5.657 | 1.052 / 1.039 | 7.780 / 5.438 | 8.196 / 5.632 |

## streams on the host-joined probes (first two builds; no longer opened)

| part | session | probe | OFF/ON exact / contract |
|---|---|---|---:|
| RTX 3090 (sm_86) | s1: first build | pt_diamond_4096 | 0.891 / 0.870 |
| RTX 3090 (sm_86) | s1: first build | pt_diamond_512 | 0.878 / 0.873 |
| RTX 3090 (sm_86) | s1: first build | pt_indep_4096 | 0.882 / 0.852 |
| RTX 3090 (sm_86) | s1: first build | pt_indep_512 | 0.955 / 0.975 |
| RTX 3090 (sm_86) | s1: first build | pt_mlp2_4096 | 0.922 / 0.903 |
| RTX 3090 (sm_86) | s1: first build | pt_mlp2_512 | 0.838 / 0.840 |
| RTX 3090 (sm_86) | s2: runtime v2 | pt_diamond_512 | 0.822 / 0.805 |
| RTX 3090 (sm_86) | s2: runtime v2 | pt_indep_512 | 0.960 / 0.901 |
| RTX 3090 (sm_86) | s2: runtime v2 | pt_mlp2_512 | 0.727 / 0.616 |
| A100 SXM4 (sm_80) | s1: first build | pt_diamond_2048 | 0.816 / 0.807 |
| A100 SXM4 (sm_80) | s1: first build | pt_diamond_512 | 0.872 / 0.882 |
| A100 SXM4 (sm_80) | s1: first build | pt_indep_2048 | 1.241 / 0.923 |
| A100 SXM4 (sm_80) | s1: first build | pt_indep_512 | 0.949 / 0.981 |
| A100 SXM4 (sm_80) | s1: first build | pt_mlp2_2048 | 0.869 / 1.020 |
| A100 SXM4 (sm_80) | s1: first build | pt_mlp2_512 | 0.802 / 0.901 |
| A100 SXM4 (sm_80) | s2: runtime v2 | pt_diamond_512 | 0.780 / 0.811 |
| A100 SXM4 (sm_80) | s2: runtime v2 | pt_indep_512 | 1.000 / 0.997 |
| A100 SXM4 (sm_80) | s2: runtime v2 | pt_mlp2_512 | 0.745 / 0.768 |
| RTX 4090 (sm_89) | s1: first build | pt_diamond_1024 | 0.933 / 0.964 |
| RTX 4090 (sm_89) | s1: first build | pt_diamond_512 | 0.853 / 0.946 |
| RTX 4090 (sm_89) | s1: first build | pt_indep_1024 | 0.942 / 1.012 |
| RTX 4090 (sm_89) | s1: first build | pt_indep_512 | 1.000 / 1.060 |
| RTX 4090 (sm_89) | s1: first build | pt_mlp2_1024 | 0.926 / 0.944 |
| RTX 4090 (sm_89) | s1: first build | pt_mlp2_512 | 0.861 / 0.847 |
| H100 PCIe (sm_90a) | s1: first build | pt_diamond_2048 | 0.803 / 0.919 |
| H100 PCIe (sm_90a) | s1: first build | pt_diamond_4096 | 0.702 / 0.796 |
| H100 PCIe (sm_90a) | s1: first build | pt_diamond_512 | 0.887 / 0.879 |
| H100 PCIe (sm_90a) | s1: first build | pt_indep_2048 | 0.832 / 0.926 |
| H100 PCIe (sm_90a) | s1: first build | pt_indep_4096 | 0.629 / 0.762 |
| H100 PCIe (sm_90a) | s1: first build | pt_indep_512 | 1.213 / 1.086 |
| H100 PCIe (sm_90a) | s1: first build | pt_mlp2_2048 | 0.920 / 0.982 |
| H100 PCIe (sm_90a) | s1: first build | pt_mlp2_4096 | 0.836 / 0.921 |
| H100 PCIe (sm_90a) | s1: first build | pt_mlp2_512 | 0.880 / 1.056 |
| H100 PCIe (sm_90a) | s2: runtime v2 | pt_diamond_512 | 2.268 / 0.666 |
| H100 PCIe (sm_90a) | s2: runtime v2 | pt_indep_512 | 0.691 / 1.155 |
| H100 PCIe (sm_90a) | s2: runtime v2 | pt_mlp2_512 | 1.028 / 0.603 |
| RTX 5090 (sm_120) | s1: first build | pt_diamond_4096 | 0.795 / 0.901 |
| RTX 5090 (sm_120) | s1: first build | pt_diamond_512 | 0.936 / 0.904 |
| RTX 5090 (sm_120) | s1: first build | pt_indep_4096 | 0.819 / 0.888 |
| RTX 5090 (sm_120) | s1: first build | pt_indep_512 | 1.021 / 1.021 |
| RTX 5090 (sm_120) | s1: first build | pt_mlp2_4096 | 0.841 / 0.944 |
| RTX 5090 (sm_120) | s1: first build | pt_mlp2_512 | 0.959 / 0.921 |
| RTX 5090 (sm_120) | s2: runtime v2 | pt_diamond_512 | 0.936 / 0.861 |
| RTX 5090 (sm_120) | s2: runtime v2 | pt_indep_512 | 0.938 / 0.966 |
| RTX 5090 (sm_120) | s2: runtime v2 | pt_mlp2_512 | 0.932 / 0.893 |

## pinned (the loop's per-iteration copies; `pt_loop_2048`)

| part | session | ON ms | OFF ms | OFF/ON |
|---|---|---:|---:|---:|
| RTX 3090 (sm_86) | s1: first build | 130.755 / 126.012 | 131.314 / 125.685 | 1.004 / 0.997 |
| RTX 3090 (sm_86) | s2: runtime v2 | 107.350 / 103.422 | 103.603 / 98.630 | 0.965 / 0.954 |
| RTX 3090 (sm_86) | s3: device-only rule | 110.819 / 106.843 | 114.186 / 108.992 | 1.030 / 1.020 |
| A100 SXM4 (sm_80) | s1: first build | 97.927 / 93.810 | 103.262 / 95.645 | 1.054 / 1.020 |
| A100 SXM4 (sm_80) | s2: runtime v2 | 100.516 / 90.376 | 105.920 / 94.261 | 1.054 / 1.043 |
| A100 SXM4 (sm_80) | s3: device-only rule | 128.398 / 119.525 | 129.161 / 118.392 | 1.006 / 0.991 |
| RTX 4090 (sm_89) | s1: first build | 108.529 / 105.036 | 105.843 / 105.855 | 0.975 / 1.008 |
| RTX 4090 (sm_89) | s3: device-only rule | 69.409 / 68.007 | 70.917 / 69.474 | 1.022 / 1.022 |
| H100 PCIe (sm_90a) | s1: first build | 75.623 / 69.702 | 72.974 / 73.441 | 0.965 / 1.054 |
| H100 PCIe (sm_90a) | s2: runtime v2 | 70.811 / 69.016 | 72.437 / 69.029 | 1.023 / 1.000 |
| H100 SXM (sm_90a, profile h100) | s3: device-only rule | 88.755 / 86.122 | 107.428 / 102.595 | 1.210 / 1.191 |
| RTX 5090 (sm_120) | s1: first build | 77.545 / 75.830 | 92.102 / 88.746 | 1.188 / 1.170 |
| RTX 5090 (sm_120) | s2: runtime v2 | 50.086 / 48.318 | 58.865 / 57.286 | 1.175 / 1.186 |
| RTX 5090 (sm_120) | s3: device-only rule | 57.786 / 56.476 | 55.170 / 53.483 | 0.955 / 0.947 |

## Mapal vs the library legs (session 1, every probe; `table.py`)

Mapal here is session 1's ON arm, which is the first build: host-joined sites on streams and every crossing pinned. On the host-joined probes, the shipped text is that session's one-stream, pageable text, which is faster by the OFF/ON ratios above. Library legs are cudaEvent device time on device-built data; Mapal's is the program's own clock, host joins and copies included. `twin` has no library leg. Its nearest is the library's `indep` streams leg (two GEMMs plus the join): 0.037 ms at 512 on the RTX 4090, against twin's 0.15–0.97 ms per box, which is dominated by four one-element copy-outs.


### RTX 3090 (sm_86)

| probe | n | Mapal iter ms exact / contract | wall ms | kernel ms | launches | bytes in / out | library legs med ms (launches) | best library | best lib / Mapal exact | gate |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| diamond | 1024 | 4.308 / 4.205 | 284.6 | 0.66 | 3 | 0 / 8388608 | eager 0.357 (4) streams 0.344 (4) lt 0.341 (3) cutlass 0.388 (3) | lt | 0.08 | pass |
| diamond | 2048 | 14.700 / 13.722 | 291.3 | 4.58 | 3 | 0 / 33554432 | eager 2.645 (4) streams 2.602 (4) lt 2.634 (3) cutlass 3.071 (3) | streams | 0.18 | pass |
| diamond | 4096 | 72.516 / 61.010 | 487.2 | 28.75 | 3 | 0 / 134217728 | eager 20.123 (4) streams 19.172 (4) lt 19.685 (3) cutlass 19.940 (3) | streams | 0.26 | pass |
| diamond | 512 | 1.708 / 1.705 | 277.3 | 0.14 | 3 | 0 / 2097152 | eager 0.071 (4) streams 0.070 (4) lt 0.068 (3) cutlass 0.163 (3) | lt | 0.04 | pass |
| fanout | 1024 | 2.367 / 2.233 | 281.4 | 0.45 | 2 | 0 / 4194312 | eager 0.244 (3) streams 0.242 (3) | streams | 0.10 | pass |
| fanout | 2048 | 8.561 / 7.375 | 293.5 | 3.05 | 2 | 0 / 16777224 | eager 1.783 (3) streams 1.746 (3) | streams | 0.20 | pass |
| fanout | 4096 | 39.169 / 31.557 | 363.2 | 19.18 | 2 | 0 / 67108872 | eager 12.978 (3) streams 12.819 (3) | streams | 0.33 | pass |
| fanout | 512 | 0.890 / 0.900 | 279.2 | 0.09 | 2 | 0 / 1048584 | eager 0.056 (3) streams 0.053 (3) | streams | 0.06 | pass |
| indep | 1024 | 3.574 / 3.638 | 280.5 | 0.45 | 2 | 0 / 8388608 | eager 0.249 (3) streams 0.238 (3) lt 0.234 (2) cutlass 0.244 (2) | lt | 0.07 | pass |
| indep | 2048 | 12.905 / 12.373 | 296.1 | 3.05 | 2 | 0 / 33554432 | eager 1.777 (3) streams 1.732 (3) lt 1.764 (2) cutlass 2.035 (2) | streams | 0.13 | pass |
| indep | 4096 | 60.889 / 53.572 | 440.9 | 19.17 | 2 | 0 / 134217728 | eager 13.334 (3) streams 12.524 (3) lt 13.177 (2) cutlass 13.403 (2) | streams | 0.21 | pass |
| indep | 512 | 1.515 / 1.486 | 278.6 | 0.10 | 2 | 0 / 2097152 | eager 0.056 (3) streams 0.052 (3) lt 0.054 (2) cutlass 0.125 (2) | streams | 0.03 | pass |
| loop | 2048 | 130.755 / 126.012 | 411.1 | 12.21 | 8 | 150994944 / 134217728 | eager 7.033 (16) lt 6.705 (8) cutlass 7.933 (8) | lt | 0.05 | pass |
| mlp2 | 1024 | 7.094 / 6.926 | 285.8 | 0.91 | 4 | 8388608 / 8388624 | eager 0.544 (6) streams 0.500 (6) lt 0.552 (4) lt2 0.534 (4) cutlass 0.538 (4) | streams | 0.07 | pass |
| mlp2 | 2048 | 24.940 / 23.034 | 305.1 | 6.10 | 4 | 33554432 / 33554448 | eager 3.596 (6) streams 3.444 (6) lt 3.521 (4) lt2 3.360 (4) cutlass 4.180 (4) | lt2 | 0.13 | pass |
| mlp2 | 4096 | 114.577 / 100.062 | 683.3 | 38.35 | 4 | 134217728 / 134217744 | eager 27.486 (6) streams 24.180 (6) lt 26.533 (4) lt2 23.375 (4) cutlass 27.189 (4) | lt2 | 0.20 | pass |
| mlp2 | 512 | 2.875 / 2.860 | 280.2 | 0.21 | 4 | 2097152 / 2097168 | eager 0.111 (6) streams 0.086 (6) lt 0.111 (4) lt2 0.085 (4) cutlass 0.248 (4) | lt2 | 0.03 | pass |
| mlp | 1024 | 3.466 / 3.368 | 283.4 | 0.45 | 2 | 4194304 / 4194312 | eager 0.241 (3) lt 0.229 (2) cutlass 0.235 (2) | lt | 0.07 | pass |
| mlp | 2048 | 12.504 / 11.337 | 295.8 | 3.05 | 2 | 16777216 / 16777224 | eager 1.747 (3) lt 1.706 (2) cutlass 2.024 (2) | lt | 0.14 | pass |
| mlp | 4096 | 50.387 / 42.687 | 384.6 | 19.17 | 2 | 67108864 / 67108872 | eager 13.252 (3) lt 13.179 (2) cutlass 13.405 (2) | lt | 0.26 | pass |
| mlp | 512 | 1.376 / 1.352 | 275.3 | 0.10 | 2 | 1048576 / 1048584 | eager 0.052 (3) lt 0.048 (2) cutlass 0.115 (2) | lt | 0.03 | pass |

### A100 SXM4 (sm_80)

| probe | n | Mapal iter ms exact / contract | wall ms | kernel ms | launches | bytes in / out | library legs med ms (launches) | best library | best lib / Mapal exact | gate |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| diamond | 1024 | 5.049 / 4.783 | 469.2 | 1.44 | 3 | 0 / 8388608 | eager 0.476 (4) streams 0.463 (4) lt 0.467 (3) cutlass 0.704 (3) | streams | 0.09 | pass |
| diamond | 2048 | 18.102 / 14.023 | 482.9 | 8.55 | 3 | 0 / 33554432 | eager 2.953 (4) streams 2.926 (4) lt 2.927 (3) cutlass 3.442 (3) | streams | 0.16 | pass |
| diamond | 4096 | 76.727 / 46.776 | 593.1 | 56.04 | 3 | 0 / 134217728 | eager 21.833 (4) streams 21.791 (4) lt 21.725 (3) cutlass 22.741 (3) | lt | 0.28 | pass |
| diamond | 512 | 3.180 / 3.122 | 442.0 | 0.23 | 3 | 0 / 2097152 | eager 0.096 (4) streams 0.087 (4) lt 0.095 (3) cutlass 0.306 (3) | streams | 0.03 | pass |
| fanout | 1024 | 4.194 / 3.937 | 464.8 | 0.93 | 2 | 0 / 4194312 | eager 0.292 (3) streams 0.288 (3) | streams | 0.07 | pass |
| fanout | 2048 | 8.869 / 8.061 | 457.6 | 5.70 | 2 | 0 / 16777224 | eager 2.391 (3) streams 2.371 (3) | streams | 0.27 | pass |
| fanout | 4096 | 54.173 / 33.310 | 544.3 | 37.39 | 2 | 0 / 67108872 | eager 14.564 (3) streams 14.514 (3) | streams | 0.27 | pass |
| fanout | 512 | 1.217 / 1.242 | 426.9 | 0.14 | 2 | 0 / 1048584 | eager 0.073 (3) streams 0.072 (3) | streams | 0.06 | pass |
| indep | 1024 | 4.139 / 3.792 | 444.6 | 0.90 | 2 | 0 / 8388608 | eager 0.300 (3) streams 0.287 (3) lt 0.291 (2) cutlass 0.438 (2) | streams | 0.07 | pass |
| indep | 2048 | 14.518 / 18.006 | 494.8 | 5.71 | 2 | 0 / 33554432 | eager 2.397 (3) streams 2.369 (3) lt 2.369 (2) cutlass 2.793 (2) | streams | 0.16 | pass |
| indep | 4096 | 66.210 / 48.404 | 559.2 | 37.36 | 2 | 0 / 134217728 | eager 14.574 (3) streams 14.567 (3) lt 14.468 (2) cutlass 15.132 (2) | lt | 0.22 | pass |
| indep | 512 | 2.816 / 2.694 | 455.7 | 0.15 | 2 | 0 / 2097152 | eager 0.076 (3) streams 0.065 (3) lt 0.074 (2) cutlass 0.234 (2) | streams | 0.02 | pass |
| loop | 2048 | 97.927 / 93.810 | 565.6 | 22.85 | 8 | 150994944 / 134217728 | eager 7.950 (16) lt 8.195 (8) cutlass 9.123 (8) | eager | 0.08 | pass |
| mlp2 | 1024 | 8.712 / 8.190 | 467.5 | 1.97 | 4 | 8388608 / 8388624 | eager 0.654 (6) streams 0.615 (6) lt 0.661 (4) lt2 0.614 (4) cutlass 0.962 (4) | lt2 | 0.07 | pass |
| mlp2 | 2048 | 30.766 / 29.358 | 503.4 | 11.41 | 4 | 33554432 / 33554448 | eager 3.939 (6) streams 3.727 (6) lt 3.998 (4) lt2 3.759 (4) cutlass 4.569 (4) | streams | 0.12 | pass |
| mlp2 | 4096 | 121.635 / 81.110 | 826.8 | 71.10 | 4 | 134217728 / 134217744 | eager 29.045 (6) streams 28.829 (6) lt 29.360 (4) lt2 28.875 (4) cutlass 30.218 (4) | streams | 0.24 | pass |
| mlp2 | 512 | 4.170 / 4.259 | 441.9 | 0.30 | 4 | 2097152 / 2097168 | eager 0.158 (6) streams 0.124 (6) lt 0.158 (4) lt2 0.125 (4) cutlass 0.498 (4) | streams | 0.03 | pass |
| mlp | 1024 | 4.589 / 4.617 | 438.3 | 0.88 | 2 | 4194304 / 4194312 | eager 0.330 (3) lt 0.321 (2) cutlass 0.481 (2) | lt | 0.07 | pass |
| mlp | 2048 | 12.084 / 10.728 | 451.9 | 5.70 | 2 | 16777216 / 16777224 | eager 2.333 (3) lt 2.308 (2) cutlass 2.719 (2) | lt | 0.19 | pass |
| mlp | 4096 | 58.999 / 39.153 | 537.1 | 37.37 | 2 | 67108864 / 67108872 | eager 14.563 (3) lt 14.482 (2) cutlass 15.168 (2) | lt | 0.25 | pass |
| mlp | 512 | 1.863 / 1.873 | 442.9 | 0.15 | 2 | 1048576 / 1048584 | eager 0.080 (3) lt 0.075 (2) cutlass 0.249 (2) | lt | 0.04 | pass |

### RTX 4090 (sm_89)

| probe | n | Mapal iter ms exact / contract | wall ms | kernel ms | launches | bytes in / out | library legs med ms (launches) | best library | best lib / Mapal exact | gate |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| diamond | 1024 | 4.067 / 3.922 | 261.7 | 0.26 | 3 | 0 / 8388608 | eager 0.148 (4) streams 0.146 (4) lt 0.145 (3) cutlass 0.249 (3) | lt | 0.04 | pass |
| diamond | 2048 | 11.620 / 11.242 | 284.6 | 1.59 | 3 | 0 / 33554432 | eager 0.967 (4) streams 0.959 (4) lt 0.949 (3) cutlass 0.947 (3) | cutlass | 0.08 | pass |
| diamond | 4096 | 39.806 / 38.649 | 448.3 | 12.00 | 3 | 0 / 134217728 | eager 8.055 (4) streams 8.028 (4) lt 7.894 (3) cutlass 7.878 (3) | cutlass | 0.20 | pass |
| diamond | 512 | 1.806 / 1.729 | 258.3 | 0.07 | 3 | 0 / 2097152 | eager 0.052 (4) streams 0.052 (4) lt 0.050 (3) cutlass 0.132 (3) | lt | 0.03 | pass |
| fanout | 1024 | 2.175 / 2.071 | 257.2 | 0.18 | 2 | 0 / 4194312 | eager 0.101 (3) streams 0.099 (3) | streams | 0.05 | pass |
| fanout | 2048 | 6.274 / 6.023 | 274.4 | 1.06 | 2 | 0 / 16777224 | eager 0.647 (3) streams 0.640 (3) | streams | 0.10 | pass |
| fanout | 4096 | 23.605 / 21.455 | 370.5 | 8.03 | 2 | 0 / 67108872 | eager 5.193 (3) streams 5.124 (3) | streams | 0.22 | pass |
| fanout | 512 | 0.889 / 0.929 | 252.4 | 0.05 | 2 | 0 / 1048584 | eager 0.037 (3) streams 0.037 (3) | eager | 0.04 | pass |
| indep | 1024 | 3.731 / 3.495 | 277.1 | 0.18 | 2 | 0 / 8388608 | eager 0.101 (3) streams 0.093 (3) lt 0.098 (2) cutlass 0.170 (2) | streams | 0.03 | pass |
| indep | 2048 | 10.611 / 10.732 | 281.8 | 1.06 | 2 | 0 / 33554432 | eager 0.652 (3) streams 0.645 (3) lt 0.639 (2) cutlass 0.638 (2) | cutlass | 0.06 | pass |
| indep | 4096 | 34.480 / 35.499 | 412.1 | 8.02 | 2 | 0 / 134217728 | eager 5.375 (3) streams 5.347 (3) lt 5.210 (2) cutlass 5.193 (2) | cutlass | 0.15 | pass |
| indep | 512 | 1.626 / 1.520 | 263.8 | 0.05 | 2 | 0 / 2097152 | eager 0.039 (3) streams 0.037 (3) lt 0.036 (2) cutlass 0.090 (2) | lt | 0.02 | pass |
| loop | 2048 | 108.529 / 105.036 | 394.6 | 4.19 | 8 | 150994944 / 134217728 | eager 2.730 (16) lt 2.637 (8) cutlass 2.548 (8) | cutlass | 0.02 | pass |
| mlp2 | 1024 | 7.466 / 7.363 | 268.0 | 0.36 | 4 | 8388608 / 8388624 | eager 0.202 (6) streams 0.218 (6) lt 0.192 (4) lt2 0.180 (4) cutlass 0.331 (4) | lt2 | 0.02 | pass |
| mlp2 | 2048 | 21.849 / 21.260 | 297.8 | 2.13 | 4 | 33554432 / 33554448 | eager 1.302 (6) streams 1.281 (6) lt 1.279 (4) lt2 1.251 (4) cutlass 1.256 (4) | lt2 | 0.06 | pass |
| mlp2 | 4096 | 75.764 / 71.706 | 621.9 | 16.05 | 4 | 134217728 / 134217744 | eager 10.714 (6) streams 10.570 (6) lt 10.629 (4) lt2 10.442 (4) cutlass 10.545 (4) | lt2 | 0.14 | pass |
| mlp2 | 512 | 2.950 / 2.977 | 265.0 | 0.10 | 4 | 2097152 / 2097168 | eager 0.072 (6) streams 0.059 (6) lt 0.070 (4) lt2 0.060 (4) cutlass 0.175 (4) | streams | 0.02 | pass |
| mlp | 1024 | 3.345 / 3.337 | 263.3 | 0.18 | 2 | 4194304 / 4194312 | eager 0.100 (3) lt 0.097 (2) cutlass 0.168 (2) | lt | 0.03 | pass |
| mlp | 2048 | 10.323 / 10.115 | 280.0 | 1.06 | 2 | 16777216 / 16777224 | eager 0.649 (3) lt 0.636 (2) cutlass 0.633 (2) | cutlass | 0.06 | pass |
| mlp | 4096 | 35.038 / 34.378 | 374.6 | 8.02 | 2 | 67108864 / 67108872 | eager 5.321 (3) lt 5.200 (2) cutlass 5.205 (2) | lt | 0.15 | pass |
| mlp | 512 | 1.398 / 1.432 | 250.3 | 0.05 | 2 | 1048576 / 1048584 | eager 0.038 (3) lt 0.036 (2) cutlass 0.089 (2) | lt | 0.03 | pass |

### H100 PCIe (sm_90a)

| probe | n | Mapal iter ms exact / contract | wall ms | kernel ms | launches | bytes in / out | library legs med ms (launches) | best library | best lib / Mapal exact | gate |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| diamond | 1024 | 3.153 / 5.682 | 877.3 | 0.60 | 3 | 0 / 8388608 | eager 0.258 (4) streams 0.238 (4) lt 0.253 (3) cutlass 0.375 (3) | streams | 0.08 | pass |
| diamond | 2048 | 10.037 / 9.153 | 875.4 | 3.30 | 3 | 0 / 33554432 | eager 1.579 (4) streams 1.451 (4) lt 1.558 (3) cutlass 2.070 (3) | streams | 0.14 | pass |
| diamond | 4096 | 40.949 / 32.869 | 571.9 | 18.70 | 3 | 0 / 134217728 | eager 11.758 (4) streams 10.977 (4) lt 11.659 (3) cutlass 12.850 (3) | streams | 0.27 | pass |
| diamond | 512 | 1.534 / 1.494 | 465.6 | 0.12 | 3 | 0 / 2097152 | eager 0.064 (4) streams 0.058 (4) lt 0.061 (3) cutlass 0.194 (3) | streams | 0.04 | pass |
| fanout | 1024 | 1.739 / 1.683 | 468.3 | 0.40 | 2 | 0 / 4194312 | eager 0.172 (3) streams 0.171 (3) | streams | 0.10 | pass |
| fanout | 2048 | 4.789 / 4.292 | 473.6 | 2.22 | 2 | 0 / 16777224 | eager 1.047 (3) streams 1.032 (3) | streams | 0.22 | pass |
| fanout | 4096 | 18.792 / 15.980 | 530.3 | 12.59 | 2 | 0 / 67108872 | eager 7.765 (3) streams 7.690 (3) | streams | 0.41 | pass |
| fanout | 512 | 0.780 / 0.793 | 475.6 | 0.08 | 2 | 0 / 1048584 | eager 0.044 (3) streams 0.042 (3) | streams | 0.05 | pass |
| indep | 1024 | 3.097 / 3.034 | 485.0 | 0.40 | 2 | 0 / 8388608 | eager 0.177 (3) streams 0.158 (3) lt 0.171 (2) cutlass 0.251 (2) | streams | 0.05 | pass |
| indep | 2048 | 8.876 / 7.860 | 863.9 | 2.21 | 2 | 0 / 33554432 | eager 1.068 (3) streams 0.945 (3) lt 1.050 (2) cutlass 1.393 (2) | streams | 0.11 | pass |
| indep | 4096 | 35.675 / 29.585 | 840.9 | 12.57 | 2 | 0 / 134217728 | eager 7.808 (3) streams 7.039 (3) lt 7.732 (2) cutlass 8.526 (2) | streams | 0.20 | pass |
| indep | 512 | 1.700 / 1.655 | 535.4 | 0.08 | 2 | 0 / 2097152 | eager 0.045 (3) streams 0.038 (3) lt 0.042 (2) cutlass 0.131 (2) | streams | 0.02 | pass |
| loop | 2048 | 75.623 / 69.702 | 648.9 | 8.90 | 8 | 150994944 / 134217728 | eager 4.213 (16) lt 4.074 (8) cutlass 5.506 (8) | lt | 0.05 | pass |
| mlp2 | 1024 | 5.660 / 5.584 | 538.5 | 0.80 | 4 | 8388608 / 8388624 | eager 0.342 (6) streams 0.308 (6) lt 0.324 (4) lt2 0.286 (4) cutlass 0.499 (4) | lt2 | 0.05 | pass |
| mlp2 | 2048 | 15.222 / 14.932 | 521.5 | 4.45 | 4 | 33554432 / 33554448 | eager 2.124 (6) streams 1.873 (6) lt 2.075 (4) lt2 1.823 (4) cutlass 2.779 (4) | lt2 | 0.12 | pass |
| mlp2 | 4096 | 70.757 / 61.237 | 703.5 | 25.00 | 4 | 134217728 / 134217744 | eager 16.528 (6) streams 14.757 (6) lt 16.296 (4) lt2 14.766 (4) cutlass 17.951 (4) | streams | 0.21 | pass |
| mlp2 | 512 | 3.059 / 2.784 | 919.2 | 0.16 | 4 | 2097152 / 2097168 | eager 0.088 (6) streams 0.072 (6) lt 0.080 (4) lt2 0.064 (4) cutlass 0.261 (4) | lt2 | 0.02 | pass |
| mlp | 1024 | 2.952 / 3.613 | 947.6 | 0.40 | 2 | 4194304 / 4194312 | eager 0.173 (3) lt 0.167 (2) cutlass 0.249 (2) | lt | 0.06 | pass |
| mlp | 2048 | 7.431 / 7.172 | 548.5 | 2.22 | 2 | 16777216 / 16777224 | eager 1.054 (3) lt 1.029 (2) cutlass 1.374 (2) | lt | 0.14 | pass |
| mlp | 4096 | 25.408 / 23.511 | 611.9 | 12.49 | 2 | 67108864 / 67108872 | eager 7.636 (3) lt 7.555 (2) cutlass 8.419 (2) | lt | 0.30 | pass |
| mlp | 512 | 1.211 / 1.174 | 506.6 | 0.07 | 2 | 1048576 / 1048584 | eager 0.044 (3) lt 0.041 (2) cutlass 0.128 (2) | lt | 0.03 | pass |

### RTX 5090 (sm_120)

| probe | n | Mapal iter ms exact / contract | wall ms | kernel ms | launches | bytes in / out | library legs med ms (launches) | best library | best lib / Mapal exact | gate |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| diamond | 1024 | 3.653 / 2.785 | 337.5 | 0.25 | 3 | 0 / 8388608 | eager 0.147 (4) streams 0.137 (4) lt 0.141 (3) cutlass 0.264 (3) | streams | 0.04 | pass |
| diamond | 2048 | 8.540 / 8.259 | 278.2 | 1.52 | 3 | 0 / 33554432 | eager 0.813 (4) streams 0.811 (4) lt 0.805 (3) cutlass 0.915 (3) | lt | 0.09 | pass |
| diamond | 4096 | 40.736 / 36.079 | 505.7 | 10.39 | 3 | 0 / 134217728 | eager 6.751 (4) streams 6.552 (4) lt 6.671 (3) cutlass 7.372 (3) | streams | 0.16 | pass |
| diamond | 512 | 1.277 / 1.304 | 251.1 | 0.07 | 3 | 0 / 2097152 | eager 0.388 (4) streams 0.444 (4) lt 0.370 (3) cutlass 0.982 (3) | lt | 0.29 | pass |
| fanout | 1024 | 6.315 / 2.341 | 495.0 | 0.31 | 2 | 0 / 4194312 | eager 0.098 (3) streams 0.095 (3) | streams | 0.01 | pass |
| fanout | 2048 | 5.449 / 5.125 | 267.0 | 1.02 | 2 | 0 / 16777224 | eager 0.537 (3) streams 0.529 (3) | streams | 0.10 | pass |
| fanout | 4096 | 21.341 / 20.010 | 399.3 | 6.91 | 2 | 0 / 67108872 | eager 4.366 (3) streams 4.333 (3) | streams | 0.20 | pass |
| fanout | 512 | 0.747 / 0.769 | 249.4 | 0.04 | 2 | 0 / 1048584 | eager 0.037 (3) streams 0.035 (3) | streams | 0.05 | pass |
| indep | 1024 | 2.457 / 2.386 | 255.3 | 0.16 | 2 | 0 / 8388608 | eager 0.100 (3) streams 0.088 (3) lt 0.094 (2) cutlass 0.178 (2) | streams | 0.04 | pass |
| indep | 2048 | 7.780 / 7.748 | 273.6 | 1.02 | 2 | 0 / 33554432 | eager 0.541 (3) streams 0.539 (3) lt 0.530 (2) cutlass 0.602 (2) | lt | 0.07 | pass |
| indep | 4096 | 33.674 / 31.751 | 457.2 | 6.91 | 2 | 0 / 134217728 | eager 4.510 (3) streams 4.319 (3) lt 4.441 (2) cutlass 4.913 (2) | streams | 0.13 | pass |
| indep | 512 | 1.553 / 1.462 | 319.9 | 0.06 | 2 | 0 / 2097152 | eager 0.039 (3) streams 0.029 (3) lt 0.035 (2) cutlass 0.102 (2) | streams | 0.02 | pass |
| loop | 2048 | 77.545 / 75.830 | 348.9 | 4.01 | 8 | 150994944 / 134217728 | eager 2.139 (16) lt 2.059 (8) cutlass 2.391 (8) | lt | 0.03 | pass |
| mlp2 | 1024 | 5.249 / 5.303 | 263.5 | 0.33 | 4 | 8388608 / 8388624 | eager 0.199 (6) streams 0.176 (6) lt 0.188 (4) lt2 0.168 (4) cutlass 0.348 (4) | lt2 | 0.03 | pass |
| mlp2 | 2048 | 15.512 / 15.944 | 290.4 | 2.03 | 4 | 33554432 / 33554448 | eager 1.102 (6) streams 1.071 (6) lt 1.083 (4) lt2 1.075 (4) cutlass 1.249 (4) | streams | 0.07 | pass |
| mlp2 | 4096 | 69.288 / 63.756 | 685.9 | 13.72 | 4 | 134217728 / 134217744 | eager 9.021 (6) streams 8.505 (6) lt 8.929 (4) lt2 8.337 (4) cutlass 9.869 (4) | lt2 | 0.12 | pass |
| mlp2 | 512 | 2.025 / 2.122 | 249.3 | 0.09 | 4 | 2097152 / 2097168 | eager 0.070 (6) streams 0.050 (6) lt 0.061 (4) lt2 0.047 (4) cutlass 0.197 (4) | lt2 | 0.02 | pass |
| mlp | 1024 | 2.383 / 2.317 | 249.5 | 0.16 | 2 | 4194304 / 4194312 | eager 0.096 (3) lt 0.094 (2) cutlass 0.178 (2) | lt | 0.04 | pass |
| mlp | 2048 | 7.001 / 6.537 | 268.1 | 1.02 | 2 | 16777216 / 16777224 | eager 0.537 (3) lt 0.526 (2) cutlass 0.598 (2) | lt | 0.08 | pass |
| mlp | 4096 | 26.310 / 25.644 | 376.5 | 6.88 | 2 | 67108864 / 67108872 | eager 4.381 (3) lt 4.342 (2) cutlass 4.783 (2) | lt | 0.17 | pass |
| mlp | 512 | 1.031 / 1.069 | 245.4 | 0.04 | 2 | 1048576 / 1048584 | eager 0.037 (3) lt 0.033 (2) cutlass 0.100 (2) | lt | 0.03 | pass |
