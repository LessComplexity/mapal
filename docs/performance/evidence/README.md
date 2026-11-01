# Evidence pass

## fusion-natural-forms sweep: the four parts on hardware

Raw: [`benches/results/evidence/fusion-natural-forms/`](../../../benches/results/evidence/fusion-natural-forms/)
(`<arch>/` main boxes, `confirm/<arch>/` a second host, `attn-confirm/<arch>/` the attention re-take,
`confirm-throttled/<arch>/` boxes that ran slow) · tables: `python3 benches/gpu/fusion_natural_report.py
benches/results/evidence/fusion-natural-forms`, `python3 benches/gpu/evidence_report.py
benches/results/evidence/fusion-natural-forms` · log: [session](../../sessions/2026-10-24-fusion-natural-forms.md).
Boxes: `PROFILE=1 E2E=0 TIMING=0 PROBE_REPS=9`, 30 reps; arms interleaved ABBA, one process each; null = max over
reps of |null / shipped − 1|; default face, value-gated first.

### Decisions

| rung | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 | evidence: OFF/ON iter, main box (confirm box) | explained by |
|---|---|---|---|---|---|---|
| `flow` | **OFF** (row) | **OFF** (row) | **OFF** (row) | **OFF** (row) | ON slower outside null on 4 / 5 / 4 / 6 of 6 programs (second host 6 / 4 / 6 / 6), faster on none; softmax_rows_4096 0.884 / 0.861 / 0.849 / 0.914, attn_2048_128 0.409 / 0.410 / 0.412 / 0.480 | the unit kernel is 1.79–4.82× the four kernels it replaces: one thread per row (1024–4096 threads) walks the T-cell members serially and evaluates `ex` twice; the OFF arm's cell kernels run T² threads |
| `window` | ON | ON | ON | ON | conv_2048_5 23.2 / 31.6 / 9.7 / 11.8 (31.5 / 42.0 / 17.7 / 11.5), ewconvew_2048_5 34.4 / 34.2 / 13.0 / 14.2 (47.9 / 44.3 / 19.9 / 17.8) | OFF runs the 5×5 conv on the host (0 launches); ON one naive kernel, 25 FFMA, 0 div/rem |
| `range-map` | ON | ON | ON | ON | decode_1024_256 3.96 / 4.39 / 2.50 / 4.15 (4.24 / 4.04 / 2.35 / 4.11) | the traced kernel ms within 8 % ON / OFF (4090 44.6 / 42.2); OFF's host map writes 262144 cells per trip, ON one 256-cell row |
| `lane-stride` (2-D `q·Kᵀ`) | ON | ON | ON | ON | attention 106–693 on every part (110–436) | OFF computes `q·Kᵀ` on the host behind 4 / 16 MB copies |

`GPU_RUNGS_OFF` rows: `crates/backends/llvm/src/profile.rs` (flow × 4). End check: the build with the rows emits, for all
25 suite programs on all four parts, exactly the measured build — fused programs = their `.off-fuse` arm, the rest =
shipped (host IR, PTX; [`endcheck-fuse-off.txt`](../../../benches/results/evidence/fusion-natural-forms/endcheck-fuse-off.txt)).

### Boxes

| arch | box | host CPU | GPU power limit, driver | instance | value gates pass / fail | intent checks (checks / mismatch) | library legs pass / FAIL | torch legs within / run | clock: worst kernel max/min (programs) |
|---|---|---|---|---|---|---|---|---|---|
| h100-pcie | attn-confirm | AMD EPYC 9354 32-Core Processor | 350.00 W, 595.71.05 | 57544627 | 4 / 0 | 3779 / 112 | 8 / 0 | 8 / 8 | - |
| rtx-5090 | attn-confirm | AMD EPYC 7543 32-Core Processor | 600.00 W, 595.71.05 | 57544628 | 4 / 0 | 3203 / 59 | 8 / 0 | 8 / 8 | - |
| rtx-3090 | confirm-throttled | AMD EPYC 7B12 64-Core Processor | 250.00 W, 580.126.09 | 57548252 | 25 / 0 | 3978 / 122 | 41 / 0 | 50 / 50 | 1.00 (5) |
| rtx-5090 | confirm-throttled | AMD Ryzen 9 7945HX with Radeon Graphics | 546.00 W, 590.48.01 | 57547762 | 25 / 0 | 3578 / 75 | 41 / 0 | 50 / 50 | 3.32 (5) |
| h100-pcie | confirm | AMD EPYC 7713 64-Core Processor | 350.00 W, 595.71.05 | 57553934 | 25 / 0 | 4143 / 112 | 41 / 0 | 50 / 50 | 1.01 (5) |
| rtx-3090 | confirm | AMD EPYC 7702 64-Core Processor | 350.00 W, 580.95.05 | 57551147 | 25 / 0 | 3978 / 122 | 41 / 0 | 50 / 50 | 1.01 (5) |
| rtx-4090 | confirm | AMD Ryzen 9 5950X 16-Core Processor | 375.00 W, 575.57.08 | 57549653 | 25 / 0 | 3269 / 26 | 41 / 0 | 50 / 50 | 1.02 (4) |
| rtx-5090 | confirm | AMD Ryzen 9 7950X 16-Core Processor | 575.00 W, 610.57.04 | 57549771 | 25 / 0 | 3578 / 76 | 41 / 0 | 50 / 50 | 1.29 (5) |
| h100-pcie | main | AMD EPYC 9354 32-Core Processor | 350.00 W, 595.71.05 | 57541723 | 25 / 0 | 4143 / 112 | 33 / 8 | 50 / 50 | 1.02 (5) |
| rtx-3090 | main | AMD EPYC 7K62 48-Core Processor | 220.00 W, 580.159.03 | 57544369 | 25 / 0 | 3978 / 122 | 41 / 0 | 0 / 0 | 1.01 (5) |
| rtx-4090 | main | AMD EPYC 7532 32-Core Processor | 450.00 W, 580.159.03 | 57544551 | 25 / 0 | 3269 / 26 | 41 / 0 | 50 / 50 | 1.02 (4) |
| rtx-5090 | main | AMD EPYC 7K62 48-Core Processor | 450.00 W, 580.173.02 | 57541724 | 25 / 0 | 3578 / 75 | 33 / 6 | 50 / 50 | 1.04 (5) |

### Each new rung ON vs its OFF arm, per chain (launches and kernel ms: traced rep 1; device buffers: static facts)

| program | arch | box | arm | launches ON / OFF | device buffers ON / OFF | kernel ms ON / OFF | iter ms ON / OFF | OFF/ON | null | sign |
|---|---|---|---|---|---|---|---|---|---|---|
| attn_1024_128 | rtx-4090 | main | off-fuse | 3 / 6 | 6 / 9 | 0.372 / 0.152 | 0.4452 / 0.2648 | 0.597 | 0.080 | ON slower |
| attn_1024_128 | rtx-4090 | main | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.372 / 0.140 | 0.4452 / 0.2706 | 0.614 | 0.080 | ON slower |
| attn_1024_128 | rtx-4090 | main | off-lane-stride | 3 / 1 | 6 / 3 | 0.372 / 0.035 | 0.4452 / 143.86 | 322.914 | 0.080 | ON faster |
| attn_1024_64 | rtx-4090 | main | off-fuse | 3 / 6 | 6 / 9 | 0.232 / 0.127 | 0.2952 / 0.2839 | 0.971 | 0.168 | inside null |
| attn_1024_64 | rtx-4090 | main | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.232 / 0.130 | 0.2952 / 0.2852 | 0.957 | 0.168 | inside null |
| attn_1024_64 | rtx-4090 | main | off-lane-stride | 3 / 1 | 6 / 3 | 0.232 / 0.038 | 0.2952 / 80.78 | 268.551 | 0.168 | ON faster |
| attn_2048_128 | rtx-4090 | main | off-fuse | 3 / 6 | 6 / 9 | 0.708 / 0.221 | 0.8427 / 0.3528 | 0.409 | 0.060 | ON slower |
| attn_2048_128 | rtx-4090 | main | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.708 / 0.229 | 0.8427 / 0.3511 | 0.417 | 0.060 | ON slower |
| attn_2048_128 | rtx-4090 | main | off-lane-stride | 3 / 1 | 6 / 3 | 0.708 / 0.065 | 0.8427 / 464.11 | 550.666 | 0.060 | ON faster |
| attn_2048_64 | rtx-4090 | main | off-fuse | 3 / 6 | 6 / 9 | 0.410 / 0.196 | 0.5561 / 0.4186 | 0.745 | 0.148 | inside null |
| attn_2048_64 | rtx-4090 | main | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.410 / 0.203 | 0.5561 / 0.3698 | 0.659 | 0.148 | inside null |
| attn_2048_64 | rtx-4090 | main | off-lane-stride | 3 / 1 | 6 / 3 | 0.410 / 0.051 | 0.5561 / 219.52 | 393.968 | 0.148 | ON faster |
| conv_2048_5 | rtx-4090 | main | off-fuse+window+range-map | 1 / 0 | 3 / 0 | 0.052 / 0.000 | 0.0762 / 2.0945 | 30.094 | 0.107 | ON faster |
| conv_2048_5 | rtx-4090 | main | off-window | 1 / 0 | 3 / 0 | 0.052 / 0.000 | 0.0762 / 1.7443 | 23.243 | 0.107 | ON faster |
| decode_1024_256 | rtx-4090 | main | off-fuse+window+range-map | 4096 / 4096 | 8 / 8 | 44.604 / 42.970 | 60.08 / 239.59 | 4.045 | 0.137 | ON faster |
| decode_1024_256 | rtx-4090 | main | off-range-map | 4096 / 4096 | 8 / 8 | 44.604 / 42.154 | 60.08 / 240.05 | 3.955 | 0.137 | ON faster |
| ewconvew_2048_5 | rtx-4090 | main | off-fuse+window+range-map | 1 / 0 | 3 / 0 | 0.054 / 0.000 | 0.1012 / 3.3607 | 33.744 | 0.120 | ON faster |
| ewconvew_2048_5 | rtx-4090 | main | off-window | 1 / 0 | 3 / 0 | 0.054 / 0.000 | 0.1012 / 3.4359 | 34.403 | 0.120 | ON faster |
| softmax_rows_2048 | rtx-4090 | main | off-fuse | 2 / 5 | 4 / 7 | 0.641 / 0.433 | 0.7181 / 0.4882 | 0.693 | 0.045 | ON slower |
| softmax_rows_2048 | rtx-4090 | main | off-fuse+window+range-map | 2 / 5 | 4 / 7 | 0.641 / 0.430 | 0.7181 / 0.4880 | 0.681 | 0.045 | ON slower |
| softmax_rows_4096 | rtx-4090 | main | off-fuse | 2 / 5 | 4 / 7 | 3.202 / 2.838 | 3.2960 / 2.9111 | 0.884 | 0.019 | ON slower |
| softmax_rows_4096 | rtx-4090 | main | off-fuse+window+range-map | 2 / 5 | 4 / 7 | 3.202 / 2.850 | 3.2960 / 2.9138 | 0.885 | 0.019 | ON slower |
| attn_1024_128 | rtx-4090 | confirm | off-fuse | 3 / 6 | 6 / 9 | 0.365 / 0.122 | 0.3927 / 0.1624 | 0.416 | 0.030 | ON slower |
| attn_1024_128 | rtx-4090 | confirm | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.365 / 0.119 | 0.3927 / 0.1621 | 0.424 | 0.030 | ON slower |
| attn_1024_128 | rtx-4090 | confirm | off-lane-stride | 3 / 1 | 6 / 3 | 0.365 / 0.075 | 0.3927 / 78.39 | 199.039 | 0.030 | ON faster |
| attn_1024_64 | rtx-4090 | confirm | off-fuse | 3 / 6 | 6 / 9 | 0.218 / 0.107 | 0.2537 / 0.1555 | 0.612 | 0.039 | ON slower |
| attn_1024_64 | rtx-4090 | confirm | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.218 / 0.108 | 0.2537 / 0.1555 | 0.627 | 0.039 | ON slower |
| attn_1024_64 | rtx-4090 | confirm | off-lane-stride | 3 / 1 | 6 / 3 | 0.218 / 0.078 | 0.2537 / 34.81 | 136.874 | 0.039 | ON faster |
| attn_2048_128 | rtx-4090 | confirm | off-fuse | 3 / 6 | 6 / 9 | 0.708 / 0.212 | 0.7845 / 0.2823 | 0.355 | 0.061 | ON slower |
| attn_2048_128 | rtx-4090 | confirm | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.708 / 0.217 | 0.7845 / 0.2865 | 0.354 | 0.061 | ON slower |
| attn_2048_128 | rtx-4090 | confirm | off-lane-stride | 3 / 1 | 6 / 3 | 0.708 / 0.077 | 0.7845 / 307.21 | 391.622 | 0.061 | ON faster |
| attn_2048_64 | rtx-4090 | confirm | off-fuse | 3 / 6 | 6 / 9 | 0.413 / 0.185 | 0.4775 / 0.2525 | 0.532 | 0.039 | ON slower |
| attn_2048_64 | rtx-4090 | confirm | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.413 / 0.183 | 0.4775 / 0.2458 | 0.515 | 0.039 | ON slower |
| attn_2048_64 | rtx-4090 | confirm | off-lane-stride | 3 / 1 | 6 / 3 | 0.413 / 0.074 | 0.4775 / 137.30 | 286.855 | 0.039 | ON faster |
| conv_2048_5 | rtx-4090 | confirm | off-fuse+window+range-map | 1 / 0 | 3 / 0 | 0.050 / 0.000 | 0.0606 / 1.8065 | 29.817 | 0.162 | ON faster |
| conv_2048_5 | rtx-4090 | confirm | off-window | 1 / 0 | 3 / 0 | 0.050 / 0.000 | 0.0606 / 1.8242 | 31.530 | 0.162 | ON faster |
| decode_1024_256 | rtx-4090 | confirm | off-fuse+window+range-map | 4096 / 4096 | 8 / 8 | 34.934 / 35.633 | 37.35 / 170.65 | 4.579 | 0.017 | ON faster |
| decode_1024_256 | rtx-4090 | confirm | off-range-map | 4096 / 4096 | 8 / 8 | 34.934 / 35.375 | 37.35 / 157.97 | 4.242 | 0.017 | ON faster |
| ewconvew_2048_5 | rtx-4090 | confirm | off-fuse+window+range-map | 1 / 0 | 3 / 0 | 0.051 / 0.000 | 0.0769 / 3.4944 | 46.026 | 0.146 | ON faster |
| ewconvew_2048_5 | rtx-4090 | confirm | off-window | 1 / 0 | 3 / 0 | 0.051 / 0.000 | 0.0769 / 3.5525 | 47.933 | 0.146 | ON faster |
| softmax_rows_2048 | rtx-4090 | confirm | off-fuse | 2 / 5 | 4 / 7 | 0.648 / 0.415 | 0.6647 / 0.4478 | 0.674 | 0.105 | ON slower |
| softmax_rows_2048 | rtx-4090 | confirm | off-fuse+window+range-map | 2 / 5 | 4 / 7 | 0.648 / 0.414 | 0.6647 / 0.4480 | 0.668 | 0.105 | ON slower |
| softmax_rows_4096 | rtx-4090 | confirm | off-fuse | 2 / 5 | 4 / 7 | 3.307 / 2.933 | 3.3505 / 2.9562 | 0.884 | 0.017 | ON slower |
| softmax_rows_4096 | rtx-4090 | confirm | off-fuse+window+range-map | 2 / 5 | 4 / 7 | 3.307 / 2.933 | 3.3505 / 2.9536 | 0.882 | 0.017 | ON slower |
| attn_1024_128 | rtx-5090 | main | off-fuse | 3 / 6 | 6 / 9 | 0.315 / 0.119 | 0.3497 / 0.1977 | 0.562 | 0.055 | ON slower |
| attn_1024_128 | rtx-5090 | main | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.315 / 0.117 | 0.3497 / 0.1970 | 0.573 | 0.055 | ON slower |
| attn_1024_128 | rtx-5090 | main | off-lane-stride | 3 / 1 | 6 / 3 | 0.315 / 0.035 | 0.3497 / 140.60 | 404.539 | 0.055 | ON faster |
| attn_1024_64 | rtx-5090 | main | off-fuse | 3 / 6 | 6 / 9 | 0.192 / 0.111 | 0.2321 / 0.1967 | 0.840 | 0.101 | inside null |
| attn_1024_64 | rtx-5090 | main | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.192 / 0.114 | 0.2321 / 0.2063 | 0.881 | 0.101 | inside null |
| attn_1024_64 | rtx-5090 | main | off-lane-stride | 3 / 1 | 6 / 3 | 0.192 / 0.034 | 0.2321 / 77.96 | 334.998 | 0.101 | ON faster |
| attn_2048_128 | rtx-5090 | main | off-fuse | 3 / 6 | 6 / 9 | 0.625 / 0.206 | 0.6694 / 0.2757 | 0.410 | 0.049 | ON slower |
| attn_2048_128 | rtx-5090 | main | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.625 / 0.205 | 0.6694 / 0.2849 | 0.436 | 0.049 | ON slower |
| attn_2048_128 | rtx-5090 | main | off-lane-stride | 3 / 1 | 6 / 3 | 0.625 / 0.058 | 0.6694 / 462.67 | 692.980 | 0.049 | ON faster |
| attn_2048_64 | rtx-5090 | main | off-fuse | 3 / 6 | 6 / 9 | 0.366 / 0.182 | 0.4272 / 0.2405 | 0.566 | 0.070 | ON slower |
| attn_2048_64 | rtx-5090 | main | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.366 / 0.182 | 0.4272 / 0.2408 | 0.562 | 0.070 | ON slower |
| attn_2048_64 | rtx-5090 | main | off-lane-stride | 3 / 1 | 6 / 3 | 0.366 / 0.052 | 0.4272 / 217.46 | 505.849 | 0.070 | ON faster |
| conv_2048_5 | rtx-5090 | main | off-fuse+window+range-map | 1 / 0 | 3 / 0 | 0.038 / 0.000 | 0.0559 / 1.7082 | 30.319 | 0.156 | ON faster |
| conv_2048_5 | rtx-5090 | main | off-window | 1 / 0 | 3 / 0 | 0.038 / 0.000 | 0.0559 / 1.7471 | 31.561 | 0.156 | ON faster |
| decode_1024_256 | rtx-5090 | main | off-fuse+window+range-map | 4096 / 4096 | 8 / 8 | 46.265 / 43.472 | 53.89 / 235.83 | 4.357 | 0.236 | ON faster |
| decode_1024_256 | rtx-5090 | main | off-range-map | 4096 / 4096 | 8 / 8 | 46.265 / 42.723 | 53.89 / 234.23 | 4.390 | 0.236 | ON faster |
| ewconvew_2048_5 | rtx-5090 | main | off-fuse+window+range-map | 1 / 0 | 3 / 0 | 0.037 / 0.000 | 0.0815 / 2.5980 | 31.529 | 0.203 | ON faster |
| ewconvew_2048_5 | rtx-5090 | main | off-window | 1 / 0 | 3 / 0 | 0.037 / 0.000 | 0.0815 / 2.7365 | 34.197 | 0.203 | ON faster |
| softmax_rows_2048 | rtx-5090 | main | off-fuse | 2 / 5 | 4 / 7 | 0.576 / 0.388 | 0.6239 / 0.4193 | 0.674 | 0.031 | ON slower |
| softmax_rows_2048 | rtx-5090 | main | off-fuse+window+range-map | 2 / 5 | 4 / 7 | 0.576 / 0.390 | 0.6239 / 0.4246 | 0.674 | 0.031 | ON slower |
| softmax_rows_4096 | rtx-5090 | main | off-fuse | 2 / 5 | 4 / 7 | 3.006 / 2.599 | 3.0516 / 2.6263 | 0.861 | 0.007 | ON slower |
| softmax_rows_4096 | rtx-5090 | main | off-fuse+window+range-map | 2 / 5 | 4 / 7 | 3.006 / 2.598 | 3.0516 / 2.6205 | 0.860 | 0.007 | ON slower |
| attn_1024_128 | rtx-5090 | confirm | off-fuse | 3 / 6 | 6 / 9 | 0.319 / 0.107 | 0.3358 / 0.1174 | 0.348 | 0.017 | ON slower |
| attn_1024_128 | rtx-5090 | confirm | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.319 / 0.105 | 0.3358 / 0.1159 | 0.345 | 0.017 | ON slower |
| attn_1024_128 | rtx-5090 | confirm | off-lane-stride | 3 / 1 | 6 / 3 | 0.319 / 0.036 | 0.3358 / 75.49 | 224.771 | 0.017 | ON faster |
| attn_1024_64 | rtx-5090 | confirm | off-fuse | 3 / 6 | 6 / 9 | 0.187 / 0.097 | 0.2043 / 0.1093 | 0.523 | 0.130 | ON slower |
| attn_1024_64 | rtx-5090 | confirm | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.187 / 0.095 | 0.2043 / 0.1006 | 0.504 | 0.130 | ON slower |
| attn_1024_64 | rtx-5090 | confirm | off-lane-stride | 3 / 1 | 6 / 3 | 0.187 / 0.035 | 0.2043 / 35.41 | 172.425 | 0.130 | ON faster |
| attn_2048_128 | rtx-5090 | confirm | off-fuse | 3 / 6 | 6 / 9 | 0.649 / 0.202 | 0.6936 / 0.2388 | 0.345 | 3.560 | inside null |
| attn_2048_128 | rtx-5090 | confirm | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.649 / 0.205 | 0.6936 / 0.2413 | 0.347 | 3.560 | inside null |
| attn_2048_128 | rtx-5090 | confirm | off-lane-stride | 3 / 1 | 6 / 3 | 0.649 / 0.055 | 0.6936 / 302.14 | 436.083 | 3.560 | ON faster |
| attn_2048_64 | rtx-5090 | confirm | off-fuse | 3 / 6 | 6 / 9 | 0.370 / 0.261 | 0.4114 / 0.2051 | 0.501 | 0.030 | ON slower |
| attn_2048_64 | rtx-5090 | confirm | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.370 / 0.175 | 0.4114 / 0.2060 | 0.500 | 0.030 | ON slower |
| attn_2048_64 | rtx-5090 | confirm | off-lane-stride | 3 / 1 | 6 / 3 | 0.370 / 0.054 | 0.4114 / 136.94 | 334.731 | 0.030 | ON faster |
| conv_2048_5 | rtx-5090 | confirm | off-fuse+window+range-map | 1 / 0 | 3 / 0 | 0.036 / 0.000 | 0.0447 / 1.8498 | 38.555 | 0.257 | ON faster |
| conv_2048_5 | rtx-5090 | confirm | off-window | 1 / 0 | 3 / 0 | 0.036 / 0.000 | 0.0447 / 1.8575 | 41.971 | 0.257 | ON faster |
| decode_1024_256 | rtx-5090 | confirm | off-fuse+window+range-map | 4096 / 4096 | 8 / 8 | 38.837 / 35.437 | 40.63 / 164.07 | 4.038 | 0.056 | ON faster |
| decode_1024_256 | rtx-5090 | confirm | off-range-map | 4096 / 4096 | 8 / 8 | 38.837 / 35.566 | 40.63 / 163.97 | 4.042 | 0.056 | ON faster |
| ewconvew_2048_5 | rtx-5090 | confirm | off-fuse+window+range-map | 1 / 0 | 3 / 0 | 0.036 / 0.000 | 0.0589 / 2.7403 | 44.983 | 0.125 | ON faster |
| ewconvew_2048_5 | rtx-5090 | confirm | off-window | 1 / 0 | 3 / 0 | 0.036 / 0.000 | 0.0589 / 2.6053 | 44.254 | 0.125 | ON faster |
| softmax_rows_2048 | rtx-5090 | confirm | off-fuse | 2 / 5 | 4 / 7 | 0.604 / 0.438 | 0.6328 / 0.4174 | 0.663 | 0.046 | ON slower |
| softmax_rows_2048 | rtx-5090 | confirm | off-fuse+window+range-map | 2 / 5 | 4 / 7 | 0.604 / 0.418 | 0.6328 / 0.4215 | 0.665 | 0.046 | ON slower |
| softmax_rows_4096 | rtx-5090 | confirm | off-fuse | 2 / 5 | 4 / 7 | 2.803 / 2.355 | 2.8255 / 2.9204 | 0.864 | 0.202 | inside null |
| softmax_rows_4096 | rtx-5090 | confirm | off-fuse+window+range-map | 2 / 5 | 4 / 7 | 2.803 / 2.375 | 2.8255 / 2.9250 | 0.865 | 0.202 | inside null |
| attn_1024_128 | h100-pcie | main | off-fuse | 3 / 6 | 6 / 9 | 0.435 / 0.158 | 0.5547 / 0.3162 | 0.527 | 0.187 | inside null |
| attn_1024_128 | h100-pcie | main | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.435 / 0.160 | 0.5547 / 0.2120 | 0.377 | 0.187 | ON slower |
| attn_1024_128 | h100-pcie | main | off-lane-stride | 3 / 1 | 6 / 3 | 0.435 / 0.036 | 0.5547 / 88.85 | 159.848 | 0.187 | ON faster |
| attn_1024_64 | h100-pcie | main | off-fuse | 3 / 6 | 6 / 9 | 0.284 / 0.150 | 0.3804 / 0.1804 | 0.536 | 0.285 | inside null |
| attn_1024_64 | h100-pcie | main | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.284 / 0.136 | 0.3804 / 0.1740 | 0.520 | 0.285 | inside null |
| attn_1024_64 | h100-pcie | main | off-lane-stride | 3 / 1 | 6 / 3 | 0.284 / 0.027 | 0.3804 / 40.17 | 105.621 | 0.285 | ON faster |
| attn_2048_128 | h100-pcie | main | off-fuse | 3 / 6 | 6 / 9 | 0.954 / 0.359 | 1.0186 / 0.4192 | 0.412 | 0.020 | ON slower |
| attn_2048_128 | h100-pcie | main | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.954 / 0.358 | 1.0186 / 0.4203 | 0.410 | 0.020 | ON slower |
| attn_2048_128 | h100-pcie | main | off-lane-stride | 3 / 1 | 6 / 3 | 0.954 / 0.080 | 1.0186 / 354.97 | 346.330 | 0.020 | ON faster |
| attn_2048_64 | h100-pcie | main | off-fuse | 3 / 6 | 6 / 9 | 0.621 / 0.301 | 0.6799 / 0.3635 | 0.537 | 0.127 | ON slower |
| attn_2048_64 | h100-pcie | main | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.621 / 0.297 | 0.6799 / 0.3623 | 0.528 | 0.127 | ON slower |
| attn_2048_64 | h100-pcie | main | off-lane-stride | 3 / 1 | 6 / 3 | 0.621 / 0.055 | 0.6799 / 161.04 | 236.817 | 0.127 | ON faster |
| conv_2048_5 | h100-pcie | main | off-fuse+window+range-map | 1 / 0 | 3 / 0 | 0.054 / 0.000 | 0.0687 / 0.7508 | 11.025 | 0.137 | ON faster |
| conv_2048_5 | h100-pcie | main | off-window | 1 / 0 | 3 / 0 | 0.054 / 0.000 | 0.0687 / 0.6664 | 9.695 | 0.137 | ON faster |
| decode_1024_256 | h100-pcie | main | off-fuse+window+range-map | 5120 / 5120 | 9 / 9 | 65.992 / 65.708 | 73.05 / 182.85 | 2.499 | 0.030 | ON faster |
| decode_1024_256 | h100-pcie | main | off-range-map | 5120 / 5120 | 9 / 9 | 65.992 / 66.307 | 73.05 / 182.78 | 2.502 | 0.030 | ON faster |
| ewconvew_2048_5 | h100-pcie | main | off-fuse+window+range-map | 1 / 0 | 3 / 0 | 0.059 / 0.000 | 0.0855 / 1.1729 | 13.740 | 0.376 | ON faster |
| ewconvew_2048_5 | h100-pcie | main | off-window | 1 / 0 | 3 / 0 | 0.059 / 0.000 | 0.0855 / 1.1226 | 13.028 | 0.376 | ON faster |
| softmax_rows_2048 | h100-pcie | main | off-fuse | 2 / 5 | 4 / 7 | 1.152 / 0.816 | 1.1974 / 0.8456 | 0.704 | 0.043 | ON slower |
| softmax_rows_2048 | h100-pcie | main | off-fuse+window+range-map | 2 / 5 | 4 / 7 | 1.152 / 0.830 | 1.1974 / 0.8816 | 0.725 | 0.043 | ON slower |
| softmax_rows_4096 | h100-pcie | main | off-fuse | 2 / 5 | 4 / 7 | 4.922 / 4.179 | 4.9445 / 4.2031 | 0.849 | 0.009 | ON slower |
| softmax_rows_4096 | h100-pcie | main | off-fuse+window+range-map | 2 / 5 | 4 / 7 | 4.922 / 4.179 | 4.9445 / 4.2036 | 0.850 | 0.009 | ON slower |
| attn_1024_128 | h100-pcie | confirm | off-fuse | 3 / 6 | 6 / 9 | 0.544 / 0.210 | 0.6055 / 0.2865 | 0.482 | 0.128 | ON slower |
| attn_1024_128 | h100-pcie | confirm | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.544 / 0.202 | 0.6055 / 0.3783 | 0.625 | 0.128 | ON slower |
| attn_1024_128 | h100-pcie | confirm | off-lane-stride | 3 / 1 | 6 / 3 | 0.544 / 0.228 | 0.6055 / 98.18 | 162.720 | 0.128 | ON faster |
| attn_1024_64 | h100-pcie | confirm | off-fuse | 3 / 6 | 6 / 9 | 0.350 / 0.177 | 0.3924 / 0.2042 | 0.515 | 0.179 | ON slower |
| attn_1024_64 | h100-pcie | confirm | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.350 / 0.165 | 0.3924 / 0.2221 | 0.559 | 0.179 | inside null |
| attn_1024_64 | h100-pcie | confirm | off-lane-stride | 3 / 1 | 6 / 3 | 0.350 / 0.036 | 0.3924 / 43.51 | 110.205 | 0.179 | ON faster |
| attn_2048_128 | h100-pcie | confirm | off-fuse | 3 / 6 | 6 / 9 | 1.175 / 0.428 | 1.2608 / 0.5542 | 0.422 | 0.033 | ON slower |
| attn_2048_128 | h100-pcie | confirm | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 1.175 / 0.429 | 1.2608 / 0.5190 | 0.408 | 0.033 | ON slower |
| attn_2048_128 | h100-pcie | confirm | off-lane-stride | 3 / 1 | 6 / 3 | 1.175 / 0.103 | 1.2608 / 392.45 | 313.240 | 0.033 | ON faster |
| attn_2048_64 | h100-pcie | confirm | off-fuse | 3 / 6 | 6 / 9 | 0.762 / 0.362 | 0.8624 / 0.4817 | 0.534 | 0.107 | ON slower |
| attn_2048_64 | h100-pcie | confirm | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.762 / 0.367 | 0.8624 / 0.4462 | 0.526 | 0.107 | ON slower |
| attn_2048_64 | h100-pcie | confirm | off-lane-stride | 3 / 1 | 6 / 3 | 0.762 / 0.073 | 0.8624 / 173.33 | 201.531 | 0.107 | ON faster |
| conv_2048_5 | h100-pcie | confirm | off-fuse+window+range-map | 1 / 0 | 3 / 0 | 0.069 / 0.000 | 0.0826 / 1.6730 | 19.691 | 0.210 | ON faster |
| conv_2048_5 | h100-pcie | confirm | off-window | 1 / 0 | 3 / 0 | 0.069 / 0.000 | 0.0826 / 1.4447 | 17.746 | 0.210 | ON faster |
| decode_1024_256 | h100-pcie | confirm | off-fuse+window+range-map | 5120 / 5120 | 9 / 9 | 82.964 / 77.125 | 82.97 / 201.07 | 2.372 | 0.079 | ON faster |
| decode_1024_256 | h100-pcie | confirm | off-range-map | 5120 / 5120 | 9 / 9 | 82.964 / 77.098 | 82.97 / 195.76 | 2.348 | 0.079 | ON faster |
| ewconvew_2048_5 | h100-pcie | confirm | off-fuse+window+range-map | 1 / 0 | 3 / 0 | 0.069 / 0.000 | 0.1264 / 2.4112 | 21.547 | 0.306 | ON faster |
| ewconvew_2048_5 | h100-pcie | confirm | off-window | 1 / 0 | 3 / 0 | 0.069 / 0.000 | 0.1264 / 2.4187 | 19.919 | 0.306 | ON faster |
| softmax_rows_2048 | h100-pcie | confirm | off-fuse | 2 / 5 | 4 / 7 | 1.402 / 0.994 | 1.4585 / 1.0390 | 0.716 | 0.039 | ON slower |
| softmax_rows_2048 | h100-pcie | confirm | off-fuse+window+range-map | 2 / 5 | 4 / 7 | 1.402 / 1.002 | 1.4585 / 1.0519 | 0.721 | 0.039 | ON slower |
| softmax_rows_4096 | h100-pcie | confirm | off-fuse | 2 / 5 | 4 / 7 | 6.059 / 5.148 | 6.1262 / 5.1764 | 0.846 | 0.008 | ON slower |
| softmax_rows_4096 | h100-pcie | confirm | off-fuse+window+range-map | 2 / 5 | 4 / 7 | 6.059 / 5.147 | 6.1262 / 5.1829 | 0.846 | 0.008 | ON slower |
| attn_1024_128 | rtx-3090 | main | off-fuse | 3 / 6 | 6 / 9 | 0.581 / 0.217 | 0.6219 / 0.2851 | 0.460 | 0.053 | ON slower |
| attn_1024_128 | rtx-3090 | main | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.581 / 0.220 | 0.6219 / 0.2716 | 0.438 | 0.053 | ON slower |
| attn_1024_128 | rtx-3090 | main | off-lane-stride | 3 / 1 | 6 / 3 | 0.581 / 0.074 | 0.6219 / 139.34 | 224.735 | 0.053 | ON faster |
| attn_1024_64 | rtx-3090 | main | off-fuse | 3 / 6 | 6 / 9 | 0.353 / 0.203 | 0.4072 / 0.2660 | 0.636 | 0.078 | ON slower |
| attn_1024_64 | rtx-3090 | main | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.353 / 0.196 | 0.4072 / 0.2647 | 0.652 | 0.078 | ON slower |
| attn_1024_64 | rtx-3090 | main | off-lane-stride | 3 / 1 | 6 / 3 | 0.353 / 0.068 | 0.4072 / 78.13 | 191.727 | 0.078 | ON faster |
| attn_2048_128 | rtx-3090 | main | off-fuse | 3 / 6 | 6 / 9 | 1.174 / 0.532 | 1.2656 / 0.6131 | 0.480 | 0.050 | ON slower |
| attn_2048_128 | rtx-3090 | main | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 1.174 / 0.549 | 1.2656 / 0.6108 | 0.482 | 0.050 | ON slower |
| attn_2048_128 | rtx-3090 | main | off-lane-stride | 3 / 1 | 6 / 3 | 1.174 / 0.136 | 1.2656 / 466.70 | 369.487 | 0.050 | ON faster |
| attn_2048_64 | rtx-3090 | main | off-fuse | 3 / 6 | 6 / 9 | 0.779 / 0.431 | 0.8493 / 0.4943 | 0.570 | 0.064 | ON slower |
| attn_2048_64 | rtx-3090 | main | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.779 / 0.430 | 0.8493 / 0.4885 | 0.576 | 0.064 | ON slower |
| attn_2048_64 | rtx-3090 | main | off-lane-stride | 3 / 1 | 6 / 3 | 0.779 / 0.107 | 0.8493 / 221.13 | 261.264 | 0.064 | ON faster |
| conv_2048_5 | rtx-3090 | main | off-fuse+window+range-map | 1 / 0 | 3 / 0 | 0.111 / 0.000 | 0.1317 / 1.6167 | 12.322 | 0.109 | ON faster |
| conv_2048_5 | rtx-3090 | main | off-window | 1 / 0 | 3 / 0 | 0.111 / 0.000 | 0.1317 / 1.5850 | 11.814 | 0.109 | ON faster |
| decode_1024_256 | rtx-3090 | main | off-fuse+window+range-map | 4096 / 4096 | 8 / 8 | 48.899 / 48.628 | 58.96 / 246.34 | 4.178 | 0.107 | ON faster |
| decode_1024_256 | rtx-3090 | main | off-range-map | 4096 / 4096 | 8 / 8 | 48.899 / 48.665 | 58.96 / 245.04 | 4.146 | 0.107 | ON faster |
| ewconvew_2048_5 | rtx-3090 | main | off-fuse+window+range-map | 1 / 0 | 3 / 0 | 0.110 / 0.000 | 0.1585 / 2.3692 | 14.627 | 0.134 | ON faster |
| ewconvew_2048_5 | rtx-3090 | main | off-window | 1 / 0 | 3 / 0 | 0.110 / 0.000 | 0.1585 / 2.2616 | 14.238 | 0.134 | ON faster |
| softmax_rows_2048 | rtx-3090 | main | off-fuse | 2 / 5 | 4 / 7 | 1.422 / 1.083 | 1.5033 / 1.1363 | 0.755 | 0.058 | ON slower |
| softmax_rows_2048 | rtx-3090 | main | off-fuse+window+range-map | 2 / 5 | 4 / 7 | 1.422 / 1.083 | 1.5033 / 1.1383 | 0.764 | 0.058 | ON slower |
| softmax_rows_4096 | rtx-3090 | main | off-fuse | 2 / 5 | 4 / 7 | 7.022 / 6.436 | 7.0979 / 6.4883 | 0.914 | 0.006 | ON slower |
| softmax_rows_4096 | rtx-3090 | main | off-fuse+window+range-map | 2 / 5 | 4 / 7 | 7.022 / 6.436 | 7.0979 / 6.4958 | 0.915 | 0.006 | ON slower |
| attn_1024_128 | rtx-3090 | confirm | off-fuse | 3 / 6 | 6 / 9 | 0.569 / 0.207 | 0.6015 / 0.2324 | 0.386 | 0.030 | ON slower |
| attn_1024_128 | rtx-3090 | confirm | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.569 / 0.209 | 0.6015 / 0.2322 | 0.387 | 0.030 | ON slower |
| attn_1024_128 | rtx-3090 | confirm | off-lane-stride | 3 / 1 | 6 / 3 | 0.569 / 0.072 | 0.6015 / 119.80 | 199.054 | 0.030 | ON faster |
| attn_1024_64 | rtx-3090 | confirm | off-fuse | 3 / 6 | 6 / 9 | 0.347 / 0.186 | 0.3799 / 0.2104 | 0.552 | 0.029 | ON slower |
| attn_1024_64 | rtx-3090 | confirm | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.347 / 0.185 | 0.3799 / 0.2110 | 0.550 | 0.029 | ON slower |
| attn_1024_64 | rtx-3090 | confirm | off-lane-stride | 3 / 1 | 6 / 3 | 0.347 / 0.064 | 0.3799 / 58.37 | 154.456 | 0.029 | ON faster |
| attn_2048_128 | rtx-3090 | confirm | off-fuse | 3 / 6 | 6 / 9 | 1.169 / 0.520 | 1.2254 / 0.5703 | 0.465 | 0.028 | ON slower |
| attn_2048_128 | rtx-3090 | confirm | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 1.169 / 0.520 | 1.2254 / 0.5714 | 0.467 | 0.028 | ON slower |
| attn_2048_128 | rtx-3090 | confirm | off-lane-stride | 3 / 1 | 6 / 3 | 1.169 / 0.130 | 1.2254 / 480.96 | 392.622 | 0.028 | ON faster |
| attn_2048_64 | rtx-3090 | confirm | off-fuse | 3 / 6 | 6 / 9 | 0.765 / 0.416 | 0.8183 / 0.4612 | 0.560 | 0.068 | ON slower |
| attn_2048_64 | rtx-3090 | confirm | off-fuse+window+range-map | 3 / 6 | 6 / 9 | 0.765 / 0.415 | 0.8183 / 0.4605 | 0.563 | 0.068 | ON slower |
| attn_2048_64 | rtx-3090 | confirm | off-lane-stride | 3 / 1 | 6 / 3 | 0.765 / 0.102 | 0.8183 / 233.29 | 285.328 | 0.068 | ON faster |
| conv_2048_5 | rtx-3090 | confirm | off-fuse+window+range-map | 1 / 0 | 3 / 0 | 0.108 / 0.000 | 0.1222 / 1.4402 | 11.714 | 0.054 | ON faster |
| conv_2048_5 | rtx-3090 | confirm | off-window | 1 / 0 | 3 / 0 | 0.108 / 0.000 | 0.1222 / 1.4065 | 11.511 | 0.054 | ON faster |
| decode_1024_256 | rtx-3090 | confirm | off-fuse+window+range-map | 4096 / 4096 | 8 / 8 | 44.148 / 44.340 | 53.98 / 222.53 | 4.113 | 0.038 | ON faster |
| decode_1024_256 | rtx-3090 | confirm | off-range-map | 4096 / 4096 | 8 / 8 | 44.148 / 44.258 | 53.98 / 221.06 | 4.112 | 0.038 | ON faster |
| ewconvew_2048_5 | rtx-3090 | confirm | off-fuse+window+range-map | 1 / 0 | 3 / 0 | 0.108 / 0.000 | 0.1457 / 2.3451 | 16.000 | 0.115 | ON faster |
| ewconvew_2048_5 | rtx-3090 | confirm | off-window | 1 / 0 | 3 / 0 | 0.108 / 0.000 | 0.1457 / 2.5799 | 17.820 | 0.115 | ON faster |
| softmax_rows_2048 | rtx-3090 | confirm | off-fuse | 2 / 5 | 4 / 7 | 1.415 / 1.057 | 1.4590 / 1.0991 | 0.754 | 0.009 | ON slower |
| softmax_rows_2048 | rtx-3090 | confirm | off-fuse+window+range-map | 2 / 5 | 4 / 7 | 1.415 / 1.058 | 1.4590 / 1.1016 | 0.756 | 0.009 | ON slower |
| softmax_rows_4096 | rtx-3090 | confirm | off-fuse | 2 / 5 | 4 / 7 | 6.928 / 6.330 | 6.9703 / 6.3708 | 0.914 | 0.001 | ON slower |
| softmax_rows_4096 | rtx-3090 | confirm | off-fuse+window+range-map | 2 / 5 | 4 / 7 | 6.928 / 6.331 | 6.9703 / 6.3692 | 0.914 | 0.001 | ON slower |
| program | archs | launches: base → new rungs off → shipped | launches: `fuse.host` intent = realized | kernels (shipped) | device buffers: base → off → shipped | kept (never stored) | stored flow members | kept: buffers(off-fuse) − buffers = kept | h2d / d2h bytes, whole program: base → shipped | per trip h2d / d2h / host copy, facts→realized: base ⇒ shipped |
|---|---|---|---|---|---|---|---|---|---|---|
| pt_attn_1024_128 | rtx-5090 | 1 → 6 → 3 | ok | fuse tile×2 | 3 → 9 → 6 | den,ex,mx | p | ok | 4.0 M / 8 → 0 / 8 | - ⇒ - |
| pt_attn_1024_128 | rtx-4090 rtx-3090 | 1 → 6 → 3 | ok | fuse tile×3 | 3 → 9 → 6 | den,ex,mx | p | ok | 4.0 M / 8 → 0 / 8 | - ⇒ - |
| pt_attn_1024_128 | h100-pcie | 1 → 6 → 3 | ok | fuse tile×4 | 3 → 9 → 6 | den,ex,mx | p | ok | 4.0 M / 8 → 0 / 8 | - ⇒ - |
| pt_attn_1024_64 | rtx-5090 | 1 → 6 → 3 | ok | fuse tile×2 | 3 → 9 → 6 | den,ex,mx | p | ok | 4.0 M / 8 → 0 / 8 | - ⇒ - |
| pt_attn_1024_64 | rtx-4090 h100-pcie rtx-3090 | 1 → 6 → 3 | ok | fuse tile×3 | 3 → 9 → 6 | den,ex,mx | p | ok | 4.0 M / 8 → 0 / 8 | - ⇒ - |
| pt_attn_2048_128 | rtx-4090 rtx-5090 | 1 → 6 → 3 | ok | fuse tile×2 | 3 → 9 → 6 | den,ex,mx | p | ok | 16.0 M / 8 → 0 / 8 | - ⇒ - |
| pt_attn_2048_128 | h100-pcie rtx-3090 | 1 → 6 → 3 | ok | fuse tile×4 | 3 → 9 → 6 | den,ex,mx | p | ok | 16.0 M / 8 → 0 / 8 | - ⇒ - |
| pt_attn_2048_64 | rtx-4090 rtx-5090 | 1 → 6 → 3 | ok | fuse tile×2 | 3 → 9 → 6 | den,ex,mx | p | ok | 16.0 M / 8 → 0 / 8 | - ⇒ - |
| pt_attn_2048_64 | rtx-3090 | 1 → 6 → 3 | ok | fuse tile×3 | 3 → 9 → 6 | den,ex,mx | p | ok | 16.0 M / 8 → 0 / 8 | - ⇒ - |
| pt_attn_2048_64 | h100-pcie | 1 → 6 → 3 | ok | fuse tile×4 | 3 → 9 → 6 | den,ex,mx | p | ok | 16.0 M / 8 → 0 / 8 | - ⇒ - |
| pt_conv_2048_5 | all four | 0 → 0 → 1 | ok | naive | 0 → 0 → 3 | - | - | ok | 0 / 0 → 0 / 8 | - ⇒ - |
| pt_decode_1024_256 | h100-pcie | 5 → 5 → 5 | ok | naive×4 reduce tile×2 | 9 → 9 → 9 | - | - | ok | 3.0 M / 1.0 M → 3.0 M / 1.0 M | 2.0 K→2.0 K / 1.0 K→1.0 K / 0→0 ⇒ 2.0 K→2.0 K / 1.0 K→1.0 K / 0→0 |
| pt_decode_1024_256 | rtx-4090 rtx-5090 | 6 → 6 → 6 | ok | naive×5 tile | 8 → 8 → 8 | - | - | ok | 3.0 M / 1.0 M → 3.0 M / 1.0 M | 2.0 K→2.0 K / 1.0 K→1.0 K / 0→0 ⇒ 2.0 K→2.0 K / 1.0 K→1.0 K / 0→0 |
| pt_decode_1024_256 | rtx-3090 | 6 → 6 → 6 | ok | naive×5 tile×2 | 8 → 8 → 8 | - | - | ok | 3.0 M / 1.0 M → 3.0 M / 1.0 M | 2.0 K→2.0 K / 1.0 K→1.0 K / 0→0 ⇒ 2.0 K→2.0 K / 1.0 K→1.0 K / 0→0 |
| pt_diamond_4096 | rtx-4090 h100-pcie rtx-3090 | 3 → 3 → 3 | ok | tile×2 | 7 → 7 → 7 | - | - | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_diamond_4096 | rtx-5090 | 3 → 3 → 3 | ok | tile×3 | 7 → 7 → 7 | - | - | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_ew_2048 | all four | 1 → 1 → 1 | ok | cell | 1 → 1 → 1 | - | - | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_ew_4096 | all four | 1 → 1 → 1 | ok | cell | 1 → 1 → 1 | - | - | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_ewconvew_2048_5 | all four | 0 → 0 → 1 | ok | naive | 0 → 0 → 3 | - | - | ok | 0 / 0 → 0 / 8 | - ⇒ - |
| pt_fanout_4096 | rtx-4090 h100-pcie rtx-3090 | 2 → 2 → 2 | ok | tile×2 | 6 → 6 → 6 | - | - | ok | 0 / 24 → 0 / 24 | - ⇒ - |
| pt_fanout_4096 | rtx-5090 | 2 → 2 → 2 | ok | tile×3 | 6 → 6 → 6 | - | - | ok | 0 / 24 → 0 / 24 | - ⇒ - |
| pt_fir_4194304_16 | all four | 1 → 1 → 1 | ok | naive | 3 → 3 → 3 | - | - | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_firew_rowsum_2048x2048_16 | all four | 2 → 2 → 2 | ok | naive rowfold | 4 → 4 → 4 | - | - | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_indep_4096 | rtx-4090 h100-pcie rtx-3090 | 2 → 2 → 2 | ok | tile×2 | 6 → 6 → 6 | - | - | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_indep_4096 | rtx-5090 | 2 → 2 → 2 | ok | tile×3 | 6 → 6 → 6 | - | - | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_loop_2048 | all four | 1 → 1 → 1 | ok | tile | 3 → 3 → 3 | - | - | ok | 0 / 8 → 0 / 8 | 0→0 / 0→0 / 0→0 ⇒ 0→0 / 0→0 / 0→0 |
| pt_loop_4096 | all four | 1 → 1 → 1 | ok | tile | 3 → 3 → 3 | - | - | ok | 0 / 8 → 0 / 8 | 0→0 / 0→0 / 0→0 ⇒ 0→0 / 0→0 / 0→0 |
| pt_loopmm_1024_8 | rtx-4090 rtx-5090 | 1 → 1 → 1 | ok | tile | 3 → 3 → 3 | - | - | ok | 0 / 4 → 0 / 4 | 0→0 / 0→0 / 0→0 ⇒ 0→0 / 0→0 / 0→0 |
| pt_loopmm_1024_8 | h100-pcie rtx-3090 | 1 → 1 → 1 | ok | tile×2 | 3 → 3 → 3 | - | - | ok | 0 / 4 → 0 / 4 | 0→0 / 0→0 / 0→0 ⇒ 0→0 / 0→0 / 0→0 |
| pt_rm_m128_d1024_h2048 | all four | 2 → 2 → 2 | ok | tile×2 | 6 → 6 → 6 | - | - | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_rm_m128_d256_h512 | all four | 2 → 2 → 2 | ok | tile×2 | 6 → 6 → 6 | - | - | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_rowsum_2048 | rtx-4090 h100-pcie rtx-3090 | 2 → 2 → 2 | ok | rowfold tile | 4 → 4 → 4 | - | - | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_rowsum_2048 | rtx-5090 | 2 → 2 → 2 | ok | rowfold tile×2 | 4 → 4 → 4 | - | - | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_rowsum_4096 | rtx-4090 h100-pcie rtx-3090 | 2 → 2 → 2 | ok | rowfold tile | 4 → 4 → 4 | - | - | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_rowsum_4096 | rtx-5090 | 2 → 2 → 2 | ok | rowfold tile×2 | 4 → 4 → 4 | - | - | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_rt_mlp2_1024 | rtx-4090 rtx-5090 | 4 → 4 → 4 | ok | tile×4 | 13 → 13 → 13 | - | - | ok | 68.0 M / 8 → 68.0 M / 8 | 8.0 M→8.0 M / 0→0 / 0→0 ⇒ 8.0 M→8.0 M / 0→0 / 0→0 |
| pt_rt_mlp2_1024 | h100-pcie rtx-3090 | 4 → 4 → 4 | ok | tile×5 | 13 → 13 → 13 | - | - | ok | 68.0 M / 8 → 68.0 M / 8 | 8.0 M→8.0 M / 0→0 / 0→0 ⇒ 8.0 M→8.0 M / 0→0 / 0→0 |
| pt_smlp_4096_256_1024 | rtx-4090 rtx-5090 | 2 → 2 → 2 | ok | tile×2 | 6 → 6 → 6 | - | - | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_smlp_4096_256_1024 | h100-pcie rtx-3090 | 2 → 2 → 2 | ok | tile×3 | 6 → 6 → 6 | - | - | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_softmax_rows_2048 | rtx-4090 h100-pcie rtx-3090 | 5 → 5 → 2 | ok | fuse tile | 7 → 7 → 4 | den,ex,mx | y | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_softmax_rows_2048 | rtx-5090 | 5 → 5 → 2 | ok | fuse tile×2 | 7 → 7 → 4 | den,ex,mx | y | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_softmax_rows_4096 | rtx-4090 h100-pcie rtx-3090 | 5 → 5 → 2 | ok | fuse tile | 7 → 7 → 4 | den,ex,mx | y | ok | 0 / 8 → 0 / 8 | - ⇒ - |
| pt_softmax_rows_4096 | rtx-5090 | 5 → 5 → 2 | ok | fuse tile×2 | 7 → 7 → 4 | den,ex,mx | y | ok | 0 / 8 → 0 / 8 | - ⇒ - |

### The unit kernel against the kernels it replaces (traced ms, median of the traced reps)

| program | arch | unit kernel: blocks × threads = rows, ms | off-fuse kernels in its place, ms | unit / those |
|---|---|---|---|---|
| attn_1024_128 | rtx-4090 | 32 × 32 = 1024, 0.308 | rowfold 0.028, cell 0.014, rowfold 0.027, cell 0.011 (Σ 0.080) | 3.85 |
| attn_1024_128 | rtx-5090 | 32 × 32 = 1024, 0.296 | rowfold 0.026, cell 0.011, rowfold 0.024, cell 0.011 (Σ 0.072) | 4.11 |
| attn_1024_128 | h100-pcie | 32 × 32 = 1024, 0.365 | rowfold 0.037, cell 0.010, rowfold 0.033, cell 0.009 (Σ 0.089) | 4.10 |
| attn_1024_128 | rtx-3090 | 32 × 32 = 1024, 0.465 | rowfold 0.036, cell 0.017, rowfold 0.038, cell 0.016 (Σ 0.107) | 4.35 |
| attn_1024_64 | rtx-4090 | 32 × 32 = 1024, 0.173 | rowfold 0.026, cell 0.010, rowfold 0.024, cell 0.010 (Σ 0.070) | 2.47 |
| attn_1024_64 | rtx-5090 | 32 × 32 = 1024, 0.152 | rowfold 0.026, cell 0.011, rowfold 0.023, cell 0.012 (Σ 0.072) | 2.11 |
| attn_1024_64 | h100-pcie | 32 × 32 = 1024, 0.235 | rowfold 0.037, cell 0.011, rowfold 0.035, cell 0.010 (Σ 0.093) | 2.53 |
| attn_1024_64 | rtx-3090 | 32 × 32 = 1024, 0.261 | rowfold 0.036, cell 0.017, rowfold 0.039, cell 0.017 (Σ 0.109) | 2.39 |
| attn_2048_128 | rtx-4090 | 64 × 32 = 2048, 0.607 | rowfold 0.046, cell 0.020, rowfold 0.042, cell 0.018 (Σ 0.126) | 4.82 |
| attn_2048_128 | rtx-5090 | 64 × 32 = 2048, 0.531 | rowfold 0.042, cell 0.017, rowfold 0.038, cell 0.017 (Σ 0.114) | 4.66 |
| attn_2048_128 | h100-pcie | 64 × 32 = 2048, 0.766 | rowfold 0.068, cell 0.022, rowfold 0.066, cell 0.020 (Σ 0.176) | 4.35 |
| attn_2048_128 | rtx-3090 | 64 × 32 = 2048, 0.890 | rowfold 0.083, cell 0.048, rowfold 0.067, cell 0.047 (Σ 0.245) | 3.63 |
| attn_2048_64 | rtx-4090 | 64 × 32 = 2048, 0.337 | rowfold 0.045, cell 0.018, rowfold 0.041, cell 0.018 (Σ 0.122) | 2.76 |
| attn_2048_64 | rtx-5090 | 64 × 32 = 2048, 0.295 | rowfold 0.042, cell 0.018, rowfold 0.038, cell 0.018 (Σ 0.116) | 2.54 |
| attn_2048_64 | h100-pcie | 64 × 32 = 2048, 0.492 | rowfold 0.066, cell 0.024, rowfold 0.064, cell 0.019 (Σ 0.173) | 2.84 |
| attn_2048_64 | rtx-3090 | 64 × 32 = 2048, 0.592 | rowfold 0.083, cell 0.046, rowfold 0.067, cell 0.047 (Σ 0.243) | 2.44 |
| softmax_rows_2048 | rtx-4090 | 64 × 32 = 2048, 0.335 | rowfold 0.047, cell 0.019, rowfold 0.042, cell 0.019 (Σ 0.127) | 2.64 |
| softmax_rows_2048 | rtx-5090 | 64 × 32 = 2048, 0.297 | rowfold 0.044, cell 0.018, rowfold 0.038, cell 0.017 (Σ 0.117) | 2.54 |
| softmax_rows_2048 | h100-pcie | 64 × 32 = 2048, 0.512 | rowfold 0.086, cell 0.018, rowfold 0.065, cell 0.018 (Σ 0.187) | 2.74 |
| softmax_rows_2048 | rtx-3090 | 64 × 32 = 2048, 0.591 | rowfold 0.082, cell 0.048, rowfold 0.068, cell 0.048 (Σ 0.246) | 2.40 |
| softmax_rows_4096 | rtx-4090 | 128 × 32 = 4096, 0.859 | rowfold 0.139, cell 0.109, rowfold 0.119, cell 0.112 (Σ 0.479) | 1.79 |
| softmax_rows_4096 | rtx-5090 | 128 × 32 = 4096, 0.720 | rowfold 0.110, cell 0.062, rowfold 0.083, cell 0.057 (Σ 0.312) | 2.31 |
| softmax_rows_4096 | h100-pcie | 64 × 64 = 4096, 1.335 | rowfold 0.223, cell 0.082, rowfold 0.205, cell 0.082 (Σ 0.592) | 2.26 |
| softmax_rows_4096 | rtx-3090 | 64 × 64 = 4096, 1.334 | rowfold 0.205, cell 0.171, rowfold 0.192, cell 0.170 (Σ 0.738) | 1.81 |

### Attention: Mapal against the library and torch legs, same box

| program | arch | box | Mapal flow ON | Mapal flow OFF (ships) | cuBLAS eager (6 launches) | cuBLAS + warp-per-row softmax (3) | torch eager | torch compile | OFF / best other |
|---|---|---|---|---|---|---|---|---|---|
| attn_1024_128 | rtx-4090 | main | 0.4452 | 0.2648 | 0.0512 | 0.0471 | 0.0983 | 0.0440 | 6.018 |
| attn_1024_64 | rtx-4090 | main | 0.2952 | 0.2839 | 0.0430 | 0.0379 | 0.0891 | 0.0328 | 8.656 |
| attn_2048_128 | rtx-4090 | main | 0.8427 | 0.3528 | 0.1300 | 0.1198 | 0.2570 | 0.0881 | 4.005 |
| attn_2048_64 | rtx-4090 | main | 0.5561 | 0.4186 | 0.1106 | 0.0993 | 0.2202 | 0.0677 | 6.183 |
| attn_1024_128 | rtx-5090 | main | 0.3497 | 0.1977 | - | - | 0.1499 | 0.0328 | 6.028 |
| attn_1024_64 | rtx-5090 | main | 0.2321 | 0.1967 | - | - | 0.1422 | 0.0689 | 2.855 |
| attn_2048_128 | rtx-5090 | main | 0.6694 | 0.2757 | - | - | 0.2028 | 0.0819 | 3.366 |
| attn_2048_64 | rtx-5090 | main | 0.4272 | 0.2405 | - | - | 0.1740 | 0.0745 | 3.228 |
| attn_1024_128 | h100-pcie | main | 0.5547 | 0.3162 | - | - | 0.1073 | 0.0584 | 5.414 |
| attn_1024_64 | h100-pcie | main | 0.3804 | 0.1804 | - | - | 0.0952 | 0.0323 | 5.586 |
| attn_2048_128 | h100-pcie | main | 1.0186 | 0.4192 | - | - | 0.2770 | 0.1179 | 3.555 |
| attn_2048_64 | h100-pcie | main | 0.6799 | 0.3635 | - | - | 0.2361 | 0.0781 | 4.655 |
| attn_1024_128 | rtx-3090 | main | 0.6219 | 0.2851 | 0.0860 | 0.0727 | - | - | 3.921 |
| attn_1024_64 | rtx-3090 | main | 0.4072 | 0.2660 | 0.0727 | 0.0594 | - | - | 4.478 |
| attn_2048_128 | rtx-3090 | main | 1.2656 | 0.6131 | 0.2929 | 0.2621 | - | - | 2.339 |
| attn_2048_64 | rtx-3090 | main | 0.8493 | 0.4943 | 0.2355 | 0.2058 | - | - | 2.402 |
| attn_1024_128 | rtx-5090 | attn-confirm | 0.3578 | 0.1846 | 0.0389 | 0.0348 | 0.0963 | 0.0307 | 6.013 |
| attn_1024_64 | rtx-5090 | attn-confirm | 0.2324 | 0.1832 | 0.0307 | 0.0266 | 0.0952 | 0.0486 | 6.887 |
| attn_2048_128 | rtx-5090 | attn-confirm | 0.7021 | 0.2555 | 0.0839 | 0.0881 | 0.1802 | 0.0850 | 3.045 |
| attn_2048_64 | rtx-5090 | attn-confirm | 0.4566 | 0.2314 | 0.0676 | 0.0696 | 0.1464 | 0.0526 | 4.399 |
| attn_1024_128 | h100-pcie | attn-confirm | 0.4895 | 0.2093 | 0.0598 | 0.0557 | 0.1080 | 0.0439 | 4.768 |
| attn_1024_64 | h100-pcie | attn-confirm | 0.3267 | 0.1756 | 0.0487 | 0.0430 | 0.0958 | 0.0330 | 5.320 |
| attn_2048_128 | h100-pcie | attn-confirm | 1.0252 | 0.4245 | 0.1488 | 0.1392 | 0.2761 | 0.1178 | 3.604 |
| attn_2048_64 | h100-pcie | attn-confirm | 0.6905 | 0.3761 | 0.1095 | 0.1001 | 0.2360 | 0.0780 | 4.822 |

### Suite: iter ms, Mapal vs the best other leg on the same box (CPU+GPU measured = flow ON; the column after the ratio = the build as it ships now)

| workload | arch | CPU-only | CPU+GPU (measured) | null | best other (ms) | Mapal / best other | shipped after the flow rows: ms, ratio | confirm box: CPU+GPU, best other, ratio | S65 ratio |
|---|---|---|---|---|---|---|---|---|---|
| attn_1024_128 | rtx-4090 | 26.51 | 0.4452 | 0.028 | compile 0.0440 | 10.118 | 0.2648, 6.018 | 0.3927, compile 0.0440, 8.924 | - |
| attn_1024_128 | rtx-5090 | 16.75 | 0.3497 | 0.011 | compile 0.0328 | 10.662 | 0.1977, 6.028 | 0.3358, compile 0.0307, 10.939 | - |
| attn_1024_128 | h100-pcie | 13.85 | 0.5547 | 0.112 | compile 0.0584 | 9.498 | 0.3162, 5.414 | 0.6055, compile 0.0520, 11.645 | - |
| attn_1024_128 | rtx-3090 | 25.73 | 0.6219 | 0.006 | fused 0.0727 | 8.554 | 0.2851, 3.921 | 0.6015, fused 0.0717, 8.389 | - |
| attn_1024_64 | rtx-4090 | 19.08 | 0.2952 | 0.012 | compile 0.0328 | 9.001 | 0.2839, 8.656 | 0.2537, compile 0.0307, 8.265 | - |
| attn_1024_64 | rtx-5090 | 10.12 | 0.2321 | 0.041 | compile 0.0689 | 3.368 | 0.1967, 2.855 | 0.2043, compile 0.0245, 8.339 | - |
| attn_1024_64 | h100-pcie | 9.0858 | 0.3804 | 0.138 | compile 0.0323 | 11.777 | 0.1804, 5.586 | 0.3924, fused 0.0503, 7.800 | - |
| attn_1024_64 | rtx-3090 | 14.30 | 0.4072 | 0.040 | fused 0.0594 | 6.855 | 0.2660, 4.478 | 0.3799, compile 0.0512, 7.420 | - |
| attn_2048_128 | rtx-4090 | 93.58 | 0.8427 | 0.013 | compile 0.0881 | 9.566 | 0.3528, 4.005 | 0.7845, compile 0.0829, 9.463 | - |
| attn_2048_128 | rtx-5090 | 59.36 | 0.6694 | 0.006 | compile 0.0819 | 8.173 | 0.2757, 3.366 | 0.6936, compile 0.0850, 8.160 | - |
| attn_2048_128 | h100-pcie | 42.17 | 1.0186 | 0.004 | compile 0.1179 | 8.639 | 0.4192, 3.555 | 1.2608, compile 0.1481, 8.513 | - |
| attn_2048_128 | rtx-3090 | 77.97 | 1.2656 | 0.006 | fused 0.2621 | 4.829 | 0.6131, 2.339 | 1.2254, compile 0.1884, 6.504 | - |
| attn_2048_64 | rtx-4090 | 64.01 | 0.5561 | 0.050 | compile 0.0677 | 8.214 | 0.4186, 6.183 | 0.4775, compile 0.0645, 7.402 | - |
| attn_2048_64 | rtx-5090 | 40.14 | 0.4272 | 0.018 | compile 0.0745 | 5.734 | 0.2405, 3.228 | 0.4114, compile 0.0502, 8.195 | - |
| attn_2048_64 | h100-pcie | 34.51 | 0.6799 | 0.011 | compile 0.0781 | 8.705 | 0.3635, 4.655 | 0.8624, compile 0.0968, 8.909 | - |
| attn_2048_64 | rtx-3090 | 51.25 | 0.8493 | 0.003 | fused 0.2058 | 4.127 | 0.4943, 2.402 | 0.8183, compile 0.1362, 6.008 | - |
| conv_2048_5 | rtx-4090 | 2.3704 | 0.0762 | 0.002 | compile 0.0881 | 0.865 | = | 0.0606, eager 0.0870, 0.696 | 15.191 |
| conv_2048_5 | rtx-5090 | 1.8230 | 0.0559 | 0.014 | eager 0.0778 | 0.718 | = | 0.0447, eager 0.0819, 0.546 | 22.679 |
| conv_2048_5 | h100-pcie | 0.7221 | 0.0687 | 0.035 | compile 0.6204 | 0.111 | = | 0.0826, compile 0.7725, 0.107 | 1.345 |
| conv_2048_5 | rtx-3090 | 1.5880 | 0.1317 | 0.008 | - | - | = | 0.1222, eager 0.1618, 0.755 | 8.919 |
| decode_1024_256 | rtx-4090 | 3815.71 | 60.08 | 0.003 | cache 30.1230 | 1.995 | = | 37.35, cache 17.32, 2.156 | 3.059 |
| decode_1024_256 | rtx-5090 | 3821.41 | 53.89 | 0.006 | nocache 42.3200 | 1.273 | = | 40.63, nocache 21.02, 1.932 | 2.264 |
| decode_1024_256 | h100-pcie | 2727.87 | 73.05 | 0.003 | cache 19.7850 | 3.692 | = | 82.97, cache 25.23, 3.289 | 3.232 |
| decode_1024_256 | rtx-3090 | 3836.87 | 58.96 | 0.010 | cache 32.9030 | 1.792 | = | 53.98, cache 30.43, 1.774 | 1.640 |
| diamond_4096 | rtx-4090 | 694.21 | 7.3423 | 0.003 | cutlass 7.4936 | 0.980 | = | 7.5684, compile 8.4552, 0.895 | 1.120 |
| diamond_4096 | rtx-5090 | 408.68 | 7.1070 | 0.001 | compile 6.7287 | 1.056 | = | 6.2832, compile 7.6288, 0.824 | 1.006 |
| diamond_4096 | h100-pcie | 238.50 | 10.74 | 0.000 | streams 10.7348 | 1.001 | = | 13.48, streams 12.96, 1.041 | 1.042 |
| diamond_4096 | rtx-3090 | 670.00 | 17.42 | 0.001 | streams 27.3551 | 0.637 | = | 17.00, streams 17.09, 0.995 | 0.992 |
| ew_2048 | rtx-4090 | 2.1955 | 0.3901 | 0.254 | fused 0.0584 | 6.679 | = | 0.2168, fused 0.0584, 3.713 | 3.462 |
| ew_2048 | rtx-5090 | 1.6317 | 0.5324 | 0.055 | fused 0.0348 | 15.298 | = | 0.2223, fused 0.0348, 6.389 | 3.061 |
| ew_2048 | h100-pcie | 0.5972 | 0.5161 | 0.270 | compile 0.0339 | 15.223 | = | 0.6674, compile 0.0329, 20.285 | 19.915 |
| ew_2048 | rtx-3090 | 1.2762 | 0.8124 | 0.030 | fused 0.0645 | 12.595 | = | 0.4269, compile 0.0635, 6.722 | 6.926 |
| ew_4096 | rtx-4090 | 8.6170 | 0.5679 | 0.071 | fused 0.2263 | 2.510 | = | 0.2823, fused 0.2191, 1.288 | 1.206 |
| ew_4096 | rtx-5090 | 4.5254 | 0.6027 | 0.092 | fused 0.1311 | 4.597 | = | 0.2140, fused 0.1311, 1.632 | 1.088 |
| ew_4096 | h100-pcie | 1.9225 | 0.5669 | 0.130 | compile 0.1143 | 4.959 | = | 0.8323, compile 0.1142, 7.288 | 6.812 |
| ew_4096 | rtx-3090 | 4.8234 | 0.9457 | 0.005 | fused 0.2417 | 3.913 | = | 0.5501, fused 0.2406, 2.286 | 2.109 |
| ewconvew_2048_5 | rtx-4090 | 4.1465 | 0.1012 | 0.000 | compile 0.1239 | 0.817 | = | 0.0769, compile 0.1241, 0.620 | - |
| ewconvew_2048_5 | rtx-5090 | 2.6344 | 0.0815 | 0.015 | compile 0.1004 | 0.811 | = | 0.0589, compile 0.1055, 0.558 | - |
| ewconvew_2048_5 | h100-pcie | 1.1452 | 0.0855 | 0.008 | compile 0.6497 | 0.132 | = | 0.1264, compile 0.8031, 0.157 | - |
| ewconvew_2048_5 | rtx-3090 | 2.4749 | 0.1585 | 0.024 | - | - | = | 0.1457, compile 0.2468, 0.590 | - |
| fanout_4096 | rtx-4090 | 501.98 | 4.8915 | 0.008 | streams 4.9132 | 0.996 | = | 5.0417, streams 5.6092, 0.899 | 1.118 |
| fanout_4096 | rtx-5090 | 294.08 | 4.7285 | 0.001 | compile 4.4872 | 1.054 | = | 4.2332, compile 5.0012, 0.846 | 1.027 |
| fanout_4096 | h100-pcie | 175.14 | 7.1524 | 0.000 | streams 7.4913 | 0.955 | = | 9.0089, streams 9.1557, 0.984 | 0.980 |
| fanout_4096 | rtx-3090 | 461.97 | 11.55 | 0.000 | streams 18.0552 | 0.640 | = | 11.42, streams 11.53, 0.990 | 0.985 |
| fir_4194304_16 | rtx-4090 | 2.0872 | 0.0553 | 0.062 | compile 0.0614 | 0.901 | = | 0.0419, compile 0.0607, 0.690 | 0.603 |
| fir_4194304_16 | rtx-5090 | 1.4204 | 0.0453 | 0.051 | compile 0.0492 | 0.921 | = | 0.0313, compile 0.0522, 0.599 | 0.522 |
| fir_4194304_16 | h100-pcie | 0.6801 | 0.0484 | 0.013 | compile 0.0770 | 0.629 | = | 0.0663, eager 0.0924, 0.717 | 0.689 |
| fir_4194304_16 | rtx-3090 | 1.3302 | 0.0913 | 0.015 | - | - | = | 0.0834, compile 0.1075, 0.776 | 0.744 |
| firew_rowsum_2048x2048_16 | rtx-4090 | 7.0707 | 0.0933 | 0.018 | compile 0.0676 | 1.380 | = | 0.0759, compile 0.0674, 1.126 | - |
| firew_rowsum_2048x2048_16 | rtx-5090 | 10.76 | 0.0838 | 0.005 | compile 0.0553 | 1.515 | = | 0.0643, compile 0.0563, 1.142 | - |
| firew_rowsum_2048x2048_16 | h100-pcie | 4.3383 | 0.1146 | 0.019 | compile 0.0838 | 1.368 | = | 0.1442, compile 0.1012, 1.425 | - |
| firew_rowsum_2048x2048_16 | rtx-3090 | 8.1015 | 0.1543 | 0.003 | - | - | = | 0.1459, compile 0.1270, 1.149 | - |
| indep_4096 | rtx-4090 | 452.54 | 4.9628 | 0.000 | lt 4.9725 | 0.998 | = | 5.0505, compile 5.6412, 0.895 | 1.141 |
| indep_4096 | rtx-5090 | 262.64 | 4.7357 | 0.000 | compile 4.4626 | 1.061 | = | 5.3814, compile 5.0022, 1.076 | 1.025 |
| indep_4096 | h100-pcie | 161.58 | 7.2716 | 0.000 | streams 6.8949 | 1.055 | = | 9.0445, streams 8.3839, 1.079 | 1.079 |
| indep_4096 | rtx-3090 | 402.78 | 11.61 | 0.016 | streams 18.9338 | 0.613 | = | 11.38, streams 11.16, 1.020 | 0.996 |
| loop_2048 | rtx-4090 | 1942.87 | 2.7315 | 0.003 | cutlass 2.3972 | 1.139 | = | 2.5913, cutlass 2.5631, 1.011 | 0.997 |
| loop_2048 | rtx-5090 | 1974.68 | 2.4830 | 0.000 | lt 2.0849 | 1.191 | = | 2.2580, lt 2.0552, 1.099 | 1.071 |
| loop_2048 | h100-pcie | 1303.60 | 5.1264 | 0.001 | lt 4.0731 | 1.259 | = | 6.3993, lt 5.0375, 1.270 | 1.270 |
| loop_2048 | rtx-3090 | 2056.11 | 7.2901 | 0.009 | lt 9.1628 | 0.796 | = | 7.0320, lt 6.0918, 1.154 | 1.150 |
| loop_4096 | rtx-4090 | 17076.19 | 19.87 | 0.003 | cutlass 19.2686 | 1.031 | = | 19.76, compile 21.68, 0.911 | 0.911 |
| loop_4096 | rtx-5090 | 16778.62 | 17.34 | 0.001 | lt 18.3634 | 0.945 | = | 15.83, lt 16.55, 0.957 | 0.971 |
| loop_4096 | h100-pcie | 11494.05 | 28.62 | 0.000 | lt 29.4334 | 0.972 | = | 36.01, lt 36.60, 0.984 | 0.981 |
| loop_4096 | rtx-3090 | 18116.27 | 59.74 | 0.010 | lt 64.1444 | 0.931 | = | 45.94, lt 45.28, 1.014 | 1.010 |
| loopmm_1024_8 | rtx-4090 | 233.26 | 0.7195 | 0.020 | eager 0.3594 | 2.002 | = | 0.5756, compile 0.3707, 1.553 | 1.287 |
| loopmm_1024_8 | rtx-5090 | 237.60 | 0.6632 | 0.011 | compile 0.3482 | 1.905 | = | 0.5143, compile 0.3779, 1.361 | 1.368 |
| loopmm_1024_8 | h100-pcie | 164.58 | 0.8121 | 0.050 | compile 0.6224 | 1.305 | = | 1.0566, eager 0.7620, 1.387 | 1.443 |
| loopmm_1024_8 | rtx-3090 | 241.07 | 1.8089 | 0.004 | - | - | = | 1.6147, compile 0.8847, 1.825 | 1.807 |
| rm_m128_d1024_h2048 | rtx-4090 | 7.6986 | 0.2250 | 0.087 | compile 0.0594 | 3.788 | = | 0.1176, compile 0.0603, 1.950 | 1.980 |
| rm_m128_d1024_h2048 | rtx-5090 | 8.7179 | 0.1603 | 0.022 | compile 0.0410 | 3.911 | = | 0.1080, compile 0.0420, 2.572 | 1.894 |
| rm_m128_d1024_h2048 | h100-pcie | 3.1171 | 0.1612 | 0.061 | compile 0.0679 | 2.374 | = | 0.2354, compile 0.0786, 2.995 | 2.410 |
| rm_m128_d1024_h2048 | rtx-3090 | 8.7985 | 0.3592 | 0.013 | - | - | = | 0.3478, eager 0.0952, 3.654 | 3.656 |
| rm_m128_d256_h512 | rtx-4090 | 0.8080 | 0.2310 | 0.143 | compile 0.0225 | 10.268 | = | 0.1031, compile 0.0225, 4.584 | 4.534 |
| rm_m128_d256_h512 | rtx-5090 | 0.7494 | 0.1316 | 0.098 | eager 0.0246 | 5.350 | = | 0.0663, eager 0.0184, 3.605 | 1.728 |
| rm_m128_d256_h512 | h100-pcie | 0.5266 | 0.1024 | 0.018 | compile 0.0263 | 3.894 | = | 0.1728, eager 0.0340, 5.082 | 4.199 |
| rm_m128_d256_h512 | rtx-3090 | 0.8434 | 0.1415 | 0.028 | - | - | = | 0.1084, eager 0.0276, 3.929 | 3.908 |
| rowsum_2048 | rtx-4090 | 46.59 | 0.3945 | 0.015 | eager 0.3256 | 1.212 | = | 0.3497, eager 0.3318, 1.054 | 1.279 |
| rowsum_2048 | rtx-5090 | 33.70 | 0.3595 | 0.001 | eager 0.2601 | 1.382 | = | 0.3629, eager 0.2867, 1.266 | 1.204 |
| rowsum_2048 | h100-pcie | 20.65 | 0.7303 | 0.001 | compile 0.5245 | 1.392 | = | 0.8997, compile 0.6432, 1.399 | 1.449 |
| rowsum_2048 | rtx-3090 | 39.47 | 0.9673 | 0.005 | gemv 1.3855 | 0.698 | = | 0.9399, eager 0.7393, 1.271 | 1.214 |
| rowsum_4096 | rtx-4090 | 274.35 | 2.5577 | 0.002 | gemv 2.4320 | 1.052 | = | 2.5978, eager 2.5784, 1.008 | 0.919 |
| rowsum_4096 | rtx-5090 | 158.29 | 2.4184 | 0.002 | compile 2.1944 | 1.102 | = | 2.7215, compile 2.4330, 1.119 | 1.083 |
| rowsum_4096 | h100-pcie | 108.80 | 3.8147 | 0.001 | gemv 3.7188 | 1.026 | = | 4.7612, gemv 4.6135, 1.032 | 1.036 |
| rowsum_4096 | rtx-3090 | 250.66 | 5.9895 | 0.001 | eager 8.7429 | 0.685 | = | 5.8526, eager 5.7426, 1.019 | 1.014 |
| rt_mlp2_1024 | rtx-4090 | 987.29 | 21.93 | 0.072 | compile 1.6179 | 13.553 | = | 9.3936, compile 1.7101, 5.493 | 5.072 |
| rt_mlp2_1024 | rtx-5090 | 992.19 | 16.99 | 0.008 | compile 1.5811 | 10.745 | = | 7.3555, compile 1.6640, 4.420 | 4.892 |
| rt_mlp2_1024 | h100-pcie | 700.44 | 9.2325 | 0.007 | compile 2.6331 | 3.506 | = | 12.10, compile 3.6277, 3.335 | 6.429 |
| rt_mlp2_1024 | rtx-3090 | 999.81 | 18.84 | 0.023 | - | - | = | 13.36, compile 4.1472, 3.222 | 3.497 |
| smlp_4096_256_1024 | rtx-4090 | 12.32 | 0.2515 | 0.026 | compile 0.1085 | 2.318 | = | 0.1619, compile 0.1034, 1.566 | - |
| smlp_4096_256_1024 | rtx-5090 | 6.6526 | 0.1698 | 0.000 | compile 0.0840 | 2.021 | = | 0.1431, compile 0.0911, 1.571 | - |
| smlp_4096_256_1024 | h100-pcie | 4.6616 | 0.2434 | 0.015 | compile 0.1562 | 1.558 | = | 0.3075, lt 0.1892, 1.625 | - |
| smlp_4096_256_1024 | rtx-3090 | 9.5739 | 0.4303 | 0.005 | cutlass 0.2775 | 1.550 | = | 0.3809, compile 0.2396, 1.590 | - |
| softmax_rows_2048 | rtx-4090 | 54.31 | 0.7181 | 0.000 | compile 0.3430 | 2.094 | 0.4882, 1.423 | 0.6647, compile 0.3442, 1.931 | 1.554 |
| softmax_rows_2048 | rtx-5090 | 40.45 | 0.6239 | 0.009 | compile 0.2622 | 2.380 | 0.4193, 1.599 | 0.6328, compile 0.2908, 2.176 | 1.412 |
| softmax_rows_2048 | h100-pcie | 30.20 | 1.1974 | 0.014 | compile 0.5274 | 2.270 | 0.8456, 1.603 | 1.4585, compile 0.6489, 2.248 | 1.628 |
| softmax_rows_2048 | rtx-3090 | 44.81 | 1.5033 | 0.009 | eager 0.9431 | 1.594 | 1.1363, 1.205 | 1.4590, compile 0.7997, 1.824 | 1.458 |
| softmax_rows_4096 | rtx-4090 | 288.13 | 3.2960 | 0.003 | compile 2.4617 | 1.339 | 2.9111, 1.183 | 3.3505, compile 2.5713, 1.303 | 1.455 |
| softmax_rows_4096 | rtx-5090 | 170.85 | 3.0516 | 0.000 | compile 2.2579 | 1.352 | 2.6263, 1.163 | 2.8255, compile 2.4504, 1.153 | 1.162 |
| softmax_rows_4096 | h100-pcie | 118.23 | 4.9445 | 0.001 | compile 3.7929 | 1.304 | 4.2031, 1.108 | 6.1262, compile 4.6696, 1.312 | 1.100 |
| softmax_rows_4096 | rtx-3090 | 255.09 | 7.0979 | 0.001 | fused 9.3379 | 0.760 | 6.4883, 0.695 | 6.9703, compile 5.9382, 1.174 | 1.088 |

### GEMM cells and MLP kernels: Mapal vs cuBLAS / cuBLASLt / CUTLASS (main boxes, median ms, both processes)

| cell | arch | Mapal default | realization | Mapal _c | Mapal _x | cuBLAS | cuBLASLt | CUTLASS | tuned CUTLASS | default / best library | null |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mm_128x2048x7168 | rtx-4090 | 0.1649 | none | 0.1638 | 0.2038 | 0.1516 | - | 0.5263 | 0.2263 | 1.088 | 0.012 |
| mm_16x1024x4096 | rtx-4090 | 0.0369 | none | 0.0369 | 0.0686 | 0.0358 | - | 0.3051 | 0.1178 | 1.031 | 0.031 |
| mm_16x18432x7168 | rtx-4090 | 0.6195 | none | 0.6195 | 0.6195 | 0.6369 | - | 1.0200 | 0.6154 | 1.007 | 0.001 |
| mm_16x2048x7168 | rtx-4090 | 0.0993 | none | 0.0993 | 0.1044 | 0.0952 | - | 0.5304 | 0.2048 | 1.043 | 0.000 |
| mm_4096x4096x4096 | rtx-4090 | 2.3748 | none | 2.3726 | 4.0069 | 2.4730 | - | 2.4852 | 6.7215 | 0.960 | 0.001 |
| pt_rm_m128_d1024_h2048 | rtx-4090 | 0.0625 | - | = | 0.0850 | 0.0512 | 0.0481 | 0.2376 | 0.0727 | 1.299 | 0.002 |
| pt_rm_m128_d256_h512 | rtx-4090 | 0.0214 | - | = | 0.0276 | 0.0307 | 0.0205 | 0.0686 | 0.0214 | 1.044 | 0.009 |
| mm_128x2048x7168 | rtx-5090 | 0.1454 | none | 0.1454 | 0.2969 | 0.0881 | - | 0.5468 | 0.2212 | 1.650 | 0.000 |
| mm_16x1024x4096 | rtx-5090 | 0.0450 | none | 0.0450 | 0.0471 | 0.0205 | - | 0.2929 | 0.1270 | 2.195 | 0.000 |
| mm_16x18432x7168 | rtx-5090 | 0.4076 | streamk | 0.4506 | 0.3379 | 0.3379 | - | 0.5427 | 0.3687 | 1.206 | 0.005 |
| mm_16x2048x7168 | rtx-5090 | 0.0799 | none | 0.0799 | 0.0901 | 0.0532 | - | 0.5018 | 0.2171 | 1.502 | 0.000 |
| mm_4096x4096x4096 | rtx-5090 | 2.5866 | streamk | 2.6040 | 4.1533 | 2.3880 | - | 2.6112 | 5.4815 | 1.083 | 0.001 |
| pt_rm_m128_d1024_h2048 | rtx-5090 | 0.0574 | - | = | 0.1092 | 0.0738 | 0.0400 | 0.2049 | 0.0615 | 1.435 | 0.002 |
| pt_rm_m128_d256_h512 | rtx-5090 | 0.0184 | - | = | 0.0293 | 0.0318 | 0.0224 | 0.0594 | 0.0205 | 0.898 | 0.005 |
| mm_128x2048x7168 | h100-pcie | 0.2672 | streamk | 0.4087 | 0.4006 | 0.1164 | - | 0.8414 | 0.3500 | 2.296 | 0.003 |
| mm_16x1024x4096 | h100-pcie | 0.0393 | streamk | 0.0889 | 0.0801 | 0.0296 | - | 0.4752 | 0.1720 | 1.328 | 0.005 |
| mm_16x18432x7168 | h100-pcie | 0.4830 | none | 0.4836 | 0.5234 | 0.3671 | - | 1.6513 | 0.3927 | 1.316 | 0.000 |
| mm_16x2048x7168 | h100-pcie | 0.0828 | workspace S=7 | 0.1302 | 0.1412 | 0.0703 | - | 0.8352 | 0.3122 | 1.178 | 0.012 |
| mm_4096x4096x4096 | h100-pcie | 3.6572 | none | 3.6912 | 6.4806 | 3.8168 | - | 4.2054 | 9.0823 | 0.958 | 0.001 |
| pt_rm_m128_d1024_h2048 | h100-pcie | 0.1247 | - | = | 0.1473 | 0.0703 | 0.0654 | 0.3283 | 0.1283 | 1.907 | 0.005 |
| pt_rm_m128_d256_h512 | h100-pcie | 0.0269 | - | = | 0.0343 | 0.0294 | 0.0236 | 0.0909 | 0.0253 | 1.140 | 0.015 |
| mm_128x2048x7168 | rtx-3090 | 0.3594 | atomic S=8 | 0.7649 | 0.4731 | 0.2181 | - | 0.9298 | 0.6502 | 1.648 | 0.003 |
| mm_16x1024x4096 | rtx-3090 | 0.0983 | atomic S=4 | 0.0983 | 0.0686 | 0.0369 | - | 0.4680 | 0.1372 | 2.664 | 0.010 |
| mm_16x18432x7168 | rtx-3090 | 1.0772 | none | 1.0916 | 1.1305 | 0.9943 | - | 3.0976 | 1.0691 | 1.083 | 0.001 |
| mm_16x2048x7168 | rtx-3090 | 0.1331 | workspace S=4 | 0.1485 | 0.1475 | 0.0901 | - | 0.8765 | 0.2488 | 1.477 | 0.008 |
| mm_4096x4096x4096 | rtx-3090 | 9.9420 | none | 9.8540 | 16.48 | 10.19 | - | 10.51 | 27.55 | 0.975 | 0.036 |
| pt_rm_m128_d1024_h2048 | rtx-3090 | 0.3082 | - | = | 0.1812 | 0.1034 | 0.0952 | 0.3543 | 0.1741 | 3.237 | 0.000 |
| pt_rm_m128_d256_h512 | rtx-3090 | 0.0696 | - | = | 0.0410 | 0.0307 | 0.0266 | 0.0993 | 0.0297 | 2.617 | 0.000 |

### Intended vs realized on the boxes (shipped programs; ok / MISMATCH / n-a)

| rung:state | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 |
|---|---|---|---|---|
| arch-rungs:on | 78 / 0 / 0 | 83 / 0 / 0 | 91 / 0 / 0 | 89 / 0 / 0 |
| arena:on | 49 / 0 / 0 | 49 / 0 / 0 | 49 / 0 / 0 | 49 / 0 / 0 |
| async:off | 30 / 0 / 0 | 68 / 0 / 0 | 83 / 0 / 0 | 80 / 0 / 0 |
| async:on | 77 / 1 / 0 | 12 / 0 / 0 | - | - |
| by-ref:on | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| by-ref:runtime | 145 / 0 / 0 | 145 / 0 / 0 | 145 / 0 / 0 | 145 / 0 / 0 |
| cell-fill:on | - | - | 8 / 0 / 0 | 5 / 0 / 0 |
| devgen:on | 52 / 0 / 0 | 52 / 0 / 0 | 52 / 0 / 0 | 52 / 0 / 0 |
| devmap:on | 2 / 0 / 0 | 2 / 0 / 0 | 2 / 0 / 0 | 2 / 0 / 0 |
| eager-init:on | 52 / 0 / 0 | 52 / 0 / 0 | 52 / 0 / 0 | 52 / 0 / 0 |
| epilogue:on | 15 / 0 / 0 | 15 / 0 / 0 | 15 / 0 / 0 | 15 / 0 / 0 |
| exec:on | 66 / 0 / 0 | 50 / 16 / 0 | 66 / 0 / 0 | 66 / 0 / 0 |
| face-fma:on | 106 / 2 / 0 | 124 / 2 / 0 | 130 / 2 / 0 | 134 / 2 / 0 |
| face-registers:on | 207 / 0 / 0 | 212 / 10 / 0 | 247 / 2 / 0 | 232 / 8 / 0 |
| face-separate:on | 48 / 0 / 0 | 40 / 0 / 0 | 50 / 0 / 0 | 42 / 0 / 0 |
| fuse-host:on | 18 / 0 / 0 | 18 / 0 / 0 | 18 / 0 / 0 | 18 / 0 / 0 |
| fuse:on | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 |
| grid-share:on | 55 / 0 / 0 | 74 / 0 / 0 | 42 / 4 / 0 | 10 / 0 / 0 |
| hostopt:on | 2 / 0 / 0 | 2 / 0 / 0 | 2 / 0 / 0 | 2 / 0 / 0 |
| idx32:on | 50 / 0 / 0 | 32 / 0 / 0 | 66 / 0 / 0 | 28 / 0 / 0 |
| interleave:on | 69 / 0 / 0 | 74 / 0 / 0 | 75 / 0 / 0 | 33 / 0 / 0 |
| min-blocks:off | - | - | 70 / 0 / 0 | - |
| min-blocks:on | 138 / 0 / 0 | 138 / 10 / 0 | 26 / 0 / 0 | 152 / 8 / 0 |
| module:on | 85 / 6 / 0 | 83 / 11 / 0 | 88 / 20 / 0 | 87 / 17 / 0 |
| panel-regs:on | 10 / 0 / 0 | 62 / 6 / 0 | 70 / 0 / 0 | 8 / 2 / 0 |
| pdl:off | - | 83 / 0 / 0 | 91 / 0 / 0 | - |
| prefetch:on | 30 / 0 / 0 | 68 / 0 / 0 | 64 / 19 / 0 | 71 / 9 / 0 |
| resident:on | 52 / 0 / 0 | 52 / 0 / 0 | 52 / 0 / 0 | 52 / 0 / 0 |
| row-fit:on | 39 / 0 / 0 | 68 / 0 / 0 | 13 / 0 / 0 | 47 / 0 / 0 |
| rowfold:on | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| splitk-atomic-zero:on | 2 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 | 7 / 0 / 0 |
| splitk-atomic:on | 8 / 0 / 0 | 4 / 0 / 0 | 2 / 0 / 0 | 24 / 0 / 0 |
| splitk:on | 4 / 0 / 0 | - | 12 / 0 / 0 | 8 / 0 / 0 |
| streamk:on | - | 27 / 0 / 0 | 39 / 0 / 0 | 3 / 0 / 0 |
| unroll:on | 138 / 0 / 0 | 148 / 0 / 0 | 166 / 0 / 0 | 160 / 0 / 0 |
| warp-fill:on | 69 / 0 / 0 | 46 / 0 / 0 | 69 / 0 / 0 | 43 / 0 / 0 |
| wide-fma:on | 60 / 0 / 0 | - | - | 48 / 0 / 0 |
| window:on | 4 / 2 / 0 | 4 / 2 / 0 | 4 / 2 / 0 | 4 / 2 / 0 |

### Mismatches on shipped programs

| arch | rung | check | programs | realized (first) |
|---|---|---|---|---|
| rtx-4090 | async:on | LDGSTS in the hot loop, no register-staged STS | pt_attn_1024_64 | sass.hot.ldgsts=0 sass.hot.sts=0 |
| rtx-5090 | exec:on | JIT image local bytes == ptxas stack frame | mm_16x1024x4096 mm_16x1024x4096_c mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_c mm_16x2048x7168_rc mm_4096x4096x4096_r mm_4096x4096x4096_x | exec.local=152 ptxas.stack=208 |
| rtx-5090 | exec:on | JIT image registers == ptxas -v | mm_16x18432x7168 mm_16x18432x7168_r mm_16x18432x7168_x mm_4096x4096x4096 pt_rm_m128_d1024_h2048_c pt_rm_m128_d1024_h2048_x pt_rm_m128_d256_h512_x | exec.regs=225 ptxas.regs=227 |
| h100-pcie | face-fma:on | PTX hot loop: fma.rn, no separate mul | pt_firew_rowsum_2048x2048_16 | ptx.hot.fma=16 ptx.hot.mul=1 |
| h100-pcie | face-fma:on | SASS hot loop: FFMA, no FMUL | pt_firew_rowsum_2048x2048_16 | sass.hot.ffma=16 sass.hot.fmul=1 |
| rtx-3090 | face-fma:on | PTX hot loop: fma.rn, no separate mul | pt_firew_rowsum_2048x2048_16 | ptx.hot.fma=16 ptx.hot.mul=1 |
| rtx-3090 | face-fma:on | SASS hot loop: FFMA, no FMUL | pt_firew_rowsum_2048x2048_16 | sass.hot.ffma=16 sass.hot.fmul=1 |
| rtx-4090 | face-fma:on | PTX hot loop: fma.rn, no separate mul | pt_firew_rowsum_2048x2048_16 | ptx.hot.fma=16 ptx.hot.mul=1 |
| rtx-4090 | face-fma:on | SASS hot loop: FFMA, no FMUL | pt_firew_rowsum_2048x2048_16 | sass.hot.ffma=16 sass.hot.fmul=1 |
| rtx-5090 | face-fma:on | PTX hot loop: fma.rn, no separate mul | pt_firew_rowsum_2048x2048_16 | ptx.hot.fma=16 ptx.hot.mul=1 |
| rtx-5090 | face-fma:on | SASS hot loop: FFMA, no FMUL | pt_firew_rowsum_2048x2048_16 | sass.hot.ffma=16 sass.hot.fmul=1 |
| h100-pcie | face-registers:on | the derived blocks/SM are resident at ptxas registers + smem | pt_attn_2048_128 pt_attn_2048_64 | occ.blocks_per_sm=1 plan.blocks_per_sm=2 |
| rtx-3090 | face-registers:on | no spill | mm_128x2048x7168_rc mm_16x1024x4096_r mm_16x1024x4096_x pt_attn_2048_128 pt_attn_2048_64 pt_loopmm_1024_8 pt_rt_mlp2_1024 pt_smlp_4096_256_1024 | ptxas.spill_st=24 ptxas.spill_ld=24 |
| rtx-5090 | face-registers:on | no spill | mm_16x1024x4096 mm_16x1024x4096_c mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_c mm_16x2048x7168_rc mm_4096x4096x4096_r mm_4096x4096x4096_x pt_attn_2048_128 pt_attn_2048_64 | ptxas.spill_st=208 ptxas.spill_ld=208 |
| h100-pcie | grid-share:on | residency sized to the grid share | mm_16x1024x4096 mm_16x1024x4096_c mm_16x1024x4096_rc | plan.blocks_per_sm=6 plan.ctas=0 plan.combine=none plan.gx=64 plan.gy=1 occ.sms=114 |
| rtx-3090 | min-blocks:on | ptxas met it without spilling | mm_128x2048x7168_rc mm_16x1024x4096_r mm_16x1024x4096_x pt_attn_2048_128 pt_attn_2048_64 pt_loopmm_1024_8 pt_rt_mlp2_1024 pt_smlp_4096_256_1024 | ptxas.spill_st=24 |
| rtx-5090 | min-blocks:on | ptxas met it without spilling | mm_16x1024x4096 mm_16x1024x4096_c mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_c mm_16x2048x7168_rc mm_4096x4096x4096_r mm_4096x4096x4096_x pt_attn_2048_128 pt_attn_2048_64 | ptxas.spill_st=208 |
| h100-pcie | module:on | every kernel in the device module is launched by the host | mm_128x2048x7168 mm_128x2048x7168_r mm_16x1024x4096 mm_16x1024x4096_r mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_rc pt_attn_1024_128 pt_attn_1024_64 pt_attn_2048_128 pt_attn_2048_64 pt_decode_1024_256 pt_loopmm_1024_8 pt_rt_mlp2_1024 pt_smlp_4096_256_1024 | mod.launched=0 |
| rtx-3090 | module:on | every kernel in the device module is launched by the host | mm_128x2048x7168 mm_128x2048x7168_rc mm_16x1024x4096 mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_rc pt_attn_1024_128 pt_attn_1024_64 pt_attn_2048_128 pt_attn_2048_64 pt_decode_1024_256 pt_loopmm_1024_8 pt_rt_mlp2_1024 pt_smlp_4096_256_1024 | mod.launched=0 |
| rtx-4090 | module:on | every kernel in the device module is launched by the host | mm_128x2048x7168_r mm_16x1024x4096_r pt_attn_1024_128 pt_attn_1024_64 pt_decode_1024_256 | mod.launched=0 |
| rtx-5090 | module:on | every kernel in the device module is launched by the host | mm_16x18432x7168 mm_4096x4096x4096 pt_decode_1024_256 pt_diamond_4096 pt_fanout_4096 pt_indep_4096 pt_rowsum_2048 pt_rowsum_4096 pt_softmax_rows_2048 pt_softmax_rows_4096 | mod.launched=0 |
| rtx-3090 | panel-regs:on | the prefetch panel fits the register file: no spill | mm_16x1024x4096_r mm_16x1024x4096_x | ptxas.spill_st=140 |
| rtx-5090 | panel-regs:on | the prefetch panel fits the register file: no spill | mm_16x1024x4096 mm_16x1024x4096_c mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_c mm_16x2048x7168_rc | ptxas.spill_st=208 |
| h100-pcie | prefetch:on | the next panel's LDGs are covered by this panel's math | mm_16x18432x7168 mm_16x18432x7168_c mm_16x18432x7168_r mm_16x18432x7168_rc mm_16x18432x7168_x mm_4096x4096x4096_r mm_4096x4096x4096_x pt_attn_1024_128 pt_attn_1024_64 pt_attn_2048_128 pt_attn_2048_64 pt_rm_m128_d1024_h2048 pt_rm_m128_d1024_h2048_c pt_rm_m128_d256_h512 pt_rm_m128_d256_h512_c | sass.hot.ldg_cover_min=0 |
| rtx-3090 | prefetch:on | the next panel's LDGs are covered by this panel's math | mm_16x1024x4096_r mm_16x1024x4096_x mm_4096x4096x4096_r mm_4096x4096x4096_x pt_attn_2048_128 pt_attn_2048_64 pt_rm_m128_d256_h512_x | sass.hot.ldg_cover_min=0 |
| h100-pcie | window:on | LLVM fully unrolled the constant-trip tap walk (loop-unroll FullyUnrolled) | pt_conv_2048_5 pt_ewconvew_2048_5 | remarks.opt.full_unroll=0 |
| rtx-3090 | window:on | LLVM fully unrolled the constant-trip tap walk (loop-unroll FullyUnrolled) | pt_conv_2048_5 pt_ewconvew_2048_5 | remarks.opt.full_unroll=0 |
| rtx-4090 | window:on | LLVM fully unrolled the constant-trip tap walk (loop-unroll FullyUnrolled) | pt_conv_2048_5 pt_ewconvew_2048_5 | remarks.opt.full_unroll=0 |
| rtx-5090 | window:on | LLVM fully unrolled the constant-trip tap walk (loop-unroll FullyUnrolled) | pt_conv_2048_5 pt_ewconvew_2048_5 | remarks.opt.full_unroll=0 |

### Findings, ranked by measured time lost (the build as it ships now; main boxes, the 3090 on S65's host)

| # | workload | ms lost, Σ 4 parts | Mapal / best other (4090 / 5090 / H100 / 3090) | explained by |
|---|---|---|---|---|
| 1 | decode_1024_256 | 118.3 | 1.99 / 1.27 / 3.69 / 1.77 (cache / no-cache harness) | 4096 one-block launches (`gx` 1–4, 7–17 µs traced each) + 2 h2d / 1 d2h synchronous copies per step (4090: 8.9 + 9.5 ms); the harness: 6 async launches per step |
| 2 | rt_mlp2_1024 | 51.5 | 13.55 / 10.75 / 3.51 / 3.22 (compile) | traced copy + host ms 20.3 / 14.8 / 5.1 (4090 / 5090 / H100): 68 MB host-made request inputs per run; 5.49 on the Ryzen 9 4090 host |
| 3 | loop_2048 | 2.7 | 1.14 / 1.19 / 1.26 / 1.15 (cuBLASLt / CUTLASS) | the 2048³ FMA tile per step |
| 4 | ew_2048, ew_4096 | 1.7, 1.6 | 6.7–15.3, 2.3–5.0 (fused kernel) | traced host 0.35–0.66 ms of the run on EPYC hosts; 0.22 ms runs on the Ryzen 9 hosts |
| 5 | softmax_rows_4096 / 2048 | 1.7, 0.9 | 1.07–1.18, 1.37–1.60 (compile) | two thread-per-row rowfold kernels, 0.08–0.22 ms each at 4096 rows |
| 6 | attention, 4 sizes | 0.7–1.1 each | 2.85–8.66 (compile) | at 2048 × 128 the kernels take 0.21–0.54 ms (strided `q·Kᵀ` tile, two rowfolds, two cells, `p·V`); compile 0.08–0.12 ms on the 4090 / 5090 / H100 |
| - | `flow` unit | (shipped OFF) | unit kernel 1.79–4.82× the kernels it replaces | one thread per row; d = 128: 6 `div.rn.f32` in the unit against 2 at d = 64 (÷11.3 vs ÷8.0 → × 0.125) |
| - | boxes | - | 3090 main 220 W (cuBLAS 4096³ 10.06 vs 5.93 ms at 350 W); a 3090 at ~27 % speed and a 5090 that lost clock mid-run (`confirm-throttled/`) | same-box ratios only; S65's own hosts re-rented for the 3090 and H100 second boxes |

### Tests

| test | result |
|---|---|
| `MAPAL_REQUIRE_CLANG=1 cargo test --workspace --release --no-fail-fast -- --test-threads=1` (with the flow rows) | 1254 passed, 0 failed, 3 ignored (93 binaries) |
| `generic_fusion.rs` (`flow` forced on: the rung's proof of values, launches, stores) | 8 passed, 2 ignored |
| value gates (default face within ADR-0040 D2's bound, every arm, before timing) | every box: 25 / 25 programs; attention re-take 4 / 4 |
| chain_bench attention legs on the first H100 / 5090 boxes | gate FAIL: the T×T `p` buffer was main()'s d×d operand `p` (fixed `233c49d`, re-taken in `attn-confirm/`) |

## fusion-natural-forms: generic fusion and natural forms, integrated (static)

Raw: [`benches/results/evidence/fusion-natural-forms/`](../../../benches/results/evidence/fusion-natural-forms/) ·
tables: `python3 benches/gpu/fusion_natural_report.py benches/results/evidence/fusion-natural-forms` · branch
`fusion-natural-forms` (generic-fusion + natural-forms). No GPU was rented: ptxas 12.8 / cuobjdump through
`cudatool.sh`, LLVM 22 remarks, values through the emulator. base = 27b4f99 (s65/residency) on its own programs;
off = the merged build with `flow`, `window`, `range-map` off; shipped = every rung on.

### Decisions as merged (static); measured in the sweep above

| rung | OFF arm | static, OFF → shipped | intent checks, ok / mismatch (4090 / 5090 / H100 / 3090) | time |
|---|---|---|---|---|
| `flow` | `--gpu-off=fuse` | softmax_rows: 5 → 2 launches, 7 → 4 device buffers; attention: 6 → 3, 9 → 6; mx, ex, den never stored | `flow` 50/0, 40/0, 50/0, 75/0; `fuse-host` 30/0, 24/0, 30/0, 45/0 | OFF on all four (sweep) |
| `window` | `--gpu-off=window` | conv, ewconvew: the 5×5 conv leaves the host: 0 → 1 launch (`taps=5x5`), host insns 4567 → 78 | `window` 4/2 on every part (finding 3) | ON on all four (sweep) |
| `range-map` | `--gpu-off=range-map` | decode: launches, buffers, per-trip bytes equal; the row write runs over `[256t, min(256t+256, 262144))`, not `[0, 262144)` | no static predicate (host loop bound, by inspection); `by-ref` 61/0, 61/0, 73/0, 69/0 | ON on all four (sweep) |
| `lane-stride` (S54, now also 2-D `q·Kᵀ`) | `--gpu-off=lane-stride` | attention: 1 → 3 launches, whole-program h2d 4 / 16 MB → 0, host insns ~590 → ~120 | its kernels carry the new mismatches (finding 4) | ON on all four (sweep) |

### Per program class: base → off → shipped

| class (programs) | launches | device buffers | kept (never stored) / stored unit members | h2d / d2h, whole program | per trip h2d / d2h / host copy, facts = realized | `fuse.host` launches intent = realized |
|---|---|---|---|---|---|---|
| softmax rows (2048, 4096) | 5 → 5 → 2 | 7 → 7 → 4 | mx, ex, den / y | 0 / 8 B → 0 / 8 B | - | ok, all four |
| attention (1024, 2048 × 64, 128) | 1 → 6 → 3 | 3 → 9 → 6 | mx, ex, den / p | 4 or 16 MB / 8 B → 0 / 8 B | - | ok, all four |
| conv, ew → conv → ew | 0 → 0 → 1 | 0 → 0 → 3 | - | 0 / 0 → 0 / 8 B | - | ok |
| decode 1024 × 256 | 6 → 6 → 6 (H100 5) | 8 → 8 → 8 (H100 9) | - | 3 MB / 1 MB, equal | 2 KB / 1 KB / 0 = facts, equal | ok |
| decode, CPU-only | - | - | - | - | 0 / 0 / 0 = facts | host insns 18205 → 1619 |
| FIR → ew → rowsum, MLP (smlp, rm, rt_mlp2), rowsum, FIR, ew, fan-out, diamond, indep, loop, loopmm | unchanged | unchanged | - | unchanged | unchanged | host IR, PTX, CPU host IR identical to base up to kernel names, all four parts (`base/identity.txt`) |

CPU-only builds (x86-64-v3, apple-m): host IR byte-identical to natural-forms' emitter on all 25 programs, and
to generic-fusion's on all but decode (its range map) (`cpu_identity.txt`): both parts' CPU A/B timings stand, no
re-run.

Unit kernels (shipped, all four parts): `mx:rowfold;ex:cell;den:rowfold;y|p:cell`, one thread per row, kept members
in registers, `smem=0`; ptxas: no spill, no call left, pointers = stored + operands (`flow` row); device buffers
off-fuse − shipped = the 3 kept members on every fused program and part.

### Natural vs unrolled on the merged build (`twin/kcmp.txt`)

| program | kernels identical by content | per-trip movement | host object insns, unrolled → natural |
|---|---|---|---|
| decode 1024 × 256 | 8 / 8 / 9 / 9 of 8 / 8 / 9 / 9 | same | 6432 → 503 (4090), 6430 → 500 (H100), 6440 → 510 (3090) |
| softmax_rows 2048, 4096 (unit kernel with the exp fold inside) | all, all four parts | same | 95 → 95 |
| attention 1024 × 64, 2048 × 128 | all, all four parts | same | 122 → 122, 119 → 119, 125 → 125 |

### Intent check totals (`static/<arch>/evidence.tsv`)

| | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 |
|---|---|---|---|---|
| checks / mismatch (every arm) | 2223 / 24 | 2444 / 35 | 2883 / 90 | 2737 / 100 |
| mismatches on shipped programs | 9 | 17 | 31 | 30 |
| of those on kernels new this sprint (strided 2-D tiles, window) | 5 | 6 | 11 | 10 |
| the rest: classes present in the base build (unlaunched twin kernels, FIR epilogue mul, prefetch cover, 3090 spills) | 4 | 11 | 20 | 20 |

### Findings

| # | finding | evidence | status |
|---|---|---|---|
| 1 | an in-place range map over an init another loop still reads wrote that init (`roots()` saw the in-place map as unowned, so no entry copy) | chain property test #158, #182, both `flow` arms; n3[0] −13.44 vs −7.22; the `Update` twin right | fixed `815c3c3`, `natural_forms.rs` regression test |
| 2 | a unit kernel called the nested fold body of a natural-form cell without carrying it | pt_attn_1024_64: `opt` "use of undefined value @fn8" | fixed `e8b45d8`; the fusion fixtures now use the natural exp |
| 3 | `window`: LLVM only partially unrolls the 25-tap walk; ptxas completes it (25 FFMA, 0 div/rem) | `remarks.opt.full_unroll=0`, `PartialUnrolled`; SASS ffma 25, ldg 50 | open: two counted loops (design §4) |
| 4 | strided 2-D `q·Kᵀ` tiles keep `SeparateMulAdd` on the default face (`Arithmetic::scalar`: `b.clane != 1`), and at 8 × 8 spill (5090 12 B, 3090 8 B), miss occupancy (H100 1 < 2) and prefetch cover (H100, 3090) | `newmm` rows in `static/<arch>/evidence.tsv` | open: judge with the `.off-lane-stride` arm on the boxes |
| 5 | a sub-row range write that reads another device array copies all of it out per trip | Part A's gap program: d2h 128 B vs 16 B for the `Update`s | open: `footprint_plan`'s rect gives the 16 B |
| 6 | the unit evaluates `ex` once per reader (den's phase, then p's / y's) | attention unit kernel: two calls of ex's body (`@fn9_u1`), one of p's | open: shared-memory staging in units |
| 7 | `flow-tile` (tile member, root re-derived) not built: FIR → ew → rowsum, MLP keep their launches | 2 → 2 | open |

### Tests

| test | result |
|---|---|
| `generic_fusion_chains_values_launches_and_stores` (256 chains, emulated 4090 / H100) | 244 with a device unit; 96 units of 202 members, 126 kept, 33 with a halo; 106 launches removed; refusals footprint 6, phase 7; exact face = interpreter ON and OFF, default face ON = OFF (16) |
| `differential_natural_twins_values_and_movement` (256 twins) | 51 `Rows` claims, all probed sound; 209 device builds, 181 in-place states; movement, launches, kernels = the unrolled twins' (62 better); values ok with `range-map` on and off |
| `cargo test -p mapal-ir` / `-p mapal-backend-llvm` / `-p mapal-cli` (debug, `6a3ee1d`) | 226 / 283 (3 ignored) / 5 passed, 0 failed |
| `MAPAL_REQUIRE_CLANG=1 cargo test --workspace --release --no-fail-fast -- --test-threads=1` | 1254 passed, 0 failed, 3 ignored (93 binaries) at `6a3ee1d`; at `815c3c3` 1253 / 1: `natural_forms::a_fold_over_a_range_stays_in_the_cell` asked for a cell kernel, `flow` makes `ex` a unit member (check widened in `6a3ee1d`) |

## S65: loop state by reference (rule R)

Raw: [`benches/results/evidence/s65/`](../../../benches/results/evidence/s65/) · tables:
`python3 benches/gpu/evidence_report.py benches/results/evidence/s65`,
`python3 benches/gpu/movement_report.py benches/results/evidence/s65`,
`python3 benches/gpu/s65_arms.py benches/results/evidence/s65/<arch>/results-<arch>.txt`.
S64 arm = `--gpu-off=by-ref` (the S64 build). Boxes: 30 interleaved reps per arm, one process each, null = max over reps
of |null / shipped − 1|. Every value gate passed (default face within its bound) on all four parts.

### Decisions

| GPU_RUNGS_OFF row | part | rung | change | evidence (static) | measured (ON / OFF, null) |
|---|---|---|---|---|---|
| new | all four | by-ref | **ON** | host memcpy 7–11 → 0 per program; by-ref checks 302 / 337 / 329 / 320 ok, 0 mismatch (4090 / 5090 / H100 / 3090) | below: 26 of 28 GPU cells ON faster outside null; rt_mlp2 inside null on 5090 (0.791) and H100 (0.812) |
| 4090 retain-resident | rtx-4090 | retain-resident | **removed: ON** | retained 0 → 2, + one-row kernel `mapal_tile_fee5d054184d5db4` (1×256×256), launch sites 4 → 6 | decode 40.58 vs 33.05 ms, 1.214 (per rep 1.192..1.252), null 0.050; traced K/V site 2048 × 22.34 ms → 2046 × 14.94 ms |
| 5090 retain-resident | rtx-5090 | retain-resident | **removed: ON** | retained 0 → 2, + `mapal_tile_e2f8ae468c059815`, launch sites 4 → 6 | decode 42.71 vs 36.69 ms, 1.167 (1.122..1.191), null 0.037; 2048 × 23.69 → 2046 × 17.49 ms |
| H100 retain-resident | h100-pcie | retain-resident | none | retained 0 → 2, + `mapal_tile_3881eb4bbf2d9cb5`, launch sites 5 → 7 | decode 87.42 vs 86.14 ms, 1.008 (0.946..1.139) inside null 0.094; 2048 × 45.83 → 2046 × 45.18 ms |
| (ships ON) | rtx-3090 | retain-resident | none | OFF arm: retained 2 → 0 | decode 54.05 vs 149.24 ms OFF, 2.747 (2.525..2.915), null 0.088; OFF 2048 × 99.89 ms → ON 2046 × 23.24 ms |

End check: the new shipped decode build on the 4090 / 5090 is byte-identical (host.ll, dev.ll) to the measured
`on-retain-resident` arm; H100 / 3090 unchanged.

### Rule R: facts → decisions

| fact (mapal-ir) | decision (existing storage layer: `frame.rs` host, `resident.rs` device) | state mode |
|---|---|---|
| `LastUsePlan::view_pos` / `view_death` / `view_dead_after` (new: the use positions with `Pair` / `Phi` / `Proj` / loop-enter pins; `algos.rs`) | a value only forwarded (`Pair`, `Proj`, `Phi`, loop enter / back / exit) is a view: its address moves, not its bytes | all |
| `loop_plan` (merge, init, back route, exits) + `view_last_use` over every buffer the state may be | interval coloring across the back edge: next state first written at W, old state last read at D — D < W one buffer, else ping-pong (one more per extra position) | `one`, `two` |
| `Update` + `view_dead_after` + parallel-entry task order (every old-value read before the write) | the row write lands in the state's own buffer: the init's when the init dies at the entry, else one entry copy | `inplace` |
| resident carries (`loop_plan`, `escapes`, `death`, `write_delta`) | a device-carried state has no host bytes; its init is made on the device | `device` |
| a consumer that reads the product whole (`Output`, a by-value body input) | bytes | `value` (S64 arm) |

### Movement: intended (facts, `; mapal.move`) vs realized (host IR), per trip

| program | build | archs | state (position:bytes:carried) | host copy / trip: facts → realized (calls) | h2d / trip: facts → realized | d2h / trip: facts → realized | MB copied per trip, S64 arm → this |
|---|---|---|---|---|---|---|---|
| pt_loop_2048 | CPU+GPU | all four | 0:16777216:device | 0 → 0 (0) | 0 → 0 | 0 → 0 | 80.000 → 0 |
| pt_loop_2048 | CPU+GPU, S64 arm | all four | 0:16777216:device | 0 → 80.000 (5) | 0 → 0 | 0 → 0 | - |
| pt_loop_2048 | CPU-only | all four | 0:16777216:one | 0 → 0 (0) | 0 → 0 | 0 → 0 | 80.000 → 0 |
| pt_loop_2048 | CPU-only, S64 arm | all four | 0:16777216:value | 0 → 80.000 (5) | 0 → 0 | 0 → 0 | - |
| pt_loop_4096 | CPU+GPU | all four | 0:67108864:device | 0 → 0 (0) | 0 → 0 | 0 → 0 | 320.000 → 0 |
| pt_loop_4096 | CPU+GPU, S64 arm | all four | 0:67108864:device | 0 → 320.000 (5) | 0 → 0 | 0 → 0 | - |
| pt_loop_4096 | CPU-only | all four | 0:67108864:one | 0 → 0 (0) | 0 → 0 | 0 → 0 | 320.000 → 0 |
| pt_loop_4096 | CPU-only, S64 arm | all four | 0:67108864:value | 0 → 320.000 (5) | 0 → 0 | 0 → 0 | - |
| pt_loopmm_1024_8 | CPU+GPU | all four | 0:4194304:device | 0 → 0 (0) | 0 → 0 | 0 → 0 | 20.000 → 0 |
| pt_loopmm_1024_8 | CPU+GPU, S64 arm | all four | 0:4194304:device | 0 → 20.000 (5) | 0 → 0 | 0 → 0 | - |
| pt_loopmm_1024_8 | CPU-only | all four | 0:4194304:two | 0 → 0 (0) | 0 → 0 | 0 → 0 | 20.000 → 0 |
| pt_loopmm_1024_8 | CPU-only, S64 arm | all four | 0:4194304:value | 0 → 20.000 (5) | 0 → 0 | 0 → 0 | - |
| pt_rt_mlp2_1024 | CPU+GPU | all four | 0:4194304:device | 0 → 0 (0) | 8.000 → 8.000 | 0 → 0 | 20.000 → 0 |
| pt_rt_mlp2_1024 | CPU+GPU, S64 arm | all four | 0:4194304:value | 0 → 20.000 (5) | 8.000 → 8.000 | 4.000 → 4.000 | - |
| pt_rt_mlp2_1024 | CPU-only | all four | 0:4194304:one | 0 → 0 (0) | 0 → 0 | 0 → 0 | 20.000 → 0 |
| pt_rt_mlp2_1024 | CPU-only, S64 arm | all four | 0:4194304:value | 0 → 20.000 (5) | 0 → 0 | 0 → 0 | - |
| pt_decode_1024_256 | CPU+GPU | all four | 0:1048576:inplace,1:1024:one | 0 → 0 (0) | 2.0 K → 2.0 K | 1.0 K → 1.0 K | 4.005 → 0 |
| pt_decode_1024_256 | CPU+GPU, S64 arm | all four | 0:1048576:value,1:1024:value | 0 → 4.005 (7) | 2.0 K → 2.0 K | 1.0 K → 1.0 K | - |
| pt_decode_1024_256 | CPU-only | all four | 0:1048576:inplace,1:1024:one | 0 → 0 (0) | 0 → 0 | 0 → 0 | 4.005 → 0 |
| pt_decode_1024_256 | CPU-only, S64 arm | all four | 0:1048576:value,1:1024:value | 0 → 4.005 (7) | 0 → 0 | 0 → 0 | - |

Host IR `llvm.memcpy` call sites, S64 arm → shipped:

| program | build | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 |
|---|---|---|---|---|---|
| pt_loop_2048 | CPU+GPU | 8 → 0 | 8 → 0 | 8 → 0 | 8 → 0 |
| pt_loop_2048 | CPU-only | 7 → 0 | 7 → 0 | 7 → 0 | 7 → 0 |
| pt_loop_4096 | CPU+GPU | 8 → 0 | 8 → 0 | 8 → 0 | 8 → 0 |
| pt_loop_4096 | CPU-only | 7 → 0 | 7 → 0 | 7 → 0 | 7 → 0 |
| pt_loopmm_1024_8 | CPU+GPU | 8 → 0 | 8 → 0 | 8 → 0 | 8 → 0 |
| pt_loopmm_1024_8 | CPU-only | 7 → 0 | 7 → 0 | 7 → 0 | 7 → 0 |
| pt_decode_1024_256 | CPU+GPU | 11 → 0 | 11 → 0 | 11 → 0 | 11 → 0 |
| pt_decode_1024_256 | CPU-only | 10 → 0 | 10 → 0 | 10 → 0 | 10 → 0 |
| pt_rt_mlp2_1024 | CPU+GPU | 8 → 0 | 8 → 0 | 8 → 0 | 8 → 0 |
| pt_rt_mlp2_1024 | CPU-only | 7 → 0 | 7 → 0 | 7 → 0 | 7 → 0 |

### by-ref in time: S65 vs its S64 arm, same box

| program | arch | build | S64 arm ms | S65 ms | S65 / S64 arm per rep: median (min..max) | null | traced residue ms, S64 arm → S65 | MB copied, S64 arm → S65 |
|---|---|---|---|---|---|---|---|---|
| pt_loop_2048 | rtx-4090 | CPU+GPU | 39.6508 | 2.5753 | 0.065 (0.061..0.104) | 0.333 | 35.34 → 0.00 | 16.0 → 0.0 |
| pt_loop_2048 | rtx-4090 | CPU-only | 1042.8764 | 941.8409 | 0.904 (0.901..0.911) | - | - | - |
| pt_loop_4096 | rtx-4090 | CPU+GPU | 179.6650 | 19.6459 | 0.110 (0.107..0.144) | 0.312 | 157.88 → 0.00 | 64.0 → 0.0 |
| pt_loop_4096 | rtx-4090 | CPU-only | 8722.8409 | 8417.0680 | 0.966 (0.957..0.971) | - | - | - |
| pt_loopmm_1024_8 | rtx-4090 | CPU+GPU | 7.4878 | 0.5312 | 0.071 (0.067..0.074) | 0.092 | 6.78 → 0.01 | 4.0 → 0.0 |
| pt_loopmm_1024_8 | rtx-4090 | CPU-only | 142.2432 | 119.0220 | 0.832 (0.806..0.847) | - | - | - |
| pt_rt_mlp2_1024 | rtx-4090 | CPU+GPU | 21.9560 | 8.7197 | 0.399 (0.367..0.405) | 0.041 | 14.17 → 1.77 | 96.0 → 68.0 |
| pt_rt_mlp2_1024 | rtx-4090 | CPU-only | 532.4533 | 510.9250 | 0.962 (0.937..0.972) | - | - | - |
| pt_decode_1024_256 | rtx-4090 | CPU+GPU | 247.6295 | 40.5796 | 0.161 (0.155..0.176) | 0.050 | 209.85 → 0.00 | 4.0 → 4.0 |
| pt_decode_1024_256 | rtx-4090 | CPU-only | 2135.4831 | 1916.3378 | 0.900 (0.895..0.906) | - | - | - |
| pt_softmax_rows_2048 | rtx-4090 | CPU+GPU | 4.9775 | 0.5157 | 0.103 (0.098..0.108) | 0.034 | 3.48 → 0.00 | 16.0 → 0.0 |
| pt_softmax_rows_4096 | rtx-4090 | CPU+GPU | 21.8570 | 3.5187 | 0.161 (0.153..0.163) | 0.004 | 12.99 → 0.00 | 64.0 → 0.0 |
| pt_loop_2048 | rtx-5090 | CPU+GPU | 38.4585 | 2.1838 | 0.057 (0.051..0.059) | 0.034 | 35.11 → 0.00 | 16.0 → 0.0 |
| pt_loop_2048 | rtx-5090 | CPU-only | 1103.6484 | 1050.2771 | 0.947 (0.933..0.958) | - | - | - |
| pt_loop_4096 | rtx-5090 | CPU+GPU | 155.0005 | 15.8282 | 0.102 (0.099..0.104) | 0.038 | 134.81 → 0.00 | 64.0 → 0.0 |
| pt_loop_4096 | rtx-5090 | CPU-only | 8833.3068 | 8553.2154 | 0.970 (0.951..0.984) | - | - | - |
| pt_loopmm_1024_8 | rtx-5090 | CPU+GPU | 8.9816 | 0.4818 | 0.055 (0.035..0.063) | 0.147 | 9.08 → 0.00 | 4.0 → 0.0 |
| pt_loopmm_1024_8 | rtx-5090 | CPU-only | 137.4616 | 129.0076 | 0.953 (0.911..1.002) | - | - | - |
| pt_rt_mlp2_1024 | rtx-5090 | CPU+GPU | 20.7864 | 7.5898 | 0.370 (0.337..0.393) | 0.791 | 12.14 → 1.45 | 96.0 → 68.0 |
| pt_rt_mlp2_1024 | rtx-5090 | CPU-only | 563.1604 | 555.0901 | 0.986 (0.964..1.017) | - | - | - |
| pt_decode_1024_256 | rtx-5090 | CPU+GPU | 249.0487 | 42.7149 | 0.172 (0.157..0.178) | 0.037 | 211.16 → 0.00 | 4.0 → 4.0 |
| pt_decode_1024_256 | rtx-5090 | CPU-only | 2361.2873 | 2171.9823 | 0.918 (0.912..0.929) | - | - | - |
| pt_softmax_rows_2048 | rtx-5090 | CPU+GPU | 5.1143 | 0.3732 | 0.073 (0.054..0.079) | 0.039 | 5.52 → 0.00 | 16.0 → 0.0 |
| pt_softmax_rows_4096 | rtx-5090 | CPU+GPU | 20.9195 | 2.3396 | 0.112 (0.107..0.119) | 0.081 | 13.52 → 0.00 | 64.0 → 0.0 |
| pt_loop_2048 | h100-pcie | CPU+GPU | 35.6640 | 6.4506 | 0.180 (0.161..0.188) | 0.019 | 25.95 → 0.02 | 16.0 → 0.0 |
| pt_loop_2048 | h100-pcie | CPU-only | 1645.9832 | 1520.3636 | 0.927 (0.646..1.095) | - | - | - |
| pt_loop_4096 | h100-pcie | CPU+GPU | 143.5566 | 36.2815 | 0.253 (0.233..0.265) | 0.006 | 103.51 → 0.05 | 64.0 → 0.0 |
| pt_loop_4096 | h100-pcie | CPU-only | 15432.5961 | 14689.9527 | 0.969 (0.900..1.080) | - | - | - |
| pt_loopmm_1024_8 | h100-pcie | CPU+GPU | 7.9867 | 1.0944 | 0.139 (0.122..0.213) | 0.333 | 6.28 → 0.59 | 4.0 → 0.0 |
| pt_loopmm_1024_8 | h100-pcie | CPU-only | 212.0991 | 179.6344 | 0.847 (0.814..0.926) | - | - | - |
| pt_rt_mlp2_1024 | h100-pcie | CPU+GPU | 24.9280 | 20.5389 | 0.819 (0.400..0.860) | 0.812 | 13.34 → 3.30 | 96.0 → 68.0 |
| pt_rt_mlp2_1024 | h100-pcie | CPU-only | 787.6105 | 767.1414 | 0.962 (0.939..1.025) | - | - | - |
| pt_decode_1024_256 | h100-pcie | CPU+GPU | 275.9214 | 87.4242 | 0.316 (0.291..0.330) | 0.094 | 186.91 → 0.00 | 4.0 → 4.0 |
| pt_decode_1024_256 | h100-pcie | CPU-only | 3114.7613 | 2932.3586 | 0.950 (0.805..0.988) | - | - | - |
| pt_softmax_rows_2048 | h100-pcie | CPU+GPU | 16.1430 | 1.0563 | 0.066 (0.052..0.103) | 0.057 | 4.18 → 0.00 | 16.0 → 0.0 |
| pt_softmax_rows_4096 | h100-pcie | CPU+GPU | 57.2642 | 5.1955 | 0.091 (0.076..0.108) | 0.009 | 23.98 → 0.00 | 64.0 → 0.0 |
| pt_loop_2048 | rtx-3090 | CPU+GPU | 51.9696 | 7.0657 | 0.136 (0.131..0.153) | 0.011 | 44.41 → 0.11 | 16.0 → 0.0 |
| pt_loop_2048 | rtx-3090 | CPU-only | 2031.1519 | 1997.4528 | 0.973 (0.963..1.030) | - | - | - |
| pt_loop_4096 | rtx-3090 | CPU+GPU | 193.6354 | 46.0564 | 0.240 (0.228..0.245) | 0.039 | 142.87 → 0.30 | 64.0 → 0.0 |
| pt_loop_4096 | rtx-3090 | CPU-only | 18030.4508 | 17627.4130 | 0.980 (0.961..0.987) | - | - | - |
| pt_loopmm_1024_8 | rtx-3090 | CPU+GPU | 16.1796 | 1.6191 | 0.100 (0.094..0.104) | 0.059 | 14.35 → 0.06 | 4.0 → 0.0 |
| pt_loopmm_1024_8 | rtx-3090 | CPU-only | 264.0542 | 238.1217 | 0.902 (0.895..0.912) | - | - | - |
| pt_rt_mlp2_1024 | rtx-3090 | CPU+GPU | 39.5315 | 14.4858 | 0.360 (0.334..0.375) | 0.024 | 28.68 → 3.98 | 96.0 → 68.0 |
| pt_rt_mlp2_1024 | rtx-3090 | CPU-only | 1051.3420 | 1028.2151 | 0.979 (0.975..0.998) | - | - | - |
| pt_decode_1024_256 | rtx-3090 | CPU+GPU | 165.3107 | 54.0491 | 0.325 (0.311..0.330) | 0.088 | 103.10 → 0.00 | 4.0 → 4.0 |
| pt_decode_1024_256 | rtx-3090 | CPU-only | 4119.9114 | 3890.9013 | 0.948 (0.939..1.028) | - | - | - |
| pt_softmax_rows_2048 | rtx-3090 | CPU+GPU | 8.4470 | 1.1036 | 0.131 (0.127..0.133) | 0.009 | 5.74 → 0.00 | 16.0 → 0.0 |
| pt_softmax_rows_4096 | rtx-3090 | CPU+GPU | 36.4531 | 6.3963 | 0.176 (0.172..0.180) | 0.005 | 24.02 → 0.00 | 64.0 → 0.0 |

Per rung (evidence_report):

| rung | arch | shipped | cells: ON/OFF (null) | ON faster / slower / inside null |
|---|---|---|---|---|
| by-ref | h100-pcie | shipped ON | pt_decode_1024_256 0.316 (0.094); pt_loop_2048 0.180 (0.019); pt_loop_4096 0.253 (0.006); pt_loopmm_1024_8 0.139 (0.333); pt_rt_mlp2_1024 0.819 (0.812); pt_softmax_rows_2048 0.066 (0.057); pt_softmax_rows_4096 0.091 (0.009) | 6 / 0 / 1 |
| by-ref | rtx-3090 | shipped ON | pt_decode_1024_256 0.325 (0.088); pt_loop_2048 0.136 (0.011); pt_loop_4096 0.240 (0.039); pt_loopmm_1024_8 0.100 (0.059); pt_rt_mlp2_1024 0.360 (0.024); pt_softmax_rows_2048 0.131 (0.009); pt_softmax_rows_4096 0.176 (0.005) | 7 / 0 / 0 |
| by-ref | rtx-4090 | shipped ON | pt_decode_1024_256 0.161 (0.050); pt_loop_2048 0.065 (0.333); pt_loop_4096 0.110 (0.312); pt_loopmm_1024_8 0.071 (0.092); pt_rt_mlp2_1024 0.399 (0.041); pt_softmax_rows_2048 0.103 (0.034); pt_softmax_rows_4096 0.161 (0.004) | 7 / 0 / 0 |
| by-ref | rtx-5090 | shipped ON | pt_decode_1024_256 0.172 (0.037); pt_loop_2048 0.057 (0.034); pt_loop_4096 0.102 (0.038); pt_loopmm_1024_8 0.055 (0.147); pt_rt_mlp2_1024 0.370 (0.791); pt_softmax_rows_2048 0.073 (0.039); pt_softmax_rows_4096 0.112 (0.081) | 6 / 0 / 1 |
| by-ref/cpu | h100-pcie | shipped ON | pt_decode_1024_256.cpu 0.950 (-); pt_loop_2048.cpu 0.927 (-); pt_loop_4096.cpu 0.969 (-); pt_loopmm_1024_8.cpu 0.847 (-); pt_rt_mlp2_1024.cpu 0.962 (-) | 0 / 0 / 5 |
| by-ref/cpu | rtx-3090 | shipped ON | pt_decode_1024_256.cpu 0.948 (-); pt_loop_2048.cpu 0.973 (-); pt_loop_4096.cpu 0.980 (-); pt_loopmm_1024_8.cpu 0.902 (-); pt_rt_mlp2_1024.cpu 0.979 (-) | 0 / 0 / 5 |
| by-ref/cpu | rtx-4090 | shipped ON | pt_decode_1024_256.cpu 0.900 (-); pt_loop_2048.cpu 0.904 (-); pt_loop_4096.cpu 0.966 (-); pt_loopmm_1024_8.cpu 0.832 (-); pt_rt_mlp2_1024.cpu 0.962 (-) | 0 / 0 / 5 |
| by-ref/cpu | rtx-5090 | shipped ON | pt_decode_1024_256.cpu 0.918 (-); pt_loop_2048.cpu 0.947 (-); pt_loop_4096.cpu 0.970 (-); pt_loopmm_1024_8.cpu 0.953 (-); pt_rt_mlp2_1024.cpu 0.986 (-) | 0 / 0 / 5 |
| retain-resident | h100-pcie | shipped OFF | pt_decode_1024_256 0.992 (0.094) | 0 / 0 / 1 |
| retain-resident | rtx-3090 | shipped ON | pt_decode_1024_256 0.364 (0.088) | 1 / 0 / 0 |
| retain-resident | rtx-4090 | shipped OFF | pt_decode_1024_256 0.824 (0.050) | 1 / 0 / 0 |
| retain-resident | rtx-5090 | shipped OFF | pt_decode_1024_256 0.857 (0.037) | 1 / 0 / 0 |
| splitk | h100-pcie | shipped ON | mm_16x1024x4096 0.462 (0.009); mm_16x2048x7168 0.624 (0.013); pt_decode_1024_256 0.475 (0.094) | 3 / 0 / 0 |
| splitk | rtx-3090 | shipped ON | mm_128x2048x7168 0.441 (0.003); mm_16x1024x4096 1.000 (0.000); mm_16x2048x7168 0.905 (0.000); pt_decode_1024_256 0.557 (0.088) | 3 / 0 / 1 |
| splitk | rtx-4090 | shipped ON | pt_decode_1024_256 0.568 (0.050) | 1 / 0 / 0 |
| splitk | rtx-5090 | shipped ON | pt_decode_1024_256 0.617 (0.037) | 1 / 0 / 0 |
| splitk-atomic | h100-pcie | shipped ON | pt_decode_1024_256 0.957 (0.094) | 0 / 0 / 1 |
| splitk-atomic | rtx-3090 | shipped ON | mm_128x2048x7168 0.907 (0.003); mm_16x1024x4096 1.000 (0.000); pt_decode_1024_256 0.640 (0.088); pt_loopmm_1024_8 1.144 (0.059); pt_rt_mlp2_1024 1.011 (0.024) | 2 / 1 / 2 |
| splitk-atomic | rtx-4090 | shipped ON | pt_decode_1024_256 0.835 (0.050) | 1 / 0 / 0 |
| splitk-atomic | rtx-5090 | shipped ON | pt_decode_1024_256 0.960 (0.037) | 0 / 0 / 1 |
| streamk | h100-pcie | shipped ON | mm_128x2048x7168 0.653 (0.004); mm_16x1024x4096 0.927 (0.009); pt_decode_1024_256 0.982 (0.094); pt_loopmm_1024_8 0.674 (0.333); pt_rt_mlp2_1024 1.648 (0.812) | 2 / 0 / 3 |
| streamk | rtx-5090 | shipped ON | mm_16x18432x7168 0.897 (0.005); mm_4096x4096x4096 0.991 (0.005); pt_diamond_4096 0.989 (0.277); pt_fanout_4096 0.995 (0.006); pt_indep_4096 0.993 (0.025); pt_rowsum_2048 0.941 (0.051); pt_rowsum_4096 0.995 (0.003); pt_softmax_rows_2048 0.953 (0.039); pt_softmax_rows_4096 0.995 (0.081) | 3 / 0 / 6 |

### Kernel and movement findings

| finding | evidence | status |
|---|---|---|
| softmax_rows: by-ref lets both row folds run on the device: launches 3 → 5, copy-out 16 / 64 MB → 0 | `rowfold:on` 4 / 0 / 0 on every part; 4090 n=2048 traced kernel 0.47 → 0.52 ms, host residue 3.48 → 0.00 ms | intended (closes S64's "normalize on the host") |
| loop programs: the carried init is made on the device (+1 gen kernel); h2d 1 → 0 per run | 4090 PTX entries + `mapal_gen_0c2c9285a1cff51b` (loop_2048), `mapal_gen_7a2ade86c804ef09` (loopmm) | intended |
| programs without a loop (conv, diamond, ew, fanout, fir, indep, rowsum, rm) | host.ll, cpu.host.ll, PTX byte-identical to S64's bundles on all four parts (H100 rm / decode PTX: S64's warp-fill change) | their S64 → S65 time differences are the boxes (4090 diamond_4096 7.44 → 9.35 ms with identical IR) |
| rt_mlp2: 8 MB h2d per trip remains | the facts require it (host-made request inputs, a device reader): `h2d=8388608` intended = realized on every part | required |
| decode: 2 KB h2d + 1 KB d2h per trip remain | intended `h2d=2048 d2h=1024` bytes = realized; traced 2048 uploads / 1024 copy-outs over 1024 trips | required |
| 3090 loopmm: splitk-atomic ON slower | 1.6191 vs 1.4133 ms OFF, 1.144 (1.124..1.186), null 0.059; ON: atomic S=2, 12.49 warps / SM, 64 `atom.add`, a zero-fill call; OFF: S=1, 6.24 warps, 8 B spill | open (same row: decode 0.640, 128×2048×7168 0.907 ON faster) |
| H100 rt_mlp2 time not stable | shipped 20.54 vs null 12.93 ms (same build), null 0.812; traced copies 9.89 vs 3.87 ms | open (box) |
| every mismatch on a shipped program is an S64 class | table below; `by-ref:on` / `by-ref:runtime` 0 mismatch on all four parts | S64 open items |

Every mismatch on a shipped program (S65 build):

| arch | rung | check | programs | realized (first) |
|---|---|---|---|---|
| rtx-5090 | exec:on | JIT image local bytes == ptxas stack frame | mm_16x1024x4096 mm_16x1024x4096_c mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_c mm_16x2048x7168_rc mm_4096x4096x4096_r mm_4096x4096x4096_x | exec.local=152 ptxas.stack=208 |
| rtx-5090 | exec:on | JIT image registers == ptxas -v | mm_16x18432x7168 mm_16x18432x7168_r mm_16x18432x7168_x mm_4096x4096x4096 pt_rm_m128_d1024_h2048_c pt_rm_m128_d1024_h2048_x pt_rm_m128_d256_h512_x | exec.regs=225 ptxas.regs=227 |
| rtx-3090 | face-registers:on | no spill | mm_128x2048x7168_rc mm_16x1024x4096_r mm_16x1024x4096_x pt_loopmm_1024_8 pt_rt_mlp2_1024 | ptxas.spill_st=24 ptxas.spill_ld=24 |
| rtx-5090 | face-registers:on | no spill | mm_16x1024x4096 mm_16x1024x4096_c mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_c mm_16x2048x7168_rc mm_4096x4096x4096_r mm_4096x4096x4096_x | ptxas.spill_st=208 ptxas.spill_ld=208 |
| h100-pcie | grid-share:on | residency sized to the grid share | mm_16x1024x4096 mm_16x1024x4096_c mm_16x1024x4096_rc | plan.blocks_per_sm=6 plan.ctas=0 plan.combine=none plan.gx=64 plan.gy=1 occ.sms=114 |
| rtx-3090 | min-blocks:on | ptxas met it without spilling | mm_128x2048x7168_rc mm_16x1024x4096_r mm_16x1024x4096_x pt_loopmm_1024_8 pt_rt_mlp2_1024 | ptxas.spill_st=24 |
| rtx-5090 | min-blocks:on | ptxas met it without spilling | mm_16x1024x4096 mm_16x1024x4096_c mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_c mm_16x2048x7168_rc mm_4096x4096x4096_r mm_4096x4096x4096_x | ptxas.spill_st=208 |
| h100-pcie | module:on | every kernel in the device module is launched by the host | mm_128x2048x7168 mm_128x2048x7168_r mm_16x1024x4096 mm_16x1024x4096_r mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_rc pt_decode_1024_256 pt_loopmm_1024_8 pt_rt_mlp2_1024 | mod.launched=0 |
| rtx-3090 | module:on | every kernel in the device module is launched by the host | mm_128x2048x7168 mm_128x2048x7168_rc mm_16x1024x4096 mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_rc pt_decode_1024_256 pt_loopmm_1024_8 pt_rt_mlp2_1024 | mod.launched=0 |
| rtx-4090 | module:on | every kernel in the device module is launched by the host | mm_128x2048x7168_r mm_16x1024x4096_r pt_decode_1024_256 | mod.launched=0 |
| rtx-5090 | module:on | every kernel in the device module is launched by the host | mm_16x18432x7168 mm_4096x4096x4096 pt_decode_1024_256 pt_diamond_4096 pt_fanout_4096 pt_indep_4096 pt_rowsum_2048 pt_rowsum_4096 pt_softmax_rows_2048 pt_softmax_rows_4096 | mod.launched=0 |
| rtx-3090 | panel-regs:on | the prefetch panel fits the register file: no spill | mm_16x1024x4096_r mm_16x1024x4096_x | ptxas.spill_st=140 |
| rtx-5090 | panel-regs:on | the prefetch panel fits the register file: no spill | mm_16x1024x4096 mm_16x1024x4096_c mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_c mm_16x2048x7168_rc | ptxas.spill_st=208 |
| h100-pcie | prefetch:on | the next panel's LDGs are covered by this panel's math | mm_16x18432x7168 mm_16x18432x7168_c mm_16x18432x7168_r mm_16x18432x7168_rc mm_16x18432x7168_x mm_4096x4096x4096_r mm_4096x4096x4096_x pt_rm_m128_d1024_h2048 pt_rm_m128_d1024_h2048_c pt_rm_m128_d256_h512 pt_rm_m128_d256_h512_c | sass.hot.ldg_cover_min=0 |
| rtx-3090 | prefetch:on | the next panel's LDGs are covered by this panel's math | mm_16x1024x4096_r mm_16x1024x4096_x mm_4096x4096x4096_r mm_4096x4096x4096_x pt_rm_m128_d256_h512_x | sass.hot.ldg_cover_min=0 |

Intended vs realized, shipped programs (ok / MISMATCH / n-a):

| rung:state | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 |
|---|---|---|---|---|
| arch-rungs:on | 62 / 0 / 0 | 69 / 0 / 0 | 70 / 0 / 0 | 70 / 0 / 0 |
| arena:on | 41 / 0 / 0 | 41 / 0 / 0 | 41 / 0 / 0 | 41 / 0 / 0 |
| async:off | 27 / 0 / 0 | 60 / 0 / 0 | 65 / 0 / 0 | 64 / 0 / 0 |
| async:on | 60 / 0 / 0 | 8 / 0 / 0 | - | - |
| by-ref:on | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| by-ref:runtime | 105 / 0 / 0 | 105 / 0 / 0 | 105 / 0 / 0 | 105 / 0 / 0 |
| cell-fill:on | - | - | 8 / 0 / 0 | 5 / 0 / 0 |
| cpu-tiling:on | 2 / 0 / 0 | 2 / 0 / 0 | 2 / 0 / 0 | 2 / 0 / 0 |
| devgen:on | 44 / 0 / 0 | 44 / 0 / 0 | 44 / 0 / 0 | 44 / 0 / 0 |
| devmap:on | 4 / 0 / 0 | 4 / 0 / 0 | 4 / 0 / 0 | 4 / 0 / 0 |
| eager-init:on | 44 / 0 / 0 | 44 / 0 / 0 | 44 / 0 / 0 | 44 / 0 / 0 |
| epilogue:on | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 |
| exec:on | 66 / 0 / 0 | 50 / 16 / 0 | 66 / 0 / 0 | 66 / 0 / 0 |
| face-fma:on | 88 / 0 / 0 | 106 / 0 / 0 | 104 / 0 / 0 | 108 / 0 / 0 |
| face-registers:on | 171 / 0 / 0 | 184 / 8 / 0 | 195 / 0 / 0 | 187 / 5 / 0 |
| face-separate:on | 36 / 0 / 0 | 32 / 0 / 0 | 36 / 0 / 0 | 32 / 0 / 0 |
| grid-share:on | 47 / 0 / 0 | 64 / 0 / 0 | 28 / 4 / 0 | 8 / 0 / 0 |
| hostopt:on | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| idx32:on | 40 / 0 / 0 | 28 / 0 / 0 | 52 / 0 / 0 | 24 / 0 / 0 |
| interleave:on | 57 / 0 / 0 | 64 / 0 / 0 | 57 / 0 / 0 | 29 / 0 / 0 |
| min-blocks:off | - | - | 55 / 0 / 0 | - |
| min-blocks:on | 114 / 0 / 0 | 120 / 8 / 0 | 20 / 0 / 0 | 123 / 5 / 0 |
| module:on | 72 / 4 / 0 | 70 / 11 / 0 | 76 / 12 / 0 | 75 / 11 / 0 |
| panel-regs:on | 10 / 0 / 0 | 54 / 6 / 0 | 55 / 0 / 0 | 6 / 2 / 0 |
| pdl:off | - | 69 / 0 / 0 | 70 / 0 / 0 | - |
| prefetch:on | 27 / 0 / 0 | 60 / 0 / 0 | 54 / 11 / 0 | 58 / 6 / 0 |
| resident:on | 44 / 0 / 0 | 44 / 0 / 0 | 44 / 0 / 0 | 44 / 0 / 0 |
| row-fit:on | 30 / 0 / 0 | 60 / 0 / 0 | 10 / 0 / 0 | 35 / 0 / 0 |
| rowfold:on | 4 / 0 / 0 | 4 / 0 / 0 | 4 / 0 / 0 | 4 / 0 / 0 |
| splitk-atomic-zero:on | - | 9 / 0 / 0 | 5 / 0 / 0 | 5 / 0 / 0 |
| splitk-atomic:on | 4 / 0 / 0 | 4 / 0 / 0 | 2 / 0 / 0 | 14 / 0 / 0 |
| splitk:on | 4 / 0 / 0 | - | 12 / 0 / 0 | 8 / 0 / 0 |
| streamk:on | - | 27 / 0 / 0 | 15 / 0 / 0 | - |
| unroll:on | 114 / 0 / 0 | 128 / 0 / 0 | 130 / 0 / 0 | 128 / 0 / 0 |
| warp-fill:on | 57 / 0 / 0 | 37 / 0 / 0 | 53 / 0 / 0 | 37 / 0 / 0 |
| wide-fma:on | 51 / 0 / 0 | - | - | 45 / 0 / 0 |

### Workload suite: iter ms (median of 30 reps), Mapal vs the best library / PyTorch, S64 beside it

`best other` = fastest of the library leg (cuBLAS `lt`, CUTLASS, fused / cache / streams harnesses) and torch eager /
compile on the same box. Decode on the 4090 / 5090 as shipped after this session (retain-resident ON, the measured
arm): 33.05 ms → 2.491, 36.69 ms → 1.944. S64 ran on other boxes of the same parts.

| workload | arch | CPU-only | GPU-only | CPU+GPU | null | best other (ms) | Mapal / best other | S64 CPU+GPU | S64 ratio |
|---|---|---|---|---|---|---|---|---|---|
| conv_2048_5 | rtx-4090 | 1.3083 | = shipped | 1.3338 | 0.040 | eager 0.0878 | 15.191 | 1.2274 | 13.775 |
| decode_1024_256 | rtx-4090 | 1916.34 | = shipped | 40.58 | 0.002 | cache 13.2660 | 3.059 | 136.76 | 5.779 |
| diamond_4096 | rtx-4090 | 287.98 | = shipped | 9.3463 | 0.001 | compile 8.3425 | 1.120 | 7.4382 | 0.857 |
| ew_2048 | rtx-4090 | 1.6136 | = shipped | 0.1984 | 0.027 | fused 0.0573 | 3.462 | 0.2966 | 5.078 |
| ew_4096 | rtx-4090 | 8.1164 | = shipped | 0.2718 | 0.135 | fused 0.2253 | 1.206 | 0.3720 | 1.650 |
| fanout_4096 | rtx-4090 | 211.86 | = shipped | 6.1545 | 0.001 | streams 5.5030 | 1.118 | 4.9482 | 0.852 |
| fir_4194304_16 | rtx-4090 | 1.3269 | = shipped | 0.0374 | 0.001 | compile 0.0620 | 0.603 | 0.0445 | 0.711 |
| indep_4096 | rtx-4090 | 191.34 | = shipped | 6.3178 | 0.001 | compile 5.5368 | 1.141 | 5.0279 | 0.873 |
| loop_2048 | rtx-4090 | 941.84 | = shipped | 2.5753 | 0.001 | cutlass 2.5836 | 0.997 | 38.46 | 15.703 |
| loop_4096 | rtx-4090 | 8417.07 | = shipped | 19.65 | 0.002 | compile 21.5572 | 0.911 | - | - |
| loopmm_1024_8 | rtx-4090 | 119.02 | = shipped | 0.5312 | 0.019 | eager 0.4127 | 1.287 | 13.12 | 33.534 |
| rm_m128_d1024_h2048 | rtx-4090 | 2.5315 | = shipped | 0.1170 | 0.089 | compile 0.0591 | 1.980 | 0.1332 | 2.204 |
| rm_m128_d256_h512 | rtx-4090 | 0.2084 | = shipped | 0.1011 | 0.288 | compile 0.0223 | 4.534 | 0.1557 | 6.921 |
| rowsum_2048 | rtx-4090 | 17.56 | = shipped | 0.4232 | 0.007 | eager 0.3308 | 1.279 | - | - |
| rowsum_4096 | rtx-4090 | 118.11 | = shipped | 2.4991 | 0.002 | compile 2.7197 | 0.919 | 2.5594 | 0.912 |
| rt_mlp2_1024 | rtx-4090 | 510.93 | = shipped | 8.7197 | 0.001 | compile 1.7191 | 5.072 | 28.04 | 16.320 |
| softmax_rows_2048 | rtx-4090 | 20.38 | = shipped | 0.5157 | 0.001 | compile 0.3318 | 1.554 | 6.2131 | 19.018 |
| softmax_rows_4096 | rtx-4090 | 124.93 | = shipped | 3.5187 | 0.001 | compile 2.4186 | 1.455 | 25.86 | 8.935 |
| conv_2048_5 | rtx-5090 | 1.7852 | = shipped | 1.8121 | 0.018 | compile 0.0799 | 22.679 | 0.4927 | 6.167 |
| decode_1024_256 | rtx-5090 | 2171.98 | = shipped | 42.71 | 0.005 | cache 18.8700 | 2.264 | 171.87 | 8.702 |
| diamond_4096 | rtx-5090 | 427.54 | = shipped | 6.1690 | 0.001 | streams 6.1328 | 1.006 | 7.5879 | 1.081 |
| ew_2048 | rtx-5090 | 1.7417 | = shipped | 0.1065 | 0.122 | fused 0.0348 | 3.061 | 0.2783 | 8.485 |
| ew_4096 | rtx-5090 | 8.5493 | = shipped | 0.1426 | 0.054 | fused 0.1311 | 1.088 | 0.3272 | 2.495 |
| fanout_4096 | rtx-5090 | 284.72 | = shipped | 4.1503 | 0.001 | compile 4.0428 | 1.027 | 4.9674 | 1.059 |
| fir_4194304_16 | rtx-5090 | 1.5815 | = shipped | 0.0267 | 0.009 | compile 0.0512 | 0.522 | 0.0309 | 0.628 |
| indep_4096 | rtx-5090 | 284.92 | = shipped | 4.1371 | 0.000 | streams 4.0345 | 1.025 | 5.0399 | 1.067 |
| loop_2048 | rtx-5090 | 1050.28 | = shipped | 2.1838 | 0.003 | compile 2.0388 | 1.071 | 22.64 | 10.481 |
| loop_4096 | rtx-5090 | 8553.22 | = shipped | 15.83 | 0.000 | compile 16.2929 | 0.971 | - | - |
| loopmm_1024_8 | rtx-5090 | 129.01 | = shipped | 0.4818 | 0.006 | compile 0.3523 | 1.368 | 4.6312 | 12.706 |
| rm_m128_d1024_h2048 | rtx-5090 | 3.7645 | = shipped | 0.0814 | 0.148 | compile 0.0430 | 1.894 | 0.0840 | 2.049 |
| rm_m128_d256_h512 | rtx-5090 | 0.2571 | = shipped | 0.0389 | 0.457 | compile 0.0225 | 1.728 | 0.0489 | 2.658 |
| rowsum_2048 | rtx-5090 | 22.55 | = shipped | 0.3156 | 0.002 | compile 0.2621 | 1.204 | - | - |
| rowsum_4096 | rtx-5090 | 144.52 | = shipped | 2.1269 | 0.000 | compile 1.9640 | 1.083 | 2.5616 | 1.112 |
| rt_mlp2_1024 | rtx-5090 | 555.09 | = shipped | 7.5898 | 0.001 | compile 1.5514 | 4.892 | 12.68 | 7.859 |
| softmax_rows_2048 | rtx-5090 | 28.90 | = shipped | 0.3732 | 0.031 | compile 0.2642 | 1.412 | 3.2561 | 11.953 |
| softmax_rows_4096 | rtx-5090 | 176.01 | = shipped | 2.3396 | 0.001 | compile 2.0142 | 1.162 | 14.75 | 6.314 |
| conv_2048_5 | h100-pcie | 0.9546 | = shipped | 1.0467 | 0.015 | compile 0.7784 | 1.345 | 1.1127 | 1.429 |
| decode_1024_256 | h100-pcie | 2932.36 | = shipped | 87.42 | 0.021 | cache 27.0520 | 3.232 | 300.43 | 11.644 |
| diamond_4096 | h100-pcie | 423.18 | = shipped | 13.50 | 0.000 | streams 12.9601 | 1.042 | 13.57 | 1.038 |
| ew_2048 | h100-pcie | 1.6441 | = shipped | 0.6552 | 0.030 | compile 0.0329 | 19.915 | 0.6566 | 19.777 |
| ew_4096 | h100-pcie | 5.1683 | = shipped | 0.7766 | 0.018 | compile 0.1140 | 6.812 | 0.7715 | 6.791 |
| fanout_4096 | h100-pcie | 279.06 | = shipped | 8.9745 | 0.001 | streams 9.1565 | 0.980 | 9.0416 | 0.977 |
| fir_4194304_16 | h100-pcie | 0.9421 | = shipped | 0.0639 | 0.044 | eager 0.0928 | 0.689 | 0.0591 | 0.637 |
| indep_4096 | h100-pcie | 275.61 | = shipped | 9.0438 | 0.003 | streams 8.3833 | 1.079 | 9.0852 | 1.075 |
| loop_2048 | h100-pcie | 1520.36 | = shipped | 6.4506 | 0.002 | lt 5.0808 | 1.270 | 35.00 | 6.890 |
| loop_4096 | h100-pcie | 14689.95 | = shipped | 36.28 | 0.000 | lt 36.9943 | 0.981 | - | - |
| loopmm_1024_8 | h100-pcie | 179.63 | = shipped | 1.0944 | 0.013 | compile 0.7586 | 1.443 | 8.2035 | 10.740 |
| rm_m128_d1024_h2048 | h100-pcie | 8.2118 | = shipped | 0.2017 | 0.047 | eager 0.0837 | 2.410 | 0.4910 | 5.832 |
| rm_m128_d256_h512 | h100-pcie | 1.0181 | = shipped | 0.1423 | 0.016 | eager 0.0339 | 4.199 | 0.2805 | 7.835 |
| rowsum_2048 | h100-pcie | 25.73 | = shipped | 0.9295 | 0.006 | compile 0.6414 | 1.449 | - | - |
| rowsum_4096 | h100-pcie | 164.26 | = shipped | 4.7771 | 0.001 | compile 4.6128 | 1.036 | 4.7801 | 1.036 |
| rt_mlp2_1024 | h100-pcie | 767.14 | = shipped | 20.54 | 0.370 | compile 3.1949 | 6.429 | 51.75 | 14.424 |
| softmax_rows_2048 | h100-pcie | 33.21 | = shipped | 1.0563 | 0.023 | compile 0.6488 | 1.628 | 16.09 | 24.789 |
| softmax_rows_4096 | h100-pcie | 171.33 | = shipped | 5.1955 | 0.000 | compile 4.7213 | 1.100 | 55.09 | 11.668 |
| conv_2048_5 | rtx-3090 | 1.4155 | = shipped | 1.4342 | 0.038 | compile 0.1608 | 8.919 | 1.8971 | 11.725 |
| decode_1024_256 | rtx-3090 | 3890.90 | = shipped | 54.05 | 0.004 | cache 32.9520 | 1.640 | 575.73 | 15.670 |
| diamond_4096 | rtx-3090 | 671.17 | = shipped | 17.04 | 0.001 | streams 17.1807 | 0.992 | 16.01 | 1.028 |
| ew_2048 | rtx-3090 | 0.8906 | = shipped | 0.4329 | 0.054 | compile 0.0625 | 6.926 | 0.2746 | 4.394 |
| ew_4096 | rtx-3090 | 4.1150 | = shipped | 0.5074 | 0.011 | compile 0.2406 | 2.109 | 0.4073 | 1.693 |
| fanout_4096 | rtx-3090 | 434.49 | = shipped | 11.36 | 0.000 | streams 11.5272 | 0.985 | 10.71 | 1.009 |
| fir_4194304_16 | rtx-3090 | 0.9845 | = shipped | 0.0838 | 0.002 | compile 0.1126 | 0.744 | 0.0832 | 0.789 |
| indep_4096 | rtx-3090 | 389.69 | = shipped | 11.21 | 0.018 | streams 11.2589 | 0.996 | 10.72 | 1.050 |
| loop_2048 | rtx-3090 | 1997.45 | = shipped | 7.0657 | 0.002 | lt 6.1430 | 1.150 | 79.11 | 13.990 |
| loop_4096 | rtx-3090 | 17627.41 | = shipped | 46.06 | 0.005 | lt 45.6028 | 1.010 | - | - |
| loopmm_1024_8 | rtx-3090 | 238.12 | = shipped | 1.6191 | 0.001 | compile 0.8960 | 1.807 | 20.15 | 23.339 |
| rm_m128_d1024_h2048 | rtx-3090 | 6.9193 | = shipped | 0.3484 | 0.000 | eager 0.0953 | 3.656 | 0.3495 | 4.064 |
| rm_m128_d256_h512 | rtx-3090 | 0.5864 | = shipped | 0.1079 | 0.004 | eager 0.0276 | 3.908 | 0.1229 | 4.622 |
| rowsum_2048 | rtx-3090 | 35.62 | = shipped | 0.9423 | 0.001 | compile 0.7762 | 1.214 | - | - |
| rowsum_4096 | rtx-3090 | 234.85 | = shipped | 5.8710 | 0.000 | compile 5.7917 | 1.014 | 5.5507 | 1.047 |
| rt_mlp2_1024 | rtx-3090 | 1028.22 | = shipped | 14.49 | 0.006 | compile 4.1421 | 3.497 | 47.07 | 12.469 |
| softmax_rows_2048 | rtx-3090 | 40.55 | = shipped | 1.1036 | 0.004 | compile 0.7567 | 1.458 | 11.36 | 14.996 |
| softmax_rows_4096 | rtx-3090 | 248.09 | = shipped | 6.3963 | 0.001 | compile 5.8808 | 1.088 | 42.52 | 7.878 |

### Tests

| test | result |
|---|---|
| `MAPAL_REQUIRE_CLANG=1 cargo test --workspace --release --no-fail-fast -- --test-threads=1` | 1231 passed, 0 failed, 1 ignored (91 binaries) |
| `differential_carried_loops_values_and_movement` (shared generator `crates/mapal-rewrite/tests/testgen`) | 320 programs (270 run to the end), 434 loops; per-trip copies 292 = required 292; values = interpreter, ON and OFF, -O0 and -O2 |
| `differential_device_loops_values_and_movement` (emulated h100-pcie) | 64 programs, 59 with a device unit, 37 device-carried states; values = interpreter ON and OFF; movement = the plan's |
| `algos.rs` `view_death_follows_projections`, `view_death_init_lives_through_the_loop_exit` | ok |
| `resident.rs` ping-pong (`two`, 0 copies), one buffer (`one`), row writes in place (`inplace`), device state without host bytes | ok |
| found by the generator, fixed | an in-place write in the parallel entry racing another task's read (S64 arm too); two device carries on one merge sharing a slot; task-local spare buffers |

## S64

Raw: [`benches/results/evidence/s64/`](../../../benches/results/evidence/s64/) · tables:
`python3 benches/gpu/evidence_report.py benches/results/evidence/s64` (and `…/s64/endcheck`) · checker:
[`benches/gpu/evidence.py`](../../../benches/gpu/evidence.py), intent: [`benches/gpu/intent.py`](../../../benches/gpu/intent.py)

### Decisions

| GPU_RUNGS_OFF row | part | rung | class | change | evidence (static) | measured (ON / OFF, two processes, null) |
|---|---|---|---|---|---|---|
| 14 | h100-pcie | warp-fill | Under, Contract | **Under → Partial** (≥ 1 warp per scheduler): ON at 0 | MLP contract kernels OFF: 8×8 tile, 32-thread blocks, 254 regs, 0.56 / 1.12 resident warps per SM; ON: 4×4, 2.25 / 4.49 | MLP d1024 0.1501/0.1500 vs 0.3726/0.3724 ms, d256 0.0332/0.0332 vs 0.1027/0.1028 (null 0.000/0.001); end check 128×2048×7168 0.3090 vs 0.5381 ms (default), 0.4742 vs 0.8966 (`_c`), 0.4728 vs 0.8957 (`_rc`), null 0.002; decode 0.914 inside its null 0.163 |
| 37 | rtx-3090 | warp-fill | Under, Contract | none (open) | as on the H100; the filled tile lowers the split count (atomic S 8 → 2, workspace S 4 → 2) | MLP d1024 0.1630/0.1629 vs 0.3092/0.3092 ms, d256 0.0338/0.0338 vs 0.0707/0.0715; 128×2048×7168 `_c` 0.523; default 1.179 and `_rc` 1.058 slower (null 0.003) |
| 93 | rtx-3090 | splitk | Filled, Exact | none (sign unstable) | resident warps per SM 3.12 → 12.49 | 16×2048×7168 `_r` 0.904/0.890 (null 0.011); S62 1.134/1.126 on another 3090 |
| all others | | | | none | judge.tsv `status` | judge.tsv `remeasured` |

### Intended vs realized (shipped programs; ok / MISMATCH / n-a)

| rung:state | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 |
|---|---|---|---|---|
| arch-rungs:on | 65 / 0 / 0 | 74 / 0 / 0 | 79 / 0 / 0 | 75 / 0 / 0 |
| arena:on | 43 / 0 / 0 | 45 / 0 / 0 | 47 / 0 / 0 | 45 / 0 / 0 |
| async:off | 30 / 0 / 0 | 62 / 0 / 0 | 74 / 0 / 0 | 69 / 0 / 0 |
| async:on | 60 / 0 / 0 | 14 / 0 / 0 | - | - |
| cell-fill:on | - | - | 8 / 0 / 0 | 5 / 0 / 0 |
| cpu-tiling:on | 2 / 0 / 0 | 2 / 0 / 0 | 2 / 0 / 0 | 2 / 0 / 0 |
| devgen:on | 47 / 0 / 0 | 49 / 0 / 0 | 51 / 0 / 0 | 49 / 0 / 0 |
| devmap:on | 4 / 0 / 0 | 4 / 0 / 0 | 4 / 0 / 0 | 4 / 0 / 0 |
| eager-init:on | 47 / 0 / 0 | 49 / 0 / 0 | 51 / 0 / 0 | 49 / 0 / 0 |
| epilogue:on | 13 / 0 / 0 | 13 / 0 / 0 | 15 / 0 / 0 | 13 / 0 / 0 |
| exec:on | 66 / 0 / 0 | 49 / 17 / 0 | 66 / 0 / 0 | 65 / 1 / 0 |
| face-fma:on | 88 / 0 / 0 | 110 / 0 / 0 | 112 / 0 / 0 | 112 / 0 / 0 |
| face-registers:on | 180 / 0 / 0 | 196 / 11 / 0 | 222 / 0 / 0 | 202 / 5 / 0 |
| face-separate:on | 42 / 0 / 0 | 38 / 0 / 0 | 46 / 0 / 0 | 38 / 0 / 0 |
| grid-share:on | 50 / 0 / 0 | 69 / 0 / 0 | 33 / 4 / 0 | 8 / 0 / 0 |
| hostopt:on | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 |
| idx32:on | 43 / 0 / 0 | 30 / 0 / 0 | 56 / 0 / 0 | 26 / 0 / 0 |
| interleave:on | 60 / 0 / 0 | 69 / 0 / 0 | 66 / 0 / 0 | 34 / 0 / 0 |
| min-blocks:off | - | - | 64 / 0 / 0 | - |
| min-blocks:on | 120 / 0 / 0 | 127 / 11 / 0 | 20 / 0 / 0 | 133 / 5 / 0 |
| module:on | 73 / 4 / 0 | 75 / 11 / 0 | 85 / 12 / 0 | 80 / 11 / 0 |
| panel-regs:on | 10 / 0 / 0 | 56 / 6 / 0 | 64 / 0 / 0 | 6 / 2 / 0 |
| pdl:off | - | 74 / 0 / 0 | 79 / 0 / 0 | - |
| prefetch:on | 30 / 0 / 0 | 62 / 0 / 0 | 62 / 12 / 0 | 60 / 9 / 0 |
| resident:on | 47 / 0 / 0 | 49 / 0 / 0 | 51 / 0 / 0 | 49 / 0 / 0 |
| row-fit:on | 30 / 0 / 0 | 62 / 0 / 0 | 10 / 0 / 0 | 35 / 0 / 0 |
| splitk-atomic-zero:on | - | 9 / 0 / 0 | 5 / 0 / 0 | 5 / 0 / 0 |
| splitk-atomic:on | 4 / 0 / 0 | 4 / 0 / 0 | 2 / 0 / 0 | 14 / 0 / 0 |
| splitk:on | 4 / 0 / 0 | - | 12 / 0 / 0 | 8 / 0 / 0 |
| streamk:on | - | 27 / 0 / 0 | 15 / 0 / 0 | - |
| unroll:on | 120 / 0 / 0 | 138 / 0 / 0 | 148 / 0 / 0 | 138 / 0 / 0 |
| warp-fill:on | 60 / 0 / 0 | 40 / 0 / 0 | 48 / 0 / 0 | 42 / 0 / 0 |
| wide-fma:on | 60 / 0 / 0 | - | - | 51 / 0 / 0 |

### Findings: every mismatch on a shipped program

| arch | rung | check | programs | realized (first) |
|---|---|---|---|---|
| rtx-3090 | exec:on | JIT image registers == ptxas -v | pt_rm_m128_d1024_h2048_x | exec.regs=194 ptxas.regs=196 |
| rtx-5090 | exec:on | JIT image local bytes == ptxas stack frame | mm_16x1024x4096 mm_16x1024x4096_c mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_c mm_16x2048x7168_rc mm_4096x4096x4096_r mm_4096x4096x4096_x | exec.local=152 ptxas.stack=208 |
| rtx-5090 | exec:on | JIT image registers == ptxas -v | mm_16x18432x7168 mm_16x18432x7168_r mm_16x18432x7168_x mm_4096x4096x4096 pt_rm_m128_d1024_h2048_c pt_rm_m128_d1024_h2048_x pt_rm_m128_d256_h512_x | exec.regs=225 ptxas.regs=227 |
| rtx-3090 | face-registers:on | no spill | mm_128x2048x7168_rc mm_16x1024x4096_r mm_16x1024x4096_x pt_loopmm_1024_8 pt_rt_mlp2_1024 | ptxas.spill_st=24 ptxas.spill_ld=24 |
| rtx-5090 | face-registers:on | no spill | mm_16x1024x4096 mm_16x1024x4096_c mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_c mm_16x2048x7168_rc mm_4096x4096x4096_r mm_4096x4096x4096_x pt_loop_2048_x pt_rowsum_2048_x pt_rowsum_4096_x | ptxas.spill_st=208 ptxas.spill_ld=208 |
| h100-pcie | grid-share:on | residency sized to the grid share | mm_16x1024x4096 mm_16x1024x4096_c mm_16x1024x4096_rc | plan.blocks_per_sm=6 plan.ctas=0 plan.combine=none plan.gx=64 plan.gy=1 occ.sms=114 |
| rtx-3090 | min-blocks:on | ptxas met it without spilling | mm_128x2048x7168_rc mm_16x1024x4096_r mm_16x1024x4096_x pt_loopmm_1024_8 pt_rt_mlp2_1024 | ptxas.spill_st=24 |
| rtx-5090 | min-blocks:on | ptxas met it without spilling | mm_16x1024x4096 mm_16x1024x4096_c mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_c mm_16x2048x7168_rc mm_4096x4096x4096_r mm_4096x4096x4096_x pt_loop_2048_x pt_rowsum_2048_x pt_rowsum_4096_x | ptxas.spill_st=208 |
| h100-pcie | module:on | every kernel in the device module is launched by the host | mm_128x2048x7168 mm_128x2048x7168_r mm_16x1024x4096 mm_16x1024x4096_r mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_rc pt_decode_1024_256 pt_loopmm_1024_8 pt_rt_mlp2_1024 | mod.launched=0 |
| rtx-3090 | module:on | every kernel in the device module is launched by the host | mm_128x2048x7168 mm_128x2048x7168_rc mm_16x1024x4096 mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_rc pt_decode_1024_256 pt_loopmm_1024_8 pt_rt_mlp2_1024 | mod.launched=0 |
| rtx-4090 | module:on | every kernel in the device module is launched by the host | mm_128x2048x7168_r mm_16x1024x4096_r pt_decode_1024_256 | mod.launched=0 |
| rtx-5090 | module:on | every kernel in the device module is launched by the host | mm_16x18432x7168 mm_4096x4096x4096 pt_decode_1024_256 pt_diamond_4096 pt_fanout_4096 pt_indep_4096 pt_rowsum_2048 pt_rowsum_4096 pt_softmax_rows_2048 pt_softmax_rows_4096 | mod.launched=0 |
| rtx-3090 | panel-regs:on | the prefetch panel fits the register file: no spill | mm_16x1024x4096_r mm_16x1024x4096_x | ptxas.spill_st=140 |
| rtx-5090 | panel-regs:on | the prefetch panel fits the register file: no spill | mm_16x1024x4096 mm_16x1024x4096_c mm_16x1024x4096_rc mm_16x2048x7168 mm_16x2048x7168_c mm_16x2048x7168_rc | ptxas.spill_st=208 |
| h100-pcie | prefetch:on | the next panel's LDGs are covered by this panel's math | mm_16x18432x7168 mm_16x18432x7168_c mm_16x18432x7168_r mm_16x18432x7168_rc mm_16x18432x7168_x mm_4096x4096x4096_r mm_4096x4096x4096_x pt_diamond_4096_x pt_loop_2048_x pt_rowsum_2048_x pt_rowsum_4096_x | sass.hot.ldg_cover_min=0 |
| rtx-3090 | prefetch:on | the next panel's LDGs are covered by this panel's math | mm_16x1024x4096_r mm_16x1024x4096_x mm_4096x4096x4096_r mm_4096x4096x4096_x pt_loop_2048_x pt_rm_m128_d256_h512_x pt_rowsum_2048_x pt_rowsum_4096_x | sass.hot.ldg_cover_min=0 |

### How each optimization turned out in time (arms timed this session: ON / OFF, median over processes)

| rung | arch | shipped | cells: ON/OFF (null) | ON faster / slower / inside null |
|---|---|---|---|---|
| asyncstage | rtx-3090 | shipped OFF | mm_128x2048x7168_c 0.830 (0.001); mm_16x2048x7168_c 0.968 (0.011); mm_16x2048x7168_x 1.032 (0.011) | 2 / 0 / 1 |
| fill-rows | h100-pcie | shipped OFF | mm_16x1024x4096_x 1.855 (0.011) | 0 / 1 / 0 |
| grid-share | h100-pcie | shipped OFF | pt_rm_m128_d1024_h2048_x 0.991 (0.001); pt_rm_m128_d256_h512_x 0.993 (0.004) | 2 / 0 / 0 |
| grid-share | rtx-3090 | shipped OFF | mm_16x2048x7168_x 1.000 (0.011); pt_rm_m128_d1024_h2048_c 0.993 (0.003); pt_rm_m128_d256_h512_c 0.952 (0.000) | 2 / 0 / 1 |
| grid-share | rtx-4090 | shipped OFF | mm_128x2048x7168_x 0.928 (0.002); mm_16x1024x4096_x 1.016 (0.027) | 1 / 0 / 1 |
| panel-regs | h100-pcie | shipped OFF | pt_rm_m128_d1024_h2048_x 1.110 (0.001); pt_rm_m128_d256_h512_x 0.981 (0.004) | 1 / 1 / 0 |
| panel-regs | rtx-3090 | shipped OFF | mm_16x2048x7168_x 1.056 (0.011) | 0 / 1 / 0 |
| row-fit | h100-pcie | shipped OFF | mm_16x1024x4096_c 4.249 (0.011); mm_16x18432x7168_x 1.147 (0.001) | 0 / 2 / 0 |
| row-fit | rtx-3090 | shipped OFF | mm_16x18432x7168_x 1.001 (0.004) | 0 / 0 / 1 |
| row-fit | rtx-4090 | shipped OFF | mm_16x18432x7168_x 1.048 (0.000) | 0 / 1 / 0 |
| row-fit | rtx-5090 | shipped OFF | mm_16x18432x7168_x 1.011 (0.004) | 0 / 1 / 0 |
| rowfold-layout | h100-pcie | shipped OFF | pt_rowsum_4096_c 1.060 (0.007); pt_rowsum_4096_x 1.190 (0.007) | 0 / 2 / 0 |
| rowfold-layout | rtx-3090 | shipped OFF | pt_rowsum_4096_c 1.024 (0.036); pt_rowsum_4096_x 1.007 (0.036) | 0 / 0 / 2 |
| rowfold-layout | rtx-5090 | shipped OFF | pt_rowsum_4096_c 1.033 (0.005); pt_rowsum_4096_x 1.029 (0.005) | 0 / 2 / 0 |
| splitk | h100-pcie | shipped ON | mm_16x1024x4096 0.463 (0.011); mm_16x2048x7168 0.622 (0.004); pt_decode_1024_256 0.751 (0.019) | 3 / 0 / 0 |
| splitk | rtx-3090 | shipped OFF/shipped ON | mm_128x2048x7168 0.429 (0.001); mm_16x1024x4096 0.989 (0.000); mm_16x2048x7168 0.887 (0.011); mm_16x2048x7168_r 0.904 (0.011); pt_decode_1024_256 0.927 (0.101) | 4 / 0 / 1 |
| splitk | rtx-4090 | shipped ON | pt_decode_1024_256 0.818 (0.013) | 1 / 0 / 0 |
| splitk | rtx-5090 | shipped ON | pt_decode_1024_256 0.895 (0.037) | 1 / 0 / 0 |
| splitk-atomic | h100-pcie | shipped OFF/shipped ON | mm_16x2048x7168 0.997 (0.004); pt_decode_1024_256 0.981 (0.019) | 0 / 0 / 2 |
| splitk-atomic | rtx-3090 | shipped OFF/shipped ON | mm_128x2048x7168 0.898 (0.001); mm_16x1024x4096 0.998 (0.000); mm_16x2048x7168 1.000 (0.011); pt_decode_1024_256 0.950 (0.101); pt_loopmm_1024_8 1.044 (0.122); pt_rt_mlp2_1024 1.025 (0.133) | 2 / 0 / 4 |
| splitk-atomic | rtx-4090 | shipped ON | pt_decode_1024_256 0.982 (0.013) | 0 / 0 / 1 |
| splitk-atomic | rtx-5090 | shipped ON | pt_decode_1024_256 0.990 (0.037) | 0 / 0 / 1 |
| streamk | h100-pcie | shipped OFF/shipped ON | mm_128x2048x7168 0.599 (0.000); mm_16x1024x4096 0.921 (0.011); mm_16x18432x7168 0.749 (0.001); mm_4096x4096x4096 1.134 (0.000); pt_decode_1024_256 0.978 (0.019); pt_loopmm_1024_8 1.025 (0.395); pt_rt_mlp2_1024 0.984 (0.509) | 3 / 1 / 3 |
| streamk | rtx-3090 | shipped OFF | mm_128x2048x7168 1.055 (0.001); mm_16x1024x4096 0.865 (0.000); mm_16x18432x7168 1.033 (0.004); mm_16x2048x7168 1.413 (0.011); mm_4096x4096x4096 1.114 (0.001) | 1 / 4 / 0 |
| streamk | rtx-4090 | shipped OFF | mm_16x18432x7168 1.058 (0.000) | 0 / 1 / 0 |
| streamk | rtx-5090 | shipped OFF/shipped ON | mm_128x2048x7168 0.861 (0.011); mm_16x1024x4096 1.000 (0.040); mm_16x18432x7168 0.901 (0.004); mm_16x2048x7168 1.076 (0.000); mm_4096x4096x4096 1.006 (0.003); pt_diamond_4096 1.025 (0.004); pt_fanout_4096 1.003 (0.004); pt_indep_4096 1.020 (0.009); pt_rowsum_4096 1.028 (0.005); pt_softmax_rows_2048 1.005 (0.064); pt_softmax_rows_4096 1.008 (0.035) | 2 / 3 / 6 |

### Follow-up: warp-fill ON on the MLP contract kernels (H100, 3090)

| rung | arch | shipped | cells: ON/OFF (null) | ON faster / slower / inside null |
|---|---|---|---|---|
| warp-fill | h100-pcie | shipped OFF | pt_rm_m128_d1024_h2048_c 0.403 (0.000); pt_rm_m128_d256_h512_c 0.323 (0.001) | 2 / 0 / 0 |
| warp-fill | rtx-3090 | shipped OFF | pt_rm_m128_d1024_h2048_c 0.527 (0.000); pt_rm_m128_d256_h512_c 0.478 (0.010) | 2 / 0 / 0 |

### GEMM cells and MLP kernels: Mapal vs cuBLAS / cuBLASLt / CUTLASS (median ms, both processes)

| cell | arch | Mapal default | realization | Mapal _c | Mapal _x | cuBLAS | cuBLASLt | CUTLASS | tuned CUTLASS | default / best library | null |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mm_128x2048x7168 | rtx-4090 | 0.1659 | none | 0.1660 | 0.2068 | 0.1526 | - | 0.5386 | 0.2296 | 1.087 | 0.002 |
| mm_16x1024x4096 | rtx-4090 | 0.0379 | none | 0.0379 | 0.0686 | 0.0366 | - | 0.3062 | 0.1168 | 1.036 | 0.027 |
| mm_16x18432x7168 | rtx-4090 | 0.6195 | none | 0.6185 | 0.6195 | 0.6369 | - | 0.9770 | 0.6154 | 1.007 | 0.000 |
| mm_16x2048x7168 | rtx-4090 | 0.1004 | none | 0.1004 | 0.1067 | 0.0961 | - | 0.5345 | 0.2089 | 1.045 | 0.000 |
| mm_4096x4096x4096 | rtx-4090 | 2.7319 | none | 2.7249 | 5.0340 | 2.9860 | - | 3.0126 | 8.1592 | 0.915 | 0.014 |
| pt_rm_m128_d1024_h2048 | rtx-4090 | 0.0614 | - | = | 0.0840 | 0.0502 | 0.0490 | 0.2376 | 0.0727 | 1.253 | 0.018 |
| pt_rm_m128_d256_h512 | rtx-4090 | 0.0205 | - | = | 0.0268 | 0.0246 | 0.0183 | 0.0676 | 0.0205 | 1.120 | 0.000 |
| mm_128x2048x7168 | rtx-5090 | 0.1451 | none | 0.1451 | 0.2929 | 0.0881 | - | 0.5509 | 0.2212 | 1.647 | 0.011 |
| mm_16x1024x4096 | rtx-5090 | 0.0430 | none | 0.0446 | 0.0471 | 0.0184 | - | 0.2945 | 0.1270 | 2.337 | 0.040 |
| mm_16x18432x7168 | rtx-5090 | 0.4092 | streamk | 0.4547 | 0.3397 | 0.3380 | - | 0.5550 | 0.3707 | 1.211 | 0.004 |
| mm_16x2048x7168 | rtx-5090 | 0.0799 | none | 0.0799 | 0.0898 | 0.0492 | - | 0.5059 | 0.2191 | 1.624 | 0.000 |
| mm_4096x4096x4096 | rtx-5090 | 2.6767 | streamk | 2.6767 | 4.2809 | 2.4259 | - | 2.6665 | 5.6174 | 1.103 | 0.003 |
| pt_rm_m128_d1024_h2048 | rtx-5090 | 0.0543 | - | = | 0.1073 | 0.0646 | 0.0318 | 0.2036 | 0.0581 | 1.708 | 0.006 |
| pt_rm_m128_d256_h512 | rtx-5090 | 0.0151 | - | = | 0.0256 | 0.0164 | 0.0140 | 0.0561 | 0.0151 | 1.079 | 0.007 |
| mm_128x2048x7168 | h100-pcie | 0.5370 | streamk | 0.8970 | 0.4738 | 0.1415 | - | 0.9783 | 0.4198 | 3.795 | 0.000 |
| mm_16x1024x4096 | h100-pcie | 0.0464 | streamk | 0.1012 | 0.0900 | 0.0330 | - | 0.5524 | 0.1924 | 1.406 | 0.011 |
| mm_16x18432x7168 | h100-pcie | 0.5650 | none | 0.5645 | 0.6299 | 0.4066 | - | 1.9127 | 0.4359 | 1.390 | 0.001 |
| mm_16x2048x7168 | h100-pcie | 0.0971 | workspace S=7 | 0.1565 | 0.1694 | 0.0759 | - | 0.9686 | 0.3438 | 1.279 | 0.004 |
| mm_4096x4096x4096 | h100-pcie | 4.4265 | none | 4.4259 | 7.7627 | 4.5843 | - | 4.8571 | 10.97 | 0.966 | 0.000 |
| pt_rm_m128_d1024_h2048 | h100-pcie | 0.3823 | - | = | 0.1690 | 0.0874 | 0.0794 | 0.4068 | 0.1583 | 4.815 | 0.001 |
| pt_rm_m128_d256_h512 | h100-pcie | 0.1035 | - | = | 0.0429 | 0.0367 | 0.0293 | 0.1143 | 0.0309 | 3.532 | 0.004 |
| mm_128x2048x7168 | rtx-3090 | 0.2693 | atomic S=8 | 0.6470 | 0.3512 | 0.1649 | - | 0.6893 | 0.4895 | 1.633 | 0.001 |
| mm_16x1024x4096 | rtx-3090 | 0.0891 | atomic S=4 | 0.0891 | 0.0625 | 0.0340 | - | 0.4239 | 0.1279 | 2.621 | 0.000 |
| mm_16x18432x7168 | rtx-3090 | 0.6205 | none | 0.6205 | 0.6451 | 0.6223 | - | 1.3640 | 0.6103 | 1.017 | 0.004 |
| mm_16x2048x7168 | rtx-3090 | 0.1116 | workspace S=4 | 0.1229 | 0.1229 | 0.0870 | - | 0.7383 | 0.2212 | 1.283 | 0.011 |
| mm_4096x4096x4096 | rtx-3090 | 5.4282 | none | 5.4651 | 9.4024 | 5.3627 | - | 5.4927 | 14.40 | 1.012 | 0.001 |
| pt_rm_m128_d1024_h2048 | rtx-3090 | 0.2949 | - | = | 0.1690 | 0.0990 | 0.0891 | 0.3318 | 0.1619 | 3.310 | 0.003 |
| pt_rm_m128_d256_h512 | rtx-3090 | 0.0644 | - | = | 0.0377 | 0.0337 | 0.0266 | 0.0922 | 0.0266 | 2.421 | 0.000 |

### Workload suite: iter ms (median of the reps), CPU-only / GPU-only / CPU+GPU, libraries, PyTorch

| workload | arch | CPU-only | GPU-only | CPU+GPU (shipped) | null | kernel / copy / host ms, launches, MB (traced rep) | best library | torch eager | torch compile | shipped / best other |
|---|---|---|---|---|---|---|---|---|---|---|
| conv_2048_5 | rtx-4090 | 1.3743 | = shipped | 1.2274 | 0.138 | - | - | 0.0891 | 0.0891 | 13.775 |
| decode_1024_256 | rtx-4090 | 3142.71 | = shipped | 136.76 | 0.008 | 46.83 / 14.83 / 74.92, 4096, 4 | cache 23.6650 | 152.8924 | 141.6541 | 5.779 |
| diamond_4096 | rtx-4090 | 931.71 | = shipped | 7.4382 | 0.001 | 7.39 / 0.03 / 0.00, 3, 0 | cutlass 9.1372 | 8.8474 | 8.6774 | 0.857 |
| ew_2048 | rtx-4090 | 1.2512 | = shipped | 0.2966 | 0.244 | 0.04 / 0.04 / 0.33, 1, 0 | fused 0.0584 | 0.0901 | 0.0596 | 5.078 |
| ew_4096 | rtx-4090 | 7.1440 | = shipped | 0.3720 | 0.133 | 0.08 / 0.05 / 0.12, 1, 0 | fused 0.2255 | 0.5202 | 0.2255 | 1.650 |
| fanout_4096 | rtx-4090 | 664.00 | = shipped | 4.9482 | 0.002 | 4.89 / 0.06 / 0.00, 2, 0 | streams 5.8061 | 5.8706 | 5.8778 | 0.852 |
| fir_4194304_16 | rtx-4090 | 1.2387 | = shipped | 0.0445 | 0.018 | 0.04 / 0.05 / 0.00, 1, 0 | - | 0.0707 | 0.0625 | 0.711 |
| indep_4096 | rtx-4090 | 614.93 | = shipped | 5.0279 | 0.001 | 5.00 / 0.04 / 0.00, 2, 0 | cutlass 5.9617 | 5.8860 | 5.7600 | 0.873 |
| loop_2048 | rtx-4090 | 1547.05 | = shipped | 38.46 | 0.038 | 2.63 / 1.02 / 34.49, 8, 16 | cutlass 2.5846 | 2.7412 | 2.4495 | 15.703 |
| loopmm_1024_8 | rtx-4090 | 189.75 | = shipped | 13.12 | 0.022 | 0.62 / 0.31 / 12.09, 8, 4 | - | 0.3912 | 0.4342 | 33.534 |
| rm_m128_d1024_h2048 | rtx-4090 | 5.0822 | = shipped | 0.1332 | 0.143 | 0.07 / 0.03 / 0.00, 2, 0 | - | 0.0768 | 0.0604 | 2.204 |
| rm_m128_d256_h512 | rtx-4090 | 0.3495 | = shipped | 0.1557 | 0.048 | 0.03 / 0.04 / 0.09, 2, 0 | - | 0.0256 | 0.0225 | 6.921 |
| rowsum_4096 | rtx-4090 | 345.42 | = shipped | 2.5594 | 0.005 | 2.51 / 0.04 / 0.00, 2, 0 | gemv 2.8078 | 2.8467 | 2.8058 | 0.912 |
| rt_mlp2_1024 | rtx-4090 | 769.07 | = shipped | 28.04 | 0.002 | 2.35 / 6.58 / 18.75, 32, 96 | - | 2.5989 | 1.7183 | 16.320 |
| softmax_rows_2048 | rtx-4090 | 50.00 | = shipped | 6.2131 | 0.006 | 0.37 / 1.37 / 4.39, 3, 16 | fused 0.3543 | 0.5140 | 0.3267 | 19.018 |
| softmax_rows_4096 | rtx-4090 | 362.23 | = shipped | 25.86 | 0.001 | 2.64 / 5.21 / 17.80, 3, 64 | fused 2.8938 | 4.4216 | 2.9768 | 8.935 |
| conv_2048_5 | rtx-5090 | 0.4877 | = shipped | 0.4927 | 0.028 | - | - | 0.0799 | 0.0799 | 6.167 |
| decode_1024_256 | rtx-5090 | 2387.80 | = shipped | 171.87 | 0.010 | 38.01 / 9.34 / 123.58, 4096, 4 | nocache 19.7510 | 54.2205 | 52.9315 | 8.702 |
| diamond_4096 | rtx-5090 | 318.84 | = shipped | 7.5879 | 0.000 | 7.54 / 0.03 / 0.00, 3, 0 | streams 7.6782 | 7.1352 | 7.0175 | 1.081 |
| ew_2048 | rtx-5090 | 0.3858 | = shipped | 0.2783 | 0.029 | 0.04 / 0.04 / 0.22, 1, 0 | fused 0.0348 | 0.0512 | 0.0328 | 8.485 |
| ew_4096 | rtx-5090 | 2.2870 | = shipped | 0.3272 | 0.417 | 0.06 / 0.03 / 0.23, 1, 0 | fused 0.1311 | 0.2416 | 0.1311 | 2.495 |
| fanout_4096 | rtx-5090 | 220.32 | = shipped | 4.9674 | 0.001 | 4.93 / 0.05 / 0.00, 2, 0 | streams 4.6921 | 4.7596 | 4.7186 | 1.059 |
| fir_4194304_16 | rtx-5090 | 0.5503 | = shipped | 0.0309 | 0.006 | 0.02 / 0.03 / 0.00, 1, 0 | - | 0.0532 | 0.0492 | 0.628 |
| indep_4096 | rtx-5090 | 194.29 | = shipped | 5.0399 | 0.000 | 5.00 / 0.04 / 0.00, 2, 0 | streams 4.8294 | 4.8076 | 4.7237 | 1.067 |
| loop_2048 | rtx-5090 | 1105.22 | = shipped | 22.64 | 0.014 | 2.31 / 0.52 / 19.54, 8, 16 | lt 2.1606 | 2.2190 | 2.1944 | 10.481 |
| loopmm_1024_8 | rtx-5090 | 138.45 | = shipped | 4.6312 | 0.011 | 0.46 / 0.14 / 3.98, 8, 4 | - | 0.3707 | 0.3645 | 12.706 |
| rm_m128_d1024_h2048 | rtx-5090 | 2.1940 | = shipped | 0.0840 | 0.021 | 0.06 / 0.03 / 0.00, 2, 0 | - | 0.0420 | 0.0410 | 2.049 |
| rm_m128_d256_h512 | rtx-5090 | 0.2294 | = shipped | 0.0489 | 0.032 | 0.02 / 0.04 / 0.00, 2, 0 | - | 0.0184 | 0.0184 | 2.658 |
| rowsum_4096 | rtx-5090 | 115.11 | = shipped | 2.5616 | 0.001 | 2.53 / 0.03 / 0.00, 2, 0 | eager 2.3235 | 2.3091 | 2.3040 | 1.112 |
| rt_mlp2_1024 | rtx-5090 | 574.01 | = shipped | 12.68 | 0.010 | 2.02 / 2.73 / 8.23, 32, 96 | - | 2.1269 | 1.6139 | 7.859 |
| softmax_rows_2048 | rtx-5090 | 20.47 | = shipped | 3.2561 | 0.007 | 0.32 / 0.60 / 2.22, 3, 16 | eager 0.2969 | 0.3779 | 0.2724 | 11.953 |
| softmax_rows_4096 | rtx-5090 | 131.06 | = shipped | 14.75 | 0.004 | 2.61 / 2.14 / 9.89, 3, 64 | fused 2.4033 | 2.9522 | 2.3357 | 6.314 |
| conv_2048_5 | h100-pcie | 1.0873 | = shipped | 1.1127 | 0.040 | - | - | 0.7786 | 0.7786 | 1.429 |
| decode_1024_256 | h100-pcie | 3089.08 | = shipped | 300.43 | 0.005 | 110.13 / 16.61 / 173.47, 5120, 4 | cache 25.8020 | 166.3550 | 178.6163 | 11.644 |
| diamond_4096 | h100-pcie | 404.19 | = shipped | 13.57 | 0.001 | 13.53 / 0.05 / 0.00, 3, 0 | streams 13.0776 | 13.9744 | 13.9209 | 1.038 |
| ew_2048 | h100-pcie | 1.2533 | = shipped | 0.6566 | 0.018 | 0.07 / 0.07 / 0.65, 1, 0 | fused 0.0376 | 0.0585 | 0.0332 | 19.777 |
| ew_4096 | h100-pcie | 4.8964 | = shipped | 0.7715 | 0.048 | 0.13 / 0.06 / 0.70, 1, 0 | fused 0.1279 | 0.2708 | 0.1136 | 6.791 |
| fanout_4096 | h100-pcie | 279.09 | = shipped | 9.0416 | 0.002 | 9.00 / 0.08 / 0.00, 2, 0 | streams 9.2540 | 9.3915 | 9.3164 | 0.977 |
| fir_4194304_16 | h100-pcie | 1.0999 | = shipped | 0.0591 | 0.028 | 0.05 / 0.05 / 0.00, 1, 0 | - | 0.0929 | 0.0930 | 0.637 |
| indep_4096 | h100-pcie | 276.31 | = shipped | 9.0852 | 0.000 | 9.05 / 0.06 / 0.00, 2, 0 | streams 8.4542 | 9.3522 | 9.2975 | 1.075 |
| loop_2048 | h100-pcie | 1511.23 | = shipped | 35.00 | 0.004 | 6.35 / 3.15 / 25.77, 8, 16 | lt 5.0806 | 5.1911 | 5.1900 | 6.890 |
| loopmm_1024_8 | h100-pcie | 209.27 | = shipped | 8.2035 | 0.014 | 0.94 / 0.61 / 6.02, 8, 4 | - | 0.7638 | 0.9353 | 10.740 |
| rm_m128_d1024_h2048 | h100-pcie | 7.4758 | = shipped | 0.4910 | 0.066 | 0.41 / 0.05 / 0.00, 2, 0 | - | 0.0842 | 0.0849 | 5.832 |
| rm_m128_d256_h512 | h100-pcie | 0.8147 | = shipped | 0.2805 | 0.179 | 0.12 / 0.05 / 0.00, 2, 0 | - | 0.0358 | 0.0652 | 7.835 |
| rowsum_4096 | h100-pcie | 169.61 | = shipped | 4.7801 | 0.002 | 4.74 / 0.05 / 0.00, 2, 0 | gemv 4.6627 | 4.6341 | 4.6128 | 1.036 |
| rt_mlp2_1024 | h100-pcie | 773.25 | = shipped | 51.75 | 0.004 | 7.15 / 31.12 / 13.81, 32, 96 | - | 4.3248 | 3.5876 | 14.424 |
| softmax_rows_2048 | h100-pcie | 30.49 | = shipped | 16.09 | 0.029 | 0.91 / 10.10 / 4.44, 3, 16 | fused 0.6873 | 0.8500 | 0.6490 | 24.789 |
| softmax_rows_4096 | h100-pcie | 176.27 | = shipped | 55.09 | 0.082 | 4.84 / 4.34 / 19.50, 3, 64 | fused 4.8548 | 5.7829 | 4.7212 | 11.668 |
| conv_2048_5 | rtx-3090 | 1.9038 | = shipped | 1.8971 | 0.014 | - | - | 0.1741 | 0.1618 | 11.725 |
| decode_1024_256 | rtx-3090 | 5109.57 | = shipped | 575.73 | 0.020 | 43.00 / 18.60 / 514.08, 4096, 4 | cache 36.7420 | 269.6673 | 257.7155 | 15.670 |
| diamond_4096 | rtx-3090 | 1293.89 | = shipped | 16.01 | 0.001 | 15.95 / 0.06 / 0.00, 3, 0 | streams 15.5812 | 16.0410 | 15.8392 | 1.028 |
| ew_2048 | rtx-3090 | 1.9157 | = shipped | 0.2746 | 0.249 | 0.06 / 0.05 / 0.23, 1, 0 | fused 0.0635 | 0.1475 | 0.0625 | 4.394 |
| ew_4096 | rtx-3090 | 8.2980 | = shipped | 0.4073 | 0.164 | 0.15 / 0.04 / 0.22, 1, 0 | fused 0.2406 | 0.5632 | 0.2406 | 1.693 |
| fanout_4096 | rtx-3090 | 916.39 | = shipped | 10.71 | 0.000 | 10.63 / 0.07 / 0.00, 2, 0 | streams 10.6168 | 10.7776 | 10.6957 | 1.009 |
| fir_4194304_16 | rtx-3090 | 1.6953 | = shipped | 0.0832 | 0.010 | 0.07 / 0.05 / 0.00, 1, 0 | - | 0.1147 | 0.1055 | 0.789 |
| indep_4096 | rtx-3090 | 844.31 | = shipped | 10.72 | 0.001 | 10.65 / 0.04 / 0.00, 2, 0 | streams 10.2111 | 10.6905 | 10.5727 | 1.050 |
| loop_2048 | rtx-3090 | 3590.10 | = shipped | 79.11 | 0.029 | 6.61 / 2.11 / 68.69, 8, 16 | lt 5.6545 | 6.0026 | 6.0334 | 13.990 |
| loopmm_1024_8 | rtx-3090 | 350.07 | = shipped | 20.15 | 0.007 | 1.37 / 0.88 / 17.81, 8, 4 | - | 0.8663 | 0.8632 | 23.339 |
| rm_m128_d1024_h2048 | rtx-3090 | 6.1963 | = shipped | 0.3495 | 0.007 | 0.31 / 0.06 / 0.00, 2, 0 | - | 0.0860 | 0.0922 | 4.064 |
| rm_m128_d256_h512 | rtx-3090 | 0.6668 | = shipped | 0.1229 | 0.008 | 0.07 / 0.04 / 0.00, 2, 0 | - | 0.0266 | 0.0673 | 4.622 |
| rowsum_4096 | rtx-3090 | 488.73 | = shipped | 5.5507 | 0.001 | 5.29 / 0.05 / 0.00, 2, 0 | eager 5.3066 | 5.3156 | 5.3023 | 1.047 |
| rt_mlp2_1024 | rtx-3090 | 1279.74 | = shipped | 47.07 | 0.005 | 5.37 / 10.28 / 30.72, 32, 96 | - | 6.0078 | 3.7747 | 12.469 |
| softmax_rows_2048 | rtx-3090 | 64.55 | = shipped | 11.36 | 0.010 | 0.91 / 2.54 / 6.62, 3, 16 | fused 0.8141 | 1.3251 | 0.7578 | 14.996 |
| softmax_rows_4096 | rtx-3090 | 510.51 | = shipped | 42.52 | 0.003 | 5.65 / 5.47 / 30.73, 3, 64 | fused 5.6105 | 7.6196 | 5.3975 | 7.878 |

### Re-judged GPU_RUNGS_OFF rows

| row | part | rung | fill | face | cells | status | evidence (ON vs shipped) | re-measured ON/shipped |
|---|---|---|---|---|---|---|---|---|
| 0 | rtx-4090 | async | Filled | Any | mm_1024_c mm_1024_x mm_2048_c mm_2048_x mm_4096_c mm_4096_x mm_8192_c mm_8192_x | explained | mm_2048_c sass.hot.ffma_per_ldg 256.0->102.4; mm_2048_c sass.hot.ffma_per_lds 21.33->20.69; mm_2048_x sass.hot.ffma_per_ldg 256.0->102.4; mm_2048_x sass.hot.ffma_per_lds 21.33->20.69; mm_4096_c sass.hot.ffma_per_ldg 256. | - |
| 1 | rtx-4090 | warp | Filled | Contract | mm_2048_c mm_4096_c mm_8192_c | no-op at cited cells | mm_2048_c: ON arm = shipped; mm_4096_c: ON arm = shipped; mm_8192_c: ON arm = shipped | - |
| 2 | rtx-4090 | ldmatrix | Any | Any | mm_1024_tf mm_2048_tf mm_4096_tf mm_8192_tf | open | mm_1024_tf: 17 other deltas (ptx.hot.ld_shared, ptx.hot.ldmatrix, ptx.i64_arith, ptx.ld_shared); mm_2048_tf: 15 other deltas (ptx.hot.ld_shared, ptx.hot.ldmatrix, ptx.i64_arith, ptx.ld_shared); mm_4096_tf: 15 other delta | - |
| 3 | rtx-4090 | idx32 | Under | Contract | mm_1024_c | open | mm_1024_c: 17 other deltas (ptx.hot.i64_arith, ptx.hot.lines, ptx.i64_arith, sass.hot.imad_wide) | - |
| 4 | rtx-4090 | affine | Under | Contract | mm_1024_c | open | mm_1024_c: 11 other deltas (ptx.i64_arith, ptxas.regs, sass.hot.insns, sass.hot.span) | - |
| 5 | h100-pcie | async | Any | Any | mm_1024_c mm_1024_x mm_2048_c mm_2048_x mm_4096_c mm_4096_x mm_8192_c mm_8192_x | explained | mm_1024_c occ.warps_per_sm 12->8; mm_1024_c sass.hot.ffma_per_ldg 170.67->56.89; mm_1024_x sass.hot.ffma_per_ldg 170.67->56.89; mm_2048_c sass.hot.ffma_per_ldg 256.0->102.4; mm_2048_c sass.hot.ffma_per_lds 16.0->15.63; m | - |
| 6 | h100-pcie | min-blocks | Under | Contract | mm_1024_c mm_2048_c mm_4096_c mm_8192_c | explained | mm_2048_c ptxas.spill_st 0->16; mm_4096_c ptxas.spill_st 0->16; mm_8192_c ptxas.spill_st 0->16; mm_1024_c: 8 other deltas (ptx.minnctapersm, ptxas.regs, sass.hot.ldg_cover_min, sass.insns) | - |
| 7 | h100-pcie | wide-fma | Any | Any | mm_2048_c mm_2048_x mm_4096_c mm_4096_x mm_8192_c mm_8192_x | explained | mm_2048_c occ.resident_warps_per_sm 16.0->8.0; mm_2048_c occ.warps_per_sm 16->8; mm_2048_c sass.hot.span 69.62->136.5; mm_2048_x occ.resident_warps_per_sm 16.0->8.0; mm_2048_x occ.warps_per_sm 16->8; mm_2048_x ptxas.spil | - |
| 8 | h100-pcie | idx32 | Any | Exact | mm_2048_x mm_4096_x mm_8192_x | explained | mm_2048_x ptxas.spill_st 0->12; mm_4096_x ptxas.spill_st 0->12; mm_8192_x ptxas.spill_st 0->12 | - |
| 9 | h100-pcie | affine | Any | Exact | mm_2048_x mm_4096_x mm_8192_x | explained | mm_2048_x occ.resident_warps_per_sm 16.0->8.0; mm_2048_x occ.warps_per_sm 16->8; mm_4096_x occ.resident_warps_per_sm 16.0->8.0; mm_4096_x occ.warps_per_sm 16->8; mm_8192_x occ.resident_warps_per_sm 16.0->8.0; mm_8192_x o | - |
| 10 | h100-pcie | affine | Under | Contract | mm_1024_c mm_2048_c mm_4096_c mm_8192_c | open | +mm_1024_c occ.warps_per_sm 12->16; mm_2048_c: rung already ON here (other class); mm_4096_c: rung already ON here (other class); mm_8192_c: rung already ON here (other class) | - |
| 11 | h100-pcie | pad | Filled | Exact | mm_1024_x mm_2048_x mm_4096_x mm_8192_x | explained | mm_2048_x occ.resident_warps_per_sm 16.0->8.0; mm_2048_x occ.warps_per_sm 16->8; mm_4096_x occ.resident_warps_per_sm 16.0->8.0; mm_4096_x occ.warps_per_sm 16->8; mm_8192_x occ.resident_warps_per_sm 16.0->8.0; mm_8192_x o | - |
| 12 | h100-pcie | min-blocks | Filled | Any | mm_1024_c mm_1024_x mm_2048_c mm_2048_x mm_4096_c mm_4096_x mm_8192_c mm_8192_x | explained | mm_2048_c ptxas.spill_st 0->16; mm_2048_x ptxas.spill_st 0->4; mm_4096_c ptxas.spill_st 0->16; mm_4096_x ptxas.spill_st 0->4; mm_8192_c ptxas.spill_st 0->16; mm_8192_x ptxas.spill_st 0->4; mm_1024_c: 8 other deltas (ptx. | - |
| 13 | h100-pcie | warp | Under | Contract | mm_1024_c mm_2048_c mm_4096_c mm_8192_c | no-op at cited cells | mm_1024_c: ON arm = shipped; mm_2048_c: rung already ON here (other class); mm_4096_c: rung already ON here (other class); mm_8192_c: rung already ON here (other class) | - |
| 14 | h100-pcie | warp-fill | Partial | Contract | mm_1024_c | explained | mm_1024_c sass.hot.ffma_per_lds 16.0->10.67; +mm_1024_c occ.resident_warps_per_sm 4.49->8.98; +mm_1024_c occ.warps_per_sm 12->24; +mm_1024_c sass.hot.span 71.12->36.81 | - |
| 15 | rtx-5090 | async | Filled | Contract | mm_1024_c mm_2048_c mm_4096_c mm_8192_c | explained | mm_1024_c occ.warps_per_sm 24->16; mm_1024_c sass.hot.ffma_per_ldg 128.0->51.2; mm_1024_c sass.hot.ffma_per_lds 8.0->7.64; mm_2048_c sass.hot.ffma_per_ldg 256.0->102.4; mm_2048_c sass.hot.ffma_per_lds 16.0->15.28; mm_409 | - |
| 16 | rtx-5090 | wide-fma | Any | Any | mm_2048_c mm_4096_c mm_8192_c | explained | mm_2048_c occ.resident_warps_per_sm 12.05->6.02; mm_2048_c occ.warps_per_sm 16->8; mm_2048_c sass.hot.span 69.62->136.5; mm_4096_c occ.resident_warps_per_sm 16.0->8.0; mm_4096_c occ.warps_per_sm 16->8; mm_4096_c sass.hot | - |
| 17 | rtx-5090 | idx32 | Any | Exact | mm_1024_x mm_2048_x mm_4096_x mm_8192_x | open | +mm_2048_x ptxas.spill_st 20->0; +mm_4096_x ptxas.spill_st 20->0; +mm_8192_x ptxas.spill_st 20->0; mm_1024_x: 22 other deltas (ptx.hot.i64_arith, ptx.hot.lines, ptx.i64_arith, ptxas.regs) | - |
| 18 | rtx-5090 | affine | Filled | Exact | mm_1024_x mm_2048_x mm_4096_x mm_8192_x | open | +mm_2048_x ptxas.spill_st 20->12; +mm_4096_x ptxas.spill_st 20->12; +mm_8192_x ptxas.spill_st 20->12; mm_1024_x: rung already ON here (other class) | - |
| 19 | rtx-5090 | pad | Any | Exact | mm_2048_x mm_4096_x mm_8192_x | no-op at cited cells | mm_2048_x: ON arm = shipped; mm_4096_x: ON arm = shipped; mm_8192_x: ON arm = shipped | - |
| 20 | rtx-5090 | idx32 | Under | Contract | mm_1024_c | open | mm_1024_c: 18 other deltas (ptx.hot.i64_arith, ptx.hot.lines, ptx.i64_arith, ptxas.regs) | - |
| 21 | rtx-5090 | affine | Under | Contract | mm_1024_c | no-op at cited cells | mm_1024_c: ON arm = shipped | - |
| 22 | rtx-5090 | warp-fill | Filled | Contract | mm_2048_c mm_2048_tf | no-op at cited cells | mm_2048_c: ON arm = shipped; mm_2048_tf: ON arm = shipped | - |
| 23 | rtx-4090 | retain-delta | Any | Any | decode_1024_256_c decode_1024_256_x | explained | decode_1024_256_c host.calls.tile 0->6; decode_1024_256_c sass.hot.span 0->3.16; decode_1024_256_x host.calls.tile 0->6; decode_1024_256_x sass.hot.span 0->6.25; +decode_1024_256_c host.calls.alloc 8->0; +decode_1024_256 | - |
| 24 | h100-pcie | retain-delta | Any | Any | decode_1024_256_c decode_1024_256_x | explained | decode_1024_256_c host.calls.tile 0->6; decode_1024_256_c sass.hot.span 0->5.75; decode_1024_256_x host.calls.tile 0->6; decode_1024_256_x sass.hot.span 0->4.25; +decode_1024_256_c host.calls.alloc 8->0; +decode_1024_256 | - |
| 25 | rtx-5090 | async | Under | Any | mm_1024_c mm_1024_x | explained | mm_1024_c occ.warps_per_sm 24->16; mm_1024_c sass.hot.ffma_per_ldg 128.0->51.2; mm_1024_c sass.hot.ffma_per_lds 8.0->7.64; mm_1024_x occ.warps_per_sm 24->16; mm_1024_x sass.hot.ffma_per_ldg 128.0->51.2; mm_1024_x sass.ho | - |
| 26 | rtx-5090 | wide | Under | Contract | mm_1024_tf | explained | mm_1024_tf occ.resident_warps_per_sm 12.05->8.0; mm_1024_tf occ.warps_per_sm 16->8 | - |
| 27 | rtx-5090 | retain-delta | Any | Any | decode_1024_256_c decode_1024_256_x | explained | decode_1024_256_c host.calls.tile 0->6; decode_1024_256_c sass.hot.span 0->3.08; decode_1024_256_x host.calls.tile 0->6; decode_1024_256_x sass.hot.span 0->4.12; +decode_1024_256_c host.calls.alloc 8->0; +decode_1024_256 | - |
| 28 | rtx-5090 | wide | Filled | Contract | mm_2048_tf mm_4096_tf mm_8192_tf | explained | mm_2048_tf occ.resident_warps_per_sm 12.05->6.02; mm_2048_tf occ.warps_per_sm 16->8; mm_4096_tf occ.resident_warps_per_sm 16.0->8.0; mm_4096_tf occ.warps_per_sm 16->8; mm_8192_tf occ.resident_warps_per_sm 16.0->8.0; mm_8 | - |
| 29 | rtx-3090 | affine | Under | Any | mm_1024_c mm_1024_x | no-op at cited cells | mm_1024_c: ON arm = shipped; mm_1024_x: ON arm = shipped | - |
| 30 | rtx-3090 | async | Any | Any | mm_1024_c mm_1024_x mm_2048_c mm_2048_x mm_4096_c mm_4096_x mm_8192_c mm_8192_x | explained | mm_1024_c sass.hot.ffma_per_ldg 170.67->56.89; mm_1024_c sass.hot.ffma_per_lds 16.0->14.63; mm_1024_c sass.hot.span 73.75->75.75; mm_1024_x sass.hot.ffma_per_ldg 170.67->56.89; mm_1024_x sass.hot.ffma_per_lds 10.67->10.0 | - |
| 31 | rtx-3090 | idx32 | Any | Exact | mm_1024_x mm_2048_x mm_4096_x mm_8192_x | explained | mm_2048_x ptxas.spill_st 0->12; mm_4096_x ptxas.spill_st 0->12; mm_8192_x ptxas.spill_st 0->12; mm_1024_x: 21 other deltas (ptx.hot.distinct_load_bases, ptx.hot.i64_arith, ptx.hot.lines, ptx.i64_arith) | - |
| 32 | rtx-3090 | idx32 | Under | Contract | mm_1024_c | open | +mm_1024_c ptxas.spill_st 8->0; +mm_1024_c sass.hot.span 73.75->71.38 | - |
| 33 | rtx-3090 | interleave | Under | Any | mm_1024_c mm_1024_x | explained | mm_1024_c ptxas.spill_st 8->12; mm_1024_x: 8 other deltas (ptx.i64_arith, sass.iadd3, sass.imad, sass.insns) | - |
| 34 | rtx-3090 | ldmatrix | Any | Contract | mm_1024_tf mm_2048_tf mm_4096_tf mm_8192_tf | open | mm_1024_tf: 14 other deltas (ptx.hot.ld_shared, ptx.hot.ldmatrix, ptx.i64_arith, ptx.ld_shared); mm_2048_tf: 16 other deltas (ptx.hot.ld_shared, ptx.hot.ldmatrix, ptx.i64_arith, ptx.ld_shared); mm_4096_tf: 16 other delta | - |
| 35 | rtx-3090 | min-blocks | Any | Tf32 | mm_1024_tf mm_2048_tf mm_4096_tf mm_8192_tf | open | mm_1024_tf: 7 other deltas (ptx.minnctapersm, ptxas.regs, sass.imad, sass.lop3); mm_2048_tf: 7 other deltas (ptx.minnctapersm, ptxas.regs, sass.imad, sass.insns); mm_4096_tf: 7 other deltas (ptx.minnctapersm, ptxas.regs, | - |
| 36 | rtx-3090 | warp | Under | Any | mm_1024_c mm_1024_x | open | mm_1024_c: ON arm = shipped; mm_1024_x: 6 other deltas (ptx.i64_arith, ptxas.regs, sass.hot.ldg_cover_min, sass.lop3) | - |
| 37 | rtx-3090 | warp-fill | Under | Contract | mm_1024_c mm_1024_rc mm_1024_tf mm_128x2048x7168_c mm_128x2048x7168_rc mm_128x2048x7168_tf | explained | mm_1024_c sass.hot.ffma_per_lds 16.0->10.67; +mm_1024_c occ.resident_warps_per_sm 6.24->12.49; +mm_1024_c ptxas.spill_st 8->0; +mm_1024_c sass.hot.span 73.75->36.94; mm_1024_rc: not built; mm_128x2048x7168_c: ON arm = sh | - |
| 38 | rtx-3090 | wide | Under | Contract | mm_1024_tf mm_2048_tf mm_4096_tf mm_8192_tf | explained | mm_1024_tf occ.resident_warps_per_sm 6.24->3.12; mm_1024_tf occ.warps_per_sm 16->10; mm_2048_tf: rung already ON here (other class); mm_4096_tf: rung already ON here (other class); mm_8192_tf: rung already ON here (other | - |
| 39 | rtx-3090 | wide-fma | Filled | Exact | mm_2048_x mm_4096_x mm_8192_x | explained | mm_2048_x occ.resident_warps_per_sm 16.0->8.0; mm_2048_x occ.warps_per_sm 16->8; mm_2048_x sass.hot.span 134.5->264.94; mm_4096_x occ.resident_warps_per_sm 16.0->8.0; mm_4096_x occ.warps_per_sm 16->8; mm_4096_x sass.hot. | - |
| 40 | rtx-3090 | retain-delta | Any | Any | decode_1024_256_c decode_1024_256_x | no-op at cited cells | decode_1024_256_c: ON arm = shipped; decode_1024_256_x: ON arm = shipped | - |
| 41 | a100 | affine | Any | Exact | - | unverified (no box) | - | - |
| 42 | a100 | async | Any | Any | - | unverified (no box) | - | - |
| 43 | a100 | idx32 | Any | Exact | - | unverified (no box) | - | - |
| 44 | a100 | idx32 | Under | Contract | - | unverified (no box) | - | - |
| 45 | a100 | interleave | Under | Any | - | unverified (no box) | - | - |
| 46 | a100 | min-blocks | Filled | Any | - | unverified (no box) | - | - |
| 47 | a100 | unroll | Any | Exact | - | unverified (no box) | - | - |
| 48 | a100 | warp | Any | Any | - | unverified (no box) | - | - |
| 49 | a100 | wide | Under | Contract | - | unverified (no box) | - | - |
| 50 | a100 | wide-fma | Filled | Any | - | unverified (no box) | - | - |
| 51 | a100 | ldmatrix | Under | Tf32 | - | unverified (no box) | - | - |
| 52 | rtx-3090 | rowfold-layout | Any | Any | pt_rowsum_2048_c pt_rowsum_2048_x pt_rowsum_4096_c pt_rowsum_4096_x | explained | pt_rowsum_2048_x ptxas.spill_st 0->12; pt_rowsum_4096_x ptxas.spill_st 0->12; pt_rowsum_2048_c: 14 other deltas (plan.col, ptx.i64_arith, ptxas.regs, sass.hot.ldg_cover_min); pt_rowsum_4096_c: 14 other deltas (plan.col,  | pt_rowsum_4096_c ON/OFF 1.024/1.013/1.023/1.029/1.024/1.018/1.026 null 0.036 inside; pt_rowsum_4096_x ON/OFF 0.945/1.001/1.008/1.011/1.008/1.005/1.007 null 0.036 inside |
| 53 | rtx-5090 | rowfold-layout | Any | Any | pt_rowsum_2048_c pt_rowsum_2048_x pt_rowsum_4096_c pt_rowsum_4096_x | open (ON slower, no static cost) | +pt_rowsum_2048_x ptxas.spill_st 20->16; +pt_rowsum_4096_x ptxas.spill_st 20->16; pt_rowsum_2048_c: 13 other deltas (plan.col, ptx.i64_arith, sass.hfma2, sass.hot.ldg_cover_min); pt_rowsum_4096_c: 13 other deltas (plan.c | pt_rowsum_4096_c ON/OFF 1.033/1.034/1.030/1.024/1.033/1.035/1.032 null 0.005 slower; pt_rowsum_4096_x ON/OFF 1.030/1.032/1.029/1.025/1.029/1.031/1.024 null 0.005 slower |
| 54 | h100-pcie | rowfold-layout | Any | Any | pt_rowsum_2048_c pt_rowsum_2048_x pt_rowsum_4096_c pt_rowsum_4096_x | explained, ON slower re-measured | pt_rowsum_2048_x occ.resident_warps_per_sm 16.0->8.0; pt_rowsum_2048_x occ.warps_per_sm 16->8; pt_rowsum_4096_x occ.resident_warps_per_sm 16.0->8.0; pt_rowsum_4096_x occ.warps_per_sm 16->8; pt_rowsum_2048_c: 10 other del | pt_rowsum_4096_c ON/OFF 1.057/1.059/1.066/1.065/1.060/1.059/1.067 null 0.007 slower; pt_rowsum_4096_x ON/OFF 1.194/1.183/1.184/1.205/1.190/1.190/1.192 null 0.007 slower |
| 55 | rtx-4090 | rowfold-layout | Any | Exact | pt_rowsum_2048_x pt_rowsum_4096_x | explained | pt_rowsum_2048_x ptxas.spill_st 0->8; pt_rowsum_4096_x ptxas.spill_st 0->8 | - |
| 56 | a100 | rowfold-layout | Any | Contract | - | unverified (no box) | - | - |
| 57 | a100 | retain-delta | Any | Any | - | unverified (no box) | - | - |
| 58 | rtx-3090 | raster | Any | Any | mm_8192_c mm_8192_x | explained | mm_8192_x ptxas.spill_st 0->20; mm_8192_c: 7 other deltas (plan.raster, sass.insns, sass.lea, sass.lop3) | - |
| 59 | rtx-3090 | l2-window | Any | Any |  | unverified (no in-class cell) |  | - |
| 60 | a100 | l2-window | Any | Any | - | unverified (no box) | - | - |
| 61 | rtx-4090 | l2-window | Any | Any | pt_diamond_2048_c pt_mlp2_2048_c | explained | pt_diamond_2048_c host.calls.launch_l2 0->2; pt_mlp2_2048_c host.calls.launch_l2 0->2; +pt_diamond_2048_c host.calls.launch 2->0; +pt_mlp2_2048_c host.calls.launch 2->0 | - |
| 62 | rtx-5090 | l2-window | Any | Any | pt_diamond_2048_c | explained | pt_diamond_2048_c host.calls.launch_l2 0->2; +pt_diamond_2048_c host.calls.launch 2->0 | - |
| 63 | h100-pcie | l2-window | Any | Any | pt_diamond_2048_c pt_diamond_2048_x pt_diamond_4096_c pt_diamond_4096_x pt_mlp2_2048_c pt_mlp2_2048_x pt_mlp2_4096_c pt_mlp2_4096_x pt_mlp_2048_c pt_mlp_2048_x pt_mlp_4096_c pt_mlp_4096_x | explained | pt_diamond_2048_c host.calls.launch_l2 0->2; pt_diamond_2048_x host.calls.launch_l2 0->2; pt_mlp2_2048_c host.calls.launch_l2 0->2; pt_mlp2_2048_x host.calls.launch_l2 0->2; pt_mlp_2048_c host.calls.launch_l2 0->1; pt_ml | - |
| 64 | h100-pcie | retain-resident | Any | Any | decode_1024_256_c decode_1024_256_x | explained | decode_1024_256_c host.calls.launch 4->6; decode_1024_256_x host.calls.launch 4->6 | - |
| 65 | rtx-5090 | retain-resident | Any | Any | decode_1024_256_c decode_1024_256_x | explained | decode_1024_256_c host.calls.launch 4->6; decode_1024_256_x host.calls.launch 4->6 | - |
| 66 | rtx-4090 | retain-resident | Any | Any | decode_1024_256_c decode_1024_256_x | explained | decode_1024_256_c host.calls.launch 4->6; decode_1024_256_x host.calls.launch 4->6 | - |
| 67 | rtx-3090 | streams | Any | Any | pt_diamond_512_c pt_diamond_512_x pt_mlp2_512_c pt_mlp2_512_x pt_twin_512_c pt_twin_512_x | explained | pt_diamond_512_c host.calls.alloc 0->9; pt_diamond_512_c host.calls.event 0->1; pt_diamond_512_c host.calls.launchv 1->3; pt_mlp2_512_c host.calls.alloc 0->14; pt_mlp2_512_c host.calls.launchv 2->4; pt_twin_512_c host.ca | - |
| 68 | rtx-4090 | streams | Any | Any | pt_diamond_512_c pt_diamond_512_x pt_mlp2_512_c pt_mlp2_512_x pt_twin_1024_c pt_twin_1024_x pt_twin_512_c pt_twin_512_x | explained | pt_diamond_512_x host.calls.alloc 0->9; pt_diamond_512_x host.calls.event 0->1; pt_diamond_512_x host.calls.launchv 1->3; pt_mlp2_512_x host.calls.alloc 0->14; pt_mlp2_512_x host.calls.launchv 2->4; pt_twin_1024_x host.c | - |
| 69 | rtx-5090 | streams | Any | Any | pt_mlp2_4096_c pt_mlp2_4096_x pt_twin_4096_c pt_twin_4096_x | explained | pt_mlp2_4096_c host.calls.alloc 0->14; pt_mlp2_4096_c host.calls.launchv 2->4; pt_mlp2_4096_x host.calls.alloc 0->14; pt_mlp2_4096_x host.calls.launchv 2->4; pt_twin_4096_c host.calls.alloc 0->8; pt_twin_4096_c host.call | - |
| 70 | h100-pcie | streams | Any | Any | pt_twin_512_c pt_twin_512_x | explained | pt_twin_512_x host.calls.alloc 0->8; pt_twin_512_x host.calls.launchv 0->2; +pt_twin_512_x host.calls.d2h 4->0; +pt_twin_512_x host.calls.launch 2->0; pt_twin_512_c: ON arm = shipped | - |
| 71 | rtx-3090 | pinned | Any | Any | pt_loop_2048_c pt_loop_2048_x | no-op at cited cells | pt_loop_2048_c: ON arm = shipped; pt_loop_2048_x: ON arm = shipped | - |
| 72 | rtx-4090 | pinned | Any | Any | pt_loop_2048_c pt_loop_2048_x | no-op at cited cells | pt_loop_2048_c: ON arm = shipped; pt_loop_2048_x: ON arm = shipped | - |
| 73 | h100-pcie | pinned | Any | Any | pt_loop_2048_c pt_loop_2048_x | no-op at cited cells | pt_loop_2048_c: ON arm = shipped; pt_loop_2048_x: ON arm = shipped | - |
| 74 | a100 | pinned | Any | Any | - | unverified (no box) | - | - |
| 75 | rtx-5090 | pinned | Any | Any | pt_loop_2048_c pt_loop_2048_x | no-op at cited cells | pt_loop_2048_c: ON arm = shipped; pt_loop_2048_x: ON arm = shipped | - |
| 76 | h100-pcie | pdl | Any | Any | pt_diamond_2048_c pt_diamond_2048_x pt_mlp_512_c pt_mlp_512_x | open | pt_diamond_2048_c: 6 other deltas (ptx.griddep, sass.nop, sass.other, ptx.griddep); pt_diamond_2048_x: 10 other deltas (ptx.griddep, sass.insns, sass.mov, sass.nop); pt_mlp_512_c: 7 other deltas (ptx.griddep, sass.insns, | - |
| 77 | rtx-5090 | pdl | Any | Any | pt_diamond_2048_c pt_diamond_2048_x pt_mlp_512_c pt_mlp_512_x | open | pt_diamond_2048_c: 8 other deltas (ptx.griddep, sass.insns, sass.nop, sass.other); pt_diamond_2048_x: 6 other deltas (ptx.griddep, sass.nop, sass.other, ptx.griddep); pt_mlp_512_c: 7 other deltas (ptx.griddep, sass.insns | - |
| 78 | rtx-5090 | wave-gate | Filled | Contract |  | unverified (no in-class cell) |  | - |
| 79 | rtx-5090 | row-fit | Filled | Exact | mm_16x18432x7168_x | explained, ON slower re-measured | mm_16x18432x7168_x occ.resident_warps_per_sm 13.55->3.39; mm_16x18432x7168_x occ.warps_per_sm 16->4; mm_16x18432x7168_x sass.hot.ffma_per_ldg 32.0->21.33; mm_16x18432x7168_x sass.hot.span 10.8->36.88; +mm_16x18432x7168_x | mm_16x18432x7168_x ON/OFF 1.007/1.011 null 0.004 slower |
| 80 | rtx-4090 | grid-share | Under | Exact | mm_128x1024x4096_x mm_128x2048x7168_x mm_16x1024x4096_x | open (ON faster at mm_128x2048x7168_x) | mm_128x1024x4096_x occ.warps_per_sm 20->2; mm_128x2048x7168_x occ.warps_per_sm 16->4; mm_16x1024x4096_x occ.warps_per_sm 18->1; +mm_128x1024x4096_x sass.hot.ffma_per_lds 7.31->7.91; +mm_128x1024x4096_x sass.hot.span 36.8 | mm_128x2048x7168_x ON/OFF 0.887/0.928 null 0.002 faster; mm_16x1024x4096_x ON/OFF 1.016/0.958 null 0.027 inside |
| 81 | rtx-4090 | row-fit | Filled | Any | mm_16x18432x7168_x | explained, ON slower re-measured | mm_16x18432x7168_x occ.resident_warps_per_sm 9.0->4.5; mm_16x18432x7168_x occ.warps_per_sm 28->24; mm_16x18432x7168_x sass.hot.span 19.44->36.88; +mm_16x18432x7168_x sass.hot.ffma_per_lds 4.0->8.0 | mm_16x18432x7168_x ON/OFF 1.045/1.048 null 0.000 slower |
| 82 | h100-pcie | grid-share | Filled | Contract |  | unverified (no in-class cell) |  | - |
| 83 | h100-pcie | row-fit | Filled | Any | mm_16x18432x7168_x | explained, ON slower re-measured | mm_16x18432x7168_x occ.resident_warps_per_sm 10.11->5.05; mm_16x18432x7168_x occ.warps_per_sm 50->28; mm_16x18432x7168_x sass.hot.span 19.94->37.38; +mm_16x18432x7168_x sass.hot.ffma_per_lds 4.0->8.0 | mm_16x18432x7168_x ON/OFF 1.147/1.144 null 0.001 slower |
| 84 | h100-pcie | wave-gate | Under | Contract |  | unverified (no in-class cell) |  | - |
| 85 | rtx-3090 | grid-share | Filled | Any | mm_16x18432x7168_x mm_16x2048x7168_x | open | mm_16x18432x7168_x: ON arm = shipped; mm_16x2048x7168_x: 2 other deltas (plan.blocks_per_sm, ptx.minnctapersm) | mm_16x2048x7168_x ON/OFF 1.000/1.000 null 0.011 inside |
| 86 | rtx-3090 | panel-regs | Filled | Any | mm_16x2048x7168_x | explained, ON slower re-measured | mm_16x2048x7168_x occ.warps_per_sm 28->16; +mm_16x2048x7168_x sass.hot.span 11.0->10.7 | mm_16x2048x7168_x ON/OFF 1.056/1.050 null 0.011 slower |
| 87 | rtx-3090 | row-fit | Filled | Any | mm_16x18432x7168_x | explained | mm_16x18432x7168_x occ.resident_warps_per_sm 14.05->7.02; mm_16x18432x7168_x occ.warps_per_sm 32->16; mm_16x18432x7168_x sass.hot.span 19.75->37.5; +mm_16x18432x7168_x sass.hot.ffma_per_lds 4.0->8.0 | mm_16x18432x7168_x ON/OFF 0.997/1.001 null 0.004 inside |
| 88 | rtx-5090 | asyncstage | Any | Any | mm_128x1024x1024_c mm_128x1024x1024_x mm_16x2048x7168_c mm_16x2048x7168_x pt_rmlp256x512_128_c pt_rmlp256x512_128_x | no-op at cited cells | mm_128x1024x1024_c: ON arm = shipped; mm_128x1024x1024_x: ON arm = shipped; mm_16x2048x7168_c: ON arm = shipped; mm_16x2048x7168_x: ON arm = shipped | - |
| 89 | rtx-4090 | asyncstage | Any | Any | mm_1024_c mm_1024_x mm_16x2048x7168_c mm_16x2048x7168_x pt_rmlp256x512_128_c pt_rmlp256x512_128_x | explained | mm_1024_x occ.warps_per_sm 16->8; pt_rmlp256x512_128_x occ.warps_per_sm 16->1; pt_rmlp256x512_128_x occ.warps_per_sm 16->1; pt_rmlp256x512_128_x occ.warps_per_sm 16->1; +pt_rmlp256x512_128_x sass.hot.span 41.25->37.75; + | - |
| 90 | h100-pcie | asyncstage | Any | Any | mm_1024_c mm_1024_x mm_128x256x256_c mm_128x256x256_x pt_rmlp1024x2048_128_c pt_rmlp1024x2048_128_x pt_rmlp256x512_128_c pt_rmlp256x512_128_x | explained | mm_128x256x256_x occ.warps_per_sm 25->4; mm_128x256x256_x sass.hot.ffma_per_ldg 42.67->14.22; mm_128x256x256_x sass.hot.ffma_per_lds 8.0->6.74; pt_rmlp1024x2048_128_x occ.warps_per_sm 24->4; pt_rmlp1024x2048_128_x sass.h | - |
| 91 | rtx-3090 | asyncstage | Any | Any | mm_1024_c mm_1024_x mm_128x2048x7168_c mm_128x2048x7168_x mm_16x2048x7168_c mm_16x2048x7168_x pt_rmlp256x512_128_c pt_rmlp256x512_128_x | open (ON faster at mm_128x2048x7168_c mm_16x2048x7168_c) | mm_1024_c occ.warps_per_sm 16->8; mm_1024_c sass.hot.ffma_per_ldg 170.67->56.89; mm_1024_c sass.hot.ffma_per_lds 16.0->14.63; mm_1024_c sass.hot.span 73.75->77.25; mm_128x2048x7168_c occ.warps_per_sm 16->2; mm_128x2048x7 | mm_128x2048x7168_c ON/OFF 0.830/0.805 null 0.001 faster; mm_16x2048x7168_c ON/OFF 0.968/0.966 null 0.011 faster; mm_16x2048x7168_x ON/OFF 1.032/0.992 null 0.011 inside |
| 92 | a100 | asyncstage | Any | Any | - | unverified (no box) | - | - |
| 93 | rtx-3090 | splitk | Filled | Exact | mm_16x2048x7168_r | open (ON faster at mm_16x2048x7168_r) | mm_16x2048x7168_r host.calls.alloc 0->1; mm_16x2048x7168_r host.calls.launch 1->2; mm_16x2048x7168_r launch.S 1->4; mm_16x2048x7168_r occ.warps_per_sm 28->24; +mm_16x2048x7168_r occ.resident_warps_per_sm 3.12->12.49 | mm_16x2048x7168_r ON/OFF 0.904/0.890 null 0.011 faster |
| 94 | h100-pcie | row-fit | Under | Contract | mm_16x1024x4096_c | explained, ON slower re-measured | mm_16x1024x4096_c occ.resident_warps_per_sm 4.49->0.07; mm_16x1024x4096_c occ.warps_per_sm 64->12; mm_16x1024x4096_c ptxas.spill_st 0->8; mm_16x1024x4096_c sass.hot.span 3.28->71.81; +mm_16x1024x4096_c sass.hot.ffma_per_ | mm_16x1024x4096_c ON/OFF 4.242/4.249 null 0.011 slower |
| 95 | h100-pcie | grid-share | Under | Exact | mm_16x1024x4096_r mm_16x1024x4096_x pt_rm_m128_d1024_h2048_x pt_rm_m128_d256_h512_x | open (ON faster at pt_rm_m128_d1024_h2048_x pt_rm_m128_d256_h512_x) | mm_16x1024x4096_r: ON arm = shipped; mm_16x1024x4096_x: ON arm = shipped; pt_rm_m128_d1024_h2048_x: 4 other deltas (plan.blocks_per_sm, ptx.minnctapersm, plan.blocks_per_sm, ptx.minnctapersm); pt_rm_m128_d256_h512_x: 4 o | pt_rm_m128_d1024_h2048_x ON/OFF 0.991/0.991 null 0.001 faster; pt_rm_m128_d256_h512_x ON/OFF 0.993/0.991 null 0.004 faster |
| 96 | h100-pcie | fill-rows | Under | Exact | mm_16x1024x4096_x | explained, ON slower re-measured | mm_16x1024x4096_x occ.resident_warps_per_sm 4.49->1.12; mm_16x1024x4096_x occ.warps_per_sm 48->32; mm_16x1024x4096_x sass.hot.ffma_per_ldg 32.0->21.33; mm_16x1024x4096_x sass.hot.span 4.39->11.62; +mm_16x1024x4096_x sass | mm_16x1024x4096_x ON/OFF 1.855/1.847 null 0.011 slower |
| 97 | h100-pcie | panel-regs | Under | Exact | pt_rm_m128_d1024_h2048_x pt_rm_m128_d256_h512_x | open (ON faster at pt_rm_m128_d256_h512_x) | pt_rm_m128_d1024_h2048_x occ.warps_per_sm 24->16; pt_rm_m128_d256_h512_x occ.warps_per_sm 25->16; pt_rm_m128_d256_h512_x occ.warps_per_sm 25->16; +pt_rm_m128_d1024_h2048_x sass.hot.span 37.12->35.14; +pt_rm_m128_d256_h51 | pt_rm_m128_d1024_h2048_x ON/OFF 1.108/1.110 null 0.001 slower; pt_rm_m128_d256_h512_x ON/OFF 0.981/0.977 null 0.004 faster |
| 98 | rtx-3090 | splitk | Under | Exact | mm_16x1024x4096_r | no-op at cited cells | mm_16x1024x4096_r: ON arm = shipped | - |
| 99 | rtx-3090 | grid-share | Under | Contract | pt_rm_m128_d1024_h2048_c pt_rm_m128_d256_h512_c | open (ON faster at pt_rm_m128_d1024_h2048_c pt_rm_m128_d256_h512_c) | pt_rm_m128_d1024_h2048_c occ.warps_per_sm 16->15; pt_rm_m128_d1024_h2048_c occ.warps_per_sm 16->15; pt_rm_m128_d256_h512_c occ.warps_per_sm 16->15; pt_rm_m128_d256_h512_c occ.warps_per_sm 16->15 | pt_rm_m128_d1024_h2048_c ON/OFF 0.993/0.993 null 0.003 faster; pt_rm_m128_d256_h512_c ON/OFF 0.952/0.952 null 0.000 faster |
| 100 | rtx-3090 | panel-regs | Under | Contract | pt_rm_m128_d1024_h2048_c pt_rm_m128_d256_h512_c | no-op at cited cells | pt_rm_m128_d1024_h2048_c: ON arm = shipped; pt_rm_m128_d256_h512_c: ON arm = shipped | - |
| 101 | rtx-5090 | streamk | Under | Contract | mm_128x2048x7168 mm_16x1024x4096 mm_16x2048x7168 | open (ON faster at mm_128x2048x7168) | mm_128x2048x7168 host.calls.zero 0->1; mm_16x1024x4096 host.calls.zero 0->1; mm_16x1024x4096 ptxas.spill_st 208->212; mm_16x1024x4096 sass.hot.span 6.89->7.12; mm_16x2048x7168 host.calls.zero 0->1; mm_16x2048x7168 ptxas. | mm_128x2048x7168 ON/OFF 0.861/0.857 null 0.011 faster; mm_16x1024x4096 ON/OFF 1.000/1.000 null 0.040 inside; mm_16x2048x7168 ON/OFF 1.051/1.076 null 0.000 slower |
| 102 | rtx-4090 | streamk | Filled | Contract | mm_16x18432x7168 | explained, ON slower re-measured | mm_16x18432x7168 host.calls.zero 0->1; mm_16x18432x7168 occ.warps_per_sm 24->22; mm_16x18432x7168 sass.red 0->8; +mm_16x18432x7168 occ.resident_warps_per_sm 9.0->9.98 | mm_16x18432x7168 ON/OFF 1.054/1.058 null 0.000 slower |
| 103 | h100-pcie | streamk | Filled | Contract | mm_16x18432x7168 mm_4096x4096x4096 | open (ON faster at mm_16x18432x7168) | mm_16x18432x7168 host.calls.zero 0->1; mm_16x18432x7168 sass.hot.span 11.25->11.56; mm_4096x4096x4096 host.calls.zero 0->1; +mm_16x18432x7168 occ.resident_warps_per_sm 10.11->16.0; +mm_16x18432x7168 occ.warps_per_sm 42-> | mm_16x18432x7168 ON/OFF 0.748/0.749 null 0.001 faster; mm_4096x4096x4096 ON/OFF 1.134/1.134 null 0.000 slower |
| 104 | h100-pcie | splitk-atomic | Any | Contract | mm_16x1024x4096 mm_16x2048x7168 | explained | mm_16x2048x7168 host.calls.zero 0->1; +mm_16x2048x7168 host.calls.alloc 1->0; +mm_16x2048x7168 host.calls.launch 2->1; mm_16x1024x4096: ON arm = shipped | mm_16x2048x7168 ON/OFF 0.997/0.992 null 0.004 inside |
| 105 | rtx-3090 | streamk | Under | Contract | mm_128x2048x7168 mm_16x1024x4096 | open (ON faster at mm_16x1024x4096) | +mm_128x2048x7168 launch.S 8->1; +mm_128x2048x7168 occ.resident_warps_per_sm 12.49->15.99; +mm_16x1024x4096 launch.S 4->1 | mm_128x2048x7168 ON/OFF 1.050/1.055 null 0.001 slower; mm_16x1024x4096 ON/OFF 0.862/0.865 null 0.000 faster |
| 106 | rtx-3090 | streamk | Filled | Contract | mm_16x18432x7168 mm_16x2048x7168 mm_4096x4096x4096 | explained, ON slower re-measured | mm_16x18432x7168 host.calls.zero 0->1; mm_16x18432x7168 occ.warps_per_sm 24->22; mm_16x18432x7168 sass.red 0->8; mm_16x2048x7168 host.calls.zero 0->1; mm_16x2048x7168 sass.red 0->4; mm_4096x4096x4096 host.calls.zero 0->1 | mm_16x18432x7168 ON/OFF 1.031/1.033 null 0.004 slower; mm_16x2048x7168 ON/OFF 1.319/1.413 null 0.011 slower; mm_4096x4096x4096 ON/OFF 1.114/1.100 null 0.001 slower |
| 107 | rtx-3090 | splitk-atomic | Filled | Contract | mm_16x2048x7168 | explained | mm_16x2048x7168 host.calls.zero 0->1; mm_16x2048x7168 sass.red 0->4; +mm_16x2048x7168 host.calls.alloc 1->0; +mm_16x2048x7168 host.calls.launch 2->1 | mm_16x2048x7168 ON/OFF 0.991/1.000 null 0.011 inside |

### End check: the warp-fill class split (ON = with the split, OFF = the pre-split build; taken on the H100, not on the 3090: `shipped` below reads the arm name)

#### How each optimization turned out in time (arms timed this session: ON / OFF, median over processes)

| rung | arch | shipped | cells: ON/OFF (null) | ON faster / slower / inside null |
|---|---|---|---|---|
| splitk | h100-pcie | shipped ON | pt_decode_1024_256 0.739 (0.163) | 1 / 0 / 0 |
| splitk | rtx-3090 | shipped ON | mm_128x2048x7168 0.994 (0.003); pt_decode_1024_256 0.797 (0.217) | 1 / 0 / 1 |
| splitk-atomic | h100-pcie | shipped ON | pt_decode_1024_256 1.004 (0.163) | 0 / 0 / 1 |
| splitk-atomic | rtx-3090 | shipped ON | mm_128x2048x7168 1.003 (0.003); pt_decode_1024_256 0.837 (0.217) | 0 / 0 / 2 |
| streamk | h100-pcie | shipped ON | mm_128x2048x7168 0.653 (0.002); pt_decode_1024_256 1.000 (0.163) | 1 / 0 / 1 |
| warp-fill | h100-pcie | shipped ON | mm_128x2048x7168 0.575 (0.002); mm_128x2048x7168_c 0.529 (0.002); mm_128x2048x7168_rc 0.528 (0.002); pt_decode_1024_256 0.914 (0.163); pt_rm_m128_d1024_h2048 0.384 (0.355); pt_rm_m128_d1024_h2048_c 0.397 (0.005); pt_rm_m128_d256_h512 0.600 (0.276); pt_rm_m128_d256_h512_c 0.320 (0.009) | 6 / 0 / 2 |
| warp-fill | rtx-3090 | shipped ON | mm_128x2048x7168 1.179 (0.003); mm_128x2048x7168_c 0.523 (0.003); mm_128x2048x7168_rc 1.058 (0.003); pt_decode_1024_256 0.995 (0.217); pt_rm_m128_d1024_h2048 0.571 (0.076); pt_rm_m128_d1024_h2048_c 0.522 (0.007); pt_rm_m128_d256_h512 0.627 (0.142); pt_rm_m128_d256_h512_c 0.462 (0.032) | 5 / 2 / 1 |

#### GEMM cells and MLP kernels: Mapal vs cuBLAS / cuBLASLt / CUTLASS (median ms, both processes)

| cell | arch | Mapal default | realization | Mapal _c | Mapal _x | cuBLAS | cuBLASLt | CUTLASS | tuned CUTLASS | default / best library | null |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mm_128x2048x7168 | h100-pcie | 0.3090 | streamk | 0.4740 | 0.4732 | 0.1405 | - | 0.9780 | 0.4208 | 2.199 | 0.002 |
| pt_rm_m128_d1024_h2048 | h100-pcie | 0.1514 | - | = | 0.1779 | 0.0865 | 0.0798 | 0.4060 | 0.1577 | 1.897 | 0.005 |
| pt_rm_m128_d256_h512 | h100-pcie | 0.0329 | - | = | 0.0423 | 0.0365 | 0.0296 | 0.1138 | 0.0306 | 1.111 | 0.009 |
| mm_128x2048x7168 | rtx-3090 | 0.3388 | atomic S=2 | 0.3410 | 0.3748 | 0.1751 | - | 0.7332 | 0.5192 | 1.935 | 0.003 |
| pt_rm_m128_d1024_h2048 | rtx-3090 | 0.1608 | - | = | 0.1812 | 0.1024 | 0.0932 | 0.3533 | 0.1741 | 1.725 | 0.007 |
| pt_rm_m128_d256_h512 | rtx-3090 | 0.0317 | - | = | 0.0399 | 0.0297 | 0.0256 | 0.0993 | 0.0297 | 1.238 | 0.032 |

#### Workload suite: iter ms (median of the reps), CPU-only / GPU-only / CPU+GPU, libraries, PyTorch

| workload | arch | CPU-only | GPU-only | CPU+GPU (shipped) | null | kernel / copy / host ms, launches, MB (traced rep) | best library | torch eager | torch compile | shipped / best other |
|---|---|---|---|---|---|---|---|---|---|---|
| decode_1024_256 | h100-pcie | 3657.57 | = shipped | 275.69 | 0.013 | 85.55 / 18.20 / 178.11, 5120, 4 | cache 24.6740 | - | - | 11.173 |
| rm_m128_d1024_h2048 | h100-pcie | 10.09 | = shipped | 0.2202 | 0.043 | 0.17 / 0.05 / 0.00, 2, 0 | - | - | - | - |
| rm_m128_d256_h512 | h100-pcie | 0.7482 | = shipped | 0.1490 | 0.086 | 0.04 / 0.06 / 0.00, 2, 0 | - | - | - | - |
| decode_1024_256 | rtx-3090 | 3950.51 | = shipped | 186.75 | 0.000 | 47.08 / 17.87 / 88.38, 4096, 4 | cache 30.9750 | - | - | 6.029 |
| rm_m128_d1024_h2048 | rtx-3090 | 30.49 | = shipped | 0.2141 | 0.005 | 0.17 / 0.07 / 0.00, 2, 0 | - | - | - | - |
| rm_m128_d256_h512 | rtx-3090 | 2.1282 | = shipped | 0.0942 | 0.036 | 0.04 / 0.07 / 0.00, 2, 0 | - | - | - | - |

### facts-on-graph GPU sweep (default float face, 1 box per arch, 15 reps)

iter ms, median. Ratio = best other / Mapal (below 1: the other leg is faster). Best other = min over cuBLAS/cuBLASLt/CUTLASS/streams/fused/library-cache legs and torch eager/compile. Raw: [facts-on-graph/](facts-on-graph/) (results-*.txt.gz, table.md).

| workload | arch | Mapal ms | Mapal CPU-only ms | best other | ratio (other/Mapal) |
|---|---|---|---|---|---|
| pt_mlp_2048 | rtx-4090 | 0.7386 | 48.25 | 0.641 (lt) | 0.87 |
| pt_mlp_2048 | rtx-5090 | 0.6452 | 55.17 | 0.5571 (lt) | 0.86 |
| pt_mlp_2048 | h100-pcie | 1.291 | 30.19 | 1.024 (lt) | 0.79 |
| pt_mlp_2048 | rtx-3090 | 1.767 | 134.8 | 2.321 (lt) | 1.31 |
| pt_mlp2_2048 | rtx-4090 | 1.404 | 88.63 | 1.249 (lt2) | 0.89 |
| pt_mlp2_2048 | rtx-5090 | n/a | | | |
| pt_mlp2_2048 | h100-pcie | 2.589 | 52.63 | 1.808 (lt2) | 0.70 |
| pt_mlp2_2048 | rtx-3090 | 3.367 | 242.2 | 4.648 (lt2) | 1.38 |
| pt_rt_mlp2_2048 | rtx-4090 | 49.98 | 5965 | - | - |
| pt_rt_mlp2_2048 | rtx-5090 | 50.49 | 8333 | - | - |
| pt_rt_mlp2_2048 | h100-pcie | 53.47 | 6998 | - | - |
| pt_rt_mlp2_2048 | rtx-3090 | 75.3 | 1.483e+04 | - | - |
| pt_pj_m16_n2048_k7168 | rtx-4090 | 0.1382 | 20.66 | - | - |
| pt_pj_m16_n2048_k7168 | rtx-5090 | 0.1229 | 18.77 | - | - |
| pt_pj_m16_n2048_k7168 | h100-pcie | n/a | | | |
| pt_pj_m16_n2048_k7168 | rtx-3090 | n/a | | | |
| pt_attn_1024_64 | rtx-4090 | 0.24 | 6.589 | 0.0325 (torch-compile) | 0.14 |
| pt_attn_1024_64 | rtx-5090 | 0.2185 | 11.95 | 0.0266 (torch-compile) | 0.12 |
| pt_attn_1024_64 | h100-pcie | 0.1803 | 5.335 | 0.0317 (torch-compile) | 0.18 |
| pt_attn_1024_64 | rtx-3090 | 0.2328 | 15.56 | 0.0594 (fused) | 0.26 |
| pt_softmax_rows_4096 | rtx-4090 | 2.956 | 215.4 | 2.615 (fused) | 0.88 |
| pt_softmax_rows_4096 | rtx-5090 | 2.714 | 231.8 | 2.259 (fused) | 0.83 |
| pt_softmax_rows_4096 | h100-pcie | 4.191 | 139.7 | 3.896 (fused) | 0.93 |
| pt_softmax_rows_4096 | rtx-3090 | 6.434 | 761.4 | 8.748 (fused) | 1.36 |
| pt_decode_1024_256 | rtx-4090 | 42.2 | 2951 | 21.77 (lib-cache) | 0.52 |
| pt_decode_1024_256 | rtx-5090 | 56.42 | 3892 | 45.07 (lib-nocache) | 0.80 |
| pt_decode_1024_256 | h100-pcie | 72.87 | 3272 | 22.19 (lib-cache) | 0.30 |
| pt_decode_1024_256 | rtx-3090 | 54.34 | 5040 | 33.39 (lib-cache) | 0.61 |
| pt_causal_1024_256 | rtx-4090 | 580.5 | 3211 | 182.1 (torch-compile) | 0.31 |
| pt_causal_1024_256 | rtx-5090 | 543.9 | 4233 | 198.6 (torch-compile) | 0.37 |
| pt_causal_1024_256 | h100-pcie | 629.5 | 3407 | 130.1 (torch-compile) | 0.21 |
| pt_causal_1024_256 | rtx-3090 | 885.7 | 5467 | - | - |
| pt_prefill_1024_512_256 | rtx-4090 | 572.9 | 1659 | - | - |
| pt_prefill_1024_512_256 | rtx-5090 | 520.9 | 2171 | - | - |
| pt_prefill_1024_512_256 | h100-pcie | 464.3 | 1749 | - | - |
| pt_prefill_1024_512_256 | rtx-3090 | 806.4 | 2887 | - | - |
| pt_rowsum_4096 | rtx-4090 | 2.61 | 196.6 | 2.441 (gemv) | 0.94 |
| pt_rowsum_4096 | rtx-5090 | 2.507 | 193.9 | 2.178 (eager) | 0.87 |
| pt_rowsum_4096 | h100-pcie | 3.782 | 116.3 | 3.717 (gemv) | 0.98 |
| pt_rowsum_4096 | rtx-3090 | 5.891 | 729.6 | 8.395 (gemv) | 1.42 |
| pt_fir_4194304_16 | rtx-4090 | 0.04555 | 2.276 | - | - |
| pt_fir_4194304_16 | rtx-5090 | 0.04419 | 1.9 | - | - |
| pt_fir_4194304_16 | h100-pcie | 0.04502 | 0.7551 | - | - |
| pt_fir_4194304_16 | rtx-3090 | 0.08776 | 3.552 | - | - |
| pt_conv_2048_5 | rtx-4090 | 0.06537 | 3.012 | 0.0891 (torch-compile) | 1.36 |
| pt_conv_2048_5 | rtx-5090 | 0.05296 | 1.86 | 0.084 (torch-eager) | 1.59 |
| pt_conv_2048_5 | h100-pcie | 0.06594 | 0.6732 | 0.1209 (torch-eager) | 1.83 |
| pt_conv_2048_5 | rtx-3090 | 0.1221 | 3.368 | - | - |
| pt_convmc_8_64_56_3 | rtx-4090 | 240 | 249.9 | 0.0535 (torch-eager) | 0.00 |
| pt_convmc_8_64_56_3 | rtx-5090 | 277.6 | 264.7 | 0.0635 (torch-eager) | 0.00 |
| pt_convmc_8_64_56_3 | h100-pcie | 158.1 | 158.2 | 0.0803 (torch-eager) | 0.00 |
| pt_convmc_8_64_56_3 | rtx-3090 | 532.9 | 537.4 | - | - |
| pt_ew_4096 | rtx-4090 | 0.2669 | 21.06 | 0.2253 (fused) | 0.84 |
| pt_ew_4096 | rtx-5090 | 0.6437 | 9.054 | 0.1311 (fused) | 0.20 |
| pt_ew_4096 | h100-pcie | 0.273 | 2.188 | 0.1238 (fused) | 0.45 |
| pt_ew_4096 | rtx-3090 | 0.4631 | 38.71 | 0.2417 (fused) | 0.52 |
| pt_epi_4096 | rtx-4090 | 2.536 | 195.4 | 2.445 (lt) | 0.96 |
| pt_epi_4096 | rtx-5090 | 2.398 | 194.2 | 2.224 (eager) | 0.93 |
| pt_epi_4096 | h100-pcie | 3.617 | 107 | 3.682 (lt) | 1.02 |
| pt_epi_4096 | rtx-3090 | 5.466 | 722.8 | 8.667 (lt) | 1.59 |
| pt_fanout_2048 | rtx-4090 | 0.7025 | 51.99 | 0.6359 (streams) | 0.91 |
| pt_fanout_2048 | rtx-5090 | 0.6603 | 59.01 | 0.5574 (streams) | 0.84 |
| pt_fanout_2048 | h100-pcie | 1.302 | 29.68 | 1.028 (streams) | 0.79 |
| pt_fanout_2048 | rtx-3090 | 1.748 | 142.4 | 1.556 (streams) | 0.89 |
| pt_diamond_2048 | rtx-4090 | 1.247 | 69.38 | 0.9472 (lt) | 0.76 |
| pt_diamond_2048 | rtx-5090 | 0.9495 | 82.39 | 0.8418 (torch-compile) | 0.89 |
| pt_diamond_2048 | h100-pcie | 1.92 | 42.07 | 1.443 (streams) | 0.75 |
| pt_diamond_2048 | rtx-3090 | 2.588 | 204.9 | 3.603 (streams) | 1.39 |
| pt_twin_2048 | rtx-4090 | 0.8475 | 45 | - | - |
| pt_twin_2048 | rtx-5090 | n/a | | | |
| pt_twin_2048 | h100-pcie | 1.288 | 26.95 | - | - |
| pt_twin_2048 | rtx-3090 | 1.721 | 144.5 | - | - |
| pt_loop_4096 | rtx-4090 | 19.59 | 1.395e+04 | 19.37 (cutlass) | 0.99 |
| pt_loop_4096 | rtx-5090 | 17.93 | 1.813e+04 | 18.54 (lt) | 1.03 |
| pt_loop_4096 | h100-pcie | 28.77 | 1.349e+04 | 29.43 (lt) | 1.02 |
| pt_loop_4096 | rtx-3090 | 56.98 | 4.335e+04 | 63.07 (lt) | 1.11 |
| pt_loopmm_1024_8 | rtx-4090 | 0.6133 | 180.1 | - | - |
| pt_loopmm_1024_8 | rtx-5090 | 0.7452 | 231.5 | - | - |
| pt_loopmm_1024_8 | h100-pcie | 0.7465 | 217.9 | - | - |
| pt_loopmm_1024_8 | rtx-3090 | 1.611 | 313.1 | - | - |
| pt_smlp_4096_256_1024 | rtx-4090 | 0.2087 | 6.976 | 0.1331 (eager) | 0.64 |
| pt_smlp_4096_256_1024 | rtx-5090 | 0.1651 | 8.512 | 0.1024 (lt) | 0.62 |
| pt_smlp_4096_256_1024 | h100-pcie | 0.2311 | 3.795 | 0.1541 (lt) | 0.67 |
| pt_smlp_4096_256_1024 | rtx-3090 | 0.3796 | 16.38 | 0.2519 (cutlass) | 0.66 |


Value gates and open rows

| item | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 |
|---|---|---|---|---|
| `_exact` face VALUE-FAIL (arm on and null; row skipped) | 0 | pt_mlp2_2048, pt_twin_2048 | pt_pj_m16_n2048_k7168 | pt_pj_m16_n2048_k7168 |
| intent check mismatches (of checks) | 5/1178 | 6/1479 | 3/1275 | 19/1340 |
| torch leg | through pt_diamond_2048 | exit 1 | exit 1 | traceback |

Rows with no torch value have none. pt_convmc torch median (0.05-0.08 ms for ~18.5 GFLOP) is above the card's fp32 peak: unexplained, so the pt_convmc ratio is open.

K/V traffic and retention (traced rep 1)

| program | arch | launches | d2h ms | bytes in / out (total) |
|---|---|---|---|---|
| pt_decode_1024_256 | rtx-4090 / h100-pcie | 4096 / 5120 | 6.1 / 6.8 | - |
| pt_causal_1024_256 | rtx-4090 / h100-pcie | 2048 / 2048 | 12.5 / 178.0 | 2096128 / 4192256 (rtx-4090) |
| pt_prefill_1024_512_256 | rtx-4090 / h100-pcie | 1026 / 1026 | 6.9 / 92.9 | - |

The static 2,097,152 B per step is not realized: the run moves 4.19 MB out in total. Prefill + 512 decode steps = 1026 launches (2 + 2 per step): the decode loop does not redo the prefill's K/V. Causal d2h on h100-pcie is 14x the rtx-4090's: open.
