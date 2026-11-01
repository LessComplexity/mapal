# Why Mapal is slower: flow sweep

Raw data is in [`benches/results/evidence/flow/`](../../../benches/results/evidence/flow/) (sweep `0398b84`). The GPU split comes from
the nsys trace in `prof-<arch>.tgz` (one traced process per program). **kernel** sums the timed kernels (`mapal_gen_*`
excluded). **copy** counts only the copies before the last kernel. **rest** is the median iter minus kernel, copy and
memset: the time the GPU waits for host work, launch gaps or allocation. Because rest mixes the median with a single
trace, it can be larger than the gap. Usage is bytes or FLOPs divided by the nsys kernel time, since vast refuses
hardware counters. Peaks for 4090 / 5090 / H100 PCIe / 3090 are FP32 82.6 / 104.8 / 51.2 / 35.6 TFLOP/s and DRAM
1008 / 1792 / 2039 / 936 GB/s.

## Top 10 (each workload at its worst arch)

| # | workload | worst arch (gap ms) | other archs (gap ms) | where the time goes | causes | ms recoverable (basis) |
|---|---|---|---|---|---|---|
| 1 | prefill_1024_512_256 | 4090 (474.7) | 3090 452.6, 5090 426.6, H100 398.4 | GPU idle 414–572 ms. After each trip's K/V row D2H the host runs the attention (d2h→h2d idle: 537 ms on 4090, 402 on H100). Kernels take 4.5–11.8 ms | C1 (+C2 on H100) | ≤ 474.7 (idle > gap). On H100, C2 measured 96.0 |
| 2 | causal_1024_256 | H100 (467.8) | 3090 372.0, 4090 367.1, 5090 344.4 | Same as prefill. H100 also moves 2148 MB D2H and 2148 MB memset (53.1 + 2.9 ms) and rebuilds K/V in full every trip (2048 × 11.5 µs) | C1, C2 | ≤ 467.8. C2 measured 183.2 (ON/OFF 0.678) |
| 3 | convmc_8_64_56_3 | 3090 (396.7) | 4090 270.7, 5090 256.5, H100 178.8 | No device kernel. The conv runs as host code: shipped 270.8 vs CPU-only 228.1 on 4090 | C3 | ≤ 396.7 (the torch leg is TF32, O6) |
| 4 | rt_mlp2_2048 | 3090 (56.7) | 5090 48.2, 4090 35.8, H100 22.8 | 285 MB of pageable H2D per run (copy 11.7–44.3 ms) plus 9–18 ms rest. Kernels take 9.4–26.4 ms | C4 (+C6a/b) | 56.7 (copy + rest = 62.3). 161.6 across the 4 archs |
| 5 | decode_1024_256 | H100 (51.2) | 4090 27.4, 3090 25.1, 5090 13.9 | About 10 ops per trip, each issued after the host acts: GPU idle 25–37 ms. H100 kernels take 37.2 ms, 23.6 of it the full K/V rebuild, plus 4.5 ms memset | C5, C4 (+C2 on H100) | ≤ 25.3 idle on H100 (the K/V share is O2); 27.4 / 25.1 / 13.9 elsewhere |
| 6 | mlp2_2048 | H100 (0.714) | 5090 0.172 (4090/3090 win) | Kernels are 2.498 of 2.583: 4 GEMMs at 619–633 µs vs cuBLASLt 508 µs | C6a | 0.466 (kernel excess) |
| 7 | loopmm_1024_8 | 3090 (0.692) | 5090 0.378, 4090 0.301, H100 0.165 | 3090: 8 × 173.7 µs vs torch-compile's 0.968 ms total. 4090/5090: rest is 0.27–0.30 ms (352 µs idle before the 1st GEMM, 0.6 µs between the next 7) | C6c, C7, C8 | 3090: 0.42 kernel + 0.22 rest |
| 8 | ew_4096 | 5090 (0.567) | H100 0.430, 4090 0.359, 3090 0.305 | The kernel takes 33–108 µs, below torch's own kernel. Rest is 0.44–0.67 ms: the arena cudaMalloc lands inside the clock (alloc_ms 0.35–0.56) | C8 | 1.66 across the 4 archs (alloc_ms, capped at the gap) |
| 9 | diamond_2048 | H100 (0.463) | 5090 0.127, 4090 0.069 (3090 wins) | 3 GEMMs at 623–639 µs vs 508 µs | C6a | 0.355 |
| 10 | softmax_rows_4096 | 5090 (0.430) | H100 0.294, 4090 0.095 (3090 wins) | 5090: GEMM 2165 vs CUTLASS 1952 µs, plus 0.39 ms rest. H100: 4 row folds at one thread per row, 80–213 µs each | C6b, C11 / C9, C12 | flow ON measured: 0.230 (5090), 0.508 (H100), 0.334 (4090) |

## Every losing workload × arch (sorted by gap)

| workload | arch | Mapal ms | best ms (leg) | gap ms | GPU kernel ms | copy ms | memset ms | rest ms (host / launch) | top Mapal kernels (n x median us) | causes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| prefill_1024_512_256 | rtx-4090 | 578.6 | 103.9 (torch-compile) | 474.7 | 4.61 | 1.56 | 0 | 572.4 | fee5d054: 1022x4.5 | C1 |
| causal_1024_256 | h100-pcie | 568.9 | 101.1 (torch-compile) | 467.8 | 23.6 | 53.1 | 2.89 | 489.3 | 8065b046: 2048x11.5 | C1, C2 |
| prefill_1024_512_256 | rtx-3090 | 553 | 100.4 (torch-compile) | 452.6 | 6.759 | 1.89 | 0.00755 | 544.3 | 0fe28881: 1022x6.4, ab736138: 4x51.2 | C1 |
| prefill_1024_512_256 | rtx-5090 | 529.5 | 102.9 (torch-compile) | 426.6 | 4.493 | 1.03 | 0 | 524 | e2f8ae46: 1022x4.3 | C1 |
| prefill_1024_512_256 | h100-pcie | 450.6 | 52.25 (torch-compile) | 398.4 | 11.78 | 23.1 | 1.45 | 414.3 | 8065b046: 1026x11.5 | C1, C2 |
| convmc_8_64_56_3 | rtx-3090 | 396.8 | 0.079 (torch) | 396.7 | - | - | - | 396.8 | no device kernel (CPU code) | C3 |
| causal_1024_256 | rtx-3090 | 568.9 | 196.9 (torch-compile) | 372 | 13.22 | 3.24 | 0.00381 | 552.4 | 0fe28881: 2046x6.4 | C1 |
| causal_1024_256 | rtx-4090 | 563.3 | 196.2 (torch-compile) | 367.1 | 9.183 | 2.63 | 0 | 551.5 | fee5d054: 2046x4.5 | C1 |
| causal_1024_256 | rtx-5090 | 542.7 | 198.3 (torch-compile) | 344.4 | 8.759 | 1.17 | 0 | 532.8 | e2f8ae46: 2046x4.3 | C1 |
| convmc_8_64_56_3 | rtx-4090 | 270.8 | 0.0523 (torch) | 270.7 | - | - | - | 270.8 | no device kernel (CPU code) | C3 |
| convmc_8_64_56_3 | rtx-5090 | 256.6 | 0.0635 (torch) | 256.5 | - | - | - | 256.6 | no device kernel (CPU code) | C3 |
| convmc_8_64_56_3 | h100-pcie | 178.9 | 0.0808 (torch) | 178.8 | - | - | - | 178.9 | no device kernel (CPU code) | C3 |
| rt_mlp2_2048 | rtx-3090 | 88.68 | 31.95 (torch-compile) | 56.73 | 26.37 | 44.3 | 0 | 18.04 | dc2dca64: 8x865.9, 1a22fa58: 8x826.2 | C4 |
| decode_1024_256 | h100-pcie | 71 | 19.85 (cache) | 51.15 | 37.2 | 4.02 | 4.52 | 25.25 | 8065b046: 2048x11.5, a21805fe: 1024x7.8 | C5, C4, C2 |
| rt_mlp2_2048 | rtx-5090 | 58.69 | 10.48 (torch-compile) | 48.21 | 9.417 | 38 | 0.0476 | 11.27 | 997700f4: 8x327.0, d2d7996f: 8x301.4 | C4, C6b |
| rt_mlp2_2048 | rtx-4090 | 47.42 | 11.62 (torch-compile) | 35.8 | 9.709 | 21.1 | 0 | 16.6 | cb8a431d: 8x315.0, c56cd5d3: 8x313.0 | C4 |
| decode_1024_256 | rtx-4090 | 56.05 | 28.65 (cache) | 27.4 | 18.4 | 1.78 | 1.36 | 34.51 | fee5d054: 2046x4.5, 55fab8ff: 1024x5.4 | C5, C4 |
| decode_1024_256 | rtx-3090 | 53.39 | 28.32 (cache) | 25.07 | 21.94 | 2.37 | 1.92 | 27.16 | 0fe28881: 2046x6.4, 16bba541: 1024x4.4 | C5, C4 |
| rt_mlp2_2048 | h100-pcie | 40.83 | 18 (torch-compile) | 22.83 | 20.02 | 11.7 | 0 | 9.149 | 5c213072: 8x639.8, 33e05412: 8x626.7 | C4, C6a |
| decode_1024_256 | rtx-5090 | 55.27 | 41.39 (nocache) | 13.88 | 16.59 | 1.07 | 0.682 | 36.93 | e2f8ae46: 2046x4.2, a0714d3d: 1024x4.9 | C5, C4 |
| mlp2_2048 | h100-pcie | 2.583 | 1.869 (lt2) | 0.714 | 2.498 | 0 | 0 | 0.08547 | 844dbc85: 2x619.1, 33e05412: 1x633.1 | C6a, C7 |
| loopmm_1024_8 | rtx-3090 | 1.66 | 0.9676 (torch-compile) | 0.6924 | 1.389 | 0 | 0.0468 | 0.2242 | ce5a973d: 8x173.7 | C6c, C7, C8 |
| ew_4096 | rtx-5090 | 0.6984 | 0.1311 (fused) | 0.5673 | 0.03325 | 0 | 0 | 0.6652 | 7088d91e: 1x33.2 | C8 |
| diamond_2048 | h100-pcie | 1.924 | 1.461 (streams) | 0.463 | 1.879 | 0 | 0 | 0.0451 | 844dbc85: 2x623.5, 02e938c5: 1x639.2 | C6a, C7 |
| ew_4096 | h100-pcie | 0.5448 | 0.1147 (torch-compile) | 0.4301 | 0.07283 | 0 | 0 | 0.472 | 07af1aed: 1x72.8 | C8 |
| softmax_rows_4096 | rtx-5090 | 2.88 | 2.45 (torch-compile) | 0.43 | 2.457 | 0 | 0.0311 | 0.3916 | 6eab42db: 1x2165.3, a1d935db: 1x103.4 | C6b, C11, C12 |
| loopmm_1024_8 | rtx-5090 | 0.7443 | 0.3666 (torch-compile) | 0.3777 | 0.4396 | 0 | 0 | 0.3047 | b3430eaa: 8x54.5 | C6c, C7, C8 |
| ew_4096 | rtx-4090 | 0.585 | 0.2261 (torch-compile) | 0.3589 | 0.05165 | 0 | 0 | 0.5334 | 2728d927: 1x51.6 | C8 |
| ew_4096 | rtx-3090 | 0.5454 | 0.2406 (torch-compile) | 0.3048 | 0.1076 | 0 | 0 | 0.4378 | f0a3b0ee: 1x107.6 | C8 |
| loopmm_1024_8 | rtx-4090 | 0.7311 | 0.4301 (torch) | 0.301 | 0.4608 | 0 | 0 | 0.2703 | 565ffde5: 8x57.4 | C6c, C7, C8 |
| softmax_rows_4096 | h100-pcie | 4.202 | 3.908 (torch-compile) | 0.294 | 4.119 | 0 | 0 | 0.08285 | eb50b87a: 1x3547.1, 0ebf3e09: 1x213.2 | C9, C12 |
| rowsum_4096 | rtx-5090 | 2.687 | 2.409 (torch-compile) | 0.278 | 2.242 | 0 | 0.0286 | 0.4166 | 6eab42db: 1x2154.0, 900a9eaa: 1x87.8 | C6b, C11 |
| fanout_2048 | h100-pcie | 1.302 | 1.034 (streams) | 0.268 | 1.247 | 0 | 0 | 0.05456 | 67c37c5c: 1x630.3, 844dbc85: 1x617.1 | C6a, C7 |
| pj_m16_n2048_k7168 | rtx-3090 | 0.3488 | 0.0819 (torch-compile) | 0.2669 | 0.1179 | 0 | 0 | 0.2309 | aabc3f35: 1x115.7, 164c6cf4: 1x2.2 | C10, C7 |
| mlp_2048 | h100-pcie | 1.291 | 1.03 (lt) | 0.261 | 1.237 | 0 | 0 | 0.05387 | b393c3ba: 1x621.9, 844dbc85: 1x615.3 | C6a, C7 |
| twin_2048 | h100-pcie | 1.287 | 1.03 (torch-compile) | 0.257 | 1.235 | 0 | 0 | 0.05243 | 844dbc85: 2x620.9 | C6a, C7 |
| attn_1024_64 | rtx-4090 | 0.2873 | 0.0328 (torch-compile) | 0.2545 | 0.08448 | 0 | 0.00259 | 0.2002 | 7e5df615: 1x23.7, cc5bd59a: 1x19.2 | C9, C7, C12 |
| attn_1024_64 | rtx-3090 | 0.2423 | 0.0584 (fused) | 0.1839 | 0.1486 | 0 | 0.00163 | 0.09203 | 3f060d72: 1x54.6, cc03fc0c: 1x29.7 | C9, C7, C12 |
| attn_1024_64 | rtx-5090 | 0.2022 | 0.0266 (torch-compile) | 0.1756 | 0.06378 | 0 | 0 | 0.1384 | 2c8b55a0: 1x18.0, a0a6a43f: 1x15.8 | C9, C7, C12 |
| epi_4096 | rtx-5090 | 2.585 | 2.412 (torch-compile) | 0.173 | 2.159 | 0 | 0 | 0.4262 | f342093f: 1x2158.8 | C6b, C11 |
| mlp2_2048 | rtx-5090 | 1.404 | 1.232 (lt2) | 0.172 | 1.136 | 0 | 0.0115 | 0.2564 | c59d5a05: 2x262.5, 997700f4: 1x315.2 | C6b, C7 |
| pj_m16_n2048_k7168 | h100-pcie | 0.2378 | 0.0693 (torch-compile) | 0.1685 | 0.08195 | 0 | 0 | 0.1558 | 62db87fb: 1x80.2, 7a30627f: 1x1.8 | C10, C7 |
| smlp_4096_256_1024 | rtx-3090 | 0.4092 | 0.2417 (torch-compile) | 0.1675 | 0.2901 | 0 | 0.00595 | 0.1131 | c849b908: 1x171.9, bf5981fa: 1x118.2 | C6c, C7 |
| loopmm_1024_8 | h100-pcie | 0.8032 | 0.638 (torch-compile) | 0.1652 | 0.6235 | 0 | 0.0191 | 0.1606 | 3591423d: 8x77.2 | C6c, C7, C8 |
| smlp_4096_256_1024 | rtx-4090 | 0.2658 | 0.1034 (torch-compile) | 0.1624 | 0.1017 | 0 | 0 | 0.1641 | a1d600c3: 1x57.8, 52fc2d71: 1x44.0 | C7 |
| attn_1024_64 | h100-pcie | 0.1844 | 0.0328 (torch-compile) | 0.1516 | 0.1024 | 0 | 0.00125 | 0.08075 | 2c9ea216: 1x30.1, 506b9cd2: 1x27.7 | C9, C7, C12 |
| diamond_2048 | rtx-5090 | 1.001 | 0.8745 (torch-compile) | 0.1265 | 0.8395 | 0 | 0.0121 | 0.1494 | c59d5a05: 2x284.0, cbb11ec5: 1x286.9 | C6b, C7 |
| rowsum_4096 | rtx-4090 | 2.515 | 2.402 (torch-compile) | 0.113 | 2.456 | 0 | 0 | 0.0589 | 86cfda25: 1x2388.4, c63583fc: 1x67.7 | C6, C7 |
| mlp_2048 | rtx-5090 | 0.6868 | 0.5796 (lt) | 0.1072 | 0.5683 | 0 | 0.00589 | 0.1127 | 78b03e0e: 1x302.9, c59d5a05: 1x265.3 | C6b, C7 |
| fanout_2048 | rtx-5090 | 0.6859 | 0.5837 (torch-compile) | 0.1022 | 0.5764 | 0 | 0.00592 | 0.1036 | e637050b: 1x311.5, c59d5a05: 1x264.9 | C6b, C7 |
| softmax_rows_4096 | rtx-4090 | 2.88 | 2.785 (torch-compile) | 0.095 | 2.812 | 0 | 0 | 0.06835 | ef9b0c1d: 1x2356.0, eb3955d2: 1x131.5 | C12 |
| twin_2048 | rtx-5090 | 0.6703 | 0.5817 (torch-compile) | 0.0886 | 0.5469 | 0 | 0.0124 | 0.111 | c59d5a05: 2x279.7 | C6b, C7 |
| smlp_4096_256_1024 | rtx-5090 | 0.1731 | 0.0942 (torch-compile) | 0.0789 | 0.09507 | 0 | 0 | 0.07803 | d3ae82cb: 1x58.1, dfe0c12e: 1x37.0 | C7 |
| smlp_4096_256_1024 | h100-pcie | 0.2329 | 0.1565 (lt) | 0.0764 | 0.1687 | 0 | 0.00243 | 0.06179 | ec31e1a8: 1x89.1, b05526af: 1x79.6 | C6a, C7 |
| mlp_2048 | rtx-4090 | 0.69 | 0.6154 (cutlass) | 0.0746 | 0.5935 | 0 | 0 | 0.09649 | 17e3127b: 1x305.7, 59027ed4: 1x287.8 | C7 |
| twin_2048 | rtx-4090 | 0.6794 | 0.6083 (torch-compile) | 0.0711 | 0.57 | 0 | 0 | 0.1094 | 59027ed4: 2x289.1 | C7 |
| pj_m16_n2048_k7168 | rtx-5090 | 0.1255 | 0.0553 (torch) | 0.0702 | 0.06723 | 0 | 0 | 0.05827 | 37c50431: 1x67.2 | C10, C7 |
| diamond_2048 | rtx-4090 | 0.9847 | 0.9155 (torch-compile) | 0.0692 | 0.9128 | 0 | 0 | 0.07189 | 59027ed4: 2x319.1, 60426c42: 1x312.9 | C7 |
| fanout_2048 | rtx-4090 | 0.6896 | 0.6308 (streams) | 0.0588 | 0.5863 | 0 | 0 | 0.1033 | 4ea5bf1c: 1x293.5, 59027ed4: 1x292.8 | C7 |
| twin_2048 | rtx-3090 | 1.73 | 1.672 (torch-compile) | 0.058 | 1.64 | 0 | 0 | 0.09004 | 1a22fa58: 2x821.2 | C7 |
| pj_m16_n2048_k7168 | rtx-4090 | 0.1487 | 0.0932 (torch-compile) | 0.0555 | 0.06822 | 0 | 0 | 0.08048 | 6e4b9bcd: 1x68.2 | C10, C7 |
| firew_rowsum_2048x2048_16 | rtx-5090 | 0.0895 | 0.0573 (torch-compile) | 0.0322 | 0.0504 | 0 | 0 | 0.0391 | b0769a61: 1x30.6, c0dee364: 1x19.8 | C12, C7 |
| firew_rowsum_2048x2048_16 | h100-pcie | 0.1121 | 0.0838 (torch-compile) | 0.0283 | 0.09245 | 0 | 0 | 0.01965 | 0f65c9fe: 1x61.3, 096f5e1d: 1x31.1 | C12, C7 |
| firew_rowsum_2048x2048_16 | rtx-4090 | 0.0884 | 0.0675 (torch-compile) | 0.0209 | 0.06278 | 0 | 0 | 0.02562 | 65631a69: 1x33.5, c4cf34b6: 1x29.3 | C12, C7 |
| firew_rowsum_2048x2048_16 | rtx-3090 | 0.1527 | 0.1423 (torch-compile) | 0.0104 | 0.127 | 0 | 0 | 0.02572 | c923147d: 1x68.2, 9b4ca251: 1x58.8 | C12, C7 |

## Causes: SASS and usage vs the best leg, and the missing fact or realization

| id | rows | evidence | missing fact / realization |
|---|---|---|---|
| C1 | causal, prefill ×4 | Each trip runs: H2D x row (1 KB), K-row kernel, D2H, V-row kernel, D2H, then 400–530 µs of host work. `facts.json` lists `host.device_sites = 2` (the K/V row maps only). Torch-compile runs the whole step on the device | `s = (r <= t) ? q·K[r] : -1e9` is a Phi over a loop-carried scalar, so it is not a `tile_plan_strided` site. Decode's unguarded `q·Kᵀ` is one and runs on the device. `mx`, `p`, `den` and `o` then read a host array (the byte rule), and `fold_candidates` returns nothing in a function that has a retained site (`func/resident.rs:1091`) |
| C2 | causal, prefill, decode on H100 | The move intent has `retained = 0`. Each trip does 2 × 1 MB D2H plus 1 MB memset and rebuilds K/V in full (stream-K `8065b046`, 228 CTAs, 11.5 µs). The other archs retain (`retained = 2`) and run a 1-CTA row kernel (4.3–6.4 µs) | The `GPU_RUNGS_OFF` row was judged on decode. This sweep's ON arms give causal 0.678 and prefill 0.787 (null 0.027 / 0.025) |
| C3 | convmc ×4 | No kernel and no trace. Torch's conv takes 0.05–0.08 ms (cuDNN, TF32) | The tap read `img[((b·C+ci)·w + i + q/3)·w + j + q%3]`, with `ci = k/9`, `q = k%9` and the cell `t` split by `/` and `%`, is affine only after a constant div/mod (mixed-radix) split. No fact (`tile_plan`, window or `cell_plan`) carries that split, so the fold is not a device site |
| C4 | rt_mlp2 ×4; `q` in decode, causal and prefill | rt_mlp2 does 17 H2D, 285 MB per run. `a` and `f` are closed forms of `(t, it)`, and torch builds them on the device | `gen_law` (`func/resident.rs:612–633`) accepts only `Map { captures: 0 }`. A law that captures the loop scalar (`it`, or `t` for `q`) is built on the host and uploaded on every trip |
| C5 | decode ×4 (and the x row in C1) | One 4090 trip takes 79 µs: H2D x, K, V, H2D q, memset, s, memset, o, D2H o, then 13.5 µs of host work. Every op waits 3–14 µs. The library cache leg runs 12 async launches per trip at 28 µs per trip, and loopmm's loop launches are 0.6 µs apart | `range-map` writes row t on the host. The row reads `o`, which lives on the device, so each trip moves `o` D2H and the row H2D (1 KB each) and the host stays in lock-step with the GPU. There is no device cell for an in-place range map (`map_law` piecewise + `Outside`) |
| C6a | H100: mlp, mlp2, diamond, fanout, twin, rt_mlp2, smlp | Mapal's 2048 tile: 8×8 = 64 accumulators/thread, 256 threads, bk 16, 1 stage, 0 LDGSTS, 122–127 regs, 17.4 KB smem, 16×16 CTAs: 615–640 µs, 52–55 % of FP32 peak. cuBLASLt `sm80_xmma 128x128x8 stage3`: 128 threads, 255 regs, 34.8 KB, split-K 4: 508 µs, 66 % | 4090 and 3090 select the 128-accumulator tile (tm 8 × tn 16, 128 threads, 218–224 regs) and tie CUTLASS (288–319 vs 306 µs on 4090). `wide-fma` does not apply on sm_90a, so the H100 profile picks the 64-accumulator, 1-stage tile |
| C6b | 5090: mlp, mlp2, diamond, fanout, twin, rt_mlp2, epi, rowsum, softmax | The plain 2048 stream-K GEMM (338 CTAs) runs 261–284 µs vs CUTLASS 256×128 split-K 5 at 255 µs. An epilogue GEMM (bias/relu, 16×16 grid, no stream-K) runs 287–327 µs. 4096 stream-K runs 2154–2165 vs 1952–1960 µs (61 vs 67 % FP32) | Stream-K is not combined with the epilogue site. Mapal runs a 64-accumulator tile where CUTLASS runs 256×128 (210 regs) |
| C6c | loopmm ×4, smlp on 3090 | 4090: tm 8 × tn 4 (32 accumulators, 95 regs, 1 CTA/SM, 78.8 KB) at 57.4 µs vs CUTLASS 128×64 (126 regs, 5 stages) at 42.7. 5090: tm 4 × tn 4 (16 accumulators, 1 stage) at 54.5 µs vs 45.8 per GEMM for torch-compile. 3090: tm 8 × tn 8, split-K 2 atomic + memset at 173.7 µs vs about 121 (torch-compile total / 8) | Tile choice at K = 1024 |
| C7 | every straight-line program | The GPU sits idle 13–352 µs before each kernel's first launch (attn on 4090: 70, 16, 16, 27, 14, 125 µs); relaunching the same kernel costs 0.6 µs (loopmm). `kernel()` calls `cuModuleGetFunction` at launch time (`crates/mapal-rt/gpu/mapal_gpu.c:226`), and CUDA lazy loading loads the function at that moment. The torch legs are warmed | s62 gap map G5 measured 0.08–0.47 ms per MLP program from LAZY to EAGER. M5 (preload the kernels) is not done |
| C8 | ew ×4, loopmm | ew does 1 arena alloc + free; alloc_ms 0.35–0.56 matches rest 0.44–0.67. No device op precedes `t0` (the gens dissolved into the cell), so the arena's cudaMalloc lands after `t0`. loopmm does 3 loop-cone cudaMallocs | The arena is allocated at the first device op, not at fn entry. `arena` lays out straight-line objects only, so loop-cone buffers stay outside it |
| C9 | attn ×4, softmax on H100 | Rowfold kernels use 32–64 CTAs × 32–64 threads, one thread walking k = 1024–4096 in order (64–88 LDG, no LDS). attn: 15–30 µs each, about 0.22 TB/s; softmax on H100: 199–213 µs. Torch-compile's fused row softmax: 4.3 µs on 1024 CTAs × 256 | The default face leaves order unspecified (ADR-0040), so a row fold may be a tree. `rowfold` keeps the in-order chain |
| C10 | pj ×4 | W is 58.7 MB. 4090: 68.2 µs, 861 GB/s (85 %). 5090: 67.2 µs, 874 GB/s (49 %), 1-warp CTAs, 208 B spill. H100: 80.2 + 1.8 µs, 716 GB/s (35 %), S = 7. 3090: 115.7 + 2.2 µs, 498 GB/s (53 %), S = 4. The torch totals are 0.055–0.093 ms; on 4090 the gap is C7 alone | The GEMV tile puts too few warps on each SM (5090, H100, 3090). The 5090 spill is already a report finding (face-registers, min-blocks, panel-regs) |
| C11 | 5090: rowsum, softmax, epi | Rest is 0.39–0.43 ms vs 0.06–0.08 on the other archs. Inside the GPU span the gaps are small: a 31 µs memset of 67 MB and gaps of 51 µs or less | open (O5) |
| C12 | attn, firew, softmax_rows | `flow` ships OFF. ON/OFF (4090 / 5090 / H100 / 3090): attn 0.79 / 0.69 / 0.81 / 0.71, firew 0.71 / 0.60 / 0.52 / 0.77, softmax 0.88 / 0.92 / 0.88 / 0.93 | The `GPU_RUNGS_OFF` rows cite the old geometry. rowsum_4096 on 4090 is 1.056 (slower, null 0.033) |

## Fixes ranked by measured ms recoverable

| rank | fix | causes | rows | ms recoverable (sum over archs) | basis |
|---|---|---|---|---:|---|
| 1 | Run guarded strided score sites on the device: put the `guard_plan` predicate in the `tile_plan_strided` cell (masked value = the Phi's constant arm), and drop the retained-site veto in `fold_candidates` | C1 | causal, prefill ×4 | ≤ 3303.6 | Σ min(gap, rest); rest = measured GPU idle, 414–572 ms per row |
| 2 | Add a mixed-radix index fact to `mapal-ir` (constant div/mod splits of `k` and `t`) so the conv fold becomes a `tile_plan` site (implicit GEMM) | C3 | convmc ×4 | ≤ 1102.7 | The shipped build runs host code; bound = gap |
| 3 | Flip H100 `retain-resident` ON for causal and prefill (check decode first, O2) | C2 | causal, prefill on H100 | 279.2 | This sweep's ON arms: 568.9 × 0.322 + 450.6 × 0.213 |
| 4 | Let devgen take scalar captures (loop scalars become kernel args) | C4 | rt_mlp2 ×4 (+ `q` in C1/C5) | 161.6 | Σ min(gap, copy + rest) |
| 5 | Add a device cell for the in-place range map (row t from `o`). With fix 4, no per-trip copy is left and the loop enqueues ahead | C5 | decode ×4 | ≤ 91.6 | Σ min(gap, rest) |
| 6 | Per-arch tiles: 128 accumulators + cp.async stages on H100 (as on 4090/3090), stream-K with the epilogue on 5090, 64-accumulator tiles at K = 1024 | C6a–c | 23 rows | 7.7 | Σ (Mapal kernel − best-leg kernel), nsys, same box: H100 5.25, 5090 1.85, loopmm 0.61 |
| 7 | Preload every function of the module before the clock (s62 M5) | C7 | straight-line rows | ≤ 2.8 | Σ min(gap, rest) over straight-line C7 rows; overlaps C12/C9 on attn and firew; s62 measured LAZY→EAGER at 0.08–0.47 ms per program |
| 8 | Flip `flow` ON for attn, firew and softmax_rows (rowsum on 4090 stays OFF) | C12 | 12 rows | 1.78 | This sweep's ON/OFF × shipped. Flipped, firew wins on all 4 archs and softmax wins on 4090/H100 |
| 9 | Allocate the arena at fn entry (its size is static in the plan) and put loop-cone buffers in it | C8 | ew ×4, loopmm | 1.66 (+ loopmm, O8) | ew alloc_ms, capped at the gap |
| 10 | Tree row folds on the default face; a GEMV tile with ≥ 4 warps/SM and no spill | C9, C10 | attn, softmax on H100, pj | ≤ 0.5 | rowfold kernels − torch's fused kernel; pj kernel − torch total |

## Derived usage (bytes or FLOPs / nsys kernel time)

| kernel | arch | Mapal µs | Mapal usage | best-leg kernel µs | best-leg usage |
|---|---|---:|---|---:|---|
| GEMM 2048³ (17.2 GFLOP) | 4090 | 288–319 | 60–54 TFLOP/s (72–65 %) | CUTLASS 306 | 56 (68 %) |
| | 5090 | 261–327 | 66–53 (63–50 %) | CUTLASS 255 | 67 (64 %) |
| | H100 | 615–640 | 28–27 (55–52 %) | cuBLASLt 508 | 34 (66 %) |
| | 3090 | 802–866 | 21–20 (60–56 %) | CUTLASS 756 | 23 (64 %) |
| GEMM 4096³ (137 GFLOP) | 5090 | 2154–2165 | 64 (61 %) | CUTLASS 1952–1960 | 70 (67 %) |
| | H100 | 3544 | 39 (76 %) | cuBLASLt 3650 | 38 (73 %) |
| GEMM 1024³ (2.15 GFLOP) | 4090 | 57.4 | 37 (45 %) | CUTLASS 42.7 | 50 (61 %) |
| | 5090 | 54.5 | 39 (38 %) | torch-compile 45.8 (total / 8) | 47 (45 %) |
| | 3090 | 173.7 | 12 (35 %) | torch-compile 121 (total / 8) | 18 (50 %) |
| pj W read (58.7 MB) | 4090 / 5090 / H100 / 3090 | 68.2 / 67.2 / 82.0 / 117.9 | 861 / 874 / 716 / 498 GB/s (85 / 49 / 35 / 53 %) | torch total 93 / 55 / 69 / 82 | ≥ 630 / 1062 / 847 / 717 GB/s |
| ew output write (67.1 MB) | 4090 / 5090 / H100 / 3090 | 51.6 / 33.2 / 72.8 / 107.6 | 1.30 / 2.02 / 0.92 / 0.62 TB/s (4090 and 5090 above DRAM peak: the write stays in L2) | torch-compile 217 (4090) | 0.93 TB/s (3 arrays) |
| attn row fold (4.2 MB) | 4090 | 19.2 | 218 GB/s | torch fused softmax 4.3 | — |

## Open (the evidence that closes each)

| id | what | evidence |
|---|---|---|
| O1 | **Closed (cpu-gpu-boundary, placement dump on rtx-4090).** Decode: `x` (range map) and `q` were no `cell_plan` site — a scalar capture (`t`) made a map no `CellSite`, and `gen_law` took `captures: 0` only. Causal: `s` and `o` were no site of any kind (a body folding over captures is neither a tile site — the guard Phi — nor a cell: `local_nest`), `p` had scalar captures (`mx`, `t`), `mx`/`den` are whole folds (no `row_fold` site; the retained-site veto was moot for them, it blocked prefill's `pmx`/`pden`). Fixed by `uniform` (q, rt_mlp2 a/f) and `nest-cell` (s, p, o, convmc's fold); `x` stays the host range write (1c) | `TMP_DUMP` placement dump of `cell_plan`/`cell_of`/`cell_sites` per map |
| O2 | H100 `retain-resident` on decode, the program the OFF row was judged on | A `.on-retain-resident` arm of pt_decode_1024_256 on h100-pcie, ABBA |
| O3 | Library kernel SASS: C6a/C6b cite nsys geometry only (regs, smem, CTA, grid) | cuobjdump of the captured cuBLASLt/CUTLASS images (`sass_index.tsv` is absent from the H100 archive) |
| O4 | C7 attribution in this build | A `CUDA_MODULE_LOADING=EAGER` arm of the straight-line programs (s62 measured MLP programs only) |
| O5 | C11: 0.39–0.43 ms rest on 5090's 4096 programs | An nsys `-t cuda` API trace on the timing box. The 5090 prof archive comes from the retake box (Xeon 8352V, 57591208), the timings from 57585317 |
| O6 | A same-face convmc ratio | A torch leg with `torch.backends.cudnn.allow_tf32 = False` |
| O7 | DRAM GB/s and FP32 % are derived, not counted | ncu counters, which vast refuses |
| O8 | `trace-<arch>.tgz` is 0 B on all 4 archs; the lib leg has no trace for causal or convmc; loopmm's in-clock share of its 3 allocs is unknown | Re-pack the trace archives; add a lib trace for causal; get loopmm's API trace |
