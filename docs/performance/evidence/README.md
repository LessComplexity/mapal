# Evidence pass (S64)

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

## End check: the warp-fill class split (ON = with the split, OFF = the pre-split build; taken on the H100, not on the 3090: `shipped` below reads the arm name)

### How each optimization turned out in time (arms timed this session: ON / OFF, median over processes)

| rung | arch | shipped | cells: ON/OFF (null) | ON faster / slower / inside null |
|---|---|---|---|---|
| splitk | h100-pcie | shipped ON | pt_decode_1024_256 0.739 (0.163) | 1 / 0 / 0 |
| splitk | rtx-3090 | shipped ON | mm_128x2048x7168 0.994 (0.003); pt_decode_1024_256 0.797 (0.217) | 1 / 0 / 1 |
| splitk-atomic | h100-pcie | shipped ON | pt_decode_1024_256 1.004 (0.163) | 0 / 0 / 1 |
| splitk-atomic | rtx-3090 | shipped ON | mm_128x2048x7168 1.003 (0.003); pt_decode_1024_256 0.837 (0.217) | 0 / 0 / 2 |
| streamk | h100-pcie | shipped ON | mm_128x2048x7168 0.653 (0.002); pt_decode_1024_256 1.000 (0.163) | 1 / 0 / 1 |
| warp-fill | h100-pcie | shipped ON | mm_128x2048x7168 0.575 (0.002); mm_128x2048x7168_c 0.529 (0.002); mm_128x2048x7168_rc 0.528 (0.002); pt_decode_1024_256 0.914 (0.163); pt_rm_m128_d1024_h2048 0.384 (0.355); pt_rm_m128_d1024_h2048_c 0.397 (0.005); pt_rm_m128_d256_h512 0.600 (0.276); pt_rm_m128_d256_h512_c 0.320 (0.009) | 6 / 0 / 2 |
| warp-fill | rtx-3090 | shipped ON | mm_128x2048x7168 1.179 (0.003); mm_128x2048x7168_c 0.523 (0.003); mm_128x2048x7168_rc 1.058 (0.003); pt_decode_1024_256 0.995 (0.217); pt_rm_m128_d1024_h2048 0.571 (0.076); pt_rm_m128_d1024_h2048_c 0.522 (0.007); pt_rm_m128_d256_h512 0.627 (0.142); pt_rm_m128_d256_h512_c 0.462 (0.032) | 5 / 2 / 1 |

### GEMM cells and MLP kernels: Mapal vs cuBLAS / cuBLASLt / CUTLASS (median ms, both processes)

| cell | arch | Mapal default | realization | Mapal _c | Mapal _x | cuBLAS | cuBLASLt | CUTLASS | tuned CUTLASS | default / best library | null |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mm_128x2048x7168 | h100-pcie | 0.3090 | streamk | 0.4740 | 0.4732 | 0.1405 | - | 0.9780 | 0.4208 | 2.199 | 0.002 |
| pt_rm_m128_d1024_h2048 | h100-pcie | 0.1514 | - | = | 0.1779 | 0.0865 | 0.0798 | 0.4060 | 0.1577 | 1.897 | 0.005 |
| pt_rm_m128_d256_h512 | h100-pcie | 0.0329 | - | = | 0.0423 | 0.0365 | 0.0296 | 0.1138 | 0.0306 | 1.111 | 0.009 |
| mm_128x2048x7168 | rtx-3090 | 0.3388 | atomic S=2 | 0.3410 | 0.3748 | 0.1751 | - | 0.7332 | 0.5192 | 1.935 | 0.003 |
| pt_rm_m128_d1024_h2048 | rtx-3090 | 0.1608 | - | = | 0.1812 | 0.1024 | 0.0932 | 0.3533 | 0.1741 | 1.725 | 0.007 |
| pt_rm_m128_d256_h512 | rtx-3090 | 0.0317 | - | = | 0.0399 | 0.0297 | 0.0256 | 0.0993 | 0.0297 | 1.238 | 0.032 |

### Workload suite: iter ms (median of the reps), CPU-only / GPU-only / CPU+GPU, libraries, PyTorch

| workload | arch | CPU-only | GPU-only | CPU+GPU (shipped) | null | kernel / copy / host ms, launches, MB (traced rep) | best library | torch eager | torch compile | shipped / best other |
|---|---|---|---|---|---|---|---|---|---|---|
| decode_1024_256 | h100-pcie | 3657.57 | = shipped | 275.69 | 0.013 | 85.55 / 18.20 / 178.11, 5120, 4 | cache 24.6740 | - | - | 11.173 |
| rm_m128_d1024_h2048 | h100-pcie | 10.09 | = shipped | 0.2202 | 0.043 | 0.17 / 0.05 / 0.00, 2, 0 | - | - | - | - |
| rm_m128_d256_h512 | h100-pcie | 0.7482 | = shipped | 0.1490 | 0.086 | 0.04 / 0.06 / 0.00, 2, 0 | - | - | - | - |
| decode_1024_256 | rtx-3090 | 3950.51 | = shipped | 186.75 | 0.000 | 47.08 / 17.87 / 88.38, 4096, 4 | cache 30.9750 | - | - | 6.029 |
| rm_m128_d1024_h2048 | rtx-3090 | 30.49 | = shipped | 0.2141 | 0.005 | 0.17 / 0.07 / 0.00, 2, 0 | - | - | - | - |
| rm_m128_d256_h512 | rtx-3090 | 2.1282 | = shipped | 0.0942 | 0.036 | 0.04 / 0.07 / 0.00, 2, 0 | - | - | - | - |

