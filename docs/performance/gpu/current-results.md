# Current GPU results and next experiments

Updated 2026-10-23 (S62). Correctness, steady timing, allocation counts and profiling are separate evidence classes. Mapal beats cuBLAS only at 4096³ Contract (3090 0.90, H100 0.96, 4090 0.99) and 4090 16x18432 (0.97); no generic win.

| Question | Verified result | Detailed report |
| --- | --- | --- |
| Does scheduling consume IR and hardware facts consistently? | A proved request/actual-arithmetic mismatch is corrected. 65 library and176 integration tests pass; eight successful LLVM/PTX controls are unchanged. No new GPU speed result. | [Fact-driven scheduling](disassembly/fact-driven-scheduling.md) |
| Are the composed clocks valid? | Completion is included; no proved unit or missing-sync defect. They measure warmed resident computation, with differing input preparation and allocation behavior across arms. | [Measurement boundaries](workloads/measurement-boundaries.md) |
| What did the S62 integration change? | Six items merged (6a0e3a0). M16 Exact/Contract kernels 1.6–10.6× faster than 4e1102c on four parts; int/cuBLAS 0.97–3.01 on M16 cells. Residual-MLP whole programs still slower than base on three parts. | [S62 integrated](#s62-integrated-6a0e3a0), [gap map](disassembly/s62-gap-map.md) |
| Does Mapal beat competitive fused libraries on a residual MLP? | No: at M128/d256/h512 and M128/d1024/h2048, Mapal takes0.073090/0.128935 ms versus tuned fused cuBLASLt0.032659/0.059620 ms (2.24×/2.16× faster). Tuned fused CUTLASS has lower observed medians too; Mapal beats tested eager/graph-disabled Inductor. All30 same-device confirmation/control rows admitted. | [Fused library comparison](workloads/fused-library-comparison.md) |
| Can known dataflow eliminate allocations? | Yes: unused raw-output allocations drop from 7→6, 8→7, 12→10 or 8→6 in the tested cases; controls stay unchanged. 640 ordinary processes and 28 separate counter processes pass. Under forced streams, iteration time improves on 12/7/5/11 of 14 affected cells (RTX 3090/4090/H100/5090, two sessions each), worsens on none; shipped policy emits the same text. | [Allocation latency](memory/allocation-latency.md) |
| Does the 8×64/K16 tile alternative help? | Against the original, FMA improves 1.253× at M16/N2048/K7168 and 1.159× at M16/N1024/K4096; Exact takes 1.227× and 1.013× as long, respectively. Tile aspect and panel depth change together. cuBLAS is faster in all four cells; no selector promotion. | [Tile geometry and panel comparison](kernels/panel-depth-comparison.md) |
| What changes when CUTLASS is tuned? | Six of twelve configurations compiled and passed gates. Selected 16×64×16 is 2.32–2.34× faster than the old default and beats the frozen Mapal baseline in all four cells; cuBLAS remains 4.31–7.01× faster than tuned CUTLASS. | [Competitive CUTLASS comparison](kernels/competitive-cutlass.md) |
| Do we have disassembly from the actual launch? | Yes: the fresh repaired observer admits Mapal1/cuBLAS2/CUTLASS1 selected entries at16×2048×7168. cuBLAS launches a staged GEMM plus split-K reduction;4,608 numerical words pass. Unknown decoder annotations remain; no new timing/counter claim. | [Executed library kernels](disassembly/executed-library-images.md) |
| Does an opt-in reassociation face (ADR-0040, deterministic split-K) close the small-M gap? | Partly. The split kernel is faster than its ordered arm in 47 of 48 split cells on four parts (5090 16×2048×7168: 0.2683 → 0.1454 ms; cuBLAS 0.0512). It beats cuBLAS in five part×cell pairs (M1, M128×1024×4096), and cuBLAS stays faster in every M16 cell. One 3090 loss ships OFF. Bits are deterministic and bit-exact to the partitioned order. | [Raw results](../../../benches/results/gpu/s62/splitk-reassoc/) (`benches/gpu/splitk_report.py`) |
| What does CPU submission cost? | Generated launch enqueue is 1.12–1.16× (4090, 5090; 10/10 kernels in both sessions) and 1.14–1.29× (3090, H100 SXM; not beyond one-process floors of 0.255/0.257) the direct `cuLaunchKernel` loop, 1.16–2.99 µs vs 0.96–2.59 µs per launch. Per-part maximum enqueue/iteration on 512–1024 probes: 0.69–1.25. | [Raw results](../../../benches/results/gpu/s62/runtime-measurements/submit-summary.md) |
| Does retention help requests? | `resident` off costs 1.13–3.36× per request on every part; `devgen` off shows no stable sign; regenerating loop-invariant inputs emits the same inventory as retaining them (no stable difference). Each request still copies two loop-variant operands in and one output out. | [Raw results](../../../benches/results/gpu/s62/runtime-measurements/rg.md) |
| What about 16K and model-shaped matrices? | Earlier large-shape measurements and all failures remain available; they are distinct from the M128 composed test. | [Large matrices](workloads/large-matrices-and-mlp-validation.md) |

The approved repaired observer succeeds on a fresh RTX5090. All504 recovered files are independently rehashed and the owned rental is closed. The first failed trace stays rejected and immutable. Mapal/CUTLASS match256/256 sampled ordered-FMA outputs; cuBLAS matches5/256 and passes the accepted error bounds. The fresh driver610.43.02 capture does not retroactively establish earlier timed-host dispatches. [Live evidence](../../../benches/results/gpu/disassembly/library-kernels/counter-free-capture/rtx-5090/README.md)

Next: the ranked gaps G1–G9 and their order in the [S62 gap map](disassembly/s62-gap-map.md#order).

The allocation correction is integrated; the kernel alternatives remain isolated. Current test status, active experiments and rental ownership are recorded in [the live handoff](../../next-session.md).


## S62 integrated (6a0e3a0)

Kernel ms, L2 flushed, value-gated (bit-exact, or the ADR-0040 bound for reassoc), ABBA; `a / b` = faces. MLP rows: iter ms over 30 processes.
Reassoc has no baseline face: its baseline is the same cell's Exact (or Contract) time. 5090/4090 from session 1 (PTX byte-identical to the final build); H100/3090 from the confirm run.
Raw: [`benches/results/gpu/s62/integrated/`](../../../benches/results/gpu/s62/integrated/), tables by `python3 benches/gpu/integrated_report.py <results files>`.

| part | cell | face | baseline | integrated | cuBLAS | CUTLASS | int/cuBLAS | null |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5090 | 16×2048×7168 | E / C / reassoc | 0.2560 / 0.2621 / — | 0.0922 / 0.0819 / 0.0717 | 0.0512 | 0.5079 | 1.80 / 1.60 / 1.40 | 0.013 |
| 5090 | 16×1024×4096 | E / C / reassoc | 0.1516 / 0.1454 | 0.0492 / 0.0451 / 0.0369 | 0.0184 | 0.2949 | 2.67 / 2.45 / 2.01 | |
| 5090 | 16×18432×7168 | E / C | 2.0828 / 2.0337 | 0.3379 / 0.4588 | 0.3379 | 0.5427 | 1.00 / 1.36 | |
| 5090 | 4096³ | E / C | 3.9004 / 2.3153 | 3.8912 / 2.3204 | 2.1494 | 2.2927 | 1.81 / 1.08 | |
| 5090 | MLP 256/512, 1024/2048 | E / C | 0.0556, 0.0962 / 0.0459, 0.0947 | 0.1069, 0.1883 / 0.0674, 0.1074 | — | — | base/int 0.52, 0.51 / 0.68, 0.88 | 0.029 |
| 5090 | composed `pt_pm_2048_32` | E / C | 1.0054 / 0.6102 | 0.9863 / 0.6017 | — | — | base/int 1.02 / 1.01 | |
| 4090 | 16×2048×7168 | E / C | 0.5153 / 0.5161 | 0.1055 / 0.0993 | 0.0961 | 0.5243 | 1.10 / 1.03 | 0.024 |
| 4090 | 16×1024×4096 | E / C / reassoc | 0.2868 / 0.1341 | 0.0696 / 0.0379 / 0.0410 | 0.0369 | 0.3021 | 1.89 / 1.03 / 1.11 | |
| 4090 | 16×18432×7168 | E / C | 2.9738 / 3.1058 | 0.6176 / 0.6185 | 0.6359 | 0.9708 | 0.97 / 0.97 | |
| 4090 | 4096³ | E / C / reassoc+c | 5.3248 / 2.9174 | 5.2941 / 2.9194 / 2.8058 | 2.9379 | 2.9736 | 1.80 / 0.99 / 0.96 | |
| 4090 | MLP 256/512, 1024/2048 | E / C | 0.0863, 0.1518 / 0.1198, 0.1323 | 0.0896, 0.1579 / 0.2746, 0.2149 | — | — | base/int 0.96, 0.96 / 0.44, 0.62 | 0.172 |
| 4090 | composed | E / C | 1.2608 / 0.7853 | 1.2230 / 0.7300 | — | — | 1.03 / 1.08 | |
| H100 PCIe | 16×2048×7168 | E / C / reassoc+c | 0.5270 / 0.8100 | 0.1417 / 0.1305 / 0.0883 | 0.0689 | 0.8349 | 2.06 / 1.89 / 1.28 | 0.008 |
| H100 PCIe | 16×1024×4096 | E / C / reassoc | 0.2868 / 0.4614 | 0.0791 / 0.0878 / 0.0381 | 0.0292 | 0.4779 | 2.71 / 3.01 / 1.31 | |
| H100 PCIe | 16×18432×7168 | E / C | 1.9785 / 2.1907 | 0.5252 / 0.4826 | 0.3679 | 1.6519 | 1.43 / 1.31 | |
| H100 PCIe | 4096³ | E / C | 6.5151 / 3.6361 | 6.4762 / 3.6354 | 3.7876 | 4.1956 | 1.71 / 0.96 | |
| H100 PCIe | MLP 256/512, 1024/2048 | E / C | 0.0684, 0.1782 / 0.1481, 0.4908 | 0.0684, 0.1783 / 0.1925, 0.3916 | — | — | base/int 1.00, 1.00 / 0.77, 1.25 | 0.005 |
| H100 PCIe | composed | E / C | 2.3164 / 1.3082 | 2.2976 / 1.2957 | — | — | 1.01 / 1.01 | |
| 3090 | 16×2048×7168 | E / C / reassoc+c | 0.4526 / 0.6605 | 0.1516 / 0.1526 / 0.1321 | 0.0922 | 0.8960 | 1.64 / 1.66 / 1.43 | 0.077 (median 0.007) |
| 3090 | 16×1024×4096 | E / C / reassoc+c | 0.1741 / 0.1649 | 0.0707 / 0.1004 / 0.0891 | 0.0369 | 0.4772 | 1.92 / 2.72 / 2.42 | |
| 3090 | 16×18432×7168 | E / C | 8.6303 / 8.7337 | 0.8161 / 0.9206 | 0.8120 | 2.5539 | 1.01 / 1.13 | |
| 3090 | 4096³ | E / C | 14.3483 / 9.3573 | 14.7804 / 8.6518 | 9.5744 | 10.2543 | 1.54 / 0.90 | |
| 3090 | MLP 256/512, 1024/2048 | E / C | 0.0800, 0.2119 / 0.1081, 0.3475 | 0.1690, 0.2534 / 0.1094, 0.3508 | — | — | base/int 0.47, 0.84 / 0.99, 0.99 | 0.146 |
| 3090 | composed | E / C | 3.1178 / 1.7763 | 3.1261 / 1.7872 | — | — | 1.00 / 0.99 | |

H100 MLP Exact: the confirm run's `off-panel-regs` arm (the only rung that changes there). 3090 clock drift on long kernels: cuBLAS 16×18432 0.812 vs 0.976 ms between processes.

| rung | 5090 | 4090 | H100 PCIe | 3090 |
| --- | --- | --- | --- | --- |
| row-fit | keep (Filled/Exact OFF) | keep (Filled OFF) | keep; Filled OFF, Under/Contract OFF | keep (Filled OFF) |
| wave-gate | keep (Filled/Contract OFF) | keep | keep (Under/Contract OFF) | keep |
| grid-share | keep | keep (Under/Exact OFF) | Filled/Contract OFF, Under/Exact OFF | Filled OFF, Under/Contract OFF |
| panel-regs | keep | keep | Under/Exact OFF | Filled OFF, Under/Contract OFF |
| fill-rows | keep | keep | Under/Exact OFF | keep |
| cell-fill | keep | keep | keep | keep |
| asyncstage | OFF | OFF | OFF | OFF |
| splitk | keep | keep | keep | Filled/Exact OFF, Under/Exact OFF |
| hostopt, place | keep | keep | keep | keep |
| runtime (arena, resident, devgen) | keep (streams OFF) | keep (streams OFF) | keep (streams OFF) | keep (streams OFF) |

| still open | measured |
| --- | --- |
| residual MLP single iteration slower than base (base/int) | 5090 E 0.51–0.52, C 0.68–0.88; 4090 C 0.44–0.62; 3090 E 0.47–0.84; H100 C d256 0.77. Turning off grid-share or panel-regs recovers it but costs 1.2–2.5× on M16 cells of the same class; traced kernel ms barely differ (5090 d256 E 0.033 vs 0.029) |
| grid-share × splitk | leaving grid-share out is faster on 8 of 11 reassoc cells, slower on 3 (1.05–1.14) |
| H100 16×1024 reassoc+contract after the row-fit flip | 0.0383 vs 0.0366 ms (noise 0.002); Contract on the same cell 4.13× faster |
| wave-gate on reassoc faces | off/on 0.40–0.95 on 16×1024; 1.25–4.75 on E/C of the same cells |

## S63 default float face (ADR-0040)

Default = fma steps, order free, ADR-0040 D2 bound. Kernel ms, L2 flushed, two processes × 30 ABBA; every arm inside D2 (290 of 290), deterministic arms bit-exact. Shipped = after the S63 `GPU_RUNGS_OFF` rows (each a session arm). Raw: [`benches/results/gpu/s63/default-float/`](../../../benches/results/gpu/s63/default-float/) (`tables-composed.txt`).

| part | cell | shipped | default | fixed (_rc) | contract | exact | cuBLAS | CUTLASS | tuned CUTLASS | default / best lib | null |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5090 | 16×1024×4096 | none | 0.0471 | 0.0471 | 0.0471 | 0.0512 | 0.0205 | 0.3011 | 0.1311 | 2.298 | 0.000 |
| 5090 | 16×2048×7168 | none | 0.0819 | 0.0839 | 0.0819 | 0.0922 | 0.0512 | 0.5181 | 0.2232 | 1.600 | 0.012 |
| 5090 | 16×18432×7168 | stream-K | 0.4137 | 0.4628 | 0.4628 | 0.3441 | 0.3400 | 0.5612 | 0.3707 | 1.217 | 0.000 |
| 5090 | 128×2048×7168 | none | 0.1495 | 0.1495 | 0.1516 | 0.3031 | 0.0942 | 0.5632 | 0.2273 | 1.587 | 0.000 |
| 5090 | 4096³ | stream-K | 2.8252 | 2.8334 | 2.8221 | 4.3909 | 2.5416 | 2.8109 | 5.7344 | 1.112 | 0.001 |
| 4090 | 16×1024×4096 | none | 0.0369 | 0.0378 | 0.0371 | 0.0677 | 0.0358 | 0.3062 | 0.1188 | 1.031 | 0.011 |
| 4090 | 16×2048×7168 | none | 0.1014 | 0.1014 | 0.1013 | 0.1065 | 0.0952 | 0.5325 | 0.2058 | 1.065 | 0.000 |
| 4090 | 16×18432×7168 | none | 0.6178 | 0.6175 | 0.6175 | 0.6175 | 0.6359 | 0.9861 | 0.6144 | 1.006 | 0.002 |
| 4090 | 128×2048×7168 | none | 0.1649 | 0.1649 | 0.1647 | 0.2068 | 0.1505 | 0.5386 | 0.2304 | 1.096 | 0.005 |
| 4090 | 4096³ | none | 3.0126 | 3.1365 | 2.9911 | 5.4333 | 3.3075 | 3.3526 | 9.0388 | 0.911 | 0.002 |
| H100 PCIe | 16×1024×4096 | stream-K | 0.0467 | 0.0507 | 0.1016 | 0.0906 | 0.0333 | 0.5561 | 0.1933 | 1.402 | 0.004 |
| H100 PCIe | 16×2048×7168 | workspace S=7 | 0.0980 | 0.0977 | 0.1581 | 0.1714 | 0.0763 | 0.9747 | 0.3451 | 1.284 | 0.002 |
| H100 PCIe | 16×18432×7168 | none | 0.5689 | 0.5684 | 0.5697 | 0.6368 | 0.4092 | 1.9254 | 0.4391 | 1.390 | 0.002 |
| H100 PCIe | 128×2048×7168 | stream-K | 0.5417 | 0.9057 | 0.9055 | 0.4778 | 0.1428 | 0.9857 | 0.4242 | 3.793 | 0.003 |
| H100 PCIe | 4096³ | none | 4.4739 | 4.4740 | 4.4740 | 7.8480 | 4.6335 | 4.8971 | 11.0880 | 0.966 | 0.000 |
| 3090 | 16×1024×4096 | atomic S=4 | 0.0891 | 0.0891 | 0.0942 | 0.0645 | 0.0348 | 0.4270 | 0.1331 | 2.560 | 0.012 |
| 3090 | 16×2048×7168 | workspace S=4 | 0.1198 | 0.1198 | 0.1331 | 0.1341 | 0.0891 | 0.7772 | 0.2304 | 1.345 | 0.007 |
| 3090 | 16×18432×7168 | none | 0.6984 | 0.6953 | 0.6943 | 0.7229 | 0.6676 | 1.9548 | 0.6912 | 1.046 | 0.002 |
| 3090 | 128×2048×7168 | atomic S=8 | 0.3195 | 0.3573 | 0.7158 | 0.4198 | 0.1956 | 0.8253 | 0.5806 | 1.633 | 0.003 |
| 3090 | 4096³ | none | 7.0277 | 7.7015 | 7.0359 | 12.6945 | 7.1741 | 7.3964 | 19.5369 | 0.980 | 0.014 |

| part | residual MLP kernels (ms, p1/p2) | default (= contract) | exact | cuBLASLt | tuned CUTLASS | default / best lib |
| --- | --- | --- | --- | --- | --- | --- |
| 5090 | d256 h512 / d1024 h2048 | 0.0162/0.0167, 0.0591/0.0587 | 0.0288/0.0279, 0.1132/0.1130 | 0.0192/0.0187, 0.0385/0.0381 | 0.0182/0.0176, 0.0633/0.0627 | 0.890/0.949, 1.535/1.541 |
| 4090 | d256 h512 / d1024 h2048 | 0.0195/0.0195, 0.0613/0.0613 | 0.0266/0.0265, 0.0837/0.0837 | 0.0165/0.0167, 0.0471/0.0470 | 0.0195/0.0195, 0.0716/0.0717 | 1.182/1.168, 1.301/1.304 |
| H100 PCIe | d256 h512 / d1024 h2048 | 0.1042/0.1043, 0.3849/0.3853 | 0.0428/0.0431, 0.1793/0.1798 | 0.0287/0.0291, 0.0800/0.0800 | 0.0309/0.0311, 0.1596/0.1603 | 3.631/3.584, 4.811/4.816 |
| 3090 | d256 h512 / d1024 h2048 | 0.0686/0.0686, 0.3072/0.3072 | 0.0399/0.0399, 0.1802/0.1802 | 0.0256/0.0256, 0.0932/0.0932 | 0.0287/0.0287, 0.1731/0.1731 | 2.680/2.680, 3.296/3.296 |

| rung (Contract face) | 5090 | 4090 | H100 PCIe | 3090 |
| --- | --- | --- | --- | --- |
| streamk | Under OFF (16×2048 1.050), Filled ON (16×18432 0.894) | Filled OFF (1.057) | Under ON (0.597–0.923), Filled OFF (4096³ 1.133) | Under OFF (128×2048 1.041), Filled OFF (16×2048 1.248) |
| splitk-atomic | — | — | OFF (0.992–1.000 vs workspace) | Under ON (128×2048 0.897), Filled OFF (1.000/0.992) |
| splitk | — | — | ON (0.500, 0.620) | ON (0.499, 0.893) |

[Browse GPU reports](README.md) · [Browse raw GPU evidence](../../../benches/results/gpu/README.md)
