# Current measured implementation vs tested libraries

Independent report check: 161 CSV rows, 282 ratio calculations and 244 medians checked
directly against saved raw records; source, proof and launched-PTX hashes matched.
See the [check receipt](../../benches/results-s54-gpu/s56-current-library-independent-check.json).

Accepted current selection runs only (2026-10-21). The final five-profile/six-size integrated results remain pending in [S54 GPU fusion](s54-gpu-fusion.md).

## Mapal face and observed hardware path

Observed in each launched tile entry at all four measured sizes (1024, 2048, 4096, 8192):

| Mapal face | Numerical contract | H100 PCIe PTX realization | RTX 5090 PTX realization |
|---|---|---|---|
| Exact | Ordered separate FP32 multiplication and addition | SIMT `mul.rn.f32` + `add.rn.f32` | SIMT `mul.rn.f32` + `add.rn.f32` |
| Contract | Ordered FP32 fused multiply-add (`fmaf` oracle) | Ordinary FP32 `fma.rn.f32` | Ordinary FP32 `fma.rn.f32` |
| TF32 | Reduced-precision tensor inputs; existing error gate | Tensor Cores through `wgmma.mma_async` (WGMMA) | Tensor Cores through `mma.sync` (MMA) |

The launched entry name and frozen PTX SHA256 are in the kernel CSV. Instruction presence
identifies the emitted path; it does not measure hardware utilization. Library numerical
contracts and tested configurations are described below and in the
[tested harness](../../benches/gpu/tile_bench.cu).

The [kernel evidence audit](s56-selection-kernel-evidence.md) compares actual launch
resources and records which machine-code artifacts exist. Its
[reproduction receipt](../../benches/results-s54-gpu/s56-selection-kernel-evidence.json)
binds 112 artifact digests, exact SQL queries and instruction line references. Mapal's
saved SASS is a separate offline assembly, not the captured driver-JIT image used for
timing. No saved hardware counters establish occupancy, cache hit rates or a cause for
the measured gaps. The audit and receipt are byte-identical copies of the original
temporary evidence files. Raw selection bundles and profiling archives have an ignored
recovery copy described in `docs/next-session.md`; the report and kernel CSV also have
the durable copies linked here.

## Single-kernel GEMM: same cudaEvent timing boundary

Each cell is **cuBLAS / Mapal, CUTLASS / Mapal**; greater than one favors Mapal. Keep hosts separate. Mapal faces are Exact (ordered separate mul/add), Contract (FMA), and TF32 tensor precision. The tested libraries do not promise Mapal's bitwise order.

| n | Mapal face | H100 PCIe | RTX 5090 A2 | RTX 5090 B |
|---:|---|---:|---:|---:|
| 1024 | Exact | 0.467, 0.660 | 0.667, 1.364 | 0.629, 1.314 |
| 1024 | FMA / Contract | 0.681, 0.954 | 0.816, 1.667 | 0.759, 1.586 |
| 1024 | TF32 | 0.564, 0.791 | 0.789, 1.316 | 0.749, 1.298 |
| 2048 | Exact | 0.481, 0.628 | 0.539, 0.608 | 0.548, 0.616 |
| 2048 | FMA / Contract | 0.847, 1.107 | 0.867, 0.979 | 0.866, 0.980 |
| 2048 | TF32 | 0.607, 0.958 | 0.860, 0.888 | 0.864, 0.891 |
| 4096 | Exact | 0.589, 0.660 | 0.574, 0.631 | 0.566, 0.607 |
| 4096 | FMA / Contract † | 1.029, 1.148 | 0.911, 1.001 | 0.932, 1.002 |
| 4096 | TF32 † | 0.661, 1.074 | 0.893, 0.890 | 0.914, 0.911 |
| 8192 | Exact | 0.520, 0.671 | 0.569, 0.645 | 0.574, 0.565 |
| 8192 | FMA / Contract † | 0.943, 1.153 | 1.047, 1.062 | 1.054, 1.057 |
| 8192 | TF32 † | 0.551, 1.187 | 0.873, 0.904 | 0.910, 0.913 |

† RTX 5090 values use the measured **pre-final shipped raster table**, which forces grouping on Contract/TF32. Final selection restores the derived L2 gate; these affected cells are not measurements of that final selection. Other kernel cells are the measured shipped configuration. Streams does not affect this single-kernel timing.

Library configurations are the pinned harness configurations: FP32 cuBLAS SGEMM and stock CUTLASS SIMT; TF32 uses cuBLAS TF32 math mode and explicit SM80 CUTLASS 128×128×16 TensorOp. These are tested configurations, not the best possible CUTLASS configuration on every GPU.

Hosts: H100 machine 81035 / host 415974 (repeat historical host); 5090 A2 machine 55053 / host 253861, driver 570.211.01, 500 W; 5090 B machine 55265 / host 33035, driver 595.104.02, 575 W. Both 5090 physical hosts are independent.

Exact medians, ratio arithmetic, source lines and raw/proof SHA256: [current-library-kernels.csv](../../benches/results-s54-gpu/s56-current-library-kernels.csv).



Observed launched PTX entry paths at all four measured sizes: H100 TF32 uses **WGMMA**; RTX5090 TF32 uses **MMA**. Exact uses separate FP32 mul/add and Contract uses FP32 FMA on all three sessions. Kernel CSV includes entry name and PTX SHA256.

## All measured workload families and sizes

Mapal columns are **untraced ordinary program iteration ms** (including its host work/transmissions). The named best library is **CUDA-event device time on device-built inputs**. Library/iteration is a cross-scope diagnostic, not a whole-program speedup. Library values passed the existing tolerance gate; they do not share Exact Mapal's bitwise-order contract. `lt`/`lt2` are the harness's cuBLASLt implementations; `eager` is eager cuBLAS, `streams` its streams leg, and `cutlass` the pinned CUTLASS leg. Twin is explicitly unsupported by the library harness.

Selection of saved rows is explicit: use measured OFF-streams on RTX5090 where present (also re-enables the arena), otherwise shipped; use H100 OFF-window only where that distinct arm exists. H100 MLP4096 remains shipped and is unaffected by the window change. Cells marked **†** still retain the pre-final RTX5090 raster policy; the restored gate is not combined with OFF-streams in a saved measurement. No combined final number is fabricated.

### H100

| Workload | n | Exact iteration ms | Contract iteration ms | Best library event ms (implementation) | library/Exact | library/Contract | saved arm Exact / Contract |
|---|---:|---:|---:|---|---:|---:|---|
| diamond | 2048 | 3.396958 | 1.934879 | 1.469 (streams) | 0.433 | 0.759 | on / off-l2-window |
| diamond | 4096 | 19.125683 | 10.820222 | 10.789 (streams) | 0.564 | 0.997 | on / on |
| mlp | 2048 | 2.296932 | 1.291543 | 1.030 (lt) | 0.448 | 0.797 | on / off-l2-window |
| mlp | 4096 | 12.836336 | 7.221243 | 7.494 (lt) | 0.584 | 1.038 | on / on |
| mlp2 | 2048 | 4.558129 | 2.589975 | 1.823 (lt2) | 0.400 | 0.704 | on / off-l2-window |
| mlp2 | 4096 | 25.616234 | 14.469079 | 14.053 (streams) | 0.549 | 0.971 | on / on |

### 5090 A2

| Workload | n | Exact iteration ms | Contract iteration ms | Best library event ms (implementation) | library/Exact | library/Contract | saved arm Exact / Contract |
|---|---:|---:|---:|---|---:|---:|---|
| diamond | 512 | 0.062519 | 0.058549 | 0.045 (streams) | 0.721 | 0.770 | off-streams / off-streams |
| diamond | 1024 | 0.229936 | 0.188477 | 0.131 (streams) | 0.570 | 0.696 | on / on |
| diamond | 2048 | 1.440115 | 0.885265 | 0.801 (lt) | 0.556 | 0.905 | on / on |
| diamond † | 4096 | 10.513357 | 6.625454 | 6.682 (streams) | 0.636 | 1.008 | off-streams / off-streams |
| fanout | 512 | 0.050199 | 0.044490 | 0.033 (streams) | 0.657 | 0.742 | on / on |
| fanout | 1024 | 0.164297 | 0.131908 | 0.090 (streams) | 0.550 | 0.685 | on / on |
| fanout | 2048 | 0.984553 | 0.605850 | 0.512 (streams) | 0.520 | 0.846 | on / on |
| fanout † | 4096 | 7.051748 | 4.435963 | 4.250 (streams) | 0.603 | 0.958 | on / on |
| indep | 512 | 0.048049 | 0.045880 | 0.029 (streams) | 0.597 | 0.626 | off-streams / off-streams |
| indep | 1024 | 0.157627 | 0.129758 | 0.084 (streams) | 0.533 | 0.647 | on / on |
| indep | 2048 | 0.969743 | 0.599220 | 0.524 (lt) | 0.541 | 0.875 | on / on |
| indep † | 4096 | 7.017179 | 4.433054 | 4.328 (streams) | 0.617 | 0.976 | off-streams / off-streams |
| mlp | 512 | 0.047919 | 0.047830 | 0.031 (lt) | 0.641 | 0.642 | on / on |
| mlp | 1024 | 0.161668 | 0.130908 | 0.090 (lt) | 0.557 | 0.688 | on / on |
| mlp | 2048 | 0.977374 | 0.603430 | 0.516 (lt) | 0.528 | 0.855 | on / on |
| mlp † | 4096 | 6.976130 | 4.411875 | 4.258 (lt) | 0.610 | 0.965 | on / on |
| mlp2 | 512 | 0.100029 | 0.096498 | 0.045 (lt2) | 0.446 | 0.462 | off-streams / off-streams |
| mlp2 | 1024 | 0.323205 | 0.266565 | 0.155 (lt2) | 0.480 | 0.583 | on / on |
| mlp2 | 2048 | 1.938897 | 1.216499 | 1.073 (lt2) | 0.553 | 0.882 | on / on |
| mlp2 † | 4096 | 13.850311 | 8.789568 | 8.547 (lt2) | 0.617 | 0.972 | off-streams / off-streams |
| twin | 512 | 0.046490 | 0.044910 | unsupported (twin) | — | — | off-streams / off-streams |
| twin | 1024 | 0.159166 | 0.128038 | unsupported (twin) | — | — | on / on |
| twin | 2048 | 0.975452 | 0.594468 | unsupported (twin) | — | — | on / on |
| twin † | 4096 | 7.011611 | 4.395169 | unsupported (twin) | — | — | off-streams / off-streams |

### 5090 B

| Workload | n | Exact iteration ms | Contract iteration ms | Best library event ms (implementation) | library/Exact | library/Contract | saved arm Exact / Contract |
|---|---:|---:|---:|---|---:|---:|---|
| diamond | 512 | 0.069560 | 0.067291 | 0.035 (streams) | 0.500 | 0.517 | off-streams / off-streams |
| diamond | 1024 | 0.255691 | 0.200341 | 0.135 (streams) | 0.528 | 0.674 | on / on |
| diamond | 2048 | 1.497306 | 0.908984 | 0.782 (streams) | 0.522 | 0.861 | on / on |
| diamond † | 4096 | 10.266605 | 6.400179 | 6.159 (streams) | 0.600 | 0.962 | off-streams / off-streams |
| fanout | 512 | 0.077780 | 0.057310 | 0.027 (streams) | 0.345 | 0.468 | on / on |
| fanout | 1024 | 0.184681 | 0.146241 | 0.097 (streams) | 0.523 | 0.660 | on / on |
| fanout | 2048 | 1.040195 | 0.641553 | 0.520 (streams) | 0.500 | 0.811 | on / on |
| fanout † | 4096 | 6.869176 | 4.287652 | 4.143 (streams) | 0.603 | 0.966 | on / on |
| indep | 512 | 0.066500 | 0.062000 | 0.027 (streams) | 0.400 | 0.429 | off-streams / off-streams |
| indep | 1024 | 0.186781 | 0.143951 | 0.092 (streams) | 0.493 | 0.639 | on / on |
| indep | 2048 | 1.037025 | 0.643144 | 0.524 (streams) | 0.505 | 0.815 | on / on |
| indep † | 4096 | 6.876279 | 4.282014 | 4.040 (streams) | 0.588 | 0.943 | off-streams / off-streams |
| mlp | 512 | 0.074091 | 0.053700 | 0.027 (lt) | 0.359 | 0.495 | on / on |
| mlp | 1024 | 0.178282 | 0.147261 | 0.094 (lt) | 0.528 | 0.640 | on / on |
| mlp | 2048 | 1.034146 | 0.630924 | 0.520 (lt) | 0.503 | 0.825 | on / on |
| mlp † | 4096 | 6.833942 | 4.262646 | 4.152 (lt) | 0.608 | 0.974 | on / on |
| mlp2 | 512 | 0.135571 | 0.124541 | 0.040 (lt2) | 0.298 | 0.324 | off-streams / off-streams |
| mlp2 | 1024 | 0.354092 | 0.293402 | 0.166 (lt2) | 0.467 | 0.564 | on / on |
| mlp2 | 2048 | 2.040852 | 1.273078 | 1.026 (streams) | 0.503 | 0.806 | on / on |
| mlp2 † | 4096 | 13.635961 | 8.472291 | 7.867 (lt2) | 0.577 | 0.929 | off-streams / off-streams |
| twin | 512 | 0.053971 | 0.052301 | unsupported (twin) | — | — | off-streams / off-streams |
| twin | 1024 | 0.173702 | 0.143191 | unsupported (twin) | — | — | on / on |
| twin | 2048 | 1.024480 | 0.625772 | unsupported (twin) | — | — | on / on |
| twin † | 4096 | 6.843751 | 4.245028 | unsupported (twin) | — | — | off-streams / off-streams |

Full min/median/max, selected-arm reasons, allocation/launch counts, source lines and hashes: [probe CSV](../../benches/results-s54-gpu/s56-current-library-probes.csv).

## Derived retention and allocation: separate measured boundaries

Mapal's relevant mechanism is dataflow-derived retention and incremental recomputation:
the existing row-change and row-locality analyses identify which projected rows remain
valid, which must be recomputed, and which host changes need transmission. Residency and
storage reuse are related but distinct decisions. The compiler cannot infer value validity
from lifetime alone. The [implementation map](../components/backend-nvptx/IMPLEMENTATION.md)
links the current retention, residency, delta-upload and arena realizations.

The explicit handwritten KV-cache comparison immediately below is a reference strategy.
It does not establish that the compiler discovered the strategy, nor measure the isolated
benefit of Mapal's analysis. L2 cache policy and allocator caching are separate mechanisms.

### Library KV-cache comparison (same host-loop wall boundary)

This compares the two handwritten library decode implementations with 1024 steps × 256 width. It is a library cache/no-cache A/B, not Mapal process wall. Both value gates passed.

| Host | Cache loop wall ms | No-cache loop wall ms | no-cache/cache wall | Cache event ms | No-cache event ms |
|---|---:|---:|---:|---:|---:|
| H100 | 21.175 | 40.411 | 1.908× | 21.167 | 40.402 |
| 5090 A2 | 24.904 | 33.587 | 1.349× | 24.899 | 33.583 |
| 5090 B | 34.842 | 34.617 | 0.994× | 34.837 | 34.612 |

Cache/no-cache wall favors cache on H100 (1.908×) and RTX5090 A2 (1.349×), while RTX5090 B shows no median benefit (0.994×). This is not a universal caching speedup.

### Mapal combined R5 controls (ordinary process wall)

These are saved selection-bundle ablations, not automatic measurements of the final table. The H100 off-retain-resident arm disables retention while retaining the other pre-final settings; notably Contract still has L2-window ON (the frozen host IR calls mapal_gpu_launch_l2), while final selection disables it. The baseline disables retain-resident+delta-h2d+lane-stride together. Thus the H100 Contract pair is conditional on the older window setting and cannot establish the final retain-OFF/window-OFF composition. RTX5090 already had retain-resident OFF; its frozen decode host IR uses legacy launches with no event/launchv calls, so the saved decode has no active stream chain. Device raster applicability is not assumed from that fact. All pairs below remain explicitly saved-configuration combined R5 comparisons until final runs establish the complete current baseline. Process startup is included; this is not an isolated caching/retention gain. Traced kernels are separate and can have the opposite sign.

| Host / face | selected saved arm | Selected process wall ms | R5-disabled process wall ms | disabled/selected wall | Selected traced kernels ms | Disabled traced kernels ms |
|---|---|---:|---:|---:|---:|---:|
| H100 _exact | off-retain-resident | 842 | 1146 | 1.361× | 131.439 | 84.121 |
| H100 _c | off-retain-resident | 878 | 1089 | 1.240× | 161.032 | 115.411 |
| 5090 A2 _exact | on | 641 | 890 | 1.388× | 75.984 | 47.819 |
| 5090 A2 _c | on | 630 | 899 | 1.427× | 73.981 | 44.749 |
| 5090 B _exact | on | 586 | 979 | 1.671× | 83.071 | 64.623 |
| 5090 B _c | on | 567 | 952 | 1.679× | 79.035 | 59.363 |

### Integrated stream/arena A/B (ordinary iteration)

OFF-streams re-enables the existing device arena because `resident_plan` only creates it with an empty stream-chain map. Actual traced allocation counts drop 14→1 for MLP2 and 6→1 for twin. Frozen MLP2 Contract ON host IR has ten allocation and twelve free call sites inside the timer interval; OFF has six arena lookups and no frees there. Ordinary nsys traces of MLP2 Contract4096 show no positive overlap between the two tile-kernel streams on either measured host. These are scoped observations; profiled API durations can include waits and must not be added to kernel durations as wholly removable overhead. The sole OFF-arm arena allocation is before the program timer; process wall still pays it. Thus these are measured integrated placement gains, **not isolated scheduling or allocator speedups**. Attributing the entire difference to either cause is unproven.

| Host / workload / face | ON iteration ms | OFF iteration ms | ON/OFF | Actual allocations ON→OFF | ON/OFF traced kernel ms |
|---|---:|---:|---:|---|---|
| 5090 A2 pt_twin_4096 _exact | 7.520672 | 7.011611 | 1.073× | 6→1 | 7.021/6.988 |
| 5090 A2 pt_twin_4096 _c | 5.110095 | 4.395169 | 1.163× | 6→1 | 4.421/4.379 |
| 5090 A2 pt_mlp2_4096 _exact | 16.560826 | 13.850311 | 1.196× | 14→1 | 13.834/13.817 |
| 5090 A2 pt_mlp2_4096 _c | 11.490262 | 8.789568 | 1.307× | 14→1 | 8.787/8.761 |
| 5090 B pt_twin_4096 _exact | 7.594467 | 6.843751 | 1.110× | 6→1 | 6.799/6.786 |
| 5090 B pt_twin_4096 _c | 5.228815 | 4.245028 | 1.232× | 6→1 | 4.241/4.214 |
| 5090 B pt_mlp2_4096 _exact | 17.242973 | 13.635961 | 1.265× | 14→1 | 13.566/13.550 |
| 5090 B pt_mlp2_4096 _c | 12.068032 | 8.472291 | 1.424× | 14→1 | 8.429/8.419 |

Caching/R5/allocation support: [cache](../../benches/results-s54-gpu/s56-current-library-cache.csv), [R5](../../benches/results-s54-gpu/s56-current-library-r5.csv), [allocation](../../benches/results-s54-gpu/s56-current-library-allocation.csv). Full structured support: [current-library-comparison.json](../../benches/results-s54-gpu/s56-current-library-comparison.json). All saved data remain tied to accepted raw/proof SHA256; no old failed integrated prefix is used. Final changed-table measurements across all five profiles and all six kernel sizes are pending.

## Durable accepted evidence

CSV/JSON preserve the original recovery-file paths as provenance. These durable raw/proof
copies have the same SHA256 values recorded in the supporting data:

| Session | Raw measurements | Acceptance proof | Provenance |
|---|---|---|---|
| H100 PCIe | [raw](../../benches/results-s54-gpu/s56-selection-h100-pcie.txt) | [verification](../../benches/results-s54-gpu/s56-selection-h100-pcie.verification.json) | [provenance](../../benches/results-s54-gpu/s56-selection-h100-pcie.json) |
| RTX 5090 A2 | [raw](../../benches/results-s54-gpu/s56-selection-rtx-5090-a2.txt) | [verification](../../benches/results-s54-gpu/s56-selection-rtx-5090-a2.verification.json) | [provenance](../../benches/results-s54-gpu/s56-selection-rtx-5090-a2.json) |
| RTX 5090 B | [raw](../../benches/results-s54-gpu/s56-selection-rtx-5090-b.txt) | [verification](../../benches/results-s54-gpu/s56-selection-rtx-5090-b.verification.json) | [provenance](../../benches/results-s54-gpu/s56-selection-rtx-5090-b.json) |

See [S54 per-architecture selection](s54-per-arch-selection.md) for the acceptance decisions
and counterevidence. [Final GPU fusion acceptance](s54-gpu-fusion.md) remains pending.
