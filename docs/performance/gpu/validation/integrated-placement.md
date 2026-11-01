# Integrated GPU placement validation

Historical provenance: S54.

Validation recovery, 2026-10-21. **All five final primary runs and both independent RTX 4090/5090 confirmations pass.** Each task rental has an independent absence receipt.
The [per-architecture report](architecture-selection.md) owns selection decisions.
The [implementation map](../../../components/backend-nvptx/IMPLEMENTATION.md) names the
realized placement rules and runtime transmissions.

## Evidence boundaries

Historical `integrated-*-prefix.txt` runs predate the arena repair and contain failures.
They do not validate the recovered compiler. Fresh bundles are built from one frozen
source revision, with current runtime/compiler hashes and regenerated reference outputs.
CUTLASS is pinned to `91843b71d90737e0a9b86f7d9df213cff5a3526f`.

An accepted run must match its intended source and arm inventories, pass value and trap
checks, preserve successful ordinary/traced process statuses, supply every requested
measurement repetition and library value gate, and include resource and profiler artifacts.
The rental's physical machine and cleanup are checked separately. A completion marker alone
does not certify these properties.

## Measurement method

- Program iteration time comes from the ordinary, untraced execution. Its boundary includes
  the program body and necessary transfers; generated inputs outside the clock are excluded.
  Runtime tracing serializes launches, so traced kernels cannot establish stream overlap.
- Process wall includes context/module startup and host work. Traced wall minus load is a
  separate diagnostic; it is not a substitute for the ordinary program clock.
- Library kernel/event timings measure their own device workloads. Compare like workloads
  and label differing timing boundaries; process wall divided by device-only library time
  is not an end-to-end speed ratio.
- Exact, Contract and TF32 remain separate. This work changes no numerical policy.
- Ratios greater than one favor Mapal or the named ON arm, as specified by each table.
  Kernel arm comparisons use their own session's library reference and null control.
- Report full distributions and every loss. Min/median/max ranges are not confidence
  intervals; interleaved repetitions and independent hosts expose different sources of noise.

The `on` arm follows the measured table. `off-arch-rungs` bypasses that table and enables
all rungs. The placement-disabled composite switches off epilogue, devmap, cell-law,
rowfold/layout, streams, pinned, PDL, arena, retain-resident, delta-h2d, lane-stride and
L2 window; raster remains independent. Neither control is an S52 compiler baseline.

## Final primary measurements

All five primary runs match frozen source `b4152c5`, pass complete values/traps, timing/status, library, resource and profiling gates, and have independent rental-absence receipts. Both required independent confirmations also pass; their measurements and limits appear below. Every exported cell retains raw source lines and proof hashes. An independent stdlib-only checker recomputed 38,577 numeric fields and 19,800 kernel samples directly from raw records with zero differences. It imports neither the exporter nor validator. Program and kernel samples are checked in full; library summaries are checked against their printed raw summaries because that harness does not save every library sample. The [older comparison](../kernels/library-comparison.md) remains selection evidence with its pre-final settings explicitly marked.

### Single-kernel medians

Each table is one machine. Times are **CUDA-event milliseconds**, lower is faster. Exact is the ordered separate-multiply/add face; FMA is Contract; TF32 uses the requested reduced face. Library reductions need not match Mapal bitwise. These are the pinned tested cuBLAS/CUTLASS configurations, not a search for each library's best possible implementation.

For these extracted kernels, the harness enforces Mapal's numeric gate; cuBLAS and
CUTLASS errors are diagnostics. TF32's gate compares Mapal's sampled error with four
times cuBLAS's error, without a fixed absolute accuracy ceiling. Its reported error
uses denominator `max(abs(f64_ref), 0.001)`. This differs from the explicit library
reference gates in the separate composed/decode harnesses.

#### RTX 3090 — machine 13014 / host 49107

| N | Face | Mapal ms | cuBLAS ms | CUTLASS ms | cuBLAS/Mapal | CUTLASS/Mapal |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | Exact | 0.1772 | 0.1044 | 0.1075 | 0.589 | 0.607 |
| 1024 | FMA | 0.1475 | 0.1106 | 0.1137 | 0.750 | 0.771 |
| 1024 | TF32 | 0.0829 | 0.0821 | 0.0819 | 0.990 | 0.988 |
| 1536 | Exact | 0.5540 | 0.3072 | 0.3185 | 0.555 | 0.575 |
| 1536 | FMA | 0.3215 | 0.3072 | 0.3185 | 0.956 | 0.991 |
| 1536 | TF32 | 0.2488 | 0.2355 | 0.2365 | 0.947 | 0.951 |
| 2048 | Exact | 1.4285 | 0.7190 | 0.8305 | 0.503 | 0.581 |
| 2048 | FMA | 0.7761 | 0.7188 | 0.8305 | 0.926 | 1.070 |
| 2048 | TF32 | 0.6287 | 0.5734 | 0.5755 | 0.912 | 0.915 |
| 3072 | Exact | 4.2908 | 2.4310 | 2.5231 | 0.567 | 0.588 |
| 3072 | FMA | 2.3951 | 2.4699 | 2.5669 | 1.031 | 1.072 |
| 3072 | TF32 | 1.8697 | 1.7172 | 1.7224 | 0.918 | 0.921 |
| 4096 | Exact | 9.5744 | 5.5060 | 5.6373 | 0.575 | 0.589 |
| 4096 | FMA | 5.7580 | 5.6556 | 5.7889 | 0.982 | 1.005 |
| 4096 | TF32 | 4.1523 | 4.1196 | 3.8584 | 0.992 | 0.929 |
| 8192 | Exact | 77.7789 | 41.9092 | 46.1896 | 0.539 | 0.594 |
| 8192 | FMA | 46.3770 | 45.9315 | 49.2554 | 0.990 | 1.062 |
| 8192 | TF32 | 42.3936 | 38.7932 | 38.3252 | 0.915 | 0.904 |

[All kernel arms, ranges and ratios](../../../../benches/results/gpu/validation/integrated/rtx-3090/primary-retry2-kernels.csv) · [raw results](../../../../benches/results/gpu/validation/integrated/rtx-3090/primary-retry2.txt) · [verification](../../../../benches/results/gpu/validation/integrated/rtx-3090/primary-retry2.verification.json)

#### A100 SXM4 — machine 142924 / host 125728

| N | Face | Mapal ms | cuBLAS ms | CUTLASS ms | cuBLAS/Mapal | CUTLASS/Mapal |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | Exact | 0.3953 | 0.1341 | 0.1976 | 0.339 | 0.500 |
| 1024 | FMA | 0.2109 | 0.1341 | 0.1987 | 0.636 | 0.942 |
| 1024 | TF32 | 0.0604 | 0.0348 | 0.0369 | 0.576 | 0.611 |
| 1536 | Exact | 1.2073 | 0.4250 | 0.5755 | 0.352 | 0.477 |
| 1536 | FMA | 0.6021 | 0.4250 | 0.5755 | 0.706 | 0.956 |
| 1536 | TF32 | 0.1352 | 0.0881 | 0.1014 | 0.652 | 0.750 |
| 2048 | Exact | 2.5723 | 1.0752 | 1.2626 | 0.418 | 0.491 |
| 2048 | FMA | 1.2042 | 0.9748 | 1.1428 | 0.810 | 0.949 |
| 2048 | TF32 | 0.2478 | 0.1710 | 0.1751 | 0.690 | 0.707 |
| 3072 | Exact | 7.0031 | 3.1549 | 3.4079 | 0.451 | 0.487 |
| 3072 | FMA | 3.5123 | 3.1539 | 3.4079 | 0.898 | 0.970 |
| 3072 | TF32 | 0.7014 | 0.4669 | 0.5038 | 0.666 | 0.718 |
| 4096 | Exact | 15.2975 | 7.2223 | 7.5561 | 0.472 | 0.494 |
| 4096 | FMA | 7.7763 | 7.2233 | 7.5561 | 0.929 | 0.972 |
| 4096 | TF32 | 1.6466 | 1.0906 | 1.1725 | 0.662 | 0.712 |
| 8192 | Exact | 115.6106 | 57.3460 | 57.3061 | 0.496 | 0.496 |
| 8192 | FMA | 58.5492 | 57.3522 | 57.3071 | 0.980 | 0.979 |
| 8192 | TF32 | 13.5311 | 8.8760 | 9.7249 | 0.656 | 0.719 |

[All kernel arms, ranges and ratios](../../../../benches/results/gpu/validation/integrated/a100/primary-kernels.csv) · [raw results](../../../../benches/results/gpu/validation/integrated/a100/primary.txt) · [verification](../../../../benches/results/gpu/validation/integrated/a100/primary.verification.json)

#### RTX 4090 — machine 146572 / host 264868

| N | Face | Mapal ms | cuBLAS ms | CUTLASS ms | cuBLAS/Mapal | CUTLASS/Mapal |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | Exact | 0.0748 | 0.0481 | 0.0826 | 0.643 | 1.104 |
| 1024 | FMA | 0.0635 | 0.0499 | 0.0860 | 0.786 | 1.354 |
| 1024 | TF32 | 0.0492 | 0.0380 | 0.0584 | 0.772 | 1.187 |
| 1536 | Exact | 0.3973 | 0.1740 | 0.2229 | 0.438 | 0.561 |
| 1536 | FMA | 0.2345 | 0.1741 | 0.2222 | 0.742 | 0.948 |
| 1536 | TF32 | 0.1516 | 0.1311 | 0.1649 | 0.865 | 1.088 |
| 2048 | Exact | 0.4833 | 0.3103 | 0.3000 | 0.642 | 0.621 |
| 2048 | FMA | 0.2866 | 0.3103 | 0.2992 | 1.083 | 1.044 |
| 2048 | TF32 | 0.2427 | 0.2150 | 0.2183 | 0.886 | 0.899 |
| 3072 | Exact | 2.0777 | 1.0506 | 1.0987 | 0.506 | 0.529 |
| 3072 | FMA | 1.4961 | 1.0455 | 1.1008 | 0.699 | 0.736 |
| 3072 | TF32 | 0.8083 | 0.7391 | 0.7444 | 0.914 | 0.921 |
| 4096 | Exact | 3.9782 | 2.4320 | 2.4668 | 0.611 | 0.620 |
| 4096 | FMA | 2.3480 | 2.4780 | 2.4727 | 1.055 | 1.053 |
| 4096 | TF32 | 1.7019 | 1.5512 | 1.5768 | 0.911 | 0.926 |
| 8192 | Exact | 33.6865 | 19.5552 | 20.0909 | 0.581 | 0.596 |
| 8192 | FMA | 20.1687 | 20.6397 | 20.8804 | 1.023 | 1.035 |
| 8192 | TF32 | 13.7912 | 12.9188 | 13.0478 | 0.937 | 0.946 |

[All kernel arms, ranges and ratios](../../../../benches/results/gpu/validation/integrated/rtx-4090/primary-kernels.csv) · [raw results](../../../../benches/results/gpu/validation/integrated/rtx-4090/primary.txt) · [verification](../../../../benches/results/gpu/validation/integrated/rtx-4090/primary.verification.json)

#### RTX 5090 — machine 44652 / host 309734

| N | Face | Mapal ms | cuBLAS ms | CUTLASS ms | cuBLAS/Mapal | CUTLASS/Mapal |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | Exact | 0.0696 | 0.0451 | 0.0959 | 0.648 | 1.378 |
| 1024 | FMA | 0.0573 | 0.0451 | 0.0960 | 0.787 | 1.675 |
| 1024 | TF32 | 0.0389 | 0.0307 | 0.0532 | 0.789 | 1.368 |
| 1536 | Exact | 0.2130 | 0.1209 | 0.1352 | 0.568 | 0.635 |
| 1536 | FMA | 0.1454 | 0.1229 | 0.1370 | 0.845 | 0.942 |
| 1536 | TF32 | 0.0922 | 0.0778 | 0.0778 | 0.844 | 0.844 |
| 2048 | Exact | 0.5059 | 0.2745 | 0.3174 | 0.543 | 0.627 |
| 2048 | FMA | 0.3195 | 0.2745 | 0.3172 | 0.859 | 0.993 |
| 2048 | TF32 | 0.2253 | 0.1966 | 0.2004 | 0.873 | 0.889 |
| 3072 | Exact | 1.7428 | 0.9953 | 1.1138 | 0.571 | 0.639 |
| 3072 | FMA | 1.1937 | 0.9769 | 1.0916 | 0.818 | 0.914 |
| 3072 | TF32 | 0.6676 | 0.5919 | 0.5898 | 0.887 | 0.883 |
| 4096 | Exact | 4.0694 | 2.3706 | 2.5948 | 0.583 | 0.638 |
| 4096 | FMA | 2.6327 | 2.3869 | 2.6171 | 0.907 | 0.994 |
| 4096 | TF32 | 1.5460 | 1.3906 | 1.3701 | 0.899 | 0.886 |
| 8192 | Exact | 34.6417 | 18.1412 | 23.1762 | 0.524 | 0.669 |
| 8192 | FMA | 21.0535 | 19.6209 | 21.1026 | 0.932 | 1.002 |
| 8192 | TF32 | 11.5948 | 10.0475 | 10.3393 | 0.867 | 0.892 |

[All kernel arms, ranges and ratios](../../../../benches/results/gpu/validation/integrated/rtx-5090/primary-retry4-kernels.csv) · [raw results](../../../../benches/results/gpu/validation/integrated/rtx-5090/primary-retry4.txt) · [verification](../../../../benches/results/gpu/validation/integrated/rtx-5090/primary-retry4.verification.json)

#### H100 PCIe — machine 147981 / host 3497

Driver 610.57.04, 350W limit, `sm_90a`. All 812 differential cases and 22 expected traps
pass, with 1,230 probe records, 252 end-to-end records, 70 decode records and 46 kernel
sections. Resource/profiling gates pass; independent rental absence was confirmed at
20:37:43 UTC. The independent checker verified 7,956 numeric fields and 4,140 raw kernel
samples with zero differences. The exports retain all 772 loss rows.

| N | Face | Mapal ms | cuBLAS ms | CUTLASS ms | cuBLAS/Mapal | CUTLASS/Mapal |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | Exact | 0.1889 | 0.0893 | 0.1240 | 0.473 | 0.656 |
| 1024 | FMA | 0.1305 | 0.0893 | 0.1237 | 0.684 | 0.948 |
| 1024 | TF32 | 0.0433 | 0.0239 | 0.0346 | 0.552 | 0.799 |
| 1536 | Exact | 0.5675 | 0.2145 | 0.3547 | 0.378 | 0.625 |
| 1536 | FMA | 0.3361 | 0.2137 | 0.3543 | 0.636 | 1.054 |
| 1536 | TF32 | 0.0991 | 0.0539 | 0.0841 | 0.544 | 0.849 |
| 2048 | Exact | 1.0958 | 0.5304 | 0.6886 | 0.484 | 0.628 |
| 2048 | FMA | 0.6259 | 0.5293 | 0.6892 | 0.846 | 1.101 |
| 2048 | TF32 | 0.1730 | 0.1054 | 0.1652 | 0.609 | 0.955 |
| 3072 | Exact | 3.2424 | 1.7863 | 2.1057 | 0.551 | 0.649 |
| 3072 | FMA | 1.9248 | 1.7863 | 2.1072 | 0.928 | 1.095 |
| 3072 | TF32 | 0.3758 | 0.2820 | 0.4481 | 0.750 | 1.192 |
| 4096 | Exact | 6.3058 | 3.6959 | 4.1564 | 0.586 | 0.659 |
| 4096 | FMA | 3.6650 | 3.7860 | 4.1975 | 1.033 | 1.145 |
| 4096 | TF32 | 0.9248 | 0.5901 | 1.0111 | 0.638 | 1.093 |
| 8192 | Exact | 52.4767 | 27.3895 | 35.2944 | 0.522 | 0.673 |
| 8192 | FMA | 30.8715 | 29.1543 | 35.4436 | 0.944 | 1.148 |
| 8192 | TF32 | 7.1841 | 3.9710 | 8.7019 | 0.553 | 1.211 |

FMA4096 has favorable median ratios against both libraries on this primary host. At 8192,
cuBLAS has the lower FMA and TF32 medians, while Mapal has lower medians than the pinned
CUTLASS kernels. The 8192 FMA Mapal/cuBLAS ranges overlap; no independent H100 confirmation
is claimed. TF32 and Exact retain the numeric-contract limitations above.

[All kernel arms/ranges](../../../../benches/results/gpu/validation/integrated/h100-pcie/primary-retry2-kernels.csv) · [raw results](../../../../benches/results/gpu/validation/integrated/h100-pcie/primary-retry2.txt) · [verification](../../../../benches/results/gpu/validation/integrated/h100-pcie/primary-retry2.verification.json) · [independent export check](../../../../benches/results/gpu/validation/integrated/h100-pcie/primary-retry2-independent-csv-check.json)

**Repeat-control limits:** RTX 5090 Exact8192 measures 34.6417 ms versus 48.3714 ms for identical PTX; cuBLAS stays 18.1412 versus 17.8237 ms. Its normalized null is 0.70363. This cell is unstable and must not support small optimization claims. Other 5090 nulls are 0.99120–1.00167. A100's largest null deviation is 2.87% (TF321024); 3090 spans 0.99756–1.00998 and 4090 spans 0.98072–1.01595. H100's largest null deviation is 3.61% (FMA1024); its FMA4096 normalized null is 1.00107. These controls expose measurement variation rather than compiler effects.

### Several computations in one program

Contract/FMA at n=4096. **The two timing columns have different boundaries:** Mapal is ordinary program iteration including its host work and necessary transfers; the named best tested library is its device-event region. They are shown side by side without an end-to-end speed ratio. Full tables include all sizes, faces, controls and losses.

| GPU | Program | Mapal iteration ms | Library device ms | Library leg |
| --- | --- | ---: | ---: | --- |
| RTX 3090 | mlp | 10.728421 | 11.0602 | lt |
| RTX 3090 | mlp2 | 20.657902 | 20.0243 | lt2 |
| RTX 3090 | diamond | 15.859821 | 16.2652 | streams |
| RTX 3090 | fanout | 10.707641 | 10.9179 | streams |
| RTX 3090 | indep | 10.749881 | 10.5687 | streams |
| A100 SXM4 | mlp | 15.648840 | 14.4415 | lt |
| A100 SXM4 | mlp2 | 31.262463 | 28.8358 | streams |
| A100 SXM4 | diamond | 23.420064 | 21.6678 | lt |
| A100 SXM4 | fanout | 15.610678 | 14.4620 | streams |
| A100 SXM4 | indep | 15.631417 | 14.4476 | lt |
| RTX 4090 | mlp | 4.785901 | 4.9111 | cutlass |
| RTX 4090 | mlp2 | 9.082079 | 9.7676 | lt2 |
| RTX 4090 | diamond | 7.058033 | 7.4376 | cutlass |
| RTX 4090 | fanout | 4.629115 | 4.9111 | streams |
| RTX 4090 | indep | 4.786664 | 4.9133 | cutlass |
| RTX 5090 | mlp | 4.984596 | 4.7493 | lt |
| RTX 5090 | mlp2 | 9.986935 | 9.0037 | lt2 |
| RTX 5090 | diamond | 7.557489 | 7.3311 | streams |
| RTX 5090 | fanout | 5.031694 | 4.7494 | streams |
| RTX 5090 | indep | 5.024872 | 4.7906 | streams |
| H100 PCIe | mlp | 7.311006 | 7.4878 | lt |
| H100 PCIe | mlp2 | 14.634645 | 14.0062 | streams |
| H100 PCIe | diamond | 10.922254 | 10.7902 | streams |
| H100 PCIe | fanout | 7.304007 | 7.5548 | streams |
| H100 PCIe | indep | 7.337121 | 6.9775 | streams |

For final RTX 5090 MLP2 Contract4096, the shipped stream-OFF/arena-ON iteration is 9.986935 ms versus 11.862434 ms with streams ON: 15.8% lower. Actual process allocations are 1 versus 14. This is a coupled placement choice, not an allocation-only causal result. Both show zero H2D bytes and 16 D2H bytes; runtime-traced kernels are separate diagnostics.

The 4090 all-rungs-enabled control has lower medians for several epilogue/residual cases, but the shipped distributions contain both fast and slow observations. Contract residual4096 is 3.251466 ms [2.488205, 3.259802] shipped versus 2.511358 [2.504265, 2.521047] for that control. A100 Contract softmax_rows2048 is 10.542956 [8.048505, 11.532130] versus 7.825156 [7.641904, 12.581315]. These overlapping ranges do not prove a uniform regression or identify a single rung. Preserve the controls and wait for independent evidence before changing selection.

### Retention, residency and transfers in recurrent decode

Contract, 1024 steps × width256. These are **ordinary whole-process wall medians**, including startup. The combined control `off-retain-delta+resident` disables `retain-delta` and base device residency; it is not the pure both-retention-OFF control required for the new composition study. Its large allocation/copy differences therefore cannot be attributed solely to inferred retention.

| GPU | Shipped wall ms | Combined control wall ms | Control/shipped | Shipped/control allocations |
| --- | ---: | ---: | ---: | ---: |
| RTX 3090 | 578 | 2310 | 3.997 | 8 / 12288 |
| A100 SXM4 | 627 | 2741 | 4.372 | 8 / 12288 |
| RTX 4090 | 564 | 2680 | 4.752 | 8 / 12288 |
| RTX 5090 | 450 | 1292 | 2.871 | 8 / 12288 |
| H100 PCIe | 973 | 2382 | 2.448 | 8 / 12288 |

Shipped traces transfer 3,144,704 bytes H2D and 1,048,576 bytes D2H per process. Turning only delta-H2D OFF raises H2D to 1,074,790,400 bytes, with the same eight observed allocations. The combined residency/retention control reaches 4,837,081,088 H2D and 2,152,726,528 D2H bytes with 12,288 allocations. These are measured traced counts; they are not measured peak memory or isolated latency contributions.

On H100, enabling only resident retention lowers the traced kernel median from 161.283
to 142.827 ms, while ordinary process-wall medians are 973 ms shipped versus 1,007 ms enabled.
The wall ranges overlap (896–1,501 ms versus 933–1,074 ms), so this final run establishes
no ordinary wall-time gain from enabling it. This preserves the rationale for the
conservative OFF selection without treating tracing as untraced performance.

- RTX 3090: [programs](../../../../benches/results/gpu/validation/integrated/rtx-3090/primary-retry2-programs.csv), [libraries](../../../../benches/results/gpu/validation/integrated/rtx-3090/primary-retry2-libraries.csv), [all loss records](../../../../benches/results/gpu/validation/integrated/rtx-3090/primary-retry2-losses.csv), [nulls](../../../../benches/results/gpu/validation/integrated/rtx-3090/primary-retry2-nulls.csv).
- A100 SXM4: [programs](../../../../benches/results/gpu/validation/integrated/a100/primary-programs.csv), [libraries](../../../../benches/results/gpu/validation/integrated/a100/primary-libraries.csv), [all loss records](../../../../benches/results/gpu/validation/integrated/a100/primary-losses.csv), [nulls](../../../../benches/results/gpu/validation/integrated/a100/primary-nulls.csv).
- RTX 4090: [programs](../../../../benches/results/gpu/validation/integrated/rtx-4090/primary-programs.csv), [libraries](../../../../benches/results/gpu/validation/integrated/rtx-4090/primary-libraries.csv), [all loss records](../../../../benches/results/gpu/validation/integrated/rtx-4090/primary-losses.csv), [nulls](../../../../benches/results/gpu/validation/integrated/rtx-4090/primary-nulls.csv).
- RTX 5090: [programs](../../../../benches/results/gpu/validation/integrated/rtx-5090/primary-retry4-programs.csv), [libraries](../../../../benches/results/gpu/validation/integrated/rtx-5090/primary-retry4-libraries.csv), [all loss records](../../../../benches/results/gpu/validation/integrated/rtx-5090/primary-retry4-losses.csv), [nulls](../../../../benches/results/gpu/validation/integrated/rtx-5090/primary-retry4-nulls.csv).

- H100 PCIe: [programs](../../../../benches/results/gpu/validation/integrated/h100-pcie/primary-retry2-programs.csv), [libraries](../../../../benches/results/gpu/validation/integrated/h100-pcie/primary-retry2-libraries.csv), [all loss records](../../../../benches/results/gpu/validation/integrated/h100-pcie/primary-retry2-losses.csv), [nulls](../../../../benches/results/gpu/validation/integrated/h100-pcie/primary-retry2-nulls.csv).

The new [16K/rectangular and composed-program extension](../workloads/large-matrices-and-mlp-validation.md) has separate inputs and gates. Its PyTorch and new composed GPU measurements remain pending; no comparison above substitutes for them.

## Independent RTX 4090 confirmation

Accepted tag `rtx-4090-confirm-retry3`, machine3358 / host1276, driver580.178.04,
300W power limit. All727 differential values and20 expected traps pass, together
with complete timing/status/resource/profiling gates; rental absence was independently
confirmed at20:02:34 UTC. Its physical machine and host differ from the primary.
The independent parser checked7,494 numeric fields and all42×3×30 raw kernel samples
with zero differences. All620 loss rows remain in the exports. The largest
library-normalized identical-PTX null deviation is1.667%.

CUDA-event medians in milliseconds; this table is one host. These results do not
establish that absolute timings transfer between different hosts or power settings.

| N | Face | Mapal ms | cuBLAS ms | CUTLASS ms | cuBLAS/Mapal | CUTLASS/Mapal |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | FMA | 0.0829 | 0.0666 | 0.1065 | 0.803 | 1.285 |
| 1024 | Exact | 0.1034 | 0.0655 | 0.1065 | 0.633 | 1.030 |
| 1024 | TF32 | 0.0554 | 0.0440 | 0.0748 | 0.794 | 1.350 |
| 1536 | FMA | 0.3029 | 0.2150 | 0.2908 | 0.710 | 0.960 |
| 1536 | Exact | 0.5130 | 0.2150 | 0.2908 | 0.419 | 0.567 |
| 1536 | TF32 | 0.1946 | 0.1659 | 0.2140 | 0.853 | 1.100 |
| 2048 | FMA | 0.3645 | 0.4004 | 0.3871 | 1.098 | 1.062 |
| 2048 | Exact | 0.6267 | 0.4004 | 0.3871 | 0.639 | 0.618 |
| 2048 | TF32 | 0.3154 | 0.2806 | 0.2847 | 0.890 | 0.903 |
| 3072 | FMA | 1.5739 | 1.3588 | 1.3998 | 0.863 | 0.889 |
| 3072 | Exact | 2.7218 | 1.3566 | 1.3989 | 0.498 | 0.514 |
| 3072 | TF32 | 1.1162 | 1.0230 | 1.0281 | 0.917 | 0.921 |
| 4096 | FMA | 3.3423 | 3.5205 | 3.5656 | 1.053 | 1.067 |
| 4096 | Exact | 5.4118 | 3.3085 | 3.3270 | 0.611 | 0.615 |
| 4096 | TF32 | 2.3491 | 2.1535 | 2.1801 | 0.917 | 0.928 |
| 8192 | FMA | 33.4305 | 33.4234 | 31.1501 | 1.000 | 0.932 |
| 8192 | Exact | 54.3201 | 27.1688 | 28.5440 | 0.500 | 0.525 |
| 8192 | TF32 | 18.4361 | 16.9964 | 17.2820 | 0.922 | 0.937 |

FMA2048 and4096 have favorable median ratios against both libraries on both physical
hosts. FMA8192 does **not** repeat the primary's win: cuBLAS is essentially tied by
median, while CUTLASS has the lower median. Its ranges are broad and overlap:
Mapal22.365–36.464ms, cuBLAS23.874–35.877ms, CUTLASS29.280–34.077ms. Preserve that
variability; neither the power limit nor any static resource count alone proves a cause.
Exact and TF32 remain separate contracts; the single-kernel library errors are diagnostics.

[All kernel arms/ranges](../../../../benches/results/gpu/validation/integrated/rtx-4090/confirm-retry3-kernels.csv) ·
[all program rows](../../../../benches/results/gpu/validation/integrated/rtx-4090/confirm-retry3-programs.csv) ·
[all library rows](../../../../benches/results/gpu/validation/integrated/rtx-4090/confirm-retry3-libraries.csv) ·
[all losses](../../../../benches/results/gpu/validation/integrated/rtx-4090/confirm-retry3-losses.csv) ·
[raw](../../../../benches/results/gpu/validation/integrated/rtx-4090/confirm-retry3.txt) ·
[verification](../../../../benches/results/gpu/validation/integrated/rtx-4090/confirm-retry3.verification.json) ·
[independent export check](../../../../benches/results/gpu/validation/integrated/rtx-4090/confirm-retry3-independent-csv-check.json)

## Independent RTX 5090 confirmation

Accepted tag `rtx-5090-confirm-retry3`, machine 145957 / host 617499, driver 590.48.01,
546 W power limit. All 812 differential values and 22 expected traps pass, with 1,240 probe,
252 end-to-end, 70 decode and 48 kernel sections, plus complete status/resource/profiling
gates. The independent rental-absence receipt is 20:55:19 UTC. The physical machine and
host differ from the primary. Independent checking covers 8,050 numeric fields and 4,320
raw kernel samples with zero differences; all 794 loss rows remain in the exports.
The largest library-normalized identical-PTX null deviation is 0.6573%.

CUDA-event medians in milliseconds, measured on this host:

| N | Face | Mapal ms | cuBLAS ms | CUTLASS ms | cuBLAS/Mapal | CUTLASS/Mapal |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | FMA | 0.0860 | 0.0676 | 0.1147 | 0.786 | 1.334 |
| 1024 | Exact | 0.1044 | 0.0676 | 0.1147 | 0.648 | 1.099 |
| 1024 | TF32 | 0.0512 | 0.0430 | 0.0737 | 0.840 | 1.439 |
| 1536 | FMA | 0.1946 | 0.1700 | 0.1741 | 0.874 | 0.895 |
| 1536 | Exact | 0.2806 | 0.1720 | 0.1741 | 0.613 | 0.620 |
| 1536 | TF32 | 0.1311 | 0.1106 | 0.1106 | 0.844 | 0.844 |
| 2048 | FMA | 0.4116 | 0.3482 | 0.4055 | 0.846 | 0.985 |
| 2048 | Exact | 0.6738 | 0.3502 | 0.4055 | 0.520 | 0.602 |
| 2048 | TF32 | 0.3174 | 0.2765 | 0.2806 | 0.871 | 0.884 |
| 3072 | FMA | 1.2677 | 1.0936 | 1.1796 | 0.863 | 0.931 |
| 3072 | Exact | 4.3643 | 1.0936 | 1.1796 | 0.251 | 0.270 |
| 3072 | TF32 | 0.9093 | 0.8192 | 0.8212 | 0.901 | 0.903 |
| 4096 | FMA | 5.0258 | 4.9490 | 5.0678 | 0.985 | 1.008 |
| 4096 | Exact | 9.3635 | 4.9920 | 5.0924 | 0.533 | 0.544 |
| 4096 | TF32 | 4.4472 | 1.8883 | 1.8883 | 0.425 | 0.425 |
| 8192 | FMA | 40.8156 | 40.3558 | 41.1587 | 0.989 | 1.008 |
| 8192 | Exact | 72.2893 | 40.6098 | 41.4167 | 0.562 | 0.573 |
| 8192 | TF32 | 31.8362 | 28.5757 | 28.2849 | 0.898 | 0.888 |

The 4K TF32 library ratio is materially worse than the primary: Mapal/cuBLAS is
4.4472/1.8883 ms here versus 1.5460/1.3906 ms there. The absolute times and hardware/software
conditions differ. The [saved resource comparison](../../../../benches/results/gpu/operations/confirmation/rtx-5090-confirm-driver-resource-evidence.md)
finds identical PTX and all 18 reported kernel-resource tuples. Actual TF32 4096 profiler
routes/resources also match. Driver-JIT machine code and hardware counters are missing,
so no measured cause is established and these observations justify no compiler fix. FMA 4K and 8K remain near the tested library medians; these small differences
are not a general claim of superiority.

All eight changed stream comparisons favor the shipped stream-OFF/arena-ON arm by median.
Six have disjoint min–max ranges; independent 4096 Exact and MLP2 4096 Contract have
faster ON observations and overlapping ranges. The table keeps that counterevidence.
These are ordinary program-iteration milliseconds, with allocation counts from separate
traces. The intervention changes scheduling and arena use together.

| Program | Face | Shipped OFF median [min–max], ms | Streams ON median [min–max], ms | ON/OFF | Allocations OFF/ON |
| --- | --- | ---: | ---: | ---: | ---: |
| pt_indep_4096 | FMA | 12.446224 [12.432607–12.448889] | 13.572797 [13.562175–13.587785] | 1.090515 | 1 / 7 |
| pt_indep_4096 | Exact | 21.088984 [21.084486–21.119995] | 22.197332 [14.928755–22.244674] | 1.052556 | 1 / 7 |
| pt_mlp2_4096 | FMA | 24.978135 [24.964078–25.040437] | 27.069233 [18.836103–27.131685] | 1.083717 | 1 / 14 |
| pt_mlp2_4096 | Exact | 39.932475 [39.909650–40.014996] | 42.010317 [41.979890–42.042761] | 1.052034 | 1 / 14 |
| pt_twin_4096 | FMA | 12.404873 [12.397799–12.417658] | 12.752519 [12.716188–12.762578] | 1.028025 | 1 / 6 |
| pt_twin_4096 | Exact | 21.022909 [21.007358–21.042687] | 21.151780 [21.056063–21.174102] | 1.006130 | 1 / 6 |
| pt_twin_512 | FMA | 2.365251 [2.343709–2.367185] | 2.460587 [2.452051–2.472500] | 1.040307 | 1 / 6 |
| pt_twin_512 | Exact | 2.368468 [2.358057–2.371894] | 2.469234 [2.456068–2.474665] | 1.042545 | 1 / 6 |

Contract decode 1024×256 has ordinary process-wall medians 5,046 ms shipped versus 14,480 ms
for the combined residency/retention-disabled control (2.870×), with 8 versus 12,288 traced
allocations. The byte counts match the primary: 3,144,704/1,048,576 H2D/D2H shipped and
4,837,081,088/2,152,726,528 for the combined control. The shipped wall range is 4,912–5,104 ms;
the control is 14,412–14,497 ms. These absolute times are much larger than the primary's
450/1,292 ms; the similar ratio does not explain the host difference. Traced shipped kernel
time is 9,098.911 ms and is a separate serialized execution, not a component to subtract
from the ordinary 5,046 ms wall time.

[All kernel arms/ranges](../../../../benches/results/gpu/validation/integrated/rtx-5090/confirm-retry3-kernels.csv) ·
[all program rows](../../../../benches/results/gpu/validation/integrated/rtx-5090/confirm-retry3-programs.csv) ·
[all library rows](../../../../benches/results/gpu/validation/integrated/rtx-5090/confirm-retry3-libraries.csv) ·
[all losses](../../../../benches/results/gpu/validation/integrated/rtx-5090/confirm-retry3-losses.csv) ·
[nulls](../../../../benches/results/gpu/validation/integrated/rtx-5090/confirm-retry3-nulls.csv) ·
[raw](../../../../benches/results/gpu/validation/integrated/rtx-5090/confirm-retry3.txt) ·
[verification](../../../../benches/results/gpu/validation/integrated/rtx-5090/confirm-retry3.verification.json) ·
[independent export check](../../../../benches/results/gpu/validation/integrated/rtx-5090/confirm-retry3-independent-csv-check.json)

## What the compiler already knows

These mechanisms derive facts from shapes and dataflow. They do not recognize benchmark
names. The implementation links below refer to frozen source `b4152c5`; the
[component map](../../../components/backend-nvptx/IMPLEMENTATION.md) owns the complete realization.

| Mechanism | Derived fact and realization | Current policy and evidence |
| --- | --- | --- |
| Device residency | Producers, device readers, host observations and carried values; `resident.rs:FnEmit::resident_plan` | Available on all five profiles. Selected 5090 MLP2, independent join, diamond and twin move no H2D data and only printed corners D2H. This establishes placement, not an isolated residency speedup. |
| Allocation arena | Static sizes and nonescaping lifetimes; `algo.rs:CategoryIr::last_use_plan` → `resident.rs:FnEmit::arena` / `color` / `dalloc` | Available on all five, currently disabled when stream chains exist. Straight-line allocation reuse is implemented; generic cross-stream pooling is not. |
| Row retention | Changed rows, row-local readers, invariant other operand and unchanged output; `retain.rs:retained_sites` | Resident retention ON for 3090/A100, OFF for 4090/H100/5090. Fresh H100 kernel savings did not establish a stable untraced program gain. |
| Delta upload | Proven host-written row span of a carried array; `resident.rs:emit_delta_h2d` | Available on all five. First upload is whole; later uploads cover the changed span. The accepted primary decode traces above measure 3,144,704 H2D bytes versus 1,074,790,400 with only delta-H2D disabled, at eight allocations in both arms. |
| L2 access window | Output reuse by later device consumers and capacity fit; `resident.rs:FnEmit::l2_window` | OFF on all five. H100 selection iteration ranges overlap; no accepted program gain. |
| Raster traversal | Affine panel bytes, grid geometry and L2 capacity; `profile.rs:Gpu::raster` | Derived gate restored. 5090 TF32/8192 has favorable approximately 1.5% kernel medians, but overlapping ranges and no established whole-program gain; the rejected override's counterevidence remains in the selection report. |

Paths in the table are relative to `crates/backends/llvm/src/func`, except `algo.rs`
(`crates/mapal-ir/src`) and `profile.rs` (`crates/backends/llvm/src`). Row retention retains
derived values; residency avoids transmissions; the arena reuses storage; L2 policies
influence hardware caching. Their benefits must be measured separately or identified as a
combined intervention.

`LastUsePlan` already derives positions, uses, escapes and carried-state facts, while
`CategoryIr::path_plan` derives dependency/task structure. The LLVM backend's emitter
consumes those facts to place operations and storage; they are not device lifetime or
completion annotations in the generated host LLVM. The
[call-effects audit](../../../../benches/results/gpu/host-dispatch/optimization-audit/host-gpu-call-effects-audit.md)
records that boundary and the existing helper/device attributes precisely.
No user lifetime hint is required for the supported static cases. The missing distinction
is logical death versus asynchronous completion: another GPU operation may still be
reading a buffer after its host launch call returns. `FnEmit::resident_plan` therefore
disables its linear arena when the stream-chain map is nonempty. This is a deliberate
conservative backend restriction, not absent lifetime information in the IR.

## Allocation and execution: observed selection results

The accepted 5090 selection compares streams ON against OFF; OFF also restores the arena.
These are coupled placement measurements from pre-final raster settings. They do not
isolate allocation from scheduling. Iteration medians are untraced; allocation counts come
from separate traced executions.

| Contract, n=4096 | Actual allocations ON→OFF | A2 iteration ON/OFF ms | B iteration ON/OFF ms |
| --- | ---: | ---: | ---: |
| MLP2 | 14→1 | 11.490 / 8.790 | 12.068 / 8.472 |
| Independent join | 7→1 | 5.780 / 4.433 | 6.046 / 4.282 |
| Diamond | 8→1 | 8.291 / 6.625 | 8.551 / 6.400 |
| Twin | 6→1 | 5.110 / 4.395 | 5.229 / 4.245 |
| Single-chain MLP | Already 1; no changed stream arm | 4.412 shipped | 4.263 shipped |

The single OFF-arm arena allocation occurs before the program timer; process wall still
pays it. MLP2 ON has eight actual allocations and twelve frees inside the measured clock
interval, with fourteen allocations across the process. Both arms perform four tile
launches, eight generators, no H2D transfer and sixteen D2H bytes. Full values, source lines
and hashes are in the [selection stream CSV](../../../../benches/results/gpu/validation/architecture-selection/selection-rtx-5090-streams.csv).

Ordinary Nsight Systems MLP2 Contract4096 timelines show **zero cross-stream tile-kernel
overlap in the two profiled ON runs**. Four tile intervals are recorded in each archive's
`prof/tl_pt_mlp2_4096_c.csv`, rows 40–43. The first stream finishes 8,192 ns (A2) / 7,552 ns
(B) before the second begins. This observation does not measure SM utilization or establish
that every workload lacks overlap. Runtime-traced kernel totals serialize launches and
cannot independently answer the overlap question.

The timed allocation/free API calls consume elapsed 0.644/2.222 ms on A2 and 1.098/2.780 ms
on B. API elapsed time can include GPU waits, driver work and scheduling. It must not be
added wholesale to kernel duration or described as entirely removable CPU allocation cost.
The chronology establishes allocation/free activity inside the clock and absent observed
tile overlap, but does not isolate every cause of the ordinary iteration difference.

## Authorized next phase

After final frozen validation, apply dependency, lifetime and placement facts across
algorithms through OpenSpec. The planned first change is
[eliminate-unmaterialized-gpu-allocations](../../../../openspec/changes/eliminate-unmaterialized-gpu-allocations/proposal.md):
stream setup should use the same existing output-storage facts as fused kernel emission,
omitting only proved unused raw output allocations. The arena guard and live allocation
positions remain unchanged. However, nonzero frees currently synchronize outstanding
chains, so removing unused allocations can also remove waits. The
[frozen ordering investigation](../memory/ordering-investigation.md) makes completion ordering a
prerequisite: establish each surviving consumer's producer dependency before eliminating
storage. Some suspected cases were refuted; GPU consequences of the remaining missing
emitted edges are still untested. Predicted allocation reductions are not measured
performance results.

Pooling concurrently live buffers into disjoint slices remains a separate candidate.
Disjointness proves nonaliasing, but a function-wide pool can increase peak reserved bytes;
that needs its own capacity/fallback decision. Any reuse must prove completion ordering of
all old consumers before the new writer. Guards, carries, aliases and host observations
remain proof boundaries. These are planned candidates, not built capabilities.

Explicit CUDA graph construction is a later candidate: derive operation topology from IR,
bind buffers and kernel handles at runtime, instantiate once and replay. This can remove
runtime stream capture; it does not remove runtime binding/instantiation or make ordinary
separate launch calls a single submission. Graph submission and kernel fusion are distinct
transformations. The NVIDIA API supports both explicit graph construction and capture
([CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)).
Profitability must be measured across workload families, sizes and hardware independently
of correctness. Numerical precision policy remains unchanged.
