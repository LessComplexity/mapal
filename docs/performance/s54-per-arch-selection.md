# S54 per-architecture selection

Selection recovery, 2026-10-21. **Final five-profile validation and both independent confirmations pass.**
The LLVM GPU implementation is mapped in
[backend-nvptx](../components/backend-nvptx/IMPLEMENTATION.md).
The [S54 handoff](../sessions/2026-10-21-s54-fusion-placement.md) records the disputed,
unverified selections that this work resolves. Historical `integrated-*-prefix.txt`
runs predate the arena repair and contain failures; they are not current acceptance evidence.

## Acceptance rule and measurement scope

The disputed S54 selections had never completed integrated acceptance. Apply the original
[plan §3](../components/backend-nvptx/plans/plan-s54-fusion-placement.md#3-the-test-matrix):
a program-level ON choice needs a repeatable gain, with independent confirmation for small
results. The recovery OpenSpec also declared before measurement that overlapping effects do
not justify a disputed override. Raster is assessed as a kernel realization; a kernel gain
is not a whole-program claim, and the same conservative overlap rule applies. This is initial acceptance of those disputed choices, not `rungs.py`'s policy of
holding an already accepted default. Other existing selections remain subject to final
integrated validation; this recovery does not restart every historical tuning experiment.

- Streams: use untraced program iteration time. Runtime tracing synchronizes launches and
  removes the overlap being measured.
- Retain-resident: keep untraced process wall separate from traced kernels and traced wall
  minus load time. Keep Exact and Contract separate; pooling can conceal a sign change.
- L2 window: distinguish iteration effects from kernel-only effects and policy-call overhead.
- Raster: compare the derived L2 gate, forced grouping and raster-disabled controls. Kernel
  A/B ratios use each arm's own cuBLAS reference and the same session's null control.
- Min/median/max ranges are distributions, not confidence intervals. Overlapping ranges do
  not prove equality; a favorable median alone does not establish a repeatable improvement.

`on` means the shipped table selection. `off-arch-rungs` bypasses the table and enables all
rungs, including S54 features. The separate placement-disabled composite disables epilogue,
devmap, cell-law, rowfold/layout, streams, pinned, PDL, arena, retain-resident, delta-h2d,
lane-stride and L2 window. It leaves raster independent. Neither control is an S52 binary.

## H100 PCIe: retention and L2 window

Fresh selection evidence: [raw result](../../benches/results-s54-gpu/s56-selection-h100-pcie.txt),
[provenance](../../benches/results-s54-gpu/s56-selection-h100-pcie.json),
[verification](../../benches/results-s54-gpu/s56-selection-h100-pcie.verification.json).
Source `4c93c8a`; CUTLASS `91843b71d90737e0a9b86f7d9df213cff5a3526f`;
Vast machine **81035**, host **415974**, instance **57081803** (destroyed and independently
confirmed absent). This repeats the earlier retention-confirm host; it is not a third
independent physical host.

The fresh bundle passed 82 value cases, 14 expected traps, 630 probe measurements,
60 decode measurements, 26 library value gates and 45 kernel sections. Resource and nsys
artifacts are retained. Selection bundles predate the additional timed-process status guard:
complete output/trace inventories passed, but missing ordinary/traced process status fields
cannot be reconstructed. Final integrated bundles record and enforce those statuses.

### Retain-resident: OFF for final acceptance

Five interleaved process repetitions per arm and face. OFF/ON >1 favors retention.

| Face | ON wall median [min–max], ms | OFF wall median [min–max], ms | OFF/ON wall | OFF/ON traced kernels |
|---|---:|---:|---:|---:|
| Exact | 819 [791–922] | 842 [824–1036] | 1.028 | 1.219 |
| Contract | 870 [822–1120] | 878 [854–886] | 1.009 | 1.130 |

Both wall ranges overlap. Historical untraced wall ratios were Exact/Contract
0.971/1.143 in the original run and 0.705/1.016 in its confirmation. The fresh measurement
therefore does not establish a stable program-level gain across runs or an independent new
confirmation. Retention does reduce traced kernel time by about **18.0% Exact / 11.5%
Contract**; that counterevidence is real and is not promoted into a wall-time claim.

The final selection disables retention on both H100 faces. **OFF is not proved faster.**
This is the conservative result under initial acceptance. Holding ON would be defensible
under a different policy—preserve an already accepted default until a stable loss appears—but
that would assume the disputed S54 choice had already passed acceptance. Forced-ON controls
remain available in final validation.

### L2 window: OFF on all H100 faces

Fifteen interleaved probe repetitions per arm. The table previously enabled Contract and
kept Exact disabled. The current Contract comparisons are:

| Probe | ON iteration ms | OFF iteration ms | OFF/ON iteration | OFF/ON traced kernels |
|---|---:|---:|---:|---:|
| diamond 2048 | 1.969309 | 1.934879 | 0.983 | 0.998 |
| mlp2 2048 | 2.643185 | 2.589975 | 0.980 | 1.007 |
| mlp 2048 | 1.357854 | 1.291543 | 0.951 | 1.007 |

All min–max ranges overlap, but no current iteration median favors ON. Prior run3 iteration
ratios were 0.999/0.994/1.002 for these probes; its 1.2–1.5% kernel savings did not establish
iteration gains. The current kernel effect is mixed and below 1%. At 4096 the window arms
are byte-identical and pruned, so there is no distinct measured effect.

Disable the unsupported Contract cache-policy selection and retain Exact OFF. This removes
an unaccepted override; it is not a claim of a proved cross-host speedup. The H100 selection
run's single-kernel null controls also include a **5.36% TF32/4096 outlier**, so percent-scale
kernel differences from that session require particular caution. That kernel null is a
separate instrument from the interleaved program comparisons above.

## RTX 5090: streams OFF; keep the derived raster gate

Both fresh selection runs passed their complete timing, value, resource and nsys inventories.
They use the same frozen bundle on independent physical hosts:

| Run | Machine / host | Driver | Power limit | Evidence |
|---|---|---|---:|---|
| A2 | 55053 / 253861 | 570.211.01 | 500 W | [raw](../../benches/results-s54-gpu/s56-selection-rtx-5090-a2.txt), [provenance](../../benches/results-s54-gpu/s56-selection-rtx-5090-a2.json), [verification](../../benches/results-s54-gpu/s56-selection-rtx-5090-a2.verification.json) |
| B | 55265 / 33035 | 595.104.02 | 575 W | [raw](../../benches/results-s54-gpu/s56-selection-rtx-5090-b.txt), [provenance](../../benches/results-s54-gpu/s56-selection-rtx-5090-b.json), [verification](../../benches/results-s54-gpu/s56-selection-rtx-5090-b.verification.json) |

Instances 57081969 (A2) and 57081794 (B) were destroyed and independently confirmed absent.
The hosts differ in driver and power limit; do not pool their raw times. Each run passed
82 value cases, 14 expected traps, 2,670 probe measurements and traces, 70 decode measurements,
74 library value gates and 46 kernel sections. Both frozen runners omit some timed-process
exit fields, disclosed by verification; twin has an explicit unsupported-library exception.
Neither limitation is hidden by the successful profiling gates.

### Streams: OFF on both faces

Fifteen interleaved repetitions per changed arm, program and face. OFF/ON below one favors
OFF in ordinary program iteration time. Every one of the 32 host × changed-cell comparisons
has disjoint ON/OFF iteration min–max ranges. No measured size meets initial ON acceptance.

| Probe | Exact OFF/ON A2 | Exact OFF/ON B | Contract OFF/ON A2 | Contract OFF/ON B |
|---|---:|---:|---:|---:|
| diamond 512 | 0.130781 | 0.086693 | 0.125684 | 0.089063 |
| indep 512 | 0.103200 | 0.089487 | 0.098717 | 0.081203 |
| mlp2 512 | 0.123868 | 0.105662 | 0.118930 | 0.094394 |
| twin 512 | 0.144278 | 0.111254 | 0.142792 | 0.105014 |
| diamond 4096 | 0.865054 | 0.826369 | 0.799113 | 0.748450 |
| indep 4096 | 0.841286 | 0.794217 | 0.767001 | 0.708214 |
| mlp2 4096 | 0.836330 | 0.790813 | 0.764958 | 0.702044 |
| twin 4096 | 0.932312 | 0.901150 | 0.860095 | 0.811853 |

For example, B's twin4096 Exact median is OFF **6.843751 ms** versus ON **7.594467 ms**;
Contract is **4.245028 / 5.228815 ms**. B's MLP2_4096 is Exact **13.635961 / 17.242973 ms**,
Contract **8.472291 / 12.068032 ms**. The stream controls at 1024 and 2048 are byte-identical
and pruned, not independent measured no-effect results.

Most 4096 loss is outside traced kernels: OFF/ON kernel ratios remain approximately
0.990–0.999 while ordinary iteration changes materially. Tracing serializes launches, so it
cannot establish stream overlap. This is an **integrated placement-selection A/B**, not an
isolated stream-scheduling experiment: `FnEmit::resident_plan` creates a device arena only
when the stream-chain map is empty ([implementation map](../components/backend-nvptx/IMPLEMENTATION.md)).
Disabling streams therefore also re-enables the arena on these programs. Both hosts' traces
show MLP2 allocations falling from 14 to 1 and twin from 6 to 1. MLP2 retains four launches,
eight generators, zero H2D operations and 16 output bytes. Frozen MLP2 Contract host IR has ten allocation
and twelve free call sites inside the ON timer interval, versus no free calls with OFF; the
sole real OFF arena allocation occurs before the timer, although process wall still pays it.
These static call sites differ from actual whole-process allocation counts. The allocation
counts are measured; attributing the entire elapsed-time difference to allocation or scheduling
is unproven.

Right tails are retained: A2 twin512 Contract ON is
0.314514 ms median with a 4.526447 ms maximum; its ON minimum still exceeds the OFF maximum.
Across all controls, B's indep512 Contract table-bypass control reaches 15.655731 ms against
a 0.773844 ms median. No samples are removed to obtain the table above.

Disable the disputed RTX 5090 streams choice for both faces. These two runs establish an
ordinary-iteration loss with the ON placement on the changed probes. The final selected
binary subsequently passed the integrated matrix. Other profiles' streams choices remain held.

### Raster/L2 gate: remove the forced-grouping override

The shipped selection bundle disables the L2 gate for Contract, which also covers TF32;
thus it forces grouping where the derived gate would retain row-major order. The final
selection removes that RTX 5090 `l2-gate` OFF row and keeps the derived gate and raster
mechanism. It does not disable raster globally or change other profiles' selections.

The table shows `(raster disabled / shipped grouping)`, normalized by each arm's own cuBLAS
leg and corrected with the same face/size's session null. Above one favors shipped grouping.

| Face | Size | A2 | B |
|---|---:|---:|---:|
| Contract | 4096 | 1.017748 | 1.001002 |
| Contract | 8192 | 0.994967 | 0.998933 |
| TF32 | 4096 | 0.997551 | 1.000000 |
| TF32 | 8192 | 1.015324 | 1.015134 |

Contract FMA does not show a >1% grouping win in both sessions. Exact retains its derived
gate; forcing grouping gives no repeatable >1% improvement. Program raster/gate control
ratios remain within 1% in both runs. Every changed single-kernel raster/gate min–max pair
overlaps, including TF32 at 8192.

The repeated **TF32/8192 ~1.5% normalized median gain is counterevidence** to removing all
forced grouping: its own null ratios are 0.998843 (A2) and 1.002681 (B). A median-only reading
of plan §3 could support a narrower Filled/TF32 row. Under the declared conservative rule,
overlapping distributions and no demonstrated program gain do not establish this override.
The final candidate therefore restores the derived gate on all RTX 5090 faces. **The gate
is not proved faster.** The largest absolute null deviations over all faces/sizes are
1.5573% (A2) and 1.6464% (B); these remain per-session diagnostics, not transferable error bars.

### Exhaustive compact evidence

The compact tables retain every measured cell, not only gains:

- [Streams](../../benches/results-s54-gpu/s56-selection-rtx-5090-streams.csv): all 32 face/host comparisons, ordinary iteration and wall distributions, traced kernels and raw source lines.
- [Programs](../../benches/results-s54-gpu/s56-selection-rtx-5090-programs.csv): all 384 probe/decode arm groups and best probe-library comparisons.
- [Libraries](../../benches/results-s54-gpu/s56-selection-rtx-5090-libraries.csv): all 152 library timing cells, keeping decode host-loop wall and device events separate.
- [Kernels](../../benches/results-s54-gpu/s56-selection-rtx-5090-kernels.csv): all 92 kernel cases, cuBLAS/CUTLASS distributions, controls and own-session nulls.
- [Losses](../../benches/results-s54-gpu/s56-selection-rtx-5090-losses.csv): every summarized ratio below one, without a tolerance or significance filter; time domains are named explicitly.

Each row binds raw result and verification SHA256 and points to raw source lines. Raw samples
remain in the result text; full sample summaries and profiling archives stay in recovery
storage. MLP2_4096 illustrates the boundary: best Lt2 device event is 8.5469/7.8670 ms on
A2/B, while OFF-streams Contract iteration is 8.789568/8.472291 ms. Library device events
exclude Mapal's host/transmission work and use device-built inputs. This is a cross-scope
diagnostic, not a comparable whole-program library wall-time gap. Decode process wall,
traced kernels and library host-loop/device-event timing likewise remain separate.

## Final integrated validation passed

The frozen `b4152c5` bundles pass on RTX 3090, A100 SXM4, RTX 4090, H100 PCIe and
RTX 5090, with independent final RTX 4090/5090 confirmations. All seven runs have
complete numerical, ordinary/traced-status, timing, library, resource and profiling
inventories, independent machine identity and rental-absence receipts. The
[integrated report](s54-gpu-fusion.md) publishes every shipped kernel row and full
loss/control CSVs. Independent checking covers 54,121 numeric fields and 27,900 raw
kernel samples with zero differences.

The final 5090 confirmation favors streams OFF by median in all eight changed cells;
six ranges are disjoint and two overlap. This is additional evidence, separate from
the 32 disjoint comparisons in the earlier selection experiment. Its absolute timings
and 4K TF32 library gap differ materially from the primary; no cause is established.
The 4090 FMA4K favorable median ratios repeat, while its FMA8K win does not. H100
retention still shows traced-kernel savings without an established ordinary wall gain.
These qualifications preserve the selected policy and do not support a universal win.

Finite benchmark acceptance does not establish generic asynchronous allocation safety.
The [targeted ordering investigation](s57-ordering-evidence.md) remains a separate open
proof duty before allocation elimination. Precision policy is unchanged.
