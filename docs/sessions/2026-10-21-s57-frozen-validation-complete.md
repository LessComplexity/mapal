# S57 — frozen validation complete; generic proof work continues

Immutable checkpoint written 2026-10-21 21:09 UTC. Branch `s54-gpu-fusion`, HEAD
`b4152c5f4d9ad802d0388c1c64b628091d494e6a`. One checkout; preserve ignored recovery
archives and untracked `oainotes.md`. No push/merge. Reports and new proof files are local
and uncommitted at this checkpoint; production sources and frozen binaries are unchanged.

## Completed validation

All five primaries and independent RTX 4090/5090 confirmations strictly pass complete
numerical, timing/status, library, resource and profiling gates. All seven corresponding
rentals are independently absent. Final confirmation instance57104993 was absent at
20:55:19 UTC. Independent recomputation checks54,121 numeric fields and27,900 raw kernel
samples with zero differences. The source manifest remains
`/private/tmp/mapal-s56-gpu/integrated/source-manifest.json`, SHA
`afdb6d16c0fbeb31789d2760a82ec48ffacd206d9f957af8d0cac72608e62133`.

[Full measured results](../performance/s54-gpu-fusion.md) publish126 shipped kernel cells,
all loss/control exports, composed timings and decode allocation/transfer counts.
[Selection](../performance/s54-per-arch-selection.md) remains5090 streams OFF,
H100 retention/window OFF, derived5090 raster gate restored. Precision policy unchanged.
OpenSpec `verify-s54-gpu-selection` was strictly validated and archived as
`openspec/changes/archive/2026-10-21-verify-s54-gpu-selection`, updating three requirements
in the main `gpu-validation` spec. Reports, component/root status and FRAMEWORK review
are reconciled. Drift was0 dead/563 references; docs/OpenSpec whitespace check passes.
Task4.3 closes with this immutable handoff. Targeted ordering duties1/4/5 remain explicitly
open in the review; the finite matrix does not prove generic asynchronous safety.

Representative CUDA-event medians, Mapal/cuBLAS/CUTLASS milliseconds:

| Host/cell | Mapal | cuBLAS | CUTLASS | Scope |
| --- | ---: | ---: | ---: | --- |
| 4090 confirmation FMA4096 |3.3423|3.5205|3.5656|Favorable median ratios repeat|
| 4090 confirmation FMA8192 |33.4305|33.4234|31.1501|Primary win does not repeat; broad overlap|
| H100 primary FMA4096 |3.6650|3.7860|4.1975|One final primary|
| H100 primary TF328192 |7.1841|3.9710|8.7019|cuBLAS faster|
| 5090 confirmation TF324096 |4.4472|1.8883|1.8883|Gap larger than primary|
| 5090 large extension FMA16384 |191.6580|154.1048|194.0900|Small CUTLASS difference not established gain|
| 5090 large extension TF3216384 |115.7243|80.9953|101.6248|Both libraries faster|

The final5090 stream-OFF medians are lower in all8 changed cells, with6 disjoint ranges
and2 overlaps. Contract decode1024×256 ordinary wall is5046ms shipped vs14480ms for the
combined residency/retention-disabled control,8 vs12288 traced allocations. The primary
was450/1292ms. Identical counts/bytes do not explain that absolute host difference.
The [saved resource comparison](../../benches/results-s54-gpu/s57-rtx-5090-confirm-driver-resource-evidence.md)
finds identical PTX and all18 shipped resource tuples, matching actual TF324096 profiler
routes/resources. Actual driver-JIT SASS and hardware counters are missing. No causal
fix follows. Note SHAa2474c325426fd121162410191cc04dcffb671ea39893099673b419ef8930117;
receipt SHA14c62917e623c8bfea3ab890c18e96d359adddd7d32ac168877a304d317f1dc0.
The primary5090 Exact8192 identical-PTX null34.64→48.37ms remains unstable.

The separate16K/rectangular extension passes98/100 invocations; both rejected arms share
a requested-TF32 small-M FP32 fallback. Library kernel errors are diagnostic, and the
relative TF32 gate has no fixed absolute ceiling. No full LLM/BF16/FP8 claim.

## Two correctness-pilot allocation failures, no GPU result

The user approved a13.4MB packet containing frozen runtime/LLVM/PTX, synthetic ordering
and small matrix comparisons. Ordinary200-process ordering precedes explicit isolated
pip25.3/Torch2.13.0cu130 installation and matrix correctness. No timing admission.
Archives are unchanged: matrix`c460670f...aa056`, ordering`2892f59e...00ce`.
The adapter's pinned pip upgrade repairs proved Jammy pip22.0.2/`--report` incompatibility;
7 adapter and8 original launcher stub tests pass. Actual installation is untested.

1. Approved initial attempt: offer56695170,m54112/h146915, contract57110891. Create reported
   `success=false`, first status intended stopped, wrapper exit1 after2.1s. No SSH/upload/
   benchmark; independent absence21:02:29 UTC.
2. User explicitly approved one additional attempt under the original cumulative$1 cap,
   <=$0.80/hour, <=55minutes, <=$8/hour aggregate. Equivalent offer51669617,m143204/h556515,
   $.660741/hour, estimated cumulative allowance$.837704 including prior$.11. Contract
   57111469 also returned unsuccessful allocation/stopped; exit1 after2.3s, before upload.
   Independent absence21:07:38 UTC. No further attempts are authorized.

At this checkpoint **zero active task rentals**. Neither ordering nor matrix ran.
Provider image-transfer/invoice facts are unavailable; estimates/reserves are not invoices.
Read-only startup diagnosis is ongoing before any additional rental proposal. Never touch
unrelated instances57065574 or57100404. Create responses may contain API credentials:
retain privately with0600 permissions; do not print, commit or upload them.

## Live local proof work and exact continuation

`audit_legacy_worktrees` owns `/private/tmp/mapal-replay-diagnostic/source` and isolated
sibling`target`. OpenSpec`repair-loop-route-replay` precedes all edits. Diagnostic traces
prove tiny674/original5907 are unmapped exit payload pairs outside the SCC but inside the
canonical loop driver cone during round1 ConstFold, not removed or redirected feeders.
The proposed generic fix uses existing LoopPlan decide/advance targets and the existing
`reconstruct_a` classifier. Five new regressions fail unchanged code and pass with the
fix; focused26 and full rewrite86 tests pass. The original recurrent source is currently
being emitted across GPU targets/faces and checked against complete small native/interpreter
outputs. No shared fix yet. Root reviewed the narrow production diff without a blocker.

`audit_s54_worktrees` now owns a separate new
`/private/tmp/mapal-ordering-fix/source` snapshot/isolated target. It is preparing runtime/
API dependency proof and negative-controlled emission regressions for grouped-writer
readback, cross-chain/capture operands and conservative legacy consumers. Existing
`eliminate-unmaterialized-gpu-allocations` and its implementation-touch-map are authority.
**No allocation removal yet.** Preserve arena guard and loop/guard boundaries. GPU baseline
proof remains unexecuted after allocation failures; do not claim observed wrong values.

GPU agent owns only read-only failure diagnosis, final sanitized receipts and cleanup.
Root owns reporting/reconciliation, independent review and eventual shared integration.
Agents must not revert each other's edits or overwrite frozen artifacts.

```sh
python3 /private/tmp/mapal-s56-gpu/final_status.py
cat /private/tmp/mapal-s56-gpu/integrated/results/correctness-pair-rtx-5090-retry2/verification.json
cat /private/tmp/mapal-s56-gpu/correctness-pair/additional-attempt-user-approval.json
cat /private/tmp/mapal-replay-diagnostic/evidence/fix.patch
cat /private/tmp/mapal-replay-diagnostic/evidence/rewrite-package.stdout
openspec list --json
supercharge-drift
```

Inspect exact new command/status receipts before resuming a build. Do not rerun either
consumed pilot launcher or write into an existing output directory. All future compiler
performance claims need actual launched-code proof and matched numerical/timing scopes.
After the replay patch passes review and full host checks, apply/reconcile/commit locally;
continue ordinary ordering/composed correctness when a justified GPU run is authorized.
Do not attribute gains to allocations alone when a control changes streams/residency too.

## Durable recovery

Earlier checkpoint inventories remain valid. Final5090 confirmation archive contains21
files/20,057,698bytes; copy receipt SHA
`0c33a4a0a6f6d6ce34887721d2e3d89a3cc34d3313c95626dbfa4a6669d8b8d7`.
The first correctness-failure preparation/operations checkpoint contains221files/
29,423,503bytes; receipt SHA
`82d451ea747d5a52c6590ef8faab7e3241811e7bf75d8da19cd696c251c9cf85`.
Both are ignored immutable recovery directories. The second failed attempt, new resource
note and evolving local compiler proof still need their final immutable recovery copies.
The mutable [continuation](../next-session.md) should reflect subsequent work; never edit
this checkpoint to make its live state appear current.
