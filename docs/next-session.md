# Continue GPU compiler and library comparison work

Start with [current results](performance/gpu/current-results.md), [fact-driven scheduling](performance/gpu/disassembly/fact-driven-scheduling.md), and [the experiment sequence](performance/gpu/disassembly/optimization-experiments.md). New reports and evidence use descriptive topic names. Historical provenance remains unchanged.

## Current state

- Branch `s54-gpu-fusion`; checkpoint `50a57e5` contains the arithmetic-selection correction, verification and docs. Preserve unrelated `oainotes.md`; no push.
- Compiler correction: existing `Launch` carries typed actual arithmetic. Strided Contract and rejected TF32 select their actual scalar policy. Two old-code regressions reproduced; 65 library and176 integration tests pass, one optional baseline ignored. Eight successful host/device LLVM/PTX controls are unchanged; five affected cases retain arithmetic inventories and PDL agreement. Independent review passes. No new GPU runtime/performance validation of this compiler source.
- The [measurement audit](performance/gpu/workloads/measurement-boundaries.md) finds no proved missing-sync or unit defect. Existing composed clocks measure warmed resident computation. Input preparation, synchronization APIs and allocation strategy differ across implementations; matched clocks and inclusive request timing are planned controls.
- Frozen M128 residual-MLP results remain Mapal73.090/128.935µs versus fused cuBLASLt32.659/59.620µs. Competitive CUTLASS primitive tuning also beats the frozen Mapal baseline; all underlying gates and limits remain in their reports. No generic cuBLAS advantage for Mapal is claimed.
- Allocation elimination is integrated and GPU correctness/count checked: 640 ordinary and28 counter processes. No allocation latency gain measured. Small-M geometry alternatives remain isolated because wins accompany regressions.
- First counter-free library pilot completes Mapal execution and all1,536 pre/post numerical words, then rejects callback attribution. Library selected-call legs are skipped; no image/SASS admission. All453 recovered files and exact-owned cleanup are independently checked. Four proved instrumentation boundaries are repaired and independently reviewed: callback invocation identity, ELF helper subranges, provisional UID0 context metadata and API activity-domain binding. The complete synthetic replay passes, with no retrospective admission.

## Active local work

| Owner | Work | Location |
| --- | --- | --- |
| Root | Compiler checkpoint, docs/evidence, reviewed operator decisions | `/private/tmp/mapal-s61-root/` |
| `/root/gap_analysis_s61` | Finished: reviewed repaired observer v2 sealed | `/private/tmp/mapal-library-image-capture/seal-v2.json` |
| `/root/composed_analysis_s61` | Finished: repair review and pinned v2 live reader | `/private/tmp/mapal-library-image-review/repair-review/live-admission-v2/` |

Compiler worker `/root/gpu_schedule_facts` is finished. Its evidence is persisted under
`benches/results/gpu/scheduling/fact-driven-selection/`. Operator `/root/comparison_run`
is idle after automatic approval rejection for attempt12; no new rental or upload occurred.
The repaired observer passes53 source cases,12 identity mutations,10 integration checks
and full synthetic replay. Independent reviews and root rehash pass. Exact revised-bundle
user approval is pending because automatic review requires it; do not retry before the reply.

## Machines and authorization

**No live task rental.** All11 task rentals are closed. Latest instance57265999,
machine135433/host346497, was independently absent2026-10-22T20:05:14.535Z.
Its injected capture failed offline attribution, not execution/numerical checks.
Safe receipt: `/private/tmp/mapal-s61-gpu/attempt-11/operator-outcome.json`.

The user removed monetary caps and explicitly approved the reviewed9.13MB disassembly
bundle plus equivalent RTX5090 hosts. Capture limit480s; instance deadline3300s.
Prior budget/upload approvals persist within their scopes. Automatic review rejected the
repaired9.16MB revision because its corrective observer delta was not explicitly included
in the earlier exact-bundle approval. A precise revised-bundle/equivalent-host approval
question is pending. If approved, preserve the frozen payload, refresh quote/inventory,
rebind root review and submit through the normal review route.

**Protect all pre-existing machines**, including57065574,57100404,57126023,57133432 and
every fresh inventory ID. Only an exact new ID from this task's own create response,
matching physical host and absent from the prelaunch inventory, permits access/cleanup.
Never print provider/account/API/create/wrapper logs. Earlier credential rotation remains
outstanding; it is unrelated to these source/result uploads.

## Evidence and resume commands

```sh
cd /Volumes/LessComplex/Personal/mapal
git status --short
openspec list --json
cat docs/performance/gpu/current-results.md
cat /private/tmp/mapal-s61-root/live-state.json
cat /private/tmp/mapal-s61-gpu/attempt-11/operator-outcome.json
cat /private/tmp/mapal-library-image-review/live-admission/attempt11-v2/REPORT.md
cat /private/tmp/mapal-s61-gpu/attempt-12/approval-review-block.json
```

Failed observer trace is immutable at `/private/tmp/mapal-s61-gpu/attempt-11/retrieved/trace`;
private persistent archive is under `.benchmark-artifacts/gpu/library-images/counter-free/rtx-5090/2026-10-22/rejected-correlation/`.
Public receipts and independent findings are under
`benches/results/gpu/disassembly/library-kernels/counter-free-capture/rejected-correlation/`.
Do not deduplicate, renumber or retrofit attribution into that failed trace.

The broader active plans remain `capture-executed-cuda-images`,
`benchmark-large-composed-gpu-workloads`, `optimize-ordered-smallm-gpu`, and
`eliminate-unmaterialized-gpu-allocations`. The scoped arithmetic correction has its own
`derive-gpu-schedules-from-facts` change, archived as `2026-10-22-derive-gpu-schedules-from-facts`.
Broader shapes, recurrence, graph-enabled PyTorch and causal retention/dispatch/allocation
measurements remain open; the experiment sequence supplies discriminating tests.

Ready repaired packet: `/private/tmp/mapal-s61-gpu/attempt-12/packet.json`, SHA256
`6138fcc648e7009c435e9744eb37b178237da9d4aa99ecc21b3e865d4ed2b637`; root review
`ae680703560f300b549e5d0806a2a3c4bc2460646f4aa2c4201df0a312b6811d`. Observer archive
SHA256 `48f288f43a9d2133a50e3572de8aa5604209fc254e5697e0c8027f15d5aaacaf`,
manifest `21d58721e96250dcf30d8dd23799cfe7e337604fbb4993c81ad5f15cbbc85eac`.
Persistent source: `.benchmark-artifacts/gpu/library-images/preparation/library-image-capture-v2.tar.gz`.
Public repair proofs: `benches/results/gpu/disassembly/library-kernels/counter-free-capture/observation-schema-repair/`.
