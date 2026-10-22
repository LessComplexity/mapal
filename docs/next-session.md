# Continue GPU compiler and library comparison work

Start with [current results](performance/gpu/current-results.md), [fact-driven scheduling](performance/gpu/disassembly/fact-driven-scheduling.md), and [the experiment sequence](performance/gpu/disassembly/optimization-experiments.md). New reports and evidence use descriptive topic names. Historical provenance remains unchanged.

## Current state

- Branch `s54-gpu-fusion`; last committed checkpoint `d80d8f5`. The arithmetic-selection correction and its verification/docs are ready for a new local checkpoint. Preserve unrelated `oainotes.md`; no push.
- Compiler correction: existing `Launch` carries typed actual arithmetic. Strided Contract and rejected TF32 select their actual scalar policy. Two old-code regressions reproduced; 65 library and176 integration tests pass, one optional baseline ignored. Eight successful host/device LLVM/PTX controls are unchanged; five affected cases retain arithmetic inventories and PDL agreement. Independent review passes. No new GPU runtime/performance validation of this compiler source.
- The [measurement audit](performance/gpu/workloads/measurement-boundaries.md) finds no proved missing-sync or unit defect. Existing composed clocks measure warmed resident computation. Input preparation, synchronization APIs and allocation strategy differ across implementations; matched clocks and inclusive request timing are planned controls.
- Frozen M128 residual-MLP results remain Mapal73.090/128.935µs versus fused cuBLASLt32.659/59.620µs. Competitive CUTLASS primitive tuning also beats the frozen Mapal baseline; all underlying gates and limits remain in their reports. No generic cuBLAS advantage for Mapal is claimed.
- Allocation elimination is integrated and GPU correctness/count checked: 640 ordinary and28 counter processes. No allocation latency gain measured. Small-M geometry alternatives remain isolated because wins accompany regressions.
- First counter-free library pilot completes Mapal execution and all1,536 pre/post numerical words, then rejects callback attribution. Library selected-call legs are skipped; no image/SASS admission. All453 recovered files and exact-owned cleanup are independently checked. Three proved instrumentation boundaries require repair: callback invocation identity, ELF helper subranges, and provisional UID0 context metadata.

## Active local work

| Owner | Work | Location |
| --- | --- | --- |
| Root | Compiler checkpoint, docs/evidence, reviewed operator decisions | `/private/tmp/mapal-s61-root/` |
| `/root/gap_analysis_s61` | Repair and test observer identity/schema boundaries; preserve sealedv1 and failed trace | `/private/tmp/mapal-library-image-capture/` |
| `/root/composed_analysis_s61` | Independent repair-contract/source review | `/private/tmp/mapal-library-image-review/repair-review/` |

Compiler worker `/root/gpu_schedule_facts` is finished. Its evidence is persisted under
`benches/results/gpu/scheduling/fact-driven-selection/`. Operator `/root/comparison_run`
is idle after cleanup; no retry is launched. The new observer requires local regression
proof, independent review and a new seal before a fresh live smoke.

## Machines and authorization

**No live task rental.** All11 task rentals are closed. Latest instance57265999,
machine135433/host346497, was independently absent2026-10-22T20:05:14.535Z.
Its injected capture failed offline attribution, not execution/numerical checks.
Safe receipt: `/private/tmp/mapal-s61-gpu/attempt-11/operator-outcome.json`.

The user removed monetary caps and explicitly approved the reviewed9.13MB disassembly
bundle plus equivalent RTX5090 hosts. Capture limit480s; instance deadline3300s.
Prior budget/upload approvals persist within their scopes. Fresh reviewed packets and
ownership checks are required; do not repeat an already granted permission question.

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
```

Failed observer trace is immutable at `/private/tmp/mapal-s61-gpu/attempt-11/retrieved/trace`;
private persistent archive is under `.benchmark-artifacts/gpu/library-images/counter-free/rtx-5090/2026-10-22/rejected-correlation/`.
Public receipts and independent findings are under
`benches/results/gpu/disassembly/library-kernels/counter-free-capture/rejected-correlation/`.
Do not deduplicate, renumber or retrofit attribution into that failed trace.

The broader active plans remain `capture-executed-cuda-images`,
`benchmark-large-composed-gpu-workloads`, `optimize-ordered-smallm-gpu`, and
`eliminate-unmaterialized-gpu-allocations`. The scoped arithmetic correction has its own
`derive-gpu-schedules-from-facts` change, ready to archive after its checkpoint handoff.
Broader shapes, recurrence, graph-enabled PyTorch and causal retention/dispatch/allocation
measurements remain open; the experiment sequence supplies discriminating tests.
