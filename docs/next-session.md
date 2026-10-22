# Continue fact-driven GPU optimization

Start with [current results](performance/gpu/current-results.md), the new [executed library comparison](performance/gpu/disassembly/executed-library-images.md), and the [controlled experiment sequence](performance/gpu/disassembly/optimization-experiments.md). Reports and evidence use descriptive topic folders.

## Completed outcome

- Fresh repaired-observer capture is admitted for all three implementations at M16/N2048/K7168 Contract: Mapal1 kernel, cuBLAS2, tuned CUTLASS1. All four exact entries have qualified SASS;4,608 exported numerical words and504 recovered files pass independent checks.
- The frozen Mapal entry uses independent output accumulation. Tuned CUTLASS stages through shared memory and uses several accumulators. cuBLAS contains asynchronous global-to-shared copies and a separately launched split-K reduction. Exact instruction anchors and static-count limits are recorded; no dynamic overlap/occupancy/traffic or causal speedup is claimed.
- Mapal/CUTLASS match256/256 sampled ordered-FMA outputs; cuBLAS matches5/256 and passes its declared componentwise bounds. Current Contract permits ordered FMA, not reassociation. The capture uses driver610.43.02 on a new host, not the earlier timed host.
- The compiler arithmetic-policy correction remains committed at50a57e5:65 library and176 integration tests pass, one optional baseline ignored. The new capture executes the earlier frozen binary, so it does not GPU-validate or time that correction.
- Frozen primitive and composed performance tables are unchanged. Fused cuBLASLt remains2.24×/2.16× faster in the two measured M128 residual MLPs; tuned CUTLASS beats the old primitive Mapal baseline. No generic Mapal win over cuBLAS is established.
- Capture OpenSpec is complete11/11 and archived as `2026-10-22-capture-executed-cuda-images`; it adds the executed-image evidence spec. No production source changes in this continuation.

## Next actions

1. Prepare a controlled ordinary validation of current compiler arithmetic selection against its baseline, then separate ordered tile aspect/output coarsening from panel depth. Preserve exact bits, frozen selection and unchanged controls before promotion.
2. Test an asynchronous-copy/pipeline realization independently, using existing affine/alignment/device facts. Keep arithmetic/output mapping fixed where possible and bind measured kernels to exact images.
3. Design any parallel partial/tree reduction under an explicit numerical permission with an a-priori error bound. Shape and hardware facts determine legal schedules after semantics permit them. Do not silently change Contract or treat TileKSplit address decomposition as reassociation permission.
4. Continue matched submission, retention, allocation-inclusive and fused-program tests. The primitive harness preallocates CUDA buffers and cannot measure Mapal runtime arena benefits. Existing composed clocks measure warmed resident computation with documented preparation differences.

Use OpenSpec for the next concrete compiler/harness change. The existing broad plans remain `benchmark-large-composed-gpu-workloads`6/15, `optimize-ordered-smallm-gpu`4/7 and `eliminate-unmaterialized-gpu-allocations`11/15. The experiment sequence gives discriminating acceptance tests; no new candidate is selected from static SASS alone.

## Live execution and authorization

**No task rental or local benchmark remains running.** All12 actual task rentals are closed. Latest exact-owned57271085, physical140049/host552627, was independently absent2026-10-22T20:48:03.025Z. Attempt12 had an unavailable offer404/3603, empty create response and unchanged full inventory; it created no instance and performed no inferred-ID deletion.

The user explicitly approved the repaired9.16MB bundle and equivalent RTX5090 hosts. That approval passed the normal review route. Eight-minute capture and55-minute instance limits were preserved. Monetary caps remain removed. Protect every pre-existing inventory ID, including57065574,57100404,57126023,57133432. Only an exact new create-response ID, matching its physical host and absent from fresh inventory, permits access/cleanup. Never print provider/account/API/create/wrapper logs. Earlier credential rotation remains outstanding.

Branch is `s54-gpu-fusion`; preserve unrelated `oainotes.md`. Local commits only; no push. Agents `/root/comparison_run`, `/root/composed_analysis_s61` and `/root/gap_analysis_s61` completed operation/admission/analysis and remain reusable. Root owns shared integration and reports.

## Evidence and exact resume commands

```sh
cd /Volumes/LessComplex/Personal/mapal
git status --short
openspec list --json
cat docs/performance/gpu/disassembly/executed-library-images.md
cat docs/performance/gpu/disassembly/optimization-experiments.md
cat /private/tmp/mapal-s61-root/live-state.json
cat /private/tmp/mapal-s61-gpu/attempt-13/results/cleanup.json
cat benches/results/gpu/disassembly/library-kernels/counter-free-capture/rtx-5090/independent-review/REPORT.md
```

Public evidence: `benches/results/gpu/disassembly/library-kernels/counter-free-capture/rtx-5090/`, with separate disassembly, analysis, independent-review, inputs and tools directories. Raw trace remains `/private/tmp/mapal-s61-gpu/attempt-13/retrieved/trace`. Persistent private archive:
`.benchmark-artifacts/gpu/library-images/counter-free/rtx-5090/2026-10-22/admitted-library-kernels/trace.tar.gz`,10,938,433bytes, SHAdab5bbba2e0f6c57d1b60289b442cd496c2f5e8f26c4f37ded6cabb42586dec6.

Repaired observer source remains `.benchmark-artifacts/gpu/library-images/preparation/library-image-capture-v2.tar.gz`, SHA48f288f43a9d2133a50e3572de8aa5604209fc254e5697e0c8027f15d5aaacaf. The failed first trace/source and prior automatic rejection receipts remain immutable; neither was retroactively repaired. Root rehash and publication mapping record every copied artifact.
