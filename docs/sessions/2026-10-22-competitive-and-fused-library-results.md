# 2026-10-22 — Competitive and fused library results

## Continuation brief

Latest GPU measurements and independent admission are complete. No task GPU is live.
The previous organization checkpoint is `dece72f`; this log accompanies the next local
evidence checkpoint. Preserve unrelated untracked `oainotes.md`; no push authorized.
Read [current results](../performance/gpu/current-results.md) and [live handoff](../next-session.md).
Next: independently review/seal the counter-free library-image observer, then run its
bounded pilot on an exactly owned new rental under existing authorization. No GPU launch
has been prepared or admitted for that extension yet.

## Completed measurements and decisions

- Primitive CUTLASS:12geometries attempted,6compile/pass. Selected16×64×16 is2.32–2.34×
  faster than the old default and beats the frozen Mapal baseline in all4cells. cuBLAS
  remains4.31–7.01× faster than tuned CUTLASS. All254,976exported values and1,044events
  pass independent readers;983payload/336ordinary/63outer hashes verified.
- Latest residual MLP:20native confirmation rows plus10original controls admitted on one
  RTX5090. At M128/d256/h512 and M128/d1024/h2048, Mapal is0.073090/0.128935ms,
  tuned fused Lt0.032659/0.059620ms, tuned fused CUTLASS0.046711/0.123867ms.
  All148native executions,770,816words,2,020clock pairs including warmups,8selections,
  exact cross-process algorithm configs and all artifact seals independently pass.
- Keep finite tuning and supported library fusion in the comparison; earlier claims of a
  general Mapal-over-CUTLASS advantage are not supported. No generic selector is promoted.
- Keep matched selected-configuration fusion deltas (Lt1.110×/1.064×;
  CUTLASS1.010×/1.033×). No confidence, generic fusion or causal Mapal retention claim.
- NCU returned ERR_NVGPUCTRPERM, no report/SASS/counters. Reject its damaged pre-gate
  rather than reconstructing it; post-gate success does not admit the profiler attempt.
  No driver/security changes. Callback image capture is the planned next mechanism.

Readable reports: `docs/performance/gpu/kernels/competitive-cutlass.md` and
`docs/performance/gpu/workloads/fused-library-comparison.md`. Raw evidence is separated
under the corresponding `benches/results/gpu/` topic/experiment directories. The ordinary
native/control archive preserves1,384files; full1,492-file trace is kept privately.

## Tests and reconciliation

Strict OpenSpec7passed/0failed; drift0dead/575refs;430report links resolve; all2,802raw
hashes from the organization migration remain unchanged. The new report received an
independent factual review. No new production code was changed after `dece72f`.
Backend/root status, current-results front door, workload indexes, live handoff and OpenSpec
tasks reconcile the admission. FRAMEWORK review is at
`docs/components/backend-nvptx/reviews/review-competitive-library-results.md`.
Code graph was already refreshed this session (9,319nodes/19,103edges); no later production
code changed. Semantic document extraction lacks an LLM key. gbrain capture previously
failed on an existing PGLite lock; do not remove it. Repository logs remain authoritative.

## Live handoff state

| Handle | State | Inspect / resume |
| --- | --- | --- |
| `s54-gpu-fusion` | Evidence/docs checkpoint; unrelated oainotes.md preserved | `git status --short` |
| Task instance57259496, machine58084/host98309 | Closed; exact-ID absence19:17:06.789Z | `cat benches/results/gpu/operations/performance-comparison/cleanup-run-10.json` |
| All10 task rentals | Closed; user machines protected | `cat /private/tmp/mapal-s61-root/live-state.json` |
| `/root/gap_analysis_s61` | Local callback extension; unsealed,32actual-callback CPU cases pass at checkpoint | `cat /private/tmp/mapal-library-image-capture/HANDOFF.md` |
| `/root/composed_analysis_s61` | Native admission complete; independent callback review resumes | `/private/tmp/mapal-library-image-review/` |
| `/root/comparison_run` | GPU operation/provenance audit complete; idle | `/private/tmp/mapal-s61-gpu/attempt-10/` |
| Full latest trace | Preserved private,0700/0600 | `.benchmark-artifacts/gpu/residual-mlp/fused-references/rtx-5090/2026-10-22/recovery-receipt.json` |

No known running local benchmark, task port or GPU job remains. Background agents may
continue local observer/reviewer work; obtain their latest saved handoff before sealing.
Known protected IDs57065574,57100404,57126023,57133432 plus ALL fresh prelaunch IDs.
Only the exact ID from our own create response, absent prelaunch and matching physical
host, permits access or cleanup. Never print provider/account/API/create/wrapper raw logs.
Earlier credential rotation remains outstanding. No new approval is needed for already
reviewed synthetic benchmark content/equivalent RTX5090 runs; user removed monetary caps.

## OpenSpec snapshot

| Change | Tasks | Next action |
| --- | --- | --- |
| benchmark-large-composed-gpu-workloads |6/15| Task3.6 complete for declared primitive+two-width matrix comparison; recurrence/retention/graph-enabled duties remain |
| capture-executed-cuda-images |8/11| Review/seal counter-free library callback extension and perform exact-image pilot |
| optimize-ordered-smallm-gpu |4/7| Use actual library-image evidence before choosing further lowering experiments |
| eliminate-unmaterialized-gpu-allocations |11/15| Allocation counts proved; isolate latency before claiming speedup |

Planning artifacts are complete; implementation tasks remain open. Do not archive any of
these changes merely because a report is published.

## Open items and exact resume

1. Image observer: complete source/runner/cap/cancellation/ambiguity review, seal hashes,
   then bounded live pilot. Done when every selected dispatch maps to an exact image/entry
   with complete metadata and numerical gates. Missing or ambiguous metadata must reject.
2. Gap experiments: compare actual Mapal/library instructions and dispatches with the
   IR fact map; test one falsifiable lowering hypothesis with negative controls before fixes.
3. Broader comparisons: recurrence, causal retention/allocation, graph-enabled PyTorch,
   representative16Kcompositions and cross-host repetition remain unmeasured here.

```sh
cd /Volumes/LessComplex/Personal/mapal
git status --short
openspec list --json
cat docs/performance/gpu/current-results.md
cat docs/next-session.md
cat /private/tmp/mapal-s61-root/live-state.json
cat /private/tmp/mapal-library-image-capture/HANDOFF.md
```

This is a new immutable log; earlier sessions and relocated raw artifact contents were not
rewritten. The path migration map remains `benches/results/path-map.json`.
