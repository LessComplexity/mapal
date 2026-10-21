# S57 checkpoint — five accepted primaries and remaining confirmation

Written 2026-10-21 20:45 UTC. Immutable progress checkpoint; work continues.
The mutable [continuation record](../next-session.md) owns subsequent live-state updates.

## Completed results

All five final primary profiles pass their complete value/trap, timing/status, library,
resource, profiling and independent rental-cleanup gates: RTX 3090, A100 SXM4,
RTX 4090, H100 PCIe and RTX 5090. The independent RTX 4090 confirmation also passes
and is cleaned up. The required independent RTX 5090 confirmation remains running.

The [integrated report](../performance/s54-gpu-fusion.md) now contains 90 primary
single-kernel rows, 25 composed FMA rows, five decode rows and the 18-row independent
4090 confirmation. It links every exported arm, loss, null, raw result and proof.
The independent parser checks 38,577 numeric fields and 19,800 raw kernel samples
across the five primaries, plus 7,494 fields and 3,780 samples for the confirmation,
with zero differences. Library summaries are checked against their printed summaries;
unrecorded library samples cannot be reconstructed.

The initial [presentation review](../../benches/results-s54-gpu/s57-report-presentation-review.md)
checked four primary tables and the 4090 confirmation. One last-digit null rounding
error was corrected. The separate [H100 review](../../benches/results-s54-gpu/s57-report-presentation-h100-review.md)
checks all later H100 additions and aggregate counts. Neither found an outstanding
table, ratio, timing-boundary or numerical-policy mismatch in its bound document version.

Key CUDA-event medians, milliseconds:

| Run / workload | Mapal | cuBLAS | CUTLASS | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 4090 confirmation / 4096 FMA | 3.3423 | 3.5205 | 3.5656 | Favorable median ratios repeat |
| 4090 confirmation / 8192 FMA | 33.4305 | 33.4234 | 31.1501 | Primary win does not repeat; broad overlapping ranges |
| H100 primary / 4096 FMA | 3.6650 | 3.7860 | 4.1975 | Favorable primary medians; no independent H100 confirmation |
| H100 primary / 8192 TF32 | 7.1841 | 3.9710 | 8.7019 | cuBLAS wins; Mapal beats pinned CUTLASS by median |
| 5090 supplement / 16384 FMA | 191.6580 | 154.1048 | 194.0900 | cuBLAS wins; small CUTLASS difference is not an established gain |
| 5090 supplement / M16,N18432,K7168 FMA | 2.0644 | 0.3400 | 0.5530 | Proven small-M route gap |

These are pinned configurations and distinct numerical contracts. Extracted-kernel
acceptance gates Mapal; library errors are diagnostics. TF32 has a library-relative
gate without a fixed absolute accuracy ceiling. The [large-shape report](../performance/s57-large-composed-evidence.md)
preserves 98/100 passing invocations, both rejected M1 TF32 invocations, all six passing
16K base/repeat invocations and the large admitted error diagnostics. No full-model,
BF16/FP8 or application-accuracy claim follows.

## Retention, allocation and compiler conclusions

The decode controls establish large allocation/transfer differences. For H100,
shipped process wall is 973 ms versus 2,382 ms for the combined retention/residency
control, with 8 versus 12,288 traced allocations. That combined control does not
isolate retention. Turning only delta-H2D off raises transferred input bytes from
3,144,704 to 1,074,790,400 at the same eight allocations.

On H100, resident retention ON reduces traced kernel median 161.283→142.827 ms,
but ordinary wall medians are 973 ms shipped versus 1,007 ms enabled with overlapping
ranges. No ordinary wall-time gain is established. The conservative OFF policy remains.
No new compiler/runtime optimization or removal of the arena guard occurred.

The [host call-effects audit](../../benches/results-s54-gpu/s57-host-optimization/host-gpu-call-effects-audit.md)
proves existing CPU vectorization and records which hints reach LLVM. Device lifetime
and completion are not represented by host stack lifetime markers. The backend consumes
dataflow facts explicitly; no missed optimization or benefit from blanket attributes
has been proved. Joint CPU/GPU scheduling remains an open architecture question.

The [disassembly report](../performance/s57-ptx-sass-gap.md) distinguishes saved PTX,
offline Mapal SASS and matched CUTLASS fatbin code. Small-M scalar fallback is proven;
its dynamic bottleneck is not. TF32 conversion tie rules differ, and the current
`.rna` contract must not be changed as an assumed optimization. Actual Mapal driver-JIT
SASS, cuBLAS SASS and useful hardware counters remain absent.

## Live execution and authorization

Frozen source/HEAD remains `b4152c5f4d9ad802d0388c1c64b628091d494e6a`, source manifest
`afdb6d16c0fbeb31789d2760a82ec48ffacd206d9f957af8d0cac72608e62133`.
All 328 source files, four compiler/runtime inputs and three LLVM tools were rechecked
unchanged during the read-only investigations. Do not replace them while validation runs.

| Active tag | Instance | Machine / host | Tool session | Latest partial state |
| --- | --- | --- | --- | --- |
| rtx-5090-confirm-retry3 | 57104993 | 145957 / 617499 | 34139 | 812 values,22 traps,1,240 probes,252 e2e,45/70 decode; no FAIL |

Its budget-derived deadline is approximately 21:02 UTC. The cumulative allowance is
$1.4497 within the $1.50 confirmation cap, including the earlier transport failure and
reserves. Do not reset the cap or accept partial records. Current listed task rate is
approximately $0.7704/hour; shared limit is five rentals/$8 per hour. Never touch
unrelated instances `57065574` or `57100404`.

H100 tag `h100-pcie-primary-retry2` finished in 3,041.8 seconds; instance57102944,
machine147981/host3497 was independently absent at20:37:43 UTC. No H100 rental remains.

The GPU validation agent owns all rental/monitor/collect/cleanup operations:

```sh
python3 /private/tmp/mapal-s56-gpu/final_status.py
```

Ledger/sessions: `/private/tmp/mapal-s56-gpu/integrated/operations/{ledger,sessions}.json`.
After `operations/execution-rtx-5090-confirm-retry3.json` exists, use the existing:

```sh
python3 /private/tmp/mapal-s56-gpu/final_ops.py collect rtx-5090-confirm-retry3
```

The new correctness-only pilot has explicit user approval: **“Approve this correctness
run within the caps.”** Receipt: `correctness-pair/explicit-user-approval.json` under
the temporary base. Scope: the exact reviewed 13,394,834-byte payload, one RTX5090
attempt, equivalent replacement if unavailable before launch, at most $0.80/hour,
$1 total, 55 minutes and the shared $8/hour cap. Do not ask again for that scope.
Automatic review's earlier rejection is retained as historical evidence, not a current
lack of authorization. The selected hosts became unavailable before launch; no pilot
reservation/rental/upload occurred at this checkpoint. The GPU owner is proving a
replacement's transfer allowance, including the explicit Torch bootstrap, before launch.

The four upload bytes remain unchanged. Ordering runs first: 200 fresh ordinary
processes, no trace or inserted clocks. Then isolated Torch2.13.0/cu130 bootstrap and
small matrix correctness against handwritten CUDA/cuBLAS and PyTorch eager/compiled.
Timeout aborts later phases. Complete outputs and arithmetic-route evidence are required;
no timing is admitted. A completed ordering mismatch may still permit the independent
matrix check, while combined success remains false.

## Next actions and recovery

1. Finish/strictly accept or reject the 5090 confirmation; retrieve and independently
   clean the rental. Export/check/report all results, including losses and nulls.
2. Run the approved correctness pilot with a fully bound destination and cost allowance.
   Use its separate ordering and matrix verifiers, not the integrated timing validator.
3. Apply the [planned allocation correction](../../openspec/changes/eliminate-unmaterialized-gpu-allocations/design.md)
   only after the ordering obligations pass. Its [touch map](../../openspec/changes/eliminate-unmaterialized-gpu-allocations/implementation-touch-map.md)
   reuses existing grouped-writer facts in all three stream lookups, preserves the
   conservative legacy-consumer boundary, then omits unstored raw output allocations.
   Group ownership and storage are separate facts; keep the arena guard.
4. Diagnose the recurrent checkpoint blocker after the freeze. The [bounded analysis](../../benches/results-s54-gpu/s57-large-composed-preparation/recurrent-checkpoint-root-cause.md)
   identifies an acyclic exit-payload reconstruction candidate, but exact failing-object
   provenance is still unconfirmed. Trace it before choosing a fix. Local diagnostic
   work is already authorized; no new permission is needed.
5. Admit composed timing only after correctness/route gates, then test shape-derived
   small-M algorithms across families with unchanged numerical contracts. No wins over
   PyTorch, generic paging or joint CPU/GPU scheduling have been established.

Completed recovery archives remain under `.claude/worktrees/_archive-s54-20261021/`.
The earlier session and living continuation list the original consolidation, five
frozen bundles, four primary directories, 4090 confirmation, rejected 5090 partial,
large-shape results, disassembly and prepared correctness payloads. New copies:

- `final-integrated-h100-recovery`:21 files/20,189,782 bytes; receipt SHA
  `adebb9f4d15d2b629c6406a53555359a379ac25d246e001c727daeec35c1a0e4`.
- `s57-correctness-pilot-preparation`:9 files/13,425,578 bytes; receipt SHA
  `3792840c0992ee115e0d140c1e71cb68cf05ad23c1e380285da1a471012df189`.
- `s57-correctness-pilot-destination-v2`:2 files/10,565 bytes, historical reviewed
  destination and rejection; receipt SHA
  `cb1fe9f5e1d558c1979fda8f9d22c6206570be9c1fa4b5b371feb685ea7f3290`.
- `s57-static-followup-notes`:3 files/26,079 bytes; receipt SHA
  `f74c53d1fe3732da6bdd0b37e8d168943472fc4537744b33d9d64b7329b1748e`.

All completed report/evidence files are saved locally; current report changes are not
committed. Active-run outputs, subsequent pilot state and the mutable operations ledger
still need final archival. Preserve `oainotes.md`. No push or merge is authorized.

OpenSpec: selection validation10/15, allocation1/15, large/composed3/14. Selection
validates strictly; latest drift is0 dead/563 references and doc whitespace checks pass.
FRAMEWORK ordering/composition duties remain explicitly open. Final acceptance and
architecture-wide correctness are separate claims; do not archive unfinished changes.
