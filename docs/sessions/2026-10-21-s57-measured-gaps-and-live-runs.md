# S57 checkpoint — measured gaps and live frozen validation

Written 2026-10-21 19:58 UTC. This is an immutable progress checkpoint, not final
acceptance or closure of the active changes.

## Source and completed work

One checkout remains: `s54-gpu-fusion`, HEAD
`b4152c5f4d9ad802d0388c1c64b628091d494e6a`. Preserve `oainotes.md` and all recovery
archives. No source edit, Cargo rebuild, replacement of frozen binaries, commit,
push or merge occurred in this checkpoint. All 328 source hashes and HEAD were
independently rechecked unchanged. Source manifest SHA256:
`afdb6d16c0fbeb31789d2760a82ec48ffacd206d9f957af8d0cac72608e62133`.

Four final primary runs are strictly accepted, staged and independently cleaned:
RTX 3090, A100 SXM4, RTX 4090 and RTX 5090. Host validation remains
1,176 passed / 0 failed / 1 ignored. All five frozen bundles are built.

- [Integrated results](../performance/s54-gpu-fusion.md): 72 shipped kernel cells,
  20 composed FMA rows, retention/allocation controls, every loss and null anomaly.
  An independent checker verified 30,621 numeric fields and 15,660 kernel samples.
- [Large-shape results](../performance/s57-large-composed-evidence.md): 100 RTX 5090
  invocations, 98 passing existing gates, two rejected requested-TF32 M1 fallback
  invocations. All six 16K base/repeat invocations pass. Raw sample checking covers
  8,820 kernel measurements. The supplemental rental is independently absent.
- [Disassembly comparison](../performance/s57-ptx-sass-gap.md): independent review,
  corrected requested-face labels, 108 input hashes, 11 output hashes and provenance
  limits. No speculative compiler fix was applied.

## Conclusions and constraints

RTX 4090 square FMA has observed primary-run wins over both libraries at 4096 and
8192; confirmation remains open. cuBLAS wins all three 16K faces. The small FMA
median lead over CUTLASS at 16K is not an established optimization gain.
The M16/N18432/K7168 FMA route is a scalar output-per-thread K-loop and takes
2.0644 ms versus cuBLAS 0.3400 ms. This motivates a shape-derived algorithm
experiment, not a benchmark-name rule or an assumed memory bottleneck.

The Blackwell TF32 conversion difference is real in saved PTX/offline SASS versus
the actual CUTLASS fatbin. However, `Reduced::Tf32` explicitly specifies `cvt.rna`;
the ties-even ADR-0039 proposal is undecided. Changing rounding is a semantic
change. Actual Mapal driver-JIT SASS, cuBLAS SASS and useful hardware counters are
not saved. There is no proven timing benefit from changing that conversion.

The extracted-kernel harness gates only Mapal numerics. Libraries have diagnostics.
TF32's gate is relative to cuBLAS, with no fixed absolute accuracy ceiling; the
diagnostic denominator is `max(abs(f64_ref), 0.001)`. Large admitted diagnostics
are now explicit in both reports. The failed M1 gate's stderr “1” is a Boolean
sentinel, not a counted bad-cell result. No gate was relaxed.

Frozen optimized host LLVM proves an eight-element query loop becomes two
four-lane vectors around preserved GPU calls. All 51 static runtime calls and
seven static launch IDs remain. This proves existing CPU optimization within a
mixed program, not overlap or performance improvement. CPU task parallelism is
disabled on GPU profiles; joint CPU/GPU cost-based placement remains an open
architecture question. Shared dependency/lifetime/alias facts establish legality;
measured compute/transfer/launch costs must establish profitability.

## Live runs and authorization

Latest agent checkpoint: 19:55 UTC. These are partial results, not accepted runs.

| Tag | Instance | Machine / host | PID / tool session | Progress |
| --- | --- | --- | --- | --- |
| rtx-5090-confirm-retry2 | 57100666 | 31810 / 95374 | 41996 / 32333 | 812/812 value passes; 252/252 e2e; 1204/1240 probes |
| rtx-4090-confirm-retry3 | 57101431 | 3358 / 1276 | 42588 / 74414 | 727/727 value passes; 252/252 e2e; 1042/1180 probes |
| h100-pcie-primary-retry2 | 57102944 | 147981 / 3497 | 44799 / 81335 | 569/812 value passes; timing not started |

No reported failures. Confirmations still owe decode/library/kernel/profiling,
final strict validation and cleanup. H100 owes the remaining complete matrix.
Listed aggregate $4.3629/hour; limits are five task rentals / $8/hour. H100 is
$2.5946/hour, within the explicitly approved $3/hour H100 cap and two-hour wrapper.
Never touch unrelated instances `57065574` or `57100404`.

The user explicitly approved the reviewed RTX/16K payloads, equivalent RTX
confirmation replacements, and H100 PCIe uploads/runs including equivalent
replacements within caps. Receipts and packet bindings are under
`/private/tmp/mapal-s56-gpu/integrated/operations/`. Do not re-ask those permissions.
New payloads require their own review; finish a concrete packet before any
approval question that automatic review requires.

The GPU validation agent owns all current runs and cleanup. Exact inspection:

```sh
python3 /private/tmp/mapal-s56-gpu/final_status.py
```

Ledger and sessions:
`/private/tmp/mapal-s56-gpu/integrated/operations/{ledger,sessions}.json`.
Per-tag request/progress:
`/private/tmp/mapal-s56-gpu/integrated/results/<tag>/`.
Only after `operations/execution-<tag>.json` exists:

```sh
python3 /private/tmp/mapal-s56-gpu/final_ops.py collect <tag>
```

Do not relaunch an existing tag or accept partial output. All authorized active
runs continue after this status checkpoint.

## Prepared next work

1. Finish frozen H100 and both confirmations; export/check complete results and
   retrieve/clean/archive before closing the validation change.
2. Run the prepared ordinary GPU ordering reproductions before altering allocation
   behavior. Frozen LLVM proves missing explicit edges in two stored-member cases;
   runtime consequences remain untested. Removing an allocation can remove an
   incidental wait. Preserve the arena guard until completion-order proof exists.
3. Use those results to apply the generic materialization/lifetime fix through the
   existing OpenSpec change. Measure allocation, transfer and compute separately.
4. Validate the small matrix composition against handwritten CUDA/cuBLAS and
   PyTorch eager/compiled before timing. A matrix-only packet is prepared at
   `/private/tmp/mapal-s56-gpu/large-composed/handwritten/matrix-packet/`;
   packet manifest SHA256
   `1afac8f989e4cf6343f7e6cc2dfaceb095e4866a4bf2195e739d29978c171fde`.
   Nine numeric-gate and six status/parser checks pass. Independent packet review,
   local frozen-LLVM build, actual CUDA verification and arithmetic proof remain
   pending. No timing admission. No recurrence in this packet.
5. Investigate small-M cooperative implementations across all shape families with
   unchanged numerical contracts. Joint CPU/GPU scheduling needs separate measured
   end-to-end evidence; kernel timing alone does not validate it.

Full recurrent checkpoints still reproduce a rewrite panic at `replay.rs:989`
with an unmapped feeder; Exact interpretation passes. Minimized repros are saved.
No speculative repair. A100 final-only recurrence proves full/row K/V emission;
RTX 5090's inspected final-only source matches both-retention-OFF LLVM. Do not
infer generic retention from the handwritten incremental reference.

## Recovery and work state

All paths below are under `.claude/worktrees/_archive-s54-20261021/`:

| Recovery copy | Hash-checked contents |
| --- | --- |
| gpu-selection-recovery | 9,874 files / 598,984,566 bytes |
| large-composed-preparation | 2,970 files / 159,284,379 bytes |
| final-integrated-primary-recovery | 41,051 files / 2,423,639,312 bytes |
| s57-supporting-proof-recovery | 850 files / 4,832,758 bytes |
| s57-ptx-sass-gap | 287 files / 186,357,087 bytes |
| s57-handwritten-matrix-preparation | 167 files / 68,150,518 bytes |

Each has a copy-verification receipt. Current raw results, CSVs, public manifests
and receipts are also staged as files under `benches/results-s54-gpu/`. Active-run
outputs and mutable operational state still require their final archive after
collection; do not claim those are already complete.

OpenSpec snapshot: `verify-s54-gpu-selection` 9/15;
`eliminate-unmaterialized-gpu-allocations` 1/15;
`benchmark-large-composed-gpu-workloads` 3/14. All three validate strictly.
Latest drift is 0 dead / 560 references; documentation whitespace check passes.
The FRAMEWORK review leaves placement/dependency/composition duties open for the
targeted ordering cases. No change is archived as complete.
Semble requests stalled during the disassembly follow-up; known files and existing
artifacts supplied the evidence. Graph/memory tooling must not be confused with
hardware acceptance or recovery of live run state.
