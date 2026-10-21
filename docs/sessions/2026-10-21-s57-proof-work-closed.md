# S57 — validation recorded; replay repaired; ordering candidate preserved

Immutable handoff written 2026-10-21 21:30:28 UTC. Branch `s54-gpu-fusion`, source HEAD
`627a17ca1a57ed9d948fd417fe99cbd5029456b0`. This handoff and ordering packet are committed
in the following documentation commit. One checkout remains after consolidating 18 worktrees
and pruning three stale registrations; all 51 branches were preserved. No push/merge.
Preserve untracked `oainotes.md` and ignored recovery archives.

## Completed and measured

- `0ca0507`: all frozen validation records and reports committed. Five primary profiles
  (3090, A100, 4090, H100 PCIe, 5090) and independent 4090/5090 confirmations pass complete
  required inventories and cleanup. Independent checks cover 54,121 numeric fields and
  27,900 raw kernel samples with zero differences. Measurements bind frozen source
  `b4152c5f4d9ad802d0388c1c64b628091d494e6a`, not the subsequent replay repair.
- `627a17c`: generic loop-route replay repair applied, reviewed and committed. Existing
  canonical loop decide/advance work is rebuilt through the ordinary replay classifier;
  no new IR mechanism or numerical relaxation. Five regressions fail old code and pass
  the repair. Rewrite tests 86/0; full required-clang workspace 1181/0/1 ignored, formatting
  and release examples pass. Three restricted hardware-detection failures disappeared
  in the full host-access run without a source change. Complete original interpreter
  checkpoints match 8408 numeric values / 8838 lines; all eight previously failing GPU
  emissions complete. No new GPU execution is claimed. Three original native emissions
  still time out at 40 seconds, including a no-tile control; cause remains open.
- Ordering-only candidate preserved, reviewed and **unapplied**. Four exact regressions
  fail old production, two controls pass; candidate 50/0 focused checks and 14 device LLVM
  units unchanged. Runtime/CUDA API proof establishes the intended dependency corrections.
  Shared resident/streams/runtime files still match frozen source. No allocation removal.

[Full tables](../performance/s54-gpu-fusion.md) include 126 shipped kernel cells and all
losses/control exports. [Large extension](../performance/s57-large-composed-evidence.md)
passes 98/100 invocations: all 16K gates pass; two requested-TF32 small-M fallback arms fail
the declared relative gate and remain rejected. CUDA-event medians in milliseconds:

| Host / cell | Mapal | cuBLAS | CUTLASS |
| --- | ---: | ---: | ---: |
| 4090 confirmation FMA4096 |3.3423|3.5205|3.5656|
| 4090 confirmation FMA8192 |33.4305|33.4234|31.1501|
| H100 primary TF328192 |7.1841|3.9710|8.7019|
| 5090 large extension FMA16384 |191.6580|154.1048|194.0900|
| 5090 large extension TF3216384 |115.7243|80.9953|101.6248|
| 5090 dense projection M16/N18432/K7168 FMA |2.0644|0.3400|0.5530|

Mapal is competitive in some 4K FMA cases. The 4090 8K primary win does not repeat;
the 16K CUTLASS FMA difference is not an established gain. cuBLAS wins the 16K cases and
the tested small-M projection. Keep separate arithmetic contracts and timing scopes.

Final 5090 Contract decode 1024×256 ordinary wall is 5,046 ms shipped versus 14,480 ms for the
combined `off-retain-delta+resident` control (2.870×); traced allocation counts 8 versus 12,288.
The primary was 450/1,292 ms. This control disables `retain-delta` and base device residency;
it is not a pure both-retention-OFF control and does not isolate allocation overhead. All 18 saved
PTX/resource tuples match between hosts; actual inspected profiler routes/resources also
match. Driver/power differences are observed, not proved causes. Actual driver-JIT SASS
and hardware counters are absent. The Exact 8192 identical-PTX null remains unstable.

## Correctness attempts closed — no third attempt authorized

The approved 13,395,490-byte packet contains frozen ordering and composed matrix correctness
checks, synthetic fixtures and explicitly installed PyTorch 2.13.0/cu130. Ordering must
run first, untraced, in 200 fresh processes. A phase failure/timeout aborts later phases;
correctness output cannot enter integrated timing acceptance.

1. Initial attempt contract 57110891: allocation failed, exit 1 after 2.1 s, no SSH/upload/
   benchmark; independently absent 21:02:29 UTC.
2. Explicitly approved additional attempt contract 57111469, equivalent offer 51669617,
   machine 143204/host 556515: same pre-upload failure, exit 1 after 2.3 s; independently
   absent 21:07:38 UTC. Original cumulative $1, rate <=$0.80/hour, 55 minutes and aggregate
   $8/hour limits were retained. Both authorized attempts are consumed.

Read-only diagnosis found identical creation options to successful controls and no explicit
account ineligibility. No detailed provider error or concrete cause was proved. The
[sanitized diagnosis](../../benches/results-s54-gpu/s57-correctness-pilot-allocation-failures/allocation-diagnosis.md)
and closed-attempt receipts are committed. No PyTorch install or ordering/matrix GPU run
occurred. Estimates are not invoices. Raw create responses can contain credentials and
remain private 0600 artifacts; never print, commit or upload them.

**Live state:** zero active task rentals, zero local task jobs, no measurement owner.
Never touch unrelated instances 57065574 or 57100404. Both launchers are closed; do not retry
them. Additional GPU work requires a newly justified run and fresh attempt authorization.

## Exact continuation

The mutable [continuation](../next-session.md) carries older CPU work separately. Current
OpenSpecs: `eliminate-unmaterialized-gpu-allocations` 3/15,
`benchmark-large-composed-gpu-workloads` 5/14. Completed validation and replay changes are
archived. Both active plans pass strict validation; drift 0 dead / 563 refs; docs whitespace clean.

1. Ordinary frozen ordering baseline remains task 1.3. Passing small fixtures establishes
   those executions only; it does not prove missing dependencies safe. Keep wrong-value
   observations separate from emitted-code/API proof.
2. After the baseline, integrate the ordering-only candidate, run required host checks and
   ordinary GPU affected/control cases. It uses existing grouped ownership and canonical
   aliases for stream lookup, feeder events/readback and conservative legacy veto. The
   arena guard and runtime remain unchanged. FRAMEWORK laws 1/4/5 stay open until validation.
3. Only then eliminate proved unmaterialized storage. A nonzero free currently synchronizes;
   dropping it can remove an incidental completion wait. Measure matched stream-policy
   before/after allocation counts, requested live bytes, iteration and process-wall time.
4. Resume complete recurrent/composed correctness and comparisons against naive, handwritten,
   cuBLAS and PyTorch eager/compiled across algorithms. Resolve native-emission timeout
   separately. Joint CPU/GPU cost-based placement and generic paged KV ownership remain
   open designs, not implemented features. Static graph topology still needs runtime
   buffer binding/instantiation/replay. No numerical-policy changes are authorized here.

```sh
cd /Volumes/LessComplex/Personal/mapal
supercharge-preflight
git status --short
git worktree list
openspec list --json
cat benches/results-s54-gpu/s57-ordering-candidate/candidate-receipt.json
git apply --check benches/results-s54-gpu/s57-ordering-candidate/ordering-only.patch

# Repaired current compiler: old target/release/examples/emit is deliberately frozen.
shasum -a 256 /private/tmp/mapal-replay-shared-host-target/release/examples/emit
# Expected b8de9090d5f3ddcdc9a36543e5385f665440134f85a4a9985d5aecc3a58bd759

# Only when changing current source and needing fresh validation:
MAPAL_REQUIRE_CLANG=1 CARGO_TARGET_DIR=/private/tmp/mapal-replay-shared-host-target cargo test --offline --locked --workspace --release --no-fail-fast -- --test-threads=1

# Reproduce isolated ordering candidate; do not apply implicitly.
cd /private/tmp/mapal-ordering-fix/source
CARGO_TARGET_DIR=/private/tmp/mapal-ordering-fix/source/target cargo test --offline --locked --release -p mapal-backend-llvm --lib grouped_stream_lookup
CARGO_TARGET_DIR=/private/tmp/mapal-ordering-fix/source/target cargo test --offline --locked --release -p mapal-backend-llvm --test streams --test epilogue --test resident --test rowfold
```

## Final durable recovery

Earlier immutable inventories remain valid. New completed artifacts are hash-verified:

- [Replay proof/review](../../benches/results-s54-gpu/s57-loop-route-replay/README.md);
  full ignored recovery under `.claude/worktrees/_archive-s54-20261021/s57-loop-route-replay/`.
  Diagnostic tar has 2,448 files / 24,287,298 bytes, SHA
  `1d4dd8c57a71373409d39d34eff32d9b631c53507f05621dfdb7ed89877e1bf0`.
  Final shared-workspace/example logs are saved beside it.
- [Ordering candidate receipt](../../benches/results-s54-gpu/s57-ordering-candidate/candidate-receipt.json),
  SHA `7ca018dfdab8114223629368bb7bcc7b16b975fd7bb86b13638288d74909ee70`;
  patch SHA `0813872210b157aabc1887ece05e7d72d68744f420a07ef2efd0b8b29c5fbbbe`.
  Full recovery `s57-ordering-candidate-20261021T212439Z` under the same archive root:
  2,385 files / 59,511,743 bytes, copy receipt SHA
  `ffcb47041ab8b0f389f913dd5d931fb3abfe8867952bf336fd4055559048885f`.
  Independent second verification checked all copies; Cargo targets and duplicate prior
  inputs are excluded. Original frozen fixture inputs remain in earlier recovery archives.
- [Final closed-pilot recovery receipt](../../benches/results-s54-gpu/s57-correctness-pilot-final-closed-recovery.json):
  ignored `s57-correctness-pilot-final-closed-recovery`,268 files / 29,686,641 bytes.
  Copy manifest SHA `2daa27116473f520bc2ef2a587a2619e2c9905e0808084858a1e44c816acacc4`.
- Final 5090 integrated confirmation21files/20,057,698bytes plus resource proof are
  preserved separately; receipts are committed. No mutable temporary directory is the
  only copy of a completed result.

The full source snapshots permit rebuilding if `/private/tmp` is cleared. Preserve exact
raw result bytes; trailing whitespace in captured output is intentional. Commit only
sanitized public proof and docs, never the private runtime/API-operation recovery trees.
