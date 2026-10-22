# Allocation gate v2 — prepared, not authorized/executed

V1 is rejected: its outer timeout could leave detached children, and results containment lacked a guard. Original packet, archive, source and receipts remain hash-verified under evidence/rejected-v1. V2 fixes both; no production/candidate source changed.

Packet: `allocation-gate-v2.tgz`; extracted root `allocation-gate-v2/`.
Archive SHA256 `a93247bef4e50b96ed148dbad6c66a77caf1f10462ab6af8b2210c7a4de545cd`.
Packet manifest SHA256 `32d618492a233f3c32708cc8c028a752dd4997413529e487ac4d0f07c8d18a87`.
Archive12,568,328 bytes;432 regular payload files. The packet binds source HEAD3a0d435, isolated DCE patch2c8cdc19…, both exact compiler hashes, unchanged runtime, fixtures/oracles, Linux host objects, LLVM/PTX, scripts and counter predictions. It does not contain a rental/uploader/launcher or alter sealedv62. No execution authorization is conferred.

## Fixed scope and order

Eight64×64 no-clock fixtures: independent join, diamond join, two-chainMLP2, single-chainMLP, materialized-raw, omitted-raw stored members, forked member/raw fanout, and three-iteration carried loop. Every source has no `time`/trace/clock call. Existing three frozen fixture sources/oracles are byte-preserved; added shapes reuse existing regression topology with bounded positive integer or half-integer arithmetic. The interpreter supplies exact complete stdout, with no numerical tolerance. Observations are corner samples, not whole-array correctness.

Before source is3a0d435 ordering correction. After differs only by the isolated raw-prologue omission. Both use identical forced streams+epilogue and stream-gate disabled; shipped streams-OFF policy is not a comparison arm. Sixteen paired device LLVM and PTX units are byte-identical. `gate-static-proof.json` shows alpha-equivalent complete main bodies after deleting only unread old allocations and normalizing entry declarations/zero initialization; free call sites remain, their omitted raw slots now initialize0 and never allocate. This directly identifies the loss of the old live-free context wait without changing the runtime. Passing small ordinary fixtures remains nondiagnostic for universal race absence; readiness still rests on the integrated explicit dependency proof.

1. Validate/hash, link and disassemble all16 before binaries; run20 fresh ordinary processes per case/face, then independent post-run verification:320 processes.
2. Same after:320 processes. Failure/timeout aborts later phases.
3. Only after both ordinary revisions pass, run28 separate fresh processes with unchanged `MAPAL_GPU_TRACE=1`, seven straight-line cases×two faces×two revisions. Exact stdout/status remains checked, but these traced executions are not ordinary correctness or timing evidence. The existing trace synchronizes launches. No trace/runtime modification is made. Loop is excluded from predicted counters because the simple static simulator does not model dynamic iteration multiplicity.

The reviewedv59 runner and independent verifier are reused with fixed inventory/count changes plus the v2 cancellation handler (diffs under evidence). Their exact archive/source hashes are checked in `v59-provenance.json`. New top-level orchestration and counter parsing remain separately reviewable in source/.

## Counter contract (predictions, not measured results)

Existing runtime traces report successful allocation requests after slot caching (`op=alloc`/`arena` with exact requested bytes), live frees, and total `allocs`. The counter parser requires exact ordered requests, their byte sum, total allocation count and successful-free count. Zero-slot frees produce no free line. Raw trace timing fields are retained but never admitted. Static allocation call-site counts differ from successful cached requests and are not conflated.

| Case, each arithmetic face | Predicted successful requests before→after | Predicted requested bytes before→after |
|---|---:|---:|
| independent |7→6|114688→98304|
| diamond |8→7|131072→114688|
|MLP2|12→10|196608→163840|
| omitted-raw |8→6|131072→98304|
| single-chain (arena) |1→1|81920→81920|
| materialized-raw |8→8|131072→131072|
| forked member/raw |8→8|131072→131072|

Predicted successful-free counts equal the successful request counts for these terminating straight-line fixtures. Requested totals/peaks concern only these runtime buffer requests, not driver/module memory or measuredVRAM. No timing or speedup acceptance is included.

## Exact future commands and bounds

Requires a separately authorized single physicalRTX5090 Linux host with existing CUDA headers/stub, driver, cc, objdump and Python3. No package installation or machine mutation is included. After authorized transfer/extraction:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 /w/allocation-gate-v2/run_gate.py /w/allocation-gate-v2 /w/allocation-gate-v2-results --check-only
PYTHONDONTWRITEBYTECODE=1 python3 /w/allocation-gate-v2/run_gate.py /w/allocation-gate-v2 /w/allocation-gate-v2-results
```

Results must be a new directory outside the packet. Ordinary GPU process timeout60s; link180s and disassembly60s inherited fromv59. Each complete ordinary revision is capped600s, counters180s, outer gate1500s including checks. The outer active-phase SIGTERM handler initiates TERM to its phase group, waits up to5s for the inner handler to kill and reap its separately sessioned child group, then uses KILL if the phase refuses TERM. This5s grace is reserved inside the phase/global wait budgets. Timeout/cancellation aborts every later phase. Resolved equal/nested/ancestor/symlink result paths are rejected before checking or writing the packet. The1500s phase budget starts after initial packet validation. Existing environment clearing and injected-preload veto are preserved. No hardware work was run during preparation.

After retrieval, verify locally with the independent pinned manifest digest:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 /private/tmp/mapal-s58-allocation-gate/packet-v2/verify_gate.py /private/tmp/mapal-s58-allocation-gate/packet-v2 /ABS/RETRIEVED/allocation-gate-results --packet-sha 32d618492a233f3c32708cc8c028a752dd4997413529e487ac4d0f07c8d18a87
```

Preserve all linked executables, disassembly, statuses, raw stdout/stderr, ordinary rows, trace rows and phase summaries. Ordinary and instrumented evidence remain separate. Task3.2 for DCE stays open until actual retrieval and independent verification; no performance task closes from this gate.

## Local verification only

- Real CPU interpreter completed all eight fixture oracles; three old immutable oracle bytes match. A rejected initial loop fixture (untyped1.0 inferredf64) is preserved under evidence; corrected1→widen_f32 uses the same intended exact arithmetic.
- Real LLVM lowering completed32 host objects and PTX units (two revisions×eight cases×two faces); no CUDA execution or Linux runtime link occurred.
- Existing verifier CPU fake-process tests pass for both revisions and reject forged success with wrong/truncated values, exit status, duplicate/missing rows and wrong hashes.
- Finalv2 full fixed gate CPU fake tools completed640 ordinary+28 trace stub processes; independent verifier passed and rejected forged counter records. Tests use explicitly fake GPU/cc/objdump and do not establish hardware behavior.
- All28 predicted counter cells pass parser checks and malformed/missing/duplicate/wrong-byte counter negatives. Two phase failure/cancellation mocks pass. Five real local boundary tests prove detached sleeping children die and are reaped after an actual outerSIGTERM and forced outer timeout, prove KILL after5s when a phase ignores TERM, and prove equal/nested/ancestor/symlink paths cause no write or process launch. Independent reviewer reproductions exercise the actual packetCLI with fake physical-GPU query plus real sleeping cc/phase processes; both processes die. Final packet check-only passes without subprocesses/GPU.

No local preparation jobs remain. Parent owns independent review, durable preservation and any future execution authorization.
