# 2026-10-21 — S58 parallel proof and validation

## 0. Continuation brief

Four GPT-6.1-sol workstreams covered ordering/allocation, small-M selection, GPU
measurement and executed-image capture. Root independently reviewed and integrated the
ordering correction and observer tooling in `3a0d435`. All task rentals are closed.
Small-M and allocation production candidates remain isolated pending their own GPU gates.
The comparison attempt stopped before its first primary process because `/usr/bin/time`
was absent. The separate PyTorch compiled small-matrix gate passes and is independently
rechecked. No new kernel speedup or universal correctness claim is admitted.

First inspect [the evidence table](../performance/s58-parallel-development.md), the
[allocation gate](../../benches/results-s54-gpu/s58-allocation-gate/README.md), and run:

```sh
cd /Volumes/LessComplex/Personal/mapal
supercharge-preflight
openspec list --json
cat benches/results-s54-gpu/s58-allocation-gate/README.md
cat benches/results-s54-gpu/s58-combined-pilot/budget-closed.json
```

Any future rental needs a fresh bound packet/physical-host/cost/deadline review within
persisting user authorization. Never rerun a consumed launcher. The last same-host-only
600-second feasibility query found no eligible offer; no new rental/upload followed.

## 1. Work completed

- Ordering: resolve aliases through the physical grouped writer for stream choice,
  feeder events and member readback; retain the conservative legacy-consumer veto.
  Arena guard, allocation/free calls, runtime ABI and precision policy are unchanged.
- Small-M: isolated seven-line thread-grid adaptation preserves thread count and all
  existing resource/divisibility checks. Two earlier candidates were rejected; final
  2,440-case inventory has 2,276 unchanged, 164 newly legal, zero valid geometries lost.
- Allocation: isolated three-line `store_out` skip omits raw storage only when the launch
  already passes raw pointer zero. Missing group facts stay conservative. Member buffer
  positions, events and surviving instructions remain unchanged. A free-time incidental
  wait disappears, so hardware correctness is still a separate obligation.
- CUPTI observer: repository tooling binds PTX, context, module, function and exact
  geometry without extra synchronization. Review caught wrong-context/geometry acceptance;
  local tests now reject both. Actual Linux injection/cubin/SASS remains unvalidated.
- Matrix: four Mapal emissions and handwritten naive/cuBLAS/PyTorch eager small-case
  checks pass. Explicit development headers repaired the compiled PyTorch prerequisite;
  six checkpoint arrays plus final output pass both frozen componentwise numerical gates.
- Future allocation harness: independent review reproduced two v1 failures before upload, then verified v2 fixes using real processes and the actual CLI; all 432 archive members match the reviewed manifest.

## 2. Decisions

| Decision | Verdict and evidence |
| --- | --- |
| Integrate ordering | Kept: four old-code failures, 50 focused checks, explicit dependency proof, 14 unchanged device LLVM units and 400 ordinary GPU processes across before/after. Both GPU revisions passed; no observed old race or speedup. |
| Ship small-M selection now | Deferred: static/host proof passes; GPU comparison never started. No shape/algorithm-specific shortcut or precision change. |
| Ship allocation elimination now | Deferred: independent static review and full host suite pass; its own no-clock GPU gate must cover removal of incidental free waits. |
| Accept failed small-M output | Rejected: 1/234 records scheduled, zero primary processes started, zero of 30 cells admitted. Three harness negative controls are separate evidence. |
| Claim actual executed SASS | Rejected: optional capture correctly reports unvalidated after incomplete ordinary comparison. Offline S57 disassembly remains distinct. |
| Extend cost or retry automatically | Rejected: original cumulative $1 planning cap retained; final same-host-only feasibility found no offer. |
| Keep harness v1 | Rejected before upload: independently reproduced escaped child after cancellation and nested/symlink result paths corrupting packet inventory. Preserve v1 and reproductions. |

## 3. Tests and numerical evidence

| Check | Result |
| --- | --- |
| Shared required-clang workspace | 1,188 passed / 0 failed / 1 ignored |
| Combined ordering + small-M snapshot | 1,190 passed / 0 failed / 1 ignored after building the unchanged runtime at the test-required fixed path; initial 40 link failures and successful reruns preserved |
| Allocation snapshot | 51 focused, 59 backend-library; full required-clang workspace 1,190/0/1 ignored; all 310 bound source files unchanged after testing |
| Allocation emitted proof | 28 device LLVM identities and surviving-host-body equivalence; independent checker preserves slot types/zero initialization and rejects mutations |
| Ordinary ordering GPU | Frozen 200 + corrected 200 fresh processes; independently rechecked raw outputs |
| Matrix values | 6,496 Mapal ordinary/traced values rechecked bitwise; naive/cuBLAS/PyTorch eager pass; compiled 1,024 checkpoint + 128 final values rechecked against fp32/fmaf budgets, not a shared association claim |
| Small-M packet | 114 combined/isolated PTX identities; correct TF32 recipe includes both `--contract` and `--precision=tf32`; 13 harness checks; independent failed-run analyzer 8 mutation tests |
| Allocation future gate | 640 ordinary + 28 counter CPU-stub processes; two 320-process verifier-negative suites; five real local boundary tests, two phase mocks; independent timeout/SIGTERM/path reproductions pass. No GPU execution. |
| Validation | Strict OpenSpec 7/0; drift 0 dead / 571 refs; source/docs/OpenSpec whitespace clean |

The compiled run is PyTorch 2.13.0+cu130, Inductor fullgraph, static shapes, CUDA graphs
off, case `matrix-m16-d8-h16`. No composed timing or large recurrent acceptance follows.
Raw evidence and rejected cases are retained. Frozen performance still binds source
`b4152c5`; current `target/release/examples/emit` intentionally remains that frozen emitter.

## 4. Live handoff state

| Type | Handle/location | State and continuation |
| --- | --- | --- |
| Checkout | `s54-gpu-fusion` | Only worktree; no push. Preserve unrelated untracked `oainotes.md`. Ordering/tooling commit `3a0d435`; follow-up results commit follows this immutable log. |
| Local jobs/agents | S58 workers and host test sessions | All workers, host tests, graph refresh and local process probes completed; no task job remains live. |
| Task rentals | 57118557, 57121897, 57122410 | Independently absent 22:21:34, 22:43:37, 23:02:58 UTC respectively. Never touch unrelated 57065574/57100404. |
| Closed operations | `/private/tmp/mapal-s58-gpu/` | v59/v61/v62 claims consumed; no restart. v62 cleanup preceded 23:18:04 absolute deadline. |
| Budget | `s58-combined-pilot/budget-closed.json` | $0.7667982573 conservative cumulative planning cost, including failure/transfer reserves; $0.2332017427 remains under $1. Invoice not observed. |
| Combined compiler | `/private/tmp/mapal-s58-combined/bin/emit` | SHA `ccdde46ca700f19a9c9d51d733f1f7e6aaa2767c6dad9240e52a79b60f865307`; source manifest binds 3,343 files. |
| Allocation candidate | `/private/tmp/mapal-s58-allocation/` | Patch SHA `2c8cdc1964abbd0516a46c5da1e02a76ba59e214e205275c8e478dc23e85ffd5`; candidate emitter SHA `61351468b9e59ce0859ded91007f15b5d37ea2a4f6603533202855452d074231`. Not applied. |
| Small-M payload | `/private/tmp/mapal-s58-smallm-bench-payload.tgz` | 5,686,832 bytes, SHA `6f4535d08a49cc92ad8ffdd7863e814e7343cbbf6a916105ad864bc85a48b9b3`. Timing section unchanged; dependency repair under `/private/tmp/mapal-s58-gpu/time-prerequisite-preparation/`. |
| Allocation gate | `/private/tmp/mapal-s58-allocation-gate/` | Use `packet-v2` and `allocation-gate-v2.tgz`, SHA `a93247bef4e50b96ed148dbad6c66a77caf1f10462ab6af8b2210c7a4de545cd`; 12,568,328 bytes. Rejected v1 retained separately; no GPU execution. |
| Private recovery | `.claude/worktrees/_archive-s54-20261021/` | Keep; hash-verified archives for correctness, failed reservation, final combined run, allocation candidate and gate. API/create records may contain credentials and stay private. |

No live task port/server or GPU job is intentionally left running. Artifact directories
are evidence, not instructions to execute consumed launchers.

## 5. In-flight OpenSpec

| Change | Tasks | Next action |
| --- | --- | --- |
| eliminate-unmaterialized-gpu-allocations | 9/15 | Candidate no-clock hardware gate, then isolated allocation/wall comparison |
| capture-executed-cuda-images | 6/8 | Real CUDA/CUPTI capture after ordinary numerical admission |
| optimize-ordered-smallm-gpu | 4/7 | Correct prerequisite, run ordinary ABBA/null/library matrix, then decide integration |
| benchmark-large-composed-gpu-workloads | 5/14 | Recurrent/incremental correctness and matched scopes before performance |

`openspec status --change <name> --json` says planning artifacts are complete. That is
not implementation completion; unchecked tasks above remain open. No incomplete change
was archived.

## 6. Open items and exact checks

1. Review the corrected allocation gate and remaining budget before choosing hardware.
   GPU acceptance needs ordinary value/trap checks separately from traced allocation
   requests. Static requested bytes are not measured VRAM or time saved.
2. Rerun small-M with the explicit GNU time prerequisite. Preserve all 30 cells,
   Exact/Contract/requested-TF32, library statuses, nulls, resources and failures. The
   original 16K frozen comparison remains accepted; the small-M packet's square control
   is 4096, so broader 16K/cross-architecture candidate acceptance remains open.
3. Capture the actual Mapal JIT cubin only after ordinary admission; library executed
   images/counters and precision-matched attribution remain separate work.
4. Complete composed recurrent/incremental references and allocation/retention controls.
   Joint CPU/GPU cost placement and generic paged KV allocation remain unimplemented.

```sh
python3 /private/tmp/mapal-s58-gpu/combined-v62/recheck_compiled.py --out /private/tmp/s58-compiled-recheck.json
python3 -B /private/tmp/mapal-s58-smallm-analysis/analyze.py --results /private/tmp/mapal-s58-gpu/combined-v62/retrieved/trace/combined-v61/smallm-values --outer-summary /private/tmp/mapal-s58-gpu/combined-v62/retrieved/trace/combined-v61/summary.json --out /private/tmp/s58-failed-attempt-recheck
python3 /private/tmp/mapal-s58-allocation-review/check_proof.py
cat /private/tmp/mapal-s58-allocation-gate/README.md
```

Hardware execution is deliberately not an unconditional resume command. Closed payloads,
cleanup deadlines and single-use launch claims cannot be reused.

## 7. Architecture and reconciliation

No new compiler Dat/Trn/Loc/Trm, component or port. `grouped_writer` realizes existing
R7 dependency intent for Gmem member values and host/stream readiness. The allocation
candidate reuses selected-group materialization facts; it does not equate logical
last use with asynchronous completion. The arena guard and Exact/Contract/TF32 policies
stay. Observer tooling maps its evidence transformations and GPU-to-host receipt carrier
in the NVPTX implementation/architecture docs; reviews record FRAMEWORK §4.5 boundaries.

Reconciled NVPTX IMPLEMENTATION → ARCHITECTURE → STATUS for the integrated ordering/tooling,
then root STATUS and continuation. Later findings update STATUS, performance evidence and
OpenSpec partial progress; they do not promote isolated candidates to built architecture.
The evidence report links all raw/public/private recovery receipts and rejected attempts.

## 8. Drift, graph and memory

Drift 0 dead / 571 refs; strict OpenSpec 7/0. Graph refresh uses `graphify update .
--no-cluster` (local AST, no LLM); 1,230 files, 9,594 nodes and 20,384 edges refreshed successfully. Preflight reports all tools installed;
Semble searches were bounded/hung earlier, so known paths were used with that limit
recorded. This immutable file remains the session source; gbrain indexing follows writing
it and its receipt is kept separately.
