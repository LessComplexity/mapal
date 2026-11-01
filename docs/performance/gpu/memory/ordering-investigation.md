# generated ordering evidence

Historical provenance: S57.

S58 update: the ordering correction is integrated and both frozen/corrected revisions
pass 200 ordinary GPU processes. See [current evidence](ordering-and-allocation-development.md).
The investigation and isolated-candidate history below retain their original evidence boundary.

2026-10-21. These are **compiler-output investigations**, using frozen source
`b4152c5f4d9ad802d0388c1c64b628091d494e6a`. A later isolated ordering candidate is documented
below; at the S57 cutoff no ordering fix had been applied and no GPU execution or disassembly
had occurred for these fixtures. Each fixture has Exact/Contract host and device LLVM,
interpreter output, exact commands and before/after source/binary hash receipts.
The original temporary artifact paths in those receipts are preserved; the linked copies
are byte-identical.

| Investigation | Observed generated code | What remains unproved |
| --- | --- | --- |
| [Omitted raw outputs](../../../../benches/results/gpu/memory/ordering/audit/README.md) | Fused writers use streams1/2; stored-member host copies use the legacy path. Two nonzero raw-slot frees precede the first copy; the runtime synchronizes outstanding chains there. | Removing the allocations also removes those free-induced waits. Generic elimination safety cannot be inferred from unused-pointer evidence alone. |
| [Raw outputs retained for later reads](../../../../benches/results/gpu/memory/ordering/audit/materialized-raw/README.md) | Last grouped launch at line157 uses stream2; first stored-member copy at174 uses the legacy path. No call intervenes. First free is at192. Later raw-output copies correctly name streams1/2. | Actual GPU values and copy/kernel overlap remain untested. The emitted code lacks an explicit producer wait for the member copy. |
| [Stored member forked into tiles](../../../../benches/results/gpu/memory/ordering/audit/consumers/README.md) | First child shares stream1 and is ordered by FIFO. Second child uses stream2 with the same stored-member pointer and no event. Intervening allocation calls use already-populated slots. | Actual GPU behavior remains untested. The same-stream concern is refuted; the different-stream edge has no explicit emitted wait. |
| Stored member into a legacy cell | Writers use streams1/2, cell uses stream0 without an event. A real output allocation precedes the cell. | The allocation may mediate or mask ordering; no runtime claim or independent fix is justified yet. |
| Stored member into a row fold | Producer-stream whole readback precedes a host fold; this shape does not use a GPU row fold. | The proposed missing-ordering concern is **refuted for this shape**. |
| Base-output cell/row-fold controls | Existing veto selects legacy stream0 for every launch. | These controls support the current veto for base outputs, not a broader stored-member conclusion. |

Line references in the table apply to both generated host LLVM faces. The detailed
[consumer proof](../../../../benches/results/gpu/memory/ordering/audit/consumers/proof.json)
and [receipt](../../../../benches/results/gpu/memory/ordering/audit/consumers/receipt.json)
record all fixture files and hashes. Interpreter outputs match the independent integer
value oracles; that does not establish asynchronous GPU ordering.

The current compiler already knows the grouped writer: `resident.rs:FnEmit::epilogues`
records member-to-site ownership while forming groups, and `FnEmit::arena` uses that
relationship to place a member's birth at its producer launch. `ResidentPlan::stream_of`
instead queries the object's original IR producer. These are investigation touch sites;
no new annotation or algorithm-specific rule has been shown necessary.

The runtime's legacy copy calls `cuMemcpyDtoH`; its chain-aware copy queues on the named
stream and waits that stream. CUDA nonblocking streams have no implicit synchronization
with stream0 ([NVIDIA driver stream flags](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__STREAM.html)).
The observations therefore establish missing explicit dependency edges in the emitted
cases, while GPU wrong-value or timing consequences require ordinary untraced execution.
Tracing and inserted clocks synchronize and can conceal the behavior being investigated.

## Prepared ordinary execution

The saved LLVM is now lowered into ten Linux host objects and matching RTX 5090
PTX modules: five cases, each on Exact and Contract. All 30 frozen-tool commands
succeeded; all 328 source entries, four existing binary/runtime inputs, three LLVM
tools and HEAD remain unchanged. The [prepared runner and exact commands](../../../../benches/results/gpu/memory/ordering/supporting/ordering-runtime-READY.md)
require 200 ordinary fresh processes (20 repetitions per cell), complete byte-exact
outputs and exact process/status inventories. Ordering runs without profiling or
injected synchronization. Five local kit tests and a separate 200-record CPU-stub
verification test check failure propagation; these are not GPU results.

The [independent peer review](../../../../benches/results/gpu/memory/ordering/supporting/ordering-bundle-peer-review.md)
found no blocker for that scope. The payload has 133 hash-bound files plus its
manifest, and its archive SHA256 is
`2892f59e49b11cfbc46f54ad355aa8b9bd010073787dfe1ace112ad734e900ce`.
The [recovery receipt](../../../../benches/results/gpu/memory/ordering/supporting/ordering-compiled-payload-recovery.json)
preserves the full local preparation. Remote linking and GPU execution remain pending.

A passing baseline would establish those observed executions only. These small kernels
may complete before the unordered read is issued, so passing does not prove an absent
dependency safe. Larger variants require their own oracle and emitted-edge checks;
they are not silently included in this baseline.

The [planned allocation change](../../../../openspec/changes/eliminate-unmaterialized-gpu-allocations/proposal.md)
must resolve this proof obligation before applying dead allocation elimination. Frozen
validation is complete; ordinary ordering execution remains pending. Future changes must
show their before/after LLVM or runtime evidence, including refuted cases and controls;
no performance gain is inferred here.

## Reviewed ordering candidate — unapplied at S57

The [candidate receipt](../../../../benches/results/gpu/memory/ordering/candidate/candidate-receipt.json)
binds the patch, exact commands and recovery snapshot. It factors the existing grouped-writer
query in `resident.rs`, canonicalizes aliases in `ResidentPlan::stream_of`, and uses that
physical producer for launch selection, feeder events and element readback. The legacy
consumer veto uses the same ownership relation. The arena guard, allocation/free calls,
runtime ABI, chain prologue and loop/guard eligibility remain unchanged.

The [runtime/API proof](../../../../benches/results/gpu/memory/ordering/candidate/producer-completion-proof.md)
ties nonblocking stream behavior, event recording/waiting and producer-stream readback to
the actual runtime. Exact final regressions against old production fail in the four intended
cases (member readback, cross-chain fork, epilogue capture and conservative legacy-cell
fallback), while two controls pass. The restored candidate passes all 50 focused checks.
Seven fixtures in Exact/Contract produce 14 device LLVM units byte-identical to baseline;
materialized-raw, fork and capture allocation/free call counts are unchanged. The legacy-cell
case intentionally selects the existing conservative placement. These establish emitted
dependencies and API guarantees, not observed GPU wrong values or performance gains.

Both authorized ordinary-execution attempts failed at Vast allocation before upload; their
instances are independently absent. At that cutoff no third attempt was authorized. The candidate remained
isolated at `/private/tmp/mapal-ordering-fix/source` pending task 1.3, with a hash-verified
2,385-file, 59,511,743-byte recovery under
`.claude/worktrees/_archive-s54-20261021/s57-ordering-candidate-20261021T212439Z`.
Build caches and duplicate prior inputs are excluded. No allocation elimination has been
applied; removing a nonzero free can remove an incidental completion wait.
