# S58 parallel development and evidence

2026-10-21. Four GPT-6.1-sol workers prepared ordering, small-M selection, GPU
measurement and executed-image capture in isolated snapshots. The root reviewed and
integrated the ordering correction. No additional worktree was created and no push occurred.
This is a live report; final execution state belongs in the immutable session handoff.

| Work | Result established | Boundary still open |
| --- | --- | --- |
| Grouped-writer ordering | Four old-code regression failures, 50 focused checks, 14 unchanged device LLVM units; integrated workspace 1,188 passed / 0 failed / 1 ignored | Allocation elimination and performance effects are separate work |
| Ordinary GPU ordering | Frozen and corrected revisions each pass 200 fresh processes, five fixtures × two numerical faces × 20 repetitions | Both passed; this does not demonstrate a race failure in the old revision or universal safety |
| Matrix values | Four Mapal emissions pass; 6,496 ordinary/traced values independently rechecked bitwise. Handwritten naive CUDA, cuBLAS and PyTorch eager reports pass. A later compiled PyTorch run passes; root rechecks 1,024 checkpoint and 128 final values against both frozen componentwise budgets | Small `M=16, d=8, h=16` correctness only; no composed timing or common bitwise association claim |
| Small-M candidate | 2,440 geometry cases: 2,276 unchanged, 164 newly legal, no previous valid geometry changed or lost; 120 emissions and panel-copy coverage; combined snapshot 1,190 passing tests, one ignored | Actual GPU values, resource use and ordinary performance pending; shared selection unchanged |
| Dead raw allocations | Isolated three-line candidate reuses `store_out`; old code fails the new regression, candidate passes 51 focused, 59 backend-library and full workspace 1,190/0/1 ignored; 28 device LLVM units unchanged | Separate GPU correctness/allocation gate required; shared allocation behavior unchanged |
| Executed-image observer | Standalone CUPTI observer and local negative tests prepared; review found and corrected missing context/geometry binding | Linux injection, real GPU cubin capture and actual executed SASS remain unvalidated |

## Ordering integration

`resident.rs:grouped_writer` factors the existing member-to-launch relation used by
arena births. `ResidentPlan::stream_of` resolves canonical aliases through that
physical writer before selecting a stream. Launch feeder events, member readback and
the legacy-consumer veto use the same relation. This realizes the existing R7 rule,
with no new compiler annotation, runtime ABI or numerical contract.

The arena guard and allocation/free calls remain unchanged. Existing frees can impose
completion, so deleting unused storage is not justified by pointer non-use alone.
The static next-step estimate is one 1 MiB allocation for each independent/diamond
pt512 case and two for MLP2; twin/single-chain controls are unchanged. These are
predicted call/byte changes, not measured savings or an applied optimization.

The isolated follow-up now realizes that estimate using the existing selected group's
`store_out` fact in `FnEmit::chain_prologue`. The candidate removes allocation only when
the launch already passes raw pointer zero and omits its raw store. Missing group facts
remain conservative. Across 28 before/after variants, the proof retains all surviving
host instructions and ordering, including member allocation positions, and finds all
device LLVM byte-identical. The old implementation fails the new regression on eight
unused allocation calls across independent, diamond and MLP2 in both arithmetic faces.
The [static packet and recovery receipt](../../benches/results-s54-gpu/s58-allocation-candidate/README.md)
are preserved; root rehashed 310 source and 261 evidence files plus both emitters and
the patch. Removing a live free's incidental wait still requires the candidate's own
ordinary GPU gate. This patch is not applied and has no measured performance result.
An [independent review](../../benches/results-s54-gpu/s58-allocation-candidate/review/review.md)
finds no static blocker and strengthens the equivalence check to preserve slot types
and zero initialization. Mutations changing `i64` to `i32` or zero to one are rejected.
Root ran the isolated candidate's full required-clang workspace: 1,190 passed, zero
failed, one ignored. A [future no-clock gate](../../benches/results-s54-gpu/s58-allocation-gate/README.md)
is prepared for 640 ordinary before/after processes and 28 separate allocation-counter
processes. CPU stubs establish harness behavior, not GPU execution or speed.
Independent review rejected that gate's first version before upload: an inner child
started in a separate session survived outer cancellation, and a nested results path
mutated the packet before rejection. Reproductions and the rejected packet are preserved;
the corrected v2 passes independent actual CLI SIGTERM, phase-timeout and resolved-path
reproductions. All 432 sealed files verify; kernels, runtime and numerical oracles are
unchanged. Hardware execution remains pending.

The [original emitted-edge proof](s57-ordering-evidence.md) remains the source for
old/new negative controls and CUDA runtime semantics. The
[independent retrieved-output validation](../../benches/results-s54-gpu/s58-correctness-pilot/root-independent-verification.json)
rehashes 1,078 adapter files and 108 matrix files and validates both 200-process runs.
The [matrix recheck](../../benches/results-s54-gpu/s58-correctness-pilot/matrix-partial-recheck.json)
and [compiled failure](../../benches/results-s54-gpu/s58-correctness-pilot/pytorch-compiled-stderr.txt)
preserve the first attempt's partial verdict. The subsequent
[compiled numerical recheck](../../benches/results-s54-gpu/s58-combined-pilot/root-compiled-recheck.json)
passes for six checkpoint arrays and the final output against both frozen componentwise
budgets. PyTorch 2.13.0/cu130 ran Inductor fullgraph, static shapes, with CUDA graphs off;
this certifies only the small matrix case and supplies no matched performance result.

## Small-M hypothesis and rejected attempts

The inspected M=16 scalar fallback is caused by the existing tile's row extent failing
divisibility. The candidate redistributes row threads to columns only after the existing
occupancy shrink, preserving thread count and retaining panel, divisibility and shared
memory admission. A previously legal positive geometry cannot enter the new loop.
Zero dimensions retain the original path.

Two earlier pre-shrink changes were rejected: one violated panel/resource constraints;
another changed 135 valid selections and lost 75 in the 2,440-case inventory. The final
seven-line candidate avoids those observed regressions. Requested TF32 can also change
route through existing rung selection, so it receives its own numerical and timing gates.
Exact still emits separate multiply/add; Contract still permits FMA. No TF32 rounding
policy is changed.

The [candidate evidence](../../benches/results-s54-gpu/s58-smallm-candidate/verification.json)
preserves the rejected and final geometries, exact commands and patch hashes. All 114
benchmark PTX files regenerated from the root's combined snapshot match the independently
built candidate byte-for-byte after correcting a preparation error: requested TF32 must
include both `--contract` and `--precision=tf32`. No GPU run used the incorrect recipe.
The [combined host suite](../../benches/results-s54-gpu/s58-smallm-candidate/combined-workspace-verification.json)
has 1,190 passing tests and one ignored. Its initial 40 failures came from the isolated
snapshot missing the fixed-path native runtime library; building that unmodified library
and rerunning the two affected targets resolved all 40. Failure/retry logs are retained.

GPU comparison preparation includes cuBLAS/CUTLASS, three rectangular families,
M=1/16/128, a square control, Exact/Contract/requested TF32, identical-PTX null arms,
30 raw CUDA-event samples per leg, and separate wall/resource receipts. Bounded shapes
check every output; actual larger shapes retain the recorded 256-cell sampled gate.
The copied harness explicitly rejects nonfinite values and checks library statuses.
Its ordinary timing section is byte-identical to the frozen harness. Failed numerical
cells are quarantined rather than assigned accepted timing.

The GPU attempt did not reach those comparisons. The adapter omitted the required GNU
`time` prerequisite; the first primary process could not start because `/usr/bin/time`
was absent. One record was scheduled out of 234, with empty streams and no process-status
receipt. Three harness negative controls did execute, but zero primary numerical cells
or timings are admitted. The [independent failure analysis](../../benches/results-s54-gpu/s58-smallm-failed-attempt-analysis/results/REPORT.md)
rehashes 2,866 payload files and 180 outer result files and preserves all 30 unmeasured
rows. Installing/checking the timing utility before the benchmark is prepared and tested;
the frozen timing section and numerical gates remain unchanged. Actual GPU resource and
speed comparisons are still open.

## Disassembly evidence

The [S57 PTX/SASS report](s57-ptx-sass-gap.md) remains the accepted static evidence.
The observer is intended to bind exact input PTX, module load, actual CUDA context,
function lookup, launch geometry and callback-supplied cubin bytes. Independent review
demonstrated that its first version accepted a different resource context and unexpected
launch dimensions; both are now explicit rejection cases. Local compilation and stub
tests do not establish CUDA injection behavior on a GPU. Instrumented captures must be
kept separate from ordinary timings, and library executed-image capture remains open.

## GPU cost and cleanup

RTX 5090 instance 57118557 ran at the listed $0.6598148148/hour rate and was independently
absent at 22:21:34 UTC. Reservation-to-confirmed-absence was 456.813 seconds, approximately
$0.08373 at that rate. Adding the prior failed-attempt allowance and transfer reserve gives
$0.40373 cumulative planning cost, leaving $0.59627 under the original $1 cap. These are
conservative planning figures, not an invoice. The
[closed budget](../../benches/results-s54-gpu/s58-correctness-pilot/budget-closed.json)
and [cleanup receipt](../../benches/results-s54-gpu/s58-correctness-pilot/cleanup.json)
record the boundary. A further bounded performance/compiled-correctness packet is being
prepared; no result from it is claimed here.

The first combined reservation, instance 57121897, failed before SSH/upload and was
independently absent at 22:43:37 UTC. Zero payload bytes reached it and no GPU phase ran.
The provider supplied no diagnostic cause. A conservative $0.11 allowance raises the
cumulative planning total to $0.51373, leaving $0.48627. The
[closed failure evidence](../../benches/results-s54-gpu/s58-combined-attempt-allocation-failure/closed-failure-receipt.json)
preserves the exact unuploaded payload and cleanup. A manually reviewed replacement with
identical static bytes, instance 57122410 on machine 154241/host 303447, completed the
compiled matrix check and was independently absent at 23:02:58 UTC, before its
23:18:04 deadline. Listed reservation-to-absence time was 894.419 seconds, approximately
$0.15307. The [closed budget](../../benches/results-s54-gpu/s58-combined-pilot/budget-closed.json)
including a further $0.10 transfer reserve totals $0.76680, leaving $0.23320. A final
read-only check for that same host under a 600-second/$0.23320 limit found no eligible
offer; no further rental/upload occurred. All task rentals are closed. The optional
executed-image phase correctly remained unvalidated because ordinary small-M comparison
had not completed. No new kernel performance result is admitted.
