# S58 parallel development and evidence

2026-10-21. Four GPT-6.1-sol workers prepared ordering, small-M selection, GPU
measurement and executed-image capture in isolated snapshots. The root reviewed and
integrated the ordering correction. No additional worktree was created and no push occurred.
This is a live report; final execution state belongs in the immutable session handoff.

| Work | Result established | Boundary still open |
| --- | --- | --- |
| Grouped-writer ordering | Four old-code regression failures, 50 focused checks, 14 unchanged device LLVM units; integrated workspace 1,188 passed / 0 failed / 1 ignored | Allocation elimination and performance effects are separate work |
| Ordinary GPU ordering | Frozen and corrected revisions each pass 200 fresh processes, five fixtures × two numerical faces × 20 repetitions | Both passed; this does not demonstrate a race failure in the old revision or universal safety |
| Matrix values | Four Mapal emissions pass; 6,496 ordinary/traced values independently rechecked bitwise. Handwritten naive CUDA, cuBLAS and PyTorch eager reports pass | PyTorch compiled fails before values: generated `cuda_utils.c` cannot include `Python.h`; no timing admitted |
| Small-M candidate | 2,440 geometry cases: 2,276 unchanged, 164 newly legal, no previous valid geometry changed or lost; 120 emissions and panel-copy coverage; combined snapshot 1,190 passing tests, one ignored | Actual GPU values, resource use and ordinary performance pending; shared selection unchanged |
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

The [original emitted-edge proof](s57-ordering-evidence.md) remains the source for
old/new negative controls and CUDA runtime semantics. The
[independent retrieved-output validation](../../benches/results-s54-gpu/s58-correctness-pilot/root-independent-verification.json)
rehashes 1,078 adapter files and 108 matrix files and validates both 200-process runs.
The [matrix recheck](../../benches/results-s54-gpu/s58-correctness-pilot/matrix-partial-recheck.json)
and [compiled failure](../../benches/results-s54-gpu/s58-correctness-pilot/pytorch-compiled-stderr.txt)
preserve the partial numerical verdict without promoting it to full acceptance.

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
identical static bytes, instance57122410, is loading on the independently checked
machine154241/host303447 at22:49:02UTC. Rate $0.6161111111/hour; absolute cleanup
deadline23:18:04UTC and cumulative planning reservation $0.92178. No performance
result from that live run is admitted yet.
