# Large shapes and composed programs: evidence in progress

Frozen source: `b4152c5f4d9ad802d0388c1c64b628091d494e6a`. These experiments use
separate temporary inputs and the existing compiler/runtime/LLVM binaries. They do not
change the final S54 validation matrix. The RTX 5090 primitive extension has completed: **98/100 invocations validate; two
requested-TF32 fallback invocations fail the unchanged value gate.** All six 16K
base/repeat invocations validate. New composed GPU/PyTorch results remain pending.

## Primitive emission

| Scope | Result |
| --- | --- |
| Shapes | 16384 square; K/N = 7168/18432, 7168/2048, 4096/1024 at M = 1, 16, 128, 2048, 8192 |
| Profiles / faces | A100 and RTX 5090; Exact, Contract, requested TF32 |
| Compiled realizations | 96 / 96 |
| Actual arithmetic | 32 separate FP32 mul/add; 32 FP32 FMA; 25 TF32 MMA; 7 requested-TF32 FP32-FMA fallbacks |
| 16K explicit harness storage | 5.25 GiB device; 3 GiB host vectors, before context/workspace overhead |
| Prepared first GPU run | RTX 5090: 48 extension realizations + 2 frozen rectangular controls, each repeated with identical PTX as a null |

All M=1 requested-TF32 realizations fall back to FP32 FMA. The RTX 5090
7168-to-18432 projection at M=16 also falls back; its A100 counterpart emits MMA.
Every M=16 Exact/Contract realization uses the scalar output-per-thread fallback.
These are emitted-code observations, not measured performance explanations.

The [shape/route CSV](../../../../benches/results/gpu/workloads/large-matrices/preparation/primitive/summary.csv)
records all 96 cells. The [manifest](../../../../benches/results/gpu/workloads/large-matrices/preparation/primitive/manifest.json)
binds source, host/device LLVM, PTX, launch geometry, commands and input-law provenance.
Model dimensions come from the pinned [model configurations](../../../../benches/results/gpu/workloads/model-shapes/receipt.json).
These are FP32-storage projection experiments, not BF16/FP8 model execution.

The unchanged `tile_bench.cu` computes error diagnostics on 256 sampled outputs per leg and times
Mapal/cuBLAS/CUTLASS with CUDA events after its 256 MiB flush. Mapal's ordered
Exact/Contract or existing TF32 gate is mandatory; cuBLAS and CUTLASS have no
independent numeric acceptance gate in this instrument. Their f64 errors remain
diagnostics. This is not an all-output 16K CPU oracle.
The harness supplies seeded random arrays directly to extracted kernels; standalone
source initializers are not those runtime inputs. Runtime capacity, unsupported library
shapes, failed processes and missing samples must remain explicit non-passing outcomes.

## RTX 5090 large/rectangular measurements

Machine40773 / host214845, driver610.57.04, power limit465W. Thirty CUDA-event samples per leg; 256MiB flush before each event region. All times below are **device-kernel milliseconds**. Each base has an identical-PTX repeat, preserved with full distributions and raw samples in [cases.csv](../../../../benches/results/gpu/workloads/large-matrices/rtx5090/cases.csv). The [verification](../../../../benches/results/gpu/workloads/large-matrices/rtx5090/primitive-verification.json) reports integrity_pass=true and overall pass=false; the suite must not be called an all-pass result. Rental absence is independently verified.

Exact means ordered separate FP32 mul/add, FMA means ordered FP32 fused multiply-add, TF32 is the requested reduced face. The actual path column exposes its FP32 fallbacks. Library reductions have different numerical contracts; their sampled f64 errors are included in the CSV. CUTLASS is the existing pinned configuration, not a newly tuned best implementation.

The existing TF32 gate bounds Mapal's sampled error by four times cuBLAS's sampled
error. It imposes no fixed absolute accuracy threshold. The reported `maxrel_vs_f64`
is `max(abs(got-ref) / max(abs(ref), 0.001))`, with a denominator floor. For example,
the M16 expert TF32 case passes that relative gate despite large diagnostics in
all three implementations. A gate pass alone does not certify application accuracy.

| Shape / requested face | Mapal diagnostic | cuBLAS diagnostic | CUTLASS diagnostic |
| --- | ---: | ---: | ---: |
| 16K square / Exact | 2.454e-4 | 1.426e-4 | 1.426e-4 |
| 16K square / FMA | 1.426e-4 | 1.426e-4 | 1.426e-4 |
| 16K square / TF32 | 0.1043 | 0.1059 | 0.1059 |
| M16, N2048, K7168 / TF32 | 3.302 | 3.302 | 3.248 |

### 16K square

| M | Face | Actual path | Mapal ms | cuBLAS ms | CUTLASS ms |
| ---: | --- | --- | ---: | ---: | ---: |
| 16384 | Exact | FP32 mul/add | 285.6049 | 157.0908 | 197.7375 |
| 16384 | FMA | FP32 FMA | 191.6580 | 154.1048 | 194.0900 |
| 16384 | TF32 | TF32 MMA | 115.7243 | 80.9953 | 101.6248 |

### K=7168, N=18432 projection

| M | Face | Actual path | Mapal ms | cuBLAS ms | CUTLASS ms |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | Exact | FP32 mul/add | 0.3891 | 0.3523 | 0.5468 |
| 1 | FMA | FP32 FMA | 0.3850 | 0.3482 | 0.5489 |
| 1 | TF32 | FP32 FMA | 0.3850 | 0.3523 | 0.3338 |
| 16 | Exact | FP32 mul/add | 2.1309 | 0.3400 | 0.5530 |
| 16 | FMA | FP32 FMA | 2.0644 | 0.3400 | 0.5530 |
| 16 | TF32 | FP32 FMA | 2.0716 | 0.3338 | 0.3359 |
| 128 | Exact | FP32 mul/add | 1.0342 | 0.5857 | 0.6513 |
| 128 | FMA | FP32 FMA | 0.7045 | 0.5653 | 0.6370 |
| 128 | TF32 | TF32 MMA | 0.4669 | 0.3461 | 0.3523 |
| 2048 | Exact | FP32 mul/add | 16.8253 | 9.4935 | 10.2001 |
| 2048 | FMA | FP32 FMA | 12.0340 | 9.4689 | 10.0567 |
| 2048 | TF32 | TF32 MMA | 6.3826 | 4.8302 | 5.0053 |
| 8192 | Exact | FP32 mul/add | 68.6889 | 38.8884 | 44.8236 |
| 8192 | FMA | FP32 FMA | 49.3117 | 39.7947 | 44.0197 |
| 8192 | TF32 | TF32 MMA | 27.0438 | 19.9660 | 22.5966 |

### K=7168, N=2048 projection

| M | Face | Actual path | Mapal ms | cuBLAS ms | CUTLASS ms |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | Exact | FP32 mul/add | 0.2642 | 0.0451 | 0.5161 |
| 1 | FMA | FP32 FMA | 0.2540 | 0.0450 | 0.5120 |
| 1 | TF32 | FP32 FMA | value gate failed | not timed | not timed |
| 16 | Exact | FP32 mul/add | 0.2580 | 0.0512 | 0.5140 |
| 16 | FMA | FP32 FMA | 0.2683 | 0.0512 | 0.5140 |
| 16 | TF32 | TF32 MMA | 0.1516 | 0.0451 | 0.3256 |
| 128 | Exact | FP32 mul/add | 0.6328 | 0.0942 | 0.5632 |
| 128 | FMA | FP32 FMA | 0.6369 | 0.0942 | 0.5632 |
| 128 | TF32 | TF32 MMA | 0.1311 | 0.0573 | 0.3277 |
| 2048 | Exact | FP32 mul/add | 1.8350 | 0.9626 | 1.1366 |
| 2048 | FMA | FP32 FMA | 1.2083 | 0.9830 | 1.1653 |
| 2048 | TF32 | TF32 MMA | 0.7660 | 0.6717 | 0.6697 |
| 8192 | Exact | FP32 mul/add | 7.0902 | 4.1544 | 4.8220 |
| 8192 | FMA | FP32 FMA | 4.4769 | 4.0253 | 4.6418 |
| 8192 | TF32 | TF32 MMA | 2.6470 | 2.3654 | 2.4156 |

### K=4096, N=1024 projection

| M | Face | Actual path | Mapal ms | cuBLAS ms | CUTLASS ms |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | Exact | FP32 mul/add | 0.1577 | 0.0225 | 0.2990 |
| 1 | FMA | FP32 FMA | 0.1495 | 0.0225 | 0.2990 |
| 1 | TF32 | FP32 FMA | 0.1495 | 0.0225 | 0.1884 |
| 16 | Exact | FP32 mul/add | 0.1536 | 0.0184 | 0.3011 |
| 16 | FMA | FP32 FMA | 0.1475 | 0.0184 | 0.3011 |
| 16 | TF32 | TF32 MMA | 0.0860 | 0.0184 | 0.1884 |
| 128 | Exact | FP32 mul/add | 0.3564 | 0.0901 | 0.3092 |
| 128 | FMA | FP32 FMA | 0.3522 | 0.0901 | 0.3092 |
| 128 | TF32 | TF32 MMA | 0.0655 | 0.0369 | 0.1884 |
| 2048 | Exact | FP32 mul/add | 0.5407 | 0.2622 | 0.3297 |
| 2048 | FMA | FP32 FMA | 0.3727 | 0.2622 | 0.3297 |
| 2048 | TF32 | TF32 MMA | 0.2355 | 0.1925 | 0.1946 |
| 8192 | Exact | FP32 mul/add | 2.1166 | 1.0793 | 1.3312 |
| 8192 | FMA | FP32 FMA | 1.4152 | 1.1510 | 1.4316 |
| 8192 | TF32 | TF32 MMA | 0.8684 | 0.7660 | 0.7844 |

16K repeat Mapal medians are287.3385ms Exact,193.2769ms FMA and115.8799ms TF32, versus285.6049/191.6580/115.7243ms in the base. cuBLAS is faster on every 16K face; the small FMA median lead over CUTLASS does not establish a repeatable optimization gain. The explicit 16K allocation formula is5.25GiB device and3GiB host vectors; this is requested storage, not a measured process peak.

**Rejected requested-TF32 case:** M1/N2048/K7168 fails in both base and repeat, before timing. Mapal reports max relative error1.787e-3 versus cuBLAS1.248e-4 and fails the existing four-times-cuBLAS gate. The passing Contract and failed requested-TF32 PTX modules are byte-identical after renaming only the entry, with the same8×1 grid/block256/no shared memory and ordinary FP32 FMA. Contract passes its ordered-FMA bitwise gate. This establishes different gate outcomes for the same fallback arithmetic; it does not establish a compiler wrong-value defect. The stderr value“1” is a Boolean failure sentinel, not an enumerated count of mismatching sampled cells. CUTLASS's TF32 relative-error diagnostic0.2652 is preserved too. No tolerance was changed and no failed-case timing is fabricated.

The [fallback investigation](../../../../benches/results/gpu/workloads/large-matrices/rtx5090/fallback-value-investigation.md)
and its hash-bound receipt preserve the full-module comparison and the exact gate distinction.

## Small composition certification

The shared [contract](../../../../benches/results/gpu/workloads/large-matrices/preparation/composition/contract.json)
defines one matrix residual MLP and one bounded, unnormalized recurrent
attention-plus-MLP program. Every reference preserves the staged association
`p=XWo; h=ReLU(pWup+b1); m=(hWdown)+b2; y=.25*(p+m)`.
The recurrent program updates one input row and derives K/V before that projection.
It is not a complete transformer model.

The following table is the preserved pre-repair `b4152c5` baseline. Post-repair evidence
is reported separately below.

| Check | Observed result |
| --- | --- |
| Exact interpreter versus scalar binary32 reference | All 4 sources bit-for-bit equal |
| Native Contract versus actual C-fmaf-fold reference | Matrix check/run and recurrent run pass; recurrent check cannot emit |
| Rewritten GPU LLVM emission | 24 / 32 succeed; 8 fail on the same recurrent checkpoint source |
| A100 final-only recurrence | Emitted first/full and subsequent/row K/V paths, 128 versus 8 output cells |
| RTX 5090 final-only recurrence | Shipped LLVM byte-identical to both-retention-OFF control |
| GPU values and timing | Unvalidated |

The full recurrent checkpoint source triggered `replay.rs:989` with unmapped
`ObjectId(5907v1)` (exit 101) in that frozen build. Its SHA256 is
`0e4c05b403550da3a3afaeebcea8323e2af1dca887aa6a94d95c1e0dd7c57142`.
A subsequent instrumented diagnostic proved that this is an acyclic token/value exit
pair inside the existing canonical driver cone but outside its SCC. The generic
[replay repair and proof](../../../../benches/results/gpu/validation/loop-route-replay/README.md)
are now applied: reconstruction includes existing decide/advance work before feeding
loop routes. No observations or numerical rules are removed or relaxed.

The eight previously failing GPU emission configurations now complete: RTX 5090/A100 ×
Exact/Contract × shipped/full-recompute. Raw, identity-replayed and fully rewritten
interpreter runs agree on all 8,408 numeric checkpoint values and 8,838 output lines.
Five new regressions fail before the fix and pass after it; the full rewrite suite passes
86/86 and the required-clang workspace suite passes 1,181/0/1 ignored. This establishes
compiler and interpreter coverage, not CUDA correctness. Tiny native Exact/Contract
runs pass; original native Exact, Contract and one separate Contract `--no-tile` emission
each reach a bounded 40-second timeout. That backend limit remains unresolved.
The earlier no-rewrite timeout and CPU-only A100 diagnostic remain historical evidence.

Checkpoint observations change placement: the matrix check source contains 12 arena
requests and 1,496 scalar readbacks; its final-only source contains 9 arena requests
and 128 scalar readbacks. The runtime may reuse those arena requests, so they are
not counts of independent physical allocations. Both emit one direct tile launch
and two grouped launch sites. Grouping alone does not establish a speedup.

The small recurrent zero-attention control changes all eight final values, but its
largest effect is only about 1.38e-7 against a final magnitude near .0079. Final-output
tolerance alone could conceal a missing branch. Intermediate gates remain required
before external-library/PyTorch timing. Full recomputation must disable both
`retain-delta` and `retain-resident`; explicit incremental references must be labeled
separately from compiler-derived retention.

## CPU optimization inside the mixed program

The frozen RTX 5090 Contract recurrence also provides a concrete host-side example.
Using the existing bundle's Clang 22.1.8 `-O2` setting, its eight-element scalar query
loop becomes two four-lane vector calculations before the query upload. All seven
static GPU launch IDs remain in order, and all 51 static GPU runtime call sites
remain. The [original and optimized LLVM, assembly and manifest](../../../../benches/results/gpu/host-dispatch/optimization-audit/README.md)
are retained with an independent byte-copy receipt. These are compiler outputs;
they do not measure CPU/GPU overlap or a speedup.

The [call-effects audit](../../../../benches/results/gpu/host-dispatch/optimization-audit/host-gpu-call-effects-audit.md)
maps the existing hints and dataflow consumers to their source symbols. In this saved
module, GPU runtime declarations expose neither memory-effect/alias contracts nor
device-completion tokens to LLVM. LLVM adds lifetime markers for host stack slots;
those markers do not describe device-buffer lifetime. Existing CPU helper and device
kernel attributes, and a separate row-fold copy hint, are recorded precisely. This
does not establish a missed optimization: the query loop already vectorizes. Runtime
global state, host writes, retained addresses and asynchronous work rule out adding
blanket purity or lifetime annotations without proving each call's actual effects.

Current emission runs shared rewrites before splitting the host and device work.
The host then receives ordinary LLVM optimization. However,
`crates/backends/llvm/src/lib.rs:emit_detailed` disables the CPU task-parallel path
for GPU profiles, and `func/gpu.rs:gpu_site` tests device eligibility rather than
whole-program placement cost. A joint policy remains an open architecture question:
use dependency, alias, lifetime and numerical facts to establish legal placement,
retention and overlap; use measured compute, transfer and launch costs to choose
among those legal schedules. Neither shapes alone nor static last use proves
profitability or completion of asynchronous device access.

## Provenance and remaining work

The [original bounded investigation](../../../../benches/results/gpu/workloads/large-matrices/preparation/recurrent-checkpoint-root-cause.md)
records the candidate before instrumentation. The later
[diagnostic and semantic proof](../../../../benches/results/gpu/validation/loop-route-replay/PROOF.md)
binds both failed objects to that candidate, preserves the five old-code failures and
all emitted artifacts, and documents the applied generic repair. The frozen certification
index below is retained unchanged; it must not be mistaken for the post-repair result.

Both approved correctness-pilot rental attempts failed at allocation before upload or
GPU execution. [Sanitized diagnosis and cleanup](../../../../benches/results/gpu/operations/correctness-allocation-failures/allocation-diagnosis.md)
record the exact boundary; no PyTorch, ordering or composed CUDA result exists yet.

The [certification index](../../../../benches/results/gpu/workloads/large-matrices/preparation/composition/certification-summary.json)
records passes, failures and frozen-input checks. An independent root
[artifact check](../../../../benches/results/gpu/workloads/large-matrices/preparation/independent-artifact-check.json)
verified 1,464 primitive and 216 composition artifact digests. The ignored recovery
archive `.claude/worktrees/_archive-s54-20261021/large-composed-preparation/` contains
2,970 hash-verified files (159,284,379 bytes), including the temporary generators,
commands, failed outputs, native executables and full emitted evidence. It excludes
active GPU results and later PyTorch work, which require their own final archival.

The active [OpenSpec change](../../../../openspec/changes/benchmark-large-composed-gpu-workloads/proposal.md)
still owes GPU value/capacity gates, ordinary timing, independent report checks,
matched handwritten and PyTorch references, and cleanup receipts. Compiler retention
legality, runtime allocation and speed are separate proof duties. Paging and
cross-request value sharing have not been implemented or measured by these experiments.
