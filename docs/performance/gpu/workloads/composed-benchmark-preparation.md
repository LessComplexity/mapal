# Composed benchmark preparation and review

Historical provenance: S60.

**GPU execution is pending the requested budget extension.** No S60 rental, upload,
performance result or production compiler change has occurred. The S59 measured
decision remains: keep the blanket small-M patch isolated because its wide-shape
wins accompany smaller-shape regressions. See [the measured table](../kernels/small-m-geometry-comparison.md).

Three GPT-6.1-sol workers prepared separate kernel, allocation/operator and composed
workstreams. Root reviewed the code, numerical contracts, emitted clocks and final
26,444,972-byte batch. Source/Cargo files at `bc7af76` still match the ordering-corrected
`3a0d435` compiler. Frozen emitter and unrelated `oainotes.md` remain untouched.

| Item | Completed local evidence | Hardware obligation |
| --- | --- | --- |
| Composed residual MLP | Two M128 shapes, d256/h512 and d1024/h2048; Mapal, naive CUDA, cuBLAS, PyTorch eager and Inductor harnesses; separate checkpoint/final graphs; nine numerical/protocol/cancellation checks | Compile native CUDA, validate values, collect 30 admitted observations per implementation/shape |
| Allocation elimination | Reuse reviewed S58 v2: unchanged device code and isolated host-storage omission; retrieval path repaired and byte-verified for successful/partial runs | 640 ordinary correctness processes and 28 separate counter processes; no latency admission |
| Kernel alternatives A/B | Each passes 918,000 geometry comparisons; 57 emitted modules per alternative; actual caller/rung coupling recorded | Optional old/S58/B comparison of four affected M16 cells, with full bounded and actual sampled gates, raw events and nulls |
| Executed image capture | Four frozen S59 old/candidate Contract cases; callback tests and real nested cancellation/spawn-race checks | First live observer smoke, then exact input/context/function/launch joins and disassembly; separate from timing |
| Independent readers | Raw-output numerical/timing reader and separate kernel/image readers with mutation controls | Run on retrieved artifacts; retain partial results and rejected cells |

## What the kernel investigation establishes

`crates/backends/llvm/src/func/gpu.rs:launch` calls
`crates/backends/llvm/src/profile.rs:TargetProfile::gpu_tile` with `wide=false, fill=false`
to derive the `Fill` key before `rungs_at`. It calls the helper again after choosing
wide/fill rungs. Changing helper admission therefore changes more than final tile
dimensions: it can change the selected arithmetic route through the existing policy.

Alternative A admits newly legal post-shrink row fits only when they cover an SM wave.
Its preliminary fill key restores the old scalar route for the wide Contract/TF32 cases,
so it does not preserve the S58 improvements. Alternative B tries row fitting before
occupancy shrinking, only after the unchanged rule rejects a shape. It produces a
different legal geometry for the smaller M16 cases, while retaining the old dense
fallback. These are alternatives to the same helper, not cumulative patches.

Both preserve every previously successful geometry in the tested inventory. A admits
9,000 new legal cases; B admits 13,208. This is finite legality evidence, not a
profitability theorem. Of 57 RTX 5090 modules, A preserves 55 old device LLVM/PTX units
and B preserves 49. B changes the two smaller families' M16 Exact/Contract bounded and
actual cases. Fifteen focused backend/panel tests pass per alternative. A broader backend
run has 58 passes and three native macOS detection failures; the unchanged-profile
control has the same failures. No all-pass workspace result is claimed for these experiments.

The next selection experiment must evaluate complete launch alternatives with numerical
face and rung context known. Grid size and resource ceilings can constrain possibilities;
they do not establish actual occupancy, traffic or the cause of a timing regression.
The existing [OpenSpec change](../../../../openspec/changes/optimize-ordered-smallm-gpu/design.md)
keeps measured admission and neighboring/large controls open.

## Composed numerical and measurement contract

Both programs compute `p=XWo; h=ReLU(pWup+b1); m=hWdown+b2; y=(p+m)/4`.
All dimensions and inputs are synthetic. The d256 oracle evaluates all 128 rows;
the larger oracle evaluates complete rows 0, 42, 84 and 127 with the full d1024/h2048
reduction lengths. Each published stage has 256 selected cells. Other large output
cells are unvalidated. The larger scalar references and bounds completed in 31.127 s.

Every gate uses both operationwise FP32 and true C `fmaf` references with the existing
componentwise `gamma(2n)` rounding/propagation derivation. Mapal Contract and naive CUDA
also require ordered-FMA bits. No fitted tolerance is introduced. Full tiny intermediate
and final values precede representative work. Representative checkpoint graphs are
separate from final-only timing graphs because observations can change materialization.
The final v3 packet exports raw PyTorch checker values for independent verification.

The measured region is repeated identical resident-input work: five warmups, then
30 observations. Each implementation synchronizes before wall/start-event, executes
the work, synchronizes completion, records/synchronizes the end event, then reads
wall time. Event elapsed includes host submission and completion-wakeup gaps. It is
not a sum of kernel durations. Compilation, input preparation and output observation
are outside these clocks; cold calls/process wall are retained separately.

Root independently checked each emitted final-only host program: six input-generation
sites precede timing; one direct and two grouped work launches occur inside; there is
no generation, D2H or free inside. Mapal regenerates immutable inputs between calls,
while references retain them, so cache state is not identical and no cold-L2 claim follows.
An external C repetition loop avoids a preserved `L1901 TokenNotLinear` failure in a
small source loop containing timer calls. Device code is unchanged by the clock wrapper.

| Storage fact | d256/h512 | d1024/h2048 | Meaning |
| --- | ---: | ---: | --- |
| Mapal arena request | 1,838,080 B | 22,032,384 B | Arena first use before clock, free after observation |
| Explicit CUDA buffers | 2,493,440 B | 25,702,400 B | Inputs and intermediates preallocated and retained across calls |
| PyTorch | Pending GPU observation | Pending GPU observation | Input upload outside; intermediate allocator calls inside; allocated/reserved snapshots and peaks outside clock |

Requested bytes are not measured driver peaks or cuBLAS workspace. This comparison
cannot establish allocation-elimination speedup. The separate allocation gate forces
streams, whereas shipped RTX 5090 selection disables streams. Its counter results must
not be attributed to the default composed mode. PyTorch uses eager or Inductor
fullgraph/static, TF32/autocast disabled and CUDA graphs off; this is an explicit
restricted configuration, not a best-possible-PyTorch claim. Independent-branch,
representative recurrence, retention/full-recompute and graph-enabled comparisons remain open.

## Review, preservation and next execution

Root review repaired evidence retrieval outside `/w/trace`, unmatched closing-event
ordering, raw checker-value omission, and child-process cleanup on cancellation.
Real local tests cover nested descendants, an exited leader, and cancellation between
spawn and assignment. Successful/partial allocation retrieval preserves 1,528/728
CPU-stub artifacts with byte hashes. These tests establish local runner behavior;
they are not GPU evidence.

The final packet is bound by SHA256
`b37f14727ac878394046d520e83ef60c1679f1efe758e9ad37cf4dac20a12432`.
The reviewed input archive and public preparation receipt are recorded under
`benches/results/gpu/workloads/residual-mlp/preparation/`. The ignored private preparation
archive preserves 885 files, SHA256
`8f52deba863af578bdcabe9f9447f5abbf3a6f00d0ce61538a454aa3f0b34903`.
Raw provider/API diagnostics are excluded from public evidence.

The pending request is one attempt on an eligible RTX 5090, at most $0.80/hour and
55 minutes from creation through cleanup, at most $1 additional/$2 cumulative and
$8/hour aggregate. The 03:13 UTC quote reserves $0.66019 including a $0.10 transfer
allowance; it is not an invoice and must be refreshed before launch. The previous
$1 cap has $0.09634 remaining. No approval is inferred from the pending request.

After explicit approval, refresh the quote, bind root review to the unchanged payload
and fresh packet, then run the one-use launcher. Phase order is allocation, explicit
PyTorch bootstrap, both composed sizes, optional B comparison, then optional capture.
The shared deadline can skip later phases. Retrieve and independently verify each
successful or failed artifact before interpreting results. Never modify unrelated
instances 57065574, 57100404, 57126023 or 57133432.
