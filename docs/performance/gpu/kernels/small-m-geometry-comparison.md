# Small-M cooperative geometry comparison

Historical provenance: S59.

2026-10-21 local / 2026-10-22 UTC. **Keep the candidate isolated.** The generic
geometry change improves the widest M16 projection by 4.82–5.18× but makes the
two smaller M16 Exact/Contract projections 37–47% slower. All seven differences
exceed the observed identical-code control variation. No production selection changed.

## Measured results

RTX 5090, ordinary CUDA-event kernel times in milliseconds, lower is better.
Each range spans two process medians in ABBA order, with 30 raw events per leg per
process. These are descriptive ranges, not confidence intervals. Libraries are
measured in the candidate processes. Exact checks ordered separate multiply/add;
Contract checks ordered `fmaf`; requested TF32 retains the frozen numerical diagnostic.
The libraries do not promise Mapal's exact association. CUTLASS means the pinned
kernels tested by this harness, not every possible CUTLASS configuration.

| M × N × K | Face | Old Mapal ms | Candidate ms | cuBLAS ms | CUTLASS ms | Candidate vs old |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 16 × 18432 × 7168 | Exact | 2.7720 | 0.5673 | 0.4444 | 0.7472–0.7475 | 4.89× faster |
| 16 × 18432 × 7168 | Contract/FMA | 2.7249–2.7423 | 0.5673 | 0.4444 | 0.7475–0.7491 | 4.82× faster |
| 16 × 18432 × 7168 | TF32 | 2.7310–2.7369 | 0.5263–0.5284 | 0.5059–0.5079 | 0.5100 | 5.18× faster |
| 16 × 2048 × 7168 | Exact | 0.3154 | 0.4444 | 0.0655 | 0.7209 | 41% slower |
| 16 × 2048 × 7168 | Contract/FMA | 0.3236 | 0.4444 | 0.0655 | 0.7206–0.7209 | 37% slower |
| 16 × 1024 × 4096 | Exact | 0.1823 | 0.2580–0.2597 | 0.0266 | 0.4116 | 42% slower |
| 16 × 1024 × 4096 | Contract/FMA | 0.1758–0.1761 | 0.2580 | 0.0266 | 0.4116 | 47% slower |

The [full 30-cell table](../../../../benches/results/gpu/kernels/small-m/cooperative-comparison/analysis/ABSOLUTE.md)
and [CSV](../../../../benches/results/gpu/kernels/small-m/cooperative-comparison/analysis/absolute-cells.csv)
include M1, M16 and M128 for all three families, Exact/Contract/TF32, and 4096-square
controls. Twenty-one admitted cells have unchanged PTX and are controls, not gains.
The global unchanged-code variation factor is 1.028579 (about 2.86%).

Across the 28 admitted cells, cuBLAS is faster beyond that observed variation in
25 and unresolved in 3; the candidate has **no established cuBLAS win**. Against
the tested CUTLASS kernels, the candidate wins 21, loses 6 and is unresolved in 1.
These counts include unchanged controls and do not count 21 new compiler improvements.
The improved wide Exact/FMA kernels beat tested CUTLASS by about 1.32×, while cuBLAS
still takes only 0.4444 ms versus Mapal's 0.5673 ms.

## Correctness and independent verification

All 234 record identities are present: 218 passing processes, four numerical failures
and twelve quarantined skips. There are 106 admitted numerical processes and 112
timed processes, yielding **28/30 admitted timing cells**. The two excluded cells
are M1 requested-TF32 for N×K = 18432×7168 and 2048×7168. Both old and candidate
fail their bounded full-output gates; neither receives a timing comparison.
The four quarantine keys are two bounded gates and two dependent timing cells.

The failures preserve the existing 4× cuBLAS diagnostic: bounded dense M1 has Mapal
maximum relative error 1.109e-4 versus cuBLAS 2.117e-5; bounded expert M1 has
9.399e-4 versus 1.085e-4. Each reports one mismatching output. Exact/Contract
bounded gates check every output against the recorded ordered fold. Actual large
timing shapes check 256 sampled outputs; no exhaustive large-matrix claim follows.
TF32's relative diagnostic is not an absolute accuracy guarantee.

The independent analyst verified 2,866 sealed payload files, 948 outer files, both
harness binaries, every record, and all three negative controls. Root separately
rehashed all 948 outer files, reparsed **10,080 CUDA-event samples** from 112 processes,
recomputed their medians and same-process library ratios, and checked statuses and
numerical pass markers. See [root verification](../../../../benches/results/gpu/kernels/small-m/cooperative-comparison/root-verification.json)
and the [complete independent verdict](../../../../benches/results/gpu/kernels/small-m/cooperative-comparison/analysis/VERDICT.md).

The runner's original `candidate_acceptance=true` field certifies only its numerical
and protocol scope. It does not reject performance regressions and is **not promotion
approval**. The whole run correctly exits 1 because two requested cells remain excluded;
there is no fatal execution error or incomplete inventory.

## Runtime facts and next proof obligations

For changed Exact/Contract cells, runtime resources move from 256-thread scalar
blocks to 32-thread cooperative blocks; shared memory changes from zero to
1,536/1,664 bytes and registers from 40 to 76–80. Local bytes remain zero.
The wide TF32 cell changes to 64 threads, 10,752 shared bytes and 48 registers.
Saved PTX routes are cooperative Exact/Contract and MMA for wide TF32.
Actual executed cubins/SASS and dynamic hardware counters were not captured.

The candidate's cooperative Exact/Contract grid is 576 blocks for the wide shape, 64 for the
2048-column shape and 32 for the 1024-column shape, against
[170 reported SMs](../../../../benches/results/gpu/kernels/small-m/cooperative-comparison/device-facts.json).
Thus the latter two can occupy at most 64 and 32 SMs simultaneously with this grid.
That is a geometry bound, not proof that it causes the entire measured regression.
The next generic selection experiment should use grid size, SM count, resident
blocks, registers and shared-memory costs. A threshold fitted only to these three
shapes would not establish a general rule. Keep scalar and cooperative alternatives
until their legal and profitable regions are measured. No model-name gate or
numerical-policy change is justified by this run.

Remaining [small-M OpenSpec duties](../../../../openspec/changes/optimize-ordered-smallm-gpu/tasks.md):
executed images, large controls, a refined generic candidate and regression checks
before integration. The separate allocation candidate still needs its ordinary GPU
and allocation-counter gate. No allocation, retention or composed PyTorch timing gain
was measured here. Previous 16K results remain in the [large-matrix report](../workloads/large-matrices-and-mlp-validation.md),
separate from this host/run.

## Execution, cost and recovery

Instance **57130775**, machine 146107 / host 213578, was independently confirmed
absent at **2026-10-22 00:12:24.048049 UTC**, before its 00:20:25.867543 deadline.
The benchmark packet ran for 140.359 seconds. Create-to-confirmed-absence was
188.181 seconds. Requested 40 GB storage is included in the $0.7051851852/hour quote,
giving a host-time estimate of $0.0368617. With the additional $0.10 transfer reserve,
cumulative planning usage is **$0.903660**, leaving **$0.096340** under the $1 cap.
These are planning figures, not an invoice. Prior receipts used default-storage
quotes; their retained transfer reserves are not itemized billing. No further rental
is inferred from the remaining estimate.

GNU `time` is now installed and verified before benchmarks. Root's 16 launcher and
10 adapter tests pass. No profiler, PyTorch installation or other GPU job ran alongside
the ordinary comparison. The frozen timing section and reviewed payload are unchanged.
Shared source and the frozen emitter are untouched.

[Public evidence, raw trace and verifier scripts](../../../../benches/results/gpu/kernels/small-m/cooperative-comparison/README.md)
and the [recovery receipt](../../../../benches/results/gpu/kernels/small-m/cooperative-comparison/recovery.json)
preserve the results. Private recovery includes the sealed payload. Provider API records
are excluded from public evidence. A CLI diagnostic exposed a Vast credential in tool
output during preparation; the user was notified and rotation is outstanding.
