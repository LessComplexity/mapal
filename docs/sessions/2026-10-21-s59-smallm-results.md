# S59 — measured small-M wins and regressions

2026-10-21 America/New_York; GPU execution 2026-10-22 UTC. Starting HEAD `ce0c201`,
branch `s54-gpu-fusion`. User requested comparison results and parallel GPT-6.1-sol
work. Root ran supercharge preflight, restored S58, reviewed the packet, and retained
the existing $1 cumulative / $0.80-per-hour / $8-per-hour aggregate limits.

## Decision and measurements

**Do not promote the blanket small-M candidate.** Wide M16 Exact/Contract/TF32
improve 4.82–5.18×, but the two smaller M16 Exact/Contract families regress37–47%.
All seven changed cells exceed observed identical-code variation (factor1.028579).
Twenty-one admitted cells are unchanged-PTX controls, not compiler improvements.

| M × N × K | Face | Old ms A/B | Candidate ms A/B | cuBLAS ms A/B | CUTLASS ms A/B |
| --- | --- | ---: | ---: | ---: | ---: |
| 16 × 18432 × 7168 | Contract/FMA | 2.7249 / 2.7423 | 0.5673 / 0.5673 | 0.4444 / 0.4444 | 0.7475 / 0.7491 |
| 16 × 18432 × 7168 | TF32 | 2.7310 / 2.7369 | 0.5284 / 0.5263 | 0.5059 / 0.5079 | 0.5100 / 0.5100 |
| 16 × 2048 × 7168 | Contract/FMA | 0.3236 / 0.3236 | 0.4444 / 0.4444 | 0.0655 / 0.0655 | 0.7206 / 0.7209 |
| 16 × 1024 × 4096 | Contract/FMA | 0.1758 / 0.1761 | 0.2580 / 0.2580 | 0.0266 / 0.0266 | 0.4116 / 0.4116 |

These are ordinary CUDA-event kernel medians, 30 samples per leg, two ABBA process
observations. No confidence intervals or identical library association are claimed.
See [the full report](../performance/s59-smallm-comparison.md) and
[all 30 cells](../../benches/results-s54-gpu/s59-smallm-comparison/analysis/ABSOLUTE.md).
Among28 admitted cells, cuBLAS wins25 and3 are unresolved; candidate wins0. Against
tested pinned CUTLASS kernels, candidate wins21, loses6 and1 is unresolved. Counts
include unchanged controls. These are not claims about every CUTLASS configuration.

## Correctness, tests and evidence

All234 records are present:218 passing,4 numeric failures,12 quarantined skips.
There are106 passing numerical processes and112 timed processes. Two M1 TF32 timing
cells are excluded because both revisions fail the bounded full-output diagnostic
for dense and2048-column expert shapes. Four quarantine keys mean two bounded gates
and two timing cells. Large timing shapes check256 sampled outputs; bounded gates
check all outputs. The existing4×cuBLAS TF32 diagnostic is not an absolute guarantee.

Independent analysis verifies2,866 payload files,948 outer files, two binaries and
three negative controls. Root independently rehashes948 files and recomputes10,080
raw event samples, medians and same-process library ratios. The runner's numerical
`candidate_acceptance=true` does not approve its performance regressions. Overall
exit1 records the two excluded cells; inventory is complete and no fatal error occurs.

Root's16 launcher tests and10 adapter tests pass; analyst's eight mutation tests
pass. An independent GPT-6.1-sol reviewer finds no factual/numerical blocker in the
final report. Strict OpenSpec:7 passed/0 failed. Drift:0 dead/571 references.
`git diff --check` passes. Graph refreshed:9,137 nodes/19,815 edges from1,243 files.
No production source changed, so the existing shared1,188/0/1-ignored workspace
result is not misrepresented as rerun. The isolated candidate's prior1,190/0/1
remains S58 evidence. The frozen emitter was untouched.

Public artifacts are under `benches/results-s54-gpu/s59-smallm-comparison/`:
raw trace, tables/CSVs, analysis, root checker, hashes, budget and cleanup receipts.
Public archive:949 files/4,764,382 uncompressed bytes, all rehashed against source.
Ignored private recovery:3,875 files/71,486,461 uncompressed bytes, also rehashed;
`recovery.json` binds its path and hash. Private archive mode0600, parent0700.
Public evidence excludes provider API/create/request/wrapper logs and passes a
provider-credential pattern scan. A CLI DNS-error diagnostic exposed a Vast key in
tool output during preparation; user notified. Rotation remains outstanding.

## Rental and live state at close

| Item | State / handle | Inspection / continuation |
| --- | --- | --- |
| Owned RTX5090 | instance57130775, machine146107/host213578; independently absent2026-10-22T00:12:24.048049Z | public `cleanup.json`; never rerun consumed launcher |
| Lease | create00:09:15.867548Z; absolute deadline00:20:25.867543Z; closed after188.181s | actual40GB-inclusive rate$0.7051851852/hour |
| Benchmark | complete in140.359s,480s cap | raw trace + independent analysis |
| Budget | additional host estimate$0.0368617 plus$0.10 reserve; cumulative$0.9036599532; remaining$0.0963400468 | planning, not invoice; old default-storage quote uncertainty disclosed; no new rental inferred |
| Operator | `/private/tmp/mapal-s59-gpu/`; no live job | `budget-closed.json`, consumed claim; private API diagnostics stay private |
| Analyst | `/private/tmp/mapal-s59-analysis/RESULTS/`; complete | frozen manifest + public copies |
| Root checks | `/private/tmp/mapal-s59-root/`; complete | `verification.json`, drift/validation logs |
| Repository | only checkout `s54-gpu-fusion`; local evidence commit follows this log | no push; preserve unrelated `oainotes.md` |
| Protected rentals |57065574 and57100404 unrelated | never modify or destroy |

The unchanged sealed payload SHA is
`6f4535d08a49cc92ad8ffdd7863e814e7343cbbf6a916105ad864bc85a48b9b3`.
The root-bound packet SHA is
`660da9f084b71c4332cea52bb6c36a9b7a45473f0fe3e325d6077bf9494cdd0e`.
GNU `time` was explicitly installed and checked before benchmarks. No PyTorch install,
profiler, executed-image capture or other GPU work ran alongside the comparison.

## OpenSpec and next work

| Change | Completed / total | Remaining boundary |
| --- | ---: | --- |
| optimize-ordered-smallm-gpu |4/7| refined generic selection, executed images/large controls, integration acceptance |
| capture-executed-cuda-images |6/8| actual GPU injection/capture proof |
| benchmark-large-composed-gpu-workloads |5/14| matched composed timing and remaining correctness/scale gates |
| eliminate-unmaterialized-gpu-allocations |9/15| prepared ordinary GPU/counter gate before allocation removal |

No change is archived. Implementation map and architecture reviewed; no promoted
model or runtime change. Component/root status, mutable continuation and comparison
index reconciled. §4.5 review is in
`docs/components/backend-nvptx/reviews/review-s59-smallm-comparison.md`.

Use generic grid/resource costs to investigate the next small-M selection. Runtime
cooperative Exact/Contract grids are576/64/32 blocks against170 SMs. The small grids
bound simultaneously occupied SMs; this does not establish the whole regression's
cause. Capture actual images/counters before attributing that cause. Keep scalar
alternatives and neighboring-shape tests; do not fit a model-name or three-point gate.
Allocation removal, retention, CPU/GPU joint placement, PyTorch composed timing and
16K results remain distinct work. Existing arena guard and numerical policies stay.

## Exact resume / inspection commands

```sh
cd /Volumes/LessComplex/Personal/mapal
supercharge-preflight
cat docs/next-session.md
openspec list --json
cat docs/performance/s59-smallm-comparison.md
cat benches/results-s54-gpu/s59-smallm-comparison/budget-closed.json
python3 -B benches/results-s54-gpu/s59-smallm-comparison/root-recheck.py \
  --trace /private/tmp/mapal-s59-gpu/retrieved/trace/smallm-s59 \
  --analysis benches/results-s54-gpu/s59-smallm-comparison/analysis/analysis.json \
  --out /private/tmp/mapal-s59-root-rerun.json
```

If temporary files disappear, follow the public evidence README and `recovery.json`;
verify archive hashes, extract locally, and pass explicit recovered paths. Inspection
requires no rented machine. Never run old one-use launchers to recover results.
