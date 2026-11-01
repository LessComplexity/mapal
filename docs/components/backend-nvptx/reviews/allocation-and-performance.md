# allocation and performance evidence review

Historical provenance: S61.

The production change is the reviewed `FnEmit::chain_prologue` storage omission and its
regressions. Geometry alternatives and benchmark harness changes remain outside production.
See [implementation](../IMPLEMENTATION.md) and [evidence](../../../performance/gpu/current-results.md).

| FRAMEWORK §4.5 duty | Evidence and boundary |
| --- | --- |
| 1. Placement honesty | Existing `Dat Epi.store_out=false` means the selected grouped launch does not write the raw array. `Trn chain_prologue` omits only its `Gmem` allocation. Surviving pointers and live member allocation positions match the reviewed before emission. |
| 2. Transmission well-typing | No runtime ABI, copy extent, pointer type, `Loc` or `Trm` changes. Twenty-eight device LLVM units remain byte-identical. |
| 3. Placement totality | Ungrouped sites and stored raw outputs keep the existing allocation path. Stored-raw/fork controls preserve their eight allocations; the single-chain arena preserves its one allocation. |
| 4. Dependency mediation | Existing grouped physical-writer stream selection, event edges, chain-aware readback and legacy veto remain. `mapal_gpu_free` returns early for the zero slot, so the removed incidental wait is real and required a separate ordinary gate. The 640 observed processes pass; this is not universal race freedom. |
| 5. Composition soundness | The change consumes the existing selected group fact rather than constructing another ownership map. Independent, diamond, two-chain, raw-observed/fork and loop cases exercise composition. The stream/arena exclusion remains intact. |
| 6. Multiple realizations | Exact and Contract share the generic storage admission and both pass the tested output/counter inventory. Forced streams are distinct from default RTX 5090 streams-OFF; no default latency claim follows. |

The frozen patch SHA is `2c8cdc1964abbd0516a46c5da1e02a76ba59e214e205275c8e478dc23e85ffd5`.
An independent agent confirms the integrated two-file diff matches it after Git metadata
and hunk-context normalization. Root independently verifies all640 ordinary records and28
separate counter records, including raw hashes, process status and observed values.
The full required-clang workspace and host-access backend recheck establish1,190 distinct
passing tests with one ignored; the original three native-detection failures remain recorded.

No latency evidence is admitted from the synchronizing counters. A performance test must
compare the same forced-stream realization, including allocation/free and completion costs,
with ordinary clocks and controls. That OpenSpec obligation remains open. Retention/paging,
joint CPU/GPU scheduling and architecture-wide profitability are not solved by this omission.

The first composed run's displaced stdout sample markers are a proved harness failure:
real pipe replay reproduces the raw ordering, and a flush before the request repairs it.
No generated kernel, numerical gate or timed region changes. The independent reader must
bind the new manifest, retain earlier failures and admit only newly complete timing rows.
Actual runtime results and limitations are recorded in the living S61 evidence report.
