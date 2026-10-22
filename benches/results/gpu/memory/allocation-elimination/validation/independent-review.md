# Narrow allocation integration review

Verdict: no static blocker found for the reviewed `store_out` allocation omission. The current two-file diff matches frozen `/private/tmp/mapal-s58-allocation/evidence/candidate.patch` after removing only Git index lines and hunk function-context suffixes. Patch SHA256 `2c8cdc1964abbd0516a46c5da1e02a76ba59e214e205275c8e478dc23e85ffd5`; source hashes and verifier receipt hash are in `allocation-review.json`. No code edits or test reruns performed by this reviewer.

The changed Trn is `crates/backends/llvm/src/func/resident.rs:FnEmit::chain_prologue` (2149): it omits eager `dalloc` only when an existing selected epilogue group has `store_out=false`. A missing group remains allocated. The Dat predicate is already computed from outside readers in `FnEmit::epilogues` (859), and `FnEmit::emit_site_resident` (2013) already uses it to pass output pointer zero and omit the raw output from `stored`. Stored epilogue members still receive their own allocations (2055 onward). The patch does not create a new liveness predicate or change device arithmetic.

The Dat slot is zero-initialized by `FnEmit::dslot` (1619). `FnEmit::resident_after` (1854) and `resident_teardown` (2265) can still emit free calls for an unallocated slot. At the runtime Loc, `crates/mapal-rt/gpu/mapal_gpu.c:mapal_gpu_free` (553) returns immediately for `*slot==0`; it neither frees an invalid pointer nor treats a zero slot as a completion fence. This means the removed allocation can indeed remove its former incidental nonzero-free synchronization. I do not infer ordering safety from pointer non-use alone.

Existing explicit Trm dependencies remain: `ResidentPlan::stream_of` (301) canonicalizes aliases and resolves grouped members through `grouped_writer`; `emit_site_resident` (2022 onward) emits cross-chain events from that producer mapping; `FnEmit::d2h` (2111) selects the producer's chain-aware copy. `mapal_gpu_d2h_on` (388) queues that copy on the named stream and calls `cuStreamSynchronize`. `FnEmit::chains` (1321) retains the conservative legacy-consumer veto and moves frees beyond the last chained launch (1447 onward). These sites are outside this diff.

For actual remaining allocations, the runtime completion guard remains intact: `launch_on` marks nonzero-stream work through `chains_busy=1` (351), and the next nonzero `mapal_gpu_free` calls `cuCtxSynchronize` before `cuMemFree` (560). The patch does not strengthen that guard, and a zero free does not run it. This supports the unchanged explicit dependency contract without promising universal asynchronous race freedom.

Root's supplied `/private/tmp/mapal-s61-root/allocation-verification.json` reports640 ordinary before/after processes and28 separate counter processes passing, with `timing_admissible=false`; this reviewer read and hash-bound the receipt rather than rerunning those gates. The observed per-invocation counters are:

| Case | Before allocations/bytes/frees | After allocations/bytes/frees |
|---|---|---|
| independent | 7 /114688 /7 | 6 /98304 /6 |
| diamond | 8 /131072 /8 | 7 /114688 /7 |
| mlp2 | 12 /196608 /12 | 10 /163840 /10 |
| omitted-raw | 8 /131072 /8 | 6 /98304 /6 |
| single-chain | 1 /81920 /1 | 1 /81920 /1 |
| materialized-raw | 8 /131072 /8 | 8 /131072 /8 |
| forked-tile | 8 /131072 /8 | 8 /131072 /8 |

These are observed allocation requests/frees, not driver peak memory or latency. The GPU gate forces streams; default RTX5090 composed selection disables them, so no default composed timing benefit follows. Existing S58 evidence reports28 unchanged device LLVM units and host slot/order equivalence; that preparation was not rerun here.

The supplied workspace receipt reports1187 passes/3 native-detection failures/1 ignored in the required-clang sandbox run, then59/0 in a complete backend library rerun with host access resolving those three failures. This is not described as a full unsandboxed workspace rerun.
