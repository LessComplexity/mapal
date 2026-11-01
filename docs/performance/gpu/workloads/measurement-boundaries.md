# Composed measurement boundary audit

The captured wall clocks do include GPU completion. I found no proved missing-sync, wrong-unit, or output-copy-in-clock accounting defect. They measure a warmed compute region, not full request latency or pure device kernel time. Attribution of the performance gap is not established by these measurements.

The frozen link command links `matrix/emission/host.o`, `mapal_wrapper.c`, `runtime/mapal_gpu.c` and `libmapal_rt.a`; it does not use a linker `--wrap` timer. Wrapped emitted IR calls `mapal_bench_time_ms`. The underlying `mapal_time_ms` at frozen source head `3a0d4350dc2f5b13b53ed86821cf427922fe99d9`, `crates/mapal-rt/src/lib.rs:514–519`, returns process-lifetime `Instant` elapsed milliseconds. The GPU runtime's separate `now_ms` trace timer is not that function.

## Exact boundaries

- Mapal source `matrix/final-only.mapal:2–13` generates six inputs before t0 at14; t1 is39, before printing/readback. Emitted `matrix/emission/callable.wrapped.ll:1086–1109` allocates one arena and generates inputs; sync1110, start1111, compute1136/1161/1191, sync1200, stop1201, arena free5318. `runtime/mapal_gpu.c:230–244,256–257` returns an existing nonzero arena slot, so arena requests inside the region do not repeat device allocation. **Input generation, initial arena allocation and final free are excluded from measured wall.**
- `mapal_wrapper.c:9` reads host start before recording begin. Line10 records/synchronizes end, then reads host stop; event elapsed and stderr reporting follow the host read. First-use event creation is in rep0, discarded with the five warmups. Emitted stop sync invokes `cuCtxSynchronize`, checked by CK (`runtime/mapal_gpu.c:578–586`). No asynchronous early-stop defect is visible.
- Original C++ reference `matrix_reference.cu:24–29` constructs/uploads/preallocates before validation and ACK. It primes events, then clocks start-record/work/stream-sync/end-record/end-sync; final-output readback follows wall stop. Native `composed_native.cu:59–79` likewise performs input load, tensor/workspace setup, two prerequisite graphs and numerical ACK before clocks. Its final graph has the same host/event placement; cleanup is after timing at82.
- Torch `matrix_torch.py:14–26,33–47` constructs resident inputs, separately compiles/checks checker and final-only graphs, primes events, and then clocks `run(*args)` between a synchronized start and completed end. Output observation and `del outputs` follow stop. Callable tensor creation/caching-allocator work can occur inside `run`; Mapal's preallocated arena and native preallocated buffers differ from that allocation strategy.

All three event clocks record the end **after** a completion synchronization. Thus events include the synchronized region's submission gaps and completion wakeup/end-enqueue gap; they are not isolated kernel sums. Mapal context synchronization versus C++ stream synchronization and Torch device synchronization are distinct APIs. Error-checked synchronizations and final output validation prevent treating a merely submitted graph as complete.

## Available paired upper medians

Each cell has 30 retained observations after five warmups. Units below are microseconds, wall/event. These are the same-host original controls from attempt10, not a cross-host substitution.

| Shape M/d/h | Mapal | Naive CUDA | cuBLAS | Torch eager | Torch compiled |
|---|---:|---:|---:|---:|---:|
|128/256/512|73.090 / 66.976|54.452 / 49.792|39.069 / 34.784|115.424 / 101.824|178.659 / 163.488|
|128/1024/2048|128.935 / 123.008|333.153 / 328.672|92.890 / 88.288|140.602 / 123.552|208.715 / 191.392|

Native selected Lt unfused/fused medians are36.255/31.808 and32.659/28.128 microseconds at the smaller shape;63.416/58.784 and59.620/55.072 at the larger shape. Native selected CUTLASS unfused/fused medians are47.192/42.720 and46.711/41.920;127.954/123.328 and123.867/119.072 respectively. Full control/default/selected rows remain in `native-medians.json`.

The large eager Torch retained sample includes a72.474ms wall/72.453ms event outlier. It is preserved, not trimmed or causally attributed. Upper medians and wall-minus-event differences do not isolate a particular runtime cost.

## Fairness and limits

Inputs are bit-matched immutable fixtures, but cache preparation is not identical: Mapal regenerates inputs outside every timed call; native/Torch reuse resident buffers. No forced L2 flush, common empty-region calibration, matched GPU clock locking, randomized/interleaved arm order, or independent-host replication was performed. Native ran first, then Torch bootstrap, then original controls; within the original block Mapal precedes naive, cuBLAS, eager and compiled. Equal naive/cuBLAS controls between blocks are useful observations, not a drift confidence interval.

Mapal Contract permits FMA while preserving increasing-K order. Libraries are admitted under the fixed dual-reference componentwise bounds and may reorder reduction; IEEE/PEDANTIC does not grant ordered-bit equivalence. Native/Torch differ in framework submission and allocation strategies. These are disclosed realizations of the same graph with different arithmetic permissions; the table is not an isolated comparison of identical reduction order or allocation-free kernel instruction efficiency.

## Selected Lt attributes

The selected unfused and fused configurations report the same following p/h/m attributes. All queries return status0; reduction scheme is numeric enum2, not an inferred dispatch. Each process allocates a32MiB explicit workspace cap. Required/checked workspace is shown below.

| Shape | Site | Eligible index | Split-K attribute | Reduction | Workspace bytes |
|---|---|---:|---:|---:|---:|
|128/256/512|p|0|4|2|524288|
|128/256/512|hraw or h-bias-relu|1|4|2|1048576|
|128/256/512|mraw or m-bias|0|8|2|1048576|
|128/1024/2048|p|0|10|2|4194304|
|128/1024/2048|hraw or h-bias-relu|3|5|2|5242880|
|128/1024/2048|mraw or m-bias|0|10|2|5242880|

These are actual selected opaque algorithm attributes passed to Lt (`native_plans.hpp:46–60,71–76`), bound to the fresh confirmation receipts. Duplicate opaque aliases exist, so extraction matches the selected supported-filter index as well as identity. Actual kernel dispatch/SASS and reduction execution were not captured. No causal speedup or numeric cost is inferred from split-K/workspace attributes. IR TileKSplit address decomposition must not be called parallel split-K reduction without separate proof.

[Audit receipts and paired medians](../../../../benches/results/gpu/workloads/residual-mlp/measurement-boundary-audit/audit.json) · [Whole-program comparison](fused-library-comparison.md) · [Existing IR facts and GPU restrictions](../disassembly/ir-facts-and-realization-gaps.md)
