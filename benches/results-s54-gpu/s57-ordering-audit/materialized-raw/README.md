# Materialized raw output variant: current emitted ordering evidence

This new temp-only fixture adds four late raw-output corner reads to the original [stored-member fixture](../stored-member-ordering.mapal). No compiler/runtime/source/bundle changes were made. Both faces emit using the already built frozen tools; the exact argv and exit0 receipts are in [commands.json](commands.json). The fixture SHA256 is `8728dc8f12ca48f54dd6cb5bd4a2ac811f1ebaa71b7418a93955ff57d1cdedcb`.

The late raw reads force both base results to remain materialized. The frozen compiler still groups each doubling map into its base tile and places the two launches on streams1/2. Both raw output pointers and both stored member pointers are real allocations. Both device kernels use `%o` and `%y0` in their bodies. This is a current compiler emission, with no hypothetical dead-allocation change.

Both [exact.host.ll](exact.host.ll) and [contract.host.ll](contract.host.ll) have this same layout:

| Evidence | Line |
|---|---:|
| First grouped launch, stream1 | 136 |
| Last grouped launch, stream2 | 157 |
| First h2 corner read, legacy `mapal_gpu_d2h` | 174 |
| Second h2 corner read, legacy `mapal_gpu_d2h` | 189 |
| First free of any live device allocation | 192 |
| First raw c1 read correctly uses `mapal_gpu_d2h_on(i32 1,...)` | 300 |
| First raw c2 read correctly uses `mapal_gpu_d2h_on(i32 2,...)` | 331 |

There are **zero calls of any kind between lines157 and174**. Therefore no emitted alloc/free/sync/event or other runtime call orders that first member readback after its writer. The original omitted-raw fixture's two nonzero frees supplied an explicit runtime context barrier; this variant retains those buffers until their late reads and has no such barrier. The two member reads precede all frees and the explicit stream-specific raw readbacks.

[proof.json](proof.json) preserves exact calls, line numbers, device kernel observations and hashes. The existing lookup causing this distinction is `ResidentPlan::stream_of` (`crates/backends/llvm/src/func/resident.rs:293`): the raw target's IR producer is a chained site, while the grouped member's own Map producer is absent from `plan.chain`. All three consumers and existing true-writer facts are listed in [the parent investigation](../README.md).

The runtime creates nonblocking streams, and the legacy copy entry calls `cuMemcpyDtoH` without an explicit producer/context wait. CUDA specifies no implicit stream0 synchronization with nonblocking streams: [CUDA driver stream flags](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__STREAM.html). This supports an ordering concern in the emitted current program. It does **not** by itself establish that the tested GPU driver actually produces a wrong value on this fixture.

The interpreter exits0 and prints the original eight lines followed by `1014, 1008, 2760, 2618`; [oracle.stdout](oracle.stdout) preserves all12 lines. [before.json](before.json) and [after.json](after.json) agree on HEAD `b4152c5f4d9ad802d0388c1c64b628091d494e6a`, frozen source-manifest SHA, all328 source hashes and the emitter/interpreter/runtime-library hashes. No cargo, GPU execution or disassembly was run.

Next decisive step after authorization: preserve this exact emission, inspect/disassemble its linked runtime on an existing authorized GPU, then compare ordinary untraced execution against this oracle. Do not add clock reads or `MAPAL_GPU_TRACE=1`, which would synchronize and mask the path. Capture stdout/exit and, if necessary, a timeline establishing whether the member copy executes before the producer completes. Larger extents require new emission/oracle receipts. No source fix is selected; actual GPU wrong-output evidence remains pending.
