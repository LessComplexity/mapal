# Stored-member consumers: frozen emitted evidence and refutations

All five fixtures were emitted in Exact and Contract using the already built frozen emitter. All five interpreter oracles exit0. HEAD, source-manifest hash, all328 source hashes and the frozen emitter/interpreter/runtime-library hashes match in [before.json](before.json) and [after.json](after.json). [commands.json](commands.json) records exact argv/exits; [proof.json](proof.json) records fixture/emission hashes, all main calls, line numbers, stream IDs and oracle output. No cargo, GPU, disassembly, source/runtime/binary change or final bundle mutation occurred.

| Fixture, both faces | Actual emitted behavior | Conclusion for this dependency |
|---|---|---|
| [forked-tile](forked-tile/forked-tile.mapal) | Root grouped writer stream1; first tile reader stream1; second tile reader stream2; no explicit event | First child has FIFO ordering. Second child has a current emitted missing cross-chain wait; GPU consequence untested. |
| [member-cell](member-cell/member-cell.mapal) | Grouped writers streams1/2; reverse-cell consumer stream0; no explicit event; a new real output allocation intervenes | Veto misses this member consumer, but allocation may mediate/mask its execution. No runtime failure established. |
| [base-cell](base-cell/base-cell.mapal) | Changing the cell's operand to raw `c1` vetoes chains; all three launches stream0 | Existing base-output veto works; suspected cross-chain issue refuted for this control. |
| [member-rowfold](member-rowfold/member-rowfold.mapal) | Member row fold remains on host; grouped producer immediately copies the whole member out on stream1 | Suspected unordered GPU row-fold consumer refuted for this shape; its actual host fold has an explicit producer-stream wait. |
| [base-rowfold](base-rowfold/base-rowfold.mapal) | Raw-output row fold is a GPU kernel, and all launches use stream0 | Existing unordered row-fold veto works for a base output. |

These conclusions concern the named dependency, not an assertion that every other observation in each fixture is safe. For example, the member-rowfold fixture also keeps another group's corner reads, which are the separately recorded host-readback concern.

## Forked tile: one safe same-stream edge, one missing cross-chain wait

[forked-tile/exact.host.ll](forked-tile/exact.host.ll) and [contract.host.ll](forked-tile/contract.host.ll) have identical relevant line positions:

- `c1`'s grouped tile stores raw `c1` and member `h1` on stream1 at143. Member output slot `%d31` was allocated at134 and supplied as `%t32` at142.
- First child loads `h1` from `%d31` at156 and launches on stream1 at165. FIFO on that same stream orders it after the root; this child does not need a cross-stream event.
- Second child loads **the same** `%d31` at170, passes it as the first operand at174, and launches on stream2 at179. No `mapal_gpu_event` is emitted anywhere in main.
- The only calls after root143 and before second-child179 are output alloc calls158/172 and the first child launch165. Slots `%d26`/`%d28` were already allocated by the prologue at131/132 and never freed in between. `dev_alloc` returns an existing nonzero slot at `crates/mapal-rt/gpu/mapal_gpu.c:232–233`, so those two calls perform no new `cuMemAlloc` or context synchronization.
- All input generation occurs before the root. Runtime automatic legacy→chain events order that generation before a chain; they do not order root stream1→child stream2. Root and second child therefore have no emitted producer dependency through this path. Actual GPU wrong output remains untested.

This distinguishes a real fork from a simple linear chain, whose next tile would continue the producer's stream and refute the proposed cross-stream concern.

## Legacy cell: missed veto with an allocator caveat

In [member-cell/exact.host.ll](member-cell/exact.host.ll) and [contract.host.ll](member-cell/contract.host.ll), the grouped `h1` writer launches stream1 at149. The independent grouped tile launches stream2 at172. A separate reverse cell (`ta -> map { t -> h1[4095 - t] }`) loads `h1`'s `%d28` at182, then launches stream0 at187. The device unit contains exactly one `; cell` kernel. No explicit event, copy or context synchronization mediates its member input.

However, `%d54` starts at zero and its cell-output allocation at181 invokes a real `cuMemAlloc` before the consumer. Unlike the forked-tile cached allocations, this allocation may synchronize execution. This emitted case proves a member consumer escaped the base-only veto; it does **not** by itself prove an observable unordered execution or wrong result on a particular driver. Preserve that distinction before selecting any fix.

The matching base-cell control substitutes `c1[4095-t]`. It still emits one device cell, but all three launches use stream0 at164/191/208. Thus the existing `chains` unordered-consumer test actually vetoes the direct base dependency.

## Row fold: narrower concern refuted

`rowfold_sites` (`crates/backends/llvm/src/func/rowfold.rs:66–78`) requires the operand's own producer to be a GPU tile site. A grouped member's own producer is a Map, so a row sum over `h1` is not admitted as a GPU row fold in this implementation.

The emitted [member-rowfold/exact.host.ll](member-rowfold/exact.host.ll) / [contract.host.ll](member-rowfold/contract.host.ll) confirm that result: only two tile kernels exist, no row-fold launch occurs, and immediately after the `h1` producer launch165 the whole member is copied with `mapal_gpu_d2h_on(i32 1,...,i64 16384)` at166. That runtime entry queues the copy on stream1 and synchronizes stream1 before returning (`mapal_gpu.c:388–393`). The later row-fold host loop therefore reads a completed member through an explicit producer wait. No row-fold-specific correction is established by this case.

The base-rowfold control reads `c1` instead. It emits the GPU row-fold kernel via plain `mapal_gpu_launch` at205, but both grouped tile launches165/192 use stream0. The existing veto prevents a concurrent-chain producer for that recognized fold.

## Existing identity facts and next proof boundary

`epilogues::made` already records `member.out → base site` (`resident.rs:825`), and `arena::site_of` independently resolves the same writer to compute birth (`resident.rs:1469–1474`). `ResidentPlan::alias` maps carry views/exits to merge storage (`resident.rs:302`); these fixtures are straight-line and use no carry aliases. No new writer map, veto or event implementation was added.

The producer-stream rules are grounded in CUDA's [nonblocking stream contract](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__STREAM.html) and [stream synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html). Each open concern still requires the requested linked artifact/runtime proof before a fix is selected. Ordinary GPU execution and disassembly were not authorized or performed here; `MAPAL_GPU_TRACE=1` would synchronize launches and cannot establish ordinary asynchronous behavior.
