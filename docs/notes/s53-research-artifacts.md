# S53 research: IR artifacts vs the GPU leg

Branch `s52-ptx-leg` @ fa77818. What the IR proves, who consumes it, and what the GPU leg leaves unused. Survey only; nothing measured. Item and gap numbers are stable (plan-s53 cites them). This revision adds the deduced-caching frame and gap 8.

## Inventory

| Artifact (file:line) | What it proves | CPU consumer | gpu.rs today | GPU use it enables |
| --- | --- | --- | --- | --- |
| `tile_plan` → `TileSite{rows,c,k,a,b:TileRead,seed,elem}` (algo.rs:785, 335) | map∘fold site, reads affine `base+ci·i+clane·lane+ck·k`; trap-free; operation order | bulk.rs:95 tiled/packed/conv/SME rungs, frame.rs:200, lib.rs:340 prefetch | **yes**: `device_module` (gpu.rs:227), `launch` (gpu.rs:92), selection tile.rs:46 | already the kernel. Unused: **32-bit indexing** (max address < operand length, both static; kernel index math is i64, gpu.rs:318-337); ksplit refused (gpu.rs:67) |
| `TileKSplit` + `reuse::i_reuse` → `Invariant`/`Sliding{q}` (algo.rs:245, reuse.rs:53) | which reads adjacent output rows share | conv.rs:89/291/443 register blocking | `launch` keys smem staging on the same `Invariant` shape (`a.clane==0 && b.ci==0`, gpu.rs:99) | conv on device: stage the halo once per block |
| `move_plan` → `MoveSite{width,rows,cq,cr,elem,len}` (algo.rs:802, 309) | fold-less permutation, affine in `(t÷C, t%C)` | bulk.rs:46 move panel via `move_block` | no | transpose folded into the consumer's `TileRead`; never materialized, never copied |
| `elem_plan` → `ElemSrc{Index,Broadcast,Pair,Apply,Load}` (algo.rs:1429, 434) | closed-form `arr[i]` | bulk.rs:173/263 recompute; frame.rs:70 array elision; `Apply` declined (core.rs:263) | no | **operand generated on device** (no h2d); **epilogue** for `Apply{inner: Load{tile out}}` |
| `last_use_plan`: `position`, `death`, `escapes`, `carried_by`, `dead_after` (algo.rs:2680, 525) | where each object dies; reaches output?; loop-carried? | ops.rs:532 in-place `Update` | no | **d2h elision / partial d2h**; **free device buffers at death**; residency across sites |
| `bounds_proof` (algo.rs:2941) | `Index` statically in `[0,n)` | ops.rs:561 guard elision, mod.rs:50 FnAttrs | indirect (tile_site needs proven reads) | `c[0]` is a constant proven index: d2h offset known at compile time |
| `path_plan` → `Task{kind,deps,rank,trap_min,pinned}`, `Checkpoint` (algo.rs:1637, 83-139) | task DAG; the `TimeMs` fence as deps (algo.rs:2008-2036) and waits (2135-2160) | drive.rs:138 `emit_parallel` | **dropped** on GPU profiles (lib.rs:331) | **clock fence**; concurrent launch of independent `Split` tasks |
| `loop_structure`/`loop_plan` (algo.rs:2450/2486) | canonical loop, invariant `init`, SCC membership, decide/advance cones | lib.rs:288, loops.rs:25, drive.rs:225 | no | **keep loop-invariant operands on device** (∉ `scc_objects`); carried arrays stay on device |
| `guard_plan` → `GuardSite`/`GuardArm{own,can_trap,heavy}` (algo.rs:3339, 169) | arm-exclusive work | core.rs:23, drive.rs | indirect (host branches around the launch) | nothing new |
| `emission_plan` (algo.rs:617) | single-consumer pure values | none in llvm; cuda crate func.rs:163 | no | weak fusion precondition |
| `packing_site` + `FrameLayout.packed` (frame.rs:200, bulk.rs:105) | backend: a site worth a packed panel, packed ahead by a prefetch task | packed rung | no | none; smem staging is the device form |
| `conflict_free_pad` (profile.rs:266) | bank-conflict-free row pad | GPU only | async/mma; **sync FMA `pad_a: 0`** (profile.rs:998) | pad the 4-way-conflicted transposed-A STS |
| cuda `arena_plan` + last-use frees (cuda/src/arena.rs:99) | one device arena per fn, compile-time offsets | other backend | no | prior art for residency vs per-call `cuMemAlloc` (mapal_gpu.c:100-115) |

Runtime today (mapal_gpu.c): per call, 3 allocations, both whole operands h2d, launch, sync, whole output d2h, 3 frees. Nothing survives between calls except the module.

## Deduced caching (the general form)

A cache is a value kept, in the layout a later operation reads, because the program proves that operation reads it again and that it has not changed in between. The pattern comes from the graph, not from annotations or runtime profiling. On an LLM decode loop the rule produces the KV cache; nothing in it names attention.

| Proof | Artifact | State |
| --- | --- | --- |
| reuse: a later read exists | out-edges; `scc_objects` (read every iteration); `i_reuse` (adjacent rows); `TileRead` (k steps) | exists |
| validity: unchanged in between | objects are immutable, every write is a new object; `carried_by`, `dead_after` | whole objects yes; **part of an object no** (gap 8) |
| layout the reader wants | `TileRead` strides, `MoveSite`, `conflict_free_pad`, `packing_site` | exists |
| lifetime / exit | `death`, `escapes` | exists |
| budget | device memory bytes, L2 bytes | missing from `Gpu` (gap 6) |

Already fires: registers (`i_reuse` → conv register blocking), CPU packed panels (prefetch task), GPU shared memory (staged panels on the `Invariant` shape). The scopes it does not reach on GPU:

- **across sites** (device memory): items 2-4.
- **across iterations, whole object**: item 7.
- **across iterations, part of an object (incremental)**: if output row `r` of a site reads only input rows unchanged since the last iteration, it has the same bits (same fold, same operands, same order), so keep it and compute only the new rows. On decode: `K = X·Wk`, `Wk ∉ scc_objects`, `X` carried and written at row `t`. Row dependence is already in `TileRead` (row `i` reads `a` at `base+ci·i+ck·k`). The missing fact is which rows of `X` the iteration wrote. `examples/matmul4_loop.mapal` (`c[t] <- v` into carried `c`) has the write shape; no GPU program does (gen_tiles.py is straight-line).

## Facts no artifact provides yet (gaps)

1. **Consumer read footprint**: which elements of X the rest of the fn reads. `Index` and `TileRead` answer it locally; nothing combines them. Needed for partial d2h with several consumers and for amendment 9's fusion boundary.
2. **Per-cell consumer with captures** (`c[t] + bias[t%C]`). `MoveSite` allows one Index and no arithmetic; `tile_site` needs a fold; `elem_plan` covers `captures==0` only. Bias/activation epilogues need it.
3. **Data location of a value.** No record says an object is on the device. Per ADR-0032 this belongs to the backend; FnEmit has no device analogue of `ptr_resident`.
4. **Cross-fn composition.** Every plan is per `FuncId`, so residency stops at a `Call`.
5. **Canonical float-reduction tree** (ADR-0028 deferred). Blocks `sum`, split-K; semantics, not geometry.
6. **Device facts missing from `Gpu`** (profile.rs:125): device memory bytes (the cache budget), L2 bytes, DRAM and host-link bandwidth, copy-engine count, concurrent-kernel support.
7. The consumer-coverage gate (mapal-ir/tests/consumer_coverage.rs) is crate-level; `gpu.rs` not consuming `elem_plan` goes unflagged.
8. **Per-iteration write set.** `bounds_proof` ranges `Index` only and does not walk canonical-loop counters; `Update` stays trap-capable and unranged (algo.rs:3229-3237). Incremental retention needs "the `Update` in loop `m` writes exactly index `f(counter)`". The interval lattice (`Rng`) is the starting point.

## Deducible today, not yet used on GPU (ranked)

1. **Clock fence from `path_plan`** (correctness). GPU profiles ignore the plan (lib.rs:331), so `e2e_4096_c.host.ll` reads t1 before the GPU call. Emit the sequential host in an order that respects the task deps.
2. **d2h elision / partial d2h** from `escapes` + out-edges + `bounds_proof`. e2e reads `c` only through `c[0]`: 4 B instead of the whole output (measured 3.2 ms at n=4096 on H100).
3. **Operands generated on device via `elem_plan`.** e2e's `a`/`b` are `Apply{(t·7+13)%101−50 → widen, Index}`; compute them on device instead of copying (measured h2d 5.5 ms pageable). A GPU rung, separate from the CPU's `APPLY_INLINE=false`.
4. **Device residency across sites + free at death** from `tile_plan` × out-edges × `dead_after`. Needs a residency-aware runtime entry and a chain program; none exists.
5. **32-bit indexing** from extents + operand length.
6. **Sync-FMA A pad** via `conflict_free_pad`; targets the measured 4-way STS conflict.
7. **Loop-invariant operands kept on device** from `loop_plan` SCC membership; carried arrays by pointer swap.
8. **Concurrent launch** from `path_plan` Split tasks/deps/rank. CPU gain from the same plan: 1.63× on two nets; no GPU number.
9. **Epilogue fusion** for `Apply{inner: Load{tile out}}`, `captures==0`. Captured epilogues need gap 2.
10. **Transpose folded into `TileRead`** from `MoveSite` ∘ `TileRead`.
11. **Conv on device** from `TileKSplit` + `Sliding`; needs a ksplit device encoding, no new fact.
12. `emission_plan`: single-consumer signal only.

Not deducible today: **incremental retention** (the decode-loop KV cache). Needs gap 8 and a decode-shaped program in gen_tiles.py; items 4 and 7 come first, since it reuses their device-resident carried buffer.

Every item is a `GpuRungs` rung with an OFF arm and a box measurement before it ships ON (hard rule 1). Item 1 is a fix, not a rung.
