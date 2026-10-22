# S53 research: the CPU↔GPU link

Evidence: `benches/results/gpu/kernels/initial-comparisons/host-trace-v5-v6.txt`; host .ll from `emit e2e_1024.mapal --rewrite --target=rtx-4090`.

**One site, n=4096, contract face, v6 (ms, 3 processes):**

| part | load | alloc | h2d | kernel | d2h | free | `iter ms` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 4090 | 139–143 | 0.21–0.25 | 6.75–7.05 | 2.74 | 3.37–4.42 | 2.31–2.65 | 22.3–22.7 |
| H100 PCIe | 334–350 | 0.32–0.33 | 5.70–5.79 | 4.22–4.24 | 3.20–3.26 | 0.43–0.51 | 21.0 |
| RTX 5090 | 211–216 | 0.50–0.51 | 9.85–15.5 | 2.26–2.27 | 6.89–7.26 | 1.02–1.08 | 32.7–35.3 |

## 1. The missing clock fence

**Cause.** `topo_order` (`algo.rs:2328`) breaks ties on the source object's position. A `TimeMs` source is the previous effect's token (`mapal-lower/src/emit.rs:1739`), so a clock read inherits that position. `t0` sits at host .ll:40, ahead of the input-generation loops. `t1` sits at .ll:110, before `@mapal_gpu_tile` at .ll:115. `iter ms` therefore times input generation, while the site itself costs 155 / 361 / 237 ms (Σ phases). `path_plan` has the fence as task edges. The sequential walk (`drive.rs:776`) has none, and GPU profiles always take it (`lib.rs:331`). CPU single-path programs share the bug.

**Fix in `topo_order`**, which `last_use_plan`, `guard_plan`, `path_plan` and the interpreter all share (a walk-only reorder would desync `dead_after`). Post-pass on Kahn's output:
- `own(m) = 2·#{clocks before m}`; a clock gets `2k+1`.
- `seg(m) = max(own(m), seg(producer(source(m))))`; a loop takes its max.
- Stable-sort by `seg`.

The sort keeps the order topological. Clock-free functions are unchanged byte for byte. Device half: `call @mapal_gpu_sync()` before each `mapal_time_ms` in GPU modules (needed once §2 makes transfers lazy). This is not a rung. Also `box_profile.sh:45` uses `bc`, absent on the box, which empties `wall_s`.

**Test:** an order test on the e2e shape, an llvm golden, and on the box fenced `iter ms` = Σ phases. **S** (~40 lines).

## 2. Device-residency model (FRAMEWORK §4)

`Loc` = {Host, Gmem}. `Trm` = `h2d(x[r])`, `d2h(x[r])` over element range `r`, plus `launch`, alloc and free. Placement is a backend decision (ADR-0032).
- **R1:** a device site reads x ⇒ Gmem ∈ DataLoc(x) at launch.
- **R2 h2d:** only under R1, with a host producer and no R5.
- **R3 d2h:** device producer, and x has a host consumer or escapes. When every host consumer is `Index` at a `Constant`, copy only those elements.
- **R4 free:** at `death(x)`. Escaping and carried objects stay to exit.
- **R5 produce on device:** `elem_plan` law `Index`/`Broadcast`/`Apply` ⇒ an elementwise kernel writes x, with no host array.

Objects are fresh, and in-place `Update` needs `dead_after`, so validity is static: no runtime coherence. ABI: split `mapal_gpu_tile` into `_alloc/_h2d/_launch/_d2h/_free/_sync` on one stream.

**e2e:** `a` and `b` are `Apply{Index}` (R5), so the 5.7–15.5 ms h2d goes. Only `c[0]` is read (R3), so d2h moves 4 B, not 64 MB (3.2–7.3 ms). Rungs: `resident` + `partial` (M), `devgen` (M–L).

## 3. Host runtime

- **`load_ms` is context creation, not JIT.** nsys on the same boxes: `cuModuleLoadData` takes 0.45 / 0.82 / 0.41 ms, and the first `cudaFree` takes 109 / 147 / 152 ms. The first of 3 processes pays nothing extra. Fixes: split the trace field. Add an `eager-init` rung: `main` starts init on a pthread, overlapping 135–350 ms with pre-site host work (S). An embedded cubin is worth at most 0.4–0.8 ms warm, so measure cold JIT first.
- **`pool`:** `cuMemAllocAsync`/`FreeAsync` on the default mempool, release threshold `UINT64_MAX`. Removes 2.5 / 0.8 / 1.6 ms per site (S).
- **`pinned`:** pageable h2d runs at 19.7 / 23.2 / 13.1 GB/s; the pinned rate is unmeasured, so add the arm first, then pin only the R2/R3 arrays (S–M).
- **`streams`:** sites with no `path_plan` path between them, when `Σ blocks ≤ sms × blocks/SM`. At n≥2048 one grid already fills each part (M).
- **`overlap`:** panel d2h under the next panel's kernel, at most 2.7 / 4.2 / 2.3 ms. Nothing is left to overlap in e2e after R3/R5. Last (M–L).

## 4. Deduced caching

**Rule.** A value is retained when the graph proves a later operation reads it again. What is kept, where, in what layout, and how long all follow from who reads it and what changes between reads. The rule names no workload. It reads five facts:

| fact | proof | status |
| --- | --- | --- |
| F1 read again | ≥2 consumers through dissolved products | `emission_plan` |
| F2 read every iteration | not merge-reachable, read in the body; the graph already computes it once before `LoopEnter` | `loop_plan` |
| F3 lives across iterations | `carried_by(x)` | `last_use_plan` |
| F4 delta | the back route's x differs from the merge's x only at Δ(x): `Update` chain → {p}; select-map `row(i)==p ? new : old` → row p; else ⊤ | **new**, ~60 lines |
| F5 row-local | output row r reads x only at row r, other operand row-independent: elementwise `elem_plan` law, or tile site with `a.clane==0`, `b.ci==0` (the staged-kernel gate) | exists |

What the backend does with each:
- **F1:** one device copy, freed at `death`.
- **F2:** one h2d before `LoopEnter`, freed after `LoopExit`. Today `mapal_gpu.c` re-uploads per call.
- **F3:** stays in Gmem across `LoopBack`. With a host producer, F3+F4 moves only Δ.
- **Derived cache (F2+F3+F4+F5).** x is carried with Δ, and y = F(x, w) is row-local with w invariant. Then `y' = Update(y, Δ, F_row(x'[Δ], w))`. y becomes a new carried object, allocated once at its static size and filled once from init x. Each iteration then runs the same kernel over Δ's rows only. y inherits Δ, so the rule propagates through row-local consumers. It stops at the first row-mixing consumer, which reads the retained y.
- **Append-only:** Δ = the loop counter (monotone by `bounds_proof`) sweeps a growing prefix. y is never re-uploaded, and the prefix bounds any R3 d2h.
- **Layout:** y is written one row per iteration and read every iteration, so it is stored in its reader's order (`TileRead`/`MoveSite` coefficients). Any transpose moves into the one-row write.

**Exactness:** exact-face kernels equal the oracle per cell, so a retained row equals its recompute. It already ran without trapping, so no trap is removed. On ⊤ nothing changes.

**On an LLM decode loop:** token rows are appended to carried x, the projections `x·w_k` and `x·w_v` are row-local with invariant weights, and attention folds across rows. The rule yields carried, device-resident, append-only k and v laid out for the scores read, with resident weights. That is the known KV cache, deduced; nothing names it.

**Rung** `derive-cache` (OFF = full site each iteration). **Test:** a decode-shaped program in `gen_tiles.py`. The differential must be N/N. The trace should show `bytes_in` = one row and a one-row launch ON vs the full grid OFF. **Saving:** no loop program has been measured yet; its OFF arm is the first number. **M** (~210 lines).

## 5. Order

| # | item | measured phase removed (4090 / H100 / 5090, ms) | size |
| --- | --- | --- | --- |
| 1 | clock fence + sync | bracket becomes 155 / 361 / 237 | S |
| 2 | trace split, `wall_s` | instrument | S |
| 3 | eager-init | 139–143 / 334–350 / 211–216 | S |
| 4 | resident + partial | 3.4 / 3.2 / 7.3 d2h | M |
| 5 | devgen | 6.8 / 5.8 / 10.2 h2d + generation | M–L |
| 6 | pool | 2.5 / 0.8 / 1.6 | S |
| 7 | derive-cache | none yet: needs a loop program | M |
| 8 | pinned | rate arm first | S–M |
| 9 | streams | sub-wave sites only | M |
| 10 | overlap | ≤ 2.7 / 4.2 / 2.3 | M–L |

Every row from 3 on is a `GpuRungs` field with `--gpu-off`, measured with the fenced e2e on vast, with the differential at N/N. After rows 1–6, the n=4096 site is kernel + launch: 2.7 / 4.2 / 2.3 ms against 15.5 / 14.1 / 21.4 ms today.
