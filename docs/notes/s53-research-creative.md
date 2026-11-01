# S53 research — what a whole-graph compiler can do that a library cannot

Date: 2026-10-20 · Branch `s52-ptx-leg` @ fa77818 · Sources: S52 log, plan-s41 S52 revision, SASS review, ADR-0028/0032/0039, `algo.rs`, `reuse.rs`, `profile.rs`, `gpu.rs`, `mapal_gpu.c`, FRAMEWORK §5.

## Where the time actually is

Kernel-only, Mapal is at 0.92–1.06× the libraries on FP32. The *program* is far behind: on H100 n=4096 one site pays JIT 337–688 ms, alloc+h2d 5.5 ms, kernel 4.2 ms, d2h 3.2 ms (`mapal_gpu.c:mapal_gpu_tile` allocs, copies, syncs and frees per call). A library sees one GEMM; Mapal sees the graph. Ranked by (effect × certainty) ÷ semantic cost.

## Ranked proposals

### 1. Deduced retention — one rule, every cache a consequence

**The rule.** FRAMEWORK §5: a cache is *a stored copy of a deduced morphism*; store only when forward navigation is hot and the copy cannot drift. The IR already carries both halves of that sentence, so the compiler, not the program, decides. A value `V` produced at position `p` is **retained** — kept, in the layout and at the location its future readers want — iff:

- **R1, used again.** `LastUsePlan::death(V) > p` with a later reader (`algo.rs:564`); or `carried_by(V) = M` (a later iteration of loop `M` reads it, `:583`); or ≥ 2 readers (`EmissionClass::Named`, and `PathPlan` tasks sharing the dep).
- **R2, unchanged or appended-to.** No writer of `V` between `p` and the reader. For carried arrays the only writer is the back route; its `Update` index is an affine function of the merge counter, so `bounds_proof`'s interval lattice (`Rng`, `:410`) gives the per-iteration **delta** — the cells iteration `t` writes. Delta disjoint from the cells read by later iterations' sites = *append*; the prefix is invariant.
- **R3, readers' footprint is affine.** `TileRead` coefficients for tile sites, `ElemSrc` for maps. That footprint chooses the retained **layout** (the reader's pack `DataLoc`, ADR-0033 D3, at the reader's `Loc` — `Gmem` for a GPU site) and the **precision face**: a `to_bf16`/`to_tf32` on the read path is an `ElemSrc::Apply` law, applied once at retention, not per read.

A second layer follows with no extra fact: if a site's input cells are unchanged since the previous iteration, its output rows that depend only on those cells are bit-identical (same fold, same order, same inputs). Which output rows depend on which input cells is arithmetic over `TileRead` — the `backends/llvm/src/reuse.rs` precedent (`i_reuse`: arithmetic over recorded coefficients, no new graph analysis). So the site recomputes only the delta rows and keeps the rest: a result memo, from the same rule.

**What falls out.** Chain `c = a·b; d = c·e`: `c` never leaves the device. `x ← x·W` eight times: `W` crosses PCIe once, packed once. Two readers of one producer: one layout chosen for both. A decode loop — carried `X` with row `t` appended, `K = X·Wk`, `V = X·Wv`, `s = q_t·Kᵀ`, `o = s·V` — yields `K`/`V` device-resident in the readers' packed layout, rows `< t` retained, one projected row per step, one output row d2h: the known KV cache, named nowhere. The same rule produces every other cache the graph admits (reused weight panels, invariant operands, shared producers) and refuses the rest (a carried array fully rewritten each iteration has delta = everything).

**New facts (ADR-0033 D4(b) findings).** (i) `delta : (carried array, merge) → affine range in the counter`, from `Update` indices through `bounds_proof`. (ii) a `Footprint` query returning the affine read range per output cell for tile sites and maps alike (shared with #3). Everything else is recorded today.

**Oracle stays the judge.** Retention is value-invariant by construction — a retained cell is one whose producing morphism's inputs did not change, and the oracle is deterministic. The differential prints per-iteration outputs; both faces must match N/N with retention ON and OFF.

**Test family** (`gen_tiles.py`; no program names a cache): `chain_mm`, `loop_mm_w8` (invariant operand), `fanout_mm` (two readers), `decode_T` (static `T`, mask `k ≤ t`, the attention-decode shape), `loop_mm_rewrite` (negative: must retain nothing). Gate: OFF arm `retain`; `MAPAL_GPU_TRACE` reports retained bytes and skipped rows; `decode_T` vs PyTorch eager with a hand-written KV cache (the user wrote it, Mapal deduced it).

**Effect.** Decode O(d·T) per step instead of O(d·T²) when the program is written naively; every multi-site program and every loop drops its round trips. **Risk.** Host arena coloring and the device map must agree on aliasing; delta on an `Update` whose index is not affine in the counter must bail to "rewritten" (sound: the status quo).

### 2. Device residency at the last-use frontier — certain, largest e2e

Per-object `Loc` map `{Host, Gmem, Both}` in the host emitter (`func/core.rs` frame); a `Trm` is emitted only on a location change; `d2h` only for the elements a host morphism consumes (`c[0] -> println` moves 4 bytes, not 64 MB); device buffers freed at `death`. Runtime splits `mapal_gpu_tile` into `alloc/h2d/launch/d2h/free`. Differential unchanged; OFF arm `resident`; e2e from ~13 ms host to ~kernel time.

### 3. Host tax: cubin cache, pinned staging, clock fence — trivial, certain

`kernel_id` is content-addressed: cache `ptxas` output per `(id, arch)` beside `MAPAL_PTX` (337–688 ms → a file read). GPU-profile host arenas via `cuMemAllocHost` → pinned transfers (5.5 → ~2.5 ms at 4096) and the overlap #7 needs. Fix the known bug: `time` on a GPU profile fences before reading the clock. No new facts.

### 4. Fusion as a placement consequence: epilogues and prologues from `elem_plan`

Plan-s41 amendment 9's rule: a consumer stays in the producer's kernel iff its per-cell read footprint lies inside the block's tile. Three classes from recorded facts: **epilogue** — a `map` over the site's output with law `Apply{inner: Load{out}}`, trap-free by the `elem_law` gate (bias, activation, `to_bf16`, residual via `Pair`) folds into the store; **prologue** — an operand whose law is `Apply` (`to_tf32`, scaling, a `MoveSite` transpose) is computed inside `g2s`, the GPU reading of `core.rs:APPLY_INLINE`; **chain** — a consumer that is itself a tile site reads a whole panel → launch boundary, device-resident (#2). Needs the `Footprint` query (#1). Effect: GEMM+bias+ReLU in one launch; prologue removes a full pass for conversions. Test: per-cell exact ops keep the exact face N/N; programs `mm_bias_relu`, `mm_chain`, `mm_cast`; OFF arms `epilogue`, `prologue`; vs cuBLASLt epilogues and Inductor. Risk: register pressure under the 128-reg cap; ship prologue ON only for bodyless laws and conversions.

### 5. Canonical-tree float fold class (ADR-0028 deferred option) — biggest unlock; semantic, Sapir's call

Split-K/Stream-K for skinny shapes, the standalone `sum`, and FlashAttention's online softmax are *all* blocked by ADR-0028 D4. A fixed, machine-independent tree over `k` (chunk a function of `k` only; the interpreter computes the same tree) makes the reduction **placeable**: any chunk on any SM, bits identical. Proposal: `tree` as a fold class in the ADR-0032 lattice, declared on the fold; ADR-0039 D3's `block` is the same idea with a hardware k-step as the unit. Today `gpu_tile` halves the block when `rows/bm · c/bn < sms` (`profile.rs:981`) — the only legal move under `exact`; with `tree` it splits `k` across `sms / tiles` blocks. Test: differential *equality* (canonical, not a bound); OFF arm `splitk`. Risk: language change; default stays `exact`.

### 6. ADR-0039 storage types → wgmma + TMA on Hopper

H100 TF32 is 0.34–0.43× because cuBLAS uses `wgmma`, which needs K-major B or 16-bit operands. With `bf16`/`f16` storage and `to_bf16` visible, `Gpu::sm() >= 90` plus the operand format selects `wgmma.m64nNk16` with TMA; TMA descriptors are literally `TileRead` strides + a `(bm, bk)` box. Expected H100 ~85 → 180–220 TF/s (cuBLAS 234–255). Test: ADR-0039 D3 bound; OFF arms `wgmma`, `tma`. Risk: sm_90a-only; ADR not decided.

### 7. Concurrency of independent sites from `path_plan`

`PathPlan.tasks` (`deps`, `rank`) and `Checkpoint.wait` are the schedule the CPU's two-nets 1.63× came from. GPU reading: one stream per chain, an event per cross-chain edge, host syncs only at a `Checkpoint`; with pinned memory (#3) site B's `h2d` overlaps site A's kernel. `func/drive.rs`'s driver, forced off on GPU profiles (`lib.rs:331`), re-enabled with streams. OFF arm `streams`; `two_nets` vs PyTorch single-stream.

### 8. L2 rasterization from an `l2_bytes` device fact

Grid order is row-major. Group `g` row-panels so one wave's `g·bm` rows of A and `(sms·blocks_per_sm/g)·bn` columns of B fit L2, `g` from `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`. +3–8% at 8192. OFF arm `raster`.

### 9. Close the FP32 5%: register accounting and 32-bit indexing

SASS review: prefetch LDGs sunk late under the 128-reg cap, 52 B of spills. `min-blocks` and `prefetch` are deduced independently and fight. Rule: `regs_est` in `gpu_tile` counts prefetch registers and addresses when the rung is on — the `wide` whole-thread rule applied to the FMA tile. `i32` indexing when extents prove `rows·c·k·e < 2³¹`. OFF arms `regs`, `idx32`; 0.95× → ≥1.0× on Ada/Blackwell.

### 10. Warp specialization as `Trm` placement (speculative, Hopper)

`g2s` is a `Trm` placed today on the `Trn`'s threads. Placing it on its own warp (TMA producer; `mbarrier` is the "transmission complete" morphism) is a second `TrnLoc` for the transmission, derivable once #6 lands: one producer warp, consumers = `warps_x·warps_y`. The remaining gap on H100. Risk: sm_90a-only, `setmaxnreg`.

### 11. FlashAttention derived, not written (speculative; needs #4, #5, #6)

`attn_256.mapal` is two tile sites with a per-row softmax between. Under the footprint rule the row fits a block only while `n·4 ≤ smem share`; beyond that the boundary falls between QKᵀ and PV — where FlashAttention puts the online-softmax reassociation, which under `tree` is the merge of a placeable per-chunk max/sum. If it emerges from the three rules, Mapal has derived FlashAttention; if not, the missing fact is named (ADR-0033 D4). With #1, decode over a retained prefix is the same derivation one loop up. Test vs FA2 on H100, 16-bit.

## What to build first (S53)

2 → 3 → 9 need no decision and no new IR fact. 1 and 4 share the `Footprint` query; 1 adds `delta`. 5 and 6 are ADRs for Sapir; everything in (d) beyond `mma.sync`, and the softmax half of 11, waits on them.

| # | item | needs | risk | expected |
| --- | --- | --- | --- | --- |
| 1 | deduced retention | `delta`, `Footprint` | medium | O(T) decode; caches without names |
| 2 | device residency | last_use_plan (have) | low | e2e ÷3–10 |
| 3 | cubin/pinned/fence | nothing | none | −400 ms/process, h2d ÷2 |
| 4 | epilogue/prologue | `Footprint` | low | beat cuBLASLt fused shapes |
| 5 | tree fold class | ADR | semantic | split-K, sum, Flash |
| 6 | bf16 + wgmma/TMA | ADR-0039 | medium | H100 0.4× → ~0.85× |
| 7 | streams | nothing | low | 1.3–1.6× independent sites |
| 8 | L2 raster | l2 fact | low | +3–8% at 8192 |
| 9 | regs + idx32 | nothing | low | +5% FP32 |
| 10 | warp specialization | #6 | high | H100 last 15% |
| 11 | derived Flash | #4,#5,#6 | high | the thesis's proof |
