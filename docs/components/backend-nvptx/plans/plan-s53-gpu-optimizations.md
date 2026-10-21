# plan-s53: GPU optimizations — the build list

Written: 2026-10-20 · S53 · judge over the four research reports (`docs/notes/s53-research-{creative,artifacts,link,kernel}.md`)
Branch: `s52-ptx-leg` @ fa77818 · Measured state: `benches/results-s52-gpu/v6-*.txt`, `host-trace-v5-v6.txt`
Rules: plan-s41 S52 revision (10 amendments) · ADR-0032 (contracts vs backend config) · ADR-0028 D4 (float folds pinned) · ADR-0039 (candidate, not decided)

## 0. Scoring

Each idea scored on three axes, the product is the rank:

- **gain** — expected effect on a *whole program* against the best library, taken from a measured S52 phase
  (kernel ratios in v6, host phases in the trace file). A ceiling from a measured number, never a guess.
- **feasible now** — buildable on `s52-ptx-leg` by one engineer-agent in one worktree without an ADR and
  without an item that is not yet built.
- **fit** — deduced from device facts / ISA rules / program geometry; exact face bit-identical; a `GpuRungs`
  OFF arm; ADR-0032 (precision in the program, placement in the backend).

Three facts the reports established, which the ranking depends on:

1. **The 0.95× headline is the contract face.** The exact face (`mul.rn`+`add.rn`, two instructions per MAC)
   is 0.42–0.64× at ≥2048 on every part. Kernel items are judged on the contract face; the exact face is the
   differential gate and gets its own arms.
2. **`load_ms` is context creation, not JIT.** nsys: `cuModuleLoadData` 0.45/0.82/0.41 ms, first `cudaFree`
   (context) 109/147/152 ms. Nothing to cache; the fix is to start the context before the program needs it.
3. **Retention is one rule at two scopes.** "Keep a value, in the layout its later readers want, when the
   graph proves it is read again and unchanged in between." At whole-object scope the facts exist today
   (out-edges, `loop_plan` SCC membership, `carried_by`, `death`, `escapes`): that is device residency.
   At part-of-object scope one fact is missing (which cells of a carried array an iteration writes): that is
   incremental recompute. Neither scope names a workload; on a decode loop the two together are the KV cache.

## 1. Every idea, ranked

Gain/feasible/fit each 1–5. Verdict: **BUILD** (in §2), **AFTER** (needs a build-list item or a fact first),
**ADR** (Sapir's call), **REJECT** (reason, one line).

| # | idea | source | gain | feas | fit | verdict |
|---|---|---|---|---|---|---|
| 1 | GPU host clock fence (topo_order segments) + device sync before `time` | link §1, artifacts 1, creative 3 | 5 (gates every e2e number) | 5 | 5 | **BUILD** `clock-fence` |
| 2 | Per-arch rung selection by measured effect, `--gpu-on`, `_exact` arms, 5090 `max_blocks_per_sm` 32→24 | kernel R1, kernel §1 | 4 (4090 contract 0.87→0.95 at 4096 is a measured off-arm row) | 5 | 5 | **BUILD** `arch-rungs` |
| 3 | Retention, whole-object scope: device location map, h2d at the producer (one copy per reader set, per loop), carried arrays stay on the device, partial d2h, free at death, device-generated closed-form operands | creative 1–2, link §2/§4 F1–F3, artifacts 2–4,7 | 5 (h2d 5.7–15.5 + d2h 3.2–7.3 of a 14–21 ms site) | 3 | 5 | **BUILD** `resident` |
| 4 | Retention, part-of-object scope: per-iteration write delta of a carried array + row-local sites → recompute only the written rows into a retained carried result | creative 1 (second layer), link §4 F4–F5, artifacts gap 8 | 4 (O(T·d) → O(d) per step on a decode loop; no number yet) | 3 | 5 | **BUILD** `retain-delta` |
| 5 | Eager context init on a thread + stream-ordered mempool + trace split | link §3(a),(b) | 4 (135–350 ms per process; alloc+free 0.8–2.5 ms per site) | 5 | 4 | **BUILD** `host-runtime` |
| 6 | 32-bit indexing + affine copy addressing + pad A on the sync path, then the whole-file FMA register tile (8×16, 128 threads, 128×128) | kernel R2, R5, R6; creative 9; artifacts 5–6; SASS review | 4 (48–68 B spills on every part; cuBLAS/CUTLASS-128 1.11–1.30× H100, 1.00–1.18× 5090) | 4 | 5 | **BUILD** `fma-regs` (R6 needs R2's registers: merged) |
| 7 | Warp-fill rule (grid warps ≥ 2·schedulers·SMs; shrink the register tile first) | kernel R3 | 3 (1024: FMA 0.48–0.78×, TF32 0.42–0.51× on all parts) | 5 | 5 | **BUILD** `warp-fill` |
| 8 | Hopper TF32 `wgmma` with swapped operands (Cᵀ = Bᵀ·Aᵀ), pre-rounded A in smem, cp.async staging, no TMA | kernel R7+R8 | 5 (H100 TF32 0.34–0.43×; cuBLAS wgmma = 1.56–2.13× CUTLASS mma.sync) | 2 | 4 | **BUILD** `wgmma-hopper` |
| 9 | Epilogue / prologue fusion from `elem_plan` | creative 4, artifacts 9 | 3 | 2 | 4 | **AFTER** `resident`: needs the footprint query (artifacts gap 1) and the captures record (gap 2); no bias/activation GPU program exists |
| 10 | Concurrent independent sites on streams from `path_plan` | creative 7, link 3(d), artifacts 8 | 2 | 3 | 4 | **REJECT for S53**: ceiling only for sub-wave sites (≤1024 on the 4090), which `warp-fill` fills instead; needs the parallel driver re-enabled on GPU profiles (`lib.rs:331`) |
| 11 | Pinned host memory | creative 3, link 3(c) | 2 | 3 | 4 | **REJECT for S53**: rate unmeasured; after `resident`+devgen the e2e programs have no transfer left. A pageable-vs-pinned probe rides in `resident`'s `tile_bench` |
| 12 | Copy/compute overlap by row panels | link 3(e) | 1 | 2 | 4 | **REJECT**: needs pinned; ceiling ≤ min(transfer, kernel), zero after devgen |
| 13 | cubin / PTX cache, embedded PTX | creative 3, link 3(a) | 1 | 4 | 4 | **REJECT**: mislabelled — module load is 0.4–0.8 ms; `host-runtime` is the fix |
| 14 | Zero-pad swizzle, stages before `bk`, cp.async only with ≥3 stages | kernel R4 | 1 | 4 | 4 | **REJECT**: H100 already runs 3 stages and async still costs 6–8% at ≥2048 (kernel §1); the measured cost is spills and 4-byte copies (`fma-regs`), not depth |
| 15 | K-major A staging with 16-byte cp.async | kernel §3 "not proposed", SASS review | 2 | 3 | 5 | **AFTER** `fma-regs`: a TMA prerequisite, no measured gap of its own (cuBLAS reaches its rate with 4-byte copies) |
| 16 | TF32 fragment double-buffer, XOR swizzle, 4–5 stages on Ada/Blackwell | kernel §2.6 | 2 | 3 | 3 | **AFTER** ADR-0039: the TF32 face is interim (ADR-0032 D5 deletes `--precision=tf32`); 0.86–0.93× where cuBLAS = CUTLASS is the consumer gap |
| 17 | L2 rasterization (`l2_bytes` device fact) | creative 8, kernel R9 | 1 | 4 | 4 | **AFTER** nsys at 8192 shows the same cuBLAS kernel at both sizes (kernel R9's own gate); evidence is one row |
| 18 | Canonical-tree float fold class (split-K, `sum`, Flash) | creative 5, plan-s41 am. 10 | 5 | 1 | — | **ADR** (ADR-0028 D4 re-pin): a language change, not a rung |
| 19 | bf16/f16 storage + wgmma/TMA via ADR-0039 | creative 6 | 4 | 1 | — | **ADR**; `wgmma-hopper` takes the TF32 route that needs no new types, and the same kernel serves bf16 when D1 lands |
| 20 | Warp specialization as `Trm` placement (producer warp + mbarrier) | creative 10 | 3 | 1 | 4 | **AFTER** `wgmma-hopper` + TMA |
| 21 | FlashAttention derived | creative 11 | 5 | 1 | 5 | **AFTER** 9, 18, 19 |
| 22 | Transpose folded into `TileRead` (`MoveSite` ∘ `TileRead`) | artifacts 10 | 1 | 3 | 4 | **REJECT for S53**: no GPU program has a transposed producer; nothing to measure |
| 23 | Conv on the device (`TileKSplit` + `Sliding`) | artifacts 11 | 2 | 2 | 5 | **REJECT for S53**: no conv GPU program or cuDNN baseline in the harness |
| 24 | `emission_plan` single-consumer signal | artifacts 12 | 0 | — | — | **REJECT**: a precondition for 9, not a rung |
| 25 | Consumer-coverage gate per realization | artifacts gap 7 | 0 | 5 | 3 | **REJECT**: test infrastructure, no measurement; do it when 9 makes `gpu.rs` an `elem_plan` consumer |
| 26 | Retained result memo across *sites* (skip rows of a non-loop site whose inputs are unchanged) | creative 1 | 1 | 2 | 4 | **REJECT for S53**: outside a loop every object is produced once, so there is no second computation to skip; the loop form is `retain-delta` |
| 27 | Trace split, `wall_s` bc fix, `mm_3072`/`mm_1536`, nsys at 1024/8192 | link §3(a), kernel §4 | — | 5 | — | folded into `host-runtime`, `clock-fence`, `warp-fill`, `arch-rungs` |

**Why these eight.** Each needs no decision from Sapir and no item that is not yet built; each has an OFF arm
and a box measurement. The three largest numbers in a whole program (context 135–350 ms, h2d+d2h 9–23 ms,
kernel 2.7–4.2 ms of a 14–21 ms site at n=4096) each have one item; the one rule the project is for (fact 3)
has one item per scope, and the decode-shaped test where the KV cache has to emerge lives in `retain-delta`.

## 2. The build list

Common to every item (hard rules 1–7):

- Branch `s53/<key>` off `s52-ptx-leg` in its own worktree. A `GpuRungs` field per rung, `emit --gpu-off=<name>`
  (and `--gpu-on=<name>` once `arch-rungs` lands; until then the OFF arm is `--gpu-off`), an entry in
  `bundle.sh`'s arm list.
- Tests, in this order: `cargo test --workspace --release` green (unit tests named per item); box differential
  **N/N PASS on every listed part** (N = 12 + the item's programs), both faces; the on/off A/B interleaved
  with cuBLAS and CUTLASS in one vast session per listed part (`PROFILE=1`); host-link and graph items also
  measure their e2e programs. Every number appended to `benches/results-s53-gpu/<key>-<part>.txt` the moment
  it lands; `vastai show instances` and `vastai destroy instance <id> -y` close every session.
- Ship a rung ON only on the parts where its arm measured positive; record the decision in the profile's
  rung table (`arch-rungs` §2.2) with the results row cited.
- e2e numbers that do not need the fence: `MAPAL_GPU_TRACE=1` phase sums and process wall time
  (`date +%s%N`). The program's own `iter ms` is valid once `clock-fence` has landed — land it first; the other
  items do not wait for it.

### 2.1 `clock-fence` — the GPU host clock fence (host-link, S)

**Deducible from.** `path_plan` already fences `TimeMs` as task deps (`algo.rs:2008`) and checkpoint waits
(`algo.rs:2139–2160`). The sequential walk (`drive.rs:776`, the only walk on GPU profiles, `lib.rs:331`) uses
`topo_order` (`algo.rs:2328`), which orders a morphism by its source object's position; a clock's source is the
previous effect's token, so `t1` is emitted before the site (`e2e_4096_c.host.ll:110` vs `:115`).

**Design.** A segment pass at the end of `CategoryIr::topo_order`, not in the walk (`last_use_plan`,
`guard_plan`, `path_plan` and the interpreter take positions from `topo_order`; reordering only the walk
would put the in-place `Update` check out of step with emission):

1. `own(m)` = `2 × (number of TimeMs morphisms whose loc.start < m.loc.start)` for a non-clock, `2k + 1` for the
   k-th clock.
2. `seg(m) = max(own(m), seg(producer of m.source))`, walked in the existing order; every morphism of a
   loop SCC (`loop_structure`) takes the SCC's maximum.
3. Stable sort the existing order by `seg`. Segments never decrease along a dependency, so the order stays a
   valid topological order; a function with no `TimeMs` has one segment and keeps its order byte for byte.
4. `debug_assert!`: no clock is sorted past its own segment.

Device half: `FnEmit::emit_time_ms` (`ops.rs:502`) emits `call void @mapal_gpu_sync()` before
`@mapal_time_ms()` when `self.profile.gpu().is_some()` and the module has a GPU site (the condition under which
`lib.rs:422` appends `GPU_DECLS`); `mapal_gpu_sync` = `cuCtxSynchronize` in `mapal_gpu.c`. Trivially correct
today (the tile call synchronizes) and required once launches become asynchronous (`resident`).

Harness: `box_profile.sh:45` computes `wall_s` with `bc`, absent on the box — use `$(( ))` on `date +%s%N`.

**Tests.** `crates/mapal-ir/tests/algos.rs`: an e2e-shaped program (iota → map → time → tile site → time →
print) asserts every morphism of the site lies between the two `TimeMs` in `topo_order`; a clock-free program's
order is unchanged (every existing snapshot in `mapal-ir`, `mapal-lower` and `backends/llvm` stays byte-identical
— no `insta` update accepted). `golden_ll.rs`: the GPU-profile e2e golden shows `mapal_time_ms` →
`mapal_gpu_tile` → `mapal_gpu_sync` → `mapal_time_ms`. Box: differential 12/12 on all three parts (clock-free
programs, order unchanged); fenced `iter ms` of `e2e_1024`/`e2e_4096`, both faces, equals the trace's
alloc+h2d+kernel+d2h+free sum within 1 ms (3 processes each). vs cuBLAS/CUTLASS: unchanged kernels — the
`mm_*` rows are the control that nothing else moved.

**Expected.** `iter ms` today 22.3 / 21.0 / 32.7 ms (4090 / H100 / 5090, n=4096) measures the host building the
inputs; after, it equals the site's phases (15.5 / 14.1 / 21.4 ms, load excluded). A correctness fix, not a
speedup. CPU timed programs get the fenced order too and need re-measuring in a CPU session (rule 5: not here).

### 2.2 `arch-rungs` — per-architecture rung decisions by measured effect (kernel, S)

**Deducible from.** `GpuRungs::default()` (`lib.rs:92`) ships every rung ON on every part; v6 shows
`cp_async` costs the 4090 8% at 4096 (0.87× vs 0.95× off-async), the H100 7%, the 5090 2%, and gains 47% / 27%
at 1024 on 4090 / 5090 (−5% on H100). `DispatchCost` (`profile.rs:393`) is the precedent: a measured number
lives only on the profile it was measured on. The box reports `max_blocks_per_sm = 24` on the 5090; the
profile says 32 (kernel §1).

**Design.**

1. `TargetProfile::gpu_rungs(&self, warps_per_scheduler: u64) -> GpuRungs` in `profile.rs`: a `match self.name`
   for `rtx-4090`, `h100-pcie`, `rtx-5090`, each `false` with a one-line citation of the results row; the
   regime key is `warps_per_scheduler < 2` (latency-bound; the `GPU_RESIDENT_BLOCKS` reason applied per
   scheduler, `schedulers = 4` an ISA fact for sm ≥ 70). From v6: `cp_async` ON only when under-filled on
   sm_89 / sm_120, OFF everywhere on sm_90; `warp` OFF on sm_89 (off-warp 0.91 vs 0.87 at 4096/8192) and within
   noise elsewhere; `ldmatrix` OFF (≈0 on every part); `wide` ON on sm_90 (+16–19%), engineer's call from the
   re-run elsewhere (+1–2% 4090, 0 5090). Unmeasured GPU profiles (`cuda-ada`, `h100`) get the intersection of
   positives: interleave, unroll, min-blocks, prefetch, mma, wide. `RTX_5090.max_blocks_per_sm = 24`.
2. `EmitOpts.gpu` becomes two masks, `gpu_off` and `gpu_on` (both default empty); effective rungs at a site =
   `(profile.gpu_rungs(wps) | on) & !off`, computed once in `gpu::launch` and `kernel_text` and passed down
   (`Kernel.rungs`, `FnEmit.gpu_rungs`). `emit --gpu-on=<rung>` is the ON arm for a rung the profile ships OFF.
3. `bundle.sh`: for every rung, the arm opposite to the profile's decision (`_c.off-<r>` or `_c.on-<r>`), and
   `_exact.off-async` / `_exact.on-async` / `_exact.off-prefetch` (async's cost differs by face: H100 exact
   0.57 → 0.46 between v4 and v6).

**Tests.** Unit: `RTX_4090.gpu_rungs(1).cp_async == true`, `.gpu_rungs(4).cp_async == false`,
`H100_PCIE.gpu_rungs(1).cp_async == false`; every `false` in a measured profile carries a citation (a static
table checked by a test); `CUDA_ADA.gpu_rungs(_)` equals the intersection; the 5090 geometry test re-pinned
for 24 blocks. Golden: `mm_4096_c` on `rtx-4090` has no `cp.async` in the device unit. Box, all three parts:
differential 12/12; the new defaults plus every opposite arm vs cuBLAS/CUTLASS at 1024–8192; then write the
table from those rows (one more session if a flip changes a decision).

**Expected.** Contract face at 4096 from the measured off arms: 4090 0.87 → 0.95×, H100 0.88 → 0.95×,
5090 0.83 → 0.85×; 1024 keeps async where it won. Every later item's arms are measured against these
defaults.

### 2.3 `resident` — retention at whole-object scope: device residency, partial d2h, free at death, device-generated operands (host-link, L)

**The rule, at this scope.** An object produced for a device site is kept on the device, in the kernel's
layout, as long as some later reader is a device site and nothing rewrites it. Objects are immutable and every
morphism writes a fresh one; in-place `Update` is legal only on a dead source (`last_use_plan`), so "unchanged
in between" holds for every whole object by construction. The three reuse facts are already recorded:
**F1** a second reader (out-edges reach two sites), **F2** a loop-invariant operand (producer outside the
loop's SCC, `loop_plan`), **F3** a carried array (`carried_by`). The location map below reads them; no
workload is named.

**Deducible from.** `tile_plan` names the device sites; `out_edges` say which objects they read and which host
morphisms read a device output; `last_use_plan.death` (`algo.rs:525`) gives the free point and `escapes` the
objects that must outlive the fn; `bounds_proof` proves `c[0]` a constant index; `elem_plan` (`algo.rs:1429`)
gives the bench's `a`, `b` a closed form (`Apply{(t·7+13)%101−50 → widen, Index}`); `FnAttrs::clean` says the
body has no trap and no call.

**Design.**

1. **Runtime** (`mapal_gpu.c`): `mapal_gpu_alloc(bytes) -> ptr`, `mapal_gpu_h2d(d, h, bytes)`,
   `mapal_gpu_launch(id, da, db, dout, gx, gy, block, smem)` (asynchronous on one stream),
   `mapal_gpu_d2h(h, d, off, bytes)` (synchronizes), `mapal_gpu_free(d)`, `mapal_gpu_sync()`. `mapal_gpu_tile`
   stays, composed of them: the OFF arm.
2. **Location map** in `FnEmit`: `dev: SecondaryMap<ObjectId, String>` (object → device pointer SSA), the GPU
   reading of `ptr_resident` (`core.rs:158`). Rules, applied in the sequential walk:
   - **h2d at production** (F1, F2): after a host morphism produces an array that some device site reads
     (`out_edges` reach a GPU tile site's capture product), alloc + h2d once and record it. The copy sits at
     the producer, so a loop-invariant operand crosses once for the whole loop and an array read by two
     sites crosses once.
   - **launch** (`emit_tiled_map_gpu`): operands from `dev`; output alloc'd on the device and recorded; no d2h.
   - **carry** (F3): at `LoopBack`, the merge object's `dev` entry becomes the back-route object's pointer
     (the merge's slot is a device pointer, like the host slot today); the previous pointer is freed at its
     death. A carried array that a device site reads and a device site produces never visits the host. A
     carried array the host writes (`Update` chain) stays host-side and is re-uploaded — `retain-delta` is
     what shrinks that upload to the written rows.
   - **d2h lazily at the first host read** (`array_operand_ptr` / `load_component` on an object in `dev`): if
     every host consumer of the object is an `Index` with a `bounds_proof`-constant index, copy those elements
     at each read (4 B for `c[0]`); otherwise copy the whole array once and mark it host-valid.
   - **free at `death`**: after the morphism at `lup.death(o)`; escaping objects at fn exit.
3. **Rung `devgen`** (second rung of the item, built after the map): a producer (`Iota`, `Fill`, `Map`) whose
   `elem_plan` law is `Index`, `Broadcast`, or `Apply{captures: 0}` with a clean body, and whose consumers are
   all device sites, is produced on the device: the body's `define internal` text (`drive.rs:133`) is appended
   to the device unit (a clean fn is scalar LLVM IR with no host symbol — `device_module` asserts it) and a
   kernel `mapal_gen_<id>(ptr out, i64 n)` calls it per element; host emits alloc + launch, no host array. Any
   host consumer → materialize on the host as today (the lazy d2h covers a mixed case). This is the GPU
   answer to `APPLY_INLINE = false` (`core.rs:263`): a separate decision, a separate rung.
4. Programs in `gen_tiles.py`: `mm_chain_2048` (`c = a·b; d = c·e; d[0]`, F1 through a chain), `mm_share_1024`
   (`d = a·b; e = a·c`, F1 fan-out), `loop_mm_1024_8` (`x ← x·W`, 8 iterations: F2 for `W`, F3 for `x`),
   `e2e_chain_4096` (timed), and differential versions at 256 (both faces). `tile_bench --program`: a cuBLAS
   whole-program floor (context, alloc, device-side generate, `sgemm`, one 4-byte d2h) — the library
   comparison for e2e. A pageable-vs-pinned h2d rate probe in `tile_bench` (measurement only, row 11).

**Tests.** Unit: the location map on an e2e-shaped IR (h2d after the producers, no d2h for `c` except a 4-byte
one at `c[0]`, free at death positions); `mm_chain` emits one h2d per input and none for `c`; `loop_mm` emits
`W`'s h2d outside the loop and no transfer of `x` inside it; the `devgen` gate refuses an unclean body. Golden
`.ll` for each. Box, all three parts: differential 12 + 4 = 16/16, both faces, with and without
`--gpu-off=resident`, `devgen`; `MAPAL_GPU_TRACE` shows `bytes_out=4` on e2e, zero transfer for `c` in the
chain, one h2d of `W` and none of `x` in the loop, `bytes_in=0` under devgen; e2e wall and phase sums ON vs
each OFF arm, vs `tile_bench --program`; `mm_*` kernel rows vs cuBLAS/CUTLASS as the control.

**Expected.** From the measured phases at n=4096 (4090 / H100 / 5090): d2h 3.4 / 3.2 / 7.3 ms → ~0; h2d 6.8 /
5.7 / 10–15 ms → 0 under devgen. The site goes from 15.5 / 14.1 / 21.4 ms to kernel + launch + alloc
(2.7 / 4.2 / 2.3 ms + alloc, which `host-runtime`'s pool removes). The chain and loop programs save one round
trip per intermediate and per iteration on top; those are the first loop numbers on a GPU profile.

### 2.4 `retain-delta` — retention at part-of-object scope: recompute only the rows an iteration changed (graph, M)

**The rule, at this scope.** Inside a loop, a site `y = F(x, w)` is recomputed every iteration although only
part of `x` changed. If (i) `x` is carried and the iteration writes a known set of its cells Δ(t), (ii) the
other operand `w` is loop-invariant, and (iii) output row `r` of the site reads `x` only at row `r`, then
every output row outside Δ(t) is bit-identical to last iteration's (same inputs, same fold, same order). So
`y` is kept as a carried array of its own and only Δ(t)'s rows are recomputed. `y` inherits Δ, so the rule
propagates through row-local consumers and stops at the first row-mixing one, which reads the retained `y`
whole. Nothing in it refers to attention; on a decode loop (token rows appended to `x`, projections `x·Wk`,
`x·Wv` row-local, scores row-mixing) the retained `K`, `V` are the KV cache.

**Deducible from.** (i) is the one missing fact (artifacts gap 8): `bounds_proof`'s interval lattice (`Rng`,
`algo.rs:2941`) ranges `Index` only; `Update` is untouched (`algo.rs:3229`). The `Update` chain in a loop body
already has the needed shape in `examples/matmul4_loop.mapal` (`c[t] <- v`, index affine in the counter). (ii)
is `loop_plan` SCC membership. (iii) is the staged-kernel gate `a.clane == 0 && b.ci == 0` (`gpu.rs:99`), or
an elementwise `elem_plan` law. Independent of `resident`: the rewrite is at emission level and runs on
today's `mapal_gpu_tile` ABI (and on the CPU rungs when a Δ-row site is below the GPU gate); when the two
merge, `y` also stays on the device.

**Design.**

1. **Fact** `CategoryIr::write_delta(f) -> SecondaryMap<ObjectId, Delta>` in `algo.rs`, keyed by `LoopMerge`
   object: `Delta::Rows { stride, start: Affine{a, b} }` meaning the back-route value differs from the merge
   value only at cells `[a·t + b, a·t + b + stride)` for counter `t`, or `Delta::Top`. Derived by walking from
   the merge to its back-route object: every step is an `Update` whose index is affine in the canonical
   counter (`loop_plan`'s counter object, through `Add`/`Mul` by constants — the `Rng` lattice with one
   symbolic variable), the union of the indices is one contiguous range per iteration; anything else is `Top`
   (the status quo). `matmul4_loop` gives `Rows{stride: 1, start: t}`.
2. **Site rewrite** in `FnEmit::emit_tiled_map` (before the realization is chosen): for a tile site in a loop
   body whose `a` operand is a merge (or its `Proj`) with `Delta::Rows{stride, start}`, `site.a.ci == stride`,
   `site.a.clane == 0`, `site.b` not in the loop's SCC, `site.b.ci == 0`, and whose output `y` dies inside the
   iteration (`death(y)` before `LoopBack`, not carried, not escaping):
   - allocate a retained buffer for `y` once, outside the loop (static size `rows·c`);
   - on the first pass (a host flag set at `LoopEnter`) run the whole site into it — the OFF path, so no
     reasoning about the init value is needed;
   - on every later pass run the site restricted to rows `[start(t), start(t)+1)` (a `TileSite` with
     `rows = 1`, `a.base += start·stride`, output offset `start·c`), through the same `emit_tiled_map` (the
     restricted site picks its realization by the same gates: GPU above the gate, CPU rung below);
   - bind `y`'s pointer to the retained buffer. A consumer of `y` that is itself row-local and in the body
     gets Δ propagated (the rewrite recurses with `y` as the carried operand); a row-mixing consumer reads the
     buffer whole.
   Rung `retain` (OFF = full site each iteration, today's code).
3. **Programs** in `gen_tiles.py`, straight-line `.mapal` text generated by Python (the canonical-loop rule is
   per fn, so the row write is `d` `Update` morphisms `x[t*d + j] <- v_j`, `j` unrolled; `d` static):
   - `decode_T_d` (`T = 1024`, `d = 256` for timing; `T = 16`, `d = 8` for the differential): carried
     `x: [f32; T·d]` from a formula; per iteration `t`: `x[t·d + j] <- f(t, j) + o[j]` for `j < d` (the new
     token row depends on the previous output, so the loop is sequential), `K = x·Wk`, `V = x·Wv`
     (`T×d` sites over row-local `x`, `Wk`/`Wv` invariant formula arrays), `s = q_t·Kᵀ` (`1×T` site over all
     `K` rows: row-mixing), `o = s·V` (`1×d`), carried `o`; after the loop print `o[0]`, `o[d−1]`, a fold sum
     of `o`. No softmax (a map on `s` is an elementwise consumer and changes nothing in the rule; add it if the
     library floor wants it).
   - `loop_rewrite_T_d` (negative): the iteration rewrites every row of `x` (`Delta::Top`) — the rule must
     retain nothing; the OFF and ON `.ll` are identical.
   - `loop_mm_1024_8` from `resident` is also a negative here (`x` is produced by the site, not written by
     `Update`: `Top`).
   - `tile_bench --decode T d`: a hand-written KV cache with cuBLAS (`sgemv` per step into preallocated
     `K`, `V`, scores, output) — the library floor the user would write by hand; Mapal must deduce it.
4. **Trace**: `MAPAL_GPU_TRACE` reports per-launch `rows`; the decode program ON shows `rows=1` launches
   after the first iteration, OFF shows `rows=T` every iteration.

**Tests.** Unit (`mapal-ir/tests/algos.rs`): `write_delta` on `matmul4_loop` is `Rows{1, t}`, on the decode IR
is `Rows{d, t·d}`, on `loop_rewrite` is `Top`, on a non-affine `Update` index is `Top`. Backend: the decode
program's `.ll` has one full-site call outside the first-pass branch and a `rows = 1` site inside the loop for
`K` and `V`, none for `s`/`o`; `loop_rewrite`'s `.ll` is byte-identical ON and OFF. Interpreter oracle: the
decode program's printed outputs equal the oracle on both faces (the exact face bit for bit) with the rung ON
and OFF — the box differential at `T = 16, d = 8` and at `T = 1024, d = 256`, all three parts. Box: e2e wall
and trace of `decode_1024_256` ON vs OFF (3 processes each), vs `tile_bench --decode`; the `mm_*` rows as the
control that the kernels are unchanged.

**Expected.** Per step the two projections go from `T·d·d` to `d·d` MACs and from `T·d` to `d` elements of
transfer (`x`'s upload shrinks to the written row once `resident`'s carry lands; until then `x` is uploaded
whole, which the OFF arm pays too). No loop program has been measured on a GPU profile; the OFF arm is the
first number, and the gap to the hand-written floor is the measure of what the rule leaves on the table
(the row-mixing `s`/`o` sites run as 1-row sites either way).

### 2.5 `host-runtime` — context before the program needs it, stream-ordered pool, trace split (host-link, S)

**Deducible from.** `tile_plan` on a GPU profile says whether any site offloads (the condition for `GPU_DECLS`,
`lib.rs:422`). nsys: context creation is 109–152 ms, module load 0.4–0.8 ms; the trace's `load_ms`
(139–350 ms) is paid inside the first site, after the host has already built the inputs. alloc+free inside
`mapal_gpu_tile` is 2.5 / 0.8 / 1.6 ms per site (4090 / H100 / 5090).

**Design.**

1. Rung `eager-init`: `mapal_gpu_init()` in `mapal_gpu.c` runs `cuInit`, primary context and module load on a
   `pthread`; `load_module()` joins it. The emitter puts `call void @mapal_gpu_init()` first in `@mapal_main`
   when the module has a GPU site. OFF = today's lazy path.
2. Rung `pool`: the three `cuMemAlloc`/`cuMemFree` in `mapal_gpu_tile` (`mapal_gpu.c:100–115`) become
   `cuMemAllocAsync`/`cuMemFreeAsync` on the stream, default mempool with
   `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = UINT64_MAX` (the reason: the program's device footprint is static, so
   nothing is gained by returning memory between sites). OFF = `cuMemAlloc`. Works inside today's
   `mapal_gpu_tile`; `resident`'s split entries call the same allocator.
3. Trace: `load_ms` becomes `init_ms`, `ctx_ms`, `module_ms`, `wait_ms` (time the first site waited for the
   thread), so the overlap is measured, not assumed.

**Tests.** Unit: a GPU-profile module with a site starts with the init call; a CPU profile and a GPU profile
with no site do not. Golden `.ll`. Box, all three parts: differential 12/12 with each rung off; process wall
time of `e2e_1024`/`e2e_4096` ON vs OFF per rung, 3 processes each; `wait_ms` and `alloc_ms`/`free_ms`
reported; kernel rows vs cuBLAS/CUTLASS unchanged.

**Expected.** Up to 135–350 ms per process moves off the site and overlaps the host's input generation (bound:
`min(load_ms, host work before the first site)`); alloc+free 2.5 / 0.8 / 1.6 ms per site → ~0.

### 2.6 `fma-regs` — spill fixes, then the whole-file FMA register tile (kernel, M–L)

**Deducible from.** Every kernel index is `i64` (`Kernel::op`, `gpu.rs:319`); the largest offset of each
operand, `base + ci·(rows−1) + clane·(c−1) + ck·(k−1)`, is a site extent and the array length is in the operand
type. Each panel copy keeps its own `(gsrc, sdst)` pair, built with `udiv`/`urem` of `tid + p·threads`
(`gpu.rs:836–848`) that LLVM cannot see is affine; SASS shows `STL.64`/`LDL.64` of four global pointers feeding
the LDGs, 48–68 B spilled on every part. The sync FMA path stores transposed A with `pad_a: 0`
(`profile.rs:998`), the 4-way bank conflict the SASS review measured; `conflict_free_pad` (`profile.rs:266`)
already pads the async and mma paths. `gpu_tile` step 1 (`profile.rs:957`) spends half the file on
accumulators (8×8) and its `regs_est = 2·tm·tn` omits the prefetch panel's registers, so `min-blocks` and
`prefetch` fight under the 128 cap (SASS review 1). cuBLAS's Hopper kernel is 128 threads × 128 accumulators,
128×128 block; cuBLAS over CUTLASS-128 is 1.11–1.30× on H100, 1.00–1.18× on the 5090, ≈1.0 on the 4090, and
Mapal ≈ CUTLASS-128. The whole-file tile needs the registers the spills take (kernel: R6 after R2), so the two
are one item.

**Design.** Four rungs, each OFF = today's code:

- `idx32`: `Kernel.idx: &str` = `i32` when every operand's max offset and `rows·c` are < 2³¹, else `i64`;
  `op` emits in that type, pointers stay 64-bit (`getelementptr` with a zero-extended index).
- `affine`: copy `p`'s coordinates with `Q = threads/lanes`: `r = tid/lanes + Q·(p mod bm/Q)`,
  `c = p div (bm/Q)` when `Q | bm`; `r = (tid/lanes) mod bm`, `c = (tid/lanes)/bm + p·Q/bm` when `bm | Q`
  (powers of two: always one of the two) — so copy `p` is `base₀ + imm(p)`: one register pair per operand,
  immediates from the lane map's period.
- `pad`: `gpu_tile` sets `pad_a = conflict_free_pad(bm, sync store pattern)`; the kernel's `sa` stride reads it.
- `wide-fma`: `gpu_tile` step 1 becomes the `wide` rule applied to FMA: a thread holds `tm·tn` accumulators +
  `2·(tm+tn)` fragments (double-buffered) + the prefetch panel's registers when that rung is on +
  `FMA_ADDR_REGS` (a stated policy constant read off ptxas, like `MMA_ADDR_REGS`) ≤ `regs_per_thread`. Largest
  `(tm, tn)` with `tn ∈ {tm, 2tm}`, square-first: 255 → 8×16 (128 + 48 + 32 = 208). Step 2: threads =
  `regs_per_sm / regs_est / GPU_RESIDENT_BLOCKS` rounded down to a power of two (65536/208/2 → 128); the thread
  grid split square-most on the block tile (`ty·tm ≈ tx·tn` → 16×8 → 128×128). `min_blocks` then emits
  `minnctapersm = 2` with 256 threads/SM = 256 registers each: no cap. Prefetch depth (`gpu.rs:prefetch_depth`)
  reads the same `regs_est`.

**Tests.** Unit: the max-offset rule picks `i32` for every S52 bench site and `i64` for a synthetic site whose
operand exceeds 2³¹ elements; the affine decomposition equals the `udiv`/`urem` one for every `(p, tid)` of
the S52 geometries (a property test over the lane map); `pad_a` on the sync path is bank-conflict-free for
the transposed store pattern; `gpu_tile_geometry_is_derived_from_device_facts` extended — Ada/Hopper/Blackwell
facts give 8×16, 128 threads, 128×128, `regs_est` 208; the fill rule still halves for small sites. Box, all
three parts: differential 12/12 (incl. `rm_128_f64`, `odd_100` naive); ptxas registers and spill bytes per arm
(gate for the first three rungs: 0 B); `_c` and `_exact` ON vs OFF per rung at 1024–8192 vs cuBLAS/CUTLASS.

**Expected.** The measured defects are the 48–68 B spills and the 4-way STS conflict (the SASS review attributes
the ≈5% contract gap on the 4090 to the stalls they cause); the whole-file tile targets cuBLAS/CUTLASS-128:
H100 1.11–1.30×, 5090 1.00–1.18×. Ceiling: contract ≈1.0× on every part at ≥2048. The 1024 regime may lose
blocks under the wide tile; `warp-fill` covers that.

### 2.7 `warp-fill` — fill the schedulers, not just the SMs (kernel, S)

**Deducible from.** `gpu_tile` step 3 and `gpu_mma_tile` step 2 halve the *thread grid* while blocks < SMs; at
1024² the 4090 runs 4 FMA warps per SM (1 per scheduler) and 2 TF32 warps (0.5 per scheduler); v6: FMA
0.48–0.78×, TF32 0.42–0.51× on all three parts, and async's only wins are there (latency-bound). An SM has 4
schedulers since sm_70 — an ISA table fact keyed by `sm()`, like `has_mma_sync`.

**Design.** `Gpu::schedulers()` (4 for sm ≥ 70). Fill rule: grid warps ≥ `2 · schedulers · sms` (2 is
`GPU_RESIDENT_BLOCKS`'s reason applied per scheduler: one warp issues while the other waits). Shrink the
register tile first — halve the longer side of `tm×tn` (FMA) or `mt×nt` (mma) down to the vector width — and
only then the thread grid as today. The rule overrides `wide` for small sites. Rung `fill`; OFF = today's
block ≥ SMs rule. `gen_tiles.py`: `mm_3072` (wave-quantization probe: 576 tiles over 256 slots on the 4090) and
`mm_1536`; `box_profile.sh`: nsys at 1024 and 8192 too (which kernel cuBLAS picks there; decides row 17).

**Tests.** Unit: on Ada facts at 1024 the rule gives ≥ 1024 warps (4×4 register tile, 64×64 blocks) and at 4096
is a no-op (geometry unchanged, the existing pin test still passes); mma: 32×32 warp tile at 1024. Box, all
three parts: differential 12/12; `_c`, `_exact`, `_tf` ON vs OFF at 1024, 1536, 2048, 3072 vs cuBLAS/CUTLASS.

**Expected.** Warps per scheduler at 1024 go 1 → 2 (FMA) and 0.5 → 2 (TF32); today 0.48–0.78× / 0.42–0.51×.
The ≥2048 rows are the control (unchanged geometry).

### 2.8 `wgmma-hopper` — Hopper TF32 through `wgmma` with swapped operands (kernel, L, H100 only)

**Deducible from.** cuBLAS's H100 TF32 kernel is `wgmma`; it measures 1.56–2.13× CUTLASS's sm80 `mma.sync`,
and Mapal's `mma.sync` kernel is 0.74× CUTLASS, 0.34–0.43× cuBLAS. `wgmma` TF32 smem operands must be K-major;
the site's `a` (`ci = k, ck = 1`) is K-major and `b` (`ck = c, clane = 1`) is N-major — so compute
**Cᵀ = bᵀ·aᵀ**: `b` is the register A-operand (loaded `ld.shared.b32` from the N-major panel, any layout,
rounded in the load path), `a` the smem B-operand in its native layout. `wgmma` reads smem operands in flight
without conversion, so the staged `a` panel must be rounded to TF32 once per element at staging (kernel R7 —
this also replaces the per-fragment `cvt.rna` that costs 128 `FSETP` per 128 `HMMA` today, and fixes the
tie mismatch: ADR-0039 says RNE, the kernel emits `rna`). No transpose pass, no ADR-0039 types; the same
kernel serves bf16 when D1 lands. `Gpu::has_wgmma()` = `sm() ≥ 90` and arch `sm_90a`.

**Design.** Profile `h100-pcie` arch → `sm_90a` (bundle `h100-pcie:sm_90a`; llc accepts it locally).
`gpu_wgmma_tile` in `profile.rs`: warpgroup = 4 warps (ISA); instruction `m64nNk8.f32.tf32.tf32`, A from
registers; `N` the largest multiple of 8 ≤ 256 with `N/2` accumulators + `2·4·(bk/8)` A fragments (double-
buffered) + `MMA_ADDR_REGS` ≤ `regs_per_thread` (whole-file rule); warpgroups per block from `regs_per_sm`
over `GPU_RESIDENT_BLOCKS`; `bk` and stages from the block's smem share (228 KB); the `a` panel in the
canonical no-swizzle K-major core-matrix layout the matrix descriptor expects (leading/stride byte offsets
derived from `bk`), `b` panel padded by `conflict_free_pad` for the fragment pattern. Kernel in `gpu.rs`:
cp.async staging (existing) with a rounding pass over the staged `a` panel, `wgmma.fence` / `mma_async` /
`commit_group` / `wait_group` as inline asm in the device unit (no TMA, no warp specialization — rows 19–20);
epilogue transposes the Cᵀ tile through the drained smem buffers and writes `c` rows with 16-byte stores.
Rung `wgmma` (TF32 face only; the exact face is untouched); OFF = the `mma.sync` kernel. First step before any
rent: the PTX ISA text confirms register-sourced A for `.tf32` (the kernel report's check), `llc -mcpu=sm_90a`
accepts the kernel and `ptxas` (on the box) reports no spills.

**Tests.** Unit: `gpu_wgmma_tile` derived from Hopper facts (pinned numbers), `None` on sm_89/sm_120; the
descriptor fields for the S52 geometries; the staged rounding is RNE (a unit test against ADR-0039's rule).
Box, H100 PCIe: differential 12/12 on the exact and contract faces (unchanged kernels); the TF32 bound gate in
`tile_bench` passes; `_tf` ON vs OFF at 1024–8192 vs cuBLAS-TF32 and CUTLASS; ptxas registers/spills; nsys
kernel time.

**Expected.** The measured headroom is cuBLAS/Mapal = 2.3–2.9× on H100 TF32; a `wgmma` kernel at CUTLASS-sm80
issue efficiency sits between today's 0.34–0.43× and cuBLAS's 1.0×. The item ships whatever it measures; a
kernel that does not beat the `mma.sync` arm stays OFF and the result is still the evidence row 19 needs.

## 3. Order and sessions

`clock-fence` first (S; makes `iter ms` meaningful for everyone), then `arch-rungs` (S; the defaults every
other arm is measured against). `resident`, `retain-delta`, `host-runtime`, `fma-regs`, `warp-fill`,
`wgmma-hopper` are independent and can run in parallel worktrees; each rents its own sessions (three parts,
or one for `wgmma-hopper`). After merge, one combined bundle re-measures every arm against the merged
defaults and writes the final rung table — the interaction between two deduced rules is measured, not
assumed; `resident` + `retain-delta` merged is where the decode program's retained `K`/`V` stay on the
device, and that composed number is the one the README gets.
