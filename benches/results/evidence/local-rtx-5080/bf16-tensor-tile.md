# local-rtx-5080 — bf16 tensor-tile: the 16-bit tensor tile walked by stream-K (`mma-walk`) (2026-11-25)

Branch `bf16/tensor-tile` from `bb9bbed7` (the full-model integration, `full-model.md`). Scope: trace (a) of the bf16
class — the projections' tensor tiles capped by their grids' rounds over the 84 SMs. Raw output (never in git):
`.claude/raw-evidence/local-rtx-5080/bf16-tensor-tile/` (`prog/` the model-shaped projections, `out/` their builds
and CUPTI traces, `model/` the bf16-class model builds and traces, `sass/` ptxas / cuobjdump, `changed/` the suite
emission diff, `campaign.log`, `tests.log`, `tools/`). Every timed process held the bench lock and ran on CPUs
16-23,48-55 (`nice 10`); the box was shared (load average 18–123 during the session).

## What changed

| what | where | reads | OFF arm |
|---|---|---|---|
| **rule `mma-walk`**: a 16-bit tensor tile (no split form, free order, no radix / window / forced cut / bulk stage) whose grid's rounds over the SMs' matrix pipes cost more than the same products spread evenly is walked by stream-K over every resident slot | `func/gpu.rs` `mma_walk`, `MmaWalk` (`ns`, `pays`, `shared_pieces`), called first in `mma_combine`; `resized` gives the walk its tile | `sms`, the matrix rate (`mma_half_per_clk` × clock), `atomic_ops_per_ns`, `zero_ns`, `fill_bytes_per_ns`, `lat_ns`, the DRAM rate — properties only; no new fact or property | `--gpu-off=mma-walk` (equal emission to `bb9bbed7`); `--gpu-on=mma-walk` walks every admitted site (measurement / emulator arm) |
| **the walk's tile**: the 16-bit tile at its whole warp tile and warps — neither `fill`'s warp-tile halving nor the halving of a block's warps until the grid has a tile an SM | `profile.rs` `gpu_mma16_walk_tile` (`gpu_mma_geom_walk`) | the same register / shared-memory residency rules | — |
| **the emitter**: `mma16` walks pieces (`Kernel::segment`): accumulators in the entry block, a predicated prologue over the piece's panels `[p0, p1)` behind a barrier, the same pipelined `k` loop; `mma_walk_store`: a whole piece stores as the ordered kernel (its group's members on an epilogue group), a shared piece adds its partials — `red` into the zeroed output, or `red` into the group's workspace slot plus the tile's ticket (`streamk-epi`'s), the last piece evaluating the members | `func/gpu.rs` `mma16`, `mma_walk_store`; `ticket_bytes` sizes the tensor tile's slots | — | — |

The price (`MmaWalk::ns`): the busiest SM's products at its pipe's rate (`rate / sms`) — the ordered grid
`⌈tiles / sms⌉` tiles, its `k` cut (`mma_pieces`) `⌈tiles·S / sms⌉` pieces, the walk `blocks_per_sm · iters`
panels plus a refill (`lat_ns`) a piece its blocks start; a cut or a walk adds the zero pass (the walk at the larger
of the output and the group workspace) and its partials at `atomic_ops_per_ns` (the 512 × 2048 × 8192 cuts measured
8 / 16 µs for 4 M / 8 M adds, the rate's 10 / 20); every time floors at the operands' bytes at the DRAM rate. Taken
where it saves `MMA_WALK_PM` (950 ‰, `MMA_CUT_PM`'s margin) of the better of the ordered grid and its cut. The walk
visits tiles row-major, so where gapfix tensor-tile's C11 gate would group the walk tile's own grid (`Gpu::raster` > 1:
a block's panels leave the L2, or row-major's `b` re-reads leave it DRAM-bound) the ordered grid keeps its grouped
walk — found by `tensor_tile_rules_on_generated_shapes` (256 × 28672 × 4096 bf16 on the 5080: its 235 MB `b` once a
block-row).

## Intent → realized → explained

**Intent.** The walked kernel keeps the ordered tile's pipelined loop (HMMA / LDSM / LDGSTS a panel, one
`cp.async` ring, no spill), launches `ctas ≤ sms × blocks_per_sm`, stores whole tiles with `STG` and adds shared
tiles' partials with fire-and-forget `REDG`; per SM it runs `blocks_per_sm · iters` panels where the ordered grid
runs `⌈tiles / sms⌉` tiles.

**Realized** (ptxas -v, cuobjdump, sm_120; `sass/`; read by hand here — the three walked `benches/algorithms` bf16
projections went through `evidence.py` since: `full-model.md` *Review fixes* 3, 232 checks, 0 mismatch):

| kernel | regs / spill | HMMA | LDSM | LDGSTS | BAR | REDG / ATOMG | STG | grid |
|---|---|---:|---:|---:|---:|---|---:|---|
| 512 × 2048 × 8192 ordered (64 × 128, 8 warps of 32 × 32, 3 stages) | 64 / 0 | 16 | 8 | 9 | 1 | 0 | 16 | 16 × 8 |
| the same walked (128 × 128, 8 warps of 64 × 32, 2 stages) | 122 / 0 | 32 | 12 | 8 | 2 | 64 / 0 | 32 | 168 |
| 512 × 8192 × 2048 ordered (128 × 128, 4 warps of 64 × 64) | 190 / 0 | 128 (2 loop copies) | 32 | 16 | 2 | 0 | 64 | 64 × 4 |
| the same walked | 208 / 0 | 64 (1 copy: a runtime trip count) | 16 | 16 | 2 | 128 / 0 | 64 | 168 |
| 2048 × 2048 × 8192 + residual epilogue walked, `atomicrmw` | 224 / 0 | 64 | 16 | 16 | 5 | 0 / 128 | 144 | 168 |
| the same, the slot's adds as inline `red` (this branch) | 224 / 0 | 64 | 16 | 16 | 5 | 128 / 0 | 144 | 168 |

ptxas keeps an `atomicrmw fadd` ahead of the ticket's release fence a returning `ATOMG`; the slot's adds are now
`red.relaxed.gpu` (the fence orders them; the emulator reads the `red` as the atomic add it is). No spill anywhere;
the walk's loop body is the ordered one's per panel. Alone the `red` form was null on the 2048 group (open 4); in the
model the group kernels ran 692 / 246 / 181 µs (down 2048 / o 2048 / down 512) with it against 715 / 256 / 190 with
`ATOMG` (two sessions, different load).

**Explained.** The ordered 512 × 2048 × 8192 grid puts two 134 MF tiles on 44 SMs (40 hold one): 268 MF on the busiest
SM, 216 µs (1.243 Tflop/s an SM). The walk puts `2 × 98` panels of 1.05 MF on every SM: 205.5 MF, 0.767 × 216 =
165.6 µs predicted, 164.2 measured. The 512 × 8192 × 2048 grid's 256 tiles put 4 on the busiest SMs (4 × 48.3 =
193 µs, measured 192.7): the walk's 205.5 MF an SM is 0.767 of it plus ~13 µs of partials and zero pass — 161.2
measured. Where the ordered grid is already within 7 % of even (1024 tiles: 13 rounds where 12.19 would do) the
walk's 64 MB zero pass and partials cost more than it saves (685 against 628 µs): refused.

## The model's projections alone (CUPTI medians of 3 processes; the program's printed cells agree)

`prog/{qo,kv,gu,dn}_{512,2048}.mapal`: the weight `N × K` bf16, `x` a stored map, `fold: bf16 | f32`. Peak: the
calibrated pipe (`mma_half_per_clk` 512 × 84 × 2.45 GHz = 105.4 Tflop/s); the part ran at ~2.7 GHz (the 1024-tile
grid's per-tile time), so % can pass 100.

| site (T × N × K) | ordered tile, grid | ordered µs | walk µs | walk Tflop/s | % of 105.4 | price |
|---|---|---:|---:|---:|---:|---|
| down 512 × 2048 × 8192 | 64 × 128, 128 | 216.0 | **164.2** | 104.6 | 99 | walk |
| gate / up 512 × 8192 × 2048 | 128 × 128, 256 | 192.7 | **161.2** | 106.6 | 101 | walk |
| q / o 512 × 2048 × 2048 | 64 × 128, 128 | **56.6** | 49.6 (forced) | 86.6 | 82 | ordered (open 1) |
| k / v 512 × 512 × 2048 | 32 × 64, 128 | **18.8** | 19.3 (forced) | — | — | ordered |
| down 2048 × 2048 × 8192 | 128 × 128, 256 | 754.1 | **626.5** | 109.7 | 104 | walk |
| q / o 2048 × 2048 × 2048 | 128 × 128, 256 | 193.1 | **161.9** | 106.1 | 101 | walk |
| k / v 2048 × 512 × 2048 | 64 × 128, 128 | **57.0** | 48.7 (forced) | 88.2 | 84 | ordered (open 1) |
| gate / up 2048 × 8192 × 2048 | 128 × 128, 1024 | **628.4** | 685.1 (forced) | — | — | ordered |

The weights flushed from the L2 first (`*c.mapal`, a 128 MB map before the clock): 512 × 8192 × 2048 196.1 → 163.8,
512 × 2048 × 8192 222.4 → 167.9, 2048 × 2048 × 8192 755.7 → 621.9 µs. With a residual epilogue (`*e.mapal`, the
group's ticket): 512 × 2048 × 8192 227.5 → 157.9 µs; 2048 × 2048 × 8192 802.3 → 811.9 (null, open 4).
`mma_walk_replay` replays all eight lines against the price: six picked, the two forced wins open.

## The full model, bf16 class (`variant.py --compute bf16`, calibrated builds, `model/`)

Emission: at 128 nothing walks (host and device IR equal to the OFF arm); at 512 gate / up (plain) and down (its
residual group, ticket) walk; at 2048 down, q and o (o's `a` f32: `bk` 16) walk, the merged gate + up (`kind=hz`) and
k / v stay ordered. The **f32 class** (512) emits identical host and device IR on both arms: its split tiles are not
walked (no split form is), so they cannot regress.

**Kernel time** (one CUPTI trace an arm, same session; per-kernel medians × launches):

| 512 | OFF µs | ON µs | % of 105.4 (ON) |
|---|---:|---:|---:|
| gate / up (32) | 194.1 | **180.1** | 90.5 |
| down (16) | 235.3 | **180.8** | 90.2 |
| q (16) | 56.9 | 56.7 | 72 |
| o (16) | 62.4 | 67.4 | 60 |
| k / v (32) | 19.3 | 19.3 | 53 |
| zero passes | — | 48 × 7.0 | |
| **projections** | **12.50 ms** | **11.60 ms** | |

| 2048 | OFF µs | ON µs | % of 105.4 (ON) |
|---|---:|---:|---:|
| gate + up merged (16) | 1238.1 | 1231.4 | 106 |
| down (16) | 800.0 | **691.8** | 94 |
| q (16) | 193.2 | **163.6** | 99.6 |
| o (16) | 265.5 | **246.1** | 66 |
| k / v (32) | 56.8 | 56.6 | 72 |
| zero passes | — | 48 × 6.5 | |
| **projections** | **41.77 ms** | **39.45 ms** | |

o at 512 is not walked, yet runs 5 µs slower: a zero pass (16 MB, for the next gate projection's output) now runs
right before it and evicts its operands from the L2 (the kernel text is equal on both arms).

**TTFT** (untraced, ABBA, ms, median [min–max]; load average 18–99 — the spread is the box): 128 OFF 7.30 [7.15–8.86],
ON 7.19 [6.78–7.26] (equal emission); 512 OFF 19.52 [18.83–22.45] (5 runs), ON 18.93 [18.50–22.86] (5); 2048 OFF
79.5 [74.6–106.3], ON 79.7 [78.7–80.7] (3 each; a quieter traced pair earlier: 71.42 / 69.82). Against
`full-model.md`'s 6.95 / 18.12 / 74.05 (load 36). Tokens: the ON arm's partials add in no fixed order, so its first
logit moves in the 4th–5th digit run to run (18.39–18.44 at 512) and its agreement with HF bf16 eager follows the
near-tie at token 67 (67 or 89 of 128); every run stays inside the bf16 references' own spread (`full-model.md`).

## The kernel suite's emission

`benches/local/gpu_changed.sh <out> --off mma-walk`: 35 programs + 2 cells × 5 faces, **same = 45, differs = 0** —
nothing to time (the suite has no 16-bit tensor site). The OFF arm's emission equals `bb9bbed7`'s (512 × 2048 × 8192:
host and device IR byte-equal). Outside the suite, three `benches/algorithms` bf16 programs walk; ABBA (CUPTI, OFF ON
then ON OFF, 3 processes an arm a pass; the table: the second pass's per-process kernel times, medians; values equal
to the printed digits but the last):

| program | OFF | ON | gaps-disasm leader |
|---|---:|---:|---|
| `pt_pj_m2048_n4096_k4096_bf16` (C4's case: 512 tiles, 7 rounds where 6.1 do) | 666.6 | **616.2** | 590.6 (cutlass 64x64_32x6) |
| `pt_pj_m2048_n4096_k14336_bf16` | 2240–2424 | **2081** | — |
| `pt_pj_m128_n4096_k14336_bf16` | 203.0 | **161.8** | 167.2 (cutlass 128x64_64x3 sK5) |

`pt_llm_blkpre_t2048_bf16` / `pt_llm_pre_t2048_bf16` keep their ordered tiles (equal emission).

## Tests

- `cargo test -p mapal-backend-llvm --lib`: `mma_walk_replay` (8 measured walks: 6 picked, 2 open), `rung_rules_replay`
  (one `mma-walk` line, `mixed`: 172 picked / 82 null / 17 mixed / 22 open), `mma_cut_replay` unchanged.
- `tests/mma_walk.rs` (new): `mma_walk_rules_on_generated_shapes` — 48 generated GEMMs × the RTX 5080 / 4090 /
  A100 / H100 properties: every walked tile within the resident slots, its units covering `tiles × panels` with no
  empty block, its warps and warp tile never below the ordered tile's, `red` and `st_whole` / `st_part` in its kernel;
  the OFF arm and `--order=fixed` walk none. `mma_walk_values_through_the_emulator` — 6 generated GEMMs at 512 × 6144
  × 64 (whole and shared tiles), plain and as an epilogue group, the forced walk equal to the interpreter in every
  emulator mode (serial, serial-2, reversed) and the OFF arm. Plans: 192 tensor kernels, 17 walked (all with shared
  tiles), 175 kept ordered. Values: 6 walked GEMMs, 6 walked epilogue groups (the ticket), 36 emulated runs, all equal.
- Neighbours (`tests.log`, `--release`, `RUST_TEST_THREADS=8 MAPAL_TESTGEN_JOBS=4`): lib 136, device_geometry 11,
  epilogue 13, golden_ll 36, half_tile 6, hardware_properties 2, large_gemm 3, site_choice 9, split_tile 7, splitk 5,
  streamk_epi 2, tf32_tile 2, tensor_tile's emulator test — all pass. Two failures found and fixed (`tests2.log`):
  `tensor_tile_rules_on_generated_shapes` caught the walk over a grid the C11 gate groups (the gate above), then a
  walk moving the grid between its `mma-fly` arms (a ring's panels change the walk's price) — that test now holds
  `mma-walk` off on both arms (193 kernels, 15 rings, 12 at the S52 residency, 22 grouped, 13 row-major kept); and
  `mma_walk_values_through_the_emulator` (the emulator now reads the `red` asm as the atomic add).

## Open (attacked, not closed)

1. **The ordered tile's realized pipe fraction is no property.** q / o at 512 and k / v at 2048 walk faster (forced:
   0.876 / 0.854) but the price reads the ordered 64 × 128 tile of 32 × 32 warp tiles at the pipe's rate, which it
   reaches only at 85–89 % (512 × 2048 × 8192: 1.243 Tflop/s an SM against the 64 × 64 tile's 1.39). Next: a
   `MatrixOp` warp-tile efficiency (calibrated per warp tile and warps, `mma_calib.cu`) both prices read.
2. **The fixup's cost.** 48 zero passes a prompt (0.31–0.34 ms) and `red` partials (priced 400 adds a ns); one zero
   pass evicts the next kernel's operands. Next: the shared tiles' partials as plain stores into per-block slots, the
   ticket's last piece summing them in registers (only the tickets zeroed — a runtime change: the zeroed bytes are not
   the workspace's), and a data-parallel first wave (only the tail tiles shared).
3. **In the model the walk gains less than alone** (gate / up −7 % against −16 %): the walk balances only where
   every SM starts at once; the ordered grid's block scheduler absorbs SMs still busy with the previous kernel (PDL).
4. **The 2048 × 2048 × 8192 group alone** (residual epilogue, `y` stored too): the walk is null (800 against 802 µs)
   where the plain walk takes 622 and the model's group 692 against 800; `ATOMG` → `REDG` (above) changed nothing.
5. Not this rule's: o's `a` stays f32 (trace (b): 60–66 % of peak); the merged gate + up at 2048 already at 106 %.

No tailored fixes (`docs/tailored-fixes.md` unchanged).

## For integration

- **CLAUDE.md facts table**, the backend `Gpu` properties row — add: "`mma_walk` / `MmaWalk` (a 16-bit tensor
  tile grid's rounds over the SMs' matrix pipes, `⌈tiles / sms⌉` tiles on the busiest SM, against stream-K over every
  resident slot at the whole warp tile — `TargetProfile::gpu_mma16_walk_tile` — its partials at `atomic_ops_per_ns`,
  its zero pass at `fill_bytes_per_ns`; replay `mma_walk_replay`)".
- **benches/results/evidence/local-rtx-5080/full-model.md** "Where Mapal still loses" item 2 / trace (a): the
  projections' rounds are this file's; the remaining tensor-tile gap is open items 1–2 above.
