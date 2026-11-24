# gapfix tensor-tile — gaps-disasm C3 / C4 / C11 on the tensor tiles (2026-11-25)

Branch `gapfix/tensor-tile` from `fac7691b` (the gaps-disasm merge). Scope: C3 (too little in flight), C4 (the tile grid
rounding over the SMs' matrix pipes), C11 (the raster gate ignoring the `b` panel's reuse distance). Raw output (never
in git): `.claude/raw-evidence/local-rtx-5080/gapfix-tensor-tile/` — `abba_local_{1..4,rounds}.txt` (RTX 5080, ABBA,
`run_mapal.sh` under `bench.lock`, value-gated), `rent-{rtx-4090,a100,h100,rtx-3090}/results-*.txt` (rented parts:
the slim `box_run.sh` — every arm value-gated against the exact-face oracle, the box's own `ptxas -v` + SASS
counts, one `MAPAL_GPU_TRACE` process, then 6 repetitions ABBA, `MAPAL_GPU_PICK=rules`).

Every A/B is `fac7691b`'s emitter (`base`) against this branch (`gf`); the base arms of the rented bundles were
built by the gaps emitter (`fbf236c6`), whose emission equals `fac7691b`'s on every measured stem of those parts
(normalized compare: kernel-name hashes, metadata numbers, `; mapal.` lines aside).

## What changed, per cause

| cause | fact / property / rule | OFF arm | tests |
|---|---|---|---|
| C11 | `Gpu::b_rereads` (row-major puts a `b` panel's readers `gx` launches apart: past a wave every `b` panel streams between them, within one `(gx²/wave·bn + bm)·k·eb`; beyond `l2_keep_bytes` each further block-row / wave re-reads `b`), `Gpu::dram_bound` + `DRAM_BOUND_PM` (0.86 of the pipe's time); `Gpu::raster` groups where one block's panels leave the L2 (the S54 anchor, kept) **or** row-major's re-read traffic leaves the site DRAM-bound at the tile's own rate (matrix op per format, the FMA tile's plan rate) and `b`'s storage bytes | `--gpu-off=l2-gate` (ungated, as before) | `raster_is_the_wave_panel_minimum_behind_the_l2_gate` (+5080 / 3090 cases), `raster_fires_where_one_blocks_panels_leave_the_l2` (the H100 PCIe's 8192 TF32 square now groups), `tensor_tile_rules_on_generated_shapes` (raster > 1 exactly where the gate says, the group divides the grid), emulator values under a grouped walk; 4 replay lines (5080, 4090, A100, H100) |
| C3 | rung `mma-fly`: `TargetProfile::gpu_mma_fly` / `mma_fly_pick` — a tensor tile whose fragments are read as stored (`ldmatrix` of the 16-bit format: no per-element conversion), with an operand past the L2 keep, whose S52 panel's DRAM time a panel (its blocks' streamed bytes over the blocks of a row-major wave that share the panel, at the SM's DRAM share) fits its products' time `t` but `t` < that time + `lat_ns`, takes the ring that keeps `stream_fly_per_sm` in flight past `wait_group(stages − 2)` (`blocks_per_sm · (stages − 2) ·` the streamed stage bytes), each panel at least a latency of products; the most blocks an SM, then the deepest panel, within the S52 residencies. Properties read: `stream_fly_per_sm`, `lat_ns`, `dram_bytes_per_ns`, `l2_keep_bytes`, the matrix rate. No new property | `--gpu-off=mma-fly` | `tensor_tile_rules_on_generated_shapes` (the ring differs from the S52 panel only on a DRAM-resident, convert-free tile; same block and grid; ≥ the part's in-flight bytes; never more blocks an SM; the kernel's `cp.async.wait.group(stages − 2)`), `tensor_tile_values_through_the_emulator`; replay line |
| C4 | attempted (rung `tile-rounds`, then reverted): the block of half the warps at the same warp tile where `⌈tiles/sms⌉·bm·bn` drops > 5 % | — | — |

Tests: `cargo test -p mapal-backend-llvm --lib` 129 pass (`rung_rules_replay` 163 picked / 74 null / 16 mixed / 23
open); `tests/tensor_tile.rs`: plans over 64 generated GEMMs × {`bf16 | f32`, `tf32 | f32`} × the RTX 5080 / 4090 /
A100 / H100 properties — 193 tensor kernels, 17 rings (14 at the S52 residency), 22 walks grouped by the reuse
gate, 13 re-reading grids the pipe hides kept row-major; values: 6 generated GEMMs (`512 × 6144 × 64`, the L2 keep /
latency / DRAM rate overridden by `--cost=` so both rules act), 6 ring kernels and 6 grouped walks, 18 emulated runs
equal to the interpreter (serial, serial-2, reversed; OFF arm too).

## Intent → realized → explained

**C11.** Intent: the kernel body is unchanged (the raster maps `ctaid` to tiles); DRAM traffic drops from `(rereads
+ 1)·|b|` to about `|b|`. Realized (box `ptxas` + SASS, both arms): identical instruction counts, `DEPBAR` operands and
registers on every part (e.g. 4090 bf16: 128 HMMA, 16 LDGSTS, `DEPBAR.LE SB0, 0x0`, 220 registers both arms; H100
TF32 `wgmma-ws`: 16 HGMMA, 255 registers both). Explained: 2048 × 28672 × 4096 re-reads its 235 MB (bf16) / 470 MB
(TF32) `b` once per block-row in row-major (224 blocks a row past the wave): 15 re-reads; grouped, once. The H100's
128-row TF32 projections (`gx` 64 within its 132-CTA wave, 2 waves) re-read `b` once past its keep.

| workload | part | OFF (fac7691b gate) ms | ON ms | leader (leg) ms | SASS / usage |
|---|---|---:|---:|---:|---|
| pj_m2048_n28672_k4096_bf16 | rtx-5080 | 4.580 | 3.988 | 4.000 (torch-bf16-compile-cg) | same kernel; `b` read once, not 16× |
| pj_m2048_n28672_k4096_tf32 | rtx-5080 | 9.073 | 8.508 | 8.057 (lt-tf32) | same kernel |
| pj_m2048_n28672_k4096_bf16 | rtx-4090 | 4.281 | 2.860 | 2.757 (torch-bf16-compile-cg) | same kernel |
| pj_m2048_n28672_k4096_tf32 | rtx-4090 | 8.313 | 6.041 | 5.500 (torch-compile-ro-tf32) | same kernel |
| pj_m2048_n28672_k4096_bf16 | a100 | 3.094 | 2.514 | 2.159 (torch-bf16-compile-ro) | same kernel |
| pj_m2048_n28672_k4096_tf32 | a100 | 5.926 | 5.157 | 4.521 (torch-eager-tf32) | same kernel |
| pj_m128_n28672_k4096 (f32 split) | a100 | 1.256 | 1.261 | — | raster 2: null |
| pj_m128_n4096_k4096_tf32 | h100 | 0.0994 | 0.0874 | 0.0440 (cublas-tf32-fresh) | same `wgmma-ws` kernel, raster 4 |
| pj_m128_n4096_k14336_tf32 | h100 | 0.2286 | 0.2077 | 0.1053 (lt-tf32) | raster 4 |
| pj_m128_n6144_k4096_tf32 | h100 | 0.1136 | 0.1038 | 0.0524 (lt-tf32-fresh) | raster 2 |
| pj_m2048_n28672_k4096_bf16 | h100 | 1.687 | 1.543 | 0.6085 (torch-bf16-compile-ro) | raster 16 |
| pj_m2048_n28672_k4096_tf32 | h100 | 1.946 | 1.902 | 1.282 (lt-tf32) | raster 8 |
| llm_dec_b32_l2048_bf16 (FMA tiles, 32 rows) | rtx-3090 | 10.788 | 10.802 | — | raster 2: null |

**C3.** Intent: the 2-stage ring waits `DEPBAR.LE SB0, 0x0` (one panel in flight, drained at every wait); the ring
keeps `stages − 2` panels in flight through the wait. Realized (RTX 5080, `sassum.sh` on the arm PTX): S52 `bk` 32 × 2
at 2 blocks — `DEPBAR.LE SB0, 0x0`, 16 LDGSTS + 64 HMMA a panel (ptxas duplicates the loop: 128 HMMA); the ring `bk`
16 × 4 at 2 blocks — `DEPBAR.LE SB0, 0x2`, 8 LDGSTS + 32 HMMA a panel, 41 984 B shared memory. Explained: the panel's
8 KiB of streamed `b` for 2 blocks takes 1546 ns at the SM's 10.6 B/ns, its products 1672 ns, the load 573 ns more —
exposed at every wait; four stages keep two panels in flight through it.

| workload | part | OFF (S52 panel) ms | ON ms | leader (leg) ms | SASS / usage |
|---|---|---:|---:|---:|---|
| pj_m128_n28672_k4096_bf16 | rtx-5080 | 0.400 | 0.344 | 0.3174 (triton-bf16: `DEPBAR.LE 0x4`, 4 stages) | `DEPBAR 0x0` → `0x2`, `bk` 32×2 → 16×4 |

How the gate was found — rings measured and refused (each a value-gated ABBA; the shipped rule keeps the S52
panel on all of them):

| site | part | S52 ms | ring ms | ring | why refused |
|---|---|---:|---:|---|---|
| pj_m128_n28672_k4096_bf16 | rtx-5080 | 0.400 | 0.372 | `bk` 32×5, 1 block | the residency first: `bk` 16×4 at 2 blocks 0.344 |
| pj_m128_n28672_k4096_tf32 | rtx-5080 | 0.699 | 0.718 / 0.725 | `bk` 16×5 @1 / 8×4 @2 | fragments converted on read (TF32) |
| pj_m128_n28672_k4096_tf32 | rtx-4090 | 0.574 | 0.608 | `bk` 8×4 | TF32 |
| pj_m128_n28672_k4096_tf32 | a100 | 0.541 | 0.564 | `bk` 16×4 | TF32 |
| pj_m128_n4096_k14336_tf32 | rtx-4090 / a100 | 0.350 / 0.349 | 0.364 / 0.351 | `bk` 16×5 / 32×4 | TF32 |
| pj_m128_n6144_k4096_tf32 | rtx-4090 / a100 | 0.196 / 0.169 | 0.194 / 0.171 | | TF32 (null) |
| pj_m128_n4096_k4096_tf32 | a100 | 0.1545 | 0.1541 | | TF32 (null) |
| pj_m128_n28672_k4096_bf16 | rtx-4090 | 0.320 | 0.339 | `bk` 16×4 | DRAM-bound: 2139 ns of DRAM > 1626 of products |
| pj_m128_n28672_k4096_bf16 | a100 | 0.285 | 0.283 | `bk` 32×4 | DRAM-bound (2378 > 1470): null |
| pj_m128_n4096_k14336_bf16 | a100 | 0.222 | 0.215 | `bk` 64×4 | DRAM-bound: **forgoes a 3.5 % win** (open) |
| pj_m128_n4096_k4096_bf16 | a100 | 0.0880 | 0.0925 | `bk` 64×4 | DRAM-bound |
| pj_m128_n4096_k14336_bf16 | rtx-4090 | 0.200 | 0.199 | `bk` 32×5 | DRAM-bound (null) |
| pj_m128_n4096_k14336_bf16 | rtx-5080 | 0.2207 | 0.2311 | `bk` 32×5 | a `bk` 32 panel of the 64 × 64 tile: 418 ns of products < 573 ns latency |
| llm_dec_b32_l2048_bf16 | a100 | 8.154 | 8.173 | 32-row decode tiles | null |

Open in C3: the A100's 64 × 64 bf16 tile at `k` 14336 (S = 4 split, 512 CTAs) won its ring 3.5 % where the same
tile at `k` 4096 (128 CTAs, 0.6 waves) lost 5.1 %; the DRAM-bound gate refuses both. The separating fact is likely
the grid's fill (an SM's blocks against its slots sets the SM's DRAM share), not read yet. Also open: why the TF32
tile's two-stage ring already hides its loads at the same panel bytes and products as the bf16 one (the 5080's TF32
twin runs at its pipe's rate: 0.699 ms against 0.642 pipe-rounded) — the realized fragment-path cost is not a
property; the gate reads the conversion as the layout fact it is.

## C4 — attempted, open

Rung `tile-rounds`: a tensor tile's block takes half its warps at the same warp tile where its grid's rounds on the
SMs' matrix pipes (`⌈tiles/sms⌉·bm·bn`) drop more than 5 % (the RTX 5080's 2048 × 4096 × 4096 bf16: 512 tiles of
128 × 128 = 7 rounds where 6.10 would do; 128 × 64: 6.5). Measured (an arm with the rule unguarded, `abba_local_rounds.txt`):

| workload | part | 128 × 128 ms | 128 × 64 ms | the half block's panel |
|---|---|---:|---:|---|
| pj_m2048_n6144_k4096_bf16 | rtx-5080 | 0.940 | 1.288 | `bk` 16 × 2 at 4 blocks (its 24 KB share) |
| pj_m2048_n4096_k4096_bf16 | rtx-5080 | 0.678 | 0.996 | `bk` 16 × 2, S = 4 atomic |
| pj_m2048_n6144_k4096_tf32 | rtx-5080 | 2.024 | 2.70 | `bk` 8 × 2 |

The half block's share of shared memory forces a one-k-step panel (+37–45 %); guarded to keep the panel depth, the
rule never fires on the five parts' properties, so it was removed (commit `00e0cef4`). The leaders' geometry for these
rows is another family — 64 × 64 blocks of four 32 × 32 warp tiles with six stages at two CTAs an SM (CUTLASS
`64x64_32x6`, 2048 CTAs) — whose choice needs the warp tile's realized matrix-pipe efficiency (`MatrixOp` issue
interval / latency from `mma_calib.cu`, the gaps doc's day-0 property) to price against the rounds it saves. Not
added here: no calibration was run.

## Side finding (not this branch)

`fac7691b`'s own emission differs from `fbf236c6`'s on the RTX 3090 for `smlp_4096_256_1024`, `indep_512`,
`mlp_512` and `causal_1024_256`: the 4096 × 1024 × 256 FMA site takes stream-K (164 CTAs) where `fbf236c6` launched
it unsplit, and the rented 3090 measured 0.279 → 0.340 ms (+22 %, `rent-rtx-3090/results-rtx-3090.txt`, `base` =
`fbf236c6`, `gf` = this branch). This branch does not change that site (its raster stays 1: `b_rereads` 0). Likely
the 3090's measured `l2_keep_bytes` landing in `fac7691b`; for the integrator to re-check.

## Rentals and spend

| part | offer $/h | instance lifetime | cost |
|---|---:|---:|---:|
| RTX 4090 (i7-13700K) | 0.402 | 3.7 min | $0.03 |
| A100-SXM4-40GB | 0.801 | 4.3 min | $0.06 |
| RTX 3090 (EPYC 7543) | 0.204 | 3.1 min | $0.01 |
| H100 80GB HBM3 (SXM) | 4.669 | 2.6 min | $0.20 |

Total ≈ $0.30 (labels `mapal-gf-tt-<gpu>`, each destroyed by `vast_run.sh`'s exit trap and verified absent).

## For integration

- **CLAUDE.md facts table**, the backend `Gpu` properties row — add: "`b_rereads` (row-major's `b` re-reads past
  `l2_keep_bytes`: a panel's readers `gx` launches apart) and `dram_bound` (`DRAM_BOUND_PM`) — the raster's reuse
  gate (C11); `gpu_mma_fly` / `mma_fly_pick` — a convert-free tensor tile's ring where its panel's DRAM time fits its
  products but not their latency, keeping `stream_fly_per_sm` past `wait_group(stages − 2)` (C3)".
- **benches/results/evidence/local-rtx-5080/README.md**: a row for this file (C11: 2048 × 28672 × 4096 bf16 / TF32
  1.15 → 1.00 / 1.13 → 1.06 on the 5080, 4090 1.55 → 1.04 / 1.51 → 1.10, A100 1.43 → 1.16 / 1.31 → 1.14, H100's
  128-row TF32 projections −9 to −12 %; C3: 128 × 28672 × 4096 bf16 1.26 → 1.08 on the 5080).
- **matrix.***: the rows above re-time on the next integrated run (5080 column: pj_m128_n28672_k4096_bf16,
  pj_m2048_n28672_k4096_{bf16,tf32}).
- **rules.rs `OPEN`**: `("l2-gate", "h100", "off", "-")` records the old gate's loss; the gate itself now groups those
  sites (the new `l2-gate h100 on` replay line) — drop the OPEN entry if the integrator re-reads that line as the new
  gate's.
- **docs/tailored-fixes.md**: nothing (no tailored fix).
