# hopper-launch — the TMA kernel's per-call cost, and a persistent tile walk

Branch `fix/hopper-launch` (base 8ff66d34). Problem (cross-arch 5fcb401b finding 3): on the H100 the
warp-specialized `wgmma` kernel fed by TMA (`wgmma-ws`) beat the non-warp-specialized kernel per kernel (1.04–1.45×)
but each program call spent 120–270 µs outside it (OFF 2–7 µs), so 9 of 12 TF32 programs lost. Raw output:
`.claude/raw-evidence/local-rtx-5080/hopper-launch/` (`h100/` box 1, `h100-r3/` box 2 run 1, `h100-r4/` box 2 run 2;
not in git).

## What the runtime did on a call (measured)

A Mapal program's timed region is one call: the operands are generated, the clock fence syncs, then the site's first
launch. For a kernel with tensor maps that first launch did four things the S53 kernel's does not:

| step (old runtime, `launch_on`) | cost | where measured |
|---|---|---|
| the recipe: `cuModuleGetGlobal` + a synchronous pageable `cuMemcpyDtoH` of `<kernel>_tmap` | 23–30 µs (RTX 5080), 7 + 41 µs (H100, CUPTI) | `micro.c`, ktrace `old` arm |
| the map table: `calloc` of 256 entries (330–770 KiB of zeroed pages) at the first recipe | 28–35 µs, up to 260 µs when the init ran on the program's thread (5080) | instrumented runtime |
| `cuTensorMapEncodeTiled` × 2 (the encoder's first call ~2 µs more than a later one) | 3–8 µs | `micro.c`, ktrace `cold` |
| **the context's first launch of any kernel holding `cp.async.bulk.tensor`** (host side of `cuLaunchKernel`) | **1.34–1.92 ms** (5080, driver 620.73), 87–197 µs (H100, driver 615.71), ~7 µs (H100, driver 580.17) | `tma_launch.c` cold vs self |

The descriptor encode itself was already cached per pointer (0 encodes a call in steady state). Not the cause: lazy
module loading (`CUDA_MODULE_LOADING=EAGER` changes nothing), JIT (an offline cubin pays the same), the map's
contents, a new allocation (a new 1 GiB buffer's first TMA launch: 8 µs), the `__grid_constant__` parameter (a
kernel taking it without the TMA instruction: 3.9 µs). One launch of *any* TMA kernel in the context — its own
module, another thread — readies the path for all of them (5080: 1.4 ms → 3.3 µs).

## The changes

- **Recipes from the module text** (`mapal_gpu.c` `tmap_scan` / `tmap_parse`): the `_tmap` initializer is read from
  the PTX at load (NVPTX spells `[n x i64]` as bytes and drops trailing zeros; 80 of 80 bundle recipes equal the IR's
  words). No device call on the launch path; a recipe the text does not spell that way falls back to the device read.
- **The map table at load** (`tmap_table`): one entry a listed recipe, allocated and touched with the module.
- **Map cache by address** (`TMAP_SLOTS` 4 a map): a buffer whose address is fixed (arena, weights, cached
  allocations) is encoded once for the program's life, a loop alternating two buffers re-encodes nothing
  (`tma_launch.c`: 4 alternating launches, 2 encodes total).
- **`tma-warm`** (host rung, `GPU_INIT_TMA_COLD` = its OFF arm): a module whose text carries a recipe has the init
  thread (`eager-init`) load the runtime's one-box TMA kernel and launch it once before the clock (one 32 × 8-word
  box, 128-byte swizzle, onto an `mbarrier`). The fact it reads is the module's (recipes present), no part property;
  the flag word of a module without recipes is unchanged.
- **`ws-persist`** (kernel rung, rule `rules::ws_persist_pays`, shipped **off**): the warp-specialized kernel on
  `sms × blocks_per_sm` CTAs, each walking tiles `cta, cta + ctas, …` in the launch's raster order on one stage ring —
  ring step `t·nk + kk`, barrier phases running on across tiles, per-tile seeds, the tile's last stage released before
  its epilogue so the producer copies the next tile's boxes under it; the walk's CTA and tile count read after each
  warpgroup's `setmaxnreg` (read once at entry they were spilled), its divisions by constants in 32 bits. The price:
  `(⌈tiles / ctas⌉ − 1) · ((stages − 1) · tma_copy_ns + tile_bytes / DRAM share)`. Measured mixed (below): the rules
  ship a CTA a tile, `--gpu-on=ws-persist` measures the walk.

## Intent / realized

| item | intent | realized |
|---|---|---|
| launch path (new runtime) | no `cuModuleGetGlobal` / `cuMemcpyDtoH` / encode inside the clock; the warm on the init thread | CUPTI clocked region (`ktrace`, KTRACE_API=1): `on` / `one`: gen kernel → sync → tile kernel → `cuLaunchKernel` 6–8 µs; `old`: `cuModuleGetGlobal` 7 µs + `cuMemcpyDtoH` 41 µs before it; `mapal_tma_warm` 1×1×1 × 32, 1.3 µs, at init; trace totals `recipes=` = the module's recipe count |
| `ws-persist` kernel (sm_90a, ptxas) | grid = 132 CTAs, hot loop unchanged, no spill | 168 registers at launch, 0 spill on all 365 kernels of the bundle (8 bytes before the walk state moved after `setmaxnreg`); `USETMAXREG` 2, `UTMALDG` 2 and `HGMMA` 4 a trip; SASS +40 instructions a kernel (IMAD index math: 960 vs 920 on 2048 × 4096 × 14336). Review — the boxes kept no `evidence.tsv` (the bundle ran no `evidence.py static`, its script deleted the cubins); rebuilt from the deterministic emission (`hopper_launch_bundle.sh` at HEAD: boxes 3 / 4 launched 12 / 12 kernels of the rebuild by name; box 1's trace carries no names): 292 kernels, 0 spill bytes; the 20 `wgmma-ws` kernels an arm 168 registers (17) / 255 (3, 16-HGMMA tiles), `USETMAXREG` 2, `UTMALDG` 2 each; the walk (`--gpu-on=ws-persist`) on 2048 × 4096 × 14336 672 SASS instructions against 632 (+40; the 960 / 920 counted another span). `evidence.py check` per box (`evidence-h100.tsv` in each raw box dir): 1 526 checks, the step's `wgmma-ws` / `wgmma-hopper` / `ws-persist` rows 0 mismatch; 88 mismatches outside it — `min-blocks:off` 60 (`.minnctapersm` 1 on wgmma kernels the checker reads as off: the rung is not in their `rungs_on`), `flow-host:off` 8, `arch-rungs:on` 20 on the `.off` arm (named `.off`, not `.off-wgmma-ws`: the checker cannot see its rung) — open, the checker's reading of this bundle's arm names. The bundle and box scripts now write and check the evidence on the box. |
| values | every arm of the warp-specialized kernel prints the shipped arm's bits; OFF within 1e-2 | value gate 0 failures, both boxes |

## Timed: H100 SXM (box 2: driver 580.173.02, 700 W, Broadwell host; ABBA 8 reps, median `iter ms`)

OFF = `--gpu-off=wgmma-ws` (S53 kernel); old = the 5fcb401b build (this code's device text, base runtime); ON = the
shipped build (new runtime, `tma-warm`, a CTA a tile); walk = `--gpu-on=ws-persist`; rast = ON with
`--gpu-off=l2-gate`. Fastest competitor: gemm_lib's TF32 legs on the same box (`-fresh`: operands regenerated, Mapal's
boundary); the MLP / LLM rows' torch-compile legs are the 5fcb401b matrix's (other box).

| workload | OFF | old | ON | ON/old | ON/OFF | walk | rast | fastest competitor (leg) | explanation |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| pj_m128_n4096_k4096_tf32 | 0.0876 | 0.2191 | 0.0941 | **2.33** | 0.93 | 0.0896 | **0.0822** | 0.0435 (cublas-tf32-fresh) | old: 155 µs outside the kernel; ON loses to OFF by the b re-read (raster: kernel 0.0645 → 0.0535) |
| pj_m128_n6144_k4096_tf32 | 0.1545 | 0.2807 | 0.1159 | **2.42** | **1.33** | 0.1119 | 0.1096 | 0.0519 (lt-tf32-fresh) | kernel 0.094 vs 0.140 |
| pj_m128_n28672_k4096_tf32 | 0.2526 | 0.3937 | 0.2206 | **1.78** | **1.15** | 0.2251 | = | 0.1850 (cublas-tf32) | |
| pj_m128_n4096_k14336_tf32 | 0.2420 | 0.3897 | 0.2300 | **1.69** | **1.05** | 0.2257 | **0.2087** | 0.1051 (lt-tf32) | raster as above |
| pj_m2048_n4096_k4096_tf32 | 0.3129 | 0.4791 | 0.3082 | **1.55** | 1.02 | 0.3090 | 0.3085 | 0.1740 (cublas-tf32-fresh) | |
| pj_m2048_n6144_k4096_tf32 | 0.4576 | 0.6037 | 0.4428 | **1.36** | 1.03 | 0.4550 | 0.4393 | 0.2570 (lt-tf32-fresh) | |
| pj_m2048_n28672_k4096_tf32 | 2.0380 | 2.0680 | 1.8876 | **1.10** | **1.08** | 1.8967 | 1.8891 | 1.2815 (lt-tf32) | kernel 1.855 vs 2.019 |
| pj_m2048_n4096_k14336_tf32 | 0.9993 | 1.1196 | 0.9655 | **1.16** | 1.04 | 0.9949 | 0.9635 | 0.5835 (lt-tf32-fresh) | |
| mlp_2048_tf32 | 0.2729 | 0.3832 | 0.2080 | **1.84** | **1.31** | 0.2029 | 0.2021 | 0.1107 (torch-compile-cg-tf32, matrix) | |
| mlp2_2048_tf32 | 0.5321 | 0.5589 | 0.3876 | **1.44** | **1.37** | 0.3886 | 0.3911 | 0.2145 (torch-compile-cg-tf32, matrix) | |
| llm_pre_t2048_tf32 | 3.0995 | 3.1803 | 3.0500 | 1.04 | 1.02 | 3.0710 | 3.0360 | 1.372 (triton-tf32, matrix) | attention on radix tiles (open P4b) |
| llm_blkpre_t2048_tf32 | 6.5677 | 6.6772 | 6.4080 | 1.04 | 1.02 | 6.4811 | 6.3678 | 3.47 (triton-tf32, matrix) | as above |

Null (`|null / ON − 1|`, the shipped arm's second copy): 0.0–1.8 %. `wgmma-ws` against OFF: **11 of 12 win**
(1.02–1.37×) where 5fcb401b measured 0 W / 3 null / 9 L; the one loss is the 128-row 4096 × 4096 projection's `b`
re-read (rast wins it, 0.0822 vs 0.0876). Box 2 run 1 (10 reps, the recipe parser then failing to its device-read
fallback) agrees: old / ON 1.01–1.92×. `tma-warm` alone (cold = `--gpu-off=tma-warm`): cold / ON 1.004–1.07
(pj_m128_n4096 0.0976 / 0.0941, n6144 0.1235 / 0.1159); first clocked TMA launch (`tma_launch.c`) cold / ON
19.5 / 13.0 µs on driver 580.17, 182 / 95 µs on box 1 (driver 615.71, the 5fcb401b driver; that run had the map
table still made at the first launch and the parser falling back to the device read: the 80–100 µs left there are those two, both moved to load since).

**The walk (`ws-persist`), mixed**: walk / ON (a CTA a tile), two boxes: the one-consumer tiles (64 columns) win
(pj_m128_n4096 0.952, n6144 0.965, k14336 0.981; box 2 run 1 0.978 / 0.985 / 0.930) and the 128 × 256 two-consumer
tiles lose (pj_m2048_n4096_k14336 1.030 / 1.035, n6144 1.028, pj_m128_n28672 1.020, blkpre 1.011 / 1.018) beyond the
null; the hot loop's SASS is the same, the kernels' traced times differ the same way (0.968 vs 0.953 ms). Not explained
by the instruction mix — the candidate fact is the walk's static round robin against the hardware placing a later
block on whichever SM frees first, at the tile's own time (no SM-level timeline here: no `ncu` on the boxes). Off by
the rules; the replay line is `mixed`.

## The library's Hopper kernels (CUPTI on the box; gemm_lib's TF32 legs)

| site | cuBLAS kernel (most launched) | grid | block, regs, smem |
|---|---|---|---|
| 2048 × {4096, 6144, 28672} × 4096, 2048 × 4096 × 14336 | `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize128x256x32_warpgroupsize2x1x1_execute_segment_k_off` | 66 × 2 = 132 (persistent) | 384, 168, 226 KiB |
| 128 × 4096 × 4096 | `…tilesize64x128x32_warpgroupsize1x1x1…` | 8 × 8 = 64 / 16 × 4 | 384, 168, 226 KiB |
| 128 × 6144 × 4096, 128 × 4096 × 14336 | `…tilesize64x128x32…` / `…tilesize128x128x32…` | 24 × 4 / 48 × 2, 13 × 5 / 64 × 2 | 384, 168, 226 KiB |
| 128 × 28672 × 4096 | `…tilesize64x128x32…` | 7 × 16 = 112 | 384, 168, 226 KiB |

The library kernels run 1.45–2.0× faster than ours (kernel against kernel, same box): the gap is the TF32 kernel body
(our producer rounds `aᵀ` in place and the consumers round `b` fragments from registers; the library's operands are
smem descriptors), not the launch path any more.

## Tests

- `tests/hopper_tma.rs` `generated_ws_kernels_equal_the_s53_kernel` (36 generated TF32 GEMMs at 1 / 2 / 8 SMs, four
  emulator modes): the walk (`--gpu-on=ws-persist`) equals the shipped kernel byte for byte; coverage (the same
  with the walk shipped and as the ON arm): 36 programs, 18 warp-specialized, **9 walks, 1 walk of ≥ 3 tiles a CTA, 2 whose ring turns over
  inside a tile's steps**, 9 byte-identical to S53, 8 two-consumer, 3 wrapping, 2 S53 fallbacks, 16 other kernels.
- `protocol_mutations_fail` mutation 3: a tile's last stage never released fails (exit 108, a phase that never
  completes) on the 1-SM walk; the walk's own bits equal the CTA-a-tile kernel's in every mode.
- `rules.rs` `ws_persist_prices_the_walk` (hand-worked H100: 477 ns fill, 5 161 ns epilogue, 13 rounds at 1792 tiles,
  one round at 132, a part with twice the SMs); `rung_rules_replay` with the 5 new lines (143 picked, 72 null,
  11 mixed, 22 open); `func::gpu` `warm_zero_flag_follows_the_modules_zeros` (the TMA-cold bit); the raster test reads
  the tile grid's height under a walk.
- Full `mapal-backend-llvm` suite (`test_par.py`, walk on by default at that point): 49 binaries, 482 tests, 0 failed.
- `benches/local/tma_launch.c` on the RTX 5080: cold first launch 1.36–1.92 ms, ON 9.8–14.6 µs, self 7–9 µs; 0 wrong
  words; 2 encodes for 5 launches over 2 buffers.

## Rentals

| box | GPU / driver / host | $/h | minutes | cost | outcome |
|---|---|---:|---:|---:|---|
| 62338875 (Idaho) | H100 SXM / 615.71.09 / EPYC 9V74 | 2.27 | 3.6 | $0.14 | probes; programs not linked (the uploader skips a non-part object dir: fixed) |
| 62343243 (AU) | H100 SXM / 580.173.02 / Broadwell | 4.05 | 17.3 | $1.17 | run 1 (recipe parse fell back: trailing zeros, fixed) |
| 62345520 (AU) | same machine | 4.67 | 15.8 | $1.23 | run 2 (the table above) |

Total **$2.54** of the $7 cap; every instance destroyed (verified: no `mapal-hl-*` instance listed).

## Open

1. `ws-persist` mixed (above): the separating fact (static walk vs dynamic block placement against the tile's time;
   needs an SM-level timeline) — attempted, open.
2. `l2-gate` for one-CTA-an-SM tiles: the gate's "one block's panels fit the L2" misses a wave reading every `b` panel
   (128-row projections: raster 5–12 % faster, null at 2048 rows) — the fact is the wave's own panel bytes against the
   L2; changing it moves every tile kernel on every part, so it is recorded (replay `l2-gate h100 off`, open) not
   switched.
3. The TF32 kernel body against cuBLAS's (1.45–2.0×): operand rounding passes (open; the 16-bit body is another
   step's).

## For integration

- `benches/results/evidence/local-rtx-5080/README.md`: a row "hopper-launch — the TMA kernel's first-launch path
  (recipes from the module text, the map table at load, map cache, `tma-warm`) and the persistent walk (`ws-persist`,
  ON arm): H100 `wgmma-ws` vs OFF 11 / 12 wins (was 0 W / 3 null / 9 L), ON vs the 5fcb401b build 1.04–2.42× →
  [hopper-launch.md](hopper-launch.md)".
- `CLAUDE.md` facts table, backend row (`backend: Gpu properties …, rules::rungs`): add `rules::ws_persist_pays`
  (`ws_persist_saves_ns`: the walk's saved fills and epilogues from `tma_copy_ns`, the DRAM share, `sms`). No new
  mapal-ir fact.
- `matrix.*`: the H100 TF32 rows (`wgmma-ws` ON) per the table above (box 2, driver 580.17; the 5fcb401b column was
  driver 615.71).
- `docs/tailored-fixes.md`: none (no tailored fix).
- `benches/results/evidence/cross-arch/README.md` open item 3 ("the TMA kernel's steady-state launch cost"): closed by
  this step; open 5 (`l2-gate`'s wave footprint) and the walk's separating fact remain.
