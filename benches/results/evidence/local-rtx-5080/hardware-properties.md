# hardware-properties — every GPU rung decided by physical properties (2026-11-21)

The rule (CLAUDE.md, Sapir 2026-11-20): hardware is physical properties, never a name or a version. Before this step
the shipped GPU rungs were `GPU_RUNGS_OFF` — 131 `RungOff` rows (rung × part × fill × face, each citing a measurement)
plus an architecture alias (`cuda-ada` took the 4090's rows, `h100` the H100 PCIe's, the rtx-5080 the 5090's). The
table, `RungOff`, `GPU_MEASURED` and the lookup are deleted. What ships now is `rules::rungs(gpu, wps, face)`
(`crates/backends/llvm/src/rules.rs`): each rung from the site's fill (warps per scheduler of its grid on the part's
SMs), its face, and the part's properties. Every cited measurement is a line of `rung_replay.tsv` (the former rows'
cites, narrowed to the class each measured, plus the "kept" clauses and the 5080's own), and the test
`rung_rules_replay` replays the rules against all of them.

Raw: `.claude/raw-evidence/local-rtx-5080/properties/` (suite runs, ABBA, calib, bundles' plan lines).

## What changed

| item | where |
| --- | --- |
| `GPU_RUNGS_OFF` / `RungOff` / `GPU_MEASURED` / `Fill` / `Face::Any` and the architecture lookup removed | `profile.rs` |
| rung rules over properties; thresholds as named constants, each between the nearest measured winner and loser | `rules.rs` |
| properties on every GPU profile: DRAM interface (`mem_clock_khz`, `mem_bus_bits` → peak), FP32 lanes and clock on every part, `schedulers`; derived: FFMA per DRAM byte, registers per resident thread, L2 per SM, DRAM per SM | `profile.rs` `Gpu` |
| capabilities as properties (`Caps`: `cp.async`, warp `mma` 16×8×8 / 32 threads, `ldmatrix`, warpgroup `wgmma` 64×256×8 / 128 threads, `griddepcontrol`, `red.f64`); `Gpu::sm()` and every version compare deleted — `caps_of_cc` is the PTX ISA table, applied once when a profile is built from a device query | `profile.rs`, `func/gpu.rs` |
| `DeviceCost::pageable_pm` — the host's pageable / page-locked copy rate; `calib.cu copy` measures it (5080 box: 695, three runs) | `profile.rs`, `benches/gpu/calib.cu` |
| `native-gpu`: device 0's `cuDeviceGetAttribute` facts (`libcuda.so.1` opened at run time) + the calibration file (`MAPAL_GPU_CALIB` or `~/.cache/mapal/gpu-calib`, keyed by the query's physical facts); this box's line: `calib/rtx-5080.calib`; on this box it equals the `rtx-5080` spelling field for field | `profile.rs` `DeviceQuery` |
| `--cost=` names extended to the physical properties (`sms`, `l2_bytes`, `smem_per_sm`, `max_threads_per_sm`, `regs_per_sm`, `max_blocks_per_sm`, `mem_clock_khz`, `mem_bus_bits`): a cross-compilation or a property sweep reshapes a part by name-free values | `profile.rs` |
| `wave_full` (a row-fitted grid's blocks over the SM slots its waves span ≥ 3/4) replaces "fewer blocks than SMs" as `wave-gate`'s geometry | `profile.rs` |
| `fold-lanes` never over a column-major operand (a layout fact, not the 4090's row) | `func/gpu.rs` `rowfold_lanes` |
| `pdl`: every kernel of the module waits (`griddepcontrol.wait`) — the runtime launches any kernel behind a kernel as a dependent launch; only the tile kernel waited before (found when the rule turned `pdl` on: 5 programs failed their value gates; 4042 / 4042 kernel entries wait now) | `func/gpu.rs` `pdl_wait_every_kernel` |
| `emit --rungs --target=<p>`: the rules' table for a part; the bundle writes `<part>/rungs.tsv` and `evidence.py` checks `plan.rungs_on` against it (it parsed `GPU_RUNGS_OFF` before) | `examples/emit.rs`, `evidence.py`, `changed.py`, `rungs.py` (prints replay lines); `rejudge.py` removed with the rows |

## Properties per part (the inputs the rules read)

| part | SMs | L2 / SM | regs / resident thread | smem / SM | FP32 lanes | clock | DRAM (peak) | FFMA / DRAM byte | DRAM / SM | `griddepcontrol` | pageable / pinned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rtx-4090 | 128 | 576 KiB | 42 | 100 KiB | 128 | 2.52 GHz | 1008 GB/s | 81.9 | 7.9 KB/µs | no | — |
| rtx-5090 | 170 | 578 KiB | 42 | 100 KiB | 128 | 2.41 | 1792 | 58.5 | 10.5 | yes | — |
| h100-pcie | 114 | 449 KiB | 32 | 228 KiB | 128 | 1.755 | 2039 | 25.1 | 17.9 | yes | — |
| rtx-3090 | 82 | 75 KiB | 42 | 100 KiB | 128 | 1.695 | 936 | 38.0 | 11.4 | no | — |
| a100 | 108 | 379 KiB | 32 | 164 KiB | 64 | 1.41 | 1555 | 12.5 | 14.4 | no | — |
| rtx-5080 | 84 | 780 KiB | 42 | 100 KiB | 128 | 2.45 | 960 | 54.9 | 11.4 | yes (PDL 486 vs plain 650 ns) | 0.695 |

Device-query facts (SMs, L2, smem, threads, registers, blocks, DRAM interface) from the boxes' `facts` lines and
datasheets; clocks under load where measured (E1), else the query's boost clock; the rented parts' pageable ratio is
not a property of the part but of the box (the replay lines carry the ratio each session measured).

## Rung → property → rule → replay

Replay counts per rung: lines whose winner the rules pick / null (inside the null or an unstable sign: no winner to
pick) / mixed (the part's own cells disagree) / open (a measured winner the rules do not pick).

| rung | property the rule reads | ships ON where | the mechanism the measurements show | replay |
| --- | --- | --- | --- | --- |
| `async` (`cp.async` FMA panels) | FFMA per DRAM byte | under-filled and ≥ 70 flop/B (`BALANCE_PM`) | under-filled the tile is latency-bound; a compute-rich part (4090, 81.9) needs the issue slots and registers a register-staged copy spends; filled, register staging won on every part | 7 / 1 / 0 / 0 |
| `kmajor-a` | (the `async` rule) | where the `cp.async` panel is | lost on every register-staged copy, 21 W / 0 L on the async path | 5 / 0 / 0 / 0 |
| `wide-fma` (8×16 tile) | FFMA per DRAM byte | ≥ 70 flop/B | more FFMA per loaded byte for less occupancy: won on the 4090, lost at 38 / 25 / 12.5, mixed on the 5090 (58.5) | 4 / 0 / 1 / 0 |
| `warp-fill` | fill; FFMA per DRAM byte | exact face; 0 warps a scheduler; 1 warp on ≥ 45 flop/B (`FILL_BALANCE_PM`) | the shrunk tile hides latency where schedulers are empty; at one warp it won on the 5080 (54.9) and lost on the 3090 / H100 (38 / 25); filled contract lost | 9 / 2 / 0 / 1 |
| `min-blocks` | registers per resident thread; smem per SM | filled: > 32 regs; under-filled: exact, or contract unless (≤ 32 and ≥ 200 KiB); not TF32 under-filled | `.minnctapersm` from the derived residency caps registers: spills where a full SM leaves 32 a thread (H100, A100) and smem admits many blocks (H100) | 8 / 2 / 0 / 0 |
| `fill-rows` | registers per resident thread | not (exact, under-filled, ≤ 32) | the scalar-row tile lost to the cell tile on the H100 | 1 / 0 / 0 / 0 |
| `grid-share`, `panel-regs` | L2 bytes per SM | ≥ 128 KiB (`L2_PER_SM`) | a deeper panel holds more bytes in flight per block; the 3090's 75 KiB a SM lost both | 2 / 6 / 0 / 0 |
| `splitk` (workspace) | L2 bytes per SM | not (exact face and < 128 KiB) | the exact face's partial round trip on 75 KiB a SM (3090); its contract face won | 3 / 0 / 0 / 0 |
| `row-split` | DRAM bytes per µs per SM | ≥ 14 600 (`DRAM_PER_SM`) | filling more SMs pays where each adds DRAM rate: H100 17 885 won, 3090 / 5090 / 4090 (≤ 11 414) lost | 4 / 0 / 0 / 0 |
| `unroll` | FP32 lanes per SM | not (exact face and < 128) | the A100's 64-lane SM lost the fully unrolled exact panel | 1 / 0 / 0 / 0 |
| `pdl` | `griddepcontrol`; measured programmatic vs plain dependent launch | the part has it and `pdl_launch_ns < dep_launch_ns` | the dependent kernel's ramp overlaps the producer's drain: 5080 486 vs 650 ns, loops 8–9 % faster; H100 / 5090 measured no launch costs | 1 / 2 / 1 / 0 |
| `pinned` | host pageable / page-locked copy rate | measured and < 0.9 (`PAGEABLE_PM`) | page-locking pays only where the pageable path is slower: 0.36–0.70 won, 0.98–1.0 lost or null | 3 / 4 / 0 / 0 |
| `row-fit` + `wave-gate` | SMs: the fitted grid's wave fill (`wave_full`) | row-fit everywhere; the gate under-filled, refusing grids that fill < 3/4 of their waves' SM slots | under-filled, sparse fitted grids lost to the cell tile; filled, the fitted grid + its split won (5090 0.46 vs 2.07 ms, 5080 16×2048×7168 1.33–1.66×) | 4 / 2 / 0 / 3 |
| `affine` | fill, face | exact under-filled; contract filled | every part agrees per class | 5 / 4 / 0 / 0 |
| `idx32` | fill, face | contract filled | every part agrees per class (mostly null) | 3 / 7 / 0 / 0 |
| `pad` | fill, face | not (exact and filled) | H100 exact: filled lost, under-filled won | 2 / 1 / 0 / 0 |
| `wide` (TF32 warp tile) | fill | filled | under-filled lost on 5090 / 3090 / A100; filled won on the 3090 | 4 / 1 / 0 / 0 |
| `streamk` | fill | filled | its forgone filled cells won (H100, 3090); under-filled lost on the 5090 and the 5080 (skinny projections) — site-level wave quantization open | 2 / 3 / 0 / 4 |
| `fold-lanes` | the operand's layout | never over a column-major operand | one thread a row already coalesces there (4090) | 1 / 0 / 0 / 0 |
| `interleave`, `ldmatrix`, `raster`, `splitk-atomic`, `warp` | — | every part | null everywhere measured, or a win (`warp` H100 filled, `ldmatrix` A100 filled) | 3 / 12 / 0 / 0 |
| `linear` | — | no part | null on the 5090, lost on the 5080 (indep_512 18 %) | 1 / 1 / 0 / 0 |
| `async-h2d`, `asyncstage`, `bank-plan`, `fit-first`, `l2-window`, `plan-cost`, `retain-delta`, `rowfold-layout`, `streams`, `tf32-radix` | — | no part | no measured part found a stable gain; the mixed ones are per-site candidates (`CHOICE_AXES`) | 22 / 19 / 2 / 5 |

**Replay: 179 lines — 95 picked, 67 null, 4 mixed, 13 open.** The open ones, each with the property that is missing:

| rung | part | measured winner | missing fact |
| --- | --- | --- | --- |
| `asyncstage` | h100-pcie, rtx-5090 | on with L2 flushed (warm L2 and the composed MLP lose) | whether the launch's operands are in L2 (`residence_plan` at the launch) |
| `fit-first` | rtx-4090 | on at the 64-row convolution (its 16-row projection lost) | site-level: the reshape against the halving per site geometry |
| `row-fit` | h100-pcie, rtx-4090 | off at 144 fitted blocks on 114 / 128 SMs (filled) | the fitted grid's wave fill counted with the split chosen after it (`wave_full` × S) |
| `rowfold-layout` | a100 (exact), rtx-4090 (contract) | on, 1–2 % of device time | the fold's strided-read cost against the tile's column store |
| `streamk` | rtx-3090 ×2, rtx-4090, rtx-5090 | site-dependent within one class | per-site wave quantization (tiles against `sms · blocks_per_sm`) |
| `warp-fill` | rtx-3090 | off on the free order at 0 warps a scheduler (the filled tile lowers the split count) | the order axis as a rule input |
| `wave-gate` | h100-pcie | off at 8 blocks, measured against one thread a cell before `cell-fill` existed | re-measure against the cell tile (the row-fit line measured the cell tile 4× faster there) |

## The rtx-5080: decisions against f3a9e792 (the 5090's rows by architecture)

| rung | f3a9e792 on the 5080 | rules on the 5080 | why |
| --- | --- | --- | --- |
| `pdl` | off (5090 row: null) | on | `griddepcontrol` and a measured programmatic launch cheaper than a plain one (486 vs 650 ns) |
| `pinned` | off | on | pageable / pinned 0.695 < 0.9 (`calib.cu copy`); inert in the suite (no host array crosses in a loop) |
| `wide` (TF32 warp tile) | off filled | on filled | under-filled lost on 5090 / 3090 / A100, filled won on the 3090 |
| `min-blocks` | on | off on TF32 under-filled | the 3090's TF32 1024 lost 11 % three times |
| `pad` | off on the exact face | off exact filled only | H100 exact under-filled won (the default face is contract: no suite change) |
| `async` | on exact filled | off filled | lost filled on every part (exact face only) |
| `wave-gate` | on under-filled (blocks < SMs) | on under-filled (`wave_full`) | the gate's geometry |
| `row-fit` | off exact filled | on | the gate's fact decides per grid |
| `warp-fill`, `streamk`, `linear`, `fit-first`, `grid-share`, ... | — | unchanged on the 5080 | the 5080 sits on the 5090's side of every other threshold |

The 5080 also measured three rules first written without it, and the replay now holds its lines: `warp-fill` at one
warp a scheduler (its shrink won loopmm / smlp 1.15–1.24×; the H100 and 3090 lost it: FFMA per DRAM byte separates
them, `FILL_BALANCE_PM`), `wave-gate` at a filled site (the fitted 16-block grid plus its split beat the cell tile
1.33–1.66×: the gate gates under-filled sites only), `linear` (one launch lost to the two site tables, 18 %).

### Time on the rtx-5080 (ms; suite ON / OFF / ON `LADDER=1 gpu_run.sh`, tuned picks; ABBA: 8 rounds = 16 processes an arm, rules' picks, same box)

OFF = HEAD before this step (11b124a2: the f3a9e792 table's decisions, the 5090's rows). Value gates: 0 programs failed
in each of the three runs (the first ON bundle, before every kernel waited under PDL, failed 5: attn, causal, decode,
diamond, softmax_rows — the race the wait fixed).

| workload | OFF | ON (two runs) | ON/OFF | ABBA (OFF / ON, no-pdl / ON) | best other | what moved it |
| --- | --- | --- | --- | --- | --- | --- |
| decode_1024_256 | 9.587 | 8.828 / 8.897 | 0.921 / 0.928 | 0.920, 0.921 | 15.48 (cache) | `pdl`: dependent launches overlap the producer's drain (the trip is ~12 launches) |
| causal_1024_256 | 16.75 | 15.34 / 15.15 | 0.916 / 0.904 | 0.908, 0.906 | no leg | `pdl` |
| gen_1024_512_256 | 8.517 | 7.756 / 7.706 | 0.911 / 0.905 | 0.908, 0.919 | no leg | `pdl` |
| prefill_1024_512_256 | 8.916 | 8.149 / 8.106 | 0.914 / 0.909 | 0.910, 0.910 | no leg | `pdl` |
| attn_1024_64 | 0.04132 | 0.03951 / 0.04141 | 0.956 / 1.002 | — | 0.0348 (fused) | `pdl` (one run beyond null) |
| pj_m16_n2048_k7168 | 0.05091 | 0.0482 / 0.04871 | 0.947 / 0.957 | 0.935 (spread 13 %) | no leg | inside the OFF sessions' 4.9 % spread |
| mlp_512 | 0.03993 | 0.04242 / 0.04351 | 1.062 / 1.090 | 1.003, 1.026 | 0.041 (lt) | `pdl` on a short chain (open: mixed) |
| fir_4194304_16 | 0.04601 | 0.04945 / 0.04875 | 1.075 / 1.060 | 1.019, 1.022 | no leg | `pdl` ~2 % (OFF sessions 4.4 % apart) |
| convmc_8_64_56_3 | 0.08363 | 0.08507 / 0.08443 | 1.017 / 1.010 | 1.013, 1.015 | no leg | `pdl` ~1.5 % |
| indep_2048 | 1.002 | 1.018 / 1.020 | 1.016 / 1.018 | — | 0.9196 (lt) | `pdl` (no tile plan moved) |
| loopmm_1024_8, smlp_4096_256_1024 | 0.6503, 0.1672 | 0.6506, 0.168 | 1.000, 1.005 | 1.003, 1.003 | —, 0.1536 (lt) | `warp-fill` at one warp kept (it lost 1.24 / 1.15 when the first rule dropped it) |
| the other 25 rows | — | — | 0.975–1.012 | — | — | unchanged tile plans; the `pdl` wait in every kernel |

Intended vs realized (bundle, `ptxas` / PTX / host text): `griddepcontrol.wait` in 4042 / 4042 kernel entries (0 before),
`mapal_gpu_init` flags 13 (PDL bit) in 43 / 43 programs; realized: decode's traced run launches 4098 kernels, 4097 of
them programmatic (`pdl=` in the trace totals; 0 in every OFF trace). `pinned`: no `mapal_gpu_pin` in any suite program (no host array crosses the bus inside a loop here). Tile
plans moved in 11 programs (the TF32 warp tile filled, the gate's geometry, `min-blocks` on TF32 under-filled); none of
them moves time beyond its null except through `pdl`. Evidence checks: 12081 (330 mismatch) ON against 12064 (433) OFF —
the intent checks now read the rules' table (`rungs.tsv`); the mismatches are the pre-existing untriaged set.

**Decision.** The rules ship. `pdl` is net positive on the suite (four loop programs 8–9 % faster; three short programs
1.5–2.5 % slower in ABBA, mlp_512 6–9 % in the suite) and recorded **mixed** in the replay: the separating fact is open
(hypothesis: `griddepcontrol.launch_dependents` lets a dependent grid's blocks take SM slots while a producer that
fills the part still runs — the trigger should read the producer's wave fill).

## Tests

| test | result |
| --- | --- |
| `rules::tests::rung_rules_replay` | 179 lines (`rung_replay.tsv`): 95 picked, 67 null, 4 mixed, 13 open — the open set asserted exactly |
| `rules::tests::rungs_read_properties_never_names` | every part renamed and re-targeted (`sm_999`) takes the same rungs at every fill and face |
| `rules::native_tests` (2) | the 5080's query + `calib/rtx-5080.calib` = the `rtx-5080` spelling field for field and rung for rung; the live driver query on this box equals it; another part's facts take no line (no measured costs) |
| `tests/hardware_properties.rs` | 36 generated programs (`geom_prog_strategy`) on parts drawn from properties (SMs, L2, FP32 lanes, DRAM bus, threads per SM, clock) over six bases: both sides of each threshold reached (FFMA/B 23 / 13, regs 15 / 21, L2 24 / 12, DRAM/SM 25 / 11), 36 emissions moved by the rules, values = the every-rung arm in 144 emulator runs (free, `MAPAL_EMU_SERIAL=1`, `=2`, `MAPAL_EMU_REVERSE=1`) |
| expectation updates | 10 tests pinned per-part rows (streams forced on where they test the mechanism, `linear` / `rowfold-layout` / stream-K forced where they test those, the chooser's picks, the M1 gate geometry); 9 insta snapshots: the 4090 under-filled exact launch's shared memory (`grid-share` on: 1664 → 13312, 3328 → 28672 B) |
| `site_choice` golden | `c0-rules.tsv` (was `c0-089f1a6.tsv`): 138 of 208 candidate-0 hashes moved (4090 30, H100 41, 5090 19, 3090 48), rewritten by `MAPAL_REGOLD=1` |
| workspace | 541 tests, 0 failing after the updates (the CUDA `differential_testgen_closed_sweep` passed here) |

## Open

1. The 13 open replay lines above, each a missing fact; and `pdl`'s mixed sign on the 5080 (loops gain 8–9 %, short
   programs lose 1.5–2.5 %): the dependents' early launch against the producer's wave fill.
2. Thresholds sit between one measured winner and one loser on five or six parts; `FILL_BALANCE_PM` and
   `BALANCE_PM` read the same property with two thresholds (a part between 45 and 70 flop/B takes the warp-fill
   shrink at one warp but no `cp.async` panel). Each new part's measurements are replay lines that can move them.
3. `pinned` is measured on this host but no suite program page-locks; its win is the rented boxes' (`pt_loop_2048`).
4. The matrix capabilities are shapes on `Caps`, but the `mma` / `wgmma` emitters still use their shape constants
   (`MMA_M`, `WGMMA_M`, …); reading the shape from `Caps` is the next step for a part with other shapes (datacenter
   Blackwell `tcgen05`).
5. `fp32_lanes` is not a device attribute: `native-gpu` takes it from the calibration file (measured FFMA peak) or the
   SM-design table by compute capability.
