# local-rtx-5080 — long prefill and under-filled grids (2026-11-22)

Gap analysis items 1–2 of `../cross-arch/README.md` (prefill T8192 9.7–18.7× behind on every part; grids under the
SM count). Machine: Threadripper 9970X + RTX 5080 (sm_120, 84 SMs, 100 KiB smem / SM, 64 MiB L2, 16 GB), CUDA 13,
clang/LLVM 23, branch `local-rtx-5080`. **Driver note:** the NVIDIA userspace was upgraded to 620.73.04 during the
day while the loaded kernel module stayed 620.70.06 (no reboot): every CUDA process failed `cuInit` ("unsupported
display driver / cuda driver combination"). Every GPU process of this step (Mapal, torch, the CUDA backend's tests)
ran with the matching 620.70.06 userspace (libcuda, ptxjitcompiler, nvml, nvidia-smi) extracted from the pacman
cache into the scratch dir and put first on `LD_LIBRARY_PATH` / `PATH` (`env.sh` in the raw dir). A reboot ends it.

## 1. The exact fact that refused the 8192-token prefill

The 8192-token program loops over the 8 KV-head groups (its 32-head score arrays would be 2³¹ cells, 8 GiB each);
inside, QKᵀ reads `q[i·4096 + (g·4 + hh)·128 + u]` and `kr[r·1024 + g·128 + u]`, P·V `qkv[r·6144 + 5120 + g·128 + u]`,
the O projection `wo[(g·512 + k)·4096 + j]`. Every index fits `i32` (2²⁸ cells a group) and `bounds_proof` proves
them in bounds (`g ∈ [0, 7]` where the loop body runs). What refused them was **`tile_plan_radix`'s read: a digit
of an address on any map capture but the cell index `t` had no form** (`radix.rs`, `e.slot != map_captures`) — the
loop counter `g`, a number captured by the map, made all three maps per-cell kernels. Replacing `g` by a literal
gave four tile sites (the 2048-token program's), so this was the one refusal; index width and digits were not.

| fact / rule | what it proves or decides | where | OFF arm |
|---|---|---|---|
| **`uniform-base`** (fact arm): a mixed-radix tile read's address may take terms of an integer the map captures, `coef · ((v ÷ stride) % radix)` (`RadixTile::ua` / `ub`), each in bounds at the top of the capture's range where the site runs (`BoundsProof::capture_range`, non-negative there) | `tile_plan_radix` (stored on the graph); `uniform_at` is the address law host and tests share | `mapal-ir/src/algo/radix.rs`, `bounds.rs` | `--gpu-off=uniform-base` (the fact arm: per-cell kernels, as before) |
| the launch moves the operand's pointer by the terms (bytes = Σ terms × element bytes); the kernel is the unmoved site's; a site whose terms step by less than a 16-byte vector is no device site (`uniform_aligned`); a moved site launches on its own (no `horizontal` / `calls` / row-block member) | the captured value at the launch | `func/resident.rs` `emit_site_resident`, `func/gpu.rs` `uniform_bytes`, `gpu_site` | same |
| **`carry-take`**: a loop carry whose init is host-built but whose device copy is valid and dies at the loop's entry (`LastUsePlan::death` ≤ the `LoopEnter`'s position) takes that buffer as its first state — the S65 move of a device-built init, widened (`; mapal.take`) | `last_use_plan` | `func/resident.rs` `resident_loop_enter` | `--gpu-off=carry-take` |
| **`slice-launches`** (item 2, long rows): a sliced unit's phases meet at kernel boundaries — one launch a phase, the kernel's `%phase` argument selecting it — where `phases × launch_at(grid)` undercuts one cooperative launch, its `phases − 1` grid barriers and (outside a loop) the process's first cooperative launch (`coop_first_ns`, 38 us here) | `DeviceCost::{dep,pdl,coop}_launch_ns`, `gsync_ns`, `coop_first_ns`, `row_iter_ns` | `func/flow.rs` `row_slices`, `sliced`, `emit_unit` | `--gpu-off=slice-launches` (cooperative only); `--gpu-on=slice-launches` with `--row-slices=<s>` forces it (tests) |

No workload-named code: the uniform terms are any integer capture's digits (a group counter, a panel index, a
sequence), checked against `bounds_proof`; the take reads the death point every value has; the launch choice is the
measured launch costs of whatever part the profile describes.

## 2. Intent → realized (PTX / SASS / ptxas -v, the shipped bundle; `evidence.py` with the new intents)

| kernel | intent | realized |
|---|---|---|
| T8192 QKᵀ (`d6210c7b`, `ubase=a,b`, head-group, guarded post) | the 2048-token program's group tile: 4 query heads a KV head's panel, `q` / `kr` moved by `g·512` / `g·128` floats, no spill | 64 × 256 blocks × 128 thr, 222 regs, 0 spill, 16.4 KB smem; 2700 FFMA / 96 LDS.128, 16 LDG.128; host: one `add i64` a pointer before the table launch |
| T8192 P·V (`5103b13a`, tuned pick; table of 7 with S = 4 workspace splits) | `qkv`'s V panel moved by `g·128`; the `/ den` epilogue in the store | 1 × 256 blocks × 256 thr, 127 regs, 0 spill, 33.8 KB, 20 LDGSTS (cp.async), 1356 FFMA / 64 LDS.128, 66 MUFU (the reciprocal) |
| T8192 O + residual (`b2ae8fe3`, tuned) | `wo`'s panel moved by `g·2 097 152`; `y[e] + yo` in the store | 32 × 64 × 256 thr, 123 regs, 0 spill, 1024 FFMA / 64 LDS.128 |
| T8192 `y` init | `x0`'s device copy dies at the loop's entry: no upload | `; mapal.take carry=0 bytes=134217728`; CUPTI: the 134 MB H2D (4.96 ms) is gone |
| lsm sliced units (`launches=1`) | two launches, no grid barrier, the phase by value | `ptx.gsync_arrive = 0`, `%phase` the last parameter, 37–38 regs, 0 spill, 2 KB smem; host: two `launchv` |

`evidence.py` over the bundle (14 programs, 38 stems): 9 105 checks, 606 mismatches — the part's standing items
(`pdl` 216, `fma-pairs` 83, `face-registers` 58, …); the new and touched intents 42 / 42 ok (`uniform-base`,
`carry-take`, `slice-launches`, `row-slices`, `exp-rescale`), `flow` 42 / 42 on the lsm stems once the intent counts a
phase launch's argument. One finding: a **T8192 P·V table candidate spills** (`4843c65c`, the S = 4 workspace split,
8 × 8 × 256, 44 B; not the rules' or the tuner's pick) — the same class the 2048-token program's radix split
candidates already showed (24 B).

## 3. Results (`model_scale_run.sh`, 30 reps; arms interleaved in the probe section, 5 processes an arm; ms)

| workload | OFF | ON | best competitor | ON / best | explanation (CUPTI, rules pick unless noted) |
|---|---:|---:|---|---:|---|
| prefill T8192 | 1064.8 (`uniform-base`, `carry-take` off) | **82.06** (rules 86.7, null 1.8 %) | 84.25 torch-compile | **0.97** (was 12.1) | per-cell QKᵀ 651.8 / P·V 238.1 / O 125.8 ms → tiles 20.2 / 24.9 / 8.1 ms (8 groups); softmax flow 24.9 (DRAM: `s` read + `pe` written, 1 GiB each a group); QKV 11.9; `y` upload 4.96 → 0 |
| prefill T2048 | — | 8.636 | 8.311 torch-compile | 1.04 | emission unchanged (normalized hash); null 18.9 % this run |
| block prefill T2048 | — | 27.45 | 27.74 torch-compile | 0.99 | unchanged |
| prefill_1024_512_256 | 8.077 (`carry-take` off) | 8.043 | 27.06 torch-compile | 0.30 | two 1 MiB uploads fewer; ON / OFF 0.996, null 0.1 %: below what one run separates |
| lsm 1 × 131072 | 0.0308 (`slice-launches` off) | **0.01525** | 0.0236 torch-compile | **0.65** (was 2.3) | the same 64 slices; cooperative 5.9 us + 17.6 us idle (the first cooperative launch's host setup in the region) → 2 × 3.2 us launches, span 26.2 → 11.0 us |
| lsm 4 × 32768 | 0.0614 | **0.01602** | 0.0184 torch-sdpa | **0.87** (was 3.6) | one block a row (4 blocks, 49.8 us) → 64 slices a row, 2 × 2.7 us (torch: `cunn_SoftMaxForward` 4 × 1024 threads, 8.3 us) |
| lsm 32 × 8192 | 0.01871 | 0.01841 (rules 0.01557) | 0.0102 torch-compile | 1.80 | one block a row (32 blocks, 9.0 us) → 8 slices, 2 × 3.06 us; ON / OFF 0.984 (null 0.6 %); the tuned pick of the q·W site is slower than its rules pick (open, tune) |
| softmax_rows 2048 / 4096 | — | 0.4872 / 3.818 | 0.471 / 3.562 torch-compile | 1.03 / 1.07 | unchanged: memory-bound flow units, 2048 / 4096 rows fill the part |
| attn 1024_64 / 2048_64 / 2048_128 | — | 0.0392 / 0.0797 / 0.1173 | 0.0287 / 0.0748 torch-compile, 0.1044 sdpa-compile | 1.37 / 1.07 / 1.12 | unchanged (see open 3) |
| pj m16 × 2048 / 18432 × 7168 | — | 0.0425 / 0.6107 | 0.0492 / 0.5898 cuBLAS | 0.86 / 1.04 | unchanged (large-gemm's fill rules) |

Values: 137 value gates pass (every arm and site candidate of the 14 programs; T8192 / T2048 / block prefill against
their exact face built for the device, `XREF`); the 8192-token prefill prints −1.3648784, −1.6895329, 0.65508735,
0.8966407 on every arm, as the per-cell build did (torch −1.36488, −1.68953, 0.655087, 0.896641). Machine: Tctl 35 C at
the start, CPUs 0–7 idle and pinned, no game, the bench lock held; the base-vs-head check over the 35-program suite,
the lsm, softmax, attention and model-scale programs: **39 of 44 emit identical kernels and host code** (names,
candidate ids and loop-metadata numbers masked — the new rungs renumber candidate kernel ids), the 5 that differ
are the ones above (T8192, the three lsm, prefill_1024_512_256).

## 4. Tests

Workspace 1459 pass, 0 fail (the CUDA backend's differential tests with the matching driver userspace). New:
`uniform_base_moves_the_operands_by_the_loop_counter` (facts, coefficients, `uniform_at`, the OFF fact arm, values
through the emulator on both arms), `uniform_base_refuses_a_slice_past_the_array`,
`carry_take_starts_the_carry_in_the_inits_device_copy` (one upload fewer, values both arms),
`uniform_base_on_generated_programs` (`testgen::SliceProg`: 1–4 KV groups a loop's trips, 1 / 2 / 4 query heads a
group, QKᵀ or P·V plain / masked where the counter moves the mask / scaled / divided per row, K/V shared or per
sequence, head widths 3–32, the group's projection into a carried `y`, a reader of the init before the loop; 64
programs, 4 emulator modes, every site candidate, CPU, the OFF arms on a third): uniform-facts 108, post 86, plain 22,
uniform tiles 41 programs (post 35, splits 6), refused (a 12-byte step) 23, carry-takes 25, candidate runs 26, OFF
runs 22. `long_rows`: the generated long rows also run `s = 8` as phase launches (`phase launches` counter), and the
rule's cases on the 5080's properties (four rows of 32768 and 32 of 8192 now slice; four of 2048 do not).

## 5. Open

1. **Flash-style streaming (long-rows open 1)** — the facts give attention flash's laws (`fold_combine`: P·V's
   `mx` Rescale, `den` Factor ÷) and the attention is now tile sites at every length, but no unit streams a row
   block over key blocks; at T8192 the scores and `pe` cross DRAM (1 GiB each, written and read: ~4 GiB a group,
   the softmax flow's 3.1 ms a group is that traffic). torch-compile already folds `exp(s − m)` into its P·V
   template (`triton_tem_fused__softmax_bmm_div_exp…`, `pe` never stored). Missing: a tile operand read through an
   element law (`elem_plan`'s `exp(s[e] − mx[row])` as P·V's `a` load: `pe` dissolved, −1 GiB a group), then the
   streamed row-block unit (per key block: q·Kᵀ, block max, rescale of the running `(den, O)`).
2. **Causal blocks** — the guarded QKᵀ post site computes every masked cell's fold (half its FFMA). Missing fact: the
   guard's affine form over a tile block's digit box, so a block whose cells all take the false arm skips its `k`
   loop.
3. **attn_1024_64** (1.37×) — on the 5080 three kernels (QKᵀ 128 blocks for 168 slots, the softmax flow, a P·V table
   of 32-thread blocks); unchanged here. lsm 32 × 8192's tuned q·W pick is slower than its rules pick (tune).
4. **`slice-launches` on the rented parts** needs `row_iter_ns` (only the rtx-5080 has it: the rule slices nothing on
   a part without it — cross-arch open 5): one `calib.cu` line per part.
5. A post site with a 64-bit cell index (`post_sites`' `i32` index) is not reached by a program that fits 16 GB
   (the 32-head score array would be 2³¹ cells, 8 GiB); left as is.

Raw (local, not in git): `.claude/raw-evidence/local-rtx-5080/long-prefill/` (results, box output, conditions, CUPTI
traces and summaries, the bundle's evidence, `commands.txt`, `env.sh`).
