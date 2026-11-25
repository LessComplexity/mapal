# local-rtx-5080 — prefill precision: the bf16-permitted model's wrong tokens; the projections on tensor cores in both precision classes (2026-11-25)

Branch `prefill/precision` from `16d668c6` (the full-model merge). RTX 5080 (sm_120, 84 SMs), shared Threadripper
9970X; timed processes on CPUs 16-23 under the bench lock, builds on 16-23,48-55 (`nice 10`). **Machine load 20–120
from other agents' jobs during every timed run** (logged per run in the raw `time_all.log`); every comparison below is
same-session ABBA, the absolute times carry that load. Raw (never in git):
`.claude/raw-evidence/local-rtx-5080/prefill-precision/build/` — the programs, builds, `r1_*_n*.json`
(`run_mapal.py`, warmup 2 + 5 reps), `r1_tr*_*.log` (`MAPAL_GPU_TRACE`, 4 processes a build), cubins, SASS,
`q_*.json` (the same builds re-timed beside the competitors); `prefill-precision/competitors-f32/` (the f32-compute
competitor sweep).

## 1. The `fold: bf16 | f32` build's wrong tokens (token 84707 repeated) — fixed

**Bisect.** A two-layer mini Llama (vocab 512, hidden 256, random bf16 weights, the same program text) reproduced it
on the device: `--gpu-off=round-store` gave the f32 values; the permission on the down projection alone broke the
prompt's logit (170.7 against 161.6), on the output head alone every decode step (drifting 162 → 139).
`--gpu-off=epilogue` hid the first. Reduced to a straight-line program: one function (gate, up, SiLU·up, a
`: bf16 | f32` down projection, a residual) called twice. Its device unit called `call float @fn5_cell` on a
function defined `bfloat @fn5_cell` and stored `<4 x float>` into the second copy's `mixed`.

**Root cause.** `CategoryIr::apply_store` (`precision_plan.store`, sweep-gemm `round-store`) rewrites map and fold
*bodies* in place — the producer's result rounded (`Narrow`), each element read widened (`Widen`), the body inputs'
capture types narrowed. The inliner and R-UN's loop replay share one body between all copies of a map (FuncId 3 used
by 4 maps in the repro; the model's 16 layers; the prompt's and a decode step's output head, both `project<1, …>`).
A store taken for one copy rewrote every copy: a kept-f32 copy's producer returned bf16 into f32 storage and its
readers widened f32 bytes as bf16. `mapal_ir::validate` rejects such a graph — nothing ran it.

**Fix** (`mapal-ir/src/algo/precision.rs`, commit `9706b4a6`): before rewriting, each body the store changes (the
producer's, every body its walk enters) is owned by the one morphism it enters through — `own_body` gives a shared
body a deep copy (objects, morphisms, admitted sets, nested bodies: `clone_body`), and the walk is taken again over
the owned bodies (no duplicate reads when one store enters a body twice: the key and value projections). `Facts::
add_func` gives a copy its fact slot. Device side: two `devgen` producers of one body and length were defined twice
(`mapal_gen_<id>`): one kernel now. Diagnostic `MAPAL_DEBUG_STORE=1`: each candidate array, taken or the site classes
it would lose (the model: 34 `normed`, 32 `normed_again`, 32 `mixed` taken; `attended` refused — its producer P·V is
a mixed-radix tile site, class 3).

**Test** `half_tile::round_store_shared_bodies_on_generated_programs`: 24 generated programs (`testgen::
SharedStoreProg`: a GEMM function inlined at 2–3 sites, its `a` read by `: bf16 | f32` products, some copies' `a`
also printed, `twin`: two folds of one body in one store's walk). Facts (one store per unread copy), the rewritten
graph valid and idempotent, emulated device values within D3 of the weakest member in every emulator mode; every
coverage counter > 0. **Without the fix it fails** (the rewritten graph does not validate).

## 2a. f32 activations × bf16 weights (the program as written, default face): `split-rate`

**Intent.** The seven projections on `mma.sync.m16n8k16.bf16` as bf16x2n (`a` = hi + lo bf16 parts, `b` exact: two
products a k-step, D2's bound admitted), where the split realizes more than the FFMA tile it replaces.

**Realized SASS** (ptxas 13.5, sm_120, the hot loop's instructions; `sassloop.py`):

| tile | loop | per pipe instr | mix |
|---|---|---:|---|
| FFMA (q / o / down, 8×8 a thread) | 1141 | 1.11 a FFMA | 1024 FFMA, 64 LDS, 16 STS, 4 LDG, 25 int |
| FFMA (gate, 8×16) | 2220 | 1.08 a FFMA | 2048 FFMA, 96 LDS, 32 STS, 8 LDG, 25 int |
| FFMA (key / value, 4×4) | 315 | 1.23 a FFMA | 256 FFMA, 32 LDS, 6 STS, 3 LDG |
| bf16x2n (mt 2 × nt 4) | 360–364 | 11.3 a HMMA | 32 HMMA, 64 FADD (flush), 32 cvt, 19 LDS, 4 LDSM, 4–6 LDGSTS, ~202 int |
| bf16x2n (key / value, mt 1 × nt 2) | 320 | 20.0 a HMMA | 16 HMMA, 40 FADD, 32 cvt, 19 LDS, 4 LDSM, 4 LDGSTS |

**Explained.** The 5080's 128 FP32 lanes on 4 schedulers take a warp FFMA every clock: the FFMA tile is issue-bound
(1.08–1.23 slots an FFMA, register-staged single panels whose load latency is exposed: gaps-disasm C3-fma) and
realizes 0.54–0.68 of its 52.7 Tflop/s peak — the calibrated `DeviceCost::flops_per_ns` 32 200 is that. The bf16
pipe takes an `HMMA.16816` every 32 clocks a scheduler: the split tile's 11.3 slots (`split_slots_pm` prices 12.0,
`issue_pm` 375) leave it matrix-bound; it realizes 0.60–0.97 of the 105 Tflop/s bf16 pipe. The rule priced the FFMA
side at the peak (512 / 2 against 1.6 × 256 flops an SM a clock) and refused all seven.

**Change** (commit `03e2251a`): `Gpu::ffma_tile_flops_per_ns` — the FFMA tile's measured rate
(`DeviceCost::flops_per_ns`), the peak where a part measured none — is the side `rules::split_rate_pays` weighs.
θ = 1.6 stays: the A100 has no measured FFMA rate and its tile reaches 0.92 of its peak (5fcb401b's 4096² FFMA tile).
Replay: the rtx-5080 tf32x3 loser (0.55) and narrow1 winner (3.3) keep their sides, tensor-cores-2's open 5080
bf16x2n line is picked, two new lines (this model, issue 375 and 671); **172 picked / 82 null / 16 mixed / 22 open**.
Test `split_rate_weighs_the_ffma_tiles_achieved_rate`. Applies only with the calibration line
(`MAPAL_GPU_CALIB=benches/results/evidence/local-rtx-5080/calib/rtx-5080.calib`; this box has no
`~/.cache/mapal/gpu-calib`, and without it the native profile has no measured FFMA rate: the FFMA tile, as before).

**Per projection, µs a layer** (calibrated builds, OFF arm `--gpu-off=split-tile` = the emission before; median of 4
traces each, interleaved):

| projection (rows × c × k) | 128: FFMA → split | 512: FFMA → split | 2048: FFMA → split |
|---|---|---|---|
| query 2048 × 2048 | 57.9 → 38.5 | 144.3 → 115.4 | 455.2 → 336.0 |
| output 2048 × 2048 (+ residual) | 59.6 → 39.5 | 163.6 → 131.1 | 491.5 → 365.9 |
| key, value 512 × 2048 (each) | 28.0 → 25.3 (FFMA kept: refused at the site) | 53.5 → 40.4 | 146.2 → 118.6 |
| gate 8192 × 2048 | 165.0 → 113.2 | 480.3 → 340.1 | 1930.9 → 1265.0 |
| up 8192 × 2048 (+ SiLU·gate) | 193.9 → 124.9 | 525.4 → 356.8 | 1945.6 → 1299.7 |
| down 2048 × 8192 (+ residual) | 184.5 → 134.4 | 545.7 → 443.6 | 1967.0 → 1340.5 |
| **prompt kernels, ms** | **14.0 → 10.6** | **38.0 → 30.0** | **169.6 → 133.8** |

## 2b. bf16 activations (`fold: bf16 | f32` on `project`: the competitors' precision)

Every projection on the `HalfMma` tile (`ldmatrix` fragments of the bf16-stored activations; `o` reads `attended`
f32, rounded at its fragment: P·V cannot store narrow). Per projection, µs a layer (calibrated, median of 4 traces),
against cuBLAS bf16 (`torch.mm` on the same shapes, 512 rows, median of 50: `torch_proj.py`):

| projection | 128 | 512 | cuBLAS 512 | 2048 | the 512 tile |
|---|---:|---:|---:|---:|---|
| query | 25.1 | 64.0 | 58.9 | 201.0 | 64×128 block, 8 warps (32×32), bk 32 × 3 stages, 128 CTAs |
| output (f32 `a`) | 27.3 | 72.7 | 58.9 | 268.7 | same, 2 stages (an f32 panel), LDS + cvt |
| key, value (each) | 14.8 | 25.9 | 23.0 | 65.2 | 32×64 block, 16×16 warp tiles, bk 64 × 3 |
| gate, up (each) | 73.1 | 201.3 | 183.9 | 621 (merged pair 1242) | 128×128, 4 warps (64×64), bk 32 × 2, 190 registers |
| down (+ residual) | 78.6 | 241.1 | 202.4 | 807.7 | 64×128, bk 32 × 3, 128 CTAs |
| **prompt kernels, ms** | **7.5** | **20.4** | | **101.8** | |

SASS of the 512 tiles: gate / up 117 instructions a 64-`HMMA` loop (1.83 a HMMA: 16 LDSM, 8 LDGSTS) with `DEPBAR.LE
SB0, 0x0` (one panel in flight); query / down 81 a 16-HMMA loop, `DEPBAR 0x1`. All matrix-pipe-bound by issue. What
separates them from cuBLAS (7–19 %) is the grid on 84 SMs: 128 or 256 blocks at 2 an SM put 2 or 4 blocks on the
busiest SM where 1.52 / 3.05 is the mean (C4, the tile grid's rounding). Tried here, measured, not kept: a forced k
cut (`--splitk=2` / `4`: key / value 25 → 40 / 33 µs, gate / up 200 → 220 µs — the zero pass and `red` partials cost
more than the rounding saves; the epilogue sites cannot cut); the ring of gapfix C3 (`mma-fly`) does not engage (its
gate is the operand's own bytes against the L2 keep; a 1 MiB keep arm left every tile's stages unchanged).

## Full model: before → after (calibrated native-gpu; ABBA medians, load 20–58)

| prompt | f32, FFMA (before) | f32, bf16x2n (after) | bf16 permitted (after) | decode ms/token (f32 off / on / bf16) | tokens = HF f32 (off / on / bf16) | bf16 run = HF bf16 eager (prefix / equal) |
|---|---:|---:|---:|---|---|---|
| 128 | 29.3 | **25.5** | **23.4** | 3.53 / 3.51 / 3.63 | 128 / 128 / 128 | 128 / 128 |
| 512 | 90.7 | **80.3** | **68.0** | 3.67 / 3.57 / 3.63 | 128 / 128 / 37 | 67 / 70 |
| 2048 | 353.6 | **310.0** | **279.1** | 3.87 / 3.72 / 3.81 | 128 / 128 / 124 | 61 / 61 |

TTFT ms. (For scale, the task's quiet-machine baseline of the uncalibrated build: 28.3 / 85.2 / 332.7 ms; HF f32
itself equals HF bf16 eager in 49 / 32 / 57 tokens, so the bf16 run follows the bf16 reference at least as closely
as the f32 math does.) Prompt launches: 242 (f32) / 258 (bf16) at 512. Peak memory (nvidia-smi): FFMA 5.2 / 5.4 / 6.8 GB,
split 4.8 / 4.9 / 6.2, bf16 4.7 / 4.9 / 6.2 (the difference not traced here; the weights are still held twice).

**Where the rest of the TTFT goes** (512, bf16: 68 ms with 20.4 ms of kernels): the prompt's cache chain through the
host — 32 uploads, 671 MB (268 MB at 128, 2.28 GB at 2048), and 33 downloads (full-model open item 3, the residency
step's); then the attention, f32 FFMA in both classes (scores + softmax + P·V ≈ 0.34 ms a layer at 512, 1.95 ms at
2048: 31 of the 102 ms of kernels; the permission is only on `project`).

## Same-class competitors (each class against tools computing at its own precision)

No gain is claimed across classes: the f32 class is compared only with f32-compute tools, and the bf16 class only
with bf16 tools.

**f32 class.** Mapal as written (f32 activations × the bf16 weights). The competitors take the bf16 checkpoint widened
to f32, with f32 activations and KV cache, and TF32 off (`allow_tf32 = False`, `float32_matmul_precision("highest")`).
Command: `compare.py --tools hf-eager,gpt-fast,hf-static-compile --batches 1 --runner-arg=--dtype
--runner-arg=f32`, with runner copies that add a `--dtype` flag (the shared harness is unchanged; the patch is raw
`competitors-f32/runners-f32.patch`). The competitors ran 16:59–17:05 at load 5–18 under the bench lock. The Mapal
builds above were re-timed right after (17:05–17:08, load 4–6, one `run_mapal.py` process each, `q_*.json`). Every
row gives **128 / 128 of the HF f32 reference's tokens** at all three prompts.

| tool, f32 compute | TTFT 128 ms | 512 | 2048 | decode 128 ms/token | 512 | 2048 | peak MB (512) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mapal, bf16x2n split (after) | 25.9 | 82.4 | 297.9 | **3.63** | **3.72** | **3.60** | 4 886 |
| Mapal, FFMA (before) | 29.8 | 86.7 | 343.6 | 3.66 | 3.85 | 3.69 | 5 392 |
| gpt-fast f32 (max-autotune, CUDA graphs) | **13.8** | 48.0 | **213.9** | 5.73 | 5.74 | 5.92 | 6 628 |
| HF eager f32 | 14.4 | **41.8** | 241.9 | 7.16 | 7.48 | 8.16 | 5 204 |
| HF static cache + compile f32 | 17.1 | 44.4 | 245.5 | 6.77 | 7.55 | 10.49 | 5 372 |

- **Decode:** Mapal is 1.54–1.64× faster than gpt-fast f32 and 1.97–2.27× faster than HF eager f32. Mapal reads the
  bf16 weight bytes and widens them in registers, which gives the same values with half the bytes. The f32 tools
  store the widened weights: 4.9 GB a step, about 5.1 ms at ~960 GB/s, and that bound sets their floor. Mapal's
  decode lead comes from that bound, not from a precision difference.
- **TTFT:** Mapal is 1.88 / 1.97 / 1.39× behind the best f32 tool. Its prompt kernels alone (10.6 / 30.0 / 133.8 ms,
  traced above) are below every f32 tool's whole TTFT. Most of the gap is outside the projections: the cache chain
  through the host and the attention's f32 FFMA.
- **Precision of the split:** Mapal's split is bf16x2n: `a` = hi + lo bf16 parts with exact bf16 weights, accurate
  within D2's admitted bound, not full f32 products. The competitors' f32 GEMMs are SGEMM (FFMA).
- **Decode noise:** Mapal's decode was 3.48–4.19 ms rep to rep in this hour (shared host), against 3.34–3.55 on the
  quiet full-model run. This step does not change the decode step.

**bf16 class.** `fold: bf16 | f32` on `project` only: attention, norms and the cache stay f32 in this program. The
competitors are fully bf16. Their numbers come from full-model.md (the same harness, this afternoon, load 0.6–3).
Mapal (`q_bf_*.json`) gives TTFT 22.5 / 71.2 / 271.5 ms against the best bf16 tool's 4.5 (gpt-fast) / 13.4 (vLLM)
/ 44.9 (SGLang): **5.0 / 5.3 / 6.0× behind**. Mapal's prompt kernels alone (7.5 / 20.4 / 101.8 ms) are 1.5–2.3×
those whole TTFTs. Decode is 3.69–3.91 ms in this hour (3.63–3.81 in the ABBA above) against gpt-fast's 2.95–3.05.

## Open

1. **C4, the tensor tile's grid on the SMs** (2b's remaining 7–19 % to cuBLAS): a block geometry that puts the
   mean, not the ceiling, on the busiest SM (the leaders' 64×64 × 6-stage family) needs the warp tile's matrix-pipe
   interval / latency property (`mma_calib.cu`, gaps-disasm day-0); the k cut measured worse. Not attempted further.
2. **The split side's realized fraction**: θ = 1.6 still carries the A100's under-realized issue-bound split tiles
   (0.39–0.55 of their pipe at issue 1.0) while the 5080's matrix-bound ones realize 0.60–0.97; the rule passes the
   5080's bf16x2n at 1.64 against 1.6 where they measured 1.12–1.48. The split tile's realized fraction as a function
   of its issue (the open fact tensor-cores-2 named) would let θ drop to a margin; an A100 round is needed to place it.
3. **`attended` stays f32** in the bf16 class (P·V is a mixed-radix tile site that cannot produce a narrow output):
   the output projection reads an f32 panel (72.7 against 64.0 µs).
4. **No calibration file on this box**: the default build (no `MAPAL_GPU_CALIB`) keeps the FFMA tile. Installing the
   committed `rtx-5080.calib` line as `~/.cache/mapal/gpu-calib` is the integrator's call (it changes every agent's
   plan-cost builds on this machine).
5. Two commits on this branch are not this step's agent's: `f56d59a7` (zero-hoist) and `0d9686a6` (a second
   shared-body bug: a body's `cell-law` and epilogue-member device variants shared the name `<body>_cell`; test
   `tests/shared_bodies.rs`). They were made in this worktree by another session while this step ran; the measured
   builds' CLI was built with those edits in the tree (16:08). The tokens agree in every build. Rebuilt with the
   CLI at `0d9686a6` (both commits in, a clean tree), all nine measured builds (three classes × three prompts) are
   byte-identical to the measured ones: host `.ll`, device `.ll` and `mapal_gpu.c`. So the numbers here measure the
   committed code.

## Tests

`cargo test --release`: `mapal-ir` 117, `mapal-lower` (lib, `compute_type` 7), `mapal-backend-llvm` lib 135 (incl.
`rung_rules_replay`, `split_rate_weighs_the_ffma_tiles_achieved_rate`), `half_tile` 6 (the new generated test: 24
programs × 4 emulator modes), `narrow` 4, `split_tile` 7, `tensor_tile` 2, `tf32_tile` 2, `large_gemm` 3,
`hardware_properties` 2 — all pass. Suite: `suite_diff.py --off split-tile` (rtx-5080, 35 programs + 2 cells × 5
faces): **same = 45** — no suite program has an f32 × bf16 product, none takes a split tile on either arm, nothing to
time; the round-store fix moves no suite program either (none admits `bf16`).

## For integration

CLAUDE.md facts table (edits, not made here):

- `precision_plan` row, `store`: append "— a store rewrites only its own copy of a body several maps share
  (`apply_store` owns each body it changes: `own_body`, a deep copy; the inliner's and R-UN's copies share bodies)".
- backend `Gpu` properties row: add "`ffma_tile_flops_per_ns` (the FFMA tile's achieved rate, `DeviceCost::
  flops_per_ns`; the peak where unmeasured) — the side `rules::split_rate_pays` weighs a split tile against".

`docs/tailored-fixes.md`: unchanged (no tailored fix).
