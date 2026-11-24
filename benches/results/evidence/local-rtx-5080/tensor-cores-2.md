# tensor-cores-2 — split tiles at their issue rate; bf16 weights against f32 activations on tensor cores

Branch `fix/tensor-cores` from `8ff66d34` (`local-rtx-5080`, the 5fcb401b confirmation round's evidence). RTX 5080
(sm_120) locally on CPU set 16-23,48-55; rented: an A100-SXM4-40GB (400 W, driver 595.71) and an H100 PCIe 80 GB
(driver 595.84). Raw output: `.claude/raw-evidence/local-rtx-5080/tensor-cores-2/<part>/` (not in git).

The step's three gaps (5fcb401b, `cross-arch/README.md` findings 2 and the matrix's bf16w rows):

1. `split-tile` (tf32x3) lost 12 / 12 on the A100 (OFF / ON 0.56–0.79), read as "panels reloaded through registers,
   40 LDG / 0 LDGSTS in the hot loop";
2. Hopper's 16-bit tiles run `mma.sync`, not `wgmma`;
3. the bf16-weight prefills ran 11.8–17.5× torch on the A100 / H100.

## 1. The A100's split loser: not its staging — its issue

**The 40 LDG were the cold path's.** The 5fcb401b split kernel (`mapal_tile_8e7ca8fa9c3513ac`, `mm_4096x4096x4096`,
the A100's own cubin) stages both panels by `cp.async`: its k loop (`0xa70`–`0x6ef0`) holds 8 `LDGSTS.E.BYPASS.128`,
8 `LDSM` and no `LDG`. The 48 `LDG` sit in the guard's cold loop (`0x7310`–`0x8030`, the warp's tile recomputed by
FFMA, unrolled 4×: 128 FFMA a trip), which `evidence.py` picked as the hot loop because it counted an `HMMA` as one
multiply against an FFMA's one: 128 > 96. Fixed: a matrix instruction weighs 32 FFMA (an m16n8k8 is 1024
multiply-adds a warp, an FFMA 32) in both the SASS and PTX hot-loop choice. On this step's bundles the intent
"LDGSTS in the hot loop, no LDG" holds on every split kernel (A100 52 / 52, H100 39 / 39).

**What the k loop did issue:** 1612 SASS instructions a 96-`HMMA` panel (mt 2 × nt 4 tiles, 4 k8-steps): 346 LOP3,
266 ISETP, 196 SEL, 192 FADD, 152 IADD3, 128 FSETP — the per-fragment cut (`cvt.rna.tf32.f32` emulated in integer
code, the overflow select, the guard's class checks: 21.7 slots an element), plus 4 FADD a tile and k-step (the flush).
A scheduler issues one instruction a clock; the A100's TF32 pipe takes an m16n8k8 in `2048 / (1012 / 4)` = 8.1
clocks: 408 slots against 194 pipe clocks a k-step — **issue-bound 2.1×**, 1012 → 481 flops an SM a clock, under the
499 the three passes need against the FFMA tile (θ · 3 · 128).

### The realization (`split-issue`, OFF arm `--gpu-off=split-issue`: the 5fcb401b cut)

- **The cut by integer rounding**: `hi = (x + 2¹²) & ¬(2¹³ − 1)` (`cvt.rna`'s ties-away rounding, `mapal_ir::split::
  tf32_hi` bit for bit on every finite `x` below the overflow threshold), `lo` the same of `x − hi`. The value
  classes go to the guard: `split_guard` sends `|x| ≥ 2¹²⁸ − 2¹⁰³` (its `hi` would round to infinity: the guard now
  takes what the overflow select did), non-finite and `0 < |x| < exact_above` to the FFMA cold path in one unsigned
  range check. 10 slots an element (sm_80 SASS).
- **The flush depth** (`split_flush`): each tile's zeroed chain carries `j` k-steps before its round-to-nearest add,
  `j` the deepest power of two `SplitForm::admits` gives at the site's `K` and pieces, at most a panel's k-steps
  (A100 2048-row GEMMs: 4; the coefficient's chain term `c_g · j · P · Σρ` is inside D2 at every admitted `j`).
- The A100 panel: **792** instructions a 96-`HMMA` panel (from 1612), 8.25 slots an `HMMA`; the model 8.00.

### The rule prices the realization (`rules::split_issue_pm`, `split_slots_pm`)

`split_rate_pays(g, op, P, issue_pm)`: the matrix rate over the tile's issue ratio (at least 1) against `θ · P` FFMA
rates. `split_issue_pm`: a warp's k-step issues `P·mt·nt` matrix instructions, `4·mt·nt / flush` adds and `per_cut`
slots for each element its thread cuts (`a`: `mt · k/2`, `b`: `nt · k/4`), one a clock on its scheduler; the pipe
takes each instruction `2·m·n·k / (matrix_flops_per_clk / schedulers)` clocks. Physical properties: the matrix and
FFMA rates (`calib.cu`), the schedulers. `per_cut` is read off ptxas 13.0's sm_80 SASS (`SPLIT_CUT_*`: TF32 10, the
5fcb401b cut 22, bf16x3n 13, bf16x2n 10; `MMA_ADDR_REGS`'s precedent) — the plan line states `flush`, `issue_pm`,
`slots_pm`, and intent.py checks the hot loop's SASS within 1.25× of `slots_pm`.

**The margin θ, from the A100's measured tiles** (`SPLIT_RATE_PM` 1300 → 1600):

| issue ratio (`issue_pm`) | sites (A100, OFF / ON = FFMA tile / split) | θ the site allows | verdict |
|---|---|---:|---|
| 0.83 (mt 2 × nt 4, small grids) | mlp_1024 1.563, loopmm_1024_8 1.449 | 3.2 | win |
| 1.00 (mt 2 × nt 4) | pj_m2048 n4096 1.426, n6144 1.453, n28672 1.480, n4096 k14336 1.450; mlp_2048 1.428, mlp2_2048 1.452, epi_4096 1.538, rt_mlp2_2048 1.466, pj_m128_n28672 1.364, llm_pre_t2048 1.230 | 2.67 | win (10 / 10) |
| 1.40 (mt 2 × nt 2) | pj_m128 n4096 1.105, n4096 k14336 1.055 (cut S = 8); n6144 **0.809** (uncut: 192 blocks on 216 slots; SASS 13.97 slots an HMMA against 11.17 priced) | 1.91 | mixed — open |
| 1.81 (mt 1 × nt 2) | mlp_512 **0.834**, pj_m128_n2048_k7168 **0.873**, causal_1024_256 0.997 | 1.47 | loss / null |
| 2.10 (the 5fcb401b cut) | the 12 programs of finding 2, 0.56–0.79 | 1.27 | loss |

θ = 1.6 separates every class but the mixed one; the rtx-5080's tf32x3 loser (θ ≤ 0.33) and narrow1 winner (2.0)
stand on either side, as do the H100 PCIe's bf16x2n winners (θ ≤ 2.78) and forced tf32x3 losers (θ ≤ 1.37). Replay
lines: 14 (A100 8, H100 PCIe 4, RTX 5080 2 — one `OPEN`; `rules::tests::rung_rules_replay` 152 picked / 72 null /
11 mixed / 22 open).

**Open (the mixed class):** `pj_m128_n6144_k4096` is the one 1.40 site `mma-combine` leaves uncut (192 tiles under
one wave of 216 slots) while its FFMA alternative fills the part by stream-K; its SASS also issues 25 % over the
priced slots. Candidate fact: the split tile's own wave fill against the FFMA tile's (the cut's `mma_pieces` price
reads the unsplit tile's rate).

## 3. bf16 weights against f32 activations: which products stayed off tensor cores

From the 5fcb401b emission of `pt_llm_pre_t2048_bf16w` (the same on every part): its two projections — QKV 2048 ×
6144 × 4096 and O 2048 × 4096 × 4096, `x · wᵀ` with `x` f32 and `w` bf16 under the default `{f32}` — took
`kind=tile face=OrderedFma narrow=b:bf16`: the FFMA tile. Neither realization applied: `half_operands` needs a 16-bit
member in the product set (`: bf16 | f32`, the `_bf16` rows) or two narrow operands (narrow1); `split_operands` named
the f32 × bf16 pair "bf16x3n: not emitted". Traced on the A100 (FFMA arm): QKV 7.34 ms, O 5.13 ms, the attention
flow kernel 8.04 ms of 20.75.

### The realization: bf16x3n / bf16x2n on the bf16 warp product

- `split_operands`: an f32 × bf16 pair (either side) is a split site on `mma.sync.m16n8k16.bf16` with panels at their
  storage widths (the f32 panel 4 bytes by `cp.async`, the bf16 one 2 bytes read by `ldmatrix`).
- **bf16x3n** (`mapal_ir::split`, e = 0): the f32 operand's pair cut as its fragment is read — `p₀ = cvt.rn.bf16x2(x)`,
  `p₁` of the exact residual, `p₂` of what is left (exact above `2⁻¹¹⁰`); three products, smallest class first,
  into the tile's zeroed chain; the narrow operand is its own exact widening.
- **bf16x2n** (new `SplitKind::Bf16x2n`: two parts, split error `|a·b − (a₀ + a₁)·b| ≤ 2⁻¹⁶|a·b|`, e = 256 units; K_min
  288 at the measured `g = 2`, 320 at `g = 0`; `bf16_two_parts_within_two_to_minus_16` on every binade edge and 20 000
  draws): two products. `split-least` (OFF: bf16x3n) takes the form of fewest products the bound admits at the site's
  `K` — D2 verbatim (ADR-0040): the realization's a-priori coefficient within `γ_{K+1}(u)·M`.
- The guard (`split_guard`): non-finite, rounding to infinity (`|x| ≥ 0x7f7f8000`) or `0 < |x| < 2⁻¹¹⁰` → the warp's
  tile recomputed by FFMA (`split_cold`, the narrow operand widened from global).
- Rule: `split_rate_pays` at the bf16 rate — A100 2048 (θ ≤ 5.3 at issue 1.50), H100 2702 (θ ≤ 2.8 at its priced
  1.90); GeForce 512 against 2 · 256 refuses (θ ≤ 1.0). The bf16w prefills on the 3090 / 4090 / 5080 keep the FFMA
  tile: no split form pays on a 512-flop bf16 pipe.

**A100, OFF (FFMA tile) / ON:** `llm_pre_t2048_bf16w` **1.741** (20.6 → 11.85 ms; QKV 7.34 → 2.19, O 5.13 → 1.46 ms
traced), `llm_blkpre_t2048_bf16w` **2.672** (72.8 → 27.26 ms), `llm_dec_b32_l2048_bf16w` 1.193; bf16x3n / bf16x2n
1.084 / 1.199 / 1.000. The prefill's remaining 8.4 ms is the attention flow kernel (f32 × f32 scores and P·V under
`{f32}`, no split realization in the stream kernels): the next gap, open.

H100 PCIe: §H100 PCIe below.

## 2. Hopper's 16-bit tiles on `wgmma` — attempted, open

What exists: the capability (`MATRIX_WARPGROUP`: `group(WGMMA_16, Bf16 / F16)`, `a` and `b` from shared-memory
descriptors, accumulator in registers), the TF32 `wgmma` kernels (S53 `wgmma-hopper`: `cᵀ = bᵀ·aᵀ`, `bᵀ` the register
operand, `aᵀ` the 128-byte-swizzled descriptor operand; hopper-wgmma-tma's warp-specialized TMA kernel). What the
16-bit tile needs on top: `wgmma.mma_async.m64nNk16.f32.bf16.bf16` with both operands as descriptors (16-bit
operands may be MN-major through the descriptor's transpose bits, so `b` stored `k × n` needs no register pass), a
64-element (128-byte) swizzle row of `k`, and — for the `_bf16` rows' f32 operands rounded once to bf16 — a
conversion of the landed f32 panel into a swizzled bf16 panel before the fence. Not attempted in code here: the
emulator models `wgmma` TF32 only (`emu_wgmma`), the RTX 5080 has no `wgmma`, and an unvalidated kernel would have
cost the H100 rental; the launch side (descriptors, persistent grid) is the other agent's. The H100's 16-bit rows
therefore keep `mma.sync` at its measured 2702 flops an SM a clock (§H100: the bf16w projections there now run on
it).

## Properties

- **H100 / H100 PCIe `mma.sync` rates**: `calib.cu matrix` on the 5fcb401b SXM box measured TF32 1351 / bf16 2702
  flops an SM a clock (`calib-h100.txt`); the profiles carried Mapal's realized 532 / 1064. Both profiles and
  `mma_rates_of_cc(90)` now take the measurement. With the priced issue it moves no TF32 decision (H100 tf32x3:
  1351 / 1.29 / (3 · 256) = 1.37 < 1.6 — the sm_90 SASS issues 10.1 slots an `HMMA` against the 7.8 priced: θ ≤
  1.05); it takes bf16x2n on the bf16w projections.
- Rungs: `split-issue`, `split-least` (lib.rs; OFF arms are measurement flags; both ship ON on every part).

## Tests

- `mapal-ir` `split`: 7 (K_min table with bf16x2n; the two-part cut's bound).
- `split_tile.rs` (5): the rule per part (tf32x3 on the A100 at an unshrunk tile, refused at the 128 × 256 site's
  1 × 1 warp tile; bf16x2n / bf16x3n by `K`, part and arm), `split_issue_prices_the_realization` (the A100's
  4096-K tile: flush 4, issue ≤ 1.1; OFF: the loser refused), and `generated_programs_within_d2` — 64 generated
  programs (`testgen::SplitProg`, store 3 = f32 × bf16 now realized), the emulator in 4 modes, `MAPAL_EMU_MMA_G=2`
  and guard-less `=0` at the kernel's own flush depth, S = 2 pieces, the OFF arm; every coverage counter non-zero
  (tf32x3, narrow1, **bf16x3n, bf16x2n**, refused-K_min, refused-rate, pieces, post, bt, guard, underflow,
  overflow-select, lo-zero, cancel, fallback, guard-less, narrow1-rule).
- Backend unit tests 124 / 124 (the replay counts; `the_chooser_on_the_measured_cells` and
  `plan_line_reads_back_as_its_launch` pin the FMA tile under `split-tile` OFF: the H100's measured rate otherwise
  priced the 4096³ cell's split); half_tile, tf32_tile, site_choice, gpu_wgmma (the reachability test under
  `split-tile` OFF), hopper_tma, radix, bulk_stage, epilogue, site_geometry, golden_ll: all pass.

## Evidence (intent check)

A100 bundle: 10 866 checks, `split-tile` 52 kernels: mma.sync, HMMA + FADD flush, the guard's VOTE, F2FP in the bf16
cuts' hot loop, LDGSTS and no LDG, no spill — all ok; slots within 1.25× of `slots_pm`: 51 / 52 (`pj_m128_n6144`:
13.97 against 11.17, the open site). The first H100 PCIe bundle (sm_90a SASS, built at θ 1.3, where tf32x3 was taken): the same intents 39 / 39; slots
20 / 39 — the tf32x3 kernels issue 10.1 slots an `HMMA` against 7.8 priced (the sm_80 constant); the bf16x2n kernels within.
A finding, not a pass: `SPLIT_CUT_TF32` read off sm_80 under-prices sm_90's code by 1.3×; no decision moves (the
H100 refuses tf32x3 at either count).

## H100 PCIe (rented, sm_90a, the H100 SM's per-clock rates)

Two sessions (5 and 7 reps ABBA, value gates first: 6 / 6 programs, every arm within D2).

| program | ON (rules) ms | FFMA tile / ON | bf16x3n / bf16x2n | forced tf32x3 / FFMA |
|---|---:|---:|---:|---:|
| llm_pre_t2048_bf16w | 5.256 | **1.690** | 1.114 / 1.119 | — |
| llm_blkpre_t2048_bf16w | 15.05 | **2.158** | 1.191 / 1.194 | — |
| llm_dec_b32_l2048_bf16w | 7.897 | 1.151 | 0.998 / 1.006 | — |
| pj_m2048_n4096_k4096 (FFMA, the rule's) | 1.941 | — | — | 1.220 (refused ✓) |
| pj_m2048_n6144_k4096 | 2.920 | — | — | 1.211 (refused ✓) |
| mlp_2048 | 1.025 | — | — | 1.200 (refused ✓) |

The measured `mma.sync` rate in the profile does not move the H100's f32 GEMMs onto tf32x3 (5fcb401b's warning):
priced at issue 1.29 the rule allows θ 1.37 < 1.6, the sm_90 SASS issues 10.1 slots an `HMMA` (θ ≤ 1.05), and the
forced arm loses 1.20–1.22. The bf16w projections take bf16x2n at issue 1.90 (θ ≤ 2.78) and win 1.69 / 2.16.

## RTX 5080 (local, forced ON: the rule refuses every split here)

9 reps ABBA, `LADDER=1` (every arm timed), the bench lock held, CPU set 16-23,48-55 (Tctl 84–89 C: other agents).

| program | rules (FFMA) ms | forced split / rules | null | the split |
|---|---:|---:|---:|---|
| pj_m2048_n4096_k4096 | 1.836 | 1.953 | 1.001 | tf32x3 (5fcb401b's cut: 2.24) — refused ✓ |
| mlp_2048 | 0.8885 | 2.093 | 1.002 | tf32x3 — refused ✓ |
| llm_pre_t2048_bf16w | 6.834 | **0.757** | 0.956 | bf16x2n — refused: **open** |
| llm_dec_b32_l2048_bf16w | 13.44 | 1.022 | 1.016 | bf16x2n — null |

**Open (a mixed result across parts):** the 5080's bf16x2n QKV kernel runs 1.86 ms against the FFMA tile's 2.6–3.9
(traced; 103 Gflop: 108 bf16 Tflop/s, ~90 % of its pipe — pipe-bound, issue 0.38), the program 0.757; the rule
(2 · 256 bf16 against 1.6 · 256 FFMA: θ ≤ 1.0) refuses it. The A100's issue-bound tiles realize ~39 % of their pipe
at issue 1.0 (19.9 useful Tflop/s at 3 products against 154 TF32 Tflop/s). One margin cannot hold both: the tensor
tile's realized fraction as a function of its issue (and the pipe clocks per instruction against its latency) is the
missing fact. Replayed as `OPEN` (`rung_replay.tsv`); the 3090 / 4090 / 5080 bf16w prefills keep the FFMA tile.

## Measurements

Competitors: the same box's cuBLAS / cuBLASLt f32 legs (`gemm_lib`, fresh and flushed, within the f32 gate) where
the bundle ran them; otherwise the 5fcb401b round's fastest leg on that part type (`matrix.md`; the H100 row's is the
SXM part's). bf16 legs are never like-for-like for an f32-activation program (they round activations too).

| workload | part | OFF | ON | fastest competitor (leg) | explanation |
|---|---|---:|---:|---|---|
| pj_m2048_n4096_k4096 | a100 | 4.91 ms (FFMA) | **3.445** | 3.672 (cuBLAS f32, this box) | tf32x3 at issue 1.00: 792 slots / 96 HMMA (5fcb401b's 1612) |
| pj_m2048_n6144_k4096 | a100 | 7.37 | **5.071** | 5.594 (cuBLASLt f32) | the same |
| pj_m2048_n28672_k4096 | a100 | 34.1 | **23.07** | 25.33 (cuBLASLt f32) | the same |
| pj_m2048_n4096_k14336 | a100 | 16.9 | **11.66** | 12.65 (cuBLAS f32) | the same |
| pj_m128_n6144_k4096 | a100 | 0.546 | 0.675 | 0.476 (cuBLASLt f32) | issue 1.40, uncut under-filled grid: open |
| mlp_512 | a100 | 0.106 | 0.127 (ships FFMA now) | — | issue 1.81: θ 1.6 refuses |
| llm_pre_t2048 | a100 | 20.5 | 16.7 | 12.94 (liger, 5fcb401b) | projections tf32x3; attention flow FFMA |
| llm_pre_t2048_bf16w | a100 | 20.6 | **11.85** | 1.082 (torch-bf16-compile-cg, 5fcb401b) | bf16x2n projections 12.47 → 3.65 ms; the attention flow kernel 8.4 ms (f32, no split realization): 15.0× → 10.9× |
| llm_blkpre_t2048_bf16w | a100 | 72.8 | **27.26** | 4.833 (torch-bf16-compile) | bf16x2n: 11.8× → 5.6× |
| llm_dec_b32_l2048_bf16w | a100 | 13.9 | 11.68 | (torch-bf16-cudnn 1.41× in 5fcb401b) | bf16x2n, issue 1.63 |
| llm_pre_t2048_bf16w | h100-pcie | 8.88 | **5.256** | 0.350 (torch-bf16-compile-ro, H100 SXM) | bf16x2n at issue 1.90 |
| llm_blkpre_t2048_bf16w | h100-pcie | 32.5 | **15.05** | 1.308 (torch-bf16-compile-cg, SXM) | the same |
| pj_m2048_n4096_k4096 | h100-pcie | **1.941** | 2.37 (forced) | 1.930 (cuBLASLt f32) | tf32x3 refused (θ 1.37 < 1.6), measured 1.22 slower |
| llm_pre_t2048_bf16w | rtx-5080 | **6.834** | 5.17 (forced) | 1.934 (torch-bf16-compile-cg, 5fcb401b) | the rule refuses bf16x2n here: open (above) |
| pj_m2048_n4096_k4096 | rtx-5080 | **1.836** | 3.59 (forced) | (5fcb401b: cuBLAS f32) | TF32 = FFMA rate: three passes lose ✓ |

## Rentals

| part | instance | offer $/h | minutes | cost |
|---|---|---:|---:|---:|
| A100-SXM4-40GB (first offer taken before start) | 62339048 | 0.668 | 0.25 | $0.00 |
| A100-SXM4-40GB (EPYC 7532, 400 W) | 62339082 | 0.843 | 13.4 | $0.19 |
| H100 PCIe 80 GB (Sapphire Rapids) | 62341187 | 2.001 | 5.3 | $0.18 |
| H100 PCIe (second session, every arm timed) | 62342001 | 2.001 | 3.9 | $0.13 |
| **total** | label `mapal-tc-*`, every one destroyed (checked by label) | | | **$0.50** |

No H100 SXM offer was under $6.67/h; the PCIe part shares the SM (per-clock rates, sm_90a code) and was rented
instead.

## Open problems attacked

| problem | status |
|---|---|
| (1) split-tile loses on the A100 | **closed**: the "register reload" was the evidence tool's hot-loop choice (fixed); the loser was issue-bound (2.1×); the integer cut + the bound's flush depth issue 792 instead of 1612 a panel, and the A100 now wins 1.43–1.54 on its 2048-row GEMMs and MLPs (beats cuBLAS f32 by 6–9 %); the rule prices the issue (`split_issue_pm`) and its margin is placed on the measured tiles. Remaining: the 1.40-issue class's uncut under-filled site (attempted, open: the split tile's own wave fill) |
| (2) H100 16-bit tiles on `wgmma` | **attempted, open**: design stated (§2); not built — no emulator model for 16-bit `wgmma`, no local Hopper; the H100's measured `mma.sync` rate now in its profile |
| (3) bf16-weight prefills | **partly closed**: the products off tensor cores were the f32 × bf16 projections (FFMA tile on every part); bf16x2n / bf16x3n take them where the bound and rate admit — A100 1.74 / 2.67, H100 PCIe 1.69 / 2.16. The prefill's remaining gap is the f32 attention flow kernel (8.4 of 11.85 ms on the A100): open. GeForce: the rule refuses a measured 1.32× win (open: the tile's realized fraction at its issue) |
| found: `pt_pc_*` / `pt_pm_*` emission panic (`stream.rs:1332` index out of bounds) | pre-existing at 8ff66d34 on the 5080 / A100 (`pt_pc_128`, `pt_pm_2048_128`, …): not this step's, reported |

## For integration

- `benches/results/evidence/local-rtx-5080/README.md`: a row — "tensor-cores-2 (`fix/tensor-cores`): split tiles
  priced at their issue (`split-issue`), f32 × bf16 on the bf16 warp product (bf16x2n / bf16x3n, `split-least`);
  A100 tf32x3 1.43–1.54 over the FFMA tile (12 / 12 losers → winners), bf16w prefills A100 1.74 / 2.67, H100 PCIe
  1.69 / 2.16; [tensor-cores-2.md](tensor-cores-2.md)".
- `CLAUDE.md` facts table: no new graph fact. The `precision_plan` row's split mention (if any, after the f32-tensor
  -cores merge) gains `SplitKind::Bf16x2n` (`mapal_ir::split`, split error `2⁻¹⁶`, K_min 288).
- `matrix.md` / `matrix.json`: the A100 rows of `pj_m2048_*`, `mlp_2048`, `mlp2_2048`, `epi_4096`, `rt_mlp2_2048`,
  `llm_pre_t2048(_bf16w)`, `llm_blkpre_t2048_bf16w`, `llm_dec_b32_l2048_bf16w` and the H100 bf16w rows move at the
  next cross-arch round (measured here on single sessions against the same box's arms, not the round's protocol).
- `docs/tailored-fixes.md`: nothing (no tailored fix).
- `cross-arch/README.md` 5fcb401b finding 2's "40 LDG and 0 LDGSTS" is the cold loop's (this file §1).
