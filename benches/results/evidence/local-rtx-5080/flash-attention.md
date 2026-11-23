# flash-attention — attention as one pass over key windows, from facts (2026-11-23)

Branch `local-rtx-5080` @ 3a8e654f + this step. RTX 5080 (sm_120, 84 SMs, 99 KiB opt-in smem, 64 MiB L2), CUDA 13,
clang/LLVM 23. Design: `.claude/sprints/flash-attention/design.md`. Raw output (sweeps, ABBA logs, SASS, calib, the
changed-programs run): `.claude/raw-evidence/local-rtx-5080/flash-attention/` (not in git).

**Result.** The model-scale prefill's attention (`QKᵀ → mask → row max → exp → row sum → P·V → ÷ den`, GQA, causal)
runs as **one kernel** walking the keys in windows — the scores and probabilities never stored, the causal mask's
wholly masked windows never visited — because the facts say it may: no kernel is named "attention". On the 5080 the
2048-token prefill's attention went from 3 kernels / 3.79 ms (CUPTI, design §2) to **1.91 ms**, the layer from 8.60 to
**6.62 ms** (ABBA medians): **0.91× the fastest f32 competitor** (torch SDPA efficient 7.295 ms, the final round's matrix; was 1.17×).

## What landed

| id | where | what it proves / does |
| --- | --- | --- |
| F4 `stream_plan` | `mapal-ir/src/algo/stream.rs` | a row-block group whose tile folds over an in-group row (its `k` = the row's width) is one pass: roles `Produce` (tile over the axis), `Cell` (same-cell maps; each read of an axis fold's value taken by its law's stand-in), `Fold` (row fold over the axis, its state carried), `Consume` (tile folding the axis), `Row`; each fold's laws from `fold_combine`; a `Rescale`'s shift must be the running extreme of the same `x` (`rescale_shift_ok`, the one test `row-slices` now reads too — moved out of `flow.rs`); a `Phase` law, a stored axis-wide value, a misaligned read refuse (`refused` with the reason); dissolved members and their bytes, per-row state bytes |
| F1 `RowMap` | `mapal-ir/src/unit.rs`, `algebra.rs::whole_rows` | a mixed-radix row operand reading **whole rows under a digit permutation** (P·V's `(token, head)` row reads the scores' row `head·T + token`): `ρ(r) = Σ (coef ÷ w)·digit(r)`, its inverse the digits with stride and coefficient exchanged, both checked on every row; an epilogue's read of a per-row member (`÷ den[head·T + token]`) checked at the same permuted row on every row and both end lanes (`radix_plan` forms) |
| F2 `unit_plan` (extended pass) | `unit.rs` | after every plain group, a second pass seeded by the other tile plans (`tile_plan_post`, `tile_plan_radix`; no captured number yet) over the members left: `UnitGroup::extended` (the row-block units take plain groups only), `UnitMember::row_of`, `UnitGroup::panel` (the most group rows over which no tile's column operand moves: a KV head's query heads, checked on every row) |
| F3 | `algebra.rs::fold_combine` | laws over every tile plan: epilogue sites (the epilogue runs on the combined value), narrow sites (an exact widening), mixed-radix sites reading whole rows (read as `ci = k`) — P·V's `mx` rescale law at model scale |
| F5 `GuardBox` | `stream.rs` | a guarded producer's epilogue `(lhs cmp rhs) ? body(fold) : c` with both sides digit sums of the cell index split into row and lane digits: over a box of rows × lanes each side's range decides `Visible` / `Masked` / `Mixed` (exact; undecided = mixed) |
| F6 absorb | `stream.rs::stream_guard` + kernel vote | a wholly masked window may be skipped when no axis-wide value is stored, every axis cell reads its index only as an address, every axis fold is a `+` or a running extreme, and every consumer's column operand is proven finite; the kernel then checks after the visible band that each fold's term at the masked constant is its identity **at the final values** (`step(m_final, c) == m_final`, `exp(…) == ±0`), the block votes, and on failure walks the skipped windows (rotated order) |
| F7 `finite_plan` | `mapal-ir/src/algo/range.rs` | interval transfer: constants, `iota`, widening, `+ − × ÷` (divisor away from 0), squares, `exp` below overflow, guard arms, map bodies over captured ranges, folds iterated ≤ 2¹⁶ trips; the prefill's `qkv` proven finite through RMSNorm's Heron fold |
| P1 `bar_ns` | `profile.rs`, `calib.cu bar` | one block barrier + dependent smem round trip: **16.9 ns** at 256 threads (44.6 cycles at 2.64 GHz; 13.8 / 23.0 ns at 128 / 512); `BAR_CYCLES_RTX_5080 = 41`, the native-gpu calib line `bar_ns=17` |
| R1/R2 `stream` rule | `func/stream.rs::stream_choice` | every geometry (threads 32–256, `b` rows dividing the rows and the panel, windows 16–64 (≥ 16: the rescales stay inside 2γ, design §7), `rt` rows a thread, smem ≤ opt-in, regs ≤ budget, ≥ 2 resident warps a scheduler) priced on the part's properties: the products' issue at `fp32_lanes` beside each cell's / row's instructions, shared-load issue at `lsu_ipc_pm`, wavefronts, the window's barriers (`bar_ns`), L2 window bytes, DRAM bytes by `residence_plan`, a wave's first-load latency (`lat_ns`); against the split realization's own price (`split_ns`); refusal on the host line (`stream=<m>:cost:<s>/<p>`) |
| R3 `split-kv` | `stream.rs::split_combine` | a row block's windows over `s` blocks; partial states to a workspace, a fence and one ticket (`atomicrmw add … acq_rel`) — the row block's last block combines in slice order by the laws (`exp(a(m_z, m))`), no second launch |
| R5 | `flow.rs` kernel start | a skipping unit's row blocks launch in reverse order (a causal mask's heaviest first) |
| emission | `func/stream.rs::stream_body` | `b` rows × `threads`, row groups of `g` lanes (a warp's aligned part): q panel + K / V windows by 16-byte `cp.async` (next window landing while this one computes; rows past the axis zero-filled branch-free), the scores `rt × kb/g` a thread in registers, the producer's epilogue (`_post` body: scale, mask), the cells by their host bodies (stand-ins by law), the window max a butterfly, `α = exp(a(m_old, m_new))` (1 where the extreme stayed; one `exp` a row shared by the laws), P through shared memory, `O = α·O + P·V`; after the last window the true-value rescale, factor laws, seeds, the consumer's epilogue (`÷ den` through a one-cell buffer) and its permuted output address |

Not landed (open below): F8 (`precision_plan.stream_bound` — the `kb ≥ 16` rule keeps §7's bound, the value gates
pass, but the bound is not a stored fact), P2/P3, R4 (`stream-mma`: TF32 / bf16 tensor cores — the TF32 rows stream on
FFMA), R6 (lazy rescale), narrow K/V windows and captured numbers in units (decode, T8192: S4), the H100 wgmma/TMA (S7),
the competitor SASS capture (design §11 step 1: the competitor venv is not on this box any more).

## OFF arms (measurement flags)

`--gpu-off=stream` (the split realization), `--gpu-off=split-kv` (one block a row block), fact arms
`--gpu-off=stream-fact` / `--no-stream`, `--gpu-off=guard-box` / `--no-guard-box` (every window computed),
`--gpu-off=finite` / `--no-finite` (nothing finite, nothing skipped); `--no-exp-rescale` (max a phase ⇒ refused);
forcing `--stream=<b>x<kb>[x<rt>[x<s>]]` (tests, sweeps).

## Tests

| test | programs | checks |
| --- | --- | --- |
| `flash_attention.rs` `stream_on_generated_attention` | 128 drawn (`testgen::FlashProg`: single-head 1–64 rows × 4–256 keys, widths 3–16, scale forms, max / min (rescale down), seeds below / inside the scores, sum seed 0 / 1, stored members, the refused forms; model-scale 16–64 tokens × 2–8 heads in groups of 1–4, causal or not, mask `−1e9` or `−8`) | values vs the interpreter (rel 1e-4) in 4 emulator modes (free, serialized in order / reversed, blocks reversed) on the forced stream, plus OFF arms (`stream`, `stream-fact`, `async`, `guard-box`, `finite`) and split-KV `s = 2 / 4`: **621 runs**; F5 classes vs every cell of **4116 boxes**. Counters (all > 0): 68 fact groups, refusals phase 8 / escape 10 / producer 17 / no-axis 25, laws rescale up 112 / down 24 / factor 21, 63 streamed, 43 model-scale, 43 digit-permuted, 25 panel batches, 19 causal with skip on, 8 with the near mask (the kernel's vote fails, the skipped windows run), 23 split-KV, 31 multi-window, 6 tails, 19 odd widths, 5 one-row blocks, 23 sync windows |
| `flash_attention.rs` `stream_fact_on_attention`, `streamed_attention_values` | `pt_attn_1024_64`, `px_attn_64_16` | roles, laws, dissolved members and bytes, stored; fact arms off; the streamed unit's emission and values in every mode |
| `stream_plan.rs` (mapal-rewrite) | `px_llm_pre_t12`, a norm chain, 48 drawn | the model-scale group (roles, F1 permutation inverse on every row, panel 8, F3 law on the epilogue site, F5/F6/F7), each fact arm off; F7 ranges contain the interpreter's values, refuses `x ÷ x'`, `exp(100/sd)`; on 30 model-scale programs every `RowMap` a bijection (20), every mask's classes vs every cell (8) |
| suites | `mapal-ir`, `mapal-rewrite`, `mapal-backend-llvm` | all pass; `row_block.rs` measures the row-block rule with `stream` off (the stream takes those groups first) |

## Measurements (RTX 5080, bench protocol: lock, load < 3, no game, `taskset 16-31,48-63`, Tctl 35–57 °C)

Changed programs (`gpu_changed.sh --off stream`): 44 suite programs and both cells **same**, `pt_attn_1024_64`
differs only by its host line's comment (`stream=s:cost:…`): the same three kernels on both arms (ABBA 0.0386 /
0.0389 ms, value gate pass) — null by construction. The full suite (`gpu_bundle.sh` + `gpu_run.sh`): below.

Model-scale rows — ABBA, 11 repetitions each arm, the arm order reversed every other repetition, values equal across
arms (rel ≤ 1e-6). Competitor: the matrix's fastest same-precision leg on this box (`matrix.md`, the 3a8e654f round, every leg re-timed
there; competitors do not depend on Mapal's code, so not re-timed here: the torch venv is not on this box).

| workload | OFF `stream` ms | ON ms | streamed kernel | fastest competitor (leg) | Mapal/best before → now | explanation |
| --- | ---: | ---: | --- | --- | --- | --- |
| llm_pre_t2048 (f32) | 8.599 | **6.622** | 1.914 ms, `b64xkb32:rt4g16t256:cp`, skip on | 7.295 (torch-sdpa-efficient) | 1.17 → **0.91** | 3 kernels (QKᵀ 1.32 over all T² cells, softmax flow 1.30, P·V 1.17) → 1: scores / `pe` (512 MB each) never in DRAM, half the windows never visited (causal band), 0 spill, 159 regs, 75 KB smem |
| llm_pre_t2048_tf32 | 7.502 | **5.506** | 1.912 ms (FFMA) | 4.607 (triton-tf32) | 1.62 → **1.20** | the attention streams on FFMA (no `stream-mma` yet); the rest is the TF32 projections |
| llm_blkpre_t2048 (f32) | 27.28 | **25.31** | 1.913 ms | 26.97 (liger) | 1.01 → **0.94** | same attention; the block's projections dominate |
| llm_blkpre_t2048_tf32 | 21.88 | **19.89** | 1.914 ms | 18.18 (triton-tf32) | 1.20 → **1.09** | as `_tf32` |
| llm_pre_t2048, `guard-box` off | 8.599 | 9.377 | 4.688 ms, `b128xkb16:rt4g8t256:cp:s8` | — | — | every window computed: the model prices it below the split (2.9 vs 4.2 ms), measured above it — a mis-ranked OFF arm (open 2) |
| attn_1024_64 (rule: split, cost) | 0.0412 | 0.0412 | forced `s8`: 0.0349 (−15 %) | 0.0287 (torch-compile-cg) | 1.33 → 1.33 (forced 1.22) | the model prices the split at 16.9 µs against 41 measured (open 1) |
| attn_2048_64 / 2048_128 / 1024_128 (split, cost) | 0.0796 / 0.118 / 0.050 | same | forced best 0.076 / 0.222 / 0.075 | 0.0696 / 0.0952 / — | unchanged | L2-resident scores: the split pays no DRAM; small-`d` tiles issue-bound; refused correctly at 128, even at 2048 × 64 |

Geometry (2048-token prefill, kernel ms, `MAPAL_GPU_TRACE`): `b32xkb64:rt4g32` 1.84, **`b64xkb32:rt4g16` 1.91 (rule)**,
`b64xkb16:rt4g16` 2.33, `b32xkb32:rt2g16` 2.79, a 32-thread `rt2g4` 5.30 (2 warps an SM: now unpriced by the
latency-hiding floor). The model ranks the top two within 4 %: its shared-load issue term at the measured
`lsu_ipc_pm` (0.5 warp-instructions a clock) is what separates them (`b32xkb32:rt2g16` issues 3.25 LSU cycles a cell
against 2.5).

Intended vs realized, the rule's streamed prefill kernel `b64xkb32:rt4g16t256` (`ptxas -v`, `cuobjdump -sass`, the
`intent.py` `stream` checks): FFMA in both products (606 static), no `st.global` but the outputs (32 STG), `s` and `pe`
dissolved; LDGSTS (32) for the windows, SHFL (32: the window max's butterflies), MUFU.EX2 (54: the cells' and the
rescale's `exp`), BAR (7 static), 159 registers, 0 spill, 75 KB shared memory (one block an SM, 8 warps).

Full suite (safety net, `gpu_bundle.sh` + `gpu_run.sh`, 35 programs + 2 cells): box_run rc 0, value gate 0 failed,
12402 intent checks / 226 mismatches (the suite's standing ones; no suite program streams, so the `stream` intent
applies to none), `attn_1024_64` 0.0421 ms, `softmax_rows_4096` 3.80 ms (`full-suite/`).

## Open

1. **The split's price on L2-resident attention** (`split_ns`, the plan model's tiles + flow segments): 18.6 µs priced
   for `pt_attn_1024_64` against 41 measured, so the rule keeps the split where the forced stream (`s = 8`) runs 15 %
   faster. Tried: the same first-load latency a launch on both sides (`lat_ns` a split launch, a stream wave) — the
   split rose 16.9 → 18.6 µs, not past the stream's 19.3; the stream side is within 1.1–1.7× of measured, the split
   side 2.2× (the plan model's small-tile `site_ns`, shared by every tile decision: kernel-merging.md's 1.4–1.9×
   optimism) — the missing fact is there, not in this rule.
2. **Unskipped streams** (`guard-box` off, any unmasked full square at `d = 128`): priced 2.9 ms, measured 4.7 —
   small-`kb` / many-block geometries' per-block costs; the default (skip on) is measured right.
3. **S4 decode** (`_bf16kv`, f32 / bf16w decodes): the P·V site reads a narrow cache through a captured position
   (`ua` / `ub`): extended groups take no captured number and no narrow element yet (`refused: no-axis`); T8192's
   group loop likewise (uniform-base sites).
4. **S6 tensor cores** (`stream-mma`, P2 / P3): the TF32 and `: bf16 | f32` prefills stream on FFMA; Triton TF32
   leads by 1.20×.
5. **F8** as a stored fact; R6 lazy rescale; S7 (H100 wgmma + one TMA box a window); the rented round with
   `calib.cu bar`; competitor SASS (needs the torch venv back).

Proposed CLAUDE.md facts rows (for the maintainer): `stream_plan` (`StreamGroup`: roles, laws, `rescale_shift_ok`,
dissolved, `guard`), `unit_plan` extended groups (`row_of` / `RowMap`, `panel`), `GuardBox::classify`, `finite_plan`.
