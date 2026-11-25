# local-rtx-5080 — bf16-class: the causal skip, the bf16 KV cache, the stored attention output, the device trap (2026-11-25, night)

Branch `bf16/class` from `bb9bbed7` (the full model after the prefill integration, `full-model.md`). The program is
the readable `examples/llama.mapal`; the bf16 class is `benches/model/variant.py --compute bf16`, which now also
stores the KV cache bf16 (below). Precision classes as in `full-model.md`: **no gain is claimed across classes**.
Raw output (never in git): `.claude/raw-evidence/local-rtx-5080/bf16-bf16-class/` (`mapal/` programs and builds,
`time/` the ABBA runs, `trace/` CUPTI traces, `tools/`, `changed/` the suite emission diff, `test-full.log`).

## Method

RTX 5080 (sm_120, 84 SMs), shared Threadripper (other agents' work on other CPUs; this step on CPUs 8-15,40-47,
`nice 10`, every timed process under the bench lock, paused above Tctl 85 C). Builds `MAPAL_GPU_CALIB=
benches/results/evidence/local-rtx-5080/calib/rtx-5080.calib mapal build --target=native-gpu`. Timing
`benches/model/run_mapal.py` (warmup 2 + 5 reps a process), **ABBA**: before (the integration's builds at
`8bf54fb3`, `prefill-integrate/mapal/F_*`) and after (this branch's head, `bfh_*` / `f32h_*`) interleaved, the order
reversed on the second pass; a cell is the mean of the two passes' medians. Tokens: greedy 128 against
`hf_reference.py`'s HF f32 / HF bf16 eager runs.

## Result (RTX 5080, batch 1, ms)

| bf16 class | TTFT 128 | TTFT 512 | TTFT 2048 | decode 128 | decode 512 | decode 2048 | peak MB 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|
| before (`8bf54fb3`, f32 cache) | 7.13 | 17.84 | 67.56 | 3.785 | 3.645 | 3.696 | 3 498 |
| **after** (`bf16/class`) | 7.17 | **17.25** | **60.45** | **3.66** | **3.45** | **3.43** | **2 954** |
| best bf16 engine (`full-model.md`) | 4.91 gpt-fast | 13.75 vLLM | 45.47 SGLang | 2.965 gpt-fast | 2.979 gpt-fast | 3.164 SGLang | — |
| after ÷ best | 1.46× | 1.25× | 1.33× | 1.23× | 1.16× | 1.08× | |

Passes (before / after, ms): 128 — 7.02, 7.24 / 6.85, 7.49 (null: inside the passes' spread); 512 — 17.80, 17.87
/ 17.30, 17.19; 2048 — 67.66, 67.45 / 60.43, 60.47. Tokens (bf16 class, after): 128/128 vs HF f32 and HF bf16 at
128; 32 / 89 at 512 (before 32 / 67); 66 / 61 at 2048 (before 82 / 61) — inside the spread HF's own two bf16 runs
show (49 / 67 / 57 agree with each other, `full-model.md`).

| f32 class (calibrated) | TTFT 128 | TTFT 512 | TTFT 2048 | decode 2048 |
|---|---:|---:|---:|---:|
| before | 9.77, 11.03 | 33.04, 28.71 | 112.65, 132.36 | 3.600, 4.007 |
| after | 9.73, 9.62 | 28.99, 31.35 | **98.43, 98.35** | 3.472, 3.472 |

The f32 class is unchanged at 128 and 512 (512's reps are bimodal on both arms, 27.5 / 32 ms: the machine, not the
build) and 13–26 % faster at 2048 (the causal skip, below); 128/128 tokens = HF f32 at every length.

## Where the time went (CUPTI, prompt pass: `tools/ktgroup.py`, `ktgaps.py`)

| bf16 class, ms | 128 before → after | 512 before → after | 2048 before → after |
|---|---|---|---|
| prompt kernels | 5.79 → 5.42 | 15.56 → 14.97 | 69.18 → **54.56** |
| attention (stream) | 0.265 → 0.144 | 1.473 → **0.895** | 16.33 → **7.90** |
| query + output projections | 0.678 → 0.629 | 1.911 → 1.886 | 8.16 → 7.02 |
| key + value projections | 0.32 → 0.32 | 0.617 → 0.618 | 2.99 → 1.81 |
| host gaps (> 20 µs) | 1.02 → 1.18 | 2.46 → 1.88 | 2.19 → 2.78 |

f32 class at 2048: prompt kernels 122.63 → 96.14 ms, attention 34.05 → 16.31 ms. The decode step's kernels are
unchanged (512: 3.36 → 3.37 ms, 288 → 292 launches); the decode is the host's (launch-bound, the residency agent's).

## What changed (facts, rules, capabilities) — intent vs realized

### (d) The causal skip over a cache finite only at run time — `finite-check`

*Fact* (`stream_plan`, `StreamGuard::check`): the consumers whose column operand `finite_plan` does not prove finite.
F6 no longer refuses them ("finite"): a skipped (wholly masked) window adds `exp(κ·(c − m))·y = 0·y`, which is `0`
only where `y` is finite — so the kernel's vote after the visible windows (each fold's term at the masked constant its
identity at the final values, the existing F6 check) also reads the skipped windows' column elements (the value cache
rows the block would have read, 16-byte chunks, exponent bits ≠ all ones, strided over the block's threads) and runs
those windows where one is not finite (the oracle's `0·∞ = NaN`). *Rule*: the stream model prices those reads
(`stream_ns`: a checked consumer's skipped-window bytes) — it moved the bf16-cache 2048 geometry from `b16xt32`
(10.5 ms) to `b32xt64` (8.8 ms). OFF: `--gpu-off=finite-check`.
*Intent*: wholly masked windows skipped (≈ half the products at 2048), split-KV given up for the skip, one vote barrier.
*Realized*: every prompt stream kernel `skip=on` (16 / 16 at every length, both classes); PTX of the 2048 kernel: the
check's `ld.global.v4` loop, one more `bar.sync` (7 vs 6); attention 16.33 → 7.90 ms (bf16 class, 2048) and 34.05 →
16.31 ms (f32 class) — the skipped half, less the check's reads.

### (e) The bf16 KV cache — `narrow-window` + the program's precision

*Program* (`variant.py`, the bf16 class; `--cache f32` keeps it f32): `KeyValueCache`'s arrays `[bf16; cache_size]`,
`write_rows` rounds each new row once (`-> to_bf16`), `attend` widens each cached element it reads (`-> widen_f32`,
exact) — the competitors' setup, written in the program, no code path. *Capability* (`stream.rs`): a stream member
whose operand is stored bf16 (the cache; the query once `round-store` stores it) folds on the matrix product with its
windows holding the operand's own bytes (`cp.async` 16-byte chunks of 8 elements, rows `k + 8` / `c + 8` elements
apart: conflict-free fragment loads), the producer's `b` pair one 32-bit shared load, the consumer's two keys
packed, the row operand's fragments packed from bf16; FFMA streams stay f32-only (refused `narrow`). OFF:
`--gpu-off=narrow-window` (no stream).
*Intent*: half the window bytes, no `cvt` of a stored operand. *Realized* (PTX, the 2048 attention kernel):
`cvt.rn.bf16x2.f32` 52 → 4 (the probabilities' packs remain), `ld.shared.u32` 33 → 17 and 32 `ld.shared.u16` (the
values' keys); peak memory at 2048 3 498 → 2 954 MB; the decode's cache reads half the bytes (decode 3.70 → 3.43 ms
at 2048).

### (b) The attention output stored bf16 — `round-post`

*Fact* (`tile_plan_post`, `TilePost::round`): an epilogue may end in the stored rounding — the body's last step a
`Narrow` from f32, the map's array that format (`precision_plan().store`); `tile_out_elem` lets the strided, epilogue
and mixed-radix sites take such a map. A site the rewrite moves from a plain plan to an epilogue one keeps its class,
so `narrow_stored` now stores an epilogue site's array whose every reader is a bf16-admitting product — attention's
`P·V ÷ total` (generic: any GEMM-with-epilogue feeding a GEMM). *Backend*: the stream consumer writes the format its
epilogue returns; the tile epilogue's value 0 is the post body's type. *Realized*: every output projection
`narrow=a:bf16+b:bf16` (before `b:bf16` only); query + output 8.16 → 7.02 ms at 2048, null at 512.

### (c) The embedding lookup on the device — `device-trap` (capability done; the prompt's gather still on the host)

*Fact* (`trap_cell_plan`, `CellSite::traps`): the maps `cell_plan` refuses only for their unproven `Index` reads.
*Capability*: such a map is a cell candidate; its device variant (`<body>_trap`) checks each unproven read — out of
bounds, `@dev_trap_record` takes the least `topo << 8 | kind` into the module's `mapal_dev_trap` word and element 0 is
read; the launch arms the next host checkpoint (`mapal_gpu_trap_arm`), and every print, the clock, a host trap and the
exit first read the armed word (`mapal_rt_checkpoint_hook`) and raise the recorded trap (exit 101, the oracle's
outcome; unarmed: one atomic load). *Realized*: the decode step's embedding runs on the device (one checking cell a
step); **the prompt's stays on the host** — the byte rule drops it (`MAPAL_DEBUG_PLACE`: `embedded host=true`, its ids
not on the device, and the 525 MB embedding not counted resident although the output head's cell uploads it: only
device sites' operands are `there`). A `shared-upload` trial (a nest cell's operands are there for every cell) made
the embedding resident but the gather still loses the byte rule to a host use of its output; that rule is the
residency agent's in-progress `shared-upload`, so it is left to that branch. The prompt's host chain stays 1.2 / 1.9
/ 2.8 ms at 128 / 512 / 2048.

## Tests

- New / extended generated-program tests (all coverage counters non-zero):
  `flash_attention::stream_on_generated_attention` and `stream_mma_on_generated_attention` — each causal program also
  with its values written so `finite_plan` cannot prove them (an overflowing arm never taken; the last key's last
  value `∞`, masked for every earlier token, the oracle's NaN reproduced), every emulator mode, and the
  `finite-check` OFF arm; the mma programs also with a bf16 KV cache (36 / 36 streamed on the matrix product, 23
  skipping; member bound) and `narrow-window` OFF (unstreamed). Counts (mma test): `check:finite unproven` 23,
  `check:masked inf` 23, `check:oracle nan` 19, `check off unskipped` 24, `mma narrow window` 36, runs 722.
  `half_tile::round_post_on_generated_chains` — 48 GEMM chains, every epilogue (÷, + row, · column, none) rounded as
  stored, emulator modes and `round-store` OFF against the weakest member's interpreter.
  `onto_device::device_trap_on_generated_gathers` — 27 gathers through run-time ids (in range / past the table /
  negative), 27 / 27 checking device cells, every emulator mode's outcome the oracle's.
- Targeted: `flash_attention` 6/6, `half_tile` 7/7, `narrow` 4/4, `site_choice` 9/9 (goldens unchanged), `epilogue`,
  `golden_ll`, `tensor_tile` 2/2.
- Full workspace (`cargo test --workspace --release --no-fail-fast`, `RUST_TEST_THREADS=8 MAPAL_PAR=2
  MAPAL_TESTGEN_JOBS=4`, at `d30e68e5`; `test-full.log`): 164 test binaries, **1708 passed, 1 failed, 10 ignored**.
  The failure was `mapal-rewrite` `stream_plan::model_scale_prefill_is_one_stream_group` asserting the old F7-off
  refusal (`why == "finite"`); under `finite-check` the skip stands with the consumer in `StreamGuard::check` — the
  assertion now says so (3/3).

## Suite emission (`benches/local/gpu_changed.sh --base bb9bbed7`)

Every suite program and cell face emits **the same host and device text** at the head as at `bb9bbed7` (35 programs +
2 cells × 5 faces, `changed/suite.tsv`: every `head_hash = on-base_hash`) — no kernel of the suite changes; null by
construction. Only the runtime differs (the checkpoint hook: unarmed, one atomic load a print or clock read), which
`suite_diff` flags for a full-suite run at both runtimes; the bundle step failed in the worktree (its CUTLASS
scratch lives in the main checkout), so that run is open.

## What stays open, and why

1. **The prompt's host chain** (1.2–2.8 ms): the device trap made the gather a device candidate; the byte rule keeps it
   on the host (above) — the residency branch's `shared-upload` plus the host use of `embedded` (the peeled layer 0's
   readers) decide it.
2. **The skip's check reads** — each block re-reads its skipped windows' values (L2): measured, the 2048 attention is
   7.9 ms against ~half of 16.3 with no check. A per-launch flag (one pass over the cache's rows the launch reads)
   would read each row once; it needs a pre-launch kernel in the stream's launch sequence.
3. **At 128 nothing moved**: the 128-token TTFT is the projections (4.3 of 5.4 ms of kernels, at the tile's 77–81 % /
   58–68 % of the pipe, `full-model.md`) and the host chain.
4. The bf16 KV cache moves tokens at 2048 against HF f32 (82 → 66 agree; HF bf16 61 = 61): the class's own rounding.

## For integration

- `local-rtx-5080/README.md` row: `bf16-class` — the causal skip over a run-time-finite cache (`finite-check`), bf16
  stream windows (`narrow-window`, the bf16 KV cache in the bf16 class), the stored attention output (`round-post`),
  the device trap (`device-trap`); link this file.
- CLAUDE.md facts table:
  - `stream_plan` row: add "`StreamGuard::check`: the consumers whose column operand is finite only at run time (the
    skip's vote reads the skipped windows' column elements)".
  - `tile_plan_post` row: add "`TilePost::round`: the epilogue ends in the stored rounding (the map writes the narrow
    format; `precision_plan().store`)".
  - `precision_plan` row, `store`: "an epilogue site's array too (its rounding the epilogue's last step)".
  - new row `trap_cell_plan` (`CellSite::traps`): "the maps `cell_plan` refuses only for unproven `Index` reads — a
    device cell checks each (the trap recorded, raised at the next host checkpoint)".
  - backend row: rungs `finite-check`, `narrow-window`, `device-trap`; runtime `mapal_rt_checkpoint_hook` /
    `mapal_gpu_trap_arm`.
- `docs/tailored-fixes.md`: nothing (no tailored fix).
- Merge note for prefill/residency: `device-trap` adds trap cells to `cell_sites`' candidates; `shared-upload` there
  decides whether the prompt's gather stays on the device.
