# sweep-attention — open problems in attention / softmax / long rows / decode / prefill (2026-11-24)

Branch `sprint/sweep-attention` from `local-rtx-5080` @ `a31cd014` (the integrated head). RTX 5080 (sm_120, 84 SMs,
99 KiB opt-in smem, 64 MiB L2), CUDA 13, clang/LLVM 23. Commits: `fc7e5907` (`unit-uniform`: attention inside a
head-group loop streams, plus three fact fixes the generator found), `fbd3f050` (the streamed kernel declares its
residency). Raw output (ABBA results, box logs, bundle evidence, test logs, the emission diffs):
`.claude/raw-evidence/local-rtx-5080/sweep-attention/` (not in git).

**Result.** The 8192-token prefill (`llm_pre_t8192`, the area's costliest row: 214.8 ms lost over the five parts) went
from **90.52 to 48.89 ms** on the 5080 (ABBA medians, same session). Against the fastest competitor (Liger, 52.91 ms)
the ratio goes from **1.69× to 0.92×**. Before, its attention ran in three kernels per KV-head group: QKᵀ, the softmax
flow and P·V, with the scores and probabilities written to DRAM (1 GiB each a group). Now each group's attention is
one streamed kernel, because a tile whose operands a loop counter moves can now join a stream unit. The other
programs only change in `; mapal.` comments (the refusal reasons): no other suite or model-scale program changes
its code.

## Open items in this area, ranked by cost

The cost is the time lost to the fastest competitor, summed over the five parts. Sources: `matrix.md`, this area's
rows; the `Open` sections of `flash-attention.md`, `long-rows.md`, `long-prefill.md`, `half-precision.md`,
`integration.md`; the open items of `cross-arch/README.md` §3a8e654f.

| # | item | rows | Σ lost (ms) | status |
| --- | --- | --- | ---: | --- |
| 1 | T8192 attention inside the head-group loop not streamed (flash-attention open 3, "T8192's group loop") | `llm_pre_t8192` | 214.8 | **closed generic** (`unit-uniform`) |
| 1a | found by the new generated loops: a consumer with `d = T` taken for the producer; a one-head trip's epilogue / row operand unproven by `Reach`; a carried sum after the consumer | generated programs | — | **closed generic** (facts) |
| 1b | found by the T8192 evidence pass: the stream kernel spilled 8 B (ptxas capped it at 128 registers) | every streamed program | — | **closed generic** (`min-blocks` on the stream kernel) |
| 2 | bf16-weight prefill/block prefill: f32 activations × bf16 weights stay on FFMA | `llm_blkpre_t2048_bf16w`, `llm_pre_t2048_bf16w` (3.7–14.5×) | 179.9 | attempted-open (below) |
| 3 | `: bf16 \| f32` prefill attention streams on FFMA (S6 `stream-mma`) | `llm_blkpre_t2048_bf16`, `llm_pre_t2048_bf16` | 52.2 | open (below) |
| 4 | TF32 prefill attention on FFMA (S6) | `llm_blkpre_t2048_tf32`, `llm_pre_t2048_tf32` | 36.4 | open: the 5080's TF32 rate equals its FFMA rate |
| 5 | decode attention over the cache (S4: narrow windows, a captured position in the mask) | 21 decode rows bf16kv / bf16 / bf16w | 53.9 (ms a step) | attempted-open (below) |
| 6 | the tune's pick varies between arms and sessions (integration open) | `llm_pre_t8192` QKV | 3.0 here | open (seen again, below) |
| 7 | small programs: lsm (3 launches), `attn_1024_64` (split under-priced: flash-attention open 1) | 6 rows | < 1 | open: µs launch latency (integration's ~1 µs runtime residual) |

## 1. `unit-uniform`: attention in a head-group loop streams (closed generic)

The long prefill loops over the KV heads (`g < 8`). Each trip's QKᵀ reads `q[i·4096 + (g·4 + hh)·128 + u]` and
`kr[r·1024 + g·128 + u]`, and its P·V reads `qkv[r·6144 + 5120 + g·128 + u]`. These are mixed-radix tile sites whose
operand address a captured number moves (`RadixTile::ua` / `ub`, long-prefill `uniform-base`). `unit_plan`'s extended
pass took no such site, so the stream fact never saw the group.

| what | where | does |
| --- | --- | --- |
| fact arm `FactArms::unit_uniform` | `mapal-ir/src/unit.rs` | the extended pass takes a radix tile whose operand a captured number moves; `--gpu-off=unit-uniform` (`--no-…` none) |
| `TilePost::after` | `mapal-ir/src/algo.rs` `tile_post` | the numbers the work after the fold reads (`feeds_only` the fold's source otherwise); `Debug` prints it only where it is not every number, so kernel ids are unchanged |
| `GMove` | `func/flow.rs` | a unit's `Gmem` entry carries its captured terms; the launch adds `uniform_at` of them to the device pointer (the tile launch's existing move); `moved=` on the flow line, `unit-uniform` in its `rungs_on` |
| stream rule | `func/stream.rs` `stream_choice` | a producer's moved operands, a consumer's moved column operand; an epilogue whose `after` is empty gets its unread numbers as zero |
| intent `unit-uniform` | `intent.py`, `evidence.py` (`flow.n_moved`) | ON: the moved operand belongs to a streamed unit, its windows arrive by LDGSTS, its axis-wide members are dissolved. OFF: no moved operand in a unit |

**Found by the generated loops and fixed generically** (`testgen::FlashProg::lp`, `flash_loop_src`):

- **Role order.** When the head width equals the axis (`d = T`), the consumer P·V was taken for the producer
  (`refused: producer`). A tile that folds the axis over an in-group axis-wide row is now the consumer.
- **Rows `Reach` cannot prove.** In a one-head trip the head digit `c ÷ d` is always 0, and `Reach` cannot see it. The
  epilogue's per-row read and the plain row operand are now proved instead by their digit forms
  (`unit.rs post_rows`, shared with F1's permuted check) and by the tile's own read
  (`a[base + ci·i + ck·k]`, `ci` = the row width). The backend's `unit_of` accepts the fact's proof for extended
  groups.
- **Trailing members.** A carried sum after the consumer (`y + attg`) has the axis width when `d = T`, so the group
  was refused. `StreamGroup::roles` now covers the members up to the last consumer. The ones after it, which read
  only the consumer's output, run as their own launches and read its stored rows.

**Intended vs realized** (`b64xkb32:rt4g16t256:cp`, `moved=q;kr;qkv`, the bundle's `evidence.tsv`):

- **Instruction mix:** FFMA in both products (608 static), LDGSTS for the windows (40), MUFU.EX2 for the rescale
  (20), 7 BAR.
- **Stores and moves:** `st.global` only for `attg` (32); `s` and `pe` are dissolved; the three pointers are moved by
  the host at the launch.
- **Resources:** 168 registers, 0 spill (after 1b), 76.3 KB shared memory, one block an SM.
- **Checks:** every `stream` and `unit-uniform` check is `ok`. The remaining mismatches are in the `.xref` reference
  build (never timed) and the standing `kmajor-a` class, on both arms.

### Measurements

RTX 5080, changed-programs ABBA (`gpu_changed.sh --off unit-uniform --stems pt_llm_pre_t8192`, one `flock` hold,
`PROBE_REPS=9`, arm order reversed every other repetition). Conditions: `taskset 8-15,40-47`, Tctl 50 °C, GPU idle
before. Value gate: 0 failed (the device exact face as the reference, `XREF`). The competitor is the matrix's fastest
same-precision leg (competitors do not depend on Mapal's code).

| workload | OFF `unit-uniform` ms | ON ms | null | fastest competitor (leg) | Mapal/best before → now | explanation (SASS / trace) |
| --- | ---: | ---: | ---: | --- | --- | --- |
| llm_pre_t8192 (f32) | 90.52 (86.1–99.7) | **48.89** (46.6–49.3) | 46.84 | 52.91 (liger) | 1.69 → **0.92** | per group: OFF QKᵀ 2.18–2.38 + softmax flow 2.69–2.87 + P·V 3.20–3.44 ms; ON one stream kernel 3.67–3.81 ms (model 3.03: within 1.23×) — 8 groups, 66 → 30 ms; O 0.99 → 0.92; QKV 13.89 vs 10.87 ms is a different tuned pick between the arms (item 6), not this change |
| llm_pre_t8192, rules pick / one candidate | — | 54.42 / 52.40 | — | — | — | the projections' table picks (the stream has no table) |

**Replay on the other parts** (the rule over each part's properties; no stream arm was measured there): on the 4090
the stream is chosen (priced 2.03 ms a group against 7.57 ms for the split). On the 3090, A100 and H100 it is
`stream=s:unpriced`: their profiles have no plan rates. That is flash-attention open 5 ("the rented round with
`calib.cu bar`"). Their prefill keeps the split realization until a rented round measures those properties.

### Tests

- **`flash_attention.rs` `stream_on_generated_attention`.** 128 generated programs, 39 of them head-group loops (the
  T8192 form at 16–64 tokens, 2–8 heads in groups of 1–4, causal or not, mask `−1e9` / `−8`). 31 loops stream with
  moved operands; 12 groups have trailing members, 4 of them streamed; every loop is also built with `unit-uniform`
  OFF. 731 emulator runs: values against the interpreter (rel 1e-4) in 4 modes. Every counter is > 0: 90 streamed
  (was 68), 65 model-scale (was 45), 38 causal. The 8 loops that do not stream are one-head trips at 16 tokens that
  run entirely on the host (no device site).
- **Suites.** `mapal-ir` + `mapal-rewrite`: 349 passed. `mapal-backend-llvm` full suite on the final code
  (`fbd3f050`): 471 passed, 0 failed.
- **Emission against `a31cd014`, all 306 programs of `gen_tiles.py`.** Only `pt_llm_pre_t8192` changes code under
  `unit-uniform`. 45 programs differ only in `; mapal.` comments, where the stream fact now names a refusal (decode
  `row_block=s:footprint`; the matmul chains `stream=c:cost`). After 1b, the 12 T2048 stream programs differ by the
  stream kernel's `nvvm.minctasm`.

## 1b. The streamed kernel's residency (closed generic)

The T8192 evidence pass found the stream kernel at 128 registers with an 8-byte spill. ptxas does not know the
launch's dynamic shared memory (76 KB, one block an SM), so it capped the registers to allow a second block that never
fits. Tile kernels already hold ptxas to their derived residency (`min-blocks`, `.minnctapersm`); the stream kernel
did not. Now `Stream::per_sm` (the residency `stream_ns` already priced: threads, shared memory and the model's
registers) gives the kernel's `nvvm.minctasm`, under the same `min-blocks` rung. Effect: T8192 168 registers and no
spill; T2048 159 → 167 registers, no spill either way.

ABBA against the integrated head (`--base-emit a31cd014`, 8 stream programs, `PROBE_REPS=9`). The stream kernel
itself went **1.914 → 1.897 ms** (traced, T2048). The program medians moved 0.95–1.00×, but that comes from the
projections' tuned picks: those kernels' text is identical between the arms. Null variation was 0.0–2.4 %.

| workload | base ms | ON ms | null | stream kernel base → ON |
| --- | ---: | ---: | ---: | --- |
| llm_pre_t2048 | 7.157 | 6.861 | 7.026 | 1.914 → 1.897 |
| llm_pre_t2048_tf32 | 5.862 | 5.846 | 5.718 | 1.911 → 1.895 |
| llm_pre_t2048_bf16 / _bf16w | 4.123 / 7.228 | 4.114 / 6.909 | 4.130 / 6.910 | — / 1.915 → 1.894 |
| llm_blkpre_t2048 / _tf32 / _bf16 / _bf16w | 27.32 / 20.83 / 12.21 / 28.24 | 26.66 / 20.85 / 12.18 / 26.94 | 26.87 / 20.85 / 12.17 / 26.96 | 1.916 → 1.901 (f32) |

## 2. bf16-weight prefill (attempted-open)

`_bf16w` keeps f32 activations against bf16 weights; its legs are bf16 throughout. On every part the projections run
on FFMA: in the A100 emission, `narrow=b:bf16` with an 8 × 16 FFMA tile. The split realization for this pair is
already in the facts: `mapal_ir::split::SplitKind::Bf16x3n` (three exact bf16 parts of the f32 operand, three exact
products) has its bound. But `split_operands` returns `None` for an f32 × narrow pair ("bf16x3n: not emitted").

The rate property decides it per part (`split_rate_pays`: `matrix ≥ 1.3·P·ffma`, P = 3):

- **5080:** 512 < 998 flops/clk/SM. Refused, so nothing on this box would time the kernel.
- **A100:** 1024 ≥ 499. Would take it.
- **H100:** the `mma.sync` rate would take it.
- **GeForce 3090 / 4090:** refused, as the 5080.

The missing piece is the emission: f32 `a` fragments from the TF32 tile's 16-byte panels, cut into three bf16 parts in
registers, against narrow1's `ldmatrix` `b` panel, three `mma.m16n8k16` a k-step, with tf32x3's cold guard. It needs a
rented A100/H100 round to measure and replay. Not built in this step.

## 3–4. Tensor cores inside the streamed kernel (S6, open)

`func/stream.rs produce` / `consume` fold the windows by FFMA in a lane layout (rows `rt`, columns `x + g·z`). A
matrix-instruction path needs the cells and folds in the accumulator's fragment layout, with the row extreme a quad
butterfly and P passed to P·V as the A fragment. That is a second emission of the kernel. On the 5080 the TF32
instruction's rate equals FFMA's (256 flops/clk/SM), so TF32 (item 4) gains nothing here. The bf16 rows (item 3)
would at most halve the 1.9 ms attention: `llm_pre_t2048_bf16` 4.11 → ~3.2 ms, against torch's 1.93. Not attempted in
this step: it is the largest emission change in the area.

## 5. Decode attention (S4, attempted-open)

With `unit-uniform` the decode groups now form (`row_block=s:footprint`, was `extended`). Three gaps keep them on the
split realization:

1. **Captured position in the mask.** The QKᵀ mask `r ≤ p` reads the captured position after the fold
   (`TilePost::after = [p]`, so `post_numbers_ok` refuses). A unit kernel takes no scalar argument.
2. **Batch digit in the window operand.** K's address carries the batch digit (`((b·L + r)·8 + h/4)·128`), so the
   window operand differs per row block. `panel` covers a KV head's query heads, not a sequence.
3. **Narrow windows.** The bf16 cache windows are not staged by the stream yet.

The data movement it would save is the scores and probabilities: 32 MB a step at B32 L8192 f32, against the 2 GiB
f32 cache (1 GiB bf16). That is about 7–15 % of the step, not the decode gap, which is the cache read itself.

## 6. The tune's pick between arms (open, seen again)

`llm_pre_t8192`'s QKV site took two different tuned picks in the two arms of one session: `5dff3a9c` at 10.87 ms (ON)
and the stream-K `cd025245` at 13.89 ms (OFF). The kernel text is identical, and the 3 ms difference is not this
change. The integration's M3 (in-context confirmation) is still the missing step.

## For integration

- **README line (local results):** `llm_pre_t8192` (f32, 5080) 89.55 → **48.89 ms**, 0.92× Liger (was 1.69×): each
  KV-head group's attention is one streamed kernel (`unit-uniform`: a tile whose operand a loop counter moves joins
  the stream unit; the launch moves the pointer).
- **CLAUDE.md facts table, rows to add/extend:**
  - `unit_plan` extended pass: + mixed-radix tiles with captured-number terms (`FactArms::unit_uniform`), epilogue /
    row reads proved by digit forms where `Reach` cannot (`post_rows`).
  - `tile_plan_post` (`TilePost::after`): the numbers the epilogue reads after the fold.
  - `stream_plan` (`StreamGroup::roles` covers the members up to the last consumer; the consumer role ahead of the
    producer's when `d = T`).
- **OFF arms:** `--gpu-off=unit-uniform` (fact arm). The stream kernel's residency is under the existing
  `--gpu-off=min-blocks`.
- **Tailored-fix registry:** nothing (no tailored fix).
