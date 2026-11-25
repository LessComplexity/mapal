# local-rtx-5080 — mask-frontend: the readable model's attention streams; the forms it was written around compile (2026-11-25)

Branch `prefill/mask-frontend` from `16d668c6` (the full model). RTX 5080 (sm_120, 84 SMs), Threadripper shared
(timed processes on CPUs 8-15/40-47, `nice 10`, the bench lock held; load 5–70 during the runs). Raw output (never in
git): `.claude/raw-evidence/local-rtx-5080/prefill-mask-frontend/` (`base*/` `v4/` `off-*/` builds and runs,
`abba_*.txt`, `*/trace_n*.err`, `tests/`, `changed/`).

## What the readable program hit, traced

`examples/llama.mapal`'s attention masks causally through the position array, `(past <= positions[t])`, its scores
token-major (`cell = (t·H + h)·T + past`, the way a person writes `[token][head][key]`). Traced on the 512 build:
every prompt layer's attention was the split realization (scores tile, three softmax launches, `P·V` tile), the
stream group refused `site` — and behind that refusal four separate faults, each fixed generically:

| fault | where | fix |
|---|---|---|
| the epilogue reads numbers after the fold (`÷ scale_root`, the layer counter in `key_row`) and the mask's array (`positions`) — the stream kernel passed `0` for numbers and refused arrays | backend `stream_choice` / `post_call` | **`stream-numbers`**: a streamed unit's epilogue numbers (`TilePost::after`) are launch arguments (`%uv`), its epilogue arrays memory operands |
| forcing the stream gave **wrong values** on the token-major layout (`0.817` for `0.75`): `tile_plan_radix` numbers the scores' tile rows KV-head-major, the group's readers (row max, `exp`, row sum) read output rows — the stream treated tile rows as output rows | `unit_plan` | **`out_rows`**: a mixed-radix seed's output rows as a digit permutation of its tile rows; the group's rows are the output's, the seed's map its `row_of` (the producer's row operand staged at its own rows); a seed writing no row a tile row seeds nothing |
| `positions[t]` is no digit form: no `GuardBox` (and `guard_box` refused any `radix.out` site) | `stream_plan` `guard_box` | **guard terms** (fact arm `guard-terms`): a side through `+`, `·literal` and reads of a captured integer array proven in bounds — an array whose element law is the index's affine image (`pos = map(iota)(x → first + x)`, `first` a literal: `elem_plan`'s law through the capture) reads as that image (the prompt's mask **is** `past ≤ t`); any other read a per-row load (`RowLoad`: a run-time position vector, its least / greatest over a block read once a block); a captured number's digit a per-launch term (`uniform`) — open sides decided in the kernel (`visible_band`, signed) |
| the stream's blocks held 4 rows (the rows sharing a KV panel in output order: `t·32 + kvh·4 + h`), so every block re-read the whole key/value window for 4 queries — slower than the split realization | `unit_plan` / backend | **walk order** (fact arm `walk-order`): `UnitGroup::order` — blocks walk the group's rows in the seed's tile order (one KV head's queries over consecutive tokens); the panel grows 4 → 2048 rows on the prompt (`b` 16) |

Every one generic: no workload name; the facts read the graph (`elem_plan`'s law, `radix_plan`, `bounds_proof`,
`tile_plan_radix`'s address forms), the rule the same stream cost model.

## Results (RTX 5080, batch 1, 128 generated; ABBA, 2 rounds, medians [range])

| prompt | TTFT before → after ms | decode before → after ms/token | prompt kernels (traced) | prompt attention kernels (traced) | prompt launches |
|---|---|---|---|---|---:|
| 128 | 29.9 [29.4–30.8] → 29.6 [29.0–30.8] | 3.465 → 3.542 (spread 3.35–3.73: null) | 14.3–16.6 → 13.3–13.8 | **1.65–1.85 → 0.77–0.93** (48 → 16 launches) | 243 → 211 |
| 512 | 88.3 [86.9–91.6] → **84.9** [83.7–86.2] | 3.393 [3.387–3.408] → 3.401 [3.401–3.406] | 39.8–40.5 → 38.3–39.2 | **5.59–5.81 → 3.42–3.68** | 243 → 211 |
| 2048 | 335.1 [325.4–344.5] → **316.8** [315.3–343.4] | 3.611 → 3.623 (spread 3.53–3.91: null) | 176.5–190.4 → 153.1 (a 191.7 run: host noise) | **55.5–57.6 → 31.2–32.2** | 244 → 212 |

Tokens: 128/128 identical to the `16d668c6` build at 512 (the program now reading its cache through the struct).
Walk order OFF (512): attention 7.60–7.70 ms — the 4-row blocks are slower than the split realization; the walk
order is what makes the stream pay. Decode: the decode step's attention groups are streams by the fact (their mask
a `RowLoad` of the run-time position) but the rule keeps the split realization (`stream=scores:cost:58848/15426` at
512: 32 rows in 8 KV panels) — unchanged, 323 launches a step.

The stream kernels: `kind=flow rows=T·32 gx=T·2 block=128 members=5 rungs_on=flow,row-block,stream,unit-uniform`
(`b` 16 rows, windows of 16–64 keys, FFMA; per layer 0.05 / 0.21 / 1.95 ms at 128 / 512 / 2048 against the split's
0.11 / 0.36 / 3.5 ms).

## What the readable program was written around — now compiled as written

| form | was | fix | test |
|---|---|---|---|
| `attend(query, positions, cache: KeyValueCache, layer)` reading `cache.keys[…]` inside its map bodies (struct fields read through a capture: no `cell_plan` read, the attention on the host — decode 112 ms) | the program passed the arrays | lowering **`flatten_captures`**: a body's struct capture is its leaf fields' captures (`name.field…`), the body rebuilding the struct from them, so a field read forwards to its capture (`proj ∘ pack`); a nested body's field capture through a struct the emitter packed takes the component itself (a guard arm's projection is not forwarded before the arm resolves) | `frontend_forms::a_captured_struct_is_its_fields` (no struct capture, the twin's tile sites and stream group, the same values); the flash generator's struct-cache form |
| `KeyValueCache { … } -> ret.1` in a callee a layer loop over records calls: replay panicked (`reconstruct_temp`, an object with no writer) | bound first | builder `build_pack`: a `Dest::Ret{slot}` pack writes the component `dest_target` minted (it minted a second product and left the first writerless) | `frontend_forms::a_struct_literal_written_to_a_ret_slot` (no writerless temporary; the oracle's values, `--static` on and off) |
| `past <= positions[t]` (the mask through the position array) | `past <= t` gave the same plan (no guard either way: `radix.out`) | above | the generator's token-major forms |

`examples/llama.mapal` now reads the cache through the struct (`attend` takes `cache: KeyValueCache`).

## Tests

- New generator forms (`testgen::flash_tm_src`, `FlashProg::tm`): the token-major scores with the mask `r ≤ i`,
  `pos[i]` with `pos`'s law literal, `pos[i]` with a run-time `first` (row load), `first + i` with it (per-launch
  number); keys and values through a captured struct where `scale == 1`.
- `mapal-rewrite` `stream_plan::stream_facts_on_generated_attention` (96 draws): the mask's classes against every
  cell at the run's values (`GuardEnv`) and `l ≤ token` itself; law masks closed, load / uniform masks open; walk
  orders bijective with panels > 1 — tm digit 5, law 17, load 7, uniform 8, walk orders 28.
- `mapal-backend-llvm` `flash_attention::stream_on_generated_attention`: 128 programs, 1087 runs over the emulator
  modes, every counter > 0 — token-major streamed 38, skipped windows on every token-major form (digit 6, law 9,
  load 5, uniform 4), struct cache streamed 12.
- `mapal-rewrite` `frontend_forms` (2 tests).
- `cargo test --release -p mapal-ir -p mapal-lower -p mapal-interp -p mapal-rewrite`: **705 passed, 0 failed**; the
  backend, CLI and runtime (`test_par.py --bins 2`): **65 binaries, 566 passed, 0 failed** (the doc tests failed
  once on an uncommitted edit made during the run — fixed, re-run clean) (`tests/`).
- Suite: `suite_diff.py --base 16d668c6` — 35 programs + 2 cells × 5 faces, **same = 45**: no suite program's host
  or device text changes (none has a struct capture, a token-major mixed-radix seed or an epilogue number in a
  stream group), so nothing to time (`changed/suite.tsv`). The OFF arms (`--gpu-off=guard-terms,walk-order,
  stream-numbers`) are measurement flags; the 512 build's device text is identical with the arms' plumbing in.

## Open (attempted; why it did not close here)

1. **The causal skip on the model** (`guard:finite`): the stream skips a wholly masked window only where the
   consumer's values are proven finite (F7: `0 · v = 0`); the cache's values are projections of the run's weights —
   no a-priori bound, so every window is computed (≈ half the prompt attention's work: ~1 ms a layer at 2048). The
   generated programs (data proven finite) skip on every mask form. Direction: a run-time F7 — one reduction of the
   consumer's `b` slice a launch (4.5 MB a layer at 2048) setting a device flag the stream's band reads.
2. **The first layer's prompt attention** (only with the arrays-passing `attend`; the struct form streams all 16):
   `topo_order` places layer 0's value-cache write after the scores (Kahn's tie-break by source position: the write's
   capture product carries the call site's position), and `unit_plan` takes outside operands made before the seed
   only. Direction: an order fact that releases a tile site's operand producers first.
3. **The decode step's attention** stays split (the rule prices the 32-row stream at 58.8 µs against 15.4 µs);
   the fact holds (a `RowLoad` mask) — split-KV with the band skip is the next measurement.
4. The prefill's big costs are elsewhere: the projections on the FFMA tile (~115 of 153 ms of kernels at 2048) and
   the host-side cache chain (TTFT − kernels ≈ 45 / 160 ms at 512 / 2048) — other steps.

No tailored fixes.

## For integration

CLAUDE.md facts table (edits to existing rows):

- `unit_plan`: add "the seed's output rows as a digit permutation of its tile rows (`out_rows`; the seed's
  `row_of`) and the group's walk order (`UnitGroup::order`, the seed's tile order: a block one KV head's queries)".
- `stream_plan`: "a causal mask's box classes" → "a guard's sides through `+`, `·literal` and reads of a captured
  integer array (its element law — `elem_plan` through the capture — else a per-row load `RowLoad`) and captured
  numbers' per-launch digits; open sides decided in the kernel (`GuardEnv`)".
- backend row: `stream-numbers` (a streamed unit's epilogue numbers as launch arguments).

DESIGN (`docs/components/lower`): a map / fold body's struct capture is its leaf fields' captures (`flatten_captures`).
