# kernel-merging — compile-time placement of GPU work (2026-11-20, rtx-5080 + Threadripper 9970X)

User decision: no runtime graph recording. The compiler decides at compile time how GPU work goes together, by calls
(vertical: a producer and its consumers, a chain of dependent kernels) and by data (horizontal: more rows a kernel,
independent sites in one launch). Kernel boundaries stay a result of dataflow facts plus `TargetProfile`. Each step
of the sprint appends its section here. Raw output (box logs, results txt, CUPTI traces, ABBA) is kept locally in
`.claude/raw-evidence/local-rtx-5080/kernel-merging/` (not in git).

## Groundwork

Code: `0e8adcdf` (graph removal, `access_plan`, `residence_plan`, `zero-fold` v1) and the follow-up commit beside this
section (16-B tails, trial-aware skip, measured fill / tail rates).

### What changed

| | fact / rule | OFF arm |
| --- | --- | --- |
| 1 | Runtime CUDA-graph replay removed: launch regions and tags (`func/region.rs`), the runtime's graph record / instantiate / patch / launch, the emulator's recording and shuffled replay, `graph_launch_ns` / `set_params_ns`, calib `graph`, `--no-launch-graph`, `MAPAL_GPU_GRAPH`, `launch_graph.rs`. The dependency analysis is kept as an IR fact: **`access_plan`** (mapal-ir, stored in `FuncFacts`, CLAUDE.md row) — per morphism the bytes it reads (through `Pair` / `Proj` of products) and writes; storage classes (a carried slot's views, next value and exit; an `Update` with its array); RAW / WAR / WAW edges of every loop trip; `edges()` lifts it onto any grouped op sequence with a backend's extra byte sharing (arena offsets). | — (removal) |
| 2 | **`residence_plan`** (mapal-ir, stored, CLAUDE.md row): per array its bytes and working set — the peak live bytes from its producer to its death (`last_use_plan`), storage classes once, index domains (`Iota`) none; `at(o, cache)` → `Chip` / `L2` / `Dram`. The row-block plan model (`block_cost`) prices every byte stream at the rate of where it lives (`l2_bytes_per_ns` / `dram_bytes_per_ns`) instead of the part-wide `bytes_per_ns`. | the old rate: `bytes_per_ns` (the rule only runs under `--gpu-on=row-block`, which every part ships off) |
| 3 | **`zero-fold`** (`func/zero.rs`, host rung): an atomic site's zero pass is the grid-stride tail (16-B stores) of the nearest earlier cell or unit kernel that the dependency fact orders after every earlier access to the output's bytes (class, residency alias, carry, arena overlap; a gated access keeps the zero), when the plan rates price the tail (bytes at the measured tail rate, few blocks capped by their own issue) below the zero op (`zero_ns` + bytes at the measured fill rate + the idle it leaves). The site launches flagged `MAPAL_SITE_ZEROED`; the runtime skips a candidate's zero unless a trial ran in that call; the direct atomic launch drops its zero. A part without the measured fill / tail rates folds nothing (`unpriced`): only the rtx-5080 has them. Host intent `; mapal.zero.host sites= folded= kernels= gated= stopped= none=`, `; mapal.zero sites= bytes=` at the folding launch. | `--gpu-off=zero-fold` |

### 1 — graph replay removed, emission unchanged

With `--gpu-off=zero-fold` the emission equals `49430fc8 --no-launch-graph` byte for byte: 5258 host/device
modules, 551 programs (191 `benches/algorithms` + 360 generated: 40 each of device / chain / wide-chain / twin /
twin-number / recur / radix / geom / tile, `tests/testgen_dump.rs`) × rtx-5080 / rtx-4090 / h100-pcie × default /
exact face (rechecked on the final build: 5258 / 5258 identical). With the rung on, 20 program × part builds changed at
`0e8adcdf` (rtx-5080 / rtx-4090: decode, attention × 4, rg / rt mlp2 / indep 2048, px mlp2 128); on the final build 8,
all rtx-5080 (decode × 2, attention × 4, px rg / rt mlp2 128): the 16 MiB zeros keep their op and the parts without the
measured rates fold nothing (their emission is the OFF arm's but for the intent comment).

### 2 — residence: attention priced by one rule on three parts (`--gpu-on=row-block`)

`(fused, split)` ns of the model's cheapest `b` (`row_block=s:cost:b<b>:<f>/<s>`; formed units print `ns_fused` /
`ns_split`). Before = the same build with the split rows at `bytes_per_ns` (490 / 494 / 2553 B/ns).

| program | rtx-4090 before | after | rtx-5090 before | after | rtx-5080 before | after |
| --- | --- | --- | --- | --- | --- | --- |
| pt_attn_1024_64 | **formed** b16 19.0 / 39.8 us | refused 18.8 / 17.0 | **formed** b16 16.1 / 34.1 | refused 15.9 / 10.6 | refused 18.9 / 18.0 | refused 18.6 / 15.9 |
| pt_attn_1024_128 | **formed** b16 37.3 / 43.3 | refused 36.9 / 20.4 | **formed** b16 31.7 / 40.1 | refused 31.5 / 16.6 | refused 35.5 / 26.0 | refused 35.1 / 23.8 |
| pt_attn_2048_64 | **formed** b8 92.1 / 143.6 | refused 91.6 / 52.1 | **formed** b8 74.9 / 130.1 | refused 74.7 / 35.9 | refused 138.4 / 63.5 | refused 137.9 / 55.0 |
| pt_attn_2048_128 | refused 183.1 / 154.1 | refused 182.3 / 62.5 | refused 149.3 / 143.5 | refused 148.8 / 49.4 | refused 276.0 / 94.5 | refused 275.1 / 86.0 |
| px_attn_64_16 | formed b16 1.37 / 3.05 | formed 1.37 / 2.97 | formed b32 1.32 / 2.24 | formed 1.32 / 2.15 | formed b64 1.41 / 2.09 | formed 1.41 / 2.08 |

The split softmax's rows (12.6 MB at 1024 × 64) have a working set ≤ 12.9 MB: L2 on all three parts (64 / 72 /
96 MB). Priced at the L2 rate the large attentions keep their boundary everywhere — the order the rtx-5080
measured (forced unit 33.3 us > split kernels 31.7 us, `compile-time-graph.md`); the launch-bound `px_attn_64_16`
forms everywhere (still untimed: open). Row-block stays off by default (`GPU_RUNGS_OFF`).

### 3 — zero-fold: intended vs realized (rtx-5080)

Intended per program from the facts; realized from the host / device text, `ptxas -v`, SASS (`cuobjdump`) and
CUPTI (`benches/local/ktrace.cpp`, one traced process an arm, each bundle's own tuned wisdom; picks traced equal
unless said).

| program | intended (fact + rule) | realized |
| --- | --- | --- |
| decode_1024_256 | `s` (4 KB) and `o` (1 KB) zero in the query cell `q` — the trip's last kernel before them that touches neither (the row `x` reads the last output, whose buffer the next may take: WAR); the retained rows `kk`, `vv` keep theirs (their range is the trip's); 2 memsets a trip fewer, launches unchanged | `; mapal.zero sites=2 bytes=5120` on `q`'s `launchv` (7 arguments: out, t, threads, 2 × pointer + bytes), sites flagged (flags 3); `mapal_tile_e4d66c7fb764e008` 22 regs (OFF `633e6b01` 10), 0 spill, SASS 2 `STG.E.128 RZ` loops + 2 `STG.E.U8` (104 vs 32 instructions); CUPTI: **6146 kernels + 2048 memsets** a run (OFF 6146 + 4096), `cuMemsetD8Async` host time 3064 vs 6136 us, `q` 0.77 vs 0.70 us, device busy 8989 vs 8886 us, traced span 18.17 vs 21.67 ms |
| attn_1024_64 | `P·V`'s output (256 KB) zero in the softmax unit (1024 × 256 threads); no memset | unit `dc2ee5ec822c6992` 26 regs (OFF 27), + 1 `STG.E.128` loop, 7.68 vs 7.68 us; CUPTI **6 kernels + 0 memsets** (OFF 6 + 1) |
| rt_mlp2_2048 | 16 MiB zero: the measured rates price the tail (16 MiB / 1740 B/ns = 9.64 us) above the fill (790 + 16 MiB / 2320 = 8.02 us): kept | not folded: device modules equal the OFF arm's; 54 kernels + 9 memsets both |

Evidence checker (`evidence.py`, `zero-fold` intent): 12 / 12 checks ok in the ON bundle (folded = flagged + direct,
one tail a folding kernel, folded + kept ≤ atomic sites; OFF: no tail). The ON and OFF bundles' mismatch sets
(stems without the extra arms) are identical — the 427 pre-existing, untriaged mismatches (`README.md`).

How the rule got there (each version measured ON / OFF / ON + ABBA, `.claude/raw-evidence/…/groundwork/v1…v3`):

| version | change | decode ABBA | attn | rt_mlp2_2048 ABBA |
| --- | --- | --- | --- | --- |
| v1 (`0e8adcdf`) | 4-B tail stores; the runtime skipped a zero only on a memo hit | 0.872 | the zero still ran (a site run once is never a memo hit) | **1.003** / min 1.002 (null 0.2 %): tail +9.6 us vs memset 8.0 us a call |
| v2 | 16-B stores; the zero skipped unless a trial ran in the call (a trial re-zeroes before each timed run and the pick) | 0.874 | memsets 1 → 0 | **1.002** / 1.001: tail still +9.64 us (1740 B/ns) vs fill 8.02 us |
| v3 | **mixed result → missing fact**: the tail's and the fill's achieved rates (`tail_bytes_per_ns` 1740, `fill_bytes_per_ns` 2320, rtx-5080, `DeviceCost`); a part without them is unpriced | **0.878** | as v2 | not folded: 1.000 / 1.001 |

### Time (ms; v3, ON / OFF / ON suite runs `LADDER=1 gpu_run.sh`; ABBA: 8 rounds = 16 processes an arm, same box)

| workload | OFF | ON | best competitor | ABBA ON/OFF (median, min; null A / B) | explanation |
| --- | ---: | ---: | --- | --- | --- |
| decode_1024_256 | 16.84 | 14.79 / 14.82 | 15.62 hand CUDA (cache) | **0.878, 0.873** (1.8 % / 2.3 %) | host-bound trip (1.6 us a launch issue, 1.5 us a memset issue): two memset issues a trip gone (≈ 3 ms of 1024 trips: `cuMemsetD8Async` 6.1 → 3.1 ms host), the tail adds 0.07 us device time a trip. Mapal now 0.95× the best competitor (was 1.08×) |
| attn_1024_64 | 0.04305 | 0.04081 / 0.04104 | 0.0287 torch-compile | tuned picks differ (QKᵀ site: OFF cand 4, ON cand 0): 0.923, 1.008; picks matched (all cand 0): 0.879, 0.979 (23 % / 11 %) | one memset op fewer (its issue inside the timed region); the unit's kernel time is unchanged. Within null: open (below) |
| rt_mlp2_2048 | 16.30 | 16.32 / 16.30 | 16.0 torch-compile | 1.000, 1.001 (0.4 % / 0.2 %) | not folded (v3): identical device module |
| epi_4096 | 4.012 | 3.584 / 3.578 | 3.75 torch-eager | — | identical emission and module key; the OFF bundle's tuned pick at site `1b4d2ba2` is cand 3, ON's cand 0 (tuning state, not the arm) |
| other 33 rows | — | ON/OFF 0.96–1.03 | — | — | identical emission: cross-run variation (ends: pj_m16_n2048 0.96, convmc_tf32 1.03) |

Probe value gate: 0 programs failed in each of the three runs.

### Tests

| test | result |
| --- | --- |
| `access_plan.rs` (mapal-rewrite) | decode trip: the edges between the makers of x, kk, vv, q, s, o are the dataflow (x→kk, x→vv, kk→s, q→s, vv→o, s→o RAW; x→o WAR through the carried buffer), the carried output's view / next / exit one class; 240 generated loops (twin, twin-number, recur, device): stored edges = brute-force pairwise conflicts, every dataflow read inside a trip is a RAW edge, a RAW edge without a dataflow read goes through a carried class — 3277 RAW (8 through a class) / 1838 WAR / 3 WAW edges, coverage asserted |
| `residence_plan.rs` (mapal-rewrite) | attention: s / ex / p 4 MB each, `s`'s working set holds s + ex + an operand, `p`'s never `s`, L2 at 64 MB and DRAM at 4 MB, a number on chip; 120 generated programs (twin, wide-chain, recur): every array's working set = a naive scan of the live classes over its interval, `at` flips exactly at it (1145 arrays, both sides of 64 KB) |
| `row_block::attention_pricing_is_one_rule_across_parts` | the table above as assertions |
| `zero_fold.rs` (3) | decode / attention decisions, flags, tails, OFF arm text; an emulated decode (16 × 256 rows, 3 trips) = interpreter under 6 picks × free / `MAPAL_EMU_SERIAL=1` / `2`, fewer zero passes ON, trials (fake clock) re-zero; with the tails' stores removed the values are wrong; 120 generated `zero_prog_strategy` programs (cells, carried reads, sites, loop or straight-line) × 4 picks × 3 modes = interpreter and OFF — 145 folds over 169 atomic sites, 37 loop / 26 straight-line programs folded, 45 kernels zeroing several sites, 15 walks stopped by an access, 9 kept; each count required > 0 |
| bugs put back | the walk ignoring accesses: the generated test fails (#56, wrong values: an arena-reused operand zeroed before its reader); trials not re-zeroing: the decode test fails (sums of 9 runs) |
| suites | mapal-ir, llvm backend, cli, mapal-rewrite: 62 binaries, 695 passed (`0e8adcdf` and the final build); `site_choice` goldens pin `zero-fold` off with the other OFF arms |

### Open

1. **The retained rows' zeros stay** (decode `kk`, `vv`: 2 memsets a trip, 2048 a run): their output range is the
   trip's row (`write_delta`) behind the first-trip branch; folding needs that range computed at the earlier kernel.
2. Zero tails ride cells and units only: a tile site's candidates, a row fold or a `devgen` launch have fixed launch
   arguments (`none` in the intent line when no such kernel precedes the site).
3. The fill / tail rates are measured on the rtx-5080 alone, from one program (rt_mlp2_2048's 16 MiB cell tail); other
   parts are unpriced (no fold). A tail on a kernel of few blocks is priced by a model bound
   (`ZERO_BYTES_PER_BLOCK_CLOCK`), not measured at size.
4. attn_1024_64: the one-memset saving is within this row's null (pick-matched spread 23 %); a tighter measurement
   (more processes, CUPTI span only) is open. The two bundles' tuners disagree at the QKᵀ site (cand 4 vs 0) on
   identical tables: per-site-choice's trial noise on a 40-us program (open there).
5. `residence_plan` feeds only the row-block plan model; the `flow` unit gate and `device_pays` still read
   `bytes_per_ns`. `px_attn_64_16` (formed by row-block on all three parts) is untimed.
6. Not in this step: the Calls (vertical) and Horizontal placement itself — they start from `access_plan` and
   `residence_plan`.

## Rows — staged panels in row-block units

Code: the commit beside this section. Raw: `.claude/raw-evidence/local-rtx-5080/kernel-merging/rows/` (CUPTI traces,
ABBA, SASS, `ptxas -v`, emitted modules, the scripts).

### What changed

| # | change | OFF arm |
| --- | --- | --- |
| 1 | **`row-stage` rung** (`func/flow.rs` `Stage`, `stage_of`, `tile_phase_staged`): a row-block unit's tile phase stages its column operand's panels through shared memory, as the separate tile kernels do. A trip folds `kc` of each k-slice's `k`. Kᵀ-like operands (`ck = 1`) are laid out `[slice][column][kc]`, with their 16-byte chunks XOR-swizzled by row so the 16-byte reads of 8 lanes hit 32 banks. V-like operands are laid out `[slice][kc][column]` (consecutive lanes read consecutive words). A row operand read from memory (q) is staged `[slice][row][kc]` and read as a broadcast. The copies are 16 bytes wide (`cp.async.cg` on sm_80+, else `ld` + `st`), with one or two buffers; with two, the next panel lands while this one folds. Column passes fold `1/p` of a thread's columns a pass with panels that much narrower. | `--gpu-off=row-stage`: the unit reads straight from memory, the emission before (byte-identical, below) |
| 2 | **Scratch sharing** (`block_geo`): once a tile is staged, the panels, the split tile's partials and the tree folds' partials share one region. They are never live in the same phase. A split tile meets a barrier before its partials overwrite its last panel, and a double-buffered phase waits for its last copy. | with nothing staged, the layout before |
| 3 | **Pricing** (`block_cost`): a staged tile is priced by the same `site_ns` terms as the separate tile kernels: `bk = kc`, `fly = stages − 1`, per-trip LSU = its LDS reads plus copies (doubled without `cp.async`), and each column pass a `b × c/p` tile of its own. One buffer adds its latency chain to the trip instead of hiding it. The sector term reads the panel's run along `k` (`kc·e` against 32 B). `block_choice` weighs every combination of the tiles' options (at most 256) per `b`, and falls back to reading every tile straight from memory only when no staged combination fits. | the direct pricing, unchanged |
| 4 | `--row-stage=<kc>x<stages>[p<passes>]` measurement override (the `--row-block` precedent) | — |

Byte identity: the new emitter with `row-stage` off equals `f18bd484` on **2292 modules**: 191 programs ×
rtx-5080/4090/h100-pcie × both faces × {default, `--gpu-on=row-block`}. A forced `--row-block=<b>` now prints the
model's `ns_fused`/`ns_split`, where before it printed 0/0.

### Intended vs realized: pt_attn_1024_64, b = 16 (`--gpu-on=row-block --row-block=16`)

| | intended | realized (SASS / `ptxas` / CUPTI) |
| --- | --- | --- |
| q·Kᵀ | Kᵀ and q panels staged by `cp.async`, Kᵀ rows of `kc·4 = 32 B` (one full sector per column per trip), conflict-free 16-B LDS | model pick `T0:kc8x1:kmaj:v:av:cp`; 17 LDGSTS, **0 LDG** in the unit (direct arm: 96 LDG, 0 LDGSTS); 104 LDS.128; adjacent lane pairs copy a column's 32 B (16 full sectors per warp copy) |
| P·V | V panels across columns, two buffers | `T5:kc16x2:krow:v:cp`; 5 `LDGDEPBAR`/`DEPBAR` waits |
| geometry | 1 launch, 64 CTAs × 256, ≤ 99 KB smem | 100 864 B smem (direct 83 968), 147 regs (direct 220), 0 spill, 27 BAR (direct 18) |
| kernel time | model: fused 18.6 us vs split 15.9 us (refused) | **28.4 us** staged (model pick) / **25.7 us** best forced (q·Kᵀ `8x2p2`, P·V direct) / 32.9 us direct / split 12.6 (P·V) + 10.6 (q·Kᵀ) + 7.6 (softmax) + small = 30.8 us |
| phases (best, isolated by a temporary emitter hook, not committed) | — | q·Kᵀ ≈ 10.4 us (direct: ~18), row segments (mx, ex, den, p) ≈ 5.8 us, P·V ≈ 13.2 us |

Per-tile variants (CUPTI, one traced process each): q·Kᵀ staged and P·V direct 27.8 us (`8x1`) / 25.7 (`8x2p2`);
q·Kᵀ direct and P·V staged 33.6; both one buffer of `8` 34.6; q·Kᵀ `4x2` (half-sector) 33.0; at 512 threads a block
24.4. Staging pays on q·Kᵀ (sectors and latency). On P·V it is null to slightly negative (+0.6 us): V is read coalesced
either way, and the phase is bound by LDS issue (16 broadcast LDS.128 of P per 4 k for 16 FFMA a column).

### Time (ABBA, 8 rounds = 16 processes an arm, iteration ms; `abba.out`)

| A | B | A med | B med | B/A med | B/A min | spread A / B |
| --- | --- | --- | --- | --- | --- | --- |
| OFF (split, shipped) | forced unit, staged (model pick) | 0.0434 | 0.0404 | **0.932** | 0.945 | 32 % / 10 % |
| OFF | forced unit, best staged | 0.0427 | 0.0366 | **0.857** | 0.848 | 11 % / 28 % |
| forced unit, direct (row-stage OFF) | forced unit, staged | 0.0437 | 0.0397 | **0.908** | 0.899 | 19 % / 19 % |

torch.compile on this shape: 0.0297 ms (`gpu-analysis.md`). Value gates: every arm prints the OFF values (exact
digits as printed).

### Decision: row-block stays off; `row-stage` ships on (it acts only inside a unit)

When forced, the staged unit beats the split on this part: kernel time 25.7 to 28.4 us against 30.8 us, and
iteration time 0.86 to 0.93 of the split's. The rule does not form it, because the plan model ranks it the other way
(fused 18.6 us against split 15.9 us). With `--gpu-on=row-block`, the rtx-5080 forms no unit on the timed default face
of any suite program (`survey.sh`). The exact face forms decode at b = 1, staged, as at HEAD. Turning the rung on would
therefore change no timed program, so no win can be cited and the `row-block` rows stay. The model's errors here,
measured against model:

| term | model | measured | ratio |
| --- | --- | --- | --- |
| split q·Kᵀ site (cand 0) | 5.56 us | 10.6 | 1.9 |
| split P·V site + zero | 7.26 | 13.4 | 1.85 |
| split softmax segment + launch | 3.06 | 7.6 | 2.5 |
| unit q·Kᵀ staged (`8x1`) | 7.7 | ~10–12 | 1.4 |
| unit P·V | 9.4 | 13.2 | 1.4 |
| unit row segments (mx, ex, den, p) | 0.1 | 5.8 | ~58 |

The site model is about 1.9× optimistic on this part's split sites: `eff_max_pm` / `eff_half` are the rtx-5090 fit,
never refit here. It is only about 1.4× optimistic on the unit's tiles. It prices the unit's row segments at peak
FLOPs, but they are bound by latency and barriers: two warps per scheduler, tree steps, and dependent shared-memory
walks. The model also ranks the staged options wrongly. `8x1` (28.4 us) is priced below `8x2p2` (25.7 us) because
the chain term charges every trip the DRAM latency (`lat_ns` 573 ns), while the panels are L2-resident (L2 latency
328 ns, calib B). These are missing facts, and they are open (below).

`row-stage` has no `GPU_RUNGS_OFF` row. It changes nothing outside a row-block unit, and every part ships row-block
off. Where a unit is forced, it measured positive on the one part timed (ABBA 0.908 against direct).

With the rung on, the 4090's model now forms pt_attn_1024_64 (staged `8x2p2`; 15.5 us against 17.0). The 4090 is
unmeasured and shipped off.

### Tests

- `row_block.rs`, 11 tests:
  - `attention_row_block_values_and_launches`: b = 1, 2, 4, 8 × {staged by register copies, by `cp.async`, direct} ×
    both faces × `MAPAL_EMU_SERIAL` = 0, 1, 2. The stage labels and the `cp.async` calls are asserted per arm.
  - `staged_column_passes_values` (new): a 512-cell row, forced `8x1p2`, `4x2p2`, `8x2p2` and `4x1p2`, by register
    copies and by `cp.async`, × b = 1, 2, 4 × both faces × serial 0, 1, 2.
  - `attn_1024_on_rtx_5080…`: the model's staged pick, a smem bound of 99 KB, and the `row-stage` OFF arm being the
    unit as before (83 968 B, no `cp.async`).
  - `attention_pricing_is_one_rule_across_parts`: the 4090's staged 1024 × 64 unit.
- `row_block_on_generated_chains`, 1000 generated chains (unchanged generators). New arms at the largest b: staged by
  `cp.async`, and `row-stage` off. Every unit was run under serial = 1 and 2. Counters, each required > 0: 608 staged
  tiles, 43 panels along `k`, 565 across columns, 249 by `cp.async`, 249 double-buffered, 501 with a staged row
  operand, 73 staged split tiles. Column passes need 512-cell rows, which the chain generator (u8 extents) does not
  produce, so they are counted (0) but not required.
- Bugs put back:
  - A double-buffered phase not waiting for its last copy fails chains #488 and #837 (`cp`, serial 1 and 2).
  - The one-buffer trip's write-after-read barrier removed fails `staged_column_passes_values`.

### Open

1. **The plan model mis-ranks the unit** (table above). It needs three facts: the rtx-5080's own site efficiency fit
   (sites measure 1.9× the model), the row segments' latency in a unit and in a split kernel (barrier and tree steps
   at the block's warps per scheduler), and the chain at the panels' residence (L2 latency, a `DeviceCost` fact
   beside `lat_ns`). Until then, row-block ships off, and the measured win is reachable only forced.
2. **The unit's tile phases are LDS-bound.** Each thread holds all `b` rows by `nj` columns, so a broadcast row load
   feeds only `nj` FFMA. P·V (`nj = 1`) issues 16 LDS.128 per 16 FFMA a column. A thread tile over rows as well as
   columns (the tile kernels' `tm × tn`) would cut LDS by 2.5× on P·V. The row segments (5.8 us) run at 8 warps per
   SM with 16-lane rows 1024 words apart, a 2-way bank conflict.
3. Column passes come from the forced test only. The chain generator's extents stop at 255 cells.
4. b = 16 keeps 64 of the 84 SMs busy. Attention at 1024 rows has no `b` that fills the part and keeps the K / V
   re-reads (16 MB each at b = 16) low.
