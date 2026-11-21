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

## Horizontal — independent launches side by side in one grid

Two device launches of one sequence (a loop trip's advance cone, the straight-line walk) that the dependency fact
(`access_plan`, with the backend's byte sharing) leaves unordered run as one launch: the grid is the concatenation of
their block ranges, a block dispatches on its index to its member's body (each kernel a device function of its own
block coordinates, inlined). The group launches at its last member; each earlier member's launch is deferred there.
No runtime graph: the merged kernels, their tables and the dispatch are compile-time text.

### What changed

| | fact / rule | OFF arm |
| --- | --- | --- |
| 1 | **`horizontal`** (`func/horizontal.rs`, host rung). Per sequence, greedily in walk order, a launch joins the open group when it is independent of every member (no RAW / WAR / WAW in `access_plan` lifted with residency aliases and carries; `Pair` / `Proj` steps assemble host products and touch no device byte) and the merged grid fits: arguments ≤ `LAUNCHV_MAX`, the block the largest member's (every block a multiple of the warp), shared memory the largest, and at the merged footprint — the largest block and shared memory, each member's own residency re-expressed in blocks of the merged size — no member takes more waves than alone (`occupancy`) and the members' last waves fit one wave (`waves`: the S54 wave rule over `k` grids). Under both the merged grid is never longer than the members back to back, and issues one launch for `k`. Members: cells (not a range cell or a batched write), units, unsplit tile sites (a table's unsplit candidates make the merged table's tuples, rank-sum order, ≤ `--cands`), a retained site's row pass (merged when its trip's row arm ran — a flag the arm sets; otherwise each member launches alone at the group's point). | `--gpu-off=horizontal` |
| 2 | Emission: a member's runtime launch line is captured (its pointer array and table, or its kernel and grid); the last member issues the merged table (`mapal_gpu_site`, `%mapal_cand` rows of merged kernels) or the one merged kernel (`launchv`); a launch the merge cannot take (a copy-out, a split's zero) issues the member's own launch in place. `; mapal.hz kernel= block= smem= members=` names each merged kernel; the device unit composes it: each member kernel once as `@hz.<kernel>(<params>, i32 %hz.bx, i32 %hz.by) alwaysinline` (its `ctaid` reads the arguments; a member of fewer threads than the block runs on its own warps, its `barrier0` counted: `barrier.sync 0, <its threads>`), the merged kernel a compare chain over the block ranges (`maxntid` the block, `minctasm` keeping each member's register cap). A member launched nowhere else leaves only its device function. | — |
| 3 | Order of the plan: `horizontal` runs before the arena, which keeps every operand of a deferred launch alive until its group launches (an arena reuse would otherwise race inside the merged grid); `zero-fold` after it: a group member is no atomic site (its tuples are unsplit), a tail rides a group only at its last member when no member touches the bytes and the merged arguments take the tail's. | — |
| 4 | Emulator: `barrier.sync 0, n` (`emu_barrier_n`, serial modes meet the block's first `n` threads) and `MAPAL_EMU_REVERSE=1` (a grid's blocks last to first: two blocks of one launch that share bytes race visibly). Measurement only: `--hz=force` (the occupancy and wave tests aside). | — |

No new IR fact (independence is `access_plan`'s; occupancy facts are `TargetProfile`'s), so no CLAUDE.md row.

Byte identity: with `--gpu-off=horizontal` the emission equals `66a10923` on all 5258 host / device modules (551
programs: 191 `benches/algorithms` + 360 generated, × rtx-5080 / rtx-4090 / h100-pcie × default / exact face). With the
rung on, 308 modules differ (groups, or a refusal's intent line); on the rtx-5080 suite, five programs form a group
(decode, causal, gen, prefill, rt_mlp2_1024) and two more print refusals only (rt_mlp2_2048 occupancy + waves,
mlp2_2048_tf32 waves).

### Intended vs realized (rtx-5080; `ptxas -v`, `cuobjdump -sass`, CUPTI `ktrace.cpp`, one traced run an arm)

| program | intended (fact + rule) | realized |
| --- | --- | --- |
| decode_1024_256 | trip: `x` (writes what the rows read) · **kk-row ∥ vv-row ∥ q** (both rows read only `x` and the weights, `q` only the counter; 1 + 1 + 1 blocks: one wave) · `s`, `o` (read them). Merged table: 4 tuples (each row's unsplit candidates 0 and 1 × `q`); `q`'s zero tail (s, o) stays on the group's last member | `; mapal.hz.host launches=3 groups=1 members=3 branchy=2 refused=0`; tuple 0 `mapal_tile_16bf…` 3 blocks × 256, **38 regs** (members 38 / 38 / 22), 0 barriers, SASS 513 insts = 2 × 222 + 93 less shared setup, FFMA 128 = 2 × 64, LDG 256 = 2 × 128, uniform dispatch `UISETP` + `BRA.U`, padded warps `@P0 EXIT`. Tuple 3 (32-thread row candidates): 255 regs, `BAR.SYNC 0x0, 0x20` × 4, 1376 B stack (the member alone: 1400). CUPTI: **4100 kernels a run (OFF 6146)**, the merged kernel 4.19 us where kk 4.16 + vv 4.16 + q 0.77 ran back to back; device busy 9.2 vs 14.2 ms |
| causal / gen / prefill | the same three per trip | 7172 / 3596 / 3595 kernels (OFF 9218 / 4620 / 4617); the merged kernel 4.19–4.22 us |
| rt_mlp2_1024 | trip: **a ∥ f** (4096 blocks each, last waves 64 + 64 ≤ 504); `d1 ∥ c2` refused (128-block GEMMs, a wave of 84 × 1: last waves 44 + 44 > 84 — no wave saved) | 54 → 46 kernels; a, f 3.84 + 3.81 us → 7.04 us (8192 blocks, 16 regs) |
| indep_512 | no group: `d`'s store evaluates `g = c + d` (epilogue) so it reads `c` | unchanged |

Evidence checker (`horizontal` / `horizontal-kernel` intents): 20 host and 44 kernel checks ok in the ON bundle. The ON
and OFF bundles' mismatch sets differ by 4 vs 3 `choice` rows (tuning noise on identical modules, both ways) and two
rt_mlp2_1024 rows (`module`, `uniform`): the merged cells' facts lines outlived their kernels — fixed (a member launched
nowhere else takes its facts lines with it). The final build's bundle, its static evidence checked against run 1's
results: those two rows gone, 20 + 44 `horizontal` checks ok, the mismatch set the OFF bundle's but for the `choice`
rows. Its emission equals the measured ON bundle's but for those facts lines and the intent line's member kinds
(re-emitted, compared); its own timed run was not taken (the box was in use: the gate held it).

### Time (ms; suite ON / OFF / ON `LADDER=1 gpu_run.sh`; ABBA: 8 rounds = 16 processes an arm, same box)

| workload | OFF | ON | best competitor | ABBA ON/OFF rules picks (null A / B) | tuned picks | explanation |
| --- | ---: | ---: | --- | --- | --- | --- |
| decode_1024_256 | 14.91 | 10.92 / 9.60 | 15.6 hand CUDA (cache) | **0.658** (1.1 / 2.0 %) | **0.654** (1.9 / 2.7 %) | the two row passes (one block each, latency-bound) and `q` run side by side instead of back to back: −4.9 us device and −2 launches a trip. Mapal 0.62× the best competitor (was 0.95×) |
| causal_1024_256 | 21.21 | 16.77 / 16.88 | 49.75 torch-compile | **0.774** (0.9 / 0.8 %) | **0.791** (1.4 / 0.9 %) | as decode |
| gen_1024_512_256 | 10.68 | 9.29 / 8.53 | 25.06 torch-compile | **0.776** (1.3 / 2.0 %) | 0.735 (13.6 / 13.3 %) | as decode, half the trips at full length |
| prefill_1024_512_256 | 11.02 | 8.89 / 8.90 | 25.42 torch-compile | **0.782** (0.9 / 0.9 %) | **0.807** (1.4 / 1.4 %) | as decode |
| rt_mlp2_1024 | 2.580 | 2.572 / 2.575 | 2.542 torch-compile | 0.998 (0.6 / 0.9 %) | 0.997 (1.8 / 0.6 %) | a ∥ f saves 0.6 us a trip (8 trips): within null |
| other 30 rows | — | ON/OFF 0.90–1.09 | — | — | — | identical modules; the ends are tuning state (fanout_2048 1.035: site `cf5ec6…` picks 5 ON vs 3 OFF; mlp_512 0.97: `3e77a9…` 4 vs 5; epi_4096 1.09 on one run only) |

Tuned picks: each arm trialled once into its own wisdom, then ABBA on it. The OFF rows tune to their atomic split
candidate (`ce70d…` pick 2, a zero pass a row), which the merged tuples do not hold; the merge still wins. A first
tuned ABBA had disturbed phases (decode OFF spread 18 %, causal ON 18.8 ms vs 16.7 untuned); the repeat is the table's.
Probe value gate: 0 programs failed in each of the three suite runs.

Refusals, measured forced (`--hz=force`, rules picks):

| program | group the rule refuses | forced ABBA / kernel time | reading |
| --- | --- | --- | --- |
| rt_mlp2_1024 | d1 ∥ c2 (`waves`) | 0.997 (1.1 / 2.3 %); merged 158.9 us vs 81.6 + 82.1 | the shared tail saves 3 % of the pair's kernel time, under the iteration's null |
| rt_mlp2_2048 | (`occupancy`, `waves`) | 1.000 (0.2 / 0.1 %) | nothing to save |
| indep_512, epilogue and splits off | c ∥ d (`waves` at the plan's 2 blocks / SM; 6 resident by registers) | 0.953 (43 % / 9 % spread); merged 30.3 us vs 2 × 16.0 | the GEMMs are throughput-bound co-resident: the plan's per-SM count is the right capacity |

### Tests

- `horizontal.rs` (3): decode's group (3 members, 2 retained rows, 4 tuples, the counted barrier, the OFF arm without
  any of it) and rt_mlp2_1024's (`a ∥ f`, `waves:1`); an emulated decode (16 × 256, 3 trips): ON = OFF = interpreter
  under 6 picks × free / `MAPAL_EMU_SERIAL=1` / `2` / `MAPAL_EMU_REVERSE=1`, two launches fewer a trip after the
  first; 120 generated `hz_prog_strategy` programs (cells of the counter, GEMV sites over different cells, row sums,
  the retained KV pattern; loop or straight-line) × 4 picks × free / serial 1 / 2 / reverse = interpreter and OFF, a
  program without a group the OFF text. Coverage, each required > 0: 127 groups, 367 members (185 cells, 72 units, 44
  tile sites, 66 retained rows), 64 programs with a group of three or more, 59 loop / 40 straight-line programs, 7
  merged tables of several tuples, 31 members padded to a larger block, 364 runs with fewer launches.
- Unit tests (`func/horizontal.rs`, 6): the rule (small grids, shared last waves, full last waves, occupancy, args),
  merged residency, kernel → device function, the merged kernel's parameters / bounds / ranges, a member launched
  nowhere else, the launch-line parser.
- `zero_fold.rs`: decode's flags now include the merged table; one kernel takes the tail.
- Tests of other rungs that count launches or pin a pre-sprint text (placement, devmap, epilogue, generic-fusion,
  retain-delta, row-block, site-choice goldens, streams, tf32-tile) pin `horizontal` off, as they pin `zero-fold` and
  row-block: their programs' independent sites would otherwise launch as one.
- Suites: workspace, release, 112 binaries, 1377 passed, 0 failed.
- Bugs put back: the dependency check off (members merged across a RAW) fails the decode decision and the generated
  values; the arena not keeping a deferred launch's operands fails the generated values in reverse block order (a
  later member's output takes an earlier member's dead operand; forward block order hides it — hence the reverse
  mode); the barrier not counted deadlocks the serial emulator (decode and generated).

### Open

1. **indep_512 needs the vertical / horizontal trade**: `g = c + d` rides `d`'s store, which orders `d` after `c`.
   Unsplit and without the epilogue the two sites run 0.773× the shipped plan (split-K atomic + epilogue; ABBA, A
   spread 47 %), merged 0.953× of that again (noisy). The epilogue rule and split-K pricing do not see a sibling site
   that could fill the part instead.
2. Merged tuples hold no atomic candidate (a zero must precede every adding block); tuned OFF decode picks one for its
   rows. A tuple with atomic members needs their zeros before the launch (a zero tail in `x`'s cell, or runtime
   zeros a member).
3. The rule's per-member capacity is the plan's (grid-capped) blocks per SM; the true register residency is a ptxas
   fact. Forced merges where it refuses measured within null (above), so it stands.
4. Groups form only among adjacent steps of the walk order; `c1 ∥ c2` in rt_mlp2 (d1 between, reading c1) would need
   the walk reordered.
5. Members are cells, units and tile sites; row folds, `devgen` generators and split launches are not (their launch
   lines are not captured). Measured on the rtx-5080 only; the rule reads only `TargetProfile` occupancy facts, so it
   ships on every part (no `GPU_RUNGS_OFF` row; the 4090 / 5090 / H100 are unmeasured).

## Calls — dependent launches as one persistent kernel

A chain of dependent launches of one sequence (a loop trip's advance cone, the straight-line walk) runs as one
persistent kernel: each launch's body is a device function called over its own blocks by a co-resident grid
(launch block `i` on persistent block `i mod G`), and a grid barrier stands only between the levels the dependency
fact (`access_plan`) orders. The host issues one cooperative launch instead of `k`. No runtime graph: the chain, its
levels, its barriers and the kernel text are compile-time.

**Result on the rtx-5080: the rule refuses every chain of the suite on this part, and the forced persistent kernels
measure slower (ABBA 1.03–1.32×), so the shipped emission equals the OFF arm's code.** (Review fix, `review-fixes.md`
§3: the boundaries alone priced chains as wins on the rtx-4090 / 3090 / a100, untimed; the rule now also prices the
bodies' measured change inside one kernel, `coop_body_ns`, and a part without it forms no chain.) The barrier is cheap (385–553 ns), but a
cooperative launch costs 2044 ns of device time against 650 for a plain dependent launch; the first version also lost
load batching in member bodies inside the grid-stride loop (fixed: each level's first block straight-line).

Code: `47ad30aa`, `fb37a9fb` and the commit beside this section. Raw: `.claude/raw-evidence/local-rtx-5080/
kernel-merging/calls/` (calib runs, suite results, CUPTI traces, ABBA, phase variants, SASS).

### What changed

| | fact / property / rule | OFF arm |
| --- | --- | --- |
| 1 | **Measured properties** (`DeviceCost`, `benches/gpu/calib.cu gsync`, rtx-5080, block 256, five runs): `gsync_ns` 553 (one grid barrier of ≤ 84 blocks: 385–401 at 1–2 blocks, 519–546 at 16–42, 530–557 at 84), `gsync_block_ps` 1890 (each block beyond `sms`, to 504); `dep_launch_ns` 650 (a dependent empty launch, start to start, ≤ 84 blocks: the low mode of a bimodal 644–652 / 842–936), `dep_launch_block_ps` 0; `pdl_launch_ns` 486 (the same as a programmatic dependent launch: 396–599); **`coop_launch_ns` 2044** (the same as a cooperative launch, flat to 504 blocks — CUPTI agrees: decode's gap before the persistent kernel 1559 ns against 272 before a plain one). Device-side tail launch (CDP2, comparison only): 7.6–8.0 us. Other parts: not measured → `unpriced` (no chain). The stopped attempt's calib edit was kept (its barrier is the emitted one) with its comment corrected (the counter's low 31 bits must start at zero; every barrier restores them); PDL, cooperative and tail-launch columns added. | — |
| 2 | **`calls`** (`func/calls.rs`, host rung, after `horizontal`): units = `horizontal`'s output in walk order (a launch or a merged group). A unit's level is one above the highest chain member it conflicts with (`access_plan` RAW / WAR / WAW with residency aliases and carries), else 0. A non-member step touching a member's bytes closes the chain (the chain launches at its last member, earlier members deferred there); a launch whose output the host copies out ends it. A retained row pass (branch arm) only at level 0; an atomic site (rule candidate splits into a zeroed output) at level ≥ 1, its zero pass grid-stride in the kernel one level before, where no member of that level touches the bytes. **Rule:** forms when `coop_launch(G) + (L−1)·gsync(G) ≤ Σ launch(unit grid)` (each separate launch as the runtime issues it: programmatic where `pdl` is on; the device never slower, the host issuing 1 launch for `k`) and no member takes more rounds of the persistent grid than waves alone (`occupancy`), a level's members fit one round (`waves`), ≤ 32 pointers (`args`). `G` = the persistent footprint's wave (`merged_per_sm`: largest block and shared memory, each member's own register residency) capped at the largest level. | `--gpu-off=calls` |
| 3 | **Emission**: the persistent group reuses `horizontal`'s capture (each member's launch deferred; a table member's unsplit candidates — and its atomic ones when its zero is in the kernel — make the tuples); `; mapal.hz kernel= … grid=G members=<k>:gx:gy:b:l<level>[:z<bytes>@<param>]`; one `mapal_gpu_launch_coop` or a table flagged `MAPAL_SITE_COOP` (4). A member that cannot be captured issues every earlier member first (they produce what it reads). The device unit: `@mapal_gsync` (block barrier, `atom.add.release.gpu` of `0x80000000 − (n − 1)` / 1 on the kernel's counter, `ld.acquire.gpu` spin on the flip bit, block barrier); per level the zero passes (16-B stores), then the level's first block straight-line and a grid-stride loop over the rest, a member of fewer threads on its own warps, `barrier.sync 1, B` between two bodies (shared memory reuse; apart from the members' barrier 0); parameters without `noalias` / `readonly` (no `ld.global.nc` of bytes an earlier level wrote). | — |
| 4 | `--calls=force`: the measurement arm (a chain forms whatever its prices; the fit tests stay). | — |
| 5 | **Runtime**: `mapal_gpu_launch_coop` — `cuOccupancyMaxActiveBlocksPerMultiprocessor` of the loaded kernel (its ptxas registers) × SMs caps the grid, `cuLaunchCooperativeKernel` (the driver refuses a grid it cannot co-schedule). Emulator: a cooperative launch runs every block's threads, one block holding a baton between grid barriers (`MAPAL_EMU_GRID`, default 3 blocks; `MAPAL_EMU_REVERSE` runs blocks last to first; a barrier some block never reaches exits 105); barrier ids. | — |
| 6 | **row-block**: the five "unmeasured" `GPU_RUNGS_OFF` rows (4090 / 5090 / 3090 / H100 PCIe / A100) are deleted; the default is the plan model's decision on every part (a part without plan rates is `unpriced`). On the rtx-5080 suite nothing changes (the model refuses every timed group); the 4090 / 5090 now form `pt_attn_1024_64` and `px_attn_64_16` (unmeasured parts: the rule's decision, no row). `horizontal`, `zero-fold`, `row-stage` had no rows; `zero-fold`'s and `calls`' gate is a measured property (a part without it is unpriced), not a part name. | `--gpu-off=row-block` |

No new IR fact (the order is `access_plan`'s; co-residency and the prices are `TargetProfile` properties), so no
CLAUDE.md row.

Byte identity: `--gpu-off=calls,row-block` equals HEAD (`f3a9e792`) on all 1146 host / device module pairs (191
`benches/algorithms` × rtx-5080 / rtx-4090 / h100-pcie × default / exact face). Code lines (intent comments aside)
that `calls` changes: 74 modules, all rtx-5080 (the parts without the measured costs: none). Code that the
`row-block` rule (no rows) changes: 15 modules — 13 rtx-4090 (attention default and exact, pc / pm / chain / decode /
rowsum / softmax exact) and the rtx-5080's exact-face decode ×2 (b = 1, staged: not a timed row).

### Intended vs realized (rtx-5080; bundle `ptxas -v`, `cuobjdump -sass`, host text)

| program | intended (fact + rule) | realized |
| --- | --- | --- |
| decode_1024_256 (forced) | trip: `x` (range cell: launches alone) · **[kk-row ∥ vv-row ∥ q] (level 0) → `s` (level 1, its 4096-B zero at level 0) → `o` (level 2, 1024-B zero at level 1)**: 3 units, 2 barriers, grid 128; 3 launches a trip → 1. Shipped: refused, 3316 vs 1950 ns | `; mapal.calls.host chains=1 formed=1 units=3 barriers=2 zeros=2`; a COOP table of 8 tuples (`i32 5` flags); tuple 0 `mapal_tile_0ae8bbe6…` **40 regs, 0 spill, 2 barriers**; SASS: 2 `ATOMG.E.ADD.STRONG.GPU` (after `MEMBAR.ALL.GPU`: the release), 2 `LDG.E.STRONG.GPU` spins each followed by `CCTL.IVALL` (the acquire), 7 `BAR.SYNC`, 144 FFMA, 288 `LDG.E` (no `LDG.E.CONSTANT`), 2 `REDG.E.ADD.F32` (`s`, `o` atomic), 4 `STG.E.128` (the zeros); PTX: 2 `atom.add.release.gpu`, 0 `ld.global.nc`, 0 `call`. Tuples with the 32-thread row candidates: 255 regs, 3644 B spill (the members' own, as in `horizontal`'s merged tuple) — the runtime's occupancy caps them at 84 blocks |
| causal / gen / prefill (forced) | trip: rows ∥ q → 5 more dependent sites / cells: 6 units, 5 barriers, grid 128 (shipped: refused, 5224 vs 3900 ns) | causal / gen: `mapal_tile_89f20ce4…` 40 regs, 0 spill; SASS 5 `ATOMG` + 5 acquire spins + 5 `CCTL.IVALL`, 34 `BAR.SYNC`, 161 FFMA; prefill: its second chain refused (`cost`: grid 504, 4690 vs 2600 ns; `occupancy`, `waves`) |
| rt_mlp2_1024 (forced) | trip: two dependent GEMM pairs (128 blocks each): 2 units, 1 barrier each (shipped: refused, 2680 vs 1300 ns) | `mapal_tile_566adc79…` 121–128 regs (members 95), 0 spill, 1 `ATOMG` + 1 spin, 8 `BAR.SYNC`, 1024 FFMA |
| indep_512, mlp_512 | — | refused `occupancy` (a member would take more rounds of the merged footprint than waves alone): emission = OFF |
| smlp_4096_256_1024, mlp_2048_tf32, rt_mlp2_2048, firew_rowsum | — | refused `cost` (grid 168–504) |

Evidence checker (`calls` / `calls-kernel` intents, ON bundle): 19 host and 164 kernel checks ok; the one mismatch
(`pt_decode.c1`, `--cands=1`: a planned chain with no cooperative launch) was the direct-atomic bug above, fixed
before the timed bundle.

### Time (ms; suite `LADDER=1 gpu_run.sh`; ABBA: 8 rounds = 16 processes an arm, rules picks, same box)

The shipped build's code equals the OFF arm's on all 35 suite programs (intent lines aside: `fin.out`), so the
shipped column is the OFF run. The forced columns measure what the rule refused: the first persistent kernel (a
grid-stride loop around every body, suite runs ON1 / ON2 against OFF) and the final one (each level's first block
straight-line, ABBA against OFF).

| workload | OFF = shipped | forced, first kernel (ON1 / ON2) | forced, final kernel (ABBA B/A, null A / B) | best competitor | explanation |
| --- | ---: | ---: | ---: | --- | --- |
| decode_1024_256 | 9.619 | 13.01 / 13.00 (ABBA 2.412) | **1.319** (1.5 / 0.4 %) | 15.46 hand CUDA (cache) | CUPTI a trip: persistent kernel 9.60 us (first version 20.73) against 4.22 + 2.27 + 1.70 = 8.19 us of the three kernels; the cooperative launch's gap 1559 ns against 272; phases (variants, `var/`): launch + zeros + 2 barriers alone 2.00 us, `s` + `o` 2.66 us (3.97 alone), rows + q 5.0 us (4.22 alone; 10.8 inside the loop) |
| causal_1024_256 | 16.91 | 20.37 / 20.40 (1.192) | **1.070** (0.9 / 0.6 %) | 50.31 torch-compile | persistent 15.10 us (first 17.15) against 15.17 us of six kernels; + the cooperative launch's 1.3 us and 5 barriers against 5 plain boundaries |
| gen_1024_512_256 | 8.547 | 10.52 / 10.33 (1.235) | **1.112** (1.6 / 0.9 %) | 25.26 torch-compile | as causal (persistent 15.20 us) |
| prefill_1024_512_256 | 8.933 | 11.63 / 10.89 (1.223) | **1.101** (0.8 / 0.8 %) | 25.63 torch-compile | as causal (15.23 us) |
| rt_mlp2_1024 | 2.570 | 3.079 / 2.824 (1.054) | **1.031** (0.4 / 1.4 %) | 2.532 torch-compile | persistent GEMM pair 165.7 us against 81.95 + 79.36 us (128 regs against 95) |
| indep_512 / mlp_512 | 0.04834 / 0.03938 | identical code | — | 0.0365 streams / 0.043 lt | refused `occupancy` (a member would take more rounds of the merged footprint than waves alone); ON/OFF spread is tuning state on identical modules |
| smlp_4096_256_1024 | 0.1672 | identical code | — | 0.1495 torch-compile | refused `cost` (grid 168) |
| other 27 rows | — | identical code | — | — | ON1/OFF 0.84–1.10 on identical modules: cross-run variation (ends: indep_512 0.84, softmax_rows_4096 1.10) |

Probe value gate: 0 programs failed in each of the three suite runs. Evidence checker on the forced bundle: 18 host and
164 kernel `calls` checks ok, its mismatch set the OFF bundle's but for rt_mlp2_1024's `horizontal` spec count (the
persistent kernels were counted as merged ones — fixed in `evidence.py`) and the tables `calls` absorbed.

How the rule got there: as first priced (a plain launch for the persistent kernel's own) it formed decode, causal,
gen, prefill and rt_mlp2_1024 at 1922 / 1950 ns etc.; every one measured slower (above): **a uniform loss is a
missing property** — the cooperative launch's device cost (`coop_launch_ns`, measured at 2044 ns; the CUPTI gap
agrees). With it priced, no chain of the suite forms. Where the runtime launches programmatically (`pdl`; the
rtx-5080 ships it off by a legacy `sm_120` row, open below) a separate boundary is 486 ns and the barrier loses
further.

### Tests

- `calls.rs` (3):
  - `decode_runs_its_trip_as_one_persistent_kernel`: pt_decode_1024_256 on the rtx-5080: shipped, the chain is
    refused by cost (plain and programmatic launches); forced, one chain (3 units, 2 barriers, 2 zero passes, grid
    128; `s` at level 1 with `z4096@2`, `o` at level 2 with `z1024@2`), every tuple a persistent kernel with 2
    `@mapal_gsync` calls, `barrier.sync 1, 256` between bodies and no `noalias` parameter; the table flagged COOP; no
    `zero-fold` tail left; OFF: none of it; the rtx-4090 (unpriced), forced: none of it.
  - `decode_values_through_the_persistent_kernel`: an emulated decode (16 × 256, 3 trips) = interpreter = OFF under
    picks 0 / 1 / 3 / 7 × persistent grids 1 / 2 / 3 × free / `MAPAL_EMU_SERIAL=1` / `2` / `MAPAL_EMU_REVERSE=1`; one
    cooperative launch for three every trip after the first (pick 0: exactly 2 launches fewer a trip).
  - `calls_on_generated_programs` (chains forced: the realized path under test): 48 programs of the new `calls_prog_strategy` (the `horizontal` programs over one to
    eight rows / columns and folds up to 1024 long: unsplit and atomic sites, cells reading sites, row sums, the
    retained KV pattern; loop or straight-line), run in parallel, × picks 0 / 3 × grids 1 / 3 (free), and at grid 3
    serial 1 / 2 and reverse = interpreter and OFF. Coverage, each required > 0 (final build): 48 chains formed in
    43 programs (21 loop / 22 straight-line), 149 units, 93 grid barriers, 39 zero passes in the kernel, 28 programs
    with 3+ levels, 18 merged tables, 20 padded members, 13 retained-row chains, 2 programs with fit refusals, 140 runs
    with fewer launches, 430 emulated runs.
- Unit tests (`func/calls.rs`, 4): the fit (decode's levels → 128; capped at one wave; args, occupancy, waves
  refusals), the prices (decode 3316 vs 1950 ns, 1458 with programmatic launches; 13 one-block launches still refused;
  without the cooperative launch's price the barriers would win; rtx-4090 unpriced), the kernel text (levels, the zero
  loop before the barrier, the straight-line first block and the stride loop, padded member, barrier ids, no
  `noalias`), the flip bit (no early flip for 1–504 blocks over 3 barriers, low bits restored).
- Workspace, release: 103 binaries, 1384 passed, 0 failed (before the pricing follow-ups; `calls.rs` and the backend's
  unit tests re-run after them).
- Bugs put back: the grid barrier removed — 7 of 12 generated programs give wrong values at a 3-block grid (a later
  level reads a block's earlier level not yet run); a chain's last member's copy-out aborting the group (found by the
  generated test: ON issued more launches than OFF) — fixed by launching the group before the copy-out and ending a
  chain at a copied-out output; a one-candidate atomic site (`--cands=1`, direct launch behind its zero op) taken as a
  member — found by the evidence checker (`calls:on` MISMATCH on `pt_decode.c1`), fixed in the shape rule.
- Tests of other rungs that count launches or pin a pre-step text (devmap, epilogue, generic-fusion, horizontal,
  retain-delta, row-block, site-choice goldens, streams, tf32-tile, zero-fold) pin `calls` off.

### row-block on the rtx-5080: the missing fact

The split arm of pt_attn_1024_64 (CUPTI, `rows/traces/off.tsv`) against the site model's terms (`--plan-cands`):

| split site (rules pick) | model f (FFMA) | memory term | model `max` | measured | f + memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| q·Kᵀ, unsplit, 128 blocks × 256, one wave, 4 warps / scheduler | 4.38 | d 5.30 (4 MB stored) | 5.30 + 0.26 | **10.56** | 9.68 + 0.26 |
| P·V, atomic ×8, 1024 blocks × 32, one wave | 4.24 | l 5.59 | 5.59 + 1.31 + 1.95 | **12.64** | 9.84 + 1.31 |
| softmax (rows), 1024 blocks × 256 | — | 3.06 (L2 bytes) | 3.06 | **7.58** | — |

In a kernel of one wave every block runs the same phases at once (panel loads, FFMA, the epilogue's stores): no later
wave's FFMA hides them, so the memory and FFMA terms add (0.94 and 0.88 of measured) instead of overlapping (`max`:
0.53 and 0.55). That is the missing fact: **phase serialization in a single wave** — overlap needs a second wave (or
k-depth), a geometry × hardware fact (waves from `sms` × residency), not a per-part constant. Applied to both sides of
`block_cost` (an experimental build, not committed: `kernel = f + max(l, d, chain)` when `waves = 1`), the model prices
the unit at 27.4 us (measured 25.7–28.4) and the split at 25.8 (measured 30.8): the tile sites now match; the rest of
the split's error is the softmax kernel (3.06 vs 7.58 us — the second missing fact, the row segments' latency: tree
steps and barriers at few warps per scheduler; the unit's row segments are 0.1 vs 5.8 us). With the first fact alone
the 5080 still refuses (27.4 / 25.8), so the decision is not yet right on this part (open). The term changes every
single-wave site's price on every part with plan rates (the 4090 would form 1024 × 64 and 2048 × 64), so it ships with
its own evidence pass (plan-cost holdout refit, suite), not inside this step.

### Open

1. **The whole loop in one kernel is not built.** A trip's chain is one persistent kernel; the host still runs the
   trip (the exit test, a range cell's offsets from the counter, the retained rows' first-trip branch and row
   pointers, the carried buffers' swap, the argument arrays). Moving them needs the trip's host scalar code compiled
   for the device. Priced by the same properties a trip boundary removed saves `dep_launch − gsync(128)` ≈ 0.01 us of
   device time (one cooperative launch for all trips instead of one a trip would repay its 2.04 us once): measurable
   only on loops whose trips are host-bound. Device-side tail launch (CDP2) is no substitute: 7.6–8.0 us a launch.
2. **row-block's missing facts** (above): single-wave phase serialization in `site_ns`, then the row segments'
   latency; until both, the 5080 refuses the attention unit that measured faster.
3. The rule compares device boundaries only. A host-bound chain (host issue 1.69 us a launch) could pay back the
   cooperative launch from the host side; deciding that needs the members' device times (the plan model) to know
   which side bounds the chain. None of the suite's chains is host-bound (CUPTI: the device is busy 88–89 % of a trip).
4. Co-residency is planned at the plan's residency and decided at launch by the loaded kernel's: decode's tuples with
   the 32-thread row candidates use 255 registers (ptxas), so the runtime launches 84 blocks, not 128 (`s` and `o`
   take two rounds), against the plan's price. A ptxas-usage fact at plan time would price it.
5. A retained-row chain's first trip launches each member alone (the `horizontal` groups the chain absorbed are gone
   that trip): +3 launches once against −2 a trip after.
6. Workspace (split-K + reduce) candidates are not taken into a chain (the reduce would be a level of its own); a
   one-candidate atomic site (direct launch) is not a member.
7. The barrier price is the plateau to `sms` blocks (553 ns); 1–2 blocks measure 398–401 ns (priced high: conservative).
8. **`pdl` ships off on the rtx-5080 by the legacy `sm_120` `GPU_RUNGS_OFF` row** (the 5090's "no stable sign").
   The measured properties say a programmatic dependent launch is 164 ns cheaper than a plain one (486 vs 650): its
   default should be that rule, not the row — the same removal as `row-block`'s, next.
9. The persistent kernel's bodies still differ from their own kernels: rows + q 5.0 us against 4.22 (the dispatch
   and barriers around them), `s` + `o` 2.66 against 3.97 (no launch ramp). Priced since `review-fixes.md` §3 as a
   measured bound (`coop_body_ns`, 2500 ns a chain on this part); the mechanism stays open (not the barrier's poll,
   not the non-coherent load path).
