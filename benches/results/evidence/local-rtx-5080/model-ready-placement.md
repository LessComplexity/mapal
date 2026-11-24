# local-rtx-5080 — model-ready `placement`: a readable Llama's decode step on the device (2026-11-25)

Machine: Threadripper 9970X + RTX 5080 (sm_120, 84 SMs, 16 GB, ~960 GB/s), CUDA 13, shared with other agents (load
10–30, Tctl 70–94 C; timed processes pinned to CPUs 4-7,12-15,36-39,44-47 under the bench lock). Branch
`model/placement` from `fac7691b`: `b23750db` (the lang scout's two local fixes, verbatim) + the placement commits.
Raw output: `.claude/raw-evidence/local-rtx-5080/model-ready-placement/` (`before/`, `after/`, `work/`, `changed*/`,
never in git).

The program: the lang scout's readable Llama-3.2-1B (`model-scout/lang/llama_3_2_1b.mapal`: descriptive names, small
functions, a `KvCache` struct state, 16 decode-layer calls with a literal layer index, synthetic bf16 weights made by
maps, prompt 512, 31 decode steps, greedy argmax). Not one line of it changed; `llama_timed.mapal` is the same text
with three `time` reads (after the weights, after the prompt's logits, after the decode loop) for the split.

## 1. What kept the decode step on the CPU, and the generic change for each

| cause (found in the fixed point's drops) | change | where |
|---|---|---|
| Every operand of an inlined call is a `Proj` of its argument tuple, every struct field a `Proj` of the struct — inside a loop the rewriter never forwarded them (P2 forbade any key in a loop SCC), so no fact saw the arrays: a cell's operand was "a view", never "the device output" | **proj∘pack inside a loop SCC**: a `Proj` of a pack forwards its slot feeder when the feeder is in the same SCC or a loop invariant; a view feeding a loop route only within its SCC (DESIGN §1.2's own "same SCC" alias rule) | `mapal-rewrite` `equations.rs` (`forward_loop_view`) |
| `mut kv: KvCache` is a loop-state slot of struct type: the backend's carries (F3) only know array slots | **scalar replacement of a loop state**: a tuple state with a product slot is carried as its leaves; each product view is rebuilt as a pack of leaf views (proj∘pack then reaches them); the init / next packs are read leaf by leaf, never replayed (a token taken once) | `replay.rs` (`flat_state`, `flat_pack`, `flat_views`, `flat_skip`) |
| The dead packs those leave behind (an inlined call's result tuple, the old struct) were pinned as DCE verdict sinks and read as "host uses" | **DCE**: a dead token-free tuple / struct pack whose feeders are read elsewhere (or are such packs) is no sink; its feeders are | `graph_rewrites.rs` (`dead_pack_feeders`) |
| RoPE's `key[partner]`, `partner = i ± 32` under `i % 64 < 32`, unproven: the condition and the arm read `i` through two `Proj` views of the body input, and the narrowing reached only one | **bounds**: sibling views of one input slot share a pin | `mapal-ir` `algo/bounds.rs` (`narrow`, `step`) |
| The embedding gather `embedding[token · 2048 + i]` unproven: `token` is an argmax fold's index component | **bounds**: a *selecting* fold's component (each step keeps the accumulator's, takes the element or a constant) is ranged by seed ∪ items ∪ constants | `bounds.rs` (`select_range`) |
| The appends `cell / 512 == layer · 544 + position` had no range law: two captures | **map_law `Outside`**: a capture the *site* passes as a constant (an inlined layer's literal index) is that constant at the site | `algo.rs` (`capture_affine` gets the site) |
| Small per-element maps (RMSNorm, RoPE, residuals, SiLU·up, the gather) were never candidates (`cell_pays`: too small) and a host-made operand dropped every cell after it | rung **`trip-cells`**: a cell a loop runs every trip is a candidate; an operand made once is there (uploaded once). Rung **`bytes-place`**: a per-trip cell runs where it moves fewer bytes per trip (tie → the device when a device kernel reads it); a nest cell that pays by its own roofline uploads / copies out like a tile site | `func/resident.rs` (`cell_sites`, `cell_round`) |
| 16 appends per trip to one cache: only an append writing the *next state* over the *view* was in place, so 15 of 16 were whole-extent cells (4.46 M cells, 17.8 MB each) | rung **`range-chain`**: a chain of range maps, each over the one before (dead past its writer), from a carried view to the next state, writes the carry's buffer (`Carry::chain`, aliased) | `resident.rs` (`range_in_place`, `range_chain`, `alias`) |
| A cell writing a carry's next state counted the back edge's pack as a host read | rung **`carry-fwd`** | `resident.rs` (`cell_round`) |
| Gate / up (8192 rows) ran one thread a row: `cell-lanes` takes a warp only when the cells leave threads idle, so a row of `w` was read `k` elements apart (313 µs, 107 GB/s) | **fact** `step_algebra` regroup reads: a captured array's affine read has its stride along the fold (`radix_plan`'s `k` digit). Rung **`stride-lanes`**: a fold reading at unit stride takes a warp a cell; a one-row narrow site over a k-major `b` (the vocabulary projection) is that nest cell, not the one-thread tile | `algebra.rs`; `resident.rs` (`cell_lanes`); `func/gpu.rs` (`tile_plan`) |
| Cells of one body shared by inlined layers' sites emitted one kernel each, same id: `opt` rejected the module | one kernel per cell id | `gpu.rs` (`device_module`) |

Every rule reads facts (`cell_plan`, `map_law`, `bounds_proof`, `step_algebra`, `last_use_plan`, `loop_plan`) and the
part's properties (`device_pays`); nothing names a workload. Each backend rule has an OFF arm
(`--gpu-off=trip-cells,bytes-place,range-chain,stride-lanes,carry-fwd`); the new rungs are host-only and stripped
from kernel keys, so every other kernel keeps its id.

## 2. Before / after on the scout's program (RTX 5080)

Before = the emitter at `b23750db` (fac7691b + the scout's fixes), after = `model/placement` HEAD. Tokens printed are
identical (`98006 98006 118186 121734 …`); the 128 256 final logits differ by at most 3.8e-3 (|logit| ≤ 74.8, the
default face's free order: device trees against host loops).

| per decode step (static `; mapal.move` + `MAPAL_GPU_TRACE`) | before | after |
|---|---:|---:|
| ops on the device | the vocabulary projection only | every op but the RMSNorm sums of squares and the argmax |
| launches | 1 | 261 |
| bytes to the device | 8 192 (the embedded token) | **0** |
| bytes from the device | 513 024 (logits) | 783 360 (logits + 33 × 8 KB for the host sums of squares) |
| KV caches (2 × 17.8 MB) | host | carried on the device, each append writes 512 cells in place |
| ms per decode step (timed harness, 31 steps) | 679.5 | **4.9** (150.9–152.2 ms / 31) |

| prefill (prompt 512) | before | after |
|---|---:|---:|
| per-layer bytes to / from the device (static) | 81.8 MB / 35.7 MB | 44.0 MB / 6.3 MB |
| ms (timed harness) | 587* | 187–204 |
| whole program wall (prefill + 31 steps + print, incl. ~0.9 s context / module / weight generation) | 18.5–26.8 s | 1.3–1.5 s |

\* the before program's host tasks overlap the clock fences; its whole-program wall is the reliable figure.

The decode step (after, `MAPAL_GPU_TRACE=1`, every launch synchronized, last step):

| kernel | launches | ms | achieved |
|---|---:|---:|---|
| gate + up (hz of two lanes cells, 8192 rows × k 2048, bf16) | 16 | 1.47 | 730 GB/s |
| vocabulary (lanes cell, 128 256 × 2048 bf16) | 1 | 0.59 | 890 GB/s (was 1.83 ms, 282 GB/s, one-thread tile) |
| down (lanes cell, 2048 × 8192) | 16 | 1.04 | 516 GB/s |
| query / output (lanes cells, 2048 × 2048) | 31 | 0.66 | 405 GB/s |
| key / value (512 × 2048, in hz groups) | 30 | ~0.5 | ~130 GB/s (512 warps in flight) |
| attention scores / weighted sum / folds, RoPE, RMSNorm scale, residuals, SiLU·up, appends (range 1×512) | ~150 | ~1.5 | latency-bound small kernels |

Against the competitors' 3.00–4.50 ms/token (prompt 512, batch 1) and the 2.57 ms DRAM floor for 2.47 GB of bf16
weights: 4.9 ms is inside 1.1–1.6× of them; the gap is §4's.

## 3. Tests

- `crates/backends/llvm/tests/placement.rs`: 24 generated decode-like programs (`testgen::decode_strategy`: a struct
  cache state, one or two literal-index layer calls, the second reading the first, rotated keys, chained appends,
  attention over the filled rows, an argmax token's embedding row) — the interpreter's output in four emulator modes,
  on the CPU and with `range-cell` / `devmap` / `cell-lanes` off; coverage `{flat-states 24, device-carries 24,
  trip-h2d-zero 24, range-cells 24, range-chains 13, rotated-on-device 9, lane-cells 24, gather-cells 24,
  off-arm-runs 8}` — every loop uploads 0 bytes per trip.
- `crates/mapal-rewrite/tests/placement_facts.rs`: each fact on a small program (the partner read proven, the argmax
  gather proven, the constant-capture range law, the strides along the fold, a struct state carried as its leaves with
  no view of a pack left in the loop) against the raw graph's interpreter output.
- Changed expectations: `caching_memory` (`same_every_trip_launches_once`: the transpose is now a device cell
  launched once plus its operand uploaded once — 2 first-trip items, 1 with `trip-cells` off; `fit-smaller` also
  counted on the `trip-cells`-off placement), `differential` (the single-range-cell launch bound is `range-chain`
  off's; with it on, no loop moves more bytes per trip).
- `cargo test --release -p mapal-ir -p mapal-rewrite -p mapal-backend-llvm --no-fail-fast` at the final commit: 847
  passed, 0 failed (82 test targets, emulator modes included).
- Suite: `benches/local/suite_diff.py --base-emit <b23750db emitter>` — 35 programs + 2 cells × 5 faces, **same=45**
  (host and device text equal): null by construction, nothing to time.

## 4. Open (each tried or measured, and why it did not close here)

1. **RMSNorm's sum of squares runs on the host** (33 syncs a step, 8 KB each): `(0.0, features) -> fold { s, i -> s + x[i]
   · x[i] }` is a whole fold whose term is a function of the read, and `whole_fold` / the row-fold kernel take `acc ⊕
   e` only. Generic direction: a row / whole fold site whose element term is a per-element body (`acc ⊕ g(x[k])`,
   `fold_tree_step` already proves `g` blind to `acc`), the kernel calling `g`.
2. **The argmax** (a tuple fold) runs on the host, so the logits cross each step (513 KB): a device select-fold (the
   `Select` row-fold op over `(value, index)` pairs) would keep them.
3. **Prefill's hidden state and cache are not carried** (44 MB up / 6 MB down per layer): the prefill loop exits with
   `(hidden, cache)`, a tuple payload of views, and a carry's exit is only a single-view payload (`Carry::exit`).
   Direction: a carry's exits as the exit object's component views along the payload path, each aliased to the
   carry's buffer and copied out only where the host reads it.
4. **Launches**: 261 a step; the small kernels between projections are separate launches (`horizontal` merges only
   independent ones). The decode chain's elementwise maps are row-block members of the projections' outputs — a
   `unit_plan` group around a nest cell (today only around tile sites) would merge them.
5. **K/V projections** (512 rows) at ~130 GB/s: a warp a row leaves the part with 512 warps; more lanes a row (or a
   split of `k`) where the rows leave warps idle.
6. **Layer index as a loop counter** (the ideal form's layer loop): the range law takes one captured scalar; two
   (`layer · P + position`) need `Outside` as a sum of captured terms.
7. **Evidence pass**: launch / movement / per-kernel bandwidth above; SASS and `ptxas -v` of the new lanes cells not
   captured in this step.

## For integration

CLAUDE.md facts table (edits to existing rows):

- `bounds_proof`: add "… sibling `Proj` views of one input slot share a condition's narrowing (a rotation's partner
  `i ± h` under `i % 2h < h`); a selecting fold's component (each step keeps the accumulator's, takes the element or a
  constant: argmax / argmin / first-match index) ranged by seed ∪ items ∪ constants".
- `elem_plan`, `map_law`: add "… an interval's bound through a capture the site passes as a constant (an inlined
  call's literal) is that constant at the site".
- `step_algebra`: "per-read stride along the fold axis" — now also a captured array's affine read (`radix_plan`'s
  fold-index digit), e.g. a 1-row projection's weight row at stride 1.
- backend row: the rungs `trip-cells`, `bytes-place`, `range-chain` (`Carry::chain`), `stride-lanes`, `carry-fwd`
  (model-ready placement).

Rewriter (DESIGN `docs/components/rewrite` §1.2 P2 / §2): proj∘pack forwards inside a loop SCC (same SCC, or a loop
invariant off the routes); a loop state with a product slot is carried as its leaves; DCE drops a dead token-free
tuple / struct pack whose feeders are read elsewhere.

`docs/tailored-fixes.md`: none (no tailored fix).

Prerequisite: `b23750db` applies the lang scout's `local_fixes.patch` verbatim (inline_return through an inlined
Call / fused Map; pre-loop waits before the entry state's host-built edges) — drop it at integration if another
branch carries the same fix.
