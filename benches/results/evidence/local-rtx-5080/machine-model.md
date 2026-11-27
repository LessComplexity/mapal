# machine-model — ADR-0050 step 1, commits C1–C5 (local RTX 5080)

Spec: `docs/decisions/ADR-0050-the-machine-model-placement-by-one-cost-model.md`; design checks
`.claude/sprints/machine-model/design-check.json`. Raw output (not in git): `.claude/raw-evidence/machine-model/`
(`place_calib-rtx-5080-run1.txt`, `-keep2.txt`, `place-tests-*.txt`, `replay-*.txt`; C4: `c4/r4/place.tsv`,
`c4/r3gpu/`; C5: `c5/` — programs, builds, `abba.tsv`, `abba-layout1.tsv`, SASS; the review's fixes: `review-fix/`).
Their run-output files are rows of `benches/results/evidence/RAW-manifest.tsv` (path, bytes, sha256) with the asset
`pending:raw-machine-model.tar.gz`: the upload (`benches/gpu/archive_raw.sh`, a GitHub release) publishes, so it is
the user's step.

**Next step (2026-11-27): `cost-model.md`** — the primitives measured in one calibration run, stored as machine
facts, and the evaluator's max-over-resources replaced by a pipeline calculation over them.

## What landed

| commit | what | check |
| --- | --- | --- |
| C1 | crate `mapal-place`: machine / schedule / fact types with provenance, the evaluator, reuse closed forms, exact LRU trace simulator | P1–P4, P6, cited walks |
| C2 | `crates/backends/llvm/src/place.rs`: `machine_of` for every GPU profile (provenance in `profile::PlaceFacts`), the five present emitters as lowerings (constraints, footprints, issue counts, ring in flight, coordinate domains), `schedule_of` / `launch_of`, the coverage probe | coverage test |
| C3 | `place_replay.tsv` (136 arms, 68 sites) and `place_replay_scores`; `benches/local/place_calib.cu` run on this RTX 5080, its values stored as measured facts (`PLACE_RTX_5080`) | replay score, probes |
| C4 (6cb71680) | `mapal_place::search` (branch and bound, best-first under the floor, the gate, minimax, opens, the band); the emitters' search spaces and declared constraints; the gate (`--place-gate=ptxas`); the rung `machine-place` (default off) at `launch_in` / `launch` / `resized`; `PLACE_BAND_PM` from the replay; R3 / R4 | search props, coverage, R3, R4 (below) |
| C5 | the hybrid traversal: one launch, the walk over the raster order's first tiles, the whole waves over the rest, the zero pass a prefix where whole pieces store; its space, evaluator, reuse and LRU layout; values; the ABBA; `hyb-*` replay arms | values, ABBA, replay (below) |

**OFF arm against 1cbcfe89 (corrected by the review).** C4 / C5 claimed no emission change, but compared HEAD with
HEAD: `launch_key` hashes the launch's text, and the new `machine_place: false` and a walk's `whole: 0` entered it, so
199 of 280 suite emissions (35 programs × 8 profiles) had new candidate and reduce kernel ids. Stripped like every
earlier field (`gpu.rs launch_key`, test in `the_off_arms_reproduce_todays_launch`): **280 of 280 emissions equal to
the 1cbcfe89 emitter's** (host + device IR less evidence comments; `review-fix/offarm.tsv`) — null by construction.

## Review fixes (2026-11-27)

The review of C1–C5 (14 findings) and what each became. Raw: `.claude/raw-evidence/machine-model/review-fix/`.

| finding | fix | check |
| --- | --- | --- |
| OFF arm not byte-identical (kernel ids) | `launch_key` strips `machine_place: false` and `whole: 0` | 280 / 280 equal to 1cbcfe89's emitter; `the_off_arms_reproduce_todays_launch` |
| replay score lenient (ties as picks, arm-order tie-breaks, a site counted twice) | one rule for the search and the replay: `mapal_place::verdict` (minimax over the value and 16 corners; `flip` where another arm is cheaper beyond the null at some corner, `tie` within the null either side); `cut-1` / `cut-7` merged into `cut-0` / `cut-6` (two runs, the mean) | `place_replay_scores` (floor 47 → **37**, 6 misses, 30 open), `place_replay_sites_are_distinct`, `the_verdict_opens_ties_and_flips_only` |
| warm operands only from a hand flag | `place::warm_sites` (each site's operands' `residence_plan` at the part's keep), read by every launch of an emission through `WarmScope` (under the rung only); the replay lowers the measured program (`program=mm_1x1024x4096`) for it | pc-0 picked with both operands warm from the program's facts |
| native-gpu lost the placement facts | the calibration line carries them (`place.start`, `keep_*`, `hit`, `lat_l2_ns`, `eff`, `latency`, `clock_khz`, `bar_cycles`; `profile::place_line`); the committed 5080 line extended | `native_gpu_translates_to_the_hand_profiles_machine` (every machine row's value, interval and marking equal), the rules' native tests compare the facts (no overwrite) |
| staggered-start closed form wrong (to −91 %) | the window missed the blocks launched after the later reader that start before it, the trace's tie order and the reader's own event; one middle step stood for all | P2 now asserts it (operands within a slice a row / column or 5 %); 300 fresh grids: worst 0.2 % (was 53 off > 5 %) |
| no value test of the picks | `picks_within_the_bound_through_the_emulator`: float operands, a cell of every 32 × 64 block against its exact sum and `M` | **found a defect**: a single-stage warp tile (`k = bk`, the search's) stored zeros — the MMA space now takes two stages at least; then all within the bound (below) |
| `REPLAY_OPEN` reason wrong for the A100 cuts | reasons rewritten from each miss's terms: the 3090 cuts — the partial adds as a burst (dearer cuts); the A100 `cut-8` — the L2's realized rate at the uncut grid's occupancy (the opposite direction); `cut-9` / `cut-10` are ties (open) | `place_replay_scores` |
| P2 tolerance p times loose; hybrid closed form off; zero pass counted twice | tolerance one slice a row / column (`gy·sa`, `gx·sb`) or 5 %; hybrids at heights 1–8, outputs compared; the walk's first reader of a slice the waves read: the distinct bytes since the waves' last read of it (every wave's history, not the last wave's); the walk's outputs count hits; the hybrid's zero pass once; the evaluator streams the zero pass through the cache and against warm operands | P2 (300 grids) and P2 large (40, gx / gy 8–32, panels to 115); the review's +10.8 % shape (2048 × 4096 × 4096 h8): 64.6 against LRU 64.4 MB |
| an incumbent below every candidate marked open | open only where the gate refused candidates | `an_incumbent_below_every_candidate_is_no_open` |
| provenance labels | `fp32_lanes` derived spans half to twice; hand profiles' geometry `declared` (the native profile's `queried`); the 5080's accumulator `measured` (`mma_calib.cu`); H100 PCIe `launch_ns` a fit; an unmeasured launch a placeholder (1 / 3 µs, ×3 either way), no part's value | `every_gpu_profile_translates_with_provenance`, native test |
| probe weaknesses | start probe under three loads — L2-resident, DRAM-contended, a GEMM's shared panels: lockstep in all (median 0–22 ‰); keep's resolution an interval `[68 784 128, 70 463 488)` (the sweep's next point) on all three patterns; the wall-time FFMA peak 59.54 TFLOP/s at 2.879 GHz, 962 ‰ — the clock × efficiency (59.8) is the throughput, so E1's 52.7 (the profile's 2.45 GHz) is an older, different measurement; ABBA null columns; raw rows in `RAW-manifest.tsv` (upload pending: `archive_raw.sh` publishes, the user's step) | `review-fix/calib/run-review.txt`, `keep-review.txt` |
| one-thread-per-cell kernel's registers | declared from `ptxas -v` on sm_80 – sm_120 (31–40 registers at 1–8 cells, 43–48 at 16): the full-occupancy budget a lane (32 where an SM holds 2048 threads, 40 at 1536 — the code generator fills it), at least `32 + ⌈4·(cells − 4)/3⌉` (48 at 16) — 16 cells a thread now 5 blocks an SM, not 6; the cell kernel emits no `min-blocks`, so no cap and the rules' assumed residency stays metadata | `naive_declared_registers_cover_ptxas` (5 parts × f32 / bf16 × 5 cell counts), coverage 0 outside |
| stray `.scaled-*.mapal` files | removed; git ignores them; content-named scratch copies; `exposition.py --check` reports a tracked one | `git check-ignore` |
| coverage fold read off the launch | the fold is the emission's face (its order) | `rules_launches_are_points_of_the_space` |
| evaluator terms assumed | the walk's prologue at its requests' latency (misses over requests); the hybrid's phases priced by their own misses (`reuse::hybrid_split`, the floor by `hybrid_first`); the minimax over 16 corners (each group — operation, cache, memory, dispatch — slow or fast) | `the_verdict_opens_ties_and_flips_only`, P4, B&B = exhaustive |

Tests (release): `mapal-place` (props P1–P6 with the new P2s, search), the backend's lib tests (148: replay score and
band, coverage, native, naive registers, OFF arm), `machine_place` (R3, C5, the picks' values) — and the workspace
once: 1975 tests, 0 failed (`review-fix/workspace-release.txt` to `tensor_tile`, where its 3 h limit ended it, and
`workspace-rest-*.txt` from `math` on and every other crate; the CUDA differential sweep needs at most 12 CPUs — at 64
parallel native runs its GPU allocations fail, infra exit 102, no parity divergence).

## Intent vs realized (evidence pass)

| intent | realized |
| --- | --- |
| reuse counted by distinct bytes equals an exact LRU (lockstep ordered / split) | P1: equal byte for byte over 400 generated grids (raster, partitions, zero pass, outputs) and the four cited grids |
| the walk's closed form within one slice a row or column, or 5 % (P2) | (review) passes on 300 generated walks / hybrids / staggered grids at raster heights 1–8, outputs too, and 40 large ones; at C1–C5 the test allowed a full extra read of the operand and hid hybrids off by up to 46 % — fixed (above); cited walks: 3090 956 MB, A100 1074 MB, 5080 3072 × 4096 × 8192 124 MB — closed form = LRU = the design check's numbers (the old `walk_bytes` read 1685 MB for the last) |
| staggered starts | (review) the window union within 0.2 % of the simulation on 300 fresh grids, asserted in P2. At C1–C5 it was reported only and wrong (to −91 %) while read on 7 of 8 parts (start unmeasured there: both timings at the slow and fast corners) — the "flip" opens on those parts rested on it |
| P3 / P4 / P6 | once ≤ reads ≤ every reader, non-increasing in the keep; time non-increasing in every rate and non-decreasing in every extent, never below the admissible bound; three-gap = brute force (2000 cases) |
| every rules' launch is a point of the space, `launch_of` reproduces it, emission byte-identical | 4272 emissions (401 programs × 8 GPU profiles, the `mm_*` cells on all five faces), 162 725 distinct launches (rule-internal ones included): 0 outside, 0 round-trip differences, 0 persistent grids in the suite |
| the evaluator reads no fit, no table by version, no other part's value | test: perturbing each fitted `DeviceCost` field leaves every machine unchanged; every value not measured on the part is `derived` / `nominal` with an interval (cuda-ada 42 of 59 facts, rtx-4090 23/60, h100 38/76, h100-pcie 58/75, rtx-5090 30/61, rtx-5080 14/163, rtx-3090 26/56, a100 24/56) |

## C3 probes (RTX 5080, `place_calib.cu`, driver 620.73.04, 360 W)

| fact | measured | was |
| --- | --- | --- |
| start timing past the first wave | lockstep: a uniform grid's wave starts within 0–10 ‰ of a run (1 and 2 blocks an SM, 4 waves, 6 reps); (review) the same under a DRAM-contended load (every block its own stream, 16 × the L2) and a GEMM's panel load (shared row / column panels, ~4.5 × the L2): median 0–22 ‰ | unmeasured |
| L2 keep, cyclic | 68 784 128 B (1.025 × 64 MiB; plateau 4130 GB/s, DRAM 905); (review) the sweep's resolution: below 70 463 488 (1.05 ×) for every pattern — each keep a fact on `[68 784 128, 70 463 488)` | 68 786 585 (`l2keep.cu`) — agrees |
| L2 keep, rotated (a walk's readers on other SMs) | 68 784 128 B (plateau 3864 GB/s) | unmeasured |
| L2 keep, strided (256 B of every 1 KiB row) | 68 784 128 B (a 5 % dip at 0.94–1.0 ×) | unmeasured |
| hit against distance (cyclic) | 1.0 to 1.0 ×, 0.85 at 1.025, 0.48 at 1.05, 0.02 at 1.1, 0 past | step assumed |
| FFMA efficiency (chains a lane × warps a scheduler) | 1 chain: 0.22 / 0.43 / 0.75 at 1 / 2 / 4 warps; ≥ 4 chains: 0.83–0.98, 0.97 at ≥ 2 warps | the 5090's fit (eff_max 0.8) |
| `mma.sync` TF32 / bf16 efficiency | ≥ 0.99 from 2 warps a scheduler at any chain count; 0.91 at 1 warp, 1 chain | none |
| dependent latency | FFMA 4.56, `mma.m16n8k8.tf32` 35.2, `mma.m16n8k16.bf16` 35.2 clocks | nominal |
| sustained SM clock under FFMA load | 2 866 818 kHz (clock64 / %globaltimer, 4 s, 428 windows) | profile 2.45 GHz (E1's FFMA peak ÷ lanes, which folds the efficiency in) — kept for the rules; the machine reads the measured one |
| (review) FFMA peak by wall time | 59.54 TFLOP/s at 2 879 094 kHz, 962 ‰ of the row rate (8 chains, 16 warps a scheduler, 2 s after 1 s warm) — the probe's clock × its efficiency (2.867 GHz × 0.97 → 59.8) is the box's throughput; E1's 52.7 TFLOP/s is an older run of another kernel, the 13 % unexplained by today's box | — |

## C3 replay score (`place_replay_scores`, 68 sites; picked / missed / open / null)

**Superseded (review):** this score counted a pick the search itself would mark open (a runner-up within the 2 %
null: 11 of the 47 at C5), broke exact ties by arm order (6 of the 8 `c11` opens were a slow-bound tie, `c11-7` a miss
with both arms at 106.4 µs) and counted two sites twice. The score by the search's own rule is under "Replay score
(review)" below.

| family (sites) | model | rules |
| --- | --- | --- |
| walk (17: WALK_ARMS) | 11 / 3 / 1 / 2 | 13 / 2 / – / 2 |
| cut (21: mma_cut_replay) | 11 / 6 / 3 / 1 | 18 / 2 / – / 1 |
| c11 l2-gate (12) | 3 / 1 / 8 / 0 | 12 / 0 / – / 0 |
| ws (4: H100 wgmma-ws) | 0 / 4 / 0 / 0 | 4 / 0 / – / 0 |
| k-fill (4), streamk-fill (2), mma-depth (4), split-tile (2) | 12 / 0 / 0 / 0 | 12 / 0 / – / 0 |
| plan-cost losers (2) | 1 / 0 / 1 / 0 | 2 / 0 / – / 0 |
| **all** | **38 / 14 / 13 / 3** | **61 / 4 / – / 3** |

`model ≥ rules`: **not met** (38 against 61). The test asserts the model's floor (38) and the committed open list
(14 misses, each with its missing term); the 13 open picks flip inside marked facts (the keep of the 4090 and the
A100 / H100 placeholders, start timing on every part but the 5080) — the ADR's minimax at work, not a bug. Model wins
the rules miss: walk-6 / walk-7 (the 5080's refused walks, 512 × 2048 × 2048 and its transpose).

**Missing evaluator terms (what the 14 misses name):**

1. **Movers are not priced.** The H100 warp-specialized arms (ws-0..3, c11-7) are L2-bound at one price for TMA
   boxes and `cp.async` lane copies; the measured 3–25 % is the copy engine's. Next: a slot's panel movement priced by
   its mover — bulk copies at `copy_ns` × boxes over `in_flight`, lane copies as Global issue.
2. **The tensor tile's efficiency at its occupancy on the 3090 / A100 / 4090** (cut-0/1/2/8/9/10): the cut wins on
   wave quantization in the model, loses on the box. Only the 5080 has a measured curve; the others run on the latency
   rule (marked). Next: the `eff` probe on a rented part (C6) — the efficiency per warp tile × occupancy, and the
   cut's partial adds as a burst at each piece's end (atomics as a resource at its place, not spread).
3. **The walk's own per-unit cost** (walk-4 / -5 / -16: the 5080 walks the model prices under ordered grids that
   won — its segment loop and the shared tiles' fixup, and its pipe efficiency; the design check's open
   2048 × 6144 × 4096 line).

## plan-cost's losers

(Review: at C3 `Site::warm` came from a `warm=b` annotation of the replay's row, and the emission hard-coded both
operands cold — the falsification test passed only under that annotation. Now the emission reads each site's operands'
`residence_plan` at the part's keep (`place::warm_sites`, `WarmScope`), and the replay lowers the measured program
(`program=mm_1x1024x4096`) for the same fact: both operands warm, pc-0 still **picked**, pc-1 open (a flip at the
all-fast corner).)

`mm_1x1024x4096`: the shipped rules (an atomic split of the one-thread-per-cell kernel) 30.7 µs on the RTX 4090
(16.4 on the 5090) against plan-cost's pick (a workspace split, two launches) 84.0 (55.3). The term `site_ns` lacked
(its own `.claude/sprints/plan-cost/terms.md` open item 1, confirmed here): **operand residency across calls** — the
16.8 MB of weights stay in the 72 MB L2 between the program's repetitions — and with it **the L2's latency against
DRAM's** for the requests that hit. The new evaluator has both (`Site::warm` from `residence_plan`; latency weighted by
the share of a wave's distinct slices fetched from memory, between the L2's and the DRAM's measured loaded latencies).
With the weights warm it prices the rules' arm below plan-cost's on the 4090 (6.4 against 7.0 µs: **picked**) and
opens on the 5090 (the two within its intervals: **open**). It does **not** explain the size of the loss: both arms
are predicted 4–12 × faster than measured, so the 84 µs is still unexplained by any term (open: a one-row grid's
per-partition launch and reduce on four blocks).

## Proposed CLAUDE.md facts-table rows (not applied — CLAUDE.md changes need the user)

| query | what it proves |
| --- | --- |
| `mapal_place::Machine` (`place::machine_of`) | ADR-0050 D2: a target's places (a graph of instances: registers, shared memory, the L2 with its keep per pattern and hit curve, DRAM), executors (device → SMs → schedulers → warps → lanes), operations as rows of geometry (FFMA, `mma.sync`, `wgmma`, `tcgen05`: shape, homes, fragment footprint, latency, rate, measured efficiency), movers (`cp.async`, TMA, the host link) and dispatch (start timing, launch modes, atomics at the L2, zero pass, sustained clock) — every value a `Fact` with its source (measured / queried / declared / derived / nominal) and interval; fits never read |
| `mapal_place::evaluate` (`Terms`) | ADR-0050 D4: a schedule's predicted time — phases × max over compute (efficiency curve), issue (declared counts), memory by distinct bytes, the L2 feed, latency (ring in flight), atomics; combine, zero pass, launches — and the marked facts it read |
| `mapal_place::reuse::reads` (+ `lru::lru_bytes`; `hybrid_split`, `hybrid_first`) | a cache instance's missed bytes over a traversal by distinct bytes: lockstep exact (= LRU), the walk by the three-gap theorem, the hybrid (its waves and its walk apart; a walked slice's first reader against the waves' last read of it), staggered starts — every form within one slice a row or column, or 5 %, of the LRU (P2), outputs and the zero pass included |
| `mapal_place::verdict` (`Corner`, `evaluate_corner`) | ADR-0050 D5's one rule for a pick and its opens, the search's and the replay's: the minimax over the value and the 16 corners of the facts' intervals (each group — operation, cache, memory, dispatch — slow or fast); `flip` where another candidate is cheaper beyond the null at some corner, `tie` where one is within the null at the values |
| `place::warm_sites` / `WarmScope` | ADR-0050 D3 warm operands: each tile site's `[a, b]` resident where it reads them — `residence_plan` at the part's keep (`MAPAL_SITE_COLD`'s fact) — one map every launch of an emission reads (under `machine-place`) |
| `profile::place_line` | the calibration line's placement facts (`place.*`: start timing, keeps and their resolution, hit curve, L2 latency, efficiency points, latencies, sustained clock, barrier cycles) — a native-gpu profile carries its probes, not a hand profile's by name |
| `place::schedule_of` / `launch_of`, `Emitter` (`Lowering`) | the rules' launch as a schedule and back (byte-identical); each present emitter's declared constraints, footprint, issue counts per class per trip, ring in flight and coordinate domains (D5 steps 2–3) |
| `profile::PlaceFacts` | a part's placement facts beyond `DeviceCost` (start timing, keeps per pattern and the sweep's resolution `keep_hi`, hit curve, L2 latency, efficiency points, latencies, sustained clock) and the provenance of every profile value |
| `mapal_place::search` (`Ranked`, `Opts`; `tile_bound`, `lanes_bound`, `point_bound`, `evaluate_floor`) | ADR-0050 D5 (C4): the schedule the one evaluator prices lowest over every lowering's coordinate space — branch and bound on admissible bounds (peak compute, the panels into the units, operands once, the chains' efficiency at the most slots the lanes allow, the declared issue), best-first settling under the floor, dominance by the lowerings' own layout rules; the top 8 through the post-lowering gate; the verdict (`verdict`: minimax over the corners, `flip` / `tie` opens); the incumbent kept within the error band |
| `place::PLACE_BAND_PM` (test `place_band_is_the_replays_residuals`) | the model's error band per bounding term: the 80th percentile of the replay's pairwise ratio residuals (all 163 ‰, compute 137, DRAM 212, L2 259 — the review's corpus, duplicates merged) — a pick replaces the rules' launch only beyond it |
| `place::placed` / `placed_for` / `place_line` (rung `machine-place`; arms `--place-gate=ptxas`, `--place-trav=…`, `--place-tile`) | C4's seam: a tile site's launch and every kernel of its realization from the search (cached per site, part, face, arms, order); the `; mapal.place` line (schedule, price at the three bounds, the rules' price, kept, points / pruned / gated, terms, open) |
| `Combine::StreamK { whole }`, `gpu::zero_bytes` (`Declared::whole_stores`) | C5's hybrid: one launch of `whole + ctas` blocks — the whole waves over the raster order's last `whole` tiles, the walk over its first `tiles − whole` — and the zero pass a prefix where whole pieces store |

## C4 — the search (what landed)

| piece | where | check |
| --- | --- | --- |
| branch and bound over each lowering's coordinates (`Lowering::dims / search_domain / point / tile_of / lanes_of`), traversals per slot point, best-first settling, top-8 through the gate, minimax over Slow / Value / Fast, opens (`flip`: the slow and fast winners differ; `tie`: a runner-up within the 2 % null) — (review) the minimax over the value and 16 corners, `flip` where another candidate is cheaper beyond the null at some corner, `tie` within the null either side (`verdict`), an incumbent below every candidate no open — the incumbent's price as the threshold from the start and kept within the band | `crates/mapal-place/src/search.rs` | `tests/search.rs`: B&B = the exhaustive argmin (12 generated sites, every point and traversal priced); P5 the pick invariant under every domain reversed, ±1 % noise on every rate moved 0 of 12 picks (each would have to be open); the gate takes the next; the band keeps / drops the incumbent |
| the evaluation floor (every operand once, no reuse closed form; admissible as no term falls as misses rise) and memoized reuse closed forms (a pure function of grid and keep) | `eval.rs` `evaluate_floor`, `reuse.rs` | P4 unchanged; the RTX 5080's 128 × 4096 × 4096 f32 search 96 s → 6.8 s (release), 512 × 2048 × 2048 bf16 MMA 20 s → 1.2 s |
| the present emitters' search spaces: the site's facts (formats the precision set admits, the stored layouts `bt` / `b_kmajor`, staged widths, the face's scalar step), the emitters' declared constraints (whole 16-byte copies a thread, `cp.async` rings of ≥ 2 stages, a narrow `a` register-staged, the walk on the staged FMA tile and the 16-bit warp tile only, no raster on the cell kernel, the warp-specialized tile's TMA boxes, a warpgroup `a` from shared memory exactly where stored in the format), dominance (pads and row lanes by the bank model: `kmajor_pad`, `gpu_tile_sync_a`, `conflict_free_pad` / `pair_pad` / `ldmatrix_pad`; `held`, register reallocation, split forms at their incumbent's), a slot of whole groups | `place.rs` `Emitter::narrowed`, `declare`; `gpu.rs` `fma_constraints`, `fma_layout` | the coverage test still holds with the new refusals: 4272 emissions, 162 725 rules' launches, **0 outside**, round trip byte-identical |
| the post-lowering gate: the candidate's kernel text lowers (the emitter under `place::forced`, a refusal its panic) and, under `--place-gate=ptxas`, `llc` + `ptxas -v` with no spill and the registers within the file | `place.rs` `gate`, `ptxas_gate`; `gpu.rs` `lowered_text`, `kernel_module` | the smoke test: H100 PCIe 512 × 2048 × 2048 bf16 gated 8 warpgroup candidates whose `a` came from registers at 16-bit width (the emitter's assert) → that declared constraint added; R3 below |
| the seam: `launch_in` (the rules' chain, `place::rules`, is the incumbent), `launch` and `resized` (every kernel of a realization runs the placed launch: the ordered kernel beside a combined pick keeps the rules' tile), a placed site's table is its pick | `gpu.rs` (~60 lines at `launch`, `launch_in`, `resized`, `site_cands`) | the OFF arm byte-identical (coverage round trip; `rules::rungs` leaves `machine-place` off, `GpuRungs::ALL` too — it replaces the chain, it is no rung of it) |

The space: the slot tile, `bk`, the depth and the traversal are the search's on every lowering; the tile *family*
choices the rules make inside a lowering (k-major `a`, the prefetch depth, whole-file `held`, split realizations of an
`{f32}` product) are narrowed to the emitter's own derivation or the incumbent's — listed above, visible in
`narrowed`. Persistence stays out (its round-robin jitter is unmeasured).

**Search cost** (release; full search over all five lowerings with the rules' incumbent): 0.03–9 s a site, the
32-site smoke test over the eight parts 55 s. The emission runs it only under the rung; a debug build's tests place
fewer shapes.

## C4 R3 — generated GEMMs on every profile

`tests/machine_place.rs` `r3_generated_gemms_place_lower_and_assemble_on_every_part` (12 `testgen::GemmProg` at plan
extents × f32 / TF32 set / `bf16 | f32` × the 8 profiles, `--gpu-on=machine-place --place-gate=ptxas`): 96
emissions, every one lowered with a `; mapal.place` line on each placed site (96: 72 kept the rules' launch, 24
replaced, 93 open; review re-run: 67 kept, 29 replaced, 89 open), the OFF arm never placed; `llc -mcpu=<arch>` + `ptxas -arch=<arch> -v` on every device module:
287 kernels with no spill — every kernel the search placed among them (asserted) — and 1 rules' kernel (kept) that
spills on its own. **On this RTX 5080** (`.claude/raw-evidence/machine-model/c4/r3gpu/`, `run.sh`): 18 generated
GEMMs (6 shapes × the three sets; integer operands, so every realization's sums are exact) built OFF and ON
(`--place-gate=ptxas`) and run: the printed cells **equal on all 18**; 7 ran a replaced pick (FMA walks and ordered
one-warp tiles, one 16-bit warp tile), 11 the kept rules' launch. (`mapal test --weakest` was the intent; its oracle
does not finish these extents within a 4·10¹⁰ budget, so the exact-sum gate stands in.)

**Values within the bound (review).** Integer operands are exact in any order, so the runs above could not show a
rounding or fold-order defect, and nothing committed checked a pick's values. `tests/machine_place.rs`
`picks_within_the_bound_through_the_emulator`: float operands (`v / 7`, `v / 9` in f32, `testgen::gemm_bound_src_at`),
a cell of every 32 × 64 block printed, each against its exact sum and `M = Σ|a·b|` (`testgen::gemm_bound_ref`, f64)
within the set's coefficient `c·M` (`bound_coef_unit` of the set's unit and of f32's). (1) The hybrid on the 16-bit
warp tile at 512 × 6144 × 64 (168 whole tiles, the raster order's first 24 walked): all 1536 cells within the bound
in the three emulator modes, 192 of them in walked tiles; the 1344 whole-tile cells **bit for bit** the ordered
kernel's (one piece, the same fold). (2) The search's own pick (the rung alone) on each of the 8 profiles (f32, TF32,
bf16 sets in turn): 512 cells within the bound, 6 placed kernels the rules' replaced. **It found a defect:** on a
single-panel fold (`k = bk`) the search took the 16-bit warp tile at one stage (the domain allowed it there), and the
kernel stored zeros on three parts — the emitter's ring waits on `stages − 2` groups; the rules never emit one stage.
The MMA lowering now declares two stages at least (`stages in [2, k/bk]`, the design check's constraint).

## C4 R4 — the suite's emission, ON against OFF, per profile (at C5: its band and the hybrid in the space)

`benches/local/place_diff.py` (raw `.claude/raw-evidence/machine-model/c4/r4/place.tsv`): the 35 timed-suite programs
on the default face. "Moved": a kernel the ON arm launches is new or differs from OFF's of its name; "table": only the
host and the dropped site-table candidates (a placed site's table is its pick). Sites: one `; mapal.place` line a
placed site and order; "replaced" = the pick cleared the band against the rules' launch.

**Review re-run** (the verdict over 16 corners, warm operands from `residence_plan`, the naive kernel's declared
registers, two stages a warp tile, the keep's interval, the launch placeholder, the band re-derived; raw
`review-fix/r4/place.tsv`):

| profile | programs moved / table / equal | sites kept / replaced / open | replaced by lowering / traversal | biggest predicted gains (rules → pick) |
| --- | --- | --- | --- | --- |
| cuda-ada | 8 / 20 / 7 | 31 / 9 / 38 | fma ord 3, split 1, walk 3; naive 2 | conv_2048_5 252 → 120 µs (naive, 16 cells, 5 blocks an SM); attn_1024_64 12.4 → 6.8 (FMA walk) |
| rtx-4090 | 8 / 21 / 6 | 28 / 14 / 36 | fma ord 9, split 3; naive 2 | conv 96.5 → 46.7; prefill 11.7 → 7.6 (FMA workspace split) |
| h100 | 14 / 18 / 3 | 23 / 17 / 38 | fma ord 10, split 2, walk 1; mma ord 2; naive 2 | pj_m16_n2048_k7168 137 → 22.0 (FMA walk); conv 109 → 53.3 |
| h100-pcie | 9 / 19 / 7 | 31 / 13 / 39 | fma ord 7, split 3, walk 1; naive 2 | prefill 22.2 → 9.1; conv 70.0 → 30.7 |
| rtx-5090 | 13 / 19 / 3 | 24 / 16 / 36 | fma ord 9, split 3, walk 1; mma split 1; naive 2 | pj_m16_n2048_k7168 29.2 → 11.2; decode / causal 7.0 → 3.2 |
| rtx-5080 | 11 / 19 / 5 | 26 / 13 / 36 | fma ord 7, split 1, walk 3; naive 2 | conv 101 → 46.4; attn_1024_64 7.7 → 4.6 (FMA walk) |
| rtx-3090 | 19 / 10 / 6 | 22 / 24 / 43 | fma ord 13, walk 5; mma ord 2, split 2; naive 2 | conv 176 → 85; decode 10.8 → 7.5 |
| a100 | 8 / 6 / 21 | 28 / 10 / 19 | fma ord 5, split 3; naive 2 | conv 155 → 75; prefill 29.1 → 20.3 |

At C5 (superseded):

| profile | programs moved / table / equal | sites kept / replaced / open | replaced by lowering / traversal | biggest predicted gains (rules → pick) |
| --- | --- | --- | --- | --- |
| cuda-ada | 7 / 21 / 7 | 31 / 9 / 38 | fma ord 4, split 1, walk 2; naive 2 | conv_2048_5 252 → 120 µs (naive, 16 cells a thread); indep_512 16.1 → 11.1 (FMA walk) |
| rtx-4090 | 8 / 21 / 6 | 28 / 14 / 36 | fma ord 9, split 3; naive 2 | conv_2048_5 96.8 → 46.9; prefill 11.7 → 7.6 (FMA workspace split) |
| h100 | 14 / 18 / 3 | 23 / 17 / 38 | fma ord 10, split 2, walk 1; mma split 2; naive 2 | pj_m16_n2048_k7168 137 → 21.9 (FMA walk); decode / causal 13.9 → 5.5 |
| h100-pcie | 9 / 19 / 7 | 31 / 13 / 37 | fma ord 7, split 1, walk 3; naive 2 | prefill 22.0 → 8.9; mlp_512 20.7 → 8.5; conv 70.4 → 31.0 |
| rtx-5090 | 12 / 20 / 3 | 26 / 14 / 37 | fma ord 8, split 1, walk 2; mma split 1; naive 2 | decode / causal / prefill 10.4 → 3.1 |
| rtx-5080 | 10 / 20 / 5 | 27 / 12 / 36 | fma ord 8, walk 2; naive 2 | indep_512 / mlp_512 19.6 → 7.3 (FMA walk); conv 102 → 46.8 |
| rtx-3090 | 19 / 10 / 6 | 22 / 24 / 44 | fma ord 13, walk 5; mma split 4; naive 2 | conv 176 → 85; decode 11.0 → 7.7 |
| a100 | 5 / 9 / 21 | 32 / 6 / 36 | fma ord 1, split 1, walk 2; naive 2 | conv 155 → 75; prefill 29.1 → 20.3 |

Why each change: the pick prices below the rules' beyond the band (16 % at compute, 21 % at DRAM). The common reasons,
from the terms: (1) the one-thread-per-cell kernel at 16 cells a thread where the rules take fewer (conv's 2048 ×
2048 × 25 window, every part: the rule `vec_cells` gates on the DRAM share an SM asks, the model prices the loads in
flight a lane); (2) FMA tiles of one warp a slot at 8–19 slots an SM (`u9`–`u19`) — the model's efficiency is the
latency rule on every part but the 5080 (marked), and one-group slots raise the occupancy the latency rule rewards;
(3) walks and splits on under-filled grids. Most picks are **open** (a runner-up within the 2 % null, or a flip inside
a marked fact): the model does not separate near-equal tiles, which the band then protects. None of these is measured
yet: the ABBA of every moved program is C6's (the predictions here are the model's; the C3 replay shows its absolute
times run 1.1–3 × under the measured ones on DRAM-bound and skinny sites).

## C5 — the hybrid traversal

**Lowering (one launch).** `Combine::StreamK { ctas, iters, whole }`: blocks `cta < whole` fold tile `tail + cta`
whole (the whole waves, the raster order's last `whole = ⌊tiles / W⌋·W` tiles, `W` the footprint's slots); blocks
`whole ..` walk `iters` units each of the first `tail = tiles − whole` tiles' `tail·P` units (`Kernel::segment`, the
block's unit range a `select` on `cta < whole`; `ticket_split`'s blocks of a tile the same map). One launch, not
two: the tail's blocks start as the last whole wave's slots free (no launch gap, no second grid to wait for), the
kernel is the walk's own (its pieces, partial adds and tickets unchanged), and the host's call is stream-K's with
`whole + ctas` blocks. The walked tiles first makes the zero pass a **prefix** of the output (the rows of the first
`h·⌈tail / (gx·h)⌉` tile rows) where whole pieces store (`mma_walk_store`; the FMA tile adds every piece, so it zeroes
all: `Declared::whole_stores`). The evaluator, the reuse closed form and the LRU trace follow the same layout (P1–P2
hold). Space: `Traversal::Hybrid` at the footprint's occupancy, every raster height; emitted by the FMA (register-
staged) and 16-bit warp-tile lowerings.

**Values.** `tests/machine_place.rs` `c5_hybrid_values_through_the_emulator`: 2 generated GEMMs × (the 16-bit tile's
hybrid at 512 × 6144 × 64 plain — 168 whole tiles, 24 walked —, the same as an epilogue group — the walked tiles'
workspace and tickets —, and the best FMA hybrid at 256 × 2048 × 128): 6 hybrid kernels, each **equal to the
interpreter in the 3 emulator modes** (integer operands: any order exact; a whole tile's block is the ordered
kernel's fold). The ABBA arms below print the
same cells as the rules' arm to ≤ 2.9e-5 relative (atomic orders and bf16 operands; the bound holds).

**Intent vs realized (SASS, `ptxas -arch=sm_120 -v`, cuobjdump):**

| kernel (2048 × 4096 × 4096 bf16, the rules' 128 × 128 warp tile) | registers | spill | HMMA | LDGSTS | RED | LDSM | instructions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ordered | 190 | 0 | 128 (2 loop copies) | 16 | 0 | 32 | 464 |
| walk | 208 | 0 | 64 | 16 | 128 | 16 | 568 |
| hybrid | 210 | 0 | 64 | 16 | 128 | 16 | 576 |

Intent: the walk's loop body unchanged, the block map a few `select`s ahead of it — realized: +2 registers, +8
instructions, the k loop identical (2048 × 2048 × 8192: the same counts).

**ABBA on this RTX 5080** (walk-arm shapes of `place_replay.tsv`, `fold: bf16 | f32`, the rules' tile: R the rules,
O ordered, W walked, H hybrid; CUPTI medians of 6 processes, R O W H then H W O R, GEMM kernel + its zero pass; raw
`.claude/raw-evidence/machine-model/c5/`, `abba.tsv`):

| site | R (rules) | O | W | **H** | H − R | null R / H (max − min of 6) | H's zero | whole + walk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 128 × 28672 × 4096 | 872.2 (ord) | = R | 892.0 | **848.6** | −2.7 % | 1.2 / 0.4 % | 6.8 | 168 + 167 × 86 |
| 512 × 8192 × 2048 | 168.6 (walk) | 194.1 | = R | **162.5** | −3.6 % | 0.9 / 0.7 % | 4.0 | 168 + 166 × 34 |
| 2048 × 2048 × 2048 | 168.4 (walk) | 195.0 | = R | **162.0** | −3.8 % | 0.5 / 0.8 % | 3.3 | 168 + 166 × 34 |
| 2048 × 2048 × 8192 | 627.7 (walk) | 748.5 | = R | **588.4** | −6.3 % | 0.8 / 0.2 % | 3.2 | 168 + 167 × 135 |
| 2048 × 4096 × 4096 | 668.5 (walk) | 686.4 | = R | **581.3** | −13.0 % | 0.6 / 0.4 % | 1.5 | 504 + 147 × 7 |
| 2048 × 4096 × 8192 | 1287.8 (walk) | 1363.5 | = R | **1146.7** | −11.0 % | 0.3 / 0.5 % | 1.5 | 504 + 158 × 13 |
| 2048 × 6144 × 4096 | 945.7 (ord) | 955.1 | 1016.1 | **871.0** | −7.9 % | 0.5 / 0.3 % | 3.3 | 672 + 167 × 74 |
| 2048 × 8192 × 2048 | 630.9 (ord) | = R | 739.6 | **581.8** | −7.8 % | 0.7 / 0.3 % | 2.4 | 1008 + 147 × 7 |
| 3072 × 4096 × 8192 | 1869.5 (ord) | 1885.8 | 2007.8 | **1714.5** | −8.3 % | 1.2 / 0.4 % | 3.3 | 672 + 168 × 147 |
| 4096 × 2048 × 8192 | 1304.1 (walk) | 1350.9 | = R | **1146.5** | −12.1 % | 0.9 / 0.1 % | 1.0 | 504 + 158 × 13 |

No hybrid where the grid has no whole wave or no tail (512 × 512 × 2048, 512 × 2048 × 2048, 2048 × 512 × 2048,
128 × 6144 × 4096, 512 × 2048 × 8192: H = the rules'). The hybrid wins **all ten** sites it exists on, beyond the
2 % null, both where the walk won (−4 to −13 %: no partial adds or zero pass over the whole waves) and where the walk
lost to the ordered grid (2048 × 6144, 2048 × 8192 × 2048, 3072 × 4096 × 8192: −8 %; the walk's 38–58 µs zero pass
and its spread readers are gone). The 2048 × 4096 × 4096 hybrid (581 µs) passes the gaps-disasm leader (590.6,
CUTLASS 64x64_32x6). The first layout (the whole waves first, the full zero pass) measured −1.5 to −10.6 % on eight
sites and **+3.0 %** on 2048 × 8192 × 2048, whose 58 µs zero pass of 64 MB the walked prefix's 2.4 µs replaced
(`abba-layout1.tsv`) — the layout is the fix.

**Replay** (`hyb-0..9` added, 32 arms): at C5 the model picked the hybrid on 9 of 10 (the rules, which have no
hybrid, 0) — under the search's own rule (review) 5 picks, 4 ties within the null (open: hyb-3 / 5 / 8 / 9), 1 miss;
hyb-0 (128 × 28672 × 4096, H by 2.7 %) is open-listed: every arm priced ~3 × under its time (a DRAM-bound skinny
site; the tile's realized memory rate under its ring is not a machine fact). **But the band keeps the rules on all
ten**: the model prices the hybrid 1–3 % under the rules' arm where it measured 3–13 % — inside the 15.5 % compute
band. So `machine-place` (default band) does not emit these hybrids yet; `--place-trav=hybrid --place-tile` does.
The missing term is the walk's own: its partial adds and the zero pass as a burst at each piece's end, and the
ordered grid's wave tail — the model underprices the ordered and walked arms' losses (open for C6).

## Replay score (review: the search's rule, 76 sites)

Scored by `mapal_place::verdict` — the rule the search picks and opens by: a pick within the 2 % null of a runner-up,
or beaten beyond it at any corner of the facts' intervals, is open (never a pick); no tie is broken by arm order; the
two sites measured twice merged (`raw review-fix/replay-report-2.txt`).

| family | model picked / missed / open / null | rules picked / missed / null |
| --- | --- | --- |
| hyb (10) | 5 / 1 / 4 / 0 | 0 / 10 / 0 |
| walk (17) | 11 / 2 / 2 / 2 | 13 / 2 / 2 |
| cut (19) | 9 / 3 / 6 / 1 | 16 / 2 / 1 |
| c11 (12) | 5 / 0 / 7 / 0 | 12 / 0 / 0 |
| ws (4) | 0 / 0 / 4 / 0 | 4 / 0 / 0 |
| kfill (4), skfill (2), depth (4), split (2) | 6 / 0 / 6 / 0 | 12 / 0 / 0 |
| pc (2) | 1 / 0 / 1 / 0 | 2 / 0 / 0 |
| **all** | **37 / 6 / 30 / 3** | **59 / 14 / 3** |

`MODEL_FLOOR` 37, `REPLAY_OPEN` 6 — `cut-0` / `cut-2` (RTX 3090: the cut's partial adds, and past the keep their
read-modify-write from DRAM, priced as overlapped with the k loop; they burst at each piece's end — dearer cuts),
`cut-8` (A100: both arms priced at the full-grid L2 read rate; the uncut grid's 128 tiles on 108 SMs do not draw it —
the L2's realized rate at the grid's occupancy, which makes the uncut dearer: the opposite direction), `hyb-0` (every
arm ~3 × under its time at the DRAM roofline: unexplained), `walk-16` / `walk-5` (the walk's per-unit cost: its k loop
one copy against the ordered kernel's two in SASS, its segment loop and fixup, its partial adds' burst). The 30 opens
are ties within the null (the model does not separate those arms) or flips at a corner of marked facts. The error band
re-derived: all 163 ‰ (88 pairs), compute 137 (48), DRAM 212 (19), L2 259 (20); issue and latency take all's.

## Replay score after C5 (`place_replay_scores`, 78 sites; superseded by the review's score above)

| family | model picked / missed / open / null | rules |
| --- | --- | --- |
| hyb (10) | 9 / 1 / 0 / 0 | 0 / 10 / 0 |
| walk (17) | 11 / 3 / 1 / 2 | 13 / 2 / 2 |
| cut (21) | 11 / 6 / 3 / 1 | 18 / 2 / 1 |
| c11 (12) | 3 / 1 / 8 / 0 | 12 / 0 / 0 |
| ws (4) | 0 / 4 / 0 / 0 | 4 / 0 / 0 |
| kfill, skfill, depth, split (12) | 12 / 0 / 0 / 0 | 12 / 0 / 0 |
| pc (2) | 1 / 0 / 1 / 0 | 2 / 0 / 0 |
| **all** | **47 / 15 / 13 / 3** | **61 / 14 / 3** |

`MODEL_FLOOR` 47, `REPLAY_OPEN` 15 (hyb-0 added with its term). The error band re-derived with the hybrid pairs:
all 163 ‰ (90 pairs), compute 155 (49), DRAM 214 (19), L2 259 (21); issue and latency take all's.

## Open for C6

1. **The band keeps the measured winners.** The hybrid's 3–13 % and most of the walk's are inside the 16 % band:
   the residuals that set it are the walk's (partial adds and zero pass as a burst at each piece's end, its segment
   loop) and the DRAM-bound skinny sites' (the ring's realized memory rate). Shrinking those terms shrinks the band.
2. **The ABBA of every R4-moved program** (local: the 5080's 10 programs; rented: the 3090's 19, the H100's 14,
   the 5090's 12 — with the efficiency probe, since the moved FMA picks lean on the latency rule's marked efficiency).
3. The split realizations of an `{f32}` product and the k-major / prefetch families are the incumbent's or the
   emitter's own rule inside the search (narrowed, listed above); the walk's per-unit cost; movers (TMA against
   `cp.async`); the tensor tile's efficiency at occupancy off the 5080 — C3's list; (review) the partial adds' burst
   at each piece's end (the 3090 cuts, the walks), the L2's realized rate at a grid's occupancy (the A100 cut), and
   the 30 replay ties the model does not separate.
4. A placed site's table is its pick: the run-time confirmation among the search's top (D5) is not built.
5. The search costs 1–9 s a site in release (the FMA lowering's 10–25 k points); a debug build is ~10 × slower.
