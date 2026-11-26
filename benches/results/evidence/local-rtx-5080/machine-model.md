# machine-model — ADR-0050 step 1, commits C1–C3 (local RTX 5080)

Spec: `docs/decisions/ADR-0050-the-machine-model-placement-by-one-cost-model.md`; design checks
`.claude/sprints/machine-model/design-check.json`. Raw output (not in git): `.claude/raw-evidence/machine-model/`
(`place_calib-rtx-5080-run1.txt`, `-keep2.txt`, `place-tests-*.txt`, `replay-*.txt`).

## What landed

| commit | what | check |
| --- | --- | --- |
| C1 | crate `mapal-place`: machine / schedule / fact types with provenance, the evaluator, reuse closed forms, exact LRU trace simulator | P1–P4, P6, cited walks |
| C2 | `crates/backends/llvm/src/place.rs`: `machine_of` for every GPU profile (provenance in `profile::PlaceFacts`), the five present emitters as lowerings (constraints, footprints, issue counts, ring in flight, coordinate domains), `schedule_of` / `launch_of`, the coverage probe | coverage test |
| C3 | `place_replay.tsv` (136 arms, 68 sites) and `place_replay_scores`; `benches/local/place_calib.cu` run on this RTX 5080, its values stored as measured facts (`PLACE_RTX_5080`) | replay score, probes |

No emission changes: the coverage test's round trip is byte-identical on every program and profile (the probe is a
test switch), so no PTX/SASS differs and no A/B is owed by this step (null by construction).

## Intent vs realized (evidence pass)

| intent | realized |
| --- | --- |
| reuse counted by distinct bytes equals an exact LRU (lockstep ordered / split) | P1: equal byte for byte over 400 generated grids (raster, partitions, zero pass, outputs) and the four cited grids |
| the walk's closed form within one read a slice or 5 % (P2) | passes on 200 generated walks / hybrids (tiny keeps are the worst: up to 47 % relative, inside one read a slice); cited walks: 3090 956 MB, A100 1074 MB, 5080 3072 × 4096 × 8192 124 MB — closed form = LRU = the design check's numbers (the old `walk_bytes` read 1685 MB for the last) |
| staggered starts | window-union form reported only: median 1.00 of the simulation, range 0.16–1.10 (open; the 5080 measured lockstep, so no 5080 decision reads it) |
| P3 / P4 / P6 | once ≤ reads ≤ every reader, non-increasing in the keep; time non-increasing in every rate and non-decreasing in every extent, never below the admissible bound; three-gap = brute force (2000 cases) |
| every rules' launch is a point of the space, `launch_of` reproduces it, emission byte-identical | 4272 emissions (401 programs × 8 GPU profiles, the `mm_*` cells on all five faces), 162 725 distinct launches (rule-internal ones included): 0 outside, 0 round-trip differences, 0 persistent grids in the suite |
| the evaluator reads no fit, no table by version, no other part's value | test: perturbing each fitted `DeviceCost` field leaves every machine unchanged; every value not measured on the part is `derived` / `nominal` with an interval (cuda-ada 42 of 59 facts, rtx-4090 23/60, h100 38/76, h100-pcie 58/75, rtx-5090 30/61, rtx-5080 14/163, rtx-3090 26/56, a100 24/56) |

## C3 probes (RTX 5080, `place_calib.cu`, driver 620.73.04, 360 W)

| fact | measured | was |
| --- | --- | --- |
| start timing past the first wave | lockstep: a uniform grid's wave starts within 0–10 ‰ of a run (1 and 2 blocks an SM, 4 waves, 6 reps) | unmeasured |
| L2 keep, cyclic | 68 784 128 B (1.025 × 64 MiB; plateau 4130 GB/s, DRAM 905) | 68 786 585 (`l2keep.cu`) — agrees |
| L2 keep, rotated (a walk's readers on other SMs) | 68 784 128 B (plateau 3864 GB/s) | unmeasured |
| L2 keep, strided (256 B of every 1 KiB row) | 68 784 128 B (a 5 % dip at 0.94–1.0 ×) | unmeasured |
| hit against distance (cyclic) | 1.0 to 1.0 ×, 0.85 at 1.025, 0.48 at 1.05, 0.02 at 1.1, 0 past | step assumed |
| FFMA efficiency (chains a lane × warps a scheduler) | 1 chain: 0.22 / 0.43 / 0.75 at 1 / 2 / 4 warps; ≥ 4 chains: 0.83–0.98, 0.97 at ≥ 2 warps | the 5090's fit (eff_max 0.8) |
| `mma.sync` TF32 / bf16 efficiency | ≥ 0.99 from 2 warps a scheduler at any chain count; 0.91 at 1 warp, 1 chain | none |
| dependent latency | FFMA 4.56, `mma.m16n8k8.tf32` 35.2, `mma.m16n8k16.bf16` 35.2 clocks | nominal |
| sustained SM clock under FFMA load | 2 866 818 kHz (clock64 / %globaltimer, 4 s, 428 windows) | profile 2.45 GHz (E1's FFMA peak ÷ lanes, which folds the efficiency in) — kept for the rules; the machine reads the measured one |

## C3 replay score (`place_replay_scores`, 68 sites; picked / missed / open / null)

| family (sites) | model | rules |
| --- | --- | --- |
| walk (17: WALK_ARMS) | 12 / 2 / 1 / 2 | 13 / 2 / – / 2 |
| cut (21: mma_cut_replay) | 11 / 6 / 3 / 1 | 18 / 2 / – / 1 |
| c11 l2-gate (12) | 3 / 1 / 8 / 0 | 12 / 0 / – / 0 |
| ws (4: H100 wgmma-ws) | 0 / 4 / 0 / 0 | 4 / 0 / – / 0 |
| k-fill (4), streamk-fill (2), mma-depth (4), split-tile (2) | 12 / 0 / 0 / 0 | 12 / 0 / – / 0 |
| plan-cost losers (2) | 1 / 0 / 1 / 0 | 2 / 0 / – / 0 |
| **all** | **39 / 13 / 13 / 3** | **61 / 4 / – / 3** |

`model ≥ rules`: **not met** (39 against 61). The test asserts the model's floor (39) and the committed open list
(13 misses, each with its missing term); the 13 open picks flip inside marked facts (the keep of the 4090 and the
A100 / H100 placeholders, start timing on every part but the 5080) — the ADR's minimax at work, not a bug. Model wins
the rules miss: walk-6 / walk-7 (the 5080's refused walks, 512 × 2048 × 2048 and its transpose).

**Missing evaluator terms (what the 13 misses name):**

1. **Movers are not priced.** The H100 warp-specialized arms (ws-0..3, c11-7) are L2-bound at one price for TMA
   boxes and `cp.async` lane copies; the measured 3–25 % is the copy engine's. Next: a slot's panel movement priced by
   its mover — bulk copies at `copy_ns` × boxes over `in_flight`, lane copies as Global issue.
2. **The tensor tile's efficiency at its occupancy on the 3090 / A100 / 4090** (cut-0/1/2/8/9/10): the cut wins on
   wave quantization in the model, loses on the box. Only the 5080 has a measured curve; the others run on the latency
   rule (marked). Next: the `eff` probe on a rented part (C6) — the efficiency per warp tile × occupancy, and the
   cut's partial adds as a burst at each piece's end (atomics as a resource at its place, not spread).
3. **The walk's zero pass and pipe efficiency** (walk-16; the design check's open 5080 2048 × 6144 × 4096 line) and
   a deep ring's prologue priced where the hardware hides it (walk-5).

## plan-cost's losers

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
| `mapal_place::reuse::reads` (+ `lru::lru_bytes`) | a cache instance's missed bytes over a traversal by distinct bytes: lockstep exact (= LRU), the walk by the three-gap theorem, the hybrid, staggered starts |
| `place::schedule_of` / `launch_of`, `Emitter` (`Lowering`) | the rules' launch as a schedule and back (byte-identical); each present emitter's declared constraints, footprint, issue counts per class per trip, ring in flight and coordinate domains (D5 steps 2–3) |
| `profile::PlaceFacts` | a part's placement facts beyond `DeviceCost` (start timing, keeps per pattern, hit curve, L2 latency, efficiency points, latencies, sustained clock) and the provenance of every profile value |

## What C4–C5 need

- Search: `mapal_place::space::candidates(m, s, lows)` (steps 1–4, `crates/mapal-place/src/space.rs`) over
  `place::Emitter` (`crates/backends/llvm/src/place.rs`, `impl Lowering for Emitter`; `slots()` enumerates the
  coordinate domains depth-first, capped at 4096 — the branch and bound replaces the cap); rank by
  `mapal_place::evaluate` at `Bound::Value`, open where `Slow` / `Fast` disagree; admissible bound
  `mapal_place::lower_bound`.
- Lowering a pick: `place::launch_of(site, profile, contract, arms, &schedule)` → `Launch`; the rung `machine-place`
  goes where `gpu.rs` `launch_in` calls `launch` (the probe's hook points `launch_split` / `resized` show the two
  places every launch passes).
- The post-lowering gate: the declared registers (`Emitter::declare`) are the present emitters' own counts, capped
  by `.minnctapersm`; the naive kernel's 32 is unverified — `ptxas -v` must check it.
- Hybrid (C5): `Traversal::Hybrid` is evaluated and generated (`space::traversals`) but no lowering emits it yet
  (`Emitter::traversals` lists none).
