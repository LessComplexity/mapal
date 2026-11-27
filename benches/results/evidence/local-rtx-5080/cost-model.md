# cost-model — the pipeline calculation over measured primitives (ADR-0050, local RTX 5080)

The user's direction: the cost model is a **calculation over the primitives** — memory placement, movement,
compute-unit capabilities, result movement — with every parameter it needs, each measured or declared. This step
measures the primitives on this RTX 5080 in one calibration run, stores them as machine facts with provenance and
intervals, replaces the evaluator's max-over-resources with a pipeline calculation, and re-scores the replay. Spec:
`docs/decisions/ADR-0050-the-machine-model-placement-by-one-cost-model.md`; where step 1 stood: `machine-model.md`.
Raw output (never in git): `.claude/raw-evidence/cost-model/` — `calib-5080-final/place_calib.txt` (the run),
`replay-before.txt` / `replay-final.txt` (`PLACE_TERMS=1` reports), `r4/` / `r4b/`, `abba/` / `abba2/`, the test logs;
rows of `benches/results/evidence/RAW-manifest.tsv` with the asset `pending:raw-cost-model.tar.gz` (the upload,
`benches/gpu/archive_raw.sh`, is the user's step).

## 1. The calibration program (one run, every primitive)

`benches/local/place_calib.sh <out>` builds and runs `place_calib.cu` (the C3 probes and the new `prim` section)
and the optional modules of `place_calib_opt.cu`. A primitive the part may lack is built to PTX for the least
target that has it (fp8 `mma.sync` compute_89, TMA boxes compute_90, `wgmma` compute_90a) and loaded through the
driver's JIT: the part has it when its JIT accepts the module — a capability of the PTX ISA, never a part's name
(on this 5080: fp8 and TMA accepted, `wgmma` refused: `Instruction 'wgmma.fence' cannot be compiled for
architecture 'sm_120a'`). Each primitive is swept over what a unit (an SM) keeps in flight — warps × independent
uses a lane, boxes a block × blocks — three timed runs a point: **rate** the sweep's best (its min / max the
interval), **conc** the least in flight reaching 90 % of it (interval: the point below the knee .. the knee),
**lat** one use's latency (one lane, a dependent chain). Output: `prim name=` lines, then the calibration file's
`place.prim=` value (`benches/local/place_calib_facts.py` turns a run into the `place.*` keys and the Rust
`PlaceFacts` fields). The eff probe's points are now fractions of the sweep's own saturated rate (no part's row
rate in the probe), and the operation rows take the measured rate.

**RTX 5080 (driver 620.73.04, 360 W, 2026-11-27; sustained clock 2.879 GHz; every row `measured`):**

| primitive | rate (saturated) | latency (one use) | concurrency (90 %) |
| --- | --- | --- | --- |
| FFMA | 250 flop/clk/SM | 4.55 clk | 32 instr/SM |
| mma.sync m16n8k8 tf32 / m16n8k16 bf16 / f16 | 256 / 512 / 512 flop/clk/SM | 35.1 / 35.2 / 34.9 clk | 4 instr/SM |
| mma.sync m16n8k32 e4m3 (JIT) | 1024 flop/clk/SM | 35.2 clk | 4 instr/SM |
| wgmma (JIT) | refused: absent | – | – |
| ld.global 4 / 8 / 16 B, L2 | 52.3 / 52.4 / 56.1 B/ns/SM | 126 / 130 / 132 ns | 8 / 8 / 16 KiB/SM |
| ld.global 4 / 8 / 16 B, DRAM | 10.9 B/ns/SM (916 GB/s) | 306 – 313 ns | 4 KiB/SM |
| cp.async 4 / 8 / 16 B, L2 | 52.3 / 52.2 / 56.0 B/ns/SM | 140 ns | 16 / 8 / 16 KiB/SM |
| cp.async 4 / 8 / 16 B, DRAM | 10.9 B/ns/SM | 321 ns | 4 KiB/SM |
| TMA box 4 KiB / 16 KiB, L2 | 43.1 / 56.7 B/ns/SM | 263 / 400 ns | 16 / 32 KiB/SM |
| TMA box 4 KiB / 16 KiB, DRAM | 10.9 / 11.0 B/ns/SM | 262 / 400 ns | 8 / 16 KiB/SM |
| ld.shared 4 / 16 B | 63.9 / 254 B/clk/SM | 67.8 clk | 1 / 16 KiB/SM |
| ldmatrix.x4 | 288 B/clk/SM | 69.0 clk | 16 KiB/SM |
| st.global 4 – 16 B, DRAM / L2-resident | 10.9 / 28.7 B/ns/SM | 303 ns (store + fence + load) | 128 – 512 B/SM |
| red.add.f32: a lane its own address, contiguous / fragment rows / a sector a lane | 5.70 / 2.60 / 0.83 op/ns/SM | 138 ns (atom) | 512 / 256 / 64 op/SM |
| red.add.f32: a warp on one address / the grid on 32 | 0.73 / 0.18 op/ns/SM | 138 ns | – |
| epilogue: a 32 KiB tile staged in shared, stored | 10.9 B/ns/SM | 474 ns a tile (one block) | 32 KiB/SM |
| bar.sync 128 / 256 / 512 / 1024 lanes | – | 20.7 / 28.7 / 44.7 / 76.7 clk | – |
| mbarrier phase (256 lanes) / grid barrier (84 blocks) | – | 145.5 clk / 799 ns | – |
| launch: host issue / device gap between dependent launches | – | 1634 ns / 584 ns | – |
| panel stream from DRAM (run × rows a unit × bytes in flight; 60 rows) | run 32 B: 3.4 – 5.1, 64 B: 7.2 – 8.7, ≥ 128 B: 10.4 – 10.9 B/ns/SM | 321 ns | non-monotone in flight at short runs |

Little's law checks: lat × rate ≈ conc for the DRAM movers (3.4 KiB against 4 KiB) and the operations (FFMA 4.55 clk
× 250/64 instr ≈ 18 against 32 at 90 %); the L2 movers' loaded latency (conc / rate, 292 ns) is twice their one
use's (140 ns) — the knee the pipeline reads is the loaded one.

## 2. Facts

`mapal_place::Prim { name, of: PrimOf, per_clk, rate, latency, conc }` (each a `Fact` with its interval) on
`Machine::prims`; `PrimOf` keys a primitive by what it does and the place it reaches — `Op`, `Load`, `Copy`, `Bulk`,
`Panel {run, rows, fly}`, `Scratch {matrix}`, `Store {to}`, `Staged`, `Reduce {sharing, sectors}`, `Barrier
{lanes}`, `Phase`, `GridSync`, `Issue`, `Gap` — never an instruction's or a part's name. `Machine::cost(of, bound)`
reads the row, else the nearest measured row of its kind, else derives it from the place's reach / dispatch facts
and marks it. `profile::PlaceFacts::prim` (milli-units) and the calibration line's `place.prim=` carry them;
`PLACE_RTX_5080` holds this run's values (and its start, keeps, hit curve, eff points, latencies, clock), the
calibration file `calib/rtx-5080.calib` the same line (`native_gpu_is_the_query_and_the_calibration` checks
equality). Every other part: no `prim` rows — every primitive cost derived and marked (its picks open where they
flip).

## 3. The pipeline calculation (replaces max-over-resources)

`mapal_place::eval` — `Flow {work, rate, latency, fly}`, `steady`, `pipeline`:

- **a trip's steady state** = max over compute (the operation at its measured efficiency for the resident
  groups), issue (declared counts against the pipes), the scratchpad's fragment loads (`ldmatrix` bytes at its
  rate, its latency against the fragments in flight), and every mover: the panels into the unit (the unit's reach),
  the ring's latency term (the declared mover's loaded latency, L2 / DRAM by the miss share, × work ÷ the ring's
  bytes in flight — Little), the device cache's feed (the mover's rate × units), the memory (the misses at the
  mover's DRAM rate, plus the result's memory bytes), the result's throughput share;
- **a wave** = each piece's fill (its loaded latency) + trips × (steady + the declared barriers at their measured
  cost) + the drain (the shorter of a trip's compute and movement);
- **the result** (stores into the cache where the output fits the keep, else the memory; partial adds at the
  reductions' rate for the lowering's declared sector pattern — the warp tile's fragment rows 8 sectors, the FMA
  tile's lanes contiguous; past the keep their read-modify-write) is issued and left (`st`, `red`): a throughput
  beside the next waves; the phase waits only for the last wave's (a lockstep wave's whole result, a walk's blocks'
  last pieces: `1 / pieces` of it);
- **the grid** = the busiest unit's waves (whole waves, then the tail at its own residency) by the traversal; the
  hybrid's two phases; the zero pass, the workspace's reduce and the launches as before; never below the phase's
  memory bytes at the memory's rate.

Lowerings declare, per schedule, the new terms (`Declared`): `mover` (`Move::Load / Copy / Bulk` and its granule),
`scratch` / `scratch_fly` / `scratch_matrix`, `barriers` a trip, `store` granule, `add_sectors`, `epi_overlap` (false
for every present emitter).

**Tests.** `crates/mapal-place/tests/pipeline.rs`: Q1 a flow never slower with its rate or flight up, never faster
with its latency or work up, latency-bound exactly below the knee (2000 cases); Q2 the pipeline monotone in every
flow's rate and latency, the fill and the serial term (2000); Q3 with every ring unbounded and no fill / serial term
the steady state is the max of work ÷ rate exactly and the loop within one trip of trips × it (2000); Q4 the
evaluator on the toy machine with every primitive a measured row: monotone in each primitive's rate and latency,
never below the admissible bound (400). P1–P6, the search tests (B&B = exhaustive, P5), the backend's 148 lib
tests (replay score and band, coverage, native, OFF arm) and `machine_place` (R3, C5 values, picks within the bound)
pass; the workspace once at the end (release, `workspace-release.txt`): **1836 passed, 0 failed**, 11 ignored
(the CUDA differential harness capped by memory; the gate's `llc` refusals of candidates, as before, are stderr).

## 4. Replay (`place_replay_scores`, the search's verdict; 76 sites)

| | model picked / missed / open / null | rules picked / missed / null |
| --- | --- | --- |
| before (C5 evaluator, max over resources) | 37 / 6 / 30 / 3 | 59 / 14 / 3 |
| **after (the pipeline over measured primitives)** | **28 / 9 / 36 / 3** | 59 / 14 / 3 |

| family | before | after |
| --- | --- | --- |
| hyb (10) | 5 / 1 / 4 / 0 | 0 / 2 / 8 / 0 |
| walk (17) | 11 / 2 / 2 / 2 | 9 / 3 / 3 / 2 |
| cut (19) | 9 / 3 / 6 / 1 | 8 / 4 / 6 / 1 |
| c11 (12) | 5 / 0 / 7 / 0 | 5 / 0 / 7 / 0 |
| depth (4) | 1 / 0 / 3 / 0 | 0 / 0 / 4 / 0 |
| kfill (4), skfill (2), split (2), pc (2), ws (4) | 6 / 0 / 8 / 0 | 6 / 0 / 8 / 0 |

**Predicted against measured time per arm** (residual `log(measured / model)`; absolute on the µs-measured sites'
126 arms, pairwise `(model_b / model_a) / (measured_b / measured_a)` against each site's measured best, 88
pairs):

| | absolute: bias, median \|log\|, p80 | pairwise: median, p80 |
| --- | --- | --- |
| before, all | +0.097, 0.126 (× 1.13), 0.416 | 0.066, 0.177 |
| **after, all** | **+0.055, 0.107 (× 1.11), 0.409** | **0.088, 0.193** |
| before, RTX 5080 (92 / 56) | +0.115, 0.115, 0.374 | 0.050, 0.116 |
| **after, RTX 5080** | **+0.078, 0.083 (× 1.09), 0.332** | **0.058, 0.173** |
| after, other parts (no `prim` rows: derived, marked) | −0.062, 0.210, 0.502 | 0.122, 0.218 |

The absolute times moved toward the measurements — on this part where every primitive is measured, the median
error fell from 12 % to 8.7 % and the bias from +12 % to +8 % (the "3–12 × optimistic" arms were the relative-unit
rows and the DRAM-bound skinny site; the µs arms were 1.1–1.9 × under) — but the **ranking of near-equal arms got
worse** (pairwise 0.066 → 0.088), and so did the score: **`model ≥ rules` is not met (28 against 59)**. The band
re-derived (`PLACE_BAND_PM`): all 176 ‰, compute 143, DRAM 217, cache 247.

**Every remaining miss and the term it lacks** (`REPLAY_OPEN`, test-checked):

| site | measured winner | the model | the term it lacks |
| --- | --- | --- | --- |
| cut-0 (RTX 3090 2048³ TF32) | uncut by 5.9 % | cut 15 % under | the 3090's own primitives (reductions, stores, movers derived and marked): the rental's `prim` rows |
| cut-8 / -9 / -10 (A100) | cut by 9.6 / 25 / 8.3 % | uncut 3–6 % under | the L2's realized rate at the uncut grid's in-flight bytes (128 tiles on 108 SMs): the A100's `prim` sweep points |
| hyb-0 (5080 128 × 28672 × 4096 bf16) | hybrid by 2.7 % | every arm 2.9 × under (DRAM roofline) | the panel stream's measured rate by run (0.35–0.5 × at 32 B) would explain it — but c11-0's `-raster` arm streams 32-byte runs at 888 GB/s: the property separating the two is unidentified, so no term reads the run |
| hyb-1, walk-16 (5080) | hybrid / ordered by 3.6 / 12 % | the walk 2 % / 6 % under | the walk's own per-unit cost: its k loop one copy where the ordered kernel's is two (64 against 128 HMMA a body), its segment loop and fixup; its partial adds are now priced (fragment rows, measured 2.60 op/ns/SM) and overlap the loop but the last pieces' |
| walk-6 / -7 (5080 512 × 2048 × 2048 and transpose) | the walk by 12–15 % | the ordered 64 × 128 tile at its compute (49 against 57 µs measured) | a small warp tile's realized rate at two warps a scheduler (fragment loads' 69-clock latency against its products) — not separated by the measured primitives |

The 36 opens: 25 ties within the 2 % null (the model prices both arms alike: the c11 rasters on parts without
`prim` rows, the depth rung, the hybrid against the walk) and 11 flips at a corner of marked facts (every other
part's derived primitives).

What was tried and why it did not close (CLAUDE.md "open problems are attacked"):

1. **The panel stream by run** (`PrimOf::Panel`: DRAM rate by contiguous run, rows a unit streams, bytes in
   flight; 60 measured rows). It brought hyb-0 to within 1.1 × of its time but priced c11-0 `-raster` 2.7 × over
   its (also 32-byte-run) time; kept as a measured fact no term reads.
2. **Results as a serial burst at each wave's end** (the C3 open item for the 3090 cuts): priced the 5080's
   cut-17 (2048³ TF32, cut wins 5.5 %) 23 % dearer — the adds are issued and left (`red` returns nothing), so they
   overlap the next waves; the model now waits only for the last wave's.
3. **Reductions by the sectors an instruction touches** (fragment rows 2.2 × slower than contiguous, a sector a
   lane 6.9 ×): the warp tile's walks priced right in isolation (walk-6's walk 52.5 against 49.6 µs) but the FMA
   tile's lanes-`tn`-apart pattern revisits its sectors over `tn` instructions, which the sector-a-lane probe does
   not; the FMA tile takes the contiguous row.
4. **An MN-major panel** (each trip new rows): the FMA register-staged walks (skfill) realize 524–667 GB/s against
   the contiguous 916 — unmeasured (a probe whose trips advance the rows is the next one).

## 5. ABBA on this RTX 5080 (`machine-place` ON against OFF)

R4 (`place_diff.py`, the final evaluator): 35 suite programs, **9 with a moved kernel**, 19 the table alone, 7
equal; 40 placed sites: 28 kept, 12 replaced, 36 open. Timed (before the fix below: 11 moved): every program whose
host or device text differs (`gpu_changed.sh --on machine-place`: 30 programs + 2 cells), value gates first, the
arms interleaved and reversed on even repetitions (ABBA), 5 processes an arm, medians (`benches/gpu/probe_ab.py`;
raw `.claude/raw-evidence/cost-model/abba/`, `probe_ab.md`). The 19 table-only programs: ON / OFF 0.985–1.005
(null).

| program (moved kernel) | the search's replacement (R4) | OFF iter ms (min–max) | ON iter ms | ON / OFF iter | ON / OFF kernel | null / OFF |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| attn_1024_64 | FMA 64 × 32 × 32 split 2, 2 an SM (1024 × 64 × 1024) | 0.0360 (0.0332–0.0368) | 0.0329 | **0.912** | 0.745 | 0.998 |
| attn_1024_64, final band (also its 1024 × 1024 × 64 site: FMA 64 × 64 × 8 d5, 3 an SM) | | 0.0353 (0.0340–0.0369) | 0.0383 | 1.085 | 0.942 | 1.047 / 0.962 |
| indep_512 | FMA 64 × 64 × 32, 1 an SM | 0.0455 (0.0449–0.0573) | 0.0412 | **0.906** | 0.921 | 0.997 |
| causal_1024_256 | FMA 64 × 64 × 16 d4, 2 an SM (1024 × 256 × 256) | 12.81 (12.56–12.84) | 12.67 | 0.989 | 0.991 | 1.003 |
| gen_1024_512_256 | the same 1024 × 256 × 256 tile | 6.500 (6.401–6.611) | 6.476 | 0.996 | 0.988 | 0.992 |
| prefill_1024_512_256 | the same; FMA 64 × 32 × 32 and 128 × 64 × 16, 1 an SM | 6.475 (6.465–6.582) | 6.491 | 1.002 | 0.997 | 1.000 |
| mlp_512 | FMA 64 × 64 × 32, 1 an SM | 0.0404 (0.0388–0.0407) | 0.0438 | 1.086 | 0.942 | 1.007 |
| decode_1024_256 | the same 1024 × 256 × 256 tile; a 1 × 32 one-warp tile (1 × 1024 × 256) | 8.697 (8.597–8.940) | 10.366 | **1.192** | 1.043 | 1.003 |
| pj_m16_n2048_k7168 | FMA 16 × 256 × 32 split 7, 1 an SM | 0.0380 (0.0340–0.0384) | 0.0482 | **1.267** | 1.150 | 0.917 |
| smlp_4096_256_1024 | FMA 64 × 64 × **4**, one warp a slot, 9 an SM | 0.1611 (0.1566–0.1620) | 0.3570 | **2.216** | 2.322 | 1.008 |
| conv_2048_5, ewconvew_2048_5 | the cell kernel at 16 cells a thread | 0.0695 / 0.0810 | — | **VALUE-FAIL** | | 0.998 / 0.996 |

Two faster beyond the null (−9 %) at the timed band, four null, three slower (+19 %, +27 %, +122 %), and **two
wrong values**. The band's DRAM term re-derived after the run (315 → 217 ‰) replaces one more site of `attn`: re-timed
alone (`abba2/`, load 2.5), null (iter 1.085 against a null of 1.047; kernel 0.942). The default stays OFF.

- **The wrong values (fixed in this step).** The search's one-thread-per-cell pick at 16 cells a thread on the
  2048 × 2048 × 25 window site: the vector-cells kernel (`naive_cells`) reads `b` as one vector of a row's
  consecutive lanes at an affine offset a `k` step; the window's `b` is mixed-radix, so the kernel read the wrong
  elements (`got 99.47, -104.84` against `272.44, -14.51`). The rules never emit it there (`vec_fit` requires an
  affine site); the lowering never declared it. Now it does (`gpu::naive_cells_admit`, refused in the NAIVE
  lowering's `declare`; test `naive_vector_cells_refuse_a_window_site` on the program's own sites): both programs
  keep the rules' cell kernel on that site (R4 above: no longer moved).
- **The losses are the model's open terms, not noise:** every one replaces a tile with a smaller or one-warp FMA
  slot (`smlp`'s `bk = 4`, one warp, 9 slots an SM priced 44 against the rules' 54 µs; `pj_m16`'s split 7 at one
  slot an SM; `decode`'s one-warp 1 × 32 tile) — the FMA tile's realized rate at one warp a slot and short panels,
  which the measured primitives do not reach (the replay's skfill / kfill opens are the same family). The wins are
  the 64 × 64 / 64 × 32 tiles on under-filled grids (`attn`, `indep`).

## 6. What the rental round must measure

On each part (RTX 3090, A100, RTX 4090, H100, RTX 5090), the same one run and its facts:

```
git clone … && cd mapal && benches/local/place_calib.sh /tmp/calib            # nvcc + driver on the box; ~6 min
python3 benches/local/place_calib_facts.py /tmp/calib/place_calib.txt          # the calibration line's place.* keys
python3 benches/local/place_calib_facts.py /tmp/calib/place_calib.txt --rust   # the part's PlaceFacts fields
```

Its `prim` rows (fp8 where the JIT accepts compute_89 PTX, TMA compute_90, `wgmma` compute_90a — the H100's
warpgroup row and its TMA boxes against `cp.async`, the ws arms' missing mover term) become the part's
`PLACE_<PART>.prim`; `place_replay_scores` then re-scores its sites with measured costs (the cut / c11 / ws opens
and misses). Then the rented A/B of the R4-moved programs (`benches/local/gpu_changed.sh <out> --on machine-place
--arch <profile>` + the box's `gpu_run.sh`).

## Proposed CLAUDE.md facts-table rows (not applied — CLAUDE.md changes need the user)

| query | what it proves |
| --- | --- |
| `mapal_place::Prim` / `PrimOf`, `Machine::cost` | the cost model's primitives: per primitive (an operation, a load / copy / box from a place, a panel stream by run, rows and flight, scratchpad loads, stores to a place, reductions by sharing and sectors, barriers, phases, grid sync, a launch's issue and gap) its saturated rate, one use's latency and the concurrency that saturates it — each a measured `Fact` with its interval (`place_calib.cu prim`), else the nearest measured row of its kind, else derived and marked |
| `mapal_place::{Flow, steady, pipeline}` | the evaluator's pipeline calculation: a trip's steady state the max over compute and every mover of work ÷ rate, or latency × work ÷ in flight below the knee; a wave fill + trips × (steady + barriers) + drain; results issued and left, the last wave's waited on |
| `profile::PlaceFacts::prim` (`place.prim=`) | a part's measured primitives in the calibration line — the native profile carries them |
