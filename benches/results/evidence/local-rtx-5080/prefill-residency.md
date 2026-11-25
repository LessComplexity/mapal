# local-rtx-5080 — prefill residency: the prompt's cache writes on the device, one copy of the weights (2026-11-25)

Step `residency` of the prefill work on the readable Llama-3.2-1B-Instruct (`examples/llama.mapal`, real weights;
`full-model.md` is the starting point: TTFT 28.3 / 85.2 / 332.7 ms, decode 3.34–3.55 ms, peak 4.9–6.3 GB). RTX 5080,
batch 1, `run_mapal.py` (warmup 2 + 5 reps, medians), CPUs 24-31,56-63, `nice 10`, the bench lock. Raw output (not in
git): `.claude/raw-evidence/local-rtx-5080/prefill-residency/` (`summary.txt`, `before/` `after/` JSON and CUPTI
traces `kt512.tsv.gz`, `tools/`).

**Conditions.** The GPU was shared during these runs (other agents' processes at 12–54 % utilization outside the lock;
load average 50–260). Decode spreads are wide (3.41–3.90 ms). Earlier quiet single runs of the same builds gave
3.37–3.43 ms. TTFT spreads stayed within 1–7 % except at 2048. Before and after ran interleaved under the same
conditions.

**What "after" is.** Branch `prefill/residency`: d4dedf0d (zero-hoist) + d3fa6a86 (CSE value scope), built together with
the uncommitted `once-range` work another agent was writing in this same worktree at the time (straight-line range
maps as device cells, in place over a dead source; `tools/after_wip.diff`, md5 fcd7ee66). The TTFT gain needs both:
`once-range` runs the writes on the device; the zero-hoist fix makes that correct.

## Before → after (full model, 128 tokens generated)

| prompt | TTFT ms | decode ms/token (noisy) | peak MB | tokens vs HF f32 |
|---|---:|---:|---:|---|
| 128 | 29.6 → **16.1** | 3.58 → 3.49 | 4 882 → **3 046** | 128 / 128 |
| 512 | 86.9 → **39.8** | 3.57 → 3.58 | 5 086 → **3 262** | 128 / 128 |
| 2048 | 365.2 → **195.8** | 3.78 → 3.64 | 6 180 → **4 404** | 128 / 128 |

Competitors (`full-model.md`, same box): TTFT 4.5 / 10.2–13.4 / 44.9 ms; decode gpt-fast 2.95 / 2.97 / 3.05;
peak gpt-fast 3 532 MB at 512. Peak memory now sits below gpt-fast's.

**CUPTI, prompt 512** (`ktwin.py`: from the end of the weight uploads to the logits copy):

| | span ms | device busy | idle | kernels | kernel ms | copies |
|---|---:|---:|---:|---:|---:|---|
| before | 90.2 | 64.4 | **25.8** | 355 | 36.8 | 69, 713.5 MB, 27.6 ms |
| after | 39.4 | 37.6 | **1.8** | 388 | 37.3 | 5, 8.9 MB, 0.3 ms |

The prompt pass is now kernels end to end: 95 % of TTFT is device time. Item (4), the host gaps, closed as a
consequence of item (1): the 40 ms of idle device was the host running the range maps and their 21 MB uploads.

## What changed

1. **zero-hoist waits for the merged launch** (d4dedf0d, `func/zero.rs`). The earlier `once-range` WIP gave wrong
   tokens (12/128; first logit 18.20 against 18.40). Bisect: `--gpu-off=horizontal` or `arena` or `zero-hoist` each
   fixed it. Cause: the hoist walk found the last step touching a site output's bytes (arena sharing) at a
   `horizontal` member's **own** step, but the member launches at its group's last member. The hoisted zero
   landed between the two and zeroed bytes the deferred member still read: the prompt's `rotated_query` cell,
   merged with `rotated_key`, once the cache writes left the host and freed the rotate cells to merge. Fix: the
   walk takes each touching step's launch point (`plan.hz_of` → the group's last member in the sequence); a group
   launching at or after the site keeps the zero at the site. A pre-existing bug that the new placement exposed.
2. **CSE value scope** (d3fa6a86, `mapal-rewrite/src/graph_rewrites.rs`). Weights uploaded twice, with peak
   4.9–6.3 GB. The prompt's `forward<512>` and the decode step's `forward<1>` are two instances. Each unrolls its
   layer loop with its own constant `l`, so `model.layers[l]` was two `Index` objects, and CSE keys a constant by
   object (P1). The fact: an element view (an `Index` whose result is an array or record) is its parent's bytes,
   so equal views at equal constant indices are one value. Outside every guard arm's own work (`guard_plan`), the
   view keys its constant index by value class (the function's first constant of that type and value). Inside an
   arm it stays keyed by object: the reverted constant-matching attempt merged across arms. One object means one
   host view, one upload (2.47 GB instead of 4.4 GB), and one device copy; every projection merges after it.
   Narrowed on purpose: keying **every** op's constants by value also merged equal `iota`s and fills across the
   suite (60 `site_choice` golden rows renumbered, an h100 coverage row lost). That broader rewrite needs its own
   suite A/B.
3. **Calibration finding (no change).** `~/.cache/mapal/gpu-calib` does not exist on this box. Every `native-gpu`
   build here, `full-model.md`'s included, ran without the box's calibration line, so PDL, the zero-fold tail
   prices and the `calls` costs were all unmeasured. With
   `MAPAL_GPU_CALIB=benches/results/evidence/local-rtx-5080/calib/rtx-5080.calib`, PDL turns on and decode
   **worsens**: 3.368 ms uncalibrated, 3.545 calibrated, 3.605 calibrated with `--gpu-off=pdl` (quiet single runs).
   The calibrated placement moves three decode cells to the host: `; mapal.move … h2d=20484` per trip against 8196,
   each round trip a sync. That is a mixed result, so a fact is missing: the placement's byte rule prices a host
   cell between device kernels by its bytes, not by the pipeline drain its copies force. Not installed here (shared
   machine; it would change other agents' builds).

## Decode (item 3): the launches are not where the time goes

CUPTI, one decode step at 512 (`ktstep.py`), identical before and after:

| | |
|---|---|
| span / device busy / idle | 3.36 / 3.26 / **0.10 ms** |
| kernels | 323, 3.25 ms; **87 µs** between kernels |
| weight-streaming kernels | gate+up 16 × 76 µs (880 GB/s), down 16 × 39 (860), head 576 µs (912), q/o 32 × 10.5 (800), k/v+rotate 2 × 16 × 4.2 (≈500): **2.89 ms** for 2.47 GB |
| 226 kernels under 4 µs | **359 µs** (RMSNorm sums of squares, scales, rotations, residual adds, SiLU·up, cache appends, the attention's five passes a layer) |

The async launch path already keeps the device busy: idle is 0.1 ms a step, so CUDA-graph-style replay would save
at most the 87 µs of inter-kernel gaps. The gap to gpt-fast (2.95) has two parts:

- **The 226 small kernels' own time (0.36 ms).** Merging them at compile time is a `flow` / epilogue question, not a
  launch-recording one. The flow pass refuses the decode's groups with `phase`
  (`; mapal.flow.host … refused=after_attention->…:phase`, 145 groups), and `calls` forms no chain (`cost:66`,
  uncalibrated; `cost:80` calibrated). The RMSNorm scale is a per-row factor of the following projection
  (`fold_combine` `DepLaw::Factor`: applied once after the combine). The rotation, cache append and residual add
  are projection epilogues (`tile_plan_post`). Those facts already exist. The rule that reads them at one decode
  row is where the next step starts.
- **The small projections' DRAM rate.** k/v run at about 500 GB/s and q/o at 800 GB/s against 880–912 GB/s for
  the large ones. `horizontal` pairs `rotated_query+key` and `rotated_key+value` instead of merging q, k and v:
  the wave rule (`waves_merge`) refuses grids whose last waves do not fit one wave. For latency-bound GEMVs below
  `fly_bytes_per_sm`, merging pays even when the waves do not fit. That property is missing from the rule.

Not changed here, for want of time after items 1, 2 and the finding above. The two bullets are measured, not
attempted.

## Tests

- New: `zero_fold::zero_hoist_waits_for_merged_launches`: 48 generated straight-line programs (`ZeroProg.drop_sites`:
  earlier site outputs die at the cells reading them; two sites, two independent cells, a site), values against the
  interpreter and the OFF arm in every pick and serial mode. It covers hoisted zeros beside merged launches and
  **fails without the fix**.
- New: `unroll::prompt_and_steps_share_their_record_views`: 32 generated programs. The records are read from a file,
  and a size-generic `step<rows>` is called over a prompt and once per decode trip. Values equal the oracle, no two
  views of one array share a constant index, and the test **fails without the scope**.
- Run green: `cargo test --release -p mapal-rewrite -p mapal-lower -p mapal-interp` (all), and the backend's
  `zero_fold`, `site_choice` (goldens unchanged), `read`, `differential`, `devmap`, `caching_memory` and
  `horizontal`.
- With the `once-range` WIP in the tree, `zero_fold_on_generated_programs` loses its `none` coverage: `once-range`
  puts the straight-line generators on the device, as `once-bytes` did. The fix is to pin `once-range` off in the
  study's `emulated()`, beside `once-bytes`. Applied in the working tree but not committed, because the rung
  lands with that work. `placement::once_range_on_generated_prompts` (the WIP's own test) was failing at the time.
- Not run: the full 66-binary backend suite (the machine was loaded; a run started on the broader CSE form was
  stopped when the scope was narrowed). The `suite_diff` suite: d3fa6a86 leaves `rm_128_f64`'s emission byte-equal
  (checked), and the site_choice goldens pin 60 suite rows unchanged.

## Open

1. Decode: the 226 small kernels (0.36 ms) through `flow` at one row (`phase` refusals) and epilogues. The q/k/v
   merge needs a latency-bound property in the wave rule (above). The target is below 2.95 ms; the weights alone at
   880–910 GB/s take about 2.75 ms.
2. Prefill kernels: 37 ms of the 39.8 ms TTFT at 512 are the FFMA projections and attention (tensor-cores-2's split
   rule, `full-model.md` open 1). This step's movement work is done: the device is idle 1.8 ms.
3. The loop's cache init: carry-take into the decode loop is part of the `once-range` WIP (`bytes_enter`). It
   lands with that work.
4. The calibration finding (above): the placement price of a host cell between device kernels.
5. The broad CSE form (every constant keyed by value outside arms): it merges equal iotas and fills, and needs
   the suite A/B.

## For integration

CLAUDE.md facts table (proposed; CLAUDE.md is not this step's to edit):

- New row: `CSE value scope` (`mapal-rewrite` `value_scope`): an element view (an `Index` yielding an array or
  record) outside every guard arm's own work keys its constant index by value, so equal views are one object
  (one host view, one device copy). Inside an arm, keys stay by object.
- Backend row, `zero-hoist`: the hoist point is after the last *launch* touching the output's bytes; a
  `horizontal` member's launch is its group's last member.

`docs/tailored-fixes.md`: unchanged (no tailored fix).

---

# Step 2 — decode launches, prompt host gaps, calibration (2026-11-25, later)

Branch `prefill/residency` after 67f440c8: 30181082 (calibration: carry-induct, shared-upload, carry-take,
calib-supply), 6badc822 (carry-take follow-up), then `fly-unroll` (item 1). Same box, harness and conditions as
above (`run_mapal.py`-equivalent single processes under the bench lock, CPUs 24-31,56-63, `nice 10`). The GPU was
shared: other agents' processes, memory clock seen at 7 GHz between runs, a 4.8 GB foreign process during one
A/B. Every comparison below ran interleaved (ABBA, rotating order) in one sitting. Absolute values moved by
up to 10 % between sittings, so compare within a table only. Raw output (not in git):
`$SCRATCH/residency/` (per-build `kt.tsv` CUPTI traces, `ab.*` run lists, `tools`: `steplist.py`,
`stepgap.py`, `apistep.py`, `promptgaps.py`).

## Item 3 — calibration: why the calibrated build was slower, and the fix

With `MAPAL_GPU_CALIB=…/rtx-5080.calib`, decode measured 3.70 ms against 3.39 ms uncalibrated (this sitting).
The CUPTI step lists show three priced rules misfiring. In each case a fact was missing, and the priced
roofline (an isolated 11.1 µs `launch_ns` against the host's flops) filled the gap:

| finding (CUPTI, decode step at 512) | missing fact | fix |
|---|---|---|
| The decode attention's scores and P·V (32 tile sites a step) became nest cells: 6.4 + 1.5 + 8.7 µs a layer against 3.2 + 0.9 + 1.5 + 3.2 + 1.0. | The KV cache's device residence. `device_made` followed the carried view to its next state and stopped at the back-edge cycle, so the sites were not `fed`. | **carry-induct** (`device_made`, `carry-made` arm): a carried view whose loop enters with a device-made init, and whose next state the device makes from it, is device-made on every trip. |
| Layer 0's RMSNorm and K/V projections ran on the host. The device waited 2 × 82 µs for `h2d 2048` copies after host GEMVs: `h2d=20484` a trip against 8196. | The bytes of an upload another device reader needs anyway. The embedding row was charged to the norm's cells. | **shared-upload** (the placement `bytes` rule): an operand another device reader loads adds no bytes. |
| (Found while testing.) plan_cost's and row_block's generated programs panicked (`fold array ptr`, `init ptr`). Already broken at 67f440c8. | A taken loop init's other readers: a host fold before the loop, a second loop. | **carry-take**: only the taking loop's init pair (and the state's way out) belongs to the carry. Other readers get the bytes copied out. |

Not changed: `cell_roofline` counts a nest cell's nested fold as one element per cell (a one-row projection
priced at 6 KB instead of its 2 MB matrix). In the Llama, the two facts above decide every such cell first.
No generated program found where the roofline alone decides, so the change is not shipped. It is open below.

**Calibration supply** (`calib-supply`, `profile.rs`): `native-gpu` reads the line for its device key from
`MAPAL_GPU_CALIB` or the user cache, else from the lines the repository commits
(`COMMITTED_CALIB`, keyed by physical facts). This box is priced with no manual step and nothing is installed
under `~/.cache`. With no line, the build warns on stderr and never falls back silently.
`MAPAL_GPU_CALIB=none` is the uncalibrated measurement arm. `; mapal.host … calib=file|committed|none`.

Decode, ABBA, 10 rounds, prompt 512 (ms/token, median [min–max]):

| build | uncalibrated | calibrated |
|---|---:|---:|
| 67f440c8 | 3.391 [3.385–3.615] | 3.70 (single runs; 3.55 in step 1) |
| 30181082 | 3.389 [3.384–3.593] | **3.329** [3.322–3.456] |

Calibrated now beats uncalibrated: PDL is on, and the between-kernel gaps fall from 86 to 13–14 µs a step
(CUPTI). Placement: `device_sites=176 host_sites=0` and `h2d=8196` a trip on both arms. TTFT at 512:
39.2 → 38.0 ms. The prompt's layer-0 norm moved to the device (shared-upload); no copy was added.

## Item 1 — decode launches

The launches per step did not drop: 323 kernels (CUPTI) before and after. Where the step's time goes
(final build, CUPTI, one step at 512): 3.24 ms of kernels, 13 µs between kernels, and a host turnaround of
126 µs from the logits copy to the next step's first kernel (75 µs of host work between the API calls: the
argmax over 128 256 logits, the print, the embedding row).

**Landed: `fly-unroll`** (OFF arm `--gpu-off=fly-unroll`). A warp-a-cell fold whose grid holds too few warps an
SM to keep the part's stream bytes in flight (`Gpu::stream_fly_per_sm`, 8 KiB on the 5080) unrolls its lane
loop deeper. The depth is the power of two Little's law asks at the narrowest operand's element, in
`[NAIVE_UNROLL, 32]`. Each variant gets its own body name (`_lanes_u<n>`) and loop count. In the Llama
decode: the RMSNorm sums of squares (1 block) go 2.0 → 1.4 µs (×33), and the K/V GEMVs get 32 loads in flight.
Their time stays 4.0 / 4.2 µs: they are bound by request count (2-byte bf16 loads, 64 B a warp
instruction), not by bytes in flight.

| ABBA, 8 rounds, calibrated | decode ms/token median [min–max] |
|---|---:|
| 30181082 state (without fly-unroll) | 3.343 [3.326–3.353] |
| + fly-unroll | **3.314** [3.308–3.322] |

Suite: `gpu_changed.sh --off fly-unroll` finds 3 of 35 programs whose text differs (`pt_causal_1024_256`,
`pt_prefill_1024_512_256`, `pt_gen_1024_512_256`) and 0 cells. Timed by `gpu_run.sh` (probe section ABBA,
5 reps, value gates pass; `$SCRATCH/residency/gc_fly/results`), as on / off-fly-unroll medians in ms:
causal 15.34 / 15.65 (null 3.1 %), gen 8.19 / 9.40 (−13 %, null 3.7 %), prefill 9.65 / 9.56 (null 2.1 %).
Emission is unchanged on 32 programs; one is better beyond null and two are within it.

**Measured, not landed: merging q/k/v.** An experiment that only moved `fn rotate`'s definition below
`fn project` in the source (not committed: the program is not bent to the compiler) let `horizontal` merge
q+k+v (384 blocks, 16.4 µs against 10.5 + 4.0 + 4.3) and rotate_q+rotate_k. That gave 323 → 307 kernels and
−50 µs of kernel time a step. Cause: `topo_order` keys an inlined callee's objects by the callee's source
positions, so every inlined `rotate` sorts before every inlined `project`, and the walk interleaves
q, rotate_q, k, rotate_k, v. `horizontal` groups greedily in walk order, so each dependent step closes the
group. The generic fix is the walk order of inlined code: call-site position first, as the unrewritten
program evaluates it. It touches every program with inlined calls (goldens, suite), so it needs its own A/B.

**Why below gpt-fast is out of reach on this path.** The weight GEMVs already run at 860–912 GB/s: 2.88 ms of
the 3.24 ms of kernels against 2.78 ms for 2.47 GB at the measured 890 GB/s. The other 0.36 ms are about 240
kernels of 0.8–2.6 µs each. With PDL, a boundary costs about what a grid barrier does (`calls` refuses every
chain on cost, correctly: `pdl_launch_ns=486` against `gsync_ns=553` + per block). So merging launches alone
saves little. Removing phases, and overlapping the idle DRAM with prefetches of the next weights, is what
would pay.

## Item 2 — prompt host gaps

The prompt pass at 512 is 4 copies, 4.7 MB, with 1.6–1.7 ms of device idle in a 39–44 ms span (CUPTI). Nearly
all of it is one host chain: the 4 MB embedding gather for the prompt's rows (0.8 ms, then the upload).
Its index comes from a file, so it is not proven in bounds, and the gather stays on the host. Layer 0's norm
no longer follows it there (shared-upload). Per the integration step, the device-side index check (a device
trap flag) belongs to the bf16 sprint's "bf16-class" agent. Nothing else in the window exceeds 20 µs.

## Tokens and TTFT (final build, calibrated by the committed line)

| prompt | TTFT ms | decode ms/token (this sitting) | tokens vs HF f32 |
|---|---:|---:|---|
| 128 | 15.7–16.5 | 3.62–3.67 | 128 / 128 |
| 512 | 38.7 (quiet) / 42–43 (busy) | 3.30 (quiet) / 3.60–3.69 (busy) | 128 / 128 |
| 2048 | 204–207 | 3.83 | 128 / 128 |

## Tests

- New: `placement::priced_placement_follows_the_carried_state`. 16 generated `testgen::CarryProg` loops: a
  carried array made by a large site, rewritten each trip through a small site, with an optional second
  carry, and optionally a host-gathered row with its norm and a bf16 one-row projection. Placed against
  x86-64-v3's measured host. It asserts no site on the host, no state crossing, and at most the row uploaded
  a trip, against the interpreter in every emulator mode and with `carry-made` off. **It fails without
  carry-induct and without shared-upload.**
- New: `profile::calib_supply_tests` (committed lines parse and resolve for their keys; a miss is named).
- Extended: `placement_on_generated_programs` counts `fly-unrolled-loops` (decode generator `h = 1024`), and
  the OFF-arm rotation includes `fly-unroll`.
- Green: the backend's full suite (59 binaries) on the item-3 state, then `--lib`, placement, plan_cost
  (green again), row_block (green again) and caching_memory on the final state.

## Open

1. **Walk order of inlined calls** (q/k/v merge, −50 µs a step measured): key inlined objects by call site
   first. Needs a suite A/B.
2. **Narrow GEMV requests**: k/v at 500 GB/s and q/o at 807 GB/s load bf16 2 bytes a lane. 16-byte vector
   loads per lane (`TileRead` unit stride, regrouped lanes) cut requests 8×. The `fly-unroll` result shows
   bytes in flight are not the bound.
3. **Host turnaround, 75–126 µs a step**: the greedy argmax runs on the host over the 513 KB of logits.
   It needs a keyed-select fold fact (associative with left priority on ties), on the device or regrouped
   on the host.
4. **Queued launch price** (integration's point): `device_pays` prices an isolated 11.1 µs launch. Behind
   other device work, a launch costs `gap_ns` (260 ns). The Llama's case is now decided by facts, so this
   rule change was not made. It needs a definition of "queued" from the sequence's device work, and a suite
   A/B.
5. **`cell_roofline`**: a nest cell's nested reads should count its walked arrays' bytes. Not shipped, for
   want of a generated case that it decides (see item 3).
6. **Calibration measured on first use**: a `mapal calibrate` that runs `benches/gpu/calib.cu` and writes the
   user cache line. Only part of a line comes from `calib.cu` (launch, issue, gap, zero, l2keep and tma come
   from other tools), so the committed line remains the supply for measured parts.

No tailored fixes (`docs/tailored-fixes.md` unchanged).
