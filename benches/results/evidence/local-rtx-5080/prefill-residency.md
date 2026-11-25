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
