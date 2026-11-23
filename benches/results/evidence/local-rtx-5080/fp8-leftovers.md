# fp8 + leftovers (RTX 5080, sm_120; branch `sprint/fp8-leftovers` from `55ba3e42`)

Design: `.claude/sprints/fp8-leftovers/design.md` (§1 fp8, §2 retain-call, §3 calls-issue, §4 tune). Raw run output:
`.claude/raw-evidence/local-rtx-5080/fp8-leftovers/` (ABBA logs, CUPTI `ktrace` TSVs, suite identity TSVs, trial
wisdom). Timing: one process per run, `MAPAL_PTX` + a wisdom written by one untimed tune run, `MAPAL_GPU_TUNE=0`,
4 interleaved ABBA repetitions per row under the bench lock, `taskset -c 0-7,32-39`; ms are a run's 16 decode steps
(`iter ms`), "ms/step" = /16. Null: the spread of one arm's 4 runs (≤ 0.5 % on every decode row but B32, ≤ 5 %).

## What landed

| item | fact / property / capability / rule | OFF arm | commit |
| --- | --- | --- | --- |
| fp8 storage | `NarrowFloat::{F8e4m3, F8e5m2}` (1 B); `NarrowForm::{saturates, least, embeds_in}`; the oracle's saturating `round` / exact `widen`; the host integer sequence `f8_from_f32` | (storage: none; tiles: `narrow-tile`, `narrow-async`, `vec-cells`) | 073d48ee |
| per-operand units (F5) | `bound_coef_units`, `TileSite::units` (an operand embedded exactly in the member: `u_t = 0`) | — (a bound, no emission) | 073d48ee |
| C1 | `Caps::cvt_f8` (sm_89+ from `caps_of_cc`): `cvt.rn.satfinite.{e4m3,e5m2}x2.f32` roundings, `cvt.rn.f16x2.e4m3x2` pair widening (inline PTX: LLVM 23 selects the intrinsics only from PTX 8.1); the integer forms elsewhere | a part without it | 073d48ee, 466dc111 |
| split post table | `make_table`: a `tile-post` group whose candidates collapse to one split keeps it (one-candidate table, no trial) | `--gpu-off=splitk` | 466dc111 |
| calls-issue (§3) | `chain_ns`: the separate form as the host issues it (unit `k` issued at `k·issue_ns`, started at the later of its predecessor's end + launch gap and its issue); `issue_ns` the host's measured property | `--gpu-off=calls-issue` | 462de6f2 |
| tune-context (§4, M1+M2) | `MAPAL_SITE_COLD` (bit 3) from `residence_plan.at(o, l2_bytes) == Dram`; trial timed by device events, a ZEROED site's zero outside the timed region, a COLD site's L2 evicted before each timed run | `MAPAL_GPU_TRIAL=isolated` / `nocold` (runtime flags) | 2ce97ed0 |
| retain-call fact (§2) | `CategoryIr::call_keep` (stored): per `Call` site, the callee slots its loops keep and the callee values kept with them | — (no consumer yet) | e18c821f |
| programs | `gen_tiles.py w8`: `pt_llm_dec_b{1,8,32}_l{2048,8192}_f8w`, `pt_llm_blkdec_b1_l4096_f8w`, px twins — f8e4m3 weights with **per-output-channel scales as dataflow** (two whole max folds a column, `w / s[j] -> to_f8e4m3`, the scale in each projection's epilogue) | | 462de6f2 |

**Identity.** `suite_diff.py --base-emit` (the `55ba3e42` emitter) over all 297 `benches/algorithms` programs and the
two cells × 5 faces: after the fp8 storage commit 307/307 same; at the final commit 305 same, 2 differ —
`px_rt_mlp2_128`, `px_rg_mlp2_128` (calls-issue forms their 4-unit chain: 0.3–0.4 µs members against a 1.69 µs
issue; untimed interpreter twins, value-gated on the device: exact face = interpreter = OFF arm). The COLD bit
changes the host text of every program with a DRAM-resident site operand (a flags constant; wisdom keys move, so
AUTO re-trials) and the runtime changed (`mapal_gpu.c`): the integration run's full suite times every program.

## Tests

| test | programs | checks | coverage (all > 0) |
| --- | --- | --- | --- |
| `mapal-ir narrow` (5 new) | every fp8 pattern; ties; saturation (448, 464, ∞; 57344, 61440); subnormals; every 97th f32 | oracle = integer sequence = an independent nearest search (every 5917th); the f16 widening path; `embeds_in` table | — |
| `mapal-ir precision::per_operand_units` | k 1/64/4096 | F5 = shared form at equal units; embedded operand drops its factor | — |
| `tests/f8.rs` facts | 1 | saturating forms; 1-byte narrow site; host sequence / device `cvt` / 3090 integer per side | — |
| `tests/f8.rs` generated (`testgen::F8Prog`) | 48 | exact face bytes: host; emulated device × 4 emulator modes; `narrow-tile` OFF, `async` forced, `vec-cells` OFF; the 3090 profile; default face within `2·coef·M` + D3 flips × 4 modes and `narrow-tile` OFF | e4m3 35, e5m2 13, fp8 `a` 20, `n×k` 21, per-tensor 18, per-column 11, Factor 8, member sets 23, saturated values 95, subnormal 14, loops 18, narrow sites 40, device fp8 tiles 10, fp8 cells 48 |
| `calls.rs calls_issue_is_monotone_on_generated_programs` | 48 (`HzProg`) | formed chains non-decreasing in `issue_ns` (1690 → 4000 → 8000 → 30000); ON ⊇ OFF, equal text where no decision moved | chains 43, newly formed 29, refused at every issue 14 |
| `calls::host_issue_prices_the_separate_form` | unit | identity at 0; monotone; long members hide it; the decode-like chain refused at 1690, formed at 12000 | — |
| `site_choice.rs cold_flag_follows_residence` | 4 | COLD exactly where `residence_plan` says DRAM (5080 64 MiB L2 vs 5090 96 MiB) | — |
| `caching_memory.rs call_keep_marks_the_slots_a_loop_keeps` | 1 | kept / `same` / rebuilt / outside-loop / nested | — |
| regressions | `half_tile` (bound now F5's), `narrow`, `calls` lib, `mapal-ir` lib (91) | pass | — |

## Rows (RTX 5080, ms per decode step; the bf16 legs are the precision's reference, no fp8 leg exists)

f32, bf16w and the OFF arm: one ABBA batch per row (`abba_rows.txt`, the integer widening's f8w beside them:
1.050 / 3.277 / 5.070 / 15.27 / 15.26 / 56.30 / 6.77 ms per 16 steps); f8w ON: the C1 batch (`abba_cvt.txt`, against the
integer form), blkdec's AUTO with the cold trial's wisdom (`trial_bd8c.txt`).

| workload | f32 | bf16w | f8w OFF (`narrow-tile`) | **f8w ON** | fastest competitor (leg, matrix 3a8e654f) | explanation |
| --- | --- | --- | --- | --- | --- | --- |
| dec B1 L2048 | 0.2405 | 0.1501 | 0.3996 | **0.0558** | bf16: 0.1448 (torch-bf16-compile) | weights 42 MB at 1 B (CUPTI: qkv 19.5 µs, wo 14.9 µs a step against bf16w's 59.4 / 39.5): 42 MB of weights + 17 MB of cache fit the 64 MiB L2 across steps (qkv at 1.29 TB/s > DRAM) |
| dec B1 L8192 | 0.338 | 0.2409 | 0.786 | **0.2013** | 0.1826 (torch-bf16-compile) | the f32 cache (67 MB) dominates; weights halved |
| dec B8 L2048 | 0.4425 | 0.3486 | 0.8424 | **0.298** | 0.218 (torch-bf16-compile) | weights halved; attention unchanged |
| dec B8 L8192 | 1.0625 | 0.9736 | 1.486 | **0.950** | 0.4728 (torch-bf16-cudnn) | f32 cache-bound (`_f8kv` not built: R5 open) |
| dec B32 L2048 | 1.030 | 0.952 | 2.27 | 0.95 (null) | 0.4584 (torch-bf16-compile) | f32 attention-bound |
| dec B32 L8192 | 3.517 | 3.4375 | 4.83 | 3.52 (1.02× bf16w) | 1.416 (torch-bf16-cudnn) | **open**: the step's top kernel (scores, 2.0 ms) is the same text in both programs (diff: function numbering only) and runs +4.6 % in the f8w process (CUPTI); unexplained |
| blkdec B1 L4096 | 1.074 | 0.6456 | 2.70 | **0.370** | 0.5483 (torch-bf16-compile) | FFN GEMVs 58.7 MB at 1 B: 76–92 µs a step against bf16w's 132–168 |

The OFF arm (`narrow-tile`): the fp8 products as per-cell kernels (one `LDG.U8` an element, no split): 4–7× the ON
time on B1, the cost of the tile path the fp8 operand takes.

**C1 widening (466dc111)**, cvt vs integer form, ms / 16 steps: dec B1 L2048 0.892 / 1.047, blkdec 6.02 / 6.77, B8 L2048
4.77 / 5.07, B8 L8192 15.2 / 16.0, B1 L8192 3.22 / 3.28, B32 null. SASS (blkdec FFN GEMV, `cuobjdump -sass`): 32
`F2FP.F16.E4M3.UNPACK_B` + 32 `HADD2.F32` per 32 FFMA, no `LOP3`/`I2F` chain (integer form: 25 LOP3 + 12 I2F per 12
FFMA). The vector cells take 16 fp8 lanes per `LDG.E.128` (`vec-cells`, plan line `cells=16`).

**Split post table (466dc111)**: the f8w `wo` projection (`u·s[j] + x[e]` after the fold) had one candidate
(`splitk16`) — and no table, so it ran as its group's ordered kernel (16 one-cell blocks, `LDG.U8`): 108 µs a step
(CUPTI); with the table 14.9 µs. dec B1 L2048 f8w 6.39 → 1.05 ms / 16 steps (before the cvt widening).

**tune-context (2ce97ed0)**: the trial into its own wisdom under `isolated` and the default, then 4 ABBA runs of the
program reading each (ms / 16 steps): dec B8 L2048 bf16w **5.040 vs 5.490** (two picks changed: the isolated trial's
candidates measured 33–49 µs warm, cold 80–157 µs), blkdec f8w 5.926 vs 6.023, blkdec f32 17.197 vs 17.281, mlp2_2048
1.939 vs 1.946 (null, same picks). The f8w FFN GEMV's trial: 21 µs warm (L2) against 75–77 µs cold — CUPTI in the
program 76–92 µs: the cold trial measures the program's rate.

## Open problems attacked

| problem | result |
| --- | --- |
| fp8 storage (F1–F5, C1, R4) | **closed** on the 5080: storage, conversions, GEMV / FMA tile over 1-byte operands, decode rows |
| the f8w `wo` projection lost its split (found here) | **closed generically**: a one-candidate split post table (no program but fp8 moved) |
| integer fp8 widening ALU share (found here) | **closed** where the part has `cvt.rn.f16x2.e4m3x2` (C1); e5m2's form is already a shift |
| calls-issue (§3) | **closed** as a rule; the per-boundary form of the design over-formed the 5080's causal chain (units 7.1, 1.1 µs then three tiny maps priced idle behind a host 7 µs ahead) — replaced by the issue queue; the measured `work_slack_pm` the design proposed is not needed for any 5080 chain (none forms); rented boxes' `issue_ns` (3090 / A100 / H100 have none) and the 20-chain replay: next round |
| tune M1 / M2 (§4) | **closed** (cold eviction, event timing, zero outside); the in-context ABBA confirmation (M3, sites in loops) **attempted-open**: not built — M1+M2 already move the two 5080 picks the cold trial explains; the 4090's power remainder needs the capped part |
| retain-call (§2) | **attempted-open**: the fact (`call_keep`) is built and tested; the consumer (a retaining entry per mask, keep block in the caller's frame, `emit_cell_once` keyed by the site's `made` bit) is not — no suite program has a call in a loop |
| fp8 tensor-core tiles (R1–R3, P1, P2) | **attempted-open**: `half_operands` still matches formats (fp8 never takes an MMA tile; F4 `embeds_in` and F5 are in place for R1); the f8 / embed MMA kernels, `mma_calib.cu rate` (P1) and the emulator's m16n8k32 were not reached. `_f8` GEMM rows and `_f8kv` (R5, gated on the vector digit copy) not built |
| B32 L8192 f8w +2.4 % vs bf16w | **open**: identical attention kernel text, +4.6 % in the f8w process (memory placement after the f32 weights die is the remaining candidate) |
| fp8 competitor legs | torchao is not installed beside torch 2.11 (not installed into the shared venv); `torch._scaled_mm` needs fp8 activations (`_f8` rows, not built): reported against the bf16 legs |

## For integration

README line (`benches/results/evidence/local-rtx-5080/README.md`):

```
- [fp8-leftovers.md](fp8-leftovers.md) — fp8 storage (f8e4m3 / f8e5m2, saturating, C1 cvt), `_f8w` decode rows (B1 L2048 0.056 ms/step, 0.37× bf16w; blkdec 0.37 ms, 0.57×), the one-candidate split post table, calls-issue (issue queue), tune-context (cold / event trials: dec B8 L2048 bf16w 0.92×), the `call_keep` fact; open: fp8 MMA tiles, retain-call's consumer
```

CLAUDE.md facts-table rows:

```
| `call_keep` (`CallKeep`: `kept`, `values`, `loops`) | retain-call: per `Call` site, the callee input slots whose argument is the same at every execution (made outside every loop containing the site, or `unchanged_plan.same`) and the callee values all of whose slots are kept (`unchanged_plan(callee).inputs`) — equal on every execution after the first |
```

and in the `precision_plan` row: "`narrow` … each `to_bf16` / `to_f16` / `to_f8e4m3` / `to_f8e5m2` rounding's
`NarrowForm` (fp8: saturating; `NarrowForm::embeds_in`: format A's values are exactly B's — the tensor-core admission
over narrow storage); `bound` with one unit per multiplicand (`bound_coef_units`, `TileSite::units`)"; in the backend
`Gpu` properties row: "`Caps::cvt_f8`", and "`MAPAL_SITE_COLD` (a site operand DRAM-resident: cold trials)".
