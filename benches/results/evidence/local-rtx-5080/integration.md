# integration — regressions + f32-tensor-cores + hopper + fp8-leftovers on one branch

Branch `local-rtx-5080`: `370637ed` (regressions) + `--no-ff` merges of `sprint/f32-tensor-cores` (`dfdcb5e1`),
`sprint/hopper` (`3092b270`) and `sprint/fp8-leftovers` (`ad451a13`), then the integration commit (the combined
one-candidate rule as a rung, the evidence-tool fixes the full suite found, the "For integration" sections: `851de1ba`),
then the two regressions the full GPU suite found (the tensor-map lookup on the launch path, the trial's cold flag at the
L2's edge). Raw: `.claude/raw-evidence/local-rtx-5080/integration/` (`tests*.log`, `resS/` `resM/` the first run,
`resS-final/` `resM-final/` the final code's, `keep-ab/`, `abba/`, `alone_norm.tsv`).

**Run in parallel (the user asked for speed).** The workspace suite ran pinned to CCDs 2–3 (`16-31,48-63`,
`RUST_TEST_THREADS=16 MAPAL_PAR=4`, 45 min) while the two GPU bundles built on CCD 1, then the first timed runs
(`model_scale_run.sh`, the bench lock, timed processes pinned to CCD 1, `CPUS=8-15`) overlapped the suite's second
half; a thermal guard stopped the suite's process tree above 85 C Tctl (26 + 33 stops) and continued it below 78 C,
never the timed run. The overlap cost nothing it measured: the session-controlled ABBA below separates the code from
the machine. The final runs (`resS-final`, `resM-final`) ran on an idle machine (load 1.2–1.4).

## Merge notes

| merge | conflicts | resolution |
| --- | --- | --- |
| f32-tensor-cores | rung table size; replay `scalar=` / `split=` conditions; replay counts; `rung_replay.tsv` lines | both rungs' names (123); the replay reads both conditions; counts add both steps' lines: 116 / 68 / 9 / 18 + regressions (6 / 3 / 1 / 1) + f32 (6 / 0 / 0 / 0) = **128 / 71 / 10 / 19** |
| hopper | rung table; `MatrixOp` (`acc_g` vs `flops_per_clk`); `DeviceCost::NAMES`; emulator (accumulator datapath vs wgmma model); generators | both sides kept (126 rungs, 42 cost names, both emulator models, both generators) — and **one matrix rate**: hopper's per-instruction `MatrixOp::flops_per_clk` (H100 datasheet) and f32's per-part `Gpu::mma_{tf32,half}_per_clk` / `wgmma_tf32_per_clk` (queried by SM design, measured by `calib.cu matrix`) were the same property twice; `stage_consume_ns` now reads `Gpu::matrix_flops_per_clk` and the per-op field is gone (H100 wgmma TF32 2048 in both, the 5080's warp TF32 256 = its FFMA rate in both: R1's price unchanged) |
| fp8-leftovers | rung table; `Caps` (`reg_realloc` vs `cvt_f8`); mapal-ir exports; c0 golden comment; **the one-candidate epilogue split** | both kept (127 → 128 with `post-table`); see below |

**The same rule, fixed twice.** regressions item 1 (any epilogue group whose one candidate splits launches through a
one-candidate table) and fp8-leftovers (`tile-post` groups only, and `--cands=1` keeps the ordered kernel) fixed the
same launch. The full suite then failed `gemv_stream`'s `post_split_launches_through_its_table` (cands 1) and
`generated_programs_keep_values_on_every_arm` (`post_split` 35 → 0: the generated programs check the rules' plan,
`--cands=1`), because fp8's gate made `--cands=1` drop the planned split. Both intents hold with the condition as
its own rung: **`post-table`** — on by rule everywhere (a realization fix, no hardware condition), `--cands=1` keeps
it (candidate 0 is the rules' plan, and the table is its only launch path: the plain path runs an epilogue group's
ordered kernel), **OFF arm `--gpu-off=post-table`** = the emission before (fp8's measured OFF: f8w decode `wo`
108 vs 39 µs a step; the H100 B1 decode 235 vs 29 µs). `first()` instead of `cands[0]`.

**Found by the merge: narrow1 over fp8 operands.** f32-tensor-cores' `narrow1` admitted any two operands of one
narrow format; fp8 storage made fp8 × fp8 `{f32}` sites reach it under the ON arm (by rule the rate check refused:
no fp8 rate is given) — the split kernel issues m16n8k16 with two-byte fragments, fp8's warp product is m16n8k32.
narrow1 now takes a format only where its warp product has the 16-bit geometry (`MMA_SYNC_16`, a capability as
geometry); `split_tile` checks both fp8 formats on the rtx-5080 and h100 by rule and under the ON arm (it fails
without the check: `rtx-5080 fp8 store 4 ["split-tile"]`).

**Evidence tooling the full suite exposed** (each step's own bundles timed only the programs it moved):

| intent / fact | symptom on the integrated bundles | fix |
| --- | --- | --- |
| `arch-rungs:on` (hopper) | 101 + 34 kernels "rungs not the rules'": `tma-digits` in the plan's `rungs_on`, absent from `evidence.py`'s `KERNEL_RUNGS` | listed |
| `mma-combine:off` (f32) | the shipped mlp_2048_tf32 / mlp2 cut tiles judged by the OFF check: the site-priced rung is not in a plan's `rungs_on` | `split-tile`, `mma-combine` gated by the plan line (`applies`), as `half-tile` |
| `mma-combine:on` RED (f32) | `sass.red = 0`: sm_120 spells the partials' add `REDG` (128 in the kernel) | `sass.red + sass.redg > 0` |
| `zero-fold:on` (fp8 tune-context) | 35 "folded site not flagged": flags 9 / 11 (COLD = 8 beside ZEROED = 2) missed by `i32 [23]` | the ZEROED bit read as a bit |
| `narrow-tile:on` (regressions) | 189 naive narrow kernels: `vec-cells`' 8 bf16 cells a thread now load 16 bytes (LDG.128 under the allocation's alignment), the check knew LDG.U16 only for one thread a cell | LDG.128 where `cells` > 1 |

## Tests

- **Full workspace suite** (`cargo test --release --workspace`, once): 1519 passed, 3 failed, 7 ignored — the three
  merge failures above (`gemv_stream` ×2: the one-candidate gate; `site_choice` c0 golden: hopper's `wgmma-ws` on
  h100-pcie). After the fixes, the 21 binaries they and the `--cands=1` users touch (lib, gemv_stream, site_choice,
  split_tile, f8, calls, bulk_stage, devmap, epilogue, golden_ll, gpu_host, hardware_properties, hopper_tma, large_gemm,
  plan_cost, radix, resident, rowfold, site_geometry, splitk, streams): **285 passed, 0 failed**; after the L2-keep
  property: lib 121, site_choice 9, golden_ll 36, caching_memory 11, hardware_properties 2, gpu_host 5, epilogue 13.
- **Generated programs, counters** (all non-zero): gemv-stream 64 programs — `post_split` 35, `spread_refused` 1,
  `spread_kept` 12, `vec_post` 9, `vec_narrow` 13 (the regressions step's counts, restored by `post-table`);
  split-tile 64 — tf32x3 9, narrow1 26, narrow1 by rule 5, guard 2, underflow 3, overflow 7, fallback 28, guardless 33;
  mma-combine 33 cut / 8 uncut of 41 tensor kernels, 8 programs × 4 emulator runs; f8 48 — e4m3 35, e5m2 13, fp8 `a` 20,
  saturated 95, subnormal 14, device fp8 tiles 10, device cells 48. Replay: 128 picked / 71 null / 10 mixed / 19 open.
- **Real device**: the hopper TMA probe on the final runtime — 0 / 256 words differ, one encode for two launches.
  Value gates: every arm of every timed program within its bound (`resS-final` 41 programs, `resM-final` 96 + 128 px
  differentials pass).

## Merged emission against each step alone

`alone_norm.tsv`: every one of the 137 timed programs emitted (rtx-5080, default face) by the merged emitter and by each
branch's (kernel ids numbered by first appearance). 13 equal every branch; 27 equal f32-tensor-cores', 6 the
regressions step's, 2 fp8's; 47 are fp8's host (the COLD flag) with f32's device text; the other 42 are narrow-operand
programs whose device text is f32's (or fp8's) plus the regressions step's operand alignment (`align 2` → the
allocation's) — superpositions of independent changes, none an interaction. So a row's timing against a step alone
is that step's code plus the others' independent changes.

## Two regressions the full GPU suite found (each step timed only the programs it moved)

**1. hopper's tensor-map lookup on the launch path (+20–30 µs a program).** The first full run had every small
program 20–30 µs slower (lsm 0.016 → 0.047 ms, mlp_512 0.037 → 0.059). Same-session ABBA, the new host object and PTX
linked against each step's runtime (`rtx/`): 55ba3e42 / f32 / fp8 runtimes 0.0396 ms, hopper's 0.0592. Cause:
`tmap_of` asked the driver for `<kernel>_tmap` at each kernel's first launch, and a failing `cuModuleGetGlobal` costs
~10 µs, inside a program's first timed iteration (with the lookup disabled: 0.0405). Fix (runtime only, host ABI
unchanged): the loaded PTX names its recipe globals — read once at module load on the init thread; a kernel without one
never asks; the map table is made at the first recipe. ABBA after (8 reps): mlp_512 0.0393 → 0.0405 (fix) / 0.0403
(heap table), lsm_32_8192 0.0230 → 0.0239, indep_512 0.0400 → 0.0411, attn_1024_64 0.0410 → 0.0421. The ~1 µs
residual is not isolated (the per-launch code differs by one inlined call returning at once) — open.

**2. The trial's cold flag at the L2's edge (pj_m16_n4096_k4096 1.74× slower).** fp8-leftovers' tune-context marks a
site COLD where `residence_plan` puts an operand in DRAM (working set > L2), and its trials evict the L2. The 4096 × 4096
f32 projection's working set is 64 MiB + 0.5 MiB on the 5080's 64 MiB L2: COLD; cold trials picked candidate 7 (in
context 0.097 ms) where warm / isolated trials pick 1 or 0 (0.056 / 0.061; `tunecheck`: picks 7,7,7 cold; 1,0,1 nocold;
1,1,1 isolated). **Measured property** `DeviceCost::l2_keep_bytes` (`benches/local/l2keep.cu`, re-reading a cyclic
working set back to back): the 5080's L2 keeps 1.010× its size at 4.1 TB/s, 1.025× at 2.5 (the L2 / DRAM midpoint),
1.05× at 1.4, 1.1× at DRAM rate (0.92) — 68 786 585 bytes; unmeasured parts keep the nominal L2. The COLD flag reads
`residence_plan` against `Gpu::l2_keep_bytes()`. It moves 3 of 137 programs (`suite_diff --base 851de1ba`):
pj_m16_n4096_k4096 0.0967 → **0.0561** ms (tune picks 1; 3a8e654f's warm pick ran 0.0559), pj_m1_n4096_k4096 0.0388
(= before), dec_b1_l2048_bf16w 2.400 / 16 (= before). No f8w program moves: fp8's measured cold case (the block decode's FFN GEMVs) stands.
Test: `cold_flag_follows_residence` — a 68 MiB weight cold, the 64 MiB one kept, the 5090's 96 MiB, a 1 MiB weight.

## The RTX 5080 column (matrix.md, 3a8e654f → integration; rented parts keep 3a8e654f)

| geomean Mapal / best (rows) | 3a8e654f | integration |
| --- | ---: | ---: |
| GEMM cells & projections | 1.108 (47) | 1.115 (47) |
| MLP chains | 1.090 (6) | 1.095 (6) |
| Attention / softmax / long rows | 0.838 (14) | 0.883 (14) |
| LLM decode / prefill / block | 1.312 (37) | 1.211 (37) |
| Conv / FIR / elementwise | 0.508 (7) | 0.549 (7) |
| Fan-in/out, loops, CPU+GPU | 1.075 (9) | 1.081 (9) |
| all tf32 rows | 1.287 (13) | 1.198 (13) |
| all bf16w / bf16kv / bf16 rows | 1.938 / 1.375 / 1.195 | 1.861 / 1.252 / 1.139 |
| **all rows** | **1.076 (120)** | **1.065 (120)** |
| win / even / loss | 25 / 21 / 74 | 23 / 29 / 68 |

Plus 7 `_f8w` rows (no fp8 leg installed: `(no leg)`; bf16 reference in the matrix). Files: the 3a8e654f runs for
legs, `resS-final` + `resM-final` for every program, `keep-ab` for the three programs the keep property moved.

**Rows that moved > 5 % slower, session-controlled** (ABBA, the 3a8e654f bundle's binaries against the final ones,
same box, 8–10 reps, `abba/`): lsm_1 / lsm_32 / lsm_4 1.055 / 1.047 / 1.023, attn_1024_64 1.039, fir 1.055, indep_512
1.017, mlp_512 1.009, ew_4096 1.006, conv / convmc / ewconvew / firew 0.999–1.008, pj_m16_n2048 1.008, pj_m1_n2048
0.994, smlp 1.003, prefill 0.999, pj_m16_n6144_bf16 0.918, pj_m16_n4096_bf16 0.993, pj_m1_n4096_bf16 1.006, pj_m128_n4096
1.000, dec_b8_l8192_bf16kv 0.936. The matrix's 1.3–1.5× on lsm and 1.1–1.2× on the other 20–90 µs rows is the
3a8e654f run's session (its old binaries re-timed here: lsm_32 0.0228 against its matrix 0.0165); what the code adds
is the ~1 µs runtime residual (1–5 % on 20–45 µs programs). Bimodal rows: prefill_1024_512_256 (7.73 or 8.3–8.8 ms
in every arm, null included), dec_b8_l8192_bf16kv (null 12 %). **llm_pre_t8192** 89.6 ms (null 5.9 %): the run's tune
picked 3,0,0,0; re-tuned twice on the idle box it picks 3,0,5,3 under both trial modes and runs 79.0 ms (3a8e654f
80.3) — a tune trial's variance, fp8's in-context confirmation (M3) still unbuilt.

| workload (5080) | 3a8e654f | step alone (step) | integration | fastest competitor (leg) | explanation |
| --- | ---: | ---: | ---: | --- | --- |
| dec B8 L2048 bf16kv (ms/step) | 0.293 | 0.257 (regressions) | 0.241 | 0.222 (torch-bf16-compile) | 16-byte scores loads |
| dec B1 L2048 bf16kv | 0.149 | 0.140 (regressions) | 0.140 | 0.145 (torch-bf16-compile) | same |
| blkdec B1 L4096 bf16kv | 0.638 | 0.617 (regressions) | 0.616 | 0.556 (torch-bf16-compile) | same |
| mlp_2048_tf32 | 0.832 | 0.785 (f32) | 0.787 | 0.760 (torch-compile-ro-tf32) | TF32 tile cut S = 8 |
| mlp2_2048_tf32 | 1.666 | 1.568 (f32) | 1.572 | 1.513 (torch-compile-ro-tf32) | same |
| pj_m128_n6144_k4096_tf32 | 0.239 | 0.184 (f32) | 0.190 | 0.129 (lt-tf32-fresh) | 96 blocks on 168 slots |
| pj_m2048_n4096_k14336_tf32 | 4.898 | 4.370 (f32) | 4.356 | 4.241 (lt-tf32-fresh) | 512 blocks |
| dec B1 L2048 f8w | – | 0.0558 (fp8) | 0.0558 | no fp8 leg (bf16 ref 0.145) | 42 MB fp8 weights + cache in L2 |
| dec B8 L2048 f8w | – | 0.298 (fp8) | 0.270 | no fp8 leg (bf16 ref 0.222) | + regressions' operand alignment on the fp8 weights |
| blkdec B1 L4096 f8w | – | 0.370 (fp8) | 0.372 | no fp8 leg (bf16 ref 0.556) | FFN weight bytes halved |
| pj_m16_n4096_k4096 | 0.0494 | – | 0.0561 | 0.0553 (cublas-fresh) | keep property: warm trial (cold pick: 0.0967) |
| lsm_32_8192 | 0.0165 | – | 0.0242 | 0.0102 (torch-compile-cg) | session (ABBA 1.047): 3 launches; open (regressions item 4) |

No row is slower than its step alone beyond null: the steps' rows reproduce (0.99–1.03×), the f8w B8 row gains the
regressions step's narrow-operand alignment (0.298 → 0.270).

## Open problems attacked

- **Closed**: the one-candidate epilogue split fixed twice (one rung, both intents); narrow1 over fp8 (geometry check);
  the matrix rate twice (one property); the tensor-map lookup (+20–30 µs a program); the cold flag at the L2's edge
  (measured `l2_keep_bytes`); five evidence-tool gaps; fp8 one-element loads in the `narrow-tile` intent (`ldg8`).
- **Attempted, open**: the ~1 µs runtime residual (lookup off, heap table: 1.1 → 0.8 µs; the per-launch code differs by
  one inlined early return); the tune's pick variance (pre_t8192; fp8's M3); the fp8 register-staged tile's
  `narrow-tile` check (its 16-byte loads are not separable by width from the f32 operand's: needs a per-operand load fact);
  the f8w rows' first evidence pass (599 mismatches, the classes the f32 / bf16w rows already carry: full-tile,
  kmajor-a, grid-share, fill-wave on the decode GEMV kernels).
- No tailored fixes (`docs/tailored-fixes.md` unchanged).

## Sweep merge

Branch `local-rtx-5080`: `a31cd014` + `--no-ff` merges of `sprint/sweep-attention` (`083d9b8d`), `sprint/sweep-gemm`
(`80e6d240`) and `sprint/sweep-runtime` (`d7cae90a`), then the integration commit (the "For integration" sections, the
two harness fixes below, this evidence). Raw: `.claude/raw-evidence/local-rtx-5080/integration2/` (`tests.log`,
`tests-collided.log`, `counters-*.log`, `resS/` `resM/`, `abba_*.txt`); the emission comparison scripts and tables in
the perf scratch `integration2/` (`alone_k.py`, `alone_k.tsv`).

**Run in parallel (the user asked for speed): 75 min from merge to the regenerated column**, against integration 1's
45 min for the workspace tests alone. One release build of every test binary and bundle tool (52 s); the workspace
through sweep-runtime's `test_par.py` on CCDs 2–3 (4 binaries at a time: **955 s**, 776 worker-seconds held for
Tctl > 85 C); both GPU bundles on CCD 1 at the same time (the bundle cache reused 58 + 87 stems; a thermal guard
stopped the bundle tree 7 times, 618 s, above 85 C — never a test binary or a timed process); the timed suite on
CCD 0 as soon as its bundle was ready (overlapping the tests' second half), the model-scale run (+ every competitor
leg) after it; the emission comparison and the counter runs on CCDs 2–3 during the timed runs.

| merge | conflicts | resolution |
| --- | --- | --- |
| sweep-attention | none | — |
| sweep-gemm | `GpuRungs::NAMES` (`unit-uniform` beside `round-store`) | both: 130 names |
| sweep-runtime | none (`testgen/mod.rs`, `evidence.py` merged clean) | — |

**Merged emission against each step alone** (`alone_k.py`: every timed program, rtx-5080, default face, emitted by
the merged emitter and by each branch's; device = the multiset of kernel bodies with ids, metadata numbers and
`; mapal.` comments dropped; host = the text with ids numbered by first appearance). Device: 105 programs equal
`a31cd014`, 7 equal sweep-attention's (T8192's stream; the streamed kernels' `nvvm.minctasm`), 23 sweep-gemm's
(round-store), 2 the superposition of both (`llm_pre_t2048_bf16`, `llm_blkpre_t2048_bf16`). Host: one program
(`llm_blkdec_b1_l4096_bf16`) matches no arm only through a `; mapal.` flow comment that carries both steps' fields;
its code equals sweep-gemm's. No interaction: each row runs its step's code.

### Tests

- **Workspace** (`test_par.py --bins 4`, `RUST_TEST_THREADS=8 MAPAL_PAR=2 MAPAL_TESTGEN_JOBS=4`): 125 binaries
  (124 + doc tests), **1524 passed, 0 failed, 7 ignored**.
- **Found: `test_par.py` named a binary's log and times key by its target only.** Seven target names exist in two
  packages (`golden`, `differential`, `acceptance`, `captures`, `narrow`, `widen`, `format`: 15 binaries), so two
  binaries shared one log file (a failure's log could be overwritten) and one times key (longest-first order). Fixed:
  the package directory is part of the name (124 distinct). The 15 rerun alone: 160 passed, 0 failed; the other logs
  1364 — the total stands.
- **Counters on the merged code** (`--nocapture`): `flash_attention` generated — 128 programs, 39 head-group loops,
  31 streamed with moved operands, 12 trailing members (4 streamed), 39 unstreamed with `unit-uniform` off, 90
  streamed, 731 emulator runs; `half_tile` round-store generated — 40 programs: stored 32, a-tiles 6, b-tiles 10, FMA
  reads 12, refused 2, epilogues 6. Both equal their step's counts; every counter > 0.
- **Evidence** (static, both bundles): mismatches bS 246 → 246, bM 5329 → 5321 (T8192's spill gone from five
  classes, one `kmajor-a`), no new class; `unit-uniform` 6 on / 221 off ok, `round-store` 36 ok; the runtime
  `trace allocs == predicted` 30 → 0 (sweep-runtime's fix). Value gates: 41 suite programs, 96 model-scale programs +
  128 px differentials pass on every arm.

### Time (`resS` / `resM` against integration 1's `resS-final` / `resM-final`)

The 32 programs whose code moved: geomean 0.934; each reproduces its step's ON time. The 105 others: 0.977 (session).

| workload (5080) | integration 1 | sweep merge | step alone (step) | fastest competitor (leg) | explanation |
| --- | ---: | ---: | ---: | --- | --- |
| llm_pre_t8192 | 89.55 | **47.93** | 48.89 (attention) | 53.22 (liger) | one streamed kernel a KV-head group; peak 2991 → 776 MB (0.41× torch-compile's 1916) |
| llm_pre_t2048 | 7.222 | 6.604 | 6.861 (attention) | 7.213 (liger) | stream kernel `minctasm` (no spill) |
| pj_m128_n4096_k4096_bf16 | 0.0773 | 0.0634 | 0.0640 (gemm) | 0.0532 (torch-bf16) | 2-byte `a` panel |
| pj_m2048_n6144_k4096_bf16 | 1.080 | 0.937 | 0.938 (gemm) | 0.883 (torch-bf16-compile-cg) | same |
| llm_blkpre_t2048_bf16 | 11.62 | 10.74 | 10.75 (gemm) | 8.036 (torch-bf16-compile-cg) | stored activations + stream `minctasm` |
| llm_blkdec_b1_l4096_bf16 (ms/step) | 0.615 | 0.582 | 0.580 (gemm) | 0.558 (torch-bf16-compile) | stored activations |
| pj_m128_n28672_k4096_bf16 | 0.3844 | 0.3995 | 0.3996 (gemm; review-fixes-2's 12-rep re-measure — the 0.4037 first cited had no raw run) | 0.3174 (triton-bf16) | panel depth (sweep-gemm open, reproduced) |

**Unchanged code slower than 5 %: none from the merge.** Four rows ran > 5 % slower than integration 1's run with
equal code; old (integration 1 bundle, its wisdom) against new binaries, same session, ABBA on CPUs 0–7:
`llm_dec_b1_l2048_f8w` 0.999 (both 1.15 ms; integration 1's run 0.892 — open, below), `pj_m1_n4096_k4096` 0.961
(16 reps; 1.129 at 8: a bimodal GEMV), `llm_dec_b32_l2048_f8w` 1.002, `llm_dec_b32_l2048` **1.047** — the tune's pick
(site 2 candidate 3 there, 0 here), the tune-variance item.

**The session gap is partly the timed CPUs.** Integration 1 timed on CPUs 8–15; 3a8e654f and this run on 0–7. The
9970X's per-core clock ceilings differ by CCD (ACPI CPPC `highest_perf`: 211–241 on CCD 0, 166 on CCDs 1 and 3,
171–206 on CCD 2). The same binaries, 0–7 against 8–15 (ABBA, 8 reps): lsm_1_131072 0.81, pj_m1_n4096_k4096 0.89,
ew_4096 0.94, mlp_512 / indep_512 0.95, lsm_32_8192 0.97, pj_m1_n2048 0.99, attn_1024_64 1.00, dec_b1_l2048_f8w 1.00.
And `lsm_32_8192` is bimodal by process (0.018 / 0.0235 ms in one batch): integration 1's "lsm 1.3–1.5× session"
is that mode plus the CPU set, not code. Harness fix: `benches/local/timed_cpus.sh` — the timed scripts'
busy check read CPUs 0–7 whatever `CPUS` was (integration 1 timed on 8–15 while checking 0–7); it now reads the
timed set, and `conditions.txt` records the set's `cppc` range.

### The RTX 5080 column (matrix.md: integration → sweep merge; legs from this session's runs)

| geomean Mapal / best (rows) | integration | sweep merge |
| --- | ---: | ---: |
| GEMM cells & projections | 1.115 (47) | 1.079 (47) |
| MLP chains | 1.095 (6) | 1.073 (6) |
| Attention / softmax / long rows | 0.883 (14) | 0.818 (14) |
| LLM decode / prefill / block | 1.211 (37) | 1.151 (37) |
| Conv / FIR / elementwise | 0.549 (7) | 0.530 (7) |
| Fan-in/out, loops, CPU+GPU | 1.081 (9) | 1.061 (9) |
| all tf32 rows | 1.198 (13) | 1.194 (13) |
| all bf16w / bf16kv / bf16 rows | 1.861 / 1.252 / 1.139 | 1.795 / 1.195 / 1.067 |
| **all rows** | **1.065 (120)** | **1.021 (120)** |
| win / even / loss | 23 / 29 / 68 | 28 / 31 / 61 |

The code moved 32 rows (above); the rest of the shift is the session (CPU set, the legs re-timed in the same session).

### Open problems attacked (integration)

- **Closed**: `test_par.py`'s shared logs (15 binaries); the timed scripts' busy check on the wrong CPUs and the
  unrecorded CPU set (`timed_cpus.sh`); sweep-runtime's T8192 peak memory (closed by sweep-attention's stream,
  measured here: 1.56× → 0.41× torch).
- **Attempted, open**: the tune's pick variance — a third row (`llm_dec_b32_l2048` 1.047, equal code; fp8's M3
  in-context confirmation unbuilt); `llm_dec_b1_l2048_f8w` 0.892 (integration 1, fp8-leftovers) against 1.15 ms
  (both bundles here, ABBA 0.999, either CPU set) — equal code and picks, so a machine state the conditions do not
  yet record (the 42 MB fp8 working set in a 64 MB L2 shared with the display).
- No tailored fixes (`docs/tailored-fixes.md` unchanged).

## Second sweep merge

Branch `local-rtx-5080`: `4c29387e` + `--no-ff` merges of `sprint/sweep-attention-2` (`0992dea6`), `sprint/sweep-gemm-2`
(`43af51cf`) and `sprint/sweep-runtime-2` (`683ae4d6`), then the integration commit (the "For integration" sections,
the fixes below, this evidence). Raw: `.claude/raw-evidence/local-rtx-5080/integration3/` (`tests.log`, `counters.log`,
`streamk_epi.log`, `horizontal-rerun.log`, `resS/` `resM/`, `resM-gpu-clocks.csv`); scripts, the emission comparison
(`alone_k.py`, `alone_k.tsv`) and the evidence re-checks (`evcheck/`) in the perf scratch `integration3/`. Bundles under
`target/int3/` (the shared `/tmp` tmpfs was near full). Timeline: tests on CCDs 2–3 (1042 s, workers held 803 s for
Tctl > 85 C) beside both bundles on CCD 1 (1629 s, a thermal guard held the tree 817 s); the suite then model scale
on CCD 0 (05:06–05:56), the A/B after them.

| merge | conflicts | resolution |
| --- | --- | --- |
| sweep-attention-2 | none | — |
| sweep-gemm-2 | `GpuRungs::NAMES` (`stream-mma`, `kmajor-b` beside `warm-zero`, `streamk-price`) | all four: 134 names |
| sweep-runtime-2 | `GpuRungs` (+ `streamk-epi`: 135), `launch_key`'s dropped fields, `kernel_text` (the group stream-K assert beside `streamk-price`'s `walk`), `evidence.py`'s plan-gated rungs, `c0-rules.tsv` | both sides; the golden regenerated: the same 6 rows sweep-runtime-2 moved |

**Merged emission against each step alone** (`alone_k.py`, as integration 2: 137 timed programs, rtx-5080, default
face). Device: 71 equal `4c29387e`, 51 attention-2's, 1 gemm-2's, 10 runtime-2's, 3 the superposition, **1 none**
(`rt_mlp2_1024`). Host: 27 base, 2 attention-2's, 96 gemm-2's (warm-zero's init word), 8 superposition, **4 none**
(`rt_mlp2_1024`, `smlp_4096_256_1024`, `epi_4096`, `loop_4096`). The interactions, each found and handled:

- **`streamk-price` × `streamk-epi`** (`rt_mlp2_1024`, `smlp`): an epilogue group at a site the part's rates now stream
  takes the ticket (`walk=price epi=1`), which neither step emitted. Generated: `streamk_epi.rs`'s plans now also draw
  each program at `streamk-price`'s band (768–1536 a side, 512–4096 deep, the shipped placement): 83 groups, 67
  tickets (walk class 27, fill 24, **price 16**), 16 ordered; the priced walk covers its tiles once. Values: 10
  ticket programs (10 whole tiles, 49 shared; 8 more at 4 SMs on the class walk), 30 emulated runs, interpreter =
  ON = OFF. Time (same session, ABBA, `--gpu-off=streamk-epi`): `rt_mlp2_1024` 2.6187 → **2.5705** (0.982, null
  0.001), `smlp` 0.1616 → 0.1599 (0.989, inside its null 0.019).
- **`warm-zero` × `streamk-epi`**: `module_zeroes` read only a table row's `zero` bytes; the ticket's workspace (`ws`,
  no reduce kernel) is zeroed by the runtime too. Fixed (a row with `rid` 0 and `ws` > 0 counts; unit test): `epi_4096`
  and `loop_4096`, whose only zero is the workspace, now warm the memset path (init word 9 → 41, 13 → 45). Their
  iterations take 3.5 / 27 ms, so the 13–17 µs moved out of the first call is below their null.
- **`streamk-price` × the `horizontal` test**: `rt_mlp2_1024`'s two 128-block GEMMs now walk by stream-K, so the
  refusal the test asserted (`waves:1`) is never weighed; sweep-gemm-2 ran 11 of the 126 binaries. The test now
  asserts no refusal under the shipped rules and `waves:1` with `streamk-price` off.

### Tests

- **Workspace** (`test_par.py --bins 4`, `RUST_TEST_THREADS=8 MAPAL_PAR=2 MAPAL_TESTGEN_JOBS=4`): 126 binaries,
  **1530 passed, 1 failed** (`horizontal`, above), 7 ignored; after the fix `horizontal` 3 / 3 and `streamk_epi` 2 / 2
  (extended) — **1531 passed, 0 failed**.
- **Counters on the merged code** (`counters.log`): `stream-mma` 48 programs (48 on the matrix instruction, 24 causal,
  19 loops, 10 split-KV, 6 tails), 250 emulator runs; `kmajor-b` 40 programs, 61 vector-copy tiles, 107 runs;
  `streamk-price` 17 priced walks of 43 under-filled kernels, 3 programs / 9 runs; `streamk-epi` as above. Each equals
  its step's count where the step drew the same programs; every counter > 0.
- `rung_replay.tsv`: 230 lines, 130 picked / 71 null / 10 mixed / 19 open (`open-rules.md`).
- `benches/gpu/test_harness.py`: 6 failures, all present without this merge's checker edits; 5 fail at `4c29387e`, the
  sixth (`search.py` plan: `--splitk=2` reaches `rowsum_2048` on the 4090) is skipped there for want of an emitter, and
  the `4c29387e` emitter emits the same split — stale harness expectations, not this merge's (open).

### Evidence (static, both bundles)

The ticket kernels and `kmajor-b`'s decode copies met checks written before them. Each new class was read in the SASS
/ PTX / remarks; the checker was wrong in these, and is fixed generically:

| check | what the checker did | fix | bS | bM |
| --- | --- | --- | ---: | ---: |
| hot loop (`face-fma`, `face-registers`, `prefetch`, `idx32`, `wide-fma`, `kmajor-a`) | took a stream-K piece loop for the hot loop: its epilogue store's FMULs (`st_whole`, laid out before the k loop) out-counted the k loop's FFMA-only body | `innermost`: from the loop with the most products, descend into a nested loop holding more than half of them (`evidence.py`, `sass_stats.py`) | −57 | −32 |
| `arch-rungs` | read an epilogue group's order (`epi_order`: atomic / workspace splits off, a `tile-post` group's stream-K off) as rungs off | those may be off on an `epi` kernel | −22 | −2 |
| `unroll` | compared the largest `FullyUnrolled` count (a ticket's 8-row store) with bk / 4 | the bk-step count among them (`full_unrolls`) | −2 | 0 |
| `epilogue` | required `launchv`; a group launched through its site table (`mapal_gpu_site`) has its n pointers there | `launchv + site` (`site` counted) | −11 | 0 |
| `narrow-tile` | a mixed-radix tile's narrow operand: 2- or 16-byte loads only; `kmajor-b` copies 4 bf16 a `LDG.64` | `LDG.64` counts | 0 | −46 |

Totals: bS 321 (old checker) → **229** (integration 2: 246); bM 5561 → **5481** (integration 2: 5321). What remains
new in bM against integration 2 is real and recorded: `idx32` on 56 `kmajor-b` kernels at i32 indices (34
`IMAD.WIDE` for 33 `LDG.64` a trip — one 64-bit address a key row, one over the rung's bound; open), the `px_…off-place`
differential arms' `kmajor-b` kernels spilling (12 rows, the OFF-place arms only, not timed), `prefetch` cover 0 on
19 more kernels — the B1 decode programs' `kmajor-b` copies (the next panel's vector LDGs consumed before the math that
should cover them) and the block prefill's ticket kernels (`llm_blkpre_t2048` 25.29 → 25.31 ms, null) — open.

### Time (`resS` / `resM` against integration 2's)

Value gates: 41 suite programs and 96 model-scale programs pass on every arm, 128 px differentials pass. Equal-code
programs (20, `alone_k` base on both sides): geomean **0.990**, none slower than 5 %. Programs whose code moved (117):
0.927. The tune's in-program confirmation ran live (`confirm_pick.py`, median rule): 13 confirms in the suite, 7 kept,
6 reverted.

| workload (5080) | sweep merge | second merge | step alone (step) | fastest competitor (leg) | explanation |
| --- | ---: | ---: | ---: | --- | --- |
| llm_pre_t2048_bf16 | 3.774 | **2.636** | 2.645 (attention-2) | 1.935 (torch-bf16-compile-cg) | stream kernel on HMMA (`stream-mma`) |
| llm_blkpre_t2048_bf16 | 10.74 | **9.593** | 9.61 (attention-2) | 8.071 (torch-bf16-compile-cg) | same |
| llm_dec_b32_l8192_bf16kv (ms/step) | 2.430 | **1.657** | 1.676 (attention-2) | 1.482 (flashinfer-bf16) | `kmajor-b`: keys copied along their rows |
| llm_dec_b8_l8192_bf16kv | 0.7351 | **0.5111** | 0.512 (attention-2) | 0.4841 (fa2-bf16) | same |
| llm_dec_b32_l8192 (f32) | 3.586 | **2.800** | 2.816 (attention-2) | 9.525 (torch-compile) | same, 16-byte vectors |
| pj_m1_n2048_k7168 | 0.0307 | **0.0249** | 0.0275 (gemm-2) | 0.0266 (cublas-fresh) | `warm-zero` |
| pj_m16_n2048_k7168 | 0.0460 | **0.0387** | 0.0392 (gemm-2) | 0.0492 (cublas-fresh) | same |
| pj_m1_n4096_k4096_bf16 | 0.0336 | **0.0231** | — | 0.0492 (torch-bf16-compile-cg) | same |
| loopmm_1024_8 | 0.6494 | **0.6117** | 0.6121 (gemm-2) | 0.5878 (torch-compile-cg) | `streamk-price` |
| rt_mlp2_2048 | 16.27 | **14.65** | 14.71 (runtime-2) | 15.57 (torch-compile-cg) | `streamk-epi` |
| fanout_2048 | 0.9853 | **0.9177** | 0.9265 (runtime-2) | 0.9132 (chain-streams) | same |
| mlp_2048 | 0.9467 | **0.8816** | 0.8921 (runtime-2) | 0.9134 (chain-lt) | same |
| loop_4096 | 28.33 | **26.88** | 27.03 (runtime-2) | 28.58 (chain-lt) | same |
| rt_mlp2_1024 | 2.559 | 2.567 | 2.512 (gemm-2) | 2.518 (torch-compile-cg) | the interaction above (ON/OFF 0.982 same session) |
| llm_dec_b1_l2048_f8w (program) | 1.1517 | 0.8477 | — | (no leg) | the session item: siblings ≤ 3 % |

Slower than integration 2 by > 2 %: `mlp_512` 1.054 (bimodal by process: in one A/B batch on 0.0395, null 0.0373 —
its null is 7 %), `llm_kvx_b8_l2048` 1.037 (null 2.9 %; `kmajor-b` code, every one of its 59 programs ON 1.01–1.45×
faster in its own A/B), `convmc_8_64_56_3` 1.023 (equal code). `pj_m1_n4096_k4096` reads 0.0411 here (its five `on`
samples 0.0378–0.0469) against sweep-gemm-2's 0.0311: the bimodal GEMV integration 2 named.

**The f8w B1 decode.** `llm_dec_b1_l2048_f8w` 1.1517 → 0.8477 ms while every sibling moved ≤ 3 % under `kmajor-b`:
the 0.89 / 1.15 split is the session, not the code. This is the first full run with the GPU's clocks sampled (the
model-scale script did not record them; it does now): P0 / P1 at 14.8–15.0 GHz memory through the run (5848 samples:
2 at P8, 46 at P3, all between phases). The slow state's clocks remain unrecorded (attempted-open).

### The RTX 5080 column (matrix.md: sweep merge → second merge; legs from this session's runs)

| geomean Mapal / best (rows) | sweep merge | second merge |
| --- | ---: | ---: |
| GEMM cells & projections | 1.079 (47) | 1.036 (47) |
| MLP chains | 1.073 (6) | 1.054 (6) |
| Attention / softmax / long rows | 0.818 (14) | 0.836 (14) |
| LLM decode / prefill / block | 1.151 (37) | 1.045 (37) |
| Conv / FIR / elementwise | 0.530 (7) | 0.525 (7) |
| Fan-in/out, loops, CPU+GPU | 1.061 (9) | 1.017 (9) |
| all tf32 rows | 1.194 (13) | 1.176 (13) |
| all bf16w / bf16kv / bf16 rows | 1.795 / 1.195 / 1.067 | 1.661 / 1.048 / 0.975 |
| **all rows** | **1.021 (120)** | **0.973 (120)** |
| win / even / loss | 28 / 31 / 61 | 30 / 38 / 52 |

Attention's 0.818 → 0.836 is the legs: `causal_1024_256`, `gen` / `prefill_1024_512_256` and `decode_1024_256`
torch-compile-cg / chain-cache ran 12–17 % faster this session on equal Mapal times.

### Open problems attacked (integration)

- **Closed**: the merge's two interactions (`warm-zero` × the ticket workspace; a group at a priced walk, now generated
  and timed); the stale `horizontal` expectation; five evidence-checker misreadings of the new kernels (table above);
  the full suite / model scale not recording GPU clocks (`model_scale_run.sh` → `gpu-clocks.csv`, `.txt`).
- **Attempted, open**: the f8w B1 session state (now 0.848 ms with clocks recorded at P0 / P1; the slow state not seen
  again).
- **Open (found here)**: `kmajor-b`'s per-row 64-bit addresses at i32 indices (`idx32`, 56 rows) and its B1 decode
  copies' prefetch cover; `test_harness.py`'s 6 stale expectations.
- No tailored fixes (`docs/tailored-fixes.md` unchanged).

## Confirmation-fixes merge

Branch `local-rtx-5080`: `54afa0b1` + `--no-ff` merges of `fix/bugs-properties` (`710eddfd`, merge `530c5231`),
`fix/tensor-cores` (`b8a31c27`, `8a9be0fc`), `fix/hopper-launch` (`ed551899`, `01911e7f`) and `fix/attention-chains`
(`adc9d1ff`, `3bd55d42`), then one interaction fix (`39cd2c87`) and this evidence. The four steps answered the
cross-arch 5fcb401b confirmation round (`cross-arch/README.md`); their evidence: `bugs-properties.md`,
`tensor-cores-2.md`, `hopper-launch.md`, `attention-chains.md`. Rental spend of the four steps: $0 + $0.50 + $2.54 +
$0.25 = **$3.29** (this merge: $0, local only). Raw: `.claude/raw-evidence/local-rtx-5080/confirmation-fixes/`
(`tests.log`, `testlogs/`, `counters.log`, `counters-split-after-fix.log`, `resS/`, `resM/`; `scripts/`: the bundle and
run scripts, `alone.py`, `kern.py`, `static_one.sh`, `cmp.py`; `analysis/`: the emission comparisons `alone-<part>.tsv`,
`kern-none.tsv`, the static pass `static.tsv`, the timing comparison `cmp.txt`, the runtime A/B `rtab-abba.txt`). (The step first wrote into `raw-evidence/local-rtx-5080/integration/`, integration 1's directory: its
`tests.log` / `counters.log` and `resS/box.out` were overwritten; its resS results are kept in `resS-integration1/`,
its resM in `resM-integration1/`.)

| merge | conflicts | resolution |
| --- | --- | --- |
| bugs-properties | none | — |
| tensor-cores | `rules.rs` (the split rule), `profile.rs` (the H100 / A100 `mma.sync` rates' comments, `mma_rates_of_cc`), `split_tile.rs`, `evidence.py` (the hot-loop weight) | **one split-issue model** (below); the A100 keeps the measured 1012 / 2024; `MMA_WEIGHT = TENSOR_MACS` (both 32) |
| hopper-launch | `rules.rs` (replay parsers, `OPEN`, counts) | both sides; `GpuRungs::NAMES` 139 (`split-issue`, `split-least`, `tma-warm`, `ws-persist`) |
| attention-chains | `rules.rs`, `gpu.rs`, `split_tile.rs` (the split rule again) | the one model; its `tile=` replay lines folded (below) |

**The split rule, three versions of one fact.** bugs-properties (`split_issue_pays`, 22 instructions an operand
element), attention-chains (`split_elem_instr`, 21 an element at the FP32 lanes) and tensor-cores-2 (`split_issue_pm`)
each priced the split realization's instruction issue. The first two read the 5fcb401b `cvt.rna` cut's SASS;
tensor-cores-2 replaced that cut (integer TF32 cut, 792 instructions a 96-HMMA panel against 1612) and prices the cut
the realization emits (`SPLIT_CUT_TF32` 10, `SPLIT_CUT_TF32_CVT` 22 under `--gpu-off=split-issue`). Kept: tensor-cores-2's
(`split_rate_pays` over `split_issue_pm`, θ = 1.6) — the other two would refuse the A100's measured tf32x3 winners
(1.43–1.54× the FFMA tile) and duplicate one fact (CLAUDE.md: no parallel model). Their measurements stay, as replay
lines: the three copies of the A100 5fcb401b loser are one line (`issue=2100`, every note), the 5080 lines drop `tile=`.
attention-chains' lane width, tried as a factor on the issue (the A100's 64 lanes take an integer instruction 2 clocks a
scheduler): it leaves θ a 1.37–1.41 window (the H100 PCIe tf32x3 loser at issue 1.29 against the A100 winners at 1.0)
and the 5080's bf16x2n winner still needs θ ≤ 1.0 — it separates nothing the realized cut does not (open line kept).
Replay: 286 lines, **158 picked / 74 null / 16 mixed / 23 open** (the four steps' lines; 1 duplicate folded).

### Merged emission against each step alone

Each of the 137 timed programs (bS + bM stems) emitted by the merge, the 5fcb401b emitter and each branch tip, per part;
host and device IR less `; mapal.` lines, kernel-name / key hashes and metadata numbers normalized (the launch key's
fields grew in three steps, renaming every kernel), lines sorted (`alone.py`); a program equal to no single step is then
split per kernel (`kern.py`): a kernel equal to some step's is a superposition, one equal to none is an interaction.

| part | device text equal to: base / one step / several steps / **none** | kernels equal to no step |
| --- | --- | --- |
| rtx-5080 | 126 / ac 7, bugs 4 / — / **0** | none: every timed program runs a step's own text |
| a100 | 71 / tc 29, bugs 8, ac 4 / 6 / **19** | 14 programs: split tiles × `cut-price` (13: the 2048-row tiles tensor-cores-2 cut `S` 2–4, now whole) and `attn_2048_64`'s stream geometry |
| h100 | 124 / ac 6, tc 4, bugs 1 / — / **2** (`llm_pre` / `blkpre_t2048_bf16w`) | none: tc's bf16x2n kernels beside ac's stream kernels |
| rtx-3090 | 105 / bugs 22, ac 8 / — / **2** (`llm_blkpre_t2048(_bf16w)`) | none: bugs' and ac's kernels side by side |
| rtx-4090 | 127 / ac 9, bugs 1 / — / **0** | none |

- **`cut-price` × the split tile (fixed, `39cd2c87`).** `mma_pieces` priced a tensor tile's waves at the matrix pipe's
  rate; a split tile runs its form's `P` products a panel at its issue ratio, so bugs-properties' ns price (fitted with
  no split tile on any part) saw the A100's tf32x3 waves 3× fast and refused the cuts tensor-cores-2 timed winning
  (`pj_m128_n4096_k4096` / `_k14336`, `S` 8: 1.105 / 1.055 over the FFMA tile). The waves are now priced at the rate
  the split tile's panels realize (`split_issue`, the existing fact); both cut again (test
  `split_cut_priced_at_the_realized_rate`). The 2048-row tiles stay whole: 4 waves against 13 of pieces saves 50 µs,
  16.8 M `red` adds past the 20 MiB the A100 keeps cost 68 µs — the mechanism the A100's TF32 MLP cuts measured losing
  (0.84–0.91). Review: tensor-cores-2's own run timed the 2048³ split tile both ways side by side (`mlp_2048` /
  `mlp2_2048`: whole `9a42b908` / `538e9763` / `6ffab719` 0.891–0.901 ms, `S` = 4 `440ac947` 0.915–0.919, one hot loop
  — 792 instructions, 96 HMMA, 8 LDGSTS — the cut's 32 RED and zero pass on top): whole 1.025×, the price's side. The
  `pj_m2048_n4096_k4096` / `_k14336` tiles (`S` = 2 measured, whole now) have no whole timing: **open**, the next A100
  round's arm (`--splitk=2` against the shipped build).
  16-bit `narrow1` tiles (P 1, issue < 1) are unchanged: the 5080's emission of all 137 programs is byte-for-byte the
  steps'.
- **`attn_2048_64` on the A100**: ac's price with bugs' measured latency (728 ns) and barrier (54 cycles) picks
  `b8xkb32:rt2g8t32:cp:s8` (141.9 µs priced) where ac picked and timed `b16xkb16:rt4g8t32:cp:s16` (0.160 ms); both
  stream against a 152.6 µs split. Unmeasured: next round's arm (`--stream=16x16x4x16` against the shipped pick).

### Tests

- **Workspace** (`test_par.py --bins 4`, `RUST_TEST_THREADS=8 MAPAL_PAR=2 MAPAL_TESTGEN_JOBS=4`, CPUs 16–31 / 48–63):
  126 binaries, **1541 passed, 0 failed** (wall 1181 s, 309 s held for Tctl > 85 C). After the cut fix: backend lib 128,
  `split_tile` 6 (+1), `golden_ll` 13, `site_choice` 9, `half_tile` 5, `tf32_tile` 36, `calls` 2, `epilogue` 5 — pass.
- **Counters** (`counters.log`, the merged code; split-tile's again after the cut fix, `counters-split-after-fix.log`:
  equal): split-tile 64 programs (tf32x3 9, narrow1 26, bf16x3n 4, bf16x2n 2,
  refused rate 9 / K_min 2, pieces 27, guard / overflow / cancel paths 3–7 each); ptx-isa 128 modules through llc +
  ptxas (fp8 instructions on the 6 profiles with the conversions, none on the 3090 / A100); flash-attention 128
  programs, 1023 runs (chunks > 1: 10, contiguous consumer 22, narrow groups streamed 73); wgmma-ws 36 programs (walks
  9, ring across tiles 2, one long walk); `mma-combine` 21 cut / 21 uncut of 42 tensor kernels, 32 emulated runs;
  stream-mma 48 programs / 250 runs; kmajor-b 40 / 107. Every counter > 0.
- **Static, the rented parts at the merge** (`static.tsv`): the 137 programs emitted for the A100, H100 SXM / PCIe,
  4090 and 3090, `opt`, `llc` at each module's declared PTX ISA (`+ptx70` … `+ptx81`), `ptxas`: **685 / 685 assemble**
  (tf32x3 tiles in 43, bf16x2n in 18 of the A100 / H100 programs). Spills (16–52 bytes) only in decode programs whose
  text equals 5fcb401b's.

### Time (`resS` / `resM`, the 5fcb401b protocol: bench lock, CPUs 0–7 idle and pinned, 30 reps, `PROBE_REPS=5`)

Value gates: 41 suite + 96 model-scale programs pass on every arm, 128 px differentials pass; 3 / 800 competitor legs
with max > 2 × med or med > 1.3 × min (all in resS, not re-timed). Against the 5fcb401b column (same protocol, same
box): **geomean 1.008** over 137 programs; the 126 with text equal to 5fcb401b's 1.008; attention-chains' 7 1.024;
bugs-properties' 4 1.002. Review: the 1.008 on equal text is session drift, not the runtime — the one runtime A/B
(`analysis/rtab-abba.txt`, `llm_kvn_b1_l2048`, old / new `mapal_gpu.c` same-box ABBA) is null (6.43 / 6.43), and the
competitor legs drifted the other way on the same objects (×1.005).

| workload (5080) | 5fcb401b | merge | fastest competitor (leg) | explanation |
| --- | ---: | ---: | --- | --- |
| llm_pre_t8192 | 47.61 | **44.65** | 52.89 (liger) | interleaved consumer chunks (2-way conflicts gone), attention-chains |
| llm_pre_t2048 | 6.648 | **6.458** | 7.216 (liger) | same |
| llm_blkpre_t2048 | 25.37 | **25.17** | 26.83 (liger) | same |
| llm_pre_t2048_bf16w | 6.508 | 6.365 | 1.931 (torch-bf16-compile-cg) | same; bf16x2n refused on GeForce (open) |
| llm_pre_t2048_tf32 | 5.484 | 5.342 | 4.543 (triton-tf32) | same |
| pj_m2048_n4096_k4096_bf16 | 0.6908 | 0.6742 | 0.598 (torch-bf16-compile-ro) | `cut-price`: whole tile (bugs-properties) |
| pj_m2048_n4096_k14336_bf16 | 2.241 | 2.267 | 2.095 (torch-bf16-compile-cg) | same, 0.988 (bugs-properties' own A/B 0.992) |
| llm_kvn_b1_l2048 (equal text) | 6.433 | 7.526 | 47.57 (naive-compile) | session: 7 of 10 processes at 7.5, both arms; old / new runtime same-box ABBA 6.43 / 6.43 |
| llm_dec_b1_l2048_f8w (equal text) | 0.861 | 0.990 | (no leg) | the known f8w B1 slow state (integration 2), open |

Slower than 5fcb401b by > 5 % beyond these: `gen_1024_512_256` 0.907 (processes at 7.7 or 8.6–8.9 ms on both
arms), `firew_rowsum_2048x2048_16` 0.896 (two first-rep outliers; null 0.115), `indep_512` 0.928 (both arms 0.040–0.043)
— equal text, session. Faster: `lsm_32_8192` 1.235, `pj_m1_n4096_k4096` 1.254 (the bimodal GEMV), `rt_mlp2_1024` 1.075.

### The RTX 5080 column (matrix.md: 5fcb401b → merge; rented parts keep 5fcb401b)

Review: the first version of this table timed all 120 rows in the merge session, so the 104 rows whose host and device
text equal 5fcb401b's (`analysis/alone-rtx-5080.tsv`) moved by the session alone (Mapal ×0.990, best leg ×1.005:
0.941 → 0.927) and carried the column 0.979 → 0.965. Equal text is the same object and PTX — null by construction —
so `matrix.md` now takes those rows from the 5fcb401b session and only the 14 changed programs from the merge's
(`confirmation-fixes/changed-res{S,M}`: their probe, torch, comp and chain lines); the merge column is **0.978**
(30 / 39 / 51), the changed rows 1.288 → 1.280. The table below is that matrix; the first version's column is kept
in brackets.

| geomean Mapal / best (rows) | 5fcb401b | merge |
| --- | ---: | ---: |
| GEMM cells & projections | 1.045 (47) | 1.045 (47) [1.033] |
| MLP chains | 1.054 (6) | 1.054 (6) [1.048] |
| Attention / softmax / long rows | 0.858 (14) | 0.855 (14) [0.840] |
| LLM decode / prefill / block | 1.036 (37) | 1.035 (37) [1.015] |
| Conv / FIR / elementwise | 0.529 (7) | 0.529 (7) [0.534] |
| Fan-in/out, loops, CPU+GPU | 1.032 (9) | 1.032 (9) [1.023] |
| all tf32 / bf16w / bf16kv / bf16 rows | 1.186 / 1.662 / 1.043 / 0.982 | 1.186 / 1.655 / 1.043 / 0.982 [1.183 / 1.615 / 1.014 / 0.967] |
| **all rows** | **0.979 (120)** | **0.978 (120)** [0.965] |
| win / even / loss | 30 / 39 / 51 | 30 / 39 / 51 [32 / 37 / 51] |

The rented columns move at the next round: the steps' single-session measurements (A100 tf32x3 1.43–1.54, bf16w
prefills A100 1.74 / 2.67 and H100 PCIe 1.69 / 2.16, H100 `wgmma-ws` 11 / 12, A100 streamed attention 1.17–2.8×)
are not the round's protocol. Review: the A100 tf32x3 ratios are tensor-cores-2's kernels (13 programs now run whole
2048-row tiles; whole timed only at 2048³) on a box whose Mapal probes ran at a ramp clock (ratios only); the A100
streamed attention boxes ran no evidence pass (timing only, not a result); the H100 `wgmma-ws` boxes' evidence is
rebuilt below.

### Open problems attacked (integration)

- **Closed**: the split rule's three copies (one model, every step's measurement a replay line); `cut-price` × the
  split tile (priced at the realized rate: the measured cuts taken again); the 18 decode rows the first comparison
  missed (zsh word splitting in the stems; rerun under bash: all equal to a step).
- **Attempted, open**: the 5080's bf16x2n winner (attention-chains' lane width as the separating property: no — above);
  `llm_kvn_b1_l2048`'s 7.5 ms processes (not the runtime: same-box ABBA null; the slow state unrecorded beyond P1
  clocks).
- **Next round's arms** (unmeasured predictions): the A100's split-tile cuts at 2048 rows (`--splitk=4`), its
  `attn_2048_64` stream geometry, the A100 rows tf32x3 retakes after attention-chains / bugs-properties refused the
  5fcb401b cut (`rowsum`, `fanout`, `diamond`, `twin`, `indep`, `softmax_rows` 2048), the steps' own queued arms
  (bugs-properties: the 3090 stream geometry and B1 row slices, `l2keep.cu` below 1.0 × L2; hopper-launch: `l2-gate`'s
  wave footprint, `ws-persist`'s separating fact).
- No tailored fixes (`docs/tailored-fixes.md` unchanged).

### Review of the merge (nine confirmed findings)

| # | finding | resolution | evidence |
| --- | --- | --- | --- |
| 1 | bf16x2n / bf16x3n: `±inf` in the narrow operand gave NaN (a split part of `0` or of the other sign times `inf`) | **fixed**: `acc_nonfinite` ORs a non-finite test of every accumulator after the `k` loop into the split guard (every such cell is left `inf` / NaN), so the warp takes the FFMA cold path. A test of each narrow fragment register (first try) cost 6 % of the kernel's SASS; this one adds 64 / 48 instructions once a tile after the loop (A100 / H100 PCIe `llm_pre_t2048_bf16w`: 800 → 864, 1 368 → 1 416), the hot loop's opcodes unchanged (A100: one `SHF` realized as `IMAD`) | test `narrow_infinity_takes_the_guard` (fails without: `NaN vs inf`); generator law 7 (`±inf` in the narrow operand) and store 6 (`bf16 × f32`), coverage `narrow_a` 8 / `narrow_inf` 1 of 64 |
| 2 | the merge ships A100 2048-row tf32x3 tiles whole; the comment claimed the `mlp_2048` cut back | **comment corrected**; the 2048³ whole tile is the measured faster side (tensor-cores-2's own run: whole 0.891–0.901 ms beside `S` = 4 0.915–0.919, one hot loop); `pj_m2048_n4096_k4096` / `_k14336` whole: open (no whole timing) | `split_cut_priced_at_the_realized_rate` docs; `func/gpu.rs` `mma_pieces` |
| 3 | `l2_keep_bytes` half the L2 on the 3090 / A100 / H100, unmeasured | **3090 fixed** from its own sweep: `l2_keep_of_sweep` reads the keep at the plateau–DRAM midpoint (the 3090's stream re-reads a kept set at 1.2 TB/s, not the derived 2.4): 1.03 × 6 MiB; the 5080's sweep from 0.25 × (today, `calib/l2keep-rtx-5080.txt`: plateau 4.1–4.3 TB/s to 1.01 ×) gives its profile's 1.025 × exactly. A100 / H100: their sweeps (from 1.0 ×) show no plateau — below 1.0 ×, the half a placeholder, **open** until `l2keep.cu`'s 0.25 × sweep (the confirmation round's `ops_nsys.sh` step 0) | test `l2_keep_read_off_the_sweeps` (fails at 3 MiB); emission moved on the 3090 only: 14 programs' site tables drop `SITE_COLD` (operands 3–6 MiB), `smlp_4096_256_1024`'s candidates (its `c0-rules.tsv` pin regolded) |
| 4 | 5080 column 0.979 → 0.965 from equal-text rows' drift | **fixed**: matrix rebuilt with equal-text rows at 5fcb401b (null by construction): **0.978** (30 / 39 / 51); changed rows 1.288 → 1.280; the runtime attribution withdrawn (its only A/B is null) | `matrix.md` command; § The RTX 5080 column |
| 5 | "12 / 12 losers → winners" | **corrected** to 11 / 12 (`pj_m128_n6144_k4096` 0.809 under the shipped tf32x3: finding 2 open for it); README, tensor-cores-2, cross-arch | — |
| 6 | tensor-cores-2's A100 probes at a ramp clock | **recorded**: absolute ms, cuBLAS margins and the pipe fraction withdrawn; ratios kept | `tensor-cores-2.md` § Measurements |
| 7 | `rung_replay.tsv` FFMA 5.21 / split 7.09 | **fixed**: 3.832 / 5.213 (raw 5fcb401b); bugs-properties' derived 86.6 (0.54) / 117.7 (0.92) | `rung_replay.tsv`, `bugs-properties.md` |
| 8 | six rented boxes without `evidence.tsv` | H100 (hopper-launch) **rebuilt** from the deterministic emission (12 / 12 kernels by name; the step's intents 0 mismatch; 88 checker mismatches on the bundle's arm names, open); bundle / box scripts now run the pass on the box. A100 (attention-chains): **timing only**, cited as such | `hopper-launch.md` § Intent / realized |
| 9 | `streamk-epi` / cut-price on an unmeasured L2 mechanism | **open** (evidence rule 3): no L2 / DRAM counter on any part (`ncu` captures empty, no local `ncu`); the gate keeps every replayed arm's measured side; the next 3090 round's `ncu` pass decides | `bugs-properties.md` § `streamk-epi` |

Re-measured: nothing on the RTX 5080 — every `benches/algorithms/pt_*` program's host and device text is byte-equal
before / after on the 5080, 4090 and 5090 (201 programs; `pc_*` / `pm_*` panic in both, pre-existing). The moved rows are rented-part rows: the A100 / H100 / H100 PCIe bf16w / bf16kv
programs (6 each: the guard's 64 / 48 post-loop instructions) and the 3090's 14 site-table programs — the next round's
A/B (the confirmation round in flight times the build before this fix). Tests: `cargo test -p mapal-backend-llvm`
(release) all pass.

## Gap-fixes merge

Branch `local-rtx-5080`: `f41d4a83` + `--no-ff` merges of `gapfix/launch-combine` (`a6938770`, merge `57f3ea8f`),
`gapfix/tensor-tile` (`7dd38795`, `c98931a1`) and `gapfix/rows-operands` (`108515bb`, `d945c641`), then the merge's
test fixes (`fe73cd45`), what the integrated run found (`d450014d`, `9abeb446`) and this evidence. The three steps
answer `gaps-disasm.md` (C1 / H / C5, C11 / C3 / C4, C2 / C7 / C8 / C12); their evidence: `gapfix-launch-combine.md`,
`gapfix-tensor-tile.md`, `gapfix-rows-operands.md`. Rental spend of the steps: $0.05 + $0.30 + $0.06 = **$0.41**
(this merge: $0, local only). Raw: `.claude/raw-evidence/local-rtx-5080/gapfix-integrate/` (`tests.log`, `testlogs/`,
`resS/`, `resM/`, the A/B runs `fixab/`, `flyab/`, `swzab/`, `zeroab/`, `zeroab2/`; `scripts/`: `alone.py`, `kern.py`,
`kernsrc.py`, `hostfn.py`, `cmp.py`, `bundle.sh`, `run.sh`; `analysis/`: `alone-<part>.tsv`, `kern-none.tsv`,
`kernsrc.tsv`, `host-none.tsv`, `cmp.txt`, `fix-moved.txt`, `zero-moved.txt`, `zero-fence-moved.txt`, the A/B tables).

| merge | conflicts | resolution |
| --- | --- | --- |
| launch-combine | none | — |
| tensor-tile | `lib.rs` (`GpuRungs::NAMES` length) | 144 names (`load-dispatch`, `zero-host`, `zero-hoist`, `split-price`, `mma-fly`) |
| rows-operands | `lib.rs` (the four rung fields, defaults, names, `rung()` arms), `profile.rs` (`NAMES_COST`), `rules.rs` (replay totals), `rung_replay.tsv` | unions: 148 names; `NAMES_COST` counted (rows-operands' fix, it also covers launch-combine's `split_cell_ps`); both branches' replay lines, totals 158/74/16/23 + 5/0/0/0 + 7/7/0/0 = **170 / 81 / 16 / 23**; the OPEN `l2-gate h100 off` line kept as the old gate's record (the new gate's `on` line beside it) |

### Merged emission against each step alone

The 137 timed programs (bS + bM stems) emitted by the merge, `fac7691b` and each branch tip on seven parts; host and
device IR less `; mapal.` lines, kernel-name / key hashes normalized, lines sorted (`alone.py`); a program equal to no
single step is split per device kernel (`kern.py`, `kernsrc.py`) and per host function and module global
(`hostfn.py`). Device text:

| part | programs | equal to base | one step | none (kernels from several steps) | kernels equal to no step | host equal to base |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| rtx-5080 | 137 | 49 | lc 4, tt 3, ro 56 | 25 | **28 in 12 programs** | 1 |
| rtx-3090 | 137 | 69 | ro 58 | 10 | 0 | 1 |
| rtx-4090 | 137 | 56 | tt 2, ro 79 | 0 | 0 | 1 |
| a100 | 137 | 62 | tt 4, ro 68 | 3 | 0 | 1 |
| h100 | 137 | 53 | tt 5, ro 77 | 2 | 0 | 1 |
| h100-pcie | 137 | 57 | tt 4, ro 69 | 7 | 0 | 1 |
| rtx-5090 | 137 | 58 | tt 1, ro 78 | 0 | 0 | 1 |

- **Host:** every merged host function and module global equals some step's (rtx-5080, rtx-3090): launch-combine's
  site registry / hoisted zeros beside rows-operands' host folds under `iota-law`. With the runtime changed
  (`mapal_gpu.c`, launch-combine) and the registry in nearly every host module, no row is null by construction: the
  whole column is this session's.
- **The 28 kernels equal to no step** are candidates of the decode / skinny-GEMV site tables on the 5080 only (the
  one part with `split_cell_ps`): launch-combine's priced split (`S` 8 → 4) carrying rows-operands' `evict=b`
  (`stream-ld`) or `swz=` (`kvec-swizzle`), and `attn_1024_64`'s flow kernel (its zero no longer folded, launch-combine,
  with the shuffle tree, rows-operands). Orthogonal changes on one kernel: every program value-gated below.
- **Interactions the run found** (not visible as text, fixed in `d450014d` / `9abeb446`, below): tensor-tile's two
  rules on one site (C11 groups the walk, C3's ring priced row-major), launch-combine's hoisted zero past the operand
  generators and the program's clock read.

### Tests

- **Workspace** (`test_par.py --bins 4`, `RUST_TEST_THREADS=16 MAPAL_PAR=4 MAPAL_TESTGEN_JOBS=4`, CPUs 16–31 / 48–63):
  129 binaries, **1549 passed**, 2 binaries failed (wall 1806 s): `site_choice` `candidate_zero_is_the_pre_sprint_emission`
  (rows-operands had not run it: `iota-law` moves all 204 pinned rows — the un-rewritten programs' host folds compute
  a captured iota's element — `stream-ld` 3, `shfl-tree` 4; their OFF arms added) and `resident`'s two host-module
  snapshots (the same iota law). Neither an interaction (`fe73cd45`).
- **After the fixes:** backend lib 133 (+1 `mma_fly_prices_the_grouped_walk`), `stream_ld` (52 generated programs:
  naive 6 / tile 2 streamed, re-read 42, keep 52, share 10, **window 2** refused), `zero_fold` 4
  (+`hoisted_zero_keeps_its_lines_and_its_clock`: 24 generated products, zeros ahead of the generators 9 / after
  them 15 / after the clock 24), `tensor_tile` 2, `site_choice` 9, `splitk` 5, `site_geometry` 8, `resident` 17,
  `gpu_host` 5, `golden_ll` 36, `horizontal` 3, `calls` 6, `half_tile` 5, `large_gemm` 3, `hardware_properties` 2,
  `device_geometry` 11 — pass. Every coverage counter > 0.

### Time (`resS` / `resM`, every competitor leg, 30 reps, `PROBE_REPS=5`, bench lock, CPUs 0–7 pinned)

Load at the start of `resS` (06:21) 5.2, of `resM` (06:33) 1.2 (1-min average; CPUs 0–7 busy 0–2 % at each boundary);
other agents raised the load to 100–125 between about 07:10 and 07:25, during `resM`'s competitor legs. Value gates:
41 + 96 programs and 128 px differentials pass on every arm; every follow-up A/B (below) value-gated, 0 failures.
**Harness boundary (H):** every competitor leg is now timed as a Mapal program times itself (device idle, host clock
around the call and its synchronize; the old event bracket kept as `dev_med_ms`); the rented columns still carry
event-bracket legs, so the 5080 column is not comparable with them until their next round.

| what (RTX 5080, 120 matrix rows) | before (matrix: 5fcb401b + confirmation-fixes) | gap-fixes merge |
| --- | ---: | ---: |
| geomean Mapal / best leg | 0.978 (30 / 39 / 51) | **0.880 (47 / 42 / 31)** |
| Mapal against the old (event-bracket) legs | 0.978 | 0.928 |
| Mapal new / old (geomean) | — | 0.949 |
| best leg new / old (geomean; attention 1.216) | — | 1.054 |
| GEMM cells & projections / MLP / attention / LLM / conv / fan | 1.045 / 1.054 / 0.855 / 1.035 / 0.529 / 1.032 | 0.927 / 0.988 / 0.642 / 0.987 / 0.493 / 0.992 |

33 rows flip better (loss → even / win, even → win), none worse. Per cause (Mapal ms, old → merge; ratio against the
best leg, new boundary):

| cause | row | Mapal ms | ratio | SASS / usage (step's evidence) |
| --- | --- | --- | --- | --- |
| C1 host path | pj_m1_n4096_k4096 | 0.0396 → 0.0301 | 1.760 → 1.263 | site picks at load, the zero a kernel (no memset API) |
| C1 + C7 | lsm_32_8192 | 0.0237 → 0.0162 | 2.326 → 0.839 | launches sooner; unit tree barriers 38 → 6 |
| C1 | softmax_rows_4096 / epi_4096 / rt_mlp2_1024 | 3.97 → 3.80 / 3.79 → 3.54 / 2.75 → 2.55 | 1.122 → 1.033 / 1.019 → 0.991 / 1.094 → 0.992 | host path; `softmax_rows`' 64 MiB zero now inside the clock (below) |
| C5 split price | pj_m16_n4096_k4096 / pj_m16_n2048_k7168 / mm_16x2048x7168 | 0.0495 → 0.0388 / 0.0416 → 0.0346 / 0.0879 → 0.0799 | 0.895 → 0.656 / 0.846 → 0.666 / 1.099 → 1.000 | `S` by the stream's spread against the partials |
| C11 raster reuse gate | pj_m2048_n28672_k4096_bf16 / _tf32 | 4.555 → 3.987 / 9.048 → 8.532 | 1.139 → 1.003 / 1.123 → 1.057 | grouped walk, `b` read once (with the ring fix below) |
| C3 mma-fly ring | pj_m128_n28672_k4096_bf16 | 0.4024 → 0.3426 | 1.268 → 1.021 | `DEPBAR.LE SB0, 0x2`, `bk` 16 × 4 |
| C2 stream-ld | pj_m1_n6144_k4096 / pj_m16_n6144_k4096 / pj_m1_n4096_k14336 / pj_m1_n4096_k14336_bf16 | 0.0827 → 0.0687 / 0.0979 → 0.0861 / 0.2481 → 0.2238 / 0.1015 → 0.0780 | 1.614 → 1.275 / 1.111 → 0.957 / 1.081 → 0.956 / 0.719 → 0.532 | `b` loads `LDG.E.EF` |
| C7 shfl-tree (+ C12) | attn_1024_64 / attn_2048_64 / attn_2048_128 | 0.0369 → 0.0356 / 0.0803 → 0.0728 / 0.1163 → 0.1131 | 1.284 → 1.022 / 1.154 → 0.776 / 1.235 → 0.992 | one barrier a tree; swizzled `b` stores |
| C8 iota-law | llm_dec_b32_l2048_bf16kv / llm_dec_b1_l2048 | 0.5105 → 0.4959 / 0.2395 → 0.2195 | 1.099 → 1.047 / 0.925 → 0.837 | the norm's index computed, not loaded |
| rows-operands (not split per rung) | causal_1024_256 / prefill_1024_512_256 | 15.16 → 13.45 / 7.78 → 6.77 | 0.344 → 0.297 / 0.344 → 0.300 | — |
| C5 + C2 / C12 (one site table, not split) | llm_dec_b32_l8192_bf16kv | 1.728 → 1.657 | 1.160 → 1.101 | candidates at `S` 4 with `evict=` / `swz=` (the 28 kernels above) |

### What the integrated run found (fixed)

- **mma-fly × the C11 walk (tensor-tile, both rules on one site).** `pj_m2048_n28672_k4096_bf16` ran 4.516 ms in
  the run, against the step's 3.99: C11 grouped its walk (`raster` 16) and C3, pricing the panel's DRAM bytes
  row-major (`T` 1442 ≤ `t` 1672 < `T` + 573 ns), took the ring `bk` 16 × 4 — the step's 3.99 was the grouped S52
  panel `bk` 32 × 2 (`gaps-disasm` arm). Same-session ABBA: the old bundle's grouped panel 3.98–3.99 against the
  merge's ring 4.51–4.53. Fix (`d450014d`): `Depth::Fly { walk }` carries the walk's raster rule; the gate divides
  each panel's bytes by its readers in the grouped wave (`b` by `h`, `a` by `wave / h`): `T` 233 ns, no ring. ABBA:
  4.516 → **3.987** ms (null 3.989); the 128-row ring kept (unit test). Also moves the A100's
  `llm_blkpre_t2048_bf16` (unmeasured).
- **stream-ld on a window (rows-operands).** `cells = k·c` priced `fir_4194304_16`'s 16 MiB signal (`s[t + k]`,
  every element read by 16 cells) as a 256 MiB read-once operand and streamed it. Read once now needs the
  operand's `tensor_form` (dense, no overlapping axes) with `cells` distinct elements (`d450014d`; test: 12 window
  programs, `testgen::gemm_window_src_at`). Moves fir / firew_rowsum on the 5080 / 4090 / 5090; ABBA null (0.0455 /
  0.0454, 0.0516 / 0.0518).
- **The hoisted zero (launch-combine C1).** `pj_m128_n6144_k4096_tf32` ran 0.180 → 0.187 ms on the same PTX (old /
  new runtime ABBA): its 3 MB output, zeroed ahead of the 100 MB `b` generator, was cold at the atomic adds. And the
  hoist crossed the program's `time` read: the product's own zero (64 MiB for `softmax_rows_4096`) left the timed
  region, which the competitor legs keep inside theirs. Fix (`9abeb446`): the hoist stops where the bytes the working
  steps after it touch (`access_plan`, `residence_plan` bytes) and the zero's own fit `l2_keep_bytes`, and never
  crosses a clock read. ABBA against `d450014d` on the 50 programs it moves: **0.979** (the zeros' issue and fill
  back in the clock: GEMVs 0.89–0.94, softmax / rowsum 4096 0.97; `pj_m128_n6144_k4096_tf32` 1.068); the residence
  bound alone (35 programs) 1.001. The column takes these 50 rows from that run (`zeroab2`): 0.871 → 0.880. Open: a
  zero folded into a kernel (`zero-fold`) is not fenced the same way (no suite program shows one ahead of a clock;
  unchecked on generated programs).
- **Evidence pass.** `evidence.py` did not list `stream-ld` / `kvec-swizzle` as kernel rungs (arch-rungs mismatches
  101 / 1067 → 0 / 297, the old count); hoisted zeros are counted (`; mapal.zero.host hoisted=`, `zero_for`), intent
  zero-fold counts them, new intent zero-hoist. Re-checked: resS 14523 checks / 275 mismatch (5fcb401b protocol run
  of the previous merge: 14433 / 250), resM 121848 / 5779 (120181 / 5764). New mismatches: **kvec-swizzle "no
  spill"** — attention score kernels spill 12–40 B with the swizzle (`attn_2048_64` two kernels 32 / 40 B,
  `attn_2048_128` 12–16, `llm_pre_t8192.xref` 8; none without it), yet the swizzle is faster there (ABBA, rules
  pick both arms: `attn_1024_64` 0.0367 → 0.0334, `attn_2048_64` 0.0820 → 0.0793, `attn_2048_128` 0.1515 → 0.1496):
  open (the spill's place in SASS unread); choice "AUTO launches the stored pick" rows doubled (the class existed
  before: the stored pick 5, launched 0).

### Open (attempted or not reached)

- The steps' own open items stand: C1 first-launch host cost and in-loop hoists; C5's DP + stream-K with workspace,
  uneven `k` cuts and `split_cell_ps` on other parts; C4 (warp-tile efficiency property), C3's A100 64 × 64 split
  (grid fill), the 3090 `smlp_4096_256_1024` stream-K (+22 %, fac7691b); C7 reciprocal (needs an ADR), C8 norm
  spread, C12 copy width; C9, C14 not reached.
- `pj_m1_n4096_k4096` (1.263), `pj_m1_n6144_k4096` (1.275), `pj_m128_n6144_k4096_tf32` (1.378) remain the 5080's
  largest GEMV / TF32 losses; with the zero back in the clock the C1 GEMV gain is the host path's alone.
- Rented columns: every row moved by the three steps and these fixes re-times at the next round, with the H harness.
- No tailored fixes (`docs/tailored-fixes.md` unchanged).
