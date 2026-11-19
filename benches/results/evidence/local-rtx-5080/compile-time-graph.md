# compile-time-graph — row-block units and launch regions (2026-11-20, rtx-5080 + Threadripper 9970X)

Code: `2521a051` + `0ca6347e`, and the review fixes beside this revision (§ Review fixes). Design:
`.claude/sprints/compile-time-graph/design.md`. Raw output (gap tables, box logs, CUPTI traces, ABBA, calibration,
goldens) is kept locally in `.claude/raw-evidence/local-rtx-5080/compile-time-graph/` (not in git).

## What changed

| | fact / rule | OFF arm |
| --- | --- | --- |
| A | `unit_plan(f)` (mapal-ir, stored in `FuncFacts`, CLAUDE.md facts table): row-block groups — tile sites + the folds and cells reading their rows; every in-group edge row-closed (`Reach::within_rows`, now also `flow`'s row test); a tile's column operand never a member; stored = escapes or read outside; on-chip rows colored by death in the group (in place along a same-cell chain). The `flow` rule makes a placed group one unit kernel of `b` rows a block: `b` from opt-in smem, accumulators and the plan model (each tile a `b × c` tile re-reading its column operand, plus the sector waste of reading it straight from L2) against the sites' best candidates + split launches; between tiles the members run as row groups (`lanes ÷ P` lanes a row, `P` rows at once); a member writing an on-chip row buffer waits (barrier) for every reader of the buffer's earlier occupant in its segment. A part without plan rates, or a shape the model cannot price, keeps the boundary (`unpriced`). Shipped default: the `row-block` rung's `GPU_RUNGS_OFF` rows keep it off on every measured part (no realized unit measured to pay; the rtx-5080 takes the rtx-5090's row). | `--no-row-block`; `--gpu-on=row-block` runs the cost-gated rule; `--row-block=<b>` forces `b` (measurement) |
| B | A device loop's trip is a launch region: each device op tagged with its compile-time dependencies (RAW/WAR/WAW over device bytes — carries, arena overlap, numbers in one-element buffers; a zero pass waits only for its output's readers/writers). The runtime records the trip as graph nodes, instantiates per topology (two alternating executables), replays with changed parameters patched; a host step inside flushes eagerly; any graph failure runs the trip eagerly. Emulator: seeded random topological replay. `TargetProfile`: `graph_launch_ns`, `set_params_ns` (region gate). | `--no-launch-graph`, `MAPAL_GPU_GRAPH=0` (2 = program-order edges) |

Both arms OFF: emission byte-identical to `4206028f` — 2230 host/device modules (191 programs × rtx-5080 / rtx-4090 /
h100-pcie × default / exact face) and 382 CPU (x86-64-v3) modules with the arms ON; the pre-sprint golden
(`site_choice::candidate_zero_is_the_pre_sprint_emission`) passes with the two new OFF flags.

Shipped default after the review fixes, against `0ca6347e` (191 programs × rtx-5080 / rtx-4090 / rtx-5090 / h100-pcie /
a100 / rtx-3090 × default / exact face, 4476 host/device modules): 4236 identical but for the host intent's
`row_block=…` refusal comment; the other 240 are the 120 program × part × face builds where `0ca6347e` formed a
row-block unit by default, and each now equals `0ca6347e --no-row-block` (the OFF arm) — the kernels the review found
shipping unmeasured (rtx-4090 / 5090 attention, h100-pcie / a100 / rtx-3090 attention, rowsum, softmax_rows, pm, pc,
chains with no cost gate at all; rtx-5080 `px_attn_64_16` and the exact-face decode) are gone. No timed suite build
(face `_default`) on the rtx-5080 changes, so no suite row is re-measured. Kernel ids are unchanged (the new rung is
not part of a kernel's key), so the box's wisdom stays valid.

## Tests

| test | result |
| --- | --- |
| `row_block.rs` (7) | the fact on attention (one group, colors s/ex/p in place, mx/den; stored only `o`), an outside reader stores, unclosed reads refused; attention values ON (b = 1, 2, 4, 8 × exact/default face, a per-row value and a row read across the second tile) = interpreter = OFF, one launch, free-running and serialized; rtx-5080 `pt_attn_1024_64`: shipped off, with the rung on refused by cost, forced one launch; every GPU part ships no unit for `pt_attn_1024_64` / `px_attn_64_16`, and the parts without plan rates refuse `unpriced`; 1000 generated chains (400 `chain_strategy` + 600 `wide_chain_strategy`, epilogue rung off): 795 groups, 278 formed (162 with row groups over several warps, 64 with tree folds, 116 split tiles, 417 FMA, 3 programs reusing a freed row buffer — each count required > 0), at b = 1 and the largest b ≤ 8, exact face = interpreter, default face within 1e-4 of the interpreter or its OFF arm, every run serialized in thread order and in reverse; each unit holds exactly the fact's members, ON launches ≤ OFF. Without the barrier fix the same test fails (chain#739, exact face, b = 1 and 2) |
| `launch_graph.rs` (4) | decode tags = the dataflow; decode values under 6 shuffled orders + runtime OFF + compile OFF, one graph a trip; tags with edges dropped are caught; causal step (whole folds): every order × site candidate; 280 generated loops (160 `twin_prog_strategy` + 120 `twin_number_strategy`): 44 with a region, 9 of them reading a whole fold's number; values under shuffled replay, tags name earlier ops only. With number-blind tags (the v1 bug) the generated test fails (twin#164, #208, #216 …), not only the hand-written causal test |
| emulator | `MAPAL_EMU_SERIAL=1` / `2` (`mapal_gpu_emu.c`): a threaded block runs one thread at a time from barrier (or warp meet) to barrier, in thread order / reverse, so a missing barrier is a wrong value on every run; the reviewer's race program (`s = a·b`, row max, a cell of the max alone, its row sum) gives 175397.92 / 350157.97 for 2677.9375 deterministically before the fix, the interpreter's values after |
| suite | mapal-ir, llvm backend, cli: 600 tests in 39 binaries, all ok (`test-final.log`); older rung tests pin the new arms off where they pin launch structure; `benches/gpu/test_harness.py` 27/27. After the review fixes: mapal-ir, llvm backend, cli, mapal-rewrite: 691 tests in 59 binaries, all ok (the pre-sprint golden included) |

## A — intended vs realized (pt_attn_1024_64)

| | intended (fact + rule) | realized |
| --- | --- | --- |
| fact | 1 group: s(tile) mx ex den p o(tile), rows 1024, 4100 B a row, 512 KB column operands | `unit_plan` as intended (`print_plan_file`) |
| forced b = 16 | 64 CTAs × 256, 83 968 B smem, 1 launch, no S/P in memory, no split-K zero, FFMA | ptxas 220 regs, 0 spill; SASS 1297 FFMA, 64 LDS.128, 80 LDG.128, 16 LDG.32 (V), 26 BAR, 16 STG; host 1 `launchv`, 0 `mapal_gpu_site`, 0 zero |
| kernel time (CUPTI) | model 15.2 us fused vs 18.0 split | fused **33.3 us** (rows in sequence: 44.2) vs split 10.5 QKᵀ + 7.5 softmax + 12.9 P·V(S=8 atomic) + 0.8 zero = 31.7 |
| phases (forced unit, CUPTI) | — | QKᵀ + mx 20.2 us, + ex/den 21.5, full 33.3: the in-unit QKᵀ costs 2× its tile site |
| cause (SASS) | — | the unit reads Kᵀ straight from L2: lanes 256 B apart, 16 B a lane (half of each 32-B sector); the site stages panels; 1 CTA/SM (220 regs, 84 KB) |
| decision | model + sector term: fused 18.9 > split 18.0 → `row_block=s:cost`, boundary kept | shipped attn = OFF emission (kernels identical); ABBA forced/shipped 1.011 (median, 16 processes). Since the review fixes the rung ships off (below); `--gpu-on=row-block` still refuses it by cost |

With the rung on (`--gpu-on=row-block`), the other rtx-5080 suite groups: `rowsum_2048/4096`, `softmax_rows_4096`,
`decode` (s, o) refused by cost (the K = 2048–4096 column operand re-read by every unit); MLPs, `epi`, `loop`, `firew`,
`indep`, `smlp` refused `placed` (the tile already has an epilogue group, which the rule leaves alone).

**Correction (review).** The first revision said "no suite program changes kernels by default". That held only for the
rtx-5080's timed (`_default` face) builds. At `0ca6347e` the arm was on by default everywhere, and it formed units
nobody had measured: on the rtx-4090 / rtx-5090 `pt_attn_1024_64/1024_128/2048_64` (5090 model 16.1 vs 34.1 us
split) and `px_attn_64_16`, plus on the exact face decode (both) and pc, pm, rowsum, softmax_rows, chain (4090); on
h100-pcie / a100 / rtx-3090 (no plan rates) attention, rowsum, softmax_rows, pm, pc and the chains, at the first `b`
that fit with no cost test; on the rtx-5080 `px_attn_64_16` and the exact-face decode; and any shape the model could
not price, at the first `b` that fit. The model's 4090 / 5090 admission of attention
comes from one term: the split softmax segment is priced at `bytes_per_ns`. That rate is a DRAM placement rate on
those parts (490 / 494 B/ns: 12.6 MB of L2-resident rows priced at 25.5 us on the 5090) but an L2 rate on the
5080 (2553 B/ns: 4.9 us; measured 7.5 us). The fix: the rung is off on every measured part (`GPU_RUNGS_OFF`
`row-block`, "unmeasured;" rows), and an unpriced group is refused. The one realized unit lost on the one part where it
was measured, so no part ships the rule until a realized unit is measured to pay there.

## B — intended vs realized (rtx-5080, CUPTI, one traced run each)

| program | intended | realized |
| --- | --- | --- |
| decode_1024_256 | 1 region, 6 ops, 7 edges: x → kk, vv; kk, q → s; s, vv, x → o; zeros off the critical path; 1 graph a trip | 1022 graphs (2 trips flush: first pick of each table), 6146 kernels + 4096 memsets unchanged, 6 patches a trip; memsets overlap kernels (idle −0.29 us), but kernel edges idle 1.1–1.7 us (eager 0.3–0.7); `cuGraphLaunch` **15.8 us a trip** (10 nodes; `MAPAL_GPU_GRAPH=2` chain: same) ≈ the 6 + 4 eager calls it replaces (traced 2.1 + 1.5 us each) |
| causal / gen / prefill | 9 ops, 10 edges (whole folds `mx`, `den` as numbers) | values within bound (v1 without number edges: `VALUE-FAIL` on 8 causal arms — fixed, test added); traced causal span 20.8 vs 24.7 ms |
| rt_mlp2_1024 | 6 ops, 2 independent chains | the graph runs the chains concurrently: busy 3211 us > span 2690 us |
| attn, mlp_2048, ew_4096, conv_2048_5 (no loop) | unchanged | identical kernels, launches, memsets |

Calibration (`calib graph`, `calib/graph-rtx-5080.txt`): eager 1.32–1.58 us an op; a 6-kernel chain replays in 4.10 us,
patching 0.16–0.19 us a node → `graph_launch_ns` 2050, `set_params_ns` 173 (the gate keeps every region of ≥ 2 ops).
**Mismatch:** the real decode graph launches in 15.8 us, not ≤ 7.4 us — the synthetic graph is not the region's shape.

## Time (ms; ON/OFF/ON suite runs: `LADDER=1 gpu_run.sh`; ABBA: same box, same wisdom, identical device modules)

The suite columns are separate runs in different tuning states: OFF = run-off2, which reused run-off1's wisdom (no
trials); ON = the mean of run-on3 (retuned: 28 sites trialled) and run-on4 (run-on3's wisdom, no trials). Of the 28
programs with site tables, three ran different picks in the two columns: attn (site 705a20a0: cand 4 OFF, 0 ON),
indep_512 (38612e7a: 6 / 5), twin_2048 (6ee3e302: 0 / 3). Their ON/OFF ratios are tuning-state differences, not the
arms' effect and not null.

| workload | OFF | ON | best competitor | ABBA ON/OFF (median, min) | explanation |
| --- | ---: | ---: | --- | --- | --- |
| attn_1024_64 | 0.0503 | 0.0434 | 0.0297 torch-compile | forced b16: 1.011, 1.084 | shipped = OFF kernels (cost refusal); suite runs differ by tuned picks + cross-run null (OFF1 0.0418) |
| decode_1024_256 | 17.51 | 17.10 | 15.81 cache | 1.044, 0.984 | host-bound either way: graph launch ≈ eager issue (above) |
| causal_1024_256 | 21.58 | 21.38 | 51.39 torch-compile | 0.987, 0.959 | same |
| gen_1024_512_256 | 11.04 | 10.78 | 26.09 torch-compile | 1.000, 0.993 | same |
| prefill_1024_512_256 | 11.26 | 11.58 | 25.99 torch-compile | 1.025, 1.036 | same (within null 5.7–7.2 %) |
| rt_mlp2_1024 | 2.581 | 2.375 | 2.53 torch-compile | **0.917, 0.899** | two independent GEMM chains overlap in the graph |
| rt_mlp2_2048 | 16.84 | 16.65 | 16.35 torch-compile | 0.994, 0.985 | the chains fill the GPU alone: nothing to overlap |
| indep_512 | 0.0495 | 0.0406 | — | picks 6 → 5 (same binary): **0.823**, 0.866 | identical emission; the 0.82 is the tuned pick at site 38612e7a (cand 6 OFF vs 5 ON), reproduced by ABBA with only the wisdom changed |
| twin_2048 | 0.988 | 0.941 | — | picks 0 → 3 (same binary): **0.955**, 0.955 | identical emission; the pick at site 6ee3e302 (cand 0 OFF vs 3 ON) |
| every other row (26) | — | ON/OFF 0.98–1.10 | — | — | identical emission and no pick difference: cross-run variation (the ends: fir 0.98, ew_4096 1.04, pj_m1_n2048 1.10) |

Null: within a run 0.0–11 % (gap tables); between separate runs up to 20 % on the 40-us rows (attn OFF 0.0418 vs 0.0503).
Pick ABBA (`review-fix/abba_picks.sh`, 16 processes an arm, `MAPAL_GPU_TUNE=0`, each arm's wisdom differing only at
that site; traced picks confirmed): arm spread indep_512 47 % / 28 % (40-us row), twin_2048 1.9 % / 1.1 %.

## Review fixes (independent review of `2521a051`..`0ca6347e`)

| finding | fix |
| --- | --- |
| row-block race: `unit_plan` gives a later kept cell the row buffer of a member that died earlier in the same `rows_loop` segment; only an in-place writer waited | `rows_loop` puts a barrier before every member whose row buffer a member of its segment read (`UnitGen::reread`: in place, or freed at an earlier death); emission changes only where a buffer is reused that way (attention's in-place chain keeps its barriers, kernels identical) |
| arm on by default, unmeasured parts, no cost gate without plan rates or for an unpriced shape | `row-block` rung, off on every measured part (`GPU_RUNGS_OFF`, "unmeasured;"); `unpriced` refusal; the 4090 / 5090 admission traced to `bytes_per_ns` (above) |
| generated test barely reached the new paths (11 / 329 groups, ≤ one warp, exact face only) | `wide_chain_strategy` (power-of-two rows to 128 cells, `k` to 64, a row broadcast node, operands made first), both faces, b = 1 and max, serialized emulation, coverage asserted; `twin_number_strategy` for whole-fold numbers in launch regions; both catch their bug when it is put back |
| "every other row … identical kernels: null; indep_512 0.82" | wrong: the suite columns compare different tuning states (below) |

## Open

1. **A's realization of a tile in a unit** stages nothing: Kᵀ crosses at half a sector a lane, so q·Kᵀ costs 2× its
   site; with shared-memory panels (the site's machinery) the unit would hold attention in one kernel profitably.
   The rule, fact and the forced kernel are in place; the model's sector term is the fact that keeps the boundary.
2. **B's graph launch cost** on this driver (15.8 us for decode's 10 nodes) is not the calibrated 2.05 us: a
   missing fact (launch cost by node count / shape). Under it the region gate would refuse decode and keep
   rt_mlp2 only where independent chains overlap — the separating fact (concurrent device work per trip) is not yet
   read by the gate.
3. Zero passes are memset nodes off the critical path, not folded into earlier kernels; straight-line regions
   (run once) stay eager (no persistent arena / load-time instantiate yet); device-side counters + a conditional
   node for whole loops are not done.
4. A tile with an epilogue group is refused (`placed`): the rule does not yet weigh dissolving an epilogue.
5. **Row-block ships nowhere** (`GPU_RUNGS_OFF` `row-block`, a stopgap): no realized unit has been measured to pay
   on any part. Missing fact for the cost model: where a split segment's bytes live (L2-resident vs DRAM), which
   `bytes_per_ns` stands in for, calibrated as an L2 rate on the 5080 and a DRAM placement rate on the 4090 / 5090.
   A small, launch-bound attention (`px_attn_64_16`: model 1.41 vs 2.09 us on the 5080) is the first candidate to
   time; a timed probe of it does not exist yet.
6. The tuner's trial at indep_512's site 38612e7a timed candidates 5 and 6 as a tie (21.59 vs 21.56 us a call,
   run-off1) and stored 6. In the program, 5 runs 18 % faster. The trial's per-call time misses what separates
   them (a per-site-choice item).
7. Many generated groups are refused `footprint` (a tile row neither dividing nor a multiple of the block's lanes)
   or `phase` (an operand made after the group's first member): the rule does not reorder independent operand
   producers before the launch.
