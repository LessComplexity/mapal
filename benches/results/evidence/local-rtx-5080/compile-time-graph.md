# compile-time-graph — row-block units and launch regions (2026-11-20, rtx-5080 + Threadripper 9970X)

Code: `2521a051` + the follow-up commit beside this report. Design: `.claude/sprints/compile-time-graph/design.md`.
Raw output (gap tables, box logs, CUPTI traces, ABBA, calibration, goldens) is kept locally in
`.claude/raw-evidence/local-rtx-5080/compile-time-graph/` (not in git).

## What changed

| | fact / rule | OFF arm |
| --- | --- | --- |
| A | `unit_plan(f)` (mapal-ir, stored in `FuncFacts`, CLAUDE.md facts table): row-block groups — tile sites + the folds and cells reading their rows; every in-group edge row-closed (`Reach::within_rows`, now also `flow`'s row test); a tile's column operand never a member; stored = escapes or read outside; on-chip rows colored by death in the group (in place along a same-cell chain). The `flow` rule makes a placed group one unit kernel of `b` rows a block: `b` from opt-in smem, accumulators and the plan model (each tile a `b × c` tile re-reading its column operand, plus the sector waste of reading it straight from L2) against the sites' best candidates + split launches; between tiles the members run as row groups (`lanes ÷ P` lanes a row, `P` rows at once). | `--no-row-block`; `--row-block=<b>` forces `b` (measurement) |
| B | A device loop's trip is a launch region: each device op tagged with its compile-time dependencies (RAW/WAR/WAW over device bytes — carries, arena overlap, numbers in one-element buffers; a zero pass waits only for its output's readers/writers). The runtime records the trip as graph nodes, instantiates per topology (two alternating executables), replays with changed parameters patched; a host step inside flushes eagerly; any graph failure runs the trip eagerly. Emulator: seeded random topological replay. `TargetProfile`: `graph_launch_ns`, `set_params_ns` (region gate). | `--no-launch-graph`, `MAPAL_GPU_GRAPH=0` (2 = program-order edges) |

Both arms OFF: emission byte-identical to `4206028f` — 2230 host/device modules (191 programs × rtx-5080 / rtx-4090 /
h100-pcie × default / exact face) and 382 CPU (x86-64-v3) modules with the arms ON; the pre-sprint golden
(`site_choice::candidate_zero_is_the_pre_sprint_emission`) passes with the two new OFF flags.

## Tests

| test | result |
| --- | --- |
| `row_block.rs` (6) | the fact on attention (one group, colors s/ex/p in place, mx/den; stored only `o`), an outside reader stores, unclosed reads refused; attention values ON (b = 1, 2, 4, 8 × exact/default face, a per-row value and a row read across the second tile) = interpreter = OFF, one launch; rtx-5080 `pt_attn_1024_64` refused by cost, forced one launch; 400 generated chains: 329 groups, 11 formed, values exact, each unit holds exactly the fact's members, ON launches ≤ OFF |
| `launch_graph.rs` (4) | decode tags = the dataflow; decode values under 6 shuffled orders + runtime OFF + compile OFF, one graph a trip; tags with edges dropped are caught; causal step (whole folds): every order × site candidate; 160 generated loops (26 with a region): values under shuffled replay, tags name earlier ops only |
| suite | mapal-ir, llvm backend, cli: 600 tests in 39 binaries, all ok (`test-final.log`); older rung tests pin the new arms off where they pin launch structure; `benches/gpu/test_harness.py` 27/27 |

## A — intended vs realized (pt_attn_1024_64)

| | intended (fact + rule) | realized |
| --- | --- | --- |
| fact | 1 group: s(tile) mx ex den p o(tile), rows 1024, 4100 B a row, 512 KB column operands | `unit_plan` as intended (`print_plan_file`) |
| forced b = 16 | 64 CTAs × 256, 83 968 B smem, 1 launch, no S/P in memory, no split-K zero, FFMA | ptxas 220 regs, 0 spill; SASS 1297 FFMA, 64 LDS.128, 80 LDG.128, 16 LDG.32 (V), 26 BAR, 16 STG; host 1 `launchv`, 0 `mapal_gpu_site`, 0 zero |
| kernel time (CUPTI) | model 15.2 us fused vs 18.0 split | fused **33.3 us** (rows in sequence: 44.2) vs split 10.5 QKᵀ + 7.5 softmax + 12.9 P·V(S=8 atomic) + 0.8 zero = 31.7 |
| phases (forced unit, CUPTI) | — | QKᵀ + mx 20.2 us, + ex/den 21.5, full 33.3: the in-unit QKᵀ costs 2× its tile site |
| cause (SASS) | — | the unit reads Kᵀ straight from L2: lanes 256 B apart, 16 B a lane (half of each 32-B sector); the site stages panels; 1 CTA/SM (220 regs, 84 KB) |
| decision | model + sector term: fused 18.9 > split 18.0 → `row_block=s:cost`, boundary kept | shipped attn = OFF emission (kernels identical); ABBA forced/shipped 1.011 (median, 16 processes) |

Other suite groups: `rowsum_2048/4096`, `softmax_rows_4096`, `decode` (s, o) refused by cost (the K = 2048–4096
column operand re-read by every unit); MLPs, `epi`, `loop`, `firew`, `indep`, `smlp` refused `placed` (the tile already
has an epilogue group, which the rule leaves alone). No suite program changes kernels by default.

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

| workload | OFF | ON | best competitor | ABBA ON/OFF (median, min) | explanation |
| --- | ---: | ---: | --- | --- | --- |
| attn_1024_64 | 0.0503 | 0.0434 | 0.0297 torch-compile | forced b16: 1.011, 1.084 | shipped = OFF kernels (cost refusal); suite runs differ by tuned picks + cross-run null (OFF1 0.0418) |
| decode_1024_256 | 17.51 | 17.10 | 15.81 cache | 1.044, 0.984 | host-bound either way: graph launch ≈ eager issue (above) |
| causal_1024_256 | 21.58 | 21.38 | 51.39 torch-compile | 0.987, 0.959 | same |
| gen_1024_512_256 | 11.04 | 10.78 | 26.09 torch-compile | 1.000, 0.993 | same |
| prefill_1024_512_256 | 11.26 | 11.58 | 25.99 torch-compile | 1.025, 1.036 | same (within null 5.7–7.2 %) |
| rt_mlp2_1024 | 2.581 | 2.375 | 2.53 torch-compile | **0.917, 0.899** | two independent GEMM chains overlap in the graph |
| rt_mlp2_2048 | 16.84 | 16.65 | 16.35 torch-compile | 0.994, 0.985 | the chains fill the GPU alone: nothing to overlap |
| every other row | — | ON/OFF 0.82–1.10 | — | — | no emission change (identical kernels and launches): null; the ends are indep_512 0.82, pj_m1_n2048 1.10 |

Null: within a run 0.0–11 % (gap tables); between separate runs up to 20 % on the 40-us rows (attn OFF 0.0418 vs 0.0503).

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
