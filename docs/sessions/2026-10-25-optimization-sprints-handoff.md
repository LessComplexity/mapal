# 2026-10-22 → 25 — optimization sprints and handoff

Immutable. One conversation, eight sprints, all local commits (never pushed). Main checkout ends on `flow-latest` @ `6369853`.

## Sprints, branches, outcome

| sprint (branch) | outcome |
| --- | --- |
| integration (`s62-integrated` @ 3ec494e) | 6 items merged (geometry, asyncstage, splitk, cpugpu, runtime, cli); reassoc slab-offset bug found by value gates and fixed; S62 gap map G1–G9 |
| default float face (`s63/default-float` @ c0103c5) | default = full precision, a-priori bound, order free (ADR-0040); exact/contract/fixed opt-in; atomic split-K + stream-K rungs |
| evidence pass (`s64/evidence` @ 5d594e9) | intent registry + checker (`intent.py`, `evidence.py`, `rejudge.py`); 92 rows re-judged; multi-workload suite |
| deduced residency (`s65/residency` → `deduced-residency` @ 27b4f99) | loop state by reference inside the existing allocators (`view_death` fact); host copies of carried state 4–320 MB/trip → 0; KV retention emerges on decode; `benches/algorithms/*.mapal` committed |
| natural forms + generic fusion (`fusion-natural-forms-latest` @ e90dd90) | idiomatic generators (decode = one guarded map), `Outside` piecewise law, `write_delta` through maps; `footprint_plan` fact; window row folds; first fuse rule lost everywhere |
| facts on graph (`facts-on-graph-latest` @ a2db415) | facts stored on the sealed IR: compile −20–32 %, property test −24 %, byte-identical; dead kernels removed (IR −14–16 %); causal decode, prefill+decode, multi-channel conv programs |
| flow (`flow-latest` @ 6369853) | harness bug behind exact-face "failures" fixed; `fuse` → `flow`, rebuilt (lanes per row, tree folds, compute-once rows) and shipped ON (attn 0.69–0.81, firew_rowsum 0.52–0.77, softmax 0.88–0.93 vs OFF); why-slow analysis |

## Decisions by the user (binding, in CLAUDE.md / memory)

- Every optimization derives from IR facts + TargetProfile, emitted so LLVM derives the schedule; no workload-named
  cases; no parallel module re-deriving existing facts; facts live on the graph (compute once).
- Generic, proven on unseen cases (testgen property tests); benchmarks are acceptance only, written in idiomatic Mapal.
- No "fusion" concept — compile by data usage in the dataflow graph; kernel boundaries are a result.
- Evidence pass mandatory: intent → remarks/PTX/SASS/ptxas → explained timing; movement + launch intents; best-of
  library legs and torch, never torch alone.
- Default float: full precision, bound, order unspecified; determinism is not a requirement.
- Priority: win/tie the leading implementation on every algorithm and chain before coverage (StableHLO/PJRT later).
- Focus next: the CPU↔GPU boundary (placement, round-trips, allocation, dispatch, transfers).
- Ops: Sonnet 5.5 for GPU-operation agents, Opus 5.5 for code/design; ~300k tokens per sprint; no mid-run restarts;
  topic names, never session numbers; main checkout switched to each sprint's result branch.

## Latest standing (flow sweep, Mapal / best other)

At or ahead everywhere: conv 5×5, ew→conv→ew, FIR, loop_4096. Far behind: prefill 5.2–8.6, causal 2.7–5.6,
attention 4.2–8.8, rt_mlp2 2.3–5.6, ew_4096 2.3–5.3, convmc host-only. Causes and recoverable ms:
`docs/performance/evidence/why-slow.md`.

## Open ends

Next sprint `cpu-gpu-boundary` (scope in `docs/next-session.md` and `.claude/sprints/cpu-gpu-boundary-scope.md`).
Rentals: none of ours left. Token use was the binding constraint (≈43 % of the weekly budget by 2026-10-24).
