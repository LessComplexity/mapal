# Sessions 49–50 — the scheduler was the kernel

Date: 2026-08-09 → 2026-08-10 · by: Claude (category-architect) · machines: Apple M4 Pro
(local), Intel i9-14900F (arch box, `100.81.226.103`).

## 0. Continuation brief

Sapir asked for a benchmark: a 3–4 layer linear NN, NumPy vs Mapal, on both machines,
at scale, plus **two nets at once** to show that independent work parallelises without
being asked to. The benchmark could not be run as specified, and finding out why became
the session.

## 1. What was asked vs what was found

**Every chained bulk program in the tree was emitting fully sequential code.** `nn2`,
`nn4`, `nn2x2` at every size emitted **zero** `mapal_par_*` calls; so did
`benches/shapes/attn_256.mapal`, the project's own chained-matmul corpus shape, in the
tree since S25. A single matmul emitted 70.

`path_plan` grouped scalar morphisms into **undirected** components of a "shares an
object in any role" graph. Every matmul site's capture tuple captures the same `k`
range, so one shared *read* fused all of them into one task; layer 2's group then reads
layer 1's output, so that task sat both before and after the layer-1 map. A cycle.
`llvm/src/lib.rs:261` filters a cyclic plan out with `path_plan_is_acyclic` and emits
the sequential form — **so the symptom is silently lost parallelism**, and
`path_plan`'s own `debug_assert` is compiled out of release. In debug: `left: 6,
right: 11`.

**And the numbers those programs printed were fiction.** Pre-fix `nn2_2048` self-timed
its forward pass at **1.81 ms** — 34.4 GFLOP in 1.81 ms is 19,000 GF/s on a part that
caps near 4,100. The clock fence (plan-s33b) is implemented as *task dependency edges*,
which exist only on the parallel path; the sequential fallback has no fence and
`topo_order` put the work outside the bracket. Whole-process wall settled it: pre-fix
49.62 ms vs post-fix `MAPAL_PAR=1` 48.77 ms — same work, mis-reported.

## 2. What was built, in order

| # | Change | Where | Kind |
| --- | --- | --- | --- |
| S49 | Scalar-component adjacency is dataflow + co-writing, never sibling-reading | `mapal-ir/src/algo.rs` | **general** |
| P0 | A packed tile site is two tasks in the outer run, not a wrapper opening two nested runs | `backend-llvm/func/drive.rs` | tile sites |
| P0b | A dispatch wakes only the lanes it filled (`LanePark` per lane) | `mapal-rt` | **general** |
| — | Dep-unlocked tasks are `Placement::Seed`, not `Local` | `mapal-rt` | **general** |
| P2 | Run graph sealed once; completion path lock-free | `mapal-rt` | **general** |
| P1 | Host spins before parking; **workers do not**; one executor per lane | `mapal-rt` | **general** |
| P3a | A one-slice task runs on the calling thread | `mapal-rt` | **general** |
| P3b | Roofline dispatch threshold, three measured `TargetProfile` fields | `backend-llvm` | tile sites |

`mapal-ir` learned **no machine fact**: its whole diff is the S49 graph rule plus tests,
and `git diff -- crates/mapal-ir/` matches `pack|lane|wake|notify|thread|dispatch` on
four lines, two of them reflowed comments and two `IrBuilder::pack` in a test.

## 3. Decisions

1. **Fix the cycle rather than report the benchmark as impossible.** The requested
   measurement was unobtainable otherwise, and the root cause was one adjacency
   expression.
2. **`Placement` collapses to one variant.** `Local(lane)` was affinity reasoning valid
   only while dep-unlocked tasks were scalar glue; P0 made them many-slice bulk
   dispatches. Measured: `nn4_128` 0.2447 → 0.1550 ms.
3. **Spinning workers are refuted, not deferred.** 0.80–0.93× on real shapes at every
   budget swept (16…4096). `handoff_floor` over-predicted 30–40× because it gave every
   thread work every wave; a real dispatch leaves most lanes idle, and an idle spinner
   burns a core the working lanes need.
4. **Host-only spin ships**, budget 8192 ≈ 8 µs, derived from the 11–41 µs park-and-wake
   it replaces, zero when oversubscribed (the 1.083 → 1640.9 µs cliff at 13 → 14
   executors).
5. **`MAPAL_PAR` now means total executors including the host.** `Pool::new` spawns
   `threads - 1` workers because `help_until` already claimed lane `threads - 1`.
   Measured neutral in isolation (0.99–1.02×). **This changes what a number taken at a
   given `MAPAL_PAR` means** and is flagged for Sapir.
6. **A profile with no measured dispatch economics carries `None` and dispatches
   everything** — no part can be made worse by a number nobody took. Verified: 0 of 174
   cells move under `--target=generic`.
7. **`dispatch_ns` is derived from a program, not a microbenchmark.** The first value
   (30 µs, the parked wave) cost three cells; the shipped 8.3 µs is
   `(144.9 − 78.8) / 8` measured on `nn4_128` with and without its dispatches.
8. **The threshold decides only for recognised tile sites.** Estimating elementwise maps
   got `transpose_512` and `fir_65536` wrong in the expensive direction; a wrong
   "inline" costs the whole speedup while a wrong "dispatch" costs one wave.

## 4. Results

### Linear NN, M4 Pro, threaded — one run, 41 cycles, `benches/results-s50final/`

Values bit-identical to NumPy on every cell (`rel = 0.000e+00`).

| shape | d | S49 start | S50 close | vs S49 | NumPy | Mapal × NumPy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nn4 | 128 | 0.3636 | **0.0810** | **4.49×** | 0.0234 | 0.29 |
| nn2 | 128 | 0.1720 | **0.0406** | **4.24×** | 0.0136 | 0.33 |
| nn2x2 | 128 | 0.2301 | **0.0758** | **3.03×** | 0.0230 | 0.30 |
| nn4 | 256 | 0.4751 | **0.2613** | 1.82× | 0.0919 | 0.35 |
| nn2x2 | 256 | 0.3161 | **0.1917** | 1.65× | 0.0910 | 0.47 |
| nn4 | 512 | 0.8718 | **0.6333** | 1.38× | 0.6450 | **1.02** |
| nn2x2 | 512 | 0.5712 | **0.4726** | 1.21× | 0.6438 | **1.36** |
| nn4 | 1024 | 3.1681 | **2.9301** | 1.08× | 3.7635 | **1.28** |
| nn2x2 | 1024 | 2.8298 | **2.7707** | 1.02× | 4.0635 | **1.47** |
| nn4 | 2048 | 20.645 | **20.363** | 1.01× | 24.970 | **1.23** |
| nn2x2 | 2048 | 19.957 | **19.677** | 1.01× | 25.646 | **1.30** |

**Read the spreads** (in `s50e` §5): d ≥ 1024 is 3–8%, d = 256–512 is 27–72%, d = 128 is
10–20% at ~80 µs. The same `nn4_128` cell reads 0.0788 / 0.0810 / 0.0827 across the
three runs quoted in the S50 records — one measurement, three samples.

### Two nets at once — same flops, only the dependency structure differs

| d | one 4-layer net | two 2-layer nets | gain from the graph alone |
| ---: | ---: | ---: | ---: |
| 256 | 0.2613 | 0.1917 | **1.36×** |
| 512 | 0.6333 | 0.4726 | **1.34×** |
| 1024 | 2.9301 | 2.7707 | 1.06× |

NumPy's equivalent ratio is 1.00–1.09 at every size **and its control matches it**, so
it has no overlap: the sequential source is what a person writes. The advantage is
filling idle width and decays to nothing once one layer saturates the machine.

### The rest of the ladder — 13 non-matmul shapes, session start → close

saxpy **1.40×** · conv2d_s1024 **1.30×** · transpose_512 **1.29×** · conv2d_512
**1.26×** · conv2d_1024 **1.25×** · conv2d_s1026 1.20× · fir_65536 1.17× ·
transpose_1024 1.13× · gather 1.13× · reduce 1.10× · transpose_1536 1.06× ·
fir_1048576 1.14× · transpose_2048 1.00×. **13 of 13 improved, zero regressions.**
Streaming, stencil, permutation, gather, reduction — no matmul in that list.

### matmul, M4 Pro, threaded (README rows)

| N | S49 start | S50 close | NumPy |
| ---: | ---: | ---: | ---: |
| 1024 | 0.81 | **0.72** | 0.67 |
| 2048 | 5.17 | **5.03** | 5.33 |
| 4096 | 38.6 | **38.7** | 44.1 |

### The runtime, priced

| | M4, 14 executors |
| --- | ---: |
| pool spawn, once per process | 121–126 µs |
| pinned task, no wake (the floor) | 0.21 µs, flat 2 → 14 lanes |
| pooled empty dispatch, before P0b | 5.1 µs |
| pooled empty dispatch, after P0b | 0.46 µs |
| nested run (deleted by P0) | 36.8 µs |
| one wave, spinning vs parked (13 executors) | 1.08 vs 41.2 µs |
| one wave, spinning at 14 executors | **1640.9 µs** |
| one real dispatch, current runtime | **8.3 µs** |

## 5. Bugs found and fixed

1. **Cyclic task plan** (§1) — every chained bulk program sequential. Fixed; pinned by
   `mapal-ir/tests/algos.rs::{assert_acyclic, path_chained_bulk_ops_do_not_fuse_their_scalar_glue}`.
2. **`retire` published `Done` before unlocking dependents** (introduced by P2). A
   waiter could beat the unlock and trip `mapal_par_run_pinned`'s assertion. Survivable
   only while the host always parked; P1's host spin made it fire on `conv2d_1024` in
   seconds. Fixed by ordering; hammered `conv2d_1024` × 300, zero failures.
3. **Zero-length-dispatch path omitted the pinned filter**, so a pinned dependent of an
   empty split could reach a worker where `execute` asserts. Pre-existing; one rule now,
   both paths.
4. **Sequential emission has no clock fence** — recorded, not fixed. Both new harnesses
   hard-error on an emission with no `mapal_par_task`.

## 6. Tests

Gate at close: **`cargo test --workspace --release` → 1056 passed / 0 failed**,
including the 1,280-run differential byte-equality oracle. `cargo fmt --all --check`
clean. **Results identity never moved** — every change was speed only.

New: `path_chained_bulk_ops_do_not_fuse_their_scalar_glue` + `assert_acyclic`
(mapal-ir), `tiny_tile_site_is_inlined_when_the_part_has_measured_costs`,
`the_dispatch_threshold_is_what_separates_the_two_profiles_here`,
`the_dispatch_threshold_is_what_separates_the_two_profiles_here` (llvm).

Re-scoped rather than deleted: `packed_wrapper_dispatches_the_b_pack_across_the_pool`
(now pins the two-outer-task shape, and pins `generic` by name per S46),
`path_wait_uses_max_threshold_and_data_completion_wins` (its two `Div`s are chained, not
siblings), `apple_m4_sme_moves_only_the_sme_field` → `..._declared_capability_fields`,
`sme_profile_is_byte_identical_to_apple_m_...` → `..._differs_from_apple_m_only_in_dispatch_kind_...`.

## 7. Measurement rules earned

> **25. A microbenchmark that gives every thread work every wave does not model a
> dispatch.** `handoff_floor` predicted 30–40× for spinning workers and delivered
> 0.80–0.93× on real shapes, because a real dispatch leaves most lanes idle and an idle
> spinner burns a core the working lanes need. Price a runtime change on real shapes
> before believing the instrument.

> **26. A constant derived from the runtime must be re-derived when the runtime
> changes.** `dispatch_ns` was correct at 30 µs when `handoff_floor` measured it and
> wrong four changes later; using the stale value cost three cells 5–20%.

> **27. Label every table with the run it came from.** Three quotes of one `nn4_128`
> cell (0.0788 / 0.0810 / 0.0827) from three runs read as a regression. Absolutes and
> A/B ratios are not interchangeable across runs; the ratio is valid within a run even
> when the absolute is not.

## 8. Live state at close

| Type | Handle | State | Inspect | Cleanup |
| --- | --- | --- | --- | --- |
| branch | `main` @ `680ce44` | **in sync with origin, nothing pushed, everything uncommitted** | `git status -sb` | Sapir's call |
| machine | arch box `100.81.226.103` | **DOWN** — ssh timed out from mid-session; `raptorlake`'s dispatch numbers are owed | `ssh -o ConnectTimeout=8 100.81.226.103 nproc` | — |
| artifact | `~/mapal-s49`, `~/mapal-s50`, `~/mapal-s50p1`, `~/mapal-s50p2` on the box | scratch, unknown size (box down) | `ssh … du -sh ~/mapal-s5*` | delete when reachable |
| worktree ×12 | `.claude/worktrees/agent-*`, 3 pre-rename | prunable, unchanged since S47 | `git worktree list` | `git worktree prune` |
| results | `benches/results-s49/`, `-s50/`, `-s50p1/`, `-s50p2/`, `-s50p3/`, `-s50final/` | untracked; **`-s50final` is the one the README quotes** | `ls benches/results-s50final` | keep `-s49` and `-s50final`, rest discardable |

**Nothing is running.** No background job, no server, no port. Measurement mutex free.

## 9. Open items

| Priority | Item | Reference | Next action | Done when |
| --- | --- | --- | --- | --- |
| **P0** | single-thread kernel is 0.28–0.35× of NumPy below d=512, 0.87× on the i9 | `s50d` §3 | reuse-structure change, not another blocking level | 1t GF/s moves off ~950 |
| **P0** | `MAPAL_PAR` semantics changed (total executors incl. host) | §3.5 | Sapir sign-off, or revert | decided |
| P1 | dispatch threshold cannot decide for elementwise maps | `s50e` §6 | per-element read count + achieved-rate class; `elem_plan` records what `out[i]` IS | `transpose`/`fir` estimated within 2× |
| P1 | `raptorlake` owes 3 dispatch numbers | `s50e` §2 | run `dispatch_cost` + `handoff_floor` on the box | profile carries `Some` |
| P1 | `path_plan_is_acyclic` still load-bearing | `runtime/suggestions.md` #4 | enforce the DAG contract or make the demotion loud | fallback deletable |
| P1 | sequential emission has no clock fence | §5.4 | fence, or refuse to self-time a sequential program | a sequential `time` bracket is honest |
| P2 | ARCHITECTURE §4 rules unpinned | `runtime/suggestions.md` #3 | one test per rule | 6 pins |
| P2 | `Work` carries an `Arc<Run>`; `take_any` miss path is O(N) | `runtime/suggestions.md` #1, #2 | — | — |
| P2 | S48 work still uncommitted alongside S49/S50 | `git status` | separate or land together | committed |

## 10. Resume commands

```sh
cargo test --workspace --release                  # 1056 passed / 0 failed
cargo fmt --all --check

# the NN benchmark, both machines
CYCLES=41 ./benches/perflock.sh bash benches/nn/nn_ab.sh
CYCLES=15 ./benches/perflock.sh bash benches/nn/nn_i9.sh 100.81.226.103

# price the runtime
MAPAL_PAR=14 ./target/release/examples/dispatch_cost 201
./target/release/examples/handoff_floor 13 201

# re-derive dispatch_ns after any handoff change: nn4_128 with vs without dispatches
#   (144.9 - 78.8) / 8 = 8.3 us  → profile.rs APPLE_M4_SME.dispatch

# the emission gate that must NOT move for unmeasured profiles
./benches/emit_sweep_ab.sh target/release/examples/emit /tmp/post.hashes --target=generic
```

## 11. Files changed

Code: `mapal-ir/src/algo.rs`, `mapal-rt/src/lib.rs`,
`backends/llvm/src/{func/drive.rs, profile.rs}`.
Tests: `mapal-ir/tests/algos.rs`, `backends/llvm/tests/{golden_ll.rs, sme_rung.rs}`,
4 golden snapshots.
New instruments: `mapal-rt/examples/{dispatch_cost.rs, handoff_floor.rs}`.
New benches: `benches/nn/` (`nn_gen.py`, `nn_numpy.py`, `nn_ab.sh`, `nn_i9.sh`,
`nn_report.py`, 15 `.mapal` sources).
New docs: `components/runtime/{ARCHITECTURE,IMPLEMENTATION,STATUS,suggestions}.md`,
`components/runtime/plans/plan-s50-scheduler-handoff.md`,
`components/backend-llvm/plans/plan-s49-linear-nn-two-nets.md`,
`performance/s49-*.md`, `s49b-*.md`, `s50-*.md`, `s50b-*.md`, `s50c-*.md`, `s50d-*.md`,
`s50e-*.md`. README results updated.
