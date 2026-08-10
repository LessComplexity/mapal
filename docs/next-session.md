# Next Session (S51)

Written: 2026-08-10 · end of S49–S50 · by: Claude (orchestrator; category-architect)
Session log: `sessions/2026-08-10-s49-s50-the-scheduler-was-the-kernel.md` — **read it**
Records that govern S51: `performance/s50e-the-roofline-threshold.md` and
`components/runtime/{ARCHITECTURE,STATUS,suggestions}.md` (**new component**).
Previous handoff: S48 (below, still open and untouched).

## READ THIS FIRST

**Everything is UNCOMMITTED on `main` @ `680ce44`, which is in sync with origin.** Two
independent bodies of work sit in the tree together:

| batch | files | state |
| --- | --- | --- |
| **S48** (not this session's) | `backends/llvm/src/{lib,profile}.rs`, `differential.rs`, `golden_ll.rs`, `move_panel.rs`, `mapal-syntax/parser*`, `mapal-lower/tests/rejection.rs`, `ADR-0038`, the two `plan-s48-*` | untouched by S49/S50 |
| **S49–S50** | `mapal-ir/src/algo.rs`, `mapal-rt/src/lib.rs`, `backends/llvm/src/{func/drive.rs, profile.rs}`, `sme_rung.rs`, 4 goldens, `benches/nn/`, `components/runtime/`, 7 `performance/s49*-s50*` | this session |

The gate is green on the **combination** (1056 / 0). Nobody has run either batch alone.

**`MAPAL_PAR` changed meaning.** It is now TOTAL executors including the host —
`Pool::new` spawns `threads - 1` workers, because `help_until` already claimed lane
`threads - 1`. Measured neutral (0.99–1.02×), but **every historical number taken at a
given `MAPAL_PAR` was taken with one more runnable thread than the same flag gives
today.** Needs Sapir's sign-off or a revert.

**Spinning workers are REFUTED. Do not retry as specified.** `handoff_floor` predicts
30–40× and delivers 0.80–0.93× on real shapes, at every budget from 16 to 4096, because
it gives every thread work every wave and a real dispatch does not. Host-only spin is
what ships.

**A profile with no MEASURED dispatch economics carries `None` and dispatches
everything.** Verified: 0 of 174 cells move under `--target=generic`. Only
`apple-m4-sme` (and `native` when it detects that part) carries `Some`.

**`dispatch_ns` tracks the RUNTIME, not the machine.** 30 µs was right when
`handoff_floor` measured it and wrong four changes later, costing three cells 5–20%. The
shipped 8.3 µs is `(144.9 − 78.8) / 8` from `nn4_128` with and without its dispatches.
**Re-derive it after any change to the handoff.**

**The arch box `100.81.226.103` went DOWN mid-session** and never came back. Everything
on it is stale, `raptorlake` never got its three dispatch numbers, and the i9 half of the
final measurements is missing.

## S51 opens on — the single-thread kernel

### 1. P0 — it is now the only term that matters

Scheduling is spent. At d=128 the parallel time is within 20% of the one-thread time;
`s50e` §5 shows d ≥ 1024 at 3–8% spread and 1.23–1.47× ahead of NumPy. What is left:

| | mapal | NumPy | ratio |
| --- | ---: | ---: | ---: |
| M4, 4-layer d=128 | 212 GF/s | 691 | **0.31** |
| M4, d=1024 | 986 | 1364 | 0.72 |
| M4, d=2048 | 939 | 1546 | 0.61 |
| i9, d=2048 | 144 | 166 | 0.87 |

**Two different gaps.** The i9's is flat and small — a micro-kernel quality gap against
the same AVX2 units. The M4's is large and size-dependent, and **its size is already
accounted for**: S43's operand-residency instrument measured **+71% at one thread**,
assembly-verified, and 939 × 1.71 = 1606 against Accelerate's 1546. It is unclaimed
because `kc` is +6.1% at 1t and **−25.5% threaded** and `nc` is +18.7% at 1t and parity —
so it needs a **reuse-structure** change, not another blocking level. Unchanged since S48.

### 2. P1 — the queue

- **The dispatch threshold cannot decide for elementwise maps.** It only fires where a
  tile record supplies `rows`/`c`/`k`/`elem`. The obvious estimate (`3·n·width` bytes at
  the streaming rate) is measurably wrong two ways — `fir_65536` by 15× on the **byte
  count** (a 64-tap window reads 65 values per output, not 3) and `transpose_512` by
  5.7× on the **rate** (a permutation strides, ~15 B/ns not 127). Both under-estimate,
  so both would be inlined and both would lose their real 1.85×/1.38×. What is needed:
  a per-element read count and an achieved-rate class the emitter can derive from the
  graph. **`elem_plan` already records what `out[i]` IS** — start there.
- **`raptorlake` owes three numbers.** `dispatch_cost` + `handoff_floor` on the box,
  then `flops_per_ns`/`bytes_per_ns`/`dispatch_ns` in `profile.rs`. Recipe in `s50e` §2.
- **`path_plan_is_acyclic` is still load-bearing.** S49 removed the sibling-merge cycle
  class, not the general one, and a cyclic plan is still silently demoted to sequential
  emission — the failure mode that hid every chained matmul for nine sessions. Enforce
  the DAG contract or make the demotion loud (`runtime/suggestions.md` #4).
- **Sequential emission has no clock fence.** A program that falls back self-times
  garbage. Both new NN harnesses hard-error on an emission with no `mapal_par_task`, but
  the emitter does not.
- The six `runtime/ARCHITECTURE.md` §4 composition rules have no automated pin; rule 1
  was violated for all of P2 and surfaced only because P1 made the host fast enough to
  see it.
- Carried from S48, all still open: `examples/vector.mapal` does not parse (3 of 177
  sweep cells) · delete or justify `kc_nest` · executing SME value check in `cargo test`.

### 3. Measurement rules — S49–S50 added three

> **25. A microbenchmark that gives every thread work every wave does not model a
> dispatch.** Price a runtime change on real shapes before believing the instrument.

> **26. A constant derived from the runtime must be re-derived when the runtime
> changes.**

> **27. Label every table with the run it came from.** Absolutes and A/B ratios are not
> interchangeable across runs; a ratio is valid within a run even when the absolute is
> not. Three quotes of one `nn4_128` cell from three runs read as a regression.

Rules 1–24 stand. The one that earned most here is **22 — a sweep needs a control arm
that should NOT move**: the `nn4` chain is what proved the `nn2x2` gain was the DAG and
not the machine, and NumPy's own control is what proved its 1.8× "overlap" was an
OpenBLAS threading artifact.

## FIRST commands

```sh
cargo test --workspace --release                  # 1056 passed / 0 failed
cargo fmt --all --check
git status --short                                # two uncommitted batches, see above
ssh -o ConnectTimeout=8 100.81.226.103 nproc      # box was DOWN at close
CYCLES=41 ./benches/perflock.sh bash benches/nn/nn_ab.sh
```

## Live state at close

| Type | Handle | State | Inspect | Cleanup |
| --- | --- | --- | --- | --- |
| branch | `main` @ `680ce44` | in sync with origin; **everything uncommitted** | `git status -sb` | Sapir's call |
| machine | arch box `100.81.226.103` | **DOWN** since mid-session | `ssh … nproc` | — |
| artifact | `~/mapal-s49`, `-s50`, `-s50p1`, `-s50p2` on the box | scratch, size unknown | `ssh … du -sh ~/mapal-s5*` | delete when up |
| worktree ×12 | `.claude/worktrees/agent-*` + 3 pre-rename | prunable, unchanged since S47 | `git worktree list` | `git worktree prune` |
| results | `benches/results-s49/ … -s50final/` | untracked; **`-s50final` is what the README quotes** | `ls benches/results-s50final` | keep `-s49`, `-s50final` |
| file | `oainotes.md` | untracked, deliberately | — | Sapir's call |

**Nothing is running.** No background job, no server, no port. Measurement mutex free.

## Open items

| Priority | Item | Reference | Next action | Done when |
| --- | --- | --- | --- | --- |
| **P0** | single-thread kernel 0.28–0.35× of NumPy below d=512 | §1 | reuse-structure change, not another blocking level | 1t GF/s moves off ~950 |
| **P0** | `MAPAL_PAR` semantics changed | READ THIS FIRST | Sapir sign-off or revert | decided |
| **P0** | two uncommitted batches in one tree | READ THIS FIRST | separate or land together | committed |
| P1 | threshold cannot decide for elementwise maps | §2 | per-element read count + rate class via `elem_plan` | `transpose`/`fir` within 2× |
| P1 | `raptorlake` owes 3 dispatch numbers | §2 | run the two instruments on the box | profile carries `Some` |
| P1 | `path_plan_is_acyclic` load-bearing | §2 | enforce or make loud | fallback deletable |
| P1 | sequential emission has no clock fence | §2 | fence, or refuse to self-time | honest bracket |
| P2 | ARCHITECTURE §4 rules unpinned | `runtime/suggestions.md` #3 | one test per rule | 6 pins |
| P2 | `Work` holds an `Arc<Run>`; `take_any` miss is O(N) | `runtime/suggestions.md` #1–2 | — | — |
| P2 | box scratch · 12 worktrees · `vector.mapal` · `kc_nest` · SME value check | S48 | — | — |

## Standing direction (Sapir — unchanged)

- Compute-only legs; NumPy in every verdict table; scale everything up.
- Parallel-first by construction — **threaded is the configuration that ships.**
- Backend-genericity contract (ADR-0032): **mapal-ir never learns machine facts.**
  Scheduling and comptime structuring belong in the backend, per backend, so the IR's
  geometry analysis stays generic. Held through all of S49/S50 — checkable by
  `git diff -- crates/mapal-ir/`.
- **Keep stuff generic, not hardcoded** — and when a constant is swept rather than
  derived, say so.
- Query, not rewrite: record that something *could* be skipped; never delete it.
- Compile time decides the SIZES, runtime decides the ASSIGNMENT.
- Nothing goes in the README that a default build does not deliver.
- Proof over suggestion — a change arrives with the measurement of what it did.
- **The README is results, not method.** Tables and a pointer; explanations live in
  `docs/`.
- Speak simply, base claims on empirical results. **Give absolute ms, not only ratios.**
