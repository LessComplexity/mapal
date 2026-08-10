# Plan S50 — the handoff, not the policy

> **Status 2026-08-09: P0 and P0b are BUILT, gated and measured.** Results:
> `docs/performance/s50-p0-the-nested-runs.md`. Headline: `nn4_128` 2.20x, `nn2_128` 2.01x,
> `nn2x2_128` 1.83x on the M4; the published ladder gained too (`conv2d_1024` 1.18x,
> `matmul1024` 1.08x); no cell regressed on that machine; gate 1054/0 with the differential
> oracle green. **P0b under-delivered against its microbenchmark** (11x on an empty dispatch,
> ~0-3% on real programs, because real dispatches fill every lane) and **a third change not in
> this plan carried most of the win**: dep-unlocked tasks are now `Placement::Seed`, not
> `Placement::Local` — see §P1b-actual below. **Open: `nn2x2` at d<=256 on the 32-lane i9 is
> 0.855-0.907x of S49**, and P2 is now the named suspect.
>
> **Status 2026-08-09 (later): P2 is BUILT and it closed that item.** Results:
> `docs/performance/s50b-p2-the-shared-lock.md`. The two regressed i9 cells were the two P2
> helped most (1.733x and 1.433x) — the diagnosis confirmed by its own repair. Cumulative
> against S49: i9 `nn2_128` 1.744x, `nn4_128` 1.716x; M4 `nn4_128` 2.285x, `nn2x2_128`
> 1.736x, `nn4_512` 1.320x. **`nn4` at d=512 reached parity with Accelerate** (0.6602 vs
> 0.6613 ms) from 0.75x at S49. Gate 1054/0. **P1 remains unbuilt** and is now the floor:
> every gain so far came from fewer waves and less work per wave, never from a cheaper wave.
>
> **Status 2026-08-09 (final): P1 measured. Spinning WORKERS is REFUTED** (0.80-0.93x on real
> shapes; `handoff_floor` over-predicted because it gave every thread work on every wave, and
> a real dispatch leaves most lanes idle). **Host-only spin ships** — `nn4_128` 1.18x, budget
> 8192 iterations derived from the 11-41 us park-and-wake it replaces, zero when
> oversubscribed. `Pool::new` now spawns `threads - 1` workers so **`MAPAL_PAR` means total
> executors including the host** (measured neutral in isolation). P1 also exposed a **race
> introduced by P2**: `retire` published `Done` before unlocking its dependents, so a spinning
> host could beat the unlock and trip `mapal_par_run_pinned`'s assertion — fixed by ordering,
> hammered 300x. Results: `docs/performance/s50c-p1-spinning-workers-refuted.md`.
> Cumulative vs S49: M4 `nn4_128` **2.51x**, `nn2_128` 2.11x; i9 `nn2_128` 2.00x, `nn2_256`
> 1.73x. Gate 1054/0.
>
> **Status 2026-08-10: P3 (the dispatch threshold) is BUILT in two halves.** The
> constant-free half — a task cut into ONE slice runs on the calling thread — is
> `s50d-dont-dispatch-what-cannot-be-shared.md`. The roofline half, with three measured
> `TargetProfile` fields, is `s50e-the-roofline-threshold.md`. `dispatch_ns` had to be
> re-derived from a program the runtime actually ran (8.3 us) rather than `handoff_floor`'s
> parked wave (30 us), which cost three cells before it was corrected — **the constant tracks
> the runtime, not the machine**. **CUMULATIVE vs S49 on the M4: `nn4_128` 4.40x, `nn2_128`
> 4.23x, `nn2x2_128` 3.06x, `nn2x2_256` 2.07x**; `nn4` at d=512 is at parity with Accelerate
> and d>=1024 is 1.22-1.51x ahead. Gate 1056/0. `raptorlake` owes its three numbers (box
> unreachable). The one-thread deficit is now the dominant term everywhere.

Written: 2026-08-09 · component: **runtime** (`crates/mapal-rt`) · reads with
`docs/performance/s49b-the-dispatch-price.md`, which is the measurement this plan acts on.

> **Note on the component.** `docs/components/runtime/` did not exist before this plan.
> `mapal-rt` is a `Cmp` with two `Loc`s (host spine, pool lane) and real `Trm`s (frame slot,
> dep edge, work queue) and none of it is in the living architecture map — the 2026-08-01
> full-project review said so. This plan is the first doc in the folder; ARCHITECTURE.md /
> IMPLEMENTATION.md / STATUS.md / suggestions.md are **owed**.

## 1. What the measurement forces

Sapir's framing: *"instead of a thread pool we need a smart scheduler — we know the
dataflow, we know what is waiting, we can schedule it smartly."* The measurement agrees with
half of that and redirects the other half.

**Agrees:** at d=128 a 4-layer net spends **81%** of its parallel wall in the runtime, and a
2-layer net **~95%**. The concurrency model is the bottleneck, not the kernel.

**Redirects:** it is not the *assignment* that costs. A **pinned** task — same run object,
same registration, same launch, same finish, executed inline by the host — costs
**0.21 µs and is flat from 2 lanes to 14**. The same task handed to the pool costs
**0.42 µs at 2 lanes and 9.8 µs at 10**. Every microsecond of the difference is *wake and
steal*. A better static assignment on top of this handoff still pays it.

And the compiler already schedules: `path_plan` computes critical-path `rank`,
`Placement::Seed` lays a dispatch rank-sorted across the compiler's `width`,
`slice_elems`/`oversub` are compile-time sizes cut on the region's quantum, and
`complete_slice` re-places an unlocked dependent `Local(lane)` — chain affinity. The
dataflow is being used. **What is not used is that the run graph is a constant, and what is
expensive is that delivering one slice touches N mutexes and wakes N threads.**

## 2. Model

`Loc` — `HostSpine`, `PoolLane(i)` for `i < N`. `Dat` — `RunGraph` (tasks, dep edges, ranks,
slice quanta: **compile-time constant**), `Frame` (per invocation), `WorkItem`. `Trn` —
`slice : Task × N → Range*`, `assign : Range → Loc`, `execute`, `complete`. `Trm` — the
enqueue of a `WorkItem` from one `Loc` to another, and the condvar wake that tells the
target it exists.

| Morphism | Signature | Partiality | Semantics |
| --- | --- | --- | --- |
| `shape` | `Site → RunGraph` | Total, **Deduced at compile time** | task count, deps, ranks, quanta — today re-registered per invocation |
| `slice` | `Task × ℕ → Range*` | Total | compile-time quantum × runtime lane count |
| `assign` | `Range → Loc` | Total | today: round-robin over `width`, then stealing |
| `deliver` | `Range × Loc → Trm` | Total | today: per-lane `Mutex<VecDeque>` + `notify_all` |

**Composition rule violated today:** `deliver` should cost O(1) in `N`. It costs O(N²) —
`Pool::enqueue` wakes every worker (`notify_all` on one condvar behind one mutex) and every
woken worker's `take_any` miss path locks all `N` lane mutexes in turn. That is FRAMEWORK §5
"one source of truth for shared structure" failing on the *idle* set: there is no shared
record of which lanes are idle, so each worker rediscovers it by polling all of them.

## 3. Steps, in measured-payoff order

### P0 — delete the nested runs (biggest single number)

The emitted tile site opens **two `mapal_par_begin` per matmul** — B pack, then tile matmul —
**inside the timed region**, per layer, per invocation. An inner run costs **36.8 µs at 14
lanes**. `nn2_128` opens 4 of them; `nn4_128`/`nn2x2_128` open 8.

They should not be nested at all. Pack and matmul are two transformations with one dep edge —
exactly what the outer run's `mapal_par_dep` already expresses. Registering them as outer
tasks makes the nesting disappear rather than making it cheaper.

- **Predicted:** removes ~147 µs of `nn2_128`'s 172 µs.
- **First check before building:** whether the inner `n` is a compile-time constant at every
  tile site or depends on the enclosing slice's bounds. If it depends, the outer task's
  `slice_elems` has to carry it, and that is the real work of this step.
- **Watch:** S43 rejected `begin(2)` + `mapal_par_dep` for the parallel B pack because
  `complete_slice` schedules dep-unlocked tasks `Placement::Local(lane)`, which would move
  every matmul slice off the rank-sorted `Seed` placement onto one deque. **That objection
  applies here verbatim** and is the reason this is a plan and not a patch. Read
  `plan-s43-parallel-bpack.md` §7 first.

### P0b — lane count is a per-dispatch decision, not a process constant

Free, measured, no redesign: `nn4_128` runs **0.2483 ms on 8 lanes and 0.3675 ms on 14**
(1.48×), `nn2_512` 0.3668 vs 0.4231, while `nn4_2048` still wants all 14. The optimum walks
8 → 12 → 14 with the work. `Pool::new` spawns `threads` workers *and* the host stays a
help-first executor, so `MAPAL_PAR=14` on a 14-core part is 15 runnable threads.

`TaskDef.width` already carries a compile-time lane count, and `Placement::Seed` honours it —
but `enqueue` still `notify_all`s the whole pool and `take_any` still scans every lane, so
`width` narrows the *placement* without narrowing the *cost*. Making `width` narrow the cost
is most of this item; deriving it from the element count and quantum is the rest.

### P1 — `deliver` must be O(1) in N — and the workers must not sleep

**Measured floor** (`examples/handoff_floor.rs`, everything but the round trip removed):

| workers | spin | spin-then-park | condvar (parked) |
| ---: | ---: | ---: | ---: |
| 8 | 0.542 | 0.542 | 22.625 |
| 13 | **1.083** | 1.083 | 41.209 |
| 14 | 1640.917 | 195.375 | 40.250 |

Two conclusions that set this plan's direction:

1. **A compile-time schedule on a parked pool buys nothing.** 11–41 µs per wave is *worse*
   than today's 5–14 µs, because today's help-first host usually runs the work itself rather
   than waiting for a wake. The cost was never bookkeeping — it is the kernel round trip.
   Deleting queues without deleting sleep is not the fix.
2. **Spinning lanes reach ~1 µs per wave, and that is the "almost zero" target** — but only
   while `workers ≤ cores − 1`. At 14 workers on 14 cores the spin arm goes 1.083 →
   **1640.9 µs**. The fallback to parking must be automatic and must trigger on
   oversubscription, not on a flag.

### P1b — static assignment needs an O(1) dynamic escape

S33's controlled experiment (8 threads held constant, only core uniformity varying): 8
E-cores → numpy 5% ahead; 8 P-cores → numpy 41% ahead; **4P+4E → mapal 1.65× ahead**, because
OpenBLAS partitions statically and waits for its slowest thread. **That is the failure mode a
purely static compile-time schedule reintroduces.** The compiler knows the dataflow; it does
not know lane 11 is an E-core at 0.6×, or that another process took a core mid-wave.

Target shape: compile-time slice→lane assignment (no queue, no wake, no steal scan) **plus**
one shared residual counter that a finished lane claims from with a single `fetch_add`. Cost
when nothing is imbalanced: one uncontended atomic read per lane per wave. That keeps the
heterogeneity tolerance that is mapal's actual measured edge.

### P1c — the old delivery path

Two independent costs, both measured:

1. **Fan-out.** `notify_all` for a dispatch carrying one slice. Probed: at most one
   `notify_one` per queued item moved the 14-lane empty dispatch 5.13 → 2.33 µs, but made
   4 lanes *worse* (0.96 → 1.67) and bought the nested case only 1.2×. **Reverted, not
   shipped** — real but partial, and it carries a lost-wakeup risk if a notified worker was
   already running.
2. **The miss path.** `take_any`/`take_for` lock every lane's `Mutex<VecDeque>` on a miss.
   Replace with either a lock-free deque per lane (Chase–Lev) or one MPMC queue plus an
   atomic idle-lane bitmap, so a failed steal is an atomic read rather than N lock/unlocks,
   and `enqueue` wakes exactly the lanes it filled.

- **Target:** 14-lane empty pooled dispatch 5.1 µs → ~0.5 µs (the pinned floor is 0.21).
- **Gate:** `dispatch_cost` is the instrument and already exists; the pinned row is the
  control that must NOT move.

### P2 — the run graph is a constant; stop rebuilding it

`Run::new` allocates a `Vec<Task>` + `Arc` + `Box` per invocation; every
`mapal_par_task`/`mapal_par_dep` retakes the state mutex; `complete_slice` heap-allocates a
`dependents.clone()` on every task completion; every `Work` carries an `Arc<Run>` (atomic
refcount pair per slice). All of it re-derives what `path_plan` already knows.

Emit the graph as a static descriptor per call site and give the runtime a reusable `Run`
whose per-invocation state is a reset, not an allocation. Small in isolation (~0.2 µs/run)
but it is the precondition for P1 being visible and it removes an allocation from the inner
completion path.

### P3 — warm the pool in the entry prologue

Spawn is **121–126 µs**, once. It is outside the bracket for today's benchmarks only because
the outer `mapal_par_begin` happens to precede the first clock read. Nothing enforces it.
`mapal_perf_begin` already pre-warms on purpose (`drop(global_pool())`); the `time` builtin
path does not. One call in the emitted prologue closes it.

### P1b-actual — what shipped instead, and why it was not predicted

`Placement::Local(lane)` — put a dep-unlocked task on the lane that finished its producer —
was affinity reasoning that only held while the only dep-unlocked tasks were scalar glue. P0
made a many-slice matmul dep-unlocked, and `Local` handed it one deque plus an index-order
sort in place of the rank-sorted spread the same task gets at launch. Measured against
seeding (M4, par, median of 41): `nn2_128` 0.1290 -> 0.0839, `nn4_128` 0.2447 -> 0.1550,
`nn2x2_128` 0.1813 -> 0.1215, `nn2x2_512` 0.6097 -> 0.5821. `Placement` now has one variant.

The lesson for the rest of this plan: **the empty-dispatch microbenchmark over-predicted P0b
by an order of magnitude** (11x on the instrument, ~0-3% on real programs) because a real
dispatch usually fills every lane and there is no fan-out to save. Price P1 on real shapes
before believing the instrument.

### P4 — assignment policy: not yet

Nothing in the measurement says the policy is wrong. Revisit after P0–P2, when the handoff is
cheap enough for a policy difference to be visible above it.

## 4. What would refute this plan

- P0 lands and `nn2_128` does **not** move ~147 µs ⇒ the nested-run attribution is wrong and
  the cost is somewhere the microbenchmark does not model.
- P1 lands, `dispatch_cost` improves, and the **real** shapes do not ⇒ the empty dispatch is
  not representative of a dispatch that carries a working set.
- Either lands and the published ladder regresses ⇒ the fixed cost was buying something
  (locality, ordering) that the cheaper handoff loses. The ladder A/B in
  `s49-two-nets-and-the-cyclic-plan.md` §0c is the harness for that check.
