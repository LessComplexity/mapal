# S49b — what a dispatch costs, and why small nets lose

Date: 2026-08-09 · instrument: `crates/mapal-rt/examples/dispatch_cost.rs`
Context: `s49-two-nets-and-the-cyclic-plan.md` measured a 4-layer net at d=128 running
**0.0822 ms on one thread and 0.3636 ms on fourteen**. Going parallel cost 4.4×. Nothing in
the tree priced the runtime on its own, so this does: the task bodies here do nothing at
all, and what the clock sees is the scheduler.

## 1. The instrument

`dispatch_cost` builds a `mapal_par_begin → task → launch → finish` cycle whose task
function is `black_box((lo, hi))` and returns. Median of 151–201 reps, one warm call
discarded. Three shapes: a **pinned** task (the host spine runs it inline — nothing is
enqueued, no worker woken, no queue scanned), a **pooled** task, and a **nested** run (a
pooled task whose body opens and finishes an inner run — the shape the emitted tile site
actually has).

## 2. The measurement

Apple M4 Pro (10P + 4E), µs, median.

| MAPAL_PAR | pinned (no wake) | pooled, 1 slice | pooled, 32 slices | nested run |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.500 | 0.625 | 10.375 | 0.541 |
| 2 | 0.208 | 0.416 | 2.833 | 1.375 |
| 3 | 0.291 | 0.667 | 3.583 | 1.541 |
| 4 | 0.250 | 0.958 | 3.958 | 2.833 |
| 6 | 0.250 | 2.084 | 10.083 | 7.750 |
| 8 | 0.209 | 4.291 | 17.000 | 14.666 |
| 10 | 0.250 | 9.792 | 25.083 | 28.667 |
| 12 | 0.208 | 7.291 | 30.375 | 37.333 |
| 14 | 0.208 | 5.125 | 30.791 | 41.917 |

Three facts fall straight out.

**(a) Building a run is free. Handing it to a thread is not.** The pinned row is **flat at
0.21–0.25 µs from 2 lanes to 14** — that is `mapal_par_begin` + registration + `launch` +
`finish` + running the body, all of it. The pooled row over the same construction goes
**0.42 µs at 2 lanes to 9.8 µs at 10**. The entire difference is wake + steal. So "the run
object allocates" is *not* the problem; the handoff is.

**(b) The cost grows with THREAD COUNT, not with work.** Every row above is the same zero
work. From 2 to 10 lanes the pooled empty dispatch grows 23× for 5× the threads — ~N^1.9,
clearly super-linear, where linear would have predicted 2.1 µs. The nested row is the clean
monotone series: 1.375 → 41.917 µs from 2 to 14 lanes, **30× for 7× the threads**.
(Above 10 lanes the single-slice row is non-monotone — 9.79 at 10, 7.29 at 12, 5.13 at 14.
This part is 10P+4E and the medians there are noisy; the shape is quoted over 2–10, where
it is clean.)

**(c) The fixed cost dwarfs the marginal cost.** At 14 lanes: 1 slice costs 5.1 µs, 256
slices cost 75.0 µs — a marginal **0.24 µs per slice** on a **5–14 µs** fixed floor. Cutting
more pieces is cheap; opening a dispatch is not.

### Why — read off the source

`Pool::enqueue` (`mapal-rt/src/lib.rs:519`) pushes each item, bumps one global
`wake_epoch`, and calls **`self.wake.notify_all()`** on one condvar. Every worker wakes, for
a dispatch that may carry one slice.

Each woken worker runs `take_any` (`:532`): pop its own `Mutex<VecDeque>`, and on a miss
**lock every other lane's mutex in turn** — up to N lock/unlock. The host, inside
`mapal_par_finish → help_until → take_for` (`:545`), does the same scan. So one dispatch
costs roughly `N` host locks + `N` workers × `N` locks = **O(N²) mutex operations,
independent of how much work there is** — 196 at 14 lanes to deliver one slice. Workers that
find nothing then contend on the single `wake_epoch` mutex to re-park.

Secondary, smaller: `Run.state` is one `Mutex<RunState>` taken twice per slice (`execute`
flips `Running`, `complete_slice` decrements), `complete_slice` **heap-allocates** a
`dependents.clone()` on every task completion, and each `Work` carries an `Arc<Run>` — an
atomic refcount pair per slice.

**Probed and reported as partial:** replacing `notify_all` with at most one `notify_one` per
queued item moved the 14-lane empty dispatch 5.13 → 2.33 µs (2.2×) but made 4 lanes *worse*
(0.96 → 1.67) and bought the nested case only 1.2× (41.9 → 35.0). Reverted, not shipped —
the fan-out is part of the cost, not the whole of it, and the O(N²) scan is the rest.

## 3. What this costs the real programs

The emitted tile site opens **two nested runs per matmul** — one for the B pack, one for the
tile matmul — and both are **inside the timed region**:

| program | outer tasks | nested runs in the bracket | SME calls |
| --- | ---: | ---: | ---: |
| `nn2_*` | 13 | 4 | 2 |
| `nn4_*` | 19 | 8 | 4 |
| `nn2x2_*` | 22 | 8 | 4 |

At 14 lanes an inner run costs ≈ 41.9 − 5.1 = **36.8 µs**. Predicted dispatch against the
measured parallel time at d=128:

| program | predicted dispatch | measured par | compute (4·d³ or 8·d³ at ~1000 GF/s) |
| --- | ---: | ---: | ---: |
| `nn2_128` | 4 × 36.8 + outer ≈ 170 µs | **172 µs** | ~8 µs |
| `nn4_128` | 8 × 36.8 + outer ≈ 330 µs | **364 µs** | ~17 µs |
| `nn2x2_128` | two chains overlap ⇒ ≈ 170–230 µs | **230 µs** | ~17 µs |

Three cells predicted within 10% from a microbenchmark that never touches the kernel. **At
d=128 the parallel number is ~95% scheduler.**

Where it stops mattering (4-layer net, 8 nested runs ≈ 294 µs):

| d | mapal par | dispatch share |
| ---: | ---: | ---: |
| 128 | 0.364 ms | **81%** |
| 256 | 0.475 ms | **62%** |
| 512 | 0.872 ms | 34% |
| 1024 | 3.168 ms | 9.3% |
| 2048 | 20.645 ms | 1.4% |

That is the whole shape of the mapal-vs-numpy curve: numpy wins below d≈512 and loses above
d≈1024, and the crossover is where dispatch stops dominating. numpy's entire 4-layer d=128
forward pass is 23.5 µs — less than one of our nested dispatches.

## 4. On "the measurement shouldn't contain thread dispatch"

Split into three, because only two of them are movable.

**Thread spawn — 121–126 µs, once per process — is already outside the bracket, but by
accident.** `global_pool()` is a `OnceLock` created at the first `mapal_par_begin`, and the
emitter puts the outer `mapal_par_begin(i32 13)` at the top of `mapal_main` while the first
clock read is task 0 *inside* that run, with dependencies on the generation tasks. So the
pool exists and its workers are warm before `t0`. Nothing enforces that: a program whose
first parallel work is inside the bracket would pay the spawn in its measurement.
`mapal_perf_begin` already pre-warms deliberately (`drop(global_pool())`, `:368`); the
`time` builtin path does not. **Cheap fix: warm the pool in the entry prologue.**

**Run construction — ~0.2 µs — is not worth moving.** The pinned row prices it, and it is
1/25 of one wake.

**The wake/steal handoff cannot be moved out, because it *is* running the program.** It has
to become cheap rather than relocated. What *can* be lifted out of the per-invocation path
is the **shape** of a run — task count, dep edges, ranks, slice sizes are all compile-time
constants that are re-registered on every call, and the nested tile runs are re-created per
layer per invocation.

## 5. So: is a "smart scheduler" the answer?

Partly, and not where it looks. The measurement says the **assignment policy is not what is
costing**: pinned execution with identical policy inputs costs 0.21 µs. A cleverer static
assignment layered on the same handoff would still pay 36.8 µs per nested dispatch.

Also worth saying plainly: the compiler already schedules. `rank` is a critical-path weight
computed in `path_plan`, `Placement::Seed` lays a dispatch out rank-sorted across the
`width` lanes the compiler asked for, `slice_elems`/`oversub` are compile-time sizes, and
`complete_slice` re-places an unlocked dependent as `Placement::Local(lane)` — chain
affinity, already there. The dataflow is being used. What is not being used is the fact that
**the run graph is a constant**, and what is expensive is that **delivering one slice to one
thread touches N mutexes and wakes N threads**.

Ordered by measured payoff — see `components/runtime/plans/plan-s50-scheduler-handoff.md`:

| # | change | measured target |
| --- | --- | --- |
| P0 | tile pack + tile matmul become **tasks in the outer run**, not two nested runs per site | ~147 µs of `nn2_128`'s 172 µs |
| P1 | one dispatch must not wake N workers and must not lock N queues | 5.1 µs → target ~0.5 µs at 14 lanes |
| P2 | run graph built once from a static descriptor, not re-registered per call | ~0.2 µs/run + the per-completion `Vec` clone |
| P3 | pool warmed in the entry prologue | removes a 126 µs cliff from any program that dispatches first inside its bracket |
| P4 | assignment policy | not yet — nothing says it is wrong |

## 6. Compile-time scheduling — what the floor actually is

Sapir: *"comptime scheduling, not runtime, such that the dispatch overhead is almost 0."*
A static schedule deletes queues, stealing, run construction and dep bookkeeping. What it
cannot delete is the moment N threads are told "go" and the host learns they finished. That
round trip is the floor, and `crates/mapal-rt/examples/handoff_floor.rs` measures it with
everything else removed: no queue, no run, no `Arc`, empty bodies, lanes pre-assigned.

µs per wave, M4 Pro (10P+4E), median of 151–201:

| workers | spin | spin-then-park | condvar (parked) |
| ---: | ---: | ---: | ---: |
| 2 | **0.125** | 0.125 | 11.083 |
| 4 | **0.291** | 0.250 | 13.458 |
| 8 | **0.542** | 0.542 | 22.625 |
| 9 | 0.958 | **0.792** | 28.333 |
| 10 | 0.958 | 0.958 | 29.459 |
| 11 | 0.958 | 0.958 | 35.125 |
| 12 | 0.958 | 0.958 | 35.375 |
| 13 | **1.083** | 1.083 | 41.209 |
| 14 | **1640.917** | 195.375 | 40.250 |

**The answer is yes, and the enabler is not the schedule — it is not sleeping.**

- **Static schedule on parked threads: 11–41 µs per wave.** That is *worse* than what mapal
  does today (5.1–9.8 µs for a pooled empty dispatch), because today's host is help-first
  and usually runs the work itself instead of waiting for a wake. **Compile-time scheduling
  on a parked pool buys nothing.** The cost was never the bookkeeping; it is the kernel
  round trip to wake a sleeping thread.
- **Static schedule on spinning threads: 0.96–1.08 µs per wave at 9–13 lanes** — 30–40× under
  the parked arm and ~5× under today's pooled dispatch. That is the "almost 0" being asked
  for, and it is a measurement, not a projection.
- **Hard constraint: workers ≤ cores − 1.** The host is a runnable thread too. At 14 workers
  on 14 cores the spin arm goes 1.083 → **1640.9 µs**, a 1500× cliff, and the hybrid still
  pays 195 µs. A spinning static schedule is only viable when the runtime owns the machine,
  and the fallback has to be automatic.

### 6b. The pool is already oversubscribed — and it is costing published numbers

`Pool::new` spawns `threads` workers and the host stays an additional help-first executor, so
`MAPAL_PAR=14` on a 14-core part is **15 runnable threads**. Every number in
`s49-two-nets-and-the-cyclic-plan.md` used 14. Self-timed forward pass, median of 11:

| shape | PAR=8 | PAR=10 | PAR=11 | PAR=12 | PAR=13 | PAR=14 | best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nn4_128` | **0.2483** | 0.2896 | 0.3080 | 0.2910 | 0.3289 | 0.3675 | 8 |
| `nn2_512` | **0.3668** | 0.3971 | 0.3864 | 0.4276 | 0.3968 | 0.4231 | 8 |
| `nn4_1024` | 3.2229 | 3.2137 | 3.1673 | **3.1355** | 3.1649 | 3.2095 | 12 |
| `nn4_2048` | 22.1220 | 21.5789 | 21.0712 | 21.0082 | 20.7237 | **20.6711** | 14 |

**`nn4_128` is 1.48× faster on 8 lanes than on 14, with no code change at all** — the
dispatch cost of the extra six lanes exceeds the compute they contribute. The optimum walks
8 → 12 → 14 as the work grows. So **lane count is a per-dispatch decision the compiler can
make** (it knows the element count and the quantum), not a process constant. `TaskDef.width`
already carries a per-dispatch lane count from the compiler, but `enqueue` still
`notify_all`s the whole pool and `take_any` still scans every lane, so today `width` narrows
the placement without narrowing the cost.

### 6c. The cost of going fully static, and why it cannot be fully static

S33 measured, on the i9 with 8 threads held constant and only core uniformity varying:
8 E-cores → numpy 5% ahead; 8 P-cores → numpy 41% ahead; **4P+4E → mapal 1.65× ahead**, and
OpenBLAS gained *nothing* from swapping in four 35%-faster cores because it partitions
statically and waits for the slowest thread. Mapal's whole advantage on hybrid silicon is
that work stealing absorbs that.

**A purely static compile-time schedule reintroduces exactly the failure mode mapal wins on.**
The compiler knows the dataflow; it does not know that lane 11 is an E-core running at 0.6×,
or that another process took a core mid-wave. So the target is not "static instead of
dynamic" — it is **static by default with an O(1) dynamic escape**: compile-time slice→lane
assignment (no queue, no wake, no steal scan) plus a single shared residual counter a
finished lane claims from with one `fetch_add`. Cost of the escape when nothing is imbalanced:
one uncontended atomic read per lane per wave.

### 6d. Projection, labelled as one

Arithmetic on two measured quantities, not a measurement: `nn4` at d=128 opens 8 dispatches
inside the bracket at ~36.8 µs each ≈ 294 µs, against ~17 µs of actual compute. At the
measured spin floor of ~1 µs per wave those 8 waves cost ~8 µs, which would put the cell near
**0.03 ms against numpy's 0.0235 ms** — from 15× behind to roughly parity. Nothing is claimed
until it is built and measured; this is here to size the prize, and it is the reason P0/P1
are ranked where they are.
