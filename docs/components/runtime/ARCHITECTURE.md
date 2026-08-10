# Runtime — `mapal-rt`

The CPU backend's execution site. `mapal-ir` deduces a task DAG (`path_plan`), the
LLVM backend places and sizes it, and this component **runs** it: it owns the threads,
the queues, and the moment work crosses from one thread to another.

Created S50 (plan-s50). Before that the runtime was the only crate with no component
folder — the 2026-08-01 review recorded that the living architecture map modelled
neither its `Loc`s nor its `Trm`s.

## 1. Why model it categorically

Because the expensive thing here is not a computation, it is a **transmission**, and
without `Loc`/`Trm` in the model there is nothing to point at when it costs 95% of the
wall. FRAMEWORK §7.6: at this level Law 1 ("no transformation reads data not present
at its location") is a literal hardware-bug detector, and Law 4 ("a cross-location
dependency is mediated by a transmission") is the thing every scheduling bug violates.

## 2. `Dat` — the objects

| Object | Semantics |
| --- | --- |
| `RunGraph` | tasks, dep edges, ranks, slice quanta. **Compile-time constant** — the emitter replays the same registration every invocation |
| `Frame` | the per-invocation arena every task reads and writes through |
| `WorkItem` | one `(task, lo, hi)` range, the unit that crosses a lane boundary |
| `TaskRt` | a task while the run executes: `def`, `dependents`, `pinned` (frozen) + `deps_left`, `state`, `slices_left`, `watermark` (atomics) |

| Morphism | Signature | Partiality | Semantics |
| --- | --- | --- | --- |
| `shape` | `Site → RunGraph` | Total, **Deduced** at compile time | what `path_plan` + the emitter decided |
| `seal` | `TaskBuild* → TaskRt*` | Total | registration → executing form; **being sealed IS being launched** |
| `slice` | `Task × ℕ → Range*` | Total | compile-time quantum × runtime lane count |
| `assign` | `Range → Loc` | Total | rank-sorted round robin over the compiler's `width` |
| `deliver` | `Range × Loc → Trm` | Total | per-lane queue push + wake of exactly that lane |
| `retire` | `Task → Task*` | Total | unlock dependents, then publish `Done` — **in that order** |
| `inline?` | `Task → Loc` | Partial | a one-slice task runs at the caller's `Loc` instead of crossing |

## 3. `Loc` and `Trm`

`Loc` — `HostSpine` (one), `PoolLane(i)` for `i < threads`. **One executor per lane,
counting the host**: `Pool::new` spawns `threads - 1` workers because `help_until`
already gives the host lane `threads - 1`.

| `Trm` | `c_from` | `c_to` | `carries` |
| --- | --- | --- | --- |
| enqueue | scheduler's lane | target lane | `WorkItem` |
| lane wake | any | target lane | the fact that a `WorkItem` exists |
| completion bump | finishing lane | every waiter | the run's `epoch` |
| frame slot | producer lane | consumer lane | the produced `Dat` |

## 4. Composition rules

1. **`retire` unlocks before it publishes.** `Done` is what a waiter tests, so storing
   it before decrementing dependents lets a waiter run ahead of the unlock. Violating
   this trips `mapal_par_run_pinned`'s `deps_left == 0` assertion — it did, and only a
   spinning host made it observable.
2. **Exactly one thread sees a `deps_left` fall to zero**, so a task is scheduled once
   with no lock and no re-check.
3. **`mapal_par_launch` only seeds.** The emitted host runs its own spine work between
   launch and finish, and the clock fence is dependency edges evaluated from that
   spine; a launch that executed tasks would run bracketed work before the host
   reached the pinned clock read.
4. **A pinned task never reaches a worker.** Both retire paths filter it; the host runs
   it via `mapal_par_run_pinned` at its topo position.
5. **`deliver` wakes exactly the lanes it filled.** The set is not a heuristic — the
   placement loop just chose it.
6. **A one-slice task is not delivered at all.** It has no parallelism to give away, so
   crossing a boundary for it is pure cost.

## 5. Bridges

| Boundary morphism | Other side | Stored? |
| --- | --- | --- |
| `mapal_par_task(kind, n, rank, slice_elems, oversub, width)` | `backend-llvm` `func/drive.rs` | the compiler's sizing, passed not re-derived |
| `mapal_par_dep` | same | the `path_plan` edge, reified |
| `mapal_par_pin` / `run_pinned` | same | host-spine placement of clock reads and trap-capable calls |
| `TargetProfile::dispatch` | `backend-llvm` `profile.rs` | **the runtime never reads it** — the emitter decides `kind` and the runtime only sees the result |

## 6. What is deliberately NOT here

The **decision** to dispatch. `MAPAL_PAR` and `MAPAL_SLICE` are experiment levers, not
design (their doc comments say so). Sizing is compile-time (plan-s32 §2.5); assignment
is runtime; the dispatch/inline verdict is compile-time and arrives as `kind`. This
component owns only the mechanism.
