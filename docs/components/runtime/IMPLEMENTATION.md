# Runtime — model → code

Functor from [`ARCHITECTURE.md`](ARCHITECTURE.md) to `crates/mapal-rt/src/lib.rs`.
One file; symbol names are exact.

| Model | `file:symbol` | State |
| --- | --- | --- |
| `RunGraph` (registration form) | `lib.rs:TaskBuild`, `Run::build: Mutex<Vec<TaskBuild>>` | built |
| `RunGraph` (executing form) | `lib.rs:TaskRt`, `Run::tasks: OnceLock<Box<[TaskRt]>>` | built |
| `seal` | `lib.rs:Run::seal`, `Run::launched` (`tasks.get().is_some()`) | built |
| `Frame` | `Run::frame: AtomicPtr<u8>` | built |
| `WorkItem` | `lib.rs:Work` | built |
| `slice` | `lib.rs:slice_ranges` | built |
| `assign` | `lib.rs:Run::schedule`, `Placement::Seed` | built |
| `deliver` | `lib.rs:Pool::enqueue` + `Pool::notify_lane` | built |
| `inline?` | `Run::schedule`'s `ranges.len() == 1` arm + `Run::run_inline` | built |
| `retire` | `lib.rs:Run::retire` (unlock, **then** publish `Done`) | built |
| `HostSpine` / `PoolLane(i)` | `WORKER_LANE`, `Pool::worker_loop`, `Run::help_until` | built |
| lane wake `Trm` | `lib.rs:LanePark { epoch, wake }`, one per lane | built |
| completion bump `Trm` | `Run::bump`, `Run::epoch: AtomicU64`, `Run::gate: Mutex<()>` | built |
| host spin | `Run::help_until`'s bounded spin, budget from `spin_iters` | built |
| one executor per lane | `Pool::new` → `with_workers(threads, threads - 1)` | built |
| trap flag | `Run::trap: AtomicU64`, `mapal_par_trap`, `check_trap` | built |
| watermark | `TaskRt::watermark`, `mapal_par_watermark` | built |
| arena | `lib.rs:ARENA`, `mapal_rt_alloc`, `reside` | built |

## Boundary (the emitted ABI)

`mapal_par_begin` · `_task` · `_dep` · `_pin` · `_launch` · `_wait` · `_run_pinned` ·
`_check` · `_trap` · `_watermark` · `_finish`. Declared once in
`crates/backends/llvm/src/module.rs` and emitted by `func/drive.rs`.

## Instruments

| Tool | Prices |
| --- | --- |
| `examples/dispatch_cost.rs` | one dispatch with no kernel under it; pinned vs pooled vs nested |
| `examples/handoff_floor.rs` | the wave round trip alone — spin vs spin-then-park vs parked |

## Levers, not design

`MAPAL_PAR` (executor count, **including the host** since S50), `MAPAL_SLICE`
(`slice_override`), `MAPAL_SPIN` (`spin_iters`). Each documents at its site that it
exists to falsify a sizing model, not to configure a build.
