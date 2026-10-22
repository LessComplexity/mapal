# P2 built: the completion path stops taking a shared lock

Historical provenance: S50b.

Date: 2026-08-09 · plan: `components/runtime/plans/plan-s50-scheduler-handoff.md` §P2
Predecessor: `s50-p0-the-nested-runs.md`, which shipped P0/P0b and left one open item —
`nn2x2` at d ≤ 256 on the 32-lane i9 running **0.855× / 0.907×** of S49 while every
single-chain shape on that box gained. The named suspect was `Run.state`. This is the fix,
and the fix confirms the diagnosis.

## 1. What was wrong

`Run` kept one `Mutex<RunState>` holding every task's `def`, `dependents`, `deps_left`,
`state`, `slices_left` and the run's `remaining`/`epoch`. Per slice that mutex was taken
**twice** — once in `execute` to flip `Waiting → Running` and read `def`, once in
`complete_slice` to decrement — and every task completion additionally
**heap-allocated** a `dependents.clone()`.

Under S49 a packed tile site opened its own nested run, so four sites meant four separate
mutexes. P0 deleted the nesting, which was worth 1.7–2.2× — and funnelled every site's
completions through **one** lock. That is why the regression appeared exactly where lanes ×
concurrently-live-tasks was highest: 32 lanes, two independent chains, eight live tasks.

## 2. What was built

**The run graph is a constant, so it is sealed once and then read with no lock.**

| field | before | after |
| --- | --- | --- |
| `def`, `dependents`, `pinned` | inside the mutex | `TaskRt`, immutable after `seal`, read lock-free |
| `deps_left` | `u32` under the mutex | `AtomicU32` |
| `state` | `TaskState` under the mutex | `AtomicU8` |
| `slices_left` | `usize` under the mutex | `AtomicUsize` |
| `remaining`, `epoch` | inside the mutex | `AtomicUsize` / `AtomicU64` |
| registration | same mutex | `Mutex<Vec<TaskBuild>>`, pre-launch only |
| launched flag | `bool` in the mutex | `OnceLock<Box<[TaskRt]>>` — being sealed **is** being launched, so there is no second fact to keep in agreement |

Hot paths now:

- **`execute`: zero locks.** `pinned` and `def` are immutable; `schedule` already published
  `Running` before queuing the slice.
- **`complete_slice`: one atomic for every slice but the last.**
  `slices_left.fetch_sub(1) != 1` returns immediately. Only the thread taking the final slice
  retires the task, and `retire` walks `dependents` (a `Box<[usize]>`, no clone) doing one
  `deps_left.fetch_sub` each. **Exactly one thread sees a count fall to zero**, so a task is
  scheduled exactly once with no lock and no defensive re-check.
- **`schedule`: the rank sort reads `def` lock-free.**
- The remaining `Mutex<()>` is the condvar's companion and nothing else: every value a waiter
  tests is an atomic, and `bump` raises the epoch *before* taking the gate, so a wake cannot
  be lost between a waiter's test and its wait.

**One latent bug closed in passing.** The zero-length-dispatch path pushed unlocked
dependents *without* filtering pinned ones, so a pinned dependent of an empty split could be
handed to a worker, where `execute` asserts. `complete_slice` had the filter; the other path
did not. One rule now, both paths.

## 3. Measured — interleaved A/B against the P0+P0b tree

Both arms are complete toolchains, run alternating, values byte-identical. M4 Pro, 41 cycles.

| shape | par | | shape | par |
| --- | ---: | --- | --- | ---: |
| `nn2x2_512` | **1.166×** | | `nn4_1024` | 1.033× |
| `nn2_512` | **1.137×** | | `nn2x2_1024` | 1.033× |
| `nn4_512` | **1.093×** | | `matmul1024` | 1.024× |
| `conv2d_1024` | 1.043× | | `nn4_128` | 1.041× |
| `nn4_2048` | 0.999 | | `matmul4096` | 0.995 |

The one-thread column is flat (0.985–1.028) — `MAPAL_PAR=1` spawns no pool, so it is the
control that must not move, and does not.

## 4. The open item from S50 is closed

**i9-14900F, 32 threads**, mapal fma par, median ms:

| shape | d | S49 | P0+P0b | **+P2** | P2 × | total × | numpy | mapal × np |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nn2 | 128 | 0.1902 | 0.1489 | **0.1091** | 1.365 | **1.744** | 0.0747 | 0.68 |
| nn4 | 128 | 0.3389 | 0.2560 | **0.1975** | 1.297 | **1.716** | 0.1382 | 0.70 |
| **nn2x2** | **128** | 0.1850 | *0.2164* | **0.1249** | **1.733** | **1.481** | 0.1378 | **1.10** |
| nn2 | 256 | 0.2489 | 0.2219 | **0.1769** | 1.255 | 1.408 | 0.4588 | 2.59 |
| nn4 | 256 | 0.4771 | 0.4097 | **0.3120** | 1.313 | 1.529 | 0.9050 | 2.90 |
| **nn2x2** | **256** | 0.3017 | *0.3326* | **0.2321** | **1.433** | **1.299** | 0.9071 | 3.91 |
| nn4 | 512 | 0.9800 | 0.9360 | 0.9247 | 1.012 | 1.060 | 6.7181 | 7.27 |
| nn2x2 | 512 | 0.8248 | 0.8245 | 0.8056 | 1.024 | 1.024 | 6.7179 | 8.34 |
| nn4 | 1024 | 6.6681 | 6.6461 | 6.6958 | 0.993 | 0.996 | 8.2279 | 1.23 |
| nn4 | 2048 | 45.8525 | 46.4807 | 46.6243 | 0.997 | 0.983 | 57.6086 | 1.24 |

The two regressed cells were the two cells P2 helped most — **1.733× and 1.433×** — which is
the diagnosis confirmed by its own repair rather than by argument. Both are now ahead of S49.

**M4 Pro, 14 threads**, same columns:

| shape | d | S49 | P0+P0b | **+P2** | total × | numpy | mapal × np |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nn2 | 128 | 0.1720 | 0.0961 | **0.0990** | **1.737** | 0.0138 | 0.14 |
| nn4 | 128 | 0.3636 | 0.1815 | **0.1592** | **2.285** | 0.0246 | 0.15 |
| nn2x2 | 128 | 0.2301 | 0.1344 | **0.1326** | **1.736** | 0.0234 | 0.18 |
| nn2 | 256 | 0.2422 | 0.1611 | **0.1545** | 1.567 | 0.0485 | 0.31 |
| nn4 | 256 | 0.4751 | 0.3199 | **0.3116** | 1.525 | 0.0920 | 0.30 |
| nn2x2 | 256 | 0.3161 | 0.2399 | **0.2062** | 1.533 | 0.0915 | 0.44 |
| nn2 | 512 | 0.4982 | 0.4489 | **0.4361** | 1.142 | 0.3348 | 0.77 |
| nn4 | 512 | 0.8718 | 0.7634 | **0.6602** | 1.320 | 0.6613 | **1.00** |
| nn2x2 | 512 | 0.5712 | 0.5774 | **0.4952** | 1.153 | 0.6673 | **1.35** |
| nn2 | 1024 | 1.5915 | 1.5262 | **1.4930** | 1.066 | 2.3883 | **1.60** |
| nn4 | 1024 | 3.1681 | 3.0874 | **2.9949** | 1.058 | 4.0114 | **1.34** |
| nn2x2 | 1024 | 2.8298 | 2.8659 | **2.7946** | 1.013 | 4.5266 | **1.62** |
| nn2 | 2048 | 10.3757 | 10.1554 | **10.1464** | 1.023 | 14.5232 | **1.43** |
| nn4 | 2048 | 20.6452 | 20.4990 | 20.5368 | 1.005 | 26.2766 | **1.28** |
| nn2x2 | 2048 | 19.9572 | 19.7195 | 20.0319 | 0.996 | 27.1399 | **1.35** |

**The 4-layer net at d=512 crossed over**: 0.6602 ms against Accelerate's 0.6613 — parity,
from 0.75× at S49. Two nets at d=512 is now 1.35× ahead where S49 was 1.15×.

## 5. Gate

`cargo test --workspace --release`: **1054 passed / 0 failed**, including the 1,280-run
differential byte-equality oracle and `mapal-rt`'s 21 tests — among them
`seeded_random_dags_complete_once`, which drives randomly generated DAGs through the new
lock-free completion path. `cargo fmt --all --check` clean. `mapal-ir` untouched: the whole
change is `crates/mapal-rt/src/lib.rs`.

## 6. Still open

- **d ≥ 1024 is unmoved on both machines**, as it must be — dispatch was already 1–9% of
  those walls. The one-thread deficit (0.60–0.87× of numpy) is untouched and remains the
  standing P0 from S48.
- **P1 is not built.** The parked-thread wake is still the floor: `handoff_floor` measures
  11–41 µs per wave parked against ~1 µs spinning, and every gain here came from doing fewer
  waves and less work per wave, not from making a wave cheaper.
- `nn2_512` on the i9 reads 0.917× for P2 alone and 0.974× against S49 — a single cell
  against the trend, not investigated.
