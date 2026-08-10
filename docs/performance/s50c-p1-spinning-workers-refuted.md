# S50c — P1: spinning workers REFUTED, the host spin ships, and a race it exposed

Date: 2026-08-09 · plan: `components/runtime/plans/plan-s50-scheduler-handoff.md` §P1
Predecessors: `s50-p0-the-nested-runs.md`, `s50b-p2-the-shared-lock.md`.
Raw: `benches/results-s50p1/` and `benches/results-s50p1/i9/`.

## 0. What P1 predicted, and what happened

`examples/handoff_floor.rs` measured a wave — the round trip where the host says "go" to N
threads and learns they finished — at **~1 µs spinning against 11–41 µs parked**, a 30–40×
prize. That prediction **did not transfer**, and the instrument is why it lied: it gave
*every* thread work on *every* wave. A real dispatch does not — `nn4` at d=128 cuts **four**
slices for **fourteen** lanes.

Workers spinning, M4 Pro, `MAPAL_PAR=14`, median of 31, against the same tree with the spin
removed:

| shape | workers spin |
| --- | ---: |
| `nn4_512` | **0.80×** |
| `nn2x2_512` | 0.87× |
| `matmul1024` | 0.88× |
| `nn4_1024` | 0.93× |
| `nn4_128` | 0.83× |

Two mechanisms, both structural: a spinning **idle** lane burns a core the working lanes
need, and every idle lane that sees the shared `pending` counter move races into `take_any`,
which locks every queue — a herd inside the spin loop, worse than the wake it replaced.
Swept 16 / 64 / 256 / 1024 / 4096: the best cell was `nn4_128` at 1.07× and everything else
sat at 0.99–1.01. **No budget recovers it. Refuted.**

## 1. What ships instead

**Host-only spin.** There is exactly one host; it publishes the dispatch and it is what
learns the wave finished, so it is *on* the wave's critical path and cannot herd. Same
conditions:

| shape | host spin 0 | 256 | 2048 | 8192 | 32768 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nn4_128` | 0.94 | 1.12 | **1.18** | 1.14 | 1.19 |
| `nn4_512` | 0.96 | 0.98 | 1.00 | 1.02 | 1.00 |
| `nn2x2_512` | 0.99 | 0.99 | 0.99 | 1.01 | 0.99 |
| `matmul1024` | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 |

**The budget is derived from the thing it replaces, not fitted.** Spinning is only worth it
up to what a park-and-wake would have cost — 11–41 µs measured — because past that you would
have been better off asleep. At roughly 1 ns per `spin_loop`, **8192 iterations ≈ 8 µs**, the
low end of that band. The sweep shows a plateau from 2048 up, not a peak, so the smallest
value on the plateau is the honest pick.

**And it is zero when the pool is oversubscribed.** `handoff_floor` at the boundary: 13
executors 1.083 µs, 14 executors **1640.917 µs** — a 1500× cliff, because spinners and host
fight for the same cores. That is a correctness bound on a performance knob.
`MAPAL_SPIN` overrides for sweeping, like `MAPAL_SLICE`.

**One executor per lane.** `Pool::new` now spawns `threads - 1` workers, because
`help_until` already gives the host lane `threads - 1` — so with `threads` workers that lane
always had two claimants and `MAPAL_PAR=14` on a 14-core part meant 15 runnable threads.
Measured in isolation (spin forced to 0): **0.99–1.02×**, neutral. **`MAPAL_PAR` now means
total executors including the host** — a deliberate semantic change to the flag, recorded
here because it changes what a number taken at a given `MAPAL_PAR` means.

## 2. The race P1 exposed — a P2 bug, not a P1 one

`retire` published `Done` **before** decrementing its dependents' `deps_left`. `Done` is what
a waiter observes, so a host that saw it could run ahead of the thread still unlocking, and
`mapal_par_run_pinned` hit its own `deps_left == 0` assertion with the count still at 1:

```
panicked at crates/mapal-rt/src/lib.rs: pinned task dependencies are incomplete
  left: 1   right: 0
```

It is a **race**, and it was survivable only because the host always parked and was released
by the `bump` that runs *after* `retire` returns. Giving the host a spin made it observe
`Done` immediately, and it fired on `conv2d_1024` within seconds of the first run.

Fixed by ordering: dependents unlock first, `Done` publishes last, so Release/Acquire on
`state` orders the whole unlock behind the flag a waiter tests. Verified by hammering the
shape that reproduced it — **`conv2d_1024` × 300, zero failures** — plus the full gate.

This is the second time in this plan that a change was cheap and its *measurement* was the
expensive part. It is also a reminder that P2's lock removal traded a mutex for an ordering
obligation, and that obligation was not fully discharged until here.

## 3. Interleaved A/B against the P2 tree — M4 Pro, 41 cycles

| shape | par | | shape | par |
| --- | ---: | --- | --- | ---: |
| `conv2d_1024` | **1.110×** | | `fir_1048576` | 1.029× |
| `nn4_128` | **1.077×** | | `matmul4096` | 1.012× |
| `nn2_128` | 1.048× | | `nn2x2_512` | 1.011× |
| `nn4_512` | 1.038× | | `nn4_1024` | 1.006× |
| `nn2_512` | 1.035× | | `matmul1024` | 1.004× |

No parallel cell regressed. The one-thread leg was re-checked at depth (min of 61, no workers
spawned): **1.001–1.073×** — the 0.914/0.957 readings in the 41-cycle run were noise on cells
with 38.7% and 9.2% spread, which is what a control that cannot move looks like when it is
measured too shallowly.

## 4. Cumulative — all four steps, both machines

mapal fma par, median ms. `numpy` is the better of its 1-thread and multi-thread legs.

**M4 Pro, 14 threads (Accelerate)**

| shape | d | S49 | P0+P0b | +P2 | **+P1** | P1 × | **total ×** | numpy | mapal × np |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nn2 | 128 | 0.1720 | 0.0961 | 0.0990 | **0.0814** | 1.216 | **2.113** | 0.0138 | 0.17 |
| nn4 | 128 | 0.3636 | 0.1815 | 0.1592 | **0.1448** | 1.099 | **2.511** | 0.0243 | 0.17 |
| nn2x2 | 128 | 0.2301 | 0.1344 | 0.1326 | **0.1268** | 1.045 | **1.814** | 0.0242 | 0.19 |
| nn2 | 256 | 0.2422 | 0.1611 | 0.1545 | **0.1530** | 1.010 | 1.583 | 0.0489 | 0.32 |
| nn4 | 256 | 0.4751 | 0.3199 | 0.3116 | **0.2848** | 1.094 | 1.668 | 0.0925 | 0.32 |
| nn2x2 | 256 | 0.3161 | 0.2399 | 0.2062 | **0.1968** | 1.048 | 1.606 | 0.0920 | 0.47 |
| nn2 | 512 | 0.4982 | 0.4489 | 0.4361 | **0.4193** | 1.040 | 1.188 | 0.3294 | 0.79 |
| nn4 | 512 | 0.8718 | 0.7634 | 0.6602 | **0.6560** | 1.006 | 1.329 | 0.6617 | **1.01** |
| nn2x2 | 512 | 0.5712 | 0.5774 | 0.4952 | **0.4701** | 1.053 | 1.215 | 0.6600 | **1.40** |
| nn2 | 1024 | 1.5915 | 1.5262 | 1.4930 | **1.4670** | 1.018 | 1.085 | 2.4126 | **1.64** |
| nn4 | 1024 | 3.1681 | 3.0874 | 2.9949 | **2.9609** | 1.011 | 1.070 | 4.1360 | **1.40** |
| nn2x2 | 1024 | 2.8298 | 2.8659 | 2.7946 | **2.7317** | 1.023 | 1.036 | 4.4923 | **1.64** |
| nn2 | 2048 | 10.3757 | 10.1554 | 10.1464 | **10.1062** | 1.004 | 1.027 | 14.3618 | **1.42** |
| nn4 | 2048 | 20.6452 | 20.4990 | 20.5368 | 20.5043 | 1.002 | 1.007 | 25.9543 | **1.27** |
| nn2x2 | 2048 | 19.9572 | 19.7195 | 20.0319 | **19.7003** | 1.017 | 1.013 | 26.9616 | **1.37** |

**i9-14900F, 32 threads (OpenBLAS)**

| shape | d | S49 | P0+P0b | +P2 | **+P1** | P1 × | **total ×** | numpy | mapal × np |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nn2 | 128 | 0.1902 | 0.1489 | 0.1091 | **0.0952** | 1.146 | **1.998** | 0.0749 | 0.79 |
| nn4 | 128 | 0.3389 | 0.2560 | 0.1975 | **0.1799** | 1.097 | **1.883** | 0.1381 | 0.77 |
| nn2x2 | 128 | 0.1850 | *0.2164* | 0.1249 | **0.1199** | 1.042 | **1.543** | 0.1381 | **1.15** |
| nn2 | 256 | 0.2489 | 0.2219 | 0.1769 | **0.1443** | 1.225 | **1.725** | 0.4597 | 3.19 |
| nn4 | 256 | 0.4771 | 0.4097 | 0.3120 | **0.2851** | 1.095 | 1.674 | 0.9048 | 3.17 |
| nn2x2 | 256 | 0.3017 | *0.3326* | 0.2321 | **0.2247** | 1.033 | 1.342 | 0.9063 | 4.03 |
| nn2 | 512 | 0.4825 | 0.4542 | 0.4955 | **0.4602** | 1.077 | 1.048 | 3.3695 | 7.32 |
| nn4 | 512 | 0.9800 | 0.9360 | 0.9247 | **0.8973** | 1.031 | 1.092 | 2.6755 | 2.98 |
| nn2x2 | 512 | 0.8248 | 0.8245 | 0.8056 | **0.7844** | 1.027 | 1.051 | 6.7186 | 8.57 |
| nn2 | 1024 | 3.1651 | 3.1079 | 3.1951 | **3.0763** | 1.039 | 1.029 | 5.2356 | **1.70** |
| nn4 | 1024 | 6.6681 | 6.6461 | 6.6958 | **6.4310** | 1.041 | 1.037 | 8.3093 | **1.29** |
| nn2x2 | 1024 | 5.9726 | 6.1399 | 6.1297 | 6.0073 | 1.020 | 0.994 | 8.3128 | **1.38** |
| nn2 | 2048 | 22.9509 | 22.9982 | 23.0431 | 22.9513 | 1.004 | 1.000 | 29.4010 | **1.28** |
| nn4 | 2048 | 45.8525 | 46.4807 | 46.6243 | **45.9129** | 1.015 | 0.999 | 57.7627 | **1.26** |
| nn2x2 | 2048 | 45.3210 | 44.9050 | 44.8255 | **44.6786** | 1.003 | 1.014 | 57.1754 | **1.28** |

**P1 is the first step that moved d ≥ 1024**, and only on the i9 (1.02–1.04×) — that is the
host's park being the last thing on the critical path once everything else is gone, and it
shows up where the wave count is highest. On the M4 d ≥ 1024 stays at 1.00–1.02× as before:
dispatch there is ~1% of the wall and there is nothing left to take.

## 5. Gate

`cargo test --workspace --release`: **1054 passed / 0 failed**, including the 1,280-run
differential byte-equality oracle and `mapal-rt`'s 21 tests. `conv2d_1024` × 300 with zero
failures against the fixed ordering. `cargo fmt --all --check` clean. `mapal-ir` untouched.

## 6. Still open

- **`nn2x2_1024` on the i9 is 0.994× of S49** — the last cell still not recovered from the
  P0 regression, now within 0.6%.
- **Below d=512 numpy still wins on the M4** (0.17–0.47×). At d=128 the whole numpy forward
  pass is 24 µs; the remaining mapal time is now genuinely close to what the parked wave
  floor allows, and P1 proved workers cannot spin their way past it. The next lever is
  fewer waves, not cheaper ones.
- The one-thread deficit (0.60–0.87× of numpy) is untouched — S48's standing P0, and now the
  dominant term at every size where dispatch is not.
