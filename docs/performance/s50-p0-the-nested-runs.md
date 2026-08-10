# S50 — P0 + P0b built and measured: the nested runs are gone

Date: 2026-08-09 · plan: `components/runtime/plans/plan-s50-scheduler-handoff.md`
Measurement this acts on: `s49b-the-dispatch-price.md`.
Raw: `benches/results-s50/` (M4 Pro) and `benches/results-s50/i9/`.

## 0. Where the scheduling decision lives

Sapir's constraint: **the comptime structuring belongs in the backend, per backend, so that
the IR's geometry analysis stays generic** (ADR-0032). Held by construction, and checkable:

| crate | P0/P0b diff | what it decides |
| --- | ---: | --- |
| `backends/llvm/src/func/drive.rs` | +123 | that a *packed* tile site is two tasks and how they are registered — a packing decision, and packing is a machine fact (`packing_site`, `TargetProfile::tile_j`) |
| `mapal-rt/src/lib.rs` | +74/−9 | how a lane is woken and where a dep-unlocked task is placed — the CPU backend's runtime |
| `mapal-ir` | **0** | unchanged by this work |

`git diff -- crates/mapal-ir/` over the whole session has four lines matching
`pack|lane|wake|notify|thread|dispatch`: two are reflowed comments and two are
`IrBuilder::pack` (the product constructor) in a test. The task DAG `path_plan` deduces is
still pure graph geometry; every scheduling and machine decision is downstream of it.

## 1. What was built

**P0 — a packed tile site is two tasks in the outer run, not a wrapper opening two nested
runs.** `drive.rs::emit_parallel` registers `@task{n}_pack` at an appended task index with
the matmul's own dependency set, `@task{n}_slice` at the site's index, and one
`mapal_par_dep(pack, matmul)`. `emit_task` stops emitting the wrapper. Nested
`mapal_par_begin` count per program: `nn2` 4 → **0**, `nn4`/`nn2x2` 8 → **0**, `matmul*`
2 → **0**.

**P0b — a dispatch wakes only the lanes it filled.** `Pool`'s single `wake_epoch`/`wake`
pair became one `LanePark` per lane; `enqueue` records the lanes it pushed to and notifies
exactly those. Empty-dispatch cost at 14 lanes, `examples/dispatch_cost.rs`:
**5.125 → 0.458 µs (11×)**; at 10 lanes 9.792 → 0.458 (21×); a 32-slice dispatch, which
fills every lane, is unchanged (33.8 → 35.2, noise) — exactly the shape predicted.

**The third change, which was not in the plan and carried most of the win.** S43 §7 refused
this shape because a dep-unlocked task is scheduled `Placement::Local(lane)` — the lane that
finished its producer — and the matmul would land on one deque instead of the rank-sorted
spread it gets at launch. The first build kept `Local` and split it for multi-piece tasks;
that was measured against seeding the same tasks, and seeding won everywhere:

| shape (M4, par, median of 41) | `Local` | `Seed` | |
| --- | ---: | ---: | ---: |
| `nn2_128` | 0.1290 | **0.0839** | 1.54× |
| `nn4_128` | 0.2447 | **0.1550** | 1.58× |
| `nn2x2_128` | 0.1813 | **0.1215** | 1.49× |
| `nn2x2_512` | 0.6097 | **0.5821** | 1.05× (and it removed a regression) |

`Local` was affinity for scalar glue and a serialization hazard for anything bulk, which is
what a packed matmul became the moment it stopped hiding inside a nested run. `Placement`
now has one variant.

## 2. Interleaved A/B against S49 — M4 Pro, 41 cycles

Both arms are complete toolchains (`emit` + `libmapal_rt.a`), built from the two trees and
run alternating; values byte-identical between arms on every shape.

| shape | 1t | par | par spread (pre) |
| --- | ---: | ---: | ---: |
| `nn4_128` | 0.994 | **2.201×** | 92.5% |
| `nn2_128` | 1.044 | **2.011×** | 150.8% |
| `nn2x2_128` | 1.080 | **1.828×** | 132.4% |
| `nn2_512` | 0.957 | **1.180×** | 43.0% |
| `nn4_512` | 0.982 | **1.147×** | 24.6% |
| `nn4_1024` | 0.996 | 1.029 | 24.4% |
| `nn4_2048` | 1.014 | 1.015 | 2.5% |
| `nn2x2_512` | 0.997 | 1.005 | 21.3% |
| `nn2x2_2048` | 0.999 | 1.003 | 3.9% |
| `conv2d_1024` | 0.999 | **1.179×** | 82.8% |
| `matmul1024` | 0.998 | **1.076×** | 13.7% |
| `fir_1048576` | 0.995 | **1.072×** | 27.8% |
| `matmul4096` | 1.000 | 1.003 | 26.7% |

**No cell regressed.** The one-thread column is flat by construction — `MAPAL_PAR=1` spawns
no pool, so it is the control that must not move, and does not. The published ladder gained
too (`conv2d_1024` 1.18×, `matmul1024` 1.08×): seeding dep-unlocked tasks helps every shape
that has a dependent bulk task, not only the packed ones.

## 3. Both machines, against S49 and against numpy

Parallel leg, median, ms. `numpy` is the better of its 1-thread and multi-thread legs.

**M4 Pro, 14 threads, numpy → Accelerate**

| shape | d | S49 | **S50** | speedup | numpy | mapal × numpy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nn2 | 128 | 0.1720 | **0.0961** | 1.789 | 0.0140 | 0.146 |
| nn4 | 128 | 0.3636 | **0.1815** | **2.004** | 0.0239 | 0.132 |
| nn2x2 | 128 | 0.2301 | **0.1344** | 1.713 | 0.0232 | 0.173 |
| nn2 | 256 | 0.2422 | **0.1611** | 1.503 | 0.0487 | 0.302 |
| nn4 | 256 | 0.4751 | **0.3199** | 1.485 | 0.0926 | 0.289 |
| nn2x2 | 256 | 0.3161 | **0.2399** | 1.318 | 0.0915 | 0.382 |
| nn2 | 512 | 0.4982 | **0.4489** | 1.110 | 0.3267 | 0.728 |
| nn4 | 512 | 0.8718 | **0.7634** | 1.142 | 0.6555 | 0.859 |
| nn2x2 | 512 | 0.5712 | 0.5774 | 0.989 | 0.6510 | **1.128** |
| nn2 | 1024 | 1.5915 | **1.5262** | 1.043 | 2.2055 | **1.445** |
| nn4 | 1024 | 3.1681 | **3.0874** | 1.026 | 3.8045 | **1.232** |
| nn2x2 | 1024 | 2.8298 | 2.8659 | 0.987 | 4.1383 | **1.444** |
| nn2 | 2048 | 10.3757 | **10.1554** | 1.022 | 13.8746 | **1.366** |
| nn4 | 2048 | 20.6452 | **20.4990** | 1.007 | 25.1095 | **1.225** |
| nn2x2 | 2048 | 19.9572 | **19.7195** | 1.012 | 25.9981 | **1.318** |

**i9-14900F, 32 threads, numpy → OpenBLAS**

| shape | d | S49 | **S50** | speedup | numpy | mapal × numpy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nn2 | 128 | 0.1902 | **0.1489** | 1.278 | 0.0751 | 0.505 |
| nn4 | 128 | 0.3389 | **0.2560** | 1.324 | 0.1379 | 0.539 |
| nn2x2 | 128 | 0.1850 | 0.2164 | **0.855** | 0.1381 | 0.638 |
| nn2 | 256 | 0.2489 | **0.2219** | 1.122 | 0.4605 | 2.075 |
| nn4 | 256 | 0.4771 | **0.4097** | 1.165 | 0.9054 | 2.210 |
| nn2x2 | 256 | 0.3017 | 0.3326 | **0.907** | 0.9069 | 2.727 |
| nn2 | 512 | 0.4825 | **0.4542** | 1.062 | 2.1153 | 4.657 |
| nn4 | 512 | 0.9800 | **0.9360** | 1.047 | 6.7184 | 7.178 |
| nn2x2 | 512 | 0.8248 | 0.8245 | 1.000 | 3.6857 | 4.470 |
| nn2 | 1024 | 3.1651 | **3.1079** | 1.018 | 5.1450 | 1.655 |
| nn4 | 1024 | 6.6681 | 6.6461 | 1.003 | 8.3369 | 1.254 |
| nn2x2 | 1024 | 5.9726 | 6.1399 | 0.973 | 8.3493 | 1.360 |
| nn2 | 2048 | 22.9509 | 22.9982 | 0.998 | 29.3461 | 1.276 |
| nn4 | 2048 | 45.8525 | 46.4807 | 0.986 | 57.5937 | 1.239 |
| nn2x2 | 2048 | 45.3210 | **44.9050** | 1.009 | 57.4839 | 1.280 |

The d≥1024 cells are unchanged, as they must be: dispatch was already 1.4–9% of those walls.

## 4. Open — the two-chain shape on 32 lanes

`nn2x2` at d ≤ 256 on the i9 is **0.855× / 0.907×** of S49, reproduced across runs, while
every single-chain shape on the same box gains 1.12–1.32×. On the M4 the same shape gains
1.71× at d=128 and sits at 0.987–1.005 at 512/1024.

The pattern is *more lanes × more concurrently live tasks*, and the standing hypothesis —
**named, not proven** — is `Run.state`: one `Mutex<RunState>` taken twice per slice, plus a
`dependents.clone()` heap allocation on every task completion. Under S49 each nested run had
its own mutex, so four sites meant four mutexes; now every site's completions funnel through
one. That is P2 in the plan ("the run graph is a constant; stop rebuilding it"), and it now
has a measurement pointing at it rather than an argument.

## 5. Gate

`cargo test --workspace --release`: **1054 passed / 0 failed**, including the 1,280-run
differential byte-equality oracle — **results are unchanged**. `cargo fmt --all --check`
clean. One test was rewritten rather than deleted:
`packed_wrapper_dispatches_the_b_pack_across_the_pool` pinned the nested-wrapper shape (S43);
it now pins the replacement — exactly one `mapal_par_begin` in the module, the pack a `Split`
dispatch with a non-zero quantum, the matmul still a `Split` with its region quantum, and the
`mapal_par_dep(pack, matmul)` edge that replaced the nested `finish` as the happens-before.
