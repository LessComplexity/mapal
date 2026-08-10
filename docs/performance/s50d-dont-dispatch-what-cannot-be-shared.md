# S50d — the dispatch threshold: what shipped, and what it cannot decide yet

Date: 2026-08-10 · plan: `components/runtime/plans/plan-s50-scheduler-handoff.md`
Predecessors: `s50-p0-the-nested-runs.md`, `s50b-p2-the-shared-lock.md`,
`s50c-p1-spinning-workers-refuted.md`.

## 0. The observation this acts on

BLAS does not parallelise small GEMMs at all — OpenBLAS has
`GEMM_MULTITHREAD_THRESHOLD`, below which it runs one thread and pays **zero**
scheduling. Measured, 4-layer net, numpy's own two legs:

| machine | d | numpy 1t | numpy mt | threading gain |
| --- | ---: | ---: | ---: | ---: |
| M4 | 128 | **0.0243** | 0.0254 | 0.96 |
| M4 | 512 | **0.6617** | 0.8592 | 0.77 |
| M4 | 1024 | 6.2966 | **4.1360** | 1.52 |
| i9 | 128 | **0.1381** | 0.5172 | **0.27** |
| i9 | 512 | 6.7199 | **2.6755** | 2.51 |

Mapal does the opposite — it dispatches everything:

| machine | d | mapal 1t | mapal par | par gain |
| --- | ---: | ---: | ---: | ---: |
| M4 | 128 | **0.0790** | 0.1448 | **0.55** |
| M4 | 256 | **0.2711** | 0.2848 | 0.95 |
| i9 | 128 | **0.1618** | 0.1799 | 0.90 |

and it is not only the NN shapes: `conv2d_512` is 1t 0.0476 against par 0.0569, and
`reduce_1048576` is 1t 0.4971 against par 0.5427.

## 1. What shipped — no threshold, no constant

**A task cut into exactly one slice is executed by the calling thread instead of being
queued.** Queueing it buys a wave (11–41 µs parked) and one lane's worth of work, which is
the work the calling thread was going to be able to do anyway. There is nothing to decide:
"one slice" is already determined by `slice_ranges` from the compiler's quantum and the lane
count.

Every scalar-glue task in a plan is this shape. `nn2_128`'s plan has fifteen tasks and eight
of them are capture tuples and print cones — each one was costing a full dispatch.

Two constraints found while building it:

- **`mapal_par_launch` must only seed.** The emitted host runs its own spine work between
  launch and finish, and the clock fence (plan-s33b) is dependency edges evaluated from that
  spine; a launch that executed tasks itself would run bracketed work before the host reached
  the pinned clock read. `mapal-rt`'s own test asserts this in so many words, and it caught
  the first build. Inlining is therefore enabled only for mid-run unlocks, where the calling
  thread is already inside the run.
- **Queue before you inline.** The pool gets its share of the batch *first*, so workers are
  already moving while the calling thread does the inline part; otherwise inlining serialises
  what it was supposed to accelerate.

### Measured, interleaved against the P1 tree, M4, 41 cycles

| shape | par | | shape | par |
| --- | ---: | --- | --- | ---: |
| `nn2_128` | **1.098×** | | `nn2x2_512` | 1.013× |
| `nn4_256` | **1.056×** | | `saxpy_1048576` | 1.013× |
| `nn2_256` | 1.045× | | `reduce_1048576` | 1.012× |
| `nn2x2_128` | 1.045× | | `matmul512` | 1.008× |
| `nn4_128` | 1.024× | | `matmul1024`, `nn4_512`, `nn4_2048` | 1.000 |

Small and real at the sizes where waves dominate, neutral everywhere else. `conv2d_512` reads
0.927 on a cell with 251% spread — noise, and the shape is one of the ones a *work* threshold
would take out of the pool entirely.

**Gate: 1054 passed / 0 failed**, differential oracle included; `mapal-rt` 21/21; fmt clean.

## 2. What is NOT shipped, and exactly why

The remaining case is the one that costs most: a task with **several** slices that is still
too small to pay for its wave. `nn4_128`'s matmul cuts 4 slices and parallel is **1.88×
slower than serial**. Deciding that needs an estimate of the task's serial time, and a static
work count cannot supply one.

**A pure operation count does not order the corpus.** Estimated flops against measured serial
time, M4:

| task | est. flops | measured serial | achieved |
| --- | ---: | ---: | ---: |
| `nn4_256` layer | 3.4e7 | 68 µs | 495 GF/s |
| `conv2d_1024` | 1.9e7 | 250 µs | 76 GF/s |

`conv2d_1024` has **half** the flops and takes **3.7× longer**, so no threshold on flops can
inline the first and dispatch the second — and those are the two calls that must go opposite
ways. The reason is that one is compute-bound and the other is memory-bound; a single work
metric cannot span both rungs.

**A roofline estimate does order it.** `est_ns = max(2·n·k / flops_per_ns, bytes / bytes_per_ns)`
with per-lane rates from the profile, against the wave cost:

| task | compute term | memory term | est | wave | verdict | measured par gain |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `nn4_128` layer | 4.2 µs | 2 µs | 4.2 µs | ~25 µs | inline | 0.55 ✓ |
| `nn4_256` layer | 34 µs | 8 µs | 34 µs | ~25 µs | marginal | 0.95 ✓ |
| `nn4_512` layer | 268 µs | 31 µs | 268 µs | ~25 µs | dispatch | 2.13 ✓ |
| `conv2d_512` | 4.7 µs | 21 µs | 21 µs | ~25 µs | marginal | 0.84 ✓ |
| `conv2d_1024` | 19 µs | 84 µs | 84 µs | ~25 µs | dispatch | 3.33 ✓ |
| `matmul1024` | 2.1 ms | 126 µs | 2.1 ms | ~25 µs | dispatch | 2.75 ✓ |

Six for six, including both cells that break the flop-only rule.

**The mechanism to ship it needs no ABI change and no runtime rule.** The emitter already
chooses `kind`, and a `kind = 0` task is one slice — which §1 now runs inline. So the whole
decision can live in `drive.rs`: estimate, and register the task `kind = 0` if it is under
the wave cost. That keeps it in the backend, per backend, with `mapal-ir` untouched, which is
the standing constraint.

**What it costs is three new `TargetProfile` fields** — `flops_per_ns`, `bytes_per_ns`,
`dispatch_ns` — across six profiles. Two of those six (`apple-m4-sme`, `raptorlake`) have
measurements behind every number; `generic`, `apple-m`, `zen3` and `cuda-ada` would be
carrying values nobody has measured, and `dispatch_ns` is genuinely lane-count dependent
while the emitter is not. That is eighteen numbers of which twelve would be invented, which
is a bigger commitment than a threshold sounds like, and it is a decision rather than an
implementation detail. **Not built. Recorded with its validation table so the next session
starts from the evidence rather than the argument.**

## 3. The one-thread gap, since it keeps showing up

Single thread only, 4-layer net:

| machine | d | mapal | numpy | mapal GF/s | numpy GF/s | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M4 | 128 | 0.0790 | 0.0243 | 212 | 691 | 0.31 |
| M4 | 512 | 1.3942 | 0.6617 | 770 | 1623 | 0.47 |
| M4 | 1024 | 8.7136 | 6.2966 | 986 | 1364 | 0.72 |
| M4 | 2048 | 73.17 | 44.44 | 939 | 1546 | 0.61 |
| i9 | 256 | 0.9464 | 0.9048 | 142 | 148 | 0.96 |
| i9 | 1024 | 59.44 | 52.16 | 145 | 165 | 0.88 |
| i9 | 2048 | 476.1 | 413.8 | 144 | 166 | 0.87 |

**Two different gaps, not one.**

*On the i9 it is small and flat — 0.87–0.96×.* Both sides are driving the same AVX2 FMA
units; a generated kernel within 12% of hand-tuned OpenBLAS is a micro-kernel quality gap,
and it is size-invariant, which S33 already identified as the signature of a steady
per-iteration deficit rather than a blocking failure.

*On the M4 it is large and size-dependent — 0.31× at d=128 rising to 0.72× at d=1024.* Both
sides reach a matrix unit (Accelerate → AMX, mapal → SME), and S42 measured this part at two
SME units of ~2000 GF/s each, so one thread's ceiling is ~2000. Mapal reaches 939–986.

**The size of that gap is already accounted for in the record, and the fix is already
measured.** S43's operand-residency instrument — two `and` masks patched into the emitted
`.ll`, assembly-verified, 174.596 → 101.854 ms — is **+71% at one thread**. 939 × 1.71 =
1606 GF/s, against Accelerate's 1546. The gap and the unclaimed win are the same size.

It is unclaimed because every lever measured so far that captures it costs the configuration
that ships: `kc` blocking is +6.1% at one thread and **−25.5% threaded**; `nc` blocking is
+18.7% at one thread and parity threaded. The 1.71× is confounded between L2 capacity and TLB
reach and no in-kernel arm separates them, so capturing it needs a **reuse-structure** change
rather than another blocking level. That is S48's standing P0, untouched by any of S49/S50,
and it is now the dominant term at every size where dispatch is not.
