# S51 — the packs are O(N²) against an O(N³) matmul, and one of them should not happen

Date: 2026-08-10 · Machine: **Apple M4 Pro**, 10 P + 4 E, 2 SME units.
L1D 128 KB · line 128 B · shared L2 16 MB. Baseline `c78c9af`, clean.
Plan: `components/backend-llvm/plans/plan-s51-the-one-thread-kernel.md`.
Instruments: `benches/nn/{pack_ab.sh, pack_report.py}` (new), `benches/sme/packconflict.c`
(new). Mutex: `benches/perflock.sh`. Samples: `benches/results-s51/`.

## 0. The headline

| | verdict |
| --- | --- |
| the P0 as handed over | **two gaps, not one** — its number is d ≤ 512, its prescription (residency) was measured at N=4096 and cannot bind below 12 MB |
| what binds at d ≤ 512 | **staging, not compute.** Both packs are O(N²) against an O(N³) matmul |
| the B pack, measured | **a 1.46–1.52× LOSS at d=128**, break-even at d=1024, a **1.53× win at d=2048** |
| where the crossover sits | **12.6 MB working set — on S43's independently measured 8–12 MB L2 knee.** Not fitted |
| what ships | `working_set > l2_bytes`, an **already-recorded profile field**. No new constant |
| threaded | **the win HOLDS and at d=256 GROWS** (1.20× at 1t → 1.46× threaded), as rule 24 pre-registered |
| `generic` | **0 of 174 cells moved** |
| gate | **1056 passed / 0 failed**, fmt clean — the same count as the S50 baseline |
| the A pack | a real 1.77–2.75× traversal win above `ap` = L1D, worth **~5–8% integrated**. Measured, not built |
| a stale instrument found | `packcost.c` claims to replicate the emitted pack and replicates the **pre-S42** one |
| S42's row-outer reorder | **worth ~2× at N ≤ 512 and a 1.455× REGRESSION at N=2048** — it moved the waste, it did not remove it |

## 1. Why the P0 had to be split before anything could be built

`next-session.md` §1 carried one P0: *"single-thread kernel is 0.28–0.35× of NumPy below
d=512 … reuse-structure change, not another blocking level."* The number and the
prescription come from different regimes.

At d=512 the three arrays are 1 MB each. S43 §4 measured this part's cache ladder with
`loadlevel.c` and found **no L1 cliff and no L2-slice cliff** — flat ~1990 GF/s from a
32 KB buffer to an 8 MB one — so *an operand living in L2 rather than L1 costs nothing
here*. A 3 MB working set is inside L2. **Residency cannot be the term the P0's own number
names.**

| regime | ratio vs NumPy, 1t | can residency bind? |
| --- | ---: | --- |
| d ≤ 512 | 0.31–0.47 | **no** — working set ≤ 3 MB |
| d ≥ 1024 | 0.72, 0.61 | yes, d=2048 crosses the knee |
| i9, all sizes | 0.87–0.96 flat | different machine, no SME, unreachable since S49 |

## 2. What was refuted before it was chased

Recorded so no future session spends a day on any of it.

| candidate | measurement | verdict |
| --- | --- | --- |
| streaming-mode ABI per panel call | `smcost.c`: 1.0 ms / 131072 calls = **7.6 ns** | **refuted** — 0.6% of the d=128 wall |
| `kc` blocking | +6.1% 1t, −25.5% threaded | ships OFF |
| `nc` blocking as built | 1t +18.7% at nc=1024; threaded loses at every width | ships OFF |
| loads vs `fmopa` issue | 4183.8 vs 4159.6 GF/s | refuted, 1.006× |
| the kernel itself | `@mapal_sme_panel` lifted: **3831 GF/s = 93% of ceiling** | not the problem |

## 3. H1 — the B pack. Measured, and it does not pay below the knee

`benches/nn/pack_ab.sh`, `--contract` face, 15 interleaved cycles, arms round-robin inside
every cycle, machine exclusive.

**Three gates before a single timing was read.** (a) Every arm's output **bit-identical**
to the pack-on arm — the pack is a layout, not arithmetic, so `cmp` is the right gate and
40 legs passed it. (b) An arm whose emission equalled the control's would be reported
**VOID**, never as "no effect" (S43 §8.4's `nc`=768 lesson); none were — the four
`@task{13..16}_pack` functions vanish entirely. (c) A **byte-identical control arm** —
ON's own flags compiled to a second binary — carried through every cell, because S39
measured −5.9%…+1.2% between byte-identical binaries on this machine.

| d | working set | 1t OFF/ON | sep | par OFF/ON | sep | CTL/ON (noise floor) |
| ---: | ---: | ---: | --- | ---: | --- | ---: |
| 128 | 0.2 MB | **1.457 / 1.517×** | DISJOINT | 1.507 / 1.451× | overlap | 1.014 / 1.024× |
| 256 | 0.8 MB | 1.199 / 1.213× | DISJOINT | **1.457 / 1.412×** | DISJOINT | 0.996 / 1.071× |
| 512 | 3.1 MB | 1.127 / 1.125× | DISJOINT | 1.317 / 1.233× | DISJOINT | 0.997 / 0.990× |
| **1024** | **12.6 MB** | **0.992 / 1.000×** | **overlap** | 1.056 / 0.996× | mixed | 0.999 / 0.996× |
| 2048 | 50 MB | 0.651 / 0.658× | DISJOINT | 0.674 / 0.662× | DISJOINT | 1.008 / 1.005× |

(`nn4 / nn2x2`. `OFF/ON > 1` ⟹ the pack is a loss.)

### The crossover was not fitted — it landed on a knee another instrument measured

`pack_ab.sh` knows nothing about caches. S43's `loadlevel.c` put this part's one real cache
boundary — shared L2 → DRAM — between **8 and 12 MB**. The pack flips sign at a **12.6 MB**
working set. Three sizes below and one above are all called correctly by
`working_set > l2_bytes`, and `l2_bytes` is a field the profile already records.

The data does not separate this from `k·c·sizeof > l2_bytes` (B alone): both predicates are
right at all five measured sizes. The whole working set is what ships, because A and the
output genuinely compete with `b` for the same L2 — if a later size separates them, that
measurement decides, not this paragraph.

### Rule 24 was pre-registered and it paid

Written in the plan **before** the run: a pack is a **per-core** cost, so by S44's rule 24
its removal should hold or grow with thread count. At d=256 it **grew** — 1.20× at one
thread, **1.46× threaded**. This is a win in the configuration that ships.

### At d=1024 the choice is within noise, and the predicate picks not-packing

1t reads 0.992× / 1.000× and threaded 1.056× / 0.996×, all overlapping or mixed. Stated
rather than glossed: at exactly the knee, neither answer is measurably better.

## 4. What shipped, and the two byte-identity proofs that make re-measuring unnecessary

`TargetProfile::pack_pays` + `packing_site(site, profile, contract)`. Six call sites read
one predicate — the buffer's existence, the copy that fills it, the slice function that owns
it, the prefetch declaration, the SME load-width clause and the copy's own assertion — because
a site packed by one and not another is a kernel reading a layout nobody wrote.

**Scope is narrower than "this profile has an SME unit", and a test forced that correction.**
The first version gated on `profile.sme.is_some()`, and
`sme_profile_differs_from_apple_m_only_in_dispatch_kind_where_the_rung_declines` failed —
correctly. An f64 or exact-face site on `apple-m4-sme` runs the **NEON** packing rung, whose
crossover nobody has measured. The threshold now applies only where the site takes the SME
rung (`func/sme.rs::sme_shape`, clauses 1–7 split out to break the clause-8 cycle). Verified:
`attn_256` on the exact face still packs.

Two `cmp`s replace a second measurement campaign:

| cell | check | result |
| --- | --- | --- |
| d ≤ 1024 | new default vs **old `--no-pack`** | **byte-identical** ⇒ the measured OFF arm *is* the shipped emission |
| d = 2048 | new default vs **old default** | **byte-identical** ⇒ the 1.53× the pack wins there cannot have been lost |
| `--target=generic` | 174-cell emission sweep | **0 cells moved** |

So §3's table is the shipped result, established by byte-identity rather than by a second
timing run — the same argument S39 and S40 used to show no perf run was owed.

*(The sweep reports 3 of 174 emissions FAILED on both sides: `examples/vector.mapal` does not
parse. Pre-existing since S43, identical on both arms, still open.)*

## 5. H2 — the A pack. A real traversal win, measured, NOT built

### 5a. The instrument that was measuring code that is not in the binary

`benches/sme/packcost.c` states it *"replicates the EMITTED nest exactly
(func/sme.rs::emit_tiled_map_sme)"* and runs **k outer, row inner**. The emitter stopped
emitting that in **S42**, which reordered the pack row-outer (`func/sme.rs:385`: *"ROW OUTER,
k INNER — and the order is the whole point"*). The comment was never updated.

Consequence: `packcost.c`'s **8.46 ms** is the cost of the *pre-S42* pack, and this session's
own first pricing of the A pack — 0.199 ms/layer at d=512, projecting 1.47× — was built on it
and is **withdrawn**. The real figure is **0.041 ms**, 4.9× cheaper.

> **28. An instrument that names the code it replicates must be re-read against that code,
> not trusted for its comment.** `packcost.c` was correct when written and silently stopped
> being correct one session later; three documents cite its number, and a plan was written
> from it.

### 5b. The corrected measurement, and its prediction held

`benches/sme/packconflict.c`, best of 7, value-gated (checksum-equal across arms, and the
gate was verified to *fail* on an injected bug before the run was trusted — rule 23).

The prediction written into the file before the run: reads are already contiguous after S42,
so the waste moved to the **stores** (`ap[pk·32+pi]` — consecutive `pk` are 128 B apart, one
store per line); whether that costs anything depends on `ap = 32·k·4` against a 128 KB L1D.

| N | `ap` KB | ctl ms | blk/ctl | old/ctl | drift | ctl ns/elem |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 16 | 0.002 | 1.000× | 0.500× | 1.000× | 0.122 |
| 256 | 32 | 0.010 | 1.000× | 0.526× | 1.000× | 0.153 |
| 512 | 64 | 0.041 | 1.051× | 0.612× | 1.000× | 0.156 |
| 1024 | **128 = L1D** | 0.274 | 1.612× | 1.046× | **1.507× VOID** | 0.261 |
| 2048 | 256 | 2.075 | **2.752×** | 1.455× | 1.000× | 0.495 |
| 4096 | 512 | 9.087 | 1.771× | 1.140× | 1.007× | 0.542 |

`ctl ns/elem` is flat at ~0.15 up to N=512 and rises from exactly where `ap` crosses L1D.
Blocking is flat below the boundary and wins above it. **N=1024 is VOID** — `ctl` measured
twice differed by 1.507×, so that cell is not quoted.

### 5c. S42's reorder is a regression at large N

The `old/ctl` column prices the S42 change on its own terms for the first time: **0.50–0.61×
at N ≤ 512** (the reorder bought ~2×) and **1.455× at N=2048, 1.140× at N=4096** — the
pre-S42 order is *faster* there. S42 fixed the read side and moved the waste to the store
side, where `ap` stops fitting L1. Blocking beats both (1.89× over `old` at N=2048), so the
answer is to block, not to revert.

### 5d. What it is worth, and why it did not ship this session

The A pack is **12–17% of the wall**, not the 60% the stale number implied:

| d | A pack ×4 layers | measured wall (1t) | share | blocking would save |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.008 | 0.0501 | 16% | nothing (flat below L1D) |
| 512 | 0.164 | 1.251 | 13% | nothing |
| 1024 | 1.096 | 8.787 | 12% | ~1.05× |
| 2048 | 8.30 | 72.52 | 11% | ~1.08× |

Real, clean, value-identical, no constant — and **~5–8% integrated at d ≥ 1024 only**. It is
scoped and ready; it is not this session's headline and it was not built on the way past.
S44's rule 3 applies to those projections: a probe over-priced its integrated win by 1.7×
last time, so they are projections until measured in the pipeline.

## 6. The table that changes

Threaded, `nn4` vs NumPy, from §3's ratios applied to the S50 `-s50final` run. **Projected,
not re-measured**, and flagged as such because absolutes are not comparable across runs
(rule 27) — the ratios are within-run and valid.

| d | S50 shipped | with the threshold | vs NumPy, before → after |
| ---: | ---: | ---: | --- |
| 128 | 0.0791 | 0.0525 | 0.30 → **0.45** |
| 256 | 0.1923 | 0.1320 | 0.48 → **0.70** |
| 512 | 0.6257 | 0.4751 | 1.03 → **1.36** |
| 1024 | 2.9260 | 2.7707 | 1.29 → **1.36** |
| 2048 | 20.41 | 20.41 (identical) | 1.22 |

The NumPy tie moves from d=512 to roughly d=384.

## 7. What is NOT claimed

- **Nothing here touches the i9.** Its gap is flat 0.87–0.96×, a micro-kernel question
  against OpenBLAS on AVX2, and the box has been unreachable since mid-S49.
- **No threaded headline beyond the ceiling.** d=2048 threaded already runs ≈3374 GF/s
  against a measured ~4100 aggregate — 82% of the machine — so at most ≈1.22× remains
  threaded on this part, ever.
- **The B-alone predicate is not excluded**, only unselected; five sizes do not separate it
  from the working-set form.
- **H2's integrated numbers are projections**, and the N=1024 probe cell is void.
- **`pack_pays` is unmeasured on the NEON rung and is not applied there.**
