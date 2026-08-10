# plan-s51 — the one-thread kernel: the P0 is not one gap, and the prescription it carries fits only one of them

Status: **RATIFIED by Sapir, H1 SHIPPED, H2 measured and scoped.** §9 reconciles the plan
against what the code became and what the machine said. Written before any S51 code.
Component: `backend-llvm` (the SME rung, `func/sme.rs`). Opens on `next-session.md` §1.
Priors that govern it: `performance/s43-residency-and-the-thermal-artifact.md`,
`performance/s44-conflict-not-capacity.md`, `plan-s43-nc-blocking.md` §8, `benches/sme/README.md`.

---

## 0. What the handoff says, and what the record says about it

`next-session.md` §1 states one P0:

> single-thread kernel is 0.28–0.35× of NumPy below d=512 … reuse-structure change, not another
> blocking level.

**The number and the prescription come from different size regimes.** The prescription is S43's
operand-residency finding, which was measured at **N=4096**. The number is the NN benchmark at
**d ≤ 512**. At d=512 the three arrays are 1 MB each — the whole working set is inside a 16 MB
shared L2, and S43 §4 measured that an operand living in L2 rather than L1 **costs nothing on this
part**. So residency cannot be the term the P0's own number names.

The measured 1-thread table, split by what can physically bind:

| regime | cells | ratio vs NumPy | can residency bind? |
| --- | --- | ---: | --- |
| **d ≤ 512, M4** | 128, 512 | **0.31, 0.47** | **no** — working set ≤ 3 MB, inside L2, and L2-vs-L1 is free (S43 §4) |
| **d ≥ 1024, M4** | 1024, 2048 | 0.72, 0.61 | yes — 2048 crosses the 8–12 MB knee |
| **i9, all sizes** | 256…2048 | 0.87–0.96 flat | different machine, different kernel (no SME), unmeasured since S48 |

Three regimes, three terms. This plan models each, gives each a falsifiable test with an existing
instrument, and pre-registers what rule 24 says each fix will do threaded.

---

## 1. The categorical model (FRAMEWORK §4)

The SME rung's placement, written out. Objects and morphisms first, because the whole diagnosis is
that **one morphism in this diagram has never been priced.**

`Dat` — `A, B, Out : Array f32`, plus two staged objects `APanel`, `BPanel`.

`Trn` — three transformations, and their `t_from`/`t_to`:

| `Trn` | signature | placed at | who emits it |
| --- | --- | --- | --- |
| `pack_a` | `A → APanel` | per i-panel, inside the task slice | `func/sme.rs::emit_tiled_map_sme` |
| `pack_b` | `B → BPanel` | once per site, before the matmul run | `func/core.rs::emit_pack_copy` + `@task{id}_pack` |
| `panel` | `APanel × BPanel → OutBlock` | per (i-panel, j-panel) | `module.rs::sme_panel`, in streaming mode |

`Loc` — `DRAM`, `sharedL2` (16 MB), `L1D` (128 KB, 128 sets × 8 ways × 128 B), `ZA` (the
accumulator file). Adjacency is the load path.

`Trm` — a line fetch `sharedL2 → L1D` (128 B), and a fill `DRAM → sharedL2`. **Every `Trm` here
carries a 128 B line, and that is the fact the whole plan turns on:** a transformation that reads
4 B per line pays 32× the `Trm` it uses.

### The `DataLoc` the packs exist to create

`pack_a` and `pack_b` compute nothing. Neither appears in the flop count. Each exists purely to
create a **second `DataLoc`** over an existing `Dat` at a friendlier `Loc` — this is FRAMEWORK
§4.2's "one type materialised as a DB row, a heap object, and bytes on a wire is three `DataLoc`s
over one `Dat`", instantiated in a cache hierarchy.

That makes their cost a **`Trm` price**, not a `Trn` price, and it is the price no record in this
repo has ever taken at small N. `packcost.c` took it at one cell only:

> **N=4096, 1 thread: `pack_a` moves 16.78M elements in 8.46 ms.**

`pack_a` moves exactly `N²` elements (`rows/32` panels × `32 × k`). So its price is
`N² × 0.50 ns`, and `panel`'s is `2N³ / 2000 GF/s = N³ × 1.0 ns / 2`. Their ratio is

```
pack_a / panel  =  (N² · 0.50) / (N³ · 0.50)  =  1/N        -- per layer, elementwise
pack_a + pack_b vs panel  ≈  0.8 / N          -- both packs, at the measured rate
```

**An O(N²) morphism against an O(N³) one.** At N=4096 the packs are 0.02% and invisible; at N=128
they are ~62% of the matmul's own time. That is the small-d gap, read off the model and one
existing measurement, with nothing new run.

### Composition rules this plan must preserve

1. Values bit-identical on every arm, checked **before** any timing is read (S43's gate).
2. `@mapal_sme_panel`'s emitted text and machine code unmoved by anything here — the kernel is
   already 93% of the two-unit ceiling (S43 §4c) and is not the subject.
3. Any threshold ships **derived from measured profile fields**, never a literal (ADR-0032; and a
   profile with no measurement carries `None` and keeps today's behaviour — S50's rule).

---

## 2. What is already refuted — do not retry

Recorded here so the session cannot spend a day on any of it. Each was measured, not argued.

| candidate | measurement | verdict |
| --- | --- | --- |
| the streaming-mode ABI per panel call (`smstart za`+`sm`, `d8–d15` spills) | `benches/sme/smcost.c`: **1.0 ms over 131072 calls = 7.6 ns/call** | **refuted.** At d=128 that is 64 calls = 0.49 µs of a 79 µs wall, 0.6% |
| `kc` blocking | +6.1% 1t, **−25.5% threaded** | ships OFF |
| `nc` blocking as built (with its `c/nc` A re-pack) | 1t optimum +18.7% at nc=1024; threaded **every arm loses**, monotone | ships OFF (`plan-s43-nc-blocking.md` §8.2) |
| loads competing with `fmopa` for issue | 4 loads per 4 `fmopa`, L1-resident: 4183.8 vs 4159.6 GF/s | refuted, 1.006× |
| L1 capacity / per-core L2 slice as a cliff | `loadlevel.c`: flat ~1990 GF/s from 32 KB to 8 MB | **no L1 cliff exists on this part** |
| B layout (whole-k slice vs kc repack) | `bslice.c` 1.065× | no |
| unrolling / software-pipelining the k loop | `pipe2.c` 1.001–1.002×; `mm4p.c` +0.6% | no |

**And the biggest one: the kernel is not the problem.** `@mapal_sme_panel` lifted verbatim runs
N=4096 at **3831 GF/s, 93% of the measured ~4100 ceiling**. Everything left is staging.

---

## 3. H1 — the small-d term: the packs are O(N²) against an O(N³) matmul

**Hypothesis.** Below some N, `pack_a + pack_b` cost more than they buy, because what they buy —
operand contiguity — is worthless when the whole working set is already L2-resident and L2 is free
on this part.

`pack_b` is the one that can simply not happen: `emit_tiled_map_sme`'s own doc records that
unpacked, "the kernel strides `b` by its row stride and reads it in place", with `bj = t` instead of
`t·k`. The legality is already there and there is already a `--no-pack` control (S27).

### The test — zero new code

`--no-pack` A/B on the NN shapes, 1 thread, machine exclusive, values gated first:

| d | pack ON (today) | pack OFF | prediction |
| ---: | --- | --- | --- |
| 128 | baseline | | **OFF wins** |
| 256 | baseline | | OFF wins |
| 512 | baseline | | crossover somewhere here |
| 1024 | baseline | | ON wins |
| 2048 | baseline | | ON wins, by more |

**Pre-registered threshold:** H1 is confirmed iff **d=128 OFF beats ON by ≥15% with disjoint
distributions** and **d=2048 ON beats OFF disjointly** — i.e. a real crossover, not a monotone
preference. A monotone result in either direction refutes the model and the session stops to
re-derive rather than shipping a threshold fitted to one end.

Control arm required (S44's lesson): **an arm whose emission is byte-identical to `ON` must be
run and must land on `ON`'s number.** `nc`=768 taught this repo that a silently-void arm reads as
"no effect".

### What ships if it confirms

A pack threshold in **exactly the shape S50 already built** for dispatch — `pack_b` fires only when
its cost is covered:

```
pack_b_ns  =  n_elems / bytes_per_ns · sizeof         -- the Trm it pays
panel_ns   =  2·rows·c·k / flops_per_ns               -- the Trn it serves
pack when   pack_b_ns  <  panel_ns · share            -- share: the fraction it can plausibly buy
```

Both constants are **already measured fields** on `TargetProfile` (`flops_per_ns`,
`bytes_per_ns`, S50). A profile carrying `None` packs as today — no part is made worse by a number
nobody took.

### Rule 24 pre-registration

Removing a pack removes a **per-core** cost (every lane packs its own panels). Rule 24 category 3
⇒ **the win holds or grows threaded.** If it shrinks, the classification is wrong and that is a
finding, recorded as such.

---

## 4. H2 — the large-d term: `pack_a` is a strided transpose that wastes 31 of every 32 bytes

`pack_a`'s emitted loop is, verbatim from `packcost.c`'s replica of the emitted nest:

```
for pk in 0..k:  for pi in 0..32:  ap[pk*32 + pi] = a[(i0+pi)*N + pk]
```

Reads stride by `N·4` bytes; writes are contiguous. **Each read consumes 4 B of a 128 B line.**
At N=4096 that is 16.78M line touches = 2.1 GB of `Trm` in 8.46 ms = **254 GB/s — exactly the
measured L1/L2 plateau (249 GB/s)**. The pack is not slow; it is running at the machine's line-fetch
ceiling and asking for 32× more line than it uses.

**And S44 already characterised this exact access pattern.** Its predictor,
`pressure = lines_live / (sets_reachable × ways)`, applied to `pack_a` (128 B lines, 128 sets,
8 ways assumed, `lines_live` = the 32 panel rows):

| N | read stride | sets/read | sets reachable | pressure | S44 predicts | measured 1t ratio vs NumPy |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 128 | 512 B | 4 | 32 | **0.125** | no conflict | 0.31 |
| 512 | 2 KB | 16 | 8 | 0.5 | no conflict | 0.47 |
| 1024 | 4 KB | 32 | 4 | **1.0** | at the threshold | 0.72 |
| **2048** | 8 KB | 64 | **2** | **2.0** | **win already near its ceiling** | **0.61** |
| 4096 | 16 KB | 128 | **1** | **4.0** | conflict | — |

**The existing data already agrees with the prediction in the one place it can be checked:** the
ratio *worsens* from 0.72 at d=1024 to 0.61 at d=2048 — the only non-monotone step in the table —
and d=2048 is exactly where pressure crosses S44's ~2 ceiling. That is a retrodiction, so it is
worth one line and no more; §4's test is what settles it.

### The test — again, an existing instrument

`benches/sme/packcost.c` already runs the two emitted pack shapes in plain C with no SME. Extend it
with a **blocked** and a **padded** arm (S44's two treatments, which do not compose — that
non-composition is what proved it was one defect) and sweep N ∈ {128, 512, 1024, 2048, 4096}.

**Pre-registered:** H2 is confirmed iff blocking the pack wins **≥1.5× at N=4096 and ≥1.3× at
N=2048**, and is **within noise at N=128 and N=512**. A win at N=128 refutes it — pressure there is
0.125 and there is nothing to break.

### What ships if it confirms

The treatment is already built and already proven a bijection: **`move_panel`** — S44's
permutation of the loop counter (`func/bulk.rs::move_panel_index`, +73 lines), whose whole
correctness argument is that `perm` is a bijection of `[0, n)` so parallel slices still partition
the outputs and `%lo`/`%hi` need no handling. Applying it to the `pack_a` counter is the same move
at a new site.

**And it retires S44's own open item in passing.** S44 recorded `w` as "program geometry supplied
by hand … a reported limitation, not a design", because a transpose-shaped map carries no fold and
`tile_site` hard-requires one. **Inside the SME rung there is no such problem: `w` is `site.c`,
already in the record.** The rung that could not derive its width has a caller that always knows it.

### Rule 24 pre-registration

An L1 set conflict is the textbook **per-core resource** — S44 measured `move_panel` growing
1.578× → 1.993× with thread count for exactly this reason, "the first optimization in this project
that does". ⇒ **the win should grow threaded.** This is the one term in the whole P0 that rule 24
predicts will pay in the configuration that ships.

---

## 5. H3 — the residency term, and the honest statement of what it is worth

This is the P0's stated prescription, kept because it is real, ranked third because of what it is
worth where it matters.

S43's masked-window instrument is **+71% at one thread** and **≤5% threaded** (54.291 → 51.788 ms,
under this machine's 6% cross-build floor). `nc` blocking as built captures 18.7% of that at 1t and
**loses threaded at every width, monotonically** — because it pays a `c/nc` A re-pack.

`plan-s43-nc-blocking.md` §8.2 names the fix and the evidence selects it: *"§7's `mc` rung is the
design that removes that price."* Concretely, the re-pack disappears if `pack_a` is hoisted out of
the `jc` loop by packing **all** of A once — the same one-shot parallel pack S43 §4d already built
for B, applied to the other operand.

**Priced from the measured sweep, not predicted fresh:** at nc=1024, 1t is 144.16 ms against 171.07
off, while paying `c/nc = 4` A-pack sweeps at ~8.5 ms. Removing three of the four projects
≈118 ms ⇒ **≈1.45× at 1 thread**, close to the 1.71× residency ceiling. Threaded, S43 says ≤5% and
rule 24 says a shared bottleneck shrinks. **Cost:** a second full-size packed buffer (64 MB at
N=4096) — a real memory price that must be stated in any verdict.

⇒ **H3 is a 1-thread result.** Worth taking for the published 1t number and for the i9, and it must
not be sold as a threaded win.

---

## 6. Order, and why

| # | item | cost | 1t value | threaded value (rule 24) |
| --- | --- | --- | --- | --- |
| **1** | H1 test — `--no-pack` A/B sweep | **zero code**, one run | decides d ≤ 512 | per-core ⇒ holds/grows |
| **2** | H2 test — `packcost.c` blocked/padded arms | ~40 lines of C | decides d ≥ 2048 | **per-core ⇒ grows** |
| 3 | ship whichever confirmed: pack threshold (H1) / `move_panel` on the A pack (H2) | emitter, both small | | |
| 4 | H3 — pack A once, then `jc` | larger; 64 MB buffer | ≈1.45× | ≤5%, shrinks |

Two measurements before any emitter change, both on instruments that already exist, both with
thresholds written down before the first number is read. That ordering is measurement rule 17 (the
simplest experiment available is run **first**, not last) and rule 25 (price it on real shapes).

---

## 7. Gates — every one runs before a timing is read

| gate | how |
| --- | --- |
| values bit-identical, every arm, every cell | before any timing; the NN harnesses already hard-error on an emission with no `mapal_par_task` |
| `@mapal_sme_panel` unmoved | byte-identical in `.ll` **and** in the `.s` (S43's gate, 54/54 instructions) |
| emission byte-identity off the changed path | `benches/emit_sweep_ab.sh … --target=generic` — 0 of 174 cells may move |
| **the instrument can fail** | inject the bug, watch the gate go red, revert (rule 23 — this repo has shipped two silent-pass gates) |
| a void arm is loud | any arm whose emission equals the control's is reported as void, never as "no effect" |
| full gate | `cargo test --workspace --release` = 1056 / 0, `cargo fmt --all --check` |

---

## 8. What this plan does NOT claim

- **Nothing here closes the i9 gap.** It is flat 0.87–0.96×, a NEON/AVX2 micro-kernel quality
  question against OpenBLAS, on a machine that has been unreachable since mid-S49. Out of scope,
  and it stays open.
- **No threaded headline is promised.** The threaded configuration at d=2048 already runs
  ≈3374 GF/s against a measured ~4100 aggregate ceiling — **82% of the machine** — so there is at
  most ≈1.22× left threaded on this part, ever. H1 and H2 are claimed to grow threaded by rule 24;
  the magnitude is bounded by that number and the plan says so before measuring.
- **No constant is fitted to one end.** A threshold ships only if a crossover is measured at both
  ends of the sweep.

---

## 9. Reconciliation — what the code became, and what the machine said

*Written after the build. Numbers and gates: `performance/s51-the-pack-is-o-n-squared.md`;
samples in `benches/results-s51/`.*

### 9.1 Where the plan held

| § | planned | built |
| --- | --- | --- |
| §3 | H1 tested with `--no-pack`, zero new emitter code | exactly that; the harness (`benches/nn/pack_ab.sh`) is the only new artifact |
| §3 | thresholds pre-registered, crossover required at BOTH ends | all four PASS, evaluated by the reporter rather than eyeballed |
| §3 | void arms loud | implemented and never fired — the four `@task*_pack` fns vanish |
| §3 | byte-identical control arm in every cell | 0.990–1.071×, all overlapping |
| §3 | ship a threshold in S50's shape, from measured profile fields | shipped as `working_set > l2_bytes` — **one existing field, no new constant** |
| §3 | rule 24: per-core ⇒ holds or grows threaded | **grew** — d=256 went 1.20× (1t) → 1.46× (threaded) |
| §7 | `generic` unmoved | **0 of 174 cells** |
| §7 | the instrument must be able to fail | injected a bug into `packconflict.c`'s blocked arm, watched the value gate go red, reverted |
| §7 | full gate | **1056 / 0**, fmt clean — the S50 count exactly |

### 9.2 What the plan got materially wrong

**(a) §4's whole cost model for the A pack was priced from a stale instrument.**
`packcost.c` advertises that it replicates the emitted nest; S42 reordered that nest
row-outer and the comment was never updated. §4's "254 GB/s of line traffic", the pressure
table, and the projected 1.47× at d=512 were all computed against **k-outer/row-inner code
that is not in the binary**. Withdrawn. The shipping pack is 4.9× cheaper than that number
at N=512, and the A pack is 12–17% of the wall rather than 60%.

**(b) S44's `pressure` predictor does not explain the pack.** Applied to the stale nest it
predicted no win at N=512 (pressure 0.5) and the treatment delivered 2.03×. Against the real
nest the governing quantity turned out to be far simpler — whether the **output** buffer
`ap = 32·k·sizeof` fits L1D — and that one predicted the corrected sweep exactly, including
its flat lower half. No mechanism was retrofitted to the refuted model.

**(c) The threshold's scope was too wide on the first build, and a test caught it.**
`profile.sme.is_some()` is not "this site takes the SME rung": an f64 or exact-face site on
`apple-m4-sme` runs the NEON packing rung, which nobody has measured.
`sme_profile_differs_from_apple_m_only_in_dispatch_kind_where_the_rung_declines` failed,
correctly, and the fix was to split `sme_shape` (clauses 1–7) out of `sme_tile_site` so the
threshold can ask the shape question without the clause-8 cycle.

**(d) H2's ordering premise was wrong in a useful direction.** §6 ranked H2 second on the
strength of the stale pricing. Re-priced, it is worth ~5–8% at d ≥ 1024 and nothing below —
so H1 was not merely first, it was the whole result.

### 9.3 What did not need doing

§6 item 3 planned a re-measurement of the shipped emitter. It was replaced by two `cmp`s:
the new default is byte-identical to the measured `--no-pack` arm at d ≤ 1024 and to the old
default at d = 2048. Byte-identity is the stronger claim and costs nothing (S39/S40's
precedent — no perf run is owed when no byte moved).

### 9.4 Still open, unchanged by this session

H3 (the residency term — pack A once, then `jc`) is untouched and remains a **1-thread**
result worth ≈1.45× at 1t and ≤5% threaded. H2 is measured, scoped, and unbuilt. Neither
touches the i9.
