# 2026-08-10 — S51: the pack is O(N²)

Date: 2026-08-10 · by: Claude (category-architect) · machine: Apple M4 Pro (local).
Baseline `c78c9af`, clean. Plan: `components/backend-llvm/plans/plan-s51-the-one-thread-kernel.md`.
Perf: `performance/s51-the-pack-is-o-n-squared.md`. Samples: `benches/results-s51/`.

## 0. Continuation brief

**Current state:** S51's P0 was split into three terms before anything was built, the smallest
one turned out to be the whole result, and it shipped. The B pack is now conditional on size
(`working_set > l2_bytes`), worth **1.13–1.52× at 1 thread and 1.23–1.51× threaded** across
d=128–512, with `--target=generic` unmoved and the gate at 1056/0. The A pack's traversal win
is measured and scoped but **not built**. Nothing is running; nothing is uncommitted except
this session's own work.

**Next step:** ship H2 (block the A pack's traversal, gated on `32·k·sizeof > l1d_bytes`) —
~5–8% at d≥1024, all the measurement already taken — or take H3 (the residency term) knowing
it is a 1-thread result.

**Resume command/check:** `bash benches/perflock.sh ./target/tmp/packconflict` reproduces H2's
table; `cargo test --workspace --release` is the gate.

## 1. What was asked vs what was found

Sapir chose the P0 from `next-session.md` §1: *"single-thread kernel is 0.28–0.35× of NumPy
below d=512 … reuse-structure change, not another blocking level."*

**The number and the prescription come from different size regimes.** The prescription is
S43's operand-residency finding, measured at **N=4096**. The number is the NN benchmark at
**d ≤ 512**, where the whole working set is ≤3 MB — inside a 16 MB shared L2, and S43's own
`loadlevel.c` measured **no cliff there at all** (flat ~1990 GF/s from 32 KB to 8 MB). So
residency cannot be the term the P0's own number names.

Split into three, each with a falsifiable test and a pre-registered threshold:

| | regime | term | outcome |
| --- | --- | --- | --- |
| H1 | d ≤ 512 | the packs are O(N²) against an O(N³) matmul | **confirmed, shipped** |
| H2 | d ≥ 1024 | the A pack's line efficiency | treatment confirmed, ~5–8%, **not built** |
| H3 | d ≥ 1024 | residency (the handoff's own prescription) | untouched; ≤5% threaded by S43 |

## 2. Decisions

| Decision | Verdict | Why |
| --- | --- | --- |
| Split the P0 before building | kept | Its prescription cannot bind at the sizes its number names; building it first would have optimised a term that is not there |
| `working_set > l2_bytes` as the pack rule | kept | The measured crossover (12.6 MB) lands on S43's independently measured 8–12 MB knee. An existing field; **no new constant** |
| `k·c·sizeof > l2_bytes` (B alone) | **not excluded, unselected** | Right at all five measured sizes too — the data does not separate them. Chose the working set because A and the output really do compete for the same L2 |
| Scope the threshold to `profile.sme.is_some()` | **discarded — a test refuted it** | An f64/exact-face site on `apple-m4-sme` runs the NEON rung, whose crossover nobody measured. Now scoped to sites taking the SME rung |
| Re-measure the shipped emitter | **unnecessary** | New default is byte-identical to the measured `--no-pack` arm at d≤1024 and to the old default at d=2048. Byte-identity is the stronger claim (S39/S40 precedent) |
| Ship H2 in the same session | **deferred, deliberately** | Re-priced against the real nest it is ~5–8% at d≥1024 only. Measured and scoped rather than half-built |
| Fix `packcost.c`'s stale loops | **discarded** | Its verdict is cited in three docs and a number's provenance includes the code that produced it (ADR-0037 D3). Marked stale in place; `packconflict.c` measures what ships |
| Retrofit a mechanism to the refuted `pressure` model | discarded | The treatment is confirmed; the model is not. Recorded as such |

## 3. Tests, checks, benchmarks

| Check | Result | What it proved |
| --- | --- | --- |
| `cargo test --workspace --release` | **1056 passed / 0 failed** | the S50 count exactly — nothing lost |
| `cargo fmt --all --check` | clean | — |
| `emit_sweep_ab.sh … --target=generic` | **0 of 174 cells moved** | no unmeasured part is made worse |
| value gate, 40 legs (2 shapes × 5 sizes × 2 arms × 2 widths) | **bit-identical** | the pack is a layout, not arithmetic |
| byte-identical control arm, every cell | 0.990–1.071×, all overlapping | this run's own noise floor |
| `cmp` new default vs old `--no-pack`, d ≤ 1024 | **identical** | the measured arm IS the shipped emission |
| `cmp` new default vs old default, d = 2048 | **identical** | the pack's 1.53× win above the knee cannot have been lost |
| injected-bug check on `packconflict.c`'s value gate | went red, then reverted | rule 23 — a gate that cannot fail is not a gate |

**H1, the shipped result** (`nn4 / nn2x2`, OFF/ON, >1 means the pack is a loss):

| d | working set | 1 thread | sep | threaded | sep |
| ---: | ---: | ---: | --- | ---: | --- |
| 128 | 0.2 MB | **1.457 / 1.517×** | DISJOINT | 1.507 / 1.451× | overlap |
| 256 | 0.8 MB | 1.199 / 1.213× | DISJOINT | **1.457 / 1.412×** | DISJOINT |
| 512 | 3.1 MB | 1.127 / 1.125× | DISJOINT | 1.317 / 1.233× | DISJOINT |
| 1024 | 12.6 MB | 0.992 / 1.000× | overlap | 1.056 / 0.996× | mixed |
| 2048 | 50 MB | 0.651 / 0.658× | DISJOINT | 0.674 / 0.662× | DISJOINT |

**H2, against the nest that actually ships** (`packconflict.c`, best of 7):

| N | `ap` KB | ctl ms | blk/ctl | old/ctl | drift |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 16 | 0.002 | 1.000× | 0.500× | 1.000× |
| 512 | 64 | 0.041 | 1.051× | 0.612× | 1.000× |
| 1024 | **128 = L1D** | 0.274 | 1.612× | 1.046× | **1.507× VOID** |
| 2048 | 256 | 2.075 | **2.752×** | 1.455× | 1.000× |
| 4096 | 512 | 9.087 | 1.771× | 1.140× | 1.007× |

## 4. Live handoff state

| Type | Handle | State | Inspect | Cleanup |
| --- | --- | --- | --- | --- |
| branch | `main` @ `c78c9af` + **uncommitted S51 work** | in sync with origin; this session's changes are unstaged | `git status -sb` | Sapir's call |
| machine | arch box `100.81.226.103` | **still DOWN** — ssh timed out again at S51 open | `ssh -o ConnectTimeout=8 100.81.226.103 nproc` | — |
| artifact | `benches/results-s51/` | untracked, the H1 samples | `ls benches/results-s51` | keep |
| artifact | `target/tmp/{emit_pre,packconflict,pack_ab}` | scratch build outputs | — | discardable |
| worktree ×12 | `.claude/worktrees/agent-*` + 3 pre-rename | prunable, unchanged since S47 | `git worktree list` | `git worktree prune` |
| file | `oainotes.md` | untracked, deliberately | — | Sapir's call |

**Nothing is running.** No background job, no server, no port. Measurement mutex free.

## 5. Open items

| Priority | Item | Reference | Next action | Done when |
| --- | --- | --- | --- | --- |
| **P0** | H2 — block the A pack's traversal | `s51` §5, `packconflict.c` | 32×32 tile in `emit_tiled_map_sme`, gated `32·k·sizeof > l1d_bytes` | d≥1024 improves, d≤512 byte-identical |
| **P0** | `MAPAL_PAR` semantics | S50 §3.5 | Sapir leans **worker count + an `auto` mode**; undecided | decided |
| P1 | H3 — residency: pack A once, then `jc` | `s51` §0, `plan-s43-nc-blocking.md` §7 | the `mc` rung | 1t ≈1.45×; ≤5% threaded expected |
| P1 | S42's row-outer reorder regresses 1.455× at N=2048 | `s51` §5c | subsumed by H2 — block, do not revert | H2 lands |
| P1 | `raptorlake` owes 3 dispatch numbers | S50 | run the two instruments on the box | **blocked — box down** |
| P1 | `path_plan_is_acyclic` still load-bearing | `runtime/suggestions.md` #4 | enforce the DAG contract or make the demotion loud | fallback deletable |
| P1 | sequential emission has no clock fence | S50 §5.4 | fence, or refuse to self-time a sequential program | honest bracket |
| P1 | `pack_pays` is unmeasured on the NEON rung | `s51` §4 | measure the crossover on the NEON kernel, or leave scoped | either |
| P2 | ARCHITECTURE §4 rules unpinned | `runtime/suggestions.md` #3 | one test per rule | 6 pins |
| P2 | `examples/vector.mapal` does not parse (3 of 174 cells) | S43/S48 | — | — |
| P2 | 12 prunable worktrees · box scratch · `kc_nest` · SME value check | S48 | — | — |

## 6. Architecture / model changes

One new morphism and one split, both in `backend-llvm`:

- **`pack_pays : (rows, c, k, elem) → 𝔹`** (`profile.rs`) — the size half of the packing
  decision. It reifies the fact that `pack_b` is a **`Trm` price, not a `Trn` price**: it
  computes nothing and exists only to create a second `DataLoc` over `b` at a friendlier
  `Loc` (FRAMEWORK §4.2). No coherence law changes; what changes is that the placement now
  has a cost model instead of being unconditional.
- **`packing_site` gains `(profile, contract)`** — shape *and* size in one predicate, read by
  six call sites. FRAMEWORK §5: a site packed by one caller and not another is a kernel
  reading a layout nobody wrote.
- **`sme_shape` split out of `sme_tile_site`** (clauses 1–7) to break a genuine cycle: clause
  8 asks `packing_site`, and `packing_site` now needs to know whether the SME rung takes the
  site. Residue documented rather than hidden.

**No `mapal-ir` change** — ADR-0032 held; the whole diff is `backends/llvm` plus benches and
docs.

## 7. Measurement rules earned

> **28. An instrument that names the code it replicates must be re-read against that code,
> not trusted for its comment.** `benches/sme/packcost.c` states it replicates the emitted A
> pack; S42 reordered that pack row-outer one session later and the comment was never
> updated. Three documents cite its number, and this session's own first pricing of the A
> pack was written from it and had to be withdrawn — the real pack is 4.9× cheaper at N=512.

Rules 1–27 stand. The ones that earned most here: **22** (a sweep needs a control arm that
should NOT move — the byte-identical control arm is what made every ratio in §3 readable),
**23** (a gate that cannot fail is not a gate — the injected-bug check), and **24** (classify
what an optimization removes and its thread-count behaviour follows — pre-registered, and
the win grew threaded exactly as the per-core classification predicted).

## 8. Docs reconciled

| Doc | Change |
| --- | --- |
| `components/backend-llvm/plans/plan-s51-the-one-thread-kernel.md` | new — the plan, plus §9 reconciling it against what the machine said (including four things it got wrong) |
| `performance/s51-the-pack-is-o-n-squared.md` | new — the full record, both hypotheses, every gate |
| `components/backend-llvm/IMPLEMENTATION.md` | new S51 section — `pack_pays`, `packing_site`, `sme_shape` → `file:symbol` |
| `components/backend-llvm/STATUS.md` | header rolled to S51 |
| `docs/STATUS.md` | S51 roll-up |
| `docs/next-session.md` | rewritten for S52 |
| `benches/sme/packcost.c` | marked STALE in place, with why it is kept rather than fixed |

## 9. Files changed

Code: `backends/llvm/src/{profile.rs, func/mod.rs, func/sme.rs, func/core.rs, func/bulk.rs,
func/frame.rs, func/drive.rs, lib.rs}`.
Tests: `backends/llvm/tests/golden_ll.rs` (one locator repointed off a packing artifact).
New instruments: `benches/nn/{pack_ab.sh, pack_report.py}`, `benches/sme/packconflict.c`.
Annotated: `benches/sme/packcost.c`.
New docs: the five above. Samples: `benches/results-s51/`.
