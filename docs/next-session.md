# Next Session (S52)

Written: 2026-08-10 · end of S51 · by: Claude (category-architect)
Session log: `sessions/2026-08-10-s51-the-pack-is-o-n-squared.md` — **read it**
Records that govern S52: `performance/s51-the-pack-is-o-n-squared.md` and
`components/backend-llvm/plans/plan-s51-the-one-thread-kernel.md` **§9** (what the plan got
wrong, so S52 does not repeat it).
Previous handoff: S49–S50, `sessions/2026-08-10-s49-s50-the-scheduler-was-the-kernel.md`.

## READ THIS FIRST

**S51's work is UNCOMMITTED on `main` @ `c78c9af`**, which is in sync with origin. S49/S50 and
S48 all landed together in `c78c9af` and were pushed, so that P0 is closed; the tree now holds
only S51's diff plus the deliberate `oainotes.md`.

**A published instrument was measuring code that is not in the binary, and it cost this session
a wrong projection.** `benches/sme/packcost.c` says it *"replicates the EMITTED nest exactly"*
and replicates the **pre-S42** A pack — S42 reordered that pack row-outer and the comment was
never updated. Its 8.46 ms is a pre-S42 number; three documents cite it. It is marked STALE in
place (its provenance is worth more than a fix) and `benches/sme/packconflict.c` measures what
ships. **Measurement rule 28** was added for this.

**`pack_pays` is scoped to the SME rung ON PURPOSE.** The crossover was measured on that kernel
and no other, so an f64 or exact-face site on `apple-m4-sme` still packs unconditionally through
the NEON rung. That scoping is not an oversight and a test enforces it
(`sme_profile_differs_from_apple_m_only_in_dispatch_kind_where_the_rung_declines`, which caught
the first, wider version).

**`MAPAL_PAR` is still undecided.** Sapir's position at S51 open: *"I think it should be worker
count, and also an `auto` version for optimal detection — I'm confused about this and need to
think about it."* It ships today as **total executors including the host**. Nothing was changed.

**The arch box `100.81.226.103` is INTERMITTENT.** Sapir enabled Tailscale at S51b; it came up,
ran the whole NN ladder, then went unreachable again before the constants run (4 retries over
60 s, all timeout). `raptorlake` now has **`flops_per_ns = 144` measured** and still owes
`bytes_per_ns` and `dispatch_ns` — see `sessions/2026-08-10-s51b-the-i9-leg.md`.

**A RENTED STAND-IN WAS TRIED FOR THE TWO OWED CONSTANTS AND REFUSED ITS OWN CONTROL.**
S51c rented an i9-14900K on vast.ai (whole machine, 32/32 effective, $0.09 total, destroyed).
`flops_per_ns` is the built-in control: the arch box reads **144.2 / 143.4** at d=1024/2048,
agreeing to 0.6%; the rental read **149.2 / 157.7**, disagreeing with *itself* by 5.7%. Cause
found, not guessed: the host runs the **`powersave` governor** (load 0.26, so not neighbours),
the artifact S33/S36b already recorded for this CPU class. The target constant came out
**13.0 µs from medians and 1.5 µs from minima — 9x apart** — because the signal
(dispatch − inline ≈ 0.1 ms) sits under ±0.6 ms of jitter, and a 41-cycle re-run moved the
medians 5x. **Nothing from that box was written into `raptorlake`.**

> **29. A timing constant may only be taken on a stand-in machine if the stand-in reproduces
> the target's own control.** `flops_per_ns` existed in this recipe as a cross-check and is what
> caught it; without it, a `dispatch_ns` off by 9x would have been written into the profile that
> compiles for the real box. And note what does NOT transfer even in principle: `bytes_per_ns`
> is a DRAM-configuration property and `dispatch_ns` is a runtime-and-scheduler property —
> neither is a property of "an i9-14900".

**If `nn4_128` looks that noisy on the arch box too, the recipe needs a bigger program.** At
0.18 ms it may be too small to carry a ~0.1 ms difference on a `powersave` machine; the fix is a
larger d or more flipped dispatches, not more cycles.

**THERE IS A MEASUREMENT SCRATCH PROFILE IN THE TREE THAT MUST NOT SHIP.**
`profile.rs:RAPTORLAKE_INLINE` is `raptorlake` with `dispatch_ns = 1_000_000_000`, so the
roofline answers "inline" for every site. It exists only because `dispatch_ns` can be measured
only as a **difference** (dispatch-everything vs inline-everything, over the tasks that flip).
It is reachable only by explicit `--target=raptorlake-inline` and shows up in the resolver's
name list, which is the tripwire. **Delete it the moment the constants land** — it is the
`kc_nest` mistake in miniature, and this repo already carries a P1 to delete that one.

**Threaded headroom on this part is bounded and small.** d=2048 threaded already runs
≈3374 GF/s against a measured ~4100 aggregate ceiling — **82% of the machine**. At most ≈1.22×
remains threaded on the M4, ever. Size any future threaded claim against that.

## S52 opens on — H2, which is measured and unbuilt

### 1. P0 — block the A pack's traversal

Everything needed is taken. The emitted pack (`func/sme.rs`, row-outer / k-inner since S42)
reads `a` contiguously and **stores `ap[pk·32 + pi]`, one 4 B store per 128 B line**. Whether
that costs anything is one question: does `ap = 32·k·sizeof` fit L1D (128 KB)?

| N | `ap` | blocked / ships | pre-S42 order / ships |
| ---: | ---: | ---: | ---: |
| 128–512 | ≤ 64 KB | 1.000–1.051× — **flat, nothing to win** | 0.50–0.61× |
| 1024 | 128 KB = L1D | 1.612× — **cell VOID**, drift 1.507× | 1.046× |
| 2048 | 256 KB | **2.752×** | **1.455× — the reorder REGRESSES here** |
| 4096 | 512 KB | 1.771× | 1.140× |

**Build:** a 32×32 tile over `(pi, pk)` in `emit_tiled_map_sme`, gated on
`32·k·sizeof > l1d_bytes` so it is a byte-identical no-op below the boundary. Value-identical
by construction (same stores, different order of the same loop nest). **Worth ~5–8% integrated
at d≥1024 and nothing below** — the A pack is 12–17% of the wall, not the 60% the stale
instrument implied.

**Do not "fix" this by reverting S42.** The pre-S42 order is faster at N≥2048 and ~2× slower at
N≤512; blocking beats both (1.89× over the pre-S42 order at N=2048).

**Rule 24, pre-registered:** a per-core cost ⇒ the win should hold or grow threaded.

### 2. P1 — the queue

- **H3, the residency term** (the S51 handoff's original prescription). Pack A once globally —
  the same one-shot parallel pack S43 §4d already built for B — then `jc` outside, which
  removes the `c/nc` re-pack that made `nc` lose. Priced from the measured sweep at **≈1.45× at
  one thread**; S43 measured residency at **≤5% threaded** and rule 24 says a shared bottleneck
  shrinks. Costs a second full-size packed buffer (64 MB at N=4096). **Take it for the 1t number
  and the i9, not as a threaded win.**
- **`pack_pays` on the NEON rung is unmeasured.** Either measure that crossover or leave it
  scoped; do not transplant the SME threshold on the strength of an argument.
- **`path_plan_is_acyclic` is still load-bearing** — a cyclic plan is still silently demoted to
  sequential emission, the failure mode that hid every chained matmul for nine sessions.
- **Sequential emission has no clock fence** — a program that falls back self-times garbage.
- Six `runtime/ARCHITECTURE.md` §4 composition rules have no automated pin.
- Carried and still open: `examples/vector.mapal` does not parse (3 of 174 sweep cells) ·
  `kc_nest` · executing SME value check in `cargo test` · 12 prunable worktrees.

### 3. What S51 proved about how to open a session

The P0 was handed over as one item and was **three**, and its own prescription fitted only the
regime its number did not name. Two things caught that before any code was written: reading
`FRAMEWORK.md` §4 and writing the placement out as `Trn`/`Loc`/`Trm` (which made the packs
visible as **`Trm` prices that no record had taken**), and reading the refuted-candidates list
in the prior records (which killed the streaming-mode hypothesis in one grep — `smcost.c`,
7.6 ns/call, 0.6% of the wall).

## FIRST commands

```sh
cargo test --workspace --release                  # 1056 passed / 0 failed
cargo fmt --all --check
git status --short                                # S51's diff, uncommitted
ssh -o ConnectTimeout=8 100.81.226.103 nproc      # INTERMITTENT
bash benches/nn/raptorlake_constants.sh 100.81.226.103 21   # the two owed constants

# reproduce S51's two measurements
CYCLES=15 bash benches/perflock.sh bash benches/nn/pack_ab.sh
clang -O2 -march=armv8-a -o /tmp/pc benches/sme/packconflict.c && bash benches/perflock.sh /tmp/pc

# the gate that must NOT move for unmeasured profiles
bash benches/emit_sweep_ab.sh target/release/examples/emit /tmp/post.hashes --target=generic
```

## Live state at close

| Type | Handle | State | Inspect | Cleanup |
| --- | --- | --- | --- | --- |
| branch | `main` @ `c78c9af` | in sync with origin; **S51 uncommitted** | `git status -sb` | Sapir's call |
| machine | arch box `100.81.226.103` | **INTERMITTENT** — ran the S51b ladder, then dropped | `ssh -o ConnectTimeout=8 … nproc` | — |
| **code debt** | `RAPTORLAKE_INLINE` in `profile.rs` | **scratch, must not ship** | the resolver's name list | delete after the constants |
| artifact | `benches/results-s51-i9/i9/` | 90 files, the i9 ladder | `ls benches/results-s51-i9/i9` | keep |
| artifact | `benches/results-s51/` | untracked, the H1 samples | `ls benches/results-s51` | keep |
| artifact | `target/tmp/{emit_pre,packconflict}` | scratch | — | discardable |
| worktree ×12 | `.claude/worktrees/agent-*` + 3 pre-rename | prunable, unchanged since S47 | `git worktree list` | `git worktree prune` |
| results | `benches/results-s49/`, `-s50final/` | untracked; **`-s50final` is what the README quotes** | `ls benches/results-s50final` | keep both |
| file | `oainotes.md` | untracked, deliberately | — | Sapir's call |

**Nothing is running.** No background job, no server, no port. Measurement mutex free.

## Open items

| Priority | Item | Reference | Next action | Done when |
| --- | --- | --- | --- | --- |
| **P0** | H2 — block the A pack traversal | §1 | 32×32 tile gated on `32·k·sizeof > l1d_bytes` | d≥1024 improves, d≤512 byte-identical |
| **P0** | `MAPAL_PAR` semantics | READ THIS FIRST | Sapir leans worker-count + `auto`; undecided | decided |
| P1 | H3 — pack A once, then `jc` | §2 | the `mc` rung | 1t ≈1.45× |
| P1 | `pack_pays` unmeasured on the NEON rung | §2 | measure or leave scoped | either |
| **P0** | delete `RAPTORLAKE_INLINE` scratch profile | S51b §2a | remove once the constants land | gone from `profile.rs` |
| **P0** | `raptorlake` owes `bytes_per_ns` + `dispatch_ns` (`flops_per_ns` = 144 is taken) | S51b §2 | `bash benches/nn/raptorlake_constants.sh 100.81.226.103 21` | profile carries `Some` |
| P1 | i9 scaling hole at d=1024 — 6.15× between 8.12× and 9.37×, overlap factor 0.730 | S51b §1d | diagnose | explained or recorded |
| P1 | `path_plan_is_acyclic` load-bearing | §2 | enforce or make loud | fallback deletable |
| P1 | sequential emission has no clock fence | §2 | fence, or refuse to self-time | honest bracket |
| P2 | ARCHITECTURE §4 rules unpinned | `runtime/suggestions.md` #3 | one test per rule | 6 pins |
| P2 | `vector.mapal` · `kc_nest` · SME value check · 12 worktrees · box scratch | S48/S43 | — | — |

## Standing direction (Sapir — unchanged)

- Compute-only legs; NumPy in every verdict table; scale everything up.
- Parallel-first by construction — **threaded is the configuration that ships.**
- Backend-genericity contract (ADR-0032): **mapal-ir never learns machine facts.** Held through
  S51 — its whole diff is `backends/llvm` plus benches and docs.
- **Keep stuff generic, not hardcoded** — and when a constant is swept rather than derived, say
  so. S51 added **no constant**: the pack threshold is `working_set > l2_bytes`, an existing
  recorded field.
- Query, not rewrite: record that something *could* be skipped; never delete it.
- Compile time decides the SIZES, runtime decides the ASSIGNMENT.
- Nothing goes in the README that a default build does not deliver.
- Proof over suggestion — a change arrives with the measurement of what it did.
- **The README is results, not method.** Tables and a pointer; explanations live in `docs/`.
- Speak simply, base claims on empirical results. **Give absolute ms, not only ratios.**
