# Current continuation — S57 generic proof work

The only checkout is `s54-gpu-fusion` at `b4152c5`. Preserve the ignored recovery archives
under `.claude/worktrees/_archive-s54-20261021/` and untracked `oainotes.md`. Do not push.
Most S57 reports/evidence are durable local files but not yet committed. The newest immutable checkpoint is
[S57 frozen validation complete](sessions/2026-10-21-s57-frozen-validation-complete.md).
The earlier20:45 checkpoint remains historical; this mutable continuation tracks later work.

## Completed frozen validation

All five primaries (RTX 3090, A100 SXM4, RTX 4090, H100 PCIe, RTX 5090) and both
independent RTX 4090/5090 confirmations pass complete numerical, timing/status, library,
resource and profiling gates. Every corresponding rental is independently absent.
The last 5090 confirmation was absent at 20:55:19 UTC. Independent checks cover
54,121 numeric fields and 27,900 raw kernel samples with zero differences.

- [Full measured tables](performance/s54-gpu-fusion.md): all 126 shipped kernel rows,
  all arms/losses/control CSVs, composed/decode comparisons and timing-boundary limits.
- [Selection decisions](performance/s54-per-arch-selection.md): 5090 streams OFF,
  H100 retention/window OFF, restored derived 5090 raster gate. No precision-policy change.
- [16K and rectangular extension](performance/s57-large-composed-evidence.md): 98/100
  invocations pass; both failures are the requested-TF32 small-M fallback. All 16K gates
  pass, but cuBLAS is faster. No application-wide BF16/FP8 or transformer claim.
- [Disassembly evidence](performance/s57-ptx-sass-gap.md): small-M scalar fallback and
  differing TF32 conversion/tie semantics are proved; a profitable replacement is not.
- [CPU/GPU call effects](../benches/results-s54-gpu/s57-host-optimization/host-gpu-call-effects-audit.md):
  host LLVM vectorization is present; device completion is not LLVM stack lifetime.
  Joint cost-based CPU/GPU placement and generic paged KV allocation are not implemented.

Final 5090 confirmation has lower stream-OFF medians in all eight changed cells, but two
ranges overlap. Its 4K TF32 Mapal/cuBLAS times are 4.4472/1.8883ms versus primary
1.5460/1.3906ms. Saved PTX, launch geometry and runtime resources match; the cause remains
unresolved. Decode allocations remain 8 versus 12,288, but ordinary wall is 5,046/14,480ms
versus primary 450/1,292ms. Never mix traced kernel duration with ordinary wall.
The primary 5090 Exact8192 identical-PTX null is unstable (34.64→48.37ms); preserve it.

All frozen results are archived with hash verification. The earlier five-bundle/four-primary
archive has 41,051 files/2,423,639,312 bytes. Separate immutable archives preserve H100,
4090 confirmation and final 5090 confirmation. The latter contains 21 files/20,057,698 bytes;
[receipt](../benches/results-s54-gpu/final-integrated-5090-confirmation-recovery.json).
Frozen host tests: 1,176 passed/0 failed/1 ignored; no production-source change since.

## Correctness pilot: one failed allocation, no GPU result

The user approved one 13.4MB correctness-only attempt: ordinary ordering first (200 fresh
processes), then small composed matrix verification against handwritten CUDA/cuBLAS and
explicit PyTorch2.13.0/cu130. Limits: $1 total, <=$0.80/hour, <=55 minutes, existing
$8/hour aggregate cap; equivalent hosts allowed before launch. No timing admission.
The reviewed adapter additionally installs hash-pinned pip25.3 because Jammy pip22.0.2
cannot produce the required `--report`; the two private archives are unchanged.
Seven adapter and eight launcher stub tests pass. Official image/wheel metadata accounts
for about7.89GB of single-copy transfers; actual installation remains untested.

The v5 launch on offer56695170, machine54112/host146915, returned `success=false` with
contract57110891. The wrapper exited after2.1s before SSH/upload/workload and destroyed
that ID. Independent absence is21:02:29 UTC. No ordering/matrix execution occurred;
provider image transfer and actual billing are not inferred from the logs. This consumed
the approved single attempt; **do not retry without additional user authorization**.
The failed attempt and approved preparation have a separate immutable 221-file,
29,423,503-byte archive; [receipt](../benches/results-s54-gpu/s57-correctness-pilot-v5-failed-recovery.json).
The user then approved one additional same-payload attempt under the original cumulative
$1 cap. It also failed at allocation before upload: offer51669617, machine143204/host556515,
contract57111469; exit1 after2.3s, independent absence21:07:38 UTC. Both attempts produced
no GPU results. No further attempt is authorized. Read-only startup diagnosis is in progress.

Operational base: `/private/tmp/mapal-s56-gpu`.

```sh
python3 /private/tmp/mapal-s56-gpu/final_status.py
cat /private/tmp/mapal-s56-gpu/integrated/results/correctness-pair-rtx-5090/verification.json
cat /private/tmp/mapal-s56-gpu/correctness-pair/staging-manifest-v5.json
```

At21:08 UTC there are zero active task rentals. Never touch unrelated instances
`57065574` or `57100404`. GPU agent owns rental/upload/cleanup. Preserve the frozen benchmark payloads/tools as evidence. Both bound launchers are now
closed; apply a source repair only after its separate proof/review. Isolated diagnostic
snapshots remain under the paths below. Do not run integrated timing acceptance on correctness-only output.
Keep numerical success, arithmetic-route proof and timing admission distinct.

## Next authorized work, in order

1. Frozen-report reconciliation is complete: new confirmation presentation checked,
   FRAMEWORK §4.5 review reconciled, drift zero-dead, strict validation passed and
   `verify-s54-gpu-selection` archived as `2026-10-21-verify-s54-gpu-selection`.
   Benchmark success does not close targeted ordering duties.
2. Diagnose the two pre-upload allocation failures before proposing any further ordinary
   ordering/matrix pilot. The approved attempts are exhausted; no GPU result exists.
   Payloads and exact validators remain under `ordering-runtime/`, `large-composed/`
   and `correctness-pair/` in the operational base. Timeout aborts all later phases.
3. Repair the recurrent checkpoint compiler failure through
   `repair-loop-route-replay`. Diagnostic snapshot:
   `/private/tmp/mapal-replay-diagnostic/source`, isolated build in sibling `target`.
   Both tiny674 and original5907 now identify an acyclic `(IoToken, Array)` exit payload
   inside the canonical loop driver cone but outside its SCC, left unmapped during
   round1 ConstFold replay. Temporary tracing establishes the exact cause; a generic
   repair/regression is being tested only in that snapshot. No shared source fix yet.
4. Apply `eliminate-unmaterialized-gpu-allocations` only after dependency proof. The
   [ordering evidence](performance/s57-ordering-evidence.md) establishes missing emitted
   edges for stored-member host readback and another-chain tile consumption. Same-stream
   and member-row-fold suspicions were refuted for inspected shapes. The planned touch map
   uses existing group ownership, alias canonicalization and output-storage facts.
   Preserve the arena guard. A nonzero free currently synchronizes; omitting an allocation
   can remove that incidental wait. Predicted allocation counts are not measured gains.
5. Continue `benchmark-large-composed-gpu-workloads`: full recurrent checkpoints after
   replay repair; matched numerical contracts and clocks across Mapal, naive/handwritten,
   cuBLAS and PyTorch eager/compiled; then retention/allocation controls across families.
   Matrix payload arithmetic is statically certified; no new GPU composition timing yet.

Proof rule: infer from shapes and dataflow, record a candidate, then prove emitted code and
runtime behavior before changing the compiler. Current `.rna` TF32 ties differ from the
undecided ties-even proposal; do not change numerical policy as a performance fix. Logical
last use differs from asynchronous completion. Static IR can build graph topology, while
buffer binding/instantiation and replay still need runtime work.

---

(S52's handoff, kept below for the CPU-side items it still carries. The S53 workflow header it replaced:
workflow `mapal-s53-gpu-optimizations`, run `wf_40012e53-58d`; item branches `s53/<key>`.)

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

## S52 opens on — H3, the last term. H2 SHIPPED in S51

### 0. H2 is DONE — `ac13439`

Blocked the A pack once `ap = panel_rows·kc·sizeof` leaves L1D. **`nn4_2048` 1 thread
74.0476 → 67.8328 ms, 1.092×, disjoint**; threaded 1.022×, overlap. Gate 1058/0, generic
unmoved, `nn4_{128,512,1024}` byte-identical.

**Its threaded prediction was WRONG and that is the lesson (rule 30).** The plan
pre-registered "per-core cost ⇒ holds or grows threaded" on rule 24, and it shrank
(1.09× → 1.02×). Rule 24 needs a second question: **is the work divisible?**
`move_panel` grew threaded because every core swept all of A whatever rows it owned;
each lane packs only its own panels, so this dilutes like any parallel term.

### 1. What H2 was (kept for the record)

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
| P1 | the reversed pack's 2.0–2.6× padding win is UNEXPLAINED; padded-reversed beats shipped blocking (3.154 vs 5.167 ms at N=4096) and **no traversal reaches it** — a 24-cell tile sweep got to 0.684× | `s51` §5f, §5g | the read-set SIZE is refuted (R=4 ≡ R=32); only the stride VALUE matters. Untested candidate: page/DRAM-bank aliasing — at N=4096 the stride is exactly the 16 KB page | mechanism identified, or dropped |
| **P0** | `MAPAL_PAR` semantics | READ THIS FIRST | Sapir leans worker-count + `auto`; undecided | decided |
| **P0** | H3 — pack A once, then `jc` — the last term | §2 | the `mc` rung | 1t ≈1.45× |
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
