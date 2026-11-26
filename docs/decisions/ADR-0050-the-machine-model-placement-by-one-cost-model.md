# ADR-0050: The machine model — hardware as abstract compute and placement facts; the dataflow placed on it by one cost model

Date: 2026-11-26 · Status: **accepted** (decided by the user). Builds on: the facts catalog (CLAUDE.md), "hardware is
physical properties" (2026-11-20), "a mixed result is a missing fact" (2026-10-26), ADR-0040 / 0041 / 0043 (numerics
admission), ADR-0049 (correct = flow + bound). Supersedes: ADR-0034's search by measurement as the source of placement
constants (measurement stays, as the model's check). Replaces, once step 1 is confirmed: the per-decision placement
rules (`.claude/sprints/plan-cost/design.md` §1) and the plan-cost model `site_ns` as a separate rung.

Design check (2026-11-26, two independent reviews; full text `.claude/sprints/machine-model/design-check.json`): the
first draft's shape held; its places were a tree, its schedule levels were fixed to executor levels, its operations had
no latency or layout, its cache reuse counted requested bytes, block start timing was unstated, and it did not cite
plan-cost. This text includes their amendments.

## Context

The GPU backend decides placement in about seventeen separate decisions — placement, kernel kind, tile shape, staging,
combine and split, raster order, the walk, epilogue and kernel boundaries, the zero pass, residency, concurrency — each
by its own rule, in a fixed order, each reading the subset of the hardware facts it was written for
(`rung_replay.tsv`: 319 measurements over about 50 rungs). The tile shape is a ladder of local rules with built-in
assumptions: two resident blocks an SM (`GPU_RESIDENT_BLOCKS`), half the register file, powers of two, square-most.
Each instruction family has its own geometry chooser. Several `DeviceCost` values are fits of particular Mapal kernels,
some rates come from tables keyed by compute capability, and parts without a measurement borrow the RTX 5080's.

The failure this produces: `mma-walk` (2026-11-25) priced how full the SMs were, not where the operands live while the
walk scatters their readers. The L2 fact existed (`l2_keep_bytes`) but was outside that rule's view, and the walk lost
1.2× on the RTX 3090 and the A100. The fix added the fact to that one rule (`MmaWalk::walk_bytes`, b3ad75f4). The
design check then showed that rule counts *requested* bytes between two reads where a cache evicts on *distinct*
bytes: an exact LRU simulation agrees on the 3090 and A100 (956 / 1074 MB) but not near the cut-off (the RTX 5080's
3072 × 4096 × 8192: 1685 MB requested, 124 MB distinct — that walk was refused for the wrong reason). None of these
byte counts was measured (no profiler counters on the rented boxes); they are the model's. Each decision patched alone
keeps producing this.

Prior art: plan-cost (`site_ns`) was already one cost model for the scalar tile site. It measured mixed (it lost 2.7×
on `mm_1x1024x4096` on the RTX 4090) and ships off. This ADR must win back its losers, naming the term it lacked.

The user's direction (2026-11-26): the compiler holds two kinds of facts — the dataflow's and the hardware's; each
hardware becomes **abstract facts about compute and placement**; the dataflow is **mapped onto** those facts, and only
then goes to the hardware's language; "choosing from a table of values, or hardcoding is not the way"; hardware-specific
optimizations that cannot become abstract facts are a separate, later step, kept to a minimum, "so the abstract model is
as complete as possible"; adding a backend or a part is defining how it translates into the model.

## Decision

**D1 — Two fact layers, one mapping.** Compilation places the dataflow (the facts of `mapal-ir`, unchanged) on a
**system** — machines joined by links — by one evaluator and one search, producing a **schedule**; the backend then
lowers the schedule into its language. A target is defined by its translation into machine facts plus its lowering.
No placement decision reads a target's name, kind or instruction-set version — only machine facts.

**D2 — The machine.** Each target defines the depth of its own trees; nothing below assumes a GPU.

- **Places** (where bytes can live) are a **graph of instances** (a tree is the common case). A place has a capacity, a
  scope (the executors that reach it, by reach edges carrying bandwidth, latency and transaction granule), banking
  (bank count, width, broadcast rule) and a management: *explicit* (a scratchpad the program addresses — registers,
  shared memory, tensor memory, an ASIC's SRAM) or *cache*. A cache lists its instances (L2 partitions, per-XCD L2s,
  NUMA nodes), the map from executor or address to instance, its replication rule (a line read through two instances is
  held in both), and its measured **hit-against-distance curve** (its `keep` for a cyclic, a rotated and a strided
  pattern — never the nominal size). **Capacity groups** whose split a schedule chooses (shared memory against L1, a
  unified register file's two kinds, a two-unit sharing mode) are places too.
- **Executors** (what computes) form a tree of the target's depth: device → units → sub-partitions → groups → lanes,
  units possibly of different kinds, with modes and their switch cost. **Operations are rows of geometry**, owned by any
  executor level (a group, a unit, a unit pair, a cluster, a unit shared by cores): shape as a set (m, the n set, k,
  batch); operand, auxiliary-operand (scales) and accumulator formats; each one's **home place**; each operand's allowed
  layouts and the fragment map (lane → element); the fragment footprint per lane; latency and async depth (groups in
  flight before a wait); the measured rate; the measured **efficiency curve** (against group tile and resident groups a
  scheduler); the group widths it issues at; the accumulator's numerics (ADR-0043). The FMA lane is the `1 × 1 × 1` row;
  `mma.sync`, `wgmma`, `tcgen05` (accumulator in tensor memory, a unit-pair form), an MFMA, a CPU vector FMA, an SME outer
  product, a systolic array are rows of the same table. Issue rates per instruction class (math, global, shared,
  barrier) are unit facts.
- **Movers** (how bytes move): program-issued (lane copies, bulk-copy engines with their box shapes, swizzles and
  multicast, their async depth), hardware-issued (prefetch streams: the run that locks them on, their reach), and
  **links** between machines (host ↔ device: rate by source place, pinned or pageable, copy engines). Per-access cache
  policies (evict first, persisting windows) are mover options.
- **Dispatch** (how work reaches executors): a unit's resources as capacities — **occupancy is derived from a
  schedule's footprint, never a constant**; the dispatch map from grid index to unit (round-robin across partitions);
  the **start timing** of a grid's blocks (lockstep or staggered — measured); launch modes (plain, dependent overlap,
  cooperative, cluster, dynamic queue) and their costs; persistence and the jitter of a static round-robin; clusters;
  barriers; atomics as a resource at the place that resolves them; the zero pass; the sustained clock under load.

**Provenance.** Every fact carries its source — *measured*, *queried*, *declared* (by a design: spec, RTL parameters,
simulation), *derived* or *nominal* — and an interval. Facts fall in three categories: **machine facts** (above);
**lowering-declared counts** (a schedule's register footprint, instructions per class per trip, in-flight discipline of
its ring, its constraints — checked against `ptxas -v` and SASS, D8); **fits** (a curve of one Mapal kernel) — never
read by the evaluator, kept only as replay lines until the evaluator reproduces them, then deleted. No value comes from a
table by version or from another part. A missing fact takes a marked derived or nominal interval, or removes the
candidates that need it — visibly.

The translation `Machine::of(target)` is the only target-specific code placement depends on (the GPU's from
`TargetProfile`; `profile::Machine`, the kind discriminator, leaves placement).

**D3 — A schedule.** For a site (first: sum-of-products tile sites; then every cell, fold and streamed unit):

- the **executor subtree** it runs on (host, device, both);
- the **operation** and the **operand roles** (which input is `a`, which `b`: the transposed product is a role swap);
- the **levels**: any number, each a loop extent bound either to an executor level (parallel) or to a place
  (sequential blocking — a CPU's register tile, L1 / L2 panels, thread split);
- each operand's **home, layout (pad, swizzle) and buffering depth** at each level; the accumulator's buffering;
  **packing** (a layout-changing copy into a resident buffer) as a scheduled mover priced by its bytes;
- **roles inside a slot** (producer and consumer groups, each one's register share within the reallocation limits);
- a **cluster tile** with multicast per operand;
- the **traversal** — ordered with a raster (group height `h`); split (`S`, atomic or workspace); walked (stream-K over
  every slot, with or without a raster); hybrid (whole waves ordered, the tail walked); persistent;
- the launch mode, the cache policy per access, the capacity-group splits;
- **warm operands** (resident from the previous call: `residence_plan`) as inputs; later steps add the **co-schedule**
  (independent sites sharing the system in one phase) and **residence** across the graph.

**Feasibility** = capacity (every place's footprint fits its share at its level) ∧ numerics admission (ADR-0041 / 0043:
an operation and its pieces — 1, `S`, or unbounded for a walk — admitted by the site's bound; the fold order of
ADR-0040; window and mixed-radix sites never cut `k`) ∧ the lowering admits it (D6).

**D4 — One evaluator for every schedule.** Predicted time = over the schedule's phases (waves, a walk, a tail, a
prologue), the maximum over resources of work ÷ rate, summed, plus fixed costs:

- **compute** — the busiest unit's work (`⌈tiles / slots⌉`: waves and tails) at the operation's measured rate and
  efficiency, at the sustained clock;
- **movement** — bytes across each place boundary ÷ its bandwidth. Into explicit places the bytes follow from the
  levels exactly. Out of a cache instance they follow from **reuse distance**: for every slice, the *distinct* bytes the
  instance sees between two reads of it — every stream through it: operands, output stores, partial sums, the zero
  pass, workspaces, epilogue operands — through the dispatch map and the start timing, against the hit curve. Closed
  forms per traversal (interval unions, `O(blocks)`; the walk's reader gaps by the three-gap theorem) are checked against
  an exact trace simulation of the cache (arithmetic); calibration probes check the hardware;
- **latency** — the bytes in flight the lowering's ring keeps per unit against the source's bandwidth × latency;
- **issue** — the lowering-declared instruction counts per class against the unit's issue rates;
- **combine** — atomics as a resource at their place, overlapped while their working set stays within the keep, a DRAM
  path when it does not; workspace bytes; the zero pass, priced once; a per-piece prologue (pipeline fill) for every
  split or walked piece;
- **fixed** — launches and gaps (plan-cost D2), barriers, grid syncs, link transfers.

The present per-rule prices (`site_ns`, `stream_ns`, `mma_pieces`, `streamk_priced`, `raster` / `b_rereads`, the walk's
price) become terms of this evaluator or are deleted. A new measurement is a replay of the evaluator, never a rule.

**D5 — The search.** Candidates are generated from the facts, bottom-up through the executor and place trees, never
from a table: (1) operations the system lists that the site's precision set admits, with their roles; (2) group tiles
whose declared footprint fits the operation's home share; (3) slot tiles, `bk`, depths and layouts within a slot's
capacities, for every feasible occupancy; (4) traversals, launch modes and policies with their parameters. Branch and
bound with admissible lower bounds (peak compute; every operand read once) and dominance pruning keep it tractable. The
search keeps its top candidates through a **post-lowering gate** (the code generator's real registers and spills: a
spill, or registers above the declared cap, makes a candidate infeasible and the next one is taken). Facts with
intervals are evaluated at their bounds (minimax); a pick that flips inside the intervals is marked open. The model's
**error band** is a fact (per-term replay residuals): a schedule replaces the incumbent, or a simpler one, only beyond
it — so a re-measured rate does not churn emission. Where a run-time confirmation is kept, it chooses only among the top
candidates and only on pairs that differ beyond the null.

**D6 — Lowering after placement; target-specific steps after lowering, kept minimal.** A lowering emits any feasible
schedule of the kinds it declares, and declares its constraints (they narrow the space visibly), its counts and its
footprints. Layout (pad, swizzle span) is in the model — its banking cost is a fact; only the bit encoding is lowering.
What cannot be a machine fact — a code generator's register-allocation quirk, a driver fault's workaround, an index-width
or unroll flag — is a separate step after lowering, keyed by properties, listed in `docs/target-specific.md` with why it
is not a fact yet and what would make it one. Step 1 records that list's starting size; it must shrink over time and
must not grow with each new target.

**D7 — New targets, and synthesis.** A new target is its translation and its lowering — AMD (MFMA, LDS, 64-lane
waves, Infinity Cache, XCDs), CPUs (cores, vector units, cache levels, prefetchers, SME), ASICs (PE arrays, SRAM banks,
a NoC). A **synthesis** target (the ASIC lab) is a machine template with free parameters (array dimensions, SRAM sizes
and banks, NoC width) and a resource budget (area, DSPs, BRAM, LUTs, fmax): the search ranges over the template's
parameters as well as the schedule, with facts declared by the design.

**D8 — Evidence.** The CLAUDE.md evidence pass is unchanged. Each site's schedule, its predicted time, its terms and
any open flip are emitted with the module (`; mapal.place`). A realized time outside the prediction beyond the band is a
finding — a missing machine fact or evaluator term, fixed in the model, never by a per-site override. The replay corpus
is **`place_replay.tsv`**: per measured site its extents and formats, the part and the box's properties (clock, power
cap, host), each arm's schedule as a plan string, the times and the null — seeded from the walk arms, the split cuts,
the stream-K prices, the raster (C11) and k-fill lines, the warp-specialized arms and the plan-cost losers.

## Step 1: GEMM

Sum-of-products tile sites on the GPU (what `launch_split` places today: the FMA tile, `mma.sync` TF32 / 16-bit /
split forms, `wgmma` TF32 / 16-bit warp-specialized). One implementer in the main tree, commits in order:

1. **C1** — crate `mapal-place` (target-agnostic): machine, schedule and fact types with provenance; reuse closed forms
   (distinct bytes, interval union, three-gap walk gaps); an exact LRU trace simulator; property tests (closed form =
   simulator; monotone in every rate; admissible bounds).
2. **C2** — `Machine::of` for every GPU profile; the lowerings' declarations (constraints, counts, footprints) for the
   present emitters; the candidate generators; a coverage test: the rules' launch for every suite site is a point of
   the space, emitting byte-identical text.
3. **C3** — `place_replay.tsv` and its scoring (picked / null / mixed / open, the model against the rules), report-only
   first, then asserted (model ≥ rules, a committed open list); the walk's bytes against the LRU simulation; the local
   RTX 5080 calibration probes the evaluator needs (start timing; rotated and per-partition keep; efficiency per
   operation × group tile × occupancy).
4. **C4** — the search and `launch_of(schedule)`; the rung `machine-place` (default off until confirmed; its OFF arm is
   the rule chain); the `; mapal.place` line; generated GEMMs on every profile lower, assemble and stay within the
   bound; suite emission diffs per profile.
5. **C5** — the hybrid traversal's lowering and value tests. Persistence stays out of step 1's space (its separating
   fact, the round-robin jitter, is unmeasured); `wgmma` takes ordered traversals only (a declared constraint).
6. **C6** (its own step) — the local RTX 5080 ABBA on changed programs and the full model in both classes; a rented
   confirmation with the calibration probes where picks change (RTX 3090, A100, H100) within the $100 cap; the default
   flipped on; later the rule chain removed.

Later steps: attention and the streamed units; cells, folds and row folds; the co-schedule and residence across the
system (filling free slots with independent work; the shared operand of sibling projections); the CPU translation;
AMD.

## Alternatives rejected

- **Keep per-decision rules, adding facts as losses appear** (status quo): interactions stay unpriced; every loss is a
  patch to one rule.
- **Search by measurement as the source of choices** (ADR-0034's autotuner, TVM-style): needs the hardware at compile
  time, does not carry to unseen parts or to an ASIC that does not exist yet, and does not say why a choice wins. Kept as
  the model's check and calibration.
- **A learned cost model** (Ansor-style): opaque, trained per target — decisions not explained by facts.
- **Per-hardware kernel templates**: the per-hardware path this ADR removes.
- **Whole-program polyhedral scheduling** for placement: its locality objective does not see slots, waves, copy engines
  or bandwidth per level. Its exact access maps are what this model uses (`tile_plan`, `tensor_form`).
- **Places as a tree, levels fixed to executor levels** (the first draft): cannot express cache partitions, per-XCD L2s,
  NUMA or a NoC, nor a CPU's blocking through several cache levels.

## What would falsify it

- After adding the missing machine facts, the evaluator predicts `place_replay.tsv`'s winners less often than the rules
  did, or does not win back plan-cost's losers — the abstraction misses something structural.
- `docs/target-specific.md` grows with each target instead of shrinking.
- The CPU's two recorded cases the first draft could not state (a hardware prefetcher's reach; conflict misses from set
  mapping) stay unexplained after the CPU translation.

## The interpreter

Unchanged. A schedule never changes a value beyond the stated bound (ADR-0049); value gates check the bound.
