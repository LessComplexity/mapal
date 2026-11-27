# Mapal — rules for every agent and workflow

## Source formatting

- Format every Mapal source file you create or edit with `mapal fmt <files...>`;
  verify it with `mapal fmt --check <files...>` before handing off. If the CLI is
  not installed, use `cargo run -p mapal-cli -- fmt <files...>` (add `--check` to
  verify). Pass only files touched by the task; do not reformat unrelated sources.
- The formatter owns layout: four spaces, multiline brace bodies, consistent
  spacing. Keep comments and descriptive names readable; avoid hand-aligned padding
  or compacting generated programs onto long lines. Format generated source too.
- Deliberately invalid syntax fixtures are exempt because formatting rejects invalid
  input. Explain that exemption in the test; do not bypass checks for valid programs.
- For code discovery, avoid recursive indexing of archived worktrees, vendored
  libraries or generated bundles. Use bounded reads of known files when indexing
  stalls; do not launch concurrent index builds for the same checkout.

## Evidence pass (mandatory for any optimization or codegen change)

No keep/remove/gate decision stands on timing alone.

1. **Intent, before measuring.** Each rung declares, machine-checkably:
   - what LLVM/NVPTX must derive from the emitted IR (pass/transform: vectorize, unroll, SROA of accumulators, fma
     contraction, LSR, cp.async lowering, …), and
   - the PTX/SASS features that must appear or disappear (instruction mix, vector widths, LDGSTS/cp.async, FFMA per
     load, unroll depth, accumulators/thread, registers, spills, smem, occupancy, launches), and
   - the data movement the facts require per program and per loop trip (host↔device copies, allocations, residency);
     any copy or allocation the dataflow facts do not require is a finding. Placement, reuse and retention (caches)
     are deduced from the graph for every object, loop-carried state included — never a by-value default.
2. **Realized.** For every measured kernel and arch collect LLVM optimization remarks (`-pass-remarks-output`), PTX,
   SASS (`cuobjdump -sass` / `nvdisasm`) and `ptxas -v`; check against the intent. A mismatch is a finding, never a pass.
   Library kernels (cuBLAS/cuBLASLt/CUTLASS) are captured from the actual launch, not assumed.
3. **Explain.** Every timing difference that drives a decision cites the SASS/usage difference that explains it.
   Unexplained ⇒ the decision is "open", not keep/remove.
4. **Breadth.** Judge on the multi-workload suite, not matmul alone: GEMM/MLP, attention/softmax decode, reductions,
   FIR/conv/window, elementwise chains, fan-in/fan-out, loops — CPU-only, GPU-only and CPU+GPU — against the best
   library (cuBLAS/cuBLASLt/CUTLASS) or PyTorch where no library exists.
   Tooling: `benches/gpu/intent.py` (intents), `evidence.py` (checker), `rejudge.py`; a GPU run without its
   `evidence.tsv` is not a result.
   Per A/B, time only the suite programs whose host or device text differs between the arms
   (`benches/local/gpu_changed.sh`, ABBA; equal text is the same object and PTX: null by construction), and run the
   full suite once per step as the safety net; bundles reuse what an unchanged emission built (`BUNDLE_CACHE`).
   Generated-program tests check their programs in parallel (`MAPAL_TESTGEN_JOBS`; same draws, same counters).
   Raw run output (profile archives, box logs, raw result text) never goes into git: `benches/gpu/archive_raw.sh`
   uploads it to a GitHub release and appends `benches/results/evidence/RAW-manifest.tsv`; commit only the evidence
   tables, reports and the manifest (`benches/results/evidence/RAW.md`).
5. **End check.** The final integrated build gets the same pass: a table of intended vs realized per optimization per
   arch, and how it turned out in time.

## Facts catalog — every decision is based on these (`crates/mapal-ir/src/algo.rs`)

| query | what it proves |
| --- | --- |
| `emission_plan` / `class` | minimal emission: which objects materialize; products dissolved (views, no bytes); consumer counts |
| `last_use_plan` → `position`, `death`, `escapes`, `carried_by`, `dead_after` | death points; escape into output; `carried_by` = value crosses a `LoopBack` (two-iteration liveness) |
| `loop_plan`, `loop_structure`, `sccs`, `topo_order` | canonical loops: init, merge, decide/advance order, exits |
| `write_delta` (`Delta::Rows`, `Delta::Pieces`; `FactArms::piece_delta`) | the contiguous range of a carried array each trip writes, through `Update`s and piecewise maps (partial update → move/retain only the delta); or its pieces — `w` cells every `step` (a row of every sequence of a `[B][S][W]` state) |
| `unchanged_plan` → `trip`, `same`, `inputs` | what is unchanged since last time: per loop-body array the cells that may differ from its previous trip (`write_delta` carried through row-local maps — row folds, per-row scales, projections, windows reaching a neighbour row: `Rows` / `Pieces` / `Top`); values equal on every trip; each value's input slots (none: the same at every call) |
| `call_keep` (`CallKeep`: `kept`, `values`, `loops`) | retain-call: per `Call` site, the callee input slots whose argument is the same at every execution (made outside every loop containing the site, or `unchanged_plan.same`) and the callee values all of whose slots are kept (`unchanged_plan(callee).inputs`) — equal on every execution after the first |
| `static_plan` (`StaticPlan`: `fixed`, `roots` → `StaticRoot { bytes, cells, work, realize }`, `work`; `Realize::{Embed, Attempt, Compute}`) | ADR-0048 D3: the values computable before the program runs (`unchanged_plan.inputs` 0, no `read` in the cone, a loop only whole); the roots a run-time value reads; each root realized by bytes against work — embedded by the oracle (≤ `EMBED_BYTES` 1 KiB, ≤ `EVAL_WORK` 2²⁰ steps, more steps than cells, no `elem_plan` law), attempted (an uncounted loop, `ATTEMPT_WORK` 2¹⁸) or computed at run time; work through maps, folds, calls and counted loops (`trip_split`) |
| `path_plan` (+ `is_single_path`) | independent parallel paths / tasks; a host-spine value's consumers follow the host (scalar cone on the spine, a reading loop on the host, a reading bulk op pinned); a host morphism reading a task's object waits for it (checkpoint-covered `Pair` / `Proj` edges excepted; a token alone reads no bytes); `Checkpoint::at`: where the host waits — the earliest value-assembling edge, never before a pinned task the wait needs; a file read and its array stay a task whatever token they take (ADR-0045 D8) |
| `tile_plan`, `tile_plan_strided`, `move_plan`, `row_fold`, `cell_plan` | affine map/fold sites, data-move sites, row folds, per-cell maps and what each cell reads |
| `tile_plan_post` (`TileSite::post`, `TilePost`) | sum-of-products sites whose map body goes on past the fold: the cell's other terms (residual, scale, mask, per-row divisor) as an epilogue derived from the body — the slots it reads, its number captures, whether the fold sits in a guard arm (then every read is in bounds over the whole iteration space); `TilePost::after`: the captured numbers the epilogue reads after the fold; `TilePost::round`: the epilogue ends in the stored rounding (the map writes the narrow format; `precision_plan().store`) |
| `elem_plan`, `map_law` (+ `Outside`) | each array's `out[i]` as a closed function of the index; one map's law, piecewise when every cell off a row / range / column / cell / row-every-`s`-rows (`Outside::w`) is a capture's own (in-place range map) — through captures: a body fn's capture takes the closed law (`Index`, pairs) of what every site passes (`capture_law`; gapfix `iota-law`); an interval's bound through a capture the site passes as a constant (an inlined call's literal) is that constant at the site |
| `radix_plan`, `tile_plan_radix` (`FactArms::head_group`) | an integer as digits `(x ÷ s) % r` of the inputs (constant div/mod splits); map/fold sites whose reads are affine per digit as tile sites over `(i, lane, k)` (implicit GEMM); digits both operands read are batch digits — the most significant rows, block-uniform row digits of `b` (a KV head's query-head group shares one `b` panel); a captured digit's digit (`h / 4`) is a digit of the map index |
| `precision_plan` (`set`, `round_at`, `cross`, `bound`, `exp`, `math`, `narrow`, `store`) | ADR-0041: the precisions each float product admits (innermost `: compute`); values rounded once at the producer; edges a rounded copy must not cross; each tile site's a-priori bound (D3 / ADR-0040 D2); each `Exp`'s form and default-face bound (ADR-0042 D4/D6: `mapal_ir::exp`, the oracle's algorithm on the exact faces); each math builtin's (`exp`, `sqrt`, `sin`, `cos`) `MathForm` — its contract and condition number (ADR-0046 D9; realizations are `mapal_ir::math::choose` over the target's declared units, never a name); each `to<bf16>` / `to<f16>` / `to<f8e4m3>` / `to<f8e5m2>` rounding's `NarrowForm` (ADR-0043 D3: exact on every face; fp8: saturating; `NarrowForm::embeds_in`: format A's values are exactly B's — the tensor-core admission over narrow storage; a default-face gate adds `2·(u·|x| + abs)` per rounding of an operand carrying error); `bound` with one unit per multiplicand (`bound_coef_units`, `TileSite::units`); `store` (`NarrowStore`, `narrow_stored`): an f32 array a map produces whose every element read (through nested captures) is a multiplicand of a bf16-admitting product — stored rounded once by its producer (bf16 storage, the reads widened) where no map leaves a site plan, unit group or f32 stream member it was on — a store rewrites only its own copy of a body several maps share (`apply_store` owns each body it changes: `own_body`, a deep copy; the inliner's and R-UN's copies share bodies); an epilogue site's array too (its rounding the epilogue's last step, `TilePost::round`) |
| `tile_plan_narrow` (`TileRead::narrow`, `elem_bytes`) | ADR-0043 D8: sum-of-products sites (strided lanes, epilogues) whose operand is the exact widening of a `bf16` / `f16` array element — each such read's storage format and bytes (the cost model's operand traffic, `PlanShape.{ea, eb}`, `profile::site_bytes`); the f32 site's bound; disjoint from every other tile plan |
| `tensor_form` (`TileRead::tensor_form` → `TensorForm`, `box_of`) | a tile operand as a dense ≤ 3-axis tensor over (I, Lane, K): dims innermost first, none overlapping; the box that is a panel and whether such boxes tile it exactly (refused: ksplit windows / im2col, overlapping axes) |
| `mapal_ir::split` (`SplitForm::{coef, admits, k_min}`, `AccModel`) | a split realization of an `{f32}` product on a matrix instruction (tf32x3, bf16x6, bf16x3n, bf16x2n — two bf16 parts of an f32 operand against a bf16 one, `2⁻¹⁶`, `K_min` 288 —, narrow1: parts, kept products smallest class first, each k-step from a zeroed accumulator then one round-to-nearest add): its a-priori coefficient `(e + C_j)(1+γ_n) + γ_n/u` against D2's `γ_{K+1}(u)` at the site's `K`, `S` pieces and the accumulator's measured guard bits; the operand range below which a part leaves the normal grid (`exact_above`: the guard's FFMA path) |
| `bounds_proof` (`proven`, `index_range`, `arm_used`; `FactArms::cond_bounds`) | in-bounds indices (droppable trap guards) from signed ranges narrowed by the conditions holding where a value is used — a guard arm's condition on its exclusive work (`guard_structure`), a loop's continue condition on its advance cone and exit condition on its exit value, a `Phi`'s per arm, run back through `+ − ·k ÷k %k` — and by loop-carried counters ranged from `loop_plan` (init, affine step, continue bound); each index's range; the guard arms a proof needs gated; sibling `Proj` views of one input slot share a condition's narrowing (a rotation's partner `i ± h` under `i % 2h < h`); a selecting fold's component (each step keeps the accumulator's, takes the element or a constant: argmax / argmin / first-match index) ranged by seed ∪ items ∪ constants |
| `guard_plan`, `gated` (`guard_structure`, `cond_reads`) | per-Phi arm exclusive work; whether an arm must be gated (trap, heavy, or an index in bounds only under its condition) |
| `trap_cell_plan` (`CellSite::traps`) | the maps `cell_plan` refuses only for unproven `Index` reads — a device cell checks each (the trap recorded in the module word, raised at the next host checkpoint — a print, the clock, a host trap, the exit or a device-to-host copy, the only way a device value reaches host control; rung `device-trap`) |
| `Ty::index_ty` | an extent's index type: `i32` up to 2³¹ cells, `i64` past it — an `iota`'s elements |
| `trip_split` | per counted loop: the trip intervals guards on the counter cut, each interval's recurrent state slots, the work outside them, batched row writes, whether the first interval's trips but the last can be skipped |
| `step_algebra` (`FmaPair`, `Regroup`, `StepAlgebra::flags`) | per function: float product–accumulate pairs (product read once) that may be one rounding (FMA) where the face contracts; a fold body's step that may regroup (lane partials + tree: op, bits kept or not, trip count, per-read stride along the fold axis, a captured array's affine read included (`radix_plan`'s fold-index digit: a 1-row projection's weight row at stride 1) — layout, never admission); the one realization rule host and device share |
| `fold_combine` (`FoldCombine`, `RowDep`, `DepLaw::{Factor, Shift, Rescale, Phase}`, `RescaleArg`) | per fold site (row fold, whole fold, tile site): its state over blocks of its axis combines by its own `⊕` (bits kept or not); each value constant along the axis its term reads through the term's cell map, by its law — a product / quotient factor applied once after the combine (`⊕ = +`), a shift of a max / min, a rescale (`⊕ = +`, the value enters only as `exp(κ·(x − v))·y`: a block's partial takes its own shift, rescaled by `exp` when combined — ADR-0042 D7, through product paths at depth; `FactArms::exp_rescale`), or a phase (the partials need it combined first) |
| `unit_plan` (`UnitGroup`, `UnitMember::color`) | row-block groups: tile sites + the folds and cells reading their rows, every in-group edge row-closed (`Reach::within_rows`), a tile's `b` operand never a member; stored = escapes or read outside; on-chip row buffers colored by death in the group (in place along a same-cell chain); bytes per unit row and the operand bytes every unit re-reads; the extended pass over epilogue and mixed-radix sites (P·V's rows a digit permutation of the scores', `RowMap`), with captured-number terms (`FactArms::unit_uniform`: a tile whose operand address a loop counter moves — the launch moves the pointer) and epilogue / row reads proved by digit forms where `Reach` cannot (`post_rows`); the seed's output rows as a digit permutation of its tile rows (`out_rows`; the seed's `row_of`) and the group's walk order (`UnitGroup::order`, the seed's tile order: a block one KV head's queries) |
| `stream_plan` (`StreamGroup`, `StreamRole`, `StreamLaw`, `GuardBox`) | one pass over windows of an axis: a row-block group whose tile folds over an in-group row's cells (tile → cells and row folds over its lanes → tile over the same axis: attention, softmax-weighted pooling) — each member's role (`Produce`, `Cell`, `Fold`, `Consume`, `Row`), each fold's `⊕` and each per-row value's law from `fold_combine` (a rescale's shift the running extreme), a guard's sides through `+`, `·literal` and reads of a captured integer array (its element law — `elem_plan` through the capture — else a per-row load `RowLoad`) and captured numbers' per-launch digits; open sides decided in the kernel (`GuardEnv`); roles cover the members up to the last consumer (later members their own launches), the consumer role ahead of the producer's where the head width equals the axis (`d = T`); axis-wide values never stored; `StreamGuard::check`: the consumers whose column operand is finite only at run time (the skip's vote reads the skipped windows' column elements, rung `finite-check`) |
| `access_plan` (`reads`, `writes`, `class`, `edges`, `trip`; `Hazard`) | the bytes each morphism reads (through `Pair` / `Proj` of products) and writes; storage classes (a carried slot's views, next value and exit; an `Update` with its array); RAW / WAR / WAW edges of each loop trip, and of any grouped op sequence with a backend's extra byte sharing (arena) — the order device work is placed from |
| `residence_plan` (`bytes`, `working_set`, `at` → `Residence::{Chip, L2, Dram}`) | where a value's bytes live between producer and last reader: the peak live bytes (storage classes once, index domains none) over its interval against a cache's capacity (`TargetProfile` L2) — the rate a cost model prices that value's traffic at |
| `whole_fold`, `fold_tree_step` (`CellSite::tree`; `FactArms::row_tree`) | a fold of the function itself over one affine read (a one-row row fold); a fold step regroupable as lane partials + tree where the order is free — a cell's one nested fold that loads array elements, beside folds that load none (a norm's root) |
| `CSE value scope` (`mapal-rewrite` `value_scope`) | an element view (an `Index` yielding an array or record) outside every guard arm's own work keys its constant index by value, so equal views are one object (one host view, one device copy); inside an arm, keys stay by object |
| backend: `ResidentPlan`, `arena`, frame slots (`prepare_storage`), `TargetProfile`, site registry (`@mapal_sites` / `mapal_gpu_sites`), `mapal_gpu_zero_for` | device residency, lifetime-colored buffers, host slots/aliases, hardware facts; each site table's pick, candidate kernels, shared-memory limits and occupancy resolved when the module loads (gapfix `load-dispatch`); a hoisted zero skipped at run time where the site's decided pick adds nothing (`zero-hoist`: only as far back as the bytes the working steps between touch and its own fit `Gpu::l2_keep_bytes`, never across a clock read; after the last *launch* touching the output's bytes — a `horizontal` member's launch is its group's last member); model-ready placement's rungs over the facts above: a per-trip cell a placement candidate, operands made once uploaded once (`trip-cells`), placed where it moves fewer bytes a trip (`bytes-place`), a trip's chain of in-place range maps writing the carried buffer (`range-chain`, `Carry::chain`), a warp a cell where the fold reads at unit stride (`stride-lanes`), a carry's next state forwarded on the device (`carry-fwd`); a streamed unit's epilogue numbers as launch arguments and its epilogue arrays as memory operands (`stream-numbers`); a piecewise range map run once outside every loop (a prompt's cache rows) a device cell, written over its source's buffer where nothing reads the source after it (`last_use_plan.view_dead_after`, `ResidentPlan::inplace`), else whole into its own, priced by the bytes it moves (`once-range`, `once-bytes`); a loop's init array read after entry only through the loop state hands its device buffer to the carry (`carry-take`); a warp-a-cell fold loop unrolled `NAIVE_UNROLL` trips, the loads in flight the stream rules price (`lanes-unroll`); rungs `finite-check`, `narrow-window` (bf16 stream windows), `device-trap` (runtime `mapal_rt_checkpoint_hook` / `mapal_gpu_trap_arm`), `cand-made` (a map an operand of which a candidate cell or device site makes is a placement candidate: the fixed point decides a chain whole) |
| backend: `Gpu` properties (`ffma_per_dram_byte_pm`, `regs_per_resident_thread`, `l2_bytes_per_sm`, `dram_bytes_per_us_per_sm`, `fill_warps`, `bar_cycles`, `Caps` — `reg_realloc`, `cvt_f8`, `ptx` (`Ptx`: the PTX ISA a capability needs; each device module declares its largest in `; mapal.ptx`, llc's `mattr`) —, `DeviceCost` — `tma_copy_ns`, `lds_bcast_pm` —), `wave_full`, `tile_capacity`, `rules::rungs`, `streamk_priced`, `streamk_epi_pays`, `ws_persist_pays`, `stream::mma_fmt` | the physical properties a GPU decision reads (queried: `cuDeviceGetAttribute`; measured: `calib.cu`); a matrix product's flops an SM a clock (`Gpu::{mma_tf32_per_clk, mma_half_per_clk, wgmma_tf32_per_clk}`, `matrix_flops_per_clk` / `_per_ns(op)`: `calib.cu matrix`) against the FFMA tile's achieved rate (`Gpu::ffma_tile_flops_per_ns`, `DeviceCost::flops_per_ns`; the peak where unmeasured — the side `rules::split_rate_pays` weighs a split tile against: a split form of `P` passes pays at `1.6·P`, the matrix rate taken over the tile's issue against its pipe — `split_issue_pm`: its products, flushes and the realized cut's slots an element) and a stage's copies (`stage_consume_ns` against the bulk-copy engine's `tma_copy_ns`, `tma_calib.cu`: the wgmma stage depth); the accumulator's guard bits (`MatrixOp::acc_g`, `mma_calib.cu`); a tensor tile's `k` cut by its waves on the slots it holds (`mma_pieces`: `⌈blocks·S/slots⌉·(P/S + stages)`); a site operand DRAM-resident (`MAPAL_SITE_COLD`: `residence_plan` against `Gpu::l2_keep_bytes`, the cyclic working set the L2 keeps, `l2keep.cu`: cold trials); a row-fitted grid's wave fill; a grid's spread over the SMs (`grid_spread_pm`; `vec_cells_pay`: the vector cells' grid against the scalar grid's where the DRAM share an SM asks); a tile's residency (the slots stream-K walks: `k-fill`); each rung's value at a site's fill and face — the replay of every cited measurement (`rung_replay.tsv`); a launched grid's own fill (`streamk-fill`); an under-filled site's stream-K walk priced by the part's `DeviceCost` rates (the FMA tile, `red`, the zero pass) against the panels it saves (`streamk_priced`, sweep-gemm-2); a streamed unit's products on the warp matrix instruction every one admits (`stream::mma_fmt`: `TileSite::half` × `Caps::matrix_op`, its geometry and accumulator, priced by `stream_ns` on `mma_half_per_clk`, sweep-attention-2); a register-staged tile's k-major `b` copied along its rows (`kvec_b`: a layout over `TileRead` / `RadixAddr` `b_k_major`, no new fact), its stored words XOR-swizzled over the 32 banks (`kvec-swizzle`); a DRAM-streaming atomic split's cost a partition cell (`DeviceCost::split_cell_ps`; `TargetProfile::split_priced`: the stream at the split grid's spread against `S · cells` partials, gapfix C5); row-major's `b` re-reads past `l2_keep_bytes` (`Gpu::b_rereads`: a panel's readers `gx` launches apart) and `Gpu::dram_bound` (`DRAM_BOUND_PM`) — the raster's reuse gate (C11); a convert-free tensor tile's ring (`gpu_mma_fly` / `mma_fly_pick`) where its panel's DRAM time — under the walk the site takes (`Depth::Fly { walk }`: a grouped wave's panel readers) — fits its products but not their latency, keeping `stream_fly_per_sm` past `wait_group(stages − 2)` (C3); L2 eviction priority on a load (`Caps::l2_evict`) and a read-once operand against the keep (`rules::stream_ld_pays`: a dense `tensor_form` whose distinct elements are the site's cells, past `l2_keep_bytes`, within `STREAM_KEEP_SHARE` of it — `ld.global.cs`, C2); `mma_walk` / `MmaWalk` (a 16-bit tensor tile grid's rounds over the SMs' matrix pipes, `⌈tiles / sms⌉` tiles on the busiest SM, against stream-K over every resident slot at the whole warp tile — `TargetProfile::gpu_mma16_walk_tile` — its partials at `atomic_ops_per_ns`, its zero pass at `fill_bytes_per_ns`; replay `mma_walk_replay`) |

New logic starts from this table; if a needed fact is missing, add it to `mapal-ir` (one query, tested) — never
re-derive it inside a backend pass.

**Facts live on the graph.** The IR is a dataflow execution graph; every fact above is stored on the sealed graph
once computed, and its dependencies on other facts form a dataflow graph of their own. A query reads the stored fact;
only a fact that is not there yet is computed (from its stored inputs). Never recompute a fact per call site.

## Standing rules

- Optimizations derive from Mapal dataflow facts (tile_plan, elem_plan, loop_plan, last_use_plan, path_plan,
  bounds_proof, use counts, death points) + TargetProfile hardware facts, emitted so LLVM can derive the schedule.
  No workload-named special cases (the only exception: a recorded tailored fix, below), and no parallel module that re-derives what an IR fact or existing allocator
  already models (e.g. a loop-only lifetime allocator next to `LastUsePlan` coloring) — extend the existing
  machinery so the case falls out. Reuse the plans/facts at hand; add a new fact only when none carries it.
- **Open problems are attacked, not just recorded** (Sapir, 2026-11-23). When work finds an open problem — a loss
  against a competitor, a mixed rule, a missing fact or property, a bug — attempt to solve or optimize it in the same
  piece of work where feasible; report only what was tried and why it did not close. Solve it generically: a fact,
  property or rule change that makes the specific case fall out and helps every program with the same structure.
  Only when no abstraction exists is a tailored fix allowed; it is recorded in `docs/tailored-fixes.md` (the case,
  where it lives, why no generic form exists yet, the generic direction) and revisited until a generic optimization
  replaces it.
- **A mixed result is a missing fact** (Sapir, 2026-10-26). An optimization that wins on some programs and loses on
  others beyond null means its condition lumps different situations together (over-abstraction). Find what separates
  winners from losers in the SASS/usage evidence (layout, extent vs SMs, reuse, occupancy, …), add it as an IR or
  `TargetProfile` fact, and make the rule read it, so each case gets its own decision.
- **Hardware is physical properties, never a name or version** (Sapir, 2026-11-20). Compilation matches the shape's
  geometry to the hardware's physical properties (SMs, threads/registers/smem per SM, L2 bytes, bandwidths, latencies,
  issue/launch/barrier costs, instruction rates) — measured or queried, stored as `TargetProfile` facts. No decision is
  switched on or off per architecture, part or version ("4090 on, 5080 off" is wrong): a rule that loses on some
  hardware is missing the property that separates the cases. The rung rules (`crates/backends/llvm/src/rules.rs`)
  read properties only; every cited measurement is replayed against them (`rung_replay.tsv`, test
  `rung_rules_replay`) — a new measurement is a replay line, never a per-part row. OFF arms are measurement flags
  (A/B, debugging), never per-hardware defaults. The GPU `native-gpu` profile is the device query + the calibration
  file (`benches/gpu/calib.cu`), never a name lookup.
  Capabilities are properties too, described by the geometry they execute, never by a version compare: matrix
  instructions as shapes (M×N×K per issuing group, operand types, where operands come from — registers, smem
  descriptors — and where the accumulator lives: registers or tensor memory, e.g. datacenter Blackwell `tcgen05` +
  TMEM), memory levels and their capacities (registers, smem, TMEM, distributed smem across a cluster, L2), copy
  engines (`cp.async`, TMA tensor boxes / swizzles / multicast), cluster shapes, barriers (`mbarrier`, grid sync).
  Rules map the dataflow's tile geometry onto whichever capabilities the target has; a new chip is a new property
  set, not new code paths keyed on its name.
- **Generic, proven on unseen cases.** Never target a benchmark. Every rule is validated by a property test over
  programs from the random generator (`crates/mapal-rewrite/tests/testgen/`, extended as needed) that the change was
  not designed on: values vs the interpreter / bound, plus movement and kernel intents. Benchmarks are acceptance
  only; a change that helps only the suite is rejected.
- **No "fusion" concept.** Compilation follows data usage in the dataflow graph: which work runs together, what stays
  in registers/shared memory and what is stored all follow from producer/consumer footprints (`footprint_plan`), death
  points, independence (`cell_plan`, `path_plan`) and `TargetProfile`. Kernel boundaries are a result, never a pass.
  The rung is `flow` (formerly `fuse`). Work units must keep the parallelism the graph proves (independent cells across
  threads, folds as tree reductions) and compute a value once when it has several readers.
- **Benchmarks are written in idiomatic Mapal** (map / fold / loop / guards / captures), never unrolled or shaped to
  what the compiler already recognizes. If the natural form compiles worse, the facts are generalized — the program
  is never rewritten to suit them. `benches/algorithms/*.mapal` is the committed, readable snapshot. Every rung has an OFF arm for measurement; its default comes from the rule over facts, never per hardware.
- **Correctness is the flow and the bound, not bit-identity** (Sapir, 2026-11-26, ADR-0049). A program is correct
  when its flow of operations is the dataflow and every value is within the stated a-priori bound (default float face:
  full precision, order unspecified, fastest for the target — ADR-0040). The compiler orders, splits and combines
  freely (atomics, stream-K, split-K included); results may differ in the last bits between runs and targets. Never
  refuse, price up or restrict an optimization to reproduce bits, and never require a target (CPU/GPU/ASIC) to provide
  deterministic reductions. Exact/contract/fixed are **development modes** (oracle, replay, debugging, value gates),
  not language promises; value gates on the default face check the bound.
- GPU archs: RTX 4090 sm_89, RTX 5090 sm_120, H100 sm_90a, RTX 3090 sm_86, rented on vast.ai on demand
  (`benches/gpu/vast_part.sh`/`vast_run.sh`). `vastai destroy instance <id>` needs `-y`; never touch instances you
  did not create — any instance labeled `stocklab-*` belongs to another project; never print keys or provider logs. Same-box ABBA A/B, value gates first, report null variation.
- This is a laptop (14 cores): leave 2 cores free for heavy local work (`CARGO_BUILD_JOBS=12 RUST_TEST_THREADS=12`).
  Background QoS (`taskpolicy -b`, efficiency cores only, ~3× slower) only when the user asks for a quiet machine.
- Token budget is tight: wait on a GPU box or long test with ONE blocking background command and read its summary
  once (grep/tail, never full logs); no tick-by-tick polling; keep returns to short tables, details in the repo.
- Parallel agents share one scratchpad: each keeps its files in its own prefixed subdir.
- Never push. Results only in README (tables + link).
- Name action items, branches, result dirs and session logs by topic (`deduced-residency`, `generic-fusion`), never
  by session number (`s65`, `S66`).
