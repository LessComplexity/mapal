# Mapal — rules for every agent and workflow

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
5. **End check.** The final integrated build gets the same pass: a table of intended vs realized per optimization per
   arch, and how it turned out in time.

## Facts catalog — every decision is based on these (`crates/mapal-ir/src/algo.rs`)

| query | what it proves |
| --- | --- |
| `emission_plan` / `class` | minimal emission: which objects materialize; products dissolved (views, no bytes); consumer counts |
| `last_use_plan` → `position`, `death`, `escapes`, `carried_by`, `dead_after` | death points; escape into output; `carried_by` = value crosses a `LoopBack` (two-iteration liveness) |
| `loop_plan`, `loop_structure`, `sccs`, `topo_order` | canonical loops: init, merge, decide/advance order, exits |
| `write_delta` | the contiguous range of a carried array each trip writes, through `Update`s and piecewise maps (partial update → move/retain only the delta) |
| `path_plan` (+ `is_single_path`) | independent parallel paths / tasks |
| `tile_plan`, `tile_plan_strided`, `move_plan`, `row_fold`, `cell_plan` | affine map/fold sites, data-move sites, row folds, per-cell maps and what each cell reads |
| `elem_plan`, `map_law` (+ `Outside`) | each array's `out[i]` as a closed function of the index; one map's law, piecewise when every cell off a row / range / column / cell is a capture's own (in-place range map) |
| `radix_plan`, `tile_plan_radix` | an integer as digits `(x ÷ s) % r` of the inputs (constant div/mod splits); map/fold sites whose reads are affine per digit as tile sites over `(i, lane, k)` (implicit GEMM) |
| `precision_plan` (`set`, `round_at`, `cross`, `bound`) | ADR-0041: the precisions each float product admits (innermost `: compute`); values rounded once at the producer; edges a rounded copy must not cross; each tile site's a-priori bound (D3 / ADR-0040 D2) |
| `bounds_proof`, `proven` | in-bounds indices; droppable trap guards |
| `guard_plan`, `gated` | per-Phi arm exclusive work; whether an arm must be gated |
| `trip_split` | per counted loop: the trip intervals guards on the counter cut, each interval's recurrent state slots, the work outside them, batched row writes, whether the first interval's trips but the last can be skipped |
| `whole_fold`, `fold_tree_step` (`CellSite::tree`) | a fold of the function itself over one affine read (a one-row row fold); a fold step regroupable as lane partials + tree where the order is free |
| backend: `ResidentPlan`, `arena`, frame slots (`prepare_storage`), `TargetProfile` | device residency, lifetime-colored buffers, host slots/aliases, hardware facts |

New logic starts from this table; if a needed fact is missing, add it to `mapal-ir` (one query, tested) — never
re-derive it inside a backend pass.

**Facts live on the graph.** The IR is a dataflow execution graph; every fact above is stored on the sealed graph
once computed, and its dependencies on other facts form a dataflow graph of their own. A query reads the stored fact;
only a fact that is not there yet is computed (from its stored inputs). Never recompute a fact per call site.

## Standing rules

- Optimizations derive from Mapal dataflow facts (tile_plan, elem_plan, loop_plan, last_use_plan, path_plan,
  bounds_proof, use counts, death points) + TargetProfile hardware facts, emitted so LLVM can derive the schedule.
  No workload-named special cases, and no parallel module that re-derives what an IR fact or existing allocator
  already models (e.g. a loop-only lifetime allocator next to `LastUsePlan` coloring) — extend the existing
  machinery so the case falls out. Reuse the plans/facts at hand; add a new fact only when none carries it.
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
  is never rewritten to suit them. `benches/algorithms/*.mapal` is the committed, readable snapshot. Every rung has an OFF arm; per-arch defaults only via cited `GPU_RUNGS_OFF` rows.
- Default float face: full precision, a-priori error bound, order unspecified, fastest for the native CPU+GPU
  (ADR-0040). Exact/contract/fixed-order are opt-in; value gates on the default face check the bound.
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
