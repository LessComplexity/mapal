# model-ready: structure — the readable Llama compiles as written

Branch `model/structure` (from `fac7691b`). The scout's readable form of Llama-3.2 (per-layer weight
structs, `[LayerCache; 2]`, a layer loop inside `prefill` / `decode_step`, nested arrays) is now a
test fixture: `crates/backends/llvm/tests/structure/llama_tiny_readable.mapal`, using the new
`() -> make_model`, `((1.0, 1.0), steps)` and `-> ret.0; caches -> ret.1;` forms.

## Before / after (tiny model, this machine, RTX 5080)

| program | before | after |
| --- | --- | --- |
| readable form, `mapal build` (native) | rewrite panic (`replay.rs:627`); with the scout's patch, a 605 KB `.ll` whose `clang -O2` ran > 600 s at 1.3 GB (killed) | 0.8 s build, no SSA aggregate holding an array; run 4 ms; `mapal test --exact` PASS (byte-equal to the interpreter) |
| readable form, native-gpu (RTX 5080) | not buildable | 1.4 s build + run; tokens = interpreter = PyTorch twin (`236 236 236 236 49 49 49 133`); logits vs PyTorch f64 max abs 5.2e-6 (CPU 5.1e-6; PyTorch f32's own: 5.0e-6) |
| scout probe `r2` (struct entry state of a printing loop) | SIGSEGV (CPU) | equal to the interpreter, without the scout's `drive.rs` patch |
| scout probe `p23` (task reading a printing loop's exit) | `mapal trap: index_oob` | equal |
| print in a loop exit arm (`s5`) | printed on every trip (0 1 1.5 1.75), returned garbage | printed once, after the loop |
| decode step's final norm + vocabulary projection in the layer loop's exit arm | computed on every layer trip (decide phase); with the GPU arena, wrong logits | computed once, after the loop |

Raw run output: `.claude/raw-evidence/local-rtx-5080/model-ready-structure/` (not in git).

## What changed

**Language (mapal-lower, mapal-ir `Value::Unit`).**
- `() -> f` calls a zero-parameter fn: the `()` literal is the `Unit` witness (an effectful
  callee takes the token alone).
- Float literal widths come from context through tuples, map bodies, captures and array
  literals: a map/fold signature resolves with the function's literal variables (`acc.1` of a
  `(1.0, 1.0)` seed, `map { i -> 0.01 } -> s: [f32; 64]`, `[1.0, 2.0]` read as `f32`).
- `ret.k` slot writes in a loop exit arm: a complete set packs into the one exit value.
- **A loop exit arm that computes runs after the loop.** The loop exits with the loop-varying
  values the arm reads (and the token); the arm's statements are then straight-line code. Its
  effects happen once, in order (they ran every trip), and its work leaves the decide phase,
  which runs every trip: a layer loop's final norm + output projection is computed once.

**Rewrite.** `inline_return`: a callee Return written by a call the plan inlines is inlined there
too (the scout's patch), regression test `mapal-rewrite/tests/inline_return.rs`.

**Facts (mapal-ir `path_plan`).** Generalized from the clock read's rule to every host-spine value:
- *host-value cone*: a value the host spine writes (a token op's product, a printing loop's
  state / exit, an effectful call's result) exists only when the host reaches it; a scalar
  consumer joins the spine, a loop reading one becomes a host loop, a bulk op reading one is a
  pinned task at its own position. (The fn input and its projections exist at entry.)
- *host reads wait*: a host-spine morphism reading a task's object waits for that task — the
  `Pair` / `Proj` edges a checkpoint's wait already precedes are covered (for a checkpoint
  inside a loop, only edges in that loop: the wait is hoisted to its entry). Supersedes the
  scout's `drive.rs` pre-loop patch.

**Runtime.** A non-entry Named fn's big blocks (≥ `heap_min_bytes`) come from a thread-local
per-call arena (`mapal_rt_frame_mark` / `_alloc` / `_release`), released at its return — a decode
step's per-layer caches no longer go on its stack (test `big_frame_step`, 4 × 256 KB caches).

**Emitter (backends/llvm): arrays at any depth by address.**
- Named fn input: every array at any depth is a pointer (`lower_named_input_ty` =
  `lower_ptr_paths(ty, array_paths(ty))`); a map/fold body's captures likewise
  (`capture_paths`). The input's pointer positions are *pinned* in `plan_views` (consumers
  needing bytes convert at their edge); the ABI is set before any view planning (`init_abi`).
- Named fn output holding arrays: written through `ptr noalias %out` into the caller's storage
  (`out_by_ptr`); the entry keeps its return.
- `Index` of an array of arrays / of products is a view of the element (its address inside the
  array), registered with the view roots so an in-place `Update` of the array waits for it;
  `Update` / `Pair` / `Proj` / `Output` of products of arrays copy field by field
  (`copy_layout`: arrays by `memcpy` or address, never an SSA aggregate).
- Call / capture positions count as addresses at any depth in the view plan.
- A product every array of which is forwarded on every side (a small struct of pointers) still
  moves whole, so the text of programs without products of array bytes is unchanged; goldens
  changed only by the out-pointer ABI (`capture_one_kernel_matmul`, `update_inplace_carried_loop`)
  and two cell-kernel ids (`devmap`: a cell kernel's id now carries its `cell-law` slots — two
  inlined copies of one body, one over an `iota` law and one over a loaded array, had one name and
  two signatures: "invalid redefinition" in the device unit).

**Nested loops — decision.** A loop nested in a loop is a call to a loop-carrying fn (the readable
form writes `decode_step` with its layer loop, called from the decode loop). With the boundary
now moving addresses only, every loop fact stays per function on flat canonical loops
(`loop_plan`, `trip_split`, `write_delta`, `unchanged_plan`, `bounds_proof`), and the call
boundary is what `call_keep` / `unchanged_plan(callee).inputs` already describe. Source-level
nesting (L1504) is to be desugared the same way (outline the inner loop into a synthesized fn of
its carried values and reads) — not done in this step.

## Tests

- `crates/backends/llvm/tests/structure.rs`: 11 fixture programs (the readable model, the scout's
  probes as differential tests) — interpreter = native raw and rewritten (exact face, byte for
  byte, no array-holding SSA aggregate) = emulated GPU (`rtx-4090`, `rtx-3090` profiles); plus
  a generated-program sweep (struct-of-arrays layers in an array, caches updated through element
  views, printing decode loop, exit arms that compute and print; sizes / fields / layers / ops
  per seed) with coverage counters (out-pointer calls, element views, exit-arm prints > 0).
- `crates/mapal-rewrite/tests/inline_return.rs`.
- Final code: 968 tests passed, 0 failed over 86 test binaries (mapal-ir / -lower / -rewrite /
  -interp / -syntax / -check and the backend's structure, differential, calls, natural_forms,
  recurrence, gpu_host, named_host_dispatch, retain_delta, onto_device, boundary, bulk_stage,
  caching_memory), plus golden_ll 36, resident 17, devmap 14. The long device-geometry suites
  (device_geometry, epilogue, exp, f8, flash_attention, fold_algebra, half_tile) passed on an
  earlier revision of this branch; hopper_tma onward were not run on the final code.

## Open

- **Real size** not run with the readable form (the per-call arena removes the stack limit; the
  copies of the carried caches across the call are the placement step's).
- **GPU placement / residency across the call** (per-element maps and 1-row projections on the
  CPU, caches re-uploaded every step): the placement step's (scout blocker 4).
- **Struct captures in device sites**: `cell_site` / tile sites admit only array and number
  captures; after inlining most struct captures are read only through `Proj`, so narrowing a
  captured product to its read fields (a rewrite over the body input) would make them plain
  array captures every fact sees.
- **Effectful slot exits**: an effectful fn's `ret.k` exit (L1307 at top level) still packs inside
  the loop, so a print in such an arm runs every trip; returning whole is the workaround.
- **path_plan cycles (pre-existing)**: the carried-loops / closed testgen sweeps hit
  `task_topo.len() != tasks.len()` (a scalar component both feeding and reading one fold task) —
  267 times on `fac7691b` as on this branch; the backend demotes such a plan to sequential.
- **Arena vs decide-phase reads**: before the exit-arm change, the readable form's GPU run was
  wrong with `arena` on (right with `--gpu-off=arena`); the exit-arm change removes the work from
  the decide phase, the arena's interval for an operand read every trip in the decide phase was
  not isolated.
- **Source-level nested loops** (above).

## For integration

CLAUDE.md facts table, `path_plan` row:

| `path_plan` (+ `is_single_path`) | independent parallel paths / tasks; a host-spine value's consumers follow the host (scalar cone on the spine, a reading loop on the host, a reading bulk op pinned); a host morphism reading a task's object waits for it (checkpoint-covered `Pair` / `Proj` edges excepted) |

`docs/tailored-fixes.md`: no tailored fix.
