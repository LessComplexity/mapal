# long-rows — the fold-state combine and rows over several blocks (2026-11-21, rtx-5080 + Threadripper 9970X)

Step goal: long reductions split into blocks even when a row does not fit on chip, from one generic fact: how a fold's
state combines across blocks of its axis, after a per-block transform where the term's dataflow gives one. Raw output
(suite runs, ABBA, CUPTI traces, SASS, identity sweep, test logs) is in `.claude/raw-evidence/local-rtx-5080/long-rows/`
(local, not in git).

## What changed

| | fact / rule | OFF arm |
| --- | --- | --- |
| 1 | **`fold_combine`** (mapal-ir `algo/algebra.rs`, stored per function, CLAUDE.md row). For every fold site (row fold, whole fold, tile site): its state over blocks of the axis combines by its own `⊕` (bits kept for integer steps), and each value constant along the axis that its term reads — through the cell map the term reads, at a row-aligned `x[t ÷ c]`, a constant cell or a number capture — gets a law from the dataflow of that cell's body: **`Factor`** (`⊕ = +`, the value reaches the output only through products and quotients whose other operand never reads it: `(Σ x)·v` or `(Σ x) ÷ v`), **`Shift`** (`⊕` a max / min, the output is `x ± v`: monotone rounding, exact), else **`Phase`** (the partials need `v` combined first). The law's identity (`1`, `∓0`) substituted for every read of `v` gives `x` bit for bit. | — (fact) |
| 2 | **`row-slices`** rung (`func/flow.rs` `row_slices`, `slice_phases`, `slice_geo`, `UnitGen::sliced`). A flow unit whose folds all regroup (tree steps over f32, the order free) may run each row over `s` blocks: per phase, the blocks stride over the `rows · s` items; a fold walks its slice into a partial (law reads at their identity) stored to a workspace (a module global); a later phase combines the row's `s` partials — one a lane, the lanes' tree, a fixed order so every block of the row gets the same bits — applies the laws' values, then the seed; cells run in the phase their sources complete in (no row kept on chip: a later phase recomputes a cell at its read). Phases meet at the `calls` step's grid barrier (`mapal_gsync`), one cooperative launch (`launch_coop`: the runtime caps the grid at its occupancy; the items loop strides). A sliced unit is no `horizontal` / `calls` member and takes no zero tail. | `--gpu-off=row-slices`; `--row-slices=<s>` forces `s` (measurement) |
| 3 | **The rule** prices one block a row against `s ∈ {2…64}` blocks a row from measured properties: a block's walk of `n ÷ s` elements a lane at `row_iter_ns` (rounds of the co-resident blocks; a walk reading only rows kept on chip costs none), one load latency (`lat_ns`) per restored fold, floored by the unit's bytes at the rate of where they live (`residence_plan`); the launch is programmatic for one block a row, cooperative (`coop_launch_ns`) plus `phases − 1` barriers (`gsync_ns`) sliced, plus `coop_first_ns` where the unit is outside every loop. New `DeviceCost` properties (rtx-5080 profile and `calib/rtx-5080.calib`): **`row_iter_ns` = 87** (CUPTI, one block a row: 1 × 131072 → 131.6 us over 1536 lane-iterations, 4 × 32768 → 34.3 us over 384), **`coop_first_ns` = 38 000** (driver-API trace: a process's first `cuLaunchCooperativeKernel` 44.8 us median of 7 [39.4–52.7] against `cuLaunchKernelEx` 6.7 us; later ones 3–7 us). A part without them slices nothing. | unpriced → one block a row |
| 4 | Bench rows `pt_lsm_{1_131072,4_32768,32_8192}` (+ `px_lsm_2_512`): q·W scores (d = 16, a tile site) → softmax over few long rows (gen_tiles.py `long_softmax`); torch legs `eager` / `compile` of the same math and `sdpa` / `sdpa-compile` (torch.softmax with exact exp); the attention rows `pt_attn_*` get `sdpa` / `sdpa-compile` legs (`scaled_dot_product_attention`, exact exp: a time reference, not gated). | — |

## What the fact derives — and why flash attention does not fall out

| program | site | deps (law) |
| --- | --- | --- |
| `pt_attn_1024_64` | `mx` (row max of the scores) | none |
| | `den` = Σ `ex`, `ex = poly(s − mx)` | `mx` **Phase**: the exponential is the program's `(1 + x/256)^256`; no IR op is a homomorphism `(+) → (×)`, so `exp(m₁ − m)` rescaling has no exact law |
| | `o` = Σ `p · V`, `p = ex ÷ den` (tile) | `den` **Factor ÷** (o = (Σ ex·V) ÷ den), `mx` Phase (through `ex`) |
| norm + projection | `y` = Σ `xn · W`, `xn = x · rinv · 2` | `rinv` **Factor ×** |
| shifted max | `max(x − m)` | `m` **Shift −** |
| `pt_causal_1024_256` | the whole-fold `den` | the max as a number capture: Phase |

Online softmax (running `(max, sum)` merged by `exp(m₁ − m)`) needs an exponential law; Mapal has no `exp` operation and
the programs' polynomial `exp` does not satisfy one, so attention's `den` is a phase after `mx`: a streamed
(flash-style) unit would be two passes over the keys (re-computing q·Kᵀ, or storing it), with `P·V`'s `÷ den` moved
after the sum by the Factor law. That streamed row-block unit is not built (open 1). At model scale the attention is not
a device tile site at all (model-scale findings 2–10): `pt_llm_pre_t2048_pv` runs q·Kᵀ and P·V as cell kernels, and
the GPU-only decode's softmax unit is `32 × 8192` (`mx, pe, den`), which the rule keeps at one block a row (below).

## Byte identity

`--gpu-off=row-slices` equals `fc56a298` on **2832 host / device modules** (247 `benches/algorithms` programs, the
model-scale ones included, × rtx-5080 / rtx-4090 / h100-pcie × default / exact face), compared as line multisets with
kernel ids and metadata numbers blanked (a new rung field changes every kernel-id hash; `ident/norm.sh`). With the rung
on, the final rule slices **one** of them: `pt_lsm_1_131072` on the rtx-5080 (the only part with `row_iter_ns`); every
other module equals the OFF arm.

## Intended vs realized (rtx-5080; `ptxas -v`, `cuobjdump -sass`, CUPTI `ktrace.cpp`)

| | intended (fact + rule) | realized |
| --- | --- | --- |
| pt_lsm_1_131072 | `mx` phase 0, `ex`/`den` phase 1 (Phase on `mx`), `y` phase 2: 3 phases, 2 grid barriers, s = 64 (model 45.0 us with the first cooperative launch vs 134.1 us one block a row) | `; mapal.flow … slices=64 phases=3 laws= ns_sliced=44957 ns_row=134118`; host `launch_coop(…, i32 64, …, i32 256, i32 2048)`; 40 regs (OFF 24), 0 spill / stack, 2048 B smem; SASS 896 instr (OFF 488): 2 `ATOM` (barrier arrivals) + 4 fences, 58 `BAR` (OFF 18), 22 `LDG`; CUPTI: flow kernel **182.3 → 7.6 us**, device busy 193.9 → 18.8 us, span 222.9 → 98.3 us |
| pt_lsm_4_32768 | first rule (no `coop_first_ns`): s = 64 (6.0 vs 33.9 us); final rule: one block a row (44.0 vs 33.9 us outside a loop) | first: flow kernel 34.2 → 6.4 us, busy 40.4 → 12.6 us, but span 80.8 → 89.2 us: the first `cuLaunchCooperativeKernel` 34.8 us of host API (missing fact → `coop_first_ns`). Final: module = OFF's (md5) |
| pt_lsm_32_8192, GPU-only decode B1 L8192 (`32 × 8192`) | one block a row: `ex`/`pe` kept on chip makes a walk free (6.1 us) against 6.95 us sliced | unsliced; forced s = 4 measured 6.66 us kernel vs 5.89 (calibration) |
| laws (`long_rows.rs`) | a Factor / Shift dep drops a phase: 2 phases where the phase rule alone takes 3 | `laws=mul` / `sub` / `div`, `phases=2` |

Calibration (CUPTI kernel time, median of 7, softmax over q·W rows, `--row-slices=<s>` forced; rule = the final rule):

| rows × len | s = 1 | 4 | 16 | 32 | 64 | rule |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 × 131072 | 131.3 | 37.1 | 12.5 | 8.5 | **6.1** | 64 |
| 4 × 131072 | 142.6 | 36.1 | 12.3 | 8.7 | **8.5** | 64 |
| 4 × 32768 | 34.3 | 12.3 | 6.1 | **5.4** | 6.4 | 64 in a loop; 1 straight-line |
| 4 × 8192 | 5.9 | 6.1 | 4.4 | 4.1 | 4.1 | 32 in a loop (+2.0 vs +0.5 us launch: ≈ even) |
| 32 × 8192 | **5.9** | 6.7 | 9.7 | 14.6 | 15.6 | 1 |
| 4 × 2048 | **2.3** | 4.4 | 3.3 | 3.3 | — | 1 |

The first slice build combined a row's partials in one lane's chain: 15 ns a partial (s = 64 lost to s = 16 at
4 × 32768); combining one a lane as a tree made the restore one load latency, the term the model prices.

## Time (ms; suite `LADDER=1 gpu_run.sh` ON / OFF / ON, then the final build; ABBA 8 rounds = 16 processes an arm)

| workload | OFF | ON run 1 / run 2 / final | best competitor | ABBA ON/OFF median, min (null A / B) | explanation |
| --- | ---: | ---: | --- | --- | --- |
| lsm_1_131072 | 0.1943 | 0.0494 / 0.0669 / 0.0552 | 0.0236 torch softmax compiled (0.0246 compile of the program's math) | **0.290**, 0.206 (8.5 % / 58.8 %) | flow kernel 182 → 7.6 us (one row on 1 SM → 64 blocks); the ON spread is the first cooperative launch's host setup (39–53 us) inside the one-iteration region. Mapal 2.3× torch (was 8.2×) |
| lsm_4_32768 | 0.0509 | 0.0516 / 0.0541 / **0.0509** | 0.0184 torch softmax | 1.010, 0.804 (12.7 % / 55.7 %) — first rule | device −28 us, host +38 us (first cooperative launch): null → `coop_first_ns`; final module = OFF |
| lsm_32_8192 | 0.0240 | 0.0243 / 0.0241 / 0.0242 | 0.0102 torch-compile | 0.981, 0.993 | identical modules |
| attn_1024_64 | 0.0425 | 0.0419 / 0.0398 | 0.0287 torch-compile; SDPA 0.0430, SDPA compiled 0.0287 | — | identical modules (SDPA: exact exp, not gated) |
| attn_2048_64 | 0.0798 | 0.0796 / 0.0798 | 0.0748 torch-compile; SDPA 0.1014 / 0.0788 | — | identical modules |
| attn_2048_128 | 0.1354 | 0.1364 / 0.1352 | 0.1034 torch-compile; SDPA 0.1331 / 0.1014 | — | identical modules |
| softmax_rows_2048 | 0.4779 | 0.4796 / 0.4754 | 0.471 torch-compile | — | identical modules (2048 rows: one block a row fills the part) |
| softmax_rows_4096 | 3.821 | 3.802 / 3.803 | 3.545 torch-compile | — | identical modules |
| llm_dec_b1_l8192_pv GPU-only (16 steps) | 21.82 | 21.81 / 21.81 | 0.344 ms/step torch-compile (model-scale.md) | — | identical modules: 1.363 ms/step; its `32 × 8192` softmax unit stays one block a row |
| llm_dec_b1_l8192_pv shipped (16 steps) | 2238 | 2147 / 2328 | as above | — | identical modules; attention on the host (model-scale findings 2–4), ±4 % run to run |
| llm_pre_t2048_pv (all device) | 86.58 | 86.57 / 86.61 | 8.322 torch-compile (model-scale.md) | — | identical modules: q·Kᵀ / P·V are cell kernels, no unit holds the rows |
| llm_pre_t8192 | — | — | 83.71 torch-compile | — | no device form (model-scale finding 10): not run |

Probe value gate: 0 programs failed in every run. Evidence checker: 2436 checks, the ON and OFF mismatch sets equal
(105 pre-existing tile-kernel mismatches — `face-registers`, `kmajor-a`, `streamk` — untriaged as before) but for one
`choice` row of attn_1024_64 (tuning noise on an identical module).

## Tests

- `crates/mapal-rewrite/tests/fold_combine.rs` (4): attention's laws (`mx` none, `den` Phase on `mx`, `o` Factor ÷ on
  `den` + Phase on `mx`; stored: the second query equal); norm / shift / phase laws (`rinv` Factor ×, `ss` Shift −,
  `x ÷ (ss + 100)` Phase, `ss ÷ (x + 100)` Factor ×, a max over `x · ss` Phase); whole folds read their row value as a
  number (Phase); 120 generated `long_row` + 40 twin + 40 wide-chain programs: every row / whole / tile site has its
  combine, every dep's law matches its step — 405 sites, 3 Factor / 10 Shift / 41 Phase deps (each required > 0).
- `crates/backends/llvm/tests/long_rows.rs` (4): the sliced softmax (3 × 1024 rows scaled per row, 1 × 2048) at
  s = 2, 4, 8 — 3 phases, 2 barriers, one cooperative launch, values = interpreter within the D2 tolerance under free
  threads, `MAPAL_EMU_SERIAL=1` / `2` and the cooperative grid capped at 1 and 2 blocks (every block strides); the OFF
  arm and the exact face one block a row; the three laws (2 phases each); 160 generated `testgen::long_row_strategy`
  programs (rows 1–8 of 256–4096, folds / law cells / per-row cells, straight-line or looped) × s = 2, 8 × 3 modes =
  interpreter and the OFF arm — 136 sliced units, 324 runs; coverage, each required > 0: 136 multi-phase, 10 three-phase,
  4 factor + 2 shift laws applied, 80 stored folds, 42 stored row cells, 78 per-row cells, 44 one-row units, 48 looped;
  the rule on measured properties (1 × 131072 → 64; 4 × 32768 → 64 in a loop, none straight-line; 4 × 2048, 32 × 8192,
  the rtx-4090 (no `row_iter_ns`) and the OFF arm → none).
- Bugs put back: the barrier between a block's items removed → 8+ generated programs give wrong values; the laws not
  applied after the combine → `laws_drop_phases` fails.
- Suites (mapal-ir, mapal-rewrite, llvm backend, cli; release): 64 binaries, **724 passed, 0 failed**; after the
  `coop_first_ns` change the backend unit tests and the profile / calls / site-geometry / plan-cost / long-rows
  binaries again (121 passed) and the long-rows rule test.

## Open

1. **Streamed (flash-style) row-block units.** The fact gives attention's laws (Phase on `mx`, Factor ÷ on `den`), but
   no consumer streams a tile phase over key blocks: rows longer than the shared-memory budget still leave a row-block
   group to the split kernels. With no exponential law it is two passes (q·Kᵀ recomputed or stored once); at 1024–2048
   keys (L2-resident scores) that costs more than it saves (forced unit 25.7–28.4 us vs split 30.8 us, kernel-merging
   Rows, already one pass), so it needs model-scale attention as device tile sites first (model-scale findings 2–10).
2. `row_iter_ns` is measured with L2-resident rows; a DRAM-resident row walk (higher latency) is priced at the same rate
   (conservative: it under-states the one-block-a-row time). Only the rtx-5080 has it; other parts slice nothing.
3. `coop_first_ns` is charged to a unit outside every loop whatever other cooperative launch the program makes (the
   `calls` step's persistent kernels pay it too, unpriced there); a loop's units are charged none (its trips share it).
4. No `evidence.py` intent for `row-slices` yet (realized checked by hand above: launch entry, barrier count, slices).
5. Split-row parallelism covers flow units; row folds outside a unit (`rowfold`, one warp or thread a row), whole folds
   (a cell that reads a number capture runs alone) and per-row cells with a nested fold (model-scale's RMSNorm `rinv`)
   do not slice.
6. **Pre-existing bug the generator found** (not this step's code; `fc56a298` emits the same): a loop whose carried
   array's next value does not read the array returns the initial value on the device (`.claude/raw-evidence/…/tests/
   fail1b.mapal`: `mut x <- b0; loop { x = b0 / 2 + a0 }` prints `b0`; the interpreter `b0 / 2 + a0`). The long-row
   generator's next state reads the state.
