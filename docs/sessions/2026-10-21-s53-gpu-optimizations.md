# S53 — GPU optimizations: eight items, one integrated build

Date: 2026-10-21 · Branch: **`s53-gpu-opt`** (local, not pushed) · base `s52-ptx-leg` @ 1f1ef87
Record: `docs/performance/s53-gpu-optimizations.md` · Results: `benches/results-s53-gpu/` ·
Plan: `docs/components/backend-nvptx/plans/plan-s53-gpu-optimizations.md`

## What was built

Workflow `mapal-s53-gpu-optimizations` (run `wf_40012e53-58d`): one branch per plan item, each built,
measured on RTX 4090 / H100 PCIe / RTX 5090 and adversarially reviewed, then merged here.

| item | branch @ commit | what it is |
|---|---|---|
| `clock-fence` | `s53/clock-fence` @ 9a0c43f | `time` reads cut `topo_order` into segments; `mapal_gpu_sync` before each read on a GPU profile |
| `arch-rungs` | `s53/arch-rungs` @ b7db3cb | `GPU_RUNGS_OFF`: per-part rung table keyed by fill (warps/scheduler) and face; `--gpu-on`; 5090 `max_blocks_per_sm` 24 |
| `resident` | `s53/resident` @ 9398968 | device residency (copy in at the producer, carried state on the device, partial d2h, free at death) + `devgen` |
| `retain-delta` | `s53/retain-delta` @ 6abfc3f | per-iteration write delta of a carried array; recompute only the written rows (KV cache on decode) |
| `host-runtime` | `s53/host-runtime` @ 077968a | `eager-init`: context + module on a thread from `main`'s first call |
| `fma-regs` | `s53/fma-regs` @ 2cfcc52 | `idx32`, `affine`, `pad`, `wide-fma` (8×16 FMA tile) |
| `warp-fill` | `s53/warp-fill` @ 8aa9944 | shrink the register / warp tile until 2 warps per scheduler |
| `wgmma-hopper` | `s53/wgmma-hopper` @ b1a4d04 | H100 TF32 through `wgmma` (swapped operands, rounded A in smem) |

## Integration decisions (code)

- **One rung set**: `GpuRungs` holds all 20 rungs; `GpuArms {on, off}` flips them for measurement
  (`emit --gpu-off=` / `--gpu-on=`, item names `fma-regs` expand). Host rungs (`resident`, `devgen`,
  `retain-delta`, `eager-init`) read the same table with `Fill::Any` rows (`gpu::host_rungs`).
- **The fill key reads the S52 geometry** (no `warp-fill`, no `wide-fma`): the table's rows were measured
  on grids classified by it.
- **`wide-fma` × `warp-fill`**: a wide tile whose grid is under `fill_warps` falls back to the S52 tile,
  which `warp-fill` then shrinks — the geometry both items measured. 5090 2048² is that case.
- **Unmeasured profiles take the measured part of their architecture** (`cuda-ada` ← 4090, `h100` SXM ←
  H100 PCIe); the intersection only when none exists.
- **`resident` yields to a retained site**: a function where `retain-delta` fires keeps S52 round trips
  (the arm `retain-delta` was measured on). Planning both together is open.
- **`devgen` refuses a float `%`** (review finding: host `fmodf` vs NVPTX `frem` expansion differ in
  2051/4096 elements). `fmod_64` added to the differential.
- **`idx32` stays off tensor-core kernels**: merged with `wgmma`, an `i32` index broke `opt` on the H100
  1024 TF32 kernel. Pinned in `gpu_wgmma.rs`.
- **`pool` deleted** (measured negative on every part; the GpuRungs rule).
- GPU declarations are emitted per called entry (a `resident` module no longer declares `mapal_gpu_tile`);
  the clock fence's sync is kept whenever the module talks to the runtime (tile or split entries).
- Runtime: `resident`'s split entries + `eager-init`'s init thread + the fence's sync in one
  `mapal_gpu.c`; one trace total line per process (`load` = the wait, plus `lead/init/ctx/module`).
- Bench harness: one `box_run.sh` (differential over every arm, trap check, e2e host arms incl. the S52
  host object, decode arms, every kernel arm), `bundle.sh` `arms` drops arms identical to the shipped
  emission, `table.py` one parser for all sections.

## Table decisions (beyond the item branches)

The integrated run (every rung's arm on all three parts) reversed or added these rows; each shipped kernel
after the flips is byte-identical to a measured arm (record §3c):

- 4090: `unroll` back ON at filled contract (+8 / +17 / +9% at 2048 / 4096 / 8192 on the `wide-fma` tile);
  `async` OFF at filled exact (register staging wins once `fma-regs` is on).
- `retain-delta` OFF on all three: on decode `resident` alone is 1.3–1.8× faster than `retain-delta` with round
  trips (resident yields to it). Kept as a rung.
- H100 PCIe: `min-blocks` OFF filled (the 2-block cap spills under `idx32`+`pad`); `affine` OFF on exact and
  under-filled contract, ON filled contract (positive only with `min-blocks` off — confirmed in a second session);
  `warp` OFF under-filled contract; the `ldmatrix` filled row dropped (review finding).
- 5090: `async` OFF under-filled (on `warp-fill`'s 4×4 tile it costs 10%); TF32 `wide` OFF under-filled
  (−21%); `warp-fill` OFF filled contract (TF32 2048 −2.7%); `wide-fma` OFF; `idx32`/`affine`/`pad` OFF on
  the exact face.
- `fma-regs` sub-rungs per part: 4090 all four on both faces (except `idx32`/`affine` under-filled contract);
  H100 `idx32`+`pad` contract filled, `affine` filled contract; 5090 `idx32`+`affine`+`pad` contract filled.

## Results

Record: `docs/performance/s53-gpu-optimizations.md`. Shipped table, n = 2048 / 4096 / 8192, medians
(`~` = ranges overlap):

| part | face | vs cuBLAS | vs CUTLASS | S52 kernel vs cuBLAS (same run) |
|---|---|---|---|---|
| RTX 4090 | exact | 0.66 / 0.61 / 0.60 | 0.64 / 0.61 / 0.60 | 0.64 / 0.59 / 0.56 |
| RTX 4090 | contract | **1.04~ / 1.05~ / 1.05~** | 1.02~ / 1.05~ / 1.02~ | 0.93 / 0.87 / 0.84 |
| RTX 4090 | TF32 | 0.89 / 0.91 / 0.94~ | 0.90 / 0.93 / 0.95~ | 0.89 / 0.91 / 0.93 |
| H100 PCIe | exact | 0.48 / 0.59 / 0.53 | 0.63 / 0.66 / 0.66 | 0.44 / 0.44 / 0.41 |
| H100 PCIe | contract | 0.85 / 1.03~ / 0.95~ | **1.11 / 1.14 / 1.12~** | 0.73 / 0.88 / 0.80 |
| H100 PCIe | TF32 (`wgmma`) | 0.62 / 0.65 / 0.56 | 0.96 / 1.08~ / 1.19 | 0.50 / 0.42 / 0.34 |
| RTX 5090 | exact | 0.54 / 0.58 / 0.58 | 0.62 / 0.64 / 0.67 | 0.54 / 0.58 / 0.57 |
| RTX 5090 | contract | 0.87 / 0.91~ / 0.92~ | 0.99~ / 1.00~ / 1.01~ | 0.80 / 0.84 / 0.81 |
| RTX 5090 | TF32 | 0.85 / 0.90 / 0.84 | 0.88 / 0.90 / 0.91~ | 0.85 / 0.90 / 0.84 |

n = 1024 (warp-fill): 4090 TF32 1.65× the S52 kernel, 5090 TF32 1.89×, 5090 contract 1.29×, H100 exact 1.23×.

Whole programs (`e2e_4096`, process wall, S52 host object → S53, same interleaving): 4090 431.6 → 298.0 ms,
H100 687.7 → 523.5 ms, 5090 258.4 → 198.6 ms (exact face); bytes in 128 MB → 0, out 64 MB → 4 B; own `iter ms`
21.9 / 21.1 / 18.0 (S52: the input generation) → 5.3 / 6.9 / 3.6 (the site). `eager-init` alone 1.17–1.37×.
Decode (T=1024, d=256): S52 round trips 2.41 / 2.47 / 1.26 s → shipped (`resident`) 0.905 / 1.109 / 0.630 s.

Correctness: differential 126/126 on every part (session 1), 134/134 (H100, 5090 confirmations); trap-before-site
10/10; every exact/contract bench bit-exact, every TF32 bench within the bound. `cargo test --workspace --release`
1101 passed / 0 failed / 1 ignored.

## Live state at close

- Branch `s53-gpu-opt` (local, not pushed); item branches `s53/*` and their worktrees
  (`.claude/worktrees/wf_40012e53-58d-{6..13}`) are left as they were.
- `vastai show instances`: empty (every rental destroyed by `vast_run.sh`; 4090 56989848/56992147, H100
  56989847/56992148/56994545, 5090 56989846/56992149/56993611/56994032/56995310).
- Results: `benches/results-s53-gpu/integrated-{rtx-4090,h100-pcie,rtx-5090}.txt` (session 1, every arm) and
  `integrated-{h100-pcie,rtx-5090}-confirm.txt` (the flipped table). Profile tarballs (ptxas, SASS, nsys) and raw
  outputs: `/private/tmp/claude-501/-Volumes-LessComplex-Personal-mapal/5796dddd-f57b-41ae-ba33-9afb66b7242d/scratchpad/s53/integrate/results*/`.
- Bundles: `.../scratchpad/s53/integrate/bundle` (session 1, three parts), `bundle-h2` (H100 confirm),
  `bundle-r2` (5090 confirm). The first attempt (`results-fail1/`) died on a `tile_bench.cu` merge break, fixed in
  7b32aa4.

## Resume

```sh
git checkout s53-gpu-opt
cargo test --workspace --release
cargo build --release -p mapal-backend-llvm --example emit && cargo build --release -p mapal-interp --example run
S=<scratch>; mkdir -p $S/b/tiles && cp -R <bundle7>/cutlass $S/b/ && cp <bundle7>/tiles/*.expect $S/b/tiles/
PREV="s52=<bundle7>" bash benches/gpu/bundle.sh $S/b rtx-4090:sm_89 h100-pcie:sm_90a rtx-5090:sm_120
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 rentable=true cuda_max_good>=12.8 reliability>0.97 inet_down>1000 disk_space>=40' -o 'dph+' --limit 3
PROFILE=1 bash benches/gpu/vast_run.sh <offer> rtx-4090 sm_89 $S/b $S/res 30
python3 benches/gpu/table.py $S/res/results-rtx-4090.txt
vastai show instances        # destroy leftovers: vastai destroy instance <id> -y
```

`bundle.sh` takes ~15 min per profile (the 64 KB loop products compile slowly in clang); profiles can
build in parallel into separate directories and be moved into one bundle (`bundle7` =
`/private/tmp/claude-501/-Volumes-LessComplex-Personal-mapal/5796dddd-f57b-41ae-ba33-9afb66b7242d/scratchpad/gpu/bundle7`).

## Open (queue for S54)

1. **`resident` × `retain-delta`**: a retained site keeps the output rows on the device and recomputes only the
   written rows there. Today the function goes to one or the other; `resident` wins (decode 0.63–1.11 s).
2. **Whole-program wall is the context**: 75–245 ms of the 200–520 ms `e2e_4096` wall is the wait for the
   context (eager-init already overlaps the rest); the cuBLAS whole-program floor is 174–448 ms. `resident` still
   allocates and pages host buffers for device-only objects (64 MB each at 4096).
3. **Exact face 0.48–0.66× cuBLAS everywhere** (two instructions per MAC). H100 exact: `pad` +4 / +2% at
   1024 / 2048 and −9% at 8192 in one session — not shipped.
4. **H100 TF32 0.56–0.65× cuBLAS** (cuBLAS: sm90 xmma 128×128×32, one warpgroup). TMA, a deeper `wgmma` pipeline,
   and bf16 storage wait on ADR-0039.
5. **5090 contract 0.87–0.92× cuBLAS**; `wide-fma` loses on Blackwell (no ncu on vast to say why).
6. **clock-fence**: an inlined helper's morphisms keep the callee's source location, so `own()` places a site in
   the wrong segment (review repro `.../scratchpad/s53/review-clock-fence/{above,below}.mapal`); the segment check
   is a `debug_assert` the release suite never runs. No timed program defines a helper today.
7. **eager-init**: on a host without a GPU (or `MAPAL_PTX` unset) the init thread's `exit(102)` can race a program
   that traps before its first site (101); the `atexit` order against libcuda's handlers is unverified.
8. **Unmeasured profiles** (`cuda-ada`, `h100` SXM) inherit their architecture's rows — measure an SXM and a 4070 Ti.
9. `bundle.sh` ~15 min per profile: loop products of 64 KB stay SSA aggregates (below `heap_min_bytes`) and clang
   takes ~1 min per such host object.
10. S52's open items not touched: fusion as placement, concurrent sites, canonical-tree float folds (ADR-0028).
