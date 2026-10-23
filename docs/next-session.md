# Continue from S63 default float face

Branch `s63/default-float` (cut from `s62-integrated` @ `3ec494e`), worktree `.claude/worktrees/wf_2d840a98-6c0-2`. Not pushed; `s54-gpu-fusion`, `s62-integrated` untouched.
Decision: [ADR-0040](decisions/ADR-0040-default-float-face.md) · results: [current results](performance/gpu/current-results.md#s63-default-float-face-adr-0040) · raw: [`benches/results/gpu/s63/default-float/`](../benches/results/gpu/s63/default-float/) · gaps: [S62 gap map](performance/gpu/disassembly/s62-gap-map.md).

## State

| item | state |
| --- | --- |
| default face | `EmitOpts::default()`: fma steps, `Order::Free`; `--exact`, `--order=sequential`, `--no-contract --order=fixed` opt-in; `mapal test` runs the fixed face, the oracle replays `Emission.orders` |
| rungs | `splitk-atomic`, `streamk` (+ `splitk` on fixed/free); S63 `GPU_RUNGS_OFF` rows: 5090 streamk Under; 4090 streamk Filled; H100 streamk Filled, splitk-atomic Any; 3090 streamk Under+Filled, splitk-atomic Filled |
| tests | see the S63 commit message (`cargo test -p mapal-backend-llvm`, `-p mapal-cli`, required-clang workspace) |
| rentals | none live. Protected, never touch: 57065574, 57100404, 57126023, 57133432, 57285249 |

## Measured, open

| item | measured |
| --- | --- |
| default (= contract) MLP kernels slower than exact on H100 / 3090 | H100 d1024 0.3849 vs 0.1793 ms, 3090 0.3072 vs 0.1802 ms (G4: per-face tile). The default face admits separate steps: a measured steps chooser per part is the next rung |
| wins the S63 keep rule forgoes (a class row) | 5090 128×2048 streamk 0.851/0.863; H100 16×18432 0.748; 3090 16×1024 0.880/0.863, 16×18432 0.924/0.928 — split the fill class by a fact that separates them before re-enabling |
| still behind cuBLAS | H100 128×2048 3.793, 3090 16×1024 2.560, 5090 16×1024 2.298 (default / best library) |
| default beats every library | 4090 4096³ 0.911, H100 4096³ 0.966, 3090 4096³ 0.980, 5090 MLP d256 0.890/0.949 |

## Next

| order | work | acceptance |
| --- | --- | --- |
| 0 | steps chooser (fma vs separate) per part from measurement — the default face permits both | G4 cells + MLP on four parts |
| 1 | gap map order 0–2 (M5 fn-cache, M3 pipeline, M4 face tile, M1 done here, M2) | T0–T4 |
| 2 | stream-K on epilogue / col / stream sites; `red.global.add.v4.f32` (sm_90) | when a measured program needs it |

## Resume commands

```sh
cd /Volumes/LessComplex/Personal/mapal/.claude/worktrees/wf_2d840a98-6c0-2
git status --short && git log --oneline -5
python3 benches/gpu/default_float_report.py benches/results/gpu/s63/default-float/*/results-*.txt
cargo test -p mapal-backend-llvm --lib -- the_chooser --nocapture   # the chooser per part and cell
MAPAL_REQUIRE_CLANG=1 cargo test --workspace --release --no-fail-fast -- --test-threads=1
vastai show instances   # expect only the five protected ids
```

Measurement rules: one box per part, ABBA, value gate first, `vastai destroy instance <id> -y`, serialize anything that measures.
Bundle: `benches/gpu/integrated_bundle.sh <out> rtx-5090:sm_120 rtx-4090:sm_89 h100-pcie:sm_90a rtx-3090:sm_86`; box env
`PROFILE=0 E2E=0 TIMING=0 PROBE_REPS=10 RUN_MIN=45` through `benches/gpu/vast_part.sh`.
