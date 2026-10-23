# Continue from S62 integrated

Branch `s62-integrated` (cut from `4e1102c`), worktree `.claude/worktrees/wf_a3ec2a1a-376-1`. Not pushed; `s54-gpu-fusion` and `oainotes.md` untouched.
Results: [current results](performance/gpu/current-results.md#s62-integrated-6a0e3a0) · gaps and designs: [S62 gap map](performance/gpu/disassembly/s62-gap-map.md) · log: [session](sessions/2026-10-23-s62-optimize-integrate.md).

## State

| item | state |
| --- | --- |
| build | 6a0e3a0 + four disasm evidence commits + this reconciliation |
| tests | `cargo test -p mapal-backend-llvm` 258/0/1; `mapal-cli` 4/0; required-clang workspace 1217/0/1 |
| rentals | none live. Protected, never touch: 57065574, 57100404, 57126023, 57133432, 57285249 |
| OpenSpec | `optimize-ordered-smallm-gpu` 2.1 open (16K controls not measured); `benchmark-large-composed-gpu-workloads` 2.2, 3.1–3.5, 4.x open; `eliminate-unmaterialized-gpu-allocations` 4.x open |

## Worse cells (Mapal / best library, or base/int for MLP)

| part | cell / face | ratio | gap |
| --- | --- | --- | --- |
| all | 4096³ E/R vs cuBLAS | 1.62–1.77 | G1 |
| 5090 | 128x2048 E/R vs cuBLAS | 3.37/3.41 | G1 |
| H100 | 128x2048 C vs cuBLAS | 6.38 | G4 |
| H100 | 128x2048 E vs cuBLAS | 3.36 | G2 |
| H100 | 16x1024 E/C vs cuBLAS | 2.71/3.01 | G2 |
| 3090 | 128x2048 C vs cuBLAS | 3.93 | G4 |
| 3090 | 16x1024 C vs cuBLAS | 2.72 | G2/G4 |
| 5090 | 16x1024 E/C vs cuBLAS | 2.67/2.45 | G2 |
| H100 | 16x18432 E/C vs cuBLAS | 1.43/1.31 | G3 |
| 3090 | 16x18432 rc vs own C | 1.41 | G6 |
| 5090 | MLP E / C base/int | 0.51–0.52 / 0.68–0.88 | G5, G7 |
| 4090 | MLP C base/int | 0.44–0.62 | G5, G9 |
| 3090 | MLP E base/int | 0.47–0.84 | G5, G7 |
| H100 | MLP C d256 base/int; shipped vs cuBLASLt | 0.77; 7.63 | G4, G5 |

## Next, in order (gap map §Order)

| order | work | acceptance |
| --- | --- | --- |
| 0 | M5 `fn-cache` (resolve launch ids on the eager-init thread, set MAX_DYNAMIC_SHARED once) for G5/G9; G7 gates on panel-regs/grid-share | T0 |
| 1 | M3 pipelined mainloop (G3), M4 face-aware tile chooser (G4), f32x2 on sm ≥ 100 (G1) | T1, T2 |
| 2 | M1 split-K sizing with one-wave clamp, M2 ordered acc-fit, MLP reassoc + fusion, L2 retention (G2/G6) | T3, T4 |

## Resume commands

```sh
cd /Volumes/LessComplex/Personal/mapal/.claude/worktrees/wf_a3ec2a1a-376-1
git status --short && git log --oneline -8
openspec list --json
cat docs/performance/gpu/disassembly/s62-gap-map.md
python3 benches/gpu/integrated_report.py benches/results/gpu/s62/integrated/*/results-*.txt
grep -in 's62' crates/backends/llvm/src/profile.rs | head -40   # GPU_RUNGS_OFF rows with cites
MAPAL_REQUIRE_CLANG=1 cargo test --workspace --release --no-fail-fast -- --test-threads=1
supercharge-drift .
vastai show instances   # expect only the five protected ids
```

Measurement rules: one box per part, ABBA, value gate first, `vastai destroy instance <id> -y`, serialize anything that measures.
