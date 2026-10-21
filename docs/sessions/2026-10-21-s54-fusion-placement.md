# S54 — fusion as placement, five architectures

Date: 2026-10-21 · Branch **`s54-gpu-fusion`** (= `s54/arch-select` + wip commit; local, not pushed) · base `s53-gpu-opt` @ 6d105b0.
Workflow `wf_25b2dee4-01e` (script `~/.claude/projects/-Volumes-LessComplex-Personal-mapal/5796dddd-f57b-41ae-ba33-9afb66b7242d/workflows/scripts/mapal-s54-fusion-placement-wf_25b2dee4-01e.js`)
**stopped by user (out of tokens)** during arch-select's final 4090/5090 table-verify sessions; verify:arch-select and integrate never ran.
Six orphaned boxes destroyed; `vastai show instances` empty at close.

## Done
- Probe before S54: nothing fused except a closed-form prologue (`benches/results-s54-gpu/fusion-probe-static.txt`).
- `s54-base`: Ampere profiles `rtx-3090` (sm_86), `a100` (sm_80); multi-computation probe suite in `gen_tiles.py`
  (`pt_*`: epi, bias, ew, pro, indep, mlp, mlp2, fanout, diamond, residual, rowsum, softmax_rows, twin, loop, decode);
  `fusion_report.py`; library chain harness (eager cuBLAS + kernels, cuBLASLt epilogues, CUTLASS, streams);
  `bundle.sh ONLY=`; placement model `docs/components/backend-nvptx/ARCHITECTURE.md`; plan `plans/plan-s54-fusion-placement.md`.
- Research notes `docs/notes/s54-research-{shapes,fusion,arch,prior}.md`.
- Built, each with OFF arm, 5 parts, adversarially verified (verdict / ships ON):

| item | verdict | ON on | headline (OFF/ON or vs library) |
|---|---|---|---|
| epilogue (per-cell consumers in tile store) | real | all 5 | pt_epi_4096 contract 2.85–6.05× vs OFF; vs cuBLASLt epilogue iter 0.65–0.93, kernel 0.76–1.03 |
| devmap (per-cell maps/chains on device) | real | all 5 | pt_ew_4096 exact 27–110× vs OFF; kernel 0.93–2.02× fused library |
| rowfold (row folds as device sites) | real | all 5; layout: 4090 contract, A100 exact | pt_rowsum_4096 2.8–8.3× vs OFF; vs eager iter 0.52–0.83 |
| arena (one device arena from death points) | real | all 5 | e2e_chain_4096 wall 1.03–1.17× vs OFF |
| retain-resident (+delta-h2d, lane-stride) | real | delta-h2d/lane-stride all 5; retain-resident 3090/A100/H100 | decode loop still 0.05–0.12× eager+cache library |
| streams (independent sites concurrent) | real | A100, 5090; pinned + PDL OFF | pt_twin 1.32× (A100), 1.05× (5090) |
| l2-raster (grid raster, L2 gate, L2 window) | **refuted** (merged anyway with per-arch rows) | raster 5090 contract; window H100 contract | ±1–5%; window loses on 4090 |
| arch-select (merge + per-arch tables) | **not verified** | — | Ampere rung rows from base+confirm; kernels table below |

Kernels after S54 tables (lib ms / Mapal ms, n=1024/2048/4096/8192; source `results-s54-gpu/arch-select-*`, `baseline-tables.md`):

| part | exact vs cuBLAS | contract vs cuBLAS | contract vs CUTLASS | TF32 vs cuBLAS |
|---|---|---|---|---|
| RTX 3090 | 0.54/0.50/0.58/0.53 | 0.78/0.86/0.96/0.84 | 0.79/1.02/0.99/0.93 | 0.89/0.91/1.00/0.79 |
| A100 SXM4 | 0.34/0.41/0.47/0.50 | 0.63/0.80/0.92/0.96 | 0.94/0.94/0.96/0.96 | 0.57/0.69/0.66/0.66 |
| RTX 4090 | 0.63/0.65/0.61/0.59 | 0.77/1.04/1.05/1.07 | 1.32/1.03/1.05/1.06 | 0.77/0.89/0.91/0.93 |
| H100 PCIe | 0.46/0.48/0.59/0.53 | 0.65/0.84/1.03/0.94 | 0.89/1.07/1.11/1.10 | 0.56/0.60/0.67/0.55 |
| RTX 5090 | 0.66/0.55/0.59/0.58 | 0.79/0.86/0.90/0.88 | 1.59/0.98/1.00/1.01 | 0.75/0.85/0.90/0.80 |

## Tests
`cargo test --workspace --release --no-fail-fast`: **1172 passed / 0 failed**. A first run (fail-fast) reported 20 failures in
an early binary and stopped; not reproduced. Suspect: `examples/emit` exists in both mapal-backend-llvm and
mapal-backend-cuda (cargo warns of an output filename collision), so a test that runs `target/release/examples/emit` can
get the wrong binary depending on build order. Also: duplicated `#[test]` at `crates/backends/llvm/src/profile.rs:3123`.

## Open (S55 queue, in order)
0. Flaky test run above: rename one `emit` example, rerun twice.
1. Verify `s54-gpu-fusion` (was arch-select): one integrated 5-part run (`integrated-*-prefix.txt` are partial), confirm 4090/5090 tables.
2. Decide l2-raster: verifier refuted (gains ≤1.6%, no e2e effect) — drop or keep rows.
3. Placement losses: pt_mlp2_4096 4090 Mapal iter 55.1 ms vs cuBLASLt 11.4 ms; decode loop 0.05–0.12× library.
4. Write `docs/performance/s54-gpu-fusion.md` + `s54-per-arch-selection.md` (integrator never ran); reconcile STATUS docs; drift check.
5. S53 queue items still open: exact face 0.34–0.66×, H100/A100 TF32 (ADR-0039 undecided), clock-fence helper bug, eager-init race.

## Resume
```sh
cd /Volumes/LessComplex/Personal/mapal && git checkout s54-gpu-fusion   # (remove worktree .claude/worktrees/wf_25b2dee4-01e-15 first, or work there)
cargo test --workspace --release
S=<scratch>; mkdir -p $S/b && PREV="" bash benches/gpu/bundle.sh $S/b rtx-3090:sm_86 a100:sm_80 rtx-4090:sm_89 h100-pcie:sm_90 rtx-5090:sm_120
PROFILE=1 bash benches/gpu/vast_run.sh <offer> <profile> <sm> $S/b $S/res 30   # per part; vastai destroy instance <id> -y
python3 benches/gpu/table.py $S/res/results-<profile>.txt
git worktree prune   # many stale .claude/worktrees/* from S53/S54 agents
```
