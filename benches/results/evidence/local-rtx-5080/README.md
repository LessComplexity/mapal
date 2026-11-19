# local-rtx-5080 — first local competitor run (2026-11-19)

Machine: AMD Ryzen Threadripper 9970X (Zen 5, 32C/64T, AVX-512, 4 × 32 MB L3), RTX 5080 (sm_120, 84 SMs,
64 MB L2, 16 GB), Arch Linux, clang/LLVM 23, CUDA 13 at `/opt/cuda`, driver 620.70. The GPU also drives the
display; a game was closed before timing, Docker Desktop's VM and a browser stayed up (~9 % CPU each).

## Commands

| pass | command |
| --- | --- |
| GPU bundle | `benches/local/gpu_bundle.sh <bundle>` (README "GPU workloads" suite, 35 programs + 2 cells, `rtx-5080` profile) |
| GPU run | `LADDER=1 LOCAL_GPU=<dir> benches/local/gpu_run.sh <bundle>` (`LADDER=1` times "carried" programs too; run 1 lacked it, so 7 rows are run 2 only) |
| CPU matmul | `CYCLES=15 PYTHON=<venv>/bin/python benches/local/cpu_matmul.sh <out>` |
| CPU shapes | `CYCLES=25 PYTHON=<venv>/bin/python benches/local/cpu_shapes.sh <out>` |
| CPU NN | `TARGET=native NN_CFLAGS="-O2 -march=native" NN_CONF_FLAGS=--no-contract PAR=64 CYCLES=41 OUT=<out> PYTHON=<venv>/bin/python benches/nn/nn_ab.sh` |

Competitors: cuBLAS / cuBLASLt / CUTLASS `b2f249a` (stock + tuned) / hand CUDA (chain_bench), PyTorch 2.11.0+cu130
eager and `torch.compile` (TF32 rows with TF32 on). CPU: NumPy 2.5.5 on scipy-openblas 0.3.34 in a venv (the
system NumPy links reference cblas, 250 ms at 1024² — not a valid opponent), C++ naive 1t / mt.

## Files

`gpu/gap-run1.md`, `gpu/gap-run2.md` (Mapal / best other per workload, null), `gpu/evidence-rtx-5080.tsv`
(run 2: 10695 checks, 401 mismatch, 66 n/a — not yet triaged), `gpu/wisdom-rtx-5080.tsv`, GPU app snapshots;
`cpu/matmul-summary.md`, `cpu/shapes-summary.md`, `cpu/nn-report.txt`.

## Known invalid / open

- **The CPU NN NumPy leg was a cold single iteration per process** (`nn_numpy.py`, no warm-up). Warm, NumPy nn2
  d=1024 is 1.52 ms vs 8.70 ms in the harness, so the NN "1.3–5.4× Mapal" ratios were a boundary artifact. The
  matmul NumPy leg was warm (`numpy_bench.py` runs 3 untimed passes); its ratios stand at 64 threads. Corrected
  tables (warm NumPy, each side at its best thread count — OpenBLAS is faster at 32 threads): `cpu-analysis.md`.
- Mapal threaded < Mapal 1t on conv2d / saxpy / reduce: the Linux `native` profile had no dispatch-cost facts
  (fixed in 1ffe0137, `cpu-analysis.md`).
- GPU: no timing difference here is explained from SASS yet (CLAUDE.md Explain rule): every ratio is "open".
  `rtx-5080` is an UNMEASURED profile (no DeviceCost; sm_120 rung rows come from rtx-5090).

Raw output (results txt, box.out, CPU logs) is kept locally in `.claude/raw-evidence/local-rtx-5080/`; not yet
uploaded with `benches/gpu/archive_raw.sh` (that publishes a GitHub release).
