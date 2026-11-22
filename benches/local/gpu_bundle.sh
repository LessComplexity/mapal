#!/bin/bash
# Local RTX 5080 bundle: the README "GPU workloads" suite (default face, shipped build + null) + two GEMM cells.
# usage: bundle.sh <outdir> [stems...]   (stems default: the README table's 35 programs, benches/local/suite.txt)
# STEMS= in the environment (empty too) replaces the stems (gpu_changed.sh: only the programs an A/B moves); CELLS= and
# ARMS= (BASE_EMIT=) pass through. Built objects come from the bundle cache where a stem's emitted text is unchanged
# (integrated_bundle.sh, BUNDLE_CACHE=0 builds everything).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=$1; shift
SUITE=${STEMS-${*:-$(tr '\n' ' ' < "$ROOT/benches/local/suite.txt")}}
mkdir -p "$OUT"
# The boxes' CUTLASS (b2f249a), copied so integrated_bundle.sh does not clone HEAD.
[ -d "$OUT/cutlass" ] || cp -R "${CUTLASS_SRC:-$ROOT/.claude/sprints/per-site-choice/scratch/harness/bundle/cutlass}" "$OUT/"
cd "$ROOT"
export PATH=/opt/cuda/bin:$PATH CUDA=/opt/cuda LLVM=/usr/bin CPU_TARGET=x86-64-v3
CELLS=${CELLS-"4096x4096x4096 16x2048x7168"} PROGS="" SUITE="$SUITE" SUITE_ARMS=0 ARMS="${ARMS-}" \
    benches/gpu/integrated_bundle.sh "$OUT" rtx-5080:sm_120 > "$OUT/bundle.log" 2>&1 || { tail -20 "$OUT/bundle.log"; exit 1; }
grep -h '^bundle cache:' "$OUT/bundle.log" || true
python3 benches/gpu/preflight.py "$OUT" rtx-5080 2>&1 | tee "$OUT/preflight.log" | tail -5
