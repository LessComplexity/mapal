#!/bin/bash
# Full timed local run on the RTX 5080: box_run.sh inside the bundle, then results + evidence + the gap table.
# usage: run.sh [bundle dir] [reps]   -> $G/results/{results,evidence}-rtx-5080.{txt,tsv}, gap.md
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; G=${LOCAL_GPU:-$ROOT/target/tmp/local-gpu}
B=${1:-$G/bundle}; REPS=${2:-30}; R=$G/results
mkdir -p "$R"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/gpu-apps-before.csv"
nvidia-smi --query-gpu=utilization.gpu,clocks.sm,temperature.gpu,power.draw --format=csv >> "$R/gpu-apps-before.csv"
cd "$B" && env PATH=/opt/cuda/bin:$PATH CUDA=/opt/cuda TORCH_PY=${TORCH_PY:-$G/venv/bin/python} TORCH_MIN=${TORCH_MIN:-120} COMP_MIN=${COMP_MIN:-60} \
    bash box_run.sh rtx-5080 sm_120 "$REPS" > "$R/box.out" 2>&1
rc=$?
cp results-rtx-5080.txt evidence-rtx-5080.tsv wisdom-rtx-5080.tsv "$R/" 2>/dev/null
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/gpu-apps-after.csv"
python3 "$ROOT/benches/local/gap.py" "$R/results-rtx-5080.txt" > "$R/gap.md"
echo "box_run rc=$rc  $(grep -E '^probe value gate|VALIDATION' "$R/results-rtx-5080.txt" | tr '\n' ' ')  $(tail -1 "$R/evidence-rtx-5080.tsv" 2>/dev/null)"
tail -n +1 "$R/gap.md" | head -60
