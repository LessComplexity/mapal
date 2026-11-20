#!/bin/bash
# model-scale timed run on the local RTX 5080 (benches/results/evidence/local-rtx-5080/model-scale.md): box_run.sh
# over a bundle (benches/local/gpu_bundle.sh with the model-scale stems), then the summary table.
# Machine protocol: the run holds the shared bench lock (BENCH_LOCK) for its whole length; it starts only with no
# game process (MapleStory.exe), `sensors` Tctl under 85 C and CPUs 0-7 under 10 % busy (/proc/stat over 2 s), and
# every timed process runs pinned to CPUs 0-7 (`taskset`): another agent's builds load CPUs 8-31 / 40-63, so the
# usual "load < 3" wait never ends. The conditions at the start and end land in <results>/conditions.txt.
# usage: model_scale_run.sh <bundle> <results dir> [reps]   (TORCH_PY, TORCH_MIN, TORCH, TORCH_MODES, PROBE_REPS,
# PROBE_ARMS pass through; LADDER=1 by default: a program whose tune keeps candidate 0 is timed too, not `carried`)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
B=$(cd "$1" && pwd); R=$2; REPS=${3:-30}
LOCK=${BENCH_LOCK:?BENCH_LOCK=<the shared bench lock file>}
CPUS=${CPUS:-0-7}
mkdir -p "$R"; R=$(cd "$R" && pwd)
busy() {  # % busy of CPUs 0-7 over 2 s
    local a b
    a=$(awk '/^cpu[0-7] / { t += $2+$3+$4+$5+$6+$7+$8+$9; i += $5+$6 } END { print t, i }' /proc/stat); sleep 2
    b=$(awk '/^cpu[0-7] / { t += $2+$3+$4+$5+$6+$7+$8+$9; i += $5+$6 } END { print t, i }' /proc/stat)
    awk -v a="$a" -v b="$b" 'BEGIN { split(a, x, " "); split(b, y, " "); dt = y[1] - x[1]; di = y[2] - x[2];
        print (dt > 0) ? int(100 * (dt - di) / dt) : 100 }'
}
tctl() { sensors 2>/dev/null | awk '/^Tctl/ { gsub(/[+°C]/, "", $2); print int($2) }'; }
why_not() {  # empty when the machine is ready
    pgrep -x MapleStory.exe > /dev/null && { echo "MapleStory.exe running"; return; }
    local t; t=$(tctl); [ "${t:-99}" -lt 85 ] || { echo "Tctl=${t}C"; return; }
    local u; u=$(busy); [ "$u" -lt 10 ] || echo "cpu$CPUS busy=${u}%"
}
cond() { echo "$1 $(date -u +%FT%TZ) tctl=$(tctl)C cpu${CPUS}_busy=$(busy)% maple=$(pgrep -x MapleStory.exe > /dev/null && echo yes || echo no)" \
    "gpu=[$(nvidia-smi --query-gpu=utilization.gpu,clocks.sm,temperature.gpu,power.draw,memory.used --format=csv,noheader)]" \
    "loadavg=[$(cut -d' ' -f1-3 /proc/loadavg)]" >> "$R/conditions.txt"; }
exec 9> "$LOCK"
while :; do
    w=$(why_not); [ -z "$w" ] || { echo "waiting: $w"; sleep 30; continue; }
    flock 9
    w=$(why_not); [ -z "$w" ] && break
    flock -u 9; echo "waiting (after the lock): $w"; sleep 30
done
cond start
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/gpu-apps-before.csv"
(cd "$B" && taskset -c "$CPUS" env PATH=/opt/cuda/bin:$PATH CUDA=/opt/cuda TORCH_PY="${TORCH_PY:?TORCH_PY}" \
    TORCH_MIN="${TORCH_MIN:-180}" PROBE_REPS="${PROBE_REPS:-5}" TIMING=0 LADDER="${LADDER:-1}" \
    bash box_run.sh rtx-5080 sm_120 "$REPS" > "$R/box.out" 2>&1)
rc=$?
cond end
flock -u 9
cp "$B"/results-rtx-5080.txt "$B"/evidence-rtx-5080.tsv "$B"/wisdom-rtx-5080.tsv "$R/" 2>/dev/null
python3 "$ROOT/benches/local/model_scale_report.py" "$R/results-rtx-5080.txt" > "$R/model-scale.md"
echo "box_run rc=$rc $(grep -E '^differential:|^probe value gate|VALIDATION' "$R/results-rtx-5080.txt" | tr '\n' ' ')"
