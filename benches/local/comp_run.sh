#!/bin/bash
# competitors: re-time competitor legs only (torch_bench.py modes, comp_bench.py legs) inside a timed run's bundle (its
# `vals/` gate them), with model_scale_run.sh's machine protocol: the bench lock for the whole pass, no game, Tctl under
# 85 C, CPUs 0-7 under 10 % busy, every process pinned to CPUS. Legs do not depend on Mapal's commit: matrix.py takes
# each leg from the newest results file that timed it. TORCH_MODES / COMP_LEGS pick the legs (empty: all; `none`:
# skip that harness).
# usage: comp_run.sh <bundle> <results file> [reps] -- <stem>...   (BENCH_LOCK, TORCH_PY; CPUS=0-7)
set -uo pipefail
B=$(cd "$1" && pwd); OUT=$2; REPS=${3:-30}; shift 3; [ "${1:-}" = -- ] && shift
LOCK=${BENCH_LOCK:?BENCH_LOCK}; CPUS=${CPUS:-0-7}
mkdir -p "$(dirname "$OUT")"; OUT=$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")
busy() {
    local a b
    a=$(awk '/^cpu[0-7] / { t += $2+$3+$4+$5+$6+$7+$8+$9; i += $5+$6 } END { print t, i }' /proc/stat); sleep 2
    b=$(awk '/^cpu[0-7] / { t += $2+$3+$4+$5+$6+$7+$8+$9; i += $5+$6 } END { print t, i }' /proc/stat)
    awk -v a="$a" -v b="$b" 'BEGIN { split(a, x, " "); split(b, y, " "); dt = y[1] - x[1]; di = y[2] - x[2];
        print (dt > 0) ? int(100 * (dt - di) / dt) : 100 }'
}
tctl() { sensors 2>/dev/null | awk '/^Tctl/ { gsub(/[+°C]/, "", $2); print int($2) }'; }
why_not() {
    pgrep -x MapleStory.exe > /dev/null && { echo "MapleStory.exe running"; return; }
    local t; t=$(tctl); [ "${t:-99}" -lt 85 ] || { echo "Tctl=${t}C"; return; }
    local u; u=$(busy); [ "$u" -lt 10 ] || echo "cpu$CPUS busy=${u}%"
}
exec 9> "$LOCK"
while :; do
    w=$(why_not); [ -z "$w" ] || { echo "waiting: $w"; sleep 30; continue; }
    flock 9; w=$(why_not); [ -z "$w" ] && break
    flock -u 9; echo "waiting (after the lock): $w"; sleep 30
done
echo "== comp_run start $(date -u +%FT%TZ) tctl=$(tctl)C TORCH_MODES=${TORCH_MODES:-} COMP_LEGS=${COMP_LEGS:-}" >> "$OUT"
cd "$B" || exit 1
[ "${TORCH_MODES:-}" = none ] || taskset -c "$CPUS" env TORCH_MODES="${TORCH_MODES:-}" "${TORCH_PY:?TORCH_PY}" \
    torch_bench.py "$REPS" vals "$@" 2> "$OUT.torch.err" | grep -E '^torch ' >> "$OUT"
[ "${COMP_LEGS:-}" = none ] || taskset -c "$CPUS" env COMP_LEGS="${COMP_LEGS:-}" "$TORCH_PY" comp_bench.py "$REPS" vals \
    "$@" 2> "$OUT.comp.err" | grep -E '^comp ' >> "$OUT"
echo "== comp_run end $(date -u +%FT%TZ) tctl=$(tctl)C" >> "$OUT"
flock -u 9
