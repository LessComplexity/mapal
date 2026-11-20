#!/bin/bash
# model-scale: one timed run per program, for the natural forms whose shipped placement runs the attention on the host
# (model-scale.md findings 2-4: 45 s - 9 min per run, so box_run.sh's value gates over every site candidate, tune
# run and repetitions would take hours). Same machine protocol as model_scale_run.sh (bench lock, CPUs 0-7 idle and
# pinned, no game, Tctl < 85 C). The values answer to the `_pv` twin's exact-face oracle (`tiles/<twin>_x.expect`,
# the device's exact face: the twins compute the same function, only the RoPE indices are written otherwise)
# within ADR-0040 D2's bound (value_gate.py, `tiles/<stem>.kmax`); a stem without a twin oracle prints `gate=none`.
# usage: model_scale_once.sh <bundle> <results dir> <stem>...   -> <results>/once-rtx-5080.txt
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
B=$(cd "$1" && pwd); R=$2; shift 2
LOCK=${BENCH_LOCK:?BENCH_LOCK=<the shared bench lock file>}
CPUS=${CPUS:-0-7}
mkdir -p "$R"; R=$(cd "$R" && pwd); LOG=$R/once-rtx-5080.txt
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
cd "$B" || exit 1
exec 9> "$LOCK"
for s in "$@"; do
    cc "rtx-5080/$s.host.o" mapal_gpu.c libmapal_rt.a -I/opt/cuda/include -L/opt/cuda/lib64/stubs -lcuda -lpthread -ldl \
        -lm -o "once_$s" || { echo "once prog=$s LINK FAILED" | tee -a "$LOG"; continue; }
    while :; do
        w=$(why_not); [ -z "$w" ] || { echo "waiting: $w"; sleep 30; continue; }
        flock 9; w=$(why_not); [ -z "$w" ] && break
        flock -u 9; sleep 30
    done
    c0="tctl=$(tctl)C cpu${CPUS}_busy=$(busy)%"
    got=$(MAPAL_WISDOM=$B/wisdom MAPAL_GPU_TUNE=0 MAPAL_PTX="rtx-5080/$s.ptx" taskset -c "$CPUS" "./once_$s" 2>/dev/null | tr '\n' ' ')
    rc=$?
    flock -u 9
    vals=${got%%iter ms=*}; twin=${s}_pv; g=none
    if [ -s "tiles/${twin}_x.expect" ] && [ -s "tiles/$s.kmax" ]; then
        g=$(python3 value_gate.py "tiles/${twin}_x.expect" "$vals" "$(cat "tiles/$s.kmax")") && g="pass ($g)" || g="FAIL ($g)"
    fi
    echo "once prog=$s rc=$rc out=[$got] gate=$g K=$(cat "tiles/$s.kmax" 2>/dev/null) $c0" | tee -a "$LOG"
done
