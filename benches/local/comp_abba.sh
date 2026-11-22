#!/bin/bash
# competitors: same-box ABBA of two competitor legs of one stem — the old best and the new best of a row whose best
# leg changed (matrix.md "What the new legs change"): A B B A, one process per leg run (a torch_bench.py mode, or a
# comp_bench.py leg written `comp=<leg>`), inside a timed run's bundle (its `vals/`). Same machine protocol as
# model_scale_run.sh (bench lock for the whole pass, no game, Tctl < 85 C, pinned to CPUS). Prints `abba prog= pos=
# leg= med_ms=` lines, then per stem `abba prog= a= b= a_med_ms= b_med_ms= b/a=` (each leg's two runs' medians).
# usage: comp_abba.sh <bundle> <out file> <stem>:<legA>:<legB>...   (BENCH_LOCK, TORCH_PY; CPUS=0-7, REPS=30)
set -uo pipefail
B=$(cd "$1" && pwd); OUT=$2; shift 2
LOCK=${BENCH_LOCK:?BENCH_LOCK}; CPUS=${CPUS:-0-7}; REPS=${REPS:-30}
mkdir -p "$(dirname "$OUT")"; OUT=$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")  # the passes run in the bundle
tctl() { sensors 2>/dev/null | awk '/^Tctl/ { gsub(/[+°C]/, "", $2); print int($2) }'; }
exec 9> "$LOCK"
while :; do
    flock 9
    pgrep -x MapleStory.exe > /dev/null || [ "$(tctl)" -ge 85 ] || break
    flock -u 9; echo "waiting: game or Tctl"; sleep 30
done
echo "abba start $(date -u +%FT%TZ) tctl=$(tctl)C loadavg=[$(cut -d' ' -f1-3 /proc/loadavg)]" >> "$OUT"
cd "$B" || exit 1
one() {  # <stem> <leg> -> its med_ms
    local s=$1 leg=$2 line
    if [ "${leg#comp=}" != "$leg" ]; then
        line=$(COMP_LEGS="${leg#comp=}" taskset -c "$CPUS" "$TORCH_PY" comp_bench.py "$REPS" vals "$s" 2>/dev/null | grep '^comp prog=')
    else
        line=$(TORCH_MODES="$leg" taskset -c "$CPUS" "$TORCH_PY" torch_bench.py "$REPS" vals "$s" 2>/dev/null | grep '^torch prog=')
    fi
    echo "$line" | grep -o 'med_ms=[0-9.]*' | head -1 | cut -d= -f2
}
for spec in "$@"; do
    IFS=: read -r s a b <<< "$spec"
    declare -A got=()
    i=0
    for leg in "$a" "$b" "$b" "$a"; do
        i=$((i + 1))
        m=$(one "$s" "$leg")
        echo "abba prog=$s pos=$i leg=$leg med_ms=${m:-FAILED}" | tee -a "$OUT"
        got[$leg]="${got[$leg]:-} ${m:-nan}"
    done
    python3 - "$s" "$a" "$b" "${got[$a]}" "${got[$b]}" <<'EOF' | tee -a "$OUT"
import statistics, sys
s, a, b, ga, gb = sys.argv[1:]
fa = [float(x) for x in ga.split() if x != 'nan']
fb = [float(x) for x in gb.split() if x != 'nan']
ma, mb = (statistics.median(fa) if fa else float('nan')), (statistics.median(fb) if fb else float('nan'))
print(f'abba prog={s} a={a} b={b} a_med_ms={ma:.4f} b_med_ms={mb:.4f} b/a={mb / ma:.3f} a_runs={ga.strip()} b_runs={gb.strip()}')
EOF
    unset got
done
echo "abba end $(date -u +%FT%TZ) tctl=$(tctl)C" >> "$OUT"
flock -u 9
