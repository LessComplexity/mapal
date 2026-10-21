#!/bin/bash
# One part's session to completion: rent the cheapest offer of the first GPU type that has one,
# run the bundle (vast_run.sh, PROFILE=1), until a results file ends in `== done`. An offer taken
# or a box that never starts: the next offer; none left: search again every 10 min for 1 h. Two
# boxes that ran the bundle without finishing: stop (the bundle, not the rental, is the problem).
# Only offers whose host link is >= 10 GB/s (the probes time transfers).
# SKIP_HOSTS="<host_id>..." excludes boxes: a confirm session runs on a different box.
#
# usage: benches/gpu/vast_part.sh <profile> <sm_NN> <bundle_dir> <results_dir> <reps> <GPU_NAME>...
#   e.g. benches/gpu/vast_part.sh a100 sm_80 $B $R 30 A100_SXM4 A100_PCIE
PROF=$1 SM=$2 B=$3 RES=$4 REPS=$5; shift 5
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$RES"
offers() { # <GPU_NAME>: id dph pcie_bw cpu_ghz host machine geo, cheapest first
    vastai search offers "gpu_name=$1 num_gpus=1 rentable=true cuda_max_good>=12.8 reliability>0.97 inet_down>500 disk_space>=40" \
        -o 'dph+' --limit 10 --raw 2>/dev/null | python3 -c 'import json,sys
for o in json.load(sys.stdin):
    if (o.get("pcie_bw") or 0) >= 10:
        print(o["id"], round(o["dph_total"], 3), o.get("pcie_bw"), o.get("cpu_ghz"), o.get("host_id"), o.get("machine_id"), o.get("geolocation"))'
}
deadline=$(($(date +%s) + 3600)); tried=" "; ran=0
while :; do
    for g in "$@"; do
        while read -r o dph pcie cpu host mach geo; do
            [ -n "$o" ] || continue
            case "$tried" in *" $o "*) continue ;; esac
            case " ${SKIP_HOSTS:-} " in *" $host "*) continue ;; esac
            tried="$tried$o "
            echo "[$(date +%T)] try $g offer=$o dph=$dph pcie=$pcie cpu_ghz=$cpu host=$host machine=$mach geo=$geo" | tee -a "$RES/attempts-$PROF.log"
            rm -f "$RES/results-$PROF.txt"
            PROFILE=${PROFILE:-1} bash "$HERE/vast_run.sh" "$o" "$PROF" "$SM" "$B" "$RES" "$REPS" < /dev/null
            if grep -q '^== done' "$RES/results-$PROF.txt" 2>/dev/null; then
                echo "[$(date +%T)] DONE $g offer=$o host=$host machine=$mach" | tee -a "$RES/attempts-$PROF.log"; exit 0
            fi
            echo "[$(date +%T)] no complete results from offer=$o" | tee -a "$RES/attempts-$PROF.log"
            if [ "$(grep -c '' "$RES/results-$PROF.txt" 2>/dev/null || echo 0)" -gt 5 ]; then
                cp "$RES/results-$PROF.txt" "$RES/results-$PROF.partial-$o.txt"; ran=$((ran + 1))
                [ "$ran" -ge 2 ] && { echo "[$(date +%T)] GAVE UP after $ran incomplete runs" | tee -a "$RES/attempts-$PROF.log"; exit 1; }
            fi
        done < <(offers "$g")
    done
    [ "$(date +%s)" -ge "$deadline" ] && { echo "[$(date +%T)] GAVE UP" | tee -a "$RES/attempts-$PROF.log"; exit 1; }
    sleep 600
done
