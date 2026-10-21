#!/bin/bash
# Rent one vast.ai GPU, run the bundle on it, fetch the results, DESTROY it.
# The instance is destroyed on every exit path (KEEP=1 to keep it for debugging,
# which bills until you run `vastai destroy instance <id>`).
#
# usage: benches/gpu/vast_run.sh <offer_id> <profile> <sm_NN> <bundle_dir> <results_dir> [reps]
set -euo pipefail
OFFER=$1; PROF=$2; SM=$3; BUNDLE=$4; RES=$5; REPS=${6:-30}
IMAGE=${IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu22.04}
mkdir -p "$RES"
log() { echo "[$(date +%T) $PROF] $*" | tee -a "$RES/vast-$PROF.log"; }

id=$(vastai create instance "$OFFER" --image "$IMAGE" --disk 40 --ssh --direct --raw |
     python3 -c 'import json,sys; print(json.load(sys.stdin)["new_contract"])')
log "instance $id from offer $OFFER"
echo "$id" > "$RES/instance-$PROF"
cleanup() {
    if [ "${KEEP:-0}" = 1 ]; then log "KEEP=1: instance $id still billing"; return; fi
    vastai destroy instance "$id" -y >/dev/null 2>&1; if vastai show instances --raw 2>/dev/null | grep -q "\"id\": $id,"; then log "DESTROY FAILED — $id STILL BILLING"; else log "destroyed $id"; fi
}
trap cleanup EXIT

for _ in $(seq 120); do
    st=$(vastai show instance "$id" --raw 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("actual_status") or "")' || true)
    [ "$st" = running ] && break
    sleep 10
done
[ "$st" = running ] || { log "never reached running (last: $st)"; exit 1; }
url=$(vastai ssh-url "$id")            # ssh://root@host:port
hp=${url#ssh://root@}; host=${hp%:*}; port=${hp##*:}
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30 -p "$port" "root@$host")
for _ in $(seq 30); do "${SSH[@]}" true 2>/dev/null && break; sleep 10; done
log "ssh $host:$port up"

up=0
for _ in 1 2 3 4 5; do
    if tar -C "$BUNDLE" -czf - . | "${SSH[@]}" 'mkdir -p /w && tar -xzf - -C /w'; then up=1; break; fi
    sleep 15
done
[ "$up" = 1 ] || { log "bundle upload failed 5 times"; exit 1; }
log "bundle uploaded"
"${SSH[@]}" "cd /w && bash box_run.sh $PROF $SM $REPS" > "$RES/run-$PROF.out" 2>&1 || log "box_run exit $?"
"${SSH[@]}" "cat /w/results-$PROF.txt" > "$RES/results-$PROF.txt" 2>/dev/null || true
"${SSH[@]}" "cat /w/ncu-$PROF.csv" > "$RES/ncu-$PROF.csv" 2>/dev/null || true
if [ "${PROFILE:-0}" = 1 ]; then
    "${SSH[@]}" "cd /w && bash box_profile.sh $PROF $SM" >> "$RES/run-$PROF.out" 2>&1 || log "box_profile exit $?"
    "${SSH[@]}" "cat /w/prof-$PROF.tgz" > "$RES/prof-$PROF.tgz" 2>/dev/null || true
    "${SSH[@]}" "cat /w/results-$PROF.txt" > "$RES/results-$PROF.txt" 2>/dev/null || true  # + the e2e rows
    log "profile fetched: $(wc -c < "$RES/prof-$PROF.tgz") bytes"
fi
log "results fetched: $(grep -c '' "$RES/results-$PROF.txt") lines"
