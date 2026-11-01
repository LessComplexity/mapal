#!/bin/bash
# Rent one vast.ai GPU, run the bundle on it, fetch the results, DESTROY it.
# The instance is destroyed on every exit path (KEEP=1 to keep it for debugging,
# which bills until you run `vastai destroy instance <id>`).
#
# usage: benches/gpu/vast_run.sh <offer_id> <profile> <sm_NN> <bundle_dir> <results_dir> [reps]
# DIFFERENTIAL_ONLY / PROBE_ARMS / PROBE_REPS / E2E_REPS pass through to box_run.sh.
set -euo pipefail
OFFER=$1; PROF=$2; SM=$3; BUNDLE=$4; RES=$5; REPS=${6:-30}
IMAGE=${IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu22.04}
mkdir -p "$RES"
log() { echo "[$(date +%T) $PROF] $*" | tee -a "$RES/vast-$PROF.log"; }
# site-geometry: a bundle box_run.sh would stop on fails here, before the rental (PREFLIGHT=0 skips).
if [ "${PREFLIGHT:-1}" = 1 ] && ! python3 "$(dirname "$0")/preflight.py" "$BUNDLE" "$PROF" >> "$RES/vast-$PROF.log"; then
    log "preflight failed ($RES/vast-$PROF.log): no rental"; exit 1
fi

id=$(vastai create instance "$OFFER" --image "$IMAGE" --disk 40 --ssh --direct --raw |
     python3 -c 'import json,sys; print(json.load(sys.stdin)["new_contract"])')
log "instance $id from offer $OFFER"
echo "$id" > "$RES/instance-$PROF"
# LABEL=<topic>: names the rental (CLAUDE.md: never touch an instance you did not create).
if [ -n "${LABEL:-}" ]; then vastai label instance "$id" "$LABEL" >/dev/null 2>&1 || log "label failed"; fi
cleanup() {
    if [ "${KEEP:-0}" = 1 ]; then log "KEEP=1: instance $id still billing"; return; fi
    vastai destroy instance "$id" -y >/dev/null 2>&1; if vastai show instances --raw 2>/dev/null | grep -q "\"id\": $id,"; then log "DESTROY FAILED — $id STILL BILLING"; else log "destroyed $id"; fi
}
trap cleanup EXIT

# START_TRIES x 10 s for the container to start (default 20 min; a slow image pull needs more).
for _ in $(seq "${START_TRIES:-120}"); do
    st=$(vastai show instance "$id" --raw 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("actual_status") or "", "intended=" + str(d.get("intended_status")), (d.get("status_msg") or "").split(chr(10))[0])' || true)
    [ "${st%% *}" = running ] && { st=running; break; }
    # A host that cannot start the container says so in status_msg, and a rental
    # whose GPU another renter took is created stopped: give the offer up now.
    case "$st" in *Error*|*intended=stopped*) log "start failed: $st"; exit 1 ;; esac
    sleep 10
done
[ "$st" = running ] || { log "never reached running (last: $st)"; exit 1; }
url=$(vastai ssh-url "$id")            # ssh://root@host:port
hp=${url#ssh://root@}; host=${hp%:*}; port=${hp##*:}
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30 -p "$port" "root@$host")
for _ in $(seq 30); do "${SSH[@]}" true 2>/dev/null && break; sleep 10; done
log "ssh $host:$port up"
# WAIT_FILE=<path>: rent while the bundle is still being built; upload once the path exists.
while [ -n "${WAIT_FILE:-}" ] && [ ! -e "$WAIT_FILE" ]; do sleep 10; done

# A bundle holds every part's directory: upload this part's only, and no LLVM
# IR (the box reads host.o, ptx and launch).
skip=(--exclude='*.ll' --exclude='*.yaml' --exclude='*.yaml.gz' --exclude='*.sass' --exclude='*.ptxas' \
    --exclude='*.host.asm' --exclude='*.alias' --exclude='evidence.tsv' \
    --exclude='./tiles/*.cpu' --exclude='./tiles/*.cpu_c')  # the oracles' macOS builds (~400 MB)
for x in "$BUNDLE"/*/; do
    n=$(basename "$x")
    if [ "$n" != "$PROF" ] && compgen -G "$x*.host.o" >/dev/null; then skip+=(--exclude="./$n"); fi
done
up=0
for _ in 1 2 3 4 5; do
    if tar -C "$BUNDLE" "${skip[@]}" -czf - . | "${SSH[@]}" 'mkdir -p /w && tar -xzf - -C /w'; then up=1; break; fi
    sleep 15
done
[ "$up" = 1 ] || { log "bundle upload failed 5 times"; exit 1; }
log "bundle uploaded"
# The e2e and probe sections' arms and repetitions, and box_profile's traced
# probes, pass through (unset: their defaults).
ENVS="DIFFERENTIAL_ONLY='${DIFFERENTIAL_ONLY:-0}' E2E_ARMS='${E2E_ARMS:-}' PROBE_ARMS='${PROBE_ARMS:-}' PROBE_REPS='${PROBE_REPS:-}' E2E_REPS='${E2E_REPS:-}' PROF_PROBES='${PROF_PROBES:-}' PROBE_TRACE_REPS='${PROBE_TRACE_REPS:-}' LANE_PROGS='${LANE_PROGS:-}' PASSES='${PASSES:-1}' FACES='${FACES:-_exact _c _tf}' WARM='${WARM:-0}' E2E='${E2E:-1}' SUBMIT='${SUBMIT:-}' TIMING='${TIMING:-1}' TORCH='${TORCH:-1}' SWEEP_PROF='${SWEEP_PROF:-0}' TORCH_MIN='${TORCH_MIN:-15}' SASS_MIN='${SASS_MIN:-12}'"
run_rc=0
# RUN_MIN (S62): the box run's ceiling in minutes, so a rental stays inside its time budget
# (start + upload + run + fetch); the results so far are still fetched.
"${SSH[@]}" "cd /w && $ENVS timeout ${RUN_MIN:-70}m bash box_run.sh $PROF $SM $REPS" >"$RES/run-$PROF.out" 2>&1 || { run_rc=$?; log "box_run exit $run_rc"; }
"${SSH[@]}" "cat /w/results-$PROF.txt" > "$RES/results-$PROF.txt" 2>/dev/null || {
    log "results retrieval failed"
    [ "$run_rc" -ne 0 ] || run_rc=1
}
"${SSH[@]}" "cat /w/results-$PROF-pass2.txt" > "$RES/results-$PROF-pass2.txt" 2>/dev/null || true
"${SSH[@]}" "cat /w/ncu-$PROF.csv" > "$RES/ncu-$PROF.csv" 2>/dev/null || true
# S64: the intent check of this run, the PyTorch install log.
"${SSH[@]}" "cat /w/evidence-$PROF.tsv" > "$RES/evidence-$PROF.tsv" 2>/dev/null || true
"${SSH[@]}" "cat /w/torch-install.log /w/torch.err" > "$RES/torch-$PROF.log" 2>/dev/null || true
"${SSH[@]}" "cd /w && [ -d trace ] && tar czf - trace" > "$RES/trace-$PROF.tgz" 2>/dev/null || true
if [ "$run_rc" -eq 0 ] && [ "${PROFILE:-0}" = 1 ] && [ "${DIFFERENTIAL_ONLY:-0}" != 1 ]; then
    "${SSH[@]}" "cd /w && $ENVS bash box_profile.sh $PROF $SM" >> "$RES/run-$PROF.out" 2>&1 || { run_rc=$?; log "box_profile exit $run_rc"; }
    "${SSH[@]}" "cat /w/prof-$PROF.tgz" > "$RES/prof-$PROF.tgz" 2>/dev/null || true
    # + the e2e rows; a box lost during the profile keeps the results already fetched.
    "${SSH[@]}" "cat /w/results-$PROF.txt" > "$RES/results-$PROF.txt.new" 2>/dev/null &&
        [ -s "$RES/results-$PROF.txt.new" ] && mv "$RES/results-$PROF.txt.new" "$RES/results-$PROF.txt"
    rm -f "$RES/results-$PROF.txt.new"
    log "profile fetched: $(wc -c < "$RES/prof-$PROF.tgz") bytes"
fi
log "results fetched: $(grep -c '' "$RES/results-$PROF.txt") lines"
exit "$run_rc"
