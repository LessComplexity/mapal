#!/bin/bash
# disasm-rtx-5090: ONE RTX 5090 rental, <= 58 min from create to destroy. Stages of
# box_disasm.sh run one ssh call each, results fetched after every stage; the instance is
# destroyed (-y) on every exit path and by a watchdog at 58 min. Only the id this script's
# create returned is ever destroyed.
set -uo pipefail
D=/private/tmp/claude-501/-Volumes-LessComplex-Personal-mapal/d48de64b-c24f-4f25-9674-928a7ce97ccb/scratchpad/disasm-rtx-5090
R=$D/box; mkdir -p "$R"
log() { echo "[$(date -u +%T)] $*" | tee -a "$R/run.log"; }
PROTECT=" 57065574 57100404 57126023 57133432 57285249 "
offers=$(vastai search offers "gpu_name=RTX_5090 num_gpus=1 rentable=true cuda_max_good>=12.8 reliability>0.98 inet_down>500 inet_up>200 disk_space>=40" \
    -o 'dph+' --limit 12 --raw 2>/dev/null | python3 -c 'import json,sys
for o in json.load(sys.stdin): print(o["id"], round(o["dph_total"], 3), o.get("host_id"), o.get("driver_version"), o.get("inet_down"))')
id=""
while read -r o dph host drv inet; do
    [ -n "$o" ] || continue
    log "try offer=$o dph=$dph host=$host driver=$drv inet_down=$inet"
    resp=$(vastai create instance "$o" --image nvidia/cuda:12.8.1-devel-ubuntu22.04 --disk 40 --ssh --direct --raw 2>/dev/null)
    id=$(python3 -c 'import json,sys
try: print(json.loads(sys.argv[1])["new_contract"])
except Exception: print("")' "$resp")
    [ -n "$id" ] && break
    log "offer $o: no instance created"
done <<< "$offers"
[ -n "$id" ] || { log "NO INSTANCE"; exit 1; }
case "$PROTECT" in *" $id "*) log "created id $id is protected?!"; exit 1 ;; esac
echo "$id" > "$R/instance"; created=$(date +%s); log "instance $id created"
destroy() {
    vastai destroy instance "$id" -y >/dev/null 2>&1
    sleep 5
    if vastai show instances --raw 2>/dev/null | python3 -c 'import json,sys; sys.exit(0 if int(sys.argv[1]) in [i["id"] for i in json.load(sys.stdin)] else 1)' "$id"; then
        log "DESTROY FAILED $id still listed"
    else log "destroyed $id (absent from show instances)"; fi
}
( sleep 3480; log "WATCHDOG 58 min"; vastai destroy instance "$id" -y >/dev/null 2>&1 ) & wd=$!
trap 'kill $wd 2>/dev/null; destroy' EXIT
trap 'exit 1' INT TERM HUP
left() { echo $(( created + 3300 - $(date +%s) )); }   # seconds until the 55 min mark
st=""
for _ in $(seq 72); do
    st=$(vastai show instance "$id" --raw 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("actual_status") or "", (d.get("status_msg") or "").split(chr(10))[0][:80])' || true)
    [ "${st%% *}" = running ] && break
    case "$st" in *rror*) log "start failed: $st"; exit 1 ;; esac
    sleep 10
done
[ "${st%% *}" = running ] || { log "never running: $st"; exit 1; }
url=$(vastai ssh-url "$id"); hp=${url#ssh://root@}; host=${hp%:*}; port=${hp##*:}
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30 -o LogLevel=ERROR -p "$port" "root@$host")
for _ in $(seq 30); do "${SSH[@]}" true 2>/dev/null && break; sleep 10; done
log "ssh up ($(( $(date +%s) - created )) s after create)"
for _ in 1 2 3; do "${SSH[@]}" 'mkdir -p /w && tar -xzf - -C /w' < "$D/bundle.tgz" && break; sleep 10; done
log "bundle uploaded"
stage() { # <name> <max_s>
    local t=$(( $(left) - 240 )); [ "$t" -gt "$2" ] && t=$2
    [ "$t" -gt 60 ] || { log "skip $1: $(left) s left"; return 1; }
    log "stage $1 (timeout ${t}s)"
    "${SSH[@]}" "cd /w && timeout ${t} bash box_disasm.sh $1" > "$R/stage-$1.out" 2>&1; log "stage $1 exit $?"
    "${SSH[@]}" "cat /w/results.txt" > "$R/results.txt" 2>/dev/null
}
stage setup 900
# A failed build: wait (<= 8 min) for $R/upload.tgz (fixed sources), unpack it over /w, set up again.
if grep -q 'BUILD FAILED' "$R/results.txt"; then
    log "build failed: waiting for $R/upload.tgz"
    for _ in $(seq 48); do [ -f "$R/upload.tgz" ] && break; sleep 10; done
    if [ -f "$R/upload.tgz" ]; then
        "${SSH[@]}" 'tar -xzf - -C /w' < "$R/upload.tgz"; mv "$R/upload.tgz" "$R/upload.used.tgz"; log "fix uploaded"
        stage setup 600
    fi
fi
stage time 1200
stage nsys 720
stage observe 480
"${SSH[@]}" "cd /w && bash box_disasm.sh pack" >> "$R/run.log" 2>&1
"${SSH[@]}" "cat /w/out.tgz" > "$R/out.tgz" 2>/dev/null; log "fetched out.tgz $(wc -c < "$R/out.tgz") bytes"
log "box minutes: $(( ($(date +%s) - created) / 60 ))"
