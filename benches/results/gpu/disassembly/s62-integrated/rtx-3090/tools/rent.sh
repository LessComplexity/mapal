#!/bin/bash
# One RTX 3090 rental for the disasm session: create, upload the bundle, run disasm_box.sh,
# fetch, DESTROY (-y) on every exit path, confirm absence. Never touches other instances.
set -uo pipefail
S=/private/tmp/claude-501/-Volumes-LessComplex-Personal-mapal/d48de64b-c24f-4f25-9674-928a7ce97ccb/scratchpad/disasm-rtx-3090
B=$S/bundle; R=$S/run; mkdir -p $R
log() { echo "[$(date -u +%T)] $*" | tee -a $R/rent.log; }
IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04
offers=$(vastai search offers "gpu_name=RTX_3090 num_gpus=1 rentable=true cuda_max_good>=12.8 reliability>0.97 inet_down>500 disk_space>=40" -o 'dph+' --limit 15 --raw 2>/dev/null |
  python3 -c 'import json,sys
for o in json.load(sys.stdin):
    if (o.get("pcie_bw") or 0) >= 10: print(o["id"], o.get("host_id"), round(o["dph_total"],3), o.get("geolocation","?").replace(" ","_"))')
log "offers: $(echo "$offers" | head -5 | tr '\n' ';')"
id=""
cleanup() {
  [ -n "$id" ] || return 0
  vastai destroy instance "$id" -y >/dev/null 2>&1
  sleep 5
  if vastai show instances --raw 2>/dev/null | grep -q "\"id\": $id,"; then log "DESTROY FAILED $id"; vastai destroy instance "$id" -y >/dev/null 2>&1; else log "destroyed $id (absent from show instances)"; fi
}
trap cleanup EXIT
while read -r offer host dph geo; do
  [ -n "$offer" ] || continue
  log "try offer=$offer host=$host dph=$dph geo=$geo"
  id=$(vastai create instance "$offer" --image "$IMAGE" --disk 40 --ssh --direct --raw 2>/dev/null | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["new_contract"])
except Exception: pass')
  [ -n "$id" ] || { log "create failed"; continue; }
  echo "$id" >> $R/created-ids.txt
  log "created instance $id"
  st=""
  for _ in $(seq 72); do
    st=$(vastai show instance "$id" --raw 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("actual_status") or "", "intended=" + str(d.get("intended_status")), (d.get("status_msg") or "").split(chr(10))[0])' 2>/dev/null)
    [ "${st%% *}" = running ] && break
    case "$st" in *Error*|*intended=stopped*) break ;; esac
    sleep 10
  done
  if [ "${st%% *}" = running ]; then break; fi
  log "instance $id did not start: $st"; cleanup; id=""
done <<< "$offers"
[ -n "$id" ] || { log "no instance"; exit 1; }
START=$(date +%s)
url=$(vastai ssh-url "$id"); hp=${url#ssh://root@}; host=${hp%:*}; port=${hp##*:}
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30 -p "$port" "root@$host")
for _ in $(seq 30); do "${SSH[@]}" true 2>/dev/null && break; sleep 10; done
log "ssh up"
up=0
for _ in 1 2 3 4; do
  if tar -C $B --exclude='*.ll' -czf - . | "${SSH[@]}" 'mkdir -p /w && tar -xzf - -C /w'; then up=1; break; fi; sleep 10
done
[ $up = 1 ] || { log "upload failed"; exit 1; }
log "uploaded"
# Build; on failure fetch the logs and wait (<= 12 min) for $R/FIXED, re-upload the sources.
for try in 1 2 3; do
  "${SSH[@]}" "cd /w && bash build.sh" > $R/build-$try.out 2>&1 && { log "build ok (try $try)"; break; }
  "${SSH[@]}" "cd /w && tail -40 out/build-tile.log; echo ====; tail -40 out/build-rm.log; echo ====; cat out/build-observer.log" > $R/build-$try.logs 2>&1
  log "build failed (try $try): see $R/build-$try.logs"
  [ $try = 3 ] && { log "giving up"; exit 1; }
  rm -f $R/FIXED
  for _ in $(seq 72); do [ -e $R/FIXED ] && break; sleep 10; done
  [ -e $R/FIXED ] || { log "no fix in time"; exit 1; }
  tar -C $B -czf - tile_bench.cu rm_bench.cu disasm_box.sh build.sh | "${SSH[@]}" 'tar -xzf - -C /w'
done
[ -e $R/STOP ] && { log "STOP before run"; exit 1; }
"${SSH[@]}" "cd /w && timeout ${RUN_MIN:-38}m bash disasm_box.sh ${REPS:-30}" > $R/box.out 2>&1; log "box exit $?"
"${SSH[@]}" "cat /w/results.txt" > $R/results.txt 2>/dev/null
"${SSH[@]}" "cd /w && tar czf - out" > $R/out.tgz 2>/dev/null
log "fetched results=$(grep -c '' $R/results.txt) lines out.tgz=$(wc -c < $R/out.tgz) bytes; box minutes=$(( ($(date +%s) - START) / 60 ))"
