#!/bin/bash
# S62 submission stage 2, ON the GPU box (uploaded as the bundle's box_run.sh, so vast_run.sh
# runs it): every launch in submit-list-<profile> (benches/gpu/submit_list.py) through the
# generated runtime entry and a minimal direct loop (submit_bench.c), three processes each.
# usage: bash box_run.sh <profile> <sm_NN> [reps]
set -uo pipefail
PROF=$1; SM=$2
CUDA=${CUDA:-/usr/local/cuda}
LOG=results-$PROF.txt
say() { echo "$@" | tee -a "$LOG"; }
: > "$LOG"
say "== $PROF $SM $(date -u +%FT%TZ) submit stage 2"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.sm,power.limit,temperature.gpu,persistence_mode --format=csv | tee -a "$LOG"
lscpu | grep -E 'Model name|^CPU\(s\)|MHz' | tee -a "$LOG"
cc -O2 submit_bench.c mapal_gpu.c -I"$CUDA/include" -L"$CUDA/lib64/stubs" -lcuda -lpthread -ldl -lm -o submit_bench 2>&1 | tail -3
[ -x submit_bench ] || { say "BUILD FAILED"; exit 1; }
while read -r prog entry id nptr bytes gx gy block smem; do
    for rep in 1 2 3; do
        out=$(MAPAL_PTX="$PROF/$prog.ptx" ./submit_bench "$entry" "$id" "$nptr" "$bytes" "$gx" "$gy" "$block" "$smem" 10 100 2>&1); rc=$?
        say "$out prog=$prog rep=$rep rc=$rc"
        [ "$rc" -eq 0 ] || { say "VALIDATION FAILED (submit_bench $prog rc=$rc)"; exit 1; }
    done
done < "submit-list-$PROF"
say "== done"
