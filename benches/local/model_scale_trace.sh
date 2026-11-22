#!/bin/bash
# model-scale: CUPTI traces (ktrace.cpp) of one process per leg, after a timed run of the same bundle (its linked
# `pt_<stem>[.<arm>]` binaries, `vals/`, `gemm_lib`): Mapal's shipped and GPU-only arms, the torch legs one mode per
# process (TORCH_MODES), gemm_lib with 3 repetitions. Same machine protocol as model_scale_run.sh (one lock for the
# whole pass: every process is seconds). Summaries: ktrace_sum.py, Mapal over its timed kernels (first to last
# `mapal_tile`); torch / gemm_lib whole-process (per call = total / (1 gate + 5 warm-up + 3 timed calls)).
# usage: model_scale_trace.sh <bundle> <out dir> <ktrace.so> <stem>:<torch mode | comp=<comp_bench leg>,...>...
#        (TRACE_ARMS="on off-place")
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
B=$(cd "$1" && pwd); O=$2; KT=$(cd "$(dirname "$3")" && pwd)/$(basename "$3"); shift 3
LOCK=${BENCH_LOCK:?BENCH_LOCK}; CPUS=${CPUS:-0-7}
mkdir -p "$O"; O=$(cd "$O" && pwd)
exec 9> "$LOCK"; flock 9
pgrep -x MapleStory.exe > /dev/null && { echo "MapleStory.exe running: no traces"; exit 1; }
echo "trace $(date -u +%FT%TZ) $(sensors 2>/dev/null | grep Tctl | tr -s ' ')" >> "$O/conditions.txt"
cd "$B" || exit 1
tr1() { local out=$1; shift; rm -f "$out"; CUDA_INJECTION64_PATH=$KT KTRACE_OUT=$out taskset -c "$CPUS" "$@" > "$out.stdout" 2>&1; }
for spec in "$@"; do
    s=${spec%%:*}; modes=${spec#*:}
    for a in ${TRACE_ARMS:-on off-place}; do  # (a host-placed shipped decode runs 45 s: TRACE_ARMS=off-place)
        arm=; [ "$a" = on ] || arm=.$a
        [ -x "pt_$s$arm" ] || continue
        tr1 "$O/$s$arm.mapal.tsv" env MAPAL_WISDOM="$B/wisdom" MAPAL_GPU_TUNE=0 MAPAL_PTX="rtx-5080/$s$arm.ptx" "./pt_$s$arm"
        python3 "$ROOT/benches/local/ktrace_sum.py" "$O/$s$arm.mapal.tsv" --window=mapal_tile:mapal_tile > "$O/$s$arm.mapal.sum"
    done
    for m in ${modes//,/ }; do
        if [ "${m#comp=}" != "$m" ]; then  # competitors: a comp_bench.py leg (`comp=<leg>`)
            tr1 "$O/$s.comp-${m#comp=}.tsv" env COMP_LEGS="${m#comp=}" "$TORCH_PY" comp_bench.py 3 vals "$s"
            python3 "$ROOT/benches/local/ktrace_sum.py" "$O/$s.comp-${m#comp=}.tsv" --top=16 > "$O/$s.comp-${m#comp=}.sum"
            continue
        fi
        tr1 "$O/$s.torch-$m.tsv" env TORCH_MODES="$m" "$TORCH_PY" torch_bench.py 3 vals "$s"
        python3 "$ROOT/benches/local/ktrace_sum.py" "$O/$s.torch-$m.tsv" --top=16 > "$O/$s.torch-$m.sum"
    done
    if [[ $s =~ ^pt_pj_m([0-9]+)_n([0-9]+)_k([0-9]+)(_tf32)?$ ]] && [ -x gemm_lib ]; then
        tr1 "$O/$s.gemm.tsv" ./gemm_lib "$s" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}" 3 ${BASH_REMATCH[4]:+--tf32}
        python3 "$ROOT/benches/local/ktrace_sum.py" "$O/$s.gemm.tsv" --top=16 > "$O/$s.gemm.sum"
    fi
    echo "traced $s"
done
flock -u 9
