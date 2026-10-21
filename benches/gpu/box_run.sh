#!/bin/bash
# Runs ON the GPU box, inside an unpacked bundle (benches/gpu/bundle.sh).
# 1. device facts vs the profile, 2. the differential (GPU stdout == oracle),
# 3. the timing harness vs cuBLAS and CUTLASS.  Every line lands in results.txt
# the moment it is produced.
#
# usage: bash box_run.sh <profile> <sm_NN> [reps]
set -uo pipefail
PROF=$1; SM=$2; REPS=${3:-30}
CUDA=${CUDA:-/usr/local/cuda}
LOG=results-$PROF.txt
say() { echo "$@" | tee -a "$LOG"; }
: > "$LOG"
say "== $PROF $SM $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.sm,power.limit,temperature.gpu,persistence_mode --format=csv | tee -a "$LOG"
nvcc --version | tail -1 | tee -a "$LOG"

nvcc -O3 -std=c++17 -arch="$SM" -Icutlass/include -Icutlass/tools/util/include \
    tile_bench.cu -lcublas -lcuda -o tile_bench 2>&1 | grep -v Wno | tail -5 | tee -a "$LOG"
[ -x tile_bench ] || { say "HARNESS BUILD FAILED"; exit 1; }

say "-- differential (GPU stdout vs oracle)"
pass=0; fail=0
for o in "$PROF"/*.host.o; do
    b=$(basename "$o" .host.o)
    case $b in mm_*|e2e_*|loop_mm_*) continue ;; esac
    # `<prog>[_c][.off-<rung>]`: every arm of a face answers to that face's oracle.
    stem=${b%%.*}; base=${stem%_c}; exp="tiles/$base.expect"; [ "$stem" != "$base" ] && exp="tiles/${base}_c.expect"
    cc "$o" mapal_gpu.c libmapal_rt.a -I"$CUDA/include" -L"$CUDA/lib64/stubs" -lcuda -lpthread -ldl -lm -o "prog_$b" 2>&1 | tail -3
    got=$(MAPAL_PTX="$PROF/$b.ptx" "./prog_$b" 2>&1); rc=$?
    if [ $rc -eq 0 ] && [ "$got" = "$(cat "$exp")" ]; then
        pass=$((pass + 1)); say "PASS $b $(tr '\n' ' ' <<< "$got")"
    else
        fail=$((fail + 1)); say "FAIL $b rc=$rc got=[$(tr '\n' ' ' <<< "$got")] want=[$(tr '\n' ' ' < "$exp")]"
    fi
done
say "differential: $pass pass, $fail fail"

# Host-link arms (S53 `resident`): each e2e-shaped program, both faces, the
# shipped emission vs each host rung off, interleaved per repetition. One
# untraced process for the wall time, one traced for the phase totals; the
# cuBLAS whole-program floor (`tile_bench --program`) next to it.
say "-- e2e: host-link arms, process wall + MAPAL_GPU_TRACE totals"
ARMS="on off-devgen off-resident"
arm_stem() { [ "$2" = on ] && echo "$1" || echo "$1.$2"; }
e2e() { # <prog> <floor: n gemms ops>
    local p=$1; shift
    for face in "" _c; do
        for arm in $ARMS; do
            b=$(arm_stem "$p$face" "$arm")
            [ -f "$PROF/$b.host.o" ] && cc "$PROF/$b.host.o" mapal_gpu.c libmapal_rt.a -I"$CUDA/include" \
                -L"$CUDA/lib64/stubs" -lcuda -lpthread -ldl -lm -o "e2e_$b" 2>&1 | tail -3
        done
        for r in $(seq "${E2E_REPS:-7}"); do  # interleaved arms; medians
            for arm in $ARMS; do
                b=$(arm_stem "$p$face" "$arm")
                [ -x "e2e_$b" ] || continue
                t0=$(date +%s%N)
                got=$(MAPAL_PTX="$PROF/$b.ptx" "./e2e_$b" 2>/dev/null | tr '\n' ' ')
                t1=$(date +%s%N)
                tr=$(MAPAL_GPU_TRACE=1 MAPAL_PTX="$PROF/$b.ptx" "./e2e_$b" 2>&1 >/dev/null)
                t2=$(date +%s%N)
                say "e2e prog=$p face=${face:-_exact} arm=$arm rep=$r wall_us=$(( (t1 - t0) / 1000 )) traced_wall_us=$(( (t2 - t1) / 1000 )) out=[$got] $(grep mapal_gpu_trace_total <<< "$tr")"
                if [ "$r" = 1 ]; then grep -v mapal_gpu_trace_total <<< "$tr" | sed "s/^/trace prog=$b /" >> "$LOG"; fi
            done
        done
    done
    for r in 1 2 3; do say "floor prog=$p rep=$r $(./tile_bench --program "$@" 2>&1)"; done
}
e2e e2e_1024 1024 1 2
e2e e2e_4096 4096 1 2
e2e mm_share_1024 1024 2 3
e2e mm_chain_2048 2048 2 3
e2e e2e_chain_4096 4096 2 3
e2e loop_mm_1024_8 1024 8 2
say "$(./tile_bench --h2d 64 10 2>&1)"

say "-- timing vs cuBLAS / CUTLASS (f32, reps=$REPS)"
bench() { # <n> <stem> <label> [--contract]
    local n=$1 stem=$2 label=$3; shift 3
    read -r k gx gy blk sm < "$stem.launch"
    say "# n=$n face=$label"
    ./tile_bench "$n" "$n" "$n" "$stem.ptx" "$k" "$gx" "$gy" "$blk" "$sm" "$REPS" "$@" 2>&1 | tee -a "$LOG"
}
for n in 1024 2048 4096 8192; do
    bench "$n" "$PROF/mm_$n" _exact
    bench "$n" "$PROF/mm_${n}_c" _c --contract
    [ -f "$PROF/mm_${n}_tf.launch" ] && bench "$n" "$PROF/mm_${n}_tf" _tf --tf32
    for v in "$PROF/mm_${n}_tf".*.launch; do
        [ -f "$v" ] || continue
        var=${v%.launch}; var=${var##*.}
        bench "$n" "${v%.launch}" "_tf.$var" --tf32
    done
    for v in "$PROF/mm_${n}_c".*.launch; do
        [ -f "$v" ] || continue
        var=${v%.launch}; var=${var##*.}
        bench "$n" "${v%.launch}" "_c.$var" --contract
    done
done

# Where the time goes: Nsight Compute on the three kernels at n=4096, FMA face.
NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
if [ -z "$NCU" ]; then
    apt-get install -y -qq "cuda-nsight-compute-$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1-\2/p')" >/dev/null 2>&1
    NCU=$(ls /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
fi
if [ -n "$NCU" ]; then
    read -r k gx gy blk sm < "$PROF/mm_4096_c.launch"
    say "-- ncu ($NCU)"
    "$NCU" --launch-count 9 --csv --page details \
        --section SpeedOfLight --section Occupancy --section LaunchStats --section SchedulerStats \
        --section WarpStateStats --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
        --section MemoryWorkloadAnalysis_Tables --section InstructionStats \
        ./tile_bench 4096 4096 4096 "$PROF/mm_4096_c.ptx" "$k" "$gx" "$gy" "$blk" "$sm" 1 --contract \
        > "ncu-$PROF.csv" 2> "ncu-$PROF.err"
    say "ncu exit $? lines $(wc -l < "ncu-$PROF.csv") $(grep -m1 -i 'err\|permission' "ncu-$PROF.err")"
else
    say "-- ncu not available"
fi
say "== done"
