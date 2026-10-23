#!/bin/bash
# s62 disasm (rtx-4090) — runs ON the box as the bundle's box_run.sh (vast_run.sh calls
# `bash box_run.sh <prof> <sm> <reps>`; everything raw lands under trace/, which vast_run.sh
# fetches as trace-<prof>.tgz; every number lands in results-<prof>.txt the moment it exists).
#   0. builds (tile_bench_s62d: cuBLAS + stock + tuned CUTLASS legs; mlp_bench; capture.so),
#      static ptxas -v / cuobjdump of every Mapal PTX, harness SASS + res-usage
#   1. kernel cells: tile_bench --splitk, every Mapal arm (E, C, r, rc, null E/C, base E/C) with
#      cuBLAS, stock CUTLASS, tuned CUTLASS in one process, value gates first, 2 processes ABBA
#   2. residual MLP whole programs: Mapal on/base/null x E/C x CUDA_MODULE_LOADING LAZY/EAGER,
#      value-gated, ABBA; traced reps; the library legs (mlp_bench: eager, lt, cutlass, cutlass_tuned)
#   3. counter-free observer (capture.cpp) per cell x leg and per MLP program/leg -> executed images
#   4. nsys: kernel names/grids/regs/smem per cell; API + kernel timeline of the MLP programs
set -uo pipefail
PROF=$1; SM=$2; REPS=${3:-30}
CUDA=${CUDA:-/usr/local/cuda}
export PATH=$PATH:$CUDA/bin NVIDIA_TF32_OVERRIDE=0
LOG=results-$PROF.txt; T=trace
mkdir -p $T/sass $T/ptxas $T/nsys $T/obs $T/mlp
say() { echo "$@" | tee -a "$LOG"; }
: > "$LOG"
say "== $PROF $SM $(date -u +%FT%TZ) s62-disasm"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,clocks.sm,power.limit,temperature.gpu,persistence_mode --format=csv | tee -a "$LOG"
nvidia-smi -q > $T/nvidia-smi-q.txt 2>&1
lscpu | grep -E '^(Model name|CPU\(s\)|CPU max MHz)' | sed 's/^/lscpu: /' | tee -a "$LOG"
nvcc --version | tail -1 | tee -a "$LOG"
( apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq cuda-nsight-systems-12-8 >/dev/null 2>&1 || apt-get install -y -qq nsight-systems >/dev/null 2>&1
  echo $? > nsys.install ) &
NSYS_PID=$!
B=(-O3 -std=c++17 -arch="$SM" -Icutlass/include -Icutlass/tools/util/include)
nvcc "${B[@]}" tile_bench_s62d.cu -lcublas -lcuda -o tile_bench > $T/build-tile.log 2>&1 & P1=$!
nvcc "${B[@]}" mlp_bench.cu -lcublasLt -lcublas -o mlp_bench > $T/build-mlp.log 2>&1 & P2=$!
# CUPTI: extras/CUPTI on some images, targets/x86_64-linux on nvidia/cuda:12.8.1-devel (the 2026-10-23 run
# built capture.so by hand with the second path after the first failed; this is that command).
CL=$(dirname "$(ls $CUDA/extras/CUPTI/lib64/libcupti.so $CUDA/targets/x86_64-linux/lib/libcupti.so 2>/dev/null | head -1)")
c++ -std=c++17 -O2 -fPIC -shared -I$CUDA/include -I$CUDA/extras/CUPTI/include -I$CUDA/targets/x86_64-linux/include capture.cpp \
    -L"$CL" -Wl,-rpath,"$CL" -lcupti -pthread -o capture.so > $T/build-capture.log 2>&1 & P3=$!
for o in "$PROF"/pt_*.host.o; do
    b=$(basename "$o" .host.o)
    cc "$o" mapal_gpu.c libmapal_rt.a -I"$CUDA/include" -L"$CUDA/lib64/stubs" -lcuda -lpthread -ldl -lm -o "pt_$b" 2>&1 | tail -3
done
# Static: ptxas (this toolkit) resources and SASS of every distinct Mapal PTX (.null is a byte copy).
for p in "$PROF"/*.ptx; do
    v=$(basename "$p" .ptx); case $v in *.null) continue ;; esac
    ptxas -arch="$SM" -O3 -v "$p" -o "$T/ptxas/$v.cubin" > "$T/ptxas/$v.ptxas" 2>&1
    cuobjdump -sass "$T/ptxas/$v.cubin" > "$T/sass/$v.ptxas.sass" 2>&1
    cuobjdump -res-usage "$T/ptxas/$v.cubin" > "$T/sass/$v.ptxas.res" 2>&1
done
say "static: $(ls $T/ptxas/*.ptxas | wc -l) ptxas reports"
wait $P1; wait $P2; wait $P3
[ -x mlp_bench ] || nvcc "${B[@]}" -DNO_TUNED mlp_bench.cu -lcublasLt -lcublas -o mlp_bench > $T/build-mlp-notuned.log 2>&1
tail -3 $T/build-tile.log $T/build-mlp.log $T/build-capture.log | tee -a "$LOG"
[ -x tile_bench ] || { say "HARNESS BUILD FAILED"; exit 1; }
for x in tile_bench mlp_bench; do
    [ -x $x ] || continue
    cuobjdump -sass $x > $T/sass/$x.sass 2>&1
    cuobjdump -res-usage $x > $T/sass/$x.res 2>&1
done

CELLS=${CELLS:-"16x2048x7168 16x1024x4096 16x18432x7168 128x2048x7168 4096x4096x4096"}
say "-- cells: every arm + cublas + cutlass + cutlass_tuned, one process, 2 processes ABBA (reps=$REPS)"
for c in $CELLS; do
    read -r M N K <<< "${c//x/ }"
    arms="mm_$c mm_$c.null mm_${c}_c mm_${c}_c.null mm_${c}_r mm_${c}_rc mm_$c.base mm_${c}_c.base"
    for pass in 1 2; do
        order=$arms; [ $pass = 2 ] && order=$(tr ' ' '\n' <<< "$arms" | tac | tr '\n' ' ')
        # shellcheck disable=SC2086
        ./tile_bench --splitk "$M" "$N" "$K" "$REPS" "$PROF" $order 2>&1 | sed "s/^/pass=$pass cell=$c /" | tee -a "$LOG"
        [ "${PIPESTATUS[0]}" -eq 0 ] || say "CELL GATE FAILED cell=$c pass=$pass"
    done
done

MLPS=${MLPS:-"pt_rm_m128_d256_h512 pt_rm_m128_d1024_h2048"}
say "-- mlp: Mapal whole programs (iter ms), on/base/null x E/C x LAZY/EAGER, ABBA, reps=${MLP_REPS:-20}"
for p in $MLPS; do
    runs=()
    for face in "" _c; do
        for arm in on base null; do
            s=$p$face; [ $arm != on ] && s=$s.$arm
            [ -x "pt_$s" ] || continue
            for ld in LAZY EAGER; do runs+=("$face:$arm:$ld"); done
        done
    done
    bad=0
    for r in "${runs[@]}"; do
        IFS=: read -r face arm ld <<< "$r"; s=$p$face; [ $arm != on ] && s=$s.$arm
        got=$(CUDA_MODULE_LOADING=$ld MAPAL_PTX="$PROF/$s.ptx" "./pt_$s" 2>/dev/null | tr '\n' ' '); got=${got%%iter ms=*}
        want=$(tr '\n' ' ' < "tiles/$p$face.expect")
        [ "$got" = "$want" ] || { bad=1; say "VALUE-FAIL prog=$p face=${face:-_exact} arm=$arm ld=$ld got=[$got] want=[$want]"; }
    done
    [ $bad = 0 ] || continue
    say "value-gate prog=$p runs=${#runs[@]} pass"
    for rep in $(seq "${MLP_REPS:-20}"); do
        order=("${runs[@]}")
        if [ $((rep % 2)) = 0 ]; then order=(); for ((i = ${#runs[@]} - 1; i >= 0; i--)); do order+=("${runs[$i]}"); done; fi
        for r in "${order[@]}"; do
            IFS=: read -r face arm ld <<< "$r"; s=$p$face; [ $arm != on ] && s=$s.$arm
            t0=$(date +%s%N)
            out=$(CUDA_MODULE_LOADING=$ld MAPAL_PTX="$PROF/$s.ptx" "./pt_$s" 2>/dev/null | tr '\n' ' ')
            t1=$(date +%s%N)
            say "mlp prog=$p face=${face:-_exact} arm=$arm ld=$ld rep=$rep wall_us=$(( (t1 - t0) / 1000 )) out=[$out]"
        done
    done
    for rep in 1 2; do
        for r in "${runs[@]}"; do
            IFS=: read -r face arm ld <<< "$r"; s=$p$face; [ $arm != on ] && s=$s.$arm
            CUDA_MODULE_LOADING=$ld MAPAL_GPU_TRACE=1 MAPAL_PTX="$PROF/$s.ptx" "./pt_$s" > "$T/mlp/$s.$ld.trace$rep" 2>&1
            say "mlp_trace prog=$p face=${face:-_exact} arm=$arm ld=$ld rep=$rep $(grep -m1 'iter ms' "$T/mlp/$s.$ld.trace$rep" | tr -d '\n') $(grep -m1 mapal_gpu_trace_total "$T/mlp/$s.$ld.trace$rep") kernels=[$(grep -o 'op=launch kernel=mapal_tile_[0-9a-f]* ms=[0-9.]*' "$T/mlp/$s.$ld.trace$rep" | sed 's/op=launch kernel=mapal_tile_//' | tr '\n' ' ')]"
        done
    done
    d=${p#*_d}; d=${d%%_*}; h=${p##*_h}
    if [ -x mlp_bench ]; then
        for pass in 1 2; do
            # shellcheck disable=SC2046
            ./mlp_bench 128 "$d" "$h" "$REPS" $(cat "tiles/$p.expect") 2>&1 | sed "s/^/pass=$pass prog=$p /" | tee -a "$LOG"
        done
    fi
done

say "-- observer: counter-free capture of the executed images (capture.cpp), one call per leg"
cap() { # <name> <ptx> <cmd...> (PROFILE_LEG from the caller's env)
    local n=$1 ptx=$2 rc; shift 2
    CUDA_INJECTION64_PATH=$PWD/capture.so MAPAL_CAPTURE_OUT=$PWD/$T/obs/$n MAPAL_CAPTURE_PTX=$PWD/$ptx \
        timeout 180 "$@" > "$T/obs/$n.out" 2>&1; rc=$?
    timeout 600 python3 -I obs_extract.py "$T/obs/$n" > "$T/obs/$n.kernels" 2>&1
    say "obs name=$n rc=$rc final=[$(tail -1 "$T/obs/$n/events.tsv" 2>/dev/null | tr '\t' ' ')] $(grep -c '' "$T/obs/$n/kernels.tsv" 2>/dev/null)"
    sed 1d "$T/obs/$n/kernels.tsv" 2>/dev/null | cut -f2-8 | sed "s/^/obs_kernel name=$n /" | tee -a "$LOG"
}
if [ -f capture.so ]; then
    for c in $CELLS; do
        read -r M N K <<< "${c//x/ }"
        for leg in cublas cutlass cutlass_tuned; do
            PROFILE_LEG=$leg cap "$c.$leg" "$PROF/mm_${c}_c.ptx" ./tile_bench --splitk "$M" "$N" "$K" 1 "$PROF" "mm_${c}_c"
        done
        for v in "" _c _r _rc; do
            PROFILE_LEG=mm_$c$v cap "$c.mapal$v" "$PROF/mm_$c$v.ptx" ./tile_bench --splitk "$M" "$N" "$K" 1 "$PROF" "mm_$c$v"
        done
    done
    for p in $MLPS; do
        for face in "" _c; do
            for arm in on base; do
                s=$p$face; [ $arm != on ] && s=$s.$arm
                MAPAL_PTX="$PROF/$s.ptx" cap "mlp.$s" "$PROF/$s.ptx" "$PWD/pt_$s"
            done
        done
        d=${p#*_d}; d=${d%%_*}; h=${p##*_h}
        for leg in lt eager cutlass cutlass_tuned; do
            PROFILE_LEG=$leg cap "mlp.$p.$leg" "$PROF/$p.ptx" ./mlp_bench 128 "$d" "$h" 1
        done
    done
fi

wait $NSYS_PID
NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys $CUDA/bin/nsys 2>/dev/null | head -1)
say "-- nsys: ${NSYS:-unavailable} install_rc=$(cat nsys.install 2>/dev/null)"
if [ -n "$NSYS" ]; then
    $NSYS --version | tee -a "$LOG"
    for c in $CELLS; do
        read -r M N K <<< "${c//x/ }"
        timeout 300 $NSYS profile -t cuda -o "$T/nsys/cell_$c" --force-overwrite true \
            ./tile_bench --splitk "$M" "$N" "$K" 3 "$PROF" "mm_$c" "mm_${c}_c" "mm_${c}_r" "mm_${c}_rc" > "$T/nsys/cell_$c.out" 2>&1
        timeout 300 $NSYS stats -q --format csv -r cuda_gpu_trace,cuda_gpu_kern_sum "$T/nsys/cell_$c.nsys-rep" > "$T/nsys/cell_$c.csv" 2>&1
        say "nsys cell=$c rows=$(grep -c '' "$T/nsys/cell_$c.csv")"
    done
    for p in $MLPS; do
        for face in "" _c; do
            for arm in on base; do
                s=$p$face; [ $arm != on ] && s=$s.$arm
                for ld in LAZY EAGER; do
                    [ $ld = EAGER ] && [ $arm != on ] && continue
                    CUDA_MODULE_LOADING=$ld MAPAL_PTX="$PROF/$s.ptx" timeout 300 $NSYS profile -t cuda,osrt -o "$T/nsys/mlp_$s.$ld" --force-overwrite true \
                        "./pt_$s" > "$T/nsys/mlp_$s.$ld.out" 2>&1
                    timeout 300 $NSYS stats -q --format csv -r cuda_api_trace,cuda_gpu_trace,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
                        "$T/nsys/mlp_$s.$ld.nsys-rep" > "$T/nsys/mlp_$s.$ld.csv" 2>&1
                    say "nsys mlp=$s ld=$ld $(tr '\n' ' ' < "$T/nsys/mlp_$s.$ld.out" | grep -o 'iter ms=[0-9.]*') rows=$(grep -c '' "$T/nsys/mlp_$s.$ld.csv")"
                done
            done
        done
        d=${p#*_d}; d=${d%%_*}; h=${p##*_h}
        timeout 300 $NSYS profile -t cuda,osrt -o "$T/nsys/mlplib_$p" --force-overwrite true ./mlp_bench 128 "$d" "$h" 3 > "$T/nsys/mlplib_$p.out" 2>&1
        timeout 300 $NSYS stats -q --format csv -r cuda_api_trace,cuda_gpu_trace,cuda_api_sum,cuda_gpu_kern_sum "$T/nsys/mlplib_$p.nsys-rep" > "$T/nsys/mlplib_$p.csv" 2>&1
        say "nsys mlplib=$p rows=$(grep -c '' "$T/nsys/mlplib_$p.csv")"
    done
fi
say "== done"
