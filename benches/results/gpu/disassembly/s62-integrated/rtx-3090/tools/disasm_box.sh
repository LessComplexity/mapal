#!/bin/bash
# ON the rented RTX 3090, inside the unpacked bundle: same-box timing of Mapal (integrated
# s62 PTX) vs cuBLAS / stock CUTLASS / tuned CUTLASS / cuBLASLt, nsys traces, and the
# counter-free observer's captured executed images. Every line lands in results.txt as
# it is produced; captures and traces land in out/ (tarred by the caller).
# usage: bash disasm_box.sh <reps>
set -uo pipefail
REPS=${1:-30}
P=rtx-3090
CUDA=${CUDA:-/usr/local/cuda}
LOG=results.txt
say() { echo "$@" | tee -a "$LOG"; }
mkdir -p out/nsys out/cap out/smi
: > "$LOG"
say "== disasm $P $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.max.sm,clocks.max.mem,clocks.sm,clocks.mem,power.limit,temperature.gpu,persistence_mode,pcie.link.gen.max,pcie.link.width.max --format=csv | tee -a "$LOG"
say "nvcc $(nvcc --version | tail -1)"
lscpu | grep -E '^(Model name|CPU\(s\))' | sed 's/^/lscpu: /' | tee -a "$LOG"

[ -x tile_bench ] || { say "no tile_bench (run build.sh)"; exit 1; }
link() { cc "$1" mapal_gpu.c libmapal_rt.a -I"$CUDA/include" -L"$CUDA/lib64/stubs" -lcuda -lpthread -ldl -lm -o "$2" 2>&1 | tail -3; }

# Clock sampling across the whole timed part (achieved-clock evidence for peak math).
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 200 > out/smi/clocks.csv 2>&1 &
SMI_PID=$!

CELLS="16x2048x7168 16x1024x4096 16x18432x7168 128x2048x7168 4096x4096x4096"
# --- 1. kernel cells: every face of the integrated build + a null copy of the contract
# arm, cuBLAS, stock CUTLASS, tuned CUTLASS in one process; value gates first; two
# processes, the second with the arm order reversed (ABBA across processes too).
say "-- cells: tile_bench --splitk, reps=$REPS, two processes"
for c in $CELLS; do
    M=${c%%x*}; r=${c#*x}; N=${r%%x*}; K=${r#*x}
    cp $P/mm_${c}_c.ptx $P/mm_${c}_c.null.ptx; cp $P/mm_${c}_c.launch $P/mm_${c}_c.null.launch
    arms="mm_$c mm_${c}_c mm_${c}_rc mm_${c}_c.null"
    for pass in 1 2; do
        order=$arms; [ $pass = 2 ] && order=$(tr ' ' '\n' <<< "$arms" | tac | tr '\n' ' ')
        # shellcheck disable=SC2086
        ./tile_bench --splitk "$M" "$N" "$K" "$REPS" $P $order 2>&1 | sed "s/^/pass=$pass cell=$c /" | tee -a "$LOG"
        [ "${PIPESTATUS[0]}" -eq 0 ] || say "GATE FAILED cell=$c pass=$pass"
    done
done
say "cells done $(date -u +%T)"

# --- 2. MLP whole programs: value gate, then ABBA processes under lazy (driver default)
# and eager module loading; one traced process per arm for kernel ms.
say "-- mlp programs"
MLP_ARMS=${MLP_ARMS:-"pt_rm_m128_d256_h512 pt_rm_m128_d256_h512.base pt_rm_m128_d256_h512.off-grid-share pt_rm_m128_d256_h512.off-panel-regs pt_rm_m128_d256_h512_c pt_rm_m128_d256_h512_c.base pt_rm_m128_d1024_h2048 pt_rm_m128_d1024_h2048.base pt_rm_m128_d1024_h2048.off-grid-share pt_rm_m128_d1024_h2048.off-panel-regs pt_rm_m128_d1024_h2048_c pt_rm_m128_d1024_h2048_c.base"}
ok_arms=()
for s in $MLP_ARMS; do
    stem=${s%%.*}
    got=$(MAPAL_PTX=$P/$s.ptx ./prog_$s 2>/dev/null | grep -v '^iter ms=' | tr '\n' ' ')
    want=$(tr '\n' ' ' < tiles/$stem.expect)
    if [ "$got" = "$want" ]; then ok_arms+=("$s"); say "mlp_gate arm=$s pass out=[$got]"; else say "mlp_gate arm=$s FAIL got=[$got] want=[$want]"; fi
done
for r in $(seq "${MLP_REPS:-12}"); do
    order=("${ok_arms[@]}")
    if [ $((r % 2)) = 0 ]; then order=(); for ((i = ${#ok_arms[@]} - 1; i >= 0; i--)); do order+=("${ok_arms[$i]}"); done; fi
    for mode in lazy eager; do
        for s in "${order[@]}"; do
            if [ $mode = eager ]; then out=$(CUDA_MODULE_LOADING=EAGER MAPAL_PTX=$P/$s.ptx ./prog_$s 2>/dev/null | tr '\n' ' ')
            else out=$(MAPAL_PTX=$P/$s.ptx ./prog_$s 2>/dev/null | tr '\n' ' '); fi
            say "mlp arm=$s mode=$mode rep=$r out=[$out]"
        done
    done
done
for s in "${ok_arms[@]}"; do
    for mode in lazy eager; do
        if [ $mode = eager ]; then tr=$(CUDA_MODULE_LOADING=EAGER MAPAL_GPU_TRACE=1 MAPAL_PTX=$P/$s.ptx ./prog_$s 2>&1 >/dev/null)
        else tr=$(MAPAL_GPU_TRACE=1 MAPAL_PTX=$P/$s.ptx ./prog_$s 2>&1 >/dev/null); fi
        say "mlp_trace arm=$s mode=$mode $(grep -m1 mapal_gpu_trace_total <<< "$tr")"
        say "mlp_trace_ops arm=$s mode=$mode $(grep -E 'op=(launch|gen|load)' <<< "$tr" | sed 's/mapal_gpu_trace //' | tr '\n' '|')"
    done
done
# Library legs of the same programs, gated against Mapal's printed y[0], y[last].
for d in 256 1024; do
    h=$((d * 2)); vals=$(head -2 tiles/pt_rm_m128_d${d}_h${h}.expect | tr '\n' ' ')
    for pass in 1 2; do
        # shellcheck disable=SC2086
        [ -x rm_bench ] && ./rm_bench 128 $d $h "$REPS" $vals 2>&1 | sed "s/^/pass=$pass /" | tee -a "$LOG"
    done
done
say "mlp done $(date -u +%T)"
kill $SMI_PID 2>/dev/null

# --- 3. nsys: kernel names, grids, registers, shared memory, per-kernel time, API timeline.
for _ in $(seq 60); do grep -q nsys-install-done out/nsys-install.log 2>/dev/null && break; sleep 5; done
command -v nsys >/dev/null || export PATH=$PATH:$(dirname "$(ls /opt/nvidia/nsight-systems/*/bin/nsys "$CUDA"/bin/nsys 2>/dev/null | head -1)" 2>/dev/null)
if command -v nsys >/dev/null; then
    say "-- nsys $(nsys --version 2>&1 | head -1)"
    ns() { # <name> <cmd...>
        local n=$1; shift
        timeout 240 nsys profile -o out/nsys/$n --force-overwrite true -t cuda,osrt --cuda-memory-usage=false "$@" > out/nsys/$n.out 2>&1
        nsys stats -q -r cuda_gpu_trace --format csv out/nsys/$n.nsys-rep > out/nsys/$n.gpu_trace.csv 2>&1
        nsys stats -q -r cuda_gpu_kern_sum --format csv out/nsys/$n.nsys-rep > out/nsys/$n.kern_sum.csv 2>&1
        nsys stats -q -r cuda_api_trace --format csv out/nsys/$n.nsys-rep > out/nsys/$n.api_trace.csv 2>&1
        say "nsys $n rc=$? kernels=$(grep -c '' out/nsys/$n.gpu_trace.csv)"
    }
    for c in $CELLS; do
        M=${c%%x*}; r=${c#*x}; N=${r%%x*}; K=${r#*x}
        ns cell_$c ./tile_bench --splitk "$M" "$N" "$K" 3 $P mm_$c mm_${c}_c mm_${c}_rc
    done
    for s in pt_rm_m128_d256_h512 pt_rm_m128_d256_h512.base pt_rm_m128_d256_h512_c pt_rm_m128_d256_h512_c.base \
             pt_rm_m128_d1024_h2048 pt_rm_m128_d1024_h2048.base pt_rm_m128_d1024_h2048_c pt_rm_m128_d1024_h2048_c.base; do
        MAPAL_PTX=$P/$s.ptx ns prog_$s ./prog_$s
    done
    for d in 256 1024; do ns rm_$d ./rm_bench 128 $d $((d * 2)) 3; done
else
    say "nsys unavailable"
fi
say "nsys done $(date -u +%T)"

# --- 4. counter-free observer: the executed image of every leg, one fresh process each.
if [ -f capture.so ]; then
    say "-- observer captures"
    cap() { # <dir> <ptx> <cmd...>
        local d=$1 ptx=$2; shift 2
        rm -rf "out/cap/$d"; mkdir -p "out/cap/$(dirname "$d")"
        CUDA_INJECTION64_PATH=$PWD/capture.so MAPAL_CAPTURE_OUT=$PWD/out/cap/$d MAPAL_CAPTURE_PTX=$PWD/$ptx NVIDIA_TF32_OVERRIDE=0 \
            timeout 120 "$@" > "out/cap/$d.stdout" 2> "out/cap/$d.stderr"
        say "capture $d rc=$? images=$(ls out/cap/$d/*.cubin 2>/dev/null | wc -l) errors=$(grep -c '^error' out/cap/$d/events.tsv 2>/dev/null)"
    }
    for c in $CELLS; do
        M=${c%%x*}; r=${c#*x}; N=${r%%x*}; K=${r#*x}
        for leg in mm_$c mm_${c}_c mm_${c}_rc; do
            TB_PROFILE_LEG=$leg cap $c/$leg $P/$leg.ptx ./tile_bench --splitk "$M" "$N" "$K" 1 $P $leg
        done
        for leg in cublas cutlass cutlass_tuned; do
            TB_PROFILE_LEG=$leg cap $c/$leg $P/mm_${c}_c.ptx ./tile_bench --splitk "$M" "$N" "$K" 1 $P mm_${c}_c
        done
    done
    for d in 256 1024; do
        for leg in cublas lt cutlass cutlass_tuned; do
            TB_PROFILE_LEG=$leg cap rm_$d/$leg $P/pt_rm_m128_d256_h512_c.ptx ./rm_bench 128 $d $((d * 2)) 1
        done
    done
fi
say "== done $(date -u +%T)"
