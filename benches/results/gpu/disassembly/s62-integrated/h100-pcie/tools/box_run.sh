#!/bin/bash
# disasm-h100-pcie: one box session ON the GPU box, in the unpacked bundle (/w).
# 0. facts; nsys installs in the background while binaries link and values are gated
# 1. MLP whole programs: value gate (stdout == CPU oracle) for every arm
# 2. kernel cells: every Mapal face (+ null copies), cuBLAS, stock CUTLASS, tuned CUTLASS in ONE
#    tile_bench --splitk process (value gates first, ABBA inside), two processes (pass 2 reversed)
# 3. MLP: Mapal arms (on, null, base, on under CUDA_MODULE_LOADING=EAGER) x REPS processes ABBA,
#    traced processes for per-launch kernel ms; rm_bench library legs, two processes
# 4. nsys: kernel names/grids/regs/smem per cell; MLP timelines (API, kernels, gaps)
# 5. counter-free observer (CUPTI module-load capture): every loaded image + launch/kernel records
# Every line lands in results-<profile>.txt as produced; bulk evidence goes to trace/.
# usage: bash box_run.sh <profile> <sm_NN> [reps]
set -uo pipefail
PROF=$1; SM=$2; REPS=${3:-30}
CUDA=${CUDA:-/usr/local/cuda}
LOG=results-$PROF.txt
say() { echo "$@" | tee -a "$LOG"; }
: > "$LOG"; mkdir -p trace
export NVIDIA_TF32_OVERRIDE=0
say "== $PROF $SM $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,uuid,driver_version,clocks.max.sm,clocks.max.memory,clocks.sm,clocks.mem,power.limit,temperature.gpu,persistence_mode,pcie.link.gen.max,pcie.link.width.max --format=csv | tee -a "$LOG"
nvidia-smi -q > trace/nvidia-smi-q.txt 2>&1
lscpu | grep -E '^(Model name|CPU\(s\)|Socket)' | sed 's/^/lscpu: /' | tee -a "$LOG"
nvcc --version | tail -2 | tee -a "$LOG"

( apt-get update -qq && ver=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1-\2/p') &&
  { apt-get install -y -qq "cuda-nsight-systems-$ver" || apt-get install -y -qq nsight-systems; } ) > trace/nsys-install.log 2>&1 &
NSYS_PID=$!

CUPTI_INC=$(dirname "$(ls $CUDA/extras/CUPTI/include/cupti.h $CUDA/targets/x86_64-linux/include/cupti.h 2>/dev/null | head -1)")
CUPTI_LIB=$(dirname "$(ls $CUDA/extras/CUPTI/lib64/libcupti.so $CUDA/targets/x86_64-linux/lib/libcupti.so 2>/dev/null | head -1)")
[ -x tile_bench ] || nvcc -O3 -std=c++17 -arch="${SM%a}" -Icutlass/include -Icutlass/tools/util/include tile_bench.cu -lcublas -lcuda -o tile_bench > trace/build-tile_bench.log 2>&1 &
[ -x rm_bench ] || nvcc -O3 -std=c++17 -arch="${SM%a}" -Icutlass/include -Icutlass/tools/util/include rm_bench.cu -lcublasLt -lcublas -o rm_bench > trace/build-rm_bench.log 2>&1 &
[ -f capture.so ] || g++ -std=c++17 -O2 -fPIC -shared -I"$CUDA/include" -I"$CUPTI_INC" capture.cpp -L"$CUPTI_LIB" -Wl,-rpath,"$CUPTI_LIB" -lcupti -pthread -o capture.so > trace/build-capture.log 2>&1 &
link() { cc "$1" mapal_gpu.c libmapal_rt.a -I"$CUDA/include" -L"$CUDA/lib64/stubs" -lcuda -lpthread -ldl -lm -o "$2" 2>&1 | tail -3; }
PROGS="pt_rm_m128_d256_h512 pt_rm_m128_d1024_h2048"
for p in $PROGS; do for face in "" _c; do for arm in "" .null .base; do
    s=$p$face$arm; [ -f "$PROF/$s.host.o" ] && [ ! -x "bin_$s" ] && link "$PROF/$s.host.o" "bin_$s"
done; done; done
wait $(jobs -p | grep -v "^$NSYS_PID$") 2>/dev/null
for b in tile_bench rm_bench capture.so; do [ -e $b ] && say "built $b $(sha256sum $b | cut -c1-16)" || { say "BUILD FAILED $b"; tail -20 trace/build-${b%.so}.log | sed 's/^/build: /' | tee -a "$LOG"; }; done
# A failed patched harness falls back to the repository's (no tuned CUTLASS leg) so the session still measures.
[ -x tile_bench ] || { nvcc -O3 -std=c++17 -arch="${SM%a}" -Icutlass/include -Icutlass/tools/util/include tile_bench_orig.cu -lcublas -lcuda -o tile_bench > trace/build-tile_bench_orig.log 2>&1; say "fallback tile_bench_orig: $([ -x tile_bench ] && echo built || echo FAILED)"; }
[ -x tile_bench ] || { say "HARNESS BUILD FAILED"; exit 1; }

say "-- 1. MLP value gate (stdout == tiles/<prog><face>.expect, iter ms dropped)"
fail=0
for p in $PROGS; do for face in "" _c; do for arm in "" .null .base; do
    s=$p$face$arm; [ -x "bin_$s" ] || continue
    got=$(MAPAL_PTX="$PROF/$s.ptx" "./bin_$s" 2>&1 | grep -v '^iter ms=' | tr '\n' ' ')
    want=$(tr '\n' ' ' < "tiles/$p$face.expect")
    if [ "$got" = "$want" ]; then say "PASS $s $got"; else say "FAIL $s got=[$got] want=[$want]"; fail=$((fail + 1)); fi
done; done; done
say "mlp value gate: $fail fail"

wait "$NSYS_PID"; export PATH=$PATH:$(dirname "$(ls /opt/nvidia/nsight-systems/*/bin/nsys $CUDA/bin/nsys 2>/dev/null | head -1)" 2>/dev/null)
command -v nsys >/dev/null && say "nsys $(nsys --version 2>&1 | tail -1)" || say "nsys unavailable"

CELLS="16x2048x7168 16x1024x4096 16x18432x7168 128x2048x7168 4096x4096x4096"
arms_of() { local c=mm_$1 a=""; for s in $c $c.null ${c}_c ${c}_c.null ${c}_r ${c}_rc; do [ -f "$PROF/$s.launch" ] && a="$a $s"; done; echo $a; }
say "-- 2. kernel cells: Mapal faces + null, cuBLAS, CUTLASS stock, CUTLASS tuned; reps=$REPS, two processes ABBA"
for cell in $CELLS; do
    M=${cell%%x*}; r=${cell#*x}; N=${r%%x*}; K=${r#*x}
    arms=$(arms_of "$cell")
    for pass in 1 2; do
        order=$arms; [ $pass = 2 ] && order=$(tr ' ' '\n' <<< "$arms" | tac | tr '\n' ' ')
        # shellcheck disable=SC2086
        ./tile_bench --splitk "$M" "$N" "$K" "$REPS" "$PROF" $order 2>&1 | sed "s/^/pass=$pass cell=$cell /" | tee -a "$LOG"
        [ "${PIPESTATUS[0]}" -eq 0 ] || say "CELL GATE FAILED cell=$cell pass=$pass"
    done
done

say "-- 3. MLP whole programs: iter ms per process (arms ABBA), traced kernel ms, library legs"
runs=()
for p in $PROGS; do for face in "" _c; do for arm in on null base eager; do
    s=$p$face; case $arm in null) s=$s.null ;; base) s=$s.base ;; esac
    [ -x "bin_$s" ] && runs+=("$p$face:$arm")
done; done; done
for r in $(seq "$REPS"); do
    order=("${runs[@]}")
    if [ $((r % 2)) = 0 ]; then order=(); for ((i = ${#runs[@]} - 1; i >= 0; i--)); do order+=("${runs[$i]}"); done; fi
    for x in "${order[@]}"; do
        pf=${x%%:*}; arm=${x#*:}; s=$pf; env=()
        case $arm in null) s=$s.null ;; base) s=$s.base ;; eager) env=(CUDA_MODULE_LOADING=EAGER) ;; esac
        t0=$(date +%s%N); out=$(env ${env[@]+"${env[@]}"} MAPAL_PTX="$PROF/$s.ptx" "./bin_$s" 2>/dev/null | tr '\n' ' '); t1=$(date +%s%N)
        say "mlp prog=$pf arm=$arm rep=$r wall_us=$(( (t1 - t0) / 1000 )) out=[$out]"
        if [ "$r" -le 3 ]; then
            tr=$(env ${env[@]+"${env[@]}"} MAPAL_GPU_TRACE=1 MAPAL_PTX="$PROF/$s.ptx" "./bin_$s" 2>&1 >/dev/null)
            say "mlp_trace prog=$pf arm=$arm rep=$r $(grep -o 'op=launch kernel=mapal_tile_[0-9a-f]* ms=[0-9.]*' <<< "$tr" | sed 's/op=launch kernel=mapal_tile_//' | tr '\n' ' ') $(grep -m1 mapal_gpu_trace_total <<< "$tr")"
        fi
    done
done
[ -x rm_bench ] && for p in $PROGS; do
    d=${p#pt_rm_m128_d}; d=${d%%_*}; h=${p##*_h}
    vals=$(MAPAL_PTX="$PROF/${p}_c.ptx" "./bin_${p}_c" 2>/dev/null | head -2 | tr '\n' ' ')
    for pass in 1 2; do
        # shellcheck disable=SC2086
        ./rm_bench 128 "$d" "$h" "$REPS" $vals 2>&1 | sed "s/^/pass=$pass prog=$p /" | tee -a "$LOG"
    done
done

if command -v nsys >/dev/null; then
    say "-- 4. nsys"
    stats() { local b=$1 r; shift; for r in "$@"; do nsys stats -q -r "$r" --format csv -o "$b" "$b.nsys-rep" > /dev/null 2>&1; done; }
    for cell in $CELLS; do
        M=${cell%%x*}; r=${cell#*x}; N=${r%%x*}; K=${r#*x}
        arms=$(for s in mm_$cell mm_${cell}_c mm_${cell}_r mm_${cell}_rc; do [ -f "$PROF/$s.launch" ] && echo -n "$s "; done)
        # shellcheck disable=SC2086
        timeout 300 nsys profile -t cuda -o "trace/nsys_cell_$cell" --force-overwrite true ./tile_bench --splitk "$M" "$N" "$K" 3 "$PROF" $arms > "trace/nsys_cell_$cell.out" 2>&1
        stats "trace/nsys_cell_$cell" cuda_gpu_trace cuda_gpu_kern_sum
        say "nsys cell=$cell files=$(ls trace/nsys_cell_${cell}_* 2>/dev/null | wc -l)"
    done
    for p in $PROGS; do for face in "" _c; do for arm in on base eager; do
        s=$p$face; env=(); case $arm in base) s=$s.base ;; eager) env=(CUDA_MODULE_LOADING=EAGER) ;; esac
        t=trace/nsys_mlp_$p$face.$arm
        env ${env[@]+"${env[@]}"} MAPAL_PTX="$PROF/$s.ptx" timeout 300 nsys profile -t cuda,osrt -o "$t" --force-overwrite true "./bin_$s" > "$t.out" 2>&1
        stats "$t" cuda_api_trace cuda_gpu_trace cuda_kern_exec_trace
        say "nsys mlp=$p$face arm=$arm out=[$(tr '\n' ' ' < "$t.out" | cut -c1-200)]"
    done; done; done
    for p in $PROGS; do
        d=${p#pt_rm_m128_d}; d=${d%%_*}; h=${p##*_h}; t=trace/nsys_rmlib_$p
        timeout 300 nsys profile -t cuda,osrt -o "$t" --force-overwrite true ./rm_bench 128 "$d" "$h" 5 > "$t.out" 2>&1
        stats "$t" cuda_api_trace cuda_gpu_trace cuda_kern_exec_trace
        say "nsys rmlib=$p files=$(ls ${t}_* 2>/dev/null | wc -l)"
    done
fi

if [ -f capture.so ]; then
    say "-- 5. counter-free observer"
    for cell in $CELLS; do
        M=${cell%%x*}; r=${cell#*x}; N=${r%%x*}; K=${r#*x}
        arms=$(for s in mm_$cell mm_${cell}_c mm_${cell}_r mm_${cell}_rc; do [ -f "$PROF/$s.launch" ] && echo -n "$s "; done)
        o=trace/cap_cell_$cell
        # shellcheck disable=SC2086
        CUDA_INJECTION64_PATH=$PWD/capture.so MAPAL_CAPTURE_OUT=$o MAPAL_CAPTURE_PTX="$PROF/mm_${cell}_c.ptx" TB_REGION=1 \
            timeout 120 ./tile_bench --splitk "$M" "$N" "$K" 1 "$PROF" $arms > "$o.out" 2>&1
        say "capture cell=$cell rc=$? images=$(ls $o/module-*.cubin 2>/dev/null | wc -l) bytes=$(du -sb $o 2>/dev/null | cut -f1) errors=[$(grep -h '^error' $o/events.tsv 2>/dev/null | sort | uniq -c | tr '\n' ' ')] $(grep -h '^finish' $o/events.tsv 2>/dev/null)"
    done
    for p in $PROGS; do
        d=${p#pt_rm_m128_d}; d=${d%%_*}; h=${p##*_h}; o=trace/cap_rmlib_$p
        CUDA_INJECTION64_PATH=$PWD/capture.so MAPAL_CAPTURE_OUT=$o MAPAL_CAPTURE_PTX="$PROF/${p}_c.ptx" TB_REGION=1 \
            timeout 120 ./rm_bench 128 "$d" "$h" 1 > "$o.out" 2>&1
        say "capture rmlib=$p rc=$? images=$(ls $o/module-*.cubin 2>/dev/null | wc -l) errors=[$(grep -h '^error' $o/events.tsv 2>/dev/null | sort | uniq -c | tr '\n' ' ')] $(grep -h '^finish' $o/events.tsv 2>/dev/null)"
        for face in "" _c; do
            o=trace/cap_mlp_$p$face
            CUDA_INJECTION64_PATH=$PWD/capture.so MAPAL_CAPTURE_OUT=$o MAPAL_CAPTURE_PTX="$PROF/$p$face.ptx" MAPAL_PTX="$PROF/$p$face.ptx" \
                timeout 120 "./bin_$p$face" > "$o.out" 2>&1
            say "capture mlp=$p$face rc=$? images=$(ls $o/module-*.cubin 2>/dev/null | wc -l) $(grep -h '^finish' $o/events.tsv 2>/dev/null)"
        done
    done
fi
cp tile_bench rm_bench capture.so trace/ 2>/dev/null
say "== done $(date -u +%FT%TZ)"
