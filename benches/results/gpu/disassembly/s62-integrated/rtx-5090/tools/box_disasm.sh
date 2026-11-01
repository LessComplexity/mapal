#!/bin/bash
# disasm-rtx-5090, ON the box in /w: bash box_disasm.sh <stage>
#   setup   GPU facts, nsys install (background), harness/observer builds, ptxas -v + SASS of
#           every Mapal PTX and of both harness binaries, the Mapal whole programs linked
#   time    per GEMM cell, every Mapal arm + cuBLAS + stock CUTLASS + tuned CUTLASS in one
#           process (value gates first), two processes, the second in reversed arm order;
#           the MLP harness twice (second reversed); Mapal whole MLP programs, 30 processes
#           per arm (on/base, Exact/Contract, CUDA lazy vs eager module loading), ABBA
#   nsys    kernel names, grids, registers, smem and times of every implementation per cell;
#           the whole MLP programs' API + kernel timeline
#   observe the repaired counter-free observer around every leg of every cell and the MLP
# Every line lands in results.txt the moment it is produced.
set -uo pipefail
cd /w
export PATH=$PATH:/usr/local/cuda/bin NVIDIA_TF32_OVERRIDE=0
P=rtx-5090; LOG=results.txt
say() { echo "$@" | tee -a "$LOG"; }
CELLS="16x2048x7168 16x1024x4096 16x18432x7168 128x2048x7168 4096x4096x4096"
MLPS="128x256x512 128x1024x2048"
arms() { local c=mm_$1; echo "$c $c.null ${c}_c ${c}_c.null ${c}_r ${c}_rc"; }
mlp_arms() { local p=$1; echo "$p ${p}_c ${p}_r ${p}_rc $p.base ${p}_c.base ${p}_c.null"; }
nsys_bin() { ls /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/cuda/bin/nsys 2>/dev/null | head -1; }

case $1 in
setup)
    : > "$LOG"
    say "== $P setup $(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=name,uuid,driver_version,clocks.max.sm,clocks.max.mem,power.limit,pcie.link.gen.max,pcie.link.width.max --format=csv | tee -a "$LOG"
    nvcc --version | tail -1 | tee -a "$LOG"
    ( apt-get update -qq && apt-get install -y -qq cuda-nsight-systems-12-8 ) > nsys-install.log 2>&1 &
    mkdir -p static
    nvcc -O3 -std=c++17 -arch=sm_120 -Icutlass/include -Icutlass/tools/util/include dis_bench.cu -lcublas -lcuda -o dis_bench > build-dis.log 2>&1 &
    nvcc -O3 -std=c++17 -arch=sm_120 -Icutlass/include -Icutlass/tools/util/include mlp_bench.cu -lcublasLt -lcublas -lcuda -o mlp_bench > build-mlp.log 2>&1 &
    c++ -std=c++17 -O2 -Wall -Wextra -Werror -fPIC -shared -I/usr/local/cuda/include -I/usr/local/cuda/extras/CUPTI/include -I/usr/local/cuda/targets/x86_64-linux/include \
        observer/capture.cpp -L/usr/local/cuda/extras/CUPTI/lib64 -L/usr/local/cuda/targets/x86_64-linux/lib -L/usr/local/cuda/lib64 \
        -Wl,-rpath,/usr/local/cuda/extras/CUPTI/lib64 -Wl,-rpath,/usr/local/cuda/targets/x86_64-linux/lib \
        -lcupti -pthread -o capture.so > build-obs.log 2>&1 &
    for p in $P/*.ptx; do
        v=$(basename "$p" .ptx)
        ptxas -arch=sm_120 -O3 -v "$p" -o "static/$v.cubin" > "static/$v.ptxas" 2>&1
        cuobjdump -sass "static/$v.cubin" > "static/$v.sass" 2>&1
    done
    say "static: $(ls static/*.sass | wc -l) Mapal SASS files"
    for p in pt_rm_m128_d256_h512 pt_rm_m128_d1024_h2048; do
        for v in "" _c _r _rc .base _c.base; do
            cc "$P/$p$v.host.o" mapal_gpu.c libmapal_rt.a -I/usr/local/cuda/include -L/usr/local/cuda/lib64/stubs \
                -lcuda -lpthread -ldl -lm -o "prog_$p$v" >> build-prog.log 2>&1 || say "LINK FAILED $p$v"
        done
    done
    wait
    for b in dis_bench mlp_bench capture.so; do [ -e $b ] && say "built $b" || { say "BUILD FAILED $b"; tail -20 build-*.log | tee -a "$LOG"; }; done
    for b in dis_bench mlp_bench; do
        [ -x $b ] || continue
        cuobjdump -sass $b > static/$b.sass 2>&1
        cuobjdump -res-usage $b > static/$b.res 2>&1
    done
    say "nsys: $(nsys_bin) $(tail -1 nsys-install.log)"
    say "== setup done $(date -u +%FT%TZ)" ;;
time)
    say "== time $(date -u +%FT%TZ) $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader)"
    for c in $CELLS; do
        M=${c%%x*}; r=${c#*x}; N=${r%%x*}; K=${r#*x}
        a=$(arms "$c")
        for pass in 1 2; do
            order=$a; [ $pass = 2 ] && order=$(tr ' ' '\n' <<< "$a" | tac | tr '\n' ' ')
            # shellcheck disable=SC2086
            ./dis_bench --splitk "$M" "$N" "$K" 30 "$P" $order 2>&1 | sed "s/^/pass=$pass cell=$c /" | tee -a "$LOG"
        done
    done
    for m in $MLPS; do
        M=${m%%x*}; r=${m#*x}; d=${r%%x*}; h=${r#*x}; p=pt_rm_m${M}_d${d}_h${h}
        for pass in 1 2; do
            rev=; [ $pass = 2 ] && rev=1
            # shellcheck disable=SC2046
            DIS_REVERSE=$rev ./mlp_bench "$M" "$d" "$h" 30 "$P" $(mlp_arms $p) 2>&1 | sed "s/^/pass=$pass mlp=$p /" | tee -a "$LOG"
        done
    done
    say "-- whole MLP programs: iter ms, 30 processes per arm, ABBA"
    for m in $MLPS; do
        M=${m%%x*}; r=${m#*x}; d=${r%%x*}; h=${r#*x}; p=pt_rm_m${M}_d${d}_h${h}
        runs=()
        for v in "" _c .base _c.base; do for ld in LAZY EAGER; do runs+=("$v:$ld"); done; done
        for v in "" _c .base _c.base; do   # value gate first
            got=$(MAPAL_PTX="$P/$p$v.ptx" ./prog_$p$v 2>/dev/null | head -2 | tr '\n' ' ')
            [ "$got" = "$(head -2 tiles/$p${v%.base}.expect | tr '\n' ' ')" ] && say "value-gate prog=$p$v pass" || say "VALUE-FAIL prog=$p$v got=[$got]"
            MAPAL_GPU_TRACE=1 MAPAL_PTX="$P/$p$v.ptx" ./prog_$p$v > /dev/null 2> trace_$p$v.txt
            say "trace prog=$p$v $(grep -E 'op=launch|mapal_gpu_trace_total' trace_$p$v.txt | tr '\n' ' ')"
        done
        for rep in $(seq 30); do
            order=("${runs[@]}")
            if [ $((rep % 2)) = 0 ]; then order=(); for ((i = ${#runs[@]} - 1; i >= 0; i--)); do order+=("${runs[$i]}"); done; fi
            for x in "${order[@]}"; do
                v=${x%%:*}; ld=${x#*:}
                t0=$(date +%s%N)
                out=$(CUDA_MODULE_LOADING=$ld MAPAL_PTX="$P/$p$v.ptx" ./prog_$p$v 2>/dev/null | tr '\n' ' ')
                t1=$(date +%s%N)
                say "prog prog=$p$v loading=$ld rep=$rep wall_us=$(( (t1 - t0) / 1000 )) out=[$out]"
            done
        done
    done
    say "== time done $(date -u +%FT%TZ)" ;;
nsys)
    N=$(nsys_bin); [ -n "$N" ] || { say "nsys unavailable: $(tail -3 nsys-install.log)"; exit 0; }
    mkdir -p nsys
    for m in $MLPS; do
        M=${m%%x*}; r=${m#*x}; d=${r%%x*}; h=${r#*x}; p=pt_rm_m${M}_d${d}_h${h}
        # shellcheck disable=SC2046
        timeout 240 "$N" profile -o nsys/mlp_$p --force-overwrite true -t cuda ./mlp_bench "$M" "$d" "$h" 5 "$P" $(mlp_arms $p) > nsys/mlp_$p.out 2>&1
        "$N" stats -q --format csv -r cuda_gpu_trace -o nsys/mlp_$p nsys/mlp_$p.nsys-rep > /dev/null 2>&1
        for v in "" _c .base _c.base; do
            MAPAL_PTX="$P/$p$v.ptx" timeout 240 "$N" profile -o nsys/prog_$p$v --force-overwrite true -t cuda,osrt ./prog_$p$v > nsys/prog_$p$v.out 2>&1
            "$N" stats -q --format csv -r cuda_api_trace,cuda_gpu_trace -o nsys/prog_$p$v nsys/prog_$p$v.nsys-rep > /dev/null 2>&1
        done
        say "nsys mlp=$p $(ls nsys/*$p* | wc -l) files"
    done
    for c in $CELLS; do
        M=${c%%x*}; r=${c#*x}; Nn=${r%%x*}; K=${r#*x}
        # shellcheck disable=SC2046
        timeout 240 "$N" profile -o nsys/cell_$c --force-overwrite true -t cuda ./dis_bench --splitk "$M" "$Nn" "$K" 5 "$P" $(arms "$c") > nsys/cell_$c.out 2>&1
        "$N" stats -q --format csv -r cuda_gpu_trace -o nsys/cell_$c nsys/cell_$c.nsys-rep > /dev/null 2>&1
        say "nsys cell=$c $(ls nsys/cell_${c}_* 2>/dev/null | tr '\n' ' ')"
    done
    say "== nsys done $(date -u +%FT%TZ)" ;;
observe)
    mkdir -p obs
    for c in $CELLS; do
        M=${c%%x*}; r=${c#*x}; Nn=${r%%x*}; K=${r#*x}
        # shellcheck disable=SC2046
        say "observe $(python3 obs.py obs/cell_$c "$P/mm_${c}_c.ptx" /w/capture.so -- /w/dis_bench --splitk "$M" "$Nn" "$K" 1 "$P" $(arms "$c") 2>&1 | tail -1)"
    done
    for m in $MLPS; do
        M=${m%%x*}; r=${m#*x}; d=${r%%x*}; h=${r#*x}; p=pt_rm_m${M}_d${d}_h${h}
        # shellcheck disable=SC2046
        say "observe $(python3 obs.py obs/mlp_$p "$P/${p}_c.ptx" /w/capture.so -- /w/mlp_bench "$M" "$d" "$h" 1 "$P" $(mlp_arms $p) 2>&1 | tail -1)"
    done
    say "== observe done $(date -u +%FT%TZ)" ;;
pack)
    tar czf /w/out.tgz results.txt static nsys obs trace_*.txt build-*.log nsys-install.log 2>/dev/null
    ls -la /w/out.tgz ;;
esac
