#!/bin/bash
# Runs ON the GPU box after box_run.sh: the gap evidence that needs no
# performance counters (vast containers refuse them: ERR_NVGPUCTRPERM).
#   1. ptxas resources (registers, spills, smem) of every n=4096 FMA variant
#   2. SASS: Mapal's kernel, and the harness binary (CUTLASS's kernel)
#   3. the kernels cuBLAS and CUTLASS actually launch, and their times (nsys)
#   4. the host side of a site: MAPAL_GPU_TRACE phases of the e2e programs
# Everything lands in prof/, tarred to prof-<profile>.tgz.
# usage: bash box_profile.sh <profile> <sm_NN>
set -uo pipefail
PROF=$1; SM=$2
CUDA=${CUDA:-/usr/local/cuda}
mkdir -p prof

for p in "$PROF"/mm_4096_*.ptx; do
    v=$(basename "$p" .ptx)
    ptxas -arch="$SM" -O3 -v "$p" -o "prof/$v.cubin" > "prof/$v.ptxas" 2>&1
done
for c in prof/mm_4096_*.cubin; do cuobjdump -sass "$c" > "${c%.cubin}.sass" 2>&1; done
cp prof/mm_4096_c.sass prof/mapal.sass
cuobjdump -sass tile_bench > prof/tile_bench.sass 2>&1

if ! command -v nsys >/dev/null; then
    ver=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1-\2/p')
    apt-get install -y -qq "cuda-nsight-systems-$ver" >/dev/null 2>&1 || apt-get install -y -qq nsight-systems >/dev/null 2>&1
    export PATH=$PATH:$(dirname "$(ls /opt/nvidia/nsight-systems/*/bin/nsys "$CUDA"/bin/nsys 2>/dev/null | head -1)" 2>/dev/null)
fi
if command -v nsys >/dev/null; then
    read -r k gx gy blk sm < "$PROF/mm_4096_c.launch"
    nsys profile -o prof/run --force-overwrite true -t cuda \
        ./tile_bench 4096 4096 4096 "$PROF/mm_4096_c.ptx" "$k" "$gx" "$gy" "$blk" "$sm" 5 --contract > prof/nsys.out 2>&1
    nsys stats -q -r cuda_gpu_kern_sum --format csv prof/run.nsys-rep > prof/kern.csv 2>&1
else
    echo "nsys unavailable" > prof/kern.csv
fi

# The e2e programs with each host-runtime rung flipped (S53): process wall time
# and the trace phases, interleaved (the arm order rotates every round). Every
# line lands in results-<profile>.txt as it is produced.
ARMS=(on off-eager-init on-pool)
for n in 1024 4096; do
    for face in "" _c; do
        for arm in "${ARMS[@]}"; do
            b=e2e_$n$face; [ "$arm" = on ] || b=$b.$arm
            cc "$PROF/$b.host.o" mapal_gpu.c libmapal_rt.a -I"$CUDA/include" -L"$CUDA/lib64/stubs" \
                -lcuda -lpthread -ldl -lm -o "prof/$b" 2>/dev/null
        done
        for r in $(seq "${ROUNDS:-9}"); do
            for i in "${!ARMS[@]}"; do
                arm=${ARMS[$(( (i + r) % ${#ARMS[@]} ))]}
                b=e2e_$n$face; [ "$arm" = on ] || b=$b.$arm
                [ -x "prof/$b" ] || continue
                t0=$(date +%s%N)
                out=$(MAPAL_GPU_TRACE=1 MAPAL_PTX="$PROF/$b.ptx" "prof/$b" 2>&1)
                t1=$(date +%s%N)
                echo "$out" >> "prof/$b.out"
                echo "e2e n=$n face=${face:-_exact} arm=$arm r=$r wall_us=$(( (t1 - t0) / 1000 ))" \
                    "value=$(grep -v mapal_gpu <<< "$out" | head -1)" \
                    "$(grep mapal_gpu_trace <<< "$out" | sed 's/^mapal_gpu_trace //')" | tee -a "results-$PROF.txt"
            done
        done
    done
done
tar czf "prof-$PROF.tgz" prof
echo "profile done: $(ls prof | wc -l) files"
