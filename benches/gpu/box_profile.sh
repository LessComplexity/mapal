#!/bin/bash
# Runs ON the GPU box after box_run.sh: the gap evidence that needs no
# performance counters (vast containers refuse them: ERR_NVGPUCTRPERM).
#   1. ptxas resources (registers, spills, smem) of every n=4096 FMA variant
#   2. SASS: Mapal's kernel, and the harness binary (CUTLASS's kernel)
#   3. the kernels cuBLAS and CUTLASS actually launch, and their times (nsys)
#   4. the host side: per-operation MAPAL_GPU_TRACE and the nsys timeline of the
#      e2e programs
# Everything lands in prof/, tarred to prof-<profile>.tgz.
# usage: bash box_profile.sh <profile> <sm_NN>
set -uo pipefail
PROF=$1; SM=$2
CUDA=${CUDA:-/usr/local/cuda}
mkdir -p prof

for p in "$PROF"/mm_4096[._]*.ptx; do
    v=$(basename "$p" .ptx)
    ptxas -arch="$SM" -O3 -v "$p" -o "prof/$v.cubin" > "prof/$v.ptxas" 2>&1
done
for c in prof/mm_4096[._]*.cubin; do cuobjdump -sass "$c" > "${c%.cubin}.sass" 2>&1; done
cp prof/mm_4096_c.sass prof/mapal.sass
cuobjdump -sass tile_bench > prof/tile_bench.sass 2>&1

if ! command -v nsys >/dev/null; then
    ver=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1-\2/p')
    apt-get install -y -qq "cuda-nsight-systems-$ver" >/dev/null 2>&1 || apt-get install -y -qq nsight-systems >/dev/null 2>&1
    export PATH=$PATH:$(dirname "$(ls /opt/nvidia/nsight-systems/*/bin/nsys "$CUDA"/bin/nsys 2>/dev/null | head -1)" 2>/dev/null)
fi
if command -v nsys >/dev/null; then
    # 1024 and 8192 too (S53): which kernel cuBLAS picks at each end.
    for n in 1024 4096 8192; do
        read -r k gx gy blk sm < "$PROF/mm_${n}_c.launch"
        # S53: nsys hung for 21 min on one 4090 host; never let it hold a rented box.
        timeout 300 nsys profile -o "prof/run-$n" --force-overwrite true -t cuda \
            ./tile_bench $n $n $n "$PROF/mm_${n}_c.ptx" "$k" "$gx" "$gy" "$blk" "$sm" 5 --contract > "prof/nsys-$n.out" 2>&1
        nsys stats -q -r cuda_gpu_kern_sum --format csv "prof/run-$n.nsys-rep" > "prof/kern-$n.csv" 2>&1
    done
    for v in "$PROF"/mm_4096_tf.launch "$PROF"/mm_4096_tf.off-wgmma-hopper.launch; do
        [ -f "$v" ] || continue
        s=$(basename "$v" .launch)
        read -r k gx gy blk sm < "$v"
        timeout 300 nsys profile -o "prof/run_$s" --force-overwrite true -t cuda \
            ./tile_bench 4096 4096 4096 "$PROF/$s.ptx" "$k" "$gx" "$gy" "$blk" "$sm" 5 --tf32 > "prof/nsys_$s.out" 2>&1
        nsys stats -q -r cuda_gpu_kern_sum --format csv "prof/run_$s.nsys-rep" > "prof/kern_$s.csv" 2>&1
    done
else
    echo "nsys unavailable" > prof/kern.csv
fi

# The per-operation trace of the e2e programs (box_run.sh keeps the totals):
# this build and the S52 host object (`.s52`, PREV), one process each.
for b in e2e_1024 e2e_1024_c e2e_4096 e2e_4096_c e2e_1024.s52 e2e_1024_c.s52 e2e_4096.s52 e2e_4096_c.s52; do
    [ -f "$PROF/$b.host.o" ] || continue
    cc "$PROF/$b.host.o" mapal_gpu.c libmapal_rt.a -I"$CUDA/include" -L"$CUDA/lib64/stubs" \
        -lcuda -lpthread -ldl -lm -o "prof/$b" 2>/dev/null || continue
    MAPAL_GPU_TRACE=1 MAPAL_PTX="$PROF/$b.ptx" "prof/$b" > "prof/$b.out" 2>&1
done
# The launch timeline of whole Mapal programs: API calls (launch overhead, JIT,
# allocation), kernels, transfers, and the gaps between them — what a program
# pays around its kernels, not just inside them.
if command -v nsys >/dev/null; then
    for b in e2e_1024_c e2e_4096_c; do
        [ -x "prof/$b" ] || continue
        MAPAL_PTX="$PROF/$b.ptx" nsys profile -o "prof/tl_$b" --force-overwrite true -t cuda,osrt \
            "prof/$b" > "prof/tl_$b.out" 2>&1
        nsys stats -q --format csv -r cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_trace \
            "prof/tl_$b.nsys-rep" > "prof/tl_$b.csv" 2>&1
    done
fi
tar czf "prof-$PROF.tgz" prof
echo "profile done: $(ls prof | wc -l) files"
