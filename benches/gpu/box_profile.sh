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

# (S62: and every `mm_<M>x<N>x<K>` cell's arms.)
for p in "$PROF"/mm_4096[._]*.ptx "$PROF"/mm_*x*x*[._]*.ptx; do
    [ -f "$p" ] || continue
    v=$(basename "$p" .ptx)
    ptxas -arch="$SM" -O3 -v "$p" -o "prof/$v.cubin" > "prof/$v.ptxas" 2>&1
done
for c in prof/mm_4096[._]*.cubin prof/mm_*x*x*[._]*.cubin; do [ -f "$c" ] && cuobjdump -sass "$c" > "${c%.cubin}.sass" 2>&1; done
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
        [ -s "$PROF/mm_${n}_c.launch" ] || continue
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
# S54: three multi-computation probes (an epilogue, a chain through a per-cell
# step, two independent sites) — where the transfers and host loops sit.
PROBES=${PROF_PROBES:-"pt_epi_4096_c pt_mlp_4096_c pt_indep_4096_c"}
for b in e2e_1024 e2e_1024_c e2e_4096 e2e_4096_c e2e_1024.s52 e2e_1024_c.s52 e2e_4096.s52 e2e_4096_c.s52 $PROBES; do
    [ -f "$PROF/$b.host.o" ] || continue
    cc "$PROF/$b.host.o" mapal_gpu.c libmapal_rt.a -I"$CUDA/include" -L"$CUDA/lib64/stubs" \
        -lcuda -lpthread -ldl -lm -o "prof/$b" 2>/dev/null || continue
    MAPAL_GPU_TRACE=1 MAPAL_PTX="$PROF/$b.ptx" "prof/$b" > "prof/$b.out" 2>&1
done
# The launch timeline of whole Mapal programs: API calls (launch overhead, JIT,
# allocation), kernels, transfers, and the gaps between them — what a program
# pays around its kernels, not just inside them.
if command -v nsys >/dev/null; then
    for b in e2e_1024_c e2e_4096_c $PROBES; do
        [ -x "prof/$b" ] || continue
        MAPAL_PTX="$PROF/$b.ptx" timeout 300 nsys profile -o "prof/tl_$b" --force-overwrite true -t cuda,osrt \
            "prof/$b" > "prof/tl_$b.out" 2>&1
        nsys stats -q --format csv -r cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_trace \
            "prof/tl_$b.nsys-rep" > "prof/tl_$b.csv" 2>&1
    done
fi
# S64 evidence pass: the library kernels each cell and MLP actually launches (cuBLAS, cuBLASLt, stock
# and tuned CUTLASS): names, grid, block, registers, smem and time per launch (nsys cuda_gpu_trace).
if command -v nsys >/dev/null; then
    for l in "$PROF"/mm_*x*x*_r.launch; do
        [ -f "$l" ] || continue
        cell=$(basename "$l" _r.launch); d=${cell#mm_}; M=${d%%x*}; d=${d#*x}; N=${d%%x*}; K=${d#*x}
        timeout 300 nsys profile -o "prof/lib_$cell" --force-overwrite true -t cuda \
            ./tile_bench --splitk "$M" "$N" "$K" 3 "$PROF" "$cell" > "prof/lib_$cell.out" 2>&1
        nsys stats -q --format csv -r cuda_gpu_kern_sum,cuda_gpu_trace "prof/lib_$cell.nsys-rep" > "prof/lib_$cell.csv" 2>&1
    done
    for k in "$PROF"/pt_rm_m*_d*_h*_c.k; do
        [ -f "$k" ] && [ -x mlp_bench ] || continue
        p=$(basename "$k" _c.k); dims=$(sed -nE 's/^pt_rm_m([0-9]+)_d([0-9]+)_h([0-9]+)$/\1 \2 \3/p' <<< "$p")
        # shellcheck disable=SC2086  # the dims are separate arguments
        timeout 300 nsys profile -o "prof/lib_$p" --force-overwrite true -t cuda ./mlp_bench $dims 3 "$PROF" "${p}_c" \
            > "prof/lib_$p.out" 2>&1
        nsys stats -q --format csv -r cuda_gpu_kern_sum,cuda_gpu_trace "prof/lib_$p.nsys-rep" > "prof/lib_$p.csv" 2>&1
    done
fi
# S64 evidence pass: the driver's JIT image vs offline ptxas on one cell (both step faces): the
# static SASS stands for the executed one only where their opcode sequences agree (S62: 36/36 on
# the H100). capture.cpp (benches/gpu/capture_images) saves the module the driver built from
# exactly that PTX; `evidence.py jit` compares it, function by function.
JIT_CELL=${JIT_CELL:-16x2048x7168}
if [ -f capture.cpp ] && [ -f evidence.py ] && [ -f "$PROF/mm_$JIT_CELL.launch" ]; then
    inc=$(dirname "$(ls "$CUDA"/extras/CUPTI/include/cupti.h "$CUDA"/targets/x86_64-linux/include/cupti.h 2>/dev/null | head -1)")
    lib=$(dirname "$(ls "$CUDA"/extras/CUPTI/lib64/libcupti.so "$CUDA"/targets/x86_64-linux/lib/libcupti.so 2>/dev/null | head -1)")
    g++ -std=c++17 -O2 -fPIC -shared -I"$CUDA/include" -I"$inc" capture.cpp -L"$lib" -Wl,-rpath,"$lib" -lcupti \
        -pthread -o capture.so > prof/capture-build.log 2>&1
    M=${JIT_CELL%%x*}; r=${JIT_CELL#*x}; N=${r%%x*}; K=${r#*x}
    for s in "mm_$JIT_CELL" "mm_${JIT_CELL}_x"; do
        o=prof/jit_$s; rm -rf "$o"
        CUDA_INJECTION64_PATH=$PWD/capture.so MAPAL_CAPTURE_OUT=$o MAPAL_CAPTURE_PTX="$PROF/$s.ptx" \
            timeout 120 ./tile_bench --splitk "$M" "$N" "$K" 1 "$PROF" "$s" > "$o.out" 2>&1
        for c in "$o"/module-*.cubin; do [ -f "$c" ] && cuobjdump -sass "$c" > "${c%.cubin}.sass"; done
        ptxas -arch="$SM" -O3 "$PROF/$s.ptx" -o "$o.ptxas.cubin" && cuobjdump -sass "$o.ptxas.cubin" > "$o.ptxas.sass"
        echo "jit $s images=$(ls "$o"/module-*.cubin 2>/dev/null | wc -l) $(python3 evidence.py jit "$o" "$o.ptxas.sass" 2>&1 | tail -1)" |
            tee -a "results-$PROF.txt"
    done
fi
tar czf "prof-$PROF.tgz" prof
echo "profile done: $(ls prof | wc -l) files"
