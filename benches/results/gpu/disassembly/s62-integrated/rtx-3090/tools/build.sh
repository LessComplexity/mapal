#!/bin/bash
# Build phase (run first, may be re-run after a source fix): nsys install in the background,
# tile_bench (+ tuned CUTLASS leg, TB_PROFILE_LEG), rm_bench, the observer, the MLP programs.
set -uo pipefail
P=rtx-3090
CUDA=${CUDA:-/usr/local/cuda}
mkdir -p out/nsys out/cap out/smi
say() { echo "$@" | tee -a build.txt; }
if [ ! -f out/nsys-install.started ]; then
  touch out/nsys-install.started
  nohup bash -c 'if ! command -v nsys >/dev/null; then ver=$(nvcc --version | sed -n "s/.*release \([0-9]*\)\.\([0-9]*\).*/\1-\2/p"); apt-get install -y -qq "cuda-nsight-systems-$ver" || { apt-get update -qq; apt-get install -y -qq "cuda-nsight-systems-$ver" || apt-get install -y -qq nsight-systems; }; fi; echo nsys-install-done' > out/nsys-install.log 2>&1 &
fi
rm -f tile_bench rm_bench
INC=(-Icutlass/include -Icutlass/tools/util/include)
( nvcc -O3 -std=c++17 -arch=sm_86 "${INC[@]}" tile_bench.cu -lcublas -lcuda -o tile_bench > out/build-tile.log 2>&1 ) &
B1=$!
( nvcc -O3 -std=c++17 -arch=sm_86 "${INC[@]}" rm_bench.cu -lcublasLt -lcublas -o rm_bench > out/build-rm.log 2>&1 ) &
B2=$!
c++ -std=c++17 -O2 -Wall -Wextra -fPIC -shared -I "$CUDA/include" -I "$CUDA/extras/CUPTI/include" observer/capture.cpp \
    -L "$CUDA/extras/CUPTI/lib64" -L "$CUDA/lib64" -Wl,-rpath,"$CUDA/extras/CUPTI/lib64" -Wl,-rpath,"$CUDA/lib64" -lcupti -pthread -o capture.so > out/build-observer.log 2>&1
[ -f capture.so ] || say "OBSERVER BUILD FAILED $(tail -3 out/build-observer.log)"
link() { cc "$1" mapal_gpu.c libmapal_rt.a -I"$CUDA/include" -L"$CUDA/lib64/stubs" -lcuda -lpthread -ldl -lm -o "$2" 2>&1 | tail -3; }
for o in $P/pt_rm_*.host.o; do b=$(basename "$o" .host.o); link "$o" "prog_$b"; done
wait $B1 $B2
rc=0
[ -x tile_bench ] || { say "HARNESS BUILD FAILED"; grep -m5 error out/build-tile.log | tee -a build.txt; rc=1; }
[ -x rm_bench ] || { say "RM HARNESS BUILD FAILED"; grep -m5 error out/build-rm.log | tee -a build.txt; rc=1; }
say "builds done $(date -u +%T) rc=$rc"
exit $rc

