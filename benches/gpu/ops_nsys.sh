#!/bin/bash
# plan-cost: device ops between the timers of the small chains, per arm (nsys cuda gpu trace, one warm-cold process each),
# plus ptxas -v and SASS of each arm's PTX. Runs on the box from box_profile.sh (bundle root = cwd).
# usage: bash ops_nsys.sh <profile> <sm_NN> ; env OPS_PROGS overrides the chains.
PROF=$1; SM=$2; CUDA=${CUDA:-/usr/local/cuda}; export PATH=$PATH:$CUDA/bin
mkdir -p prof/ops
command -v nsys >/dev/null || { ver=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1-\2/p'); apt-get install -y -qq "cuda-nsight-systems-$ver" >/dev/null 2>&1; export PATH=$PATH:$(dirname "$(ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)"); }
for c in ${OPS_PROGS:-pt_indep_512 pt_twin_512 pt_mlp_512 pt_mlp2_512 pt_fanout_512 pt_diamond_512}; do
  for arm in "" .off-plan-cost+linear .on-base; do
    s=$c$arm; [ -f "$PROF/$s.host.o" ] || continue
    cc "$PROF/$s.host.o" mapal_gpu.c libmapal_rt.a -I"$CUDA/include" -L"$CUDA/lib64/stubs" -lcuda -lpthread -ldl -lm -o "ops_$s" 2>/dev/null || continue
    [ -f "$PROF/$s.ptx" ] && { ptxas -arch="$SM" -O3 -v "$PROF/$s.ptx" -o "prof/ops/$s.cubin" > "prof/ops/$s.ptxas" 2>&1; cuobjdump -sass "prof/ops/$s.cubin" > "prof/ops/$s.sass" 2>&1; rm -f "prof/ops/$s.cubin"; }
    for r in 1 2 3; do
      MAPAL_PTX="$PROF/$s.ptx" timeout 300 nsys profile -o "prof/ops/$s.$r" --force-overwrite true -t cuda "./ops_$s" > "prof/ops/$s.$r.out" 2>&1
      nsys stats -q --format csv -r cuda_gpu_trace "prof/ops/$s.$r.nsys-rep" > "prof/ops/$s.$r.gpu.csv" 2>&1
      rm -f "prof/ops/$s.$r.nsys-rep" "prof/ops/$s.$r.sqlite"
    done
  done
done
tar czf ops-$PROF.tgz prof/ops 2>/dev/null
