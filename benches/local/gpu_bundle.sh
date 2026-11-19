#!/bin/bash
# Local RTX 5080 bundle: the README "GPU workloads" suite (default face, shipped build + null) + two GEMM cells.
# usage: bundle.sh <outdir> [stems...]   (stems default: the README table's 35 programs)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=$1; shift
README_SUITE="pt_decode_1024_256 pt_convmc_8_64_56_3_tf32 pt_indep_512 pt_mlp_512 pt_pj_m16_n18432_k7168 pt_pj_m1_n18432_k7168
pt_pj_m1_n2048_k7168 pt_attn_1024_64 pt_mlp2_2048_tf32 pt_mlp_2048_tf32 pt_convmc_8_64_56_3 pt_loopmm_1024_8 pt_pj_m16_n2048_k7168
pt_rt_mlp2_1024 pt_mlp2_2048 pt_indep_2048 pt_diamond_2048 pt_smlp_4096_256_1024 pt_fanout_2048 pt_mlp_2048 pt_rowsum_2048
pt_twin_2048 pt_rt_mlp2_2048 pt_softmax_rows_4096 pt_rowsum_4096 pt_epi_4096 pt_ew_4096 pt_loop_4096 pt_firew_rowsum_2048x2048_16
pt_conv_2048_5 pt_fir_4194304_16 pt_causal_1024_256 pt_prefill_1024_512_256 pt_ewconvew_2048_5 pt_gen_1024_512_256"
SUITE=${*:-$(echo $README_SUITE)}
mkdir -p "$OUT"
# The boxes' CUTLASS (b2f249a), copied so integrated_bundle.sh does not clone HEAD.
[ -d "$OUT/cutlass" ] || cp -R "${CUTLASS_SRC:-$ROOT/.claude/sprints/per-site-choice/scratch/harness/bundle/cutlass}" "$OUT/"
cd "$ROOT"
export PATH=/opt/cuda/bin:$PATH CUDA=/opt/cuda LLVM=/usr/bin CPU_TARGET=x86-64-v3
CELLS=${CELLS-"4096x4096x4096 16x2048x7168"} PROGS="" SUITE="$SUITE" SUITE_ARMS=0 ARMS="" \
    benches/gpu/integrated_bundle.sh "$OUT" rtx-5080:sm_120 > "$OUT/bundle.log" 2>&1 || { tail -20 "$OUT/bundle.log"; exit 1; }
python3 benches/gpu/preflight.py "$OUT" rtx-5080 2>&1 | tee "$OUT/preflight.log" | tail -5
