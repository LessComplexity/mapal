#!/bin/bash
# Competitors built from source on a rented box (box_run.sh, after every timed Mapal and torch section, so the build
# never loads the CPUs beside a timed run): FlashAttention-2 (the flash-attn package: no published wheel matches torch
# 2.11 + CUDA 13) on every part, and FlashAttention-3 (the same tag's hopper/ build) on sm_90a. Each build is capped
# by COMP_BUILD_MIN minutes (default 25); a capped or failed build leaves its leg reported unavailable by
# comp_bench.py. Prints `comp build <leg> rc= s= jobs= arch=` lines.
# Local (benches/local): the torch venv is prepared once instead (benches/results/evidence/local-rtx-5080/competitors.md).
# usage: comp_build.sh <sm_NN>   (FA_TAG: the flash-attention tag, default v2.8.3.post2; PY: python3)
set -uo pipefail
SM=$1
TAG=${FA_TAG:-v2.8.3.post2}
CAP=${COMP_BUILD_MIN:-25}
PY=${PY:-python3}
cores=$(nproc)
mem=$(awk '/MemAvailable/ { print int($2 / 1048576) }' /proc/meminfo)
jobs=$((cores / 2))
[ "$jobs" -gt $((mem / 6)) ] && jobs=$((mem / 6)) # an nvcc job of these kernels peaks near 5 GB
[ "$jobs" -lt 1 ] && jobs=1
# flash-attn's setup.py knows gencode families 80 / 90 / 100 / 120; sm_86 / sm_89 run the sm_80 cubins.
case $SM in sm_8*) arch=80 ;; sm_90*) arch=90 ;; sm_100*) arch=100 ;; sm_12*) arch=120 ;; *) arch= ;; esac
"$PY" -m pip install -q setuptools wheel packaging psutil ninja einops
t0=$(date +%s)
if [ ! -d fa-src ]; then
    timeout 10m git clone -q --depth 1 --branch "$TAG" https://github.com/Dao-AILab/flash-attention fa-src &&
        (cd fa-src && timeout 10m git submodule update -q --init --depth 1 csrc/cutlass)
fi
(cd fa-src && FLASH_ATTN_CUDA_ARCHS=$arch FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=$jobs NVCC_THREADS=2 \
    timeout "${CAP}m" "$PY" -m pip install --no-build-isolation .) > fa2-build.log 2>&1
echo "comp build fa2 rc=$? s=$(($(date +%s) - t0)) jobs=$jobs arch=$arch tag=$TAG"
if [ "${SM%a}" = sm_90 ]; then
    # FA3: forward kernels at head dim 128 in bf16 (the LLM layer's), with split-KV, append-KV and packed GQA (the
    # decode path); its setup.py fetches NVIDIA's nvcc 12.8 for the sm_90a kernels when the image's CUDA differs.
    t1=$(date +%s)
    (cd fa-src/hopper && FLASH_ATTENTION_DISABLE_BACKWARD=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE \
        FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
        FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
        FLASH_ATTENTION_DISABLE_LOCAL=TRUE FLASH_ATTENTION_DISABLE_CLUSTER=TRUE FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE \
        FLASH_ATTENTION_DISABLE_SM80=TRUE MAX_JOBS=$jobs timeout "${CAP}m" "$PY" setup.py install) > fa3-build.log 2>&1
    echo "comp build fa3 rc=$? s=$(($(date +%s) - t1)) jobs=$jobs arch=sm_90a tag=$TAG"
fi
