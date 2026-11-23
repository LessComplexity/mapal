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
# A rented container sees the host's CPUs and memory in nproc / meminfo; its share is the cgroup's CPU quota and memory
# limit (v2 cpu.max / memory.max, v1 cfs_quota / limit_in_bytes): a job count from the host's would oversubscribe the
# quota and run nvcc out of the container's memory.
q=$(awk '$1 != "max" { print int(($1 + $2 - 1) / $2) }' /sys/fs/cgroup/cpu.max 2>/dev/null ||
    awk -v p="$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null)" '$1 > 0 && p > 0 { print int(($1 + p - 1) / p) }' \
        /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null)
[ -n "$q" ] && [ "$q" -gt 0 ] && [ "$q" -lt "$cores" ] && cores=$q
lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)
use=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
case $lim in '' | max | *[!0-9]*) ;; *)
    lg=$(((lim - use) / 1073741824)); [ "$lg" -lt "$mem" ] && mem=$lg ;; esac
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
    # decode path). Its setup.py swaps a toolkit other than 12.8 for NVIDIA's nvcc 12.6 front end + ptxas 12.8 but
    # compiles against the image's headers, and CUDA 13 moved the CCCL headers to include/cccl (`fatal error:
    # cuda/std/utility`, cross-arch 19a8a44e): on CUDA 13 or newer the image's own nvcc builds it (FA3_SYSTEM_NVCC).
    t1=$(date +%s)
    nv=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
    sysnv=0
    if [ "${nv:-0}" -ge 13 ]; then
        sed -i 's/^    if bare_metal_version != Version("12.8"):/    if os.getenv("FA3_SYSTEM_NVCC") != "1" and bare_metal_version != Version("12.8"):/' \
            fa-src/hopper/setup.py && grep -q FA3_SYSTEM_NVCC fa-src/hopper/setup.py && sysnv=1
    fi
    (cd fa-src/hopper && FA3_SYSTEM_NVCC=$sysnv FLASH_ATTENTION_DISABLE_BACKWARD=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE \
        FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
        FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
        FLASH_ATTENTION_DISABLE_LOCAL=TRUE FLASH_ATTENTION_DISABLE_CLUSTER=TRUE FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE \
        FLASH_ATTENTION_DISABLE_SM80=TRUE MAX_JOBS=$jobs NVCC_THREADS=2 timeout "${CAP}m" "$PY" setup.py install) > fa3-build.log 2>&1
    echo "comp build fa3 rc=$? s=$(($(date +%s) - t1)) jobs=$jobs arch=sm_90a tag=$TAG system_nvcc=$sysnv"
fi
