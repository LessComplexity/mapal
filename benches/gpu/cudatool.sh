#!/bin/bash
# Run a command with the CUDA CPU tools on PATH (ptxas, cuobjdump, nvdisasm). On a CUDA box they are
# already there; on a machine without CUDA (S62 h100-pcie/tools/cudatool.sh) the command runs in an
# amd64 container over the pypi wheels (ptxas 12.8.93, cuobjdump / nvdisasm 13.4.92).
# usage: cudatool.sh <command> args...   e.g. cudatool.sh bash -c 'ptxas -arch=sm_89 -v x.ptx -o x.cubin'
# The wheels, once, into CUDA_WHEELS: pip download --platform manylinux2014_x86_64 --only-binary=:all:
#   --no-deps nvidia-cuda-nvcc-cu12==12.8.93 nvidia-cuda-cuobjdump==13.4.92 nvidia-cuda-nvdisasm==13.4.92
#   -d <dir>; unzip each into <dir>/x.  /private/tmp and /Volumes are mounted at the same path.
CUDA=${CUDA:-/usr/local/cuda}
if command -v ptxas >/dev/null 2>&1; then exec "$@"; fi
if [ -x "$CUDA/bin/ptxas" ]; then PATH=$PATH:$CUDA/bin exec "$@"; fi
W=${CUDA_WHEELS:-/private/tmp/claude-501/-Volumes-LessComplex-Personal-mapal/s64-evidence/cuda-wheels}/x/nvidia
exec docker run --rm --platform linux/amd64 -v "$W":/t:ro -v /private/tmp:/private/tmp -v /Volumes:/Volumes \
    -e PATH=/t/cuda_nvcc/bin:/t/cu13/bin:/usr/bin:/bin -w "$PWD" debian:bookworm-slim "$@"
