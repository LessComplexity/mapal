#!/bin/bash
# Run a CUDA CPU tool (ptxas 12.8.93, cuobjdump/nvdisasm 13.4.92) in an amd64 container.
# usage: cudatool.sh <tool> args...   (paths under the scratchpad are mounted at the same path)
S=/private/tmp/claude-501/-Volumes-LessComplex-Personal-mapal/d48de64b-c24f-4f25-9674-928a7ce97ccb/scratchpad
T=$S/disasm-h100-pcie/wheels/tools/nvidia
t=$1; shift
case $t in ptxas) b=/t/cuda_nvcc/bin/ptxas ;; *) b=/t/cu13/bin/$t ;; esac
exec docker run --rm --platform linux/amd64 -v "$T":/t:ro -v "$S":"$S" -w "$PWD" debian:bookworm-slim "$b" "$@"
