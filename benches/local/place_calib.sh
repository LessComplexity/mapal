#!/usr/bin/env bash
# ADR-0050 cost model: build and run the placement calibration on this box's GPU (one run: every probe, `prim`
# included). Usage: benches/local/place_calib.sh <out-dir> [probes...]  (no probes: all). The optional primitives'
# modules are PTX for the least target that has each (fp8 compute_89, TMA compute_90, wgmma compute_90a); the binary
# loads them through the driver's JIT, and a part whose JIT refuses one lacks it (`prim module=... avail=0`).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?out dir}
shift || true
mkdir -p "$out"
cd "$out"
nvcc -ptx -arch=compute_89 -DPRIM_FP8 "$here/place_calib_opt.cu" -o place_calib_fp8.ptx
nvcc -ptx -arch=compute_90 -DPRIM_TMA "$here/place_calib_opt.cu" -o place_calib_tma.ptx
nvcc -ptx -arch=compute_90a -DPRIM_WGMMA "$here/place_calib_opt.cu" -o place_calib_wgmma.ptx
nvcc -O3 -arch=native -std=c++17 "$here/place_calib.cu" -lcuda -o place_calib
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm --format=csv,noheader > box.txt
./place_calib "$@" | tee place_calib.txt
