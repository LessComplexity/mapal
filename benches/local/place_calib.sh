#!/usr/bin/env bash
# ADR-0050 cost model: build and run the placement calibration on this box's GPU (one run: every probe, `prim`
# included). Usage: benches/local/place_calib.sh <out-dir> [probes...]  (no probes: all). The optional primitives'
# modules are PTX for the least target that has each (fp8 compute_89, TMA compute_90, wgmma compute_90a); the binary
# loads them through the driver's JIT. A JIT that accepts a module does not make it native (the H100's JIT accepted
# the fp8 mma.sync module and ran it as F2FP unpacks + f16 HMMA: the probe counted 17 175 flop/clk/SM, above the
# part's fp8 peak), so a module runs only where this box's own SASS of it holds the instruction it probes — an MMA
# reading E4M3 (fp8), UTMALDG (the tensor copy engine), HGMMA (the warpgroup product) — else it is moved aside and
# reported `prim module=... avail=0 sass=...` (no row measured).
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
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
checks=()
# native <module> <SASS regex>: assemble the module for this box's SM (its arch-specific form first) and keep it
# only where the SASS holds the probed instruction.
native() {
  local ptx=$1 want=$2 sass=""
  for a in "sm_${cc}a" "sm_${cc}"; do
    if ptxas -arch="$a" "$ptx" -o "$ptx.$a.cubin" 2>/dev/null; then
      sass=$(cuobjdump -sass "$ptx.$a.cubin")
      break
    fi
  done
  if [[ -n $sass ]] && grep -qE "$want" <<<"$sass"; then
    checks+=("prim module=$ptx sass=native")
  else
    local got
    got=$(grep -oE '\b[A-Z0-9]*MMA[A-Z0-9._]*|F2FP[A-Z0-9._]*|UTMA[A-Z0-9._]*' <<<"$sass" | sort -u | head -4 | tr '\n' ' ' || true)
    checks+=("prim module=$ptx avail=0 sass=\"no ${want} on sm_${cc}: ${got:-ptxas refuses}\"")
    mv "$ptx" "$ptx.emulated"
  fi
}
native place_calib_fp8.ptx 'MMA[A-Z0-9._]*E4M3'
native place_calib_tma.ptx 'UTMALDG'
native place_calib_wgmma.ptx 'HGMMA'
{
  printf '%s\n' "${checks[@]}"
  ./place_calib "$@"
} | tee place_calib.txt
