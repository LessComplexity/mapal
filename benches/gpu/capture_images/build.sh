#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_HOME:?set CUDA_HOME to the pinned CUDA 12.8 toolkit}"
: "${CUPTI_HOME:=$CUDA_HOME/extras/CUPTI}"
: "${CAPTURE_BUILD:?set CAPTURE_BUILD to a new output directory}"
mkdir -m 700 "$CAPTURE_BUILD"
"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror -fPIC -shared \
  -I "$CUDA_HOME/include" -I "$CUPTI_HOME/include" \
  "$(dirname "$0")/capture.cpp" -L "$CUPTI_HOME/lib64" \
  -Wl,-rpath,"$CUPTI_HOME/lib64" -lcupti -pthread -o "$CAPTURE_BUILD/capture.so"
