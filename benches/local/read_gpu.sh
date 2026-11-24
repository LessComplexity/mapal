#!/bin/bash
# ADR-0045 device legs of the `read` builtin on the local GPU, under the bench lock:
#   - the golden program (every element type) and a failing read through `mapal run --target=<gpu>`;
#   - a 1 GiB weight file read once and used by every trip of a bracketed loop: the bytes the runtime moves
#     (`mapal_read_trace`, `mapal_gpu_trace_total`) and the host IR's uploads per trip (`evidence.py` move_facts).
# usage: read_gpu.sh [out file]   (env: LOCK, MAPAL_READ_TARGET=rtx-5080, CARGO_TARGET_DIR)
# The measured lines go to the out file (default target/tmp/read-gpu.txt); raw output stays out of git.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=${1:-$ROOT/target/tmp/read-gpu.txt}
LOCK=${LOCK:-/tmp/mapal-bench.lock}
mkdir -p "$(dirname "$OUT")"
cd "$ROOT"
# Build first, outside the lock: the lock covers the device runs only.
cargo test -p mapal-cli --test read --no-run 2>&1 | tail -1
exec 9>"$LOCK"
flock 9
env MAPAL_GPU_LIVE=1 MAPAL_READ_GIB=1 MAPAL_READ_GIB_OUT="$OUT" MAPAL_CUDA=${MAPAL_CUDA:-/opt/cuda} \
    RUST_TEST_THREADS=1 cargo test -p mapal-cli --test read -- --nocapture live_gpu gib_read 2>&1 |
    grep -E '^test |test result|trips=|panicked|left|right'
rc=${PIPESTATUS[0]}
flock -u 9
echo "read_gpu rc=$rc -> $OUT"
cat "$OUT" 2>/dev/null
exit "$rc"
