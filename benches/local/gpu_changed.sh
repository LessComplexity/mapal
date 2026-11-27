#!/bin/bash
# Changed programs: a bundle of only the timed-suite programs and cells an A/B moves (benches/local/README.md).
# usage: gpu_changed.sh <outdir> --off <rung>[,<rung>] | --on <rung>[,<rung>] | --base <ref> | --base-emit <emit>
#   1. suite_diff.py emits every suite program (benches/local/suite.txt) and both cells' faces on both arms and
#      keeps those whose host or device text differs (<outdir>/suite.tsv);
#   2. gpu_bundle.sh builds <outdir>/bundle with only those, each with its other arm (`.off-<rung>` / `.on-<rung>` /
#      `.on-base`), from the bundle cache where a stem's text is unchanged.
# Then time it: LADDER=1 LOCAL_GPU=<outdir> TORCH_PY=.. benches/local/gpu_run.sh <outdir>/bundle — the probe section
# value-gates every arm and times them interleaved, the order reversed on even repetitions (ABBA); the cells' arms
# run in one tile_bench --splitk process, twice, the second reversed. A program whose text is equal on both arms is
# the same host object and PTX on both: null by construction, not timed. Nothing differs: no bundle, nothing to time.
# The full suite (gpu_bundle.sh + gpu_run.sh) still runs once per step: the safety net for what text equality
# cannot see (the runtime, the driver, the machine's state).
# `--arch <profile>` (a rented part) needs PART=<profile>:<sm_NN> for the bundle (gpu_bundle.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=$1; shift
mkdir -p "$OUT"
arch=rtx-5080; prev=; for a in "$@"; do [ "$prev" = --arch ] && arch=$a; prev=$a; done
PART=${PART:-rtx-5080:sm_120}
[ "${PART%%:*}" = "$arch" ] || { echo "gpu_changed: --arch $arch needs PART=$arch:<sm_NN> (PART=$PART)" >&2; exit 2; }
eval "$(python3 "$ROOT/benches/local/suite_diff.py" "$OUT" "$@")"
if [ -z "$STEMS$CELLS" ]; then
    echo "gpu_changed: no suite program or cell differs on the two arms: nothing to time"
    exit 0
fi
echo "gpu_changed: STEMS='$STEMS' CELLS='$CELLS'"
PART="$PART" STEMS="$STEMS" CELLS="$CELLS" ARMS="$ARMS" BASE_EMIT="${BASE_EMIT:-}" "$ROOT/benches/local/gpu_bundle.sh" "$OUT/bundle"
echo "gpu_changed: time with  LADDER=1 LOCAL_GPU=$OUT benches/local/gpu_run.sh $OUT/bundle"
