#!/bin/bash
# per-site-choice scale ladders: the bundle of every ladder step (gen_tiles.py --ladder: tiles/ladder.tsv) for one or
# more parts. GEMM cells go to box_run.sh's splitk section (cuBLAS / CUTLASS legs), every other step to its probe
# section (AUTO vs RULES from one binary, chain_bench and torch legs). A step whose exact CPU oracle would take too
# long (ladder.tsv gate `xref`) carries its exact face built for the device instead (integrated_bundle.sh XREF).
# Per part, scale_report.py mem drops a step whose predicted device bytes pass 90 % of the part's memory and records
# why (`<part>/ladder-mem.tsv`); then preflight.
# usage: benches/gpu/ladder_bundle.sh <bundle dir> <profile:sm>...
# EMIT_ONLY=1: emission alone (host + device IR per step and part: does it compile, its site tables, its device
# bytes) -> `<part>/ladder-mem.tsv`, no clang / llc / oracles.
# On the box: LADDER=1 TORCH_MIN=240 benches/gpu/vast_part.sh <part> <sm> <bundle> <results> 30 <GPU_NAME>..
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=$1; shift
mkdir -p "$OUT/tiles"
python3 "$ROOT/benches/gpu/gen_tiles.py" "$OUT/tiles" --ladder > /dev/null
L=$OUT/tiles/ladder.tsv
col() { awk -F'\t' -v g="$1" 'NR > 1 && (g == "" || $6 == g || (g == "prog" && $6 != "cell")) { print $1 }' "$L" | awk '!s[$0]++' | tr '\n' ' '; }
if [ "${EMIT_ONLY:-0}" = 1 ]; then
    EMIT="${EMIT:-$ROOT/target/release/examples/emit}"
    for spec in "$@"; do
        IFS=: read -r prof sm <<< "$spec"; d="$OUT/$prof"; mkdir -p "$d"
        for s in $(col ""); do
            "$EMIT" "$OUT/tiles/$s.mapal" - --rewrite "--target=$prof" --host=x86-64-v3 "--device-out=$d/$s.dev.ll" \
                > "$d/$s.host.ll" 2> "$d/$s.emit.err" || rm -f "$d/$s.host.ll"
        done
        python3 "$ROOT/benches/gpu/scale_report.py" mem "$d" "$prof"
    done
    exit 0
fi
cells=$(col cell); cells=${cells//mm_/}
LADDER=1 CELLS="$cells" PROGS="" SUITE="$(col prog)" XREF="$(col xref)" SUITE_ARMS=0 \
    SUITE_CPU="$(col prog | tr ' ' '\n' | grep -E '^pt_p[cm]_' | tr '\n' ' ')" ARMS="" \
    "$ROOT/benches/gpu/integrated_bundle.sh" "$OUT" "$@" > "$OUT/bundle.log" 2>&1 || { tail -5 "$OUT/bundle.log"; exit 1; }
for spec in "$@"; do python3 "$ROOT/benches/gpu/scale_report.py" mem "$OUT/${spec%%:*}" "${spec%%:*}" --drop; done
# The CPU+GPU programs' competitor is their own CPU-only build (`cpu` arm): no library leg.
NO_LEG='pt_pc_* pt_pm_*' python3 "$ROOT/benches/gpu/preflight.py" "$OUT" "${@%%:*}"
