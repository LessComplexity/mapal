#!/bin/bash
# site-geometry A2: a bundle of only what changed (changed.py diff), checked before any rental.
# Per part, changed.py env picks the programs and arms: bundle.sh builds the differential programs (ONLY, with
# their oracles), integrated_bundle.sh the timed ones (SUITE / SUITE_C / PROGS / CELLS and their ARMS); a part
# with nothing changed is skipped (no rental). Then preflight.py checks every built part.
#
# usage: benches/gpu/changed_bundle.sh <outdir> <diff dir> <profile:sm>...
# ALL_ARMS=1 also bundles the `carried` arms; EXTRA_ARMS="<ARMS entries>" adds arms (search.py's geom arms).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=$1; DIFF=$2; shift 2
mkdir -p "$OUT"
built=()
for spec in "$@"; do
    IFS=: read -r prof sm <<< "$spec"
    eval "$(python3 "$ROOT/benches/gpu/changed.py" env "$DIFF" "$prof")"
    extra=$(tr ' ' '\n' <<< "${EXTRA_ARMS:-}" | grep "^$prof:" | tr '\n' ' ' || true)
    if [ -z "$SUITE$PROGS$CELLS$ONLY$ARMS$extra" ]; then echo "$prof: nothing changed, no rental"; continue; fi
    if [ -n "$ONLY" ]; then
        ONLY="$ONLY" NOARMS="$NOARMS" HOST_RUNGS="$HOST_RUNGS" KERNEL_RUNGS="$KERNEL_RUNGS" TF32_RUNGS="$TF32_RUNGS" \
            "$ROOT/benches/gpu/bundle.sh" "$OUT" "$prof:$sm" > "$OUT/bundle-$prof.log" 2>&1
    fi
    SUITE="$SUITE" SUITE_C="$SUITE_C" PROGS="$PROGS" CELLS="$CELLS" ARMS="$ARMS $extra" \
        "$ROOT/benches/gpu/integrated_bundle.sh" "$OUT" "$prof:$sm" > "$OUT/integrated-$prof.log" 2>&1
    echo "$prof: suite=$(wc -w <<< "$SUITE $PROGS") cells=$(wc -w <<< "$CELLS") differential=$(wc -w <<< "$ONLY") arms=$(wc -w <<< "$ARMS $extra")"
    built+=("$prof")
done
cp "$DIFF"/emission.tsv "$DIFF"/arms.tsv "$DIFF"/changed.tsv "$OUT/"
[ ${#built[@]} -eq 0 ] || python3 "$ROOT/benches/gpu/preflight.py" "$OUT" "${built[@]}"
