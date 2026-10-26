#!/bin/bash
# site-geometry A2: a bundle of only what changed (changed.py diff), checked before any rental.
# Per part, changed.py env picks the programs and arms: bundle.sh builds the differential programs (ONLY, with
# their oracles), integrated_bundle.sh the timed ones (SUITE / SUITE_C / PROGS / CELLS and their ARMS); a part
# with nothing changed is skipped (no rental). Then preflight.py checks every built part.
#
# usage: benches/gpu/changed_bundle.sh <outdir> <diff dir> <profile:sm>...
# ALL_ARMS=1 also bundles the `carried` arms; EXTRA_ARMS="<ARMS entries>" adds arms (search.py's geom arms).
# Arms come only from arms.tsv (SUITE_ARMS=0), plus `cpu` where the CPU-only build changed — measured on the first
# part listed alone (CPU_BOX= picks another). A default face no K bounds (UNCHECKED) stops the bundle unless
# UNCHECKED_OK=1. The emitter and both runtimes are rebuilt first (cargo: a no-op when fresh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=$1; DIFF=$2; shift 2
mkdir -p "$OUT"
(cd "$ROOT" && cargo build -q --release --example emit --example run -p mapal-backend-llvm &&
    cargo build -q --release -p mapal-rt && cargo build -q --release -p mapal-rt --target x86_64-unknown-linux-gnu)
CPU_BOX=${CPU_BOX:-${1%%:*}}
built=()
for spec in "$@"; do
    IFS=: read -r prof sm <<< "$spec"
    eval "$(python3 "$ROOT/benches/gpu/changed.py" env "$DIFF" "$prof" --cpu-box "$CPU_BOX")"
    if [ -n "$UNCHECKED" ] && [ "${UNCHECKED_OK:-0}" != 1 ]; then
        echo "$prof: default face changed, unchecked (no K): $UNCHECKED — state its K in gen_tiles.py or UNCHECKED_OK=1"
        exit 1
    fi
    extra=$(tr ' ' '\n' <<< "${EXTRA_ARMS:-}" | grep "^$prof:" | tr '\n' ' ' || true)
    if [ -z "$SUITE$PROGS$CELLS$ONLY$ARMS$extra" ]; then echo "$prof: nothing changed, no rental"; continue; fi
    if [ -n "$ONLY" ]; then
        ONLY="$ONLY" NOARMS="$NOARMS" DEFAULT="$DEFAULT" HOST_RUNGS="$HOST_RUNGS" KERNEL_RUNGS="$KERNEL_RUNGS" TF32_RUNGS="$TF32_RUNGS" \
            "$ROOT/benches/gpu/bundle.sh" "$OUT" "$prof:$sm" > "$OUT/bundle-$prof.log" 2>&1
    fi
    SUITE="$SUITE" SUITE_C="$SUITE_C" SUITE_CPU="$SUITE_CPU" SUITE_ARMS=0 PROGS="$PROGS" CELLS="$CELLS" ARMS="$ARMS $extra" \
        "$ROOT/benches/gpu/integrated_bundle.sh" "$OUT" "$prof:$sm" > "$OUT/integrated-$prof.log" 2>&1
    echo "$prof: suite=$(wc -w <<< "$SUITE $PROGS") cells=$(wc -w <<< "$CELLS") differential=$(wc -w <<< "$ONLY") default=$(wc -w <<< "$DEFAULT") arms=$(wc -w <<< "$ARMS $extra")"
    built+=("$prof")
done
cp "$DIFF"/emission.tsv "$DIFF"/arms.tsv "$DIFF"/changed.tsv "$OUT/"
[ ${#built[@]} -eq 0 ] || python3 "$ROOT/benches/gpu/preflight.py" "$OUT" "${built[@]}"
