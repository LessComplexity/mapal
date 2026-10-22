#!/bin/bash
# plan-s49 — linear-NN A/B on THIS machine (Apple M4 Pro), Mapal vs NumPy.
#
# Legs are compute-only and self-timed: the Mapal sources bracket the forward
# pass with `time`, nn_numpy.py times the matmul chain alone. Data generation is
# outside every timed region on both sides.
#
# Every cell is value-gated before it is timed, against the Mapal conformance
# leg, and the OBSERVED maximum relative error is printed rather than "passed".
#
# Usage: benches/nn/nn_ab.sh [d ...]        # default 128 256 512 1024 2048
#        CYCLES=15 benches/nn/nn_ab.sh 1024
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TMP="$ROOT/target/tmp/nn_ab"
OUT="${OUT:-$ROOT/benches/results/cpu/scheduling/dependency-cycles}"
CYCLES="${CYCLES:-15}"
SIZES=("$@"); [ ${#SIZES[@]} -gt 0 ] || SIZES=(128 256 512 1024 2048)
EMIT="$ROOT/target/release/examples/emit"
RT="$ROOT/target/release/libmapal_rt.a"
TARGET="${TARGET:-apple-m4-sme}"
CFLAGS="-O2 -march=armv8-a+sme2"
PAR="${PAR:-14}"

mkdir -p "$TMP" "$OUT"
ulimit -s unlimited 2>/dev/null || ulimit -s hard 2>/dev/null || true

# Two examples are named `emit`; the cuda one rejects --contract. Prove the binary.
for probe in --contract "--target=$TARGET"; do
    "$EMIT" "$ROOT/benches/nn/nn2_128.mapal" - --rewrite "$probe" >/dev/null 2>&1 ||
        { echo "nn_ab: wrong emit binary — it rejects $probe" >&2; exit 2; }
done
[ -f "$RT" ] || { echo "no runtime $RT — cargo build -p mapal-rt --release" >&2; exit 2; }

echo "== linear-NN A/B =="
echo "machine: $(sysctl -n machdep.cpu.brand_string) · $(sysctl -n hw.ncpu) threads"
echo "numpy:   $(python3 -c 'import numpy;print(numpy.__version__)') / $(python3 -c "import numpy;print(numpy.__config__.show(mode='dicts')['Build Dependencies']['blas']['name'])")"
echo "target:  $TARGET · MAPAL_PAR=$PAR · CYCLES=$CYCLES · sizes: ${SIZES[*]}"
echo

build() { # <shape> <d>
    local shape=$1 d=$2 src="$ROOT/benches/nn/${shape}_${d}.mapal"
    [ -f "$src" ] || { echo "no source $src" >&2; exit 2; }
    for face in conf fma; do
        local flags=(--rewrite "--target=$TARGET")
        [ "$face" = fma ] && flags+=(--contract)
        "$EMIT" "$src" - "${flags[@]}" > "$TMP/${shape}_${d}_${face}.ll" || { echo "emit failed $shape $d $face" >&2; exit 1; }
        [ -s "$TMP/${shape}_${d}_${face}.ll" ] || { echo "EMPTY EMISSION $shape $d $face" >&2; exit 1; }
        # A program whose plan went cyclic emits NO parallel driver and would be
        # timed as a scheduling result while measuring the sequential fallback.
        grep -q 'mapal_par_task' "$TMP/${shape}_${d}_${face}.ll" ||
            { echo "NO PARALLEL DRIVER: $shape $d $face — plan is cyclic or single-path" >&2; exit 1; }
        clang $CFLAGS "$TMP/${shape}_${d}_${face}.ll" "$RT" -lpthread -o "$TMP/${shape}_${d}_${face}" 2>/dev/null ||
            { echo "link failed $shape $d $face" >&2; exit 1; }
    done
}

relerr() { # <ref-file> <got-file> -> prints max relative error, exits 1 over 1e-3
    python3 - "$1" "$2" <<'PY'
import sys
ref = [float(v) for v in open(sys.argv[1]).read().split()]
got = [float(v) for v in open(sys.argv[2]).read().split()]
if len(ref) != len(got) or not ref:
    raise SystemExit(f"FAIL: value count {len(ref)} != {len(got)}")
worst = max(abs(b - a) / max(abs(a), 1.0) for a, b in zip(ref, got))
print(f"{worst:.3e}")
raise SystemExit(0 if worst <= 1e-3 else 1)
PY
}

vals() { grep -v '^iter ms=' ; }

for d in "${SIZES[@]}"; do
    for shape in nn2 nn4 nn2x2; do
        build "$shape" "$d"
        MAPAL_PAR=1 "$TMP/${shape}_${d}_conf" | vals > "$TMP/${shape}_${d}.ref"
        for leg in "env MAPAL_PAR=$PAR $TMP/${shape}_${d}_conf" \
                   "env MAPAL_PAR=1 $TMP/${shape}_${d}_fma" \
                   "env MAPAL_PAR=$PAR $TMP/${shape}_${d}_fma" \
                   "env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python3 $ROOT/benches/nn/nn_numpy.py $shape $d 1" \
                   "python3 $ROOT/benches/nn/nn_numpy.py $shape $d 1"; do
            $leg | vals > "$TMP/.g" || { echo "LEG FAILED: $leg" >&2; exit 1; }
            e=$(relerr "$TMP/${shape}_${d}.ref" "$TMP/.g") ||
                { echo "VALUE GATE FAILED $shape d=$d rel=$e :: $leg" >&2; exit 1; }
            printf 'gate %-6s d=%-5s rel=%s  %s\n' "$shape" "$d" "$e" "${leg%% *}"
        done
    done
done
echo "-- value gate passed for every cell"
echo

sample() { # <name> <command...>
    local name=$1; shift
    local ms
    ms=$("$@" 2>/dev/null | sed -n 's/^iter ms=//p')
    [ -n "$ms" ] || { echo "NO TIMING: $*" >&2; return 1; }
    echo "$ms" >> "$OUT/$name"
}

rm -f "$OUT"/S_*
echo "-- timing: $CYCLES interleaved cycles"
for ((c = 0; c < CYCLES; ++c)); do
    for d in "${SIZES[@]}"; do
        for shape in nn2 nn4 nn2x2; do
            sample "S_${shape}_${d}_mapal-conf-1t"  env MAPAL_PAR=1    "$TMP/${shape}_${d}_conf"
            sample "S_${shape}_${d}_mapal-conf-par" env MAPAL_PAR=$PAR "$TMP/${shape}_${d}_conf"
            sample "S_${shape}_${d}_mapal-fma-1t"   env MAPAL_PAR=1    "$TMP/${shape}_${d}_fma"
            sample "S_${shape}_${d}_mapal-fma-par"  env MAPAL_PAR=$PAR "$TMP/${shape}_${d}_fma"
            sample "S_${shape}_${d}_numpy-1t"       env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
                                                    python3 "$ROOT/benches/nn/nn_numpy.py" "$shape" "$d" 1
            sample "S_${shape}_${d}_numpy-mt"       python3 "$ROOT/benches/nn/nn_numpy.py" "$shape" "$d" 1
        done
    done
    printf '.' >&2
done
echo >&2
echo "samples in $OUT — one file per leg"
python3 "$ROOT/benches/nn/nn_report.py" "$OUT"
