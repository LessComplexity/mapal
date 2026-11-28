#!/bin/bash
# The README "same shapes" table, run LOCALLY on an x86-64 Linux box with clang:
# benches/shapes/i9_ladder.sh's ladder section (no move-panel sweep), emit + link + run here.
# Legs per shape: mapal conf (FMA off: --no-contract; since ADR-0040 the flagless default contracts) / fma (--contract), 1t pinned and par whole box;
# C++ 1t pinned / mt whole box, each built strict (-ffp-contract=fast) and with -ffast-math — the baselines are
# written as careful C++ (taps outside outputs, blocked transpose, a thread pool started before the timer);
# NumPy 1t pinned. Value gate before any timing.
# Statistic: median of the interleaved runs for every leg; each comparison gets a 95 % bootstrap interval and a
# Mann–Whitney p-value (benches/local/shapes_summary.py, stats.py) — a verdict only when the interval excludes 1.
#
# usage: shapes_local.sh <outdir>      env: CYCLES (25) PAR (64) P1CPU (4) PYTHON TARGET (native)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="${1:?usage: shapes_local.sh <outdir>}"
CYCLES="${CYCLES:-25}"
PAR="${PAR:-$(nproc)}"
P1CPU="${P1CPU:-4}"
PYTHON="${PYTHON:-python3}"
TARGET="${TARGET:-native}"
EMIT="${EMIT:-$ROOT/target/release/examples/emit}"
RT="$ROOT/target/release/libmapal_rt.a"
CFLAGS="-O3 -march=native -ffp-contract=fast"
mkdir -p "$OUTDIR/bin" "$OUTDIR/ll" "$OUTDIR/log"
ulimit -s unlimited 2>/dev/null || ulimit -s hard 2>/dev/null || true

{
echo "== shapes local =="
echo "machine: $(lscpu | sed -n 's/^Model name: *//p') · $(nproc) logical"
echo "clang:   $(clang --version | head -1)"
echo "numpy:   $("$PYTHON" -c 'import numpy,threadpoolctl as t;i=[d for d in t.threadpool_info() if d["user_api"]=="blas"];print(numpy.__version__,[(d["internal_api"],d["version"],d["architecture"]) for d in i])' 2>/dev/null || "$PYTHON" -c 'import numpy;print(numpy.__version__)')"
echo "target=$TARGET PAR=$PAR P1CPU=$P1CPU CYCLES=$CYCLES"
echo "git: $(git -C "$ROOT" rev-parse --short HEAD) $(git -C "$ROOT" status --short -- crates | wc -l) dirty crate files"
} | tee "$OUTDIR/header.txt"

for probe in --contract "--target=$TARGET"; do
    "$EMIT" "$ROOT/benches/shapes/saxpy_1048576.mapal" - --rewrite "$probe" >/dev/null 2>&1 ||
        { echo "wrong emit binary — it rejects $probe" >&2; exit 2; }
done

echo "-- emit + build"
for spec in fir:fir_1048576 conv2d:conv2d_1024 saxpy:saxpy_1048576 \
            reduce:reduce_1048576 transpose:transpose_1024 gather:gather_1048576; do
    name=${spec%%:*}; src="$ROOT/benches/shapes/${spec#*:}.mapal"
    for face in conf fma; do
        fl=(--rewrite "--target=$TARGET"); [ "$face" = fma ] && fl+=(--contract); [ "$face" = conf ] && fl+=(--no-contract)
        "$EMIT" "$src" - "${fl[@]}" > "$OUTDIR/ll/${name}_$face.ll"
        [ -s "$OUTDIR/ll/${name}_$face.ll" ] || { echo "EMPTY EMISSION: ${name}_$face" >&2; exit 1; }
        clang $CFLAGS "$OUTDIR/ll/${name}_$face.ll" "$RT" -lpthread -ldl -lm -o "$OUTDIR/bin/${name}_$face" 2>/dev/null
    done
done
for base in ladder2 shapes; do
    clang++ -std=c++20 -O3 -march=native -ffp-contract=fast "$ROOT/benches/shapes/${base}_baseline.cpp" -o "$OUTDIR/bin/${base}_cpp" -pthread
    clang++ -std=c++20 -O3 -march=native -ffast-math "$ROOT/benches/shapes/${base}_baseline.cpp" -o "$OUTDIR/bin/${base}_cpp_fast" -pthread
done

P1="taskset -c $P1CPU"; PA=""
NP="env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PYTHON"
strip_ms() { grep -v '^iter ms=' ; }
cfg() {
    case $1 in
        fir)       sz=1048576; cpp=shapes_cpp;  py=shapes_numpy.py;  pa=(--1t) ;;
        conv2d)    sz=1024;    cpp=shapes_cpp;  py=shapes_numpy.py;  pa=(--1t) ;;
        transpose) sz=1024;    cpp=ladder2_cpp; py=ladder2_numpy.py; pa=() ;;
        *)         sz=1048576; cpp=ladder2_cpp; py=ladder2_numpy.py; pa=() ;;
    esac
}
B="$OUTDIR/bin"; L="$OUTDIR/log"

echo "== VALUE GATE — nothing is timed until every leg agrees ==" | tee "$OUTDIR/gate.txt"
gate=0
for shape in fir conv2d saxpy reduce transpose gather; do
    cfg $shape
    $P1 env MAPAL_PAR=1 "$B/${shape}_conf" | strip_ms > "$L/$shape.ref"
    for leg in "env MAPAL_PAR=$PAR $B/${shape}_conf" \
               "$P1 env THREADS=1 $B/$cpp $shape 1t 1 $sz" \
               "env THREADS=$PAR $B/$cpp $shape mt 1 $sz" \
               "$P1 env THREADS=1 $B/${cpp}_fast $shape 1t 1 $sz" \
               "env THREADS=$PAR $B/${cpp}_fast $shape mt 1 $sz" \
               "$P1 $NP $ROOT/benches/shapes/$py $shape ${pa[*]:-} 1 $sz"; do
        $leg | strip_ms > "$L/.g" || true
        if ! cmp -s "$L/$shape.ref" "$L/.g"; then
            # Bytes first (the i9 rule); a reassociated reduction may differ in the last
            # bits, so report the relative error instead of passing silently.
            e=$("$PYTHON" -c '
import sys
a=[float(x) for x in open(sys.argv[1]).read().split()]; b=[float(x) for x in open(sys.argv[2]).read().split()]
print("count" if len(a)!=len(b) else max(abs(y-x)/max(abs(x),1.0) for x,y in zip(a,b)))' "$L/$shape.ref" "$L/.g")
            echo "  $shape not byte-equal rel=$e :: $leg" | tee -a "$OUTDIR/gate.txt"
            "$PYTHON" -c 'import sys; e=sys.argv[1]; sys.exit(0 if e!="count" and float(e)<=1e-4 else 1)' "$e" || gate=1
        fi
    done
    $P1 env MAPAL_PAR=1 "$B/${shape}_fma" | strip_ms > "$L/.g"
    "$PYTHON" -c '
import sys
a=[float(x) for x in open(sys.argv[1]).read().split()]
b=[float(x) for x in open(sys.argv[2]).read().split()]
assert len(a)==len(b), "count"
for x,y in zip(a,b): assert abs(y-x)/max(abs(x),1.0) <= 1e-4, (x,y)
' "$L/$shape.ref" "$L/.g" || { echo "FMA DRIFT $shape" | tee -a "$OUTDIR/gate.txt"; gate=1; }
    printf '%-10s values OK ref=[%s]\n' "$shape" "$(tr '\n' ' ' < "$L/$shape.ref")" | tee -a "$OUTDIR/gate.txt"
done
[ $gate -eq 0 ] || { echo "GATE FAILED — no timing"; exit 1; }

run() {
    local name="$1"; shift
    local ms; ms=$("$@" 2>/dev/null | sed -n 's/^iter ms=//p')
    [ -n "$ms" ] || { echo "NO TIMING: $*" >&2; return 1; }
    echo "$ms" >> "$L/$name"
}
rm -f "$L"/L_*
echo "== timing: $CYCLES interleaved cycles =="
for w in 1 2 3; do env MAPAL_PAR=$PAR "$B/transpose_conf" >/dev/null 2>&1; done
for ((c = 0; c < CYCLES; ++c)); do
    for shape in fir conv2d saxpy reduce transpose gather; do
        cfg $shape
        run "L_${shape}_mapal-conf-1t"  $P1 env MAPAL_PAR=1    "$B/${shape}_conf"
        run "L_${shape}_mapal-fma-1t"   $P1 env MAPAL_PAR=1    "$B/${shape}_fma"
        run "L_${shape}_mapal-conf-par"     env MAPAL_PAR=$PAR "$B/${shape}_conf"
        run "L_${shape}_mapal-fma-par"      env MAPAL_PAR=$PAR "$B/${shape}_fma"
        run "L_${shape}_cpp-1t"         $P1 env THREADS=1      "$B/$cpp" "$shape" 1t 1 "$sz"
        run "L_${shape}_cpp-mt"             env THREADS=$PAR   "$B/$cpp" "$shape" mt 1 "$sz"
        run "L_${shape}_cpp-fast-1t"    $P1 env THREADS=1      "$B/${cpp}_fast" "$shape" 1t 1 "$sz"
        run "L_${shape}_cpp-fast-mt"        env THREADS=$PAR   "$B/${cpp}_fast" "$shape" mt 1 "$sz"
        run "L_${shape}_numpy-1t"       $P1 $NP "$ROOT/benches/shapes/$py" "$shape" ${pa[@]+"${pa[@]}"} 1 "$sz"
    done
    printf '.' >&2
done
echo >&2
"$PYTHON" "$ROOT/benches/local/shapes_summary.py" "$L" "$OUTDIR/shapes.json" | tee "$OUTDIR/summary.md"
