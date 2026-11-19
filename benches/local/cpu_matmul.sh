#!/bin/bash
# The README matmul rows, run LOCALLY (x86-64 Linux, clang): Mapal f32 vs NumPy (OpenBLAS),
# threaded whole box at N = 1024 / 2048 / 4096, plus single-threaded at 1024. Same sources and
# flags as benches/matmul/matmul_ab.sh (`emit --rewrite [--contract]`, clang -O2 -march=native),
# plus a value gate (c[0], c[last] exact: integer inputs, exact sums) and interleaved cycles.
# Statistic: min for 1t, median for par/mt (both printed). NumPy: matmul_ab's iters=1 per process.
#
# usage: matmul_local.sh <outdir>   env: CYCLES (15) PAR (nproc) P1CPU (4) PYTHON SIZES ("1024 2048 4096")
#                                         ST_SIZES ("1024") CPP (0: 1 adds the naive C++ mt leg)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="${1:?usage: matmul_local.sh <outdir>}"
CYCLES="${CYCLES:-15}"
PAR="${PAR:-$(nproc)}"
P1CPU="${P1CPU:-4}"
PYTHON="${PYTHON:-python3}"
SIZES="${SIZES:-1024 2048 4096}"
ST_SIZES="${ST_SIZES:-1024}"
CPP="${CPP:-0}"
EMIT="$ROOT/target/release/examples/emit"
RT="$ROOT/target/release/libmapal_rt.a"
mkdir -p "$OUTDIR/bin" "$OUTDIR/ll" "$OUTDIR/log"
ulimit -s unlimited 2>/dev/null || ulimit -s hard 2>/dev/null || true
B="$OUTDIR/bin"; L="$OUTDIR/log"; P1="taskset -c $P1CPU"
NP1="env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PYTHON"
NPT="env OPENBLAS_NUM_THREADS=$PAR OMP_NUM_THREADS=$PAR $PYTHON"

{
echo "== matmul local =="
echo "machine: $(lscpu | sed -n 's/^Model name: *//p') · $(nproc) logical"
echo "clang:   $(clang --version | head -1)"
echo "numpy:   $("$PYTHON" -c 'import numpy,threadpoolctl as t;i=[d for d in t.threadpool_info() if d["user_api"]=="blas"];print(numpy.__version__,[(d["internal_api"],d["version"],d["architecture"]) for d in i])' 2>/dev/null || "$PYTHON" -c 'import numpy;print(numpy.__version__)')"
echo "PAR=$PAR P1CPU=$P1CPU CYCLES=$CYCLES SIZES=$SIZES ST_SIZES=$ST_SIZES"
echo "git: $(git -C "$ROOT" rev-parse --short HEAD) $(git -C "$ROOT" status --short -- crates | wc -l) dirty crate files"
} | tee "$OUTDIR/header.txt"

echo "-- emit + build"
for N in $SIZES; do
    src="$ROOT/benches/matmul/matmul${N}_cap_f32.mapal"
    "$EMIT" "$src" - --rewrite --no-contract > "$OUTDIR/ll/${N}_conf.ll"   # FMA off (the flagless default contracts)
    "$EMIT" "$src" - --rewrite --contract > "$OUTDIR/ll/${N}_fma.ll"
    for face in conf fma; do
        [ -s "$OUTDIR/ll/${N}_$face.ll" ] || { echo "EMPTY EMISSION ${N}_$face" >&2; exit 1; }
        clang -O2 -march=native -ffp-contract=fast "$OUTDIR/ll/${N}_$face.ll" "$RT" -o "$B/mm${N}_$face" -lpthread -ldl -lm 2>/dev/null
    done
done
[ "$CPP" = 1 ] && clang++ -std=c++17 -O3 -march=native -ffp-contract=fast "$ROOT/benches/matmul/cpp_mt.cpp" -o "$B/cpp_mt" -pthread

# Mapal prints c[0], c[last], `iter ms=<v>`; numpy_bench prints `numpy N=.. <ms> ms .. c0=<v> clast=<v>`.
mapal_vals() { "$@" | grep -v '^iter ms=' | awk '{printf "%.1f ", $1}'; }
np_vals()    { "$@" | sed -n 's/.*c0=\([-0-9.]*\) clast=\([-0-9.]*\).*/\1 \2 /p'; }
echo "== VALUE GATE ==" | tee "$OUTDIR/gate.txt"
gate=0
for N in $SIZES; do
    ref=$(mapal_vals env MAPAL_PAR=1 "$B/mm${N}_conf")
    for got in "$(mapal_vals env MAPAL_PAR=$PAR "$B/mm${N}_conf")" "$(mapal_vals env MAPAL_PAR=$PAR "$B/mm${N}_fma")" \
               "$(np_vals $NPT "$ROOT/benches/matmul/numpy_bench.py" "$N" 1)"; do
        [ "$got" = "$ref" ] || { echo "MISMATCH N=$N ref=[$ref] got=[$got]" | tee -a "$OUTDIR/gate.txt"; gate=1; }
    done
    echo "N=$N values OK [$ref]" | tee -a "$OUTDIR/gate.txt"
done
[ $gate -eq 0 ] || { echo "GATE FAILED — no timing"; exit 1; }

mapal_run() { local n=$1; shift; local v; v=$("$@" 2>/dev/null | sed -n 's/^iter ms=//p'); [ -n "$v" ] && echo "$v" >> "$L/$n"; }
np_run()    { local n=$1; shift; local v; v=$("$@" 2>/dev/null | sed -n 's/.* N=[0-9]* \([0-9.]*\) ms.*/\1/p'); [ -n "$v" ] && echo "$v" >> "$L/$n"; }
rm -f "$L"/M_*
echo "== timing: $CYCLES interleaved cycles =="
for ((c = 0; c < CYCLES; ++c)); do
    for N in $SIZES; do
        mapal_run "M_${N}_mapal-fma-par"  env MAPAL_PAR=$PAR "$B/mm${N}_fma"
        mapal_run "M_${N}_mapal-conf-par" env MAPAL_PAR=$PAR "$B/mm${N}_conf"
        np_run    "M_${N}_numpy-mt"       $NPT "$ROOT/benches/matmul/numpy_bench.py" "$N" 1
        [ "$CPP" = 1 ] && np_run "M_${N}_cpp-mt" env THREADS=$PAR "$B/cpp_mt" "$N" 1 f32
        if [[ " $ST_SIZES " == *" $N "* ]]; then
            mapal_run "M_${N}_mapal-fma-1t"  $P1 env MAPAL_PAR=1 "$B/mm${N}_fma"
            mapal_run "M_${N}_mapal-conf-1t" $P1 env MAPAL_PAR=1 "$B/mm${N}_conf"
            np_run    "M_${N}_numpy-1t"      $P1 $NP1 "$ROOT/benches/matmul/numpy_bench.py" "$N" 1
        fi
    done
    printf '.' >&2
done
echo >&2
"$PYTHON" - "$L" "$SIZES" <<'PY' | tee "$OUTDIR/summary.md"
import statistics, sys
from pathlib import Path
L = Path(sys.argv[1]); sizes = sys.argv[2].split()
def s(N, leg):
    f = L / f"M_{N}_{leg}"
    if not f.exists(): return None
    v = [float(x) for x in f.read_text().split()]
    return min(v), statistics.median(v), len(v)
print("ms; par/mt: median (min); 1t: min (median)\n")
print("| N | threads | Mapal fma | Mapal conf | NumPy (OpenBLAS) | NumPy / best Mapal | Mapal GFLOP/s | NumPy GFLOP/s |")
print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for N in sizes:
    for mode, mk, nk, i in (("whole box", "par", "numpy-mt", 1), ("1 thread", "1t", "numpy-1t", 0)):
        f, c, n = s(N, f"mapal-fma-{mk}"), s(N, f"mapal-conf-{mk}"), s(N, nk)
        if not (f and c and n): continue
        fmt = (lambda t: f"{t[1]:.3f} ({t[0]:.3f})") if i else (lambda t: f"{t[0]:.3f} ({t[1]:.3f})")
        best = min(f[i], c[i]); fl = 2 * int(N) ** 3 / 1e6
        print(f"| {N} | {mode} | {fmt(f)} | {fmt(c)} | {fmt(n)} | {n[i]/best:.2f}× | {fl/best:,.0f} | {fl/n[i]:,.0f} |")
    cm = s(N, "cpp-mt")
    if cm: print(f"| {N} | naive C++ mt | {cm[1]:.3f} ({cm[0]:.3f}) | | | | | |")
print("\n(NumPy / best Mapal > 1 = Mapal faster)")
PY
