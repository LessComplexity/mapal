#!/bin/bash
# Mapal against multi-threaded NumPy (OpenBLAS) on the whole machine, where NumPy actually uses threads:
# matrix multiply (1024 / 2048 / 4096, f32) and small linear networks (nn2 / nn4 at d = 512 / 1024 / 2048).
# Each side runs at 32 and at 64 threads (OpenBLAS loses at 64 on this SMT machine) and gets its better one.
#
# Value gate before timing (rel ≤ 1e-3; matmul inputs are integers, exact). Then CYCLES interleaved cycles:
# every leg of every workload once per cycle, one process per sample. NumPy warms up 3 untimed passes in its
# own process (numpy_bench.py / nn_numpy.py); Mapal times its first call after its pool starts — the
# difference favours NumPy. Summary: medians, 95 % bootstrap intervals, Mann–Whitney (stats.py).
#
# usage: cpu_numpy_mt.sh <outdir>      env: CYCLES (21) PYTHON TARGET (native)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="${1:?usage: cpu_numpy_mt.sh <outdir>}"
CYCLES="${CYCLES:-21}"
PYTHON="${PYTHON:-python3}"
TARGET="${TARGET:-native}"
EMIT="${EMIT:-$ROOT/target/release/examples/emit}"
RT="$ROOT/target/release/libmapal_rt.a"
mkdir -p "$OUTDIR/bin" "$OUTDIR/log"
WORK=(mm:1024 mm:2048 mm:4096 nn2:512 nn4:512 nn2:1024 nn4:1024 nn2:2048 nn4:2048)
THREADS=(32 64)

# NumPy must call a threaded BLAS: a distribution NumPy linked to the reference BLAS (single-threaded,
# ~250 ms for a 1024² matmul) makes every result meaningless. Use the official wheel (bundled OpenBLAS),
# e.g. `uv venv np && VIRTUAL_ENV=np uv pip install numpy threadpoolctl`, then PYTHON=np/bin/python.
blas=$("$PYTHON" -c 'import numpy, threadpoolctl as t; print(" ".join(d["internal_api"] + "-" + str(d.get("version")) for d in t.threadpool_info() if d["user_api"] == "blas"))' 2>/dev/null || true)
case "$blas" in
    *openblas*|*mkl*|*blis*) ;;
    *) echo "refusing to run: NumPy's BLAS is '${blas:-unknown}' (need OpenBLAS / MKL / BLIS via threadpoolctl)" >&2; exit 2 ;;
esac
{
echo "== Mapal vs multi-threaded NumPy =="
echo "blas:    $blas"
echo "machine: $(lscpu | sed -n 's/^Model name: *//p') · $(nproc) logical"
echo "numpy:   $("$PYTHON" -c 'import numpy;print(numpy.__version__)') / $("$PYTHON" -c "import numpy;print(numpy.__config__.show(mode='dicts')['Build Dependencies']['blas']['name'])")"
echo "clang:   $(clang --version | head -1)"
echo "git:     $(git -C "$ROOT" rev-parse --short HEAD) · CYCLES=$CYCLES · threads ${THREADS[*]}"
} | tee "$OUTDIR/header.txt"

src_of() { case $1 in mm) echo "$ROOT/benches/matmul/matmul$2_cap_f32.mapal" ;; *) echo "$ROOT/benches/nn/$1_$2.mapal" ;; esac; }
np_cmd() { case $1 in mm) echo "$ROOT/benches/matmul/numpy_bench.py $2 1" ;; *) echo "$ROOT/benches/nn/nn_numpy.py $1 $2 1" ;; esac; }

echo "-- emit + build"
for w in "${WORK[@]}"; do
    s=${w%%:*}; n=${w#*:}
    "$EMIT" "$(src_of "$s" "$n")" - --rewrite "--target=$TARGET" --contract > "$OUTDIR/bin/$s$n.ll"
    clang -O2 -march=native -ffp-contract=fast "$OUTDIR/bin/$s$n.ll" "$RT" -lpthread -ldl -lm -o "$OUTDIR/bin/$s$n" 2>/dev/null
done

mapal_vals() { "$@" | grep -v '^iter ms=' | tr '\n' ' '; }
np_vals() { "$@" | sed -n -e 's/.*c0=\([-0-9.e+]*\) clast=\([-0-9.e+]*\).*/\1 \2/p' -e '/^[-0-9.e+]*$/p' | tr '\n' ' '; }
echo "== value gate =="
for w in "${WORK[@]}"; do
    s=${w%%:*}; n=${w#*:}
    a=$(mapal_vals env MAPAL_PAR=64 "$OUTDIR/bin/$s$n")
    b=$(np_vals env OPENBLAS_NUM_THREADS=64 OMP_NUM_THREADS=64 $PYTHON $(np_cmd "$s" "$n"))
    "$PYTHON" -c '
import sys
a = [float(x) for x in sys.argv[1].split()]; b = [float(x) for x in sys.argv[2].split()]
assert a and len(a) == len(b), ("count", a, b)
e = max(abs(x - y) / max(abs(y), 1.0) for x, y in zip(a, b))
print(f"{sys.argv[3]:<9} values OK  max rel {e:.2e}  [{sys.argv[1].strip()}]") if e <= 1e-3 else sys.exit(f"MISMATCH {sys.argv[3]} {a} {b}")
' "$a" "$b" "$s$n"
done | tee "$OUTDIR/gate.txt"

L="$OUTDIR/log"; rm -f "$L"/S_*
echo "== timing: $CYCLES interleaved cycles =="
for ((c = 0; c < CYCLES; ++c)); do
    for w in "${WORK[@]}"; do
        s=${w%%:*}; n=${w#*:}
        for t in "${THREADS[@]}"; do
            v=$(env MAPAL_PAR=$t "$OUTDIR/bin/$s$n" 2>/dev/null | sed -n 's/^iter ms=//p' | head -1)
            [ -n "$v" ] && echo "$v" >> "$L/S_$s${n}_mapal-$t"
            v=$(env OPENBLAS_NUM_THREADS=$t OMP_NUM_THREADS=$t $PYTHON $(np_cmd "$s" "$n") 2>/dev/null |
                sed -n -e 's/.* N=[0-9]* \([0-9.]*\) ms.*/\1/p' -e 's/^iter ms=//p' | head -1)
            [ -n "$v" ] && echo "$v" >> "$L/S_$s${n}_numpy-$t"
        done
    done
    printf '.' >&2
done
echo >&2
"$PYTHON" - "$L" "$OUTDIR/numpy-mt.json" "$ROOT/benches/local" <<'PY' | tee "$OUTDIR/summary.md"
import json, pathlib, sys
L, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sys.path.insert(0, sys.argv[3]); import stats
NAME = {"mm": "matrix multiply {n}²", "nn2": "2-layer network, width {n}", "nn4": "4-layer network, width {n}"}
rows = []
print("ms, median of interleaved runs; each side at its better of 32 / 64 threads. Ratio = NumPy ÷ Mapal (> 1 = Mapal")
print("faster), [95 % bootstrap interval], Mann–Whitney p; a verdict only when the interval excludes 1 and p < 0.05.\n")
print("| workload | Mapal 32 | Mapal 64 | NumPy 32 | NumPy 64 | NumPy ÷ Mapal |\n| --- | ---: | ---: | ---: | ---: | --- |")
for w in ["mm1024", "mm2048", "mm4096", "nn2512", "nn4512", "nn21024", "nn41024", "nn22048", "nn42048"]:
    s = {f.name.split("_", 2)[2]: [float(x) for x in f.read_text().split()] for f in L.glob(f"S_{w}_*")}
    m = min(("mapal-32", "mapal-64"), key=lambda k: stats.median(s[k]))
    p = min(("numpy-32", "numpy-64"), key=lambda k: stats.median(s[k]))
    c = stats.compare(s[p], s[m])
    kind = "mm" if w.startswith("mm") else w[:3]
    name = NAME[kind].format(n=w[len(kind):])
    print(f"| {name} | " + " | ".join(f"{stats.median(s[k]):.3f}" for k in ("mapal-32", "mapal-64", "numpy-32", "numpy-64"))
          + f" | {stats.fmt(c)} (Mapal {m[6:]}, NumPy {p[6:]} threads) |")
    rows.append({"name": name, "mapal_threads": int(m[6:]), "numpy_threads": int(p[6:]), "compare": c,
                 "median_ms": {k: stats.median(v) for k, v in s.items()}, "samples_ms": s})
out.write_text(json.dumps({"stat": "median of interleaved runs; bootstrap 95 % interval; Mann–Whitney", "workloads": rows}, indent=1))
PY
