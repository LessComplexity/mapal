#!/bin/bash
# plan-s49 — the linear-NN ladder on the remote x86-64 box (i9-14900F).
#
# Same split as benches/shapes/i9_ladder.sh, and for the same reason: the box has
# gcc but NO clang, so EMIT and .ll -> .o happen HERE and only the LINK and the RUN
# happen there. The box's machine facts come from a NAMED profile (`raptorlake`)
# because nothing about that box is detectable from this one — the cross-compile
# case named profiles exist for.
#
# Usage: benches/nn/nn_i9.sh <host> [d ...]
set -euo pipefail

HOST="${1:?usage: nn_i9.sh <host> [d ...]}"; shift
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="$ROOT/target/tmp/nn_i9"
OUT="${OUT:-$ROOT/benches/results-s49}"
REMOTE="${REMOTE:-mapal-s49}"
CYCLES="${CYCLES:-15}"
SIZES=("$@"); [ ${#SIZES[@]} -gt 0 ] || SIZES=(128 256 512 1024 2048)
EMIT="$ROOT/target/release/examples/emit"
TARGET="${TARGET:-raptorlake}"
CFLAGS="-O3 -target x86_64-unknown-linux-gnu -march=raptorlake -ffp-contract=fast"

cargo build -q --release -p mapal-backend-llvm --example emit --manifest-path "$ROOT/Cargo.toml"
cargo build -q -p mapal-rt --release --target x86_64-unknown-linux-gnu --manifest-path "$ROOT/Cargo.toml"
for probe in --contract "--target=$TARGET"; do
    "$EMIT" "$ROOT/benches/nn/nn2_128.mapal" - --rewrite "$probe" >/dev/null 2>&1 ||
        { echo "nn_i9: wrong emit binary — it rejects $probe" >&2; exit 2; }
done

rm -rf "$STAGE"; mkdir -p "$STAGE/ll" "$OUT"
for d in "${SIZES[@]}"; do
    for shape in nn2 nn4 nn2x2; do
        src="$ROOT/benches/nn/${shape}_${d}.mapal"
        for face in conf fma; do
            name="${shape}_${d}_${face}"
            flags=(--rewrite "--target=$TARGET"); [ "$face" = fma ] && flags+=(--contract)
            "$EMIT" "$src" - "${flags[@]}" > "$STAGE/ll/$name.ll"
            [ -s "$STAGE/ll/$name.ll" ] || { echo "EMPTY EMISSION: $name" >&2; exit 1; }
            # Timing the sequential fallback as if it were a scheduling result is
            # exactly the silent-pass class this tree keeps closing. Hard error.
            grep -q 'mapal_par_task' "$STAGE/ll/$name.ll" ||
                { echo "NO PARALLEL DRIVER: $name" >&2; exit 1; }
            clang $CFLAGS -c "$STAGE/ll/$name.ll" -o "$STAGE/$name.o" 2>/dev/null
        done
    done
done
echo "-- emitted + cross-compiled $(ls "$STAGE"/*.o | wc -l | tr -d ' ') objects"

cp "$ROOT/benches/nn/nn_numpy.py" "$STAGE/"
cp "$ROOT/target/x86_64-unknown-linux-gnu/release/libmapal_rt.a" "$STAGE/"

ssh "$HOST" "rm -rf ~/$REMOTE && mkdir -p ~/$REMOTE/log"
scp -q "$STAGE"/*.o "$STAGE"/*.a "$STAGE"/*.py "$HOST:~/$REMOTE/"

ssh "$HOST" "CYCLES=$CYCLES REMOTE=$REMOTE SIZES='${SIZES[*]}' bash -s" <<'REMOTE_SCRIPT'
set -u
cd ~/"$REMOTE" || exit 2
ulimit -s unlimited 2>/dev/null || ulimit -s hard 2>/dev/null || true
mkdir -p bin log; rm -f log/S_*

for o in *.o; do gcc -O3 "$o" libmapal_rt.a -lpthread -ldl -lm -o "bin/${o%.o}" || exit 1; done

# PINNING, reported not assumed: cpu4 is a 5500 MHz P-core, not cpu0/cpu2 (the
# favoured cores where the boost lottery and interrupts live).
P1="taskset -c 4"; PA="taskset -c 0-31"
NP1="env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3"
vals() { grep -v '^iter ms='; }

echo "== VALUE GATE =="
gate=0
for d in $SIZES; do
  for shape in nn2 nn4 nn2x2; do
    $P1 env MAPAL_PAR=1 "bin/${shape}_${d}_conf" | vals > "log/$shape$d.ref"
    for leg in "$PA env MAPAL_PAR=32 bin/${shape}_${d}_conf" \
               "$P1 env MAPAL_PAR=1  bin/${shape}_${d}_fma" \
               "$PA env MAPAL_PAR=32 bin/${shape}_${d}_fma" \
               "$P1 $NP1 nn_numpy.py $shape $d 1" \
               "$PA python3 nn_numpy.py $shape $d 1"; do
      $leg 2>/dev/null | vals > log/.g
      python3 -c '
import sys
a=[float(x) for x in open(sys.argv[1]).read().split()]
b=[float(x) for x in open(sys.argv[2]).read().split()]
assert len(a)==len(b) and a, "count %d %d"%(len(a),len(b))
w=max(abs(y-x)/max(abs(x),1.0) for x,y in zip(a,b))
print("  rel=%.3e"%w)
assert w<=1e-3, w
' "log/$shape$d.ref" log/.g || { echo "VALUE GATE FAILED $shape d=$d :: $leg"; gate=1; }
    done
  done
done
[ $gate -eq 0 ] || { echo "GATE FAILED — no timing"; exit 1; }

sample() { local name=$1; shift; local ms
  ms=$("$@" 2>/dev/null | sed -n 's/^iter ms=//p')
  [ -n "$ms" ] || { echo "NO TIMING: $*" >&2; return 1; }
  echo "$ms" >> "log/$name"; }

echo "== timing: $CYCLES interleaved cycles =="
for w in 1 2 3; do $PA env MAPAL_PAR=32 bin/nn2_512_fma >/dev/null 2>&1; done
for ((c = 0; c < CYCLES; ++c)); do
  for d in $SIZES; do
    for shape in nn2 nn4 nn2x2; do
      sample "S_${shape}_${d}_mapal-conf-1t"  $P1 env MAPAL_PAR=1  "bin/${shape}_${d}_conf"
      sample "S_${shape}_${d}_mapal-conf-par" $PA env MAPAL_PAR=32 "bin/${shape}_${d}_conf"
      sample "S_${shape}_${d}_mapal-fma-1t"   $P1 env MAPAL_PAR=1  "bin/${shape}_${d}_fma"
      sample "S_${shape}_${d}_mapal-fma-par"  $PA env MAPAL_PAR=32 "bin/${shape}_${d}_fma"
      sample "S_${shape}_${d}_numpy-1t"       $P1 $NP1 nn_numpy.py "$shape" "$d" 1
      sample "S_${shape}_${d}_numpy-mt"       $PA python3 nn_numpy.py "$shape" "$d" 1
    done
  done
  printf '.' >&2
done
echo >&2; rm -f log/.g; echo DONE
REMOTE_SCRIPT

rm -rf "$STAGE/log"
ssh "$HOST" "tar cf - -C ~/$REMOTE log" | tar xf - -C "$STAGE"
mkdir -p "$OUT/i9"; rm -f "$OUT"/i9/S_*; cp "$STAGE"/log/S_* "$OUT/i9/"
echo "samples in $OUT/i9"
python3 "$ROOT/benches/nn/nn_report.py" "$OUT/i9"
