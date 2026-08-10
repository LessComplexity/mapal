#!/bin/bash
# Derive `raptorlake`'s three `DispatchCost` fields on the arch box.
#
# WHY THIS IS A SCRIPT AND NOT A ONE-OFF. Measurement rule 26: *a constant
# derived from the runtime must be re-derived when the runtime changes.*
# `dispatch_ns` was correct at 30 us when `handoff_floor` measured it and wrong
# four changes later, and using the stale value cost three cells 5-20%. So the
# recipe has to be runnable, not remembered.
#
# THE THREE FIELDS, AND WHERE EACH COMES FROM (docs/performance/s50e §2):
#
#   flops_per_ns  the SME/AVX rung's ONE-LANE rate, from the real kernel:
#                 nn4_1024 and nn4_2048 at MAPAL_PAR=1. Already taken on this
#                 box (144.2 / 143.4 GF/s) -- recomputed here as a control.
#   bytes_per_ns  the streaming rate one lane reaches: saxpy_1048576 at
#                 MAPAL_PAR=1 moves 3*n*4 bytes.
#   dispatch_ns   NOT a microbenchmark. `handoff_floor` measures a PARKED wave,
#                 which over-priced a real dispatch by 3.6x on the M4 and
#                 regressed three cells. It is measured as a DIFFERENCE on a
#                 program the runtime actually ran: nn4_128 with its dispatches
#                 against nn4_128 with all of them inlined, over the number that
#                 flipped.
#
#                 The inline arm needs an emitter that answers "inline" for every
#                 site, which is what the `raptorlake-inline` MEASUREMENT SCRATCH
#                 profile exists for. It is not a machine and must not outlive
#                 this measurement.
#
# Usage: benches/nn/raptorlake_constants.sh <host> [cycles]
set -euo pipefail

HOST="${1:?usage: raptorlake_constants.sh <host> [cycles]}"
CYCLES="${2:-21}"
# SSH_OPTS lets this run against a box that is not on port 22 — e.g. a rented
# stand-in when the arch box is unreachable. The recipe is the same either way;
# only the transport differs.
SSH_OPTS="${SSH_OPTS:-}"
SCP_OPTS="${SCP_OPTS:-}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="$ROOT/target/tmp/rl_const"
REMOTE="${REMOTE:-mapal-s51-const}"
EMIT="$ROOT/target/release/examples/emit"
RT="$ROOT/target/x86_64-unknown-linux-gnu/release/libmapal_rt.a"
CFLAGS="-O3 -target x86_64-unknown-linux-gnu -march=raptorlake -ffp-contract=fast"
SAXPY_N=1048576

cargo build -q --release -p mapal-backend-llvm --example emit --manifest-path "$ROOT/Cargo.toml"
cargo build -q -p mapal-rt --release --target x86_64-unknown-linux-gnu --manifest-path "$ROOT/Cargo.toml"
[ -f "$RT" ] || { echo "no x86 runtime $RT" >&2; exit 2; }

rm -rf "$STAGE"; mkdir -p "$STAGE"

emit_one() { # <name> <src> <target>
    local name=$1 src=$2 target=$3
    "$EMIT" "$src" - --rewrite "--target=$target" --contract > "$STAGE/$name.ll"
    [ -s "$STAGE/$name.ll" ] || { echo "EMPTY EMISSION: $name" >&2; exit 1; }
    grep -q 'mapal_par_task' "$STAGE/$name.ll" ||
        { echo "NO PARALLEL DRIVER: $name — a cyclic plan would be timed as a scheduling result" >&2; exit 1; }
    clang $CFLAGS -c "$STAGE/$name.ll" -o "$STAGE/$name.o" 2>/dev/null
}

emit_one saxpy   "$ROOT/benches/shapes/saxpy_${SAXPY_N}.mapal" raptorlake
emit_one nn4_128_dispatch "$ROOT/benches/nn/nn4_128.mapal"     raptorlake
emit_one nn4_128_inline   "$ROOT/benches/nn/nn4_128.mapal"     raptorlake-inline
emit_one nn4_1024 "$ROOT/benches/nn/nn4_1024.mapal"            raptorlake
emit_one nn4_2048 "$ROOT/benches/nn/nn4_2048.mapal"            raptorlake

# HOW MANY DISPATCHES ACTUALLY FLIPPED — counted from the emission, not assumed.
# The divisor IS the measurement; assuming 8 because the M4 had 8 would make the
# constant wrong by whatever this box's program shape differs by.
FLIPPED=$(diff \
    <(grep 'call void @mapal_par_task' "$STAGE/nn4_128_dispatch.ll" | awk -F', ' '{print $2, $3}') \
    <(grep 'call void @mapal_par_task' "$STAGE/nn4_128_inline.ll"   | awk -F', ' '{print $2, $3}') \
    | grep -c '^<' || true)
[ "$FLIPPED" -gt 0 ] || { echo "NO TASK FLIPPED between the two arms — the scratch profile is not doing anything" >&2; exit 1; }
echo "-- $FLIPPED task(s) flip from dispatched to inlined between the two arms"

cp "$RT" "$STAGE/"
ssh $SSH_OPTS "$HOST" "rm -rf ~/$REMOTE && mkdir -p ~/$REMOTE"
scp -q $SCP_OPTS "$STAGE"/*.o "$STAGE"/*.a "$HOST:~/$REMOTE/"

ssh $SSH_OPTS "$HOST" "CYCLES=$CYCLES REMOTE=$REMOTE bash -s" <<'REMOTE_SCRIPT'
set -u
cd ~/"$REMOTE" || exit 2
ulimit -s unlimited 2>/dev/null || ulimit -s hard 2>/dev/null || true
mkdir -p bin
for o in *.o; do gcc -O3 "$o" libmapal_rt.a -lpthread -ldl -lm -o "bin/${o%.o}" || exit 1; done

# cpu4 is a 5500 MHz P-core, not cpu0/cpu2 where the boost lottery and
# interrupts live (the pinning benches/nn/nn_i9.sh already reports).
P1="taskset -c 4"; PA="taskset -c 0-31"
vals() { grep -v '^iter ms='; }

# VALUE GATE — the two nn4_128 arms differ only in WHERE tasks run, so their
# output must be identical. Checked before any timing is read.
$PA env MAPAL_PAR=32 bin/nn4_128_dispatch | vals > .a
$PA env MAPAL_PAR=32 bin/nn4_128_inline   | vals > .b
cmp -s .a .b || { echo "VALUE GATE FAILED: the two dispatch arms disagree"; exit 1; }
echo "-- value gate: the two arms are bit-identical"

for w in 1 2 3; do $PA env MAPAL_PAR=32 bin/nn4_128_dispatch >/dev/null 2>&1; done

rm -f s_*
sample() { local name=$1; shift; local ms
  ms=$("$@" 2>/dev/null | sed -n 's/^iter ms=//p')
  [ -n "$ms" ] || { echo "NO TIMING: $*" >&2; return 1; }
  echo "$ms" >> "s_$name"; }

for ((c = 0; c < CYCLES; ++c)); do
  sample saxpy_1t        $P1 env MAPAL_PAR=1  bin/saxpy
  sample nn4_1024_1t     $P1 env MAPAL_PAR=1  bin/nn4_1024
  sample nn4_2048_1t     $P1 env MAPAL_PAR=1  bin/nn4_2048
  sample nn128_dispatch  $PA env MAPAL_PAR=32 bin/nn4_128_dispatch
  sample nn128_inline    $PA env MAPAL_PAR=32 bin/nn4_128_inline
done
echo "== RAW =="
for f in s_*; do
  python3 -c "
import sys,statistics as st
v=[float(x) for x in open('$f').read().split()]
print(f'$f  n={len(v)}  min={min(v):.4f}  med={st.median(v):.4f}  max={max(v):.4f}')"
done
REMOTE_SCRIPT
