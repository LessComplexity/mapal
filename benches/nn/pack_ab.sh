#!/bin/bash
# plan-s51 H1 — does the B pack pay at small d?
#
# THE HYPOTHESIS. `pack_b` is O(N²) and `panel` is O(N³), so below some N the
# pack costs more than the operand contiguity it buys — which buys nothing
# anyway while the whole working set is L2-resident, because S43 §4 measured
# L2-vs-L1 as FREE on this part (flat ~1990 GF/s, 32 KB → 8 MB).
#
# PRE-REGISTERED (plan-s51 §3), written before the first number is read:
#   confirmed iff  d=128  OFF beats ON by >=15%, distributions DISJOINT
#            and   d=2048 ON  beats OFF,         distributions DISJOINT
#   i.e. a real CROSSOVER. A monotone result in either direction REFUTES the
#   model, and no threshold is fitted to one end of a sweep.
#
# TWO GATES THIS REPO LEARNED THE HARD WAY:
#
#   * VOID ARMS ARE LOUD (S43 §8.4). `nc`=768 silently emitted the baseline's
#     bytes and read as "nc had no effect at this width", which is a different
#     claim. Here: if OFF's emission is byte-identical to ON's, the cell is
#     VOID and says so — it is not a null result.
#   * A BYTE-IDENTICAL CONTROL RUNS EVERY CELL. S39 measured -5.9%..+1.2%
#     between byte-identical binaries on this machine. `ctl` is ON's flags
#     compiled to a second binary; ON-vs-CTL is this run's own noise floor, and
#     any ON-vs-OFF effect smaller than it is not an effect.
#
# The SME rung is a contract-face realization (fmopa fuses), so only the --contract
# face is measured: without it the NEON rung runs and the question does not arise.
#
# Usage: benches/perflock.sh bash benches/nn/pack_ab.sh [d ...]
#        CYCLES=21 benches/perflock.sh bash benches/nn/pack_ab.sh 128 2048
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TMP="$ROOT/target/tmp/pack_ab"
OUT="${OUT:-$ROOT/benches/results-s51}"
CYCLES="${CYCLES:-15}"
SIZES=("$@"); [ ${#SIZES[@]} -gt 0 ] || SIZES=(128 256 512 1024 2048)
SHAPES=(${SHAPES:-nn4 nn2x2})
EMIT="$ROOT/target/release/examples/emit"
RT="$ROOT/target/release/libmapal_rt.a"
TARGET="${TARGET:-apple-m4-sme}"
CFLAGS="-O2 -march=armv8-a+sme2"
PAR="${PAR:-14}"

mkdir -p "$TMP" "$OUT"
ulimit -s unlimited 2>/dev/null || ulimit -s hard 2>/dev/null || true

# Two examples are named `emit`; the cuda one rejects --contract. Prove the binary.
for probe in --contract "--target=$TARGET" --no-pack; do
    "$EMIT" "$ROOT/benches/nn/nn2_128.mapal" - --rewrite "$probe" >/dev/null 2>&1 ||
        { echo "pack_ab: wrong emit binary — it rejects $probe" >&2; exit 2; }
done
[ -f "$RT" ] || { echo "no runtime $RT — cargo build -p mapal-rt --release" >&2; exit 2; }

echo "== plan-s51 H1: the B pack A/B, --contract face =="
echo "machine: $(sysctl -n machdep.cpu.brand_string) · $(sysctl -n hw.ncpu) threads"
echo "target:  $TARGET · MAPAL_PAR=1 and $PAR · CYCLES=$CYCLES · sizes: ${SIZES[*]}"
echo "commit:  $(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet || echo ' +dirty')"
echo

VOID=()

build() { # <shape> <d> -> emits+links on/off/ctl, gates emission difference
    local shape=$1 d=$2 src="$ROOT/benches/nn/${shape}_${d}.mapal"
    [ -f "$src" ] || { echo "no source $src" >&2; exit 2; }
    local base=(--rewrite "--target=$TARGET" --contract)
    for arm in on off ctl; do
        local flags=("${base[@]}")
        [ "$arm" = off ] && flags+=(--no-pack)
        "$EMIT" "$src" - "${flags[@]}" > "$TMP/${shape}_${d}_${arm}.ll" ||
            { echo "emit failed $shape $d $arm" >&2; exit 1; }
        [ -s "$TMP/${shape}_${d}_${arm}.ll" ] ||
            { echo "EMPTY EMISSION $shape $d $arm" >&2; exit 1; }
        # A cyclic plan emits no parallel driver and would be timed as a
        # scheduling result while measuring the sequential fallback (S49).
        grep -q 'mapal_par_task' "$TMP/${shape}_${d}_${arm}.ll" ||
            { echo "NO PARALLEL DRIVER: $shape $d $arm" >&2; exit 1; }
        clang $CFLAGS "$TMP/${shape}_${d}_${arm}.ll" "$RT" -lpthread -o "$TMP/${shape}_${d}_${arm}" 2>/dev/null ||
            { echo "link failed $shape $d $arm" >&2; exit 1; }
    done

    # GATE: ctl MUST be byte-identical to on — it is the noise-floor arm.
    cmp -s "$TMP/${shape}_${d}_on.ll" "$TMP/${shape}_${d}_ctl.ll" ||
        { echo "BUG: ctl emission differs from on — $shape d=$d" >&2; exit 1; }

    # GATE: off MUST differ from on, or the cell measures nothing.
    if cmp -s "$TMP/${shape}_${d}_on.ll" "$TMP/${shape}_${d}_off.ll"; then
        echo "  VOID  $shape d=$d — --no-pack emitted ON's bytes (this site does not pack)"
        VOID+=("${shape}_${d}")
        return 1
    fi
    # Say what actually moved, so "the pack went away" is observed, not assumed.
    local p_on p_off
    p_on=$(grep -c 'pack' "$TMP/${shape}_${d}_on.ll" || true)
    p_off=$(grep -c 'pack' "$TMP/${shape}_${d}_off.ll" || true)
    echo "  emit  $shape d=$d — pack mentions $p_on -> $p_off, .ll $(wc -l < "$TMP/${shape}_${d}_on.ll") -> $(wc -l < "$TMP/${shape}_${d}_off.ll") lines"
    return 0
}

vals() { grep -v '^iter ms=' ; }

echo "-- emission gates"
CELLS=()
for d in "${SIZES[@]}"; do
    for shape in "${SHAPES[@]}"; do
        build "$shape" "$d" && CELLS+=("${shape}_${d}")
    done
done
[ ${#CELLS[@]} -gt 0 ] || { echo "every cell VOID — nothing to measure" >&2; exit 3; }
echo

echo "-- value gate: OFF must be BIT-identical to ON (the pack is a layout, not arithmetic)"
for cell in "${CELLS[@]}"; do
    MAPAL_PAR=1 "$TMP/${cell}_on" | vals > "$TMP/.ref"
    for arm in off ctl; do
        for par in 1 "$PAR"; do
            MAPAL_PAR=$par "$TMP/${cell}_${arm}" | vals > "$TMP/.got" ||
                { echo "LEG FAILED: $cell $arm par=$par" >&2; exit 1; }
            cmp -s "$TMP/.ref" "$TMP/.got" ||
                { echo "VALUE GATE FAILED: $cell $arm par=$par is not bit-identical to on" >&2; exit 1; }
        done
    done
    printf '  ok    %s\n' "$cell"
done
echo

sample() { # <name> <command...>
    local name=$1; shift
    local ms
    ms=$("$@" 2>/dev/null | sed -n 's/^iter ms=//p')
    [ -n "$ms" ] || { echo "NO TIMING: $*" >&2; return 1; }
    echo "$ms" >> "$OUT/$name"
}

rm -f "$OUT"/P_*
echo "-- timing: $CYCLES interleaved cycles, arms round-robin inside every cycle"
for ((c = 0; c < CYCLES; ++c)); do
    for cell in "${CELLS[@]}"; do
        for arm in on off ctl; do
            sample "P_${cell}_${arm}_1t"  env MAPAL_PAR=1    "$TMP/${cell}_${arm}"
            sample "P_${cell}_${arm}_par" env MAPAL_PAR=$PAR "$TMP/${cell}_${arm}"
        done
    done
    printf '.' >&2
done
echo >&2
echo

[ ${#VOID[@]} -eq 0 ] || echo "VOID cells (measured nothing, excluded): ${VOID[*]}"
python3 "$ROOT/benches/nn/pack_report.py" "$OUT"
