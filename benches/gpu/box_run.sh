#!/bin/bash
# Runs ON the GPU box, inside an unpacked bundle (benches/gpu/bundle.sh).
# 1. the differential (GPU stdout == oracle, every program, every arm),
# 2. whole programs: process wall, own `iter ms`, MAPAL_GPU_TRACE totals, per
#    host-rung arm, interleaved; the S54 multi-computation probes the same way,
#    each followed by its library legs (chain_bench.cu); the decode loop,
# 3. the timing harness vs cuBLAS and CUTLASS, every kernel arm.
# PROBE_ARMS / PROBE_REPS narrow the probe section (default: on, off-resident,
# off-devgen; 5 repetitions).
# Every line lands in results-<profile>.txt the moment it is produced.
#
# usage: bash box_run.sh <profile> <sm_NN> [reps]
set -uo pipefail
PROF=$1; SM=$2; REPS=${3:-30}
CUDA=${CUDA:-/usr/local/cuda}
LOG=results-$PROF.txt
say() { echo "$@" | tee -a "$LOG"; }
link() { cc "$1" mapal_gpu.c libmapal_rt.a -I"$CUDA/include" -L"$CUDA/lib64/stubs" -lcuda -lpthread -ldl -lm -o "$2" 2>&1 | tail -3; }
: > "$LOG"
say "== $PROF $SM $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.sm,power.limit,temperature.gpu,persistence_mode --format=csv | tee -a "$LOG"
nvcc --version | tail -1 | tee -a "$LOG"

# The harness is built for the plain arch (sm_90a -> sm_90): the Mapal leg is
# PTX the driver JITs for the device, and the library legs stay as measured before.
nvcc -O3 -std=c++17 -arch="${SM%a}" -Icutlass/include -Icutlass/tools/util/include \
    tile_bench.cu -lcublas -lcuda -o tile_bench 2>&1 | grep -v Wno | tail -5 | tee -a "$LOG"
[ -x tile_bench ] || { say "HARNESS BUILD FAILED"; exit 1; }
# S54: the library legs of the multi-computation probes (a failed build skips
# only the probe section's library rows).
nvcc -O3 -std=c++17 -arch="${SM%a}" -Icutlass/include -Icutlass/tools/util/include \
    chain_bench.cu -lcublasLt -lcublas -o chain_bench 2>&1 | grep -v Wno | tail -5 | tee -a "$LOG"
[ -x chain_bench ] || say "CHAIN HARNESS BUILD FAILED"

say "-- differential (GPU stdout vs oracle)"
pass=0; fail=0
for o in "$PROF"/*.host.o; do
    b=$(basename "$o" .host.o)
    case $b in mm_*|e2e_*|loop_mm_*|pt_*) continue ;; esac
    # `<prog>[_c][.<arm>]`: every arm of a face answers to that face's oracle.
    stem=${b%%.*}; base=${stem%_c}; exp="tiles/$base.expect"; [ "$stem" != "$base" ] && exp="tiles/${base}_c.expect"
    link "$o" "prog_$b"
    got=$(MAPAL_PTX="$PROF/$b.ptx" "./prog_$b" 2>&1); rc=$?
    if [ $rc -eq 0 ] && [ "$got" = "$(cat "$exp")" ]; then
        pass=$((pass + 1)); say "PASS $b $(tr '\n' ' ' <<< "$got")"
    else
        fail=$((fail + 1)); say "FAIL $b rc=$rc got=[$(tr '\n' ' ' <<< "$got")] want=[$(tr '\n' ' ' < "$exp")]"
    fi
done
say "differential: $pass pass, $fail fail"

# eager-init: a program that traps before its first site exits 101 in every
# arm, with the init thread possibly still running (the runtime joins it).
for o in "$PROF"/e2e_trap*.host.o; do
    b=$(basename "$o" .host.o)
    link "$o" "prog_$b"
    got=$(MAPAL_PTX="$PROF/$b.ptx" timeout 60 "./prog_$b" 2>&1); rc=$?
    if [ $rc -eq 101 ]; then say "TRAP-PASS $b rc=$rc $got"; else say "TRAP-FAIL $b rc=$rc $got"; fi
done

# Whole programs, both faces: the shipped emission, each host rung off, and
# the S52 host object (`s52`, PREV), interleaved per repetition. One untraced
# process for the wall time and the program's own `iter ms`, one traced for
# the phase totals; the cuBLAS whole-program floor (`tile_bench --program`).
say "-- e2e: host-link arms, process wall + MAPAL_GPU_TRACE totals"
ARMS="on off-resident off-devgen off-eager-init s52"
arm_stem() { [ "$2" = on ] && echo "$1" || echo "$1.$2"; }
e2e() { # <prog> <floor: n gemms ops>
    local p=$1; shift
    for face in "" _c; do
        for arm in $ARMS; do
            b=$(arm_stem "$p$face" "$arm")
            [ -f "$PROF/$b.host.o" ] && link "$PROF/$b.host.o" "e2e_$b"
        done
        for r in $(seq "${E2E_REPS:-7}"); do
            for arm in $ARMS; do
                b=$(arm_stem "$p$face" "$arm")
                [ -x "e2e_$b" ] || continue
                t0=$(date +%s%N)
                got=$(MAPAL_PTX="$PROF/$b.ptx" "./e2e_$b" 2>/dev/null | tr '\n' ' ')
                t1=$(date +%s%N)
                tr=$(MAPAL_GPU_TRACE=1 MAPAL_PTX="$PROF/$b.ptx" "./e2e_$b" 2>&1 >/dev/null)
                t2=$(date +%s%N)
                say "e2e prog=$p face=${face:-_exact} arm=$arm rep=$r wall_us=$(( (t1 - t0) / 1000 )) traced_wall_us=$(( (t2 - t1) / 1000 )) out=[$got] $(grep -m1 mapal_gpu_trace_total <<< "$tr")"
            done
        done
    done
    for r in 1 2 3; do say "floor prog=$p rep=$r $(./tile_bench --program "$@" 2>&1)"; done
}
e2e e2e_1024 1024 1 2
e2e e2e_4096 4096 1 2
e2e mm_share_1024 1024 2 3
e2e mm_chain_2048 2048 2 3
e2e e2e_chain_4096 4096 2 3
e2e loop_mm_1024_8 1024 8 2
say "$(./tile_bench --h2d 64 10 2>&1)"

# S54 multi-computation probes (`pt_<probe>_<n>`, gen_tiles.py): per probe, both
# faces × the host arms, interleaved per repetition — one untraced process for
# the wall and the program's own `iter ms`, one under MAPAL_GPU_TRACE for the
# phases, launches and bytes — then the library legs (chain_bench.cu) on the
# same probe, gated against the exact face's printed values.
say "-- probes: Mapal (wall, iter ms, MAPAL_GPU_TRACE) vs the library legs (chain_bench)"
PROBE_ARMS=${PROBE_ARMS:-on off-resident off-devgen}
for o in "$PROF"/pt_*.host.o; do
    [ -f "$o" ] || continue
    b=$(basename "$o" .host.o)
    case $b in *.*|*_c) continue ;; esac
    probe=${b#pt_}; n=${probe##*_}; probe=${probe%_*}
    for face in "" _c; do
        for arm in $PROBE_ARMS; do
            s=$(arm_stem "$b$face" "$arm")
            [ -f "$PROF/$s.host.o" ] && link "$PROF/$s.host.o" "pt_$s"
        done
    done
    vals=""
    for r in $(seq "${PROBE_REPS:-5}"); do
        for face in "" _c; do
            for arm in $PROBE_ARMS; do
                s=$(arm_stem "$b$face" "$arm")
                [ -x "pt_$s" ] || continue
                t0=$(date +%s%N)
                got=$(MAPAL_PTX="$PROF/$s.ptx" "./pt_$s" 2>/dev/null | tr '\n' ' ')
                t1=$(date +%s%N)
                tr=$(MAPAL_GPU_TRACE=1 MAPAL_PTX="$PROF/$s.ptx" "./pt_$s" 2>&1 >/dev/null)
                t2=$(date +%s%N)
                say "probe prog=$b face=${face:-_exact} arm=$arm rep=$r wall_us=$(( (t1 - t0) / 1000 )) traced_wall_us=$(( (t2 - t1) / 1000 )) out=[$got] $(grep -m1 mapal_gpu_trace_total <<< "$tr")"
                [ -z "$vals" ] && [ "$face$arm" = on ] && vals=${got%%iter ms=*}
            done
        done
    done
    # shellcheck disable=SC2086  # the values are separate arguments
    [ -x chain_bench ] && ./chain_bench "$probe" "$n" "$REPS" $vals 2>&1 | tee -a "$LOG"
done

# retain-delta: the decode loop end to end, shipped vs the rung off (`resident`
# then takes the function) vs every site a round trip (the arm the rung was
# measured against), both faces, interleaved; the launches per kernel from rep 1.
if [ -x prog_decode_1024_256 ]; then
    say "-- decode e2e (T=1024 d=256), MAPAL_GPU_TRACE=1"
    for r in 1 2 3; do
        for face in "" _c; do
            for arm in "" .off-retain-delta .off-retain-delta+resident; do
                b=decode_1024_256$face$arm
                [ -x "prog_$b" ] || continue
                t0=$(date +%s%N)
                tr=$(MAPAL_GPU_TRACE=1 MAPAL_PTX="$PROF/$b.ptx" "./prog_$b" 2>&1 >"dec_$b.out"); rc=$?
                t1=$(date +%s%N)
                say "decode prog=$b rep=$r rc=$rc wall_s=$(awk -v d=$((t1 - t0)) 'BEGIN { printf "%.3f", d / 1e9 }') out=[$(tr '\n' ' ' < "dec_$b.out")] $(grep -m1 mapal_gpu_trace_total <<< "$tr")"
                [ "$r" = 1 ] && grep 'op=launch' <<< "$tr" | grep -o 'kernel=[^ ]*' | sort | uniq -c |
                    while read -r n k; do say "decode_launch prog=$b $k n=$n"; done
            done
        done
    done
    say "-- decode floor: hand-written cuBLAS sgemv KV cache (tile_bench --decode)"
    ./tile_bench --decode 1024 256 "$REPS" 2>&1 | tee -a "$LOG"
fi

say "-- timing vs cuBLAS / CUTLASS (f32, reps=$REPS)"
bench() { # <n> <stem> <label> [--contract|--tf32]
    local n=$1 stem=$2 label=$3; shift 3
    read -r k gx gy blk sm < "$stem.launch"
    say "# n=$n face=$label"
    ./tile_bench "$n" "$n" "$n" "$stem.ptx" "$k" "$gx" "$gy" "$blk" "$sm" "$REPS" "$@" 2>&1 | tee -a "$LOG"
}
for n in 1024 1536 2048 3072 4096 8192; do
    for face in _exact _c _tf; do
        case $face in _exact) s=mm_$n; flag=() ;; _c) s=mm_${n}_c; flag=(--contract) ;; _tf) s=mm_${n}_tf; flag=(--tf32) ;; esac
        [ -f "$PROF/$s.launch" ] || continue
        bench "$n" "$PROF/$s" "$face" "${flag[@]}"
        for v in "$PROF/$s".*.launch; do
            [ -f "$v" ] || continue
            var=${v%.launch}; var=${var#"$PROF/$s".}
            bench "$n" "${v%.launch}" "$face.$var" "${flag[@]}"
        done
    done
done
say "== done"
