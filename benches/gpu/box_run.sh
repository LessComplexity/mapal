#!/bin/bash
# Runs ON the GPU box, inside an unpacked bundle (benches/gpu/bundle.sh).
# 1. the differential (GPU stdout == oracle, every program, every arm),
# 2. whole programs: process wall, own `iter ms`, MAPAL_GPU_TRACE totals, per
#    host-rung arm, interleaved; the S54 multi-computation probes the same way,
#    each followed by its library legs (chain_bench.cu); the decode loop,
# 3. the timing harness vs cuBLAS and CUTLASS, every kernel arm.
# E2E_ARMS narrows the e2e section's arms; PROBE_ARMS / PROBE_REPS the probe section
# (default: the bundle's `probe_arms` file, else on + every off-<rung> arm the bundle
# built for the probe + `cpu` and `null` where built; 5 repetitions). S62: every probe
# arm is value-gated against the program's oracle (`tiles/<stem>.expect`, else its
# `cpu` arm) before any timing, the arm order is reversed on even repetitions (ABBA),
# and PROBE_TRACE_REPS (default 2) repetitions also run a traced process.
# DIFFERENTIAL_ONLY=1 stops after value and trap checks, without compiling harnesses.
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
# S62 `hostopt`: the host CPU the host objects run on (the host profile's facts are checked
# against these; a run on another CPU family is another host).
lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core|Socket|L1d|L2|L3|CPU max MHz)' | sed 's/^/lscpu: /' | tee -a "$LOG"
say "cpu_flags $(grep -o -w -E 'avx2|fma|avx512f' /proc/cpuinfo | sort | uniq -c | tr -s ' \n' ' ') nproc=$(nproc)"
for x in index0 index2; do
    c=/sys/devices/system/cpu/cpu0/cache/$x
    say "cache $x size=$(cat $c/size 2>/dev/null) line=$(cat $c/coherency_line_size 2>/dev/null) sets=$(cat $c/number_of_sets 2>/dev/null) shared=$(cat $c/shared_cpu_list 2>/dev/null)"
done

say "-- differential (GPU stdout vs oracle)"
pass=0; fail=0
for o in "$PROF"/*.host.o; do
    [ -f "$o" ] || continue
    b=$(basename "$o" .host.o)
    case $b in mm_*|e2e_*|loop_mm_*|pt_*) continue ;; esac
    # `<prog>[_c][.<arm>]`: every arm of a face answers to that face's oracle.
    stem=${b%%.*}; base=${stem%_c}; exp="tiles/$base.expect"; [ "$stem" != "$base" ] && exp="tiles/${base}_c.expect"
    # S54: a TF32 program answers to its `off-epilogue` arm (bundle.sh), run first.
    case $stem in *_tf)
        exp="tiles/$stem.ref"
        if [ ! -s "$exp" ]; then
            if ! { link "$PROF/$stem.off-epilogue.host.o" "prog_ref_$stem" &&
                MAPAL_PTX="$PROF/$stem.off-epilogue.ptx" "./prog_ref_$stem" > "$exp" 2>&1; }; then
                rm -f "$exp"
                fail=$((fail + 1)); say "FAIL $b TF32 reference failed"; continue
            fi
        fi ;;
    esac
    if [ ! -f "$exp" ]; then
        fail=$((fail + 1)); say "FAIL $b missing oracle $exp"; continue
    fi
    if ! link "$o" "prog_$b"; then
        fail=$((fail + 1)); say "FAIL $b link failed"; continue
    fi
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
trap_pass=0; trap_fail=0
for o in "$PROF"/e2e_trap*.host.o; do
    [ -f "$o" ] || continue
    b=$(basename "$o" .host.o)
    if ! link "$o" "prog_$b"; then
        trap_fail=$((trap_fail + 1)); say "TRAP-FAIL $b link failed"; continue
    fi
    got=$(MAPAL_PTX="$PROF/$b.ptx" timeout 60 "./prog_$b" 2>&1); rc=$?
    if [ $rc -eq 101 ]; then
        trap_pass=$((trap_pass + 1)); say "TRAP-PASS $b rc=$rc $got"
    else
        trap_fail=$((trap_fail + 1)); say "TRAP-FAIL $b rc=$rc $got"
    fi
done
say "traps: $trap_pass pass, $trap_fail fail"
if [ "$fail" -ne 0 ] || [ "$trap_fail" -ne 0 ]; then
    say "VALIDATION FAILED (value/trap failures)"
    exit 1
fi
if [ "${DIFFERENTIAL_ONLY:-0}" = 1 ]; then
    [ "$pass" -gt 0 ] || { say "VALIDATION FAILED (empty differential scope)"; exit 1; }
    say "== done differential-only"
    exit 0
fi

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

# Whole programs, both faces: the shipped emission, each host rung off, and
# the S52 host object (`s52`, PREV), interleaved per repetition. One untraced
# process for the wall time and the program's own `iter ms`, one traced for
# the phase totals; the cuBLAS whole-program floor (`tile_bench --program`).
# E2E=0 (S62): skip this section (a kernel item's session).
[ "${E2E:-1}" = 1 ] && say "-- e2e: host-link arms, process wall + MAPAL_GPU_TRACE totals"
ARMS=${E2E_ARMS:-on off-resident off-devgen off-eager-init off-arena off-arch-rungs s52}
arm_stem() { [ "$2" = on ] && echo "$1" || echo "$1.$2"; }
e2e() { # <prog> <floor: n gemms ops>
    local p=$1; shift
    compgen -G "$PROF/$p.*host.o" > /dev/null || compgen -G "$PROF/${p}_c.*host.o" > /dev/null ||
        [ -f "$PROF/$p.host.o" ] || return 0
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
                got=$(MAPAL_PTX="$PROF/$b.ptx" "./e2e_$b" 2>/dev/null | tr '\n' ' '); rc=$?
                t1=$(date +%s%N)
                tr=$(MAPAL_GPU_TRACE=1 MAPAL_PTX="$PROF/$b.ptx" "./e2e_$b" 2>&1 >/dev/null); trace_rc=$?
                t2=$(date +%s%N)
                say "e2e prog=$p face=${face:-_exact} arm=$arm rep=$r wall_us=$(( (t1 - t0) / 1000 )) traced_wall_us=$(( (t2 - t1) / 1000 )) out=[$got] $(grep -m1 mapal_gpu_trace_total <<< "$tr") rc=$rc trace_rc=$trace_rc"
                [ "$rc" -eq 0 ] && [ "$trace_rc" -eq 0 ] || { say "VALIDATION FAILED (e2e $b rep=$r process exit)"; exit 1; }
            done
        done
    done
    for r in 1 2 3; do say "floor prog=$p rep=$r $(./tile_bench --program "$@" 2>&1)"; done
}
if [ "${E2E:-1}" = 1 ]; then
    e2e e2e_1024 1024 1 2
    e2e e2e_4096 4096 1 2
    e2e mm_share_1024 1024 2 3
    e2e mm_chain_2048 2048 2 3
    e2e e2e_chain_4096 4096 2 3
    e2e loop_mm_1024_8 1024 8 2
    say "$(./tile_bench --h2d 64 10 2>&1)"
fi

# S54 multi-computation probes (`pt_<probe>_<n>`, gen_tiles.py): per probe, both
# faces × the host arms, interleaved per repetition — one untraced process for
# the wall and the program's own `iter ms`, one under MAPAL_GPU_TRACE for the
# phases, launches and bytes — then the library legs (chain_bench.cu) on the
# same probe, gated against the exact face's printed values.
say "-- probes: Mapal (wall, iter ms, MAPAL_GPU_TRACE) vs the library legs (chain_bench)"
[ -z "${PROBE_ARMS:-}" ] && [ -s probe_arms ] && PROBE_ARMS=$(cat probe_arms)
# S54 arch-select: by default every OFF arm the bundle built for the probe —
# bundle.sh drops an arm equal to the shipped text, so these are exactly the
# rungs this part ships ON there.
off_arms() { local o; for o in "$PROF/$1".off-*.host.o; do [ -f "$o" ] && { o=${o%.host.o}; echo "${o##*.}"; }; done; }
# S62: the CPU-only build and the null copy, where the bundle built them.
xtra_arms() { local x; for x in cpu cpu-inline null; do [ -f "$PROF/$1.$x.host.o" ] && echo "$x"; done; }
# probe_out <stem>: one untraced process's stdout, one line (`iter ms` included).
probe_out() { MAPAL_PTX="$PROF/$1.ptx" "./pt_$1" 2>/dev/null | tr '\n' ' '; }
val_fail=0
for o in "$PROF"/pt_*.host.o; do
    [ -f "$o" ] || continue
    b=$(basename "$o" .host.o)
    case $b in *.*|*_c) continue ;; esac
    pa=$( (off_arms "$b"; off_arms "${b}_c"; xtra_arms "$b") | sort -u | tr "\n" " "); pa=${PROBE_ARMS:-on $pa}
    probe=${b#pt_}; n=${probe##*_}; probe=${probe%_*}
    runs=()
    for face in "" _c; do
        for arm in $pa; do
            s=$(arm_stem "$b$face" "$arm")
            [ -f "$PROF/$s.host.o" ] && link "$PROF/$s.host.o" "pt_$s"
            [ -x "pt_$s" ] && runs+=("$face:$arm")
        done
    done
    # The value gate, before any timing: every arm prints its face's oracle.
    bad=0
    for fa in "${runs[@]}"; do
        face=${fa%%:*}; arm=${fa#*:}; s=$(arm_stem "$b$face" "$arm")
        got=$(probe_out "$s"); got=${got%%iter ms=*}
        exp="tiles/$b$face.expect"
        if [ -s "$exp" ]; then want=$(tr '\n' ' ' < "$exp"); else want=$(probe_out "$b$face.cpu"); want=${want%%iter ms=*}; fi
        if [ "$got" != "$want" ]; then
            bad=1; say "VALUE-FAIL prog=$b face=${face:-_exact} arm=$arm got=[$got] want=[$want]"
        fi
    done
    [ "$bad" = 0 ] || { val_fail=$((val_fail + 1)); continue; }
    say "value-gate prog=$b arms=${#runs[@]} pass"
    vals=""
    for r in $(seq "${PROBE_REPS:-5}"); do
        order=("${runs[@]}")
        if [ $((r % 2)) = 0 ]; then
            order=(); for ((i = ${#runs[@]} - 1; i >= 0; i--)); do order+=("${runs[$i]}"); done
        fi
        for fa in "${order[@]}"; do
            face=${fa%%:*}; arm=${fa#*:}; s=$(arm_stem "$b$face" "$arm")
            t0=$(date +%s%N)
            got=$(probe_out "$s"); rc=$?
            t1=$(date +%s%N)
            tr=""; trace_rc=0
            if [ "$r" -le "${PROBE_TRACE_REPS:-2}" ]; then
                tr=$(MAPAL_GPU_TRACE=1 MAPAL_PTX="$PROF/$s.ptx" "./pt_$s" 2>&1 >/dev/null); trace_rc=$?
            fi
            say "probe prog=$b face=${face:-_exact} arm=$arm rep=$r wall_us=$(( (t1 - t0) / 1000 )) out=[$got] $(grep -m1 mapal_gpu_trace_total <<< "$tr") rc=$rc trace_rc=$trace_rc"
            # S54: each launch's kernel time under the trace, in launch order.
            [ -n "$tr" ] && say "probe_kernels prog=$b face=${face:-_exact} arm=$arm rep=$r $(grep -o 'op=launch kernel=mapal_tile_[0-9a-f]* ms=[0-9.]*' <<< "$tr" | sed 's/op=launch kernel=mapal_tile_//' | tr '\n' ' ')"
            [ "$rc" -eq 0 ] && [ "$trace_rc" -eq 0 ] || { say "VALIDATION FAILED (probe $s rep=$r process exit)"; exit 1; }
            [ -z "$vals" ] && [ "$face$arm" = on ] && vals=${got%%iter ms=*}
            # S62 SUBMIT="<arm>...": a third process under MAPAL_GPU_TRACE=2 (no
            # post-launch synchronize), so each launch's ms is its CPU enqueue.
            case " ${SUBMIT:-} " in *" $arm "*)
                got2=$(MAPAL_GPU_TRACE=2 MAPAL_PTX="$PROF/$s.ptx" "./pt_$s" 2>submit.err | tr '\n' ' '); submit_rc=$?
                say "submit prog=$b face=${face:-_exact} arm=$arm rep=$r out=[$got2] $(grep -m1 mapal_gpu_trace_total submit.err) rc=$submit_rc"
                say "submit_ops prog=$b face=${face:-_exact} arm=$arm rep=$r $(grep -oE 'op=(launch kernel=mapal_tile_[0-9a-f]*|gen kernel=mapal_gen_[0-9a-f]*|event record=[0-9]* wait=[0-9]* auto=[0-9]*) ms=[0-9.]*' submit.err | sed -E 's/op=(launch|gen) kernel=mapal_//; s/op=event /event:/; s/ (wait|auto)=/,\1=/g' | tr '\n' ' ')"
                [ "$submit_rc" -eq 0 ] || { say "VALIDATION FAILED (submit $s rep=$r process exit)"; exit 1; } ;;
            esac
        done
    done
    # S62: one host lane's rates (`place`'s host facts): the CPU-only build at MAPAL_PAR=1.
    case " ${LANE_PROGS:-pt_pc_128 pt_pj_m1_n2048_k7168} " in *" $b "*)
        for r in $(seq 10); do
            for face in "" _c; do
                [ -x "pt_$b$face.cpu" ] || continue
                say "lane prog=$b face=${face:-_exact} rep=$r out=[$(MAPAL_PAR=1 MAPAL_PTX=none "./pt_$b$face.cpu" 2>/dev/null | tr '\n' ' ')]"
            done
        done ;;
    esac
    # shellcheck disable=SC2086  # the values are separate arguments
    case $n in *[!0-9]*) continue ;; esac  # S62 `rmlp_<M>x<d>x<h>`: no library leg
    [ -x chain_bench ] && ./chain_bench "$probe" "$n" "$REPS" $vals 2>&1 | tee -a "$LOG"
done
say "probe value gate: $val_fail program(s) failed"

# The decode loop end to end, both faces, interleaved: shipped vs each R5 rung
# off (S54 `retain-resident`, `delta-h2d`, `lane-stride`), the s54-base placement
# (all three off), and every site a round trip (S53's arm); one untraced process
# (`wall_s`) and one under MAPAL_GPU_TRACE (`traced_wall_s`: a traced launch
# synchronizes); the launches per kernel from rep 1. Five repetitions: an arm
# within 10% of ON needs its sign stable across them and across boxes.
if [ -x prog_decode_1024_256 ]; then
    say "-- decode e2e (T=1024 d=256), MAPAL_GPU_TRACE=1"
    for r in 1 2 3 4 5; do
        for face in "" _c; do
            for arm in "" .null .cpu .off-hostopt .off-place .off-retain-resident .on-retain-resident .off-delta-h2d .off-lane-stride \
                .off-retain-resident+delta-h2d+lane-stride .off-retain-delta .on-retain-delta .off-retain-delta+resident; do
                b=decode_1024_256$face$arm
                [ -x "prog_$b" ] || continue
                t0=$(date +%s%N)
                MAPAL_PTX="$PROF/$b.ptx" "./prog_$b" >/dev/null 2>&1; wall_rc=$?
                t1=$(date +%s%N)
                tr=$(MAPAL_GPU_TRACE=1 MAPAL_PTX="$PROF/$b.ptx" "./prog_$b" 2>&1 >"dec_$b.out"); rc=$?
                t2=$(date +%s%N)
                say "decode prog=$b rep=$r rc=$rc wall_s=$(awk -v d=$((t1 - t0)) 'BEGIN { printf "%.3f", d / 1e9 }') traced_wall_s=$(awk -v d=$((t2 - t1)) 'BEGIN { printf "%.3f", d / 1e9 }') out=[$(tr '\n' ' ' < "dec_$b.out")] $(grep -m1 mapal_gpu_trace_total <<< "$tr") wall_rc=$wall_rc"
                [ "$rc" -eq 0 ] && [ "$wall_rc" -eq 0 ] || { say "VALIDATION FAILED (decode $b rep=$r process exit)"; exit 1; }
                [ "$r" = 1 ] && grep 'op=launch' <<< "$tr" | grep -o 'kernel=[^ ]*' | sort | uniq -c |
                    while read -r n k; do say "decode_launch prog=$b $k n=$n"; done
            done
        done
    done
    say "-- decode floor: hand-written cuBLAS sgemv KV cache (tile_bench --decode)"
    ./tile_bench --decode 1024 256 "$REPS" 2>&1 | tee -a "$LOG"
    # S54: the eager loop a framework user writes, with and without a
    # hand-written cache (host wall, launches), gated on Mapal's out[0], out[d-1].
    vals=$(head -2 dec_decode_1024_256.out | tr '\n' ' ')
    # shellcheck disable=SC2086  # the values are separate arguments
    [ -x chain_bench ] && ./chain_bench decode_1024 256 "$REPS" $vals 2>&1 | tee -a "$LOG"
fi

# S62 `splitk` (ADR-0040): per `mm_MxNxK` cell, every arm (exact, contract, the
# reassociation face on both step faces, its OFF / null / forced-S arms) with
# cuBLAS and CUTLASS in ONE process (tile_bench --splitk: value gates first,
# round-robin, ABBA inside), twice, the second in reversed arm order. Then the
# whole programs of the split arms on this driver (workspace + two launches):
# each must exit 0 and print the two cells its OFF arm prints within 1e-3
# relative (the first line, a sum over every cell, cancels: not compared).
if compgen -G "$PROF/mm_*x*x*_r.launch" > /dev/null; then
    say "-- splitk: mm_MxNxK cells, every arm in one process, two processes ABBA (reps=$REPS)"
    for l in "$PROF"/mm_*x*x*_r.launch; do
        cell=$(basename "$l" _r.launch); d=${cell#mm_}; M=${d%%x*}; d=${d#*x}; N=${d%%x*}; K=${d#*x}
        # S62 integration: every arm the bundle built for the cell (`.base`, `.off-<rung>`, `.null`).
        arms=$(cd "$PROF" && ls "$cell".launch "$cell".*.launch "${cell}"_c.launch "${cell}"_c.*.launch "${cell}"_r*.launch 2>/dev/null |
            sed 's/\.launch$//' | tr '\n' ' ')
        say "# splitk cell=$cell M=$M N=$N K=$K arms=$arms"
        for pass in 1 2; do
            order=$arms; [ $pass = 2 ] && order=$(tr ' ' '\n' <<< "$arms" | sed '/^$/d' | tac | tr '\n' ' ')
            # shellcheck disable=SC2086  # the arms are separate arguments
            ./tile_bench --splitk "$M" "$N" "$K" "$REPS" "$PROF" $order 2>&1 | sed "s/^/pass=$pass cell=$cell /" | tee -a "$LOG"
            [ "${PIPESTATUS[0]}" -eq 0 ] || say "SPLITK GATE FAILED cell=$cell pass=$pass"
        done
        for v in _r _rc; do
            [ -f "$PROF/$cell$v.host.o" ] || continue
            for s in "$cell$v" "$cell$v.off-splitk"; do
                [ -f "$PROF/$s.host.o" ] && link "$PROF/$s.host.o" "prog_$s"
            done
            on=$(MAPAL_PTX="$PROF/$cell$v.ptx" "./prog_$cell$v" 2>&1 | tr '\n' ' '); rc=$?
            off=$on; [ -x "prog_$cell$v.off-splitk" ] && off=$(MAPAL_PTX="$PROF/$cell$v.off-splitk.ptx" "./prog_$cell$v.off-splitk" 2>&1 | tr '\n' ' ')
            ok=$(awk -v a="$on" -v b="$off" 'BEGIN { n = split(a, x, " "); m = split(b, y, " "); ok = n == 3 && m == 3
                for (i = 2; i <= n && ok; i++) { d = x[i] - y[i]; s = y[i] < 0 ? -y[i] : y[i]; if ((d < 0 ? -d : d) > 1e-3 * (s > 1 ? s : 1)) ok = 0 }
                print ok ? "yes" : "NO" }')
            say "splitk_program prog=$cell$v rc=$rc out=[$on] off=[$off] within_1e-3=$ok"
        done
    done
fi

say "-- timing vs cuBLAS / CUTLASS (f32, reps=$REPS, passes=${PASSES:-1})"
# A cell is `n` (square) or `MxNxK` (S62: the `mm_<M>x<N>x<K>` programs).
dims() { case $1 in *x*x*) echo "${1//x/ }" ;; *) echo "$1 $1 $1" ;; esac; }
bench() { # <cell> <stem> <label> [--contract|--tf32]
    local n=$1 stem=$2 label=$3; shift 3
    read -r k gx gy blk sm < "$stem.launch"
    say "# n=$n face=$label pass=$pass"
    # shellcheck disable=SC2046  # M N K are three arguments
    ./tile_bench $(dims "$n") "$stem.ptx" "$k" "$gx" "$gy" "$blk" "$sm" "$REPS" "$@" 2>&1 | tee -a "$LOG"
    # WARM=1 (S62): the same bench again with operands left in L2 (`tile_bench --warm`).
    if [ "${WARM:-0}" = 1 ]; then
        say "# n=$n face=$label pass=$pass l2=warm"
        # shellcheck disable=SC2046
        ./tile_bench $(dims "$n") "$stem.ptx" "$k" "$gx" "$gy" "$blk" "$sm" "$REPS" "$@" --warm 2>&1 | tee -a "$LOG"
    fi
}
# S54 arch-select: the session's first bench ran 3.6% off its own repeat (the `null` arm,
# arch-select-rtx-3090.txt _exact 1024): one discarded bench first.
[ -f "$PROF/mm_1024.launch" ] && { read -r k gx gy blk sm < "$PROF/mm_1024.launch"
    ./tile_bench 1024 1024 1024 "$PROF/mm_1024.ptx" "$k" "$gx" "$gy" "$blk" "$sm" "$REPS" > /dev/null 2>&1; }
cells="1024 1536 2048 3072 4096 8192 $(for f in "$PROF"/mm_*x*x*.launch; do [ -f "$f" ] && basename "$f" .launch; done |
    sed -n 's/^mm_\([0-9]*x[0-9]*x[0-9]*\)$/\1/p' | sort -t x -n -k1,1 -k2,2 -k3,3)"
# PASSES=2 (S62): every cell again with its arm order reversed (ABBA), so an arm's
# place in the bench order is not read as its effect. FACES narrows the faces.
for pass in $(seq "${PASSES:-1}"); do
    for n in $cells; do
        for face in ${FACES:-_exact _c _tf}; do
            case $face in _exact) s=mm_$n; flag=() ;; _c) s=mm_${n}_c; flag=(--contract) ;; _tf) s=mm_${n}_tf; flag=(--tf32) ;; esac
            [ -f "$PROF/$s.launch" ] || continue
            legs=(shipped)
            for v in "$PROF/$s".*.launch; do
                [ -f "$v" ] || continue
                var=${v%.launch}; legs+=("${var#"$PROF/$s".}")
            done
            order=("${legs[@]}")
            if [ $((pass % 2)) = 0 ]; then
                order=(); for ((i = ${#legs[@]} - 1; i >= 0; i--)); do order+=("${legs[i]}"); done
            fi
            for var in "${order[@]}"; do
                if [ "$var" = shipped ]; then bench "$n" "$PROF/$s" "$face" "${flag[@]}"
                else bench "$n" "$PROF/$s.$var" "$face.$var" "${flag[@]}"; fi
            done
        done
    done
done
say "== done"
