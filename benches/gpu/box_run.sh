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
# S64 order: differential, splitk cells, mlp, e2e, probes, decode, the evidence check
# (evidence-<profile>.tsv: the bundle's facts + this box's JIT image registers), the PyTorch legs
# (torch_bench.py, installed in the background from the start; TORCH=0 skips them), timing.
# Every line lands in results-<profile>.txt the moment it is produced.
#
# per-site-choice B5: a binary with site tables (`<stem>.sites`) is one build whose arms the environment picks:
# `on` = AUTO (the wisdom, ./wisdom), `rules` (MAPAL_GPU_PICK=rules: candidate 0), `pick<i>` (candidate i). Every run
# reads the wisdom and never trials (MAPAL_GPU_TUNE=0) except the probe section's one untimed tune run per stem
# (`tune` lines; RETUNE=1 ignores a wisdom file left by an earlier run). Value gates cover every candidate index
# up to the stem's largest table. A program whose every pick is candidate 0 is `carried` (no timing, no library
# leg); a looped program keeps its `.c1` (`--cands=1`) arm: RULES vs `.c1` is the cost of `mapal_gpu_site`.
# `load` lines: module load ms on a cold driver JIT cache and on a later (warm) run. PICKS=1 also times every
# candidate. The driver's JIT cache must be on and writable (JIT_CACHE_OK=1 runs without it).
#
# usage: bash box_run.sh <profile> <sm_NN> [reps]
set -uo pipefail
PROF=$1; SM=$2; REPS=${3:-30}
CUDA=${CUDA:-/usr/local/cuda}
LOG=results-$PROF.txt
# S64: a looped program's CPU build (pt_rt_mlp2_1024 `.cpu`) overflows the default 8 MB stack.
ulimit -s unlimited 2>/dev/null || ulimit -s 65536 2>/dev/null || true
say() { echo "$@" | tee -a "$LOG"; }
# per-site-choice: the wisdom of this box; only the tune pass trials. The tables grow the PTX x4-11: the JIT
# cache must hold every image across the runs (the driver's default size does not).
export MAPAL_WISDOM=$PWD/wisdom MAPAL_GPU_TUNE=0 CUDA_CACHE_MAXSIZE=${CUDA_CACHE_MAXSIZE:-4294967296}
# The candidate indexes a stem's value gates run (`auto` without a table).
picks() { local n; n=$(awk '{ if ($3 > m) m = $3 } END { print m + 0 }' "$1.sites" 2>/dev/null)
    if [ "${n:-0}" -gt 0 ]; then seq 0 $((n - 1)); else echo auto; fi; }
# MAPAL_GPU_PICK of an arm: `rules`, `pick<i>` -> i, every other arm AUTO.
sel() { case $1 in rules) echo rules ;; pick*) echo "${1#pick}" ;; *) echo auto ;; esac; }
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

jc=${CUDA_CACHE_PATH:-$HOME/.nv/ComputeCache}
if [ "${CUDA_CACHE_DISABLE:-0}" = 1 ] || ! { mkdir -p "$jc" && touch "$jc/.mapal_w" && rm -f "$jc/.mapal_w"; }; then
    say "JIT CACHE OFF OR NOT WRITABLE path=$jc disable=${CUDA_CACHE_DISABLE:-0}"; [ "${JIT_CACHE_OK:-0}" = 1 ] || exit 1
fi
say "jit_cache path=$jc max_bytes=$CUDA_CACHE_MAXSIZE used_kb=$(du -sk "$jc" 2>/dev/null | cut -f1)"

say "-- differential (GPU stdout vs oracle)"
pass=0; fail=0
for o in "$PROF"/*.host.o; do
    [ -f "$o" ] || continue
    b=$(basename "$o" .host.o)
    case $b in mm_*|e2e_*|loop_mm_*|pt_*) continue ;; esac
    # `<prog>[_c|_d][.<arm>]`: every arm of a face answers to that face's oracle; site-geometry: the default
    # face (`_d`) to the exact one within ADR-0040 D2's bound (value_gate.py, tiles/<prog>.kmax).
    stem=${b%%.*}; base=${stem%_c}; exp="tiles/$base.expect"; [ "$stem" != "$base" ] && exp="tiles/${base}_c.expect"
    dflt=; [ "${stem%_d}" != "$stem" ] && [ -s "tiles/${stem%_d}.kmax" ] && { base=${stem%_d}; dflt=1; exp="tiles/$base.expect"; }
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
    # per-site-choice: every candidate index of a tabled stem answers to the oracle (`<stem>@<i>`).
    for pk in $(picks "$PROF/$b"); do
    bp=$b; [ "$pk" = auto ] || bp="$b@$pk"
    got=$(MAPAL_GPU_PICK=$pk MAPAL_PTX="$PROF/$b.ptx" "./prog_$b" 2>&1); rc=$?
    # The clock is not a value: a program's `iter ms=` line drops on both sides.
    got=$(grep -v '^iter ms=' <<< "$got"); want=$(grep -v '^iter ms=' "$exp")
    if [ -n "$dflt" ]; then
        tf=(); [ -s "tiles/${base}_xabs.expect" ] && tf=("tiles/${base}_xabs.expect")  # D2 absolute, with M
        grep -q ': tf32' "tiles/$base.mapal" 2>/dev/null && tf+=(--tf32)
        grep -q ': bf16' "tiles/$base.mapal" 2>/dev/null && tf+=(--bf16)
        g="process exit $rc"
        if [ $rc -eq 0 ] && g=$(python3 value_gate.py "$exp" "$(tr '\n' ' ' <<< "$got")" "$(cat "tiles/$base.kmax")" ${tf[@]+"${tf[@]}"}); then
            pass=$((pass + 1)); say "PASS-BOUND $bp K=$(cat "tiles/$base.kmax") $g $(tr '\n' ' ' <<< "$got")"
        else
            fail=$((fail + 1)); say "FAIL $bp rc=$rc default face outside D2's bound: $g got=[$(tr '\n' ' ' <<< "$got")] want=[$(tr '\n' ' ' < "$exp")]"
        fi
        continue
    fi
    if [ $rc -eq 0 ] && [ "$got" = "$want" ]; then
        pass=$((pass + 1)); say "PASS $bp $(tr '\n' ' ' <<< "$got")"
    # TF32: a split-K atomic combine adds in no fixed order, so two runs of one build differ in the
    # last bits; within 1e-4 relative of the reference (TF32 itself is 2^-11).
    elif [ $rc -eq 0 ] && [[ $stem == *_tf ]] && python3 -c 'import sys
g, w = sys.argv[1].split(), sys.argv[2].split()
sys.exit(not (len(g) == len(w) and all(abs(float(a) - float(b)) <= 1e-4 * max(abs(float(a)), abs(float(b)), 1.0) for a, b in zip(g, w))))' "$got" "$want"; then
        pass=$((pass + 1)); say "PASS-ORDER $bp $(tr '\n' ' ' <<< "$got")"
    else
        fail=$((fail + 1)); say "FAIL $bp rc=$rc got=[$(tr '\n' ' ' <<< "$got")] want=[$(tr '\n' ' ' < "$exp")]"
    fi
    done
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

# S64: the PyTorch legs (torch_bench.py) run last; their install overlaps the Mapal sections.
if [ "${TORCH:-1}" = 1 ] && [ -f torch_bench.py ] && [ -n "${TORCH_PY:-}" ]; then
    # A local box: TORCH_PY=<python with torch> is used as it is (no apt-get / pip).
    "$TORCH_PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)' > torch-install.log 2>&1 &
    TORCH_PID=$!
elif [ "${TORCH:-1}" = 1 ] && [ -f torch_bench.py ]; then
    # The pin: 2.11.0 for the image's Python 3.10 (the version of the local model-scale legs), the cu130 wheel
    # (cross-arch: the CUDA 13 image; TORCH_INDEX overrides). torch.compile's Triton builds a C launcher
    # against Python.h: python3-dev.
    ( { apt-get update -qq && apt-get install -y -qq python3-pip python3-dev; } || true
      # fusion-natural-forms: a 3090 box's download timed out once (no torch legs): retry the wheel.
      python3 -m pip install -q --timeout 120 --retries 10 "torch==${TORCH_VERSION:-2.11.0}" \
          --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
      # competitors (comp_bench.py): the pip-wheel legs — Liger, FlashInfer and its prebuilt kernels for this part's
      # family (flashinfer.ai's cu130 index; a part without one JIT-compiles on first use). The source-built legs
      # (FlashAttention-2 / -3: comp_build.sh) build after the timed sections.
      if [ "${COMP:-1}" = 1 ]; then
          t0=$(date +%s)
          fv=${FLASHINFER_VERSION:-0.7.5.post1}
          python3 -m pip install -q --timeout 120 --retries 5 "liger-kernel==${LIGER_VERSION:-0.8.5}" "flashinfer-python==$fv"
          case $SM in sm_80 | sm_86 | sm_87) fj=sm80 ;; sm_89) fj=sm89 ;; sm_90*) fj=sm90a ;; sm_100*) fj=sm100a ;;
              sm_12*) fj=sm120f ;; *) fj= ;; esac
          [ -n "$fj" ] && python3 -m pip install -q --timeout 120 --retries 5 --no-deps "flashinfer-jit-cache==$fv+cu130" \
              "flashinfer-jit-cache-$fj==$fv+cu130" --extra-index-url https://flashinfer.ai/whl/cu130
          echo "comp pip install s=$(($(date +%s) - t0)) flashinfer_jit_cache=${fj:-none}"
      fi
    ) > torch-install.log 2>&1 &
    TORCH_PID=$!
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
# model-scale: the projections' cuBLAS / cuBLASLt legs (gemm_lib.cu).
if [ -f gemm_lib.cu ]; then
    nvcc -O3 -std=c++17 -arch="${SM%a}" gemm_lib.cu -lcublasLt -lcublas -o gemm_lib 2>&1 | grep -v Wno | tail -5 | tee -a "$LOG"
    [ -x gemm_lib ] || say "GEMM HARNESS BUILD FAILED"
fi
# cross-arch: the device calibration (calib.cu, every kind once, then `gsync`, `copy` and `body` twice more: the medians
# of three the profile's launch, copy and body properties cite) -> calib-<profile>.txt, while the GPU is idle (the torch
# install downloads beside it). CALIB=0 skips it.
if [ "${CALIB:-1}" = 1 ] && [ -f calib.cu ]; then
    nvcc -O3 -std=c++17 -arch="${SM%a}" -rdc=true calib.cu -o calib -lcudadevrt 2>&1 | grep -v Wno | tail -5 | tee -a "$LOG"
    if [ -x calib ]; then
        { timeout 20m ./calib; for r in 2 3; do timeout 5m ./calib gsync copy body | sed "s/^/run=$r /"; done; } > "calib-$PROF.txt" 2>&1
        say "calib: $(grep -c '' "calib-$PROF.txt") lines; $(grep -m1 '^facts' "calib-$PROF.txt"); $(grep -m1 '^ffma' "calib-$PROF.txt"); $(grep -m1 '^copy pageable_pm' "calib-$PROF.txt"); $(grep -m1 '^body coop_body_cycles' "calib-$PROF.txt")"
    else
        say "CALIB BUILD FAILED"
    fi
    # exp (ADR-0042 D5): the part's `ex2.approx.ftz.f32` error (`Caps::ex2`) and the hardware form's ulp,
    # exhaustively (exp_calib.cu), appended to the calibration as `exp ` lines.
    if [ -f exp_calib.cu ]; then
        nvcc -O3 -arch="${SM%a}" exp_calib.cu -o exp_calib 2>&1 | grep -v Wno | tail -5 | tee -a "$LOG"
        if [ -x exp_calib ]; then
            timeout 10m ./exp_calib 2>&1 | sed 's/^/exp /' >> "calib-$PROF.txt"
            say "exp_calib: $(grep '^exp ' "calib-$PROF.txt" | tr '\n' ' ')"
        else
            say "EXP_CALIB BUILD FAILED"
        fi
    fi
fi
# S63: the residual MLP's kernels vs cuBLAS, cuBLASLt and CUTLASS (mlp_bench.cu).
if [ -f mlp_bench.cu ]; then
    nvcc -O3 -std=c++17 -arch="${SM%a}" -Icutlass/include -Icutlass/tools/util/include \
        mlp_bench.cu -lcublasLt -lcublas -lcuda -o mlp_bench 2>&1 | grep -v Wno | tail -5 | tee -a "$LOG"
    [ -x mlp_bench ] || say "MLP HARNESS BUILD FAILED"
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
        # S64: and the exact face's re-judge arms (`_x.on-<rung>`).
        arms=$(cd "$PROF" && ls "$cell".launch "$cell".*.launch "${cell}"_x.launch "${cell}"_x.*.launch "${cell}"_c.launch "${cell}"_c.*.launch \
            "${cell}"_r*.launch 2>/dev/null | sed 's/\.launch$//' | tr '\n' ' ')
        say "# splitk cell=$cell M=$M N=$N K=$K arms=$arms"
        for pass in 1 2; do
            order=$arms; [ $pass = 2 ] && order=$(tr ' ' '\n' <<< "$arms" | sed '/^$/d' | tac | tr '\n' ' ')
            # shellcheck disable=SC2086  # the arms are separate arguments
            ./tile_bench --splitk "$M" "$N" "$K" "$REPS" "$PROF" $order 2>&1 | sed "s/^/pass=$pass cell=$cell /" | tee -a "$LOG"
            [ "${PIPESTATUS[0]}" -eq 0 ] || say "SPLITK GATE FAILED cell=$cell pass=$pass"
        done
        # S63: the whole program of every arm through mapal_gpu.c (the output's
        # `mapal_gpu_zero` + the atomic launch on the default face, the workspace +
        # reduce on the fixed one) prints the exact face's value within D2's bound.
        [ -f "$PROF/${cell}_x.host.o" ] && link "$PROF/${cell}_x.host.o" "prog_${cell}_x"
        # A three-line program's first line is a host sum over every cell, which cancels
        # (S62): the cells it prints after it are compared.
        cells_of() { local x; read -r -a x <<< "$1"; if [ ${#x[@]} -eq 3 ]; then echo "${x[@]:1}"; else echo "${x[@]}"; fi; }
        ref=$(MAPAL_PTX="$PROF/${cell}_x.ptx" "./prog_${cell}_x" 2>&1 | tr '\n' ' ')
        cells_of "$ref" > "prog_${cell}_x.out"
        for l2 in "$PROF/$cell".launch "$PROF/$cell".off-*.launch "$PROF/${cell}"_c.launch "$PROF/${cell}"_r.launch "$PROF/${cell}"_rc.launch; do
            [ -f "$l2" ] || continue
            # cross-arch: an arm's `.pick<i>.launch` is a tile_bench leg (no program of its own).
            case $l2 in *.pick*.launch) continue ;; esac
            s=$(basename "$l2" .launch)
            [ -f "$PROF/$s.host.o" ] && link "$PROF/$s.host.o" "prog_$s"
            on=$(MAPAL_PTX="$PROF/$s.ptx" "./prog_$s" 2>&1 | tr '\n' ' '); rc=$?
            g=$(python3 value_gate.py "prog_${cell}_x.out" "$(cells_of "$on")" "$K")
            say "splitk_program prog=$s rc=$rc out=[$on] exact=[$ref] launch=[$(cat "$l2")] $g"
            [ "$rc" -eq 0 ] && [ "${g%% *}" = within ] || say "SPLITK PROGRAM FAILED prog=$s"
        done
    done
fi

# S63: the residual MLP's kernels in one process with cuBLAS, cuBLASLt, stock and tuned CUTLASS
# (mlp_bench.cu: Mapal arms bit-exact vs their face's ordered fold, then 30 iterations ABBA),
# twice, the second with the legs reversed. The default face's MLP is its contract face (both
# sites are epilogue groups: no combine), so the `_c` stems stand for it (bundle: `same`).
if [ -x mlp_bench ]; then
    say "-- mlp: residual MLP kernels vs cuBLAS / cuBLASLt / CUTLASS (mlp_bench, reps=$REPS)"
    for k in "$PROF"/pt_rm_m*_d*_h*_c.k; do
        [ -f "$k" ] || continue
        p=$(basename "$k" _c.k); dims=$(sed -nE 's/^pt_rm_m([0-9]+)_d([0-9]+)_h([0-9]+)$/\1 \2 \3/p' <<< "$p")
        [ -n "$dims" ] || continue
        legs=$(cd "$PROF" && ls "${p}"_c.k "${p}"_c.*.k "${p}"_x.k "${p}"_x.*.k 2>/dev/null | sed 's/\.k$//' | tr '\n' ' ')
        for pass in 1 2; do
            rev=; [ $pass = 2 ] && rev=1
            # shellcheck disable=SC2086  # dims and legs are separate arguments
            DIS_REVERSE=$rev ./mlp_bench $dims "$REPS" "$PROF" $legs 2>&1 | sed "s/^/pass=$pass mlp=$p /" | tee -a "$LOG"
        done
    done
fi

# Whole programs, both faces: the shipped emission, each host rung off, and
# the S52 host object (`s52`, PREV), interleaved per repetition. One untraced
# process for the wall time and the program's own `iter ms`, one traced for
# the phase totals; the cuBLAS whole-program floor (`tile_bench --program`).
# E2E=0 (S62): skip this section (a kernel item's session).
[ "${E2E:-1}" = 1 ] && say "-- e2e: host-link arms, process wall + MAPAL_GPU_TRACE totals"
ARMS=${E2E_ARMS:-on off-resident off-devgen off-eager-init off-arena off-arch-rungs s52}
arm_stem() { case $2 in on|rules|pick*) echo "$1" ;; *) echo "$1.$2" ;; esac; }
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
# S64: and every `.on-<rung>` arm (the re-judged rows' flipped arms).
off_arms() { local o; for o in "$PROF/$1".off-*.host.o "$PROF/$1".on-*.host.o; do [ -f "$o" ] && { o=${o%.host.o}; echo "${o##*.}"; }; done; }
# S62: the CPU-only build and the null copy, where the bundle built them.
xtra_arms() { local x; for x in cpu cpu-inline cpu-off-by-ref null c1; do [ -f "$PROF/$1.$x.host.o" ] && echo "$x"; done; }
# probe_out <stem> [arm]: one untraced process's stdout, one line (`iter ms` included).
probe_out() { MAPAL_GPU_PICK=$(sel "${2:-on}") MAPAL_PTX="$PROF/$1.ptx" "./pt_$1" 2>/dev/null | tr '\n' ' '; }
# A trace's site decisions: `key:pick:src` per `op=site` line.
site_picks() { sed -nE 's/.*op=site site=[0-9a-f]+ key=([0-9a-f]+) n=[0-9]+ pick=([0-9]+) src=([a-z]+).*/\1:\2:\3/p' <<< "$1" | tr '\n' ' '; }
module_ms() { grep -m1 mapal_gpu_trace_total <<< "$1" | grep -o 'module_ms=[0-9.]*' | cut -d= -f2; }
val_fail=0
for o in "$PROF"/pt_*.host.o; do
    [ -f "$o" ] || continue
    b=$(basename "$o" .host.o)
    case $b in *.*|*_c|*_x) continue ;; esac
    pa=$( (off_arms "$b"; off_arms "${b}_c"; off_arms "${b}_x"; xtra_arms "$b") | sort -u | tr "\n" " ")
    tabled=; compgen -G "$PROF/$b.sites" > /dev/null || compgen -G "$PROF/${b}_[cx].sites" > /dev/null && tabled=1
    pa=${PROBE_ARMS:-on ${tabled:+rules }$pa}
    probe=${b#pt_}; n=${probe##*_}; probe=${probe%_*}
    # S63 (ADR-0040 D2): the program's longest fold, off the stem (`_d<d>_h<h>`).
    # S64: `tiles/<stem>.kmax` (gen_tiles.py) first.
    kmax=$(cat "tiles/$b.kmax" 2>/dev/null || { sed -nE 's/.*_d([0-9]+)_h([0-9]+).*/\1 \2/p' <<< "$b" | awk '{ print ($1 > $2) ? $1 : $2 }'; })
    faces=("" _c); [ -f "$PROF/${b}_x.host.o" ] && faces=("" _x _c)
    # per-site-choice ladders: a step too large for the CPU oracle carries its exact face built for the device
    # (`.xref`, the interpreter's order bit for bit; `.xabsref` D2's magnitudes): its output is the oracle.
    for x in xref:_x xabsref:_xabs; do
        if [ -f "$PROF/$b.${x%%:*}.host.o" ] && [ ! -s "tiles/$b${x#*:}.expect" ]; then
            link "$PROF/$b.${x%%:*}.host.o" "pt_$b.${x%%:*}"
            MAPAL_PTX="$PROF/$b.${x%%:*}.ptx" "./pt_$b.${x%%:*}" 2>/dev/null | grep -v '^iter ms=' > "tiles/$b${x#*:}.expect"
            say "oracle prog=$b src=device-exact file=$b${x#*:}.expect rc=${PIPESTATUS[0]} out=[$(tr '\n' ' ' < "tiles/$b${x#*:}.expect")]"
        fi
    done
    runs=(); gates=()
    for face in "${faces[@]}"; do
        for arm in $pa; do
            s=$(arm_stem "$b$face" "$arm")
            case $arm in
                rules|pick*) [ -s "$PROF/$b$face.sites" ] || continue; [ -x "pt_$s" ] || link "$PROF/$s.host.o" "pt_$s" ;;
                *) [ -f "$PROF/$s.host.o" ] && link "$PROF/$s.host.o" "pt_$s" ;;
            esac
            [ -x "pt_$s" ] && runs+=("$face:$arm")
        done
        # per-site-choice: the value gate runs every candidate index (timed only under PICKS=1).
        if [ -s "$PROF/$b$face.sites" ] && [ -x "pt_$b$face" ]; then
            for pk in $(picks "$PROF/$b$face"); do
                gates+=("$face:pick$pk"); [ "${PICKS:-0}" = 1 ] && runs+=("$face:pick$pk")
            done
        fi
    done
    gates=(${runs[@]+"${runs[@]}"} ${gates[@]+"${gates[@]}"})
    # The value gate, before any timing: every arm prints its face's oracle. site-geometry: a failing arm
    # drops alone (its VALUE-FAIL line stays); the others are timed (a search bundle's other knob values).
    bad=0; failed=" "
    for fa in ${gates[@]+"${gates[@]}"}; do
        case " $failed " in *" $fa "*) continue ;; esac
        face=${fa%%:*}; arm=${fa#*:}; s=$(arm_stem "$b$face" "$arm")
        got=$(probe_out "$s" "$arm"); got=${got%%iter ms=*}
        # A failing candidate fails the arms that may run it (AUTO and its null) with it.
        fails() { bad=1; failed+="$fa "; case $arm in pick*) failed+="$face:on $face:null " ;; esac; }
        # S63: the default face answers to the exact face within D2's bound (value_gate.py).
        if [ -z "$face" ] && [ -s "tiles/${b}_x.expect" ] && [ -n "$kmax" ]; then
            # S65: against D2's M (absolute) where the bundle has the program's magnitude oracle.
            xabs=(); [ -s "tiles/${b}_xabs.expect" ] && xabs=("tiles/${b}_xabs.expect")
            # ADR-0041 D3: a program whose products admit TF32 answers to D3's bound.
            grep -q ': tf32' "tiles/$b.mapal" 2>/dev/null && xabs+=(--tf32)
            # ADR-0043 D4: a bf16 member answers to D3's bound with u_t = 2⁻⁸.
            grep -q ': bf16' "tiles/$b.mapal" 2>/dev/null && xabs+=(--bf16)
            if ! g=$(python3 value_gate.py "tiles/${b}_x.expect" "$got" "$kmax" ${xabs[@]+"${xabs[@]}"}); then
                fails; say "VALUE-FAIL prog=$b face=_default arm=$arm got=[$got] $g"
            else
                say "value-bound prog=$b face=_default arm=$arm K=$kmax $g got=[$got] exact=[$(tr '\n' ' ' < "tiles/${b}_x.expect")]"
            fi
            continue
        fi
        exp="tiles/$b$face.expect"
        # flow: a default face without its bound is a gate failure, never a bit comparison against
        # another order-free build (the facts-on-graph sweep failed split-K / stream-K that way).
        if [ -z "$face" ] && [ ! -s "$exp" ]; then
            fails; say "VALUE-FAIL prog=$b face=_default arm=$arm got=[$got] no bound (tiles/${b}_x.expect, K)"
            continue
        fi
        if [ -s "$exp" ]; then want=$(tr '\n' ' ' < "$exp"); else want=$(probe_out "$b$face.cpu"); want=${want%%iter ms=*}; fi
        if [ "$got" != "$want" ]; then
            fails; say "VALUE-FAIL prog=$b face=${face:-_exact} arm=$arm got=[$got] want=[$want]"
        fi
    done
    kept=(); for fa in "${runs[@]}"; do [[ "$failed" == *" $fa "* ]] || kept+=("$fa"); done
    if [ "$bad" != 0 ]; then
        val_fail=$((val_fail + 1)); say "value-gate prog=$b arms=${#gates[@]} failed=$(wc -w <<< "$failed") timing the rest"
    else
        say "value-gate prog=$b arms=${#gates[@]} pass"
    fi
    [ ${#kept[@]} -gt 0 ] || continue
    runs=("${kept[@]}")
    # per-site-choice: the tune run (untimed, traced) of each tabled face, and its module load on a cold JIT
    # cache (a fresh CUDA_CACHE_PATH) against this run's (the gates warmed it); `.c1` the same.
    tuned=
    for face in "${faces[@]}"; do
        s=$b$face; [ -s "$PROF/$s.sites" ] && [ -x "pt_$s" ] || continue
        mkdir -p tune
        tr=$(MAPAL_GPU_TUNE=1 MAPAL_GPU_PICK=$([ "${RETUNE:-0}" = 1 ] && echo tune || echo auto) MAPAL_GPU_TRACE=1 \
            MAPAL_PTX="$PROF/$s.ptx" "./pt_$s" 2>&1 >/dev/null); rc=$?
        echo "$tr" > "tune/$s.trace"
        sp=$(site_picks "$tr")
        say "tune prog=$b face=${face:-_default} rc=$rc sites=$(( $(wc -w <<< "$sp") )) trials=$(grep -c 'op=trial ' <<< "$tr") picks=[$sp] module_ms=$(module_ms "$tr")"
        [ "$rc" -eq 0 ] || { say "VALIDATION FAILED (tune $s process exit)"; exit 1; }
        grep -qE ':[1-9][0-9]*:' <<< "$sp" && tuned=1
        for t in "$s" "$s.c1"; do
            if [ -x "pt_$t" ] && { [ -x "pt_$s.c1" ] || [ "${LOAD_ALL:-0}" = 1 ]; }; then
                cold=$(mktemp -d)
                a=$(CUDA_CACHE_PATH=$cold MAPAL_GPU_PICK=rules MAPAL_GPU_TRACE=1 MAPAL_PTX="$PROF/$t.ptx" "./pt_$t" 2>&1 >/dev/null)
                w=$(MAPAL_GPU_PICK=rules MAPAL_GPU_TRACE=1 MAPAL_PTX="$PROF/$t.ptx" "./pt_$t" 2>&1 >/dev/null)
                rm -rf "$cold"
                say "load prog=$b face=${face:-_default} stem=$t ptx_kb=$(( $(wc -c < "$PROF/$t.ptx") / 1024 )) cold_module_ms=$(module_ms "$a") warm_module_ms=$(module_ms "$w")"
            fi
        done
    done
    # LADDER=1: every step is timed (AUTO = RULES where nothing was tuned: the scale table needs the row).
    if [ -n "$tabled" ] && [ -z "$tuned" ] && [ -z "${PROBE_ARMS:-}" ] && [ "${LADDER:-0}" != 1 ] && ! compgen -G "pt_$b*.c1" > /dev/null; then
        say "carried prog=$b (pick = rules everywhere)"; continue
    fi
    mkdir -p vals
    vals=""
    for r in $(seq "${PROBE_REPS:-5}"); do
        order=("${runs[@]}")
        if [ $((r % 2)) = 0 ]; then
            order=(); for ((i = ${#runs[@]} - 1; i >= 0; i--)); do order+=("${runs[$i]}"); done
        fi
        for fa in "${order[@]}"; do
            face=${fa%%:*}; arm=${fa#*:}; s=$(arm_stem "$b$face" "$arm")
            t0=$(date +%s%N)
            got=$(probe_out "$s" "$arm"); rc=$?
            t1=$(date +%s%N)
            tr=""; trace_rc=0
            if [ "$r" -le "${PROBE_TRACE_REPS:-2}" ]; then
                # caching-memory: the traced process also reports the peak device bytes (`mapal_gpu_peak`).
                tr=$(MAPAL_GPU_PICK=$(sel "$arm") MAPAL_GPU_TRACE=1 MAPAL_GPU_PEAK=1 MAPAL_PTX="$PROF/$s.ptx" "./pt_$s" 2>&1 >/dev/null); trace_rc=$?
            fi
            sp=; [ -n "$tr" ] && grep -q 'op=site ' <<< "$tr" && sp=" sites=[$(site_picks "$tr")]"
            say "probe prog=$b face=${face:-_default} arm=$arm rep=$r wall_us=$(( (t1 - t0) / 1000 )) out=[$got] $(grep -m1 mapal_gpu_trace_total <<< "$tr") $(grep -m1 -o 'mapal_gpu_peak bytes=[0-9]*' <<< "$tr") rc=$rc trace_rc=$trace_rc$sp"
            # S54: each launch's kernel time under the trace, in launch order.
            [ -n "$tr" ] && say "probe_kernels prog=$b face=${face:-_default} arm=$arm rep=$r $(grep -o 'op=launch kernel=mapal_tile_[0-9a-f]* ms=[0-9.]*' <<< "$tr" | sed 's/op=launch kernel=mapal_tile_//' | tr '\n' ' ')"
            [ "$rc" -eq 0 ] && [ "$trace_rc" -eq 0 ] || { say "VALIDATION FAILED (probe $s rep=$r process exit)"; exit 1; }
            [ -z "$vals" ] && [ "$face$arm" = on ] && vals=${got%%iter ms=*}
            [ "$face$arm" = on ] && echo "$vals" > "vals/$b.txt"
            # S62 SUBMIT="<arm>...": a third process under MAPAL_GPU_TRACE=2 (no
            # post-launch synchronize), so each launch's ms is its CPU enqueue.
            case " ${SUBMIT:-} " in *" $arm "*)
                got2=$(MAPAL_GPU_PICK=$(sel "$arm") MAPAL_GPU_TRACE=2 MAPAL_PTX="$PROF/$s.ptx" "./pt_$s" 2>submit.err | tr '\n' ' '); submit_rc=$?
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
    # model-scale: a projection's cuBLAS / cuBLASLt legs (gemm_lib.cu; `_tf32`: the TF32 legs), gated on c[0], c[last].
    if [[ $b =~ ^pt_pj_m([0-9]+)_n([0-9]+)_k([0-9]+)(_tf32)?$ ]] && [ -x gemm_lib ]; then
        # shellcheck disable=SC2086  # the values are separate arguments
        ./gemm_lib "$b" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}" "$REPS" \
            ${BASH_REMATCH[4]:+--tf32} $vals 2>&1 | tee -a "$LOG"
    fi
    # shellcheck disable=SC2086  # the values are separate arguments
    case $n in *[!0-9]*) continue ;; esac  # S62 `rmlp_<M>x<d>x<h>`: no library leg
    # S64: only the probes chain_bench.cu builds (the suite's fir / conv / loopmm / rt legs are torch_bench.py's;
    # of the generic-fusion chains, firew_rowsum / ewconvew too).
    case $probe in epi|bias|ew|pro|indep|mlp|fanout|diamond|residual|rowsum|softmax_rows|mlp2|loop|decode_*|attn_*|smlp_*) ;; *) continue ;; esac
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

# S64: the intent check over this run (evidence.py: the bundle's static facts + this box's JIT
# images, `exec`), then the PyTorch legs of every timed probe that printed values.
if [ -f evidence.py ] && [ -f "$PROF/evidence/facts.json" ]; then
    python3 evidence.py check "$PROF" "$LOG" > "evidence-$PROF.tsv" 2> evidence-check.err
    say "evidence: $(tail -1 "evidence-$PROF.tsv") $(head -c 300 evidence-check.err)"
fi
if [ -n "${TORCH_PID:-}" ]; then
    say "-- torch: PyTorch eager + torch.compile legs (torch_bench.py, reps=$REPS)"
    wait "$TORCH_PID"; say "torch install: $(tail -2 torch-install.log | tr '\n' ' ')"
    # cross-arch 60f0f8ce: a box whose TORCH_MIN ends the legs early keeps the rows that need them: the bf16-weight
    # rows (torch bf16 is their only reference) first, then the README rows, the f32 layer, the projections last
    # (their gemm_lib legs are the like-for-like competitor already).
    # TORCH_SKIP=<regex>: programs whose torch / comp legs are another program's (e.g. the bf16-storage twins
    # `_bf16kv` / `_bf16`: every bf16 leg rounds weights, cache and activations to bfloat16, so their bf16 legs are
    # the f32 program's; matrix.py takes them from the twin) are not timed twice.
    tp=$(ls vals 2>/dev/null | sed 's/\.txt$//' | grep -Ev "${TORCH_SKIP:-^\$}" |
        awk '{ k = /_bf16w$/ ? 0 : /^pt_pj_/ ? 3 : /^pt_llm_/ ? 2 : 1; print k, $0 }' |
        sort -s -k1,1n | cut -d' ' -f2 | tr '\n' ' ')
    # shellcheck disable=SC2086  # the stems are separate arguments
    [ -n "$tp" ] && timeout "${TORCH_MIN:-15}m" "${TORCH_PY:-python3}" torch_bench.py "$REPS" vals $tp 2>torch.err | tee -a "$LOG"
    say "torch exit=${PIPESTATUS[0]} $(tail -c 300 torch.err | tr '\n' ' ')"
    # competitors (comp_bench.py): hand Triton, Liger, FlashAttention-2 / -3, FlashInfer on the same stems and the GEMM
    # cells. A rented box builds the source legs first (comp_build.sh, in the foreground: nothing timed runs beside
    # it; COMP_BUILD=0 skips them); a local box's TORCH_PY venv carries every package already. COMP=0 skips all.
    if [ "${COMP:-1}" = 1 ] && [ -f comp_bench.py ]; then
        if [ -z "${TORCH_PY:-}" ] && [ "${COMP_BUILD:-1}" = 1 ] && [ -f comp_build.sh ]; then
            say "-- competitor builds: flash-attn ${FA_TAG:-v2.8.3.post2} (FA2; FA3 on sm_90a), cap ${COMP_BUILD_MIN:-25}m each"
            bash comp_build.sh "$SM" > comp-build.log 2>&1
            say "comp build: $(grep '^comp build' comp-build.log | tr '\n' ' ') $(grep '^comp pip' torch-install.log)"
        fi
        cells=$(for f in "$PROF"/mm_*x*x*.launch; do [ -f "$f" ] && basename "$f" .launch; done | grep -E '^mm_[0-9]+x[0-9]+x[0-9]+$' | tr '\n' ' ')
        say "-- competitors: hand Triton, Liger, FlashAttention-2/3, FlashInfer (comp_bench.py, reps=$REPS)"
        # shellcheck disable=SC2086  # the stems are separate arguments
        timeout "${COMP_MIN:-20}m" "${TORCH_PY:-python3}" comp_bench.py "$REPS" vals $tp $cells 2>comp.err | tee -a "$LOG"
        say "comp exit=${PIPESTATUS[0]} $(grep -v -e POSIX -e features.h -e define -e '^ ' -e '^In file' comp.err | tail -c 300 | tr '\n' ' ')"
    fi
fi
# TIMING=0 (S63): skip the single-arm square-cell section (a cell session runs tile_bench --splitk).
if [ "${TIMING:-1}" = 1 ]; then
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
            # ADR-0040: the unsuffixed module is the default face (order free); the in-order check needs `_x`.
            case $face in _exact) s=mm_${n}_x; flag=() ;; _c) s=mm_${n}_c; flag=(--contract) ;; _tf) s=mm_${n}_tf; flag=(--tf32) ;; esac
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
fi
# per-site-choice: this box's decisions (benches/results/evidence/per-site-choice/wisdom-<arch>.tsv via plancost.py).
if compgen -G "wisdom/wisdom-*.tsv" > /dev/null; then
    cat wisdom/wisdom-*.tsv > "wisdom-$PROF.tsv"
    say "wisdom: $(( $(wc -l < "wisdom-$PROF.tsv") )) decisions, $(awk -F'\t' '$4 != 0' "wisdom-$PROF.tsv" | wc -l | tr -d ' ') not candidate 0; carried $(grep -c '^carried prog=' "$LOG") programs; jit_cache used_kb=$(du -sk "$jc" 2>/dev/null | cut -f1)"
fi
say "== done"
