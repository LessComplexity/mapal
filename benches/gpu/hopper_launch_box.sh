#!/bin/bash
# hopper-launch: the box half of hopper_launch_bundle.sh (shipped as the bundle's box_run.sh; vast_run.sh runs it as
# `box_run.sh <prof> <sm> <reps>` in /w). Sections, each appending to results-<prof>.txt:
#   1. the part (nvidia-smi, CPU);
#   2. the launch probes (tma_launch.c on both runtimes' flag words: warm / cold, the device busy before the clock);
#   3. every program's arms (bundle `arms`), `reps` repetitions each, the arm order reversed every other repetition
#      (ABBA), `null` the shipped arm's second copy; then one MAPAL_GPU_TRACE=1 run an arm (the totals line);
#   4. the value gate: every arm of the warp-specialized kernel prints the shipped arm's bits, `off` (the S53
#      kernel) within 1e-2 of them;
#   5. CUPTI (ktrace.cpp, KTRACE_API=1) on two programs: each arm's driver calls and kernels;
#   6. the library legs (gemm_lib.cu, TF32: cuBLAS / cuBLASLt, plain and `-fresh`) of the projections, and the
#      kernels they launch (ktrace).
set -uo pipefail
PROF=$1 SM=$2 REPS=${3:-10}
CUDA=${CUDA:-/usr/local/cuda}
LOG=results-$PROF.txt
: > "$LOG"
say() { echo "$*" | tee -a "$LOG"; }
say "== $PROF $SM $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit,persistence_mode --format=csv,noheader | tee -a "$LOG"
lscpu | grep -E 'Model name|^CPU\(s\)' | sed 's/^/lscpu: /' | tee -a "$LOG"
export MAPAL_GPU_TUNE=0
for v in new old; do
    cp "mapal_gpu_$v.c" "rt_$v.c"
    cc -O2 -c "rt_$v.c" -I"$CUDA/include" -o "rt_$v.o" 2>&1 | tail -3 | tee -a "$LOG"
done
link() { cc "$1" "$2" libmapal_rt.a -L"$CUDA/lib64/stubs" -lcuda -lpthread -ldl -lm -o "$3" 2>&1 | tail -3; }

say "-- launch probes (tma_launch.c: flags 9 = eager + preload, 73 = the same with tma-warm OFF; BUSY: a plain kernel before the clock)"
cc -O2 tma_launch.c rt_new.o -I"$CUDA/include" -L"$CUDA/lib64/stubs" -lcuda -lpthread -ldl -lm -o tma_launch 2>&1 | tail -3
for r in 1 2 3 4 5; do
    for fl in 9 73; do
        say "probe rt=new rep=$r $(BUSY=1 MAPAL_GPU_TRACE=2 ./tma_launch $fl 2>&1 | grep -v 'op=' | tr '\n' ' ')"
    done
    say "probe rt=new rep=$r self $(BUSY=1 SELF=1 ./tma_launch 73 2>&1 | tr '\n' ' ')"
done

mkdir -p bin
declare -A ARMS
while read -r p rest; do
    ARMS[$p]="$rest"
    for tok in $rest; do
        case $tok in
        *=*) ;;  # an alias: the same binary as the arm it names
        old:*) link "$PROF/$p.${tok#old:}.host.o" rt_old.o "bin/$p.old" ;;
        *) link "$PROF/$p.$tok.host.o" rt_new.o "bin/$p.$tok" ;;
        esac
    done
done < arms
ptx_of() { # <prog> <arm token>: the PTX the arm loads
    case $2 in old:*) echo "$PROF/$1.${2#old:}.ptx" ;; *) echo "$PROF/$1.$2.ptx" ;; esac
}
name_of() { case $1 in old:*) echo old ;; *) echo "$1" ;; esac; }

say "-- programs: $REPS repetitions, ABBA, null = the shipped arm's second copy"
for p in $(cut -d' ' -f1 arms); do
    timed=()
    for tok in ${ARMS[$p]}; do case $tok in *=*) ;; *) timed+=("$tok") ;; esac; done
    timed+=(null)
    for r in $(seq "$REPS"); do
        order=("${timed[@]}")
        if [ $((r % 2)) = 0 ]; then
            order=(); for ((i = ${#timed[@]} - 1; i >= 0; i--)); do order+=("${timed[$i]}"); done
        fi
        for tok in "${order[@]}"; do
            a=$(name_of "$tok"); bin=$a; pt=$(ptx_of "$p" "$tok")
            [ "$tok" = null ] && { bin=on; pt="$PROF/$p.on.ptx"; }
            out=$(MAPAL_PTX="$pt" "./bin/$p.$bin" 2>/dev/null | tr '\n' ' ')
            say "run prog=$p arm=$a rep=$r out=[$out]"
        done
    done
    for tok in ${ARMS[$p]}; do
        case $tok in *=*) continue ;; esac
        a=$(name_of "$tok")
        tr=$(MAPAL_GPU_TRACE=1 MAPAL_PTX="$(ptx_of "$p" "$tok")" "./bin/$p.$a" 2>&1 >/dev/null |
            grep -E 'mapal_gpu_trace_total|op=launch' | sed 's/mapal_gpu_trace //' | tr '\n' ' ')
        say "trace prog=$p arm=$a $tr"
    done
done

say "-- value gate"
python3 - "$LOG" <<'EOF' | tee -a "$LOG"
import re, sys, collections
vals = collections.defaultdict(lambda: collections.defaultdict(set))
for l in open(sys.argv[1]):
    m = re.match(r'run prog=(\S+) arm=(\S+) rep=\d+ out=\[(.*)\]', l)
    if m:
        v = re.sub(r'iter ms=\s*\S+', '', m.group(3)).split()
        vals[m.group(1)][m.group(2)].add(tuple(v))
bad = 0
for p, arms in vals.items():
    ref = arms.get('on')
    if not ref or len(ref) != 1:
        print(f'gate prog={p} FAIL on prints {len(ref or [])} value sets'); bad += 1; continue
    ref = next(iter(ref))
    for a, sets in arms.items():
        for s in sets:
            if a == 'off':
                dev = max((abs(float(x) - float(y)) / max(1.0, abs(float(y))) for x, y in zip(s, ref)), default=0.0)
                ok = dev <= 1e-2 and len(s) == len(ref) and len(s) > 0
                print(f'gate prog={p} arm=off {"ok" if ok else "FAIL"} maxdev={dev:.3g}')
            else:
                ok = s == ref
                if not ok:
                    print(f'gate prog={p} arm={a} FAIL {s} vs {ref}')
            bad += not ok
print(f'value gate: {bad} failure(s)')
EOF

if [ -f ktrace.cpp ]; then
    say "-- CUPTI: driver calls and kernels per arm (KTRACE_API=1)"
    g++ -O2 -shared -fPIC -DKTRACE_KERNEL=CUpti_ActivityKernel9 -DKTRACE_MEMCPY=CUpti_ActivityMemcpy5 \
        -DKTRACE_MEMSET=CUpti_ActivityMemset4 -I"$CUDA/include" -I"$CUDA/extras/CUPTI/include" ktrace.cpp -o ktrace.so \
        -L"$CUDA/lib64" -L"$CUDA/extras/CUPTI/lib64" -lcupti 2>&1 | tail -3 | tee -a "$LOG"
    export LD_LIBRARY_PATH="$CUDA/extras/CUPTI/lib64:$CUDA/lib64:${LD_LIBRARY_PATH:-}"
    for p in pt_pj_m2048_n4096_k4096_tf32 pt_pj_m128_n4096_k4096_tf32; do
        for tok in ${ARMS[$p]:-}; do
            case $tok in *=*) continue ;; esac
            a=$(name_of "$tok"); rm -f "kt-$p.$a.tsv"
            CUDA_INJECTION64_PATH=$PWD/ktrace.so KTRACE_API=1 KTRACE_OUT=$PWD/kt-$p.$a.tsv \
                MAPAL_PTX="$(ptx_of "$p" "$tok")" "./bin/$p.$a" > /dev/null 2>&1
            # From the last operand generator's start on (the clocked call): the driver calls longer than 5 us,
            # and the kernels (name, grid, block, us).
            say "ktrace prog=$p arm=$a $(awk -F'\t' 'NR == FNR { if ($1 == "K" && $2 ~ /^mapal_gen/ && $8 > g) g = $8; next }
                $1 == "D" && $3 >= g && ($4 - $3) > 5000 { printf "%s=%.0fus ", $2, ($4 - $3) / 1000 }
                $1 == "K" && $8 >= g { printf "K:%s:%s:%s=%.1fus ", substr($2, 1, 28), $3, $4, ($9 - $8) / 1000 }' \
                "kt-$p.$a.tsv" "kt-$p.$a.tsv" 2>/dev/null)"
        done
    done
fi

if [ -f gemm_lib.cu ]; then
    say "-- library legs (gemm_lib.cu --tf32) and their kernels"
    nvcc -O3 -std=c++17 -arch="${SM%a}" gemm_lib.cu -lcublasLt -lcublas -o gemm_lib 2>&1 | grep -v Wno | tail -3 | tee -a "$LOG"
    for p in $(cut -d' ' -f1 arms); do
        [[ $p =~ ^pt_pj_m([0-9]+)_n([0-9]+)_k([0-9]+)_tf32$ ]] || continue
        v=$(grep -m1 "run prog=$p arm=on " "$LOG" | sed -E 's/.*out=\[(.*)iter ms.*/\1/')
        ./gemm_lib "$p" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}" 15 --tf32 $v 2>&1 | tee -a "$LOG"
        if [ -f ktrace.so ]; then
            rm -f "kt-lib-$p.tsv"
            CUDA_INJECTION64_PATH=$PWD/ktrace.so KTRACE_OUT=$PWD/kt-lib-$p.tsv ./gemm_lib "$p" "${BASH_REMATCH[1]}" \
                "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}" 3 --tf32 > /dev/null 2>&1
            say "libkernels prog=$p $(awk -F'\t' '$1 == "K" && $2 !~ /fill|flush|gen/ { print substr($2, 1, 90) "|" $3 "|" $4 "|" $5 "|" $7 }' \
                "kt-lib-$p.tsv" 2>/dev/null | sort | uniq -c | sort -rn | head -4 | tr '\n' ';')"
        fi
    done
fi
# review: the intent check over this run (evidence.py: the bundle's static facts, its rung table) — a run without
# its evidence-<prof>.tsv is not a result (CLAUDE.md); vast_run.sh fetches /w/evidence-<prof>.tsv.
if [ -f evidence.py ] && [ -f "$PROF/evidence/facts.json" ]; then
    python3 evidence.py check "$PROF" "$LOG" > "evidence-$PROF.tsv" 2> evidence-check.err
    say "evidence: $(tail -1 "evidence-$PROF.tsv")"
fi
say "== done"
