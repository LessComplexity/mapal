#!/bin/bash
# caching-memory: torch's peak device bytes, one process per (stem, leg) — a leg's peak counted from the bytes the
# process holds before its inputs (torch_bench.py), with nothing an earlier stem or leg left behind.
# usage: torch_peak_proc.sh <bundle> <out.txt> <stem>...   (under the bench lock, CPUs 0-7)
B=$1; O=$2; shift 2
cd "$B" || exit 1
: > "$O"
for s in "$@"; do
    case $s in
        *_llm_kvn_*) modes="naive naive-compile" ;;
        *_llm_kvx_*) modes="eager compile sdpa bf16 bf16-compile" ;;
        pt_llm_*|pt_pj_*) modes="eager compile sdpa bf16 bf16-compile" ;;
        *) modes="eager compile" ;;
    esac
    for m in $modes; do
        flock "$BENCH_LOCK" taskset -c 0-7 env TORCH_MODES=$m "$TORCH_PY" torch_bench.py 3 vals "$s" 2>/dev/null | grep '^torch prog=' >> "$O"
    done
done
