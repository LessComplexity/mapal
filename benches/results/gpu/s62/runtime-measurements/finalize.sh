#!/bin/bash
# finalize.sh: collect the 20:00 completion sessions, merge each cut session into one file per
# (part, session) with every program's rows from exactly one box, then the tables.
set -eu
S=/private/tmp/claude-501/-Volumes-LessComplex-Personal-mapal/d48de64b-c24f-4f25-9674-928a7ce97ccb/scratchpad/runtime
R=/Volumes/LessComplex/Personal/mapal/.claude/worktrees/wf_3402ad23-67f-9/benches/results/gpu/s62/runtime-measurements
f() { grep -v -E '^(Warning: Permanently added|Welcome to vast|Have fun)' "$1" > "$2"; echo "$2 $(grep -c '' "$2")"; }
src() { # the box's finished file, else the partial copy, else the streamed stdout
    local d=$S/res/$1 p=$2
    if grep -q '^== done' "$d/results-$p.txt" 2>/dev/null; then echo "$d/results-$p.txt"
    else ls -t "$d"/results-$p.partial-*.txt 2>/dev/null | head -1 || echo "$d/run-$p.out"; fi
}
f "$(src rtx-3090-s3 rtx-3090)" "$R/rtx-3090/results-rtx-3090-part3.txt"
cp "$S/res/rtx-3090-s3/attempts-rtx-3090.log" "$R/rtx-3090/attempts-rtx-3090-part3.log"
f "$(src rtx-3090-confirm-s3 rtx-3090)" "$R/rtx-3090/results-rtx-3090-confirm-d.txt"
cp "$S/res/rtx-3090-confirm-s3/attempts-rtx-3090.log" "$R/rtx-3090/attempts-rtx-3090-confirm-d.log"
f "$(src h100-pcie-confirm-s3 h100-pcie)" "$R/h100-pcie/results-h100-sxm-confirm-part4.txt"
cp "$S/res/h100-pcie-confirm-s3/attempts-h100-pcie.log" "$R/h100-pcie/attempts-h100-sxm-confirm-part4.log"

M=$S/merge.py
cd "$R"
python3 "$M" rtx-3090/merged-rtx-3090.txt \
    'rtx-3090/results-rtx-3090-part1.txt:pt_(diamond|indep|mlp2|mlp)_\d+|pt_rg_\w+|pt_rq_(diamond|indep|mlp2)_\d+' \
    'rtx-3090/results-rtx-3090-part2.txt:pt_rq_(mlp_\d+|twin_(1024|2048))' \
    'rtx-3090/results-rtx-3090-part3.txt:pt_\w+'
python3 "$M" rtx-3090/merged-rtx-3090-confirm.txt \
    'rtx-3090/results-rtx-3090-confirm-a.txt:pt_(diamond|indep|mlp2|mlp)_\d+' \
    'rtx-3090/results-rtx-3090-confirm-b.txt:pt_rg_\w+|pt_rq_\w+|pt_rt_indep_1024' \
    'rtx-3090/results-rtx-3090-confirm-d.txt:pt_\w+'
python3 "$M" h100-pcie/merged-h100-sxm-confirm.txt \
    'h100-pcie/results-h100-sxm-confirm-part1.txt:pt_(diamond|indep|mlp2|mlp)_\d+|pt_rg_\w+|pt_rq_(diamond|indep|mlp2|mlp)_\d+' \
    'h100-pcie/results-h100-sxm-confirm-part2.txt:pt_rq_twin_\d+|pt_rt_indep_1024' \
    'h100-pcie/results-h100-sxm-confirm-part4.txt:pt_\w+'
python3 "$S/coverage.py" rtx-3090/merged-rtx-3090.txt rtx-3090/merged-rtx-3090-confirm.txt h100-pcie/merged-h100-sxm-confirm.txt | grep -E 'INCOMPLETE|merged' || true
