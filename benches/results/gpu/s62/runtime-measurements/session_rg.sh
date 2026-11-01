#!/bin/bash
# session_rg.sh <profile> <sm> <res_subdir> <GPU_NAME>...: the retained-vs-regenerated completion
# (bundle b8): pt_rt_* with arms on / null / rg (the pt_rg_* objects as an interleaved arm) and the
# pt_rg_* programs themselves, on one box. SKIP_HOSTS passes through.
S=/private/tmp/claude-501/-Volumes-LessComplex-Personal-mapal/d48de64b-c24f-4f25-9674-928a7ce97ccb/scratchpad/runtime
W=/Volumes/LessComplex/Personal/mapal/.claude/worktrees/wf_3402ad23-67f-9
PROF=$1 SM=$2 SUB=$3; shift 3
mkdir -p "$S/res/$SUB"
PROFILE=0 SUBMIT="" PROBE_REPS=12 PROBE_ARMS="on null rg" \
    bash "$W/benches/gpu/vast_part.sh" "$PROF" "$SM" "$S/b8" "$S/res/$SUB" 30 "$@" > "$S/res/$SUB/session.log" 2>&1
echo "exit $?" >> "$S/res/$SUB/session.log"
