#!/bin/bash
# disasm-h100-pcie: ONE H100 session on a chosen offer (vast_run.sh rents, runs bundle/box_run.sh,
# fetches results + trace/, destroys on every exit path).  usage: run_part.sh <offer_id>
cd /Volumes/LessComplex/Personal/mapal/.claude/worktrees/wf_a3ec2a1a-376-1 || exit 1
D=/private/tmp/claude-501/-Volumes-LessComplex-Personal-mapal/d48de64b-c24f-4f25-9674-928a7ce97ccb/scratchpad/disasm-h100-pcie
export PROFILE=0 RUN_MIN=40 START_TRIES=60
mkdir -p "$D/run"
echo "[$(date -u +%FT%TZ)] start offer=$1" >> "$D/run/part.log"
bash benches/gpu/vast_run.sh "$1" h100-pcie sm_90a "$D/bundle" "$D/run" 30 < /dev/null >> "$D/run/part.log" 2>&1
echo "[$(date -u +%FT%TZ)] exit $?" >> "$D/run/part.log"
