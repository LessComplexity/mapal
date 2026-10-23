#!/bin/bash
# Re-emit the 5090 cells from s62-integrated HEAD (integrated_bundle.sh's emit_one, one part).
set -euo pipefail
W=/Volumes/LessComplex/Personal/mapal/.claude/worktrees/wf_a3ec2a1a-376-1
D=/private/tmp/claude-501/-Volumes-LessComplex-Personal-mapal/d48de64b-c24f-4f25-9674-928a7ce97ccb/scratchpad/disasm-rtx-5090
I=/private/tmp/claude-501/-Volumes-LessComplex-Personal-mapal/d48de64b-c24f-4f25-9674-928a7ce97ccb/scratchpad/integrated/bundle
LLVM=/opt/homebrew/opt/llvm/bin; EMIT=$W/target/release/examples/emit; O=$D/bundle/rtx-5090
emit_one() { # <src> <stem> [flags]
    local src=$1 stem=$2; shift 2
    rm -f "$stem.dev.ll"
    "$EMIT" "$src" - --rewrite --target=rtx-5090 --host=x86-64-v3 "--device-out=$stem.dev.ll" "$@" > "$stem.host.ll"
    clang -O2 -c --target=x86_64-unknown-linux-gnu -march=x86-64-v3 -Wno-override-module "$stem.host.ll" -o "$stem.host.o"
    [ -f "$stem.dev.ll" ] || return 0
    "$LLVM/opt" -O3 "$stem.dev.ll" -S -o "$stem.dev.opt.ll"
    "$LLVM/llc" -O3 -mcpu=sm_120 -nvptx-fma-level=0 "$stem.dev.opt.ll" -o "$stem.ptx"
    { grep -oE 'call void @mapal_gpu_(tile|launch|launch_l2)\(i64 [-0-9]*.*' "$stem.host.ll" || true; } | head -2 |
        python3 -c 'import sys,re
ls = sys.stdin.read().splitlines()
k = lambda l: "mapal_tile_%016x" % (int(re.search(r"i64 (-?\d+)", l).group(1)) & (2**64 - 1))
n = lambda l: re.findall(r"i32 (\d+)", l.split(";")[0])
s = ls and re.search(r"; splitk=(\d+)", ls[0])
ls and print(k(ls[0]), *n(ls[0]), *([s.group(1), k(ls[1]), *n(ls[1])] if s else []))' > "$stem.launch"
}
for c in 16x2048x7168 16x1024x4096 16x18432x7168 128x2048x7168 4096x4096x4096; do
    f=$I/tiles/mm_$c.mapal; cp $f $D/bundle/tiles/
    emit_one $f $O/mm_$c
    emit_one $f $O/mm_${c}_c --contract
    emit_one $f $O/mm_${c}_r --precision=reassoc
    emit_one $f $O/mm_${c}_rc --contract --precision=reassoc
done
for p in pt_rm_m128_d256_h512 pt_rm_m128_d1024_h2048; do
    f=$I/tiles/$p.mapal; cp $f $I/tiles/$p.expect $I/tiles/${p}_c.expect $D/bundle/tiles/
    emit_one $f $O/$p
    emit_one $f $O/${p}_c --contract
    for v in "" _c; do for x in host.o ptx host.ll; do cp $I/rtx-5090/$p$v.base.$x $O/ 2>/dev/null || true; done; done
done
echo emitted
# MLP reassoc faces (S62 ADR-0040 face on whole programs)
for p in pt_rm_m128_d256_h512 pt_rm_m128_d1024_h2048; do
    f=$D/bundle/tiles/$p.mapal
    emit_one $f $O/${p}_r --precision=reassoc
    emit_one $f $O/${p}_rc --contract --precision=reassoc
done
echo emitted-mlp-reassoc
