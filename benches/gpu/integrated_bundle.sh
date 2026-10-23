#!/bin/bash
# S62 integration session bundle: per part, the baseline emitter's build (BASE_EMIT, e.g.
# 4e1102c's `emit`) next to the integrated one, the integrated build with each LOO rung off,
# and a null copy — for box_run.sh's splitk section (every kernel arm of a cell, cuBLAS and
# CUTLASS in one tile_bench process, value gates first, two processes ABBA) and its probe
# section (whole programs, value-gated against the CPU oracle, ABBA per repetition).
#
# usage: BASE_EMIT=<emit> benches/gpu/integrated_bundle.sh <outdir> <profile:sm:loo-rungs>...
#   loo-rungs: comma list of `--gpu-off=` arms (dropped where the text equals the integrated build)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=$1; shift
LLVM="${LLVM:-/opt/homebrew/opt/llvm/bin}"
EMIT="$ROOT/target/release/examples/emit"
CELLS=${CELLS:-"16x2048x7168 16x1024x4096 16x18432x7168 4096x4096x4096"}
PROGS=${PROGS:-"pt_rm_m128_d256_h512 pt_rm_m128_d1024_h2048 pt_pm_2048_32"}
mkdir -p "$OUT/tiles"
python3 "$ROOT/benches/gpu/gen_tiles.py" "$OUT/tiles" >/dev/null
# The whole programs' oracle: the CPU build (the interpreter bit for bit on Exact, per-step
# fmaf on Contract), `iter ms` dropped — bundle.sh's `pt_*` rule.
for p in $PROGS; do
    for face in "" _c; do
        fl=(); [ "$face" = _c ] && fl=(--contract)
        "$EMIT" "$OUT/tiles/$p.mapal" - --rewrite ${fl[@]+"${fl[@]}"} --target=apple-m > "$OUT/tiles/$p$face.cpu.ll"
        clang -O2 -Wno-override-module "$OUT/tiles/$p$face.cpu.ll" "$ROOT/target/release/libmapal_rt.a" -o "$OUT/tiles/$p$face.cpu"
        "$OUT/tiles/$p$face.cpu" | grep -v '^iter ms=' > "$OUT/tiles/$p$face.expect"
    done
done

emit_one() { # <emitter> <src> <stem> <prof> <sm> [flags...]
    local em=$1 src=$2 stem=$3 prof=$4 sm=$5 host=(); shift 5
    [ "$em" = "$EMIT" ] && host=(--host=x86-64-v3)
    rm -f "$stem.dev.ll"
    "$em" "$src" - --rewrite "--target=$prof" ${host[@]+"${host[@]}"} "--device-out=$stem.dev.ll" "$@" > "$stem.host.ll"
    clang -O2 -c --target=x86_64-unknown-linux-gnu -march=x86-64-v3 -Wno-override-module "$stem.host.ll" -o "$stem.host.o"
    [ -f "$stem.dev.ll" ] || return 0
    "$LLVM/opt" -O3 "$stem.dev.ll" -S -o "$stem.dev.opt.ll"
    "$LLVM/llc" -O3 -mcpu="$sm" -nvptx-fma-level=0 "$stem.dev.opt.ll" -o "$stem.ptx"
    { grep -oE 'call void @mapal_gpu_(tile|launch|launch_l2)\(i64 [-0-9]*.*' "$stem.host.ll" || true; } | head -2 |
        python3 -c 'import sys,re
ls = sys.stdin.read().splitlines()
k = lambda l: "mapal_tile_%016x" % (int(re.search(r"i64 (-?\d+)", l).group(1)) & (2**64 - 1))
n = lambda l: re.findall(r"i32 (\d+)", l.split(";")[0])
s = ls and re.search(r"; splitk=(\d+)", ls[0])
ls and print(k(ls[0]), *n(ls[0]), *([s.group(1), k(ls[1]), *n(ls[1])] if s else []))' > "$stem.launch"
}
same() { local x; for x in host.ll ptx launch; do
    if [ -f "$1.$x" ] || [ -f "$2.$x" ]; then cmp -s "$1.$x" "$2.$x" || return 1; fi; done; }
loo() { # <src> <stem> <prof> <sm> <rungs> [flags...]: one OFF arm per rung, dropped where equal
    local src=$1 stem=$2 prof=$3 sm=$4 rungs=$5 r; shift 5
    for r in ${rungs//,/ }; do
        emit_one "$EMIT" "$src" "$stem.off-$r" "$prof" "$sm" "$@" "--gpu-off=$r"
        if same "$stem.off-$r" "$stem"; then rm -f "$stem.off-$r".*; else echo "loo $stem.off-$r"; fi
    done
}

for spec in "$@"; do
    IFS=: read -r prof sm rungs <<< "$spec"; d="$OUT/$prof"; mkdir -p "$d"
    for c in $CELLS; do
        f="$OUT/tiles/mm_$c.mapal"; b="$d/mm_$c"
        for face in "" _c; do
            fl=(); [ "$face" = _c ] && fl=(--contract)
            emit_one "$EMIT" "$f" "$b$face" "$prof" "$sm" ${fl[@]+"${fl[@]}"}
            emit_one "$BASE_EMIT" "$f" "$b$face.base" "$prof" "$sm" ${fl[@]+"${fl[@]}"}
            cp "$b$face.ptx" "$b$face.null.ptx"; cp "$b$face.launch" "$b$face.null.launch"
            loo "$f" "$b$face" "$prof" "$sm" "$rungs" ${fl[@]+"${fl[@]}"}
        done
        for v in _r _rc; do
            fl=(--precision=reassoc); [ $v = _rc ] && fl=(--contract --precision=reassoc)
            emit_one "$EMIT" "$f" "$b$v" "$prof" "$sm" "${fl[@]}"
            loo "$f" "$b$v" "$prof" "$sm" "$rungs" "${fl[@]}"
        done
        # PREV=<bundle>: that bundle's integrated kernels as the `.pre` arm (a confirm run).
        for v in "" _c _r _rc; do
            [ -n "${PREV:-}" ] && [ -f "$PREV/$prof/mm_$c$v.ptx" ] || continue
            cp "$PREV/$prof/mm_$c$v.ptx" "$b$v.pre.ptx"; cp "$PREV/$prof/mm_$c$v.launch" "$b$v.pre.launch"
        done
    done
    for p in $PROGS; do
        f="$OUT/tiles/$p.mapal"; b="$d/$p"
        for face in "" _c; do
            fl=(); [ "$face" = _c ] && fl=(--contract)
            emit_one "$EMIT" "$f" "$b$face" "$prof" "$sm" ${fl[@]+"${fl[@]}"}
            emit_one "$BASE_EMIT" "$f" "$b$face.base" "$prof" "$sm" ${fl[@]+"${fl[@]}"}
            cp "$b$face.host.o" "$b$face.null.host.o"
            [ -f "$b$face.ptx" ] && cp "$b$face.ptx" "$b$face.null.ptx"
            loo "$f" "$b$face" "$prof" "$sm" "$rungs" ${fl[@]+"${fl[@]}"}
            if [ -n "${PREV:-}" ] && [ -f "$PREV/$prof/$p$face.host.o" ]; then
                cp "$PREV/$prof/$p$face.host.o" "$b$face.pre.host.o"
                [ -f "$PREV/$prof/$p$face.ptx" ] && cp "$PREV/$prof/$p$face.ptx" "$b$face.pre.ptx"
            fi
        done
    done
done
cp "$ROOT/target/x86_64-unknown-linux-gnu/release/libmapal_rt.a" "$OUT/"
cp "$ROOT/crates/mapal-rt/gpu/mapal_gpu.c" "$ROOT/benches/gpu/tile_bench.cu" "$ROOT/benches/gpu/chain_bench.cu" \
    "$ROOT/benches/gpu/box_run.sh" "$ROOT/benches/gpu/box_profile.sh" "$OUT/"
if [ ! -d "$OUT/cutlass" ]; then
    git clone -q --depth 1 https://github.com/NVIDIA/cutlass "$OUT/cutlass.git"
    mkdir -p "$OUT/cutlass/tools/util"
    cp -R "$OUT/cutlass.git/include" "$OUT/cutlass/"
    cp -R "$OUT/cutlass.git/tools/util/include" "$OUT/cutlass/tools/util/"
    (cd "$OUT/cutlass.git" && git rev-parse HEAD) > "$OUT/cutlass/COMMIT"
    rm -rf "$OUT/cutlass.git"
fi
echo "bundle ready: $OUT"
