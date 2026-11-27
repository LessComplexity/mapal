#!/bin/bash
# hopper-launch: the H100 bundle for the TMA launch path and the persistent walk — the 12 TF32 programs whose
# `wgmma-ws` kernel won per kernel and lost per call (cross-arch 5fcb401b finding 3), each in the arms
#   on     the shipped build (tma-warm, ws-persist where its price wins)
#   one    --gpu-off=ws-persist (a CTA a tile: the device text before this step)
#   cold   --gpu-off=ws-persist,tma-warm (the recipes from the module text and the map cache, no warm)
#   old    `one`'s objects on the runtime before this step (base 8ff66d34: the 5fcb401b build)
#   order  --gpu-off=l2-gate (the walk in the l2-gate's raster order) where `on` walks
#   rast   --gpu-off=ws-persist,l2-gate (a CTA a tile in that raster order) where `on` walks
#   off    --gpu-off=wgmma-ws (the S53 kernel)
# An arm whose host text and PTX equal an earlier arm's is that arm (`arms` lists the aliases). Then the box
# script (hopper_launch_box.sh as box_run.sh), the runtimes, gemm_lib / ktrace and the launch probes.
# usage: benches/gpu/hopper_launch_bundle.sh <out>   (env: PROGS, LLVM=/usr/bin, CUDA=/opt/cuda, BASE=8ff66d34)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=$1
LLVM=${LLVM:-/usr/bin}
CUDA=${CUDA:-/opt/cuda}
BASE=${BASE:-8ff66d34}
EMIT="$ROOT/target/release/examples/emit"
# The objects sit in a directory named for the part: vast_run.sh uploads only the part's own.
PART=${PART:-h100}
PROGS=${PROGS:-"pt_pj_m128_n4096_k4096_tf32 pt_pj_m128_n6144_k4096_tf32 pt_pj_m128_n28672_k4096_tf32
pt_pj_m128_n4096_k14336_tf32 pt_pj_m2048_n4096_k4096_tf32 pt_pj_m2048_n6144_k4096_tf32 pt_pj_m2048_n28672_k4096_tf32
pt_pj_m2048_n4096_k14336_tf32 pt_mlp_2048_tf32 pt_mlp2_2048_tf32 pt_llm_pre_t2048_tf32 pt_llm_blkpre_t2048_tf32"}
mkdir -p "$OUT/$PART"
: > "$OUT/arms"
build() { # <prog> <arm> [flags]
    local p=$1 arm=$2 s="$OUT/$PART/$1.$2"; shift 2
    "$EMIT" "$ROOT/benches/algorithms/$p.mapal" - --rewrite --target=h100 --host=x86-64-v3 \
        "--device-out=$s.dev.ll" "$@" > "$s.host.ll"
    "$LLVM/opt" -O3 "$s.dev.ll" -S -o "$s.dev.opt.ll"
    "$LLVM/llc" -O3 -mcpu=sm_90a -nvptx-fma-level=0 "$s.dev.opt.ll" -o "$s.ptx"
    "$CUDA/bin/ptxas" -v -arch=sm_90a "$s.ptx" -o "$s.cubin" 2> "$s.ptxas"
    clang -O2 -c --target=x86_64-unknown-linux-gnu -march=x86-64-v3 -Wno-override-module "$s.host.ll" -o "$s.host.o"
    grep -h '^; mapal.plan' "$s.dev.ll" > "$s.plan" || true
    rm -f "$s.dev.opt.ll" "$s.cubin"
}
same() { cmp -s <(grep -v '^; mapal\.' "$OUT/$PART/$1.host.ll") <(grep -v '^; mapal\.' "$OUT/$PART/$2.host.ll") &&
    cmp -s "$OUT/$PART/$1.ptx" "$OUT/$PART/$2.ptx"; }
for p in $PROGS; do
    build "$p" on
    build "$p" one --gpu-off=ws-persist
    build "$p" cold --gpu-off=ws-persist,tma-warm
    build "$p" off --gpu-off=wgmma-ws
    line="$p on one cold old:one off"
    if same "$p.on" "$p.one"; then line="$p on one=on cold old:one off"; fi
    if grep -q ' persist=' "$OUT/$PART/$p.on.plan"; then
        build "$p" order --gpu-off=l2-gate
        same "$p.on" "$p.order" && line="$line order=on" || line="$line order"
        build "$p" rast --gpu-off=ws-persist,l2-gate
        same "$p.one" "$p.rast" && line="$line rast=one" || line="$line rast"
    fi
    echo "$line" >> "$OUT/arms"
    echo "built $p: $(grep -o 'persist=[0-9]* tiles=[0-9]*' "$OUT/$PART/$p.on.plan" | sort | uniq -c | tr '\n' ' ')"
done
cp "$ROOT/crates/mapal-rt/gpu/mapal_gpu.c" "$OUT/mapal_gpu_new.c"
cp "$ROOT/crates/mapal-rt/gpu/mapal_choice.h" "$ROOT"/crates/mapal-rt/gpu/mapal_gpu_*.inc "$OUT/"
git -C "$ROOT" show "$BASE:crates/mapal-rt/gpu/mapal_gpu.c" > "$OUT/mapal_gpu_old.c"
cmp -s <(git -C "$ROOT" show "$BASE:crates/mapal-rt/gpu/mapal_choice.h") "$ROOT/crates/mapal-rt/gpu/mapal_choice.h" ||
    { echo "mapal_choice.h changed since $BASE: the old runtime needs its own"; exit 1; }
cp "$ROOT/target/release/libmapal_rt.a" "$ROOT/benches/gpu/gemm_lib.cu" "$ROOT/benches/local/ktrace.cpp" \
    "$ROOT/benches/local/tma_launch.c" "$OUT/"
cp "$ROOT/benches/gpu/hopper_launch_box.sh" "$OUT/box_run.sh"
# review: the evidence pass's inputs — the part's rung table, ptxas -v / SASS per PTX and the facts (the cubins are
# rebuilt there), with the checker, so the box writes evidence-<part>.tsv (the three hopper-launch H100 runs had none).
"$EMIT" --rungs "--target=$PART" > "$OUT/$PART/rungs.tsv"
python3 "$ROOT/benches/gpu/evidence.py" static "$OUT/$PART" sm_90a
python3 "$ROOT/benches/gpu/evidence.py" facts "$OUT/$PART" sm_90a
cp "$ROOT/benches/gpu/evidence.py" "$ROOT/benches/gpu/intent.py" "$OUT/"
echo "bundle: $(wc -l < "$OUT/arms") programs, $(ls "$OUT/$PART"/*.host.o | wc -l) objects"
