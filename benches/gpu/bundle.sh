#!/bin/bash
# Build everything a GPU box needs, on THIS machine (no CUDA here): per profile,
# the emitted PTX and the host object of every program in benches/gpu, the
# x86-64 Linux runtime, the oracle outputs, the harness source and CUTLASS.
#
# usage: benches/gpu/bundle.sh <outdir> <profile:sm>...   e.g. rtx-4090:sm_89 h100:sm_90
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=$1; shift
LLVM="${LLVM:-/opt/homebrew/opt/llvm/bin}"
EMIT="$ROOT/target/release/examples/emit"
RUN="$ROOT/target/release/examples/run"
mkdir -p "$OUT/tiles"
python3 "$ROOT/benches/gpu/gen_tiles.py" "$OUT/tiles" >/dev/null

# The oracle: the interpreter on the conformance face; the CPU build on the
# contract face (per-step fmaf — the same fused step the GPU emits).
for f in "$OUT"/tiles/*.mapal; do
    b=${f%.mapal}
    case $(basename "$b") in mm_*|e2e_*) continue ;; esac
    [ -s "$b.expect" ] || "$RUN" "$f" --budget 20000000000 > "$b.expect"
    if [ ! -s "${b}_c.expect" ]; then
        "$EMIT" "$f" - --rewrite --contract --target=apple-m > "$b.cpu_c.ll"
        clang -O2 "$b.cpu_c.ll" "$ROOT/target/release/libmapal_rt.a" -o "$b.cpu_c" 2>/dev/null
        "$b.cpu_c" > "${b}_c.expect"
    fi
done

# emit_one <src.mapal> <out-stem> <profile> <sm> [emit flags...]: host object,
# PTX, and the launch line (kernel gx gy block smem) the host call passes.
emit_one() {
    local src=$1 stem=$2 prof=$3 sm=$4; shift 4
    rm -f "$stem.dev.ll"
    "$EMIT" "$src" - --rewrite "--target=$prof" "--device-out=$stem.dev.ll" "$@" > "$stem.host.ll"
    clang -O2 -c --target=x86_64-unknown-linux-gnu -Wno-override-module "$stem.host.ll" -o "$stem.host.o"
    [ -f "$stem.dev.ll" ] || return 0
    "$LLVM/opt" -O3 "$stem.dev.ll" -S -o "$stem.dev.opt.ll"
    "$LLVM/llc" -O3 -mcpu="$sm" -nvptx-fma-level=0 "$stem.dev.opt.ll" -o "$stem.ptx"
    grep -o 'call void @mapal_gpu_tile(i64 [-0-9]*.*' "$stem.host.ll" | head -1 |
        python3 -c 'import sys,re; s=sys.stdin.read(); i=int(re.search(r"i64 (-?\d+)",s).group(1)) & (2**64-1); n=re.findall(r"i32 (\d+)",s); print(f"mapal_tile_{i:016x}", *n)' \
        > "$stem.launch"
}

# arms <src> <base-stem> <prof> <sm> <face flags> -- <rung>...: for every rung,
# the kernel with it forced off and forced on (S53). An arm whose kernel and
# launch equal the shipped one is not an arm and is dropped, so what stays is
# exactly the opposite of the profile's decision at that size.
arms() {
    local src=$1 base=$2 prof=$3 sm=$4 flags=() s; shift 4
    while [ "$1" != -- ]; do flags+=("$1"); shift; done; shift
    for rung in "$@"; do
        for dir in off on; do
            s="$base.$dir-$rung"
            emit_one "$src" "$s" "$prof" "$sm" ${flags[@]+"${flags[@]}"} "--gpu-$dir=$rung"
            if cmp -s "$s.ptx" "$base.ptx" && cmp -s "$s.launch" "$base.launch"; then rm -f "$s".*; fi
        done
    done
}

for spec in "$@"; do
    prof=${spec%%:*}; sm=${spec##*:}; d="$OUT/$prof"; mkdir -p "$d"
    for f in "$OUT"/tiles/*.mapal; do
        b=$(basename "$f" .mapal)
        emit_one "$f" "$d/$b" "$prof" "$sm"
        emit_one "$f" "$d/${b}_c" "$prof" "$sm" --contract
        # Measurement arms: the shipped kernel with one rung flipped, on the
        # FMA face, the TF32 face (tensor cores) and the exact face.
        case $b in mm_*)
            emit_one "$f" "$d/${b}_tf" "$prof" "$sm" --contract --precision=tf32
            arms "$f" "$d/${b}_tf" "$prof" "$sm" --contract --precision=tf32 -- wide ldmatrix warp-fill arch-rungs
            arms "$f" "$d/${b}_c" "$prof" "$sm" --contract -- \
                interleave unroll min-blocks warp prefetch async warp-fill arch-rungs
            arms "$f" "$d/$b" "$prof" "$sm" -- async prefetch warp-fill arch-rungs
            # Earlier shipped kernels, for a within-run comparison: PREV="v1=<bundle> v2=<bundle>".
            for prev in ${PREV:-}; do
                tag=${prev%%=*}; from=${prev#*=}
                if [ -f "$from/$prof/${b}_c.ptx" ]; then
                    cp "$from/$prof/${b}_c.ptx" "$d/${b}_c.$tag.ptx"
                    cp "$from/$prof/${b}_c.launch" "$d/${b}_c.$tag.launch"
                fi
            done
            ;;
        esac
    done
done

cp "$ROOT/target/x86_64-unknown-linux-gnu/release/libmapal_rt.a" "$OUT/"
cp "$ROOT/crates/mapal-rt/gpu/mapal_gpu.c" "$ROOT/benches/gpu/tile_bench.cu" "$ROOT/benches/gpu/box_run.sh" \
    "$ROOT/benches/gpu/box_profile.sh" "$OUT/"
if [ ! -d "$OUT/cutlass" ]; then
    git clone -q --depth 1 https://github.com/NVIDIA/cutlass "$OUT/cutlass.git"
    mkdir -p "$OUT/cutlass/tools/util"
    cp -R "$OUT/cutlass.git/include" "$OUT/cutlass/"
    cp -R "$OUT/cutlass.git/tools/util/include" "$OUT/cutlass/tools/util/"
    (cd "$OUT/cutlass.git" && git rev-parse HEAD) > "$OUT/cutlass/COMMIT"
    rm -rf "$OUT/cutlass.git"
fi
echo "bundle ready: $OUT"
