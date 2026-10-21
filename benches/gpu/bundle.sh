#!/bin/bash
# Build everything a GPU box needs, on THIS machine (no CUDA here): per profile,
# the emitted PTX and the host object of every program in benches/gpu, the
# x86-64 Linux runtime, the oracle outputs, the harness source and CUTLASS.
#
# usage: benches/gpu/bundle.sh <outdir> <profile:sm>...
#   e.g. rtx-4090:sm_89 h100-pcie:sm_90a rtx-5090:sm_120 rtx-3090:sm_86 a100:sm_80
# ONLY="<glob>..." builds only the programs whose stem matches one of the globs
# (e.g. ONLY="px_* pt_*_2048 mm_4096"); every program by default.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=$1; shift
LLVM="${LLVM:-/opt/homebrew/opt/llvm/bin}"
EMIT="$ROOT/target/release/examples/emit"
RUN="$ROOT/target/release/examples/run"
mkdir -p "$OUT/tiles"
python3 "$ROOT/benches/gpu/gen_tiles.py" "$OUT/tiles" >/dev/null
read -r -a ONLY_GLOBS <<< "${ONLY:-}"
keep() {
    [ ${#ONLY_GLOBS[@]} -eq 0 ] && return 0
    local g
    for g in "${ONLY_GLOBS[@]}"; do
        # shellcheck disable=SC2254  # the glob is the point
        case $1 in $g) return 0 ;; esac
    done
    return 1
}

# The oracle: the interpreter on the conformance face; the CPU build on the
# contract face (per-step fmaf — the same fused step the GPU emits). The timed
# probes (`pt_*`) have none: chain_bench.cu gates the libraries against them.
for f in "$OUT"/tiles/*.mapal; do
    b=${f%.mapal}
    keep "$(basename "$b")" || continue
    case $(basename "$b") in mm_*|e2e_*|loop_mm_*|pt_*) continue ;; esac
    if [ ! -s "$b.expect" ]; then
        case $(basename "$b") in
        # Too big for the interpreter (1024 decode steps; a 2048² filled site):
        # the CPU build's conformance face is the interpreter bit for bit (the
        # backend differential suite's invariant).
        decode_1024_*|wide_*)
            "$EMIT" "$f" - --rewrite --target=apple-m > "$b.cpu.ll"
            clang -O2 "$b.cpu.ll" "$ROOT/target/release/libmapal_rt.a" -o "$b.cpu" 2>/dev/null
            "$b.cpu" > "$b.expect" ;;
        *) "$RUN" "$f" --budget 20000000000 > "$b.expect" ;;
        esac
    fi
    if [ ! -s "${b}_c.expect" ]; then
        "$EMIT" "$f" - --rewrite --contract --target=apple-m > "$b.cpu_c.ll"
        clang -O2 "$b.cpu_c.ll" "$ROOT/target/release/libmapal_rt.a" -o "$b.cpu_c" 2>/dev/null
        "$b.cpu_c" > "${b}_c.expect"
    fi
done

# emit_one <src.mapal> <out-stem> <profile> <sm> [emit flags...]: host object,
# PTX, and the launch line (kernel gx gy block smem) of the first site.
emit_one() {
    local src=$1 stem=$2 prof=$3 sm=$4; shift 4
    rm -f "$stem.dev.ll"
    "$EMIT" "$src" - --rewrite "--target=$prof" "--device-out=$stem.dev.ll" "$@" > "$stem.host.ll"
    clang -O2 -c --target=x86_64-unknown-linux-gnu -Wno-override-module "$stem.host.ll" -o "$stem.host.o"
    [ -f "$stem.dev.ll" ] || return 0
    "$LLVM/opt" -O3 "$stem.dev.ll" -S -o "$stem.dev.opt.ll"
    "$LLVM/llc" -O3 -mcpu="$sm" -nvptx-fma-level=0 "$stem.dev.opt.ll" -o "$stem.ptx"
    grep -oE 'call void @mapal_gpu_(tile|launch)\(i64 [-0-9]*.*' "$stem.host.ll" | head -1 |
        python3 -c 'import sys,re; s=sys.stdin.read(); i=int(re.search(r"i64 (-?\d+)",s).group(1)) & (2**64-1); n=re.findall(r"i32 (\d+)",s); print(f"mapal_tile_{i:016x}", *n)' \
        > "$stem.launch"
}

# same <stem> <stem>: the two emissions are one program (host, PTX, launch).
same() {
    local x
    for x in host.ll ptx launch; do
        if [ -f "$1.$x" ] || [ -f "$2.$x" ]; then cmp -s "$1.$x" "$2.$x" || return 1; fi
    done
}

# arms <src> <base-stem> <prof> <sm> <face flags> -- <rung>...: for every rung,
# the emission with it forced off and forced on (S53). An arm equal to the
# shipped emission is not an arm and is dropped, so what stays is exactly the
# opposite of the profile's decision at that size. A comma joins rungs (file
# tag `+`).
arms() {
    local src=$1 base=$2 prof=$3 sm=$4 flags=() s; shift 4
    while [ "$1" != -- ]; do flags+=("$1"); shift; done; shift
    for rung in "$@"; do
        for dir in off on; do
            s="$base.$dir-${rung//,/+}"
            emit_one "$src" "$s" "$prof" "$sm" ${flags[@]+"${flags[@]}"} "--gpu-$dir=$rung"
            if same "$s" "$base"; then rm -f "$s".*; fi
        done
    done
}

KERNEL_RUNGS="interleave unroll min-blocks warp prefetch async warp-fill idx32 affine pad wide-fma fma-regs arch-rungs"
for spec in "$@"; do
    prof=${spec%%:*}; sm=${spec##*:}; d="$OUT/$prof"; mkdir -p "$d"
    for f in "$OUT"/tiles/*.mapal; do
        b=$(basename "$f" .mapal)
        keep "$b" || continue
        emit_one "$f" "$d/$b" "$prof" "$sm"
        emit_one "$f" "$d/${b}_c" "$prof" "$sm" --contract
        case $b in
        mm_[0-9]*)
            # The kernel benches: every rung flipped on all three faces; arms
            # at the four sizes the decisions cite, the shipped kernel at all.
            emit_one "$f" "$d/${b}_tf" "$prof" "$sm" --contract --precision=tf32
            case $b in mm_1536|mm_3072) ;; *)
                arms "$f" "$d/${b}_tf" "$prof" "$sm" --contract --precision=tf32 -- \
                    mma wide ldmatrix min-blocks warp-fill wgmma-hopper arch-rungs
                arms "$f" "$d/${b}_c" "$prof" "$sm" --contract -- $KERNEL_RUNGS
                arms "$f" "$d/$b" "$prof" "$sm" -- $KERNEL_RUNGS
                ;;
            esac ;;
        *)
            # Whole programs: both staging paths for the differential, and every
            # host rung off for the differential and the e2e wall-time A/B.
            for face in "" _c; do
                flags=(); [ "$face" = _c ] && flags=(--contract)
                arms "$f" "$d/$b$face" "$prof" "$sm" ${flags[@]+"${flags[@]}"} -- \
                    async resident devgen eager-init retain-delta
                # retain-delta's measured OFF arm: every site a round trip.
                case $b in decode_*)
                    emit_one "$f" "$d/$b$face.off-retain-delta+resident" "$prof" "$sm" \
                        ${flags[@]+"${flags[@]}"} --gpu-off=retain-delta,resident ;;
                esac
            done ;;
        esac
        # Earlier builds, for a within-run comparison: PREV="v1=<bundle> v2=<bundle>".
        # mm_*: the shipped kernels (tile_bench arms). e2e_*: the whole program,
        # host object included (box_run.sh times it next to this build's).
        for prev in ${PREV:-}; do
            tag=${prev%%=*}; from=${prev#*=}
            case $b in
            mm_*)
                for v in "$b" "${b}_c" "${b}_tf"; do
                    [ -f "$from/$prof/$v.ptx" ] || continue
                    cp "$from/$prof/$v.ptx" "$d/$v.$tag.ptx"
                    cp "$from/$prof/$v.launch" "$d/$v.$tag.launch"
                done ;;
            e2e_*)
                for v in "$b" "${b}_c"; do
                    [ -f "$from/$prof/$v.host.o" ] || continue
                    cp "$from/$prof/$v.host.o" "$d/$v.$tag.host.o"
                    cp "$from/$prof/$v.ptx" "$d/$v.$tag.ptx"
                done ;;
            esac
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
