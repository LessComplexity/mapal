#!/bin/bash
# Build everything a GPU box needs, on THIS machine (no CUDA here): per profile,
# the emitted PTX and the host object of every program in benches/gpu, the
# x86-64 Linux runtime, the oracle outputs, the harness source and CUTLASS.
#
# usage: benches/gpu/bundle.sh <outdir> <profile:sm>...
#   e.g. rtx-4090:sm_89 h100-pcie:sm_90a rtx-5090:sm_120 rtx-3090:sm_86 a100:sm_80
# ONLY="<glob>..." builds only the programs whose stem matches one of the globs
# (e.g. ONLY="px_* pt_*_2048 mm_4096"); every program by default.
# HOST_RUNGS="<rung>..." narrows the whole programs' arms (default: every host rung).
# NOARMS="<glob>..." builds those whole programs shipped only (both faces, no rung
# arms): the regression programs an item's rungs leave byte-identical.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=$1; shift
LLVM="${LLVM:-/opt/homebrew/opt/llvm/bin}"
EMIT="$ROOT/target/release/examples/emit"
RUN="$ROOT/target/release/examples/run"
mkdir -p "$OUT/tiles"
python3 "$ROOT/benches/gpu/gen_tiles.py" "$OUT/tiles" >/dev/null
read -r -a ONLY_GLOBS <<< "${ONLY:-}"
read -r -a NOARMS_GLOBS <<< "${NOARMS:-}"
matches() { # <stem> <glob>...
    local s=$1 g; shift
    for g in "$@"; do
        # shellcheck disable=SC2254  # the glob is the point
        case $s in $g) return 0 ;; esac
    done
    return 1
}
keep() {
    [ ${#ONLY_GLOBS[@]} -eq 0 ] && return 0
    matches "$1" "${ONLY_GLOBS[@]}"
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
        decode_1024_*|wide_*|deep_*|skinny_*)
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
    # (A program whose every site has an epilogue launches only `launchv`: no line.)
    # S62 `splitk`: a split site's launch ends `; splitk=S` and the reduce launch
    # follows: `kernel gx gy block smem S rkernel rgx rgy rblock rsmem`.
    { grep -oE 'call void @mapal_gpu_(tile|launch|launch_l2)\(i64 [-0-9]*.*' "$stem.host.ll" || true; } | head -2 |
        python3 -c 'import sys,re
ls = sys.stdin.read().splitlines()
k = lambda l: "mapal_tile_%016x" % (int(re.search(r"i64 (-?\d+)", l).group(1)) & (2**64 - 1))
n = lambda l: re.findall(r"i32 (\d+)", l.split(";")[0])
s = ls and re.search(r"; splitk=(\d+)", ls[0])
ls and print(k(ls[0]), *n(ls[0]), *([s.group(1), k(ls[1]), *n(ls[1])] if s else []))' \
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

# The arms per program kind; an item's bundle narrows them to its own rungs
# (e.g. KERNEL_RUNGS="raster l2-gate" TF32_RUNGS="raster l2-gate" HOST_RUNGS="l2-window raster").
KERNEL_RUNGS=${KERNEL_RUNGS:-"interleave unroll min-blocks warp prefetch async warp-fill idx32 affine pad wide-fma fma-regs arch-rungs raster l2-gate cell-fill asyncstage"}
TF32_RUNGS=${TF32_RUNGS:-"mma wide ldmatrix min-blocks warp-fill wgmma-hopper arch-rungs raster l2-gate"}
HOST_RUNGS=${HOST_RUNGS:-"async resident devgen eager-init retain-delta arena arch-rungs l2-window raster retain-resident delta-h2d lane-stride rowfold rowfold-layout epilogue devmap cell-law streams stream-gate pinned pdl cell-fill asyncstage"}
for spec in "$@"; do
    prof=${spec%%:*}; sm=${spec##*:}; d="$OUT/$prof"; mkdir -p "$d"
    for f in "$OUT"/tiles/*.mapal; do
        b=$(basename "$f" .mapal)
        keep "$b" || continue
        emit_one "$f" "$d/$b" "$prof" "$sm"
        emit_one "$f" "$d/${b}_c" "$prof" "$sm" --contract
        case $b in
        mm_*x*x*)
            # S62: a projection-shaped kernel bench (M × N × K), Exact and Contract:
            # every kernel arm, then the shipped kernel again (`null`).
            arms "$f" "$d/${b}_c" "$prof" "$sm" --contract -- $KERNEL_RUNGS
            arms "$f" "$d/$b" "$prof" "$sm" -- $KERNEL_RUNGS
            for v in "$b" "${b}_c"; do
                cp "$d/$v.ptx" "$d/$v.null.ptx"; cp "$d/$v.launch" "$d/$v.null.launch"
            done
            # S62 `splitk` (ADR-0040): the reassociation face on both step faces
            # (`_r` exact steps, `_rc` fma steps), its OFF arm (dropped where it
            # equals ON: S = 1), the box's null (the same PTX twice) and, as a
            # diagnostic, S forced to a quarter, a half and twice the derived one
            # (dropped where it falls back to the derived S). tile_bench --splitk.
            for v in _r _rc; do
                fl=(--precision=reassoc); [ $v = _rc ] && fl=(--contract --precision=reassoc)
                emit_one "$f" "$d/$b$v" "$prof" "$sm" "${fl[@]}"
                arms "$f" "$d/$b$v" "$prof" "$sm" "${fl[@]}" -- splitk
                cp "$d/$b$v.ptx" "$d/$b$v.null.ptx"; cp "$d/$b$v.launch" "$d/$b$v.null.launch"
                s=$(awk '{print $6}' "$d/$b$v.launch")
                for x in ${s:+$((s / 4)) $((s / 2)) $((s * 2))}; do
                    [ "$x" -ge 2 ] || continue
                    emit_one "$f" "$d/$b$v.s$x" "$prof" "$sm" "${fl[@]}" "--splitk=$x"
                    if same "$d/$b$v.s$x" "$d/$b$v"; then rm -f "$d/$b$v.s$x".*; fi
                done
            done ;;
        mm_[0-9]*)
            # The kernel benches: every rung flipped on all three faces; arms
            # at the four sizes the decisions cite, the shipped kernel at all.
            emit_one "$f" "$d/${b}_tf" "$prof" "$sm" --contract --precision=tf32
            case $b in mm_1536|mm_3072) ;; *)
                arms "$f" "$d/${b}_tf" "$prof" "$sm" --contract --precision=tf32 -- $TF32_RUNGS
                arms "$f" "$d/${b}_c" "$prof" "$sm" --contract -- $KERNEL_RUNGS
                arms "$f" "$d/$b" "$prof" "$sm" -- $KERNEL_RUNGS
                # The shipped kernel again as an arm: one A/B's noise floor in
                # this session, bench position included (rungs.py `null_ab`).
                for v in "$b" "${b}_c" "${b}_tf"; do
                    cp "$d/$v.ptx" "$d/$v.null.ptx"; cp "$d/$v.launch" "$d/$v.null.launch"
                done
                ;;
            esac ;;
        *)
            # Whole programs: both staging paths for the differential, every
            # host rung off for the differential and the e2e wall-time A/B, and
            # the part's whole rung table off (`arch-rungs`, S54 arch-select:
            # what the per-part selection is worth on a multi-computation program).
            [ ${#NOARMS_GLOBS[@]} -gt 0 ] && matches "$b" "${NOARMS_GLOBS[@]}" && continue
            for face in "" _c; do
                flags=(); [ "$face" = _c ] && flags=(--contract)
                arms "$f" "$d/$b$face" "$prof" "$sm" ${flags[@]+"${flags[@]}"} -- $HOST_RUNGS
                # retain-delta's measured OFF arm: every site a round trip. S54:
                # the s54-base placement (R5's three rungs off) in the same run.
                case $b in decode_*)
                    emit_one "$f" "$d/$b$face.off-retain-delta+resident" "$prof" "$sm" \
                        ${flags[@]+"${flags[@]}"} --gpu-off=retain-delta,resident
                    emit_one "$f" "$d/$b$face.off-retain-resident+delta-h2d+lane-stride" "$prof" "$sm" \
                        ${flags[@]+"${flags[@]}"} --gpu-off=retain-resident,delta-h2d,lane-stride ;;
                esac
            done
            # S54 `epilogue` on the TF32 face (mma.sync, wgmma): no oracle there (a
            # bound), so the ON arm answers to the OFF arm, bit for bit — the same
            # body over the same accumulators. Programs without a group drop both.
            case $b in px_*)
                emit_one "$f" "$d/${b}_tf" "$prof" "$sm" --contract --precision=tf32
                emit_one "$f" "$d/${b}_tf.off-epilogue" "$prof" "$sm" --contract --precision=tf32 \
                    --gpu-off=epilogue
                if same "$d/${b}_tf.off-epilogue" "$d/${b}_tf"; then rm -f "$d/${b}_tf".*; fi ;;
            esac ;;
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
# PROBE_ARMS="on off-<rung>...": the probe arms box_run.sh times (its default otherwise).
if [ -n "${PROBE_ARMS:-}" ]; then echo "$PROBE_ARMS" > "$OUT/probe_arms"; fi
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
