#!/bin/bash
# Build everything a GPU box needs, on THIS machine (no CUDA here): per profile,
# the emitted PTX and the host object of every program in benches/gpu, the
# x86-64 Linux runtime, the oracle outputs, the harness source and CUTLASS.
#
# usage: benches/gpu/bundle.sh <outdir> <profile:sm[:host]>...
#   e.g. rtx-4090:sm_89 h100-pcie:sm_90a rtx-5090:sm_120 rtx-3090:sm_86 a100:sm_80
# `host` (S62 `hostopt`, default x86-64-v3) is the CPU profile the host code is emitted
# against; host objects are compiled for that ISA level (`-march`), and every timed whole
# program also gets `<stem>.cpu` (the CPU-only build, `--target=<host>`) and `<stem>.null`
# (a copy of the shipped build: the box's null variation).
# ONLY="<glob>..." builds only the programs whose stem matches one of the globs
# (e.g. ONLY="px_* pt_*_2048 mm_4096"); every program by default.
# HOST_RUNGS="<rung>..." narrows the whole programs' arms (default: every host rung).
# NOARMS="<glob>..." builds those whole programs shipped only (both faces, no rung
# arms): the regression programs an item's rungs leave byte-identical.
# DEFAULT="<stem>..." (site-geometry, changed.py env) also builds their default face (`<stem>_d`): the
# differential gates it against the exact oracle within ADR-0040 D2's bound (tiles/<stem>.kmax).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=$1; shift
LLVM="${LLVM:-/opt/homebrew/opt/llvm/bin}"
EMIT="${EMIT:-$ROOT/target/release/examples/emit}"  # S62: another build's emitter (a before arm)
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

# The oracle: the interpreter on the exact face; the CPU build on the contract
# face (per-step fmaf — the same fused step the GPU emits). S63 (ADR-0040): the
# shipped `""` and `_c` builds here are `--exact` and `--order=sequential`, so
# the differential stays bit for bit; the default face is integrated_bundle.sh's. The timed
# probes (`pt_*`) have none: chain_bench.cu gates the libraries against them.
for f in "$OUT"/tiles/*.mapal; do
    b=${f%.mapal}
    keep "$(basename "$b")" || continue
    case $(basename "$b") in
    # S62: the timed programs answer to the CPU build too (`iter ms` dropped).
    pt_*)
        for face in "" _c; do
            [ -s "$b$face.expect" ] && continue
            flags=(--exact); [ "$face" = _c ] && flags=(--order=sequential)
            "$EMIT" "$f" - --rewrite ${flags[@]+"${flags[@]}"} --target=apple-m > "$b$face.cpu.ll"
            clang -O2 "$b$face.cpu.ll" "$ROOT/target/release/libmapal_rt.a" -o "$b$face.cpu" 2>/dev/null
            "$b$face.cpu" | grep -v '^iter ms=' > "$b$face.expect"
        done
        continue ;;
    mm_*|e2e_*|loop_mm_*) continue ;;
    esac
    if [ ! -s "$b.expect" ]; then
        case $(basename "$b") in
        # Too big for the interpreter (1024 decode steps; a 2048² filled site):
        # the CPU build's conformance face is the interpreter bit for bit (the
        # backend differential suite's invariant).
        decode_1024_*|wide_*|deep_*|skinny_*)
            "$EMIT" "$f" - --rewrite --exact --target=apple-m > "$b.cpu.ll"
            clang -O2 "$b.cpu.ll" "$ROOT/target/release/libmapal_rt.a" -o "$b.cpu" 2>/dev/null
            "$b.cpu" > "$b.expect" ;;
        *) "$RUN" "$f" --budget 20000000000 > "$b.expect" ;;
        esac
    fi
    if [ ! -s "${b}_c.expect" ]; then
        "$EMIT" "$f" - --rewrite --order=sequential --target=apple-m > "$b.cpu_c.ll"
        clang -O2 "$b.cpu_c.ll" "$ROOT/target/release/libmapal_rt.a" -o "$b.cpu_c" 2>/dev/null
        "$b.cpu_c" > "${b}_c.expect"
    fi
    # site-geometry: a DEFAULT program's magnitude oracle — every closed-form input by its magnitude, so each
    # printed value of the exact build is its M (gen_tiles.py `mabs`; only a program of sums of products,
    # bias and relu over such inputs: no other subtraction). box_run.sh `_d` gates on D2's absolute bound with it.
    case " ${DEFAULT:-} " in *" $(basename "$b") "*)
        if [ ! -s "${b}_xabs.expect" ] && python3 -c 'import re, sys
s = open(sys.argv[1]).read()
o, n = re.subn(r"-> widen_(f32|f64) -> (\w+); \2 / ([0-9.]+) \}",
              r"-> widen_\1 -> \2; (\2 > 0.0) -> { -true-> \2; -false-> 0.0 - \2; } -> va; va / \3 }", s)
rest = re.sub(r"[^\n]*widen_f(32|64)[^\n]*", "", s)
sys.exit(1 if n == 0 or " - " in rest or "max" in rest or "exp" in rest else open(sys.argv[2], "w").write(o) * 0)' \
                "$f" "${b}_xabs.mapal"; then
            "$EMIT" "${b}_xabs.mapal" - --rewrite --exact --target=apple-m > "${b}_xabs.cpu.ll"
            clang -O2 "${b}_xabs.cpu.ll" "$ROOT/target/release/libmapal_rt.a" -o "${b}_xabs.cpu" 2>/dev/null
            "${b}_xabs.cpu" > "${b}_xabs.expect"
        fi ;;
    esac
done

# The ISA level a host profile describes (S62): its host objects are compiled for it, so
# the vector width and FMA the IR was emitted for lower as written. ponytail: by name — add
# a row with each new host profile.
march() { case $1 in x86-64-v3) echo x86-64-v3 ;; *) echo x86-64 ;; esac; }

# emit_one <src.mapal> <out-stem> <profile> <sm> [emit flags...]: host object,
# PTX, and the launch line (kernel gx gy block smem) of the first site.
# HOST: the host profile; an arm with `hostopt` off is the S52-S54 build (generic, x86-64).
# ADR-0041 D7: the TF32 face is the program's — `<src>` with `: tf32 | f32` on every fold step, written
# beside it; `--precision=tf32` is gone.
tf32_src() { # <src> -> path of its TF32-admitting twin
    local o="${1%.mapal}_tfsrc.mapal"
    sed 's/-> fold {/-> fold: tf32 | f32 {/g' "$1" > "$o"; echo "$o"
}
emit_one() {
    local src=$1 stem=$2 prof=$3 sm=$4 a m; shift 4
    m=$(march "$HOST")
    for a in "$@"; do case $a in --gpu-off=*hostopt*) m=x86-64 ;; esac; done
    rm -f "$stem.dev.ll"
    "$EMIT" "$src" - --rewrite "--target=$prof" "--host=$HOST" "--device-out=$stem.dev.ll" "$@" > "$stem.host.ll"
    clang -O2 -c --target=x86_64-unknown-linux-gnu -march="$m" -Wno-override-module "$stem.host.ll" -o "$stem.host.o"
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
s = ls and re.search(r"; splitk=(\d+)(?: (atomic|streamk))?", ls[0])
# An atomic / stream-K combine has no reduce launch (the next line may be a cell'"'"'s launchv).
x = [] if not s else [s.group(1), s.group(2)] if s.group(2) else [s.group(1), k(ls[1]), *n(ls[1])] if len(ls) > 1 else []
ls and print(k(ls[0]), *n(ls[0]), *x)' \
        > "$stem.launch"
}

# emit_cpu <src.mapal> <out-stem> [emit flags...]: the CPU-only build on the host profile.
emit_cpu() {
    local src=$1 stem=$2; shift 2
    "$EMIT" "$src" - --rewrite "--target=$HOST" "$@" > "$stem.host.ll"
    clang -O2 -c --target=x86_64-unknown-linux-gnu -march="$(march "$HOST")" -Wno-override-module \
        "$stem.host.ll" -o "$stem.host.o"
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
HOST_RUNGS=${HOST_RUNGS:-"async resident devgen eager-init retain-delta arena arch-rungs l2-window raster retain-resident delta-h2d lane-stride rowfold rowfold-layout epilogue devmap cell-law streams stream-gate pinned pdl cell-fill asyncstage preload entry-arena async-h2d stage chain-h2d uniform nest-cell tree-gate retained-fold"}
for spec in "$@"; do
    IFS=: read -r prof sm HOST <<< "$spec"; HOST=${HOST:-x86-64-v3}; d="$OUT/$prof"; mkdir -p "$d"
    for f in "$OUT"/tiles/*.mapal; do
        b=$(basename "$f" .mapal)
        keep "$b" || continue
        emit_one "$f" "$d/$b" "$prof" "$sm" --exact
        emit_one "$f" "$d/${b}_c" "$prof" "$sm" --order=sequential
        case " ${DEFAULT:-} " in *" $b "*) emit_one "$f" "$d/${b}_d" "$prof" "$sm" ;; esac
        # S62: the CPU-only build and the null arm of every timed whole program.
        case $b in pt_*|e2e_*|decode_1024_*)
            emit_cpu "$f" "$d/$b.cpu" --exact
            emit_cpu "$f" "$d/${b}_c.cpu" --order=sequential
            for v in "$b" "${b}_c"; do
                cp "$d/$v.host.o" "$d/$v.null.host.o"
                if [ -f "$d/$v.ptx" ]; then cp "$d/$v.ptx" "$d/$v.null.ptx"; fi
            done ;;
        esac
        case $b in
        mm_*x*x*)
            # S62: a projection-shaped kernel bench (M × N × K), Exact and Contract:
            # every kernel arm, then the shipped kernel again (`null`).
            arms "$f" "$d/${b}_c" "$prof" "$sm" --order=sequential -- $KERNEL_RUNGS
            arms "$f" "$d/$b" "$prof" "$sm" --exact -- $KERNEL_RUNGS
            for v in "$b" "${b}_c"; do
                cp "$d/$v.ptx" "$d/$v.null.ptx"; cp "$d/$v.launch" "$d/$v.null.launch"
            done
            # S62 `splitk` (ADR-0040): the reassociation face on both step faces
            # (`_r` exact steps, `_rc` fma steps), its OFF arm (dropped where it
            # equals ON: S = 1), the box's null (the same PTX twice) and, as a
            # diagnostic, S forced to a quarter, a half and twice the derived one
            # (dropped where it falls back to the derived S). tile_bench --splitk.
            for v in _r _rc; do
                fl=(--no-contract --order=fixed); [ $v = _rc ] && fl=(--order=fixed)
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
            emit_one "$(tf32_src "$f")" "$d/${b}_tf" "$prof" "$sm" --contract
            case $b in mm_1536|mm_3072) ;; *)
                arms "$(tf32_src "$f")" "$d/${b}_tf" "$prof" "$sm" --contract -- $TF32_RUNGS
                arms "$f" "$d/${b}_c" "$prof" "$sm" --order=sequential -- $KERNEL_RUNGS
                arms "$f" "$d/$b" "$prof" "$sm" --exact -- $KERNEL_RUNGS
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
                flags=(--exact); [ "$face" = _c ] && flags=(--order=sequential)
                arms "$f" "$d/$b$face" "$prof" "$sm" ${flags[@]+"${flags[@]}"} -- $HOST_RUNGS
                # retain-delta's measured OFF arm: every site a round trip. S54:
                # the s54-base placement (R5's three rungs off) in the same run.
                case $b in decode_*)
                    emit_one "$f" "$d/$b$face.off-retain-delta+resident" "$prof" "$sm" \
                        ${flags[@]+"${flags[@]}"} --gpu-off=retain-delta,resident
                    emit_one "$f" "$d/$b$face.off-retain-resident+delta-h2d+lane-stride" "$prof" "$sm" \
                        ${flags[@]+"${flags[@]}"} --gpu-off=retain-resident,delta-h2d,lane-stride ;;
                esac
                # S62: `fs` = streams forced open (gate off), the realization the
                # allocation elimination changes (PREV copies an earlier emitter's
                # `fs`); `null` = the shipped probe again, the session's noise floor.
                case $b in pt_*|px_*)
                    emit_one "$f" "$d/$b$face.fs" "$prof" "$sm" ${flags[@]+"${flags[@]}"} \
                        --gpu-on=streams --gpu-off=stream-gate ;;
                esac
                case $b in pt_*)
                    cp "$d/$b$face.host.o" "$d/$b$face.null.host.o"; cp "$d/$b$face.ptx" "$d/$b$face.null.ptx" ;;
                esac
            done
            # S54 `epilogue` on the TF32 face (mma.sync, wgmma): no oracle there (a
            # bound), so the ON arm answers to the OFF arm, bit for bit — the same
            # body over the same accumulators. Programs without a group drop both.
            case $b in px_*)
                emit_one "$(tf32_src "$f")" "$d/${b}_tf" "$prof" "$sm" --contract
                emit_one "$(tf32_src "$f")" "$d/${b}_tf.off-epilogue" "$prof" "$sm" --contract \
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
            # S62: the earlier emitter's `fs` arm (its forced-stream realization).
            pt_*|px_*)
                for v in "$b" "${b}_c"; do
                    [ -f "$from/$prof/$v.fs.host.o" ] || continue
                    cp "$from/$prof/$v.fs.host.o" "$d/$v.$tag.host.o"
                    cp "$from/$prof/$v.fs.ptx" "$d/$v.$tag.ptx"
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
