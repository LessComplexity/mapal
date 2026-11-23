#!/bin/bash
# S63 default-float session bundle (ADR-0040), S64 evidence pass: per part, every face of every cell and
# whole program, the default face's rung arms, and a null copy — for box_run.sh's splitk section (every
# kernel arm of a cell, cuBLAS, stock and tuned CUTLASS in one tile_bench process, value gates first, two
# processes ABBA), its MLP section (mlp_bench: the programs' kernels vs cuBLAS, cuBLASLt, CUTLASS) and its
# probe section (whole programs, value-gated, ABBA per repetition) — and, per part, the evidence
# (evidence.py: ptxas -v, SASS, LLVM remarks, the intent check) in `<part>/evidence/`.
#
# per-site-choice B5: a program with site tables is one binary whose arms box_run.sh picks by environment (AUTO,
# RULES, each candidate); `.sites` lists its candidates, `.pick<i>.launch` the first table's for tile_bench, `.c1`
# a looped program's `--cands=1` build (the overhead arm); a rung arm the tables cover is dropped (`= table`).
#
# usage: benches/gpu/integrated_bundle.sh <outdir> <profile:sm>...
# Faces (stem suffix): "" default (no flags: fma, order free), _x --exact, _c --order=sequential
# (contract), _r --no-contract --order=fixed (the fixed test/debug face), _rc --order=fixed.
# Default-face arms: .off-streamk, .off-streamk+splitk-atomic, .off-streamk+splitk-atomic+splitk
# (each dropped where it equals the arm above it: the rung does not change that launch), .null.
# S64: SUITE="<pt stem>..." whole programs on the default face only, each with `.cpu` (the CPU-only
# build on the host profile), `.off-place` (GPU-only: every capable site on the device; dropped where
# it equals the shipped build) and `.null`; ARMS="<profile>:<stem>:<face>:<rung>[,<rung>]:<on|off>..."
# extra arms (`<stem><face>.<on|off>-<rung>`, dropped where equal to the shipped emission) — the re-judged
# rows' cells (evidence.py rejudge prints them). site-geometry: SUITE_C="<stem>.." also builds those suite programs'
# contract face (`_c`, its oracle too); an ARMS entry `<profile>:<stem>:<face>:<knob>=<value>:geom` is the arm
# `--geom=<knob>=<value>` (`<stem><face>.on-g-<knob>-<value>`, `.` -> `p`: search.py's candidates).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=$1; shift
LLVM="${LLVM:-/opt/homebrew/opt/llvm/bin}"
# The exact-face CPU oracle's profile: the bundling host (apple-m on the Mac; CPU_TARGET=x86-64-v3 on an x86-64 host).
CPU_TARGET="${CPU_TARGET:-apple-m}"
EMIT="${EMIT:-$ROOT/target/release/examples/emit}"
MAIN_EMIT=$EMIT
CELLS=${CELLS-"16x2048x7168 16x1024x4096 16x18432x7168 128x2048x7168 4096x4096x4096"}
PROGS=${PROGS-"pt_rm_m128_d256_h512 pt_rm_m128_d1024_h2048"}
SUITE=${SUITE-""}
SUITE_C=${SUITE_C-""}
ARMS=${ARMS-""}
mkdir -p "$OUT/tiles"
# Incremental bundles: a stem's built objects (host object, optimized device IR, PTX and their remarks) and a
# CPU oracle's printed values (`.expect`; a hit leaves no `tiles/*.cpu` executable — an intermediate nothing reads,
# excluded from rentals) are content-addressed under BUNDLE_CACHE (default target/bundle-cache;
# BUNDLE_CACHE=0 builds everything) by the emitted IR text itself, the CPU / SM, the runtime library (oracles)
# and a toolchain key (clang, opt, llc and this script, whose text holds every compile step's flags); ptxas -v
# and SASS per distinct PTX likewise (evidence.py static, EVIDENCE_CACHE). An emission whose text is unchanged
# takes the bytes its compile steps made before; any change to the text, the tools or the flags is a new key.
# The emitter always runs: its text is the key.
BUNDLE_CACHE=${BUNDLE_CACHE-$ROOT/target/bundle-cache}
TOOLS_KEY="" RT_KEY=""
[ "$BUNDLE_CACHE" = 0 ] || TOOLS_KEY=$({ clang --version; "$LLVM/opt" --version; "$LLVM/llc" --version
    cat "${BASH_SOURCE[0]}"; } 2>&1 | sha256sum | cut -c1-16)
# evidence.py static: each distinct PTX's ptxas -v and SASS by the PTX's bytes, the SM and the ptxas release.
[ "$BUNDLE_CACHE" = 0 ] || export EVIDENCE_CACHE=${EVIDENCE_CACHE:-$BUNDLE_CACHE/evidence}
# The CPU oracles' runtime library, by content.
[ "$BUNDLE_CACHE" = 0 ] || RT_KEY=$(cat "$ROOT/target/release/libmapal_rt.a" 2>/dev/null | sha256sum | cut -c1-16)
OBJS=(host.o host.remarks.yaml dev.opt.ll dev.opt.remarks.yaml ptx dev.llc.remarks.yaml)
cache_key() { # <kind> <salt> <files...>: the entry of these inputs' text (not their paths); "" when off
    [ "$BUNDLE_CACHE" = 0 ] && return 0
    local k=$1 salt=$2 f; shift 2
    echo "$BUNDLE_CACHE/$k/$( { echo "$TOOLS_KEY $salt"; for f in "$@"; do
        if [ -f "$f" ]; then echo "== file"; cat "$f"; else echo "== none"; fi; done; } | sha256sum | cut -c1-32)"
}
cache_put() { # <entry> <dir> <prefix> <names...>: store <dir>/<prefix><name> that exist (atomic rename)
    local c=$1 dir=$2 pre=$3 x; shift 3
    [ -n "$c" ] || return 0
    mkdir -p "$c.$$"
    for x in "$@"; do [ -f "$dir/$pre$x" ] && cp "$dir/$pre$x" "$c.$$/$x"; done
    touch "$c.$$/ok"; mkdir -p "$(dirname "$c")"; mv -T "$c.$$" "$c" 2>/dev/null || rm -rf "$c.$$"
}
cache_get() { # <entry> <dir> <prefix>: copy a complete entry's files out; 1 when absent
    local c=$1 x
    [ -n "$c" ] && [ -f "$c/ok" ] || return 1
    for x in "$c"/*; do case ${x##*/} in ok | stem) ;; *) cp "$x" "$2/$3${x##*/}" ;; esac; done
    echo "$3" >> "$OUT/.bundle-cache-hits"
}
# per-site-choice ladders: LADDER=1 adds the scale-ladder programs (gen_tiles.py --ladder, ladder_bundle.sh).
# XREF="<stem>..": suite programs too large for the exact CPU oracle — no `.expect` here; `<stem>.xref` (the exact
# face on the device: the interpreter's order bit for bit; `.xabsref` its D2 magnitude program) writes it on the box.
XREF=${XREF-""}
python3 "$ROOT/benches/gpu/gen_tiles.py" "$OUT/tiles" ${LADDER:+--ladder} >/dev/null
# S64: the CPU build of a looped program (pt_rt_mlp2_1024) needs more than an 8 MB stack (rc 139 at
# the default); macOS's hard limit is 64 MB.
ulimit -s 65520 2>/dev/null || true
# The whole programs' oracles: the CPU build on the exact face (the interpreter bit for bit) and
# on the contract face (per-step fmaf), `iter ms` dropped. The default face answers to the exact
# one within ADR-0040 D2's bound (value_gate.py).
for p in $PROGS $SUITE; do
    [[ " $XREF " == *" $p "* ]] && continue
    # The suite runs the default face alone: its gate is the exact face's oracle.
    # (and the contract face where a re-judge arm runs on it).
    faces=(_x _c); [[ " $SUITE " == *" $p "* ]] && [[ " $ARMS " != *":$p:_c:"* ]] && [[ " $SUITE_C " != *" $p "* ]] && faces=(_x)
    for face in "${faces[@]}"; do
        [ -s "$OUT/tiles/$p$face.expect" ] && continue
        fl=(--exact); [ "$face" = _c ] && fl=(--order=sequential)
        "$EMIT" "$OUT/tiles/$p.mapal" - --rewrite "${fl[@]}" "--target=$CPU_TARGET" > "$OUT/tiles/$p$face.cpu.ll"
        c=$(cache_key expect "$RT_KEY" "$OUT/tiles/$p$face.cpu.ll")
        cache_get "$c" "$OUT/tiles" "$p$face." && continue
        clang -O2 -Wno-override-module "$OUT/tiles/$p$face.cpu.ll" "$ROOT/target/release/libmapal_rt.a" -o "$OUT/tiles/$p$face.cpu"
        "$OUT/tiles/$p$face.cpu" | grep -v '^iter ms=' > "$OUT/tiles/$p$face.expect"
        cache_put "$c" "$OUT/tiles" "$p$face." expect
    done
    # S65: ADR-0040 D2's M per printed value (gen_tiles.py `xabs`), where the program has one.
    if [ -f "$OUT/tiles/${p}_xabs.mapal" ] && [ ! -s "$OUT/tiles/${p}_xabs.expect" ]; then
        "$EMIT" "$OUT/tiles/${p}_xabs.mapal" - --rewrite --exact "--target=$CPU_TARGET" > "$OUT/tiles/${p}_xabs.cpu.ll"
        clang -O2 -Wno-override-module "$OUT/tiles/${p}_xabs.cpu.ll" "$ROOT/target/release/libmapal_rt.a" -o "$OUT/tiles/${p}_xabs.cpu"
        "$OUT/tiles/${p}_xabs.cpu" | grep -v '^iter ms=' > "$OUT/tiles/${p}_xabs.expect"
    fi
done

# model-scale: PX="<px stem>..." small differentials (gen_tiles.py `px_llm_*`): the interpreter is the oracle
# (`.expect`); box_run.sh's differential compares the exact face bit for bit and the default face (`_d`) within
# ADR-0040 D2's bound (`.kmax`), each also GPU-only (`.off-place`: at this size the shipped placement keeps
# everything on the host, so the arm is what runs the device kernels of the full-scale program's shape).
PX=${PX-""}
for p in $PX; do
    [ -s "$OUT/tiles/$p.expect" ] || "$ROOT/target/release/examples/run" "$OUT/tiles/$p.mapal" > "$OUT/tiles/$p.expect"
done

# ADR-0041 D7: the TF32 face is the program's — `<src>` with `: tf32 | f32` on every fold step, written
# beside it; `--precision=tf32` is gone.
tf32_src() { # <src> -> path of its TF32-admitting twin
    local o="${1%.mapal}_tfsrc.mapal"
    sed 's/-> fold {/-> fold: tf32 | f32 {/g' "$1" > "$o"; echo "$o"
}
flags_of() { # <face>
    case $1 in _x) echo --exact ;; _c) echo --order=sequential ;; _r) echo --no-contract --order=fixed ;;
    _rc) echo --order=fixed ;; _tf) echo --contract ;; *) ;; esac
}
# S64: every compile step leaves LLVM's optimization remarks beside its output (evidence.py).
REMARK_PASSES='loop-vectorize|loop-unroll|slp-vectorizer|inline|sroa|licm'
host_cc() { # <stem> <march>
    clang -O2 -c --target=x86_64-unknown-linux-gnu -march="$2" -Wno-override-module "$1.host.ll" -o "$1.host.o" \
        -fsave-optimization-record=yaml -foptimization-record-file="$1.host.remarks.yaml" \
        "-foptimization-record-passes=$REMARK_PASSES"
}
emit_one() { # <src> <stem> <prof> <sm> [flags...]
    local src=$1 stem=$2 prof=$3 sm=$4 m=x86-64-v3 a; shift 4
    for a in "$@"; do case $a in --gpu-off=*hostopt*) m=x86-64 ;; esac; done
    rm -f "$stem.dev.ll"
    "$EMIT" "$src" - --rewrite "--target=$prof" --host=x86-64-v3 "--device-out=$stem.dev.ll" "$@" > "$stem.host.ll"
    local c; c=$(cache_key obj "$m $sm" "$stem.host.ll" "$stem.dev.ll")
    if cache_get "$c" "$(dirname "$stem")" "$(basename "$stem")."; then
        # `opt -S` names its input file (`; ModuleID`, `source_filename`): this stem's, as a build here writes it.
        [ -f "$stem.dev.opt.ll" ] && python3 -c 'import sys; p, o, n = sys.argv[1:]; t = open(p).read().split("\n", 2)
t[:2] = [l.replace(o, n) for l in t[:2]]; open(p, "w").write("\n".join(t))' "$stem.dev.opt.ll" "$(cat "$c/stem").dev.ll" "$stem.dev.ll"
    else
        host_cc "$stem" "$m"
        if [ -f "$stem.dev.ll" ]; then
            "$LLVM/opt" -O3 "$stem.dev.ll" -S -o "$stem.dev.opt.ll" -pass-remarks-output="$stem.dev.opt.remarks.yaml" \
                -pass-remarks-filter='loop-unroll|sroa|licm|gvn|instcombine|loop-vectorize'
            "$LLVM/llc" -O3 -mcpu="$sm" -nvptx-fma-level=0 "$stem.dev.opt.ll" -o "$stem.ptx" \
                -pass-remarks-output="$stem.dev.llc.remarks.yaml"
        fi
        echo "$stem" > "$stem.stem"
        cache_put "$c" "$(dirname "$stem")" "$(basename "$stem")." "${OBJS[@]}" stem
        rm -f "$stem.stem"
    fi
    [ -f "$stem.dev.ll" ] || return 0
    # `kernel gx gy block smem`, then `S rkernel rgx rgy rblock rsmem` (workspace split),
    # `S atomic` (splitk-atomic) or `1 streamk` (stream-K). per-site-choice: a site table's candidate 0 when the
    # table comes first; `.sites` (every table's candidates) and `.pick<i>.launch` (the first table's candidate i,
    # its `.ptx` a link to this module: tile_bench's per-candidate legs).
    python3 "$ROOT/benches/gpu/evidence.py" launch "$stem"
}
# The CPU-only build on the host profile (S62 `emit_cpu`).
emit_cpu() { # <src> <stem> [flags...]
    local src=$1 stem=$2; shift 2
    "$EMIT" "$src" - --rewrite --target=x86-64-v3 "$@" > "$stem.host.ll"
    host_cc "$stem" x86-64-v3
}
# Two emissions are one program when host IR (less the `; mapal.` evidence comments), PTX and launch agree.
same() { local x; for x in host.ll ptx launch; do
    if [ -f "$1.$x" ] || [ -f "$2.$x" ]; then
        cmp -s <(grep -sv '^; mapal\.' "$1.$x") <(grep -sv '^; mapal\.' "$2.$x") || return 1
    fi; done; }
drop() { rm -f "$1".host.* "$1".dev.* "$1".ptx "$1".launch "$1".k "$1".sites "$1".pick*.launch "$1".pick*.ptx; }
# per-site-choice B5 overhead arm: a looped program with site tables also as its `--cands=1` build (`.c1`):
# box_run.sh times its RULES arm against it (the cost of going through mapal_gpu_site).
c1() { # <src> <stem> <prof> <sm>
    [ -s "$2.sites" ] && grep -q 'loop {' "$1" || return 0
    emit_one "$1" "$2.c1" "$3" "$4" --cands=1
}
# The default face's realization chain: each arm turns the next rung off; one equal to the arm
# above it (the rung leaves that launch alone) is dropped.
chain() { # <src> <stem> <prof> <sm>
    local src=$1 stem=$2 prof=$3 sm=$4 prev=$2 off="" r
    for r in streamk splitk-atomic splitk; do
        off=${off:+$off,}$r
        emit_one "$src" "$stem.off-${off//,/+}" "$prof" "$sm" "--gpu-off=$off"
        if same "$stem.off-${off//,/+}" "$prev"; then drop "$stem.off-${off//,/+}"
        else prev="$stem.off-${off//,/+}"; echo "arm $prev $(cat "$prev.launch" 2>/dev/null)"; fi
    done
}

for spec in "$@"; do
    IFS=: read -r prof sm <<< "$spec"; d="$OUT/$prof"; mkdir -p "$d"
    for c in $CELLS; do
        f="$OUT/tiles/mm_$c.mapal"; b="$d/mm_$c"
        # (`_r` / `_rc`: the reassociation faces box_run.sh's split-aware cell section times — always built.)
        faces=("" _x _c _r _rc)
        for face in "${faces[@]}"; do
            # shellcheck disable=SC2046  # the face's flags are separate arguments
            emit_one "$f" "$b$face" "$prof" "$sm" $(flags_of "$face")
        done
        cp "$b.ptx" "$b.null.ptx"; cp "$b.launch" "$b.null.launch"
        echo "cell $prof $c default $(cat "$b.launch")"
        if [ "${SUITE_ARMS:-1}" = 1 ]; then chain "$f" "$b" "$prof" "$sm"; fi
    done
    for p in $PROGS; do
        f="$OUT/tiles/$p.mapal"; b="$d/$p"
        for face in "" _x _c; do
            # shellcheck disable=SC2046
            emit_one "$f" "$b$face" "$prof" "$sm" $(flags_of "$face")
        done
        cp "$b.host.o" "$b.null.host.o"; [ -f "$b.ptx" ] && cp "$b.ptx" "$b.null.ptx"
        c1 "$f" "$b" "$prof" "$sm"
        if same "$b" "${b}_c"; then echo "prog $prof $p default = contract (host, PTX identical)"; fi
        chain "$f" "$b" "$prof" "$sm"
        # S64: CPU-only and GPU-only arms of the default face.
        emit_cpu "$f" "$b.cpu"
        emit_one "$f" "$b.off-place" "$prof" "$sm" --gpu-off=place
        if same "$b.off-place" "$b"; then drop "$b.off-place"; fi
        # mlp_bench: each launch with its argument roles (benches/gpu/mlp_k.py).
        # plan-cost: the default face may now pick a split plan (workspace launches) that mlp_k.py cannot describe;
        # mlp_bench uses only the _x / _c stems, so its `.k` is best-effort.
        python3 "$ROOT/benches/gpu/mlp_k.py" "$b.host.ll" >/dev/null 2>&1 || true
        python3 "$ROOT/benches/gpu/mlp_k.py" "${b}_x.host.ll" "${b}_c.host.ll" >/dev/null
        cp "${b}_c.ptx" "${b}_c.null.ptx"; cp "${b}_c.k" "${b}_c.null.k"
    done
    # S64 evidence suite: the default face, CPU-only, GPU-only, the realization chain, null.
    for p in $SUITE; do
        f="$OUT/tiles/$p.mapal"; b="$d/$p"
        emit_one "$f" "$b" "$prof" "$sm"
        cp "$b.host.o" "$b.null.host.o"; [ -f "$b.ptx" ] && cp "$b.ptx" "$b.null.ptx"
        if [[ " $SUITE_C " == *" $p "* ]]; then emit_one "$f" "${b}_c" "$prof" "$sm" --order=sequential; fi
        if [[ " $XREF " == *" $p "* ]]; then
            emit_one "$f" "$b.xref" "$prof" "$sm" --exact
            [ -f "$OUT/tiles/${p}_xabs.mapal" ] && emit_one "$OUT/tiles/${p}_xabs.mapal" "$b.xabsref" "$prof" "$sm" --exact
        fi
        c1 "$f" "$b" "$prof" "$sm"
        # site-geometry: SUITE_ARMS=0 (search.py, changed_bundle.sh) keeps the shipped build and its null, plus the
        # `cpu` arm of SUITE_CPU (its CPU-only build changed); every other arm comes from ARMS.
        if [ "${SUITE_ARMS:-1}" != 1 ]; then
            case " ${SUITE_CPU:-} " in *" $p "*) emit_cpu "$f" "$b.cpu" ;; esac
            continue
        fi
        emit_cpu "$f" "$b.cpu"
        # S65: the CPU-only build's S64 arm (`by-ref` off), where it differs.
        emit_cpu "$f" "$b.cpu-off-by-ref" --gpu-off=by-ref
        if same "$b.cpu-off-by-ref" "$b.cpu"; then drop "$b.cpu-off-by-ref"; fi
        emit_one "$f" "$b.off-place" "$prof" "$sm" --gpu-off=place
        if same "$b.off-place" "$b"; then drop "$b.off-place"; fi
        chain "$f" "$b" "$prof" "$sm"
        echo "suite $prof $p $(cat "$b.launch" 2>/dev/null) cpu=$([ -f "$b.cpu.host.o" ] && echo y) gpu-only=$([ -f "$b.off-place.host.o" ] && echo arm || echo same)"
    done
    # model-scale: the small differentials, exact face and default face, shipped and GPU-only.
    for p in $PX; do
        f="$OUT/tiles/$p.mapal"
        emit_one "$f" "$d/$p" "$prof" "$sm" --exact
        emit_one "$f" "$d/$p.off-place" "$prof" "$sm" --exact --gpu-off=place
        emit_one "$f" "$d/${p}_d" "$prof" "$sm"
        emit_one "$f" "$d/${p}_d.off-place" "$prof" "$sm" --gpu-off=place
        echo "px $prof $p gpu-only kernels=$(grep -c '^define ptx_kernel' "$d/${p}_d.off-place.dev.ll" 2>/dev/null)"
    done
    # S64 re-judge arms: `<stem><face>.<on|off>-<rung>` on this part.
    for a in $ARMS; do
        IFS=: read -r ap stem face rung dir <<< "$a"
        [ "$ap" = "$prof" ] || continue
        EMIT=$MAIN_EMIT
        flag="--gpu-$dir=$rung"; tag="$dir-${rung//,/+}"
        # plan-cost: `<profile>:<stem>:<face>:base:base` is the arm emitted by BASE_EMIT (the shipped baseline's emitter), tag on-base.
        if [ "$dir" = base ]; then flag=""; tag=on-base; fi
        # tile-efficiency: `<stem>:<face>:<off-rungs>:pcoff` is plan-cost,linear ON with those rungs OFF (full-tile: arm PC), tag pc-off-<rungs>.
        if [ "$dir" = pcoff ]; then flag="--gpu-on=plan-cost,linear --gpu-off=$rung"; tag="pc-off-${rung//,/+}"; fi
        if [ "$dir" = geom ]; then flag="--geom=$rung"; tag=$(tr '=.' '-p' <<< "on-g-$rung"); fi
        # cross-arch: `<stem>:<face>:<rung>:force` is the measurement override `--<rung>=force` (`--calls=force`: the
        # chains the price refuses, formed), tag on-<rung>-force.
        if [ "$dir" = force ]; then flag="--$rung=force"; tag="on-$rung-force"; fi
        f="$OUT/tiles/$stem.mapal"; b="$d/$stem$face"; s="$b.$tag"
        [ "$face" = _tf ] && f=$(tf32_src "$f")
        [ -f "$b.host.ll" ] || emit_one "$f" "$b" "$prof" "$sm" $(flags_of "$face")
        # cpu-gpu-boundary sweep: emission alone (0.3 s) decides first; an arm whose host and device IR
        # equal the shipped build's is dropped before clang / opt / llc (hundreds of rung arms per part).
        [ "$dir" = base ] && EMIT=${BASE_EMIT:?BASE_EMIT}
        q="$d/.q"; rm -f "$q".*
        "$EMIT" "$f" - --rewrite "--target=$prof" --host=x86-64-v3 "--device-out=$q.dev.ll" $(flags_of "$face") ${flag} > "$q.host.ll"
        if cmp -s <(grep -sv '^; mapal\.' "$q.host.ll") <(grep -sv '^; mapal\.' "$b.host.ll") &&
           { { [ ! -f "$q.dev.ll" ] && [ ! -f "$b.dev.ll" ]; } || cmp -s "$q.dev.ll" "$b.dev.ll"; }; then
            rm -f "$q".*; echo "rejudge $prof $stem$face $tag = shipped"; continue
        fi
        # per-site-choice B5: a CHOICE_AXES rung whose every kernel is already a candidate of the shipped
        # tables is measured by the tune pass, not as an emission arm.
        if { [ "$dir" = on ] || [ "$dir" = off ]; } && python3 "$ROOT/benches/gpu/evidence.py" covered "$q" "$b" "$rung"; then
            rm -f "$q".*; echo "rejudge $prof $stem$face $tag = table"; continue
        fi
        rm -f "$q".*
        emit_one "$f" "$s" "$prof" "$sm" $(flags_of "$face") ${flag}
        if same "$s" "$b"; then drop "$s"; echo "rejudge $prof $stem$face $tag = shipped"; continue; fi
        echo "rejudge $prof $stem$face $tag $(cat "$s.launch" 2>/dev/null)"
        case $stem in pt_rm_*) [ "$face" = _c ] || [ "$face" = _x ] && python3 "$ROOT/benches/gpu/mlp_k.py" "$s.host.ll" >/dev/null ;; esac
    done
    EMIT=$MAIN_EMIT
    # cross-arch: an arm equal to the shipped build up to kernel names (the hash covers the rung set) is dropped too.
    [ -n "$ARMS" ] && python3 "$ROOT/benches/gpu/armdedup.py" "$d"
    # hardware-properties: the rungs the rules ship on this part (the intent check's `table.rungs_on`).
    "$MAIN_EMIT" --rungs "--target=$prof" > "$d/rungs.tsv"
    # S64: the evidence — ptxas -v / SASS of every distinct PTX (cudatool.sh), the facts, the intent check.
    python3 "$ROOT/benches/gpu/evidence.py" static "$d" "$sm"
    python3 "$ROOT/benches/gpu/evidence.py" facts "$d" "$sm"
    python3 "$ROOT/benches/gpu/evidence.py" check "$d" > "$d/evidence/evidence.tsv"
    echo "evidence $prof: $(tail -1 "$d/evidence/evidence.tsv")"
    gzip -f "$d"/*.remarks.yaml 2>/dev/null || true
done
cp "$ROOT/target/x86_64-unknown-linux-gnu/release/libmapal_rt.a" "$OUT/"
cp "$ROOT/crates/mapal-rt/gpu/mapal_gpu.c" "$ROOT/crates/mapal-rt/gpu/mapal_choice.h" "$ROOT/benches/gpu/tile_bench.cu" "$ROOT/benches/gpu/chain_bench.cu" \
    "$ROOT/benches/gpu/mlp_bench.cu" "$ROOT/benches/gpu/gemm_lib.cu" "$ROOT/benches/gpu/value_gate.py" \
    "$ROOT/benches/gpu/box_run.sh" "$ROOT/benches/gpu/box_profile.sh" "$OUT/"
# cross-arch: the device calibration box_run.sh runs before the timed sections (the native-gpu profile's properties).
cp "$ROOT/benches/gpu/calib.cu" "$ROOT/benches/local/exp_calib.cu" "$ROOT/benches/local/tma_calib.cu" "$ROOT/benches/local/ktrace.cpp" "$OUT/"
# S64: the checker and its inputs, for the box's `exec` / `jit` checks; the torch legs.
cp "$ROOT/benches/gpu/"{confirm_pick.py,evidence.py,intent.py,sass_stats.py,occ.py,ptx_static.py,cudatool.sh,torch_bench.py,comp_bench.py,triton_kernels.py,comp_build.sh} "$OUT/"
cp "$ROOT/benches/gpu/capture_images/capture.cpp" "$OUT/"
if [ ! -d "$OUT/cutlass" ]; then
    git clone -q --depth 1 https://github.com/NVIDIA/cutlass "$OUT/cutlass.git"
    mkdir -p "$OUT/cutlass/tools/util"
    cp -R "$OUT/cutlass.git/include" "$OUT/cutlass/"
    cp -R "$OUT/cutlass.git/tools/util/include" "$OUT/cutlass/tools/util/"
    (cd "$OUT/cutlass.git" && git rev-parse HEAD) > "$OUT/cutlass/COMMIT"
    rm -rf "$OUT/cutlass.git"
fi
echo "bundle cache: $(wc -l < "$OUT/.bundle-cache-hits" 2>/dev/null || echo 0) built stems and oracles reused (${BUNDLE_CACHE})"
echo "bundle ready: $OUT"
