#!/bin/bash
# S63 default-float session bundle (ADR-0040), S64 evidence pass: per part, every face of every cell and
# whole program, the default face's rung arms, and a null copy — for box_run.sh's splitk section (every
# kernel arm of a cell, cuBLAS, stock and tuned CUTLASS in one tile_bench process, value gates first, two
# processes ABBA), its MLP section (mlp_bench: the programs' kernels vs cuBLAS, cuBLASLt, CUTLASS) and its
# probe section (whole programs, value-gated, ABBA per repetition) — and, per part, the evidence
# (evidence.py: ptxas -v, SASS, LLVM remarks, the intent check) in `<part>/evidence/`.
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
# rows' cells (evidence.py rejudge prints them).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=$1; shift
LLVM="${LLVM:-/opt/homebrew/opt/llvm/bin}"
EMIT="${EMIT:-$ROOT/target/release/examples/emit}"
CELLS=${CELLS-"16x2048x7168 16x1024x4096 16x18432x7168 128x2048x7168 4096x4096x4096"}
PROGS=${PROGS-"pt_rm_m128_d256_h512 pt_rm_m128_d1024_h2048"}
SUITE=${SUITE-""}
ARMS=${ARMS-""}
mkdir -p "$OUT/tiles"
python3 "$ROOT/benches/gpu/gen_tiles.py" "$OUT/tiles" >/dev/null
# S64: the CPU build of a looped program (pt_rt_mlp2_1024) needs more than an 8 MB stack (rc 139 at
# the default); macOS's hard limit is 64 MB.
ulimit -s 65520 2>/dev/null || true
# The whole programs' oracles: the CPU build on the exact face (the interpreter bit for bit) and
# on the contract face (per-step fmaf), `iter ms` dropped. The default face answers to the exact
# one within ADR-0040 D2's bound (value_gate.py).
for p in $PROGS $SUITE; do
    # The suite runs the default face alone: its gate is the exact face's oracle.
    # (and the contract face where a re-judge arm runs on it).
    faces=(_x _c); [[ " $SUITE " == *" $p "* ]] && [[ " $ARMS " != *":$p:_c:"* ]] && faces=(_x)
    for face in "${faces[@]}"; do
        [ -s "$OUT/tiles/$p$face.expect" ] && continue
        fl=(--exact); [ "$face" = _c ] && fl=(--order=sequential)
        "$EMIT" "$OUT/tiles/$p.mapal" - --rewrite "${fl[@]}" --target=apple-m > "$OUT/tiles/$p$face.cpu.ll"
        clang -O2 -Wno-override-module "$OUT/tiles/$p$face.cpu.ll" "$ROOT/target/release/libmapal_rt.a" -o "$OUT/tiles/$p$face.cpu"
        "$OUT/tiles/$p$face.cpu" | grep -v '^iter ms=' > "$OUT/tiles/$p$face.expect"
    done
    # S65: ADR-0040 D2's M per printed value (gen_tiles.py `xabs`), where the program has one.
    if [ -f "$OUT/tiles/${p}_xabs.mapal" ] && [ ! -s "$OUT/tiles/${p}_xabs.expect" ]; then
        "$EMIT" "$OUT/tiles/${p}_xabs.mapal" - --rewrite --exact --target=apple-m > "$OUT/tiles/${p}_xabs.cpu.ll"
        clang -O2 -Wno-override-module "$OUT/tiles/${p}_xabs.cpu.ll" "$ROOT/target/release/libmapal_rt.a" -o "$OUT/tiles/${p}_xabs.cpu"
        "$OUT/tiles/${p}_xabs.cpu" | grep -v '^iter ms=' > "$OUT/tiles/${p}_xabs.expect"
    fi
done

flags_of() { # <face>
    case $1 in _x) echo --exact ;; _c) echo --order=sequential ;; _r) echo --no-contract --order=fixed ;;
    _rc) echo --order=fixed ;; _tf) echo --contract --precision=tf32 ;; *) ;; esac
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
    host_cc "$stem" "$m"
    [ -f "$stem.dev.ll" ] || return 0
    "$LLVM/opt" -O3 "$stem.dev.ll" -S -o "$stem.dev.opt.ll" -pass-remarks-output="$stem.dev.opt.remarks.yaml" \
        -pass-remarks-filter='loop-unroll|sroa|licm|gvn|instcombine|loop-vectorize'
    "$LLVM/llc" -O3 -mcpu="$sm" -nvptx-fma-level=0 "$stem.dev.opt.ll" -o "$stem.ptx" \
        -pass-remarks-output="$stem.dev.llc.remarks.yaml"
    # `kernel gx gy block smem`, then `S rkernel rgx rgy rblock rsmem` (workspace split),
    # `S atomic` (splitk-atomic) or `1 streamk` (stream-K).
    { grep -oE 'call void @mapal_gpu_(tile|launch|launch_l2)\(i64 [-0-9]*.*' "$stem.host.ll" || true; } | head -2 |
        python3 -c 'import sys,re
ls = sys.stdin.read().splitlines()
k = lambda l: "mapal_tile_%016x" % (int(re.search(r"i64 (-?\d+)", l).group(1)) & (2**64 - 1))
n = lambda l: re.findall(r"i32 (\d+)", l.split(";")[0])
s = ls and re.search(r"; splitk=(\d+)( atomic)?", ls[0])
sk = ls and re.search(r"; streamk=", ls[0])
tail = ([s.group(1), "atomic"] if s and s.group(2) else [s.group(1), k(ls[1]), *n(ls[1])]) if s else (["1", "streamk"] if sk else [])
ls and print(k(ls[0]), *n(ls[0]), *tail)' > "$stem.launch"
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
drop() { rm -f "$1".host.* "$1".dev.* "$1".ptx "$1".launch "$1".k; }
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
        for face in "" _x _c _r _rc; do
            # shellcheck disable=SC2046  # the face's flags are separate arguments
            emit_one "$f" "$b$face" "$prof" "$sm" $(flags_of "$face")
        done
        cp "$b.ptx" "$b.null.ptx"; cp "$b.launch" "$b.null.launch"
        echo "cell $prof $c default $(cat "$b.launch")"
        chain "$f" "$b" "$prof" "$sm"
    done
    for p in $PROGS; do
        f="$OUT/tiles/$p.mapal"; b="$d/$p"
        for face in "" _x _c; do
            # shellcheck disable=SC2046
            emit_one "$f" "$b$face" "$prof" "$sm" $(flags_of "$face")
        done
        cp "$b.host.o" "$b.null.host.o"; [ -f "$b.ptx" ] && cp "$b.ptx" "$b.null.ptx"
        if same "$b" "${b}_c"; then echo "prog $prof $p default = contract (host, PTX identical)"; fi
        chain "$f" "$b" "$prof" "$sm"
        # S64: CPU-only and GPU-only arms of the default face.
        emit_cpu "$f" "$b.cpu"
        emit_one "$f" "$b.off-place" "$prof" "$sm" --gpu-off=place
        if same "$b.off-place" "$b"; then drop "$b.off-place"; fi
        # mlp_bench: each launch with its argument roles (benches/gpu/mlp_k.py).
        python3 "$ROOT/benches/gpu/mlp_k.py" "$b.host.ll" "${b}_x.host.ll" "${b}_c.host.ll" >/dev/null
        cp "${b}_c.ptx" "${b}_c.null.ptx"; cp "${b}_c.k" "${b}_c.null.k"
    done
    # S64 evidence suite: the default face, CPU-only, GPU-only, the realization chain, null.
    for p in $SUITE; do
        f="$OUT/tiles/$p.mapal"; b="$d/$p"
        emit_one "$f" "$b" "$prof" "$sm"
        cp "$b.host.o" "$b.null.host.o"; [ -f "$b.ptx" ] && cp "$b.ptx" "$b.null.ptx"
        emit_cpu "$f" "$b.cpu"
        # S65: the CPU-only build's S64 arm (`by-ref` off), where it differs.
        emit_cpu "$f" "$b.cpu-off-by-ref" --gpu-off=by-ref
        if same "$b.cpu-off-by-ref" "$b.cpu"; then drop "$b.cpu-off-by-ref"; fi
        emit_one "$f" "$b.off-place" "$prof" "$sm" --gpu-off=place
        if same "$b.off-place" "$b"; then drop "$b.off-place"; fi
        chain "$f" "$b" "$prof" "$sm"
        echo "suite $prof $p $(cat "$b.launch" 2>/dev/null) cpu=$([ -f "$b.cpu.host.o" ] && echo y) gpu-only=$([ -f "$b.off-place.host.o" ] && echo arm || echo same)"
    done
    # S64 re-judge arms: `<stem><face>.<on|off>-<rung>` on this part.
    for a in $ARMS; do
        IFS=: read -r ap stem face rung dir <<< "$a"
        [ "$ap" = "$prof" ] || continue
        f="$OUT/tiles/$stem.mapal"; b="$d/$stem$face"; s="$b.$dir-${rung//,/+}"
        [ -f "$b.host.ll" ] || emit_one "$f" "$b" "$prof" "$sm" $(flags_of "$face")
        # cpu-gpu-boundary sweep: emission alone (0.3 s) decides first; an arm whose host and device IR
        # equal the shipped build's is dropped before clang / opt / llc (hundreds of rung arms per part).
        q="$d/.q"; rm -f "$q".*
        "$EMIT" "$f" - --rewrite "--target=$prof" --host=x86-64-v3 "--device-out=$q.dev.ll" $(flags_of "$face") "--gpu-$dir=$rung" > "$q.host.ll"
        if cmp -s <(grep -sv '^; mapal\.' "$q.host.ll") <(grep -sv '^; mapal\.' "$b.host.ll") &&
           { { [ ! -f "$q.dev.ll" ] && [ ! -f "$b.dev.ll" ]; } || cmp -s "$q.dev.ll" "$b.dev.ll"; }; then
            rm -f "$q".*; echo "rejudge $prof $stem$face $dir-$rung = shipped"; continue
        fi
        rm -f "$q".*
        emit_one "$f" "$s" "$prof" "$sm" $(flags_of "$face") "--gpu-$dir=$rung"
        if same "$s" "$b"; then drop "$s"; echo "rejudge $prof $stem$face $dir-$rung = shipped"; continue; fi
        echo "rejudge $prof $stem$face $dir-$rung $(cat "$s.launch" 2>/dev/null)"
        case $stem in pt_rm_*) [ "$face" = _c ] || [ "$face" = _x ] && python3 "$ROOT/benches/gpu/mlp_k.py" "$s.host.ll" >/dev/null ;; esac
    done
    # S64: the evidence — ptxas -v / SASS of every distinct PTX (cudatool.sh), the facts, the intent check.
    python3 "$ROOT/benches/gpu/evidence.py" static "$d" "$sm"
    python3 "$ROOT/benches/gpu/evidence.py" facts "$d" "$sm"
    python3 "$ROOT/benches/gpu/evidence.py" check "$d" > "$d/evidence/evidence.tsv"
    echo "evidence $prof: $(tail -1 "$d/evidence/evidence.tsv")"
    gzip -f "$d"/*.remarks.yaml 2>/dev/null || true
done
cp "$ROOT/target/x86_64-unknown-linux-gnu/release/libmapal_rt.a" "$OUT/"
cp "$ROOT/crates/mapal-rt/gpu/mapal_gpu.c" "$ROOT/benches/gpu/tile_bench.cu" "$ROOT/benches/gpu/chain_bench.cu" \
    "$ROOT/benches/gpu/mlp_bench.cu" "$ROOT/benches/gpu/value_gate.py" \
    "$ROOT/benches/gpu/box_run.sh" "$ROOT/benches/gpu/box_profile.sh" "$OUT/"
# S64: the checker and its inputs, for the box's `exec` / `jit` checks; the torch legs.
cp "$ROOT/benches/gpu/"{evidence.py,intent.py,sass_stats.py,occ.py,ptx_static.py,cudatool.sh,torch_bench.py} "$OUT/"
cp "$ROOT/benches/gpu/capture_images/capture.cpp" "$OUT/"
if [ ! -d "$OUT/cutlass" ]; then
    git clone -q --depth 1 https://github.com/NVIDIA/cutlass "$OUT/cutlass.git"
    mkdir -p "$OUT/cutlass/tools/util"
    cp -R "$OUT/cutlass.git/include" "$OUT/cutlass/"
    cp -R "$OUT/cutlass.git/tools/util/include" "$OUT/cutlass/tools/util/"
    (cd "$OUT/cutlass.git" && git rev-parse HEAD) > "$OUT/cutlass/COMMIT"
    rm -rf "$OUT/cutlass.git"
fi
echo "bundle ready: $OUT"
