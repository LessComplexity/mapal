#!/bin/bash
# S63 default-float session bundle (ADR-0040): per part, every face of every cell and whole
# program, the default face's rung arms, and a null copy — for box_run.sh's splitk section
# (every kernel arm of a cell, cuBLAS, stock and tuned CUTLASS in one tile_bench process, value
# gates first, two processes ABBA), its MLP section (mlp_bench: the programs' kernels vs cuBLAS,
# cuBLASLt, CUTLASS) and its probe section (whole programs, value-gated, ABBA per repetition).
#
# usage: benches/gpu/integrated_bundle.sh <outdir> <profile:sm>...
# Faces (stem suffix): "" default (no flags: fma, order free), _x --exact, _c --order=sequential
# (contract), _r --no-contract --order=fixed (the fixed test/debug face), _rc --order=fixed.
# Default-face arms: .off-streamk, .off-streamk+splitk-atomic, .off-streamk+splitk-atomic+splitk
# (each dropped where it equals the arm above it: the rung does not change that launch), .null.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT=$1; shift
LLVM="${LLVM:-/opt/homebrew/opt/llvm/bin}"
EMIT="$ROOT/target/release/examples/emit"
CELLS=${CELLS:-"16x2048x7168 16x1024x4096 16x18432x7168 128x2048x7168 4096x4096x4096"}
PROGS=${PROGS:-"pt_rm_m128_d256_h512 pt_rm_m128_d1024_h2048"}
mkdir -p "$OUT/tiles"
python3 "$ROOT/benches/gpu/gen_tiles.py" "$OUT/tiles" >/dev/null
# The whole programs' oracles: the CPU build on the exact face (the interpreter bit for bit) and
# on the contract face (per-step fmaf), `iter ms` dropped. The default face answers to the exact
# one within ADR-0040 D2's bound (value_gate.py).
for p in $PROGS; do
    for face in _x _c; do
        fl=(--exact); [ "$face" = _c ] && fl=(--order=sequential)
        "$EMIT" "$OUT/tiles/$p.mapal" - --rewrite "${fl[@]}" --target=apple-m > "$OUT/tiles/$p$face.cpu.ll"
        clang -O2 -Wno-override-module "$OUT/tiles/$p$face.cpu.ll" "$ROOT/target/release/libmapal_rt.a" -o "$OUT/tiles/$p$face.cpu"
        "$OUT/tiles/$p$face.cpu" | grep -v '^iter ms=' > "$OUT/tiles/$p$face.expect"
    done
done

flags_of() { # <face>
    case $1 in _x) echo --exact ;; _c) echo --order=sequential ;; _r) echo --no-contract --order=fixed ;;
    _rc) echo --order=fixed ;; *) ;; esac
}
emit_one() { # <src> <stem> <prof> <sm> [flags...]
    local src=$1 stem=$2 prof=$3 sm=$4; shift 4
    rm -f "$stem.dev.ll"
    "$EMIT" "$src" - --rewrite "--target=$prof" --host=x86-64-v3 "--device-out=$stem.dev.ll" "$@" > "$stem.host.ll"
    clang -O2 -c --target=x86_64-unknown-linux-gnu -march=x86-64-v3 -Wno-override-module "$stem.host.ll" -o "$stem.host.o"
    [ -f "$stem.dev.ll" ] || return 0
    "$LLVM/opt" -O3 "$stem.dev.ll" -S -o "$stem.dev.opt.ll"
    "$LLVM/llc" -O3 -mcpu="$sm" -nvptx-fma-level=0 "$stem.dev.opt.ll" -o "$stem.ptx"
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
same() { local x; for x in host.ll ptx launch; do
    if [ -f "$1.$x" ] || [ -f "$2.$x" ]; then cmp -s "$1.$x" "$2.$x" || return 1; fi; done; }
# The default face's realization chain: each arm turns the next rung off; one equal to the arm
# above it (the rung leaves that launch alone) is dropped.
chain() { # <src> <stem> <prof> <sm>
    local src=$1 stem=$2 prof=$3 sm=$4 prev=$2 off="" r
    for r in streamk splitk-atomic splitk; do
        off=${off:+$off,}$r
        emit_one "$src" "$stem.off-${off//,/+}" "$prof" "$sm" "--gpu-off=$off"
        if same "$stem.off-${off//,/+}" "$prev"; then rm -f "$stem.off-${off//,/+}".*
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
        # mlp_bench: each launch with its argument roles (benches/gpu/mlp_k.py).
        python3 "$ROOT/benches/gpu/mlp_k.py" "$b.host.ll" "${b}_x.host.ll" "${b}_c.host.ll" >/dev/null
        cp "${b}_c.ptx" "${b}_c.null.ptx"; cp "${b}_c.k" "${b}_c.null.k"
    done
done
cp "$ROOT/target/x86_64-unknown-linux-gnu/release/libmapal_rt.a" "$OUT/"
cp "$ROOT/crates/mapal-rt/gpu/mapal_gpu.c" "$ROOT/benches/gpu/tile_bench.cu" "$ROOT/benches/gpu/chain_bench.cu" \
    "$ROOT/benches/gpu/mlp_bench.cu" "$ROOT/benches/gpu/value_gate.py" \
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
