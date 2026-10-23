#!/bin/bash
# CLI `build --emit-only` vs bundle.sh's emit_one path (emit example + opt -O3 + llc -O3
# -mcpu=<sm> -nvptx-fma-level=0): host .ll and PTX must be byte-identical.
set -u
ROOT=$1; TILES=$2; OUT=$3
LLVM=/opt/homebrew/opt/llvm/bin
EMIT=$ROOT/target/debug/examples/emit
CLI=$ROOT/target/debug/mapal
mkdir -p "$OUT"
for spec in rtx-4090:sm_89 rtx-5090:sm_120 h100-pcie:sm_90a rtx-3090:sm_86; do
  prof=${spec%%:*}; sm=${spec##*:}
  for p in mm_4096 pt_mlp_2048 pt_residual_2048 rm_128x384x96; do
    for face in exact contract; do
      flag=(); [ $face = contract ] && flag=(--contract)
      d="$OUT/$prof/$p.$face"; mkdir -p "$d"
      "$EMIT" "$TILES/$p.mapal" - --rewrite "--target=$prof" "--device-out=$d/b.dev.ll" ${flag[@]+"${flag[@]}"} > "$d/b.host.ll"
      "$LLVM/opt" -O3 "$d/b.dev.ll" -S -o "$d/b.dev.opt.ll"
      "$LLVM/llc" -O3 -mcpu="$sm" -nvptx-fma-level=0 "$d/b.dev.opt.ll" -o "$d/b.ptx"
      LLVM=$LLVM "$CLI" build "$TILES/$p.mapal" -o "$d/c" --emit-only "--target=$prof" ${flag[@]+"${flag[@]}"} 2>/dev/null
      h=same; cmp -s "$d/b.host.ll" "$d/c.ll" || h=DIFF
      x=same; cmp -s "$d/b.ptx" "$d/c.ptx" || x=DIFF
      echo "$prof $p $face host=$h ptx=$x target=$(grep -m1 '^.target' "$d/c.ptx")"
    done
  done
done
