#!/bin/bash
# Old (4e1102c) emit.rs vs new emit.rs (flags::apply): stdout, device unit, stderr, exit.
set -u
ROOT=$1; TILES=$2; OUT=$3
OLD=$ROOT/target/debug/examples/emit_old
NEW=$ROOT/target/debug/examples/emit
mkdir -p "$OUT"
n=0; bad=0
while IFS= read -r line; do
  read -r -a f <<< "$line"
  for p in "$TILES/pt_mlp_2048.mapal" "$ROOT/examples/fir.mapal" "$ROOT/examples/matmul4_loop.mapal"; do
    "$OLD" "$p" - "${f[@]}" "--device-out=$OUT/o.dev" > "$OUT/o.ll" 2> "$OUT/o.err"; oc=$?
    "$NEW" "$p" - "${f[@]}" "--device-out=$OUT/n.dev" > "$OUT/n.ll" 2> "$OUT/n.err"; nc=$?
    s=same
    cmp -s "$OUT/o.ll" "$OUT/n.ll" || s=DIFF
    cmp -s "$OUT/o.err" "$OUT/n.err" || s=DIFF
    [ "$oc" = "$nc" ] || s=DIFF
    if [ -f "$OUT/o.dev" ] || [ -f "$OUT/n.dev" ]; then cmp -s "$OUT/o.dev" "$OUT/n.dev" || s=DIFF; fi
    rm -f "$OUT/o.dev" "$OUT/n.dev"
    n=$((n+1)); [ $s = same ] || { bad=$((bad+1)); echo "DIFF: $(basename "$p") ${f[*]} (exit $oc/$nc)"; }
  done
done <<'EOF'
--rewrite --target=generic
--rewrite --target=apple-m --perf
--rewrite --target=zen3 --no-tile
--rewrite --target=generic --no-pack --kc --contract
--rewrite --target=apple-m --move-panel=off
--rewrite --target=apple-m --move-panel=64:8
--move-panel=0:8
--rewrite --target=rtx-4090
--rewrite --target=rtx-4090 --contract --gpu-off=fma-regs
--rewrite --target=h100-pcie --gpu-on=arch-select,raster
--rewrite --target=rtx-5090 --precision=tf32 --gpu-off=mma
--rewrite --target=rtx-3090 --gpu-off=nosuchrung
--rewrite --target=nosuchprofile
--bogus
EOF
echo "cases=$n diffs=$bad"
