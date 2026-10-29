#!/usr/bin/env bash
# Move a sweep's raw output (profile archives, box logs, raw results) to a GitHub release instead of git.
# usage: benches/gpu/archive_raw.sh <release-tag> <evidence-dir>...
# Appends path / bytes / sha256 / asset rows to benches/results/evidence/RAW-manifest.tsv; the raw files stay on disk
# (git ignores them, see .gitignore) and the asset is raw-<dir name>.tar.gz.
set -euo pipefail
tag=$1; shift
root=$(git rev-parse --show-toplevel); man="$root/benches/results/evidence/RAW-manifest.tsv"
[ -s "$man" ] || printf 'path\tbytes\tsha256\tlast_commit\tasset\n' > "$man"
gh release view "$tag" >/dev/null 2>&1 || gh release create "$tag" --target main --title "Raw benchmark evidence ($tag)" --notes "See benches/results/evidence/RAW.md"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for dir in "$@"; do
  rel=$(cd "$root" && realpath --relative-to=. "$dir" 2>/dev/null || python3 -c 'import os,sys;print(os.path.relpath(sys.argv[1],sys.argv[2]))' "$dir" "$root")
  asset="raw-$(basename "$rel").tar.gz"
  (cd "$root" && find "$rel" -type f \( -name '*.tgz' -o -name '*.out' -o -name '*.log' -o -name 'results-*' -o -name 'run-*' \)) | sort > "$tmp/list"
  [ -s "$tmp/list" ] || { echo "$rel: no raw files"; continue; }
  (cd "$root" && tar -czf "$tmp/$asset" -T "$tmp/list")
  while IFS= read -r p; do
    printf '%s\t%s\t%s\t-\t%s\n' "$p" "$(wc -c < "$root/$p" | tr -d ' ')" "$(shasum -a 256 "$root/$p" | cut -d' ' -f1)" "$asset" >> "$man"
  done < "$tmp/list"
  gh release upload "$tag" "$tmp/$asset" --clobber
  echo "$rel: $(wc -l < "$tmp/list" | tr -d ' ') files -> $tag/$asset"
done
