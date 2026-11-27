#!/usr/bin/env bash
# check_file_size.sh: every tracked code file under crates/ and editors/ is at
# most MAX lines (default 1,000, counted by `wc -l`): Rust, C (with the GPU
# runtime's `#include`d `.inc` parts), C headers, CUDA, Lua and TypeScript.
# Lists the offenders, longest first, and exits 1 when there are any.
# A file near the cap is split into submodules (pure moves; the original file
# stays the module root and re-exports), its tests in their own module files.
set -euo pipefail

MAX=${MAX:-1000}
cd "$(git rev-parse --show-toplevel)"

patterns=()
for dir in crates editors; do
    for ext in rs c h inc cu lua ts; do
        patterns+=("$dir/*.$ext")
    done
done

offenders=$(
    git ls-files -z -- "${patterns[@]}" |
        while IFS= read -r -d '' f; do
            n=$(($(wc -l <"$f")))
            if [ "$n" -gt "$MAX" ]; then
                printf '%7d  %s\n' "$n" "$f"
            fi
        done | sort -rn
)

if [ -n "$offenders" ]; then
    echo "check_file_size: files over $MAX lines (split them into submodules):" >&2
    echo "$offenders" >&2
    exit 1
fi
echo "check_file_size: every code file under crates/ and editors/ is at most $MAX lines"
