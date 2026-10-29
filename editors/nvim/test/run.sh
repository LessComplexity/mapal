#!/usr/bin/env sh
# Assert highlighting and formatting. Build mapal-cli before running.
set -e
cd "$(dirname "$0")/../../.."
nvim --headless -u NONE -i NONE -c "luafile editors/nvim/test/syntax_test.lua"
exec nvim --headless -u NONE -i NONE -c "luafile editors/nvim/test/format_test.lua"
