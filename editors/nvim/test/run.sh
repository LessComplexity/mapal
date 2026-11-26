#!/usr/bin/env sh
# Highlighting, formatting and the language server in a headless Neovim.
# Build mapal-cli first (cargo build -p mapal-cli), or set MAPAL_BIN.
set -e
cd "$(dirname "$0")/../../.."
nvim --headless -u NONE -i NONE -c "luafile editors/nvim/test/syntax_test.lua"
nvim --headless -u NONE -i NONE -c "luafile editors/nvim/test/lsp_test.lua"
exec nvim --headless -u NONE -i NONE -c "luafile editors/nvim/test/format_test.lua"
