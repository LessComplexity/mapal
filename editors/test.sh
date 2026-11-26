#!/usr/bin/env sh
# Both editors' highlighting, language server and formatting.
# Build mapal-cli first (cargo build -p mapal-cli, or set MAPAL_BIN), and run
# `npm install` in editors/vscode once (the real TextMate engine and the
# language client's protocol stack).
#
# The Vim syntax and the TextMate grammar are hand-maintained with OPPOSITE
# precedence (Vim last-match-wins, TextMate earliest-match), so "consistent"
# is something to check, not to hope for: both are held to the class of every
# token of editors/test/fixture.mapal (fixture.tokens.tsv, classes.tsv), the
# same table the server's semantic tokens are tested against
# (cargo test -p mapal-lsp).
set -e
cd "$(dirname "$0")/.."
if [ ! -d editors/vscode/node_modules ]; then
  echo "editors/vscode/node_modules is missing: run (cd editors/vscode && npm install)" >&2
  exit 1
fi
echo "== neovim"
sh editors/nvim/test/run.sh
echo
echo "== vscode"
python3 editors/vscode/test/scope_test.py
node editors/vscode/test/grammar_test.js
node editors/vscode/test/lsp_test.js
node editors/vscode/test/format_test.js
