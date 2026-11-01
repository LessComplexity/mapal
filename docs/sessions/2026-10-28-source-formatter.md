# 2026-10-28 — source formatter

## 0. Continuation brief

Implemented shared Mapal source formatting and both editor integrations on
`site-geometry-latest`. The CLI is built at `target/debug/mapal`; the VSIX is
`editors/vscode/mapal-lang-0.2.0.vsix`. Nothing committed or pushed. The user's
partial edits to `benches/algorithms/gen_16_8_8.mapal` were left unchanged.
Next step: use the formatter/editor setup or review the diff; no implementation
work remains for this change.

```sh
git status --short
target/debug/mapal fmt --help
target/debug/mapal fmt --stdout benches/algorithms/gen_16_8_8.mapal
```

## 1. Work completed

- Pure `mapal_syntax::format_source` uses the existing lexer/parser and token
  spans; four spaces, multiline braces, statement/record-field breaks, consistent
  punctuation spacing. Comments and literal spelling remain intact. Deliberate
  line/paragraph breaks are retained; no width-based expression reflow.
- `mapal fmt <files...>`, `--check`, `--stdout` and stdin `-`. All requested
  inputs are validated before writes. Atomic replacement preserves symlinks and
  permissions; duplicate paths/aliases and already-formatted inputs are no-ops.
- Neovim `:MapalFormat`/`gggqG`, executable setting `vim.g.mapal_executable`.
  VS Code **Format Document**, setting `mapal.executablePath`. Both invoke the
  same CLI, preserve errors/stale buffers, and bound process time/output.
- Updated packaging, tests, README/contributor guidance, `.editorconfig` and
  `CLAUDE.md` (also read through the existing `AGENTS.md` symlink).

## 2. Decisions

| Decision | Verdict | Reason |
| --- | --- | --- |
| Existing CLI formatter | none existed | source dispatch and both prior binary help outputs had only build/run/dump-ir/test |
| Shared token formatter | kept | original source spans preserve comments/literals; no second AST or editor formatting policy |
| Separate editor layout rules / new LSP | not added | stdin/stdout CLI port meets the requested editor behavior |
| Existing benchmark reformat | not performed | user requested tooling; gen preview is a separate temporary file |
| Recursive indexing | stopped | user reported killing an approximately 80 GB Python process during stalled concurrent Semble requests |

The memory incident's killed PID was not recovered. Eight concurrent MCP searches
and a Semble CLI fallback had been started across agents. The fallback traceback
showed traversal into `.claude/worktrees/_archive-s54-20261021/.../bundles/a100/cutlass`.
Cancelling orchestration cells did not establish that server-side Python stopped.
After the user's intervention, process inspection showed a remaining idle Semble
Python process at approximately 1.3 MB resident memory. These facts make indexing
the strongest suspect, not a proven identification of the killed process.
The resumed work used bounded direct reads and no further Semble/graph indexing.

## 3. Tests and checks

| Check | Result |
| --- | --- |
| `CARGO_BUILD_JOBS=4 RUST_TEST_THREADS=4 cargo test -p mapal-syntax` | 215 passed, including 12 formatter tests |
| Formatter breadth | 192 generated whitespace variants; 205 valid files in explicit nonrecursive directories; only existing invalid `examples/vector.mapal` exempted |
| `CARGO_BUILD_JOBS=4 RUST_TEST_THREADS=4 cargo test -p mapal-cli --test format` | 8 passed, no compiler/GPU toolchain needed |
| `NVIM_LOG_FILE=/tmp/mapal-format-nvim.log sh editors/test.sh` | formatting integration passes for both editors, plus 29 Neovim / 32 VS Code highlighting assertions |
| Rust formatting on changed Rust files; `git diff --check` | clean |
| OpenSpec strict validation and archive | valid; archived as `2026-10-28-add-source-formatter`, 3 requirements merged |
| supercharge drift | 0 dead / 701 refs |

The drift script uses `git ls-files`; new source paths were supplied through a
temporary index copied to `/tmp/mapal-source-formatting/drift.index` and populated
with `git update-index --info-only --cacheinfo` using real file hashes. The real
repository index was not changed. No whole-workspace/GPU suite was needed for this
source-only tooling change. VS Code provider tests use an API test double with the
real CLI, rather than launching the full editor UI.

## 4. Live execution state

| Type | Handle / location | State | Inspect / resume |
| --- | --- | --- | --- |
| branch | `site-geometry-latest` | implementation/doc changes uncommitted | `git status --short` |
| task processes | Rust/editor tests and drift | all completed; no task jobs remain | logs below |
| artifact | `target/debug/mapal` | built | `target/debug/mapal fmt --help` |
| artifact | `editors/vscode/mapal-lang-0.2.0.vsix` | built/verified, not globally installed | `code --install-extension editors/vscode/mapal-lang-0.2.0.vsix --force` |
| preview/logs | `/tmp/mapal-source-formatting/` | gen preview, syntax/CLI/editor test summaries and drift output | `cat /tmp/mapal-source-formatting/drift.log` |
| user source | `benches/algorithms/gen_16_8_8.mapal` | pre-existing modifications preserved | `shasum benches/algorithms/gen_16_8_8.mapal` → `0d581f1f9bedd344c0466510f0183dc5b2e93c8a` |

Existing untracked `.claude/`, `oainotes.md` and `summary.md` were untouched. No
remote machines, providers or ports were used. No global installation was made.
Install the CLI with `cargo install --path crates/mapal-cli --locked`, or point
the editor executable setting to the absolute `target/debug/mapal` path.

## 5. OpenSpec state

`add-source-formatter` is complete (7/7), archived and merged into
`openspec/specs/source-formatting/spec.md`. Other changes were not touched:

| Change | Tasks | Status |
| --- | --- | --- |
| optimize-ordered-smallm-gpu | 6/7 | in-progress |
| benchmark-large-composed-gpu-workloads | 6/15 | in-progress |
| eliminate-unmaterialized-gpu-allocations | 12/15 | in-progress |

Inspect with `openspec list --json`; those unrelated GPU continuations remain in
their prior session/evidence records.

## 6. Open items

No formatter implementation items remain. Graph refresh and gbrain indexing were
deliberately skipped after the memory incident; do not restart recursive indexing
without exclusions/resource bounds for archived worktrees and vendor bundles.
The code, OpenSpec contract and this immutable log are the authoritative state.

## 7. Docs reconciled

Syntax and CLI IMPLEMENTATION mappings, then their DESIGN models and STATUS;
root IMPLEMENTATION, STATUS and architecture map; CLI coherence review at
`docs/components/cli/reviews/review-source-formatter.md`; README, CONTRIBUTING,
agent guidance and both editor READMEs. The older components use DESIGN.md as
their architecture document, as recorded in `openspec/config.yaml`.
