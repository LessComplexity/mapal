## Context

The CLI currently dispatches only build/run/dump-ir/test. The syntax crate has a total lexer and parser with byte spans, but the AST does not retain comments. Both editor plugins currently provide highlighting only. See the proposal for motivation.

## Goals / Non-Goals

Provide one deterministic layout and safe CLI/editor transports. Preserve source semantics and literal spelling. Keep file selection explicit and source inspection bounded; do not index the checkout, rewrite benchmark sources, add an LSP server, or touch code generation.

## Decisions

- Extend existing `Dat Source` with pure `Trn format_source : Source → Source ⊕ Diagnostic*` in `mapal-syntax`. Validate syntax, use original token spans and comment gaps, then validate output tokens. An AST-only printer would lose comments; regex replacement would misinterpret strings and guard arrows.
- Canonical layout uses four spaces, multiline brace interiors, newline-separated statements and record fields, consistent punctuation/operator spacing, and one final newline for nonempty source. Retain meaningful source breaks/comments where appropriate; do not impose arbitrary wrapping inside strings.
- Add `Trn fmt` in `mapal-cli`: explicit file paths write in place, `--check` reports differences without writes, `--stdout` previews one file, and `-` reads stdin/writes stdout. Validate all requested inputs before writing. Separate this argument handling from compiler flags.
- `Trn format_buffer` in each existing editor plugin sends unsaved buffer contents through `Trm` stdin/stdout to the CLI process (`Loc`); stderr carries errors. Apply output only after success and when the buffer has not changed. Use argument arrays, never a shell command assembled from source or paths. Support a configured executable path.
- Document commands in agent guidance and editor/CLI documentation. Format only touched Mapal files; no repository-wide formatting migration.

Normative models and planned mappings precede code in `docs/components/{syntax,cli}/DESIGN.md`. FRAMEWORK §4.5 checks: source is present in the formatter process; all editor/CLI boundaries use typed text pipes; formatting is one shared transformation; adapters add transport rather than independent layout policies. Identity-on-tokens and idempotence are the composition laws.

## Risks / Trade-offs

- Context-sensitive guards or merged numeric/projection tokens → compare input/output token kinds and spellings and parse both; reject unsafe output.
- Comments lost or moved across code → preserve comment text/order and test trailing, standalone and delimiter-adjacent comments.
- Concurrent editor edits or process failure → return no edit and keep the buffer intact.
- Existing files differ from the new style → check only explicit paths; leave current benchmarks untouched.

## Migration Plan

Build/install the CLI, then reload the Neovim plugin or install the updated VSIX. Both plugins use the same CLI command. Existing compilation commands retain their behavior.
