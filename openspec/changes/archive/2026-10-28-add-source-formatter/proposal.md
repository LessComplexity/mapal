## Why

Mapal has highlighting in Neovim and VS Code but no shared source formatter. Hand-written and agent-written programs need consistent indentation, brace layout and spacing without changing program meaning.

## What Changes

- Add `mapal fmt` for explicit files, stdin/stdout, and a non-writing `--check` mode.
- Use four spaces, multiline brace bodies, statement line breaks and consistent token spacing; preserve comments and literal spelling and reject invalid input without writing it.
- Connect both existing editor plugins to the same CLI using buffer text on stdin.
- Document formatting commands for contributors and agents. Do not reformat existing benchmarks in this change.

## Capabilities

### New Capabilities

- `source-formatting`: deterministic source formatting through the CLI and both editor plugins.

### Modified Capabilities

None.

## Impact

`mapal-syntax` supplies the pure formatting operation over existing tokens; `mapal-cli` owns file and stream effects. Existing Neovim and VS Code plugins gain formatting adapters. No compiler lowering, optimization or code generation changes, new runtime dependency, or whole-repository scan is needed.
