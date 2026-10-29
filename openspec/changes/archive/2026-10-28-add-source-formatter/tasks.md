## 1. Shared formatter

- [x] 1.1 Implement token-preserving source formatting in mapal-syntax; verify exact layouts, invalid-input refusal, comment preservation and idempotence with syntax tests.
- [x] 1.2 Add bounded generated-program and tracked source corpus checks for token equivalence, parse success and idempotence without modifying fixtures.

## 2. CLI and editors

- [x] 2.1 Add fmt file/check/stdin/stdout modes and useful diagnostics; verify subprocess tests including invalid batch no-write and existing dispatch compatibility.
- [x] 2.2 Add Neovim and VS Code formatting adapters and executable configuration; verify buffer success, error preservation and packaging through editor tests.
- [x] 2.3 Document contributor/agent and editor usage; verify commands against the built CLI and keep benchmark sources unchanged.

## 3. Reconciliation

- [x] 3.1 Reconcile syntax/CLI implementation mappings, design and status, root mappings/status and coherence review; verify symbol references and run drift check.
- [x] 3.2 Validate/archive OpenSpec and write immutable session handoff with checks and exact resume commands; verify no task processes remain.
