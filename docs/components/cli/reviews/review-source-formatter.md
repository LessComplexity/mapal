# Source formatter — coherence and verification

2026-10-28. Scope: shared syntax formatting, CLI file/stream modes, Neovim and
VS Code adapters. No lowering, codegen, runtime or GPU changes.

## FRAMEWORK §4.5 checklist

- [x] Placement honesty — the pure formatter receives complete source in the CLI
  process; editor adapters deliver unsaved buffer text through stdin.
- [x] Transmission well-typing — source and output travel over text pipes;
  diagnostics use stderr. File replacement crosses the filesystem boundary.
- [x] Placement totality — syntax owns layout, CLI owns I/O, editors own buffer
  replacement. Their code mappings are in the component IMPLEMENTATION files.
- [x] Dependency mediation — editor-to-CLI calls use argument arrays and pipes,
  never shell-interpolated source. No external compiler toolchain is required.
- [x] Composition soundness — CLI and editors share `format_source`; every
  successful output reparses and preserves token kinds/spelling. Idempotence is
  checked across 192 generated whitespace cases and 205 valid source files.
- [x] runsAt is a relation — the pure library is called in both CLI and test
  processes; no singleton physical placement is encoded in syntax data.

## Independent review and fixes

The bounded CLI review found duplicate path/alias handling and temporary file
permission ordering. Both are fixed: already-formatted destinations are no-ops;
permissions are set before source bytes are written to the temporary file.
CLI tests cover repeated paths and a symlink alias in the same batch.

The bounded formatter review found that corpus tests could silently skip newly
invalid files. Only the existing `examples/vector.mapal` rejection is now allowed.
Parameter indentation also now recognizes actual map/fold headers, including
precision annotations, rather than guessing from the first body expression.

## Evidence

| Check | Result |
| --- | --- |
| `CARGO_BUILD_JOBS=4 RUST_TEST_THREADS=4 cargo test -p mapal-syntax` | 215 pass, including 12 formatter tests |
| `CARGO_BUILD_JOBS=4 RUST_TEST_THREADS=4 cargo test -p mapal-cli --test format` | 8 pass; toolchain-free modes, errors/no-write, file permissions/symlinks and original interpreter dispatch |
| `NVIM_LOG_FILE=/tmp/mapal-format-nvim.log sh editors/test.sh` | Neovim/VS Code formatting integration and 29/32 highlighting assertions pass |
| Rust formatting on changed Rust files | clean |
| `openspec validate add-source-formatter --strict --json` | valid, no issues |
| supercharge drift | 0 dead / 701 refs; temporary index includes new source paths, real index unchanged |
| `gen_16_8_8.mapal` | read-only preview; original SHA-1 remains `0d581f1f9bedd344c0466510f0183dc5b2e93c8a` |

No performance claim is made. Syntax testing reads only explicit, nonrecursive
example/algorithm directories; no Semble or graph refresh runs after the reported
indexing memory incident. VS Code behavior is checked through its provider with a
small API test double and real CLI, not a downloaded editor UI.

## Model delta actually shipped

Existing source/token data gains a pure formatting transformation; there is no
second AST or editor-specific layout policy. Atomic source replacement is separate
from the formatter. Comments/literal spelling and deliberate line/paragraph breaks
are preserved; layout does not introduce width-based expression reflow. Invalid or
out-of-Core syntax is reported rather than rewritten.
