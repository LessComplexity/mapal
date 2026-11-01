# cli — implementation map

> The functor DESIGN.md ("Categorical model") → code.

## Objects (Dat) → code
| Object | Form / shape | Realized at | State |
| --- | --- | --- | --- |
| `Source` | file text | `crates/mapal-cli/src/main.rs:frontend` | built |
| formatting `Source` | original and formatted UTF-8 text, explicit file name | `crates/mapal-cli/src/fmt.rs:Input` | built |
| `Program` | `mapal_syntax::Program` | `crates/mapal-cli/src/main.rs:frontend` | built |
| `CategoryIr(lowered)` | `mapal_ir::CategoryIr` | `crates/mapal-cli/src/main.rs:frontend` | built |
| `CategoryIr(rewritten)` | `mapal_ir::CategoryIr` | `crates/mapal-cli/src/main.rs:rewritten` | built |
| `Emission` | `mapal_backend_llvm::Emission` → `<exe>.ll`, `<exe>.dev.ll` | `crates/mapal-cli/src/main.rs:build` | built |
| `Ptx` | `<exe>.ptx` | `crates/mapal-cli/src/main.rs:build` | built |
| `Executable` | `<exe>` (temp dir for `run`/`test`) | `crates/mapal-cli/src/main.rs:build` · `in_temp` | built |
| `Observable` | `(Option<String>, i32)` | `crates/mapal-interp/src/lib.rs:native_observable` · `crates/mapal-cli/src/main.rs:wait_boxed` | built |

## Morphisms (Trn / relations) → code
| Morphism | Signature | Realizing code | State |
| --- | --- | --- | --- |
| `parse` / `lower` / `check` | `Source → CategoryIr ⊕ Diagnostic*` | `crates/mapal-cli/src/main.rs:frontend` | built |
| `rewrite` | `CategoryIr → CategoryIr` | `crates/mapal-cli/src/main.rs:rewritten` | built |
| `emit` | `CategoryIr × EmitOpts → Emission` | `crates/mapal-cli/src/main.rs:emit` (flags: `crates/backends/llvm/src/flags.rs:apply`) | built |
| `opt∘llc` | `device → Ptx` | `crates/mapal-cli/src/main.rs:build` | built |
| `link` | `host (× Ptx) → Executable` | `crates/mapal-cli/src/main.rs:build` | built |
| `exec` | `Executable → Observable` | `crates/mapal-cli/src/main.rs:run_cmd` · `test_cmd` | built |
| `interp` | `CategoryIr × Budget → Observable ⊕ Diverged` | `crates/mapal-cli/src/main.rs:run_cmd` (`--interp`) | built |
| `render` | `Diagnostic → 𝕊` | `crates/mapal-cli/src/main.rs:frontend` (`report`) | built |
| equalizer `test` | `exec∘link∘emit∘rewrite ≡ interp` | `crates/mapal-cli/src/main.rs:test_cmd` | built |
| `fmt` | `Source × Mode → Source ⊕ Diagnostic*` | `crates/mapal-cli/src/fmt.rs:run` | built |
| file replacement | `Source × Path → File ⊕ Error` | `crates/mapal-cli/src/fmt.rs:replace_file` | built |
| Neovim `format_buffer` | `EditorBuffer → EditorBuffer ⊕ Error` | `editors/nvim/lua/mapal/format.lua:format` | built |
| VS Code `format_buffer` | `EditorBuffer → TextEdit* ⊕ Error` | `editors/vscode/extension.js:createProvider` | built |
| VS Code source transport | `Source → Source ⊕ Error` through CLI stdin/stdout | `editors/vscode/extension.js:formatText` | built |

## Formatting placements and checks

The pure syntax formatter executes in the CLI process. Explicit file reads and
atomic replacement use the filesystem port; stdin/stdout transport unsaved buffer
text between editor and CLI processes. Stderr transports diagnostics. No compiler
toolchain, LSP, backend, or GPU is invoked. `--check` and `--stdout` never write.
Neovim `changedtick` and VS Code document-version checks reject stale output;
both adapters bound subprocess time/output and keep the buffer on errors.

`crates/mapal-cli/tests/format.rs` verifies all modes, invalid/unreadable batch
no-write, atomic permissions/symlink preservation and original interpreter dispatch.
Editor tests exercise actual CLI output as well as missing executable, parse
failure, no-op and stale-document behavior. No existing benchmark was reformatted.

## Notes / divergences
- `libmapal_rt.a` is found via `MAPAL_RT` or next to the binary. cargo uplifts a
  staticlib only when it is built as a root (checked S62: as a dependency of
  mapal-cli it stays in `deps/` under a hash), so the build is
  `cargo build -p mapal-cli -p mapal-rt`.
- A GPU build under `hostopt` carries the profile name `<gpu>+<host>`; `build` resolves the
  device arch from the GPU half (`crates/mapal-cli/src/main.rs:build`, integration fix on
  `s62-integrated`). The GPU test passes `--gpu-off=place` because `place` keeps the 64³ site on the host.
