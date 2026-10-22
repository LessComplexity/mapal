# cli — implementation map

> The functor DESIGN.md ("Categorical model") → code.

## Objects (Dat) → code
| Object | Form / shape | Realized at | State |
| --- | --- | --- | --- |
| `Source` | file text | `crates/mapal-cli/src/main.rs:frontend` | built |
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

## Notes / divergences
- `libmapal_rt.a` is found via `MAPAL_RT` or next to the binary. cargo uplifts a
  staticlib only when it is built as a root (checked S62: as a dependency of
  mapal-cli it stays in `deps/` under a hash), so the build is
  `cargo build -p mapal-cli -p mapal-rt`.
