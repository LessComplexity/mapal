# Component: cli

Status: implemented
Last updated: 2026-10-22 · S62
Spec references: architecture.md §5 (Tooling ecosystem — `mapal` CLI / build integration) + HANDOFF §6 (repository layout — the `mapal-cli` crate provides `mapal build|run|dump-ir|test`). Supporting: architecture.md §2.3 (compiler driver); architecture.md §7.1 (typical dev cycle / `mapal build`).
Depends on: syntax, ir, lower, check, interp, rewrite, backend-llvm Depended on by: (none)

## What works
- `mapal build <f> [-o exe] [--emit-only]`: host `.ll` + link; on a GPU profile also `.dev.ll` and `.ptx` (`-mcpu` from the profile's `arch`).
- `mapal run <f>` (compile + execute; exit code passes through) and `mapal run --interp <f>` (the oracle; 101 on a trap, as the compiled protocol).
- `mapal dump-ir <f> --stage=lowered|rewritten|llvm|device`.
- `mapal test <f>`: compiled (rewritten, Exact, `-O2`) vs the interpreter, exit code and stdout byte for byte.
- Every emitter flag and rung arm (`--target=`, `--gpu-off=`, `--gpu-on=`, `--move-panel=`, ...) through `mapal_backend_llvm::flags`.

- GPU builds under `hostopt` (`<gpu>+<host>` profile name) resolve the device arch from the GPU half (fixed at integration; before, a panic).

## What does not / known issues
- No CUDA on the host: a GPU build writes `.ll`/`.dev.ll`/`.ptx` and exits 3.
- `dump-ir --stage=lowered|rewritten` is `Debug` text, no JSON.

## Invariants enforced (and where in code)
- Diagnostics are rendered only here (syntax I5 / C3): `main.rs:frontend`.
- `test` judges the Exact face only; `--contract` / `--precision=` exit 2.

## Test coverage (golden / property / differential / skipped+why)
- `crates/mapal-cli/tests/examples.rs`: 12 examples `test` PASS at `--target=generic`; trap exit 101 both ways; `dump-ir` emits `@main`; `rtx-4090` build writes the device unit and `.target sm_89` PTX. `matmul128`/`matmul4096` absent (oracle budget in debug), `vector` absent (P0001, not Core).

## Performance notes (numbers + bench name + date; regressions flagged)
None — a tooling item; no codegen change.

## Open questions (→ ADR candidates)
