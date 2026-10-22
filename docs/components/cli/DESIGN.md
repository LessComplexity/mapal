# Component: cli

`mapal build|run|dump-ir|test` — a driver over the existing crates. It makes no
codegen decision: every optimization in its output is derived inside
`mapal-backend-llvm` from the IR artifacts against the named `TargetProfile`.
The CLI routes `EmitOpts` (the shared parser `mapal_backend_llvm::flags`, so
every rung's OFF arm is reachable from the product binary) and runs the
toolchain. Diagnostics are rendered here and only here (syntax I5 / C3).

## Categorical model (Dat + Trn)

Objects:

| Object | Meaning |
| --- | --- |
| `Source` | the `.mapal` text |
| `Program` | the parsed tree (`mapal_syntax::Program`) |
| `CategoryIr(lowered)` | sealed, checked IR — the oracle's input |
| `CategoryIr(rewritten)` | the compiled arm's input |
| `Emission{host, device?, profile}` | the translation unit(s) and the resolved profile |
| `Ptx` | the device unit after `opt -O3 ∘ llc -O3 -mcpu=<profile.gpu.arch> -nvptx-fma-level=0` |
| `Executable` | host unit + `libmapal_rt.a` (+ `mapal_gpu.c`, libcuda on a GPU profile) |
| `Observable(stdout?, exit)` | what a run shows: stdout byte for byte (absent on a trap) and the exit code |

Morphisms:

| Morphism | Signature |
| --- | --- |
| `parse` | `Source → Program ⊕ Diagnostic*` |
| `lower` | `Source × Program → CategoryIr(lowered) ⊕ Diagnostic*` |
| `check` | `Source × Program × CategoryIr → Diagnostic*` (empty = accept) |
| `rewrite` | `CategoryIr(lowered) → CategoryIr(rewritten)` (skipped by `--no-rewrite`) |
| `emit` | `CategoryIr × EmitOpts → Emission ⊕ EmitError` |
| `opt∘llc` | `Emission.device → Ptx` (GPU profile only) |
| `link` | `Emission.host (× Ptx) → Executable` |
| `exec` | `Executable (× MAPAL_PTX) → Observable` |
| `interp` | `CategoryIr(lowered) × Budget → Observable ⊕ Diverged` (`mapal_interp::run` then `native_observable`) |
| `render` | `Diagnostic → 𝕊` (`file:line:col severity[CODE] (stage) message`) |

`test` is the equalizer check `exec∘link∘emit∘rewrite ≡ interp` on `Observable`,
on the Exact face only (the contract and reduced faces are bound-checked, not
bit-equal, so `test` rejects them). A diverged oracle is no verdict (exit 1,
`SKIP`); a missing toolchain is exit 3, never PASS/FAIL.

Exit codes: 0 ok · 1 rejected program / failed tool / FAIL · 2 usage · 3 toolchain
missing (clang, opt/llc, libmapal_rt.a, CUDA) · otherwise the program's own
(101 trap, 102 GPU driver).
