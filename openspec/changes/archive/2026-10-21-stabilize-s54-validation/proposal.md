## Why

The LLVM and CUDA example emitters currently share Cargo's `emit` output path, so a workspace build can overwrite the executable used by GPU bundles. A duplicated test attribute also runs one profile check twice while leaving its neighboring check unregistered. S54 needs unambiguous build artifacts and a trustworthy local baseline before final GPU measurements.

## What Changes

- **BREAKING (developer example command):** give the CUDA example a unique `emit_cuda` name and update active consumers; retain LLVM's `emit` name for the current GPU bundle path.
- Register each of the two profile checks once.
- Initialize OpenSpec tracking and repair the six known stale implementation references so the validation change can pass drift before archive.
- Verify both emitter artifacts, required-clang workspace tests, and the documentation mappings.
- Fix the reproducible constant-folding trap-preservation defect found by the full validation run; retain the generated counterexample and add a focused regression.

## Capabilities

### New Capabilities
- `emitter-examples`: backend-specific developer example executables remain independently addressable after workspace builds.
- `rewrite-equivalence`: optimization preserves successful results and whether evaluation traps, including guarded computations whose dependencies simplify.

### Modified Capabilities
None; this is the first OpenSpec contract in the repository.

## Impact

CUDA example target and active scripts/docs that invoke it; LLVM profile test attributes; existing implementation-map references; the rewriter's existing trap-preservation contract. GPU selection, public Rust APIs and dependencies remain unchanged.
