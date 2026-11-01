## Why

S54's integrated GPU validation stopped after an arena correctness fix; the saved prefix results predate that fix and contain failures. Three shipped selection decisions also conflict with independent verification, so the final tables cannot yet support release claims.

## What Changes

- Add a differential-only GPU smoke mode that reports failure reliably and skips performance work; propagate remote failures through the rental wrapper and retain cleanup.
- Validate result inventory and actual correctness gates before accepting performance data; make forced-ON retain-resident measurable alongside OFF.
- Rebuild current sources, smoke the previously failing cases, then resolve RTX 5090 streams, H100 retain-resident and raster selections from integrated A/B evidence.
- Complete five-profile measurements and independent RTX 4090/5090 confirmation runs, using H100 sm_90a and pinned CUTLASS.
- Publish both S54 reports with losses, uncertainty and provenance; reconcile architecture mappings/status and session continuation.

## Capabilities

### New Capabilities
- `gpu-validation`: a GPU verification run reports success only when its requested correctness scope completed and passed.

### Modified Capabilities
None.

## Impact

GPU benchmark scripts and validation tools; measured GPU rung selection rows in LLVM's TargetProfile; backend-nvptx living docs and S54 reports. Vast rentals are temporary and destroyed after results retrieval. Existing unrelated rentals are outside scope. Numerical precision policy and ADR-0039 remain unchanged.
