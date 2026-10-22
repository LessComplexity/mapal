## Why

S57 saved offline ptxas SASS, not the production driver-JIT image. A bounded observation packet must bind the exact authorized PTX, copied loaded image and successful launch submission before any future disassembly claim.

## What Changes

- Prepare an isolated observer and fail-closed validator for the existing benchmark's driver PTX path, without runtime/compiler changes or synchronization.
- Retain exact bytes, chronology, context/module/function identity and completed harness receipt; forbid timing admission from observed runs.
- Leave GPU validation and library-fatbin observation explicitly open.

## Capabilities

### New Capabilities
- `executed-cuda-image-evidence`: admission rules for a bounded driver-PTX observation packet.

### Modified Capabilities
None.

## Impact

Only change artifacts are added to the repository at preparation time. Prototype sources and stub evidence reside under `/private/tmp/mapal-s58-capture/`; integration and future GPU execution require root review. No production code or existing evidence is changed. Linux CUDA/CUPTI 12.8 is the future observation target; macOS permits real-header compilation and stub tests only.
