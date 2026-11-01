## Why
S57 proves the ordered M16 FMA fallback is 6.07× and 8.02× slower than pinned cuBLAS on two projection families. Existing cooperative tile emission can preserve each output’s ordered fold while fitting its thread grid to skinny shapes.

## What Changes
- Experiment with shape-derived redistribution of the existing GPU tile thread grid, retaining the existing panel, resource and divisibility checks.
- Verify all three projection families at M1/M16/M128 and unchanged square controls, then accept only measured value-gated improvements.
- Preserve separate multiply/add versus FMA contracts and all existing fallback, alias, bounds and epilogue behavior.

## Capabilities
### New Capabilities
None.
### Modified Capabilities
None; this is a placement optimization with no observable contract change. `skip_specs: true` records that choice.

## Impact
Planned production touch site: `crates/backends/llvm/src/profile.rs:TargetProfile::gpu_tile`, plus its unit tests. Existing `crates/backends/llvm/src/func/gpu.rs:Kernel::tiled` remains the sole cooperative emitter. GPU measurements and integration are coordinated by the parent agent; implementation occurs only in an isolated git-archive snapshot until review.
