# Arithmetic-aware GPU selection evidence

[Report](../../../../../docs/performance/gpu/disassembly/fact-driven-scheduling.md)

The source and hardware audits precede the correction. `pre-fix.log` records the two original behavioral failures. `verification-receipt.json` records exact commands, tool/source hashes, tests and limits; `independent-review/` contains the separate source and raw-emission comparison.

`emission-evidence.tar.gz` contains source programs and all before/after host LLVM, device LLVM, optimized LLVM and PTX. `emission-manifest.json` binds each member. Historical temporary paths and source line positions remain in the original receipts.

No GPU execution or performance evidence is supplied by these compile-time checks.
