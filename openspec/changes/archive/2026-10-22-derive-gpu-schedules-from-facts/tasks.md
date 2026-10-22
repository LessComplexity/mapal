## 1. Prove and model the missing connection
- [x] 1.1 Audit existing IR geometry/order, hardware consumers and actual arithmetic fallback; exclude previously rejected geometry changes and intentional conservative resource policy.
- [x] 1.2 Record the planned Launch arithmetic invariant and tensor-to-scalar resolution in component architecture and this design before production edits.
- [x] 1.3 Add meaningful regressions that fail on the original policy/emission mismatch; preserve the pre-fix failure evidence.

## 2. Implement and verify
- [x] 2.1 Derive typed actual arithmetic on Launch and use it consistently in per-site policy and scalar emission, preserving module PDL, ordering, legacy Fill and forced-arm precedence.
- [x] 2.2 Cover strided/contiguous Exact/Contract, tensor type/layout/geometry/device/forced-off fallbacks and supported tensor controls; inspect emitted arithmetic and resources.
- [x] 2.3 Run relevant backend tests and independently review source and emitted-control changes; report hardware validation and performance limits explicitly.

## 3. Reconcile
- [x] 3.1 Reconcile implementation mappings, architecture, component/root status and FRAMEWORK review; retain audit and verification evidence in descriptive topic directories.
- [x] 3.2 Run strict OpenSpec/drift and write immutable handoff; archive only when the correction's required implementation checks pass.
