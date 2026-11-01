## 1. Isolated diagnosis

- [x] 1.1 Validate this plan; bind shared HEAD/source/tool/input hashes and create a git-archive snapshot plus isolated target directory.
- [x] 1.2 Instrument only temporary driver/loop replay, reproduce tiny674 and original5907, and bind actual pass/route/feeder provenance before selecting a fix.

## 2. Temporary repair and reviewable proof

- [x] 2.1 If diagnostics confirm a repair, implement the smallest generic correction and focused regressions in the temporary snapshot; otherwise revise the design with demonstrated cause.
- [x] 2.2 Negative-control the focused regression on unmodified code, then check identity/per-pass/full rewrite values/effects/traps and relevant loop boundaries on the corrected snapshot.
- [x] 2.3 Remove temporary tracing from the proposed patch, save exact commands/hashes/proof note, and independently confirm shared frozen inputs remain unchanged.

## 3. Shared integration, deferred

- [x] 3.1 Review and apply the demonstrated narrow patch to shared production source after the exact-baseline pilot requirement is satisfied; run appropriate host checks.
- [x] 3.2 Reconcile implementation mapping, architecture and component/root status; record §4.5 review and verification evidence.
- [x] 3.3 Run supercharge drift, resolve dead mappings, validate and archive only after shared integration is complete.
