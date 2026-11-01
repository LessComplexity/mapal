# API activity domain delta independent review

PASS for final parser ce12b781… with unchanged observer b1dd934a…. Previous review artifacts remain immutable; this receipt covers the fourth predicate discovered by full synthetic replay.

Kernel9's correlation may name a driver API or runtime API record. Equal numeric driver/runtime-parent IDs now produce explicit candidates, not a driver-priority branch. Each candidate must match the exact CBID/correlation, unique eligible callback invocation token across the process, successful callback and activity statuses and already validated parent/context interval. Exactly one domain must survive; both valid, missing/wrong identity, repeated eligible parents or contradictory status reject. The chosen API activity binding is retained in the launch report.

Eight independent actual-source CPU-fixture cases pass: different IDs with runtime binding; equal IDs with runtime-only or driver-only binding; and rejection of both valid domains, wrong runtime CBID, failed parent, ambiguous repeated parent and conflicting failed-driver activity. Owner53sourcecases+12bindingmutations bind the same bytes and pass. Full synthetic replay preserves3156callback rows,1799activity rows,5images and13kernel records, but supplies synthetic invocation storage; it is preparation only, not an admission of v1's original trace.

No real trace was altered or deduplicated. No GPU was run and no image/SASS leg is newly admitted. The original pilot remains0/3. Seal the final reviewed candidate before any separately authorized live execution.
