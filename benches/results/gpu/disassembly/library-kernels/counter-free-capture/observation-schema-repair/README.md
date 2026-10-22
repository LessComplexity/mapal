# Reviewed observation schema repair

The first live trace remains rejected in `../rejected-correlation/`. This directory
records four source-supported observation repairs: unique callback invocation identity,
bounded ELF helpers, provisional context metadata and explicit API activity-domain binding.
Production changes are confined to the standalone observer's `capture.cpp` and `admit.py`.

[Repair and proof](REPAIR.md) · [Independent final delta review](independent-review/DOMAIN-DELTA-REVIEW.md)

53 callback/parser cases,12 identity mutations,10 integration checks and the complete
synthetic replay pass. Independent reviews add9 callback/context,8 ELF-range and8 activity
domain cases. The full replay preserves original event fields but supplies new CPU-owned
invocation storage: it is compatibility evidence, never retrospective image admission.

The sealed source archive and manifest are bound in `recovery.json` and `seal-v2.json`.
All198 payload files and199 archive files are independently rehashed. Frozen benchmark
binaries, input/numerical contracts, CLI, prerequisites, runner and resource caps stay
unchanged. A fresh live smoke remains required before admitting library images or SASS.
