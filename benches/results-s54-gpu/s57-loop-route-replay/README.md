# S57 loop-route replay repair

Shared source now includes the reviewed generic repair. The full required-clang workspace
suite passes **1,181 / 0 / 1 ignored**; [exact receipt](evidence/shared-workspace-verification.json).
The first restricted run failed three native hardware-detection tests because sysctl was
denied. With host access, the unchanged source passes the full suite. Formatting passes.

The [isolated proof](PROOF.md) records the diagnosed acyclic exit-payload omission, five
negative-controlled regressions,86 passing rewrite tests and eight restored GPU emissions.
The [independent review](review-gpu-owner.md) found no blocker. Full interpreter output
matches8,408 numeric values/8,838 lines; this does not certify GPU execution or the large
native backend. Its three bounded native emission timeouts remain unresolved.

This directory is a compact exact-byte copy of proof and logs. The complete2448-file
reproduction archive is retained locally under
`.claude/worktrees/_archive-s54-20261021/s57-loop-route-replay/`; it includes fixtures,
saved host/device LLVM and temporary diagnostic tools. Frozen benchmark binaries and
payloads remain unchanged. [Copy hashes](copy-verification.json) bind both records.
