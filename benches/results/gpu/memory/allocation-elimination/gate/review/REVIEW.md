What this change does: This future gate compares exact observed outputs before/after the allocation change, then separately checks traced allocation requests. It makes no timing or universal race-safety claim.

The rejected v1 had two blockers: detached subprocesses survived outer timeout/SIGTERM, and an output path inside the packet could mutate its inventory before execution. Both are fixed in v2.

- Cancellation: `packet-v2/run_gate.py:13` propagates TERM with bounded grace; `packet-v2/box_ordering.py:33` kills and waits its active child. Independent real-process tests pass for phase timeout and actual outer CLI SIGTERM.
- Paths: `packet-v2/run_gate.py:20` resolves and rejects overlapping paths before validation or writes. Equal, nested and symlink cases pass independent negative checks.
- Prerequisites: `packet-v2/box_ordering.py:142` checks cc/objdump, and lines155–181 require all16 CUDA/runtime links and disassemblies before a GPU fixture. This packet invokes neither clang nor `/usr/bin/time`.
- Evidence: exact output/status/file bindings are rechecked by `verify_results.py:21`; traced counters run only afterward (`run_gate.py:50`, `counters.py:23`) and stay in separate outputs. Parser checks ordered requests, bytes and free/total counts (`counters.py:8`). Only documented corner observations are checked; no whole-array or speedup inference is admitted.

Reproduce:

```sh
python3 /private/tmp/mapal-s58-allocation-gate-review/check_packet.py /private/tmp/mapal-s58-allocation-gate/packet-v2 --expect-fixed --out /private/tmp/mapal-s58-allocation-gate-review/v2-independent-reproductions.json
python3 /private/tmp/mapal-s58-allocation-gate-review/check_sigterm.py /private/tmp/mapal-s58-allocation-gate/packet-v2 --expect-fixed --out /private/tmp/mapal-s58-allocation-gate-review/v2-sigterm.json
PYTHONDONTWRITEBYTECODE=1 python3 /private/tmp/mapal-s58-allocation-gate/packet-v2/run_gate.py /private/tmp/mapal-s58-allocation-gate/packet-v2 /private/tmp/mapal-s58-allocation-gate-review/unused-v2 --check-only
```

Manifest SHA256: `32d618492a233f3c32708cc8c028a752dd4997413529e487ac4d0f07c8d18a87`.

Verdict: No remaining blocker for preserving the reviewed future packet. This does not authorize hardware execution.

Not checked: actual Linux/CUDA linking, GPU outputs or runtime counters. No GPU/network/rental work occurred.
