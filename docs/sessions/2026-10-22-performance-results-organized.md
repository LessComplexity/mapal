# Performance reports and evidence organized

Checkpoint recorded 2026-10-22 after the organization review. The GPU comparison continues; this log does not claim that the cuBLAS gap is closed. See the [earlier execution checkpoint](2026-10-22-gpu-comparison-and-evidence-organization.md) for the admitted measurements and integration details, and [live continuity](../next-session.md) for changing state.

## Organization completed

The user requested descriptive titles and separate topic folders. Readable reports now start at [the performance index](../performance/README.md), with [current GPU results](../performance/gpu/current-results.md), then GPU kernels/workloads/memory/disassembly/validation, CPU topics, compiler cost, measurement methodology and matrix history. Raw evidence starts at [the evidence index](../../benches/results/README.md). The historical matrix chronology is preserved behind a concise experiment index.

The path migration covers 2,849 files. All 2,802 raw-file SHA256 hashes remain unchanged; no historical session log was edited. The [path map](../../benches/results/path-map.json) resolves original captured paths. There are zero session-number filenames beneath either performance reports or benchmark results. Raw labels remain provenance. Current comparison tables use actual tile geometries and panel depths; paired timings identify their shapes. Topic indexes distinguish admitted timing, diagnostics, preparation and historical evidence.

Final checks: 372 local report links resolve, zero broken links, all 2,802 raw hashes match; strict OpenSpec 7 pass / 0 fail. The real GPU baseline generator is equivalent except its descriptive title; NN and packing report outputs are byte-identical after relocation. Shell/Python syntax checks and cargo fmt passed earlier in this session. No broader code test was repeated for the final prose changes. The independently validated allocation correction and its 1,190 distinct local passing tests are included in the checkpoint; no allocation latency gain is claimed.

The read-only navigation review found ambiguous candidate names, generic scope columns, unlabelled paired shapes and competing “current” entry points. Root corrected those without changing measured values or captured artifacts. Review evidence remains under `/private/tmp/mapal-s61-gap-analysis/navigation-review/`.

## Live work at this checkpoint

- Branch `s54-gpu-fusion`, previous committed head `29ad618`; root is saving a local checkpoint. No push. Preserve unrelated `oainotes.md`.
- Newly owned instance **57256816**, machine **56076 / host 391071**, created **18:49:03.684Z** with deadline **19:44:03.684Z**. Ownership guard passed **18:49:04.710Z**. Launcher session **43594**, directory `/private/tmp/mapal-s61-gpu/attempt-09/`. Provider startup is pending; no new timing admitted.
- All eight earlier task-owned rentals are independently absent, including latest failed startup instance **57256063**, absent **18:43:05.966Z**. Every pre-existing instance remains protected. Only an exact own create-response ID plus physical match and prelaunch exclusion authorizes task cleanup.
- The operator performed read-only diagnosis after repeated startup failures. Installed-CLI offline evidence proves adding `--cancel-unavail` changes only that field, requesting explicit scheduling rejection instead of a stopped contract. This is the sole request change. Payload and guard hashes are unchanged; no underlying provider/account cause or causal repair is claimed. Account/API data and raw errors stay private.
- Root bound packet **9786978505e5a580f7c11f5cd650ced54ea9e308fcf1807e870cd6856d885e1e** after independently rehashing every staged file and comparing the launcher/wrapper. Competitive CUTLASS timing runs before the separate selected-binary NCU diagnostic. User upload/equivalent-host authorization and removed monetary caps persist.
- `/root/composed_analysis_s61` owns fused/unfused cuBLASLt/CUTLASS reference preparation in `/private/tmp/mapal-composed-fused-references/`. Source review found no inspected layout, bias-stride, epilogue or timing semantic blocker. An independent checker audit exposed request/candidate/config/nonce admission holes; corrections and cross-process opaque-algorithm binding are being verified. No CUDA compile or GPU result for this new packet.
- `/root/gap_analysis_s61` independently verifies those identity corrections and real subprocess protocol checks. Root owns shared documentation and integration.

## Tool state and open work

Code-only `graphify update .` succeeds: 9,319 nodes and 19,103 edges. Full semantic document extraction could not run without an LLM API key; no document-graph refresh is claimed. `gbrain capture` of the earlier checkpoint timed out on the existing PGLite lock; the lock was not removed. Repository session files remain authoritative.

OpenSpec remains in progress: benchmark-large-composed-gpu-workloads **5/15**, optimize-ordered-smallm-gpu **4/7**, eliminate-unmaterialized-gpu-allocations **11/15**, capture-executed-cuda-images **8/11**. Do not archive incomplete experiments. Next: retrieve and independently admit competitive timing, actual library SASS/counters, then supported fused-library comparison. Continue to distinguish ordered bits, numerical-budget admission, ordinary timing, static instructions and diagnostic counters.

## Inspect and resume

```sh
cd /Volumes/LessComplex/Personal/mapal
git status --short
openspec list --json
cat docs/performance/README.md
cat docs/next-session.md
cat /private/tmp/mapal-s61-root/live-state.json
cat /private/tmp/mapal-s61-gpu/attempt-09/packet.json
```

Do not print raw provider/account/create/wrapper logs. The operator owns retrieval and exact-instance cleanup. If the current attempt has finished, inspect its sanitized `results/cleanup.json` before starting another rental. Public new reports/results must use descriptive experiment names; immutable captured receipts and historical logs retain provenance.
