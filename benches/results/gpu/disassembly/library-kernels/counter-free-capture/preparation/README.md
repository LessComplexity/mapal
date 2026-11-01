# Executed GPU library images

This counter-free diagnostic prototype captures actual loaded CUDA images and attributes every internal kernel launched by one already admitted Mapal/cuBLAS/CUTLASS call. Ordinary timing comes from a separate uninstrumented run. GPU admission is pending.

Read CONTRACT.md for supported paths, exact attribution rules, caps, and explicit unsupported cases. HANDOFF.md in the preparation root records local review state and exact resume commands.

The payload contains no installer or remote control. It needs Linux, CUDA12.8 development/CUPTI packages in the documented targets/x86_64-linux layout, c++, nvdisasm and an already admitted ordinary binary/source/oracle packet. It adds only the existing diagnostic --profile-leg selector to the admitted argv; it never rebuilds the benchmark or changes math options.

After packet-integrity verification, build a request from actual immutable ordinary results:

```sh
python3 check_packet.py
python3 build_request.py --packet /w/competitive-reference \
 --ordinary /w/trace/competitive-ordinary \
 --case rtx-5090__timing__deepseek_expert_m16_n2048_k7168__contract \
 --out /w/trace/library-image-request.json
python3 run_capture.py --request /w/trace/library-image-request.json \
 --cuda-home /usr/local/cuda-12.8 --out /w/trace/library-images --check-only
```

Original absolute runtime paths inside ordinary receipts must be preserved when staging the exact retained binary. A new diagnostic host does not prove what instructions ran on an earlier timed host. Keep host/driver/library identities separate.

A root-reviewed live pilot uses the same command without --check-only, with `--total-timeout 480`. Subprocess waits share a480second deadline, plus bounded cleanup; each profile process has a120second cap. Aggregate output admission is checked at512MiB after each command, with128MiB per-file RLIMIT_FSIZE; this is not an instantaneous aggregate disk quota. Mapal is the mandatory first smoke. A failed smoke skips both libraries; later failure preserves previous successful legs. All diagnostic durations remain inadmissible for ordinary timing.

Outputs include summary.json, exact request/command hashes, build/tool receipts, per-leg raw stdout/stderr, observer events.tsv, exact callback-saved module bytes, pre/post numeric reports, attribution JSON, and exact-function nvdisasm output. Nonzero exit, unknown dispatch, missing/ambiguous identity, drop/pending buffer, numerical/protocol failure, cap, timeout or cancellation produces non-admission. Raw artifacts are retained.

Local CPU checks use the actual callback implementation and pinned official headers, plus independent strict parser mutations and real subprocess lifecycle functions. They establish protocol behavior only. They do not establish Linux injection, live CUPTI getter/FUNCTION support, library-image attribution or complete decoder compatibility. Exact source/test hashes and independent review receipts are sealed in the manifest/evidence.
