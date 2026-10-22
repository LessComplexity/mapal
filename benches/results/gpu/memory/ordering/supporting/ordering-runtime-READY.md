# Ordering correctness pilot — upload-ready, GPU not executed

Use only the reviewed unchanged bundle and the default20 repetitions. This pilot tests observed ordinary values/statuses, with no clocks or tracing added. At64×64, even all200 correct processes are nondiagnostic for race absence and do not justify a source fix or certify universal ordering safety.

- Bundle: `/private/tmp/mapal-s56-gpu/ordering-runtime/bundle-rtx-5090-20261021`.
- Manifest SHA256: `141f7bdca06b2f770761b735b02ae12f6891ba80cb9b461ec88fcd59780209e8`.
- Upload archive: `/private/tmp/mapal-s56-gpu/ordering-runtime/ordering-runtime-rtx-5090-20261021.tgz`.
- Archive SHA256: `2892f59e49b11cfbc46f54ad355aa8b9bd010073787dfe1ace112ad734e900ce`.
- Archive:6,234,381 bytes; exactly134 regular files (all133 manifest-bound files plus manifest), no special/link members. Extraction creates `ordering-runtime-bundle/`.

Local preparation completed30 saved-LLVM lowering commands into NEW host objects/optimized LLVM/PTX:5 preserved cases × Exact/Contract ×3 commands. No source re-emission, Cargo, existing-binary rebuild/replacement, GPU, rental or upload occurred. All ten objects are ELF64 x86-64; every host kernel/generator ID joins to a bound sm_120 PTX entry. Both freeze snapshots and the final preparation snapshot match HEAD b4152c5 and all328 source files,4 frozen inputs and3 LLVM tools in the original frozen manifest.

Five `test_kit.py` CPU-only tests pass and payload check-only passes. A separate `test_result_verifier.py` CPU-only fake-tool test completed200 fresh Python stub processes, then proved post-retrieval rejection of forged success containing a nonzero exit, wrong/truncated output, duplicate records or missing records. These are explicitly CPU test results, not ordinary GPU observations. The reviewed bundled runner is unchanged: its pass is for the requested repetition count; baseline acceptance additionally requires20 repetitions and exactly200 unique process records, enforced by the separate post-retrieval verifier.

After separate admission/upload, check then execute on the authorized single-RTX5090 Linux x86-64 host (CUDA headers/driver stub, real driver, cc/objdump/Python3 required):

```sh
python3 /w/ordering-runtime-bundle/box_ordering.py /w/ordering-runtime-bundle /w/unused --check-only
python3 /w/ordering-runtime-bundle/box_ordering.py /w/ordering-runtime-bundle /w/ordering-runtime-results-001 --reps 20 --timeout 60
```

The runner links all ten executables and captures disassembly before any GPU program. Any failed link/disassembly blocks GPU cases. Each ordinary fresh process has exact complete stdout, empty stderr and zero-status gates; the loop preserves all failures rather than skipping cases. Keep all outputs and linked binaries. Profiler/debugger/injected-sync execution is excluded; any later traced diagnostic needs separate results and scope.

After retrieving the complete result directory, run the local CPU-only baseline verifier; save its stdout outside the retrieved directory:

```sh
python3 /private/tmp/mapal-s56-gpu/ordering-runtime/verify_results.py \
  /private/tmp/mapal-s56-gpu/ordering-runtime/bundle-rtx-5090-20261021 \
  /ABS/RETRIEVED/ordering-runtime-results-001 \
  --manifest-sha 141f7bdca06b2f770761b735b02ae12f6891ba80cb9b461ec88fcd59780209e8 \
  > /ABS/OUTSIDE-RESULTS/ordering-runtime-verification.json
```

The five preserved sources/oracles are unchanged. `materialized-raw` and the second-chain child in `forked-tile` exercise demonstrated missing explicit edges. The same-stream child, omitted-raw real-free barriers, rowfold producer-copy subset and all-legacy base-rowfold preserve the previously established controls; member-rowfold's other host read remains a separate concern. No DCE variant, larger fixture or tolerance relaxation is included. Optional `make_larger.py` remains unexecuted and unadmitted.
