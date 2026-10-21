## 1. Isolated preparation

- [x] 1.1 Verify pinned official CUPTI12.8 structures and existing harness path; retain header hashes and source references in `/private/tmp/mapal-s58-capture/evidence/header-manifest.json` and source README.
- [x] 1.2 Prepare bounded exact-PTX observer with no runtime/compiler changes or added synchronization; verify real-header syntax/object compilation and stub callback lifetime tests.
- [x] 1.3 Prepare fail-closed launcher/validator; verify happy joins and missing/corrupt/unlinked/oversized/overwrite/variant cases using `python3 /private/tmp/mapal-s58-capture/source/test_capture.py`.

## 2. Review and integration (not yet completed)

- [x] 2.1 Independently review prototype and integrate a concrete repository tooling path; verify reviewed diff and dependency boundary.
- [x] 2.2 Map every integrated morphism into component IMPLEMENTATION.md, reconcile ARCHITECTURE.md then component/root STATUS.md, and record section4.5 review; verify mapped symbols exist and docs describe actual integration.
- [ ] 2.3 Run supercharge drift, fix/record dead rows and write immutable session continuity; verify checks before archiving.

## 3. Separately authorized future GPU proof

- [ ] 3.1 Link against real Linux CUDA/CUPTI12.8 and smoke on the actual GPU in a separately budgeted proof run; verify callback layout, launch/completion joins, code hashes, tool/GPU/driver identity and actual nvdisasm function output.
- [ ] 3.2 Reject missing/deferred callbacks without modifying production loading/JIT behavior; preserve offline cubin and library-fatbin evidence as distinct variants, and verify instrumented outputs cannot enter ordinary timing admission.

S58 integration: `benches/gpu/capture_images/` and sanitized evidence under
`benches/results-s54-gpu/s58-executed-image-capture/`. Independent review reproduced
and then rejected wrong resource context and launch geometry. Fresh relocated real-header
compilation and eleven local test groups pass; GPU proof remains unexecuted.
