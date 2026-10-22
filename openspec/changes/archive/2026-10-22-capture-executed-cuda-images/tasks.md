## 1. Isolated preparation

- [x] 1.1 Verify pinned official CUPTI12.8 structures and existing harness path; retain header hashes and source references in `/private/tmp/mapal-s58-capture/evidence/header-manifest.json` and source README.
- [x] 1.2 Prepare bounded exact-PTX observer with no runtime/compiler changes or added synchronization; verify real-header syntax/object compilation and stub callback lifetime tests.
- [x] 1.3 Prepare fail-closed launcher/validator; verify happy joins and missing/corrupt/unlinked/oversized/overwrite/variant cases using `python3 /private/tmp/mapal-s58-capture/source/test_capture.py`.

## 2. Review and integration (not yet completed)

- [x] 2.1 Independently review prototype and integrate a concrete repository tooling path; verify reviewed diff and dependency boundary.
- [x] 2.2 Map every integrated morphism into component IMPLEMENTATION.md, reconcile ARCHITECTURE.md then component/root STATUS.md, and record section4.5 review; verify mapped symbols exist and docs describe actual integration.
- [x] 2.3 Run supercharge drift, fix/record dead rows and write immutable session continuity; verify checks before archiving.

## 3. Separately authorized future GPU proof

- [x] 3.1 Link against real Linux CUDA/CUPTI12.8 and smoke on the actual GPU in a separately budgeted proof run; verify callback layout, launch/completion joins, code hashes, tool/GPU/driver identity and actual nvdisasm function output.
- [x] 3.2 Reject missing/deferred callbacks without modifying production loading/JIT behavior; preserve offline cubin and library-fatbin evidence as distinct variants, and verify instrumented outputs cannot enter ordinary timing admission.

S58 integration: `benches/gpu/capture_images/` and sanitized evidence under
`benches/results/gpu/disassembly/capture-tooling/`. Independent review reproduced
and then rejected wrong resource context and launch geometry. Fresh relocated real-header
compilation and eleven local test groups pass; GPU proof remains unexecuted.
The later optional hardware phase correctly reports unvalidated because its prerequisite
ordinary small-M run was incomplete; no callback/cubin/SASS claim follows from its exit0.
S58 closes local continuity with drift 0 dead / 571 refs, strict OpenSpec 7/0 and
`docs/sessions/2026-10-21-s58-parallel-proof-and-validation.md`. That S58 status is historical. S61 now independently admits four real Linux/GPU
old/S58 Contract captures; exact PTX, context/module/function/launch, callback ELF and
exact-entry SASS join, and instrumented timing remains inadmissible. Full 924-input and
69-output inventories rehash. The library extension below keeps this change in flight.

## 4. User-authorized library comparison extension

- [x] 4.1 After live Mapal observer admission, prepare exact library module/function/dispatch joins for cuBLAS and CUTLASS with leg markers; reject ambiguity, unsupported variants, dropped records and missing completion.
- [x] 4.2 Independently verify captured functions and disassemble the actual executed entries; keep static embedded library images and runtime-selected images distinct.
- [x] 4.3 Compare instructions/resources and separate diagnostic counters, then map each evidenced difference to existing IR facts, their consumers, backend realization choices or an explicit unknown; test proposed causal explanations before changing compiler policy.

The first approved counter-free extension pilot is preserved with zero image/SASS
admissions. Its Mapal process and1536 exported pre/post numeric words pass; library legs
are skipped after callback identity rejection. Before completing4.1, repair the three
source-proved observation boundaries in `design.md` (invocation identity, executable
helper ranges and provisional context metadata), test actual source against positive and
negative fixtures, independently review and reseal. Do not patch or deduplicate the failed
trace. All11 task rentals are closed; the latest exact-owned57265999 is independently
absent at2026-10-22T20:05:14.535Z.

The four proved observation defects are repaired in a sealed standalone v2, with53 source
cases,12 identity mutations,10 integration checks and full actual-source synthetic replay
passing. Independent review passes; the original trace remains rejected. The user then
explicitly approved the repaired bundle and equivalent hosts; normal upload approval
passed. Attempt12 encountered an unavailable offer before creation. Fresh capture and
independent admission then succeed on attempt13. All three legs and four selected entries
are admitted;4,608 numerical words and504 recovered files are independently checked.
The decoded cuBLAS GEMM plus reduction, ordered independent-output Mapal and shared-memory
CUTLASS entries are compared with instruction anchors and queried resource limits.
Unknown decoder annotations and unavailable cuBLAS register attributes remain explicit.
No counters, new ordinary timings or causal policy changes are claimed. The report maps
each observed difference to existing facts and a controlled follow-up experiment; those
experiments remain future performance work, not prerequisites for accepting the capture.
Exact-owned57271085 is independently absent20:48:03.025Z. All12 actual rentals are closed;
attempt12 created no instance. Evidence: `docs/performance/gpu/disassembly/executed-library-images.md`
and its linked raw admission, analysis and independent review.
