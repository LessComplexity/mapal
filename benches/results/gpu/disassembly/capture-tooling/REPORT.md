# Driver-loaded image observer preparation

Status: isolated prototype only. No production changes, GPU execution, rental, upload, CUPTI runtime link, executed-image capture, hardware counters, or speedup evidence.

Source: ../source/. `source-manifest.json` binds all source files. `header-manifest.json` binds official NVIDIA CUPTI/runtime12.8.90 and compiler12.8.93 wheel/header inputs. `local-checks.json` records commands and full test output. Real-header syntax and object compilation pass on macOS; the callback test executable links only CUPTI stubs. Eight unittest groups pass, covering at least24 negative variations plus callback-lifetime and exact identity happy paths. `drift-check`: exit0,563 rows,0 dead. OpenSpec `capture-executed-cuda-images --strict --json`: valid,0 issues.

The exact existing harness path is observed through cuModuleLoadData -> nested resource module-loaded bytes -> returned module -> returned function -> successful cuLaunchKernel. Only exact allowlistedPTX matching bytes are eligible. The validator checks bounded CUDAELF64 sections and exact nonempty `.text.<kernel>`, context/module/function joins, chronology, hashes and completed existing-harness exit0. Submission callback alone is not execution completion. Numeric validity retains the existing harness256-sample limits.

Fixed16MiB PTX and total cubin caps;4096events;1024-byte kernel names;16MiB child output file caps;512MiB executable bound;default60s / maximum300s child process timeout; process-group timeout termination; exclusive output paths. No arbitrary memory dump, credentials or environment inventory. Unknown/library modules are not persisted.

Unexecuted limits: Linux shared-library link/injection; real callback wrapper layout; resource callback timing under lazy loading; actual GPU execution/completion; real ELF disassembly; GPU UUID/driver/tool identity supplied by the broader future packet. Missing/deferred callbacks fail without changing JIT/loading mode. cuBLAS/CUTLASS library images and fatbins remain absent from this scoped tool and cannot receive disassembly claims. Offlineptxas images remain separate. Any instrumented run is permanently ineligible for ordinary timing. Counter collection uses existing ncu in a distinct future run; CUPTI subscriber conflicts are not bypassed.

Coherence review: placement honesty copies callback-owned bytes before return (tested by mutation); typed transmissions persist only authorized module code; totality rejects incomplete identity projections; dependency mediation uses explicit callback/load/function ports; composition admission is deduced from raw joins; no assumption of a unique execution location. No production categorical model delta. OpenSpec integration/doc-reconciliation/archive/GPU tasks remain unchecked. Parent owns final integration and immutable session log.

## Independent-review correction: schema v2

The first prototype omitted the actual resource callback context check and exact expected launch geometry. Independent review demonstrated both could admit an incorrectly bound packet. They are fixed in the isolated source only:

- Pending loads retain actualCUcontext, contextUID, API correlationID, returned-module address and input-image address. Resource callbacks must have the same nonnull actualCUcontext. Load exit must match all retained entry identities before associating a module. Schema v2 retains actual context pointers in enter/image/exit events and repeats the exit correlationID, so the validator independently checks these joins.
- CLI now requires `--grid x,y,z --block x,y,z --shared-bytes N`. The receipt preserves this reviewed expected launch contract; every admitted launch must match all seven dimensions/bytes exactly. Missing/malformed contracts reject. Validation hashes the receipt, preserving the launch contract in the admitted packet identity.
- `local-checks-v2.json` records real-header syntax/object compilation, stub executable build,11 passing test groups and shell syntax. All previous test groups remain. `review-fixes-v2/rejections.json` and nine retained negative packets prove wrong resource context, wrong exit contextUID/context/correlation/output/image identity, wrong grid, wrong block and wrong shared bytes reject. `source-manifest.json` now binds corrected v2 sources; original manifest retained as `source-manifest-v1.json`.

All original limits remain: no Linux runtime link/injection, CUDA execution, actual callback-layout verification, JIT image capture or disassembly has occurred. Root final review and persistence are still required before any future observation run.

## Repository persistence

Reusable source is `benches/gpu/capture_images/`. The only functional persistence adjustment makes tests relocatable using source-relative run_capture.py and explicit CAPTURE_TEST_CALLBACK. `relocated-checks.json` records a fresh real-header stub build and11 passing groups against the repository sources. No wheels, binary objects, stub executables or cubins are included. Original evidence is retained separately in the temporary workspace; sanitized receipts here substitute $REPO, $CAPTURE_WORK and $CAPTURE_CHECK_DIR for absolute local paths. Header/wheel bytes and hashes are unchanged by path sanitization. Raw synthetic negative events are embedded in targeted-negative-v2.json.
