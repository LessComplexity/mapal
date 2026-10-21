## Context

See proposal.md. S57's offline image cannot identify the driver-loaded machine code. The existing target path is `benches/gpu/tile_bench.cu:main` (`cuModuleLoadData` then `cuModuleGetFunction` and `cuLaunchKernel`). Production code remains unchanged. Isolated prototype files are under `/private/tmp/mapal-s58-capture/source/`.

## Model delta

Extend Dat `ObservationPacket` with optional PTX/image/launch morphisms; no new production runtime model or parallel profiling framework. Loc benchmark process owns callback data; Loc evidence directory receives copied code through Trm `writefile`. Section4.5 placement honesty requires copying inside callback lifetime; dependency mediation requires explicit event joins; composition soundness requires deriving admission from those joins.

```text
ObservationPacket --input--> ExactPTX
ObservationPacket --image?--> LoadedCubin
ObservationPacket --launch?--> LaunchSubmission
ObservationPacket --receipt--> HarnessCompletion
ObservationPacket --admit?--> AdmittedObservation
```

| Morphism | Signature | Partiality | Semantics / realization |
| --- | --- | --- | --- |
| input | ObservationPacket → ExactPTX | Total | Exact authorized bytes, `capture.cpp:InitializeInjection` |
| image? | ObservationPacket → LoadedCubin | Partial | Only nested callback matching exactPTX, `capture.cpp:callback` / `writefile` |
| launch? | ObservationPacket → LaunchSubmission | Partial | Successful function/module/context join, `capture.cpp:callback` |
| receipt | ObservationPacket → HarnessCompletion | Total | Executable/PTX/observer identity and completed process status, `run_capture.py` |
| admit? | ObservationPacket → AdmittedObservation | Partial | Hashes, chronology, CUDAELF64 exact kernel text and complete joins, `validate.py:validate` / `elf_kernel` |

These file:symbol mappings describe isolated prototype code, not shipped production architecture. Component IMPLEMENTATION/ARCHITECTURE/STATUS reconciliation remains an integration task.

## Goals / Non-Goals

Observe the exact existing driver-PTX path without CUDA calls, synchronization, JIT-option mutation or ordinary timing admission. No rentals, uploads, compiler edits, library fatbin claims or counter framework. A launch callback establishes submission; the existing reviewed harness's successful completion supplies its own synchronization and numeric-gate evidence.

## Decisions

- Pinned unmodified NVIDIA CUPTI/runtime12.8.90 and compiler12.8.93 headers verify field and parameter names. `CUpti_ResourceData::resourceDescriptor` carries `CUpti_ModuleResourceData`; callback bytes are copied before return. Official wrapper/module descriptions conflict about raw argument type; actual GPU verification is mandatory.
- Exact allowlisted PTX plus terminatingNUL is compared using a safe Linux `/proc/self/mem` read. Unrelated module bytes are not retained. Alternatives that blindly inspect binary image strings or save every module expand collection beyond scope.
- Thread-local matching load, resource moduleID, driver contextUID, returned CUmodule, returned CUfunction and successful launch establish explicit identity joins. Handle reuse, missing/deferred/ambiguous callbacks fail closed. Names alone cannot establish module identity.
- Validator requires structurally bounded CUDAELF64 and one nonempty `.text.<expected kernel>` section. This validates basic structure and identity, not instruction semantics or successful disassembly.
- Finite input/image/event/name/time/output caps and exclusive destinations prevent unbounded collection or overwrite. Exact SHA256s and raw files are preserved. No environment or credential collection.

## Risks / Trade-offs

- Lazy loading may defer resource callbacks outside the matched call → reject packet, preserve absent production-JIT evidence; do not change module-loading mode.
- Other CUPTI subscriber or unsupported injection path → missing initialization rejects. Counter runs use ncu separately.
- macOS cannot execute CUDA or link Linux CUPTI → real-header compile and stub tests are preparation evidence only.
- Harness sampling does not prove all outputs → retain its existing accuracy limits.
- Current capability omits library images → no cuBLAS/CUTLASS disassembly claims from it.

## Migration Plan

Root reviews the isolated prototype. Integrate only after component docs reconcile and independent review. Future explicitly budgeted Linux GPU smoke must verify callback layout, load chronology, completion, nonempty cubin, exact kernel section and real nvdisasm output. Ordinary timings remain separate uninstrumented runs. Rollback removes the observer invocation; production runtime has no changes.
