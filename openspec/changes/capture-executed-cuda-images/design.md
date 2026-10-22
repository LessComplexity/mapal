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

## Authorized extension: library execution and compiler mapping

The user explicitly requests comparing actually executed cuBLAS/CUTLASS and Mapal code,
then identifying which differences require IR facts, which already-derived facts are not
consumed, and which are backend algorithm/placement choices. The original no-library-image
scope above describes the initial prototype; this section adds a separately reviewed
diagnostic capability after that prototype's live smoke succeeds. No compiler correction
or speedup is inferred from a static library image.

Extend existing `Dat ObservationPacket` with a benchmark-leg identity, observed module
lifetimes, activity function records and dispatch records. `Trn library_admit?` is planned:
`(completed harness leg, active context/module/function/launch evidence, copied cubin) →
unique executed function or rejection`. Its domain excludes missing, ambiguous, dropped,
unsupported or unloaded identities. It must not join by symbol name alone or invent a
kernel function ID that the pinned activity record lacks. Exact SDK field evidence and
planned joins are recorded in the S61 library-capture analysis attachment.

The existing benchmark-process and evidence-directory `Loc`s and callback-owned-byte
`Trm writefile` remain. Capture library/runtime registration and launch variants only
with explicit support and bounded output; unsupported paths remain non-admission.
The pinned CUPTI kernel activity record has context/name/correlation but no function ID,
so a unique active context+symbol→module/function association needs additional proof.
No credential/environment collection or production JIT/loading-mode alteration is allowed.
Actual disassembly and separate runtime counters establish different facts; static
instruction counts cannot establish dynamic traffic, stalls or achieved occupancy.

FRAMEWORK §4.5 laws1/4/5 require callback-lifetime copies, complete identity mediation and
composition with a completed, numerically checked harness leg. Instrumented results are
excluded from ordinary timing. Every claimed compiler gap must map to existing IR or
backend `file:symbol`, a planned implementation, or an explicit open question.

The preferred additional realization is now the pinned NVIDIA Nsight Compute tool,
not a custom library callback implementation. Planned `Trn profile_leg` runs one explicit,
pre/post numerically gated library/Mapal call in a dedicated process with profiler control
markers, then preserves all its internal kernels in a raw `.ncu-rep`. Planned `Trn
read_report` maps that report plus exact harness/PTX/oracle/candidate identities to the
tool's context/stream/kernel identities, SASS, resources and separately selected counters.
Both operate on existing benchmark/evidence `Loc`s. A tool report gives runtime attribution,
not an independently extracted cubin hash. No such hash will be invented.

This independent tool realization has its own actual-version/options/sections and live
permission/smoke obligations; it does not depend on the custom Mapal observer succeeding.
Clock control stays disabled. All internal kernels are preserved rather than assuming one
SGEMM call launches one kernel. Unsupported profiler access, missing SASS, failed numeric
markers or ambiguous identities cause non-admission. The exact 2025.1 CLI is the reviewed
baseline; options introduced by later versions are not silently assumed.

## Counter-denied host: callback image realization

The actual RTX 5090 diagnostic reached the exact selected ordinary binary with Nsight
Compute 2025.1.1, then failed with `ERR_NVGPUCTRPERM`. No report or SASS was produced.
This proves access denial, not a parser defect or a specific host configuration. Do not
change host driver/security controls to repair this diagnostic. Ordinary timings remain
separate and admitted.

The next planned realization returns to `library_admit?` above using the existing
successful Mapal callback observer as its base. `Dat LibraryObservationPacket` binds
the selected ordinary binary, explicit harness leg and profiler-control interval to
copied module lifetimes, successful dispatches and completed numerical gates.
`Trn observe_library` and `Trn admit_library` are planned under the standalone
benchmark tooling, with a prototype in `/private/tmp/mapal-library-image-capture/`.
They use the existing benchmark-process/evidence-directory `Loc`s and bounded
callback-lifetime `Trm writefile`; no counter interface or production runtime change.

Pinned CUDA/CUPTI header fields must support every identity edge. Resource module
records lack a CUmodule handle, so they cannot alone establish a driver-handle join.
Accept either an observed direct loader/function association or additional function
metadata yielding one complete, unambiguous active context/module/exact-symbol join.
Missing, reused, dropped, deferred or ambiguous identities reject. All supported internal
library launch variants and every image required by the selected leg must be retained.
Static SASS resources/opcodes cannot establish dynamic traffic, occupancy or stalls.

Root reviews the exact extension before a live smoke. No loading-mode, compiler-option,
numerical-budget or ordinary-clock changes are implied. FRAMEWORK §4.5 laws 1/4/5 retain
their callback-lifetime, complete identity mediation and completed-leg obligations.
