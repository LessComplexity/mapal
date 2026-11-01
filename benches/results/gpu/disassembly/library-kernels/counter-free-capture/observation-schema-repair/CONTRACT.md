# Executed library image capture contract

Status: local counter-free observer prototype with independent CPU/source review; no GPU library-image admission yet. The prior Mapal callback observer passed four old/shared-tile Contract captures. Attempt09 Nsight Compute failed with ERR_NVGPUCTRPERM before producing a report, so it supplies no executed SASS or counters.

## Smallest execution boundary

Reuse the exact ordinary-selected binary and its build/selection SHA, complete original admitted argv, input/oracle/PTX/manifests, and existing diagnostic-only `--profile-leg mapal|cublas|cutlass`. Never rebuild or add `--values-only`: the actual harness rejects combining those modes. The existing branch calls cudaProfilerStart, exactly one selected implementation call, cudaDeviceSynchronize, cudaProfilerStop, then emits all post-gate words and the completion marker. No additional kernel, math option, library selection or ordinary timing change is proposed.

Run one fresh injected process per leg. Runtime profilerStart/Stop callbacks bracket a process-global region, retaining worker-thread launches. Nested/failed/missing/multiple regions reject. Enable observation before relevant module loading. Keep all resource module load/unload events from initialization onward because libraries can load eagerly. No NCU process, Event/Metric/Profiler API, PC sampling, SASS instrumentation, counter query, driver control, admin operation or host configuration change is part of this experiment. Activity timestamps, if present, are diagnostic only.

## Data and joins

| Atom | Planned source realization | Admission |
|---|---|---|
| Dat SelectedRun | existing ordinary receipts + exact executable/argv/request | hashes before/after; exact schema and dual-reference numeric gates pre/post; final protocol marker; no timing admission |
| Dat LoadedImage | MODULE_LOADED resource payload copied before callback returns | exact CUcontext, moduleId, lifetime generation, ELF bytes/hash; all loads retained within limits; MODULE_UNLOAD_STARTING closes lifetime |
| Dat Function | successful loader/GetFunction association where exposed; otherwise FUNCTION activity contextId,moduleId,functionIndex,name | exact authoritative function/module record; no invented module handle in resource payload |
| Dat Launch | driver launch callback successful enter/exit + concurrent-kernel activity | exact API/correlation, context/stream, function-handle or explicitly supported handle kind, symbol, grid/block/shared; no unexplained kernel or dispatch |
| Trn join_image | direct loader/GetFunction chain preferred | otherwise require complete active-image inventory, authoritative FUNCTION metadata, unique context/symbol/function-index→module and an exact ELF symbol/text inside that module; ambiguity or missing metadata rejects |
| Trn disassemble | saved callback ELF → exact launched function text → tool output | module SHA, function symbol/index/text SHA, decoder version/output SHA, same admitted launch; unused module functions not called executed |
| Loc | one fresh Linux process, exact CUDA/CUPTI build and GPU/driver identity | observer readiness before CUDA initialization; context/lifetime mapping must agree across callback/activity records |
| Trm | selected library call→all successful internal launches→activity records→saved module | complete inventories with zero dropped records and no silent truncation |

Kernel9 has correlationId/contextId/name and geometry, but has neither functionId nor moduleId. FUNCTION has authoritative moduleId/functionIndex/contextId/name. These are separate records. The fallback therefore requires an explicit uniqueness argument over the complete active module/function inventory; it must not report a direct Kernel9.functionId join. A same-name duplicate active module is an explicit unsupported case. Resources do not expose a CUmodule handle. CUDA calls from inside callbacks remain forbidden in this design.

Support ordinary cuLaunchKernel and cuLaunchKernelEx, including per-thread-stream variants, with their actual distinct parameter structures. Subscribe to driver domain to detect other launch paths. Cooperative/graph/internal/context-independent paths are rejected unless their complete context/function/module association is explicitly implemented and tested. CUPTI CONCURRENT_KERNEL and DRIVER/RUNTIME activity provide correlation/completeness checks. FUNCTION delivery without counters on this driver is a live smoke obligation, not established by its header declaration. A successful direct association may still admit a kernel if FUNCTION is absent, but unexplained activity kernels never disappear from the inventory.

## Bounded pilot and failures

First pilot only the already ordinary-admitted 16×2048×7168 Contract case and selected old Mapal / selected CUTLASS geometry, three legs. Suggested reviewable pilot limits: 64 MiB per image, 256 MiB aggregate images, 128 image lifetimes, 512 selected-region kernels, 65,536 callback/activity records, 512 MiB admitted aggregate output (checked after each subprocess), 128 MiB RLIMIT_FSIZE per file, and 120 seconds per profile process. Limits are explicit inputs sealed before execution, not automatically enlarged. Cap failure records observed size and preserves a non-admission receipt; no partial success is called complete. These proposed limits require root review before packaging.

Buffers are copied in callbacks, serialized outside where possible; exclusive/no-symlink task-owned output, no arbitrary library writes. Reuse bounded owned-process-group timeout/cancellation from the successful capture runner. Activity completion callbacks count dropped records, record every enabled-kind return code, and flush only outside callback code before final admission. A final flush/readiness/record inventory failure rejects. Returned data remains useful as partial evidence with pass=false.

CPU obligations before sealing: exact-header compile; real callback-entry fixtures using those types; selected-region state, regular/Ex geometry, library module capture without Mapal pending, handle reuse/unload and context mismatches; authoritative-function join ambiguity, missing/dropped/unexplained records, malformed ELF/section, numerical/protocol failure, cap overflow, timeout/cancellation. Actual same-binary smoke must prove region callbacks, initial load completeness, all internal-kernel joins and post-gates. The pilot may return unsupported cleanly; no ordinary run is repeated or invalidated by optional capture failure.

## Compatibility evidence

`header-proof/interface.cpp` compiles with C++17/-Wall/-Wextra/-Werror against unmodified official Linux CUDA12.8.90 cudart headers, CUDA12.8.93 crt headers and CUPTI12.8.90 headers. This verifies field/CBID/API names only, not ABI behavior on the current GPU/driver. The ordinary host recorded these same package versions. Runtime profilerStart/Stop CBIDs171/172, driver Ex652/653, resource fields and FUNCTION/Kernel9 fields are checked. Package URLs/hashes and exact compiler invocation are in `header-proof/receipt.json`.

The prior Mapal live observer proves callback cubin copying plus direct function/module/launch attribution on its recorded driver; it does not prove library dispatch or current-driver activity compatibility. Any new result records actual CUPTI/toolkit/driver API versions and NVIDIA driver build separately. Existing JIT-produced ELF notes versus older nvdisasm support remain a decoder limitation; preserve exact bytes and unknown decoder annotations. Static SASS supports instruction/realization comparisons only. It proves no dynamic traffic, occupancy, throughput or latency cause.

Primary sources: [CUPTI12.8 callback and activity usage](https://docs.nvidia.com/cupti/12.8.1/main/main.html) documents asynchronous buffer delivery (no guaranteed record order), explicit dropped-record handling, single subscriber and prohibition on ordinary CUDA calls inside callbacks. [CUPTI12.8 activity API](https://docs.nvidia.com/cupti/12.8.1/api/group__CUPTI__ACTIVITY__API.html) documents the separate function/kernel records and enable-return failure modes. Raw event order cannot therefore be treated as activity execution order; joins use correlation IDs and complete lifetime records.

The CUPTI functionIndex is retained as authoritative CUPTI metadata. It is not assumed to equal an ELF symbol-table ordinal: the exact named ELF symbol/text must be resolved and hashed within the already attributed module.

## Prototype review refinements

Admission now additionally requires observed selected-context creation before its first known use, an explicit successful context+PTSZ-aware stream CUPTI metadata mapping matching activity IDs, and zero outstanding requested activity buffers after non-forced final flush. Any retained buffer is non-admission, even if empty or final dropped count is zero. Ex launches support only numAttrs==0 for this pilot; all other attributes reject. Defined ELF STT_FUNC symbols must have nonempty bounded ranges in executable CUDA function sections; their ELF symbol ordinals, independently computed, select nvdisasm output. CUPTI functionIndex remains separate metadata.

Both regular and Ex CUDA12.8 APIs explicitly accept CUfunction or context-independent CUkernel cast to CUfunction. The parser therefore calls this field opaque_launch_handle. A direct successful GetFunction chain proves its CUfunction origin. The fallback explicitly supports an otherwise unclassified launch argument only when actual context/stream metadata, API/kernel correlation, unique active-image ELF symbol and authoritative FUNCTION module identity independently prove execution attribution. It never infers the underlying handle kind from the parameter's C++ type. Null/mismatched executing context or stream, unsupported dispatch, or unresolved image/function rejects. This is narrower than generic context-independent launch support.

The adapter deadline bounds subprocess waits, with bounded TERM/KILL/reap cleanup in addition. The aggregate512MiB check is an admission check after each command; it is not an instantaneous filesystem quota. Per-file RLIMIT_FSIZE and observer image/event/buffer caps provide separate active-process bounds.

## Callback compatibility revision

API invocation identity uses CUPTI's documented `correlationData`64-bit ENTER/EXIT storage, separate from activity correlation. Runtime parents carry an exact invocation token. Unique domain/CBID/correlation/callback-status/activity-status joins are required; repeated ambiguous eligible API invocations and multiple valid activity domains reject. TSV schema is `library-image-callback-v2`.

Defined device-helper STT_FUNC ranges may share one uniquely named entry section only when fully contained in that entry's bounded executable function range. Helpers are inventoried, never treated as launched entries or decoder indices.

Only exact SetCurrent303/PrimaryCtxRetain386 driver callbacks with UID0 may occur in the prefix before authoritative context creation. They provide no initialized-context or module identity. Missing creation, positive-UID/unknown/module/dispatch precreation use or a provisional event after creation rejects.

These changes have CPU protocol proof, including a full synthetic replay of the failed pilot with fresh invocation storage. The failed pilot remains0/3 admitted image legs; no actual cuBLAS/CUTLASS SASS has been admitted. See REPAIR.md.
