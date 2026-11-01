# Exact executed library-image capture: grounding notes

Status: waiting for the existing four-case Mapal observer to pass on a GPU. No capture extension source or payload change has been implemented. These notes identify the minimal missing evidence for a later narrow proposal; they do not assert a working library-image observer.

## What the current source proves

`/private/tmp/mapal-s60-kernel/diagnostics/capture_images/capture.cpp:callback` returns immediately from resource-module events unless its thread-local `pending` matched the exact authorized Mapal PTX. That state is entered only through `cuModuleLoadData`; the observer subscribes only to that API, `cuModuleGetFunction` and `cuLaunchKernel`. It deliberately never persists library modules. Existing capture results therefore cannot answer the user's requested cuBLAS/CUTLASS executed-image comparison even if all four Mapal captures succeed.

The frozen harness `diagnostics/harness/tile_bench_checked.cu:main` initializes all three implementations before its warmup calls. Its `run` lambda invokes Mapal via `cuLaunchKernel`, cuBLAS via `cublasSgemm` and the pinned CUTLASS instantiated operator. It performs three warmups per leg and then existing numerical checks; `--values-only` returns before timing. cuBLAS math mode/version, requested shapes, pinned CUTLASS instantiation and build inputs must remain bound to the capture artifact. Library f64 errors are diagnostic under this harness; there is no ordered-bit promise for the library legs.

## Primary API facts and a rejected shortcut

Exact official12.8.90 development-package headers are copied read-only into this owned analysis directory:

- `cupti_callbacks.h:350`: launch callbacks provide the kernel's `symbolName`; callback parameter/return fields are valid only during the callback.
- `cupti_callbacks.h:443`: module resource payload provides `moduleId`, `cubinSize` and `pCubin`; resource data carries actual `CUcontext`. It does not provide a `CUmodule` handle in its resource-handle union.
- `cupti_activity.h:4706`: `CUpti_ActivityFunction` carries contextId, moduleId, functionIndex and name.
- `cupti_activity.h:2823`: `CUpti_ActivityKernel9` carries contextId, exact name, launch geometry and correlationId, but no functionId/moduleId field. A direct kernel.functionId→module join would be invented.
- `cupti_driver_cbid.h:665`: `cuLaunchKernelEx` and its per-thread-stream form have distinct callbacks. Their geometry lives in `CUlaunchConfig`, rather than ordinary `cuLaunchKernel` arguments.

[CUDA12.8 execution-control documentation](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__EXEC.html) documents context-independent `CUkernel` launch handles as well as `CUfunction`, and `cuFuncGetModule`/`cuFuncGetName`. This does not authorize making CUDA calls inside CUPTI callbacks. The existing observer makes none. Its `cuModuleGetFunction` path cannot be assumed to cover library/runtime registration or context-independent module paths.

Consequently, dumping arbitrary ELF sections from installed `libcublas*.so`, finding plausible kernel names, or using a name-only guess is insufficient. Those would be static library artifacts, not executed-kernel evidence.

## Narrow proposed boundary, conditional on the Mapal smoke

| Model atom | Planned realization and exact evidence needed |
|---|---|
| Dat `CaptureLeg` | Explicit diagnostic leg marker `mapal/cublas/cutlass`, bound to the unchanged `run` lambda's API call. Initialization is a separate stage because modules can load before the first leg. |
| Dat `LoadedImage` | Resource callback's exact cubin bytes, actual context, moduleId, load/unload lifetime, byte hash and size; bounded process-owned capture inventory. No assumption that a library module fits the current16MiB aggregate cap. |
| Dat `Launch` | Callback API ID, actual context/context UID, function or kernel handle kind, exact symbol, correlationId, stream, launch grid/block/shared; successful API exit and numerically successful values-only harness. |
| Trn `join_executed_image` | Preserve direct module-loader/GetFunction associations when exposed. Otherwise require proven complete active module inventory plus an unambiguous context/exact-symbol association to function/module metadata and exact ELF text; duplicates, missing function metadata or unseen dispatch APIs are non-admission. Availability of FUNCTION activity records is itself a live proof obligation. |
| Trn `disassemble` | `nvdisasm` the callback-saved cubin and select exactly the admitted launched function. Bind ELF text hash, SASS hash and module/function identity. Other functions in a loaded module are not labeled executed. |
| Loc | One fresh harness process and its CUDA context; an observer initialized before any relevant module load. A late/unproven start invalidates the completeness-based fallback. |
| Trm | API submission→successful kernel launch, and resource cubin delivery→saved image. Every accepted artifact must name its exact leg/process/context/module/function/launch joins. |

Potential harness change should be diagnostic-only: require `--values-only` and add leg markers around existing warmup API calls without changing dimensions, inputs, numerical checks, library parameters or kernel selection. The ordinary timing binary/source path stays frozen. A marker can use a weak exported observer hook rather than adding new GPU work. Record setup loads even before a selected leg so lazy/eager library loading is not silently missed.

Do not assume every library launch occurs on the caller's host thread. A thread-local marker alone would not classify worker-thread dispatch. A diagnostic phase can instead remain globally active until the selected leg's GPU work is explicitly complete, with any extra completion fence labeled as instrumentation; alternatively prove the per-thread API correlation. Likewise, a context-independent launch's stream can determine its context: callback current-context fields must agree with authoritative activity context/stream information rather than being assumed to identify the executing context.

The first library extension should pilot one already numerically admitted shape, retaining every kernel launched by the selected GEMM rather than assuming one kernel per operation. It must reject unsupported Graph/LaunchEx/cooperative/context-independent dispatch instead of dropping it silently. New byte/event caps require measured module-size inventory or a bounded failure receipt before enlargement. Any activity/profiler durations stay diagnostic; ordinary performance comparisons continue to use the existing uninstrumented run.

Required negative checks before an immutable packet: wrong context/module/function handle, renamed/missing exact ELF section, duplicate symbol across active modules, unsupported launch API, missing/dropped activity records, late observer start, failed numerical process, image/event-cap overflow and nested cancellation. A CPU-only header compile is necessary but cannot substitute for real module/function/launch joins.

After actual SASS arrives, compare only directly observed hardware mappings and instructions. Static opcode presence/count does not establish dynamic issue counts, traffic, achieved occupancy, memory overlap or the cause of a latency gap. cuBLAS's original IR is closed; missing Mapal IR facts must be established in Mapal's actual source/IR and tested, not reconstructed by guessing from library SASS.
