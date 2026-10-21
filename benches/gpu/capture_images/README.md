# Bounded production driver-PTX image observer — preparation only

No GPU observation has run. No speedup, actual JIT-image capture, or Linux shared-library link is established by these files. Local evidence uses NVIDIA CUDA/CUPTI12.8.90 real headers and CUDA compiler headers12.8.93 with macOS clang; CUPTI calls are stubbed in tests. Offline ptxas evidence remains a different variant.

## Boundary and model

Dat `ObservationPacket` extends the existing performance packet with exact input PTX, module cubin bytes, callback chronology, contextUID/moduleID, function handle/name, correlation ID, launch dimensions and completed harness receipt. Trn `callback` in capture.cpp runs in Loc benchmark process; Trm `writefile` copies callback-owned code bytes to Loc isolated output directory before callback lifetime ends (FRAMEWORK §4.5 placement honesty). Trn `validate` in validate.py deduces admitted image/launch joins from events, rather than persisting a parallel identity model (composition soundness). Trn `run_capture` in run_capture.py binds exact executable/PTX/observer hashes and process completion. No production architecture delta.

Only the `benches/gpu/tile_bench.cu:main` path at lines345–346 and385 is supported: `cuModuleLoadData` of exactly the allowlisted NUL-terminated PTX; nested `MODULE_LOADED`; successful `cuModuleGetFunction`; successful `cuLaunchKernel`. Schema v2 binds the resource callback actualCUcontext to the pending load, and requires identical actual context, contextUID, correlationID, output-module address and input-image address at load exit. Every launch must exactly match the reviewed CLI grid/block/shared-bytes contract preserved in the receipt. The validation result hashes that receipt; geometry in the example is illustrative and must be supplied from the actual reviewed invocation. Unmatched library modules are never saved. This does not claim cuBLAS/CUTLASS images or link their fatbins. Every library claim still needs its own exact executed-image/function proof. Other loading APIs, CUDA graphs, deferred resource callbacks and handle reuse fail admission or remain absent. Do not enable eager loading or alter JIT options to make observation work: that would change the variant.

Driver launch success is submission evidence. Completion comes separately from the reviewed existing harness's exit0 after its current synchronization and correctness gates. This does not expand its256-sample numeric acceptance into whole-output accuracy.

## Reviewable future use (separate authorization/budget)

Build on Linux with pinned CUDA12.8 and CUPTI12.8, setting CUDA_HOME, optional CUPTI_HOME, and a new CAPTURE_BUILD directory, then run `bash build.sh`. This links the real CUPTI library and does not link or call CUDA directly.

Run only a reviewed existing benchmark executable, using its normal arguments and one expected kernel:

```
python3 run_capture.py --out /new/isolated/run --ptx /exact/input.ptx \
  --observer /new/build/capture.so --kernel expected_entry --grid 32,32,1 --block 256,1,1 \
  --shared-bytes 16640 --timeout 60 \
  -- /absolute/tile_bench [existing arguments]
```

The wrapper exclusively creates the destination. The observer exclusively creates `capture/` and every code/event file. Fixed limits:16MiB input PTX;16MiB total captured cubins;4096 callback events;1024-byte kernel name;16MiB per child output file;1–300s process timeout (default60s). `/proc/self/mem` reads only the authorized PTX length plus NUL for equality, preventing an invalid/short unknown image pointer from crashing the observer. Unknown input bytes are not persisted. No environment dump, credentials, tensors, memory payloads, timing API, replay, artificial synchronization, compiler mutation or new rentals.

`receipt.json` and `validation.json` are retained. Missing subscriber, module callback, exactPTX join, resource/load context or correlation identity, function join, exact expected launch contract, successful launch, completed harness, or nonemptyELF bytes fails. Resource data is copied within callback lifetime; validator hashes exact saved bytes. Hash executable, observer and PTX before/after. Output from this process is **never ordinary timing evidence**. CUPTI's single-subscriber restriction means run separately from ncu/nsys; later counter collection uses existing ncu, separately.

For a later admitted packet, run recorded `nvdisasm --version` / `cuobjdump --version`, disassemble each saved `module-N.cubin` with a process timeout, retain raw output/stdout/stderr/exit status, hash cubin and SASS, and verify the exact requested function appears before making instruction claims. No tool execution or SASS claim is supplied here.

## Primary API sources and header provenance

Consulted2026-10-21:
- https://docs.nvidia.com/cupti/12.8/api/structCUpti__ModuleResourceData.html (`moduleId`, `cubinSize`, `pCubin`, callback lifetime).
- https://docs.nvidia.com/cupti/12.8/api/structCUpti__ResourceData.html (`context`, `resourceDescriptor`).
- https://docs.nvidia.com/cupti/12.8/api/group__CUPTI__CALLBACK__API.html (ResourceData wrapper, module-loaded ID, subscription conflicts).
- Pinned PyPI NVIDIA wheels: `nvidia-cuda-cupti-cu12==12.8.90`, `nvidia-cuda-runtime-cu12==12.8.90`, `nvidia-cuda-nvcc-cu12==12.8.93`; package and relevant header SHA256s are recorded in ../../results-s54-gpu/s58-executed-image-capture/header-manifest.json. Official wrapper docs and ModuleResourceData prose are inconsistent about raw cbdata; actual GPU callback-layout verification remains mandatory, and the prototype is not admitted production observation evidence.

Additional source: https://docs.nvidia.com/cupti/12.8/main/main.html#sass-source-correlation explicitly identifies module callbacks as CUDA-binary capture points. GPU/driver/toolchain identity must be supplied by the broader future proof packet; this narrow tool does not inventory the system.

Local checks: `clang++ -std=c++17 -fsyntax-only -Wall -Wextra -Werror` against unmodified real headers passes; same headers build the stub executable; `CAPTURE_TEST_CALLBACK=/path/to/stub python3 test_capture.py` exercises callback lifetime, happy join, missing/empty/corrupt/duplicate module data, absent/unlinked/failed launch, variant/hash/status corruption, chronology, symlinks, resource/load-context/correlation corruption and exact launch-geometry mismatches. Tests use synthetic structurally validELF sections; nvdisasm acceptance is deliberately unproven.

## Reproduce the local preparation checks

From the repository root, use unmodified pinned NVIDIA headers. The retained receipt used these extracted packages: CUDA runtime/CUPTI12.8.90 and CUDA compiler12.8.93. The compiler header package supplies `crt/host_defines.h` when using wheel headers. No dependency wheels or prebuilt executables are committed.

```bash
CUDA_HEADER_ROOT=/private/tmp/mapal-s58-capture/headers/unpacked/nvidia
CAPTURE_CHECK_DIR=$(mktemp -d /private/tmp/mapal-capture-check.XXXXXX)
clang++ -std=c++17 -Wall -Wextra -Werror -fsyntax-only \
  -I "$CUDA_HEADER_ROOT/cuda_runtime/include" \
  -I "$CUDA_HEADER_ROOT/cuda_nvcc/include" \
  -I "$CUDA_HEADER_ROOT/cuda_cupti/include" \
  benches/gpu/capture_images/capture.cpp
clang++ -std=c++17 -Wall -Wextra -Werror \
  -I "$CUDA_HEADER_ROOT/cuda_runtime/include" \
  -I "$CUDA_HEADER_ROOT/cuda_nvcc/include" \
  -I "$CUDA_HEADER_ROOT/cuda_cupti/include" \
  benches/gpu/capture_images/test_callback.cpp -o "$CAPTURE_CHECK_DIR/test_callback"
CAPTURE_TEST_CALLBACK="$CAPTURE_CHECK_DIR/test_callback" \
  python3 benches/gpu/capture_images/test_capture.py
bash -n benches/gpu/capture_images/build.sh
```

The tests resolve `run_capture.py` relative to this source directory and require an explicit separately built stub executable through `CAPTURE_TEST_CALLBACK`; no test relies on a checked-in binary or a fixed temporary evidence path. With a complete CUDA12.8 toolkit installation, use its include directory plus the CUPTI include directory instead of the three wheel include directories. These checks invoke synthetic callbacks only. The future Linux shared-library build still uses `build.sh` and remains unexecuted.
