# S61 capture prerequisite boundary and reviewed setup option

Attempt03's independent reader verifies924 sealed input files and9 output files. All four capture observations are non-admitted because no completed capture receipt exists. The runner exits before observer compilation with `CUPTI headers absent: no install attempted`; observer smoke is false. No executed cubin/SASS, injection behavior or dynamic counter result was obtained.

Frozen `/private/tmp/mapal-s60-kernel/diagnostics/run_diagnostics.py` derives CUPTI as `$CUDA_HOME/extras/CUPTI` and requires `include/cupti.h`. Its environment guard rejects every nonempty `CUPTI_` variable, including `CUPTI_HOME`; an external override is therefore not a working unchanged-script remedy. `capture_images/build.sh` compiles with `$CUDA_HOME/include` and `$CUPTI_HOME/include`, and links/RPATHs `$CUPTI_HOME/lib64`. `run_capture.py` inherits the caller environment and adds the intended injection/identity variables; it does not add a CUPTI library search path. Hence the build RPATH must remain valid during actual children.

NVIDIA's official Ubuntu22.04 CUDA repository provides the exact installed versions observed in attempt02's package receipt. Downloaded archives were inspected locally without installation or executing their content:

| Package | Archive SHA256 | Relevant payload |
|---|---|---|
| `cuda-cupti-dev-12-8_12.8.90-1_amd64.deb` | `a7dc61e7b18dc48b3a2fb180c67ef4efa80bf874b6079ce5fe4183f4f2dc5102` | `/usr/local/cuda-12.8/targets/x86_64-linux/include/cupti.h` |
| `cuda-cupti-12-8_12.8.90-1_amd64.deb` | `cc73a4e090945b020fb0caa65830b9d12421d5cf802db5f977b1a3d7a78e9a68` | target `lib/libcupti.so → libcupti.so.12 → libcupti.so.2025.1.1`; root `lib64 → targets/x86_64-linux/lib` |

Archive sources: [official development package](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-cupti-dev-12-8_12.8.90-1_amd64.deb), [official runtime package](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-cupti-12-8_12.8.90-1_amd64.deb). `cupti-package-layout.json` preserves exact relevant members. The archived CUDA documentation's legacy `extras/CUPTI/lib64` example is not a substitute for checking the actual distribution package; see [NVIDIA CUPTI12.8 installation overview](https://developer.nvidia.com/cupti-ctk12_8).

The failure is consistent with looking only in the legacy layout while the installed package uses target paths. An actual file inventory was not recorded on attempt03, so this is not a claim that those files existed in that closed instance. Next-host `dpkg -L` and file/hash checks must precede any build; no missing-package installation is justified by the current evidence.

Minimal unchanged-payload option: create a fresh task-owned compatibility toolkit view with three directory symlinks:

```text
<view>/include                    → <cuda>/targets/x86_64-linux/include
<view>/extras/CUPTI/include       → <cuda>/targets/x86_64-linux/include
<view>/extras/CUPTI/lib64         → <cuda>/targets/x86_64-linux/lib
```

Set only the already allowed `CUDA_HOME=<view>` when invoking the frozen diagnostic runner. Preserve the active-observer environment guard, numerical/identity/join validators, frozen PTX and both frozen capture scripts. The observer's RPATH points to the task-owned compatibility `lib64`, which must live through all capture processes. No system CUDA symlink or package change is needed if actual package files validate.

`prepare_cupti_layout.py` is a reviewable CPU-only prelude, not evidence of a successful Linux/GPU execution. It requires exact existing package versions, records `dpkg -L`, validates package paths, creates the fresh view, compiles the frozen observer, checks `ldd` and ELF dynamic entries, and verifies packet hashes before/after. Syntax was parsed locally; the prelude has not been run on a Linux host. The operator must run it under the batch's phase timeout/cancellation boundary. If it fails, retain its receipt and skip capture. Main bootstrap/composed correctness work remains the principal phase; no B rerun is needed.

Review artifact: `cupti-layout-readiness.tgz`, SHA256 `b63e86247f3c1eb0724e43c1883a28d5f94851ead67fbcea5b019bff1eaf87af`. Actual smoke, capture identities and exact-entry disassembly remain mandatory after CPU readiness succeeds. That follow-up still covers only the four old/S58 Contract cases, not B.
