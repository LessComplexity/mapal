# Independent peer review: matrix-only verification packet

No concrete blocker found for the separately admitted local saved-LLVM build
and subsequent small matrix value run. Compilation, linking, runtime values,
actual arithmetic routes and optimization remain unproved. No build, install,
upload, rental or GPU action was performed in this review; packet/author files
were not edited. Only this review was written.

Reviewed packet manifest SHA
`1afac8f989e4cf6343f7e6cc2dfaceb095e4866a4bf2195e739d29978c171fde`.
All60 payload hashes and60 source/destination provenance bindings match. The
four emissions are exactly check/run × Exact/Contract, copied byte-for-byte
from their immutable composition certification manifest. No recurrent source
or emission is admitted. Both kits' matrix scalar references and input hashes
agree; policy14e704561e756669f2d1d883ea9dc48018965f4cb84b86a100799859bdaef4ae
and numerics3469aa34f1580520bcfac0e3516e8afcc9930eee87d2a559d160066948d4092a
are the conditionally reviewed versions.

Checked build/link boundaries:

- build_local.py:13–31 verifies packet and frozen clang/opt/llc bytes, then
  compiles saved IR with the declared host-O2/device-O3 commands. Exact outputs
  and every command status are recorded; tool/payload hashes are rechecked.
  No Mapal emission or cargo command exists. Local compilation is still pending.
- runtime/libmapal_rt.a is a normal archive and matches the frozen Linux
  x86_64 runtime SHA5b26eea594347505cd45d7aed48b750753d812ca583b7a28698441d2594b25cb.
  runtime/mapal_gpu.c also matches the frozen profile manifest. The link command
  at verify_box.py:53 reproduces the established driver-API C runtime recipe,
  with CUDA headers/driver stub plus pthread/dl/m. Host platform is explicitly
  restricted to Linux x86_64. Driver stubs are a link search path; actual CUDA
  lib64 is declared on the runtime path. A working driver/link is not assumed:
  dependency hashes, versions, ldd and runtime failures remain recorded.

Checked value/scope boundaries:

- mapal_gate at17–24 enforces every checkpoint label in order, all input and
  intermediate scalar bits, and complete final-only output with no extras.
  Check mode covers1496 scalar values; run mode128. Exact selects operation-wise
  scalar bits; Contract selects true-fmaf bits. Original emission/source hashes
  bind their staged bias/residual association to the pinned contract.
- Ordinary untraced and traced executions must both exit0 and pass their full
  gates. The traced launch parser matches the actual runtime format and binds
  nonempty launch names to entries in that row's exact PTX. This is a trace/PTX
  association, not a driver-JIT SASS capture or overlap proof.
- Handwritten naive/cuBLAS commands select only the matrix/full/verify scope.
  Their copied external validator requires immutable inputs, all shapes/counts,
  every checkpoint, full final output, both scalar budgets and naive's extra
  exact fmaf gate. The previously reported cold-flush/buffer-free counter scope
  correction is present in the copied source; none of its timing CLI is invoked.
- PyTorch eager/compiled commands are explicitly matrix-only and omit --time.
  The copied runner has no install, eager fallback or missing-checkpoint bypass.
  It checks both scalar budgets, full outputs and shapes. Compiled fullgraph and
  precision settings are requests until actual generated/runtime evidence is
  inspected. Missing torch/compiler support produces a nonzero result with
  preserved output/error; it cannot count as a passed realization.

Checked status/proof boundaries:

- support.command preserves stdout/stderr/hashes and exact nonzero exits,
  missing-executable127 and timeout124; live child groups are terminated on
  timeout/interruption. Required command failures stop numeric acceptance.
  Existing pure parser/status tests cover success, plausible output followed
  by nonzero exit, timeout, absent executable, payload mutation and strict output
  truncation/extra values. No compiler/GPU tests were rerun during this review.
- Numeric acceptance requires all4 Mapal,2 handwritten and2 PyTorch entries.
  No recurrence or timing argument is added by the packet. Optional profiler
  captures are separate verify-only runs with values re-gated; gaps/statuses
  stay explicit. Missing profiling does not fabricate arithmetic evidence.
- proof_complete and timing_admitted are initialized false and never set true.
  CLI0 can establish only the complete numerical inventory, with proof gaps
  still visible. It does not establish the conditional policy's actual kernel
  adherence, a performance result or timing admission.

The next review boundary is the saved local build receipt and then actual small
GPU outputs/tool versions. No source or protocol expansion is needed from this
static review, and no larger or recurrent admission follows from it.
