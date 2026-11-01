# Executed library kernels on RTX 5090

Fresh repaired-observer capture on2026-10-22: Mapal, cuBLAS and selected CUTLASS all pass
live admission. The selected call has one Mapal kernel, two cuBLAS kernels and one
CUTLASS kernel. All four exact entries have hash-bound decoder outputs. This is
diagnostic evidence; no captured duration is admitted as a performance measurement.

Case: M16/N2048/K7168 Contract, frozen independent-output Mapal and tuned standalone
CUTLASS `m16-n64-k16`. GPU UUID `GPU-6f5a16b4-bef2-c3b0-6c58-c80edb2eed53`, driver610.43.02.
The actual cuBLAS entries are a CUTLASS-named GEMM and `splitKreduce_kernel`.

- [Capture summary](capture-summary.json), [exact request](capture-request.json).
- [Independent admission report](independent-review/REPORT.md), [summary](independent-review/ADMISSION-SUMMARY.json), [decoder limits](independent-review/decoder-limitations.json).
- [Final independent instruction/report review](independent-review/FINAL-REVIEW.md), [bound receipt](independent-review/FINAL-REVIEW.json).
- [Instruction analysis and action matrix](analysis/REPORT.md), [machine-readable anchors/counts](analysis/comparison.json), [publication hash mapping](analysis/publication.json).
- Attribution: [Mapal](mapal-attribution.json), [cuBLAS](cublas-attribution.json), [CUTLASS](cutlass-attribution.json).
- Exact decoder stdout: [Mapal](disassembly/mapal-sass-0/stdout), [cuBLAS GEMM](disassembly/cublas-sass-0/stdout), [cuBLAS reduction](disassembly/cublas-sass-1/stdout), [CUTLASS](disassembly/cutlass-sass-0/stdout). Each directory also preserves the command/status and stderr. All contain unknown `?trans` and `?WAIT` annotations; limits accompany the analysis.
- [504-file recovery inventory](recovery.json), [root independent rehash](root-recovery-verification.json), [cleanup](cleanup.json).
- Reviewed operation: [packet](operator-packet.json), [root binding](root-review.json), [explicit repaired-bundle approval](user-approval.json).

The complete immutable trace archive is preserved privately at
`.benchmark-artifacts/gpu/library-images/counter-free/rtx-5090/2026-10-22/admitted-library-kernels/trace.tar.gz`,
10,938,433bytes, SHA256 `dab5bbba2e0f6c57d1b60289b442cd496c2f5e8f26c4f37ded6cabb42586dec6`.
It includes callback/image bytes, numerical outputs, command/tool provenance and the
original ordinary artifacts. It does not include provider/account logs. The raw failed
first capture and its rejected verdict remain separately immutable.

Mapal/CUTLASS match256/256 sampled ordered-FMA outputs; cuBLAS matches5/256 and passes
the frozen componentwise bounds. Independent checks cover4,608 exported words. These
are sampled output guarantees, not all32,768 cells or a universal library contract.

[Analysis and compiler actions](../../../../../../../docs/performance/gpu/disassembly/executed-library-images.md)
