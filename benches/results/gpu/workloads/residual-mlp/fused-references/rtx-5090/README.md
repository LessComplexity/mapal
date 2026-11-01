# RTX5090 residual MLP: fused references and original controls

[Readable comparison](../../../../../../../docs/performance/gpu/workloads/fused-library-comparison.md) · [Workload evidence](../../README.md)

All20 native confirmation rows and10 original Mapal/CUDA/cuBLAS/PyTorch rows are independently admitted on the same GPU. Tuned fused cuBLASLt wins both shapes; Mapal does not establish a fusion advantage.

- [Summary](summary.json) retains18 tiny gates,110 tuning trials,8 selections and20 held-out native confirmations. [Table](table.json) contains same-host whole-program wall medians.
- [Ordinary archive](raw/ordinary.tar.gz) contains1,384 unchanged files under `composed-native/` and `composed-controls/`, including builds, raw stdout, requests, source/binary seals, tuning and all controls. Extract into a new directory; do not overwrite historical evidence.
- [Native independent admission](checks/native-admission.json) and [reproducer](checks/audit_native.py) replay request/ACK/numeric/timing/selection contracts and all hash inventories.
- [Original-control admission](checks/controls-admission.json) and [report](checks/controls-report.md) retain the ten control rows and their scope.
- [Separate value/clock implementation](checks/review_values_clocks.py) imports no harness checker. Its [result](checks/raw-value-clock-review.json) verifies770,816 words and2,020 paired clock samples, including warmups, across all148 native executions.
- [Matched cuBLASLt configurations](checks/matched-lt-configurations.json) bind exact selected algorithms across fused/unfused modes, without claiming identical executed library kernels.
- [Provenance audit](checks/operator-provenance.json), [reproducer](checks/audit_provenance.py), [cleanup](checks/cleanup.json) and [recovery hashes](recovery.json) bind hardware, tools, source archives and exact-owned instance absence.

The check scripts preserve original absolute paths for traceability. Reproduction requires mapping their trace/packet/output roots to a fresh extracted archive and the sealed source packet identified in [preparation](../preparation/recovery.json). Full private trace and source packet are kept under the repository's ignored `.benchmark-artifacts/` tree; this public archive contains the complete ordinary measurement trees.

No static-image, counter, allocation-latency, causal-retention or graph-enabled PyTorch claim follows from these measurements.
