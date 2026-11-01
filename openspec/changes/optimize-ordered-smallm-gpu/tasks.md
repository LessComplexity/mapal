## 1. Bounded proof and isolated implementation
- [x] 1.1 Bind S57 measured fallback gap and locate shared emitter/selection; verify proposal/design cite real symbols.
- [x] 1.2 Add geometry regression covering M16 three projection families and boundary/large controls; preserve old-code failure log.
- [x] 1.3 Fit thread-grid shape using existing geometry checks in isolated snapshot; verify new test and LLVM backend tests pass.
- [x] 1.4 Preserve before/after LLVM/PTX and patch hashes; verify ordered operations, complete tile coverage and unchanged controls.

## 2. Integrated acceptance and reconciliation
- [ ] 2.1 Root integrates reviewed patch and GPU owner measures M1/M16/M128 all projection families and 16K controls with value gates, actual executed-image evidence and both libraries; record losses and wins.
- [x] 2.2 Reconcile component IMPLEMENTATION → ARCHITECTURE → STATUS then root STATUS; record §4.5 review and immutable session evidence.
- [x] 2.3 Run supercharge drift and OpenSpec validation; fix dead rows before archive.

S58 GPU attempt stopped before the first primary process because the adapter omitted
the required `/usr/bin/time` installation. Independent analysis verifies all 2,866
sealed payload and 180 result files: 1/234 records scheduled, zero primary processes
started, zero of 30 timing cells admitted. This is not a numerical quarantine or
performance result. The prerequisite correction is prepared/tested; the candidate stays
isolated. Both the failed run and closed rental are preserved in the S58 evidence table.

S59 completed the unchanged packet after explicitly installing/checking GNU `time`:
234/234 records, 28/30 admitted timing cells. Three wide M16 cells improve 4.82–5.18×;
four smaller M16 Exact/Contract cells regress 37–47%, beyond null/control variation.
Root independently verified 948 files and 10,080 raw samples. The blanket candidate
is rejected for promotion; task 2.1 remains open, including executed images and large
controls. No new selection threshold is accepted. See
`docs/performance/gpu/kernels/small-m-geometry-comparison.md`. Instance 57130775 is independently absent.

S62 integrates the reviewed geometry patch (s62/geometry, merge 40f858c) with the five other
items at 6a0e3a0 and measures, value-gated, on RTX 5090/4090/H100 PCIe/3090 against cuBLAS and
CUTLASS: M1 (splitk cells), all three M16 projection families (16x2048x7168, 16x1024x4096,
16x18432x7168), M128 and 4096³, with executed SASS per part
(`benches/results/gpu/disassembly/s62-integrated/`). M16 Exact/Contract kernels are 1.6–10.6×
faster than 4e1102c; losses (residual-MLP iteration, per-face tile choice) are in
`docs/performance/gpu/current-results.md` and `docs/performance/gpu/disassembly/s62-gap-map.md`.
2.1 stays open: the 16K controls were not run in S62. 2.2 and 2.3 are closed by the S62
reconciliation (component IMPLEMENTATION → ARCHITECTURE §10 with the §4.5 review → STATUS →
root STATUS, `docs/sessions/2026-10-23-s62-optimize-integrate.md`; drift 0 dead; strict validation passes).

