## 1. Bounded proof and isolated implementation
- [x] 1.1 Bind S57 measured fallback gap and locate shared emitter/selection; verify proposal/design cite real symbols.
- [x] 1.2 Add geometry regression covering M16 three projection families and boundary/large controls; preserve old-code failure log.
- [x] 1.3 Fit thread-grid shape using existing geometry checks in isolated snapshot; verify new test and LLVM backend tests pass.
- [x] 1.4 Preserve before/after LLVM/PTX and patch hashes; verify ordered operations, complete tile coverage and unchanged controls.

## 2. Integrated acceptance and reconciliation
- [ ] 2.1 Root integrates reviewed patch and GPU owner measures M1/M16/M128 all projection families and 16K controls with value gates, actual executed-image evidence and both libraries; record losses and wins.
- [ ] 2.2 Reconcile component IMPLEMENTATION → ARCHITECTURE → STATUS then root STATUS; record §4.5 review and immutable session evidence.
- [ ] 2.3 Run supercharge drift and OpenSpec validation; fix dead rows before archive.

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
