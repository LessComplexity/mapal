## 1. Reliable smoke execution

- [x] 1.1 Add differential-only mode, aggregate value/trap failures and propagate remote failure after result retrieval; verify pass, mismatch, wrong-trap, empty-scope and remote-failure cleanup cases using local stubs plus shell syntax checks.
- [ ] 1.2 Time the forced-ON retain-resident arm and explicitly select both stream directions; verify available-arm enumeration from fresh bundles.
- [x] 1.4 Repair the existing mixed probe/decode report parsing boundary; verify saved decode, streams and L2-window result files summarize without dropping their intended records.
- [x] 1.3 Build fresh current smoke bundles and run on Vast; require exact inventory coverage and zero value/trap failures, recording source/tool provenance and instance cleanup.

## 2. Resolve the disputed selections

- [ ] 2.1 Measure integrated RTX 5090 streams on independent hosts, with null/reference controls and 15 interleaved probe repetitions; record per-probe distributions and a supported ON/OFF decision.
- [ ] 2.2 Measure H100 PCIe retain-resident on the integrated decode path, separating untraced wall from traced kernels; record a supported decision and any overlapping/sign-unstable cases.
- [ ] 2.3 Verify raster/l2-gate and H100 L2-window evidence against integrated placement; record kernel-only and iteration effects separately and update only justified selection rows.
- [ ] 2.4 Report completed selection decisions with fresh correctness/measurement evidence; verify applicable profile tests and emitted selection arms after any table edits.

## 3. Final integrated verification

- [ ] 3.1 Build and run the final selection on RTX 3090, A100 SXM4, RTX 4090, H100 PCIe sm_90a and RTX 5090; require complete value gates, expected timing rows/repetitions and profiling artifacts.
- [ ] 3.2 Complete independent final RTX 4090/5090 confirmations; verify machine independence, value gates, controls and reported uncertainty.
- [ ] 3.3 Report integrated validation results and confirm all task-created Vast instances are destroyed.

## 4. Reports and reconciliation

- [ ] 4.1 Write s54-gpu-fusion.md and s54-per-arch-selection.md with provenance, all losses, control behavior and limits; independently check report numbers against raw results.
- [ ] 4.2 Reconcile component implementation, architecture and status, then root status and continuation; record the FRAMEWORK section 4.5 review and zero-dead-reference drift output.
- [ ] 4.3 Strictly validate and archive the completed change, then write the immutable session handoff with exact resume commands and live resource state.
