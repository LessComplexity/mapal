## Context

See proposal.md. The S54 placement model already exists in `docs/components/backend-nvptx/ARCHITECTURE.md`. Current `box_run.sh` prints `== done` even after correctness failures; `vast_run.sh` swallows remote failure. Post-arena-fix bundles were built historically but no post-fix GPU results were collected. Prior CUTLASS is cached at commit `91843b71d90737e0a9b86f7d9df213cff5a3526f`.

## Goals / Non-Goals

**Goals:** explicit smoke correctness status, current integrated evidence, justified conservative selection and reproducible reports.

**Non-Goals:** new fusion features, precision-policy changes, unbounded kernel tuning, reuse of old compiled objects, or a general benchmark orchestration service.

## Decisions

- Extend the existing scripts with `DIFFERENTIAL_ONLY=1`, passed through the rental wrapper. Skip CUDA benchmark harness compilation and timing/floors in this mode. Run every generated differential/trap case, require nonempty differential coverage, aggregate failures, and emit completion only on success. Normal full runs also stop before timing on differential/trap failures. Preserve logs before propagating a failed remote exit and let the existing EXIT cleanup destroy the rental.
- Add the existing forced-ON retain-resident arm to decode timing so a conservative OFF default remains measurable. Explicitly set probe arms including both on-streams and off-streams.
- Validate each collected run against an inventory derived from its frozen bundle. The smoke contract is implemented in the script; full timing acceptance also requires complete expected rows/repetitions, finite timings, value/library gates and readable profiling artifacts. A completion marker alone is not evidence. Preserve verification summaries with raw results.
- Final integrated timing records must preserve and enforce both untraced and traced subprocess exit status for probes and end-to-end programs. Existing selection bundles predate this check: their complete output/value/trace inventories remain required, and their unavailable per-process status must be disclosed rather than inferred.
- Repair the existing `table.py` probe/decode parsing boundary if needed for mixed result files: decode library rows omit the probe-size field and belong to the separate decode summary. Verify against saved decode and probe files; never invent missing sizes or silently discard ordinary malformed probe rows. Use untraced iteration/wall for streams/retain-resident decisions; traced kernels are a separate diagnostic and serialize streams.
- Build into fresh directories. No stale program/arm/oracle reuse. The newly discovered rewrite repair changes the reference pipeline, so regenerate oracles from current interpreter/CPU emission. Pin CUTLASS; record source revision/diff, compiler versions, bundle hashes, offer/machine/instance identifiers and cleanup evidence.
- Smoke `px_bias_100`, `px_mlp2_64_f64`, `px_residual_128`, `px_fan_128`, `e2e_trap`, both faces/applicable TF32 and arena/epilogue/resident/devgen/eager-init/async/arch-rungs/streams arms before timing.
- Resolve 5090 streams using integrated probes plus twin, with 15 interleaved repetitions and a distinct confirmation host. Resolve H100 retain-resident on PCIe via untraced and traced decode results separately. Compare raster/l2-gate with null controls at the recorded sizes and distinguish kernel from iteration effects. Overlapping or sign-unstable effects do not justify an override. Existing evidence favors removing the 5090 forced raster override; verify the actual integrated emission before deciding.
- Full run profiles: `rtx-3090:sm_86`, `a100:sm_80`, `rtx-4090:sm_89`, `h100-pcie:sm_90a`, `rtx-5090:sm_120`. Require final independent 4090/5090 checks. Keep rentals bounded to active work; do not affect unrelated instances.
- Model delta: no new runtime Dat/Trn/Loc/Trm. Existing `TargetProfile::gpu_rung` selects realizations of the documented placement rule; any table changes alter measured selection, not numerical semantics. For FRAMEWORK §4.5 laws 1–6, smoke and integrated differential tests check placement/transmission and composition coherence; profiling corroborates which existing transmissions run. Benchmark records are existing tool outputs, not new runtime objects. Map changed tooling in the existing component implementation map.

## Risks / Trade-offs

- Real correctness defects may stop performance work → fix and revalidate before claiming results.
- Slow full bundles/runs → use targeted correctness first, then disjoint compilation passes into new frozen directories; parallel profiles can build independently.
- Host variability can dominate small gains → null controls and independent confirmation; report uncertainty and conservative defaults.
- Profilers may be unavailable on a rental → distinguish missing evidence and obtain another host when needed for the required claim.

## Migration Plan

Finish local validation first. Implement/test the small harness changes, smoke current code, collect selection A/B results, update only supported table rows, then rebuild changed profiles and run the final five-profile/confirmation gates. Reconcile reports and component docs, require drift and OpenSpec validation, archive, and write immutable live-state handoff.
