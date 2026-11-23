# benches/local — the RTX 5080 desktop flow

Threadripper 9970X + RTX 5080 (sm_120), CUDA at `/opt/cuda`. Evidence lands in
`benches/results/evidence/local-rtx-5080/`; raw output in `.claude/raw-evidence/local-rtx-5080/<step>/` (never in git).
Before a timed step: load < 3, no game, the bench lock held (`flock`); heavy runs pinned (`taskset -c 16-31,48-63`),
Tctl watched (`sensors`), parallelism lowered above 85 C.
Timed processes run on the cores with the highest clock ceiling, `CPUS=0-7` (CCD 0, CPPC `highest_perf` 211–241;
CCD 1 / 3 166): the same binaries run host-bound 15–90 µs programs 0.81–1.00× as long there as on 8–15, so runs
compare only on one set — `timed_cpus.sh` (sourced by the timed scripts) checks that set's busy share and writes its
`cppc` range into `conditions.txt`. Every timed run (`gpu_run.sh`, `model_scale_run.sh`) samples the GPU's P-state and
clocks five times a second into `gpu-clocks.csv` (summary `gpu-clocks.txt`): between processes the 5080 may drop to
P8, a state the other conditions do not show.

## Per step: changed programs, then one full suite

| what | command | times |
| --- | --- | --- |
| A/B of a rung | `benches/local/gpu_changed.sh <out> --off <rung>[,<rung>]` (or `--on` for a gated rung) | only the suite programs / cells whose host or device text differs between the arms, each against its other arm |
| HEAD vs a base | `benches/local/gpu_changed.sh <out> --base <ref>` (`--base-emit <emit>`) | only those whose text differs between the two emitters (`on` vs `on-base`) |
| then | `LADDER=1 LOCAL_GPU=<out> TORCH_PY=<venv>/bin/python benches/local/gpu_run.sh <out>/bundle` | probe section: value gates, then the arms interleaved, order reversed on even repetitions (ABBA); cells: every arm in one `tile_bench --splitk`, twice, the second reversed |
| safety net, once per step | `benches/local/gpu_bundle.sh <bundle>` + `gpu_run.sh <bundle>` | the whole suite (`suite.txt`, 35 programs + 2 cells) |

Why the changed-programs run checks the same thing as timing everything: `suite_diff.py` emits every suite program
and every face of each cell on both arms and compares host IR and device IR less the `; mapal.` evidence comments
(`changed.py emission`). Equal device IR is equal PTX under one LLVM; equal host IR is the same host object. A
program equal on both arms runs the same bytes on both: its A/B difference is null by construction, so timing it
only measures noise. Everything the text cannot see — the runtime (`changed.py` `RT_PATHS`: a base whose runtime or
timing harnesses differ moves every program, and its base side needs a full run at the base), the driver, the
machine's state — is what the per-step full-suite run is for. `<out>/suite.tsv` lists every row (`same` / `differs`
/ `runtime` / `err`) with both hashes; an arm the emitter refuses (`err`) is reported, not timed.

## Incremental bundles

`integrated_bundle.sh` (all bundles, local and rented) content-addresses what it builds under `BUNDLE_CACHE`
(default `target/bundle-cache`; `BUNDLE_CACHE=0` builds everything):

| cached | key |
| --- | --- |
| a stem's host object, optimized device IR, PTX and their LLVM remarks | emitted host IR + device IR text, CPU, SM, toolchain key (clang / opt / llc versions + the script's own text, which holds every flag) |
| a CPU oracle's `.expect` | its emitted CPU IR text + `libmapal_rt.a` bytes + toolchain key |
| ptxas -v and SASS of each distinct PTX (`evidence.py static`, `EVIDENCE_CACHE`) | PTX bytes + SM + ptxas release |

The emitter always runs (its text is the key), as do `evidence.py launch / facts / check` (cheap, and they read the
bundle's own files). A hit copies the bytes the same steps made for the same input, so a cached bundle equals a fresh
one file for file (`opt -S`'s `; ModuleID` line is rewritten to the stem's own input name). The bundle log ends with
`bundle cache: N built stems and oracles reused`. Delete the directory to reclaim space; nothing else reads it.

## Tests: generated programs in parallel

| knob | default | bounds |
| --- | --- | --- |
| `MAPAL_TESTGEN_JOBS` | CPUs this process may run on / 4 (8 under the 32-CPU pin) | generated programs checked at once per test binary, across its tests (`testgen::par_check`) |
| `MAPAL_TEST_JOBS` | CPUs / 4 (was (CPUs − 2) / 2: 89–93 C under the pin) | child processes (clang, emulated or native programs) in flight per test binary (`emu::slot`) |
| `RUST_TEST_THREADS` | 16 here | tests at once per binary |

`par_check` draws nothing itself: the programs come from the test's `TestRunner::deterministic()` before the call, so
they are the serial run's; results return in index order, so every coverage counter is the serial run's at any job
count, and the lowest-index failure is the one raised (`generated program #i of n failed` on stderr). An emulated
program's modes (`MAPAL_EMU_SERIAL=0|1|2`, `MAPAL_EMU_REVERSE`, picks) run side by side (`Emulated::run_each`), each
holding a child slot. The emulator's serialized modes hand the turn to exactly the next thread (one condition per
thread; one shared broadcast used to wake the whole block per hand-off) — the same order, so the same values.

Typical: `RUST_TEST_THREADS=16 MAPAL_PAR=4 taskset -c 16-31,48-63 cargo test --release -p mapal-backend-llvm --test <t>`.

The whole workspace: `benches/local/test_par.py [--bins N] [--logs DIR] [cargo selection...]` (default `--workspace`).
`cargo test` runs its binaries one after another; this builds them once and runs N at a time (default CPUs / 8),
longest first by the last run's times (`target/test-par-times.tsv`), each in its package directory, doc tests as one
more job — the same binaries and environment. Above `TEST_HOT` C (85) no binary starts until `TEST_COOL` (78); a
running one is never stopped (its tests time their children by the wall clock). Emulated programs link the emulator
compiled once (`target/tmp/emu-<key>.o`) with `ld.lld` where installed (`MAPAL_EMU_LLD=0`: the default linker).

## Other scripts

`comp_run.sh` (re-time competitor legs only — torch_bench / comp_bench — in a timed run's bundle, same protocol),
`comp_abba.sh` (same-box ABBA of two competitor legs of a row), the competitor venv (`competitors.md`),
`cpu_matmul.sh`, `cpu_shapes.sh` (CPU competitors), `model_scale_*.sh` (model-scale bundle / run / trace / report;
the trace takes `comp=<leg>` too),
`ktrace.cpp` + `ktrace_sum.py` (CUPTI kernel traces), `exp_calib.cu`, `gap.py` (gap table of a run).
