# harness — cheaper steps, same checks (RTX 5080 desktop, 2026-11-22)

Last sprint's agent time: ~43 % tests, ~22 % GPU timing, ~7 % bundle builds. Three changes, each shown to check the
same thing as before. Machine as in `README.md`; every run pinned to `taskset -c 16-31,48-63` (32 CPUs),
`RUST_TEST_THREADS=16 MAPAL_PAR=4 MAPAL_TEST_JOBS=8`, Tctl > 86 C stops the run's process group until < 78 C (paused
seconds subtracted, "net"). Raw output: `.claude/raw-evidence/local-rtx-5080/harness/` (scripts, summaries, counters,
bundle comparisons). Flow and knobs: `benches/local/README.md`.

## 1. Tests: the emulator's serialized modes, generated programs in parallel

**Cause.** `MAPAL_EMU_SERIAL=1|2` (a block's threads one at a time between meets) handed the turn with one
`pthread_cond_broadcast` on the block's shared condition: every hand-off woke every waiting thread of the block to
see whose turn it was — b_n² wake-ups per phase, 65 536 for a 256-thread block. Load average 180–210 on 32 CPUs,
Tctl 89–93 C, and almost no progress: `staged_column_passes_values` (one program) 129 s.
**Fix** (`mapal_gpu_emu.c`): one condition per thread; a hand-off signals the one thread the turn goes to (`s_give`).
The order (`s_pass` / `s_first`) is untouched, so every run executes the same interleaving: 129 → 9.6 s.

**Programs in parallel** (`testgen::par_check`): the programs are drawn first from the test's
`TestRunner::deterministic()` (the serial stream), then checked at most `MAPAL_TESTGEN_JOBS` at a time per test binary
(default CPUs / 4), each worker taking the next index; results return in index order (counters are the serial
run's at any job count), the lowest-index failure is raised. An emulated program's modes and picks run side by side
(`Emulated::run_each`, one child slot each). Converted: the generated tests of row_block, long_rows, onto_device (3),
device_geometry, large_gemm (2, were serial), exp, narrow, calls, horizontal, zero_fold, hardware_properties,
fold_algebra, generic_fusion (2), site_choice (values). `emu::job_slots` default (CPUs − 2) / 2 → CPUs / 4 (half the
CPUs ran the pinned cores at 89–93 C).

| test binary | before, net s (paused, peak C) | after, net s | × | tests | counters |
| --- | --- | --- | --- | --- | --- |
| row_block | 237.4 (84, 89.5) | 94.8 | 2.5 | 11 / 11 | identical |
| long_rows | 636.3 (22, 89.6) | 33.8 | 18.8 | 4 / 4 | identical |
| onto_device | 46.1 (6, 87.4) | 43.3 (10, 89.0) | 1.1 | 9 + 2 ignored | identical (3 lines) |
| device_geometry | 30.6 | 12.3 | 2.5 | 7 / 7 | identical |
| large_gemm | 614.2 | 113.2 | 5.4 | 2 / 2 | identical (2 lines) |
| **the five** | **1564.6** | **297.4** | **5.3** | | |
| exp | 180.7 | 15.3 | 11.8 | 4 / 4 | identical |
| narrow | 24.5 | 18.4 | 1.3 | 3 / 3 | identical |
| calls | 440.1 | 33.6 | 13.1 | 12 + 2 ignored | identical |
| horizontal | 512.3 | 30.7 | 16.7 | 4 / 4 | identical |
| zero_fold | 110.4 | 21.5 | 5.1 | 2 / 2 | identical |
| hardware_properties | 290.4 | 15.4 | 18.9 | 3 / 3 | identical |
| fold_algebra | 146.9 | 33.8 | 4.3 | 3 / 3 | identical |
| generic_fusion | 159.2 | 21.5 | 7.4 | 3 / 3 | identical (2 lines) |
| site_choice | 360.4 | 70.5 | 5.1 | 8 / 8 | identical (4 lines) |
| **all fourteen** | **3789.5** | **558.1** | **6.8** | | |

Whole workspace after (`cargo test --workspace --release --no-fail-fast`, same pin): 125 binaries, 1445 passed,
7 ignored, 10 failed — all `mapal-backend-cuda` `differential` (the real GPU: "unsupported display driver / cuda driver
combination" after the 18:42 system update below); 935 s net (35 s paused, one peak at 92 C), against ~30 min and
2174–2523 s summed per binary in this sprint's earlier workspace logs.

"Before" is HEAD 60f0f8ce's test binaries and emulator (a scratch worktree), "after" this change, one after the
other on the same pin; counters are every coverage line the tests print on the fixed seed
(`TestRunner::deterministic()`), diffed. Another project's pytest (~9 CPUs) ran during the "before" long_rows; the
long_rows "after" with the new emulator alone (first, aborted baseline attempt at 15 child slots) was 27 s.
onto_device is bound by its own pool-lane runs (`host_maps_of_a_gpu_build_split_on_the_pool`, 39 s, one program).

## 2. GPU timing: changed programs per A/B, one full suite per step

`benches/local/gpu_changed.sh <out> --off <rung> | --on <rung> | --base <ref> | --base-emit <emit>` emits every suite
program (`benches/local/suite.txt`, 35) and every face of both cells on both arms (`suite_diff.py`, host IR + device
IR less `; mapal.` comments) and bundles only those whose text differs, each with its other arm; `gpu_run.sh` then
value-gates and times them ABBA (probe order reversed on even repetitions; cells in one `tile_bench --splitk`, twice).
Equal text on both arms is the same host object and PTX: the A/B is null by construction. The full suite still runs
once per step (the runtime, driver and machine state text equality cannot see; a base whose runtime differs moves
every program).

| A/B | programs timed | cells | diff + bundle |
| --- | --- | --- | --- |
| HEAD emitter vs itself (`--base-emit`) | 0 of 35 | 0 of 2 | 16 s (nothing to bundle) |
| `--off exp-hw` | 5 of 35 (attn, softmax_rows, causal, prefill, gen) | 0 | 16 s (13 cache hits), preflight OK |

The GPU side of this was not timed here: a system update at 18:42 (`cuda 13.4.3 → 13.5.0`, nvidia-utils 620.73
against the loaded 620.70 module) left NVML at "Driver/library version mismatch" until a reboot. The full-suite run
is ~10 min (last sprint); a changed run times only its programs' arms (5 of 35 here) plus the run's fixed sections.

## 3. Incremental bundles

`integrated_bundle.sh` content-addresses each stem's host object, optimized device IR, PTX and remarks by the emitted
host + device IR text, CPU, SM and a toolchain key (clang / opt / llc versions + the script's text), each CPU oracle's
`.expect` by its CPU IR + `libmapal_rt.a` bytes, and `evidence.py static` its ptxas -v / SASS by PTX bytes + SM +
ptxas release (`BUNDLE_CACHE`, default `target/bundle-cache`; `=0` off).

| local suite bundle (35 programs + 2 cells) | wall s | reused |
| --- | --- | --- |
| `BUNDLE_CACHE=0` (HEAD's commands) | 154.7 | — |
| cold cache | 156.8 | 3 (equal emissions inside the bundle) |
| warm (rebuild, emission unchanged) | 47.1 | 88 |

Equivalence: the warm and cold bundles equal the uncached one file for file (2157 / 2189 files, 0 differ) after
removing what differs between two uncached builds too — ptxas -v's `Compile time` line and the set-valued cells of
`facts.json` / `evidence.tsv` printed in Python's hash order — and apart from the 35 `tiles/*.cpu` oracle executables
a cached oracle does not leave (intermediates nothing reads; rentals exclude them). On a hit `opt -S`'s `; ModuleID`
and `source_filename` lines are rewritten to the stem's own input path. The toolchain update at 18:42 changed the
ptxas key: the next bundle re-ran ptxas 13.5 instead of serving 13.4's SASS.

## Open

- The changed-programs run on the GPU (driver/library mismatch since the 18:42 update; reboot needed).
- Generated loops still serial (each now on the fast emulator): site_geometry (4), plan_cost (2), recurrence (2),
  boundary (2), radix, tf32_tile, dispatch_width, generic_fusion's third (48), hardware_properties' `pdl` test (24),
  site_choice's three emission-only loops.
- A workspace run still touched 92 C once (guard paused 35 s): the binary is not identified.
- A warm bundle's remaining 47 s: the emitter per stem and arm (serial) and `evidence.py facts / check`.
