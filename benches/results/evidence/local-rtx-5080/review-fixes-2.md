# review-fixes-2 — the second sweep's review: escapes, unmeasured parts, replay lines, evidence claims

Branch `local-rtx-5080` @ `d950efb3` (the second sweep merge). RTX 5080 (sm_120, 84 SMs, 64 MB L2), Threadripper
9970X, CUDA 13, clang/LLVM 23; CPU set 16-31,48-63. Raw output: `.claude/raw-evidence/local-rtx-5080/review-fixes-2/`
(not in git).

An independent review confirmed four issues. Each is fixed (or shown corrected) below, each with a test that fails
without the fix; two more defects of the same classes were found while fixing them and are fixed too.

| # | issue | what was wrong | fix | test (fails without) |
| --- | --- | --- | --- | --- |
| 1 | `round-store` stores an escaping array narrow | `narrow_store` accepted a `Pair` into a function's return tuple (no out-edges): a kept callee returned `[bf16; n]` to a caller reading `[f32; n]` | `PrecisionPlan::store` refuses an array any of whose carriers escapes (`last_use_plan.escapes`, parameters aside) | `compute_type::precision_plan_store_narrow` (escaping callee: 0 stores; non-escaping: 1); `half_tile::round_store_on_generated_programs` (one program in five returns its `a` from a kept callee, counter `escaped`) |
| 1b | found: an epilogue group's escaping output is never stored | `read_outside` saw no reader of a value paired into the return tuple: `store_out = false`, the kernel wrote nowhere; the caller read zeros on the device (`--gpu-off=place`: `-0` / `0` against `-21.27581`, real RTX 5080) | `read_outside` counts an escaping value as read by the caller (as `uses` already did; flow units already did) | the same generated test (its first failing draw was this program: ON emulated `0` vs `-21.2`) |
| 2 | new priced rungs fall back to the old behaviour where a rate is unmeasured | `streamk_priced` returned `None` without `atomic_ops_per_ns` / `zero_ns` (3090, A100, H100: no stream-K for under-filled sites); `stream_choice` refused every group `unpriced` without plan rates (3090, A100, H100 SXM / PCIe: `unit-uniform` and `stream-mma` effectively off) | `Gpu::plan_cost`: each plan rate the part measured, every other derived from the rtx-5080's measurement at the part's own properties; `TargetProfile::priced` for the stream rule | `gpu::tests::streamk_price_separates_the_measured_sites` (priced on all four parts); `flash_attention::stream_priced_on_every_part` (the 2048-token prefill streams on each, values in 4 emulator modes) |
| 2b | found: the first cooperative launch free in a loop, and free where unmeasured | `chain_ns` / `row_slices` priced `coop_first_ns` at 0 for any looped chain ("the trips share it"), and `chain_cost` left it 0 on every part but the 5080 | a counted loop's trips (`trip_split`'s `[lo, hi)`) each pay `1/n` (`launches_in`; a loop of unknown trips no term, as before); `chain_cost` derives the 5080's measured 38 µs where unmeasured | `calls::counted_trips_share_the_first_cooperative_launch` (the 5080's two 2-trip programs form no chain), `calls::first_cooperative_launch_shared_by_counted_trips` |
| 3 | measured rungs without replay lines | wgmma-ws / tma-digits / ws-regs (H100), calls-issue, streamk-epi, stream-mma, kmajor-b, round-store, unit-uniform, post-table (5080) | 13 lines; the two losses the rules still take are `OPEN` with their missing fact | `rules::tests::rung_rules_replay` (243 lines: 140 picked / 72 null / 10 mixed / 21 open) |
| 4 | two evidence claims not backed by raw output | streamk-epi "ABBA 8 reps" (raw: 5); round-store's `pj_m128_n28672_k4096_bf16` "12-rep ABBA 0.3914 / 0.4037" (not in raw) | text corrected to the raw (5 reps); the row re-measured, 12 reps ABBA (below) | — |

## 1. round-store and escaping arrays

`narrow_store` (crates/mapal-ir/src/algo/precision.rs) now asks the stored fact `last_use_plan(f).escapes(c)` of
every non-parameter carrier the walk collected (the array, its capture tuples and projections, in their own
functions). A body's input is a borrowed parameter and always escapes; it is skipped. Repro (the review's
`esc4.mapal`, now in `compute_type.rs`): a callee with a counted loop (the inliner keeps it) returns `(a, c[0])`, `a`
read only by a `fold: bf16 | f32` — 0 stores; returning only `c[0]`, 1 store.

The generated test's escaping variant (`testgen::half_src_escaping`: the GEMM in `fn gemm(n: i32) -> ([a; mk],
[f32; mn], [f32; mn], i32)` with a counted loop, two elements of the returned `a` printed exactly) first failed on a
program with an epilogue (`u * 0.5 + a[i * k]`) and no round-store at all: the device left `y` zero. The real GPU
reproduced it with every site on the device (`--gpu-off=place`; the default build places these small sites on the
host): `y[166]` printed `-0` where the interpreter prints `-21.27581`. `read_outside` (resident.rs) decides whether an
epilogue group stores the site's output (`store_out`) and each member's; a value whose only use is a `Pair` into the
return tuple had no reader there. It now counts `last_use_plan.escapes` (as `uses` does for the copy-out, and as flow
units' `mb.store` already did). After the fix the device prints `-21.27581` (`p3`, `esc_full`).

Suite effect: none. `suite_diff.py --base HEAD` over all 199 committed `pt_*` programs + both cells × 5 faces on the
rtx-5080: same text everywhere (6 programs do not emit on either arm, as before) — null by construction, no timing.

## 2. Unmeasured parts are priced, not left on the old rule

`Gpu::plan_cost` (profile.rs) derives each unmeasured plan rate from the rtx-5080's at this part's properties, the
pattern `chain_cost` / `bar_ns` / `slice_cost` already follow: the FMA tile at the 5080's fraction of its FFMA peak;
the L2 rate at its bytes an SM a clock; DRAM at its fraction of the DRAM peak; `red.global.add.f32` at its adds per L2
byte (400 / 4500 on the 5080 and the 4090, 560 / 6500 on the 5090: one ratio on three measured parts); the zero pass,
the in-stream gap and the loaded latency in the 5080's cycles at the part's clock; the host issue as measured; the
fitted efficiency and LSU issue as the 5080's. `streamk_priced` reads it (a measured part's three rates are its own:
asserted for the 5080 / 4090 / 5090); `stream_choice` prices its unit and the split realization on
`TargetProfile::priced` (the profile itself where it carries plan rates — the 4090, 5090 and 5080 emit as before).
The plan tables keep D8 (the rules' plan where no rates are measured): that is a plan, not an OFF arm.

`chain_cost` (the `calls` price) now derives the process's first cooperative launch (the 5080's measured 38 µs, host
time) where unmeasured, and `chain_ns` / `row_slices` charge each launch `coop_first_ns / n` for a counted loop of `n`
trips (`trip_split`). Found by the calls-issue measurement in §3: on the 5080 the two programs calls-issue moved formed
a chain whose price had no first-launch term (in a loop: "shared"); timed, the first cooperative launch took 57 µs
against 14 µs for the second (`MAPAL_GPU_TRACE`), and the chains ran 1.77× / 1.48× their separate launches.

Emission moved by §2 (HEAD's emitter vs this one, all 199 `pt_*` programs, default face, host + device IR less the
`; mapal.` evidence lines; `parts-diff.tsv`):

| part | same | differs | what moves |
| --- | ---: | ---: | --- |
| rtx-5080 | 193 | **0** | nothing (6 do not emit on either side) |
| rtx-3090 | 143 | 50 | streamed attention / prefill (priced), under-filled stream-K, the derived first launch |
| a100 | 178 | 18 | streamed attention / prefill |
| h100 | 173 | 26 | streamed prefill; straight-line chains (`pt_pc_*`, `pt_pm_*`) now pay the first launch: refused |
| h100-pcie | 149 | 50 | as the 3090 |
| rtx-4090 | 162 | 31 | decode chains calls-issue formed at fp8-leftovers (unmeasured) now pay their 16 decode steps' share: refused |
| rtx-5090 | 161 | 32 | as the 4090 |

The causal family's chains (measured winning on the 4090 1.16× and the 3090 1.36×, cross-arch) are unchanged: 1024
trips share 37 ns each. `decode_1024_256` keeps its chain decisions. Every moved row on a rented part is unmeasured:
the next rented round times them (the c0 golden regolded 26 rows: 4090 / 5090 px_mlp2_64_f64 back to its
pre-calls-issue hash, the 4090 / 5090 / 3090 decode_1024_256 contract / exact rows, the 3090 / H100 PCIe attention,
twin, rq_mlp2, smlp and two cell rows).

## 3. Replay lines

`rung_replay.tsv` +13 lines, each citing its evidence file and raw run:

| rung | part | fill / face | winner | measurement |
| --- | --- | --- | --- | --- |
| post-table | rtx-5080 | U / C | on | f8w `wo` projection 108 → 14.9 µs a step (fp8-leftovers) |
| streamk-epi | rtx-5080 | F / C | on | 8 programs 0.906–0.965, ABBA **5** reps (sweep-runtime-2) |
| stream-mma | rtx-5080 | A / C | on | prefill bf16 3.775 → 2.645, block 10.742 → 9.611 ms |
| kmajor-b | rtx-5080 | A / C | on | decode B32 L8192 bf16kv 2.425 → 1.676 ms, B8 / B1 / f32 likewise |
| unit-uniform | rtx-5080 | A / C | on | llm_pre_t8192 90.52 → 48.89 ms |
| round-store | rtx-5080 | U / C, F / C | on | m128 / m1 bf16 projections 1.11–1.19×, m2048 n6144 1.15×, block prefill 1.08× |
| round-store | rtx-5080 | U / C | **off (open)** | pj_m128_n28672_k4096_bf16 (§4) — the panel depth, not the storage |
| wgmma-ws | h100 | A / T | on | MLP TF32 1.19–1.25×, m128 n6144 / n28672 1.21 / 1.12×, m2048 1.01–1.05× |
| wgmma-ws | h100 | U / T | **off (open)** | pj_m128_n4096_k4096_tf32 0.96× (s2) / 0.87× (s3): 256 CTAs in 2 waves re-read the 64 MB `b` against 50 MB of L2 |
| tma-digits | h100 | A / T | on | slabs → digits: MLP 1.21 / 1.18×, m128 1.48 / 2.04× |
| ws-regs | h100 | A / T | on | even share → `setmaxnreg`: 0.560 → 0.353, 0.348 → 0.307 ms |
| calls-issue | rtx-5080 | U / C | null | the two programs it moved lost 1.77× / 1.48× (12 reps ABBA, timed twins of `px_rt_mlp2_128` / `px_rg_mlp2_128`): the first-launch share (§2) refuses both chains now — ON's text is OFF's |

The two `off` lines are open in `rules::tests::OPEN` with their missing fact: round-store's loss is the tail wave's
residency × the operand's residence against the stage depth (sweep-gemm §2, `mma-depth`'s split); wgmma-ws's is the
operand's re-read across waves against the L2 (hopper.md open 5: a persistent grid in raster order or R6's split reads
`b` once). Neither is the rung's own condition, so neither gets a per-part switch.

## 4. The two evidence claims

- `sweep-runtime-2.md`: the streamk-epi table says "ABBA 5 reps" now. `timed.sh` passed `REPS` 8 to `gpu_run.sh`;
  the probe section repeats `PROBE_REPS` times (default 5) — the raw `sk-epi` / `sk-epi2` results hold rep 1–5 of
  every program and arm; the medians quoted were those.
- `sweep-gemm.md` §2 and its table row: the 12-rep figures (0.3914 / 0.4037) had no raw run. The row is re-measured
  here, changed-programs ABBA, `PROBE_REPS=12` (`gpu_changed.sh --off round-store --stems
  pt_pj_m128_n28672_k4096_bf16`, raw `review-fixes-2/rs12`), and the text now cites it and the traceable `rs/` run.

**Time** (RTX 5080; value gate first: 3 arms pass; GPU at P1, 2865–2925 MHz; Tctl 41–44 °C; competitor from
`matrix.md`, competitors do not depend on Mapal's code):

| workload | OFF | ON | fastest competitor (leg) | explanation |
| --- | ---: | ---: | --- | --- |
| pj_m128_n28672_k4096_bf16 (`round-store`, 12 reps, null 0.3995) | **0.3846** | 0.3996 (1.039×) | 0.3174 (triton-bf16) | kernel 0.383 / 0.399 ms: the stored 2-byte panel fits `bk` 32 × 2 where the f32 one left 16 × 3 (sweep-gemm §2: forced 16 × 4 0.3431) — a loss, replayed `off` and open |
| px_rt_mlp2_128 timed twin (`calls-issue` at d950efb3, 12 reps) | **0.0475** | 0.0842 (1.77×) | — (an interpreter twin; no leg) | the chain's first cooperative launch 57 against 14 µs; now refused (§2): ON's text is OFF's |
| px_rg_mlp2_128 timed twin (same) | **0.0810** | 0.1201 (1.48×) | — | as above |

Every other program on the 5080 has equal text before and after this change (§1, §2): null by construction, not
timed. No rented part was available; the rows §2 moved there are the next rented round's.

## Tests

Workspace subset (`mapal-ir`, `mapal-lower`, `mapal-rewrite`, `mapal-backend-llvm`, 90 test binaries, release,
`RUST_TEST_THREADS=8 MAPAL_TESTGEN_JOBS=4`): **1007 passed, 2 failed, 7 ignored** (raw `tests-3.log`); the two were
this change's intended moves and pass after it — `site_choice::candidate_zero_is_the_pre_sprint_emission` (the c0
golden regolded, 26 rented-part rows, §2) and `large_gemm::streamk_price_on_generated_shapes` (its "no rate → no
stream-K" assertion became "a missing `red` rate prices as the derived one": 17 priced walks of 43 under-filled tile
kernels, every one equal under `--cost=atomic_ops_per_ns=0`).

| test | programs | checks | coverage (all > 0) |
| --- | --- | --- | --- |
| `compute_type::precision_plan_store_narrow` (+2 cases) | hand | an escaping callee's `a`: 0 stores, a call kept; the same callee returning its cell only: 1 | — |
| `half_tile::round_store_on_generated_programs` | 40 (8 in a kept callee returning `a`) | the fact per operand (a returned `a` never stored), graph valid / idempotent, values vs the weakest member's interpreter in 4 emulator modes and the OFF arm; the returned `a` exact | programs 40, stored 31, `a` 2-byte tile 5, `b` 10, FMA reads 12, refused 2, epilogues 6, **escaped 1** |
| `gpu::tests::streamk_price_separates_the_measured_sites` | units | the 5080's sites as before; the A100 / H100 / H100 PCIe / 3090 priced (derived rates > 0), 512³ refused; the 4090 / 5090 / 5080 rates their own | — |
| `flash_attention::stream_priced_on_every_part` | 2 bench programs × 4 parts | the 2048-token prefill streams on the 3090 / A100 / H100 / H100 PCIe (never `unpriced`); `px_attn_64_16`'s values under each part's own rule in 4 emulator modes | — |
| `calls::counted_trips_share_the_first_cooperative_launch` | 2 bench programs | `trip_split` = one 2-trip loop; no chain, no cooperative launch shipped; `--calls=force` still forms it | — |
| `calls::first_cooperative_launch_shared_by_counted_trips` | unit | whole at 1 launch, half at 2, decreasing in trips, none for unknown trips | — |
| `rules::tests::rung_rules_replay` | 243 lines | 140 picked / 72 null / 10 mixed / 21 open, `OPEN` = the 21 | — |

Each new assertion was run against the code without its fix and failed (escaping store 1 ≠ 0; the 3090's prefill
`unpriced`; the 2-trip chain formed, `per_ns=3180` against `sep_ns=4762` with no first-launch term; the generated
round-store program's device `y` 0 against −21.2). `rustfmt --check` clean on every changed file; `cargo clippy` adds
no warning on a changed line. No `.mapal` source is committed (the timed twins live in the raw directory).

## Open problems attacked

| problem | result |
| --- | --- |
| round-store on an escaping array | **closed generic** (the escape fact) |
| an epilogue group's escaping output never stored (found) | **closed generic** (`read_outside` reads the escape fact) |
| streamk-price / stream off where unmeasured | **closed generic** (`Gpu::plan_cost`, `TargetProfile::priced`); the moved rented rows are unmeasured — next rented round |
| the first cooperative launch free in a loop and where unmeasured (found) | **closed generic** (`trip_split`'s trips share it; `chain_cost` derives it) |
| replay lines for the measured rungs | **closed** (13 lines) |
| round-store's m128 n28672 loss; wgmma-ws's m128 n4096 loss | **attempted, open**: both are named as their missing fact (panel depth × tail-wave residency; the operand's re-read across waves) in `OPEN`; neither can be measured on a part here (the H100) or pinned from five points (the depth) |
| calls-issue has no measured gain on any part | **open**: its term is a host property (issue time); on the 5080 it moves nothing now; the 4090 / 5090 decode chains it formed at fp8-leftovers are refused by §2's share and are the next rented round's rows |
| no tailored fix | — (`docs/tailored-fixes.md` unchanged) |
