# Open problems — attention, GEMM, runtime (after the second sweep merge, RTX 5080, 2026-11-24)

One table across the areas of both sweeps (`sweep-attention.md`, `sweep-gemm.md`, `sweep-runtime.md`,
`sweep-attention-2.md`, `sweep-gemm-2.md`, `sweep-runtime-2.md`) and what the two integrations found (`integration.md`
§ Sweep merge, § Second sweep merge). Status: **closed** (generic: fact / property / capability / rule; no tailored
fix in any of them — `docs/tailored-fixes.md` unchanged), **attempted-open** (built or measured, not closed, the
blocker named), **open** (not attempted, the reason named). Rows moved: RTX 5080 ms, the step's own run or the
second merge's full suite (`matrix.md` for the legs).

| area | item | status | rows moved |
| --- | --- | --- | --- |
| attention | T8192 attention inside the KV-head-group loop not streamed | **closed** (`FactArms::unit_uniform`, `TilePost::after`; OFF `--gpu-off=unit-uniform`) | `llm_pre_t8192` 89.55 → 47.93 (0.90× Liger) |
| attention | facts the generated head-group loops broke (`d = T` consumer, one-head row reads, a carried sum after the consumer) | **closed** (`stream_plan` roles, `post_rows` digit forms) | generated programs (39 loops, 31 streamed) |
| attention | the stream kernel spilled (ptxas capped it at 128 registers) | **closed** (`nvvm.minctasm` from its geometry) | `llm_pre_t2048` 7.22 → 6.60 |
| attention | tensor cores in the streamed kernel, `: bf16 \| f32` prefill | **closed** (`stream-mma`: `stream::mma_fmt`, `TileSite::half` × `Caps::matrix_op`; OFF `--gpu-off=stream-mma`) | `llm_pre_t2048_bf16` 3.774 → 2.636 (1.95× → 1.36×), `llm_blkpre_t2048_bf16` 10.74 → 9.59 |
| attention | the decode cache read (the scores kernel copied its k-major keys one element a thread down a column) | **closed** (`kmajor-b`: `kvec_b`, vectors along each key's row; OFF `--gpu-off=kmajor-b`) | `llm_dec_b32_l8192_bf16kv` 2.430 → 1.657 a step (1.60× → 1.12×), B8 0.735 → 0.511, f32 B32 L8192 3.586 → 2.800 |
| attention | stream-mma at 40 % of the bf16 peak (f32-staged windows, no `ldmatrix`, the model's 32-row blocks) | open (found by sweep-attention-2; not attempted) | — |
| attention | bf16-weight prefill (f32 × bf16 on FFMA; `Bf16x3n` bounded, unemitted) | attempted-open: the rate rule refuses it on every GeForce part (3 bf16 products at 512 cost more than FFMA at 256); the A100 / H100 have no stream plan rates until a rented round measures them | — |
| attention | TF32 prefill attention on the matrix product | attempted-open: needs a quad permutation of P; no part with plan rates has TF32 above FFMA where the stream is priced | — |
| attention | decode attention as one stream (captured position, batch-moved window, narrow windows) | attempted-open: `kmajor-b` removed the larger cost it was meant to attack; the groups still form and are refused | — |
| attention | small programs: lsm (3 dependent launches), `attn_1024_64` (split under-priced) | attempted-open: lsm needs a host-time-per-dependent-launch property; `attn_1024_64` the plan model's optimistic small-tile `site_ns` | `attn_1024_64` 0.0433 → 0.0394 (session) |
| attention | naive-cache twins vs the HF-cache program (1.19–1.54×) | open: the demand fact, not reached | — |
| GEMM | f32 activation rounded per fragment in bf16 tiles | **closed** (`precision_plan.store`, rung `round-store`) | 23 `_bf16` programs 0.82–0.96 |
| GEMM | evidence: sm_120's `FHFMA.BF16` not counted as FFMA | **closed** (`sass_stats.py`) | — |
| GEMM | m1 f32 GEMVs: the per-call zero's driver set-up in the clock | **closed** (`warm-zero`, init flag 32; OFF `--gpu-off=warm-zero`) | `pj_m1_n2048_k7168` 0.0307 → 0.0249, `pj_m16_n2048_k7168` 0.0460 → 0.0387, `pj_m1_n4096_k4096_bf16` 0.0336 → 0.0231 |
| GEMM | m1 f32 GEMV kernels' L2-edge geometry (doubling the split wins only at 64 MB of weights) | attempted-open: whether a just-written operand is still in L2 is not pinned as a fact | `pj_m1_n4096_k4096` 1.83× lt-fresh |
| GEMM | loopmm 1024³ / under-filled stream-K (site-level wave quantization) | **closed** on parts whose profile has the rates (`streamk-price`; OFF `--gpu-off=streamk-price`); open on the 3090 / A100 / H100: no `red` rate (`calib.cu` has no mode for it) | `loopmm_1024_8` 0.6494 → 0.6117 |
| GEMM | m128 TF32 projections (per-fragment TF32 rounding: compare + predicated add) | attempted-open: a "TF32-exact f32" storage fact (round once at the producer) not built | — |
| GEMM | `pj_m128_n28672_k4096_bf16` panel depth | attempted-open: bk 16 × 4 is 12 % faster but `mma-depth` loses on 4 other sites; not re-attempted (the TF32 point contradicts the candidate fact) | 0.3995 → 0.3991 |
| GEMM | mlp_512 / indep_512 (their atomic split's zero) | **closed** in part (`warm-zero`); mlp_512 is bimodal by process (on 0.0394 / null 0.0366 in one batch) | — |
| GEMM | bf16 projections on A100 / H100, convmc on the rented parts, m16 on the 4090 | open: rented hardware, no round this sweep | — |
| runtime | epilogue groups forfeit stream-K (fan / MLP rows 4–9 %) | **closed** (`streamk-epi`: the stream-K ticket; OFF `--gpu-off=streamk-epi`). Its state is the call site's workspace (`ws_slot`, zeroed per launch), not a module-global ticket: sweep-gemm-2's race (one kernel on two streams) does not arise | `rt_mlp2_2048` 16.27 → 14.65, `fanout_2048` 0.985 → 0.918, `mlp_2048` 0.947 → 0.882, `loop_4096` 28.33 → 26.88 |
| runtime | an ordered launch behind a stream-K plan launched the stream-K tile's geometry | **closed** (`c113cc18`) | — |
| runtime | merge: a group at a priced stream-K site (`streamk-price` × `streamk-epi`: `rt_mlp2_1024`, `smlp`) | **closed**: the ticket walks the priced site (plans on generated GEMMs, 16 priced tickets); timed in § Second sweep merge | see `integration.md` |
| runtime | merge: a module whose only zero is the ticket's workspace did not warm the memset path (`module_zeroes` read `zero` only) | **closed** (`module_zeroes` counts a workspace without a reduce kernel) | `epi_4096`, `loop_4096` init word 9 → 41, 13 → 45 |
| runtime | the tune's pick varies between sessions | **closed** (`confirm_pick.py`, median of the in-program pairs; OFF `CONFIRM=0`) — run live in the second merge's full suite: 13 confirms in the suite (7 kept, 6 reverted) | — |
| runtime | `pdl`'s sign on short programs; `calls` pricing | **closed** earlier (verified by sweep-runtime-2) | — |
| runtime | prefill peak memory (T2048 / block prefill); T8192 peak | **closed** (flash-attention; sweep-attention's stream) | peaks 0.42× / 0.74× / 0.41× torch |
| runtime | the tests' wall time | **closed** (emulator compiled once, lld, `test_par.py`) | 45 → 16–17 min |
| runtime | `llm_dec_b1_l2048_f8w` 0.89 vs 1.15 ms on equal code | attempted-open: this run 0.848 ms (its siblings moved ≤ 3 % under `kmajor-b`, so the 0.74× is the state, not the code) with the GPU at P0 / P1 throughout (5848 samples, 2 at P8) — the first full run with clocks; the slow state's clocks are still unrecorded | 1.1517 → 0.8477 |
| runtime | first 4-byte host↔device copies 11–21 µs cold (sweep-gemm-2 §7) | open: not reached; bounded — only a program with a copy inside its clock pays it, once (the GEMV rows' copies run after the clock) | — |
| runtime | the ~1 µs runtime residual between equal-set binaries (1–5 %) | open: not reached (needs a host-issue trace per launch on idle CPUs) | — |
| harness | evidence: a stream-K piece loop's epilogue taken for the hot loop (the ticket kernels' FMULs out-counted the k loop) | **closed** (`evidence.py` / `sass_stats.py` `innermost`: descend into a nested loop holding most of the products) | bS −57, bM −32 mismatches |
| harness | evidence: an epilogue group's order (`epi_order`) read as a rung mismatch; a table-launched group read as not launched over n pointers; one bk-step count among several `FullyUnrolled` | **closed** (`arch-rungs`, `epilogue`, `unroll` checks; `full_unrolls`) | bS −35, bM −2; all fixes: bS 321 → 229, bM 5561 → 5481 |
| harness | evidence: `kmajor-b`'s 8-byte bf16 vectors read as a widened operand (`narrow-tile`) | **closed** (the radix branch counts `LDG.64`) | bM −46 |
| harness | evidence: `kmajor-b` kernels at i32 indices take one `IMAD.WIDE` a row vector + 1 (34 / 33 LDG.64: `idx32`, 56 rows) | open (found at the merge): the per-row 64-bit address from an i32 row index; one base + 32-bit row offsets not tried | — |
| attention | `kmajor-b`'s B1 decode copies: the next panel's vector LDGs consumed before the math that should cover them (`prefetch` cover 0, 19 kernels) | open (found at the merge) | — |
| harness | `horizontal` test: `rt_mlp2_1024`'s refusal (`waves:1`) stale under `streamk-price` | **closed** (the shipped rule streams both GEMMs; the refusal asserted with `streamk-price` off) | — |
| harness | the full suite / model scale did not record GPU clocks (only `gpu_run.sh` did) | **closed** (`model_scale_run.sh` → `gpu-clocks.csv` / `.txt`) | — |
| harness | `test_harness.py`: 6 failing expectations (vast wrapper mock, a missing `_xabs` K file, `rt_indep_1024`'s clock / retention markers, `search.py`'s `--splitk=2` on `rowsum_2048`) | open: present before this merge (5 at `4c29387e`; the sixth skipped there without an emitter, the same split from its emitter); not reached | — |
| harness | `test_par.py` shared logs; the timed CPU set | **closed** (integration 2) | — |
