# Open problems — attention, GEMM, runtime (after the sweep merge, RTX 5080, 2026-11-24)

One table across the three sweeps' areas (`sweep-attention.md`, `sweep-gemm.md`, `sweep-runtime.md`) and what the
integration found (`integration.md` § Sweep merge). Status: **closed** (generic: fact / property / capability / rule;
no tailored fix in any of them), **attempted-open** (built or measured, not closed, the blocker named), **open** (not
attempted in these steps). Rows moved: RTX 5080 rows whose time the item changed, integration 1 → sweep merge (ms,
same protocol; `matrix.md` for the legs).

| area | item | status | rows moved |
| --- | --- | --- | --- |
| attention | T8192 attention inside the KV-head-group loop not streamed | **closed** (`FactArms::unit_uniform`, `TilePost::after`; OFF `--gpu-off=unit-uniform`) | `llm_pre_t8192` 89.55 → 47.93 (0.90× Liger, was 1.69×) |
| attention | facts the generated head-group loops broke: `d = T` consumer taken for the producer; one-head row reads `Reach` could not prove; a carried sum after the consumer | **closed** (`stream_plan` roles, `post_rows` digit forms) | generated programs (39 loops, 31 streamed) |
| attention | the stream kernel spilled (ptxas capped it at 128 registers) | **closed** (`nvvm.minctasm` from its geometry, under `min-blocks`) | `llm_pre_t2048` 7.22 → 6.60, `_bf16w` 7.15 → 6.51, `_tf32` 6.07 → 5.68 |
| attention | bf16-weight prefill (f32 × bf16 on FFMA; `Bf16x3n` bounded, unemitted) | attempted-open: the rate rule refuses it on GeForce parts; needs an A100 / H100 round | — |
| attention | tensor cores in the stream kernel (`_bf16`, `_tf32` prefill) | open: a second emission in the fragment layout; TF32 = FFMA rate on the 5080 | — |
| attention | decode attention as a stream (captured position in the mask, batch-moved K window, bf16 cache windows) | attempted-open: groups form, the backend refuses them (three named reasons) | — |
| attention | small programs (lsm 3 launches, `attn_1024_64` split under-priced) | open (µs launch latency) | — |
| GEMM | f32 activation rounded per fragment in bf16 tiles | **closed** (`precision_plan.store` / `narrow_stored`, rung `round-store`, OFF `--gpu-off=round-store`) | 23 `_bf16` programs: m128 0.82–0.89, m2048 n6144 0.87, m16 0.89–0.95, m1 0.92–0.96, blkpre / blkdec 0.92 / 0.95 |
| GEMM | evidence: sm_120's `FHFMA.BF16` not counted as FFMA (92 false mismatches) | **closed** (`sass_stats.py`) | — |
| GEMM | `pj_m128_n28672_k4096_bf16` slower under round-store (the panel rule's bk 32 × 2) | attempted-open: bk 16 × 4 is 12 % faster, `mma-depth` loses on 4 other sites; the deciding fact (a partial last wave at one block an SM × DRAM residence) unpinned | 0.3844 → 0.3995 (reproduced) |
| GEMM | m1 f32 GEMVs (1.04–1.89×) | attempted-open: the L2 knee does not bind; the per-call zero + synchronize remains (an in-block split would remove it) | — |
| GEMM | m128 TF32 projections | open: swizzled 96 KiB panels or a TF32-exact array fact | — |
| runtime | epilogue groups forfeit stream-K (fan / MLP rows 4–9 %) | attempted-open: `absorb_pays` replayed 9 / 13; needs a stream-K epilogue or a refit plan model | — |
| runtime | prefill peak memory (T2048 / block prefill) | **closed** (by flash-attention) | peaks 0.42× / 0.74× torch |
| runtime | T8192 peak memory (scores / probabilities stored per group) | **closed** by sweep-attention's stream, measured at the merge | 2991 → 776 MB, 1.56× → 0.41× torch |
| runtime | the tests' wall time | **closed** (emulator compiled once, lld, `test_par.py`, site_geometry in parallel) | integration's workspace run 45 → 16 min |
| runtime | evidence: one arena allocation predicted as two (`trace allocs == predicted`) | **closed** (`evidence.py`: an arena is a call site) | 282 checks → ok |
| runtime | bundles stopped at `gen_tiles.py` on a stale `mapal` | **closed** (the checkout's own CLI) | — |
| runtime | the tune's pick varies between arms and sessions (fp8's M3, in-context confirmation) | attempted-open, seen a third time: `llm_dec_b32_l2048` 1.047 on equal code (site 2: candidate 3 vs 0) | `llm_dec_b32_l2048` 16.53 → 17.30 |
| runtime | ~1 µs runtime residual; the 5080 column's small-program session gap | **closed** in part: the timed CPU set (CPPC ceilings 211–241 vs 166; 0–7 vs 8–15: 0.81–1.00×) and `lsm_32_8192`'s per-process modes; the residual 1–5 % between equal-set binaries stays open | lsm 0.72–0.77, small GEMVs / ew 0.85–0.93 (session, equal code) |
| harness | `test_par.py`: 15 binaries in 7 shared log names | **closed** (package directory in the name) | — |
| harness | timed scripts' busy check read CPUs 0–7 whatever `CPUS` was; the set's clock unrecorded | **closed** (`timed_cpus.sh`, `cppc` in `conditions.txt`) | — |
| harness | `llm_dec_b1_l2048_f8w` 0.892 (two earlier sessions) against 1.15 ms (two here) with equal code and picks | attempted-open: neither the CPU set nor the binaries (ABBA 0.999); a machine state the conditions do not record | 0.892 → 1.152 (session) |
| runtime | `pdl`'s sign on short programs, `calls` pricing, convmc on rented parts, loopmm 1024³, remaining serial test loops | open (not reached) | — |
