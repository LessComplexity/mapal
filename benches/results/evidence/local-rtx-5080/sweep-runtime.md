# sweep-runtime — open problems in memory / launches / runtime / rules / harness (RTX 5080, 2026-11-23)

Branch `sprint/sweep-runtime` from `local-rtx-5080` @ `a31cd014`. Machine shared with other sweep agents the whole
time: load 15–98, Tctl up to 91 C from their work; every build / test / bench of this step pinned to CPUs
24-31,56-63 (`nice -n 5`, `CARGO_BUILD_JOBS=12 RUST_TEST_THREADS=8 MAPAL_PAR=2 MAPAL_TESTGEN_JOBS=4`), the one GPU
run under the bench lock. Raw output (not in git): `.claude/raw-evidence/local-rtx-5080/sweep-runtime/`
(`ab-epilogue/` the GPU A/B, `harness/` the test logs, the scripts).

## Ranked items and status

Cost = time lost × rows × parts (matrix.md, integration.md); the harness item is the user's ask (the agents' wall
time: the integration's workspace test run alone was 45 min).

| # | item (source) | rows × parts | status |
| --- | --- | --- | --- |
| 1 | fan-in/out + MLP rows lose 4–17 %: an epilogue group's site forfeits stream-K (matrix: fan/loops geomean 1.06–1.21, MLP 1.10–1.55) | 8 rows × 5 | **attempted, open** — cause measured, the priced rule replayed (9 / 13), not shipped (§1) |
| 2 | prefill peak 1.2–1.6× torch (cross-arch "Peak device memory", caching-memory open 3) | 3 rows × 5 | **T2048 / block prefill closed by flash-attention** (measured: 0.42× / 0.74× torch); **T8192 open** (1.56×: §2) |
| 3 | test wall time (harness.md open: serial loops; integration: 45 min sequential binaries) | every step | **closed generic** (4 commits, §3) |
| 4 | evidence: `by-ref:runtime trace allocs == predicted` (caching-memory standing item; 160 misses a part on the rented rounds) | 282 checks | **closed** (§4) |
| 5 | harness: every bundle stopped at `gen_tiles.py` (a stale `mapal` on PATH) | every bundle | **closed** (§3) |
| 6 | tune pick variance (integration: pre_t8192 89.6 vs 79.0 ms; indep_512's tie) — fp8's in-context confirmation M3 | 2 rows | not reached |
| 7 | ~1 µs runtime residual a small program (integration) | ~20 rows | not reached |
| 8 | `pdl`'s sign on short programs (hardware-properties open 1), `calls` pricing (cross-arch open 6) | — | not reached |
| 9 | convmc 1.68–2.25× on the rented parts (0.76 here: torch's sm_120 cuDNN pick is 2× its 4090 time), loopmm 1024³ 1.11–1.37 (64 tiles on 168 slots) | 2 rows × 4–5 | not reached (GEMM / conv tile geometry) |

No tailored fix; nothing for `docs/tailored-fixes.md`.

## 1. Epilogue groups and stream-K (attempted, open)

**Cause (SASS / plan evidence).** In `indep_2048` the two 2048³ sites get different kernels: the plain one
`76c66be4` is `combine=streamk ctas=168 iters=196` (0.456 ms), the one carrying the join `g = c + d` as its epilogue
group `ebb48a23` is `combine=none` (0.531 ms): same tile (8 × 16, bk 16, 2 blocks / SM, 128 accumulators, no spill,
occupancy as planned — every evidence check ok on both). 256 tiles on 168 slots = 1.52 waves: the ordered kernel
idles half its second wave, stream-K does not. An S54 epilogue group runs `Order::Sequential` (`group_order`: only
a `tile-post` group may split, into a workspace whose reduce evaluates the epilogue), so the site can never take
stream-K. Same in diamond / fanout (two of three, one of two sites) and the 2048 MLPs (bias + relu).

**A/B (`gpu_changed.sh --off epilogue`, 17 programs differ, 8 ABBA reps, value gates pass):** epilogue OFF = the
site plain (stream-K where the rules give it) + its members as their own cell launch.

| program | ON ms | OFF / ON | absorb_pays (plan model) says | right? |
| --- | ---: | ---: | --- | --- |
| fanout_2048 | 0.9879 | **0.941** | split | yes |
| rt_mlp2_2048 | 16.72 | **0.914** | split | yes |
| indep_2048 | 1.0116 | **0.959** | split | yes |
| mlp2_2048 / _tf32 | 1.9499 / 1.5697 | **0.959 / 0.964** | split | yes |
| diamond_2048 | 1.4519 | **0.971** | split | yes |
| mlp_2048 / _tf32 | 0.9445 / 0.7872 | **0.982 / 0.976** | split | yes |
| rt_mlp2_1024 | 2.6019 | 1.032 | absorb | yes |
| smlp_4096_256_1024 | 0.1680 | 1.219 | absorb | yes |
| ewconvew_2048_5 | 0.0819 | 1.138 | (no tile candidates: absorb) | yes |
| epi_4096 / loop_4096 | 3.5879 / 28.65 | 1.064 / 1.046 | split (stream-K 3.40 vs ordered 3.57 ms) | **no** |
| indep_512 / mlp_512 | 0.0411 / 0.0403 | 1.473 / 1.471 | split (atomic S=2 12.5 vs 18.6 µs) | **no** |
| firew, prefill | — | 1.002 / 1.005 (null 0.97) | — | null |

A mixed result: the group wins at 4096 / 1024 / 512 and loses at 2048. The rule that decides it exists —
`absorb_pays` (resident.rs, plan-cost design #10: the unsplit site + the members' loads against the split + one
dependent op a member) — but only under `plan-cost`, which ships on no part, so every group is absorbed. Replayed
here without that gate (a temporary emitter, not committed): **9 of 13 right**. The four misses are the plan
model's, not the rule's: it prices stream-K at 6 waves 4.5 % faster than the ordered kernel where the measured
split + member is 6.4 % slower (`wide-fma`'s 6-wave epilogue tiles, large-gemm open 1), and the 512 sites' atomic
split at 0.67× where its zero pass and the member's extra launch measure +19 µs on a 40 µs program (kernel-merging's
1.4–1.9× small-tile optimism). Shipping `absorb_pays` as it is would trade +4–9 % on six rows for +5–47 % on four.

**Open, what closes it.** Either (a) a stream-K epilogue for groups — the tile's last CTA (ticket, as `split-kv`'s
combine) applies the chain to the whole sum, or for a chain affine in the site value (`fold·α + r`, α a power of
two: indep / diamond / fanout) the CTA owning `k`-segment 0 adds `r` and every partial is scaled by α (atomic
safe, the order-free face's reassociation only) — which removes the trade-off; or (b) the plan model's stream-K
fixup and small-launch terms fitted (plancost.py over a 4096 / 512 sweep), after which `absorb_pays` ships with its
own rung. (a) is tile codegen (`gpu.rs`, the sweep-gemm area: cross-arch open 1 `stream-fly` on post sites is the
same capability); (b) the plan model's refit. Rows it would move on the 5080 (OFF arm today): fanout 1.08 → ~1.02,
indep_2048 1.09 → ~1.06, diamond 1.06 → ~1.03, rt_mlp2_2048 1.04 → ~0.98, mlp2_2048 1.11 → ~1.07.

## 2. Prefill peak memory

From the integration run's traced processes (`resM-final`, `mapal_gpu_peak`), against caching-memory's torch peaks:

| row | before flash-attention (MB) | integration (MB) | torch (MB, leg) | Mapal / torch |
| --- | ---: | ---: | ---: | ---: |
| pt_llm_pre_t2048 | 1296 | **353** | 841 (compile) | 1.54 → **0.42** |
| pt_llm_blkpre_t2048 | 2032 | **1125** | 1513 (compile) | 1.34 → **0.74** |
| pt_llm_pre_t8192 | 2852 | 2991 | 1916 (compile) | 1.49 → 1.56 |

T2048 and the block prefill: the scores and probabilities are no longer stored (flash-attention's stream unit), so
the arena lost them — closed, with no change here. **T8192**: the attention runs in the KV-group loop
(uniform-base sites, flash-attention open 3 / S4: the stream unit takes no captured number yet), so each group
still stores `s` and `pe` (4 × 8192² f32 = 1 GiB each). The arena (2 991 MB = `mapal_gpu_arena`'s 2 990 604 288 B) is
the straight-line members plus the loop's buffers, each loop-cone buffer live from its allocation to exit
(`entry_arena`: "per-loop spans … when that footprint matters"), so `s` and `pe` never share bytes with each other
or with anything the trip frees. Two generic routes, neither taken here: S4 (sweep-attention is on the T8192
program: it removes both) or per-trip coloring of a loop's cone buffers plus a same-cell in-place output (`pe[e]`
reads `s` only at `e`, `s` dead after it: −1 GiB). Status: open, waiting on S4.

## 3. Harness: the tests' wall time (closed)

Where the time went (generic_fusion, 1 082 clang runs: 197 s of clang for a 52.6 s binary; clang ≈ 95 % of its
CPU):

| commit | change | measured (same pin and env) |
| --- | --- | --- |
| `fea4aecc` | the emulator compiled once per clang + source text into the tests' tmpdir; every emulated program links the object (was: `mapal_gpu_emu.c` compiled inside every program's link, 0.245 s each) | generic_fusion 52.6 → 18.3 s, clang summed 197 → 60 s; coverage lines identical |
| `ed262036` | `-fuse-ld=lld` where `ld.lld` is installed (`MAPAL_EMU_LLD=0`: the default linker); `benches/local/test_par.py` | links 47.0 / 46.6 → 24.4 / 23.4 s summed, wall 18.0 / 17.9 → 12.4 / 11.9 s (ABBA); identical lines |
| `774cc20a` | site_geometry's five emulated loops through `testgen::par_check` (`par_count`: summed counters) | 163 → 104 s (its kmajor-a test alone was 166 s); identical lines |
| `17cde58c` | `test_par.py`'s thermal guard holds new binaries instead of SIGSTOPping running ones; `gen_tiles.py` formats with this checkout's CLI | a stopped group failed its own wall-clock timeouts on resume (bulk_stage: "emulated run timed out after 120 s"); a Nov 20 `mapal` on PATH rejected the generator's syntax, stopping every bundle |

Tried and dropped: a per-launch thread pool in the emulator (threads made once a launch instead of once a block):
3.2 → 0.21 ms a 256-thread block in a micro-benchmark, but no change on the tests (exp 21.2 / 21.0 s, generic_fusion
56.4 / 56.2 s): their launches are a few blocks and clang dominated. Reverted.

`test_par.py` (cargo test runs its binaries one after another; the integration's 45 min had 6.9 of 32 CPUs busy on
average): builds the binaries once, runs N at a time longest-first by the last run's times, each in its package
directory, doc tests as one job; the binaries and environment are cargo's. Full workspace, this branch, this step's
16 CPUs, 3 binaries at once, under the other agents' load (15–98) and heat: **1651 s wall, of which 761 s the
workers held for Tctl > 85 C**; the per-binary walls sum to 2461 s. The integration's sequential run of the same
suite: 2692 s on 32 CPUs at `RUST_TEST_THREADS=16`. Per binary, even while sharing the CPUs three ways: split_tile
497 → 259 s, site_geometry 194 → 104 s, hopper_tma 230 → 296 s (six of its seven generated loops are still
serial: the next conversion), half_tile 137 → 248 s, long_rows 146 → 189 s (contention).

## 4. Evidence checker: one arena allocation predicted as two (closed)

`move_facts` counted `@mapal_gpu_arena(` anywhere in a function's text; the module's `declare` lines land in the
chunk of whichever function precedes them, so the runtime check predicted two allocations where the entry arena
makes one. Counting the call (`call i64 @mapal_gpu_arena(`): the integration run's 434 traced checks re-judged
(`alloc_recheck.py` over `bM2` / `bS2` and `resM-final` / `resS-final`) — 282 MISMATCH → ok, the 152 ok stay ok, none
left. The rented rounds' 160 a part are the same class.

## Tests

Full workspace through `test_par.py` on this branch: 124 binaries, **1502 passed, 0 failed** (7 ignored, as before),
doc tests included. Coverage: every line generic_fusion (12 tests) and site_geometry (7) print on the fixed seed is
identical before / after each harness change (`harness/gf.lld*.log`, `sg.old.log` / `sg.new2.log`); the GPU A/B's
value gates: 17 programs, every arm within its bound (`ab-epilogue/results-rtx-5080.txt`). No IR fact or rule was
added, so no new generated-program property test; the evidence fix is checked on the integration run's 434 traced
rows.

## For integration

- README (local-rtx-5080, after **integration**): **sweep-runtime** (`sweep-runtime.md`): tests 3–4× faster per
  emulated binary (the emulator compiled once, lld) and run side by side (`benches/local/test_par.py`);
  site_geometry's loops in parallel; the runtime alloc check's false mismatch fixed (282 → 0); bundles no longer
  stop on a stale `mapal` CLI. Prefill peak: T2048 0.42×, block prefill 0.74× torch since flash-attention; T8192
  1.56× until S4. Epilogue groups forfeit stream-K on 2048 sites (fan / MLP rows 4–9 %): measured, the priced rule
  replayed 9 / 13, open (stream-K epilogue or the plan model's refit).
- CLAUDE.md facts table: no new fact. Tailored-fix registry: no row.
