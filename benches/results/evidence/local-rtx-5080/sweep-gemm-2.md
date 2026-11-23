# sweep-gemm-2 — the second open-problems sweep in GEMM cells / projections / MLP chains / GEMVs / tensor tiles (RTX 5080, 2026-11-24)

Branch `sprint/sweep-gemm-2` from `local-rtx-5080` @ `4c29387e` (integration 2): `1f102a34` (warm-zero), `81a13cfa`
(streamk-price), this file. RTX 5080 (sm_120, 84 SMs, 64 MB L2), CUDA 13, LLVM 23. Raw output (not in git):
`.claude/raw-evidence/local-rtx-5080/sweep-gemm-2/` (`warm-zero/`, `streamk-forced/`, `streamk-price/` — results,
evidence, suite diffs; `tools/` — the probes: `zprobe.cu`, `zfirst*.cu`, `zthread.cu`, `cfirst.cu`, `kprobe.cu`).

**Conditions.** Several agents shared the box: builds and tests on CPUs 16–23 / 48–55 (`CARGO_BUILD_JOBS=12
RUST_TEST_THREADS=8 MAPAL_TESTGEN_JOBS=4 MAPAL_PAR=2`, `nice 5`), timed processes on 16–23 under the bench lock (one
ABBA batch a hold). Load 14–67 from the other agents and Tctl 76–93 °C during the session (heavy local work paused
above 85 °C; the timed runs are GPU-bound). Every A/B is same-session ABBA with a `null` arm (the ON binary again);
decisions read ratios. The changed-programs flow ran over the 125 matrix programs (`suite_diff.py --stems`), not
only `suite.txt`'s 35: the area's rows are mostly outside the suite.

## The area's open items, ranked by cost (Σ ln(Mapal / best) over the five parts, `matrix.md`) and their status

| item (source) | cost | status here | rows moved (5080) |
| --- | ---: | --- | --- |
| m = 1 f32 GEMVs 1.04–2.23× (sweep-gemm §3: "per-call zero + synchronize") | 9.2 | **closed in part, generic** (§1 `warm-zero`: the zero's driver init in the timed region); the kernel's L2-edge geometry attempted, open (§3) | pj_m1_n4096_k4096 0.0384 → 0.0311 (1.71 → 1.38× lt-fresh), n2048_k7168 0.0355 → 0.0275 (1.03×) |
| m128 TF32 projections 1.19–2.99× (sweep-gemm §4) | 8.3 | attempted, open: cause measured (§4), the TF32 round-store not built | — |
| bf16 projections on A100 / H100 (2–3.8×) | ≥ 10 | not reachable here (no rented round this sweep; the H100's missing 16-bit `wgmma`) | — |
| loopmm 1024³ 1.10–1.37× (sweep-runtime §9), under-filled stream-K ("site-level wave quantization", `rules.rs`) | 1.5 | **closed generic on parts with the rates** (§2 `streamk-price`); the 3090 / A100 / H100 rows need the `red` rate measured (open) | loopmm 0.6492 → 0.6121, rt_mlp2_1024 2.673 → 2.512, smlp 0.1650 → 0.1587 |
| epilogue groups forfeit stream-K (fan / MLP rows 4–9 %) | 1.0 | attempted, open: the ticket design's blocker named (§5) | — |
| 2-byte panels' depth (`pj_m128_n28672_k4096_bf16` 1.26×) | 0.3 | not re-attempted: the TF32 point at the same shape contradicts the candidate fact (§6) | — |
| mlp_512 / indep_512 1.14–2.08× | 0.9 | **closed in part** (§1: their atomic split's zero) | mlp_512 0.0392 → 0.0371, indep_512 0.0392 → 0.0380 |
| first-call copy costs (found here, the runtime area) | — | measured, handed over (§7) | — |
| convmc on the rented parts, m16 on the 4090, H100 bf16 | — | not reached (rented hardware) | — |

## 1. `warm-zero` — the driver's memset path initialized before the clock (closed, generic)

**Cause.** The m = 1 GEMV (`pt_pj_m1_n4096_k4096`, atomic split S = 32) traced `op=zero ms=0.011` of a 0.042 ms
program; CUPTI: the zero's device time 0.5–0.8 µs, but 7.4–8.7 µs from the memset's start to the kernel's. A probe
with the zero removed (`NOZERO`, wrong values, timing only) ran 0.0296 against 0.037 ms. Hand probes on the bare
driver (`zfirst*.cu`, `zthread.cu`, fresh processes):

| first `cuMemsetD8Async` of the process | host API time |
| --- | ---: |
| cold, ≥ 8 KiB (whatever size comes first: 8 KiB 17.6, 16 KiB 13.0 µs) | 13–17 µs |
| every later one, any size, fresh buffer or not | 1.4–3.8 µs |
| first fill < 8 KiB (4 B) | 6.3 µs |
| after one 8 KiB–1 MiB fill on the same thread | 2.0–3.8 µs |
| after one fill on another thread (the runtime's init thread) | 7.2–8.0 µs |

The driver initializes its memset path lazily, partly per thread: a one-call program pays it inside its clock.

**Rung `warm-zero`** (host, on; OFF `--gpu-off=warm-zero`): the emitter sets init flag 32 where the module's
launches zero an output (`module_zeroes`: a `mapal_gpu_zero` call, or a site table row with zero bytes); the runtime
fills a kept 64 KiB scratch buffer (a 4-byte and a 64 KiB fill) on the init thread, then again on the thread that
loads the module (before the first site, as `choice_ready_dev` does for the site choice). `warm_ms` in the trace
total (0.10–0.20 ms, outside the clock). In-situ the traced zero went 11–12 → 3–6 µs.

**Changed programs:** 92 of the 125 differ (the init word only; device IR equal). Timed: the 21 under 0.5 ms
(8 reps ABBA; value gates 0 failed; the session's null threshold 11.6 %, the m16 n2048 row's own; longer programs
gain ≤ 10 µs once, below their null):

| workload | OFF | ON | null | fastest competitor (leg) | ON / best (was) | explanation |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| pj_m1_n4096_k4096 | 0.0384 | **0.0311** | 0.0329 | 0.0225 (lt-fresh) | 1.38 (1.93) | the zero's first-fill init out of the clock; kernel unchanged |
| pj_m1_n2048_k7168 | 0.0355 | **0.0275** | 0.0284 | 0.0266 (cublas-fresh) | 1.03 (1.15) | same |
| pj_m1_n4096_k4096_bf16 | 0.0301 | **0.0232** | 0.0253 | 0.0492 (torch-bf16-compile) | 0.47 | same |
| pj_m1_n6144_k4096 / _bf16 | 0.0916 / 0.0313 | 0.0873 / 0.0295 | 0.0862 / 0.0302 | 0.0512 (cublas-fresh) / 0.0635 | 1.71 / 0.46 | same; n6144's kernel is the gap (§3) |
| pj_m16_n2048_k7168 | 0.0465 | **0.0392** | 0.0437 | 0.0492 (cublas-fresh) | 0.80 (0.93) | same |
| pj_m16_n4096_k4096 / _bf16 | 0.0549 / 0.0487 | 0.0509 / 0.0462 | 0.0550 / 0.0460 | 0.0553 (cublas-fresh) / 0.043 | 0.92 / 1.07 | same |
| pj_m16_n6144_k4096_bf16 | 0.0563 | 0.0514 | 0.0535 | 0.0635 (torch-bf16-compile) | 0.81 | same |
| pj_m128_n4096_k4096 / n6144 | 0.1493 / 0.2145 | 0.1417 / 0.2023 | 0.1406 / 0.2048 | 0.1352 / 0.1802 (cublas-fresh) | 1.05 / 1.12 | the stream-K / atomic zero |
| pj_m128_n6144_k4096_bf16 | 0.0960 | 0.0895 | 0.0872 | 0.0881 (torch-bf16) | 1.02 | same |
| mlp_512 / indep_512 | 0.0392 / 0.0392 | 0.0371 / 0.0380 | 0.0372 / 0.0381 | 0.0328 (torch-compile) | 1.13 / 1.16 | their S = 2 atomic sites' zero |
| attn_1024_64 / 2048_64 / 2048_128, rowsum_2048 | | | | | null | the zero is not the first fill's / a long program |

Evidence (`evidence.py` over both arms): 8785 checks, every mismatch class the same on both arms (the device IR is
equal). No hardware property enters the rule (it reads the module's own zeros); the driver behaviour is CUDA's, so
every part runs it — the rented rows move at the next round.

## 2. `streamk-price` — an under-filled site's stream-K priced by the part's rates (closed generic where the rates exist)

`rules.rs` shipped `streamk` in the filled class only: "under-filled it lost on the 5080's skinny projections and
the 5090's 16 × 2048 × 7168 — site-level wave quantization is the open fact (both sides have measured exceptions)".
loopmm's 1024³ (128 tiles of `128 × 64` on 168 slots, `wps` 1) ran ordered: 44 SMs two tiles, 40 one.

**Forced (every class, `--gpu-on=streamk`), 13 changed programs, ABBA 8 reps — a mixed result:** loopmm 0.949, smlp
0.963 (wins); the 2048 × 2048 × 128 scores 1.41, the 512³ cells (indep / mlp_512) 1.24 / 1.15, rt_mlp2_1024 1.11,
attn_2048_64 1.06 (losses). Per kernel (CUPTI-free trace): loopmm's walk 0.076–0.082 against 0.085–0.090 ms; the
scores' 0.076 against 0.047. What separates them is the walk's fixed cost, which the panel-count rule (`iters + 1 <
split`: one panel of fixup) does not price: each piece adds `bm × bn` partials by `red`, and the output's zero pass
precedes the kernel — the scores' 8-panel tiles give 414 pieces × 64 KB = 26.5 MB of `red` plus a 16 MiB zero on a
41 µs ordered kernel; loopmm's 64-panel tiles 296 × 32 KB = 9.5 MB.

**The rule (`streamk_priced`, gpu.rs).** From the part's measured `DeviceCost` (FMA tile `flops_per_ns`,
`atomic_ops_per_ns`, `zero_ns`, the fill or L2 rate): one block's panel `t = 2·bm·bn·bk·wave / flops_per_ns`;
stream-K costs `iters·t + (ctas + tiles)·bm·bn / atomic_ops_per_ns + zero`; the S63 count costs `split·t` (+ its
`s·tiles` partials and the zero when it cuts). f32 only; `None` without the rates. Rung **`streamk-price`** (on):
an under-filled site (its class ships `streamk` off) streams where the price says so; a class that ships `streamk`
keeps its panel count (its replayed wins); an arm that turns `streamk` off (a group's order, a measurement) stays
off. OFF: `--gpu-off=streamk-price`. Replayed on the forced run's points with the rtx-5080's rates: loopmm 75.6 vs
87.5 µs (measured 77 / 85: streams), the scores 57.6 vs 41 (76 / 47: ordered), 512³ 11.1 vs 10.7 (ordered) — every
sign right.

**Changed programs** (125 matrix programs, `--off streamk-price`): 5 differ. ABBA 10 reps, value gates 0 failed,
null within 1 %:

| workload | OFF | ON | null | fastest competitor (leg) | ON / best (was) | explanation |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| loopmm_1024_8 | 0.6492 | **0.6121** | 0.6151 | 0.5878 (torch-compile-cg) | 1.04 (1.10) | 168 CTAs × 49 panels over the 128 tiles (was 1.52 tiles an SM): 512 FFMA / 32 REDG a piece, 125 registers, no spill (OFF 112) |
| rt_mlp2_1024 | 2.673 | **2.512** | 2.512 | 2.528 (torch-compile-cg) | 0.99 (1.01) | its plain 1024³ site the same walk |
| smlp_4096_256_1024 | 0.1650 | **0.1587** | 0.1582 | 0.1495 (torch-compile) | 1.06 (1.10) | the 4096 × 256 × 1024 site (128 tiles, 64 panels) |
| attn_2048_64 | 0.0859 | 0.0808 | 0.0814 | 0.0696 (torch-compile-ro) | — | a table candidate only; candidate 0 null (0.995): the tune's pick |
| attn_2048_128 | 0.1684 | 0.1200 | 0.1191 | 0.0983 (torch-sdpa-compile) | — | tune-pick variance (the OFF arm's text is integration 2's, which timed 0.1192): candidate 0 null (1.003) |

Candidate 0 (`rules`, no tune): loopmm 0.947, rt_mlp2 0.979, smlp 0.977 — all stable signs. Evidence over both arms:
4266 checks, the mismatch classes identical on the two arms; the new `streamk-price` intent (one wave, every unit of
the launched grid once, `red` combine, no spill) 10 / 10 ok; the `k-fill` intent now applies to its own walks only
(plan lines name the walk: `walk=class|fill|price`) — its 5 false mismatches on the priced walks are gone.

**Other parts.** The rule reads properties only. The rtx-5090's profile carries the rates: its loopmm / smlp /
rt_mlp2 take the walk (emitted, validated at the next round); the rtx-4090's prices refuse them; the rtx-3090, A100
and H100 profiles carry no `atomic_ops_per_ns` (`DeviceCost` never calibrated the `red` rate there), so their sites
keep the class's rule — their loopmm rows (1.20 / 1.37 / 1.26×) stay open until a rented round measures the rate
(plan-cost E1's calib method; `calib.cu` has no `red` mode yet — the next round's one addition).

**Tests.** `streamk_price_separates_the_measured_sites` (gpu.rs unit: loopmm streams, 1024 × 1024 × 256 and 512³ do
not, the OFF arm and a part without the rates keep the class's rule); large_gemm
**`streamk_price_on_generated_shapes`**: 96 generated GEMMs (half at drawn extents in the band where the rule acts,
768–1536 a side, 32–256 panels): **17 priced walks of 43 under-filled tile kernels**, each gone under the OFF arm,
none without the rates, the launched tile and grid kept, one wave, every unit once; values through the emulator on
10 generated programs at 256–512 a side with the `red` rate raised (`--cost=`): **3 programs, 9 emulated runs**
(free threads, serialized, reversed) equal the interpreter and the OFF arm. `large_gemm_rules_on_generated_shapes`
scoped (`k-fill`'s assertions to `k-fill`'s walks): 6 k-fill, 34 filled stream-K, 10 declined, 24 whole-file tiles.
`rung_rules_replay`: +2 lines (warm-zero A, streamk-price U on the 5080): 130 picked / 71 null / 10 mixed / 19 open.
`candidate_zero_is_the_pre_sprint_emission`: `c0-rules.tsv` regolded, 59 of 204 rows — warm-zero's init word on
every part whose program zeroes (host IR only: checked on skinny_16x16384x512 for the 3090 / 4090 / 5090, device IR
equal), the 5090's priced walks. Backend subset run (11 binaries, 214 tests, 0 failed): lib 123, large_gemm 3,
site_choice 9, splitk 5, device_geometry 11, golden_ll 36, caching_memory 11, gemv_stream 4, hardware_properties 2,
site_geometry 7, zero_fold 3 (one large_gemm run hit an LLVM crash in the emulator's link while /tmp was out of
quota; the rerun of the same programs passed). The full workspace suite is the integration's.

## 3. m = 1 f32 GEMV kernels (attempted, open)

After §1 the m = 1 f32 rows' gap is the kernel and the call's shape. Event-timed in the program's state (`kprobe.cu`:
Mapal's PTX, `b` rewritten by a fill kernel before each repetition, as the program's generator does):

| site | rules (S, blocks) | 2× S | ½ S | cuBLASLt-fresh (matrix) |
| --- | --- | --- | --- | ---: |
| 1 × 4096 × 4096 (64 MB) | 32.8 µs (32, 128) | **24.6** (64, 256) | 28.7 (16 one-cell, 256) | 22.5 |
| 1 × 6144 × 4096 (96 MB) | **73.7** (16, 96) | 81.9 (32, 192) | 79.9 (8 one-cell) | 51.2 |
| 1 × 2048 × 7168 (56 MB) | **16.4** (56, 112) | 18.4 (112, 224) | 16.4 (28 one-cell) | 26.6 |
| 1 × 4096 × 14336, 28672 × 4096, 18432 × 7168 | 238 / 516 / 578 | 244 / 508 / 578 | — | DRAM-bound: null |

A mixed result: doubling the cut wins only where the operand just written is about the L2's size (64 MB on 64 MB) and
loses below and above it. The separating fact is the operand's L2 residence at its read after a producer that wrote
it (`residence_plan` prices a value's interval against capacity, not the producer's write-back state); not pinned on
three points, no rule ships. The rest of the m1 f32 gap is the call: a launch and a synchronize the event-timed
library leg does not pay (gemv-tma.md).

## 4. m128 TF32 projections (attempted, open)

`pj_m128_n4096_k4096_tf32` (1.66×): the shipped kernel is `mma` `mt 2 nt 2`, `bk` 32 × 2, 64 × 64 tiles — 128 blocks
on 168 slots (1.52 an SM). Its hot loop: 32 HMMA.1688.F32.TF32, 16 LDSM, 32 LDS, and **96 FSETP + 96 predicated
IADD** — `cvt.rna.tf32.f32` lowers on sm_120 to a finite check and a rounding add per element, three elements a
HMMA at this warp tile (a fragment pair per 2 × 2 mma). A TF32 `round-store` (operands rounded once by the producer,
stored f32 with zero low bits, read raw: the hardware truncation is then exact) removes them; it needs a TF32-exact
storage fact beside `precision_plan.store`'s bf16 (no format carries "f32, low 13 bits zero"). Not built this sweep.

## 5. Epilogue groups forfeit stream-K (attempted, open)

sweep-runtime §1's option (a): the tile's last CTA (a ticket) applies the group's chain to the whole sum. Built as
`split-kv`'s ticket (module-global counters, reset by the last arriver) it would race: kernel ids key the site's
structure (`site_key`), so two identical sites on two chain streams launch one kernel with one counter array at
once (e.g. two equal projections on independent paths). What closes it: per-call-site ticket state — the runtime
passing a counter block per call site (a launch on one stream never overlaps itself; PDL waits at entry), or counters
indexed by the launching stream slot — then the chain evaluated by the tile's last arriver over the output the
pieces accumulated into. The affine-chain variant (`fold·α + r`, α a power of two: indep / diamond / fanout) needs a
body fact (the member affine in the site value) that `linear_plan` does not carry. §1 lowers the 512 sites' zero
(one of `absorb_pays`' four misses).

## 6. The 2-byte panels' depth (not re-attempted)

sweep-gemm §2's candidate fact (a partial last wave at one block an SM × DRAM residence) has a counter-example at the
same shape: `mma-depth` lost on 128 × 28672 × 4096 TF32 (0.722 / 0.696, `rung_replay.tsv`) where it won bf16. The
element width separates them too; with two shapes nothing pins it.

## 7. Found for the runtime area: first-call copy costs

`cfirst.cu` (fresh process, pageable): the first 4-byte HtoD 11.6 µs (3.6 after one warm copy), the first 4-byte DtoH
18–21 µs (6.8–10 after). The GEMV rows' copies run after their clock; a program with a copy inside its clock pays
them once — the same init-thread warm as §1 would cover them (not done here: the runtime area's item).

## Rows moved (RTX 5080, ms, same session)

| workload | OFF | ON | fastest competitor (leg) | explanation |
| --- | ---: | ---: | --- | --- |
| pj_m1_n4096_k4096 | 0.0384 | 0.0311 | 0.0225 (lt-fresh) | warm-zero: the first fill's driver init out of the clock |
| pj_m1_n2048_k7168 | 0.0355 | 0.0275 | 0.0266 (cublas-fresh) | warm-zero |
| pj_m16_n2048_k7168 | 0.0465 | 0.0392 | 0.0492 (cublas-fresh) | warm-zero |
| mlp_512 | 0.0392 | 0.0371 | 0.0328 (torch-compile) | warm-zero (its S = 2 atomic sites) |
| loopmm_1024_8 | 0.6492 | 0.6121 | 0.5878 (torch-compile-cg) | streamk-price: 168 CTAs × 49 panels over 128 tiles |
| rt_mlp2_1024 | 2.673 | 2.512 | 2.528 (torch-compile-cg) | streamk-price |
| smlp_4096_256_1024 | 0.1650 | 0.1587 | 0.1495 (torch-compile) | streamk-price |

## For integration

README line (`local-rtx-5080/README.md`):

`| sweep-gemm-2 | [sweep-gemm-2.md](sweep-gemm-2.md) | warm-zero: the driver's lazy memset init (13-17 us a process's first >= 8 KiB fill) moved before the clock — one-call atomic-split programs 0.77-0.98x (pj_m1 GEMVs 0.77-0.81x); streamk-price: an under-filled site's stream-K priced by the part's flops / red / zero rates (site-level wave quantization) — loopmm 0.94x, rt_mlp2_1024 0.94x, smlp 0.96x; m1 GEMV L2-edge geometry, TF32 per-fragment cvt, epilogue tickets: attempted, open |`

CLAUDE.md facts table — extend the backend `Gpu` properties row's rule list:

`| backend: Gpu properties (…), wave_full, tile_capacity, rules::rungs, streamk_priced | … ; an under-filled site's stream-K walk priced by the part's DeviceCost rates (FMA tile, red, zero pass) against the S63 count (sweep-gemm-2) |`

`open-problems.md` rows: m1 f32 GEMVs → closed in part (warm-zero), the L2-edge geometry open (§3); loopmm → closed on
parts with the rates (streamk-price), open on 3090 / A100 / H100 until their `red` rate is measured; m128 TF32 →
attempted, open (§4); epilogue groups → attempted, open (§5: per-call-site ticket state); first-call copies → new,
runtime (§7). `open-rules.md`'s replay count: 130 picked / 71 null / 10 mixed / 19 open.

Tailored-fix registry: none (no tailored fix in this sweep).
